import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix
import numpy as np
import pandas as pd
import re
import tldextract
from collections import Counter
from typing import Dict, Tuple, List, Optional
import logging
import psutil
import os
import gc
import glob
import multiprocessing as mp

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Number of parallel worker processes for feature extraction.
# Leave 1-2 cores free for the main process + OS.
N_WORKERS = max(1, mp.cpu_count() - 2)

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(PROJECT_DIR, "content.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def log_memory_usage(stage=""):
    process = psutil.Process(os.getpid())
    mem_gb = process.memory_info().rss / 1024 / 1024 / 1024
    logger.info(f"[Memory] {stage}: {mem_gb:.2f} GB")

# ─────────────────────────────────────────────────────────────────────────────
# DOM Tree Builder
# ─────────────────────────────────────────────────────────────────────────────
from dom_tree_builder import DOMTreeBuilder, DOMTreeComparator

torch.manual_seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# MODULE-LEVEL worker functions
# (Must be at module level — not inside a class — so multiprocessing can
#  pickle them on Windows, which uses 'spawn' instead of 'fork'.)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_html_features_single(html: str) -> Dict:
    """Parse one HTML string and return feature dict. Runs in worker process."""
    try:
        builder = DOMTreeBuilder()
        builder.build_from_html_string(html)
        return builder.extract_all_html_features()
    except Exception:
        return {
            'num_tags': 0, 'num_forms': 0, 'num_inputs': 0, 'num_links': 0,
            'num_scripts': 0, 'num_iframes': 0, 'num_images': 0, 'num_divs': 0,
            'num_password_fields': 0, 'num_external_links': 0, 'external_link_ratio': 0,
            'suspicious_keyword_count': 0, 'has_meta_description': 0, 'has_meta_keywords': 0,
            'title_length': 0, 'num_hidden_elements': 0,
            'max_dom_depth': 0, 'avg_dom_depth': 0, 'dom_balance': 0,
            'leaf_node_ratio': 0, 'avg_children_per_node': 0, 'dom_complexity': 0
        }


def _extract_url_features_single(url: str) -> Dict:
    """Extract URL features for one URL. Runs in worker process."""
    features = {}
    url_lower = url.lower()
    features['url_length'] = len(url)
    features['num_dots'] = url.count('.')
    features['num_hyphens'] = url.count('-')
    features['num_underscores'] = url.count('_')
    features['num_slashes'] = url.count('/')
    features['num_question_marks'] = url.count('?')
    features['num_equals'] = url.count('=')
    features['num_ampersands'] = url.count('&')
    features['num_at_symbols'] = url.count('@')
    features['num_digits'] = sum(c.isdigit() for c in url)
    features['has_ip_address'] = 1 if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', url) else 0
    features['has_https'] = 1 if url.startswith('https') else 0
    try:
        extracted = tldextract.extract(url)
        domain = extracted.domain
        subdomain = extracted.subdomain
        tld = extracted.suffix
        features['domain_length'] = len(domain)
        features['subdomain_length'] = len(subdomain)
        features['tld_length'] = len(tld)
        features['num_subdomains'] = subdomain.count('.') + 1 if subdomain else 0
        suspicious_tlds = ['tk', 'ml', 'ga', 'cf', 'gq', 'xyz', 'top', 'click']
        features['is_suspicious_tld'] = 1 if tld in suspicious_tlds else 0
        if domain:
            char_counts = Counter(domain)
            entropy = 0.0
            for count in char_counts.values():
                prob = count / len(domain)
                entropy -= prob * np.log2(prob)
            features['domain_entropy'] = entropy
        else:
            features['domain_entropy'] = 0
    except Exception:
        for key in ['domain_length', 'subdomain_length', 'tld_length', 'num_subdomains',
                    'is_suspicious_tld', 'domain_entropy']:
            features[key] = 0
    features['has_suspicious_keywords'] = 1 if any(
        kw in url_lower for kw in ['login', 'verify', 'account', 'secure', 'update']
    ) else 0
    return features


def _worker_extract_batch(args):
    """
    Worker entry point called by Pool.map().
    args = (urls_sublist, htmls_sublist)
    Returns (html_feats_array, url_feats_array)
    """
    urls, htmls = args
    html_feats = pd.DataFrame([_extract_html_features_single(h) for h in htmls]).fillna(0).values.astype(np.float32)
    url_feats  = pd.DataFrame([_extract_url_features_single(u)  for u in urls ]).fillna(0).values.astype(np.float32)
    return html_feats, url_feats


# ─────────────────────────────────────────────────────────────────────────────
# Feature Extractors (kept for single-sample inference)
# ─────────────────────────────────────────────────────────────────────────────
class HTMLFeatureExtractor:
    def __init__(self, use_dom_tree: bool = True):
        self.use_dom_tree = use_dom_tree
        self.dom_builder = DOMTreeBuilder() if use_dom_tree else None

    def extract(self, html: str) -> Dict:
        if self.use_dom_tree:
            self.dom_builder.build_from_html_string(html)
            return self.dom_builder.extract_all_html_features()
        return _extract_html_features_single.__wrapped__(html) if hasattr(_extract_html_features_single, '__wrapped__') else {k: 0 for k in [
            'num_tags','num_forms','num_inputs','num_links','num_scripts','num_iframes',
            'num_images','num_divs','num_password_fields','num_external_links',
            'external_link_ratio','suspicious_keyword_count','has_meta_description',
            'has_meta_keywords','title_length','num_hidden_elements',
            'max_dom_depth','avg_dom_depth','dom_balance','leaf_node_ratio',
            'avg_children_per_node','dom_complexity'
        ]}


class URLFeatureExtractor:
    def extract(self, url: str) -> Dict:
        return _extract_url_features_single(url)


# ─────────────────────────────────────────────────────────────────────────────
# Neural Network
# ─────────────────────────────────────────────────────────────────────────────
class PhishingNet(nn.Module):
    def __init__(self, html_dim, url_dim, tfidf_dim, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.html_net = nn.Sequential(
            nn.Linear(html_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64),       nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(dropout)
        )
        self.url_net = nn.Sequential(
            nn.Linear(url_dim, 128),  nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64),       nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(dropout)
        )
        self.tfidf_net = nn.Sequential(
            nn.Linear(tfidf_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128),       nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout)
        )
        fusion_dim = 64 + 64 + 128
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 128),        nn.BatchNorm1d(128),        nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 2)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, html_feat, url_feat, tfidf_feat):
        return self.fusion(torch.cat([
            self.html_net(html_feat),
            self.url_net(url_feat),
            self.tfidf_net(tfidf_feat)
        ], dim=1))


# ─────────────────────────────────────────────────────────────────────────────
# Main Classifier
# ─────────────────────────────────────────────────────────────────────────────
class PhishingClassifier:
    def __init__(self, use_dom_tree: bool = True, brand_database_path: Optional[str] = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.html_extractor = HTMLFeatureExtractor(use_dom_tree=use_dom_tree)
        self.url_extractor  = URLFeatureExtractor()
        self.url_vectorizer  = HashingVectorizer(n_features=500,  analyzer='char', ngram_range=(2, 4), alternate_sign=False)
        self.html_vectorizer = HashingVectorizer(n_features=1000, alternate_sign=False)
        self.html_scaler = StandardScaler()
        self.url_scaler  = StandardScaler()
        self.model = None
        self.use_dom_tree = use_dom_tree
        self.dom_comparator = DOMTreeComparator() if use_dom_tree else None
        self.brand_database = self._load_brand_database(brand_database_path) if brand_database_path else {}

        # Compute feature dims from a sample
        sample_html = "<html><body><div>test</div></body></html>"
        sample_url  = "https://example.com/test"
        self.html_dim  = len(self.html_extractor.extract(sample_html))
        self.url_dim   = len(self.url_extractor.extract(sample_url))
        self.tfidf_dim = 1500  # 500 url-char-ngrams + 1000 html-word-hashes

        print("\n" + "-"*50)
        print(f"PyTorch {torch.__version__} | Device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}  CUDA: {torch.version.cuda}")
        print(f"CPU cores available: {mp.cpu_count()}  →  Using {N_WORKERS} worker processes")
        print(f"\nFeature dims:  HTML={self.html_dim}  URL={self.url_dim}  TF-IDF={self.tfidf_dim}")
        print(f"DOM Tree: {'ENABLED' if use_dom_tree else 'DISABLED'}")
        print("-"*50 + "\n")

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _load_brand_database(self, path: str) -> Dict:
        try:
            import json
            with open(path, 'r') as f:
                return json.load(f)
        except Exception:
            return {}

    def extract_batch(self, urls: List[str], htmls: List[str]):
        """
        Parallel feature extraction using a multiprocessing Pool.
        Splits the batch into N_WORKERS sub-batches, runs them in parallel,
        then concatenates results.
        """
        n = len(urls)
        if n == 0:
            return np.empty((0, self.html_dim), dtype=np.float32), \
                   np.empty((0, self.url_dim),  dtype=np.float32), \
                   np.empty((0, self.tfidf_dim),dtype=np.float32)

        # Split into sub-batches for each worker
        chunk_size = max(1, (n + N_WORKERS - 1) // N_WORKERS)
        chunks = [
            (urls[i:i+chunk_size], htmls[i:i+chunk_size])
            for i in range(0, n, chunk_size)
        ]

        # Parallel DOM parsing + URL feature extraction
        if N_WORKERS > 1 and n > 50:
            with mp.Pool(processes=N_WORKERS) as pool:
                results = pool.map(_worker_extract_batch, chunks)
        else:
            # Single-process fallback for small batches or 1-core machines
            results = [_worker_extract_batch(c) for c in chunks]

        html_feats = np.vstack([r[0] for r in results]).astype(np.float32)
        url_feats  = np.vstack([r[1] for r in results]).astype(np.float32)

        # TF-IDF hashing (sklearn is already fast, no need to parallelize)
        url_hashed  = self.url_vectorizer.transform(urls).toarray().astype(np.float32)
        html_hashed = self.html_vectorizer.transform(htmls).toarray().astype(np.float32)
        tfidf_feats = np.hstack([url_hashed, html_hashed])

        return html_feats, url_feats, tfidf_feats

    def compare_with_brands(self, html: str, threshold: float = 0.7) -> Dict:
        if not self.use_dom_tree or not self.brand_database:
            return {'brand_match': None, 'similarity': 0.0}
        try:
            builder = DOMTreeBuilder()
            builder.build_from_html_string(html)
            input_tree = [
                {'id': n.id, 'tag': n.tag, 'parent_id': n.parent_id, 'child_ids': n.child_ids,
                 'child_count': n.child_count, 'attributes': n.attributes, 'depth': n.depth}
                for n in builder.dom_array
            ]
            best = {'brand': None, 'similarity': 0.0}
            for brand, brand_tree in self.brand_database.items():
                m = self.dom_comparator.compare_structure(input_tree, brand_tree)
                sim_nodes = self.dom_comparator.find_similar_nodes(input_tree, brand_tree, threshold)
                size_sim  = 1 - abs(m['tree1_nodes'] - m['tree2_nodes']) / max(m['tree1_nodes'], m['tree2_nodes'], 1)
                depth_sim = 1 - abs(m['max_depth_tree1'] - m['max_depth_tree2']) / max(m['max_depth_tree1'], m['max_depth_tree2'], 1)
                node_sim  = len(sim_nodes) / max(len(input_tree), len(brand_tree), 1)
                combined  = size_sim * 0.3 + depth_sim * 0.2 + node_sim * 0.5
                if combined > best['similarity']:
                    best = {'brand': brand, 'similarity': combined}
            return {
                'brand_match': best['brand'] if best['similarity'] >= threshold else None,
                'similarity':  best['similarity']
            }
        except Exception:
            return {'brand_match': None, 'similarity': 0.0}

    # ── Checkpointing ──────────────────────────────────────────────────────────

    def _checkpoint_path(self, split_name: str, count: int) -> str:
        return os.path.join(PROJECT_DIR, f"checkpoint_{split_name}_{count}.npz")

    def save_checkpoint(self, split_name: str, count: int,
                        pending_html: list, pending_url: list,
                        pending_tfidf: list, pending_labels: list):
        filename = self._checkpoint_path(split_name, count)
        try:
            np.savez_compressed(
                filename,
                html   = np.vstack(pending_html),
                url    = np.vstack(pending_url),
                tfidf  = np.vstack(pending_tfidf),
                labels = np.concatenate(pending_labels)
            )
            n = sum(len(l) for l in pending_labels)
            logger.info(f"✅ Checkpoint saved: {os.path.basename(filename)}  ({n:,} new samples)")
        except Exception as e:
            logger.error(f"❌ Checkpoint save failed ({filename}): {e}")

    def load_checkpoints(self, split_name: str):
        """
        Scan for existing checkpoint files and return total samples already saved.
        Data is NOT loaded into RAM here — it stays on disk until final stacking.
        """
        pattern = os.path.join(PROJECT_DIR, f"checkpoint_{split_name}_*.npz")
        files = sorted(
            glob.glob(pattern),
            key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0])
        )
        if not files:
            return 0, []

        total = 0
        for f in files:
            try:
                d = np.load(f)
                n = len(d['labels'])
                total += n
                logger.info(f"  Found: {os.path.basename(f)}  ({n:,} samples)")
            except Exception as e:
                logger.error(f"  ❌ Corrupt checkpoint {f}: {e}")

        logger.info(f"  → Resuming from {total:,} already-processed samples.")
        return total, files

    def _stack_all_checkpoints(self, checkpoint_files: list, split_name: str) -> Optional[Dict]:
        """Load checkpoint files one-by-one from disk and stack. RAM-efficient."""
        if not checkpoint_files:
            logger.warning(f"No checkpoints for '{split_name}'!")
            return None

        logger.info(f"Stacking {len(checkpoint_files)} checkpoint(s) from disk...")
        log_memory_usage("Before Stack")

        html_parts, url_parts, tfidf_parts, label_parts = [], [], [], []
        for f in checkpoint_files:
            try:
                d = np.load(f)
                html_parts.append(d['html'])
                url_parts.append(d['url'])
                tfidf_parts.append(d['tfidf'])
                label_parts.append(d['labels'])
            except Exception as e:
                logger.error(f"  ❌ Failed to load {f}: {e}")

        if not html_parts:
            return None

        result = {
            'html':  np.vstack(html_parts),
            'url':   np.vstack(url_parts),
            'tfidf': np.vstack(tfidf_parts),
            'y':     np.concatenate(label_parts)
        }
        logger.info(f"Stacked shapes → HTML:{result['html'].shape}  URL:{result['url'].shape}  TFIDF:{result['tfidf'].shape}")
        log_memory_usage("After Stack")
        return result

    # ── Processing ─────────────────────────────────────────────────────────────

    def process_split(self, dataset, split_name: str, max_samples=None) -> Optional[Dict]:
        logger.info(f"Processing split: '{split_name}'")
        logger.info(f"Parallel workers: {N_WORKERS}  (CPU cores: {mp.cpu_count()})")
        logger.info("(Note: '206 Partial Content' = normal streaming byte-range requests)")
        log_memory_usage(f"Start {split_name}")

        # 1. Find existing checkpoints
        processed_count, checkpoint_files = self.load_checkpoints(split_name)

        # 2. Fast-forward past already-processed samples
        iterator = iter(dataset)
        if processed_count > 0:
            logger.info(f"Fast-forwarding {processed_count:,} samples (streaming — may take a few minutes)...")
            skipped = 0
            try:
                for _ in range(processed_count):
                    next(iterator)
                    skipped += 1
                logger.info(f"Fast-forward complete ({skipped:,} samples skipped).")
            except StopIteration:
                logger.warning("Dataset ended during fast-forward — all data already checkpointed.")
                return self._stack_all_checkpoints(checkpoint_files, split_name)
            except Exception as e:
                logger.error(f"Fast-forward error: {e}")

        # 3. Stream and process
        batch_html, batch_url, batch_labels = [], [], []
        pending_html, pending_url, pending_tfidf, pending_labels = [], [], [], []

        current_session_count = 0
        batch_size            = 1000
        checkpoint_interval   = 50_000

        try:
            for sample in iterator:
                total_processed = processed_count + current_session_count

                if max_samples is not None and total_processed >= max_samples:
                    logger.info(f"Reached max_samples ({max_samples:,}). Stopping.")
                    break

                batch_html.append(sample.get('html', '') or '')
                batch_url.append(sample.get('url', '')  or '')

                try:
                    raw = sample.get('label', 0)
                    val = 1 if (raw.lower() in ['phish', 'phishing', '1'] if isinstance(raw, str) else int(raw) == 1) else 0
                    batch_labels.append(val)
                except Exception:
                    batch_labels.append(0)

                current_session_count += 1

                # ── Process mini-batch ──────────────────────────────────────
                if len(batch_html) >= batch_size:
                    try:
                        h, u, t = self.extract_batch(batch_url, batch_html)
                        pending_html.append(h)
                        pending_url.append(u)
                        pending_tfidf.append(t)
                        pending_labels.append(np.array(batch_labels, dtype=np.int8))
                    except Exception as e:
                        logger.error(f"❌ Batch error at sample {total_processed}: {e}. Skipping.")
                    finally:
                        batch_html, batch_url, batch_labels = [], [], []

                    # ── Checkpoint every 50k new samples ───────────────────
                    if current_session_count % checkpoint_interval == 0 and pending_html:
                        ckpt_total = processed_count + current_session_count
                        logger.info(f"Checkpointing at {ckpt_total:,} total samples...")
                        self.save_checkpoint(split_name, ckpt_total,
                                             pending_html, pending_url,
                                             pending_tfidf, pending_labels)
                        checkpoint_files.append(self._checkpoint_path(split_name, ckpt_total))
                        pending_html, pending_url, pending_tfidf, pending_labels = [], [], [], []
                        gc.collect()

                    # ── Progress log every 5k samples ──────────────────────
                    if current_session_count % 5000 == 0:
                        logger.info(f"Processed {processed_count + current_session_count:,} samples for '{split_name}'...")
                        log_memory_usage(f"Progress {processed_count + current_session_count:,}")

        except KeyboardInterrupt:
            logger.warning("🛑 Interrupted by user!")
        except Exception as e:
            logger.critical(f"🔥 Stream error: {e}", exc_info=True)

        # 4. Flush remaining mini-batch
        if batch_html:
            try:
                h, u, t = self.extract_batch(batch_url, batch_html)
                pending_html.append(h)
                pending_url.append(u)
                pending_tfidf.append(t)
                pending_labels.append(np.array(batch_labels, dtype=np.int8))
            except Exception as e:
                logger.error(f"Final mini-batch error: {e}")

        # 5. Save remaining pending data
        if pending_html:
            final_count = processed_count + current_session_count
            logger.info(f"Saving final partial checkpoint at {final_count:,} samples...")
            self.save_checkpoint(split_name, final_count,
                                 pending_html, pending_url,
                                 pending_tfidf, pending_labels)
            checkpoint_files.append(self._checkpoint_path(split_name, final_count))

        logger.info(f"✅ '{split_name}' done. Total: {processed_count + current_session_count:,} samples.")

        # 6. Stack everything from disk
        return self._stack_all_checkpoints(checkpoint_files, split_name)

    # ── Data Preparation ───────────────────────────────────────────────────────

    def prepare_data(self, max_samples=None) -> Dict:
        logger.info("Preparing data...")
        hf_token = "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"

        try:
            logger.info("Loading 'train' split stream...")
            train_stream = load_dataset('phreshphish/phreshphish', split='train', streaming=True, token=hf_token)
            logger.info("Loading 'test' split stream...")
            test_stream  = load_dataset('phreshphish/phreshphish', split='test',  streaming=True, token=hf_token)
        except Exception as e:
            logger.error(f"Dataset load error: {e}")
            raise

        # Train
        logger.info("─" * 60)
        logger.info("--- Starting Train Split Processing ---")
        logger.info("─" * 60)
        train_data = self.process_split(train_stream, "train", max_samples)
        if train_data is None:
            raise RuntimeError("Train processing returned None.")

        logger.info("Fitting scalers on training data...")
        self.html_scaler.fit(train_data['html'])
        self.url_scaler.fit(train_data['url'])
        train_data['html'] = self.html_scaler.transform(train_data['html'])
        train_data['url']  = self.url_scaler.transform(train_data['url'])

        # Test
        logger.info("─" * 60)
        logger.info("--- Starting Test Split Processing ---")
        logger.info("─" * 60)
        test_data = self.process_split(test_stream, "test", max_samples)
        if test_data is None:
            raise RuntimeError("Test processing returned None.")

        test_data['html'] = self.html_scaler.transform(test_data['html'])
        test_data['url']  = self.url_scaler.transform(test_data['url'])

        logger.info("Data Preparation Complete.")
        logger.info(f"  Train: {len(train_data['y']):,} samples")
        logger.info(f"  Test:  {len(test_data['y']):,} samples")
        return {'train': train_data, 'test': test_data}

    # ── Training ───────────────────────────────────────────────────────────────

    def train(self, data: Dict, epochs: int = 20, batch_size: int = 128, lr: float = 0.001):
        print("\n" + "="*80)
        print("TRAINING")
        print("="*80)

        class DS(Dataset):
            def __init__(self, html, url, tfidf, y):
                self.html  = torch.FloatTensor(html)
                self.url   = torch.FloatTensor(url)
                self.tfidf = torch.FloatTensor(tfidf)
                self.y     = torch.LongTensor(y)
            def __len__(self): return len(self.y)
            def __getitem__(self, i): return self.html[i], self.url[i], self.tfidf[i], self.y[i]

        train_loader = DataLoader(
            DS(data['train']['html'], data['train']['url'], data['train']['tfidf'], data['train']['y']),
            batch_size=batch_size, shuffle=True, num_workers=0
        )
        test_loader = DataLoader(
            DS(data['test']['html'], data['test']['url'], data['test']['tfidf'], data['test']['y']),
            batch_size=batch_size, num_workers=0
        )

        self.model = PhishingNet(self.html_dim, self.url_dim, self.tfidf_dim).to(self.device)
        criterion  = nn.CrossEntropyLoss()
        optimizer  = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler  = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3)
        best_acc   = 0.0

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            for h, u, t, y in train_loader:
                h, u, t, y = h.to(self.device), u.to(self.device), t.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(h, u, t), y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            self.model.eval()
            preds, true_labels, probs = [], [], []
            with torch.no_grad():
                for h, u, t, y in test_loader:
                    h, u, t = h.to(self.device), u.to(self.device), t.to(self.device)
                    out  = self.model(h, u, t)
                    prob = F.softmax(out, dim=1)
                    _, pred = torch.max(out, 1)
                    preds.extend(pred.cpu().numpy())
                    true_labels.extend(y.numpy())
                    probs.extend(prob[:, 1].cpu().numpy())

            acc = accuracy_score(true_labels, preds)
            auc = roc_auc_score(true_labels, probs)
            scheduler.step(acc)

            if acc > best_acc:
                best_acc = acc
                torch.save({
                    'model':    self.model.state_dict(),
                    'html_scaler': self.html_scaler,
                    'url_scaler':  self.url_scaler,
                    'html_dim':  self.html_dim,
                    'url_dim':   self.url_dim,
                    'tfidf_dim': self.tfidf_dim,
                    'acc': acc
                }, os.path.join(PROJECT_DIR, 'phishing_model.pth'))

            print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {train_loss/len(train_loader):.4f} | Acc: {acc:.4f} | AUC: {auc:.4f}")

        logger.info(f"Best Accuracy: {best_acc:.4f}")
        print("\n" + "="*80)
        print("FINAL TEST RESULTS")
        print("="*80)
        report = classification_report(true_labels, preds, target_names=['Benign', 'Phishing'])
        print(report)
        logger.info("Classification Report:\n" + report)
        logger.info(f"Confusion Matrix:\n{confusion_matrix(true_labels, preds)}")

        csv_path = os.path.join(PROJECT_DIR, 'test_results.csv')
        pd.DataFrame({
            'true_label':      ['Benign' if l == 0 else 'Phishing' for l in true_labels],
            'predicted_label': ['Benign' if p == 0 else 'Phishing' for p in preds],
            'confidence_score': probs,
            'correct': [1 if true_labels[i] == preds[i] else 0 for i in range(len(true_labels))]
        }).to_csv(csv_path, index=False)
        print(f"\n✅ Test results → {csv_path}")

    # ── Inference ──────────────────────────────────────────────────────────────

    def predict(self, url: str, html: str) -> Tuple[str, float]:
        if self.model is None:
            raise ValueError("Model not trained yet.")
        h, u, t = self.extract_batch([url], [html])
        h = self.html_scaler.transform(h)
        u = self.url_scaler.transform(u)
        self.model.eval()
        with torch.no_grad():
            out  = self.model(torch.FloatTensor(h).to(self.device),
                              torch.FloatTensor(u).to(self.device),
                              torch.FloatTensor(t).to(self.device))
            conf = F.softmax(out, dim=1)[0, 1].item()
        return ('Phishing' if conf > 0.5 else 'Benign'), conf

    def predict_with_similarity(self, url: str, html: str) -> Dict:
        label, conf = self.predict(url, html)
        brand_info  = self.compare_with_brands(html)
        if brand_info['brand_match'] and label == 'Phishing':
            return {'label': 'Phishing', 'confidence': min(conf + 0.1, 1.0),
                    'brand_match': brand_info['brand_match'],
                    'similarity':  brand_info['similarity'],
                    'warning': f"⚠️ Impersonating {brand_info['brand_match']}"}
        return {'label': label, 'confidence': conf,
                'brand_match': brand_info['brand_match'],
                'similarity':  brand_info['similarity'], 'warning': None}


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# IMPORTANT: The `if __name__ == '__main__':` guard is REQUIRED on Windows.
# Without it, each spawned worker process would re-run the top-level code,
# creating an infinite fork-bomb of processes.
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # Required for Windows multiprocessing (spawn method)
    mp.freeze_support()

    clf = PhishingClassifier(use_dom_tree=True)
    logger.info("Starting Full Pipeline Execution")
    logger.info(f"Worker processes: {N_WORKERS}")

    try:
        data = clf.prepare_data(max_samples=None)
        clf.train(data, epochs=20, batch_size=128, lr=0.001)
    except Exception as e:
        logger.critical(f"Pipeline crashed: {e}", exc_info=True)
        raise

    # Quick inference examples
    print("\n" + "="*80)
    print("EXAMPLES")
    print("="*80)

    url1   = "https://secure-verify-account.tk/login"
    html1  = "<html><body><form><input type='password'></form><h1>Verify Your Account Now</h1></body></html>"
    l1, c1 = clf.predict(url1, html1)
    print(f"\n[Basic]    {url1}\n           → {l1} ({c1:.2%})")
    r1 = clf.predict_with_similarity(url1, html1)
    print(f"[Enhanced] → {r1['label']} ({r1['confidence']:.2%})")
    if r1['warning']:
        print(f"           {r1['warning']}")

    url2   = "https://www.google.com/search"
    html2  = "<html><head><title>Google</title></head><body><div>Results</div></body></html>"
    l2, c2 = clf.predict(url2, html2)
    print(f"\n[Basic]    {url2}\n           → {l2} ({c2:.2%})")
    r2 = clf.predict_with_similarity(url2, html2)
    print(f"[Enhanced] → {r2['label']} ({r2['confidence']:.2%})")
    if r2['brand_match']:
        print(f"           Matches: {r2['brand_match']} (sim: {r2['similarity']:.2%})")

    print(f"\n✅ Model saved:        phishing_model.pth")
    print(f"✅ Test results saved: test_results.csv")
    print(f"✅ Features: HTML({clf.html_dim}) + URL({clf.url_dim}) + TF-IDF({clf.tfidf_dim})")
    print(f"✅ Workers used: {N_WORKERS}")
