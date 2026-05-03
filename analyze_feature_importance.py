#!/usr/bin/env python3
"""
analyze_feature_importance.py
==============================
Experiment 14 — Feature Importance Analysis across CharCNN + FeatureMLP streams.

Answers: which features (structured URL features) and which character-level signals
contribute most to phishing detection in the BinaryURLPhishNet dual-stream model?

Three complementary analyses:
  A. FeatureMLP permutation importance (67 features)
       Permute each feature on test set → measure accuracy / F1 drop.
       Features that cause the largest drop are most important.

  B. Gradient × Input saliency for FeatureMLP features (67 features)
       Backprop through the model with respect to the input feature vector.
       Attribution = mean |gradient × input| over test samples.
       Ranks features by their average signed contribution to phishing prediction.

  C. CharCNN analysis
       C1. Position importance: mean |gradient| at each of 256 character positions
           (averaged over phishing-classified URLs from test set).
           Shows which URL positions carry the most discriminative signal.
       C2. Character importance: mean |embedding weight| per character index (1-97).
           Shows which printable characters have highest learned representation magnitude.
       C3. Filter pattern extraction: most-activated character sequences per CNN filter bank
           (sample up to 1000 phishing URLs, extract top k-gram patterns per kernel size 3,5,7).

Loads:
  binary_phishing/models/model_best.pth    (BinaryURLPhishNet best checkpoint)
  binary_phishing/models/scaler.pkl        (fitted StandardScaler)
  phishphresh/multiclass_brand/data/       (phishphresh test chunks)

Output:
  feature_importance/logs/permutation_importance.csv
  feature_importance/logs/gradient_saliency.csv
  feature_importance/logs/char_position_importance.csv
  feature_importance/logs/char_vocab_importance.csv
  feature_importance/logs/top_char_ngrams.txt
  feature_importance/logs/feature_importance_report.txt
  feature_importance/logs/run.log
  feature_importance/logs/stdout.log
"""

from __future__ import annotations
import gc, glob, logging, math, os, re, sys, urllib.parse
from collections import Counter
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(PROJECT_DIR, "phishphresh", "multiclass_brand", "data")
MODEL_PATH  = os.path.join(PROJECT_DIR, "binary_phishing", "models", "model_best.pth")
SCALER_PATH = os.path.join(PROJECT_DIR, "binary_phishing", "models", "scaler.pkl")

OUT_DIR  = os.path.join(PROJECT_DIR, "feature_importance")
OUT_LOGS = os.path.join(OUT_DIR, "logs")
os.makedirs(OUT_LOGS, exist_ok=True)

STDOUT_LOG      = os.path.join(OUT_LOGS, "stdout.log")
PERM_CSV        = os.path.join(OUT_LOGS, "permutation_importance.csv")
GRAD_CSV        = os.path.join(OUT_LOGS, "gradient_saliency.csv")
CHAR_POS_CSV    = os.path.join(OUT_LOGS, "char_position_importance.csv")
CHAR_VOC_CSV    = os.path.join(OUT_LOGS, "char_vocab_importance.csv")
NGRAM_TXT       = os.path.join(OUT_LOGS, "top_char_ngrams.txt")
REPORT_TXT      = os.path.join(OUT_LOGS, "feature_importance_report.txt")

# ── Tee stdout → file ──────────────────────────────────────────────────────────
class _Tee:
    def __init__(self, *files): self.files = files
    def write(self, obj):
        for f in self.files: f.write(obj); f.flush()
    def flush(self):
        for f in self.files: f.flush()

_stdout_fh = open(STDOUT_LOG, "w", encoding="utf-8")
sys.stdout  = _Tee(sys.__stdout__, _stdout_fh)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(os.path.join(OUT_LOGS, "run.log"), encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ── Model constants (must match train_binary.py exactly) ──────────────────────
MAX_URL_LEN  = 256
VOCAB_SIZE   = 98
EMBED_DIM    = 64
CNN_CHANNELS = 128
FEAT_MLP_DIM = 256
FEAT_MLP_OUT = 128
DROPOUT      = 0.40
RANDOM_STATE = 42
TEST_SIZE    = 0.20
BATCH_SIZE   = 512
N_PERM_REPEATS = 5   # number of random shuffles per feature for permutation importance
GRAD_N_SAMPLES = 5000  # test samples to use for gradient saliency (speed)
CHAR_N_SAMPLES = 2000  # phishing samples for CNN filter analysis

# ── 67 feature names (order matches extract_features + _structural) ────────────
FEATURE_NAMES = [
    # Group 1: Basic (features 0-14)
    "domain_length", "dot_count", "dash_count", "underscore_count", "digit_count",
    "uppercase_count", "digit_ratio", "special_char_ratio", "domain_parts",
    "longest_part", "shortest_part", "avg_part_length", "vowel_count",
    "consonant_count", "has_consecutive_chars",
    # Group 2: Entropy (features 15-24)
    "char_entropy", "part_length_variance", "part_length_std", "bigram_entropy",
    "unique_char_ratio", "transition_entropy", "trigram_count", "unique_trigram_ratio",
    "vowel_consonant_ratio", "consonant_clusters",
    # Group 3: TLD (features 25-29)
    "tld_length", "is_common_tld", "is_country_tld", "is_suspicious_tld", "is_indian_tld",
    # Group 4: CSE Pattern (features 30-39)
    "legitimate_domain_similarity", "matches_legitimate_domain",
    "legitimate_keyword_count", "contains_legitimate_keyword",
    "brand_keyword_count", "has_brand_variation",
    "finance_keywords", "gov_keywords", "telecom_keywords", "tech_keywords",
    # Group 5: Lexical (features 40-50)
    "dictionary_word_count", "char_repetition_score", "has_year_pattern",
    "starts_with_number", "ends_with_number", "max_numeric_sequence",
    "has_subdomain", "subdomain_count", "suspicious_pattern_count",
    "has_homograph", "complexity_score",
    # Group 6: Structural (features 51-66)
    "url_length_normalized", "path_length_normalized", "query_length_normalized",
    "fragment_length_normalized", "path_slash_count", "query_param_count",
    "is_https", "has_nonstandard_port", "has_userinfo_in_netloc", "is_ip_address",
    "percent_encoded_ratio", "has_redirect_param", "has_double_slash_in_path",
    "shannon_entropy_url", "shannon_entropy_query", "shannon_entropy_path",
]

FEATURE_GROUPS = {
    "Basic (domain structure)":    list(range(0, 15)),
    "Entropy & complexity":        list(range(15, 25)),
    "TLD signals":                 list(range(25, 30)),
    "CSE pattern / brand":         list(range(30, 40)),
    "Lexical patterns":            list(range(40, 51)),
    "Structural / URL-level":      list(range(51, 67)),
}

assert len(FEATURE_NAMES) == 67, f"Expected 67 features, got {len(FEATURE_NAMES)}"


# ── Model architecture (identical to train_binary.py) ─────────────────────────
class _ResBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(dim, dim),
        )
    def forward(self, x): return x + self.block(x)


class FeatureMLP(nn.Module):
    def __init__(self, in_dim, proj=FEAT_MLP_DIM, out=FEAT_MLP_OUT, drop=DROPOUT):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, proj), nn.LayerNorm(proj), nn.GELU())
        self.res  = nn.Sequential(_ResBlock(proj, drop), _ResBlock(proj, drop))
        self.out  = nn.Sequential(nn.Linear(proj, out), nn.LayerNorm(out), nn.GELU(), nn.Dropout(drop))
        self.output_dim = out
    def forward(self, x): return self.out(self.res(self.proj(x)))


class CharCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb  = nn.Embedding(VOCAB_SIZE, EMBED_DIM, padding_idx=0)
        self.c3   = nn.Conv1d(EMBED_DIM, CNN_CHANNELS, 3, padding=1)
        self.c5   = nn.Conv1d(EMBED_DIM, CNN_CHANNELS, 5, padding=2)
        self.c7   = nn.Conv1d(EMBED_DIM, CNN_CHANNELS, 7, padding=3)
        self.bn3  = nn.BatchNorm1d(CNN_CHANNELS)
        self.bn5  = nn.BatchNorm1d(CNN_CHANNELS)
        self.bn7  = nn.BatchNorm1d(CNN_CHANNELS)
        fused     = 3 * CNN_CHANNELS
        self.deep = nn.Conv1d(fused, fused, 3, padding=1)
        self.bnd  = nn.BatchNorm1d(fused)
        self.drop = nn.Dropout(0.15)
        self.output_dim = 2 * fused
    def forward(self, x):
        e = self.emb(x).transpose(1, 2)
        c = torch.cat([F.relu(self.bn3(self.c3(e))),
                       F.relu(self.bn5(self.c5(e))),
                       F.relu(self.bn7(self.c7(e)))], dim=1)
        f = self.drop(F.relu(self.bnd(self.deep(c))))
        return torch.cat([F.adaptive_max_pool1d(f, 1).squeeze(-1),
                          F.adaptive_avg_pool1d(f, 1).squeeze(-1)], dim=1)


class BinaryURLPhishNet(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.feat_mlp = FeatureMLP(feat_dim)
        self.char_cnn = CharCNN()
        fusion = self.feat_mlp.output_dim + self.char_cnn.output_dim
        self.clf = nn.Sequential(
            nn.Linear(fusion, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(DROPOUT),
            nn.Linear(512, 256),   nn.LayerNorm(256), nn.GELU(), nn.Dropout(DROPOUT * 0.7),
            nn.Linear(256, 2),
        )
    def forward(self, feat, chars):
        return self.clf(torch.cat([self.feat_mlp(feat), self.char_cnn(chars)], dim=1))


# ── Data loading ──────────────────────────────────────────────────────────────
def load_test_data(data_dir, scaler, max_samples=None):
    """Load phishphresh chunks, apply same 80/20 split, return test split."""
    train_chunks = sorted(glob.glob(os.path.join(data_dir, "train", "chunk_*.npz")))
    test_chunks  = sorted(glob.glob(os.path.join(data_dir, "test",  "chunk_*.npz")))
    feat_parts, char_parts, tgt_parts = [], [], []
    for chunk in tqdm(train_chunks + test_chunks, desc="  Loading chunks"):
        d = np.load(chunk, allow_pickle=True)
        feat_parts.append(d["features"])
        char_parts.append(d["char_ids"])
        tgt_parts.extend(d["targets"].tolist())
    X_feat = np.vstack(feat_parts).astype(np.float32)
    X_char = np.vstack(char_parts).astype(np.int64)
    y_all  = np.array([0 if str(t or "").strip() == "" else 1 for t in tgt_parts],
                      dtype=np.int64)
    del feat_parts, char_parts; gc.collect()

    idx = np.arange(len(y_all))
    tr_idx, te_idx = train_test_split(idx, test_size=TEST_SIZE,
                                      random_state=RANDOM_STATE, stratify=y_all)
    Xf_te = scaler.transform(X_feat[te_idx]).astype(np.float32)
    Xc_te = X_char[te_idx]
    y_te  = y_all[te_idx]
    del X_feat, X_char; gc.collect()

    if max_samples and len(y_te) > max_samples:
        rng = np.random.default_rng(42)
        sel = rng.choice(len(y_te), max_samples, replace=False)
        return Xf_te[sel], Xc_te[sel], y_te[sel]
    return Xf_te, Xc_te, y_te


def run_inference_np(model, X_feat, X_char, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_feat), BATCH_SIZE):
            xf = torch.from_numpy(X_feat[i:i+BATCH_SIZE]).float().to(device)
            xc = torch.from_numpy(X_char[i:i+BATCH_SIZE].astype(np.int64)).to(device)
            preds.extend(model(xf, xc).argmax(1).cpu().numpy().tolist())
    return np.array(preds)


# ══════════════════════════════════════════════════════════════════════════════
# Analysis A: Permutation Feature Importance (FeatureMLP features)
# ══════════════════════════════════════════════════════════════════════════════

def permutation_importance(model, X_feat, X_char, y_true, device,
                            n_repeats=N_PERM_REPEATS):
    """
    For each of 67 features, shuffle its values N_PERM_REPEATS times,
    measure mean drop in F1-Macro. Higher drop = more important feature.
    """
    model.eval()
    baseline_preds = run_inference_np(model, X_feat, X_char, device)
    baseline_f1 = f1_score(y_true, baseline_preds, average="macro", zero_division=0)
    baseline_acc = accuracy_score(y_true, baseline_preds)
    log.info(f"  Baseline F1={baseline_f1:.4f}  Acc={baseline_acc*100:.2f}%")

    rng = np.random.default_rng(42)
    n_feat = X_feat.shape[1]
    results = []

    for fi in tqdm(range(n_feat), desc="  Permuting features"):
        f1_drops, acc_drops = [], []
        for _ in range(n_repeats):
            X_perm = X_feat.copy()
            X_perm[:, fi] = rng.permutation(X_perm[:, fi])
            preds = run_inference_np(model, X_perm, X_char, device)
            f1_drops.append(baseline_f1 - f1_score(y_true, preds, average="macro", zero_division=0))
            acc_drops.append(baseline_acc - accuracy_score(y_true, preds))
        results.append({
            "feature_idx": fi,
            "feature_name": FEATURE_NAMES[fi],
            "group": next(g for g, idxs in FEATURE_GROUPS.items() if fi in idxs),
            "f1_drop_mean": float(np.mean(f1_drops)),
            "f1_drop_std":  float(np.std(f1_drops)),
            "acc_drop_mean": float(np.mean(acc_drops)),
            "acc_drop_std":  float(np.std(acc_drops)),
        })
        del X_perm; gc.collect()

    return pd.DataFrame(results).sort_values("f1_drop_mean", ascending=False)


# ══════════════════════════════════════════════════════════════════════════════
# Analysis B: Gradient × Input Saliency (FeatureMLP features)
# ══════════════════════════════════════════════════════════════════════════════

def gradient_saliency(model, X_feat, X_char, y_true, device):
    """
    Compute mean |gradient × input| for each feature on phishing-prediction
    samples. Measures each feature's average signed contribution.
    """
    model.eval()
    feat_attr_pos = np.zeros(X_feat.shape[1])  # attribution for phishing class
    feat_attr_all = np.zeros(X_feat.shape[1])  # unsigned attribution
    n_samples = 0

    for i in tqdm(range(0, len(X_feat), BATCH_SIZE), desc="  Gradient saliency"):
        xf_np = X_feat[i:i+BATCH_SIZE]
        xc_np = X_char[i:i+BATCH_SIZE]

        xf = torch.from_numpy(xf_np).float().to(device).requires_grad_(True)
        xc = torch.from_numpy(xc_np.astype(np.int64)).to(device)

        logits = model(xf, xc)
        # Backprop w.r.t. phishing class (class 1) logit
        phish_logit = logits[:, 1].sum()
        phish_logit.backward()

        grad = xf.grad.detach().cpu().numpy()  # (batch, 67)
        feat_attr_pos += (grad * xf_np).sum(axis=0)   # gradient × input, signed
        feat_attr_all += np.abs(grad * xf_np).sum(axis=0)
        n_samples += len(xf_np)

        xf.grad = None
        del xf, xc, logits, grad; gc.collect()

    results = []
    for fi in range(X_feat.shape[1]):
        results.append({
            "feature_idx": fi,
            "feature_name": FEATURE_NAMES[fi],
            "group": next(g for g, idxs in FEATURE_GROUPS.items() if fi in idxs),
            "mean_grad_x_input": feat_attr_pos[fi] / n_samples,
            "mean_abs_grad_x_input": feat_attr_all[fi] / n_samples,
        })

    return pd.DataFrame(results).sort_values("mean_abs_grad_x_input", ascending=False)


# ══════════════════════════════════════════════════════════════════════════════
# Analysis C1: CharCNN position importance (gradient magnitude per position)
# ══════════════════════════════════════════════════════════════════════════════

def char_position_importance(model, X_feat, X_char, y_pred, device):
    """
    For samples predicted as phishing, compute mean gradient magnitude
    at each of 256 character positions (via embedding gradient).
    Shows which URL positions carry the most discriminative signal.
    """
    phish_mask = y_pred == 1
    Xf_ph = X_feat[phish_mask][:CHAR_N_SAMPLES]
    Xc_ph = X_char[phish_mask][:CHAR_N_SAMPLES]
    if len(Xf_ph) == 0:
        log.warning("  No phishing-predicted samples for position analysis")
        return pd.DataFrame()

    model.eval()
    pos_importance = np.zeros(MAX_URL_LEN)
    n_samples = 0

    for i in tqdm(range(0, len(Xf_ph), BATCH_SIZE), desc="  Char position grad"):
        xf = torch.from_numpy(Xf_ph[i:i+BATCH_SIZE]).float().to(device)
        xc = torch.from_numpy(Xc_ph[i:i+BATCH_SIZE].astype(np.int64)).to(device)

        # Get embedding, enable grad
        emb = model.char_cnn.emb(xc).float()  # (batch, 256, 64)
        emb.retain_grad()
        emb_t = emb.transpose(1, 2)  # (batch, 64, 256)

        # Forward through CharCNN manually with gradient tracking
        c3 = F.relu(model.char_cnn.bn3(model.char_cnn.c3(emb_t)))
        c5 = F.relu(model.char_cnn.bn5(model.char_cnn.c5(emb_t)))
        c7 = F.relu(model.char_cnn.bn7(model.char_cnn.c7(emb_t)))
        c  = torch.cat([c3, c5, c7], dim=1)
        f  = model.char_cnn.drop(F.relu(model.char_cnn.bnd(model.char_cnn.deep(c))))
        char_out = torch.cat([F.adaptive_max_pool1d(f, 1).squeeze(-1),
                              F.adaptive_avg_pool1d(f, 1).squeeze(-1)], dim=1)

        feat_out = model.feat_mlp(xf)
        fused    = torch.cat([feat_out, char_out], dim=1)
        logits   = model.clf(fused)

        logits[:, 1].sum().backward()

        if emb.grad is not None:
            # emb.grad shape: (batch, 256, 64) → magnitude across embedding dim
            pos_grad = emb.grad.abs().mean(dim=2).detach().cpu().numpy()  # (batch, 256)
            pos_importance += pos_grad.sum(axis=0)
            n_samples += len(xf)

        del xf, xc, emb, logits; gc.collect()

    if n_samples == 0:
        return pd.DataFrame()

    pos_importance /= n_samples
    df = pd.DataFrame({
        "position": list(range(MAX_URL_LEN)),
        "mean_grad_magnitude": pos_importance,
        "importance_rank": pd.Series(pos_importance).rank(ascending=False, method="dense").astype(int),
    }).sort_values("mean_grad_magnitude", ascending=False)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Analysis C2: Character vocabulary importance (embedding weight norms)
# ══════════════════════════════════════════════════════════════════════════════

def char_vocab_importance(model):
    """
    Compute L2 norm of each character's embedding vector.
    High-norm characters have more expressive representations → potentially
    more discriminative.  Maps char index back to ASCII character.
    """
    emb_weights = model.char_cnn.emb.weight.detach().cpu().numpy()  # (98, 64)
    norms = np.linalg.norm(emb_weights, axis=1)  # (98,)

    rows = []
    for idx in range(1, VOCAB_SIZE):  # 0 = padding
        ascii_code = idx + 31  # reverse of encode_url: code - 31 = idx, so code = idx + 31
        if 32 <= ascii_code < 128:
            char = chr(ascii_code)
        else:
            char = "<other>"
        rows.append({
            "char_idx": idx,
            "ascii_code": ascii_code,
            "character": char,
            "embedding_norm": float(norms[idx]),
        })

    return pd.DataFrame(rows).sort_values("embedding_norm", ascending=False)


# ══════════════════════════════════════════════════════════════════════════════
# Analysis C3: Top activated character n-gram patterns
# ══════════════════════════════════════════════════════════════════════════════

def extract_top_ngrams(X_char_phish, top_k=20):
    """
    From phishing-classified URL char sequences, extract the most common
    character n-grams (k=3,5,7) to surface patterns the CharCNN is sensitive to.
    These are the actual URL substrings most frequently seen in phishing URLs.
    """
    results = {}
    for k in [3, 5, 7]:
        ngram_counter: Counter = Counter()
        for seq in X_char_phish:
            # Decode: char_idx → ASCII char (reverse of encode_url)
            chars = []
            for idx in seq:
                if idx == 0:
                    break  # padding
                ascii_code = int(idx) + 31
                if 32 <= ascii_code < 128:
                    chars.append(chr(ascii_code))
                else:
                    chars.append("?")
            url_str = "".join(chars)
            # Extract k-grams
            for pos in range(len(url_str) - k + 1):
                ngram_counter[url_str[pos:pos+k]] += 1
        results[k] = ngram_counter.most_common(top_k)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("=" * 80)
    log.info("Experiment 14 — Feature Importance Analysis (CharCNN + FeatureMLP)")
    log.info(f"Device : {device}")
    log.info(f"Output : {OUT_DIR}")
    log.info("=" * 80)

    # ── Load model ─────────────────────────────────────────────────────────────
    log.info("\n[1/6] Loading trained BinaryURLPhishNet from Exp 2...")
    if not os.path.exists(MODEL_PATH):
        log.error(f"Model not found at {MODEL_PATH}. Run train_binary.py first.")
        sys.exit(1)
    if not os.path.exists(SCALER_PATH):
        log.error(f"Scaler not found at {SCALER_PATH}. Run train_binary.py first.")
        sys.exit(1)

    scaler = joblib.load(SCALER_PATH)
    ckpt   = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    feat_dim = scaler.mean_.shape[0]
    model    = BinaryURLPhishNet(feat_dim).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    log.info(f"  Model loaded (epoch {ckpt['epoch']}, F1={ckpt['f1_macro']:.4f})")
    log.info(f"  Feature dim: {feat_dim}  Expected: 67")

    # ── Load phishphresh test data ─────────────────────────────────────────────
    log.info("\n[2/6] Loading phishphresh test data...")
    train_flag = os.path.exists(os.path.join(DATA_DIR, "_train_complete.flag"))
    test_flag  = os.path.exists(os.path.join(DATA_DIR, "_test_complete.flag"))
    if not (train_flag and test_flag):
        log.error("Chunks incomplete — re-run train_multiclass_brand.py first.")
        sys.exit(1)

    Xf_te, Xc_te, y_te = load_test_data(DATA_DIR, scaler)
    log.info(f"  Test samples: {len(y_te):,}  "
             f"Benign: {(y_te==0).sum():,}  Phishing: {(y_te==1).sum():,}")

    preds_te = run_inference_np(model, Xf_te, Xc_te, device)
    baseline_f1  = f1_score(y_te, preds_te, average="macro", zero_division=0)
    baseline_acc = accuracy_score(y_te, preds_te)
    log.info(f"  Baseline model: Acc={baseline_acc*100:.2f}%  F1={baseline_f1:.4f}")

    # ── Analysis A: Permutation importance ─────────────────────────────────────
    log.info("\n[3/6] Analysis A — Permutation importance (67 features)...")
    if os.path.exists(PERM_CSV):
        log.info(f"  Checkpoint found — loading existing results from {PERM_CSV}")
        perm_df = pd.read_csv(PERM_CSV).sort_values("f1_drop_mean", ascending=False)
    else:
        log.info(f"  Using {len(y_te):,} test samples, {N_PERM_REPEATS} repeats per feature")
        perm_df = permutation_importance(model, Xf_te, Xc_te, y_te, device)
        perm_df.to_csv(PERM_CSV, index=False)
    log.info(f"  Saved -> {PERM_CSV}")
    log.info("  Top 10 features by permutation F1-drop:")
    for _, row in perm_df.head(10).iterrows():
        log.info(f"    {row['feature_name']:<40} F1-drop={row['f1_drop_mean']:.4f} ± {row['f1_drop_std']:.4f}")

    # ── Analysis B: Gradient saliency ──────────────────────────────────────────
    log.info(f"\n[4/6] Analysis B — Gradient × Input saliency ({GRAD_N_SAMPLES} samples)...")
    if os.path.exists(GRAD_CSV):
        log.info(f"  Checkpoint found — loading existing results from {GRAD_CSV}")
        grad_df = pd.read_csv(GRAD_CSV).sort_values("mean_abs_grad_x_input", ascending=False)
    else:
        rng  = np.random.default_rng(42)
        sel  = rng.choice(len(y_te), min(GRAD_N_SAMPLES, len(y_te)), replace=False)
        grad_df = gradient_saliency(model, Xf_te[sel], Xc_te[sel], y_te[sel], device)
        grad_df.to_csv(GRAD_CSV, index=False)
    log.info(f"  Saved -> {GRAD_CSV}")
    log.info("  Top 10 features by |grad × input|:")
    for _, row in grad_df.head(10).iterrows():
        log.info(f"    {row['feature_name']:<40} |g×x|={row['mean_abs_grad_x_input']:.4f}  "
                 f"signed={row['mean_grad_x_input']:.4f}")

    # ── Analysis C1: Char position importance ─────────────────────────────────
    log.info(f"\n[5/6] Analysis C — CharCNN Analysis...")
    log.info(f"  C1: Position importance (up to {CHAR_N_SAMPLES} phishing-predicted samples)...")
    if os.path.exists(CHAR_POS_CSV):
        log.info(f"  Checkpoint found — loading existing results from {CHAR_POS_CSV}")
        pos_df = pd.read_csv(CHAR_POS_CSV).sort_values("mean_grad_magnitude", ascending=False)
    else:
        pos_df = char_position_importance(model, Xf_te, Xc_te, preds_te, device)
    if len(pos_df) > 0:
        pos_df.sort_values("position").to_csv(CHAR_POS_CSV, index=False)
        log.info(f"  Saved -> {CHAR_POS_CSV}")
        log.info("  Top 15 most important URL character positions:")
        for _, row in pos_df.head(15).iterrows():
            log.info(f"    Position {int(row['position']):3d}  mean_grad={row['mean_grad_magnitude']:.5f}  "
                     f"rank={int(row['importance_rank'])}")
    else:
        log.warning("  Position analysis produced no output")

    # ── Analysis C2: Char vocab importance ─────────────────────────────────────
    log.info("  C2: Character vocabulary importance (embedding norms)...")
    vocab_df = char_vocab_importance(model)
    vocab_df.to_csv(CHAR_VOC_CSV, index=False)
    log.info(f"  Saved -> {CHAR_VOC_CSV}")
    log.info("  Top 20 highest-norm characters:")
    for _, row in vocab_df.head(20).iterrows():
        log.info(f"    '{row['character']}'  (idx={row['char_idx']}, ascii={row['ascii_code']})  "
                 f"norm={row['embedding_norm']:.4f}")

    # ── Analysis C3: Top n-gram patterns in phishing-predicted URLs ────────────
    log.info("  C3: Top character n-grams in phishing-classified URLs...")
    phish_mask = preds_te == 1
    Xc_phish   = Xc_te[phish_mask][:CHAR_N_SAMPLES]
    ngram_results = extract_top_ngrams(Xc_phish, top_k=25)

    with open(NGRAM_TXT, "w", encoding="utf-8") as f:
        f.write("Top character n-grams in phishing-classified URLs\n")
        f.write("=" * 60 + "\n\n")
        for k, ngrams in ngram_results.items():
            f.write(f"k={k} (kernel size {k} — {len(ngrams)} top patterns)\n")
            f.write("-" * 40 + "\n")
            for ng, cnt in ngrams:
                f.write(f"  '{ng}'  count={cnt:,}\n")
            f.write("\n")
    log.info(f"  Saved -> {NGRAM_TXT}")
    for k in [3, 5, 7]:
        top5 = [f"'{ng}'" for ng, _ in ngram_results[k][:5]]
        log.info(f"  k={k} top patterns: {', '.join(top5)}")

    # ── Write comprehensive report ──────────────────────────────────────────────
    log.info("\n[6/6] Writing feature importance report...")

    with open(REPORT_TXT, "w", encoding="utf-8") as rpt:
        rpt.write("=" * 70 + "\n")
        rpt.write("Experiment 14 — Feature Importance Report\n")
        rpt.write("Model: BinaryURLPhishNet (CharCNN + FeatureMLP, Exp 2)\n")
        rpt.write("=" * 70 + "\n\n")

        rpt.write(f"Baseline model: Acc={baseline_acc*100:.2f}%  F1={baseline_f1:.4f}\n")
        rpt.write(f"Test samples: {len(y_te):,}\n\n")

        rpt.write("=" * 70 + "\n")
        rpt.write("ANALYSIS A: Permutation Feature Importance (top 20)\n")
        rpt.write("=" * 70 + "\n")
        rpt.write(f"{'Rank':<5} {'Feature':<40} {'Group':<30} {'F1-drop':>8} {'±std':>8}\n")
        rpt.write("-" * 95 + "\n")
        for rank, (_, row) in enumerate(perm_df.head(20).iterrows(), 1):
            rpt.write(f"{rank:<5} {row['feature_name']:<40} {row['group']:<30} "
                      f"{row['f1_drop_mean']:>8.4f} {row['f1_drop_std']:>8.4f}\n")

        rpt.write("\n\nPer-group permutation importance:\n")
        rpt.write("-" * 50 + "\n")
        for group in FEATURE_GROUPS:
            grp_df = perm_df[perm_df["group"] == group]
            rpt.write(f"  {group:<35}  mean_drop={grp_df['f1_drop_mean'].mean():.4f}  "
                      f"max_drop={grp_df['f1_drop_mean'].max():.4f}\n")

        rpt.write("\n\n")
        rpt.write("=" * 70 + "\n")
        rpt.write("ANALYSIS B: Gradient × Input Saliency (top 20)\n")
        rpt.write("=" * 70 + "\n")
        rpt.write(f"{'Rank':<5} {'Feature':<40} {'|g×x|':>10} {'signed g×x':>12}\n")
        rpt.write("-" * 70 + "\n")
        for rank, (_, row) in enumerate(grad_df.head(20).iterrows(), 1):
            direction = "+phishing" if row["mean_grad_x_input"] > 0 else "-benign"
            rpt.write(f"{rank:<5} {row['feature_name']:<40} "
                      f"{row['mean_abs_grad_x_input']:>10.5f} "
                      f"{row['mean_grad_x_input']:>12.5f}  {direction}\n")

        rpt.write("\n\n")
        rpt.write("=" * 70 + "\n")
        rpt.write("ANALYSIS C: CharCNN Character-Level Analysis\n")
        rpt.write("=" * 70 + "\n\n")

        rpt.write("C1. Most important URL character positions (top 20):\n")
        rpt.write("-" * 40 + "\n")
        if len(pos_df) > 0:
            for _, row in pos_df.head(20).iterrows():
                rpt.write(f"  Position {int(row['position']):3d}  mean_grad={row['mean_grad_magnitude']:.5f}\n")
        else:
            rpt.write("  (no data)\n")

        rpt.write("\nC2. Characters with highest embedding norms (top 20):\n")
        rpt.write("-" * 40 + "\n")
        for _, row in vocab_df.head(20).iterrows():
            rpt.write(f"  '{row['character']:<3}' (ascii {row['ascii_code']:3d})  "
                      f"norm={row['embedding_norm']:.4f}\n")

        rpt.write("\nC3. Top character n-grams in phishing-classified URLs:\n")
        rpt.write("-" * 40 + "\n")
        for k, ngrams in ngram_results.items():
            rpt.write(f"\n  kernel-size k={k}:\n")
            for ng, cnt in ngrams[:15]:
                rpt.write(f"    '{ng}'  ({cnt:,})\n")

        rpt.write("\n\n")
        rpt.write("=" * 70 + "\n")
        rpt.write("KEY FINDINGS SUMMARY\n")
        rpt.write("=" * 70 + "\n\n")

        top5_perm = perm_df.head(5)["feature_name"].tolist()
        top5_grad = grad_df.head(5)["feature_name"].tolist()
        top5_chars = [row["character"] for _, row in vocab_df.head(5).iterrows()]

        rpt.write(f"Top 5 features by permutation importance: {', '.join(top5_perm)}\n\n")
        rpt.write(f"Top 5 features by gradient saliency:      {', '.join(top5_grad)}\n\n")
        rpt.write(f"Top 5 characters by embedding norm:        {', '.join(top5_chars)}\n\n")

        overlap = set(top5_perm) & set(top5_grad)
        rpt.write(f"Features in BOTH top-5 lists (stable importance): {overlap}\n\n")

        # Group-level summary
        rpt.write("Most impactful feature groups (by avg permutation F1-drop):\n")
        group_scores = []
        for group in FEATURE_GROUPS:
            grp_df = perm_df[perm_df["group"] == group]
            group_scores.append((group, grp_df["f1_drop_mean"].mean()))
        group_scores.sort(key=lambda x: x[1], reverse=True)
        for rank, (group, score) in enumerate(group_scores, 1):
            rpt.write(f"  {rank}. {group:<35}  avg F1-drop={score:.4f}\n")

    log.info(f"Report saved -> {REPORT_TXT}")
    log.info("=" * 80)
    log.info("Experiment 14 complete.")
    log.info("=" * 80)
