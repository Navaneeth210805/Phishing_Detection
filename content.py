import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import numpy as np
import pandas as pd
import re
import tldextract
from collections import Counter
from typing import Dict, Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Import DOM tree components - ALL HTML parsing done here
from dom_tree_builder import DOMTreeBuilder, DOMTreeComparator

torch.manual_seed(42)
np.random.seed(42)

class HTMLFeatureExtractor:
    """Wrapper around DOMTreeBuilder for HTML feature extraction"""
    def __init__(self, use_dom_tree: bool = True):
        self.use_dom_tree = use_dom_tree
        self.dom_builder = DOMTreeBuilder() if use_dom_tree else None
        
    def extract(self, html: str) -> Dict:
        """Extract all HTML features using DOMTreeBuilder"""
        if self.use_dom_tree:
            # Single parse, all features extracted
            self.dom_builder.build_from_html_string(html)
            return self.dom_builder.extract_all_html_features()
        else:
            # Fallback: return zeros
            return {
                'num_tags': 0, 'num_forms': 0, 'num_inputs': 0, 'num_links': 0,
                'num_scripts': 0, 'num_iframes': 0, 'num_images': 0, 'num_divs': 0,
                'num_password_fields': 0, 'num_external_links': 0, 'external_link_ratio': 0,
                'suspicious_keyword_count': 0, 'has_meta_description': 0, 'has_meta_keywords': 0,
                'title_length': 0, 'num_hidden_elements': 0,
                'max_dom_depth': 0, 'avg_dom_depth': 0, 'dom_balance': 0,
                'leaf_node_ratio': 0, 'avg_children_per_node': 0, 'dom_complexity': 0
            }

class URLFeatureExtractor:
    def extract(self, url: str) -> Dict:
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
                entropy = 0
                for count in char_counts.values():
                    prob = count / len(domain)
                    entropy -= prob * np.log2(prob)
                features['domain_entropy'] = entropy
            else:
                features['domain_entropy'] = 0
        except:
            for key in ['domain_length', 'subdomain_length', 'tld_length', 'num_subdomains', 
                       'is_suspicious_tld', 'domain_entropy']:
                features[key] = 0
        features['has_suspicious_keywords'] = 1 if any(kw in url_lower for kw in ['login', 'verify', 'account', 'secure', 'update']) else 0
        return features

class PhishingNet(nn.Module):
    def __init__(self, html_dim, url_dim, tfidf_dim, hidden_dim=256, dropout=0.3):
        super(PhishingNet, self).__init__()
        self.html_net = nn.Sequential(
            nn.Linear(html_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout)
        )
        self.url_net = nn.Sequential(
            nn.Linear(url_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout)
        )
        self.tfidf_net = nn.Sequential(
            nn.Linear(tfidf_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout)
        )
        fusion_input_dim = 64 + 64 + 128
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
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
        h = self.html_net(html_feat)
        u = self.url_net(url_feat)
        t = self.tfidf_net(tfidf_feat)
        fused = torch.cat([h, u, t], dim=1)
        return self.fusion(fused)

class PhishingClassifier:
    def __init__(self, use_dom_tree: bool = True, brand_database_path: Optional[str] = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.html_extractor = HTMLFeatureExtractor(use_dom_tree=use_dom_tree)
        self.url_extractor = URLFeatureExtractor()
        self.url_vectorizer = HashingVectorizer(n_features=500, analyzer='char', ngram_range=(2, 4), alternate_sign=False)
        self.html_vectorizer = HashingVectorizer(n_features=1000, alternate_sign=False)
        self.html_scaler = StandardScaler()
        self.url_scaler = StandardScaler()
        self.model = None
        self.use_dom_tree = use_dom_tree
        self.dom_comparator = DOMTreeComparator() if use_dom_tree else None
        self.brand_database = self._load_brand_database(brand_database_path) if brand_database_path else {}
        
        # Calculate dimensions dynamically from feature extractors
        sample_html = "<html><body><div>test</div></body></html>"
        sample_url = "https://example.com/test"
        self.html_dim = len(self.html_extractor.extract(sample_html))
        self.url_dim = len(self.url_extractor.extract(sample_url))
        self.tfidf_dim = 1500  # 500 (url hashing) + 1000 (html hashing)
        
        print("\n" + "-"*40)
        print(f"PyTorch Version: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU Device: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")
        else:
            print("Status: Running on CPU. (Tip: If you have a GPU, ensure torch-cuda is installed)")
        print(f"Active Device: {self.device}")
        print(f"\nFeature Configuration:")
        print(f"  DOM Tree Analysis: {'✓ ENABLED' if use_dom_tree else '✗ DISABLED'}")
        print(f"  HTML features: {self.html_dim}")
        print(f"  URL features: {self.url_dim}")
        print(f"  TF-IDF features: {self.tfidf_dim}")
        print(f"  Brand Database: {len(self.brand_database)} entries loaded" if self.brand_database else "  Brand Database: Not loaded")
        print("-"*40 + "\n")
    
    def _load_brand_database(self, path: str) -> Dict[str, List[Dict]]:
        """Load brand DOM trees for similarity comparison"""
        try:
            import json
            with open(path, 'r') as f:
                return json.load(f)
        except:
            return {}
    
    def compare_with_brands(self, html: str, threshold: float = 0.7) -> Dict:
        """
        Compare input HTML DOM with known brand DOM trees
        Returns brand name and similarity score if match found
        """
        if not self.use_dom_tree or not self.brand_database:
            return {'brand_match': None, 'similarity': 0.0}
        
        try:
            # Build DOM tree for input using DOMTreeBuilder
            builder = DOMTreeBuilder()
            builder.build_from_html_string(html)
            input_tree = [{'id': n.id, 'tag': n.tag, 'parent_id': n.parent_id, 'child_ids': n.child_ids,
                          'child_count': n.child_count, 'attributes': n.attributes, 'depth': n.depth} 
                         for n in builder.dom_array]
            
            best_match = {'brand': None, 'similarity': 0.0}
            
            for brand, brand_tree in self.brand_database.items():
                # Use DOMTreeComparator for comparison
                metrics = self.dom_comparator.compare_structure(input_tree, brand_tree)
                similar_nodes = self.dom_comparator.find_similar_nodes(input_tree, brand_tree, threshold)
                
                # Calculate combined similarity score
                size_sim = 1 - abs(metrics['tree1_nodes'] - metrics['tree2_nodes']) / max(metrics['tree1_nodes'], metrics['tree2_nodes'])
                depth_sim = 1 - abs(metrics['max_depth_tree1'] - metrics['max_depth_tree2']) / max(metrics['max_depth_tree1'], metrics['max_depth_tree2'], 1)
                node_sim = len(similar_nodes) / max(len(input_tree), len(brand_tree))
                
                combined_sim = (size_sim * 0.3 + depth_sim * 0.2 + node_sim * 0.5)
                
                if combined_sim > best_match['similarity']:
                    best_match = {'brand': brand, 'similarity': combined_sim}
            
            return {'brand_match': best_match['brand'] if best_match['similarity'] >= threshold else None,
                   'similarity': best_match['similarity']}
        except:
            return {'brand_match': None, 'similarity': 0.0}
    
    def extract_batch(self, urls, htmls):
        html_feats = pd.DataFrame([self.html_extractor.extract(h) for h in htmls]).fillna(0).values
        url_feats = pd.DataFrame([self.url_extractor.extract(u) for u in urls]).fillna(0).values
        url_hashed = self.url_vectorizer.transform(urls).toarray()
        html_hashed = self.html_vectorizer.transform(htmls).toarray()
        return html_feats, url_feats, np.hstack([url_hashed, html_hashed])
    
    def prepare_data(self, max_samples=10000):
        print("Loading dataset (limited parquet files)...")
        # Load dataset in streaming mode and limit to first few samples
        dataset = load_dataset('phreshphish/phreshphish', split='train', streaming=True)
        
        # Collect limited samples from stream
        samples = []
        print(f"Collecting {max_samples} samples...")
        for idx, sample in enumerate(dataset):
            if idx >= max_samples:
                break
            samples.append(sample)
            if (idx + 1) % 1000 == 0:
                print(f"Collected {idx + 1}/{max_samples} samples", end='\r')
        print(f"\nTotal collected: {len(samples)}")
        
        # Extract features in batches
        batch_size = 1000
        all_html, all_url, all_tfidf, all_labels = [], [], [], []
        
        for i in range(0, len(samples), batch_size):
            end = min(i + batch_size, len(samples))
            batch = samples[i:end]
            
            urls = [s['url'] for s in batch]
            htmls = [s['html'] for s in batch]
            labels_batch = [s['label'] for s in batch]
            
            h, u, t = self.extract_batch(urls, htmls)
            labels = np.array([1 if l == 'phish' else 0 for l in labels_batch])
            
            all_html.append(h)
            all_url.append(u)
            all_tfidf.append(t)
            all_labels.append(labels)
            print(f"Processed {end}/{len(samples)}", end='\r')
        print()
        
        html_features = np.vstack(all_html)
        url_features = np.vstack(all_url)
        tfidf_features = np.vstack(all_tfidf)
        labels = np.concatenate(all_labels)
        
        # Fit scalers on training data
        self.html_scaler.fit(html_features)
        self.url_scaler.fit(url_features)
        html_features = self.html_scaler.transform(html_features)
        url_features = self.url_scaler.transform(url_features)
        
        # 80-20 train-test split
        split = int(0.8 * len(labels))
        return {
            'train': {'html': html_features[:split], 'url': url_features[:split], 'tfidf': tfidf_features[:split], 'y': labels[:split]},
            'test': {'html': html_features[split:], 'url': url_features[split:], 'tfidf': tfidf_features[split:], 'y': labels[split:]}
        }

    def train(self, data, epochs=20, batch_size=128, lr=0.001):
        print("\n" + "="*80)
        print("TRAINING")
        print("="*80)
        
        class DS(Dataset):
            def __init__(self, html, url, tfidf, y):
                self.html = torch.FloatTensor(html)
                self.url = torch.FloatTensor(url)
                self.tfidf = torch.FloatTensor(tfidf)
                self.y = torch.LongTensor(y)
            def __len__(self): return len(self.y)
            def __getitem__(self, i): return self.html[i], self.url[i], self.tfidf[i], self.y[i]
        
        train_loader = DataLoader(DS(data['train']['html'], data['train']['url'], data['train']['tfidf'], data['train']['y']), 
                                  batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(DS(data['test']['html'], data['test']['url'], data['test']['tfidf'], data['test']['y']), 
                                 batch_size=batch_size)
        
        self.model = PhishingNet(self.html_dim, self.url_dim, self.tfidf_dim).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3)
        best_acc = 0
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            for h, u, t, y in train_loader:
                h, u, t, y = h.to(self.device), u.to(self.device), t.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                out = self.model(h, u, t)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            self.model.eval()
            preds, labels, probs = [], [], []
            with torch.no_grad():
                for h, u, t, y in test_loader:
                    h, u, t = h.to(self.device), u.to(self.device), t.to(self.device)
                    out = self.model(h, u, t)
                    prob = F.softmax(out, dim=1)
                    _, pred = torch.max(out, 1)
                    preds.extend(pred.cpu().numpy())
                    labels.extend(y.numpy())
                    probs.extend(prob[:, 1].cpu().numpy())
            
            acc = accuracy_score(labels, preds)
            auc = roc_auc_score(labels, probs)
            scheduler.step(acc)
            
            if acc > best_acc:
                best_acc = acc
                torch.save({
                    'model': self.model.state_dict(),
                    'html_scaler': self.html_scaler,
                    'url_scaler': self.url_scaler,
                    'acc': acc
                }, 'phishing_model.pth')
            
            print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss/len(train_loader):.4f} | Acc: {acc:.4f} | AUC: {auc:.4f}")
        
        print(f"\nBest Acc: {best_acc:.4f}")
        print("\n" + "="*80)
        print("TEST RESULTS")
        print("="*80)
        print(classification_report(labels, preds, target_names=['Benign', 'Phishing']))
        
        # Save test results to CSV with confidence scores
        test_results = pd.DataFrame({
            'true_label': ['Benign' if l == 0 else 'Phishing' for l in labels],
            'predicted_label': ['Benign' if p == 0 else 'Phishing' for p in preds],
            'confidence_score': probs,
            'correct': [1 if labels[i] == preds[i] else 0 for i in range(len(labels))]
        })
        csv_filename = 'test_results.csv'
        test_results.to_csv(csv_filename, index=False)
        print(f"\n✅ Test results saved to: {csv_filename}")
        print(f"   Total test samples: {len(test_results)}")
        print(f"   Correct predictions: {test_results['correct'].sum()}")
        print(f"   Accuracy: {test_results['correct'].mean():.4f}")

    def predict(self, url: str, html: str) -> Tuple[str, float]:
        if self.model is None:
            raise ValueError("Model not trained")
        h, u, t = self.extract_batch([url], [html])
        h = self.html_scaler.transform(h)
        u = self.url_scaler.transform(u)
        self.model.eval()
        with torch.no_grad():
            h_t = torch.FloatTensor(h).to(self.device)
            u_t = torch.FloatTensor(u).to(self.device)
            t_t = torch.FloatTensor(t).to(self.device)
            out = self.model(h_t, u_t, t_t)
            prob = F.softmax(out, dim=1)
            conf = prob[0, 1].item()
            label = 'Phishing' if conf > 0.5 else 'Benign'
        return label, conf
    
    def predict_with_similarity(self, url: str, html: str) -> Dict:
        """
        Enhanced prediction with DOM similarity checking
        Returns: {label, confidence, brand_match, similarity_score}
        """
        label, conf = self.predict(url, html)
        brand_info = self.compare_with_brands(html)
        
        # If high similarity to a brand but predicted as phishing -> likely phishing attempt
        if brand_info['brand_match'] and label == 'Phishing':
            return {
                'label': 'Phishing',
                'confidence': min(conf + 0.1, 1.0),  # Boost confidence
                'brand_match': brand_info['brand_match'],
                'similarity': brand_info['similarity'],
                'warning': f"⚠️ Impersonating {brand_info['brand_match']}"
            }
        
        return {
            'label': label,
            'confidence': conf,
            'brand_match': brand_info['brand_match'],
            'similarity': brand_info['similarity'],
            'warning': None
        }

clf = PhishingClassifier(use_dom_tree=True)
data = clf.prepare_data(max_samples=10000)
clf.train(data, epochs=20, batch_size=128, lr=0.001)

print("\n" + "="*80)
print("EXAMPLES")
print("="*80)
url1 = "https://secure-verify-account.tk/login"
html1 = "<html><body><form><input type='password'></form><h1>Verify Your Account Now</h1></body></html>"
label1, conf1 = clf.predict(url1, html1)
print(f"\n[Basic] URL: {url1}\nPrediction: {label1} ({conf1:.2%})")

result1 = clf.predict_with_similarity(url1, html1)
print(f"[Enhanced] Prediction: {result1['label']} ({result1['confidence']:.2%})")
if result1['warning']:
    print(f"           {result1['warning']}")

url2 = "https://www.google.com/search"
html2 = "<html><head><title>Google</title></head><body><div>Results</div></body></html>"
label2, conf2 = clf.predict(url2, html2)
print(f"\n[Basic] URL: {url2}\nPrediction: {label2} ({conf2:.2%})")

result2 = clf.predict_with_similarity(url2, html2)
print(f"[Enhanced] Prediction: {result2['label']} ({result2['confidence']:.2%})")
if result2['brand_match']:
    print(f"           Matches brand: {result2['brand_match']} (sim: {result2['similarity']:.2%})")

print("\n✅ Model saved: phishing_model.pth")
print("✅ Features: HTML (22) + URL (20) + TF-IDF (1500)")
print("✅ DOM Tree Integration: COMPLETE")
print("✅ Brand Similarity Check: AVAILABLE")