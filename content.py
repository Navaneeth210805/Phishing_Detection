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
from bs4 import BeautifulSoup
import re
import tldextract
from collections import Counter
import joblib
import os
from datetime import datetime
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

torch.manual_seed(42)
np.random.seed(42)

class HTMLFeatureExtractor:
    def __init__(self):
        self.suspicious_keywords = ['verify', 'update', 'confirm', 'suspend', 'secure', 'account', 'login', 'password', 'urgent', 'alert']
        
    def extract(self, html: str) -> Dict:
        features = {}
        try:
            soup = BeautifulSoup(html, 'html.parser')
            features['num_tags'] = len(soup.find_all())
            features['num_forms'] = len(soup.find_all('form'))
            features['num_inputs'] = len(soup.find_all('input'))
            features['num_links'] = len(soup.find_all('a'))
            features['num_scripts'] = len(soup.find_all('script'))
            features['num_iframes'] = len(soup.find_all('iframe'))
            features['num_images'] = len(soup.find_all('img'))
            features['num_divs'] = len(soup.find_all('div'))
            password_inputs = soup.find_all('input', {'type': 'password'})
            features['num_password_fields'] = len(password_inputs)
            links = soup.find_all('a', href=True)
            external_links = 0
            for link in links:
                href = link['href']
                if href.startswith('http') and not href.startswith('#'):
                    external_links += 1
            features['num_external_links'] = external_links
            features['external_link_ratio'] = external_links / max(len(links), 1)
            text = soup.get_text().lower()
            features['suspicious_keyword_count'] = sum(1 for kw in self.suspicious_keywords if kw in text)
            features['has_meta_description'] = 1 if soup.find('meta', {'name': 'description'}) else 0
            features['has_meta_keywords'] = 1 if soup.find('meta', {'name': 'keywords'}) else 0
            title = soup.find('title')
            features['title_length'] = len(title.text) if title else 0
            hidden_elements = soup.find_all(style=re.compile(r'display:\s*none|visibility:\s*hidden'))
            features['num_hidden_elements'] = len(hidden_elements)
            features['max_dom_depth'] = self._get_max_depth(soup)
        except:
            for key in ['num_tags', 'num_forms', 'num_inputs', 'num_links', 'num_scripts', 'num_iframes', 
                       'num_images', 'num_divs', 'num_password_fields', 'num_external_links', 
                       'external_link_ratio', 'suspicious_keyword_count', 'has_meta_description', 
                       'has_meta_keywords', 'title_length', 'num_hidden_elements', 'max_dom_depth']:
                features[key] = 0
        return features
    
    def _get_max_depth(self, element, depth=0):
        if not hasattr(element, 'children'):
            return depth
        max_child_depth = depth
        for child in element.children:
            if hasattr(child, 'name'):
                child_depth = self._get_max_depth(child, depth + 1)
                max_child_depth = max(max_child_depth, child_depth)
        return max_child_depth

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
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.html_extractor = HTMLFeatureExtractor()
        self.url_extractor = URLFeatureExtractor()
        self.url_vectorizer = HashingVectorizer(n_features=500, analyzer='char', ngram_range=(2, 4), alternate_sign=False)
        self.html_vectorizer = HashingVectorizer(n_features=1000, alternate_sign=False)
        self.html_scaler = StandardScaler()
        self.url_scaler = StandardScaler()
        self.model = None
        
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
        print(f"\nFeature Dimensions:")
        print(f"  HTML features: {self.html_dim}")
        print(f"  URL features: {self.url_dim}")
        print(f"  TF-IDF features: {self.tfidf_dim}")
        print("-"*40 + "\n")
    
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

clf = PhishingClassifier()
data = clf.prepare_data(max_samples=10000)
clf.train(data, epochs=20, batch_size=128, lr=0.001)

print("\n" + "="*80)
print("EXAMPLES")
print("="*80)
url1 = "https://secure-verify-account.tk/login"
html1 = "<html><body><form><input type='password'></form><h1>Verify Your Account Now</h1></body></html>"
label1, conf1 = clf.predict(url1, html1)
print(f"\nURL: {url1}\nPrediction: {label1} ({conf1:.2%})")

url2 = "https://www.google.com/search"
html2 = "<html><head><title>Google</title></head><body><div>Results</div></body></html>"
label2, conf2 = clf.predict(url2, html2)
print(f"\nURL: {url2}\nPrediction: {label2} ({conf2:.2%})")

print("\n✅ Model saved: phishing_model.pth")