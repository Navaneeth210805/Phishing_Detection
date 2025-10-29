#!/usr/bin/env python3
"""
HYBRID PHISHING DETECTOR
========================

Combines:
1. Rule-based CSE mapping from test.py (keyword matching, typosquatting)
2. ML model prediction from clean_real_data_model.py (Phishing/Suspected classification)
3. Evidence collection from test.py (screenshots, WHOIS)

WORKFLOW:
1. Train ML model on combined_dataset.csv
2. Load 1M shortlisting domains
3. Apply rule-based CSE mapping
4. For CSE-targeted domains, predict Phishing/Suspected with ML model
5. Collect evidence (screenshots + WHOIS)
6. Generate submission
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import warnings
import re
import tldextract
import os
import sys
import time
import threading
from datetime import datetime
from pathlib import Path
import Levenshtein
import joblib
from collections import Counter
import urllib.parse

warnings.filterwarnings('ignore')

# Playwright
try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

# WHOIS
try:
    import whois as python_whois
    WHOIS_AVAILABLE = True
except ImportError:
    WHOIS_AVAILABLE = False

# PDF
try:
    from PIL import Image
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader
    import io
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)


# ==================== NEURAL NETWORK ====================

class PhishingDataset(Dataset):
    """Custom PyTorch Dataset for phishing detection."""
    
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class PhishingNet(nn.Module):
    """Neural Network for binary phishing classification."""

    def __init__(self, input_size=51, hidden_sizes=[128, 64, 32, 16], dropout_rate=0.3):
        super(PhishingNet, self).__init__()
        
        # Input layer
        self.input_layer = nn.Linear(input_size, hidden_sizes[0])
        self.input_dropout = nn.Dropout(dropout_rate)
        self.input_bn = nn.BatchNorm1d(hidden_sizes[0])
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        self.hidden_dropouts = nn.ModuleList()
        self.hidden_bns = nn.ModuleList()
        
        for i in range(len(hidden_sizes) - 1):
            self.hidden_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            self.hidden_dropouts.append(nn.Dropout(dropout_rate))
            self.hidden_bns.append(nn.BatchNorm1d(hidden_sizes[i + 1]))
        
        # Output layer - binary classification
        self.output = nn.Linear(hidden_sizes[-1], 2)  # Suspected vs Phishing
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        # Input layer
        x = F.relu(self.input_layer(x))
        x = self.input_bn(x)
        x = self.input_dropout(x)
        
        # Hidden layers
        for layer, dropout, bn in zip(self.hidden_layers, self.hidden_dropouts, self.hidden_bns):
            x = F.relu(layer(x))
            x = bn(x)
            x = dropout(x)
        
        # Output
        return self.output(x)


# ==================== CSE MAPPER (from test.py) ====================

class CSEMapper:
    """
    Rule-based CSE mapper using keyword matching and typosquatting detection.
    From test.py logic.
    """
    
    def __init__(self):
        # CSE keywords and official domains
        self.cse_data = {
            'State Bank of India (SBI)': {
                'keywords': ['sbi', 'statebank', 'yono', 'onlinesbi', 'sbicard', 'statebankof'],
                'official': ['sbi.co.in', 'onlinesbi.sbi', 'sbicard.com']
            },
            'ICICI Bank': {
                'keywords': ['icici', 'icicibank', 'icicidirect', 'icicipru'],
                'official': ['icicibank.com', 'icicidirect.com']
            },
            'HDFC Bank': {
                'keywords': ['hdfc', 'hdfcbank', 'hdfcergo'],
                'official': ['hdfcbank.com', 'hdfc.com']
            },
            'Punjab National Bank (PNB)': {
                'keywords': ['pnb', 'punjabnational', 'netpnb', 'pnbindia'],
                'official': ['pnbindia.in', 'netpnb.com']
            },
            'Bank of Baroda (BoB)': {
                'keywords': ['baroda', 'bankofbaroda', 'bobibanking'],
                'official': ['bankofbaroda.in', 'bobibanking.com']
            },
            'Airtel': {
                'keywords': ['airtel', 'airtelmoney', 'airtelbank', 'airtelpayments'],
                'official': ['airtel.in', 'airtel.com']
            },
            'National Informatics Centre (NIC)': {
                'keywords': ['nicgov', 'govnic', 'nicindia', 'kavachmail', 'emailgov'],
                'official': ['nic.gov.in', 'email.gov.in']
            },
            'Registrar General and Census Commissioner of India (RGCCI)': {
                'keywords': ['censusindia', 'crsorgi', 'rgcci'],
                'official': ['dc.crsorgi.gov.in']
            },
            'Indian Railway Catering and Tourism Corporation (IRCTC)': {
                'keywords': ['irctc', 'indianrailway'],
                'official': ['irctc.co.in']
            },
            'Indian Oil Corporation Limited (IOCL)': {
                'keywords': ['indianoil', 'iocl'],
                'official': ['iocl.com']
            }
        }
    
    def map_domain_to_cse(self, domain):
        """
        Map domain to CSE using keyword matching and typosquatting.
        Returns: (cse_name, confidence_score) or (None, 0) if no match
        """
        domain_lower = domain.lower()
        extracted = tldextract.extract(domain)
        domain_part = extracted.domain.lower()
        tld = extracted.suffix.lower()
        
        best_cse = None
        best_score = 0
        
        # Check each CSE
        for cse, data in self.cse_data.items():
            score = 0
            
            # Check keywords
            for keyword in data['keywords']:
                if len(keyword) >= 4:  # Only check keywords 4+ chars
                    # Exact match
                    if domain_part == keyword:
                        score += 30
                    # Starts/ends with
                    elif domain_part.startswith(keyword) or domain_part.endswith(keyword):
                        score += 25
                    # Contains
                    elif keyword in domain_part and len(domain_part) > len(keyword):
                        score += 20
            
            # Check typosquatting against official domains
            for official in data['official']:
                official_domain = tldextract.extract(official).domain
                official_tld = tldextract.extract(official).suffix
                
                # Skip if it's the official domain itself
                if domain_part == official_domain and tld == official_tld:
                    return None, 0  # Official domain, not phishing
                
                # Typosquatting detection
                similarity = Levenshtein.ratio(domain_part, official_domain)
                if similarity >= 0.75:
                    score += int(similarity * 30)
            
            if score > best_score:
                best_score = score
                best_cse = cse
        
        # Only return CSE if score is significant (>= 20)
        if best_score >= 20:
            return best_cse, best_score
        return None, 0


# ==================== FEATURE EXTRACTOR ====================

class FeatureExtractor:
    """Extract 51 features from domains."""
    
    def __init__(self):
        # Legitimate domains and keywords for CSE matching
        self.legitimate_domains = [
            'sbi.co.in', 'onlinesbi.sbi', 'sbicard.com',
            'icicibank.com', 'icicidirect.com',
            'hdfcbank.com', 'hdfc.com',
            'pnbindia.in', 'netpnb.com',
            'bankofbaroda.in', 'bobibanking.com',
            'airtel.in', 'airtel.com',
            'nic.gov.in', 'email.gov.in',
            'dc.crsorgi.gov.in',
            'irctc.co.in',
            'iocl.com'
        ]
        
        self.legitimate_keywords = [
            'sbi', 'statebank', 'icici', 'hdfc', 'pnb', 'baroda', 
            'airtel', 'nic', 'crsorgi', 'irctc', 'iocl'
        ]
    
    def extract_features(self, domain):
        """Extract 51 features from a single domain."""
        features = {}
        domain_clean = str(domain).lower().strip()
        
        # =================== BASIC FEATURES (1-15) ===================
        features['domain_length'] = len(domain_clean)
        features['dot_count'] = domain_clean.count('.')
        features['dash_count'] = domain_clean.count('-')
        features['underscore_count'] = domain_clean.count('_')
        features['digit_count'] = sum(c.isdigit() for c in domain_clean)
        features['uppercase_count'] = 0  # Already lowercased
        features['digit_ratio'] = features['digit_count'] / max(len(domain_clean), 1)
        features['special_char_ratio'] = (features['dash_count'] + features['underscore_count']) / max(len(domain_clean), 1)
        
        # Domain structure
        parts = domain_clean.split('.')
        features['domain_parts'] = len(parts)
        features['longest_part'] = max(len(part) for part in parts) if parts else 0
        features['shortest_part'] = min(len(part) for part in parts) if parts else 0
        features['avg_part_length'] = np.mean([len(part) for part in parts]) if parts else 0
        
        # Character patterns
        features['vowel_count'] = sum(1 for c in domain_clean if c in 'aeiou')
        features['consonant_count'] = sum(1 for c in domain_clean if c.isalpha() and c not in 'aeiou')
        features['has_consecutive_chars'] = 1 if any(domain_clean[j] == domain_clean[j+1] for j in range(len(domain_clean)-1) if len(domain_clean) > 1) else 0
        
        # =================== ENTROPY FEATURES (16-25) ===================
        # Character entropy
        if len(domain_clean) > 0:
            char_counts = Counter(domain_clean)
            entropy = 0
            for count in char_counts.values():
                prob = count / len(domain_clean)
                if prob > 0:
                    entropy -= prob * np.log2(prob)
            features['char_entropy'] = entropy
        else:
            features['char_entropy'] = 0
        
        # Part entropy
        if len(parts) > 0:
            part_lengths = [len(part) for part in parts]
            features['part_length_variance'] = np.var(part_lengths)
            features['part_length_std'] = np.std(part_lengths)
        else:
            features['part_length_variance'] = 0
            features['part_length_std'] = 0
        
        # Bigram entropy
        bigrams = [domain_clean[j:j+2] for j in range(len(domain_clean)-1)]
        if bigrams:
            bigram_counts = Counter(bigrams)
            bigram_entropy = 0
            for count in bigram_counts.values():
                prob = count / len(bigrams)
                if prob > 0:
                    bigram_entropy -= prob * np.log2(prob)
            features['bigram_entropy'] = bigram_entropy
        else:
            features['bigram_entropy'] = 0
        
        features['unique_char_ratio'] = len(set(domain_clean)) / max(len(domain_clean), 1)
        
        # Transition entropy
        if len(domain_clean) > 1:
            transitions = [domain_clean[j:j+2] for j in range(len(domain_clean)-1)]
            transition_counts = Counter(transitions)
            transition_entropy = 0
            for count in transition_counts.values():
                prob = count / len(transitions)
                if prob > 0:
                    transition_entropy -= prob * np.log2(prob)
            features['transition_entropy'] = transition_entropy
        else:
            features['transition_entropy'] = 0
        
        # Trigram patterns
        trigrams = [domain_clean[j:j+3] for j in range(len(domain_clean)-2)]
        features['trigram_count'] = len(trigrams)
        features['unique_trigram_ratio'] = len(set(trigrams)) / max(len(trigrams), 1) if trigrams else 0
        
        # Consonant/vowel patterns
        features['vowel_consonant_ratio'] = features['vowel_count'] / max(features['consonant_count'], 1)
        features['consonant_clusters'] = len(re.findall(r'[bcdfghjklmnpqrstvwxyz]{3,}', domain_clean))
        
        # =================== TLD FEATURES (26-30) ===================
        try:
            tld_extract = tldextract.extract(domain_clean)
            tld = tld_extract.suffix.lower()
            
            features['tld_length'] = len(tld)
            features['is_common_tld'] = 1 if tld in ['com', 'org', 'net', 'edu', 'gov', 'mil'] else 0
            features['is_country_tld'] = 1 if len(tld) == 2 and tld.isalpha() else 0
            features['is_suspicious_tld'] = 1 if tld in ['tk', 'ml', 'ga', 'cf', 'bit', 'click', 'top'] else 0
            features['is_indian_tld'] = 1 if tld in ['in', 'co.in', 'org.in', 'net.in', 'gov.in'] else 0
        except:
            features['tld_length'] = 0
            features['is_common_tld'] = 0
            features['is_country_tld'] = 0
            features['is_suspicious_tld'] = 0
            features['is_indian_tld'] = 0
        
        # =================== CSE PATTERN FEATURES (31-40) ===================
        # Legitimate domain similarity
        max_similarity = 0
        keyword_matches = 0
        
        for legit_domain in self.legitimate_domains:
            if domain_clean == legit_domain.lower():
                max_similarity = 1.0
                break
            # Character set similarity
            set1 = set(domain_clean)
            set2 = set(legit_domain.lower())
            if set1 and set2:
                similarity = len(set1.intersection(set2)) / len(set1.union(set2))
                max_similarity = max(max_similarity, similarity)
        
        features['legitimate_domain_similarity'] = max_similarity
        features['matches_legitimate_domain'] = 1 if max_similarity > 0.8 else 0
        
        # Keyword matching
        for keyword in self.legitimate_keywords:
            if keyword.lower() in domain_clean:
                keyword_matches += 1
        
        features['legitimate_keyword_count'] = keyword_matches
        features['contains_legitimate_keyword'] = 1 if keyword_matches > 0 else 0
        
        # Brand variations
        common_brands = ['bank', 'pay', 'secure', 'login', 'account', 'mail', 'google', 'microsoft', 'apple']
        features['brand_keyword_count'] = sum(1 for brand in common_brands if brand in domain_clean)
        features['has_brand_variation'] = 1 if features['brand_keyword_count'] > 0 else 0
        
        # Sector-specific patterns
        features['finance_keywords'] = sum(1 for word in ['bank', 'pay', 'card', 'loan', 'credit', 'finance'] if word in domain_clean)
        features['gov_keywords'] = sum(1 for word in ['gov', 'nic', 'india', 'bharat', 'sarkari'] if word in domain_clean)
        features['telecom_keywords'] = sum(1 for word in ['airtel', 'jio', 'bsnl', 'mobile', 'telecom'] if word in domain_clean)
        features['tech_keywords'] = sum(1 for word in ['tech', 'digital', 'online', 'web', 'net'] if word in domain_clean)
        
        # =================== LEXICAL FEATURES (41-51) ===================
        # Dictionary words
        common_words = ['the', 'and', 'com', 'org', 'net', 'www', 'http', 'mail', 'info', 'home']
        features['dictionary_word_count'] = sum(1 for word in common_words if word in domain_clean)
        
        # Repetition patterns
        features['char_repetition_score'] = 0
        if len(domain_clean) > 1:
            repetitions = sum(1 for j in range(len(domain_clean)-1) if domain_clean[j] == domain_clean[j+1])
            features['char_repetition_score'] = repetitions / len(domain_clean)
        
        # Numeric patterns
        features['has_year_pattern'] = 1 if re.search(r'(19|20)\d{2}', domain_clean) else 0
        features['starts_with_number'] = 1 if domain_clean and domain_clean[0].isdigit() else 0
        features['ends_with_number'] = 1 if domain_clean and domain_clean[-1].isdigit() else 0
        
        # Longest numeric sequence
        numeric_sequences = re.findall(r'\d+', domain_clean)
        features['max_numeric_sequence'] = max(len(seq) for seq in numeric_sequences) if numeric_sequences else 0
        
        # Subdomain patterns
        features['has_subdomain'] = 1 if features['domain_parts'] > 2 else 0
        features['subdomain_count'] = max(0, features['domain_parts'] - 2)
        
        # Suspicious patterns
        suspicious_patterns = ['verify', 'update', 'secure', 'alert', 'urgent', 'suspended', 'blocked']
        features['suspicious_pattern_count'] = sum(1 for pattern in suspicious_patterns if pattern in domain_clean)
        
        # Homograph detection
        homograph_chars = {'0': 'o', '1': 'l', '3': 'e', '5': 's', '6': 'g'}
        features['has_homograph'] = 0
        for digit, letter in homograph_chars.items():
            if digit in domain_clean and letter in domain_clean:
                features['has_homograph'] = 1
                break
        
        # Final complexity score
        features['complexity_score'] = (
            features['char_entropy'] * 0.3 +
            features['bigram_entropy'] * 0.3 +
            features['unique_char_ratio'] * 0.2 +
            features['transition_entropy'] * 0.2
        )
        
        return features


# ==================== HYBRID DETECTOR ====================

class HybridPhishingDetector:
    """
    Hybrid detector combining:
    - Rule-based CSE mapping (test.py)
    - ML model prediction (clean_real_data_model.py)
    - Evidence collection (test.py)
    """
    
    def __init__(self, application_id="AIGR-123456"):
        self.application_id = application_id
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Components
        self.cse_mapper = CSEMapper()
        self.feature_extractor = FeatureExtractor()
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        
        # Submission folders
        self.submission_folder = f"PS-02_{application_id}_Submission"
        self.evidence_folder = f"{self.submission_folder}/PS-02_{application_id}_Evidences"
        
        # CSE folders
        self.cse_folders = {
            'ICICI Bank': f"{self.evidence_folder}/ICICI",
            'HDFC Bank': f"{self.evidence_folder}/HDFC",
            'State Bank of India (SBI)': f"{self.evidence_folder}/SBI",
            'Punjab National Bank (PNB)': f"{self.evidence_folder}/PNB",
            'Bank of Baroda (BoB)': f"{self.evidence_folder}/BOB",
            'Airtel': f"{self.evidence_folder}/AIRTEL",
            'National Informatics Centre (NIC)': f"{self.evidence_folder}/NIC",
            'Registrar General and Census Commissioner of India (RGCCI)': f"{self.evidence_folder}/RGCCI",
            'Indian Railway Catering and Tourism Corporation (IRCTC)': f"{self.evidence_folder}/IRCTC",
            'Indian Oil Corporation Limited (IOCL)': f"{self.evidence_folder}/IOCL"
        }
        
        for folder in [self.submission_folder, self.evidence_folder]:
            os.makedirs(folder, exist_ok=True)
        
        for cse_folder in self.cse_folders.values():
            os.makedirs(cse_folder, exist_ok=True)
        
        # Browser
        self.playwright = None
        self.browser = None
        
        # Stats
        self.total_domains = 0
        self.cse_mapped = 0
        self.phishing_detected = 0
        self.suspected_detected = 0
        self.screenshots_captured = 0
        self.whois_collected = 0
        
        print(f"Hybrid Phishing Detector initialized")
        print(f"Device: {self.device}")
    
    def train_model(self, dataset_path=None, epochs=300):
        """Train ML model on combined dataset."""
        print("\n" + "="*80)
        print("TRAINING ML MODEL")
        print("="*80)
        
        if dataset_path is None:
            dataset_path = os.environ.get("DATASET_PATH")
        print(f"\n[1/4] Loading Training Dataset: {dataset_path}")
        # Guard: dataset must exist
        if not dataset_path or not os.path.exists(dataset_path):
            print(f"   [WARN] Dataset not found at: {dataset_path}. Skipping training.")
            return
        
        df = pd.read_csv(dataset_path)
        print(f"   Total samples: {len(df):,}")
        print(f"   Columns: {df.columns.tolist()}")
        
        # Label distribution
        label_counts = df['label'].value_counts()
        print(f"\n   Label distribution:")
        for label, count in label_counts.items():
            print(f"      {label}: {count:,} ({count/len(df)*100:.1f}%)")
        
        print(f"\n[2/4] Extracting Features...")
        
        # Extract features
        domains = df['url'].tolist()
        labels = df['label'].tolist()
        
        feature_data = []
        for i, domain in enumerate(domains):
            if i % 500 == 0:
                print(f"   Processing {i:,}/{len(domains):,}...", end='\r')
            features = self.feature_extractor.extract_features(domain)
            feature_data.append(features)
        
        print(f"   Extracted features from {len(domains):,} domains")
        
        # Convert to DataFrame
        X_df = pd.DataFrame(feature_data)
        self.feature_names = list(X_df.columns)
        X = X_df.values
        
        print(f"   Feature matrix shape: {X.shape}")
        print(f"   Total features: {len(self.feature_names)}")
        
        # Convert labels to binary (Phishing=1, Suspected=0)
        y = np.array([1 if label.lower() == 'phishing' else 0 for label in labels])
        
        print(f"\n[3/4] Training PyTorch Model...")
        print(f"   Epochs: {epochs}")
        print(f"   Architecture: 51 -> 128 -> 64 -> 32 -> 16 -> 2")
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create datasets
        train_dataset = PhishingDataset(X_train_scaled, y_train)
        test_dataset = PhishingDataset(X_test_scaled, y_test)
        
        # Class weights
        class_counts = np.bincount(y_train)
        class_weights = len(y_train) / (2.0 * class_counts)
        
        print(f"\n   Training samples: {len(X_train):,}")
        print(f"   Test samples: {len(X_test):,}")
        print(f"   Class distribution: Suspected: {np.sum(y_train == 0):,}, Phishing: {np.sum(y_train == 1):,}")
        print(f"   Class weights: [Suspected: {class_weights[0]:.3f}, Phishing: {class_weights[1]:.3f}]")
        
        # Weighted sampler
        sample_weights = class_weights[y_train]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        
        # Data loaders
        train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Create model
        self.model = PhishingNet(
            input_size=X_train_scaled.shape[1],
            hidden_sizes=[128, 64, 32, 16],
            dropout_rate=0.3
        ).to(self.device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss(
            weight=torch.FloatTensor(class_weights).to(self.device)
        )
        optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # Training loop
        best_test_acc = 0
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Testing
            self.model.eval()
            correct = 0
            total = 0
            test_loss = 0
            
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    test_loss += loss.item()
                    
                    _, predicted = torch.max(outputs, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()
            
            test_acc = correct / total
            avg_train_loss = train_loss / len(train_loader)
            
            scheduler.step(test_loss / len(test_loader))
            
            if test_acc > best_test_acc:
                best_test_acc = test_acc
            
            if epoch % 20 == 0 or epoch == epochs - 1:
                print(f"   Epoch {epoch:3d}/{epochs}: Loss: {avg_train_loss:.4f}, Test Acc: {test_acc:.4f}")
        
        print(f"\n   Training completed!")
        print(f"   Best test accuracy: {best_test_acc:.4f} ({best_test_acc*100:.1f}%)")
        
        # Final evaluation
        print(f"\n[4/4] Final Evaluation...")
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)
                probs = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.numpy())
                all_probs.extend(probs.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Metrics
        accuracy = accuracy_score(all_labels, all_preds)
        roc_auc = roc_auc_score(all_labels, all_probs[:, 1])
        
        print(f"\n   Final Test Results:")
        print(f"      Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
        print(f"      ROC AUC: {roc_auc:.4f}")
        
        print(f"\n   Classification Report:")
        print(classification_report(all_labels, all_preds, target_names=['Suspected', 'Phishing']))
        
        cm = confusion_matrix(all_labels, all_preds)
        print(f"\n   Confusion Matrix:")
        print("         Predicted")
        print("       Sus   Phi")
        print(f"Act Sus {cm[0,0]:4d}  {cm[0,1]:4d}")
        print(f"Act Phi {cm[1,0]:4d}  {cm[1,1]:4d}")
        
        # Save model
        self.save_model()
        
        print(f"\n" + "="*80)
        print("MODEL TRAINING COMPLETE!")
        print("="*80)
    
    def save_model(self, filepath='submission/hybrid_model.pkl'):
        """Save trained model."""
        print(f"\nSaving model to {filepath}...")
        
        model_data = {
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_type': 'hybrid_phishing_detector',
            'device': str(self.device),
            'timestamp': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, filepath)
        torch.save(self.model.state_dict(), 'submission/hybrid_weights.pth')
        
        print(f"   Model saved successfully")
    
    def load_model(self, filepath='submission/hybrid_model.pkl'):
        """Load trained model."""
        print(f"\nLoading model from {filepath}...")
        
        model_data = joblib.load(filepath)
        
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        
        # Create and load model
        self.model = PhishingNet(
            input_size=len(self.feature_names),
            hidden_sizes=[128, 64, 32, 16],
            dropout_rate=0.3
        ).to(self.device)
        
        self.model.load_state_dict(model_data['model_state_dict'])
        self.model.eval()
        
        print(f"   Model loaded successfully")
        print(f"   Features: {len(self.feature_names)}")
    
    def predict_phishing(self, domain):
        """
        Predict if domain is Phishing or Suspected using ML model.
        Returns: (classification, confidence)
        """
        if self.model is None:
            return 'Suspected', 0.5
        
        # Extract features
        features = self.feature_extractor.extract_features(domain)
        feature_vector = np.array([list(features.values())])
        
        # Scale
        feature_scaled = self.scaler.transform(feature_vector)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(feature_scaled).to(self.device)
            output = self.model(X_tensor)
            probs = F.softmax(output, dim=1)
            _, predicted = torch.max(output, 1)
        
        predicted_class = predicted.cpu().numpy()[0]
        confidence = probs.cpu().numpy()[0][predicted_class]
        
        classification = 'Phishing' if predicted_class == 1 else 'Suspected'
        
        return classification, float(confidence)
    
    def init_browser(self):
        """Initialize browser for screenshots."""
        if not PLAYWRIGHT_AVAILABLE:
            return False
        
        try:
            self.playwright = sync_playwright().start()
            self.browser = self.playwright.chromium.launch(headless=True)
            return True
        except Exception as e:
            print(f"   [WARN] Browser init failed: {e}")
            return False
    
    def capture_screenshot(self, domain, cse_name, serial_no):
        """Capture screenshot (from test.py)."""
        if not self.browser:
            return None
        
        try:
            cse_short = self._get_cse_short_name(cse_name)
            extracted = tldextract.extract(domain)
            subdomain = extracted.subdomain if extracted.subdomain else ''
            domain_name = extracted.domain
            tld = extracted.suffix
            
            # Build filename
            subdomain_parts = subdomain.split('.') if subdomain else []
            if len(subdomain_parts) > 2:
                subdomain_parts = subdomain_parts[-2:]
            subdomain_str = '.'.join(subdomain_parts) if subdomain_parts else domain_name
            
            filename = f"{cse_short}_{subdomain_str}.{domain_name}.{tld}_{serial_no}.pdf"
            filename = filename.replace('..', '.').replace(' ', '_')
            
            cse_folder = self.cse_folders.get(cse_name, self.evidence_folder)
            pdf_path = f"{cse_folder}/{filename}"
            
            # Try both protocols
            screenshot_bytes = None
            for protocol in ['https', 'http']:
                try:
                    url = f"{protocol}://{domain}"
                    page = self.browser.new_page(viewport={"width": 1280, "height": 800})
                    page.goto(url, wait_until="networkidle", timeout=20000)
                    screenshot_bytes = page.screenshot(full_page=True)
                    page.close()
                    break
                except:
                    try:
                        page.close()
                    except:
                        pass
                    continue
            
            if not screenshot_bytes or not PDF_AVAILABLE:
                return None
            
            # Convert to PDF
            img = Image.open(io.BytesIO(screenshot_bytes))
            img_width, img_height = img.size
            aspect = img_height / float(img_width)
            page_width, page_height = A4
            img_display_width = page_width - 40
            img_display_height = img_display_width * aspect
            
            c = canvas.Canvas(pdf_path, pagesize=A4)
            c.setFont("Helvetica-Bold", 14)
            c.drawString(20, page_height - 30, f"Phishing Evidence: {domain}")
            c.setFont("Helvetica", 10)
            c.drawString(20, page_height - 50, f"CSE: {cse_name}")
            c.drawString(20, page_height - 65, f"Captured: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            y_position = page_height - 85 - img_display_height
            if y_position < 0:
                img_display_height = page_height - 105
                y_position = 20
            
            c.drawImage(ImageReader(img), 20, y_position, 
                      width=img_display_width, height=img_display_height,
                      preserveAspectRatio=True)
            c.save()
            
            self.screenshots_captured += 1
            return filename
            
        except Exception as e:
            return None
    
    def collect_whois(self, domain):
        """Collect WHOIS (from test.py)."""
        whois_data = {}
        
        if not WHOIS_AVAILABLE:
            return whois_data
        
        try:
            result = [None]
            
            def whois_lookup():
                try:
                    import logging
                    logging.getLogger('whois.whois').setLevel(logging.CRITICAL)
                    result[0] = python_whois.whois(domain)
                except:
                    pass
            
            thread = threading.Thread(target=whois_lookup)
            thread.daemon = True
            thread.start()
            thread.join(timeout=5)
            
            if result[0]:
                w = result[0]
                if hasattr(w, 'creation_date'):
                    whois_data['registration_date'] = w.creation_date[0] if isinstance(w.creation_date, list) else w.creation_date
                if hasattr(w, 'registrar'):
                    whois_data['registrar'] = w.registrar[0] if isinstance(w.registrar, list) else w.registrar
                if hasattr(w, 'name'):
                    whois_data['registrant'] = w.name[0] if isinstance(w.name, list) else w.name
                if hasattr(w, 'country'):
                    whois_data['country'] = w.country[0] if isinstance(w.country, list) else w.country
                if hasattr(w, 'name_servers'):
                    whois_data['name_servers'] = ', '.join(w.name_servers) if isinstance(w.name_servers, list) else str(w.name_servers)
                
                self.whois_collected += 1
        except:
            pass
        
        return whois_data
    
    def test_on_shortlisting(self, shortlisting_dir=None):
        """
        Run hybrid detection on shortlisting dataset.
        1. Map domains to CSE (rule-based)
        2. Predict Phishing/Suspected (ML model)
        3. Collect evidence (screenshots + WHOIS)
        """
        
        print("\n" + "="*80)
        print("HYBRID PHISHING DETECTION ON SHORTLISTING DATA")
        print("="*80)
        
        if shortlisting_dir is None:
            shortlisting_dir = os.environ.get("SHORTLIST_DIR")
        print("\n[1/5] Loading Shortlisting Dataset...")
        
        files = list(Path(shortlisting_dir).glob("*.xlsx"))
        if not files:
            files = list(Path(shortlisting_dir).glob("*.csv"))
        
        print(f"   Found {len(files)} file(s)")
        if not shortlisting_dir or not files:
            print("   [WARN] No shortlisting files found. Skipping testing.")
            return
        
        all_domains = []
        for file in files:
            print(f"   Loading: {file.name}...")
            if file.suffix == '.xlsx':
                df = pd.read_excel(file)
            else:
                df = pd.read_csv(file)
            
            domain_col = df.columns[0]
            domains = df[domain_col].tolist()
            all_domains.extend(domains)
            print(f"   [OK] Loaded: {file.name} ({len(domains):,} domains)")
        
        self.total_domains = len(all_domains)
        print(f"\n   Total domains: {self.total_domains:,}")
        
        print("\n[2/5] Running Hybrid Detection (CSE Mapping + ML Prediction)...")
        print(f"   Step 1: Rule-based CSE mapping")
        print(f"   Step 2: ML model prediction (Phishing/Suspected)")
        
        detected_domains = []
        start_time = time.time()
        
        for idx, domain in enumerate(all_domains):
            try:
                # Step 1: CSE mapping (rule-based from test.py)
                cse_name, cse_score = self.cse_mapper.map_domain_to_cse(domain)
                
                if cse_name:
                    self.cse_mapped += 1
                    
                    # Step 2: ML prediction
                    classification, confidence = self.predict_phishing(domain)
                    
                    detected_domains.append({
                        'domain': domain,
                        'classification': classification,
                        'confidence': confidence,
                        'cse_name': cse_name,
                        'cse_score': cse_score
                    })
                    
                    if classification == 'Phishing':
                        self.phishing_detected += 1
                    else:
                        self.suspected_detected += 1
            except:
                continue
            
            if (idx + 1) % 10000 == 0:
                elapsed = time.time() - start_time
                rate = (idx + 1) / elapsed
                remaining = (self.total_domains - idx - 1) / rate if rate > 0 else 0
                
                print(f"\r   [{idx+1:,}/{self.total_domains:,}] "
                      f"CSE: {self.cse_mapped:,} | "
                      f"Phishing: {self.phishing_detected:,} | "
                      f"Suspected: {self.suspected_detected:,} | "
                      f"Rate: {rate:.0f}/s | "
                      f"ETA: {remaining/60:.0f}m", end='', flush=True)
        
        print(f"\n\n   [OK] Detection Complete!")
        if self.total_domains > 0:
            print(f"   CSE-mapped domains: {self.cse_mapped:,} ({self.cse_mapped/self.total_domains*100:.2f}%)")
        else:
            print("   CSE-mapped domains: 0 (0.00%)")
        print(f"   Phishing: {self.phishing_detected:,}")
        print(f"   Suspected: {self.suspected_detected:,}")
        
        # Save all detections
        detected_df = pd.DataFrame(detected_domains)
        detected_df.to_csv(f"{self.submission_folder}/all_detected.csv", index=False)
        print(f"   [OK] Saved: {self.submission_folder}/all_detected.csv")
        
        # CSE distribution
        cse_counts = {}
        for item in detected_domains:
            cse = item['cse_name']
            cse_counts[cse] = cse_counts.get(cse, 0) + 1
        
        print(f"\n   CSE Distribution:")
        for cse, count in sorted(cse_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"      {cse}: {count:,}")
        
        print("\n[3/5] Initializing Browser for Screenshots...")
        
        if self.init_browser():
            print(f"   [OK] Browser ready")
        else:
            print(f"   [WARN] Browser not available")
        
        print("\n[4/5] Capturing Screenshots + WHOIS (Phishing only)...")
        
        # Process only Phishing domains for evidence
        phishing_only = [item for item in detected_domains if item['classification'] == 'Phishing']
        
        print(f"   [INFO] Processing {len(phishing_only)} Phishing domains")
        
        results = []
        serial_no = 1
        start_time = time.time()
        
        for idx, item in enumerate(phishing_only):
            domain = item['domain']
            cse_name = item['cse_name']
            confidence = item['confidence']
            classification = item['classification']
            
            if (idx + 1) % 10 == 0:
                elapsed = time.time() - start_time
                rate = (idx + 1) / elapsed if elapsed > 0 else 0
                remaining = (len(phishing_only) - idx - 1) / rate if rate > 0 else 0
                
                print(f"\r   [{idx+1}/{len(phishing_only):,}] "
                      f"📸 {self.screenshots_captured} | "
                      f"📋 {self.whois_collected} | "
                      f"⏱ {remaining/60:.0f}m", end='', flush=True)
            
            # Collect WHOIS
            whois_data = self.collect_whois(domain)
            
            # Capture screenshot
            evidence_file = self.capture_screenshot(domain, cse_name, serial_no)
            
            # Add to results
            results.append({
                'Application_ID': self.application_id,
                'Source_of_detection': 'Hybrid Detection (Rule-based CSE + ML Model)',
                'Identified_Domain': domain,
                'Corresponding_CSE_Domain': self._get_official_domain(cse_name),
                'CSE_Name': cse_name,
                'Classification': classification,
                'Registration_Date': str(whois_data.get('registration_date', '')),
                'Registrar_Name': whois_data.get('registrar', ''),
                'Registrant_Name': whois_data.get('registrant', ''),
                'Registrant_Country': whois_data.get('country', ''),
                'Name_Servers': whois_data.get('name_servers', ''),
                'Hosting_IP': '',
                'Hosting_ISP': '',
                'Hosting_Country': '',
                'DNS_Records': '',
                'Evidence_File': evidence_file or '',
                'Date_of_Detection': datetime.now().strftime('%d-%m-%Y'),
                'Time_of_Detection': datetime.now().strftime('%H-%M-%S'),
                'Date_of_Post': '',
                'Remarks': f'ML Confidence: {confidence:.2%}, CSE Score: {item["cse_score"]}'
            })
            
            serial_no += 1
        
        # Add Suspected domains (without evidence)
        suspected_only = [item for item in detected_domains if item['classification'] == 'Suspected']
        print(f"\n   [INFO] Adding {len(suspected_only)} Suspected domains to submission...")
        
        for item in suspected_only:
            results.append({
                'Application_ID': self.application_id,
                'Source_of_detection': 'Hybrid Detection (Rule-based CSE + ML Model)',
                'Identified_Domain': item['domain'],
                'Corresponding_CSE_Domain': self._get_official_domain(item['cse_name']),
                'CSE_Name': item['cse_name'],
                'Classification': item['classification'],
                'Registration_Date': '',
                'Registrar_Name': '',
                'Registrant_Name': '',
                'Registrant_Country': '',
                'Name_Servers': '',
                'Hosting_IP': '',
                'Hosting_ISP': '',
                'Hosting_Country': '',
                'DNS_Records': '',
                'Evidence_File': '',
                'Date_of_Detection': datetime.now().strftime('%d-%m-%Y'),
                'Time_of_Detection': datetime.now().strftime('%H-%M-%S'),
                'Date_of_Post': '',
                'Remarks': f'ML Confidence: {item["confidence"]:.2%}, CSE Score: {item["cse_score"]}'
            })
        
        print(f"\n\n[5/5] Generating Submission...")
        print(f"   [OK] Total domains: {self.total_domains:,}")
        print(f"   [OK] CSE-mapped: {self.cse_mapped:,} ({self.cse_mapped/self.total_domains*100:.2f}%)")
        print(f"   [OK] Phishing: {self.phishing_detected:,}")
        print(f"   [OK] Suspected: {self.suspected_detected:,}")
        print(f"   [OK] Screenshots: {self.screenshots_captured:,}")
        print(f"   [OK] WHOIS: {self.whois_collected:,}")
        
        # Save submission
        self._save_submission(results)
        
        # Cleanup
        if self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.stop()
        
        print(f"\n" + "="*80)
        print("TESTING COMPLETE!")
        print("="*80)
        print(f"   Submission: {self.submission_folder}/")
        print("="*80)
    
    def _get_cse_short_name(self, cse_name):
        """Get short CSE name."""
        mapping = {
            'ICICI Bank': 'ICICI',
            'HDFC Bank': 'HDFC',
            'State Bank of India (SBI)': 'SBI',
            'Punjab National Bank (PNB)': 'PNB',
            'Bank of Baroda (BoB)': 'BOB',
            'Airtel': 'AIRTEL',
            'National Informatics Centre (NIC)': 'NIC',
            'Registrar General and Census Commissioner of India (RGCCI)': 'RGCCI',
            'Indian Railway Catering and Tourism Corporation (IRCTC)': 'IRCTC',
            'Indian Oil Corporation Limited (IOCL)': 'IOCL'
        }
        return mapping.get(cse_name, 'UNKNOWN')
    
    def _get_official_domain(self, cse_name):
        """Get official CSE domain."""
        official = {
            'State Bank of India (SBI)': 'sbi.co.in',
            'ICICI Bank': 'icicibank.com',
            'HDFC Bank': 'hdfcbank.com',
            'Punjab National Bank (PNB)': 'pnbindia.in',
            'Bank of Baroda (BoB)': 'bankofbaroda.in',
            'National Informatics Centre (NIC)': 'nic.gov.in',
            'Registrar General and Census Commissioner of India (RGCCI)': 'dc.crsorgi.gov.in',
            'Indian Railway Catering and Tourism Corporation (IRCTC)': 'irctc.co.in',
            'Airtel': 'airtel.in',
            'Indian Oil Corporation Limited (IOCL)': 'iocl.com'
        }
        return official.get(cse_name, '')
    
    def _save_submission(self, results):
        """Save submission files."""
        output_excel = f"{self.submission_folder}/PS-02_{self.application_id}_Submission_Set.xlsx"
        
        df = pd.DataFrame(results)
        df.to_excel(output_excel, index=False, sheet_name='Phishing_Domains')
        
        csv_path = output_excel.replace('.xlsx', '.csv')
        df.to_csv(csv_path, index=False)
        
        print(f"   [OK] Excel: {output_excel}")
        print(f"   [OK] CSV: {csv_path}")


def main():
    """Main execution."""
    
    print("="*80)
    print("HYBRID PHISHING DETECTION SYSTEM")
    print("="*80)
    print("Combines:")
    print("  1. Rule-based CSE mapping (from test.py)")
    print("  2. ML model prediction (from clean_real_data_model.py)")
    print("  3. Evidence collection (screenshots + WHOIS)")
    print("="*80)
    
    detector = HybridPhishingDetector(application_id="AIGR-123456")
    
    # Step 1: Train model on combined dataset
    print("\n" + "="*80)
    print("STEP 1: TRAIN ML MODEL")
    print("="*80)
    detector.train_model(
        dataset_path=None,
        epochs=300
    )
    
    # Step 2: Test on shortlisting data
    print("\n" + "="*80)
    print("STEP 2: TEST ON SHORTLISTING DATA")
    print("="*80)
    detector.test_on_shortlisting()
    
    print("\n" + "="*80)
    print("ALL COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
