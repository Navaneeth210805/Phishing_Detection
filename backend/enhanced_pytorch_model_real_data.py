#!/usr/bin/env python3
"""
Enhanced PyTorch Phishing Detection Model - Real NCIIPC Dataset
================================================================

USING REAL NCIIPC DATASET:
1. Loads actual phishing/suspected domains from Excel files
2. Handles both URLs and social media links
3. 51-feature extraction from real data
4. PyTorch neural network with proper regularization  
5. 80-20 train-test split with realistic evaluation

Performance Target: Realistic accuracy based on real data
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import os
import glob
import json
import joblib
import warnings
import re
from collections import Counter
import tldextract
import random
from datetime import datetime
import urllib.parse
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

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
    """Neural Network for phishing detection with proper regularization."""
    
    def __init__(self, input_size=51, hidden_sizes=[128, 64, 32], dropout_rate=0.3):
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
        
        # Output layer
        self.output_layer = nn.Linear(hidden_sizes[-1], 2)  # Binary classification
        
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
        
        # Output layer
        x = self.output_layer(x)
        return x

class RealDataPhishingDetector:
    """Enhanced phishing detector using real NCIIPC dataset."""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.cse_data = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load CSE whitelist data
        self.load_cse_whitelist()
    
    def load_cse_whitelist(self):
        """Load CSE whitelist data for legitimate domain patterns."""
        try:
            with open('cse_whitelist.json', 'r') as f:
                self.cse_data = json.load(f)
            
            print("CSE Whitelist loaded:")
            for cse_name, data in self.cse_data.items():
                print(f"   • {cse_name}: {len(data['whitelisted_domains'])} domains, {data['sector']} sector")
            
            # Create comprehensive legitimate domain patterns
            self.legitimate_domains = []
            self.legitimate_keywords = []
            self.sectors = set()
            
            for cse_name, data in self.cse_data.items():
                self.legitimate_domains.extend(data['whitelisted_domains'])
                self.legitimate_keywords.extend(data['keywords'])
                self.sectors.add(data['sector'])
            
            print(f"   📊 Total: {len(self.legitimate_domains)} legitimate domains, {len(self.legitimate_keywords)} keywords")
            
        except Exception as e:
            print(f"⚠️ Could not load CSE whitelist: {e}")
            self.cse_data = {}
            self.legitimate_domains = []
            self.legitimate_keywords = []
            self.sectors = set()
    
    def load_nciipc_dataset(self, dataset_path="dataset/Mock data along with Ground Truth for NCIIPC AI Grand Challenge - PS02"):
        """Load all NCIIPC dataset files."""
        print(f"\n📂 Loading NCIIPC Dataset from: {dataset_path}")
        
        all_data = []
        excel_files = glob.glob(os.path.join(dataset_path, "*.xlsx"))
        
        print(f"   Found {len(excel_files)} Excel files")
        
        for file_path in excel_files:
            try:
                print(f"   Loading: {os.path.basename(file_path)}")
                df = pd.read_excel(file_path)
                
                # Standardize column names
                df.columns = df.columns.str.strip()
                
                # Extract relevant columns
                if 'Identified Phishing/Suspected Domain Name' in df.columns:
                    domain_col = 'Identified Phishing/Suspected Domain Name'
                elif 'Domain' in df.columns:
                    domain_col = 'Domain'
                else:
                    domain_col = df.columns[1]  # Usually second column
                
                if 'Phishing/Suspected Domains (i.e. Class Label)' in df.columns:
                    label_col = 'Phishing/Suspected Domains (i.e. Class Label)'
                elif 'Label' in df.columns:
                    label_col = 'Label'
                else:
                    label_col = df.columns[-1]  # Usually last column
                
                # Clean and filter data
                df_clean = df[[domain_col, label_col]].dropna()
                df_clean.columns = ['domain', 'label']
                
                # Clean domains
                df_clean['domain'] = df_clean['domain'].astype(str).str.strip()
                df_clean = df_clean[df_clean['domain'] != '']
                
                # Clean labels
                df_clean['label'] = df_clean['label'].astype(str).str.strip()
                
                print(f"      Rows: {len(df_clean)}")
                print(f"      Labels: {df_clean['label'].value_counts().to_dict()}")
                
                all_data.append(df_clean)
                
            except Exception as e:
                print(f"   ⚠️ Error loading {file_path}: {e}")
                continue
        
        if not all_data:
            raise ValueError("No data loaded from NCIIPC dataset!")
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        print(f"\n📊 Combined Dataset Statistics:")
        print(f"   Total samples: {len(combined_df):,}")
        print(f"   Label distribution:")
        label_counts = combined_df['label'].value_counts()
        for label, count in label_counts.items():
            print(f"      {label}: {count:,} ({count/len(combined_df)*100:.1f}%)")
        
        return combined_df
    
    def clean_and_preprocess_domains(self, df):
        """Clean and preprocess domains, handling URLs and social media links."""
        print(f"\n🧹 Cleaning and Preprocessing Domains...")
        
        cleaned_domains = []
        cleaned_labels = []
        domain_types = []
        
        for idx, row in df.iterrows():
            domain = str(row['domain']).strip()
            label = str(row['label']).strip()
            
            # Skip empty domains
            if not domain or domain.lower() in ['nan', 'none', '']:
                continue
            
            # Determine domain type
            domain_type = self.classify_domain_type(domain)
            domain_types.append(domain_type)
            
            # Extract clean domain
            clean_domain = self.extract_clean_domain(domain)
            
            if clean_domain:
                cleaned_domains.append(clean_domain)
                cleaned_labels.append(label)
        
        print(f"   ✅ Cleaned {len(cleaned_domains):,} domains")
        print(f"   Domain types:")
        type_counts = Counter(domain_types)
        for domain_type, count in type_counts.items():
            print(f"      {domain_type}: {count:,}")
        
        return cleaned_domains, cleaned_labels
    
    def classify_domain_type(self, domain):
        """Classify the type of domain/URL."""
        domain_lower = domain.lower()
        
        # Social media patterns
        social_patterns = ['@', 'facebook.com', 'twitter.com', 'instagram.com', 'linkedin.com', 
                          'youtube.com', 'tiktok.com', 'snapchat.com', 'whatsapp.com']
        
        if any(pattern in domain_lower for pattern in social_patterns):
            return 'social_media'
        
        # Email patterns
        if '@' in domain and '.' in domain:
            return 'email'
        
        # URL patterns
        if domain_lower.startswith(('http://', 'https://', 'www.')):
            return 'url'
        
        # Domain patterns
        if '.' in domain and not '@' in domain:
            return 'domain'
        
        return 'other'
    
    def extract_clean_domain(self, domain):
        """Extract clean domain from various formats."""
        domain_lower = domain.lower().strip()
        
        # Handle email addresses
        if '@' in domain_lower:
            # Extract domain from email
            email_parts = domain_lower.split('@')
            if len(email_parts) > 1:
                return email_parts[1].split('/')[0].split('?')[0]
            return None
        
        # Handle URLs
        if domain_lower.startswith(('http://', 'https://')):
            try:
                parsed = urllib.parse.urlparse(domain_lower)
                return parsed.netloc.split(':')[0]  # Remove port
            except:
                return None
        
        # Handle www prefix
        if domain_lower.startswith('www.'):
            domain_lower = domain_lower[4:]
        
        # Remove paths and parameters
        domain_lower = domain_lower.split('/')[0].split('?')[0].split('#')[0]
        
        # Basic validation
        if '.' in domain_lower and len(domain_lower) > 3:
            return domain_lower
        
        return None
    
    def extract_51_features(self, domains):
        """Extract exactly 51 comprehensive features from domains."""
        print(f"\n⚙️ Extracting 51 Comprehensive Features from Real Data...")
        
        feature_data = []
        
        for i, domain in enumerate(domains):
            if i % 1000 == 0:
                print(f"   Processing {i:,}/{len(domains):,} domains...")
            
            features = {}
            domain_clean = str(domain).lower().strip()
            
            # =================== BASIC FEATURES (1-15) ===================
            features['domain_length'] = len(domain_clean)
            features['dot_count'] = domain_clean.count('.')
            features['dash_count'] = domain_clean.count('-')
            features['underscore_count'] = domain_clean.count('_')
            features['digit_count'] = sum(c.isdigit() for c in domain_clean)
            features['uppercase_count'] = sum(c.isupper() for c in domain)
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
            features['has_consecutive_chars'] = 1 if any(domain_clean[j] == domain_clean[j+1] for j in range(len(domain_clean)-1)) else 0
            
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
            
            # Part entropy and statistics
            if len(parts) > 0:
                part_lengths = [len(part) for part in parts]
                features['part_length_variance'] = np.var(part_lengths)
                features['part_length_std'] = np.std(part_lengths)
            else:
                features['part_length_variance'] = 0
                features['part_length_std'] = 0
            
            # N-gram entropy
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
            
            # Character transition entropy
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
            
            feature_data.append(features)
        
        # Convert to DataFrame
        features_df = pd.DataFrame(feature_data)
        self.feature_names = list(features_df.columns)
        
        print(f"✅ Extracted {len(self.feature_names)} comprehensive features")
        if len(self.feature_names) != 51:
            print(f"⚠️ Expected 51 features, got {len(self.feature_names)}")
        
        return features_df
    
    def create_pytorch_model(self, input_size=51):
        """Create PyTorch neural network model."""
        print(f"\n🧠 Creating PyTorch Neural Network (Input: {input_size} features)...")
        
        model = PhishingNet(
            input_size=input_size,
            hidden_sizes=[128, 64, 32],
            dropout_rate=0.3
        ).to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        
        print(f"✅ Model Architecture:")
        print(f"   Input → 128 → 64 → 32 → 2 (Binary Output)")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Dropout rate: 30%")
        print(f"   Device: {self.device}")
        
        return model
    
    def train_pytorch_model(self, X, y, epochs=100, batch_size=32):
        """Train PyTorch model with proper validation."""
        print(f"\n🏋️ Training PyTorch Model...")
        
        # Convert labels to binary (Phishing=1, Suspected=0)
        y_binary = [1 if 'phishing' in str(label).lower() else 0 for label in y]
        y_binary = np.array(y_binary)
        
        # Train-test split (80-20)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
        )
        
        # Further split training for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create datasets
        train_dataset = PhishingDataset(X_train_scaled, y_train)
        val_dataset = PhishingDataset(X_val_scaled, y_val)
        
        # Calculate class weights
        class_counts = np.bincount(y_train)
        class_weights = len(y_train) / (2.0 * class_counts)
        
        # Create weighted sampler
        sample_weights = class_weights[y_train]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        
        # Data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Create model
        self.model = self.create_pytorch_model(input_size=X_train_scaled.shape[1])
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(self.device))
        optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        print(f"📊 Training Configuration:")
        print(f"   Training samples: {len(X_train):,}")
        print(f"   Validation samples: {len(X_val):,}")
        print(f"   Test samples: {len(X_test):,}")
        print(f"   Class distribution: Suspected: {np.sum(y_train == 0):,}, Phishing: {np.sum(y_train == 1):,}")
        print(f"   Class weights: [Suspected: {class_weights[0]:.3f}, Phishing: {class_weights[1]:.3f}]")
        
        # Training loop
        best_val_acc = 0
        train_losses = []
        val_accuracies = []
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            val_correct = 0
            val_total = 0
            val_loss = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs, 1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()
            
            val_acc = val_correct / val_total
            avg_train_loss = train_loss / len(train_loader)
            
            train_losses.append(avg_train_loss)
            val_accuracies.append(val_acc)
            
            scheduler.step(val_loss / len(val_loader))
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
            
            if epoch % 20 == 0 or epoch == epochs - 1:
                print(f"   Epoch {epoch:3d}/{epochs}: Train Loss: {avg_train_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        print(f"✅ Training completed. Best validation accuracy: {best_val_acc:.4f}")
        
        # Store test data for final evaluation
        self.X_test_scaled = X_test_scaled
        self.y_test = y_test
        
        return {'best_val_accuracy': best_val_acc}
    
    def evaluate_model(self):
        """Comprehensive model evaluation on test set."""
        print(f"\n📈 Final Model Evaluation (80-20 Split)...")
        
        if not hasattr(self, 'X_test_scaled') or not hasattr(self, 'y_test'):
            print("❌ No test data available for evaluation")
            return None
        
        # Predictions
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(self.X_test_scaled).to(self.device)
            outputs = self.model(X_tensor)
            probabilities = F.softmax(outputs, dim=1)
            _, predictions = torch.max(outputs, 1)
        
        predictions = predictions.cpu().numpy()
        probabilities = probabilities.cpu().numpy()
        
        # Metrics
        accuracy = accuracy_score(self.y_test, predictions)
        roc_auc = roc_auc_score(self.y_test, probabilities[:, 1])
        
        print(f"🎯 Final Test Results:")
        print(f"   Test Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
        print(f"   ROC AUC: {roc_auc:.4f}")
        print(f"   Test Samples: {len(self.y_test):,}")
        
        # Classification report
        print(f"\n📋 Detailed Classification Report:")
        class_names = ['Suspected', 'Phishing']
        print(classification_report(self.y_test, predictions, target_names=class_names))
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, predictions)
        print(f"\n🔍 Confusion Matrix:")
        print("         Predicted")
        print("       Sus   Phi")
        print(f"Act Sus {cm[0,0]:4d}  {cm[0,1]:4d}")
        print(f"Act Phi {cm[1,0]:4d}  {cm[1,1]:4d}")
        
        # Calculate additional metrics
        precision = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
        recall = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\n📊 Additional Metrics:")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        print(f"   F1-Score: {f1_score:.4f}")
        
        return {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'confusion_matrix': cm
        }
    
    def run_complete_training(self):
        """Run the complete training pipeline with real NCIIPC data."""
        print("🚀 REAL DATA PYTORCH PHISHING DETECTION SYSTEM")
        print("=" * 80)
        print("🎯 USING REAL NCIIPC DATASET:")
        print("  ✅ Real phishing/suspected domains from Excel files")
        print("  ✅ Handles URLs, emails, and social media links")
        print("  ✅ 51 advanced features from real data")
        print("  ✅ PyTorch neural network with regularization")
        print("  ✅ 80-20 train-test split with proper evaluation")
        print("=" * 80)
        
        # Load real NCIIPC dataset
        df = self.load_nciipc_dataset()
        
        # Clean and preprocess domains
        domains, labels = self.clean_and_preprocess_domains(df)
        
        # Extract 51 features
        X = self.extract_51_features(domains)
        
        print(f"\n📊 Final Dataset Statistics:")
        print(f"   Total samples: {len(X):,}")
        print(f"   Features: {X.shape[1]}")
        print(f"   Label distribution:")
        label_counts = Counter(labels)
        for label, count in label_counts.items():
            print(f"      {label}: {count:,} ({count/len(labels)*100:.1f}%)")
        
        # Train model with 80-20 split
        training_results = self.train_pytorch_model(X, labels, epochs=100)
        
        # Evaluate model
        evaluation_results = self.evaluate_model()
        
        # Save model
        self.save_model()
        
        print(f"\n🎉 TRAINING COMPLETE!")
        print(f"   Final Test Accuracy: {evaluation_results['accuracy']:.1%}")
        print(f"   ROC AUC: {evaluation_results['roc_auc']:.3f}")
        print(f"   Precision: {evaluation_results['precision']:.3f}")
        print(f"   Recall: {evaluation_results['recall']:.3f}")
        print(f"   F1-Score: {evaluation_results['f1_score']:.3f}")
        print(f"   Model Type: Real Data PyTorch with 51 features")
        
        return evaluation_results
    
    def save_model(self, filepath='real_data_pytorch_model.pkl'):
        """Save the complete model."""
        print(f"\n💾 Saving Real Data PyTorch Model...")
        
        model_data = {
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_type': 'real_data_pytorch_51_features',
            'device': str(self.device),
            'timestamp': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, filepath)
        torch.save(self.model.state_dict(), 'real_data_pytorch_weights.pth')
        
        print(f"Model saved successfully")

def main():
    """Main execution function."""
    print("Real Data PyTorch Phishing Detection - NCIIPC Dataset")
    print("=" * 60)
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    detector = RealDataPhishingDetector()
    results = detector.run_complete_training()

if __name__ == "__main__":
    main()
