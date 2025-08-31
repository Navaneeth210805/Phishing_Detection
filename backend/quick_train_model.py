#!/usr/bin/env python3
"""
Quick Model Training Script
===========================

This script generates sample data and trains a basic phishing detection model
to get the system up and running quickly.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import json
from datetime import datetime, timedelta
import os
import sys

def generate_sample_training_data(n_samples=1000):
    """Generate sample training data for phishing detection."""
    np.random.seed(42)
    
    # Generate features similar to what the feature extractor produces
    features = []
    labels = []
    
    for i in range(n_samples):
        # Legitimate domains (60% of data)
        if i < n_samples * 0.6:
            feature_vector = {
                'url_length': np.random.normal(25, 8),  # Shorter URLs
                'subdomain_count': np.random.poisson(1),  # Fewer subdomains
                'suspicious_chars': np.random.poisson(0.5),  # Very few suspicious chars
                'domain_age': np.random.normal(2000, 500),  # Older domains
                'ssl_certificate': 1,  # Usually have SSL
                'whois_privacy': 0,  # Usually not private
                'redirect_count': 0,  # Usually no redirects
                'external_links': np.random.normal(10, 5),  # Moderate external links
                'form_count': np.random.poisson(1),  # Few forms
                'has_login_form': np.random.choice([0, 1], p=[0.7, 0.3]),  # Sometimes login
                'similarity_score': np.random.normal(0.1, 0.05)  # Low similarity to known sites
            }
            label = 0  # Legitimate
            
        # Suspected phishing (25% of data)
        elif i < n_samples * 0.85:
            feature_vector = {
                'url_length': np.random.normal(40, 10),  # Medium length URLs
                'subdomain_count': np.random.poisson(2),  # More subdomains
                'suspicious_chars': np.random.poisson(2),  # Some suspicious chars
                'domain_age': np.random.normal(100, 200),  # Newer domains
                'ssl_certificate': np.random.choice([0, 1], p=[0.4, 0.6]),  # Sometimes SSL
                'whois_privacy': np.random.choice([0, 1], p=[0.3, 0.7]),  # Often private
                'redirect_count': np.random.poisson(1),  # Some redirects
                'external_links': np.random.normal(20, 8),  # More external links
                'form_count': np.random.poisson(2),  # More forms
                'has_login_form': np.random.choice([0, 1], p=[0.3, 0.7]),  # Often login
                'similarity_score': np.random.normal(0.4, 0.1)  # Medium similarity
            }
            label = 1  # Suspected
            
        # Definite phishing (15% of data)
        else:
            feature_vector = {
                'url_length': np.random.normal(60, 15),  # Long URLs
                'subdomain_count': np.random.poisson(4),  # Many subdomains
                'suspicious_chars': np.random.poisson(5),  # Many suspicious chars
                'domain_age': np.random.normal(30, 50),  # Very new domains
                'ssl_certificate': np.random.choice([0, 1], p=[0.7, 0.3]),  # Often no SSL
                'whois_privacy': 1,  # Usually private
                'redirect_count': np.random.poisson(3),  # Many redirects
                'external_links': np.random.normal(40, 15),  # Many external links
                'form_count': np.random.poisson(3),  # Many forms
                'has_login_form': 1,  # Always has login
                'similarity_score': np.random.normal(0.8, 0.1)  # High similarity
            }
            label = 2  # Phishing
        
        # Ensure non-negative values
        for key in feature_vector:
            if key != 'ssl_certificate' and key != 'whois_privacy' and key != 'has_login_form':
                feature_vector[key] = max(0, feature_vector[key])
        
        features.append(feature_vector)
        labels.append(label)
    
    # Convert to DataFrame
    df = pd.DataFrame(features)
    df['label'] = labels
    
    return df

def train_model():
    """Train and save the phishing detection model."""
    print("🔄 Generating sample training data...")
    df = generate_sample_training_data(1000)
    
    # Prepare features and labels
    feature_columns = [col for col in df.columns if col != 'label']
    X = df[feature_columns]
    y = df['label']
    
    print(f"📊 Training data shape: {X.shape}")
    print(f"📊 Label distribution: {y.value_counts().to_dict()}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Label encoder for consistent prediction labels
    label_encoder = LabelEncoder()
    label_encoder.fit(['Legitimate', 'Suspected', 'Phishing'])
    
    # Train Random Forest model
    print("🤖 Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate the model
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    y_pred = model.predict(X_test_scaled)
    
    print(f"✅ Training accuracy: {train_score:.3f}")
    print(f"✅ Testing accuracy: {test_score:.3f}")
    print("\n📈 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Suspected', 'Phishing']))
    
    # Package model data in expected format
    model_data = {
        'model': model,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'model_name': 'RandomForestClassifier',
        'feature_names': feature_columns,
        'training_date': datetime.now().isoformat(),
        'test_accuracy': float(test_score)
    }
    
    # Save the model
    model_path = 'phishing_detection_model.pkl'
    joblib.dump(model_data, model_path)
    print(f"💾 Model saved to: {model_path}")
    
    # Save feature info
    feature_info = {
        'feature_names': feature_columns,
        'feature_count': len(feature_columns),
        'label_mapping': {0: 'Legitimate', 1: 'Suspected', 2: 'Phishing'},
        'training_date': datetime.now().isoformat(),
        'model_type': 'RandomForestClassifier',
        'training_samples': len(df),
        'test_accuracy': float(test_score)
    }
    
    with open('model_info.json', 'w') as f:
        json.dump(feature_info, f, indent=2)
    
    print("✅ Model training completed successfully!")
    return model_data, feature_info

if __name__ == "__main__":
    try:
        model, info = train_model()
        print(f"\n🎯 Model ready for use!")
        print(f"📁 Files created:")
        print(f"   - phishing_detection_model.pkl")
        print(f"   - model_info.json")
        
    except Exception as e:
        print(f"❌ Error training model: {e}")
        sys.exit(1)
