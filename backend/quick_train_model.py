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
    """Generate sample training data for phishing detection using real feature names."""
    np.random.seed(42)
    
    # Generate features matching what the feature extractor produces
    features = []
    labels = []
    
    for i in range(n_samples):
        # Legitimate domains (60% of data)
        if i < n_samples * 0.6:
            feature_vector = {
                # URL features (with url_ prefix)
                'url_length': np.random.normal(25, 8),
                'url_domain_length': np.random.normal(12, 4),
                'url_path_length': np.random.normal(5, 3),
                'url_query_length': np.random.normal(0, 2),
                'url_has_ip_address': 0,
                'url_has_suspicious_tld': 0,
                'url_subdomain_count': np.random.poisson(1),
                'url_dash_count': np.random.poisson(0.5),
                'url_dot_count': np.random.poisson(2),
                'url_slash_count': np.random.poisson(2),
                'url_at_symbol': 0,
                'url_double_slash_redirecting': 0,
                'url_has_suspicious_keywords': 0,
                'url_has_url_shortening': 0,
                'url_has_suspicious_port': 0,
                'url_digit_ratio': np.random.normal(0.1, 0.05),
                'url_special_char_ratio': np.random.normal(0.1, 0.05),
                'url_entropy': np.random.normal(3.5, 0.5),
                'url_uses_https': np.random.choice([0, 1], p=[0.3, 0.7]),
                
                # Domain features (with domain_ prefix)
                'domain_age_days': np.random.normal(2000, 500),
                'domain_is_new_domain': 0,
                'domain_days_to_expire': np.random.normal(365, 100),
                'domain_expires_soon': 0,
                'domain_has_registrar': 1,
                'domain_has_registrant': 1,
                'domain_a_record_count': np.random.poisson(3),
                'domain_dns_resolves': 1,
                'domain_has_ssl': 1,
                'domain_ssl_days_to_expire': np.random.normal(90, 30),
                'domain_ssl_expires_soon': 0,
                
                # Content features (with content_ prefix)
                'content_status_code': 200,
                'content_length': np.random.normal(50000, 20000),
                'content_has_title': 1,
                'content_title_length': np.random.normal(50, 20),
                'content_form_count': np.random.poisson(1),
                'content_has_password_field': np.random.choice([0, 1], p=[0.7, 0.3]),
                'content_has_hidden_fields': np.random.choice([0, 1], p=[0.5, 0.5]),
                'content_total_links': np.random.normal(50, 20),
                'content_external_links': np.random.normal(10, 5),
                'content_image_count': np.random.normal(20, 10),
                'content_external_images': np.random.normal(2, 2),
                'content_script_count': np.random.normal(5, 3),
                'content_has_external_scripts': np.random.choice([0, 1], p=[0.3, 0.7]),
                'content_has_suspicious_content': 0,
                'content_has_copyright': 1,
                'content_redirect_count': 0,
                'content_has_sitemap': np.random.choice([0, 1], p=[0.5, 0.5]),
                'content_sitemap_url_count': np.random.poisson(2),
                'content_sitemap_structure_complexity': np.random.choice([0, 1], p=[0.7, 0.3]),
                'content_has_sitemap_index': np.random.choice([0, 1], p=[0.8, 0.2]),
                'content_sitemap_last_modified': np.random.choice([0, 1], p=[0.6, 0.4])
            }
            label = 0  # Legitimate
            
        # Suspected phishing (25% of data)
        elif i < n_samples * 0.85:
            feature_vector = {
                # URL features
                'url_length': np.random.normal(40, 10),
                'url_domain_length': np.random.normal(20, 8),
                'url_path_length': np.random.normal(15, 8),
                'url_query_length': np.random.normal(5, 5),
                'url_has_ip_address': np.random.choice([0, 1], p=[0.8, 0.2]),
                'url_has_suspicious_tld': np.random.choice([0, 1], p=[0.7, 0.3]),
                'url_subdomain_count': np.random.poisson(2),
                'url_dash_count': np.random.poisson(2),
                'url_dot_count': np.random.poisson(4),
                'url_slash_count': np.random.poisson(4),
                'url_at_symbol': np.random.choice([0, 1], p=[0.9, 0.1]),
                'url_double_slash_redirecting': np.random.choice([0, 1], p=[0.8, 0.2]),
                'url_has_suspicious_keywords': np.random.choice([0, 1], p=[0.6, 0.4]),
                'url_has_url_shortening': np.random.choice([0, 1], p=[0.8, 0.2]),
                'url_has_suspicious_port': np.random.choice([0, 1], p=[0.9, 0.1]),
                'url_digit_ratio': np.random.normal(0.2, 0.1),
                'url_special_char_ratio': np.random.normal(0.2, 0.1),
                'url_entropy': np.random.normal(4.0, 0.5),
                'url_uses_https': np.random.choice([0, 1], p=[0.4, 0.6]),
                
                # Domain features
                'domain_age_days': np.random.normal(100, 200),
                'domain_is_new_domain': np.random.choice([0, 1], p=[0.3, 0.7]),
                'domain_days_to_expire': np.random.normal(180, 100),
                'domain_expires_soon': np.random.choice([0, 1], p=[0.7, 0.3]),
                'domain_has_registrar': np.random.choice([0, 1], p=[0.2, 0.8]),
                'domain_has_registrant': np.random.choice([0, 1], p=[0.3, 0.7]),
                'domain_a_record_count': np.random.poisson(2),
                'domain_dns_resolves': 1,
                'domain_has_ssl': np.random.choice([0, 1], p=[0.4, 0.6]),
                'domain_ssl_days_to_expire': np.random.normal(60, 30),
                'domain_ssl_expires_soon': np.random.choice([0, 1], p=[0.6, 0.4]),
                
                # Content features
                'content_status_code': np.random.choice([200, 302, 404], p=[0.7, 0.2, 0.1]),
                'content_length': np.random.normal(30000, 15000),
                'content_has_title': 1,
                'content_title_length': np.random.normal(40, 15),
                'content_form_count': np.random.poisson(2),
                'content_has_password_field': np.random.choice([0, 1], p=[0.3, 0.7]),
                'content_has_hidden_fields': np.random.choice([0, 1], p=[0.2, 0.8]),
                'content_total_links': np.random.normal(30, 15),
                'content_external_links': np.random.normal(20, 8),
                'content_image_count': np.random.normal(15, 8),
                'content_external_images': np.random.normal(5, 3),
                'content_script_count': np.random.normal(8, 4),
                'content_has_external_scripts': np.random.choice([0, 1], p=[0.2, 0.8]),
                'content_has_suspicious_content': np.random.choice([0, 1], p=[0.6, 0.4]),
                'content_has_copyright': np.random.choice([0, 1], p=[0.5, 0.5]),
                'content_redirect_count': np.random.poisson(1),
                'content_has_sitemap': np.random.choice([0, 1], p=[0.7, 0.3]),
                'content_sitemap_url_count': np.random.poisson(1),
                'content_sitemap_structure_complexity': np.random.choice([0, 1], p=[0.8, 0.2]),
                'content_has_sitemap_index': np.random.choice([0, 1], p=[0.9, 0.1]),
                'content_sitemap_last_modified': np.random.choice([0, 1], p=[0.7, 0.3])
            }
            label = 1  # Suspected
            
        # Definite phishing (15% of data)
        else:
            feature_vector = {
                # URL features
                'url_length': np.random.normal(60, 15),
                'url_domain_length': np.random.normal(30, 10),
                'url_path_length': np.random.normal(25, 10),
                'url_query_length': np.random.normal(10, 8),
                'url_has_ip_address': np.random.choice([0, 1], p=[0.5, 0.5]),
                'url_has_suspicious_tld': np.random.choice([0, 1], p=[0.4, 0.6]),
                'url_subdomain_count': np.random.poisson(4),
                'url_dash_count': np.random.poisson(5),
                'url_dot_count': np.random.poisson(6),
                'url_slash_count': np.random.poisson(6),
                'url_at_symbol': np.random.choice([0, 1], p=[0.7, 0.3]),
                'url_double_slash_redirecting': np.random.choice([0, 1], p=[0.6, 0.4]),
                'url_has_suspicious_keywords': np.random.choice([0, 1], p=[0.2, 0.8]),
                'url_has_url_shortening': np.random.choice([0, 1], p=[0.6, 0.4]),
                'url_has_suspicious_port': np.random.choice([0, 1], p=[0.8, 0.2]),
                'url_digit_ratio': np.random.normal(0.3, 0.1),
                'url_special_char_ratio': np.random.normal(0.3, 0.1),
                'url_entropy': np.random.normal(4.5, 0.5),
                'url_uses_https': np.random.choice([0, 1], p=[0.7, 0.3]),
                
                # Domain features
                'domain_age_days': np.random.normal(30, 50),
                'domain_is_new_domain': 1,
                'domain_days_to_expire': np.random.normal(90, 50),
                'domain_expires_soon': np.random.choice([0, 1], p=[0.4, 0.6]),
                'domain_has_registrar': np.random.choice([0, 1], p=[0.3, 0.7]),
                'domain_has_registrant': np.random.choice([0, 1], p=[0.5, 0.5]),
                'domain_a_record_count': np.random.poisson(1),
                'domain_dns_resolves': 1,
                'domain_has_ssl': np.random.choice([0, 1], p=[0.7, 0.3]),
                'domain_ssl_days_to_expire': np.random.normal(30, 20),
                'domain_ssl_expires_soon': np.random.choice([0, 1], p=[0.3, 0.7]),
                
                # Content features
                'content_status_code': np.random.choice([200, 302, 403, 404], p=[0.5, 0.3, 0.1, 0.1]),
                'content_length': np.random.normal(20000, 10000),
                'content_has_title': np.random.choice([0, 1], p=[0.2, 0.8]),
                'content_title_length': np.random.normal(30, 10),
                'content_form_count': np.random.poisson(3),
                'content_has_password_field': 1,
                'content_has_hidden_fields': 1,
                'content_total_links': np.random.normal(20, 10),
                'content_external_links': np.random.normal(40, 15),
                'content_image_count': np.random.normal(10, 5),
                'content_external_images': np.random.normal(8, 4),
                'content_script_count': np.random.normal(12, 6),
                'content_has_external_scripts': 1,
                'content_has_suspicious_content': np.random.choice([0, 1], p=[0.2, 0.8]),
                'content_has_copyright': np.random.choice([0, 1], p=[0.8, 0.2]),
                'content_redirect_count': np.random.poisson(3),
                'content_has_sitemap': np.random.choice([0, 1], p=[0.9, 0.1]),
                'content_sitemap_url_count': 0,
                'content_sitemap_structure_complexity': 0,
                'content_has_sitemap_index': 0,
                'content_sitemap_last_modified': 0
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
