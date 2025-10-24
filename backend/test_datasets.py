#!/usr/bin/env python3
"""
Dataset Analysis and Model Testing Script
==========================================

This script combines all the NCIIPC datasets and tests the phishing model.
"""

import pandas as pd
import numpy as np
import os
import glob
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Import our feature extractor
from phishing_feature_extractor import PhishingFeatureExtractor

def combine_datasets():
    """Combine all dataset files into one DataFrame."""
    print("🔍 Searching for dataset files...")
    
    dataset_dir = "dataset/Mock data along with Ground Truth for NCIIPC AI Grand Challenge - PS02/"
    pattern = os.path.join(dataset_dir, "*.xlsx")
    files = glob.glob(pattern)
    
    print(f"📁 Found {len(files)} dataset files:")
    for file in files:
        print(f"  - {os.path.basename(file)}")
    
    combined_data = []
    
    for file in files:
        print(f"\n📖 Reading {os.path.basename(file)}...")
        try:
            df = pd.read_excel(file)
            print(f"   Shape: {df.shape}")
            print(f"   Columns: {df.columns.tolist()}")
            
            # Add source file column
            df['source_file'] = os.path.basename(file)
            combined_data.append(df)
            
        except Exception as e:
            print(f"   ❌ Error reading {file}: {e}")
    
    if combined_data:
        combined_df = pd.concat(combined_data, ignore_index=True)
        print(f"\n✅ Combined dataset shape: {combined_df.shape}")
        print(f"📊 Columns in combined dataset:")
        for i, col in enumerate(combined_df.columns):
            print(f"   {i+1}. {col}")
        
        return combined_df
    else:
        print("❌ No datasets could be loaded!")
        return None

def analyze_dataset(df):
    """Analyze the combined dataset."""
    print("\n" + "="*60)
    print("📊 DATASET ANALYSIS")
    print("="*60)
    
    # Basic info
    print(f"Total records: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    
    # Check for the label column
    label_columns = [col for col in df.columns if 'class' in col.lower() or 'label' in col.lower() or 'phishing' in col.lower()]
    print(f"\nPossible label columns: {label_columns}")
    
    if label_columns:
        label_col = label_columns[0]
        print(f"\nUsing '{label_col}' as label column")
        print("Label distribution:")
        print(df[label_col].value_counts())
        print("\nLabel distribution (%):")
        print(df[label_col].value_counts(normalize=True) * 100)
    
    # Check for domain column
    domain_columns = [col for col in df.columns if 'domain' in col.lower() or 'url' in col.lower()]
    print(f"\nPossible domain columns: {domain_columns}")
    
    if domain_columns:
        domain_col = domain_columns[0]
        print(f"\nUsing '{domain_col}' as domain column")
        print(f"Sample domains:")
        print(df[domain_col].head(10).tolist())
    
    # CSE analysis
    cse_columns = [col for col in df.columns if 'cse' in col.lower() or 'entity' in col.lower()]
    print(f"\nPossible CSE columns: {cse_columns}")
    
    if cse_columns:
        cse_col = cse_columns[0]
        print(f"\nUsing '{cse_col}' as CSE column")
        print("CSE distribution:")
        print(df[cse_col].value_counts())
    
    return df, label_col if label_columns else None, domain_col if domain_columns else None

def extract_features_for_dataset(df, domain_col, sample_size=100):
    """Extract features for a sample of the dataset."""
    print(f"\n🔧 Extracting features for {sample_size} samples...")
    
    # Take a sample for feature extraction (can be time-consuming)
    sample_df = df.sample(n=min(sample_size, len(df)), random_state=42)
    
    extractor = PhishingFeatureExtractor()
    extracted_features = []
    
    for idx, (i, row) in enumerate(sample_df.iterrows()):
        domain = row[domain_col]
        print(f"  {idx+1}/{len(sample_df)}: {domain}")
        
        try:
            features = extractor.extract_all_features(domain)
            features['original_index'] = i
            features['domain'] = domain
            extracted_features.append(features)
        except Exception as e:
            print(f"    ❌ Error extracting features for {domain}: {e}")
    
    if extracted_features:
        features_df = pd.DataFrame(extracted_features)
        print(f"✅ Extracted features for {len(features_df)} domains")
        print(f"Feature count: {len(features_df.columns)}")
        return features_df
    else:
        print("❌ No features could be extracted!")
        return None

def test_model(features_df, original_df, label_col):
    """Test the phishing detection model."""
    print("\n" + "="*60)
    print("🧪 MODEL TESTING")
    print("="*60)
    
    # Load the model
    try:
        model = joblib.load('phishing_detection_model.pkl')
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Load model info if available
    try:
        import json
        with open('model_info.json', 'r') as f:
            model_info = json.load(f)
        print(f"📋 Model info: {model_info}")
    except Exception as e:
        print(f"⚠️ Could not load model info: {e}")
        model_info = {}
    
    # Prepare features for prediction
    # Remove non-feature columns
    feature_cols = [col for col in features_df.columns if col not in ['original_index', 'domain', 'url']]
    X_test = features_df[feature_cols].copy()
    
    print(f"Features for prediction: {len(feature_cols)}")
    print(f"Sample shape: {X_test.shape}")
    
    # Handle missing values
    X_test = X_test.fillna(0)
    
    # Convert boolean columns to numeric
    for col in X_test.columns:
        if X_test[col].dtype == bool:
            X_test[col] = X_test[col].astype(int)
    
    # Get true labels
    true_labels = []
    for idx in features_df['original_index']:
        true_labels.append(original_df.loc[idx, label_col])
    
    print(f"\nTrue label distribution:")
    print(pd.Series(true_labels).value_counts())
    
    # Make predictions
    try:
        # Check if model expects scaled features
        if hasattr(model, 'named_steps') and 'scaler' in model.named_steps:
            print("🔧 Model includes scaler")
            predictions = model.predict(X_test)
            prediction_proba = model.predict_proba(X_test)
        else:
            print("🔧 Applying standard scaling")
            scaler = StandardScaler()
            X_test_scaled = scaler.fit_transform(X_test)
            predictions = model.predict(X_test_scaled)
            prediction_proba = model.predict_proba(X_test_scaled)
        
        print("✅ Predictions made successfully")
        
        # Convert predictions to labels if needed
        if hasattr(model, 'classes_'):
            pred_labels = model.classes_[predictions]
        else:
            pred_labels = predictions
        
        print(f"\nPredicted label distribution:")
        print(pd.Series(pred_labels).value_counts())
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, pred_labels)
        print(f"\n📊 RESULTS:")
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        print(f"\n📋 Detailed Classification Report:")
        print(classification_report(true_labels, pred_labels))
        
        print(f"\n🔍 Confusion Matrix:")
        cm = confusion_matrix(true_labels, pred_labels)
        print(cm)
        
        # Show some sample predictions
        print(f"\n🔍 Sample Predictions:")
        results_df = pd.DataFrame({
            'domain': features_df['domain'].values,
            'true_label': true_labels,
            'predicted_label': pred_labels,
            'confidence': np.max(prediction_proba, axis=1)
        })
        
        print(results_df.head(10))
        
        # Show misclassified examples
        misclassified = results_df[results_df['true_label'] != results_df['predicted_label']]
        if len(misclassified) > 0:
            print(f"\n❌ Misclassified examples ({len(misclassified)}):")
            print(misclassified.head(10))
        else:
            print("\n✅ Perfect classification on this sample!")
        
        return results_df
        
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        print(f"X_test shape: {X_test.shape}")
        print(f"X_test columns: {X_test.columns.tolist()}")
        return None

def main():
    """Main function to run the analysis."""
    print("🚀 NCIIPC Dataset Analysis and Model Testing")
    print("=" * 60)
    
    # Step 1: Combine datasets
    combined_df = combine_datasets()
    if combined_df is None:
        return
    
    # Step 2: Analyze dataset
    analyzed_df, label_col, domain_col = analyze_dataset(combined_df)
    
    if not domain_col or not label_col:
        print("❌ Could not identify domain and label columns!")
        return
    
    # Step 3: Extract features for a sample
    features_df = extract_features_for_dataset(analyzed_df, domain_col, sample_size=50)
    if features_df is None:
        return
    
    # Step 4: Test model
    results = test_model(features_df, analyzed_df, label_col)
    
    print("\n🎉 Analysis complete!")

if __name__ == "__main__":
    main()
