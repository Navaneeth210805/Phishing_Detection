#!/usr/bin/env python3
"""
Improved Dataset Analysis and Model Testing Script
=================================================

This script properly handles feature extraction and model testing for NCIIPC datasets.
"""

import pandas as pd
import numpy as np
import os
import glob
import joblib
import json
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
            
            # Add source file column
            df['source_file'] = os.path.basename(file)
            combined_data.append(df)
            
        except Exception as e:
            print(f"   ❌ Error reading {file}: {e}")
    
    if combined_data:
        combined_df = pd.concat(combined_data, ignore_index=True)
        print(f"\n✅ Combined dataset shape: {combined_df.shape}")
        return combined_df
    else:
        print("❌ No datasets could be loaded!")
        return None

def prepare_features_for_model(features_df, model_info):
    """Prepare features to match the model's expected format."""
    print("🔧 Preparing features for model...")
    
    expected_features = model_info['feature_names']
    print(f"Model expects {len(expected_features)} features")
    
    # Create a DataFrame with the expected features
    model_features = pd.DataFrame()
    
    for feature_name in expected_features:
        if feature_name in features_df.columns:
            model_features[feature_name] = features_df[feature_name]
        else:
            # Try to find the feature with different naming
            found = False
            for col in features_df.columns:
                if feature_name.replace('url_', '').replace('domain_', '').replace('content_', '') in col:
                    model_features[feature_name] = features_df[col]
                    found = True
                    break
            
            if not found:
                # Set default value for missing features
                print(f"   ⚠️  Missing feature '{feature_name}', setting to 0")
                model_features[feature_name] = 0
    
    print(f"✅ Prepared {len(model_features.columns)} features for model")
    return model_features

def extract_features_for_sample(df, domain_col, sample_size=50):
    """Extract features for a sample of the dataset."""
    print(f"\n🔧 Extracting features for {sample_size} samples...")
    
    # Take a stratified sample if possible
    try:
        label_col = None
        for col in df.columns:
            if 'class' in col.lower() or 'label' in col.lower() or 'phishing' in col.lower():
                label_col = col
                break
        
        if label_col:
            # Get balanced sample
            sample_df = df.groupby(label_col).apply(
                lambda x: x.sample(n=min(sample_size//2, len(x)), random_state=42)
            ).reset_index(drop=True)
        else:
            sample_df = df.sample(n=min(sample_size, len(df)), random_state=42)
    except:
        sample_df = df.sample(n=min(sample_size, len(df)), random_state=42)
    
    print(f"Sample size: {len(sample_df)}")
    
    extractor = PhishingFeatureExtractor()
    extracted_features = []
    
    for idx, (i, row) in enumerate(sample_df.iterrows()):
        domain = str(row[domain_col]).strip()
        if not domain or domain.lower() in ['nan', 'none']:
            continue
            
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
        return features_df, sample_df
    else:
        print("❌ No features could be extracted!")
        return None, None

def test_model(features_df, sample_df, label_col):
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
    
    # Load model info
    try:
        with open('model_info.json', 'r') as f:
            model_info = json.load(f)
        print(f"📋 Model expects {model_info['feature_count']} features")
        print(f"📋 Label mapping: {model_info['label_mapping']}")
    except Exception as e:
        print(f"⚠️ Could not load model info: {e}")
        return
    
    # Prepare features for the model
    model_features = prepare_features_for_model(features_df, model_info)
    
    # Handle missing values
    model_features = model_features.fillna(0)
    
    # Convert boolean columns to numeric
    for col in model_features.columns:
        if model_features[col].dtype == bool:
            model_features[col] = model_features[col].astype(int)
        elif model_features[col].dtype == 'object':
            try:
                model_features[col] = pd.to_numeric(model_features[col], errors='coerce').fillna(0)
            except:
                model_features[col] = 0
    
    print(f"Final feature matrix shape: {model_features.shape}")
    
    # Get true labels
    true_labels = []
    for idx in features_df['original_index']:
        label = sample_df.loc[sample_df.index == idx, label_col].iloc[0]
        true_labels.append(label)
    
    print(f"\nTrue label distribution:")
    label_counts = pd.Series(true_labels).value_counts()
    print(label_counts)
    
    # Make predictions
    try:
        # Apply scaling
        scaler = StandardScaler()
        X_test_scaled = scaler.fit_transform(model_features)
        
        predictions = model.predict(X_test_scaled)
        prediction_proba = model.predict_proba(X_test_scaled)
        
        print("✅ Predictions made successfully")
        
        # Convert predictions to labels
        if hasattr(model, 'classes_'):
            pred_labels = [model_info['label_mapping'][str(p)] for p in predictions]
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
        
        # Show sample predictions with confidence
        print(f"\n🔍 Sample Predictions:")
        results_df = pd.DataFrame({
            'domain': features_df['domain'].values,
            'true_label': true_labels,
            'predicted_label': pred_labels,
            'confidence': np.max(prediction_proba, axis=1)
        })
        
        print(results_df.head(15))
        
        # Show misclassified examples
        misclassified = results_df[results_df['true_label'] != results_df['predicted_label']]
        if len(misclassified) > 0:
            print(f"\n❌ Misclassified examples ({len(misclassified)}):")
            print(misclassified)
        else:
            print("\n✅ Perfect classification on this sample!")
        
        # Performance by label
        print(f"\n📈 Performance by Label:")
        for label in set(true_labels):
            mask = np.array(true_labels) == label
            if mask.any():
                label_acc = accuracy_score(np.array(true_labels)[mask], np.array(pred_labels)[mask])
                print(f"  {label}: {label_acc:.4f} ({label_acc*100:.2f}%)")
        
        return results_df
        
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_dataset_statistics(df):
    """Provide detailed dataset statistics."""
    print("\n" + "="*60)
    print("📊 DATASET STATISTICS")
    print("="*60)
    
    print(f"Total records: {len(df):,}")
    print(f"Total unique domains: {df.iloc[:, 1].nunique():,}")
    
    # Label analysis
    label_col = None
    for col in df.columns:
        if 'class' in col.lower() or 'label' in col.lower() or 'phishing' in col.lower():
            label_col = col
            break
    
    if label_col:
        print(f"\nLabel Distribution:")
        label_counts = df[label_col].value_counts()
        print(label_counts)
        print(f"\nLabel Percentages:")
        print((label_counts / len(df) * 100).round(2))
    
    # CSE analysis
    cse_col = None
    for col in df.columns:
        if 'cse' in col.lower() or 'entity' in col.lower():
            cse_col = col
            break
    
    if cse_col:
        print(f"\nTop 10 CSEs by frequency:")
        cse_counts = df[cse_col].value_counts().head(10)
        print(cse_counts)
    
    # Source file analysis
    if 'source_file' in df.columns:
        print(f"\nRecords per source file:")
        file_counts = df['source_file'].value_counts()
        print(file_counts)
    
    return df, label_col, cse_col

def main():
    """Main function to run the analysis."""
    print("🚀 NCIIPC Dataset Analysis and Model Testing")
    print("=" * 60)
    
    # Step 1: Combine datasets
    combined_df = combine_datasets()
    if combined_df is None:
        return
    
    # Step 2: Analyze dataset statistics
    analyzed_df, label_col, cse_col = analyze_dataset_statistics(combined_df)
    
    # Find domain column
    domain_col = None
    for col in analyzed_df.columns:
        if 'domain' in col.lower() or 'url' in col.lower():
            domain_col = col
            break
    
    if not domain_col or not label_col:
        print("❌ Could not identify domain and label columns!")
        print(f"Available columns: {list(analyzed_df.columns)}")
        return
    
    print(f"\n✅ Using columns:")
    print(f"  Domain: {domain_col}")
    print(f"  Label: {label_col}")
    if cse_col:
        print(f"  CSE: {cse_col}")
    
    # Step 3: Extract features for a sample
    features_df, sample_df = extract_features_for_sample(analyzed_df, domain_col, sample_size=100)
    if features_df is None:
        return
    
    # Step 4: Test model
    results = test_model(features_df, sample_df, label_col)
    
    print("\n🎉 Analysis complete!")
    
    # Save results
    if results is not None:
        results.to_csv('model_test_results.csv', index=False)
        print("💾 Results saved to 'model_test_results.csv'")

if __name__ == "__main__":
    main()
