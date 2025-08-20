#!/usr/bin/env python3
"""
Setup and Validation Script for Phishing Detection System
=========================================================

This script validates the installation and runs a complete pipeline test.
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists and report status."""
    if os.path.exists(filepath):
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description} not found: {filepath}")
        return False

def check_python_packages():
    """Check if required Python packages are installed."""
    required_packages = [
        'pandas', 'numpy', 'sklearn', 'requests', 'bs4',
        'tldextract', 'whois', 'flask', 'matplotlib', 'seaborn'
    ]
    
    print("\n📦 Checking Python packages...")
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    return len(missing_packages) == 0, missing_packages

def validate_datasets():
    """Validate that the required datasets exist."""
    print("\n📊 Checking datasets...")
    
    datasets = [
        ("/home/vk/phishing/phishing/PS02_Training_set/PS02_Training_set/PS02_Training_set.xlsx", "Training dataset"),
        ("/home/vk/phishing/phishing/PS-02 Phishing Detection CSE_Domains_Dataset_for_Stage_1.xlsx", "Domains dataset")
    ]
    
    all_exist = True
    for filepath, description in datasets:
        exists = check_file_exists(filepath, description)
        all_exist = all_exist and exists
    
    return all_exist

def run_feature_extraction_test():
    """Run a quick feature extraction test."""
    print("\n🧪 Running feature extraction test...")
    
    try:
        from phishing_feature_extractor import PhishingFeatureExtractor
        
        extractor = PhishingFeatureExtractor()
        features = extractor.extract_all_features("example.com")
        
        if len(features) > 30:  # Should have 50+ features
            print(f"✅ Feature extraction successful: {len(features)} features extracted")
            return True
        else:
            print(f"⚠️ Feature extraction may be incomplete: only {len(features)} features")
            return False
            
    except Exception as e:
        print(f"❌ Feature extraction failed: {e}")
        return False

def check_project_structure():
    """Check if all project files are in place."""
    print("\n📁 Checking project structure...")
    
    required_files = [
        "phishing_feature_extractor.py",
        "train_model.py", 
        "web_app.py",
        "demo.py",
        "requirements.txt",
        "README.md",
        "templates/index.html"
    ]
    
    all_exist = True
    for filepath in required_files:
        exists = check_file_exists(filepath, f"Project file")
        all_exist = all_exist and exists
    
    return all_exist

def run_training_pipeline():
    """Run the complete training pipeline."""
    print("\n🚀 Running complete training pipeline...")
    
    # Step 1: Feature extraction
    print("\n1️⃣ Starting feature extraction...")
    try:
        result = subprocess.run([
            sys.executable, "phishing_feature_extractor.py", "--batch"
        ], capture_output=True, text=True, timeout=300)  # 5 minute timeout
        
        if result.returncode == 0:
            print("✅ Feature extraction completed")
        else:
            print(f"❌ Feature extraction failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⚠️ Feature extraction timed out (this is normal for large datasets)")
        # Continue anyway as partial extraction might be useful
    except Exception as e:
        print(f"❌ Feature extraction error: {e}")
        return False
    
    # Step 2: Model training
    print("\n2️⃣ Starting model training...")
    try:
        result = subprocess.run([
            sys.executable, "train_model.py"
        ], capture_output=True, text=True, timeout=600)  # 10 minute timeout
        
        if result.returncode == 0:
            print("✅ Model training completed")
            return True
        else:
            print(f"❌ Model training failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⚠️ Model training timed out")
        return False
    except Exception as e:
        print(f"❌ Model training error: {e}")
        return False

def main():
    """Main setup and validation function."""
    print("🛡️ Phishing Detection System - Setup & Validation")
    print("=" * 60)
    
    # Change to project directory
    project_dir = "/home/vk/phishing/phishing_detection_project"
    os.chdir(project_dir)
    print(f"📂 Working directory: {project_dir}")
    
    # Check project structure
    structure_ok = check_project_structure()
    
    # Check Python packages
    packages_ok, missing = check_python_packages()
    
    # Check datasets
    datasets_ok = validate_datasets()
    
    # Run feature extraction test
    extraction_ok = run_feature_extraction_test()
    
    print(f"\n📋 Setup Validation Summary:")
    print(f"{'✅' if structure_ok else '❌'} Project structure")
    print(f"{'✅' if packages_ok else '❌'} Python packages")
    print(f"{'✅' if datasets_ok else '❌'} Required datasets")
    print(f"{'✅' if extraction_ok else '❌'} Feature extraction")
    
    if not packages_ok:
        print(f"\n📦 Missing packages: {', '.join(missing)}")
        print("💡 Install with: pip install " + " ".join(missing))
    
    if not datasets_ok:
        print(f"\n📊 Please ensure the training datasets are available in the correct location")
    
    # Ask user if they want to run the full pipeline
    if all([structure_ok, packages_ok, datasets_ok, extraction_ok]):
        print(f"\n✅ All checks passed! System is ready.")
        
        response = input("\n🚀 Would you like to run the complete training pipeline? (y/N): ").strip().lower()
        
        if response in ['y', 'yes']:
            print("\n🔄 Starting complete pipeline...")
            print("⚠️ This may take 10-30 minutes depending on your system and network speed.")
            
            pipeline_ok = run_training_pipeline()
            
            if pipeline_ok:
                print(f"\n🎉 Setup completed successfully!")
                print(f"\n🎯 Next steps:")
                print(f"1. Test single URL: python demo.py <url>")
                print(f"2. Start web interface: python web_app.py")
                print(f"3. Analyze custom URLs: python phishing_feature_extractor.py --url <url>")
            else:
                print(f"\n⚠️ Pipeline had issues. You can still use individual components.")
        else:
            print(f"\n💡 Manual steps to complete setup:")
            print(f"1. Extract features: python phishing_feature_extractor.py --batch")
            print(f"2. Train models: python train_model.py")
            print(f"3. Test system: python demo.py")
    else:
        print(f"\n⚠️ Some issues found. Please resolve them before proceeding.")
    
    print(f"\n📖 For detailed instructions, see README.md")

if __name__ == "__main__":
    main()
