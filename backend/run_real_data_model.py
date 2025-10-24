#!/usr/bin/env python3
"""
Run Real Data PyTorch Phishing Detection Model
==============================================

Simple script to run the real data model without Unicode issues
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_pytorch_model_real_data import RealDataPhishingDetector

def main():
    """Main execution function."""
    print("Real Data PyTorch Phishing Detection - NCIIPC Dataset")
    print("=" * 60)
    
    try:
        detector = RealDataPhishingDetector()
        results = detector.run_complete_training()
        
        print("\n" + "=" * 60)
        print("FINAL RESULTS SUMMARY:")
        print(f"Test Accuracy: {results['accuracy']:.1%}")
        print(f"ROC AUC: {results['roc_auc']:.3f}")
        print(f"Precision: {results['precision']:.3f}")
        print(f"Recall: {results['recall']:.3f}")
        print(f"F1-Score: {results['f1_score']:.3f}")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error running model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
