#!/usr/bin/env python3
import os
import os.path as osp
from main import HybridPhishingDetector


def main():
    print("="*80)
    print("TEST-ONLY: HYBRID PHISHING DETECTION")
    print("="*80)

    detector = HybridPhishingDetector(application_id="AIGR-123456")

    # Try to load model if available
    model_path = 'submission/hybrid_model.pkl'
    if osp.exists(model_path):
        try:
            detector.load_model(filepath=model_path)
        except Exception as e:
            print(f"[WARN] Could not load model from {model_path}: {e}")
    else:
        print(f"[WARN] No trained model found at {model_path}. Predictions will default to 'Suspected'.")

    print("\n" + "="*80)
    print("STEP: TEST ON SHORTLISTING DATA")
    print("="*80)
    detector.test_on_shortlisting(shortlisting_dir=None)

    print("\n" + "="*80)
    print("TESTING COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
