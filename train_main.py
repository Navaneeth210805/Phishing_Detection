#!/usr/bin/env python3
import os
from main import HybridPhishingDetector


def main():
    print("="*80)
    print("TRAIN-ONLY: HYBRID PHISHING DETECTION")
    print("="*80)

    detector = HybridPhishingDetector(application_id="AIGR-123456")

    print("\n" + "="*80)
    print("STEP: TRAIN ML MODEL")
    print("="*80)
    detector.train_model(dataset_path=None, epochs=300)

    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
