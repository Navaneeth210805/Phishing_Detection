#!/usr/bin/env python3
"""
Prepare training and testing data paths and train model without majority classes.
"""

import subprocess
import sys
from pathlib import Path

def main():
    # Get all training and test chunks
    train_chunks = sorted(Path(".").glob("dom_unsup_features/train_domfeat_chunk_*.npz"))
    train_chunks = [str(c) for c in train_chunks if not str(c).endswith("_y11.npz")]
    
    test_chunks = sorted(Path(".").glob("dom_unsup_features/test_domfeat_chunk_*.npz"))
    test_chunks = [str(c) for c in test_chunks if not str(c).endswith("_y11.npz")]
    
    print(f"Found {len(train_chunks)} training chunks")
    print(f"Found {len(test_chunks)} test chunks")
    
    if not train_chunks or not test_chunks:
        print("ERROR: Could not find training or test chunks")
        sys.exit(1)
    
    # Build command
    cmd = [
        "python3",
        "train_without_majority.py",
        "--train-chunks", *train_chunks,
        "--test-chunks", *test_chunks,
        "--mapping-path", "dom_stage1_mapping.json",
        "--model-path", "dom_mc_no_majority_rf.pkl",
        "--algorithm", "random_forest",
    ]
    
    print("\n" + "=" * 80)
    print("Running training without majority classes (Facebook, Meta, USPS)...")
    print("=" * 80 + "\n")
    
    result = subprocess.run(cmd)
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()
