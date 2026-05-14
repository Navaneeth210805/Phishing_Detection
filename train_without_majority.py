#!/usr/bin/env python3
"""
Train DOM multiclass detector excluding majority classes (Facebook, Meta, USPS)
and evaluate on full test set to see how model performs.
"""

import argparse
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from dom_target_multiclass_detector import HTMLTargetMulticlassDOMClassifier

EXCLUDED_CLASSES = {"facebook", "meta", "usps"}


def load_balanced_mapping(mapping_path: str) -> Dict[str, object]:
    """Load the class mapping from baseline."""
    with open(mapping_path, 'r') as f:
        return json.load(f)


def create_filtered_classifier(
    mapping: Dict[str, object],
    hashed_dim: int = 4096,
    n_estimators: int = 300,
    algorithm: str = "random_forest",
    exclude_classes: set = None,
) -> HTMLTargetMulticlassDOMClassifier:
    """Create classifier excluding certain classes."""
    exclude_classes = exclude_classes or set()
    
    # Build filtered class list
    all_targets = mapping.get("top_targets", [])
    filtered_targets = [t for t in all_targets if t.lower() not in exclude_classes]
    
    # Build new class_names and target_to_id
    class_names = filtered_targets + ["unknown_agg"]
    target_to_id = {name: i for i, name in enumerate(filtered_targets)}
    
    print(f"Filtered class names: {class_names}")
    print(f"Excluded classes: {exclude_classes}")
    
    # Create classifier
    clf = HTMLTargetMulticlassDOMClassifier(
        class_names=class_names,
        target_to_id=target_to_id,
        label_id_remap=None,
        hashed_dim=hashed_dim,
        n_estimators=n_estimators,
        random_state=42,
        algorithm=algorithm,
        prefer_gpu=True,
    )
    
    return clf


def load_chunks_filter_excluded(
    chunk_files: Sequence[str],
    exclude_class_ids: set,
    label_suffix: str = "_y11.npz",
) -> Tuple[np.ndarray, np.ndarray]:
    """Load chunks and filter out excluded classes."""
    x_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    
    for i, chunk_path in enumerate(chunk_files, start=1):
        chunk_p = Path(chunk_path)
        label_path = chunk_p.with_name(chunk_p.stem + label_suffix)
        
        try:
            with np.load(chunk_path) as data:
                if "X" not in data:
                    print(f"  Skipping {chunk_path}: missing X key")
                    continue
                X_chunk = data["X"]
            
            with np.load(label_path) as label_data:
                if "y11" not in label_data:
                    print(f"  Skipping {label_path}: missing y11 key")
                    continue
                y_chunk = label_data["y11"]
            
            # Filter to only keep samples NOT in excluded classes
            mask = ~np.isin(y_chunk, list(exclude_class_ids))
            X_filtered = X_chunk[mask]
            y_filtered = y_chunk[mask]
            
            print(f"  [{i:3d}] Chunk {chunk_p.name}: {len(y_chunk):7,} → {len(y_filtered):7,} (kept {len(y_filtered)/len(y_chunk)*100:.1f}%)")
            
            if len(X_filtered) > 0:
                x_parts.append(X_filtered)
                y_parts.append(y_filtered)
        
        except Exception as e:
            print(f"  Error loading chunk {chunk_path}: {e}")
            continue
    
    if not x_parts:
        raise ValueError("No chunks loaded successfully")
    
    return np.vstack(x_parts).astype(np.float32), np.concatenate(y_parts).astype(np.int16)


def remap_labels_exclude(
    y: np.ndarray,
    original_class_names: List[str],
    exclude_classes: set,
) -> np.ndarray:
    """Remap labels after excluding classes, and replace excluded with unknown."""
    exclude_classes = {c.lower() for c in exclude_classes}
    
    # Build remap dict
    remap = {}
    new_id = 0
    unknown_id = len(original_class_names) - 1  # Last class is 'unknown_agg'
    
    for old_id, class_name in enumerate(original_class_names):
        if class_name.lower() in exclude_classes:
            remap[old_id] = unknown_id
        else:
            remap[old_id] = new_id
            new_id += 1
    
    # Apply remap
    y_remapped = np.empty_like(y)
    for old_id, new_id in remap.items():
        y_remapped[y == old_id] = new_id
    
    return y_remapped


def main():
    parser = argparse.ArgumentParser(description="Train DOM classifier excluding majority classes")
    parser.add_argument("--train-chunks", nargs="+", required=True, help="Training chunk files")
    parser.add_argument("--test-chunks", nargs="+", required=True, help="Test chunk files")
    parser.add_argument("--mapping-path", required=True, help="Class mapping JSON")
    parser.add_argument("--model-path", default="dom_mc_no_majority.pkl", help="Output model path")
    parser.add_argument("--algorithm", default="random_forest", help="Model algorithm")
    parser.add_argument("--exclude-classes", nargs="+", default=list(EXCLUDED_CLASSES), help="Classes to exclude")
    
    args = parser.parse_args()
    
    # Load baseline mapping
    print("=" * 80)
    print("Loading baseline class mapping...")
    mapping = load_balanced_mapping(args.mapping_path)
    all_class_names = mapping.get("top_targets", []) + ["unknown_agg"]
    
    # Get original class IDs for excluded classes
    exclude_set = {c.lower() for c in args.exclude_classes}
    exclude_ids = {i for i, name in enumerate(all_class_names[:-1]) if name.lower() in exclude_set}
    
    print(f"Baseline classes: {all_class_names}")
    print(f"Excluding: {args.exclude_classes} (IDs: {exclude_ids})")
    
    # Create filtered classifier
    print("\n" + "=" * 80)
    print("Creating filtered classifier...")
    clf = create_filtered_classifier(
        mapping,
        algorithm=args.algorithm,
        exclude_classes=exclude_set,
    )
    filtered_class_names = clf.class_names
    print(f"Filtered class names: {filtered_class_names}")
    
    # Load and filter training data
    print("\n" + "=" * 80)
    print("Loading and filtering training chunks...")
    X_train, y_train = load_chunks_filter_excluded(
        args.train_chunks,
        exclude_ids,
        label_suffix="_y11.npz",
    )
    print(f"Total training samples after filtering: {len(y_train):,}")
    
    # Remap labels (excluded classes → unknown_agg)
    print("Remapping labels...")
    y_train_remapped = remap_labels_exclude(y_train, all_class_names, exclude_set)
    
    # Check class distribution after remap
    unique, counts = np.unique(y_train_remapped, return_counts=True)
    print("\nFiltered training class distribution:")
    for class_id, count in sorted(zip(unique, counts)):
        class_name = filtered_class_names[int(class_id)] if int(class_id) < len(filtered_class_names) else "???"
        print(f"  Class {int(class_id):2d} ({class_name:15s}): {count:8,}")
    
    # Train model
    print("\n" + "=" * 80)
    print("Training model...")
    
    # Manually fit using the classifier's internals
    clf.scaler.fit(X_train)
    X_train_scaled = clf.scaler.transform(X_train)
    
    from sklearn.utils.class_weight import compute_sample_weight
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train_remapped)
    
    clf._fit_estimator(clf.model, X_train_scaled, y_train_remapped, sample_weight=sample_weights)
    clf.is_fitted = True
    
    y_pred_train = clf.model.predict(X_train_scaled)
    train_acc = accuracy_score(y_train_remapped, y_pred_train)
    train_macro_f1 = f1_score(y_train_remapped, y_pred_train, average="macro", zero_division=0)
    train_weighted_f1 = f1_score(y_train_remapped, y_pred_train, average="weighted", zero_division=0)
    
    print(f"Training complete:")
    print(f"  Train Accuracy: {train_acc:.4f}")
    print(f"  Train Macro-F1: {train_macro_f1:.4f}")
    print(f"  Train Weighted-F1: {train_weighted_f1:.4f}")
    
    # Load and filter test chunks (keep ALL samples, just get full test set)
    print("\n" + "=" * 80)
    print("Loading full test chunks (no filtering)...")
    x_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    
    for i, chunk_path in enumerate(args.test_chunks, start=1):
        chunk_p = Path(chunk_path)
        label_path = chunk_p.with_name(chunk_p.stem + "_y11.npz")
        
        try:
            with np.load(chunk_path) as data:
                if "X" not in data:
                    continue
                x_parts.append(data["X"])
            
            with np.load(label_path) as label_data:
                if "y11" not in label_data:
                    continue
                y_parts.append(label_data["y11"])
            
            print(f"  [{i:3d}] Loaded {chunk_p.name}")
        except Exception as e:
            print(f"  Error loading {chunk_path}: {e}")
            continue
    
    X_test = np.vstack(x_parts).astype(np.float32)
    y_test = np.concatenate(y_parts).astype(np.int16)
    print(f"Total test samples: {len(y_test):,}")
    
    # For fair comparison: remap test labels the same way
    y_test_remapped = remap_labels_exclude(y_test, all_class_names, exclude_set)
    
    # Evaluate on full test set
    print("\n" + "=" * 80)
    print("Evaluating on full test set...")
    X_test_scaled = clf.scaler.transform(X_test)
    y_test_pred = clf.model.predict(X_test_scaled)
    
    test_acc = accuracy_score(y_test_remapped, y_test_pred)
    test_macro_f1 = f1_score(y_test_remapped, y_test_pred, average="macro", zero_division=0)
    test_weighted_f1 = f1_score(y_test_remapped, y_test_pred, average="weighted", zero_division=0)
    
    print(f"Test Results (model trained WITHOUT {args.exclude_classes}):")
    print(f"  Test Accuracy:    {test_acc:.4f}")
    print(f"  Test Macro-F1:    {test_macro_f1:.4f}")
    print(f"  Test Weighted-F1: {test_weighted_f1:.4f}")
    
    # Get binary metrics (exclude unknown_agg from phishing detection)
    unknown_id = len(filtered_class_names) - 1
    yb_test_remapped = (y_test_remapped != unknown_id).astype(np.int16)
    yb_pred = (y_test_pred != unknown_id).astype(np.int16)
    
    test_binary_acc = accuracy_score(yb_test_remapped, yb_pred)
    test_binary_f1 = f1_score(yb_test_remapped, yb_pred, average="binary", zero_division=0)
    
    print(f"  Test Binary Acc:  {test_binary_acc:.4f}")
    print(f"  Test Binary F1:   {test_binary_f1:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test_remapped, y_test_pred, labels=list(range(len(filtered_class_names))))
    
    print("\n" + "=" * 80)
    print("Confusion Matrix:")
    print(f"Shape: {cm.shape}")
    
    # Save results
    results = {
        "cmd": "mc-train-excluded",
        "algorithm": args.algorithm,
        "excluded_classes": args.exclude_classes,
        "class_names": filtered_class_names,
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "train_accuracy": float(train_acc),
        "train_macro_f1": float(train_macro_f1),
        "train_weighted_f1": float(train_weighted_f1),
        "test_accuracy": float(test_acc),
        "test_macro_f1": float(test_macro_f1),
        "test_weighted_f1": float(test_weighted_f1),
        "test_binary_accuracy": float(test_binary_acc),
        "test_binary_f1": float(test_binary_f1),
        "confusion_matrix": cm.tolist(),
        "report": classification_report(
            y_test_remapped,
            y_test_pred,
            labels=list(range(len(filtered_class_names))),
            target_names=filtered_class_names,
            zero_division=0,
            output_dict=True,
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    
    # Append to results log
    results_log = Path("dom_brand_stage1_results.jsonl")
    with open(results_log, "a") as f:
        f.write(json.dumps(results) + "\n")
    
    print(f"\nResults saved to {results_log}")
    
    # Save model
    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump({"model": clf.model, "scaler": clf.scaler, "class_names": filtered_class_names}, f)
    print(f"Model saved to {model_path}")
    
    print("\n" + "=" * 80)
    print("SUMMARY: Model trained without Facebook, Meta, USPS")
    print("=" * 80)


if __name__ == "__main__":
    main()
