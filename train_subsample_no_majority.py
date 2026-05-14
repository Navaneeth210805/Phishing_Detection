#!/usr/bin/env python3
"""Train multiclass on subsample dataset, excluding facebook/meta/usps classes."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from tqdm import tqdm

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    import xgboost as xgb
except ImportError:
    xgb = None

from dom_target_multiclass_detector import HTMLTargetMulticlassDOMClassifier, _append_jsonl


def load_subsample_chunks(feature_root: str, split: str, label_suffix: str = "_y11.npz") -> tuple:
    """Load feature chunks from subsample directory."""
    root = Path(feature_root)
    split_dir = root / split
    
    chunk_files = sorted([f for f in split_dir.glob("*domfeat_chunk_*.npz") if not f.name.endswith(label_suffix)])
    if not chunk_files:
        raise ValueError(f"No chunks found in {split_dir}")
    
    x_parts = []
    y_parts = []
    
    for chunk_path in tqdm(chunk_files, desc=f"Loading {split} chunks"):
        with np.load(chunk_path) as data:
            x_parts.append(data['X'].astype(np.float32))
        
        sidecar_path = chunk_path.with_name(chunk_path.stem + label_suffix)
        if sidecar_path.exists():
            with np.load(sidecar_path) as data:
                y_parts.append(data['y11'].astype(np.int16))
    
    if not x_parts:
        raise ValueError(f"No feature data loaded for {split}")
    
    X = np.vstack(x_parts).astype(np.float32)
    y = np.concatenate(y_parts).astype(np.int16) if y_parts else None
    
    return X, y


def filter_exclude_classes(X: np.ndarray, y: np.ndarray, exclude_class_ids: List[int]) -> tuple:
    """Filter out samples from excluded classes."""
    mask = ~np.isin(y, exclude_class_ids)
    X_filtered = X[mask]
    y_filtered = y[mask]
    
    # Remap labels to contiguous range
    unique_ids = np.unique(y_filtered)
    remap = {old_id: new_id for new_id, old_id in enumerate(unique_ids)}
    y_remapped = np.array([remap[old_id] for old_id in y_filtered], dtype=np.int16)
    
    return X_filtered, y_remapped, remap


def train_eval_subsample_excluded(
    feature_root: str,
    mapping_path: str,
    algorithm: str,
    model_path: str,
    results_log: str,
    exclude_classes: List[str],
    prefer_gpu: bool = True,
    n_estimators: int = 300,
):
    """Train and evaluate model on subsample, excluding specified classes."""
    
    # Load mapping to get class name -> id mapping
    with open(mapping_path) as f:
        mapping = json.load(f)
    
    top_targets = mapping.get("top_targets", [])
    unknown_id = mapping.get("unknown_class_id", len(top_targets))
    
    # Build class name -> id map (from mapping)
    class_to_id_orig = {name: i for i, name in enumerate(top_targets)}
    class_to_id_orig[mapping.get("unknown_class_name", "unknown_agg")] = unknown_id
    
    # Determine which class IDs to exclude
    exclude_ids = [class_to_id_orig[cls] for cls in exclude_classes if cls in class_to_id_orig]
    print(f"[train-subsample-excluded] Excluding class IDs {exclude_ids} for classes {exclude_classes}")
    
    # Load train and test chunks
    print("[train-subsample-excluded] Loading train chunks...")
    X_train, y_train = load_subsample_chunks(feature_root, "train")
    print(f"  Loaded X_train: {X_train.shape}, y_train: {y_train.shape}")
    
    print("[train-subsample-excluded] Loading test chunks...")
    X_test, y_test = load_subsample_chunks(feature_root, "test")
    print(f"  Loaded X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    # Filter
    print(f"[train-subsample-excluded] Filtering train set...")
    X_train_filtered, y_train_filtered, train_remap = filter_exclude_classes(X_train, y_train, exclude_ids)
    print(f"  Filtered X_train: {X_train_filtered.shape}, y_train: {y_train_filtered.shape}")
    
    print(f"[train-subsample-excluded] Filtering test set...")
    X_test_filtered, y_test_filtered, test_remap = filter_exclude_classes(X_test, y_test, exclude_ids)
    print(f"  Filtered X_test: {X_test_filtered.shape}, y_test: {y_test_filtered.shape}")
    
    # Build new class names (only non-excluded)
    remaining_class_names = [name for name, id_ in class_to_id_orig.items() if id_ not in exclude_ids]
    print(f"[train-subsample-excluded] Remaining classes: {remaining_class_names}")
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_filtered)
    X_test_scaled = scaler.transform(X_test_filtered)
    
    # Build model
    print(f"[train-subsample-excluded] Building {algorithm} model...")
    if algorithm == "random_forest":
        model = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, random_state=42)
    elif algorithm == "extra_trees":
        model = ExtraTreesClassifier(n_estimators=n_estimators, n_jobs=-1, random_state=42)
    elif algorithm == "logistic_regression":
        model = LogisticRegression(max_iter=1000, n_jobs=-1, random_state=42)
    elif algorithm == "lightgbm" and lgb is not None:
        model = lgb.LGBMClassifier(n_estimators=n_estimators, num_leaves=31, random_state=42, verbose=-1)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    # Train
    print(f"[train-subsample-excluded] Training {algorithm}...")
    model.fit(X_train_scaled, y_train_filtered)
    
    # Evaluate
    print(f"[train-subsample-excluded] Evaluating on test set...")
    y_pred = model.predict(X_test_scaled)
    
    test_acc = float(accuracy_score(y_test_filtered, y_pred))
    test_macro_f1 = float(f1_score(y_test_filtered, y_pred, average="macro", zero_division=0))
    test_weighted_f1 = float(f1_score(y_test_filtered, y_pred, average="weighted", zero_division=0))
    
    cm = confusion_matrix(y_test_filtered, y_pred, labels=list(range(len(remaining_class_names))))
    report = classification_report(
        y_test_filtered, y_pred,
        labels=list(range(len(remaining_class_names))),
        target_names=remaining_class_names,
        zero_division=0,
        output_dict=True
    )
    
    print(f"[train-subsample-excluded] Test Accuracy: {test_acc:.4f}, Macro F1: {test_macro_f1:.4f}, Weighted F1: {test_weighted_f1:.4f}")
    
    # Save model
    print(f"[train-subsample-excluded] Saving model to {model_path}...")
    import pickle
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"  Model saved.")
    
    # Append result
    result = {
        "cmd": "mc-train-excluded-subsample",
        "algorithm": algorithm,
        "excluded_classes": exclude_classes,
        "class_names": remaining_class_names,
        "train_samples": int(X_train_filtered.shape[0]),
        "test_samples": int(X_test_filtered.shape[0]),
        "test_accuracy": test_acc,
        "test_macro_f1": test_macro_f1,
        "test_weighted_f1": test_weighted_f1,
        "confusion_matrix": cm.tolist(),
        "report": report,
        "model_path": model_path,
    }
    
    _append_jsonl(results_log, result)
    print(f"[train-subsample-excluded] Result appended to {results_log}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train multiclass on subsample, excluding majority classes")
    parser.add_argument("--feature-root", required=True, help="Root dir with train/ and test/ subdirs")
    parser.add_argument("--mapping-path", required=True, help="Path to top10_target_map.json")
    parser.add_argument("--algorithm", required=True, choices=["random_forest", "extra_trees", "logistic_regression", "lightgbm"])
    parser.add_argument("--model-path", required=True, help="Where to save model")
    parser.add_argument("--results-log", default="dom_brand_stage1_results.jsonl")
    parser.add_argument("--exclude-classes", nargs="+", default=["facebook", "meta", "usps"], help="Classes to exclude")
    parser.add_argument("--prefer-gpu", action="store_true", default=True)
    parser.add_argument("--n-estimators", type=int, default=300)
    
    args = parser.parse_args()
    
    train_eval_subsample_excluded(
        feature_root=args.feature_root,
        mapping_path=args.mapping_path,
        algorithm=args.algorithm,
        model_path=args.model_path,
        results_log=args.results_log,
        exclude_classes=args.exclude_classes,
        prefer_gpu=args.prefer_gpu,
        n_estimators=args.n_estimators,
    )
