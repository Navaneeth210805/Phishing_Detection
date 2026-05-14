#!/usr/bin/env python3
"""
Train and evaluate DOM classifier excluding majority classes (Facebook, Meta, USPS).
Compare metrics against baseline to see impact of removing imbalanced classes.
"""
import json
import pickle
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight

# Configuration
EXCLUDED_CLASSES = {"facebook", "meta", "usps"}
TRAIN_CHUNKS_DIR = Path("dom_unsup_features")
MAPPING_FILE = "dom_stage1_mapping.json"

print("=" * 100)
print(" " * 30 + "TRAINING WITHOUT MAJORITY CLASSES")
print("=" * 100)

# Load mapping
with open(MAPPING_FILE, 'r') as f:
    mapping = json.load(f)

all_classes = mapping["top_targets"] + ["unknown_agg"]
exclude_ids = {i for i, name in enumerate(all_classes[:-1]) if name.lower() in EXCLUDED_CLASSES}
unknown_id_orig = len(all_classes) - 1

print(f"Excluded classes: {', '.join(EXCLUDED_CLASSES)} (orig IDs: {sorted(exclude_ids)})")
print(f"Remaining classes: {[c for c in all_classes if c.lower() not in EXCLUDED_CLASSES]} + unknown_agg\n")

# ============================================================================
# LOAD TRAINING DATA
# ============================================================================
print("=" * 100)
print("LOADING TRAINING DATA")  
print("=" * 100)

train_chunks = sorted(TRAIN_CHUNKS_DIR.glob("train_domfeat_chunk_*.npz"))
train_chunks = [c for c in train_chunks if not str(c).endswith("_y11.npz")]

print(f"Found {len(train_chunks)} training chunks\n")

X_parts = []
y_parts = []
class_counts_all = Counter()
class_counts_kept = Counter()

for i, chunk_path in enumerate(train_chunks, 1):
    label_path = chunk_path.with_name(chunk_path.stem + "_y11.npz")
    
    try:
        with np.load(chunk_path) as data:
            X_chunk = data["X"]
        
        with np.load(label_path) as label_data:
            y_chunk = label_data["y11"]
        
        # Count before filtering
        unique_orig, counts_orig = np.unique(y_chunk, return_counts=True)
        for uid, cnt in zip(unique_orig, counts_orig):
            class_counts_all[int(uid)] += cnt
        
        # Filter excluded classes
        mask = ~np.isin(y_chunk, list(exclude_ids))
        X_filtered = X_chunk[mask]
        y_filtered = y_chunk[mask]
        
        # Count after filtering
        unique_filt, counts_filt = np.unique(y_filtered, return_counts=True)
        for uid, cnt in zip(unique_filt, counts_filt):
            class_counts_kept[int(uid)] += cnt
        
        X_parts.append(X_filtered)
        y_parts.append(y_filtered)
        
        if i % 100 == 0 or i == len(train_chunks):
            print(f"  [{i:4d}/{len(train_chunks)}] Loaded {chunk_path.name}")
        
    except Exception as e:
        print(f"  [{i:4d}] ERROR: {e}")
        continue

X_train = np.vstack(X_parts).astype(np.float32)
y_train = np.concatenate(y_parts).astype(np.int16)

print(f"\nTraining data loaded:")
print(f"  X shape: {X_train.shape}")
print(f"  y shape: {y_train.shape}")
print(f"  Total samples kept: {len(y_train):,} / {sum(class_counts_all.values()):,}")
print(f"  Removal rate: {(1 - len(y_train)/sum(class_counts_all.values()))*100:.1f}%")

print(f"\nClass distribution (kept):")
for class_id in sorted(set(y_train)):
    class_name = all_classes[class_id] if class_id < len(all_classes) else "?"
    count = class_counts_kept[class_id]
    pct = count / len(y_train) * 100
    print(f"  {class_id:2d} ({class_name:15s}): {count:7,} ({pct:5.1f}%)")

# ============================================================================
# REMAP LABELS AND TRAIN
# ============================================================================
print("\n" + "=" * 100)
print("TRAINING MODEL")
print("=" * 100)

# Build filtered class list and remap
filtered_classes = [c for c in all_classes if c.lower() not in EXCLUDED_CLASSES]
filtered_classes.append("unknown_agg")
unknown_new_id = len(filtered_classes) - 1

print(f"Training with {len(filtered_classes)} classes: {filtered_classes}\n")

# Remap: excluded classes → unknown_agg
y_train_remapped = np.empty_like(y_train, dtype=np.int16)
remap = {}
new_id = 0
for old_id in range(len(all_classes)):
    if old_id in exclude_ids:
        remap[old_id] = unknown_new_id
    else:
        remap[old_id] = new_id
        new_id += 1

for old_id, new_id in remap.items():
    y_train_remapped[y_train == old_id] = new_id

# Check remapped distribution
unique_remap, counts_remap = np.unique(y_train_remapped, return_counts=True)
print("Remapped class distribution:")
for uid, cnt in zip(unique_remap, counts_remap):
    class_name = filtered_classes[uid] if uid < len(filtered_classes) else "?"
    pct = cnt / len(y_train_remapped) * 100
    print(f"  {uid:2d} ({class_name:15s}): {cnt:7,} ({pct:5.1f}%)")

# Scale and train
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

print(f"\nTraining RandomForest with {len(X_train)} samples...")
model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1, verbose=0)
weights = compute_sample_weight("balanced", y_train_remapped)
model.fit(X_train_scaled, y_train_remapped, sample_weight=weights)

y_pred_train = model.predict(X_train_scaled)
train_acc = accuracy_score(y_train_remapped, y_pred_train)
train_macro_f1 = f1_score(y_train_remapped, y_pred_train, average="macro", zero_division=0)
train_weighted_f1 = f1_score(y_train_remapped, y_pred_train, average="weighted", zero_division=0)

print(f"✓ Training complete!")
print(f"  TRAIN Accuracy:    {train_acc:.4f}")
print(f"  TRAIN Macro-F1:    {train_macro_f1:.4f}")
print(f"  TRAIN Weighted-F1: {train_weighted_f1:.4f}")

# ============================================================================
# LOAD AND EVALUATE ON TESTSET
# ============================================================================
print("\n" + "=" * 100)
print("LOADING TEST DATA")
print("=" * 100)

test_chunks = sorted(TRAIN_CHUNKS_DIR.glob("test_domfeat_chunk_*.npz"))
test_chunks = [c for c in test_chunks if not str(c).endswith("_y11.npz")]

print(f"Found {len(test_chunks)} test chunks\n")

X_test_parts = []
y_test_parts = []

for i, chunk_path in enumerate(test_chunks, 1):
    label_path = chunk_path.with_name(chunk_path.stem + "_y11.npz")
    
    try:
        with np.load(chunk_path) as data:
            X_test_parts.append(data["X"])
        
        with np.load(label_path) as label_data:
            y_test_parts.append(label_data["y11"])
        
        if i % 100 == 0 or i == len(test_chunks):
            print(f"  [{i:4d}/{len(test_chunks)}] Loaded {chunk_path.name}")
    except Exception as e:
        print(f"  [{i:4d}] ERROR: {e}")
        continue

X_test = np.vstack(X_test_parts).astype(np.float32)
y_test = np.concatenate(y_test_parts).astype(np.int16)

print(f"\nTest data loaded: X{X_test.shape}, y{y_test.shape}")

# Remap test labels with same logic
y_test_remapped = np.empty_like(y_test, dtype=np.int16)
for old_id, new_id in remap.items():
    y_test_remapped[y_test == old_id] = new_id

print(f"\nTest class distribution (after remapping):")
unique_test, counts_test = np.unique(y_test_remapped, return_counts=True)
for uid, cnt in zip(unique_test, counts_test):
    class_name = filtered_classes[uid] if uid < len(filtered_classes) else "?"
    pct = cnt / len(y_test_remapped) * 100
    print(f"  {uid:2d} ({class_name:15s}): {cnt:7,} ({pct:5.1f}%)")

# Evaluate
print("\n" + "=" * 100)
print("EVALUATION ON TEST SET")
print("=" * 100 + "\n")

X_test_scaled = scaler.transform(X_test)
y_test_pred = model.predict(X_test_scaled)

test_acc = accuracy_score(y_test_remapped, y_test_pred)
test_macro_f1 = f1_score(y_test_remapped, y_test_pred, average="macro", zero_division=0)
test_weighted_f1 = f1_score(y_test_remapped, y_test_pred, average="weighted", zero_division=0)

print(f"TEST METRICS (Model trained WITHOUT {', '.join(EXCLUDED_CLASSES)}):")
print(f"  Accuracy:    {test_acc:.4f}")
print(f"  Macro-F1:    {test_macro_f1:.4f}")
print(f"  Weighted-F1: {test_weighted_f1:.4f}")

# Binary detection (phishing vs benign)
yb_test = (y_test_remapped != unknown_new_id).astype(np.int16)
yb_pred = (y_test_pred != unknown_new_id).astype(np.int16)
test_binary_acc = accuracy_score(yb_test, yb_pred)
test_binary_f1 = f1_score(yb_test, yb_pred, average="binary", zero_division=0)

print(f"\nBINARY METRICS (Phishing Detection):")
print(f"  Binary Accuracy: {test_binary_acc:.4f}")
print(f"  Binary F1:       {test_binary_f1:.4f}")

# Confusion matrix
cm = confusion_matrix(y_test_remapped, y_test_pred, labels=list(range(len(filtered_classes))))

# Save results
results = {
    "cmd": "mc-train-excluded",
    "algorithm": "random_forest",
    "excluded_classes": list(EXCLUDED_CLASSES),
    "class_names": filtered_classes,
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
        labels=list(range(len(filtered_classes))),
        target_names=filtered_classes,
        zero_division=0,
        output_dict=True,
    ),
    "timestamp": datetime.now(timezone.utc).isoformat(),
}

# Append to results log
with open("dom_brand_stage1_results.jsonl", "a") as f:
    f.write(json.dumps(results) + "\n")

print(f"\n✓ Results saved to dom_brand_stage1_results.jsonl")

# Save model
with open("dom_mc_no_majority_rf.pkl", "wb") as f:
    pickle.dump({
        "model": model,
        "scaler": scaler,
        "class_names": filtered_classes,
    }, f)

print(f"✓ Model saved to dom_mc_no_majority_rf.pkl")

# ============================================================================
# COMPARISON WITH BASELINE
# ============================================================================
print("\n" + "=" * 100)
print("COMPARISON WITH BASELINE")
print("=" * 100 + "\n")

# Read baseline
with open("dom_brand_stage1_results.jsonl", 'r') as f:
    lines = f.readlines()
    baseline = json.loads(lines[0])  # First entry should be baseline RF

baseline_acc = baseline["test_accuracy"]
baseline_macro_f1 = baseline["test_macro_f1"]
baseline_binary_acc = baseline["test_binary_accuracy"]

print("BASELINE (trained WITH all 10 classes):")
print(f"  Test Accuracy:    {baseline_acc:.4f}")
print(f"  Macro-F1:         {baseline_macro_f1:.4f}")
print(f"  Binary Accuracy:  {baseline_binary_acc:.4f}")

print("\nNO-MAJORITY (trained WITHOUT Facebook, Meta, USPS):")
print(f"  Test Accuracy:    {test_acc:.4f}")
print(f"  Macro-F1:         {test_macro_f1:.4f}")
print(f"  Binary Accuracy:  {test_binary_acc:.4f}")

print("\nDIFFERENCE (improvement = positive):")
print(f"  Accuracy Δ:      {test_acc - baseline_acc:+.4f} ({(test_acc - baseline_acc)*100:+.2f}%)")
print(f"  Macro-F1 Δ:      {test_macro_f1 - baseline_macro_f1:+.4f} ({(test_macro_f1 - baseline_macro_f1)*100:+.2f}%)")
print(f"  Binary Acc Δ:    {test_binary_acc - baseline_binary_acc:+.4f} ({(test_binary_acc - baseline_binary_acc)*100:+.2f}%)")

print("\n" + "=" * 100)
print("COMPLETE - Results saved")
print("=" * 100)
