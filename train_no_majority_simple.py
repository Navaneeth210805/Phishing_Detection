#!/usr/bin/env python3
"""
Simpler training script focusing on essentials.
"""
import json
import pickle
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight

# Configuration
EXCLUDED_CLASSES = {"facebook", "meta", "usps"}
TRAIN_CHUNKS_DIR = Path("dom_unsup_features")
MAPPING_FILE = "dom_stage1_mapping.json"

print("=" * 80)
print("TRAINING DOM CLASSIFIER WITHOUT MAJORITY CLASSES")
print("=" * 80)

# Load mapping
with open(MAPPING_FILE, 'r') as f:
    mapping = json.load(f)

all_classes = mapping["top_targets"] + ["unknown_agg"]
exclude_ids = {i for i, name in enumerate(all_classes[:-1]) if name.lower() in EXCLUDED_CLASSES}
filtered_classes = [c for c in all_classes if c.lower() not in EXCLUDED_CLASSES or c == "unknown_agg"]

print(f"All classes: {all_classes}")
print(f"Excluded: {EXCLUDED_CLASSES}")
print(f"Filtered classes: {filtered_classes}")
print(f"Exclude IDs: {exclude_ids}")

# Load training data
print("\n" + "=" * 80)
print("LOADING TRAINING DATA")
print("=" * 80)

train_chunks = sorted(TRAIN_CHUNKS_DIR.glob("train_domfeat_chunk_*.npz"))
train_chunks = [c for c in train_chunks if not str(c).endswith("_y11.npz")]

print(f"Found {len(train_chunks)} training chunks")

X_parts = []
y_parts = []
loaded_count = 0
skipped_count = 0

for i, chunk_path in enumerate(train_chunks[:5], 1):  # Start with just 5 chunks for testing
    label_path = chunk_path.with_name(chunk_path.stem + "_y11.npz")
    
    try:
        with np.load(chunk_path) as data:
            if "X" not in data:
                print(f"  [{i}] Skipping: missing X key")
                skipped_count += 1
                continue
            X_chunk = data["X"]
        
        with np.load(label_path) as label_data:
            if "y11" not in label_data:
                print(f"  [{i}] Skipping: missing y11 key") 
                skipped_count += 1
                continue
            y_chunk = label_data["y11"]
        
        # Filter excluded classes
        mask = ~np.isin(y_chunk, list(exclude_ids))
        X_filtered = X_chunk[mask]
        y_filtered = y_chunk[mask]
        
        print(f"  [{i}] {chunk_path.name}: {len(y_chunk)} → {len(y_filtered)} samples")
        
        X_parts.append(X_filtered)
        y_parts.append(y_filtered)
        loaded_count += 1
        
    except FileNotFoundError as e:
        print(f"  [{i}] File not found: {e}")
        skipped_count += 1
    except Exception as e:
        print(f"  [{i}] Error: {e}")
        skipped_count += 1

if not X_parts:
    print("ERROR: No chunks loaded!")
    exit(1)

X_train = np.vstack(X_parts).astype(np.float32)
y_train = np.concatenate(y_parts).astype(np.int16)

print(f"\nLoaded: {loaded_count}, Skipped: {skipped_count}")
print(f"Training data: X shape {X_train.shape}, y shape {y_train.shape}")
print(f"Class distribution: {np.unique(y_train, return_counts=True)}")

# Remap labels (exclude IDs → unknown_agg ID)
print("\nRemapping excluded classes to unknown...")
y_train_remapped = np.empty_like(y_train)
unknown_new_id = len(filtered_classes) - 1

# Build remap
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

print(f"Remapped distribution: {dict(zip(*np.unique(y_train_remapped, return_counts=True)))}")

# Train model
print("\n" + "=" * 80)
print("TRAINING RANDOM FOREST")
print("=" * 80)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1, verbose=1)
weights = compute_sample_weight("balanced", y_train_remapped)
model.fit(X_train_scaled, y_train_remapped, sample_weight=weights)

y_pred_train = model.predict(X_train_scaled)
train_acc = accuracy_score(y_train_remapped, y_pred_train)
train_f1 = f1_score(y_train_remapped, y_pred_train, average="macro", zero_division=0)

print(f"✓ Training complete!")
print(f"  Train Accuracy: {train_acc:.4f}")
print(f"  Train Macro-F1: {train_f1:.4f}")

#Save for now
model_data = {
    "model": model,
    "scaler": scaler,
    "class_names": filtered_classes,
    "original_classes": all_classes,
    "excluded_classes": list(EXCLUDED_CLASSES),
}

with open("dom_mc_no_majority_test.pkl", "wb") as f:
    pickle.dump(model_data, f)

print(f"\n✓ Model saved to dom_mc_no_majority_test.pkl")
print("=" * 80)
print("SUCCESS")
print("=" * 80)
