#!/usr/bin/env python3
"""
Stream-sample training excluding majority classes to avoid OOM.
- Scans label counts to set per-class quotas (cap).
- Streams through train chunks, sampling up to quota per class.
- Trains RandomForest (reduced n_estimators to save memory) on the sampled set.
- Evaluates on full test set and appends results to jsonl and saves model.
"""
import json
import pickle
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_sample_weight

# Config
EXCLUDED = {"facebook", "meta", "usps"}
MAPPING_FILE = Path("dom_stage1_mapping.json")
TRAIN_DIR = Path("dom_unsup_features")
CAP_PER_CLASS = 10000  # maximum samples per kept class
N_ESTIMATORS = 200
RESULTS_LOG = Path("dom_brand_stage1_results.jsonl")
MODEL_OUT = Path("dom_mc_no_majority_stream_rf.pkl")


def main():
    assert MAPPING_FILE.exists(), "Mapping file missing"
    mapping = json.loads(MAPPING_FILE.read_text())
    all_classes = mapping["top_targets"] + ["unknown_agg"]

    exclude_ids = {i for i, name in enumerate(all_classes[:-1]) if name.lower() in EXCLUDED}
    kept_class_ids = [i for i in range(len(all_classes)) if i not in exclude_ids]
    kept_names = [all_classes[i] for i in kept_class_ids]

    print(f"Excluding classes: {EXCLUDED} -> ids {sorted(exclude_ids)}")
    print(f"Kept classes ({len(kept_names)}): {kept_names}")

    # First pass: compute raw counts per class to set reasonable caps
    counts = Counter()
    print("Scanning label files for class counts (fast pass)...")
    y_files = sorted(TRAIN_DIR.glob("train_domfeat_chunk_*_y11.npz"))
    for yf in y_files:
        try:
            with np.load(yf) as d:
                if "y11" in d:
                    ys = d["y11"]
                    for k, v in zip(*np.unique(ys, return_counts=True)):
                        counts[int(k)] += int(v)
        except Exception:
            continue
    print("Raw class counts (top):")
    for cid, cnt in counts.most_common(12):
        name = all_classes[cid] if cid < len(all_classes) else str(cid)
        print(f"  {cid:2d} {name:15s}: {cnt:,}")

    # Determine per-class quotas for kept classes
    quotas = {}
    for cid in kept_class_ids:
        orig = counts.get(cid, 0)
        quotas[cid] = min(orig, CAP_PER_CLASS) if orig > 0 else 0
    print("Per-class quotas (kept classes):")
    for cid in quotas:
        print(f"  {cid:2d} {all_classes[cid]:15s}: {quotas[cid]:,}")

    # Stream and sample
    X_parts = []
    y_parts = []
    filled = defaultdict(int)
    total_needed = sum(quotas.values())
    total_collected = 0

    print("Streaming train chunks and sampling to quotas...")
    train_chunks = sorted(TRAIN_DIR.glob("train_domfeat_chunk_*.npz"))
    # filter out sidecar label files that match the same glob ("*_y11.npz")
    train_chunks = [c for c in train_chunks if not c.name.endswith("_y11.npz")]
    for i, chunk in enumerate(train_chunks, 1):
        label_path = chunk.with_name(chunk.stem + "_y11.npz")
        try:
            with np.load(chunk) as d:
                Xc = d["X"]
            with np.load(label_path) as ld:
                yc = ld["y11"]
        except Exception:
            continue

        # Exclude unwanted classes first
        mask_excluded = np.isin(yc, list(exclude_ids))
        keep_idx = np.where(~mask_excluded)[0]
        if keep_idx.size == 0:
            continue

        # For each kept class in this chunk, sample up to remaining quota
        selected_idx = []
        for cid in np.unique(yc[keep_idx]):
            needed = quotas.get(int(cid), 0) - filled[int(cid)]
            if needed <= 0:
                continue
            idxs = np.where(yc == cid)[0]
            # intersect with keep_idx
            idxs = np.intersect1d(idxs, keep_idx, assume_unique=True)
            if idxs.size == 0:
                continue
            take = min(len(idxs), needed)
            # random sample
            if take == len(idxs):
                pick = idxs
            else:
                pick = np.random.default_rng(42 + i).choice(idxs, size=take, replace=False)
            selected_idx.extend(pick.tolist())
            filled[int(cid)] += len(pick)
            total_collected += len(pick)

        if selected_idx:
            X_parts.append(Xc[selected_idx])
            y_parts.append(yc[selected_idx])

        if total_collected >= total_needed:
            print(f"Filled all quotas after chunk {i}")
            break

        if i % 500 == 0:
            print(f"  processed {i} chunks, collected {total_collected:,}/{total_needed:,}")

    if total_collected == 0:
        raise RuntimeError("No samples collected; check quotas or data files")

    X_train = np.vstack(X_parts).astype(np.float32)
    y_train = np.concatenate(y_parts).astype(np.int16)
    print(f"Collected training set: X={X_train.shape}, y={y_train.shape}")

    # Remap labels to contiguous 0..K-1 and set excluded -> unknown mapping
    # Build new class list from original classes but avoid duplicating original unknown
    old_unknown = len(all_classes) - 1
    kept_old_ids = [cid for cid in kept_class_ids if cid != old_unknown]
    new_class_list = [all_classes[cid] for cid in kept_old_ids] + [all_classes[old_unknown]]
    unknown_new = len(new_class_list) - 1
    remap = {}
    nid = 0
    # assign new ids for kept old ids (excluding original unknown), then unknown at the end
    for old in range(len(all_classes)):
        if old in exclude_ids:
            remap[old] = unknown_new
        elif old == old_unknown:
            remap[old] = unknown_new
        else:
            remap[old] = nid
            nid += 1

    # Vectorized remap: default to unknown_new, then assign kept ids
    y_train_remap = np.full(y_train.shape, unknown_new, dtype=np.int16)
    # assign for old ids within known range and not excluded
    for old in range(len(all_classes)):
        if old in exclude_ids:
            continue
        if old == old_unknown:
            # original unknown already maps to unknown_new
            continue
        newid = remap[old]
        y_train_remap[y_train == old] = newid

    # Train
    print("Scaling and training RandomForest...")
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)

    clf = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=42, n_jobs=-1)
    sw = compute_sample_weight("balanced", y_train_remap)
    clf.fit(Xs, y_train_remap, sample_weight=sw)

    # Load full test set (smaller)
    print("Loading test set for evaluation...")
    test_chunks = sorted(TRAIN_DIR.glob("test_domfeat_chunk_*.npz"))
    Xt_parts = []
    yt_parts = []
    for chunk in test_chunks:
        label_path = chunk.with_name(chunk.stem + "_y11.npz")
        try:
            with np.load(chunk) as d:
                Xt_parts.append(d["X"])
            with np.load(label_path) as ld:
                yt_parts.append(ld["y11"])
        except Exception:
            continue
    X_test = np.vstack(Xt_parts).astype(np.float32)
    y_test = np.concatenate(yt_parts).astype(np.int16)

    # Remap test labels
    # Remap test labels similarly, default unknown for out-of-range ids
    y_test_remap = np.full(y_test.shape, unknown_new, dtype=np.int16)
    for old in range(len(all_classes)):
        if old in exclude_ids:
            continue
        if old == old_unknown:
            continue
        newid = remap[old]
        y_test_remap[y_test == old] = newid

    X_test_s = scaler.transform(X_test)
    y_pred = clf.predict(X_test_s)

    test_acc = accuracy_score(y_test_remap, y_pred)
    test_macro = f1_score(y_test_remap, y_pred, average="macro", zero_division=0)
    test_weighted = f1_score(y_test_remap, y_pred, average="weighted", zero_division=0)

    # Binary metrics
    yb_test = (y_test_remap != unknown_new).astype(int)
    yb_pred = (y_pred != unknown_new).astype(int)
    test_bin_acc = accuracy_score(yb_test, yb_pred)
    test_bin_f1 = f1_score(yb_test, yb_pred, average="binary", zero_division=0)

    report = classification_report(y_test_remap, y_pred, target_names=new_class_list, zero_division=0, output_dict=True)
    cm = confusion_matrix(y_test_remap, y_pred).tolist()

    results = {
        "cmd": "mc-train-excluded-stream",
        "algorithm": "random_forest",
        "excluded_classes": list(EXCLUDED),
        "class_names": new_class_list,
        "train_samples": int(len(y_train_remap)),
        "test_samples": int(len(y_test_remap)),
        "test_accuracy": float(test_acc),
        "test_macro_f1": float(test_macro),
        "test_weighted_f1": float(test_weighted),
        "test_binary_accuracy": float(test_bin_acc),
        "test_binary_f1": float(test_bin_f1),
        "confusion_matrix": cm,
        "report": report,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    # Append results and save model
    with open(RESULTS_LOG, "a") as f:
        f.write(json.dumps(results) + "\n")
    with open(MODEL_OUT, "wb") as f:
        pickle.dump({"model": clf, "scaler": scaler, "class_names": new_class_list}, f)

    print("Training+eval complete. Results appended and model saved.")
    print(json.dumps({"test_accuracy": test_acc, "test_macro_f1": test_macro, "test_binary_f1": test_bin_f1}, indent=2))


if __name__ == "__main__":
    main()
