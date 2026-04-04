#!/usr/bin/env python3
"""
train_phreshphish_url.py
========================

Trains the SAME neural network (PhishingNet) and uses the SAME feature
extractor (FeatureExtractor) as main.py -- imported directly to guarantee
100% parity -- but applied to the phreshphish/phreshphish HuggingFace
dataset using ONLY the `url` and `label` columns.

Pipeline
--------
1. Stream phreshphish (url+label only via select_columns -- no HTML loaded).
2. Apply FeatureExtractor.extract_features() -> 51 URL features per sample.
3. Checkpoint every CHUNK_SIZE rows to  phishphresh/urls/
4. Stack all chunks, do 80-20 train/test split.
5. StandardScaler (fit on train, transform both).
6. Train PhishingNet for EPOCHS epochs with best-model checkpointing.
7. Final evaluation: accuracy, AUC, classification report.

Key design decisions
--------------------
* select_columns(['url', 'label']): Each parquet batch goes from ~2.3 GB
  to ~50 MB, eliminating the ArrowMemoryError during fast-forward / streaming.
* _complete.flag sentinel: Once a split is fully streamed, we write a flag
  file so the next run skips streaming entirely (no more fast-forward OOM).
* Per-split chunk prefixes: train chunks are data_train_*.npz,
  test chunks are data_test_*.npz -- no monkey-patching needed.
"""

import os
import sys
import gc
import glob
import logging
import time
import io as _io

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, accuracy_score,
)
import joblib
from datasets import load_dataset

# ── Import the exact classes from main.py ─────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from main import PhishingNet, PhishingDataset, FeatureExtractor  # exact same

# ── Configuration ──────────────────────────────────────────────────────────────
HF_TOKEN     = ""
DATASET_NAME = "phreshphish/phreshphish"

STORAGE_DIR  = os.path.join(PROJECT_DIR, "phishphresh", "urls")
os.makedirs(STORAGE_DIR, exist_ok=True)

CHUNK_SIZE   = 10_000   # rows per .npz chunk
MAX_SAMPLES  = None     # None = entire split

EPOCHS       = 100
BATCH_SIZE   = 256
LR           = 0.001
WEIGHT_DECAY = 1e-4
HIDDEN_SIZES = [128, 64, 32, 16]
DROPOUT_RATE = 0.3
TEST_SIZE    = 0.20
RANDOM_STATE = 42

# ── Logging ────────────────────────────────────────────────────────────────────
log_path = os.path.join(STORAGE_DIR, "training.log")

# UTF-8 console handler so Windows cp1252 doesn't crash on unicode chars
_console_stream = (
    _io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                      errors="replace", line_buffering=True)
    if hasattr(sys.stdout, "buffer") else sys.stdout
)
_fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
_ch  = logging.StreamHandler(_console_stream)
_ch.setFormatter(_fmt)
_fh  = logging.FileHandler(log_path, mode="a", encoding="utf-8")
_fh.setFormatter(_fmt)

logging.basicConfig(level=logging.INFO, handlers=[_fh, _ch])
logger = logging.getLogger(__name__)


# ── Label helper ───────────────────────────────────────────────────────────────
def normalise_label(raw) -> int:
    try:
        if isinstance(raw, str):
            return 1 if raw.strip().lower() in {"phish", "phishing", "1"} else 0
        return 1 if int(raw) == 1 else 0
    except Exception:
        return 0


# ── Chunk helpers ──────────────────────────────────────────────────────────────
def _chunk_path(split: str, idx: int) -> str:
    return os.path.join(STORAGE_DIR, f"data_{split}_{idx:05d}.npz")


def _flag_path(split: str) -> str:
    return os.path.join(STORAGE_DIR, f"_{split}_complete.flag")


def _find_chunks(split: str) -> list:
    return sorted(glob.glob(os.path.join(STORAGE_DIR, f"data_{split}_*.npz")))


def _count_samples(chunk_files: list) -> int:
    total = 0
    for f in chunk_files:
        try:
            total += len(np.load(f)["labels"])
        except Exception as e:
            logger.warning(f"Could not read {f}: {e}")
    return total


# ── STEP 1: stream + extract ───────────────────────────────────────────────────
def stream_and_extract(extractor: FeatureExtractor, split: str) -> list:
    """
    Stream `split` from phreshphish, extract 51 URL features per row,
    save chunks.  Returns list of chunk file paths for this split.

    Key fix vs previous version:
    * select_columns(['url', 'label']) -- pyarrow loads ~50 MB per batch
      instead of ~2.3 GB, so fast-forward NEVER OOMs.
    * _complete.flag -- if set, skip ALL streaming (no fast-forward at all).
    """
    flag = _flag_path(split)
    existing = _find_chunks(split)

    # ── Already fully done? ────────────────────────────────────────────────────
    if os.path.exists(flag):
        logger.info(f"[{split}] Complete flag found -- skipping streaming. "
                    f"({len(existing)} chunks on disk)")
        return existing

    already_done   = _count_samples(existing)
    next_chunk_idx = len(existing)

    if already_done:
        logger.info(f"[{split}] Resuming: {len(existing)} chunks, "
                    f"{already_done:,} samples already on disk.")

    # ── Load dataset (url + label columns ONLY) ────────────────────────────────
    # select_columns() in modern datasets pushes column selection down to the
    # pyarrow parquet reader, keeping batches ~50 MB instead of ~2.3 GB.
    # Using features= is too strict -- it fails on schema mismatches.
    logger.info(f"[{split}] Loading HuggingFace stream (url+label only)...")
    dataset = load_dataset(
        DATASET_NAME,
        split=split,
        streaming=True,
        token=HF_TOKEN,
    ).select_columns(["url", "label"])

    # ── Fast-forward (skip entire shards, no per-row next() calls) ─────────────
    if already_done > 0:
        logger.info(f"[{split}] Fast-forwarding {already_done:,} rows via skip()...")
        dataset = dataset.skip(already_done)
        logger.info(f"[{split}] Fast-forward done.")

    iterator = iter(dataset)

    # ── Stream + extract ────────────────────────────────────────────────────────
    buf_feats:  list = []
    buf_labels: list = []
    chunk_files = list(existing)
    session_count = 0

    def flush():
        nonlocal next_chunk_idx, buf_feats, buf_labels
        if not buf_feats:
            return
        fpath = _chunk_path(split, next_chunk_idx)
        np.savez_compressed(
            fpath,
            features=np.vstack(buf_feats).astype(np.float32),
            labels=np.array(buf_labels, dtype=np.int8),
        )
        logger.info(f"  [{split}] chunk #{next_chunk_idx:05d} saved "
                    f"({len(buf_labels):,} rows | total {already_done+session_count:,})")
        chunk_files.append(fpath)
        next_chunk_idx += 1
        buf_feats, buf_labels = [], []
        gc.collect()

    t0 = time.time()
    complete = False
    try:
        for sample in iterator:
            if MAX_SAMPLES is not None and (already_done + session_count) >= MAX_SAMPLES:
                logger.info(f"[{split}] Reached MAX_SAMPLES={MAX_SAMPLES:,}.")
                break

            url   = str(sample.get("url", "") or "")
            label = normalise_label(sample.get("label", 0))

            try:
                fvec = np.array(list(extractor.extract_features(url).values()),
                                dtype=np.float32)
            except Exception:
                fvec = np.zeros(51, dtype=np.float32)

            buf_feats.append(fvec)
            buf_labels.append(label)
            session_count += 1

            if len(buf_feats) >= CHUNK_SIZE:
                flush()

            if session_count % 50_000 == 0:
                rate = session_count / max(time.time() - t0, 1)
                logger.info(f"  [{split}] progress: "
                            f"{already_done+session_count:,} rows | {rate:.0f} rows/s")

        complete = True  # reached end of iterator cleanly

    except KeyboardInterrupt:
        logger.warning(f"[{split}] Interrupted -- saving partial buffer...")
    except Exception as e:
        logger.critical(f"[{split}] Stream error: {e}", exc_info=True)

    flush()  # final partial chunk

    logger.info(f"[{split}] Done. New rows this session: {session_count:,}. "
                f"Total chunks: {len(chunk_files)}.")

    if complete:
        open(flag, "w").close()
        logger.info(f"[{split}] Written complete flag: {flag}")

    return chunk_files


# ── STEP 2: load all chunks ────────────────────────────────────────────────────
def load_all_chunks(chunk_files: list):
    logger.info(f"Loading {len(chunk_files)} chunk file(s) from disk...")
    feat_parts, label_parts = [], []
    for f in chunk_files:
        try:
            d = np.load(f)
            feat_parts.append(d["features"])
            label_parts.append(d["labels"])
        except Exception as e:
            logger.error(f"Could not load {f}: {e}")

    X = np.vstack(feat_parts).astype(np.float32)
    y = np.concatenate(label_parts).astype(np.int64)
    logger.info(f"Dataset loaded: X={X.shape}  y={y.shape}")
    logger.info(f"Labels: benign={int((y==0).sum()):,}  "
                f"phishing={int((y==1).sum()):,}")
    return X, y


# ── STEP 3: train ─────────────────────────────────────────────────────────────
def train(X_train_scaled, X_test_scaled, y_train, y_test, device):
    n_feat = X_train_scaled.shape[1]
    logger.info(f"PhishingNet: {n_feat} -> {HIDDEN_SIZES} -> 2")

    train_ds = PhishingDataset(X_train_scaled, y_train)
    test_ds  = PhishingDataset(X_test_scaled,  y_test)

    class_counts  = np.bincount(y_train)
    class_weights = len(y_train) / (2.0 * class_counts)
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(class_weights[y_train]),
        num_samples=len(y_train),
        replacement=True,
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=0, pin_memory=(device.type == "cuda"))
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=(device.type == "cuda"))

    model = PhishingNet(
        input_size=n_feat,
        hidden_sizes=HIDDEN_SIZES,
        dropout_rate=DROPOUT_RATE,
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=torch.FloatTensor(class_weights).to(device)
    )
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=8, factor=0.5
    )

    best_acc        = 0.0
    best_model_path = os.path.join(STORAGE_DIR, "model_best.pth")
    log_rows        = []

    logger.info(f"Training {EPOCHS} epochs | batch={BATCH_SIZE} | lr={LR}")
    logger.info(f"  Train={len(y_train):,}  Test={len(y_test):,}")
    logger.info(f"  Class weights: benign={class_weights[0]:.4f}  "
                f"phishing={class_weights[1]:.4f}")
    print("=" * 80)

    for epoch in range(1, EPOCHS + 1):
        # ── train ──────────────────────────────────────────────────────────────
        model.train()
        total_loss = 0.0
        for bX, by in train_loader:
            bX, by = bX.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bX), by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)

        # ── eval ───────────────────────────────────────────────────────────────
        model.eval()
        all_preds, all_true, all_probs = [], [], []
        test_loss = 0.0
        with torch.no_grad():
            for bX, by in test_loader:
                bX, by = bX.to(device), by.to(device)
                out  = model(bX)
                test_loss += criterion(out, by).item()
                prob = F.softmax(out, dim=1)
                _, pred = torch.max(out, 1)
                all_preds.extend(pred.cpu().numpy())
                all_true.extend(by.cpu().numpy())
                all_probs.extend(prob[:, 1].cpu().numpy())

        acc = accuracy_score(all_true, all_preds)
        auc = roc_auc_score(all_true, all_probs)
        avg_test_loss = test_loss / len(test_loader)
        scheduler.step(acc)

        # ── best checkpoint ────────────────────────────────────────────────────
        if acc > best_acc:
            best_acc = acc
            torch.save({
                "epoch": epoch, "model_state": model.state_dict(),
                "acc": acc, "auc": auc,
            }, best_model_path)
            logger.info(f"  [epoch {epoch}] New best -> Acc={acc:.4f} AUC={auc:.4f} (saved)")

        log_rows.append({
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 6),
            "test_loss":  round(avg_test_loss, 6),
            "test_acc":   round(acc, 6),
            "test_auc":   round(auc, 6),
        })

        print(f"Epoch {epoch:3d}/{EPOCHS} | "
              f"TrainLoss={avg_train_loss:.4f} | "
              f"TestLoss={avg_test_loss:.4f} | "
              f"Acc={acc:.4f} | AUC={auc:.4f}")

    # ── final model ────────────────────────────────────────────────────────────
    final_path = os.path.join(STORAGE_DIR, "model_final.pth")
    torch.save({
        "epoch": EPOCHS, "model_state": model.state_dict(),
        "acc": acc, "auc": auc,
    }, final_path)
    logger.info(f"Final model saved: {final_path}")

    pd.DataFrame(log_rows).to_csv(
        os.path.join(STORAGE_DIR, "training_log.csv"), index=False
    )

    # ── final eval (best checkpoint) ───────────────────────────────────────────
    print("\n" + "=" * 80)
    print("FINAL EVALUATION (best checkpoint)")
    print("=" * 80)
    ckpt = torch.load(best_model_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    all_preds, all_true, all_probs = [], [], []
    with torch.no_grad():
        for bX, by in test_loader:
            bX = bX.to(device)
            out = model(bX)
            prob = F.softmax(out, dim=1)
            _, pred = torch.max(out, 1)
            all_preds.extend(pred.cpu().numpy())
            all_true.extend(by.numpy())
            all_probs.extend(prob[:, 1].cpu().numpy())

    fa  = accuracy_score(all_true, all_preds)
    fau = roc_auc_score(all_true, all_probs)

    print(f"\nTest Accuracy : {fa:.4f} ({fa*100:.2f}%)")
    print(f"Test ROC-AUC  : {fau:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(all_true, all_preds,
                                target_names=["Benign/Suspected", "Phishing"]))
    cm = confusion_matrix(all_true, all_preds)
    print("Confusion Matrix:")
    print(f"            Predicted")
    print(f"          Benign  Phish")
    print(f"Act Benign {cm[0,0]:6d}  {cm[0,1]:5d}")
    print(f"Act Phish  {cm[1,0]:6d}  {cm[1,1]:5d}")

    logger.info(f"Final Acc={fa:.4f} AUC={fau:.4f} | Best Acc={best_acc:.4f}")
    return model


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("train_phreshphish_url.py -- started")
    logger.info(f"Storage : {STORAGE_DIR}")
    logger.info(f"Epochs={EPOCHS}  Batch={BATCH_SIZE}  LR={LR}  TestSplit={TEST_SIZE}")
    logger.info("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)} | CUDA {torch.version.cuda}")

    extractor = FeatureExtractor()   # exact same as main.py

    # ── Step 1: stream both splits (each has its own chunk prefix + flag) ──────
    logger.info("\n[STEP 1a] Train split...")
    train_chunks = stream_and_extract(extractor, split="train")

    logger.info("\n[STEP 1b] Test split (pooled into 80-20 split later)...")
    test_chunks  = stream_and_extract(extractor, split="test")

    all_chunks = train_chunks + test_chunks

    # ── Step 2: load ───────────────────────────────────────────────────────────
    logger.info("\n[STEP 2] Loading all chunks into RAM...")
    X, y = load_all_chunks(all_chunks)

    # ── Step 3: 80-20 split ────────────────────────────────────────────────────
    logger.info("\n[STEP 3] 80-20 train/test split (stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info(f"  Train={len(y_train):,}  Test={len(y_test):,}")
    del X, y; gc.collect()

    # ── Step 4: scale ──────────────────────────────────────────────────────────
    logger.info("\n[STEP 4] StandardScaler (fit on train)...")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_test_s  = scaler.transform(X_test).astype(np.float32)
    del X_train, X_test; gc.collect()

    scaler_path = os.path.join(STORAGE_DIR, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    logger.info(f"  Scaler saved: {scaler_path}")

    # ── Step 5: train ──────────────────────────────────────────────────────────
    logger.info("\n[STEP 5] Training...")
    model = train(X_train_s, X_test_s, y_train, y_test, device)

    logger.info("\nAll done!")
    logger.info(f"  Best model  : {os.path.join(STORAGE_DIR, 'model_best.pth')}")
    logger.info(f"  Final model : {os.path.join(STORAGE_DIR, 'model_final.pth')}")
    logger.info(f"  Scaler      : {scaler_path}")
    logger.info(f"  Training log: {os.path.join(STORAGE_DIR, 'training_log.csv')}")
