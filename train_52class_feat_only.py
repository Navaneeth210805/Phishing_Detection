#!/usr/bin/env python3
"""
train_52class_feat_only.py
==========================
Ablation: 52-class phishing detection using ONLY the 67 handcrafted features.
No CharCNN — pure FeatureMLP baseline. Same phishphresh dataset, same split.

Goal: compare against train_multiclass_brand.py (CharCNN + FeatureMLP) to
quantify how much the character CNN contributes to the full 52-class task.

Outputs: multiclass_feat_only/models/, multiclass_feat_only/logs/
"""

from __future__ import annotations

import gc
import glob
import json
import logging
import os
import sys

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
PHISH_ROOT  = os.path.join(PROJECT_DIR, "phishphresh", "multiclass_brand")
DATA_DIR    = os.path.join(PHISH_ROOT, "data")
CLASS_MAP_JSON = os.path.join(PHISH_ROOT, "models", "class_map.json")

OUT_MODELS  = os.path.join(PROJECT_DIR, "multiclass_feat_only", "models")
OUT_LOGS    = os.path.join(PROJECT_DIR, "multiclass_feat_only", "logs")
MODEL_PATH  = os.path.join(OUT_MODELS, "model_best.pth")
SCALER_PATH = os.path.join(OUT_MODELS, "scaler.pkl")
EPOCH_LOG   = os.path.join(OUT_LOGS, "training_log.csv")
REPORT_TXT  = os.path.join(OUT_LOGS, "results.txt")

# ── Hyper-parameters ───────────────────────────────────────────────────────────
FEAT_MLP_DIM = 256
FEAT_MLP_OUT = 128
DROPOUT      = 0.40
TEST_SIZE    = 0.20
RANDOM_STATE = 42
BATCH_SIZE   = 512
EPOCHS       = 30
LR           = 3e-4
WEIGHT_DECAY = 3e-4

# ── Logging ────────────────────────────────────────────────────────────────────
os.makedirs(OUT_MODELS, exist_ok=True)
os.makedirs(OUT_LOGS,   exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(os.path.join(OUT_LOGS, "run.log"), encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

sys.path.insert(0, PROJECT_DIR)


# ── Model (FeatureMLP only) ────────────────────────────────────────────────────
class _ResBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(dim, dim),
        )
    def forward(self, x): return x + self.block(x)


class MulticlassFeatNet(nn.Module):
    def __init__(self, feat_dim, n_classes):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, FEAT_MLP_DIM), nn.LayerNorm(FEAT_MLP_DIM), nn.GELU()
        )
        self.res = nn.Sequential(
            _ResBlock(FEAT_MLP_DIM, DROPOUT),
            _ResBlock(FEAT_MLP_DIM, DROPOUT),
            _ResBlock(FEAT_MLP_DIM, DROPOUT),
        )
        self.mid = nn.Sequential(
            nn.Linear(FEAT_MLP_DIM, FEAT_MLP_OUT), nn.LayerNorm(FEAT_MLP_OUT),
            nn.GELU(), nn.Dropout(DROPOUT)
        )
        self.clf = nn.Sequential(
            nn.Linear(FEAT_MLP_OUT, 128), nn.GELU(), nn.Dropout(DROPOUT * 0.5),
            nn.Linear(128, n_classes)
        )

    def forward(self, feat):
        return self.clf(self.mid(self.res(self.proj(feat))))


# ── Dataset ────────────────────────────────────────────────────────────────────
class FeatDataset(Dataset):
    def __init__(self, Xf, y):
        self.Xf = torch.from_numpy(Xf)
        self.y  = torch.from_numpy(y.astype(np.int64))
    def __len__(self):        return len(self.y)
    def __getitem__(self, i): return self.Xf[i], self.y[i]


# ── Helpers ────────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, n_classes):
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    f1w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    # Binary phishing detection rate (class 0 = benign, everything else = phishing)
    y_bin_true = (y_true > 0).astype(int)
    y_bin_pred = (y_pred > 0).astype(int)
    tp = ((y_bin_true == 1) & (y_bin_pred == 1)).sum()
    fp = ((y_bin_true == 0) & (y_bin_pred == 1)).sum()
    tn = ((y_bin_true == 0) & (y_bin_pred == 0)).sum()
    fn = ((y_bin_true == 1) & (y_bin_pred == 0)).sum()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    return dict(Accuracy=acc, F1_Macro=f1m, F1_Weighted=f1w, MCC=mcc, FPR=fpr, FNR=fnr)


def run_inference(model, Xf, device):
    model.eval()
    Xft   = torch.from_numpy(Xf)
    preds = []
    with torch.no_grad():
        for i in range(0, len(Xft), BATCH_SIZE):
            preds.extend(model(Xft[i:i+BATCH_SIZE].to(device)).float().argmax(1).cpu().numpy())
    return np.array(preds)


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    log.info("ABLATION: FeatureMLP only (no CharCNN) — 52-class, phishphresh")

    # Load class map from existing 52-class model
    log.info(f"[1/5] Loading class map from {CLASS_MAP_JSON}...")
    with open(CLASS_MAP_JSON, encoding="utf-8") as f:
        cmap_data = json.load(f)
    class_map  = cmap_data["class_map"]        # brand_str -> class_id
    n_classes  = cmap_data["n_classes"]        # 52
    other_cls  = cmap_data["other_class"]      # 51
    class_names = cmap_data["class_names"]
    log.info(f"  {n_classes} classes (including benign + other_phishing)")

    # Load phishphresh chunks (features only)
    log.info("[2/5] Loading phishphresh chunks (features only)...")
    all_chunks = sorted(
        glob.glob(os.path.join(DATA_DIR, "train", "chunk_*.npz")) +
        glob.glob(os.path.join(DATA_DIR, "test",  "chunk_*.npz"))
    )
    log.info(f"  Found {len(all_chunks)} chunks")

    feat_parts, tgt_parts = [], []
    for fp in tqdm(all_chunks, desc="  Loading chunks"):
        d = np.load(fp, allow_pickle=True)
        feat_parts.append(d["features"])
        tgt_parts.extend(d["targets"].tolist())

    X_feat = np.vstack(feat_parts).astype(np.float32)
    del feat_parts; gc.collect()

    # Map targets to class IDs (same logic as train_multiclass_brand.py)
    labels = np.array(
        [class_map.get(str(t or "").strip().lower(), other_cls) for t in tgt_parts],
        dtype=np.int64
    )
    total = len(labels)
    n_ben = (labels == 0).sum()
    n_phi = (labels > 0).sum()
    log.info(f"  Total: {total:,}  benign={n_ben:,}  phishing={n_phi:,}")

    # 3. Same 80/20 split (RANDOM_STATE=42)
    log.info("[3/5] 80/20 split (RANDOM_STATE=42)...")
    idx = np.arange(total)
    tr_idx, te_idx = train_test_split(idx, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    Xf_tr, Xf_te = X_feat[tr_idx], X_feat[te_idx]
    y_tr,  y_te  = labels[tr_idx], labels[te_idx]
    del X_feat; gc.collect()
    log.info(f"  Train={len(tr_idx):,}  Test={len(te_idx):,}")

    # 4. Scale
    log.info("[4/5] Fitting StandardScaler...")
    scaler = StandardScaler()
    Xf_tr  = scaler.fit_transform(Xf_tr).astype(np.float32)
    Xf_te  = scaler.transform(Xf_te).astype(np.float32)
    joblib.dump(scaler, SCALER_PATH)
    FEAT_DIM = Xf_tr.shape[1]
    log.info(f"  Feature dim: {FEAT_DIM}  scaler -> {SCALER_PATH}")

    # 5. Train
    log.info(f"[5/5] Training MulticlassFeatNet ({EPOCHS} epochs, {n_classes} classes)...")
    counts   = np.bincount(y_tr, minlength=n_classes).astype(np.float64)
    counts   = np.maximum(counts, 1)
    eff_num  = 1.0 - np.power(0.9999, counts)
    cls_w    = torch.tensor((1.0 / eff_num) / (1.0 / eff_num).sum() * n_classes,
                             dtype=torch.float32, device=device)
    cls_w[0] *= 1.5   # boost benign class weight

    sample_w = (1.0 / counts)[y_tr]
    sampler  = WeightedRandomSampler(sample_w, num_samples=len(y_tr), replacement=True)
    train_dl = DataLoader(FeatDataset(Xf_tr, y_tr),
                          batch_size=BATCH_SIZE, sampler=sampler,
                          num_workers=0, pin_memory=True)

    model     = MulticlassFeatNet(feat_dim=FEAT_DIM, n_classes=n_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, steps_per_epoch=len(train_dl),
        epochs=EPOCHS, pct_start=0.1
    )
    criterion = nn.CrossEntropyLoss(weight=cls_w)

    use_amp    = device.type == "cuda"
    amp_scaler = torch.amp.GradScaler("cuda") if use_amp else None
    amp_dtype  = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16

    best_f1, best_epoch = 0.0, 0
    epoch_rows = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for Xf, y in train_dl:
            Xf, y = Xf.to(device), y.to(device)
            optimizer.zero_grad()
            if use_amp:
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    loss = criterion(model(Xf).float(), y)
                amp_scaler.scale(loss).backward()
                amp_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                amp_scaler.step(optimizer)
                amp_scaler.update()
            else:
                loss = criterion(model(Xf).float(), y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_dl)
        preds = run_inference(model, Xf_te, device)
        m = compute_metrics(y_te, preds, n_classes)
        epoch_rows.append({"epoch": epoch, "loss": round(avg_loss, 6), **m})
        log.info(f"  Epoch {epoch:2d}/{EPOCHS}  loss={avg_loss:.4f}  "
                 f"acc={m['Accuracy']*100:.2f}%  F1={m['F1_Macro']:.4f}  "
                 f"MCC={m['MCC']:.4f}  FPR={m['FPR']*100:.2f}%  FNR={m['FNR']*100:.2f}%")

        if m["F1_Macro"] > best_f1:
            best_f1    = m["F1_Macro"]
            best_epoch = epoch
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "f1_macro": m["F1_Macro"], "mcc": m["MCC"],
                        "accuracy": m["Accuracy"], "n_classes": n_classes}, MODEL_PATH)

    pd.DataFrame(epoch_rows).to_csv(EPOCH_LOG, index=False)
    log.info(f"  Epoch log -> {EPOCH_LOG}")
    log.info(f"  Best model: epoch {best_epoch}  F1={best_f1:.4f}")

    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    preds = run_inference(model, Xf_te, device)
    m = compute_metrics(y_te, preds, n_classes)

    sep = "=" * 80
    lines = [sep,
             "ABLATION: FeatureMLP Only (no CharCNN) — 52-class, phishphresh",
             f"Model: MulticlassFeatNet | {FEAT_DIM} features | {n_classes} classes | {EPOCHS} epochs",
             f"Best epoch: {best_epoch}  |  Best F1-Macro: {best_f1:.4f}",
             sep, "",
             f"  Accuracy    : {m['Accuracy']*100:.2f}%",
             f"  F1-Macro    : {m['F1_Macro']:.4f}",
             f"  F1-Weighted : {m['F1_Weighted']:.4f}",
             f"  MCC         : {m['MCC']:.4f}",
             f"  FPR         : {m['FPR']*100:.2f}%",
             f"  FNR         : {m['FNR']*100:.2f}%",
             "", sep]

    report = "\n".join(lines)
    log.info("\n" + report)
    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write(report)
    log.info(f"Report -> {REPORT_TXT}")
    log.info("Done.")
