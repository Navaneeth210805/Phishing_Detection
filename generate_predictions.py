#!/usr/bin/env python3
"""
generate_predictions.py
=======================
Runs the best trained URLPhishNet on the test split and saves a CSV with:
  url, actual_brand, actual_is_phishing, predicted_brand, predicted_is_phishing,
  brand_correct, phishing_correct

The test split is reconstructed using the same seed (RANDOM_STATE=42) and
same 80/20 ratio used during training, so rows are identical to training eval.
"""

from __future__ import annotations

import gc
import glob
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT        = os.path.join(PROJECT_DIR, "phishphresh", "multiclass_brand")
DATA_DIR    = os.path.join(ROOT, "data")
MODEL_DIR   = os.path.join(ROOT, "models")
OUT_CSV     = os.path.join(ROOT, "logs", "test_predictions.csv")

BEST_MODEL  = os.path.join(MODEL_DIR, "model_best.pth")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
MAP_PATH    = os.path.join(MODEL_DIR, "class_map.json")

# Must match training constants exactly
MAX_URL_LEN  = 256
VOCAB_SIZE   = 98
EMBED_DIM    = 64
CNN_CHANNELS = 128
FEAT_MLP_DIM = 256
FEAT_MLP_OUT = 128
DROPOUT      = 0.40
TEST_SIZE    = 0.20
RANDOM_STATE = 42
BATCH_SIZE   = 1024


# ── URL decoder (reverses encode_url) ─────────────────────────────────────────
def decode_url(char_ids: np.ndarray) -> str:
    chars = []
    for code in char_ids:
        if code == 0:
            break
        elif 1 <= code <= 96:
            chars.append(chr(code + 31))   # ASCII 32-127
        else:
            chars.append("?")
    return "".join(chars)


# ── Load all chunks ────────────────────────────────────────────────────────────
def load_chunks() -> tuple[np.ndarray, np.ndarray, list[str]]:
    all_chunks = sorted(
        glob.glob(os.path.join(DATA_DIR, "train", "chunk_*.npz")) +
        glob.glob(os.path.join(DATA_DIR, "test",  "chunk_*.npz"))
    )
    feat_parts, char_parts, tgt_parts = [], [], []
    for f in tqdm(all_chunks, desc="Loading chunks"):
        d = np.load(f, allow_pickle=True)
        feat_parts.append(d["features"])
        char_parts.append(d["char_ids"])
        tgt_parts.extend(d["targets"].tolist())
    X_feat = np.vstack(feat_parts).astype(np.float32)
    X_char = np.vstack(char_parts).astype(np.int64)
    return X_feat, X_char, tgt_parts


# ── Model definition (must match training exactly) ────────────────────────────
class _ResBlock(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(dim, dim),
        )
    def forward(self, x):
        return x + self.block(x)

class FeatureMLP(nn.Module):
    def __init__(self, input_dim, proj_dim=FEAT_MLP_DIM, out_dim=FEAT_MLP_OUT, dropout=DROPOUT):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(input_dim, proj_dim), nn.LayerNorm(proj_dim), nn.GELU())
        self.res_blocks = nn.Sequential(_ResBlock(proj_dim, dropout), _ResBlock(proj_dim, dropout))
        self.out = nn.Sequential(nn.Linear(proj_dim, out_dim), nn.LayerNorm(out_dim), nn.GELU(), nn.Dropout(dropout))
        self.output_dim = out_dim
    def forward(self, x):
        return self.out(self.res_blocks(self.proj(x)))

class CharCNN(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM, n_channels=CNN_CHANNELS):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv3 = nn.Conv1d(embed_dim, n_channels, 3, padding=1)
        self.conv5 = nn.Conv1d(embed_dim, n_channels, 5, padding=2)
        self.conv7 = nn.Conv1d(embed_dim, n_channels, 7, padding=3)
        self.bn3 = nn.BatchNorm1d(n_channels)
        self.bn5 = nn.BatchNorm1d(n_channels)
        self.bn7 = nn.BatchNorm1d(n_channels)
        fused = 3 * n_channels
        self.conv_deep = nn.Conv1d(fused, fused, 3, padding=1)
        self.bn_deep = nn.BatchNorm1d(fused)
        self.drop = nn.Dropout(0.15)
        self.output_dim = 2 * fused
    def forward(self, x):
        emb = self.embedding(x).transpose(1, 2)
        cat = torch.cat([F.relu(self.bn3(self.conv3(emb))),
                         F.relu(self.bn5(self.conv5(emb))),
                         F.relu(self.bn7(self.conv7(emb)))], dim=1)
        feat = self.drop(F.relu(self.bn_deep(self.conv_deep(cat))))
        return torch.cat([F.adaptive_max_pool1d(feat, 1).squeeze(-1),
                          F.adaptive_avg_pool1d(feat, 1).squeeze(-1)], dim=1)

class URLPhishNet(nn.Module):
    def __init__(self, feature_dim, n_classes, dropout=DROPOUT):
        super().__init__()
        self.feat_mlp = FeatureMLP(feature_dim, FEAT_MLP_DIM, FEAT_MLP_OUT, dropout)
        self.char_cnn = CharCNN()
        fusion = self.feat_mlp.output_dim + self.char_cnn.output_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 256),   nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout * 0.7),
            nn.Linear(256, n_classes),
        )
    def forward(self, features, char_ids):
        return self.classifier(torch.cat([self.feat_mlp(features), self.char_cnn(char_ids)], dim=1))


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load class map
    with open(MAP_PATH, encoding="utf-8") as f:
        meta = json.load(f)
    class_names  = meta["class_names"]   # list[str], index → name
    n_classes    = meta["n_classes"]
    other_class  = meta["other_class"]
    print(f"Classes: {n_classes}  ({class_names[0]} / {class_names[1]} / ... / {class_names[-1]})")

    # Load & decode chunks
    print("\nLoading chunks...")
    X_feat, X_char, targets = load_chunks()
    print(f"Total rows: {len(targets):,}")

    # Reconstruct URLs from char_ids
    print("Decoding URLs from char_ids...")
    urls = [decode_url(row) for row in tqdm(X_char, desc="Decoding")]

    # Same split as training
    idx = np.arange(len(targets))
    _, te_idx = train_test_split(idx, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    print(f"Test split: {len(te_idx):,} rows")

    X_feat_te = X_feat[te_idx]
    X_char_te = X_char[te_idx]
    urls_te   = [urls[i]    for i in te_idx]
    tgts_te   = [targets[i] for i in te_idx]
    del X_feat, X_char, urls, targets; gc.collect()

    # Scale features
    scaler    = joblib.load(SCALER_PATH)
    X_feat_te = scaler.transform(X_feat_te).astype(np.float32)

    # Load model
    ckpt  = torch.load(BEST_MODEL, map_location=device, weights_only=True)
    model = URLPhishNet(feature_dim=X_feat_te.shape[1], n_classes=n_classes).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded model from epoch {ckpt['epoch']}  (F1={ckpt['f1_macro']:.4f})")

    # Run inference in batches
    all_preds = []
    Xf_t = torch.from_numpy(X_feat_te)
    Xc_t = torch.from_numpy(X_char_te.astype(np.int64))

    with torch.no_grad():
        for i in tqdm(range(0, len(Xf_t), BATCH_SIZE), desc="Inference"):
            Xf = Xf_t[i:i+BATCH_SIZE].to(device)
            Xc = Xc_t[i:i+BATCH_SIZE].to(device)
            logits = model(Xf, Xc)
            preds  = logits.argmax(1).cpu().numpy()
            all_preds.extend(preds.tolist())

    # Build output DataFrame
    print("\nBuilding output CSV...")
    rows = []
    for i, (url, actual_tgt, pred_idx) in enumerate(zip(urls_te, tgts_te, all_preds)):
        actual_tgt = str(actual_tgt or "").strip()

        # Actual class
        actual_brand     = actual_tgt if actual_tgt else "benign"
        actual_phishing  = actual_tgt != ""

        # Predicted class
        pred_brand    = class_names[pred_idx] if pred_idx < len(class_names) else f"cls{pred_idx}"
        pred_phishing = pred_idx != 0

        rows.append({
            "url":                  url,
            "actual_brand":         actual_brand,
            "actual_is_phishing":   actual_phishing,
            "predicted_brand":      pred_brand,
            "predicted_is_phishing": pred_phishing,
            "phishing_correct":     actual_phishing == pred_phishing,
            "brand_correct":        (pred_brand == actual_brand) or
                                    (actual_tgt == "" and pred_idx == 0),
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {len(df):,} rows -> {OUT_CSV}")

    # Quick summary stats
    total      = len(df)
    ph_correct = df["phishing_correct"].sum()
    br_correct = df["brand_correct"].sum()
    print(f"\n{'='*60}")
    print(f"Phishing detection accuracy : {ph_correct/total*100:.2f}%  ({ph_correct:,}/{total:,})")
    print(f"Brand mapping accuracy      : {br_correct/total*100:.2f}%  ({br_correct:,}/{total:,})")

    # Confusion breakdown
    tp = ((df.actual_is_phishing) & (df.predicted_is_phishing)).sum()
    tn = ((~df.actual_is_phishing) & (~df.predicted_is_phishing)).sum()
    fp = ((~df.actual_is_phishing) & (df.predicted_is_phishing)).sum()
    fn = ((df.actual_is_phishing) & (~df.predicted_is_phishing)).sum()
    print(f"\nPhishing confusion matrix:")
    print(f"  True Positive  (caught phishing correctly) : {tp:>8,}")
    print(f"  True Negative  (benign correctly safe)     : {tn:>8,}")
    print(f"  False Positive (benign called phishing)    : {fp:>8,}")
    print(f"  False Negative (phishing missed)           : {fn:>8,}")
    print(f"\nCSV columns: url | actual_brand | actual_is_phishing | predicted_brand | predicted_is_phishing | phishing_correct | brand_correct")
    print(f"{'='*60}")
