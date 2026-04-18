#!/usr/bin/env python3
"""
run_52class_adversary.py
========================
Loads the trained 52-class URLPhishNet (model_best.pth) and runs it on
the three adversarial datasets from Sabir et al. (arXiv:2005.08454).

Since all adversarial URLs are phishing (class != benign), we measure:
  - Phishing detection rate (% adversarial URLs flagged as NOT benign)
  - Accuracy, F1-Macro, MCC, FPR, FNR vs the binary ground truth

Saves results to:
  phishphresh/multiclass_brand/logs/adversarial_results.txt
  phishphresh/multiclass_brand/logs/adversarial_predictions_<type>.csv
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import urllib.parse

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, f1_score, matthews_corrcoef, confusion_matrix,
)
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT         = os.path.join(PROJECT_DIR, "phishphresh", "multiclass_brand")
MODEL_DIR    = os.path.join(ROOT, "models")
LOGS_DIR     = os.path.join(ROOT, "logs")

BEST_MODEL   = os.path.join(MODEL_DIR, "model_best.pth")
SCALER_PATH  = os.path.join(MODEL_DIR, "scaler.pkl")
MAP_PATH     = os.path.join(MODEL_DIR, "class_map.json")

REPACK_DATA  = os.path.join(
    PROJECT_DIR, "Replication_Package", "Replication_Package", "Datasets"
)
ADV_DIR      = os.path.join(REPACK_DATA, "Adversary_Dataset")
DOMAIN_ADV   = os.path.join(ADV_DIR, "DomainAdversary.csv")
PATH_ADV     = os.path.join(ADV_DIR, "PathAdversary.csv")
TLD_ADV      = os.path.join(ADV_DIR, "TLDAdversary.csv")

OUT_TXT      = os.path.join(LOGS_DIR, "adversarial_results.txt")

# ── Must match training ────────────────────────────────────────────────────────
MAX_URL_LEN  = 256
VOCAB_SIZE   = 98
EMBED_DIM    = 64
CNN_CHANNELS = 128
FEAT_MLP_DIM = 256
FEAT_MLP_OUT = 128
DROPOUT      = 0.40
BATCH_SIZE   = 1024
MAX_ADV_PATH = 50_000   # path adversary has 1M rows; sample for speed

# ── Feature extractor ─────────────────────────────────────────────────────────
sys.path.insert(0, PROJECT_DIR)
from main import FeatureExtractor as _BaseFeatureExtractor  # noqa: E402

_RE_IP       = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
_RE_HEX      = re.compile(r"%[0-9a-fA-F]{2}")
_RE_REDIRECT = re.compile(r"(url|redirect|redir|next|return|goto|target)=", re.I)
_EXTRA_DIM   = 16


def _safe_port(parsed) -> bool:
    try:
        return bool(parsed.port and parsed.port not in (80, 443))
    except Exception:
        return False


def _shannon(s: str) -> float:
    if not s:
        return 0.0
    freq: dict[str, int] = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    n = len(s)
    return -sum((v / n) * math.log2(v / n) for v in freq.values())


class URLFeatureExtractorV2:
    def __init__(self) -> None:
        self._base = _BaseFeatureExtractor()
        try:
            _probe = list(self._base.extract_features("http://example.com").values())
            self._base_dim = len(_probe)
        except Exception:
            self._base_dim = 51

    @property
    def n_features(self) -> int:
        return self._base_dim + _EXTRA_DIM

    def extract(self, url: str) -> np.ndarray:
        url = str(url or "").strip()
        try:
            vals = list(self._base.extract_features(url).values())
            if len(vals) < self._base_dim:
                vals += [0.0] * (self._base_dim - len(vals))
            base = np.array(vals[:self._base_dim], dtype=np.float32)
        except Exception:
            base = np.zeros(self._base_dim, dtype=np.float32)
        return np.concatenate([base, self._structural(url)])

    @staticmethod
    def _structural(url: str) -> np.ndarray:
        try:
            parsed = urllib.parse.urlparse(
                url if "://" in url else "http://" + url
            )
        except Exception:
            return np.zeros(_EXTRA_DIM, dtype=np.float32)
        path    = parsed.path    or ""
        query   = parsed.query   or ""
        fragment = parsed.fragment or ""
        netloc  = parsed.netloc  or ""
        scheme  = (parsed.scheme or "http").lower()
        host    = netloc.split("@")[-1].split(":")[0]
        return np.array([
            min(len(url), 2000) / 2000.0,
            min(len(path), 500) / 500.0,
            min(len(query), 500) / 500.0,
            min(len(fragment), 200) / 200.0,
            float(path.count("/")),
            float(len(urllib.parse.parse_qs(query))),
            float(scheme == "https"),
            float(_safe_port(parsed)),
            float("@" in netloc),
            float(bool(_RE_IP.fullmatch(host))),
            len(_RE_HEX.findall(url)) / max(len(url), 1),
            float(bool(_RE_REDIRECT.search(query))),
            float("//" in path),
            _shannon(url),
            _shannon(query),
            _shannon(path),
        ], dtype=np.float32)


def encode_url(url: str, max_len: int = MAX_URL_LEN) -> np.ndarray:
    arr = np.zeros(max_len, dtype=np.int16)
    for i, ch in enumerate(url[:max_len]):
        code = ord(ch)
        arr[i] = (code - 31) if 32 <= code < 128 else 97
    return arr


# ── Model (identical to training) ─────────────────────────────────────────────
class _ResBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(dim, dim),
        )
    def forward(self, x): return x + self.block(x)

class FeatureMLP(nn.Module):
    def __init__(self, input_dim, proj_dim=FEAT_MLP_DIM, out_dim=FEAT_MLP_OUT, dropout=DROPOUT):
        super().__init__()
        self.proj       = nn.Sequential(nn.Linear(input_dim, proj_dim), nn.LayerNorm(proj_dim), nn.GELU())
        self.res_blocks = nn.Sequential(_ResBlock(proj_dim, dropout), _ResBlock(proj_dim, dropout))
        self.out        = nn.Sequential(nn.Linear(proj_dim, out_dim), nn.LayerNorm(out_dim), nn.GELU(), nn.Dropout(dropout))
        self.output_dim = out_dim
    def forward(self, x): return self.out(self.res_blocks(self.proj(x)))

class CharCNN(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM, n_channels=CNN_CHANNELS):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv3 = nn.Conv1d(embed_dim, n_channels, 3, padding=1)
        self.conv5 = nn.Conv1d(embed_dim, n_channels, 5, padding=2)
        self.conv7 = nn.Conv1d(embed_dim, n_channels, 7, padding=3)
        self.bn3   = nn.BatchNorm1d(n_channels)
        self.bn5   = nn.BatchNorm1d(n_channels)
        self.bn7   = nn.BatchNorm1d(n_channels)
        fused      = 3 * n_channels
        self.conv_deep = nn.Conv1d(fused, fused, 3, padding=1)
        self.bn_deep   = nn.BatchNorm1d(fused)
        self.drop  = nn.Dropout(0.15)
        self.output_dim = 2 * fused
    def forward(self, x):
        emb  = self.embedding(x).transpose(1, 2)
        cat  = torch.cat([F.relu(self.bn3(self.conv3(emb))),
                          F.relu(self.bn5(self.conv5(emb))),
                          F.relu(self.bn7(self.conv7(emb)))], dim=1)
        feat = self.drop(F.relu(self.bn_deep(self.conv_deep(cat))))
        return torch.cat([F.adaptive_max_pool1d(feat, 1).squeeze(-1),
                          F.adaptive_avg_pool1d(feat, 1).squeeze(-1)], dim=1)

class URLPhishNet(nn.Module):
    def __init__(self, feature_dim, n_classes, dropout=DROPOUT):
        super().__init__()
        self.feat_mlp   = FeatureMLP(feature_dim)
        self.char_cnn   = CharCNN()
        fusion          = self.feat_mlp.output_dim + self.char_cnn.output_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 256),   nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout * 0.7),
            nn.Linear(256, n_classes),
        )
    def forward(self, features, char_ids):
        return self.classifier(
            torch.cat([self.feat_mlp(features), self.char_cnn(char_ids)], dim=1)
        )


# ── Run inference on a URL list ───────────────────────────────────────────────
def run_inference(model, scaler, urls: list[str], extractor, device, label: int,
                  class_names: list[str], desc: str) -> pd.DataFrame:
    feats, chars = [], []
    for url in tqdm(urls, desc=f"  Features [{desc}]"):
        feats.append(extractor.extract(url))
        chars.append(encode_url(url))
    X_feat = scaler.transform(np.vstack(feats).astype(np.float32)).astype(np.float32)
    X_char = np.vstack(chars).astype(np.int64)

    Xf_t = torch.from_numpy(X_feat)
    Xc_t = torch.from_numpy(X_char)
    preds, scores = [], []
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(Xf_t), BATCH_SIZE), desc=f"  Inference [{desc}]"):
            Xf = Xf_t[i:i+BATCH_SIZE].to(device)
            Xc = Xc_t[i:i+BATCH_SIZE].to(device)
            logits = model(Xf, Xc).float()
            prob_phish = torch.softmax(logits, dim=1)[:, 0]   # class 0 = benign
            p = logits.argmax(1).cpu().numpy()
            preds.extend(p.tolist())
            scores.extend(prob_phish.cpu().numpy().tolist())

    pred_names = [class_names[p] if p < len(class_names) else f"cls{p}" for p in preds]
    pred_binary = [0 if p == 0 else 1 for p in preds]   # 0=benign, 1=phishing
    return pd.DataFrame({
        "url": urls,
        "actual_binary": label,
        "predicted_class": pred_names,
        "predicted_binary": pred_binary,
        "correct": [label == b for b in pred_binary],
    })


def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    cm  = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    return dict(Accuracy=acc, F1_Macro=f1m, MCC=mcc, FPR=fpr, FNR=fnr,
                TP=int(tp), TN=int(tn), FP=int(fp), FN=int(fn))


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load class map
    with open(MAP_PATH, encoding="utf-8") as f:
        meta = json.load(f)
    class_names = meta["class_names"]
    n_classes   = meta["n_classes"]
    print(f"Classes: {n_classes}  |  benign_idx=0")

    # Load scaler + model
    scaler   = joblib.load(SCALER_PATH)
    extractor = URLFeatureExtractorV2()
    FEAT_DIM  = extractor.n_features

    ckpt  = torch.load(BEST_MODEL, map_location=device, weights_only=True)
    model = URLPhishNet(feature_dim=FEAT_DIM, n_classes=n_classes).to(device)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded model from epoch {ckpt['epoch']}  (F1={ckpt['f1_macro']:.4f})")

    results = {}

    adversaries = [
        ("Domain_Adversary", DOMAIN_ADV, None),
        ("Path_Adversary",   PATH_ADV,   MAX_ADV_PATH),
        ("TLD_Adversary",    TLD_ADV,    None),
    ]

    for name, csv_path, cap in adversaries:
        print(f"\n{'='*60}")
        print(f"Dataset: {name}")
        df_adv = pd.read_csv(csv_path, usecols=["craftedurl"])
        urls   = df_adv["craftedurl"].dropna().astype(str).tolist()
        if cap and len(urls) > cap:
            print(f"  Capping {len(urls):,} -> {cap:,} for speed")
            urls = urls[:cap]
        print(f"  URLs: {len(urls):,}  (all phishing=1)")

        df_result = run_inference(model, scaler, urls, extractor, device,
                                  label=1, class_names=class_names, desc=name)

        # Save per-URL predictions
        out_csv = os.path.join(LOGS_DIR, f"adversarial_52class_{name}.csv")
        df_result.to_csv(out_csv, index=False)
        print(f"  Saved -> {out_csv}")

        y_true = df_result["actual_binary"].values
        y_pred = df_result["predicted_binary"].values
        m = compute_metrics(y_true, y_pred)
        results[name] = m

        detect_rate = df_result["correct"].mean() * 100
        print(f"  Detection rate : {detect_rate:.2f}%")
        print(f"  Accuracy       : {m['Accuracy']*100:.2f}%")
        print(f"  F1-Macro       : {m['F1_Macro']:.4f}")
        print(f"  MCC            : {m['MCC']:.4f}")
        print(f"  FPR            : {m['FPR']*100:.2f}%  FNR: {m['FNR']*100:.2f}%")
        print(f"  TP={m['TP']:,}  FN={m['FN']:,}  (FN = adversarial URLs that evaded detection)")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("52-CLASS URLPhishNet — Adversarial Detection Summary")
    print(f"{'='*70}")
    hdr = f"  {'Dataset':<22} {'Accuracy':>9} {'F1-Macro':>9} {'MCC':>7} {'FNR%':>7} (FNR = miss rate)"
    print(hdr)
    print("  " + "-" * (len(hdr)-2))
    for name, m in results.items():
        print(f"  {name:<22} {m['Accuracy']*100:>8.2f}% {m['F1_Macro']:>9.4f} "
              f"{m['MCC']:>7.4f} {m['FNR']*100:>6.2f}%")

    # Write text report
    lines = [
        "52-class URLPhishNet — Adversarial Evaluation Report",
        f"Model: epoch {ckpt['epoch']}  train-F1={ckpt['f1_macro']:.4f}",
        "=" * 60,
        "",
        f"{'Dataset':<24} {'Acc%':>7} {'F1-Macro':>9} {'MCC':>7} {'FPR%':>7} {'FNR%':>7}",
        "-" * 60,
    ]
    for name, m in results.items():
        lines.append(
            f"{name:<24} {m['Accuracy']*100:>6.2f}% {m['F1_Macro']:>9.4f} "
            f"{m['MCC']:>7.4f} {m['FPR']*100:>6.2f}% {m['FNR']*100:>6.2f}%"
        )
    lines += [
        "",
        "Note: FNR = False Negative Rate = % adversarial phishing URLs that",
        "      evaded detection (model predicted benign when URL is phishing).",
        "Lower FNR is better. Paper models have FNR ~2-8% on normal data.",
    ]
    report = "\n".join(lines)
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport saved -> {OUT_TXT}")
