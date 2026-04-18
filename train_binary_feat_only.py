#!/usr/bin/env python3
"""
train_binary_feat_only.py
=========================
Ablation: Binary phishing detection using ONLY the 67 handcrafted features.
No CharCNN — pure FeatureMLP baseline. Same phishphresh dataset, same split.

Goal: compare against train_binary.py (CharCNN + FeatureMLP) to quantify
how much the character-level CNN contributes.

Outputs: binary_feat_only/models/, binary_feat_only/logs/
"""

from __future__ import annotations

import gc
import glob
import logging
import os
import sys

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
PHISH_ROOT  = os.path.join(PROJECT_DIR, "phishphresh", "multiclass_brand")
DATA_DIR    = os.path.join(PHISH_ROOT, "data")

REPACK_DATA = os.path.join(
    PROJECT_DIR, "Replication_Package", "Replication_Package", "Datasets"
)
ADV_DIR    = os.path.join(REPACK_DATA, "Adversary_Dataset")
DOMAIN_ADV = os.path.join(ADV_DIR, "DomainAdversary.csv")
PATH_ADV   = os.path.join(ADV_DIR, "PathAdversary.csv")
TLD_ADV    = os.path.join(ADV_DIR, "TLDAdversary.csv")

OUT_MODELS  = os.path.join(PROJECT_DIR, "binary_feat_only", "models")
OUT_LOGS    = os.path.join(PROJECT_DIR, "binary_feat_only", "logs")
MODEL_PATH  = os.path.join(OUT_MODELS, "model_best.pth")
SCALER_PATH = os.path.join(OUT_MODELS, "scaler.pkl")
EPOCH_LOG   = os.path.join(OUT_LOGS, "training_log.csv")
REPORT_TXT  = os.path.join(OUT_LOGS, "adversarial_results.txt")

# ── Hyper-parameters ───────────────────────────────────────────────────────────
FEAT_MLP_DIM = 256
FEAT_MLP_OUT = 128
DROPOUT      = 0.40
TEST_SIZE    = 0.20
RANDOM_STATE = 42
BATCH_SIZE   = 1024
EPOCHS       = 30
LR           = 3e-4
WEIGHT_DECAY = 3e-4
MAX_ADV_PATH = 50_000

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

# ── Model (FeatureMLP only, no CharCNN) ────────────────────────────────────────
class _ResBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(dim, dim),
        )
    def forward(self, x): return x + self.block(x)


class BinaryFeatNet(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, FEAT_MLP_DIM), nn.LayerNorm(FEAT_MLP_DIM), nn.GELU()
        )
        self.res = nn.Sequential(
            _ResBlock(FEAT_MLP_DIM, DROPOUT),
            _ResBlock(FEAT_MLP_DIM, DROPOUT),
        )
        self.mid = nn.Sequential(
            nn.Linear(FEAT_MLP_DIM, FEAT_MLP_OUT), nn.LayerNorm(FEAT_MLP_OUT),
            nn.GELU(), nn.Dropout(DROPOUT)
        )
        self.clf = nn.Sequential(
            nn.Linear(FEAT_MLP_OUT, 64), nn.GELU(), nn.Dropout(DROPOUT * 0.5),
            nn.Linear(64, 2)
        )

    def forward(self, feat):
        return self.clf(self.mid(self.res(self.proj(feat))))


# ── Dataset (features only) ────────────────────────────────────────────────────
class FeatDataset(Dataset):
    def __init__(self, Xf, y):
        self.Xf = torch.from_numpy(Xf)
        self.y  = torch.from_numpy(y.astype(np.int64))
    def __len__(self):        return len(self.y)
    def __getitem__(self, i): return self.Xf[i], self.y[i]


# ── Helpers ────────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    cm  = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    return dict(Accuracy=acc, F1_Macro=f1m, MCC=mcc,
                FPR=fpr, FNR=fnr, TP=int(tp), TN=int(tn), FP=int(fp), FN=int(fn))


def run_inference(model, Xf, device):
    model.eval()
    Xft   = torch.from_numpy(Xf)
    preds = []
    with torch.no_grad():
        for i in range(0, len(Xft), BATCH_SIZE):
            preds.extend(model(Xft[i:i+BATCH_SIZE].to(device)).float().argmax(1).cpu().numpy())
    return np.array(preds)


def adv_eval(model, scaler, device, adv_path, adv_name, cap=None):
    """Extract features for adversarial URLs and evaluate."""
    import math, re, urllib.parse
    from main import FeatureExtractor as _FE

    _RE_IP       = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
    _RE_HEX      = re.compile(r"%[0-9a-fA-F]{2}")
    _RE_REDIRECT = re.compile(r"(url|redirect|redir|next|return|goto|target)=", re.I)
    _EXTRA_DIM   = 16

    def _safe_port(p):
        try: return bool(p.port and p.port not in (80, 443))
        except: return False

    def _shannon(s):
        if not s: return 0.0
        freq = {}
        for c in s: freq[c] = freq.get(c, 0) + 1
        n = len(s)
        return -sum((v/n)*math.log2(v/n) for v in freq.values())

    base_fe = _FE()
    try:
        probe     = list(base_fe.extract_features("http://example.com").values())
        base_dim  = len(probe)
    except Exception:
        base_dim  = 51

    def extract(url):
        url = str(url or "").strip()
        try:
            vals = list(base_fe.extract_features(url).values())
            if len(vals) < base_dim: vals += [0.0] * (base_dim - len(vals))
            base = np.array(vals[:base_dim], dtype=np.float32)
        except Exception:
            base = np.zeros(base_dim, dtype=np.float32)
        try:
            parsed   = urllib.parse.urlparse(url if "://" in url else "http://" + url)
            path     = parsed.path or ""
            query    = parsed.query or ""
            fragment = parsed.fragment or ""
            netloc   = parsed.netloc or ""
            host     = netloc.split("@")[-1].split(":")[0]
            extra = np.array([
                min(len(url), 2000)/2000.0, min(len(path), 500)/500.0,
                min(len(query), 500)/500.0, min(len(fragment), 200)/200.0,
                float(path.count("/")), float(len(urllib.parse.parse_qs(query))),
                float((parsed.scheme or "http").lower() == "https"),
                float(_safe_port(parsed)), float("@" in netloc),
                float(bool(_RE_IP.fullmatch(host))),
                len(_RE_HEX.findall(url)) / max(len(url), 1),
                float(bool(_RE_REDIRECT.search(query))),
                float("//" in path), _shannon(url), _shannon(query), _shannon(path),
            ], dtype=np.float32)
        except Exception:
            extra = np.zeros(_EXTRA_DIM, dtype=np.float32)
        return np.concatenate([base, extra])

    try:
        urls = pd.read_csv(adv_path, usecols=["craftedurl"])["craftedurl"].dropna().astype(str).tolist()
        if cap and len(urls) > cap:
            log.info(f"  {adv_name}: {len(urls):,} URLs, capping to {cap:,}")
            urls = urls[:cap]
        else:
            log.info(f"  {adv_name}: {len(urls):,} URLs")
        Xf = np.vstack([extract(u) for u in tqdm(urls, desc=f"  {adv_name}", mininterval=2)]).astype(np.float32)
        Xf = scaler.transform(Xf).astype(np.float32)
        y  = np.ones(len(urls), dtype=np.int64)
        p  = run_inference(model, Xf, device)
        m  = compute_metrics(y, p)
        m["Dataset"] = adv_name
        out = os.path.join(OUT_LOGS, f"adversarial_binary_{adv_name}.csv")
        pd.DataFrame({"url": urls, "predicted": p.tolist(), "correct": (p == 1).tolist()}).to_csv(out, index=False)
        log.info(f"  {adv_name}: acc={m['Accuracy']*100:.2f}%  F1={m['F1_Macro']:.4f}  "
                 f"MCC={m['MCC']:.4f}  FNR={m['FNR']*100:.2f}%  -> {out}")
        return m
    except Exception as e:
        log.error(f"  {adv_name} failed: {e}")
        return None


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    log.info("ABLATION: FeatureMLP only (no CharCNN) — binary, phishphresh")

    # 1. Load phishphresh chunks (features only, skip char_ids)
    log.info("[1/5] Loading phishphresh chunks (features only)...")
    all_chunks = sorted(
        glob.glob(os.path.join(DATA_DIR, "train", "chunk_*.npz")) +
        glob.glob(os.path.join(DATA_DIR, "test",  "chunk_*.npz"))
    )
    log.info(f"  Found {len(all_chunks)} chunks")

    feat_parts, tgt_parts = [], []
    for f in tqdm(all_chunks, desc="  Loading chunks"):
        d = np.load(f, allow_pickle=True)
        feat_parts.append(d["features"])
        tgt_parts.extend(d["targets"].tolist())

    X_feat = np.vstack(feat_parts).astype(np.float32)
    del feat_parts; gc.collect()

    labels  = np.array([0 if str(t or "").strip() == "" else 1 for t in tgt_parts], dtype=np.int64)
    total   = len(labels)
    n_ben   = (labels == 0).sum()
    n_phish = (labels == 1).sum()
    log.info(f"  Total: {total:,}  benign={n_ben:,}  phishing={n_phish:,}")

    # 2. Same 80/20 split (RANDOM_STATE=42)
    log.info("[2/5] 80/20 split (RANDOM_STATE=42)...")
    idx = np.arange(total)
    tr_idx, te_idx = train_test_split(idx, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    Xf_tr, Xf_te = X_feat[tr_idx], X_feat[te_idx]
    y_tr,  y_te  = labels[tr_idx], labels[te_idx]
    del X_feat; gc.collect()
    log.info(f"  Train={len(tr_idx):,}  Test={len(te_idx):,}")

    # 3. Scale
    log.info("[3/5] Fitting StandardScaler...")
    scaler = StandardScaler()
    Xf_tr  = scaler.fit_transform(Xf_tr).astype(np.float32)
    Xf_te  = scaler.transform(Xf_te).astype(np.float32)
    joblib.dump(scaler, SCALER_PATH)
    FEAT_DIM = Xf_tr.shape[1]
    log.info(f"  Feature dim: {FEAT_DIM}  scaler -> {SCALER_PATH}")

    # 4. Train
    log.info(f"[4/5] Training BinaryFeatNet ({EPOCHS} epochs)...")
    counts   = np.bincount(y_tr)
    sample_w = (1.0 / counts)[y_tr]
    sampler  = WeightedRandomSampler(sample_w, num_samples=len(y_tr), replacement=True)
    train_dl = DataLoader(FeatDataset(Xf_tr, y_tr),
                          batch_size=BATCH_SIZE, sampler=sampler,
                          num_workers=0, pin_memory=True)

    model     = BinaryFeatNet(feat_dim=FEAT_DIM).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, steps_per_epoch=len(train_dl),
        epochs=EPOCHS, pct_start=0.1
    )
    phish_w   = float(np.clip(counts[0] / counts[1], 1.0, 3.0))
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, phish_w], device=device, dtype=torch.float32)
    )
    log.info(f"  Phishing class weight: {phish_w:.3f}")

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
        m = compute_metrics(y_te, run_inference(model, Xf_te, device))
        epoch_rows.append({"epoch": epoch, "loss": round(avg_loss, 6), **m})
        log.info(f"  Epoch {epoch:2d}/{EPOCHS}  loss={avg_loss:.4f}  "
                 f"acc={m['Accuracy']*100:.2f}%  F1={m['F1_Macro']:.4f}  "
                 f"MCC={m['MCC']:.4f}  FPR={m['FPR']*100:.2f}%  FNR={m['FNR']*100:.2f}%")

        if m["F1_Macro"] > best_f1:
            best_f1    = m["F1_Macro"]
            best_epoch = epoch
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "f1_macro": m["F1_Macro"], "mcc": m["MCC"],
                        "accuracy": m["Accuracy"]}, MODEL_PATH)

    pd.DataFrame(epoch_rows).to_csv(EPOCH_LOG, index=False)
    log.info(f"  Epoch log -> {EPOCH_LOG}")
    log.info(f"  Best model: epoch {best_epoch}  F1={best_f1:.4f}")

    # 5. Adversarial evaluation
    log.info("[5/5] Adversarial evaluation...")
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])

    normal_m = compute_metrics(y_te, run_inference(model, Xf_te, device))
    log.info(f"  Normal test: acc={normal_m['Accuracy']*100:.2f}%  "
             f"F1={normal_m['F1_Macro']:.4f}  MCC={normal_m['MCC']:.4f}")

    adv_results = []
    for adv_path, adv_name, cap in [
        (DOMAIN_ADV, "Domain_Adversary", None),
        (PATH_ADV,   "Path_Adversary",   MAX_ADV_PATH),
        (TLD_ADV,    "TLD_Adversary",    None),
    ]:
        m = adv_eval(model, scaler, device, adv_path, adv_name, cap)
        if m:
            adv_results.append(m)

    sep = "=" * 80
    lines = [sep,
             "ABLATION: FeatureMLP Only (no CharCNN) — Binary, phishphresh",
             f"Model: BinaryFeatNet | 67 features | {EPOCHS} epochs",
             f"Best epoch: {best_epoch}  |  Best F1: {best_f1:.4f}",
             sep, "",
             f"  {'Dataset':<35} {'Accuracy':>9} {'F1-Macro':>9} {'MCC':>7} {'FPR%':>6} {'FNR%':>6}",
             "  " + "-" * 76,
             f"  {'Normal_Test (phishphresh)':<35} "
             f"{normal_m['Accuracy']*100:>8.2f}% {normal_m['F1_Macro']:>9.4f} "
             f"{normal_m['MCC']:>7.4f} {normal_m['FPR']*100:>5.2f}% {normal_m['FNR']*100:>5.2f}%"]
    for m in adv_results:
        lines.append(f"  {m['Dataset']:<35} "
                     f"{m['Accuracy']*100:>8.2f}% {m['F1_Macro']:>9.4f} "
                     f"{m['MCC']:>7.4f} {'N/A':>5}  {m['FNR']*100:>5.2f}%")
    lines += [sep]

    report = "\n".join(lines)
    log.info("\n" + report)
    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write(report)
    log.info(f"Report -> {REPORT_TXT}")
    log.info("Done.")
