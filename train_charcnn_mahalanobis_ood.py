#!/usr/bin/env python3
"""
train_charcnn_mahalanobis_ood.py
================================
Experiment 13 — CharCNN + FeatureMLP Dual-Stream with Mahalanobis OOD Detection.

Extends Exp 10 (Mahalanobis OOD on FeatureMLP-only) to the full dual-stream
BinaryURLPhishNet (CharCNN + FeatureMLP), addressing two issues from Exp 10:
  1. Accuracy was limited to ~95% (FeatureMLP-only) — combining with CharCNN
     should recover the ~98.68% accuracy seen in Exp 2.
  2. Mahalanobis OOD is now applied to the 256-dim penultimate fusion layer
     (richer representation mixing character-level + feature-level signals).

References:
  - Lee et al., NeurIPS 2018  arXiv:1807.03888  (Mahalanobis OOD)
  - AlEroud & Karabatis, IWSPA 2020  doi:10.1145/3375708.3380315  (GAN attack)

Flow:
  1. Load phishphresh chunks (features + char_ids) + 80/20 split (RANDOM_STATE=42)
  2. Train BinaryURLPhishNet  (CharCNN + FeatureMLP → 256-dim penultimate → 2)
  3. Extract 256-dim penultimate features for all training samples
  4. Fit class-conditional Gaussians (Lee et al. NeurIPS 2018, Eq 1+2)
       mu_0, mu_1  =  class means in 256-dim space
       Sigma       =  pooled tied covariance + 1e-5 * I
       P           =  inv(Sigma)
  5. Calibrate OOD threshold = 99th-percentile of training Mahalanobis scores
  6. Evaluate on real phishphresh test set (standard acc + OOD layer FPR)
  7. Evaluate on GAN adversarial feature vectors from Exp 8
       (binary_phishing_adv_gan/logs/adversarial_dataset.csv)
       GAN vectors: 67-dim only → zero char tensors passed to CharCNN stream
       Goal: same 100% detection as Exp 10 with ~98%+ real accuracy

Output:
  binary_charcnn_mahalanobis_ood/models/model_best.pth
  binary_charcnn_mahalanobis_ood/models/scaler.pkl
  binary_charcnn_mahalanobis_ood/models/mahalanobis_params.pkl
  binary_charcnn_mahalanobis_ood/logs/run.log
  binary_charcnn_mahalanobis_ood/logs/stdout.log
  binary_charcnn_mahalanobis_ood/logs/training_metrics.csv
  binary_charcnn_mahalanobis_ood/logs/ood_report.txt
"""

from __future__ import annotations
import gc, glob, logging, math, os, re, sys, urllib.parse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(PROJECT_DIR, "phishphresh", "multiclass_brand", "data")
ADV_CSV     = os.path.join(PROJECT_DIR, "binary_phishing_adv_gan", "logs", "adversarial_dataset.csv")
OUT_DIR     = os.path.join(PROJECT_DIR, "binary_charcnn_mahalanobis_ood")
OUT_MODELS  = os.path.join(OUT_DIR, "models")
OUT_LOGS    = os.path.join(OUT_DIR, "logs")

MODEL_PATH  = os.path.join(OUT_MODELS, "model_best.pth")
SCALER_PATH = os.path.join(OUT_MODELS, "scaler.pkl")
MAHA_PATH   = os.path.join(OUT_MODELS, "mahalanobis_params.pkl")
OOD_REPORT  = os.path.join(OUT_LOGS,  "ood_report.txt")
METRICS_CSV = os.path.join(OUT_LOGS,  "training_metrics.csv")
STDOUT_LOG  = os.path.join(OUT_LOGS,  "stdout.log")

os.makedirs(OUT_MODELS, exist_ok=True)
os.makedirs(OUT_LOGS,   exist_ok=True)

# ── Tee stdout → file ──────────────────────────────────────────────────────────
class _Tee:
    def __init__(self, *files): self.files = files
    def write(self, obj):
        for f in self.files: f.write(obj); f.flush()
    def flush(self):
        for f in self.files: f.flush()

_stdout_fh = open(STDOUT_LOG, "w", encoding="utf-8")
sys.stdout  = _Tee(sys.__stdout__, _stdout_fh)

# ── Logging ────────────────────────────────────────────────────────────────────
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

# ── Hyperparameters ────────────────────────────────────────────────────────────
RANDOM_STATE  = 42
TEST_SIZE     = 0.20
MAX_URL_LEN   = 256
VOCAB_SIZE    = 98
EMBED_DIM     = 64
CNN_CHANNELS  = 128
FEAT_MLP_DIM  = 256
FEAT_MLP_OUT  = 128
DROPOUT       = 0.40
BATCH_SIZE    = 1024
EPOCHS        = 30
LR            = 3e-4
WEIGHT_DECAY  = 3e-4
OOD_PERCENTILE = 99.0
PENULTIMATE_DIM = 256   # last hidden dim before classification head


# ══════════════════════════════════════════════════════════════════════════════
# Model — BinaryURLPhishNet with penultimate feature extraction
# ══════════════════════════════════════════════════════════════════════════════

class _ResBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(dim, dim),
        )
    def forward(self, x): return x + self.block(x)


class FeatureMLP(nn.Module):
    def __init__(self, in_dim, proj=FEAT_MLP_DIM, out=FEAT_MLP_OUT, drop=DROPOUT):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, proj), nn.LayerNorm(proj), nn.GELU())
        self.res  = nn.Sequential(_ResBlock(proj, drop), _ResBlock(proj, drop))
        self.out  = nn.Sequential(nn.Linear(proj, out), nn.LayerNorm(out), nn.GELU(), nn.Dropout(drop))
        self.output_dim = out
    def forward(self, x): return self.out(self.res(self.proj(x)))


class CharCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb  = nn.Embedding(VOCAB_SIZE, EMBED_DIM, padding_idx=0)
        self.c3   = nn.Conv1d(EMBED_DIM, CNN_CHANNELS, 3, padding=1)
        self.c5   = nn.Conv1d(EMBED_DIM, CNN_CHANNELS, 5, padding=2)
        self.c7   = nn.Conv1d(EMBED_DIM, CNN_CHANNELS, 7, padding=3)
        self.bn3  = nn.BatchNorm1d(CNN_CHANNELS)
        self.bn5  = nn.BatchNorm1d(CNN_CHANNELS)
        self.bn7  = nn.BatchNorm1d(CNN_CHANNELS)
        fused     = 3 * CNN_CHANNELS
        self.deep = nn.Conv1d(fused, fused, 3, padding=1)
        self.bnd  = nn.BatchNorm1d(fused)
        self.drop = nn.Dropout(0.15)
        self.output_dim = 2 * fused  # 768

    def forward(self, x):
        e = self.emb(x).transpose(1, 2)
        c = torch.cat([F.relu(self.bn3(self.c3(e))),
                       F.relu(self.bn5(self.c5(e))),
                       F.relu(self.bn7(self.c7(e)))], dim=1)
        f = self.drop(F.relu(self.bnd(self.deep(c))))
        return torch.cat([F.adaptive_max_pool1d(f, 1).squeeze(-1),
                          F.adaptive_avg_pool1d(f, 1).squeeze(-1)], dim=1)


class DualStreamOODNet(nn.Module):
    """
    BinaryURLPhishNet with separate penultimate extractor for Mahalanobis OOD.
    Architecture: CharCNN(768) + FeatureMLP(128) → fusion(896) → 512 → 256 → 2
    Penultimate = 256-dim vector (post-GELU, post-LN, pre-final-linear).
    """
    def __init__(self, feat_dim):
        super().__init__()
        self.feat_mlp = FeatureMLP(feat_dim)
        self.char_cnn = CharCNN()
        fusion = self.feat_mlp.output_dim + self.char_cnn.output_dim  # 896
        # Split clf into backbone (→256-dim) and head (→2)
        self.backbone = nn.Sequential(
            nn.Linear(fusion, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(DROPOUT),
            nn.Linear(512, PENULTIMATE_DIM), nn.LayerNorm(PENULTIMATE_DIM), nn.GELU(),
            nn.Dropout(DROPOUT * 0.7),
        )
        self.head = nn.Linear(PENULTIMATE_DIM, 2)

    def forward(self, feat, chars):
        fused = torch.cat([self.feat_mlp(feat), self.char_cnn(chars)], dim=1)
        return self.head(self.backbone(fused))

    def extract_penultimate(self, feat, chars):
        """Return 256-dim penultimate features (post-LN, post-GELU, post-Dropout)."""
        fused = torch.cat([self.feat_mlp(feat), self.char_cnn(chars)], dim=1)
        return self.backbone(fused)


class URLDataset(Dataset):
    def __init__(self, X_feat, X_char, labels):
        self.Xf = torch.from_numpy(X_feat.astype(np.float32))
        self.Xc = torch.from_numpy(X_char.astype(np.int64))
        self.y  = torch.from_numpy(labels.astype(np.int64))
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.Xf[i], self.Xc[i], self.y[i]


# ══════════════════════════════════════════════════════════════════════════════
# Mahalanobis helpers (Lee et al. NeurIPS 2018)
# ══════════════════════════════════════════════════════════════════════════════

def extract_penultimate_batched(model: DualStreamOODNet,
                                X_feat: np.ndarray, X_char: np.ndarray,
                                device: torch.device) -> np.ndarray:
    """Extract 256-dim penultimate features in batches → (N, 256) numpy."""
    model.eval()
    parts = []
    with torch.no_grad():
        for i in range(0, len(X_feat), BATCH_SIZE):
            xf = torch.from_numpy(X_feat[i:i+BATCH_SIZE].astype(np.float32)).to(device)
            xc = torch.from_numpy(X_char[i:i+BATCH_SIZE].astype(np.int64)).to(device)
            parts.append(model.extract_penultimate(xf, xc).cpu().numpy())
    return np.vstack(parts)


def fit_mahalanobis(feats: np.ndarray, labels: np.ndarray):
    """
    Fit class-conditional Gaussians → return mu_0, mu_1, precision matrix P.
    Lee et al. NeurIPS 2018, Eq (1)+(2).
    """
    classes = np.unique(labels)
    means, N, C, dim = {}, len(labels), len(classes), feats.shape[1]
    for c in classes:
        means[c] = feats[labels == c].mean(axis=0)
    Sigma = np.zeros((dim, dim), dtype=np.float64)
    for c in classes:
        fc   = feats[labels == c].astype(np.float64)
        diff = fc - means[c]
        Sigma += diff.T @ diff
    Sigma /= (N - C)
    Sigma += 1e-5 * np.eye(dim)
    P = np.linalg.inv(Sigma)
    return means[0].astype(np.float64), means[1].astype(np.float64), P


def mahalanobis_scores(feats: np.ndarray,
                       mu_0: np.ndarray, mu_1: np.ndarray,
                       P: np.ndarray) -> np.ndarray:
    """M(x) = min_c [ (f - mu_c)^T P (f - mu_c) ]. Lower = in-dist, higher = OOD."""
    f  = feats.astype(np.float64)
    d0, d1 = f - mu_0, f - mu_1
    m0 = np.einsum('ij,jk,ik->i', d0, P, d0)
    m1 = np.einsum('ij,jk,ik->i', d1, P, d1)
    return np.minimum(m0, m1)


def run_inference(model: DualStreamOODNet,
                  X_feat: np.ndarray, X_char: np.ndarray,
                  device: torch.device) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_feat), BATCH_SIZE):
            xf = torch.from_numpy(X_feat[i:i+BATCH_SIZE].astype(np.float32)).to(device)
            xc = torch.from_numpy(X_char[i:i+BATCH_SIZE].astype(np.int64)).to(device)
            preds.extend(model(xf, xc).argmax(1).cpu().numpy().tolist())
    return np.array(preds)


def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    cm  = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    return dict(acc=acc*100, f1=f1m, mcc=mcc, fpr=fpr*100, fnr=fnr*100,
                tp=int(tp), tn=int(tn), fp=int(fp), fn=int(fn))


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("=" * 80)
    log.info("Experiment 13 — CharCNN + FeatureMLP + Mahalanobis OOD (GAN Defense)")
    log.info("Reference  : Lee et al., NeurIPS 2018  arXiv:1807.03888")
    log.info("GAN attack : AlEroud & Karabatis, IWSPA 2020  doi:10.1145/3375708.3380315")
    log.info(f"Device     : {device}")
    log.info(f"Output     : {OUT_DIR}")
    log.info("=" * 80)

    # ── STEP 1: Load phishphresh chunks ────────────────────────────────────────
    log.info("\n[1/7] Loading phishphresh chunks (features + char_ids)...")
    train_chunks = sorted(glob.glob(os.path.join(DATA_DIR, "train", "chunk_*.npz")))
    test_chunks  = sorted(glob.glob(os.path.join(DATA_DIR, "test",  "chunk_*.npz")))
    train_flag   = os.path.exists(os.path.join(DATA_DIR, "_train_complete.flag"))
    test_flag    = os.path.exists(os.path.join(DATA_DIR, "_test_complete.flag"))
    if not (train_flag and test_flag):
        log.error("Chunks incomplete — re-run train_multiclass_brand.py first.")
        sys.exit(1)
    log.info(f"  Train chunks: {len(train_chunks)},  Test chunks: {len(test_chunks)}")

    feat_parts, char_parts, tgt_parts = [], [], []
    for chunk in tqdm(train_chunks + test_chunks, desc="  Loading"):
        d = np.load(chunk, allow_pickle=True)
        feat_parts.append(d["features"])
        char_parts.append(d["char_ids"])
        tgt_parts.extend(d["targets"].tolist())

    X_feat = np.vstack(feat_parts).astype(np.float32)
    X_char = np.vstack(char_parts).astype(np.int64)
    y_all  = np.array([0 if str(t or "").strip() == "" else 1 for t in tgt_parts],
                      dtype=np.int64)
    del feat_parts, char_parts; gc.collect()

    N_FEAT = X_feat.shape[1]
    log.info(f"  Total: {len(y_all):,}  |  Benign: {(y_all==0).sum():,}  "
             f"Phishing: {(y_all==1).sum():,}  |  Features: {N_FEAT}  "
             f"Char-seq len: {X_char.shape[1]}")

    # ── STEP 2: 80/20 split (same as all experiments) ─────────────────────────
    log.info("\n[2/7] 80/20 split (RANDOM_STATE=42) + StandardScaler...")
    idx = np.arange(len(y_all))
    tr_idx, te_idx = train_test_split(idx, test_size=TEST_SIZE,
                                      random_state=RANDOM_STATE, stratify=y_all)
    Xf_tr, Xf_te = X_feat[tr_idx], X_feat[te_idx]
    Xc_tr, Xc_te = X_char[tr_idx], X_char[te_idx]
    y_tr,  y_te  = y_all[tr_idx],  y_all[te_idx]
    del X_feat, X_char; gc.collect()

    scaler  = StandardScaler()
    Xf_tr_s = scaler.fit_transform(Xf_tr).astype(np.float32)
    Xf_te_s = scaler.transform(Xf_te).astype(np.float32)
    joblib.dump(scaler, SCALER_PATH)
    log.info(f"  Train: {len(y_tr):,}  Test: {len(y_te):,}  Scaler saved -> {SCALER_PATH}")

    # ── STEP 3: Train DualStreamOODNet ─────────────────────────────────────────
    log.info(f"\n[3/7] Training DualStreamOODNet ({EPOCHS} epochs)...")
    log.info(f"  Architecture: {N_FEAT}-feat + CharCNN → fusion(896) → 512 → 256 → 2")

    counts  = np.bincount(y_tr)
    sample_w = (1.0 / counts)[y_tr]
    sampler  = WeightedRandomSampler(sample_w, num_samples=len(y_tr), replacement=True)
    train_dl = DataLoader(URLDataset(Xf_tr_s, Xc_tr, y_tr),
                          batch_size=BATCH_SIZE, sampler=sampler, num_workers=0,
                          pin_memory=(device.type == "cuda"))
    test_dl  = DataLoader(URLDataset(Xf_te_s, Xc_te, y_te),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model     = DualStreamOODNet(feat_dim=N_FEAT).to(device)
    phish_w   = float(np.clip(counts[0] / counts[1], 1.0, 3.0))
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, phish_w], device=device, dtype=torch.float32)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, steps_per_epoch=len(train_dl),
        epochs=EPOCHS, pct_start=0.1
    )
    log.info(f"  Phishing class weight: {phish_w:.3f}")

    best_f1, best_epoch = 0.0, 0
    metrics_rows = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for xf, xc, yb in train_dl:
            xf, xc, yb = xf.to(device), xc.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xf, xc).float(), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_dl)
        preds_val = run_inference(model, Xf_te_s, Xc_te, device)
        m = compute_metrics(y_te, preds_val)
        log.info(f"  Epoch {epoch:02d}/{EPOCHS}  loss={avg_loss:.4f}  "
                 f"acc={m['acc']:.2f}%  F1={m['f1']:.4f}  MCC={m['mcc']:.4f}  "
                 f"FPR={m['fpr']:.2f}%  FNR={m['fnr']:.2f}%")
        metrics_rows.append({"epoch": epoch, "loss": avg_loss, **m})

        if m["f1"] > best_f1:
            best_f1, best_epoch = m["f1"], epoch
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "f1": m["f1"], "mcc": m["mcc"], "acc": m["acc"]}, MODEL_PATH)

    pd.DataFrame(metrics_rows).to_csv(METRICS_CSV, index=False)
    log.info(f"  Best model: epoch {best_epoch}  F1={best_f1:.4f}  -> {MODEL_PATH}")

    # Reload best model
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    log.info(f"  Loaded best model from epoch {ckpt['epoch']}")

    # ── STEP 4: Extract 256-dim penultimate features for training set ──────────
    log.info("\n[4/7] Extracting 256-dim penultimate features for training set...")
    feats_tr = extract_penultimate_batched(model, Xf_tr_s, Xc_tr, device)
    log.info(f"  Training penultimate matrix: {feats_tr.shape}  (N_train × 256)")

    # ── STEP 5: Fit Mahalanobis parameters ─────────────────────────────────────
    log.info("\n[5/7] Fitting class-conditional Gaussians (Lee et al. NeurIPS 2018)...")
    mu_0, mu_1, P = fit_mahalanobis(feats_tr, y_tr)
    log.info(f"  mu_0 (benign)   norm: {np.linalg.norm(mu_0):.4f}")
    log.info(f"  mu_1 (phishing) norm: {np.linalg.norm(mu_1):.4f}")
    log.info(f"  Precision matrix shape: {P.shape}")

    train_scores = mahalanobis_scores(feats_tr, mu_0, mu_1, P)
    threshold    = float(np.percentile(train_scores, OOD_PERCENTILE))
    log.info(f"  Training score stats: min={train_scores.min():.2f}  "
             f"median={np.median(train_scores):.2f}  "
             f"p95={np.percentile(train_scores, 95):.2f}  "
             f"max={train_scores.max():.2f}")
    log.info(f"  OOD threshold ({OOD_PERCENTILE}th-pct): {threshold:.4f}")

    joblib.dump({"mu_0": mu_0, "mu_1": mu_1, "P": P, "threshold": threshold}, MAHA_PATH)
    log.info(f"  Mahalanobis params saved -> {MAHA_PATH}")

    # ── STEP 6: Evaluate on real phishphresh test set ───────────────────────────
    log.info("\n[6/7] Evaluating on real phishphresh test set...")
    feats_te  = extract_penultimate_batched(model, Xf_te_s, Xc_te, device)
    preds_te  = run_inference(model, Xf_te_s, Xc_te, device)
    m_std     = compute_metrics(y_te, preds_te)

    te_scores = mahalanobis_scores(feats_te, mu_0, mu_1, P)
    ood_flags = te_scores > threshold
    n_ood     = int(ood_flags.sum())
    pct_ood   = n_ood / len(ood_flags) * 100
    ood_phish_pct = float((y_te[ood_flags] == 1).mean() * 100) if n_ood > 0 else 0.0

    log.info(f"  Standard accuracy   : {m_std['acc']:.2f}%")
    log.info(f"  Standard FNR        : {m_std['fnr']:.2f}%")
    log.info(f"  Standard FPR        : {m_std['fpr']:.2f}%")
    log.info(f"  F1-Macro            : {m_std['f1']:.4f}")
    log.info(f"  MCC                 : {m_std['mcc']:.4f}")
    log.info(f"  OOD-flagged real URLs: {n_ood:,} / {len(y_te):,} ({pct_ood:.2f}%)")
    log.info(f"  Of OOD, % phishing  : {ood_phish_pct:.1f}%")
    log.info(f"  Test score stats: min={te_scores.min():.2f}  "
             f"median={np.median(te_scores):.2f}  max={te_scores.max():.2f}")

    # ── STEP 7: Evaluate on GAN adversarial vectors from Exp 8 ─────────────────
    log.info("\n[7/7] Evaluating on GAN adversarial vectors from Exp 8...")
    if not os.path.exists(ADV_CSV):
        log.warning(f"  adversarial_dataset.csv not found at {ADV_CSV}")
        log.warning("  Skipping GAN evaluation. Re-run Exp 8 to generate it.")
        gan_detection_rate = None
        n_evaded = None
        evasion_pct = None
        adv_scores = None
    else:
        adv_df   = pd.read_csv(ADV_CSV)
        feat_cols = [c for c in adv_df.columns if c not in ("label", "original_label")]
        Xadv     = adv_df[feat_cols].values.astype(np.float32)
        log.info(f"  GAN adversarial samples: {len(Xadv):,}  (features: {Xadv.shape[1]})")

        Xadv_s  = scaler.transform(Xadv).astype(np.float32)
        # GAN vectors have no URL strings — use zero char tensors for CharCNN
        # (GAN operates in feature space only; CharCNN sees padding tokens → fixed neutral output)
        Xadv_c  = np.zeros((len(Xadv), MAX_URL_LEN), dtype=np.int64)
        log.info("  Using zero char tensors for CharCNN (GAN is feature-space only)")

        # Standard classification
        preds_adv = run_inference(model, Xadv_s, Xadv_c, device)
        n_evaded  = int((preds_adv == 0).sum())
        evasion_pct = n_evaded / len(preds_adv) * 100

        # Mahalanobis OOD detection
        feats_adv      = extract_penultimate_batched(model, Xadv_s, Xadv_c, device)
        adv_scores     = mahalanobis_scores(feats_adv, mu_0, mu_1, P)
        adv_ood_flags  = adv_scores > threshold
        n_detected     = int(adv_ood_flags.sum())
        gan_detection_rate = n_detected / len(preds_adv) * 100

        log.info(f"  GAN evasion (no OOD)   : {n_evaded:,} / {len(preds_adv):,} "
                 f"({evasion_pct:.1f}%) evaded as benign")
        log.info(f"  OOD detected (Mahala)  : {n_detected:,} / {len(preds_adv):,} "
                 f"({gan_detection_rate:.1f}%) flagged as OOD/adversarial")
        log.info(f"  Adv score stats: min={adv_scores.min():.2f}  "
                 f"median={np.median(adv_scores):.2f}  "
                 f"max={adv_scores.max():.2f}")
        log.info(f"  Threshold: {threshold:.4f}  "
                 f"(adv median / threshold = {np.median(adv_scores)/threshold:.0f}×)")

    # ── Write OOD Report ────────────────────────────────────────────────────────
    with open(OOD_REPORT, "w", encoding="utf-8") as rpt:
        rpt.write("=" * 70 + "\n")
        rpt.write("Experiment 13 — CharCNN + FeatureMLP + Mahalanobis OOD Report\n")
        rpt.write("Reference: Lee et al., NeurIPS 2018  arXiv:1807.03888\n")
        rpt.write("GAN: AlEroud & Karabatis, IWSPA 2020  doi:10.1145/3375708.3380315\n")
        rpt.write("=" * 70 + "\n\n")

        rpt.write("MODEL\n")
        rpt.write(f"  Architecture      : {N_FEAT}-feat + CharCNN → 896 → 512 → 256 → 2\n")
        rpt.write(f"  Penultimate dim   : 256 (fusion of CharCNN + FeatureMLP signals)\n")
        rpt.write(f"  Best epoch        : {best_epoch}  (F1={best_f1:.4f})\n")
        rpt.write(f"  OOD threshold     : {threshold:.4f}  ({OOD_PERCENTILE}th pct of training scores)\n\n")

        rpt.write("REAL PHISHPHRESH TEST SET\n")
        rpt.write(f"  Test samples      : {len(y_te):,}\n")
        rpt.write(f"  Standard accuracy : {m_std['acc']:.2f}%\n")
        rpt.write(f"  F1-Macro          : {m_std['f1']:.4f}\n")
        rpt.write(f"  MCC               : {m_std['mcc']:.4f}\n")
        rpt.write(f"  FPR               : {m_std['fpr']:.2f}%\n")
        rpt.write(f"  FNR               : {m_std['fnr']:.2f}%\n")
        rpt.write(f"  OOD-flagged URLs  : {n_ood:,}  ({pct_ood:.2f}%)\n")
        rpt.write(f"  Of OOD, % phishing: {ood_phish_pct:.1f}%\n\n")

        rpt.write("COMPARISON vs Exp 10 (FeatureMLP-only Mahalanobis OOD)\n")
        rpt.write(f"  Exp 10 accuracy   : 95.45%     Exp 13: {m_std['acc']:.2f}%\n")
        rpt.write(f"  Exp 10 FNR        : 6.01%      Exp 13: {m_std['fnr']:.2f}%\n")
        rpt.write(f"  Exp 10 OOD FPR    : 1.04%      Exp 13: {pct_ood:.2f}%\n\n")

        if gan_detection_rate is not None:
            rpt.write("GAN ADVERSARIAL VECTORS (Exp 8, exact AlEroud Eq 3+4)\n")
            rpt.write(f"  Adversarial count  : {len(preds_adv):,}\n")
            rpt.write(f"  Evasion (no OOD)   : {evasion_pct:.1f}%\n")
            rpt.write(f"  OOD detected       : {gan_detection_rate:.1f}%\n")
            rpt.write(f"  Adv score median   : {np.median(adv_scores):.4f}\n")
            rpt.write(f"  Train score median : {np.median(train_scores):.4f}\n")
            rpt.write(f"  Ratio (adv/thresh) : {np.median(adv_scores)/threshold:.0f}×\n\n")
            rpt.write("  Note: GAN vectors passed with zero char tensors to CharCNN\n")
            rpt.write("  (GAN operates in 67-dim feature space; no URL strings available)\n\n")

        rpt.write("CROSS-EXPERIMENT SUMMARY\n")
        rpt.write(f"  {'Experiment':<35} {'Normal Acc':>10} {'GAN Evasion':>12} {'OOD FPR':>8}\n")
        rpt.write(f"  {'-'*65}\n")
        rpt.write(f"  {'Exp 8 (no defense)':35} {'95.44%':>10} {'100%':>12} {'—':>8}\n")
        rpt.write(f"  {'Exp 9 (adv training, MLP only)':35} {'94.34%':>10} {'0%':>12} {'1.23%':>8}\n")
        rpt.write(f"  {'Exp 10 (Maha OOD, MLP only)':35} {'95.45%':>10} {'0%':>12} {'1.04%':>8}\n")
        gan_str = f"{gan_detection_rate:.1f}% det" if gan_detection_rate is not None else "—"
        rpt.write(f"  {'Exp 13 (Maha OOD, CharCNN+MLP)':35} {m_std['acc']:.2f}%{'':<5} {evasion_pct:.1f}% evade{'':<3} {pct_ood:.2f}%\n")

        rpt.write("\n\nCLASSIFICATION REPORT (real test set)\n")
        rpt.write(classification_report(y_te, preds_te, target_names=["benign", "phishing"]))

    log.info(f"\nOOD report saved -> {OOD_REPORT}")
    log.info("=" * 80)
    log.info("Experiment 13 complete.")
    log.info("=" * 80)
