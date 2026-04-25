#!/usr/bin/env python3
"""
train_mahalanobis_ood.py
========================
Experiment 10 — Mahalanobis OOD Detection as a GAN Defense.

Reference: Lee et al., NeurIPS 2018
  "A Simple Unified Framework for Detecting Out-of-Distribution Samples
   and Adversarial Attacks"
  arXiv: https://arxiv.org/abs/1807.03888
  NeurIPS: https://proceedings.neurips.cc/paper/2018/hash/abdeb6f575ac5c6676b747bca8d09cc2-Abstract.html

Idea:
  Train a FeatureMLP classifier. After training, fit class-conditional Gaussians
  on the penultimate layer (64-dim) representations.  At inference, compute the
  minimum Mahalanobis distance to any class mean (using a pooled precision matrix).
  Samples with distance above the 99th-percentile training threshold are flagged
  as OOD / adversarial — without ever seeing GAN data during training.

Flow:
  1. Load phishphresh chunks + 80/20 split (RANDOM_STATE=42, same as all exps)
  2. Train FeatureMLP (67 -> [256, 128, 64] -> 2) on scaled continuous features
  3. Extract 64-dim penultimate features for all training samples
  4. Fit class-conditional Gaussians:
       mu_0  = mean of penultimate features for benign class
       mu_1  = mean of penultimate features for phishing class
       Sigma = pooled (tied) covariance + 1e-5 * I (regularisation)
       P     = inv(Sigma)  (precision matrix)
  5. Calibrate OOD threshold = 99th-percentile of training Mahalanobis scores
  6. Evaluate on real test set:
       - Standard classification accuracy (without OOD layer)
       - With OOD layer: samples above threshold flagged as adversarial/rejected
  7. Evaluate on GAN adversarial vectors from Exp 8
       (binary_phishing_adv_gan/logs/adversarial_dataset.csv)
       Goal: high detection rate with no GAN data in training

Output:
  binary_mahalanobis_ood/models/feature_mlp.pth
  binary_mahalanobis_ood/models/scaler.pkl
  binary_mahalanobis_ood/models/mahalanobis_params.pkl   (mu_0, mu_1, P, threshold)
  binary_mahalanobis_ood/logs/run.log
  binary_mahalanobis_ood/logs/ood_report.txt
  binary_mahalanobis_ood/logs/training_metrics.csv
"""

from __future__ import annotations
import gc, glob, logging, os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR      = os.path.join(PROJECT_DIR, "phishphresh", "multiclass_brand", "data")
ADV_CSV       = os.path.join(PROJECT_DIR, "binary_phishing_adv_gan", "logs", "adversarial_dataset.csv")
OUT_DIR       = os.path.join(PROJECT_DIR, "binary_mahalanobis_ood")
OUT_MODELS    = os.path.join(OUT_DIR, "models")
OUT_LOGS      = os.path.join(OUT_DIR, "logs")

MODEL_PATH    = os.path.join(OUT_MODELS, "feature_mlp.pth")
SCALER_PATH   = os.path.join(OUT_MODELS, "scaler.pkl")
MAHA_PATH     = os.path.join(OUT_MODELS, "mahalanobis_params.pkl")
OOD_REPORT    = os.path.join(OUT_LOGS,   "ood_report.txt")
METRICS_CSV   = os.path.join(OUT_LOGS,   "training_metrics.csv")
STDOUT_LOG    = os.path.join(OUT_LOGS,   "stdout.log")

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
RANDOM_STATE = 42
TEST_SIZE    = 0.20
BATCH_SIZE   = 512
EPOCHS       = 20
LR           = 3e-4
HIDDEN       = [256, 128, 64]
DROPOUT      = 0.3
OOD_PERCENTILE = 99.0   # threshold: flag top 1% of training as OOD calibration


# ══════════════════════════════════════════════════════════════════════════════
# Model with penultimate feature extractor
# ══════════════════════════════════════════════════════════════════════════════

class FeatureMLP(nn.Module):
    """
    Same architecture as Exp 8 FeatureMLPPD (67 -> [256,128,64] -> 2).
    Extra method: extract_penultimate() returns 64-dim pre-logit features.
    These 64-dim vectors are used for Mahalanobis OOD fitting.
    """
    def __init__(self, in_dim: int, hidden: list = HIDDEN, dropout: float = DROPOUT):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.head     = nn.Linear(prev, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

    def extract_penultimate(self, x: torch.Tensor) -> torch.Tensor:
        """Return 64-dim penultimate features (post-BN, post-ReLU, post-Dropout)."""
        return self.backbone(x)


class FeatDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
    def __len__(self):        return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


# ══════════════════════════════════════════════════════════════════════════════
# Mahalanobis helpers (Lee et al. NeurIPS 2018)
# ══════════════════════════════════════════════════════════════════════════════

def extract_features_batched(model: FeatureMLP, X: np.ndarray,
                              device: torch.device) -> np.ndarray:
    """Run backbone over X in batches, return (N, 64) numpy array."""
    model.eval()
    parts = []
    with torch.no_grad():
        for i in range(0, len(X), BATCH_SIZE):
            xb = torch.from_numpy(X[i:i+BATCH_SIZE]).float().to(device)
            parts.append(model.extract_penultimate(xb).cpu().numpy())
    return np.vstack(parts)


def fit_mahalanobis(feats: np.ndarray, labels: np.ndarray):
    """
    Fit class-conditional Gaussians on penultimate features.
    Returns: mu_0, mu_1, precision_matrix P = inv(pooled_Sigma + reg*I)
    Lee et al. 2018, Eq. (1)+(2).
    """
    classes = np.unique(labels)
    means   = {}
    N       = len(labels)
    C       = len(classes)
    dim     = feats.shape[1]

    # Class means
    for c in classes:
        means[c] = feats[labels == c].mean(axis=0)

    # Pooled (tied) covariance
    Sigma = np.zeros((dim, dim), dtype=np.float64)
    for c in classes:
        fc   = feats[labels == c].astype(np.float64)
        diff = fc - means[c]
        Sigma += diff.T @ diff
    Sigma /= (N - C)

    # Regularise
    Sigma += 1e-5 * np.eye(dim)

    # Precision matrix
    P = np.linalg.inv(Sigma)
    return means[0].astype(np.float64), means[1].astype(np.float64), P


def mahalanobis_scores(feats: np.ndarray,
                       mu_0: np.ndarray, mu_1: np.ndarray,
                       P: np.ndarray) -> np.ndarray:
    """
    Compute min-class Mahalanobis distance for each sample.
    M(x) = min_c [ (f - mu_c)^T P (f - mu_c) ]
    Lower  = in-distribution. Higher = OOD / adversarial.
    """
    f  = feats.astype(np.float64)
    d0 = f - mu_0
    d1 = f - mu_1
    m0 = np.einsum('ij,jk,ik->i', d0, P, d0)   # (N,)
    m1 = np.einsum('ij,jk,ik->i', d1, P, d1)   # (N,)
    return np.minimum(m0, m1)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("=" * 80)
    log.info("Experiment 10 — Mahalanobis OOD Detection (GAN Defense)")
    log.info("Reference  : Lee et al., NeurIPS 2018  arXiv:1807.03888")
    log.info(f"Device     : {device}")
    log.info(f"Output     : {OUT_DIR}")
    log.info("=" * 80)

    # ── STEP 1: Load phishphresh chunks ────────────────────────────────────────
    log.info("\n[1/7] Loading phishphresh chunks...")
    train_chunks = sorted(glob.glob(os.path.join(DATA_DIR, "train", "chunk_*.npz")))
    test_chunks  = sorted(glob.glob(os.path.join(DATA_DIR, "test",  "chunk_*.npz")))
    train_flag   = os.path.exists(os.path.join(DATA_DIR, "_train_complete.flag"))
    test_flag    = os.path.exists(os.path.join(DATA_DIR, "_test_complete.flag"))
    if not (train_flag and test_flag):
        log.error("Chunks incomplete — re-run train_multiclass_brand.py first.")
        sys.exit(1)
    log.info(f"  Train chunks : {len(train_chunks)}, Test chunks : {len(test_chunks)}")

    feat_parts, tgt_parts = [], []
    for chunk in tqdm(train_chunks + test_chunks, desc="  Loading"):
        d = np.load(chunk, allow_pickle=True)
        feat_parts.append(d["features"])
        tgt_parts.extend(d["targets"].tolist())

    X_all = np.vstack(feat_parts).astype(np.float32)
    y_all = np.array(
        [0 if str(t or "").strip() == "" else 1 for t in tgt_parts],
        dtype=np.int64
    )
    del feat_parts; gc.collect()

    N_FEAT = X_all.shape[1]
    log.info(f"  Total    : {len(y_all):,}  |  Benign: {int((y_all==0).sum()):,}  "
             f"Phishing: {int((y_all==1).sum()):,}  |  Features: {N_FEAT}")

    # ── STEP 2: 80/20 split + scale ────────────────────────────────────────────
    log.info("\n[2/7] 80/20 split (RANDOM_STATE=42) + StandardScaler...")
    idx            = np.arange(len(y_all))
    tr_idx, te_idx = train_test_split(idx, test_size=TEST_SIZE,
                                      random_state=RANDOM_STATE, stratify=y_all)
    Xf_tr, Xf_te = X_all[tr_idx], X_all[te_idx]
    y_tr,  y_te  = y_all[tr_idx], y_all[te_idx]
    del X_all; gc.collect()

    scaler   = StandardScaler()
    Xf_tr_s  = scaler.fit_transform(Xf_tr).astype(np.float32)
    Xf_te_s  = scaler.transform(Xf_te).astype(np.float32)
    joblib.dump(scaler, SCALER_PATH)
    log.info(f"  Train: {len(y_tr):,}  Test: {len(y_te):,}  Scaler saved -> {SCALER_PATH}")

    # ── STEP 3: Train FeatureMLP ────────────────────────────────────────────────
    log.info(f"\n[3/7] Training FeatureMLP {N_FEAT} -> {HIDDEN} -> 2  ({EPOCHS} epochs)...")

    counts_tr = np.bincount(y_tr)
    sw        = (1.0 / counts_tr)[y_tr]
    sampler   = WeightedRandomSampler(sw, num_samples=len(y_tr), replacement=True)
    train_dl  = DataLoader(FeatDataset(Xf_tr_s, y_tr), batch_size=BATCH_SIZE,
                           sampler=sampler, num_workers=0, pin_memory=(device.type=="cuda"))
    test_dl   = DataLoader(FeatDataset(Xf_te_s, y_te), batch_size=BATCH_SIZE,
                           shuffle=False, num_workers=0)

    model     = FeatureMLP(N_FEAT).to(device)
    criterion = nn.CrossEntropyLoss()
    optim     = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=3, factor=0.5)

    metrics_rows = []
    for epoch in range(1, EPOCHS + 1):
        # -- train --
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            logits = model(xb)
            loss   = criterion(logits, yb)
            loss.backward()
            optim.step()
            train_loss    += loss.item() * len(yb)
            train_correct += (logits.argmax(1) == yb).sum().item()
            train_total   += len(yb)

        # -- eval --
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in test_dl:
                xb, yb  = xb.to(device), yb.to(device)
                logits  = model(xb)
                val_loss    += criterion(logits, yb).item() * len(yb)
                val_correct += (logits.argmax(1) == yb).sum().item()
                val_total   += len(yb)

        tr_acc = train_correct / train_total * 100
        va_acc = val_correct   / val_total   * 100
        tr_l   = train_loss    / train_total
        va_l   = val_loss      / val_total
        scheduler.step(va_l)
        log.info(f"  Epoch {epoch:02d}/{EPOCHS}  "
                 f"train_loss={tr_l:.4f} acc={tr_acc:.2f}%  |  "
                 f"val_loss={va_l:.4f} acc={va_acc:.2f}%")
        metrics_rows.append({"epoch": epoch, "train_loss": tr_l, "train_acc": tr_acc,
                              "val_loss": va_l, "val_acc": va_acc})

    pd.DataFrame(metrics_rows).to_csv(METRICS_CSV, index=False)
    torch.save(model.state_dict(), MODEL_PATH)
    log.info(f"  Model saved -> {MODEL_PATH}")

    # ── STEP 4: Extract penultimate features ────────────────────────────────────
    log.info("\n[4/7] Extracting 64-dim penultimate features for all training samples...")
    feats_tr = extract_features_batched(model, Xf_tr_s, device)
    log.info(f"  Training feature matrix : {feats_tr.shape}  (N_train x 64)")

    # ── STEP 5: Fit Mahalanobis parameters ─────────────────────────────────────
    log.info("\n[5/7] Fitting class-conditional Gaussians (Lee et al. NeurIPS 2018)...")
    mu_0, mu_1, P = fit_mahalanobis(feats_tr, y_tr)
    log.info(f"  mu_0 (benign)   norm : {np.linalg.norm(mu_0):.4f}")
    log.info(f"  mu_1 (phishing) norm : {np.linalg.norm(mu_1):.4f}")
    log.info(f"  Precision matrix shape : {P.shape}")

    # Calibrate threshold on training data
    train_scores = mahalanobis_scores(feats_tr, mu_0, mu_1, P)
    threshold    = float(np.percentile(train_scores, OOD_PERCENTILE))
    log.info(f"  Training score stats : min={train_scores.min():.2f}  "
             f"median={np.median(train_scores):.2f}  "
             f"max={train_scores.max():.2f}")
    log.info(f"  OOD threshold ({OOD_PERCENTILE}th-pct) : {threshold:.4f}")

    # Save Mahalanobis params
    joblib.dump({"mu_0": mu_0, "mu_1": mu_1, "P": P, "threshold": threshold}, MAHA_PATH)
    log.info(f"  Mahalanobis params saved -> {MAHA_PATH}")

    # ── STEP 6: Evaluate on real phishphresh test set ───────────────────────────
    log.info("\n[6/7] Evaluating on real phishphresh test set...")
    feats_te = extract_features_batched(model, Xf_te_s, device)

    # Standard classification (no OOD)
    model.eval()
    preds_te = []
    with torch.no_grad():
        for i in range(0, len(Xf_te_s), BATCH_SIZE):
            xb     = torch.from_numpy(Xf_te_s[i:i+BATCH_SIZE]).float().to(device)
            preds_te.extend(model(xb).argmax(1).cpu().numpy().tolist())
    preds_te  = np.array(preds_te)
    std_acc   = accuracy_score(y_te, preds_te) * 100
    tn, fp, fn, tp = confusion_matrix(y_te, preds_te).ravel()
    std_fnr   = fn / (fn + tp) * 100

    # OOD layer: score test samples
    te_scores = mahalanobis_scores(feats_te, mu_0, mu_1, P)
    ood_flags = te_scores > threshold   # True = flagged as OOD
    n_ood_test = ood_flags.sum()
    pct_ood   = n_ood_test / len(ood_flags) * 100

    # Among OOD-flagged real test samples — what fraction are actually phishing?
    ood_phish_pct = (y_te[ood_flags] == 1).mean() * 100 if n_ood_test > 0 else 0.0

    log.info(f"  Standard accuracy      : {std_acc:.2f}%")
    log.info(f"  Standard FNR           : {std_fnr:.2f}%")
    log.info(f"  OOD-flagged real URLs  : {n_ood_test:,} / {len(y_te):,} ({pct_ood:.2f}%)")
    log.info(f"  Of those, % phishing   : {ood_phish_pct:.1f}%  "
             f"(high = OOD layer is alerting on borderline phishing)")

    # ── STEP 7: Evaluate on GAN adversarial vectors ─────────────────────────────
    log.info("\n[7/7] Evaluating on GAN adversarial vectors from Exp 8...")
    if not os.path.exists(ADV_CSV):
        log.warning(f"  adversarial_dataset.csv not found at {ADV_CSV}")
        log.warning("  Skipping GAN evaluation. Re-run Exp 8 to generate it.")
        gan_detection_rate = None
    else:
        adv_df = pd.read_csv(ADV_CSV)
        feat_cols = [c for c in adv_df.columns if c not in ("label", "original_label")]
        Xadv   = adv_df[feat_cols].values.astype(np.float32)
        log.info(f"  GAN adversarial samples : {len(Xadv):,}  (features: {Xadv.shape[1]})")

        # Scale using same scaler
        Xadv_s = scaler.transform(Xadv).astype(np.float32)

        # Extract penultimate features for adversarial samples
        feats_adv = extract_features_batched(model, Xadv_s, device)

        # Standard classifier prediction on adversarial vectors
        preds_adv = []
        with torch.no_grad():
            for i in range(0, len(Xadv_s), BATCH_SIZE):
                xb = torch.from_numpy(Xadv_s[i:i+BATCH_SIZE]).float().to(device)
                preds_adv.extend(model(xb).argmax(1).cpu().numpy().tolist())
        preds_adv   = np.array(preds_adv)
        n_evaded    = int((preds_adv == 0).sum())
        evasion_pct = n_evaded / len(preds_adv) * 100

        # Mahalanobis OOD detection on adversarial vectors
        adv_scores   = mahalanobis_scores(feats_adv, mu_0, mu_1, P)
        adv_ood_flags = adv_scores > threshold
        n_detected   = int(adv_ood_flags.sum())
        gan_detection_rate = n_detected / len(preds_adv) * 100

        log.info(f"  GAN evasion (no OOD)    : {n_evaded:,} / {len(preds_adv):,} "
                 f"({evasion_pct:.1f}%) evaded as benign")
        log.info(f"  OOD detected (Mahala)   : {n_detected:,} / {len(preds_adv):,} "
                 f"({gan_detection_rate:.1f}%) flagged as OOD/adversarial")
        log.info(f"  Adv score stats : min={adv_scores.min():.2f}  "
                 f"median={np.median(adv_scores):.2f}  max={adv_scores.max():.2f}")
        log.info(f"  Training threshold : {threshold:.4f}")

    # ── Write OOD Report ────────────────────────────────────────────────────────
    with open(OOD_REPORT, "w", encoding="utf-8") as rpt:
        rpt.write("=" * 60 + "\n")
        rpt.write("Experiment 10 — Mahalanobis OOD Detection Report\n")
        rpt.write("Reference: Lee et al., NeurIPS 2018  arXiv:1807.03888\n")
        rpt.write("=" * 60 + "\n\n")

        rpt.write("MODEL\n")
        rpt.write(f"  Architecture      : {N_FEAT} -> {HIDDEN} -> 2\n")
        rpt.write(f"  Penultimate dim   : 64\n")
        rpt.write(f"  OOD threshold     : {threshold:.4f}  ({OOD_PERCENTILE}th pct)\n\n")

        rpt.write("REAL PHISHPHRESH TEST SET\n")
        rpt.write(f"  Test samples      : {len(y_te):,}\n")
        rpt.write(f"  Standard accuracy : {std_acc:.2f}%\n")
        rpt.write(f"  Standard FNR      : {std_fnr:.2f}%\n")
        rpt.write(f"  OOD-flagged URLs  : {n_ood_test:,}  ({pct_ood:.2f}%)\n")
        rpt.write(f"  Of OOD, % phishing: {ood_phish_pct:.1f}%\n\n")

        if gan_detection_rate is not None:
            rpt.write("GAN ADVERSARIAL VECTORS (Exp 8)\n")
            rpt.write(f"  Adversarial count : {len(preds_adv):,}\n")
            rpt.write(f"  Evasion (no OOD)  : {evasion_pct:.1f}%\n")
            rpt.write(f"  OOD detected      : {gan_detection_rate:.1f}%\n")
            rpt.write(f"  Adv score median  : {np.median(adv_scores):.4f}\n")
            rpt.write(f"  Train score median: {np.median(train_scores):.4f}\n\n")

        rpt.write("CLASSIFICATION REPORT (real test)\n")
        rpt.write(classification_report(y_te, preds_te,
                                        target_names=["benign", "phishing"]))

    log.info(f"\nOOD report saved -> {OOD_REPORT}")
    log.info("=" * 80)
    log.info("Experiment 10 complete.")
    log.info("=" * 80)
