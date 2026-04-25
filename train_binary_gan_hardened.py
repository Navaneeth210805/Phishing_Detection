#!/usr/bin/env python3
"""
train_binary_gan_hardened.py
============================
Adversarial hardening of FeatureMLP against AlEroud 2020 GAN attacks.

Story:
  Exp 8 showed: GAN achieves 100% evasion on a standalone FeatureMLP PD.
  This experiment answers: if we inject those GAN adversarial vectors into
  training, does FeatureMLP learn to detect them?

Flow:
  1. Load phishphresh (real data) — same 80/20 split as Exp 8
  2. Load adversarial_dataset.csv (238,729 GAN-generated feature vectors)
  3. Train hardened FeatureMLP on real + adversarial combined data
  4. Evaluate:
       (a) Normal accuracy on phishphresh test split
       (b) GAN evasion rate: how many adversarial vectors still fool it?

Output dir: binary_gan_hardened/
"""

from __future__ import annotations
import gc, glob, logging, os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                              confusion_matrix, classification_report)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(PROJECT_DIR, "phishphresh", "multiclass_brand", "data")
ADV_CSV      = os.path.join(PROJECT_DIR, "binary_phishing_adv_gan", "logs", "adversarial_dataset.csv")

OUT_DIR      = os.path.join(PROJECT_DIR, "binary_gan_hardened")
OUT_MODELS   = os.path.join(OUT_DIR, "models")
OUT_LOGS     = os.path.join(OUT_DIR, "logs")

MODEL_PATH   = os.path.join(OUT_MODELS, "hardened_mlp.pth")
SCALER_PATH  = os.path.join(OUT_MODELS, "scaler.pkl")
STDOUT_LOG   = os.path.join(OUT_LOGS,   "stdout.log")

os.makedirs(OUT_MODELS, exist_ok=True)
os.makedirs(OUT_LOGS,   exist_ok=True)

# ── Tee stdout ─────────────────────────────────────────────────────────────────
class _Tee:
    def __init__(self, *files): self.files = files
    def write(self, obj):
        for f in self.files: f.write(obj); f.flush()
    def flush(self):
        for f in self.files: f.flush()

_stdout_fh = open(STDOUT_LOG, "w", encoding="utf-8")
sys.stdout  = _Tee(sys.__stdout__, _stdout_fh)

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


# ── Model (same as FeatureMLPPD in Exp 8) ─────────────────────────────────────
class FeatureMLPHardened(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in HIDDEN:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(DROPOUT)]
            prev = h
        layers.append(nn.Linear(prev, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x): return self.net(x)

    def predict(self, x_np, device):
        self.eval()
        with torch.no_grad():
            t = torch.from_numpy(x_np).float().to(device)
            out = []
            for i in range(0, len(t), BATCH_SIZE):
                out.extend(self.net(t[i:i+BATCH_SIZE]).argmax(1).cpu().numpy())
        return np.array(out)


class FeatDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
    def __len__(self):        return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log.info("=" * 80)
    log.info("Exp 9 — Hardened FeatureMLP (GAN Adversarial Training Defense)")
    log.info("Injects GAN adversarial vectors from Exp 8 into training.")
    log.info("Goal: does adversarial training close the 100% evasion gap?")
    log.info(f"Device : {device}")
    log.info(f"Output : {OUT_DIR}")
    log.info("=" * 80)

    # ── STEP 1: Load phishphresh chunks (features only) ────────────────────────
    log.info("\n[1/5] Loading phishphresh chunks...")
    train_chunks = sorted(glob.glob(os.path.join(DATA_DIR, "train", "chunk_*.npz")))
    test_chunks  = sorted(glob.glob(os.path.join(DATA_DIR, "test",  "chunk_*.npz")))

    feat_parts, tgt_parts = [], []
    for f in tqdm(train_chunks + test_chunks, desc="  Loading"):
        d = np.load(f, allow_pickle=True)
        feat_parts.append(d["features"])
        tgt_parts.extend(d["targets"].tolist())

    X_real = np.vstack(feat_parts).astype(np.float32)
    y_real = np.array(
        [0 if str(t or "").strip() == "" else 1 for t in tgt_parts], dtype=np.int64
    )
    del feat_parts; gc.collect()

    N_FEAT = X_real.shape[1]
    log.info(f"  Real data : {len(y_real):,}  "
             f"(benign={int((y_real==0).sum()):,}  phishing={int((y_real==1).sum()):,})")

    # 80/20 split — SAME random state as Exp 8 for fair comparison
    idx              = np.arange(len(y_real))
    tr_idx, te_idx   = train_test_split(idx, test_size=TEST_SIZE,
                                        random_state=RANDOM_STATE, stratify=y_real)
    Xf_tr, Xf_te = X_real[tr_idx], X_real[te_idx]
    y_tr,  y_te  = y_real[tr_idx], y_real[te_idx]
    del X_real; gc.collect()

    log.info(f"  Train : {len(y_tr):,}  Test : {len(y_te):,}")

    # ── STEP 2: Load GAN adversarial dataset ──────────────────────────────────
    log.info("\n[2/5] Loading GAN adversarial dataset (from Exp 8)...")
    df_adv   = pd.read_csv(ADV_CSV)
    feat_cols = [c for c in df_adv.columns if c != "label"]
    X_adv    = df_adv[feat_cols].values.astype(np.float32)
    y_adv    = df_adv["label"].values.astype(np.int64)
    log.info(f"  Adversarial vectors : {len(X_adv):,}  (all label=1/phishing)")
    log.info(f"  Feature dim         : {X_adv.shape[1]}")

    # ── STEP 3: Combine + scale ───────────────────────────────────────────────
    log.info("\n[3/5] Combining real train + adversarial vectors...")
    X_tr_combined = np.vstack([Xf_tr, X_adv])
    y_tr_combined = np.concatenate([y_tr, y_adv])

    real_phish  = int((y_tr == 1).sum())
    real_benign = int((y_tr == 0).sum())
    log.info(f"  Real    : {len(y_tr):,}  (benign={real_benign:,}  phishing={real_phish:,})")
    log.info(f"  GAN adv : {len(X_adv):,}  (all phishing)")
    log.info(f"  Combined: {len(y_tr_combined):,}  "
             f"(benign={int((y_tr_combined==0).sum()):,}  "
             f"phishing={int((y_tr_combined==1).sum()):,})")

    scaler    = StandardScaler()
    X_tr_s    = scaler.fit_transform(X_tr_combined).astype(np.float32)
    X_te_s    = scaler.transform(Xf_te).astype(np.float32)
    X_adv_s   = scaler.transform(X_adv).astype(np.float32)
    joblib.dump(scaler, SCALER_PATH)
    log.info(f"  Scaler saved")

    # ── STEP 4: Train hardened FeatureMLP ─────────────────────────────────────
    log.info(f"\n[4/5] Training hardened FeatureMLP ({EPOCHS} epochs)...")
    log.info(f"  Architecture : {N_FEAT} -> {HIDDEN} -> 2")
    log.info(f"  This model has SEEN GAN adversarial vectors during training.")

    counts = np.bincount(y_tr_combined)
    sw     = (1.0 / counts)[y_tr_combined]
    train_dl = DataLoader(FeatDataset(X_tr_s, y_tr_combined),
                          batch_size=BATCH_SIZE,
                          sampler=WeightedRandomSampler(sw, len(y_tr_combined), replacement=True),
                          num_workers=0, pin_memory=(device.type=="cuda"))
    test_dl  = DataLoader(FeatDataset(X_te_s, y_te),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model     = FeatureMLPHardened(in_dim=N_FEAT).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, float(counts[0]/counts[1])], device=device)
    )

    best_acc = 0.0
    rows = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for Xb, yb in train_dl:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            criterion(model(Xb), yb).backward()
            optimizer.step()

        model.eval()
        preds = []
        with torch.no_grad():
            for Xb, _ in test_dl:
                preds.extend(model(Xb.to(device)).argmax(1).cpu().numpy())
        preds = np.array(preds)
        acc   = accuracy_score(y_te, preds)
        f1    = f1_score(y_te, preds, average="macro")
        mcc   = matthews_corrcoef(y_te, preds)
        cm    = confusion_matrix(y_te, preds, labels=[0, 1])
        fpr   = cm[0,1] / max(cm[0].sum(), 1)
        fnr   = cm[1,0] / max(cm[1].sum(), 1)

        rows.append({"epoch": epoch, "acc": acc, "f1": f1, "mcc": mcc,
                     "fpr": fpr, "fnr": fnr})

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), MODEL_PATH)

        if epoch % 5 == 0 or epoch == EPOCHS:
            log.info(f"  Epoch {epoch:2d}/{EPOCHS}  "
                     f"acc={acc*100:.2f}%  F1={f1:.4f}  MCC={mcc:.4f}  "
                     f"FPR={fpr*100:.2f}%  FNR={fnr*100:.2f}%")

    pd.DataFrame(rows).to_csv(os.path.join(OUT_LOGS, "training_metrics.csv"), index=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    log.info(f"  Best accuracy : {best_acc*100:.2f}%")

    # ── STEP 5: Evaluate — normal + GAN evasion ───────────────────────────────
    log.info("\n[5/5] Evaluating hardened model...")

    # (a) Normal test accuracy
    normal_preds  = model.predict(X_te_s, device)
    normal_acc    = accuracy_score(y_te, normal_preds)
    normal_f1     = f1_score(y_te, normal_preds, average="macro")
    normal_mcc    = matthews_corrcoef(y_te, normal_preds)
    normal_cm     = confusion_matrix(y_te, normal_preds, labels=[0, 1])
    normal_fpr    = normal_cm[0,1] / max(normal_cm[0].sum(), 1)
    normal_fnr    = normal_cm[1,0] / max(normal_cm[1].sum(), 1)

    log.info(f"\n  Normal test results:")
    log.info(f"    Accuracy : {normal_acc*100:.2f}%")
    log.info(f"    F1-Macro : {normal_f1:.4f}")
    log.info(f"    MCC      : {normal_mcc:.4f}")
    log.info(f"    FPR      : {normal_fpr*100:.2f}%")
    log.info(f"    FNR      : {normal_fnr*100:.2f}%")

    # (b) GAN adversarial evasion (key question)
    adv_preds      = model.predict(X_adv_s, device)
    adv_detect     = adv_preds.mean()       # fraction correctly called phishing
    evasion_rate   = 1.0 - adv_detect       # fraction that still evades
    adv_cm         = confusion_matrix(
        np.ones(len(adv_preds), dtype=np.int64), adv_preds, labels=[0, 1]
    )

    log.info(f"\n  GAN adversarial evasion (KEY RESULT vs Exp 8):")
    log.info(f"    Adversarial vectors tested : {len(adv_preds):,}")
    log.info(f"    Correctly caught (phishing): {int(adv_detect*len(adv_preds)):,}  ({adv_detect*100:.2f}%)")
    log.info(f"    Still evading (benign pred): {int(evasion_rate*len(adv_preds)):,}  ({evasion_rate*100:.2f}%)")
    log.info(f"    Evasion rate BEFORE hardening (Exp 8): 100.00%")
    log.info(f"    Evasion rate AFTER  hardening (Exp 9): {evasion_rate*100:.2f}%")

    # Compare baseline (unhardened) vs hardened
    # Load unhardened model from Exp 8 if available for direct side-by-side
    pd_path = os.path.join(PROJECT_DIR, "binary_phishing_adv_gan", "models", "feature_mlp_pd.pth")
    if os.path.exists(pd_path):
        from train_binary_gan_adv import FeatureMLPPD
        pd_unhardened = FeatureMLPPD(in_dim=N_FEAT)
        pd_unhardened.load_state_dict(torch.load(pd_path, map_location=device, weights_only=True))
        pd_unhardened = pd_unhardened.to(device)
        # Unhardened was trained with its own scaler — need to re-scale with Exp8 scaler
        exp8_scaler_path = os.path.join(PROJECT_DIR, "binary_phishing_adv_gan", "models", "pd_scaler.pkl")
        exp8_scaler = joblib.load(exp8_scaler_path)
        X_adv_exp8_s = exp8_scaler.transform(X_adv).astype(np.float32)
        uh_preds = pd_unhardened.predict(X_adv_exp8_s, device)
        uh_evasion = 1.0 - uh_preds.mean()
        log.info(f"\n  Side-by-side comparison:")
        log.info(f"    Unhardened FeatureMLP (Exp 8) evasion: {uh_evasion*100:.2f}%")
        log.info(f"    Hardened   FeatureMLP (Exp 9) evasion: {evasion_rate*100:.2f}%")
        delta = uh_evasion - evasion_rate
        log.info(f"    Reduction in evasion                : -{delta*100:.2f}%")

    log.info("\n" + "=" * 80)
    log.info("SUMMARY — Exp 9 Hardened FeatureMLP")
    log.info("=" * 80)
    log.info(f"  Normal accuracy (test)   : {normal_acc*100:.2f}%  (Exp 8 baseline: 95.52%)")
    log.info(f"  F1-Macro                 : {normal_f1:.4f}")
    log.info(f"  MCC                      : {normal_mcc:.4f}")
    log.info(f"  FPR                      : {normal_fpr*100:.2f}%")
    log.info(f"  FNR                      : {normal_fnr*100:.2f}%")
    log.info(f"  GAN evasion BEFORE       : 100.00%  (Exp 8)")
    log.info(f"  GAN evasion AFTER        : {evasion_rate*100:.2f}%  (this exp)")
    log.info(f"  Adversarial dataset used : {ADV_CSV}")
    log.info("=" * 80)
    log.info("Done.")
    _stdout_fh.close()
