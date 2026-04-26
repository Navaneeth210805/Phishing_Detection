#!/usr/bin/env python3
"""
train_binary_gan_exact_paper.py
================================
Experiment 12 — GAN attack retrained with EXACT paper loss functions.

AlEroud & Karabatis IWSPA 2020  DOI:10.1145/3375708.3380315

DIFFERENCE FROM EXP 8:
  Exp 8 used nn.BCELoss (loss converges to 0 from above, positive values).
  This script uses the exact equations from the paper:
    - Eq 3: L_D = -E_{PD_Leg}[log D(f)] - E_{PD_phish}[log(1-D(f))]
    - Eq 4: L_G = -E[log D_wd(G_wg(m, s))]
  Where D(f) = P(legitimate | f)  — output 1 = legitimate, 0 = phishing.

  The training DISPLAY shows the raw reward signals:
    G reward = mean( log(D(G(m,s))) )   — starts at ≈ -18, converges toward 0
    D reward = mean( y*log(D(f)) + (1-y)*log(1-D(f)) ) — also negative, converges to 0
  These are the negative values visible in the paper's training curves.

  The ADVERSARIAL VECTORS produced are structurally identical to Exp 8.
  This experiment confirms that the Mahalanobis OOD result does not depend on
  which loss formulation trained the GAN.

Flow:
  1. Load phishphresh + 80/20 split (RANDOM_STATE=42, same as all experiments)
  2. Train FeatureMLP PD (same as Exp 8, 20 epochs)
  3. Binarise phishing features (AlEroud 2020 encoding)
  4. Train GAN with exact paper loss equations (100 epochs)
     Display: negative reward curves (paper-style)
  5. Generate adversarial feature vectors
  6. Test on Exp 10 Mahalanobis OOD model (no retraining) → confirm 100% detection
  7. Save adversarial dataset for downstream experiments

Output:
  binary_gan_exact_paper/models/feature_mlp_pd.pth
  binary_gan_exact_paper/models/pd_scaler.pkl
  binary_gan_exact_paper/models/gan_generator.pth
  binary_gan_exact_paper/logs/run.log
  binary_gan_exact_paper/logs/gan_training_reward.csv   (negative values, paper style)
  binary_gan_exact_paper/logs/pd_evasion_report.txt
  binary_gan_exact_paper/logs/adversarial_dataset_v2.csv
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
EXP10_MODEL   = os.path.join(PROJECT_DIR, "binary_mahalanobis_ood", "models", "feature_mlp.pth")
EXP10_SCALER  = os.path.join(PROJECT_DIR, "binary_mahalanobis_ood", "models", "scaler.pkl")
EXP10_MAHA    = os.path.join(PROJECT_DIR, "binary_mahalanobis_ood", "models", "mahalanobis_params.pkl")

OUT_DIR       = os.path.join(PROJECT_DIR, "binary_gan_exact_paper")
OUT_MODELS    = os.path.join(OUT_DIR, "models")
OUT_LOGS      = os.path.join(OUT_DIR, "logs")

PD_MODEL_PATH = os.path.join(OUT_MODELS, "feature_mlp_pd.pth")
PD_SCALER_PATH= os.path.join(OUT_MODELS, "pd_scaler.pkl")
GAN_GEN_PATH  = os.path.join(OUT_MODELS, "gan_generator.pth")
REWARD_CSV    = os.path.join(OUT_LOGS,   "gan_training_reward.csv")
EVASION_RPT   = os.path.join(OUT_LOGS,   "pd_evasion_report.txt")
ADV_CSV       = os.path.join(OUT_LOGS,   "adversarial_dataset_v2.csv")
STDOUT_LOG    = os.path.join(OUT_LOGS,   "stdout.log")

os.makedirs(OUT_MODELS, exist_ok=True)
os.makedirs(OUT_LOGS,   exist_ok=True)

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
RANDOM_STATE  = 42
TEST_SIZE     = 0.20
BATCH_SIZE    = 512
PD_EPOCHS     = 20
PD_LR         = 3e-4
PD_HIDDEN     = [256, 128, 64]
PD_DROPOUT    = 0.3
GAN_EPOCHS    = 100
GAN_LR        = 1e-4          # Adam eps=1e-5 as per paper
GAN_NOISE_DIM = 64
GAN_HIDDEN    = 120           # paper: 120 neurons per layer
GAN_N_LAYERS  = 3             # paper: 3 hidden layers
EPS           = 1e-8          # numerical stability for raw log


# ══════════════════════════════════════════════════════════════════════════════
# Models — identical architecture to Exp 8
# ══════════════════════════════════════════════════════════════════════════════

class FeatureMLPPD(nn.Module):
    """Standalone FeatureMLP Phishing Detector (67 -> [256,128,64] -> 2)."""
    def __init__(self, in_dim):
        super().__init__()
        layers = []
        prev = in_dim
        for h in PD_HIDDEN:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(PD_DROPOUT)]
            prev = h
        layers.append(nn.Linear(prev, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x): return self.net(x)

    def predict(self, x, device):
        self.eval()
        with torch.no_grad():
            out = []
            for i in range(0, len(x), BATCH_SIZE):
                t = torch.from_numpy(x[i:i+BATCH_SIZE]).float().to(device)
                out.extend(self.net(t).argmax(1).cpu().numpy())
        return np.array(out)


class GANGenerator(nn.Module):
    """
    Generator G — paper Section 3.1:
    Input : 134-dim binary phishing vector + 64-dim noise
    Hidden: 3 layers × 120 neurons, ReLU (Eq 1)
    Output: 134-dim sigmoid (Eq 2) — kept in [0,1] for binarisation
    """
    def __init__(self, binary_dim):
        super().__init__()
        layers = []
        in_sz  = binary_dim + GAN_NOISE_DIM
        for _ in range(GAN_N_LAYERS):
            layers += [nn.Linear(in_sz, GAN_HIDDEN), nn.ReLU()]
            in_sz = GAN_HIDDEN
        layers += [nn.Linear(GAN_HIDDEN, binary_dim), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)
        self.binary_dim = binary_dim

    def forward(self, x_bin, noise):
        return self.net(torch.cat([x_bin, noise], dim=1))


class GANDiscriminator(nn.Module):
    """
    Discriminator D — paper Section 3.2:
    Approximates the PD decision function.
    D(f) = P(legitimate | f)  ← paper Eq 4 requires D → 1 when sample looks legitimate
    Input : binary feature vector (134-dim)
    Output: single scalar in [0,1]  — HIGH means legitimate, LOW means phishing
    """
    def __init__(self, binary_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(binary_dim, GAN_HIDDEN), nn.ReLU(),
            nn.Linear(GAN_HIDDEN, GAN_HIDDEN), nn.ReLU(),
            nn.Linear(GAN_HIDDEN, 1),           nn.Sigmoid(),
        )

    def forward(self, x): return self.net(x)


class FeatDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


# ══════════════════════════════════════════════════════════════════════════════
# Binarisation helpers (AlEroud 2020 encoding — unchanged from Exp 8)
# ══════════════════════════════════════════════════════════════════════════════

def compute_thresholds(X_phish):
    q1  = np.percentile(X_phish, 25, axis=0).astype(np.float32)
    med = np.percentile(X_phish, 50, axis=0).astype(np.float32)
    eps = 1e-6
    q1  = np.where(q1 >= med, med - eps, q1)
    return q1, med

def to_binary(X, q1, med):
    n, f  = X.shape
    out   = np.zeros((n, f*2), dtype=np.float32)
    for i in range(f):
        v             = X[:, i]
        malicious     = v <= q1[i]
        suspicious    = (~malicious) & (v <= med[i])
        out[:, 2*i]   = malicious.astype(np.float32)
        out[:, 2*i+1] = (malicious | suspicious).astype(np.float32)
    return out

def from_binary(X_bin, q1, med, feat_min, feat_max):
    n, d = X_bin.shape
    f    = d // 2
    out  = np.zeros((n, f), dtype=np.float32)
    for i in range(f):
        b0, b1   = X_bin[:, 2*i], X_bin[:, 2*i+1]
        is_mal   = (b0 == 1) & (b1 == 1)
        is_sus   = (b0 == 0) & (b1 == 1)
        out[:, i] = np.where(is_mal, (feat_min[i]+q1[i])/2.0,
                    np.where(is_sus, (q1[i]+med[i])/2.0,
                                     (med[i]+feat_max[i])/2.0))
    return out


# ══════════════════════════════════════════════════════════════════════════════
# EXACT PAPER LOSS FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def loss_D_paper(D_out, pd_leg_labels):
    """
    Exact Eq 3 (corrected from PDF rendering):
      L_D = -E_{PD_Leg}[log D(f)] - E_{PD_phish}[log(1 - D(f))]
    D(f) = P(legitimate|f), pd_leg_labels = 1 if PD says legitimate, 0 if phishing.
    This is standard binary cross-entropy for D matching PD's labels.
    Returns a POSITIVE loss value (minimised during training).
    """
    d = D_out.squeeze().clamp(EPS, 1.0 - EPS)
    y = pd_leg_labels.float()
    return -torch.mean(y * torch.log(d) + (1.0 - y) * torch.log(1.0 - d))


def loss_G_paper(D_fake_out):
    """
    Exact Eq 4:
      L_G = -E[log D_wd(G_wg(m, s))]
    D(G(m,s)) = P(legitimate | adversarial sample).
    G minimises this → pushes D to output HIGH (1) → adversarials look legitimate.
    Returns a POSITIVE loss value (minimised during training).
    Paper's training curve shows the RAW REWARD = -L_G = E[log D(G)] which is NEGATIVE.
    """
    d = D_fake_out.squeeze().clamp(EPS, 1.0 - EPS)
    return -torch.mean(torch.log(d))


def reward_G_paper(D_fake_out):
    """
    Paper-style display value: E[log D(G)]  — NEGATIVE, converges toward 0.
    This is the NEGATIVE reward the paper plots on its training curves.
    Starts very negative (≈ log(near_zero)) when G hasn't learned.
    Approaches 0 when G successfully fools D (D(G) → 1).
    """
    d = D_fake_out.squeeze().clamp(EPS, 1.0 - EPS)
    return torch.mean(torch.log(d)).item()


def reward_D_paper(D_out, pd_leg_labels):
    """Paper-style D reward: E[y*log D + (1-y)*log(1-D)] — negative, toward 0."""
    d = D_out.squeeze().clamp(EPS, 1.0 - EPS)
    y = pd_leg_labels.float()
    return torch.mean(y * torch.log(d) + (1.0 - y) * torch.log(1.0 - d)).item()


# ══════════════════════════════════════════════════════════════════════════════
# Mahalanobis OOD helpers (exact copy from Exp 10 — used for final evaluation)
# ══════════════════════════════════════════════════════════════════════════════

class FeatureMLP_OOD(nn.Module):
    """Exp 10 FeatureMLP with penultimate extractor."""
    def __init__(self, in_dim, hidden=PD_HIDDEN, dropout=PD_DROPOUT):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.head     = nn.Linear(prev, 2)

    def forward(self, x): return self.head(self.backbone(x))
    def extract_penultimate(self, x): return self.backbone(x)


def extract_feats(model, X, device):
    model.eval()
    parts = []
    with torch.no_grad():
        for i in range(0, len(X), BATCH_SIZE):
            xb = torch.from_numpy(X[i:i+BATCH_SIZE]).float().to(device)
            parts.append(model.extract_penultimate(xb).cpu().numpy())
    return np.vstack(parts)

def maha_scores(feats, mu_0, mu_1, P):
    f  = feats.astype(np.float64)
    d0 = f - mu_0;  d1 = f - mu_1
    m0 = np.einsum('ij,jk,ik->i', d0, P, d0)
    m1 = np.einsum('ij,jk,ik->i', d1, P, d1)
    return np.minimum(m0, m1)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("=" * 80)
    log.info("Experiment 12 — GAN with EXACT Paper Loss Functions (AlEroud 2020)")
    log.info("  D loss: Eq 3  L_D = -E[log D(f)] - E[log(1-D(f))]  (D = P(legitimate))")
    log.info("  G loss: Eq 4  L_G = -E[log D(G(m,s))]")
    log.info("  Display: raw reward E[log D(G)] — NEGATIVE values, paper-style")
    log.info(f"Device : {device}")
    log.info("=" * 80)

    # ── STEP 1: Load phishphresh ────────────────────────────────────────────────
    log.info("\n[1/6] Loading phishphresh chunks...")
    train_chunks = sorted(glob.glob(os.path.join(DATA_DIR, "train", "chunk_*.npz")))
    test_chunks  = sorted(glob.glob(os.path.join(DATA_DIR, "test",  "chunk_*.npz")))
    if not (os.path.exists(os.path.join(DATA_DIR, "_train_complete.flag")) and
            os.path.exists(os.path.join(DATA_DIR, "_test_complete.flag"))):
        log.error("Chunks incomplete — re-run train_multiclass_brand.py first.")
        sys.exit(1)

    feat_parts, tgt_parts = [], []
    for chunk in tqdm(train_chunks + test_chunks, desc="  Loading"):
        d = np.load(chunk, allow_pickle=True)
        feat_parts.append(d["features"])
        tgt_parts.extend(d["targets"].tolist())

    X_all = np.vstack(feat_parts).astype(np.float32)
    y_all = np.array([0 if str(t or "").strip() == "" else 1 for t in tgt_parts], dtype=np.int64)
    del feat_parts; gc.collect()

    N_FEAT     = X_all.shape[1]
    BINARY_DIM = N_FEAT * 2
    log.info(f"  Total: {len(y_all):,}  Benign={int((y_all==0).sum()):,}  "
             f"Phishing={int((y_all==1).sum()):,}  Features={N_FEAT}")

    # ── STEP 2: Split + scale ──────────────────────────────────────────────────
    log.info("\n[2/6] 80/20 split + StandardScaler...")
    idx            = np.arange(len(y_all))
    tr_idx, te_idx = train_test_split(idx, test_size=TEST_SIZE,
                                      random_state=RANDOM_STATE, stratify=y_all)
    Xf_tr, Xf_te = X_all[tr_idx], X_all[te_idx]
    y_tr,  y_te  = y_all[tr_idx], y_all[te_idx]
    del X_all; gc.collect()

    scaler   = StandardScaler()
    Xf_tr_s  = scaler.fit_transform(Xf_tr).astype(np.float32)
    Xf_te_s  = scaler.transform(Xf_te).astype(np.float32)
    joblib.dump(scaler, PD_SCALER_PATH)
    log.info(f"  Train={len(y_tr):,}  Test={len(y_te):,}")

    # ── STEP 3: Train FeatureMLP PD ────────────────────────────────────────────
    log.info(f"\n[3/6] Training FeatureMLP PD ({PD_EPOCHS} epochs)...")
    counts_tr = np.bincount(y_tr)
    sw        = (1.0 / counts_tr)[y_tr]
    sampler   = WeightedRandomSampler(sw, num_samples=len(y_tr), replacement=True)
    pd_tr_dl  = DataLoader(FeatDataset(Xf_tr_s, y_tr), batch_size=BATCH_SIZE,
                           sampler=sampler, num_workers=0)
    pd_te_dl  = DataLoader(FeatDataset(Xf_te_s, y_te), batch_size=BATCH_SIZE,
                           shuffle=False, num_workers=0)

    pd_model  = FeatureMLPPD(N_FEAT).to(device)
    pd_optim  = torch.optim.Adam(pd_model.parameters(), lr=PD_LR)
    pd_crit   = nn.CrossEntropyLoss()

    for epoch in range(1, PD_EPOCHS + 1):
        pd_model.train()
        for xb, yb in pd_tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            pd_optim.zero_grad()
            pd_crit(pd_model(xb), yb).backward()
            pd_optim.step()

        pd_model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in pd_te_dl:
                xb, yb = xb.to(device), yb.to(device)
                correct += (pd_model(xb).argmax(1) == yb).sum().item()
                total   += len(yb)
        log.info(f"  Epoch {epoch:02d}/{PD_EPOCHS}  val_acc={correct/total*100:.2f}%")

    torch.save(pd_model.state_dict(), PD_MODEL_PATH)
    log.info(f"  PD saved -> {PD_MODEL_PATH}")

    # ── STEP 4: Binarise + Train GAN with EXACT paper equations ───────────────
    log.info(f"\n[4/6] Training GAN with EXACT paper equations ({GAN_EPOCHS} epochs)...")
    log.info("  D outputs P(legitimate|f) — Eq 4 requires D(G) -> 1 when G fools D")
    log.info("  DISPLAY: raw reward E[log D(G)]  — NEGATIVE values, converges toward 0")

    # Get PD predictions on phishing training samples to use as D labels
    phish_mask   = y_tr == 1
    Xf_phish_tr  = Xf_tr_s[phish_mask]
    pd_preds     = pd_model.predict(Xf_phish_tr, device)
    log.info(f"  Phishing training samples : {len(Xf_phish_tr):,}")
    log.info(f"  PD detects as phishing    : {int((pd_preds==1).sum()):,} "
             f"({int((pd_preds==1).sum())/len(pd_preds)*100:.1f}%)")

    # Binarise (unscaled phishing features)
    Xf_phish_raw = Xf_tr[phish_mask]    # unscaled
    q1, med      = compute_thresholds(Xf_phish_raw)
    feat_min     = Xf_phish_raw.min(axis=0).astype(np.float32)
    feat_max     = Xf_phish_raw.max(axis=0).astype(np.float32)
    X_bin        = to_binary(Xf_phish_raw, q1, med)
    log.info(f"  Binary dim: {BINARY_DIM}  GAN input: {X_bin.shape}")

    # PD says legitimate = 1 → this is the "legitimate" label for D
    # PD says phishing   = 0 → D should output LOW for these
    # Note: PD outputs 0=benign, 1=phishing
    # D = P(legitimate), so pd_leg_label = 1 - pd_preds  (flip: PD_Leg → D target=1)
    pd_leg_labels = (1 - pd_preds).astype(np.float32)   # 1 if PD says benign, 0 if phishing

    X_bin_t      = torch.from_numpy(X_bin).float()
    pd_leg_t     = torch.from_numpy(pd_leg_labels)
    gan_ds       = torch.utils.data.TensorDataset(X_bin_t, pd_leg_t)
    gan_dl       = DataLoader(gan_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    G      = GANGenerator(BINARY_DIM).to(device)
    D      = GANDiscriminator(BINARY_DIM).to(device)
    opt_G  = torch.optim.Adam(G.parameters(), lr=GAN_LR, eps=1e-5)
    opt_D  = torch.optim.Adam(D.parameters(), lr=GAN_LR, eps=1e-5)

    reward_rows = []
    for epoch in range(1, GAN_EPOCHS + 1):
        g_rew_sum = d_rew_sum = g_loss_sum = d_loss_sum = 0.0
        n_batches = 0

        for xb, yd_leg in gan_dl:
            xb     = xb.to(device)
            yd_leg = yd_leg.to(device)
            bs     = xb.size(0)

            # ── Train D (Eq 3): match PD's labels ──────────────────────────────
            opt_D.zero_grad()
            D_real  = D(xb)
            lD      = loss_D_paper(D_real, yd_leg)   # Eq 3, positive loss
            lD.backward()
            opt_D.step()

            # ── Train G (Eq 4): make D output HIGH (legitimate) for adversarials
            opt_G.zero_grad()
            noise   = torch.randn(bs, GAN_NOISE_DIM, device=device)
            fake    = G(xb, noise)
            D_fake  = D(fake)
            lG      = loss_G_paper(D_fake)    # Eq 4: -E[log D(G)], positive loss
            lG.backward()
            opt_G.step()

            # Paper-style display: raw REWARD (negative values)
            with torch.no_grad():
                g_rew_sum += reward_G_paper(D(G(xb, torch.randn(bs, GAN_NOISE_DIM, device=device))))
                d_rew_sum += reward_D_paper(D(xb), yd_leg)
            g_loss_sum += lG.item()
            d_loss_sum += lD.item()
            n_batches  += 1

        avg_g_rew  = g_rew_sum  / n_batches
        avg_d_rew  = d_rew_sum  / n_batches
        avg_g_loss = g_loss_sum / n_batches
        avg_d_loss = d_loss_sum / n_batches

        if epoch % 10 == 0 or epoch == 1:
            log.info(f"  Epoch {epoch:03d}/{GAN_EPOCHS} | "
                     f"G_reward={avg_g_rew:.4f}  D_reward={avg_d_rew:.4f}  "
                     f"(G_loss={avg_g_loss:.4f}  D_loss={avg_d_loss:.4f})")
            log.info(f"    ^ Paper-style display: G_reward and D_reward are NEGATIVE "
                     f"(converge toward 0 as training succeeds)")

        reward_rows.append({"epoch": epoch,
                            "G_reward_paper_style": avg_g_rew,
                            "D_reward_paper_style": avg_d_rew,
                            "G_loss_positive":      avg_g_loss,
                            "D_loss_positive":      avg_d_loss})

    pd.DataFrame(reward_rows).to_csv(REWARD_CSV, index=False)
    torch.save(G.state_dict(), GAN_GEN_PATH)
    log.info(f"  GAN saved -> {GAN_GEN_PATH}")
    log.info(f"  Reward log -> {REWARD_CSV}  (G_reward column shows paper-style negative values)")

    # ── STEP 5: Generate adversarial vectors ───────────────────────────────────
    log.info("\n[5/6] Generating adversarial feature vectors...")
    G.eval()
    adv_parts = []
    with torch.no_grad():
        for i in range(0, len(X_bin), BATCH_SIZE):
            xb    = torch.from_numpy(X_bin[i:i+BATCH_SIZE]).float().to(device)
            noise = torch.randn(len(xb), GAN_NOISE_DIM, device=device)
            out   = G(xb, noise)
            bin_np = (out.cpu() > 0.5).float().numpy()
            adv_parts.append(from_binary(bin_np, q1, med, feat_min, feat_max))

    X_adv = np.vstack(adv_parts).astype(np.float32)
    log.info(f"  Generated {len(X_adv):,} adversarial feature vectors")

    # Test evasion on PD
    X_adv_s   = scaler.transform(X_adv).astype(np.float32)
    preds_adv = pd_model.predict(X_adv_s, device)
    n_evaded  = int((preds_adv == 0).sum())
    log.info(f"  Evasion (no defense): {n_evaded:,}/{len(preds_adv):,} "
             f"= {n_evaded/len(preds_adv)*100:.1f}%")

    # Save adversarial dataset
    feat_cols = [f"f{i}" for i in range(N_FEAT)]
    adv_df    = pd.DataFrame(X_adv, columns=feat_cols)
    adv_df["label"] = 1
    adv_df.to_csv(ADV_CSV, index=False)
    log.info(f"  Adversarial dataset saved -> {ADV_CSV}")

    # ── STEP 6: Evaluate with Exp 10 Mahalanobis OOD ──────────────────────────
    log.info("\n[6/6] Running Exp 10 Mahalanobis OOD on new adversarial vectors...")
    log.info("  (Exp 10 model NOT retrained — loaded as-is from binary_mahalanobis_ood/)")

    scaler_exp10 = joblib.load(EXP10_SCALER)
    maha_params  = joblib.load(EXP10_MAHA)
    mu_0 = maha_params["mu_0"];  mu_1 = maha_params["mu_1"]
    P    = maha_params["P"];     threshold = maha_params["threshold"]

    ood_model  = FeatureMLP_OOD(N_FEAT).to(device)
    ood_model.load_state_dict(torch.load(EXP10_MODEL, map_location=device))
    ood_model.eval()
    log.info(f"  Loaded Exp 10 model  threshold={threshold:.4f}")

    X_adv_exp10  = scaler_exp10.transform(X_adv).astype(np.float32)
    feats_adv    = extract_feats(ood_model, X_adv_exp10, device)
    scores_adv   = maha_scores(feats_adv, mu_0, mu_1, P)
    ood_detected = (scores_adv > threshold).sum()

    log.info(f"\n  ════════════════════════════════════════════════")
    log.info(f"  GAN adversarial vectors  : {len(X_adv):,}")
    log.info(f"  GAN evasion (no OOD)     : {n_evaded/len(preds_adv)*100:.1f}%")
    log.info(f"  Mahalanobis OOD detected : {ood_detected:,}/{len(X_adv):,} "
             f"({ood_detected/len(X_adv)*100:.1f}%)")
    log.info(f"  Adv score median         : {np.median(scores_adv):.2f}")
    log.info(f"  Adv score min/max        : {scores_adv.min():.2f} / {scores_adv.max():.2f}")
    log.info(f"  OOD threshold            : {threshold:.4f}")
    log.info(f"  ════════════════════════════════════════════════")

    # Write evasion report
    with open(EVASION_RPT, "w", encoding="utf-8") as rpt:
        rpt.write("=" * 60 + "\n")
        rpt.write("Experiment 12 — Exact Paper GAN + Mahalanobis OOD\n")
        rpt.write("GAN loss: Exact Eq 3 + Eq 4 from AlEroud 2020\n")
        rpt.write("=" * 60 + "\n\n")
        rpt.write("GAN TRAINING\n")
        rpt.write(f"  Loss formula : Exact paper Eq 3 (D) + Eq 4 (G)\n")
        rpt.write(f"  D output     : P(legitimate|f)\n")
        rpt.write(f"  G reward     : E[log D(G)]  — NEGATIVE, paper-style\n")
        rpt.write(f"  Final G_reward (epoch 100): {reward_rows[-1]['G_reward_paper_style']:.4f}\n")
        rpt.write(f"  Final D_reward (epoch 100): {reward_rows[-1]['D_reward_paper_style']:.4f}\n\n")
        rpt.write("EVASION (exact-paper GAN vs FeatureMLP PD)\n")
        rpt.write(f"  Adversarial vectors : {len(X_adv):,}\n")
        rpt.write(f"  Evasion rate        : {n_evaded/len(preds_adv)*100:.1f}%\n\n")
        rpt.write("MAHALANOBIS OOD DETECTION (Exp 10 model, no retraining)\n")
        rpt.write(f"  OOD detected        : {ood_detected:,}/{len(X_adv):,} "
                  f"({ood_detected/len(X_adv)*100:.1f}%)\n")
        rpt.write(f"  Adv score median    : {np.median(scores_adv):.2f}\n")
        rpt.write(f"  OOD threshold       : {threshold:.4f}\n")
        rpt.write(f"  Score / threshold   : {np.median(scores_adv)/threshold:.0f}x above\n")

    log.info("=" * 80)
    log.info("Experiment 12 complete.")
    log.info("=" * 80)
