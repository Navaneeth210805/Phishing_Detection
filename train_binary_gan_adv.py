#!/usr/bin/env python3
"""
train_binary_gan_adv.py
=======================
GAN adversarial training on FeatureMLP feature space.
Reference: AlEroud & Karabatis, IWSPA '20 (DOI: 10.1145/3375708.3380315)

Components (exactly as paper):
  PD  = Phishing Detector  — YOUR standalone FeatureMLP (2-class, feature-only)
                              This is the blackbox the GAN tries to fool.
  G   = Generator          — 3 hidden layers × 120 neurons, ReLU, sigmoid output
                              Input: binary phishing vector (134-dim) + noise (64-dim)
                              Output: adversarial binary vector (134-dim)
  D   = Discriminator      — 2-layer NN that approximates PD's decision boundary
                              Trained to match PD labels, NOT to distinguish real/fake.
                              Gradient flows back to G through D.

Flow:
  1.  Load + verify phishphresh chunks (print full stats)
  2.  80/20 split (RANDOM_STATE=42, same as all experiments)
  3.  Train standalone FeatureMLP (2-class) as PD on continuous features
  4.  Binarize 67-dim features → 134-dim binary (AlEroud 2-bit encoding)
  5.  Train GAN (Algorithm 1 from paper) targeting YOUR FeatureMLP PD
  6.  Generate adversarial feature vectors for all phishing training samples
  7.  Evaluate: test adversarial vectors through PD → print evasion rate
  8.  Save adversarial dataset → logs/adversarial_dataset.csv (for multiclass reuse)

Dataset: phishphresh only (cached chunks). No Sabir datasets.
Output:
  binary_phishing_adv_gan/models/feature_mlp_pd.pth   — trained FeatureMLP PD
  binary_phishing_adv_gan/models/pd_scaler.pkl          — scaler for PD
  binary_phishing_adv_gan/models/gan_generator.pth      — GAN Generator weights
  binary_phishing_adv_gan/logs/run.log
  binary_phishing_adv_gan/logs/stdout.log
  binary_phishing_adv_gan/logs/gan_training_loss.csv
  binary_phishing_adv_gan/logs/pd_evasion_report.txt    — evasion rates
  binary_phishing_adv_gan/logs/adversarial_dataset.csv  — reusable adv dataset
"""

from __future__ import annotations
import gc, glob, logging, math, os, sys, urllib.parse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(PROJECT_DIR, "phishphresh", "multiclass_brand", "data")
OUT_DIR      = os.path.join(PROJECT_DIR, "binary_phishing_adv_gan")
OUT_MODELS   = os.path.join(OUT_DIR, "models")
OUT_LOGS     = os.path.join(OUT_DIR, "logs")

PD_MODEL_PATH  = os.path.join(OUT_MODELS, "feature_mlp_pd.pth")
PD_SCALER_PATH = os.path.join(OUT_MODELS, "pd_scaler.pkl")
GAN_GEN_PATH   = os.path.join(OUT_MODELS, "gan_generator.pth")
GAN_LOG        = os.path.join(OUT_LOGS,   "gan_training_loss.csv")
EVASION_REPORT = os.path.join(OUT_LOGS,   "pd_evasion_report.txt")
ADV_DATASET    = os.path.join(OUT_LOGS,   "adversarial_dataset.csv")
STDOUT_LOG     = os.path.join(OUT_LOGS,   "stdout.log")

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

# PD (FeatureMLP Phishing Detector) training
PD_EPOCHS    = 20
PD_LR        = 3e-4
PD_HIDDEN    = [256, 128, 64]   # same style as URLPhishNet FeatureMLP
PD_DROPOUT   = 0.3

# GAN (AlEroud & Karabatis 2020)
GAN_EPOCHS    = 100
GAN_BATCH     = 512
GAN_LR        = 1e-4            # Adam, eps=1e-5 as per paper
GAN_NOISE_DIM = 64              # noise vector dimension
GAN_HIDDEN    = 120             # neurons per hidden layer (paper: 120)
GAN_N_LAYERS  = 3               # hidden layers (paper: 3)


# ══════════════════════════════════════════════════════════════════════════════
# Model definitions
# ══════════════════════════════════════════════════════════════════════════════

class FeatureMLPPD(nn.Module):
    """
    Standalone FeatureMLP Phishing Detector (2-class, feature-only).
    This IS the PD from AlEroud's paper in our context.
    Same hidden-layer style as URLPhishNet's FeatureMLP branch.
    Input: scaled continuous features (67-dim)
    Output: logits [benign, phishing]
    """
    def __init__(self, in_dim: int, hidden: list = PD_HIDDEN, dropout: float = PD_DROPOUT):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def predict(self, x: np.ndarray, device: torch.device) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            t = torch.from_numpy(x).float().to(device)
            out = []
            for i in range(0, len(t), BATCH_SIZE):
                out.extend(self.net(t[i:i+BATCH_SIZE]).argmax(1).cpu().numpy())
        return np.array(out)


class GANGenerator(nn.Module):
    """
    Generator G (AlEroud 2020):
    Input : binary phishing vector (BINARY_DIM) + noise (GAN_NOISE_DIM)
    Hidden: GAN_N_LAYERS × GAN_HIDDEN neurons, ReLU activation
    Output: GAN_BINARY_DIM neurons, sigmoid (keeps output in [0,1] for binarisation)
    """
    def __init__(self, binary_dim: int):
        super().__init__()
        layers: list[nn.Module] = []
        in_sz = binary_dim + GAN_NOISE_DIM
        for _ in range(GAN_N_LAYERS):
            layers += [nn.Linear(in_sz, GAN_HIDDEN), nn.ReLU()]
            in_sz = GAN_HIDDEN
        layers += [nn.Linear(GAN_HIDDEN, binary_dim), nn.Sigmoid()]
        self.net        = nn.Sequential(*layers)
        self.binary_dim = binary_dim

    def forward(self, x_bin: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x_bin, noise], dim=1))


class GANDiscriminator(nn.Module):
    """
    Discriminator D (AlEroud 2020):
    Approximates PD's decision boundary — NOT a real/fake classifier.
    Trained to match PD's predictions so that gradient can flow to G.
    Input : binary feature vector (BINARY_DIM)
    Output: single scalar (probability of being phishing, matching PD)
    """
    def __init__(self, binary_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(binary_dim, GAN_HIDDEN), nn.ReLU(),
            nn.Linear(GAN_HIDDEN, GAN_HIDDEN), nn.ReLU(),
            nn.Linear(GAN_HIDDEN, 1),           nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ══════════════════════════════════════════════════════════════════════════════
# Binarisation helpers (AlEroud 2020 encoding)
# ══════════════════════════════════════════════════════════════════════════════

def compute_thresholds(X_phish: np.ndarray):
    """
    Per-feature Q1 and Median from phishing training samples.
    Defines 3 regions:
      val <= Q1          → malicious  → bits [1, 1]  (= 11)
      Q1 < val <= Median → suspicious → bits [0, 1]  (= 01)
      val > Median       → benign     → bits [0, 0]  (= 00)
    """
    q1  = np.percentile(X_phish, 25, axis=0).astype(np.float32)
    med = np.percentile(X_phish, 50, axis=0).astype(np.float32)
    # Prevent zero-width bins
    eps = 1e-6
    q1  = np.where(q1 >= med, med - eps, q1)
    return q1, med


def to_binary(X: np.ndarray, q1: np.ndarray, med: np.ndarray) -> np.ndarray:
    """Continuous (n, f) → binary (n, 2f)."""
    n, f  = X.shape
    out   = np.zeros((n, f * 2), dtype=np.float32)
    for i in range(f):
        v              = X[:, i]
        malicious      = v <= q1[i]
        suspicious     = (~malicious) & (v <= med[i])
        out[:, 2*i]    = malicious.astype(np.float32)           # bit-0: 1 only malicious
        out[:, 2*i+1]  = (malicious | suspicious).astype(np.float32)  # bit-1: 1 if mal or sus
    return out


def from_binary(X_bin: np.ndarray, q1: np.ndarray, med: np.ndarray,
                feat_min: np.ndarray, feat_max: np.ndarray) -> np.ndarray:
    """Binary (n, 2f) → continuous (n, f) using region midpoints."""
    n, d = X_bin.shape
    f    = d // 2
    out  = np.zeros((n, f), dtype=np.float32)
    for i in range(f):
        b0 = X_bin[:, 2*i];   b1 = X_bin[:, 2*i+1]
        is_mal = (b0 == 1) & (b1 == 1)
        is_sus = (b0 == 0) & (b1 == 1)
        out[:, i] = np.where(is_mal,  (feat_min[i] + q1[i])  / 2.0,
                    np.where(is_sus,  (q1[i]  + med[i])      / 2.0,
                                      (med[i] + feat_max[i]) / 2.0))
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Simple dataset wrapper for PD training
# ══════════════════════════════════════════════════════════════════════════════

class FeatDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
    def __len__(self):        return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("=" * 80)
    log.info("GAN Adversarial Training — FeatureMLP Feature Space")
    log.info("Reference : AlEroud & Karabatis, IWSPA '20")
    log.info(f"Device    : {device}")
    log.info(f"Output    : {OUT_DIR}")
    log.info("=" * 80)

    # ── STEP 1: Load + verify phishphresh chunks ───────────────────────────────
    log.info("\n[1/7] Loading + verifying phishphresh chunks...")
    train_chunks = sorted(glob.glob(os.path.join(DATA_DIR, "train", "chunk_*.npz")))
    test_chunks  = sorted(glob.glob(os.path.join(DATA_DIR, "test",  "chunk_*.npz")))

    log.info(f"  Train chunks      : {len(train_chunks)}")
    log.info(f"  Test chunks       : {len(test_chunks)}")

    # Verify complete flags
    train_flag = os.path.exists(os.path.join(DATA_DIR, "_train_complete.flag"))
    test_flag  = os.path.exists(os.path.join(DATA_DIR, "_test_complete.flag"))
    log.info(f"  Train complete    : {train_flag}")
    log.info(f"  Test complete     : {test_flag}")
    if not (train_flag and test_flag):
        log.error("Chunks incomplete — re-run train_multiclass_brand.py first.")
        sys.exit(1)

    # Peek at chunk structure
    d0 = np.load(train_chunks[0], allow_pickle=True)
    log.info(f"  Chunk keys        : {list(d0.keys())}")
    log.info(f"  features shape    : {d0['features'].shape}  (per chunk)")
    log.info(f"  char_ids shape    : {d0['char_ids'].shape}  (per chunk)")
    log.info(f"  targets sample    : {d0['targets'][:5].tolist()}")

    # Load all chunks
    feat_parts, tgt_parts = [], []
    for f in tqdm(train_chunks + test_chunks, desc="  Loading chunks"):
        d = np.load(f, allow_pickle=True)
        feat_parts.append(d["features"])
        tgt_parts.extend(d["targets"].tolist())

    X_all = np.vstack(feat_parts).astype(np.float32)
    y_all = np.array(
        [0 if str(t or "").strip() == "" else 1 for t in tgt_parts],
        dtype=np.int64
    )
    del feat_parts; gc.collect()

    N_FEAT     = X_all.shape[1]
    BINARY_DIM = N_FEAT * 2

    log.info(f"\n  ── Dataset stats ──────────────────────────────────────────")
    log.info(f"  Total samples     : {len(y_all):,}")
    log.info(f"  Benign (0)        : {int((y_all==0).sum()):,}  ({(y_all==0).mean()*100:.1f}%)")
    log.info(f"  Phishing (1)      : {int((y_all==1).sum()):,}  ({(y_all==1).mean()*100:.1f}%)")
    log.info(f"  Feature dim       : {N_FEAT}")
    log.info(f"  Binary dim (2-bit): {BINARY_DIM}")
    log.info(f"  ───────────────────────────────────────────────────────────")

    # ── STEP 2: 80/20 split ────────────────────────────────────────────────────
    log.info("\n[2/7] 80/20 split (RANDOM_STATE=42)...")
    idx            = np.arange(len(y_all))
    tr_idx, te_idx = train_test_split(idx, test_size=TEST_SIZE,
                                      random_state=RANDOM_STATE, stratify=y_all)

    Xf_tr, Xf_te = X_all[tr_idx], X_all[te_idx]
    y_tr,  y_te  = y_all[tr_idx], y_all[te_idx]
    del X_all; gc.collect()

    log.info(f"  Train : {len(y_tr):,}  "
             f"(benign={int((y_tr==0).sum()):,}  phishing={int((y_tr==1).sum()):,})")
    log.info(f"  Test  : {len(y_te):,}  "
             f"(benign={int((y_te==0).sum()):,}  phishing={int((y_te==1).sum()):,})")

    # Scale features (PD is trained on scaled features)
    scaler   = StandardScaler()
    Xf_tr_s  = scaler.fit_transform(Xf_tr).astype(np.float32)
    Xf_te_s  = scaler.transform(Xf_te).astype(np.float32)
    joblib.dump(scaler, PD_SCALER_PATH)
    log.info(f"  Scaler saved → {PD_SCALER_PATH}")

    # ── STEP 3: Train FeatureMLP PD ────────────────────────────────────────────
    log.info(f"\n[3/7] Training FeatureMLP PD ({PD_EPOCHS} epochs)...")
    log.info(f"  Architecture : {N_FEAT} → {PD_HIDDEN} → 2")
    log.info(f"  This IS the Phishing Detector the GAN will try to fool.")

    counts_tr = np.bincount(y_tr)
    sw        = (1.0 / counts_tr)[y_tr]
    sampler   = WeightedRandomSampler(sw, num_samples=len(y_tr), replacement=True)
    pd_train_dl = DataLoader(FeatDataset(Xf_tr_s, y_tr),
                             batch_size=BATCH_SIZE, sampler=sampler,
                             num_workers=0, pin_memory=(device.type=="cuda"))
    pd_test_dl  = DataLoader(FeatDataset(Xf_te_s, y_te),
                             batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    pd_model  = FeatureMLPPD(in_dim=N_FEAT).to(device)
    pd_optim  = torch.optim.Adam(pd_model.parameters(), lr=PD_LR)
    pd_crit   = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, float(counts_tr[0]/counts_tr[1])],
                            device=device)
    )
    best_pd_acc = 0.0

    for epoch in range(1, PD_EPOCHS + 1):
        pd_model.train()
        for Xb, yb in pd_train_dl:
            Xb, yb = Xb.to(device), yb.to(device)
            pd_optim.zero_grad()
            pd_crit(pd_model(Xb), yb).backward()
            pd_optim.step()

        pd_model.eval()
        preds = []
        with torch.no_grad():
            for Xb, _ in pd_test_dl:
                preds.extend(pd_model(Xb.to(device)).argmax(1).cpu().numpy())
        acc = accuracy_score(y_te, preds)
        if acc > best_pd_acc:
            best_pd_acc = acc
            torch.save(pd_model.state_dict(), PD_MODEL_PATH)

        if epoch % 5 == 0 or epoch == PD_EPOCHS:
            log.info(f"  Epoch {epoch:2d}/{PD_EPOCHS}  test_acc={acc*100:.2f}%")

    pd_model.load_state_dict(torch.load(PD_MODEL_PATH, map_location=device))
    log.info(f"  Best PD accuracy : {best_pd_acc*100:.2f}%")
    log.info(f"  PD saved → {PD_MODEL_PATH}")

    # ── STEP 4: Binarize features ──────────────────────────────────────────────
    log.info("\n[4/7] Binarizing features (AlEroud 2-bit encoding)...")

    phish_mask     = (y_tr == 1)
    X_phish_tr_raw = Xf_tr[phish_mask]     # RAW (unscaled) phishing features
    X_benign_tr_raw= Xf_tr[~phish_mask]    # RAW benign features
    feat_min       = Xf_tr.min(axis=0)
    feat_max       = Xf_tr.max(axis=0)

    q1, med        = compute_thresholds(X_phish_tr_raw)
    X_phish_bin    = to_binary(X_phish_tr_raw, q1, med)
    X_benign_bin   = to_binary(X_benign_tr_raw, q1, med)

    log.info(f"  Phishing binary  : {X_phish_bin.shape}")
    log.info(f"  Benign binary    : {X_benign_bin.shape}")
    log.info(f"  Encoding: val<=Q1 → 11 (malicious) | Q1<val<=Med → 01 (suspicious) | else → 00 (benign)")

    # ── STEP 5: Train GAN (Algorithm 1 from AlEroud 2020) ─────────────────────
    log.info(f"\n[5/7] Training GAN ({GAN_EPOCHS} epochs)...")
    log.info(f"  G : input ({BINARY_DIM}+{GAN_NOISE_DIM}) → {GAN_N_LAYERS}×{GAN_HIDDEN} ReLU → {BINARY_DIM} sigmoid")
    log.info(f"  D : {BINARY_DIM} → 2×{GAN_HIDDEN} ReLU → 1 sigmoid  (approximates PD)")
    log.info(f"  PD: FeatureMLP ({N_FEAT}→{PD_HIDDEN}→2)  ← the blackbox we are fooling")
    log.info(f"  Batch={GAN_BATCH}  LR={GAN_LR}  noise_dim={GAN_NOISE_DIM}")

    G = GANGenerator(binary_dim=BINARY_DIM).to(device)
    D = GANDiscriminator(binary_dim=BINARY_DIM).to(device)

    opt_G = torch.optim.Adam(G.parameters(), lr=GAN_LR, eps=1e-5)
    opt_D = torch.optim.Adam(D.parameters(), lr=GAN_LR, eps=1e-5)
    bce   = nn.BCELoss()

    # Pre-load tensors onto device
    X_phish_t  = torch.from_numpy(X_phish_bin).float().to(device)
    X_benign_t = torch.from_numpy(X_benign_bin).float().to(device)
    n_phish    = len(X_phish_bin)
    n_benign   = len(X_benign_bin)
    rng        = np.random.default_rng(RANDOM_STATE)

    gan_rows = []

    for epoch in range(1, GAN_EPOCHS + 1):
        p_perm = rng.permutation(n_phish)
        b_perm = rng.permutation(n_benign)
        ep_g = ep_d = n_batches = 0

        for start in range(0, n_phish, GAN_BATCH):
            p_idx = p_perm[start : start + GAN_BATCH]
            b_start = start % n_benign
            b_idx   = b_perm[b_start : b_start + len(p_idx)]
            if len(b_idx) < len(p_idx):
                b_idx = np.concatenate([b_idx, b_perm[: len(p_idx) - len(b_idx)]])

            Xp = X_phish_t[p_idx]
            Xb = X_benign_t[b_idx]

            # ── Generate adversarial vectors ──────────────────────────────────
            noise  = torch.rand(len(Xp), GAN_NOISE_DIM, device=device)
            Xp_adv = G(Xp, noise)  # sigmoid → [0,1]

            # Get PD labels for adversarial (binary threshold → continuous → PD)
            # PD labels drive Discriminator training (not ground truth)
            with torch.no_grad():
                Xp_adv_bin  = (Xp_adv.detach() > 0.5).float().cpu().numpy()
                Xp_adv_cont = from_binary(Xp_adv_bin, q1, med, feat_min, feat_max)
                Xp_adv_s    = scaler.transform(Xp_adv_cont).astype(np.float32)
                pd_adv_lbl  = torch.from_numpy(
                    pd_model.predict(Xp_adv_s, device).astype(np.float32)
                ).unsqueeze(1).to(device)

                # PD labels for benign (should be mostly 0=benign)
                Xb_cont     = from_binary(Xb.cpu().numpy(), q1, med, feat_min, feat_max)
                Xb_s        = scaler.transform(Xb_cont).astype(np.float32)
                pd_ben_lbl  = torch.from_numpy(
                    pd_model.predict(Xb_s, device).astype(np.float32)
                ).unsqueeze(1).to(device)

            # ── Train Discriminator (match PD predictions) ────────────────────
            # Loss (eq 3): D learns to mimic PD on adversarial + benign samples
            opt_D.zero_grad()
            loss_D = bce(D(Xp_adv.detach()), pd_adv_lbl) + bce(D(Xb), pd_ben_lbl)
            loss_D.backward()
            opt_D.step()

            # ── Train Generator (fool D → fool PD) ───────────────────────────
            # Loss (eq 4): G wants D to classify output as benign (target=0)
            opt_G.zero_grad()
            noise_new    = torch.rand(len(Xp), GAN_NOISE_DIM, device=device)
            Xp_adv_new   = G(Xp, noise_new)
            loss_G       = bce(D(Xp_adv_new), torch.zeros(len(Xp_adv_new), 1, device=device))
            loss_G.backward()
            opt_G.step()

            ep_g += loss_G.item(); ep_d += loss_D.item(); n_batches += 1

        avg_g = ep_g / max(n_batches, 1)
        avg_d = ep_d / max(n_batches, 1)
        gan_rows.append({"epoch": epoch, "loss_G": round(avg_g, 6), "loss_D": round(avg_d, 6)})

        if epoch % 10 == 0 or epoch in (1, GAN_EPOCHS):
            log.info(f"  [GAN] Epoch {epoch:3d}/{GAN_EPOCHS}  "
                     f"loss_G={avg_g:.4f}  loss_D={avg_d:.4f}")

    pd.DataFrame(gan_rows).to_csv(GAN_LOG, index=False)
    torch.save(G.state_dict(), GAN_GEN_PATH)
    log.info(f"  GAN Generator saved → {GAN_GEN_PATH}")
    log.info(f"  GAN loss log       → {GAN_LOG}")

    # ── STEP 6: Generate adversarial dataset ───────────────────────────────────
    log.info("\n[6/7] Generating adversarial feature vectors (all phishing training samples)...")
    G.eval()
    adv_cont_parts = []
    with torch.no_grad():
        for start in range(0, n_phish, GAN_BATCH):
            batch = X_phish_t[start : start + GAN_BATCH]
            noise = torch.rand(len(batch), GAN_NOISE_DIM, device=device)
            out   = G(batch, noise)
            bin_np = (out.cpu() > 0.5).float().numpy()
            adv_cont_parts.append(from_binary(bin_np, q1, med, feat_min, feat_max))

    X_adv_cont  = np.vstack(adv_cont_parts).astype(np.float32)
    X_adv_s     = scaler.transform(X_adv_cont).astype(np.float32)
    log.info(f"  Generated {len(X_adv_cont):,} adversarial feature vectors")

    # ── STEP 7: Evaluate evasion on FeatureMLP PD ─────────────────────────────
    log.info("\n[7/7] Evaluating evasion — how well does PD detect adversarial vectors?")

    # Original phishing test on PD (baseline)
    Xf_te_phish_s = Xf_te_s[y_te == 1]
    y_te_phish    = y_te[y_te == 1]
    orig_preds    = pd_model.predict(Xf_te_phish_s, device)
    orig_detect   = orig_preds.mean()

    # Adversarial phishing on PD
    adv_preds   = pd_model.predict(X_adv_s, device)
    adv_detect  = adv_preds.mean()          # fraction correctly detected (should DROP after GAN)
    evasion_rate= 1.0 - adv_detect          # fraction that EVADED the PD

    log.info(f"  PD detection rate on ORIGINAL phishing (test)  : {orig_detect*100:.2f}%")
    log.info(f"  PD detection rate on ADVERSARIAL phishing (GAN): {adv_detect*100:.2f}%")
    log.info(f"  Evasion rate (fooled PD)                        : {evasion_rate*100:.2f}%")
    log.info(f"  → Higher evasion = GAN is working as expected")

    # Full classification report on adversarial vectors
    y_adv_true = np.ones(len(adv_preds), dtype=np.int64)
    cm         = confusion_matrix(y_adv_true, adv_preds, labels=[0, 1])
    log.info(f"\n  Confusion Matrix (adversarial vectors vs FeatureMLP PD):")
    log.info(f"                 Pred Benign  Pred Phishing")
    log.info(f"  True Phishing  {cm[1,0]:>11,}  {cm[1,1]:>13,}")
    log.info(f"  → {cm[1,0]:,} adversarial phishing examples evaded the FeatureMLP PD")

    # Save evasion report
    report_lines = [
        "=" * 70,
        "GAN Evasion Report — FeatureMLP PD (2-class, feature-only)",
        f"Reference: AlEroud & Karabatis, IWSPA '20",
        "=" * 70,
        f"PD architecture       : {N_FEAT} → {PD_HIDDEN} → 2",
        f"PD best accuracy      : {best_pd_acc*100:.2f}%",
        "",
        f"GAN config            : G = {GAN_N_LAYERS}×{GAN_HIDDEN} | D = 2×{GAN_HIDDEN} | {GAN_EPOCHS} epochs",
        f"Binary encoding       : {N_FEAT} features × 2 bits = {BINARY_DIM} dims",
        "",
        f"Original phishing detection rate (test set) : {orig_detect*100:.2f}%",
        f"Adversarial detection rate (GAN output)     : {adv_detect*100:.2f}%",
        f"Evasion rate                                : {evasion_rate*100:.2f}%",
        f"Adversarial vectors generated               : {len(X_adv_cont):,}",
        "",
        "Confusion Matrix (adversarial phishing vs FeatureMLP PD):",
        f"  Evaded PD (pred=benign)    : {cm[1,0]:,}  ({cm[1,0]/len(adv_preds)*100:.1f}%)",
        f"  Caught by PD (pred=phish)  : {cm[1,1]:,}  ({cm[1,1]/len(adv_preds)*100:.1f}%)",
        "",
        "NEXT STEP: Use adversarial_dataset.csv to retrain URLPhishNet",
        "           so FeatureMLP learns to catch these evasion patterns.",
        "=" * 70,
    ]
    with open(EVASION_REPORT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(report_lines))
    log.info(f"\n  Evasion report → {EVASION_REPORT}")

    # Save adversarial dataset (feature vectors + label=phishing)
    # Columns = feature_0 .. feature_N-1 + label
    feat_cols = [f"feature_{i}" for i in range(N_FEAT)]
    df_adv    = pd.DataFrame(X_adv_cont, columns=feat_cols)
    df_adv["label"] = 1   # all adversarial examples are phishing
    df_adv.to_csv(ADV_DATASET, index=False)
    log.info(f"  Adversarial dataset → {ADV_DATASET}")
    log.info(f"  Shape: {df_adv.shape}  (reusable for multiclass training)")

    log.info("\n" + "=" * 80)
    log.info("SUMMARY")
    log.info("=" * 80)
    log.info(f"  PD (FeatureMLP) accuracy            : {best_pd_acc*100:.2f}%")
    log.info(f"  PD detection BEFORE GAN attack      : {orig_detect*100:.2f}%")
    log.info(f"  PD detection AFTER  GAN attack      : {adv_detect*100:.2f}%")
    log.info(f"  Evasion rate                        : {evasion_rate*100:.2f}%")
    log.info(f"  Adversarial dataset saved           : {ADV_DATASET}")
    log.info(f"  → Next: retrain URLPhishNet with adversarial vectors (close the gap)")
    log.info("=" * 80)
    log.info("Done.")
    _stdout_fh.close()
