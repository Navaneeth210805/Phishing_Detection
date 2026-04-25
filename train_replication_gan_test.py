#!/usr/bin/env python3
"""
train_replication_gan_test.py
==============================
Experiment 11 — Cross-Dataset Validation of Mahalanobis OOD Defense
                using the Sabir et al. Replication Package datasets.

This experiment does NOT retrain the Mahalanobis model.
It loads Exp 10's trained weights (binary_mahalanobis_ood/) and tests them on
three completely different datasets — proving the defense generalises beyond
the phishphresh training distribution.

Three evaluations (all using Exp 10 model, zero retraining):
  A. Real URL classification: Leg_Training.csv + Phish_Training.csv
       Feature-extracted → run through Exp 10 model → accuracy, FNR, FPR, OOD rate
  B. GAN adversarial vectors trained on Replication Package phishing features
       Same AlEroud 2020 GAN → adversarial vectors generated from Replication
       Package phishing data → run through Exp 10 Mahalanobis OOD → detection rate
  C. Sabir adversarial URLs (DomainAdversary / PathAdversary / TLDAdversary)
       Feature-extracted crafted URLs → standard + OOD detection on Exp 10 model

Key research question:
  Does the Mahalanobis OOD detector, trained only on phishphresh, catch GAN adversarial
  vectors generated from a completely different phishing dataset (Replication Package)?
  If yes → the defense is dataset-agnostic (works because of the binary-encoding quantization
  artifact, not because of dataset-specific feature distributions).

References:
  - Lee et al., NeurIPS 2018, arXiv:1807.03888 (Mahalanobis OOD)
  - AlEroud & Karabatis, IWSPA 2020, DOI:10.1145/3375708.3380315 (GAN attack)
  - Sabir et al. 2020, arXiv:2005.08454 (Replication Package)

Output:
  binary_replication_gan_test/logs/run.log
  binary_replication_gan_test/logs/eval_A_real_urls.txt
  binary_replication_gan_test/logs/eval_B_gan_vectors.txt
  binary_replication_gan_test/logs/eval_C_sabir_adversarial.txt
  binary_replication_gan_test/logs/gan_training_loss.csv
  binary_replication_gan_test/models/gan_generator_reppack.pth
"""

from __future__ import annotations
import gc, glob, logging, math, os, re, sys, urllib.parse
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
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Replication Package
REPACK      = os.path.join(PROJECT_DIR, "Replication_Package", "Replication_Package", "Datasets")
LEG_CSV     = os.path.join(REPACK, "Training_Dataset", "Legitimate", "Leg_Training.csv")
PHISH_CSV   = os.path.join(REPACK, "Training_Dataset", "Phishing",   "Phish_Training.csv")
DOMAIN_ADV  = os.path.join(REPACK, "Adversary_Dataset", "DomainAdversary.csv")
PATH_ADV    = os.path.join(REPACK, "Adversary_Dataset", "PathAdversary.csv")
TLD_ADV     = os.path.join(REPACK, "Adversary_Dataset", "TLDAdversary.csv")

# Exp 10 model artifacts (already trained — NOT retrained here)
EXP10_DIR    = os.path.join(PROJECT_DIR, "binary_mahalanobis_ood")
EXP10_MODEL  = os.path.join(EXP10_DIR, "models", "feature_mlp.pth")
EXP10_SCALER = os.path.join(EXP10_DIR, "models", "scaler.pkl")
EXP10_MAHA   = os.path.join(EXP10_DIR, "models", "mahalanobis_params.pkl")

# Output
OUT_DIR     = os.path.join(PROJECT_DIR, "binary_replication_gan_test")
OUT_MODELS  = os.path.join(OUT_DIR, "models")
OUT_LOGS    = os.path.join(OUT_DIR, "logs")

os.makedirs(OUT_MODELS, exist_ok=True)
os.makedirs(OUT_LOGS,   exist_ok=True)

STDOUT_LOG  = os.path.join(OUT_LOGS, "stdout.log")

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
RANDOM_STATE  = 42
TEST_SIZE     = 0.20
BATCH_SIZE    = 512
MAX_LEG       = 100_000   # cap legitimate URLs for extraction speed
GAN_EPOCHS    = 100
GAN_LR        = 1e-4
GAN_NOISE_DIM = 64
GAN_HIDDEN    = 120
GAN_N_LAYERS  = 3
OOD_THRESHOLD_OVERRIDE = None   # None = use saved threshold from Exp 10


# ══════════════════════════════════════════════════════════════════════════════
# Feature extraction (same URLFeatureExtractorV2 as Exp 4/5)
# ══════════════════════════════════════════════════════════════════════════════

sys.path.insert(0, PROJECT_DIR)
from main import FeatureExtractor as _BaseFE

_RE_IP       = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
_RE_HEX      = re.compile(r"%[0-9a-fA-F]{2}")
_RE_REDIRECT = re.compile(r"(url|redirect|redir|next|return|goto|target)=", re.I)
_EXTRA_DIM   = 16


def _shannon(s: str) -> float:
    if not s: return 0.0
    freq: dict[str, int] = {}
    for c in s: freq[c] = freq.get(c, 0) + 1
    n = len(s)
    return -sum((v/n) * math.log2(v/n) for v in freq.values())

def _safe_port(parsed) -> bool:
    try: return bool(parsed.port and parsed.port not in (80, 443))
    except: return False


class URLFeatureExtractorV2:
    """67-dim URL feature extractor (51 base + 16 structural). Same as Exp 4/5."""
    def __init__(self):
        self._base = _BaseFE()
        try:
            probe = list(self._base.extract_features("http://example.com").values())
            self._base_dim = len(probe)
        except:
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
        except:
            base = np.zeros(self._base_dim, dtype=np.float32)
        return np.concatenate([base, self._structural(url)])

    @staticmethod
    def _structural(url: str) -> np.ndarray:
        try:
            parsed = urllib.parse.urlparse(url if "://" in url else "http://" + url)
        except:
            return np.zeros(_EXTRA_DIM, dtype=np.float32)
        path    = parsed.path     or ""
        query   = parsed.query    or ""
        fragment= parsed.fragment or ""
        netloc  = parsed.netloc   or ""
        scheme  = (parsed.scheme  or "http").lower()
        host    = netloc.split("@")[-1].split(":")[0]
        return np.array([
            min(len(url), 2000) / 2000.0,
            min(len(path), 500)  / 500.0,
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


def extract_batch(urls, extractor, desc="Features"):
    feats = []
    for url in tqdm(urls, desc=f"  {desc}", mininterval=2.0):
        feats.append(extractor.extract(url))
    return np.vstack(feats).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Exp 10 FeatureMLP (must match architecture exactly)
# ══════════════════════════════════════════════════════════════════════════════

HIDDEN  = [256, 128, 64]
DROPOUT = 0.3

class FeatureMLP(nn.Module):
    def __init__(self, in_dim: int, hidden=HIDDEN, dropout=DROPOUT):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.head     = nn.Linear(prev, 2)

    def forward(self, x):
        return self.head(self.backbone(x))

    def extract_penultimate(self, x):
        return self.backbone(x)


# ══════════════════════════════════════════════════════════════════════════════
# AlEroud 2020 GAN components (identical to Exp 8)
# ══════════════════════════════════════════════════════════════════════════════

class GANGenerator(nn.Module):
    def __init__(self, binary_dim):
        super().__init__()
        layers = []
        in_sz  = binary_dim + GAN_NOISE_DIM
        for _ in range(GAN_N_LAYERS):
            layers += [nn.Linear(in_sz, GAN_HIDDEN), nn.ReLU()]
            in_sz = GAN_HIDDEN
        layers += [nn.Linear(GAN_HIDDEN, binary_dim), nn.Sigmoid()]
        self.net        = nn.Sequential(*layers)
        self.binary_dim = binary_dim

    def forward(self, x_bin, noise):
        return self.net(torch.cat([x_bin, noise], dim=1))


class GANDiscriminator(nn.Module):
    def __init__(self, binary_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(binary_dim, GAN_HIDDEN), nn.ReLU(),
            nn.Linear(GAN_HIDDEN, GAN_HIDDEN), nn.ReLU(),
            nn.Linear(GAN_HIDDEN, 1),           nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


# ══════════════════════════════════════════════════════════════════════════════
# Binarisation helpers (AlEroud 2020)
# ══════════════════════════════════════════════════════════════════════════════

def compute_thresholds(X_phish):
    q1  = np.percentile(X_phish, 25, axis=0).astype(np.float32)
    med = np.percentile(X_phish, 50, axis=0).astype(np.float32)
    eps = 1e-6
    q1  = np.where(q1 >= med, med - eps, q1)
    return q1, med

def to_binary(X, q1, med):
    n, f = X.shape
    out  = np.zeros((n, f*2), dtype=np.float32)
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
        b0, b1  = X_bin[:, 2*i], X_bin[:, 2*i+1]
        is_mal  = (b0 == 1) & (b1 == 1)
        is_sus  = (b0 == 0) & (b1 == 1)
        out[:, i] = np.where(is_mal, (feat_min[i]+q1[i])/2.0,
                    np.where(is_sus, (q1[i]+med[i])/2.0,
                                     (med[i]+feat_max[i])/2.0))
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Mahalanobis helpers
# ══════════════════════════════════════════════════════════════════════════════

def extract_features_batched(model, X, device):
    model.eval()
    parts = []
    with torch.no_grad():
        for i in range(0, len(X), BATCH_SIZE):
            xb = torch.from_numpy(X[i:i+BATCH_SIZE]).float().to(device)
            parts.append(model.extract_penultimate(xb).cpu().numpy())
    return np.vstack(parts)

def mahalanobis_scores(feats, mu_0, mu_1, P):
    f  = feats.astype(np.float64)
    d0 = f - mu_0;  d1 = f - mu_1
    m0 = np.einsum('ij,jk,ik->i', d0, P, d0)
    m1 = np.einsum('ij,jk,ik->i', d1, P, d1)
    return np.minimum(m0, m1)

def predict_batch(model, X, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), BATCH_SIZE):
            xb = torch.from_numpy(X[i:i+BATCH_SIZE]).float().to(device)
            preds.extend(model(xb).argmax(1).cpu().numpy().tolist())
    return np.array(preds)

def eval_with_ood(model, X_scaled, y_true, mu_0, mu_1, P, threshold, device, label=""):
    """Run standard classification + OOD detection, print + return metrics dict."""
    preds  = predict_batch(model, X_scaled, device)
    feats  = extract_features_batched(model, X_scaled, device)
    scores = mahalanobis_scores(feats, mu_0, mu_1, P)
    ood    = scores > threshold

    acc    = accuracy_score(y_true, preds) * 100
    cm     = confusion_matrix(y_true, preds)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        fnr = fn/(fn+tp)*100 if (fn+tp) > 0 else 0.0
        fpr = fp/(fp+tn)*100 if (fp+tn) > 0 else 0.0
    else:
        fnr = fpr = 0.0

    n_ood      = int(ood.sum())
    pct_ood    = n_ood / len(ood) * 100
    if n_ood > 0 and len(y_true) > 0:
        ood_phish  = (y_true[ood] == 1).mean() * 100
    else:
        ood_phish  = 0.0

    log.info(f"  [{label}]")
    log.info(f"    Samples         : {len(y_true):,}")
    log.info(f"    Standard acc    : {acc:.2f}%  FNR={fnr:.2f}%  FPR={fpr:.2f}%")
    log.info(f"    OOD flagged     : {n_ood:,} / {len(ood):,} ({pct_ood:.2f}%)")
    log.info(f"    Of OOD, % phish : {ood_phish:.1f}%")
    log.info(f"    Maha score med  : {np.median(scores):.2f}  max={scores.max():.2f}")

    return dict(label=label, n=len(y_true), acc=acc, fnr=fnr, fpr=fpr,
                n_ood=n_ood, pct_ood=pct_ood, ood_phish=ood_phish,
                score_med=float(np.median(scores)), score_max=float(scores.max()),
                cls_report=classification_report(y_true, preds,
                                                  target_names=["benign","phishing"]))


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("=" * 80)
    log.info("Experiment 11 — Cross-Dataset Validation: Mahalanobis OOD + Replication Package")
    log.info("Exp 10 model loaded (no retraining).  Ref: Lee 2018 NeurIPS + AlEroud 2020 IWSPA")
    log.info(f"Device : {device}")
    log.info("=" * 80)

    # ── Load Exp 10 model + Mahalanobis params ─────────────────────────────────
    log.info("\n[SETUP] Loading Exp 10 Mahalanobis OOD model...")
    scaler_exp10 = joblib.load(EXP10_SCALER)
    maha_params  = joblib.load(EXP10_MAHA)
    mu_0      = maha_params["mu_0"]
    mu_1      = maha_params["mu_1"]
    P         = maha_params["P"]
    threshold = OOD_THRESHOLD_OVERRIDE or maha_params["threshold"]

    # Determine feature dim from scaler
    N_FEAT = scaler_exp10.n_features_in_
    model  = FeatureMLP(N_FEAT).to(device)
    model.load_state_dict(torch.load(EXP10_MODEL, map_location=device))
    model.eval()
    log.info(f"  Loaded: {EXP10_MODEL}")
    log.info(f"  Feature dim: {N_FEAT}  |  OOD threshold: {threshold:.4f}")

    # ── Feature extractor ──────────────────────────────────────────────────────
    fe = URLFeatureExtractorV2()
    log.info(f"  Feature extractor dim: {fe.n_features}")

    # ══════════════════════════════════════════════════════════════════════════
    # PART A — Real URL classification on Replication Package data
    # ══════════════════════════════════════════════════════════════════════════
    log.info("\n" + "="*60)
    log.info("PART A — Real URL Classification (Replication Package)")
    log.info("="*60)

    log.info(f"\n  Loading Leg_Training.csv (cap={MAX_LEG:,}) + Phish_Training.csv...")
    leg_df   = pd.read_csv(LEG_CSV, encoding="utf-8").sample(
        n=min(MAX_LEG, 1_048_574), random_state=RANDOM_STATE)
    phish_df = pd.read_csv(PHISH_CSV, encoding="latin-1")
    log.info(f"  Legitimate: {len(leg_df):,}  |  Phishing: {len(phish_df):,}")

    leg_urls   = leg_df["url"].tolist()
    phish_urls = phish_df["url"].tolist()

    log.info("  Extracting features for legitimate URLs...")
    X_leg   = extract_batch(leg_urls,   fe, "Leg features")
    log.info("  Extracting features for phishing URLs...")
    X_phish = extract_batch(phish_urls, fe, "Phish features")

    X_A = np.vstack([X_leg, X_phish]).astype(np.float32)
    y_A = np.array([0]*len(leg_urls) + [1]*len(phish_urls), dtype=np.int64)
    del X_leg, X_phish; gc.collect()

    log.info(f"  Combined: {len(y_A):,} samples  Benign={int((y_A==0).sum()):,}  Phishing={int((y_A==1).sum()):,}")

    # Scale using Exp 10 scaler (trained on phishphresh — intentional cross-dataset test)
    X_A_s = scaler_exp10.transform(X_A).astype(np.float32)

    # 80/20 split for evaluation
    _, te_idx = train_test_split(np.arange(len(y_A)), test_size=TEST_SIZE,
                                  random_state=RANDOM_STATE, stratify=y_A)
    X_A_te = X_A_s[te_idx]
    y_A_te = y_A[te_idx]
    log.info(f"  Test split: {len(y_A_te):,} URLs")

    res_A = eval_with_ood(model, X_A_te, y_A_te, mu_0, mu_1, P, threshold, device,
                          "Part A: Replication Package real URLs (test split)")

    # Also keep full phishing features for GAN training (unscaled, raw)
    # Re-extract just the phishing portion
    _, te_ph_idx = train_test_split(np.arange(len(phish_urls)), test_size=TEST_SIZE,
                                     random_state=RANDOM_STATE)
    ph_all_idx   = np.arange(len(phish_urls))
    tr_ph_idx    = np.setdiff1d(ph_all_idx, te_ph_idx)

    X_phish_all = X_A[len(leg_urls):]   # unscaled phishing features
    X_phish_tr  = X_phish_all[tr_ph_idx]
    del X_A; gc.collect()
    log.info(f"  Phishing training features for GAN: {len(X_phish_tr):,}")

    # ══════════════════════════════════════════════════════════════════════════
    # PART B — GAN trained on Replication Package phishing features → test on Exp 10
    # ══════════════════════════════════════════════════════════════════════════
    log.info("\n" + "="*60)
    log.info("PART B — AlEroud GAN on Replication Package phishing features")
    log.info("="*60)

    # Binarise phishing training features
    log.info("\n  Binarising phishing features (AlEroud 2020 encoding)...")
    q1, med     = compute_thresholds(X_phish_tr)
    feat_min    = X_phish_tr.min(axis=0).astype(np.float32)
    feat_max    = X_phish_tr.max(axis=0).astype(np.float32)
    X_bin_tr    = to_binary(X_phish_tr, q1, med)
    BINARY_DIM  = X_bin_tr.shape[1]
    log.info(f"  Binary dim: {BINARY_DIM} | Phishing samples for GAN: {len(X_bin_tr):,}")

    # Scale phishing features for PD evaluation
    X_phish_tr_s = scaler_exp10.transform(X_phish_tr).astype(np.float32)

    # Train GAN (AlEroud 2020 exact architecture)
    G = GANGenerator(BINARY_DIM).to(device)
    D = GANDiscriminator(BINARY_DIM).to(device)
    opt_G = torch.optim.Adam(G.parameters(), lr=GAN_LR, eps=1e-5)
    opt_D = torch.optim.Adam(D.parameters(), lr=GAN_LR, eps=1e-5)
    bce   = nn.BCELoss()

    # Build PD labels for phishing training set (what Exp 10 model predicts)
    log.info("\n  Computing PD predictions on phishing training set...")
    pd_labels = predict_batch(model, X_phish_tr_s, device)
    log.info(f"  PD detects {int((pd_labels==1).sum()):,}/{len(pd_labels):,} as phishing "
             f"({int((pd_labels==1).sum())/len(pd_labels)*100:.1f}%)")

    # GAN training dataset: binary phishing vectors + PD labels
    X_bin_t  = torch.from_numpy(X_bin_tr).float()
    y_pd_t   = torch.from_numpy(pd_labels.astype(np.float32))
    gan_ds   = torch.utils.data.TensorDataset(X_bin_t, y_pd_t)
    gan_dl   = DataLoader(gan_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    log.info(f"\n  Training GAN ({GAN_EPOCHS} epochs, batch={BATCH_SIZE})...")
    gan_loss_rows = []
    for epoch in range(1, GAN_EPOCHS + 1):
        g_loss_sum = d_loss_sum = 0.0
        n_batches  = 0
        for xb, yd in gan_dl:
            xb = xb.to(device); yd = yd.to(device)
            bs = xb.size(0)

            # Train D to match PD
            opt_D.zero_grad()
            D_real = D(xb).squeeze()
            loss_D = bce(D_real, yd)
            loss_D.backward()
            opt_D.step()

            # Train G to fool D (make D predict benign = 0)
            opt_G.zero_grad()
            noise  = torch.randn(bs, GAN_NOISE_DIM, device=device)
            fake   = G(xb, noise)
            D_fake = D(fake).squeeze()
            loss_G = bce(D_fake, torch.zeros(bs, device=device))
            loss_G.backward()
            opt_G.step()

            g_loss_sum += loss_G.item(); d_loss_sum += loss_D.item(); n_batches += 1

        avg_g = g_loss_sum / n_batches; avg_d = d_loss_sum / n_batches
        if epoch % 10 == 0 or epoch == 1:
            log.info(f"  GAN Epoch {epoch:03d}/{GAN_EPOCHS}  loss_G={avg_g:.4f}  loss_D={avg_d:.4f}")
        gan_loss_rows.append({"epoch": epoch, "loss_G": avg_g, "loss_D": avg_d})

    pd.DataFrame(gan_loss_rows).to_csv(
        os.path.join(OUT_LOGS, "gan_training_loss.csv"), index=False)
    torch.save(G.state_dict(), os.path.join(OUT_MODELS, "gan_generator_reppack.pth"))
    log.info(f"  GAN generator saved -> {OUT_MODELS}/gan_generator_reppack.pth")

    # Generate adversarial vectors
    log.info("\n  Generating adversarial feature vectors...")
    G.eval()
    adv_parts = []
    with torch.no_grad():
        for i in range(0, len(X_bin_tr), BATCH_SIZE):
            xb    = torch.from_numpy(X_bin_tr[i:i+BATCH_SIZE]).float().to(device)
            noise = torch.randn(len(xb), GAN_NOISE_DIM, device=device)
            out   = G(xb, noise)
            binarised = (out.cpu() > 0.5).float().numpy()
            adv_parts.append(from_binary(binarised, q1, med, feat_min, feat_max))

    X_adv     = np.vstack(adv_parts).astype(np.float32)
    y_adv_true = np.ones(len(X_adv), dtype=np.int64)   # all are phishing-origin
    log.info(f"  Generated {len(X_adv):,} adversarial feature vectors")

    X_adv_s   = scaler_exp10.transform(X_adv).astype(np.float32)

    # Evaluate: standard evasion + Mahalanobis detection
    preds_adv = predict_batch(model, X_adv_s, device)
    n_evaded  = int((preds_adv == 0).sum())
    evasion   = n_evaded / len(preds_adv) * 100

    feats_adv = extract_features_batched(model, X_adv_s, device)
    scores_adv = mahalanobis_scores(feats_adv, mu_0, mu_1, P)
    ood_adv    = scores_adv > threshold
    n_detected = int(ood_adv.sum())
    det_rate   = n_detected / len(preds_adv) * 100

    log.info(f"\n  GAN evasion (no OOD layer)  : {evasion:.1f}%  ({n_evaded:,}/{len(preds_adv):,})")
    log.info(f"  OOD detected (Mahalanobis)  : {det_rate:.1f}%  ({n_detected:,}/{len(preds_adv):,})")
    log.info(f"  Adv score median  : {np.median(scores_adv):.2f}")
    log.info(f"  OOD threshold     : {threshold:.4f}")

    res_B = dict(n_adv=len(X_adv), evasion_pct=evasion, detection_pct=det_rate,
                 score_med=float(np.median(scores_adv)), threshold=threshold)

    # ══════════════════════════════════════════════════════════════════════════
    # PART C — Sabir Adversarial URLs (Domain / Path / TLD)
    # ══════════════════════════════════════════════════════════════════════════
    log.info("\n" + "="*60)
    log.info("PART C — Sabir Adversarial URLs (DomainAdversary / PathAdversary / TLDAdversary)")
    log.info("="*60)

    adv_configs = [
        ("DomainAdversary", DOMAIN_ADV, 12_569),
        ("PathAdversary",   PATH_ADV,   50_000),
        ("TLDAdversary",    TLD_ADV,    9_768),
    ]
    res_C_list = []
    for name, csv_path, cap in adv_configs:
        log.info(f"\n  Loading {name} (cap={cap:,})...")
        df  = pd.read_csv(csv_path, encoding="latin-1").head(cap)
        urls = df["craftedurl"].tolist()
        log.info(f"    URLs: {len(urls):,}")
        X_c = extract_batch(urls, fe, f"{name} feats")
        X_c_s = scaler_exp10.transform(X_c).astype(np.float32)
        # These are adversarial phishing → ground truth = 1
        y_c = np.ones(len(urls), dtype=np.int64)
        res = eval_with_ood(model, X_c_s, y_c, mu_0, mu_1, P, threshold, device, name)
        res_C_list.append(res)
        del X_c, X_c_s; gc.collect()

    # ══════════════════════════════════════════════════════════════════════════
    # Write reports
    # ══════════════════════════════════════════════════════════════════════════
    def write_report(path, title, content):
        with open(path, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write(title + "\n")
            f.write("Exp 10 model: binary_mahalanobis_ood/ (phishphresh-trained)\n")
            f.write("=" * 60 + "\n\n")
            f.write(content)

    # Part A report
    write_report(os.path.join(OUT_LOGS, "eval_A_real_urls.txt"),
                 "Part A — Real Replication Package URLs",
                 f"Samples     : {res_A['n']:,}\n"
                 f"Accuracy    : {res_A['acc']:.2f}%\n"
                 f"FNR         : {res_A['fnr']:.2f}%\n"
                 f"FPR         : {res_A['fpr']:.2f}%\n"
                 f"OOD flagged : {res_A['n_ood']:,} ({res_A['pct_ood']:.2f}%)\n"
                 f"Of OOD, %phish : {res_A['ood_phish']:.1f}%\n\n"
                 + res_A["cls_report"])

    # Part B report
    write_report(os.path.join(OUT_LOGS, "eval_B_gan_vectors.txt"),
                 "Part B — GAN Adversarial Vectors (Replication Package phishing data)",
                 f"GAN architecture  : AlEroud & Karabatis IWSPA 2020\n"
                 f"GAN training data : Phish_Training.csv ({len(X_phish_tr):,} phishing)\n"
                 f"Adversarial vecs  : {res_B['n_adv']:,}\n"
                 f"Evasion (no OOD)  : {res_B['evasion_pct']:.1f}%\n"
                 f"OOD detected      : {res_B['detection_pct']:.1f}%\n"
                 f"Adv score median  : {res_B['score_med']:.2f}\n"
                 f"OOD threshold     : {res_B['threshold']:.4f}\n")

    # Part C report
    c_content = ""
    for res in res_C_list:
        c_content += (f"[{res['label']}]\n"
                      f"  Samples     : {res['n']:,}\n"
                      f"  Accuracy    : {res['acc']:.2f}%\n"
                      f"  FNR (missed phishing) : {res['fnr']:.2f}%\n"
                      f"  OOD flagged : {res['n_ood']:,} ({res['pct_ood']:.2f}%)\n"
                      f"  Maha score median : {res['score_med']:.2f}\n\n")
    write_report(os.path.join(OUT_LOGS, "eval_C_sabir_adversarial.txt"),
                 "Part C — Sabir et al. Adversarial URLs (Domain/Path/TLD)",
                 c_content)

    log.info("\n" + "=" * 80)
    log.info("Experiment 11 complete. Reports in binary_replication_gan_test/logs/")
    log.info("=" * 80)
