#!/usr/bin/env python3
"""
train_binary.py
===============
Binary URLPhishNet trained on the SAME phishphresh dataset as the 52-class model.
Labels: benign=0 (target==""), phishing=1 (any brand target).
Same 80/20 split (RANDOM_STATE=42) as train_multiclass_brand.py.

After training, evaluates on the three adversarial datasets from
Sabir et al. (arXiv:2005.08454) to compare with the paper's models.

Outputs:
  binary_phishing/models/model_best.pth
  binary_phishing/models/scaler.pkl
  binary_phishing/logs/run.log
  binary_phishing/logs/training_log.csv
  binary_phishing/logs/adversarial_results.txt
  binary_phishing/logs/adversarial_binary_<type>.csv
"""

from __future__ import annotations

import gc
import glob
import logging
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

PAPER_RESULTS = os.path.join(
    PROJECT_DIR, "Replication_Package", "Replication_Package",
    "TrainedModels", "TraditionalModels"
)

OUT_MODELS = os.path.join(PROJECT_DIR, "binary_phishing", "models")
OUT_LOGS   = os.path.join(PROJECT_DIR, "binary_phishing", "logs")
MODEL_PATH  = os.path.join(OUT_MODELS, "model_best.pth")
SCALER_PATH = os.path.join(OUT_MODELS, "scaler.pkl")
EPOCH_LOG   = os.path.join(OUT_LOGS, "training_log.csv")
REPORT_TXT  = os.path.join(OUT_LOGS, "adversarial_results.txt")

# ── Must match 52-class training exactly ──────────────────────────────────────
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
EPOCHS       = 40
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

# ── Feature extractor (for adversarial URLs — same 67 features) ───────────────
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
            probe = list(self._base.extract_features("http://example.com").values())
            self._base_dim = len(probe)
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
        path     = parsed.path     or ""
        query    = parsed.query    or ""
        fragment = parsed.fragment or ""
        netloc   = parsed.netloc   or ""
        scheme   = (parsed.scheme  or "http").lower()
        host     = netloc.split("@")[-1].split(":")[0]
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


# ── Dataset ────────────────────────────────────────────────────────────────────
class URLDataset(Dataset):
    def __init__(self, X_feat, X_char, labels):
        self.Xf = torch.from_numpy(X_feat)
        self.Xc = torch.from_numpy(X_char.astype(np.int64))
        self.y  = torch.from_numpy(labels.astype(np.int64))

    def __len__(self):          return len(self.y)
    def __getitem__(self, i):   return self.Xf[i], self.Xc[i], self.y[i]


# ── Model (identical to URLPhishNet, 2 output classes) ────────────────────────
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
        self.output_dim = 2 * fused

    def forward(self, x):
        e = self.emb(x).transpose(1, 2)
        c = torch.cat([F.relu(self.bn3(self.c3(e))),
                       F.relu(self.bn5(self.c5(e))),
                       F.relu(self.bn7(self.c7(e)))], dim=1)
        f = self.drop(F.relu(self.bnd(self.deep(c))))
        return torch.cat([F.adaptive_max_pool1d(f, 1).squeeze(-1),
                          F.adaptive_avg_pool1d(f, 1).squeeze(-1)], dim=1)


class BinaryURLPhishNet(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.feat_mlp = FeatureMLP(feat_dim)
        self.char_cnn = CharCNN()
        fusion = self.feat_mlp.output_dim + self.char_cnn.output_dim
        self.clf = nn.Sequential(
            nn.Linear(fusion, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(DROPOUT),
            nn.Linear(512, 256),   nn.LayerNorm(256), nn.GELU(), nn.Dropout(DROPOUT * 0.7),
            nn.Linear(256, 2),
        )

    def forward(self, feat, chars):
        return self.clf(torch.cat([self.feat_mlp(feat), self.char_cnn(chars)], dim=1))


# ── Helpers ────────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    f1w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    cm  = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    return dict(Accuracy=acc, F1_Macro=f1m, F1_Weighted=f1w, MCC=mcc,
                FPR=fpr, FNR=fnr, TP=int(tp), TN=int(tn), FP=int(fp), FN=int(fn))


def run_inference(model, X_feat, X_char, device):
    model.eval()
    Xf = torch.from_numpy(X_feat)
    Xc = torch.from_numpy(X_char.astype(np.int64))
    preds = []
    with torch.no_grad():
        for i in range(0, len(Xf), BATCH_SIZE):
            logits = model(Xf[i:i+BATCH_SIZE].to(device),
                           Xc[i:i+BATCH_SIZE].to(device)).float()
            preds.extend(logits.argmax(1).cpu().numpy())
    return np.array(preds)


def load_paper_results():
    rows = []
    for method in sorted(os.listdir(PAPER_RESULTS)):
        res_dir = os.path.join(PAPER_RESULTS, method, "Results")
        if not os.path.isdir(res_dir):
            continue
        for fname in os.listdir(res_dir):
            if not fname.endswith(".csv"):
                continue
            try:
                df = pd.read_csv(os.path.join(res_dir, fname))
                for _, r in df.iterrows():
                    if not isinstance(r.get("Accuracy"), float):
                        continue
                    rows.append({"Method": method, "File": fname,
                                 **{c: r[c] for c in ["Classifier", "Accuracy",
                                    "FscoreMacro", "MCC", "FPR", "FNR"]
                                    if c in r}})
            except Exception:
                pass
    return rows


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # ── 1. Load phishphresh chunks (same data as 52-class model) ──────────────
    log.info("[1/6] Loading phishphresh chunks...")
    all_chunks = sorted(
        glob.glob(os.path.join(DATA_DIR, "train", "chunk_*.npz")) +
        glob.glob(os.path.join(DATA_DIR, "test",  "chunk_*.npz"))
    )
    log.info(f"  Found {len(all_chunks)} chunks")

    feat_parts, char_parts, tgt_parts = [], [], []
    for f in tqdm(all_chunks, desc="  Loading chunks"):
        d = np.load(f, allow_pickle=True)
        feat_parts.append(d["features"])
        char_parts.append(d["char_ids"])
        tgt_parts.extend(d["targets"].tolist())

    X_feat = np.vstack(feat_parts).astype(np.float32)
    X_char = np.vstack(char_parts).astype(np.int64)
    del feat_parts, char_parts; gc.collect()

    # Convert multiclass targets -> binary: "" = benign=0, any brand = phishing=1
    labels = np.array([0 if str(t or "").strip() == "" else 1
                       for t in tgt_parts], dtype=np.int64)
    total   = len(labels)
    n_ben   = (labels == 0).sum()
    n_phish = (labels == 1).sum()
    log.info(f"  Total: {total:,}  benign={n_ben:,} ({n_ben/total*100:.1f}%)  "
             f"phishing={n_phish:,} ({n_phish/total*100:.1f}%)")

    # ── 2. Same 80/20 split as 52-class model (RANDOM_STATE=42) ──────────────
    log.info("[2/6] Reconstructing same 80/20 split (RANDOM_STATE=42)...")
    idx = np.arange(total)
    tr_idx, te_idx = train_test_split(idx, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    Xf_tr, Xf_te = X_feat[tr_idx], X_feat[te_idx]
    Xc_tr, Xc_te = X_char[tr_idx], X_char[te_idx]
    y_tr,  y_te  = labels[tr_idx], labels[te_idx]
    del X_feat, X_char; gc.collect()

    log.info(f"  Train={len(tr_idx):,}  Test={len(te_idx):,}  "
             f"phish%_train={y_tr.mean()*100:.1f}%  phish%_test={y_te.mean()*100:.1f}%")

    # ── 3. Scale features ─────────────────────────────────────────────────────
    log.info("[3/6] Fitting StandardScaler...")
    scaler = StandardScaler()
    Xf_tr  = scaler.fit_transform(Xf_tr).astype(np.float32)
    Xf_te  = scaler.transform(Xf_te).astype(np.float32)
    joblib.dump(scaler, SCALER_PATH)
    log.info(f"  Scaler saved -> {SCALER_PATH}")

    FEAT_DIM = Xf_tr.shape[1]
    log.info(f"  Feature dim: {FEAT_DIM}")

    # ── 4. Train ──────────────────────────────────────────────────────────────
    log.info(f"[4/6] Training BinaryURLPhishNet ({EPOCHS} epochs)...")
    counts   = np.bincount(y_tr)
    sample_w = (1.0 / counts)[y_tr]
    sampler  = WeightedRandomSampler(sample_w, num_samples=len(y_tr), replacement=True)
    train_dl = DataLoader(URLDataset(Xf_tr, Xc_tr, y_tr),
                          batch_size=BATCH_SIZE, sampler=sampler,
                          num_workers=0, pin_memory=True)

    model     = BinaryURLPhishNet(feat_dim=FEAT_DIM).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, steps_per_epoch=len(train_dl),
        epochs=EPOCHS, pct_start=0.1
    )
    # Phishing class weight: benign is ~55% so slightly boost phishing
    phish_w  = float(np.clip(counts[0] / counts[1], 1.0, 3.0))
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
        for Xf, Xc, y in train_dl:
            Xf, Xc, y = Xf.to(device), Xc.to(device), y.to(device)
            optimizer.zero_grad()
            if use_amp:
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    loss = criterion(model(Xf, Xc).float(), y)
                amp_scaler.scale(loss).backward()
                amp_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                amp_scaler.step(optimizer)
                amp_scaler.update()
            else:
                loss = criterion(model(Xf, Xc).float(), y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_dl)
        m = compute_metrics(y_te, run_inference(model, Xf_te, Xc_te, device))
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
    log.info(f"  Epoch log  -> {EPOCH_LOG}")
    log.info(f"  Best model: epoch {best_epoch}  F1={best_f1:.4f}  -> {MODEL_PATH}")

    # ── 5. Adversarial evaluation ─────────────────────────────────────────────
    log.info("[5/6] Adversarial evaluation (Sabir et al. attack datasets)...")
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    log.info(f"  Loaded best model from epoch {ckpt['epoch']}")

    extractor = URLFeatureExtractorV2()

    # Normal test baseline first
    normal_m = compute_metrics(y_te, run_inference(model, Xf_te, Xc_te, device))
    normal_m["Dataset"] = "Normal_Test (phishphresh)"
    log.info(f"  Normal test: acc={normal_m['Accuracy']*100:.2f}%  "
             f"F1={normal_m['F1_Macro']:.4f}  MCC={normal_m['MCC']:.4f}")

    adv_results = [normal_m]

    for adv_path, adv_name in [(DOMAIN_ADV, "Domain_Adversary"),
                                (PATH_ADV,   "Path_Adversary"),
                                (TLD_ADV,    "TLD_Adversary")]:
        try:
            df_adv = pd.read_csv(adv_path, usecols=["craftedurl"])
            urls   = df_adv["craftedurl"].dropna().astype(str).tolist()
            if len(urls) > MAX_ADV_PATH:
                log.info(f"  {adv_name}: {len(urls):,} URLs, capping to {MAX_ADV_PATH:,}")
                urls = urls[:MAX_ADV_PATH]
            else:
                log.info(f"  {adv_name}: {len(urls):,} URLs (all phishing=1)")

            feats, chars = [], []
            for url in tqdm(urls, desc=f"  Features [{adv_name}]", mininterval=2.0):
                feats.append(extractor.extract(url))
                chars.append(encode_url(url))
            Xf_adv = scaler.transform(np.vstack(feats).astype(np.float32)).astype(np.float32)
            Xc_adv = np.vstack(chars).astype(np.int64)
            y_adv  = np.ones(len(urls), dtype=np.int64)

            preds = run_inference(model, Xf_adv, Xc_adv, device)
            m     = compute_metrics(y_adv, preds)
            m["Dataset"] = adv_name

            out_csv = os.path.join(OUT_LOGS, f"adversarial_binary_{adv_name}.csv")
            pd.DataFrame({"url": urls, "predicted": preds.tolist(),
                          "correct": (preds == 1).tolist()}).to_csv(out_csv, index=False)

            adv_results.append(m)
            log.info(f"  {adv_name}: acc={m['Accuracy']*100:.2f}%  F1={m['F1_Macro']:.4f}  "
                     f"MCC={m['MCC']:.4f}  FNR={m['FNR']*100:.2f}%  -> {out_csv}")
        except Exception as e:
            log.error(f"  {adv_name} failed: {e}")

    # ── 6. Comparison report ──────────────────────────────────────────────────
    log.info("[6/6] Building comparison report...")
    paper_rows = load_paper_results()

    sep = "=" * 80
    lines = [
        sep,
        "Binary URLPhishNet vs. Sabir et al. (arXiv:2005.08454)",
        f"Model: CharCNN (k=3,5,7) + FeatureMLP | 67 features | {EPOCHS} epochs",
        f"Train data: phishphresh dataset ({n_ben:,} benign + {n_phish:,} phishing = {total:,} total)",
        f"Best epoch: {best_epoch}  |  Best F1: {best_f1:.4f}",
        sep, "",
        "OUR BINARY MODEL RESULTS",
        f"  {'Dataset':<30} {'Accuracy':>9} {'F1-Macro':>9} {'MCC':>8} {'FPR%':>7} {'FNR%':>7}",
        "  " + "-" * 76,
    ]
    for r in adv_results:
        lines.append(
            f"  {r['Dataset']:<30} {r['Accuracy']*100:>8.2f}% "
            f"{r['F1_Macro']:>9.4f} {r['MCC']:>8.4f} "
            f"{r['FPR']*100:>6.2f}% {r['FNR']*100:>6.2f}%"
        )

    lines += ["", "PAPER MODELS — Normal Validation (best per feature set)",
              f"  {'Feature Set':<32} {'Classifier':<8} {'Accuracy':>9} {'F1-Macro':>9}",
              "  " + "-" * 62]
    best_normal: dict[str, dict] = {}
    for r in paper_rows:
        f = r.get("File", "")
        if any(k in f for k in ("Adversar", "kfold", "K_fold")):
            continue
        key = r["Method"]
        if key not in best_normal or float(r.get("Accuracy", 0)) > float(best_normal[key].get("Accuracy", 0)):
            best_normal[key] = r
    for key, r in sorted(best_normal.items(), key=lambda x: -float(x[1].get("Accuracy", 0))):
        lines.append(f"  {key:<32} {str(r.get('Classifier','?')):<8} "
                     f"{float(r.get('Accuracy',0))*100:>8.2f}% "
                     f"{float(r.get('FscoreMacro', r.get('MCC', 0))):>9.4f}")

    lines += ["", "PAPER MODELS — Adversarial Validation (best per feature set)",
              f"  {'Feature Set':<32} {'Classifier':<8} {'Accuracy':>9} {'F1-Macro':>9}",
              "  " + "-" * 62]
    best_adv: dict[str, dict] = {}
    for r in paper_rows:
        if "Adversar" not in r.get("File", ""):
            continue
        key = r["Method"]
        if key not in best_adv or float(r.get("Accuracy", 0)) > float(best_adv[key].get("Accuracy", 0)):
            best_adv[key] = r
    for key, r in sorted(best_adv.items(), key=lambda x: -float(x[1].get("Accuracy", 0))):
        lines.append(f"  {key:<32} {str(r.get('Classifier','?')):<8} "
                     f"{float(r.get('Accuracy',0))*100:>8.2f}% "
                     f"{float(r.get('FscoreMacro', r.get('MCC', 0))):>9.4f}")

    lines += ["", sep,
              "NOTE: FNR = miss rate (% adversarial phishing URLs the model called benign).",
              "      Lower FNR = harder to fool. Paper models train on adversarial data too.",
              "      Our model was NOT adversarially trained — this is a fair cold comparison."]

    report = "\n".join(lines)
    log.info("\n" + report)
    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write(report)
    log.info(f"Report -> {REPORT_TXT}")
    log.info("Done.")
