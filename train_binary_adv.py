#!/usr/bin/env python3
"""
train_binary_adv.py
===================
Adversarially-trained binary URLPhishNet.

Same as train_binary.py but adds the paper's adversarial URLs into the
training set (label=phishing=1). The test set stays as the original
phishphresh split — unchanged — so results are directly comparable.

Key question: does training on adversarial examples make the model robust
to the paper's adversarial attacks, even though our base data is different?

Outputs:
  binary_phishing_adv/models/model_best.pth
  binary_phishing_adv/models/scaler.pkl
  binary_phishing_adv/logs/run.log
  binary_phishing_adv/logs/training_log.csv
  binary_phishing_adv/logs/adversarial_results.txt
  binary_phishing_adv/logs/adversarial_binary_<type>.csv
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

OUT_MODELS = os.path.join(PROJECT_DIR, "binary_phishing_adv", "models")
OUT_LOGS   = os.path.join(PROJECT_DIR, "binary_phishing_adv", "logs")
MODEL_PATH  = os.path.join(OUT_MODELS, "model_best.pth")
SCALER_PATH = os.path.join(OUT_MODELS, "scaler.pkl")
EPOCH_LOG   = os.path.join(OUT_LOGS, "training_log.csv")
REPORT_TXT  = os.path.join(OUT_LOGS, "adversarial_results.txt")

# ── Hyper-parameters (match train_binary.py) ──────────────────────────────────
MAX_URL_LEN    = 256
VOCAB_SIZE     = 98
EMBED_DIM      = 64
CNN_CHANNELS   = 128
FEAT_MLP_DIM   = 256
FEAT_MLP_OUT   = 128
DROPOUT        = 0.40
TEST_SIZE      = 0.20
RANDOM_STATE   = 42
BATCH_SIZE     = 1024
EPOCHS         = 15
LR             = 3e-4
WEIGHT_DECAY   = 3e-4
MAX_PATH_TRAIN = 100_000   # cap path adversary for training (1M is too slow)
MAX_PATH_EVAL  = 50_000

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

# ── Feature extractor (same 67-dim as all other scripts) ──────────────────────
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


def extract_batch(urls: list[str], extractor: URLFeatureExtractorV2, desc: str = "Features"):
    feats, chars = [], []
    for url in tqdm(urls, desc=f"  {desc}", mininterval=2.0):
        feats.append(extractor.extract(url))
        chars.append(encode_url(url))
    return np.vstack(feats).astype(np.float32), np.vstack(chars).astype(np.int16)


# ── Dataset ────────────────────────────────────────────────────────────────────
class URLDataset(Dataset):
    def __init__(self, X_feat, X_char, labels):
        self.Xf = torch.from_numpy(X_feat)
        self.Xc = torch.from_numpy(X_char.astype(np.int64))
        self.y  = torch.from_numpy(labels.astype(np.int64))

    def __len__(self):          return len(self.y)
    def __getitem__(self, i):   return self.Xf[i], self.Xc[i], self.y[i]


# ── Model ──────────────────────────────────────────────────────────────────────
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
    log.info("MODE: Adversarial Training — paper's attack URLs injected into training set")

    # ── 1. Load phishphresh chunks ─────────────────────────────────────────────
    log.info("[1/7] Loading phishphresh chunks...")
    all_chunks = sorted(
        glob.glob(os.path.join(DATA_DIR, "train", "chunk_*.npz")) +
        glob.glob(os.path.join(DATA_DIR, "test",  "chunk_*.npz"))
    )
    feat_parts, char_parts, tgt_parts = [], [], []
    for f in tqdm(all_chunks, desc="  Loading chunks"):
        d = np.load(f, allow_pickle=True)
        feat_parts.append(d["features"])
        char_parts.append(d["char_ids"])
        tgt_parts.extend(d["targets"].tolist())

    X_feat_base = np.vstack(feat_parts).astype(np.float32)
    X_char_base = np.vstack(char_parts).astype(np.int64)
    labels_base = np.array([0 if str(t or "").strip() == "" else 1
                            for t in tgt_parts], dtype=np.int64)
    del feat_parts, char_parts; gc.collect()

    n_base  = len(labels_base)
    n_ben   = (labels_base == 0).sum()
    n_phish = (labels_base == 1).sum()
    log.info(f"  Phishphresh: {n_base:,}  benign={n_ben:,}  phishing={n_phish:,}")

    # ── 2. Same 80/20 split as train_binary.py ────────────────────────────────
    log.info("[2/7] Splitting phishphresh (same RANDOM_STATE=42 split)...")
    idx = np.arange(n_base)
    tr_idx, te_idx = train_test_split(idx, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    Xf_te = X_feat_base[te_idx]
    Xc_te = X_char_base[te_idx]
    y_te  = labels_base[te_idx]

    # Training portion — will be augmented with adversarial URLs
    Xf_tr_base = X_feat_base[tr_idx]
    Xc_tr_base = X_char_base[tr_idx]
    y_tr_base  = labels_base[tr_idx]
    del X_feat_base, X_char_base; gc.collect()
    log.info(f"  Train (base)={len(tr_idx):,}  Test={len(te_idx):,}")

    # ── 3. Load & extract adversarial URLs for training ───────────────────────
    log.info("[3/7] Extracting features for adversarial training URLs...")
    extractor = URLFeatureExtractorV2()

    adv_train_feats, adv_train_chars, adv_counts = [], [], {}

    for adv_path, adv_name, cap in [
        (DOMAIN_ADV, "Domain", None),
        (PATH_ADV,   "Path",   MAX_PATH_TRAIN),
        (TLD_ADV,    "TLD",    None),
    ]:
        df_adv = pd.read_csv(adv_path, usecols=["craftedurl"])
        urls   = df_adv["craftedurl"].dropna().astype(str).tolist()
        if cap and len(urls) > cap:
            # Shuffle before capping so we get variety across seed URLs
            rng = np.random.default_rng(RANDOM_STATE)
            urls = [urls[i] for i in rng.permutation(len(urls))[:cap]]
        log.info(f"  {adv_name}: {len(urls):,} adversarial URLs -> training set")
        Xf_adv, Xc_adv = extract_batch(urls, extractor, desc=f"Extract [{adv_name}]")
        adv_train_feats.append(Xf_adv)
        adv_train_chars.append(Xc_adv.astype(np.int64))
        adv_counts[adv_name] = len(urls)

    Xf_adv_all = np.vstack(adv_train_feats).astype(np.float32)
    Xc_adv_all = np.vstack(adv_train_chars).astype(np.int64)
    y_adv_all  = np.ones(len(Xf_adv_all), dtype=np.int64)
    del adv_train_feats, adv_train_chars; gc.collect()
    log.info(f"  Total adversarial training URLs: {len(y_adv_all):,}")

    # ── 4. Combine base + adversarial training data ───────────────────────────
    log.info("[4/7] Combining phishphresh train + adversarial URLs...")
    Xf_tr = np.vstack([Xf_tr_base, Xf_adv_all]).astype(np.float32)
    Xc_tr = np.vstack([Xc_tr_base, Xc_adv_all]).astype(np.int64)
    y_tr  = np.concatenate([y_tr_base, y_adv_all])
    del Xf_tr_base, Xc_tr_base, Xf_adv_all, Xc_adv_all; gc.collect()

    n_tr_ben   = (y_tr == 0).sum()
    n_tr_phish = (y_tr == 1).sum()
    log.info(f"  Combined train: {len(y_tr):,}  benign={n_tr_ben:,}  phishing={n_tr_phish:,}")

    # ── 5. Scale features ─────────────────────────────────────────────────────
    log.info("[5/7] Fitting StandardScaler on combined training data...")
    scaler = StandardScaler()
    Xf_tr  = scaler.fit_transform(Xf_tr).astype(np.float32)
    Xf_te  = scaler.transform(Xf_te).astype(np.float32)
    joblib.dump(scaler, SCALER_PATH)
    log.info(f"  Scaler saved -> {SCALER_PATH}")

    FEAT_DIM = Xf_tr.shape[1]

    # ── 6. Train ──────────────────────────────────────────────────────────────
    log.info(f"[6/7] Training adversarially-augmented BinaryURLPhishNet ({EPOCHS} epochs)...")
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
    log.info(f"  Best model: epoch {best_epoch}  F1={best_f1:.4f}")

    # ── 7. Adversarial evaluation (same 3 attack sets) ────────────────────────
    log.info("[7/7] Adversarial evaluation...")
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    log.info(f"  Loaded best model from epoch {ckpt['epoch']}")

    # Normal test first
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
            if len(urls) > MAX_PATH_EVAL:
                log.info(f"  {adv_name}: {len(urls):,} URLs, capping to {MAX_PATH_EVAL:,}")
                urls = urls[:MAX_PATH_EVAL]
            else:
                log.info(f"  {adv_name}: {len(urls):,} URLs")

            Xf_adv, Xc_adv = extract_batch(urls, extractor, desc=f"Eval [{adv_name}]")
            Xf_adv = scaler.transform(Xf_adv).astype(np.float32)
            Xc_adv = Xc_adv.astype(np.int64)
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

    # ── Comparison report ──────────────────────────────────────────────────────
    paper_rows = load_paper_results()
    sep = "=" * 80

    lines = [
        sep,
        "Adversarially-Trained Binary URLPhishNet vs. Sabir et al. (arXiv:2005.08454)",
        f"Model: CharCNN (k=3,5,7) + FeatureMLP | 67 features | {EPOCHS} epochs",
        f"Training: phishphresh ({n_base:,}) + adversarial URLs "
        f"(Domain={adv_counts['Domain']:,}, Path={adv_counts['Path']:,}, TLD={adv_counts['TLD']:,})",
        f"Best epoch: {best_epoch}  |  Best F1: {best_f1:.4f}",
        sep, "",
        "OUR ADV-TRAINED MODEL RESULTS  (vs. non-adv-trained binary — for comparison)",
        f"  {'Dataset':<32} {'Accuracy':>9} {'F1-Macro':>9} {'MCC':>8} {'FPR%':>7} {'FNR%':>7}",
        "  " + "-" * 78,
    ]
    prev = {
        "Normal_Test (phishphresh)": (98.68, 0.9867, 0.9734),
        "Domain_Adversary":          (48.23, 0.3254, 0.0000),
        "Path_Adversary":            (23.78, 0.1921, 0.0000),
        "TLD_Adversary":             (43.02, 0.3008, 0.0000),
    }
    for r in adv_results:
        p = prev.get(r["Dataset"], (0, 0, 0))
        delta_acc = r["Accuracy"] * 100 - p[0]
        delta_mcc = r["MCC"] - p[2]
        arrow = f"  [acc {delta_acc:+.1f}%  mcc {delta_mcc:+.4f} vs non-adv]"
        lines.append(
            f"  {r['Dataset']:<32} {r['Accuracy']*100:>8.2f}% "
            f"{r['F1_Macro']:>9.4f} {r['MCC']:>8.4f} "
            f"{r['FPR']*100:>6.2f}% {r['FNR']*100:>6.2f}%{arrow}"
        )

    lines += ["", "PAPER MODELS — Adversarially Trained (best per feature set)",
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
                     f"{float(r.get('FscoreMacro', 0)):>9.4f}")

    lines += ["", sep,
              "KEY FINDING: Does adversarial training on paper's attacks help our model?",
              "Normal test accuracy may drop slightly (model trades some normal accuracy",
              "for robustness). Adversarial FNR should drop from 52-76% toward <10%."]

    report = "\n".join(lines)
    log.info("\n" + report)
    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write(report)
    log.info(f"Report -> {REPORT_TXT}")
    log.info("Done.")
