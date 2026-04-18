#!/usr/bin/env python3
"""
train_multiclass_brand.py
=========================
Multi-class phishing detection: Benign (0) + N brand-specific phishing classes.

Key design decisions
--------------------
* No CSEMapper.  Single end-to-end URLPhishNet handles all detection.
* n+1 output classes: 0 = benign, 1..N = brand being impersonated (phishing).
  Classes are discovered from the dataset's `target` column — never hardcoded.
* Dual-stream architecture:
    Stream A: 67 handcrafted features (51 domain-level from main.py's
              FeatureExtractor + 16 URL-level structural features that are
              provably non-overlapping) → residual MLP (128-dim)
    Stream B: Raw URL chars → TextCNN (k=3/5/7 parallel, max+avg pooling)
              → 768-dim representation
    Fusion  : concat(128, 768) = 896 → classifier → n_classes logits
* No brand/phishing keyword regex in features.  The CharCNN learns those
  directly from character sequences.  Hardcoding them would duplicate signal
  and introduce brittleness when new brands appear.
* Pre-compiled structural regex (IP detection, hex encoding, redirect params)
  is correct — these detect URL structure, not semantic content.
* Focal Loss (Lin et al. ICCV 2017) — down-weights easy examples, focuses
  training on hard misclassifications.  γ=2 with per-class α weights.
* Class-Balanced Loss weights (Cui et al. CVPR 2019) — effective number of
  samples weighting (β=0.9999) instead of naive inverse-frequency.  Avoids
  over-penalising the majority benign class.
* Benign class explicit boost (3×) in both loss weights and sampler — false
  positives (benign→phishing) are more damaging in a security context.
* Partial-rebalancing sampler (√n weighting) — preserves some natural class
  frequency so benign is not crushed to 1/52 during training.
* Mixup augmentation (Zhang et al. ICLR 2018, α=0.2) on handcrafted features
  — smooths decision boundaries and reduces overfitting.
* AdamW + OneCycleLR (10% warmup → cosine decay) — per-batch scheduler.
* AMP (torch.amp) with GradScaler for BF16/FP16 acceleration on RTX 50xx.
* Per-epoch checkpoint in checkpoints/epoch_NNN/model.pth + metrics.json.

Dataset : phreshphish/phreshphish (columns used: url, target)
Storage : phishphresh/multiclass_brand/
"""

from __future__ import annotations

import csv as _csv
import gc
import glob
import json
import logging
import io as _io
import os
import re
import sys
import time
import urllib.parse
from collections import Counter
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

# ── Import base feature extractor (exact 51 features from main.py) ────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from main import FeatureExtractor as _BaseFeatureExtractor

import tldextract  # used by _BaseFeatureExtractor; listed in requirements.txt


# ════════════════════════════════════════════════════════════════════════════════
# ENVIRONMENT & PATHS
# ════════════════════════════════════════════════════════════════════════════════

def _load_dotenv(path: str = os.path.join(PROJECT_DIR, ".env")) -> None:
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


_load_dotenv()
HF_TOKEN     = os.environ.get("HF_TOKEN", "")
DATASET_NAME = "phreshphish/phreshphish"

ROOT      = os.path.join(PROJECT_DIR, "phishphresh", "multiclass_brand")
DATA_DIR  = os.path.join(ROOT, "data")
CKPT_DIR  = os.path.join(ROOT, "checkpoints")
MODEL_DIR = os.path.join(ROOT, "models")
LOG_DIR   = os.path.join(ROOT, "logs")

for _d in [
    os.path.join(DATA_DIR, "train"),
    os.path.join(DATA_DIR, "test"),
    CKPT_DIR, MODEL_DIR, LOG_DIR,
]:
    os.makedirs(_d, exist_ok=True)


# ════════════════════════════════════════════════════════════════════════════════
# HYPERPARAMETERS
# ════════════════════════════════════════════════════════════════════════════════

CHUNK_SIZE        = 10_000   # rows per .npz chunk
MAX_URL_LEN       = 256      # URL truncated / padded to this length
VOCAB_SIZE        = 98       # 0=pad, 1-96=printable ASCII 32-127, 97=unknown
EMBED_DIM         = 64
CNN_CHANNELS      = 128      # channels per kernel; CharCNN out = 2 × 3 × 128 = 768
FEAT_MLP_DIM      = 256      # projection dimension before residual blocks
FEAT_MLP_OUT      = 128      # output dim of FeatureMLP
MIN_CLASS_SAMPLES = 30       # brands below this threshold → "other_phishing"
TOP_N_BRANDS      = 50       # keep only top-N brands as separate classes; rest → "other_phishing"
MAX_SAMPLES       = None     # None = full dataset

EPOCHS         = 60
BATCH_SIZE     = 512
LR             = 5e-4
WEIGHT_DECAY   = 3e-4        # ↑ from 1e-4 — stronger regularisation
DROPOUT        = 0.40        # ↑ from 0.35 — reduce overfitting
TEST_SIZE      = 0.20
RANDOM_STATE   = 42
NUM_WORKERS    = 0           # 0 required on Windows

# ── Class imbalance & augmentation ───────────────────────────────────────────
FOCAL_GAMMA    = 2.0         # Focal Loss γ (Lin et al. ICCV 2017)
EFF_NUM_BETA   = 0.9999      # Effective-number β (Cui et al. CVPR 2019)
BENIGN_BOOST   = 1.5         # Extra weight multiplier for benign class
MIXUP_ALPHA    = 0.2         # Mixup Beta-distribution α (Zhang et al. ICLR 2018)


# ════════════════════════════════════════════════════════════════════════════════
# LOGGING
# ════════════════════════════════════════════════════════════════════════════════

_ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = os.path.join(LOG_DIR, f"training_{_ts}.log")

_con = (
    _io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                      errors="replace", line_buffering=True)
    if hasattr(sys.stdout, "buffer") else sys.stdout
)
_fmt = logging.Formatter(
    "%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_ch = logging.StreamHandler(_con);  _ch.setFormatter(_fmt)
_fh = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"); _fh.setFormatter(_fmt)
logging.basicConfig(level=logging.INFO, handlers=[_fh, _ch])
logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ════════════════════════════════════════════════════════════════════════════════
#
# Base extractor (from main.py) produces 51 domain-level features operating on
# the hostname only (via tldextract).  It knows nothing about the URL path,
# query string, scheme, port, or fragment.
#
# URLFeatureExtractorV2 appends 16 URL-level structural features that are
# provably disjoint from the 51 base features:
#
#   Base 51 — all domain-level (from main.py FeatureExtractor):
#     domain_length, dot_count, dash_count, underscore_count, digit_count,
#     uppercase_count, digit_ratio, special_char_ratio, domain_parts,
#     longest_part, shortest_part, avg_part_length, vowel_count,
#     consonant_count, has_consecutive_chars,
#     char_entropy, part_length_variance, part_length_std, bigram_entropy,
#     unique_char_ratio, transition_entropy, trigram_count, unique_trigram_ratio,
#     vowel_consonant_ratio, consonant_clusters,
#     tld_length, is_common_tld, is_country_tld, is_suspicious_tld, is_indian_tld,
#     legitimate_domain_similarity, matches_legitimate_domain,
#     legitimate_keyword_count, contains_legitimate_keyword,
#     brand_keyword_count, has_brand_variation,
#     finance_keywords, gov_keywords, telecom_keywords, tech_keywords,
#     dictionary_word_count, char_repetition_score, has_year_pattern,
#     starts_with_number, ends_with_number, max_numeric_sequence,
#     has_subdomain, subdomain_count, suspicious_pattern_count,
#     has_homograph, complexity_score
#
#   Extra 16 — all URL-level structural (this file only):
#     total_url_len_norm  — len(full_url) / 2000
#     path_len_norm       — len(path) / 500
#     query_len_norm      — len(query) / 500
#     fragment_len_norm   — len(fragment) / 200
#     path_depth          — count of '/' in path
#     num_query_params    — number of key=value pairs in query
#     is_https            — scheme == "https"
#     has_explicit_port   — non-standard port present (not 80/443)
#     has_at_in_netloc    — '@' in netloc (pre-host redirect trick)
#     is_ip_host          — host is a raw IP address
#     hex_encoding_ratio  — count(%XX) / max(len(url), 1)
#     has_redirect_param  — redirect/return/url=/next= in query string
#     double_slash_in_path— '//' appears in path (≠ scheme separator)
#     url_full_entropy    — Shannon entropy of entire URL string
#     query_entropy       — Shannon entropy of query string
#     path_entropy        — Shannon entropy of path string
#
#   NOTE: No brand names, phishing keywords, or sector-specific terms are
#   used in the extra features.  The CharCNN learns those directly from URL 
#   character sequences — hardcoding them would duplicate learned signal and
#   break when new brands appear.
# ────────────────────────────────────────────────────────────────────────────────

# Structural regex only — no keywords, no brand names
_RE_IP       = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
_RE_HEX      = re.compile(r"%[0-9A-Fa-f]{2}")
_RE_REDIRECT = re.compile(
    r"(?:^|&)(?:redirect|return|next|url|continue)=https?", re.IGNORECASE
)
_EXTRA_DIM   = 16


def _shannon(s: str) -> float:
    """Shannon entropy of a string (bits per character)."""
    if not s:
        return 0.0
    freq = np.fromiter(Counter(s).values(), dtype=np.float32)
    freq /= freq.sum()
    return float(-(freq * np.log2(freq + 1e-10)).sum())


class URLFeatureExtractorV2:
    """
    Extends main.py's FeatureExtractor with 16 non-overlapping URL-level
    structural features.  Total: base_dim + 16 features.
    """

    def __init__(self) -> None:
        self._base = _BaseFeatureExtractor()
        # Probe actual base dimension (guards against future changes in main.py)
        try:
            _probe = list(self._base.extract_features("http://example.com").values())
            self._base_dim = len(_probe)
        except Exception:
            self._base_dim = 51
        logger.info(f"Base FeatureExtractor dim: {self._base_dim} "
                    f"(expected 51)")

    @property
    def n_features(self) -> int:
        return self._base_dim + _EXTRA_DIM

    def extract(self, url: str) -> np.ndarray:
        url = str(url or "").strip()

        # ── 51 base domain-level features ─────────────────────────────────────
        try:
            vals = list(self._base.extract_features(url).values())
            if len(vals) < self._base_dim:
                vals += [0.0] * (self._base_dim - len(vals))
            base = np.array(vals[: self._base_dim], dtype=np.float32)
        except Exception:
            base = np.zeros(self._base_dim, dtype=np.float32)

        # ── 16 URL-level structural features ──────────────────────────────────
        extra = self._url_structural(url)
        return np.concatenate([base, extra])

    @staticmethod
    def _url_structural(url: str) -> np.ndarray:
        """
        Purely structural URL features — no semantic content, no keywords.
        All disjoint from the domain-level base features.
        """
        try:
            parsed = urllib.parse.urlparse(
                url if "://" in url else "http://" + url
            )
        except Exception:
            return np.zeros(_EXTRA_DIM, dtype=np.float32)

        path     = parsed.path   or ""
        query    = parsed.query  or ""
        fragment = parsed.fragment or ""
        netloc   = parsed.netloc or ""
        scheme   = (parsed.scheme or "http").lower()

        # Extract host without port for IP check
        host = netloc.split("@")[-1].split(":")[0]

        total_len_norm  = min(len(url),      2000) / 2000.0
        path_len_norm   = min(len(path),      500) / 500.0
        query_len_norm  = min(len(query),     500) / 500.0
        fragment_norm   = min(len(fragment),  200) / 200.0
        path_depth      = float(path.count("/"))
        num_q_params    = float(len(urllib.parse.parse_qs(query)))
        is_https        = float(scheme == "https")
        has_port        = float(
            bool(parsed.port and parsed.port not in (80, 443))
        )
        has_at          = float("@" in netloc)
        is_ip           = float(bool(_RE_IP.fullmatch(host)))
        hex_ratio       = len(_RE_HEX.findall(url)) / max(len(url), 1)
        has_redirect    = float(bool(_RE_REDIRECT.search(query)))
        double_slash    = float("//" in path)          # path only, not scheme
        url_entropy     = _shannon(url)
        query_entropy   = _shannon(query)
        path_entropy    = _shannon(path)

        return np.array([
            total_len_norm, path_len_norm, query_len_norm, fragment_norm,
            path_depth, num_q_params,
            is_https, has_port, has_at, is_ip, hex_ratio,
            has_redirect, double_slash,
            url_entropy, query_entropy, path_entropy,
        ], dtype=np.float32)


# ── Probe once at import time so FEATURE_DIM is available for model init ───────
_probe_ext  = URLFeatureExtractorV2()
FEATURE_DIM = _probe_ext.n_features   # 67 = 51 + 16
del _probe_ext
logger.info(f"FEATURE_DIM = {FEATURE_DIM}  (base={FEATURE_DIM - _EXTRA_DIM}, extra={_EXTRA_DIM})")


# ════════════════════════════════════════════════════════════════════════════════
# CHARACTER ENCODING
# ════════════════════════════════════════════════════════════════════════════════

def encode_url(url: str, max_len: int = MAX_URL_LEN) -> np.ndarray:
    """
    Map URL characters to indices for the Embedding layer.
      0        → padding (unused positions)
      1-96     → printable ASCII (ord 32-127)
      97       → non-printable / non-ASCII character
    """
    arr = np.zeros(max_len, dtype=np.int16)
    for i, ch in enumerate(url[:max_len]):
        code = ord(ch)
        arr[i] = (code - 31) if 32 <= code < 128 else 97
    return arr


# ════════════════════════════════════════════════════════════════════════════════
# CHUNK HELPERS
# ════════════════════════════════════════════════════════════════════════════════

def _chunk_path(split: str, idx: int) -> str:
    return os.path.join(DATA_DIR, split, f"chunk_{idx:05d}.npz")

def _flag_path(split: str) -> str:
    return os.path.join(DATA_DIR, f"_{split}_complete.flag")

def _find_chunks(split: str) -> list:
    return sorted(glob.glob(os.path.join(DATA_DIR, split, "chunk_*.npz")))

def _count_samples(chunks: list) -> int:
    total = 0
    for f in chunks:
        try:
            total += len(np.load(f, allow_pickle=True)["targets"])
        except Exception as exc:
            logger.warning(f"Unreadable chunk {f}: {exc}")
    return total


# ════════════════════════════════════════════════════════════════════════════════
# STEP 1 — STREAM & EXTRACT
# ════════════════════════════════════════════════════════════════════════════════

def stream_and_extract(extractor: URLFeatureExtractorV2, split: str) -> list:
    """
    Stream `split` from HuggingFace, extract features + char encoding per row,
    save compressed .npz chunks.  Fully resumable.

    Chunk layout:
      features  float32  (N, FEATURE_DIM)   handcrafted URL features
      char_ids  int16    (N, MAX_URL_LEN)   character indices
      targets   object   (N,)               brand name string / "" for benign
    """
    flag     = _flag_path(split)
    existing = _find_chunks(split)

    if os.path.exists(flag):
        logger.info(f"[{split}] Already complete — {len(existing)} chunks, skipping stream.")
        return existing

    already  = _count_samples(existing)
    nxt_idx  = len(existing)
    if already:
        logger.info(f"[{split}] Resuming from {already:,} rows ({len(existing)} chunks).")

    logger.info(f"[{split}] Loading HuggingFace stream (url + target)…")
    ds = load_dataset(
        DATASET_NAME, split=split, streaming=True,
        token=HF_TOKEN or None,
    ).select_columns(["url", "target"])

    if already > 0:
        logger.info(f"[{split}] Skipping {already:,} already-processed rows…")
        ds = ds.skip(already)

    buf_feat:  list[np.ndarray] = []
    buf_char:  list[np.ndarray] = []
    buf_tgt:   list[str]        = []
    chunk_files                 = list(existing)
    n_done   = 0
    complete = False
    t0       = time.time()

    def flush() -> None:
        nonlocal nxt_idx, buf_feat, buf_char, buf_tgt
        if not buf_feat:
            return
        path = _chunk_path(split, nxt_idx)
        np.savez_compressed(
            path,
            features = np.vstack(buf_feat).astype(np.float32),
            char_ids = np.vstack(buf_char).astype(np.int16),
            targets  = np.array(buf_tgt, dtype=object),
        )
        logger.info(
            f"  [{split}] chunk #{nxt_idx:05d} saved "
            f"({len(buf_tgt):,} rows | running total ≈ {already + n_done:,})"
        )
        chunk_files.append(path)
        nxt_idx += 1
        buf_feat.clear(); buf_char.clear(); buf_tgt.clear()
        gc.collect()

    try:
        for sample in ds:
            if MAX_SAMPLES is not None and (already + n_done) >= MAX_SAMPLES:
                logger.info(f"[{split}] MAX_SAMPLES={MAX_SAMPLES:,} reached.")
                break

            url    = str(sample.get("url")    or "").strip()
            target = str(sample.get("target") or "").strip()
            if target.lower() in {"none", "nan", "null", ""}:
                target = ""  # normalise benign

            buf_feat.append(extractor.extract(url))
            buf_char.append(encode_url(url))
            buf_tgt.append(target)
            n_done += 1

            if len(buf_feat) >= CHUNK_SIZE:
                flush()

            if n_done % 50_000 == 0:
                rate = n_done / max(time.time() - t0, 1)
                logger.info(
                    f"  [{split}] {already + n_done:,} rows processed | {rate:.0f} rows/s"
                )

        complete = True

    except KeyboardInterrupt:
        logger.warning(f"[{split}] Interrupted — flushing partial buffer.")
    except Exception as exc:
        logger.critical(f"[{split}] Stream error: {exc}", exc_info=True)

    flush()
    logger.info(
        f"[{split}] Done. +{n_done:,} new rows, {len(chunk_files)} total chunks."
    )
    if complete:
        open(flag, "w").close()
        logger.info(f"[{split}] Complete flag written.")

    return chunk_files


# ════════════════════════════════════════════════════════════════════════════════
# STEP 2 — LOAD ALL CHUNKS
# ════════════════════════════════════════════════════════════════════════════════

def load_chunks(chunks: list) -> tuple[np.ndarray, np.ndarray, list]:
    feat_parts, char_parts, tgt_parts = [], [], []
    for f in tqdm(chunks, desc="Loading chunks", unit="file"):
        try:
            d = np.load(f, allow_pickle=True)
            feat_parts.append(d["features"])
            char_parts.append(d["char_ids"])
            tgt_parts.extend(d["targets"].tolist())
        except Exception as exc:
            logger.error(f"Cannot load {f}: {exc}")

    X_feat = np.vstack(feat_parts).astype(np.float32)
    # int64 required by nn.Embedding; int16 on disk saves space
    X_char = np.vstack(char_parts).astype(np.int64)
    logger.info(
        f"Loaded: X_feat={X_feat.shape}  X_char={X_char.shape}  targets={len(tgt_parts):,}"
    )
    return X_feat, X_char, tgt_parts


# ════════════════════════════════════════════════════════════════════════════════
# STEP 3 — CLASS MAP
# ════════════════════════════════════════════════════════════════════════════════

def build_class_map(
    targets: list[str],
    min_samples: int = MIN_CLASS_SAMPLES,
    top_n: int = TOP_N_BRANDS,
) -> tuple[dict, int, list[str]]:
    """
    Returns:
      class_map   dict[str, int]   "" → 0 (benign); brand → 1..top_n; else → other_class
      other_class int              index for all non-top-N phishing
      class_names list[str]        human-readable name per index
    Strategy: top_n brands by frequency get their own class; everything else
    (rare brands AND brands below min_samples) → "other_phishing".
    """
    counter  = Counter(t for t in targets if t)
    n_benign = sum(1 for t in targets if not t)

    logger.info(f"Benign samples (empty target) : {n_benign:,}")
    logger.info(f"Unique brand targets found    : {len(counter)}")

    # top_n by frequency (must also meet min_samples)
    top_brands = [b for b, c in counter.most_common(top_n) if c >= min_samples]

    other_brands = [b for b in counter if b not in set(top_brands)]
    other_total  = sum(counter[b] for b in other_brands)

    logger.info(f"\nTop-{top_n} brands (own class each):")
    for rank, brand in enumerate(top_brands, 1):
        pct = 100 * counter[brand] / len(targets)
        logger.info(f"  {rank:>3}  {brand:<45s}  {counter[brand]:>8,}  {pct:6.3f}%")

    logger.info(
        f"\n  Remaining {len(other_brands)} brands → 'other_phishing' : "
        f"{other_total:,} samples"
    )

    class_map: dict[str, int] = {"": 0}
    for idx, brand in enumerate(top_brands, start=1):
        class_map[brand] = idx

    other_class = len(top_brands) + 1
    class_names = ["benign"] + top_brands + ["other_phishing"]

    return class_map, other_class, class_names


def targets_to_labels(
    targets: list[str],
    class_map: dict[str, int],
    other_class: int,
) -> np.ndarray:
    y = np.zeros(len(targets), dtype=np.int64)
    for i, t in enumerate(targets):
        t = str(t or "").strip()
        if t in class_map:
            y[i] = class_map[t]
        elif t:                               # phishing brand not in top-N → other_phishing
            y[i] = other_class
    return y


# ════════════════════════════════════════════════════════════════════════════════
# TORCH DATASET
# ════════════════════════════════════════════════════════════════════════════════

class URLDataset(Dataset):
    def __init__(
        self,
        X_feat: np.ndarray,
        X_char: np.ndarray,
        y:      np.ndarray,
    ) -> None:
        self.Xf = torch.from_numpy(X_feat)
        self.Xc = torch.from_numpy(X_char.astype(np.int64))
        self.y  = torch.from_numpy(y.astype(np.int64))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.Xf[idx], self.Xc[idx], self.y[idx]


# ════════════════════════════════════════════════════════════════════════════════
# MODEL COMPONENTS
# ════════════════════════════════════════════════════════════════════════════════

class _ResBlock(nn.Module):
    """Pre-activation residual block for the feature MLP."""

    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)           # residual skip


class FeatureMLP(nn.Module):
    """
    Residual MLP for the 67 handcrafted features.

    Architecture:
      Input(67) → Linear(256) + LayerNorm + GELU
      → ResBlock(256) × 2
      → Linear(128) + LayerNorm + GELU + Dropout
    """

    def __init__(
        self,
        input_dim: int,
        proj_dim:  int   = FEAT_MLP_DIM,
        out_dim:   int   = FEAT_MLP_OUT,
        dropout:   float = DROPOUT,
    ) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
        )
        self.res_blocks = nn.Sequential(
            _ResBlock(proj_dim, dropout),
            _ResBlock(proj_dim, dropout),
        )
        self.out = nn.Sequential(
            nn.Linear(proj_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.output_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.res_blocks(self.proj(x)))


class CharCNN(nn.Module):
    """
    TextCNN (Kim 2014) applied at character level with max+avg pooling.

    Parallel Conv1d (k=3, 5, 7) each producing CNN_CHANNELS feature maps,
    concatenated to 3×CNN_CHANNELS, followed by a deep conv and global
    max+average pooling (concatenated).

    Output: (B, 2 × 3 × CNN_CHANNELS)  =  (B, 768)
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        embed_dim:  int = EMBED_DIM,
        n_channels: int = CNN_CHANNELS,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.conv3 = nn.Conv1d(embed_dim, n_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(embed_dim, n_channels, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(embed_dim, n_channels, kernel_size=7, padding=3)
        self.bn3   = nn.BatchNorm1d(n_channels)
        self.bn5   = nn.BatchNorm1d(n_channels)
        self.bn7   = nn.BatchNorm1d(n_channels)

        fused = 3 * n_channels                          # 384
        self.conv_deep = nn.Conv1d(fused, fused, kernel_size=3, padding=1)
        self.bn_deep   = nn.BatchNorm1d(fused)
        self.drop      = nn.Dropout(0.15)

        # max-pool + avg-pool concatenated → 2 × fused output
        self.output_dim = 2 * fused                     # 768

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len)  int64
        emb = self.embedding(x).transpose(1, 2)         # (B, E, L)
        c3  = F.relu(self.bn3(self.conv3(emb)))
        c5  = F.relu(self.bn5(self.conv5(emb)))
        c7  = F.relu(self.bn7(self.conv7(emb)))
        cat = torch.cat([c3, c5, c7], dim=1)            # (B, 3C, L)
        feat = F.relu(self.bn_deep(self.conv_deep(cat))) # (B, 3C, L)
        feat = self.drop(feat)

        # Global max+avg pooling (both capture different statistics)
        g_max = F.adaptive_max_pool1d(feat, 1).squeeze(-1)   # (B, 3C)
        g_avg = F.adaptive_avg_pool1d(feat, 1).squeeze(-1)   # (B, 3C)
        return torch.cat([g_max, g_avg], dim=1)              # (B, 6C = 768)


class URLPhishNet(nn.Module):
    """
    Dual-stream multi-class URL phishing classifier.

    Inputs:
      features  (B, FEATURE_DIM)  float32   67 handcrafted URL features
      char_ids  (B, MAX_URL_LEN)  int64     character indices

    Output:
      logits    (B, n_classes)              raw class scores (no softmax)
    """

    def __init__(
        self,
        feature_dim: int,
        n_classes:   int,
        dropout:     float = DROPOUT,
        vocab_size:  int   = VOCAB_SIZE,
        embed_dim:   int   = EMBED_DIM,
        cnn_ch:      int   = CNN_CHANNELS,
    ) -> None:
        super().__init__()
        self.feat_mlp = FeatureMLP(feature_dim, FEAT_MLP_DIM, FEAT_MLP_OUT, dropout)
        self.char_cnn = CharCNN(vocab_size, embed_dim, cnn_ch)

        fusion = self.feat_mlp.output_dim + self.char_cnn.output_dim  # 128 + 768 = 896

        self.classifier = nn.Sequential(
            nn.Linear(fusion, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout * 0.7),
            nn.Linear(256, n_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Embedding):
                nn.init.uniform_(m.weight, -0.05, 0.05)
                m.weight.data[0].zero_()      # padding idx stays zero

    def forward(
        self,
        features: torch.Tensor,
        char_ids: torch.Tensor,
    ) -> torch.Tensor:
        f = self.feat_mlp(features)                        # (B, 128)
        c = self.char_cnn(char_ids)                        # (B, 768)
        return self.classifier(torch.cat([f, c], dim=1))  # (B, n_classes)


# ════════════════════════════════════════════════════════════════════════════════
# TRAINING
# ════════════════════════════════════════════════════════════════════════════════

BEST_MODEL_PATH  = os.path.join(MODEL_DIR, "model_best.pth")
FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "model_final.pth")


# ════════════════════════════════════════════════════════════════════════════════
# FOCAL LOSS  (Lin et al. ICCV 2017)
# ════════════════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss with per-class α weights.

    FL(p_t) = -α_t · (1 − p_t)^γ · log(p_t)

    • γ > 0 down-weights easy examples (high p_t) so the model focuses on
      hard / misclassified samples — critical for severe class imbalance.
    • α_t are per-class weights computed via Effective Number of Samples
      (Cui et al. CVPR 2019) with an explicit boost for the benign class.
    """

    def __init__(self, alpha: torch.Tensor, gamma: float = FOCAL_GAMMA) -> None:
        super().__init__()
        self.register_buffer("alpha", alpha.float())
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Per-sample CE loss weighted by α (uses the registered buffer)
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        # Probability assigned to the true class
        pt = torch.exp(-ce)
        return ((1.0 - pt) ** self.gamma * ce).mean()


# ════════════════════════════════════════════════════════════════════════════════
# MIXUP AUGMENTATION  (Zhang et al. ICLR 2018)
# ════════════════════════════════════════════════════════════════════════════════

def mixup_features(
    Xf: torch.Tensor,
    y:  torch.Tensor,
    alpha: float = MIXUP_ALPHA,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Mixup on the handcrafted feature vector only.

    Character IDs are discrete (token indices), so they cannot be linearly
    mixed.  Mixing the continuous feature branch alone already smooths
    decision boundaries and reduces overfitting without corrupting char IDs.

    Returns: (Xf_mixed, y_a, y_b, λ)
    Loss = λ · L(out, y_a) + (1−λ) · L(out, y_b)
    """
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    idx = torch.randperm(Xf.size(0), device=Xf.device)
    return lam * Xf + (1.0 - lam) * Xf[idx], y, y[idx], lam


def _build_sampler_and_weights(
    y_tr: np.ndarray,
    n_classes: int,
    device: torch.device,
    beta: float = EFF_NUM_BETA,
    benign_boost: float = BENIGN_BOOST,
) -> tuple[WeightedRandomSampler, torch.Tensor]:
    """
    Loss weights : Effective Number of Samples (Cui et al. CVPR 2019).
      EN_i = (1 − β^n_i) / (1 − β)
      w_i  = 1 / EN_i,  then normalised and multiplied by benign_boost for class 0.

    Sampler      : √n partial rebalancing.
      Pure inverse-frequency makes benign (55 % of data) only 1/52 of batches.
      √n weighting preserves benign dominance while still upsampling rare brands.
    """
    counts = np.bincount(y_tr, minlength=n_classes).astype(np.float64)
    counts = np.maximum(counts, 1)  # guard against empty classes

    # ── Effective Number of Samples loss weights ───────────────────────────────
    eff_num = (1.0 - np.power(beta, counts)) / (1.0 - beta)
    weights = 1.0 / eff_num
    weights[0] *= benign_boost          # boost benign — false positives are costly
    weights = (weights / weights.sum() * n_classes).astype(np.float32)
    wt_tensor = torch.tensor(weights, dtype=torch.float32, device=device)

    # ── √n partial-rebalancing sampler ────────────────────────────────────────
    sqrt_counts  = np.sqrt(counts)
    sample_probs = sqrt_counts / sqrt_counts.sum()
    sample_wt    = torch.tensor(sample_probs[y_tr], dtype=torch.double)
    sampler      = WeightedRandomSampler(sample_wt, len(y_tr), replacement=True)

    logger.info("Class weights (Cui et al. effective-number, top 5 by weight):")
    top5 = np.argsort(-weights)[:5]
    for ci in top5:
        logger.info(f"  class {ci:3d}  weight={weights[ci]:.4f}  n={int(counts[ci]):,}")

    return sampler, wt_tensor


def train_model(
    X_feat_tr: np.ndarray, X_char_tr: np.ndarray, y_tr: np.ndarray,
    X_feat_te: np.ndarray, X_char_te: np.ndarray, y_te: np.ndarray,
    n_classes:   int,
    class_names: list[str],
    device:      torch.device,
) -> nn.Module:

    train_ds = URLDataset(X_feat_tr, X_char_tr, y_tr)
    test_ds  = URLDataset(X_feat_te, X_char_te, y_te)

    sampler, class_wt = _build_sampler_and_weights(y_tr, n_classes, device)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"),
    )

    model = URLPhishNet(
        feature_dim=FEATURE_DIM,
        n_classes=n_classes,
        dropout=DROPOUT,
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        cnn_ch=CNN_CHANNELS,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"URLPhishNet — {n_params:,} trainable parameters")
    logger.info(
        f"  feature_dim={FEATURE_DIM}  cnn_ch={CNN_CHANNELS}  "
        f"cnn_out={model.char_cnn.output_dim}  "
        f"n_classes={n_classes}  device={device}"
    )

    # Focal Loss (Lin et al. 2017) with Effective-Number α weights
    criterion = FocalLoss(alpha=class_wt, gamma=FOCAL_GAMMA)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # OneCycleLR: linear warmup (10%) → cosine decay — best for stable multi-class
    total_steps = EPOCHS * len(train_loader)
    scheduler   = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LR,
        total_steps=total_steps,
        pct_start=0.10,
        anneal_strategy="cos",
        div_factor=10,       # initial_lr = LR / 10
        final_div_factor=100,
    )

    # AMP: bfloat16 on Blackwell (RTX 50xx), float16 on older GPUs
    use_amp   = device.type == "cuda"
    amp_dtype = torch.bfloat16 if (
        use_amp and torch.cuda.is_bf16_supported()
    ) else torch.float16
    scaler    = torch.amp.GradScaler("cuda", enabled=use_amp)
    logger.info(f"AMP: {'ON (' + str(amp_dtype) + ')' if use_amp else 'OFF'}")

    best_f1 = 0.0
    log_rows: list[dict] = []

    logger.info(
        f"Train={len(y_tr):,}  Test={len(y_te):,}  "
        f"Epochs={EPOCHS}  Batch={BATCH_SIZE}  LR={LR}  "
        f"FocalGamma={FOCAL_GAMMA}  BenignBoost={BENIGN_BOOST}x  "
        f"MixupAlpha={MIXUP_ALPHA}"
    )
    logger.info("=" * 80)

    for epoch in range(1, EPOCHS + 1):
        t_epoch = time.time()

        # ── Train ──────────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for Xf, Xc, yb in train_loader:
            Xf, Xc, yb = Xf.to(device), Xc.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)

            # Mixup on handcrafted features (char IDs stay original)
            Xf_mix, y_a, y_b, lam = mixup_features(Xf, yb, MIXUP_ALPHA)

            with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                out  = model(Xf_mix, Xc).float()
                loss = lam * criterion(out, y_a) + (1.0 - lam) * criterion(out, y_b)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss += loss.item()

        avg_train = train_loss / max(len(train_loader), 1)

        # ── Eval ───────────────────────────────────────────────────────────────
        model.eval()
        test_loss  = 0.0
        preds_all: list[int] = []
        true_all:  list[int] = []

        with torch.no_grad():
            for Xf, Xc, yb in test_loader:
                Xf, Xc, yb = Xf.to(device), Xc.to(device), yb.to(device)
                with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                    out = model(Xf, Xc).float()
                test_loss += criterion(out, yb).item()
                preds_all.extend(out.argmax(1).cpu().tolist())
                true_all.extend(yb.cpu().tolist())

        avg_test = test_loss / max(len(test_loader), 1)
        acc      = accuracy_score(true_all, preds_all)
        # macro F1 treats all classes equally — more meaningful than accuracy
        # when classes are highly imbalanced
        f1_macro = f1_score(true_all, preds_all, average="macro", zero_division=0)
        lr_now   = scheduler.get_last_lr()[0]
        elapsed  = time.time() - t_epoch

        # ── Per-epoch checkpoint (always saved) ────────────────────────────────
        epoch_dir = os.path.join(CKPT_DIR, f"epoch_{epoch:03d}")
        os.makedirs(epoch_dir, exist_ok=True)
        torch.save(
            {
                "epoch":           epoch,
                "model_state":     model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "train_loss":      avg_train,
                "test_loss":       avg_test,
                "accuracy":        acc,
                "f1_macro":        f1_macro,
                "n_classes":       n_classes,
                "feature_dim":     FEATURE_DIM,
                "class_names":     class_names,
            },
            os.path.join(epoch_dir, "model.pth"),
        )

        # ── Full per-class accuracy ────────────────────────────────────────────
        true_arr  = np.array(true_all)
        pred_arr  = np.array(preds_all)
        counts    = np.bincount(true_arr, minlength=n_classes)

        per_class_rows = []
        for ci in range(n_classes):
            n_true = int(counts[ci])
            if n_true == 0:
                continue
            mask    = true_arr == ci
            n_corr  = int((pred_arr[mask] == ci).sum())
            cname   = class_names[ci] if ci < len(class_names) else f"cls{ci}"
            per_class_rows.append({
                "class_index": ci,
                "class_name":  cname,
                "n_samples":   n_true,
                "n_correct":   n_corr,
                "accuracy":    round(n_corr / n_true, 4),
            })

        # Save per-class CSV for this epoch
        class_acc_csv = os.path.join(epoch_dir, "class_accuracy.csv")
        with open(class_acc_csv, "w", newline="", encoding="utf-8") as fh:
            w = _csv.DictWriter(fh, fieldnames=["class_index", "class_name", "n_samples", "n_correct", "accuracy"])
            w.writeheader()
            w.writerows(per_class_rows)

        # Log top-10 worst-performing classes to spot problems early
        worst10 = sorted(per_class_rows, key=lambda r: r["accuracy"])[:10]
        logger.info("  Per-class accuracy (10 worst):")
        for r in worst10:
            logger.info(f"    [{r['class_index']:3d}] {r['class_name']:<40s}  "
                        f"{r['n_correct']:>6}/{r['n_samples']:<6}  {r['accuracy']:.4f}")

        # Summarise into metrics.json (full list + overall stats)
        per_class_acc_dict = {r["class_name"]: r["accuracy"] for r in per_class_rows}
        metrics = {
            "epoch":          epoch,
            "train_loss":     round(avg_train,  6),
            "test_loss":      round(avg_test,   6),
            "accuracy":       round(acc,        6),
            "f1_macro":       round(f1_macro,   6),
            "lr":             lr_now,
            "elapsed_s":      round(elapsed,    1),
            "per_class_acc":  per_class_acc_dict,
        }
        with open(os.path.join(epoch_dir, "metrics.json"), "w") as fh:
            json.dump(metrics, fh, indent=2)

        # ── Best model (by macro F1 — correct metric for imbalanced multi-class) ─
        is_best = f1_macro > best_f1
        if is_best:
            best_f1 = f1_macro
            torch.save(
                {
                    "epoch":       epoch,
                    "model_state": model.state_dict(),
                    "accuracy":    acc,
                    "f1_macro":    f1_macro,
                    "n_classes":   n_classes,
                    "feature_dim": FEATURE_DIM,
                    "class_names": class_names,
                },
                BEST_MODEL_PATH,
            )

        tag = " *** BEST ***" if is_best else ""
        logger.info(
            f"Epoch {epoch:3d}/{EPOCHS} | "
            f"TLoss={avg_train:.4f} | "
            f"VLoss={avg_test:.4f} | "
            f"Acc={acc:.4f} | "
            f"F1={f1_macro:.4f} | "
            f"LR={lr_now:.2e} | "
            f"{elapsed:.1f}s{tag}"
        )

        log_rows.append({k: v for k, v in metrics.items() if k != "per_class_acc"})

    # ── Save final model & metrics CSV ────────────────────────────────────────
    torch.save(
        {
            "epoch":       EPOCHS,
            "model_state": model.state_dict(),
            "accuracy":    acc,
            "f1_macro":    f1_macro,
            "n_classes":   n_classes,
            "feature_dim": FEATURE_DIM,
            "class_names": class_names,
        },
        FINAL_MODEL_PATH,
    )
    pd.DataFrame(log_rows).to_csv(
        os.path.join(LOG_DIR, "training_metrics.csv"), index=False
    )
    logger.info(f"Training complete. Best val macro-F1: {best_f1:.4f}")

    # ── Final eval on best checkpoint ─────────────────────────────────────────
    logger.info("=" * 80)
    logger.info("FINAL EVALUATION — best checkpoint")
    ckpt = torch.load(BEST_MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    preds_all, true_all = [], []
    with torch.no_grad():
        for Xf, Xc, yb in test_loader:
            Xf, Xc = Xf.to(device), Xc.to(device)
            with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                out = model(Xf, Xc)
            preds_all.extend(out.argmax(1).cpu().tolist())
            true_all.extend(yb.tolist())

    final_acc     = accuracy_score(true_all, preds_all)
    final_f1      = f1_score(true_all, preds_all, average="macro", zero_division=0)
    present_cls   = sorted(set(true_all))
    names_present = [
        class_names[i] if i < len(class_names) else f"class_{i}"
        for i in present_cls
    ]
    report = classification_report(
        true_all, preds_all,
        labels=present_cls,          # integer class indices (must match y dtype)
        target_names=names_present,  # human-readable names aligned to labels
        zero_division=0,
        digits=4,
    )
    logger.info(f"Final Test Accuracy : {final_acc:.4f} ({final_acc * 100:.2f}%)")
    logger.info(f"Final Macro F1      : {final_f1:.4f}")
    logger.info(f"\n{report}")

    report_path = os.path.join(LOG_DIR, "final_report.txt")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(f"Best epoch    : {ckpt['epoch']}\n")
        fh.write(f"Test accuracy : {final_acc:.4f} ({final_acc * 100:.2f}%)\n")
        fh.write(f"Macro F1      : {final_f1:.4f}\n")
        fh.write(f"n_classes     : {n_classes}\n\n")
        fh.write(report)
    logger.info(f"Report saved → {report_path}")

    return model


# ════════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("train_multiclass_brand.py — started")
    logger.info(f"Storage root : {ROOT}")
    logger.info(f"Log file     : {LOG_FILE}")
    logger.info(
        f"Config: EPOCHS={EPOCHS}  BATCH={BATCH_SIZE}  LR={LR}  "
        f"DROPOUT={DROPOUT}  WEIGHT_DECAY={WEIGHT_DECAY}  "
        f"FOCAL_GAMMA={FOCAL_GAMMA}  BENIGN_BOOST={BENIGN_BOOST}x  "
        f"MIXUP_ALPHA={MIXUP_ALPHA}  EFF_NUM_BETA={EFF_NUM_BETA}"
    )
    logger.info("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device : {device}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        logger.info(f"GPU    : {props.name}")
        logger.info(f"VRAM   : {props.total_memory // 1024 ** 2:,} MB")
        logger.info(f"CUDA   : {torch.version.cuda}")
        logger.info(f"BF16   : {torch.cuda.is_bf16_supported()}")
        torch.backends.cudnn.benchmark = True

    extractor = URLFeatureExtractorV2()
    logger.info(f"Feature extractor ready — {extractor.n_features} features per URL")

    # ── Step 1: Stream both HuggingFace splits ────────────────────────────────
    logger.info("\n[STEP 1a] Streaming 'train' split…")
    train_chunks = stream_and_extract(extractor, "train")

    logger.info("\n[STEP 1b] Streaming 'test' split…")
    test_chunks  = stream_and_extract(extractor, "test")

    # ── Step 2: Load into RAM ─────────────────────────────────────────────────
    logger.info("\n[STEP 2] Loading all chunks into RAM…")
    X_feat, X_char, targets = load_chunks(train_chunks + test_chunks)

    # ── Step 3: Build class map from target column ────────────────────────────
    logger.info("\n[STEP 3] Building class map from 'target' column…")
    class_map, other_class, class_names = build_class_map(targets, MIN_CLASS_SAMPLES, TOP_N_BRANDS)
    n_classes = len(class_names)
    logger.info(f"\nTotal output classes: {n_classes}  (top-{TOP_N_BRANDS} brands + benign + other_phishing)")

    class_map_path = os.path.join(MODEL_DIR, "class_map.json")
    with open(class_map_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "class_map":   class_map,
                "other_class": other_class,
                "class_names": class_names,
                "n_classes":   n_classes,
                "top_n_brands": TOP_N_BRANDS,
            },
            fh, indent=2,
        )
    logger.info(f"Class map saved → {class_map_path}")

    # CSV version for easy inspection
    class_map_csv = os.path.join(MODEL_DIR, "class_map.csv")
    with open(class_map_csv, "w", encoding="utf-8", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["class_index", "class_name", "brand_target"])
        w.writerow([0, "benign", ""])
        for brand, idx in sorted(class_map.items(), key=lambda kv: kv[1]):
            if brand:
                w.writerow([idx, brand, brand])
        w.writerow([other_class, "other_phishing", "<all remaining brands>"])
    logger.info(f"Class map CSV  → {class_map_csv}")

    # ── Step 4: Convert targets → integer class labels ────────────────────────
    logger.info("\n[STEP 4] Converting targets to class indices…")
    y = targets_to_labels(targets, class_map, other_class)
    del targets; gc.collect()

    dist = Counter(y.tolist())
    logger.info("Class distribution (top 20 by count):")
    for ci, cnt in sorted(dist.items(), key=lambda kv: -kv[1])[:20]:
        name = class_names[ci] if ci < len(class_names) else f"class_{ci}"
        logger.info(f"  [{ci:3d}] {name:45s} {cnt:>10,}")

    # ── Step 5: Stratified 80/20 split ───────────────────────────────────────
    logger.info("\n[STEP 5] 80/20 stratified split…")
    idx = np.arange(len(y))
    try:
        tr_idx, te_idx = train_test_split(
            idx, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
    except ValueError as exc:
        logger.warning(f"Stratified split failed ({exc}) — falling back to random split.")
        tr_idx, te_idx = train_test_split(
            idx, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )

    X_feat_tr, X_feat_te = X_feat[tr_idx], X_feat[te_idx]
    X_char_tr, X_char_te = X_char[tr_idx], X_char[te_idx]
    y_tr, y_te            = y[tr_idx],     y[te_idx]
    del X_feat, X_char, y; gc.collect()
    logger.info(f"  Train={len(y_tr):,}  Test={len(y_te):,}")

    # ── Step 6: Scale handcrafted features ───────────────────────────────────
    logger.info("\n[STEP 6] StandardScaler — fit on train, transform both…")
    scaler = StandardScaler()
    X_feat_tr = scaler.fit_transform(X_feat_tr).astype(np.float32)
    X_feat_te = scaler.transform(X_feat_te).astype(np.float32)

    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    logger.info(f"  Scaler saved → {scaler_path}")

    # ── Step 7: Train ─────────────────────────────────────────────────────────
    logger.info(f"\n[STEP 7] Training URLPhishNet for {EPOCHS} epochs…")
    model = train_model(
        X_feat_tr, X_char_tr, y_tr,
        X_feat_te, X_char_te, y_te,
        n_classes, class_names, device,
    )

    logger.info("\n" + "=" * 80)
    logger.info("All done!")
    logger.info(f"  Best model    : {BEST_MODEL_PATH}")
    logger.info(f"  Final model   : {FINAL_MODEL_PATH}")
    logger.info(f"  Scaler        : {scaler_path}")
    logger.info(f"  Class map     : {class_map_path}")
    logger.info(f"  Log           : {LOG_FILE}")
    logger.info(f"  Metrics CSV   : {os.path.join(LOG_DIR, 'training_metrics.csv')}")
    logger.info(f"  Checkpoints   : {CKPT_DIR}/epoch_NNN/")
    logger.info("=" * 80)
