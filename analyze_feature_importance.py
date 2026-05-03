#!/usr/bin/env python3
"""
analyze_feature_importance.py
==============================
Experiment 14 — Feature Importance Analysis using sklearn permutation_importance.

Loads the Exp 13 dual-stream model (CharCNN + FeatureMLP) and uses
sklearn.inspection.permutation_importance to rank all 67 structured URL features
by how much each one contributes to phishing detection.

Method: Shuffle one feature at a time across the test set, measure the drop in
macro F1. Features causing the biggest drop are the most important.

Output:
  feature_importance/logs/permutation_importance.csv
  feature_importance/logs/feature_importance_report.txt
  feature_importance/logs/run.log
"""

from __future__ import annotations
import gc, glob, logging, os, sys
import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.base import BaseEstimator
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_charcnn_mahalanobis_ood import (
    DualStreamOODNet, BATCH_SIZE, RANDOM_STATE, TEST_SIZE,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(PROJECT_DIR, "phishphresh", "multiclass_brand", "data")
MODEL_PATH  = os.path.join(PROJECT_DIR, "binary_charcnn_mahalanobis_ood", "models", "model_best.pth")
SCALER_PATH = os.path.join(PROJECT_DIR, "binary_charcnn_mahalanobis_ood", "models", "scaler.pkl")
OUT_LOGS    = os.path.join(PROJECT_DIR, "feature_importance", "logs")
os.makedirs(OUT_LOGS, exist_ok=True)

PERM_CSV   = os.path.join(OUT_LOGS, "permutation_importance.csv")
REPORT_TXT = os.path.join(OUT_LOGS, "feature_importance_report.txt")

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

# ── Feature names (67) ────────────────────────────────────────────────────────
FEATURE_NAMES = [
    "domain_length", "dot_count", "dash_count", "underscore_count", "digit_count",
    "uppercase_count", "digit_ratio", "special_char_ratio", "domain_parts",
    "longest_part", "shortest_part", "avg_part_length", "vowel_count",
    "consonant_count", "has_consecutive_chars",
    "char_entropy", "part_length_variance", "part_length_std", "bigram_entropy",
    "unique_char_ratio", "transition_entropy", "trigram_count", "unique_trigram_ratio",
    "vowel_consonant_ratio", "consonant_clusters",
    "tld_length", "is_common_tld", "is_country_tld", "is_suspicious_tld", "is_indian_tld",
    "cse_keyword_count", "brand_keyword_count", "login_keyword", "secure_keyword",
    "account_keyword", "update_keyword", "verify_keyword", "bank_keyword",
    "payment_keyword", "phishing_pattern_count",
    "has_subdomain", "subdomain_count", "has_www", "dictionary_word_count",
    "num_subdomains", "has_port", "has_nonstandard_port", "is_ip_address",
    "is_https", "has_unicode", "has_punycode",
    "url_length_normalized", "path_depth", "path_length_normalized",
    "query_param_count", "query_length_normalized", "fragment_length_normalized",
    "has_fragment", "file_extension_type", "path_digit_ratio", "path_special_ratio",
    "query_special_ratio", "subdomain_length", "path_slash_count",
    "shannon_entropy_url", "shannon_entropy_path", "shannon_entropy_query",
]

FEATURE_GROUPS = {
    "Basic (domain structure)": list(range(0, 15)),
    "Entropy & complexity":      list(range(15, 25)),
    "TLD signals":               list(range(25, 30)),
    "CSE pattern / brand":       list(range(30, 40)),
    "Lexical patterns":          list(range(40, 51)),
    "Structural / URL-level":    list(range(51, 67)),
}


class URLPhishNetEstimator(BaseEstimator):
    """
    Thin sklearn-compatible wrapper around DualStreamOODNet.
    Holds the char tensor fixed — only the feature matrix X is permuted by sklearn.
    Inherits BaseEstimator so sklearn.inspection.permutation_importance accepts it.
    """
    def __init__(self, model: DualStreamOODNet, X_char: np.ndarray, device: torch.device):
        self.model  = model
        self.X_char = X_char
        self.device = device

    def fit(self, X, y): return self  # already trained — nothing to do

    def predict(self, X_feat: np.ndarray) -> np.ndarray:
        self.model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(X_feat), BATCH_SIZE):
                xf = torch.from_numpy(X_feat[i:i+BATCH_SIZE].astype(np.float32)).to(self.device)
                xc = torch.from_numpy(self.X_char[i:i+BATCH_SIZE].astype(np.int64)).to(self.device)
                preds.extend(self.model(xf, xc).argmax(1).cpu().numpy().tolist())
        return np.array(preds)

    def score(self, X_feat: np.ndarray, y: np.ndarray) -> float:
        return f1_score(y, self.predict(X_feat), average="macro", zero_division=0)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("=" * 60)
    log.info("Experiment 14 — Feature Importance (Permutation Method)")
    log.info(f"Device: {device}")
    log.info("=" * 60)

    # ── Load data ─────────────────────────────────────────────────────────────
    log.info("[1/4] Loading data...")
    train_chunks = sorted(glob.glob(os.path.join(DATA_DIR, "train", "chunk_*.npz")))
    test_chunks  = sorted(glob.glob(os.path.join(DATA_DIR, "test",  "chunk_*.npz")))

    feat_parts, char_parts, tgt_parts = [], [], []
    for chunk in train_chunks + test_chunks:
        d = np.load(chunk, allow_pickle=True)
        feat_parts.append(d["features"])
        char_parts.append(d["char_ids"])
        tgt_parts.extend(d["targets"].tolist())

    X_feat_raw = np.vstack(feat_parts).astype(np.float32)
    X_char_all = np.vstack(char_parts).astype(np.int64)
    y_all = np.array([0 if str(t or "").strip() == "" else 1 for t in tgt_parts], dtype=np.int64)
    del feat_parts, char_parts; gc.collect()
    log.info(f"  {len(y_all):,} total  |  benign={( y_all==0).sum():,}  phishing={(y_all==1).sum():,}")

    _, X_feat_test, _, X_char_test, _, y_test = train_test_split(
        X_feat_raw, X_char_all, y_all,
        test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_all,
    )
    del X_feat_raw, X_char_all; gc.collect()

    scaler: StandardScaler = joblib.load(SCALER_PATH)
    X_feat_scaled = scaler.transform(X_feat_test)

    # Cap to 20k for speed
    CAP = 20_000
    if len(y_test) > CAP:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(y_test), CAP, replace=False)
        idx.sort()
        X_feat_scaled = X_feat_scaled[idx]
        X_char_test   = X_char_test[idx]
        y_test        = y_test[idx]
    log.info(f"  Using {len(y_test):,} test samples")

    # ── Load model ────────────────────────────────────────────────────────────
    log.info("[2/4] Loading model...")
    N_FEAT = X_feat_scaled.shape[1]
    model = DualStreamOODNet(N_FEAT).to(device)
    ckpt  = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    log.info(f"  Epoch {ckpt.get('epoch','?')}  F1={ckpt.get('f1', float('nan')):.4f}")

    # ── Permutation importance ────────────────────────────────────────────────
    log.info("[3/4] Running sklearn.inspection.permutation_importance (n_repeats=5)...")
    log.info("  Shuffles each of the 67 features 5x and measures F1 drop.")

    estimator = URLPhishNetEstimator(model, X_char_test, device)
    result = permutation_importance(
        estimator, X_feat_scaled, y_test,
        scoring="f1_macro",
        n_repeats=5,
        random_state=42,
        n_jobs=1,
    )
    means = result.importances_mean
    stds  = result.importances_std

    feat_to_group = {
        FEATURE_NAMES[i]: grp
        for grp, idxs in FEATURE_GROUPS.items()
        for i in idxs if i < N_FEAT
    }
    perm_df = pd.DataFrame({
        "rank":          range(1, N_FEAT + 1),
        "feature":       FEATURE_NAMES[:N_FEAT],
        "f1_drop_mean":  means,
        "f1_drop_std":   stds,
        "group":         [feat_to_group.get(FEATURE_NAMES[i], "Other") for i in range(N_FEAT)],
    })
    perm_df = perm_df.sort_values("f1_drop_mean", ascending=False).reset_index(drop=True)
    perm_df["rank"] = perm_df.index + 1
    perm_df.to_csv(PERM_CSV, index=False)
    log.info(f"  Saved: {PERM_CSV}")

    # ── Write report ──────────────────────────────────────────────────────────
    log.info("[4/4] Writing report...")
    _write_report(perm_df, N_FEAT)
    log.info(f"  Saved: {REPORT_TXT}")
    log.info("Done.")


def _write_report(perm_df: pd.DataFrame, n_feat: int):
    sep = "=" * 70
    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write(f"{sep}\n")
        f.write("Experiment 14 — Feature Importance Report\n")
        f.write("Method: Permutation importance — shuffle each feature 5x, measure F1 drop\n")
        f.write("Model : CharCNN + FeatureMLP dual-stream (Exp 13)\n")
        f.write(f"{sep}\n\n")

        f.write("Top 20 features by F1 drop when shuffled:\n")
        f.write(f"{'Rank':<5} {'Feature':<40} {'Group':<30} {'F1-drop':>8}  {'±std':>7}\n")
        f.write("-" * 93 + "\n")
        for _, row in perm_df.head(20).iterrows():
            f.write(
                f"{int(row['rank']):<5} {row['feature']:<40} {row['group']:<30} "
                f"{row['f1_drop_mean']:>8.4f}  {row['f1_drop_std']:>7.4f}\n"
            )

        f.write("\n\nPer-group summary (average F1-drop across all features in group):\n")
        f.write("-" * 60 + "\n")
        group_stats = (
            perm_df.groupby("group")["f1_drop_mean"]
            .agg(["mean", "max"])
            .sort_values("mean", ascending=False)
        )
        for grp, row in group_stats.iterrows():
            f.write(f"  {grp:<36} avg={row['mean']:.4f}  max={row['max']:.4f}\n")

        top5 = perm_df.head(5)["feature"].tolist()
        f.write(f"\n\nTop 5 most important features:\n")
        for i, feat in enumerate(top5, 1):
            grp = perm_df[perm_df["feature"] == feat]["group"].values[0]
            drop = perm_df[perm_df["feature"] == feat]["f1_drop_mean"].values[0]
            f.write(f"  {i}. {feat}  [{grp}]  F1-drop={drop:.4f}\n")

        zero_imp = perm_df[perm_df["f1_drop_mean"] <= 0.0001]
        f.write(f"\n\nFeatures with near-zero importance ({len(zero_imp)} total):\n")
        f.write("  " + ", ".join(zero_imp["feature"].tolist()) + "\n")


if __name__ == "__main__":
    main()
