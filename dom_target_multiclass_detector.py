from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime, timezone
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from datasets import load_dataset
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.utils.class_weight import compute_sample_weight
from tqdm import tqdm

from dom_tree_node_detector import DOMOnlyRobustDetector

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None


SUPPORTED_MC_ALGORITHMS = {
    "random_forest",
    "extra_trees",
    "gradient_boosting",
    "logistic_regression",
    "xgboost",
    "lightgbm",
}

UNKNOWN_CLASS_NAME = "unknown"
DEFAULT_LABEL_SUFFIX = "_y11.npz"


def _normalize_binary_label(raw) -> int:
    try:
        if isinstance(raw, str):
            return 1 if raw.strip().lower() in {"phish", "phishing", "1"} else 0
        return 1 if int(raw) == 1 else 0
    except Exception:
        return 0


def _normalize_target(raw_target: Optional[str]) -> str:
    if raw_target is None:
        return ""
    value = str(raw_target).strip().lower()
    if value in {"", "none", "null", "nan", "other"}:
        return ""
    return value


def _expand_chunk_files(patterns: Sequence[str]) -> List[str]:
    out: List[str] = []
    for pattern in patterns:
        if any(ch in pattern for ch in "*?[]"):
            out.extend(sorted(str(p) for p in Path().glob(pattern)))
        else:
            out.append(pattern)
    filtered = [p for p in out if p.endswith(".npz") and not p.endswith(DEFAULT_LABEL_SUFFIX)]
    deduped = list(dict.fromkeys(filtered))
    if not deduped:
        raise ValueError("No chunk files found from provided patterns.")
    return deduped


def _resolve_split_chunks(feature_root: Optional[str], split: str, explicit_patterns: Optional[Sequence[str]]) -> List[str]:
    if feature_root:
        root = Path(feature_root)
        split_dir = root / split
        if split_dir.exists():
            return _expand_chunk_files([str(split_dir / f"{split}_domfeat_chunk_*.npz")])
        return _expand_chunk_files([str(root / f"{split}_domfeat_chunk_*.npz")])
    if explicit_patterns:
        return _expand_chunk_files(explicit_patterns)
    raise ValueError(f"Provide --feature-root or --{split}-chunks")


def _sidecar_label_path(chunk_path: str, suffix: str = DEFAULT_LABEL_SUFFIX) -> str:
    p = Path(chunk_path)
    return str(p.with_name(p.stem + suffix))


def _load_top_map(mapping_path: str) -> Dict[str, object]:
    with Path(mapping_path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_top_map(mapping_path: str, payload: Dict[str, object]) -> None:
    with Path(mapping_path).open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _build_runtime_classes_and_remap(mapping: Dict[str, object]) -> Tuple[List[str], Dict[str, int], Dict[int, int]]:
    raw_top_targets = [str(t) for t in mapping.get("top_targets", [])]
    unknown_name = str(mapping.get("unknown_class_name", UNKNOWN_CLASS_NAME))
    old_unknown_id = int(mapping.get("unknown_class_id", len(raw_top_targets)))

    sanitized_targets: List[str] = []
    label_id_remap: Dict[int, int] = {}

    for old_id, target in enumerate(raw_top_targets):
        norm = _normalize_target(target)
        if norm:
            new_id = len(sanitized_targets)
            sanitized_targets.append(norm)
            label_id_remap[old_id] = new_id

    new_unknown_id = len(sanitized_targets)
    for old_id, target in enumerate(raw_top_targets):
        if not _normalize_target(target):
            label_id_remap[old_id] = new_unknown_id
    label_id_remap[old_unknown_id] = new_unknown_id

    class_names = sanitized_targets + [unknown_name]
    target_to_id = {name: i for i, name in enumerate(sanitized_targets)}
    return class_names, target_to_id, label_id_remap


def build_top_targets(split: str, top_k: int, hf_token: Optional[str] = None) -> Counter:
    print(f"[top-targets] starting scan split={split} top_k={top_k}", flush=True)
    ds = load_dataset("phreshphish/phreshphish", split=split, streaming=True, token=hf_token)
    counts: Counter = Counter()
    seen = 0
    phishing_seen = 0
    for sample in ds:
        seen += 1
        if _normalize_binary_label(sample.get("label", 0)) != 1:
            continue
        phishing_seen += 1
        target_name = _normalize_target(sample.get("target"))
        if target_name:
            counts[target_name] += 1
        if seen % 1000 == 0:
            print(
                f"[top-targets] scanned={seen:,} phishing={phishing_seen:,} unique_targets={len(counts):,}",
                flush=True,
            )
    if not counts:
        raise ValueError("No phishing targets found in dataset stream.")
    print(f"[top-targets] completed unique_targets={len(counts):,} top_{top_k}={counts.most_common(top_k)}", flush=True)
    return counts


def _class_id_for_sample(sample: Dict[str, object], class_to_id: Dict[str, int], unknown_class_id: int) -> int:
    if _normalize_binary_label(sample.get("label", 0)) != 1:
        return unknown_class_id
    target_name = _normalize_target(sample.get("target"))
    if not target_name:
        return unknown_class_id
    return int(class_to_id.get(target_name, unknown_class_id))


def create_or_update_multiclass_labels(
    split: str,
    chunk_files: Sequence[str],
    mapping_path: str,
    top_k: int = 10,
    label_suffix: str = DEFAULT_LABEL_SUFFIX,
    inplace: bool = False,
    resume: bool = False,
    hf_token: Optional[str] = None,
) -> Dict[str, object]:
    print(
        f"[y11-build] starting split={split} chunks={len(chunk_files)} top_k={top_k} inplace={inplace} resume={resume}",
        flush=True,
    )

    if resume and not inplace:
        original_count = len(chunk_files)
        chunk_files = [c for c in chunk_files if not Path(_sidecar_label_path(c, suffix=label_suffix)).exists()]
        skipped = original_count - len(chunk_files)
        if skipped > 0:
            print(f"[y11-build] resume mode: skipping {skipped} chunks with existing sidecars", flush=True)

    mapping_file = Path(mapping_path)
    if mapping_file.exists():
        mapping = _load_top_map(mapping_path)
        top_targets = mapping.get("top_targets", [])
        if not isinstance(top_targets, list) or len(top_targets) == 0:
            raise ValueError("Invalid mapping file: missing top_targets list.")

        sanitized_top_targets = [t for t in top_targets if _normalize_target(str(t))]
        if len(sanitized_top_targets) != len(top_targets):
            mapping["top_targets"] = sanitized_top_targets
            mapping["top_k"] = int(len(sanitized_top_targets))
            mapping["unknown_class_id"] = int(len(sanitized_top_targets))
            _save_top_map(mapping_path, mapping)
            print(f"[y11-build] removed non-target labels from mapping and updated {mapping_path}", flush=True)
        print(f"[y11-build] using existing mapping from {mapping_path}", flush=True)
    else:
        if split != "train":
            raise ValueError("Mapping file does not exist; build train mapping first or provide mapping_path.")
        target_counts = build_top_targets(split="train", top_k=top_k, hf_token=hf_token)
        top_targets = [name for name, _ in target_counts.most_common(top_k)]
        mapping = {
            "source_split": "train",
            "top_k": int(top_k),
            "top_targets": top_targets,
            "unknown_class_name": UNKNOWN_CLASS_NAME,
            "unknown_class_id": int(top_k),
            "target_counts_top_k": [{"target": n, "count": int(c)} for n, c in target_counts.most_common(top_k)],
        }
        _save_top_map(mapping_path, mapping)
        print(f"[y11-build] saved new mapping to {mapping_path}", flush=True)

    unknown_class_id = int(mapping.get("unknown_class_id", len(mapping["top_targets"])))
    class_to_id = {name: i for i, name in enumerate(mapping["top_targets"])}

    ds = load_dataset("phreshphish/phreshphish", split=split, streaming=True, token=hf_token)
    ds_iter = iter(ds)

    total_rows = 0
    total_unknown = 0
    class_hist = Counter()
    total_chunks = len(chunk_files)

    for i, chunk_path in enumerate(chunk_files, start=1):
        with np.load(chunk_path) as data:
            X = data["X"]
            y = data["y"]
            n_rows = int(X.shape[0])

        y11 = np.empty((n_rows,), dtype=np.int16)
        for row_idx in range(n_rows):
            sample = next(ds_iter)
            cid = _class_id_for_sample(sample, class_to_id, unknown_class_id)
            y11[row_idx] = cid
            class_hist[cid] += 1
            if cid == unknown_class_id:
                total_unknown += 1
            if (row_idx + 1) % 250 == 0 or row_idx + 1 == n_rows:
                print(
                    f"[y11-build] chunk={i}/{total_chunks} row={row_idx + 1:,}/{n_rows:,} total_rows={total_rows + row_idx + 1:,} unknown={total_unknown:,}",
                    flush=True,
                )

        if inplace:
            np.savez_compressed(chunk_path, X=X, y=y, y11=y11)
            out_path = chunk_path
        else:
            out_path = _sidecar_label_path(chunk_path, suffix=label_suffix)
            np.savez_compressed(out_path, y11=y11)

        total_rows += n_rows
        if i % 200 == 0 or i == total_chunks:
            print(
                f"[y11-build] split={split} chunk={i}/{total_chunks} rows={total_rows:,} unknown={total_unknown:,} out={Path(out_path).name}",
                flush=True,
            )

    class_names = list(mapping["top_targets"]) + [UNKNOWN_CLASS_NAME]
    return {
        "split": split,
        "chunks": int(total_chunks),
        "rows": int(total_rows),
        "unknown_class_id": int(unknown_class_id),
        "class_names": class_names,
        "class_hist": {class_names[k]: int(v) for k, v in sorted(class_hist.items())},
        "mapping_path": mapping_path,
        "inplace": bool(inplace),
    }


class HTMLTargetMulticlassDOMClassifier:
    """11-class DOM classifier: top-10 phishing targets + unknown."""

    def __init__(
        self,
        class_names: List[str],
        target_to_id: Dict[str, int],
        label_id_remap: Optional[Dict[int, int]] = None,
        hashed_dim: int = 4096,
        n_estimators: int = 300,
        random_state: int = 42,
        algorithm: str = "random_forest",
        prefer_gpu: bool = True,
    ):
        self.class_names = class_names
        self.target_to_id = target_to_id
        self.hashed_dim = hashed_dim
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.algorithm = algorithm
        self.prefer_gpu = bool(prefer_gpu)
        self.label_id_remap = dict(label_id_remap or {})

        if self.algorithm not in SUPPORTED_MC_ALGORITHMS:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")

        self.dom_detector = DOMOnlyRobustDetector()
        self.hasher = FeatureHasher(n_features=hashed_dim, input_type="dict", alternate_sign=False)
        self.scaler = StandardScaler()
        self.model = self._build_model()
        self.is_fitted = False

    def _unknown_class_id(self) -> int:
        # Unknown class is appended as the final class in mapping/runtime setup.
        return max(0, len(self.class_names) - 1)

    def _binary_from_class_ids(self, y_class: np.ndarray) -> np.ndarray:
        unknown_id = self._unknown_class_id()
        y_arr = np.asarray(y_class, dtype=np.int16)
        return (y_arr != unknown_id).astype(np.int16)

    def _build_model(self, force_cpu: bool = False):
        use_gpu = bool(self.prefer_gpu and not force_cpu)
        if self.algorithm == "random_forest":
            return RandomForestClassifier(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                n_jobs=-1,
                class_weight="balanced_subsample",
                max_depth=24,
                min_samples_split=10,
            )
        if self.algorithm == "extra_trees":
            return ExtraTreesClassifier(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                n_jobs=-1,
                class_weight="balanced_subsample",
                max_depth=24,
                min_samples_split=10,
            )
        if self.algorithm == "gradient_boosting":
            return GradientBoostingClassifier(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                learning_rate=0.05,
                max_depth=3,
            )
        if self.algorithm == "logistic_regression":
            return LogisticRegression(
                random_state=self.random_state,
                max_iter=1200,
                class_weight="balanced",
                multi_class="auto",
                solver="lbfgs",
                n_jobs=None,
            )
        if self.algorithm == "xgboost":
            if xgb is None:
                raise ImportError("xgboost not installed; install with: pip install xgboost")
            params = {
                "n_estimators": self.n_estimators,
                "scale_pos_weight": 1.0,
                "learning_rate": 0.05,
                "max_depth": 6,
                "random_state": self.random_state,
            }
            if use_gpu:
                params.update({"tree_method": "gpu_hist", "gpu_id": 0})
            else:
                params.update({"tree_method": "hist"})
            return xgb.XGBClassifier(
                **params,
            )
        if self.algorithm == "lightgbm":
            if lgb is None:
                raise ImportError("lightgbm not installed; install with: pip install lightgbm")
            device_type = "gpu" if use_gpu else "cpu"
            return lgb.LGBMClassifier(
                n_estimators=self.n_estimators,
                device=device_type,
                gpu_device_id=0 if use_gpu else -1,
                learning_rate=0.05,
                num_leaves=31,
                max_depth=6,
                random_state=self.random_state,
            )
        raise ValueError(f"Unsupported algorithm: {self.algorithm}")

    @staticmethod
    def _is_gpu_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        gpu_terms = ["cuda", "gpu", "device", "no visible gpu", "not compiled with gpu", "gpu_hist"]
        return any(t in msg for t in gpu_terms)

    def _flatten_profile(self, profile) -> Dict[str, float]:
        feat: Dict[str, float] = {}
        for k, v in profile.metrics.items():
            feat[f"metric::{k}"] = float(v)
        groups = [
            ("role", profile.role_counts),
            ("path", profile.path_ngrams),
            ("depth", profile.depth_hist),
            ("behavior", profile.behavior_edges),
            ("text", profile.text_keywords),
            ("attr", profile.attribute_features),
        ]
        for prefix, counter in groups:
            for k, v in counter.items():
                feat[f"{prefix}::{k}"] = float(v)
        return feat

    def extract_html_feature_dict(self, html: str) -> Dict[str, float]:
        try:
            profile = self.dom_detector.build_profile(html or "", page_domain="")
            return self._flatten_profile(profile)
        except Exception:
            return {}

    def extract_html_feature_matrix(self, html_batch: List[str]) -> np.ndarray:
        dict_batch = [self.extract_html_feature_dict(h) for h in html_batch]
        return self.hasher.transform(dict_batch).toarray().astype(np.float32)

    def _apply_label_id_remap(self, y: np.ndarray) -> np.ndarray:
        if not self.label_id_remap:
            return y.astype(np.int16)
        unknown_id = len(self.class_names) - 1
        out = np.empty((len(y),), dtype=np.int16)
        for i, val in enumerate(y.tolist()):
            out[i] = int(self.label_id_remap.get(int(val), unknown_id))
        return out

    def _class_name_for_id(self, class_id: int) -> str:
        if 0 <= int(class_id) < len(self.class_names):
            return self.class_names[int(class_id)]
        return UNKNOWN_CLASS_NAME

    def _load_chunks_with_y11(
        self,
        chunk_files: Sequence[str],
        label_suffix: str = DEFAULT_LABEL_SUFFIX,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_parts: List[np.ndarray] = []
        y_parts: List[np.ndarray] = []
        y_bin_parts: List[np.ndarray] = []

        for i, chunk_path in enumerate(chunk_files, start=1):
            with np.load(chunk_path) as data:
                X = data["X"].astype(np.float32)
                y_bin = data["y"].astype(np.int16)
                if "y11" in data.files:
                    y11 = data["y11"].astype(np.int16)
                else:
                    sidecar = _sidecar_label_path(chunk_path, suffix=label_suffix)
                    with np.load(sidecar) as sdata:
                        y11 = sdata["y11"].astype(np.int16)

            y11 = self._apply_label_id_remap(y11)
            x_parts.append(X)
            y_parts.append(y11)
            y_bin_parts.append(y_bin)

            if i % 200 == 0 or i == len(chunk_files):
                print(f"[mc-load] loaded_chunks={i}/{len(chunk_files)}", flush=True)

        if not x_parts:
            return (
                np.empty((0, self.hashed_dim), dtype=np.float32),
                np.empty((0,), dtype=np.int16),
                np.empty((0,), dtype=np.int16),
            )
        return (
            np.vstack(x_parts).astype(np.float32),
            np.concatenate(y_parts).astype(np.int16),
            np.concatenate(y_bin_parts).astype(np.int16),
        )

    def _fit_estimator(self, model, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> None:
        try:
            if sample_weight is None:
                model.fit(X, y)
                return
            try:
                model.fit(X, y, sample_weight=sample_weight)
            except TypeError:
                model.fit(X, y)
            return
        except Exception as exc:
            if self.algorithm not in {"xgboost", "lightgbm"} or not self.prefer_gpu or not self._is_gpu_error(exc):
                raise

            print(f"[mc-train] GPU unavailable for {self.algorithm}; falling back to CPU model. error={exc}", flush=True)
            cpu_model = self._build_model(force_cpu=True)
            if sample_weight is None:
                cpu_model.fit(X, y)
            else:
                try:
                    cpu_model.fit(X, y, sample_weight=sample_weight)
                except TypeError:
                    cpu_model.fit(X, y)

            if model is self.model:
                self.model = cpu_model
            self.prefer_gpu = False

    def fit_from_feature_store(
        self,
        train_chunk_files: Sequence[str],
        label_suffix: str = DEFAULT_LABEL_SUFFIX,
        validation_fraction: float = 0.1,
        validation_min_samples: int = 2000,
        unknown_max_multiplier: float = 2.0,
        max_samples_per_class: int = 0,
        auto_max_samples_per_class: bool = False,
        max_class_ratio: float = 0.0,
    ) -> Dict[str, object]:
        if validation_fraction < 0 or validation_fraction >= 0.5:
            raise ValueError("validation_fraction must be in [0.0, 0.5)")
        if validation_min_samples < 0:
            raise ValueError("validation_min_samples must be >= 0")
        if not train_chunk_files:
            raise ValueError("No train chunks found.")

        unknown_class_id = len(self.class_names) - 1
        x_parts: List[np.ndarray] = []
        y_parts: List[np.ndarray] = []
        hist_raw: Counter = Counter()

        for chunk_path in tqdm(train_chunk_files, desc="Loading train chunks"):
            with np.load(chunk_path) as data:
                X = data["X"].astype(np.float32)
                if "y11" in data.files:
                    y11 = data["y11"].astype(np.int16)
                else:
                    sidecar = _sidecar_label_path(chunk_path, suffix=label_suffix)
                    with np.load(sidecar) as sdata:
                        y11 = sdata["y11"].astype(np.int16)

            y11 = self._apply_label_id_remap(y11)
            x_parts.append(X)
            y_parts.append(y11)
            hist_raw.update(int(v) for v in y11.tolist())

        X_all = np.vstack(x_parts).astype(np.float32)
        y_all = np.concatenate(y_parts).astype(np.int16)
        total_raw = int(len(y_all))

        rng = np.random.default_rng(self.random_state)
        classes, counts = np.unique(y_all, return_counts=True)
        class_count_map = {int(cid): int(cnt) for cid, cnt in zip(classes.tolist(), counts.tolist())}
        known_counts = [int(c) for c in counts[classes != unknown_class_id]]
        known_ref = int(np.median(known_counts)) if known_counts else int(np.max(counts))
        unknown_cap = int(max(known_ref * unknown_max_multiplier, 1))
        auto_cap = known_ref if auto_max_samples_per_class and max_samples_per_class <= 0 else 0
        ratio_cap = 0
        if max_class_ratio > 0:
            known_nonzero = [
                int(v)
                for k, v in class_count_map.items()
                if int(k) != unknown_class_id and int(v) > 0
            ]
            if known_nonzero:
                min_known = int(min(known_nonzero))
                ratio_cap = int(max(1, round(float(min_known) * float(max_class_ratio))))

        picked_mc: List[np.ndarray] = []
        for cid in classes:
            cid_i = int(cid)
            idx = np.where(y_all == cid_i)[0]
            cap = len(idx)
            if max_samples_per_class > 0:
                cap = min(cap, max_samples_per_class)
            elif auto_cap > 0:
                cap = min(cap, auto_cap)
            if ratio_cap > 0 and cid_i != unknown_class_id:
                cap = min(cap, ratio_cap)
            if cid_i == unknown_class_id:
                cap = min(cap, unknown_cap)
            if cap < len(idx):
                idx = rng.choice(idx, size=cap, replace=False)
            picked_mc.append(np.asarray(idx, dtype=np.int64))
        mc_idx = np.concatenate(picked_mc)
        rng.shuffle(mc_idx)

        X_mc = X_all[mc_idx]
        y_mc = y_all[mc_idx]

        if validation_fraction > 0 and len(y_mc) >= validation_min_samples:
            try:
                X_fit, X_val, y_fit, y_val = train_test_split(
                    X_mc, y_mc, test_size=validation_fraction, random_state=self.random_state, stratify=y_mc
                )
            except Exception:
                X_fit, X_val, y_fit, y_val = train_test_split(
                    X_mc, y_mc, test_size=validation_fraction, random_state=self.random_state, stratify=None
                )
        else:
            X_fit, y_fit = X_mc, y_mc
            X_val = np.empty((0, X_mc.shape[1]), dtype=np.float32)
            y_val = np.empty((0,), dtype=np.int16)

        self.scaler.fit(X_fit)
        X_fit_scaled = self.scaler.transform(X_fit)
        X_val_scaled = self.scaler.transform(X_val) if len(X_val) > 0 else X_val

        self._fit_estimator(
            self.model,
            X_fit_scaled,
            y_fit,
            sample_weight=compute_sample_weight(class_weight="balanced", y=y_fit),
        )

        y_train_pred = self.model.predict(X_fit_scaled)
        yb_fit = self._binary_from_class_ids(y_fit)
        yb_train_pred = self._binary_from_class_ids(y_train_pred)
        train_acc = float(accuracy_score(y_fit, y_train_pred))
        train_macro_f1 = float(f1_score(y_fit, y_train_pred, average="macro", zero_division=0))
        train_weighted_f1 = float(f1_score(y_fit, y_train_pred, average="weighted", zero_division=0))
        train_binary_acc = float(accuracy_score(yb_fit, yb_train_pred))
        train_binary_f1 = float(f1_score(yb_fit, yb_train_pred, average="binary", zero_division=0))

        if len(y_val) > 0:
            y_val_pred = self.model.predict(X_val_scaled)
            yb_val = self._binary_from_class_ids(y_val)
            yb_val_pred = self._binary_from_class_ids(y_val_pred)
            val_acc = float(accuracy_score(y_val, y_val_pred))
            val_macro_f1 = float(f1_score(y_val, y_val_pred, average="macro", zero_division=0))
            val_weighted_f1 = float(f1_score(y_val, y_val_pred, average="weighted", zero_division=0))
            val_binary_acc = float(accuracy_score(yb_val, yb_val_pred))
            val_binary_f1 = float(f1_score(yb_val, yb_val_pred, average="binary", zero_division=0))
        else:
            val_acc = float("nan")
            val_macro_f1 = float("nan")
            val_weighted_f1 = float("nan")
            val_binary_acc = float("nan")
            val_binary_f1 = float("nan")

        self.is_fitted = True

        print(
            f"[mc-train] samples_raw={total_raw:,} samples_balanced={len(y_mc):,} train_acc={train_acc:.4f} train_macro_f1={train_macro_f1:.4f} val_acc={val_acc:.4f} val_macro_f1={val_macro_f1:.4f}",
            flush=True,
        )

        return {
            "algorithm": self.algorithm,
            "prefer_gpu": bool(self.prefer_gpu),
            "batch_ensemble": False,
            "warm_start": False,
            "fit_batch_size": int(total_raw),
            "fit_rounds": 1,
            "mlp_epochs": 0,
            "auto_max_samples_per_class": bool(auto_max_samples_per_class),
            "max_class_ratio": float(max_class_ratio),
            "max_class_ratio_cap": int(ratio_cap),
            "validation_fraction": float(validation_fraction),
            "validation_min_samples": int(validation_min_samples),
            "train_samples_raw": int(total_raw),
            "train_samples_balanced": int(len(y_mc)),
            "train_accuracy": train_acc,
            "train_macro_f1": train_macro_f1,
            "train_weighted_f1": train_weighted_f1,
            "train_binary_accuracy": train_binary_acc,
            "train_binary_f1": train_binary_f1,
            "val_accuracy": val_acc,
            "val_macro_f1": val_macro_f1,
            "val_weighted_f1": val_weighted_f1,
            "val_binary_accuracy": val_binary_acc,
            "val_binary_f1": val_binary_f1,
            "class_hist_raw": {self._class_name_for_id(k): int(v) for k, v in sorted(hist_raw.items())},
            "class_hist_balanced": {self._class_name_for_id(k): int(v) for k, v in sorted(Counter(int(v) for v in y_mc.tolist()).items())},
            "curve": [
                {
                    "fit_round": 1,
                    "train_samples": int(len(y_fit)),
                    "val_samples": int(len(y_val)),
                    "train_accuracy": train_acc,
                    "train_macro_f1": train_macro_f1,
                    "train_weighted_f1": train_weighted_f1,
                    "val_accuracy": val_acc,
                    "val_macro_f1": val_macro_f1,
                    "val_weighted_f1": val_weighted_f1,
                    "train_binary_accuracy": train_binary_acc,
                    "train_binary_f1": train_binary_f1,
                    "val_binary_accuracy": val_binary_acc,
                    "val_binary_f1": val_binary_f1,
                }
            ],
        }

    def evaluate_from_feature_store(self, test_chunk_files: Sequence[str], label_suffix: str = DEFAULT_LABEL_SUFFIX) -> Dict[str, object]:
        if not self.is_fitted:
            raise ValueError("Model must be trained/loaded before evaluation.")

        X_test, y_test, _ = self._load_chunks_with_y11(test_chunk_files, label_suffix=label_suffix)
        if len(X_test) == 0:
            raise ValueError("No test chunks found.")

        X_scaled = self.scaler.transform(X_test)
        y_prob = self.model.predict_proba(X_scaled)
        y_pred = np.argmax(y_prob, axis=1).astype(np.int16)

        out: Dict[str, object] = {
            "title": f"test_{self.algorithm}_multiclass",
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
            "confusion_matrix": confusion_matrix(y_test, y_pred, labels=list(range(len(self.class_names)))).tolist(),
            "class_names": self.class_names,
            "report": classification_report(
                y_test,
                y_pred,
                labels=list(range(len(self.class_names))),
                target_names=self.class_names,
                zero_division=0,
                output_dict=True,
            ),
        }

        unknown_id = self._unknown_class_id()
        yb_test = self._binary_from_class_ids(y_test)
        yb_prob = 1.0 - y_prob[:, unknown_id]
        yb_pred = self._binary_from_class_ids(y_pred)
        out["binary"] = {
            "accuracy": float(accuracy_score(yb_test, yb_pred)),
            "f1": float(f1_score(yb_test, yb_pred, average="binary", zero_division=0)),
            "confusion_matrix": confusion_matrix(yb_test, yb_pred, labels=[0, 1]).tolist(),
            "report": classification_report(
                yb_test,
                yb_pred,
                labels=[0, 1],
                target_names=["Benign", "Phishing"],
                zero_division=0,
                output_dict=True,
            ),
        }

        try:
            y_true_bin = label_binarize(y_test, classes=list(range(len(self.class_names))))
            out["roc_auc_ovr_macro"] = float(roc_auc_score(y_true_bin, y_prob, multi_class="ovr", average="macro"))
        except Exception:
            out["roc_auc_ovr_macro"] = float("nan")

        return out

    def predict_html(self, html: str, top_k: int = 3) -> Dict[str, object]:
        if not self.is_fitted:
            raise ValueError("Model must be trained/loaded before prediction.")

        X = self.extract_html_feature_matrix([html])
        X_scaled = self.scaler.transform(X)
        probs = self.model.predict_proba(X_scaled)[0]
        unknown_id = self._unknown_class_id()
        pred_id = int(np.argmax(probs))
        bprob = float(1.0 - probs[unknown_id])
        binary_id = int(pred_id != unknown_id)

        top_idx = np.argsort(probs)[::-1][: max(top_k, 1)]
        return {
            "predicted_class_id": pred_id,
            "predicted_class_name": self.class_names[pred_id],
            "predicted_binary_label": "phishing" if binary_id == 1 else "benign",
            "predicted_binary_id": binary_id,
            "phishing_probability": bprob,
            "top_predictions": [
                {"class_id": int(idx), "class_name": self.class_names[int(idx)], "probability": float(probs[int(idx)])}
                for idx in top_idx.tolist()
            ],
        }

    def save(self, model_path: str) -> None:
        payload = {
            "class_names": self.class_names,
            "target_to_id": self.target_to_id,
            "label_id_remap": self.label_id_remap,
            "hashed_dim": self.hashed_dim,
            "n_estimators": self.n_estimators,
            "random_state": self.random_state,
            "algorithm": self.algorithm,
            "scaler": self.scaler,
            "model": self.model,
            "is_fitted": self.is_fitted,
        }
        with Path(model_path).open("wb") as f:
            pickle.dump(payload, f)

    def load(self, model_path: str) -> None:
        with Path(model_path).open("rb") as f:
            payload = pickle.load(f)

        self.class_names = list(payload["class_names"])
        self.target_to_id = dict(payload["target_to_id"])
        self.label_id_remap = dict(payload.get("label_id_remap", {}))
        self.hashed_dim = int(payload["hashed_dim"])
        self.n_estimators = int(payload["n_estimators"])
        self.random_state = int(payload["random_state"])
        self.algorithm = str(payload.get("algorithm", "random_forest"))
        self.scaler = payload["scaler"]
        self.model = payload["model"]
        self.is_fitted = bool(payload["is_fitted"])
        self.hasher = FeatureHasher(n_features=self.hashed_dim, input_type="dict", alternate_sign=False)


def _read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def _append_jsonl(log_path: str, payload: Dict[str, object]) -> None:
    """Append one JSON object per line for run-by-run tracking."""
    p = Path(log_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DOM multiclass target detector (top-10 targets + unknown)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_y11 = sub.add_parser("build-y11", help="Create/update y11 labels aligned to existing X chunks")
    p_y11.add_argument("--split", choices=["train", "test"], required=True)
    p_y11.add_argument("--chunk-files", nargs="+", required=True)
    p_y11.add_argument("--mapping-path", required=True)
    p_y11.add_argument("--top-k", type=int, default=10)
    p_y11.add_argument("--label-suffix", default=DEFAULT_LABEL_SUFFIX)
    p_y11.add_argument("--inplace", action="store_true")
    p_y11.add_argument("--resume", action="store_true")
    p_y11.add_argument("--hf-token", default=None)

    p_train = sub.add_parser("mc-train", help="Train multiclass model from existing feature chunks + y11 labels")
    p_train.add_argument("--train-chunks", nargs="+")
    p_train.add_argument("--feature-root", default=None, help="Dataset root containing train/test chunk folders")
    p_train.add_argument("--mapping-path", required=True)
    p_train.add_argument("--model-path", required=True)
    p_train.add_argument("--algorithm", choices=sorted(SUPPORTED_MC_ALGORITHMS), default="random_forest")
    p_train.add_argument(
        "--prefer-gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use GPU where supported (xgboost/lightgbm); CPU-only for sklearn algorithms",
    )
    p_train.add_argument("--hashed-dim", type=int, default=4096)
    p_train.add_argument("--n-estimators", type=int, default=300)
    p_train.add_argument("--label-suffix", default=DEFAULT_LABEL_SUFFIX)
    p_train.add_argument("--unknown-max-multiplier", type=float, default=2.0)
    p_train.add_argument(
        "--max-class-ratio",
        type=float,
        default=0.0,
        help="Optional cap: each known class is limited to min_known_count * ratio (<=0 disables)",
    )
    p_train.add_argument("--max-samples-per-class", type=int, default=0)
    p_train.add_argument("--auto-max-samples-per-class", action="store_true")
    p_train.add_argument("--validation-fraction", type=float, default=0.1)
    p_train.add_argument("--validation-min-samples", type=int, default=2000)
    p_train.add_argument("--subsample-ratio", type=float, default=1.0, help="Kept for compatibility; unused")
    p_train.add_argument(
        "--results-log",
        default="mc_results_log.jsonl",
        help="Append train/eval outputs as JSON lines to this file",
    )

    p_eval = sub.add_parser("mc-eval", help="Evaluate multiclass model")
    p_eval.add_argument("--model-path", required=True)
    p_eval.add_argument("--test-chunks", nargs="+")
    p_eval.add_argument("--feature-root", default=None, help="Dataset root containing train/test chunk folders")
    p_eval.add_argument("--label-suffix", default=DEFAULT_LABEL_SUFFIX)
    p_eval.add_argument(
        "--results-log",
        default="mc_results_log.jsonl",
        help="Append train/eval outputs as JSON lines to this file",
    )

    p_pred = sub.add_parser("mc-predict", help="Predict target class from one HTML file")
    p_pred.add_argument("--model-path", required=True)
    p_pred.add_argument("--html", required=True)
    p_pred.add_argument("--top-k", type=int, default=3)

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.cmd == "build-y11":
        chunk_files = _expand_chunk_files(args.chunk_files)
        out = create_or_update_multiclass_labels(
            split=args.split,
            chunk_files=chunk_files,
            mapping_path=args.mapping_path,
            top_k=args.top_k,
            label_suffix=args.label_suffix,
            inplace=args.inplace,
            resume=args.resume,
            hf_token=args.hf_token,
        )
        print(json.dumps(out, indent=2))
        return

    if args.cmd == "mc-train":
        mapping = _load_top_map(args.mapping_path)
        class_names, target_to_id, label_id_remap = _build_runtime_classes_and_remap(mapping)
        clf = HTMLTargetMulticlassDOMClassifier(
            class_names=class_names,
            target_to_id=target_to_id,
            label_id_remap=label_id_remap,
            hashed_dim=args.hashed_dim,
            n_estimators=args.n_estimators,
            algorithm=args.algorithm,
            prefer_gpu=bool(args.prefer_gpu),
        )
        train_chunks = _resolve_split_chunks(args.feature_root, "train", args.train_chunks)
        info = clf.fit_from_feature_store(
            train_chunks,
            label_suffix=args.label_suffix,
            validation_fraction=args.validation_fraction,
            validation_min_samples=args.validation_min_samples,
            unknown_max_multiplier=args.unknown_max_multiplier,
            max_class_ratio=args.max_class_ratio,
            max_samples_per_class=args.max_samples_per_class,
            auto_max_samples_per_class=args.auto_max_samples_per_class,
        )
        clf.save(args.model_path)
        info["model_path"] = args.model_path
        info["class_names"] = class_names
        info["command"] = "mc-train"
        info["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
        info["results_log"] = args.results_log
        _append_jsonl(args.results_log, info)
        print(json.dumps(info, indent=2))
        return

    if args.cmd == "mc-eval":
        clf = HTMLTargetMulticlassDOMClassifier(class_names=[UNKNOWN_CLASS_NAME], target_to_id={})
        clf.load(args.model_path)
        test_chunks = _resolve_split_chunks(args.feature_root, "test", args.test_chunks)
        metrics = clf.evaluate_from_feature_store(test_chunks, label_suffix=args.label_suffix)
        metrics["model_path"] = args.model_path
        metrics["algorithm"] = clf.algorithm
        metrics["command"] = "mc-eval"
        metrics["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
        metrics["results_log"] = args.results_log
        _append_jsonl(args.results_log, metrics)
        print(json.dumps(metrics, indent=2))
        return

    if args.cmd == "mc-predict":
        clf = HTMLTargetMulticlassDOMClassifier(class_names=[UNKNOWN_CLASS_NAME], target_to_id={})
        clf.load(args.model_path)
        pred = clf.predict_html(_read_text(args.html), top_k=args.top_k)
        pred["model_path"] = args.model_path
        pred["algorithm"] = clf.algorithm
        print(json.dumps(pred, indent=2))
        return


if __name__ == "__main__":
    main()
