#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Sampler, TensorDataset, WeightedRandomSampler
from tqdm import tqdm

DEFAULT_FEATURE_ROOT = "dom_unsup_subsample_combined"
DEFAULT_SUBSAMPLE_TEST_FEATURE_ROOT = "dom_unsup_subsample_combined"
DEFAULT_LABEL_SUFFIX = "_y11.npz"

logging.basicConfig(level=logging.INFO, format="[mlp_train] %(message)s")
logger = logging.getLogger(__name__)


def _expand_chunks(folder: Path, split: str) -> List[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Missing feature root: {folder}")
    if folder.is_dir() and (folder / split).exists():
        folder = folder / split
    chunks = sorted(
        p for p in folder.glob(f"{split}_domfeat_chunk_*.npz") if not p.name.endswith(DEFAULT_LABEL_SUFFIX)
    )
    if not chunks:
        raise ValueError(f"No {split} chunks found under {folder}")
    return chunks


def _load_split(chunk_paths: Sequence[Path], label_suffix: str = DEFAULT_LABEL_SUFFIX) -> Tuple[np.ndarray, np.ndarray]:
    x_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []

    for chunk_path in tqdm(chunk_paths, desc=f"Loading {chunk_paths[0].parent.name}"):
        with np.load(chunk_path, allow_pickle=False) as data:
            x_parts.append(np.asarray(data["X"], dtype=np.float32))
            if "y11" in data.files:
                y_parts.append(np.asarray(data["y11"], dtype=np.int64))
            else:
                sidecar = chunk_path.with_name(chunk_path.stem + label_suffix)
                with np.load(sidecar, allow_pickle=False) as sdata:
                    y_parts.append(np.asarray(sdata["y11"], dtype=np.int64))

    X = np.vstack(x_parts).astype(np.float32)
    y = np.concatenate(y_parts).astype(np.int64)
    return X, y


def _encode_labels(y: np.ndarray, class_to_index: dict[int, int], split_name: str) -> np.ndarray:
    uniq = np.unique(y)
    missing = [int(v) for v in uniq if int(v) not in class_to_index]
    if missing:
        raise ValueError(
            f"{split_name} split contains unseen labels not present in train classes: {missing}"
        )
    encoded = np.fromiter((class_to_index[int(v)] for v in y), dtype=np.int64, count=len(y))
    return encoded


def _remap_unseen_labels_to_unknown(
    y: np.ndarray,
    train_labels: Sequence[int],
    unknown_label: int,
    split_name: str,
) -> np.ndarray:
    """Map labels absent from train to the designated unknown label."""
    train_set = set(int(v) for v in train_labels)
    unseen = sorted(int(v) for v in np.unique(y) if int(v) not in train_set)
    if not unseen:
        return y
    out = np.asarray(y, dtype=np.int64).copy()
    for lbl in unseen:
        out[out == int(lbl)] = int(unknown_label)
    logger.info(
        f"{split_name}: remapped unseen labels {unseen} -> unknown_label={int(unknown_label)}"
    )
    return out


def _split_chunk_paths(chunk_paths: Sequence[Path], validation_fraction: float, seed: int) -> Tuple[List[Path], List[Path]]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError(f"validation_fraction must be in (0, 1), got {validation_fraction}")
    if len(chunk_paths) < 2:
        raise ValueError("Need at least two training chunks to create a chunk-level validation split")

    rng = np.random.default_rng(seed)
    indices = np.arange(len(chunk_paths))
    rng.shuffle(indices)

    val_count = max(1, int(round(len(chunk_paths) * validation_fraction)))
    val_count = min(val_count, len(chunk_paths) - 1)
    train_subset = [chunk_paths[i] for i in indices[val_count:]]
    val_subset = [chunk_paths[i] for i in indices[:val_count]]
    return train_subset, val_subset


def _collapse_to_top_known_classes(
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_subsample_test: np.ndarray,
    y_test: np.ndarray,
    known_classes: int,
    unknown_label: int | None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int], int]:
    if known_classes < 1:
        raise ValueError(f"known_classes must be >= 1, got {known_classes}")

    inferred_unknown = int(np.max(y_train)) if unknown_label is None else int(unknown_label)
    if inferred_unknown not in set(int(v) for v in np.unique(y_train)):
        inferred_unknown = int(np.max(y_train))

    train_counts = np.bincount(y_train.astype(np.int64))
    candidate_labels = [lbl for lbl in range(len(train_counts)) if lbl != inferred_unknown and train_counts[lbl] > 0]
    candidate_labels.sort(key=lambda lbl: int(train_counts[lbl]), reverse=True)

    selected_known = [int(lbl) for lbl in candidate_labels[:known_classes]]
    if len(selected_known) < known_classes:
        raise ValueError(
            f"Requested {known_classes} known classes but only {len(selected_known)} are available in train split"
        )

    known_to_new = {orig: idx for idx, orig in enumerate(selected_known)}
    new_unknown = known_classes

    def _map_split(y: np.ndarray) -> np.ndarray:
        mapped = np.full_like(y, fill_value=new_unknown, dtype=np.int64)
        for orig, new_id in known_to_new.items():
            mapped[y == orig] = new_id
        return mapped

    return _map_split(y_train), _map_split(y_val), _map_split(y_subsample_test), _map_split(y_test), selected_known, inferred_unknown


def _format_distribution(y: np.ndarray) -> str:
    if y.size == 0:
        return "empty"
    counts = np.bincount(y.astype(np.int64))
    total = int(counts.sum())
    parts = []
    for cls, cnt in enumerate(counts.tolist()):
        if cnt <= 0:
            continue
        pct = 100.0 * float(cnt) / float(total)
        parts.append(f"{cls}:{int(cnt):,} ({pct:.2f}%)")
    return ", ".join(parts)


def _log_split_distribution(name: str, y: np.ndarray) -> None:
    logger.info(f"{name} distribution: {_format_distribution(y)}")


def _augment_with_synthetic_minority_samples(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    fill_fraction: float,
    max_multiplier: float,
    noise_std: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, int]]:
    """Create a small number of synthetic samples by interpolating within each minority class.

    This is intentionally conservative: it only fills a fraction of the gap to the median class count,
    and caps the synthetic count relative to the original class size.
    """
    if fill_fraction <= 0 or max_multiplier <= 0 or noise_std < 0:
        return X, y, {}

    rng = np.random.default_rng(seed)
    counts = np.bincount(y.astype(np.int64))
    median_count = max(1, int(np.median(counts[counts > 0])))

    synth_parts_x: List[np.ndarray] = []
    synth_parts_y: List[np.ndarray] = []
    synth_counts: Dict[int, int] = {}

    for cls, count in enumerate(counts.tolist()):
        cls = int(cls)
        count = int(count)
        if count < 2 or count >= median_count:
            continue

        gap = median_count - count
        synth_n = int(round(gap * fill_fraction))
        synth_n = min(synth_n, int(round(count * max_multiplier)))
        if synth_n <= 0:
            continue

        cls_idx = np.flatnonzero(y == cls)
        if len(cls_idx) < 2:
            continue

        x_cls = X[cls_idx]
        x1 = x_cls[rng.integers(0, len(x_cls), size=synth_n)]
        x2 = x_cls[rng.integers(0, len(x_cls), size=synth_n)]
        lam = rng.uniform(0.35, 0.65, size=(synth_n, 1)).astype(np.float32)
        x_syn = lam * x1 + (1.0 - lam) * x2
        if noise_std > 0:
            x_syn = x_syn + rng.normal(0.0, noise_std, size=x_syn.shape).astype(np.float32)

        y_syn = np.full((synth_n,), cls, dtype=np.int64)
        synth_parts_x.append(x_syn.astype(np.float32))
        synth_parts_y.append(y_syn)
        synth_counts[cls] = synth_n

    if not synth_parts_x:
        return X, y, {}

    X_aug = np.vstack([X] + synth_parts_x).astype(np.float32)
    y_aug = np.concatenate([y] + synth_parts_y).astype(np.int64)
    return X_aug, y_aug, synth_counts


class ClassBalancedBatchSampler(Sampler[List[int]]):
    def __init__(self, labels: np.ndarray, batch_size: int, seed: int) -> None:
        self.labels = np.asarray(labels, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.classes = np.unique(self.labels).astype(np.int64).tolist()
        if self.batch_size < len(self.classes):
            raise ValueError(
                f"batch_size={self.batch_size} is smaller than the number of classes={len(self.classes)}"
            )
        self.class_to_indices = {
            int(cls): np.flatnonzero(self.labels == int(cls)).astype(np.int64).tolist() for cls in self.classes
        }
        if any(len(v) == 0 for v in self.class_to_indices.values()):
            missing = [int(cls) for cls, idxs in self.class_to_indices.items() if len(idxs) == 0]
            raise ValueError(f"Each batch requires at least one example per class, but these classes are empty: {missing}")
        self._num_batches = int(math.ceil(len(self.labels) / float(self.batch_size)))

    def __len__(self) -> int:
        return self._num_batches

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        class_pools = {}
        class_counts = {}

        for cls, indices in self.class_to_indices.items():
            shuffled = np.asarray(indices, dtype=np.int64)
            rng.shuffle(shuffled)
            class_pools[int(cls)] = shuffled.tolist()
            class_counts[int(cls)] = len(shuffled)

        base_quota = {int(cls): 1 for cls in self.classes}
        remaining = self.batch_size - len(self.classes)
        if remaining > 0:
            inv_sqrt = np.asarray([1.0 / math.sqrt(max(class_counts[int(cls)], 1)) for cls in self.classes], dtype=np.float64)
            probs = inv_sqrt / inv_sqrt.sum()
            extra = rng.multinomial(remaining, probs)
            quotas = {int(cls): base_quota[int(cls)] + int(extra[idx]) for idx, cls in enumerate(self.classes)}
        else:
            quotas = base_quota

        for _ in range(self._num_batches):
            batch: List[int] = []
            for cls in self.classes:
                need = int(quotas[int(cls)])
                pool = class_pools[int(cls)]
                for _ in range(need):
                    if not pool:
                        replenished = np.asarray(self.class_to_indices[int(cls)], dtype=np.int64)
                        rng.shuffle(replenished)
                        pool.extend(replenished.tolist())
                    batch.append(int(pool.pop()))
            rng.shuffle(batch)
            yield batch


class MLPClassifierNet(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_sizes: Sequence[int] = (512, 256), dropout: float = 0.25):
        super().__init__()
        layers: List[nn.Module] = []
        last_dim = input_dim
        for hidden in hidden_sizes:
            layers.append(nn.Linear(last_dim, hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            last_dim = hidden
        layers.append(nn.Linear(last_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNN1DClassifierNet(nn.Module):
    """1D-CNN over hashed feature vectors treated as a 1D signal."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        channels: Sequence[int] = (64, 128, 256),
        kernel_size: int = 5,
        dropout: float = 0.25,
    ):
        super().__init__()
        if len(channels) < 1:
            raise ValueError("channels must contain at least one value")
        k = max(3, int(kernel_size))
        if k % 2 == 0:
            k += 1

        layers: List[nn.Module] = []
        in_ch = 1
        for out_ch in channels:
            layers.append(nn.Conv1d(in_ch, int(out_ch), kernel_size=k, padding=k // 2))
            layers.append(nn.BatchNorm1d(int(out_ch)))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
            layers.append(nn.Dropout(dropout))
            in_ch = int(out_ch)
        self.features = nn.Sequential(*layers)
        # Global pooling keeps parameter count small and avoids OOM from huge flatten+dense tensors.
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(in_ch, max(64, in_ch // 2)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(64, in_ch // 2), num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        feats = self.features(x)
        feats = self.pool(feats).squeeze(-1)
        return self.classifier(feats)


@dataclass
class Metrics:
    loss: float
    accuracy: float
    macro_f1: float


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = torch.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)
        targets = targets.long()
        idx = torch.arange(targets.shape[0], device=targets.device)
        pt = probs[idx, targets]
        loss = -((1.0 - pt).pow(self.gamma)) * log_probs[idx, targets]
        if self.weight is not None:
            loss = loss * self.weight[targets]
        return loss.mean()


def _apply_unknown_threshold(preds: torch.Tensor, probs: torch.Tensor, unknown_class_id: int | None, unknown_threshold: float | None) -> torch.Tensor:
    if unknown_class_id is None or unknown_threshold is None:
        return preds
    if unknown_class_id < 0 or unknown_class_id >= probs.shape[1]:
        return preds
    max_probs, _ = torch.max(probs, dim=1)
    out = preds.clone()
    out[max_probs < float(unknown_threshold)] = int(unknown_class_id)
    return out


def _apply_class_thresholds_np(
    preds: np.ndarray,
    probs: np.ndarray,
    class_thresholds: np.ndarray | None,
    unknown_class_id: int | None,
) -> np.ndarray:
    """Apply per-class minimum confidence thresholds; fallback to unknown when below threshold."""
    if class_thresholds is None or unknown_class_id is None:
        return preds
    out = preds.astype(np.int64).copy()
    for i in range(len(out)):
        cls = int(out[i])
        if cls < 0 or cls >= len(class_thresholds):
            continue
        if float(probs[i, cls]) < float(class_thresholds[cls]):
            out[i] = int(unknown_class_id)
    return out


@torch.no_grad()
def _collect_probs_targets(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        probs = torch.softmax(logits, dim=1)
        all_probs.append(probs.detach().cpu().numpy())
        all_targets.append(yb.detach().cpu().numpy())
    if not all_probs:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64)
    return np.concatenate(all_probs).astype(np.float32), np.concatenate(all_targets).astype(np.int64)


def _tune_class_thresholds(
    y_true: np.ndarray,
    probs: np.ndarray,
    unknown_class_id: int,
    grid: np.ndarray,
    rounds: int = 2,
) -> Tuple[np.ndarray, float]:
    """Coordinate-search class thresholds to maximize macro-F1 on validation set."""
    n_classes = int(probs.shape[1])
    thresholds = np.zeros((n_classes,), dtype=np.float32)

    def _score(th: np.ndarray) -> float:
        pred = np.argmax(probs, axis=1).astype(np.int64)
        pred = _apply_class_thresholds_np(pred, probs, th, unknown_class_id)
        return float(f1_score(y_true, pred, average="macro", zero_division=0))

    best = _score(thresholds)
    for _ in range(max(1, int(rounds))):
        improved = False
        for cls in range(n_classes):
            cls_best_t = float(thresholds[cls])
            cls_best_score = best
            for t in grid.tolist():
                cand = thresholds.copy()
                cand[cls] = float(t)
                s = _score(cand)
                if s > cls_best_score + 1e-8:
                    cls_best_score = s
                    cls_best_t = float(t)
            if cls_best_score > best + 1e-8:
                thresholds[cls] = cls_best_t
                best = cls_best_score
                improved = True
        if not improved:
            break
    return thresholds.astype(np.float32), float(best)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    unknown_class_id: int | None = None,
    unknown_threshold: float | None = None,
    class_thresholds: np.ndarray | None = None,
) -> Metrics:
    model.eval()
    total_loss = 0.0
    all_preds: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []
    total = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        total_loss += float(loss.item()) * len(xb)
        total += len(xb)

        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        preds = _apply_unknown_threshold(preds, probs, unknown_class_id, unknown_threshold)
        all_preds.append(preds.detach().cpu().numpy())
        all_targets.append(yb.detach().cpu().numpy())

    if total == 0:
        return Metrics(loss=float("nan"), accuracy=float("nan"), macro_f1=float("nan"))

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    if class_thresholds is not None and unknown_class_id is not None:
        # Recompute probabilities once for calibrated threshold application.
        probs, y_t = _collect_probs_targets(model, loader, device)
        y_pred = _apply_class_thresholds_np(np.argmax(probs, axis=1), probs, class_thresholds, unknown_class_id)
        y_true = y_t
    return Metrics(
        loss=total_loss / total,
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    )


@torch.no_grad()
def _predict_and_metrics(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    unknown_class_id: int | None = None,
    unknown_threshold: float | None = None,
    class_thresholds: np.ndarray | None = None,
) -> Tuple[Metrics, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    all_preds: List[np.ndarray] = []
    all_probs: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []
    total = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        total_loss += float(loss.item()) * len(xb)
        total += len(xb)

        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        preds = _apply_unknown_threshold(preds, probs, unknown_class_id, unknown_threshold)
        all_preds.append(preds.detach().cpu().numpy())
        all_probs.append(probs.detach().cpu().numpy())
        all_targets.append(yb.detach().cpu().numpy())

    if total == 0:
        metrics = Metrics(loss=float("nan"), accuracy=float("nan"), macro_f1=float("nan"))
        return metrics, np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64)

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    if class_thresholds is not None and unknown_class_id is not None:
        y_prob = np.concatenate(all_probs)
        y_pred = _apply_class_thresholds_np(np.argmax(y_prob, axis=1), y_prob, class_thresholds, unknown_class_id)
    metrics = Metrics(
        loss=total_loss / total,
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    )
    return metrics, y_true, y_pred


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    root = Path(args.feature_root)
    subsample_test_root = Path(args.subsample_test_feature_root) if args.subsample_test_feature_root else root
    train_chunks = _expand_chunks(root, "train")
    subsample_test_chunks = _expand_chunks(subsample_test_root, "test")
    use_full_test = bool(args.evaluate_full_test and args.test_feature_root)
    test_chunks: List[Path] = []
    if use_full_test:
        test_root = Path(args.test_feature_root)
        test_chunks = _expand_chunks(test_root, "test")

    if args.validation_by_chunk:
        train_chunk_paths, val_chunk_paths = _split_chunk_paths(train_chunks, args.validation_fraction, args.seed)
        X_train, y_train_raw = _load_split(train_chunk_paths, label_suffix=args.label_suffix)
        X_val, y_val_raw = _load_split(val_chunk_paths, label_suffix=args.label_suffix)
        X_train_full = X_train
    else:
        X_train_full, y_train_full = _load_split(train_chunks, label_suffix=args.label_suffix)
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_full,
                y_train_full,
                test_size=args.validation_fraction,
                random_state=args.seed,
                stratify=y_train_full,
            )
        except Exception:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_full,
                y_train_full,
                test_size=args.validation_fraction,
                random_state=args.seed,
                stratify=None,
            )
        y_train_raw = y_train
        y_val_raw = y_val

    X_subsample_test, y_subsample_test = _load_split(subsample_test_chunks, label_suffix=args.label_suffix)
    X_test: np.ndarray | None = None
    y_test: np.ndarray | None = None
    if use_full_test:
        X_test, y_test = _load_split(test_chunks, label_suffix=args.label_suffix)

    _log_split_distribution("raw-train", y_train_raw)
    _log_split_distribution("raw-subsample-test", y_subsample_test)
    if y_test is not None:
        _log_split_distribution("raw-test", y_test)

    classes = np.unique(y_train_raw)
    inferred_unknown_for_encoding = int(args.unknown_label)
    if inferred_unknown_for_encoding not in set(int(v) for v in classes.tolist()):
        inferred_unknown_for_encoding = int(np.max(classes))

    y_subsample_test = _remap_unseen_labels_to_unknown(
        y_subsample_test,
        train_labels=classes.tolist(),
        unknown_label=inferred_unknown_for_encoding,
        split_name="subsample test",
    )
    if y_test is not None:
        y_test = _remap_unseen_labels_to_unknown(
            y_test,
            train_labels=classes.tolist(),
            unknown_label=inferred_unknown_for_encoding,
            split_name="test",
        )

    class_to_index = {int(label): idx for idx, label in enumerate(classes.tolist())}
    y_train = _encode_labels(y_train_raw, class_to_index, split_name="train")
    y_val = _encode_labels(y_val_raw, class_to_index, split_name="validation")
    y_subsample_test = _encode_labels(y_subsample_test, class_to_index, split_name="subsample test")
    if y_test is not None:
        y_test = _encode_labels(y_test, class_to_index, split_name="test")

    selected_known_original: List[int] | None = None
    inferred_unknown_original: int | None = None
    if args.known_classes > 0:
        y_train, y_val, y_subsample_test, y_test, selected_known_original, inferred_unknown_original = _collapse_to_top_known_classes(
            y_train,
            y_val,
            y_subsample_test,
            y_test if y_test is not None else y_subsample_test,
            known_classes=args.known_classes,
            unknown_label=args.unknown_label,
        )

        if y_test is None:
            y_test = None

        classes = np.arange(args.known_classes + 1, dtype=np.int64)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_subsample_test = scaler.transform(X_subsample_test).astype(np.float32)
    if X_test is not None:
        X_test = scaler.transform(X_test).astype(np.float32)

    synth_counts: Dict[int, int] = {}
    if args.synthetic_oversampling:
        X_train, y_train, synth_counts = _augment_with_synthetic_minority_samples(
            X_train,
            y_train,
            seed=args.seed,
            fill_fraction=args.synthetic_fill_fraction,
            max_multiplier=args.synthetic_max_multiplier,
            noise_std=args.synthetic_noise_std,
        )
        if synth_counts:
            logger.info(
                "Synthetic augmentation added: "
                + ", ".join(f"class {cls} +{cnt}" for cls, cnt in sorted(synth_counts.items()))
            )

    train_counts = np.bincount(y_train, minlength=len(classes)).astype(np.float32)
    inv_freq = 1.0 / np.maximum(train_counts, 1.0)
    class_weights_np = inv_freq / np.mean(inv_freq)

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    subsample_test_ds = TensorDataset(torch.from_numpy(X_subsample_test), torch.from_numpy(y_subsample_test))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)) if (X_test is not None and y_test is not None) else None

    train_batch_sampler = None
    if args.balanced_batches:
        train_batch_sampler = ClassBalancedBatchSampler(y_train, batch_size=args.batch_size, seed=args.seed)
        train_loader = DataLoader(train_ds, batch_sampler=train_batch_sampler)
    else:
        train_sampler = None
        if args.use_weighted_sampler:
            sample_weights = class_weights_np[y_train]
            train_sampler = WeightedRandomSampler(
                weights=torch.from_numpy(sample_weights).double(),
                num_samples=len(sample_weights),
                replacement=True,
            )
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            drop_last=False,
        )

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    subsample_test_loader = DataLoader(subsample_test_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False) if test_ds is not None else None

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    num_classes = int(len(classes))
    if args.model_type == "cnn1d":
        model = CNN1DClassifierNet(
            input_dim=X_train.shape[1],
            num_classes=num_classes,
            channels=tuple(args.cnn_channels),
            kernel_size=args.cnn_kernel_size,
            dropout=args.dropout,
        ).to(device)
    else:
        model = MLPClassifierNet(
            input_dim=X_train.shape[1],
            num_classes=num_classes,
            hidden_sizes=tuple(args.hidden_sizes),
            dropout=args.dropout,
        ).to(device)

    class_weights = None
    if args.class_weighting:
        class_weights = torch.tensor(class_weights_np, dtype=torch.float32, device=device)
    if args.loss_type == "focal":
        criterion = FocalLoss(gamma=args.focal_gamma, weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.monitor not in {"balanced_val_loss", "balanced_val_macro_f1"}:
        raise ValueError("Validation is now balanced-only; use balanced_val_loss or balanced_val_macro_f1 for --monitor")
    monitor_mode = "max" if args.monitor.endswith("macro_f1") else "min"
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=monitor_mode, factor=0.5, patience=3)

    best_metric = float("-inf") if monitor_mode == "max" else float("inf")
    best_epoch = 0
    best_state = None
    history = []
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * len(xb)
            running_total += len(xb)
            running_correct += int((logits.argmax(dim=1) == yb).sum().item())

        train_loss = running_loss / max(running_total, 1)
        train_acc = running_correct / max(running_total, 1)
        train_metrics = Metrics(
            loss=train_loss,
            accuracy=train_acc,
            macro_f1=float("nan"),
        )
        balanced_val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
            unknown_class_id=(num_classes - 1 if args.unknown_threshold is not None else None),
            unknown_threshold=args.unknown_threshold,
        )
        monitored_value = balanced_val_metrics.macro_f1 if args.monitor == "balanced_val_macro_f1" else balanced_val_metrics.loss
        scheduler.step(monitored_value)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics.loss,
                "train_accuracy": train_metrics.accuracy,
                "balanced_val_loss": balanced_val_metrics.loss,
                "balanced_val_accuracy": balanced_val_metrics.accuracy,
                "balanced_val_macro_f1": balanced_val_metrics.macro_f1,
            }
        )

        print(
            f"epoch={epoch:03d}/{args.epochs} "
            f"train_loss={train_metrics.loss:.4f} train_acc={train_metrics.accuracy:.4f} "
            f"val_loss={balanced_val_metrics.loss:.4f} val_acc={balanced_val_metrics.accuracy:.4f} val_macro_f1={balanced_val_metrics.macro_f1:.4f} "
            f"monitor={monitored_value:.4f}",
            flush=True,
        )

        if monitor_mode == "max":
            improved = monitored_value > (best_metric + args.min_delta)
        else:
            improved = monitored_value < (best_metric - args.min_delta)

        if improved:
            best_metric = monitored_value
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state = {
                "model": model.state_dict(),
                "scaler_mean": scaler.mean_,
                "scaler_scale": scaler.scale_,
                "model_type": args.model_type,
                "input_dim": X_train.shape[1],
                "num_classes": num_classes,
                "hidden_sizes": list(args.hidden_sizes),
                "cnn_channels": list(args.cnn_channels),
                "cnn_kernel_size": int(args.cnn_kernel_size),
                "dropout": args.dropout,
                "feature_root": args.feature_root,
                "label_suffix": args.label_suffix,
                "classes": [int(v) for v in classes.tolist()],
            }
        else:
            epochs_without_improvement += 1

        if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
            print(
                f"Early stopping at epoch {epoch} after {epochs_without_improvement} epochs without improvement "
                f"on {args.monitor}.",
                flush=True,
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state["model"])

    calibrated_thresholds: np.ndarray | None = None
    calibration_macro_f1 = None
    if args.enable_class_threshold_tuning:
        val_probs, val_targets = _collect_probs_targets(model, val_loader, device)
        if len(val_targets) > 0:
            unknown_id = num_classes - 1
            grid = np.linspace(args.class_threshold_min, args.class_threshold_max, num=args.class_threshold_steps, dtype=np.float32)
            calibrated_thresholds, calibration_macro_f1 = _tune_class_thresholds(
                y_true=val_targets,
                probs=val_probs,
                unknown_class_id=unknown_id,
                grid=grid,
                rounds=args.class_threshold_rounds,
            )
            logger.info(
                "Calibrated class thresholds macro_f1=%.4f thresholds=%s",
                calibration_macro_f1,
                [round(float(v), 3) for v in calibrated_thresholds.tolist()],
            )

    subsample_test_metrics, y_subsample_test_true, y_subsample_test_pred = _predict_and_metrics(
        model,
        subsample_test_loader,
        criterion,
        device,
        unknown_class_id=(num_classes - 1 if args.unknown_threshold is not None else None),
        unknown_threshold=args.unknown_threshold,
        class_thresholds=calibrated_thresholds,
    )
    test_metrics = None
    y_test_true = np.asarray([], dtype=np.int64)
    y_test_pred = np.asarray([], dtype=np.int64)
    if test_loader is not None:
        test_metrics, y_test_true, y_test_pred = _predict_and_metrics(
            model,
            test_loader,
            criterion,
            device,
            unknown_class_id=(num_classes - 1 if args.unknown_threshold is not None else None),
            unknown_threshold=args.unknown_threshold,
            class_thresholds=calibrated_thresholds,
        )
    target_names = [str(int(v)) for v in classes.tolist()]
    if selected_known_original is not None:
        target_names = [f"known_{orig}" for orig in selected_known_original] + ["unknown_agg"]

    subsample_report = classification_report(
        y_subsample_test_true,
        y_subsample_test_pred,
        labels=list(range(num_classes)),
        target_names=target_names,
        zero_division=0,
        output_dict=True,
    )
    report = None
    if len(y_test_true) > 0:
        report = classification_report(
            y_test_true,
            y_test_pred,
            labels=list(range(num_classes)),
            target_names=target_names,
            zero_division=0,
            output_dict=True,
        )
    subsample_cm = confusion_matrix(y_subsample_test_true, y_subsample_test_pred, labels=list(range(num_classes)))
    cm = confusion_matrix(y_test_true, y_test_pred, labels=list(range(num_classes))) if len(y_test_true) > 0 else None

    save_path = Path(args.output_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "model_type": args.model_type,
            "input_dim": X_train.shape[1],
            "num_classes": num_classes,
            "hidden_sizes": list(args.hidden_sizes),
            "cnn_channels": list(args.cnn_channels),
            "cnn_kernel_size": int(args.cnn_kernel_size),
            "dropout": args.dropout,
            "feature_root": args.feature_root,
            "label_suffix": args.label_suffix,
            "class_weighting": args.class_weighting,
            "use_weighted_sampler": args.use_weighted_sampler,
            "class_weights": class_weights_np.tolist(),
            "classes": [int(v) for v in classes.tolist()],
            "known_classes": args.known_classes,
            "selected_known_original": selected_known_original,
            "inferred_unknown_original": inferred_unknown_original,
            "history": history,
        },
        save_path,
    )

    print(
        json.dumps(
            {
                "best_epoch": best_epoch,
                "best_monitor_metric": best_metric,
                "monitor": args.monitor,
                "subsample_test_loss": subsample_test_metrics.loss,
                "subsample_test_accuracy": subsample_test_metrics.accuracy,
                "subsample_test_macro_f1": subsample_test_metrics.macro_f1,
                "class_threshold_tuning_enabled": bool(args.enable_class_threshold_tuning),
                "class_thresholds": (calibrated_thresholds.tolist() if calibrated_thresholds is not None else None),
                "class_threshold_val_macro_f1": calibration_macro_f1,
                "test_loss": (test_metrics.loss if test_metrics is not None else None),
                "test_accuracy": (test_metrics.accuracy if test_metrics is not None else None),
                "test_macro_f1": (test_metrics.macro_f1 if test_metrics is not None else None),
                "full_test_evaluated": bool(test_metrics is not None),
                "model_path": str(save_path),
                "subsample_confusion_matrix": subsample_cm.tolist(),
                "confusion_matrix": (cm.tolist() if cm is not None else None),
                "selected_known_original": selected_known_original,
                "inferred_unknown_original": inferred_unknown_original,
                "subsample_report": subsample_report,
                "report": report,
            },
            indent=2,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MLP training on phishing DOM feature subsamples")
    parser.add_argument("--feature-root", default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--subsample-test-feature-root", default=DEFAULT_SUBSAMPLE_TEST_FEATURE_ROOT)
    parser.add_argument("--test-feature-root", default="")
    parser.add_argument(
        "--evaluate-full-test",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Evaluate additionally on --test-feature-root/test; disabled by default",
    )
    parser.add_argument("--label-suffix", default=DEFAULT_LABEL_SUFFIX)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument(
        "--known-classes",
        type=int,
        default=9,
        help="Collapse task to the top-K known classes + one aggregated unknown class; set 0 to disable",
    )
    parser.add_argument(
        "--unknown-label",
        type=int,
        default=10,
        help="Original encoded label treated as unknown before collapsing; ignored when --known-classes=0",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--class-weighting", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-weighted-sampler", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--balanced-batches", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--synthetic-oversampling", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--synthetic-fill-fraction", type=float, default=0.15)
    parser.add_argument("--synthetic-max-multiplier", type=float, default=1.05)
    parser.add_argument("--synthetic-noise-std", type=float, default=0.01)
    parser.add_argument("--loss-type", choices=["cross_entropy", "focal"], default="focal")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--unknown-threshold", type=float, default=0.60)
    parser.add_argument(
        "--monitor",
        choices=["balanced_val_loss", "balanced_val_macro_f1"],
        default="balanced_val_macro_f1",
    )
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--validation-by-chunk", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="")
    parser.add_argument("--output-path", default="mlp_subsample_model.pt")
    parser.add_argument("--hidden-sizes", nargs="+", type=int, default=[512, 256])
    parser.add_argument("--model-type", choices=["mlp", "cnn1d"], default="mlp")
    parser.add_argument("--cnn-channels", nargs="+", type=int, default=[64, 128, 256])
    parser.add_argument("--cnn-kernel-size", type=int, default=5)
    parser.add_argument("--enable-class-threshold-tuning", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--class-threshold-min", type=float, default=0.0)
    parser.add_argument("--class-threshold-max", type=float, default=0.6)
    parser.add_argument("--class-threshold-steps", type=int, default=7)
    parser.add_argument("--class-threshold-rounds", type=int, default=2)
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
