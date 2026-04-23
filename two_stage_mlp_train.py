#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from mlp_subsample_train import (
    ClassBalancedBatchSampler,
    FocalLoss,
    MLPClassifierNet,
    _augment_with_synthetic_minority_samples,
    _encode_labels,
    _expand_chunks,
    _load_split,
    _split_chunk_paths,
)

DEFAULT_FEATURE_ROOT = "dom_unsup_subsample"
DEFAULT_SUBSAMPLE_TEST_ROOT = "dom_unsup_subsample"
DEFAULT_TEST_ROOT = "dom_unsup_features"
DEFAULT_LABEL_SUFFIX = "_y11.npz"
DEFAULT_UNKNOWN_LABEL = 10


@dataclass
class Metrics:
    loss: float
    accuracy: float
    macro_f1: float


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
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
        preds = torch.argmax(logits, dim=1)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(yb.cpu().numpy())

    if total == 0:
        return Metrics(loss=float("nan"), accuracy=float("nan"), macro_f1=float("nan"))

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    return Metrics(
        loss=total_loss / total,
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    )


@torch.no_grad()
def _predict(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[Metrics, np.ndarray, np.ndarray]:
    metrics = _evaluate(model, loader, criterion, device)
    y_true_parts: List[np.ndarray] = []
    y_pred_parts: List[np.ndarray] = []
    model.eval()

    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        preds = torch.argmax(logits, dim=1)
        y_true_parts.append(yb.cpu().numpy())
        y_pred_parts.append(preds.cpu().numpy())

    if not y_true_parts:
        return metrics, np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64)
    return metrics, np.concatenate(y_true_parts), np.concatenate(y_pred_parts)


class TwoStageMLP:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.known_original_labels: List[int] = []
        self.unknown_original_label: int = int(args.unknown_label)
        self.scaler: StandardScaler | None = None
        self.stage1_model: nn.Module | None = None
        self.stage2_model: nn.Module | None = None
        self.stage1_classes: List[int] = [0, 1]
        self.stage2_classes: List[int] = []
        self.stage1_input_dim = 0
        self.stage2_input_dim = 0
        self.selected_unknown_threshold = float(args.unknown_threshold)
        self.selected_stage2_class_thresholds: np.ndarray | None = None

    def _augment_with_targeted_boundary_samples(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[int, int]]:
        """Augment minority known classes using boundary-focused in-class interpolation."""
        if (
            self.args.synthetic_fill_fraction <= 0
            or self.args.synthetic_max_multiplier <= 0
            or self.args.synthetic_noise_std < 0
        ):
            return X, y, {}

        rng = np.random.default_rng(self.args.seed)
        counts = np.bincount(y.astype(np.int64))
        nonzero = counts[counts > 0]
        if len(nonzero) == 0:
            return X, y, {}
        median_count = max(1, int(np.median(nonzero)))

        classes = np.flatnonzero(counts > 0).astype(np.int64)
        centroids = {int(c): X[y == int(c)].mean(axis=0) for c in classes.tolist()}

        synth_x_parts: List[np.ndarray] = []
        synth_y_parts: List[np.ndarray] = []
        synth_counts: Dict[int, int] = {}

        for cls in classes.tolist():
            cls = int(cls)
            count = int(counts[cls])
            if count < 3 or count >= median_count:
                continue

            gap = median_count - count
            synth_n = int(round(gap * float(self.args.synthetic_fill_fraction)))
            synth_n = min(synth_n, int(round(count * float(self.args.synthetic_max_multiplier))))
            if synth_n <= 0:
                continue

            x_cls = X[y == cls]
            other_centroids = [v for k, v in centroids.items() if k != cls]
            if not other_centroids:
                continue

            other_centers = np.vstack(other_centroids)
            # Boundary anchors are class samples closest to another class centroid.
            dists = np.linalg.norm(x_cls[:, None, :] - other_centers[None, :, :], axis=2).min(axis=1)
            anchor_count = max(2, int(round(len(x_cls) * float(self.args.boundary_focus_fraction))))
            anchor_count = min(anchor_count, len(x_cls))
            anchor_idx = np.argsort(dists)[:anchor_count]
            anchors = x_cls[anchor_idx]

            x1 = anchors[rng.integers(0, len(anchors), size=synth_n)]
            x2 = x_cls[rng.integers(0, len(x_cls), size=synth_n)]
            lam = rng.uniform(0.4, 0.75, size=(synth_n, 1)).astype(np.float32)
            x_syn = lam * x1 + (1.0 - lam) * x2
            if self.args.synthetic_noise_std > 0:
                x_syn = x_syn + rng.normal(0.0, self.args.synthetic_noise_std, size=x_syn.shape).astype(np.float32)

            synth_x_parts.append(x_syn.astype(np.float32))
            synth_y_parts.append(np.full((synth_n,), cls, dtype=np.int64))
            synth_counts[cls] = synth_n

        if not synth_x_parts:
            return X, y, {}

        X_aug = np.vstack([X] + synth_x_parts).astype(np.float32)
        y_aug = np.concatenate([y] + synth_y_parts).astype(np.int64)
        return X_aug, y_aug, synth_counts

    def _build_loaders(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_subsample_test: np.ndarray,
        y_subsample_test: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        known_classes: int,
    ) -> Dict[str, object]:
        classes = np.unique(y_train)
        class_to_index = {int(label): idx for idx, label in enumerate(classes.tolist())}
        y_train = _encode_labels(y_train, class_to_index, split_name="train")
        y_val = _encode_labels(y_val, class_to_index, split_name="validation")
        y_subsample_test = _encode_labels(y_subsample_test, class_to_index, split_name="subsample test")
        y_test = _encode_labels(y_test, class_to_index, split_name="test")

        if known_classes < 1:
            raise ValueError("known_classes must be >= 1")

        train_counts = np.bincount(y_train.astype(np.int64))
        candidate_labels = [lbl for lbl in range(len(train_counts)) if lbl != self.unknown_original_label and train_counts[lbl] > 0]
        candidate_labels.sort(key=lambda lbl: int(train_counts[lbl]), reverse=True)
        selected_known = [int(lbl) for lbl in candidate_labels[:known_classes]]
        if len(selected_known) < known_classes:
            raise ValueError(
                f"Requested {known_classes} known classes but only {len(selected_known)} are available in train split"
            )
        self.known_original_labels = selected_known

        known_to_idx = {orig: i for i, orig in enumerate(selected_known)}
        unknown_idx = known_classes

        def pipeline_map(y: np.ndarray) -> np.ndarray:
            out = np.full_like(y, fill_value=unknown_idx, dtype=np.int64)
            for orig, idx in known_to_idx.items():
                out[y == orig] = idx
            return out

        def stage1_map(y: np.ndarray) -> np.ndarray:
            out = np.zeros_like(y, dtype=np.int64)
            out[np.isin(y, selected_known)] = 1
            return out

        def stage2_map(y: np.ndarray) -> np.ndarray:
            keep = np.isin(y, selected_known)
            return np.asarray([known_to_idx[int(v)] for v in y[keep]], dtype=np.int64), keep

        y_pipeline_train = pipeline_map(y_train)
        y_pipeline_val = pipeline_map(y_val)
        y_pipeline_sub_test = pipeline_map(y_subsample_test)
        y_pipeline_test = pipeline_map(y_test)

        y1_train = stage1_map(y_train)
        y1_val = stage1_map(y_val)
        y1_sub_test = stage1_map(y_subsample_test)
        y1_test = stage1_map(y_test)

        y2_train, keep_train = stage2_map(y_train)
        y2_val, keep_val = stage2_map(y_val)
        y2_sub_test, keep_sub_test = stage2_map(y_subsample_test)
        y2_test, keep_test = stage2_map(y_test)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train).astype(np.float32)
        X_val = scaler.transform(X_val).astype(np.float32)
        X_subsample_test = scaler.transform(X_subsample_test).astype(np.float32)
        X_test = scaler.transform(X_test).astype(np.float32)
        self.scaler = scaler

        # Conservative synthetic augmentation for stage 2 only.
        synth_counts: Dict[int, int] = {}
        if self.args.synthetic_oversampling:
            X2_train_raw = X_train[keep_train]
            y2_train_raw = y2_train
            if self.args.targeted_synthetic_oversampling:
                X2_train_aug, y2_train_aug, synth_counts = self._augment_with_targeted_boundary_samples(
                    X2_train_raw,
                    y2_train_raw,
                )
            else:
                X2_train_aug, y2_train_aug, synth_counts = _augment_with_synthetic_minority_samples(
                    X2_train_raw,
                    y2_train_raw,
                    seed=self.args.seed,
                    fill_fraction=self.args.synthetic_fill_fraction,
                    max_multiplier=self.args.synthetic_max_multiplier,
                    noise_std=self.args.synthetic_noise_std,
                )
        else:
            X2_train_aug, y2_train_aug = X_train[keep_train], y2_train

        self.stage1_input_dim = X_train.shape[1]
        self.stage2_input_dim = X2_train_aug.shape[1]
        self.stage2_classes = selected_known

        loaders = {
            "train_stage1": TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y1_train)),
            "val_stage1": TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y1_val)),
            "subsample_test_stage1": TensorDataset(torch.from_numpy(X_subsample_test), torch.from_numpy(y1_sub_test)),
            "test_stage1": TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y1_test)),
            "train_stage2": TensorDataset(torch.from_numpy(X2_train_aug), torch.from_numpy(y2_train_aug)),
            "val_stage2": TensorDataset(torch.from_numpy(X_val[keep_val]), torch.from_numpy(y2_val)),
            "subsample_test_stage2": TensorDataset(torch.from_numpy(X_subsample_test[keep_sub_test]), torch.from_numpy(y2_sub_test)),
            "test_stage2": TensorDataset(torch.from_numpy(X_test[keep_test]), torch.from_numpy(y2_test)),
            "synth_counts": synth_counts,
            "selected_known": selected_known,
            "stage1_unknown_idx": unknown_idx,
            "pipeline_train": y_pipeline_train,
            "pipeline_val": y_pipeline_val,
            "pipeline_subsample_test": y_pipeline_sub_test,
            "pipeline_test": y_pipeline_test,
            "y_val_stage2_mask": keep_val,
            "y_subsample_test_stage2_mask": keep_sub_test,
            "y_test_stage2_mask": keep_test,
        }
        return loaders

    def _train_stage(
        self,
        train_ds: TensorDataset,
        val_ds: TensorDataset,
        num_classes: int,
        stage_name: str,
        classes_for_weights: Sequence[int],
        class_weights_np: np.ndarray,
    ) -> Tuple[nn.Module, List[dict], float, int]:
        model = MLPClassifierNet(
            input_dim=train_ds.tensors[0].shape[1],
            num_classes=num_classes,
            hidden_sizes=tuple(self.args.hidden_sizes),
            dropout=self.args.dropout,
        ).to(self.device)

        class_weights = None
        if self.args.class_weighting:
            class_weights = torch.tensor(class_weights_np, dtype=torch.float32, device=self.device)

        if self.args.loss_type == "focal":
            criterion = FocalLoss(gamma=self.args.focal_gamma, weight=class_weights)
        else:
            criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=self.args.label_smoothing)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

        if self.args.balanced_batches:
            train_loader = DataLoader(
                train_ds,
                batch_sampler=ClassBalancedBatchSampler(train_ds.tensors[1].numpy(), batch_size=self.args.batch_size, seed=self.args.seed),
            )
        else:
            sample_weights = class_weights_np[train_ds.tensors[1].numpy()]
            train_loader = DataLoader(
                train_ds,
                batch_size=self.args.batch_size,
                shuffle=True,
                sampler=torch.utils.data.WeightedRandomSampler(
                    weights=torch.from_numpy(sample_weights).double(),
                    num_samples=len(sample_weights),
                    replacement=True,
                ),
            )

        val_loader = DataLoader(val_ds, batch_size=self.args.batch_size, shuffle=False)
        best_metric = float("-inf")
        best_state = None
        epochs_without_improvement = 0
        history: List[dict] = []

        for epoch in range(1, self.args.epochs + 1):
            model.train()
            running_loss = 0.0
            running_correct = 0
            running_total = 0
            for xb, yb in tqdm(train_loader, desc=f"{stage_name} {epoch}/{self.args.epochs}", leave=False):
                xb = xb.to(self.device)
                yb = yb.to(self.device)
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
            val_metrics = _evaluate(model, val_loader, criterion, self.device)
            scheduler.step(val_metrics.macro_f1)

            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "val_loss": val_metrics.loss,
                    "val_accuracy": val_metrics.accuracy,
                    "val_macro_f1": val_metrics.macro_f1,
                }
            )
            print(
                f"{stage_name} epoch={epoch:03d}/{self.args.epochs} train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_metrics.loss:.4f} val_acc={val_metrics.accuracy:.4f} val_macro_f1={val_metrics.macro_f1:.4f}",
                flush=True,
            )

            improved = val_metrics.macro_f1 > (best_metric + self.args.min_delta)
            if improved:
                best_metric = val_metrics.macro_f1
                epochs_without_improvement = 0
                best_state = {
                    "model": model.state_dict(),
                    "history": history,
                }
            else:
                epochs_without_improvement += 1

            if self.args.early_stopping_patience > 0 and epochs_without_improvement >= self.args.early_stopping_patience:
                print(f"{stage_name} early stopping at epoch {epoch}", flush=True)
                break

        if best_state is not None:
            model.load_state_dict(best_state["model"])
        return model, history, best_metric, len(history)

    def _collect_probs(self, model: nn.Module, loader_ds: TensorDataset) -> Tuple[np.ndarray, np.ndarray]:
        loader = DataLoader(loader_ds, batch_size=self.args.batch_size, shuffle=False)
        probs: List[np.ndarray] = []
        targets: List[np.ndarray] = []
        model.eval()
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(self.device)
                logits = model(xb)
                probs.append(torch.softmax(logits, dim=1).cpu().numpy())
                targets.append(yb.numpy())
        if not probs:
            return np.zeros((0, 0), dtype=np.float32), np.asarray([], dtype=np.int64)
        return np.concatenate(probs), np.concatenate(targets)

    def _pipeline_metrics_from_probs(
        self,
        probs1_arr: np.ndarray,
        probs2_arr: np.ndarray,
        y2_true: np.ndarray,
        pipeline_true: np.ndarray,
        selected_known: Sequence[int],
        unknown_idx: int,
        threshold: float,
        stage2_class_thresholds: Optional[np.ndarray] = None,
    ) -> Dict[str, object]:
        pipeline_pred = np.full_like(pipeline_true, fill_value=unknown_idx)
        true_known_positions = np.flatnonzero(pipeline_true != unknown_idx)
        if len(true_known_positions) and len(probs2_arr):
            if len(true_known_positions) != len(probs2_arr):
                raise ValueError(
                    f"stage2 prediction count mismatch: got {len(probs2_arr)} predictions for {len(true_known_positions)} known samples"
                )
            if len(y2_true) != len(probs2_arr):
                raise ValueError(
                    f"stage2 target count mismatch: got {len(y2_true)} targets for {len(probs2_arr)} predictions"
                )
            stage2_pred = np.argmax(probs2_arr, axis=1)
            stage2_score = probs2_arr[np.arange(len(stage2_pred)), stage2_pred]
            if stage2_class_thresholds is not None and len(stage2_class_thresholds) == probs2_arr.shape[1]:
                cls_thr = stage2_class_thresholds[stage2_pred]
                reject_mask = stage2_score < cls_thr
                stage2_pred = np.asarray(stage2_pred, dtype=np.int64)
                stage2_pred[reject_mask] = int(unknown_idx)
            pipeline_pred[true_known_positions] = stage2_pred

        if len(probs1_arr):
            stage1_known_mask = probs1_arr[:, 1] >= float(threshold)
            pipeline_pred[~stage1_known_mask] = unknown_idx

        labels = list(range(len(selected_known) + 1))
        report = classification_report(
            pipeline_true,
            pipeline_pred,
            labels=labels,
            target_names=[f"known_{k}" for k in selected_known] + ["unknown_agg"],
            zero_division=0,
            output_dict=True,
        )
        cm = confusion_matrix(pipeline_true, pipeline_pred, labels=labels)
        return {
            "accuracy": float(accuracy_score(pipeline_true, pipeline_pred)),
            "macro_f1": float(f1_score(pipeline_true, pipeline_pred, average="macro", zero_division=0)),
            "confusion_matrix": cm.tolist(),
            "report": report,
        }

    def _calibrate_unknown_threshold(
        self,
        probs1_arr: np.ndarray,
        probs2_arr: np.ndarray,
        y2_true: np.ndarray,
        pipeline_true: np.ndarray,
        selected_known: Sequence[int],
        unknown_idx: int,
        stage2_class_thresholds: Optional[np.ndarray] = None,
    ) -> float:
        if not len(probs1_arr):
            return float(self.args.unknown_threshold)

        search_min = max(0.05, float(self.args.unknown_threshold) - 0.35)
        search_max = min(0.95, float(self.args.unknown_threshold) + 0.35)
        steps = max(7, int(self.args.threshold_search_steps))
        candidates = np.unique(np.round(np.linspace(search_min, search_max, steps), 4))

        best_threshold = float(self.args.unknown_threshold)
        best_macro_f1 = float("-inf")
        best_accuracy = float("-inf")
        for threshold in candidates:
            metrics = self._pipeline_metrics_from_probs(
                probs1_arr,
                probs2_arr,
                y2_true,
                pipeline_true,
                selected_known,
                unknown_idx,
                float(threshold),
                stage2_class_thresholds=stage2_class_thresholds,
            )
            macro_f1 = float(metrics["macro_f1"])
            accuracy = float(metrics["accuracy"])
            if macro_f1 > best_macro_f1 + 1e-12 or (abs(macro_f1 - best_macro_f1) <= 1e-12 and accuracy > best_accuracy):
                best_macro_f1 = macro_f1
                best_accuracy = accuracy
                best_threshold = float(threshold)

        return best_threshold

    def _calibrate_stage2_class_thresholds(
        self,
        probs2_arr: np.ndarray,
        y2_true: np.ndarray,
        num_classes: int,
    ) -> np.ndarray:
        if len(probs2_arr) == 0 or len(y2_true) == 0:
            return np.zeros((num_classes,), dtype=np.float32)

        pred_cls = np.argmax(probs2_arr, axis=1)
        thr_min = float(self.args.class_threshold_min)
        thr_max = float(self.args.class_threshold_max)
        steps = max(5, int(self.args.class_threshold_steps))
        candidates = np.unique(np.round(np.linspace(thr_min, thr_max, steps), 4))

        out = np.zeros((num_classes,), dtype=np.float32)
        for cls in range(num_classes):
            best_thr = 0.0
            best_f1 = float("-inf")
            best_recall = float("-inf")
            score_cls = probs2_arr[:, cls]
            is_true = y2_true == cls
            for thr in candidates:
                pred_pos = (pred_cls == cls) & (score_cls >= float(thr))
                tp = float(np.sum(pred_pos & is_true))
                fp = float(np.sum(pred_pos & (~is_true)))
                fn = float(np.sum((~pred_pos) & is_true))
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                if f1 > best_f1 + 1e-12 or (abs(f1 - best_f1) <= 1e-12 and recall > best_recall):
                    best_f1 = f1
                    best_recall = recall
                    best_thr = float(thr)
            out[cls] = np.float32(best_thr)
        return out

    def fit(self) -> Dict[str, object]:
        root = Path(self.args.feature_root)
        subsample_test_root = Path(self.args.subsample_test_feature_root) if self.args.subsample_test_feature_root else root
        test_root = Path(self.args.test_feature_root) if self.args.test_feature_root else root

        train_chunks = _expand_chunks(root, "train")
        subsample_test_chunks = _expand_chunks(subsample_test_root, "test")
        test_chunks = _expand_chunks(test_root, "test")

        if self.args.validation_by_chunk:
            train_chunk_paths, val_chunk_paths = _split_chunk_paths(train_chunks, self.args.validation_fraction, self.args.seed)
            X_train, y_train = _load_split(train_chunk_paths, label_suffix=self.args.label_suffix)
            X_val, y_val = _load_split(val_chunk_paths, label_suffix=self.args.label_suffix)
        else:
            X_full, y_full = _load_split(train_chunks, label_suffix=self.args.label_suffix)
            try:
                X_train, X_val, y_train, y_val = train_test_split(
                    X_full,
                    y_full,
                    test_size=self.args.validation_fraction,
                    random_state=self.args.seed,
                    stratify=y_full,
                )
            except Exception:
                X_train, X_val, y_train, y_val = train_test_split(
                    X_full,
                    y_full,
                    test_size=self.args.validation_fraction,
                    random_state=self.args.seed,
                    stratify=None,
                )

        X_subsample_test, y_subsample_test = _load_split(subsample_test_chunks, label_suffix=self.args.label_suffix)
        X_test, y_test = _load_split(test_chunks, label_suffix=self.args.label_suffix)

        loaders = self._build_loaders(
            X_train,
            y_train,
            X_val,
            y_val,
            X_subsample_test,
            y_subsample_test,
            X_test,
            y_test,
            known_classes=self.args.known_classes,
        )

        self.stage1_model, stage1_history, stage1_best, _ = self._train_stage(
            loaders["train_stage1"],
            loaders["val_stage1"],
            num_classes=2,
            stage_name="stage1",
            classes_for_weights=[0, 1],
            class_weights_np=self._class_weights(loaders["train_stage1"].tensors[1].numpy(), 2),
        )

        stage2_train_ds = loaders["train_stage2"]
        stage2_val_ds = loaders["val_stage2"]
        self.stage2_model, stage2_history, stage2_best, _ = self._train_stage(
            stage2_train_ds,
            stage2_val_ds,
            num_classes=self.args.known_classes,
            stage_name="stage2",
            classes_for_weights=list(range(self.args.known_classes)),
            class_weights_np=self._class_weights(stage2_train_ds.tensors[1].numpy(), self.args.known_classes),
        )

        val_probs1, _ = self._collect_probs(self.stage1_model, loaders["val_stage1"])
        val_probs2, y2_val_true = self._collect_probs(self.stage2_model, loaders["val_stage2"])
        self.selected_stage2_class_thresholds = self._calibrate_stage2_class_thresholds(
            val_probs2,
            y2_val_true,
            self.args.known_classes,
        )
        self.selected_unknown_threshold = self._calibrate_unknown_threshold(
            val_probs1,
            val_probs2,
            y2_val_true,
            loaders["pipeline_val"],
            loaders["selected_known"],
            loaders["stage1_unknown_idx"],
            stage2_class_thresholds=self.selected_stage2_class_thresholds,
        )
        print(
            f"selected unknown threshold={self.selected_unknown_threshold:.4f} (initial={self.args.unknown_threshold:.4f})",
            flush=True,
        )
        print(
            f"selected stage2 class thresholds={self.selected_stage2_class_thresholds.round(4).tolist()}",
            flush=True,
        )

        subsample_metrics = self._evaluate_pipeline(
            self.stage1_model,
            self.stage2_model,
            loaders["subsample_test_stage1"],
            loaders["subsample_test_stage2"],
            loaders["pipeline_subsample_test"],
            loaders["y_subsample_test_stage2_mask"],
            loaders["selected_known"],
            loaders["stage1_unknown_idx"],
        )
        full_metrics = self._evaluate_pipeline(
            self.stage1_model,
            self.stage2_model,
            loaders["test_stage1"],
            loaders["test_stage2"],
            loaders["pipeline_test"],
            loaders["y_test_stage2_mask"],
            loaders["selected_known"],
            loaders["stage1_unknown_idx"],
        )

        save_path = Path(self.args.output_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "stage1_state_dict": self.stage1_model.state_dict(),
                "stage2_state_dict": self.stage2_model.state_dict(),
                "scaler_mean": self.scaler.mean_ if self.scaler is not None else None,
                "scaler_scale": self.scaler.scale_ if self.scaler is not None else None,
                "known_original_labels": self.known_original_labels,
                "unknown_original_label": self.unknown_original_label,
                "known_classes": self.args.known_classes,
                "selected_unknown_threshold": self.selected_unknown_threshold,
                "selected_stage2_class_thresholds": self.selected_stage2_class_thresholds,
                "history_stage1": stage1_history,
                "history_stage2": stage2_history,
                "selected_known": loaders["selected_known"],
            },
            save_path,
        )

        output = {
            "selected_known_original": loaders["selected_known"],
            "unknown_original_label": self.unknown_original_label,
            "selected_unknown_threshold": self.selected_unknown_threshold,
            "selected_stage2_class_thresholds": (
                self.selected_stage2_class_thresholds.tolist() if self.selected_stage2_class_thresholds is not None else []
            ),
            "stage1_best_val_macro_f1": stage1_best,
            "stage2_best_val_macro_f1": stage2_best,
            "subsample_test": subsample_metrics,
            "test": full_metrics,
            "model_path": str(save_path),
        }
        print(json.dumps(output, indent=2))
        return output

    def _class_weights(self, y: np.ndarray, num_classes: int) -> np.ndarray:
        counts = np.bincount(y.astype(np.int64), minlength=num_classes).astype(np.float32)
        inv = 1.0 / np.maximum(counts, 1.0)
        return inv / np.mean(inv)

    def _evaluate_pipeline(
        self,
        stage1_model: nn.Module,
        stage2_model: nn.Module,
        stage1_loader_ds: TensorDataset,
        stage2_loader_ds: TensorDataset,
        pipeline_true: np.ndarray,
        stage2_mask: np.ndarray,
        selected_known: Sequence[int],
        unknown_idx: int,
    ) -> Dict[str, object]:
        stage1_loader = DataLoader(stage1_loader_ds, batch_size=self.args.batch_size, shuffle=False)
        stage2_loader = DataLoader(stage2_loader_ds, batch_size=self.args.batch_size, shuffle=False)

        probs1_parts: List[np.ndarray] = []
        stage1_model.eval()
        with torch.no_grad():
            for xb, _ in stage1_loader:
                xb = xb.to(self.device)
                probs1_parts.append(torch.softmax(stage1_model(xb), dim=1).cpu().numpy())
        probs1_arr = np.concatenate(probs1_parts) if probs1_parts else np.zeros((0, 2), dtype=np.float32)

        probs2_parts: List[np.ndarray] = []
        y2_parts: List[np.ndarray] = []
        stage2_model.eval()
        with torch.no_grad():
            for xb, yb in stage2_loader:
                xb = xb.to(self.device)
                probs2_parts.append(torch.softmax(stage2_model(xb), dim=1).cpu().numpy())
                y2_parts.append(yb.numpy())
        probs2_arr = np.concatenate(probs2_parts) if probs2_parts else np.zeros((0, len(selected_known)), dtype=np.float32)
        y2_true = np.concatenate(y2_parts) if y2_parts else np.asarray([], dtype=np.int64)

        return self._pipeline_metrics_from_probs(
            probs1_arr,
            probs2_arr,
            y2_true,
            pipeline_true,
            selected_known,
            unknown_idx,
            self.selected_unknown_threshold,
            stage2_class_thresholds=self.selected_stage2_class_thresholds,
        )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Two-stage MLP training for phishing DOM features")
    p.add_argument("--feature-root", default=DEFAULT_FEATURE_ROOT)
    p.add_argument("--subsample-test-feature-root", default=DEFAULT_SUBSAMPLE_TEST_ROOT)
    p.add_argument("--test-feature-root", default=DEFAULT_TEST_ROOT)
    p.add_argument("--label-suffix", default=DEFAULT_LABEL_SUFFIX)
    p.add_argument("--known-classes", type=int, default=5)
    p.add_argument("--unknown-label", type=int, default=DEFAULT_UNKNOWN_LABEL)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--hidden-sizes", nargs="+", type=int, default=[256, 128])
    p.add_argument("--validation-fraction", type=float, default=0.1)
    p.add_argument("--validation-by-chunk", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--class-weighting", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--balanced-batches", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--synthetic-oversampling", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--targeted-synthetic-oversampling", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--synthetic-fill-fraction", type=float, default=0.10)
    p.add_argument("--synthetic-max-multiplier", type=float, default=1.05)
    p.add_argument("--synthetic-noise-std", type=float, default=0.01)
    p.add_argument("--boundary-focus-fraction", type=float, default=0.4)
    p.add_argument("--loss-type", choices=["cross_entropy", "focal"], default="focal")
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--unknown-threshold", type=float, default=0.60)
    p.add_argument("--threshold-search-steps", type=int, default=21)
    p.add_argument("--class-threshold-min", type=float, default=0.15)
    p.add_argument("--class-threshold-max", type=float, default=0.85)
    p.add_argument("--class-threshold-steps", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="")
    p.add_argument("--output-path", default="two_stage_mlp_model.pt")
    p.add_argument("--min-delta", type=float, default=1e-4)
    p.add_argument("--early-stopping-patience", type=int, default=8)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    runner = TwoStageMLP(args)
    runner.fit()
