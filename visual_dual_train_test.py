from __future__ import annotations

import argparse
import json
import pickle
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from visual_supervised_detector import DINOv2FeatureExtractor

UNKNOWN_BRAND = "unknown"
SUPPORTED_MODELS = {
    "random_forest",
    "extra_trees",
    "gradient_boosting",
    "logistic_regression",
    "mlp",
    "cnn1d",
}

# Keep gradient_boosting available explicitly, but skip it in --model-type all.
DEFAULT_ALL_MODELS = [
    "cnn1d",
    "extra_trees",
    "logistic_regression",
    "mlp",
    "random_forest",
]


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_torch_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but not available on this machine")
        return torch.device("cuda")
    return torch.device("cpu")


class MLPHead(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: Sequence[int], dropout: float, out_dim: int):
        super().__init__()
        layers: List[nn.Module] = []
        prev = int(input_dim)
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, int(h)))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = int(h)
        layers.append(nn.Linear(prev, int(out_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNN1DHead(nn.Module):
    def __init__(self, input_dim: int, dropout: float, out_dim: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(64),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 64, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, int(out_dim)),
        )
        self.input_dim = int(input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.features(x)
        return self.classifier(x)


def _parse_hidden_sizes(raw: str) -> List[int]:
    vals = [v.strip() for v in raw.split(",") if v.strip()]
    if not vals:
        raise ValueError("--mlp-hidden-sizes must contain at least one integer")
    return [int(v) for v in vals]


def _train_torch_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    num_classes: int,
    args: argparse.Namespace,
) -> Tuple[nn.Module, np.ndarray, np.ndarray | None]:
    device = _resolve_torch_device(args.device)
    input_dim = int(X_train.shape[1])
    binary = num_classes == 2
    out_dim = 1 if binary else int(num_classes)

    if model_name == "mlp":
        model = MLPHead(input_dim=input_dim, hidden_sizes=_parse_hidden_sizes(args.mlp_hidden_sizes), dropout=args.dropout, out_dim=out_dim)
    elif model_name == "cnn1d":
        model = CNN1DHead(input_dim=input_dim, dropout=args.dropout, out_dim=out_dim)
    else:
        raise ValueError(f"Unsupported torch model: {model_name}")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32 if binary else torch.long)
    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    if binary:
        neg = float(np.sum(y_train == 0))
        pos = float(np.sum(y_train == 1))
        pos_weight = torch.tensor([(neg / max(pos, 1.0))], dtype=torch.float32, device=device)
        criterion: nn.Module = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        counts = np.bincount(y_train.astype(np.int64), minlength=num_classes).astype(np.float32)
        counts[counts <= 0] = 1.0
        weights = counts.sum() / counts
        weights = weights / weights.mean()
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n_seen = 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            if binary:
                logits = logits.squeeze(1)
                loss = criterion(logits, yb)
            else:
                loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            batch_size = int(xb.shape[0])
            running += float(loss.item()) * batch_size
            n_seen += batch_size

        if epoch == 1 or epoch == args.epochs or epoch % max(1, args.epochs // 5) == 0:
            avg_loss = running / max(n_seen, 1)
            print(f"[train][{model_name}] epoch={epoch}/{args.epochs} loss={avg_loss:.6f}", flush=True)

    model.eval()
    infer_loader = DataLoader(
        TensorDataset(torch.tensor(X_eval, dtype=torch.float32)),
        batch_size=max(32, min(int(args.batch_size), 256)),
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    pred_parts: List[np.ndarray] = []
    prob_parts: List[np.ndarray] = []
    with torch.no_grad():
        for (xb,) in infer_loader:
            xb = xb.to(device, non_blocking=True)
            logits = model(xb)
            if binary:
                logits = logits.squeeze(1)
                probs_pos = torch.sigmoid(logits)
                preds = (probs_pos >= 0.5).long()
                pred_parts.append(preds.detach().cpu().numpy().astype(np.int64))
                prob_parts.append(probs_pos.detach().cpu().numpy().astype(np.float32))
            else:
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                pred_parts.append(preds.detach().cpu().numpy().astype(np.int64))
                prob_parts.append(probs.detach().cpu().numpy().astype(np.float32))

    preds_all = np.concatenate(pred_parts, axis=0)
    probs_all = np.concatenate(prob_parts, axis=0)
    return model, preds_all, probs_all


def _build_model(model_type: str, random_state: int, binary: bool):
    if model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=250,
            max_depth=24,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced",
        )
    if model_type == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=350,
            max_depth=24,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced",
        )
    if model_type == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=3,
            random_state=random_state,
        )
    if model_type == "logistic_regression":
        return LogisticRegression(
            random_state=random_state,
            max_iter=1500,
            class_weight="balanced",
            solver="lbfgs" if not binary else "liblinear",
            multi_class="auto",
        )
    raise ValueError(f"Unsupported model type: {model_type}")


def _read_manifest(manifest_path: Path, workspace_root: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        expected = ["new_path", "source_path", "is_phishing", "brand_primary", "brand_all"]
        if header != expected:
            raise ValueError(f"Unexpected manifest header: {header}")
        for line in f:
            parts = line.rstrip("\n").split(",", maxsplit=4)
            if len(parts) != 5:
                continue
            row = dict(zip(expected, parts))
            img_path = workspace_root / row["new_path"]
            if not img_path.exists():
                continue
            row["image_path"] = str(img_path)
            rows.append(row)
    if not rows:
        raise ValueError("No valid rows found in manifest with existing images.")
    return rows


def _top_k_brands(rows: Sequence[Dict[str, str]], top_k: int) -> List[str]:
    counts: Counter = Counter()
    for row in rows:
        if row["is_phishing"].strip() != "1":
            continue
        brand = row["brand_primary"].strip().lower()
        if brand and brand != UNKNOWN_BRAND:
            counts[brand] += 1
    if not counts:
        raise ValueError("Could not compute top brands from phishing rows.")
    return [name for name, _ in counts.most_common(top_k)]


def _encode_labels(rows: Sequence[Dict[str, str]], top_brands: Sequence[str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    brand_to_id = {b: i for i, b in enumerate(top_brands)}
    class_names = list(top_brands) + [UNKNOWN_BRAND]
    unknown_id = len(class_names) - 1

    y_brand = np.zeros((len(rows),), dtype=np.int16)

    for i, row in enumerate(rows):
        is_phish = 1 if row["is_phishing"].strip() == "1" else 0
        if is_phish:
            brand = row["brand_primary"].strip().lower()
            y_brand[i] = int(brand_to_id.get(brand, unknown_id))
        else:
            y_brand[i] = unknown_id

    # Keep binary labels strictly consistent with brand gating:
    # known-brand classes -> phishing(1), unknown -> benign(0).
    y_binary = (y_brand != unknown_id).astype(np.int16)

    return y_binary, y_brand, class_names


def _safe_stratified_split(
    y_binary: np.ndarray,
    y_brand: np.ndarray,
    test_size: float,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    idx = np.arange(len(y_binary), dtype=np.int64)
    strat_keys = np.array([f"{int(b)}__{int(c)}" for b, c in zip(y_binary.tolist(), y_brand.tolist())], dtype=object)

    key_counts = Counter(strat_keys.tolist())
    eligible_mask = np.array([key_counts[k] >= 2 for k in strat_keys], dtype=bool)

    idx_eligible = idx[eligible_mask]
    idx_rare = idx[~eligible_mask]

    if len(idx_eligible) == 0:
        raise ValueError("No eligible samples for stratified split (all classes are rare).")

    train_idx, test_idx = train_test_split(
        idx_eligible,
        test_size=test_size,
        random_state=random_state,
        stratify=strat_keys[eligible_mask],
    )

    if len(idx_rare) > 0:
        train_idx = np.concatenate([train_idx, idx_rare])

    return np.sort(train_idx), np.sort(test_idx)


def _extract_features(extractor: DINOv2FeatureExtractor, image_paths: Sequence[str]) -> np.ndarray:
    feats: List[np.ndarray] = []
    total = len(image_paths)
    for i, path in enumerate(image_paths, start=1):
        if i % 200 == 0 or i == total:
            print(f"[features] extracted {i}/{total}", flush=True)
        feats.append(extractor.extract(path))
    return np.asarray(feats, dtype=np.float32)


def train_and_eval(args: argparse.Namespace) -> Dict[str, object]:
    _set_global_seed(args.random_state)
    workspace_root = Path(args.workspace_root).resolve()
    manifest_path = Path(args.manifest_path).resolve()

    rows = _read_manifest(manifest_path, workspace_root)
    top_brands = _top_k_brands(rows, top_k=args.top_k)
    y_binary, y_brand, class_names = _encode_labels(rows, top_brands)

    train_idx, test_idx = _safe_stratified_split(
        y_binary=y_binary,
        y_brand=y_brand,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    train_paths = [rows[i]["image_path"] for i in train_idx.tolist()]
    test_paths = [rows[i]["image_path"] for i in test_idx.tolist()]

    yb_train = y_binary[train_idx]
    yb_test = y_binary[test_idx]
    yc_train = y_brand[train_idx]
    yc_test = y_brand[test_idx]

    print(
        f"[split] total={len(rows)} train={len(train_idx)} test={len(test_idx)} "
        f"binary_train={dict(Counter(yb_train.tolist()))} binary_test={dict(Counter(yb_test.tolist()))}",
        flush=True,
    )

    extractor = DINOv2FeatureExtractor(model_name=args.model_name, device=args.device)
    X_train = _extract_features(extractor, train_paths)
    X_test = _extract_features(extractor, test_paths)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    eval_models = DEFAULT_ALL_MODELS if args.model_type == "all" else [args.model_type]
    all_results: Dict[str, Dict[str, object]] = {}
    saved_models: Dict[str, Dict[str, object]] = {}
    unknown_id = len(class_names) - 1

    for model_name in eval_models:
        print(f"[train] Training model={model_name}", flush=True)
        if model_name in {"mlp", "cnn1d"}:
            model_brand, yc_pred, yc_prob = _train_torch_model(
                model_name=model_name,
                X_train=X_train_s,
                y_train=yc_train,
                X_eval=X_test_s,
                num_classes=len(class_names),
                args=args,
            )
            model_binary = None
        else:
            model_brand = _build_model(model_name, args.random_state, binary=False)
            model_brand.fit(X_train_s, yc_train)
            yc_pred = model_brand.predict(X_test_s)
            try:
                yc_prob = model_brand.predict_proba(X_test_s)
            except Exception:
                yc_prob = None

        # Requested behavior: known brand => phishing, unknown => benign.
        yb_pred = (yc_pred != unknown_id).astype(np.int64)
        yb_prob = None
        if yc_prob is not None:
            yb_prob = 1.0 - np.asarray(yc_prob, dtype=np.float32)[:, unknown_id]

        binary_metrics: Dict[str, object] = {
            "accuracy": float(accuracy_score(yb_test, yb_pred)),
            "f1": float(f1_score(yb_test, yb_pred, average="binary", zero_division=0)),
            "confusion_matrix": confusion_matrix(yb_test, yb_pred, labels=[0, 1]).tolist(),
            "report": classification_report(
                yb_test,
                yb_pred,
                labels=[0, 1],
                target_names=["benign", "phishing"],
                output_dict=True,
                zero_division=0,
            ),
        }

        try:
            binary_metrics["roc_auc"] = float(roc_auc_score(yb_test, yb_prob)) if yb_prob is not None else None
        except Exception:
            binary_metrics["roc_auc"] = None
        binary_metrics["rule"] = "phishing if predicted_brand != unknown else benign"

        brand_metrics: Dict[str, object] = {
            "accuracy": float(accuracy_score(yc_test, yc_pred)),
            "macro_f1": float(f1_score(yc_test, yc_pred, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(yc_test, yc_pred, average="weighted", zero_division=0)),
            "class_names": class_names,
            "confusion_matrix": confusion_matrix(
                yc_test,
                yc_pred,
                labels=list(range(len(class_names))),
            ).tolist(),
            "report": classification_report(
                yc_test,
                yc_pred,
                labels=list(range(len(class_names))),
                target_names=class_names,
                output_dict=True,
                zero_division=0,
            ),
        }

        all_results[model_name] = {
            "binary": binary_metrics,
            "brand": brand_metrics,
        }

        if model_name in {"mlp", "cnn1d"}:
            model_brand = model_brand.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        saved_models[model_name] = {
            "binary_model": model_binary,
            "brand_model": model_brand,
        }

    out: Dict[str, object] = {
        "manifest_path": str(manifest_path),
        "model_type": args.model_type,
        "model_name": args.model_name,
        "device": args.device,
        "binary_rule": "phishing if predicted_brand != unknown else benign",
        "top_k": int(args.top_k),
        "top_brands": top_brands,
        "class_names": class_names,
        "split": {
            "total": int(len(rows)),
            "train": int(len(train_idx)),
            "test": int(len(test_idx)),
            "train_binary_hist": {str(k): int(v) for k, v in sorted(Counter(yb_train.tolist()).items())},
            "test_binary_hist": {str(k): int(v) for k, v in sorted(Counter(yb_test.tolist()).items())},
            "train_brand_hist": {class_names[k]: int(v) for k, v in sorted(Counter(yc_train.tolist()).items())},
            "test_brand_hist": {class_names[k]: int(v) for k, v in sorted(Counter(yc_test.tolist()).items())},
        },
        "results": all_results,
    }

    payload = {
        "scaler": scaler,
        "models": saved_models,
        "top_brands": top_brands,
        "class_names": class_names,
        "model_type": args.model_type,
        "model_name": args.model_name,
        "device": args.device,
        "binary_rule": "phishing if predicted_brand != unknown else benign",
    }
    model_out = Path(args.output_model_path)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    with model_out.open("wb") as f:
        pickle.dump(payload, f)

    if args.report_json:
        report_path = Path(args.report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train/test visual dual-task models (binary + brand) on consolidated screenshots")
    p.add_argument("--workspace-root", default=".")
    p.add_argument("--manifest-path", default="./visual_consolidated_screenshots/manifest.csv")
    p.add_argument("--top-k", type=int, default=10, help="Top-k phishing brands to model; benign is always mapped to unknown")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--model-type", choices=sorted(SUPPORTED_MODELS) + ["all"], default="extra_trees")
    p.add_argument("--model-name", default="facebook/dinov2-base")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--output-model-path", default="./visual_dual_detector.pkl")
    p.add_argument("--report-json", default="./visual_dual_report.json")
    p.add_argument("--epochs", type=int, default=25, help="Training epochs for torch models (mlp/cnn1d)")
    p.add_argument("--batch-size", type=int, default=128, help="Batch size for torch models (mlp/cnn1d)")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate for torch models (mlp/cnn1d)")
    p.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for torch models (mlp/cnn1d)")
    p.add_argument("--dropout", type=float, default=0.2, help="Dropout for torch models (mlp/cnn1d)")
    p.add_argument("--mlp-hidden-sizes", default="1024,512,256", help="Comma-separated hidden sizes for MLP")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if not 0.05 <= args.test_size <= 0.5:
        raise ValueError("--test-size must be between 0.05 and 0.5")
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    result = train_and_eval(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
