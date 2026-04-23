"""
Supervised visual phishing detector using local screenshot dataset.
Trains DINOv2 embeddings on phishing vs benign screenshots.
Supports comparison against target company reference.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from transformers import AutoImageProcessor, AutoModel


class DINOv2FeatureExtractor:
    """Extract DINOv2 embeddings from images."""

    def __init__(self, model_name: str = "facebook/dinov2-base", device: str = "auto"):
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be one of: auto, cpu, cuda")

        if device == "auto":
            resolved = "cuda" if torch.cuda.is_available() else "cpu"
        elif device == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("CUDA requested but not available on this machine")
            resolved = "cuda"
        else:
            resolved = "cpu"

        self.device = torch.device(resolved)
        print(f"[DINOv2] Using device: {self.device}")

        self.processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def extract(self, image_path: str) -> np.ndarray:
        """Extract CLS token and patch-wise stats as feature vector."""
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)

            outputs = self.model(**inputs, output_hidden_states=True)

            # CLS token: global semantic representation (1, 768)
            cls_token = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()

            # Patch tokens: spatial features excluding CLS
            patch_tokens = outputs.last_hidden_state[:, 1:, :].cpu()  # (1, num_patches, 768)

            # Aggregate patch statistics
            patch_mean = patch_tokens.mean(dim=1).numpy().flatten()  # (768,)
            patch_std = patch_tokens.std(dim=1).numpy().flatten()  # (768,)
            patch_max = patch_tokens.max(dim=1).values.numpy().flatten()  # (768,)

            # Combine into single feature vector
            features = np.concatenate([cls_token, patch_mean, patch_std, patch_max], axis=0)
            return features.astype(np.float32)

        except Exception as e:
            print(f"[ERROR] Failed to extract features from {image_path}: {e}")
            return np.zeros(768 * 4, dtype=np.float32)


def discover_images(root_dir: Path, label: int) -> List[Tuple[str, int]]:
    """Find all clean_js_on.jpg files in directory tree."""
    samples = []
    for img_path in root_dir.rglob("clean_js_on.jpg"):
        samples.append((str(img_path), label))
    return samples


def discover_images_multi(root_dirs: List[Path], label: int) -> List[Tuple[str, int]]:
    """Collect clean_js_on.jpg files from multiple root directories."""
    all_samples: List[Tuple[str, int]] = []
    for root in root_dirs:
        if not root.exists():
            print(f"[warn] Directory not found, skipping: {root}")
            continue
        samples = discover_images(root, label)
        print(f"[data] {root}: {len(samples)} images")
        all_samples.extend(samples)
    return all_samples


class VisualSupervisedDetector:
    """Supervised phishing detector trained on local screenshot dataset."""

    def __init__(self, model_type: str = "random_forest"):
        self.model_type = model_type  # "random_forest" or "logistic_regression"
        self.feature_extractor = DINOv2FeatureExtractor()
        self.scaler = StandardScaler()
        self.model = self._build_model()
        self.is_fitted = False

    def _build_model(self):
        if self.model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced",
            )
        elif self.model_type == "logistic_regression":
            return LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight="balanced",
                solver="liblinear",
            )
        elif self.model_type == "extra_trees":
            return ExtraTreesClassifier(
                n_estimators=300,
                max_depth=24,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced",
            )
        elif self.model_type == "gradient_boosting":
            return GradientBoostingClassifier(
                n_estimators=250,
                learning_rate=0.05,
                max_depth=3,
                random_state=42,
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _extract_dataset_features(
        self,
        phishing_dirs: List[Path],
        benign_dirs: List[Path],
    ) -> Tuple[np.ndarray, np.ndarray]:
        print("[train] Discovering images...")
        phishing_samples = discover_images_multi(phishing_dirs, label=1)
        benign_samples = discover_images_multi(benign_dirs, label=0)

        print(f"[train] Found: phishing={len(phishing_samples)}, benign={len(benign_samples)}")
        all_samples = phishing_samples + benign_samples
        np.random.shuffle(all_samples)

        print("[train] Extracting features for full dataset...")
        X_all = []
        y_all = []
        for img_path, label in all_samples:
            feat = self.feature_extractor.extract(img_path)
            X_all.append(feat)
            y_all.append(label)

        X = np.array(X_all, dtype=np.float32)
        y = np.array(y_all, dtype=np.int8)
        return X, y

    @staticmethod
    def _detailed_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, object]:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=[0, 1],
            zero_division=0,
        )

        metrics: Dict[str, object] = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "roc_auc": float(roc_auc_score(y_true, y_pred_proba)) if len(np.unique(y_true)) > 1 else 0.0,
            "confusion_matrix": {
                "tn": int(cm[0, 0]),
                "fp": int(cm[0, 1]),
                "fn": int(cm[1, 0]),
                "tp": int(cm[1, 1]),
            },
            "per_class": {
                "benign": {
                    "precision": float(precision[0]),
                    "recall": float(recall[0]),
                    "f1": float(f1[0]),
                    "support": int(support[0]),
                },
                "phishing": {
                    "precision": float(precision[1]),
                    "recall": float(recall[1]),
                    "f1": float(f1[1]),
                    "support": int(support[1]),
                },
            },
            "classification_report": classification_report(
                y_true,
                y_pred,
                target_names=["Benign", "Phishing"],
                output_dict=True,
                zero_division=0,
            ),
        }
        return metrics

    def fit_from_directory(
        self,
        phishing_dirs: List[Path],
        benign_dirs: List[Path],
        test_fraction: float = 0.2,
        random_state: int = 42,
    ) -> Dict[str, object]:
        """Train on phishing and benign screenshots."""
        X_all, y_all = self._extract_dataset_features(phishing_dirs, benign_dirs)
        X_train, X_test, y_train, y_test = train_test_split(
            X_all,
            y_all,
            test_size=test_fraction,
            random_state=random_state,
            stratify=y_all,
        )

        print(f"[train] train={len(X_train)}, test={len(X_test)}")

        print(f"[train] Fitting scaler...")
        self.scaler.fit(X_train)

        print(f"[train] Scaling and fitting model...")
        X_train_scaled = self.scaler.transform(X_train)
        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True

        print(f"[train] Evaluating on test set...")
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]

        metrics = self._detailed_metrics(y_test, y_pred, y_pred_proba)
        metrics["train_samples"] = int(len(X_train))
        metrics["test_samples"] = int(len(X_test))
        metrics["model_type"] = self.model_type

        print(f"[train] Metrics:")
        print(f"  accuracy: {metrics['accuracy']}")
        print(f"  roc_auc: {metrics['roc_auc']}")
        print(f"  confusion_matrix: {metrics['confusion_matrix']}")
        print(f"\n[train] Classification Report:")
        print(classification_report(y_test, y_pred, target_names=["Benign", "Phishing"]))

        return metrics

    def benchmark_from_directory(
        self,
        phishing_dirs: List[Path],
        benign_dirs: List[Path],
        model_types: List[str],
        test_fraction: float = 0.2,
        random_state: int = 42,
    ) -> List[Dict[str, object]]:
        """Train and compare multiple model types on the exact same train/test split."""
        X_all, y_all = self._extract_dataset_features(phishing_dirs, benign_dirs)
        X_train, X_test, y_train, y_test = train_test_split(
            X_all,
            y_all,
            test_size=test_fraction,
            random_state=random_state,
            stratify=y_all,
        )
        print(f"[benchmark] Shared split train={len(X_train)} test={len(X_test)}")

        results: List[Dict[str, object]] = []
        for mt in model_types:
            print(f"\n[benchmark] model={mt}")
            self.model_type = mt
            self.model = self._build_model()
            self.scaler = StandardScaler()

            self.scaler.fit(X_train)
            X_train_scaled = self.scaler.transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            self.model.fit(X_train_scaled, y_train)

            y_pred = self.model.predict(X_test_scaled)
            y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
            metrics = self._detailed_metrics(y_test, y_pred, y_pred_proba)
            metrics["model_type"] = mt
            metrics["train_samples"] = int(len(X_train))
            metrics["test_samples"] = int(len(X_test))
            results.append(metrics)

            print(f"  accuracy: {metrics['accuracy']}")
            print(f"  roc_auc: {metrics['roc_auc']}")
            print(f"  confusion_matrix: {metrics['confusion_matrix']}")
            print(classification_report(y_test, y_pred, target_names=["Benign", "Phishing"]))

        self.is_fitted = True
        best = max(results, key=lambda r: float(r["roc_auc"]))
        self.model_type = str(best["model_type"])
        self.model = self._build_model()
        self.scaler = StandardScaler()
        self.scaler.fit(X_train)
        self.model.fit(self.scaler.transform(X_train), y_train)

        return results

    def predict_image(self, image_path: str) -> Dict[str, float]:
        """Predict if a single image is phishing or benign."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction.")

        feat = self.feature_extractor.extract(image_path)
        X = np.array([feat], dtype=np.float32)
        X_scaled = self.scaler.transform(X)

        pred_label = int(self.model.predict(X_scaled)[0])
        pred_proba = float(self.model.predict_proba(X_scaled)[0, 1])

        return {
            "label": "Phishing" if pred_label == 1 else "Benign",
            "phishing_probability": pred_proba,
            "benign_probability": 1.0 - pred_proba,
        }

    def compare_to_reference(self, suspect_image: str, reference_image: str) -> Dict[str, float]:
        """Compare a suspect image against a reference (target company) screenshot."""
        suspect_feat = self.feature_extractor.extract(suspect_image)
        reference_feat = self.feature_extractor.extract(reference_image)

        # Cosine similarity
        similarity = float(
            np.dot(suspect_feat, reference_feat) / (np.linalg.norm(suspect_feat) * np.linalg.norm(reference_feat) + 1e-8)
        )

        # Get anomaly scores from trained model
        X_suspect = np.array([suspect_feat], dtype=np.float32)
        X_suspect_scaled = self.scaler.transform(X_suspect)
        suspect_proba = float(self.model.predict_proba(X_suspect_scaled)[0, 1])

        return {
            "visual_similarity": (similarity + 1) / 2,  # Normalize to [0, 1]
            "suspect_phishing_score": suspect_proba,
            "verdict": "possible_phishing" if similarity > 0.7 and suspect_proba > 0.5 else "likely_benign",
        }

    def save(self, model_path: str) -> None:
        """Save fitted model and scaler."""
        payload = {
            "model_type": self.model_type,
            "scaler": self.scaler,
            "model": self.model,
            "is_fitted": self.is_fitted,
        }
        with Path(model_path).open("wb") as f:
            pickle.dump(payload, f)
        print(f"[save] Model saved to {model_path}")

    def load(self, model_path: str) -> None:
        """Load fitted model and scaler."""
        with Path(model_path).open("rb") as f:
            payload = pickle.load(f)

        self.model_type = str(payload["model_type"])
        self.scaler = payload["scaler"]
        self.model = payload["model"]
        self.is_fitted = bool(payload["is_fitted"])
        print(f"[load] Model loaded from {model_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Supervised visual phishing detector")
    parser.add_argument("--mode", default="train", choices=["train", "benchmark", "predict", "compare"])
    parser.add_argument("--phishing-dir", help="Path to phishing screenshots directory")
    parser.add_argument("--benign-dir", help="Path to benign screenshots directory")
    parser.add_argument("--phishing-dirs", nargs="+", help="Paths to multiple phishing screenshot directories")
    parser.add_argument("--benign-dirs", nargs="+", help="Paths to multiple benign screenshot directories")
    parser.add_argument("--model-path", default="./visual_sup_detector.pkl", help="Path to save/load model")
    parser.add_argument(
        "--model-type",
        default="random_forest",
        choices=["random_forest", "logistic_regression", "extra_trees", "gradient_boosting"],
    )
    parser.add_argument(
        "--model-types",
        nargs="+",
        choices=["random_forest", "logistic_regression", "extra_trees", "gradient_boosting"],
        help="Used in benchmark mode; compare multiple models",
    )
    parser.add_argument("--image", help="Path to image for prediction")
    parser.add_argument("--reference", help="Path to reference image for comparison")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--report-json", help="Optional path to save detailed metrics JSON")
    args = parser.parse_args()

    if args.mode == "train":
        phishing_dirs = [Path(p) for p in args.phishing_dirs] if args.phishing_dirs else ([] if not args.phishing_dir else [Path(args.phishing_dir)])
        benign_dirs = [Path(p) for p in args.benign_dirs] if args.benign_dirs else ([] if not args.benign_dir else [Path(args.benign_dir)])

        if not phishing_dirs or not benign_dirs:
            raise ValueError("Provide --phishing-dirs/--benign-dirs or --phishing-dir/--benign-dir for training")

        detector = VisualSupervisedDetector(model_type=args.model_type)
        metrics = detector.fit_from_directory(
            phishing_dirs,
            benign_dirs,
            test_fraction=args.test_fraction,
            random_state=args.random_state,
        )
        detector.save(args.model_path)
        if args.report_json:
            Path(args.report_json).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(json.dumps(metrics, indent=2))

    elif args.mode == "benchmark":
        phishing_dirs = [Path(p) for p in args.phishing_dirs] if args.phishing_dirs else ([] if not args.phishing_dir else [Path(args.phishing_dir)])
        benign_dirs = [Path(p) for p in args.benign_dirs] if args.benign_dirs else ([] if not args.benign_dir else [Path(args.benign_dir)])

        if not phishing_dirs or not benign_dirs:
            raise ValueError("Provide --phishing-dirs/--benign-dirs or --phishing-dir/--benign-dir for benchmark")

        detector = VisualSupervisedDetector(model_type=args.model_type)
        model_types = args.model_types if args.model_types else ["random_forest", "extra_trees", "logistic_regression", "gradient_boosting"]
        reports = detector.benchmark_from_directory(
            phishing_dirs,
            benign_dirs,
            model_types=model_types,
            test_fraction=args.test_fraction,
            random_state=args.random_state,
        )
        detector.save(args.model_path)
        if args.report_json:
            Path(args.report_json).write_text(json.dumps(reports, indent=2), encoding="utf-8")
        print(json.dumps(reports, indent=2))

    elif args.mode == "predict":
        if not args.image:
            raise ValueError("--image required for prediction")

        detector = VisualSupervisedDetector()
        detector.load(args.model_path)
        result = detector.predict_image(args.image)
        print(json.dumps(result, indent=2))

    elif args.mode == "compare":
        if not args.image or not args.reference:
            raise ValueError("--image and --reference required for comparison")

        detector = VisualSupervisedDetector()
        detector.load(args.model_path)
        result = detector.compare_to_reference(args.image, args.reference)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
