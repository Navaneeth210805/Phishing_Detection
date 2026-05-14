#!/usr/bin/env python3
"""Generate training visualizations for visual and DOM model pipelines.

Outputs PNG files under ./training_visualizations/{visual,dom}.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT_ROOT = ROOT / "training_visualizations"
VIS_OUT = OUT_ROOT / "visual"
DOM_OUT = OUT_ROOT / "dom"


@dataclass
class EpochPoint:
    epoch: int
    total_epochs: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    val_macro_f1: float


def _ensure_dirs() -> None:
    VIS_OUT.mkdir(parents=True, exist_ok=True)
    DOM_OUT.mkdir(parents=True, exist_ok=True)


def _parse_visual_epoch_log(log_path: Path) -> List[EpochPoint]:
    if not log_path.exists():
        return []

    pattern = re.compile(
        r"epoch=(\d+)/(\d+)\s+"
        r"train_loss=([-+]?\d*\.?\d+)\s+"
        r"train_acc=([-+]?\d*\.?\d+)\s+"
        r"val_loss=([-+]?\d*\.?\d+)\s+"
        r"val_acc=([-+]?\d*\.?\d+)\s+"
        r"val_macro_f1=([-+]?\d*\.?\d+)"
    )

    points: List[EpochPoint] = []
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = pattern.search(line)
        if not match:
            continue
        points.append(
            EpochPoint(
                epoch=int(match.group(1)),
                total_epochs=int(match.group(2)),
                train_loss=float(match.group(3)),
                train_acc=float(match.group(4)),
                val_loss=float(match.group(5)),
                val_acc=float(match.group(6)),
                val_macro_f1=float(match.group(7)),
            )
        )
    return points


def _plot_visual_log(points: List[EpochPoint], model_label: str, out_path: Path) -> bool:
    if not points:
        return False

    epochs = np.array([p.epoch for p in points])
    train_loss = np.array([p.train_loss for p in points])
    val_loss = np.array([p.val_loss for p in points])
    train_acc = np.array([p.train_acc for p in points])
    val_acc = np.array([p.val_acc for p in points])
    val_macro_f1 = np.array([p.val_macro_f1 for p in points])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, train_loss, label="Train Loss", linewidth=2)
    axes[0].plot(epochs, val_loss, label="Val Loss", linewidth=2)
    axes[0].set_title(f"{model_label}: Epoch-wise Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(epochs, train_acc, label="Train Acc", linewidth=2)
    axes[1].plot(epochs, val_acc, label="Val Acc", linewidth=2)
    axes[1].plot(epochs, val_macro_f1, label="Val Macro-F1", linewidth=2)
    axes[1].set_title(f"{model_label}: Epoch-wise Accuracy/F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return True


def _plot_visual_comparison(mlp: List[EpochPoint], cnn: List[EpochPoint], out_path: Path) -> bool:
    if not mlp and not cnn:
        return False

    fig, ax = plt.subplots(figsize=(9, 5))
    if mlp:
        ax.plot([p.epoch for p in mlp], [p.val_macro_f1 for p in mlp], label="MLP Val Macro-F1", linewidth=2)
    if cnn:
        ax.plot([p.epoch for p in cnn], [p.val_macro_f1 for p in cnn], label="CNN1D Val Macro-F1", linewidth=2)

    ax.set_title("Visual Neural Models: Validation Macro-F1 Across Epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Macro-F1")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return True


def _plot_visual_eval_overview(report_path: Path, out_path: Path) -> bool:
    if not report_path.exists():
        return False

    data = json.loads(report_path.read_text(encoding="utf-8"))
    results = data.get("results", {})
    if not results:
        return False

    models = sorted(results.keys())
    brand_acc = [float(results[m]["brand"]["accuracy"]) for m in models]
    brand_macro_f1 = [float(results[m]["brand"]["macro_f1"]) for m in models]
    binary_acc = [float(results[m]["binary"]["accuracy"]) for m in models]
    binary_f1 = [float(results[m]["binary"]["f1"]) for m in models]

    x = np.arange(len(models))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - 1.5 * width, brand_acc, width, label="Brand Acc")
    ax.bar(x - 0.5 * width, brand_macro_f1, width, label="Brand Macro-F1")
    ax.bar(x + 0.5 * width, binary_acc, width, label="Binary Acc")
    ax.bar(x + 1.5 * width, binary_f1, width, label="Binary F1")

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ") for m in models], rotation=20)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Visual Models: Evaluation Overview")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return True


def _parse_ts(ts: Optional[str]) -> datetime:
    if not ts:
        return datetime.min
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return datetime.min


def _load_dom_latest_entries(jsonl_path: Path) -> Dict[Tuple[str, str], Dict[str, object]]:
    if not jsonl_path.exists():
        return {}

    best: Dict[Tuple[str, str], Dict[str, object]] = {}
    for line in jsonl_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        algo = obj.get("algorithm")
        cmd = obj.get("command")
        if not algo or cmd not in {"mc-train", "mc-eval"}:
            continue

        key = (str(algo), str(cmd))
        ts_new = _parse_ts(obj.get("timestamp_utc"))
        ts_old = _parse_ts(best.get(key, {}).get("timestamp_utc") if key in best else None)
        if key not in best or ts_new >= ts_old:
            best[key] = obj
    return best


def _plot_dom_train_bars(entries: Dict[Tuple[str, str], Dict[str, object]], out_path: Path) -> bool:
    train_items = [(k[0], v) for k, v in entries.items() if k[1] == "mc-train"]
    if not train_items:
        return False

    train_items.sort(key=lambda x: x[0])
    models = [m for m, _ in train_items]
    train_acc = [float(v.get("train_accuracy", 0.0)) for _, v in train_items]
    val_acc = [float(v.get("val_accuracy", 0.0)) for _, v in train_items]
    train_macro = [float(v.get("train_macro_f1", 0.0)) for _, v in train_items]
    val_macro = [float(v.get("val_macro_f1", 0.0)) for _, v in train_items]

    x = np.arange(len(models))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - 1.5 * width, train_acc, width, label="Train Acc")
    ax.bar(x - 0.5 * width, val_acc, width, label="Val Acc")
    ax.bar(x + 0.5 * width, train_macro, width, label="Train Macro-F1")
    ax.bar(x + 1.5 * width, val_macro, width, label="Val Macro-F1")

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ") for m in models], rotation=20)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("DOM Models: Training/Validation Metrics")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return True


def _plot_dom_eval_bars(entries: Dict[Tuple[str, str], Dict[str, object]], out_path: Path) -> bool:
    eval_items = [(k[0], v) for k, v in entries.items() if k[1] == "mc-eval"]
    if not eval_items:
        return False

    eval_items.sort(key=lambda x: x[0])
    models = [m for m, _ in eval_items]
    acc = [float(v.get("accuracy", 0.0)) for _, v in eval_items]
    macro = [float(v.get("macro_f1", 0.0)) for _, v in eval_items]
    weighted = [float(v.get("weighted_f1", 0.0)) for _, v in eval_items]
    bin_acc = [float(v.get("binary", {}).get("accuracy", 0.0)) for _, v in eval_items]

    x = np.arange(len(models))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - 1.5 * width, acc, width, label="Brand Acc")
    ax.bar(x - 0.5 * width, macro, width, label="Brand Macro-F1")
    ax.bar(x + 0.5 * width, weighted, width, label="Brand Weighted-F1")
    ax.bar(x + 1.5 * width, bin_acc, width, label="Binary Acc")

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ") for m in models], rotation=20)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("DOM Models: Evaluation Metrics")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return True


def _plot_dom_curve_rounds(entries: Dict[Tuple[str, str], Dict[str, object]], out_path: Path) -> bool:
    train_items = [(k[0], v) for k, v in entries.items() if k[1] == "mc-train"]
    if not train_items:
        return False

    has_any = False
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for model, obj in sorted(train_items, key=lambda x: x[0]):
        curve = obj.get("curve", [])
        if not isinstance(curve, list) or not curve:
            continue

        rounds = [int(p.get("fit_round", i + 1)) for i, p in enumerate(curve)]
        train_vals = [float(p.get("train_accuracy", np.nan)) for p in curve]
        val_vals = [float(p.get("val_accuracy", np.nan)) for p in curve]
        train_f1 = [float(p.get("train_macro_f1", np.nan)) for p in curve]
        val_f1 = [float(p.get("val_macro_f1", np.nan)) for p in curve]

        axes[0].plot(rounds, train_vals, marker="o", label=f"{model} train")
        axes[0].plot(rounds, val_vals, marker="x", linestyle="--", label=f"{model} val")
        axes[1].plot(rounds, train_f1, marker="o", label=f"{model} train")
        axes[1].plot(rounds, val_f1, marker="x", linestyle="--", label=f"{model} val")
        has_any = True

    if not has_any:
        plt.close(fig)
        return False

    axes[0].set_title("DOM Curve: Accuracy by Fit Round")
    axes[0].set_xlabel("Fit Round")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    axes[1].set_title("DOM Curve: Macro-F1 by Fit Round")
    axes[1].set_xlabel("Fit Round")
    axes[1].set_ylabel("Macro-F1")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return True


def _write_summary(created: List[Path]) -> None:
    summary_path = OUT_ROOT / "README.md"
    lines = [
        "# Training Visualizations",
        "",
        "Generated charts for both visual and DOM model pipelines.",
        "",
        "## Generated Files",
    ]
    for p in sorted(created):
        rel = p.relative_to(OUT_ROOT)
        lines.append(f"- `{rel.as_posix()}`")

    lines.extend(
        [
            "",
            "## Notes",
            "- Epoch-wise plots are available for visual neural models (`mlp`, `cnn1d`) from run logs.",
            "- DOM training records currently include one fit round per model; round-curve plot uses available points.",
            "",
        ]
    )
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    _ensure_dirs()
    created: List[Path] = []

    mlp_points = _parse_visual_epoch_log(ROOT / "mlp_brand_stage1_run.log")
    cnn_points = _parse_visual_epoch_log(ROOT / "cnn1d_brand_stage1_run.log")

    if _plot_visual_log(mlp_points, "MLP", VIS_OUT / "mlp_epoch_training.png"):
        created.append(VIS_OUT / "mlp_epoch_training.png")
    if _plot_visual_log(cnn_points, "CNN1D", VIS_OUT / "cnn1d_epoch_training.png"):
        created.append(VIS_OUT / "cnn1d_epoch_training.png")
    if _plot_visual_comparison(mlp_points, cnn_points, VIS_OUT / "neural_val_macro_f1_comparison.png"):
        created.append(VIS_OUT / "neural_val_macro_f1_comparison.png")

    if _plot_visual_eval_overview(ROOT / "visual_dual_report_all_brand_gated.json", VIS_OUT / "visual_model_eval_overview.png"):
        created.append(VIS_OUT / "visual_model_eval_overview.png")

    dom_entries = _load_dom_latest_entries(ROOT / "dom_brand_stage1_results.jsonl")
    if _plot_dom_train_bars(dom_entries, DOM_OUT / "dom_train_val_metrics.png"):
        created.append(DOM_OUT / "dom_train_val_metrics.png")
    if _plot_dom_eval_bars(dom_entries, DOM_OUT / "dom_eval_metrics.png"):
        created.append(DOM_OUT / "dom_eval_metrics.png")
    if _plot_dom_curve_rounds(dom_entries, DOM_OUT / "dom_fit_round_curves.png"):
        created.append(DOM_OUT / "dom_fit_round_curves.png")

    _write_summary(created)
    created.append(OUT_ROOT / "README.md")

    print("Generated files:")
    for p in sorted(created):
        print(f" - {p}")


if __name__ == "__main__":
    main()
