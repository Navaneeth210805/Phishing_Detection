#!/usr/bin/env python3
"""Generate LightGBM built-in visualizations for the DOM LightGBM model.

Outputs are written to:
  training_visualizations/lightgbm_builtin/
"""

from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

try:
    import lightgbm as lgb
except Exception as exc:  # pragma: no cover
    raise RuntimeError("lightgbm is required to run this script") from exc


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "dom_mc_brand_stage1_lightgbm.pkl"
DOM_RESULTS_PATH = ROOT / "dom_brand_stage1_results.jsonl"
OUT_DIR = ROOT / "training_visualizations" / "lightgbm_builtin"


def _load_checkpoint_model(path: Path):
    obj = pickle.loads(path.read_bytes())
    if isinstance(obj, dict) and "model" in obj:
        return obj["model"]
    return obj


def _safe_name(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in s)


def _latest_dom_eval_and_train(jsonl_path: Path) -> Tuple[Optional[Dict[str, object]], Optional[Dict[str, object]]]:
    if not jsonl_path.exists():
        return None, None

    def parse_ts(v: Optional[str]) -> datetime:
        if not v:
            return datetime.min.replace(tzinfo=timezone.utc)
        try:
            return datetime.fromisoformat(v.replace("Z", "+00:00"))
        except Exception:
            return datetime.min.replace(tzinfo=timezone.utc)

    latest_train = None
    latest_eval = None
    ts_train = datetime.min.replace(tzinfo=timezone.utc)
    ts_eval = datetime.min.replace(tzinfo=timezone.utc)

    for line in jsonl_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue

        if row.get("algorithm") != "lightgbm":
            continue

        cmd = row.get("command")
        ts = parse_ts(row.get("timestamp_utc"))
        if cmd == "mc-train" and ts >= ts_train:
            latest_train = row
            ts_train = ts
        if cmd == "mc-eval" and ts >= ts_eval:
            latest_eval = row
            ts_eval = ts

    return latest_train, latest_eval


def _plot_feature_importance(booster, out_dir: Path, created: List[Path]) -> None:
    for imp_type in ("split", "gain"):
        fig, ax = plt.subplots(figsize=(10, 8))
        lgb.plot_importance(
            booster,
            importance_type=imp_type,
            max_num_features=30,
            height=0.4,
            ax=ax,
            title=f"LightGBM Feature Importance ({imp_type})",
        )
        fig.tight_layout()
        out = out_dir / f"feature_importance_{imp_type}.png"
        fig.savefig(out, dpi=180)
        plt.close(fig)
        created.append(out)


def _plot_tree_views(booster, out_dir: Path, created: List[Path]) -> None:
    plotted = False
    try:
        fig, ax = plt.subplots(figsize=(20, 10))
        lgb.plot_tree(
            booster,
            tree_index=0,
            figsize=(20, 10),
            show_info=["split_gain", "internal_count", "leaf_count", "data_percentage"],
            ax=ax,
        )
        fig.tight_layout()
        out = out_dir / "tree_structure_tree0_matplotlib.png"
        fig.savefig(out, dpi=180)
        plt.close(fig)
        created.append(out)
        plotted = True
    except Exception:
        pass

    # Optional high-quality Graphviz rendering.
    try:
        graph = lgb.create_tree_digraph(
            booster,
            tree_index=0,
            show_info=["split_gain", "internal_count", "leaf_count", "data_percentage"],
        )
        out_base = out_dir / "tree_structure_tree0_graphviz"
        # Render both source and png when graphviz binary is available.
        graph.save(str(out_base.with_suffix(".dot")))
        created.append(out_base.with_suffix(".dot"))
        rendered = graph.render(filename=str(out_base), format="png", cleanup=True)
        created.append(Path(rendered))
    except Exception:
        # Graphviz is optional; do not fail the whole script.
        if not plotted:
            # Fallback: export tree-0 structure to JSON so structure is still inspectable.
            model_dump = booster.dump_model()
            tree0 = {}
            if isinstance(model_dump, dict):
                tree_info = model_dump.get("tree_info", [])
                if isinstance(tree_info, list) and tree_info:
                    tree0 = tree_info[0]
            out_json = out_dir / "tree_structure_tree0.json"
            out_json.write_text(json.dumps(tree0, indent=2), encoding="utf-8")
            created.append(out_json)


def _plot_metric_if_available(model, out_dir: Path, created: List[Path]) -> bool:
    # LightGBM sklearn API stores eval history in evals_result_ when eval_set was provided.
    has_evals = hasattr(model, "evals_result_") and isinstance(getattr(model, "evals_result_", None), dict)
    if not has_evals or not model.evals_result_:
        return False

    evals_result = model.evals_result_

    # Try common metrics; if absent use first available metric key.
    metric_candidates = ["multi_logloss", "multi_error", "binary_logloss", "auc"]
    first_metric = None
    for dataset_name, metrics in evals_result.items():
        if isinstance(metrics, dict) and metrics:
            first_metric = next(iter(metrics.keys()))
            break

    metric = next((m for m in metric_candidates if first_metric and m == first_metric), first_metric)
    if metric is None:
        return False

    fig, ax = plt.subplots(figsize=(10, 6))
    lgb.plot_metric(evals_result, metric=metric, ax=ax, title=f"LightGBM Learning Curve ({metric})")
    fig.tight_layout()
    out = out_dir / f"learning_curve_{_safe_name(metric)}.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    created.append(out)
    return True


def _plot_metric_from_dom_curve_fallback(dom_train: Optional[Dict[str, object]], out_dir: Path, created: List[Path]) -> bool:
    if not dom_train:
        return False
    curve = dom_train.get("curve")
    if not isinstance(curve, list) or not curve:
        return False

    rounds = [int(p.get("fit_round", i + 1)) for i, p in enumerate(curve)]
    train_acc = [float(p.get("train_accuracy", np.nan)) for p in curve]
    val_acc = [float(p.get("val_accuracy", np.nan)) for p in curve]
    train_macro = [float(p.get("train_macro_f1", np.nan)) for p in curve]
    val_macro = [float(p.get("val_macro_f1", np.nan)) for p in curve]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(rounds, train_acc, marker="o", label="Train Acc")
    axes[0].plot(rounds, val_acc, marker="x", linestyle="--", label="Val Acc")
    axes[0].set_title("DOM LightGBM Curve (Train/Val Accuracy)")
    axes[0].set_xlabel("Fit Round")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(rounds, train_macro, marker="o", label="Train Macro-F1")
    axes[1].plot(rounds, val_macro, marker="x", linestyle="--", label="Val Macro-F1")
    axes[1].set_title("DOM LightGBM Curve (Train/Val Macro-F1)")
    axes[1].set_xlabel("Fit Round")
    axes[1].set_ylabel("Macro-F1")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    out = out_dir / "learning_curve_dom_curve_fallback.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    created.append(out)
    return True


def _pick_feature_for_histogram(booster) -> int:
    gain = booster.feature_importance(importance_type="gain")
    split = booster.feature_importance(importance_type="split")
    if gain is None or len(gain) == 0:
        return 0

    best_gain = int(np.argmax(gain))
    if float(gain[best_gain]) > 0.0:
        return best_gain

    if split is None or len(split) == 0:
        return 0
    return int(np.argmax(split))


def _plot_split_value_histogram(booster, out_dir: Path, created: List[Path]) -> None:
    feat_idx = _pick_feature_for_histogram(booster)
    fig, ax = plt.subplots(figsize=(10, 6))
    lgb.plot_split_value_histogram(
        booster,
        feature=feat_idx,
        bins=None,
        ax=ax,
        title=f"LightGBM Split Value Histogram (feature index {feat_idx})",
    )
    fig.tight_layout()
    out = out_dir / f"split_value_histogram_feature_{feat_idx}.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    created.append(out)


def _write_readme(out_dir: Path, created: List[Path], used_native_metric: bool, used_fallback: bool) -> None:
    readme = out_dir / "README.md"
    lines = [
        "# LightGBM Built-in Visualizations",
        "",
        "Generated using LightGBM plotting APIs where possible:",
        "- `plot_importance` (`split` and `gain`)",
        "- `plot_tree` (Matplotlib)",
        "- `create_tree_digraph` (Graphviz, optional)",
        "- `plot_metric` (if eval history exists)",
        "- `plot_split_value_histogram`",
        "",
        "## Files",
    ]
    for p in sorted(created):
        lines.append(f"- `{p.relative_to(out_dir).as_posix()}`")

    lines.extend(
        [
            "",
            "## Metric Plot Source",
            f"- Native LightGBM eval history used: `{used_native_metric}`",
            f"- DOM JSONL fallback used: `{used_fallback}`",
        ]
    )
    readme.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    model = _load_checkpoint_model(MODEL_PATH)
    if not hasattr(model, "booster_"):
        raise RuntimeError("Loaded model does not expose booster_.")

    booster = model.booster_
    created: List[Path] = []

    _plot_feature_importance(booster, OUT_DIR, created)
    _plot_tree_views(booster, OUT_DIR, created)
    _plot_split_value_histogram(booster, OUT_DIR, created)

    dom_train, _dom_eval = _latest_dom_eval_and_train(DOM_RESULTS_PATH)
    used_native_metric = _plot_metric_if_available(model, OUT_DIR, created)
    used_fallback = False
    if not used_native_metric:
        used_fallback = _plot_metric_from_dom_curve_fallback(dom_train, OUT_DIR, created)

    _write_readme(OUT_DIR, created, used_native_metric=used_native_metric, used_fallback=used_fallback)

    print("Generated LightGBM visuals:")
    for p in sorted(created):
        print(f" - {p}")
    print(f" - {OUT_DIR / 'README.md'}")


if __name__ == "__main__":
    main()
