from __future__ import annotations

import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_phishing_detection")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "report_figures_jpg"
OUT_DIR.mkdir(exist_ok=True)

PHRESHPHISH_TEST_BENIGN = 73598
PHRESHPHISH_TEST_PHISH = 59665


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_json_with_prefix_lines(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    start = None
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped in {"[", "{"}:
            start = idx
            break
    if start is None:
        raise ValueError(f"Could not find JSON payload in {path}")
    return json.loads("\n".join(lines[start:]))


def load_jsonl(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def percent(value: float) -> float:
    return round(value * 100.0, 2)


def derive_precision_recall(fpr_pct: float, fnr_pct: float):
    fp = PHRESHPHISH_TEST_BENIGN * (fpr_pct / 100.0)
    tp = PHRESHPHISH_TEST_PHISH * (1.0 - fnr_pct / 100.0)
    precision = 100.0 * tp / (tp + fp)
    recall = 100.0 * tp / PHRESHPHISH_TEST_PHISH
    return round(precision, 2), round(recall, 2)


def setup_axes(ax, title: str, ylabel: str, ylim=(0, 100)):
    ax.set_title(title, fontsize=12, weight="bold")
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)


def percentage_ylim(*value_groups, lower=0.0):
    values = [float(v) for group in value_groups for v in group]
    if not values:
        return (lower, 100.0)
    max_value = max(values)
    if max_value >= 97.0:
        upper = 112.0
    elif max_value >= 90.0:
        upper = 105.0
    else:
        upper = math.ceil((max_value + 8.0) / 5.0) * 5.0
        upper = max(60.0, upper)
    return (lower, upper)


def annotate_bars(ax, bars, fmt="{:.2f}"):
    ymax = ax.get_ylim()[1]
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(height + ymax * 0.02, ymax * 0.96),
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=9,
            rotation=0,
        )


def save(fig: plt.Figure, name: str):
    fig.tight_layout(pad=1.4)
    fig.savefig(OUT_DIR / name, dpi=220, format="jpg", bbox_inches="tight")
    plt.close(fig)


def plot_feature_mlp_binary():
    precision, recall = derive_precision_recall(3.31, 4.36)
    metrics = {
        "Accuracy": 96.22,
        "Precision*": precision,
        "Recall*": recall,
        "Macro-F1": 96.18,
    }
    fig, ax = plt.subplots(figsize=(8.8, 6.1))
    colors = ["#1f4e79", "#2e8b57", "#c97b00", "#8b3a3a"]
    bars = ax.bar(list(metrics.keys()), list(metrics.values()), color=colors, width=0.6)
    setup_axes(ax, "FeatureMLP Only: Binary URL Detection", "Score (%)", percentage_ylim(metrics.values()))
    annotate_bars(ax, bars)
    ax.text(
        0.5,
        -0.16,
        "* Precision and recall derived from logged FPR/FNR on the shared 133,263-sample PhreshPhish test split.",
        transform=ax.transAxes,
        ha="center",
        fontsize=8,
    )
    save(fig, "url_feature_mlp_binary_metrics.jpg")


def plot_feature_mlp_multiclass():
    metrics = {
        "Accuracy": 38.20,
        "Macro-F1": 32.80,
        "Weighted-F1": 46.15,
        "MCC": 31.51,
    }
    fig, ax = plt.subplots(figsize=(8.8, 6.0))
    colors = ["#1f4e79", "#2e8b57", "#c97b00", "#8b3a3a"]
    bars = ax.bar(list(metrics.keys()), list(metrics.values()), color=colors, width=0.6)
    setup_axes(ax, "FeatureMLP Only: 52-Class URL Attribution", "Score (%)", percentage_ylim(metrics.values()))
    annotate_bars(ax, bars)
    ax.text(
        0.5,
        -0.14,
        "Committed logs report only aggregate exact-class metrics for this ablation.",
        transform=ax.transAxes,
        ha="center",
        fontsize=8,
    )
    save(fig, "url_feature_mlp_multiclass_metrics.jpg")


def plot_sabir_feature_vs_charcnn():
    feature_precision, feature_recall = derive_precision_recall(3.31, 4.36)
    char_precision, char_recall = derive_precision_recall(1.11, 1.57)

    clean_metrics = {
        "Accuracy": [96.22, 98.68],
        "Precision*": [feature_precision, char_precision],
        "Recall*": [feature_recall, char_recall],
        "Macro-F1": [96.18, 98.67],
    }
    attack_detection = {
        "Domain": [92.69, 48.23],
        "Path": [1.77, 23.78],
        "TLD": [70.37, 43.02],
    }

    labels = ["FeatureMLP", "CharCNN+FeatureMLP"]
    colors = ["#8b3a3a", "#1f4e79"]
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 6.8))

    x = np.arange(len(clean_metrics))
    vals_feature = [v[0] for v in clean_metrics.values()]
    vals_char = [v[1] for v in clean_metrics.values()]
    bars1 = axes[0].bar(x - width / 2, vals_feature, width, label=labels[0], color=colors[0])
    bars2 = axes[0].bar(x + width / 2, vals_char, width, label=labels[1], color=colors[1])
    axes[0].set_xticks(x, list(clean_metrics.keys()))
    setup_axes(
        axes[0],
        "Clean-Test Performance",
        "Score (%)",
        percentage_ylim(vals_feature, vals_char),
    )
    annotate_bars(axes[0], bars1)
    annotate_bars(axes[0], bars2)

    x2 = np.arange(len(attack_detection))
    vals_feature_attack = [v[0] for v in attack_detection.values()]
    vals_char_attack = [v[1] for v in attack_detection.values()]
    bars3 = axes[1].bar(x2 - width / 2, vals_feature_attack, width, label=labels[0], color=colors[0])
    bars4 = axes[1].bar(x2 + width / 2, vals_char_attack, width, label=labels[1], color=colors[1])
    axes[1].set_xticks(x2, list(attack_detection.keys()))
    setup_axes(
        axes[1],
        "Sabir-Style Attack Detection Rate",
        "Detection rate (%)",
        percentage_ylim(vals_feature_attack, vals_char_attack),
    )
    annotate_bars(axes[1], bars3)
    annotate_bars(axes[1], bars4)

    fig.suptitle(
        "FeatureMLP vs CharCNN+FeatureMLP Under Sabir-Style URL Attacks",
        fontsize=14,
        weight="bold",
        y=1.02,
    )
    fig.legend([bars1, bars2], labels, loc="upper center", bbox_to_anchor=(0.5, 0.98), ncols=2, frameon=False)
    save(fig, "url_sabir_feature_vs_charcnn.jpg")


def plot_gan_defenses():
    methods = [
        "MLP\nNo defense",
        "MLP\nAdv training",
        "MLP\nMahalanobis",
        "Exact-paper\nGAN control",
        "CharCNN+MLP\nMahalanobis",
    ]
    normal_accuracy = [95.44, 94.34, 95.45, 95.41, 98.68]
    gan_evasion = [100.0, 0.0, 0.0, 0.0, 0.0]

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 6.8))
    colors = ["#8b3a3a", "#c97b00", "#2e8b57", "#6a5acd", "#1f4e79"]

    x = np.arange(len(methods))
    bars1 = axes[0].bar(x, normal_accuracy, color=colors, width=0.65)
    axes[0].set_xticks(x, methods)
    setup_axes(axes[0], "Normal-Test Accuracy", "Accuracy (%)", percentage_ylim(normal_accuracy))
    annotate_bars(axes[0], bars1)
    axes[0].tick_params(axis="x", pad=8)

    bars2 = axes[1].bar(x, gan_evasion, color=colors, width=0.65)
    axes[1].set_xticks(x, methods)
    setup_axes(axes[1], "GAN Evasion Rate", "Evasion rate (%)", percentage_ylim(gan_evasion))
    annotate_bars(axes[1], bars2)
    axes[1].set_ylabel("Evasion rate (%)  [lower is better]")
    axes[1].tick_params(axis="x", pad=8)

    fig.suptitle(
        "GAN Defense Comparison on PhreshPhish Feature-Space Attacks",
        fontsize=14,
        weight="bold",
        y=1.02,
    )
    fig.text(
        0.5,
        0.01,
        "Cross-dataset Mahalanobis validation is discussed in the report table and omitted here because it uses a different test distribution.",
        ha="center",
        fontsize=8,
    )
    save(fig, "url_gan_defense_comparison.jpg")


def plot_dom_binary(dom_sup, dom_iforest, dom_ocsvm):
    models = ["Random Forest", "Extra Trees", "Isolation Forest", "One-Class SVM"]
    by_name = {item["algorithm"]: item for item in dom_sup}
    metrics = {
        "Accuracy": [
            percent(by_name["random_forest"]["accuracy"]),
            percent(by_name["extra_trees"]["accuracy"]),
            percent(dom_iforest["accuracy"]),
            percent(dom_ocsvm["accuracy"]),
        ],
        "Balanced Acc.": [
            percent(by_name["random_forest"]["balanced_accuracy"]),
            percent(by_name["extra_trees"]["balanced_accuracy"]),
            percent(dom_iforest["balanced_accuracy"]),
            percent(dom_ocsvm["balanced_accuracy"]),
        ],
        "ROC-AUC": [
            percent(by_name["random_forest"]["roc_auc"]),
            percent(by_name["extra_trees"]["roc_auc"]),
            percent(dom_iforest["roc_auc"]),
            percent(dom_ocsvm["roc_auc"]),
        ],
        "Phishing F1": [
            percent(by_name["random_forest"]["per_class"]["phishing"]["f1"]),
            percent(by_name["extra_trees"]["per_class"]["phishing"]["f1"]),
            percent(dom_iforest["per_class"]["phishing"]["f1"]),
            percent(dom_ocsvm["per_class"]["phishing"]["f1"]),
        ],
    }

    fig, ax = plt.subplots(figsize=(11.8, 6.6))
    x = np.arange(len(models))
    width = 0.18
    colors = ["#1f4e79", "#2e8b57", "#c97b00", "#8b3a3a"]
    for idx, (metric, values) in enumerate(metrics.items()):
        bars = ax.bar(x + (idx - 1.5) * width, values, width, label=metric, color=colors[idx])
        annotate_bars(ax, bars)
    ax.set_xticks(x, models)
    setup_axes(
        ax,
        "DOM Binary Model Comparison",
        "Score (%)",
        percentage_ylim(*metrics.values()),
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), frameon=False, ncols=2)
    save(fig, "dom_binary_model_comparison.jpg")


def plot_dom_majority_removal():
    models = ["Random Forest", "Extra Trees", "Logistic Regression", "LightGBM"]
    acc_10 = [69.81, 45.83, 65.71, 71.03]
    acc_7 = [75.28, 74.94, 67.46, 75.04]
    f1_10 = [68.40, 49.59, 62.31, 69.55]
    f1_7 = [77.02, 76.67, 68.43, 76.78]

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 6.8))
    x = np.arange(len(models))
    width = 0.35

    bars1 = axes[0].bar(x - width / 2, acc_10, width, label="10 classes", color="#8b3a3a")
    bars2 = axes[0].bar(x + width / 2, acc_7, width, label="7 classes", color="#1f4e79")
    axes[0].set_xticks(x, models, rotation=10)
    setup_axes(axes[0], "Accuracy Before vs After Majority Removal", "Accuracy (%)", percentage_ylim(acc_10, acc_7))
    annotate_bars(axes[0], bars1)
    annotate_bars(axes[0], bars2)

    bars3 = axes[1].bar(x - width / 2, f1_10, width, label="10 classes", color="#8b3a3a")
    bars4 = axes[1].bar(x + width / 2, f1_7, width, label="7 classes", color="#1f4e79")
    axes[1].set_xticks(x, models, rotation=10)
    setup_axes(axes[1], "Macro-F1 Before vs After Majority Removal", "Macro-F1 (%)", percentage_ylim(f1_10, f1_7))
    annotate_bars(axes[1], bars3)
    annotate_bars(axes[1], bars4)

    fig.suptitle("Effect of Removing the Three Majority DOM Classes", fontsize=14, weight="bold", y=1.02)
    fig.legend([bars1, bars2], ["10 classes", "7 classes"], loc="upper center", bbox_to_anchor=(0.5, 0.98), ncols=2, frameon=False)
    save(fig, "dom_majority_removal_comparison.jpg")


def plot_visual_binary(visual_binary):
    models = []
    accuracy = []
    roc_auc = []
    macro_f1 = []
    phishing_f1 = []
    for item in visual_binary:
        name = item["model_type"].replace("_", " ").title()
        models.append(name)
        accuracy.append(percent(item["accuracy"]))
        roc_auc.append(percent(item["roc_auc"]))
        macro_f1.append(percent(item["classification_report"]["macro avg"]["f1-score"]))
        phishing_f1.append(percent(item["per_class"]["phishing"]["f1"]))

    fig, ax = plt.subplots(figsize=(11.8, 6.6))
    x = np.arange(len(models))
    width = 0.18
    metrics = {
        "Accuracy": accuracy,
        "ROC-AUC": roc_auc,
        "Macro-F1": macro_f1,
        "Phishing F1": phishing_f1,
    }
    colors = ["#1f4e79", "#2e8b57", "#c97b00", "#8b3a3a"]
    for idx, (metric, values) in enumerate(metrics.items()):
        bars = ax.bar(x + (idx - 1.5) * width, values, width, label=metric, color=colors[idx])
        annotate_bars(ax, bars)
    ax.set_xticks(x, models)
    setup_axes(ax, "Visual Binary Model Comparison", "Score (%)", percentage_ylim(*metrics.values()))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), frameon=False, ncols=2)
    save(fig, "visual_binary_model_comparison.jpg")


def plot_visual_multiclass(visual_dual):
    results = visual_dual["results"]
    ordered = ["cnn1d", "extra_trees", "logistic_regression", "mlp", "random_forest"]
    labels = ["CNN1D", "Extra Trees", "Logistic Regression", "MLP", "Random Forest"]
    brand_acc = [percent(results[key]["brand"]["accuracy"]) for key in ordered]
    macro_f1 = [percent(results[key]["brand"]["macro_f1"]) for key in ordered]
    weighted_f1 = [percent(results[key]["brand"]["weighted_f1"]) for key in ordered]

    fig, ax = plt.subplots(figsize=(12.0, 6.6))
    x = np.arange(len(labels))
    width = 0.22
    metrics = {
        "Brand Accuracy": brand_acc,
        "Macro-F1": macro_f1,
        "Weighted-F1": weighted_f1,
    }
    colors = ["#1f4e79", "#2e8b57", "#c97b00"]
    for idx, (metric, values) in enumerate(metrics.items()):
        bars = ax.bar(x + (idx - 1) * width, values, width, label=metric, color=colors[idx])
        annotate_bars(ax, bars)
    ax.set_xticks(x, labels)
    setup_axes(ax, "Visual Multiclass Brand Comparison", "Score (%)", percentage_ylim(*metrics.values()))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.14), frameon=False, ncols=3)
    save(fig, "visual_multiclass_model_comparison.jpg")


def main():
    dom_sup = load_json_with_prefix_lines(ROOT / "dom_model_compare_sup_eval.json")
    dom_iforest = load_json_with_prefix_lines(ROOT / "dom_model_compare_iforest_eval.json")
    dom_ocsvm = load_json_with_prefix_lines(ROOT / "dom_model_compare_ocsvm_eval.json")
    visual_binary = load_json(ROOT / "visual_benchmark_report_full.json")
    visual_dual = load_json(ROOT / "visual_dual_report_all.json")

    plot_feature_mlp_binary()
    plot_feature_mlp_multiclass()
    plot_sabir_feature_vs_charcnn()
    plot_gan_defenses()
    plot_dom_binary(dom_sup, dom_iforest, dom_ocsvm)
    plot_dom_majority_removal()
    plot_visual_binary(visual_binary)
    plot_visual_multiclass(visual_dual)

    print(f"generated {len(list(OUT_DIR.glob('*.jpg')))} jpg files in {OUT_DIR}")


if __name__ == "__main__":
    main()
