#!/usr/bin/env python3
"""
modules/visualization.py
Centralized, publication-quality visualization functions.
All plots use consistent styling, colors, and dual-format saving (PDF + PNG).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    FIGURE_DPI, FIGURE_FORMATS, FIGURE_STYLE,
    MODEL_COLORS, MODEL_DISPLAY_NAMES,
)

# ──────────────────────────────────────────────
#  Global style setup
# ──────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": FIGURE_DPI,
    "savefig.dpi": FIGURE_DPI,
    "savefig.bbox": "tight",
})
sns.set_style(FIGURE_STYLE)


def _save_fig(fig, output_dir, filename_stem, formats=None):
    """Save a figure in multiple formats."""
    fmts = formats or FIGURE_FORMATS
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in fmts:
        path = output_dir / f"{filename_stem}.{fmt}"
        fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] Saved: {filename_stem} ({', '.join(fmts)})")


def _get_color(model_name):
    return MODEL_COLORS.get(model_name, "#607D8B")


def _get_display_name(model_name):
    return MODEL_DISPLAY_NAMES.get(model_name, model_name)


# ──────────────────────────────────────────────
#  Confusion Matrix
# ──────────────────────────────────────────────
def plot_confusion_matrix(cm, labels, output_dir, model_name="model", title=None):
    """Plot and save a confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax,
        linewidths=0.5, linecolor="white",
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    # ax.set_title(title or f"Confusion Matrix — {_get_display_name(model_name)}")
    _save_fig(fig, output_dir, f"confusion_matrix_{model_name}")


# ──────────────────────────────────────────────
#  ROC Curves (overlay multiple models)
# ──────────────────────────────────────────────
def plot_roc_curves(roc_data, output_dir, filename="roc_curves_overlay"):
    """
    Plot ROC curves for multiple models on one figure.
    roc_data: dict of {model_name: {"fpr": array, "tpr": array, "auc": float}}
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    for model_name, data in roc_data.items():
        ax.plot(
            data["fpr"], data["tpr"],
            label=f'{_get_display_name(model_name)} (AUC = {data["auc"]:.4f})',
            color=_get_color(model_name), linewidth=2,
        )
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random Baseline")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    # ax.set_title("ROC Curves — Model Comparison")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(True, alpha=0.3)
    _save_fig(fig, output_dir, filename)


# ──────────────────────────────────────────────
#  PR Curves (overlay multiple models)
# ──────────────────────────────────────────────
def plot_pr_curves(pr_data, output_dir, filename="pr_curves_overlay"):
    """
    Plot Precision-Recall curves for multiple models.
    pr_data: dict of {model_name: {"precision": array, "recall": array, "auprc": float}}
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    for model_name, data in pr_data.items():
        ax.plot(
            data["recall"], data["precision"],
            label=f'{_get_display_name(model_name)} (AUPRC = {data["auprc"]:.4f})',
            color=_get_color(model_name), linewidth=2,
        )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    # ax.set_title("Precision-Recall Curves — Model Comparison")
    ax.legend(loc="lower left", framealpha=0.9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(True, alpha=0.3)
    _save_fig(fig, output_dir, filename)


# ──────────────────────────────────────────────
#  Calibration Curves
# ──────────────────────────────────────────────
def plot_calibration_curves(cal_data, output_dir, filename="calibration_curves"):
    """
    Plot calibration (reliability) diagrams for multiple models.
    cal_data: dict of {model_name: {"prob_true": array, "prob_pred": array, "brier": float}}
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Perfectly Calibrated")
    for model_name, data in cal_data.items():
        ax.plot(
            data["prob_pred"], data["prob_true"],
            "s-", label=f'{_get_display_name(model_name)} (Brier = {data["brier"]:.4f})',
            color=_get_color(model_name), linewidth=1.5, markersize=5,
        )
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    # ax.set_title("Calibration Curves (Reliability Diagrams)")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(True, alpha=0.3)
    _save_fig(fig, output_dir, filename)


# ──────────────────────────────────────────────
#  CV Box Plots
# ──────────────────────────────────────────────
def plot_cv_boxplots(cv_scores, output_dir, metric_name="f1_macro", filename="cv_boxplots"):
    """
    Plot box plots of CV fold scores for multiple models.
    cv_scores: dict of {model_name: list_of_fold_scores}
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = []
    data = []
    colors = []
    for model_name, scores in cv_scores.items():
        labels.append(_get_display_name(model_name))
        data.append(scores)
        colors.append(_get_color(model_name))

    bp = ax.boxplot(data, patch_artist=True, labels=labels, widths=0.6)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel(metric_name.replace("_", " ").title())
    # ax.set_title(f"Cross-Validation {metric_name.replace('_', ' ').title()} — Model Comparison")
    ax.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=25, ha="right")
    _save_fig(fig, output_dir, filename)


# ──────────────────────────────────────────────
#  Model Comparison Heatmap
# ──────────────────────────────────────────────
def plot_model_comparison_heatmap(comparison_df, output_dir, filename="model_comparison_heatmap"):
    """
    Plot a heatmap of model performance across multiple metrics.
    comparison_df: DataFrame with models as rows and metrics as columns.
    """
    fig, ax = plt.subplots(figsize=(10, max(4, len(comparison_df) * 0.8)))
    sns.heatmap(
        comparison_df, annot=True, fmt=".4f", cmap="YlGnBu",
        linewidths=0.5, linecolor="white", ax=ax,
    )
    # ax.set_title("Model Performance Comparison")
    ax.set_xlabel("Metrics")
    ax.set_ylabel("Models")
    _save_fig(fig, output_dir, filename)


# ──────────────────────────────────────────────
#  Lorenz Curve
# ──────────────────────────────────────────────
def plot_lorenz_curve(values, output_dir, title="Lorenz Curve", filename="lorenz_curve", gini=None):
    """Plot a Lorenz curve for revenue concentration analysis."""
    sorted_vals = np.sort(values)
    cumulative = np.cumsum(sorted_vals)
    cumulative = cumulative / cumulative[-1]
    n = len(sorted_vals)
    x = np.arange(1, n + 1) / n

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(x, cumulative, color="#2196F3", linewidth=2, label="Lorenz Curve")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Perfect Equality")
    ax.fill_between(x, cumulative, np.linspace(0, 1, n), alpha=0.15, color="#2196F3")
    label = title
    if gini is not None:
        label += f" (Gini = {gini:.4f})"
    ax.set_xlabel("Cumulative Share of SKUs/Products")
    ax.set_ylabel("Cumulative Share of Revenue")
    # ax.set_title(label)
    ax.legend(loc="upper left")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    _save_fig(fig, output_dir, filename)


# ──────────────────────────────────────────────
#  Venn-like Overlap Chart (ASIN Matching)
# ──────────────────────────────────────────────
def plot_asin_overlap(reviews_only, sales_only, both, output_dir, filename="asin_overlap"):
    """Plot a bar chart showing ASIN overlap between reviews and sales."""
    fig, ax = plt.subplots(figsize=(7, 4))
    categories = ["Reviews Only", "Both (Matched)", "Sales Only"]
    counts = [reviews_only, both, sales_only]
    colors = ["#FF7043", "#66BB6A", "#42A5F5"]
    bars = ax.bar(categories, counts, color=colors, edgecolor="white", linewidth=1.5)
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.02,
            f"{count:,}", ha="center", fontweight="bold",
        )
    ax.set_ylabel("Number of Unique ASINs")
    # ax.set_title("ASIN Coverage: Reviews ↔ Sales Data Linkage")
    ax.grid(True, alpha=0.3, axis="y")
    _save_fig(fig, output_dir, filename)


# ──────────────────────────────────────────────
#  Bar charts (generic, for top SKUs, regions, etc.)
# ──────────────────────────────────────────────
def plot_horizontal_bar(data, x_col, y_col, output_dir, title="", filename="barplot", color="#2196F3"):
    """Generic horizontal bar chart."""
    fig, ax = plt.subplots(figsize=(8, max(4, len(data) * 0.35)))
    sns.barplot(data=data, x=x_col, y=y_col, ax=ax, color=color)
    # ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="x")
    _save_fig(fig, output_dir, filename)


# ──────────────────────────────────────────────
#  Boxplot (generic, for MRP dispersion, etc.)
# ──────────────────────────────────────────────
def plot_boxplot(data, x_col, y_col, output_dir, title="", filename="boxplot"):
    """Generic grouped boxplot."""
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=data, x=x_col, y=y_col, ax=ax, palette="Set2", hue=y_col, linewidth=1.5)
    # ax.set_title(title)
    plt.xticks(rotation=30, ha="right")
    _save_fig(fig, output_dir, filename)


# ──────────────────────────────────────────────
#  Imbalance strategy comparison bar chart
# ──────────────────────────────────────────────
def plot_imbalance_comparison(results_df, output_dir, filename="imbalance_recall_comparison"):
    """Plot negative-class recall across different imbalance handling strategies."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Negative recall
    ax1 = axes[0]
    bars = ax1.barh(
        results_df["strategy"], results_df["negative_recall"],
        color="#FF5722", edgecolor="white", linewidth=1,
    )
    ax1.set_xlabel("Negative Class Recall")
    # ax1.set_title("Negative Recall by Imbalance Strategy")
    ax1.set_xlim([0, 1])
    ax1.grid(True, alpha=0.3, axis="x")
    for bar, val in zip(bars, results_df["negative_recall"]):
        ax1.text(val + 0.01, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center")

    # Plot 2: Overall F1 macro
    ax2 = axes[1]
    bars2 = ax2.barh(
        results_df["strategy"], results_df["f1_macro"],
        color="#2196F3", edgecolor="white", linewidth=1,
    )
    ax2.set_xlabel("Macro F1 Score")
    # ax2.set_title("Macro F1 by Imbalance Strategy")
    ax2.set_xlim([0, 1])
    ax2.grid(True, alpha=0.3, axis="x")
    for bar, val in zip(bars2, results_df["f1_macro"]):
        ax2.text(val + 0.01, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center")

    plt.tight_layout()
    _save_fig(fig, output_dir, filename)


# ──────────────────────────────────────────────
#  Error analysis plot
# ──────────────────────────────────────────────
def plot_error_analysis(error_df, output_dir, model_name="model"):
    """
    Plot error analysis visualizations for misclassified samples.
    error_df must have columns: 'text_clean', 'true_label', 'pred_label', 'text_length'
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1. Text length distribution of errors vs correct
    ax1 = axes[0]
    if "text_length" in error_df.columns and "is_correct" in error_df.columns:
        for label, color in [("Correct", "#66BB6A"), ("Misclassified", "#FF5722")]:
            subset = error_df[error_df["is_correct"] == (label == "Correct")]
            ax1.hist(subset["text_length"], bins=50, alpha=0.6, color=color, label=label, density=True)
        ax1.set_xlabel("Text Length (chars)")
        ax1.set_ylabel("Density")
        # ax1.set_title("Text Length: Correct vs Errors")
        ax1.legend()

    # 2. Score distribution of misclassified samples
    ax2 = axes[1]
    if "Score" in error_df.columns and "is_correct" in error_df.columns:
        errors = error_df[~error_df["is_correct"]]
        if not errors.empty:
            errors["Score"].value_counts().sort_index().plot(kind="bar", ax=ax2, color="#FF5722")
            ax2.set_xlabel("Original Score")
            ax2.set_ylabel("Count")
            # ax2.set_title("Score Distribution of Misclassified Reviews")

    # 3. Confusion pattern
    ax3 = axes[2]
    if "true_label" in error_df.columns and "pred_label" in error_df.columns:
        errors = error_df[~error_df["is_correct"]] if "is_correct" in error_df.columns else error_df
        if not errors.empty:
            pattern = errors.groupby(["true_label", "pred_label"]).size().reset_index(name="count")
            pattern["pattern"] = pattern["true_label"] + " → " + pattern["pred_label"]
            sns.barplot(data=pattern, x="count", y="pattern", ax=ax3, palette="Reds_r", hue="true_label")
            ax3.set_xlabel("Count")
            # ax3.set_title("Misclassification Patterns")

    # plt.suptitle(f"Error Analysis — {_get_display_name(model_name)}", fontsize=14, y=1.02)
    plt.tight_layout()
    _save_fig(fig, output_dir, f"error_analysis_{model_name}")
