"""
Evaluation module: computes and visualizes all performance metrics.
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- Per-fold performance bar chart
- ROC Curve
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
)

import config


def compute_metrics(y_true, y_pred, y_probs=None):
    """Compute all classification metrics and print a summary."""
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="binary", pos_label=1)
    recall = recall_score(y_true, y_pred, average="binary", pos_label=1)
    f1 = f1_score(y_true, y_pred, average="binary", pos_label=1)

    print("\n" + "=" * 60)
    print("  CLASSIFICATION METRICS (Aggregated across all folds)")
    print("=" * 60)
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print("\n  Detailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=config.CLASS_NAMES))

    metrics = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }

    return metrics


def plot_confusion_matrix(y_true, y_pred):
    """Plot and save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw counts
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=config.CLASS_NAMES, yticklabels=config.CLASS_NAMES,
        ax=axes[0], cbar=True,
    )
    axes[0].set_xlabel("Predicted Label", fontsize=12)
    axes[0].set_ylabel("True Label", fontsize=12)
    axes[0].set_title("Confusion Matrix (Counts)", fontsize=14)

    # Normalized
    sns.heatmap(
        cm_normalized, annot=True, fmt=".2%", cmap="Blues",
        xticklabels=config.CLASS_NAMES, yticklabels=config.CLASS_NAMES,
        ax=axes[1], cbar=True,
    )
    axes[1].set_xlabel("Predicted Label", fontsize=12)
    axes[1].set_ylabel("True Label", fontsize=12)
    axes[1].set_title("Confusion Matrix (Normalized)", fontsize=14)

    plt.tight_layout()
    path = os.path.join(config.PLOTS_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix saved to {path}")


def plot_fold_accuracies(fold_accuracies):
    """Plot per-fold accuracy bar chart."""
    fig, ax = plt.subplots(figsize=(8, 5))
    folds = [f"Fold {i+1}" for i in range(len(fold_accuracies))]
    colors = sns.color_palette("viridis", len(fold_accuracies))

    bars = ax.bar(folds, fold_accuracies, color=colors, edgecolor="black", linewidth=0.5)

    # Add value labels on bars
    for bar, acc in zip(bars, fold_accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{acc:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    mean_acc = np.mean(fold_accuracies)
    ax.axhline(y=mean_acc, color="red", linestyle="--", linewidth=1.5, label=f"Mean: {mean_acc:.4f}")
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Cross-Validation Fold Accuracies", fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11)

    plt.tight_layout()
    path = os.path.join(config.PLOTS_DIR, "fold_accuracies.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Fold accuracy chart saved to {path}")


def plot_roc_curve(y_true, y_probs):
    """Plot ROC curve using probability scores."""
    # Probability of the positive class (HR = class 1)
    if y_probs.ndim == 2:
        y_score = y_probs[:, 1]
    else:
        y_score = y_probs

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random Classifier")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve — DR vs HR Classification", fontsize=14)
    ax.legend(loc="lower right", fontsize=11)

    plt.tight_layout()
    path = os.path.join(config.PLOTS_DIR, "roc_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ROC curve saved to {path}")
    print(f"  AUC Score: {roc_auc:.4f}")


def plot_metrics_summary(metrics):
    """Plot a summary bar chart of all metrics."""
    fig, ax = plt.subplots(figsize=(7, 5))
    metric_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
    values = [metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1_score"]]
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]

    bars = ax.bar(metric_names, values, color=colors, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{val:.4f}", ha="center", va="bottom", fontsize=12, fontweight="bold",
        )

    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Performance Metrics", fontsize=14)

    plt.tight_layout()
    path = os.path.join(config.PLOTS_DIR, "metrics_summary.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Metrics summary chart saved to {path}")


def run_evaluation(y_true, y_pred, y_probs, fold_accuracies):
    """Run full evaluation pipeline: metrics + all plots."""
    print("\n" + "#" * 60)
    print("  EVALUATION")
    print("#" * 60)

    metrics = compute_metrics(y_true, y_pred, y_probs)
    plot_confusion_matrix(y_true, y_pred)
    plot_fold_accuracies(fold_accuracies)
    plot_roc_curve(y_true, y_probs)
    plot_metrics_summary(metrics)

    # Save a machine-readable summary for reporting and reproducibility.
    summary = {
        "accuracy": float(metrics["accuracy"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1_score": float(metrics["f1_score"]),
        "fold_accuracies": [float(acc) for acc in fold_accuracies],
        "mean_fold_accuracy": float(np.mean(fold_accuracies)),
        "std_fold_accuracy": float(np.std(fold_accuracies)),
    }
    metrics_json_path = os.path.join(config.PLOTS_DIR, "metrics_summary.json")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  Metrics JSON saved to {metrics_json_path}")

    print(f"\n  All plots saved to: {config.PLOTS_DIR}")
    return metrics
