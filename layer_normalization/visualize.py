"""
Visualization utilities for layer normalization ablation study.
"""

import os
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import numpy as np


# Color scheme for different variants
VARIANT_COLORS = {
    "none": "#e74c3c",   # Red
    "post": "#3498db",   # Blue
    "pre": "#2ecc71",    # Green
    "rms": "#9b59b6",    # Purple
}

VARIANT_LABELS = {
    "none": "No-LN",
    "post": "Post-LN",
    "pre": "Pre-LN",
    "rms": "RMSNorm",
}


def plot_training_curves(
    results: Dict[str, Dict[str, Any]],
    save_path: str,
):
    """
    Plot training curves for all variants on the same plot.

    Creates 2x2 subplot with:
    - Training loss
    - Training accuracy
    - Test loss
    - Test accuracy
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for variant, r in results.items():
        history = r["history"]
        color = VARIANT_COLORS.get(variant, "#333333")
        label = VARIANT_LABELS.get(variant, variant)

        epochs = range(1, len(history["train_loss"]) + 1)

        # Training loss
        axes[0, 0].plot(epochs, history["train_loss"], color=color,
                        label=label, linewidth=2)

        # Training accuracy
        axes[0, 1].plot(epochs, history["train_acc"], color=color,
                        label=label, linewidth=2)

        # Test loss
        axes[1, 0].plot(epochs, history["test_loss"], color=color,
                        label=label, linewidth=2)

        # Test accuracy
        axes[1, 1].plot(epochs, history["test_acc"], color=color,
                        label=label, linewidth=2)

    # Configure subplots
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Training Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy (%)")
    axes[0, 1].set_title("Training Accuracy")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].set_title("Test Loss")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Accuracy (%)")
    axes[1, 1].set_title("Test Accuracy")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle("Layer Normalization Ablation Study - Training Curves",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved training curves to {save_path}")


def plot_gradient_flow(
    results: Dict[str, Dict[str, Any]],
    save_path: str,
):
    """
    Plot gradient flow comparison across variants.

    Creates:
    - Gradient norm over epochs for each variant
    - Layer-wise gradient heatmap
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Gradient norm over epochs
    ax = axes[0, 0]
    for variant, r in results.items():
        history = r["history"]
        color = VARIANT_COLORS.get(variant, "#333333")
        label = VARIANT_LABELS.get(variant, variant)

        epochs = range(1, len(history["gradient_norms"]) + 1)
        ax.plot(epochs, history["gradient_norms"], color=color,
                label=label, linewidth=2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Gradient Norm (L2)")
    ax.set_title("Gradient Norm Over Training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    # Plot 2: Gradient mean over epochs
    ax = axes[0, 1]
    for variant, r in results.items():
        history = r["history"]
        color = VARIANT_COLORS.get(variant, "#333333")
        label = VARIANT_LABELS.get(variant, variant)

        epochs = range(1, len(history["gradient_means"]) + 1)
        ax.plot(epochs, history["gradient_means"], color=color,
                label=label, linewidth=2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Gradient Mean")
    ax.set_title("Gradient Mean Over Training")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Layer-wise gradient heatmap (last epoch)
    ax = axes[1, 0]
    # Get layer gradients from the last epoch for each variant
    layer_data = {}
    for variant, r in results.items():
        layer_grads = r["history"]["layer_gradients"][-1]
        if layer_grads:
            # Average gradient norm per layer
            layer_norms = []
            for layer_idx in sorted(layer_grads.keys(), key=int):
                layer_norms.append(np.mean(layer_grads[layer_idx]))
            layer_data[variant] = layer_norms

    if layer_data:
        # Create heatmap data
        variants = list(layer_data.keys())
        max_layers = max(len(v) for v in layer_data.values())
        heatmap_data = np.zeros((len(variants), max_layers))

        for i, variant in enumerate(variants):
            for j, val in enumerate(layer_data[variant]):
                heatmap_data[i, j] = val

        im = ax.imshow(heatmap_data, aspect="auto", cmap="viridis")
        ax.set_yticks(range(len(variants)))
        ax.set_yticklabels([VARIANT_LABELS.get(v, v) for v in variants])
        ax.set_xlabel("Layer Index")
        ax.set_title("Layer-wise Gradient Norm (Last Epoch)")
        plt.colorbar(im, ax=ax, label="Gradient Norm")

    # Plot 4: Gradient stability (std over training)
    ax = axes[1, 1]
    bar_width = 0.2
    x = np.arange(len(results))

    grad_means = []
    grad_stds = []
    labels = []

    for variant in ["none", "post", "pre", "rms"]:
        if variant in results:
            grad = results[variant].get("gradient_analysis", {})
            grad_means.append(grad.get("grad_mean", 0))
            grad_stds.append(grad.get("grad_std", 0))
            labels.append(VARIANT_LABELS.get(variant, variant))

    x = np.arange(len(labels))
    ax.bar(x - bar_width/2, grad_means, bar_width, label="Mean", alpha=0.8)
    ax.bar(x + bar_width/2, grad_stds, bar_width, label="Std", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Gradient Statistic")
    ax.set_title("Gradient Statistics Summary")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Gradient Flow Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved gradient flow plot to {save_path}")


def plot_performance_summary(
    results: Dict[str, Dict[str, Any]],
    save_path: str,
):
    """
    Create a performance summary bar chart.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    variants = []
    final_accs = []
    best_accs = []
    times = []

    for variant in ["none", "post", "pre", "rms"]:
        if variant in results:
            r = results[variant]
            variants.append(VARIANT_LABELS.get(variant, variant))
            final_accs.append(r["final_test_acc"])
            best_accs.append(r["best_test_acc"])
            times.append(r["total_time"])

    colors = [VARIANT_COLORS.get(v, "#333333") for v in ["none", "post", "pre", "rms"]
              if v in results]

    # Final accuracy
    x = np.arange(len(variants))
    axes[0].bar(x, final_accs, color=colors, alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(variants)
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("Final Test Accuracy")
    axes[0].grid(True, alpha=0.3, axis="y")

    # Best accuracy
    axes[1].bar(x, best_accs, color=colors, alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(variants)
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Best Test Accuracy")
    axes[1].grid(True, alpha=0.3, axis="y")

    # Training time
    axes[2].bar(x, times, color=colors, alpha=0.8)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(variants)
    axes[2].set_ylabel("Time (seconds)")
    axes[2].set_title("Total Training Time")
    axes[2].grid(True, alpha=0.3, axis="y")

    plt.suptitle("Performance Summary", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved performance summary to {save_path}")


def plot_convergence_comparison(
    results: Dict[str, Dict[str, Any]],
    save_path: str,
):
    """
    Plot convergence speed comparison.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot test accuracy curves with convergence annotations
    for variant, r in results.items():
        history = r["history"]
        color = VARIANT_COLORS.get(variant, "#333333")
        label = VARIANT_LABELS.get(variant, variant)

        epochs = range(1, len(history["test_acc"]) + 1)
        ax.plot(epochs, history["test_acc"], color=color,
                label=label, linewidth=2)

        # Mark best accuracy point
        best_idx = np.argmax(history["test_acc"])
        ax.scatter([best_idx + 1], [history["test_acc"][best_idx]],
                   color=color, s=100, zorder=5, marker="*")

    # Add threshold lines
    for thresh in [50, 60, 70]:
        ax.axhline(y=thresh, color="gray", linestyle="--", alpha=0.5)
        ax.text(len(history["test_acc"]) + 0.5, thresh, f"{thresh}%",
                va="center", fontsize=9, color="gray")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Convergence Speed Comparison\n(stars mark best accuracy)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved convergence comparison to {save_path}")


def create_results_table(
    results: Dict[str, Dict[str, Any]],
    save_path: str,
):
    """
    Create a formatted results table as an image.
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("off")

    # Prepare table data
    headers = ["Variant", "Final Acc", "Best Acc", "Best Epoch",
               "Train Time", "Grad Mean", "Grad Std", "Issues"]

    table_data = []
    for variant in ["none", "post", "pre", "rms"]:
        if variant not in results:
            continue

        r = results[variant]
        grad = r.get("gradient_analysis", {})

        issues = []
        if grad.get("vanishing_detected"):
            issues.append("vanishing")
        if grad.get("exploding_detected"):
            issues.append("exploding")
        issue_str = ", ".join(issues) if issues else "none"

        row = [
            VARIANT_LABELS.get(variant, variant),
            f"{r['final_test_acc']:.2f}%",
            f"{r['best_test_acc']:.2f}%",
            str(r["best_epoch"]),
            f"{r['total_time']:.1f}s",
            f"{grad.get('grad_mean', 0):.4f}",
            f"{grad.get('grad_std', 0):.4f}",
            issue_str,
        ]
        table_data.append(row)

    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # Style header row
    for i in range(len(headers)):
        table[(0, i)].set_facecolor("#4a90d9")
        table[(0, i)].set_text_props(color="white", fontweight="bold")

    # Color code the variant column
    for i, variant in enumerate(["none", "post", "pre", "rms"]):
        if variant in results:
            color = VARIANT_COLORS.get(variant, "#333333")
            table[(i + 1, 0)].set_facecolor(color)
            table[(i + 1, 0)].set_text_props(color="white", fontweight="bold")

    plt.title("Layer Normalization Ablation Study - Results Summary",
              fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved results table to {save_path}")


def generate_all_plots(
    results: Dict[str, Dict[str, Any]],
    output_dir: str,
):
    """
    Generate all visualization plots.
    """
    os.makedirs(output_dir, exist_ok=True)

    plot_training_curves(
        results,
        os.path.join(output_dir, "training_curves.png")
    )

    plot_gradient_flow(
        results,
        os.path.join(output_dir, "gradient_flow.png")
    )

    plot_performance_summary(
        results,
        os.path.join(output_dir, "performance_summary.png")
    )

    plot_convergence_comparison(
        results,
        os.path.join(output_dir, "convergence_comparison.png")
    )

    create_results_table(
        results,
        os.path.join(output_dir, "results_table.png")
    )
