"""
Evaluation and metrics collection for ablation study.
"""

from typing import Dict, List, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np


def compute_convergence_metrics(history: Dict[str, List[float]]) -> Dict[str, Any]:
    """
    Compute convergence-related metrics from training history.

    Returns:
        Dictionary with convergence metrics
    """
    test_acc = history["test_acc"]
    train_loss = history["train_loss"]

    # Epochs to reach accuracy thresholds
    thresholds = [50, 60, 70, 80]
    epochs_to_threshold = {}

    for thresh in thresholds:
        for i, acc in enumerate(test_acc):
            if acc >= thresh:
                epochs_to_threshold[f"epochs_to_{thresh}%"] = i + 1
                break
        else:
            epochs_to_threshold[f"epochs_to_{thresh}%"] = None

    # Loss variance (stability metric)
    if len(train_loss) > 1:
        loss_variance = np.var(train_loss)
        loss_trend = train_loss[-1] - train_loss[0]
    else:
        loss_variance = 0
        loss_trend = 0

    # Accuracy improvement rate
    if len(test_acc) > 1:
        acc_improvement_rate = (test_acc[-1] - test_acc[0]) / len(test_acc)
    else:
        acc_improvement_rate = 0

    return {
        "epochs_to_threshold": epochs_to_threshold,
        "loss_variance": loss_variance,
        "loss_trend": loss_trend,
        "acc_improvement_rate": acc_improvement_rate,
    }


def detect_gradient_issues(
    gradient_norms: List[float],
    threshold_vanishing: float = 1e-7,
    threshold_exploding: float = 100.0,
) -> Dict[str, Any]:
    """
    Detect vanishing or exploding gradients from gradient history.

    Returns:
        Dictionary with gradient analysis
    """
    norms = np.array(gradient_norms)

    # Check for vanishing gradients
    vanishing_count = np.sum(norms < threshold_vanishing)
    vanishing_epochs = [
        i for i, n in enumerate(norms) if n < threshold_vanishing
    ]

    # Check for exploding gradients
    exploding_count = np.sum(norms > threshold_exploding)
    exploding_epochs = [
        i for i, n in enumerate(norms) if n > threshold_exploding
    ]

    # Gradient stability
    if len(norms) > 0:
        grad_mean = np.mean(norms)
        grad_std = np.std(norms)
        grad_min = np.min(norms)
        grad_max = np.max(norms)
    else:
        grad_mean = grad_std = grad_min = grad_max = 0

    return {
        "vanishing_detected": vanishing_count > 0,
        "vanishing_epochs": vanishing_epochs,
        "exploding_detected": exploding_count > 0,
        "exploding_epochs": exploding_epochs,
        "grad_mean": grad_mean,
        "grad_std": grad_std,
        "grad_min": grad_min,
        "grad_max": grad_max,
    }


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: str,
) -> Dict[str, Any]:
    """
    Comprehensive evaluation of a trained model.

    Returns:
        Dictionary with detailed evaluation metrics
    """
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Per-class accuracy
    class_correct = [0] * 10
    class_total = [0] * 10
    for i in range(len(all_labels)):
        label = all_labels[i]
        class_correct[label] += (all_preds[i] == label)
        class_total[label] += 1

    class_accuracies = [
        100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        for i in range(10)
    ]

    return {
        "accuracy": 100.0 * correct / total,
        "per_class_accuracy": class_accuracies,
        "predictions": all_preds,
        "labels": all_labels,
        "probabilities": all_probs,
    }


def generate_summary_report(
    results: Dict[str, Dict[str, Any]],
    config,
) -> str:
    """
    Generate a text summary report of all experiments.

    Returns:
        Formatted string with summary report
    """
    report = []
    report.append("=" * 70)
    report.append("LAYER NORMALIZATION ABLATION STUDY - SUMMARY REPORT")
    report.append("=" * 70)
    report.append("")

    # Configuration summary
    report.append("Configuration:")
    report.append(f"  Epochs: {config.base_config.epochs}")
    report.append(f"  Batch size: {config.base_config.batch_size}")
    report.append(f"  Learning rate: {config.base_config.learning_rate}")
    report.append(f"  Model: {config.base_config.num_layers} layers, "
                  f"{config.base_config.embed_dim} dim, "
                  f"{config.base_config.num_heads} heads")
    report.append("")

    # Results table
    report.append("-" * 70)
    report.append(f"{'Variant':<12} {'Test Acc':<12} {'Best Acc':<12} "
                  f"{'Time':<12} {'Grad Issues':<15}")
    report.append("-" * 70)

    for variant in config.variants:
        if variant in results:
            r = results[variant]
            grad_issues = r.get("gradient_analysis", {})
            issues = []
            if grad_issues.get("vanishing_detected"):
                issues.append("vanishing")
            if grad_issues.get("exploding_detected"):
                issues.append("exploding")
            issue_str = ", ".join(issues) if issues else "none"

            report.append(
                f"{variant:<12} "
                f"{r['final_test_acc']:.2f}%      "
                f"{r['best_test_acc']:.2f}%      "
                f"{r['total_time']:.1f}s       "
                f"{issue_str:<15}"
            )

    report.append("-" * 70)
    report.append("")

    # Detailed analysis per variant
    for variant in config.variants:
        if variant not in results:
            continue

        r = results[variant]
        report.append(f"\n{'=' * 70}")
        report.append(f"Variant: {variant.upper()}")
        report.append(f"{'=' * 70}")

        # Training summary
        report.append(f"\nTraining Summary:")
        report.append(f"  Total training time: {r['total_time']:.1f}s")
        report.append(f"  Final test accuracy: {r['final_test_acc']:.2f}%")
        report.append(f"  Best test accuracy: {r['best_test_acc']:.2f}% "
                      f"(epoch {r['best_epoch']})")

        # Convergence
        conv = r.get("convergence_metrics", {})
        epochs_to = conv.get("epochs_to_threshold", {})
        report.append(f"\nConvergence:")
        for thresh, epoch in epochs_to.items():
            if epoch is not None:
                report.append(f"  {thresh}: epoch {epoch}")
            else:
                report.append(f"  {thresh}: not reached")

        # Gradient analysis
        grad = r.get("gradient_analysis", {})
        report.append(f"\nGradient Analysis:")
        report.append(f"  Mean gradient norm: {grad.get('grad_mean', 0):.6f}")
        report.append(f"  Std gradient norm: {grad.get('grad_std', 0):.6f}")
        report.append(f"  Min gradient norm: {grad.get('grad_min', 0):.6f}")
        report.append(f"  Max gradient norm: {grad.get('grad_max', 0):.6f}")

        if grad.get("vanishing_detected"):
            report.append(f"  WARNING: Vanishing gradients detected at epochs "
                          f"{grad['vanishing_epochs']}")
        if grad.get("exploding_detected"):
            report.append(f"  WARNING: Exploding gradients detected at epochs "
                          f"{grad['exploding_epochs']}")

    report.append("")
    report.append("=" * 70)
    report.append("END OF REPORT")
    report.append("=" * 70)

    return "\n".join(report)
