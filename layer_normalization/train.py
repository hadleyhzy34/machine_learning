"""
Training utilities with gradient tracking for ablation study.
"""

import time
from typing import Dict, List, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def compute_gradient_stats(model: nn.Module) -> Dict[str, float]:
    """
    Compute gradient statistics across all parameters.

    Returns:
        Dictionary with gradient norm, mean, std, max per layer
    """
    total_norm = 0.0
    grad_means = []
    grad_stds = []
    grad_maxs = []
    layer_grads = {}

    for name, param in model.named_parameters():
        if param.grad is not None:
            # L2 norm of gradients
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2

            # Statistics
            grad_means.append(param.grad.data.abs().mean().item())
            grad_stds.append(param.grad.data.std().item())
            grad_maxs.append(param.grad.data.abs().max().item())

            # Per-layer gradient norm for transformer blocks
            if "blocks" in name and "weight" in name:
                block_idx = name.split(".")[1]
                if block_idx not in layer_grads:
                    layer_grads[block_idx] = []
                layer_grads[block_idx].append(param_norm.item())

    total_norm = total_norm ** 0.5

    return {
        "total_norm": total_norm,
        "mean": sum(grad_means) / len(grad_means) if grad_means else 0.0,
        "std": sum(grad_stds) / len(grad_stds) if grad_stds else 0.0,
        "max": max(grad_maxs) if grad_maxs else 0.0,
        "layer_grads": layer_grads,
    }


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    epoch: int,
    total_epochs: int,
    track_gradients: bool = True,
) -> Dict[str, Any]:
    """
    Train for one epoch and return metrics.

    Returns:
        Dictionary with loss, accuracy, gradient stats, and timing
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    gradient_stats = []

    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{total_epochs}")
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()

        # Track gradients before optimizer step
        if track_gradients:
            grad_stats = compute_gradient_stats(model)
            gradient_stats.append(grad_stats)

        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{100.0 * correct / total:.2f}%"
        })

    # Aggregate gradient statistics
    if gradient_stats:
        avg_grad_stats = {
            "total_norm": sum(g["total_norm"] for g in gradient_stats) / len(gradient_stats),
            "mean": sum(g["mean"] for g in gradient_stats) / len(gradient_stats),
            "std": sum(g["std"] for g in gradient_stats) / len(gradient_stats),
            "max": max(g["max"] for g in gradient_stats),
            "layer_grads": gradient_stats[-1]["layer_grads"],  # Last batch
        }
    else:
        avg_grad_stats = {}

    return {
        "loss": total_loss / len(train_loader),
        "accuracy": 100.0 * correct / total,
        "gradient_stats": avg_grad_stats,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> Dict[str, float]:
    """
    Evaluate model on test set.

    Returns:
        Dictionary with loss and accuracy
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(test_loader, desc="Evaluating")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({"acc": f"{100.0 * correct / total:.2f}%"})

    return {
        "loss": total_loss / len(test_loader),
        "accuracy": 100.0 * correct / total,
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    config,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Train model and return complete training history.

    Returns:
        Dictionary with training history, gradient history, timing, etc.
    """
    device = config.device
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )

    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "gradient_norms": [],
        "gradient_means": [],
        "gradient_stds": [],
        "layer_gradients": [],
        "learning_rates": [],
        "epoch_times": [],
    }

    if verbose:
        print(f"\nTraining with norm_type={config.norm_type}")
        print(f"Model parameters: {model.get_num_parameters():,}")

    start_time = time.time()

    for epoch in range(config.epochs):
        epoch_start = time.time()

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device,
            epoch, config.epochs, track_gradients=True
        )

        # Evaluate
        test_metrics = evaluate(model, test_loader, criterion, device)

        # Update scheduler
        scheduler.step()

        epoch_time = time.time() - epoch_start

        # Record history
        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["test_loss"].append(test_metrics["loss"])
        history["test_acc"].append(test_metrics["accuracy"])
        history["learning_rates"].append(optimizer.param_groups[0]["lr"])
        history["epoch_times"].append(epoch_time)

        # Record gradient statistics
        grad_stats = train_metrics["gradient_stats"]
        history["gradient_norms"].append(grad_stats.get("total_norm", 0))
        history["gradient_means"].append(grad_stats.get("mean", 0))
        history["gradient_stds"].append(grad_stats.get("std", 0))
        history["layer_gradients"].append(grad_stats.get("layer_grads", {}))

        if verbose:
            print(
                f"Epoch {epoch + 1}/{config.epochs}: "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train Acc: {train_metrics['accuracy']:.2f}%, "
                f"Test Loss: {test_metrics['loss']:.4f}, "
                f"Test Acc: {test_metrics['accuracy']:.2f}%, "
                f"Grad Norm: {grad_stats.get('total_norm', 0):.4f}, "
                f"Time: {epoch_time:.1f}s"
            )

    total_time = time.time() - start_time

    return {
        "history": history,
        "total_time": total_time,
        "final_test_acc": history["test_acc"][-1],
        "best_test_acc": max(history["test_acc"]),
        "best_epoch": history["test_acc"].index(max(history["test_acc"])) + 1,
    }
