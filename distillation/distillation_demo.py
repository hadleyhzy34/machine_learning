"""
Knowledge Distillation Demo for Vision Tasks

This demo shows how knowledge distillation helps a small student model
learn from a larger teacher model, achieving better performance than
training the student alone.

Dataset: CIFAR-10
Teacher: ResNet-18 (pre-trained)
Student: A small custom CNN
Comparison: Student with distillation vs. Student without distillation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import time
import matplotlib.pyplot as plt
from tqdm import tqdm


# ==================== Models ====================


class SmallStudent(nn.Module):
    """A small CNN suitable for CIFAR-10, much smaller than ResNet-18"""

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def get_teacher_model(num_classes=10, device="cpu"):
    """Get a pre-trained ResNet-18 as teacher, adapted for CIFAR-10"""
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # Adapt first conv for smaller images (CIFAR-10 is 32x32)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # Remove maxpool for small images
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    return model.to(device)


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ==================== Distillation Loss ====================


class DistillationLoss(nn.Module):
    """
    Knowledge Distillation Loss

    Combines:
    - Hard target loss (cross-entropy with true labels)
    - Soft target loss (KL divergence with teacher's soft predictions)

    Args:
        temperature: Higher temperature produces softer probability distributions
        alpha: Weight for hard target loss (1-alpha for soft target loss)
    """

    # def __init__(self, temperature=4.0, alpha=0.7):
    def __init__(self, temperature=4.0, alpha=0.4):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_logits, teacher_logits, labels):
        # Hard target loss
        hard_loss = self.ce_loss(student_logits, labels)

        # Soft target loss (distillation)
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        soft_loss = self.kl_loss(soft_student, soft_targets) * (self.temperature**2)

        # Combined loss
        total_loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss
        return total_loss, hard_loss, soft_loss


# ==================== Training Functions ====================


def train_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    teacher_model=None,
    distillation_loss=None,
):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Training", leave=False)
    for data, target in pbar:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)

        if teacher_model is not None and distillation_loss is not None:
            # Distillation training
            with torch.no_grad():
                teacher_output = teacher_model(data)
            loss, hard_loss, soft_loss = distillation_loss(
                output, teacher_output, target
            )
        else:
            # Standard training
            loss = criterion(output, target)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        pbar.set_postfix(
            {"loss": f"{loss.item():.4f}", "acc": f"{100.0 * correct / total:.2f}%"}
        )

    return total_loss / len(loader), 100.0 * correct / total


def evaluate(model, loader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    return total_loss / len(loader), 100.0 * correct / total


def train_model(
    model,
    train_loader,
    test_loader,
    epochs,
    device,
    teacher_model=None,
    use_distillation=False,
    lr=0.01,
    temperature=4.0,
):
    """Train a model with or without distillation"""

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    distillation_loss = (
        DistillationLoss(temperature=temperature) if use_distillation else None
    )

    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "epoch_time": [],
    }

    for epoch in range(epochs):
        start_time = time.time()

        # Train
        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            teacher_model,
            distillation_loss,
        )

        # Evaluate
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        scheduler.step()

        epoch_time = time.time() - start_time

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["epoch_time"].append(epoch_time)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}% | "
            f"Time: {epoch_time:.1f}s"
        )

    return history


# ==================== Main Experiment ====================


def run_experiment(epochs=15, batch_size=128, data_dir="./data", temperature=4.0):
    """Run the full distillation experiment"""

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}\n")

    # Data transforms
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )

    # Load data
    print("Loading CIFAR-10 dataset...")
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    # ==================== Teacher Model ====================
    print("\n" + "=" * 60)
    print("TEACHER MODEL (ResNet-18)")
    print("=" * 60)

    teacher = get_teacher_model(device=device)
    teacher_params = count_parameters(teacher)
    print(f"Parameters: {teacher_params:,}")

    # Train teacher (or you can load pre-trained weights)
    print("\nTraining Teacher Model...")
    teacher_history = train_model(
        teacher, train_loader, test_loader, epochs=epochs, device=device, lr=0.01
    )

    # ==================== Student with Distillation ====================
    print("\n" + "=" * 60)
    print("STUDENT MODEL WITH DISTILLATION")
    print("=" * 60)

    student_distill = SmallStudent().to(device)
    student_distill_params = count_parameters(student_distill)
    print(f"Parameters: {student_distill_params:,}")
    print(
        f"Compression ratio: {teacher_params / student_distill_params:.1f}x smaller than teacher"
    )

    print("\nTraining Student with Distillation...")
    student_distill_history = train_model(
        student_distill,
        train_loader,
        test_loader,
        epochs=epochs,
        device=device,
        teacher_model=teacher,
        use_distillation=True,
        lr=0.05,
        temperature=temperature,
    )

    # ==================== Student without Distillation (Baseline) ====================
    print("\n" + "=" * 60)
    print("STUDENT MODEL WITHOUT DISTILLATION (Baseline)")
    print("=" * 60)

    student_baseline = SmallStudent().to(device)
    print(f"Parameters: {student_distill_params:,}")

    print("\nTraining Student without Distillation...")
    student_baseline_history = train_model(
        student_baseline,
        train_loader,
        test_loader,
        epochs=epochs,
        device=device,
        lr=0.05,
    )

    # ==================== Results Summary ====================
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    results = {
        "Teacher (ResNet-18)": {
            "params": teacher_params,
            "test_acc": teacher_history["test_acc"][-1],
            "total_time": sum(teacher_history["epoch_time"]),
        },
        "Student + Distillation": {
            "params": student_distill_params,
            "test_acc": student_distill_history["test_acc"][-1],
            "total_time": sum(student_distill_history["epoch_time"]),
        },
        "Student (Baseline)": {
            "params": student_distill_params,
            "test_acc": student_baseline_history["test_acc"][-1],
            "total_time": sum(student_baseline_history["epoch_time"]),
        },
    }

    print(f"\n{'Model':<25} {'Parameters':>15} {'Test Acc':>12} {'Total Time':>12}")
    print("-" * 65)
    for name, metrics in results.items():
        print(
            f"{name:<25} {metrics['params']:>15,} {metrics['test_acc']:>11.2f}% {metrics['total_time']:>11.1f}s"
        )

    # Distillation benefit
    distill_benefit = (
        student_distill_history["test_acc"][-1]
        - student_baseline_history["test_acc"][-1]
    )
    print(f"\n*** Distillation Benefit: +{distill_benefit:.2f}% accuracy ***")

    # Plot results
    plot_results(
        teacher_history, student_distill_history, student_baseline_history, results
    )

    return results


def plot_results(
    teacher_history, student_distill_history, student_baseline_history, results
):
    """Plot training curves and comparison"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Test Accuracy
    ax1 = axes[0, 0]
    epochs = range(1, len(teacher_history["test_acc"]) + 1)
    ax1.plot(
        epochs,
        teacher_history["test_acc"],
        "b-",
        label="Teacher (ResNet-18)",
        linewidth=2,
    )
    ax1.plot(
        epochs,
        student_distill_history["test_acc"],
        "g-",
        label="Student + Distillation",
        linewidth=2,
    )
    ax1.plot(
        epochs,
        student_baseline_history["test_acc"],
        "r--",
        label="Student (Baseline)",
        linewidth=2,
    )
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Test Accuracy (%)")
    ax1.set_title("Test Accuracy Comparison")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Training Loss
    ax2 = axes[0, 1]
    ax2.plot(epochs, teacher_history["train_loss"], "b-", label="Teacher", linewidth=2)
    ax2.plot(
        epochs,
        student_distill_history["train_loss"],
        "g-",
        label="Student + Distillation",
        linewidth=2,
    )
    ax2.plot(
        epochs,
        student_baseline_history["train_loss"],
        "r--",
        label="Student (Baseline)",
        linewidth=2,
    )
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Training Loss")
    ax2.set_title("Training Loss Comparison")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Model Size vs Accuracy
    ax3 = axes[1, 0]
    models_names = list(results.keys())
    params = [results[m]["params"] / 1e6 for m in models_names]  # In millions
    accs = [results[m]["test_acc"] for m in models_names]
    colors = ["blue", "green", "red"]

    bars = ax3.bar(range(len(models_names)), accs, color=colors, alpha=0.7)
    ax3.set_xticks(range(len(models_names)))
    ax3.set_xticklabels(
        ["Teacher\n(ResNet-18)", "Student\n+ Distillation", "Student\n(Baseline)"]
    )
    ax3.set_ylabel("Test Accuracy (%)")
    ax3.set_title("Final Test Accuracy Comparison")

    # Add parameter count labels
    for i, (bar, p) in enumerate(zip(bars, params)):
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{p:.2f}M params",
            ha="center",
            fontsize=9,
        )

    ax3.grid(True, alpha=0.3, axis="y")

    # Plot 4: Summary Table
    ax4 = axes[1, 1]
    ax4.axis("off")

    # Create summary text
    distill_acc = results["Student + Distillation"]["test_acc"]
    baseline_acc = results["Student (Baseline)"]["test_acc"]
    teacher_acc = results["Teacher (ResNet-18)"]["test_acc"]

    summary_text = f"""
    KNOWLEDGE DISTILLATION BENEFITS
    ═════════════════════════════════════

    Teacher Model (ResNet-18):
      • Parameters: {results["Teacher (ResNet-18)"]["params"]:,}
      • Test Accuracy: {teacher_acc:.2f}%

    Student with Distillation:
      • Parameters: {results["Student + Distillation"]["params"]:,}
      • Test Accuracy: {distill_acc:.2f}%
      • Compression: {results["Teacher (ResNet-18)"]["params"] / results["Student + Distillation"]["params"]:.1f}x

    Student Baseline (no distillation):
      • Parameters: {results["Student (Baseline)"]["params"]:,}
      • Test Accuracy: {baseline_acc:.2f}%

    ═════════════════════════════════════
    KEY FINDINGS:

    ✦ Distillation improves accuracy by +{distill_acc - baseline_acc:.2f}%

    ✦ Student + Distillation achieves {distill_acc / teacher_acc * 100:.1f}% of
      teacher's accuracy with {results["Teacher (ResNet-18)"]["params"] / results["Student + Distillation"]["params"]:.1f}x fewer parameters

    ✦ Training time reduced by {sum(teacher_history["epoch_time"]) / sum(student_distill_history["epoch_time"]):.1f}x
    """

    ax4.text(
        0.1,
        0.95,
        summary_text,
        transform=ax4.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.3),
    )

    plt.tight_layout()
    plt.savefig("distillation_results.png", dpi=150, bbox_inches="tight")
    print(f"\nResults saved to 'distillation_results.png'")
    plt.show()


if __name__ == "__main__":
    # Run the experiment with default parameters
    results = run_experiment(
        # epochs=15,  # Number of training epochs
        epochs=30,  # Number of training epochs
        batch_size=256,  # Batch size
        data_dir="./data",  # Data directory
        temperature=4.0,  # Distillation temperature
    )
