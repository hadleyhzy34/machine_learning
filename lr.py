"""
PyTorch Learning Rate Scheduling Comparison Demo
Author: Deep Learning Interview Assistant
Date: 2026-02-17
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import time
from typing import List, Dict, Tuple
import warnings

warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


# Device configuration
def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    return device


# Check for GPU availability
# ipdb.set_trace()
device = get_device()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ==================== 1. Dataset & Model Setup ====================
def load_cifar10(batch_size=128):
    """Load and prepare CIFAR-10 dataset"""
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    train_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    return train_loader, test_loader


class SimpleCNN(nn.Module):
    """Simple CNN for CIFAR-10 classification"""

    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.25)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 128 * 4 * 4)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


# ==================== 2. Training Function with Scheduler Tracking ====================
def train_model(
    model,
    train_loader,
    test_loader,
    optimizer,
    scheduler,
    num_epochs=50,
    scheduler_name="Fixed",
    warmup_epochs=0,
):
    """Train model with given scheduler and track metrics"""
    criterion = nn.CrossEntropyLoss()

    # Track metrics
    train_losses = []
    test_accuracies = []
    learning_rates = []

    # Warmup scheduler (if needed)
    if warmup_epochs > 0:
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
        )

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        # Training phase
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Apply warmup scheduler before main scheduler
        if warmup_epochs > 0 and epoch < warmup_epochs:
            warmup_scheduler.step()
        elif scheduler is not None:
            # For ReduceLROnPlateau, need validation loss
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for inputs, targets in test_loader:
                        inputs, targets = inputs.to(device), targets.to(device)
                        outputs = model(inputs)
                        val_loss += criterion(outputs, targets).item()
                val_loss /= len(test_loader)
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Track learning rate
        current_lr = optimizer.param_groups[0]["lr"]
        learning_rates.append(current_lr)

        # Calculate average training loss
        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)

        # Test accuracy
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        accuracy = 100.0 * correct / total
        test_accuracies.append(accuracy)

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, "
                f"Test Acc: {accuracy:.2f}%, LR: {current_lr:.6f}"
            )

    return {
        "train_losses": train_losses,
        "test_accuracies": test_accuracies,
        "learning_rates": learning_rates,
        "final_accuracy": test_accuracies[-1],
    }


# ==================== 3. Scheduler Implementations ====================
def get_schedulers(optimizer, num_epochs=50):
    """Create dictionary of different scheduler configurations"""

    schedulers = {}

    # 1. Fixed LR (Baseline)
    schedulers["Fixed"] = None

    # 2. Step Decay
    schedulers["Step Decay"] = optim.lr_scheduler.StepLR(
        optimizer, step_size=15, gamma=0.1
    )

    # 3. Step Decay with Warmup (5 epochs)
    schedulers["Step Decay + Warmup"] = optim.lr_scheduler.StepLR(
        optimizer, step_size=15, gamma=0.1
    )

    # 4. Exponential Decay
    schedulers["Exponential Decay"] = optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=0.95
    )

    # 5. Cosine Annealing
    schedulers["Cosine Annealing"] = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )

    # 6. Cosine Annealing with Warm Restarts (Cyclical)
    schedulers["Cosine Warm Restarts"] = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    # 7. ReduceLROnPlateau
    schedulers["ReduceLROnPlateau"] = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, verbose=False
    )

    # 8. One-Cycle Policy
    schedulers["One-Cycle"] = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.1,
        total_steps=num_epochs * len(train_loader),
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy="cos",
    )

    return schedulers


# ==================== 4. Main Comparison Function ====================
def compare_schedulers(num_epochs=30):
    """Compare all scheduling strategies"""
    print("\n" + "=" * 60)
    print("COMPARING LEARNING RATE SCHEDULING STRATEGIES")
    print("=" * 60)

    results = {}

    # Load data
    global train_loader, test_loader
    train_loader, test_loader = load_cifar10(batch_size=128)

    # Test each scheduler
    schedulers = [
        ("Fixed (Baseline)", None, 0),
        ("Step Decay", "step", 0),
        ("Step Decay + Warmup", "step", 5),
        ("Exponential Decay", "exp", 0),
        ("Cosine Annealing", "cosine", 0),
        ("Cosine Warm Restarts", "cosine_warm", 0),
        ("ReduceLROnPlateau", "plateau", 0),
        ("One-Cycle Policy", "onecycle", 0),
    ]

    for name, sched_type, warmup in schedulers:
        print(f"\n\n{'=' * 50}")
        print(f"Training with: {name}")
        print(f"{'=' * 50}")

        # Initialize fresh model and optimizer
        model = SimpleCNN().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Create scheduler based on type
        scheduler = None
        if sched_type == "step":
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        elif sched_type == "exp":
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        elif sched_type == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=num_epochs, eta_min=1e-6
            )
        elif sched_type == "cosine_warm":
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=10, T_mult=2, eta_min=1e-6
            )
        elif sched_type == "plateau":
            # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            #     optimizer, mode="max", factor=0.5, patience=5, verbose=False
            # )
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, patience=5
            )
        elif sched_type == "onecycle":
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=0.01,
                total_steps=num_epochs * len(train_loader),
                epochs=num_epochs,
                steps_per_epoch=len(train_loader),
                pct_start=0.3,
                anneal_strategy="cos",
            )

        # Train and track
        start_time = time.time()
        metrics = train_model(
            model,
            train_loader,
            test_loader,
            optimizer,
            scheduler,
            num_epochs=num_epochs,
            scheduler_name=name,
            warmup_epochs=warmup,
        )
        training_time = time.time() - start_time

        results[name] = {**metrics, "training_time": training_time}

    return results


# ==================== 5. Hyperparameter Tuning Demo ====================
def hyperparameter_tuning_demo():
    """Demonstrate hyperparameter tuning for one scheduler"""
    print("\n" + "=" * 60)
    print("HYPERPARAMETER TUNING DEMO: Step Decay")
    print("=" * 60)

    train_loader, test_loader = load_cifar10(batch_size=128)

    # Test different step sizes
    step_sizes = [5, 10, 15, 20]
    gammas = [0.1, 0.2, 0.5, 0.8]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for idx, (step_size, gamma) in enumerate(zip(step_sizes, gammas)):
        model = SimpleCNN().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )

        metrics = train_model(
            model,
            train_loader,
            test_loader,
            optimizer,
            scheduler,
            num_epochs=30,
            scheduler_name=f"Step{step_size}_Gamma{gamma}",
        )

        ax = axes[idx]
        ax.plot(metrics["learning_rates"], label="LR")
        ax.set_title(f"Step Size: {step_size}, Gamma: {gamma}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate")
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig("hyperparameter_tuning.png", dpi=150, bbox_inches="tight")
    plt.show()


# ==================== 6. Visualization Functions ====================
def visualize_results(results: Dict):
    """Create comprehensive visualizations of scheduler comparison"""

    fig = plt.figure(figsize=(18, 12))

    # 1. Learning Rate Schedules
    ax1 = plt.subplot(2, 3, 1)
    for name, data in results.items():
        ax1.plot(data["learning_rates"], label=name, linewidth=2)
    ax1.set_title("Learning Rate Schedules Over Epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Learning Rate (log scale)")
    ax1.set_yscale("log")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # 2. Test Accuracy Comparison
    ax2 = plt.subplot(2, 3, 2)
    for name, data in results.items():
        ax2.plot(
            data["test_accuracies"],
            label=f"{name} ({data['final_accuracy']:.2f}%)",
            linewidth=2,
        )
    ax2.set_title("Test Accuracy Comparison")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # 3. Training Loss Comparison
    ax3 = plt.subplot(2, 3, 3)
    for name, data in results.items():
        ax3.plot(data["train_losses"], label=name, linewidth=2, alpha=0.7)
    ax3.set_title("Training Loss Comparison")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Loss")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # 4. Final Accuracy Bar Chart
    ax4 = plt.subplot(2, 3, 4)
    names = list(results.keys())
    final_accs = [results[name]["final_accuracy"] for name in names]
    colors = plt.cm.Set3(np.linspace(0, 1, len(names)))
    bars = ax4.barh(names, final_accs, color=colors)
    ax4.set_title("Final Test Accuracy")
    ax4.set_xlabel("Accuracy (%)")

    # Add value labels
    for bar, acc in zip(bars, final_accs):
        width = bar.get_width()
        ax4.text(
            width + 0.1,
            bar.get_y() + bar.get_height() / 2,
            f"{acc:.2f}%",
            ha="left",
            va="center",
        )

    # 5. Performance Summary Table
    ax5 = plt.subplot(2, 3, 5)
    ax5.axis("tight")
    ax5.axis("off")

    table_data = []
    for name, data in results.items():
        table_data.append(
            [
                name,
                f"{data['final_accuracy']:.2f}%",
                f"{min(data['train_losses']):.4f}",
                f"{data['training_time']:.1f}s",
            ]
        )

    table = ax5.table(
        cellText=table_data,
        colLabels=["Scheduler", "Final Acc", "Min Loss", "Time"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    ax5.set_title("Performance Summary")

    # 6. Learning Rate Distribution
    ax6 = plt.subplot(2, 3, 6)
    lr_data = []
    labels = []
    for name, data in results.items():
        lr_data.append(data["learning_rates"])
        labels.append(name)

    box = ax6.boxplot(lr_data, labels=labels, patch_artist=True)
    colors = plt.cm.Pastel1(np.linspace(0, 1, len(labels)))
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)

    ax6.set_title("Learning Rate Distribution")
    ax6.set_ylabel("Learning Rate")
    ax6.set_yscale("log")
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig("lr_scheduler_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()


# ==================== 7. Custom Learning Rate Schedulers ====================
class CustomWarmupScheduler:
    """Custom scheduler with warmup + cosine annealing"""

    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr=1e-3):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.current_epoch = 0

    def step(self):
        self.current_epoch += 1

        if self.current_epoch <= self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (self.current_epoch / self.warmup_epochs)
        else:
            # Cosine annealing after warmup
            progress = (self.current_epoch - self.warmup_epochs) / (
                self.total_epochs - self.warmup_epochs
            )
            lr = self.base_lr * 0.5 * (1 + np.cos(np.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr


class CyclicalScheduler:
    """Cyclical learning rate scheduler"""

    def __init__(self, optimizer, base_lr=1e-4, max_lr=1e-2, step_size=10):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.current_step = 0

    def step(self):
        self.current_step += 1
        cycle = np.floor(1 + self.current_step / (2 * self.step_size))
        x = np.abs(self.current_step / self.step_size - 2 * cycle + 1)
        lr = self.base_lr + (self.max_lr - self.base_lr) * max(0, 1 - x)

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr


# ==================== 8. Run the Complete Demo ====================
if __name__ == "__main__":
    print("Deep Learning Learning Rate Scheduling Demo")
    print("=" * 60)

    # Set training parameters
    NUM_EPOCHS = 30

    # Run main comparison
    results = compare_schedulers(num_epochs=NUM_EPOCHS)

    # Visualize results
    print("\n\nGenerating visualizations...")
    visualize_results(results)

    # Run hyperparameter tuning demo
    hyperparameter_tuning_demo()

    # Demo custom schedulers
    print("\n" + "=" * 60)
    print("CUSTOM SCHEDULER DEMONSTRATION")
    print("=" * 60)

    train_loader, test_loader = load_cifar10(batch_size=128)

    # Test custom warmup + cosine scheduler
    print("\nTesting Custom Warmup + Cosine Scheduler...")
    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    custom_scheduler = CustomWarmupScheduler(
        optimizer, warmup_epochs=5, total_epochs=20, base_lr=0.001
    )

    # Track learning rates
    custom_lrs = []
    for epoch in range(20):
        custom_scheduler.step()
        custom_lrs.append(optimizer.param_groups[0]["lr"])

    # Test cyclical scheduler
    print("\nTesting Cyclical Learning Rate Scheduler...")
    model2 = SimpleCNN().to(device)
    optimizer2 = optim.Adam(model2.parameters(), lr=0.001)
    cyclical_scheduler = CyclicalScheduler(
        optimizer2, base_lr=1e-4, max_lr=1e-2, step_size=5
    )

    cyclical_lrs = []
    for epoch in range(20):
        cyclical_scheduler.step()
        cyclical_lrs.append(optimizer2.param_groups[0]["lr"])

    # Plot custom schedulers
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(custom_lrs, "b-", linewidth=2)
    axes[0].set_title("Custom Warmup + Cosine Scheduler")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Learning Rate")
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(x=5, color="r", linestyle="--", alpha=0.5, label="Warmup End")
    axes[0].legend()

    axes[1].plot(cyclical_lrs, "g-", linewidth=2)
    axes[1].set_title("Cyclical Learning Rate Scheduler")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Learning Rate")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("custom_schedulers.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("\n" + "=" * 60)
    print("KEY INSIGHTS FROM THE DEMO:")
    print("=" * 60)
    print("""
    1. Fixed LR (Baseline): 
       - Simple but often suboptimal
       - May converge slowly or overshoot
    
    2. Step Decay:
       - Sharp drops at specific epochs
       - Works well when you know optimal drop points
       - + Warmup helps stabilize early training
    
    3. Exponential Decay:
       - Smooth, continuous decay
       - Good for gradual fine-tuning
    
    4. Cosine Annealing:
       - Smooth decay following cosine curve
       - Often leads to better convergence
       - Warm Restarts add cyclical behavior
    
    5. ReduceLROnPlateau:
       - Adaptive based on validation performance
       - May delay learning rate reduction
    
    6. One-Cycle Policy:
       - Fastest convergence in many cases
       - Combines increasing and decreasing phases
       - Requires careful hyperparameter tuning
    
    Best Practices:
    - Always use validation set to monitor performance
    - Start with Cosine Annealing or One-Cycle for modern architectures
    - Add warmup (3-5 epochs) when using large batch sizes
    - Use ReduceLROnPlateau when training dynamics are unpredictable
    """)
