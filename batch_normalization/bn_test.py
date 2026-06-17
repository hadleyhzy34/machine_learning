import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time
import seaborn as sns
from sklearn.metrics import confusion_matrix
import ipdb

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


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

# ====================== Data Loading ======================
print("\n" + "=" * 50)
print("Loading CIFAR-10 Dataset")
print("=" * 50)

# Data transformations
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

# Load CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)

# Create data loaders
batch_size = 512
train_loader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=2
)
test_loader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False, num_workers=2
)

# Class names
classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

print(f"Training samples: {len(trainset)}")
print(f"Test samples: {len(testset)}")
print(f"Batch size: {batch_size}")


# ====================== Model Definitions ======================
class CNNWithoutBN(nn.Module):
    """CNN without Batch Normalization"""

    def __init__(self):
        super(CNNWithoutBN, self).__init__()
        self.conv_layers = nn.Sequential(
            # Layer 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Layer 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Layer 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


class CNNWithBN(nn.Module):
    """CNN with Batch Normalization"""

    def __init__(self):
        super(CNNWithBN, self).__init__()
        self.conv_layers = nn.Sequential(
            # Layer 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Layer 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Layer 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


# ====================== Training Function ======================
def train_model(model, train_loader, test_loader, model_name, lr=0.001, epochs=20):
    """Train a model and return training metrics"""
    model = model.to(device)

    # Use the same optimizer settings for fair comparison
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Track metrics
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    learning_rates = []

    print(f"\n{'=' * 50}")
    print(f"Training {model_name}")
    print(f"{'=' * 50}")

    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} - Training")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()

            pbar.set_postfix({"loss": loss.item()})

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(test_loader)
        val_acc = 100 * correct / total

        # Record metrics
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(val_loss)
        test_accs.append(val_acc)
        learning_rates.append(optimizer.param_groups[0]["lr"])

        print(
            f"Epoch {epoch + 1}/{epochs}: "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
        )

    return {
        "train_losses": train_losses,
        "train_accs": train_accs,
        "test_losses": test_losses,
        "test_accs": test_accs,
        "learning_rates": learning_rates,
        "model": model,
    }


# ====================== Evaluate Function ======================
def evaluate_model(model, test_loader, model_name):
    """Comprehensive evaluation of a trained model"""
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc=f"Evaluating {model_name}"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total

    # Calculate per-class accuracy
    class_correct = [0] * 10
    class_total = [0] * 10
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += (predicted[i] == label).item()
                class_total[label] += 1

    class_accuracies = [100 * class_correct[i] / class_total[i] for i in range(10)]

    return {
        "accuracy": accuracy,
        "all_preds": all_preds,
        "all_labels": all_labels,
        "class_accuracies": class_accuracies,
    }


# ====================== Visualization Functions ======================
def plot_training_comparison(metrics_bn, metrics_no_bn, epochs):
    """Plot comparison of training metrics"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Plot training loss
    axes[0, 0].plot(
        metrics_no_bn["train_losses"], "b-", label="Without BN", linewidth=2
    )
    axes[0, 0].plot(metrics_bn["train_losses"], "r-", label="With BN", linewidth=2)
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Training Loss")
    axes[0, 0].set_title("Training Loss Comparison")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot training accuracy
    axes[0, 1].plot(metrics_no_bn["train_accs"], "b-", label="Without BN", linewidth=2)
    axes[0, 1].plot(metrics_bn["train_accs"], "r-", label="With BN", linewidth=2)
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Training Accuracy (%)")
    axes[0, 1].set_title("Training Accuracy Comparison")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot test loss
    axes[0, 2].plot(metrics_no_bn["test_losses"], "b-", label="Without BN", linewidth=2)
    axes[0, 2].plot(metrics_bn["test_losses"], "r-", label="With BN", linewidth=2)
    axes[0, 2].set_xlabel("Epoch")
    axes[0, 2].set_ylabel("Test Loss")
    axes[0, 2].set_title("Test Loss Comparison")
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Plot test accuracy
    axes[1, 0].plot(metrics_no_bn["test_accs"], "b-", label="Without BN", linewidth=2)
    axes[1, 0].plot(metrics_bn["test_accs"], "r-", label="With BN", linewidth=2)
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Test Accuracy (%)")
    axes[1, 0].set_title("Test Accuracy Comparison")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot class-wise comparison
    x = np.arange(10)
    width = 0.35
    axes[1, 1].bar(
        x - width / 2, eval_no_bn["class_accuracies"], width, label="Without BN"
    )
    axes[1, 1].bar(x + width / 2, eval_bn["class_accuracies"], width, label="With BN")
    axes[1, 1].set_xlabel("Class")
    axes[1, 1].set_ylabel("Accuracy (%)")
    axes[1, 1].set_title("Per-Class Accuracy Comparison")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(classes, rotation=45)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis="y")

    # Plot confusion matrix for model with BN
    cm = confusion_matrix(eval_bn["all_labels"], eval_bn["all_preds"])
    im = axes[1, 2].imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    axes[1, 2].set_title("Confusion Matrix (With BN)")
    axes[1, 2].set_xlabel("Predicted")
    axes[1, 2].set_ylabel("True")
    axes[1, 2].set_xticks(range(10))
    axes[1, 2].set_yticks(range(10))
    axes[1, 2].set_xticklabels(classes, rotation=45)
    axes[1, 2].set_yticklabels(classes)

    # Add text annotations
    thresh = cm.max() / 2
    for i in range(10):
        for j in range(10):
            axes[1, 2].text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()
    plt.savefig("batch_norm_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_gradient_flow_comparison(model_bn, model_no_bn, test_loader):
    """Compare gradient flow in both models"""
    # Get a batch of data
    data_iter = iter(test_loader)
    inputs, labels = next(data_iter)
    inputs, labels = inputs.to(device), labels.to(device)

    # Set models to training mode
    model_bn.train()
    model_no_bn.train()

    # Forward and backward pass for both models
    criterion = nn.CrossEntropyLoss()

    # Model with BN
    outputs_bn = model_bn(inputs)
    loss_bn = criterion(outputs_bn, labels)
    loss_bn.backward()

    # Model without BN
    outputs_no_bn = model_no_bn(inputs)
    loss_no_bn = criterion(outputs_no_bn, labels)
    loss_no_bn.backward()

    # Collect gradient statistics
    grad_stats_bn = []
    grad_stats_no_bn = []
    layer_names = []

    # Get gradients for each layer
    for name, param in model_bn.named_parameters():
        if param.grad is not None and "weight" in name:
            grad_mean = param.grad.abs().mean().item()
            grad_std = param.grad.abs().std().item()
            grad_stats_bn.append((grad_mean, grad_std))
            layer_names.append(
                name.split(".")[-2] if len(name.split(".")) > 2 else "fc"
            )

    for name, param in model_no_bn.named_parameters():
        if param.grad is not None and "weight" in name:
            grad_mean = param.grad.abs().mean().item()
            grad_std = param.grad.abs().std().item()
            grad_stats_no_bn.append((grad_mean, grad_std))

    # Plot gradient statistics
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    x = range(len(grad_stats_bn))
    bn_means = [s[0] for s in grad_stats_bn]
    no_bn_means = [s[0] for s in grad_stats_no_bn]

    axes[0].bar(x, bn_means, alpha=0.7, label="With BN")
    axes[0].bar(x, no_bn_means, alpha=0.7, label="Without BN")
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Mean Gradient Magnitude")
    axes[0].set_title("Gradient Flow Comparison")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(layer_names, rotation=45)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis="y")

    # Plot gradient ratios
    ratios = [
        bn_means[i] / no_bn_means[i] if no_bn_means[i] > 0 else 0
        for i in range(len(bn_means))
    ]
    axes[1].plot(x, ratios, "o-", linewidth=2, markersize=8)
    axes[1].axhline(y=1, color="r", linestyle="--", alpha=0.5, label="Equal gradients")
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Gradient Ratio (BN / No BN)")
    axes[1].set_title("Gradient Preservation with BN")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(layer_names, rotation=45)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("gradient_flow_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()


# ====================== Main Training and Evaluation ======================
if __name__ == "__main__":
    # Initialize models
    model_without_bn = CNNWithoutBN()
    model_with_bn = CNNWithBN()

    # Print model architectures
    print("\n" + "=" * 50)
    print("Model Architectures")
    print("=" * 50)
    print("\nModel WITHOUT BatchNorm:")
    print(model_without_bn)

    print("\nModel WITH BatchNorm:")
    print(model_with_bn)

    # Count parameters
    params_no_bn = sum(p.numel() for p in model_without_bn.parameters())
    params_bn = sum(p.numel() for p in model_with_bn.parameters())
    print(f"\nParameters - Without BN: {params_no_bn:,}")
    print(f"Parameters - With BN: {params_bn:,}")
    print(f"Additional parameters from BN: {params_bn - params_no_bn:,}")

    # Train both models
    epochs = 20

    # Train model without BN
    start_time = time.time()
    metrics_no_bn = train_model(
        model_without_bn,
        train_loader,
        test_loader,
        "Model WITHOUT BatchNorm",
        lr=0.001,
        epochs=epochs,
    )
    time_no_bn = time.time() - start_time

    # Train model with BN
    start_time = time.time()
    metrics_bn = train_model(
        model_with_bn,
        train_loader,
        test_loader,
        "Model WITH BatchNorm",
        lr=0.001,
        epochs=epochs,
    )
    time_bn = time.time() - start_time

    # Evaluate both models
    eval_no_bn = evaluate_model(
        metrics_no_bn["model"], test_loader, "Model WITHOUT BatchNorm"
    )
    eval_bn = evaluate_model(metrics_bn["model"], test_loader, "Model WITH BatchNorm")

    # Print results summary
    print("\n" + "=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)
    print(f"\nTraining Time:")
    print(f"  Without BN: {time_no_bn:.2f} seconds")
    print(f"  With BN:    {time_bn:.2f} seconds")
    print(f"  Difference: {time_bn - time_no_bn:.2f} seconds")

    print(f"\nFinal Test Accuracy:")
    print(f"  Without BN: {eval_no_bn['accuracy']:.2f}%")
    print(f"  With BN:    {eval_bn['accuracy']:.2f}%")
    print(f"  Improvement: {eval_bn['accuracy'] - eval_no_bn['accuracy']:.2f}%")

    print(f"\nTraining Convergence:")
    print(
        f"  Best accuracy without BN: {max(metrics_no_bn['test_accs']):.2f}% at epoch {metrics_no_bn['test_accs'].index(max(metrics_no_bn['test_accs'])) + 1}"
    )
    print(
        f"  Best accuracy with BN:    {max(metrics_bn['test_accs']):.2f}% at epoch {metrics_bn['test_accs'].index(max(metrics_bn['test_accs'])) + 1}"
    )

    print(f"\nTraining Stability:")
    print(f"  Final loss without BN: {metrics_no_bn['train_losses'][-1]:.4f}")
    print(f"  Final loss with BN:    {metrics_bn['train_losses'][-1]:.4f}")

    # Plot results
    print("\n" + "=" * 50)
    print("Generating Visualizations")
    print("=" * 50)

    plot_training_comparison(metrics_bn, metrics_no_bn, epochs)
    plot_gradient_flow_comparison(
        metrics_bn["model"], metrics_no_bn["model"], test_loader
    )

    # Print key insights
    print("\n" + "=" * 50)
    print("KEY INSIGHTS FROM BATCH NORMALIZATION")
    print("=" * 50)
    print("\n1. FASTER CONVERGENCE:")
    print("   - BatchNorm typically reaches similar accuracy in fewer epochs")
    print("   - Allows for higher learning rates without training instability")

    print("\n2. IMPROVED GRADIENT FLOW:")
    print("   - Prevents vanishing/exploding gradients in deep networks")
    print("   - Maintains more stable gradient magnitudes across layers")

    print("\n3. BETTER GENERALIZATION:")
    print("   - Often achieves higher test accuracy")
    print("   - Acts as a regularizer, reducing overfitting")

    print("\n4. REDUCED SENSITIVITY TO INITIALIZATION:")
    print("   - Makes training less dependent on careful weight initialization")
    print("   - More robust to different learning rates")

    print("\n5. SLIGHT COMPUTATIONAL OVERHEAD:")
    print(f"   - Additional parameters: {params_bn - params_no_bn:,}")
    print(f"   - Slightly longer training time: {time_bn - time_no_bn:.2f} seconds")
    print("   - This is usually worth the improved performance")
