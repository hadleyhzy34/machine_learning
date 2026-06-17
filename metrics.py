import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
)
import ipdb
import seaborn as sns
from collections import Counter

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ==================== 1. Create Imbalanced Dataset ====================
print("=" * 60)
print("Creating Imbalanced Dataset")
print("=" * 60)

# Create synthetic imbalanced dataset (90% negative, 10% positive)
X, y = make_classification(
    n_samples=10000,
    n_features=20,
    n_informative=8,
    n_redundant=2,
    n_clusters_per_class=2,
    weights=[0.9, 0.1],  # 90% class 0, 10% class 1
    flip_y=0.05,
    random_state=42,
)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_test, y_test, test_size=0.5, stratify=y_test, random_state=42
)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train)
X_val_tensor = torch.FloatTensor(X_val)
y_val_tensor = torch.LongTensor(y_val)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.LongTensor(y_test)

print(f"Train set: {Counter(y_train)}")
print(f"Validation set: {Counter(y_val)}")
print(f"Test set: {Counter(y_test)}")
print(f"Class distribution: {100 * sum(y_train) / len(y_train):.1f}% positive")
print()


# ==================== 2. Create PyTorch Dataset ====================
class ImbalancedDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# Create datasets
train_dataset = ImbalancedDataset(X_train_tensor, y_train_tensor)
val_dataset = ImbalancedDataset(X_val_tensor, y_val_tensor)
test_dataset = ImbalancedDataset(X_test_tensor, y_test_tensor)


# ==================== 3. Handle Class Imbalance ====================
# Method 1: Weighted Random Sampler
def get_weighted_sampler(labels):
    class_counts = Counter(labels.numpy())
    total_samples = len(labels)
    class_weights = {cls: total_samples / count for cls, count in class_counts.items()}

    sample_weights = [class_weights[label.item()] for label in labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )
    return sampler


# Method 2: Class weights for loss function
def get_class_weights(labels):
    class_counts = Counter(labels.numpy())
    total = sum(class_counts.values())
    weights = {cls: total / count for cls, count in class_counts.items()}
    weight_tensor = torch.tensor([weights[0], weights[1]], dtype=torch.float32)
    return weight_tensor


# Create data loaders
batch_size = 64
train_sampler = get_weighted_sampler(y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# ==================== 4. Define Neural Network ====================
class ImbalancedClassifier(nn.Module):
    def __init__(self, input_size):
        super(ImbalancedClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 2),  # 2 classes
        )

    def forward(self, x):
        return self.network(x)


# ==================== 5. Training Function ====================
def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs=20,
    use_class_weights=True,
    learning_rate=0.001,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Use class weights in loss if specified
    if use_class_weights:
        class_weights = get_class_weights(y_train_tensor).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss, correct, total = 0, 0, 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

        train_losses.append(train_loss / len(train_loader))
        train_accs.append(100 * correct / total)

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()

        val_losses.append(val_loss / len(val_loader))
        val_accs.append(100 * val_correct / val_total)

        if (epoch + 1) % 5 == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], "
                f"Train Loss: {train_losses[-1]:.4f}, "
                f"Val Loss: {val_losses[-1]:.4f}, "
                f"Train Acc: {train_accs[-1]:.2f}%, "
                f"Val Acc: {val_accs[-1]:.2f}%"
            )

    return model, train_losses, val_losses, train_accs, val_accs


# ==================== 6. Evaluation Functions ====================
def evaluate_model(model, data_loader, threshold=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            outputs = model(batch_X)
            probs = torch.softmax(outputs, dim=1)

            all_probs.extend(probs[:, 1].cpu().numpy())  # Positive class probabilities
            all_preds.extend((probs[:, 1] > threshold).cpu().numpy().astype(int))
            all_labels.extend(batch_y.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def plot_metrics(y_true, y_pred, y_probs, title_suffix=""):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"Model Evaluation Metrics {title_suffix}", fontsize=16)

    ipdb.set_trace()

    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0, 0])
    axes[0, 0].set_title("Confusion Matrix")
    axes[0, 0].set_xlabel("Predicted")
    axes[0, 0].set_ylabel("Actual")

    # 2. ROC Curve
    # ipdb.set_trace()
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    axes[0, 1].plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})"
    )
    axes[0, 1].plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    axes[0, 1].set_xlim([0.0, 1.0])
    axes[0, 1].set_ylim([0.0, 1.05])
    axes[0, 1].set_xlabel("False Positive Rate")
    axes[0, 1].set_ylabel("True Positive Rate (Recall)")
    axes[0, 1].set_title("ROC Curve")
    axes[0, 1].legend(loc="lower right")

    # 3. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    avg_precision = average_precision_score(y_true, y_probs)

    axes[0, 2].plot(
        recall,
        precision,
        color="green",
        lw=2,
        label=f"Avg Precision = {avg_precision:.3f}",
    )
    axes[0, 2].set_xlim([0.0, 1.0])
    axes[0, 2].set_ylim([0.0, 1.05])
    axes[0, 2].set_xlabel("Recall")
    axes[0, 2].set_ylabel("Precision")
    axes[0, 2].set_title("Precision-Recall Curve")
    axes[0, 2].legend(loc="lower left")

    # 4. Calculate metrics for different thresholds
    thresholds = np.linspace(0, 1, 50)
    precisions = []
    recalls = []

    for thresh in thresholds:
        y_pred_thresh = (y_probs > thresh).astype(int)
        if sum(y_pred_thresh) > 0:  # Avoid division by zero
            precisions.append(precision_score(y_true, y_pred_thresh, zero_division=0))
        else:
            precisions.append(0)
        recalls.append(recall_score(y_true, y_pred_thresh, zero_division=0))

    axes[1, 0].plot(thresholds, precisions, "b-", label="Precision", alpha=0.7)
    axes[1, 0].plot(thresholds, recalls, "r-", label="Recall", alpha=0.7)
    axes[1, 0].set_xlabel("Threshold")
    axes[1, 0].set_ylabel("Score")
    axes[1, 0].set_title("Precision & Recall vs Threshold")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 5. Probability distributions
    axes[1, 1].hist(
        [y_probs[y_true == 0], y_probs[y_true == 1]],
        bins=30,
        alpha=0.7,
        label=["Negative", "Positive"],
        color=["blue", "red"],
    )
    axes[1, 1].axvline(x=0.5, color="black", linestyle="--", alpha=0.7)
    axes[1, 1].set_xlabel("Predicted Probability")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].set_title("Probability Distribution by Class")
    axes[1, 1].legend()

    # 6. Metrics Summary
    axes[1, 2].axis("off")
    metrics_text = (
        f"Metrics Summary:\n\n"
        f"Accuracy: {100 * np.mean(y_true == y_pred):.2f}%\n"
        f"Precision: {precision_score(y_true, y_pred, zero_division=0):.3f}\n"
        f"Recall: {recall_score(y_true, y_pred, zero_division=0):.3f}\n"
        f"F1-Score: {f1_score(y_true, y_pred, zero_division=0):.3f}\n"
        f"AUC-ROC: {roc_auc:.3f}\n"
        f"Avg Precision: {avg_precision:.3f}\n\n"
        f"Class Distribution:\n"
        f"  Positive: {sum(y_true)} samples\n"
        f"  Negative: {len(y_true) - sum(y_true)} samples\n"
        f"  Ratio: {100 * sum(y_true) / len(y_true):.1f}% positive"
    )
    axes[1, 2].text(
        0.1,
        0.5,
        metrics_text,
        fontsize=10,
        verticalalignment="center",
        transform=axes[1, 2].transAxes,
    )

    plt.tight_layout()
    plt.show()

    return roc_auc, avg_precision


# ==================== 7. Train and Evaluate Models ====================
print("\n" + "=" * 60)
print("Training Model with Class Weights")
print("=" * 60)

# Model with class weights
model_weighted = ImbalancedClassifier(input_size=X.shape[1])
model_weighted, train_loss_w, val_loss_w, train_acc_w, val_acc_w = train_model(
    model_weighted, train_loader, val_loader, num_epochs=20, use_class_weights=True
)

print("\n" + "=" * 60)
print("Training Model without Class Weights (Baseline)")
print("=" * 60)

# Baseline model without class weights
model_baseline = ImbalancedClassifier(input_size=X.shape[1])
model_baseline, train_loss_b, val_loss_b, train_acc_b, val_acc_b = train_model(
    model_baseline, train_loader, val_loader, num_epochs=20, use_class_weights=False
)

# ==================== 8. Compare Results ====================
print("\n" + "=" * 60)
print("Evaluation Results")
print("=" * 60)

# Evaluate both models
print("\n1. Model WITH Class Weights:")
y_true_w, y_pred_w, y_probs_w = evaluate_model(model_weighted, test_loader)
roc_auc_w, avg_prec_w = plot_metrics(
    y_true_w, y_pred_w, y_probs_w, "(With Class Weights)"
)

print("\n2. Model WITHOUT Class Weights (Baseline):")
y_true_b, y_pred_b, y_probs_b = evaluate_model(model_baseline, test_loader)
roc_auc_b, avg_prec_b = plot_metrics(y_true_b, y_pred_b, y_probs_b, "(Baseline)")

# ==================== 9. Training Curves Comparison ====================
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Training Curves Comparison", fontsize=16)

# Loss curves
axes[0, 0].plot(train_loss_w, "b-", label="Train (Weighted)", alpha=0.7)
axes[0, 0].plot(val_loss_w, "b--", label="Val (Weighted)", alpha=0.7)
axes[0, 0].plot(train_loss_b, "r-", label="Train (Baseline)", alpha=0.7)
axes[0, 0].plot(val_loss_b, "r--", label="Val (Baseline)", alpha=0.7)
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("Loss")
axes[0, 0].set_title("Loss Curves")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Accuracy curves
axes[0, 1].plot(train_acc_w, "b-", label="Train (Weighted)", alpha=0.7)
axes[0, 1].plot(val_acc_w, "b--", label="Val (Weighted)", alpha=0.7)
axes[0, 1].plot(train_acc_b, "r-", label="Train (Baseline)", alpha=0.7)
axes[0, 1].plot(val_acc_b, "r--", label="Val (Baseline)", alpha=0.7)
axes[0, 1].set_xlabel("Epoch")
axes[0, 1].set_ylabel("Accuracy (%)")
axes[0, 1].set_title("Accuracy Curves")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# ROC Curve Comparison
axes[1, 0].axis("off")

fpr_w, tpr_w, _ = roc_curve(y_true_w, y_probs_w)
fpr_b, tpr_b, _ = roc_curve(y_true_b, y_probs_b)

axes[1, 1].plot(
    fpr_w, tpr_w, "b-", lw=2, label=f"Weighted (AUC = {roc_auc_w:.3f})", alpha=0.7
)
axes[1, 1].plot(
    fpr_b, tpr_b, "r-", lw=2, label=f"Baseline (AUC = {roc_auc_b:.3f})", alpha=0.7
)
axes[1, 1].plot([0, 1], [0, 1], "k--", lw=1, alpha=0.3)
axes[1, 1].set_xlim([0.0, 1.0])
axes[1, 1].set_ylim([0.0, 1.05])
axes[1, 1].set_xlabel("False Positive Rate")
axes[1, 1].set_ylabel("True Positive Rate (Recall)")
axes[1, 1].set_title("ROC Curves Comparison")
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# Create a comparison table
table_data = [
    ["Model", "Precision", "Recall", "F1-Score", "AUC-ROC", "Avg Precision"],
    [
        "With Weights",
        f"{precision_score(y_true_w, y_pred_w, zero_division=0):.3f}",
        f"{recall_score(y_true_w, y_pred_w, zero_division=0):.3f}",
        f"{f1_score(y_true_w, y_pred_w, zero_division=0):.3f}",
        f"{roc_auc_w:.3f}",
        f"{avg_prec_w:.3f}",
    ],
    [
        "Baseline",
        f"{precision_score(y_true_b, y_pred_b, zero_division=0):.3f}",
        f"{recall_score(y_true_b, y_pred_b, zero_division=0):.3f}",
        f"{f1_score(y_true_b, y_pred_b, zero_division=0):.3f}",
        f"{roc_auc_b:.3f}",
        f"{avg_prec_b:.3f}",
    ],
]

# Add the table
table = axes[1, 0].table(
    cellText=table_data,
    colWidths=[0.2, 0.15, 0.15, 0.15, 0.15, 0.2],
    cellLoc="center",
    loc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.5)
axes[1, 0].set_title("Performance Comparison", fontsize=12, pad=20)

plt.tight_layout()
plt.show()

# ==================== 10. Key Observations ====================
print("\n" + "=" * 60)
print("Key Observations for Imbalanced Classification")
print("=" * 60)
print("""
1. **Precision vs Recall Trade-off:**
   - In imbalanced datasets, high accuracy can be misleading
   - The model might predict all samples as the majority class
   - Need to look at both precision and recall

2. **Class Weights Impact:**
   - Weighted loss function gives more importance to minority class
   - Often increases recall (catching more positive cases)
   - May slightly decrease precision

3. **ROC Curve vs PR Curve:**
   - ROC: Good for balanced datasets, shows TPR vs FPR
   - PR Curve: Better for imbalanced data, shows precision-recall tradeoff
   - High AUC-ROC but low Avg Precision indicates class imbalance issues

4. **Threshold Selection:**
   - Default threshold (0.5) may not be optimal
   - Adjust threshold based on business requirements
   - Higher threshold → Higher precision, Lower recall
   - Lower threshold → Lower precision, Higher recall

5. **Recommendations:**
   - Use PR Curve instead of ROC for severe imbalance
   - Consider oversampling/undersampling techniques
   - Try different thresholds based on use case
   - Monitor both precision and recall
""")

# Clean up
plt.close("all")
