import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import copy
import seaborn as sns
from sklearn.metrics import confusion_matrix

if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Load and preprocess CIFAR-10 dataset
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),  # Add data augmentation
            transforms.RandomRotation(10),  # Add data augmentation
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    transform_test = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    batch_size = 64

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )

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

    # Define a simple CNN model (same for all experiments)
    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
            )

            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 4 * 4, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 10),
            )

        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x

        def get_weights(self):
            """Get all trainable parameters for regularization calculation"""
            weights = []
            for param in self.parameters():
                if param.requires_grad and len(param.shape) > 1:  # Skip biases
                    weights.append(param.view(-1))
            return torch.cat(weights) if weights else torch.tensor([])

    # Training function with regularization
    def train_model_with_regularization(
        model,
        trainloader,
        testloader,
        epochs=30,
        lr=0.001,
        reg_type="none",
        lambda_reg=0.001,
        model_name="Model",
    ):
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()

        # Use weight_decay for L2 regularization (built into optimizer)
        if reg_type == "l2":
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=lambda_reg)
        else:
            optimizer = optim.Adam(model.parameters(), lr=lr)

        # Store metrics
        train_losses, test_losses = [], []
        train_accs, test_accs = [], []
        reg_losses = []
        weight_magnitudes = []

        best_acc = 0
        best_model = None

        for epoch in range(epochs):
            # Training phase
            model.train()
            running_loss, reg_loss, correct, total = 0.0, 0.0, 0, 0

            pbar = tqdm(
                trainloader, desc=f"{model_name} Epoch {epoch + 1}/{epochs} [Train]"
            )
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Add regularization term
                if reg_type == "l1":
                    l1_lambda = lambda_reg
                    l1_norm = sum(p.abs().sum() for p in model.parameters())
                    loss = loss + l1_lambda * l1_norm
                    reg_loss = l1_lambda * l1_norm.item()
                elif reg_type == "l2":
                    # L2 handled by optimizer's weight_decay
                    reg_loss = 0.0
                else:
                    reg_loss = 0.0

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                pbar.set_postfix(
                    {
                        "loss": running_loss / (len(pbar) + 1e-8),
                        "acc": 100.0 * correct / total,
                        "reg": reg_loss,
                    }
                )

            train_loss = running_loss / len(trainloader)
            train_acc = 100.0 * correct / total

            # Calculate average weight magnitude
            with torch.no_grad():
                total_weight = 0
                count = 0
                for param in model.parameters():
                    if (
                        param.requires_grad and len(param.shape) > 1
                    ):  # Weight matrices only
                        total_weight += param.abs().mean().item()
                        count += 1
                avg_weight_mag = total_weight / count if count > 0 else 0

            # Evaluation phase
            model.eval()
            running_loss, correct, total = 0.0, 0, 0

            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    running_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

            test_loss = running_loss / len(testloader)
            test_acc = 100.0 * correct / total

            # Save metrics
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            train_accs.append(train_acc)
            test_accs.append(test_acc)
            reg_losses.append(reg_loss)
            weight_magnitudes.append(avg_weight_mag)

            # Save best model
            if test_acc > best_acc:
                best_acc = test_acc
                best_model = copy.deepcopy(model.state_dict())

            if (epoch + 1) % 5 == 0:
                print(
                    f"{model_name} Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, "
                    f"Train Acc: {train_acc:.2f}%, Test Loss: {test_loss:.4f}, "
                    f"Test Acc: {test_acc:.2f}%, Weight Mag: {avg_weight_mag:.4f}"
                )

        # Load best model
        model.load_state_dict(best_model)
        return (
            model,
            train_losses,
            test_losses,
            train_accs,
            test_accs,
            reg_losses,
            weight_magnitudes,
            best_acc,
        )

    # Analyze weight distribution
    def analyze_weights(models_dict, model_names):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for idx, (name, model) in enumerate(models_dict.items()):
            all_weights = []
            weight_ranges = []

            for param_name, param in model.named_parameters():
                if param.requires_grad and len(param.shape) > 1:  # Weight matrices only
                    weights = param.data.cpu().numpy().flatten()
                    all_weights.extend(weights)
                    weight_ranges.append(
                        (
                            param_name,
                            weights.min(),
                            weights.mean(),
                            weights.max(),
                            weights.std(),
                        )
                    )

            all_weights = np.array(all_weights)

            # Plot histogram
            axes[idx].hist(all_weights, bins=100, alpha=0.7, edgecolor="black")
            axes[idx].set_xlabel("Weight Value")
            axes[idx].set_ylabel("Frequency")
            axes[idx].set_title(
                f"{name} Weight Distribution\n"
                f"Mean: {all_weights.mean():.4f}, Std: {all_weights.std():.4f}"
            )
            axes[idx].axvline(x=0, color="r", linestyle="--", alpha=0.5)
            axes[idx].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Print weight statistics
        print("\n" + "=" * 60)
        print("WEIGHT STATISTICS ANALYSIS")
        print("=" * 60)

        for name, model in models_dict.items():
            print(f"\n{name}:")
            total_params = 0
            non_zero_params = 0
            for param_name, param in model.named_parameters():
                if param.requires_grad and len(param.shape) > 1:
                    weights = param.data.cpu().numpy().flatten()
                    n_params = len(weights)
                    n_nonzero = np.sum(
                        np.abs(weights) > 0.001
                    )  # Threshold for "non-zero"
                    sparsity = 1.0 - (n_nonzero / n_params)
                    total_params += n_params
                    non_zero_params += n_nonzero

                    print(f"  {param_name}:")
                    print(
                        f"    Shape: {param.shape}, Non-zero: {n_nonzero}/{n_params} "
                        f"({(1 - sparsity) * 100:.1f}%), "
                        f"Mean: {weights.mean():.4f}, Std: {weights.std():.4f}"
                    )

            overall_sparsity = 1.0 - (non_zero_params / total_params)
            print(f"  Overall Sparsity: {overall_sparsity * 100:.1f}%")

    # Train three models: Base, L1, L2
    print("\n" + "=" * 60)
    print("Training BASE Model (No Regularization)")
    print("=" * 60)
    model_base = SimpleCNN()
    (
        model_base,
        train_loss_base,
        test_loss_base,
        train_acc_base,
        test_acc_base,
        reg_base,
        weight_base,
        best_acc_base,
    ) = train_model_with_regularization(
        model_base,
        trainloader,
        testloader,
        epochs=30,
        reg_type="none",
        lambda_reg=0.001,
        model_name="Base",
    )

    print("\n" + "=" * 60)
    print("Training L1 Regularized Model (λ=0.001)")
    print("=" * 60)
    model_l1 = SimpleCNN()
    (
        model_l1,
        train_loss_l1,
        test_loss_l1,
        train_acc_l1,
        test_acc_l1,
        reg_l1,
        weight_l1,
        best_acc_l1,
    ) = train_model_with_regularization(
        model_l1,
        trainloader,
        testloader,
        epochs=30,
        reg_type="l1",
        lambda_reg=0.001,
        model_name="L1",
    )

    print("\n" + "=" * 60)
    print("Training L2 Regularized Model (λ=0.001)")
    print("=" * 60)
    model_l2 = SimpleCNN()
    (
        model_l2,
        train_loss_l2,
        test_loss_l2,
        train_acc_l2,
        test_acc_l2,
        reg_l2,
        weight_l2,
        best_acc_l2,
    ) = train_model_with_regularization(
        model_l2,
        trainloader,
        testloader,
        epochs=30,
        reg_type="l2",
        lambda_reg=0.001,
        model_name="L2",
    )

    # Final evaluation
    def evaluate_model_final(model, testloader, model_name):
        model.eval()
        all_preds, all_labels, all_probs = [], [], []

        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        # Calculate metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        accuracy = 100.0 * np.mean(all_preds == all_labels)

        # Calculate confidence metrics
        correct_probs = all_probs[np.arange(len(all_preds)), all_preds]
        avg_confidence = np.mean(correct_probs)

        return accuracy, all_preds, all_labels, avg_confidence

    print("\n" + "=" * 60)
    print("FINAL MODEL EVALUATION")
    print("=" * 60)

    models = {"Base": model_base, "L1": model_l1, "L2": model_l2}

    results = {}
    for name, model in models.items():
        accuracy, preds, labels, confidence = evaluate_model_final(
            model, testloader, name
        )
        results[name] = {
            "accuracy": accuracy,
            "preds": preds,
            "labels": labels,
            "confidence": confidence,
        }
        print(f"{name}:")
        print(f"  Test Accuracy: {accuracy:.2f}%")
        print(f"  Average Confidence: {confidence:.4f}")
        print(f"  Best Epoch Accuracy: {locals()[f'best_acc_{name.lower()}']:.2f}%")

    # Plotting results
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Accuracy curves
    epochs = range(1, 31)
    axes[0, 0].plot(
        epochs, train_acc_base, "b-", label="Base (Train)", linewidth=2, alpha=0.8
    )
    axes[0, 0].plot(
        epochs, test_acc_base, "b--", label="Base (Test)", linewidth=2, alpha=0.8
    )
    axes[0, 0].plot(
        epochs, train_acc_l1, "r-", label="L1 (Train)", linewidth=2, alpha=0.8
    )
    axes[0, 0].plot(
        epochs, test_acc_l1, "r--", label="L1 (Test)", linewidth=2, alpha=0.8
    )
    axes[0, 0].plot(
        epochs, train_acc_l2, "g-", label="L2 (Train)", linewidth=2, alpha=0.8
    )
    axes[0, 0].plot(
        epochs, test_acc_l2, "g--", label="L2 (Test)", linewidth=2, alpha=0.8
    )
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Accuracy (%)")
    axes[0, 0].set_title("Training and Test Accuracy")
    axes[0, 0].legend(loc="lower right")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([40, 100])

    # 2. Loss curves
    axes[0, 1].plot(
        epochs, train_loss_base, "b-", label="Base (Train)", linewidth=2, alpha=0.8
    )
    axes[0, 1].plot(
        epochs, test_loss_base, "b--", label="Base (Test)", linewidth=2, alpha=0.8
    )
    axes[0, 1].plot(
        epochs, train_loss_l1, "r-", label="L1 (Train)", linewidth=2, alpha=0.8
    )
    axes[0, 1].plot(
        epochs, test_loss_l1, "r--", label="L1 (Test)", linewidth=2, alpha=0.8
    )
    axes[0, 1].plot(
        epochs, train_loss_l2, "g-", label="L2 (Train)", linewidth=2, alpha=0.8
    )
    axes[0, 1].plot(
        epochs, test_loss_l2, "g--", label="L2 (Test)", linewidth=2, alpha=0.8
    )
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].set_title("Training and Test Loss")
    axes[0, 1].legend(loc="upper right")
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Overfitting gap (Train-Test Accuracy Difference)
    gap_base = [
        train_acc_base[i] - test_acc_base[i] for i in range(len(train_acc_base))
    ]
    gap_l1 = [train_acc_l1[i] - test_acc_l1[i] for i in range(len(train_acc_l1))]
    gap_l2 = [train_acc_l2[i] - test_acc_l2[i] for i in range(len(train_acc_l2))]

    axes[0, 2].plot(epochs, gap_base, "b-", label="Base", linewidth=3, alpha=0.8)
    axes[0, 2].plot(epochs, gap_l1, "r-", label="L1", linewidth=3, alpha=0.8)
    axes[0, 2].plot(epochs, gap_l2, "g-", label="L2", linewidth=3, alpha=0.8)
    axes[0, 2].set_xlabel("Epoch")
    axes[0, 2].set_ylabel("Train-Test Accuracy Gap (%)")
    axes[0, 2].set_title("Overfitting Measure\n(Smaller Gap = Less Overfitting)")
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].axhline(y=0, color="k", linestyle="-", alpha=0.2)

    # 4. Weight magnitude over time
    axes[1, 0].plot(epochs, weight_base, "b-", label="Base", linewidth=3, alpha=0.8)
    axes[1, 0].plot(epochs, weight_l1, "r-", label="L1", linewidth=3, alpha=0.8)
    axes[1, 0].plot(epochs, weight_l2, "g-", label="L2", linewidth=3, alpha=0.8)
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Average Weight Magnitude")
    axes[1, 0].set_title(
        "Average Weight Magnitude Over Time\n(L1 reduces magnitude most aggressively)"
    )
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 5. Final test accuracy comparison
    models_names = ["Base", "L1", "L2"]
    final_accs = [test_acc_base[-1], test_acc_l1[-1], test_acc_l2[-1]]
    best_accs = [best_acc_base, best_acc_l1, best_acc_l2]

    x = np.arange(len(models_names))
    width = 0.35
    bars1 = axes[1, 1].bar(
        x - width / 2, final_accs, width, label="Final Epoch", color="skyblue"
    )
    bars2 = axes[1, 1].bar(
        x + width / 2, best_accs, width, label="Best Epoch", color="lightcoral"
    )
    axes[1, 1].set_xlabel("Model")
    axes[1, 1].set_ylabel("Test Accuracy (%)")
    axes[1, 1].set_title("Final vs Best Test Accuracy Comparison")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(models_names)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            axes[1, 1].text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.5,
                f"{height:.1f}%",
                ha="center",
                va="bottom",
            )

    # 6. Model confidence comparison
    confidences = [results[name]["confidence"] for name in models_names]
    colors = ["skyblue", "lightcoral", "lightgreen"]
    bars = axes[1, 2].bar(models_names, confidences, color=colors, alpha=0.8)
    axes[1, 2].set_xlabel("Model")
    axes[1, 2].set_ylabel("Average Confidence")
    axes[1, 2].set_title("Model Confidence on Correct Predictions\n(Higher ≠ Better)")
    axes[1, 2].set_ylim([0.8, 1.0])
    axes[1, 2].grid(True, alpha=0.3, axis="y")

    for bar, conf in zip(bars, confidences):
        axes[1, 2].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{conf:.4f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.show()

    # Analyze weight distributions
    print("\n" + "=" * 60)
    print("WEIGHT DISTRIBUTION ANALYSIS")
    print("=" * 60)
    analyze_weights(models, ["Base", "L1", "L2"])

    # Plot confusion matrices
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, (name, result) in enumerate(results.items()):
        cm = confusion_matrix(result["labels"], result["preds"])
        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        im = axes[idx].imshow(
            cm_normalized, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=1
        )
        axes[idx].set_title(f"{name} Model\nAccuracy: {result['accuracy']:.1f}%")
        axes[idx].set_xlabel("Predicted Label")
        axes[idx].set_ylabel("True Label")

        # Add text annotations
        thresh = cm_normalized.max() / 2.0
        for i in range(cm_normalized.shape[0]):
            for j in range(cm_normalized.shape[1]):
                axes[idx].text(
                    j,
                    i,
                    f"{cm_normalized[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if cm_normalized[i, j] > thresh else "black",
                )

        axes[idx].set_xticks(range(10))
        axes[idx].set_yticks(range(10))
        axes[idx].set_xticklabels(classes, rotation=45, ha="right")
        axes[idx].set_yticklabels(classes)

    plt.colorbar(im, ax=axes, orientation="horizontal", fraction=0.05, pad=0.1)
    plt.suptitle("Confusion Matrices (Normalized by Row)", y=1.02, fontsize=16)
    plt.tight_layout()
    plt.show()

    # Key insights summary
    print("\n" + "=" * 80)
    print("DEEP INSIGHTS: HOW L1 AND L2 REGULARIZATION MITIGATE OVERFITTING")
    print("=" * 80)

    print("\n1. OVERFITTING REDUCTION ANALYSIS:")
    overfit_gap_base = train_acc_base[-1] - test_acc_base[-1]
    overfit_gap_l1 = train_acc_l1[-1] - test_acc_l1[-1]
    overfit_gap_l2 = train_acc_l2[-1] - test_acc_l2[-1]

    print(f"   Base Model overfitting gap:  {overfit_gap_base:.1f}%")
    print(
        f"   L1 Model overfitting gap:    {overfit_gap_l1:.1f}% ({overfit_gap_base - overfit_gap_l1:+.1f}% reduction)"
    )
    print(
        f"   L2 Model overfitting gap:    {overfit_gap_l2:.1f}% ({overfit_gap_base - overfit_gap_l2:+.1f}% reduction)"
    )

    print("\n2. GENERALIZATION PERFORMANCE:")
    print(
        f"   Base Model test accuracy:     {test_acc_base[-1]:.1f}% (Best: {best_acc_base:.1f}%)"
    )
    print(
        f"   L1 Model test accuracy:      {test_acc_l1[-1]:.1f}% (Best: {best_acc_l1:.1f}%)"
    )
    print(
        f"   L2 Model test accuracy:      {test_acc_l2[-1]:.1f}% (Best: {best_acc_l2:.1f}%)"
    )

    print("\n3. WEIGHT ANALYSIS (Key Difference):")
    print("   • L1 (Lasso) Regularization:")
    print("     - Creates SPARSE solutions: many weights ≈ 0")
    print("     - Performs implicit feature selection")
    print("     - More aggressive weight shrinkage")
    print("     - Results in simpler models with fewer effective parameters")

    print("\n   • L2 (Ridge) Regularization:")
    print("     - Creates SMOOTH solutions: all weights are small but non-zero")
    print("     - Distributes weight magnitude evenly")
    print("     - More stable training, less aggressive than L1")
    print("     - Better for correlated features")

    print("\n4. PRACTICAL RECOMMENDATIONS:")
    print("   • Use L1 when:")
    print("     - You suspect many features are irrelevant")
    print("     - You want feature selection and interpretability")
    print("     - Model compression is important")
    print("     - Warning: May be unstable with correlated features")

    print("\n   • Use L2 when:")
    print("     - All features likely contribute to prediction")
    print("     - Features are correlated")
    print("     - You want stable, well-behaved optimization")
    print("     - General default choice for deep learning")

    print("\n   • Use Elastic Net (L1 + L2) when:")
    print("     - You want benefits of both methods")
    print("     - You have many correlated but potentially irrelevant features")

    print("\n5. HYPERPARAMETER TUNING TIPS:")
    print("   • Start with λ = 0.001 for L1/L2")
    print("   • Grid search range: λ ∈ [1e-5, 1e-2]")
    print("   • Monitor validation loss, not just accuracy")
    print("   • Combine with dropout for even better regularization")
    print("   • Use learning rate scheduling with regularization")

    print("\n6. COMMON PITFALLS TO AVOID:")
    print("   • Too high λ: Underfitting (high bias)")
    print("   • Too low λ: Overfitting (high variance)")
    print("   • Forgetting to disable regularization during inference")
    print("   • Applying L1 to biases (usually not recommended)")
    print("   • Not standardizing inputs before L2 regularization")
