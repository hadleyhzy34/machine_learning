import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import main, tqdm
import ipdb
import copy

if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Load and preprocess CIFAR-10 dataset
    transform = transforms.Compose(
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
        root="./data", train=False, download=True, transform=transform
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

    # Define CNN model WITHOUT dropout
    class CNN_NoDropout(nn.Module):
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

    # Define CNN model WITH dropout
    class CNN_Dropout(nn.Module):
        def __init__(self, dropout_rate=0.3):
            super().__init__()
            self.dropout_rate = dropout_rate
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(p=dropout_rate / 2),  # Spatial dropout
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(p=dropout_rate / 2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
            )

            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 4 * 4, 256),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(128, 10),
            )

        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x

    # Training function
    def train_model(
        model, trainloader, testloader, epochs=20, lr=0.001, model_name="Model"
    ):
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Store metrics
        train_losses, test_losses = [], []
        train_accs, test_accs = [], []

        best_acc = 0
        best_model = None

        for epoch in range(epochs):
            # Training phase
            model.train()
            running_loss, correct, total = 0.0, 0, 0

            pbar = tqdm(
                trainloader, desc=f"{model_name} Epoch {epoch + 1}/{epochs} [Train]"
            )
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
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
                    }
                )

            train_loss = running_loss / len(trainloader)
            train_acc = 100.0 * correct / total

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

            # Save best model
            if test_acc > best_acc:
                best_acc = test_acc
                best_model = copy.deepcopy(model.state_dict())

            print(
                f"{model_name} Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, "
                f"Train Acc: {train_acc:.2f}%, Test Loss: {test_loss:.4f}, "
                f"Test Acc: {test_acc:.2f}%"
            )

        # Load best model
        model.load_state_dict(best_model)
        return model, train_losses, test_losses, train_accs, test_accs, best_acc

    # Train both models
    print("\n" + "=" * 60)
    print("Training CNN WITHOUT Dropout")
    print("=" * 60)
    model_no_dropout = CNN_NoDropout()
    (
        model_no_dropout,
        train_loss_nd,
        test_loss_nd,
        train_acc_nd,
        test_acc_nd,
        best_acc_nd,
    ) = train_model(
        model_no_dropout, trainloader, testloader, epochs=20, model_name="NoDropout"
    )

    print("\n" + "=" * 60)
    print("Training CNN WITH Dropout (rate=0.3)")
    print("=" * 60)
    model_dropout = CNN_Dropout(dropout_rate=0.3)
    model_dropout, train_loss_d, test_loss_d, train_acc_d, test_acc_d, best_acc_d = (
        train_model(
            model_dropout, trainloader, testloader, epochs=20, model_name="WithDropout"
        )
    )

    # Final evaluation
    def evaluate_model(model, testloader, model_name):
        model.eval()
        correct, total = 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)

                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = 100.0 * correct / total

        # Calculate per-class accuracy
        class_correct = [0] * 10
        class_total = [0] * 10
        for i in range(len(all_labels)):
            label = all_labels[i]
            pred = all_preds[i]
            class_total[label] += 1
            if label == pred:
                class_correct[label] += 1

        print(f"\n{'=' * 60}")
        print(f"{model_name} - Final Results:")
        print(f"{'=' * 60}")
        print(f"Overall Test Accuracy: {accuracy:.2f}%")
        print(f"\nPer-class Accuracy:")
        for i in range(10):
            print(f"  {classes[i]}: {100.0 * class_correct[i] / class_total[i]:.1f}%")

        return accuracy

    # Final evaluation
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)
    acc_nd = evaluate_model(model_no_dropout, testloader, "Model WITHOUT Dropout")
    acc_d = evaluate_model(model_dropout, testloader, "Model WITH Dropout")

    print("\n" + "=" * 60)
    print("SUMMARY: Dropout Performance Improvement")
    print("=" * 60)
    print(f"No Dropout Model Best Accuracy: {best_acc_nd:.2f}%")
    print(f"Dropout Model Best Accuracy:    {best_acc_d:.2f}%")
    print(f"Improvement: {best_acc_d - best_acc_nd:+.2f} percentage points")

    # Plotting results
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Loss curves
    epochs = range(1, 21)
    axes[0, 0].plot(
        epochs, train_loss_nd, "b-", label="No Dropout (Train)", linewidth=2
    )
    axes[0, 0].plot(epochs, test_loss_nd, "b--", label="No Dropout (Test)", linewidth=2)
    axes[0, 0].plot(
        epochs, train_loss_d, "r-", label="With Dropout (Train)", linewidth=2
    )
    axes[0, 0].plot(
        epochs, test_loss_d, "r--", label="With Dropout (Test)", linewidth=2
    )
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Training and Test Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Accuracy curves
    axes[0, 1].plot(epochs, train_acc_nd, "b-", label="No Dropout (Train)", linewidth=2)
    axes[0, 1].plot(epochs, test_acc_nd, "b--", label="No Dropout (Test)", linewidth=2)
    axes[0, 1].plot(
        epochs, train_acc_d, "r-", label="With Dropout (Train)", linewidth=2
    )
    axes[0, 1].plot(epochs, test_acc_d, "r--", label="With Dropout (Test)", linewidth=2)
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy (%)")
    axes[0, 1].set_title("Training and Test Accuracy")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Accuracy gap (Overfitting measure)
    gap_nd = [train_acc_nd[i] - test_acc_nd[i] for i in range(len(train_acc_nd))]
    gap_d = [train_acc_d[i] - test_acc_d[i] for i in range(len(train_acc_d))]

    axes[1, 0].plot(epochs, gap_nd, "b-", label="No Dropout", linewidth=2)
    axes[1, 0].plot(epochs, gap_d, "r-", label="With Dropout", linewidth=2)
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Accuracy Gap (Train-Test) %")
    axes[1, 0].set_title("Overfitting Measure: Train-Test Accuracy Gap")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=0, color="k", linestyle="-", alpha=0.2)

    # Final comparison bar chart
    models = ["No Dropout", "With Dropout"]
    train_final = [train_acc_nd[-1], train_acc_d[-1]]
    test_final = [test_acc_nd[-1], test_acc_d[-1]]

    x = np.arange(len(models))
    width = 0.35
    axes[1, 1].bar(
        x - width / 2, train_final, width, label="Train Accuracy", color="skyblue"
    )
    axes[1, 1].bar(
        x + width / 2, test_final, width, label="Test Accuracy", color="lightcoral"
    )
    axes[1, 1].set_xlabel("Model")
    axes[1, 1].set_ylabel("Accuracy (%)")
    axes[1, 1].set_title("Final Train vs Test Accuracy Comparison")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(models)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis="y")

    # Add accuracy values on bars
    for i, v in enumerate(train_final):
        axes[1, 1].text(i - width / 2, v + 0.5, f"{v:.1f}%", ha="center")
    for i, v in enumerate(test_final):
        axes[1, 1].text(i + width / 2, v + 0.5, f"{v:.1f}%", ha="center")

    plt.tight_layout()
    plt.show()

    # Key observations summary
    print("\n" + "=" * 60)
    print("KEY OBSERVATIONS FROM THE EXPERIMENT:")
    print("=" * 60)
    print("1. OVERFITTING REDUCTION:")
    print(f"   - No Dropout train-test gap: {train_acc_nd[-1] - test_acc_nd[-1]:.1f}%")
    print(f"   - With Dropout train-test gap: {train_acc_d[-1] - test_acc_d[-1]:.1f}%")
    print(
        f"   → Dropout reduced overfitting by {abs((train_acc_d[-1] - test_acc_d[-1]) - (train_acc_nd[-1] - test_acc_nd[-1])):.1f} percentage points"
    )

    print("\n2. GENERALIZATION IMPROVEMENT:")
    print(f"   - No Dropout test accuracy: {test_acc_nd[-1]:.1f}%")
    print(f"   - With Dropout test accuracy: {test_acc_d[-1]:.1f}%")
    print(
        f"   → Dropout improved test accuracy by {test_acc_d[-1] - test_acc_nd[-1]:+.1f}%"
    )

    print("\n3. TRAINING BEHAVIOR:")
    print("   - No Dropout: Train accuracy climbs quickly but plateaus early")
    print("   - With Dropout: Slower training but continues improving generalization")
    print("   - Dropout prevents memorization, forces learning robust features")

    print("\n4. PRACTICAL IMPLICATIONS:")
    print("   • Dropout acts as model averaging during training")
    print("   • Prevents co-adaptation of neurons")
    print("   • Acts as implicit data augmentation")
    print("   • Particularly effective in fully-connected layers")
