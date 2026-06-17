import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import time
import ipdb

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device setup (CUDA/MPS/CPU)
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device: {device}")

# Data loading
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST("./data", train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)


# CORRECTED MODEL ARCHITECTURE (with proper dimension: 64*5*5 = 1600)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(1600, 128)  # Corrected: 64*5*5 = 1600
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # ipdb.set_trace()
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)  # Flatten from dimension 1 (channels)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# Training function (with device handling)
def train(model, device, train_loader, optimizer, epoch, scheduler=None):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += data.size(0)

    train_loss /= len(train_loader)
    train_accuracy = 100.0 * correct / total

    if scheduler:
        scheduler.step()

    return train_loss, train_accuracy


# Validation function
def validate(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += data.size(0)

    test_loss /= total
    test_accuracy = 100.0 * correct / total
    return test_loss, test_accuracy


# ========================
# a. Step Decay
# ========================
print("\n=== Step Decay ===")
step_size = 30
gamma = 0.1
model = CNN().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

train_losses, test_losses = [], []
train_accuracies, test_accuracies = [], []
learning_rates = []

num_epochs = 60
start_time = time.time()

for epoch in range(num_epochs):
    train_loss, train_acc = train(
        model, device, train_loader, optimizer, epoch, scheduler
    )
    test_loss, test_acc = validate(model, device, test_loader)

    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)
    learning_rates.append(optimizer.param_groups[0]['lr'])

    print(
        f"Epoch {epoch + 1}/{num_epochs}, "
        f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
        f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%, "
        f"LR: {optimizer.param_groups[0]['lr']:.6f}"
    )

# Plot results
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(train_losses, label="Training Loss")
plt.plot(test_losses, label="Validation Loss")
plt.title("Step Decay Learning Rate")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(train_accuracies, label="Training Accuracy")
plt.plot(test_accuracies, label="Validation Accuracy")
plt.title("Step Decay Learning Rate")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(learning_rates, label="Learning Rate", color='green')
plt.title("Learning Rate Schedule")
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.legend()
plt.tight_layout()
plt.savefig("step_decay.png")
print(f"Step Decay completed in {time.time() - start_time:.2f} seconds")
plt.close()

# ========================
# b. Exponential Decay
# ========================
print("\n=== Exponential Decay ===")
model = CNN().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
gamma = 0.95
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

train_losses, test_losses = [], []
train_accuracies, test_accuracies = [], []
learning_rates = []

num_epochs = 60
start_time = time.time()

for epoch in range(num_epochs):
    train_loss, train_acc = train(
        model, device, train_loader, optimizer, epoch, scheduler
    )
    test_loss, test_acc = validate(model, device, test_loader)

    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)
    learning_rates.append(optimizer.param_groups[0]['lr'])

    print(
        f"Epoch {epoch + 1}/{num_epochs}, "
        f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
        f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%, "
        f"LR: {optimizer.param_groups[0]['lr']:.6f}"
    )

# Plot results
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(train_losses, label="Training Loss")
plt.plot(test_losses, label="Validation Loss")
plt.title("Exponential Decay Learning Rate")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(train_accuracies, label="Training Accuracy")
plt.plot(test_accuracies, label="Validation Accuracy")
plt.title("Exponential Decay Learning Rate")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(learning_rates, label="Learning Rate", color='green')
plt.title("Learning Rate Schedule")
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.legend()
plt.tight_layout()
plt.savefig("exponential_decay.png")
print(f"Exponential Decay completed in {time.time() - start_time:.2f} seconds")
plt.close()

# ========================
# c. Cosine Annealing
# ========================
print("\n=== Cosine Annealing ===")
model = CNN().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
T_max = 30
# cosine curve from the initial value to a minimum value over a set of number of epochs(T_max)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)

train_losses, test_losses = [], []
train_accuracies, test_accuracies = [], []
learning_rates = []

num_epochs = 60
start_time = time.time()

for epoch in range(num_epochs):
    train_loss, train_acc = train(
        model, device, train_loader, optimizer, epoch, scheduler
    )
    test_loss, test_acc = validate(model, device, test_loader)

    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)
    learning_rates.append(optimizer.param_groups[0]['lr'])

    print(
        f"Epoch {epoch + 1}/{num_epochs}, "
        f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
        f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%, "
        f"LR: {optimizer.param_groups[0]['lr']:.6f}"
    )

# Plot results
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(train_losses, label="Training Loss")
plt.plot(test_losses, label="Validation Loss")
plt.title("Cosine Annealing Learning Rate")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(train_accuracies, label="Training Accuracy")
plt.plot(test_accuracies, label="Validation Accuracy")
plt.title("Cosine Annealing Learning Rate")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(learning_rates, label="Learning Rate", color='green')
plt.title("Learning Rate Schedule")
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.legend()
plt.tight_layout()
plt.savefig("cosine_annealing.png")
print(f"Cosine Annealing completed in {time.time() - start_time:.2f} seconds")
plt.close()

# ========================
# d. ReduceLROnPlateau
# ========================
print("\n=== ReduceLROnPlateau ===")
model = CNN().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer, mode="min", factor=0.5, patience=5, verbose=True
# )
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=5
)

train_losses, test_losses = [], []
train_accuracies, test_accuracies = [], []
learning_rates = []

num_epochs = 60
start_time = time.time()

for epoch in range(num_epochs):
    # not adding scheduler, scheduler updated outside of train
    train_loss, train_acc = train(model, device, train_loader, optimizer, epoch)
    test_loss, test_acc = validate(model, device, test_loader)

    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)
    learning_rates.append(optimizer.param_groups[0]['lr'])

    # Update scheduler based on validation loss
    scheduler.step(test_loss)

    print(
        f"Epoch {epoch + 1}/{num_epochs}, "
        f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
        f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%, "
        f"LR: {optimizer.param_groups[0]['lr']:.6f}"
    )

# Plot results
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(train_losses, label="Training Loss")
plt.plot(test_losses, label="Validation Loss")
plt.title("ReduceLROnPlateau Learning Rate")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(train_accuracies, label="Training Accuracy")
plt.plot(test_accuracies, label="Validation Accuracy")
plt.title("ReduceLROnPlateau Learning Rate")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(learning_rates, label="Learning Rate", color='green')
plt.title("Learning Rate Schedule")
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.legend()
plt.tight_layout()
plt.savefig("reduce_on_plateau.png")
print(f"ReduceLROnPlateau completed in {time.time() - start_time:.2f} seconds")
plt.close()

# ========================
# e. One Cycle Policy
# ========================
print("\n=== One Cycle Policy ===")
model = CNN().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
max_lr = 0.1
total_steps = len(train_loader) * 30  # 30 epochs
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=max_lr, steps_per_epoch=len(train_loader), epochs=30
)

train_losses, test_losses = [], []
train_accuracies, test_accuracies = [], []
learning_rates = []

num_epochs = 30
start_time = time.time()

for epoch in range(num_epochs):
    train_loss, train_acc = train(
        model, device, train_loader, optimizer, epoch, scheduler
    )
    test_loss, test_acc = validate(model, device, test_loader)

    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)
    learning_rates.append(optimizer.param_groups[0]['lr'])

    print(
        f"Epoch {epoch + 1}/{num_epochs}, "
        f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
        f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%, "
        f"LR: {optimizer.param_groups[0]['lr']:.6f}"
    )

# Plot results
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(train_losses, label="Training Loss")
plt.plot(test_losses, label="Validation Loss")
plt.title("One Cycle Learning Rate")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(train_accuracies, label="Training Accuracy")
plt.plot(test_accuracies, label="Validation Accuracy")
plt.title("One Cycle Learning Rate")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(learning_rates, label="Learning Rate", color='green')
plt.title("Learning Rate Schedule")
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.legend()
plt.tight_layout()
plt.savefig("one_cycle.png")
print(f"One Cycle completed in {time.time() - start_time:.2f} seconds")
plt.close()
