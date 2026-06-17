import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ====================== 1D BATCH NORM (for Linear Layers) ======================
print("=" * 50)
print("BATCHNORM 1D (for Linear/Fully Connected Layers)")
print("=" * 50)

# Create sample batch data: [batch_size, features]
batch_size, features = 4, 5
data_1d = torch.tensor(
    [
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [2.0, 3.0, 4.0, 5.0, 6.0],
        [3.0, 4.0, 5.0, 6.0, 7.0],
        [4.0, 5.0, 6.0, 7.0, 8.0],
    ],
    dtype=torch.float32,
)

print(f"\n1D Input shape: {data_1d.shape}")  # [4, 5]
print(f"Data (4 samples, 5 features):\n{data_1d}")

# 1. PyTorch's BatchNorm1d
print("\n" + "-" * 50)
print("PyTorch's BatchNorm1d:")
print("-" * 50)
bn1d = nn.BatchNorm1d(features, affine=False)  # affine=False means no gamma/beta
bn1d.train()  # Training mode uses batch statistics

output_pytorch = bn1d(data_1d)
print(f"Normalized output:\n{output_pytorch}")
print(f"Mean per feature: {output_pytorch.mean(dim=0)}")
print(f"Std per feature: {output_pytorch.std(dim=0)}")

# 2. Manual implementation
print("\n" + "-" * 50)
print("Manual BatchNorm1d (Training Mode):")
print("-" * 50)

# Step 1: Calculate mean and variance per feature (across batch)
batch_mean = data_1d.mean(dim=0, keepdim=True)  # Shape: [1, 5]
batch_var = data_1d.var(dim=0, keepdim=True, unbiased=False)  # Shape: [1, 5]
eps = 1e-5

print(f"Batch mean (per feature): {batch_mean.squeeze()}")
print(f"Batch variance (per feature): {batch_var.squeeze()}")

# Step 2: Normalize
output_manual = (data_1d - batch_mean) / torch.sqrt(batch_var + eps)
print(f"\nNormalized output (manual):\n{output_manual}")

# Check if they match
print(f"\nMatch? {torch.allclose(output_pytorch, output_manual, rtol=1e-5)}")

# ====================== 2D BATCH NORM (for Conv Layers) ======================
print("\n\n" + "=" * 50)
print("BATCHNORM 2D (for Convolutional Layers)")
print("=" * 50)

# Create sample batch data: [batch_size, channels, height, width]
batch_size, channels, height, width = 2, 3, 4, 4
data_2d = torch.randn(batch_size, channels, height, width) * 2 + 1  # Mean=1, Std=2

print(f"\n2D Input shape: {data_2d.shape}")  # [2, 3, 4, 4]
print(f"Data - Batch 0, Channel 0:\n{data_2d[0, 0]}")

# 1. PyTorch's BatchNorm2d
print("\n" + "-" * 50)
print("PyTorch's BatchNorm2d:")
print("-" * 50)
bn2d = nn.BatchNorm2d(channels, affine=False)
bn2d.train()

output_2d_pytorch = bn2d(data_2d)
print(f"Normalized - Batch 0, Channel 0:\n{output_2d_pytorch[0, 0]}")
print(f"\nMean per channel: {output_2d_pytorch.mean(dim=[0, 2, 3])}")
print(f"Std per channel: {output_2d_pytorch.std(dim=[0, 2, 3])}")

# 2. Manual implementation
print("\n" + "-" * 50)
print("Manual BatchNorm2d (Training Mode):")
print("-" * 50)

# For 2D: Calculate mean and variance per channel (across batch, height, width)
batch_mean_2d = data_2d.mean(dim=[0, 2, 3], keepdim=True)  # Shape: [1, 3, 1, 1]
batch_var_2d = data_2d.var(
    dim=[0, 2, 3], keepdim=True, unbiased=False
)  # Shape: [1, 3, 1, 1]

print(f"Batch mean (per channel): {batch_mean_2d.squeeze()}")
print(f"Batch variance (per channel): {batch_var_2d.squeeze()}")

# Normalize
output_2d_manual = (data_2d - batch_mean_2d) / torch.sqrt(batch_var_2d + eps)
print(f"\nNormalized - Batch 0, Channel 0 (manual):\n{output_2d_manual[0, 0]}")

# Check if they match
print(f"\nMatch? {torch.allclose(output_2d_pytorch, output_2d_manual, rtol=1e-5)}")

# ====================== TRAINING vs INFERENCE ======================
print("\n\n" + "=" * 50)
print("TRAINING vs INFERENCE Mode")
print("=" * 50)

# Create a BatchNorm layer that tracks running statistics
bn_with_stats = nn.BatchNorm1d(features, momentum=0.1)

# Training: Use batch statistics, update running statistics
print("\n1. TRAINING MODE (bn.training = True):")
bn_with_stats.train()

# Process multiple batches
running_means = []
running_vars = []

for i in range(3):
    # Generate random batch with different statistics
    batch = torch.randn(batch_size, features) * (i + 1) + i * 2

    with torch.no_grad():
        output = bn_with_stats(batch)

    batch_mean = batch.mean(dim=0)
    batch_var = batch.var(dim=0, unbiased=False)

    print(
        f"\nBatch {i + 1}: Original mean={batch_mean.mean():.3f}, std={batch.std():.3f}"
    )
    print(f"  Running mean: {bn_with_stats.running_mean[:3]}")
    print(f"  Running var:  {bn_with_stats.running_var[:3]}")
    print(f"  Output mean: {output.mean():.3f}, std={output.std():.3f}")

    running_means.append(bn_with_stats.running_mean.clone())
    running_vars.append(bn_with_stats.running_var.clone())

# Inference: Use running statistics
print("\n\n2. INFERENCE MODE (bn.eval()):")
bn_with_stats.eval()

# New batch with different statistics
new_batch = torch.randn(batch_size, features) * 3 + 10
print(f"\nNew batch: Original mean={new_batch.mean():.3f}, std={new_batch.std():.3f}")

with torch.no_grad():
    output_eval = bn_with_stats(new_batch)

print(f"Using running mean: {bn_with_stats.running_mean[:3]}")
print(f"Using running var:  {bn_with_stats.running_var[:3]}")
print(f"Output mean: {output_eval.mean():.3f}, std={output_eval.std():.3f}")

# ====================== COMPLETE IMPLEMENTATION ======================
print("\n\n" + "=" * 50)
print("COMPLETE BATCHNORM IMPLEMENTATION FROM SCRATCH")
print("=" * 50)


class SimpleBatchNorm1d(nn.Module):
    """BatchNorm1d implementation from scratch"""

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # Trainable parameters
        self.gamma = nn.Parameter(torch.ones(num_features))  # Scale
        self.beta = nn.Parameter(torch.zeros(num_features))  # Shift

        # Running statistics (updated during training, used during inference)
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

        # Track training/inference mode
        self.training_mode = True

    def forward(self, x):
        if self.training_mode:
            # Training: use batch statistics
            if x.dim() == 2:  # [batch, features]
                batch_mean = x.mean(dim=0)
                batch_var = x.var(dim=0, unbiased=False)
            else:  # [batch, features, length]
                batch_mean = x.mean(dim=[0, 2])
                batch_var = x.var(dim=[0, 2], unbiased=False)

            # Update running statistics
            self.running_mean = (
                1 - self.momentum
            ) * self.running_mean + self.momentum * batch_mean
            self.running_var = (
                1 - self.momentum
            ) * self.running_var + self.momentum * batch_var

            mean = batch_mean
            var = batch_var
        else:
            # Inference: use running statistics
            mean = self.running_mean
            var = self.running_var

        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # Scale and shift
        if x.dim() == 2:
            output = self.gamma * x_norm + self.beta
        else:
            output = self.gamma.view(1, -1, 1) * x_norm + self.beta.view(1, -1, 1)

        return output

    def train(self, mode=True):
        self.training_mode = mode
        return self

    def eval(self):
        return self.train(False)


# Test our implementation
print("\nTesting SimpleBatchNorm1d:")
simple_bn = SimpleBatchNorm1d(features, momentum=0.1)
pytorch_bn = nn.BatchNorm1d(features, momentum=0.1)

# Copy parameters
simple_bn.gamma.data = pytorch_bn.weight.data.clone()
simple_bn.beta.data = pytorch_bn.bias.data.clone()
simple_bn.running_mean = pytorch_bn.running_mean.clone()
simple_bn.running_var = pytorch_bn.running_var.clone()

# Test batch
test_batch = torch.randn(4, features)

# Training mode
simple_bn.train()
pytorch_bn.train()

output_simple = simple_bn(test_batch)
output_pytorch = pytorch_bn(test_batch)

print(
    f"Training mode - Outputs match? {torch.allclose(output_simple, output_pytorch, rtol=1e-5)}"
)

# Inference mode
simple_bn.eval()
pytorch_bn.eval()

output_simple_eval = simple_bn(test_batch)
output_pytorch_eval = pytorch_bn(test_batch)

print(
    f"Inference mode - Outputs match? {torch.allclose(output_simple_eval, output_pytorch_eval, rtol=1e-5)}"
)

# ====================== VISUALIZATION ======================
print("\n\n" + "=" * 50)
print("VISUALIZATION: Batch Normalization Effect")
print("=" * 50)

# Create data with different batch statistics
torch.manual_seed(42)
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

# Original batches with different statistics
batches = []
for i in range(3):
    mean = i * 2
    std = 0.5 + i * 0.3
    batch = torch.randn(100, 1) * std + mean
    batches.append(batch)

    axes[0, i].hist(
        batch.numpy(), bins=30, alpha=0.7, color="skyblue", edgecolor="black"
    )
    axes[0, i].axvline(x=mean, color="red", linestyle="--", label=f"Mean: {mean:.1f}")
    axes[0, i].set_title(f"Batch {i + 1}: N({mean}, {std})")
    axes[0, i].set_xlabel("Value")
    axes[0, i].set_ylabel("Frequency")
    axes[0, i].legend()
    axes[0, i].grid(True, alpha=0.3)

# Apply batch normalization
bn_layer = nn.BatchNorm1d(1, affine=False)
bn_layer.train()

for i, batch in enumerate(batches):
    with torch.no_grad():
        normalized = bn_layer(batch)

    axes[1, i].hist(
        normalized.numpy(), bins=30, alpha=0.7, color="lightgreen", edgecolor="black"
    )
    axes[1, i].axvline(x=0, color="red", linestyle="--", label="Mean: 0")
    axes[1, i].axvline(x=1, color="orange", linestyle=":", label="Std: 1")
    axes[1, i].axvline(x=-1, color="orange", linestyle=":")
    axes[1, i].set_title(f"After BatchNorm")
    axes[1, i].set_xlabel("Value")
    axes[1, i].set_ylabel("Frequency")
    axes[1, i].set_xlim([-3, 3])
    axes[1, i].legend()
    axes[1, i].grid(True, alpha=0.3)

axes[0, 0].set_ylabel("Before BatchNorm", fontweight="bold")
axes[1, 0].set_ylabel("After BatchNorm", fontweight="bold")

plt.suptitle(
    "Batch Normalization: Different Distributions → Same Distribution",
    fontsize=12,
    fontweight="bold",
    y=1.02,
)
plt.tight_layout()
plt.show()

# ====================== SUMMARY ======================
print("\n" + "=" * 50)
print("QUICK REFERENCE: BatchNorm in PyTorch")
print("=" * 50)

print("\n1D (Linear Layers):")
print("  Input shape: [batch_size, features]")
print("  Normalize across: batch dimension (dim=0)")
print("  Code:")
print("    bn = nn.BatchNorm1d(num_features)")
print("    output = bn(input)")

print("\n2D (Conv Layers):")
print("  Input shape: [batch_size, channels, height, width]")
print("  Normalize across: batch, height, width (dim=[0,2,3])")
print("  Per channel normalization")
print("  Code:")
print("    bn = nn.BatchNorm2d(num_channels)")
print("    output = bn(input)")

print("\nKey Parameters:")
print("  - num_features/channels: Number of features/channels")
print("  - eps: Small constant for numerical stability (default: 1e-5)")
print("  - momentum: For updating running stats (default: 0.1)")
print("  - affine: Learnable γ (scale) and β (shift) (default: True)")

print("\nTraining vs Inference:")
print("  Training mode (model.train()):")
print("    - Uses current batch statistics")
print("    - Updates running_mean and running_var")
print("  ")
print("  Inference mode (model.eval()):")
print("    - Uses saved running statistics")
print("    - No gradient computation")

print("\nManual Calculation (Training Mode):")
print("  # For 1D input [batch, features]:")
print("  mean = input.mean(dim=0)")
print("  var = input.var(dim=0, unbiased=False)")
print("  normalized = (input - mean) / sqrt(var + eps)")
print("  output = gamma * normalized + beta")
