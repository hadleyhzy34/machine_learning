"""
Vision Transformer (ViT) from Scratch Demo
=========================================
This implements a Vision Transformer from scratch and trains it on CIFAR-10.
Reference: Dosovitskiy et al., "An Image is Worth 16x16 Words", ICLR 2021
"""

import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import ipdb


# =============================================================================
# Configuration
# =============================================================================
@dataclass
class ViTConfig:
    # Model architecture
    patch_size: int = 4  # Size of each patch (4x4 for CIFAR-10)
    embed_dim: int = 256  # Dimension of token embeddings
    num_heads: int = 8  # Number of attention heads
    num_layers: int = 6  # Number of transformer blocks
    mlp_ratio: int = 4  # Ratio of MLP hidden dim to embed_dim
    dropout: float = 0.1  # Dropout rate

    # Training
    batch_size: int = 128
    epochs: int = 10
    learning_rate: float = 3e-4
    weight_decay: float = 0.05

    # Data
    image_size: int = 32  # CIFAR-10 images are 32x32
    num_classes: int = 10

    # Device
    device: str = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )


def get_device():
    """Get the best available device: MPS (Apple Silicon), CUDA, or CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


config = ViTConfig()


# =============================================================================
# Vision Transformer Components
# =============================================================================


class PatchEmbedding(nn.Module):
    """
    Split image into patches and embed them.

    Args:
        img_size: Size of the input image (assumed square)
        patch_size: Size of each patch
        embed_dim: Dimension of the embedding
    """

    def __init__(self, img_size: int, patch_size: int, embed_dim: int):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        # Use a convolution to extract patches and project to embed_dim
        # This is equivalent to splitting into patches and applying a linear layer
        self.proj = nn.Conv2d(
            in_channels=3,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images of shape (B, C, H, W)
        Returns:
            Patch embeddings of shape (B, N, D) where N is number of patches
        """
        # (B, C, H, W) -> (B, D, H/P, W/P)
        x = self.proj(x)
        # (B, D, H/P, W/P) -> (B, D, N)
        x = x.flatten(2)
        # (B, D, N) -> (B, N, D)
        x = x.transpose(1, 2)
        return x


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism.

    Args:
        embed_dim: Dimension of embeddings
        num_heads: Number of attention heads
        dropout: Dropout rate
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, (
            "embed_dim must be divisible by num_heads"
        )

        # Linear projections for Q, K, V
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        # Scale factor for scaled dot-product attention
        self.scale = self.head_dim**-0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tokens of shape (B, N, D)
        Returns:
            Output tokens of shape (B, N, D)
        """
        # ipdb.set_trace()
        B, N, D = x.shape

        # Compute Q, K, V in one go
        # (B, N, 3D) -> (B, N, 3, H, D/H) -> (3, B, H, N, D/H)
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Compute attention scores
        # (B, H, N, D/H) @ (B, H, D/H, N) -> (B, H, N, N)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        # Apply attention to values
        # (B, H, N, N) @ (B, H, N, D/H) -> (B, H, N, D/H)
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)

        # Output projection
        x = self.proj(x)
        x = self.proj_dropout(x)
        return x


class MLP(nn.Module):
    """
    Feed-forward network with two linear layers and GELU activation.

    Args:
        in_features: Input dimension
        hidden_features: Hidden dimension (usually 4x in_features)
        dropout: Dropout rate
    """

    def __init__(self, in_features: int, hidden_features: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    Transformer encoder block with self-attention and MLP.

    Args:
        embed_dim: Dimension of embeddings
        num_heads: Number of attention heads
        mlp_ratio: Ratio of MLP hidden dim to embed_dim
        dropout: Dropout rate
    """

    def __init__(
        self, embed_dim: int, num_heads: int, mlp_ratio: int = 4, dropout: float = 0.0
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(
            embed_dim,
            embed_dim * mlp_ratio,
            dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tokens of shape (B, N, D)
        Returns:
            Output tokens of shape (B, N, D)
        """
        # Self-attention with residual connection
        x = x + self.attn(self.norm1(x))
        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) model.

    Args:
        img_size: Size of input images
        patch_size: Size of patches
        embed_dim: Dimension of embeddings
        num_heads: Number of attention heads
        num_layers: Number of transformer blocks
        mlp_ratio: Ratio of MLP hidden dim to embed_dim
        num_classes: Number of output classes
        dropout: Dropout rate
    """

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        mlp_ratio: int,
        num_classes: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, embed_dim)

        # Class token and position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.n_patches + 1, embed_dim)
        )

        # Dropout
        self.pos_drop = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
                for _ in range(num_layers)
            ]
        )

        # Final layer norm
        self.norm = nn.LayerNorm(embed_dim)

        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images of shape (B, C, H, W)
        Returns:
            Class logits of shape (B, num_classes)
        """
        ipdb.set_trace()
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # (B, N, D)

        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, N+1, D)

        # Add position embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        # Layer norm
        x = self.norm(x)

        # Extract class token output
        cls_token = x[:, 0]  # (B, D)

        # Classification head
        logits = self.head(cls_token)  # (B, num_classes)
        return logits

    def get_attention_maps(self, x: torch.Tensor, layer_idx: int = -1):
        """
        Get attention maps from a specific layer for visualization.

        Args:
            x: Input images
            layer_idx: Index of transformer block to get attention from
        Returns:
            Attention maps of shape (B, num_heads, N+1, N+1)
        """
        B = x.shape[0]

        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for i, block in enumerate(self.blocks):
            if i == layer_idx:
                # Return attention from this layer
                x_norm = block.norm1(x)
                qkv = (
                    block.attn.qkv(x_norm)
                    .reshape(B, -1, 3, block.attn.num_heads, block.attn.head_dim)
                    .permute(2, 0, 3, 1, 4)
                )
                q, k, v = qkv[0], qkv[1], qkv[2]
                attn = (q @ k.transpose(-2, -1)) * block.attn.scale
                attn = F.softmax(attn, dim=-1)
                return attn
            x = block(x)

        return None


# =============================================================================
# Data Loading
# =============================================================================


def get_dataloaders(batch_size: int = 128, num_workers: int = 2, device: str = "cpu"):
    """
    Create train and test dataloaders for CIFAR-10.

    Args:
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading
        device: Device being used (MPS requires num_workers=0)
    """
    # MPS doesn't support multiprocessing, set num_workers to 0
    if device == "mps" and num_workers > 0:
        print("Note: MPS requires num_workers=0 for dataloaders")
        num_workers = 0

    # Data augmentation and normalization for training
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ]
    )

    # Only normalization for testing
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ]
    )

    # Load datasets
    train_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=train_transform,
    )

    test_dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=test_transform,
    )

    # MPS doesn't support pin_memory
    pin_memory = device == "cuda"

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, test_loader, train_dataset.classes


# =============================================================================
# Training
# =============================================================================


def train_epoch(model, train_loader, optimizer, criterion, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.epochs}")
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Update progress bar
        pbar.set_postfix(
            {"loss": f"{loss.item():.4f}", "acc": f"{100.0 * correct / total:.2f}%"}
        )

    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, test_loader, criterion, device):
    """Evaluate the model on test data."""
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

    avg_loss = total_loss / len(test_loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def train_model(model, train_loader, test_loader, config):
    """Train the model and return training history."""
    device = config.device
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    for epoch in range(config.epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        print(f"\nEpoch {epoch + 1}/{config.epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Test Loss:  {test_loss:.4f}, Test Acc:  {test_acc:.2f}%")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}\n")

    return history


# =============================================================================
# Visualization
# =============================================================================


def visualize_training_history(history):
    """Plot training and test metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss plot
    axes[0].plot(history["train_loss"], label="Train Loss", marker="o")
    axes[0].plot(history["test_loss"], label="Test Loss", marker="s")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Progress - Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy plot
    axes[1].plot(history["train_acc"], label="Train Acc", marker="o")
    axes[1].plot(history["test_acc"], label="Test Acc", marker="s")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Training Progress - Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("vit_training_history.png", dpi=150, bbox_inches="tight")
    print("Saved training history plot to vit_training_history.png")
    plt.close()


def visualize_patches(images, patch_size=4):
    """Visualize image patches to understand patch embedding."""
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))

    # Denormalize for visualization
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.247, 0.243, 0.261]).view(3, 1, 1)

    for i in range(8):
        img = images[i].cpu() * std + mean
        img = img.clamp(0, 1).permute(1, 2, 0).numpy()

        # Show original image
        ax = axes[i // 4, i % 4]
        ax.imshow(img)
        ax.set_title(f"Patch Size: {patch_size}x{patch_size}")
        ax.axis("off")

        # Draw patch grid
        n_patches = 32 // patch_size
        for j in range(n_patches):
            for k in range(n_patches):
                rect = plt.Rectangle(
                    (k * patch_size, j * patch_size),
                    patch_size,
                    patch_size,
                    fill=False,
                    edgecolor="r",
                    linewidth=0.5,
                    alpha=0.7,
                )
                ax.add_patch(rect)

    plt.suptitle("CIFAR-10 Images with Patch Grid", fontsize=14)
    plt.tight_layout()
    plt.savefig("vit_patches.png", dpi=150, bbox_inches="tight")
    print("Saved patches visualization to vit_patches.png")
    plt.close()


def visualize_attention(model, test_loader, classes, device, num_samples=4):
    """Visualize attention maps from the model."""
    model.eval()

    # Get a batch of images
    images, labels = next(iter(test_loader))
    images, labels = images[:num_samples].to(device), labels[:num_samples].to(device)

    # Get attention from the last layer
    attention_maps = model.get_attention_maps(images, layer_idx=-1)
    if attention_maps is None:
        return

    # Denormalize for visualization
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.247, 0.243, 0.261]).view(3, 1, 1)

    patch_size = config.patch_size
    n_patches = config.image_size // patch_size

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))

    for i in range(num_samples):
        # Original image
        img = images[i].cpu() * std + mean
        img = img.clamp(0, 1).permute(1, 2, 0).numpy()
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Original\n({classes[labels[i]]})")
        axes[i, 0].axis("off")

        # Average attention from class token to patches
        attn = attention_maps[i, :, 0, 1:].mean(0).cpu().numpy()  # Shape: (n_patches,)
        attn = attn.reshape(n_patches, n_patches)

        # Upsample attention to image size
        attn_upsampled = (
            F.interpolate(
                torch.from_numpy(attn).unsqueeze(0).unsqueeze(0),
                size=(config.image_size, config.image_size),
                mode="bicubic",
            )
            .squeeze()
            .numpy()
        )

        # Show attention map
        im = axes[i, 1].imshow(attn_upsampled, cmap="hot", vmin=0, vmax=attn.max())
        axes[i, 1].set_title("Attention Map")
        axes[i, 1].axis("off")

        # Overlay attention on image
        axes[i, 2].imshow(img)
        axes[i, 2].imshow(
            attn_upsampled, cmap="hot", alpha=0.5, vmin=0, vmax=attn.max()
        )
        axes[i, 2].set_title("Attention Overlay")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig("vit_attention.png", dpi=150, bbox_inches="tight")
    print("Saved attention visualization to vit_attention.png")
    plt.close()


# =============================================================================
# Main
# =============================================================================


def print_model_info(model):
    """Print model architecture and parameter count."""
    print("=" * 60)
    print("Vision Transformer Architecture")
    print("=" * 60)
    print(model)
    print("=" * 60)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size (MB): {total_params * 4 / (1024**2):.2f}")
    print("=" * 60)


def main():
    """Main training and evaluation loop."""
    print("\n" + "=" * 60)
    print("Vision Transformer from Scratch - CIFAR-10 Demo")
    print("=" * 60 + "\n")

    # Print configuration
    print("Configuration:")
    for key, value in config.__dict__.items():
        print(f"  {key}: {value}")
    print()

    # Set device
    print(f"Using device: {config.device}\n")

    # Load data
    print("Loading CIFAR-10 dataset...")
    train_loader, test_loader, classes = get_dataloaders(
        config.batch_size, device=config.device
    )
    print(f"Classes: {classes}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}\n")

    # Visualize patches
    print("Visualizing patches...")
    sample_images, _ = next(iter(train_loader))
    visualize_patches(sample_images, config.patch_size)
    print()

    # Create model
    print("Creating Vision Transformer model...")
    model = VisionTransformer(
        img_size=config.image_size,
        patch_size=config.patch_size,
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        mlp_ratio=config.mlp_ratio,
        num_classes=config.num_classes,
        dropout=config.dropout,
    ).to(config.device)

    print_model_info(model)
    print()

    # Train model
    print("Starting training...\n")
    history = train_model(model, train_loader, test_loader, config)

    # Visualize training history
    print("Visualizing training history...")
    visualize_training_history(history)

    # Visualize attention maps
    print("Visualizing attention maps...")
    visualize_attention(model, test_loader, classes, config.device)

    # Final evaluation
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Final Test Accuracy: {history['test_acc'][-1]:.2f}%")
    print(f"Best Test Accuracy: {max(history['test_acc']):.2f}%")
    print("=" * 60 + "\n")

    return model, history


if __name__ == "__main__":
    model, history = main()
