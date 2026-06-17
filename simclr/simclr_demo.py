"""
SimCLR Interactive Demo - Minimal Educational Version
======================================================

This simplified demo focuses on the CORE INSIGHT of contrastive learning:
- Pull positive pairs (same image, different augmentations) together
- Push negative pairs (different images) apart

Run this first to quickly see how contrastive learning works!
"""

# Use non-interactive backend to prevent plt.show() from blocking
import matplotlib
matplotlib.use('Agg')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


# ============================================================================
# SIMPLE CONTRASTIVE LEARNING DEMO
# ============================================================================

def create_simple_augmentations():
    """
    Create two augmented views of an image.

    KEY INSIGHT: These two views look different but represent the SAME object.
    The model must learn to recognize them as the same despite:
    - Different crops
    - Different colors
    - Different orientations (flip)
    """
    # Transform expects PIL images, converts to tensor internally
    transform = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return lambda img: (transform(img), transform(img))


def create_augmented_dataset(base_dataset):
    """
    Create a dataset that returns two augmented views per image.

    IMPORTANT: CIFAR-10 with transform=None returns PIL Images,
    which is what we need for our augmentations.
    """
    class AugmentedDataset(torch.utils.data.Dataset):
        def __init__(self, dataset):
            self.dataset = dataset
            self.augment = create_simple_augmentations()

        def __getitem__(self, idx):
            img, label = self.dataset[idx]
            view1, view2 = self.augment(img)
            return (view1, view2), label

        def __len__(self):
            return len(self.dataset)

    return AugmentedDataset(base_dataset)


class SimpleEncoder(nn.Module):
    """
    Minimal encoder network for demonstration.

    Architecture: Conv -> ReLU -> Pool -> FC -> Embedding
    """
    def __init__(self, embed_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x16x16

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64x8x8

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # 128x4x4

            nn.Flatten(),
            nn.Linear(128 * 4 * 4, embed_dim)
        )

    def forward(self, x):
        return self.encoder(x)


class SimpleContrastiveLoss(nn.Module):
    """
    Simplified NT-Xent contrastive loss.

    INTUITION:
    - For each image i, we have two views: view1[i] and view2[i]
    - These form a POSITIVE pair
    - All other combinations are NEGATIVE pairs

    GOAL:
    - Maximize similarity between positive pairs
    - Minimize similarity between negative pairs
    """
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, embed1, embed2):
        """
        Args:
            embed1: Embeddings of view1 [N, D]
            embed2: Embeddings of view2 [N, D]

        Returns:
            contrastive_loss: scalar
        """
        N = embed1.shape[0]

        # Normalize embeddings (for cosine similarity)
        embed1 = F.normalize(embed1, p=2, dim=1)
        embed2 = F.normalize(embed2, p=2, dim=1)

        # Concatenate all embeddings: [2N, D]
        all_embeds = torch.cat([embed1, embed2], dim=0)

        # Compute similarity matrix: [2N, 2N]
        sim_matrix = torch.matmul(all_embeds, all_embeds.T) / self.temperature

        # Labels: for view i, positive pair is at position (i + N) % 2N
        labels = torch.arange(N)
        labels = torch.cat([labels + N, labels])

        # Cross entropy: maximize similarity of positive pairs
        loss = F.cross_entropy(sim_matrix, labels)

        return loss


def demo_contrastive_learning():
    """
    Run a minimal contrastive learning demonstration.
    """
    print("="*60)
    print("SimCLR: Contrastive Learning Demo")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load CIFAR-10 WITHOUT transform (get PIL images for augmentation)
    print("\nLoading CIFAR-10...")
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=None
    )

    # Create augmented dataset
    aug_dataset = create_augmented_dataset(train_dataset)
    loader = DataLoader(aug_dataset, batch_size=128, shuffle=True, num_workers=0)

    # Initialize model
    print("Initializing model...")
    model = SimpleEncoder(embed_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = SimpleContrastiveLoss(temperature=0.5)

    # Training tracking
    losses = []
    pos_sims = []
    neg_sims = []

    # Training loop
    n_epochs = 15
    print(f"\nTraining for {n_epochs} epochs...")
    print("(Contrastive learning - NO LABELS USED!)")
    print("-" * 60)

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        epoch_pos_sim = 0
        epoch_neg_sim = 0
        n_batches = 0

        for (view1, view2), _ in loader:
            view1, view2 = view1.to(device), view2.to(device)

            optimizer.zero_grad()

            # Get embeddings
            embed1 = model(view1)
            embed2 = model(view2)

            # Compute contrastive loss
            loss = criterion(embed1, embed2)

            loss.backward()
            optimizer.step()

            # Track metrics
            epoch_loss += loss.item()

            # Track similarities
            with torch.no_grad():
                all_embeds = torch.cat([embed1, embed2], dim=0)
                all_embeds = F.normalize(all_embeds, p=2, dim=1)
                sim_matrix = torch.matmul(all_embeds, all_embeds.T)

                N = embed1.shape[0]
                pos_sim = torch.diag(sim_matrix[:N, N:]).mean().item()

                # Sample negative similarities
                neg_mask = ~torch.eye(2*N, dtype=bool, device=device)
                neg_mask[:N, N:] = False
                neg_mask[N:, :N] = False
                neg_sim = sim_matrix[neg_mask].mean().item()

                epoch_pos_sim += pos_sim
                epoch_neg_sim += neg_sim

            n_batches += 1
            epoch_loss += loss.item()

        # Average metrics
        avg_loss = epoch_loss / n_batches
        avg_pos_sim = epoch_pos_sim / n_batches
        avg_neg_sim = epoch_neg_sim / n_batches

        losses.append(avg_loss)
        pos_sims.append(avg_pos_sim)
        neg_sims.append(avg_neg_sim)

        print(f"Epoch {epoch+1:2d}/{n_epochs}: "
              f"Loss={avg_loss:.4f}, "
              f"PosSim={avg_pos_sim:.3f}, "
              f"NegSim={avg_neg_sim:.3f}")

    print("-" * 60)
    print("Training complete!")

    # Plot results
    plot_demo_results(model, loader, device, losses, pos_sims, neg_sims)

    return model, losses


def plot_demo_results(model, loader, device, losses, pos_sims, neg_sims):
    """
    Visualize the results of contrastive learning.
    """
    print("\nGenerating visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Training curves
    ax = axes[0, 0]
    ax.plot(losses, 'b-', linewidth=2, label='Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('NT-Xent Loss')
    ax.set_title('Training Loss')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 2. Similarity curves
    ax = axes[0, 1]
    ax.plot(pos_sims, 'g-', linewidth=2, label='Positive Pairs')
    ax.plot(neg_sims, 'r-', linewidth=2, label='Negative Pairs')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('Pairwise Similarity Over Time')
    ax.set_ylim(-0.2, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # 3. PCA visualization
    ax = axes[1, 0]
    model.eval()

    all_embeds = []
    all_labels = []

    with torch.no_grad():
        for (view1, view2), labels in loader:
            view1 = view1.to(device)
            embeds = model(view1)
            all_embeds.append(embeds.cpu())
            all_labels.append(labels)

    embeds_np = torch.cat(all_embeds, dim=0).numpy()[:500]
    labels_np = torch.cat(all_labels, dim=0).numpy()[:500]

    pca = PCA(n_components=2)
    embeds_2d = pca.fit_transform(embeds_np)

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for class_idx in range(10):
        mask = labels_np == class_idx
        ax.scatter(embeds_2d[mask, 0], embeds_2d[mask, 1],
                  c=[colors[class_idx]], alpha=0.5, s=15,
                  label=class_names[class_idx])

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
    ax.set_title('Learned Embeddings (PCA)\nColors = true labels (NOT used in training!)')
    ax.legend(fontsize=6, loc='upper left')

    # 4. Show sample augmented pairs
    ax = axes[1, 1]
    ax.axis('off')

    # Get a few samples
    (view1, view2), labels = next(iter(loader))

    n_show = 4
    for i in range(n_show):
        # View 1
        ax_view1 = plt.subplot(n_show, 2, 2*i + 1)
        img1 = view1[i].permute(1, 2, 0).numpy() * 0.5 + 0.5
        img1 = np.clip(img1, 0, 1)
        ax_view1.imshow(img1)
        ax_view1.set_title(f'View 1\nClass: {class_names[labels[i]]}')
        ax_view1.axis('off')

        # View 2
        ax_view2 = plt.subplot(n_show, 2, 2*i + 2)
        img2 = view2[i].permute(1, 2, 0).numpy() * 0.5 + 0.5
        img2 = np.clip(img2, 0, 1)
        ax_view2.imshow(img2)
        ax_view2.set_title('View 2 (different augmentation)')
        ax_view2.axis('off')

    plt.suptitle('Contrastive Learning Results\n'
                 'The model learns to recognize these pairs as the SAME image',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('simclr_demo_results.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("Saved results to 'simclr_demo_results.png'")

    # Print summary
    print("\n" + "="*60)
    print("KEY OBSERVATIONS")
    print("="*60)
    print("""
1. LOSS DECREASES: Model gets better at contrastive task

2. POSITIVE SIMILARITY INCREASES:
   - Augmented views of same image become more similar
   - Model learns invariance to augmentation

3. NEGATIVE SIMILARITY DECREASES/STAYS LOW:
   - Different images remain dissimilar
   - Model distinguishes between objects

4. PCA CLUSTERING:
   - Even without labels, classes separate somewhat!
   - Model learns semantic features, not random patterns

WHY CONTRASTIVE LEARNING WORKS:
- Must recognize same image under different augmentations
- Forces learning of INVARIANT features (shape, structure)
- Ignores superficial features (exact color, position)
- These invariant features are useful for downstream tasks!
""")


def interactive_embedding_playground():
    """
    Interactive demo: See how embeddings change during training.
    """
    print("\n" + "="*60)
    print("Interactive: Watch Embeddings Evolve")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data WITHOUT transform (get PIL images)
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=None
    )

    # Track embeddings at different training stages
    stages = [0, 5, 10, 20]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    model = SimpleEncoder(embed_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = SimpleContrastiveLoss(temperature=0.5)

    aug_dataset = create_augmented_dataset(train_dataset)
    loader = DataLoader(aug_dataset, batch_size=128, shuffle=True, num_workers=0)

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    print("Training and visualizing...")

    for epoch in range(25):
        model.train()
        for (view1, view2), _ in loader:
            view1, view2 = view1.to(device), view2.to(device)

            optimizer.zero_grad()
            embed1 = model(view1)
            embed2 = model(view2)
            loss = criterion(embed1, embed2)
            loss.backward()
            optimizer.step()

        # Visualize at specific epochs
        if epoch in stages:
            stage_idx = stages.index(epoch)
            ax = axes[stage_idx]

            model.eval()
            all_embeds = []
            all_labels = []

            with torch.no_grad():
                for (view1, view2), labels in loader:
                    view1 = view1.to(device)
                    embeds = model(view1)
                    all_embeds.append(embeds.cpu())
                    all_labels.append(labels)
                    break  # Just first batch

            embeds_np = torch.cat(all_embeds, dim=0).numpy()
            labels_np = torch.cat(all_labels, dim=0).numpy()

            pca = PCA(n_components=2)
            embeds_2d = pca.fit_transform(embeds_np)

            for class_idx in range(10):
                mask = labels_np == class_idx
                ax.scatter(embeds_2d[mask, 0], embeds_2d[mask, 1],
                          c=[colors[class_idx]], alpha=0.4, s=20,
                          label=class_names[class_idx] if epoch == 0 else None)

            ax.set_xlabel(f'Epoch {epoch}')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim(-3, 3)
            ax.set_ylim(-3, 3)

    axes[0].legend(fontsize=6, loc='best')
    plt.suptitle('How Embeddings Evolve During Contrastive Training', y=1.02)
    plt.tight_layout()
    plt.savefig('simclr_embedding_evolution.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("Saved embedding evolution to 'simclr_embedding_evolution.png'")


if __name__ == "__main__":
    # Run the main demo
    model, losses = demo_contrastive_learning()

    # Run interactive visualization
    interactive_embedding_playground()

    print("\n" + "="*60)
    print("Demo Complete!")
    print("="*60)
    print("\nGenerated files:")
    print("  - simclr_demo_results.png: Training curves + PCA visualization")
    print("  - simclr_embedding_evolution.png: How embeddings change over time")
    print("\nNext steps:")
    print("  - Run simclr_tutorial.py for full implementation")
    print("  - Try different temperatures, augmentations, architectures")
