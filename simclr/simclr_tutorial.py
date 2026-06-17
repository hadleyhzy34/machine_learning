"""
SimCLR: Simple Framework for Contrastive Learning of Visual Representations
============================================================================

This tutorial demonstrates the key concepts behind SimCLR from scratch.

CONTRASTIVE LEARNING INTUITION:
-------------------------------
Traditional supervised learning: X (image) -> Y (label)
Problem: Requires expensive labeled data

Contrastive learning: Learn representations by comparing examples
- Positive pairs: Similar examples (augmented views of same image)
- Negative pairs: Different examples (different images)

Goal: Pull positive pairs together, push negative pairs apart in embedding space

SIMCLR KEY INSIGHTS:
--------------------
1. Large batch sizes - need many negatives in each batch
2. Stronger data augmentations - creates more challenging positive pairs
3. Small MLP projection head - transforms representation before contrastive loss
4. Cosine similarity + NT-Xent loss - better than dot product

NT-Xent LOSS (Normalized Temperature-scaled Cross Entropy):
-----------------------------------------------------------
For a positive pair (i, j):
    loss = -log[exp(sim(z_i, z_j)/τ) / Σexp(sim(z_i, z_k)/τ)]
                        k≠i

Where:
- z_i, z_j = embeddings of positive pair
- τ = temperature parameter (controls concentration)
- sim = cosine similarity
- Denominator sums over all other samples (negatives)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
import seaborn as sns


# ============================================================================
# PART 1: DATA AUGMENTATION PIPELINE
# ============================================================================
# SimCLR relies heavily on data augmentation to create positive pairs

class SimCLRTransform:
    """
    SimCLR Data Augmentation Pipeline

    Creates two correlated views of the same image using random augmentations.
    These two views form a POSITIVE pair for contrastive learning.

    Key augmentations:
    - RandomResizedCrop: Changes scale and aspect ratio
    - ColorJitter: Changes brightness, contrast, saturation, hue
    - RandomGrayscale: Removes color information (50% chance)
    - GaussianBlur: Adds blur (50% chance)
    - HorizontalFlip: Mirrors image (50% chance)
    """

    def __init__(self, input_size=32, min_scale=0.2):
        # Transform for first view
        self.transform1 = transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=(min_scale, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # Transform for second view (different random augmentation)
        self.transform2 = transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=(min_scale, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __call__(self, img):
        """Return two augmented views of the same image"""
        return self.transform1(img), self.transform2(img)


def visualize_augmentations(dataset, n_samples=5):
    """
    Visualize how augmentations create positive pairs

    For each original image, we create 2 augmented views.
    These views look different but represent the SAME semantic content.
    """
    fig, axes = plt.subplots(n_samples, 3, figsize=(9, 2*n_samples))

    if n_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_samples):
        # Get original image
        img, _ = dataset[n_samples + i]

        # Apply SimCLR transforms
        transform = SimCLRTransform(input_size=32)
        view1, view2 = transform(img)

        # Convert to displayable format
        img_np = img.permute(1, 2, 0).numpy() * 0.5 + 0.5
        v1_np = view1.permute(1, 2, 0).numpy() * 0.5 + 0.5
        v2_np = view2.permute(1, 2, 0).numpy() * 0.5 + 0.5

        # Clip values
        img_np = np.clip(img_np, 0, 1)
        v1_np = np.clip(v1_np, 0, 1)
        v2_np = np.clip(v2_np, 0, 1)

        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title('Original')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(v1_np)
        axes[i, 1].set_title('View 1 (augmented)')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(v2_np)
        axes[i, 2].set_title('View 2 (augmented)')
        axes[i, 2].axis('off')

    plt.suptitle('SimCLR Data Augmentation: Creating Positive Pairs', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('simclr_augmentations.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("Saved augmentation visualization to 'simclr_augmentations.png'")


# ============================================================================
# PART 2: SIMCLR MODEL ARCHITECTURE
# ============================================================================

class ResNet18Encoder(nn.Module):
    """
    ResNet-18 backbone for feature extraction

    We remove the final classification layer to get representations.
    """
    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet18(weights=None)
        # Remove the final FC layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        """
        Args:
            x: Input images [B, 3, H, W]
        Returns:
            Features [B, 512] (after squeezing)
        """
        features = self.features(x)
        return features.squeeze(-1).squeeze(-1)


class ProjectionHead(nn.Module):
    """
    MLP Projection Head

    SimCLR uses a 2-layer MLP to project features into the space where
    contrastive loss is applied. This is REMOVED after pretraining!

    Why use it?
    - The projection space is where NT-Xent loss operates
    - The actual representations come from BEFORE this head
    - This helps the backbone learn better features
    """
    def __init__(self, input_dim=512, hidden_dim=512, output_dim=128):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.projection(x)


class SimCLR(nn.Module):
    """
    Complete SimCLR Model

    Architecture:
    1. Encoder (ResNet-18): Extract features from images
    2. Projection Head (MLP): Transform features for contrastive loss

    During inference, we only use the encoder!
    """
    def __init__(self, input_dim=512, hidden_dim=512, output_dim=128):
        super().__init__()
        self.encoder = ResNet18Encoder()
        self.projection = ProjectionHead(input_dim, hidden_dim, output_dim)

    def forward(self, x):
        """Forward pass through encoder and projection head"""
        features = self.encoder(x)
        projections = self.projection(features)
        return features, projections


# ============================================================================
# PART 3: NT-Xent CONTRASTIVE LOSS
# ============================================================================

class NTXentLoss(nn.Module):
    """
    NT-Xent Loss: Normalized Temperature-scaled Cross Entropy Loss

    This is the key innovation of SimCLR!

    INTUITION:
    ----------
    Given a batch of N images, we create 2N augmented views (2 per image).

    For each view i:
    - Positive: The other view from the SAME original image
    - Negatives: All other 2N-2 views

    The loss encourages:
    1. Positive pairs to have HIGH similarity (small angle between them)
    2. Negative pairs to have LOW similarity (large angle)

    MATHEMATICS:
    ------------
    For positive pair (i, j):

        loss(i,j) = -log[exp(sim(z_i, z_j)/τ) / Σ_{k≠i} exp(sim(z_i, z_k)/τ)]

    Where:
    - z_i, z_j = L2-normalized projections
    - sim(a,b) = cosine similarity = a·b / (||a||·||b||)
    - τ = temperature (typically 0.5)

    The denominator includes ALL other samples as negatives.

    WHY IT WORKS:
    -------------
    - Forces model to recognize same image under different augmentations
    - Must learn invariant features (shape, structure) not superficial ones (color, position)
    - Implicitly learns semantic similarity without labels!
    """

    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, projections):
        """
        Args:
            projections: [2N, D] - projected features from both views

        Returns:
            contrastive_loss: scalar
        """
        N = projections.shape[0] // 2  # Number of original images

        # L2 normalize the projections (required for cosine similarity)
        projections = F.normalize(projections, p=2, dim=1)

        # Compute similarity matrix [2N, 2N]
        # sim[i,j] = cosine similarity between projection i and j
        sim_matrix = torch.matmul(projections, projections.T)

        # Scale by temperature
        sim_matrix = sim_matrix / self.temperature

        # Create labels: for each view i, its positive pair is (i + N) % 2N
        # View 0 pairs with View N, View 1 pairs with View N+1, etc.
        labels = torch.arange(N, device=projections.device)
        labels = torch.cat([labels + N, labels])  # [0..N-1, N..2N-1] -> [N..2N-1, 0..N-1]

        # NT-Xent loss: cross entropy with similarity scores as logits
        # For each row i, we want the positive pair to have highest similarity
        loss = F.cross_entropy(sim_matrix, labels)

        return loss


def visualize_similarity_matrix(model, dataloader, device, n_images=64):
    """
    Visualize the similarity matrix to understand contrastive learning
    """
    model.eval()

    # Get a batch
    (view1, view2), _ = next(iter(dataloader))
    view1, view2 = view1[:n_images].to(device), view2[:n_images].to(device)

    with torch.no_grad():
        # Get projections for both views
        _, proj1 = model(view1)
        _, proj2 = model(view2)

        # Concatenate all projections
        all_proj = torch.cat([proj1, proj2], dim=0)
        all_proj = F.normalize(all_proj, p=2, dim=1)

        # Compute similarity matrix
        sim_matrix = torch.matmul(all_proj, all_proj.T).cpu().numpy()

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create mask to show structure
    N = n_images
    mask = np.zeros((2*N, 2*N), dtype=bool)
    for i in range(N):
        mask[i, i+N] = True
        mask[i+N, i] = True

    im = ax.imshow(sim_matrix, cmap='viridis', vmin=0, vmax=1)

    # Mark positive pairs
    for i in range(N):
        ax.plot(i+N, i, 'rx', markersize=15, markeredgewidth=2)
        ax.plot(i, i+N, 'rx', markersize=15, markeredgewidth=2)

    plt.colorbar(im, label='Cosine Similarity')
    plt.xlabel('Projection Index')
    plt.ylabel('Projection Index')
    plt.title(f'Similarity Matrix (2N={2*N} projections)\n'
              f'Red X marks positive pairs (same original image)', fontsize=12)
    plt.tight_layout()
    plt.savefig('simclr_similarity_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("Saved similarity matrix visualization to 'simclr_similarity_matrix.png'")

    # Print some statistics
    N = n_images
    positive_sims = [sim_matrix[i, i+N] for i in range(N)]
    negative_sims = [sim_matrix[i, j] for i in range(2*N) for j in range(2*N)
                     if i != j and abs(i - j) != N]

    print(f"\nSimilarity Statistics:")
    print(f"  Positive pairs (same image):  mean={np.mean(positive_sims):.3f}, "
          f"std={np.std(positive_sims):.3f}")
    print(f"  Negative pairs (diff image):  mean={np.mean(negative_sims):.3f}, "
          f"std={np.std(negative_sims):.3f}")
    print(f"  Separation: {np.mean(positive_sims) - np.mean(negative_sims):.3f}")


# ============================================================================
# PART 4: TRAINING LOOP
# ============================================================================

def train_simclr(model, dataloader, criterion, optimizer, scheduler,
                 device, n_epochs, save_interval=5):
    """
    Training loop for SimCLR

    Key differences from supervised learning:
    1. No labels needed! (self-supervised)
    2. Each image produces TWO views
    3. Loss pulls positive pairs together, pushes negatives apart
    """

    history = {
        'loss': [],
        'positive_sim': [],
        'negative_sim': []
    }

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        epoch_pos_sim = 0
        epoch_neg_sim = 0
        n_batches = 0

        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{n_epochs}')
        for (view1, view2), _ in pbar:
            view1, view2 = view1.to(device), view2.to(device)

            optimizer.zero_grad()

            # Forward pass through both views
            _, proj1 = model(view1)
            _, proj2 = model(view2)

            # Concatenate projections for contrastive loss
            projections = torch.cat([proj1, proj2], dim=0)

            # Compute NT-Xent loss
            loss = criterion(projections)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Track metrics
            epoch_loss += loss.item()

            # Track similarity for monitoring
            with torch.no_grad():
                all_proj = F.normalize(projections, p=2, dim=1)
                sim_matrix = torch.matmul(all_proj, all_proj.T)
                N = projections.shape[0] // 2

                # Positive pair similarities
                pos_sims = [sim_matrix[i, i+N].item() for i in range(N)]
                epoch_pos_sim += np.mean(pos_sims)

                # Sample negative similarities
                neg_sims = []
                for i in range(min(N, 10)):
                    for j in range(N):
                        if j != i:
                            neg_sims.append(sim_matrix[i, j+N].item())
                if neg_sims:
                    epoch_neg_sim += np.mean(neg_sims)

            n_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Average metrics
        avg_loss = epoch_loss / n_batches
        avg_pos_sim = epoch_pos_sim / n_batches
        avg_neg_sim = epoch_neg_sim / n_batches

        history['loss'].append(avg_loss)
        history['positive_sim'].append(avg_pos_sim)
        history['negative_sim'].append(avg_neg_sim)

        print(f'Epoch {epoch+1}: Loss={avg_loss:.4f}, '
              f'PosSim={avg_pos_sim:.3f}, NegSim={avg_neg_sim:.3f}')

        # Save checkpoint
        if (epoch + 1) % save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f'simclr_checkpoint_epoch{epoch+1}.pth')
            print(f'Saved checkpoint: simclr_checkpoint_epoch{epoch+1}.pth')

    return history


def plot_training_history(history):
    """Plot training history"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Loss curve
    axes[0].plot(history['loss'], 'b-', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('NT-Xent Loss')
    axes[0].set_title('Training Loss')
    axes[0].grid(True, alpha=0.3)

    # Positive similarity
    axes[1].plot(history['positive_sim'], 'g-', linewidth=2, label='Positive pairs')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Cosine Similarity')
    axes[1].set_title('Similarity of Positive Pairs\n(should increase)')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1)

    # Negative similarity
    axes[2].plot(history['negative_sim'], 'r-', linewidth=2, label='Negative pairs')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Cosine Similarity')
    axes[2].set_title('Similarity of Negative Pairs\n(should decrease)')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig('simclr_training_history.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("Saved training history to 'simclr_training_history.png'")


# ============================================================================
# PART 5: EVALUATION - LINEAR PROBING
# ============================================================================

def evaluate_representations(model, train_loader, test_loader, device, n_epochs=10):
    """
    Evaluate learned representations using linear probing

    After contrastive pretraining:
    1. Freeze the encoder
    2. Train a linear classifier on top
    3. Evaluate on test set

    This measures how GOOD the learned representations are!
    """

    print("\n" + "="*60)
    print("EVALUATION: Linear Probing")
    print("="*60)

    # Freeze encoder, train only linear classifier
    model.encoder.eval()
    for param in model.encoder.parameters():
        param.requires_grad = False

    # Create linear classifier
    classifier = nn.Linear(512, 10).to(device)  # CIFAR-10 has 10 classes
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(n_epochs):
        classifier.train()
        total_loss = 0
        correct = 0
        total = 0

        for (x1, x2), y in train_loader:
            x = torch.cat([x1, x2], dim=0).to(device)
            y = torch.cat([y, y], dim=0).to(device)

            optimizer.zero_grad()

            # Get features (no gradient through encoder)
            with torch.no_grad():
                features, _ = model(x)

            # Classify
            logits = classifier(features)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = logits.max(1).max(0)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

        train_acc = 100. * correct / total
        print(f'Epoch {epoch+1}/{n_epochs}: Loss={total_loss/len(train_loader):.4f}, '
              f'Train Acc={train_acc:.2f}%')

    # Evaluate on test set
    model.encoder.eval()
    classifier.eval()

    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            features, _ = model(x)
            logits = classifier(features)

            _, predicted = logits.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    test_acc = 100. * correct / total
    print(f'\nTest Accuracy: {test_acc:.2f}%')

    return test_acc, all_preds, all_labels


def plot_embedding_visualization(model, test_loader, device, n_samples=1000):
    """
    Visualize learned embeddings using PCA

    Good representations should:
    1. Cluster by class (even without labels during training!)
    2. Separate different classes
    """

    print("\nExtracting features for visualization...")

    model.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            features, _ = model(x)
            all_features.append(features.cpu())
            all_labels.append(y)

            if len(all_features) * features.shape[0] >= n_samples:
                break

    features = torch.cat(all_features, dim=0)[:n_samples].numpy()
    labels = torch.cat(all_labels, dim=0)[:n_samples].numpy()

    # PCA to 2D
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features)

    # Plot
    plt.figure(figsize=(10, 8))

    # CIFAR-10 class names
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    # Plot each class
    for class_idx in range(10):
        mask = labels == class_idx
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1],
                   alpha=0.5, s=20, label=class_names[class_idx])

    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.title('PCA Visualization of Learned Representations\n'
              'Colors = true labels (NOT used during training!)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig('simclr_embeddings_pca.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("Saved embedding visualization to 'simclr_embeddings_pca.png'")

    # Compute clustering quality (silhouette score)
    from sklearn.metrics import silhouette_score
    if len(np.unique(labels)) > 1:
        sil_score = silhouette_score(features, labels)
        print(f"Silhouette Score: {sil_score:.3f} (higher = better clustering)")


# ============================================================================
# PART 6: BASELINE COMPARISON - SUPERVISED TRAINING
# ============================================================================

def train_supervised_baseline(model, train_loader, test_loader, device, n_epochs=20):
    """
    Train the same architecture with supervised learning for comparison

    This shows the difference between:
    1. Contrastive pretraining (no labels) + linear probe
    2. End-to-end supervised training
    """

    print("\n" + "="*60)
    print("BASELINE: Supervised Training")
    print("="*60)

    # Reset model weights
    model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)

    # Full model training (encoder + classifier)
    classifier = nn.Linear(512, 10).to(device)
    params = list(model.parameters()) + list(classifier.parameters())
    optimizer = torch.optim.Adam(params, lr=0.001)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0

    for epoch in range(n_epochs):
        model.train()
        classifier.train()
        correct = 0
        total = 0

        for (x1, x2), y in train_loader:
            x = torch.cat([x1, x2], dim=0).to(device)
            y = torch.cat([y, y], dim=0).to(device)

            optimizer.zero_grad()

            features, _ = model(x)
            logits = classifier(features)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            _, predicted = logits.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

        # Evaluate
        model.eval()
        classifier.eval()
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                features, _ = model(x)
                logits = classifier(features)
                _, predicted = logits.max(1)
                test_total += y.size(0)
                test_correct += predicted.eq(y).sum().item()

        test_acc = 100. * test_correct / test_total
        best_acc = max(best_acc, test_acc)
        print(f'Epoch {epoch+1}/{n_epochs}: Train Acc={100.*correct/total:.2f}%, '
              f'Test Acc={test_acc:.2f}%')

    print(f'\nBest Test Accuracy: {best_acc:.2f}%')
    return best_acc


# ============================================================================
# MAIN: RUN THE TUTORIAL
# ============================================================================

def main():
    """
    Run the complete SimCLR tutorial
    """

    print("="*70)
    print("SimCLR: Contrastive Learning Tutorial")
    print("="*70)
    print("""
This tutorial demonstrates SimCLR, a contrastive learning approach.

KEY CONCEPTS:
1. Self-supervised: No labels needed during pretraining!
2. Positive pairs: Two augmented views of the same image
3. Negative pairs: Views from different images
4. NT-Xent Loss: Pull positives together, push negatives apart
5. Linear probing: Evaluate by training a classifier on frozen features
""")

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ========================================================================
    # Step 1: Load CIFAR-10 dataset
    # ========================================================================
    print("\n" + "="*60)
    print("STEP 1: Loading CIFAR-10 Dataset")
    print("="*60)

    # Standard transform for supervised baseline
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Download and load dataset
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )

    # For SimCLR, we need paired augmentations
    simclr_transform = SimCLRTransform(input_size=32)

    class SimCLRCIFAR10(torch.utils.data.Dataset):
        """Wrapper dataset that applies SimCLR transforms"""
        def __init__(self, base_dataset):
            self.base_dataset = base_dataset

        def __getitem__(self, idx):
            img, label = self.base_dataset[idx]
            return simclr_transform(img), label

        def __len__(self):
            return len(self.base_dataset)

    simclr_train_dataset = SimCLRCIFAR10(train_dataset)

    # DataLoaders - NOTE: Large batch size important for contrastive learning!
    batch_size = 256  # Larger batches = more negatives = better learning

    train_loader = DataLoader(simclr_train_dataset, batch_size=batch_size,
                             shuffle=True, num_workers=2, drop_last=True)

    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=2)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Batch size: {batch_size}")

    # ========================================================================
    # Step 2: Visualize augmentations
    # ========================================================================
    print("\n" + "="*60)
    print("STEP 2: Visualizing Data Augmentations")
    print("="*60)

    visualize_augmentations(train_dataset, n_samples=5)

    # ========================================================================
    # Step 3: Initialize model
    # ========================================================================
    print("\n" + "="*60)
    print("STEP 3: Initializing SimCLR Model")
    print("="*60)

    model = SimCLR(input_dim=512, hidden_dim=512, output_dim=128).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    projection_params = sum(p.numel() for p in model.projection.parameters())

    print(f"Total parameters: {total_params:,}")
    print(f"Encoder parameters: {encoder_params:,}")
    print(f"Projection head parameters: {projection_params:,}")

    # ========================================================================
    # Step 4: Train with SimCLR (contrastive learning)
    # ========================================================================
    print("\n" + "="*60)
    print("STEP 4: Contrastive Pretraining with SimCLR")
    print("="*60)
    print("""
Training without any labels! The model learns by:
1. Comparing augmented views of the same image (positives)
2. Contrasting with views from different images (negatives)
3. Using NT-Xent loss to organize the embedding space
""")

    criterion = NTXentLoss(temperature=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    # For demo purposes, train for fewer epochs
    # In practice, train for 100+ epochs
    n_epochs = 20  # Increase to 50-100 for better results

    history = train_simclr(model, train_loader, criterion, optimizer,
                          scheduler, device, n_epochs)

    plot_training_history(history)

    # Visualize similarity matrix
    print("\nVisualizing similarity matrix...")
    visualize_similarity_matrix(model, train_loader, device, n_images=32)

    # ========================================================================
    # Step 5: Evaluate representations (linear probing)
    # ========================================================================
    print("\n" + "="*60)
    print("STEP 5: Evaluating Learned Representations")
    print("="*60)

    # Note: Reload test dataset without SimCLR transform for evaluation
    test_dataset_eval = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )

    class FeatureDataset(torch.utils.data.Dataset):
        """Dataset for extracting features"""
        def __init__(self, base_dataset, model, device):
            self.model = model
            self.device = device
            self.features = []
            self.labels = []

            loader = DataLoader(base_dataset, batch_size=256, shuffle=False)
            model.eval()

            with torch.no_grad():
                for x, y in loader:
                    x = x.to(device)
                    features, _ = model(x)
                    self.features.append(features.cpu())
                    self.labels.append(y)

            self.features = torch.cat(self.features, dim=0)
            self.labels = torch.cat(self.labels, dim=0)

        def __getitem__(self, idx):
            return self.features[idx], self.labels[idx]

        def __len__(self):
            return len(self.labels)

    # Create feature datasets for linear probing
    print("Extracting features for linear probing...")
    feature_train = FeatureDataset(train_dataset, model, device)
    feature_test = FeatureDataset(test_dataset_eval, model, device)

    feature_train_loader = DataLoader(feature_train, batch_size=256, shuffle=True)
    feature_test_loader = DataLoader(feature_test, batch_size=256, shuffle=False)

    # Linear probe evaluation
    test_acc, preds, labels = evaluate_representations(
        model, feature_train_loader, feature_test_loader, device, n_epochs=20
    )

    # Visualize embeddings
    plot_embedding_visualization(model, test_loader, device, n_samples=1000)

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*70)
    print("TUTORIAL SUMMARY")
    print("="*70)
    print("""
SimCLR demonstrates the power of contrastive learning:

1. NO LABELS NEEDED: Learns representations from unlabeled data
   - Only needs images and augmentations

2. CONTRASTIVE OBJECTIVE: NT-Xent loss
   - Pulls augmented views of same image together
   - Pushes views of different images apart

3. LARGE BATCH SIZES: More negatives = better learning
   - Typical: 256-512 batch size

4. STRONG AUGMENTATIONS: Key to good representations
   - Random crop, color jitter, blur, grayscale

5. PROJECTION HEAD: Removed after training
   - Only the encoder is kept for downstream tasks

6. LINEAR PROBING: Evaluates representation quality
   - Freeze encoder, train linear classifier
   - Good features = good linear probe accuracy

KEY INSIGHT:
The model learns SEMANTIC features (shape, object parts) because
superficial features (color, position) change across augmentations.
This is why contrastive learning works without labels!
""")

    print("\nGenerated files:")
    print("  - simclr_augmentations.png: Shows augmentation pairs")
    print("  - simclr_training_history.png: Training curves")
    print("  - simclr_similarity_matrix.png: Learned similarities")
    print("  - simclr_embeddings_pca.png: PCA visualization of embeddings")

    return model, history


if __name__ == "__main__":
    model, history = main()
