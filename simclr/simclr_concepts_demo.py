"""
SimCLR Concepts - Quick Educational Demo
=========================================

This demo quickly shows the KEY CONCEPTS of contrastive learning
without requiring long training times.

Run this first to understand the concepts, then try simclr_demo.py
for full training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

print("="*60)
print("SimCLR: Contrastive Learning Concepts Demo")
print("="*60)


# ============================================================================
# CONCEPT 1: Positive Pairs via Augmentation
# ============================================================================

print("\n" + "="*60)
print("CONCEPT 1: Creating Positive Pairs")
print("="*60)
print("""
In SimCLR, we create TWO augmented views of each image.
These two views form a POSITIVE PAIR.

The augmentations include:
- Random crop and resize
- Color jitter (brightness, contrast, saturation)
- Random grayscale
- Horizontal flip
- Gaussian blur

The KEY: These views look DIFFERENT but represent the SAME object.
""")

# Load CIFAR-10
train_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=None
)

# Define augmentation transform
augment_transform = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Visualize positive pairs
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

fig, axes = plt.subplots(5, 3, figsize=(10, 12))

for i in range(5):
    img, label = train_dataset[i + 20]

    # Create two augmented views
    view1 = augment_transform(img)
    view2 = augment_transform(img)

    # Convert for display
    orig_np = np.array(img)
    v1_np = view1.permute(1, 2, 0).numpy() * 0.5 + 0.5
    v2_np = view2.permute(1, 2, 0).numpy() * 0.5 + 0.5

    v1_np = np.clip(v1_np, 0, 1)
    v2_np = np.clip(v2_np, 0, 1)

    axes[i, 0].imshow(orig_np)
    axes[i, 0].set_title(f'Original: {class_names[label]}')
    axes[i, 0].axis('off')

    axes[i, 1].imshow(v1_np)
    axes[i, 1].set_title('Augmented View 1')
    axes[i, 1].axis('off')

    axes[i, 2].imshow(v2_np)
    axes[i, 2].set_title('Augmented View 2')
    axes[i, 2].axis('off')

plt.suptitle('Positive Pairs: Same Object, Different Augmentations\n'
             'The model must learn to recognize these as the SAME image',
             fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig('simclr_concept_1_positive_pairs.png', dpi=150, bbox_inches='tight')
plt.show()

print("Visualization saved to 'simclr_concept_1_positive_pairs.png'")


# ============================================================================
# CONCEPT 2: Contrastive Loss (NT-Xent)
# ============================================================================

print("\n" + "="*60)
print("CONCEPT 2: NT-Xent Contrastive Loss")
print("="*60)
print("""
The NT-Xent loss is the core of SimCLR.

For a batch of N images with 2N augmented views:

For each positive pair (i, j):
                    exp(sim(z_i, z_j)/τ)
    loss = -log --------------------------
              Σ_k≠i exp(sim(z_i, z_k)/τ)

Where:
- z_i, z_j = embeddings of positive pair
- τ = temperature (typically 0.5)
- sim = cosine similarity
- Denominator sums over ALL other samples (negatives)

VISUALIZATION:

Batch of 4 images → 8 embeddings

  view1_0  view1_1  view1_2  view1_3  view2_0  view2_1  view2_2  view2_3
     |        |        |        |        |        |        |        |
     +--------+--------+--------+--------+--------+--------+--------+
                          SIMILARITY MATRIX (8x8)

  Positive pairs: (0,4), (1,5), (2,6), (3,7)  ← same original image
  Negative pairs: all other combinations      ← different images

  Goal: Maximize positive similarity, minimize negative similarity
""")


class NTXentLoss(nn.Module):
    """NT-Xent: Normalized Temperature-scaled Cross Entropy Loss"""

    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        N = z_i.shape[0]

        # Normalize embeddings
        z_i = F.normalize(z_i, p=2, dim=1)
        z_j = F.normalize(z_j, p=2, dim=1)

        # Concatenate: [2N, D]
        all_z = torch.cat([z_i, z_j], dim=0)

        # Similarity matrix: [2N, 2N]
        sim_matrix = torch.matmul(all_z, all_z.T) / self.temperature

        # Labels: positive pair for i is at (i + N) % 2N
        labels = torch.cat([torch.arange(N, device=z_i.device) + N,
                            torch.arange(N, device=z_i.device)])

        return F.cross_entropy(sim_matrix, labels)


# Demonstrate the loss
print("\nDemonstrating NT-Xent Loss:")
print("-" * 40)

criterion = NTXentLoss(temperature=0.5)

# Case 1: Random embeddings (no learning)
z1_random = torch.randn(16, 64)
z2_random = torch.randn(16, 64)
loss_random = criterion(z1_random, z2_random)
print(f"Random embeddings loss: {loss_random.item():.4f}")

# Case 2: Similar positive pairs (good)
z1_good = torch.randn(16, 64)
z2_good = z1_good + 0.1 * torch.randn(16, 64)  # Small noise
loss_good = criterion(z1_good, z2_good)
print(f"Similar pairs loss: {loss_good.item():.4f}")

# Case 3: Identical positive pairs (perfect)
z1_perfect = torch.randn(16, 64)
z2_perfect = z1_perfect.clone()  # Identical
loss_perfect = criterion(z1_perfect, z2_perfect)
print(f"Identical pairs loss: {loss_perfect.item():.4f}")

print("\nObservation: Loss decreases as positive pairs become more similar!")


# ============================================================================
# CONCEPT 3: Similarity Matrix Visualization
# ============================================================================

print("\n" + "="*60)
print("CONCEPT 3: Visualizing the Similarity Matrix")
print("="*60)

# Create example similarity matrix
N = 16
z1 = torch.randn(N, 64)
z2 = z1 + 0.3 * torch.randn(N, 64)  # Some noise

z1_norm = F.normalize(z1, p=2, dim=1)
z2_norm = F.normalize(z2, p=2, dim=1)

all_z = torch.cat([z1_norm, z2_norm], dim=0)
sim_matrix = torch.matmul(all_z, all_z.T).numpy()

# Plot
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(sim_matrix, cmap='RdYlGn', vmin=-0.2, vmax=1.0)

# Mark positive pairs
for i in range(N):
    ax.plot(i + N, i, 'x', color='blue', markersize=15, markeredgewidth=2)
    ax.plot(i, i + N, 'x', color='blue', markersize=15, markeredgewidth=2)

plt.colorbar(im, label='Cosine Similarity')
ax.set_xlabel('Embedding Index')
ax.set_ylabel('Embedding Index')
ax.set_title('Similarity Matrix\n'
             'Blue X = positive pairs (same original image)\n'
             'Green = high similarity, Red = low similarity')

plt.tight_layout()
plt.savefig('simclr_concept_2_similarity_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

print("Visualization saved to 'simclr_concept_2_similarity_matrix.png'")

# Print statistics
positive_sims = [sim_matrix[i, i + N] for i in range(N)]
negative_sims = [sim_matrix[i, j] for i in range(2*N) for j in range(2*N)
                 if i != j and abs(i - j) != N]

print(f"\nSimilarity Statistics:")
print(f"  Positive pairs (same image): mean={np.mean(positive_sims):.3f}")
print(f"  Negative pairs (diff image): mean={np.mean(negative_sims):.3f}")
print(f"  Separation: {np.mean(positive_sims) - np.mean(negative_sims):.3f}")


# ============================================================================
# CONCEPT 4: Architecture
# ============================================================================

print("\n" + "="*60)
print("CONCEPT 4: SimCLR Architecture")
print("="*60)
print("""
SimCLR uses a two-component architecture:

                    ┌─────────────────┐
    Image ─────────>│    Encoder      │──> Representation (512-D)
                    │   (ResNet-18)   │        ↑
                    └─────────────────┘        │
                                               │ (keep for downstream)
                    ┌─────────────────┐        │
    Image ─────────>│    Encoder      │──> Representation (512-D)
                    │   (ResNet-18)   │        │
                    └─────────────────┘        │
                                               ↓
                                      ┌─────────────────┐
                                      │ Projection Head │──> Embedding (128-D)
                                      │   (2-layer MLP) │        ↓
                                      └─────────────────┘        │
                                                                 │
                                                                 ↓
                                                          NT-Xent Loss

KEY INSIGHT: The projection head is REMOVED after training!
- Only the encoder is kept for downstream tasks
- The projection head helps learn better representations
- Final features come from BEFORE the projection head
""")


class SimCLR(nn.Module):
    """SimCLR Model"""

    def __init__(self, input_dim=512, hidden_dim=512, output_dim=128):
        super().__init__()

        # Encoder
        resnet = torchvision.models.resnet18(weights=None)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])

        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        representation = self.encoder(x).squeeze(-1).squeeze(-1)
        embedding = self.projection(representation)
        return representation, embedding


model = SimCLR()

# Count parameters
total = sum(p.numel() for p in model.parameters())
encoder_params = sum(p.numel() for p in model.encoder.parameters())
projection_params = sum(p.numel() for p in model.projection.parameters())

print(f"Parameter counts:")
print(f"  Total: {total:,}")
print(f"  Encoder: {encoder_params:,} (kept after training)")
print(f"  Projection: {projection_params:,} (discarded after training)")


# ============================================================================
# CONCEPT 5: Linear Probing Evaluation
# ============================================================================

print("\n" + "="*60)
print("CONCEPT 5: Linear Probing")
print("="*60)
print("""
After contrastive pretraining, how do we evaluate the representations?

LINEAR PROBING:
1. Freeze the encoder (no gradient updates)
2. Train a simple linear classifier on top
3. Evaluate on test set

Why this works:
- If representations are good, a linear classifier should work well
- If representations are bad, even a complex classifier will struggle

This measures representation quality without fine-tuning!

CONTRASTIVE vs SUPERVISED:

┌────────────────────────────────────────────────────────────────┐
│ Supervised Training                                            │
│ Image → [Encoder] → Features → [Classifier] → Loss → Update all│
│                        ↑                    ↑                  │
│                        └──── Both updated ──┘                  │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ Contrastive + Linear Probe                                     │
│ Phase 1 (Contrastive):                                         │
│ Image → [Encoder] → Features → [Projection] → Loss → Update all│
│                                                                 │
│ Phase 2 (Linear Probe):                                        │
│ Image → [Encoder] → Features → [Classifier] → Loss             │
│         ↑ (FROZEN!)                          ↑ Update only     │
└────────────────────────────────────────────────────────────────┘
""")


# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*60)
print("SUMMARY: Key Concepts of SimCLR")
print("="*60)
print("""
1. POSITIVE PAIRS
   - Two augmented views of the same image
   - Model must recognize them as semantically identical

2. NT-XENT LOSS
   - Pulls positive pairs together in embedding space
   - Pushes negative pairs apart
   - Uses cosine similarity with temperature scaling

3. ARCHITECTURE
   - Encoder (ResNet): Extracts features
   - Projection Head (MLP): Transforms for contrastive loss
   - Projection head is REMOVED after training

4. LARGE BATCH SIZES
   - More negative samples per batch = better learning
   - Typical: 256-512 batch size

5. LINEAR PROBING
   - Evaluate by training linear classifier on frozen encoder
   - Measures representation quality

WHY IT WORKS:
- Must learn INVARIANT features (shape, structure)
- Ignores superficial features (exact color, position)
- These invariant features transfer to downstream tasks!

NEXT STEPS:
- Run simclr_demo.py for full training demonstration
- Run simclr_tutorial.ipynb for interactive Jupyter notebook
""")

print("\nGenerated files:")
print("  - simclr_concept_1_positive_pairs.png: Augmentation examples")
print("  - simclr_concept_2_similarity_matrix.png: Similarity visualization")

print("\nDemo complete!")
