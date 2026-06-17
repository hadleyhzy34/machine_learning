---
tags:
  - ml/contrastive-learning
  - neural-network
  - architecture
  - simclr
created: 2026-03-04
---

# SimCLR Architecture

> [!abstract] Overview
> SimCLR uses a two-component architecture: an **encoder** for feature extraction and a **projection head** for contrastive learning. The projection head is discarded after training!

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     SimCLR Architecture                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Image x ──┐                                                   │
│             │    ┌──────────────┐    ┌──────────────────┐       │
│             └───►│   Encoder    │───►│ Representation   │       │
│                  │  (ResNet-18) │    │     (512-D)      │       │
│                  └──────────────┘    └────────┬─────────┘       │
│                                               │                 │
│                                               │ KEEP            │
│                                               │ (for downstream)│
│                                               ↓                 │
│                                   ┌──────────────────┐          │
│                                   │ Projection Head  │          │
│                                   │   (2-layer MLP)  │          │
│                                   └────────┬─────────┘          │
│                                            │                    │
│                                            ↓                    │
│                                   ┌──────────────────┐          │
│                                   │   Embedding      │          │
│                                   │    (128-D)       │          │
│                                   └────────┬─────────┘          │
│                                            │                    │
│                                            │ DISCARD            │
│                                            │ (after training)   │
│                                            ↓                    │
│                                     NT-Xent Loss                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Component 1: Encoder

### Purpose

The encoder $f(\cdot)$ extracts meaningful features from images:

$$
\mathbf{h} = f(x) \in \mathbb{R}^d
$$

### ResNet-18 Backbone

```python
class ResNet18Encoder(nn.Module):
    """
    ResNet-18 backbone for feature extraction.

    We remove the final classification layer to get representations.
    Output: 512-dimensional feature vector
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
            Features [B, 512]
        """
        features = self.features(x)  # [B, 512, 1, 1]
        return features.squeeze(-1).squeeze(-1)  # [B, 512]
```

### ResNet-18 Architecture Details

```
Input: [B, 3, 32, 32]
    │
    ├── Conv1 (7×7, 64) + BN + ReLU
    │   Output: [B, 64, 16, 16]
    │
    ├── MaxPool (3×3)
    │   Output: [B, 64, 8, 8]
    │
    ├── Layer1 (2× BasicBlock, 64 channels)
    │   Output: [B, 64, 8, 8]
    │
    ├── Layer2 (2× BasicBlock, 128 channels)
    │   Output: [B, 128, 4, 4]
    │
    ├── Layer3 (2× BasicBlock, 256 channels)
    │   Output: [B, 256, 2, 2]
    │
    ├── Layer4 (2× BasicBlock, 512 channels)
    │   Output: [B, 512, 1, 1]
    │
    └── AdaptiveAvgPool
        Output: [B, 512]
```

---

## Component 2: Projection Head

### Purpose

The projection head $g(\cdot)$ transforms features into the space where contrastive loss is applied:

$$
\mathbf{z} = g(\mathbf{h}) \in \mathbb{R}^{d'}
$$

> [!important] Key Insight
> The projection head is **DISCARDED** after training! We only use the encoder for downstream tasks.

### Why Use a Projection Head?

> [!success] Three Reasons
> 1. **Feature space separation**: The projection space can be optimized for contrastive learning while the representation space remains general
> 2. **Non-linearity**: MLP can transform features non-linearly, improving the contrastive task
> 3. **Dimensionality**: Lower dimension (128-D vs 512-D) makes similarity computation more efficient

### Implementation

```python
class ProjectionHead(nn.Module):
    """
    MLP Projection Head

    Architecture: Linear → ReLU → Linear
    Input: 512-D features from encoder
    Output: 128-D embeddings for contrastive loss

    This is REMOVED after pretraining!
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
```

### Mathematical Formulation

$$
g(\mathbf{h}) = W_2 \cdot \text{ReLU}(W_1 \mathbf{h} + \mathbf{b}_1) + \mathbf{b}_2
$$

Where:
- $W_1 \in \mathbb{R}^{512 \times 512}$ (hidden layer)
- $W_2 \in \mathbb{R}^{128 \times 512}$ (output layer)

---

## Complete SimCLR Model

```python
class SimCLR(nn.Module):
    """
    Complete SimCLR Model

    Two stages:
    1. Pretraining: encoder + projection head trained with NT-Xent
    2. Downstream: only encoder kept, projection head discarded
    """

    def __init__(self, input_dim=512, hidden_dim=512, output_dim=128):
        super().__init__()
        self.encoder = ResNet18Encoder()
        self.projection = ProjectionHead(input_dim, hidden_dim, output_dim)

    def forward(self, x):
        """
        Forward pass through encoder and projection head

        Args:
            x: Input images [B, 3, H, W]

        Returns:
            features: Representation [B, 512] - KEEP for downstream
            projections: Embedding [B, 128] - USE for contrastive loss
        """
        features = self.encoder(x)        # [B, 512]
        projections = self.projection(features)  # [B, 128]
        return features, projections
```

---

## Parameter Count

```python
model = SimCLR()

# Count parameters
total = sum(p.numel() for p in model.parameters())
encoder_params = sum(p.numel() for p in model.encoder.parameters())
projection_params = sum(p.numel() for p in model.projection.parameters())

print(f"Total: {total:,}")           # ~11.7M
print(f"Encoder: {encoder_params:,}")  # ~11.2M (kept)
print(f"Projection: {projection_params:,}")  # ~0.5M (discarded)
```

---

## Two Views Processing

For contrastive learning, we process two augmented views:

```python
def forward_batch(model, view1, view2):
    """
    Process two augmented views through SimCLR

    Args:
        model: SimCLR model
        view1: First augmented view [B, 3, H, W]
        view2: Second augmented view [B, 3, H, W]

    Returns:
        features1, features2: Representations for each view
        proj1, proj2: Projections for contrastive loss
    """
    # Process both views through encoder
    features1, proj1 = model(view1)
    features2, proj2 = model(view2)

    # Concatenate projections for loss computation
    projections = torch.cat([proj1, proj2], dim=0)  # [2B, 128]

    return features1, features2, projections
```

---

## Why This Architecture Works

### Encoder-Projection Separation

> [!success] Information Bottleneck Theory
> The projection head creates an **information bottleneck**:
> - Forces the encoder to learn features that are **transferable**
> - Prevents the encoder from learning features only useful for contrastive task
> - Similar to how autoencoder bottleneck forces meaningful compression

### Ablation Results (from paper)

| Projection Head | CIFAR-10 Accuracy |
|-----------------|-------------------|
| None (direct) | ~68% |
| Linear (1 layer) | ~72% |
| Non-linear (2 layers) | **~76%** |

---

## Alternative Architectures

### SimCLR with Different Encoders

| Encoder | Params | ImageNet Accuracy |
|---------|--------|-------------------|
| ResNet-18 | 11M | ~60% |
| ResNet-50 | 25M | ~69% |
| ResNet-101 | 44M | ~71% |
| ResNet-152 | 60M | ~72% |

### Projection Head Variants

```python
# Variant 1: No projection (baseline)
class NoProjection(nn.Module):
    def forward(self, x):
        return x  # Identity

# Variant 2: Linear projection
class LinearProjection(nn.Module):
    def __init__(self, dim=512, out_dim=128):
        super().__init__()
        self.proj = nn.Linear(dim, out_dim)
    def forward(self, x):
        return self.proj(x)

# Variant 3: Non-linear (SimCLR default)
class NonlinearProjection(nn.Module):
    def __init__(self, dim=512, hidden=512, out_dim=128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, x):
        return self.proj(x)
```

---

## Key Insights

> [!important] Remember
> 1. **Encoder**: Learns general-purpose features (keep for downstream)
> 2. **Projection Head**: Helps contrastive learning (discard after training)
> 3. **Shared Weights**: Both views use the SAME encoder (weight sharing)

---

## Related Notes

- [[02 - NT-Xent Loss]] - How projections are used
- [[08 - Code Part 3 - Model]] - Detailed code walkthrough
- [[05 - Linear Probing Evaluation]] - How to use encoder for downstream tasks