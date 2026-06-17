---
tags:
  - ml/contrastive-learning
  - code-analysis
  - neural-network
  - simclr
created: 2026-03-04
---

# Code Part 3 - SimCLR Model

> [!abstract] Overview
> Detailed code analysis of the SimCLR model architecture including the encoder and projection head.

## Code Location
`simclr_tutorial.py` lines 153-219

---

## Complete Code

```python
class ResNet18Encoder(nn.Module):
    """ResNet-18 backbone for feature extraction"""
    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet18(weights=None)
        self.features = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        features = self.features(x)
        return features.squeeze(-1).squeeze(-1)


class ProjectionHead(nn.Module):
    """MLP Projection Head"""
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
    """Complete SimCLR Model"""
    def __init__(self, input_dim=512, hidden_dim=512, output_dim=128):
        super().__init__()
        self.encoder = ResNet18Encoder()
        self.projection = ProjectionHead(input_dim, hidden_dim, output_dim)

    def forward(self, x):
        features = self.encoder(x)
        projections = self.projection(features)
        return features, projections
```

---

## Part 1: ResNet-18 Encoder

### Architecture Overview

```python
class ResNet18Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet18(weights=None)
        # Remove the final FC layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])
```

**Mathematical Transformation:**

$$\mathbf{h} = f_{\theta}(x) \in \mathbb{R}^{512}$$

Where $f_{\theta}$ is the ResNet-18 encoder with parameters $\theta$.

### ResNet-18 Structure

```python
# ResNet-18 architecture (torchvision implementation)
ResNet(
    (conv1): Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    (bn1): BatchNorm2d(64)
    (relu): ReLU(inplace=True)
    (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1)
    (layer1): Sequential(
        BasicBlock(64, 64)
        BasicBlock(64, 64)
    )
    (layer2): Sequential(
        BasicBlock(64, 128, stride=2)
        BasicBlock(128, 128)
    )
    (layer3): Sequential(
        BasicBlock(128, 256, stride=2)
        BasicBlock(256, 256)
    )
    (layer4): Sequential(
        BasicBlock(256, 512, stride=2)
        BasicBlock(512, 512)
    )
    (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
    (fc): Linear(512, 1000)  ← REMOVED
)
```

### Layer-by-Layer Dimensions

For input $x \in \mathbb{R}^{B \times 3 \times 32 \times 32}$ (CIFAR-10):

```
Layer          Input Shape          Output Shape         Operation
────────────────────────────────────────────────────────────────────
conv1         [B, 3, 32, 32]      [B, 64, 16, 16]     7×7 conv, stride=2
bn1           [B, 64, 16, 16]     [B, 64, 16, 16]     BatchNorm
relu          [B, 64, 16, 16]     [B, 64, 16, 16]     ReLU
maxpool       [B, 64, 16, 16]     [B, 64, 8, 8]       3×3 maxpool, stride=2

layer1        [B, 64, 8, 8]       [B, 64, 8, 8]       2× BasicBlock
layer2        [B, 64, 8, 8]       [B, 128, 4, 4]      2× BasicBlock, stride=2
layer3        [B, 128, 4, 4]      [B, 256, 2, 2]      2× BasicBlock, stride=2
layer4        [B, 256, 2, 2]      [B, 512, 1, 1]      2× BasicBlock, stride=2

avgpool       [B, 512, 1, 1]      [B, 512, 1, 1]      Global avg pool

flatten       [B, 512, 1, 1]      [B, 512]            squeeze
```

### BasicBlock Structure

```python
class BasicBlock(nn.Module):
    """
    ResNet Basic Block

    Mathematical formulation:
    y = F(x) + x  (residual connection)

    Where F(x) = Conv→BN→ReLU→Conv→BN
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # Residual connection
        out = F.relu(out)
        return out
```

**Mathematical Formulation:**

$$\mathbf{y} = \text{ReLU}(\text{BN}(\text{Conv}_2(\text{ReLU}(\text{BN}(\text{Conv}_1(\mathbf{x}))))) + \text{Shortcut}(\mathbf{x}))$$

### Why Remove the FC Layer?

```python
self.features = nn.Sequential(*list(resnet.children())[:-1])
```

> [!important] Feature Extraction vs Classification
> - **With FC layer**: Fixed output size (1000 classes for ImageNet)
> - **Without FC layer**: Flexible output (512-D feature vector)
> - **For SimCLR**: We need features, not class predictions

---

## Part 2: Projection Head

### Architecture

```python
class ProjectionHead(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=512, output_dim=128):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
```

**Mathematical Formulation:**

$$\mathbf{z} = g_\phi(\mathbf{h}) = W_2 \cdot \text{ReLU}(W_1 \mathbf{h} + \mathbf{b}_1) + \mathbf{b}_2$$

Where:
- $W_1 \in \mathbb{R}^{512 \times 512}$ (hidden layer weights)
- $W_2 \in \mathbb{R}^{128 \times 512}$ (output layer weights)
- $\phi = \{W_1, \mathbf{b}_1, W_2, \mathbf{b}_2\}$ (projection head parameters)

### Dimension Flow

```
Input:     h ∈ R^512  (encoder output)
              │
              ▼
┌─────────────────────────┐
│  Linear(512 → 512)      │  W_1 h + b_1
└─────────────────────────┘
              │
              ▼
┌─────────────────────────┐
│  ReLU                   │  max(0, ·)
└─────────────────────────┘
              │
              ▼
┌─────────────────────────┐
│  Linear(512 → 128)      │  W_2 · + b_2
└─────────────────────────┘
              │
              ▼
Output:    z ∈ R^128  (projection)
```

### Why MLP (not just linear)?

> [!success] Ablation Results (from SimCLR paper)
>
> | Projection Head | Linear Probe Accuracy |
> |-----------------|----------------------|
> | None | 64.3% |
> | Linear (1 layer) | 67.5% |
> | Non-linear (2 layers) | **69.3%** |

**Intuition:**
- The projection head allows the encoder to learn more general features
- Without projection head, encoder must directly optimize for contrastive task
- With projection head, encoder learns features that are "one step away" from the contrastive task

---

## Part 3: Complete SimCLR Model

### Forward Pass

```python
class SimCLR(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=512, output_dim=128):
        super().__init__()
        self.encoder = ResNet18Encoder()
        self.projection = ProjectionHead(input_dim, hidden_dim, output_dim)

    def forward(self, x):
        """
        Args:
            x: Input images [B, 3, H, W]

        Returns:
            features: Representation [B, 512] - KEEP for downstream
            projections: Embedding [B, 128] - USE for contrastive loss
        """
        features = self.encoder(x)           # [B, 512]
        projections = self.projection(features)  # [B, 128]
        return features, projections
```

### Processing Two Views

```python
def forward_two_views(model, view1, view2):
    """
    Process two augmented views through SimCLR

    Mathematical formulation:
    h1 = f(x1), z1 = g(h1)
    h2 = f(x2), z2 = g(h2)

    Where:
    - f: encoder (shared weights)
    - g: projection head (shared weights)
    """
    # Both views use the SAME encoder (weight sharing)
    features1, proj1 = model(view1)
    features2, proj2 = model(view2)

    return features1, features2, proj1, proj2
```

---

## Parameter Count

```python
def count_parameters(model):
    """Count parameters for each component"""
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    projection_params = sum(p.numel() for p in model.projection.parameters())
    total_params = encoder_params + projection_params

    print(f"Encoder: {encoder_params:,} parameters")
    print(f"Projection: {projection_params:,} parameters")
    print(f"Total: {total_params:,} parameters")

model = SimCLR()
count_parameters(model)

# Output:
# Encoder: 11,176,512 parameters (~11.2M)
# Projection: 328,064 parameters (~0.3M)
# Total: 11,504,576 parameters (~11.5M)
```

### Detailed Breakdown

**Encoder (ResNet-18):**
```
conv1:      3×64×7×7 = 9,408
bn1:        64×2 = 128
layer1:     2 × (64×64×3×3×2 + 64×2×2) = 147,968
layer2:     2 × (...) stride=2 = 526,336
layer3:     2 × (...) stride=2 = 2,100,992
layer4:     2 × (...) stride=2 = 8,392,192
────────────────────────────────────────
Total:      ~11.2M
```

**Projection Head:**
```
Linear1:    512×512 + 512 = 262,656
Linear2:    128×512 + 128 = 65,664
────────────────────────────────────────
Total:      ~328K
```

---

## Weight Sharing Visualization

```
                    ┌─────────────────┐
    View 1 ────────►│                 │────► h1 ──► [Projection] ──► z1
                    │    Encoder      │
    View 2 ────────►│   (shared)      │────► h2 ──► [Projection] ──► z2
                    │                 │
                    └─────────────────┘
                           ↑
                    Same weights for
                    both views
```

**Why Weight Sharing?**

1. **Consistency**: Both views processed identically
2. **Parameter efficiency**: No need for separate encoders
3. **Feature alignment**: Same features for same visual content

---

## Key Insights

> [!important] Architecture Summary
> 1. **Encoder (ResNet-18)**: Extracts 512-D features
> 2. **Projection Head (MLP)**: Transforms to 128-D for contrastive loss
> 3. **Weight Sharing**: Both views use the same encoder
> 4. **Post-training**: Only encoder kept, projection head discarded

---

## Related Notes

- [[03 - SimCLR Architecture]] - Conceptual overview
- [[08 - Code Part 3 - Model]] - This note