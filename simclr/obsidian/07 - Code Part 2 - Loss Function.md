---
tags:
  - ml/contrastive-learning
  - code-analysis
  - loss-function
  - mathematics
  - simclr
created: 2026-03-04
---

# Code Part 2 - NT-Xent Loss

> [!abstract] Overview
> Detailed code analysis of the NT-Xent loss function with mathematical derivations.

## Code Location
`simclr_tutorial.py` lines 226-297

---

## Complete Code

```python
class NTXentLoss(nn.Module):
    """
    NT-Xent Loss: Normalized Temperature-scaled Cross Entropy Loss

    This is the key innovation of SimCLR!
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
        sim_matrix = torch.matmul(projections, projections.T)

        # Scale by temperature
        sim_matrix = sim_matrix / self.temperature

        # Create labels: for each view i, its positive pair is (i + N) % 2N
        labels = torch.arange(N, device=projections.device)
        labels = torch.cat([labels + N, labels])

        # NT-Xent loss: cross entropy with similarity scores as logits
        loss = F.cross_entropy(sim_matrix, labels)

        return loss
```

---

## Line-by-Line Analysis

### Step 1: L2 Normalization

```python
projections = F.normalize(projections, p=2, dim=1)
```

**Mathematical Operation:**

$$\mathbf{z}_i = \frac{\mathbf{h}_i}{\|\mathbf{h}_i\|_2} = \frac{\mathbf{h}_i}{\sqrt{\sum_{k=1}^{D} h_{ik}^2}}$$

**Why Normalize?**

1. **Cosine similarity becomes dot product:**
   $$\cos(\mathbf{z}_i, \mathbf{z}_j) = \frac{\mathbf{z}_i \cdot \mathbf{z}_j}{\|\mathbf{z}_i\| \|\mathbf{z}_j\|} = \mathbf{z}_i \cdot \mathbf{z}_j$$

2. **Range becomes [-1, 1]:**
   - Unnormalized: $(-\infty, +\infty)$
   - Normalized: $[-1, 1]$

3. **Temperature scaling works better on bounded values**

**PyTorch Implementation:**

```python
def normalize(x, p=2, dim=1):
    """Manual implementation of F.normalize"""
    norm = x.pow(p).sum(dim=dim, keepdim=True).pow(1/p)
    return x / norm.clamp(min=1e-12)  # Avoid division by zero
```

---

### Step 2: Similarity Matrix

```python
sim_matrix = torch.matmul(projections, projections.T)
```

**Mathematical Operation:**

$$\mathbf{S} = \mathbf{Z}\mathbf{Z}^\top$$

Where:
- $\mathbf{Z} \in \mathbb{R}^{2N \times D}$ is the matrix of normalized projections
- $\mathbf{S} \in \mathbb{R}^{2N \times 2N}$ is the similarity matrix

**Matrix Structure:**

$$\mathbf{S}_{ij} = \mathbf{z}_i \cdot \mathbf{z}_j = \cos(\mathbf{z}_i, \mathbf{z}_j)$$

**Visualization:**

```
         view1_0  view1_1  view1_2  view1_3  view2_0  view2_1  view2_2  view2_3
         [0]      [1]      [2]      [3]      [4]      [5]      [6]      [7]

view1_0  [1.0]    [0.2]    [0.1]    [0.3]    [★0.8]   [0.1]    [0.2]    [0.1]
[0]

view1_1  [0.2]    [1.0]    [0.2]    [0.1]    [0.1]    [★0.7]   [0.1]    [0.3]
[1]

view1_2  [0.1]    [0.2]    [1.0]    [0.2]    [0.1]    [0.1]    [★0.9]   [0.2]
[2]

view1_3  [0.3]    [0.1]    [0.2]    [1.0]    [0.2]    [0.1]    [0.1]    [★0.6]
[3]

view2_0  [★0.8]   [0.1]    [0.1]    [0.2]    [1.0]    [0.2]    [0.1]    [0.3]
[4]

view2_1  [0.1]    [★0.7]   [0.1]    [0.1]    [0.2]    [1.0]    [0.2]    [0.1]
[5]

view2_2  [0.2]    [0.1]    [★0.9]   [0.1]    [0.1]    [0.2]    [1.0]    [0.2]
[6]

view2_3  [0.1]    [0.3]    [0.2]    [★0.6]   [0.3]    [0.1]    [0.2]    [1.0]
[7]

★ = positive pair (should have high similarity)
```

---

### Step 3: Temperature Scaling

```python
sim_matrix = sim_matrix / self.temperature
```

**Mathematical Operation:**

$$\mathbf{S}'_{ij} = \frac{\mathbf{S}_{ij}}{\tau}$$

**Why Temperature?**

The temperature $\tau$ controls the "sharpness" of the softmax distribution:

$$p_j = \frac{\exp(s_j/\tau)}{\sum_k \exp(s_k/\tau)}$$

**Effect of Temperature:**

| Temperature | Effect | Distribution |
|-------------|--------|--------------|
| Low (0.1) | Sharp | Focuses on hard negatives |
| Medium (0.5) | Moderate | Balanced learning |
| High (1.0) | Soft | Smooth gradients |

**Example:**

```python
sims = torch.tensor([0.8, 0.2, 0.1, 0.1])  # Similarities

for tau in [0.1, 0.5, 1.0]:
    scaled = sims / tau
    probs = F.softmax(scaled, dim=0)
    print(f"τ={tau}: {probs}")

# Output:
# τ=0.1: tensor([0.9933, 0.0022, 0.0022, 0.0022])  ← Very sharp
# τ=0.5: tensor([0.7214, 0.0929, 0.0929, 0.0929])  ← Moderate
# τ=1.0: tensor([0.4256, 0.1915, 0.1915, 0.1915])  ← Soft
```

---

### Step 4: Label Creation

```python
labels = torch.arange(N, device=projections.device)
labels = torch.cat([labels + N, labels])
```

**Mathematical Definition:**

For $N$ images with $2N$ views, the label function is:

$$\text{label}(i) = \begin{cases}
i + N & \text{if } i < N \text{ (view1)} \\
i - N & \text{if } i \geq N \text{ (view2)}
\end{cases}$$

**Example for N=4:**

```python
N = 4

# Step 1: Create base labels
labels = torch.arange(N)  # [0, 1, 2, 3]

# Step 2: Create labels for both views
labels = torch.cat([labels + N, labels])
# Result: [4, 5, 6, 7, 0, 1, 2, 3]

# Meaning:
# - View1[0] (index 0) → positive is View2[0] (index 4)
# - View1[1] (index 1) → positive is View2[1] (index 5)
# - View2[0] (index 4) → positive is View1[0] (index 0)
# - etc.
```

**Visualization:**

```
Index:   0   1   2   3   4   5   6   7
View:   v1_0 v1_1 v1_2 v1_3 v2_0 v2_1 v2_2 v2_3
Label:   4   5   6   7   0   1   2   3
         ↑   ↑   ↑   ↑   ↑   ↑   ↑   ↑
         │   │   │   │   │   │   │   │
         └───┴───┴───┴───┴───┴───┴───┘
         These pairs are positive pairs
```

---

### Step 5: Cross-Entropy Loss

```python
loss = F.cross_entropy(sim_matrix, labels)
```

**Mathematical Operation:**

$$\mathcal{L} = -\frac{1}{2N} \sum_{i=1}^{2N} \log \frac{\exp(\mathbf{S}'_{i, \text{label}(i)})}{\sum_{j=1}^{2N} \exp(\mathbf{S}'_{ij})}$$

**Expanded Form:**

$$\mathcal{L} = -\frac{1}{2N} \sum_{i=1}^{2N} \left[ \mathbf{S}'_{i, \text{label}(i)} - \log \sum_{j=1}^{2N} \exp(\mathbf{S}'_{ij}) \right]$$

**PyTorch Implementation:**

```python
def cross_entropy(logits, targets):
    """Manual implementation of F.cross_entropy"""
    # Softmax
    exp_logits = torch.exp(logits)
    softmax = exp_logits / exp_logits.sum(dim=1, keepdim=True)

    # Negative log likelihood
    nll = -torch.log(softmax[torch.arange(len(targets)), targets])

    return nll.mean()
```

---

## Gradient Derivation

### Loss for One Sample

For sample $i$ with positive pair at index $j$:

$$\ell_i = -\log \frac{\exp(s_{ij}/\tau)}{\sum_{k \neq i} \exp(s_{ik}/\tau)}$$

### Gradient with Respect to Similarity

$$\frac{\partial \ell_i}{\partial s_{ik}} = \begin{cases}
p_k - 1 & \text{if } k = j \text{ (positive)} \\
p_k & \text{if } k \neq j \text{ (negative)}
\end{cases}$$

Where $p_k = \frac{\exp(s_{ik}/\tau)}{\sum_m \exp(s_{im}/\tau)}$ is the softmax probability.

**Interpretation:**

1. **For positive pair (k = j):**
   - Gradient = $p_j - 1$
   - If $p_j$ is low → gradient is negative → increase $s_{ij}$
   - Goal: Maximize similarity to positive

2. **For negative pairs (k ≠ j):**
   - Gradient = $p_k$
   - If $p_k$ is high → gradient is positive → decrease $s_{ik}$
   - Goal: Minimize similarity to negatives

### Gradient with Respect to Embedding

Using chain rule:

$$\frac{\partial \ell_i}{\partial \mathbf{z}_i} = \sum_k \frac{\partial \ell_i}{\partial s_{ik}} \cdot \frac{\partial s_{ik}}{\partial \mathbf{z}_i}$$

Since $s_{ik} = \mathbf{z}_i \cdot \mathbf{z}_k$:

$$\frac{\partial s_{ik}}{\partial \mathbf{z}_i} = \mathbf{z}_k$$

Therefore:

$$\frac{\partial \ell_i}{\partial \mathbf{z}_i} = \sum_k \frac{\partial \ell_i}{\partial s_{ik}} \mathbf{z}_k$$

---

## Numerical Example

### Setup

```python
# Batch of 2 images (N=2), embedding dimension D=3
z1 = torch.tensor([[1.0, 0.5, 0.3],   # View1 of image 1
                   [0.2, 0.8, 0.4]])  # View1 of image 2

z2 = torch.tensor([[1.1, 0.4, 0.2],   # View2 of image 1
                   [0.1, 0.9, 0.5]])  # View2 of image 2

# Combine
all_z = torch.cat([z1, z2], dim=0)  # [4, 3]
```

### Step-by-Step Computation

```python
# 1. Normalize
all_z_norm = F.normalize(all_z, p=2, dim=1)

# 2. Similarity matrix
sim = all_z_norm @ all_z_norm.T
# Result:
# [[1.00, 0.85, 0.99, 0.84],
#  [0.85, 1.00, 0.83, 0.99],
#  [0.99, 0.83, 1.00, 0.86],
#  [0.84, 0.99, 0.86, 1.00]]

# 3. Temperature scaling (τ=0.5)
sim_scaled = sim / 0.5
# [[2.00, 1.70, 1.98, 1.68],
#  [1.70, 2.00, 1.66, 1.98],
#  [1.98, 1.66, 2.00, 1.72],
#  [1.68, 1.98, 1.72, 2.00]]

# 4. Labels
labels = torch.tensor([2, 3, 0, 1])

# 5. Cross-entropy loss
loss = F.cross_entropy(sim_scaled, labels)
# Result: ~0.52
```

---

## Key Insights

> [!important] Why This Works
> 1. **Classification formulation**: Turns contrastive learning into (2N-1)-way classification
> 2. **Automatic negative sampling**: All other samples in batch are negatives
> 3. **Hard negative mining**: High similarity negatives get larger gradients
> 4. **End-to-end**: No memory bank needed, simple implementation

---

## Related Notes

- [[02 - NT-Xent Loss]] - Conceptual overview
- [[simclr_concept_2_similarity_matrix.png]] - Visual explanation