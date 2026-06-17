---
tags:
  - ml/contrastive-learning
  - training
  - simclr
created: 2026-03-04
---

# Training Pipeline

> [!abstract] Overview
> This note covers the complete training pipeline for SimCLR, including data loading, the training loop, and monitoring metrics.

## Training Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    SimCLR Training Pipeline                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  For each epoch:                                                │
│      For each batch:                                            │
│          1. Load batch of N images                              │
│          2. Create 2 augmented views per image → 2N views       │
│          3. Forward pass through encoder + projection head      │
│          4. Compute NT-Xent loss                                │
│          5. Backward pass and update weights                    │
│          6. Track metrics (loss, similarities)                  │
│                                                                 │
│  After training:                                                │
│      - Discard projection head                                  │
│      - Keep encoder for downstream tasks                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Training Parameters

| Parameter | Typical Value | Notes |
|-----------|--------------|-------|
| Batch Size | 256-512 | Larger = more negatives = better |
| Learning Rate | 0.001-0.003 | Adam optimizer |
| Temperature | 0.5 | For NT-Xent loss |
| Epochs | 100-1000 | More epochs = better representations |
| Weight Decay | 1e-6 | L2 regularization |

> [!warning] Batch Size Matters!
> In contrastive learning, batch size determines the number of negative samples:
> - Batch size 256 → 510 negatives per positive
> - Batch size 512 → 1022 negatives per positive
>
> More negatives = better representation learning!

---

## Data Loading

### Dataset Wrapper

```python
class SimCLRCIFAR10(torch.utils.data.Dataset):
    """
    Wrapper dataset that applies SimCLR transforms to CIFAR-10.

    Returns TWO augmented views per image + the label.
    Labels are NOT used during contrastive training!
    """

    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        view1, view2 = self.transform(img)  # Two augmented views
        return (view1, view2), label

    def __len__(self):
        return len(self.base_dataset)
```

### DataLoader Setup

```python
# SimCLR transform
simclr_transform = SimCLRTransform(input_size=32)

# Create dataset wrapper
simclr_dataset = SimCLRCIFAR10(base_cifar10, simclr_transform)

# DataLoader with LARGE batch size
train_loader = DataLoader(
    simclr_dataset,
    batch_size=256,      # Important: large batch size!
    shuffle=True,
    num_workers=4,
    drop_last=True       # Discard incomplete batches
)
```

---

## Training Loop

### Implementation

```python
def train_simclr(model, dataloader, criterion, optimizer, device, n_epochs):
    """
    SimCLR Training Loop

    KEY: No labels are used! This is self-supervised learning.
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

        for (view1, view2), _ in dataloader:  # Labels ignored!
            view1, view2 = view1.to(device), view2.to(device)

            # Forward pass through both views
            _, proj1 = model(view1)
            _, proj2 = model(view2)

            # Concatenate projections for loss
            projections = torch.cat([proj1, proj2], dim=0)

            # Compute NT-Xent loss
            loss = criterion(projections)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track metrics
            epoch_loss += loss.item()

            # Track similarities (for monitoring)
            with torch.no_grad():
                pos_sim, neg_sim = compute_similarities(proj1, proj2)
                epoch_pos_sim += pos_sim
                epoch_neg_sim += neg_sim

        # Record history
        history['loss'].append(epoch_loss / len(dataloader))
        history['positive_sim'].append(epoch_pos_sim / len(dataloader))
        history['negative_sim'].append(epoch_neg_sim / len(dataloader))

        print(f"Epoch {epoch+1}: Loss={history['loss'][-1]:.4f}")

    return history
```

### Similarity Monitoring

```python
def compute_similarities(proj1, proj2):
    """
    Compute positive and negative similarities for monitoring.

    Positive similarity: Similarity between positive pairs (should increase)
    Negative similarity: Similarity between negative pairs (should decrease)
    """
    # Normalize projections
    proj1 = F.normalize(proj1, p=2, dim=1)
    proj2 = F.normalize(proj2, p=2, dim=1)

    # Compute similarity matrix
    all_proj = torch.cat([proj1, proj2], dim=0)
    sim_matrix = torch.matmul(all_proj, all_proj.T)

    N = proj1.shape[0]

    # Positive similarities: diagonal of the off-diagonal blocks
    positive_sims = torch.diag(sim_matrix[:N, N:])

    # Negative similarities: off-diagonal elements
    # (excluding self-similarity and positive pairs)
    negative_sims = []
    for i in range(2*N):
        for j in range(2*N):
            if i != j and abs(i - j) != N:
                negative_sims.append(sim_matrix[i, j])

    return positive_sims.mean().item(), torch.stack(negative_sims).mean().item()
```

---

## Mathematical Formulation

### Batch Processing

For a batch of $N$ images, we create $2N$ views:

$$
\mathcal{B} = \{(\tilde{x}_1^1, \tilde{x}_1^2), (\tilde{x}_2^1, \tilde{x}_2^2), ..., (\tilde{x}_N^1, \tilde{x}_N^2)\}
$$

Where $\tilde{x}_i^1, \tilde{x}_i^2$ are the two augmented views of image $x_i$.

### Forward Pass

For each view:

$$
\mathbf{h}_i^k = f(\tilde{x}_i^k), \quad \mathbf{z}_i^k = g(\mathbf{h}_i^k)
$$

Where $k \in \{1, 2\}$ denotes the view, $f$ is the encoder, $g$ is the projection head.

### Loss Computation

$$
\mathcal{L} = \frac{1}{2N} \sum_{i=1}^{N} \left[ \ell(\mathbf{z}_i^1, \mathbf{z}_i^2) + \ell(\mathbf{z}_i^2, \mathbf{z}_i^1) \right]
$$

### Gradient Update

$$
\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}
$$

---

## Learning Rate Schedule

SimCLR uses cosine annealing:

$$
\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 + \cos(\frac{t}{T}\pi))
$$

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=n_epochs,    # Total epochs
    eta_min=1e-6       # Minimum learning rate
)
```

### Visualization

```
Learning Rate
    │
1e-3├──────╮
    │       ╲
    │        ╲
    │         ╲
    │          ╲
    │           ╲
    │            ╲
    │             ╲
1e-6├──────────────╯
    └───────────────► Epoch
    0            100
```

---

## Training Metrics Visualization

![[simclr_demo_results.png]]

### Expected Behavior

| Metric | Initial | Final | Desired Trend |
|--------|---------|-------|---------------|
| Loss | ~2.0 | ~0.5 | Decreasing |
| Positive Sim | ~0.0 | ~0.8 | Increasing |
| Negative Sim | ~0.0 | ~0.1 | Staying low |

---

## Checkpoint Saving

```python
def save_checkpoint(model, optimizer, epoch, loss, path):
    """Save training checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)

def load_checkpoint(model, optimizer, path):
    """Load training checkpoint"""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']
```

---

## Tips for Better Training

> [!tip] Best Practices
> 1. **Use large batch sizes** (256+) with `drop_last=True`
> 2. **Train for enough epochs** (100+ for CIFAR, 1000+ for ImageNet)
> 3. **Use cosine annealing** for learning rate
> 4. **Monitor positive/negative similarities** to diagnose issues
> 5. **Save checkpoints** regularly for long training runs

> [!warning] Common Issues
> - **Loss not decreasing**: Check augmentations are strong enough
> - **Positive sim not increasing**: Model may be underfitting, try longer training
> - **Negative sim too high**: Increase temperature or batch size

---

## Related Notes

- [[04 - Training Pipeline]] - This note
- [[09 - Code Part 4 - Training]] - Detailed code walkthrough
- [[simclr_demo_results.png]] - Example training visualization