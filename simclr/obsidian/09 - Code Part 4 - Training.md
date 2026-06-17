---
tags:
  - ml/contrastive-learning
  - code-analysis
  - training
  - simclr
created: 2026-03-04
---

# Code Part 4 - Training Pipeline

> [!abstract] Overview
> Detailed code analysis of the SimCLR training pipeline including data loading, optimization, and metric tracking.

## Code Location
`simclr_tutorial.py` lines 368-459

---

## Complete Code

```python
def train_simclr(model, dataloader, criterion, optimizer, scheduler,
                 device, n_epochs, save_interval=5):
    """Training loop for SimCLR"""

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

                pos_sims = [sim_matrix[i, i+N].item() for i in range(N)]
                epoch_pos_sim += np.mean(pos_sims)

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

    return history
```

---

## Line-by-Line Analysis

### Step 1: History Tracking

```python
history = {
    'loss': [],
    'positive_sim': [],
    'negative_sim': []
}
```

**Purpose:**
Track training progress over epochs. These metrics reveal if the model is learning correctly.

| Metric | Good Sign | Bad Sign |
|--------|-----------|----------|
| Loss | Decreasing | Stuck or increasing |
| Positive Sim | Increasing toward 1 | Not increasing |
| Negative Sim | Low (near 0) | Increasing |

---

### Step 2: Batch Processing

```python
for (view1, view2), _ in pbar:
    view1, view2 = view1.to(device), view2.to(device)
```

**Data Flow:**

```
DataLoader returns:
    (view1, view2): Tuple of two tensors [B, 3, H, W] each
    _: Labels (IGNORED in self-supervised learning!)

Move to device:
    view1: [B, 3, 32, 32] on GPU/CPU
    view2: [B, 3, 32, 32] on GPU/CPU
```

> [!important] No Labels!
> The `_` indicates we explicitly ignore labels. SimCLR is **self-supervised** - it learns from the images themselves, not from labels.

---

### Step 3: Forward Pass

```python
# Forward pass through both views
_, proj1 = model(view1)
_, proj2 = model(view2)

# Concatenate projections for contrastive loss
projections = torch.cat([proj1, proj2], dim=0)
```

**Mathematical Formulation:**

$$\mathbf{z}_i^1 = g(f(\tilde{x}_i^1)), \quad \mathbf{z}_i^2 = g(f(\tilde{x}_i^2))$$

$$\mathbf{Z} = [\mathbf{z}_1^1; \mathbf{z}_2^1; ...; \mathbf{z}_N^1; \mathbf{z}_1^2; \mathbf{z}_2^2; ...; \mathbf{z}_N^2]$$

Where $\mathbf{Z} \in \mathbb{R}^{2N \times 128}$

**Visualization:**

```
Batch of N images
       │
       ├── Augment → View 1 → Encoder → Projection → proj1 [N, 128]
       │
       └── Augment → View 2 → Encoder → Projection → proj2 [N, 128]
                                                        │
                            Concatenate ←───────────────┘
                                                        │
                                                        ▼
                                             projections [2N, 128]
```

---

### Step 4: Loss Computation

```python
# Compute NT-Xent loss
loss = criterion(projections)
```

**Mathematical Formulation:**

$$\mathcal{L} = \frac{1}{2N} \sum_{i=1}^{N} \left[ \ell(\mathbf{z}_i^1, \mathbf{z}_i^2) + \ell(\mathbf{z}_i^2, \mathbf{z}_i^1) \right]$$

Where:
$$\ell(\mathbf{z}_i, \mathbf{z}_j) = -\log \frac{\exp(\text{sim}(\mathbf{z}_i, \mathbf{z}_j)/\tau)}{\sum_{k \neq i} \exp(\text{sim}(\mathbf{z}_i, \mathbf{z}_k)/\tau)}$$

---

### Step 5: Backward Pass

```python
# Backward pass
loss.backward()
optimizer.step()
```

**Mathematical Formulation:**

Gradient computation:
$$\frac{\partial \mathcal{L}}{\partial \theta} = \nabla_\theta \mathcal{L}$$

Parameter update (Adam optimizer):
$$\theta_{t+1} = \theta_t - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

Where $\hat{m}_t$ and $\hat{v}_t$ are bias-corrected first and second moment estimates.

---

### Step 6: Similarity Tracking

```python
with torch.no_grad():
    all_proj = F.normalize(projections, p=2, dim=1)
    sim_matrix = torch.matmul(all_proj, all_proj.T)
    N = projections.shape[0] // 2

    # Positive similarities
    pos_sims = [sim_matrix[i, i+N].item() for i in range(N)]

    # Negative similarities
    neg_sims = []
    for i in range(min(N, 10)):
        for j in range(N):
            if j != i:
                neg_sims.append(sim_matrix[i, j+N].item())
```

**Mathematical Formulation:**

Positive similarity:
$$\text{pos\_sim} = \frac{1}{N} \sum_{i=1}^{N} \cos(\mathbf{z}_i^1, \mathbf{z}_i^2)$$

Negative similarity:
$$\text{neg\_sim} = \frac{1}{N(N-1)} \sum_{i=1}^{N} \sum_{j \neq i} \cos(\mathbf{z}_i^1, \mathbf{z}_j^2)$$

**Visualization:**

```
Similarity Matrix [2N, 2N]:

     0   1   2   3   4   5   6   7   (indices)
   ┌───┬───┬───┬───┬───┬───┬───┬───┐
 0 │1.0│neg│neg│neg│POS│neg│neg│neg│  ← pos_sim = sim[0,4]
   ├───┼───┼───┼───┼───┼───┼───┼───┤
 1 │neg│1.0│neg│neg│neg│POS│neg│neg│  ← pos_sim = sim[1,5]
   ├───┼───┼───┼───┼───┼───┼───┼───┤
 2 │neg│neg│1.0│neg│neg│neg│POS│neg│
   ├───┼───┼───┼───┼───┼───┼───┼───┤
 3 │neg│neg│neg│1.0│neg│neg│neg│POS│
   ├───┼───┼───┼───┼───┼───┼───┼───┤
 4 │POS│neg│neg│neg│1.0│neg│neg│neg│
   ├───┼───┼───┼───┼───┼───┼───┼───┤
 5 │neg│POS│neg│neg│neg│1.0│neg│neg│
   ├───┼───┼───┼───┼───┼───┼───┼───┤
 6 │neg│neg│POS│neg│neg│neg│1.0│neg│
   ├───┼───┼───┼───┼───┼───┼───┼───┤
 7 │neg│neg│neg│POS│neg│neg│neg│1.0│
   └───┴───┴───┴───┴───┴───┴───┴───┘

POS = positive pair (same image, different augmentation)
neg = negative pair (different images)
```

---

## Optimizer Configuration

### Adam Optimizer

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,          # Learning rate
    weight_decay=1e-6  # L2 regularization
)
```

**Mathematical Formulation:**

Adam combines momentum and adaptive learning rates:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$
$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}$$
$$\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

Where:
- $g_t = \nabla_\theta \mathcal{L}$ (gradient)
- $\beta_1 = 0.9$, $\beta_2 = 0.999$ (default)
- $\eta = 0.001$ (learning rate)

### Weight Decay

```python
weight_decay=1e-6
```

**Mathematical Formulation:**

$$\mathcal{L}_{total} = \mathcal{L} + \lambda \|\theta\|_2^2$$

Where $\lambda = 10^{-6}$

**Effect:** Prevents overfitting by penalizing large weights.

---

## Learning Rate Schedule

### Cosine Annealing

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=n_epochs,  # Period
    eta_min=1e-6     # Minimum LR
)
```

**Mathematical Formulation:**

$$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 + \cos(\frac{t}{T}\pi))$$

**Visualization:**

```
Learning Rate (η)
    │
0.001├──────╮
     │       ╲
     │        ╲
     │         ╲
     │          ╲
     │           ╲
     │            ╲
     │             ╲
1e-6 ├──────────────╯
     └───────────────► Epoch (t)
     0            100
```

**Why Cosine Annealing?**

1. **Smooth decay**: Gradual reduction helps convergence
2. **Exploration**: High LR early for exploration
3. **Exploitation**: Low LR late for fine-tuning

---

## Batch Size Analysis

### Why Large Batch Size?

```python
batch_size = 256  # Typical for SimCLR
```

**Number of Negatives:**

For batch size $B$, each sample has:
- **Positive pairs**: 1 (the other view)
- **Negative pairs**: $2B - 2$ (all other views)

| Batch Size | Negatives per Sample |
|------------|---------------------|
| 64 | 126 |
| 128 | 254 |
| 256 | 510 |
| 512 | 1022 |
| 1024 | 2046 |

> [!important] More Negatives = Better Learning
> More negative samples provide more contrastive signal, leading to better representations.

### Memory Considerations

```python
# Memory calculation for batch_size=256, image_size=32x32x3
image_memory = 256 * 3 * 32 * 32 * 4  # 4 bytes per float32
# = 39,321,600 bytes ≈ 37.5 MB (just for images)

# Model memory (ResNet-18)
model_memory = 11.5 * 10^6 * 4  # parameters
# = 46,000,000 bytes ≈ 44 MB

# Total GPU memory needed: ~2-4 GB for batch_size=256
```

---

## Checkpoint Saving

```python
if (epoch + 1) % save_interval == 0:
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
    }, f'simclr_checkpoint_epoch{epoch+1}.pth')
```

**What's Saved:**
- `epoch`: Current training progress
- `model_state_dict`: All model weights
- `optimizer_state_dict`: Adam momentum buffers
- `loss`: Current loss value

**To Resume Training:**

```python
checkpoint = torch.load('simclr_checkpoint_epoch50.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1
```

---

## Expected Training Curves

![[simclr_demo_results.png]]

### Healthy Training Signs

```
Epoch  1: Loss=1.85, PosSim=0.15, NegSim=0.05
Epoch  5: Loss=1.20, PosSim=0.45, NegSim=0.08
Epoch 10: Loss=0.75, PosSim=0.65, NegSim=0.10
Epoch 15: Loss=0.50, PosSim=0.78, NegSim=0.12
Epoch 20: Loss=0.35, PosSim=0.85, NegSim=0.15

↑ Loss decreasing, positive similarity increasing
```

### Warning Signs

```
Epoch  1: Loss=2.0, PosSim=0.0, NegSim=0.0
Epoch  5: Loss=2.0, PosSim=0.1, NegSim=0.0  ← Not learning!
Epoch 10: Loss=2.0, PosSim=0.1, NegSim=0.0

Potential issues:
- Learning rate too high/low
- Augmentations too strong/weak
- Model capacity insufficient
```

---

## Key Insights

> [!important] Training Pipeline Summary
> 1. **No labels needed**: Self-supervised from augmented image pairs
> 2. **Large batch size**: More negatives = better learning
> 3. **Cosine LR schedule**: Smooth convergence
> 4. **Monitor similarities**: Track positive/negative separation
> 5. **Save checkpoints**: Long training runs need recovery points

---

## Related Notes

- [[04 - Training Pipeline]] - Conceptual overview
- [[simclr_demo_results.png]] - Example training visualization