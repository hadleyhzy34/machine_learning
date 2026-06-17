---
tags:
  - ml/contrastive-learning
  - code-analysis
  - evaluation
  - simclr
created: 2026-03-04
---

# Code Part 5 - Linear Probing Evaluation

> [!abstract] Overview
> Detailed code analysis of the linear probing evaluation method for assessing learned representations.

## Code Location
`simclr_tutorial.py` lines 500-649

---

## Complete Code

```python
def evaluate_representations(model, train_loader, test_loader, device, n_epochs=10):
    """Evaluate learned representations using linear probing"""

    print("\n" + "="*60)
    print("EVALUATION: Linear Probing")
    print("="*60)

    # Freeze encoder, train only linear classifier
    model.encoder.eval()
    for param in model.encoder.parameters():
        param.requires_grad = False

    # Create linear classifier
    classifier = nn.Linear(512, 10).to(device)
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
            _, predicted = logits.max(1)
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
```

---

## Line-by-Line Analysis

### Step 1: Freeze Encoder

```python
# Freeze encoder, train only linear classifier
model.encoder.eval()
for param in model.encoder.parameters():
    param.requires_grad = False
```

**Mathematical Implication:**

During linear probing, the encoder weights $\theta_{enc}$ are fixed:

$$\theta_{enc} = \theta_{enc}^{*} \text{ (frozen)}$$

Only the linear classifier weights $W, b$ are updated.

**Why Freeze?**

> [!important] Evaluating Representation Quality
> By freezing the encoder, we measure:
> - **How good are the learned features?**
> - **Are they linearly separable?**
> - **Do they transfer to new tasks?**
>
> If accuracy is high → good representations
> If accuracy is low → poor representations

---

### Step 2: Create Linear Classifier

```python
# Create linear classifier
classifier = nn.Linear(512, 10).to(device)
```

**Mathematical Formulation:**

$$\hat{y} = \text{softmax}(W \mathbf{h} + b)$$

Where:
- $\mathbf{h} \in \mathbb{R}^{512}$ (encoder features)
- $W \in \mathbb{R}^{10 \times 512}$ (weight matrix)
- $b \in \mathbb{R}^{10}$ (bias vector)
- $\hat{y} \in \mathbb{R}^{10}$ (class probabilities)

**Parameter Count:**

$$|W| + |b| = 512 \times 10 + 10 = 5,130 \text{ parameters}$$

> [!note] Minimal Parameters
> A linear classifier has very few parameters compared to the encoder (~11M).
> This ensures we measure feature quality, not classifier capacity.

---

### Step 3: Feature Extraction

```python
# Get features (no gradient through encoder)
with torch.no_grad():
    features, _ = model(x)
```

**Mathematical Operation:**

$$\mathbf{h} = f_{\theta^*}(x)$$

Where $f_{\theta^*}$ is the frozen encoder with fixed parameters.

**Why `torch.no_grad()`?**

1. **Memory efficiency**: No need to store intermediate activations
2. **Speed**: No backward pass through encoder
3. **Correctness**: Ensures gradients don't flow to encoder

---

### Step 4: Classification and Loss

```python
# Classify
logits = classifier(features)
loss = criterion(logits, y)
```

**Mathematical Formulation:**

Logits (unnormalized scores):
$$\mathbf{s} = W \mathbf{h} + b$$

Softmax probabilities:
$$p_c = \frac{\exp(s_c)}{\sum_{k=1}^{C} \exp(s_k)}$$

Cross-entropy loss:
$$\mathcal{L} = -\sum_{c=1}^{C} y_c \log(p_c)$$

For one-hot labels:
$$\mathcal{L} = -\log(p_{y_{true}})$$

---

### Step 5: Optimization

```python
loss.backward()
optimizer.step()
```

**Gradient Computation:**

$$\frac{\partial \mathcal{L}}{\partial W} = (\mathbf{p} - \mathbf{y}) \otimes \mathbf{h}$$

$$\frac{\partial \mathcal{L}}{\partial b} = \mathbf{p} - \mathbf{y}$$

Where $\otimes$ is the outer product.

**Parameter Update:**

$$W \leftarrow W - \eta \frac{\partial \mathcal{L}}{\partial W}$$
$$b \leftarrow b - \eta \frac{\partial \mathcal{L}}{\partial b}$$

---

## Embedding Visualization

### PCA Projection

```python
def plot_embedding_visualization(model, test_loader, device, n_samples=1000):
    """Visualize learned embeddings using PCA"""
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
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    for class_idx in range(10):
        mask = labels == class_idx
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1],
                   alpha=0.5, s=20, label=class_names[class_idx])

    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.title('PCA Visualization of Learned Representations')
    plt.legend()
    plt.show()
```

**Mathematical Formulation:**

PCA finds principal components by eigendecomposition of the covariance matrix:

$$\mathbf{C} = \frac{1}{N} \mathbf{H}^\top \mathbf{H}$$

$$\mathbf{C} = \mathbf{V} \mathbf{\Lambda} \mathbf{V}^\top$$

Projection to 2D:
$$\mathbf{H}_{2D} = \mathbf{H} \mathbf{V}_{[:, 1:2]}$$

---

## Comparison: Contrastive vs Supervised

### Supervised Baseline

```python
def train_supervised_baseline(model, train_loader, test_loader, device, n_epochs=20):
    """Train the same architecture with supervised learning"""

    # Full model training (encoder + classifier)
    classifier = nn.Linear(512, 10).to(device)
    params = list(model.parameters()) + list(classifier.parameters())
    optimizer = torch.optim.Adam(params, lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(n_epochs):
        model.train()
        classifier.train()

        for (x1, x2), y in train_loader:
            x = torch.cat([x1, x2], dim=0).to(device)
            y = torch.cat([y, y], dim=0).to(device)

            optimizer.zero_grad()

            features, _ = model(x)  # Encoder updated
            logits = classifier(features)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()
```

**Key Differences:**

| Aspect | Supervised | Contrastive + Linear Probe |
|--------|------------|----------------------------|
| Labels | Required during training | Not required during pretraining |
| Encoder | Updated during training | Frozen during evaluation |
| Features | Task-specific | General-purpose |
| Transfer | Limited | Better for downstream tasks |

---

## Performance Metrics

### Linear Probe Accuracy

```python
def evaluate_linear_probe(classifier, features, labels):
    """Compute linear probe accuracy"""
    with torch.no_grad():
        logits = classifier(features)
        _, predicted = logits.max(1)
        accuracy = (predicted == labels).float().mean()

    return accuracy.item() * 100
```

**Mathematical Formulation:**

$$\text{Accuracy} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[\hat{y}_i = y_i] \times 100\%$$

### Silhouette Score

```python
from sklearn.metrics import silhouette_score

def compute_silhouette(features, labels):
    """Compute silhouette score for clustering quality"""
    score = silhouette_score(features, labels)
    return score
```

**Mathematical Formulation:**

$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

Where:
- $a(i)$ = mean intra-cluster distance
- $b(i)$ = mean nearest-cluster distance

Range: $[-1, 1]$, higher is better.

---

## Expected Results

### Linear Probe Accuracy Over Training

```
Linear Probe Accuracy vs Pretraining Epochs

Accuracy (%)
    │
80%├─────────────────────────────╮
    │                            │
70%├─────────────────────────╭───┤
    │                        │    │
60%├─────────────────────╭───┤    │
    │                    │    │    │
50%├─────────────────╭───┤    │    │
    │                │    │    │    │
40%├─────────────╭───┤    │    │    │
    │            │    │    │    │    │
    └────────────┴────┴────┴────┴────┴──► Pretraining Epoch
     0           20   40   60   80  100
```

### Comparison Table

| Method | CIFAR-10 Accuracy | ImageNet Accuracy |
|--------|-------------------|-------------------|
| Random Init + Linear | ~35% | ~5% |
| Supervised (full) | ~90% | ~76% |
| SimCLR + Linear Probe | ~75-80% | ~69% |

---

## Visualization Examples

![[simclr_embedding_evolution.png]]

### Good Embeddings

```
Characteristics:
- Clear class clusters
- Inter-class separation
- Intra-class compactness
- Smooth boundaries
```

### Poor Embeddings

```
Characteristics:
- Mixed classes
- No clear clusters
- High overlap
- Random scatter
```

---

## Key Insights

> [!important] Linear Probing Summary
> 1. **Measures representation quality**: How linearly separable are the features?
> 2. **Simple and fair**: No benefit from complex classifiers
> 3. **Standard benchmark**: Accepted in self-supervised learning
> 4. **Transfer indicator**: Good linear probe → good transfer learning

> [!success] Good Results Indicate
> - Features are linearly separable
> - Model learned semantic structure
> - Representations transfer well

> [!warning] Poor Results May Indicate
> - Insufficient pretraining
> - Too strong/weak augmentations
> - Model capacity issues
> - Wrong temperature

---

## Related Notes

- [[05 - Linear Probing Evaluation]] - Conceptual overview
- [[simclr_embedding_evolution.png]] - Visualization example