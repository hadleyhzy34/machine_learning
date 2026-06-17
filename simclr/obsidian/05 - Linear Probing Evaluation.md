---
tags:
  - ml/contrastive-learning
  - evaluation
  - linear-probing
  - simclr
created: 2026-03-04
---

# Linear Probing Evaluation

> [!abstract] Overview
> Linear probing is the standard evaluation method for self-supervised learning. It measures how well the learned representations transfer to downstream tasks.

## What is Linear Probing?

Linear probing evaluates learned representations by:

1. **Freezing** the pretrained encoder (no gradient updates)
2. **Training** a simple linear classifier on top
3. **Evaluating** on a test set

> [!important] Key Insight
> If representations are good, a linear classifier should achieve high accuracy. If representations are bad, even a complex classifier will struggle.

---

## Why Linear Probing?

### Information Theory View

Linear probing measures if the representations are **linearly separable**:

$$
\hat{y} = \mathbf{W}\mathbf{h} + \mathbf{b}
$$

If accuracy is high, the encoder has learned features where:
- Same class samples are clustered together
- Different classes are linearly separable

### Comparison to Fine-tuning

| Method | Encoder Weights | Classifier | Measures |
|--------|----------------|------------|----------|
| Linear Probe | Frozen | Linear | Representation quality |
| Fine-tuning | Unfrozen | Any | End-task performance |

Linear probing isolates representation quality from optimization capability.

---

## Implementation

### Step 1: Freeze Encoder

```python
def freeze_encoder(model):
    """Freeze encoder weights for linear probing"""
    model.encoder.eval()  # Set to eval mode
    for param in model.encoder.parameters():
        param.requires_grad = False
```

### Step 2: Create Linear Classifier

```python
# Simple linear classifier
classifier = nn.Linear(512, 10)  # 512 features → 10 classes (CIFAR-10)
```

### Step 3: Training Loop

```python
def linear_probe(model, train_loader, test_loader, device, n_epochs=20):
    """
    Linear probing evaluation

    Args:
        model: SimCLR model with pretrained encoder
        train_loader: Training data loader
        test_loader: Test data loader
        device: Device to use
        n_epochs: Number of epochs for linear classifier training

    Returns:
        test_accuracy: Final test accuracy
    """
    # Freeze encoder
    freeze_encoder(model)

    # Linear classifier
    classifier = nn.Linear(512, 10).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Training
    for epoch in range(n_epochs):
        classifier.train()
        for (x1, x2), y in train_loader:
            x = torch.cat([x1, x2], dim=0).to(device)
            y = torch.cat([y, y], dim=0).to(device)

            optimizer.zero_grad()

            # Extract features (no gradient through encoder)
            with torch.no_grad():
                features, _ = model(x)  # Only use encoder output

            # Train classifier
            logits = classifier(features)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

    # Evaluation
    return evaluate_linear_probe(model, classifier, test_loader, device)
```

### Step 4: Evaluation

```python
def evaluate_linear_probe(model, classifier, test_loader, device):
    """Evaluate linear probe on test set"""
    model.encoder.eval()
    classifier.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            features, _ = model(x)
            logits = classifier(features)

            _, predicted = logits.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

    accuracy = 100. * correct / total
    print(f"Linear Probe Accuracy: {accuracy:.2f}%")
    return accuracy
```

---

## Mathematical Formulation

### Feature Extraction

For input image $x$:

$$
\mathbf{h} = f(x) \in \mathbb{R}^d
$$

Where $f$ is the frozen encoder.

### Linear Classification

$$
\hat{y} = \text{softmax}(\mathbf{W}\mathbf{h} + \mathbf{b})
$$

Where:
- $\mathbf{W} \in \mathbb{R}^{C \times d}$ (weight matrix)
- $\mathbf{b} \in \mathbb{R}^C$ (bias vector)
- $C$ = number of classes

### Cross-Entropy Loss

$$
\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} y_{ic} \log(\hat{y}_{ic})
$$

---

## Contrastive vs Supervised Comparison

### Training Paradigm

```
┌────────────────────────────────────────────────────────────────┐
│ SUPERVISED TRAINING                                            │
│                                                                │
│  Image → [Encoder] → Features → [Classifier] → Loss → Update all│
│                          ↑                    ↑                 │
│                          └──── Both updated ──┘                 │
│                                                                │
│  Result: Features optimized for specific classification task   │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ CONTRASTIVE PRETRAINING + LINEAR PROBE                         │
│                                                                │
│  Phase 1 (Contrastive):                                        │
│  Image → [Encoder] → Features → [Projection] → Loss → Update all│
│                                                                │
│  Phase 2 (Linear Probe):                                       │
│  Image → [Encoder] → Features → [Classifier] → Loss            │
│          ↑ (FROZEN!)                       ↑ Update only        │
│                                                                │
│  Result: Features are general-purpose (no labels needed!)      │
└────────────────────────────────────────────────────────────────┘
```

### Performance Comparison

| Method | CIFAR-10 Accuracy | Notes |
|--------|-------------------|-------|
| Random Init + Linear | ~35% | Baseline, poor representations |
| Supervised (full) | ~85-90% | Upper bound |
| SimCLR + Linear Probe | ~70-80% | Self-supervised, competitive! |

---

## Embedding Visualization

![[simclr_embedding_evolution.png]]

### PCA Visualization

```python
def visualize_embeddings(model, test_loader, device, n_samples=1000):
    """
    Visualize learned embeddings using PCA

    Good representations should:
    1. Cluster by class (even without labels during training!)
    2. Separate different classes
    """
    model.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            features, _ = model(x)
            all_features.append(features.cpu())
            all_labels.append(y)

            if len(all_features) * x.shape[0] >= n_samples:
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
    plt.title('PCA Visualization of Learned Representations\n'
              'Colors = true labels (NOT used during training!)')
    plt.legend()
    plt.show()
```

---

## Clustering Quality Metrics

### Silhouette Score

Measures how well-separated the clusters are:

$$
s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}
$$

Where:
- $a(i)$ = mean distance to same-class samples
- $b(i)$ = mean distance to nearest different class

```python
from sklearn.metrics import silhouette_score

sil_score = silhouette_score(features, labels)
print(f"Silhouette Score: {sil_score:.3f}")
# Higher = better clustering (range: -1 to 1)
```

---

## Expected Results

### Linear Probe Accuracy Over Training

```
Accuracy
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
    └────────────┴────┴────┴────┴────┴──► Epoch
     0           20   40   60   80  100
```

### What Good Embeddings Look Like

![[simclr_embedding_evolution.png]]

> [!success] Good Signs
> - Classes cluster together
> - Different classes are separated
> - Smooth boundaries between clusters

> [!warning] Bad Signs
> - All points mixed together
> - Only some classes separated
> - Heavy overlap between classes

---

## Key Insights

> [!important] Why Linear Probing?
> 1. **Simple**: Easy to implement and interpret
> 2. **Fair**: Doesn't benefit from more classifier capacity
> 3. **Standard**: Accepted evaluation in self-supervised learning
> 4. **Informative**: Directly measures representation quality

---

## Related Notes

- [[03 - SimCLR Architecture]] - Understanding the encoder
- [[10 - Code Part 5 - Evaluation]] - Detailed code walkthrough
- [[simclr_embedding_evolution.png]] - Visualization example