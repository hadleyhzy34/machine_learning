---
tags:
  - ml/contrastive-learning
  - data-augmentation
  - simclr
created: 2026-03-04
---

# Data Augmentation Pipeline

> [!abstract] Overview
> Data augmentation is the foundation of contrastive learning. By creating different "views" of the same image, we generate the positive pairs that the model must learn to recognize as similar.

## The Core Idea

For each image $x$, we generate two augmented views:

$$
x_1 = \mathcal{T}_1(x), \quad x_2 = \mathcal{T}_2(x)
$$

Where $\mathcal{T}_1$ and $\mathcal{T}_2$ are stochastic augmentation functions.

---

## SimCLR Augmentation Pipeline

### Augmentation Types

| Augmentation | Purpose | Probability |
|--------------|---------|-------------|
| RandomResizedCrop | Scale & position invariance | Always |
| RandomHorizontalFlip | Left-right invariance | 50% |
| ColorJitter | Color invariance | 80% |
| RandomGrayscale | Remove color info | 20% |
| GaussianBlur | Texture invariance | 50% |

### Visual Example
![[simclr_concept_1_positive_pairs.png|600]]

---

## Code Implementation

```python
class SimCLRTransform:
    """
    Creates two correlated views of the same image.

    Each view undergoes INDEPENDENT random augmentation,
    resulting in different but semantically equivalent images.
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

        # Transform for second view (same structure, different random seed)
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
```

---

## Mathematical Formulation

### Augmentation Distribution

The augmentation function samples from a distribution:

$$
\mathcal{T} \sim \mathcal{A}
$$

Where $\mathcal{A}$ is the augmentation distribution defined by:
- Crop scale: $s \sim \text{Uniform}(0.2, 1.0)$
- Flip: $f \sim \text{Bernoulli}(0.5)$
- Color jitter: $(b, c, s, h) \sim \text{Uniform}(\pm 0.4)$ with $p=0.8$
- Grayscale: $g \sim \text{Bernoulli}(0.2)$
- Blur: $\sigma \sim \text{Uniform}(0.1, 2.0)$

### Positive Pair Definition

Given an image $x$ and two augmentation functions $\mathcal{T}_1, \mathcal{T}_2$:

$$
\text{Positive Pair} = (\tilde{x}_1, \tilde{x}_2) \text{ where } \tilde{x}_1 = \mathcal{T}_1(x), \tilde{x}_2 = \mathcal{T}_2(x)
$$

### Negative Pair Definition

For a batch of $N$ images with $2N$ views:

$$
\text{Negative pairs for } \tilde{x}_i = \{ \tilde{x}_k : k \neq i, k \neq i' \}
$$

Where $i'$ is the index of the positive pair (the other view of the same image).

---

## Why These Augmentations?

> [!success] Feature Invariance Hierarchy
> Different augmentations force different types of invariance:
>
> 1. **RandomCrop** → Scale & position invariance
> 2. **ColorJitter** → Color & lighting invariance
> 3. **Grayscale** → Color independence
> 4. **Blur** → Texture invariance
> 5. **Flip** → Horizontal symmetry

### Ablation Study Results (from SimCLR paper)

| Augmentation | Linear Probe Accuracy |
|--------------|----------------------|
| Crop only | ~55% |
| Crop + Color | ~65% |
| Crop + Color + Blur | ~70% |
| Full pipeline | **~76%** |

---

## Key Insights

> [!important] Strong Augmentation is Critical
> The strength of augmentation directly impacts representation quality. Too weak → model memorizes trivial features. Too strong → model fails to learn.

> [!warning] Augmentation Consistency
> Both views should be augmented with the SAME pipeline (but different random seeds). This ensures the task remains learnable.

---

## Related Notes

- [[02 - NT-Xent Loss]] - How we optimize the positive/negative relationships
- [[simclr_concept_1_positive_pairs.png]] - Visual examples
- [[06 - Code Part 1 - Augmentation]] - Detailed code walkthrough