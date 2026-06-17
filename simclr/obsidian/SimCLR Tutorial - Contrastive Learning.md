---
tags:
  - ml/contrastive-learning
  - ml/self-supervised
  - simclr
  - tutorial
created: 2026-03-04
status: complete
---

# SimCLR Tutorial - Contrastive Learning

> [!tip] Overview
> This tutorial provides a comprehensive guide to **SimCLR (Simple Framework for Contrastive Learning of Visual Representations)**, a groundbreaking self-supervised learning method that learns powerful visual representations without labels.

## What You'll Learn

1. [[#What is Contrastive Learning?]]
2. [[#Key Components of SimCLR]]
3. [[#Tutorial Structure]]

---

## What is Contrastive Learning?

Contrastive learning is a **self-supervised** learning paradigm that learns representations by comparing samples:

- **Positive pairs**: Samples that should be similar (e.g., different views of the same image)
- **Negative pairs**: Samples that should be different (e.g., different images)

> [!quote] Core Intuition
> "Pull positive pairs together, push negative pairs apart in the embedding space."

### Supervised vs Contrastive Learning

| Aspect | Supervised Learning | Contrastive Learning |
|--------|--------------------|-----------------------|
| Labels | Required | Not required |
| Objective | Predict class | Learn similarity |
| Data efficiency | Lower | Higher |
| Transferability | Task-specific | General representations |

---

## Key Components of SimCLR

### 1. Data Augmentation
![[simclr_concept_1_positive_pairs.png|500]]

- **What**: Create two augmented views of each image
- **Why**: Forces model to learn invariant features
- **See**: [[01 - Data Augmentation Pipeline|Data Augmentation Details]]

### 2. NT-Xent Loss
![[simclr_concept_2_similarity_matrix.png|500]]

- **What**: Normalized Temperature-scaled Cross-Entropy Loss
- **Why**: Optimize the embedding space organization
- **See**: [[02 - NT-Xent Loss|Loss Function Details]]

### 3. Encoder + Projection Head
```
Image → [Encoder] → Features (512-D) → [Projection Head] → Embedding (128-D)
                                           ↑
                                    Discarded after training
```

- **What**: Two-part neural network architecture
- **Why**: Projection head improves representation learning
- **See**: [[03 - SimCLR Architecture|Architecture Details]]

### 4. Linear Probing
- **What**: Evaluate by training linear classifier on frozen features
- **Why**: Measures representation quality
- **See**: [[05 - Linear Probing Evaluation|Evaluation Details]]

---

## Tutorial Structure

### Part 1: Foundations
- [[01 - Data Augmentation Pipeline]] - Creating positive pairs
- [[02 - NT-Xent Loss]] - Mathematical formulation
- [[03 - SimCLR Architecture]] - Model design

### Part 2: Implementation
- [[04 - Training Pipeline]] - Complete training loop
- [[05 - Linear Probing Evaluation]] - How to evaluate

### Part 3: Code Walkthrough
- [[06 - Code Part 1 - Augmentation]]
- [[07 - Code Part 2 - Loss Function]]
- [[08 - Code Part 3 - Model]]
- [[09 - Code Part 4 - Training]]
- [[10 - Code Part 5 - Evaluation]]

### Part 4: Visualizations
- [[simclr_concept_1_positive_pairs.png|Positive Pairs Visualization]]
- [[simclr_concept_2_similarity_matrix.png|Similarity Matrix]]
- [[simclr_demo_results.png|Training Results]]
- [[simclr_embedding_evolution.png|Embedding Evolution]]

---

## Quick Start

```bash
# Run concepts demo first (quick, no training)
python simclr_concepts_demo.py

# Then run full training demo
python simclr_demo.py

# Or comprehensive implementation
python simclr_tutorial.py
```

---

## Key Insights

> [!important] Why SimCLR Works
> The model must recognize the same image under **different augmentations**. This forces learning of:
> - **Invariant features**: Shape, structure, object parts
> - **Not superficial features**: Exact color, position, texture

> [!warning] Important Considerations
> - **Large batch sizes** (256-512) are crucial for enough negative samples
> - **Strong augmentations** are key to good representations
> - **Projection head** is removed after training - only encoder is kept

---

## Related Topics

- [[MoCo]] - Momentum Contrast
- [[BYOL]] - Bootstrap Your Own Latent
- [[SwAV]] - Swapping Assignments between Views
- [[CLIP]] - Contrastive Language-Image Pre-training

---

## References

- Chen, T., et al. (2020). "A Simple Framework for Contrastive Learning of Visual Representations"
- Paper: https://arxiv.org/abs/2002.05709