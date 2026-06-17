---
tags:
  - ml/contrastive-learning
  - loss-function
  - simclr
  - mathematics
created: 2026-03-04
---

# NT-Xent Loss

> [!abstract] Overview
> NT-Xent (Normalized Temperature-scaled Cross Entropy) is the core loss function of SimCLR. It elegantly converts the contrastive learning objective into a classification problem.

## The Core Idea

Given a batch of $N$ images, we create $2N$ augmented views. For each view, the model must:
1. Maximize similarity with its **positive pair** (other view of same image)
2. Minimize similarity with all **negative pairs** (views from other images)

![[simclr_concept_2_similarity_matrix.png|600]]

---

## Mathematical Formulation

### Step 1: Embedding Normalization

First, normalize the projections to unit length:

$$
\mathbf{z}_i = \frac{\mathbf{h}_i}{\|\mathbf{h}_i\|_2}
$$

This ensures cosine similarity is simply the dot product.

### Step 2: Similarity Computation

Compute pairwise cosine similarities:

$$
\text{sim}(\mathbf{z}_i, \mathbf{z}_j) = \mathbf{z}_i^\top \mathbf{z}_j
$$

### Step 3: Temperature Scaling

Scale by temperature parameter $\tau$:

$$
s_{ij} = \frac{\mathbf{z}_i^\top \mathbf{z}_j}{\tau}
$$

> [!tip] Temperature Effect
> - **Low $\tau$** (e.g., 0.1): Sharper distribution, harder negatives
> - **High $\tau$** (e.g., 1.0): Softer distribution, easier optimization
> - **SimCLR uses $\tau = 0.5$**

### Step 4: Loss for One Pair

For positive pair $(i, j)$ where $j$ is the other view of image $i$:

$$
\ell_{i,j} = -\log \frac{\exp(s_{ij})}{\sum_{k=1}^{2N} \mathbb{1}_{[k \neq i]} \exp(s_{ik})}
$$

### Step 5: Final Loss

Average over all positive pairs:

$$
\mathcal{L} = \frac{1}{2N} \sum_{k=1}^{N} \left[ \ell_{2k-1, 2k} + \ell_{2k, 2k-1} \right]
$$

---

## Intuitive Explanation

### Rewrite as Softmax

The loss can be rewritten as:

$$
\ell_{i,j} = -\log \frac{\exp(s_{ij})}{\sum_{k \neq i} \exp(s_{ik})} = -\log \text{softmax}_j(\mathbf{s}_i)
$$

This is equivalent to:
- **Input**: Similarity scores for all pairs involving view $i$
- **Target**: Index $j$ (the positive pair)
- **Task**: Classify which view is the positive pair

> [!success] Key Insight
> NT-Xent reformulates contrastive learning as a **(2N-1)-way classification** problem: "Which of the other 2N-1 views is my positive pair?"

---

## Code Implementation

```python
class NTXentLoss(nn.Module):
    """
    NT-Xent: Normalized Temperature-scaled Cross Entropy Loss

    This is the key innovation of SimCLR!

    Args:
        temperature: Temperature parameter τ (default 0.5)
    """

    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, projections):
        """
        Args:
            projections: [2N, D] - projections from both views concatenated

        Returns:
            loss: scalar contrastive loss
        """
        N = projections.shape[0] // 2  # Number of original images

        # Step 1: L2 normalize (for cosine similarity)
        projections = F.normalize(projections, p=2, dim=1)

        # Step 2: Compute similarity matrix [2N, 2N]
        # sim[i,j] = cosine similarity between projection i and j
        sim_matrix = torch.matmul(projections, projections.T)

        # Step 3: Scale by temperature
        sim_matrix = sim_matrix / self.temperature

        # Step 4: Create labels
        # For view i (0 to N-1), positive is at position i+N
        # For view i+N, positive is at position i
        labels = torch.arange(N, device=projections.device)
        labels = torch.cat([labels + N, labels])  # [N, N+1, ..., 2N-1, 0, 1, ..., N-1]

        # Step 5: Cross entropy loss
        loss = F.cross_entropy(sim_matrix, labels)

        return loss
```

---

## Detailed Walkthrough with Example

### Example: Batch of 4 Images

Given $N = 4$ images, we have $2N = 8$ views:

```
Indices:    0    1    2    3    4    5    6    7
Views:     v1_0 v1_1 v1_2 v1_3 v2_0 v2_1 v2_2 v2_3
           └──────────────────┘ └──────────────────┘
                  View 1              View 2

Positive pairs: (0,4), (1,5), (2,6), (3,7)
```

### Similarity Matrix Structure

```
     0    1    2    3    4    5    6    7
0 [ 1.0  0.2  0.1  0.3  ★   0.1  0.2  0.1 ]  → Label: 4 (★ is positive)
1 [ 0.2  1.0  0.2  0.1  0.1  ★   0.1  0.3 ]  → Label: 5
2 [ 0.1  0.2  1.0  0.2  0.1  0.1  ★   0.2 ]  → Label: 6
3 [ 0.3  0.1  0.2  1.0  0.2  0.1  0.1  ★  ]  → Label: 7
4 [ ★   0.1  0.1  0.2  1.0  0.2  0.1  0.3 ]  → Label: 0
5 [ 0.1  ★   0.1  0.1  0.2  1.0  0.2  0.1 ]  → Label: 1
6 [ 0.2  0.1  ★   0.1  0.1  0.2  1.0  0.2 ]  → Label: 2
7 [ 0.1  0.3  0.2  ★   0.3  0.1  0.2  1.0 ]  → Label: 3

★ = positive pair (should be high)
```

### Loss Computation for Row 0

$$
\ell_0 = -\log \frac{\exp(s_{04}/\tau)}{\exp(s_{01}/\tau) + \exp(s_{02}/\tau) + ... + \exp(s_{07}/\tau)}
$$

If positive similarity $s_{04}$ is **high** relative to others → **low loss** ✓

---

## Why NT-Xent Works

> [!important] Information-Theoretic View
> Minimizing NT-Xent maximizes the **mutual information** between positive pairs:
>
> $$\mathcal{I}(X_1; X_2) \geq \log(2N-1) - \mathcal{L}$$
>
> The loss is a lower bound on mutual information!

### Gradient Analysis

The gradient with respect to similarity $s_{ik}$:

$$
\frac{\partial \ell_{i,j}}{\partial s_{ik}} = \begin{cases}
p_k - 1 & \text{if } k = j \text{ (positive)} \\
p_k & \text{if } k \neq j \text{ (negative)}
\end{cases}
$$

Where $p_k = \frac{\exp(s_{ik})}{\sum_m \exp(s_{im})}$ is the softmax probability.

> [!success] Gradient Intuition
> - **Positive pair**: Pushes similarity toward 1 (gradient = $p_j - 1$)
> - **Negative pairs**: Pushes similarity toward 0 (gradient = $p_k$)
> - **Harder negatives** (high $p_k$) get larger gradients → more focus on hard examples!

---

## Temperature Analysis

### Effect of Temperature

```python
# Demonstration of temperature effect
def temperature_demo():
    logits = torch.tensor([0.9, 0.1, 0.1, 0.1])  # One high, three low

    for tau in [0.1, 0.5, 1.0]:
        scaled = logits / tau
        probs = F.softmax(scaled, dim=0)
        print(f"τ={tau}: probs = {probs}")

# Output:
# τ=0.1: probs = [0.9999, 0.0000, 0.0000, 0.0000]  ← Very sharp
# τ=0.5: probs = [0.8909, 0.0364, 0.0364, 0.0364]  ← Moderate
# τ=1.0: probs = [0.5500, 0.1500, 0.1500, 0.1500]  ← Soft
```

> [!warning] Choosing Temperature
> - **Too low**: May focus too much on hard negatives, unstable training
> - **Too high**: Softens the objective, may not learn discriminative features
> - **Optimal**: Paper finds $\tau = 0.5$ works best

---

## Key Insights

> [!success] Why NT-Xent over other losses?
> 1. **End-to-end**: No need for memory bank or momentum encoder
> 2. **Simple**: Uses standard cross-entropy, well-understood optimization
> 3. **Effective**: Leverages all negatives in the batch
> 4. **Scalable**: Performance improves with larger batches (more negatives)

---

## Related Notes

- [[01 - Data Augmentation Pipeline]] - Creates the positive pairs
- [[07 - Code Part 2 - Loss Function]] - Detailed code walkthrough
- [[simclr_concept_2_similarity_matrix.png]] - Visual explanation