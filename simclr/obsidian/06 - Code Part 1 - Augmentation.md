---
tags:
  - ml/contrastive-learning
  - code-analysis
  - simclr
created: 2026-03-04
---

# Code Part 1 - Data Augmentation

> [!abstract] Overview
> Detailed code analysis of the data augmentation pipeline in SimCLR.

## Code Location
`simclr_tutorial.py` lines 56-97

---

## Complete Code

```python
class SimCLRTransform:
    """
    SimCLR Data Augmentation Pipeline

    Creates two correlated views of the same image using random augmentations.
    These two views form a POSITIVE pair for contrastive learning.
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

        # Transform for second view (different random augmentation)
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

## Line-by-Line Analysis

### Line 1: RandomResizedCrop

```python
transforms.RandomResizedCrop(input_size, scale=(min_scale, 1.0))
```

**Mathematical Operation:**

1. **Random crop selection:**
   $$\text{crop} \sim \text{Uniform}(x_{min}, x_{max}, w, h)$$

   Where the crop area ratio:
   $$\frac{w \times h}{W \times H} \sim \text{Uniform}(s_{min}, 1.0)$$

2. **Resize to target size:**
   $$\text{output} = \text{Resize}(\text{crop}, \text{size}=input\_size)$$

**Effect:**
- Randomly selects a region of the image
- Scales it to the original size
- Forces the model to recognize objects at different scales and positions

> [!example] Example
> For a 32×32 image with `min_scale=0.2`:
> - Crop area can range from $0.2 \times 32 \times 32 = 205$ to $1.0 \times 32 \times 32 = 1024$ pixels
> - Crop size ranges from $\approx 14 \times 14$ to $32 \times 32$

---

### Line 2: RandomHorizontalFlip

```python
transforms.RandomHorizontalFlip(p=0.5)
```

**Mathematical Operation:**

$$\text{output} = \begin{cases}
\text{flip}(x) & \text{with probability } p = 0.5 \\
x & \text{with probability } 1-p = 0.5
\end{cases}$$

Where $\text{flip}(x)[i,j,c] = x[i, W-1-j, c]$

**Effect:**
- Mirrors the image horizontally
- Forces the model to learn horizontal invariance
- Note: Not used when horizontal direction matters (e.g., text)

---

### Line 3: ColorJitter

```python
transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)
```

**Mathematical Operation:**

Color jitter applies random adjustments to brightness, contrast, saturation, and hue:

1. **Brightness:**
   $$\text{img}_{new} = \text{img} + \Delta_b, \quad \Delta_b \sim \text{Uniform}(-0.4, 0.4)$$

2. **Contrast:**
   $$\text{img}_{new} = \text{img} \times \Delta_c, \quad \Delta_c \sim \text{Uniform}(\max(0, 1-0.4), 1+0.4)$$

3. **Saturation:**
   $$\text{img}_{new} = \text{mix}(\text{img}, \text{grayscale}(\text{img}), \Delta_s)$$
   $$\Delta_s \sim \text{Uniform}(\max(0, 1-0.4), 1+0.4)$$

4. **Hue:**
   $$\text{img}_{new} = \text{adjust\_hue}(\text{img}, \Delta_h)$$
   $$\Delta_h \sim \text{Uniform}(-0.1, 0.1)$$

**Effect:**
- Dramatically changes the appearance of the image
- Forces the model to focus on shape and structure, not color

---

### Line 4: RandomGrayscale

```python
transforms.RandomGrayscale(p=0.2)
```

**Mathematical Operation:**

With probability $p=0.2$:

$$\text{gray}(x) = 0.2989 \cdot R + 0.5870 \cdot G + 0.1140 \cdot B$$

$$\text{output} = \begin{cases}
[\text{gray}(x), \text{gray}(x), \text{gray}(x)] & \text{with prob } 0.2 \\
x & \text{with prob } 0.8
\end{cases}$$

**Effect:**
- Occasionally removes all color information
- Forces the model to learn shape-based features
- Prevents over-reliance on color for recognition

---

### Line 5: GaussianBlur

```python
transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
```

**Mathematical Operation:**

Convolution with Gaussian kernel:

$$G(x, y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2 + y^2}{2\sigma^2}}$$

$$\text{output}(x, y) = \sum_{i=-k}^{k} \sum_{j=-k}^{k} G(i, j) \cdot \text{input}(x-i, y-j)$$

Where $\sigma \sim \text{Uniform}(0.1, 2.0)$

**Effect:**
- Softens texture and fine details
- Forces the model to focus on larger structures
- Simulates out-of-focus or low-resolution images

---

### Line 6-7: ToTensor and Normalize

```python
transforms.ToTensor(),
transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
```

**Mathematical Operation:**

1. **ToTensor:** Converts PIL Image (0-255) to Tensor (0-1)
   $$\text{tensor} = \frac{\text{PIL\_Image}}{255.0}$$

2. **Normalize:** Standardizes the values
   $$\text{output} = \frac{\text{tensor} - \mu}{\sigma}$$

   With $\mu = 0.5$ and $\sigma = 0.5$:
   $$\text{output} = \frac{\text{tensor} - 0.5}{0.5} = 2 \cdot \text{tensor} - 1$$

   This maps [0, 1] to [-1, 1]

**Effect:**
- Standardizes input range for stable training
- Centered around 0 with range [-1, 1]

---

## Full Pipeline Visualization

```
Original Image (PIL, 32×32, RGB)
         │
         ▼
┌─────────────────────┐
│  RandomResizedCrop  │ ← Random crop + resize
│   scale=(0.2, 1.0)  │   Forces scale/position invariance
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│ RandomHorizontalFlip│ ← p=0.5
│                     │   Forces left-right invariance
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│    ColorJitter      │ ← p=0.8
│  (0.4,0.4,0.4,0.1)  │   Forces color invariance
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│  RandomGrayscale    │ ← p=0.2
│                     │   Forces shape-based recognition
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│   GaussianBlur      │ ← σ∈(0.1, 2.0)
│   kernel_size=3     │   Forces structure-based recognition
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│     ToTensor        │ ← Converts to tensor [0,1]
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│     Normalize       │ ← Maps to [-1,1]
│    (0.5, 0.5, 0.5)  │
└─────────────────────┘
         │
         ▼
Augmented Tensor (3×32×32, range[-1,1])
```

---

## Why Two Separate Transforms?

```python
self.transform1 = transforms.Compose([...])
self.transform2 = transforms.Compose([...])
```

> [!important] Key Design Choice
> We use TWO separate transform objects with the SAME pipeline.
> This ensures:
> 1. **Independent randomness**: Each transform has its own random state
> 2. **Different augmentations**: The same image gets TWO different views
> 3. **Consistent structure**: Both views undergo the same type of transforms

### Alternative (Wrong Approach)

```python
# WRONG: Same random state = identical augmentations
transform = transforms.Compose([...])
view1 = transform(img)  # Random crop at position A
view2 = transform(img)  # Same crop at position A (same random seed!)
# Result: view1 ≈ view2 (too easy for model!)
```

### Correct Approach (SimCLR)

```python
# CORRECT: Different random states = different augmentations
transform1 = transforms.Compose([...])
transform2 = transforms.Compose([...])
view1 = transform1(img)  # Random crop at position A
view2 = transform2(img)  # Random crop at position B (different random seed!)
# Result: view1 ≠ view2 (meaningful contrastive task!)
```

---

## Mathematical Summary

The augmentation pipeline can be expressed as:

$$\tilde{x}_1 = \mathcal{T}_1(x), \quad \tilde{x}_2 = \mathcal{T}_2(x)$$

Where each $\mathcal{T}$ is a composition:

$$\mathcal{T} = \mathcal{N} \circ \text{ToTensor} \circ \mathcal{G} \circ \mathcal{C} \circ \mathcal{F} \circ \mathcal{R}$$

With:
- $\mathcal{R}$: RandomResizedCrop
- $\mathcal{F}$: RandomHorizontalFlip
- $\mathcal{C}$: ColorJitter (stochastic)
- $\mathcal{G}$: RandomGrayscale/GaussianBlur (stochastic)
- $\mathcal{N}$: Normalize

---

## Related Notes

- [[01 - Data Augmentation Pipeline]] - Conceptual overview
- [[simclr_concept_1_positive_pairs.png]] - Visual examples