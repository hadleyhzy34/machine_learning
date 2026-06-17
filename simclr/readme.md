1. simclr_concepts_demo.py - Quick Educational Demo (Run this first!)

A fast, conceptual demonstration that explains:

- Concept 1: How augmentations create positive pairs
- Concept 2: NT-Xent loss mathematics and intuition
- Concept 3: Similarity matrix visualization
- Concept 4: SimCLR architecture (encoder + projection head)
- Concept 5: Linear probing evaluation

Generates visualizations:

- simclr_concept_1_positive_pairs.png - Shows augmentation pairs
- simclr_concept_2_similarity_matrix.png - Visualizes the contrastive loss

1. simclr_demo.py - Full Training Demo

A complete training demonstration with:

- Simple encoder network
- Full contrastive training loop
- PCA visualization of learned embeddings
- Interactive embedding evolution visualization

1. simclr_tutorial.py - Comprehensive Implementation

Full SimCLR implementation including:

- ResNet-18 encoder
- Complete NT-Xent loss implementation
- Training history tracking
- Linear probing evaluation
- Comparison with supervised baseline

1. simclr_tutorial.ipynb - Interactive Notebook

Jupyter notebook version with:

- Step-by-step explanations
- Mathematical formulas
- Interactive code cells
- Exercises for experimentation

Key Concepts Taught

1. Positive Pairs: Two augmented views of the same image
2. NT-Xent Loss: Pulls positives together, pushes negatives apart
3. Architecture: Encoder (kept) + Projection Head (discarded)
4. Self-Supervised: No labels needed during pretraining
5. Linear Probing: Evaluate representation quality

Run python simclr_concepts_demo.py first for a quick overview, then explore the other files for deeper implementation details.
