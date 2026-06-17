# Vision Transformer (ViT) from Scratch Demo

This demo implements a Vision Transformer from scratch and trains it on CIFAR-10, demonstrating the core concepts of the ViT architecture.

## Architecture Overview

The Vision Transformer consists of several key components:

1. **Patch Embedding**: Splits images into non-overlapping patches and projects them to embedding space
2. **Position Embeddings**: Adds positional information to each patch
3. **Class Token**: Special token that aggregates global information for classification
4. **Transformer Blocks**: Multiple blocks containing:
   - Multi-Head Self-Attention
   - Layer Normalization
   - Feed-Forward Network (MLP)
   - Residual Connections
5. **Classification Head**: Linear layer mapping class token to output logits

## Key Features

- Built entirely from scratch using PyTorch
- Implements all ViT components from first principles
- Trains on CIFAR-10 (32x32 RGB images, 10 classes)
- Includes data augmentation (random flip, rotation)
- Uses AdamW optimizer with cosine learning rate scheduling
- Visualizes patches, training progress, and attention maps

## Configuration

```python
patch_size: 4              # 4x4 patches for CIFAR-10
embed_dim: 256             # Token embedding dimension
num_heads: 8               # Number of attention heads
num_layers: 6              # Transformer blocks
mlp_ratio: 4               # MLP hidden dimension ratio
dropout: 0.1               # Dropout rate
batch_size: 128
epochs: 10
learning_rate: 3e-4
weight_decay: 0.05
```

## Usage

Run the demo:

```bash
python vit_demo.py
```

The script will:
1. Download and prepare CIFAR-10 dataset
2. Visualize how images are split into patches
3. Build and initialize the ViT model
4. Train for 10 epochs with progress bars
5. Evaluate on test set
6. Generate visualization plots

## Outputs

- `vit_training_history.png` - Training and validation loss/accuracy curves
- `vit_patches.png` - Visualization of patch grid on sample images
- `vit_attention.png` - Attention maps showing what the model focuses on

## Model Architecture Details

### Patch Embedding
- Uses a convolutional layer to extract patches efficiently
- Equivalent to splitting image into patches and applying a linear projection
- For CIFAR-10 (32x32) with patch_size=4: 64 patches total

### Multi-Head Self-Attention
- Computes queries, keys, values for each token
- Applies scaled dot-product attention: QK^T / sqrt(d_k)
- Uses softmax to create attention weights
- Multiple heads capture different types of relationships

### Transformer Block
- Pre-norm architecture (LayerNorm before attention/MLP)
- Residual connections for gradient flow
- GELU activation in MLP

### Position Embeddings
- Learnable sinusoidal-like embeddings
- Critical because attention has no notion of position
- Special [CLS] token prepended for classification

## Expected Performance

With the default configuration, you should see test accuracy in the range of 70-80% on CIFAR-10 after 10 epochs. The model has ~4.6M parameters.

## Understanding the Code

The code is organized into clear sections:

1. `ViTConfig` - Configuration dataclass
2. `PatchEmbedding` - Converts images to patch tokens
3. `MultiHeadAttention` - Self-attention mechanism
4. `MLP` - Feed-forward network
5. `TransformerBlock` - Complete transformer layer
6. `VisionTransformer` - Full model
7. Training and evaluation functions
8. Visualization functions

## References

Dosovitskiy, A., et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." ICLR 2021.

## Extension Ideas

- Experiment with different patch sizes (2, 4, 8)
- Add more transformer layers
- Try different attention mechanisms
- Implement ViT-Lite or DeiT variants
- Add learnable position embeddings interpolation for different image sizes
- Implement token pruning for efficiency