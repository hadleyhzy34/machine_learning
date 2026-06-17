"""
Vision Transformer with configurable layer normalization strategies.

Implements:
- No LayerNorm (baseline)
- Post-LN (original transformer)
- Pre-LN (modern approach)
- RMSNorm (used in LLaMA)
"""

import math
from typing import Literal, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Simpler than LayerNorm - normalizes by RMS without mean centering.
    Used in LLaMA and other modern architectures.

    Formula: x * rsqrt(mean(x^2) + eps) * weight
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calculate RMS
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class IdentityNorm(nn.Module):
    """Identity layer for 'no normalization' variant."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def get_norm_layer(
    norm_type: str, dim: int, eps: float = 1e-6
) -> nn.Module:
    """Factory function to create the appropriate normalization layer."""
    if norm_type == "none":
        return IdentityNorm()
    elif norm_type == "rms":
        return RMSNorm(dim, eps)
    else:
        return nn.LayerNorm(dim, eps)


class PatchEmbedding(nn.Module):
    """Split image into patches and embed them."""

    def __init__(self, img_size: int, patch_size: int, embed_dim: int):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels=3,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) -> (B, D, H/P, W/P)
        x = self.proj(x)
        # (B, D, H/P, W/P) -> (B, D, N)
        x = x.flatten(2)
        # (B, D, N) -> (B, N, D)
        x = x.transpose(1, 2)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention mechanism."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, \
            "embed_dim must be divisible by num_heads"

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        x = self.proj(x)
        x = self.proj_dropout(x)
        return x


class MLP(nn.Module):
    """Feed-forward network with GELU activation."""

    def __init__(self, in_features: int, hidden_features: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    Transformer encoder block with configurable normalization placement.

    Args:
        embed_dim: Dimension of embeddings
        num_heads: Number of attention heads
        mlp_ratio: Ratio of MLP hidden dim to embed_dim
        dropout: Dropout rate
        norm_type: "none", "post", "pre", or "rms"

    Normalization variants:
        - "none": No normalization (baseline)
        - "post": Post-LN: norm after attention/MLP (before residual add)
        - "pre": Pre-LN: norm before attention/MLP
        - "rms": RMSNorm before attention/MLP
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        norm_type: str = "pre",
    ):
        super().__init__()
        self.norm_type = norm_type

        # Create normalization layers
        if norm_type == "post":
            # Post-LN: norm after residual connection
            self.norm1 = nn.LayerNorm(embed_dim)
            self.norm2 = nn.LayerNorm(embed_dim)
            self.pre_norm1 = None
            self.pre_norm2 = None
        else:
            # Pre-LN, RMSNorm, or none: norm before attention/MLP
            self.norm1 = get_norm_layer(norm_type, embed_dim)
            self.norm2 = get_norm_layer(norm_type, embed_dim)
            self.pre_norm1 = None
            self.pre_norm2 = None

        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.mlp = MLP(embed_dim, embed_dim * mlp_ratio, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm_type == "post":
            # Post-LN: attention -> residual -> norm
            x = self.norm1(x + self.attn(x))
            x = self.norm2(x + self.mlp(x))
        else:
            # Pre-LN, RMSNorm, or none: norm -> attention/MLP -> residual
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer with configurable layer normalization.

    Args:
        img_size: Size of input images
        patch_size: Size of patches
        embed_dim: Dimension of embeddings
        num_heads: Number of attention heads
        num_layers: Number of transformer blocks
        mlp_ratio: Ratio of MLP hidden dim to embed_dim
        num_classes: Number of output classes
        dropout: Dropout rate
        norm_type: "none", "post", "pre", or "rms"
    """

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        mlp_ratio: int,
        num_classes: int,
        dropout: float = 0.0,
        norm_type: str = "pre",
    ):
        super().__init__()
        self.norm_type = norm_type
        self.num_layers = num_layers

        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, embed_dim)

        # Class token and position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.n_patches + 1, embed_dim)
        )

        self.pos_drop = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout, norm_type)
            for _ in range(num_layers)
        ])

        # Final layer norm (only for pre-LN and RMSNorm variants)
        if norm_type in ["pre", "rms"]:
            self.norm = get_norm_layer(norm_type, embed_dim)
        else:
            self.norm = None

        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using truncated normal initialization."""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        def _init_layer(m):
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, RMSNorm):
                nn.init.ones_(m.weight)
            elif isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.apply(_init_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)

        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add position embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final layer norm (for pre-LN variants)
        if self.norm is not None:
            x = self.norm(x)

        # Extract class token and classify
        cls_token = x[:, 0]
        logits = self.head(cls_token)
        return logits

    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_vit_model(config) -> VisionTransformer:
    """Create a ViT model from config."""
    return VisionTransformer(
        img_size=config.image_size,
        patch_size=config.patch_size,
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        mlp_ratio=config.mlp_ratio,
        num_classes=config.num_classes,
        dropout=config.dropout,
        norm_type=config.norm_type,
    )
