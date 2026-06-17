"""
Configuration for Layer Normalization Ablation Study.
"""

from dataclasses import dataclass, field
from typing import Literal
import torch


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment variant."""

    # Model architecture
    patch_size: int = 4
    embed_dim: int = 256
    num_heads: int = 8
    num_layers: int = 6
    mlp_ratio: int = 4
    dropout: float = 0.1

    # Training
    batch_size: int = 128
    epochs: int = 20
    learning_rate: float = 3e-4
    weight_decay: float = 0.05

    # Data
    image_size: int = 32
    num_classes: int = 10

    # Normalization variant
    norm_type: Literal["none", "post", "pre", "rms"] = "pre"

    @property
    def device(self) -> str:
        # CUDA has highest priority for best performance
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"


@dataclass
class AblationConfig:
    """Configuration for running all ablation experiments."""

    # All normalization variants to test
    variants: list[str] = field(
        default_factory=lambda: ["none", "post", "pre", "rms"]
    )

    # Base config (shared settings)
    base_config: ExperimentConfig = field(default_factory=ExperimentConfig)

    # Output directory
    results_dir: str = "results"

    # Random seed for reproducibility
    seed: int = 42


def get_variant_description(norm_type: str) -> str:
    """Get a human-readable description for each normalization variant."""
    descriptions = {
        "none": "No LayerNorm (baseline without normalization)",
        "post": "Post-LN (original transformer: norm after attention/MLP)",
        "pre": "Pre-LN (modern approach: norm before attention/MLP)",
        "rms": "RMSNorm (root mean square normalization, used in LLaMA)",
    }
    return descriptions.get(norm_type, f"Unknown variant: {norm_type}")


def get_variant_short_name(norm_type: str) -> str:
    """Get a short name for plotting."""
    names = {
        "none": "No-LN",
        "post": "Post-LN",
        "pre": "Pre-LN",
        "rms": "RMSNorm",
    }
    return names.get(norm_type, norm_type)
