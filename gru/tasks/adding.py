"""
The Adding Problem (Hochreiter & Schmidhuber, 1997).

Each input is a sequence of length T with TWO channels:
    - channel 0: random scalars in [0, 1]
    - channel 1: a 0/1 mask, exactly two 1s placed at random positions

Target: the SUM of the two scalars at the masked positions.

Why this task is the right benchmark:
- Trivial for any model that can route information across T steps.
- Vanilla RNNs fail beyond T~50 due to vanishing gradients; gated
  models (GRU/LSTM) keep working out to T=500+.
- A model that always predicts 1.0 (the mean of two uniforms) gets
  MSE ~0.167 — that's our "did it learn anything?" floor.

Generated on the fly: no dataset to download, infinite samples.
"""

from typing import Tuple

import torch


def make_adding_batch(
    batch_size: int,
    seq_len: int,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        x: (B, T, 2) — channel 0 random scalars, channel 1 the 0/1 mask
        y: (B, 1)   — sum of the two masked scalars
    """
    # Channel 0: scalars
    vals = torch.rand(batch_size, seq_len, device=device, generator=generator)

    # Channel 1: mask with exactly two 1s per sample. We pick two distinct
    # positions per row by argsort'ing random noise — vectorized, no Python loop.
    noise = torch.rand(batch_size, seq_len, device=device, generator=generator)
    idx = noise.argsort(dim=1)[:, :2]  # (B, 2) — first two of the random perm
    mask = torch.zeros(batch_size, seq_len, device=device)
    mask.scatter_(1, idx, 1.0)

    x = torch.stack([vals, mask], dim=-1)  # (B, T, 2)
    # Target: sum of vals at the masked positions.
    y = (vals * mask).sum(dim=1, keepdim=True)  # (B, 1)
    return x, y


# Useful constant: variance of (U + U') where U, U' ~ Uniform(0, 1) is 1/6,
# and the mean is 1.0. So MSE for a constant predictor of 1.0 is ~0.1667.
# A model "has learned the task" once MSE drops well below this.
NAIVE_BASELINE_MSE = 1.0 / 6.0


if __name__ == "__main__":
    # Sanity check.
    g = torch.Generator().manual_seed(0)
    x, y = make_adding_batch(4, 20, generator=g)
    print(f"x shape: {tuple(x.shape)}, y shape: {tuple(y.shape)}")
    print(f"mask sums per row (should all be 2): {x[..., 1].sum(dim=1).tolist()}")
    print(f"targets (should match manual sum): {y.squeeze().tolist()}")
    manual = (x[..., 0] * x[..., 1]).sum(dim=1)
    print(f"manual recompute:                  {manual.tolist()}")
    print(f"naive constant-1.0 baseline MSE:   {NAIVE_BASELINE_MSE:.4f}")
