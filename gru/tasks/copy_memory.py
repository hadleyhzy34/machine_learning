"""
Copy-Memory Problem (Hochreiter & Schmidhuber 1997; Arjovsky et al. 2016).

Setup. Vocabulary of 10 symbols:
    1..8  : data symbols
    0     : blank
    9     : "go" cue (also called the delimiter)

Each sample has length T + 2K where T is the delay:
    positions 0..K-1            : K random data symbols  (the message)
    positions K..K+T-2          : T-1 blanks             (the delay)
    position  K+T-1             : single go cue          (the prompt)
    positions K+T..K+T+K-1      : K blanks               (model writes here)

The TARGET sequence has the same length:
    positions 0..K+T-1          : blanks
    positions K+T..K+T+K-1      : the original K data symbols, in order

So: the model must memorize K symbols, hold them across T-1 silent steps,
and reproduce them exactly when prompted by the go cue.

Why this task is the right benchmark:
- Pure memory test — no pattern to extrapolate, no statistical regularity
  to exploit. Either the model retained the symbols or it didn't.
- Constant-blank baseline gets a known, easily-computed loss; any model
  that hasn't beaten it is doing nothing.
- Vanilla RNN saturates at the baseline for moderate T. LSTM/GRU break
  through with enough capacity and training. Sharper signal than adding.

Inputs are returned as one-hot tensors (B, T+2K, 10) so they slot into
RecurrentSeq directly. Targets are class indices (B, T+2K) suitable for
cross_entropy.
"""

from dataclasses import dataclass
from typing import Tuple

import torch


N_DATA_SYMBOLS = 8     # values 1..8 are data
BLANK = 0
GO_CUE = 9
VOCAB_SIZE = 10        # 0 (blank), 1..8 (data), 9 (go cue)


@dataclass(frozen=True)
class CopyConfig:
    K: int = 10        # number of symbols to remember
    T: int = 100       # delay (sequence length grows as T + 2K)

    @property
    def total_len(self) -> int:
        return self.T + 2 * self.K


def make_copy_batch(
    cfg: CopyConfig,
    batch_size: int,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        x: (B, L, vocab_size) one-hot
        y: (B, L) long class indices in [0, vocab_size)
    where L = T + 2K.
    """
    L = cfg.total_len
    K = cfg.K

    # Targets default to blank everywhere.
    y = torch.zeros(batch_size, L, dtype=torch.long, device=device)
    # Sample the K data symbols (values 1..8).
    msg = torch.randint(
        low=1,
        high=N_DATA_SYMBOLS + 1,
        size=(batch_size, K),
        device=device,
        generator=generator,
    )
    # Place the answer in the last K positions of the target.
    y[:, -K:] = msg

    # Build the input as integer ids first (it's clearer), then one-hot.
    x_ids = torch.zeros(batch_size, L, dtype=torch.long, device=device)
    x_ids[:, :K] = msg                           # message at the start
    x_ids[:, -K - 1] = GO_CUE                    # go cue right before the answer slot
    # Everything else stays blank (0).

    x = torch.nn.functional.one_hot(x_ids, num_classes=VOCAB_SIZE).float()
    return x, y


def naive_baseline_loss(cfg: CopyConfig) -> float:
    """
    Cross-entropy of "always predict blank" mixed with the right uniform
    distribution over data symbols at the K answer positions.

    A model that's learned NOTHING about the message will at best predict
    blanks at K+T positions and a uniform-over-data distribution at the K
    answer positions. Average cross-entropy:

        avg = (K / L) * log(N_DATA_SYMBOLS)

    Per-position loss below this means the model is starting to recall
    the message.
    """
    import math

    return (cfg.K / cfg.total_len) * math.log(N_DATA_SYMBOLS)


if __name__ == "__main__":
    cfg = CopyConfig(K=5, T=20)
    g = torch.Generator().manual_seed(0)
    x, y = make_copy_batch(cfg, batch_size=2, generator=g)
    print(f"cfg: K={cfg.K}, T={cfg.T}, total_len={cfg.total_len}")
    print(f"x shape: {tuple(x.shape)}, y shape: {tuple(y.shape)}")

    # Pretty-print one sample to make the structure obvious.
    ids = x.argmax(-1)[0].tolist()
    target = y[0].tolist()
    L = cfg.total_len
    pos_labels = "".join("M" if i < cfg.K else "G" if i == L - cfg.K - 1 else "_"
                          for i in range(L))
    print(f"slots:   {pos_labels}     (M=message, G=go cue, _=blank slot)")
    print(f"input:   {''.join(str(v) for v in ids)}")
    print(f"target:  {''.join(str(v) for v in target)}")

    print(f"\nnaive constant-blank baseline loss: {naive_baseline_loss(cfg):.4f} nats/pos")
