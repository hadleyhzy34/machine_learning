"""
Sequence wrapper — runs a cell over a (B, T, D) input.

Cells must implement:
  - forward(x_t, state) -> new_state
  - init_state(batch_size, device, dtype) -> state
  - state_to_h(state) -> (B, hidden_size) tensor

For RNN/GRU "state == h" so these are trivial. For LSTM "state == (h, c)"
and the wrapper treats it opaquely. This is what lets the same wrapper drive
every benchmark task.

Readout modes:
  - 'last': use h_T → head, returns (B, out_size). For tasks where the
    answer is a single scalar/class (adding problem, sentiment).
  - 'mean': mean of h over time → head, returns (B, out_size).
  - 'all':  apply head at every timestep, returns (B, T, out_size).
    For sequence-to-sequence tasks (copy memory, language modeling).
"""

from typing import Callable, Literal, Optional

import torch
import torch.nn as nn


CellFactory = Callable[[int, int], nn.Module]


class RecurrentSeq(nn.Module):
    """
    Run a cell across time, project hidden states to `out_size`.

    Forward:
        x: (B, T, input_size)
        returns:
            (B, out_size)        if readout in {'last', 'mean'}
            (B, T, out_size)     if readout == 'all'
    """

    def __init__(
        self,
        cell_factory: CellFactory,
        input_size: int,
        hidden_size: int,
        out_size: int,
        readout: Literal["last", "mean", "all"] = "last",
    ):
        super().__init__()
        self.cell = cell_factory(input_size, hidden_size)
        self.hidden_size = hidden_size
        self.readout = readout
        self.head = nn.Linear(hidden_size, out_size)

    def forward(self, x: torch.Tensor, state0=None) -> torch.Tensor:
        B, T, _ = x.shape
        state = (
            state0
            if state0 is not None
            else self.cell.init_state(B, x.device, x.dtype)
        )

        if self.readout == "last":
            for t in range(T):
                state = self.cell(x[:, t], state)
            return self.head(self.cell.state_to_h(state))

        if self.readout == "mean":
            acc = x.new_zeros(B, self.hidden_size)
            for t in range(T):
                state = self.cell(x[:, t], state)
                acc = acc + self.cell.state_to_h(state)
            return self.head(acc / T)

        if self.readout == "all":
            # Stack h at every step, then one batched head call.
            # Memory cost is O(B*T*H) for activations — fine at the scales here.
            hs = []
            for t in range(T):
                state = self.cell(x[:, t], state)
                hs.append(self.cell.state_to_h(state))
            H = torch.stack(hs, dim=1)  # (B, T, hidden)
            return self.head(H)

        raise ValueError(f"unknown readout: {self.readout}")

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
