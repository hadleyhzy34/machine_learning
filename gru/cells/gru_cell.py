"""
GRU cell from scratch — naive and fused implementations.

Follows the exact PyTorch convention so we can verify numerical equivalence
against torch.nn.GRUCell. Two gotchas worth flagging up front:

1. PyTorch keeps TWO bias vectors (b_ih and b_hh), not one.
   Many textbook write-ups fold them into a single bias.

2. The reset gate r is applied AFTER W_hn @ h + b_hn, NOT before:
       n = tanh(W_in x + b_in + r * (W_hn h + b_hn))
   Applying r to h first (n = tanh(W_in x + W_hn (r * h))) is a different
   model and is the #1 source of "my GRU doesn't match PyTorch" bugs.

Equations (PyTorch convention):
    r_t = σ(W_ir x_t + b_ir + W_hr h_{t-1} + b_hr)        # reset gate
    z_t = σ(W_iz x_t + b_iz + W_hz h_{t-1} + b_hz)        # update gate
    n_t = tanh(W_in x_t + b_in + r_t ⊙ (W_hn h_{t-1} + b_hn))   # candidate
    h_t = (1 - z_t) ⊙ n_t + z_t ⊙ h_{t-1}                 # new hidden
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUCellNaive(nn.Module):
    """
    Naive GRU cell — one matmul per gate.

    Easier to read, slower to run. Useful as a reference and for ablations
    (e.g. removing a gate is just deleting a few lines).
    """

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        # Six weight matrices: one per gate, one for input and one for hidden.
        # Names match PyTorch: W_i* maps input, W_h* maps hidden.
        self.W_ir = nn.Parameter(torch.empty(hidden_size, input_size))
        self.W_iz = nn.Parameter(torch.empty(hidden_size, input_size))
        self.W_in = nn.Parameter(torch.empty(hidden_size, input_size))

        self.W_hr = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.W_hz = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.W_hn = nn.Parameter(torch.empty(hidden_size, hidden_size))

        if bias:
            self.b_ir = nn.Parameter(torch.empty(hidden_size))
            self.b_iz = nn.Parameter(torch.empty(hidden_size))
            self.b_in = nn.Parameter(torch.empty(hidden_size))
            self.b_hr = nn.Parameter(torch.empty(hidden_size))
            self.b_hz = nn.Parameter(torch.empty(hidden_size))
            self.b_hn = nn.Parameter(torch.empty(hidden_size))
        else:
            for name in ("b_ir", "b_iz", "b_in", "b_hr", "b_hz", "b_hn"):
                self.register_parameter(name, None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Match PyTorch's GRUCell init: U(-1/sqrt(hidden), +1/sqrt(hidden))
        # for every parameter, including biases.
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for p in self.parameters():
            nn.init.uniform_(p, -stdv, stdv)

    def forward(
        self, x: torch.Tensor, h: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        x: (batch, input_size)
        h: (batch, hidden_size) — defaults to zeros
        returns: (batch, hidden_size) — h_next
        """
        if h is None:
            h = x.new_zeros(x.size(0), self.hidden_size)

        # Reset gate: should this hidden be partially forgotten?
        r = torch.sigmoid(
            F.linear(x, self.W_ir, self.b_ir) + F.linear(h, self.W_hr, self.b_hr)
        )
        # Update gate: how much of old h to keep vs new candidate?
        z = torch.sigmoid(
            F.linear(x, self.W_iz, self.b_iz) + F.linear(h, self.W_hz, self.b_hz)
        )
        # Candidate hidden — note r multiplies (W_hn h + b_hn), NOT h alone.
        n = torch.tanh(
            F.linear(x, self.W_in, self.b_in)
            + r * F.linear(h, self.W_hn, self.b_hn)
        )
        h_next = (1 - z) * n + z * h
        return h_next

    # -- state convention used by RecurrentSeq -----------------------------
    # For RNN/GRU "state == h", so these are trivial. They exist so the
    # wrapper can treat all cells uniformly alongside LSTM (which has
    # state == (h, c)).
    def init_state(self, batch_size: int, device, dtype) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)

    @staticmethod
    def state_to_h(state: torch.Tensor) -> torch.Tensor:
        return state


class GRUCellFused(nn.Module):
    """
    Fused GRU cell — one matmul of size (3*hidden) for input, one for hidden.

    Matches what cuDNN does internally. Numerically identical to the naive
    version up to floating-point reorderings.
    """

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        # Stacked: rows [0:H] = r-gate, [H:2H] = z-gate, [2H:3H] = n-gate.
        self.weight_ih = nn.Parameter(torch.empty(3 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(3 * hidden_size, hidden_size))
        if bias:
            self.bias_ih = nn.Parameter(torch.empty(3 * hidden_size))
            self.bias_hh = nn.Parameter(torch.empty(3 * hidden_size))
        else:
            self.register_parameter("bias_ih", None)
            self.register_parameter("bias_hh", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for p in self.parameters():
            nn.init.uniform_(p, -stdv, stdv)

    def forward(
        self, x: torch.Tensor, h: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        x: (batch, input_size)
        h: (batch, hidden_size) — defaults to zeros
        returns: (batch, hidden_size)
        """
        if h is None:
            h = x.new_zeros(x.size(0), self.hidden_size)

        # One big matmul each for input and hidden, then chunk into 3.
        gi = F.linear(x, self.weight_ih, self.bias_ih)  # (B, 3H)
        gh = F.linear(h, self.weight_hh, self.bias_hh)  # (B, 3H)

        i_r, i_z, i_n = gi.chunk(3, dim=-1)
        h_r, h_z, h_n = gh.chunk(3, dim=-1)

        r = torch.sigmoid(i_r + h_r)
        z = torch.sigmoid(i_z + h_z)
        # Same gotcha: r multiplies the hidden contribution INCLUDING its bias.
        n = torch.tanh(i_n + r * h_n)
        h_next = (1 - z) * n + z * h
        return h_next

    def init_state(self, batch_size: int, device, dtype) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)

    @staticmethod
    def state_to_h(state: torch.Tensor) -> torch.Tensor:
        return state

    @torch.no_grad()
    def copy_from_torch_cell(self, ref: nn.GRUCell) -> None:
        """
        Copy parameters from a torch.nn.GRUCell into this fused cell.

        Used by the equivalence test — PyTorch's GRUCell stores params with
        the exact same layout (3*H stacked as r, z, n), so this is a direct
        tensor copy.
        """
        assert ref.input_size == self.input_size
        assert ref.hidden_size == self.hidden_size
        self.weight_ih.copy_(ref.weight_ih)
        self.weight_hh.copy_(ref.weight_hh)
        if self.bias and ref.bias:
            self.bias_ih.copy_(ref.bias_ih)
            self.bias_hh.copy_(ref.bias_hh)
