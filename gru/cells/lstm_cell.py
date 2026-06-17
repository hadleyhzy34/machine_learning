"""
LSTM cell from scratch (fused).

Equations (PyTorch convention, gates ordered i, f, g, o):
    i_t = σ(W_ii x + b_ii + W_hi h + b_hi)        # input gate
    f_t = σ(W_if x + b_if + W_hf h + b_hf)        # forget gate
    g_t = tanh(W_ig x + b_ig + W_hg h + b_hg)     # cell candidate
    o_t = σ(W_io x + b_io + W_ho h + b_ho)        # output gate
    c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
    h_t = o_t ⊙ tanh(c_t)

Two gotchas relative to GRU:
- LSTM has TWO state tensors (h, c). The cell's forward signature is
  (x, (h, c)) -> (h, c), and the wrapper treats state opaquely via
  init_state / state_to_h.
- Gates stack in PyTorch as [i | f | g | o] in that exact order — getting
  the order wrong is the #1 source of "passes forward, fails backward" bugs.

Verified against torch.nn.LSTMCell in gru/test_lstm_cell.py.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


LSTMState = Tuple[torch.Tensor, torch.Tensor]  # (h, c)


class LSTMCellFused(nn.Module):
    """Fused LSTM cell — single (4*hidden) matmul each for input and hidden."""

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        # Stacked: rows [0:H]=i, [H:2H]=f, [2H:3H]=g, [3H:4H]=o.
        self.weight_ih = nn.Parameter(torch.empty(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(4 * hidden_size, hidden_size))
        if bias:
            self.bias_ih = nn.Parameter(torch.empty(4 * hidden_size))
            self.bias_hh = nn.Parameter(torch.empty(4 * hidden_size))
        else:
            self.register_parameter("bias_ih", None)
            self.register_parameter("bias_hh", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Match PyTorch's LSTMCell init.
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for p in self.parameters():
            nn.init.uniform_(p, -stdv, stdv)

    def forward(
        self, x: torch.Tensor, state: Optional[LSTMState] = None
    ) -> LSTMState:
        """
        x: (batch, input_size)
        state: (h, c), each (batch, hidden_size). Defaults to zeros.
        returns: (h_next, c_next)
        """
        if state is None:
            h = x.new_zeros(x.size(0), self.hidden_size)
            c = x.new_zeros(x.size(0), self.hidden_size)
        else:
            h, c = state

        gi = F.linear(x, self.weight_ih, self.bias_ih)  # (B, 4H)
        gh = F.linear(h, self.weight_hh, self.bias_hh)  # (B, 4H)
        gates = gi + gh
        i_g, f_g, g_g, o_g = gates.chunk(4, dim=-1)

        i = torch.sigmoid(i_g)
        f = torch.sigmoid(f_g)
        g = torch.tanh(g_g)
        o = torch.sigmoid(o_g)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    # -- state convention used by RecurrentSeq -------------------------------
    def init_state(self, batch_size: int, device, dtype) -> LSTMState:
        h = torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
        c = torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
        return h, c

    @staticmethod
    def state_to_h(state: LSTMState) -> torch.Tensor:
        return state[0]

    @torch.no_grad()
    def copy_from_torch_cell(self, ref: nn.LSTMCell) -> None:
        """PyTorch stores LSTMCell params with the SAME [i, f, g, o] layout."""
        assert ref.input_size == self.input_size
        assert ref.hidden_size == self.hidden_size
        self.weight_ih.copy_(ref.weight_ih)
        self.weight_hh.copy_(ref.weight_hh)
        if self.bias and ref.bias:
            self.bias_ih.copy_(ref.bias_ih)
            self.bias_hh.copy_(ref.bias_hh)
