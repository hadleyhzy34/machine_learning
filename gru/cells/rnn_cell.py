"""
Vanilla RNN cell from scratch.

Equation (PyTorch convention, two-bias):
    h_t = tanh(W_ih x_t + b_ih + W_hh h_{t-1} + b_hh)

Verified against torch.nn.RNNCell with nonlinearity='tanh' in
gru/test_rnn_cell.py.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNCellScratch(nn.Module):
    """
    Single-step Elman RNN cell. Useful as a baseline — same wrapper, same
    training loop, but no gating. On the adding/copy tasks it should fail
    to learn at long sequence lengths, which is the whole point.
    """

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.weight_ih = nn.Parameter(torch.empty(hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(hidden_size, hidden_size))
        if bias:
            self.bias_ih = nn.Parameter(torch.empty(hidden_size))
            self.bias_hh = nn.Parameter(torch.empty(hidden_size))
        else:
            self.register_parameter("bias_ih", None)
            self.register_parameter("bias_hh", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Match PyTorch's RNNCell init.
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
        return torch.tanh(
            F.linear(x, self.weight_ih, self.bias_ih)
            + F.linear(h, self.weight_hh, self.bias_hh)
        )

    # state convention: for vanilla RNN, state == h.
    def init_state(self, batch_size: int, device, dtype) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)

    @staticmethod
    def state_to_h(state: torch.Tensor) -> torch.Tensor:
        return state

    @torch.no_grad()
    def copy_from_torch_cell(self, ref: nn.RNNCell) -> None:
        assert ref.input_size == self.input_size
        assert ref.hidden_size == self.hidden_size
        self.weight_ih.copy_(ref.weight_ih)
        self.weight_hh.copy_(ref.weight_hh)
        if self.bias and ref.bias:
            self.bias_ih.copy_(ref.bias_ih)
            self.bias_hh.copy_(ref.bias_hh)
