"""
Numerical-equivalence tests for the from-scratch LSTM cell.

Run:
    .venv/bin/python -m gru.test_lstm_cell

Same idea as test_gru_cell.py — match torch.nn.LSTMCell to ~machine epsilon
in float64, including gradients.
"""

import torch
import torch.nn as nn

from gru.cells.lstm_cell import LSTMCellFused


def _seed(s: int = 0) -> None:
    torch.manual_seed(s)


def _make_inputs(batch: int, input_size: int, hidden_size: int, dtype=torch.float64):
    x = torch.randn(batch, input_size, dtype=dtype)
    h = torch.randn(batch, hidden_size, dtype=dtype)
    c = torch.randn(batch, hidden_size, dtype=dtype)
    return x, h, c


def test_fused_matches_torch():
    _seed(0)
    input_size, hidden_size, batch = 7, 11, 3
    ref = nn.LSTMCell(input_size, hidden_size).double()
    fused = LSTMCellFused(input_size, hidden_size).double()
    fused.copy_from_torch_cell(ref)

    x, h, c = _make_inputs(batch, input_size, hidden_size)
    h_ref, c_ref = ref(x, (h, c))
    h_my, c_my = fused(x, (h, c))

    err_h = (h_ref - h_my).abs().max().item()
    err_c = (c_ref - c_my).abs().max().item()
    assert err_h < 1e-10 and err_c < 1e-10, f"err: h={err_h}, c={err_c}"
    print(f"[ok] fused vs torch.nn.LSTMCell    max abs err h={err_h:.2e}, c={err_c:.2e}")


def test_gradients_match():
    _seed(1)
    input_size, hidden_size, batch = 5, 8, 4
    ref = nn.LSTMCell(input_size, hidden_size).double()
    fused = LSTMCellFused(input_size, hidden_size).double()
    fused.copy_from_torch_cell(ref)

    x, h, c = _make_inputs(batch, input_size, hidden_size)
    x_a = x.clone().requires_grad_(True)
    h_a = h.clone().requires_grad_(True)
    c_a = c.clone().requires_grad_(True)
    x_b = x.clone().requires_grad_(True)
    h_b = h.clone().requires_grad_(True)
    c_b = c.clone().requires_grad_(True)

    # Sum h^2 + c^2 so gradients flow through both outputs.
    h_ref, c_ref = ref(x_a, (h_a, c_a))
    (h_ref.pow(2).sum() + c_ref.pow(2).sum()).backward()
    h_my, c_my = fused(x_b, (h_b, c_b))
    (h_my.pow(2).sum() + c_my.pow(2).sum()).backward()

    err_x = (x_a.grad - x_b.grad).abs().max().item()
    err_h = (h_a.grad - h_b.grad).abs().max().item()
    err_c = (c_a.grad - c_b.grad).abs().max().item()
    assert err_x < 1e-10 and err_h < 1e-10 and err_c < 1e-10, (
        f"grad err: x={err_x}, h={err_h}, c={err_c}"
    )
    print(
        f"[ok] grads vs torch.nn.LSTMCell    "
        f"max abs err x={err_x:.2e}, h={err_h:.2e}, c={err_c:.2e}"
    )


def test_zero_initial_state():
    """state=None should match passing zero h and zero c."""
    _seed(2)
    fused = LSTMCellFused(4, 6).double()
    x = torch.randn(2, 4, dtype=torch.float64)
    z = torch.zeros(2, 6, dtype=torch.float64)
    h_none, c_none = fused(x, None)
    h_zero, c_zero = fused(x, (z, z))
    assert (h_none - h_zero).abs().max().item() == 0.0
    assert (c_none - c_zero).abs().max().item() == 0.0
    print(f"[ok] state=None == zeros           max abs err = 0.00e+00")


def test_state_shape_helpers():
    """init_state and state_to_h should produce the right shapes."""
    fused = LSTMCellFused(3, 5)
    state = fused.init_state(batch_size=2, device="cpu", dtype=torch.float32)
    assert isinstance(state, tuple) and len(state) == 2
    h, c = state
    assert h.shape == (2, 5) and c.shape == (2, 5)
    h_only = fused.state_to_h(state)
    assert h_only.shape == (2, 5)
    print(f"[ok] init_state / state_to_h shapes OK")


if __name__ == "__main__":
    test_fused_matches_torch()
    test_gradients_match()
    test_zero_initial_state()
    test_state_shape_helpers()
    print("\nAll LSTM equivalence checks passed.")
