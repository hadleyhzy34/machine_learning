"""
Numerical-equivalence tests for the from-scratch GRU cells.

Run:
    python -m gru.test_gru_cell

Passing here means your equations and the bias/gate ordering match PyTorch's
reference implementation exactly. Do this BEFORE moving on to the sequence
wrapper or training — debugging gradient issues with a wrong cell is misery.
"""

import torch
import torch.nn as nn

from gru.cells.gru_cell import GRUCellFused, GRUCellNaive


def _seed(seed: int = 0) -> None:
    torch.manual_seed(seed)


def _make_inputs(batch: int, input_size: int, hidden_size: int, dtype=torch.float64):
    """Use float64 — keeps round-off below the 1e-6 tolerance comfortably."""
    x = torch.randn(batch, input_size, dtype=dtype)
    h = torch.randn(batch, hidden_size, dtype=dtype)
    return x, h


def _copy_torch_into_naive(naive: GRUCellNaive, ref: nn.GRUCell) -> None:
    """
    PyTorch stores stacked params as [r | z | n] rows. Slice them back into
    the per-gate matrices the naive cell uses.
    """
    H = ref.hidden_size
    with torch.no_grad():
        Wih = ref.weight_ih  # (3H, in)
        Whh = ref.weight_hh  # (3H, H)
        naive.W_ir.copy_(Wih[0:H])
        naive.W_iz.copy_(Wih[H : 2 * H])
        naive.W_in.copy_(Wih[2 * H : 3 * H])
        naive.W_hr.copy_(Whh[0:H])
        naive.W_hz.copy_(Whh[H : 2 * H])
        naive.W_hn.copy_(Whh[2 * H : 3 * H])
        if ref.bias:
            bih = ref.bias_ih
            bhh = ref.bias_hh
            naive.b_ir.copy_(bih[0:H])
            naive.b_iz.copy_(bih[H : 2 * H])
            naive.b_in.copy_(bih[2 * H : 3 * H])
            naive.b_hr.copy_(bhh[0:H])
            naive.b_hz.copy_(bhh[H : 2 * H])
            naive.b_hn.copy_(bhh[2 * H : 3 * H])


def test_naive_matches_torch():
    _seed(0)
    input_size, hidden_size, batch = 7, 11, 3
    ref = nn.GRUCell(input_size, hidden_size).double()
    naive = GRUCellNaive(input_size, hidden_size).double()
    _copy_torch_into_naive(naive, ref)

    x, h = _make_inputs(batch, input_size, hidden_size)
    out_ref = ref(x, h)
    out_naive = naive(x, h)

    err = (out_ref - out_naive).abs().max().item()
    assert err < 1e-10, f"naive vs torch max abs err = {err}"
    print(f"[ok] naive  vs torch.nn.GRUCell    max abs err = {err:.2e}")


def test_fused_matches_torch():
    _seed(1)
    input_size, hidden_size, batch = 7, 11, 3
    ref = nn.GRUCell(input_size, hidden_size).double()
    fused = GRUCellFused(input_size, hidden_size).double()
    fused.copy_from_torch_cell(ref)

    x, h = _make_inputs(batch, input_size, hidden_size)
    out_ref = ref(x, h)
    out_fused = fused(x, h)

    err = (out_ref - out_fused).abs().max().item()
    assert err < 1e-10, f"fused vs torch max abs err = {err}"
    print(f"[ok] fused  vs torch.nn.GRUCell    max abs err = {err:.2e}")


def test_naive_matches_fused():
    """Sanity: the two scratch impls must agree with each other too."""
    _seed(2)
    input_size, hidden_size, batch = 7, 11, 3
    ref = nn.GRUCell(input_size, hidden_size).double()
    naive = GRUCellNaive(input_size, hidden_size).double()
    fused = GRUCellFused(input_size, hidden_size).double()
    _copy_torch_into_naive(naive, ref)
    fused.copy_from_torch_cell(ref)

    x, h = _make_inputs(batch, input_size, hidden_size)
    err = (naive(x, h) - fused(x, h)).abs().max().item()
    assert err < 1e-10, f"naive vs fused max abs err = {err}"
    print(f"[ok] naive  vs fused               max abs err = {err:.2e}")


def test_gradients_match():
    """
    A correct forward isn't enough — the gradients must match too.
    A wrong gate placement can pass forward by luck on random init but
    will diverge in backward.
    """
    _seed(3)
    input_size, hidden_size, batch = 5, 8, 4
    ref = nn.GRUCell(input_size, hidden_size).double()
    fused = GRUCellFused(input_size, hidden_size).double()
    fused.copy_from_torch_cell(ref)

    x, h = _make_inputs(batch, input_size, hidden_size)
    x_a = x.clone().requires_grad_(True)
    h_a = h.clone().requires_grad_(True)
    x_b = x.clone().requires_grad_(True)
    h_b = h.clone().requires_grad_(True)

    ref(x_a, h_a).pow(2).sum().backward()
    fused(x_b, h_b).pow(2).sum().backward()

    err_x = (x_a.grad - x_b.grad).abs().max().item()
    err_h = (h_a.grad - h_b.grad).abs().max().item()
    assert err_x < 1e-10 and err_h < 1e-10, f"grad err: x={err_x}, h={err_h}"
    print(f"[ok] grads vs torch.nn.GRUCell     max abs err x={err_x:.2e}, h={err_h:.2e}")


def test_zero_initial_hidden():
    """h=None should be the same as passing zeros."""
    _seed(4)
    fused = GRUCellFused(4, 6).double()
    x = torch.randn(2, 4, dtype=torch.float64)
    out_none = fused(x, None)
    out_zero = fused(x, torch.zeros(2, 6, dtype=torch.float64))
    err = (out_none - out_zero).abs().max().item()
    assert err == 0.0, f"h=None vs zeros differ: {err}"
    print(f"[ok] h=None == zeros               max abs err = {err:.2e}")


if __name__ == "__main__":
    test_naive_matches_torch()
    test_fused_matches_torch()
    test_naive_matches_fused()
    test_gradients_match()
    test_zero_initial_hidden()
    print("\nAll equivalence checks passed.")
