# GRU From-Scratch Benchmark

A from-scratch implementation of the GRU recurrent cell, plus a small benchmark
harness comparing it against vanilla RNN and LSTM (also from scratch) on two
classic synthetic memory tasks. The point is hands-on: every piece of math is
written out, every cell is verified against PyTorch's reference implementation
to machine epsilon, and the comparisons are designed so the *why* of gating
shows up visually in the plots.

---

## Table of contents

1. [Quick start](#quick-start)
2. [Repository layout](#repository-layout)
3. [What's implemented](#whats-implemented)
4. [Benchmarks](#benchmarks)
5. [How the pieces fit together](#how-the-pieces-fit-together)
6. [Equivalence tests — what passing means](#equivalence-tests--what-passing-means)
7. [Equations reference](#equations-reference)
8. [Roadmap](#roadmap)

---

## Quick start

```bash
cd /Users/hadley/Developments/dl_basics

# 1. Verify scratch cells match PyTorch references (machine-epsilon equivalence).
.venv/bin/python -m gru.test_gru_cell
.venv/bin/python -m gru.test_lstm_cell

# 2. Adding problem (regression) — vanilla RNN plateaus at long T, gated wins.
.venv/bin/python -m gru.train_adding --seq_len 100 --steps 2000
.venv/bin/python -m gru.train_adding --seq_len 200 --steps 4000

# 3. Copy memory (sequence-to-sequence classification).
.venv/bin/python -m gru.train_copy --K 10 --T 50  --steps 4000
.venv/bin/python -m gru.train_copy --K 10 --T 100 --steps 8000
```

Plots are written to `gru/plots/`. Console output includes per-step loss, a
final summary table (params, final metric, wall-clock seconds), and the path
of the saved plot.

---

## Repository layout

```
gru/
├── README.md                  ← this file
├── __init__.py
│
├── cells/                     ← single-step cell implementations
│   ├── __init__.py
│   ├── rnn_cell.py            ← RNNCellScratch (vanilla Elman)
│   ├── gru_cell.py            ← GRUCellNaive + GRUCellFused
│   └── lstm_cell.py           ← LSTMCellFused (tuple state)
│
├── models/                    ← composed sequence models
│   ├── __init__.py
│   └── recurrent.py           ← RecurrentSeq — wraps any cell over (B, T, D)
│
├── tasks/                     ← synthetic data generators
│   ├── __init__.py
│   ├── adding.py              ← make_adding_batch + NAIVE_BASELINE_MSE
│   └── copy_memory.py         ← CopyConfig, make_copy_batch, naive_baseline_loss
│
├── train_adding.py            ← driver: RNN/GRU/LSTM on adding problem
├── train_copy.py              ← driver: RNN/GRU/LSTM on copy memory
├── test_gru_cell.py           ← 5 equivalence tests vs torch.nn.GRUCell
├── test_lstm_cell.py          ← 4 equivalence tests vs torch.nn.LSTMCell
│
└── plots/                     ← generated training curves
```

---

## What's implemented

### Cells (`gru/cells/`)

All cells follow PyTorch's convention exactly:

- **Two bias vectors** (`b_ih` and `b_hh`), not one. Many textbooks fold these
  together; PyTorch keeps them separate, and so do we.
- **Initialization** is `Uniform(-1/√H, +1/√H)` for every parameter (matches
  `nn.RNNCell.reset_parameters()`).
- Each cell exposes a `copy_from_torch_cell(ref)` helper that copies parameters
  from the corresponding `torch.nn.*Cell`, used by the equivalence tests.
- Each cell exposes a small **state convention** (see [How the pieces fit
  together](#how-the-pieces-fit-together)) so the same wrapper can drive both
  single-tensor states (RNN/GRU) and tuple states (LSTM).

#### `RNNCellScratch`

Standard Elman cell:

```
h_t = tanh(W_ih x_t + b_ih + W_hh h_{t-1} + b_hh)
```

Used as the "no gating" baseline. Expected to fail on long-range tasks because
`tanh`'s derivative is bounded above by 1 and gradients shrink across time.

#### `GRUCellNaive`

The textbook implementation: six independent weight matrices, one per gate.
Easier to read and to ablate ("kill the reset gate" is a 2-line edit).

#### `GRUCellFused`

The performance-oriented version: a single `(3·hidden, input)` matmul produces
all three input gate pre-activations at once, chunked into r/z/n. Same for the
hidden side. This is what cuDNN does internally. Numerically equivalent to the
naive version up to floating-point reorderings.

```
gi = W_ih @ x + b_ih           # (B, 3H)
gh = W_hh @ h + b_hh           # (B, 3H)
i_r, i_z, i_n = chunk(gi, 3)
h_r, h_z, h_n = chunk(gh, 3)
r = σ(i_r + h_r)
z = σ(i_z + h_z)
n = tanh(i_n + r * h_n)        # ← reset gate applied AFTER W_hn h + b_hn
h' = (1-z) * n + z * h
```

The single most common bug in scratch GRU implementations: applying `r` to
`h` directly (`tanh(W_in x + W_hn (r*h))`) instead of to `W_hn h + b_hn`.
That's a different model; it passes forward by luck on random init but
diverges in backward.

#### `LSTMCellFused`

```
gi = W_ih @ x + b_ih           # (B, 4H)
gh = W_hh @ h + b_hh           # (B, 4H)
i, f, g, o = chunk(gi + gh, 4)
i, f, o = σ(i), σ(f), σ(o)
g = tanh(g)
c' = f * c + i * g
h' = o * tanh(c')
```

State is the tuple `(h, c)`. Gate order in the stacked weight matrix is
`[i, f, g, o]` — same as PyTorch.

### Sequence wrapper (`gru/models/recurrent.py`)

`RecurrentSeq` wraps any cell into a model over a `(B, T, D)` input.
Constructor takes a `cell_factory` callable, `input_size`, `hidden_size`,
`out_size`, and a `readout` mode:

| Readout | Output shape | When to use |
|---|---|---|
| `'last'` | `(B, out_size)` | Final hidden state → head. Adding problem, sentiment. |
| `'mean'` | `(B, out_size)` | Time-averaged hidden → head. Bag-of-tokens style. |
| `'all'`  | `(B, T, out_size)` | Apply head at every timestep. Copy memory, char-LM. |

The wrapper holds **one** linear head (`nn.Linear(hidden_size, out_size)`)
and applies it batched after the time loop in the `'all'` case.

### Tasks (`gru/tasks/`)

#### Adding problem (`adding.py`)

Each sample is a length-T sequence with two channels:
- channel 0: random scalars in [0, 1]
- channel 1: a 0/1 mask with exactly two 1s at random positions

Target: the sum of the two scalars at the masked positions.

```python
x, y = make_adding_batch(batch_size=64, seq_len=200, device="mps")
# x: (64, 200, 2)   y: (64, 1)
```

The mask is built vectorized (no Python loop): `argsort` over uniform noise
gives two distinct random positions per row, then `scatter_` writes the 1s.

**Trivial baseline.** A model that always predicts 1.0 (the mean of two
i.i.d. Uniform(0,1) variables) gets MSE = Var(U+U') = 1/6 ≈ 0.1667. Beating
this means the model has learned to attend to the masked positions.

#### Copy memory (`copy_memory.py`)

Vocabulary of 10 symbols:
- `1..8` — data symbols
- `0` — blank
- `9` — go cue

Each sample has length `T + 2K`:

```
positions   role                          fill
─────────   ───────────────────────       ──────────────
0..K-1      message (random data)         random in 1..8
K..K+T-2    delay                         blank (0)
K+T-1       go cue                        9
K+T..L-1    answer slot                   blank (0) — model writes here
```

The target is blanks everywhere except the last K positions, which contain
the original message. The model must hold K symbols across `T-1` blank steps,
then write them out in order when prompted by the go cue.

**Trivial baseline.** A model that emits "always blank, uniform-over-data at
the answer slot" achieves average per-position cross-entropy
`(K/L) · log 8`. Below this = recall has started.

Both batch generators take a `torch.Generator` for reproducibility and run
fully on-device — no host↔device copies during training.

### Training drivers (`train_adding.py`, `train_copy.py`)

Each driver:
1. Picks a device (CUDA → MPS → CPU).
2. Builds one `RecurrentSeq` per requested model kind.
3. Trains each on the same data stream (shared seeded `Generator`) so
   comparisons are not luck-of-the-draw.
4. Uses Adam, fixed lr=1e-3, gradient clipping at 1.0 (vanilla RNN
   explodes without it).
5. Logs per-step loss every `--eval_every` steps.
6. Prints a summary table and saves a comparison plot.

Both drivers share the same model factory pattern and the same
`get_device()` helper. CLI flags are nearly identical, only the
task-specific ones differ (`--seq_len` vs `--K`/`--T`).

---

## Benchmarks

We currently use **two synthetic memory probes**. Real-world tasks
(char-LM, sentiment) are roadmap items.

### Why these two

These are the canonical probes from the Hochreiter/Schmidhuber LSTM paper
and follow-up work (Arjovsky et al. 2016 for the modern copy-memory
setup). They were designed specifically to distinguish gated from
ungated recurrent models. They have three properties that make them ideal
for hands-on benchmarking:

1. **No data to download, infinite samples.** Generation is on-device,
   so a single training run takes seconds-to-minutes.
2. **Known trivial baselines.** Closed-form expression for the loss of
   "the model that learned nothing" — so the plot has a meaningful
   reference line, not just a vague "lower is better".
3. **Sharp transition.** Beyond a moderate sequence length, vanilla RNN
   does *not* learn the task at all in a reasonable budget, while gated
   models do. The split is dramatic on a log plot — not a 5% accuracy
   gap, more like a decade of MSE.

### What we measure

Per `(model, task)`:

| Quantity | How |
|---|---|
| Final task metric | MSE for adding; cross-entropy + answer-position accuracy for copy memory |
| Parameter count | `sum(p.numel() for p in model.parameters() if p.requires_grad)` |
| Wall-clock training time | `time.time()` around the training loop |
| Loss curve | Saved every `--eval_every` steps and plotted on log scale |
| Comparison-vs-baseline | Reference line drawn at the trivial baseline value |

What we **don't** measure yet (roadmap):
- GPU memory peak
- Inference throughput (tokens/sec at batch=1 vs batch=64)
- Gradient norm over training (the vanishing-gradient story made visible)

### Difficulty knobs

| Task | Knob | Effect |
|---|---|---|
| Adding | `--seq_len` | Longer = more steps to backprop through. RNN starts to fail around T~50–100, gated models hold to T~500+. |
| Copy   | `--K`      | More symbols to hold simultaneously. |
| Copy   | `--T`      | Longer silent delay between message and go cue. |
| Both   | `--hidden_size` | Capacity. |
| Both   | `--steps`, `--lr`, `--batch_size` | Optimization. |

### Running an experiment

The pattern is:

```bash
.venv/bin/python -m gru.train_<task> [task-specific flags] [optimizer flags] \
                                    --models rnn gru lstm
```

`--models` defaults to `rnn gru lstm`. Pass a subset to skip models. Each
model is trained from scratch under its own seed; data is shared across
models so loss differences reflect the architecture, not different
batches.

---

## How the pieces fit together

The whole point of the design is that **the same `RecurrentSeq` wrapper
drives every cell and every task**. Adding a new cell or a new task
requires touching only one file.

### The state convention

LSTM's state is a tuple `(h, c)`; RNN's and GRU's is just `h`. To keep the
wrapper agnostic, every cell exposes:

```python
class Cell(nn.Module):
    def forward(self, x_t, state) -> state'         # x_t: (B, in), state: opaque
    def init_state(self, B, device, dtype) -> state # build the t=0 state
    @staticmethod
    def state_to_h(state) -> Tensor                 # extract (B, H) for the head
```

For RNN/GRU, `state == h` and `state_to_h(s) == s`. For LSTM, `state == (h, c)`
and `state_to_h((h, c)) == h`. The wrapper never inspects state directly —
it just threads it through the loop and asks the cell for `h` when it needs
to read out.

### The wrapper's loop

```python
state = self.cell.init_state(B, x.device, x.dtype)
for t in range(T):
    state = self.cell(x[:, t], state)
    # readout='last': nothing — just keep going
    # readout='mean': accumulate state_to_h(state)
    # readout='all':  collect state_to_h(state) into a list
return self.head(<readout result>)
```

Plain Python loop, no `torch.jit.script` or anything fancy. On a 200-step
sequence with hidden=64 and batch=64, each step takes a few microseconds;
the overhead is well below the matmul cost.

### Adding a new cell

1. Implement `forward(x_t, state) -> state`, `init_state`, `state_to_h`.
2. Match PyTorch's parameter layout if you want to verify equivalence.
3. Add it to `cells/__init__.py`.
4. Add a branch to the `build_model` function in each driver.

### Adding a new task

1. Write `make_<task>_batch(...) -> (x, y)` in `tasks/`.
2. If the task needs per-step output, use `readout='all'` in the driver.
3. Either reuse `train_adding.py` / `train_copy.py` as templates, or
   factor out the loop if the patterns start to converge.

---

## Equivalence tests — what passing means

The tests build a `torch.nn.GRUCell` (or `LSTMCell`), copy its parameters
into our scratch cell, run both on the same input in float64, and compare:
forward outputs, gradients with respect to inputs and to hidden states,
and the `state=None` semantics.

Current results:

| Test | Max abs error |
|---|---|
| `naive_GRU` vs `torch.nn.GRUCell`, forward | 2.22e-16 |
| `fused_GRU` vs `torch.nn.GRUCell`, forward | 2.22e-16 |
| `naive_GRU` vs `fused_GRU`, forward | 2.22e-16 |
| `fused_GRU` vs `torch.nn.GRUCell`, gradients (x, h) | 1.11e-16, 2.78e-16 |
| `fused_LSTM` vs `torch.nn.LSTMCell`, forward (h, c) | **0.0**, **0.0** |
| `fused_LSTM` vs `torch.nn.LSTMCell`, gradients (x, h, c) | **0.0**, **0.0**, **0.0** |

LSTM hits exactly zero — gate ordering `[i, f, g, o]` and the single
fused matmul `gi + gh` produce the same float operations in the same
order as PyTorch's reference. GRU has 1-ulp differences because the
candidate gate `n = tanh(i_n + r * h_n)` involves an extra elementwise
multiply that gets reordered slightly relative to PyTorch's internal
implementation.

This level of equivalence is stronger than just "the loss curve looks
similar" — it means **every weight, every bias, every operation is in
the right place in the right order**. If you're going to copy this
scaffold for another architecture, the same pattern (initialize, copy
params from reference, compare in float64 with gradients) is worth
running first.

---

## Equations reference

### Vanilla RNN (Elman)

```
h_t = tanh(W_ih x_t + b_ih + W_hh h_{t-1} + b_hh)
```

Parameters: 2 matrices + 2 biases. Total: `H·in + H·H + 2H`.

### GRU (PyTorch convention)

```
r_t = σ(W_ir x_t + b_ir + W_hr h_{t-1} + b_hr)              reset gate
z_t = σ(W_iz x_t + b_iz + W_hz h_{t-1} + b_hz)              update gate
n_t = tanh(W_in x_t + b_in + r_t ⊙ (W_hn h_{t-1} + b_hn))   candidate
h_t = (1 - z_t) ⊙ n_t + z_t ⊙ h_{t-1}                       new hidden
```

Parameters: 6 matrices + 6 biases (or in fused form, 2 matrices of
shape `(3H, ·)` + 2 biases of shape `3H`). Total: `3·H·in + 3·H·H + 6H`.

### LSTM (PyTorch convention)

```
i_t = σ(W_ii x + b_ii + W_hi h + b_hi)
f_t = σ(W_if x + b_if + W_hf h + b_hf)
g_t = tanh(W_ig x + b_ig + W_hg h + b_hg)
o_t = σ(W_io x + b_io + W_ho h + b_ho)
c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
h_t = o_t ⊙ tanh(c_t)
```

Parameters: in fused form, 2 matrices of shape `(4H, ·)` + 2 biases
of shape `4H`. Total: `4·H·in + 4·H·H + 8H`.

For a fair benchmark across the three, use roughly equal parameter
counts, **not** equal hidden sizes. With `hidden_size = H`:

- RNN  ≈ `H·(in + H + 2)`
- GRU  ≈ `3·H·(in + H + 2)` — about 3× the RNN
- LSTM ≈ `4·H·(in + H + 2)` — about 4× the RNN

The current drivers don't yet auto-balance for this; they just use the
same `--hidden_size` for all three and report params in the summary
table. Auto-balancing is on the roadmap.

---

## Roadmap

Done:
- [x] Naive + fused GRU cell, matching PyTorch
- [x] Vanilla RNN cell
- [x] Fused LSTM cell with tuple state
- [x] State-agnostic sequence wrapper with three readouts
- [x] Adding problem + driver
- [x] Copy memory + driver
- [x] Equivalence tests (machine-epsilon vs `torch.nn.{GRU,LSTM}Cell`)

Not yet:
- [ ] **CNN baseline** — 1D causal CNN with kernel size matching the
  longest dependency the task can reach
- [ ] **Tiny Transformer baseline** — 1–2 encoder layers
- [ ] **`nn.GRU` baseline** — to see how far our naive-Python time loop
  is from the cuDNN-backed reference
- [ ] **Char-level LM** on tiny-shakespeare (real text, BPC metric)
- [ ] **IMDB sentiment** (real text, accuracy metric)
- [ ] **Unified `benchmark.py`** — one driver, all models × all tasks,
  Markdown table output for paste-in
- [ ] **Memory and throughput** measurements alongside accuracy
- [ ] **Gradient-norm visualization** for the vanishing-gradient story
- [ ] **Ablations** — kill the reset gate, kill the update gate, tie
  `z = 1-r` (minimal GRU), orthogonal vs Xavier init on `W_hh`
