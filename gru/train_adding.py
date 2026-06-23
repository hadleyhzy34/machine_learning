"""
Adding-problem comparison: vanilla RNN vs scratch GRU/LSTM.

Trains scratch cells on the adding problem at a chosen sequence length and
reports a side-by-side learning curve. The point is memory/optimization, not
raw backend speed: these cells are run in a Python loop over timesteps, so they
do not use PyTorch's fused nn.GRU/nn.LSTM kernels.

Run:
    python -m gru.train_adding              # default T=100
    python -m gru.train_adding --seq_len 200 --steps 4000

Plots are written to gru/plots/.
"""

import argparse
import os
import time

import torch
import torch.nn as nn

from gru.cells import GRUCellFused, LSTMCellFused, RNNCellScratch
from gru.models import RecurrentSeq
from gru.tasks import NAIVE_BASELINE_MSE, make_adding_batch


def get_device(device_name: str = "auto") -> torch.device:
    if device_name != "auto":
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def build_model(kind: str, hidden_size: int) -> RecurrentSeq:
    """Build a regression model for the adding task (input_size=2, out=1)."""
    if kind == "rnn":
        factory = lambda i, h: RNNCellScratch(i, h)
    elif kind == "gru":
        factory = lambda i, h: GRUCellFused(i, h)
    elif kind == "lstm":
        factory = lambda i, h: LSTMCellFused(i, h)
    else:
        raise ValueError(f"unknown kind: {kind}")
    return RecurrentSeq(
        cell_factory=factory,
        input_size=2,
        hidden_size=hidden_size,
        out_size=1,
        readout="last",
    )


def train_one(
    kind: str,
    seq_len: int,
    hidden_size: int,
    steps: int,
    batch_size: int,
    lr: float,
    eval_every: int,
    device: torch.device,
    seed: int,
    eval_batches: int,
) -> dict:
    """Train one model, return its loss curve and timing info."""
    torch.manual_seed(seed)
    model = build_model(kind, hidden_size).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # Independent RNG for data so the two models see the same stream of batches.
    data_gen = torch.Generator(device=device).manual_seed(1234)

    losses, eval_steps = [], []
    sync_device(device)
    t0 = time.time()
    for step in range(1, steps + 1):
        x, y = make_adding_batch(batch_size, seq_len, device=device, generator=data_gen)
        pred = model(x)
        loss = loss_fn(pred, y)

        opt.zero_grad()
        loss.backward()
        # Clip — vanilla RNN will explode without this on longer sequences.
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        if step % eval_every == 0 or step == 1:
            losses.append(loss.item())
            eval_steps.append(step)
            print(f"  [{kind}] step {step:5d}  loss={loss.item():.4f}")

    sync_device(device)
    elapsed = time.time() - t0
    val_mse = evaluate_model(
        model, seq_len, batch_size, eval_batches, device, seed + 10_000
    )
    return {
        "kind": kind,
        "params": model.num_params(),
        "elapsed_s": elapsed,
        "steps": eval_steps,
        "losses": losses,
        "final_loss": losses[-1],
        "val_mse": val_mse,
    }


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    seq_len: int,
    batch_size: int,
    batches: int,
    device: torch.device,
    seed: int,
) -> float:
    """Average MSE over fresh batches; less noisy than the last train batch."""
    model.eval()
    loss_fn = nn.MSELoss()
    gen = torch.Generator(device=device).manual_seed(seed)
    total = 0.0
    for _ in range(batches):
        x, y = make_adding_batch(batch_size, seq_len, device=device, generator=gen)
        total += loss_fn(model(x), y).item()
    model.train()
    return total / batches


@torch.no_grad()
def estimate_baselines(
    seq_len: int,
    batch_size: int,
    batches: int,
    device: torch.device,
    seed: int,
) -> dict[str, float]:
    """Empirical no-information baselines on the same target distribution."""
    gen = torch.Generator(device=device).manual_seed(seed)
    const_total = 0.0
    random_total = 0.0
    for _ in range(batches):
        _, y = make_adding_batch(batch_size, seq_len, device=device, generator=gen)
        const_total += ((torch.ones_like(y) - y) ** 2).mean().item()
        random_guess = torch.rand(
            batch_size, 2, device=device, generator=gen
        ).sum(dim=1, keepdim=True)
        random_total += ((random_guess - y) ** 2).mean().item()
    return {
        "constant_mean_mse": const_total / batches,
        "independent_random_mse": random_total / batches,
    }


def plot_curves(results: list[dict], seq_len: int, out_path: str) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 4.5))
    for r in results:
        plt.plot(
            r["steps"],
            r["losses"],
            label=f"{r['kind'].upper()} ({r['params']:,} params)",
        )
    plt.axhline(
        NAIVE_BASELINE_MSE,
        color="gray",
        linestyle="--",
        label=f"constant-1.0 baseline ({NAIVE_BASELINE_MSE:.3f})",
    )
    plt.xlabel("training step")
    plt.ylabel("MSE")
    plt.yscale("log")
    plt.title(f"Adding problem, T={seq_len}")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=120)
    print(f"\nplot saved to {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq_len", type=int, default=100)
    ap.add_argument("--hidden_size", type=int, default=64)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--eval_batches", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--models", nargs="+", default=["rnn", "gru", "lstm"])
    ap.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    args = ap.parse_args()

    device = get_device(args.device)
    print(f"device: {device}")
    print(f"task: adding problem, T={args.seq_len}, hidden={args.hidden_size}\n")

    baselines = estimate_baselines(
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        batches=args.eval_batches,
        device=device,
        seed=args.seed + 20_000,
    )
    print(
        "baselines: "
        f"constant mean MSE={baselines['constant_mean_mse']:.4f} "
        f"(theory {NAIVE_BASELINE_MSE:.4f}), "
        f"independent random-sum MSE={baselines['independent_random_mse']:.4f}\n"
    )

    results = []
    for kind in args.models:
        print(f"--- training {kind.upper()} ---")
        r = train_one(
            kind=kind,
            seq_len=args.seq_len,
            hidden_size=args.hidden_size,
            steps=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
            eval_every=args.eval_every,
            device=device,
            seed=args.seed,
            eval_batches=args.eval_batches,
        )
        print(
            f"  done in {r['elapsed_s']:.1f}s, "
            f"params={r['params']:,}, "
            f"last train MSE={r['final_loss']:.4f}, "
            f"validation MSE={r['val_mse']:.4f}\n"
        )
        results.append(r)

    print("\n=== summary ===")
    print(f"{'model':<6} {'params':>8} {'train MSE':>11} {'val MSE':>9} {'time(s)':>9}")
    for r in results:
        print(
            f"{r['kind']:<6} {r['params']:>8,} "
            f"{r['final_loss']:>11.4f} {r['val_mse']:>9.4f} {r['elapsed_s']:>9.1f}"
        )
    print(
        f"\nA validation MSE below the constant-mean baseline "
        f"({baselines['constant_mean_mse']:.3f}) means the model learned signal."
    )
    print(
        "Timing is for scratch Python-loop cells; it is not a fused-kernel "
        "throughput benchmark."
    )

    out = os.path.join(
        os.path.dirname(__file__), "plots", f"adding_T{args.seq_len}.png"
    )
    plot_curves(results, args.seq_len, out)


if __name__ == "__main__":
    main()
