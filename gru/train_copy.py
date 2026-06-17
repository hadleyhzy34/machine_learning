"""
Copy-memory comparison: vanilla RNN vs scratch GRU vs scratch LSTM.

Trains all three on the copy task and reports per-position cross-entropy
plus accuracy on the K answer positions. The interesting plot is the
loss curve: vanilla RNN should sit at the constant-blank baseline almost
forever; gated models break through.

Run:
    .venv/bin/python -m gru.train_copy                       # K=10, T=50
    .venv/bin/python -m gru.train_copy --K 10 --T 100 --steps 6000
"""

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from gru.cells import GRUCellFused, LSTMCellFused, RNNCellScratch
from gru.models import RecurrentSeq
from gru.tasks import (
    VOCAB_SIZE,
    CopyConfig,
    make_copy_batch,
    naive_baseline_loss,
)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(kind: str, hidden_size: int) -> RecurrentSeq:
    """Sequence-to-sequence model: input one-hot, output logits per step."""
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
        input_size=VOCAB_SIZE,
        hidden_size=hidden_size,
        out_size=VOCAB_SIZE,
        readout="all",
    )


def answer_accuracy(logits: torch.Tensor, y: torch.Tensor, K: int) -> float:
    """Fraction of correct symbols at the last K positions, averaged over batch."""
    pred = logits[:, -K:].argmax(dim=-1)        # (B, K)
    target = y[:, -K:]
    return (pred == target).float().mean().item()


def train_one(
    kind: str,
    cfg: CopyConfig,
    hidden_size: int,
    steps: int,
    batch_size: int,
    lr: float,
    eval_every: int,
    device: torch.device,
    seed: int,
) -> dict:
    torch.manual_seed(seed)
    model = build_model(kind, hidden_size).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    data_gen = torch.Generator(device=device).manual_seed(1234)

    losses, accs, eval_steps = [], [], []
    t0 = time.time()
    for step in range(1, steps + 1):
        x, y = make_copy_batch(cfg, batch_size, device=device, generator=data_gen)
        logits = model(x)                                     # (B, L, V)
        # Cross-entropy averaged over all positions — same convention as
        # the constant-blank baseline.
        loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        if step % eval_every == 0 or step == 1:
            with torch.no_grad():
                acc = answer_accuracy(logits, y, cfg.K)
            losses.append(loss.item())
            accs.append(acc)
            eval_steps.append(step)
            print(
                f"  [{kind}] step {step:5d}  "
                f"loss={loss.item():.4f}  ans_acc={acc*100:5.1f}%"
            )

    return {
        "kind": kind,
        "params": model.num_params(),
        "elapsed_s": time.time() - t0,
        "steps": eval_steps,
        "losses": losses,
        "accs": accs,
        "final_loss": losses[-1],
        "final_acc": accs[-1],
    }


def plot_curves(results: list[dict], cfg: CopyConfig, out_path: str) -> None:
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    baseline = naive_baseline_loss(cfg)

    for r in results:
        ax1.plot(r["steps"], r["losses"], label=f"{r['kind'].upper()} ({r['params']:,})")
        ax2.plot(r["steps"], [a * 100 for a in r["accs"]], label=r["kind"].upper())

    ax1.axhline(baseline, color="gray", linestyle="--",
                label=f"constant-blank baseline ({baseline:.3f})")
    ax1.set_xlabel("training step")
    ax1.set_ylabel("cross-entropy / position (nats)")
    ax1.set_title(f"Copy memory loss, K={cfg.K}, T={cfg.T}")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.axhline(100.0 / 8.0, color="gray", linestyle="--",
                label="random-guess (12.5%)")
    ax2.set_xlabel("training step")
    ax2.set_ylabel("accuracy on answer positions (%)")
    ax2.set_ylim(0, 105)
    ax2.set_title("Answer accuracy")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=120)
    print(f"\nplot saved to {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=10, help="symbols to memorize")
    ap.add_argument("--T", type=int, default=50, help="delay length")
    ap.add_argument("--hidden_size", type=int, default=64)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--models", nargs="+", default=["rnn", "gru", "lstm"])
    args = ap.parse_args()

    cfg = CopyConfig(K=args.K, T=args.T)
    device = get_device()
    print(f"device: {device}")
    print(
        f"task: copy memory K={cfg.K}, T={cfg.T} "
        f"(seq_len={cfg.total_len}), hidden={args.hidden_size}\n"
    )
    print(f"naive constant-blank baseline loss: {naive_baseline_loss(cfg):.4f}\n")

    results = []
    for kind in args.models:
        print(f"--- training {kind.upper()} ---")
        r = train_one(
            kind=kind,
            cfg=cfg,
            hidden_size=args.hidden_size,
            steps=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
            eval_every=args.eval_every,
            device=device,
            seed=args.seed,
        )
        print(
            f"  done in {r['elapsed_s']:.1f}s, params={r['params']:,}, "
            f"final loss={r['final_loss']:.4f}, ans_acc={r['final_acc']*100:.1f}%\n"
        )
        results.append(r)

    print("\n=== summary ===")
    print(f"{'model':<6} {'params':>8} {'final loss':>11} {'ans_acc':>8} {'time(s)':>9}")
    for r in results:
        print(
            f"{r['kind']:<6} {r['params']:>8,} {r['final_loss']:>11.4f} "
            f"{r['final_acc']*100:>7.1f}% {r['elapsed_s']:>9.1f}"
        )

    out = os.path.join(
        os.path.dirname(__file__), "plots", f"copy_K{cfg.K}_T{cfg.T}.png"
    )
    plot_curves(results, cfg, out)


if __name__ == "__main__":
    main()
