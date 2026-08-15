"""Is the missing information present in the hidden state, or never computed?

`full` (real x*x) gets stuck specifically at the high-fan-in middle places,
transitioning from "carry slipped" (edges) to "pure guessing" (middle) --
see analyze_error_magnitude.py / plot_carry_ripple.py. carry_only and
raw_sum each solve their half of the problem cleanly in isolation. The
natural next question: when the model fails at a middle place, has it
actually failed to COMPUTE the right intermediate values (the raw
convolution sum, the incoming carry), or did it compute them fine and just
fail to READ them out into the final digit? Those imply completely
different fixes.

This trains a tiny linear probe -- one nn.Linear per (place, target) -- on
the FROZEN final hidden state (post ln_f, pre output head) of a trained
`full`-variant checkpoint, predicting three things per read-out position:

    true_digit    the correct output digit (upper-reference: this is
                  literally what the model's own head is also trying to
                  predict, from the same vector)
    raw_mod10     (sum_{i+j=k} x_i*x_j) mod 10 -- the pairing/convolution
                  step's own last digit, BEFORE any carry is added
    carry_in      the true carry flowing INTO this place from lower places

A linear probe can only recover what's linearly present. If it decodes
raw_mod10/carry_in accurately at exactly the places the model's real
output is wrong, the information exists in the representation and the
model's own head just isn't using it -- a readout problem. If the probe
also fails there, the information was never computed -- a computation
problem, and readout fixes wouldn't help.

Usage:
  python experiments/probe_carry_readout.py
  python experiments/probe_carry_readout.py --checkpoint exp5_full_rope_L2_loop4_s0.pt --cohort id_5d
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.tasks import build_square, raw_place_sums  # noqa: E402
from common.tokenizer import SQUARE  # noqa: E402
from common.train import load_checkpoint, pick_device, tensors_from_records  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
CKPT_DIR = RESULTS / "ckpt"


def true_carry_chain(x: int, n_places: int) -> tuple[list[int], list[int]]:
    """raw per-place sums and the carry flowing INTO each place, LSD-first,
    extended with implicit zero-valued higher places so every place index
    used anywhere in a ragged-answer-length batch has a defined answer
    (some x in a cohort produce one extra leading digit via final carry-out;
    others don't -- both are covered by the same extension)."""
    raw = raw_place_sums(x)
    raw = raw + [0] * max(0, n_places - len(raw))
    carry_in: list[int] = []
    carry = 0
    for s in raw[:n_places]:
        carry_in.append(carry)
        carry = (s + carry) // 10
    return raw[:n_places], carry_in


def _label_hidden(hidden: torch.Tensor, target_positions: torch.Tensor, labels: torch.Tensor,
                   valid: torch.Tensor, records: list[dict]) -> dict[int, list]:
    """place -> list of (hidden_vec[d_model], true_digit, raw_mod10, carry_in), for
    ANY hidden-state snapshot (final or intermediate) of shape (n, seq, d_model)."""
    n, width = target_positions.shape
    gathered = hidden.gather(
        1, target_positions.unsqueeze(-1).expand(-1, -1, hidden.size(-1))
    ).cpu()  # (n, width, d_model)
    valid, labels = valid.cpu(), labels.cpu()

    by_place: dict[int, list] = {}
    for i, record in enumerate(records):
        x = record["answer"]
        raw, carry_in = true_carry_chain(x, width)
        for col in range(width):
            if not bool(valid[i, col]):
                continue
            place = width - 1 - col
            if place >= len(raw):
                continue
            true_digit = int(labels[i, col].item()) - SQUARE.digit_range[0]
            by_place.setdefault(place, []).append((
                gathered[i, col], true_digit, raw[place] % 10, float(carry_in[place]),
            ))
    return by_place


@torch.no_grad()
def extract_features(model, tensors: dict, records: list[dict], device) -> dict[int, list]:
    """place -> list of (hidden_vec[d_model], true_digit, raw_mod10, carry_in), from
    the FINAL hidden state (post ln_f, what the real output head actually sees)."""
    captured = {}

    def hook(module, inp, out):
        captured["h"] = out

    handle = model.ln_f.register_forward_hook(hook)
    batch = {k: v.to(device) for k, v in tensors.items()}
    model(batch["input_ids"], batch["attention_mask"])
    handle.remove()

    return _label_hidden(captured["h"], batch["target_positions"], batch["labels"],
                         batch["labels"] != -100, records)


@torch.no_grad()
def extract_features_per_loop(model, tensors: dict, records: list[dict], device) -> dict[int, dict[int, list]]:
    """depth (1..n_loops, "hidden state after this many full loop iterations") ->
    place -> samples. Hooks every unique Block; since weights are tied and the loop
    reuses the same module objects, a hook fires once per iteration -- the LAST
    block's firing in each iteration is that iteration's output (the residual
    stream BEFORE ln_f, which is only ever applied once, at the very end)."""
    n_layers = len(model.blocks)
    captured: list[torch.Tensor] = []

    def hook(module, inp, out):
        captured.append(out)

    handles = [block.register_forward_hook(hook) for block in model.blocks]
    batch = {k: v.to(device) for k, v in tensors.items()}
    model(batch["input_ids"], batch["attention_mask"])
    for h in handles:
        h.remove()

    loop_ends = captured[n_layers - 1::n_layers]  # one per completed loop iteration
    valid = batch["labels"] != -100
    return {
        depth: _label_hidden(hidden, batch["target_positions"], batch["labels"], valid, records)
        for depth, hidden in enumerate(loop_ends, start=1)
    }


def train_classifier_probe(X: torch.Tensor, y: torch.Tensor, n_classes: int,
                           steps: int = 400, lr: float = 0.05) -> nn.Linear:
    probe = nn.Linear(X.size(-1), n_classes)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = F.cross_entropy(probe(X), y)
        loss.backward()
        opt.step()
    return probe


def train_regression_probe(X: torch.Tensor, y: torch.Tensor,
                           steps: int = 400, lr: float = 0.05) -> nn.Linear:
    probe = nn.Linear(X.size(-1), 1)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    y_mean, y_std = y.mean(), y.std().clamp_min(1e-6)
    y_norm = (y - y_mean) / y_std
    for _ in range(steps):
        opt.zero_grad()
        loss = F.mse_loss(probe(X).squeeze(-1), y_norm)
        loss.backward()
        opt.step()
    probe.y_mean, probe.y_std = y_mean, y_std  # type: ignore[attr-defined]
    return probe


def probe_place(samples: list, split: float = 0.8) -> dict:
    n = len(samples)
    cut = int(n * split)
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    train_idx, eval_idx = idx[:cut], idx[cut:]

    X = torch.stack([s[0] for s in samples])
    true_digit = torch.tensor([s[1] for s in samples], dtype=torch.long)
    raw_mod10 = torch.tensor([s[2] for s in samples], dtype=torch.long)
    carry_in = torch.tensor([s[3] for s in samples], dtype=torch.float)

    def eval_classifier(probe, y):
        with torch.no_grad():
            pred = probe(X[eval_idx]).argmax(-1)
        return (pred == y[eval_idx]).float().mean().item()

    def eval_regression(probe, y):
        with torch.no_grad():
            pred = probe(X[eval_idx]).squeeze(-1) * probe.y_std + probe.y_mean
        exact = (pred.round() == y[eval_idx]).float().mean().item()
        mae = (pred - y[eval_idx]).abs().mean().item()
        return exact, mae

    p_digit = train_classifier_probe(X[train_idx], true_digit[train_idx], 10)
    p_raw = train_classifier_probe(X[train_idx], raw_mod10[train_idx], 10)
    p_carry = train_regression_probe(X[train_idx], carry_in[train_idx])

    carry_exact, carry_mae = eval_regression(p_carry, carry_in)
    return {
        "n_eval": len(eval_idx),
        "probe_true_digit_acc": eval_classifier(p_digit, true_digit),
        "probe_raw_mod10_acc": eval_classifier(p_raw, raw_mod10),
        "probe_carry_exact_acc": carry_exact,
        "probe_carry_mae": carry_mae,
        "chance_digit": 0.10, "chance_raw_mod10": 0.10,
    }


def run_per_loop(model, tensors, records, device, args) -> None:
    by_depth = extract_features_per_loop(model, tensors, records, device)
    depths = sorted(by_depth)
    places = sorted(by_depth[depths[-1]])

    grid = {"digit": {}, "raw": {}, "carry": {}}
    print(f"\nprobing every loop depth 1..{depths[-1]} x every place -- "
          f"{len(depths) * len(places)} linear probes per target\n")
    for depth in depths:
        row_digit, row_raw, row_carry = [], [], []
        for place in places:
            r = probe_place(by_depth[depth][place])
            grid["digit"][(depth, place)] = r["probe_true_digit_acc"]
            grid["raw"][(depth, place)] = r["probe_raw_mod10_acc"]
            grid["carry"][(depth, place)] = r["probe_carry_exact_acc"]
            row_digit.append(r["probe_true_digit_acc"])
            row_raw.append(r["probe_raw_mod10_acc"])
            row_carry.append(r["probe_carry_exact_acc"])
        print(f"depth={depth}  true_digit=" + " ".join(f"{v:.2f}" for v in row_digit))
        print(f"         raw_mod10=" + " ".join(f"{v:.2f}" for v in row_raw))
        print(f"         carry_in=" + " ".join(f"{v:.2f}" for v in row_carry))

    if not args.plot:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, key, title in zip(
        axes, ["digit", "raw", "carry"],
        ["probe: true output digit", "probe: raw pairing-sum mod 10", "probe: carry-in (exact)"],
    ):
        arr = np.array([[grid[key][(d, p)] for p in places] for d in depths])
        im = ax.imshow(arr, cmap="viridis", vmin=0, vmax=1, aspect="auto", origin="lower")
        ax.set_xticks(range(len(places))); ax.set_xticklabels(places)
        ax.set_yticks(range(len(depths))); ax.set_yticklabels(depths)
        ax.set_xlabel("place (0 = ones digit)")
        ax.set_title(title, fontsize=10)
    axes[0].set_ylabel("loop depth (hidden state after this many iterations)")
    figure.colorbar(im, ax=axes, label="probe accuracy", shrink=0.85)
    figure.suptitle(
        f"Does the missing info exist at an earlier depth and get destroyed, or never exist?"
        f"  ({args.checkpoint} / {args.cohort})",
        fontsize=12, y=1.05, x=0.01, ha="left")
    out = RESULTS / "plots" / f"probe_perloop_{Path(args.checkpoint).stem}_{args.cohort}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    print(f"\nwrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="exp5_full_rope_L2_loop4_s0.pt")
    parser.add_argument("--train-digits", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--ood-digits", type=int, nargs="+", default=[6, 7, 8])
    parser.add_argument("--cohort", default="id_5d")
    parser.add_argument("--n-eval", type=int, default=2_000)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--per-loop", action="store_true",
                        help="probe the hidden state after EVERY loop iteration, not just "
                             "the final one -- does the information exist earlier and get "
                             "destroyed, or never exist at all?")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = pick_device(args.device)
    model = load_checkpoint(CKPT_DIR / args.checkpoint, SQUARE).to(device)
    model.eval()
    print(f"loaded {args.checkpoint}  cohort={args.cohort}", flush=True)

    data = build_square(train_digits=tuple(args.train_digits), ood_digits=tuple(args.ood_digits),
                        n_train=100_000, n_eval=args.n_eval, carryless=False)
    max_seq_len = data.max_seq_len
    cohort = next(c for c in data.cohorts if c.name == args.cohort)
    tensors = tensors_from_records(cohort.records, data.tokenizer, max_seq_len)

    if args.per_loop:
        run_per_loop(model, tensors, cohort.records, device, args)
        return

    by_place = extract_features(model, tensors, cohort.records, device)

    print(f"\n{'place':>6}{'n':>7}{'probe:true_digit':>19}{'model:exact (pass4)':>21}"
          f"{'probe:raw_mod10':>18}{'probe:carry(exact/mae)':>25}")
    results = {}
    for place in sorted(by_place):
        r = probe_place(by_place[place])
        results[place] = r
        print(f"{place:>6}{r['n_eval']:>7}{r['probe_true_digit_acc']:>19.3f}"
              f"{'':<21}{r['probe_raw_mod10_acc']:>18.3f}"
              f"  {r['probe_carry_exact_acc']:.3f} / {r['probe_carry_mae']:.2f}")

    print("\n(fill in the model's actual per-place exact accuracy at this pass count from "
          "exp5_carry_mechanism.csv -- e.g. `grep ',full,.*,id_5d,.*,4,' results/exp5_carry_mechanism.csv` "
          "-- to compare probe recoverability against what the model's own head actually gets right)")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        places = sorted(results)
        figure, ax = plt.subplots(figsize=(8, 4.6))
        ax.plot(places, [results[p]["probe_true_digit_acc"] for p in places],
               marker="o", color="#104281", label="probe: true output digit")
        ax.plot(places, [results[p]["probe_raw_mod10_acc"] for p in places],
               marker="o", color="#1E6F5C", label="probe: raw pairing-sum mod 10")
        ax.plot(places, [results[p]["probe_carry_exact_acc"] for p in places],
               marker="o", color="#B03A2E", label="probe: carry-in (exact, rounded)")
        ax.axhline(0.10, color="grey", ls="--", lw=1, label="chance (10-way)")
        ax.set_xlabel("place (0 = ones digit)")
        ax.set_ylabel("linear probe accuracy (held-out)")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.25, linewidth=0.6); ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=9)
        ax.set_title(f"Can a linear probe recover it, even where the model's own head can't?  "
                     f"({args.checkpoint} / {args.cohort})", fontsize=10.5, loc="left")
        out = RESULTS / "plots" / f"probe_{Path(args.checkpoint).stem}_{args.cohort}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
