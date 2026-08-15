"""Experiment 2 — can the 3M-param transformer learn x -> x*x?

Run AFTER exp1, using whichever positional scheme won there (pass it with
--positional; default `learned` reproduces the competition baseline).

Squaring is multiplication without a modulus: harder than addition (every
output digit depends on many input digits, not just its neighbours) but with
no reduction step. If this fails, x^2 mod N cannot work and the wall is
multiplication, not composition.

Cohorts:
    id_Nd   seen digit length, unseen x   -> did it learn to multiply?
    ood_Nd  unseen digit length           -> did it learn to count places?

Errors are captured for the plot: for every wrong answer we keep the true and
predicted digit strings and how many digits were wrong, written to
results/exp2_errors_<positional>.json.

Usage:
  python experiments/exp2_square.py --positional abacus
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.model import POSITIONAL_MODES  # noqa: E402
from common.tasks import build_square  # noqa: E402
from common.train import (  # noqa: E402
    TrainConfig, append_csv, build_model, evaluate, pick_device,
    save_checkpoint, tensors_from_records, train,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS = RESULTS_DIR / "exp2_square.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positional", nargs="+", default=["learned", "abacus"],
                        choices=list(POSITIONAL_MODES))
    parser.add_argument("--train-digits", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--ood-digits", type=int, nargs="+", default=[5, 6])
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-train", type=int, default=100_000)
    parser.add_argument("--n-eval", type=int, default=2_000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--abacus-max-k", type=int, default=8,
                        help="random digit-index shift during training; the "
                             "paper uses 99 for ~120-digit operands, ours are <=7")
    parser.add_argument("--loss-reduction", default="token", choices=["token", "example", "example_sum"])
    parser.add_argument("--carryless", action="store_true",
                        help="target is the digit-wise convolution mod 10: same\nfan-in per place, no carry propagation at all")
    parser.add_argument("--attention-sink", action="store_true",
                        help="give each head a learned per-head softmax-denominator "
                             "scalar (init so it contributes 1), so it can dump "
                             "attention mass on nothing instead of over irrelevant digits")
    parser.add_argument("--n-loops", type=int, default=1,
                        help="--n-layers becomes the number of UNIQUE blocks; the "
                             "stack runs through them --n-loops times with shared "
                             "weights (universal-transformer-style recurrence)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--save-checkpoints", action="store_true",
                        help="write results/ckpt/<task>_<positional>_L<n>_d<n>_s<seed>.pt")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = pick_device(args.device)
    data = build_square(
        train_digits=tuple(args.train_digits), ood_digits=tuple(args.ood_digits),
        n_train=args.n_train, n_eval=args.n_eval, carryless=args.carryless,
    )
    max_seq_len = data.max_seq_len
    print(f"device={device}  train={len(data.train)}  max_seq_len={max_seq_len}",
          flush=True)

    train_tensors = tensors_from_records(data.train, data.tokenizer, max_seq_len)
    cohort_tensors = {
        c.name: (tensors_from_records(c.records, data.tokenizer, max_seq_len), c.records)
        for c in data.cohorts
    }

    # tag distinguishes sink runs in filenames/CSV so they never collide
    # with (or silently overwrite, per rows_for()'s "last row wins") a
    # plain run of the same positional mode; append_csv's schema is fixed
    # by the first row ever written, so attention_sink can't be its own
    # column without breaking every existing results file.
    for positional in args.positional:
        label = f"{positional}+sink" if args.attention_sink else positional
        if args.n_loops > 1:
            label += f"+loop{args.n_loops}"
        for seed in args.seeds:
            model = build_model(data.tokenizer, max_seq_len, positional, args.d_model,
                                    args.n_layers, abacus_max_k=args.abacus_max_k,
                                    attention_sink=args.attention_sink,
                                    n_loops=args.n_loops)
            config = TrainConfig(
                steps=args.steps, batch_size=args.batch_size, lr=args.lr,
                loss_reduction=args.loss_reduction, seed=seed,
            )
            print(f"\n=== square positional={label} seed={seed} "
                  f"(n_layers={args.n_layers} x n_loops={args.n_loops} = "
                  f"{args.n_layers * args.n_loops} effective depth) ===", flush=True)
            summary = train(model, train_tensors, config, device)
            if args.save_checkpoints:
                save_checkpoint(model, RESULTS_DIR / "ckpt" /
                    f"{data.task}_{label}_L{args.n_layers}_d{args.d_model}_s{seed}.pt",
                    {"max_seq_len": max_seq_len, "positional": positional,
                     "d_model": args.d_model, "n_layers": args.n_layers,
                     "abacus_max_k": args.abacus_max_k, "task": "square",
                     "attention_sink": args.attention_sink, "n_loops": args.n_loops,
                     "train_digits": args.train_digits})
            print(f"  loss {summary['first_loss']} -> {summary['final_loss']}  "
                  f"({summary['params']:,} params)", flush=True)

            rows, all_errors = [], {}
            for name, (tensors, records) in cohort_tensors.items():
                metrics, errors = evaluate(
                    model, tensors, records, device, capture_errors=True
                )
                all_errors[name] = {
                    "n_wrong": len(errors),
                    "n_total": metrics.n,
                    "wrong_digit_histogram": dict(
                        Counter(e["n_wrong_digits"] for e in errors)
                    ),
                    "per_place_acc": metrics.per_place,
                    "examples": errors[:5000],
                }
                rows.append({
                    "task": data.task, "positional": label, "seed": seed,
                    "train_digits": "-".join(map(str, args.train_digits)),
                    "ood_digits": "-".join(map(str, args.ood_digits)),
                    "n_train": len(data.train),
                    "cohort": name, "ood": name.startswith("ood"),
                    "digits": int(name.split("_")[1].rstrip("d")) if name[-1] == "d" else -1,
                    "d_model": args.d_model, "n_layers": args.n_layers,
                    "steps": summary["steps"], "params": summary["params"],
                    "lr": args.lr, "abacus_max_k": args.abacus_max_k,
                    "loss_reduction": args.loss_reduction,
                    "final_loss": summary["final_loss"],
                    "diverged": summary["diverged"], "flat": summary["flat"],
                    "device": summary["device"],
                    **metrics.as_row(),
                })
                print(f"  {name:<10} exact={metrics.exact:.4f} "
                      f"digit={metrics.digit:.4f} (chance {metrics.chance_exact:.4f})  "
                      f"per-place={['%.2f' % v for v in metrics.per_place]}",
                      flush=True)
            append_csv(RESULTS, rows)
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            (RESULTS_DIR / f"exp2_errors_{data.task}_{label}_L{args.n_layers}_d{args.d_model}_s{seed}.json").write_text(
                json.dumps(all_errors, indent=2)
            )

    print(f"\nwrote {RESULTS} and per-cohort error dumps")


if __name__ == "__main__":
    main()
