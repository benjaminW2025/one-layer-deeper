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
    tensors_from_records, train,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS = RESULTS_DIR / "exp2_square.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positional", nargs="+", default=["learned"],
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
    parser.add_argument("--loss-reduction", default="token", choices=["token", "example"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = pick_device(args.device)
    data = build_square(
        train_digits=tuple(args.train_digits), ood_digits=tuple(args.ood_digits),
        n_train=args.n_train, n_eval=args.n_eval,
    )
    max_seq_len = max(
        len(r["input_ids"])
        for group in [data.train] + [c.records for c in data.cohorts]
        for r in group
    )
    print(f"device={device}  train={len(data.train)}  max_seq_len={max_seq_len}",
          flush=True)

    train_tensors = tensors_from_records(data.train, max_seq_len)
    cohort_tensors = {
        c.name: (tensors_from_records(c.records, max_seq_len), c.records)
        for c in data.cohorts
    }

    for positional in args.positional:
        for seed in args.seeds:
            model = build_model(max_seq_len, positional, args.d_model, args.n_layers)
            config = TrainConfig(
                steps=args.steps, batch_size=args.batch_size, lr=args.lr,
                loss_reduction=args.loss_reduction, seed=seed,
            )
            print(f"\n=== square positional={positional} seed={seed} ===", flush=True)
            summary = train(model, train_tensors, config, device)
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
                    "examples": errors[:200],
                }
                rows.append({
                    "task": "square", "positional": positional, "seed": seed,
                    "cohort": name, "ood": name.startswith("ood"),
                    "digits": int(name.split("_")[1].rstrip("d")),
                    "d_model": args.d_model, "n_layers": args.n_layers,
                    "steps": summary["steps"], "params": summary["params"],
                    "lr": args.lr, "loss_reduction": args.loss_reduction,
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
            (RESULTS_DIR / f"exp2_errors_{positional}_s{seed}.json").write_text(
                json.dumps(all_errors, indent=2)
            )

    print(f"\nwrote {RESULTS} and per-cohort error dumps")


if __name__ == "__main__":
    main()
