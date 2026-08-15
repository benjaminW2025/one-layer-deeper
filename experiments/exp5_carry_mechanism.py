"""Which half of x*x actually breaks: the convolution, or the carry chain?

Trains four separate targets, same architecture (2 unique layers, looped),
so a difference between them isolates magnitude from propagation from
routing instead of leaving it all tangled inside one "does x*x work" number:

    full        x -> x*x                          real answer, real carries
    carryless   x -> per-place conv sums mod 10    convolution, carries removed
    raw_sum     x -> per-place conv sums, UNREDUCED  convolution alone, not
                                                      even folded mod 10
    carry_only  per-place conv sums -> x*x          carry propagation ALONE,
                                                      convolution already done

Because the block is looped with tied weights, pass count is a `forward()`
argument, not part of the model -- one training run gets evaluated at every
pass count in --pass-counts for free, no retraining. That's the axis the
carry-ripple hypothesis lives on: if carries genuinely ripple place by
place, per-place accuracy should trace a diagonal front across (place, pass)
-- place 0 solved by pass 1, place 1 by pass 2, etc. If every place jumps to
its ceiling at the same pass and then flatlines, there's no ripple and the
"iterative carry propagation" story is wrong.

Usage:
  python experiments/exp5_carry_mechanism.py
  python experiments/exp5_carry_mechanism.py --variants carry_only raw_sum --steps 4000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.model import POSITIONAL_MODES  # noqa: E402
from common.tasks import build_carry_only, build_raw_sum, build_square  # noqa: E402
from common.train import (  # noqa: E402
    TrainConfig, append_csv, build_model, evaluate, pick_device,
    save_checkpoint, tensors_from_records, train,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS = RESULTS_DIR / "exp5_carry_mechanism.csv"

VARIANTS = ["full", "carryless", "carry_only", "raw_sum"]


def build_dataset(variant: str, train_digits: tuple, ood_digits: tuple,
                  n_train: int, n_eval: int):
    if variant == "full":
        return build_square(train_digits=train_digits, ood_digits=ood_digits,
                            n_train=n_train, n_eval=n_eval, carryless=False)
    if variant == "carryless":
        return build_square(train_digits=train_digits, ood_digits=ood_digits,
                            n_train=n_train, n_eval=n_eval, carryless=True)
    if variant == "carry_only":
        return build_carry_only(train_digits=train_digits, ood_digits=ood_digits,
                                n_train=n_train, n_eval=n_eval)
    if variant == "raw_sum":
        return build_raw_sum(train_digits=train_digits, ood_digits=ood_digits,
                             n_train=n_train, n_eval=n_eval)
    raise ValueError(f"unknown variant {variant!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", default=VARIANTS, choices=VARIANTS)
    parser.add_argument("--train-digits", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--ood-digits", type=int, nargs="+", default=[6, 7, 8])
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=2,
                        help="number of UNIQUE blocks (looped --n-loops times)")
    parser.add_argument("--n-loops", type=int, default=4,
                        help="trained loop count; also the pass count error dumps are captured at")
    parser.add_argument("--pass-counts", type=int, nargs="+", default=[2, 4, 8, 16, 32],
                        help="loop counts to EVALUATE at (free -- forward() arg, no retraining)")
    parser.add_argument("--positional", default="rope", choices=list(POSITIONAL_MODES))
    parser.add_argument("--n-train", type=int, default=100_000)
    parser.add_argument("--n-eval", type=int, default=2_000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--loss-reduction", default="token", choices=["token", "example", "example_sum"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--save-checkpoints", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = pick_device(args.device)

    for variant in args.variants:
        data = build_dataset(variant, tuple(args.train_digits), tuple(args.ood_digits),
                             args.n_train, args.n_eval)
        max_seq_len = data.max_seq_len
        print(f"\n=== variant={variant}  train={len(data.train)}  "
              f"max_seq_len={max_seq_len} ===", flush=True)

        train_tensors = tensors_from_records(data.train, data.tokenizer, max_seq_len)
        cohort_tensors = {
            c.name: (tensors_from_records(c.records, data.tokenizer, max_seq_len), c.records)
            for c in data.cohorts if c.records
        }

        for seed in args.seeds:
            model = build_model(data.tokenizer, max_seq_len, args.positional, args.d_model,
                                args.n_layers, abacus_max_k=8, n_loops=args.n_loops)
            config = TrainConfig(steps=args.steps, batch_size=args.batch_size, lr=args.lr,
                                 loss_reduction=args.loss_reduction, seed=seed)
            print(f"  seed={seed}  L{args.n_layers} x loop{args.n_loops} "
                  f"(effective depth {args.n_layers * args.n_loops})", flush=True)
            summary = train(model, train_tensors, config, device)
            print(f"    loss {summary['first_loss']} -> {summary['final_loss']}  "
                  f"({summary['params']:,} params)", flush=True)

            if args.save_checkpoints:
                save_checkpoint(model, RESULTS_DIR / "ckpt" /
                    f"exp5_{variant}_{args.positional}_L{args.n_layers}_loop{args.n_loops}_s{seed}.pt",
                    {"max_seq_len": max_seq_len, "positional": args.positional,
                     "d_model": args.d_model, "n_layers": args.n_layers,
                     "n_loops": args.n_loops, "task": variant,
                     "train_digits": args.train_digits})

            rows, all_errors = [], {}
            for name, (tensors, records) in cohort_tensors.items():
                digit_len = int(name.split("_")[1].rstrip("d")) if name[-1] == "d" else -1
                per_pass = []
                for pass_count in args.pass_counts:
                    capture = pass_count == args.n_loops
                    metrics, errors = evaluate(model, tensors, records, device,
                                               capture_errors=capture, n_loops=pass_count)
                    rows.append({
                        "variant": variant, "positional": args.positional, "seed": seed,
                        "cohort": name, "digit_length": digit_len, "ood": name.startswith("ood"),
                        "n_layers": args.n_layers, "n_loops_trained": args.n_loops,
                        "pass_count": pass_count, "d_model": args.d_model,
                        "steps": summary["steps"], "params": summary["params"],
                        **metrics.as_row(),
                    })
                    per_pass.append(f"{pass_count}:{metrics.exact:.2f}")
                    if capture:
                        all_errors[name] = {
                            "n_wrong": len(errors), "n_total": metrics.n,
                            "per_place_acc": metrics.per_place, "examples": errors[:2000],
                        }
                print(f"    {name:<10} digit_len={digit_len:>2}  exact@pass=" + " ".join(per_pass),
                      flush=True)
            append_csv(RESULTS, rows)
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            (RESULTS_DIR / f"exp5_errors_{variant}_{args.positional}_L{args.n_layers}"
                          f"_loop{args.n_loops}_s{seed}.json").write_text(json.dumps(all_errors, indent=2))

    print(f"\nwrote {RESULTS} and per-variant error dumps")


if __name__ == "__main__":
    main()
