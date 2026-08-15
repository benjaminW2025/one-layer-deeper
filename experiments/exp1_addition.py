"""Experiment 1 — can the transformer learn addition, and what positional
information does it need to generalize?

This runs FIRST. Addition is the simplest possible digit task: no modulus, no
composition, carries only. If a positional scheme cannot generalize here it
will not generalize on x^2 or x^2 mod N, so this is the cheapest place to
learn the lesson.

Train on operands of 1-4 digits. Evaluate on:
    id_Nd   seen digit length, unseen operand values  -> did it learn addition?
    ood_Nd  5, 6, 7 digits, never seen                -> did it learn to COUNT?

Sweeps positional schemes:
    none            no position at all (bidirectional -> should fail; floor)
    learned         what the competition baseline uses (absolute)
    sinusoidal      fixed absolute
    rope            relative
    abacus          place value: each digit's index within its own number
    abacus_learned  place value + absolute. The combination used by
                    "Transformers Can Do Arithmetic with the Right Embeddings"
                    (McLeish et al., NeurIPS 2024, arXiv 2405.17399), whose
                    best results add Abacus alongside a standard positional
                    embedding rather than replacing it.
    abacus_rope     place value + relative field order

Read the result as: everything should do well on id_*; the interesting column
is ood_*, which is where absolute schemes are expected to collapse.

Usage:
  python experiments/exp1_addition.py                          # all 7 modes
  python experiments/exp1_addition.py --positional abacus_learned
  python experiments/exp1_addition.py --positional learned abacus abacus_learned
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.model import POSITIONAL_MODES  # noqa: E402
from common.tasks import build_addition  # noqa: E402
from common.train import (  # noqa: E402
    TrainConfig, append_csv, build_model, evaluate, pick_device,
    save_checkpoint, tensors_from_records, train,
)

RESULTS = Path(__file__).resolve().parent / "results" / "exp1_addition.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positional", nargs="+", default=list(POSITIONAL_MODES),
                        choices=list(POSITIONAL_MODES))
    parser.add_argument("--train-digits", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--ood-digits", type=int, nargs="+", default=[5, 6, 7])
    parser.add_argument("--steps", type=int, default=6000)
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
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--save-checkpoints", action="store_true",
                        help="write results/ckpt/<task>_<positional>_L<n>_d<n>_s<seed>.pt")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = pick_device(args.device)
    print(f"device: {device}", flush=True)

    data = build_addition(
        train_digits=tuple(args.train_digits), ood_digits=tuple(args.ood_digits),
        n_train=args.n_train, n_eval=args.n_eval,
    )
    # One sequence width across train and every cohort so positions mean the
    # same thing everywhere (the OOD cohorts are the long ones).
    max_seq_len = data.max_seq_len
    print(f"train={len(data.train)}  max_seq_len={max_seq_len}  "
          f"cohorts={[c.name for c in data.cohorts]}", flush=True)

    train_tensors = tensors_from_records(data.train, data.tokenizer, max_seq_len)
    cohort_tensors = {
        c.name: (tensors_from_records(c.records, data.tokenizer, max_seq_len), c.records)
        for c in data.cohorts
    }

    for positional in args.positional:
        for seed in args.seeds:
            model = build_model(data.tokenizer, max_seq_len, positional, args.d_model,
                                    args.n_layers, abacus_max_k=args.abacus_max_k)
            config = TrainConfig(
                steps=args.steps, batch_size=args.batch_size, lr=args.lr,
                loss_reduction=args.loss_reduction, seed=seed,
            )
            print(f"\n=== positional={positional} seed={seed} ===", flush=True)
            summary = train(model, train_tensors, config, device)
            if args.save_checkpoints:
                save_checkpoint(model, RESULTS.parent / "ckpt" /
                    f"addition_{positional}_L{args.n_layers}_d{args.d_model}_s{seed}.pt",
                    {"max_seq_len": max_seq_len, "positional": positional,
                     "d_model": args.d_model, "n_layers": args.n_layers,
                     "abacus_max_k": args.abacus_max_k, "task": "addition",
                     "train_digits": args.train_digits})
            print(f"  loss {summary['first_loss']} -> {summary['final_loss']}  "
                  f"({summary['params']:,} params, {summary['train_seconds']}s)",
                  flush=True)

            rows = []
            for name, (tensors, records) in cohort_tensors.items():
                metrics, _ = evaluate(model, tensors, records, device)
                rows.append({
                    "task": "addition", "positional": positional, "seed": seed,
                    "train_digits": "-".join(map(str, args.train_digits)),
                    "ood_digits": "-".join(map(str, args.ood_digits)),
                    "n_train": len(data.train),
                    "cohort": name,
                    "ood": name.startswith("ood"),
                    "digits": int(name.split("_")[1].rstrip("d")),
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
                      f"digit={metrics.digit:.4f} (chance {metrics.chance_exact:.4f})",
                      flush=True)
            append_csv(RESULTS, rows)

    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    main()
