"""Experiment 3 — can the transformer learn y mod N?

The other half of x^2 mod N. y is drawn from [0, N^2) so the reduction is
never the identity (if y < N then y mod N = y and the task is a copy).

Three variants, because "OOD" means something different in each:

    fixed_n    one modulus, y varies      -> can it divide by a KNOWN constant?
    fixed_y    one y, modulus varies      -> can it divide a KNOWN value?
    vary_both  both vary, N held out      -> can it divide, generally?

fixed_n is the easiest and is the direct analogue of the competition's
fixed-modulus tiers. vary_both is the one that needs a real algorithm: its
ood_n cohort uses moduli never seen in training.

Usage:
  python experiments/exp3_reduce.py
  python experiments/exp3_reduce.py --variants vary_both --positional abacus
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.model import POSITIONAL_MODES  # noqa: E402
from common.tasks import build_reduce  # noqa: E402
from common.train import (  # noqa: E402
    TrainConfig, append_csv, build_model, evaluate, pick_device,
    save_checkpoint, tensors_from_records, train,
)

RESULTS = Path(__file__).resolve().parent / "results" / "exp3_reduce.csv"
VARIANTS = ("fixed_n", "fixed_y", "vary_both")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    parser.add_argument("--positional", nargs="+", default=["learned"],
                        choices=list(POSITIONAL_MODES))
    parser.add_argument("--bits", type=int, default=14)
    parser.add_argument("--n-moduli", type=int, default=32)
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
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--save-checkpoints", action="store_true",
                        help="write results/ckpt/<task>_<positional>_L<n>_d<n>_s<seed>.pt")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = pick_device(args.device)
    print(f"device: {device}", flush=True)

    for variant in args.variants:
        data = build_reduce(
            variant=variant, bits=args.bits, n_moduli=args.n_moduli,
            n_train=args.n_train, n_eval=args.n_eval,
        )
        max_seq_len = data.max_seq_len
        print(f"\n##### {data.task}  {data.meta}  max_seq_len={max_seq_len}", flush=True)

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
                print(f"=== {variant} positional={positional} seed={seed} ===", flush=True)
                summary = train(model, train_tensors, config, device)
                print(f"  loss {summary['first_loss']} -> {summary['final_loss']}",
                      flush=True)

                rows = []
                for name, (tensors, records) in cohort_tensors.items():
                    metrics, _ = evaluate(model, tensors, records, device)
                    rows.append({
                        "task": data.task, "variant": variant,
                        "positional": positional, "seed": seed, "cohort": name,
                        "ood": name.startswith("ood"), "bits": args.bits,
                        "n_train_moduli": data.meta["n_train_moduli"],
                        "n_ood_moduli": data.meta["n_ood_moduli"],
                        "d_model": args.d_model, "n_layers": args.n_layers,
                        "steps": summary["steps"], "params": summary["params"],
                        "lr": args.lr, "abacus_max_k": args.abacus_max_k,
                    "loss_reduction": args.loss_reduction,
                        "final_loss": summary["final_loss"],
                        "diverged": summary["diverged"], "flat": summary["flat"],
                        "device": summary["device"],
                        **metrics.as_row(),
                    })
                    print(f"  {name:<12} exact={metrics.exact:.4f} "
                          f"digit={metrics.digit:.4f} (chance {metrics.chance_exact:.4f})",
                          flush=True)
                append_csv(RESULTS, rows)

    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    main()
