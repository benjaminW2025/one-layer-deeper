"""Experiment 4 — x^2 mod N at fixed N. This is the competition's T=1.

Run LAST, with whatever exp1-3 established. If square (exp2) and reduce
(exp3 fixed_n) both work but this does not, the wall is composition and the
depth question becomes live. If either primitive already fails, this will
fail for that reason and tells us nothing new.

Two cohorts, so memorization is measured rather than inferred:
    train_seen   x sampled from the training pool  -> ~1.0 if it memorized
    test_fresh   x never trained on                -> the real number

`x_coverage` (printed and recorded) is n_train / phi(N): the fraction of the
whole input space that was trained on. If it is large, a lookup table is
available and a low test_fresh score means nothing more than "it memorized".
Raise --bits to shrink coverage without changing anything else about the task.

Usage:
  python experiments/exp4_squaremod.py
  python experiments/exp4_squaremod.py --bits 18 22 --positional abacus
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.model import POSITIONAL_MODES  # noqa: E402
from common.tasks import build_squaremod  # noqa: E402
from common.train import (  # noqa: E402
    TrainConfig, append_csv, build_model, evaluate, pick_device,
    tensors_from_records, train,
)

RESULTS = Path(__file__).resolve().parent / "results" / "exp4_squaremod.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bits", type=int, nargs="+", default=[18])
    parser.add_argument("--positional", nargs="+", default=["learned"],
                        choices=list(POSITIONAL_MODES))
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-train", type=int, default=50_000)
    parser.add_argument("--n-eval", type=int, default=2_000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--loss-reduction", default="token", choices=["token", "example"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = pick_device(args.device)
    print(f"device: {device}", flush=True)

    for bits in args.bits:
        data = build_squaremod(
            bits=bits, n_train=args.n_train, n_eval=args.n_eval,
        )
        max_seq_len = max(
            len(r["input_ids"])
            for group in [data.train] + [c.records for c in data.cohorts]
            for r in group
        )
        meta = data.meta
        print(f"\n##### bits={bits} N={meta['modulus']} x_space={meta['x_space']} "
              f"coverage={meta['x_coverage']:.2%} max_seq_len={max_seq_len}", flush=True)
        if meta["x_coverage"] > 0.10:
            print("  WARNING: coverage >10%; a lookup table is available, so a "
                  "low test_fresh score will not distinguish 'cannot compute' "
                  "from 'memorized'. Raise --bits.", flush=True)

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
                print(f"=== squaremod b{bits} positional={positional} seed={seed} ===",
                      flush=True)
                summary = train(model, train_tensors, config, device)

                scores, rows = {}, []
                for name, (tensors, records) in cohort_tensors.items():
                    metrics, _ = evaluate(model, tensors, records, device)
                    scores[name] = metrics.exact
                    rows.append({
                        "task": "squaremod", "bits": bits, "modulus": meta["modulus"],
                        "positional": positional, "seed": seed, "cohort": name,
                        "x_space": meta["x_space"],
                        "x_coverage": round(meta["x_coverage"], 6),
                        "d_model": args.d_model, "n_layers": args.n_layers,
                        "steps": summary["steps"], "params": summary["params"],
                        "lr": args.lr, "loss_reduction": args.loss_reduction,
                        "final_loss": summary["final_loss"],
                        "diverged": summary["diverged"], "flat": summary["flat"],
                        "device": summary["device"],
                        **metrics.as_row(),
                    })
                    print(f"  {name:<12} exact={metrics.exact:.4f} "
                          f"digit={metrics.digit:.4f} (chance {metrics.chance_exact:.4f})",
                          flush=True)
                append_csv(RESULTS, rows)
                gap = scores.get("train_seen", 0) - scores.get("test_fresh", 0)
                print(f"  loss={summary['final_loss']}  memorization gap="
                      f"{gap:+.4f}", flush=True)

    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    main()
