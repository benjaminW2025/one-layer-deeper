"""Experiment 1: baseline transformer performance surface.

One representative architecture (4 layers, d_model 256) trained on two regimes
that exp0 predicts sit in *opposite* structural camps:

  - e1_fixed323        : tiny fixed N; cycle period is small, so the shortcut
                         regime — accuracy at large T is achievable without
                         sequential depth if the model finds the cycle.
  - easy_sampled_b1011 : sampled 10-11-bit N; periods should dwarf the ladder,
                         the (conjectured) sequential regime.

Every (regime, budget-mode) cell evaluates the full dense T ladder on both the
seen-N fresh-x cohort and the OOD-N cohort, giving accuracy-vs-T curves with
chance baselines attached (harness attaches them per eval set).

Budget modes (methodology requirement — both, labeled):
  - steps   : fixed 1500 optimizer steps (expressivity, cost ignored)
  - seconds : fixed 60 s wall-clock including model construction (the
              competition's actual constraint, scaled to the Easy tier)

Run-to-run variance: seeds {0, 1, 2} on easy_sampled_b1011/steps; every other
cell runs seed 0 only.

Usage:
  python explorations/exp1_baseline.py                # full pass
  python explorations/exp1_baseline.py --regimes e1_fixed323 --modes steps
"""

from __future__ import annotations

import argparse
import time

from harness import REGIMES, DataBundle, RunConfig, append_rows, build_data, run_experiment

DEFAULT_REGIMES = ("e1_fixed323", "easy_sampled_b1011")
MODES = {"steps": 1500.0, "seconds": 60.0}
VARIANCE_CELL = ("easy_sampled_b1011", "steps")  # gets seeds (0, 1, 2)
N_LAYERS = 4
D_MODEL = 256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regimes", nargs="+", default=list(DEFAULT_REGIMES),
                        choices=sorted(REGIMES))
    parser.add_argument("--modes", nargs="+", default=list(MODES), choices=list(MODES))
    parser.add_argument("--steps-budget", type=float, default=MODES["steps"])
    parser.add_argument("--seconds-budget", type=float, default=MODES["seconds"])
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    budgets = {"steps": args.steps_budget, "seconds": args.seconds_budget}

    # Data is generated once per regime and shared across budget modes and
    # seeds so curves differ only by training procedure, never by data.
    bundles: dict[str, DataBundle] = {}
    for regime_name in args.regimes:
        print(f"building data for {regime_name} ...", flush=True)
        bundles[regime_name] = build_data(REGIMES[regime_name], seed=45)

    total_start = time.monotonic()
    for regime_name in args.regimes:
        for mode in args.modes:
            seeds = (0, 1, 2) if (regime_name, mode) == VARIANCE_CELL else (0,)
            for seed in seeds:
                config = RunConfig(
                    exp="exp1", regime=regime_name, n_layers=N_LAYERS,
                    d_model=D_MODEL, budget_mode=mode, budget=budgets[mode],
                    lr=args.lr, seed=seed, device=args.device,
                )
                print(f"running {config.run_id} ...", flush=True)
                rows, summary = run_experiment(config, bundles[regime_name])
                append_rows(rows)
                flags = []
                if summary["diverged"]:
                    flags.append("DIVERGED")
                if summary["flat_from_start"]:
                    flags.append("FLAT-FROM-START (LR suspect)")
                print(f"  steps={summary['steps']} "
                      f"train_s={summary['train_seconds']:.1f} "
                      f"final_loss={summary['final_train_loss']:.4f} "
                      f"params={summary['params']:,} {' '.join(flags)}")
    print(f"exp1 done in {(time.monotonic() - total_start) / 60:.1f} min; "
          f"rows appended to results/results.csv")


if __name__ == "__main__":
    main()
