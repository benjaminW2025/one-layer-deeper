"""Experiment 3: depth x width against T — the 2D surface.

For each (n_layers, d_model) cell, train under BOTH budget modes and evaluate
exact-match accuracy on every ladder rung (seen-N fresh-x and OOD-N cohorts).
The derived quantity — largest solvable T per depth, and whether it scales like
T ~ L, T ~ 2^L, or tracks the exp0 period instead — is computed by plots.py
from results.csv, not here.

Grid trimming (~2 h cap, as requested): the full requested grid was
  n_layers in {1,2,3,4,6,8} x d_model in {64,128,256,512}  x 2 regimes x 2 modes
  = 96 runs, roughly 3+ h on one L40.
Default here cuts depth 3 and 6 (the {1,2,4,8} geometric subset still resolves
T~L vs T~2^L; 3 and 6 only refine the boundary) and width 64 (exp1's d256
baseline plus {128,256,512} brackets capacity; d64 mostly measures
under-capacity, not depth). That is 4x3x2x2 = 48 runs ~= 1.5 h. Restore the
full grid with:  --depths 1 2 3 4 6 8 --widths 64 128 256 512

Both regimes run by default because they sit in opposite structural camps
(fixed-323: cycle shortcut available; sampled 10-11 bit: sequential regime).

Usage:
  python explorations/exp3_grid.py
  python explorations/exp3_grid.py --regimes e1_fixed323 --modes seconds
"""

from __future__ import annotations

import argparse
import time

from harness import REGIMES, DataBundle, RunConfig, append_rows, build_data, run_experiment

DEFAULT_DEPTHS = (1, 2, 4, 8)
DEFAULT_WIDTHS = (128, 256, 512)
DEFAULT_REGIMES = ("e1_fixed323", "easy_sampled_b1011")
MODE_BUDGETS = {"steps": 1000.0, "seconds": 45.0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depths", type=int, nargs="+", default=list(DEFAULT_DEPTHS))
    parser.add_argument("--widths", type=int, nargs="+", default=list(DEFAULT_WIDTHS))
    parser.add_argument("--regimes", nargs="+", default=list(DEFAULT_REGIMES),
                        choices=sorted(REGIMES))
    parser.add_argument("--modes", nargs="+", default=list(MODE_BUDGETS),
                        choices=list(MODE_BUDGETS))
    parser.add_argument("--steps-budget", type=float, default=MODE_BUDGETS["steps"])
    parser.add_argument("--seconds-budget", type=float, default=MODE_BUDGETS["seconds"])
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="one LR for every cell, deliberately untuned; "
                        "flat/diverged cells are flagged in the CSV instead")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    budgets = {"steps": args.steps_budget, "seconds": args.seconds_budget}
    n_runs = len(args.depths) * len(args.widths) * len(args.regimes) * len(args.modes)
    # Rough per-run cost: seconds-mode = budget + eval; steps-mode varies with size.
    rough_minutes = n_runs * (args.seconds_budget + 30) / 60
    print(f"grid: {len(args.depths)} depths x {len(args.widths)} widths x "
          f"{len(args.regimes)} regimes x {len(args.modes)} modes = {n_runs} runs "
          f"(very roughly ~{rough_minutes:.0f} min)", flush=True)

    bundles: dict[str, DataBundle] = {}
    for regime_name in args.regimes:
        print(f"building data for {regime_name} ...", flush=True)
        bundles[regime_name] = build_data(REGIMES[regime_name], seed=45)

    total_start = time.monotonic()
    completed = 0
    for regime_name in args.regimes:
        for mode in args.modes:
            for n_layers in args.depths:
                for d_model in args.widths:
                    config = RunConfig(
                        exp="exp3", regime=regime_name, n_layers=n_layers,
                        d_model=d_model, budget_mode=mode, budget=budgets[mode],
                        lr=args.lr, seed=args.seed, device=args.device,
                    )
                    rows, summary = run_experiment(config, bundles[regime_name])
                    append_rows(rows)
                    completed += 1
                    flags = []
                    if summary["diverged"]:
                        flags.append("DIVERGED")
                    if summary["flat_from_start"]:
                        flags.append("FLAT (LR suspect)")
                    elapsed = (time.monotonic() - total_start) / 60
                    print(f"[{completed}/{n_runs} {elapsed:.0f}m] {config.run_id}: "
                          f"steps={summary['steps']} "
                          f"loss={summary['final_train_loss']:.4f} "
                          f"{' '.join(flags)}", flush=True)

    print(f"exp3 done in {(time.monotonic() - total_start) / 60:.1f} min; "
          f"rows in results/results.csv — run plots.py for the depth x T heatmap "
          f"and the max-solvable-T-vs-depth verdict")


if __name__ == "__main__":
    main()
