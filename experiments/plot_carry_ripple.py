"""Does carry propagation actually ripple through passes?

Reads exp5_carry_mechanism.csv (four task variants x many pass counts, one
looped-weight-shared model per variant -- see exp5_carry_mechanism.py) and
plots per-place accuracy against pass count, one heatmap per variant.

The question this answers directly: if carries genuinely propagate
sequentially through the loop (place 0 resolves, feeds into place 1, which
resolves, feeds into place 2, ...), accuracy should trace a diagonal front --
place p only reaches its ceiling once pass count exceeds p by some margin.
If instead every place jumps to its ceiling at the same pass and then
flatlines, there's no ripple: whatever the loop is doing, it isn't
iteratively propagating carries place by place.

carry_only isolates this cleanly (input already has the convolution done,
so any ripple in ITS heatmap is unconfounded by also having to learn
pairing). full and carryless are the entangled comparisons; raw_sum is the
floor -- no carry step exists there at all, so its heatmap is the null
pattern the others should be compared against.

Usage:
  python experiments/plot_carry_ripple.py
  python experiments/plot_carry_ripple.py --cohort id_5d
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

RESULTS = Path(__file__).resolve().parent / "results"


def load_rows(csv_path: Path, cohort: str) -> dict[str, dict[int, list[float]]]:
    """variant -> {pass_count: per_place_acc list}, last row per (variant,pass) wins."""
    out: dict[str, dict[int, list[float]]] = {}
    with csv_path.open() as handle:
        for row in csv.DictReader(handle):
            if row["cohort"] != cohort:
                continue
            variant = row["variant"]
            pass_count = int(row["pass_count"])
            values = [float(v) for v in row["per_place_acc"].split("|") if v]
            out.setdefault(variant, {})[pass_count] = values
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=str(RESULTS / "exp5_carry_mechanism.csv"))
    parser.add_argument("--cohort", default="id_5d",
                        help="which cohort's per-place accuracy to plot against pass count")
    parser.add_argument("--variants", nargs="+",
                        default=["full", "carryless", "carry_only", "raw_sum"])
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"no {csv_path} yet -- run exp5_carry_mechanism.py first")
        return
    data = load_rows(csv_path, args.cohort)
    if not data:
        print(f"no rows for cohort={args.cohort} in {csv_path}. "
              f"available: run without --cohort to see options, or check the CSV directly.")
        return

    variants = [v for v in args.variants if v in data]
    if not variants:
        print(f"none of {args.variants} present; found {list(data)}")
        return

    print(f"=== per-place accuracy by pass count, cohort={args.cohort} ===")
    for variant in variants:
        passes = sorted(data[variant])
        n_places = max(len(data[variant][p]) for p in passes)
        print(f"\n{variant}")
        header = "  place" + "".join(f"{p:>8}" for p in passes)
        print(header)
        for place in range(n_places):
            row = []
            for p in passes:
                values = data[variant][p]
                row.append(f"{values[place]:>8.2f}" if place < len(values) else f"{'--':>8}")
            print(f"  {place:>5}" + "".join(row))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figure, axes = plt.subplots(1, len(variants), figsize=(4.4 * len(variants), 4.6),
                                squeeze=False)
    for ax, variant in zip(axes[0], variants):
        passes = sorted(data[variant])
        n_places = max(len(data[variant][p]) for p in passes)
        grid = np.full((n_places, len(passes)), np.nan)
        for ci, p in enumerate(passes):
            values = data[variant][p]
            grid[:len(values), ci] = values
        im = ax.imshow(grid, cmap="viridis", aspect="auto", vmin=0, vmax=1,
                       origin="lower")
        ax.set_xticks(range(len(passes))); ax.set_xticklabels(passes)
        ax.set_yticks(range(n_places))
        ax.set_xlabel("pass count (eval-time loop count)")
        ax.set_title(variant, fontsize=11)
    axes[0][0].set_ylabel("place (0 = ones digit)")
    figure.colorbar(im, ax=axes[0], label="accuracy", shrink=0.85)
    figure.suptitle(
        f"Does accuracy ripple diagonally across (place, pass)? cohort={args.cohort}\n"
        f"diagonal front = carries genuinely propagate through the loop; "
        f"uniform columns = they don't",
        fontsize=12, y=1.06, x=0.01, ha="left")
    out = RESULTS / "plots" / f"carry_ripple_{args.cohort}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
