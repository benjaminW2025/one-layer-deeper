"""Figure: depth does not move the compute/guess boundary.

Two panels over place value for x^2 with 5-digit x (cohort id_5d), for RoPE at
2, 4 and 8 layers:

  top     exact accuracy per place
  bottom  near_rate = P(off by +-1 or +-2 | digit wrong)
          High near_rate means the sum is right and only the carry slipped.
          near_rate ~ 0.444 is what uniform guessing gives.

The point of the figure: the U in the top panel and the cliff in the bottom
panel sit at the same place regardless of depth.

Usage:  python experiments/plot_depth_boundary.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_error_magnitude import UNIFORM_NEAR, deltas_by_place  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
COHORT = "id_5d"
DEPTHS = (2, 4, 8)
COLORS = {2: "#8FA3C8", 4: "#3B5EA8", 8: "#12234A"}


def accuracy_by_place(depth: int) -> list[float] | None:
    """Last matching row wins, so a rerun supersedes an earlier one."""
    path = RESULTS / "exp2_square.csv"
    found = None
    with path.open() as handle:
        for row in csv.DictReader(handle):
            if (row["positional"] == "rope" and int(row["n_layers"]) == depth
                    and row["cohort"] == COHORT
                    and row.get("train_digits", "") == "1-2-3-4-5"):
                found = [float(v) for v in row["per_place_acc"].split("|")]
    return found


def near_rate_by_place(depth: int, min_errors: int = 30) -> dict[int, tuple[float, int]]:
    path = RESULTS / f"exp2_errors_rope_L{depth}_d256_s0.json"
    if not path.exists():
        return {}
    blob = json.loads(path.read_text())
    if COHORT not in blob:
        return {}
    out = {}
    for place, counts in deltas_by_place(blob[COHORT]).items():
        total = sum(counts.values())
        if total >= min_errors:
            out[place] = (sum(counts[d] for d in (1, 2, 8, 9)) / total, total)
    return out


def main() -> None:
    accuracy = {d: accuracy_by_place(d) for d in DEPTHS}
    near = {d: near_rate_by_place(d) for d in DEPTHS}
    if not any(accuracy.values()):
        print("no matching rows in exp2_square.csv; run the depth sweep first")
        return

    figure, (top, bottom) = plt.subplots(
        2, 1, figsize=(9, 7), sharex=True,
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.12},
    )

    # Shade the region that stays at chance no matter the depth.
    for axis in (top, bottom):
        axis.axvspan(3.5, 5.5, color="#C8443A", alpha=0.07, zorder=0)
        axis.grid(alpha=0.25, linewidth=0.6)
        axis.set_axisbelow(True)

    for depth in DEPTHS:
        values = accuracy.get(depth)
        if values:
            top.plot(range(len(values)), values, marker="o", markersize=5,
                     color=COLORS[depth], label=f"{depth} layers")
    top.set_ylabel("exact accuracy at this place")
    top.set_ylim(-0.03, 1.05)
    top.set_title(
        "Depth does not move the boundary\n"
        r"$x^2$ for 5-digit $x$, unseen inputs (id_5d), RoPE",
        fontsize=13, loc="left",
    )
    top.legend(frameon=False, fontsize=9)
    top.text(4.5, 0.62, "pinned at chance\nat every depth", ha="center",
             fontsize=9, color="#8C2F27")

    for depth in DEPTHS:
        points = sorted(near.get(depth, {}).items())
        if points:
            places = [p for p, _ in points]
            rates = [v for _, (v, _) in points]
            counts = [n for _, (_, n) in points]
            bottom.plot(places, rates, color=COLORS[depth], linewidth=1.4,
                        label=f"{depth} layers", zorder=3)
            # Marker area tracks how many wrong digits the estimate is built
            # from: tiny markers are near-noise, big ones are solid.
            bottom.scatter(places, rates, s=[min(160, 12 + n / 12) for n in counts],
                           color=COLORS[depth], zorder=4, edgecolor="white",
                           linewidth=0.6)
    bottom.axhline(UNIFORM_NEAR, color="#8C2F27", linestyle="--", linewidth=1.2)
    bottom.text(0.15, UNIFORM_NEAR + 0.02, "uniform guessing (0.444)",
                fontsize=9, color="#8C2F27")
    bottom.set_ylabel("P(off by ±1 or ±2 | wrong)")
    bottom.set_xlabel("place value  (0 = ones digit)")
    bottom.set_ylim(0.0, 1.08)
    bottom.legend(frameon=False, fontsize=9, loc="lower right")
    bottom.annotate("carry slips\n(sum is right)", xy=(7, 0.95), xytext=(7.6, 0.72),
                    fontsize=9, color="#1B4D3E", ha="center",
                    arrowprops=dict(arrowstyle="->", color="#1B4D3E", lw=1))
    bottom.annotate("guessing\n(nothing computed)", xy=(4.5, 0.50),
                    xytext=(2.2, 0.25), fontsize=9, color="#8C2F27", ha="center",
                    arrowprops=dict(arrowstyle="->", color="#8C2F27", lw=1))
    bottom.text(0.05, 0.06, "marker size = number of wrong digits behind the estimate",
                transform=bottom.transAxes, fontsize=8, color="#555555")

    out = RESULTS / "plots" / "depth_boundary.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")

    # Same content as text, for pasting into notes.
    print(f"\n{'place':>6}" + "".join(f"{f'L{d} acc':>9}" for d in DEPTHS)
          + "".join(f"{f'L{d} near/n':>13}" for d in DEPTHS))
    width = max((len(v) for v in accuracy.values() if v), default=0)
    for place in range(width - 1, -1, -1):
        line = f"{place:>6}"
        for depth in DEPTHS:
            v = accuracy.get(depth)
            line += f"{v[place]:>9.2f}" if v and place < len(v) else f"{'-':>9}"
        for depth in DEPTHS:
            hit = near.get(depth, {}).get(place)
            line += f"{hit[0]:>7.3f}/{hit[1]:<5}" if hit else f"{'-':>13}"
        print(line)


if __name__ == "__main__":
    main()
