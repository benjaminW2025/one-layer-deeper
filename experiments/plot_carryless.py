"""Figure: carries are the entire bottleneck, and length is a separate wall.

Compares two targets trained with an identical model (RoPE, 4 layers, d=256,
digits 1-5, 8000 steps). The only difference is the target function:

    x * x                  full multiplication: convolution AND carries
    carryless convolution  place k = (sum_{i+j=k} x_i*x_j) mod 10, no carries

Same fan-in per place in both. If wide convolutions were the problem, both
would fail. They do not: carryless is solved, real multiplication is not.

Panel 2 shows that removing carries does NOT fix length extrapolation, so the
two failures are independent.

Usage:  python experiments/plot_carryless.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

RESULTS = Path(__file__).resolve().parent / "results"
CARRY = "#B03A2E"      # full multiplication
NOCARRY = "#1E6F5C"    # carryless
GREY = "#6B7280"


def rows_for(task: str) -> dict[str, dict]:
    """Last matching row per cohort wins, so reruns supersede."""
    out: dict[str, dict] = {}
    with (RESULTS / "exp2_square.csv").open() as handle:
        for row in csv.DictReader(handle):
            if (row["task"] == task and row["positional"] == "rope"
                    and int(row["n_layers"]) == 4
                    and row.get("train_digits", "") == "1-2-3-4-5"):
                out[row["cohort"]] = row
    return out


def main() -> None:
    full = rows_for("square")
    none = rows_for("carryless_square")
    if not full or not none:
        print("need both `square` and `carryless_square` rows; run exp2 with "
              "and without --carryless")
        return

    figure = plt.figure(figsize=(11, 4.6))
    grid = figure.add_gridspec(1, 2, width_ratios=[1.35, 1], wspace=0.28)
    left, right = figure.add_subplot(grid[0]), figure.add_subplot(grid[1])

    # --- panel 1: per-place accuracy on id_5d -----------------------------
    for task, rows, color, label in (
        ("carryless_square", none, NOCARRY, "carryless convolution"),
        ("square", full, CARRY, r"full $x^2$ (with carries)"),
    ):
        row = rows.get("id_5d")
        if not row:
            continue
        values = [float(v) for v in row["per_place_acc"].split("|")]
        left.plot(range(len(values)), values, marker="o", markersize=6,
                  color=color, linewidth=2,
                  label=f"{label}   exact = {float(row['exact_acc']):.3f}")
    left.set_xlabel("place value  (0 = ones digit)")
    left.set_ylabel("accuracy at this place")
    left.set_ylim(-0.04, 1.06)
    left.grid(alpha=0.25, linewidth=0.6)
    left.set_axisbelow(True)
    left.legend(frameon=False, fontsize=9, loc="lower center")
    left.set_title(r"Same convolution, carries removed — $x^2$, 5-digit $x$, unseen inputs",
                   fontsize=11, loc="left")
    left.annotate("carries removed:\nevery place solved", xy=(4.5, 1.0),
                  xytext=(3.4, 0.60), fontsize=9, color=NOCARRY, ha="center",
                  arrowprops=dict(arrowstyle="->", color=NOCARRY, lw=1.2))
    left.annotate("fan-in is IDENTICAL here —\nonly the carry chain differs",
                  xy=(5, 0.17), xytext=(6.6, 0.34), fontsize=9, color=CARRY,
                  ha="center",
                  arrowprops=dict(arrowstyle="->", color=CARRY, lw=1.2))

    # --- panel 2: exact match, in-distribution vs longer -------------------
    cohorts = ["id_4d", "id_5d", "ood_6d", "ood_7d"]
    labels = ["4 digits\n(trained)", "5 digits\n(trained)",
              "6 digits\n(unseen)", "7 digits\n(unseen)"]
    width = 0.38
    for offset, (rows, color, label) in enumerate((
        (none, NOCARRY, "carryless"),
        (full, CARRY, r"full $x^2$"),
    )):
        values = [float(rows[c]["exact_acc"]) if c in rows else 0.0 for c in cohorts]
        positions = [i + (offset - 0.5) * width for i in range(len(cohorts))]
        bars = right.bar(positions, values, width, color=color, label=label)
        for bar, value in zip(bars, values):
            right.text(bar.get_x() + bar.get_width() / 2,
                       max(value, 0) + 0.03, f"{value:.2f}",
                       ha="center", fontsize=8, color=color)
    right.axvline(1.5, color=GREY, linestyle="--", linewidth=1)
    right.text(1.55, 0.86, "trained lengths  |  longer", fontsize=8.5, color=GREY)
    right.set_xticks(range(len(cohorts)), labels, fontsize=9)
    right.set_ylabel("exact-match accuracy")
    right.set_ylim(0, 1.1)
    right.grid(alpha=0.25, axis="y", linewidth=0.6)
    right.set_axisbelow(True)
    right.legend(frameon=False, fontsize=9, loc="upper right")
    right.set_title("Removing carries does not buy length generalization",
                    fontsize=11, loc="left")

    figure.suptitle(
        "Carries — not wide convolutions — are what break multiplication",
        fontsize=13.5, y=1.03, x=0.008, ha="left",
    )
    out = RESULTS / "plots" / "carryless.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}\n")

    print(f"{'cohort':>9}{'full x^2':>11}{'carryless':>12}")
    for cohort in cohorts + ["train_seen"]:
        a = f"{float(full[cohort]['exact_acc']):.4f}" if cohort in full else "-"
        b = f"{float(none[cohort]['exact_acc']):.4f}" if cohort in none else "-"
        print(f"{cohort:>9}{a:>11}{b:>12}")


if __name__ == "__main__":
    main()
