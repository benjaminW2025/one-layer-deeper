"""Figure: digit-wise (per-place) accuracy of carryless multiplication.

Unlike exact-match accuracy, which is all-or-nothing across the whole answer,
per-place accuracy shows *where* the model breaks down as digit length grows
past the trained range (1-5 digits). Carryless multiplication is solved
in-distribution at every place, but degrades unevenly by position once the
input is longer than anything seen in training.

Usage:  python experiments/plot_carryless_digitwise.py
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

# sequential blue ramp, light -> dark, one step per trained digit length (1-5)
ID_RAMP = ["#9ec5f4", "#6da7ec", "#3987e5", "#1c5cab", "#104281"]
OOD_COLOR = "#B03A2E"
GREY = "#6B7280"

COHORTS = [
    ("id_1d", "1 digit", ID_RAMP[0], False),
    ("id_2d", "2 digits", ID_RAMP[1], False),
    ("id_3d", "3 digits", ID_RAMP[2], False),
    ("id_4d", "4 digits", ID_RAMP[3], False),
    ("id_5d", "5 digits", ID_RAMP[4], False),
    ("ood_6d", "6 digits (unseen)", OOD_COLOR, True),
    ("ood_7d", "7 digits (unseen)", "#7A1F14", True),
]


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
    rows = rows_for("carryless_square")
    missing = [c for c, *_ in COHORTS if c not in rows]
    if missing:
        print(f"missing cohorts {missing}; run exp2_square.py --carryless "
              f"with --positional rope --n_layers 4 --train_digits 1-2-3-4-5")
        return

    figure, ax = plt.subplots(figsize=(9.5, 5))

    for cohort, label, color, is_ood in COHORTS:
        row = rows[cohort]
        values = [float(v) for v in row["per_place_acc"].split("|")]
        ax.plot(range(len(values)), values,
                 marker="o", markersize=5,
                 linestyle="--" if is_ood else "-",
                 linewidth=2 if is_ood else 1.8,
                 color=color,
                 label=f"{label}  (exact={float(row['exact_acc']):.2f})")

    ax.set_xlabel("place value  (0 = ones digit)")
    ax.set_ylabel("digit-wise accuracy at this place")
    ax.set_ylim(-0.04, 1.06)
    ax.set_xlim(-0.4, 13.2)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="center left",
              bbox_to_anchor=(1.01, 0.5))
    ax.set_title(
        "Carryless multiplication: digit-wise accuracy by place",
        fontsize=12.5, loc="left",
    )
    ax.text(0.99, 0.97, "solid = trained lengths (1-5d)\ndashed = unseen lengths, extrapolation (6-7d)",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color=GREY)

    out = RESULTS / "plots" / "carryless_digitwise.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}\n")

    for cohort, label, *_ in COHORTS:
        row = rows[cohort]
        print(f"{label:>20}  exact={float(row['exact_acc']):.4f}  "
              f"digit={float(row['digit_acc']):.4f}  "
              f"per_place={row['per_place_acc']}")


if __name__ == "__main__":
    main()
