"""Is the OOD failure a positional bug in disguise?

The ood_6d/ood_7d per-place accuracy curve is not flat noise — it has bumps
(e.g. high near place 0 and places 8-9 for ood_6d). One explanation: the
arithmetic is basically right but the model reads off the answer from the
wrong starting position for lengths it never trained on (no absolute
position anchor for "where does the answer begin"). If that's true, shifting
every prediction by some constant number of places before comparing to the
truth should recover much higher accuracy than shift=0.

For each example: pred[place] vs true[place - shift], compared only over the
places where both sides exist (so the denominator shrinks with |shift|; a
shift that's "cheating" by shrinking the overlap to a lucky handful of
positions would show as huge n-drop, printed alongside the accuracy).

If accuracy is flat across shifts (roughly matching shift=0, chance-level
elsewhere) -> the errors are not a translation, it's genuinely wrong digits.
If one shift clearly dominates -> it's an alignment/indexing bug, not an
arithmetic one.

Usage:
  python experiments/analyze_shift_alignment.py
  python experiments/analyze_shift_alignment.py --cohorts ood_6d ood_7d --plot
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.tokenizer import SQUARE  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
DIGIT_LO = SQUARE.digit_range[0]


def to_place_array(tokens: list[int]) -> list[int]:
    """token index 0 = most significant digit -> array indexed by place, 0=ones."""
    width = len(tokens)
    out = [0] * width
    for index, token in enumerate(tokens):
        out[width - 1 - index] = token - DIGIT_LO
    return out


def shifted_accuracy(examples: list[dict], shift: int) -> tuple[float, int, dict[int, tuple[int, int]]]:
    """pred[place] vs true[place - shift], over the overlap only.

    Returns (overall_acc, n_compared, {place: (correct, total)}).
    """
    correct = 0
    total = 0
    by_place: dict[int, list[int]] = {}
    for example in examples:
        true_tok, pred_tok = example["true_tokens"], example["predicted_tokens"]
        if len(true_tok) != len(pred_tok):
            continue
        width = len(true_tok)
        true_p = to_place_array(true_tok)
        pred_p = to_place_array(pred_tok)
        for place in range(width):
            src = place - shift
            if not (0 <= src < width):
                continue
            slot = by_place.setdefault(place, [0, 0])
            slot[1] += 1
            total += 1
            if pred_p[place] == true_p[src]:
                slot[0] += 1
                correct += 1
    acc = correct / total if total else 0.0
    per_place = {p: (c, n) for p, (c, n) in by_place.items()}
    return acc, total, per_place


def report(label: str, cohort: dict, shifts: range) -> list[dict]:
    examples = cohort.get("examples", [])
    if not examples:
        return []
    baseline = float(cohort.get("n_total", len(examples)) - cohort.get("n_wrong", 0)) / max(
        cohort.get("n_total", len(examples)), 1
    )
    print(f"\n=== {label}   ({cohort['n_wrong']}/{cohort['n_total']} wrong, "
          f"exact_match={baseline:.4f}) ===")
    print(f"{'shift':>6}{'acc (overlap only)':>22}{'n compared':>13}   "
          f"per-place (place 0 = ones)")
    rows = []
    for shift in shifts:
        acc, n, per_place = shifted_accuracy(examples, shift)
        places = sorted(per_place)
        curve = " ".join(f"{per_place[p][0] / per_place[p][1]:.2f}" for p in places)
        marker = "  <-- shift=0 (no correction)" if shift == 0 else ""
        print(f"{shift:>+6d}{acc:>22.4f}{n:>13}   {curve}{marker}")
        rows.append({"label": label, "shift": shift, "acc": acc, "n": n,
                     "per_place": {p: per_place[p][0] / per_place[p][1] for p in places}})
    best = max(rows, key=lambda r: r["acc"])
    flat = max(r["acc"] for r in rows) - min(r["acc"] for r in rows) < 0.03
    verdict = ("no shift stands out -> not a simple translation" if flat else
               f"shift={best['shift']:+d} is best ({best['acc']:.4f} vs "
               f"{next(r['acc'] for r in rows if r['shift'] == 0):.4f} at shift=0)")
    print(f"    verdict: {verdict}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glob", default="exp2_errors_*.json")
    parser.add_argument("--cohorts", nargs="+", default=["ood_6d", "ood_7d"])
    parser.add_argument("--shifts", type=int, default=3,
                        help="test shift in [-shifts, +shifts]")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    files = sorted(glob.glob(str(RESULTS / args.glob)))
    if not files:
        print(f"no dumps matching {args.glob} in {RESULTS}")
        return

    shifts = range(-args.shifts, args.shifts + 1)
    all_rows: list[dict] = []
    for path in files:
        name = re.sub(r"^exp2_errors_|_s\d+\.json$", "", Path(path).name)
        blob = json.loads(Path(path).read_text())
        for cohort_name in args.cohorts:
            if cohort_name in blob:
                all_rows += report(f"{name} / {cohort_name}", blob[cohort_name], shifts)

    if args.plot and all_rows:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels = sorted({r["label"] for r in all_rows})
        figure, axes = plt.subplots(1, len(labels), figsize=(6 * len(labels), 4.2),
                                    squeeze=False)
        RAMP = ["#9ec5f4", "#6da7ec", "#3987e5", "#1c5cab", "#104281",
                "#0d366b", "#08213f"]
        for axis, label in zip(axes[0], labels):
            rows = sorted([r for r in all_rows if r["label"] == label],
                          key=lambda r: r["shift"])
            xs = [r["shift"] for r in rows]
            ys = [r["acc"] for r in rows]
            axis.plot(xs, ys, marker="o", color="#104281", linewidth=2)
            zero = next(r for r in rows if r["shift"] == 0)
            axis.scatter([0], [zero["acc"]], color="#B03A2E", zorder=5, s=60,
                        label=f"shift=0: {zero['acc']:.3f}")
            best = max(rows, key=lambda r: r["acc"])
            if best["shift"] != 0:
                axis.scatter([best["shift"]], [best["acc"]], color="#1E6F5C",
                            zorder=5, s=60, label=f"best shift={best['shift']:+d}: {best['acc']:.3f}")
            axis.set_xlabel("shift (places); positive = predicted answer read too far left")
            axis.set_ylabel("digit accuracy, overlap only")
            axis.set_ylim(-0.02, 1.02)
            axis.grid(alpha=0.25, linewidth=0.6)
            axis.set_axisbelow(True)
            axis.set_title(label, fontsize=10, loc="left")
            axis.legend(frameon=False, fontsize=8)
        figure.suptitle("Does shifting predicted digits realign them with the truth?",
                        fontsize=12.5, y=1.03)
        figure.tight_layout()
        out = RESULTS / "plots" / "shift_alignment.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
