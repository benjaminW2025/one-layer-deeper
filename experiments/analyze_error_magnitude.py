"""When a digit is wrong, HOW wrong is it?

Reads the error dumps written by exp2_square.py and, for every place value,
histograms (predicted - true) mod 10 over the digits that were wrong.

The question this answers:

  concentrated at +/-1, +/-2  ->  the partial products and their sum are
                                  essentially right and only the carry is off.
                                  The model is computing, just imprecisely.
  uniform over 1..9           ->  nothing is being computed at that place;
                                  it is guessing.

Those are completely different failure modes and imply different fixes.

A uniform-guessing baseline is 1/9 = 0.111 for each non-zero delta, so any bar
meaningfully above that is signal. `near_rate` = P(delta in {+-1, +-2}), which
is 4/9 = 0.444 under uniform guessing.

Usage:
  python experiments/analyze_error_magnitude.py
  python experiments/analyze_error_magnitude.py --cohorts id_4d id_5d --plot
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

RESULTS = Path(__file__).resolve().parent / "results"
UNIFORM = 1 / 9          # per non-zero delta, if guessing
UNIFORM_NEAR = 4 / 9     # P(delta in {-2,-1,1,2}) if guessing


def deltas_by_place(cohort: dict) -> dict[int, Counter]:
    """place -> Counter of (predicted - true) mod 10 over WRONG digits only.
    predicted/true token ids share whatever offset their tokenizer uses, so
    the difference is offset-invariant -- no need to know which tokenizer a
    given dump came from (exp2's SQUARE, or exp5's SQUARE/CARRY_ONLY mix).

    Note for exp5's raw_sum: each semantic place is WIDTH=3 raw token
    columns (a zero-padded chunk), so "place" here is a token column, finer
    grained than the real per-place-sum unit -- still informative (are the
    hundreds/tens/ones digits of the chunk wrong the same way?), just not
    1:1 with the other variants' place numbering.
    """
    out: dict[int, Counter] = {}
    for example in cohort.get("examples", []):
        true = example["true_tokens"]
        pred = example["predicted_tokens"]
        if len(true) != len(pred):
            continue
        width = len(true)
        for index, (t, p) in enumerate(zip(true, pred)):
            if t == p:
                continue
            place = width - 1 - index
            out.setdefault(place, Counter())[(p - t) % 10] += 1
    return out


def bar(fraction: float, width: int = 28) -> str:
    filled = int(round(fraction * width))
    return "#" * filled + "." * (width - filled)


def report(label: str, cohort: dict, min_errors: int) -> list[dict]:
    by_place = deltas_by_place(cohort)
    if not by_place:
        return []
    print(f"\n=== {label}   ({cohort['n_wrong']}/{cohort['n_total']} examples wrong) ===")
    print(f"    delta = (predicted - true) mod 10, over wrong digits only")
    print(f"    uniform guessing would give {UNIFORM:.3f} per delta, "
          f"near_rate {UNIFORM_NEAR:.3f}")
    rows = []
    for place in sorted(by_place, reverse=True):
        counts = by_place[place]
        total = sum(counts.values())
        if total < min_errors:
            continue
        near = sum(counts[d] for d in (1, 2, 8, 9)) / total
        verdict = ("COMPUTING (carry-ish)" if near > 0.60 else
                   "guessing" if near < 0.50 else "mixed")
        print(f"  place {place:>2}  n={total:>6}  near_rate={near:.3f}  {verdict}")
        for delta in (1, 2, 3, 4, 5, 6, 7, 8, 9):
            share = counts[delta] / total
            signed = delta if delta <= 5 else delta - 10
            mark = " *" if delta in (1, 2, 8, 9) else "  "
            print(f"        {signed:+2d}{mark} {share:5.3f} {bar(share)}")
        rows.append({"label": label, "place": place, "n": total, "near_rate": near,
                     "dist": {d: counts[d] / total for d in range(1, 10)}})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glob", default="exp2_errors_*.json")
    parser.add_argument("--cohorts", nargs="+", default=["id_4d", "id_5d"])
    parser.add_argument("--min-errors", type=int, default=30,
                        help="skip places with fewer wrong digits than this")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    files = sorted(glob.glob(str(RESULTS / args.glob)))
    if not files:
        print(f"no dumps matching {args.glob} in {RESULTS}")
        return

    all_rows = []
    for path in files:
        name = re.sub(r"^exp[25]_errors_|_s\d+\.json$", "", Path(path).name)
        blob = json.loads(Path(path).read_text())
        for cohort_name in args.cohorts:
            if cohort_name in blob:
                all_rows += report(f"{name} / {cohort_name}", blob[cohort_name],
                                   args.min_errors)

    if all_rows:
        print("\n=== summary: near_rate = P(off by +-1 or +-2 | digit wrong) ===")
        print(f"{'model / cohort':38}{'place':>6}{'n':>8}{'near_rate':>11}")
        for r in sorted(all_rows, key=lambda r: (r["label"], -r["place"])):
            flag = "  <- computing" if r["near_rate"] > 0.60 else ""
            print(f"{r['label']:38}{r['place']:>6}{r['n']:>8}{r['near_rate']:>11.3f}{flag}")

    if args.plot and all_rows:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        labels = sorted({r["label"] for r in all_rows})
        figure, axes = plt.subplots(len(labels), 1, figsize=(9, 3 * len(labels)),
                                    squeeze=False)
        for axis, label in zip(axes[:, 0], labels):
            for r in sorted([x for x in all_rows if x["label"] == label],
                            key=lambda r: -r["place"]):
                xs = [d if d <= 5 else d - 10 for d in range(1, 10)]
                order = sorted(range(9), key=lambda i: xs[i])
                axis.plot([xs[i] for i in order],
                          [r["dist"][i + 1] for i in order],
                          marker="o", label=f"place {r['place']} (n={r['n']})")
            axis.axhline(UNIFORM, color="grey", ls="--", lw=1, label="uniform guess")
            axis.set_title(label)
            axis.set_xlabel("(predicted - true) mod 10, signed")
            axis.set_ylabel("share of wrong digits")
            axis.legend(fontsize=7)
        figure.tight_layout()
        out = RESULTS / "plots" / "error_magnitude.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(out, dpi=150)
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
