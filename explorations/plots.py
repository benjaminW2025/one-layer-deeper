"""Analysis + plots from results.csv, exp0_task_analysis.json, exp2_profile.json.

Produces (into results/plots/):
  1. acc_vs_T_<regime>_<mode>.png — exact-match and per-digit accuracy vs T,
     seen-N vs OOD-N, with the chance baseline and the exp0 median cycle
     length (pre-period + period) marked. From exp1 rows.
  2. depth_by_T_<regime>_<mode>.png — heatmap of exact accuracy, depth x
     ladder-T (best width per cell), separately for seen-N and OOD-N. From
     exp3 rows.
  3. compute_breakdown.png — per-phase ms/step stacked bars per width. From
     exp2_profile.json.

Also prints, with raw numbers (feed these into FINDINGS.md):
  - max solvable T per depth (threshold and certified-prefix variants), and
    whether it looks like T~L, T~2^L, or flat (period-tracking)
  - run-to-run variance across exp1's repeated seeds
  - flagged rows (diverged / flat-from-start / at-chance) so they are not
    presented as clean data points

Safe to run with any subset of inputs present; missing files are skipped.

Usage:  python explorations/plots.py [--threshold 0.95]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from harness import CSV_PATH, LADDER, RESULTS_DIR  # noqa: E402

PLOTS_DIR = RESULTS_DIR / "plots"

# harness regime -> closest exp0 regime for the period marker
EXP0_REGIME_MAP = {
    "e1_fixed323": "e1_fixed_323",
    "easy_sampled_b1011": "easy_sampled_10b",
    "m1_fixed10403": "m1_fixed_10403",
    "med_sampled_b121416": "med_sampled_16b",
}


def load_rows() -> list[dict]:
    if not CSV_PATH.exists():
        print(f"no {CSV_PATH}; run experiments first")
        return []
    with CSV_PATH.open() as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ("digit_acc", "exact_acc", "chance_digit", "chance_exact", "budget"):
            row[key] = float(row[key])
        for key in ("n_layers", "d_model", "steps_completed", "seed"):
            row[key] = int(row[key])
        row["T"] = int(row["T"]) if row["T"] not in ("", "None") else None
        row["ood_n"] = row["ood_n"] == "True"
        row["diverged"] = row["diverged"] == "True"
        row["flat_from_start"] = row["flat_from_start"] == "True"
    return rows


def load_exp0_periods() -> dict[str, float]:
    path = RESULTS_DIR / "exp0_task_analysis.json"
    if not path.exists():
        return {}
    report = json.loads(path.read_text())
    return {
        entry["regime"]: entry["pre_period_s"]["median"] + entry["period"]["median"]
        for entry in report["regimes"]
    }


def flag_suspect_rows(rows: list[dict]) -> None:
    print("\n=== flagged rows (do not read as clean architecture results) ===")
    flagged = 0
    for row in rows:
        reasons = []
        if row["diverged"]:
            reasons.append("diverged")
        if row["flat_from_start"]:
            reasons.append("flat-from-start (LR suspect)")
        if row["exact_acc"] <= row["chance_exact"] * 1.5 and row["T"] is not None:
            reasons.append("at/near chance")
        if reasons:
            flagged += 1
            if flagged <= 40:
                print(f"  {row['run_id']} {row['eval_set']}: "
                      f"exact={row['exact_acc']:.3f} chance={row['chance_exact']:.4f} "
                      f"[{', '.join(reasons)}]")
    print(f"  ({flagged} flagged rows total)")


def plot_acc_vs_t(rows: list[dict], periods: dict[str, float]) -> None:
    exp1 = [r for r in rows if r["exp"] == "exp1" and r["T"] is not None and r["seed"] == 0]
    cells = sorted({(r["regime"], r["budget_mode"]) for r in exp1})
    for regime, mode in cells:
        cell = [r for r in exp1 if r["regime"] == regime and r["budget_mode"] == mode]
        figure, axis = plt.subplots(figsize=(8, 5))
        for ood_n, style, label in ((False, "-o", "seen N (fresh x)"),
                                    (True, "--s", "OOD N")):
            series = sorted([r for r in cell if r["ood_n"] == ood_n],
                            key=lambda r: r["T"])
            if not series:
                continue
            axis.plot([r["T"] for r in series], [r["exact_acc"] for r in series],
                      style, label=f"exact, {label}")
            axis.plot([r["T"] for r in series], [r["digit_acc"] for r in series],
                      style, alpha=0.35, label=f"per-digit, {label}")
            axis.axhline(series[0]["chance_exact"], color="gray", linewidth=0.8,
                         linestyle=":" if ood_n else "-.",
                         label=f"chance exact, {label}")
        marker = periods.get(EXP0_REGIME_MAP.get(regime, ""))
        if marker is not None:
            axis.axvline(marker, color="red", linewidth=1,
                         label=f"median cycle s+P ≈ {marker:g} (exp0)")
        axis.set_xscale("log", base=2)
        axis.set_xticks(list(LADDER))
        axis.set_xticklabels([str(t) for t in LADDER])
        axis.set_xlabel("T (composition depth)")
        axis.set_ylabel("accuracy")
        axis.set_ylim(-0.02, 1.02)
        axis.set_title(f"exp1: accuracy vs T — {regime}, budget={mode}")
        axis.legend(fontsize=7)
        figure.tight_layout()
        output = PLOTS_DIR / f"acc_vs_T_{regime}_{mode}.png"
        figure.savefig(output, dpi=150)
        plt.close(figure)
        print(f"wrote {output}")


def max_solvable_t(series: dict[int, float], threshold: float) -> tuple[int, int]:
    """(best rung >= threshold anywhere, certified consecutive prefix rung)."""
    best = 0
    certified = 0
    prefix_intact = True
    for t in LADDER:
        accuracy = series.get(t)
        if accuracy is None:
            prefix_intact = False
            continue
        if accuracy >= threshold:
            best = max(best, t)
            if prefix_intact:
                certified = t
        else:
            prefix_intact = False
    return best, certified


def plot_depth_grid(rows: list[dict], threshold: float) -> None:
    exp3 = [r for r in rows if r["exp"] == "exp3" and r["T"] is not None]
    cells = sorted({(r["regime"], r["budget_mode"]) for r in exp3})
    for regime, mode in cells:
        cell = [r for r in exp3 if r["regime"] == regime and r["budget_mode"] == mode]
        depths = sorted({r["n_layers"] for r in cell})
        for ood_n in (False, True):
            # best exact accuracy across widths per (depth, T)
            surface = defaultdict(dict)
            for r in cell:
                if r["ood_n"] != ood_n:
                    continue
                current = surface[r["n_layers"]].get(r["T"], -1.0)
                surface[r["n_layers"]][r["T"]] = max(current, r["exact_acc"])
            if not surface:
                continue
            grid = [[surface[d].get(t, float("nan")) for t in LADDER] for d in depths]
            figure, axis = plt.subplots(figsize=(9, 4))
            image = axis.imshow(grid, aspect="auto", vmin=0, vmax=1, cmap="viridis")
            axis.set_xticks(range(len(LADDER)), [str(t) for t in LADDER])
            axis.set_yticks(range(len(depths)), [str(d) for d in depths])
            axis.set_xlabel("T")
            axis.set_ylabel("n_layers")
            variant = "ood_n" if ood_n else "seen_n"
            axis.set_title(f"exp3 exact acc (best width) — {regime}, {mode}, {variant}")
            figure.colorbar(image)
            figure.tight_layout()
            output = PLOTS_DIR / f"depth_by_T_{regime}_{mode}_{variant}.png"
            figure.savefig(output, dpi=150)
            plt.close(figure)
            print(f"wrote {output}")

            print(f"\n  max solvable T (exact >= {threshold}) — "
                  f"{regime}, {mode}, {variant}:")
            scaling_pairs = []
            for depth in depths:
                best, certified = max_solvable_t(surface[depth], threshold)
                scaling_pairs.append((depth, certified))
                print(f"    L={depth}: certified-prefix T={certified}, "
                      f"best rung T={best}")
            solved = [(d, t) for d, t in scaling_pairs if t > 0]
            if len(solved) >= 2:
                ratios = [t / d for d, t in solved]
                log_ratios = [math.log2(t) / d for d, t in solved if t >= 1]
                print(f"    T/L values: {[round(v, 2) for v in ratios]} "
                      f"(constant => T~L);  log2(T)/L: "
                      f"{[round(v, 2) for v in log_ratios]} (constant => T~2^L); "
                      f"flat T across L => period-tracking shortcut")


def report_variance(rows: list[dict]) -> None:
    exp1 = [r for r in rows if r["exp"] == "exp1"]
    by_cell = defaultdict(list)
    for r in exp1:
        by_cell[(r["regime"], r["budget_mode"], r["eval_set"])].append(r)
    print("\n=== run-to-run variance (exp1 repeated seeds) ===")
    reported = False
    for (regime, mode, eval_set), cell_rows in sorted(by_cell.items()):
        seeds = {r["seed"] for r in cell_rows}
        if len(seeds) < 2 or eval_set != "id_test":
            continue
        values = [r["exact_acc"] for r in sorted(cell_rows, key=lambda r: r["seed"])]
        print(f"  {regime}/{mode}/{eval_set}: exact per seed {values} "
              f"stdev={statistics.stdev(values):.4f}")
        reported = True
    if not reported:
        print("  (no repeated-seed cells found yet)")


def plot_compute_breakdown() -> None:
    path = RESULTS_DIR / "exp2_profile.json"
    if not path.exists():
        print(f"no {path}; skipping compute breakdown plot")
        return
    profiles = json.loads(path.read_text())
    phases = ["data_h2d", "forward", "backward", "optimizer"]
    labels = [f"d{p['d_model']}" for p in profiles]
    figure, axis = plt.subplots(figsize=(7, 5))
    bottoms = [0.0] * len(profiles)
    for phase in phases:
        values = [p["step_ms"][phase] for p in profiles]
        axis.bar(labels, values, bottom=bottoms, label=phase)
        bottoms = [b + v for b, v in zip(bottoms, values)]
    for index, profile in enumerate(profiles):
        note = (f"bwd/fwd={profile['bwd_fwd_ratio']}\n"
                f"util={profile['forward_utilization_pct']}%")
        axis.text(index, bottoms[index] * 1.02, note, ha="center", fontsize=8)
    axis.set_ylabel("ms per step")
    axis.set_title("exp2: per-step compute allocation")
    axis.legend()
    figure.tight_layout()
    output = PLOTS_DIR / "compute_breakdown.png"
    figure.savefig(output, dpi=150)
    plt.close(figure)
    print(f"wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=0.95,
                        help="exact-match threshold for 'solvable T' "
                        "(competition certification is 1.0; 0.95 is more "
                        "stable at these cohort sizes — both worth checking)")
    args = parser.parse_args()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    if rows:
        periods = load_exp0_periods()
        plot_acc_vs_t(rows, periods)
        plot_depth_grid(rows, args.threshold)
        report_variance(rows)
        flag_suspect_rows(rows)
    plot_compute_breakdown()


if __name__ == "__main__":
    main()
