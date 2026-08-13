"""Experiment 0: characterize the squaring-mod task with pure number theory.

No model, CPU only, minutes of runtime. For each competition-like regime it
samples (x, N) pairs the way the repo generator does (semiprime N = p*q,
x a unit mod N) and reports:

  - the multiplicative order d = ord_N(x), its 2-adic valuation s (pre-period)
    and the cycle period P = ord_{d'}(2) of T -> x^(2^T) mod N (d' = odd part
    of d),
  - how those periods compare to the trained T ranges and the scored ladder
    T = 1,2,4,8,16,32,64,
  - the fraction of ladder T answerable by cycle detection (T >= s, and the
    stronger T >= s + P where the value literally repeats an earlier rung),
  - gcd(x, N) behavior (the generator only emits gcd = 1; we report the
    would-be rate of gcd > 1 under uniform x for context),
  - output-digit marginals and the honest chance baselines (best constant
    predictor per digit slot, and modal-answer-string frequency for exact
    match), plus the number of distinct output values seen.

Writes results/exp0_task_analysis.json and prints a readable summary. The
verdict paragraph for task_analysis.md gets written after reading the output.

Usage:  python explorations/exp0_characterize.py [--samples 2000] [--seed 45]
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.squaring_mod import _sample_rsa_factors, _sample_unit, is_probable_prime

LADDER: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)
RESULTS_DIR = Path(__file__).resolve().parent / "results"


# --------------------------------------------------------------------------
# Number theory helpers (all inputs here are <= ~24 bits, trial division is fine)
# --------------------------------------------------------------------------

def factorize(value: int) -> dict[int, int]:
    """Prime factorization by trial division."""
    factors: dict[int, int] = {}
    remaining = value
    candidate = 2
    while candidate * candidate <= remaining:
        while remaining % candidate == 0:
            factors[candidate] = factors.get(candidate, 0) + 1
            remaining //= candidate
        candidate += 1 if candidate == 2 else 2
    if remaining > 1:
        factors[remaining] = factors.get(remaining, 0) + 1
    return factors


def carmichael_semiprime(p: int, q: int) -> int:
    return math.lcm(p - 1, q - 1)


def multiplicative_order(base: int, modulus: int, group_exponent: int) -> int:
    """ord_modulus(base) given a multiple of it (the group exponent)."""
    order = group_exponent
    for prime in factorize(group_exponent):
        while order % prime == 0 and pow(base, order // prime, modulus) == 1:
            order //= prime
    return order


def carmichael_general(value: int) -> int:
    """lambda(n) for general odd n (used for ord_{d'}(2); d' is small)."""
    result = 1
    for prime, exponent in factorize(value).items():
        if prime == 2:
            block = 1 if exponent == 1 else (2 if exponent == 2 else 2 ** (exponent - 2))
        else:
            block = (prime - 1) * prime ** (exponent - 1)
        result = math.lcm(result, block)
    return result


def cycle_parameters(x: int, p: int, q: int) -> tuple[int, int, int]:
    """Return (order d, pre-period s, period P) of T -> x^(2^T) mod p*q."""
    modulus = p * q
    d = multiplicative_order(x, modulus, carmichael_semiprime(p, q))
    s = (d & -d).bit_length() - 1  # 2-adic valuation
    d_odd = d >> s
    if d_odd == 1:
        period = 1
    else:
        period = multiplicative_order(2, d_odd, carmichael_general(d_odd))
    return d, s, period


# --------------------------------------------------------------------------
# Regimes mirroring the public tiers (Hard is hidden; Medium is the best proxy)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class AnalysisRegime:
    name: str
    tier: str
    fixed_pq: tuple[int, int] | None  # exact competition moduli where public
    sampled_bits: int | None          # or repo-style sampled semiprime size
    train_T: tuple[int, ...]          # T values the corresponding tier trains on
    role: str                         # "id" or "ood_n" (OOD-N depth-cohort size)


REGIMES: tuple[AnalysisRegime, ...] = (
    AnalysisRegime("e1_fixed_323", "easy", (17, 19), None, (1, 2, 3), "id"),
    AnalysisRegime("e2_fixed_899", "easy", (29, 31), None, (1, 2, 4), "id"),
    AnalysisRegime("easy_sampled_10b", "easy", None, 10, (1, 2, 3), "id"),
    AnalysisRegime("easy_sampled_12b", "easy", None, 12, (2,), "id"),
    AnalysisRegime("easy_ood_n_13b", "easy", None, 13, (2,), "ood_n"),
    AnalysisRegime("m1_fixed_10403", "medium", (101, 103), None, (4, 8, 16), "id"),
    AnalysisRegime("m2_fixed_38021", "medium", (193, 197), None, (4, 8, 16), "id"),
    AnalysisRegime("med_sampled_16b", "medium", None, 16, (2, 4, 8), "id"),
    AnalysisRegime("med_sampled_22b", "medium", None, 22, (8,), "id"),
    AnalysisRegime("med_ood_n_24b", "medium", None, 24, (8,), "ood_n"),
)


# --------------------------------------------------------------------------
# Per-regime analysis
# --------------------------------------------------------------------------

def percentile(sorted_values: list[int], fraction: float) -> int:
    index = min(len(sorted_values) - 1, int(fraction * len(sorted_values)))
    return sorted_values[index]


def summarize_distribution(values: list[int]) -> dict:
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "min": ordered[0],
        "median": percentile(ordered, 0.50),
        "p90": percentile(ordered, 0.90),
        "p99": percentile(ordered, 0.99),
        "max": ordered[-1],
    }


def analyze_regime(regime: AnalysisRegime, samples: int, rng: random.Random) -> dict:
    orders: list[int] = []
    pre_periods: list[int] = []
    periods: list[int] = []
    ladder_in_cycle = 0          # ladder T with T >= s
    ladder_repeats_rung = 0      # ladder T with T >= s + P (value repeats an earlier rung)
    pairs_cycle_visible = 0      # (x, N) with s + P <= max ladder T
    pairs_period_le_train = 0    # (x, N) with s + P <= max(train_T): cycle fully
    #                              observable inside the *training* T range
    gcd_gt_one_uniform = 0       # uniform x in [1, N): how often gcd(x, N) > 1
    outputs: Counter[str] = Counter()
    digit_position_counts: list[Counter[str]] = []  # index 0 = least significant
    distinct_moduli: set[int] = set()

    max_ladder_t = max(LADDER)
    max_train_t = max(regime.train_T)

    # For fixed small moduli, enumerate every unit exhaustively instead of sampling.
    if regime.fixed_pq is not None and regime.fixed_pq[0] * regime.fixed_pq[1] < 5000:
        p, q = regime.fixed_pq
        modulus = p * q
        pair_iter = [(p, q, x) for x in range(1, modulus) if math.gcd(x, modulus) == 1]
    else:
        pair_iter = []
        for _ in range(samples):
            if regime.fixed_pq is not None:
                p, q = regime.fixed_pq
            else:
                p, q = _sample_rsa_factors(modulus_bits=regime.sampled_bits, rng=rng)
            pair_iter.append((p, q, _sample_unit(modulus=p * q, rng=rng)))

    for p, q, x in pair_iter:
        modulus = p * q
        distinct_moduli.add(modulus)
        d, s, period = cycle_parameters(x, p, q)
        orders.append(d)
        pre_periods.append(s)
        periods.append(period)

        for t in LADDER:
            if t >= s:
                ladder_in_cycle += 1
            if t >= s + period:
                ladder_repeats_rung += 1
        if s + period <= max_ladder_t:
            pairs_cycle_visible += 1
        if s + period <= max_train_t:
            pairs_period_le_train += 1

        # gcd under uniform x (what the generator filters away)
        if math.gcd(rng.randrange(1, modulus), modulus) > 1:
            gcd_gt_one_uniform += 1

        # Output digit statistics across the ladder (the strings a model must emit)
        for t in LADDER:
            result = str(pow(x, pow(2, t, carmichael_semiprime(p, q)), modulus))
            outputs[result] += 1
            for position, char in enumerate(reversed(result)):
                while len(digit_position_counts) <= position:
                    digit_position_counts.append(Counter())
                digit_position_counts[position][char] += 1

    total_pairs = len(pair_iter)
    total_ladder_cases = total_pairs * len(LADDER)
    total_outputs = sum(outputs.values())

    # Chance baselines. Per-digit: best constant token per slot, averaged over
    # slots weighted by slot occupancy. Exact: the single most common answer
    # string (a constant predictor's best possible exact-match rate).
    slot_hits = sum(counter.most_common(1)[0][1] for counter in digit_position_counts)
    slot_total = sum(sum(counter.values()) for counter in digit_position_counts)
    modal_string, modal_count = outputs.most_common(1)[0]

    return {
        "regime": regime.name,
        "tier": regime.tier,
        "role": regime.role,
        "modulus": (
            regime.fixed_pq[0] * regime.fixed_pq[1] if regime.fixed_pq else None
        ),
        "sampled_bits": regime.sampled_bits,
        "train_T": list(regime.train_T),
        "n_pairs": total_pairs,
        "n_distinct_moduli": len(distinct_moduli),
        "order": summarize_distribution(orders),
        "pre_period_s": summarize_distribution(pre_periods),
        "period": summarize_distribution(periods),
        "frac_pairs_cycle_visible_within_ladder": pairs_cycle_visible / total_pairs,
        "frac_pairs_cycle_within_train_T": pairs_period_le_train / total_pairs,
        "frac_ladder_T_in_cycle": ladder_in_cycle / total_ladder_cases,
        "frac_ladder_T_repeating_earlier_rung": ladder_repeats_rung / total_ladder_cases,
        "frac_gcd_gt_one_uniform_x": gcd_gt_one_uniform / total_pairs,
        "n_distinct_outputs": len(outputs),
        "n_output_samples": total_outputs,
        "chance_per_digit_constant": slot_hits / slot_total,
        "chance_exact_constant": modal_count / total_outputs,
        "modal_output": modal_string,
        "digit_marginals_lsd_first": [
            {digit: count / sum(counter.values()) for digit, count in sorted(counter.items())}
            for counter in digit_position_counts
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=2000,
                        help="(x, N) pairs per sampled regime (fixed small N is exhaustive)")
    parser.add_argument("--seed", type=int, default=45)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    report = {
        "ladder": list(LADDER),
        "samples_per_regime": args.samples,
        "seed": args.seed,
        "regimes": [],
    }
    for regime in REGIMES:
        result = analyze_regime(regime, args.samples, rng)
        report["regimes"].append(result)
        print(f"\n=== {regime.name} ({regime.tier}, {result['n_pairs']} pairs, "
              f"{result['n_distinct_moduli']} moduli) ===")
        print(f"  order d        : median={result['order']['median']} "
              f"p90={result['order']['p90']} max={result['order']['max']}")
        print(f"  pre-period s   : median={result['pre_period_s']['median']} "
              f"p90={result['pre_period_s']['p90']} max={result['pre_period_s']['max']}")
        print(f"  period P       : median={result['period']['median']} "
              f"p90={result['period']['p90']} max={result['period']['max']}")
        print(f"  cycle (s+P) fits inside ladder<=64 : "
              f"{result['frac_pairs_cycle_visible_within_ladder']:.3f} of pairs")
        print(f"  cycle fits inside trained T<={max(regime.train_T)} : "
              f"{result['frac_pairs_cycle_within_train_T']:.3f} of pairs")
        print(f"  ladder T past pre-period (in cycle): "
              f"{result['frac_ladder_T_in_cycle']:.3f}")
        print(f"  ladder T repeating an earlier rung : "
              f"{result['frac_ladder_T_repeating_earlier_rung']:.3f}")
        print(f"  uniform-x gcd>1 rate (excluded by generator): "
              f"{result['frac_gcd_gt_one_uniform_x']:.3f}")
        print(f"  distinct outputs={result['n_distinct_outputs']}  "
              f"chance/digit={result['chance_per_digit_constant']:.4f}  "
              f"chance/exact={result['chance_exact_constant']:.6f}")

    output_path = RESULTS_DIR / "exp0_task_analysis.json"
    output_path.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {output_path}")


if __name__ == "__main__":
    main()
