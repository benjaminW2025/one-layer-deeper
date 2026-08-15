"""Datasets for the four experiments.

Each task has its own tokenizer (see tokenizer.py) and its own held-out axis,
because "OOD" means something different in each:

    addition   [X] x [Y] y     x + y        OOD = operand DIGIT LENGTH
    square     [X] x [Y] x     x * x        OOD = unseen x, then digit length
    reduce     [N] N [Y] y     y mod N      OOD = unseen y / unseen N / both
    squaremod  [N] N [X] x     x*x mod N    OOD = unseen x at fixed N

Square repeats x in both operand slots: x*x genuinely has two operands, the
prompt has to be long enough to read a 2d-digit answer off a d-digit input,
and it makes square structurally identical to addition so the only difference
between exp1 and exp2 is the operator.

y in `reduce` is drawn from [0, N^2) - exactly what squaring produces - so the
reduction is never the identity (if y < N then y mod N = y is a copy).
"""

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .tokenizer import ADDITION, CARRY_ONLY, REDUCE, SQUARE, SQUAREMOD, DigitTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass
class Cohort:
    name: str
    records: list[dict[str, Any]]
    note: str = ""


@dataclass
class Dataset:
    task: str
    tokenizer: DigitTokenizer
    train: list[dict[str, Any]]
    cohorts: list[Cohort]
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def max_seq_len(self) -> int:
        groups = [self.train] + [c.records for c in self.cohorts]
        return max(len(r["input_ids"]) for g in groups for r in g)


def _range_with_digits(n_digits: int) -> tuple[int, int]:
    return (1, 10) if n_digits == 1 else (10 ** (n_digits - 1), 10 ** n_digits)


def _split_budget(available: int, wanted_train: int, wanted_eval: int) -> tuple[int, int]:
    """Clamp requested counts to what actually exists at this digit length.

    Short lengths are tiny: 1-digit addition has only 9*9 = 81 distinct pairs,
    1-digit squaring only 9 distinct x. Asking for more unique draws than exist
    would loop forever, so cap instead, and keep train and eval disjoint.
    """
    train = min(wanted_train, max(1, available // 2))
    evaluation = min(wanted_eval, max(0, available - train))
    return train, evaluation


# ---------------------------------------------------------------------------
# 1. addition — OOD axis is digit length
# ---------------------------------------------------------------------------

def build_addition(
    *,
    train_digits: tuple[int, ...] = (1, 2, 3, 4),
    ood_digits: tuple[int, ...] = (5, 6, 7),
    n_train: int = 100_000,
    n_eval: int = 2_000,
    seed: int = 45,
) -> Dataset:
    rng = random.Random(seed)
    tokenizer = ADDITION
    used: set[tuple[int, int]] = set()

    def space(n_digits: int) -> int:
        low, high = _range_with_digits(n_digits)
        return (high - low) ** 2

    def sample(n_digits: int, count: int, reserve: bool) -> list[dict[str, Any]]:
        low, high = _range_with_digits(n_digits)
        out: list[dict[str, Any]] = []
        attempts = 0
        while len(out) < count:
            attempts += 1
            if attempts > 200 * max(count, 1) + 10_000:
                raise ValueError(
                    f"addition: exhausted unique {n_digits}-digit pairs after "
                    f"{len(out)}/{count}; only {space(n_digits)} exist"
                )
            x, y = rng.randrange(low, high), rng.randrange(low, high)
            if (x, y) in used:
                continue
            if reserve:
                used.add((x, y))
            out.append(tokenizer.record([("X", x), ("Y", y)], x + y))
        return out

    wanted = max(1, n_train // len(train_digits))
    train: list[dict[str, Any]] = []
    cohorts: list[Cohort] = []
    budget: dict[int, tuple[int, int]] = {}
    for n_digits in train_digits:
        n_tr, n_ev = _split_budget(space(n_digits), wanted, n_eval)
        budget[n_digits] = (n_tr, n_ev)
        train.extend(sample(n_digits, n_tr, reserve=True))
    rng.shuffle(train)

    for n_digits in train_digits:
        cohorts.append(Cohort(
            f"id_{n_digits}d", sample(n_digits, budget[n_digits][1], reserve=True),
            "seen length, unseen values",
        ))
    for n_digits in ood_digits:
        cohorts.append(Cohort(
            f"ood_{n_digits}d", sample(n_digits, n_eval, reserve=False),
            "unseen length",
        ))
    return Dataset("addition", tokenizer, train, cohorts, {
        "train_digits": list(train_digits), "ood_digits": list(ood_digits),
        "space_per_length": {d: space(d) for d in train_digits},
        "train_per_length": {d: budget[d][0] for d in train_digits},
    })


# ---------------------------------------------------------------------------
# 2. square — same format as addition, different operator
# ---------------------------------------------------------------------------

def carryless_square(x: int) -> int:
    """Digit-wise convolution of x with itself, each place taken mod 10.

    Identical to x*x except carries never propagate: place k is
    (sum_{i+j=k} x_i*x_j) mod 10 and depends on NOTHING else. So the fan-in per
    place is unchanged while the sequential carry chain is removed entirely --
    the ablation that separates "cannot do the convolution" from "cannot
    propagate the carry".

    The leading place is x_{d-1}^2 mod 10, and no nonzero digit squares to 0
    mod 10, so the result always has exactly 2d-1 digits and never loses a
    leading zero to str().
    """
    digits = [int(c) for c in str(x)][::-1]        # least significant first
    width = 2 * len(digits) - 1
    out = [0] * width
    for i, a in enumerate(digits):
        for j, b in enumerate(digits):
            out[i + j] = (out[i + j] + a * b) % 10
    return int("".join(str(v) for v in reversed(out)))


def build_square(
    *,
    train_digits: tuple[int, ...] = (1, 2, 3, 4),
    ood_digits: tuple[int, ...] = (5, 6),
    n_train: int = 100_000,
    n_eval: int = 2_000,
    seed: int = 45,
    carryless: bool = False,
) -> Dataset:
    rng = random.Random(seed)
    tokenizer = SQUARE
    operation = carryless_square if carryless else (lambda v: v * v)
    used: set[int] = set()

    def space(n_digits: int) -> int:
        low, high = _range_with_digits(n_digits)
        return high - low

    def sample(n_digits: int, count: int, reserve: bool) -> list[dict[str, Any]]:
        low, high = _range_with_digits(n_digits)
        out: list[dict[str, Any]] = []
        attempts = 0
        while len(out) < count:
            attempts += 1
            if attempts > 200 * max(count, 1) + 10_000:
                raise ValueError(
                    f"square: exhausted unique {n_digits}-digit x after "
                    f"{len(out)}/{count}; only {space(n_digits)} exist"
                )
            x = rng.randrange(low, high)
            if x in used:
                continue
            if reserve:
                used.add(x)
            out.append(tokenizer.record([("X", x), ("Y", x)], operation(x)))
        return out

    wanted = max(1, n_train // len(train_digits))
    train: list[dict[str, Any]] = []
    budget: dict[int, tuple[int, int]] = {}
    for n_digits in train_digits:
        n_tr, n_ev = _split_budget(space(n_digits), wanted, n_eval)
        budget[n_digits] = (n_tr, n_ev)
        train.extend(sample(n_digits, n_tr, reserve=True))
    rng.shuffle(train)

    # Held-in cohort: x drawn from the training pool itself. A memorizing model
    # scores ~1.0 here and ~0 on the id_* cohorts, so the gap is measured
    # directly instead of inferred from the training loss.
    seen_sample = rng.sample(train, min(n_eval, len(train)))
    cohorts = [Cohort("train_seen", seen_sample, "x from the training pool")] + [
        Cohort(f"id_{d}d", sample(d, budget[d][1], reserve=True), "seen length, unseen x")
        for d in train_digits
    ] + [
        Cohort(f"ood_{d}d", sample(d, n_eval, reserve=False), "unseen length")
        for d in ood_digits
    ]
    return Dataset("carryless_square" if carryless else "square",
                   tokenizer, train, cohorts, {
        "carryless": carryless,
        "train_digits": list(train_digits), "ood_digits": list(ood_digits),
        "space_per_length": {d: space(d) for d in train_digits},
        "train_per_length": {d: budget[d][0] for d in train_digits},
        # x*x has only `space` distinct inputs per length, so short lengths are
        # trivially memorizable. Check this before reading exp2's id_* scores.
        "x_coverage_per_length": {
            d: round(budget[d][0] / space(d), 4) for d in train_digits
        },
    })


# ---------------------------------------------------------------------------
# 2b. carry-mechanism ablations — split "compute the convolution" from
# "propagate the carry" into two separate tasks, so a model that fails at
# full x*x can be pointed at whichever half is actually broken.
#
#   raw_sum     x -> the UN-reduced per-place convolution sums (no mod 10,
#               no carries -- literally sum_{i+j=k} x_i*x_j, which can
#               exceed 9). Tests the convolution/pairing step in isolation:
#               can it gather the right digit pairs and multiply-sum them,
#               without also needing to fold the result back into 0-9?
#   carry_only  the raw per-place sums (same numbers raw_sum outputs) ->
#               the correct carry-resolved decimal digits of x*x. Tests
#               carry propagation in isolation: GIVEN the convolution is
#               already done, can it do the sequential add-and-carry?
#
# Each place's raw sum can exceed one digit (up to ~n*81 for n-digit x, e.g.
# 648 at n=8), so it's encoded as a FIXED-WIDTH zero-padded chunk of digits
# (`digits_padded`, not `digits` -- padding must survive concatenation, or
# place boundaries would be ambiguous / a leading zero would silently vanish
# into the previous chunk). WIDTH=3 covers every place sum up to n=12.
# ---------------------------------------------------------------------------

WIDTH = 3


def raw_place_sums(x: int) -> list[int]:
    """place k (0 = ones) = sum_{i+j=k} x_i*x_j, UNREDUCED -- no mod 10, no
    carries. Same pairing carryless_square uses, just without folding each
    place back into a single digit. Returned LSD-first (index 0 = place 0)."""
    digits = [int(c) for c in str(x)][::-1]  # LSD first
    width = 2 * len(digits) - 1
    out = [0] * width
    for i, a in enumerate(digits):
        for j, b in enumerate(digits):
            out[i + j] += a * b
    return out


def _padded_sum_field(tokenizer: DigitTokenizer, x: int, width: int = WIDTH) -> list[int]:
    """The raw per-place sums of x, MSD place first (matching every other
    field's order), each place zero-padded to `width` digits and
    concatenated -- one flat digit-id list, chunk boundaries implicit in
    position (every `width` tokens is one place)."""
    sums = raw_place_sums(x)[::-1]  # MSD place first
    out: list[int] = []
    for s in sums:
        out.extend(tokenizer.digits_padded(s, width))
    return out


def _square_fields_padded(x: int, answer_len: int) -> list[tuple[str, int]]:
    """[("X",x),("Y",x)] plus as many redundant ("X",x) copies as needed so
    the encoded prompt is at least `answer_len` tokens. raw_sum's answer
    (width*(2n-1) tokens) is longer than the bare 2-field prompt (2n+2) once
    width>1, and the tokenizer's "read the answer off the prompt's own
    tail" convention requires prompt_len >= answer_len. The copies carry no
    new information (x is already fully present in X and Y) -- they exist
    purely to give the answer somewhere to be read from."""
    base = [("X", x), ("Y", x)]
    base_len = len(SQUARE.encode(base))
    field_cost = 1 + len(str(x))  # marker + x's own digits
    extra = max(0, -(-(answer_len - base_len) // field_cost)) if field_cost else 0
    return base + [("X", x)] * extra


def build_raw_sum(
    *,
    train_digits: tuple[int, ...] = (1, 2, 3, 4),
    ood_digits: tuple[int, ...] = (5, 6),
    n_train: int = 100_000,
    n_eval: int = 2_000,
    seed: int = 45,
    width: int = WIDTH,
) -> Dataset:
    """x -> raw (un-reduced) per-place convolution sums. Reuses SQUARE's
    input format (X, Y both = x) since only the OUTPUT encoding changes."""
    rng = random.Random(seed)
    tokenizer = SQUARE
    used: set[int] = set()

    def space(n_digits: int) -> int:
        low, high = _range_with_digits(n_digits)
        return high - low

    def sample(n_digits: int, count: int, reserve: bool) -> list[dict[str, Any]]:
        low, high = _range_with_digits(n_digits)
        out: list[dict[str, Any]] = []
        attempts = 0
        while len(out) < count:
            attempts += 1
            if attempts > 200 * max(count, 1) + 10_000:
                raise ValueError(
                    f"raw_sum: exhausted unique {n_digits}-digit x after "
                    f"{len(out)}/{count}; only {space(n_digits)} exist"
                )
            x = rng.randrange(low, high)
            if x in used:
                continue
            if reserve:
                used.add(x)
            labels = _padded_sum_field(tokenizer, x, width)
            input_ids = tokenizer.encode(_square_fields_padded(x, len(labels)))
            out.append({"input_ids": input_ids, "labels": labels, "answer": x})
        return out

    wanted = max(1, n_train // len(train_digits))
    train: list[dict[str, Any]] = []
    budget: dict[int, tuple[int, int]] = {}
    for n_digits in train_digits:
        n_tr, n_ev = _split_budget(space(n_digits), wanted, n_eval)
        budget[n_digits] = (n_tr, n_ev)
        train.extend(sample(n_digits, n_tr, reserve=True))
    rng.shuffle(train)

    seen_sample = rng.sample(train, min(n_eval, len(train)))
    cohorts = [Cohort("train_seen", seen_sample, "x from the training pool")] + [
        Cohort(f"id_{d}d", sample(d, budget[d][1], reserve=True), "seen length, unseen x")
        for d in train_digits
    ] + [
        Cohort(f"ood_{d}d", sample(d, n_eval, reserve=False), "unseen length")
        for d in ood_digits
    ]
    return Dataset("raw_sum", tokenizer, train, cohorts, {
        "train_digits": list(train_digits), "ood_digits": list(ood_digits), "width": width,
    })


def build_carry_only(
    *,
    train_digits: tuple[int, ...] = (1, 2, 3, 4),
    ood_digits: tuple[int, ...] = (5, 6),
    n_train: int = 100_000,
    n_eval: int = 2_000,
    seed: int = 45,
    width: int = WIDTH,
) -> Dataset:
    """The raw per-place sums (same numbers raw_sum outputs) -> the correct,
    carry-resolved decimal digits of x*x. Single-field ("S") input; the
    pairing/multiplication is already done, only carry propagation remains."""
    rng = random.Random(seed)
    tokenizer = CARRY_ONLY
    used: set[int] = set()

    def space(n_digits: int) -> int:
        low, high = _range_with_digits(n_digits)
        return high - low

    def sample(n_digits: int, count: int, reserve: bool) -> list[dict[str, Any]]:
        low, high = _range_with_digits(n_digits)
        out: list[dict[str, Any]] = []
        attempts = 0
        while len(out) < count:
            attempts += 1
            if attempts > 200 * max(count, 1) + 10_000:
                raise ValueError(
                    f"carry_only: exhausted unique {n_digits}-digit x after "
                    f"{len(out)}/{count}; only {space(n_digits)} exist"
                )
            x = rng.randrange(low, high)
            if x in used:
                continue
            if reserve:
                used.add(x)
            input_ids = [tokenizer.marker_id("S")] + _padded_sum_field(tokenizer, x, width)
            labels = tokenizer.digits(x * x)
            out.append({"input_ids": input_ids, "labels": labels, "answer": x})
        return out

    wanted = max(1, n_train // len(train_digits))
    train: list[dict[str, Any]] = []
    budget: dict[int, tuple[int, int]] = {}
    for n_digits in train_digits:
        n_tr, n_ev = _split_budget(space(n_digits), wanted, n_eval)
        budget[n_digits] = (n_tr, n_ev)
        train.extend(sample(n_digits, n_tr, reserve=True))
    rng.shuffle(train)

    seen_sample = rng.sample(train, min(n_eval, len(train)))
    cohorts = [Cohort("train_seen", seen_sample, "x from the training pool")] + [
        Cohort(f"id_{d}d", sample(d, budget[d][1], reserve=True), "seen length, unseen x")
        for d in train_digits
    ] + [
        Cohort(f"ood_{d}d", sample(d, n_eval, reserve=False), "unseen length")
        for d in ood_digits
    ]
    return Dataset("carry_only", tokenizer, train, cohorts, {
        "train_digits": list(train_digits), "ood_digits": list(ood_digits), "width": width,
    })


# ---------------------------------------------------------------------------
# 3. reduce — y mod N, three variants
# ---------------------------------------------------------------------------

def _semiprime_pool(bits: int, count: int, rng: random.Random) -> list[int]:
    from data.squaring_mod import _sample_rsa_factors

    pool: list[int] = []
    seen: set[int] = set()
    for _ in range(50_000):
        p, q = _sample_rsa_factors(modulus_bits=bits, rng=rng)
        if p * q not in seen:
            seen.add(p * q)
            pool.append(p * q)
        if len(pool) >= count:
            break
    if len(pool) < count:
        raise ValueError(
            f"only {len(pool)} distinct {bits}-bit semiprimes exist; "
            "lower --n-moduli or raise --bits"
        )
    return pool


def build_reduce(
    *,
    variant: str = "fixed_n",
    bits: int = 14,
    n_moduli: int = 32,
    n_train: int = 100_000,
    n_eval: int = 2_000,
    seed: int = 45,
) -> Dataset:
    if variant not in ("fixed_n", "fixed_y", "vary_both"):
        raise ValueError("variant must be fixed_n, fixed_y or vary_both")
    rng = random.Random(seed)
    tokenizer = REDUCE
    pool = _semiprime_pool(bits, 1 if variant == "fixed_n" else n_moduli, rng)

    if variant == "fixed_n":
        train_moduli, ood_moduli = pool, []
    else:
        split = max(1, int(0.8 * len(pool)))
        train_moduli, ood_moduli = pool[:split], pool[split:]

    # fixed_y: one value reduced by many moduli. y must exceed every modulus.
    held_y = rng.randrange(max(pool), max(pool) ** 2) if variant == "fixed_y" else None
    used: set[tuple[int, int]] = set()

    def sample(moduli: list[int], count: int, reserve: bool) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        guard = 0
        while len(out) < count:
            guard += 1
            if guard > 500 * count:
                raise ValueError(f"exhausted unique (N, y) pairs for {variant}")
            modulus = rng.choice(moduli)
            y = held_y if variant == "fixed_y" else rng.randrange(0, modulus * modulus)
            if (modulus, y) in used:
                continue
            if reserve:
                used.add((modulus, y))
            out.append(tokenizer.record([("N", modulus), ("Y", y)], y % modulus))
        return out

    train = sample(train_moduli, n_train, reserve=True)
    cohorts = [Cohort("id_fresh", sample(train_moduli, n_eval, True), "seen N, unseen y")]
    if ood_moduli:
        cohorts.append(Cohort("ood_n", sample(ood_moduli, n_eval, False), "unseen N"))
    return Dataset(f"reduce_{variant}", tokenizer, train, cohorts,
                   {"variant": variant, "bits": bits,
                    "n_train_moduli": len(train_moduli),
                    "n_ood_moduli": len(ood_moduli)})


# ---------------------------------------------------------------------------
# 4. squaremod — the competition's T=1
# ---------------------------------------------------------------------------

def build_squaremod(
    *,
    bits: int = 18,
    n_train: int = 50_000,
    n_eval: int = 2_000,
    seed: int = 45,
) -> Dataset:
    from data.squaring_mod import _sample_rsa_factors

    rng = random.Random(seed)
    tokenizer = SQUAREMOD
    p, q = _sample_rsa_factors(modulus_bits=bits, rng=rng)
    modulus = p * q
    x_space = (p - 1) * (q - 1)
    if n_train + 2 * n_eval > x_space:
        raise ValueError(
            f"bits={bits} gives only {x_space} units; lower --n-train or raise --bits"
        )

    used: set[int] = set()

    def draw_unit() -> int:
        while True:
            value = rng.randrange(1, modulus)
            if value not in used and math.gcd(value, modulus) == 1:
                used.add(value)
                return value

    train_x = [draw_unit() for _ in range(n_train)]
    fresh_x = [draw_unit() for _ in range(n_eval)]
    held_in = rng.sample(train_x, min(n_eval, len(train_x)))

    def records(values: list[int]) -> list[dict[str, Any]]:
        return [
            tokenizer.record([("N", modulus), ("X", x)], (x * x) % modulus)
            for x in values
        ]

    return Dataset(
        "squaremod", tokenizer, records(train_x),
        [
            Cohort("train_seen", records(held_in), "x from the training pool"),
            Cohort("test_fresh", records(fresh_x), "x never trained on"),
        ],
        {"bits": bits, "modulus": modulus, "x_space": x_space,
         "x_coverage": n_train / x_space},
    )
