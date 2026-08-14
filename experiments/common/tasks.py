"""Datasets for the four experiments.

Every task produces records in the competition's prompt format (see format.py)
and every task defines its own held-out axis, because "OOD" means something
different in each:

    addition   OOD = operand DIGIT LENGTH (train short, test longer)
    square     OOD = unseen x, then unseen digit length
    reduce     OOD = unseen x / unseen N / both
    squaremod  OOD = unseen x at fixed N (= competition T=1)

A Cohort is a named set of records plus what makes it out-of-distribution.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

from .format import make_record


@dataclass
class Cohort:
    name: str
    records: list[dict[str, Any]]
    note: str = ""


@dataclass
class Dataset:
    task: str
    train: list[dict[str, Any]]
    cohorts: list[Cohort]
    meta: dict[str, Any] = field(default_factory=dict)


def _digits_in_range(n_digits: int) -> tuple[int, int]:
    """Inclusive-exclusive range of integers with exactly n_digits digits."""
    return (1, 10) if n_digits == 1 else (10 ** (n_digits - 1), 10 ** n_digits)


# ---------------------------------------------------------------------------
# 1. addition:  x + y.  OOD axis = number of digits.
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

    def draw(n_digits: int) -> tuple[int, int]:
        low, high = _digits_in_range(n_digits)
        return rng.randrange(low, high), rng.randrange(low, high)

    seen: set[tuple[int, int]] = set()

    def sample(n_digits: int, count: int, exclude: bool) -> list[dict[str, Any]]:
        out = []
        while len(out) < count:
            x, y = draw(n_digits)
            if (x, y) in seen:
                continue
            if exclude:
                seen.add((x, y))
            out.append(make_record(x, y, 0, x + y))
        return out

    per_length = max(1, n_train // len(train_digits))
    train: list[dict[str, Any]] = []
    for n_digits in train_digits:
        train.extend(sample(n_digits, per_length, exclude=True))
    rng.shuffle(train)

    cohorts = [
        Cohort(f"id_{d}d", sample(d, n_eval, exclude=True), "seen length, unseen values")
        for d in train_digits
    ] + [
        Cohort(f"ood_{d}d", sample(d, n_eval, exclude=False), "unseen length")
        for d in ood_digits
    ]
    return Dataset(
        task="addition",
        train=train,
        cohorts=cohorts,
        meta={"train_digits": list(train_digits), "ood_digits": list(ood_digits)},
    )


# ---------------------------------------------------------------------------
# 2. square:  x -> x*x.  OOD axis = unseen x, then unseen digit length.
# ---------------------------------------------------------------------------

def build_square(
    *,
    train_digits: tuple[int, ...] = (1, 2, 3, 4),
    ood_digits: tuple[int, ...] = (5, 6),
    n_train: int = 100_000,
    n_eval: int = 2_000,
    seed: int = 45,
) -> Dataset:
    rng = random.Random(seed)
    seen: set[int] = set()

    def sample(n_digits: int, count: int, exclude: bool) -> list[dict[str, Any]]:
        low, high = _digits_in_range(n_digits)
        if exclude and count > (high - low) // 2:
            raise ValueError(f"{n_digits}-digit space too small for {count} unique x")
        out = []
        while len(out) < count:
            x = rng.randrange(low, high)
            if x in seen:
                continue
            if exclude:
                seen.add(x)
            # Field a is unused by this task; hold it constant so the prompt
            # shape matches the others without leaking anything.
            out.append(make_record(0, x, 1, x * x))
        return out

    per_length = max(1, n_train // len(train_digits))
    train: list[dict[str, Any]] = []
    for n_digits in train_digits:
        low, high = _digits_in_range(n_digits)
        train.extend(sample(n_digits, min(per_length, (high - low) // 2), exclude=True))
    rng.shuffle(train)

    cohorts = [
        Cohort(f"id_{d}d", sample(d, min(n_eval, 400 if d == 1 else n_eval), exclude=True),
               "seen length, unseen x")
        for d in train_digits
    ] + [
        Cohort(f"ood_{d}d", sample(d, n_eval, exclude=False), "unseen length")
        for d in ood_digits
    ]
    return Dataset(
        task="square",
        train=train,
        cohorts=cohorts,
        meta={"train_digits": list(train_digits), "ood_digits": list(ood_digits)},
    )


# ---------------------------------------------------------------------------
# 3. reduce:  y mod N.  Three variants, because "OOD" can mean x, N, or both.
#    y is drawn from [0, N^2) so the reduction is never the identity.
# ---------------------------------------------------------------------------

def _semiprime_pool(bits: int, count: int, rng: random.Random) -> list[int]:
    from data.squaring_mod import _sample_rsa_factors

    pool: list[int] = []
    seen: set[int] = set()
    for _ in range(20_000):
        p, q = _sample_rsa_factors(modulus_bits=bits, rng=rng)
        if p * q not in seen:
            seen.add(p * q)
            pool.append(p * q)
        if len(pool) >= count:
            break
    if not pool:
        raise ValueError(f"no {bits}-bit semiprimes found")
    return pool


def build_reduce(
    *,
    variant: str = "fixed_n",           # fixed_n | fixed_y | vary_both
    bits: int = 14,
    n_moduli: int = 32,
    n_train: int = 100_000,
    n_eval: int = 2_000,
    seed: int = 45,
) -> Dataset:
    if variant not in ("fixed_n", "fixed_y", "vary_both"):
        raise ValueError("variant must be fixed_n, fixed_y or vary_both")
    rng = random.Random(seed)
    pool = _semiprime_pool(bits, 1 if variant == "fixed_n" else n_moduli, rng)

    # Held-out moduli for the variants where N varies.
    if variant == "fixed_n":
        train_moduli, ood_moduli = pool, []
    else:
        split = max(1, int(0.8 * len(pool)))
        train_moduli, ood_moduli = pool[:split], pool[split:]

    fixed_y = rng.randrange(0, max(pool) ** 2) if variant == "fixed_y" else None
    seen: set[tuple[int, int]] = set()

    def sample(moduli: list[int], count: int, fresh_y: bool, exclude: bool) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        guard = 0
        while len(out) < count and guard < count * 200:
            guard += 1
            modulus = rng.choice(moduli)
            if variant == "fixed_y" and not fresh_y:
                y = fixed_y
            else:
                y = rng.randrange(0, modulus * modulus)
            if (modulus, y) in seen:
                continue
            if exclude:
                seen.add((modulus, y))
            out.append(make_record(modulus, y, 1, y % modulus))
        if len(out) < count:
            raise ValueError(f"could not draw {count} unique reduce examples")
        return out

    train = sample(train_moduli, n_train, fresh_y=True, exclude=True)
    cohorts = [Cohort("id_fresh_y", sample(train_moduli, n_eval, True, True),
                      "seen N, unseen y")]
    if ood_moduli:
        cohorts.append(Cohort("ood_n", sample(ood_moduli, n_eval, True, False),
                              "unseen N"))
    return Dataset(
        task=f"reduce_{variant}",
        train=train,
        cohorts=cohorts,
        meta={"variant": variant, "bits": bits,
              "n_train_moduli": len(train_moduli), "n_ood_moduli": len(ood_moduli)},
    )


# ---------------------------------------------------------------------------
# 4. squaremod:  x*x mod N at fixed N.  This is the competition's T=1.
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
    p, q = _sample_rsa_factors(modulus_bits=bits, rng=rng)
    modulus = p * q
    x_space = (p - 1) * (q - 1)
    if n_train + 2 * n_eval > x_space:
        raise ValueError(
            f"bits={bits} gives only {x_space} units; lower n_train or raise bits"
        )

    seen: set[int] = set()

    def draw_unit() -> int:
        while True:
            value = rng.randrange(1, modulus)
            if math.gcd(value, modulus) == 1 and value not in seen:
                return value

    train_x = []
    while len(train_x) < n_train:
        value = draw_unit()
        seen.add(value)
        train_x.append(value)
    fresh_x = []
    while len(fresh_x) < n_eval:
        value = draw_unit()
        seen.add(value)
        fresh_x.append(value)
    held_in = rng.sample(train_x, min(n_eval, len(train_x)))

    def records(values: list[int]) -> list[dict[str, Any]]:
        return [make_record(modulus, x, 1, (x * x) % modulus) for x in values]

    return Dataset(
        task="squaremod",
        train=records(train_x),
        cohorts=[
            Cohort("train_seen", records(held_in), "x drawn from the training pool"),
            Cohort("test_fresh", records(fresh_x), "x never trained on"),
        ],
        meta={"bits": bits, "modulus": modulus, "x_space": x_space,
              "x_coverage": n_train / x_space},
    )
