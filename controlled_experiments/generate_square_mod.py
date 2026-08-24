"""Generate controlled one-step modular-squaring diagnostics.

The model receives decimal ``x`` and a semiprime modulus ``N`` and predicts
``x**2 mod N``. Train and test use disjoint moduli of the same decimal widths;
``ood_n_long`` uses wider, unseen moduli. This isolates one transition of the
competition task without adding the repeated-composition requirement.

These datasets are for local architecture research only and must not be used to
construct a competition submission.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random


EXPERIMENT_ROOT = Path(__file__).resolve().parent
TOKEN_IDS = {"PAD": 0, "BOS": 1, "X": 2, "N": 3, "ANS": 4, "EOS": 5}
DIGIT_OFFSET = 7
VOCAB_SIZE = DIGIT_OFFSET + 10


@dataclass(frozen=True)
class Preset:
    name: str
    id_modulus_digits: tuple[int, ...]
    ood_modulus_digits: tuple[int, ...]
    train_moduli_per_width: int
    test_moduli_per_width: int
    ood_moduli_per_width: int
    examples_per_modulus: int
    seen_test_examples_per_modulus: int


PRESETS = {
    "easy": Preset("easy", (3, 4), (5,), 12, 4, 8, 128, 32),
    "medium": Preset("medium", (4, 5), (6,), 32, 8, 16, 512, 64),
}


def digit_tokens(value: int) -> list[int]:
    return [DIGIT_OFFSET + int(char) for char in str(value)]


def primes_between(lower: int, upper: int) -> list[int]:
    """Return primes in the inclusive interval; preset bounds are small."""
    primes: list[int] = []
    for candidate in range(max(2, lower), upper + 1):
        if candidate > 2 and candidate % 2 == 0:
            continue
        limit = math.isqrt(candidate)
        if all(candidate % divisor for divisor in range(3, limit + 1, 2)):
            primes.append(candidate)
    return primes


def semiprimes_with_digits(digit_count: int) -> list[tuple[int, int, int]]:
    """Enumerate distinct balanced p*q values with exactly the requested width."""
    if digit_count < 2:
        raise ValueError("modulus digit count must be at least two")
    minimum = 10 ** (digit_count - 1)
    maximum = 10**digit_count - 1
    factors = primes_between(math.isqrt(minimum), math.isqrt(maximum) + 1)
    values: list[tuple[int, int, int]] = []
    seen: set[int] = set()
    for index, p in enumerate(factors):
        for q in factors[index + 1 :]:
            modulus = p * q
            if minimum <= modulus <= maximum and modulus not in seen:
                seen.add(modulus)
                values.append((modulus, p, q))
    return values


def make_record(x: int, modulus: int, p: int, q: int, split: str) -> dict[str, object]:
    square = x * x
    quotient, result = divmod(square, modulus)
    return {
        "input_ids": [
            TOKEN_IDS["X"],
            *digit_tokens(x),
            TOKEN_IDS["N"],
            *digit_tokens(modulus),
        ],
        "labels": digit_tokens(result),
        "split": split,
        "x": x,
        "modulus": modulus,
        # Factors are diagnostic metadata only; they are never input tokens.
        "p": p,
        "q": q,
        "square": square,
        "z": square,
        "quotient": quotient,
        "result": result,
        "x_digits": len(str(x)),
        "modulus_digits": len(str(modulus)),
        "square_digits": len(str(square)),
        "z_digits": len(str(square)),
        "quotient_digits": len(str(quotient)),
        "result_digits": len(str(result)),
    }


def records_for_moduli(
    rng: random.Random,
    moduli: list[tuple[int, int, int]],
    examples_per_modulus: int,
    split: str,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for modulus, p, q in moduli:
        if examples_per_modulus > modulus - 1:
            raise ValueError(
                f"cannot draw {examples_per_modulus} unique nonzero x values modulo {modulus}"
            )
        x_values = rng.sample(range(1, modulus), examples_per_modulus)
        records.extend(make_record(x, modulus, p, q, split) for x in x_values)
    rng.shuffle(records)
    return records


def train_and_seen_test_records(
    rng: random.Random,
    moduli: list[tuple[int, int, int]],
    examples_per_modulus: int,
    seen_test_examples_per_modulus: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if not 0 < seen_test_examples_per_modulus < examples_per_modulus:
        raise ValueError("seen-modulus test count must be between zero and total examples")
    train: list[dict[str, object]] = []
    test_seen_n: list[dict[str, object]] = []
    train_count = examples_per_modulus - seen_test_examples_per_modulus
    for modulus, p, q in moduli:
        if examples_per_modulus > modulus - 1:
            raise ValueError(
                f"cannot draw {examples_per_modulus} unique nonzero x values modulo {modulus}"
            )
        x_values = rng.sample(range(1, modulus), examples_per_modulus)
        train.extend(
            make_record(x, modulus, p, q, "train") for x in x_values[:train_count]
        )
        test_seen_n.extend(
            make_record(x, modulus, p, q, "test_seen_n")
            for x in x_values[train_count:]
        )
    rng.shuffle(train)
    rng.shuffle(test_seen_n)
    return train, test_seen_n


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True))
            handle.write("\n")


def generate(preset: Preset, output_dir: Path, seed: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    train_moduli: list[tuple[int, int, int]] = []
    test_moduli: list[tuple[int, int, int]] = []
    ood_moduli: list[tuple[int, int, int]] = []

    for width in preset.id_modulus_digits:
        candidates = semiprimes_with_digits(width)
        rng.shuffle(candidates)
        required = preset.train_moduli_per_width + preset.test_moduli_per_width
        if len(candidates) < required:
            raise ValueError(
                f"only {len(candidates)} balanced semiprimes available for {width} digits; "
                f"preset requires {required}"
            )
        train_moduli.extend(candidates[: preset.train_moduli_per_width])
        test_moduli.extend(candidates[preset.train_moduli_per_width : required])

    for width in preset.ood_modulus_digits:
        candidates = semiprimes_with_digits(width)
        rng.shuffle(candidates)
        if len(candidates) < preset.ood_moduli_per_width:
            raise ValueError(
                f"only {len(candidates)} balanced semiprimes available for {width} digits; "
                f"preset requires {preset.ood_moduli_per_width}"
            )
        ood_moduli.extend(candidates[: preset.ood_moduli_per_width])

    train, test_seen_n = train_and_seen_test_records(
        rng,
        train_moduli,
        preset.examples_per_modulus,
        preset.seen_test_examples_per_modulus,
    )
    splits = {
        "train": train,
        "test_seen_n": test_seen_n,
        "test": records_for_moduli(
            rng, test_moduli, preset.examples_per_modulus, "test"
        ),
        "ood_n_long": records_for_moduli(
            rng, ood_moduli, preset.examples_per_modulus, "ood_n_long"
        ),
    }
    for split, records in splits.items():
        write_jsonl(output_dir / f"{split}.jsonl", records)

    config = {
        "kind": "controlled_decimal_square_mod",
        "operation": "x_squared_mod_n",
        "preset": asdict(preset),
        "seed": seed,
        "token_ids": TOKEN_IDS | {"DIGIT_OFFSET": DIGIT_OFFSET},
        "vocab_size": VOCAB_SIZE,
        "separate_input_output": True,
        "output_format": "variable_width_most_significant_first",
        "modulus_family": "product_of_two_distinct_balanced_primes",
        "test_moduli_disjoint_from_train": True,
        "test_seen_n_uses_held_out_x_on_training_moduli": True,
        "splits": {name: len(records) for name, records in splits.items()},
    }
    with (output_dir / "dataset_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=tuple(PRESETS), required=True)
    parser.add_argument("--output_root", type=Path, default=EXPERIMENT_ROOT / "data")
    parser.add_argument("--seed", type=int, default=45)
    args = parser.parse_args()
    preset = PRESETS[args.preset]
    output_dir = args.output_root / f"square_mod_{preset.name}"
    generate(preset, output_dir, args.seed)
    print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()
