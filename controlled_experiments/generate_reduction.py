"""Generate controlled decimal modular-reduction diagnostics.

The model receives a true multiplication product ``z`` and a modulus ``N`` and
must predict ``z mod N``. The hidden factors are retained only as metadata so
later experiments can compare a true-product reducer with a learned product
representation without changing the reduction distribution.

These datasets are for local architecture research only and must not be used to
construct a competition submission.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random


EXPERIMENT_ROOT = Path(__file__).resolve().parent
TOKEN_IDS = {"PAD": 0, "BOS": 1, "Z": 2, "N": 3, "ANS": 4, "EOS": 5}
DIGIT_OFFSET = 7
VOCAB_SIZE = DIGIT_OFFSET + 10


@dataclass(frozen=True)
class Preset:
    name: str
    id_factor_digits: tuple[int, ...]
    id_modulus_digits: tuple[int, ...]
    ood_factor_digits: tuple[int, ...]
    ood_modulus_digits: tuple[int, ...]
    examples_per_id_cell: int
    examples_per_ood_z_cell: int
    examples_per_ood_n_cell: int
    examples_per_ood_both_cell: int
    train_fraction: float


PRESETS = {
    "easy": Preset(
        "easy", (3, 4), (3, 4), (5,), (5,), 1_000, 1_000, 125, 500, 0.8
    ),
    "medium": Preset(
        "medium", (4, 5), (4, 5), (6,), (6,), 11_250, 4_500, 1_125, 2_250, 0.9
    ),
}


def digit_tokens(value: int) -> list[int]:
    return [DIGIT_OFFSET + int(char) for char in str(value)]


def random_value(rng: random.Random, digit_count: int) -> int:
    if digit_count < 1:
        raise ValueError("digit_count must be positive")
    return rng.randrange(10 ** (digit_count - 1), 10**digit_count)


def make_record(a: int, b: int, modulus: int, split: str) -> dict[str, object]:
    z = a * b
    quotient, remainder = divmod(z, modulus)
    return {
        "input_ids": [
            TOKEN_IDS["Z"],
            *digit_tokens(z),
            TOKEN_IDS["N"],
            *digit_tokens(modulus),
        ],
        "labels": digit_tokens(remainder),
        "split": split,
        "a": a,
        "b": b,
        "z": z,
        "modulus": modulus,
        "quotient": quotient,
        "remainder": remainder,
        "a_digits": len(str(a)),
        "b_digits": len(str(b)),
        "z_digits": len(str(z)),
        "modulus_digits": len(str(modulus)),
        "quotient_digits": len(str(quotient)),
        "remainder_digits": len(str(remainder)),
    }


def sample_cell(
    rng: random.Random,
    a_digits: int,
    b_digits: int,
    modulus_digits: int,
    count: int,
    seen_prompts: set[tuple[int, int]],
) -> list[tuple[int, int, int]]:
    samples: list[tuple[int, int, int]] = []
    attempts = 0
    max_attempts = max(10_000, count * 1_000)
    while len(samples) < count:
        attempts += 1
        if attempts > max_attempts:
            raise ValueError(
                "could not sample enough unique (z, N) prompts for cell "
                f"a_digits={a_digits}, b_digits={b_digits}, "
                f"modulus_digits={modulus_digits}"
            )
        a = random_value(rng, a_digits)
        b = random_value(rng, b_digits)
        z = a * b
        modulus = random_value(rng, modulus_digits)
        # Quotient-zero examples teach copying rather than reduction and can
        # dominate longer-modulus cells.
        if modulus >= z:
            continue
        prompt = (z, modulus)
        if prompt in seen_prompts:
            continue
        seen_prompts.add(prompt)
        samples.append((a, b, modulus))
    return samples


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True))
            handle.write("\n")


def generate(preset: Preset, output_dir: Path, seed: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    seen_prompts: set[tuple[int, int]] = set()
    train: list[dict[str, object]] = []
    test: list[dict[str, object]] = []
    ood_z_long: list[dict[str, object]] = []
    ood_n_long: list[dict[str, object]] = []
    ood_both_long: list[dict[str, object]] = []

    for a_width in preset.id_factor_digits:
        for b_width in preset.id_factor_digits:
            for modulus_width in preset.id_modulus_digits:
                samples = sample_cell(
                    rng,
                    a_width,
                    b_width,
                    modulus_width,
                    preset.examples_per_id_cell,
                    seen_prompts,
                )
                rng.shuffle(samples)
                train_count = round(len(samples) * preset.train_fraction)
                train.extend(
                    make_record(a, b, modulus, "train")
                    for a, b, modulus in samples[:train_count]
                )
                test.extend(
                    make_record(a, b, modulus, "test")
                    for a, b, modulus in samples[train_count:]
                )

    # Longer z only: both hidden factors are longer while N remains in range.
    # Because the model sees only their product, this guarantees the visible z
    # is beyond the complete trained digit range.
    for factor_width in preset.ood_factor_digits:
        for modulus_width in preset.id_modulus_digits:
            ood_z_long.extend(
                make_record(a, b, modulus, "ood_z_long")
                for a, b, modulus in sample_cell(
                    rng,
                    factor_width,
                    factor_width,
                    modulus_width,
                    preset.examples_per_ood_z_cell,
                    seen_prompts,
                )
            )

    # Longer N only: hidden factors retain trained widths.
    for a_width in preset.id_factor_digits:
        for b_width in preset.id_factor_digits:
            for modulus_width in preset.ood_modulus_digits:
                ood_n_long.extend(
                    make_record(a, b, modulus, "ood_n_long")
                    for a, b, modulus in sample_cell(
                        rng,
                        a_width,
                        b_width,
                        modulus_width,
                        preset.examples_per_ood_n_cell,
                        seen_prompts,
                    )
                )

    for factor_width in preset.ood_factor_digits:
        for modulus_width in preset.ood_modulus_digits:
            ood_both_long.extend(
                make_record(a, b, modulus, "ood_both_long")
                for a, b, modulus in sample_cell(
                    rng,
                    factor_width,
                    factor_width,
                    modulus_width,
                    preset.examples_per_ood_both_cell,
                    seen_prompts,
                )
            )

    splits = {
        "train": train,
        "test": test,
        "ood_z_long": ood_z_long,
        "ood_n_long": ood_n_long,
        "ood_both_long": ood_both_long,
    }
    for records in splits.values():
        rng.shuffle(records)
    for split, records in splits.items():
        write_jsonl(output_dir / f"{split}.jsonl", records)

    config = {
        "kind": "controlled_decimal_modular_reduction",
        "preset": asdict(preset),
        "seed": seed,
        "token_ids": TOKEN_IDS | {"DIGIT_OFFSET": DIGIT_OFFSET},
        "vocab_size": VOCAB_SIZE,
        "separate_input_output": True,
        "output_format": "variable_width_most_significant_first",
        "numerator_source": "product_of_hidden_decimal_factors",
        "exclude_quotient_zero": True,
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
    output_dir = args.output_root / f"reduction_{preset.name}"
    generate(preset, output_dir, args.seed)
    print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()
