"""Generate local decimal-multiplication diagnostics.

These datasets are for architecture research only. They deliberately use the
same separate prompt/output shape as the competition task, but have their own
field markers and must never be used to construct a competition submission.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random


EXPERIMENT_ROOT = Path(__file__).resolve().parent
TOKEN_IDS = {"PAD": 0, "BOS": 1, "A": 2, "B": 3, "ANS": 4, "EOS": 5}
DIGIT_OFFSET = 7
VOCAB_SIZE = DIGIT_OFFSET + 10


@dataclass(frozen=True)
class Preset:
    name: str
    id_digits: tuple[int, ...]
    ood_digits: tuple[int, ...]
    examples_per_id_cell: int
    examples_per_ood_cell: int
    train_fraction: float


PRESETS = {
    # Four input-size cells, 8,000 total prompts. This is large enough to
    # expose length generalization without making local iteration slow.
    "easy": Preset("easy", (3, 4), (5,), 2_000, 500, 0.8),
    # Four larger cells and 90,000 prompts, matching Medium's order of scale.
    "medium": Preset("medium", (4, 5), (6,), 22_500, 2_250, 0.9),
}


def digits(value: int) -> list[int]:
    return [DIGIT_OFFSET + int(char) for char in str(value)]


def make_record(a: int, b: int, split: str) -> dict[str, object]:
    product = a * b
    return {
        "input_ids": [TOKEN_IDS["A"], *digits(a), TOKEN_IDS["B"], *digits(b)],
        "labels": digits(product),
        "split": split,
        # These fields make later error analysis possible; training loaders can
        # ignore them and use only input_ids/labels.
        "a_digits": len(str(a)),
        "b_digits": len(str(b)),
        "product_digits": len(str(product)),
        "a": a,
        "b": b,
    }


def random_value(rng: random.Random, digit_count: int) -> int:
    if digit_count < 1:
        raise ValueError("digit_count must be positive")
    return rng.randrange(10 ** (digit_count - 1), 10**digit_count)


def sample_cell(
    rng: random.Random,
    a_digits: int,
    b_digits: int,
    count: int,
) -> list[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    while len(pairs) < count:
        pairs.add((random_value(rng, a_digits), random_value(rng, b_digits)))
    return list(pairs)


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True))
            handle.write("\n")


def generate(preset: Preset, output_dir: Path, seed: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    train: list[dict[str, object]] = []
    test: list[dict[str, object]] = []
    ood_one_long: list[dict[str, object]] = []
    ood_both_long: list[dict[str, object]] = []

    for a_width in preset.id_digits:
        for b_width in preset.id_digits:
            pairs = sample_cell(rng, a_width, b_width, preset.examples_per_id_cell)
            rng.shuffle(pairs)
            train_count = round(len(pairs) * preset.train_fraction)
            train.extend(make_record(a, b, "train") for a, b in pairs[:train_count])
            test.extend(make_record(a, b, "test") for a, b in pairs[train_count:])

    for long_width in preset.ood_digits:
        for id_width in preset.id_digits:
            for a_width, b_width in ((long_width, id_width), (id_width, long_width)):
                ood_one_long.extend(
                    make_record(a, b, "ood_one_long")
                    for a, b in sample_cell(
                        rng, a_width, b_width, preset.examples_per_ood_cell
                    )
                )
        for other_long_width in preset.ood_digits:
            ood_both_long.extend(
                make_record(a, b, "ood_both_long")
                for a, b in sample_cell(
                    rng, long_width, other_long_width, preset.examples_per_ood_cell)
            )

    for records in (train, test, ood_one_long, ood_both_long):
        rng.shuffle(records)
    write_jsonl(output_dir / "train.jsonl", train)
    write_jsonl(output_dir / "test.jsonl", test)
    write_jsonl(output_dir / "ood_one_long.jsonl", ood_one_long)
    write_jsonl(output_dir / "ood_both_long.jsonl", ood_both_long)
    config = {
        "kind": "controlled_decimal_multiplication",
        "preset": asdict(preset),
        "seed": seed,
        "token_ids": TOKEN_IDS | {"DIGIT_OFFSET": DIGIT_OFFSET},
        "vocab_size": VOCAB_SIZE,
        "separate_input_output": True,
        "splits": {
            "train": len(train),
            "test": len(test),
            "ood_one_long": len(ood_one_long),
            "ood_both_long": len(ood_both_long),
        },
    }
    with (output_dir / "dataset_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=tuple(PRESETS), required=True)
    parser.add_argument(
        "--output_root", type=Path, default=EXPERIMENT_ROOT / "data"
    )
    parser.add_argument("--seed", type=int, default=45)
    args = parser.parse_args()
    preset = PRESETS[args.preset]
    output_dir = args.output_root / f"multiplication_{preset.name}"
    generate(preset, output_dir, args.seed)
    print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()
