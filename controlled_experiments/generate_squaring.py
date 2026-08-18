"""Generate direct decimal-squaring diagnostics with explicit answer slots."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random


EXPERIMENT_ROOT = Path(__file__).resolve().parent
TOKEN_IDS = {"PAD": 0, "BOS": 1, "X": 2, "ANS": 3, "OUT": 4, "EOS": 5}
DIGIT_OFFSET = 7
VOCAB_SIZE = DIGIT_OFFSET + 10


@dataclass(frozen=True)
class Preset:
    name: str
    id_cells: tuple[tuple[int, int], ...]
    ood_cells: tuple[tuple[int, int], ...]
    train_fraction: float


PRESETS = {
    # Three-digit inputs have only 900 unique values, so Easy uses all of
    # them plus a larger four-digit cell without train/test overlap.
    "easy": Preset("easy", ((3, 900), (4, 8_000)), ((5, 1_000),), 0.8),
    "medium": Preset("medium", ((4, 9_000), (5, 81_000)), ((6, 4_500),), 0.9),
}


def digit_tokens(text: str) -> list[int]:
    return [DIGIT_OFFSET + int(char) for char in text]


def make_record(x: int, width: int, split: str) -> dict[str, object]:
    output_width = 2 * width
    # Right-to-left target: the first answer slot is always the ones place.
    target = str(x * x).zfill(output_width)[::-1]
    return {
        "input_ids": [TOKEN_IDS["X"], *digit_tokens(str(x)), TOKEN_IDS["ANS"], *([TOKEN_IDS["OUT"]] * output_width)],
        "labels": digit_tokens(target),
        "split": split,
        "x": x,
        # Shared runner metadata. Squaring is multiplication with a == b.
        "a": x,
        "b": x,
        "x_digits": width,
        "output_width": output_width,
    }


def sample_values(rng: random.Random, width: int, count: int) -> list[int]:
    lower = 10 ** (width - 1)
    upper = 10**width
    if count > upper - lower:
        raise ValueError("requested more unique values than this digit width contains")
    return rng.sample(range(lower, upper), count)


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
    ood_long: list[dict[str, object]] = []
    for width, count in preset.id_cells:
        values = sample_values(rng, width, count)
        rng.shuffle(values)
        train_count = round(len(values) * preset.train_fraction)
        train.extend(make_record(x, width, "train") for x in values[:train_count])
        test.extend(make_record(x, width, "test") for x in values[train_count:])
    for width, count in preset.ood_cells:
        ood_long.extend(
            make_record(x, width, "ood_long")
            for x in sample_values(rng, width, count)
        )
    for records in (train, test, ood_long):
        rng.shuffle(records)
    write_jsonl(output_dir / "train.jsonl", train)
    write_jsonl(output_dir / "test.jsonl", test)
    write_jsonl(output_dir / "ood_long.jsonl", ood_long)
    config = {
        "kind": "controlled_decimal_squaring",
        "preset": asdict(preset),
        "seed": seed,
        "token_ids": TOKEN_IDS | {"DIGIT_OFFSET": DIGIT_OFFSET},
        "vocab_size": VOCAB_SIZE,
        "output_format": "fixed_width_right_to_left",
        "splits": {"train": len(train), "test": len(test), "ood_long": len(ood_long)},
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
    output_dir = args.output_root / f"squaring_{preset.name}"
    generate(preset, output_dir, args.seed)
    print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()
