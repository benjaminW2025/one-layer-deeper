from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from controlled_experiments.generate_reduction import DIGIT_OFFSET, Preset, generate


class ReductionGenerationTests(unittest.TestCase):
    def test_arithmetic_splits_and_prompt_uniqueness(self) -> None:
        preset = Preset("test", (2,), (1,), (3,), (2,), 20, 8, 8, 8, 0.75)
        with tempfile.TemporaryDirectory() as temporary_dir:
            output = Path(temporary_dir) / "reduction_test"
            generate(preset, output, seed=45)
            expected_counts = {
                "train": 15,
                "test": 5,
                "ood_z_long": 8,
                "ood_n_long": 8,
                "ood_both_long": 8,
            }
            seen: set[tuple[int, int]] = set()
            for split, expected_count in expected_counts.items():
                records = [
                    json.loads(line)
                    for line in (output / f"{split}.jsonl").read_text().splitlines()
                ]
                self.assertEqual(len(records), expected_count)
                for record in records:
                    prompt = (record["z"], record["modulus"])
                    self.assertNotIn(prompt, seen)
                    seen.add(prompt)
                    quotient, remainder = divmod(record["z"], record["modulus"])
                    self.assertGreaterEqual(quotient, 1)
                    self.assertEqual(record["quotient"], quotient)
                    self.assertEqual(record["remainder"], remainder)
                    decoded = int(
                        "".join(str(token - DIGIT_OFFSET) for token in record["labels"])
                    )
                    self.assertEqual(decoded, remainder)
            config = json.loads((output / "dataset_config.json").read_text())
            self.assertEqual(config["splits"], expected_counts)

    def test_generation_is_deterministic(self) -> None:
        preset = Preset("tiny", (2,), (1,), (3,), (2,), 10, 3, 3, 3, 0.8)
        with tempfile.TemporaryDirectory() as temporary_dir:
            first = Path(temporary_dir) / "first"
            second = Path(temporary_dir) / "second"
            generate(preset, first, seed=74)
            generate(preset, second, seed=74)
            for filename in (
                "train.jsonl",
                "test.jsonl",
                "ood_z_long.jsonl",
                "ood_n_long.jsonl",
                "ood_both_long.jsonl",
                "dataset_config.json",
            ):
                self.assertEqual(
                    (first / filename).read_bytes(),
                    (second / filename).read_bytes(),
                )


if __name__ == "__main__":
    unittest.main()
