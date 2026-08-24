from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from controlled_experiments.generate_square_mod import DIGIT_OFFSET, Preset, generate


class SquareModGenerationTests(unittest.TestCase):
    def test_arithmetic_disjoint_moduli_and_counts(self) -> None:
        preset = Preset("test", (3,), (4,), 2, 1, 1, 8, 2)
        with tempfile.TemporaryDirectory() as temporary_dir:
            output = Path(temporary_dir) / "square_mod_test"
            generate(preset, output, seed=45)
            expected_counts = {
                "train": 12,
                "test_seen_n": 4,
                "test": 8,
                "ood_n_long": 8,
            }
            moduli_by_split: dict[str, set[int]] = {}
            for split, expected_count in expected_counts.items():
                records = [
                    json.loads(line)
                    for line in (output / f"{split}.jsonl").read_text().splitlines()
                ]
                self.assertEqual(len(records), expected_count)
                moduli_by_split[split] = {int(record["modulus"]) for record in records}
                expected_moduli = 2 if split in ("train", "test_seen_n") else 1
                self.assertEqual(len(moduli_by_split[split]), expected_moduli)
                for record in records:
                    self.assertEqual(record["modulus"], record["p"] * record["q"])
                    self.assertNotEqual(record["p"], record["q"])
                    expected = pow(record["x"], 2, record["modulus"])
                    self.assertEqual(record["result"], expected)
                    decoded = int(
                        "".join(str(token - DIGIT_OFFSET) for token in record["labels"])
                    )
                    self.assertEqual(decoded, expected)
            self.assertTrue(moduli_by_split["train"].isdisjoint(moduli_by_split["test"]))
            self.assertEqual(moduli_by_split["train"], moduli_by_split["test_seen_n"])
            train_prompts = {
                (record["x"], record["modulus"])
                for record in [
                    json.loads(line)
                    for line in (output / "train.jsonl").read_text().splitlines()
                ]
            }
            seen_test_prompts = {
                (record["x"], record["modulus"])
                for record in [
                    json.loads(line)
                    for line in (output / "test_seen_n.jsonl").read_text().splitlines()
                ]
            }
            self.assertTrue(train_prompts.isdisjoint(seen_test_prompts))
            self.assertTrue(moduli_by_split["ood_n_long"].isdisjoint(moduli_by_split["train"]))
            config = json.loads((output / "dataset_config.json").read_text())
            self.assertEqual(config["splits"], expected_counts)

    def test_generation_is_deterministic(self) -> None:
        preset = Preset("tiny", (3,), (4,), 1, 1, 1, 4, 1)
        with tempfile.TemporaryDirectory() as temporary_dir:
            first = Path(temporary_dir) / "first"
            second = Path(temporary_dir) / "second"
            generate(preset, first, seed=74)
            generate(preset, second, seed=74)
            for filename in (
                "train.jsonl",
                "test_seen_n.jsonl",
                "test.jsonl",
                "ood_n_long.jsonl",
                "dataset_config.json",
            ):
                self.assertEqual(
                    (first / filename).read_bytes(),
                    (second / filename).read_bytes(),
                )


if __name__ == "__main__":
    unittest.main()
