"""Per-task digit tokenizers.

Same strategy as the competition, sized to each task:

  * numbers are decimal digits, most-significant first, VARIABLE width
  * a prompt is a sequence of (marker, number) fields
  * the answer is NOT appended: it is read off the last len(answer) positions
    of the prompt, right-aligned, so the final position is always the ones
    digit, the one before it the tens digit, and so on

Ids: markers occupy 0..m-1, digits 0-9 occupy m..m+9, PAD is m+10.

    addition   markers ("X","Y")   X=0 Y=1  digits 2..11  PAD=12  vocab 13
    square     markers ("X","Y")   X=0 Y=1  digits 2..11  PAD=12  vocab 13
    reduce     markers ("N","Y")   N=0 Y=1  digits 2..11  PAD=12  vocab 13
    squaremod  markers ("N","X")   N=0 X=1  digits 2..11  PAD=12  vocab 13

PAD sits above the digits so the marker/digit ids stay compact; padded
positions are excluded by attention_mask, never by id.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class DigitTokenizer:
    markers: tuple[str, ...]

    @property
    def digit_offset(self) -> int:
        return len(self.markers)

    @property
    def digit_range(self) -> tuple[int, int]:
        """[lo, hi) ids that are digits. Excludes PAD."""
        return self.digit_offset, self.digit_offset + 10

    @property
    def pad_id(self) -> int:
        return self.digit_offset + 10

    @property
    def vocab_size(self) -> int:
        return self.digit_offset + 11

    def marker_id(self, name: str) -> int:
        return self.markers.index(name)

    def digits(self, value: int) -> list[int]:
        if value < 0:
            raise ValueError("only non-negative integers are tokenized")
        return [self.digit_offset + int(c) for c in str(value)]

    def digits_padded(self, value: int, width: int) -> list[int]:
        """Zero-padded to a FIXED width, unlike `digits`. For encoding one
        chunk of a multi-chunk field (e.g. one place's raw, un-reduced
        convolution sum) where chunk boundaries must stay fixed-width so
        concatenating chunks doesn't silently swallow a leading zero."""
        if not 0 <= value < 10 ** width:
            raise ValueError(f"{value} does not fit in {width} digits")
        return [self.digit_offset + int(c) for c in str(value).zfill(width)]

    def encode(self, fields: list[tuple[str, int]]) -> list[int]:
        ids: list[int] = []
        for name, value in fields:
            ids.append(self.marker_id(name))
            ids.extend(self.digits(value))
        return ids

    def decode(self, ids: list[int]) -> int | None:
        """Digit ids -> integer, or None if any id is not a digit."""
        low, high = self.digit_range
        text = ""
        for token in ids:
            if not low <= token < high:
                return None
            text += str(token - low)
        return int(text) if text else None

    def record(self, fields: list[tuple[str, int]], answer: int) -> dict[str, Any]:
        return {
            "input_ids": self.encode(fields),
            "labels": self.digits(answer),
            "answer": answer,
        }

    def describe(self, ids: list[int]) -> list[str]:
        """Human-readable tokens, for debugging."""
        low, high = self.digit_range
        out = []
        for token in ids:
            if low <= token < high:
                out.append(str(token - low))
            elif token == self.pad_id:
                out.append("PAD")
            else:
                out.append(self.markers[token])
        return out


ADDITION = DigitTokenizer(("X", "Y"))
SQUARE = DigitTokenizer(("X", "Y"))
REDUCE = DigitTokenizer(("N", "Y"))
SQUAREMOD = DigitTokenizer(("N", "X"))
CARRY_ONLY = DigitTokenizer(("S",))  # S = the raw, un-reduced per-place sums


def collate(
    records: list[dict[str, Any]],
    tokenizer: DigitTokenizer,
    max_seq_len: int | None = None,
) -> dict[str, torch.Tensor]:
    """Pad prompts; right-align answers onto the prompt tail.

    target_positions[row] = arange(prompt_len - answer_len, prompt_len), which
    is exactly what the competition's collate produces. Labels are also
    right-aligned so column j refers to the same place value in every row.
    """
    batch = len(records)
    width = max_seq_len or max(len(r["input_ids"]) for r in records)
    max_answer = max(len(r["labels"]) for r in records)

    input_ids = torch.full((batch, width), tokenizer.pad_id, dtype=torch.long)
    attention_mask = torch.zeros((batch, width), dtype=torch.bool)
    labels = torch.full((batch, max_answer), -100, dtype=torch.long)
    target_positions = torch.zeros((batch, max_answer), dtype=torch.long)

    for row, record in enumerate(records):
        prompt, answer = record["input_ids"], record["labels"]
        prompt_len, answer_len = len(prompt), len(answer)
        if prompt_len > width:
            raise ValueError(f"prompt of {prompt_len} exceeds max_seq_len {width}")
        if answer_len > prompt_len:
            raise ValueError(
                f"answer of {answer_len} digits cannot be read off a "
                f"{prompt_len}-token prompt"
            )
        input_ids[row, :prompt_len] = torch.tensor(prompt)
        attention_mask[row, :prompt_len] = True
        labels[row, -answer_len:] = torch.tensor(answer)
        target_positions[row, -answer_len:] = torch.arange(
            prompt_len - answer_len, prompt_len
        )
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "target_positions": target_positions,
    }
