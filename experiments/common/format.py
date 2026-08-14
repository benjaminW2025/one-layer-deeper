"""Prompt format — reuses the competition tokenizer verbatim.

We import `tokenize_squaring_mod_with_result` and `collate_squaring_mod` from
`data/squaring_mod.py` rather than writing our own, so vocab, digit order,
padding, and the right-aligned answer readout are identical to the evaluator's.

The template is always the competition's three fields:

    [N] digits(a)  [X] digits(b)  [T] digits(c)      answer read off the tail

Each task just decides what a, b, c and the answer mean:

    addition    a = x,  b = y,  c = 0        answer = x + y
    square      a = N,  b = x,  c = 1        answer = x * x
    reduce      a = N,  b = y,  c = 1        answer = y mod N
    squaremod   a = N,  b = x,  c = 1        answer = x*x mod N

Marker names are just delimiters as far as the model is concerned; keeping the
same three keeps vocab_size = 17 and every downstream lesson transferable.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.squaring_mod import (  # noqa: E402
    DIGIT_OFFSET,
    TOKEN_IDS,
    VOCAB_SIZE,
    collate_squaring_mod,
    tokenize_squaring_mod_with_result,
)

__all__ = [
    "DIGIT_OFFSET", "TOKEN_IDS", "VOCAB_SIZE",
    "make_record", "collate", "prompt_length", "digits_to_int",
]


def make_record(a: int, b: int, c: int, answer: int) -> dict[str, Any]:
    """One example in the competition's separate-input/output form."""
    input_ids, labels = tokenize_squaring_mod_with_result(
        a, b, c, answer, separate_input_output=True
    )
    return {"input_ids": input_ids, "labels": labels, "answer": answer}


def collate(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad and right-align exactly as the evaluator does."""
    return collate_squaring_mod(
        [{"input_ids": r["input_ids"], "labels": r["labels"]} for r in records]
    )


def prompt_length(a: int, b: int, c: int) -> int:
    return 3 + len(str(a)) + len(str(b)) + len(str(c))


def digits_to_int(token_ids: list[int]) -> int | None:
    """Digit tokens -> integer, or None if any token is not a digit.

    Used for error analysis: how far off was the prediction numerically.
    """
    text = ""
    for token in token_ids:
        digit = token - DIGIT_OFFSET
        if not 0 <= digit <= 9:
            return None
        text += str(digit)
    return int(text) if text else None
