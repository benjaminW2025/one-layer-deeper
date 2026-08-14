"""Training loop, metrics, and error capture.

Loss and readout match the evaluator exactly: gather logits at
target_positions, cross-entropy over valid answer digits only, per-digit
argmax, exact match = every valid digit correct.

`loss_reduction` is the one deliberate knob:
    token    mean over every answer digit in the batch (evaluator default;
             long answers contribute more terms)
    example  mean per example, then over examples (matches exact-match scoring)
"""

from __future__ import annotations

import csv
import math
import random
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .format import DIGIT_OFFSET, VOCAB_SIZE, collate
from .model import Encoder, count_parameters


# ---------------------------------------------------------------------------
# batching
# ---------------------------------------------------------------------------

def tensors_from_records(records: list[dict[str, Any]], max_seq_len: int | None = None):
    batch = collate(records)
    if max_seq_len is not None and batch["input_ids"].size(1) < max_seq_len:
        pad = max_seq_len - batch["input_ids"].size(1)
        batch["input_ids"] = F.pad(batch["input_ids"], (0, pad), value=0)
        batch["attention_mask"] = F.pad(batch["attention_mask"], (0, pad), value=False)
    return batch


def gather_answer_logits(logits: torch.Tensor, target_positions: torch.Tensor) -> torch.Tensor:
    index = target_positions.clamp_min(0).unsqueeze(-1).expand(-1, -1, logits.size(-1))
    return logits.gather(1, index)


def answer_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    target_positions: torch.Tensor,
    reduction: str = "token",
) -> torch.Tensor:
    answer_logits = gather_answer_logits(logits, target_positions).float()
    if reduction == "token":
        return F.cross_entropy(answer_logits.transpose(1, 2), labels, ignore_index=-100)
    per_token = F.cross_entropy(
        answer_logits.transpose(1, 2), labels, ignore_index=-100, reduction="none"
    )
    valid = labels != -100
    counts = valid.sum(dim=1)
    per_example = (per_token * valid).sum(dim=1) / counts.clamp_min(1)
    return per_example[counts > 0].mean()


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

@dataclass
class Metrics:
    n: int
    exact: float
    digit: float
    chance_exact: float
    chance_digit: float
    # accuracy per place value, index 0 = ones digit
    per_place: list[float] = field(default_factory=list)
    # of the wrong answers, how many were off by exactly one place value
    mean_wrong_digits: float = 0.0

    def as_row(self) -> dict[str, Any]:
        return {
            "n_eval": self.n,
            "exact_acc": round(self.exact, 5),
            "digit_acc": round(self.digit, 5),
            "chance_exact": round(self.chance_exact, 6),
            "chance_digit": round(self.chance_digit, 5),
            "per_place_acc": "|".join(f"{v:.3f}" for v in self.per_place),
            "mean_wrong_digits": round(self.mean_wrong_digits, 3),
        }


def chance_baselines(records: list[dict[str, Any]]) -> tuple[float, float]:
    """Best constant predictor: per place value, and whole answer."""
    place_counts: list[Counter[int]] = []
    answers: Counter[int] = Counter()
    for record in records:
        answers[record["answer"]] += 1
        for place, token in enumerate(reversed(record["labels"])):
            while len(place_counts) <= place:
                place_counts.append(Counter())
            place_counts[place][token] += 1
    hits = sum(c.most_common(1)[0][1] for c in place_counts)
    total = sum(sum(c.values()) for c in place_counts)
    return answers.most_common(1)[0][1] / len(records), hits / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    tensors: dict[str, torch.Tensor],
    records: list[dict[str, Any]],
    device: torch.device,
    batch_size: int = 1024,
    capture_errors: bool = False,
) -> tuple[Metrics, list[dict[str, Any]]]:
    model.eval()
    n = tensors["input_ids"].size(0)
    digit_hits = digit_total = exact_hits = 0
    wrong_digit_counts: list[int] = []
    place_hits: Counter[int] = Counter()
    place_total: Counter[int] = Counter()
    errors: list[dict[str, Any]] = []

    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        batch = {k: v[start:stop].to(device) for k, v in tensors.items()}
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            logits = model(batch["input_ids"], batch["attention_mask"])
        predictions = gather_answer_logits(
            logits.float(), batch["target_positions"]
        ).argmax(dim=-1)
        labels = batch["labels"]
        valid = labels != -100
        hits = (predictions == labels) & valid

        digit_hits += int(hits.sum())
        digit_total += int(valid.sum())
        row_exact = (hits | ~valid).all(dim=1)
        exact_hits += int(row_exact.sum())

        # Place value: labels are right-aligned, so column j from the right is
        # the 10^j place for every row.
        width = labels.size(1)
        for column in range(width):
            place = width - 1 - column
            column_valid = valid[:, column]
            place_total[place] += int(column_valid.sum())
            place_hits[place] += int((hits[:, column] & column_valid).sum())

        wrong = (~hits & valid).sum(dim=1)
        wrong_digit_counts.extend(wrong[~row_exact].tolist())

        if capture_errors:
            for offset in range(stop - start):
                if bool(row_exact[offset]):
                    continue
                record = records[start + offset]
                predicted = [
                    int(t) for t, keep in zip(predictions[offset].tolist(),
                                              valid[offset].tolist()) if keep
                ]
                errors.append({
                    "answer": record["answer"],
                    "predicted_tokens": predicted,
                    "true_tokens": [
                        int(t) for t, keep in zip(labels[offset].tolist(),
                                                  valid[offset].tolist()) if keep
                    ],
                    "n_wrong_digits": int(wrong[offset]),
                })

    model.train()
    chance_exact, chance_digit = chance_baselines(records)
    per_place = [
        place_hits[p] / place_total[p] if place_total[p] else float("nan")
        for p in sorted(place_total)
    ]
    metrics = Metrics(
        n=n,
        exact=exact_hits / max(n, 1),
        digit=digit_hits / max(digit_total, 1),
        chance_exact=chance_exact,
        chance_digit=chance_digit,
        per_place=per_place,
        mean_wrong_digits=(
            sum(wrong_digit_counts) / len(wrong_digit_counts) if wrong_digit_counts else 0.0
        ),
    )
    return metrics, errors


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    steps: int = 6000
    batch_size: int = 512
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    grad_clip: float = 1.0
    loss_reduction: str = "token"
    seed: int = 0
    log_every: int = 1000


def pick_device(explicit: str | None = None) -> torch.device:
    if explicit:
        return torch.device(explicit)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable; refusing to fall back to CPU silently "
            f"(torch {torch.__version__} built for CUDA {torch.version.cuda}). "
            "Use /usr/bin/python, or pass device='cpu' to opt in."
        )
    free = [(torch.cuda.mem_get_info(i)[0], i) for i in range(torch.cuda.device_count())]
    return torch.device(f"cuda:{max(free)[1]}")


def train(
    model: nn.Module,
    train_tensors: dict[str, torch.Tensor],
    config: TrainConfig,
    device: torch.device,
    verbose: bool = True,
) -> dict[str, Any]:
    torch.manual_seed(config.seed)
    model.to(device).train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay,
        fused=(device.type == "cuda"),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / max(config.warmup_steps, 1))
    )
    data = {k: v.to(device) for k, v in train_tensors.items()}
    n = data["input_ids"].size(0)
    generator = torch.Generator(device="cpu").manual_seed(config.seed)

    losses: list[float] = []
    started = time.monotonic()
    for step in range(config.steps):
        index = torch.randint(0, n, (config.batch_size,), generator=generator).to(device)
        batch = {k: v[index] for k, v in data.items()}
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            logits = model(batch["input_ids"], batch["attention_mask"])
        loss = answer_loss(
            logits.float(), batch["labels"], batch["target_positions"],
            config.loss_reduction,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.detach()))
        if not math.isfinite(losses[-1]):
            break
        if verbose and config.log_every and (step + 1) % config.log_every == 0:
            recent = sum(losses[-100:]) / len(losses[-100:])
            print(f"    step {step + 1:>6}  loss {recent:.4f}", flush=True)

    first = sum(losses[:20]) / max(len(losses[:20]), 1)
    final = sum(losses[-100:]) / max(len(losses[-100:]), 1)
    return {
        "steps": len(losses),
        "train_seconds": round(time.monotonic() - started, 2),
        "first_loss": round(first, 5),
        "final_loss": round(final, 5),
        "diverged": not math.isfinite(losses[-1]) if losses else True,
        "flat": bool(losses) and final > 0.95 * first,
        "params": count_parameters(model),
        "device": str(device),
    }


def build_model(dataset_max_seq_len: int, positional: str, d_model: int,
                n_layers: int, n_heads: int = 4) -> nn.Module:
    return Encoder(
        vocab_size=VOCAB_SIZE, max_seq_len=dataset_max_seq_len, d_model=d_model,
        n_layers=n_layers, n_heads=n_heads, positional=positional,
    )


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------

def append_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
