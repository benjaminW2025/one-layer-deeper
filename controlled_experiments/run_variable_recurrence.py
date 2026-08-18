"""Train a reusable cross-attention macro-step at variable recurrence depths.

The model has a frozen prompt representation C and a mutable workspace S.
Each macro-step applies a fixed sequence of distinct blocks that query S and
read keys/values from [C, S].  Training randomly selects the number of
macro-steps per batch; evaluation sweeps a requested set of depths.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import time

import torch
from torch import Tensor
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark import ModelSpec
from controlled_experiments.run_context_reinjection import ContextReinjectionTransformer
from controlled_experiments.run_multiplication import (
    MultiplicationDataset,
    collate,
    evaluate,
    loss_and_predictions,
)


class ExperimentConfig:
    def __init__(self, vocab_size: int, max_seq_len: int) -> None:
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len


class ExperimentSource:
    """The fixed controlled-model configuration, independent of submission.py."""

    D_MODEL = 256
    NUM_HEADS = 4
    ROPE_BASE = 1000.0
    TIE_EMBEDDINGS = True
    EMBED_INIT_STD = 0.02
    Config = ExperimentConfig

    @classmethod
    def rope_tables(
        cls,
        length: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        half = head_dim // 2
        inv_freq = cls.ROPE_BASE ** (
            -torch.arange(half, device=device, dtype=torch.float32) / half
        )
        positions = torch.arange(length, device=device, dtype=torch.float32)
        angles = positions[:, None] * inv_freq[None, :]
        angles = torch.cat((angles, angles), dim=-1)
        return angles.cos().to(dtype), angles.sin().to(dtype)


def parse_recurrences(value: str) -> list[int]:
    recurrences = [int(item) for item in value.split(",") if item]
    if not recurrences or any(item < 1 for item in recurrences):
        raise argparse.ArgumentTypeError("expected comma-separated positive recurrence counts")
    return recurrences


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("multiplication", "squaring"), default="squaring")
    parser.add_argument("--preset", choices=("easy", "medium"), required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--min_train_recurrences", type=int, default=2)
    parser.add_argument("--max_train_recurrences", type=int, default=4)
    parser.add_argument(
        "--eval_recurrences",
        type=parse_recurrences,
        default=parse_recurrences("1,2,3,4,6,8"),
    )
    parser.add_argument("--example_count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=74)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.steps < 1 or args.layers < 1 or args.min_train_recurrences < 1:
        raise ValueError("steps, layers, and min_train_recurrences must be positive")
    if args.max_train_recurrences < args.min_train_recurrences:
        raise ValueError("max_train_recurrences must be at least min_train_recurrences")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    root = ROOT / "controlled_experiments" / "data" / f"{args.task}_{args.preset}"
    datasets = {"train": MultiplicationDataset(root / "train.jsonl")}
    for split in ("test", "ood_long", "ood_one_long", "ood_both_long"):
        path = root / f"{split}.jsonl"
        if path.exists():
            datasets[split] = MultiplicationDataset(path)
    loaders = {
        split: DataLoader(dataset, args.batch_size, collate_fn=collate)
        for split, dataset in datasets.items()
    }
    train_loader = DataLoader(
        datasets["train"], args.batch_size, shuffle=True, drop_last=True, collate_fn=collate
    )
    iterator = iter(train_loader)
    max_seq_len = max(
        len(record["input_ids"])
        for dataset in datasets.values()
        for record in dataset.records
    )
    model = ContextReinjectionTransformer(
        ExperimentSource,
        ModelSpec(17, max_seq_len, 500_000_000),
        args.layers,
        args.max_train_recurrences,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1
    )
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
    started = time.monotonic()
    model.train()
    for step in range(1, args.steps + 1):
        try:
            host_batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            host_batch = next(iterator)
        batch = {name: value.to(device) for name, value in host_batch.items()}
        model.rounds = int(
            torch.randint(args.min_train_recurrences, args.max_train_recurrences + 1, ()).item()
        )
        optimizer.zero_grad(set_to_none=True)
        with autocast:
            loss, _, _ = loss_and_predictions(model, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 100 == 0:
            print(
                json.dumps(
                    {
                        "step": step,
                        "loss": loss.item(),
                        "recurrences": model.rounds,
                        "seconds": time.monotonic() - started,
                    }
                )
            )

    sweep: dict[str, object] = {}
    for recurrences in args.eval_recurrences:
        model.rounds = recurrences
        sweep[str(recurrences)] = {
            split: evaluate(model, loader, device, args.example_count)
            for split, loader in loaders.items()
        }
    summary = {
        "task": args.task,
        "preset": args.preset,
        "architecture": "variable_recurrence_context_reinjection",
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "layers_per_macro_step": args.layers,
        "train_recurrences": [args.min_train_recurrences, args.max_train_recurrences],
        "eval_recurrences": args.eval_recurrences,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "evaluation": sweep,
    }
    output = ROOT / "controlled_experiments" / "results" / (
        f"{args.task}_{args.preset}_variable_recurrence_l{args.layers}"
        f"_train{args.min_train_recurrences}-{args.max_train_recurrences}"
        f"_s{args.steps}_seed{args.seed}.json"
    )
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
