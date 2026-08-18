"""Test a two-tape recurrent pipeline: input -> product -> accumulator."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark import ModelSpec
from controlled_experiments.run_multiplication import (
    LocalRMSNorm,
    MultiplicationDataset,
    apply_rope,
    collate,
    evaluate,
    load_submission,
    loss_and_predictions,
)


class TapeBlock(torch.nn.Module):
    """Updates one tape by cross-attending to an ordered list of sources."""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.width = width
        self.heads = heads
        self.attention_norm = LocalRMSNorm(width)
        self.qkv = torch.nn.Linear(width, 3 * width)
        self.out = torch.nn.Linear(width, width)
        self.mixer_norm = LocalRMSNorm(width)
        self.up = torch.nn.Linear(width, 4 * width)
        self.down = torch.nn.Linear(4 * width, width)

    def forward(
        self,
        tape: Tensor,
        sources: tuple[Tensor, ...],
        attention_mask: Tensor | None,
        rope: tuple[Tensor, Tensor],
    ) -> Tensor:
        residual = tape
        tape = self.attention_norm(tape)
        batch, length, _ = tape.shape
        q = self.qkv(tape)[..., : self.width]
        kv = self.qkv(torch.cat(sources, dim=1))
        k, v = kv[..., self.width : 2 * self.width], kv[..., 2 * self.width :]
        q = q.view(batch, length, self.heads, -1).transpose(1, 2)
        k = k.view(batch, len(sources) * length, self.heads, -1).transpose(1, 2)
        v = v.view(batch, len(sources) * length, self.heads, -1).transpose(1, 2)
        cos, sin = rope
        q = apply_rope(q, cos, sin)
        k = apply_rope(
            k,
            torch.cat((cos,) * len(sources), dim=0),
            torch.cat((sin,) * len(sources), dim=0),
        )
        if attention_mask is not None:
            key_mask = torch.cat((attention_mask.bool(),) * len(sources), dim=1)
            mask = key_mask[:, None, None, :]
        else:
            mask = None
        tape = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        tape = tape.transpose(1, 2).contiguous().view(batch, length, self.width)
        tape = residual + self.out(tape)
        return tape + self.down(F.gelu(self.up(self.mixer_norm(tape))))


class TwoTapeTransformer(torch.nn.Module):
    """Two shared blocks with a one-way product-to-accumulator interface."""

    def __init__(self, source, spec: ModelSpec, recurrences: int) -> None:
        super().__init__()
        self.source = source
        self.recurrences = recurrences
        self.config = source.Config(spec.vocab_size, spec.max_seq_len)
        self.token_embedding = torch.nn.Embedding(spec.vocab_size, source.D_MODEL)
        self.product_block = TapeBlock(source.D_MODEL, source.NUM_HEADS)
        self.accumulator_block = TapeBlock(source.D_MODEL, source.NUM_HEADS)
        self.final_norm = LocalRMSNorm(source.D_MODEL)
        self.head = torch.nn.Linear(source.D_MODEL, spec.vocab_size, bias=False)
        if source.TIE_EMBEDDINGS:
            self.head.weight = self.token_embedding.weight
        if source.EMBED_INIT_STD is not None:
            torch.nn.init.normal_(self.token_embedding.weight, std=source.EMBED_INIT_STD)

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None):
        fixed_context = self.token_embedding(input_ids)
        product_tape = fixed_context.clone()
        accumulator_tape = fixed_context.clone()
        rope = self.source.rope_tables(
            input_ids.shape[1],
            self.source.D_MODEL // self.source.NUM_HEADS,
            fixed_context.device,
            fixed_context.dtype,
        )
        for _ in range(self.recurrences):
            # P can use input and its own prior state, but not A.
            product_tape = self.product_block(
                product_tape, (fixed_context, product_tape), attention_mask, rope
            )
            # A can use input, the freshly updated P, and its own prior state.
            accumulator_tape = self.accumulator_block(
                accumulator_tape,
                (fixed_context, product_tape, accumulator_tape),
                attention_mask,
                rope,
            )
        # The accumulator tape directly supplies the answer-slot logits.
        return self.head(self.final_norm(accumulator_tape)), None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("multiplication", "squaring"), default="squaring")
    parser.add_argument("--preset", choices=("easy", "medium"), required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--recurrences", type=int, default=4)
    parser.add_argument("--example_count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=74)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.steps < 1 or args.recurrences < 1:
        raise ValueError("steps and recurrences must be positive")

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
    source = load_submission()
    model = TwoTapeTransformer(
        source, ModelSpec(17, max_seq_len, 500_000_000), args.recurrences
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
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
        optimizer.zero_grad(set_to_none=True)
        with autocast:
            loss, _, _ = loss_and_predictions(model, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 100 == 0:
            print(json.dumps({"step": step, "loss": loss.item(), "seconds": time.monotonic() - started}))

    summary = {
        "task": args.task,
        "preset": args.preset,
        "architecture": "two_tape",
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "recurrences": args.recurrences,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    summary.update(
        {split: evaluate(model, loader, device, args.example_count) for split, loader in loaders.items()}
    )
    output = ROOT / "controlled_experiments" / "results" / (
        f"{args.task}_{args.preset}_two_tape_r{args.recurrences}"
        f"_s{args.steps}_seed{args.seed}.json"
    )
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
