"""Test recurrent full-token updates with a fixed copy of the original prompt."""

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


class ContextReinjectionBlock(torch.nn.Module):
    """Updates work tokens by reading both work and frozen input tokens.

    It has the same parameter shapes as a normal full-token Transformer block.
    The only change is that its keys/values concatenate fixed context and
    mutable work, rather than using mutable work alone.
    """

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
        work: Tensor,
        fixed_context: Tensor,
        attention_mask: Tensor | None,
        rope: tuple[Tensor, Tensor],
    ) -> Tensor:
        residual = work
        work = self.attention_norm(work)
        batch, length, _ = work.shape
        q = self.qkv(work)[..., : self.width]
        kv = self.qkv(torch.cat((fixed_context, work), dim=1))
        k, v = kv[..., self.width : 2 * self.width], kv[..., 2 * self.width :]
        q = q.view(batch, length, self.heads, -1).transpose(1, 2)
        k = k.view(batch, 2 * length, self.heads, -1).transpose(1, 2)
        v = v.view(batch, 2 * length, self.heads, -1).transpose(1, 2)
        cos, sin = rope
        q = apply_rope(q, cos, sin)
        # Fixed and work tokens deliberately share their input positions.
        k = apply_rope(k, torch.cat((cos, cos), dim=0), torch.cat((sin, sin), dim=0))
        if attention_mask is not None:
            key_mask = torch.cat((attention_mask.bool(), attention_mask.bool()), dim=1)
            mask = key_mask[:, None, None, :]
        else:
            mask = None
        work = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        work = work.transpose(1, 2).contiguous().view(batch, length, self.width)
        work = residual + self.out(work)
        return work + self.down(F.gelu(self.up(self.mixer_norm(work))))


class ContextReinjectionTransformer(torch.nn.Module):
    def __init__(self, source, spec: ModelSpec, layers: int, rounds: int) -> None:
        super().__init__()
        self.source = source
        self.rounds = rounds
        self.config = source.Config(spec.vocab_size, spec.max_seq_len)
        self.token_embedding = torch.nn.Embedding(spec.vocab_size, source.D_MODEL)
        self.blocks = torch.nn.ModuleList(
            ContextReinjectionBlock(source.D_MODEL, source.NUM_HEADS)
            for _ in range(layers)
        )
        self.final_norm = LocalRMSNorm(source.D_MODEL)
        self.head = torch.nn.Linear(source.D_MODEL, spec.vocab_size, bias=False)
        if source.TIE_EMBEDDINGS:
            self.head.weight = self.token_embedding.weight
        if source.EMBED_INIT_STD is not None:
            torch.nn.init.normal_(self.token_embedding.weight, std=source.EMBED_INIT_STD)

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None):
        fixed_context = self.token_embedding(input_ids)
        work = fixed_context.clone()
        rope = self.source.rope_tables(
            input_ids.shape[1],
            self.source.D_MODEL // self.source.NUM_HEADS,
            work.device,
            work.dtype,
        )
        for _ in range(self.rounds):
            for block in self.blocks:
                work = block(work, fixed_context, attention_mask, rope)
        return self.head(self.final_norm(work)), None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("multiplication", "squaring"), default="squaring")
    parser.add_argument("--preset", choices=("easy", "medium"), required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--recurrences", type=int, default=4)
    parser.add_argument("--example_count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=74)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.steps < 1 or args.layers < 1 or args.recurrences < 1:
        raise ValueError("steps, layers, and recurrences must be positive")

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
    model = ContextReinjectionTransformer(
        source, ModelSpec(17, max_seq_len, 500_000_000), args.layers, args.recurrences
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
        "architecture": "context_reinjection",
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "layers": args.layers,
        "recurrences": args.recurrences,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    summary.update(
        {split: evaluate(model, loader, device, args.example_count) for split, loader in loaders.items()}
    )
    output = ROOT / "controlled_experiments" / "results" / (
        f"{args.task}_{args.preset}_context_reinjection_l{args.layers}"
        f"_r{args.recurrences}_s{args.steps}_seed{args.seed}.json"
    )
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
