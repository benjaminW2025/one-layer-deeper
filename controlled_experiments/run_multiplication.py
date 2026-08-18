"""Train and evaluate the current scratchpad architecture on multiplication."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path
import sys
import time

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
# Allow `python path/to/run_multiplication.py ...` from a remote shell. Python
# otherwise adds only controlled_experiments/ to sys.path, not the repo root.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark import ModelSpec


class MultiplicationDataset(Dataset):
    def __init__(self, path: Path) -> None:
        with path.open("r", encoding="utf-8") as handle:
            self.records = [json.loads(line) for line in handle if line.strip()]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        return self.records[index]


def collate(records: list[dict[str, object]]) -> dict[str, Tensor]:
    input_length = max(len(record["input_ids"]) for record in records)
    target_length = max(len(record["labels"]) for record in records)
    batch = len(records)
    input_ids = torch.zeros((batch, input_length), dtype=torch.long)
    labels = torch.full((batch, target_length), -100, dtype=torch.long)
    mask = torch.zeros((batch, input_length), dtype=torch.bool)
    positions = torch.full((batch, target_length), -1, dtype=torch.long)
    a_values = torch.empty(batch, dtype=torch.long)
    b_values = torch.empty(batch, dtype=torch.long)
    for row, record in enumerate(records):
        inputs = torch.tensor(record["input_ids"], dtype=torch.long)
        targets = torch.tensor(record["labels"], dtype=torch.long)
        input_ids[row, : inputs.numel()] = inputs
        labels[row, : targets.numel()] = targets
        mask[row, : inputs.numel()] = True
        positions[row, : targets.numel()] = torch.arange(
            inputs.numel() - targets.numel(), inputs.numel()
        )
        a_values[row] = int(record["a"])
        b_values[row] = int(record["b"])
    return {
        "input_ids": input_ids,
        "labels": labels,
        "mask": mask,
        "positions": positions,
        "a": a_values,
        "b": b_values,
    }


def load_submission():
    path = ROOT / "submissions" / "baseline_adamw" / "submission.py"
    spec = importlib.util.spec_from_file_location("controlled_submission", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Compilation is a submission-time variable, not part of this diagnostic.
    module.COMPILE_MODE = None
    return module


class FullTokenTransformer(torch.nn.Module):
    """A conventional bidirectional Transformer for the learnability check."""

    def __init__(self, source, spec: ModelSpec, layers: int) -> None:
        super().__init__()
        self.source = source
        self.config = source.Config(spec.vocab_size, spec.max_seq_len)
        self.token_embedding = torch.nn.Embedding(spec.vocab_size, source.D_MODEL)
        self.blocks = torch.nn.ModuleList(source.PromptReaderBlock() for _ in range(layers))
        self.final_norm = source.RMSNorm(source.D_MODEL)
        self.head = torch.nn.Linear(source.D_MODEL, spec.vocab_size, bias=False)
        if source.TIE_EMBEDDINGS:
            self.head.weight = self.token_embedding.weight
        if source.EMBED_INIT_STD is not None:
            torch.nn.init.normal_(self.token_embedding.weight, std=source.EMBED_INIT_STD)

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None):
        context = self.token_embedding(input_ids)
        rope = self.source.rope_tables(
            input_ids.shape[1],
            self.source.D_MODEL // self.source.NUM_HEADS,
            context.device,
            context.dtype,
        )
        for block in self.blocks:
            context = block(context, attention_mask, rope)
        return self.head(self.final_norm(context)), None


def loss_and_predictions(model, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
    logits, _ = model(batch["input_ids"], attention_mask=batch["mask"])
    row = torch.arange(logits.shape[0], device=logits.device)[:, None]
    selected = logits[row, batch["positions"].clamp_min(0)]
    valid = batch["labels"] != -100
    loss = torch.nn.functional.cross_entropy(selected[valid], batch["labels"][valid])
    return loss, selected.argmax(dim=-1), valid


@torch.no_grad()
def evaluate(model, loader: DataLoader, device: torch.device, example_count: int) -> dict[str, object]:
    model.eval()
    exact = 0
    examples = 0
    correct_digits = 0
    digit_count = 0
    passed: list[dict[str, object]] = []
    failed: list[dict[str, object]] = []
    context = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
    for host_batch in loader:
        batch = {name: value.to(device) for name, value in host_batch.items()}
        with context:
            _, prediction, valid = loss_and_predictions(model, batch)
        matches = (prediction == batch["labels"]) | ~valid
        exact += matches.all(dim=1).sum().item()
        examples += prediction.shape[0]
        correct_digits += ((prediction == batch["labels"]) & valid).sum().item()
        digit_count += valid.sum().item()
        for row in range(prediction.shape[0]):
            is_correct = matches[row].all().item()
            if len(passed if is_correct else failed) >= example_count:
                continue
            valid_tokens = valid[row]
            predicted = "".join(
                str(token - 7) if 7 <= token <= 16 else f"[{token}]"
                for token in prediction[row, valid_tokens].tolist()
            )
            example = {
                "a": batch["a"][row].item(),
                "b": batch["b"][row].item(),
                "expected": str(batch["a"][row].item() * batch["b"][row].item()),
                "predicted": predicted,
            }
            (passed if is_correct else failed).append(example)
    return {
        "exact_accuracy": exact / examples,
        "digit_accuracy": correct_digits / digit_count,
        "passed_examples": passed,
        "failed_examples": failed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("easy", "medium"), required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--architecture", choices=("scratchpad", "full_token"), default="scratchpad")
    parser.add_argument("--full_token_layers", type=int, default=8)
    parser.add_argument("--scratchpad_slots", type=int, default=4)
    parser.add_argument("--recurrences", type=int, default=4)
    parser.add_argument("--prompt_reader_layers", type=int, choices=(0, 1), default=0)
    parser.add_argument("--example_count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=74)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be positive")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    root = ROOT / "controlled_experiments" / "data" / f"multiplication_{args.preset}"
    train = MultiplicationDataset(root / "train.jsonl")
    loaders = {
        "train": DataLoader(train, args.batch_size, collate_fn=collate),
        "test": DataLoader(MultiplicationDataset(root / "test.jsonl"), args.batch_size, collate_fn=collate),
        "ood_one_long": DataLoader(MultiplicationDataset(root / "ood_one_long.jsonl"), args.batch_size, collate_fn=collate),
        "ood_both_long": DataLoader(MultiplicationDataset(root / "ood_both_long.jsonl"), args.batch_size, collate_fn=collate),
    }
    train_loader = DataLoader(train, args.batch_size, shuffle=True, drop_last=True, collate_fn=collate)
    iterator = iter(train_loader)

    submission = load_submission()
    submission.NUM_SCRATCH_TOKENS = args.scratchpad_slots
    submission.NUM_RECURRENCES = args.recurrences
    submission.NUM_PROMPT_READER_LAYERS = args.prompt_reader_layers
    model_spec = ModelSpec(17, 14, 500_000_000)
    if args.architecture == "scratchpad":
        model = submission.build_model(model_spec)
    else:
        model = FullTokenTransformer(submission, model_spec, args.full_token_layers)
    model = model.to(device)
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
        "preset": args.preset,
        "architecture": args.architecture,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "scratchpad_slots": args.scratchpad_slots,
        "recurrences": args.recurrences,
        "prompt_reader_layers": args.prompt_reader_layers,
        "full_token_layers": args.full_token_layers,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    summary.update(
        {
            split: evaluate(model, loader, device, args.example_count)
            for split, loader in loaders.items()
        }
    )
    output = ROOT / "controlled_experiments" / "results" / (
        f"multiply_{args.preset}_{args.architecture}_slots{args.scratchpad_slots}_r{args.recurrences}"
        f"_reader{args.prompt_reader_layers}_s{args.steps}_seed{args.seed}.json"
    )
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
