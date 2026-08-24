"""Train and evaluate controlled decimal arithmetic tasks."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F
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
    z_values = torch.full((batch,), -1, dtype=torch.long)
    modulus_values = torch.full((batch,), -1, dtype=torch.long)
    quotient_values = torch.full((batch,), -1, dtype=torch.long)
    z_digits = torch.full((batch,), -1, dtype=torch.long)
    modulus_digits = torch.full((batch,), -1, dtype=torch.long)
    quotient_digits = torch.full((batch,), -1, dtype=torch.long)
    x_values = torch.full((batch,), -1, dtype=torch.long)
    x_digits = torch.full((batch,), -1, dtype=torch.long)
    for row, record in enumerate(records):
        inputs = torch.tensor(record["input_ids"], dtype=torch.long)
        targets = torch.tensor(record["labels"], dtype=torch.long)
        input_ids[row, : inputs.numel()] = inputs
        labels[row, : targets.numel()] = targets
        mask[row, : inputs.numel()] = True
        positions[row, : targets.numel()] = torch.arange(
            inputs.numel() - targets.numel(), inputs.numel()
        )
        a_values[row] = int(record.get("a", -1))
        b_values[row] = int(record.get("b", -1))
        if "x" in record:
            x_values[row] = int(record["x"])
            x_digits[row] = int(record["x_digits"])
        if "z" in record:
            z_values[row] = int(record["z"])
            modulus_values[row] = int(record["modulus"])
            quotient_values[row] = int(record["quotient"])
            z_digits[row] = int(record["z_digits"])
            modulus_digits[row] = int(record["modulus_digits"])
            quotient_digits[row] = int(record["quotient_digits"])
    return {
        "input_ids": input_ids,
        "labels": labels,
        "mask": mask,
        "positions": positions,
        "a": a_values,
        "b": b_values,
        "z": z_values,
        "modulus": modulus_values,
        "quotient": quotient_values,
        "z_digits": z_digits,
        "modulus_digits": modulus_digits,
        "quotient_digits": quotient_digits,
        "x": x_values,
        "x_digits": x_digits,
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


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.shape[-1] // 2
    rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return x * cos + rotated * sin


class LocalRMSNorm(torch.nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(width))

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.shape[-1],), self.weight)


class FullTokenBlock(torch.nn.Module):
    """One normal bidirectional Transformer block, independent of submission code."""

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
        context: Tensor,
        attention_mask: Tensor | None,
        rope: tuple[Tensor, Tensor],
    ) -> Tensor:
        residual = context
        context = self.attention_norm(context)
        batch, length, _ = context.shape
        q, k, v = self.qkv(context).chunk(3, dim=-1)
        q = q.view(batch, length, self.heads, -1).transpose(1, 2)
        k = k.view(batch, length, self.heads, -1).transpose(1, 2)
        v = v.view(batch, length, self.heads, -1).transpose(1, 2)
        cos, sin = rope
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        mask = attention_mask[:, None, None, :].bool() if attention_mask is not None else None
        context = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        context = context.transpose(1, 2).contiguous().view(batch, length, self.width)
        context = residual + self.out(context)
        return context + self.down(F.gelu(self.up(self.mixer_norm(context))))


class FullTokenTransformer(torch.nn.Module):
    """A conventional bidirectional Transformer for the learnability check."""

    def __init__(
        self,
        source,
        spec: ModelSpec,
        layers: int,
        rounds: int = 1,
        use_round_embeddings: bool = False,
    ) -> None:
        super().__init__()
        self.source = source
        self.rounds = rounds
        self.round_embeddings = (
            torch.nn.Parameter(torch.empty(rounds, source.D_MODEL))
            if use_round_embeddings
            else None
        )
        if self.round_embeddings is not None:
            torch.nn.init.normal_(self.round_embeddings, std=0.02)
        self.config = source.Config(spec.vocab_size, spec.max_seq_len)
        self.token_embedding = torch.nn.Embedding(spec.vocab_size, source.D_MODEL)
        self.blocks = torch.nn.ModuleList(
            FullTokenBlock(source.D_MODEL, source.NUM_HEADS) for _ in range(layers)
        )
        self.final_norm = LocalRMSNorm(source.D_MODEL)
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
        for round_index in range(self.rounds):
            if self.round_embeddings is not None:
                context = context + self.round_embeddings[round_index]
            for block in self.blocks:
                context = block(context, attention_mask, rope)
        return self.head(self.final_norm(context)), None


def loss_and_predictions(
    model,
    batch: dict[str, Tensor],
    loss_kind: str = "token_ce",
) -> tuple[Tensor, Tensor, Tensor]:
    logits, _ = model(batch["input_ids"], attention_mask=batch["mask"])
    row = torch.arange(logits.shape[0], device=logits.device)[:, None]
    selected = logits[row, batch["positions"].clamp_min(0)]
    valid = batch["labels"] != -100
    token_losses = F.cross_entropy(
        selected.transpose(1, 2),
        batch["labels"],
        ignore_index=-100,
        reduction="none",
    )
    if loss_kind == "token_ce":
        loss = token_losses[valid].mean()
    elif loss_kind == "hard_sequence_05":
        counts = valid.sum(dim=1)
        sequence_losses = (
            (token_losses * valid).sum(dim=1) / counts.clamp_min(1)
        )[counts > 0]
        weights = sequence_losses.detach().clamp_min(1e-8).sqrt()
        weights = weights / weights.mean().clamp_min(1e-8)
        loss = (weights * sequence_losses).mean()
    else:
        raise ValueError(f"unknown loss kind: {loss_kind}")
    return loss, selected.argmax(dim=-1), valid


def _bucket_metrics(
    totals: dict[int, list[int]],
    key: int,
    exact_correct: int,
    digit_correct: int,
    digit_total: int,
) -> None:
    bucket = totals.setdefault(key, [0, 0, 0, 0])
    bucket[0] += exact_correct
    bucket[1] += 1
    bucket[2] += digit_correct
    bucket[3] += digit_total


def _finish_buckets(totals: dict[int, list[int]]) -> dict[str, dict[str, float | int]]:
    return {
        str(key): {
            "examples": values[1],
            "exact_accuracy": values[0] / values[1],
            "digit_accuracy": values[2] / values[3],
        }
        for key, values in sorted(totals.items())
    }


@torch.no_grad()
def evaluate(
    model,
    loader: DataLoader,
    device: torch.device,
    example_count: int,
    task: str = "multiplication",
    loss_kind: str = "token_ce",
) -> dict[str, object]:
    model.eval()
    exact = 0
    examples = 0
    correct_digits = 0
    digit_count = 0
    passed: list[dict[str, object]] = []
    failed: list[dict[str, object]] = []
    position_left: dict[int, list[int]] = {}
    position_right: dict[int, list[int]] = {}
    quotient_digit_buckets: dict[int, list[int]] = {}
    z_digit_buckets: dict[int, list[int]] = {}
    modulus_digit_buckets: dict[int, list[int]] = {}
    x_digit_buckets: dict[int, list[int]] = {}
    context = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
    for host_batch in loader:
        batch = {name: value.to(device) for name, value in host_batch.items()}
        with context:
            _, prediction, valid = loss_and_predictions(model, batch, loss_kind)
        matches = (prediction == batch["labels"]) | ~valid
        exact += matches.all(dim=1).sum().item()
        examples += prediction.shape[0]
        correct_digits += ((prediction == batch["labels"]) & valid).sum().item()
        digit_count += valid.sum().item()
        for row in range(prediction.shape[0]):
            is_correct = matches[row].all().item()
            row_valid = valid[row]
            row_matches = prediction[row, row_valid] == batch["labels"][row, row_valid]
            row_digit_count = int(row_valid.sum().item())
            row_digit_correct = int(row_matches.sum().item())
            for position, correct in enumerate(row_matches.tolist()):
                left = position_left.setdefault(position, [0, 0])
                left[0] += int(correct)
                left[1] += 1
                from_right = row_digit_count - position - 1
                right = position_right.setdefault(from_right, [0, 0])
                right[0] += int(correct)
                right[1] += 1
            if task in ("reduction", "square_mod"):
                exact_value = int(is_correct)
                _bucket_metrics(
                    quotient_digit_buckets,
                    int(batch["quotient_digits"][row].item()),
                    exact_value,
                    row_digit_correct,
                    row_digit_count,
                )
                _bucket_metrics(
                    z_digit_buckets,
                    int(batch["z_digits"][row].item()),
                    exact_value,
                    row_digit_correct,
                    row_digit_count,
                )
                _bucket_metrics(
                    modulus_digit_buckets,
                    int(batch["modulus_digits"][row].item()),
                    exact_value,
                    row_digit_correct,
                    row_digit_count,
                )
                if task == "square_mod":
                    _bucket_metrics(
                        x_digit_buckets,
                        int(batch["x_digits"][row].item()),
                        exact_value,
                        row_digit_correct,
                        row_digit_count,
                    )
            if len(passed if is_correct else failed) >= example_count:
                continue
            valid_tokens = row_valid
            predicted = "".join(
                str(token - 7) if 7 <= token <= 16 else f"[{token}]"
                for token in prediction[row, valid_tokens].tolist()
            )
            if task == "reduction":
                example_inputs = {
                    "z": batch["z"][row].item(),
                    "modulus": batch["modulus"][row].item(),
                    "quotient": batch["quotient"][row].item(),
                }
                mathematical_answer = str(
                    batch["z"][row].item() % batch["modulus"][row].item()
                )
            elif task == "square_mod":
                example_inputs = {
                    "x": batch["x"][row].item(),
                    "modulus": batch["modulus"][row].item(),
                }
                mathematical_answer = str(
                    pow(batch["x"][row].item(), 2, batch["modulus"][row].item())
                )
            else:
                example_inputs = {
                    "a": batch["a"][row].item(),
                    "b": batch["b"][row].item(),
                }
                mathematical_answer = str(
                    batch["a"][row].item() * batch["b"][row].item()
                )
            example = {
                **example_inputs,
                "mathematical_answer": mathematical_answer,
                "expected_tokens": "".join(
                    str(token - 7)
                    for token in batch["labels"][row, valid_tokens].tolist()
                ),
                "predicted": predicted,
            }
            (passed if is_correct else failed).append(example)
    result = {
        "exact_accuracy": exact / examples,
        "digit_accuracy": correct_digits / digit_count,
        "accuracy_by_position_from_left": {
            str(key): {"correct": values[0], "total": values[1], "accuracy": values[0] / values[1]}
            for key, values in sorted(position_left.items())
        },
        "accuracy_by_position_from_right": {
            str(key): {"correct": values[0], "total": values[1], "accuracy": values[0] / values[1]}
            for key, values in sorted(position_right.items())
        },
        "passed_examples": passed,
        "failed_examples": failed,
    }
    if task == "reduction":
        result.update(
            {
                "accuracy_by_quotient_digits": _finish_buckets(quotient_digit_buckets),
                "accuracy_by_z_digits": _finish_buckets(z_digit_buckets),
                "accuracy_by_modulus_digits": _finish_buckets(modulus_digit_buckets),
            }
        )
    elif task == "square_mod":
        result.update(
            {
                "accuracy_by_quotient_digits": _finish_buckets(quotient_digit_buckets),
                "accuracy_by_square_digits": _finish_buckets(z_digit_buckets),
                "accuracy_by_modulus_digits": _finish_buckets(modulus_digit_buckets),
                "accuracy_by_x_digits": _finish_buckets(x_digit_buckets),
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=("multiplication", "squaring", "reduction", "square_mod"),
        default="multiplication",
    )
    parser.add_argument("--preset", choices=("easy", "medium"), required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument(
        "--loss",
        choices=("token_ce", "hard_sequence_05"),
        default="token_ce",
    )
    parser.add_argument(
        "--architecture",
        choices=("scratchpad", "full_token", "full_token_recurrent"),
        default="scratchpad",
    )
    parser.add_argument("--full_token_layers", type=int, default=8)
    parser.add_argument("--round_embeddings", action="store_true")
    parser.add_argument("--untie_embeddings", action="store_true")
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
    root = ROOT / "controlled_experiments" / "data" / f"{args.task}_{args.preset}"
    train = MultiplicationDataset(root / "train.jsonl")
    datasets = {"train": train}
    for split in (
        "test",
        "test_seen_n",
        "ood_long",
        "ood_one_long",
        "ood_z_long",
        "ood_n_long",
        "ood_both_long",
    ):
        path = root / f"{split}.jsonl"
        if path.exists():
            datasets[split] = MultiplicationDataset(path)
    loaders = {
        split: DataLoader(dataset, args.batch_size, collate_fn=collate)
        for split, dataset in datasets.items()
    }
    train_loader = DataLoader(train, args.batch_size, shuffle=True, drop_last=True, collate_fn=collate)
    iterator = iter(train_loader)

    submission = load_submission()
    submission.NUM_SCRATCH_TOKENS = args.scratchpad_slots
    submission.NUM_RECURRENCES = args.recurrences
    submission.NUM_PROMPT_READER_LAYERS = args.prompt_reader_layers
    if args.untie_embeddings:
        submission.TIE_EMBEDDINGS = False
    max_seq_len = max(
        len(record["input_ids"])
        for dataset in datasets.values()
        for record in dataset.records
    )
    model_spec = ModelSpec(17, max_seq_len, 500_000_000)
    if args.architecture == "scratchpad":
        model = submission.build_model(model_spec)
    else:
        rounds = args.recurrences if args.architecture == "full_token_recurrent" else 1
        model = FullTokenTransformer(
            submission,
            model_spec,
            args.full_token_layers,
            rounds,
            args.round_embeddings,
        )
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
            loss, _, _ = loss_and_predictions(model, batch, args.loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 100 == 0:
            print(json.dumps({"step": step, "loss": loss.item(), "seconds": time.monotonic() - started}))

    summary = {
        "task": args.task,
        "preset": args.preset,
        "architecture": args.architecture,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "loss": args.loss,
        "scratchpad_slots": args.scratchpad_slots,
        "recurrences": args.recurrences,
        "prompt_reader_layers": args.prompt_reader_layers,
        "full_token_layers": args.full_token_layers,
        "round_embeddings": args.round_embeddings,
        "untie_embeddings": args.untie_embeddings,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    summary.update(
        {
            split: evaluate(
                model,
                loader,
                device,
                args.example_count,
                args.task,
                args.loss,
            )
            for split, loader in loaders.items()
        }
    )
    output = ROOT / "controlled_experiments" / "results" / (
        f"{args.task}_{args.preset}_{args.architecture}_slots{args.scratchpad_slots}_r{args.recurrences}"
        f"_reader{args.prompt_reader_layers}_clock{int(args.round_embeddings)}"
        f"_loss{args.loss}_untied{int(args.untie_embeddings)}"
        f"_s{args.steps}_seed{args.seed}.json"
    )
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
