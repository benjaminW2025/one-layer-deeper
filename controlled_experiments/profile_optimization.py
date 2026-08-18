"""Profile optimization of a full-token Transformer or its recurrent variant.

The default is the current four-distinct-block, two-repeat reference.  It
writes JSONL records during training, so an interrupted run still preserves the
profile collected so far.
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
from controlled_experiments.run_multiplication import (
    FullTokenTransformer,
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
    """Controlled-model settings, kept independent of competition submission.py."""

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


def parse_positive_csv(value: str) -> list[int]:
    values = [int(item) for item in value.split(",") if item]
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def rms(tensor: Tensor) -> float:
    return tensor.detach().float().square().mean().sqrt().item()


def parameter_group(name: str) -> str:
    if name.startswith("blocks."):
        return "block_" + name.split(".")[1]
    if name.startswith("token_embedding"):
        return "embedding"
    if name.startswith("head"):
        return "head"
    return "final_norm"


def aggregate_parameter_metrics(
    named_parameters: list[tuple[str, Tensor]],
    before_step: dict[str, Tensor],
) -> dict[str, dict[str, float]]:
    totals: dict[str, dict[str, float]] = {}
    for name, parameter in named_parameters:
        group = parameter_group(name)
        values = totals.setdefault(group, {"count": 0.0, "grad_sq": 0.0, "param_sq": 0.0, "update_sq": 0.0})
        values["count"] += parameter.numel()
        values["param_sq"] += parameter.detach().float().square().sum().item()
        if parameter.grad is not None:
            values["grad_sq"] += parameter.grad.detach().float().square().sum().item()
        update = parameter.detach().float() - before_step[name]
        values["update_sq"] += update.square().sum().item()
    return {
        group: {
            "gradient_rms": (values["grad_sq"] / values["count"]) ** 0.5,
            "parameter_rms": (values["param_sq"] / values["count"]) ** 0.5,
            "update_rms": (values["update_sq"] / values["count"]) ** 0.5,
            "update_to_parameter": (
                (values["update_sq"] / values["param_sq"]) ** 0.5
                if values["param_sq"] > 0
                else 0.0
            ),
        }
        for group, values in totals.items()
    }


def norm_parameter_metrics(
    named_parameters: list[tuple[str, Tensor]], before_step: dict[str, Tensor]
) -> dict[str, dict[str, float]]:
    """Keep the learned RMSNorm scales separate from their surrounding blocks."""
    output = {}
    for name, parameter in named_parameters:
        if not (name.endswith("attention_norm.weight") or name.endswith("mixer_norm.weight") or name == "final_norm.weight"):
            continue
        update = parameter.detach().float() - before_step[name]
        output[name] = {
            "parameter_rms": rms(parameter),
            "gradient_rms": rms(parameter.grad) if parameter.grad is not None else 0.0,
            "update_rms": rms(update),
        }
    return output


def gradient_alignment(
    model: FullTokenTransformer,
    batch: dict[str, Tensor],
    horizons: list[int],
    autocast_context,
) -> dict[str, object]:
    """Compare gradients from answers read after different recurrence counts."""
    original_rounds = model.rounds
    named_parameters = list(model.named_parameters())
    gradient_sets: dict[int, dict[str, list[Tensor | None]]] = {}
    losses: dict[str, float] = {}
    gradient_norms: dict[str, float] = {}
    for horizon in horizons:
        model.rounds = horizon
        with autocast_context():
            loss, _, _ = loss_and_predictions(model, batch)
        gradients = torch.autograd.grad(loss, [parameter for _, parameter in named_parameters])
        grouped: dict[str, list[Tensor | None]] = {}
        for (name, _), gradient in zip(named_parameters, gradients, strict=True):
            grouped.setdefault(parameter_group(name), []).append(gradient.detach())
        gradient_sets[horizon] = grouped
        losses[str(horizon)] = loss.item()
        gradient_norms[str(horizon)] = sum(
            gradient.detach().float().square().sum().item() for gradient in gradients
        ) ** 0.5
    model.rounds = original_rounds

    alignment: dict[str, dict[str, float]] = {}
    first = horizons[0]
    for later in horizons[1:]:
        pair = f"{first}_vs_{later}"
        alignment[pair] = {}
        for group, first_gradients in gradient_sets[first].items():
            dot = 0.0
            first_sq = 0.0
            later_sq = 0.0
            for first_gradient, later_gradient in zip(
                first_gradients, gradient_sets[later][group], strict=True
            ):
                if first_gradient is None or later_gradient is None:
                    continue
                dot += (first_gradient.float() * later_gradient.float()).sum().item()
                first_sq += first_gradient.float().square().sum().item()
                later_sq += later_gradient.float().square().sum().item()
            alignment[pair][group] = dot / ((first_sq * later_sq) ** 0.5 + 1e-12)
    return {
        "loss_by_horizon": losses,
        "gradient_norm_by_horizon": gradient_norms,
        "cosine_by_group": alignment,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("multiplication", "squaring"), default="squaring")
    parser.add_argument("--preset", choices=("easy", "medium"), required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--architecture", choices=("recurrent", "full_token"), default="recurrent")
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--recurrences", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--probe_every", type=int, default=500)
    parser.add_argument("--probe_horizons", type=parse_positive_csv, default=parse_positive_csv("1,2"))
    parser.add_argument("--round_embeddings", action="store_true")
    parser.add_argument("--example_count", type=int, default=5)
    parser.add_argument("--skip_final_evaluation", action="store_true")
    parser.add_argument("--seed", type=int, default=74)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if min(args.steps, args.layers, args.recurrences, args.log_every) < 1:
        raise ValueError("steps, layers, recurrences, and log_every must be positive")
    if args.probe_every < 0:
        raise ValueError("probe_every must be non-negative")
    if args.architecture == "full_token" and args.probe_every:
        args.probe_every = 0
    if any(horizon > args.recurrences for horizon in args.probe_horizons):
        raise ValueError("probe horizons cannot exceed training recurrences")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    data_root = ROOT / "controlled_experiments" / "data" / f"{args.task}_{args.preset}"
    datasets = {"train": MultiplicationDataset(data_root / "train.jsonl")}
    for split in ("test", "ood_long", "ood_one_long", "ood_both_long"):
        path = data_root / f"{split}.jsonl"
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
    fixed_probe_host_batch = collate(
        [datasets["train"][index] for index in range(min(args.batch_size, len(datasets["train"])))])
    fixed_probe_batch = {name: value.to(device) for name, value in fixed_probe_host_batch.items()}
    max_seq_len = max(
        len(record["input_ids"])
        for dataset in datasets.values()
        for record in dataset.records
    )
    rounds = args.recurrences if args.architecture == "recurrent" else 1
    model = FullTokenTransformer(
        ExperimentSource,
        ModelSpec(17, max_seq_len, 500_000_000),
        args.layers,
        rounds,
        args.round_embeddings,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1
    )
    named_parameters = list(model.named_parameters())
    activation_stats: dict[str, dict[str, float]] = {}
    activation_calls = [0 for _ in model.blocks]
    capture_activations = False

    def make_hook(index: int):
        def hook(_module, inputs, output):
            if not capture_activations:
                return
            activation_calls[index] += 1
            key = f"block_{index}_call_{activation_calls[index]}"
            incoming = inputs[0]
            activation_stats[key] = {
                "input_rms": rms(incoming),
                "output_rms": rms(output),
                "change_rms": rms(output - incoming),
            }

        return hook

    handles = [block.register_forward_hook(make_hook(index)) for index, block in enumerate(model.blocks)]
    autocast_context = (
        (lambda: torch.autocast("cuda", dtype=torch.bfloat16))
        if device.type == "cuda"
        else nullcontext
    )
    output = ROOT / "controlled_experiments" / "results" / (
        f"profile_{args.task}_{args.preset}_{args.architecture}_l{args.layers}"
        f"_r{rounds}_s{args.steps}_seed{args.seed}.jsonl"
    )
    started = time.monotonic()
    model.train()
    with output.open("w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "kind": "configuration",
                    "task": args.task,
                    "preset": args.preset,
                    "architecture": args.architecture,
                    "layers": args.layers,
                    "recurrences": rounds,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "steps": args.steps,
                    "parameter_count": sum(parameter.numel() for _, parameter in named_parameters),
                }
            )
            + "\n"
        )
        for step in range(1, args.steps + 1):
            try:
                host_batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                host_batch = next(iterator)
            batch = {name: value.to(device) for name, value in host_batch.items()}
            log_step = step == 1 or step % args.log_every == 0 or step == args.steps
            activation_stats.clear()
            activation_calls[:] = [0] * len(model.blocks)
            capture_activations = log_step
            optimizer.zero_grad(set_to_none=True)
            with autocast_context():
                loss, _, _ = loss_and_predictions(model, batch)
            loss.backward()
            capture_activations = False
            before_step = (
                {name: parameter.detach().float().clone() for name, parameter in named_parameters}
                if log_step
                else {}
            )
            raw_global_gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
            optimizer.step()
            if not log_step:
                continue
            record: dict[str, object] = {
                "kind": "step",
                "step": step,
                "loss": loss.item(),
                "seconds": time.monotonic() - started,
                "global_gradient_norm_before_clip": raw_global_gradient_norm,
                "activation": dict(activation_stats),
                "parameter_groups": aggregate_parameter_metrics(named_parameters, before_step),
                "rmsnorm_parameters": norm_parameter_metrics(named_parameters, before_step),
            }
            if args.probe_every and (step == 1 or step % args.probe_every == 0 or step == args.steps):
                record["gradient_alignment"] = gradient_alignment(
                    model, fixed_probe_batch, args.probe_horizons, autocast_context
                )
            line = json.dumps(record)
            handle.write(line + "\n")
            handle.flush()
            print(line)
    for handle in handles:
        handle.remove()

    summary = {"kind": "final_evaluation", "seconds": time.monotonic() - started}
    if not args.skip_final_evaluation:
        summary["splits"] = {
            split: evaluate(model, loader, device, args.example_count)
            for split, loader in loaders.items()
        }
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(summary) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
