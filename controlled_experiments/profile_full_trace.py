"""Write an every-step, component-level optimization trace for controlled models.

This deliberately prioritizes observability over speed.  It records raw
gradients, Adam moments, parameter updates, and module activations for every
optimizer step.  Use it for diagnosis, not competition-time comparisons.
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
from controlled_experiments.profile_optimization import ExperimentSource, gradient_alignment, parse_positive_csv
from controlled_experiments.run_multiplication import (
    FullTokenBlock,
    FullTokenTransformer,
    LocalRMSNorm,
    MultiplicationDataset,
    collate,
    evaluate,
    loss_and_predictions,
)


def magnitude(tensor: Tensor) -> dict[str, float]:
    values = tensor.detach().float()
    return {
        "rms": values.square().mean().sqrt().item(),
        "norm": values.square().sum().sqrt().item(),
    }


def is_profiled_module(module: torch.nn.Module) -> bool:
    return isinstance(module, (torch.nn.Embedding, torch.nn.Linear, LocalRMSNorm, FullTokenBlock))


def recurrent_loss(
    model: FullTokenTransformer,
    batch: dict[str, Tensor],
    *,
    detach_after_first_repeat: bool,
) -> Tensor:
    """Final-answer loss, optionally blocking gradients through repeat one.

    With shared parameters and exactly two repeats, subtracting the resulting
    round-two-only gradient from the ordinary full gradient gives the gradient
    contribution from the first use of each parameter.
    """
    if model.rounds != 2:
        raise ValueError("per-recurrence attribution currently requires exactly two repeats")
    context = model.token_embedding(batch["input_ids"])
    rope = model.source.rope_tables(
        context.shape[1],
        model.source.D_MODEL // model.source.NUM_HEADS,
        context.device,
        context.dtype,
    )
    for repeat in range(2):
        if model.round_embeddings is not None:
            context = context + model.round_embeddings[repeat]
        for block in model.blocks:
            context = block(context, batch["mask"], rope)
        if repeat == 0 and detach_after_first_repeat:
            context = context.detach()
    logits = model.head(model.final_norm(context))
    row = torch.arange(logits.shape[0], device=logits.device)[:, None]
    selected = logits[row, batch["positions"].clamp_min(0)]
    valid = batch["labels"] != -100
    return torch.nn.functional.cross_entropy(selected[valid], batch["labels"][valid])


def per_recurrence_gradient_contributions(
    model: FullTokenTransformer,
    batch: dict[str, Tensor],
    autocast_context,
) -> dict[str, object]:
    """Attribute final-loss gradients to the first versus second repeat."""
    named_parameters = list(model.named_parameters())
    parameters = [parameter for _, parameter in named_parameters]
    with autocast_context():
        full_loss = recurrent_loss(model, batch, detach_after_first_repeat=False)
    full_gradients = torch.autograd.grad(full_loss, parameters)
    with autocast_context():
        second_loss = recurrent_loss(model, batch, detach_after_first_repeat=True)
    second_gradients = torch.autograd.grad(second_loss, parameters)
    parameter_metrics: dict[str, dict[str, float]] = {}
    for (name, _), total, second in zip(named_parameters, full_gradients, second_gradients, strict=True):
        first = total - second
        first_norm = first.float().square().sum().sqrt().item()
        second_norm = second.float().square().sum().sqrt().item()
        cosine = (first.float() * second.float()).sum().item() / (first_norm * second_norm + 1e-12)
        parameter_metrics[name] = {
            "repeat_1_gradient_norm": first_norm,
            "repeat_2_gradient_norm": second_norm,
            "repeat_gradient_cosine": cosine,
        }
    return {
        "full_loss": full_loss.item(),
        "detached_first_repeat_loss": second_loss.item(),
        "parameters": parameter_metrics,
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
    parser.add_argument("--round_embeddings", action="store_true")
    parser.add_argument("--fixed_probe_every", type=int, default=100)
    parser.add_argument("--probe_horizons", type=parse_positive_csv, default=parse_positive_csv("1,2"))
    parser.add_argument("--flush_every", type=int, default=25)
    parser.add_argument("--example_count", type=int, default=5)
    parser.add_argument("--skip_final_evaluation", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=74)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if min(args.steps, args.layers, args.recurrences, args.flush_every) < 1:
        raise ValueError("steps, layers, recurrences, and flush_every must be positive")
    if args.fixed_probe_every < 0:
        raise ValueError("fixed_probe_every must be non-negative")
    if args.architecture == "full_token":
        args.fixed_probe_every = 0
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
    fixed_probe_host_batch = collate(
        [datasets["train"][index] for index in range(min(args.batch_size, len(datasets["train"])))])
    fixed_probe_batch = {name: value.to(device) for name, value in fixed_probe_host_batch.items()}
    iterator = iter(train_loader)
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
    module_calls: dict[str, int] = {}
    activation: dict[str, dict[str, float | None]] = {}

    def make_hook(name: str):
        def hook(_module, inputs, output):
            if not isinstance(output, Tensor) or not inputs or not isinstance(inputs[0], Tensor):
                return
            call = module_calls.get(name, 0) + 1
            module_calls[name] = call
            incoming = inputs[0]
            outgoing = output
            statistics: dict[str, float | None] = {
                "input_rms": magnitude(incoming)["rms"],
                "output_rms": magnitude(outgoing)["rms"],
                "change_rms": magnitude(outgoing - incoming)["rms"]
                if incoming.shape == outgoing.shape
                else None,
            }
            if name.endswith(".qkv") and outgoing.shape[-1] % 3 == 0:
                query, key, value = outgoing.chunk(3, dim=-1)
                statistics.update(
                    {
                        "query_rms": magnitude(query)["rms"],
                        "key_rms": magnitude(key)["rms"],
                        "value_rms": magnitude(value)["rms"],
                    }
                )
            activation[f"{name}#{call}"] = statistics

        return hook

    handles = [
        module.register_forward_hook(make_hook(name))
        for name, module in model.named_modules()
        if name and is_profiled_module(module)
    ]
    autocast_context = (
        (lambda: torch.autocast("cuda", dtype=torch.bfloat16))
        if device.type == "cuda"
        else nullcontext
    )
    output = args.output or ROOT / "controlled_experiments" / "results" / (
        f"trace_{args.task}_{args.preset}_{args.architecture}_l{args.layers}"
        f"_r{rounds}_s{args.steps}_seed{args.seed}.jsonl"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
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
                    "fixed_probe_every": args.fixed_probe_every,
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
            module_calls.clear()
            activation.clear()
            optimizer.zero_grad(set_to_none=True)
            with autocast_context():
                loss, prediction, valid = loss_and_predictions(model, batch)
            loss.backward()
            raw_gradients = {
                name: magnitude(parameter.grad) if parameter.grad is not None else {"rms": 0.0, "norm": 0.0}
                for name, parameter in named_parameters
            }
            before_step = {name: parameter.detach().float().clone() for name, parameter in named_parameters}
            raw_global_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
            optimizer.step()
            parameter_trace: dict[str, dict[str, float]] = {}
            for name, parameter in named_parameters:
                parameter_values = parameter.detach().float()
                update = parameter_values - before_step[name]
                state = optimizer.state[parameter]
                parameter_trace[name] = {
                    "parameter_rms": magnitude(parameter_values)["rms"],
                    "parameter_norm": magnitude(parameter_values)["norm"],
                    "gradient_rms_raw": raw_gradients[name]["rms"],
                    "gradient_norm_raw": raw_gradients[name]["norm"],
                    "update_rms": magnitude(update)["rms"],
                    "update_norm": magnitude(update)["norm"],
                    "update_to_parameter": magnitude(update)["norm"] / (magnitude(parameter_values)["norm"] + 1e-12),
                    "adam_exp_avg_rms": magnitude(state["exp_avg"])["rms"],
                    "adam_exp_avg_sq_rms": magnitude(state["exp_avg_sq"])["rms"],
                }
            matches = (prediction == batch["labels"]) | ~valid
            record: dict[str, object] = {
                "kind": "step",
                "step": step,
                "seconds": time.monotonic() - started,
                "loss": loss.item(),
                "batch_digit_accuracy": ((prediction == batch["labels"]) & valid).sum().item()
                / valid.sum().item(),
                "batch_exact_accuracy": matches.all(dim=1).float().mean().item(),
                "global_gradient_norm_raw": raw_global_norm,
                "clipped": raw_global_norm > 1.0,
                "activation": dict(activation),
                "parameters": parameter_trace,
            }
            if args.fixed_probe_every and (
                step == 1 or step % args.fixed_probe_every == 0 or step == args.steps
            ):
                record["fixed_probe"] = gradient_alignment(
                    model, fixed_probe_batch, args.probe_horizons, autocast_context
                )
                if rounds == 2:
                    record["per_recurrence_gradient_contributions"] = (
                        per_recurrence_gradient_contributions(model, fixed_probe_batch, autocast_context)
                    )
            handle.write(json.dumps(record) + "\n")
            if step % args.flush_every == 0 or step == args.steps:
                handle.flush()
            if step == 1 or step % 100 == 0 or step == args.steps:
                print(
                    json.dumps(
                        {
                            "step": step,
                            "loss": loss.item(),
                            "global_gradient_norm_raw": raw_global_norm,
                            "seconds": time.monotonic() - started,
                        }
                    )
                )
    for handle in handles:
        handle.remove()
    summary: dict[str, object] = {"kind": "final_evaluation", "seconds": time.monotonic() - started}
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
