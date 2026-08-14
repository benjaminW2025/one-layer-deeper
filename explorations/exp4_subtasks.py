"""Experiment 4: decompose x^2 mod N into its two primitive operations.

T=1 at fixed N is already a composition of two operations:

    A  square   x -> x*x          x in [0, N)      -> answer in [0, N^2)
    B  reduce   y -> y mod N      y in [0, N^2)    -> answer in [0, N)
    C  squaremod  x -> x*x mod N  x in [0, N)      -> answer in [0, N)   (= A then B)

B's input range is exactly A's output range, so C is their exact composition.
If A and B are learnable but C is not, the wall is composition. If A or B
fails alone, the wall is that primitive and nothing about depth matters yet.

Everything else is held identical to the competition: same 17-token vocab,
same digit-wise MSD-first numbers, same `[N] d(N) [X] d(x) [T] d(T)` prompt,
same right-aligned answer readout, same bidirectional padding mask. Only the
target function changes. (For A the modulus is an irrelevant field the model
must learn to ignore; pass --drop-irrelevant to strip it instead.)

Every task reports THREE cohorts so memorization is measured, not inferred:
  train_seen  - x values drawn from the training pool
  test_fresh  - x values never trained on
  (gap between them is the memorization signal)

x-space coverage is printed and recorded per config: with a fixed N the input
space is finite, so coverage says whether a lookup table is even available.

Usage:
  python explorations/exp4_subtasks.py                       # default sweep
  python explorations/exp4_subtasks.py --tasks square --bits 14
  python explorations/exp4_subtasks.py --widths 128 256 --depths 2 4
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import (  # noqa: E402
    RESULTS_DIR,
    DataBundle,
    Regime,
    EvalSet,
    RunConfig,
    _chance_baselines,
    _tokenize,
    run_experiment,
)
from data.squaring_mod import (  # noqa: E402
    _sample_rsa_factors,
    tokenize_squaring_mod_with_result,
)

CSV_PATH = RESULTS_DIR / "results_exp4.csv"
CSV_FIELDS = [
    "run_id", "task", "bits", "modulus", "cohort", "n_layers", "d_model",
    "params", "device", "budget_mode", "budget", "steps_completed",
    "train_seconds", "lr", "n_train", "x_space", "x_coverage",
    "first_train_loss", "final_train_loss", "diverged", "flat_from_start",
    "n_eval", "digit_acc", "exact_acc", "chance_digit", "chance_exact",
]

TaskFn = Callable[[int, int], int]
TASKS: dict[str, TaskFn] = {
    "square": lambda x, N: x * x,
    "reduce": lambda y, N: y % N,
    "squaremod": lambda x, N: (x * x) % N,
}
# Which range the varying input is drawn from, per task.
INPUT_RANGE = {
    "square": "units",     # x in [0, N), coprime to N (matches competition)
    "reduce": "n_squared",  # y in [0, N^2), i.e. exactly what squaring produces
    "squaremod": "units",
}


def build_subtask_data(
    *,
    task: str,
    bits: int,
    n_train: int,
    n_eval: int,
    seed: int,
    drop_irrelevant: bool,
) -> tuple[DataBundle, dict[str, Any]]:
    """Fixed modulus, varying input, disjoint train/fresh x pools."""
    rng = random.Random(seed)
    p, q = _sample_rsa_factors(modulus_bits=bits, rng=rng)
    modulus = p * q
    fn = TASKS[task]

    if INPUT_RANGE[task] == "units":
        x_space = (p - 1) * (q - 1)

        def draw() -> int:
            while True:
                value = rng.randrange(1, modulus)
                if math.gcd(value, modulus) == 1:
                    return value
    else:
        x_space = modulus * modulus

        def draw() -> int:
            return rng.randrange(0, modulus * modulus)

    if n_train + n_eval > x_space:
        raise ValueError(
            f"task={task} bits={bits}: input space is only {x_space}; "
            "reduce n_train/n_eval or raise --bits"
        )

    seen: set[int] = set()
    train_x: list[int] = []
    while len(train_x) < n_train:
        value = draw()
        if value not in seen:
            seen.add(value)
            train_x.append(value)
    fresh_x: list[int] = []
    while len(fresh_x) < n_eval:
        value = draw()
        if value not in seen:
            seen.add(value)
            fresh_x.append(value)
    # Held-in cohort: sampled from the training pool, so a memorizing model
    # scores ~1.0 here and ~0 on fresh. Makes the gap measured, not inferred.
    seen_x = rng.sample(train_x, min(n_eval, len(train_x)))

    def records(values: list[int]) -> list[tuple[int, int, int, int, int]]:
        return [(p, q, value, 1, fn(value, modulus)) for value in values]

    train_records = records(train_x)
    cohorts = {"train_seen": records(seen_x), "test_fresh": records(fresh_x)}

    def prompt_len(record: tuple[int, int, int, int, int]) -> int:
        pp, qq, value, t, result = record
        ids, _ = tokenize_squaring_mod_with_result(
            pp * qq, value, t, result, separate_input_output=True
        )
        return len(ids)

    all_records = train_records + [r for rs in cohorts.values() for r in rs]
    max_seq_len = max(prompt_len(r) for r in all_records)
    longest_target = max(len(str(r[4])) for r in all_records)
    if longest_target > max_seq_len:
        raise ValueError("answer longer than prompt; format cannot express it")

    eval_sets = []
    for name, recs in cohorts.items():
        chance_digit, chance_exact = _chance_baselines(recs)
        eval_sets.append(EvalSet(
            name=name, tensors=_tokenize(recs, max_seq_len), T=1,
            ood_n=False, ood_t=False, n=len(recs),
            chance_digit=chance_digit, chance_exact=chance_exact,
        ))

    bundle = DataBundle(
        regime=Regime(name=f"{task}_b{bits}", train_T=(1,), fixed_pq=(p, q)),
        seed=seed,
        max_seq_len=max_seq_len,
        train=_tokenize(train_records, max_seq_len),
        train_records=train_records,
        eval_sets=eval_sets,
    )
    meta = {
        "task": task, "bits": bits, "modulus": modulus, "x_space": x_space,
        "n_train": n_train, "x_coverage": n_train / x_space,
        "max_seq_len": max_seq_len,
    }
    return bundle, meta


def append_rows(rows: list[dict[str, Any]]) -> None:
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not CSV_PATH.exists()
    with CSV_PATH.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    parser.add_argument("--bits", type=int, nargs="+", default=[14, 18],
                        help="modulus bit size; larger = bigger x-space = less memorizable")
    parser.add_argument("--depths", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--widths", type=int, nargs="+", default=[256])
    parser.add_argument("--n-train", type=int, default=50_000)
    parser.add_argument("--n-eval", type=int, default=2_000)
    parser.add_argument("--budget-mode", default="steps", choices=["steps", "seconds"])
    parser.add_argument("--budget", type=float, default=6000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-seed", type=int, default=45)
    parser.add_argument("--drop-irrelevant", action="store_true",
                        help="strip fields the task does not need (changes format)")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if args.drop_irrelevant:
        raise NotImplementedError(
            "--drop-irrelevant changes the prompt format; not implemented yet. "
            "Default keeps the competition format so results transfer."
        )

    cells = [(t, b, d, w) for t in args.tasks for b in args.bits
             for d in args.depths for w in args.widths]
    print(f"{len(cells)} cells: tasks={args.tasks} bits={args.bits} "
          f"depths={args.depths} widths={args.widths}", flush=True)

    bundles: dict[tuple[str, int], tuple[DataBundle, dict]] = {}
    for task in args.tasks:
        for bits in args.bits:
            key = (task, bits)
            bundles[key] = build_subtask_data(
                task=task, bits=bits, n_train=args.n_train,
                n_eval=args.n_eval, seed=args.data_seed,
                drop_irrelevant=args.drop_irrelevant,
            )
            meta = bundles[key][1]
            print(f"  {task:10s} bits={bits:3d} N={meta['modulus']:<12d} "
                  f"x-space={meta['x_space']:<14d} "
                  f"coverage={meta['x_coverage']:.3%}  seq_len={meta['max_seq_len']}",
                  flush=True)

    started = time.monotonic()
    for index, (task, bits, depth, width) in enumerate(cells, start=1):
        bundle, meta = bundles[(task, bits)]
        config = RunConfig(
            exp="exp4", regime=f"{task}_b{bits}", n_layers=depth, d_model=width,
            budget_mode=args.budget_mode, budget=args.budget,
            batch_size=args.batch_size, lr=args.lr, seed=args.seed,
            device=args.device,
        )
        rows, summary = run_experiment(config, bundle)
        out = []
        for row in rows:
            out.append({
                "run_id": row["run_id"], "task": task, "bits": bits,
                "modulus": meta["modulus"], "cohort": row["eval_set"],
                "n_layers": depth, "d_model": width, "params": row["params"],
                "device": row["device"], "budget_mode": args.budget_mode,
                "budget": args.budget, "steps_completed": row["steps_completed"],
                "train_seconds": row["train_seconds"], "lr": args.lr,
                "n_train": args.n_train, "x_space": meta["x_space"],
                "x_coverage": round(meta["x_coverage"], 6),
                "first_train_loss": row["first_train_loss"],
                "final_train_loss": row["final_train_loss"],
                "diverged": row["diverged"], "flat_from_start": row["flat_from_start"],
                "n_eval": row["n_eval"], "digit_acc": row["digit_acc"],
                "exact_acc": row["exact_acc"],
                "chance_digit": row["chance_digit"],
                "chance_exact": row["chance_exact"],
            })
        append_rows(out)

        by_cohort = {r["cohort"]: r for r in out}
        seen_acc = by_cohort["train_seen"]["exact_acc"]
        fresh_acc = by_cohort["test_fresh"]["exact_acc"]
        chance = by_cohort["test_fresh"]["chance_exact"]
        elapsed = (time.monotonic() - started) / 60
        flags = " DIVERGED" if summary["diverged"] else ""
        flags += " FLAT(LR?)" if summary["flat_from_start"] else ""
        print(f"[{index}/{len(cells)} {elapsed:.0f}m] {task:10s} b{bits} "
              f"L{depth} d{width}: loss={summary['final_train_loss']:.4f} "
              f"seen={seen_acc:.3f} fresh={fresh_acc:.3f} "
              f"(chance {chance:.3f}) gap={seen_acc - fresh_acc:+.3f}{flags}",
              flush=True)

    print(f"\ndone in {(time.monotonic() - started) / 60:.1f} min -> {CSV_PATH}")
    print("read: fresh >> chance means real structure; "
          "seen >> fresh means memorization; both low means it learned nothing")


if __name__ == "__main__":
    main()
