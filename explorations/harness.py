"""Shared harness for One Layer Deeper exploration experiments.

Data format mirrors the competition exactly by importing the repo's own
generator/tokenizer (`data/squaring_mod.py`): decimal digit tokens (MSD first,
variable width), prompt = [N] d(N) [X] d(x) [T] d(T), separate tail-aligned
output positions, bidirectional attention with a padding mask, bf16 autocast.

Provides:
  - Regime definitions (fixed-N and sampled-N, mirroring public tiers, scaled)
  - build_data(): train pool + eval sets with hard disjointness asserts
  - EncoderModel: standard pre-LN bidirectional transformer
  - run_experiment(): trains under a steps- or seconds-budget, evaluates the
    full T ladder (ID-N fresh-x, OOD-N) plus the ID test split, and returns
    rows for results.csv with chance baselines attached
  - append_rows(): results.csv writer

Nothing in this module runs on import.
"""

from __future__ import annotations

import csv
import math
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.squaring_mod import (  # noqa: E402
    TOKEN_IDS,
    VOCAB_SIZE,
    _sample_rsa_factors,
    _sample_unit,
    tokenize_squaring_mod_with_result,
    trapdoor_squaring_mod,
)

# Denser than the competition's {1,2,4,8,16,32,64} so the break point and the
# small-N period (~<=8 for N=323) are both resolvable.
LADDER: tuple[int, ...] = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
CSV_PATH = RESULTS_DIR / "results.csv"
CSV_FIELDS = [
    "run_id", "exp", "regime", "seed", "device", "n_layers", "d_model", "n_heads",
    "params", "budget_mode", "budget", "steps_completed", "construct_seconds",
    "train_seconds", "lr", "batch_size", "first_train_loss", "final_train_loss",
    "diverged", "flat_from_start", "eval_set", "T", "ood_n", "ood_t", "n_eval",
    "digit_acc", "exact_acc", "chance_digit", "chance_exact",
]


# --------------------------------------------------------------------------
# Regimes
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Regime:
    """A scaled-down mirror of one competition dataset family."""

    name: str
    train_T: tuple[int, ...]
    fixed_pq: tuple[int, int] | None = None   # exact public modulus, or...
    train_bits: tuple[int, ...] | None = None  # ...sampled semiprime bit sizes
    ood_bits: tuple[int, ...] = ()             # OOD-N cohort bit sizes
    n_train: int = 20_000
    n_id_test: int = 1_000
    depth_cohort: int = 128     # matched (N, x) cohort size per ladder rung
    n_ood_moduli: int = 16


REGIMES: dict[str, Regime] = {
    # E1 mirror: tiny fixed modulus. Only 288 units exist, so pools are small
    # and the depth cohort is reserved before training prompts are drawn.
    "e1_fixed323": Regime(
        name="e1_fixed323", train_T=(1, 2, 3), fixed_pq=(17, 19),
        ood_bits=(10, 11), n_train=500, n_id_test=100, depth_cohort=64,
    ),
    # E3/E5 mirror: sampled 10-11 bit semiprimes.
    "easy_sampled_b1011": Regime(
        name="easy_sampled_b1011", train_T=(1, 2, 3), train_bits=(10, 11),
        ood_bits=(12, 13),
    ),
    # M1 mirror: fixed N=10403, geometric T.
    "m1_fixed10403": Regime(
        name="m1_fixed10403", train_T=(4, 8, 16), fixed_pq=(101, 103),
        ood_bits=(15, 16), n_train=10_000, n_id_test=500,
    ),
    # M5-ish mirror: sampled 12-16 bit semiprimes, joint N/T conditioning.
    "med_sampled_b121416": Regime(
        name="med_sampled_b121416", train_T=(2, 4, 8),
        train_bits=(12, 14, 16), ood_bits=(13, 15, 18), n_train=30_000,
    ),
}


# --------------------------------------------------------------------------
# Data generation (repo-faithful records -> padded tensors)
# --------------------------------------------------------------------------

RawRecord = tuple[int, int, int, int, int]  # p, q, x, T, result


def _make_record(p: int, q: int, x: int, t: int) -> RawRecord:
    return p, q, x, t, trapdoor_squaring_mod(x, t, p, q)


def _tokenize(records: list[RawRecord], max_seq_len: int) -> dict[str, torch.Tensor]:
    """Tokenize + pad exactly like `collate_squaring_mod` (separate output)."""
    rows = [
        tokenize_squaring_mod_with_result(p * q, x, t, result, separate_input_output=True)
        for p, q, x, t, result in records
    ]
    max_target_len = max(len(labels) for _, labels in rows)
    batch = len(rows)
    input_ids = torch.full((batch, max_seq_len), TOKEN_IDS["PAD"], dtype=torch.long)
    attention_mask = torch.zeros((batch, max_seq_len), dtype=torch.bool)
    labels = torch.full((batch, max_target_len), -100, dtype=torch.long)
    target_positions = torch.full((batch, max_target_len), -1, dtype=torch.long)
    for row, (prompt, target) in enumerate(rows):
        prompt_len, target_len = len(prompt), len(target)
        if prompt_len > max_seq_len:
            raise ValueError("max_seq_len too small for a generated prompt")
        input_ids[row, :prompt_len] = torch.tensor(prompt, dtype=torch.long)
        attention_mask[row, :prompt_len] = True
        labels[row, :target_len] = torch.tensor(target, dtype=torch.long)
        target_positions[row, :target_len] = torch.arange(
            prompt_len - target_len, prompt_len, dtype=torch.long
        )
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "target_positions": target_positions,
    }


def _chance_baselines(records: list[RawRecord]) -> tuple[float, float]:
    """Best constant predictor: per-digit-slot and exact-match rates."""
    slot_counts: list[Counter[str]] = []
    strings: Counter[str] = Counter()
    for _, _, _, _, result in records:
        text = str(result)
        strings[text] += 1
        for position, char in enumerate(reversed(text)):
            while len(slot_counts) <= position:
                slot_counts.append(Counter())
            slot_counts[position][char] += 1
    slot_hits = sum(counter.most_common(1)[0][1] for counter in slot_counts)
    slot_total = sum(sum(counter.values()) for counter in slot_counts)
    return slot_hits / slot_total, strings.most_common(1)[0][1] / len(records)


@dataclass
class EvalSet:
    name: str
    tensors: dict[str, torch.Tensor]
    T: int | None          # None for the mixed-T ID test split
    ood_n: bool
    ood_t: bool
    n: int
    chance_digit: float
    chance_exact: float


@dataclass
class DataBundle:
    regime: Regime
    seed: int
    max_seq_len: int
    train: dict[str, torch.Tensor]
    train_records: list[RawRecord]
    eval_sets: list[EvalSet]


def build_data(regime: Regime, seed: int) -> DataBundle:
    """Build train pool + eval sets with explicit disjointness guarantees.

    - train vs every eval set: disjoint (N, x, T) prompt triples (asserted)
    - depth cohorts: (N, x) pairs never used in training (asserted)
    - OOD-N cohorts: modulus identities never used in training (asserted)
    - ladder rungs with T not in train_T are flagged ood_t
    """
    rng = random.Random(seed)
    seen_prompts: set[tuple[int, int, int]] = set()
    used_pairs: set[tuple[int, int]] = set()

    # ---- moduli pools -----------------------------------------------------
    if regime.fixed_pq is not None:
        train_factor_pool = [regime.fixed_pq]
    else:
        train_factor_pool = []
        seen_moduli: set[int] = set()
        # A few dozen identities per bit size, like the sampled-N datasets.
        for bits in regime.train_bits:
            for _ in range(200):
                p, q = _sample_rsa_factors(modulus_bits=bits, rng=rng)
                if p * q not in seen_moduli:
                    seen_moduli.add(p * q)
                    train_factor_pool.append((p, q))
                if sum(1 for pp, qq in train_factor_pool
                       if (pp * qq).bit_length() == bits) >= 24:
                    break
    train_moduli = {p * q for p, q in train_factor_pool}

    ood_factor_pool: list[tuple[int, int]] = []
    for bits in regime.ood_bits:
        count = 0
        for _ in range(10_000):
            p, q = _sample_rsa_factors(modulus_bits=bits, rng=rng)
            if p * q not in train_moduli and all(p * q != pp * qq for pp, qq in ood_factor_pool):
                ood_factor_pool.append((p, q))
                count += 1
            if count >= max(1, regime.n_ood_moduli // len(regime.ood_bits)):
                break
    assert {p * q for p, q in ood_factor_pool}.isdisjoint(train_moduli), \
        "OOD-N moduli overlap training moduli"

    # ---- depth cohorts (reserved before training prompts) -----------------
    def sample_cohort(pool: list[tuple[int, int]], size: int) -> list[tuple[int, int, int]]:
        cohort: list[tuple[int, int, int]] = []
        for _ in range(100_000):
            if len(cohort) >= size:
                break
            p, q = rng.choice(pool)
            x = _sample_unit(modulus=p * q, rng=rng)
            if (p * q, x) not in used_pairs:
                used_pairs.add((p * q, x))
                cohort.append((p, q, x))
        if len(cohort) < size:
            raise ValueError("could not reserve a fresh depth cohort")
        return cohort

    seen_n_cohort = sample_cohort(train_factor_pool, regime.depth_cohort)
    ood_n_cohort = sample_cohort(ood_factor_pool, regime.depth_cohort)

    # ---- train + ID-test prompts ------------------------------------------
    def sample_prompt_records(count: int, exclude_pairs: bool) -> list[RawRecord]:
        records: list[RawRecord] = []
        for _ in range(count):
            for _ in range(100_000):
                p, q = rng.choice(train_factor_pool)
                x = _sample_unit(modulus=p * q, rng=rng)
                t = rng.choice(regime.train_T)
                if exclude_pairs and (p * q, x) in used_pairs:
                    continue
                if (p * q, x, t) not in seen_prompts:
                    seen_prompts.add((p * q, x, t))
                    records.append(_make_record(p, q, x, t))
                    break
            else:
                raise ValueError("prompt space exhausted; shrink n_train for this regime")
        return records

    # Depth-cohort (N, x) pairs are excluded from training entirely (repo
    # behavior for fixed-N depth profiles); ID-test shares moduli but not prompts.
    train_records = sample_prompt_records(regime.n_train, exclude_pairs=True)
    id_test_records = sample_prompt_records(regime.n_id_test, exclude_pairs=True)

    train_prompts = {(p * q, x, t) for p, q, x, t, _ in train_records}
    assert train_prompts.isdisjoint(
        {(p * q, x, t) for p, q, x, t, _ in id_test_records}
    ), "ID test prompts overlap training prompts"

    # ---- ladder eval sets --------------------------------------------------
    max_result_digits = 0
    all_records: list[tuple[str, list[RawRecord], int | None, bool]] = [
        ("id_test", id_test_records, None, False),
    ]
    for t in LADDER:
        seen_records = [_make_record(p, q, x, t) for p, q, x in seen_n_cohort]
        ood_records = [_make_record(p, q, x, t) for p, q, x in ood_n_cohort]
        all_records.append((f"depth_seen_n_t{t}", seen_records, t, False))
        all_records.append((f"depth_ood_n_t{t}", ood_records, t, True))

    # Global max_seq_len across everything, so all tensors share one width.
    def prompt_len(record: RawRecord) -> int:
        p, q, x, t, result = record
        prompt, _ = tokenize_squaring_mod_with_result(
            p * q, x, t, result, separate_input_output=True
        )
        return len(prompt)

    max_seq_len = max(
        max(prompt_len(record) for record in records)
        for _, records, _, _ in all_records
    )
    max_seq_len = max(max_seq_len, max(prompt_len(r) for r in train_records))

    eval_sets: list[EvalSet] = []
    for name, records, t, ood_n in all_records:
        chance_digit, chance_exact = _chance_baselines(records)
        eval_sets.append(EvalSet(
            name=name,
            tensors=_tokenize(records, max_seq_len),
            T=t,
            ood_n=ood_n,
            ood_t=(t is not None and t not in regime.train_T),
            n=len(records),
            chance_digit=chance_digit,
            chance_exact=chance_exact,
        ))

    # Depth cohorts must be fresh pairs relative to training.
    train_pairs = {(p * q, x) for p, q, x, _, _ in train_records}
    assert train_pairs.isdisjoint({(p * q, x) for p, q, x in seen_n_cohort}), \
        "seen-N depth cohort leaks training (N, x) pairs"
    assert train_pairs.isdisjoint({(p * q, x) for p, q, x in ood_n_cohort}), \
        "OOD-N depth cohort leaks training (N, x) pairs"

    return DataBundle(
        regime=regime,
        seed=seed,
        max_seq_len=max_seq_len,
        train=_tokenize(train_records, max_seq_len),
        train_records=train_records,
        eval_sets=eval_sets,
    )


# --------------------------------------------------------------------------
# Model: standard pre-LN bidirectional transformer encoder
# --------------------------------------------------------------------------

class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x: torch.Tensor, keep_mask: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)

        def heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(batch, length, self.n_heads, width // self.n_heads).transpose(1, 2)

        attended = F.scaled_dot_product_attention(
            heads(q), heads(k), heads(v), attn_mask=keep_mask
        )
        x = x + self.proj(attended.transpose(1, 2).reshape(batch, length, width))
        return x + self.mlp(self.ln2(x))


class EncoderModel(nn.Module):
    def __init__(self, vocab_size: int, max_seq_len: int, d_model: int,
                 n_layers: int, n_heads: int) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must divide evenly into heads")
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList(Block(d_model, n_heads) for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(input_ids.size(1), device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        # [B, 1, 1, L] boolean, True = may be attended to (padding mask only:
        # bidirectional, exactly like the competition evaluator).
        keep_mask = attention_mask[:, None, None, :]
        for block in self.blocks:
            x = block(x, keep_mask)
        return self.head(self.ln_f(x))


def gather_target_logits(logits: torch.Tensor, target_positions: torch.Tensor) -> torch.Tensor:
    index = target_positions.clamp_min(0).unsqueeze(-1).expand(-1, -1, logits.size(-1))
    return logits.gather(1, index)


def sequence_loss(logits: torch.Tensor, labels: torch.Tensor,
                  target_positions: torch.Tensor) -> torch.Tensor:
    target_logits = gather_target_logits(logits, target_positions)
    return F.cross_entropy(
        target_logits.transpose(1, 2), labels, ignore_index=-100
    )


# --------------------------------------------------------------------------
# Training / evaluation
# --------------------------------------------------------------------------

@dataclass
class RunConfig:
    exp: str
    regime: str
    n_layers: int
    d_model: int
    budget_mode: str            # "steps" | "seconds"
    budget: float               # step count or wall-clock seconds
    n_heads: int | None = None  # default: max(2, d_model // 64)
    batch_size: int = 256
    eval_batch_size: int = 1024
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 50
    grad_clip: float = 1.0
    seed: int = 0
    device: str | None = None   # default: freest CUDA device, else CPU
    use_compile: bool = False

    @property
    def resolved_heads(self) -> int:
        return self.n_heads or max(2, self.d_model // 64)

    @property
    def run_id(self) -> str:
        return (f"{self.exp}_{self.regime}_L{self.n_layers}_d{self.d_model}"
                f"_{self.budget_mode}{self.budget:g}_s{self.seed}")


def pick_device(explicit: str | None = None) -> torch.device:
    """Freest CUDA device, or an explicit one.

    Refuses to silently fall back to CPU: a CPU run is ~200x slower, which
    silently invalidates every wall-clock number and quietly starves every
    seconds-budget run. Pass device="cpu" to opt in deliberately.
    """
    if explicit:
        return torch.device(explicit)
    if not torch.cuda.is_available():
        built_for = torch.version.cuda
        raise RuntimeError(
            "CUDA is unavailable, refusing to fall back to CPU silently.\n"
            f"  torch {torch.__version__} built for CUDA {built_for}\n"
            "  If this is the repo's .venv (torch 2.13.0+cu130) the driver "
            "(12.8) is too old for it.\n"
            "  Run explorations with /usr/bin/python (torch 2.7.0+cu128), or "
            "pass device='cpu' to accept a CPU run."
        )
    free_bytes = []
    for index in range(torch.cuda.device_count()):
        free, _ = torch.cuda.mem_get_info(index)
        free_bytes.append((free, index))
    return torch.device(f"cuda:{max(free_bytes)[1]}")


@torch.no_grad()
def evaluate_set(model: nn.Module, tensors: dict[str, torch.Tensor],
                 device: torch.device, batch_size: int) -> tuple[float, float]:
    model.eval()
    digit_correct = digit_total = exact_correct = 0
    n = tensors["input_ids"].size(0)
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        batch = {key: value[start:stop].to(device) for key, value in tensors.items()}
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            logits = model(batch["input_ids"], batch["attention_mask"])
        predictions = gather_target_logits(
            logits.float(), batch["target_positions"]
        ).argmax(dim=-1)
        labels = batch["labels"]
        valid = labels != -100
        hits = (predictions == labels) & valid
        digit_correct += int(hits.sum())
        digit_total += int(valid.sum())
        exact_correct += int((hits | ~valid).all(dim=1).sum())
    model.train()
    return digit_correct / max(digit_total, 1), exact_correct / max(n, 1)


def run_experiment(config: RunConfig, data: DataBundle) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Train one model under the given budget and evaluate every eval set.

    Returns (csv_rows, summary). The wall-clock budget includes model
    construction (competition rule 11); evaluation time is excluded and
    reported separately, mirroring the separate evaluation budget.
    """
    device = pick_device(config.device)
    torch.manual_seed(config.seed)

    start = time.monotonic()
    model = EncoderModel(
        VOCAB_SIZE, data.max_seq_len, config.d_model,
        config.n_layers, config.resolved_heads,
    ).to(device)
    if config.use_compile:
        model = torch.compile(model)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / max(config.warmup_steps, 1))
    )
    construct_seconds = time.monotonic() - start

    train = {key: value.to(device) for key, value in data.train.items()}
    n_train = train["input_ids"].size(0)
    generator = torch.Generator(device="cpu").manual_seed(config.seed)

    losses: list[float] = []
    diverged = False
    deadline = start + config.budget if config.budget_mode == "seconds" else None
    max_steps = int(config.budget) if config.budget_mode == "steps" else 10**9

    step = 0
    while step < max_steps:
        if deadline is not None and time.monotonic() >= deadline:
            break
        index = torch.randint(0, n_train, (config.batch_size,), generator=generator).to(device)
        batch = {key: value[index] for key, value in train.items()}
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = sequence_loss(logits.float(), batch["labels"], batch["target_positions"])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        scheduler.step()
        loss_value = float(loss.detach())
        losses.append(loss_value)
        if math.isnan(loss_value) or math.isinf(loss_value):
            diverged = True
            break
        step += 1
    train_seconds = time.monotonic() - start

    first_loss = sum(losses[:10]) / max(len(losses[:10]), 1) if losses else float("nan")
    final_loss = sum(losses[-10:]) / max(len(losses[-10:]), 1) if losses else float("nan")
    # An LR that never moved the loss is an optimization failure, not an
    # architectural result; flag it rather than letting it masquerade as one.
    flat_from_start = bool(losses) and not diverged and final_loss > 0.95 * first_loss

    rows: list[dict[str, Any]] = []
    for eval_set in data.eval_sets:
        digit_acc, exact_acc = evaluate_set(
            model, eval_set.tensors, device, config.eval_batch_size
        )
        rows.append({
            "run_id": config.run_id, "exp": config.exp, "regime": config.regime,
            "seed": config.seed, "device": str(device), "n_layers": config.n_layers,
            "d_model": config.d_model, "n_heads": config.resolved_heads,
            "params": params, "budget_mode": config.budget_mode,
            "budget": config.budget, "steps_completed": step,
            "construct_seconds": round(construct_seconds, 3),
            "train_seconds": round(train_seconds, 3), "lr": config.lr,
            "batch_size": config.batch_size,
            "first_train_loss": round(first_loss, 5),
            "final_train_loss": round(final_loss, 5),
            "diverged": diverged, "flat_from_start": flat_from_start,
            "eval_set": eval_set.name, "T": eval_set.T,
            "ood_n": eval_set.ood_n, "ood_t": eval_set.ood_t,
            "n_eval": eval_set.n,
            "digit_acc": round(digit_acc, 5), "exact_acc": round(exact_acc, 5),
            "chance_digit": round(eval_set.chance_digit, 5),
            "chance_exact": round(eval_set.chance_exact, 6),
        })
    summary = {
        "run_id": config.run_id, "params": params, "steps": step,
        "train_seconds": train_seconds, "diverged": diverged,
        "flat_from_start": flat_from_start, "final_train_loss": final_loss,
        "device": str(device),
    }
    return rows, summary


def append_rows(rows: list[dict[str, Any]], csv_path: Path = CSV_PATH) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
