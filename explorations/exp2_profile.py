"""Experiment 2: compute-allocation profile.

Where does wall-clock go for a representative config? Measures, with
torch.cuda.synchronize() bracketing every phase:

  - torch.compile time (construction + compile + first step), timed separately
    because competition rule 11 charges it against the training budget
  - per-step data assembly + H2D copy, forward, backward, optimizer step
  - backward/forward ratio (theory ~2.0 when compute-bound)
  - achieved TFLOP/s vs an analytic matmul-FLOP estimate, and where the
    projection arithmetic intensity (~d_model/2) sits against the GPU's ridge
    point (peak FLOPs / memory bandwidth — both CLI args, defaults are L40
    numbers because that is the local GPU; pass H100 numbers on H100)
  - a torch.profiler top-kernel table sorted by CUDA time

Profiles two widths, d_model 128 and 640, to straddle the ridge-point
prediction: on H100 (ridge ~295 FLOP/byte) both should be bandwidth-bound at
128 and near-ridge at 640; on the local L40 (ridge ~105) d640 should be
comfortably compute-bound. If measured utilization does not track that
prediction, flag it loudly.

Writes results/exp2_profile.json and prints everything.

Usage:
  python explorations/exp2_profile.py [--compile] [--peak-tflops 90.5]
      [--mem-bw-gbs 864] [--widths 128 640] [--batch-size 512]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from harness import (
    REGIMES,
    RESULTS_DIR,
    EncoderModel,
    RunConfig,
    build_data,
    pick_device,
    sequence_loss,
)
from data.squaring_mod import VOCAB_SIZE

N_LAYERS = 4
WARMUP_STEPS = 5
TIMED_STEPS = 30
PROFILER_STEPS = 5


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def estimate_forward_flops(n_layers: int, d_model: int, seq_len: int,
                           batch: int, vocab: int) -> float:
    """Analytic matmul FLOPs for one forward pass (2*M*K*N per matmul)."""
    tokens = batch * seq_len
    per_token_layer = 2 * (3 * d_model * d_model    # qkv
                           + d_model * d_model      # attn out proj
                           + 8 * d_model * d_model)  # mlp (4x up + 4x down)
    attention = 2 * 2 * batch * seq_len * seq_len * d_model  # qk^T and att@v
    head = 2 * tokens * d_model * vocab
    embed = 0  # lookups, not matmuls
    return n_layers * (per_token_layer * tokens + attention) + head + embed


def profile_width(d_model: int, args: argparse.Namespace, device: torch.device,
                  data) -> dict:
    config = RunConfig(
        exp="exp2", regime=args.regime, n_layers=N_LAYERS, d_model=d_model,
        budget_mode="steps", budget=TIMED_STEPS, batch_size=args.batch_size,
        seed=0,
    )
    torch.manual_seed(0)

    # --- construction (+ optional compile), timed the way rule 11 charges it
    start = time.monotonic()
    model = EncoderModel(VOCAB_SIZE, data.max_seq_len, d_model, N_LAYERS,
                         config.resolved_heads).to(device)
    construct_seconds = time.monotonic() - start

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    train_cpu = data.train  # kept on CPU deliberately: H2D is a measured phase
    n_train = train_cpu["input_ids"].size(0)
    generator = torch.Generator().manual_seed(0)

    def make_batch():
        index = torch.randint(0, n_train, (config.batch_size,), generator=generator)
        return {key: value[index] for key, value in train_cpu.items()}

    def to_device(batch):
        return {key: value.to(device, non_blocking=False) for key, value in batch.items()}

    def forward(batch):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            logits = model(batch["input_ids"], batch["attention_mask"])
            return sequence_loss(logits.float(), batch["labels"],
                                 batch["target_positions"])

    compile_seconds = 0.0
    if args.compile:
        start = time.monotonic()
        model = torch.compile(model)
        loss = forward(to_device(make_batch()))   # first call triggers compile
        loss.backward()
        optimizer.zero_grad(set_to_none=True)
        synchronize(device)
        compile_seconds = time.monotonic() - start

    # --- warmup
    for _ in range(WARMUP_STEPS):
        loss = forward(to_device(make_batch()))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    synchronize(device)

    # --- timed phases
    phase_seconds = {"data_h2d": 0.0, "forward": 0.0, "backward": 0.0, "optimizer": 0.0}
    for _ in range(TIMED_STEPS):
        synchronize(device)
        t0 = time.monotonic()
        batch = to_device(make_batch())
        synchronize(device)
        t1 = time.monotonic()
        loss = forward(batch)
        synchronize(device)
        t2 = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        synchronize(device)
        t3 = time.monotonic()
        optimizer.step()
        synchronize(device)
        t4 = time.monotonic()
        phase_seconds["data_h2d"] += t1 - t0
        phase_seconds["forward"] += t2 - t1
        phase_seconds["backward"] += t3 - t2
        phase_seconds["optimizer"] += t4 - t3

    step_ms = {name: seconds / TIMED_STEPS * 1000 for name, seconds in phase_seconds.items()}
    total_ms = sum(step_ms.values())
    bwd_fwd_ratio = step_ms["backward"] / step_ms["forward"] if step_ms["forward"] else float("nan")

    forward_flops = estimate_forward_flops(
        N_LAYERS, d_model, data.max_seq_len, config.batch_size, VOCAB_SIZE
    )
    achieved_fwd_tflops = forward_flops / (step_ms["forward"] / 1000) / 1e12
    ridge_flop_per_byte = args.peak_tflops * 1e12 / (args.mem_bw_gbs * 1e9)
    projection_intensity = d_model / 2  # bf16 [M,d]@[d,d] cap as M -> inf

    # --- torch.profiler top kernels
    from torch.profiler import ProfilerActivity, profile
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    with profile(activities=activities) as prof:
        for _ in range(PROFILER_STEPS):
            loss = forward(to_device(make_batch()))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        synchronize(device)
    sort_key = "cuda_time_total" if device.type == "cuda" else "cpu_time_total"
    kernel_table = prof.key_averages().table(sort_by=sort_key, row_limit=15)

    result = {
        "d_model": d_model,
        "n_layers": N_LAYERS,
        "batch_size": config.batch_size,
        "seq_len": data.max_seq_len,
        "params": sum(p.numel() for p in model.parameters()),
        "construct_seconds": round(construct_seconds, 3),
        "compile_seconds": round(compile_seconds, 3),
        "compiled": bool(args.compile),
        "step_ms": {name: round(value, 3) for name, value in step_ms.items()},
        "step_ms_total": round(total_ms, 3),
        "step_pct": {name: round(100 * value / total_ms, 1) for name, value in step_ms.items()},
        "bwd_fwd_ratio": round(bwd_fwd_ratio, 3),
        "est_forward_tflop": round(forward_flops / 1e12, 4),
        "achieved_forward_tflops": round(achieved_fwd_tflops, 2),
        "peak_tflops_assumed": args.peak_tflops,
        "forward_utilization_pct": round(100 * achieved_fwd_tflops / args.peak_tflops, 1),
        "ridge_flop_per_byte": round(ridge_flop_per_byte, 1),
        "projection_intensity_flop_per_byte": projection_intensity,
        "predicted_regime": (
            "compute-bound" if projection_intensity > ridge_flop_per_byte
            else "bandwidth-bound"
        ),
        "kernel_table": kernel_table,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--widths", type=int, nargs="+", default=[128, 640])
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--regime", default="easy_sampled_b1011", choices=sorted(REGIMES))
    parser.add_argument("--compile", action="store_true",
                        help="also measure torch.compile time (charged to budget)")
    parser.add_argument("--peak-tflops", type=float, default=90.5,
                        help="dense bf16 peak (L40 90.5; H100 SXM ~989)")
    parser.add_argument("--mem-bw-gbs", type=float, default=864.0,
                        help="memory bandwidth GB/s (L40 864; H100 SXM ~3350)")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = pick_device(args.device)
    print(f"profiling on {device} "
          f"(assumed peak {args.peak_tflops} TFLOP/s, {args.mem_bw_gbs} GB/s)")
    print(f"building data for {args.regime} ...", flush=True)
    data = build_data(REGIMES[args.regime], seed=45)

    results = []
    for width in args.widths:
        print(f"\n===== d_model={width} =====", flush=True)
        result = profile_width(width, args, device, data)
        results.append(result)
        print(f"construct={result['construct_seconds']}s "
              f"compile={result['compile_seconds']}s")
        print(f"per-step ms: {result['step_ms']}  (total {result['step_ms_total']} ms)")
        print(f"per-step % : {result['step_pct']}")
        print(f"bwd/fwd ratio: {result['bwd_fwd_ratio']}  "
              f"(~2.0 expected when compute-bound)")
        print(f"forward: est {result['est_forward_tflop']} TFLOP -> "
              f"{result['achieved_forward_tflops']} TFLOP/s achieved "
              f"({result['forward_utilization_pct']}% of assumed peak)")
        print(f"projection intensity d/2 = {result['projection_intensity_flop_per_byte']} "
              f"FLOP/B vs ridge {result['ridge_flop_per_byte']} FLOP/B "
              f"-> predicted {result['predicted_regime']}")
        mismatch = (
            (result["predicted_regime"] == "compute-bound"
             and result["forward_utilization_pct"] < 30)
            or (result["predicted_regime"] == "bandwidth-bound"
                and result["forward_utilization_pct"] > 60)
        )
        if mismatch:
            print("  *** UTILIZATION DOES NOT TRACK THE RIDGE PREDICTION — "
                  "flagging as requested; see kernel table. ***")
        print("\ntop kernels:")
        print(result["kernel_table"])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "exp2_profile.json"
    output_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {output_path}")


if __name__ == "__main__":
    main()
