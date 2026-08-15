"""Look INSIDE trained `rope` baselines instead of only reading off predictions.

Every positional-encoding experiment so far assumed the OOD failure is about
*where* attention points once digit length leaves the trained range. A
leave-one-length-out check (train on {1,2,4,5}, eval zero-shot on 3, which is
pure interpolation -- not even extrapolation) showed the model gets this
almost entirely wrong (exact=0.0095), which means the "one shared algorithm,
bad addressing" story is probably incomplete: it looks like the model learns
close to a separate program per trained digit length rather than one
length-general procedure.

This script compares internals across checkpoints and cohorts to find where
that shows up mechanically, per LAYER and per HEAD (not just aggregated):

  1. attention entropy at read-out (target) positions -- diffuse (high
     entropy, low top-prob) means "gives up and spreads out"; confident-but-
     wrong would look different (low entropy, high top-prob, wrong offset).
  2. the relative offset (key position - query position) of each read-out
     query's TOP-attended key, bucketed against whether that cohort's
     underlying field-offset (digit_length + 1, since X and Y are the same
     length in x*x) was ever seen exactly during training, seen only via
     interpolation (bracketed by two trained offsets, never itself), or is
     pure extrapolation (beyond every trained offset). If confidence/entropy
     tracks "was this offset trained" cleanly regardless of interpolation vs
     extrapolation, that's evidence of a memorized-per-offset mechanism
     rather than a smooth function of offset.
  3. hidden-state RMS norm per layer -- catches length-driven activation
     drift independent of attention entirely.
  4. per-head breakdown of all of the above, since aggregating over heads
     can hide a head that's doing something specific.

Usage:
  python experiments/diagnose_internals.py
  python experiments/diagnose_internals.py --checkpoints square_rope_L4_d256_s0.pt --plot
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.model import apply_rope, rope_tables  # noqa: E402
from common.tasks import build_square  # noqa: E402
from common.tokenizer import SQUARE  # noqa: E402
from common.train import load_checkpoint, pick_device, tensors_from_records  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
CKPT_DIR = RESULTS / "ckpt"

DEFAULT_CHECKPOINTS = [
    "square_rope_L4_d256_s0.pt",       # baseline WITH carries (real x*x)
    "carryless_square_rope_L4_d256_s0.pt",  # leave-3-out carryless model
]


@torch.no_grad()
def forward_with_diagnostics(model, input_ids: torch.Tensor, attention_mask: torch.Tensor):
    """Mirrors Encoder.forward + Block.forward for positional="rope" (rel
    mode, no abacus/learned/sinusoidal terms, no sink, no loop) -- both
    checkpoints this script targets -- but also returns, per layer:
    attention weights (batch, heads, len, len) and post-block hidden RMS
    norm (batch, len). Not a general reimplementation of every positional
    mode; asserts the model matches the one case it handles."""
    assert model.positional == "rope" and model.rope_mode == "rel" and not model.blocks[0].use_sink, \
        "this diagnostic only mirrors plain positional='rope', no sink"

    length = input_ids.size(1)
    x = model.token_embedding(input_ids)
    cos, sin = rope_tables(length, model.head_dim, input_ids.device)
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    keep_mask = attention_mask[:, None, None, :]

    layers = []
    for block in model.blocks:
        q, k, v = block.qkv(block.ln1(x)).chunk(3, dim=-1)
        batch, seq_len, width = x.shape

        def heads(t):
            return t.view(batch, seq_len, block.n_heads, width // block.n_heads).transpose(1, 2)

        q, k, v = heads(q), heads(k), heads(v)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        scale = 1.0 / math.sqrt(q.size(-1))
        scores = (q @ k.transpose(-2, -1)) * scale
        scores = scores.masked_fill(~keep_mask, float("-inf"))
        weights = scores.softmax(dim=-1)
        attended = weights @ v
        x = x + block.proj(attended.transpose(1, 2).reshape(batch, seq_len, width))
        x = x + block.mlp(block.ln2(x))
        layers.append({"weights": weights, "hidden_rms": x.pow(2).mean(dim=-1).sqrt()})

    logits = model.head(model.ln_f(x))
    return logits, layers


def entropy_bits(weights: torch.Tensor) -> torch.Tensor:
    """weights: (..., keys). Shannon entropy in bits, safe for zero-mass keys."""
    safe = weights.clamp_min(1e-12)
    return -(weights * safe.log2()).sum(dim=-1)


def analyze_cohort(model, tensors: dict, device: torch.device) -> dict:
    batch = {k: v.to(device) for k, v in tensors.items()}
    logits, layers = forward_with_diagnostics(model, batch["input_ids"], batch["attention_mask"])

    target_positions = batch["target_positions"]  # (n, answer_len)
    valid = batch["labels"] != -100
    n, answer_len = target_positions.shape
    query_pos = target_positions

    out = {"n_examples": n, "per_layer": []}
    for li, layer in enumerate(layers):
        w = layer["weights"]  # (n, heads, seq, seq)
        heads = w.size(1)
        gather_idx = query_pos[:, None, :, None].expand(-1, heads, -1, w.size(-1))
        target_weights = w.gather(2, gather_idx)  # (n, heads, answer_len, seq)

        ent = entropy_bits(target_weights)  # (n, heads, answer_len)
        top_key = target_weights.argmax(dim=-1)  # (n, heads, answer_len)
        offset = (top_key - query_pos[:, None, :].expand(-1, heads, -1)).float()
        top_prob = target_weights.max(dim=-1).values

        valid_all = valid[:, None, :].expand(-1, heads, -1)
        # per-head means: mask each head's own valid slice, keep a python list
        entropy_per_head, top_prob_per_head, offset_mean_per_head = [], [], []
        for h in range(heads):
            m = valid[:, :]  # (n, answer_len), same mask for every head
            entropy_per_head.append(ent[:, h, :][m].mean().item())
            top_prob_per_head.append(top_prob[:, h, :][m].mean().item())
            offset_mean_per_head.append(offset[:, h, :][m].mean().item())

        hidden_rms = layer["hidden_rms"]  # (n, seq)
        target_rms = hidden_rms.gather(1, query_pos)

        out["per_layer"].append({
            "layer": li,
            "mean_entropy_bits": ent[valid_all].mean().item(),
            "mean_top_attn_prob": top_prob[valid_all].mean().item(),
            "mean_hidden_rms": target_rms[valid].mean().item(),
            "offset_mean": offset[valid_all].mean().item(),
            "offset_std": offset[valid_all].std().item(),
            "offsets": offset[valid_all].cpu(),
            "entropy_per_head": entropy_per_head,
            "top_prob_per_head": top_prob_per_head,
            "offset_mean_per_head": offset_mean_per_head,
        })
    return out


@torch.no_grad()
def mean_attention_maps(model, tensors: dict, device: torch.device) -> dict:
    """Full (query x key) attention matrix per layer/head, averaged over the
    batch and cropped to the cohort's real (unpadded) length -- the actual
    attention heads, not just summary statistics of them."""
    batch = {k: v.to(device) for k, v in tensors.items()}
    _, layers = forward_with_diagnostics(model, batch["input_ids"], batch["attention_mask"])
    valid_len = int(batch["attention_mask"][0].sum().item())
    target_start = int(batch["target_positions"].amin().item())
    field_split = None
    # Y marker sits right after X's digit run; find it as the second non-digit
    # token id shared across the batch (marker ids are < SQUARE.digit_range[0]).
    ids0 = batch["input_ids"][0, :valid_len].tolist()
    markers = [i for i, t in enumerate(ids0) if t < SQUARE.digit_range[0]]
    if len(markers) >= 2:
        field_split = markers[1]
    maps = []
    for layer in layers:
        w = layer["weights"][:, :, :valid_len, :valid_len].mean(dim=0)  # (heads, L, L)
        maps.append(w.cpu())
    return {"maps": maps, "valid_len": valid_len, "target_start": target_start,
            "field_split": field_split}


def offset_category(digit_len: int, train_digits: list[int]) -> str:
    """x*x has X and Y the same length, so the field offset between a digit
    and its same-significance counterpart is a fixed function of digit
    length (digit_len + 1, from the marker + digit run in between)."""
    train_offsets = sorted(d + 1 for d in train_digits)
    offset = digit_len + 1
    if offset in train_offsets:
        return "trained (unseen x)"
    if train_offsets[0] < offset < train_offsets[-1]:
        return "interpolated"
    return "extrapolated"


def x_digit_length(record: dict, digit_lo: int = SQUARE.digit_range[0]) -> int:
    """Length of the X operand (field between the two markers) for a square
    task record -- used to pull out literally-trained-on examples of a
    specific length from `data.train`, which mixes every trained length."""
    marker_positions = [i for i, t in enumerate(record["input_ids"]) if t < digit_lo]
    return marker_positions[1] - marker_positions[0] - 1


def analyze_checkpoint(ckpt_name: str, args, device: torch.device) -> None:
    ckpt_path = CKPT_DIR / ckpt_name
    model = load_checkpoint(ckpt_path, SQUARE).to(device)
    model.eval()
    meta = torch.load(ckpt_path, map_location="cpu", weights_only=False)["meta"]
    train_digits = meta["train_digits"]
    carryless = ckpt_name.startswith("carryless_")
    ood_digits = sorted(set(range(1, 8)) - set(train_digits))
    id_digit = max(train_digits)
    id_cohort = f"id_{id_digit}d"
    train_cohort = f"train_{id_digit}d"

    print(f"\n{'=' * 90}\n{ckpt_name}\n"
          f"  train_digits={train_digits}  carryless={carryless}  "
          f"held_out/ood_digits={ood_digits}", flush=True)

    # n_train/n_eval must match exp2_square.py's defaults (used, unoverridden,
    # for every checkpoint here) so build_square's seeded RNG reproduces the
    # EXACT train/eval split those checkpoints were actually trained/evaluated
    # on -- otherwise "train_Nd" below would just be freshly-sampled examples
    # that happen to share a length, not examples the model actually saw.
    data = build_square(
        train_digits=tuple(train_digits), ood_digits=tuple(ood_digits),
        n_train=args.n_train, n_eval=args.n_eval, carryless=carryless,
    )
    max_seq_len = data.max_seq_len

    cohort_meta = {train_cohort: (id_digit, "trained (seen x)"),
                   id_cohort: (id_digit, offset_category(id_digit, train_digits))}
    for d in ood_digits:
        cohort_meta[f"ood_{d}d"] = (d, offset_category(d, train_digits))
    cohort_names = list(cohort_meta)

    results = {}
    train_records = [r for r in data.train if x_digit_length(r) == id_digit][:args.n_eval]
    if train_records:
        tensors = tensors_from_records(train_records, data.tokenizer, max_seq_len)
        results[train_cohort] = analyze_cohort(model, tensors, device)
    else:
        print(f"  (no length-{id_digit} examples found in data.train at n_train="
              f"{args.n_train} -- skipping {train_cohort}; try a larger --n-train)")
    for cohort in data.cohorts:
        if cohort.name not in cohort_meta:
            continue
        tensors = tensors_from_records(cohort.records, data.tokenizer, max_seq_len)
        results[cohort.name] = analyze_cohort(model, tensors, device)

    print(f"\n{'cohort':<12}{'category':<18}{'layer':>6}{'entropy':>10}{'top_prob':>10}"
          f"{'hidden_rms':>12}{'offset_mean':>13}{'offset_std':>13}")
    for name in cohort_names:
        if name not in results:
            continue
        _, cat = cohort_meta[name]
        for layer in results[name]["per_layer"]:
            print(f"{name:<12}{cat:<18}{layer['layer']:>6}{layer['mean_entropy_bits']:>10.3f}"
                  f"{layer['mean_top_attn_prob']:>10.3f}{layer['mean_hidden_rms']:>12.3f}"
                  f"{layer['offset_mean']:>13.2f}{layer['offset_std']:>13.2f}")

    final = len(results[id_cohort]["per_layer"]) - 1
    print(f"\n  per-head, final layer (layer {final}):")
    print(f"  {'cohort':<12}{'category':<18}" + "".join(f"head{h:<7}" for h in
          range(len(results[id_cohort]["per_layer"][final]["entropy_per_head"]))))
    for name in cohort_names:
        if name not in results:
            continue
        _, cat = cohort_meta[name]
        ent_h = results[name]["per_layer"][final]["entropy_per_head"]
        print(f"  {name:<12}{cat:<18}" + "".join(f"{e:<11.3f}" for e in ent_h) + "  (entropy, bits)")
    for name in cohort_names:
        if name not in results:
            continue
        _, cat = cohort_meta[name]
        prob_h = results[name]["per_layer"][final]["top_prob_per_head"]
        print(f"  {name:<12}{cat:<18}" + "".join(f"{p:<11.3f}" for p in prob_h) + "  (top_attn_prob)")

    print(f"\n  === {id_cohort} (ID, unseen x) vs each cohort, final layer ===")
    id_final = results[id_cohort]["per_layer"][final]
    for name in cohort_names:
        if name == id_cohort or name not in results:
            continue
        _, cat = cohort_meta[name]
        f_ = results[name]["per_layer"][final]
        d_ent = f_["mean_entropy_bits"] - id_final["mean_entropy_bits"]
        d_prob = f_["mean_top_attn_prob"] - id_final["mean_top_attn_prob"]
        print(f"  {name} [{cat}]: d_entropy={d_ent:+.3f}  d_top_prob={d_prob:+.3f}")

    if args.plot:
        plot_checkpoint(ckpt_name, results, cohort_names, id_cohort, cohort_meta)

    if args.attention_maps:
        attn_maps = {}
        if train_records:
            tensors = tensors_from_records(train_records, data.tokenizer, max_seq_len)
            attn_maps[train_cohort] = mean_attention_maps(model, tensors, device)
        for cohort in data.cohorts:
            if cohort.name not in cohort_meta:
                continue
            tensors = tensors_from_records(cohort.records, data.tokenizer, max_seq_len)
            attn_maps[cohort.name] = mean_attention_maps(model, tensors, device)
        plot_attention_maps(ckpt_name, attn_maps, cohort_names, cohort_meta)


def plot_checkpoint(ckpt_name, results, cohort_names, id_cohort, cohort_meta) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    tag = Path(ckpt_name).stem
    colors = {id_cohort: "#104281"}
    ood_colors = ["#B03A2E", "#7A1F14", "#D68910", "#6B7280"]
    others = [n for n in cohort_names if n != id_cohort]
    for i, n in enumerate(others):
        colors[n] = ood_colors[i % len(ood_colors)]

    n_layers = len(results[id_cohort]["per_layer"])
    n_heads = len(results[id_cohort]["per_layer"][0]["entropy_per_head"])

    # per-head entropy heatmaps: one subplot per cohort, layer x head grid
    figure, axes = plt.subplots(1, len(cohort_names), figsize=(4.2 * len(cohort_names), 4),
                                sharey=True)
    if len(cohort_names) == 1:
        axes = [axes]
    for ax, name in zip(axes, cohort_names):
        if name not in results:
            continue
        grid = np.array([layer["entropy_per_head"] for layer in results[name]["per_layer"]])
        im = ax.imshow(grid, cmap="viridis", aspect="auto", vmin=0.5, vmax=3.5)
        _, cat = cohort_meta[name]
        ax.set_title(f"{name}  [{cat}]", fontsize=10)
        ax.set_xlabel("head"); ax.set_xticks(range(n_heads))
        ax.set_yticks(range(n_layers))
    axes[0].set_ylabel("layer")
    figure.colorbar(im, ax=axes, label="entropy (bits)", shrink=0.8)
    figure.suptitle(f"{tag}: per-head attention entropy at read-out slots",
                    fontsize=12.5, y=1.03, x=0.01, ha="left")
    out = RESULTS / "plots" / f"internals_{tag}_perhead_entropy.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    print(f"  wrote {out}")

    # offset histograms, final layer -- where attention actually points
    figure2, axes2 = plt.subplots(1, len(cohort_names), figsize=(5 * len(cohort_names), 4),
                                  sharey=True)
    if len(cohort_names) == 1:
        axes2 = [axes2]
    for ax, name in zip(axes2, cohort_names):
        if name not in results:
            continue
        offsets = results[name]["per_layer"][-1]["offsets"].numpy()
        _, cat = cohort_meta[name]
        ax.hist(offsets, bins=60, color=colors[name])
        ax.set_title(f"{name}  [{cat}]  (final layer)", fontsize=10)
        ax.set_xlabel("top-attended key pos - query pos")
        ax.grid(alpha=0.25, linewidth=0.6); ax.set_axisbelow(True)
    axes2[0].set_ylabel("count")
    figure2.suptitle(f"{tag}: where the top-attended key sits, relative to the read-out query",
                     fontsize=12.5, y=1.03, x=0.01, ha="left")
    figure2.tight_layout()
    out2 = RESULTS / "plots" / f"internals_{tag}_offsets.png"
    figure2.savefig(out2, dpi=170, bbox_inches="tight", facecolor="white")
    print(f"  wrote {out2}")


def plot_attention_maps(ckpt_name, attn_maps, cohort_names, cohort_meta) -> None:
    """The actual attention matrices (query x key), averaged over the batch,
    one grid (layers x heads) per cohort -- not summary statistics, the maps
    themselves. Position-driven structure (a diagonal band, a fixed-offset
    stripe) survives averaging over many different-content examples; pure
    content-driven attention would wash out to noise instead."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tag = Path(ckpt_name).stem
    n_layers = len(next(iter(attn_maps.values()))["maps"])
    n_heads = next(iter(attn_maps.values()))["maps"][0].size(0)

    for name in cohort_names:
        if name not in attn_maps:
            continue
        info = attn_maps[name]
        _, cat = cohort_meta[name]
        figure, axes = plt.subplots(n_layers, n_heads,
                                    figsize=(2.6 * n_heads, 2.6 * n_layers),
                                    squeeze=False)
        for li in range(n_layers):
            for h in range(n_heads):
                ax = axes[li][h]
                m = info["maps"][li][h].numpy()
                ax.imshow(m, cmap="viridis", vmin=0, vmax=max(m.max(), 1e-6), aspect="equal")
                if info["field_split"] is not None:
                    ax.axhline(info["field_split"] - 0.5, color="white", lw=0.6, alpha=0.6)
                    ax.axvline(info["field_split"] - 0.5, color="white", lw=0.6, alpha=0.6)
                ax.axhline(info["target_start"] - 0.5, color="#FF6B6B", lw=0.6, alpha=0.7)
                ax.axvline(info["target_start"] - 0.5, color="#FF6B6B", lw=0.6, alpha=0.7)
                ax.set_xticks([]); ax.set_yticks([])
                if li == 0:
                    ax.set_title(f"head {h}", fontsize=9)
                if h == 0:
                    ax.set_ylabel(f"layer {li}", fontsize=9)
        figure.suptitle(
            f"{tag} / {name} [{cat}]  --  mean attention (query rows x key cols); "
            f"white=X/Y field split, red=read-out window start",
            fontsize=11, y=1.01, x=0.01, ha="left")
        figure.tight_layout()
        out = RESULTS / "plots" / f"internals_{tag}_{name}_attnmaps.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
        plt.close(figure)
        print(f"  wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--n-train", type=int, default=100_000,
                        help="must match exp2_square.py's --n-train at the time the "
                             "checkpoint was trained (its default, unless overridden) "
                             "for the seeded RNG to reproduce the real train/eval split")
    parser.add_argument("--n-eval", type=int, default=500)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--attention-maps", action="store_true",
                        help="plot the actual (query x key) attention matrices per "
                             "layer/head/cohort, averaged over the batch")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = pick_device(args.device)
    for ckpt_name in args.checkpoints:
        analyze_checkpoint(ckpt_name, args, device)


if __name__ == "__main__":
    main()
