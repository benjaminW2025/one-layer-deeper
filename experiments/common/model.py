"""Bidirectional encoder with pluggable positional information.

Same shape as the competition baseline (pre-LN, padding mask only, logits at
every position). The only thing that varies is how position is encoded, which
is the subject of experiment 1.

Schemes
-------
none        no positional signal. In a bidirectional encoder this makes the
            model permutation-invariant, so it should fail. Kept as a floor.
learned     nn.Embedding(max_seq_len, d). What the competition baseline uses.
sinusoidal  fixed sin/cos absolute positions.
rope        rotary embeddings applied to q/k inside attention (relative).
abacus      learned embedding of each digit's index *within its own number*,
            counted from the right, i.e. its place value. Adds no absolute
            position at all.
abacus_rope abacus for place value + rope for field order.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .format import DIGIT_OFFSET

POSITIONAL_MODES = (
    "none", "learned", "sinusoidal", "rope", "abacus", "abacus_rope",
)


def digit_index_from_right(is_digit: torch.Tensor) -> torch.Tensor:
    """For each digit token, its index within its own contiguous digit run,
    counted from the right. That index IS the base-10 place value.

    [N] 1 8 9 1 [X] 1 8 9 0 [T] 6 4
     .  3 2 1 0  .  3 2 1 0  .  1 0
    """
    length = is_digit.size(1)
    positions = torch.arange(length, device=is_digit.device).expand_as(is_digit)
    flipped = is_digit.flip(1)
    # Index of the most recent non-digit at or before each position.
    marker = torch.where(flipped, torch.full_like(positions, -1), positions)
    last_marker = marker.cummax(dim=1).values
    index_in_run = positions - last_marker - 1
    return (index_in_run * flipped).flip(1)


def sinusoidal_table(length: int, width: int) -> torch.Tensor:
    position = torch.arange(length).unsqueeze(1).float()
    scale = torch.exp(torch.arange(0, width, 2).float() * (-math.log(10000.0) / width))
    table = torch.zeros(length, width)
    table[:, 0::2] = torch.sin(position * scale)
    table[:, 1::2] = torch.cos(position * scale)
    return table


def rope_tables(length: int, head_dim: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    inverse = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    angles = torch.arange(length, device=device).float().unsqueeze(1) * inverse.unsqueeze(0)
    embedded = torch.cat([angles, angles], dim=-1)
    return embedded.cos()[None, None], embedded.sin()[None, None]


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.size(-1) // 2
    rotated = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
    return x * cos + rotated * sin


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, use_rope: bool) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.use_rope = use_rope
        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x, keep_mask, rope=None):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)

        def heads(t):
            return t.view(batch, length, self.n_heads, width // self.n_heads).transpose(1, 2)

        q, k, v = heads(q), heads(k), heads(v)
        if self.use_rope and rope is not None:
            cos, sin = rope
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=keep_mask)
        x = x + self.proj(attended.transpose(1, 2).reshape(batch, length, width))
        return x + self.mlp(self.ln2(x))


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        positional: str = "learned",
        abacus_max_offset: int = 8,
    ) -> None:
        super().__init__()
        if positional not in POSITIONAL_MODES:
            raise ValueError(f"positional must be one of {POSITIONAL_MODES}")
        if d_model % n_heads:
            raise ValueError("d_model must divide evenly into heads")
        self.positional = positional
        self.max_seq_len = max_seq_len
        self.abacus_max_offset = abacus_max_offset
        self.head_dim = d_model // n_heads

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        if positional == "learned":
            self.position_embedding = nn.Embedding(max_seq_len, d_model)
        elif positional == "sinusoidal":
            self.register_buffer(
                "position_table", sinusoidal_table(max_seq_len, d_model), persistent=False
            )
        if positional.startswith("abacus"):
            # +offset headroom so random training offsets stay in range.
            self.abacus_embedding = nn.Embedding(max_seq_len + abacus_max_offset + 1, d_model)

        self.blocks = nn.ModuleList(
            Block(d_model, n_heads, use_rope=positional in ("rope", "abacus_rope"))
            for _ in range(n_layers)
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        length = input_ids.size(1)
        x = self.token_embedding(input_ids)

        if self.positional == "learned":
            x = x + self.position_embedding(torch.arange(length, device=input_ids.device))
        elif self.positional == "sinusoidal":
            x = x + self.position_table[:length].to(x.dtype)

        if self.positional.startswith("abacus"):
            is_digit = input_ids >= DIGIT_OFFSET
            index = digit_index_from_right(is_digit)
            if self.training and self.abacus_max_offset > 0:
                # One random shift per sequence: the model must read place
                # value from relative structure, not from absolute index.
                shift = torch.randint(
                    0, self.abacus_max_offset + 1, (input_ids.size(0), 1),
                    device=input_ids.device,
                )
                index = index + shift * is_digit
            x = x + self.abacus_embedding(index) * is_digit.unsqueeze(-1)

        rope = None
        if self.positional in ("rope", "abacus_rope"):
            rope = rope_tables(length, self.head_dim, input_ids.device)
            rope = (rope[0].to(x.dtype), rope[1].to(x.dtype))

        keep_mask = attention_mask[:, None, None, :]
        for block in self.blocks:
            x = block(x, keep_mask, rope)
        return self.head(self.ln_f(x))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
