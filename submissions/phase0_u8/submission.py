"""Phase 0 U8: eight untied full-token Transformer blocks."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from benchmark import (
    ModelSpec,
    OptimizerBundle,
    OptimizerSpec,
    Submission,
    assert_model_state,
)


D_MODEL = 256
NUM_HEADS = 4
NUM_LAYERS = 8
MLP_HIDDEN = 4 * D_MODEL
ROPE_BASE = 1000.0
EMBED_INIT_STD = 0.02


def rope_tables(
    length: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    half = head_dim // 2
    inv_freq = ROPE_BASE ** (
        -torch.arange(half, device=device, dtype=torch.float32) / half
    )
    positions = torch.arange(length, device=device, dtype=torch.float32)
    angles = positions[:, None] * inv_freq[None, :]
    angles = torch.cat((angles, angles), dim=-1)
    return angles.cos().to(dtype), angles.sin().to(dtype)


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.shape[-1] // 2
    rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return x * cos + rotated * sin


class Config:
    def __init__(self, vocab_size: int, max_seq_len: int) -> None:
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len


class RMSNorm(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.shape[-1],), self.weight)


class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(D_MODEL)
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL)
        self.out = nn.Linear(D_MODEL, D_MODEL)
        self.mixer_norm = RMSNorm(D_MODEL)
        self.up = nn.Linear(D_MODEL, MLP_HIDDEN)
        self.down = nn.Linear(MLP_HIDDEN, D_MODEL)

    def forward(
        self,
        x: Tensor,
        attention_mask: Tensor | None,
        rope: tuple[Tensor, Tensor],
    ) -> Tensor:
        residual = x
        x = self.attention_norm(x)
        batch, length, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch, length, NUM_HEADS, -1).transpose(1, 2)
        k = k.view(batch, length, NUM_HEADS, -1).transpose(1, 2)
        v = v.view(batch, length, NUM_HEADS, -1).transpose(1, 2)
        cos, sin = rope
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        mask = None
        if attention_mask is not None:
            if attention_mask.shape != (batch, length):
                raise ValueError("invalid attention_mask shape")
            mask = attention_mask[:, None, None, :].bool()

        x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x.transpose(1, 2).contiguous().view(batch, length, D_MODEL)
        x = residual + self.out(x)
        return x + self.down(F.gelu(self.up(self.mixer_norm(x))))


class Model(nn.Module):
    num_loops = 1

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.token_embedding = nn.Embedding(spec.vocab_size, D_MODEL)
        nn.init.normal_(self.token_embedding.weight, std=EMBED_INIT_STD)
        self.blocks = nn.ModuleList(Block() for _ in range(NUM_LAYERS))
        self.final_norm = RMSNorm(D_MODEL)
        # Input and output embeddings are deliberately untied. Reading a digit
        # in the prompt and classifying an answer digit are different roles.
        self.head = nn.Linear(D_MODEL, spec.vocab_size, bias=False)
        nn.init.normal_(self.head.weight, std=EMBED_INIT_STD)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, None]:
        x = self.token_embedding(input_ids)
        rope = rope_tables(
            input_ids.shape[1],
            D_MODEL // NUM_HEADS,
            input_ids.device,
            x.dtype,
        )
        for block in self.blocks:
            x = block(x, attention_mask, rope)
        return self.head(self.final_norm(x)), None


def build_model(spec: ModelSpec) -> Model:
    model = Model(spec)
    assert_model_state(model, spec)
    return model


def build_optimizer(model: nn.Module, spec: OptimizerSpec) -> OptimizerBundle:
    return OptimizerBundle(
        torch.optim.AdamW(
            model.parameters(),
            lr=3e-3,
            betas=(0.9, 0.95),
            weight_decay=0.1,
            capturable=spec.device_type == "cuda",
        )
    )


SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
    batch_size=512,
    eval_batch_size=512,
)
