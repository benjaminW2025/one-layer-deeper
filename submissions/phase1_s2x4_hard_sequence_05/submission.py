"""Phase 1 S2x4 scratchpad with exponent-0.5 hard-sequence CE."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from benchmark import (
    ModelSpec,
    OptimizerBundle,
    OptimizerSpec,
    Submission,
    TokenLossBatch,
    assert_model_state,
)


D_MODEL = 256
NUM_HEADS = 4

# Two different update rules, applied in this order eight times. The prompt is
# kept fixed; only the scratchpad changes during these rounds.
NUM_RECURRENT_LAYERS = 2
NUM_RECURRENCES = 4
NUM_SCRATCH_TOKENS = 4
# Local controlled experiments can enable this before constructing Model. Keep
# the submission baseline at zero so prompt reading is an explicit ablation.
NUM_PROMPT_READER_LAYERS = 0

# Applied per-Block, not to the whole forward, so compile cost is O(1) in depth
# instead of O(NUM_LAYERS) -- a tied recurrent block compiles once no matter how
# many times it is applied.
#
# "default" gets inductor fusion. "reduce-overhead" adds CUDA graphs, which is
# what really attacks launch overhead, but capture is re-paid per distinct input
# shape: evaluation walks ~16 splits (test, ood, 7 depth rungs, 7 OOD-N rungs) at
# differing lengths and ragged final batches, all inside one 30s deadline that
# raises rather than degrades. That is what failed the 8-layer submission.
# Compilation was too expensive for Easy's 60-second budget, but is worth
# retesting for Medium's 10-minute budget.
COMPILE_MODE: str | None = None

# Rotary base. The standard 10000 is tuned for contexts in the thousands; these
# prompts are ~20 tokens, so most of the spectrum rotates barely at all across a
# sequence. Lower values spend more of the head dimension on positions that
# actually occur here.
ROPE_BASE = 1000.0 # EXPERIMENT WITH

# Attention only ever produces weighted *sums* of values, so any digit product
# x_i * x_j has to be formed in the pointwise path. A gated unit computes
# products directly; GELU only approximates them.
#   "swiglu"   -> silu(W_a h) * (W_b h)
#   "bilinear" -> (W_a h) * (W_b h), the purest product form
#   "gelu"     -> the original
MLP_KIND = "gelu" # try bilinear next

MLP_HIDDEN = 4 * D_MODEL

# Answer digits are emitted at positions holding arbitrary prompt tokens, so the
# geometry for *reading* a token and for *classifying* it are unrelated here --
# unlike a normal LM, where tying helps. Untying costs vocab_size * D_MODEL.
TIE_EMBEDDINGS = False

# nn.Embedding initialises to N(0, 1). Tied to the head that puts initial logits
# at scale ~sqrt(D_MODEL) = 16, which is why step-1 loss is ~165 rather than
# ln(17) = 2.83; the first steps are spent shrinking the embedding norm. Setting
# this decouples that from the tying question. None keeps PyTorch's default so
# earlier runs stay reproducible.
EMBED_INIT_STD: float | None = 0.02


def rope_tables(
    length: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    """cos/sin of shape [length, head_dim], broadcastable over [B, H, L, D]."""
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


class PromptReaderBlock(nn.Module):
    """Lets prompt digits learn about one another before recurrence begins."""

    def __init__(self) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(D_MODEL)
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL)
        self.out = nn.Linear(D_MODEL, D_MODEL)
        self.mixer_norm = RMSNorm(D_MODEL)
        self.up = nn.Linear(D_MODEL, 4 * D_MODEL)
        self.down = nn.Linear(4 * D_MODEL, D_MODEL)

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
        q = q.view(batch, length, NUM_HEADS, -1).transpose(1, 2)
        k = k.view(batch, length, NUM_HEADS, -1).transpose(1, 2)
        v = v.view(batch, length, NUM_HEADS, -1).transpose(1, 2)
        cos, sin = rope
        q = apply_rope(q, cos[:length], sin[:length])
        k = apply_rope(k, cos[:length], sin[:length])
        if attention_mask is not None:
            if attention_mask.shape != (batch, length):
                raise ValueError("invalid attention_mask shape")
            mask = attention_mask[:, None, None, :].bool()
        else:
            mask = None
        context = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        context = context.transpose(1, 2).contiguous().view(batch, length, D_MODEL)
        context = residual + self.out(context)
        return context + self.down(F.gelu(self.up(self.mixer_norm(context))))


class ScratchpadBlock(nn.Module):
    """One update of the mutable scratchpad.

    Queries come only from the scratchpad.  The fixed prompt and the previous
    scratchpad are keys/values, so this module has no way to overwrite prompt
    tokens while it works.
    """

    def __init__(self) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(D_MODEL)
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL)
        self.out = nn.Linear(D_MODEL, D_MODEL)
        self.mixer_norm = RMSNorm(D_MODEL)
        if MLP_KIND == "gelu":
            self.up = nn.Linear(D_MODEL, 4 * D_MODEL)
            self.down = nn.Linear(4 * D_MODEL, D_MODEL)
        else:
            # One projection producing both halves: cheaper than two matmuls and
            # it keeps the gate and value paths sharing an input read.
            self.up = nn.Linear(D_MODEL, 2 * MLP_HIDDEN)
            self.down = nn.Linear(MLP_HIDDEN, D_MODEL)

    def mlp(self, h: Tensor) -> Tensor:
        if MLP_KIND == "gelu":
            return self.down(F.gelu(self.up(h)))
        gate, value = self.up(h).chunk(2, dim=-1)
        if MLP_KIND == "swiglu":
            gate = F.silu(gate)
        return self.down(gate * value)

    def forward(
        self,
        state: Tensor,
        context: Tensor,
        attention_mask: Tensor | None,
        rope: tuple[Tensor, Tensor],
    ) -> Tensor:
        residual = state
        state = self.attention_norm(state)
        batch, state_length, _ = state.shape
        context_length = context.shape[1]
        # The single projection is split so state supplies queries, while both
        # sources supply keys and values.
        q = self.qkv(state)[..., :D_MODEL]
        kv = self.qkv(torch.cat((context, state), dim=1))
        k, v = kv[..., D_MODEL : 2 * D_MODEL], kv[..., 2 * D_MODEL :]
        q = q.view(batch, state_length, NUM_HEADS, -1).transpose(1, 2)
        k = k.view(batch, context_length + state_length, NUM_HEADS, -1).transpose(1, 2)
        v = v.view(batch, context_length + state_length, NUM_HEADS, -1).transpose(1, 2)
        cos, sin = rope
        # Scratchpad positions are placed immediately after the prompt.
        q = apply_rope(q, cos[context_length:], sin[context_length:])
        k = apply_rope(k, cos, sin)
        if attention_mask is not None:
            if attention_mask.shape != (batch, context_length):
                raise ValueError("invalid attention_mask shape")
            # Padding is hidden, but every scratchpad slot is always visible.
            state_mask = torch.ones(
                (batch, state_length), device=state.device, dtype=torch.bool
            )
            mask = torch.cat((attention_mask.bool(), state_mask), dim=1)
            mask = mask[:, None, None, :]
        else:
            mask = None
        state = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        state = state.transpose(1, 2).contiguous().view(batch, state_length, D_MODEL)
        state = residual + self.out(state)
        return state + self.mlp(self.mixer_norm(state))


class Readout(nn.Module):
    """Lets each output position read the fixed prompt and final scratchpad."""

    def __init__(self) -> None:
        super().__init__()
        self.norm = RMSNorm(D_MODEL)
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL)
        self.out = nn.Linear(D_MODEL, D_MODEL)

    def forward(
        self,
        context: Tensor,
        state: Tensor,
        attention_mask: Tensor | None,
        rope: tuple[Tensor, Tensor],
    ) -> Tensor:
        batch, context_length, _ = context.shape
        state_length = state.shape[1]
        query = self.norm(context)
        q = self.qkv(query)[..., :D_MODEL]
        kv = self.qkv(torch.cat((context, state), dim=1))
        k, v = kv[..., D_MODEL : 2 * D_MODEL], kv[..., 2 * D_MODEL :]
        q = q.view(batch, context_length, NUM_HEADS, -1).transpose(1, 2)
        k = k.view(batch, context_length + state_length, NUM_HEADS, -1).transpose(1, 2)
        v = v.view(batch, context_length + state_length, NUM_HEADS, -1).transpose(1, 2)
        cos, sin = rope
        q = apply_rope(q, cos[:context_length], sin[:context_length])
        k = apply_rope(k, cos, sin)
        if attention_mask is not None:
            state_mask = torch.ones(
                (batch, state_length), device=context.device, dtype=torch.bool
            )
            mask = torch.cat((attention_mask.bool(), state_mask), dim=1)
            mask = mask[:, None, None, :]
        else:
            mask = None
        read = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        read = read.transpose(1, 2).contiguous().view(batch, context_length, D_MODEL)
        return context + self.out(read)


class Model(nn.Module):
    num_loops = 1

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.token_embedding = nn.Embedding(spec.vocab_size, D_MODEL)
        self.prompt_reader = nn.ModuleList(
            PromptReaderBlock() for _ in range(NUM_PROMPT_READER_LAYERS)
        )
        self.scratchpad = nn.Parameter(torch.empty(NUM_SCRATCH_TOKENS, D_MODEL))
        nn.init.normal_(self.scratchpad, std=0.02)
        self.recurrent_blocks = nn.ModuleList(
            ScratchpadBlock() for _ in range(NUM_RECURRENT_LAYERS)
        )
        self.readout = Readout()
        self.final_norm = RMSNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, spec.vocab_size, bias=False)
        if TIE_EMBEDDINGS:
            self.head.weight = self.token_embedding.weight
        if EMBED_INIT_STD is not None:
            # When tied, this re-initialises the head too -- same tensor.
            nn.init.normal_(self.token_embedding.weight, std=EMBED_INIT_STD)
            if not TIE_EMBEDDINGS:
                nn.init.normal_(self.head.weight, std=EMBED_INIT_STD)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, None]:
        context = self.token_embedding(input_ids)
        batch, context_length, _ = context.shape
        state = self.scratchpad.unsqueeze(0).expand(batch, -1, -1)
        rope = rope_tables(
            context_length + NUM_SCRATCH_TOKENS,
            D_MODEL // NUM_HEADS,
            context.device,
            context.dtype,
        )
        for block in self.prompt_reader:
            context = block(context, attention_mask, rope)
        for _ in range(NUM_RECURRENCES):
            for block in self.recurrent_blocks:
                state = block(state, context, attention_mask, rope)
        x = self.readout(context, state, attention_mask, rope)
        return self.head(self.final_norm(x)), None


def build_model(spec: ModelSpec) -> Model:
    model = Model(spec)
    assert_model_state(model, spec)
    if COMPILE_MODE is not None and torch.cuda.is_available():
        # Each tied scratchpad block is compiled once, regardless of the number
        # of recurrent uses.
        for block in model.prompt_reader:
            block.forward = torch.compile(block.forward, mode=COMPILE_MODE)
        for block in model.recurrent_blocks:
            block.forward = torch.compile(block.forward, mode=COMPILE_MODE)
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


def token_training_loss(batch: TokenLossBatch) -> Tensor:
    """Give moderately greater weight to difficult complete answers."""
    token_losses = F.cross_entropy(
        batch.logits.transpose(1, 2),
        batch.labels,
        ignore_index=-100,
        reduction="none",
    )
    valid = batch.valid_mask.to(token_losses.dtype)
    target_counts = batch.valid_mask.sum(dim=1)
    sequence_losses = (token_losses * valid).sum(dim=1) / target_counts.clamp_min(1)
    sequence_losses = sequence_losses[target_counts > 0]
    weights = sequence_losses.detach().clamp_min(1e-8).sqrt()
    weights = weights / weights.mean().clamp_min(1e-8)
    return (weights * sequence_losses).mean()


SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
    token_training_loss=token_training_loss,
    batch_size=512,
    eval_batch_size=512,
)
