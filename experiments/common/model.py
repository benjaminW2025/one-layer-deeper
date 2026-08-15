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
abacus_learned  abacus place value + learned absolute position. This is the
            combination used in "Transformers Can Do Arithmetic with the Right
            Embeddings" (McLeish et al., NeurIPS 2024, arXiv 2405.17399), whose
            best results add Abacus alongside a standard positional embedding
            rather than replacing it.
abacus_rope abacus for place value + relative rope for field order.

rope_sum    RoPE with the query rotation negated, so attention scores depend on
            m+n rather than m-n. Multiplication is a convolution in place-value
            space (answer place k collects pairs with i+j=k), and those pairs
            sit at constant m+n, so this geometry can express the multiplication
            pairing that relative position cannot. It gives up translation
            invariance, which is what makes relative RoPE extrapolate in length.
rope_mixed  half the heads relative, half sum: both geometries at once.
abacus_rope_sum / abacus_rope_mixed
            the same two, with Abacus place value added.

rope_significance / rope_significance_sum / rope_significance_mixed
            RoPE, but the rotation angle is each digit's place value (distance
            from the ones digit of ITS OWN number, counted from the right;
            Abacus's index, not embedded but fed straight into the rotation)
            instead of raw sequence position. Non-digit tokens (markers) all
            rotate by angle 0.

            Plain rope's m-n is "how far apart in the sequence", which is
            affine in place value but with an offset that depends on the
            field's start position -- itself a function of how many digits
            precede it, which shifts with digit length. That is a plausible
            reason relative RoPE fails to extrapolate to unseen lengths: the
            model has never seen the offsets that longer inputs produce, even
            though the *place-value* relationship it needs is unchanged.
            rope_significance removes that offset entirely: two digits of
            equal significance always rotate identically, no matter how many
            digits are in front of them or how long the number is. Same
            trick as rope_sum's m+n insight, but applied at the level RoPE
            actually needs it -- place value, not raw position.

n_loops (orthogonal flag, any positional mode; default 1 = no change)
            `n_layers` becomes the number of UNIQUE Block instances; the
            forward pass runs through that stack `n_loops` times, weights
            shared across iterations (universal-transformer-style
            recurrence). Two hypotheses this tests: (1) a loop is a single
            generalizable "step" the model can re-apply, so OOD sequence
            lengths might just need more of the same step rather than
            positions/weights it never trained; (2) more sequential steps
            gives the model room to do carry propagation, which is
            inherently iterative (add, check overflow, carry, repeat) and
            may not fit in a fixed handful of feedforward layers. The loop
            count is a `forward()` argument, not baked into the model, so it
            can be scaled up at eval time on longer inputs without retraining.

attention_sink (orthogonal flag, any positional mode)
            each head gets a learned per-head scalar competing in the softmax
            denominator alongside the real keys (the "off-by-one softmax" /
            attention-sink trick), so a head can dump attention mass on
            nothing instead of being forced to spread weight over digits
            that aren't actually relevant to the query. Initialized so it
            contributes exactly 1, matching plain softmax at step 0.

rope_dual   first half of the heads plain rel-RoPE on raw sequence position,
            second half rel-RoPE on significance. Different axis of mixing
            than rope_mixed (which mixes rel vs sum within ONE position
            source) -- this mixes the position SOURCE itself. The idea:
            rope_significance alone conflates "what digit am I" (needs
            significance, so pairing is length-invariant) with "which
            output place am I" (needs a globally unique address, which
            significance doesn't give -- see rope_significance's docstring)
            into a single number, and query/key are forced to share it.
            Splitting across heads gives the model both signals separately
            and lets attention combine them however training finds useful,
            instead of hand-deriving the combination (e.g. distance from the
            end of the sequence, added only at read-out slots).

Note on indexing direction: the paper encodes a digit's position relative to
the START of its number, which equals place value there because their inputs
are least-significant-digit first. Our digit order is MSD-first (fixed by the
competition), so we index from the RIGHT end of each digit run instead. That
preserves the property that matters — digits of equal significance share an
embedding — rather than the literal rule.
"""

from __future__ import annotations

import math
import random

import torch
import torch.nn.functional as F
from torch import nn

POSITIONAL_MODES = (
    "none", "learned", "sinusoidal",
    "rope", "rope_sum", "rope_mixed",
    "abacus", "abacus_learned",
    "abacus_rope", "abacus_rope_sum", "abacus_rope_mixed",
    "rope_significance", "rope_significance_sum", "rope_significance_mixed",
    "rope_dual",
)
# Which components each mode switches on.
USES_LEARNED = ("learned", "abacus_learned")

# How each mode rotates q/k.
#   rel    standard RoPE: q and k both by +theta, so the score depends on m-n
#          (relative position). Translation invariant; extrapolates in length.
#   sum    q by -theta, k by +theta, so the score depends on m+n.
#          Multiplication is a convolution in place-value space -- answer place
#          k collects pairs with i+j=k -- and since place = field_start+len-1-pos,
#          those pairs sit at constant m+n. So this geometry can express the
#          multiplication pairing that relative position cannot. It gives up
#          translation invariance to do so.
#   mixed  first half of the heads rel, second half sum: both geometries at once.
ROPE_MODE = {
    "rope": "rel", "abacus_rope": "rel", "rope_significance": "rel",
    "rope_sum": "sum", "abacus_rope_sum": "sum", "rope_significance_sum": "sum",
    "rope_mixed": "mixed", "abacus_rope_mixed": "mixed", "rope_significance_mixed": "mixed",
    "rope_dual": "dual",
}
# rope_significance* rotates by place value (Abacus's index) instead of raw
# sequence position -- see module docstring.
USES_SIGNIFICANCE_ROPE = (
    "rope_significance", "rope_significance_sum", "rope_significance_mixed",
)


def digit_run_positions(mask: torch.Tensor, device: torch.device) -> torch.Tensor:
    """1-indexed position within each run of consecutive True values, left to
    right, 0 elsewhere. Shared by Abacus (embeds this) and significance RoPE
    (rotates by this directly). See `Abacus` for the reverse-indexing trick
    that turns this into place value for MSD-first digit runs."""
    mask_shape = mask.shape

    # Create a shifted version of the mask to detect changes from 0 to 1
    shifted_mask = torch.cat(
        [torch.zeros((mask_shape[0], 1), device=device, dtype=mask.dtype),
         mask[:, :-1]], dim=1
    )
    starts = (shifted_mask != mask) & mask

    # Generate IDs for each segment of 1s, processing row-wise
    segment_ids = torch.cumsum(starts, dim=1)

    # Generate an index array row-wise
    index = torch.arange(mask.size(1)).repeat(mask.size(0), 1).to(device)

    # Reset index at the start of each segment
    reset_index = torch.zeros_like(mask).long()
    second_term = index * starts.long()
    reset_index = reset_index.scatter_add(1, segment_ids, second_term)

    # Calculate positions in segment
    positions = index - reset_index.gather(1, segment_ids) + 1

    # Ensure only values within 1-segments are non-zero
    return positions * mask


def significance_index(
    input_ids: torch.Tensor, digit_range: tuple[int, int],
    training: bool = False, max_k: int = 0,
) -> torch.Tensor:
    """Each digit's place value (1 = ones digit, 2 = tens, ...), counted from
    the right within its own number; 0 for non-digit tokens. Same reversed
    indexing as Abacus, without the embedding -- fed straight into RoPE's
    rotation angle so digits of equal significance rotate identically no
    matter how many digits, or how many other fields, precede them."""
    lo, hi = digit_range
    mask = (input_ids >= lo) & (input_ids < hi)
    mask = mask.flip(1)
    positions = digit_run_positions(mask, input_ids.device)
    positions = positions.flip(1)
    if training and max_k > 0:
        k = random.randint(0, max_k)
        positions = torch.where(positions > 0, positions + k, positions)
    return positions


class Abacus(nn.Module):
    """Port of https://github.com/mcleish7/arithmetic/blob/main/abacus.py

    `helper` is their algorithm verbatim: it turns a binary digit mask into
    1-indexed positions within each run of consecutive digits, left to right,
    with 0 for non-digit tokens (which therefore all share embedding[0]).

    ONE adaptation. Their docstring says "Integers must be reversed for this to
    work correctly" — their inputs are least-significant-digit first, so a
    left-to-right index equals place value. Our digit order is MSD-first and
    fixed by the competition, so `reverse=True` flips the mask, runs their
    helper, and flips back. That yields place value for our order; using their
    raw direction would index the most significant digit as 1, which does not
    align digits of equal significance across numbers of different lengths.

    `max_k` also needs rescaling. Theirs is 99 against operands up to ~120
    digits, so the random shift is comparable to the index range. Our numbers
    are <=7 digits, where a 0-99 shift would swamp the signal entirely, so the
    default here is small. Pass --abacus-max-k to sweep it.
    """

    def __init__(
        self,
        digit_range: tuple[int, int],
        embedding_dim: int,
        max_seq_length: int = 1024,
        max_k: int = 8,
        reverse: bool = True,
    ) -> None:
        super().__init__()
        self.digit_lo, self.digit_hi = digit_range
        self.embedding = nn.Embedding(max_seq_length, embedding_dim)
        self.max_k = max_k
        self.reverse = reverse

    def helper(self, mask: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Converts a binary mask of digit locations into spans of consecutive digits."""
        return digit_run_positions(mask, device)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        mask = (input_ids >= self.digit_lo) & (input_ids < self.digit_hi)
        if self.reverse:
            mask = mask.flip(1)
        output = self.helper(mask, input_ids.device)
        if self.reverse:
            output = output.flip(1)

        if self.training and self.max_k > 0:
            k = random.randint(0, self.max_k)
            # already 1-indexed, so values become k+1 upward
            output[output > 0] += k

        return self.embedding(output)


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


def rope_tables_from_index(index: torch.Tensor, head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Same rotation as `rope_tables`, but the angle comes from a per-token,
    per-example index (e.g. `significance_index`) instead of raw sequence
    position, so cos/sin vary across the batch."""
    inverse = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=index.device).float() / head_dim))
    angles = index.float().unsqueeze(-1) * inverse[None, None, :]
    embedded = torch.cat([angles, angles], dim=-1)
    return embedded.cos().unsqueeze(1), embedded.sin().unsqueeze(1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.size(-1) // 2
    rotated = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
    return x * cos + rotated * sin


def rotate_qk(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor,
              sin: torch.Tensor, mode: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to q/k. Keys always rotate by +theta; the mode
    decides the sign on the queries, which is what selects m-n versus m+n."""
    if mode == "rel":
        return apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    if mode == "sum":
        return apply_rope(q, cos, -sin), apply_rope(k, cos, sin)
    if mode == "mixed":
        half = q.size(1) // 2
        return (
            torch.cat(
                [apply_rope(q[:, :half], cos, sin),
                 apply_rope(q[:, half:], cos, -sin)],
                dim=1,
            ),
            apply_rope(k, cos, sin),
        )
    raise ValueError(f"unknown rope mode {mode!r}")


def rotate_qk_dual(
    q: torch.Tensor, k: torch.Tensor,
    rope_abs: tuple[torch.Tensor, torch.Tensor],
    rope_sig: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """First half of the heads get plain rel-RoPE on raw sequence position
    (a globally unique address for every token, including read-out slots --
    the "which output place" signal); second half get rel-RoPE on
    significance (place value, shared across fields -- the "what digit am I"
    signal, length-invariant). Both halves rotate q and k identically (rel,
    not sum), so the two signals stay separable per head rather than
    entangled into one number the way a single significance-only index
    forces query and key to share."""
    cos_a, sin_a = rope_abs
    cos_b, sin_b = rope_sig
    half = q.size(1) // 2
    q_out = torch.cat(
        [apply_rope(q[:, :half], cos_a, sin_a), apply_rope(q[:, half:], cos_b, sin_b)],
        dim=1,
    )
    k_out = torch.cat(
        [apply_rope(k[:, :half], cos_a, sin_a), apply_rope(k[:, half:], cos_b, sin_b)],
        dim=1,
    )
    return q_out, k_out


def sink_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                    keep_mask: torch.Tensor, sink_log: torch.Tensor) -> torch.Tensor:
    """Same result as scaled_dot_product_attention, but every head gets an
    extra virtual key with logit 0 and value 0 in the softmax -- a place to
    dump attention mass when nothing in the real sequence is relevant,
    instead of being forced to spread weight over keys that don't apply.

    sink_log is a per-head log-scalar; exp(sink_log) is what effectively
    gets added to the softmax denominator, so init 0 => contributes exactly
    1 (the fixed "off-by-one softmax" constant), and it's free to move
    (positive by construction) as training finds a head needs more or less
    of a sink.
    """
    scale = 1.0 / math.sqrt(q.size(-1))
    scores = (q @ k.transpose(-2, -1)) * scale
    scores = scores.masked_fill(~keep_mask, float("-inf"))
    row_max = scores.amax(dim=-1, keepdim=True)
    exp_scores = (scores - row_max).exp()
    sink = sink_log.exp().view(1, -1, 1, 1)
    denom = exp_scores.sum(dim=-1, keepdim=True) + sink * (-row_max).exp()
    return (exp_scores / denom) @ v


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, rope_mode: str = "",
                 use_sink: bool = False) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.rope_mode = rope_mode
        self.use_sink = use_sink
        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        if use_sink:
            self.sink_log = nn.Parameter(torch.zeros(n_heads))

    def forward(self, x, keep_mask, rope=None):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)

        def heads(t):
            return t.view(batch, length, self.n_heads, width // self.n_heads).transpose(1, 2)

        q, k, v = heads(q), heads(k), heads(v)
        if self.rope_mode == "dual" and rope is not None:
            rope_abs, rope_sig = rope
            q, k = rotate_qk_dual(q, k, rope_abs, rope_sig)
        elif self.rope_mode and rope is not None:
            cos, sin = rope
            q, k = rotate_qk(q, k, cos, sin, self.rope_mode)
        if self.use_sink:
            attended = sink_attention(q, k, v, keep_mask, self.sink_log)
        else:
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
        abacus_max_k: int = 8,
        digit_range: tuple[int, int] = (0, 0),
        attention_sink: bool = False,
        n_loops: int = 1,
    ) -> None:
        super().__init__()
        if positional not in POSITIONAL_MODES:
            raise ValueError(f"positional must be one of {POSITIONAL_MODES}")
        if d_model % n_heads:
            raise ValueError("d_model must divide evenly into heads")
        self.positional = positional
        self.max_seq_len = max_seq_len
        # [lo, hi) token ids that are digits. PAD sits above this range, so a
        # plain `>= offset` test would wrongly count padding as a digit.
        self.digit_lo, self.digit_hi = digit_range
        self.digit_range = digit_range
        self.head_dim = d_model // n_heads
        self.significance_max_k = abacus_max_k

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        if positional in USES_LEARNED:
            self.position_embedding = nn.Embedding(max_seq_len, d_model)
        elif positional == "sinusoidal":
            self.register_buffer(
                "position_table", sinusoidal_table(max_seq_len, d_model), persistent=False
            )
        if positional.startswith("abacus"):
            self.abacus = Abacus(
                digit_range=digit_range, embedding_dim=d_model,
                max_seq_length=max_seq_len + abacus_max_k + 2, max_k=abacus_max_k,
            )

        self.rope_mode = ROPE_MODE.get(positional, "")
        if self.rope_mode in ("mixed", "dual") and n_heads < 2:
            raise ValueError(f"{positional} needs at least 2 heads to split")
        self.blocks = nn.ModuleList(
            Block(d_model, n_heads, rope_mode=self.rope_mode, use_sink=attention_sink)
            for _ in range(n_layers)
        )
        self.n_loops = n_loops
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                n_loops: int | None = None) -> torch.Tensor:
        length = input_ids.size(1)
        x = self.token_embedding(input_ids)

        if self.positional in USES_LEARNED:
            x = x + self.position_embedding(torch.arange(length, device=input_ids.device))
        elif self.positional == "sinusoidal":
            x = x + self.position_table[:length].to(x.dtype)

        if self.positional.startswith("abacus"):
            # Note: non-digit positions get embedding[0], as in the reference
            # implementation -- they are not masked out.
            x = x + self.abacus(input_ids)

        def sig_table():
            index = significance_index(
                input_ids, self.digit_range,
                training=self.training, max_k=self.significance_max_k,
            )
            cos, sin = rope_tables_from_index(index, self.head_dim)
            return cos.to(x.dtype), sin.to(x.dtype)

        def abs_table():
            cos, sin = rope_tables(length, self.head_dim, input_ids.device)
            return cos.to(x.dtype), sin.to(x.dtype)

        rope = None
        if self.rope_mode == "dual":
            rope = (abs_table(), sig_table())
        elif self.rope_mode and self.positional in USES_SIGNIFICANCE_ROPE:
            rope = sig_table()
        elif self.rope_mode:
            rope = abs_table()

        keep_mask = attention_mask[:, None, None, :]
        for _ in range(n_loops if n_loops is not None else self.n_loops):
            for block in self.blocks:
                x = block(x, keep_mask, rope)
        return self.head(self.ln_f(x))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
