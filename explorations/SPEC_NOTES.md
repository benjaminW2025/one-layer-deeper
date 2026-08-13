# Spec verification against the repo generator

Checked against `data/squaring_mod.py`, `data/counting.py`, `scripts/generate_datasets.sh`,
`service/tiers.py`, `benchmark/runner.py`, and the README. Corrections to the working
summary, most important first.

## What the prompt summary got wrong (or under-specified)

1. **`x` is always coprime to `N`.** The generator samples units only
   (`_sample_unit` rejects `gcd(x, N) != 1`). The `gcd ≠ 1` / CRT case in the
   experiment-0 plan **never occurs in competition data**. Exp0 still reports the
   would-be rate for uniform `x` as context, but no model handling is needed.

2. **No fixed-width digit slots, and the output is not appended after the prompt.**
   Numbers are tokenized as *variable-length* decimal digit strings,
   most-significant digit first (`number_tokens` = `str(value)` per char). In the
   `separate_input_output` format used by every tier, the target is the digit
   string of the result, **tail-aligned onto the last `len(result)` positions of
   the prompt sequence itself** (`target_positions = arange(input_len - target_len,
   input_len)` in `collate_squaring_mod`). The model reads the whole prompt
   bidirectionally and must emit answer digits at those tail positions in one
   forward pass — there are no autoregressive answer slots at all. Exact match =
   every one of the (variable number of) answer digits correct.

3. **Prompt token order is `N, x, T`, not `(x, N, T)`:**
   `[N] d(N)… [X] d(x)… [T] d(T)…` with marker tokens N=2, X=3, T=4; digits are
   tokens 7–16 (`DIGIT_OFFSET=7`); vocab size 17. `T` is itself a decimal digit
   string (T=16 is two tokens).

4. **Train and eval `T` are NOT disjoint in the real competition.** The scored
   depth ladder is `T = 1,2,4,8,16,32,64` (hard-coded `DEPTH_LADDER`,
   `service/views.py:122`) and it *includes* trained depths — e.g. M1 trains on
   T ∈ {4,8,16} and is certified on the full ladder from T=1 up. Ranking is the
   largest *consecutively certified* prefix (100% exact per rung). What is
   disjoint: (a) depth-eval `(N, x)` prompt pairs are excluded from training,
   (b) the OOD-N profile uses modulus identities/bit-sizes never trained on.
   Our harness keeps the requested strict OOD-T assertion as an extra
   diagnostic (`ood_t` flag on ladder rows with T ∉ train_T), but evaluates the
   full ladder and labels rows, mirroring the competition.

5. **"Evaluation is on OOD N and T" is only exactly true for the ranked Hard
   profiles.** Easy/Medium *score* mean exact accuracy on an in-distribution
   test split; the depth ladder (`Max T` on seen moduli with fresh x, and
   `OOD N Max T` on unseen moduli) is diagnostic there and ranking-relevant on
   Hard only.

6. **Actual ranges used by the public tiers** (Hard's dataset is hidden — no
   manifest in the repo):
   - Easy: fixed N=323 (17·19, T train {1,2,3}), fixed N=899 (29·31, {1,2,4}),
     sampled 10–12-bit N at T=2 or T∈{1,2,3}. OOD-N depth cohorts at ~1–2 bits
     above the training sizes. 60 s budget.
   - Medium: fixed N=10403 (101·103), N=38021 (193·197) at T∈{4,8,16}; sampled
     N up to 22 bits at T ∈ {2,…,16}. OOD-N cohorts up to 24 bits. 600 s.
   - Moduli are semiprimes p·q with p, q primes of ~half the bit width; labels
     are computed via the φ(N) trapdoor (`trapdoor_squaring_mod`), so deep-T
     labels are exact and cheap for the generator.

7. **Bidirectional attention confirmed** — evaluator supplies a padding mask
   only (README, "Compute tiers"), so encoder-style is right. Manifests also
   set bf16 + AMP and `compile: false` for the evaluator's own runtime
   (participants may still `torch.compile` inside `build_model`; rule 11 counts
   it against the budget either way).

## What the summary got right

- `f_N(x) = x² mod N` composed T times = `x^(2^T) mod N`; one dataset kind in
  the whole repo, enforced in `data/config.py`.
- Decimal digit targets, exact-match scoring, per-digit vs exact divergence.
- 500M trainable-parameter/state ceiling; 60/600/3600 s H100 wall-clock; the
  evaluator owns the outer loop; depth/recurrence/halting explicitly legal.
- The periodicity-vs-time-lock dichotomy is real and is exactly the seen-N vs
  unseen-N distinction: the trapdoor/cycle structure is learnable per modulus,
  and the OOD-N profile is what forces a uniform algorithm.

## Environment caveats for these experiments (local, not competition)

- Local GPUs are **NVIDIA L40s, not H100s** (and GPU 0 was busy at ~80%
  utilization when checked; scripts auto-pick the freest device). Wall-clock
  numbers here calibrate *relative* costs only. For exp2's roofline math the
  relevant L40 numbers are ~90.5 TFLOP/s dense bf16 and ~864 GB/s HBM →
  ridge ≈ 105 FLOP/byte → `d_model ≈ 210` breakeven, versus H100's ≈ 295 →
  `d_model ≈ 590`. Both are CLI-configurable in `exp2_profile.py`; conclusions
  about *where the H100 breakeven sits* must be re-derived on an H100.
