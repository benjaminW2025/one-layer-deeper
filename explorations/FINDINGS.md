# One Layer Deeper — exploration findings

All data collected 2026-08-13. Hardware: NVIDIA L40 (not H100 — see Caveats).
Raw data: `results/results.csv`, `results/exp0_task_analysis.json`,
`results/exp2_profile.json`, `results/plots/`.

---

## Terms

| term | meaning |
|---|---|
| **T** | how many times `x -> x² mod N` is applied. The composition depth |
| **N** | the modulus, always a semiprime `p·q`. "Seen N" = a modulus used in training; "OOD N" = one never trained on |
| **exact accuracy** | every decimal digit of the answer correct. This is what the competition scores |
| **per-digit accuracy** | fraction of individual digits correct. Always higher than exact; not what is scored |
| **chance** | the best a constant predictor could do (always guess the most common answer / digit). Not 10% — the answer distribution is skewed, so chance is sometimes as high as 9% exact |
| **period P** | how many steps of T before the answer starts repeating |
| **pre-period s** | how many steps before that repetition begins |
| **steps budget** | fixed number of optimizer steps. Measures *can the model represent this* |
| **seconds budget** | fixed wall-clock. Measures *is it worth the cost* — the competition's real constraint |

---

## Headline

A standard 4-layer, d=256 transformer **cannot do a single modular squaring**
on a fresh input, let alone compose several. It memorizes its training set
perfectly and generalizes at or below chance — including in-distribution.
143 of 150 evaluation rows are at or below chance.

The task itself requires genuine step-by-step computation: the periodicity
shortcut that would let a model skip ahead is not learnable from the training
distribution.

---

## How the data is tokenized

Fixed by the evaluator — we import the repo's own tokenizer rather than
reimplementing it, so this is exact, not an approximation.

**Vocabulary is 17 tokens.** Seven markers (`PAD`=0, `BOS`=1, `N`=2, `X`=3,
`T`=4, `ANS`=5, `EOS`=6) and ten digits (`0`–`9` → ids 7–16).

**Every number is written as decimal digits, most significant first, with no
padding and no fixed width.** `899` → three tokens. `2` → one token. There is
no separate symbol per value — the model sees arithmetic the way it is written
down, digit by digit.

**A prompt is `[N] digits(N) [X] digits(x) [T] digits(T)`.** Worked example,
N=899, x=123, T=2, answer 342:

```
prompt  : N  d8 d9 d9  X  d1 d2 d3  T  d2      (10 tokens)
ids     : 2  15 16 16  3   8  9 10  4   9
target  : d3 d4 d2                             (3 tokens: "342")
```

**The answer is not appended.** In the `separate_input_output` format every
tier uses, the answer digits are read off the **last `len(answer)` positions of
the prompt itself** — here positions 7, 8, 9. Those positions physically hold
the last digit of `x`, the `T` marker, and the digit of `T`. The model attends
bidirectionally over the whole prompt (padding mask only, no causal mask) and
emits all answer digits in one forward pass. There are no autoregressive answer
slots, so there is nowhere to put intermediate work — every step of the
computation must happen inside the forward pass.

**Two consequences worth knowing:**

- **The answer is right-aligned to the end of the sequence, and length never
  has to be predicted.** The model is called as
  `model(input_ids, attention_mask=...)` and must return logits at *every*
  position; it never receives `target_positions`. The evaluator keeps that
  tensor and gathers the last `len(answer)` positions itself. Because the
  answer is right-aligned, the readout is consistent across examples — the
  final position always carries the ones digit, the one before it the tens
  digit, and so on — so positions beyond a short answer are simply never
  scored. Length is neither an input nor something the model must output.
- **Digit tokenization implies a smoothness that the task does not have.**
  `x=123` and `x=124` share two of three tokens but have completely unrelated
  answers mod N. Nearby inputs are not nearby outputs.

---

## Experiment 0 — what kind of problem is this? (CPU, no model)

Applying `x -> x² mod N` T times equals `x^(2^T) mod N`. That sequence
eventually repeats as T grows. If a model learned the repeat pattern, it could
answer any T instantly without doing the work. **Does that shortcut exist?**

| regime | median period P | cycle visible within T≤64 | **cycle visible inside trained T** |
|---|---|---|---|
| fixed N=323 | 6 | 100% | **5.6%** |
| fixed N=899 | 12 | 100% | 7.1% |
| sampled 10-bit | 10 | 100% | 4.1% |
| fixed N=10403 | 40 | 100% | 19.8% |
| fixed N=38021 | 42 | 100% | 14.1% |
| sampled 16-bit | 84 | 42.7% | 1.7% |
| sampled 22-bit | 1212 | 7.1% | 0.2% |
| OOD-N 24-bit | 2948 | 3.3% | **0.0%** |

**Verdict: the shortcut is unavailable, for two different reasons.**

- At **small N**, the cycle is short enough to appear on the scored ladder
  (100% of cases), but the model almost never sees a repeat *during training*
  (4–20%). It would have to invent periodicity it was never shown.
- At **large N**, the cycle simply isn't there within the tested range —
  median period 2948 versus a maximum tested T of 64.

So depth is not a trap. Sequential computation is genuinely required, which
is consistent with the competition explicitly permitting recurrence,
iterative refinement, and adaptive halting.

Two further facts from this experiment:

- **`gcd(x, N) = 1` always.** The generator only emits units, so the
  non-coprime edge case never occurs and needs no handling.
- **Chance is high and uneven.** Best-constant-predictor exact accuracy ranges
  from 0.02% to 8.5% depending on regime. Any accuracy number without its
  chance baseline beside it is meaningless.

---

## Experiment 1 — does a baseline transformer learn it?

4 layers, d=256, 3.17M parameters, bidirectional, bf16.

| run | steps | train loss | id_test exact | chance | verdict |
|---|---|---|---|---|---|
| fixed N=323, 1500 steps | 1,500 | 0.0002 | 0.020 | 0.090 | below chance |
| fixed N=323, 60 s | 7,079 | 0.00001 | 0.030 | 0.090 | below chance |
| sampled 10–11 bit, 1500 steps | 1,500 | 1.989 | 0.006 | 0.008 | at chance |
| sampled 10–11 bit, **60 s** | **7,518** | **0.061** | **0.004** | 0.008 | **fit training, still at chance** |

Accuracy by composition depth, for the best run (sampled N, 60 s):

| T | 1 | 2 | 3 | 4 | 6 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|---|---|
| seen N | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 0.01 | 0.00 | 0.00 |
| OOD N | 0.00 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

**Findings:**

1. **The failure is not a compute shortage.** Going from 1,500 to 7,518 steps
   drove training loss from 1.99 to 0.061 — a complete fit of the 20,000
   training examples — and moved held-out accuracy from 0.006 to 0.004. More
   compute buys memorization, not understanding.

2. **It fails in-distribution, not just out.** `id_test` uses the *same*
   moduli and *same* T values as training, differing only in `x`. Accuracy
   there is 0.004 against chance 0.008. The OOD splits aren't what break it.

3. **It cannot do even one squaring on an x it hasn't seen.** At **T=1** — a
   single `x² mod N`, no composition at all — accuracy on the fresh-x cohort
   is 0.01. On memorized `(N, x)` pairs it is essentially perfect, which is
   what the training loss reports. Composition depth is not currently the
   bottleneck; generalizing the base operation over x is.

4. **Below chance means confident memorization.** A constant predictor beats
   the model, because the model outputs memorized answers for inputs it has
   never seen.

5. **Noise floor: ±0.0015.** Three seeds gave 0.006 / 0.003 / 0.004. Any
   difference under ~0.005 in these tables is not real.

**Why it memorizes:** 3.17M parameters against 20,000 training examples is
159 parameters per example. The lookup table is cheaper to learn than the
algorithm.

---

## Experiment 2 — where does the compute go?

Per-step timings, 4 layers, batch 512, L40.

| | d_model=128 | d_model=640 |
|---|---|---|
| forward | 4.32 ms | 5.44 ms |
| backward | 5.54 ms | 7.83 ms |
| optimizer | 0.93 ms | 2.60 ms |
| backward/forward ratio | 1.28 | 1.44 |
| achieved throughput | 2.47 TFLOP/s | 48.3 TFLOP/s |
| **% of hardware peak** | **2.7%** | **53.4%** |
| CPU dispatch vs GPU work per step | 18.6 ms vs 1.9 ms | 23.6 ms vs 11.8 ms |

**Findings:**

1. **The roofline prediction held exactly.** A model is bandwidth-bound when
   its width is below the hardware's "ridge point" (L40: ~105 FLOP/byte, so
   d≈210). Predicted bandwidth-bound at d=128 → measured 2.7% of peak.
   Predicted compute-bound at d=640 → measured 53.4%. Both correct.

2. **Narrow models waste the GPU.** d=640 does **25× the arithmetic of d=128
   for 1.5× the wall-clock**. At d=128 the GPU sits idle ~90% of each step,
   waiting on CPU-side kernel dispatch.

3. **The optimizer is a large share at small widths** — 45% of GPU time at
   d=128. `fused=True` would help.

4. Backward/forward ratios of 1.28–1.44 (theory says ~2.0 when compute-bound)
   independently confirm neither width is purely compute-bound.

*Not a real cost:* the profiler shows a large `data_h2d` phase. That is an
artifact of the profiler deliberately keeping data on the CPU to measure the
transfer path. The actual training loop holds the dataset on the GPU and never
pays it.

---

## Problem inventory

Everything found so far, with its cause. The model must generalize along three
independent axes — **x** (new starting value), **N** (new modulus), **T** (new
composition depth) — and these are **nested, not parallel**: OOD-N is only
testable once OOD-x works, since a new modulus also brings new x values, and
OOD-T is only meaningful once the base operation works at all. Right now all
three fail simultaneously, so none of the three failures can be attributed.

### Generalization failures

| # | Problem | Evidence | Cause |
|---|---|---|---|
| G1 | **Fails on new x** at fixed N, fixed T | id_test 0.004 vs chance 0.008 | Memorization is near-optimal for the training objective (see D1, D2). Eval draws from the *complement* of trained prompts, so a lookup table scores ~0 by construction |
| G2 | **Fails on new N** | OOD-N rungs all 0.00 | Per-modulus tables carry no information about an unseen modulus. Transferring requires learning modular arithmetic as an algorithm parameterized by N — nothing in the training signal rewards that over tables |
| G3 | **Fails on new T** | ladder flat at 0.00–0.01 through T=64 | No shortcut exists (see T1), so deep T requires genuine sequential computation. Untestable anyway while T=1 fails |
| G4 | **Cannot do T=1 on a fresh x** — a single squaring | 0.01 on the fresh-x T=1 cohort; near-perfect on memorized (N,x) | It has learned 27 lookup tables, not the operation `x² mod N`. Generalizing T=1 over x *is* multi-digit modular multiplication — a known-hard target for transformers. **This is the foundation question and blocks G2/G3** |
| G5 | **Scores *below* chance** | 0.004 vs 0.008; 0.030 vs 0.090 | Confident interpolation between memorized points. The constant predictor picks the modal answer; the model picks a wrong specific one |

### Why memorization wins (data/capacity)

| # | Problem | Evidence | Cause |
|---|---|---|---|
| D1 | Training set is a large fraction of the whole task | 20k prompts = **21.6%** of all 92,544 possible (N,x,T); up to 51% for the smallest moduli | At 10–11 bits only **27** balanced semiprimes exist. The universe is small |
| D2 | Model capacity dwarfs the data | 3.17M params / 20k examples = **159 params per example** | Table lookup is cheaper to represent than the algorithm |
| D3 | More compute deepens memorization, not understanding | 1,500→7,518 steps: train loss 1.99→0.061, eval 0.006→0.004 | Nothing in the objective distinguishes the two solutions; SGD takes the cheaper one |
| D4 | Cannot be fixed with more data | — | Rule 14 forbids augmentation; the evaluator supplies a fixed dataset. The lever must be removing the *ability* to memorize, not adding pressure |

### Task-structural obstacles (inherent, not fixable)

| # | Problem | Evidence | Cause |
|---|---|---|---|
| T1 | The periodicity shortcut is unusable | cycle visible inside trained T for only 0–20% of pairs | At small N the cycle exists but is never *observed* in training; at large N it exceeds the tested range entirely |
| T2 | No local structure to interpolate | below-chance scores (G5) | `x → x² mod N` is pseudorandom w.r.t. any smooth metric. Adjacent inputs have unrelated outputs — hostile to every smooth function approximator |
| T3 | Digit tokenization suggests false smoothness | — | `x=123`/`x=124` share tokens but not answers. The representation actively misleads |
| T4 | Exact match is all-or-nothing | per-digit 0.17 vs exact 0.004 | Scoring requires every digit. Per-digit accuracy looks far healthier and is not what is scored |
| T5 | Chance is high and uneven | up to 8.5% exact | The squaring map's image shrinks as T grows, concentrating outputs. Any accuracy reported without its baseline is misleading |

### Experimental-setup problems (ours, several already fixed)

| # | Problem | Status | Cause |
|---|---|---|---|
| S1 | Easy sampled regime too small to test OOD-N | open | Only 27 moduli exist at 10–11 bits. Need 13–14 bit (99 moduli) for range |
| S2 | `e1_fixed323` is structurally a memorization benchmark | open, by design | Fixed tiny N has just 288 units total; faithful to real E1, but it can never show generalization |
| S3 | Silent CPU fallback invalidated a full run | **fixed** | venv torch built for CUDA 13 vs 12.8 driver; harness didn't assert device. Now hard-fails and logs device per row |
| S4 | Wall-clock results are L40, not H100 | open | Hardware availability. Steps-mode results transfer; seconds-mode do not |
| S5 | Depth × width surface unmeasured | open | Not yet run — and currently uninterpretable while G4 holds |

### Compute efficiency (real, but not the bottleneck)

| # | Problem | Evidence | Cause |
|---|---|---|---|
| C1 | Narrow models waste the GPU | 2.7% of peak at d=128 | Below the roofline ridge point (L40 d≈210); GPU idles ~90% of each step on CPU dispatch |
| C2 | Optimizer dominates at small width | 45% of GPU time at d=128 | Many small tensors, non-fused AdamW |

---

## Caveats

- **Hardware is L40, not H100.** Relative costs and the roofline *method*
  transfer; absolute timings and the ridge point do not. On H100 the
  bandwidth/compute crossover sits near d≈590, not d≈210.
- **One earlier run was invalid.** The repo's `.venv` has a CUDA 13 torch build
  and the driver is CUDA 12.8, so it silently ran on CPU (~200× slower). Those
  rows are quarantined in `results/results_CPU_INVALID.csv`. Use
  `/usr/bin/python`. The harness now hard-fails instead of falling back, and
  records the device in every row.
- **The scored T ladder overlaps training T.** The real competition does not
  hold T out; it certifies a consecutive prefix from T=1 upward on fresh
  prompts and unseen moduli. Our splits mirror that and flag T-extrapolation
  separately.
- **Untuned LR (3e-4) throughout, by design.** No run diverged or stalled from
  step one, so no result here is explained by a broken learning rate.
- **Experiment 3 (depth × width) has not been run.**

---

## Open questions

1. Can *any* configuration learn a single modular squaring (T=1) on fresh
   inputs? Nothing else matters until this is yes.
2. Does reducing capacity (fewer parameters) force generalization instead of
   memorization, given the dataset size is fixed?
3. Does weight-tied recurrence — same parameters applied k times — beat
   stacked layers at equal parameter count, on both generalization and
   reachable T?
4. Can a recurrent model iterate *more* times at test than during training,
   which is what OOD-T certification demands?
5. Does the model learn per-modulus lookup tables rather than modular
   arithmetic? Testable by holding total rows fixed while varying the number
   of distinct moduli.
