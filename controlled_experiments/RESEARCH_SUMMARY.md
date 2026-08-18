# Controlled arithmetic research summary

## Goal

We are using direct decimal squaring as a controlled proxy for the main
semiprime modular-squaring/composition challenge. The question is not merely
whether a model can fit examples, but whether an architecture can learn a
reusable arithmetic computation that transfers to unseen values, longer
numbers, and eventually repeated composition.

The direct-squaring prompts expose explicit output slots. Targets are emitted
least-significant digit first, so the first output slot always represents the
ones digit. Easy training uses three- and four-digit inputs; `ood_long` uses
five-digit inputs.

## Main result so far

An ordinary full-token Transformer can learn direct squaring substantially
better than the scratchpad variants. The most promising recurrent compromise
is **four distinct Transformer blocks repeated twice**. It preserves some
parameter sharing while giving the model enough separate stages to learn the
in-distribution computation.

All figures below are a single seed, 2,000 AdamW updates, width 256, four
attention heads, batch size 256, and learning rate 2e-3. OOD exact accuracy is
zero in every listed direct-squaring run, so OOD per-digit accuracy is the
useful early signal.

| Model | Train exact | Held-out exact | OOD-long digit |
| --- | ---: | ---: | ---: |
| Scratchpad: 2 blocks x 4 repeats | 0.01% | 0.00% | 20.7% |
| Two-tape product/accumulator | 0.00% | 0.00% | 13.3% |
| Full Transformer: 8 distinct blocks | 42.6% | 8.6% | 29.9% |
| Full-token recurrence: 2 blocks x 4 repeats | 4.9% | 2.2% | 29.4% |
| 2 blocks x 4 repeats + round embedding | 8.9% | 3.5% | 33.3% |
| **4 distinct blocks x 2 repeats** | **46.0%** | 7.7% | **32.5%** |
| 4 distinct blocks x 4 repeats | **52.0%** | 5.6% | 14.2% |
| Frozen-context reinjection: 4 blocks x 2 repeats | 30.6% | **9.9%** | 21.1% |
| Variable 2-4-repeat reinjection, evaluated at 3 | 13.7% | 5.8% | 27.6% |

## What these experiments say

### 1. Specialization is important

The 8-layer ordinary Transformer and the 4-block x 2-repeat model both apply
eight Transformer blocks in total. Their difference is that the former has
eight independent parameter sets, while the latter has four parameter sets
used twice.

The 4x2 model nearly matches the 8-layer model's in-distribution fitting and
has stronger OOD per-digit accuracy. In contrast, 2x4 fails to fit well.
Therefore, the main failure of the original tied recurrent model is not simply
insufficient effective depth: it lacks enough distinct computational stages.

### 2. More recurrence currently acts more like depth than an algorithm

Changing 4x2 to 4x4 increases training exact accuracy from 46.0% to 52.0%,
but reduces OOD-long digit accuracy from 32.5% to 14.2%. The extra repeated
passes are helping training fit while harming length transfer.

Variable-depth training also produced a stable but weak update rule. When
trained with 2-4 repeats and evaluated at 1, 2, 3, 4, 6, and 8 repeats,
performance improved sharply from one to two/three repeats, then gradually
declined. OOD per-digit accuracy was relatively flat after two repeats
(27.6% at two, 26.1% at eight), but did not improve with extra recurrence.

This is not yet a learned iterative algorithm that keeps improving when
unrolled. It is a mostly stable fixed-depth computation.

### 3. Scratchpads and the first two-tape pipeline are not the unlock

The compact scratchpad model does not fit direct squaring, and the attempted
product-tape -> accumulator-tape structure collapses to near-baseline digit
predictions. Merely naming different writable states does not make the model
discover an arithmetic plan from final-answer supervision alone.

The productive reference family is currently full-token computation, not a
small isolated scratchpad.

### 4. Stable context access helps in-range answers, but not length OOD

Context reinjection freezes a prompt representation `C`, maintains a mutable
workspace `S`, and updates `S` with queries from `S` and keys/values from
`[C, S]`.

For four blocks repeated twice, this gives the best held-out exact score
(9.9%) but worse OOD-long digit accuracy (21.1%). The current implementation
uses the same positional coordinates for `C` and `S` with no source/type
marker. A clean next ablation is:

```text
C[p] = token + position(p) + context-type embedding
S[p] = token + position(p) + workspace-type embedding
```

The positions should remain aligned; the new signal should identify immutable
context versus mutable workspace.

### 5. Round identity helps, but is not enough

Adding learned round embeddings to the weak 2x4 recurrent model improves
train exactness, held-out exactness, and OOD per-digit accuracy. This supports
the idea that reused weights benefit from knowing their current computational
phase. A stronger future version is a small per-repeat adapter or modulation,
not only an added vector.

## Optimization profile findings

We built an every-step profiler for the 4-block x 2-repeat model. It logs raw
gradients, Adam moments, parameter updates, module activations, norm scales,
and fixed-probe recurrence metrics.

The current trace suggests a forward role split:

```text
repeat 1, block 0: large initial feature rewrite
repeat 2, block 3: large late refinement/readout change
```

Across the final 500 steps, mean residual-change RMS was:

```text
repeat 1: block 0 6.64, block 1 2.45, block 2 1.98, block 3 3.08
repeat 2: block 0 4.20, block 1 2.90, block 2 3.38, block 3 6.20
```

Block 0 receives the largest average QKV gradient signal, but the other
blocks remain active. Therefore, it would be inaccurate to say the first
round performs all useful computation.

Gradient comparisons between answers read after one versus two repeats are
usually near zero, with occasional negative values. This indicates that the
two uses of shared weights are often being asked to serve different roles.
It is suggestive of optimization pressure from tying, but it is not yet proof
of catastrophic gradient conflict.

The profile's mini-batch loss was still decreasing late in training. This
makes additional training steps a plausible lever for improving *fit*.
However, the existing train/held-out gap means lower training loss alone is
not evidence of improved arithmetic generalization.

## Current interpretation

There is probably both an optimization problem and an architectural problem:

1. **Optimization:** recurrence ties gradients from distinct stages together;
   current training is only 2,000 steps and remains sensitive to seed/data
   order.
2. **Architecture:** the models still do not generalize reliably to unseen
   same-length inputs, and fail completely on exact longer-number outputs.
   Better positional/source structure and stable iterative refinement are
   likely needed.

It is too early to claim that one missing architectural idea is the sole
bottleneck. More training may substantially improve fit, but the 4x4 result
shows that more computation can also worsen OOD behavior.

## Next experiments, in priority order

1. **Training-duration curve:** fixed 4x2 at 2k, 4k, and 8k updates, tracking
   train, held-out, and OOD metrics. This distinguishes undertraining from
   overfitting/representation limits.
2. **Rerun complete profiles with multiple seeds:** include per-repeat gradient
   attribution and profile the 8-distinct-layer reference as a control.
3. **Per-use evaluation ablation:** skip each block application separately,
   e.g. block 0 on repeat 1 or block 3 on repeat 2, to measure causal rather
   than correlational importance.
4. **Context/workspace type embeddings:** fixed-context 4x2 with aligned
   positions and explicit source identity.
5. **Optimization sweep after profiling:** learning rate, warmup/decay,
   residual/update scaling, and potentially Muon on matrix weights.
6. **Recurrence-specific flexibility:** small per-repeat modulation or
   low-rank adapters on top of a shared four-block core.

## Important caveats

- Most comparisons are one seed and should be treated as directional.
- The full optimization profiler is diagnostic and much slower than normal
  training; it should not be used for time-budget comparisons.
- The first full trace altered the shuffled training order while constructing
  its fixed probe. This has since been fixed; use the revised profiler for
  performance comparisons.
- The controlled multiplication task is materially harder than direct
  squaring under this output interface. Direct squaring is currently the more
  informative diagnostic for the main challenge.
