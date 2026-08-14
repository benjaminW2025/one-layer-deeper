# Experiments

Bottom-up decomposition of the competition task. Each experiment is the
simplest thing that could fail; run them in order and stop where it breaks.

```
exp1  addition    x + y                 which positional encoding generalizes?
exp2  square      x -> x*x              can it multiply?
exp3  reduce      y -> y mod N          can it divide?          (3 variants)
exp4  squaremod   x -> x*x mod N        the competition's T=1    (= exp2 ∘ exp3)
```

**Run exp1 first.** It is the cheapest place to learn the positional lesson,
and every later experiment inherits it via `--positional`.

## Run

```bash
cd <repo-root>
/usr/bin/python experiments/exp1_addition.py          # NOT .venv/bin/python
/usr/bin/python experiments/exp2_square.py    --positional <winner>
/usr/bin/python experiments/exp3_reduce.py    --positional <winner>
/usr/bin/python experiments/exp4_squaremod.py --positional <winner>
```

Use `/usr/bin/python` (torch 2.7.0+cu128). The repo's `.venv` has a CUDA 13
build against a 12.8 driver and silently falls back to CPU — `pick_device`
now raises instead of letting that happen.

Results append to `results/exp*.csv`, one row per (config × cohort).

## Where the data comes from

**All of it is generated in memory at run time — no files, no `data/generated/`.**

| what | where |
|---|---|
| These four tasks | [`common/tasks.py`](common/tasks.py) — `build_addition`, `build_square`, `build_reduce`, `build_squaremod` |
| Tokenization | [`common/format.py`](common/format.py), which imports the competition's own `tokenize_squaring_mod_with_result` and `collate_squaring_mod` from [`data/squaring_mod.py`](../data/squaring_mod.py) — not reimplemented |
| Earlier exploration data | [`explorations/harness.py`](../explorations/harness.py) → `build_data()` |
| The real competition datasets | [`data/squaring_mod.py`](../data/squaring_mod.py) CLI, driven by [`scripts/generate_datasets.sh`](../scripts/generate_datasets.sh), writing JSONL to `data/generated/` |

## Format (fixed by the evaluator, not by us)

Vocab is 17 tokens: `PAD BOS N X T ANS EOS` then digits 0-9 at ids 7-16.
Numbers are decimal digits, most-significant first, **variable width**.

```
prompt : [N] digits(a) [X] digits(b) [T] digits(c)
answer : read off the LAST len(answer) positions of the prompt, right-aligned
```

The answer is never appended — it is emitted on top of prompt positions. The
final position is always the ones digit, the one before it the tens digit, and
so on, so the model can just lay out the zero-padded answer and let the
evaluator take whatever suffix it needs. Digits are decoded by independent
per-position argmax: **no autoregression between answer digits**, so every
carry has to be resolved inside the forward pass.

Each task reuses the three fields differently:

| task | a | b | c | answer |
|---|---|---|---|---|
| addition | x | y | 0 | x + y |
| square | 0 | x | 1 | x·x |
| reduce | N | y | 1 | y mod N |
| squaremod | N | x | 1 | x·x mod N |

## Positional schemes (`--positional`)

| mode | what it encodes |
|---|---|
| `none` | nothing. Bidirectional ⇒ permutation-invariant ⇒ should fail. Floor. |
| `learned` | absolute position. **What the competition baseline uses.** |
| `sinusoidal` | fixed absolute position |
| `rope` | relative position, applied to q/k |
| `abacus` | each digit's index within its own number, counted from the right — i.e. its **place value**. No absolute position. Random per-sequence offset during training. |
| `abacus_rope` | place value + relative field order |

The motivation: numbers are variable-width, so a digit's place value is its
distance from the *end of its own field*, and every field boundary shifts when
`N` has a different digit count. Absolute position cannot express that
directly, which is the leading suspect for why the baseline fails.

## Reading the results

- **Always compare against `chance_exact`** in the same row. Answer
  distributions are skewed; chance is not 10^-k.
- `per_place_acc` is accuracy by place value, index 0 = ones digit. Carries
  propagate right-to-left, so a right-to-left decay is the signature of a
  carry failure.
- `exact` vs `digit`: exact match needs every digit. A model can be at 0.95
  per-digit and ~0 exact.
- exp4's `train_seen` vs `test_fresh` gap is the memorization measurement, and
  `x_coverage` says whether memorization was even available.
- `flat` / `diverged` flag runs where the LR, not the architecture, is the
  story. One untuned LR (3e-4) is used everywhere by design.

## Knobs worth knowing

- `--loss-reduction token|example`. `token` is the evaluator default and
  weights long answers more; `example` matches exact-match scoring. Free A/B.
- `--steps`, `--d-model`, `--n-layers`, `--n-train`, `--seeds`.
- exp4 `--bits` controls modulus size, which controls `x_coverage` — the only
  clean way to make memorization impossible without changing the arithmetic.
