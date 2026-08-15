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
| Tokenization | [`common/tokenizer.py`](common/tokenizer.py) — one `DigitTokenizer` per task, same strategy as the competition |
| Earlier exploration data | [`explorations/harness.py`](../explorations/harness.py) → `build_data()` |
| The real competition datasets | [`data/squaring_mod.py`](../data/squaring_mod.py) CLI, driven by [`scripts/generate_datasets.sh`](../scripts/generate_datasets.sh), writing JSONL to `data/generated/` |

## Format

Same *strategy* as the competition, sized per task. Each task has its own
tokenizer in [`common/tokenizer.py`](common/tokenizer.py):

| task | prompt | vocab | answer |
|---|---|---|---|
| addition | `[X] d(x) [Y] d(y)` | 13 | x + y |
| square | `[X] d(x) [Y] d(x)` | 13 | x·x |
| reduce | `[N] d(N) [Y] d(y)` | 13 | y mod N |
| squaremod | `[N] d(N) [X] d(x)` | 13 | x·x mod N |

Ids: markers `0..m-1`, digits 0-9 at `m..m+9`, PAD at `m+10`. Numbers are
decimal digits, most-significant first, **variable width** — never zero-padded
to a fixed size.

The answer is **not appended**. It is read off the last `len(answer)`
positions of the prompt, right-aligned, exactly as the competition does: the
final position is always the ones digit, the one before it the tens digit. So
the model can lay out the zero-padded answer and let the evaluator take
whatever suffix it needs — it never has to predict the answer's length.

Digits are decoded by **independent per-position argmax**. There is no
autoregression between answer digits, so every carry must be resolved inside
the forward pass. Loss is cross-entropy over those answer positions only;
every other position gets zero gradient.

Two deliberate deviations, both forced:

- **Square repeats x in both operand slots.** The format requires
  `answer_len ≤ prompt_len`, and a 4-digit x squared is 8 digits against a
  5-token `[X] d(x)` prompt. Writing x·x as two operands fixes that and makes
  square structurally identical to addition, so exp1 and exp2 differ only in
  the operator.
- **`reduce` draws y from [0, N²)**, which is exactly the range squaring
  produces. If y < N then y mod N = y and the task is a copy.

## Positional schemes (`--positional`)

| mode | what it encodes |
|---|---|
| `none` | nothing. Bidirectional ⇒ permutation-invariant ⇒ should fail. Floor. |
| `learned` | absolute position. **What the competition baseline uses.** |
| `sinusoidal` | fixed absolute position |
| `rope` | relative position, applied to q/k |
| `abacus` | each digit's index within its own number, counted from the right — i.e. its **place value**. No absolute position. Random per-sequence offset during training. |
| `abacus_learned` | place value **+** absolute. The combination used by [McLeish et al., NeurIPS 2024](https://arxiv.org/abs/2405.17399), whose best results add Abacus alongside a standard positional embedding rather than replacing it. |
| `abacus_rope` | place value + relative field order |

The motivation: numbers are variable-width, so a digit's place value is its
distance from the *end of its own field*, and every field boundary shifts when
`N` has a different digit count. Absolute position cannot express that
directly, which is the leading suspect for why the baseline fails.

Abacus is from ["Transformers Can Do Arithmetic with the Right
Embeddings"](https://arxiv.org/abs/2405.17399) (McLeish et al., NeurIPS 2024;
code at [mcleish7/arithmetic](https://github.com/mcleish7/arithmetic)), which
trains on <=20-digit operands and generalizes to 120 digits. Note that repo's
handle matches the author of this competition's scoring commits.

Our indexing direction differs from the paper by necessity: they encode a
digit's position relative to the START of its number, which equals place value
because their inputs are least-significant-digit first. Ours are MSD-first, so
we index from the RIGHT end of each digit run — preserving the property that
matters (digits of equal significance share an embedding) rather than the
literal rule.

Not implemented, and the paper's next-biggest win: **input injection** (skip
connections from the input layer into every block), worth another ~50% error
reduction on top of Abacus.

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
