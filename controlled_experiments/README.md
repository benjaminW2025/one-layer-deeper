# Controlled arithmetic experiments

This directory is for local diagnostic experiments. Its purpose is to identify
which part of modular composition fails before changing the competition model.

Keep these experiments separate from submissions: do not copy trained weights,
generated labels, or hard-coded arithmetic procedures into a submission.

## Experiment ladder

Run the same model family through these tasks in order:

1. `multiply`: decimal `a, b -> a * b`.
2. `reduce`: decimal `z, N -> z mod N`, with `z` as large as a product.
3. `square_mod`: decimal `x, N -> x^2 mod N`.
4. `repeat_square_mod`: decimal `x, N, T -> x^(2^T) mod N`.

Every task should have an in-range split and a length-generalization split.
For example, train on 2--3 digit values and test on 4--5 digit values.

## The key comparison

For modular reduction, compare these three conditions:

```text
A. Reducer receives the true product as decimal tokens.
B. A recurrent scratchpad creates a product representation, then the reducer
   receives that representation.
C. One model receives only the final x^2 mod N target.
```

Interpret the results as follows:

| Result | Likely bottleneck |
| --- | --- |
| A fails | Learning modular reduction itself. |
| A works but B fails | The hidden scratchpad is not a usable number representation. |
| B works but C fails | Final-answer training does not discover the two-stage plan. |
| One square works but repeated squaring fails | Recurrence or error accumulation. |

## What to record

For every run, record:

- exact sequence accuracy;
- per-digit accuracy;
- accuracy by input digit length;
- accuracy by carry count for multiplication;
- accuracy by quotient size for modular reduction;
- train versus held-out-length accuracy;
- model width, scratchpad slots, recurrence rounds, batch size, learning rate,
  and training steps.

Store machine-generated run outputs under `results/`, named for the task and
configuration, for example `reduce_scratch4_len3to5.jsonl`.

## First implementation target

The multiplication generator is `generate_multiplication.py`. Create the two
presets with:

```bash
python3 controlled_experiments/generate_multiplication.py --preset easy
python3 controlled_experiments/generate_multiplication.py --preset medium
```

Each preset produces `train.jsonl`, held-out in-range `test.jsonl`,
`ood_one_long.jsonl` (one factor is longer), and `ood_both_long.jsonl` (both
factors are longer). Add reduction only after the multiplication baseline and
its length-generalization measurements are working.

Run the current scratchpad architecture locally with, for example:

```bash
python3 controlled_experiments/run_multiplication.py --preset easy --steps 2000
```

## Baseline protocol

The first multiplication baseline is the current compact scratchpad model:

```text
width:                    256
scratchpad slots:         4
different shared layers:  2
recurrent rounds:         4
scratchpad updates:       8
prompt reader:            none (raw fixed prompt embeddings)
answer reader:            one layer
optimizer:                AdamW, lr 2e-3
batch size:               256
training budget:          2,000 updates
compilation:              off
seed:                     74
```

Evaluate exact and per-digit accuracy on `test`, `ood_one_long`, and
`ood_both_long`. Record wall-clock time, but do not use it to choose the first
architecture winner: all initial comparisons use the same update count.

## First architecture test: prompt reader

**Question:** Does the scratchpad fail because it repeatedly reads raw digits
that do not yet know their field or decimal-place role?

Add exactly one bidirectional self-attention/MLP layer over the fixed prompt
before initializing recurrence:

```text
raw A/B decimal prompt -> one prompt-reader layer -> fixed context -> scratchpad
```

The recurrent scratchpad, answer reader, optimizer, batch size, and 2,000-step
budget stay unchanged. Compare its result to the baseline, especially on the
two OOD splits.

Interpretation:

```text
better held-out and OOD accuracy:
    making digit roles explicit in the fixed context is useful

only better in-range accuracy:
    likely extra capacity, not length generalization

no improvement:
    prioritize the recurrent number representation or output-writing path
```

Commands for the matched pair:

```bash
# Baseline
python3 controlled_experiments/run_multiplication.py \
  --preset easy --steps 2000 --scratchpad_slots 4 --recurrences 4 \
  --prompt_reader_layers 0

# First experiment: one fixed prompt-reader layer
python3 controlled_experiments/run_multiplication.py \
  --preset easy --steps 2000 --scratchpad_slots 4 --recurrences 4 \
  --prompt_reader_layers 1
```
