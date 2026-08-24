# Competition experiment suite

This suite is designed to spend the Easy submission allowance as a sequence of
decisions, not as one large Cartesian sweep. Run each phase in order and use its
promotion rule before spending submissions on the next phase.

The main questions are:

1. Which basic architecture learns the actual benchmark rather than only the
   controlled arithmetic tasks?
2. Does matching the loss to sequence-level exact accuracy improve results?
3. Does coordinating output digits help more than changing the loss?
4. Which improvements transfer across modulus identities and repeated-squaring
   depths?

## Dataset roles

| Dataset | What it isolates |
| --- | --- |
| E1 | Repeated squaring with one tiny fixed modulus. |
| E2 | Repeated squaring with a larger fixed modulus. |
| E3 | Generalization across moduli at fixed `T=2`. |
| E4 | Generalization across larger moduli at fixed `T=2`. |
| E5 | Joint conditioning on varying modulus and varying `T`. |
| M1 | Deeper repeated squaring with a fixed 14-bit modulus. |
| M3 | Modulus variation at a fixed shallow depth. |
| M5 | Joint modulus/depth variation at medium scale. |

Use E1, E3, and E5 for screening. Use E2 and E4 only to confirm ideas that
survive screening.

## Fixed experimental controls

Unless a row explicitly changes a setting, keep all of these fixed:

- evaluator seed and data order;
- model width and number of attention heads;
- positional representation;
- untied input embedding and output classifier;
- initialization;
- AdamW hyperparameters;
- batch size;
- compilation setting;
- training and evaluation batch sizes;
- total evaluator time budget.

Record the exact source revision and all constants for every run. A comparison
does not count if more than its named variable changed.

## Metrics to record

For each accepted run, save:

- submission ID and filename;
- completed optimizer steps and training seconds;
- final training loss and training exact accuracy;
- test loss and exact accuracy;
- OOD loss and exact accuracy, when reported;
- Max T and accuracy at the first uncertified rung;
- OOD N Max T and accuracy at the first uncertified rung;
- failure, timeout, or compilation notes.

Do not select a winner from rounded mean exact accuracy alone. Prefer, in order:

1. E1 test exact accuracy;
2. E3 test exact accuracy;
3. E3 OOD N depth profile;
4. E5 test exact accuracy;
5. E5 Max T and OOD N Max T;
6. throughput and training loss as diagnostics.

Suggested artifact name:

```text
<phase>_<architecture>_<loss>_<optimizer-tag>_<dataset>.jsonl
```

## Phase 0: architecture controls

Use ordinary token-mean cross-entropy in every row.

| Tag | Architecture |
| --- | --- |
| `u4` | Four untied full-token Transformer blocks. |
| `u8` | Eight untied full-token Transformer blocks. |
| `r4x2` | Four distinct full-token blocks, each stack applied twice. |
| `s2x4` | Current two-block, four-repeat compressed scratchpad control. |

Run matrix:

| Run | E1 | E3 | E5 |
| --- | --- | --- | --- |
| `p0_u4_token_ce` | [ ] | [ ] | [ ] |
| `p0_u8_token_ce` | [ ] | [ ] | [ ] |
| `p0_r4x2_token_ce` | [ ] | [ ] | [ ] |
| `p0_s2x4_token_ce` | [ ] | [ ] | [ ] |

Budget: 12 Easy submissions.

Key comparisons:

- `u4` versus `u8`: value of additional untied depth;
- `u8` versus `r4x2`: cost or benefit of weight sharing;
- `u8` versus `s2x4`: full-token workspace versus compressed scratch state;
- E1 versus E3: fixed-modulus computation versus modulus generalization.

Promotion rule: choose the better of `u8` and `r4x2` as the loss-test backbone.
It must show useful learning on E1. Prefer E3/OOD-N performance over a small E1
advantage. Keep `u4` if it performs similarly while completing materially more
updates.

### Phase 0 result (2026-08-18)

All twelve screening runs completed. The checkboxes above are retained as a
reusable template; the observed evaluation results were:

| Architecture | E1 test / OOD | E3 test / OOD | E5 test / OOD |
| --- | ---: | ---: | ---: |
| `u4` | 0.040 / 0.100 | 0.005 / 0.020 | 0.006 / 0.002 |
| `u8` | 0.033 / 0.060 | 0.007 / 0.011 | 0.005 / 0.005 |
| `r4x2` | 0.047 / 0.050 | 0.004 / 0.004 | 0.008 / 0.000 |
| `s2x4` | 0.013 / 0.100 | 0.003 / 0.006 | **0.013 / 0.015** |

`u8` is promoted over `r4x2` for the controlled loss comparison because it is
better on E3 test and OOD, does not collapse on E5 OOD, and unexpectedly
completed at least as many updates as the shallower full-token variants in
these runs. This is a weak promotion rather than evidence that U8 has learned
modular arithmetic: E3/E5 losses and exact accuracies remain close to a
digit-marginal baseline.

Keep `s2x4` as a secondary branch. It produced the strongest E5 test and OOD
exact accuracy while also having the worst E5 loss among the four models. That
is precisely the kind of loss-versus-exact discrepancy the next phase is meant
to investigate. First run the loss suite on U8 for architectural control; test
the winning loss on S2x4 before rejecting the scratchpad family.

E1 throughput is confounded by its small training split. With batch size 512
and `drop_last=true`, the loader supplies only about one full batch before the
runner recreates its iterator and workers. The full-token models completed only
79--87 updates on E1 versus roughly 379--680 on E3/E5. Treat E1 as a weak
screen at this batch size; a later E1 optimizer control should test batch 256.

## Phase 1A: supervised loss benchmark

Apply each loss to the promoted backbone. Ensure every loss returns a scalar
and averages over examples so gradient scale remains comparable.

### `token_ce`

Current control: flatten all valid answer digits and average their
cross-entropies. Longer answers consequently receive more total weight.

### `sequence_ce`

Compute token CE without reduction, average valid digits within each answer,
then average answers:

```text
token CE -> masked mean per sequence -> mean over sequences
```

### `hard_sequence_05`

Start with per-sequence mean CE. Weight each sequence by its detached loss to
the power `0.5`, normalize weights to mean one, and take the weighted mean.

### `hard_sequence_10`

As above, with detached per-sequence loss to the power `1.0`.

### `focal_05`

Token focal loss with `gamma=0.5`, followed by the same per-sequence and
per-batch averaging as `sequence_ce`.

Run matrix:

| Run | E1 | E3 | E5 |
| --- | --- | --- | --- |
| `p1_token_ce` | [ ] | [ ] | [ ] |
| `p1_sequence_ce` | [ ] | [ ] | [ ] |
| `p1_hard_sequence_05` | [ ] | [ ] | [ ] |
| `p1_hard_sequence_10` | [ ] | [ ] | [ ] |
| `p1_focal_05` | [ ] | [ ] | [ ] |

Budget: 15 submissions, of which three repeat the control. If the Phase 0
control is directly comparable, omit those three repeats.

Interpretation:

- improvement only on E1 indicates better fitting, not modular generalization;
- higher exact accuracy at similar loss indicates better metric alignment;
- lower loss without higher exact accuracy indicates confidence improvement
  without answer coordination;
- hard weighting that hurts E3/E5 suggests the hard examples overwhelm the
  model rather than form a useful curriculum.

Promotion rule: promote a loss only if it improves E3 or E5 exact/depth metrics,
not merely training loss.

### Sequence-balanced CE result (2026-08-20)

| Loss | E3 test / OOD | E5 test / OOD |
| --- | ---: | ---: |
| U8 token CE control | 0.007 / 0.011 | 0.005 / 0.005 |
| U8 sequence CE | 0.005 / 0.008 | 0.007 / 0.003 |

Sequence balancing did not improve exact accuracy. The custom loss is also used
during evaluator loss reporting, so its numerical loss values are not directly
comparable with token-CE evaluation loss. The sequence-CE runs completed far
more updates (1,093 on E3 and 1,504 on E5 versus 421 and 680 for the controls),
but the loss callback is too small to explain that speedup; treat it as
run-environment or service variance, not a property of the loss.

## Phase 1B: sequence-objective and RL-style fine-tuning

Run this phase only after selecting a strong supervised loss. Do not replace CE
entirely: exact-answer rewards are sparse while the model is weak.

### `soft_exact_mix`

For each sequence, sum the log-probabilities assigned to all correct digits.
This is the log probability of an entirely correct answer under the parallel
factorized output policy. Mix the resulting sequence objective with supervised
loss:

```text
loss = sequence_ce + lambda * soft_exact_loss
```

Test `lambda` values `0.05` and `0.2`. Use a numerically stable log-space
implementation. Do not use an unprotected probability product.

### `pg_exact_mix`

Use supervised training for approximately the first 75% of optimizer updates.
For the remainder, sample multiple answer sequences from the same logits and
mix a self-critical policy-gradient term with supervised loss:

```text
loss = sequence_ce + lambda * policy_gradient_loss
```

Initial settings:

- exact-answer reward `1`, otherwise `0`;
- optional small fractional-digit reward as a separate ablation;
- four samples per prompt from the same forward-pass logits;
- detached per-prompt or batch reward baseline;
- `lambda=0.1`;
- no pure-RL phase.

Run matrix:

| Run | E1 | E3 | E5 |
| --- | --- | --- | --- |
| `p1b_soft_exact_005` | [ ] | [ ] | [ ] |
| `p1b_soft_exact_020` | [ ] | [ ] | [ ] |
| `p1b_pg_exact_010` | [ ] | [ ] | [ ] |
| `p1b_pg_exact_digitmix_010` | [ ] | [ ] | [ ] |

Budget: 12 Easy submissions.

Promotion rule: RL-style training must improve E3/E5 test or depth metrics. An
E1-only gain is treated as metric sharpening or memorization.

## Phase 2: answer-generation structure

Use the winning backbone and supervised loss from Phase 1A. Keep the amount of
core computation as comparable as practical.

| Tag | Output mechanism |
| --- | --- |
| `parallel` | Current simultaneous, conditionally independent output slots. |
| `writer_shared` | Sequential/iterative answer states updated by one shared writer block. |
| `writer_untied` | Sequential/iterative answer states with position- or stage-specific writer blocks. |

The writer must remain differentiable and execute inside one model invocation.
Do not sample discrete digits inside the writer for this comparison. Later
answer states may attend to earlier answer states so carries and global answer
constraints have an explicit communication path.

Run matrix:

| Run | E1 | E3 | E5 |
| --- | --- | --- | --- |
| `p2_parallel` | [ ] | [ ] | [ ] |
| `p2_writer_shared` | [ ] | [ ] | [ ] |
| `p2_writer_untied` | [ ] | [ ] | [ ] |

Budget: 9 Easy submissions, including three controls that may be reused when
directly comparable.

Promotion rule: prefer improvement on E3/E5 over E1. If a writer improves only
E1, it likely helps digit coordination without learning reusable modular
arithmetic.

## Phase 3: optimizer and batch-size controls

Run a compact sweep on E3 using the current winner:

| Learning rate / batch size | 256 | 512 | 1024 |
| --- | --- | --- | --- |
| `1e-3` | [ ] | [ ] | [ ] |
| `2e-3` | [ ] | [ ] | [ ] |
| `3e-3` | [ ] | [ ] | [ ] |

Budget: 9 Easy submissions. Take the best three configurations to E1 and E5
for 6 additional submissions.

Because custom losses can change gradient magnitude, compare gradient scale or
at least early loss curves before attributing a result to the loss itself.

## Phase 4: confirmation across all Easy datasets

Select the best three genuinely distinct ideas, rather than three nearby
hyperparameter settings. Run each on E1 through E5.

| Finalist | E1 | E2 | E3 | E4 | E5 |
| --- | --- | --- | --- | --- | --- |
| `finalist_a` | [ ] | [ ] | [ ] | [ ] | [ ] |
| `finalist_b` | [ ] | [ ] | [ ] | [ ] | [ ] |
| `finalist_c` | [ ] | [ ] | [ ] | [ ] | [ ] |

Budget: 15 Easy submissions.

## Medium promotion

Use the six daily Medium submissions to separate depth and modulus behavior:

| Configuration | M1 | M3 | M5 |
| --- | --- | --- | --- |
| `medium_finalist_a` | [ ] | [ ] | [ ] |
| `medium_finalist_b` | [ ] | [ ] | [ ] |

Interpret M1 as fixed-modulus repeated computation, M3 as modulus
generalization, and M5 as their joint test. Do not spend the Hard submission on
an approach whose gain appears only on M1 training/test loss.

## Suggested first-day allocation

| Work | Maximum submissions |
| --- | ---: |
| Phase 0 architecture controls | 12 |
| Phase 1A supervised losses | 15 |
| Phase 1B sequence/RL objectives | 12 |
| Phase 2 answer writers | 9 |
| Failures and targeted follow-ups | 12 |
| Total | 60 |

Run adaptively. Submit Phase 0 first and inspect it before constructing every
later variant. Stop sweeping any backbone that cannot demonstrate useful E1
learning, and stop promoting changes that improve only training loss.

## Results ledger

| Date | Run tag | Dataset | Submission ID | Steps | Train loss | Test exact | Max T | OOD N Max T | Notes |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| | | | | | | | | | |
