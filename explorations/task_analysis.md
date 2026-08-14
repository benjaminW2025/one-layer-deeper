# Experiment 0 — task characterization and the periodicity verdict

Source: `results/exp0_task_analysis.json` (seed 45, 2000 sampled (x, N) pairs per
regime; fixed moduli below 5000 enumerated exhaustively over all units).

Composing `x -> x^2 mod N` T times gives `x^(2^T) mod N`. With `d = ord_N(x)`
and `d = 2^s * d'` (d' odd), the sequence `T -> x^(2^T) mod N` is eventually
periodic with pre-period `s` and period `P = ord_{d'}(2)`. A model that
identifies the cycle answers arbitrarily large T in constant depth. The
question is whether that shortcut is reachable.

## Measured periods

| regime | median ord | median s | median P | p90 P | max P | cycle closes ≤ T=64 | **cycle closes within trained T** |
|---|---|---|---|---|---|---|---|
| fixed N=323 (E1) | 72 | 4 | 6 | 6 | 6 | 100% | **5.6%** |
| fixed N=899 (E2) | 140 | 2 | 12 | 12 | 12 | 100% | 7.1% |
| sampled 10-bit | 90 | 1 | 10 | 30 | 30 | 100% | 4.1% |
| sampled 12-bit | 406 | 2 | 28 | 132 | 308 | 70.9% | 0.3% |
| OOD-N 13-bit | 615 | 2 | 22 | 140 | 572 | 74.1% | 0.2% |
| fixed N=10403 (M1) | 1700 | 1 | 40 | 40 | 40 | 100% | 19.8% |
| fixed N=38021 (M2) | 3136 | 6 | 42 | 42 | 42 | 100% | 14.1% |
| sampled 16-bit | 5162 | 1 | 84 | 700 | 4100 | 42.7% | 1.7% |
| sampled 22-bit | 313628 | 2 | 1212 | 14996 | 365462 | 7.1% | 0.2% |
| OOD-N 24-bit | 1193358 | 2 | 2948 | 56388 | 1671380 | 3.3% | **0.0%** |

Pre-periods are uniformly tiny (median 1–6, max 8), so essentially every ladder
rung sits inside the cyclic part. The period is what varies, over five orders
of magnitude.

## gcd and chance baselines

`gcd(x, N) = 1` for **100%** of competition data — the generator's
`_sample_unit` rejects non-units, so the CRT/non-coprime case never arises and
needs no model handling. (Under uniform x it would occur ~5% of the time at
10 bits, ~0.4% at 22 bits; reported for context only.)

Chance baselines are far from uniform and must accompany every accuracy number.
Best-constant-predictor rates: per-digit 0.12–0.33, exact-match 0.0002–0.085.
Fixed N=323 is the worst offender at **0.085 exact** — the squaring map's image
shrinks with T, concentrating outputs, so a constant predictor looks strong.

## Verdict: sequential computation, not periodicity

**OOD-T generalization is not achievable by learning periodicity in any regime
that matters.** The shortcut fails for two different reasons at the two ends of
the scale, and the failure is total across the middle.

At small N the cycle is short enough to be visible on the scored ladder (100%
of pairs at N ≤ 38021 close within T=64), but it is **invisible from the
training distribution**: only 4–20% of pairs complete a full cycle inside the
trained T range, so the model essentially never observes a repeat during
training. It would have to *extrapolate* periodicity it was never shown —
which is strictly harder than the sequential computation it could instead
learn from the same data.

At larger N the cycle is not merely unobserved but absent from the evaluation
window: at 22 bits only 7.1% of pairs cycle within T ≤ 64, and at the OOD-N
24-bit sizes just 3.3%. Median period there is 1212 and 2948 respectively,
against a maximum tested T of 64. There is nothing periodic to find.

Both ranked Hard profiles therefore demand genuine iterated computation.
Depth is not a trap; deep or recurrent architectures are the indicated
direction, and the competition's explicit blessing of recurrence, iterative
refinement, and adaptive halting reads as the organizers pointing at this.

One caveat this analysis does not settle: it rules out the *cycle-detection*
shortcut, not the *trapdoor* shortcut. A model that internalizes λ(N) or φ(N)
for a seen modulus computes any T in constant depth. That remains available on
the seen-N profile and is precisely why the competition ranks OOD-N Max T
separately — on unseen moduli, recovering the group order is factoring.
