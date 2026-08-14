# One Layer Deeper — exploration experiments

Exploratory data collection for the squaring-mod competition. **Nothing here is
a submission.** Read `SPEC_NOTES.md` first — it lists where the working task
summary differed from the repo's actual generator (biggest: `x` is always
coprime to `N`; output digits are variable-width, MSD-first, tail-aligned onto
the prompt; the scored ladder shares T values with training — OOD-ness is in
fresh prompts and unseen moduli, plus T-extrapolation past the trained range).

## Which interpreter (important)

**Run everything here with `/usr/bin/python`, not the repo's `.venv`.**

| interpreter | torch | CUDA | usable |
|---|---|---|---|
| `/usr/bin/python` | 2.7.0 | cu128 | ✅ matches the 12.8 driver |
| `.venv/bin/python` | 2.13.0 | cu130 | ❌ needs a CUDA 13 driver → CPU fallback |

The `.venv` build is newer than this box's driver, so `torch.cuda.is_available()`
is False there and runs land on CPU ~200x slower. That silently produced one
round of invalid wall-clock data (see `results/README_CPU_INVALID.md`).
`pick_device` now raises rather than falling back, and `results.csv` records
the device per row.

The `.venv` is still what the *competition* runner needs (pinned deps); it just
can't reach the GPU on this machine, which will also block local
tier-faithful `benchmark.runner` runs until its torch is rebuilt for cu128.

## Dependencies

`torch` and `matplotlib` only (everything else is stdlib) — both already
present in the local interpreter, so `explorations/requirements.txt` is
documentation rather than something you need to install. `jsonargparse` is
missing locally but is only required to regenerate the *official* datasets via
`python -m data.squaring_mod`; these scripts call the generator functions
directly and don't need it.

```bash
pip install -r explorations/requirements.txt   # no-op on this machine
```

## Run order

```bash
cd <repo-root>              # scripts add repo root to sys.path themselves

# 0. Task characterization: CPU, ~minutes. Everything depends on this.
python explorations/exp0_characterize.py

# 1. Baseline accuracy-vs-T curves (~15-20 min on one L40)
python explorations/exp1_baseline.py

# 2. Compute profile (~5 min; add --compile for the compile-time number)
python explorations/exp2_profile.py
python explorations/exp2_profile.py --compile

# 3. Depth x width grid (~1.5 h default; see trimming note in the script)
python explorations/exp3_grid.py

# Analysis + plots + the explicit verdicts, from whatever has been run so far
python explorations/plots.py
```

## Outputs

- `results/exp0_task_analysis.json` — periods, pre-periods, ladder coverage,
  chance baselines per regime. `task_analysis.md` (the verdict) is written by
  hand after reading it.
- `results/results.csv` — one row per (run, eval set): full config, per-digit +
  exact accuracy, chance baselines, ID/OOD flags, steps completed, wall-clock,
  params, budget mode, diverged/flat flags.
- `results/exp2_profile.json` — phase timings, bwd/fwd ratio, utilization,
  kernel tables.
- `results/plots/*.png` — accuracy-vs-T (period marked), depth×T heatmaps,
  compute breakdown.
- `FINDINGS.md` — written after the runs, from `plots.py` output + the CSV.

## Design decisions (and why)

- **Tokenization is the repo's own** (`tokenize_squaring_mod_with_result`,
  imported, not reimplemented): digit-wise decimal, MSD first, variable width,
  `[N] d(N) [X] d(x) [T] d(T)` prompt, answer read out at the tail positions of
  the prompt in one bidirectional forward pass. Digit-wise is also the choice
  most likely to generalize across N — but here it isn't a choice at all: the
  evaluator fixes it, so we match it exactly.
- **Bidirectional attention, padding mask only** — matches the evaluator.
- **Exact match = all answer digits correct**; per-digit reported alongside.
  Chance baselines are computed per eval set from the actual label
  distribution (best constant predictor), because leading digits are heavily
  skewed (results < N) and a constant model can look deceptively strong.
- **Budget modes**: every sweep runs both `steps` (expressivity) and `seconds`
  (the real competition constraint; the clock includes model construction, as
  rule 11 charges it). Curves are labeled with their mode in the CSV and plots.
- **Splits**: depth cohorts are matched `(N, x)` pairs excluded from training
  (asserted); OOD-N moduli are disjoint from training moduli (asserted); ladder
  rungs with `T ∉ train_T` carry an `ood_t` flag. This mirrors the competition
  rather than enforcing full train/eval T-disjointness, which the real
  evaluator does not do (see SPEC_NOTES §4).
- **One untuned LR (3e-4)** across all cells; diverged and flat-from-start runs
  are flagged in the CSV and by `plots.py` instead of silently reported.
- **Ladder is denser than the competition's** (adds 3, 6, 12, 24, 48) so the
  break point and the small-N cycle length are resolvable on a log-T axis.

## Environment caveats

- Local GPUs are **L40s, not H100s**; GPU 0 was busy when checked, so the
  harness auto-picks the freest device (`pick_device`). Absolute wall-clock
  numbers and the roofline ridge point differ from the competition target;
  `exp2_profile.py` takes `--peak-tflops/--mem-bw-gbs` (defaults are L40:
  90.5 TFLOP/s, 864 GB/s; pass ~989/3350 on an H100).
- Grid trimming vs the original request: exp3 defaults drop depths {3, 6} and
  width 64 to fit ~1.5 h (rationale in the script header; full grid via flags).
