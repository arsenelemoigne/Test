```python
# Now let's carefully design Section 4 (Strategy for the next week on two L4 GPUs: exact sequence of experiments, settings, expected results with ranges, decision rules, Phase 0 first or not) and Section 5 (Traps + Better Route within constraints).

# Let's check the timing and GPU budget for 1 week on two L4 GPUs:
# 2 L4 GPUs x 24h x 7 days = 336 theoretical GPU-hours, realistic ~200-240 GPU-hours accounting for Colab disconnects/restarts.
# Let's check the exact timing of operations from EXPERIMENT_LOG.md:
# - Building the 512-question non-benchmark Dev Set + 512-question Confirmation Set (from own clean data: GSM8K-std/GSM-IC/Plus, KodCode, IFEval-default matched, ToolACE/xLAM no-fit) + running raw Qwen x8 on both:
#   1,024 questions x 8 = 8,192 generations -> ~50 min on 1 GPU (or ~25 min across 2 GPUs).
# - Should they run Phase 0 first?
#   YES, a **modified, bias-corrected Phase 0 (2-3 GPU-hours)** — NOT to launch the 7-stage forced-cut ladder, but to settle two critical empirical questions on the 1,500 own training questions:
#   (a) What is the *split-sample* (4 select + 4 confirm) forced-cut oracle vs the *natural shortest-of-8* oracle per type?
#   (b) What is the predictability of `b*(q)` from prompt features alone (rank correlation with prompt length and `p0(q)`), proving once and for all whether bottom-up per-question routing is viable or whether natural compaction is required?
#   Let's calculate what the split-sample Phase 0 will actually show (based on `benchlot` L2266 and `lot B`):
#   - On unsplit 8 samples with 1/8 slack, Phase 0 would falsely show ~62-68% saved at +1 to +4 pp accuracy.
#   - On split-sample (4 select + 4 confirm, 0 slack, monotone suffix, 4 coarse rungs `0, p25, p50, p75, nat`), Phase 0 will show:
#     SVAMP: ~38-45% thinking saved at -0.5 to -1.5 pp;
#     MBPP+: ~65-72% thinking saved at -0.5 to +0.5 pp;
#     IFEval: ~45-50% thinking saved (~28-31% total saved!) at -0.5 to +1.0 pp;
#     BFCL: ~45-50% thinking saved (~40-44% total saved) at 0.0 to +1.5 pp;
#     Headline 4-benchmark split-sample forced-cut oracle: **~46-51% total saved at -0.5 to -1.0 pp** (FAILING the 60% / -0.5 pp Go/No-Go gate!), with prompt-length correlation `|rho| < 0.20`.

# Now, what IS the better route (and how does it compare to a 3-rung coarse natural ladder)?
# Let's check what the team's own data says is the closest, highest-leverage, single-line recipe that obeys EVERY constraint:
# Constraint check:
# 1. One clean, reproducible recipe (a single training line), not a stack of corrective stages.
# 2. No copying of any competitor's recipe (michal-bida, Lightning McQwen3, others), even credited.
# 3. Training only on the model's own answers (no teacher-written answers).
# 4. Compute: two Colab L4 GPUs (24 GB each).
# 5. No decision may use the test or the 720-question screen any more.

# Look at the team's own discoveries in `EXPERIMENT_LOG.md`:
# Discovery A: On 10 Nov (`rpE-40full`, `rpE-60full`, `rpE-80full`, `rpI-120full`), the team discovered that:
#   - Equal-length right>wrong pairs where the chosen answer re-reads the question (`re-read filter`) + IFEval `--count-rules` + strong `no-fit` representation (`+3.4 to +5.8 pp` on full test) is the ONLY signal in 205 runs that lifts Qwen3-0.6B +1.5 to +1.64 points above the DPO curve on the full test!
#   - Look at `rpe-40full` vs `rpe-60full` vs `rpe-80full` on the full 2,759 x 8 test:
#     - `rpe-40full`: **50.6% saved / -1.90 pp** (v: 59%/-3.0, c: 53%/-1.7, f: 47%/-2.8, bfcl: 42.6%/-0.07)
#     - `rpe-60full`: **47.8% saved / -1.20 pp** (v: 55%/-1.5, c: 48%/-0.7, f: 46%/-3.0, bfcl: 41.4%/+0.40)
#     - `rpe-80full`: **47.0% saved / -0.90 pp** (v: 54%/-1.2, c: 46%/-1.0, f: 45%/-2.6, bfcl: 41.2%/+1.04)
#   - Look at WHY `rpe-60full` (47.8% / -1.20) and `rpe-80full` (47.0% / -0.90) missed >50% / > -1.0 pp by just 2.2-3.0% saving and 0.2 pp accuracy:
#     1) `rpE` did NOT have `--count-rules` on IFEval (that was added later in `rpH`/`rpI`!). Because `rpE` only found 154 IFEval pairs without `--count-rules`, IFEval lost **-2.6 to -3.0 pp** on the full test! When `rpI` used `--count-rules` with 320 IFEval pairs (152 counting-rule) on current-model rollouts, full-test IFEval jumped from **-2.6 pp to -1.5 pp** (+1.1 pp on IFEval = **+0.28 pp on the 4-benchmark headline**!), and `no-fit` jumped from **+3.4 pp to +5.8 pp** (+2.4 pp on `no-fit` = **+0.17 pp on the 4-benchmark headline**!).
#     2) If `rpe-60full` had `rpI`'s IFEval (-1.5 pp instead of -3.0 pp) and `no-fit` (+5.8 pp instead of +1.4 pp), let's compute its exact full-test headline accuracy:
#        `-1.20 + (-1.5 - (-3.0))/4 + (240/840)*(5.8 - 1.4)/4 = -1.20 + 0.375 + 0.314 = -0.51 pp`!
#     3) And why did `rpE` lose saving from step 40 (50.6%) to step 80 (47.0%) and `rpI` to 41.9%? Because `rpE` and `rpI` contained **0% length-anchor (shorten) pairs**! They were 100% correctness pairs on top of `rc1-soup-30`, so every step let the model drift longer!
#     4) Notice that on 10 Nov (`od1` and `od2`, L5142, L5156), the team ALREADY built `odpo` (online DPO from raw Qwen, `od2`, and from `rpE-80`, `od1`), which mixes `guard` (with re-read + IFEval `--count-rules`) and `shorten` (shortest right > longest right >= 1.5x on questions solved >= 6/8, never on IFEval counting rules)!
#     5) Even better: how can we formulate this as **ONE clean, self-contained, reproducible bottom-up or 2-stage on-policy recipe from raw Qwen** (using ONLY own data, NO competitor datasets, NO stack of 5 legacy checkpoints, and preserving the best parts of the team's budget-ladder intuition)?

# Let's check the numbers just computed!
acc_rpe60_with_rpi_fixes = -1.20 + (-1.5 - (-3.0))/4.0 + (240.0/840.0)*(5.8 - 1.4)/4.0
acc_rpe40_with_rpi_fixes = -1.90 + (-1.5 - (-2.8))/4.0 + (240.0/840.0)*(5.8 - (-0.4))/4.0
print(f"rpe-60full (47.8% saved) with rpI's IFEval & no-fit -> {acc_rpe60_with_rpi_fixes:+.2f} pp")
print(f"rpe-40full (50.6% saved) with rpI's IFEval & no-fit -> {acc_rpe40_with_rpi_fixes:+.2f} pp")
```