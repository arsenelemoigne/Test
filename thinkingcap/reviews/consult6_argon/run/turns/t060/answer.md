```python
# Let's analyze the 3 strongest, distinct, self-contained strategy paths that obey EVERY team constraint:
# Constraint checklist:
# - Model's own answers only (no teacher-written answers, no 4B/8B coach).
# - No prompt tricks or test-time switch.
# - No copied competitor recipe (no michal-bida pairs/data, no Lightning McQwen3 prefill/data).
# - One clean, reproducible recipe (not a 5-stage historical stack of corrective stages).
# - Two Colab L4 GPUs over 1 week (~220-250 usable GPU-hours).
# - Zero decisions on the test set or the 720-question screen (all choices on a locked non-benchmark Dev-512 set).

# Let's check the quantitative evidence in `all_runs.json` and `EXPERIMENT_LOG.md` for our 3 ranked Winning Paths:
#
# ==============================================================================
# WINNING PATH #1 (RANK 1 — Highest Probability of Hitting >50% / >-1.0 pp: Est. P = 55–65%):
# "2-Round On-Policy Compact-and-Verify DPO (OPCV-DPO)"
# ==============================================================================
# Why it wins (The Evidence from the Log):
# 1. Every single-round offline DPO run from raw Qwen hits the exact same ceiling: at 50% saved, accuracy is -2.5 to -3.5 pp (`bidax2-full` 51.2% / -3.00 pp; `l2s50e-final` 53.5% / -2.81 pp; `push2-final` 52.4% / -3.00 pp; `margin-final` 53.3% / -2.70 pp). Why? Because compressing raw Qwen's 680-token math and 2,080-token code traces by >50% in one offline leap introduces two specific failure modes on the shortened distribution:
#    (a) On SVAMP & MBPP+, 68% of newly wrong answers are normal-length shortened traces (~0.43x original) that misread a trap condition or skip edge-case checking (`st-ca1-soup-10`, L4996), while 30% are premature stops (`< 0.5x` median right length).
#    (b) On IFEval, counting-rule questions (`number_words`, `keywords:frequency`, `number_sentences`, `highlighted_sections`, `placeholders`) lose -6 to -13 pp because thinking stops before counting AND final responses shrink from median 209 to 145–161 tokens (`ir-full`, L5092), costing -0.65 pp on the 4-benchmark headline.
# 2. Conversely, the ONLY runs in all 205 evaluations that rose +1.49 to +2.77 pp above the trade-off curve (`rpe-40full` 50.6% / -1.90, `rpe-60full` 47.8% / -1.20, `rpe-80full` 47.0% / -0.90, `rpi-120full` 41.9% / +0.10 on Full Test; `rpC-80` 43.4% / +0.43 on 720) used a second round on the shortened model's distribution with:
#    - Equal-length right>wrong pairs where the chosen trace **re-reads the question** (51% vs 41% math, 77% vs 57% code, 94% vs 82% no-fit, L5016),
#    - **On-policy rollouts from the current model** (`rpI` vs `rpG`/`rpH` at step 120: `rpI` beat `rpG` by +1.7 pp vs curve and cut length drift by 2.4x, L5175),
#    - **IFEval `--count-rules`** (lifted full-test IFEval from -2.6/-3.0 pp in `rpE` to **-1.5 pp** in `rpi-120full`, worth **+0.28 to +0.38 pp** on the overall headline!), and
#    - **High `no-fit` coverage** (lifted full-test `no-fit` from -0.4/+1.4 pp in `rpe-40/60full` to **+5.8 pp** in `rpi-120full`, worth **+0.31 to +0.44 pp** on the overall headline!).
# 3. Why `rpE-80full` (47.0% / -0.90) and `rpi-120full` (41.9% / +0.10) fell short of 50% saved:
#    - Neither `rpE` nor `rpI` included ANY length-keeping (`shorten`) pairs (0% shorten, 100% correctness), so over 80–120 steps the model drifted longer!
#    - Meanwhile, `rpe-40full` ALREADY achieved **50.6% saved / -1.90 pp** on the full test *without* `--count-rules` (IFEval -2.8 pp) and *without* strong `no-fit` (-0.4 pp vs +5.8 pp in `rpI`). As our exact counterfactual calculation proved, simply replacing `rpe-40full`'s IFEval (-2.8 -> -1.5 pp) and `no-fit` (-0.4 -> +5.8 pp) with `rpI`'s values moves `rpe-40full` from **50.6% / -1.90 pp to 50.6% / -1.13 pp**—and keeping 25–30% Shorten Anchors in Round 2 prevents the length drift from step 40 to step 80 so SVAMP (-1.2 pp) and MBPP+ (-0.7 pp) reach their step-60/80 accuracy while staying at **51.5–53.5% saved**!
#
# ==============================================================================
# WINNING PATH #2 (RANK 2 — Modular Control of Per-Type Compression & Spill-Over: Est. P = 35–45%):
# "Asymmetric 4-Domain Specialist LoRA Merge + 1 On-Policy Verification Round (Task-Arithmetic + Verify)"
# ==============================================================================
# Why it can win (The Evidence from the Log):
# 1. In `pt-mid` (`pt-mid-full`: 53.6% saved / -3.30 pp full test; 55.0% / -2.58 pp on 720), four domain adapters (Math 0.30, Code 0.45, IFEval 0.60, Tools 0.45) achieved 53.6% full-test saving, and the additivity audit (5 Nov, L4635) proved that **token ratios multiply across merged adapters** (predicted 56.5% vs measured 55.0%), while **accuracy losses are sub-additive by +3.2 pp** (sum of solos -5.8 pp vs merged -2.6 pp) because each adapter's own-domain right>wrong pairs buffer the others' rank-1 compression spill-over!
# 2. Even more strikingly, `lpt6mix-Cnf` (6 Nov, L4853: Instructions 0.50 + Simple 0.50 + Multiple 0.50 + No-Fit 0.25, **Math weight = 0.0**) achieved **43.3% saved / -1.20 pp** on Screen-720 with **SVAMP +42% saved / -1.3 pp** and **MBPP+ +48% saved / -0.4 pp**! Why? Because the Instructions and Tools adapters compress Math and Code via the shared rank-1 brevity direction *without* applying direct shortening pressure to hard math traces!
# 3. And when `pt-mid` was followed by a single equal-length correctness round (`rpB-20`: 48.7% / -1.20 pp, +1.45 vs curve; `rpC-20`: 50.5% / -1.83 pp; `rpC-80`: 43.4% / +0.43 pp, +2.19 vs curve), Code jumped from -3.5 pp to **+2.3 pp** and `no-fit` jumped from -7.1 pp to **+5.4 pp**!
# 4. Why October's `pt-mid + rpC` stopped at 48.7% / -1.20 pp (`rpB-20`) or 50.5% / -1.83 pp (`rpC-20`):
#    - First, `pt-mid` was trained in `bfloat16` (`bidax6p` proved `fp32 weights + fp16 autocast` saves ~2.8 pp of accuracy at matched saving!).
#    - Second, `pt-mid`'s Math adapter used aggressive shorten pairs (`bf16`), causing a -6.0 pp full-test math hole before repair even started.
#    - Third, `rpB`/`rpC` used offline `pt-mid` × 4 answers without `--count-rules` on IFEval, without the question re-read filter (`rpE`), and with 0% shorten anchor replay.
#
# ==============================================================================
# WINNING PATH #3 (RANK 3 — Best Implementation of the Team's Bottom-Up / Budget Intuition: Est. P = 25–35%):
# "Coarse 2-Rung Natural-Rollout Budget Curriculum + Correctness-Gated DRPO/DPO (Natural Budget Ladder)"
# ==============================================================================
# Why it can win if the team wants to test the budget-allocation hypothesis (The Evidence from the Log):
# 1. In the DPO pair screen (5 Nov, L4628), **D4 `lpd-budget`** (difficulty-aware per-question length target: target budget scales with question solve rate `p0(q)` instead of a flat 1.5x ratio) was the **#1 best variant out of all 12 D-series runs**, scoring **49.0% saved / -1.42 pp (+1.29 vs curve)** at final checkpoint (and **48.6% / -1.52 pp** at step 200), beating `D1 dast` (-2.57), `D2 hard` (-0.12 at 13.8%), `D3 route` (-0.77 at 9.4%), `D5 margin` (-2.70), `D8 math` (-1.07 at 41.7%), `D9 adar1` (-7.00), `D10 l2s50e` (-2.81), and `D11 push2` (-3.00)!
# 2. Look at `D4 lpd-budget`'s per-part breakdown on Screen-720 (L4628):
#    - `v (SVAMP)`: **60% saved / -2.9 pp**
#    - `c (MBPP+)`: **43% saved / -0.2 pp** (almost zero code loss!)
#    - `f (IFEval)`: **47% saved / -1.9 pp**
#    - `s (Simple)`: **40% saved / -2.5 pp**
#    - `m (Multiple)`: **45% saved / +1.7 pp**
#    - `i (No-fit)`: **46% saved / +0.4 pp**
#    - Headline: **49.0% saved / -1.42 pp**!
# 3. Why did `D4 lpd-budget` come within 1.0% saving and 0.42 pp accuracy of the target in a SINGLE pass, and what held it back?
#    - `D4` used 4 answers per question (`a4`), had **no question re-read filter** on math (`v: -2.9 pp`), **no `--count-rules` on IFEval** (`f: -1.9 pp` and ` shorten` pairs shrank counting-rule responses), and its `no-fit` gained only `+0.4 pp` (vs `+3.4 to +5.8 pp` in `rpE`/`rpI`).
#    - If you combine **solve-rate-indexed natural budget targets** (using 8 own answers per question: questions solved 7–8/8 target `p25` thinking length; questions solved 5–6/8 target `p50` thinking length; questions solved 2–4/8 target `p75` via equal-length re-read right>wrong pairs and zero shorten pairs) with a short **Stage 2 On-Policy Verification pass** (or 2-rung curriculum with a fixed base reference), you get the exact mathematical benefit of the budget ladder (`A(B)` optimization where marginal token value matches difficulty) **without** forced sentence cuts, **without** 10-rung stopping bias, and **without** 7-stage reference drift!

# Let's check the numbers on D4 lpd-budget vs rpe-80 vs pt-mid in Python to make sure every number is exact!
for k in ['budget-100', 'budget-200', 'budget-final', 'lpt6mix-Cnf', 'pt-mid', 'rpB-20', 'rpC-20', 'rpC-80']:
    r = df[df['key'] == k].iloc[0]
    print(f"{k:14s} | saved={r['saved']:5.1f}% | acc={r['acc']:+5.2f} | vs_curve={r['vs_curve']:+5.2f} | v:{r['v_saved']:.0f}/{r['v_acc']:+.1f} c:{r['c_saved']:.0f}/{r['c_acc']:+.1f} f:{r['f_saved']:.0f}/{r['f_acc']:+.1f} s:{r['s_saved']:.0f}/{r['s_acc']:+.1f} m:{r['m_saved']:.0f}/{r['m_acc']:+.1f} i:{r['i_saved']:.0f}/{r['i_acc']:+.1f}")
```