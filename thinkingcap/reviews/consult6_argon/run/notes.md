# Consult 6 (Argon) - Quantitative Research Notes

## 1. Task & Official Evaluation Math
- Model: Qwen3-0.6B, thinking mode ON (`<think>...</think>` + final answer).
- Target: > 50.0% tokens saved (thinking + answer combined) AND > -1.00 pp accuracy change on the full test (2,759 questions x 8 samples = 22,072 generations), using ONE clean, reproducible recipe (single training line, own answers only, no competitor copying, no test-time prompt/switch).
- Official evaluation breakdown (2,759 questions):
  - SVAMP (v): 1,000 questions (weight 0.25)
  - MBPP+ (c): 378 questions (weight 0.25)
  - IFEval (f): 541 questions (weight 0.25)
  - BFCL (pooled weight 0.25): 840 questions = simple (s: 400, weight 400/3360 = 11.90%), multiple (m: 200, weight 200/3360 = 5.95%), irrelevance/no-fit (i: 240, weight 240/3360 = 7.14%).
- Headline accuracy change = `(acc_v + acc_c + acc_f + (400*acc_s + 200*acc_m + 240*acc_i)/840) / 4`.
- Headline tokens saved = `1 - (ratio_v * ratio_c * ratio_f * ratio_bfcl)**0.25`.

## 2. Original Model (Raw Qwen3-0.6B) Full Test Baseline (`full-base`, 2,759 x 8) & Thinking-Off (`ntfull2`, 2,759 x 8)
- Raw thinking ON vs OFF accuracy and mean thinking tokens:
  - SVAMP (1,000): ON 88.7% (677 think tok) -> OFF 20.6% (-68.1 pp, 89-98% saved)
  - MBPP+ (378): ON 52.4% (2,079 think tok) -> OFF 40.8% (-11.6 pp, 89-98% saved)
  - IFEval (541): ON 58.7% (433 think tok, ~247 resp tok) -> OFF 57.5% (-1.2 pp, 62% total tok saved because response stays ~247 tok!)
  - BFCL simple (400): ON 82.2% (171 think tok) -> OFF 71.4% (-10.8 pp)
  - BFCL multiple (200): ON 76.1% (192 think tok) -> OFF 55.3% (-20.8 pp)
  - BFCL irrelevance / no-fit (240): ON 83.4% (186 think tok) -> OFF 90.3% (+6.9 pp! Thinking hurts no-fit by talking model into calling a function).
  - Pooled BFCL (840): ON 81.09% -> OFF 72.97% (-8.12 pp).
  - Overall 4-benchmark mean accuracy: ON 70.22% -> OFF 47.97% (-22.25 pp).

## 3. All Full-Test Runs (2,759 x 8, Official 4-Benchmark Weighting)
| Model | Saved (%) | Acc (pp) | vs Curve | SVAMP (v) | MBPP+ (c) | IFEval (f) | BFCL Pool | Simple (s) | Mult (m) | No-fit (i) |
|---|---|---|---|---|---|---|---|---|---|---|
| cleanA-rc1-full (rc1-soup-30) | 57.4 | -3.50 | +0.95 | 67 / -6.6 | 63 / -3.5 | 51 / -2.5 | 44.6 / -1.41 | 41 / -0.1 | 44 / -0.8 | 51 / -4.1 |
| pt-mid-full (4-adapter merge) | 53.6 | -3.30 | +0.30 | 61 / -6.0 | 62 / -3.9 | 50 / -1.7 | 37.1 / -1.41 | 35 / +0.2 | 38 / +0.4 | 40 / -5.6 |
| bidax2-full (bida copy on dp13) | 51.2 | -3.00 | +0.11 | 64 / -6.0 | 42 / -2.4 | 53 / -1.1 | 41.9 / -2.52 | 40 / -2.2 | 43 / -2.0 | 44 / -3.5 |
| rpe-40full (rc1 + 40 repair steps) | 50.6 | -1.90 | +1.11 | 59 / -3.0 | 53 / -1.7 | 47 / -2.8 | 42.6 / -0.07 | 39 / +0.4 | 41 / -0.6 | 50 / -0.4 |
| cleanA-ca1-full (ca1-soup-10) | 49.1 | -2.40 | +0.32 | 58 / -4.3 | 53 / -2.2 | 45 / -2.4 | 37.9 / -0.58 | 36 / +0.4 | 38 / +0.6 | 41 / -3.2 |
| lpc-l2s50e-soup-100 (full) | 48.0 | -2.08 | +0.45 | 55.9 / -3.2 | 51.9 / -2.2 | 45.8 / -1.6 | 36.3 / -1.37 | 33.4 / -0.8 | 36.6 / -0.2 | 40.9 / -3.3 |
| rpe-60full (rc1 + 60 repair steps) | 47.8 | -1.20 | +1.29 | 55 / -1.5 | 48 / -0.7 | 46 / -3.0 | 41.4 / +0.40 | 38 / +0.1 | 39 / -0.2 | 49 / +1.4 |
| rpe-80full (rc1 + 80 repair steps) | 47.0 | -0.90 | +1.49 | 54 / -1.2 | 46 / -1.0 | 45 / -2.6 | 41.2 / +1.04 | 37 / +0.1 | 39 / +0.1 | 50 / +3.4 |
| michal-bida (public ref) | 47.0 | -0.78 | +1.61 | 56 / -2.41 | 33 / -1.22 | 55 / -0.28 | 39.8 / +0.79 | 35 / -1.41 | 42 / -1.12 | 46 / +6.04 |
| rpi-120full (rpE-80 + 40 on-policy) | 41.9 | +0.10 | +1.64 | 47 / +0.0 | 38 / +0.5 | 41 / -1.5 | 41.0 / +1.54 | 37 / +0.2 | 38 / -0.9 | 50 / +5.8 |

## 4. Screen-720 vs Full-Test Optimism Bias (9 Paired Evaluations)
- Across the 9 models evaluated on both Screen-720 (x4) and Full Test (2,759 x 8):
  - Mean saving bias (720 minus Full): +1.52% (range -0.1% on rpE-40 to +3.4% on rpE-80 and +3.3% on ca1-soup-10).
  - Mean accuracy bias (720 minus Full): +0.47 pp (range -0.30 pp on rc1-soup-30 to +1.50 pp on ca1-soup-10, +0.80 pp on rpI-120, +0.72 pp on pt-mid, +0.70 pp on rpE-80).
  - Crucial lesson: When a checkpoint is selected because it looks good on a screen (e.g., rpE-80: 50.4% / -0.20 on 720; ca1-soup-10: 52.4% / -0.90 on 720), it regresses by ~3.3-3.4% on saving and ~0.7-1.5 pp on accuracy on the full test (rpE-80 full: 47.0% / -0.90; ca1 full: 49.1% / -2.40).

## 5. Mathematical & Empirical Flaws Found in the Budget Ladder Design
1. **Fundamental Mechanism Mismatch (Cut-Prefix Conditioning vs Unconditioned Generation):**
   - In Phase 0 and Phase 1 data collection, a "budget B answer" is created by generating a natural (long, leisurely) thinking trace, cutting it at the last sentence end <= B tokens, inserting `</think>`, and generating the final answer.
   - Wait: at test time, there is NO external cutter inserting `</think>` at B tokens, AND there is NO budget token B in the prompt!
   - Even worse: in Phase 1 step 2 ("The CURRENT model answers each question 8 times at the stage's budget B_k and 8 times at each lower budget"), if the model is trained at Stage 1 (B_0 = 0, empty thinking `<think></think>`), then at Stage 1 the model learns to close `</think>` immediately! When you move to Stage 2 (B_1 = 5th percentile, ~10-60 tokens), how does the current model "generate thinking and get cut at B_1" if Stage 1 just trained it to emit `</think>` at token 0?! Conversely, if Stage 1 did NOT make it stop at 0, why did Stage 1 exist?
   - And for DPO pairs built from forced cuts (e.g. cutting a 600-token reasoning chain at token 120 and appending `</think>`), the prefix up to token 120 was written at a 600-token pace (setup, restating the question), NOT a compact 120-token calculation! Only the single token `</think>` after the cut differs from the natural continuation. The team ALREADY tested this exact family of methods in Track 2g:
     - `stopB..stopB5`: DPO on forced-cut pairs -> global stop habit that either barely moves (10-16% in `stopV3` soups 25-200) or causes a sudden cliff where code collapses (+11% -> +69% cut, -5.6 pp in `stopB5` soup 25; +85% cut, -8.4 pp in `stopB3` soup 30).
     - `stopV2`: adding "keep thinking" (hard/guard) pairs to cut pairs -> model got 10-16% LONGER (`-9.9%` to `-15.8%` saved)!
     - `gold1` / `teach1c`: training `p(</think>)` at cut points -> catastrophic collapse (96% saved, -63 pp accuracy) or OOM/weak transfer.
2. **The "Cannot Predict Budget from Prompt Alone" & Uncalibrated Confidence Wall:**
   - The ladder assumes each question q has a level `b*(q)` and the 0.6B model will learn to stop at `b*(q)` on unseen questions from the prompt alone.
   - Team's own data refutes this:
     - `zbench` (25 Oct, L1535): position/length predictor dev AUC 0.722 -> benchmark test AUC **0.508** (pure chance!).
     - Early confidence AUC for predicting whether an answer will end wrong (26 Oct, L2295): in the 1st/2nd paragraph or first 25-50% of thinking, AUC on GSM8K is **0.38-0.50** (chance/inverted!), MetaMath **0.51-0.55**, IFEval **0.57-0.60**.
     - `lpt6` spill-over analysis (5 Nov, L4560, L4605): the compression matrix across the 6 data types is **rank-1 (first component explains 95% of variance; cosine 0.991)**. Even across DIFFERENT data types with blatantly different prompt formats (math vs Python code vs JSON tool schemas), a LoRA r16/r32 on Qwen3-0.6B cannot isolate length habits by type (math-only adapter shortens code 51% vs math 59%; tools-only adapter shortens code 59% and math 53%). Expecting a single LoRA adapter to learn 10 fine-grained per-question budget levels *within* each type when it cannot even separate math from Python code is contradicted by all 205 runs.
3. **Level Definition `b*(q)` Statistical Flaw (1/8 Slack + 10-Rung Stopping Bias):**
   - Allowing `p_rung(q) >= p0(q) - 1/8` (1 answer of slack out of 8 = -12.5 pp!) across 10 rungs (`0, p5, p10, p20, p30, p40, p50, p65, p80, p90`) creates massive multiple-testing stopping bias:
   - Exact binomial calculation: if a question has true natural accuracy `p_nat = 0.80` and an early rung has true accuracy `p_early = 0.60` (a **-20.0 pp** true accuracy loss!), a single rung passes `k_r >= k_0 - 1` **47.3%** of the time, and across 4 early rungs it is falsely assigned to one of those failing rungs **81.1%** of the time! Even if `p_early = 0.50` (**-30.0 pp** loss), it passes >=1 of 4 early rungs **62.4%** of the time!
   - In Phase 0, evaluating the "oracle curve" on the *same* 8 re-answers used to pick `b*(q)` inflates oracle accuracy by **+6 to +8 pp** (and inflates token savings by ~15-20%). That will trigger a false "GO" on Phase 0! (The team already discovered this exact bug on 26 Oct L2266 in `benchlot` and fixed it then with a 4+4 half-split check, which cut SVAMP oracle saving from 53%/+8.9 to 41%/0.0!).
4. **Total Tokens vs Thinking Tokens Arithmetic (IFEval Response Floor):**
   - Headline savings count **thinking + final answer tokens**.
   - On IFEval, final answers are ~247 tokens out of ~667 total tokens (37-38% of total tokens; `ntfull2` thinking OFF saves only 62% on IFEval vs 89-98% on other types).
   - If IFEval final responses are not shrunk (essential to avoid the -0.65 pp counting-rule loss), cutting IFEval thinking by 65% saves only `0.62 * 65% = 40.3%` of total IFEval tokens (`rpE-50`: think 420->153, resp 247->245 = 40.3% total saved).
   - Because BFCL also plateaus around 40-43% total saved (due to fixed JSON tool call tokens and short 171-192 base thinking), hitting >50.0% geometric-mean total savings requires **SVAMP + MBPP+ to average >= 57.5% total savings** (e.g. SVAMP >= 58%, MBPP+ >= 57%).
   - Yet in `benchlot` (L1515, L2266), SVAMP's *strict oracle forced-cut ceiling* on the original model is only **41% (0.0 pp) to 47% (-1.0 pp)** because raw Qwen's SVAMP thinking settles late (accuracy is 0.34 at 20-40% kept vs 0.73 full)! Thus, forced cuts of raw Qwen thinking CANNOT reach 58% math saving without compacting/rewriting the reasoning density itself.
5. **7-Stage Chained Reference Drift & Compute Explosion:**
   - Setting `reference = previous stage` across 7-10 stages means total drift from the raw model is unconstrained (`sum KL(pi_k || pi_{k-1})` allows linear drift in log-odds). In rounds 1->5 (`Track 2e/2f`), chained DPO rounds compounded length cuts (42% -> 54% -> 61% -> 65% -> 69%) while eroding code/math accuracy to -3.9..-5.4 pp, OR when right>wrong pairs dominated (`rpG`, `rpH`, `mf1`), drifted 15-24% LONGER.
   - Compute cost: 7 stages x 800 questions x 8 samples x (k+1) rungs + Tracking Panel (300 questions x 8 samples x 11 rungs = 26,400 generations after *every* stage) + up to 2 retry batches per stage x 2 seeds x 2 arms (online + offline) = **>1.5 million generations (~150-200 GPU-hours)**, which exceeds one week on 2 L4 GPUs (~336 GPU-hours total wall-clock, less with Colab disconnects).

## 6. Counterfactual & Strategy Numbers
- On the full test (2,759 x 8):
  - `rpe-40full` scored **50.6% saved / -1.90 pp** (v: 59%/-3.0, c: 53%/-1.7, f: 47%/-2.8, bfcl: 42.6%/-0.07 with no-fit -0.4).
  - `rpe-60full` scored **47.8% saved / -1.20 pp** (v: 55%/-1.5, c: 48%/-0.7, f: 46%/-3.0, bfcl: 41.4%/+0.40 with no-fit +1.4).
  - `rpe-80full` scored **47.0% saved / -0.90 pp** (v: 54%/-1.2, c: 46%/-1.0, f: 45%/-2.6, bfcl: 41.2%/+1.04 with no-fit +3.4).
  - `rpi-120full` (which added `--count-rules` on 320 IFEval pairs incl. 152 counting-rule pairs and fresh current-model rollouts) moved IFEval from -2.6/-3.0 pp to **-1.5 pp** (+0.28 to +0.38 pp headline gain) and `no-fit` from +1.4/+3.4 pp to **+5.8 pp** (+0.17 to +0.31 pp headline gain).
  - Counterfactual: `rpe-60full` with `rpI`'s IFEval (-1.5 pp) and `no-fit` (+5.8 pp) reaches **-0.51 pp** headline accuracy; `rpe-40full` with those two fixes reaches **50.6% saved / -1.13 pp** (only 0.13 pp from the goal before adding length-anchor pairs!).
  - Why `rpE` lost saving from step 40 (50.6%) to step 80 (47.0%) and `rpI` to 41.9%: both had **0% shorten (length-anchor) pairs**, so training drifted back toward longer answers. Adding 25-30% shorten pairs on reliably solved questions (`>= 6/8`, never on IFEval counting rules) holds saving at 51-53% while the equal-length re-read + `--count-rules` + `no-fit` pairs lift accuracy to -0.5..-0.9 pp.

## 7. Second Analysis (11 Nov): Fact-Check, Claude Review, Code Bugs, and Path 1 Redesign

### A. Fact-Check (`FACT_CHECK.md`) Assessment
- **Accept 9 of 10 corrections unconditionally:**
  1. `D4 lpd-budget` was built on `bidax5q` (`--bida-exact` + bida math + DAST target) and scored 49.0/-1.42 (+1.29 vs curve) on 720, below `bidax5q-final` 46.7/-0.59 (+1.71 vs curve). Path 3 was right to be dropped under the no-competitor rule.
  2. `rpI` drifted from 50.4% (720) / 47.0% (full) at step 80 down to 42.9% at step 100, holding flat at 43.0% (41.9% full) only from step 100 to 120; `rpH` on 720 did not show an IFEval gain (-2.5 at step 100, -2.1 at step 120); mixing gains measured at 42% saved into a 50.6%-saved checkpoint is a heuristic motivation, not an additive proof; the 55-65% subjective win probability had no frequentist basis.
  3. `bidax6p` (`fp32 + fp16 autocast`) moved along the curve (-11.4 pts saving for +3.3 pts accuracy at step 100, +0.3 to +1.1 vs curve), not a 2.8 pp vertical jump at fixed saving.
  4. `zbench` 0.508 was the position baseline; `zbench` MLP was 0.736 and own confidence 0.717, and `zbench2` timing-only AUC on SVAMP was 0.963-0.977 for own confidence (`0.850` on MBPP+, `0.806` MLP on BFCL); `gold1`/`teach1c` failed from unanchored BCE; `stopV2` lengthened from long-chosen guard pairs + NLL; the 4+4 split was `lot V2` (whereas `benchlot` used 20 forced re-answers split into two halves of 10 at L2266).
  5. `l2s50e-final`, `push2-final`, `margin-final`, `rpB`, `rpC`, `lpt6mix` are 720 screens; `rc1-rl-40` had a full test (`55.4% / -2.6 pp`, L3622), making 11 full-test runs total; `rpG` lost 17.7 pts of saving (50.4 -> 32.7), not 25; `rpA` used `lr 5e-6` on `pt-mid`; upstream commit `965a54c` only requires agent traces; re-read rates `51/77/94 vs 41/57/82` were observed in `rpE` (though `--prefer-reread` IS an active selection filter in `repair_pairs.py` L168).
- **One partial nuance on Point 5 (IFEval response lengths):** While L5042 (from the 720 x 4 screen) noted `rpE @ 80` shortened mean response tokens (`247 -> 190`) and beat raw Qwen on the 120-question 720 slice (`59.0 -> 59.4`), the subsequent **full-test (541 x 8) rule diagnostic at L5095-5100 (`ir-full/ifeval_rules.json`)** explicitly found: *"every shortened model (rc1, ca1, rpE-40/60/80) loses on COUNTING rules: length_constraints:number_words -11 to -13 pts... Too-short responses: original 265, rc1 399, ca1 364, rpE-80 379; responses on length-rule questions median 209 -> 145-161 tokens."* And `consult5` (L5227) computed that this IFEval deficit costs 0.65 pt of `rpE-80`'s full-test aggregate. So both facts in the log are real: overall IFEval errors are driven primarily by early thinking stops (`<0.5x` median: 626-815 errors vs 62 in raw Qwen), AND on length/counting rules specifically, too-short responses (379 vs 265) compound the loss.

### B. Weighing `CLAUDE_REVIEW.md`
- **Strong Agree (Confirmed by Smoking-Gun Log Evidence `D12 lpd-chain1` at L4767-4773):**
  1. **Round 1 as written in `PATH1_PLAN.md` (`39.4%` shorten, `46.9%` equal-length right, `13.7%` capped guard) will severely under-shorten (~18-33% saved, not 45-50%):** Beyond `lpc-runa2` (32.5-33.4% saved) and `D6 simright` (28.1% saved), **`D12 lpd-chain1` (L4767-4773)** tested a 40% shorten / 60% shortest-right>shortest-wrong mix in `fp32+fp16` and landed at **21.7% saved (step 100) -> 18.4% (step 200) -> 17.8% (final)**!
  2. **Round 2 cannot add saving AND accuracy simultaneously (`od1`: 50.4/-0.2 -> 53.3/-1.8 on 720, falling from +2.8 to +1.75 vs curve):** Every successful repair in the log (`rc1 -> rpE-40/60/80`, `pt-mid -> rpB-20/rpC-20`) works in the opposite direction: **Round 1 must overshoot saving to 54-58% saved**, and **Round 2 trades 3-5 pts of excess saving (`55-57% -> 51-52%`) for +1.8 to +2.5 pp of accuracy**.
  3. **Round 1 math supply in `p1src` is broken:** Dropping `metamath_*` and `math1`/`math2` leaves only 661 math questions in `dp13` (420 of them `gsm8k_easy`), and taking 60% leaves **~396 math questions** (~85% always-right), which cannot fill 800 math pairs (520 requiring mixed outcomes).
  4. **Code bugs in `track.py` / `repair_pairs.py`:** Ignored CLI flags in `ptrepair` (`--rp-exclude-ids`, `--rp-warmup`, `--rp-lr-min`, `--rp-dpo-seed`), non-distinct seeds, text-level overlap between `gsm8k_std` and `dp13` GSM8K, and missing paired candidate-vs-reference bootstrap in `dev_score.py`.
- **Nuanced Disagree with Claude on Two Points:**
  1. **Do NOT drop `rpJ` if it is already queued/running on GPU 2 (~1.5 GPU-h with zero new generation), AFTER fixing `--kinds mix`'s `right` pair bug:** While `rc1-soup-30` (57.4% saved) is not representative of a 30%-saved Round 1, it **IS** representative of the **redesigned 55-57%-saved Round 1** that both Claude and we agree Path 1 must use! Thus, testing whether adding 20-30% shorten anchors to `rpE`'s guard recipe arrests `rpE`'s 57.4% -> 47.0% length drift at ~51-53% saved on `rc1-soup-30` is the exact right test for Round 2—provided `repair_pairs.py --kinds mix` uses `rpE`'s actual compact-right (`checked[0] or rights[0] > wrongs[0]`) pair instead of `right_long`/`right_short`!
  2. **Why `rpJ`'s current `--kinds mix` code in `repair_pairs.py` is STILL broken even after the 2 fixes (`eb712bc`):**
     Even with kind-interleaving and `tmed` fixed, `--kinds mix` in `repair_pairs.py` (Lines 134-138, 210-213) replaces `rpE`'s proven `--kinds guard` pair (`shortest right with reread within 1.5x > shortest finished wrong`) with `right_long` and `right_short` (`max/min <= 1.2`, half with `chosen > rejected`), where `take()` calls `rng.shuffle(pool)` across questions (destroying `--prefer-reread` priority across questions and allowing long right answers!). In `rpE`, `--kinds guard` was used specifically because `right_long`/`right_short` was worse.
