# Consult 15 Clean Recipe — Working Notes

## Files Read
- `README_PACK.md`
- `NEW_RESULTS.md`
- `NEW_RESULTS_11nov.md`
- `NEW_RESULTS_11nov_evening.md`
- `NEW_RESULTS_12nov_morning.md`
- `EXPERIMENT_LOG.md` (tail: 11 Nov - 12 Nov)

## Key Full-Test Results (2,759 questions x 8)
- `original`: 0% saved / 0.0 acc
- `nothink` (thinking off): 94% saved / -23.0 acc
- `rc1-soup-30`: 57.4% saved / -3.5 acc (A1/A2/ca1 DPO rounds + "stop when sure" cut pairs)
- `rpE-40 / 60 / 80`: 50.6% / -1.9; 47.8% / -1.2; 47.0% / -0.9 (rc1 + right-beats-wrong repair DPO)
- `rpI-120`: 41.9% / +0.1 (rpE-80 + more repair on new questions)
- `rpK-20 / 40 / 60`: 49.9% / -1.3; **51.2% / -0.8 (PASSES)**; 54.4% / -1.4 (rpE-80 + typed repair: "think less" only on IFEval + tools)
- `rpL-40 / 60 / 80`: 55.1% / -1.37; 59.3% / -1.7; 62.8% / -2.06 (rpE-80 + typed2: IFEval pairs no longer reward longer responses + anti-loop)
- `rpM-20 / 40 / 60`: **50.1% / -0.94 (PASSES)**; 52.9% / -1.10; **54.1% / -0.93 (PASSES, BEST)** (rpE-80 + typed3: math protected with wide guards & length-matched be-right; counting pairs only on true length rules)
- `p1r1s0-200` (Path 1 round 1 from original): 43.4% / -1.5
- `p1cr1s0` (Path 1 round 1 v3 from original, more protective pairs): Dev-512 34.4% / +2.9 (barely shortens)
- `bidax2` (1:1 shortest-correct DPO replica from original): 51.2% / -3.0
- `rpN` (hold repair on rpL-60): no effect on Dev-512 (~59.9% / +1.28 vs rpL-60 58.9% / +0.92).

## Next Steps in Investigation
1. Read `PATH1_FINAL.md`, `REPAIR_IFEVAL_TOOLS.md`, `papers/INDEX.txt`, `plans_and_docs/THINKINGCAP_FULL_DOSSIER.md`.
2. Inspect `code/repair_pairs.py`, `code/dpo.py`, `code/rl_train.py`, `code/sft.py`, `code/track.py`.
3. Analyze `tables_full/ten_models_answer_statistics.json`, `tables_full/all_runs_with_per_type_scores.json`, `tables/rpM60_vs_original_analysis.json`, and the per-question / per-answer tables.
4. Read the 26 papers in `papers/` to ground our analysis and recipe design in published findings.

## Additional Files Read (Turn 8)
- `PATH1_FINAL.md` (complete)
- `REPAIR_IFEVAL_TOOLS.md` (complete)
- `papers/INDEX.txt`
- `plans_and_docs/THINKINGCAP_FULL_DOSSIER.md` (Parts 0, A, G3.2, G3.3, G4, G5, G6, G7, G8, G9, G10, G11, G12)
- `plans_and_docs/FINDINGS_ALL.md`
- `plans_and_docs/CURRENT_STATE_2026-11-10.md`
- `EXPERIMENT_LOG.md` (complete 10 Nov - 12 Nov sections)

## Detailed Findings from Dossier & Log
1. **Why Single-Round Mixed DPO from Original Failed (Path 1 Round 1 & D1-D12):**
   - `p1r1s0-200` (from original, 60-70% shorten / 30-40% compact be-right, 2,964 pairs, 200 steps, fp32+fp16): landed at **43.4% saved / -1.5 acc** on full test (math 59/-3.2, code 44/-3.3, IFEval 23/+0.4, simple 40/-0.2, multiple 42/+1.2, no-fit 45/+0.4). Why? Two reasons: (a) under fp32+fp16 DPO, one offline round moves the model to the token length of its chosen answers (weighted chosen/rejected token ratio in the pairs was 46% saved -> Dev landed at 45.9%); (b) IFEval's response guard (`--resp-guard 0.9`) and counting-rule pairs selected chosen answers with responses 47 tokens LONGER than rejected (76% of pairs had longer responses: 273 -> 330 response tokens on Dev), killing IFEval saving (23% on full test).
   - `p1cr1s0` (Path 1 Round 1 v3, adding widened guards, length-held be-right, anti-loop pairs): on Dev-512 landed at **34.6% / +2.03** (step 100), **32.4% / +2.51** (step 150), **34.4% / +2.90** (step 200). Adding protective pairs where chosen is longer (widened guards 700 vs 487 tok, anti-loop, length-held right) holds length and prevents shortening! Shortening and protecting in the same single round from the original cancels out shortening.
   - `bidax2` (1:1 shortest-correct vs longest-wrong from original, no min-solve): **51.2% / -3.0** on full test. `lpc-bidax5q` (720 screen: 46.7% / -0.59; michal-bida's recipe replica, off-limits to copy).
   - `D1-D12` pair screen: D6 (similar-length right>wrong + shorten) -> 28.1% / +1.7; D12 (shortest right > longest right >=1.5x + shortest right > shortest finished wrong 40:60) -> 17.8% / +0.8; D9 (Ada-R1) -> 65.3% / -6.8.

2. **Why the Multi-Stage Line (`rc1-soup-30` -> `rpE-80` -> `rpK/L/M`) Worked:**
   - `rc1-soup-30` (57.4% / -3.5) overshot saving first (via A1 -> A2 -> ca1 iterative on-policy DPO + rc1 cut-calibration DPO).
   - `rpE-80` (47.0% / -0.9) repaired accuracy on `rc1-soup-30`'s own answers using equal-length / re-read-preferring "be right" pairs (giving back ~10.4 pts of saving to gain +2.6 pts of accuracy, lifting +1.45 pts above the curve).
   - `rpK / rpL / rpM` (typed repair on `rpE-80`'s own answers) recognized that after `rpE-80`:
     - Math (54% / -1.2) and Code (46% / -1.0) already had substantial saving and needed accuracy protection (early-stop guards + length-matched be-right + anti-loop), NOT more shortening.
     - IFEval (45% / -2.6) and BFCL tools (37-50% saved) still had bloated thinking where thinking length does not help accuracy (or hurts on no-fit!), while IFEval's accuracy loss was 2/3 caused by shortened RESPONSES on 58 "at least N words/sentences" questions, plus runaway response loops (2.4% of answers = 26% of tokens).
   - `rpM-60` achieved **54.1% saved / -0.93 acc** (math 57/-1.2, code 58/-0.9, IFEval 50/-2.3, simple 45/-0.3, multiple 47/-1.6, no-fit 61/+4.2), clearing the >50% / > -1.0 box with +4.1% saving margin! Even better, `rpM` moved from step 20 (50.1% / -0.94) to step 60 (54.1% / -0.93) gaining +4.0 pts saving at ZERO accuracy cost.
   - `rpN` (accuracy-only hold repair on `rpL-60`, 59.3% / -1.7) had ZERO effect on Dev-512 (~59.9% / +1.28 vs 58.9% / +0.92), and IFEval responses kept shrinking (273 -> 157 tokens), proving that once a model is over-shortened to the point where response generation collapses and tool calls start dropping, a gentle hold-DPO cannot pull response lengths back up because the model's own rollouts no longer contain long enough responses!

## Additional Files Read (Turn 21)
- `code/repair_pairs.py` (complete)
- `code/dpo.py` (complete)
- `code/rl_train.py` (excerpt)
- `code/sft.py` (excerpt)
- `code/track_excerpts.py` (complete)
- `code/probe_curve.py` (excerpt)
- `tables/rpM60_vs_original_analysis.json` (complete)
- `tables_full/ten_models_answer_statistics.json` (headers/keys)
- `tables_full/fulltest_per_question_all_models.csv` (analyzed across all 14 models & 6 datasets)
- `tables/path1_round1_pairs_local_build_summary.txt`, `tables/path1_round1_v3_p1cr1s0_pairs_summary.txt`, `tables/rpK_pairs_summary.txt`, `tables/rpL_pairs_summary.txt`, `tables/rpM_pairs_local_build_summary.txt`
- `tables_full/all_runs_with_per_type_scores.json` (complete)
- All 26 papers in `papers/` (`2501.12599` through `2609.29664`)

## Literature Synthesis (26 Papers) & Connection to Our Data
1. **Two-Stage Paradigm & The "Short-is-Correct" Trap (`2602.20945`, Wu et al. 2026 - "The Art of Efficient Reasoning"):**
   - Identifies that efficient reasoning naturally decomposes into Stage I (Length Adaptation: rapid exponential compression of CoT length) and Stage II (Reasoning Refinement: increasing information density per token to recover/improve accuracy within the compressed length budget).
   - Identifies the **"Short-is-Correct Trap"**: when training couples correctness with brevity (e.g., rewarding short-correct and penalizing long-wrong or long-correct together), the model hacks the causal bias ("short = right") and omits verification/re-checking steps, collapsing accuracy on multi-step problems.
   - Finds that training on **high-pass-rate (easier) prompts** with a larger rollout count ($N \ge 8\text{--}24$) provides dense reward signals and transfers length compression to harder tasks (like AIME'25 and LiveCodeBench) without collapsing reasoning.
2. **Non-Monotonic Length-Accuracy & Selective Sensitivity on Partially Solvable Questions (`2609.29664` CARE, He et al. 2026; `2602.09591` Nohara et al. 2026):**
   - `2609.29664` (CARE) proves on Qwen3 models that reasoning length primarily affects accuracy on **partially solvable questions** (empirical solve rate $p \in (0.25, 0.75]$). On high-solve-rate questions ($p > 0.75$), thinking can be compressed heavily with negligible accuracy loss; on partially solvable questions, accuracy peaks near the question's median correct thinking length $t_{\text{med}}$ and degrades sharply when thinking falls below $t_{\text{med}}$ (premature termination / skipped verification) or far above it (cyclic repetition).
   - `2602.09591` shows that excessively long outputs suffer from sample dispersion (cyclic loops / wandering), while overly compressed outputs shift the mode wrong.
3. **Decoupling Efficiency from Correctness (`2510.04474` DRPO; `2505.13379` Thinkless/DeGRPO; `2506.05256` ALP; `2504.21370` ShorterBetter):**
   - `2510.04474` (DRPO) shows that mixing length penalties with correctness in one step corrupts the learning signal (penalizing verbose-correct answers pushes them toward negative advantage, discouraging valid reasoning). Decoupling "efficient right vs. inefficient right" (on high-solve questions) from "right vs. wrong" (without length penalty) prevents reasoning degradation.
   - `2506.05256` (ALP) scales length penalty inversely with per-prompt solve rate $\hat{p}(x)$: easy/confident prompts get strong shortening pressure, while partially-solved prompts get zero shortening pressure.
   - `2504.21370` (ShorterBetter) uses Sample Optimal Length (shortest correct among $N$ rollouts) to strip repetition and excessive self-verification.
4. **Why Naive RL / Truncation Fails (`2510.15110` DLER; `2503.14476` DAPO; `2503.20783` Dr. GRPO; `2504.05185` Fatemi et al.; `2504.01296` ThinkPrune; `2508.09726` GFPO):**
   - `2503.20783` (Dr. GRPO) & `2504.05185` prove that standard length-normalized losses bias wrong answers to become longer and cause collapse when unsolvable questions dominate.
   - `2510.15110` (DLER) & `2503.14476` (DAPO) show that RL length compression requires dynamic sampling (filtering out all-wrong/all-right when needed), batch-wise normalization, clip-higher, and small KL.
5. **Multi-Stage SFT/DPO + Refinement & Long2Short (`2501.12599` Kimi k1.5; `2601.02972` Rakotonirina et al.; `2507.15758` LAPO; `2607.09786` Little):**
   - Kimi k1.5 (`2501.12599`) introduces Long2Short DPO (shortest correct vs. longest correct $\ge 1.5\times$ and long wrong) and model merging.
   - `2601.02972` shows multi-stage training (concise initialization + verification-preserving refinement that only penalizes tokens after the answer is settled) beats single-stage methods.
   - `2607.09786` shows length penalties strip explicit verbalization of cues unless verification is explicitly protected.

## Quantitative Breakdown Across Models, Types, and Solve-Rate Bands (Full Test, 22,072 answers/model)

1. **Where the Original Model Spends Tokens (Normal vs. Runaway Loops):**
   - **SVAMP (1,000 q x 8):** 686.1 mean tok (676.5 thinking, 9.6 response), 88.7% acc. Answers >2,000 tok are 7.25% of outputs (14.9% have 4+ repeated 8-grams, consuming 45.4% of all SVAMP tokens) and add **227.0 tokens** to the mean (normal thinking <=2,000 is 486.1 tok; median total is 391 tok).
   - **MBPP+ (378 q x 8):** 2,125.3 mean tok (2,078.7 thinking, 46.7 response), 52.4% acc. Answers >4,000 tok are 16.90% of outputs (40.8% have 4+ repeated 8-grams, consuming **76.5%** of all MBPP tokens; 7.80% hit the 8,192 cap) and add **1,172.5 tokens** to the mean (normal thinking <=4,000 is 1,099.3 tok; median total is 1,058 tok).
   - **IFEval (541 q x 8):** 696.0 mean tok (432.8 thinking, 263.1 response), 58.73% acc. Answers >1,500 tok are 7.53% of outputs (+234.4 tok). Critically, **2.91% of IFEval responses exceed 1,500 tokens** (runaway response loops, 3.60% hit cap), adding **99.1 tokens** to the mean response (normal response <=1,500 is 168.9 tok; median response is 105 tok).
   - **BFCL Simple (400 q x 8):** 194.9 mean tok (170.6 thinking, 24.2 response), 82.2% acc. >500 tok = 1.69% (+15.8 tok).
   - **BFCL Multiple (200 q x 8):** 215.9 mean tok (192.2 thinking, 23.6 response), 76.1% acc. >500 tok = 2.25% (+18.1 tok).
   - **BFCL No-Fit / Irrelevance (240 q x 8):** 212.8 mean tok (185.8 thinking, 27.0 response), 83.4% acc. >500 tok = 4.74% (+40.7 tok).

2. **Why `p1r1s0` (43.4% / -1.5) Fell Short and How It Illuminates the 2-Step Solution:**
   - On **IFEval**, `p1r1s0` cut thinking from 432.8 to **193.7 tok (-55.2%)**, but its blanket response guard (`--resp-guard 0.9` + counting pairs on all IFEval prompts) selected longer responses in 76% of pairs (+47.3 tok), driving runaway response loops (>1,500 tok) from 2.91% up to **4.99%** (+173.7 tok added to mean response, raising mean response from 263.1 to **345.3 tok**). That single flaw reduced IFEval total saving to 22.5% and dragged the 4-benchmark geometric mean down by 3.6 points (from 47.05% to 43.43%).
   - On **SVAMP (59.4% / -3.24)** and **MBPP+ (44.0% / -3.31)**, `p1r1s0` lost accuracy on **high-solve (7-8/8)** questions (-4.74 pts on SVAMP, -6.05 pts on MBPP) and on **mid-solve (2-6/8)** MBPP questions (-4.69 pts), because mixing shortening with compact `be_right` in a single round from the original fell into the "Short-is-Correct Trap" (`2602.20945`): the model shortened below the per-question verification threshold on multi-step math/code problems.
   - Compare `p1r1s0` vs `rpM-60` (54.1% / -0.93) by base solve-rate group (`hi` 7-8/8, `mid` 2-6/8, `lo` 0-1/8):
     - **SVAMP:** `p1r1s0` is `hi -4.74 / mid +2.31 / lo +6.91` (overall -3.24 at 59.4% saved); `rpM-60` is `hi -3.44 / mid +8.52 / lo +9.84` (overall **-1.20 at 57.0% saved**). The entire +2.04 pt accuracy recovery on SVAMP comes from restoring verification on `mid` (+6.21 pts) and `hi` (+1.30 pts) questions while keeping >2,000-token loops at 0.51%!
     - **MBPP+:** `p1r1s0` is `hi -6.05 / mid -4.69 / lo +0.74` (overall -3.31 at 44.0% saved); `rpM-60` is `hi -6.69 / mid +3.55 / lo +2.96` (overall **-0.86 at 58.2% saved**). Repairing `mid` (+8.24 pts) and cutting >4,000-token loops from 8.30% (adds 522.5 tok) to 3.87% (adds 212.0 tok) simultaneously boosted MBPP saving by +14.2 pts AND accuracy by +2.45 pts!
     - **BFCL No-Fit:** `rpM-60` cuts response tokens from 27.0 to **5.4 tok** (refusals need a 1-sentence response, not a 27-token explanation or false tool call) and thinking from 185.8 to 78.3 tok, achieving **60.7% saved / +4.17 acc** (`mid +17.00, lo +23.21`).

3. **Three Clean Recipes Ranked:**
   - **Recipe 1 (Primary Recommended — 2 Steps: "Overshoot Saving, Then Calibrate & Protect Accuracy" via 2-Stage On-Policy DPO):**
     - Pathway: `original -> Step 1 (Confidence-Gated Compression DPO) -> Step 2 (Length-Matched Accuracy & Anti-Loop Repair DPO) -> final`.
     - Step 1 Job: Compress thinking into the 53-57% total saving band by pairing shortest-correct against verbose-correct ($\ge 1.45	imes$) and looped/long-wrong on confident rollouts ($c \ge 5/8$), with response-neutrality (`resp_c <= 1.05 * resp_r`) and explicit response/thinking anti-loop pairs so IFEval responses never bloat.
     - Step 2 Job: Recover +1.8 to +2.5 pts of accuracy on Step 1's own rollouts without giving back saving, using per-question observable rules: (a) length-matched `be_right` ($\pm 15\%$ thinking length, preferring verification markers `"check"/"verify"/"test"` when verification density dropped) + premature-cut guard ($c_{	ext{think}} < 0.70\, t_{	ext{med}}$ wrong vs $pprox t_{	ext{med}}$ right) on partially-solved questions ($1 \le c \le 6/8$), and (b) pure thinking trim ($c \ge 6/8$, $\ge 1.35	imes$) + response/thinking loop-killers ($>2.5	imes$ median length) + minimum-length-constraint satisfaction (detected directly from prompt text regex `at least N words/sentences/paragraphs/sections` or response length variance).
   - **Recipe 2 (Alternative — 3 Steps: Two-Round Progressive On-Policy Compression DPO -> Pure Accuracy-Hold Repair DPO):**
     - Pathway: `original -> Step 1 (Round-1 Compression DPO to ~46% saved) -> Step 2 (Round-2 On-Policy Compression DPO to ~57% saved) -> Step 3 (Length-Matched Verification Repair DPO) -> final`.
     - Avoids single-step overshoot risk by using the exact proven two-round on-policy compression mechanics (each fp32+fp16 DPO round moves length to the chosen mean of the current policy's rollouts: Round 1 $	o 46	ext{--}48\%$, Round 2 $	o 56	ext{--}59\%$), followed by a single pure accuracy-repair DPO stage (`rpE`/`rpM` style).
   - **Recipe 3 (Alternative — 2 Steps: Confidence-Gated ShorterBetter GRPO -> Length-Matched Verification DPO):**
     - Pathway: `original -> Step 1 (Dynamic-Sampled ShorterBetter GRPO with Solve-Rate-Scaled Length Penalty) -> Step 2 (Length-Matched Verification & Loop-Killer DPO) -> final`.
     - Uses on-policy RL (`rl_train.py`, Group-8, Dr.GRPO/ALP/ShorterBetter reward where length penalty scales with empirical solve rate $\hat{p}(x)$ and applies only to thinking tokens) to reach ~55% saving continuously on-policy, then runs one short offline DPO repair step to lock in verification and format precision.
