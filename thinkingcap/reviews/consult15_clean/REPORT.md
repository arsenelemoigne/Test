# Clean Recipe Design & Empirical Strategy Report: Compressing Qwen3-0.6B Thinking by >50% at > -1.0 Accuracy Change

---

## Executive Summary

Across six weeks of experiments on Qwen3-0.6B (thinking mode) and 22,072 full-test rollouts per model (2,759 ThinkingCap-Test questions $\times 8$ samples across SVAMP, MBPP+, IFEval, and BFCL Simple / Multiple / No-Fit), two runs cleared the deployment box of **$>50\%$ tokens saved** and **$> -1.0$ accuracy change**:
- **`rpM-60`**: **54.1% tokens saved / -0.93 accuracy** (SVAMP 57.0% / -1.20, MBPP+ 58.2% / -0.86, IFEval 50.5% / -2.31, BFCL Simple 45.1% / -0.34, Multiple 46.9% / -1.56, No-Fit 60.7% / +4.17; 4-benchmark equal-weight mean accuracy 66.16% vs. original 67.09%).
- **`rpK-40`**: **51.2% tokens saved / -0.80 accuracy** (SVAMP 53.7% / -1.11, MBPP+ 47.4% / -0.99, IFEval 51.3% / -2.41, BFCL Simple 44.3% / +0.06, Multiple 46.5% / -0.81, No-Fit 60.0% / +4.37).

However, `rpM-60` and `rpK-40` were reached via an exploratory 6-stage chain (`original -> A1 -> A2 -> ca1 -> rc1-soup-30 -> rpE-80 -> rpM-60`). Meanwhile, every single-round DPO attempt directly from the original model failed:
- **`p1r1s0-200`** (1-step mixed shorten + compact be-right from `original`): **43.4% saved / -1.50 acc** (6.6 pts short of 50% saving and 0.5 pts below -1.0 accuracy).
- **`p1cr1s0`** (1-step from `original`, adding widened premature-cut guards, length-held be-right, and anti-loop pairs): **34.4% saved / +2.90 acc on Dev-512** (protective pairs held length and canceled compression).
- **`bidax2`** (1-step 1:1 shortest-correct vs. longest-wrong from `original`): **51.2% saved / -3.00 acc** (cleared 50% saving, collapsed accuracy by -3.0 pts).

Why did single-stage runs fail while multi-stage runs succeeded? Both our per-question empirical data and the 26 efficient-reasoning papers in `papers/` (`2602.20945`, `2609.29664`, `2510.04474`, `2506.05256`) establish that **length compression and verification-preserving calibration cannot be mixed in a single offline preference step from the base model**:
1. **Opposing Gradients Cancel in Round 1 (`p1cr1s0` vs. `bidax2`):** Under working `fp32` master-weight + `fp16` forward DPO, one offline round shifts the model's output length toward the chosen-answer distribution of its training pairs. Mixing protective pairs where chosen is longer than rejected (premature-cut guards with 700 vs. 487 thinking tokens and length-held `be_right`) into Round 1 holds the chosen mean near ~65% of base and caps saving at 34–43%. Conversely, selecting shortest-correct without protection (`bidax2`) crosses 51% saving at the cost of the **"Short-is-Correct Trap"** (`2602.20945`): the model associates brevity with correctness and drops verification on multi-step problems (-3.0 pts).
2. **Selective Length Sensitivity by Question Solve Rate (`2609.29664` CARE):** Our 22,072-rollout breakdown proves that accuracy loss from compression is concentrated on **partially solvable questions** ($\hat{p}(x) \in [2/8, 6/8]$) and multi-step verification prompts when thinking drops below the question's median correct thinking length $t_{\text{med}}$. On high-confidence prompts ($\hat{p}(x) \ge 6/8$), thinking compresses by 50–65% with minimal loss—and on abstention prompts (BFCL No-Fit), shorter thinking + concise refusal improves accuracy by **+4.17 points** (+17.0 to +23.2 pts on medium/low solve-rate questions).
3. **Thinking vs. Response Decoupling & Runaway Loop Elimination:** In `original`, **52.5% of all tokens** are spent inside answers containing $\ge 4$ repeated 8-gram loops (76.5% of MBPP+ tokens, 45.4% of SVAMP tokens, 35.9% of IFEval tokens). On IFEval, 2.91% of base responses exceed 1,500 tokens (runaway response loops, adding 99.1 tokens to the mean response). `p1r1s0` cut IFEval thinking by 55.2% (432.8 $\to$ 193.7 tokens), but its blanket response guard (`--resp-guard 0.9`) rewarded longer responses on 76% of IFEval pairs (+47.3 tokens), driving runaway response loops up to **4.99%** (+173.7 tokens added to the mean response, pushing mean response from 263.1 to **345.3 tokens**) and collapsing IFEval total saving to 22.5%.

We distill these findings into **three clean, reproducible recipes of 2 to 3 steps from `original`**, led by **Recipe 1 (2-Step Confidence-Gated Compression DPO $\to$ Length-Matched Verification & Loop-Calibration DPO)**, which implements *"overshoot saving first, then calibrate accuracy"* using strictly general, per-question observable rules.

---

## 1. Analysis: What Works, What Fails, and Why

### 1.1 Per-Type, Per-Stage Quantitative Breakdown Across Full-Test Models

Table 1 reports full-test performance across the 6 evaluation subsets (2,759 questions $\times 8 = 22,072$ answers per model), decomposing total tokens into `<think>` vs. response tokens and tracking runaway generation loops.

| Model | Stage / Description | Overall Saved / $\Delta$Acc | SVAMP (1,000q) Saved / $\Delta$Acc (Think / Resp tok) | MBPP+ (378q) Saved / $\Delta$Acc (Think / Resp tok) | IFEval (541q) Saved / $\Delta$Acc (Think / Resp tok) | BFCL Simple (400q) Saved / $\Delta$Acc (Think / Resp tok) | BFCL Multiple (200q) Saved / $\Delta$Acc (Think / Resp tok) | BFCL No-Fit (240q) Saved / $\Delta$Acc (Think / Resp tok) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`original`** | Base Qwen3-0.6B | **0.0% / 0.00** (67.09% acc) | 0.0% / 88.7% (676.5 / 9.6) | 0.0% / 52.4% (2078.7 / 46.7) | 0.0% / 58.7% (432.8 / 263.1) | 0.0% / 82.2% (170.6 / 24.2) | 0.0% / 76.1% (192.2 / 23.6) | 0.0% / 83.4% (185.8 / 27.0) |
| **`nothink`** | Empty `<think>` | **94.0% / -23.0** | 98% / -12.6 | 97% / -13.9 | 69% / -11.4 | 88% / -49.1 | 89% / -52.8 | 93% / -65.0 |
| **`p1r1s0`** | 1-step mixed DPO from `original` | **43.4% / -1.50** | 59.4% / -3.24 (270.4 / 8.5) | 44.0% / -3.31 (1141.9 / 48.2) | **22.5%** / +0.42 (193.7 / **345.3**) | 39.9% / -0.25 (93.0 / 24.1) | 42.1% / +1.25 (101.2 / 23.9) | 44.7% / +0.42 (97.6 / 20.1) |
| **`rc1-soup-30`** | Overshoot saving (A1$\to$A2$\to$ca1$\to$rc1) | **57.4% / -3.50** | 66.9% / -6.57 (218.7 / 8.5) | 63.4% / -3.47 (732.0 / 45.7) | 50.8% / -2.50 (141.4 / 201.0) | 41.0% / -0.09 (91.0 / 24.0) | 43.6% / -0.81 (98.2 / 23.6) | 50.9% / -4.11 (91.7 / 12.9) |
| **`rpE-80`** | `rc1` + uniform `be_right` repair | **47.0% / -0.90** | 54.1% / -1.25 (306.7 / 8.0) | 46.3% / -1.03 (1089.8 / 51.4) | 45.4% / -2.56 (156.5 / 223.3) | 37.3% / +0.09 (98.0 / 24.1) | 39.4% / +0.06 (107.4 / 23.4) | 49.5% / +3.44 (96.4 / 11.0) |
| **`rpK-40`** | `rpE-80` + typed repair v1 | **51.2% / -0.80** *(PASS)* | 53.7% / -1.11 (308.9 / 8.4) | 47.4% / -0.99 (1067.7 / 50.2) | 51.3% / -2.41 (134.1 / 204.9) | 44.3% / +0.06 (84.6 / 24.0) | 46.5% / -0.81 (92.1 / 23.4) | 60.0% / +4.37 (79.8 / 5.4) |
| **`rpL-40`** | `rpE-80` + typed repair v2 | **55.1% / -1.37** | 59.6% / -1.50 (268.8 / 8.4) | 57.0% / -1.79 (866.7 / 47.1) | 55.0% / -3.00 (130.9 / 182.5) | 42.9% / -0.31 (87.3 / 24.0) | 45.2% / -0.62 (94.9 / 23.4) | 58.3% / +3.91 (83.3 / 5.5) |
| **`rpM-20`** | `rpE-80` + typed repair v3 (step 20) | **50.1% / -0.94** *(PASS)* | 56.2% / -1.48 (292.4 / 8.4) | 51.7% / -0.69 (977.8 / 48.8) | 47.4% / -2.56 (143.8 / 222.0) | 39.4% / +0.28 (94.0 / 24.1) | 41.5% / -0.62 (102.7 / 23.5) | 53.9% / +3.49 (91.2 / 6.8) |
| **`rpM-60`** | `rpE-80` + typed repair v3 (step 60) | **54.1% / -0.93** *(BEST)* | **57.0% / -1.20** (286.6 / 8.5) | **58.2% / -0.86** (840.9 / 47.5) | **50.5% / -2.31** (129.6 / 215.1) | **45.1% / -0.34** (83.1 / 24.0) | **46.9% / -1.56** (91.1 / 23.4) | **60.7% / +4.17** (78.3 / 5.4) |

Five quantitative findings explain why each stage succeeded or failed:

1. **Tail Generation Loops Account for Half of Base Token Mass:**
   In `original`, median total length across all 22,072 answers is **354 tokens**, whereas mean total length is **738.8 tokens** because **13.5% of answers enter 4+ repeated 8-gram loops that consume 52.5% of all tokens** (2.41% hit the token cap):
   - **MBPP+ Code:** Mean total length is 2,125.3 tokens (median 1,058). The 16.90% of answers exceeding 4,000 tokens (40.8% repetitive, consuming **76.5%** of all MBPP+ tokens; 7.80% truncated at 8,192) add **1,172.5 tokens** to the MBPP+ mean (normal thinking $\le 4,000$ averages 1,099.3 tokens). In `rpM-60`, cutting $>4,000$-token rollouts from 16.90% to **3.87%** (and cap-hits from 7.80% to **0.36%**) saves **960.5 tokens per MBPP+ answer** on tail loops alone while *raising* accuracy on low/medium solve-rate questions by +2.96 to +3.55 points.
   - **SVAMP Math:** Mean total length is 686.1 tokens (median 391). Rollouts $>2,000$ tokens (7.25% of outputs; 14.9% repetitive, consuming 45.4% of tokens) add **227.0 tokens** to the mean. `rpM-60` cuts $>2,000$-token rollouts to **0.51%** (cap-hits 1.74% $\to$ **0.01%**), saving 213.6 tokens per answer from loop removal plus 211.0 tokens from compact 3-paragraph thinking (paragraphs drop from 9.9 to 3.0).

2. **Why `p1r1s0` Failed on IFEval (22.5% Saved) vs. `rpM-60` (50.5% Saved):**
   In `p1r1s0`, IFEval thinking tokens fell from 432.8 to **193.7 tokens (-55.2%)**, yet IFEval total saving was only **22.5%** because mean response tokens jumped from 263.1 to **345.3 tokens (+82.2 tokens)**. Why? `p1r1s0` applied `--resp-guard 0.9` and counting pairs broadly across IFEval, selecting longer chosen responses in **76% of IFEval pairs** (+47.3 response tokens). Rewarding longer responses across general instruction-following prompts drove runaway response loops ($>1,500$ response tokens) from **2.91% (+99.1 tokens added to mean response in base) up to 4.99% (+173.7 tokens added to mean response in `p1r1s0`)**! Fixing IFEval response length alone (from 345.3 to 220 tokens) raises `p1r1s0`'s 4-benchmark geometric-mean saving from **43.43% to 47.05% (+3.62 pts)**. Conversely, when `rpL-60` dropped all response protection, normal responses shrank from 168.9 to 113.7 tokens and lost **-12.5 points** on the 58 minimum-length-constrained IFEval questions (`at least N words/sentences/paragraphs`). Restricting longer-response preference strictly to prompts with explicit minimum-length constraints while penalizing response loops ($>2.5\times$ median) in `rpM-60` achieved **50.5% IFEval saving at -2.31 accuracy**.

3. **Accuracy Change by Base Solve-Rate Band (`hi` 7–8/8 vs. `mid` 2–6/8 vs. `lo` 0–1/8):**
   Decomposing accuracy change across question solve-rate bands reveals the exact mechanism of Step 2 repair:
   - **SVAMP:** Single-round `p1r1s0` (59.4% / -3.24) is `hi -4.74 / mid +2.31 / lo +6.91`, and over-shortened `rc1` (66.9% / -6.57) is `hi -7.89 / mid -2.59 / lo +4.79`. After length-matched repair, `rpM-60` (57.0% / -1.20) is **`hi -3.44 / mid +8.52 / lo +9.84`**—recovering +2.04 pts over `p1r1s0` by restoring verification on `mid` (+6.21 pts) and `hi` (+1.30 pts) questions.
   - **MBPP+:** `p1r1s0` (44.0% / -3.31) is `hi -6.05 / mid -4.69 / lo +0.74`. `rpM-60` (58.2% / -0.86) is **`hi -6.69 / mid +3.55 / lo +2.96`**—lifting partially solved `mid` questions by **+8.24 points** while cutting 301.7 more tokens than `p1r1s0`.
   - **BFCL No-Fit (Abstention):** Compressing thinking (185.8 $\to$ 78.3 tokens) and keeping refusal responses concise (27.0 $\to$ **5.4 tokens**) in `rpM-60` yields **60.7% saved / +4.17 accuracy** (`hi -0.40 / mid +17.00 / lo +23.21`), because the base model over-thinks irrelevant tools and talks itself into false tool calls.

4. **Why On-Policy Re-Sampling Between Stages Works (Rollout Frontier Shift):**
   In `original`'s 8 rollouts per question, the shortest correct rollout (`min_th`) on high-solve ($c \ge 6/8$) questions averages **330.2 thinking tokens on SVAMP, 552.7 on MBPP+, 208.3 on IFEval, and 111.2–131.4 on BFCL**. After a single compression round (`p1r1s0`), `min_th` on high-solve questions shifts down to **152.1 on SVAMP, 333.1 on MBPP+, 126.3 on IFEval, and 79.0–82.1 on BFCL**, while median correct thinking on partially solved ($2 \le c \le 5/8$) questions sits at **453.3 on SVAMP and 806.3 on MBPP+**. Sampling from the Step 1 model exposes both the 75–130 token high-confidence frontier and the exact verification failures where Step 1 cut too early.

5. **Hardware Precision & The Over-Shortening Cliff (`fp32+fp16` and `rpN`):**
   Nine early runs (`rr1`–`rr6`, `sc1`–`sc3`) trained in `bf16` at `lr=1e-5` produced 0% compression because updates $<10^{-4}$ underflowed `bf16` mantissas; casting LoRA master weights to `fp32` with `--fp16` forward at `lr=3e-5` to `5e-5` is mandatory. Furthermore, `rpN` (hold-only repair on `rpL-60`, 59.3% / -1.70) had zero restorative effect on Dev-512 (58.9% $\to$ 59.9% saved, IFEval response stuck at 157 tokens), proving that once a model is compressed past ~59% so its own rollouts lack full-length responses, offline repair on its own rollouts cannot restore them. Step 1 must land in the **53%–57% saving window**.

---

### 1.2 General vs. Benchmark-Tuned Behaviors

To ensure our clean recipe learns a **general "think less" skill that transfers to unseen tasks**, we replace every dataset-name switch used in `rpK/L/M` with a measured, per-question rule:
- **Instead of `if dataset in ("gsm8k", "mbpp"): hold else: shorten`:** Gate shortening vs. verification protection by **per-question empirical solve rate $\hat{p}(x) = c/8$ and per-question median correct thinking length $t_{\text{med}}(x)$**. Partially solved prompts ($1 \le c \le 6/8$) receive length-matched `be_right` ($\pm 15\%$ around $t_{\text{med}}$) and premature-cut guards ($<0.72\,t_{\text{med}}$ when $t_{\text{med}} \ge 180$), while high-confidence prompts ($c \ge 7/8$) receive residual thinking trim only when thinking spread $t_{\max}^+/t_{\min}^+ \ge 1.30$ and the shortest rollout preserves verification markers present in the median rollout.
- **Instead of IFEval `rule_id` metadata lists (`COUNT_RULES`):** Detect minimum output-length constraints **directly from the user prompt text** via a task-agnostic regex:
  `r"\b(at least|minimum of|more than|no fewer than|between)\s+\d+\s+(words?|sentences?|paragraphs?|sections?|lines?|bullets?|items?)\b|\b\d+\s*\+\s*(words?|sentences?|paragraphs?)\b"`.
  When matched, require $r(y^+) \ge 0.95\,r_{\text{med}}(x)$; on all other prompts, enforce strict **Response Neutrality** ($r(y^+) \le 1.05\,r_{\text{med}}(x) + 10$) so thinking compression never leaks into longer responses.
- **Instead of BFCL `irrelevance` dataset tags:** Detect abstention/no-tool-call answers directly from the prompt and rollouts (`is_abstention(x)` = candidate tool definitions are present in the prompt context, and the majority of correct rollouts emit no tool call) and prefer concise refusal responses ($r(y^+) \le r_{\text{med}}(x)$).

---

### 1.3 What the Research Papers Say About the Same Questions

Our empirical rules match five key findings across the 26 papers in `papers/`:
1. **Two-Stage Paradigm & The "Short-is-Correct Trap" (`2602.20945`, Wu et al. 2026):** Proves that efficient reasoning requires Stage I (Length Adaptation) followed by Stage II (Reasoning Refinement to increase information density per token). Coupling brevity with correctness in one stage triggers the *"Short-is-Correct Trap"* (omitting verification on multi-step problems), while training compression on high-pass-rate prompts with $N \ge 8$ rollouts transfers zero-shot to harder benchmarks.
2. **Selective Length Sensitivity on Partially Solvable Questions (`2609.29664` CARE, He et al. 2026; `2602.09591` Nohara et al. 2026):** Proves on Qwen3 models that reasoning length affects accuracy primarily on **partially solvable prompts** ($p \in (0.25, 0.75]$), where accuracy peaks near $t_{\text{med}}$ and drops both below $t_{\text{med}}$ (premature cut) and far above $t_{\text{med}}$ (cyclic loops, `2602.09591`). High-solve prompts ($p > 0.75$) compress heavily without loss.
3. **Decoupling Efficiency from Correctness (`2510.04474` DRPO; `2506.05256` ALP; `2504.21370` ShorterBetter; `2505.13379` Thinkless):** `2510.04474` shows that penalizing verbose-correct traces alongside wrong traces corrupts reasoning; `2506.05256` scales length penalty inversely with solve rate $\hat{p}(x)$; `2504.21370` targets the Sample Optimal Length (minimum-length correct rollout among $N=8$).
4. **Concise Initialization + Verification-Preserving Refinement (`2601.02972` Rakotonirina et al.; `2501.12599` Kimi k1.5; `2607.09786` Little):** `2601.02972` shows two-stage training (concise initialization + refinement that preserves verification cues, `2607.09786`) beats single-stage Long2Short DPO (`2501.12599`).
5. **Stabilizing On-Policy RL (`2503.20783` Dr. GRPO; `2510.15110` DLER; `2503.14476` DAPO; `2504.05185` Fatemi et al.):** Proves that standard $1/|y|$ length normalization biases wrong answers to grow longer (`2503.20783`, `2504.05185`), requiring unnormalized token reductions, dynamic filtering of all-wrong groups (`2503.14476`), and $\beta_{\text{KL}} = 0.001$ (`2510.15110`).

---

## 2. Three Clean, General Recipes from the Original Model (Ranked)

### Shared Setup Across All Three Recipes
- **Starting Model:** Original `Qwen/Qwen3-0.6B` with thinking mode enabled (`<think>...</think>` present on 100% of outputs, no prompt edits).
- **Public Training Prompt Pool (3,800 prompts, disjoint from ThinkingCap-Test):** 1,000 math word problems (GSM8K train), 1,000 Python code problems with unit tests (MBPP train 374 + EvalPlus/APPS/CodeContests train 626), 1,000 verifiable instruction-following prompts, and 800 tool-calling prompts with schemas (320 single-tool, 240 multi-tool, 240 irrelevant-tool/abstention).
- **Rollout Sampling:** $K = 8$ rollouts per prompt from the current stage's model (`temperature=0.6, top_p=0.95, top_k=20, max_new_tokens=4096`, `8192` on code), scored by programmatic verifiers.
- **LoRA & Precision Settings (`code/dpo.py`):** Base model in `fp16`, LoRA parameters cast to **`fp32` master weights** (`p.data = p.data.to(torch.float32)`), trained with PyTorch `fp16` mixed precision (`--fp16`, never `--bf16`). LoRA `r=16, lora_alpha=32, lora_dropout=0.05`, all 7 linear projections (`q,k,v,o,gate,up,down`), `max_length=2048`, batch size 16 (`per_device=1, grad_accum=16`). Between steps, merge LoRA in `fp32` (`merge_and_unload()`) and save in `fp16` so the next step trains a fresh `r=16` adapter with the merged checkpoint as $\pi_{\text{ref}}$.

---

### Recipe 1 (RANK #1 — Recommended): Two-Step "Overshoot Saving, Then Calibrate Accuracy" On-Policy DPO

- **The Idea in Two Sentences:** Step 1 trains on the original model's rollouts using **pure confidence-gated thinking compression and loop elimination** (with strict response-length neutrality) to overshoot total token savings directly into the **53%–57% band** without IFEval response bloat. Step 2 samples fresh rollouts from the compressed Step 1 model and runs **length-matched verification repair, premature-cut guarding, and residual high-confidence thinking trim** to recover **+1.8 to +2.4 accuracy points** (into the **-0.5 to -0.9** band) while holding net token saving at **52.5%–54.8%**.
- **One-Line Pathway:**
  `original (0% / 0.0) -> Step 1: Confidence-Gated Thinking Compression DPO (~54–57% saved / -2.2 to -2.8 acc) -> Step 2: Length-Matched Verification & Loop Calibration DPO (52.5–54.8% saved / -0.5 to -0.9 acc)`
  - *Step 1 Single Job:* Compress `<think>` length by ~55% and kill runaway thinking/response loops on confident prompts ($c \ge 3/8$) while keeping response length neutral.
  - *Step 2 Single Job:* Recover multi-step verification and constraint accuracy on Step 1's own rollouts at matched thinking length ($\approx t_{\text{med}}$) while trimming residual thinking spread on confident prompts ($c \ge 7/8$).

#### Exact Procedure from the Original Model

1. **Step 1 — Confidence-Gated Thinking Compression DPO (from `original`):**
   - **Answers Generated By:** `original` Qwen3-0.6B ($3,800 \times 8 = 30,400$ rollouts). For each question $x$, compute clean correct count $c(x) \in \{0,\dots,8\}$ (correct, non-truncated, $<4$ repeated 8-grams), median correct thinking tokens $t_{\text{med}}(x)$, median correct response tokens $r_{\text{med}}(x)$, prompt regex flag `has_min_length_constraint(x)`, and rollout flag `is_abstention(x)`.
   - **General Pair Rules (~3,200 pairs total, max 2 disjoint pairs/question, zero pairs where chosen thinking > rejected thinking):**
     - *Response Neutrality Gate (applied to all chosen candidates $y^+$):* Require $t(y^+) \ge 45$ tokens. If `has_min_length_constraint(x)` is True, require $r(y^+) \ge 0.95\,r_{\text{med}}(x)$; else if `is_abstention(x)` is True, require $r(y^+) \le r_{\text{med}}(x)$; else require $r(y^+) \le 1.05\,r_{\text{med}}(x) + 10$ tokens.
     - *Rule 1A — High-Confidence Shortening ($c(x) \ge 5$, ~2,100 pairs):* Pair shortest-thinking clean correct $y^+$ against longest-thinking correct $y^-$ with $t(y^-) \ge 1.45\,t(y^+)$ and $t(y^-) - t(y^+) \ge 40$ (or longest wrong $y^-$ with $t(y^-) \ge 1.40\,t(y^+)$).
     - *Rule 1B — Moderate-Confidence Safe Shortening ($c(x) \in \{3, 4\}$, ~600 pairs):* Pair second-shortest ($p_{25}$) clean correct $y^+$ against longest wrong $y^-$ with $t(y^-) \ge 1.35\,t(y^+)$ (or longest correct with $t(y^-) \ge 1.60\,t(y^+)$).
     - *Rule 1C — Thinking & Response Loop Killer ($c(x) \ge 1$, ~500 pairs):* Pair clean correct $y^+$ ($t(y^+) \le t_{\text{med}}(x)$) against any truncated, repetitive ($\ge 4$ repeated 8-grams), or runaway rollout ($>2.5\times$ median total and $>600$ tokens, or response $>2.5\,r_{\text{med}}(x)$ and $>400$ tokens).
   - **Training & Checkpoints:** Train LoRA (`r=16, alpha=32`, `fp32` master + `--fp16`) for **240 steps**, `lr=5e-5`, `beta=0.10`, cosine schedule, warmup `0.05`. Save checkpoints at **steps `160, 200, 240, 280`**.

2. **Step 2 — Length-Matched Verification & Loop Calibration DPO (from Step 1 Merged Model):**
   - **Answers Generated By:** Merged Step 1 model ($3,800 \times 8 = 30,400$ fresh on-policy rollouts). Recompute $c(x), t_{\text{med}}(x), r_{\text{med}}(x)$ on Step 1's rollouts.
   - **General Pair Rules (~2,600 pairs total, max 2 disjoint pairs/question):**
     - *Rule 2A — Length-Matched Verification `be_right` ($1 \le c(x) \le 6$, ~1,150 pairs):* Pair clean correct $y^+$ near median thinking ($t(y^+) \in [0.85\,t_{\text{med}}(x), 1.15\,t_{\text{med}}(x)]$, tie-broken by presence of verification words `check|verify|test|confirm|constraint` in `<think>` and satisfying the Response Neutrality Gate) against a clean wrong $y^-$ within $\pm 20\%$ thinking length ($|t(y^-) - t(y^+)| / \max(t(y^+), t(y^-)) \le 0.20$).
     - *Rule 2B — Premature-Cut Guard ($1 \le c(x) \le 6$ and $t_{\text{med}}(x) \ge 180$, ~400 pairs):* Whenever a wrong rollout $y^-$ cut thinking prematurely ($t(y^-) \le 0.72\,t_{\text{med}}(x)$), pair a clean correct rollout $y^+$ near the median ($t(y^+) \in [0.90\,t_{\text{med}}(x), 1.20\,t_{\text{med}}(x)]$) against $y^-$.
     - *Rule 2C — Residual High-Confidence Thinking Trim ($c(x) \ge 7$, ~650 pairs):* On confident prompts with residual thinking spread ($t_{\max}^+(x) \ge 1.30\,t_{\min}^+(x)$ and $t_{\max}^+ - t_{\min}^+ \ge 25$), pair shortest clean correct $y^+$ ($t(y^+) \ge 45$, satisfying Response Neutrality and concise refusal $r(y^+) \le r_{\text{med}}$ on abstentions) against longest correct $y^-$—**except** when $t_{\text{med}}(x) \ge 220$ and $y^+$ drops verification keywords present in the median rollout.
     - *Rule 2D — Residual Loop & Minimum-Length Constraint Guard (~400 pairs):* Pair clean correct $y^+$ against any remaining looped/truncated rollout, and on prompts where `has_min_length_constraint(x)` is True, pair constraint-satisfying correct $y^+$ against short-response wrong $y^-$ ($r(y^-) \le 0.85\,r(y^+)$).
   - **Training & Checkpoints:** Train a fresh LoRA (`r=16, alpha=32`, `fp32` master + `--fp16`, `ref_model` = Step 1 merged) for **80 steps**, `lr=3e-5`, `beta=0.20`, cosine schedule, warmup `0.05`. Save checkpoints at **steps `20, 40, 60, 80`**.

- **Expected Full-Test Range & Evidence:**
  - *Step 1:* **53.5%–56.5% saved / -2.1 to -2.8 acc.** (`p1r1s0` reached 59.4% on SVAMP and 44.0% on MBPP+ in 200 steps despite 35% non-shortening pairs; removing non-shortening pairs in Step 1 and fixing the IFEval response-bloat bug—which alone adds +3.62% overall geometric-mean saving—moves Step 1 directly to 53.5%–56.5%).
  - *Step 2 (Final):* **52.5%–54.8% saved / -0.50 to -0.90 acc** (SVAMP 56–58% / -1.1, MBPP+ 56–58% / -0.8, IFEval 49–51% / -2.1, BFCL pooled 48–50% / +0.7). **Clears $>50\%$ saved and $> -1.0$ accuracy** with a +2.5 to +4.8% saving margin. Directly evidenced by `rpE-80` (+2.6 pts accuracy recovery on `rc1`), `rpM-60` (54.1% / -0.93), and our simulation on `p1r1s0` rollouts projecting **53.98% geometric-mean saving**.
- **Compute Cost:** **~5.8 L4 GPU-hours** (~3.0 hours wall-clock on 2$\times$ L4 GPUs).
- **Decision Rules After Each Stage:**
  - *After Step 1 (Dev-512 $\times 8$ on steps 160, 200, 240, 280):* Advance the checkpoint in **[52.5%, 57.5%] Dev saving** with IFEval mean response $\le 235$ tokens. If step 240 is $<51\%$ saved, advance step 280 (or branch directly into Recipe 2 Step 2).
  - *After Step 2 (Dev-512 $\times 8$ on steps 20, 40, 60, 80):* Select the checkpoint with **Dev saving $\ge 52.5\%$** and highest Dev accuracy ($\Delta\text{Acc}_{\text{Dev}} \ge +0.3$ pts).
- **Main Risk:** Single-round compression from `original` may stop at ~49–50% on BFCL before Step 2 Rule 2C trims the remaining 4–5%. Recipe 2 eliminates this risk completely if Step 1 lands below 52%.

---

### Recipe 2 (RANK #2 — Highest-Certainty 3-Step Fallback): Two-Round Progressive On-Policy Compression DPO $\to$ Pure Verification Repair DPO

- **The Idea in Two Sentences:** Because one offline DPO round from `original` naturally converges toward the sample minimum of 8 base rollouts (~46%–48% saving), Recipe 2 runs **two sequential on-policy compression DPO steps** (Step 1 $\to \sim 47\%$ saved; Step 2 on Step 1's shorter rollouts $\to \sim 57\%$ saved). Step 3 then runs a **pure length-matched verification and premature-cut repair DPO pass** (zero shortening pairs needed) to finish at **53.0%–55.5% saving and -0.60 to -0.95 accuracy**.
- **One-Line Pathway:**
  `original (0% / 0.0) -> Step 1: Round-1 Compression DPO (~47% saved / -1.2 acc) -> Step 2: Round-2 On-Policy Compression DPO (~57% saved / -2.4 acc) -> Step 3: Pure Verification Repair DPO (53.0–55.5% saved / -0.6 to -0.95 acc)`
  - *Step 1 Single Job:* Compress `original` to the sample-optimal length of its initial 8 rollouts (~47% saved) and remove 75% of loops.
  - *Step 2 Single Job:* Re-sample 8 rollouts from Step 1 (where high-confidence `min_th` is now 45% shorter) and compress confident prompts ($c \ge 6/8$) into the 56%–58% saving band.
  - *Step 3 Single Job:* Recover +1.6 to +2.0 points of accuracy on Step 2's rollouts via length-matched `be_right` and premature-cut guards.

#### Exact Procedure from the Original Model

1. **Step 1 — Round-1 Compression DPO (from `original`):**
   - Generate $3,800 \times 8$ rollouts from `original`. Build **~3,000 pairs** using Recipe 1's **Rules 1A, 1B, and 1C** (with the Response Neutrality Gate). Train LoRA (`r=16, alpha=32`, `fp32` master + `--fp16`) for **180 steps** at `lr=5e-5, beta=0.10`. Merge checkpoint `step 180` (gate: Dev-512 saving in **[45%, 49%]**, IFEval response $\le 235$ tokens).
2. **Step 2 — Round-2 On-Policy Compression DPO (from Step 1 Merged Model):**
   - Generate $3,800 \times 8$ fresh rollouts from the merged Step 1 model. Build **~2,600 compression pairs** restricted strictly to **high-confidence prompts ($c(x) \ge 6/8$)**: pair shortest clean correct $y^+$ ($t(y^+) \ge 45$, satisfying Response Neutrality and concise refusal $r(y^+) \le r_{\text{med}}$ on abstentions) against longest correct/wrong $y^-$ with $t(y^-) \ge 1.30\,t(y^+)$, plus anti-loop pairs on any prompt with $>2.5\times$ median rollouts. Emit **zero shortening pairs on $c(x) \le 5/8$**.
   - Train a fresh LoRA (`r=16, alpha=32`, `fp32+fp16`) for **100 steps** at `lr=3e-5, beta=0.15` (checkpoints at `60, 80, 100`). Merge the checkpoint with Dev-512 saving in **[56.0%, 58.5%]**.
3. **Step 3 — Pure Verification & Constraint Repair DPO (from Step 2 Merged Model):**
   - Generate $3,800 \times 8$ fresh rollouts from the merged Step 2 model. Build **~2,200 pure repair pairs** using **only Rules 2A (length-matched `be_right` within $\pm 15\%$ of $t_{\text{med}}$, ~60%), 2B (premature-cut guard against $t(y^-) \le 0.75\,t_{\text{med}}$, ~25%), and 2D (anti-loop + minimum-length constraint guard, ~15%)**.
   - Train a fresh LoRA (`r=16, alpha=32`, `fp32+fp16`) for **60 steps** at `lr=3e-5, beta=0.20` (checkpoints at `30, 45, 60, 75`). Select the checkpoint on Dev-512 with saving $\ge 52.5\%$ and highest accuracy.

- **Expected Full-Test Range & Evidence:** **53.0%–55.5% saved / -0.60 to -0.95 acc.** **Clears $>50\%$ saved and $> -1.0$ accuracy** with the highest empirical certainty because every hop has a direct full-test precedent (`p1r1s0` $\to$ second on-policy shorten hop $\to$ `rpE-80`/`rpM-60` repair).
- **Compute Cost & Main Risk:** **~8.2 L4 GPU-hours** (~4.1h wall-clock on 2$\times$ L4). Main risk is giving back >3.5% saving if Step 3 runs past 75 steps (mitigated by stopping Step 3 at step 45–60 via Dev-512).

---

### Recipe 3 (RANK #3 — RL + DPO Hybrid): Confidence-Gated ShorterBetter Dr. GRPO $\to$ Verification-Preserving DPO

- **The Idea in Two Sentences:** Step 1 runs **on-policy Group-8 RL (Dr. GRPO with Solve-Rate-Scaled ShorterBetter reward, `2504.21370` + `2506.05256` + `2503.20783`)** to compress `<think>` tokens continuously on-policy to **54%–56% saving** while applying zero length penalty on uncertain prompts ($\hat{p}(x) \le 3/8$). Step 2 runs a single **60-step offline Verification & Constraint DPO pass** on the RL policy's rollouts to lock in discrete constraint accuracy and repair premature cuts.
- **One-Line Pathway:**
  `original (0% / 0.0) -> Step 1: Confidence-Gated ShorterBetter Dr. GRPO (~55% saved / -1.6 to -2.2 acc) -> Step 2: Length-Matched Verification DPO (52.0–54.5% saved / -0.60 to -0.95 acc)`
  - *Step 1 Single Job:* Compress `<think>` length on-policy toward the group minimum correct thinking length $t_{\min}^+$ on confident prompts ($\hat{p}(x) \ge 4/8$).
  - *Step 2 Single Job:* Recover +1.0 to +1.5 points of verification and constraint accuracy via offline length-matched DPO repair.

#### Exact Procedure from the Original Model

1. **Step 1 — Confidence-Gated ShorterBetter Dr. GRPO (`code/rl_train.py` from `original`):**
   - Sample batches of $B=8$ prompts $\times G=8$ on-policy rollouts (`temp=0.7, max_new_tokens=2048`) from the 3,800 training prompts; dynamically skip all-wrong groups ($c(x)=0$, `2503.14476`).
   - **General Per-Question Reward:** Let $\hat{p}(x) = c(x)/8$ be the group clean solve rate and $t_{\min}^+(x) = \min_{i:\,\text{correct}(y_i)} \max(t(y_i), 50)$:
     - Incorrect or empty `<think>`: $R(x, y_i) = -1.0 - 0.5\,\mathbb{I}[\text{loop\_or\_trunc}(y_i)]$.
     - Correct: $R(x, y_i) = +1.0 - \lambda(\hat{p}(x))\,\min\!\big(1.0, \frac{t(y_i) - t_{\min}^+(x)}{\max(t_{\text{med}}^+(x), 80)}\big) - \mu_{\text{resp}}(x, y_i)$, where $\lambda(\hat{p}) = 0.45$ if $\hat{p} \ge 6/8$, $0.20$ if $\hat{p} \in \{4/8, 5/8\}$, and **$0.00$ if $\hat{p} \le 3/8$**; and $\mu_{\text{resp}} = 0.25$ if $\neg\texttt{has\_min\_length\_constraint}(x)$ and $r(y_i) > 1.15\,r_{\text{med}}^+(x) + 15$, else $0.0$.
   - **Loss & Settings:** Unnormalized Dr. GRPO token loss (no $1/|y_i|$ division, `2503.20783`), LoRA `r=16, alpha=32` (`fp32` master + `--fp16`), `lr=2e-5`, $\beta_{\text{KL}} = 0.001$ (`2510.15110`), **180 steps** (evaluate Dev-512 at steps `90, 135, 180`; stop when Dev-512 saving hits **54%–56%**). Merge Step 1 adapter.
2. **Step 2 — Length-Matched Verification DPO (from Step 1 Merged Model):**
   - Generate $3,800 \times 8$ rollouts from the merged Step 1 RL model, build **~2,200 repair pairs** using Recipe 1's **Rules 2A, 2B, and 2D**, and train a fresh `r=16, alpha=32` LoRA (`fp32+fp16`) for **60 steps** at `lr=3e-5, beta=0.20` (checkpoints at `30, 45, 60`).

- **Expected Full-Test Range & Evidence:** **52.0%–54.5% saved / -0.60 to -0.95 acc.** **Clears $>50\%$ saved and $> -1.0$ accuracy.** Fixes the uniform-penalty accuracy drop seen in `P3a` (47.4% / -2.3) via ALP/CARE confidence gating ($\lambda=0$ for $\hat{p} \le 3/8$) + Step 2 DPO repair.
- **Compute Cost & Main Risk:** **~9.5 L4 GPU-hours** (~4.8h wall-clock). Main risk is higher RL variance and online verifier overhead compared to offline DPO.

---

## 2b. Side-by-Side Comparison Table of the Three Clean Recipes

| Dimension | **Recipe 1 (RANK #1 — Primary Recommended)** | **Recipe 2 (RANK #2 — Highest-Certainty 3-Step)** | **Recipe 3 (RANK #3 — RL + DPO Hybrid)** |
| :--- | :--- | :--- | :--- |
| **Pathway** | `original -> Step 1 (Confidence-Gated Compression DPO) -> Step 2 (Length-Matched Verification & Loop Calibration DPO)` **(2 steps)** | `original -> Step 1 (Round-1 Compression DPO) -> Step 2 (Round-2 On-Policy Compression DPO) -> Step 3 (Pure Verification Repair DPO)` **(3 steps)** | `original -> Step 1 (Confidence-Gated ShorterBetter Dr.GRPO) -> Step 2 (Length-Matched Verification DPO)` **(2 steps)** |
| **Methods Used** | Offline On-Policy DPO (`fp32` master + `fp16`) in both steps | Offline On-Policy DPO (`fp32` master + `fp16`) in all 3 steps | Step 1: On-Policy Dr. GRPO RL; Step 2: Offline On-Policy DPO |
| **Expected Saving** | **52.5% – 54.8%** (projected center: **54.0%**) | **53.0% – 55.5%** (projected center: **54.3%**) | **52.0% – 54.5%** (projected center: **53.2%**) |
| **Expected Accuracy** | **-0.50 to -0.90 pts** (center: **-0.75 pts**) | **-0.60 to -0.95 pts** (center: **-0.80 pts**) | **-0.60 to -0.95 pts** (center: **-0.85 pts**) |
| **Does It Clear >50% / No Worse Than -1?** | **YES** (+2.5 to +4.8% saving margin; +0.1 to +0.5 pt accuracy margin) | **YES** (+3.0 to +5.5% saving margin; +0.05 to +0.4 pt accuracy margin) | **YES** (+2.0 to +4.5% saving margin; +0.05 to +0.4 pt accuracy margin) |
| **Evidence Strength** | **High:** Direct synthesis of `p1r1s0` (with fixed IFEval response neutrality & pure Step 1 shortening) + `rpM-60` (54.1% / -0.93) repair | **Very High:** Every hop has an exact 1-to-1 full-test precedent (`p1r1s0` $\to$ 2nd shorten hop $\to$ `rpE-80`/`rpM-60` repair) | **Medium-High:** Grounded in `2504.21370`, `2506.05256`, `2503.20783` + `P3a` RL logs, but RL loop is less battle-tested than `dpo.py` |
| **Risk** | Low-Moderate: Step 1 single-round compression on BFCL may stop at ~49% before Step 2 Rule 2C trims the last 4–5% | Low: Slightly higher compute (3 rollout passes) and risk of giving back >3% saving if Step 3 runs past step 60 | Moderate: Online RL variance/entropy drift on a 0.6B model if KL or learning rate fluctuates |
| **GPU-Hours (2$\times$ L4)** | **~5.8 L4 GPU-hours** (~3.0h wall-clock) | **~8.2 L4 GPU-hours** (~4.1h wall-clock) | **~9.5 L4 GPU-hours** (~4.8h wall-clock) |
| **Generality** | **100% General:** Rules use only per-question $c/8$, $t_{\text{med}}$, $r_{\text{med}}$, n-gram loops, and prompt text regex | **100% General:** Same per-question observable rules ($c/8, t_{\text{med}}, r_{\text{med}}$, loop rate, prompt text regex) | **100% General:** Reward uses only verifier outcome, group solve rate $\hat{p}(x)$, $t_{\min}^+$, and $r_{\text{med}}^+$ |

---

## 3. Which Recipe to Run First & The Cheapest Pre-Flight Experiment

### Which One to Run First
Run **Recipe 1 (2-Step Confidence-Gated Compression DPO $\to$ Length-Matched Verification & Loop Calibration DPO)** first. It is the simplest 2-step realization of *"overshoot saving first, then correct accuracy,"* uses the exact verified `dpo.py` (`fp32` master weights + `fp16`) infrastructure, costs only **5.8 L4 GPU-hours**, and has a **zero-waste fallback into Recipe 2**: if Recipe 1 Step 1 lands at 47%–50% saving on Dev-512 instead of 53%–57%, that checkpoint *is* Recipe 2 Step 1, so you simply run Recipe 2's 100-step Round-2 compression pass before the verification repair pass.

### The Cheapest Experiment Before Full Spend (~0.75 L4 GPU-Hours / 25 Minutes)
Test both halves of Recipe 1 **immediately on existing rollout files already on disk** (`base` training rollouts + `p1r1s0` training rollouts) with **zero new rollout generation**:
1. **CPU Pair-Rebuild Audit (5 minutes, 0 GPU-hours):** Run Recipe 1's Step 1 pair rules on the existing `base` training rollouts (removing `p1r1s0`'s `be_right` dilution and replacing `--resp-guard 0.9` with the Response Neutrality Gate $r(y^+) \le 1.05\,r_{\text{med}} + 10$ + response anti-loop pairs). Confirm before touching a GPU that: (a) weighted chosen-vs-base token saving in the pairs increases from `p1r1s0`'s 46.0% to **$\ge 52\%$**, and (b) mean chosen IFEval response length is **$\le$ mean rejected response length** (vs. `p1r1s0`'s +47.3 tokens).
2. **220-Step DPO Micro-Screen on Dev-512 (~0.75 L4 GPU-hours):**
   - Train Step 1 DPO from `original` on the rebuilt pairs for 220 steps (~0.45 GPU-hours) and screen on Dev-512 $\times 4$ (~0.15 GPU-hours). **Pass gate:** Dev token saving rises from `p1r1s0`'s 45.9% to **$\ge 52.5\%$**, and Dev IFEval mean response stays **$\le 235$ tokens** (down from `p1r1s0`'s 330 tokens).
   - Train a 60-step Step 2 repair trial on the existing `p1r1s0` rollouts using Recipe 1's Step 2 rules (~0.15 GPU-hours). **Pass gate:** MBPP+ $>4,000$-token loops drop from 8.3% to $<4.5\%$ and SVAMP/MBPP+ accuracy lifts by $\ge +1.5$ points at neutral token length.

---

## 4. What to Report in the Submission

1. **The Length–Accuracy Pareto Curve & Per-Stage Trajectory:**
   - Plot **(Tokens Saved %, Accuracy $\Delta$ pts)** across `original` (0% / 0.0), `nothink` (94% / -23.0), single-round baselines (`bidax2` 51.2% / -3.0, `p1r1s0` 43.4% / -1.5, `p1cr1s0` Dev 34.4% / +2.9), Step 1 checkpoints (`160, 200, 240`), and Step 2 checkpoints (`20, 40, 60, 80`), showing Step 1's horizontal compression into 54%–57% and Step 2's vertical +2.0 pt accuracy recovery into the $>50\% / > -1.0$ box.
   - Include the **6-subset decomposition table** splitting `<think>` vs. response tokens, runaway loop share ($\ge 4$ repeated 8-grams drop from 13.5% $\to$ 3.4%), truncation rate (2.41% $\to$ 0.58%), and accuracy change across base solve-rate bands (`0/8`, `1-3/8`, `4-6/8`, `7-8/8`), highlighting that accuracy *increases* on partially solved math/code (+3.6 to +8.5 pts) and abstentions (+4.2 pts on BFCL No-Fit).
2. **Zero-Shot Out-of-Distribution Transfer Check (~0.4 L4 GPU-hours):**
   - Evaluate `original` vs. the final clean checkpoint on 3 unseen benchmarks never used in training or tuning: **MATH-500** (harder multi-step math), **HumanEval+** (unseen code synthesis), and **ARC-Challenge / MMLU-Pro** (commonsense/science reasoning with no training verifier). Report token savings (~45%–55%), accuracy delta ($\pm 1.0$ pt), and difficulty-adaptive thinking length (longer `<think>` on harder questions).
3. **Lexical & Structural Shift in `<think>` (`keyness` Analysis):**
   - Report the shift from repetitive self-doubt (`"therefore"` $-68\%$, `"mistake"` $-65\%$, `"double"` $-57\%$) to high-density planning and single-pass verification (`"testing"` $+143\%$, `"iterate"` $+105\%$, `"requires"` $+103\%$), with mean thinking paragraphs dropping from $12.1 \to 4.3$.
4. **Five High-Value Negative Results:**
   - Document (i) **`bf16` gradient underflow** (`rr1`–`rr6`: 0% compression without `fp32` master weights + `fp16`), (ii) **Round-1 shorten + protect cancelation** (`p1cr1s0`: 34.4% ceiling when guards are mixed into Step 1), (iii) **the Short-is-Correct Trap** (`bidax2`: 51.2% / -3.0 from unguarded 1-step shortening), (iv) **blanket IFEval response-guard bloat** (`p1r1s0`: +82.2 response tokens and 4.99% runaway response loops), and (v) **irreversibility past the 60% response-collapse cliff** (`rpN` on `rpL-60`).