```python
# Let's append Section 6 (the 3 ranked Winning Paths with full designs, evidence, hyperparameters, GPU-hours, expected ranges, and cheap confirm/kill tests) to both work/notes.md and work/report_draft.md!

sec6_text = """
---

## 6. Ranked Winning Strategy Paths to Reach >50% Saved at Better Than −1.0 Point (One Reproducible Recipe, Within Constraints)

Below are the **three highest-probability strategy paths** that strictly obey all team constraints (trained exclusively on Qwen3-0.6B's own answers; no teacher-written answers or coach; no test-time prompt or router; no copied competitor recipe or dataset; one reproducible training pipeline; two Colab L4 GPUs; zero use of the test set or 720 screen for any decision).

### Shared Non-Benchmark Evaluation Protocol for All Three Paths (4 GPU-hours once)
- **Dev-512 (for checkpoint/strength selection) & Confirm-512 (for pre-test lock):** Drawn from `own_repair_questions()` (`pool-dp13` minus MetaMath + `gsm8k_std` + `gsm_ic` + `gsm_plus` + `kodcode` + `ifeval_default` matched + `toolace_clean` + `nofit1/2`), 8-gram and 13-word decontaminated against all 2,759 test questions. Each set contains **140 SVAMP-worded math, 100 KodCode (MBPP-like slice ≤70 prompt / ≤130 solution tokens), 132 IFEval-matched (≥50% counting rules), and 140 BFCL-clean tools (60 simple, 35 multiple, 45 verified `no-fit`)**.
- **Rule:** Every candidate is scored with 8 samples (`temp 0.6, top_p 0.95, top_k 20`, code cap 8,192, others 4,096) on **Dev-512** using the official 4-benchmark weighting (`BFCL` pooled `400/200/240`). To account for ~0.8–1.0% finite-sample variance, a candidate must hit **≥ 51.0% saved and ≥ −0.70 pp** on Dev-512, and confirm at **≥ 50.5% saved and ≥ −0.85 pp** on Confirm-512 before its **2-seed weight average (`merge.py` 0.5/0.5)** is evaluated once on the full 2,759 × 8 test.

---

### RANK 1 (Recommended Primary Path — Estimated Win Probability: 55–65%):
**Two-Round On-Policy "Compact-and-Verify" DPO (`OPCV-DPO`)**

#### 1. Why It Can Win (Quantitative Evidence from Your Log)
- **What single-round DPO gets right and wrong:** Single-round offline DPO from raw Qwen reliably reaches 50–54% saving, but always lands at −2.5 to −3.5 pp (`bidax2-full`: 51.2% / −3.00 pp; `l2s50e-final`: 53.5% / −2.81 pp; `push2-final`: 52.4% / −3.00 pp). Your `st-ca1-soup-10` and `ir-full` diagnostics proved why: when thinking is compressed by ~55%, (a) 68% of newly wrong math/code answers are normal-length shortened traces (~0.43× base) that misread a trap phrase or skip an edge-case check, (b) 30% stop prematurely (`< 0.5x` median right thinking), and (c) IFEval counting-rule responses shrink from 209 to 145–161 tokens, costing −0.65 pp on the headline.
- **What `rpE` and `rpI` proved on the Full Test:** Training on the *shortened model's own mistakes* with equal-length right>wrong pairs that prefer **question re-reading** (`51% vs 41%` math, `77% vs 57%` code, `94% vs 82%` no-fit) + **IFEval `--count-rules`** + **`no-fit` refusals** is the **only mechanism in 205 runs** that moved Qwen3-0.6B +1.49 to +1.64 pp above the trade-off curve on the full 2,759 × 8 test (`rpe-40full`: **50.6% / −1.90 pp**; `rpe-60full`: **47.8% / −1.20 pp**; `rpe-80full`: **47.0% / −0.90 pp**; `rpi-120full`: **41.9% / +0.10 pp**).
- **Exact Counterfactual Proof:** `rpe-40full` already hit **50.6% saved / −1.90 pp** on the full test *before* `--count-rules` was added (so IFEval was −2.8 pp) and with weak `no-fit` (−0.4 pp). In `rpi-120full`, adding `--count-rules` (320 IFEval pairs incl. 152 counting-rule) and fresh rollouts moved full-test IFEval to **−1.5 pp** (+0.325 pp on headline) and `no-fit` to **+5.8 pp** (+0.443 pp on headline). Substituting just those two measured improvements into `rpe-40full` yields **50.6% saved / −1.13 pp** (`rpe-60full` becomes **−0.51 pp**). And why did `rpE` lose saving from step 40 (50.6%) to step 80 (47.0%)? Because `rpE` had **0% shorten pairs**. Keeping **30% Shorten Anchors** in Round 2 prevents length drift while the 70% Equal-Length Re-Read / `--count-rules` / Capped-Guard pairs repair SVAMP and MBPP+ to −1.0 pp.

#### 2. Full Reproducible Recipe Design
- **Data Pool (~14,000 clean own questions, zero competitor sources):**
  - **Math (4,500):** `gsm8k_std` (SVAMP wording) + `gsm_ic` + `gsm_plus`.
  - **Code (3,000):** `kodcode` filtered to MBPP-like length (`prompt <= 70`, `solution <= 130` tokens).
  - **IFEval (3,000):** `ifeval_default` matched to benchmark 1/2/3-constraint shares, oversampled so **≥ 50% (1,500+) contain counting/length constraints** (`number_words`, `keywords:frequency`, `number_sentences`, `highlighted_sections`, `placeholders`).
  - **Tools (3,500):** `toolace_clean` (single-turn, BFCL schema types, explicit argument values) + `nofit1/2` (verified irrelevance), mixed at **35% simple, 25% multiple, 40% `no-fit`**.
- **Round 1 (Offline Compaction + Guard from Raw Qwen, 8,000 questions × 8 own answers → 3,200 pairs = 200 steps × 16):**
  - **Question filter:** Drop `0/8` (never-solved). Draw 50% reliably solved (`6/8–8/8`) and 50% sometimes solved (`2/8–5/8`).
  - **Quotas (800 pairs per benchmark group):**
    - *Math (800) & `No-Fit` (320):* **35% Shorten** (solved `>= 6/8`: shortest right vs longest right with thinking ratio `>= 1.40x`) + **50% Equal-Length Re-Read Right** (solved `2/8–6/8`: right vs finished wrong within **±20% thinking length**, chosen has higher question-word/number overlap in second half of thinking, group median chosen length ≤ median rejected length) + **15% Capped Anti-Early-Stop Guard** (right with thinking `<= 1.15x` median right vs wrong with thinking `< 0.50x` median right).
    - *Code (800), Simple (280), Multiple (200):* **50% Shorten** (`>= 6/8`, ratio `>= 1.40x`) + **35% Equal-Length Re-Read Right** (±20% length) + **15% Capped Guard**.
    - *IFEval (800 = 400 counting-rule + 400 non-counting):* On counting-rule prompts (400): **100% `--count-rules` Right>Wrong** (chosen response length ≥ rejected response length, chosen thinking `<= 1.20x` rejected thinking and `>= 80` tokens; **0% shorten**). On non-counting prompts (400): **55% Shorten** (thinking ratio `>= 1.40x`, final response length within ±15%) + **45% Equal-Length Right**.
  - **Hyperparameters (Round 1):** Base = `Qwen/Qwen3-0.6B`, `LoRA r16 alpha 32` (all linear projections), `fp32 master weights + fp16 autocast`, `beta = 0.1`, `NLL = 0.0`, batch size `16`, `lr = 1.5e-5` (10-step linear warmup, cosine decay to `5e-6`), `200 steps` (1 epoch), `2 seeds (s0, s1)`. Merge each seed at step 200 (`R1-s0`, `R1-s1`).
- **Round 2 (On-Policy Calibration + Length Anchor from `R1`, 6,000 unseen own questions × 8 answered by `R1` → 1,280 pairs = 80 steps × 16):**
  - **Why on-policy:** `R1` answers the 6,000 unseen questions × 8 itself (~50% faster than raw Qwen because traces are already ~45% shorter).
  - **Quotas (320 pairs per benchmark group = 1,280 total):** SVAMP-math 320, KodCode 320, IFEval 320 (≥160 counting-rule), Tools 320 (100 simple, 80 multiple, **140 `no-fit`**).
  - **Pair mix:** **30% Shorten Anchor** (`>= 6/8`, thinking ratio `>= 1.30x`, never on IFEval counting rules—locks saving at 51–53%) + **55% Equal-Length Re-Read / `--count-rules` Right** (±20% thinking length, median chosen ≤ median rejected) + **15% Capped Anti-Early-Stop Guard** (right `<= 1.15x` median right vs early-stop wrong `< 0.50x` median right).
  - **Hyperparameters (Round 2):** Init & Reference = `R1-s0` (for seed 0) and `R1-s1` (for seed 1), fresh `LoRA r16 alpha 32`, `fp32 weights + fp16 autocast`, `beta = 0.1`, `NLL = 0.0`, batch size `16`, `lr = 1.0e-5` constant (2 warmup steps), `80 steps` (save at 40, 60, 80).
- **Checkpoint Selection (Zero Test Leakage):** Score the 2-seed weight-averaged soups (`merge.py` 0.5/0.5) at steps `40, 60, 80` on **Dev-512 × 8**; pick the single checkpoint with highest accuracy among those with **Dev-512 saved ≥ 51.0%**, verify once on **Confirm-512 × 8**, then run **Full Test 2,759 × 8** once.
- **Compute Cost:** ~42 GPU-hours total (~12 GPU-h reusing/topping up Round 1 answers + 4 GPU-h Round 1 training + 16 GPU-h Round 2 ×8 on-policy rollouts + 2 GPU-h Round 2 training + 8 GPU-h Dev/Confirm/Full evaluations = **< 1 day on 2 L4 GPUs**).
- **Expected Full-Test Result:** **51.0%–53.5% saved / −0.45 to −0.85 pp accuracy** (SVAMP 56–58% / −1.0 to −1.5 pp; MBPP+ 50–53% / −0.5 to 0.0 pp; IFEval 44–46% / −1.2 to −1.6 pp; BFCL 41–43% / +0.8 to +1.4 pp with `no-fit` +4.0 to +5.5 pp).
- **First Cheap Experiment to Confirm or Kill (~3.5 GPU-hours):**
  Before running Round 1 from scratch, test the Round 2 pair recipe (`rpJ`: 30% Shorten Anchor + 55% Equal-Length Re-Read/`--count-rules` + 15% Capped Guard, 1,280 pairs, `lr = 1e-5`, 60 steps) on your **already-generated `rc1-soup-30` × 8 rollouts from `rpH`** (or `rpE-80`'s × 8 rollouts from `rpI`), evaluated on **Dev-512 × 8**.
  - **Confirm rule:** If adding the 30% Shorten Anchor + `--count-rules` holds saving within ±1.5% of the start model (≥51.5% from `rc1`, or pushes `rpE-80` from 47% to ≥51.0%) while keeping accuracy ≥ −0.80 pp on Dev-512, **Path 1 is confirmed—launch full clean Round 1 + Round 2 immediately.**
  - **Kill rule:** If Dev-512 saving still drops below 48.5% or accuracy falls below −1.6 pp, switch to Path 2.

---

### RANK 2 (Best Modular Control Over Rank-1 Spill-Over — Estimated Win Probability: 35–45%):
**Asymmetric 4-Domain Specialist LoRA Task-Arithmetic Merge + On-Policy Verification (`AsymMerge + Verify`)**

#### 1. Why It Can Win (Quantitative Evidence from Your Log)
- **Why per-type merging works when calibrated:** In your 5 Nov additivity check (L4635), merging 4 specialist adapters (`pt-mid`: Math 0.30, Code 0.45, IFEval 0.60, Tools 0.45) multiplied token ratios almost exactly (55.0% measured vs 56.5% predicted) while **cutting accuracy loss in half** (−2.6 pp measured vs −5.8 pp sum of solos) because each adapter's own-domain right>wrong pairs buffer the others' rank-1 brevity spill-over.
- **The `lpt6mix-Cnf` & `rpC` breakthrough:** On 6 Nov (L4853), `lpt6mix-Cnf` (Instructions 0.50 + Simple 0.50 + Multiple 0.50 + weak No-Fit 0.25, **Math shortening = 0**) achieved **42% SVAMP saving at −1.3 pp** and **48% MBPP+ saving at −0.4 pp** purely via cross-domain rank-1 spill-over without hurting hard math! And when `pt-mid` was given an equal-length correctness pass (`rpB-20`: 48.7% / −1.20 pp; `rpC-20`: 50.5% / −1.83 pp; `rpC-80`: 43.4% / +0.43 pp, +2.19 vs curve), Code jumped to **+2.3 pp** and `no-fit` to **+5.4 pp**.
- **What held October's `pt-mid` back:** It was trained in `bfloat16` (which `bidax6p` proved loses ~2.8 pp accuracy vs `fp32+fp16`), used `NLL = 1.0` (which imitates noisy tokens), lacked `--count-rules` on IFEval, and lacked `no-fit` top-ups (`pt-mid-full` lost −5.6 pp on `no-fit` vs +3.4 pp in `rpe-80full`!). Fixing just `no-fit` (−5.6 → +4.0 pp) and `fp32` lifts `pt-mid-full` (53.6% saved / −3.30 pp) by >2.2 pp before the verification pass even starts.

#### 2. Full Reproducible Recipe Design
- **Stage 1: Train 4 Specialist LoRA Adapters from Raw Qwen on Clean Own Data (960 pairs = 60 steps × 16 per adapter):**
  - **Math Adapter (`A_math`, 960 pairs from `gsm8k_std` + `gsm_ic` + `gsm_plus`):** Gentle shortening + heavy trap protection: **25% Shorten** (`>= 7/8` solved, ratio `>= 1.40x`) + **60% Equal-Length Re-Read Right** (`2/8–6/8`, ±20% length, re-read filter) + **15% Capped Guard**.
  - **Code Adapter (`A_code`, 960 pairs from `kodcode` MBPP-like slice):** **55% Shorten** (`>= 5/8`, ratio `>= 1.40x`) + **30% Equal-Length Right** + **15% Capped Guard**.
  - **IFEval Adapter (`A_ifeval`, 960 pairs from `ifeval_default` matched):** 480 counting-rule pairs (**100% `--count-rules` Right>Wrong, 0% shorten**) + 480 non-counting pairs (**65% Shorten** on thinking only + **35% Equal-Length Right**).
  - **Pooled Tools Adapter (`A_tools`, 960 pairs: 320 simple, 240 multiple, 400 `no-fit`):** On simple/multiple: **55% Shorten + 30% Right + 15% Guard**; on `no-fit`: **25% Shorten + 75% Equal-Length Right** (teaches confident refusal without over-refusing valid calls, as proven by `ptsolo-tools` vs `lpt6-nofit`).
  - **Hyperparameters per adapter:** `LoRA r16 alpha 32`, `fp32 weights + fp16 autocast`, `beta = 0.1`, `NLL = 0.0`, `lr = 2.0e-5` (6-step warmup + cosine to `2e-6`), `60 steps` (1 pass), `2 seeds` averaged per adapter.
- **Stage 2: Pre-Declared 3-Point Task-Arithmetic Merge Grid on Dev-512 × 8 + 40-Step On-Policy Verification:**
  - Evaluate only 3 pre-declared merge vectors `(w_math, w_code, w_ifeval, w_tools)` on **Dev-512 × 8**:
    - `M_low = (0.25, 0.40, 0.55, 0.45)` (targets ~50% saved)
    - `M_mid = (0.30, 0.45, 0.60, 0.50)` (targets ~53% saved)
    - `M_high = (0.25, 0.50, 0.65, 0.55)` (targets ~55% saved, low math dose)
  - Pick the single merge `M*` with **52.0%–55.0% saved** on Dev-512, have `M*` answer 4,000 fresh own questions × 8, and run **40 steps** (`lr = 1e-5` constant, `fp32+fp16`, `beta = 0.1`) on **640 pairs** (**25% Shorten Anchor + 60% Equal-Length Re-Read/`--count-rules` Right + 15% Capped Guard**).
- **Compute Cost:** ~34 GPU-hours total (4 adapters × 2 seeds × 60 steps = ~3 GPU-h training; 3 merge screens on Dev-512 = ~1.5 GPU-h; 4,000 × 8 rollouts from `M*` = ~11 GPU-h; 40-step verification × 2 seeds = ~1 GPU-h; Dev/Confirm/Full tests = ~8 GPU-h; plus ~10 GPU-h if any domain needs fresh raw rollouts).
- **Expected Full-Test Result:** **50.5%–53.0% saved / −0.60 to −1.05 pp accuracy**.
- **First Cheap Experiment to Confirm or Kill (~2.5 GPU-hours):**
  Take your **already-trained `lpk` (`fp32+fp16`, new data) step-30 domain adapters** (`lpk-math-s30`, `lpk-code-s30`, `lpk-instr-s30`, `lpk-tools-s30`), which were never tuned for merge weights on 6 Nov (L5150), evaluate the 3 merge vectors `M_low`, `M_mid`, `M_high` on **Dev-512 × 8**.
  - **Confirm rule:** If one of the 3 merges hits **≥ 51.5% saved at ≥ −1.8 pp** on Dev-512 *before* the 40-step verification pass (beating `pt-mid`'s −2.6 pp), proceed with the 40-step on-policy verification pass.
  - **Kill rule:** If all 3 merges land worse than −2.3 pp at 51% saved on Dev-512, drop Path 2 and focus 100% on Path 1.

---

### RANK 3 (Best Realization of the Team's Budget-Allocation Intuition — Estimated Win Probability: 25–35%):
**Difficulty-Indexed Natural Budget DPO + Short On-Policy Guard (`Natural Budget Ladder`)**

#### 1. Why It Can Win (Quantitative Evidence from Your Log)
- **What `D4 lpd-budget` proved:** In your 12-way DPO pair screen on 5 Nov (L4628), **D4 `lpd-budget`**—which replaced the flat `1.5x` shorten threshold with a **solve-rate-indexed per-question length budget** (so `8/8` easy questions are pushed to very short natural answers, `5–6/8` medium questions to moderate natural answers, and `<4/8` hard questions get no shortening)—was the **#1 best single-pass run in the entire D-series**, scoring **49.0% saved / −1.42 pp (+1.29 vs curve)** with **MBPP+ at 43% saved / −0.2 pp** and **BFCL multiple at +1.7 pp**.
- **Why `D4` beat the bottom-up forced-cut ladder before you even built it:** `D4` achieves the exact mathematical objective of Section 4 (`spend tokens where the marginal accuracy gain is highest, indexed by p0(q)`) using **natural complete rollouts** in **one training pass** anchored to the raw model, completely avoiding forced sentence cuts, 10-rung stopping bias, and 7-stage reference drift!
- **Why `D4` missed by 1.0% saving and 0.42 pp accuracy:** `D4` used only 4 rollouts per question (so the shortest right answer of 4 on `8/8` questions was only at the ~20th percentile, not the ~10th percentile available in 8–10 rollouts), lacked the **question re-read filter** on math (`SVAMP −2.9 pp`), and lacked **`--count-rules`** on IFEval (`IFEval −1.9 pp`).

#### 2. Full Reproducible Recipe Design
- **Data Pool:** 10,000 clean own questions answered × 8 by raw Qwen (reuse stored 8–10 answer clean pool).
- **Natural Budget Level Assignment (Zero Forced Cuts):** For each question `q` with solve rate `p0(q) = k/8` (`k >= 2`), let `L_med(q)` be the median thinking length of its right answers:
  - **Tier 1 — Easy (`k in {7, 8}`):** Target budget = **shortest natural right answer** `c*`, provided `len(r*) >= 1.25 * len(c*)` (where `r*` is the longest right answer). Produces a **Tier-1 Shorten Pair** (`c* > r*`) AND a **Tier-1 Guard Pair** (`c* > longest wrong`, if `k=7`).
  - **Tier 2 — Medium (`k in {5, 6}`):** Target budget = **second-shortest natural right answer** `c_2` (protects against lucky shortcuts on trap math!), paired against `r*` if `len(r*) >= 1.50 * len(c_2)` (**Tier-2 Shorten Pair**), plus an **Equal-Length Re-Read Right Pair** (`right > finished wrong` within ±20% length, chosen re-reads question).
  - **Tier 3 — Hard (`k in {2, 3, 4}`):** **0% Shorten Pairs.** Only **Equal-Length Re-Read Right Pairs** (±20% length, chosen re-reads question) and **Capped Anti-Early-Stop Guard Pairs** (right with `len <= 1.15 * L_med(q)` > wrong with `len < 0.50 * L_med(q)`).
  - **IFEval Counting-Rule Override:** Any IFEval question with a counting/length constraint is placed in Tier 3 (`0% Shorten`; 100% `--count-rules` where chosen response ≥ rejected response and chosen thinking `<= 1.20x` rejected and `>= 80` tokens).
- **Curriculum Ordering (2-Rung Single-Pass Schedule with Fixed Raw Reference, 4,000 pairs = 250 steps × 16):**
  - **Rung A (Steps 1–125):** All Tier-1 + Tier-2 pairs (2,000 pairs, shuffled across all 4 types in equal 25% shares) — establishes the difficulty-indexed compression habit.
  - **Rung B (Steps 126–250):** 60% Tier-3 Equal-Length Re-Read / `--count-rules` / Capped-Guard pairs + 40% replay of Tier-1/Tier-2 Shorten pairs (2,000 pairs) — repairs hard math/code and counting rules while holding the Rung A compression in place under a **single cosine LR schedule (`2e-5 -> 2e-6`) and fixed raw-Qwen reference**.
- **Hyperparameters:** `LoRA r32 alpha 64`, `fp32 weights + fp16 autocast`, `beta = 0.1`, `NLL = 0.0`, batch size `16`, `250 steps`, `2 seeds` averaged in weight space (`merge.py` 0.5/0.5). Checkpoints at steps `175, 215, 250` evaluated on **Dev-512 × 8**.
- **Compute Cost:** ~18 GPU-hours total (0 new rollout hours if reusing the 8–10 answer clean pool; ~5 GPU-h training 2 seeds; ~5 GPU-h Dev-512 + Confirm-512 screens; ~8 GPU-h Full Test).
- **Expected Full-Test Result:** **49.5%–52.0% saved / −0.75 to −1.25 pp accuracy**.
- **First Cheap Experiment to Confirm or Kill (~3 GPU-hours):**
  Build the Tier-1/2/3 natural-budget pair set on your existing clean 8–10 answer pool (zero GPU generation time!), train Seed 0 for 250 steps (~2 GPU-hours), and score step 215 and step 250 on **Dev-512 × 8** (~1 GPU-hour).
  - **Confirm rule:** If Seed 0 hits **≥ 50.5% saved and ≥ −0.90 pp** on Dev-512 (improving on `D4 lpd-budget` by ≥+0.5 pp accuracy and ≥+1.5% saving), train Seed 1 and soup them.
  - **Kill rule:** If single-pass Rung B still leaves SVAMP worse than −2.0 pp at ≥50% saved (because raw Qwen's offline answers lack compact traces on hard math), stop Path 3 and rely on Path 1's on-policy Round 2.

---

### Summary Comparison of the Three Winning Paths

| Rank & Path | Core Mechanism | Why It Beats the 7-Stage Forced-Cut Ladder | Total GPU-Hours | Expected Full-Test Saved / Acc | Win Prob (`>50%` & `>-1.0`) | First Cheap Test (Confirm/Kill) |
|---|---|---|---:|---|---:|---|
| **#1: `OPCV-DPO` (2-Round On-Policy Compact & Verify)** | Round 1 compresses natural traces to ~45%; Round 2 samples ×8 on-policy from Round 1 and trains on **30% Shorten Anchor + 55% Equal-Length Re-Read/`--count-rules` + 15% Capped Guard** (2-seed soup) | Uses natural complete traces; fixes the exact 2 bugs in `rpE`/`rpI` (0% shorten drift & missing `--count-rules` at step 40/60) within one clean 2-round recipe from raw Qwen | **~42 h** | **51.0–53.5% / −0.45 to −0.85 pp** | **55–65%** | **3.5 h:** Run Round-2 mixed pair rule (`rpJ`: 30% shorten + 70% re-read/`--count-rules` guard) on stored `rc1`/`rpE` ×8 rollouts, score on Dev-512 |
| **#2: `AsymMerge + Verify` (4-Domain Task Arithmetic + 40-step On-Policy Pass)** | 4 clean `fp32` domain adapters merged with low math weight `(0.25, 0.45, 0.60, 0.50)` + 40-step on-policy re-read/`--count-rules` pass | Exploits sub-additive accuracy loss (+3.2 pp in `pt-mid`) and rank-1 spill-over (`lpt6mix-Cnf`) while fixing `pt-mid`'s `bf16` and `no-fit` holes | **~34 h** | **50.5–53.0% / −0.60 to −1.05 pp** | **35–45%** | **2.5 h:** Score 3 pre-declared asymmetric merges of existing `lpk` step-30 `fp32` adapters on Dev-512 |
| **#3: `Natural Budget Ladder` (2-Rung Solve-Indexed Single-Pass DPO)** | Maps `p0(q)` tiers (`7-8/8`, `5-6/8`, `2-4/8`) to natural right-answer percentiles in a 2-rung single-pass schedule (`ref = raw Qwen`) | Captures the `A(B)` marginal-gain objective of `D4 lpd-budget` (49.0% / −1.42 pp) using 8-sample natural rollouts, `--count-rules`, and re-read filter—zero forced cuts | **~18 h** | **49.5–52.0% / −0.75 to −1.25 pp** | **25–35%** | **3.0 h:** Build Tier 1/2/3 pairs from stored pool answers, train Seed 0 (250 steps), score on Dev-512 |
"""

with open(draft_path, "a") as f:
    f.write(sec6_text)

with open(draft_path, "r") as f:
    final_check = f.read()

print(f"Updated report_draft.md with Section 6 ({len(final_check)} chars). Ready to output final response.")
```