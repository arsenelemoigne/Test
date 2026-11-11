# Post-Launch Analysis of Path 1 Round 1 (`p1r1s0`) and `rpK`

---

## 1. Path 1’s Round 1 (`p1r1s0`): What Happened and Why It Missed the 53–57% Forecast

### 1.1 What Happened in Numbers (Log Facts)
On `Dev-512` ($\times 8$), against the original `Qwen3-0.6B` and paired against `rpE-80` (`Dev-512`: **46.9% saved / +1.36 acc**; Full test: **47.0% / -0.9**):

| Checkpoint | Pairs Ordered Right (Train) | `Dev-512` Headline Saved [95% CI] | `Dev-512` Think-Only Saved | `Dev-512` Acc Change [95% CI] | Paired vs `rpE-80`: Saved [95% CI] | Paired vs `rpE-80`: Acc [95% CI] | Implied Full-Test Equivalent (Saved / Acc) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`p1r1s0-100`** | 82% | **47.5%** [44.8, 50.1] | 54.2% | **-0.27** [-1.81, +1.36] | **+0.5** [-2.0, +3.3] | **-1.63** [-3.15, -0.18] | ~47.4% / **-2.5** |
| **`p1r1s0-150`** | 84% | **43.7%** [41.1, 46.3] | 51.3% | **-0.28** [-1.92, +1.32] | **-3.2** [-5.3, -1.0] | **-1.65** [-3.17, -0.09] | ~43.6% / **-2.5** |
| **`p1r1s0-200`** | 91% | **45.9%** [43.2, 48.7] | 53.6% | **+0.12** [-1.43, +1.80] | **-1.0** [-3.0, +1.2] | **-1.25** [-2.92, +0.33] | ~45.8% / **-2.1** |

Wait-gate outcome: `p1r2s0` was gated on `p1r1s0-200 >= rpE-80 + 6.0` (= **52.9%** on `Dev-512`). At **45.9%**, the gate properly closed and saved the Round 2 generation and training budget.

Per-part breakdown on `Dev-512` (tokens split into `thinking + response = total`):

| Part ($n$ on Dev-512) | Original (`base`) Think+Resp (Acc) | `rpE-80` Think+Resp (Saved / $\Delta$Acc) | `p1r1s0-100` Think+Resp (Saved / $\Delta$Acc) | `p1r1s0-150` Think+Resp (Saved / $\Delta$Acc) | `p1r1s0-200` Think+Resp (Saved / $\Delta$Acc) | `rc1-soup-30` (Dev-512 Saved / $\Delta$Acc) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **SVAMP** ($n=140$) | `1115 + 20 = 1135` (74.7%) | `564 + 9 = 573` (**49.5%** / +2.1) | `442 + 9 = 452` (**60.2%** / -3.7) | `504 + 9 = 513` (**54.8%** / -2.7) | `470 + 9 = 480` (**57.7%** / -2.4) | **65%** / -4.7 |
| **MBPP+** ($n=100$) | `1934 + 80 = 2014` (61.5%) | `954 + 67 = 1021` (**49.3%** / +3.1) | `941 + 65 = 1006` (**50.1%** / -1.8) | `1004 + 63 = 1067` (**47.0%** / -3.1) | `888 + 67 = 955` (**52.6%** / -3.2) | **67%** / -3.2 |
| **IFEval** ($n=132$) | `436 + 273 = 708` (58.0%) | `146 + 245 = 392` (**44.7%** / -2.8) | `184 + 279 = 463` (**34.7%** / +2.3) | `194 + 312 = 506` (**28.5%** / +0.4) | `197 + 330 = 527` (**25.6%** / +1.8) | **52%** / -1.2 |
| **BFCL Simple** ($n=60$) | `159 + 21 = 180` (89.4%) | `83 + 21 = 104` (**42.5%** / +2.9) | `84 + 21 = 105` (**41.5%** / +0.4) | `86 + 21 = 108` (**40.4%** / +5.6) | `82 + 21 = 104` (**42.5%** / +4.6) | **46%** / +0.8 |
| **BFCL Multiple** ($n=35$) | `186 + 19 = 204` (77.5%) | `92 + 17 = 109` (**46.5%** / +6.1) | `94 + 19 = 113` (**44.8%** / +6.1) | `90 + 18 = 109` (**46.7%** / +7.5) | `90 + 19 = 108` (**46.9%** / +8.2) | **50%** / +7.9 |
| **BFCL No-Fit** ($n=45$) | `155 + 28 = 183` (90.6%) | `90 + 11 = 101` (**44.6%** / +1.4) | `94 + 20 = 114` (**37.9%** / +1.7) | `92 + 21 = 113` (**38.4%** / -0.3) | `92 + 20 = 112` (**38.5%** / +0.6) | **46%** / -6.1 |
| **BFCL Pooled** (400/200/240) | `164 + 22 = 186` | `87 + 17 = 104` (**44.0%** / +3.1) | `89 + 20 = 109` (**41.4%** / +2.2) | `89 + 20 = 109` (**41.5%** / +4.3) | `87 + 20 = 107` (**42.5%** / +4.3) | **47.0%** / +0.5 |

### 1.2 Where the Saving Is Missing and Where Accuracy Went (Log Facts vs Diagnosis)
My `consult8` accuracy forecast (**-2.2 to -3.3** full-test equivalent) was spot on: `p1r1s0` landed at **-0.27 to +0.12** on `Dev-512`, which translates to **-2.1 to -2.5** on the full test (given the +2.2-point `Dev-512` accuracy bias measured across all 5 calibration models), with losses concentrated in SVAMP (-2.4 to -3.7) and MBPP+ (-1.8 to -3.2) and gains in BFCL (+2.2 to +4.3) and IFEval (+0.4 to +2.3).

However, my `consult8` saving forecast (**53–57%**) was wrong by **7 to 9.5 percentage points**. The per-question and pair-table data show two distinct causes—one specifically in IFEval, and one structural across all parts:

1. **The IFEval Collapse (costing -5.6 points of headline geomean saving at Step 200, and -3.9 points at Step 100):**
   - Look at `p1r1s0` vs `rpE-80`: on SVAMP (**57.7%** vs **49.5%**) and MBPP+ (**52.6%** vs **49.3%**), `p1r1s0-200` actually shortened **more** than `rpE-80`. On BFCL thinking tokens (`87` vs `87`), `p1r1s0-200` tied `rpE-80` (losing 1.5 points of BFCL total saving solely because `bfcl_irrelevance` responses averaged `20.1` tokens vs `rpE-80`'s `11.1` tokens, where `rpE-80` often outputs `[]`).
   - The catastrophic gap vs `rpE-80` is **IFEval**: `p1r1s0` saved only **34.7%** at step 100, **28.5%** at step 150, and **25.6%** at step 200 (vs `rpE-80`'s **44.7%** and `rc1-soup-30`'s **52.0%**). Had IFEval alone saved 52.0%, `p1r1s0-200` would be at **51.5% headline saved** (and `p1r1s0-100` at **51.4%**).
   - **Why IFEval failed (verified directly from `path1_round1_pairs_local_build.csv` and `dev512_per_answer.csv`):**
     - **Only 25% of IFEval pairs were `shorten` pairs (`161 / 644`):** Because 50% of the IFEval quota was reserved for `right_count` (`322 / 644`) and the remaining 50% was split 50/50 between `shorten` (`161`) and `right_cr` (`161`), IFEval received a 25% shortening / 75% correctness mix. Furthermore, because `right_count` (`322`) was limited by supply, the proportionality cap cut total IFEval pairs from `800` to `644`.
     - **75.7% of all IFEval pairs pushed the response to be LONGER (`+46.9` response tokens on average):**
       - In `right` pairs (`right_count` + `right_cr`), `chosen_response > rejected_response` in **80.6%** of pairs (`>=` in **82.8%**), with mean `chosen_response = 201.3` vs `rejected_response = 145.1` (**+56.3 tokens**, 90th percentile **+152.5**, max **+2,322 tokens**). Why? First, `COUNT_RULES` was applied to **all** counting rules (60% of the pool, including `other_counting` like letter/keyword frequency and `fewer_than_N`), not just the 11% of `at_least_N` questions where short responses actually cause failures. Second, `right_count` required `response_len(c) >= response_len(w)` with **no upper cap**, and then selected `min(fit, key=thinking_len)`—frequently pairing a normal wrong answer with a 1,000+ token verbose right answer that happened to have a slightly shorter `<think>` block.
       - Even in IFEval `shorten` pairs (`--think-only ifeval --resp-guard 0.9`), `chosen_response > rejected_response` in **61.1%** of pairs (`>=` in **76.1%**), with mean `117.8` vs `99.0` (**+18.8 tokens**)! Because `long = rights[-1]` is the answer with the longest *thinking*, requiring `response_len(short) >= 0.9 * response_len(long)` truncates the lower half of the response-length distribution whenever `long` has a median or above-median response.
     - **The double result on `Dev-512` IFEval responses (`272.6 -> 278.8 -> 312.4 -> 330.1` tokens):**
       - On the 95.2% of `Dev-512` IFEval answers that finished normally (`truncated == 0`), mean response length grew from `153.1` (median `93.0`) in `base` and `119.1` (median `76.0`) in `rpE-80` to `147.3` (step 100) $\to$ `160.2` (step 150) $\to$ **`175.4` (median `111.0`) at step 200** (+22.3 tokens vs `base`, +56.3 tokens vs `rpE-80`).
       - Worse, because DPO penalized EOS inside the response (75.7% chosen response longer) and `repair_pairs.py` defines `wrongs = [r for r in q if finished(r) and not right(r)]`—**excluding truncated answers (`truncated == 1`) from ever serving as rejected examples**—runaway response loops that hit the 4,096-token cap (`startend:quotation`, `language:response_language`) rose from `41 / 1,056` (3.9%, adding `125.5` tokens to mean response) in `base` and `38 / 1,056` (3.6%, adding `130.6` tokens) in `rpE-80` to **`51 / 1,056` (4.8%, adding `163.2` tokens to mean response)** at step 200. Just the top 10 runaway questions out of 132 accounted for `+10,116.5` total response tokens (`+76.6` tokens per question across all 132 questions).
     - And on IFEval *thinking*, because `p1r1s0` built pairs from the raw model's 10 natural rollouts (where `shorten` chosen thinking averaged `244.7` tokens and `right` chosen thinking averaged `280.0` tokens), `p1r1s0` thinking stalled at `184–197` tokens (55–58% thinking saved) vs `rpE-80`'s `146` tokens (66.4% thinking saved).

2. **The Single-Round Natural-Rollout Ceiling under `fp32` Master Weights (why even SVAMP, MBPP+, and BFCL fell 5–14 points short of `rc1-soup-30`):**
   - Look at the chosen/rejected token ratios inside `path1_round1_pairs_local_build.csv` itself:
     - `chosen / rejected` mean total tokens by part: SVAMP `0.477` (**52.3% saved**), MBPP+ `0.441` (**55.9% saved**), IFEval `0.715` (**28.5% saved**), BFCL Simple `0.617` (**38.3%**), BFCL Multiple `0.638` (**36.2%**), BFCL No-Fit `0.458` (**54.2%**; BFCL pooled `0.577` = **42.3% saved**).
     - **The exact headline geometric mean of the chosen-vs-rejected mean token ratios in `p1_pairs` is 45.73% (median ratio: 40.09%)—and `p1r1s0-200` landed at 45.90%!**
   - Why do pairs built from 10 natural rollouts of raw `Qwen3-0.6B` with the `round1` 60/40 and 70/30 mix have an intrinsic target saving of ~46%?
     - In `shorten` pairs (`>= 8/10` right, `rights[0]` vs `rights[-1]`), the shortest of 10 natural rollouts of raw Qwen has mean thinking `375` tokens on SVAMP, `517` on MBPP+, `245` on IFEval, and `103–131` on BFCL.
     - In `right_cr` pairs (30–40% of pairs on non-IFEval parts, built on harder `< 8/10` questions), `rights[0]` has much **longer** thinking (`625` tokens on SVAMP, `887` on MBPP+, `132–209` on BFCL), and in 20–30% of `right_cr` pairs `chosen` is longer than `rejected` (`chosen longer` overall: 15% SVAMP, 10% MBPP+, 33% IFEval, 7–9% BFCL).
     - Furthermore, with `--bida-precision` (`fp32` master weights + `fp16` autocast), DPO does not suffer the `bf16` weight-rounding drift that artificially collapsed length by an extra ~8–11 points in `bidax4` (57.1% at -5.1 acc) vs `bidax6p` (45.7–48.9% at -1.6 to -2.2 acc). Every single-round `fp32` DPO run from raw Qwen's natural rollouts in the log (`bidax6p`: 45.7–48.9%, `bidax5q` s0: 41.8–48.5%, `michal-bida` Db: 47.0%) plateaued at **45–49% saved**. And between step 100 and step 200, as `p1r1s0` fitted the `right_cr` pairs (pair accuracy rising from 82% to 91%), SVAMP thinking actually rebounded from `442` (step 100) to `504` (step 150) and `470` (step 200).

### 1.3 What I Got Wrong in `consult8`
I owe a blunt accounting of my own errors in `consult8`:
1. **I failed to compute the actual chosen-vs-rejected token ratios of the `round1` pair table before forecasting 53–57%.** Had I calculated the geometric mean of `chosen_total / rejected_total` across the 4 benchmarks in `p1_pairs` (**45.7%**), I would have seen immediately that `p1r1s0` could not reach 53–57% without overshooting its own chosen responses.
2. **I benchmarked against `bidax4` (54.6–57.1%, which used `bf16` master weights) and `rc1-soup-30` (57.4%, which was 3 DPO rounds + 1 synthetic cut-calibration round) instead of `bidax6p` (45.7–48.9%) and `bidax5q` (46.7–48.5%).** Under `--bida-precision` (`fp32` weights), a single round of offline DPO on raw Qwen’s 1-of-10 natural rollouts with 30–40% `right` pairs has an empirical ceiling of **46–49% saved**, not 53–57%.
3. **I endorsed `--count-rules` (50% of IFEval quota) and `--resp-guard 0.9` for Round 1 without checking their net effect on IFEval response length.** I did not catch that applying `response_len(c) >= response_len(w)` (with no upper cap) to all counting rules (60% of IFEval) plus `--resp-guard 0.9` on `shorten` caused **75.7% of all IFEval pairs to prefer longer responses (+46.9 tokens)** and cut IFEval’s `shorten` share to **25%**, driving IFEval saving down to 25.6%.

---

## 2. Path 1 From Here: Stop or Continue?

| Option | Expected `Dev-512` Result | Expected Full-Test Result | GPU Cost | Judgment |
| :--- | :---: | :---: | :---: | :--- |
| **A. Run Seed 1 (`p1r1s1` + `p1r2s1`) as queued on GPU 1** | `45–48%` saved / `-0.5 to +0.3` acc (`p1r2s1` gate closes) | `45–48%` / `-2.0 to -2.7` | ~2.5 L4-h (`p1src` is already cached) | **Kill immediately.** Seed 0 did not fail because of seed noise; its trajectory (47.5% $\to$ 43.7% $\to$ 45.9%) converged directly to its pair table's 45.7% target ratio. Seed 1 will hit ~46% and close the Round 2 gate again. |
| **B. Pre-registered fallback: 60 more Round 1 steps (to step 260)** | `44–46%` saved / `-0.2 to +0.3` acc | `44–46%` / `-2.0 to -2.4` | ~1.2 L4-h | **Do not run.** Training pair accuracy is already **91%** at step 200, and headline saving *decreased* from step 100 (47.5%) to step 150 (43.7%) and step 200 (45.9%) as the model fitted the longer `right_cr` and IFEval pairs. More steps on the same pairs will not increase saving. |
| **C. Round 2 (`p1r2s0`) on top of `p1r1s0-100` or `p1r1s0-200` anyway (lowering the gate)** | `47–49.5%` saved / `+0.5 to +1.2` acc | `47–49.5%` / `-1.0 to -1.7` | ~3.0 L4-h | **Do not run as designed.** `p1r2s0` uses the Round 2 repair mix (30% shorten on non-counting IFEval, 50% counting-rule IFEval, 30% shorten on others). Starting from 45.9% (or 47.5% at step 100) where IFEval is at 25.6–34.7% and `p1r2s0` uses the same broken IFEval response rules, it cannot cross 50% saved and > -1.0 accuracy at the same time. |
| **D. Redesigned Round 1 + Round 2 from original Qwen (Clean Recipe)** | Round 1: `50–52%` / `-0.5` Dev; Round 2: `53–55%` / `+1.3` Dev | Round 2: `52–54%` / `-0.7 to -0.9` | ~1.5 L4-h (R1 reuses `p1src`) + ~3.0 L4-h (R2) | **Defer until `rpK` full tests and the targeted `rpL` repair finish, then run as the Clean Recipe test (Section 5).** |

**Single Recommendation on Path 1:** **STOP Path 1 as currently queued** (remove `p1r1s1` and `p1r2s1` from GPU 1's queue—and also stop `od2` on GPU 1 once its current round finishes, as `od1` already proved online loops slide along the frontier). Also **drop `p1r1s0-200` from the 4 queued full tests**—spending ~2.5 GPU-hours to full-test a model at `45.9% / +0.12` on `Dev-512` (~`45.8% / -2.1` full test, strictly dominated by `rpE-80` on both axes) teaches us nothing we do not already know from `Dev-512`.

---

## 3. `rpK`: Is the Result Real, and What Will the Full Tests Show?

### 3.1 `rpK` Results on `Dev-512` ($\times 8$)
`rpK` started from `rpE-80` (`Dev-512`: **46.9% / +1.36**; Full test: **47.0% / -0.9**) and trained a fresh LoRA for 60 steps (`lr=1e-5`, 2 epochs over 1,060 pairs built from `rpI`'s $\times 8$ rollouts of `rpE-80`, `--mix-preset typed`):

| Model | `Dev-512` Saved [95% CI] | `Dev-512` Acc [95% CI] | Paired vs `rpE-80`: Saved [95% CI] | Paired vs `rpE-80`: Acc [95% CI] | SVAMP Saved / Acc | MBPP+ Saved / Acc | IFEval Saved / Acc | BFCL Simple / Mult / No-Fit (Pooled) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`rpE-80`** | **46.9%** [44.2, 49.5] | **+1.36** [-0.21, +2.89] | — | — | 49.5% / +2.1 | 49.3% / +3.1 | 44.7% / -2.8 | 42.5 / 46.5 / 44.6 (**44.0% / +3.1**) |
| **`rpK-20`** | **50.5%** [48.1, 52.8] | **+1.51** [-0.02, +3.11] | **+3.6** [+2.0, +5.1] | **+0.15** [-0.94, +1.25] | 50.2% / +1.2 | 56.6% / +2.8 | 47.6% / -0.7 | 43.3 / 49.8 / 50.2 (**47.1% / +2.7**) |
| **`rpK-40`** | **50.4%** [47.9, 52.9] | **+0.95** [-0.55, +2.53] | **+3.5** [+1.8, +5.0] | **-0.41** [-1.59, +0.76] | 50.2% / +2.5 | 54.6% / +0.6 | 45.6% / -2.4 | 47.3 / 52.7 / 54.9 (**50.9% / +3.0**) |
| **`rpK-60`** | **53.8%** [51.3, 56.2] | **+0.76** [-0.73, +2.38] | **+6.8** [+5.2, +8.5] | **-0.60** [-1.72, +0.58] | 53.6% / +0.8 | 60.8% / +3.3 | 45.3% / -3.4 | 50.7 / 55.3 / 58.4 (**54.1% / +2.4**) |

### 3.2 Is `rpK`’s Result Real? (Weighing Paired Intervals, 3 Checkpoints, and Cross-Type Shortening)
**Yes, the shortening gain is unequivocally real; the accuracy is a small (-0.2 to -0.6 pt) trade-off from `rpE-80`, placing `rpK` right on the `-1.0` full-test boundary.** Specifically:

1. **The saving gain over `rpE-80` is statistically airtight (4.3$\sigma$ to 8.2$\sigma$ paired):**
   - Because `rpK` is evaluated on the exact same 512 questions as `rpE-80`, the paired bootstrap 95% confidence interval on saving gain has a half-width of only **$\pm 1.6$ points**: `+3.6% [+2.0, +5.1]` at step 20, `+3.5% [+1.8, +5.0]` at step 40, and **`+6.8% [+5.2, +8.5]` at step 60**. Even after subtracting a 3-checkpoint multiple-testing haircut (~0.5–0.8 points), step 20/40 saves at least +3.0 points more than `rpE-80`, and step 60 saves at least +6.0 points more.
2. **Why did Math (SVAMP: 49.5% $\to$ 53.6%) and Code (MBPP+: 49.3% $\to$ 56.6% / 54.6% / 60.8%) shorten when `--mix-preset typed` gave them 0% `shorten` pairs (80% `right_cr` / 20% `capguard`)?**
   - I inspected `rpK_pairs.csv` to verify whether `right_cr` on SVAMP and MBPP+ was secretly length- neutral. **It was NOT length-neutral:**
     - In `rpK_pairs.csv` on **MBPP+** (`128 right_cr + 32 guard`), `right_cr` had mean `chosen_thinking = 501.7` (median `397.5`) vs `rejected_thinking = 635.0` (median `425.0`)—a **-133.3 token shortening signal in the mean** (`chosen < rejected` in 53% of `right_cr` pairs), because failed code attempts on `rpE-80` often ramble into long debugging loops! And on `Dev-512` MBPP+, `rpE-80`'s mean thinking was `954.4` tokens (pulled up by the long tail: 95th percentile `3,480` tokens, `7.6%` truncated at 4,096). In `rpK`, two things reduced MBPP+ mean thinking from `954.4` to `762.5` (step 20), `808.2` (step 40), and `690.5` (step 60):
       1. Median MBPP+ thinking dropped smoothly (`476.0` in `rpE-80` $\to$ `425.0` at step 20 $\to$ `412.5` at step 40 $\to$ `382.0` at step 60)—a genuine 11%–20% cross-task + `right_cr` compression.
       2. Truncated 4,096-token MBPP+ rollouts (`800` samples on `Dev-512`) dropped from `61` (`7.6%`) in `rpE-80` to `43` (`5.4%`) at step 20, `49` (`6.1%`) at step 40, and `40` (`5.0%`) at step 60. Each truncated code sample avoided saves ~3,000 tokens (`~3.75` tokens on the part mean per sample). Part of the step-20 (`56.6%`) $\to$ step-40 (`54.6%`) $\to$ step-60 (`60.8%`) oscillation in MBPP+ is sampling variance in how many of those ~15 borderline hard code problems hit the 4,096 cap (43 vs 49 vs 40). On the 378-question full test, expect MBPP+ saving to regress ~2 points toward its median/trimmed trajectory (`~53–54%` at steps 20/40, `~57–58%` at step 60).
     - In `rpK_pairs.csv` on **SVAMP** (`128 right_cr + 32 guard`), `right_cr` had mean `chosen_thinking = 405.4` (median `286.0`) vs `rejected_thinking = 523.0` (median `330.0`)—a **-117.6 token shortening signal** (`chosen < rejected` in **62%** of `right_cr` pairs!). Combined with shared LoRA representation updates from the 45% shorten pairs on IFEval and BFCL, SVAMP median thinking moved from `308.5` (`rpE-80`) to `309.0` (step 20), `303.5` (step 40), and `290.0` (step 60).
3. **Where `rpK` clearly succeeded vs where it backfired:**
   - **Tool calls (`bfcl_simple`, `bfcl_multiple`, `bfcl_irrelevance`) are a home run:** Pooled BFCL saving rose monotonically from **44.0%** (`rpE-80`) $\to$ **47.1%** (step 20) $\to$ **50.9%** (step 40) $\to$ **54.1%** (step 60), while `bfcl_irrelevance` accuracy rose from **+1.4** (`rpE-80`) to **+2.8** (step 20), **+5.0** (step 40), and **+6.4** (step 60), with `bfcl_irrelevance` response tokens dropping from `11.1` to `4.6` as the model learned to abstain cleanly without verbose apologies.
   - **IFEval thinking dropped monotonically (`146.4 -> 135.4 -> 125.9 -> 112.7` tokens, reaching 74.1% thinking saved at step 60), completely cancelling out in total tokens because IFEval response tokens rose (`245.4 -> 235.6 -> 261.4 -> 274.8`):**
     - As proven in Section 1.2 from `rpK_pairs.csv`, across all 320 IFEval pairs in `rpK`, `chosen_thinking - rejected_thinking = -45.9` tokens (`108.5` vs `154.5`), while `chosen_response - rejected_response = +42.1` tokens (`161.4` vs `119.2`, with `chosen_response > rejected_response` in **75.9%** of pairs!). The net total-token difference in `rpK`'s IFEval pairs was literally **`-3.8` tokens** (`269.9` vs `273.7`).
     - Crucially, on `Dev-512` (`dev512_per_answer.csv`), `rpK`'s **non-truncated** IFEval responses did **not** get longer (`119.1` tokens in `rpE-80` $\to$ `120.3` at step 20 $\to$ `115.6` at step 40 $\to$ `114.0` at step 60). **100% of the response increase from step 20 (`235.6`) to step 60 (`274.8`) came from 13 extra runaway/truncated responses (`36 -> 43 -> 49` out of 1,056 answers hitting the 4,096 token cap**, mostly on `startend:quotation` and `language:response_language`), because preferring longer responses in 75.9% of pairs suppressed the EOS probability inside IFEval responses! And because truncated answers are automatically graded wrong (`right = graded_correct and not truncated`), those 13 extra truncated responses also caused **-1.2 points** of `rpK`'s IFEval accuracy drop from step 20 (-0.7) to step 60 (-3.4).

### 3.3 Full-Test Forecasts for the Queued Models and Confirm/Kill Criteria
Across the 5 calibration models (`rc1-soup-30`, `rpE-40`, `rpE-60`, `rpE-80`, `rpI-120`), `Full - Dev` averages **-0.1 pts** on saving (range `[-1.3, +1.3]`, with `rpE-80` at `+0.07` and `rpE-60` at `+1.3`) and **-2.24 pts** on accuracy (tight range `[-2.5, -1.7]`; specifically `-2.26` for `rpE-80`). Anchoring on `rpE-80`'s exact full test (**47.0% / -0.90**) plus the paired `Dev-512` deltas (with a ~0.7-point regression adjustment on step 20/60 for MBPP+ tail-truncation variance):

| Queued Model | `Dev-512` Score | Expected Full-Test Saving (Central / 90% Range) | Expected Full-Test Accuracy (Central / 90% Range) | Probability of Landing Inside Goal Box (`>50.0%` & `>-1.00`) |
| :--- | :---: | :---: | :---: | :---: |
| **`rpK-20`** | `50.5% / +1.51` | **50.1%** (`49.0% to 51.3%`) | **-0.80** (`-1.35 to -0.25`) | **~45%** (best accuracy, borderline on `>50.0%` saving) |
| **`rpK-40`** | `50.4% / +0.95` | **50.3%** (`49.2% to 51.5%`) | **-1.25** (`-1.80 to -0.70`) | **~25%** (saving ~50.3%, accuracy slightly below `-1.0`) |
| **`rpK-60`** | `53.8% / +0.76` | **53.2%** (`52.0% to 54.5%`) | **-1.45** (`-2.00 to -0.90`) | **~15%** (comfortably `>50%` saved, likely `-1.2 to -1.6` acc) |
| *(If run)* `p1r1s0-200` | `45.9% / +0.12` | **45.8%** (`44.5% to 47.1%`) | **-2.10** (`-2.70 to -1.50`) | **<1%** (**Cancel this full test**) |

- **What confirms `rpK` as a winner directly:** Either `rpK-20` or `rpK-40` scores **`> 50.0%` saved and `> -1.00` accuracy** on the full test, **OR** `rpK-20` and `rpK-60` bracket the box cleanly (e.g., `rpK-20` at `~50.0% / -0.8` and `rpK-60` at `~53.2% / -1.4`), which means a linear LoRA weight interpolation (`0.65 * rpK-20 + 0.35 * rpK-60`, or step 30) or fixing IFEval's response inflation (`rpL` below) puts the model squarely inside the box at **52–53% / -0.6 to -0.8**.
- **What kills the `rpK` line:** If `rpK-20` falls below `-1.5` accuracy on the full test **and** `rpK-60` falls below `50.5%` saved (which would imply severe overfitting to the `rpI` pool rather than true generalization).

---

## 4. The Next Runs in Order (Exact Settings) & The Sub-Floor Thinking Question

### 4.1 Should We Cut Thinking Below the Natural Floor (20 / 40 / 60 Thinking Tokens) Right Now?
**No—do NOT use synthetic 20/40/60-token thinking cuts (`rc1`-style cut calibration or the Budget Ladder) to get `rpK` into the box.** Here is why, from the data:
1. **`rpK-60` is ALREADY at 62.7% thinking-only saved on `Dev-512` (and 74.1% thinking saved on IFEval: `112.7` thinking tokens vs `base`'s `435.8`)!** Even if you cut IFEval thinking from `113` tokens all the way to `40` tokens, `73` fewer tokens out of `708` adds only `+10.3` points to IFEval saving (`+2.6` points to the 4-benchmark headline), and the `rc1` cut-calibration log (Section C8 of the Dossier) proves that synthetic mid-thought cuts degrade accuracy (`ca1-soup-10` `-2.4` $\to$ `rc1-soup-30` `-3.5` on the full test) because spliced `</think>` transitions are off-policy and cause premature stopping on the 241 counting-rule IFEval questions and multi-step math/code.
2. **Look at why `rpK-60`'s IFEval total saving stayed at 45.3% despite 74.1% thinking saving:** IFEval's response averaged **`274.8` tokens** (`114.0` tokens from non-truncated answers + **`166.1` tokens from the 4.6% of answers that loop to the 4,096 token cap**; vs `rc1-soup-30`'s `201.0` response tokens on the full test). Reducing `rpK`'s IFEval response from `275` back to `215–225` tokens (by fixing the pair builder so it stops preferring longer responses in 76% of IFEval pairs and starts rejecting truncated response loops) recovers **+7 to +8 points of IFEval saving (+2.0 points of headline saving) AND +1.5 to +2.0 points of IFEval accuracy (+0.4 to +0.5 points of headline accuracy)** with zero risk to math or code!

### 4.2 Exact Code Fixes in `repair_pairs.py` Before Launching Run 1 (`rpL`)
Make three surgical changes to `repair_pairs.py` (under a flag `--ifeval-strict-resp` or directly in `typed`):
1. **Restrict `COUNT_RULES` to minimum-length (`at_least_N`) rules only, and cap response expansion:**
   - Replace `COUNT_RULES` with `AT_LEAST_RULES = ("length_constraints:number_words", "length_constraints:number_sentences", "length_constraints:number_paragraphs", "detectable_format:number_highlighted_sections", "detectable_format:multiple_sections", "detectable_content:number_placeholders")`, excluding questions whose prompt contains `"less than"`, `"fewer than"`, `"at most"`, `"no more than"`, or `"under "`. (This narrows `is_cr` from 60% of IFEval to the ~12–15% of IFEval where short responses actually fail, and drops `right_count`'s share of the IFEval quota from 50% to **20%**).
   - Inside `is_cr`, cap the chosen response length so it never picks runaway verbose responses: `fit = [r for r in rights if id(r) not in taken and response_len(rej) <= response_len(r) <= 1.35 * response_len(rej) + 25 and thinking_len(r) <= 1.15 * thinking_len(rej)]`.
2. **Fix `--resp-guard` on IFEval `shorten` pairs so it guards against the question's median right response (`rmed`), not `long` (`rights[-1]`), and prevent response inflation:**
   - For `part in think_only`: let `rmed = sorted(response_len(r) for r in rights)[len(rights) // 2]`. Require `0.85 * rmed <= response_len(short) <= 1.05 * response_len(long)` (so `shorten` pairs never drop truncated stubs, **and** never push `chosen_response > 1.05 * rejected_response`).
   - For non-counting `right_cr` on IFEval: require `response_len(c) <= 1.10 * response_len(w) + 15` so correctness pairs on non-counting IFEval do not inflate responses.
3. **Include response-truncated answers (`closed_thinking and truncated`) in `wrongs` on IFEval (`resp_loop_guard`):**
   - When an IFEval question has both `rights` and response-truncated rollouts (`r["closed_thinking"] and r["truncated"]`), allow up to 1 `right_cr` pair per question where `rej` is a response-truncated rollout (`4,096` tokens) and `chosen` is `rights[0]`. This directly teaches DPO to emit EOS instead of looping on `startend:quotation` and `language:response_language`.

---

### Run 1: `rpL` — IFEval Response-Controlled Repair from `rpE-80` (Fastest Path into the Box with Margin)
- **Goal:** Keep `rpK`'s exact gains on SVAMP (`50–53.6%`), MBPP+ (`55–60%`), and BFCL (`51–54%` at `+2.4 to +3.0` acc), while lifting IFEval from `45.3% / -3.4` to **`52–54% / -0.5 to 0.0`** on `Dev-512`—putting the model at **53.5–55.5% saved and +1.3 to +1.8 accuracy on `Dev-512`** (**53–55% saved and -0.4 to -0.9 accuracy on the Full Test**, safely inside the box with margin).
- **Start model & answers used:** Base = `models/rpE-80`. Rollouts = **reuse existing `runs/bon/rp-rpI/rollouts.jsonl`** (the 5,700 questions $\times 8$ already generated by `rpE-80`—**0 generation cost**).
- **Quotas & Pair Mix (1,060 pairs total, matching `rpK`):**
  - `svamp=160, mbpp=160`: **0% `shorten` / 80% `right_cr` (`--prefer-reread`, `ratio <= 1.20`) / 20% `capguard`**.
  - `bfcl_simple_python=160, bfcl_multiple=120, bfcl_irrelevance=140`: **50% `shorten` (`ratio >= 1.30`, bar `>= 7/8` = `0.875`) / 50% `right_cr`**.
  - `ifeval=320`: **20% `right_count`** (64 pairs, on true `at_least_N` rules only, response capped at `1.35x + 25`) + **50% `shorten`** (160 pairs, `--think-only ifeval`, `ratio >= 1.30`, bar `>= 7/8`, response in `[0.85 * rmed, 1.05 * resp(long)]`) + **30% `right_cr`** (96 pairs, including response-truncated rollouts as losers, `resp(c) <= 1.10 * resp(w) + 15`).
  - **Pre-training sanity check on `rpL` pair summary:** Verify that on `ifeval`, `chosen_response > rejected_response` is **45–52%** (down from 75.9% in `rpK`) and mean `chosen_total < rejected_total` by at least **50 tokens** (vs `-3.8` tokens in `rpK`).
- **Training settings:** Fresh LoRA ($r=16, \alpha=32$), `--bida-precision` (`fp32` weights + `fp16` compute), $\beta=0.1$, batch 16, `lr=1e-5` constant (`--warmup 6 --lr-min 1e-5`), **60 steps** (~1 epoch), seed 0 (and seed 1 if seed 0 passes `Dev-512` gate, to 2-seed soup).
- **Checkpoints & `Dev-512` Decision Rule:** Score on `Dev-512` ($\times 8$) at **steps 25, 40, 55**.
  - **Pass gate for Full Test:** Pick the checkpoint (or 2-seed soup) with `Dev-512` **saved $\ge 52.5\%$** (paired vs `rpE-80` $\ge +5.5$ pts) AND **accuracy $\ge +1.25$** (paired vs `rpE-80` $\ge -0.1$ pts), with IFEval response tokens $\le 245$.
- **Cost:** **~1.2 L4 GPU-hours** (0h generation on pool, ~8 min DPO, ~65 min for 3 `Dev-512` screens).

---

### Run 2: Checkpoint Interpolation (`rpK-soup-20-60`) — Zero-Training Insurance If `rpK-20` and `rpK-60` Bracket the Box
- **When to run:** Immediately after the 3 queued `rpK` full tests (`rpK-20`, `rpK-40`, `rpK-60`) finish, if `rpK-20` is at `~50.0% / -0.8` and `rpK-60` is at `~53.2% / -1.4`.
- **Exact settings:** Both `rpK-20` and `rpK-60` are along the **same single LoRA trajectory** on top of `rpE-80`. Merge their LoRA adapter weights linearly at `0.60 * adapters/rpK-s0/step-20 + 0.40 * adapters/rpK-s0/step-60` (or score `adapters/rpK-s0/step-30` if saved on disk) on `Dev-512` first (~20 min); if `Dev-512` reads `>= 51.8% / >= +1.25`, run 1 full test (~2.5 L4-h).

---

### Run 3: Two-Round Clean Recipe From Original `Qwen3-0.6B` (`clean2r-r1` $\to$ `clean2r-r2`)
- **Goal:** Fulfill the user's requirement for **one clean, reproducible recipe starting from the original `Qwen3-0.6B`** that reaches `rpK` / `rpL`'s point (`52–54% saved / > -1.0 accuracy`) without going through October's 4-stage `rc1-soup-30` pipeline. (Full design and exact settings in Section 5 below.)

---

## 5. The Clean Recipe From Original `Qwen3-0.6B`: Why `rpK`'s Lineage Is 5 Stages Deep, and How to Compress It to 2 Clean Rounds

### 5.1 Why `rpK`'s Current Lineage Took 5 Stages—and Why 1 Round From Raw Qwen Cannot Replace It
Today, `rpK`'s lineage from `Qwen/Qwen3-0.6B` is:
$$\text{Original Qwen} \xrightarrow{\text{3 DPO rounds}} \text{ca1-soup-10 } (49.1\% / -2.4) \xrightarrow{\text{cut-cal DPO}} \text{rc1-soup-30 } (57.4\% / -3.5) \xrightarrow{\text{repair DPO}} \text{rpE-80 } (47.0\% / -0.9) \xrightarrow{\text{typed DPO}} \text{rpK}$$

Why did `p1r1s0` (and `bidax6p`, `bidax5q`, and `michal-bida` Db) stop at **46–49% saved** in 1 round from the original model?
1. **The 1-of-10 Sampling Horizon of Raw Qwen:** In raw Qwen (`1,115` thinking tokens on SVAMP, `1,934` on MBPP+, `436` on IFEval), the shortest right answer out of 10 rollouts (`rights[0]`) still averages `375` thinking tokens on SVAMP, `517` on MBPP+, and `245` on IFEval (where <1% of raw Qwen rollouts think under 100 tokens). A single offline DPO round under `fp32` master weights moves the model's mean thinking to the mean of its chosen examples (~46–49% saved), **shifting the model's length distribution down by ~2$\times$**.
2. **Once the Round 1 model is at ~48–50% saved, its OWN $\times 8$ rollouts have a new 10th-percentile floor that is ~2$\times$ shorter again** (e.g., IFEval 5th percentile drops to 75 tokens and median to 128 tokens; BFCL drops to 50–70 tokens; MBPP+ median drops to 400 tokens)—AND its failure modes at that shorter length become visible so `right_cr` and `capguard` pairs can repair them on-policy!
3. Look at why `p1r1s0` failed to reach 50–52% in Round 1:
   - In `p1r1s0`, mixing 30–40% `right_cr` pairs (which had much longer thinking: `625` on SVAMP, `887` on MBPP+) and 75% `right` pairs on IFEval (which inflated responses by `+56` tokens) into Round 1 **diluted Round 1's shortening from ~52% down to 45.9%**, while **still losing -2.1 to -2.5 points of accuracy** on the full test!
   - That is the core lesson of `p1r1s0`, `D6` (`simright`), and `runa2`: **Do not mix heavy correctness/guard pairs into Round 1 on the raw model's long answers.** Correctness pairs built on a long-thinking model teach the model to think *long* on hard problems, fighting the shortening signal in Round 1, whereas correctness/guard pairs built in Round 2 on the *already-shortened* Round 1 model's answers repair accuracy at the *short* length (just as `rpE-80` repaired +2.6 points of accuracy from `rc1-soup-30` while keeping 47.0% saved, and `rpK` pushed saving back to 53.8%!).

### 5.2 The Shortest Clean Recipe From Original `Qwen3-0.6B` (2 Rounds, Using Already-Generated `p1src` Rollouts for Round 1)

#### Stage 1 (`clean2r-r1`): Pure Shorten-Dominant Round 1 From `p1src` (Reusing `runs/bon/rp-p1src/rollouts.jsonl` — 0 Generation Cost!)
- **Base model & data:** `Qwen/Qwen3-0.6B`, reusing the already-generated `runs/bon/rp-p1src/rollouts.jsonl` (12,668 training questions $\times 10$, disjoint from `Dev-512` and the 25% Round 2 pool).
- **Pair construction (3,200 pairs: `svamp=800, mbpp=800, ifeval=800, bfcl_simple_python=280, bfcl_multiple=200, bfcl_irrelevance=240`):**
  - **90% `shorten` / 10% `right_cr` across all 6 parts** (with `--shorten-min-right 0.70` = `7/10` right so medium-difficulty math/code questions also get shortened in Round 1, `--shorten-ratio 1.40`, and up to 2 `shorten` pairs per question: `rights[0] > rights[-1]` and `rights[1] > rights[-2]`).
  - **With the fixed IFEval response rule (Section 4.2):** `--think-only ifeval` on `shorten` with `0.85 * rmed <= response_len(short) <= 1.05 * response_len(long)`, and the 10% `right` pairs on IFEval restricted to true `at_least_N` counting questions with `response_len(c) <= 1.35 * response_len(w) + 25`.
  - **Why this hits 51–53% in 1 round:** By removing the 30–40% long `right_cr` drag on SVAMP/MBPP+/BFCL and raising IFEval's `shorten` share from 25% to **90%** (with response inflation blocked), the pair table's chosen/rejected mean ratio moves from `45.7%` (`p1r1s0`) to **`54.5%`**, and IFEval `Dev-512` saving jumps from `25.6–34.7%` to **`48–51%`**, putting `clean2r-r1` at **51.5–53.5% saved and -0.8 to -1.5 accuracy on `Dev-512`** (~`51–53% / -3.0 to -3.7` full-test equivalent—matching `pt-mid` and `bidax2` cleanly in one `fp32` round).
- **Training:** LoRA ($r=16, \alpha=32$), `--bida-precision`, $\beta=0.1$, batch 16, `lr=5e-5` cosine to `5e-6`, **150 steps** (we saw in `p1r1s0` that step 100 had the highest saving, 47.5%, before steps 150–200 overfitted the longer pairs; screen on `Dev-512` at **steps 100 and 150** and pick the checkpoint with `Dev-512` saved $\ge 51.0\%$).

#### Stage 2 (`clean2r-r2`): On-Policy `rpL`-Style Typed Repair + Shorten on the Held-Out 25% Pool
- **Base model & data:** Merge the chosen `clean2r-r1` checkpoint; generate $\times 8$ rollouts on the disjoint 25% Round 2 pool (`3,480` questions, `--rp-exclude-dev --rp-exclude-ids data/p1-round1-ids.json`).
- **Pair construction (1,280 pairs: `svamp=320, mbpp=320, ifeval=320, bfcl_simple_python=110, bfcl_multiple=80, bfcl_irrelevance=130`) using `rpL`'s exact `typed` rules (Section 4.2):**
  - **SVAMP & MBPP+:** `0% shorten / 75% right_cr (--prefer-reread, ratio <= 1.20) / 25% capguard` (pure accuracy recovery on the Round 1 model's mistakes, exactly how `rpE-80` recovered +2.6 points of accuracy from `rc1-soup-30` while `right_cr` naturally trims rambling wrong code/math traces).
  - **BFCL (Simple / Multiple / No-Fit):** `50% shorten (bar >= 7/8, ratio >= 1.30) / 50% right_cr` (pushes BFCL from ~43% to **52–54% saved** and lifts `no-fit` accuracy by +3 to +5 pts, just as in `rpK`).
  - **IFEval:** `20% right_count` (true `at_least_N` only, capped response) + `50% shorten` (`--think-only ifeval`, `0.85 * rmed <= resp(c) <= 1.05 * resp(w)`) + `30% right_cr` (including truncated response loops as rejected). This pushes Round 1's ~185-token IFEval thinking down to `~115–125` tokens while keeping responses at `~220` tokens (**52–54% IFEval saved at -0.5 acc**).
- **Training:** Fresh LoRA ($r=16, \alpha=32$) on merged `clean2r-r1`, `--bida-precision`, $\beta=0.1$, batch 16, `lr=1e-5` constant, **80 steps**, scored on `Dev-512` at **steps 40, 60, 80**.

### 5.3 The Cheapest Experiment That Tests the Clean Recipe
Because `runs/bon/rp-p1src/rollouts.jsonl` (all 12,668 questions $\times 10$ from the original `Qwen3-0.6B`) **is already on disk from last night**, testing Stage 1 (`clean2r-r1`) requires **zero rollout generation**:
1. Build the `clean2r-r1` 90/10 pair file from `runs/bon/rp-p1src/rollouts.jsonl` with the fixed IFEval response rule (~1 minute on CPU).
2. Run 150 DPO steps from `Qwen/Qwen3-0.6B` (~18 minutes on L4) and score steps 100 and 150 on `Dev-512` (~42 minutes total).
3. **Total cost of the Stage 1 test: 1.0 L4 GPU-hour.** If `clean2r-r1` hits **$\ge 51.0\%$ saved on `Dev-512`** (with IFEval $\ge 47\%$), immediately run Stage 2 (`clean2r-r2`, ~2.8 L4 GPU-hours including generation on the 3,480 Round 2 questions).

---

## Summary Action Checklist for Both GPUs Today
1. **Queue / Full-Test Triage:** Let the 3 `rpK` full tests (`rpK-20`, `rpK-40`, `rpK-60`) finish; **cancel the `p1r1s0-200` full test**, **cancel `p1r1s1` / `p1r2s1`**, and **stop `od2`** on GPU 1.
2. **Launch Run 1 (`rpL`, ~1.2 L4-h) immediately on the first free GPU:** Reuse `runs/bon/rp-rpI/rollouts.jsonl` on `models/rpE-80` with the 3 IFEval response fixes in `repair_pairs.py` (Section 4.2). Screen steps 25 / 40 / 55 on `Dev-512`.
3. **Launch the Clean Recipe test (`clean2r-r1` $\to$ `clean2r-r2`, ~1.0 L4-h for Stage 1 + ~2.8 L4-h for Stage 2) on the second GPU:** Reuse `runs/bon/rp-p1src/rollouts.jsonl` with the 90/10 shorten mix + fixed IFEval response rule for Stage 1, gated at `51.0%` `Dev-512` saving into Stage 2 (`typed` repair + IFEval/BFCL shorten).