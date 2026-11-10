```python
# Let's finish writing Sections 4 and 5 into work/report_draft.md and update work/notes.md so that both files are 100% complete right now!
import os

work_dir = "/Users/arsenelemoigne/Claude Build/Test-mvt-fading/thinkingcap/reviews/consult6_argon/work"
notes_path = os.path.join(work_dir, "notes.md")
draft_path = os.path.join(work_dir, "report_draft.md")

with open(notes_path, "a") as f:
    f.write("""
## 6. Counterfactual & Strategy Numbers
- On the full test (2,759 x 8):
  - `rpe-40full` scored **50.6% saved / -1.90 pp** (v: 59%/-3.0, c: 53%/-1.7, f: 47%/-2.8, bfcl: 42.6%/-0.07 with no-fit -0.4).
  - `rpe-60full` scored **47.8% saved / -1.20 pp** (v: 55%/-1.5, c: 48%/-0.7, f: 46%/-3.0, bfcl: 41.4%/+0.40 with no-fit +1.4).
  - `rpe-80full` scored **47.0% saved / -0.90 pp** (v: 54%/-1.2, c: 46%/-1.0, f: 45%/-2.6, bfcl: 41.2%/+1.04 with no-fit +3.4).
  - `rpi-120full` (which added `--count-rules` on 320 IFEval pairs incl. 152 counting-rule pairs and fresh current-model rollouts) moved IFEval from -2.6/-3.0 pp to **-1.5 pp** (+0.28 to +0.38 pp headline gain) and `no-fit` from +1.4/+3.4 pp to **+5.8 pp** (+0.17 to +0.31 pp headline gain).
  - Counterfactual: `rpe-60full` with `rpI`'s IFEval (-1.5 pp) and `no-fit` (+5.8 pp) reaches **-0.51 pp** headline accuracy; `rpe-40full` with those two fixes reaches **50.6% saved / -1.13 pp** (only 0.13 pp from the goal before adding length-anchor pairs!).
  - Why `rpE` lost saving from step 40 (50.6%) to step 80 (47.0%) and `rpI` to 41.9%: both had **0% shorten (length-anchor) pairs**, so training drifted back toward longer answers. Adding 25-30% shorten pairs on reliably solved questions (`>= 6/8`, never on IFEval counting rules) holds saving at 51-53% while the equal-length re-read + `--count-rules` + `no-fit` pairs lift accuracy to -0.5..-0.9 pp.
""")

sec4_5_text = """
---

## 4. One-Week Strategy on Two Colab L4 GPUs (Sequence, Exact Settings, Expected Ranges, Decision Rules)

You have ~14 GPU-days (~336 raw GPU-hours, or ~220–250 usable GPU-hours after Colab reconnects/Drive syncs). Do **not** spend 150+ GPU-hours running 7 stages × 11 rungs of forced cuts. Run the following 4-step sequence, where every step has a hard quantitative gate.

### Step 0: Build the Locked Non-Benchmark Dev Set (512 q) + Confirmation Set (512 q) & Run Bias-Corrected Phase 0 (~6 GPU-hours total; Day 1 morning)
- **Should you run Phase 0 first?** **Yes, but ONLY with the 4+4 split-sample fix and 4 coarse rungs (`0, p25, p50, p75, nat`)**—because it takes only ~2.5 GPU-hours and will immediately prove to the whole team that (a) the unsplit `1/8`-slack oracle is inflated by +6 to +8 pp, and (b) forced cuts on raw Qwen SVAMP cannot reach the ~58% math saving needed for a >50% headline without compacting the reasoning itself.
- **Exact Setup:**
  1. **Dev-512 & Confirm-512 (never touched by training, zero overlap with the 2,759 test):** Draw from `own_repair_questions()` (`pool-dp13` minus MetaMath + `gsm8k_std` + `nofit1/2` + `tools1` + `code1`, 8-gram & 13-word decontaminated against all 2,759 test questions): per set, **140 SVAMP-worded GSM8K/GSM-IC/Plus, 100 KodCode (MBPP-like slice), 132 IFEval-default (matched instruction distribution, ≥50% counting rules), 140 BFCL-clean tools (60 simple, 35 multiple, 45 verified `no-fit`)**. Score `full-base` (raw Qwen) × 8 on both (~50 min total across 2 GPUs).
  2. **Bias-Corrected Phase 0 (1,500 own training questions × 8):** On the 8 uncapped rollouts, split into **Selection (4)** and **Confirmation (4)**. Test 4 coarse cut rungs (`0, p25, p50, p75`) + `natural`. Compute both the *unsplit `1/8`-slack oracle* and the *split-sample `0`-slack monotone-suffix oracle*, plus the Spearman rank correlation `rho` between `b*(q)` and prompt length, and compare against the **natural shortest-of-8 oracle** on the same questions.
- **Expected Result (Range):**
  - Unsplit `1/8`-slack forced-cut oracle: **58–66% saved / −0.5 to +3.0 pp** (artifact of stopping-time bias).
  - **Split-sample (4+4) `0`-slack forced-cut oracle:** **44–50% total saved / −0.5 to −1.5 pp** (SVAMP 38–45% saved, MBPP+ 65–72%, IFEval 28–33% total saved, BFCL 40–45% saved), and prompt-length rank correlation **`|rho| < 0.20`**.
  - **Split-sample natural shortest-right oracle (on questions with ≥2 right):** **32–42% saved in 1 round at 0.0 pp loss**, confirming that multi-round natural compaction is needed rather than forced truncation.
- **Decision Rule:**
  - If the **split-sample forced-cut oracle** saves `< 55%` total tokens at `>= -0.5 pp` OR `|rho| < 0.30` (expected: both fail), **drop forced-cut ladder training immediately** and proceed to **Experiment 1 (Coarse 2-Stage Natural Budget Ladder vs 2-Round On-Policy Compaction)** below.

### Step 1 (Main Bet — GPU 1 & GPU 2 in Parallel): 2-Stage Natural Compaction + On-Policy Calibration from Raw Qwen (~45 GPU-hours; Day 1 afternoon – Day 3)
To satisfy **Constraint 1 (one clean, reproducible single training line from raw Qwen)**, **Constraint 2 (no competitor data or recipe)**, and **Constraint 3 (own answers only)** while combining the two things that actually worked in your log—on-policy multi-round shortening (`cleanA` / `odpo`) and equal-length re-read + `--count-rules` + `no-fit` calibration (`rpE` / `rpI`)—run two clean arms from raw Qwen (2 seeds each, `s0` on GPU 1, `s1` on GPU 2):

- **Shared Clean Own Data Pool (~14,000 own questions, zero MetaMath/bida files):** `gsm8k_std` + `gsm_ic` + `gsm_plus` (math), `kodcode` (code), `ifeval_default` matched (instructions, ≥1,500 prompts), `toolace_clean` + `nofit1/2` (tools, with 35–40% `no-fit`).
- **Stage 1 (Base Compaction + Budget Guard, from Raw Qwen, 8,000 questions × 8 stored own answers):**
  - **Pair rules (all natural complete answers, `</think>` emitted naturally; 3,200 pairs = 200 steps × 16):**
    - **Quotas by type:** SVAMP-math **800** (25%), KodCode **800** (25%), IFEval **800** (25%, of which ≥350 counting-rule prompts), Tools **800** (25%: 280 simple, 200 multiple, **320 `no-fit`**).
    - **Pair mix within Math & `No-Fit`:** **35% Shorten Anchor** (on questions solved `>= 6/8`: shortest right > longest right `>= 1.4x` thinking) + **50% Equal-Length Re-Read Right** (on questions solved `2/8–6/8`: right > finished wrong within **±20% thinking length**, chosen prefers question re-read) + **15% Capped Anti-Early-Stop Guard** (right with thinking `<= 1.15x` median right > wrong that stopped early `< 0.5x` median right).
    - **Pair mix within Code, Simple, Multiple:** **50% Shorten Anchor** (`>= 6/8`, ratio `>= 1.4x`) + **35% Equal-Length Re-Read Right** (±20% length) + **15% Capped Guard**.
    - **Pair mix within IFEval:** On **counting-rule questions (~350 pairs)**: **100% `--count-rules` Right>Wrong** (chosen response ≥ rejected response length, chosen thinking `<= 1.2x` rejected thinking and `>= 80` tokens; **0% shorten**). On **non-counting questions (~450 pairs)**: **55% Shorten Anchor** (thinking ratio `>= 1.4x`, response length within ±15%) + **45% Equal-Length Right**.
  - **Training settings:** `LoRA r16 alpha 32` (all linear), `fp32 weights + fp16 autocast`, `beta = 0.1`, `NLL = 0`, `lr = 1.5e-5` (10-step warmup, cosine to `5e-6`), 200 steps (1 pass), `reference = raw Qwen`. Merge each seed at step 200 (`Stage1-s0`, `Stage1-s1`).
- **Expected Result at End of Stage 1 (on Dev-512 × 8):**
  - **42–47% total tokens saved, −0.6 to −1.4 pp accuracy** (Math ~48% / −1.5 pp, Code ~45% / −0.5 pp, IFEval ~40% / −1.0 pp, BFCL ~38% / +0.5 pp with `no-fit` +3 to +5 pp).
- **Decision Gate 1 (Dev-512 × 8):**
  - Must achieve **≥ 38% saved and ≥ −2.0 pp** on Dev-512 (both seeds). If saving is `< 38%`, increase Stage 2 Shorten share from 30% to 45%; if saving is `> 48%` with accuracy `< -1.8 pp`, set Stage 2 Shorten share to 20%.

- **Stage 2 (On-Policy Budget Compaction + Re-Read Calibration, 6,000 fresh own questions × 8 answered by `Stage1` model):**
  - Why Stage 2 works where 1-round DPO stalls: In Stage 1, raw Qwen's right answers on math and code had a wide spread around 680 and 2,080 tokens; in Stage 2, the `Stage1` model generates already-compact traces (~350 math, ~1,000 code) AND exposes its own current misreads and early stops (the exact mechanism that made `rpI` beat `rpG` by +1.7 pp vs curve!).
  - **Two controlled arms on Stage 2 (to test the bottom-up vs top-down hypothesis cleanly on identical rollouts):**
    - **Arm A (Joint Balanced On-Policy Stage 2, 2 seeds):** 1,600 pairs (100 steps × 16) from `Stage1`'s own ×8 answers using the same 4-type quotas (400 per benchmark, including 200 counting-rule IFEval and 160 `no-fit`), with **30% Shorten Anchor** (`>= 6/8`, ratio `>= 1.30x` on current model's shorter distribution, keeps saving above 51–53% so it cannot drift back like `rpE`/`rpI`) + **55% Equal-Length Re-Read / `--count-rules` Right** (±20% thinking length, median chosen ≤ median rejected) + **15% Capped Anti-Early-Stop Guard**. Fresh `LoRA r16` initialized on `Stage1` merged model, `reference = Stage1 merged`, `lr = 1e-5` constant, 80 steps (checkpoints at 40, 60, 80 evaluated on Dev-512 × 8).
    - **Arm B (2-Rung Natural Budget Ladder Stage 2, 2 seeds):** Same 1,600 pairs from `Stage1`'s ×8 answers, but ordered as two 40-step curriculum blocks (`lr = 1e-5` constant, `reference = Stage1 merged` throughout both blocks so reference never drifts!):
      - *Block 2a (Low-Budget Rung, steps 1–40, `B <= p50` of `Stage1` right thinking per type):* 800 pairs from questions solved `>= 6/8` within `p50` thinking (50% Shorten Anchor toward `p25–p50`, 50% Equal-Length Right within `p50`).
      - *Block 2b (High-Budget Rung + Guard, steps 41–80, `B > p50`):* 800 pairs from harder questions (`2/8–6/8`) where `p50` failed (`< p0 - 1/8` on 4+4 split) plus replay of 25% Block 2a Shorten Anchors (60% Equal-Length Re-Read Right + `--count-rules`, 15% Capped Guard where `p50` wrong < `p85` right, 25% Shorten Anchor).
- **Expected Result at End of Stage 2 (Dev-512 × 8):**
  - **Arm A:** **50.5–53.5% total saved, −0.4 to −0.9 pp accuracy** (Math 55–58% / −1.0 to −1.5 pp, Code 50–54% / −0.5 to +0.5 pp, IFEval 43–46% / −1.0 to −1.5 pp, BFCL 41–43% / +0.8 to +1.5 pp with `no-fit` +4 to +6 pp).
  - **Arm B:** **49.0–52.5% total saved, −0.5 to −1.2 pp accuracy** (tests whether ordering pairs by natural budget rung helps or suffers from Block 2b overwriting Block 2a).
- **Decision Gate 2 (Dev-512 × 8 → Confirm-512 × 8):**
  - Pick the single best arm + predetermined step (`40, 60, or 80`) on **Dev-512 × 8** that maximizes accuracy subject to **saved ≥ 51.0% on Dev-512** (the +1.0% buffer covers finite-sample variance). Verify both seeds on **Confirm-512 × 8**.

### Step 2 (Fast Backup / Alternative Track, Only If Needed): Balanced Single-Step Repair of `rc1-soup-30` WITH 25% Length Anchors (~12 GPU-hours; Day 4)
If the team also wants a direct comparator on the `rc1` line (or if Step 1 lands at 48–49.5% saved):
- Why `rpE-40full` (50.6% / −1.90 pp) and `rpE-80full` (47.0% / −0.90 pp) bracketed the target: `rpE` had **zero** shorten pairs, so between step 40 and step 80 it gave back 3.6% saving to buy +1.0 pp accuracy, and lacked `--count-rules` on IFEval (−2.6 to −3.0 pp).
- Run `rpJ`: `rc1-soup-30` answers 5,700 own questions × 8 (reuse `rpG`/`rpH` rollouts!), build **1,600 pairs** with **25% Shorten Anchor** (`>= 6/8`, ratio `>= 1.35x`, never on IFEval counting rules) + **75% Equal-Length Re-Read Guard + `--count-rules`** (with 320 IFEval pairs incl. ≥160 counting-rule and 120 `no-fit` pairs), `lr = 1e-5` constant, 60 steps.
- **Expected Result:** Holds saving at **51.0–52.5%** while reaching **−0.5 to −0.9 pp** on the full test (replaces `rpE-40full`'s −2.8 pp IFEval and −0.4 pp `no-fit` with −1.5 pp and +4.5 pp). *(Note: keep Step 1 as your primary submission because Step 1 is a clean 2-stage recipe from raw Qwen, whereas `rc1 + rpJ` is a 5-stage legacy stack.)*

### Step 3: Final Locked Full-Test Evaluation (2,759 × 8, Both Seeds + 2-Seed Weight Soup) (~8 GPU-hours; Day 5)
- Run the official 2,759 × 8 evaluation **once** on Seed 0, Seed 1, and their 50/50 weight-space average (`merge.py`, which remains a single merged LoRA checkpoint and reduces single-seed GPU non-determinism by `1/sqrt(2)`). Compute question-level paired bootstrap 95% CIs against `full-base`.

---

## 5. Traps in Evaluation, Data, and Method — and the Better Route You Are Missing

### 5.1 Eight Concrete Traps Identified in Your Log & Codebase
1. **Trap 1 — The Phase 0 In-Sample Stopping Mirage (+6 to +8 pp Accuracy Inflation):**
   In `LADDER_DESIGN_FOR_REVIEW.md` §5–6, `b*(q)` is defined as the lowest of 10 rungs where `k_rung >= k_0 - 1` on 8 samples, and the Phase 0 "oracle curve" reports tokens and accuracy at `b*(q)` on those **same** 8 samples. By binomial order statistics, testing up to 10 rungs with 1/8 slack guarantees that even if early rungs have a true −20 pp accuracy loss, 81.1% of questions stop at a lucky early rung, and reporting the in-sample accuracy of that lucky rung hides the entire loss (+6 to +8 pp optimism bias). **Fix:** Always use a 4+4 split-sample monotone-suffix rule (`k_sel(r') >= k_sel(nat)` for all `r' >= r`, 0 slack) and report oracle accuracy exclusively on the held-out 4 confirmation samples.
2. **Trap 2 — The Stage 1 → Stage 2 Rollout Deadlock in Bottom-Up Cut Training:**
   If Stage 1 trains the model at `B_0 = 0` (`<think></think>`) until it exits `<think>` immediately on low-level questions, then at Stage 2 (`B_1 = p5`) and Stage 3 (`B_2 = p10`), when the current model generates uncapped thinking to be cut at `B_k`, it will close `</think>` early on its own! You cannot discover whether a question benefits from `B_k` tokens by cutting a model that has already been trained to stop at `B_{k-1}`.
3. **Trap 3 — Total Output Tokens vs Thinking Tokens on IFEval (The 62% Ceiling):**
   Your ladder budgets count *thinking* tokens only, but the official benchmark scores *thinking + final response* tokens. Because IFEval final responses are ~247 tokens (38% of total IFEval tokens) and *must not shrink* (or counting rules lose −11 to −13 pp), cutting IFEval thinking by 65% yields only **40.3% total saving on IFEval** (`rpE-50`). If you budget SVAMP and MBPP+ for only 50% thinking savings, your 4-benchmark geometric mean will land at **~44–45% total saved**, missing the >50% bar. Math and Code must save **56–60%** to offset IFEval and BFCL's ~41% ceiling.
4. **Trap 4 — Sequential Reference Drift (`reference = previous stage` across 7 stages):**
   In DPO, the KL penalty `beta * KL(pi || pi_ref)` only constrains the step from `pi_{k-1}` to `pi_k`. Chaining 7 stages (`pi_0 -> pi_1 -> ... -> pi_7`) allows total log-probability drift to grow linearly with stage count, causing later stages to overwrite earlier stages (`stopV3` Block 2b overwriting Block 2a, or `rpG` undoing 25 points of `rc1`'s saving). **Fix:** Never exceed 2 chained reference hops from raw Qwen, and include 25–30% Shorten Anchor replay in Stage 2.
5. **Trap 5 — Hidden Length Signals in "Guard" and "Correctness" Pairs:**
   Your 10 Nov `copdanalysis` proved that on raw Qwen's training answers, wrong answers are **1.6× to 2.0× longer** than right answers (SVAMP right 880 vs wrong 1,773 tokens, wrong longer on 72% of questions; MBPP+ 1,449 vs 2,271, 62%). Conversely, in `stopV2` and `mf1`, "longest right > shortest wrong" guard pairs had chosen answers 1.5–2.0× longer than rejected answers, which made the model think **15–24% longer** across all types! **Fix:** Every correctness pair (`Equal-Length Right`) must enforce `|len(chosen) - len(rejected)| <= 0.20 * len(rejected)` AND verify that `median(len(chosen)) <= median(len(rejected))` per data type before training starts.
6. **Trap 6 — Rank-1 Cross-Type Spill-Over (`lpt6` Cosine 0.991):**
   On Qwen3-0.6B, DPO length pressure on *any* data type spills over globally (`ptsolo-math` shortened MBPP+ by 51%; `ptsolo-tools` shortened MBPP+ by 59% and SVAMP by 53%). Therefore, dynamically giving more questions in later ladder stages to "types further from the original's accuracy" (which is always Math) will either over-shorten or over-lengthen Code and Tools simultaneously. **Fix:** Keep type quotas strictly fixed across stages.
7. **Trap 7 — GPU Non-Determinism (~1.2–3.5 pp) and Open Code Bugs (`consult5` Audit):**
   Your `lpc-bidax2` vs `lpc-bidax3` (identical pairs, identical seed 0, pure GPU non-determinism) differed by **2.7% saving and 1.2 pp accuracy** (51.8% / −2.7 vs 54.5% / −3.9), and `lpc-runa2` vs `lpc-runa2r` differed by **7.9% saving** at step 200. Single-seed checkpoints cannot be trusted; **averaging 2 seeds in weight space (`merge.py` 0.5/0.5)** is a single clean LoRA adapter that cuts variance in half. Also fix the two still-open bugs from the 10 Nov audit before training: (a) `dpo.py` `ref.json` cache fingerprint must hash full token IDs of chosen+rejected (not just prefixes + lengths), and (b) log explicit SHA/path identities of `rollout_model`, `init_model`, and `ref_model` in every run header.
8. **Trap 8 — Upstream Submission Requirement (Commit `965a54c`, 6 Nov):**
   BottleCapAI's 6 Nov upstream commit requires every submission to include the coding agent's traces and conversation with the harness as a text document, and forbids any recipe built on another competitor's data/pipeline. Ensure your final 2-stage script runs end-to-end from clean own data (`own_repair_questions` pool) with a clean audit trail.

### 5.2 The Better Route You Are Missing (Within All Team Constraints)
Instead of trying to make a 0.6B model learn a **10-rung per-question budget router from forced cuts** (which fights the model's uncalibrated confidence and rank-1 length representation), exploit the exact separation of concerns that your 205 runs proved:
- **Global Density Compaction + Local Self-Verification (`2-Stage On-Policy Compact-and-Calibrate DPO`):**
  1. **Why every 1-round DPO run hit the −2.9 pp wall at 50% saved:** In a single offline DPO round from raw Qwen, shortening pairs compress thinking by ~50–60%, which causes two collateral damages on the newly shortened distribution: (a) on SVAMP and MBPP+, the compressed trace skips re-reading the question's trap condition or edge case (accounting for 68% of math regressions in `ca1`/`rc1`), and (b) on IFEval counting rules, the compression spills into the final response or aborts thinking before counting words/sections (−0.65 pp headline loss).
  2. **Why `rpE` + `rpI` broke through the curve (+1.64 vs curve) but lost saving (57.4% → 47.0% → 41.9%):** `rpE` and `rpI` proved that training on the **shortened model's own mistakes** using **equal-length right>wrong pairs with a question re-read filter + IFEval `--count-rules` + `no-fit` refusals** repairs math (−6.6 → 0.0 pp), code (−3.5 → +0.5 pp), and `no-fit` (−4.1 → +5.8 pp) *without needing long thinking*. They only lost token savings because they had **0% shorten anchor pairs**.
  3. **The Winning Synthesis:** A **single, self-contained 2-stage on-policy DPO pipeline from raw Qwen** (Step 1 Arm A above: Stage 1 compresses raw Qwen to ~45% saved with balanced Shorten + Equal-Length Re-Read + `--count-rules` + `no-fit`; Stage 2 samples ×8 from Stage 1 on fresh own questions and trains for 60–80 steps on **30% Shorten Anchors + 55% Equal-Length Re-Read/`--count-rules` Right + 15% Capped Anti-Early-Stop Guards**, averaged across 2 seeds). This requires no forced cuts, no test-time router, no teacher model, and no competitor recipe—and directly targets **51–53% saved at −0.4 to −0.8 pp** on the full test within 45 GPU-hours.
"""

with open(draft_path, "a") as f:
    f.write(sec4_5_text)

with open(draft_path, "r") as f:
    full_draft = f.read()

print(f"Full report_draft.md written successfully ({len(full_draft)} chars). Ready to deliver final report.")
```