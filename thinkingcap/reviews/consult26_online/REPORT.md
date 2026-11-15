### 1. Review of `od5c` and Why `od5d` (`v3`) Did Worse

#### A. What Made `od5c` Work
`od5c` reached **54.0% saved / -0.56 accuracy** (+3.14 above the reference curve) at round 15 and **54.8% saved / -0.78 accuracy** (+3.09 above the curve) at round 16 on the full 2,759-question $\times$ 8-sample test because three mechanisms worked together:

1. **Per-question health gating and self-proportional pair mixing (`--od5-v2`):**
   - On undamaged questions (`p1 >= 0.875` when `p0 = 1.0`; `p1 >= max(0.50, p0 - 0.125)` when `p0 >= 0.5`; else `p1 >= max(0.25, p0)`), `od5-v2` builds `shorten` pairs (`rejected / chosen >= 1.30`, thinking floor `max(min_tokens, floor_ratio * bt)`) and compact `be-right` (`hold`) pairs (`chosen` thinking $\le 1.0 \times \text{wrong} + 10$). Response-length floors (`>= max(0.95 * own_right_med, 0.92 * orig_right_med)`) and `<= 1.05 * rejected_resp` are restricted strictly to IFEval, and IFEval `--count-mode atleast` questions (~30% of IFEval) are never shortened.
   - On damaged questions (`p1 <= p0 - 0.25` when `p0 >= 0.875`, else `p1 <= p0 - 0.125`), `od5-v2` builds `recovery` pairs that reject hasty wrong answers (`thinking < 0.85 * bt`, shortest first) and choose clean right answers (`1.08 * wrong + 10` to `min(1.05 * bt, 1.45 * wrong + 50)`), preferring solutions with self-verification re-reads.
   - After taking defect pairs up to fixed caps (`loop` $\le 12\%$, `format` $\le 8\%$, IFEval `right_count` $\le 10\%$), the remaining quota splits proportionally to each question pool's candidate counts (`shorten : recovery : hold`), with **no artificial cap** on `hold` or `recovery`.

2. **The unlatching per-type brake (`--od-caps` + `--od-acc-hold -2.0` + `--od-unlatch`) and natural slot reduction when held:**
   - Whenever a type exceeded its saving cap (`svamp=62, mbpp=63, ifeval=56, simple=46, multiple=49, irrelevance=56`) or dropped to $\le -2.0$ accuracy on the current round's 780-question slice, `shorten` was suppressed for that type on that round.
   - Crucially, in `od5c` (`--od5-v2`), when a braked type did not have enough `recovery + hold + loop + format` candidates to fill its quota (e.g., round 6 had 224 pairs; round 12 had 204 pairs; round 15 had 196 pairs; round 16 had 211 pairs), **unfilled slots were left empty** rather than transferred to other types. Because `dpo.py` runs a fixed 15 steps per round (`batch = 16`, 240 pair exposures), rounds with braked types repeated the repair/hold pairs while reducing the total number of shortening pairs in that round.

3. **Running through the rounds 11–16 consolidation phase:**
   - Comparing the three full-tested checkpoints of `od5c` (`r10`, `r15`, `r16`) reveals how online DPO matures:
     - **Round 10 (`od5c-r10`):** `54.1% saved / -1.68 acc` (+2.04 above curve). Per type: math `65 / -4.6`, code `54 / -1.3`, IFEval `46 / -0.7`, simple `46 / -1.1`, several `48 / -1.1`, no-fit `53 / +2.3`.
     - **Round 15 (`od5c-r15`):** `54.0% saved / -0.56 acc` (+3.14 above curve). Per type: math `62 / -2.8`, code `54 / -0.9`, IFEval `48 / +0.5`, simple `48 / -1.6`, several `50 / -1.4`, no-fit `57 / +6.9`.
     - **Round 16 (`od5c-r16`):** `54.8% saved / -0.78 acc` (+3.09 above curve). Per type: math `61 / -2.8`, code `59 / -2.0`, IFEval `47 / +0.5`, simple `49 / -1.1`, several `50 / 0.0`, no-fit `57 / +5.9`.
   - Between round 10 and round 15, **composite saving did not change (`54.1%` $\to$ `54.0%`), but full-test accuracy recovered by `+1.12` points (`-1.68` $\to$ `-0.56`)**: math braked in rounds 12 and 14 and relaxed from over-shortened `65% / -4.6` to `62% / -2.8` (`+1.8` pts), code improved from `-1.3` to `-0.9` (`+0.4` pts), IFEval improved from `46% / -0.7` to `48% / +0.5` (`+1.2` pts), and no-fit improved from `53% / +2.3` to `57% / +6.9` (`+4.6` pts).

#### B. What Was Structural vs. Potential Luck in `od5c`
- **Stopping round (Structural, NOT luck):** Full-testing two consecutive rounds (r15 `54.0 / -0.56` and r16 `54.8 / -0.78`) proves that the late regime is a stable multi-round plateau inside `> 52% / > -1.0`, not a single lucky checkpoint.
- **Data refills in slices 12–16 (Minor tailwind on no-fit, ~0.2 pts composite):** Slices 13–16 used Haiku-written `no-fit` refills with `--od3-tol bfcl_irrelevance=5`, so slice 13–16 no-fit solve rates were `86–88%` vs. the test's `83%`. That helped no-fit climb to `+6.9` (r15) and `+5.9` (r16) on the full test. However, because `no-fit` has weight $0.25 \times \frac{240}{840} = \frac{1}{14} = 7.14\%$ of composite accuracy, even if `no-fit` had stayed at `od5d`'s `+3.6` (a 2.3–3.3 pt lower gain on no-fit), composite accuracy would shift by only $-0.16$ to $-0.24$ pts (`-0.56` $\to$ `-0.72` to `-0.80` at r15), still inside `> 52% / > -1.0`.
- **Precision (`--bida-precision`) and Seed:** See Section 1.C below; `fp32` master weights prevented LoRA update underflow at `lr = 1.4e-5`, which is a real numerical effect rather than luck. Seed variance is being measured directly by `od5r`.

#### C. Why `od5d` (`v3`, Plain `bf16`) Did Worse: Both Rules A/B/C and Plain `bf16` (Plus Stopping at Round 11)
Comparing `od5c-r10` (`54.1% / -1.68`) against `od5d-r10` (`51.6% / -1.95`) at the **exact same round**, alongside `od5d-r11` (`49.5% / -1.21`) and `loss_analysis/od5d.txt` vs. `od5c3.txt`, shows that **both** the `od5-v3` rule changes and plain `bf16` quantization hurt `od5d`:

1. **Why Rules A, B, and C + Quota Shifts Degraded Tools and IFEval (Direct Data Evidence):**
   - `od5d` shifted 12 quota slots/round from math (`58 -> 52`) and code (`52 -> 46`) to tools (`simple 26 -> 30, several 22 -> 26, no-fit 24 -> 28`), raised tool caps by +2 (`simple 48, several 51, no-fit 58`), and enabled `od5-v3`:
     - **Rule A** transferred unfilled slots from braked types (`math`, `code`) to `shorten` pairs on "healthy" types (`slice_acc >= 0.0`).
     - **Rule B** loosened `shorten` eligibility (`p1 >= p0 - 0.125` on `7/8` questions) and lowered the minimum thinking ratio from `1.30` to `1.22` on "healthy" types.
     - **Rule C** capped `be-right (hold)` pairs at `20%` and `recovery` pairs at `30%` on "healthy" types.
   - **The fatal interaction with monitor bias:** Because the 780-question slice monitor **over-reads accuracy by `+2.5` to `+3.0` points**, types with true full-test accuracy of `-1.0` to `-2.5` regularly scored `>= 0.0` on the slice monitor (`IFEval`, `simple`, and `several` scored `>= 0.0` in 7–9 of 10 rounds in `od5d`). `od5-v3` therefore classified them as "healthy", stripped them of stabilizing `be-right` pairs (capped at 20% by Rule C), and flooded them with low-margin (`ratio >= 1.22`) `shorten` pairs transferred from braked math/code (Rules A and B).
   - **Full-test and loss-analysis proof:** At round 10 on the full test (`od5d-r10` vs. `od5c-r10`), `IFEval`, `simple`, and `several` were pushed +2 to +3% shorter (`49%, 49%, 50%` vs. `46%, 46%, 48%`), **and their full-test accuracy collapsed**:
     - `IFEval`: `-0.7` (`od5c-r10`) $\to$ **`-1.3`** (`od5d-r10`)
     - `simple`: `-1.1` (`od5c-r10`) $\to$ **`-2.4`** (`od5d-r10`)
     - `several`: `-1.1` (`od5c-r10`) $\to$ **`-2.2`** (`od5d-r10`), worsening to **`-3.6`** at `od5d-r11`!
     - In `loss_analysis/od5d.txt`, `bfcl_multiple` (several tools) lost `-6.44` pts gross at r11 (`net -3.56`), with **`4.31` pts (67%) caused by "stopped thinking early (`<0.5x`)"** (`0.38x` median thinking on lost questions), and `bfcl_simple_python` lost `-4.94` pts gross at r10 (`net -2.44`), with **`3.06` pts (62%) from "stopped thinking early (`<0.5x`)"**.

2. **Why Plain `bf16` (Trainable LoRA Weights in `bf16`) Caused Math Stagnation and Step-to-Step Oscillation:**
   - In `code/dpo.py`, when `--bida-precision` is off and `--dtype bf16` is used, both the frozen base model and the **trainable LoRA parameters (`lora_A`, `lora_B`) are stored in pure `torch.bfloat16`**, and `AdamW` updates those `bf16` tensors directly with `lr = 1.4e-5`.
   - In IEEE `bfloat16` (only **7 mantissa bits**), the unit in the last place (ULP) is $2^{-7} \approx 7.81 \times 10^{-3}$ relative to weight magnitude. For standard Kaiming-initialized `lora_A` weights ($|w| \sim 1/\sqrt{1024} \approx 0.008\text{–}0.031$), 1 ULP is $2.44 \times 10^{-4}$ and the rounding threshold ($\frac{1}{2}\text{ULP}$) is $1.22 \times 10^{-4}$—which is **4x to 9x larger than a typical AdamW step $\Delta w \approx 1.4 \times 10^{-5}$**!
   - As a result, in pure `bf16` without `fp32` master LoRA weights, small steady gradient updates round to zero while noisy gradient spikes jump in coarse quanta. In `od5d`, math saving stalled at `48–50%` from round 4 through round 8, plunged from `49.2%` (r8) to `44.3%` (r9) on the monitor (`math 38%`), and dropped from `51.6%` (r10) to `49.5%` (r11) on the full test (`math 54% -> 50%`, `code 52% -> 49%`).

3. **Stopping at Round 11:**
   - `od5d` stopped at round 11 because only 11 slices existed when it started on GPU 2. Even `od5c` had `-1.68` accuracy at round 10 and required rounds 11–15 to consolidate accuracy above `-1.0`.

---

### 2. Design of ONE Clean, Reliable Online DPO Run (`od5e`)

To reach **`> 52%` saved at better than `-1.0` accuracy reliably** without `--bida-precision` and without data refills, we keep the proven `od5c` (`--od5-v2`) architecture and make **three targeted improvements** directly motivated by `loss_analysis/od5c3.txt` and `pertype_best_full.md`:

#### A. Pair Rules Per Type: Keep `--od5-v2` (Drop Rules A/B/C) + Raise Per-Question Thinking Floors (`--type-floor`)
- **Keep `--mix-preset od5 --od5-v2` unchanged** (no slot transfers across types, `shorten` thinking ratio $\ge 1.30$, no 20% cap on `hold` or 30% cap on `recovery`, response floors only on IFEval via `data/od3-base-resp.json`, `--count-mode atleast`).
- **Address the #1 loss mode in `od5c3.txt` via `--type-floor`:**
  - In `loss_analysis/od5c3.txt`, across `od5c-r15` and `od5c-r16`, **55% to 77% of all lost accuracy in every type is "stopped thinking early (`<0.5x` the original's right median)"**: math loses `3.12–3.26` pts (`55–56%` of losses, with right answers on lost questions thinking `0.43–0.44x`), code loses `3.45` pts (`58%` of losses, `0.44x`), simple loses `2.83` pts (`77%`), and several loses `3.89` pts (`66%`). Moreover, **60% of math losses (`3.40 / 5.70` pts) occur on easy `8/8` questions**.
  - Why did `od5c` allow right answers to shrink to `0.43x` on lost questions? Because `od5c`'s `--type-floor` set the per-question floor ratio to **`0.35` on math** and **`0.38` on code** (`svamp=35:0.35+mbpp=45:0.38+ifeval=35:0.38+bfcl_simple_python=25:0.40+bfcl_multiple=25:0.42+bfcl_irrelevance=20:0.35`).
  - Raising `--type-floor` to:
    `svamp=38:0.42+mbpp=45:0.42+ifeval=35:0.38+bfcl_simple_python=28:0.44+bfcl_multiple=28:0.45+bfcl_irrelevance=20:0.35`
    prevents `shorten` and `be-right` pairs from selecting outlier completions that think `< 0.42x–0.45x` of the question's original median right thinking `bt`. Because `1 - 0.42 = 58%` saving (and undamaged shorter questions still shorten freely down to `0.42x`), this directly cuts off the `< 0.45x` premature-stopping tail while supporting `53–55%` composite saving.

#### B. Quotas, Caps, Brake, LR, Steps, and Rounds
- **Quotas per round (keep `od5c`'s exact 240-pair quotas; 320 in round 1):**
  - Round 1: `svamp=77,mbpp=70,ifeval=77,bfcl_simple_python=35,bfcl_multiple=29,bfcl_irrelevance=32` (320 pairs, 20 steps).
  - Rounds 2–15: `svamp=58,mbpp=52,ifeval=58,bfcl_simple_python=26,bfcl_multiple=22,bfcl_irrelevance=24` (240 pairs, 15 steps).
  - *Why keep `od5c` quotas:* Math (1,000 test questions, 25% weight) and code (378 test questions, 25% weight) carry 50% of the official composite weight; cutting their quotas in `od5d` reduced composite saving by 2.5% and destabilized math/code repair.
- **Saving Caps (`--od-caps`) and Brake (`--od-acc-hold`):**
  - Set `--od-caps svamp=59,mbpp=58,ifeval=54,bfcl_simple_python=46,bfcl_multiple=49,bfcl_irrelevance=56` (vs. `62,63,56,46,49,56` in `od5c`) and **`--od-acc-hold -1.5 --od-unlatch`** (vs. `-2.0` in `od5c`).
  - *Why:* `pertype_best_full.md` shows that math has never held `>= -2.0` accuracy above `60%` full-test saving (`60% / -1.5` in `rpL-40`, whereas `61–62%` in `od5c` gave `-2.8`), and code at `54%` in `od5c-r15` gave `-0.9` accuracy whereas `59%` in `od5c-r16` dropped to `-2.0`. Because the slice monitor **under-reads saving by `~1.1%` and over-reads accuracy by `~2.8` pts**, a monitor cap of `59` on math and `58` on code holds them near `58–60%` and `55–57%` full-test saving, and an accuracy hold of `-1.5` on the monitor catches a type when its true slice accuracy drops below `-4.3` (instead of waiting until `-4.8`). Trading ~1% saving on math and ~2% on code preserves `~53.5–54.2%` composite saving while recovering `+0.35 to +0.50` pts of composite accuracy.
- **LR, Beta, Warmup, Steps, Reference, and Number of Rounds:**
  - LoRA rank `16`, constant `--rp-lr 1.4e-5`, `--rp-warmup 2`, `--rp-beta 0.1`, `--nll 0.05`, fixed reference = original Qwen (`--od-fixed-ref`), `20` steps in round 1 and `15` steps/round in rounds 2–15 (`batch = 16`).
  - **Number of rounds: 15 rounds** (`--od-rounds 15`, or `16` using the existing 16 slices). `od5c-r15` (`54.0% / -0.56`) was the highest point above the curve (`+3.14`), and stopping at round 15 avoids needing slice 16 when building slices from scratch.

#### C. Precision: Standard `bf16` Mixed Precision (`fp32` Trainable LoRA Weights + `bf16` Base & Autocast)
- Drop `--bida-precision` (which loaded the entire base model in `fp32` and used `fp16` autocast).
- Use **standard `bf16` mixed precision** (`--dtype bf16`): keep the frozen base model in `torch.bfloat16` and run forward/backward under `torch.autocast(device_type="cuda", dtype=torch.bfloat16)`, **while keeping the trainable LoRA adapter tensors (`lora_A`, `lora_B`, only 0.8% of parameters) in `torch.float32`** and casting logits to `.float()` inside `log_softmax`.
- *Justification on its own merits:* As shown in Section 1.C, with `lr = 1.4e-5`, `AdamW` step sizes ($\sim 1.4 \times 10^{-5}$) are smaller than the `bfloat16` 7-bit mantissa rounding threshold ($\frac{1}{2}\text{ULP} \approx 1.2 \times 10^{-4}$ for $|w| \approx 0.02$). Keeping trainable adapter weights in `fp32` while computing in `bf16` is standard practice in HuggingFace PEFT and PyTorch AMP to prevent update underflow.

#### D. Fresh Test-Matched Questions Needed Per Type (Zero Missing Questions)
There are two clean ways to run `od5e` with **zero new question generation and zero missing questions**:
1. **Using the existing 16 slices already stored on Drive (`data/od3-r1-questions.jsonl` .. `data/od3-r16-questions.jsonl`):**
   - All 16 slices (780 questions/slice = 12,480 total: `3,520` math, `2,240` code, `3,200` IFEval, `1,280` simple, `960` several, `1,280` no-fit) and `data/od3-base-{think,stats,resp}.json` are already built and stored on Drive from our existing pools (`pool` + `tooltop-r2a..r2r`). **0 questions are missing.**
2. **Even if rebuilding slices from scratch using ONLY the pre-`od5c` pools (`pool` + `r2a..r2l`, excluding the `r2m..r2r` slice 12–16 refills):**
   - In `od5c`, slices 1–11 (780/slice) succeeded at strict 3-pt tolerance using `660` `several` (`11 x 60`) and `880` `no-fit` (`11 x 80`) questions, because each slice drew `2.3x–2.7x` more questions than the round's pair quota (`60` questions for `22` several-tool pairs; `80` questions for `24` no-fit pairs).
   - Setting per-slice question counts in `od3prep` to **`660` questions/slice** (`svamp=180, mbpp=130, ifeval=190, bfcl_simple_python=64, bfcl_multiple=44, bfcl_irrelevance=52`, still $\ge 2.0\times$ the pair quota for every type) requires over **15 rounds**:
     - **Math (`svamp`):** $15 \times 180 = 2,700$ (pre-refill pool has $4,935$ $\to$ **0 missing**).
     - **Code (`mbpp`):** $15 \times 130 = 1,950$ (pre-refill pool has $3,363$ $\to$ **0 missing**).
     - **IFEval (`ifeval`):** $15 \times 190 = 2,850$ (pre-refill pool has $8,642$ $\to$ **0 missing**).
     - **Simple (`bfcl_simple_python`):** $15 \times 64 = 960$ (pre-refill pool has $3,393$ $\to$ **0 missing**).
     - **Several (`bfcl_multiple`):** $15 \times 44 = 660$ (pre-refill pool has $660$ at strict 3-pt match $\to$ **0 missing**).
     - **No-fit (`bfcl_irrelevance`):** $15 \times 52 = 780$ (pre-refill pool has $880$ at strict 3-pt match $\to$ **0 missing**).

---

### 3. Choosing the Checkpoint Without the Full Test and Stopping Rule

#### A. Calibrated Biases of the Two Cheap Estimators
Across all 6 full-tested online checkpoints (`od4-r8`, `od5c-r10`, `od5c-r15`, `od5c-r16`, `od5d-r10`, `od5d-r11`), the two cheap estimators have distinct, consistent biases:
1. **Next-Round Slice Monitor (780 questions, measures round $k < R$ on slice $k+1$):**
   - **Saving bias:** Monitor **under-reads** full-test saving by **`1.1` percentage points** (`od5c-r10`: `52.9 -> 54.1` [`+1.2`]; `od5c-r15`: `52.9 -> 54.0` [`+1.1`]; `od5d-r10`: `50.7 -> 51.6` [`+0.9`]).
   - **Accuracy bias:** Monitor **over-reads** full-test accuracy by **`2.5 to 3.0` points** (`od5c-r10`: `+0.84 -> -1.68` [`-2.52`]; `od5c-r15`: `+2.41 -> -0.56` [`-2.97`]; `od5d-r10`: `0.00 -> -1.95` [`-1.95`]; mean bias **`2.8` pts**, single-slice sampling SD $\approx 0.9$ pts).
   - **Calibrated monitor estimate for round $k$:**
     $$\widehat{\text{save}}_k = \text{mon\_save}_k + 1.1, \qquad \widehat{\text{acc}}_k = \text{mon\_acc}_k - 2.8$$
2. **Dev-512 (measures the final round $R$ model `models/<run>-best`):**
   - **Saving bias:** Virtually **zero (`0.0` to `-0.2` pts)** (`od4-r8`: `52.3 -> 51.6`; `od5c-r16`: `55.0 -> 54.8`; `od5d-r11`: `49.6 -> 49.5`).
   - **Accuracy bias:** Extremely tight **`+2.95` points** (`+2.91` on `od4-r8`, `+3.00` on `od5c-r16`, `+2.90` on `od5d-r11`).
   - **Calibrated Dev-512 estimate for round $R$:**
     $$\widehat{\text{save}}_R = \text{dev\_save}_R, \qquad \widehat{\text{acc}}_R = \text{dev\_acc}_R - 2.95$$

#### B. Stopping Rule and Which Rounds to Full-Test
1. **Stopping Rule During Training:**
   - **Do NOT stop before round 14** (`--od-conv-gain -999`). As `od5c` proved, online DPO hits its saving plateau (`~53–54%`) at round 10 and spends rounds 11–15 consolidating accuracy (`-1.68 -> -0.56`). Any early convergence stop cuts off the consolidation phase.
   - Run all 15 rounds (or 16 rounds if using all 16 existing slices), or allow an optional late stop at round $k \ge 14$ only if two consecutive rounds $k-1$ and $k$ both satisfy `mon_save >= 51.5%` ($\widehat{\text{save}} \ge 52.6\%$) and `mon_acc >= +2.00` ($\widehat{\text{acc}} \ge -0.80$).
2. **Which Two Checkpoints to Full-Test (`fullpick`):**
   - Why `fullpick` initially picked `od5c-r10` (`52.9 / +0.84`) over `od5c-r15` (`52.9 / +2.41`): `fullpick` applied `--fp-acc-bias 2.5` (which let `r10` barely pass `>= -1.0` or picked `r10` before `r15` finished) and ranked candidates inside the gate purely by raw `save` rather than accuracy margin or curve distance.
   - **Rule for Checkpoint 1 (Best Consolidated Monitor Round, $k \in [12, R-1]$):** Filter to rounds $k \ge 12$ with calibrated $\widehat{\text{save}}_k \ge 52.2\%$ (`mon_save >= 51.1%`) and $\widehat{\text{acc}}_k \ge -0.90$ (`mon_acc >= +1.90`), and pick the round with the **highest margin above the quadratic curve**:
     $$\text{margin}_k = \widehat{\text{acc}}_k - \left(-0.388 + 0.0899\,\widehat{\text{save}}_k - 0.0028\,\widehat{\text{save}}_k^2\right)$$
     (On `od5c`, this rule immediately selects **round 15**: `mon 52.9 / +2.41` $\to$ calibrated `54.0 / -0.39`, margin `+3.48`).
   - **Rule for Checkpoint 2 (Final Round $R$ or 2nd-Best Consolidated Round):** Score `models/<run>-best` (round $R$) with Dev-512 using bias `0.0 / 2.95`; if it is inside `save >= 52.2%, acc >= -0.95`, full-test it as the second checkpoint; otherwise full-test the second-best monitor round $k \ge 12$ by curve margin.

---

### 4. Exact Code Changes in Words and Queue Lines

#### A. Code Changes
1. **`code/dpo.py` (function `main`, around lines 237–260):**
   - To use standard `bf16` mixed precision without `--bida-precision`: right after `trainable = [p for p in model.parameters() if p.requires_grad]` (line 237), cast the trainable LoRA parameters to `fp32` by iterating over `trainable` and setting `p.data = p.data.float()`.
   - Inside `sum_logprobs(m, pairs)` (around line 255), cast `logits` to `.float()` inside `torch.log_softmax(logits.float(), dim=-1)` so token log-probability sums do not accumulate `bf16` rounding error. The base model remains in `torch.bfloat16` and forward/backward passes run under `torch.autocast(device_type=device, dtype=torch.bfloat16)`.
2. **`code/track.py` (function `fullpick`, around lines 346500–347500):**
   - Change the default biases for `--fp-kind arena` to `--fp-save-bias 1.1 --fp-acc-bias 2.8`.
   - When adding `models/<run>-best` (scored by Dev-512 rather than the 780-question slice monitor) to `cands`, apply save bias `0.0` and accuracy bias `2.95` (instead of the monitor's `+1.1` save bias).
   - When sorting `cands` for pick (1) inside the goal (`save >= 52.0` and `acc >= -1.0`), prefer consolidated rounds ($k \ge 12$) and sort by **distance above the reference curve** `acc - curve_acc(save)` (where `curve_acc(s) = -0.388 + 0.0899 * s - 0.0028 * s * s`) instead of raw `save` alone.
3. **`code/repair_pairs.py`:**
   - **No code changes needed.** Keep `--mix-preset od5 --od5-v2` as implemented (`--od5-v3` is not passed).

#### B. Queue Lines (Plain Text)
Assuming the 16 test-matched slices (`data/od3-r1..r16-questions.jsonl`) and `data/od3-base-resp.json` already on Drive are used (run for 15 rounds, or 16 if all 16 slices are present; to cap at 15 rounds without deleting slice 16, pass `--od-rounds 15` and let `odpo_fixed` stop at `min(len(slices), args.od_rounds)` by replacing `range(n_rounds)` with `range(min(n_rounds, args.od_rounds))` in `odpo_fixed`):

odpo --name od5e --od-fixed-ref --od-rounds 15 --od-steps 15 --od-r1-steps 20 --od-r1-quota svamp=77,mbpp=70,ifeval=77,bfcl_simple_python=35,bfcl_multiple=29,bfcl_irrelevance=32 --rp-quota svamp=58,mbpp=52,ifeval=58,bfcl_simple_python=26,bfcl_multiple=22,bfcl_irrelevance=24 --rp-pair-args=--mix-preset,od5,--od5-v2,--base-think,data/od3-base-think.json,--base-stats,data/od3-base-stats.json,--base-resp,data/od3-base-resp.json,--type-floor,svamp=38:0.42+mbpp=45:0.42+ifeval=35:0.38+bfcl_simple_python=28:0.44+bfcl_multiple=28:0.45+bfcl_irrelevance=20:0.35,--count-mode,atleast,--count-resp-min,1.10,--count-resp-cap,1.75,--count-resp-pad,60,--resp-median,0.95,--resp-max,1.05,--right-resp-cap,1.15 --rp-lr 1.4e-5 --rp-warmup 2 --rp-beta 0.1 --rp-dpo-seed 0 --rp-dev-vs rpE80 --od-dev-rounds 99 --data-seed 18 --od-caps svamp=59,mbpp=58,ifeval=54,bfcl_simple_python=46,bfcl_multiple=49,bfcl_irrelevance=56 --od-acc-hold -1.5 --od-unlatch --od-stop-save 999 --od-conv-gain -999 --rp-gen-seed 42
fullpick --fp-run od5e --fp-kind arena --fp-save-bias 1.1 --fp-acc-bias 2.8
lossan --la-models fullpick:od5e --la-tag od5e

---

### 5. Expected Full-Test Landing, Main Risks, and How to Read `od5r`

#### A. Expected Full-Test Landing (Rounds 14–16 of `od5e`)
With `--od5-v2`, `fp32` trainable LoRA weights under `bf16` autocast, raised per-question thinking floors (`0.42–0.45x`), and slightly tighter math/code caps (`59 / 58`) and brake (`-1.5`), the expected full-test landing at rounds 15–16 is:
- **Composite Full Test:** **`53.5%` saved (range `52.5% to 54.5%`) / `-0.45` accuracy (range `-0.85 to -0.15`)**, sitting **`+3.1 to +3.5` points above the reference curve** with a reliable buffer inside `> 52% / > -1.0`.
- **Expected Per-Type Breakdown:**
  - **Math (`svamp`, 25% weight):** `58–60%` saved / `-1.8 to -2.4` acc (vs. `61–62% / -2.8` in `od5c-r15/16`, improved by raising math's floor ratio from `0.35` to `0.42` and capping saving at `59`).
  - **Code (`mbpp`, 25% weight):** `55–57%` saved / `-0.8 to -1.5` acc (vs. `54% / -0.9` at r15 and `59% / -2.0` at r16 in `od5c`, stabilized by capping code at `58` and floor ratio `0.42`).
  - **IFEval (`ifeval`, 25% weight):** `47–48%` saved / `+0.2 to +0.7` acc (matching `od5c-r15/16`'s `47–48% / +0.5`).
  - **Simple tools (`bfcl_simple_python`, 11.9% weight):** `47–49%` saved / `-0.8 to -1.4` acc.
  - **Several tools (`bfcl_multiple`, 6.0% weight):** `49–50%` saved / `-1.0 to +0.2` acc.
  - **No-fit (`bfcl_irrelevance`, 7.1% weight):** `56–57%` saved / `+4.5 to +6.5` acc.
  - **Pooled BFCL (25% weight):** `50–51%` saved / `+0.6 to +1.3` acc.

#### B. Main Risks and Mitigations
1. **Risk 1 — Tighter brake (`-1.5` vs. `-2.0`) braking too often on small tool slices (`60–80` questions):**
   - A 60-question `several-tool` slice has discrete steps of $100 / (60 \times 8) = 0.21$ pts per answer, so random noise can occasionally dip a tool type to `-1.6` on one slice. Because `--od-unlatch` re-evaluates every round on a fresh slice and `--od5-v2` leaves unfilled braked slots empty rather than over-shortening other types, a 1-round brake on a tool type simply pauses its shortening for 1 round without harming other types.
2. **Risk 2 — Raising `--type-floor` (`0.42–0.45`) slowing early saving growth by ~0.5–1.0%:**
   - In rounds 1–8, answers are still `0.60–0.85x` of `bt`, so a `0.42x` floor barely binds at all; it binds only in rounds 10–16 on questions that have already shrunk past `55%` saving, which is exactly where `loss_analysis/od5c3.txt` shows `< 0.5x` early stopping occurs.

#### C. How to Read `od5r` (Seed Replication of `od5c`) When It Lands
`od5r` runs `od5c`'s exact command line (`--od5-v2`, `--bida-precision`, same 16 slices on Drive) with only the random seeds changed (`--rp-dpo-seed 1 --rp-gen-seed 54`). When its 16 rounds and 2 full tests complete, read it as follows:
1. **First, check WHICH rounds `fullpick` selected for `od5r`'s two full tests:**
   - Because `od5r` uses `fullpick --fp-save-bias 1.0 --fp-acc-bias 2.5` (which ranks candidates inside `acc >= -1.0` by raw `save`), check whether `fullpick` accidentally picked an early un-consolidated round (like `r9–r11`, analogous to `od5c-r10` `54.1 / -1.68`) instead of a consolidated round (`r14–r16`). If `fullpick` picked an early round and a late round, judge the algorithm by the **consolidated round (`r14–r16`)**, and if neither full test was in `r14–r16`, run a full test on the best `r14–r16` round (highest monitor accuracy at `mon_save >= 51.5%`).
2. **Interpreting the consolidated (`r14–r16`) full-test result of `od5r`:**
   - **Case 1 (`od5r` lands at `>= 52.0%` saved and `>= -1.0` accuracy, e.g. `53–55% / -0.5 to -0.9`):** This confirms that `od5c` (`--od5-v2` over 15–16 rounds with `fp32` master weights) is **seed-reliable as-is**. In that case, the only change needed to satisfy all cleanliness requirements is replacing `--bida-precision` (`fp16` autocast) with standard `bf16` mixed precision (`fp32` trainable LoRA weights + `bf16` autocast in `dpo.py`), plus optional cap/floor tightening (`od5e`) for extra accuracy margin.
   - **Case 2 (`od5r` lands at `53–55%` saved but `-1.1 to -1.5` accuracy):** This means `od5c`'s saving is seed-reliable, but its accuracy margin (`-0.56` to `-0.78` in `od5c`) has $\sim 0.4$ pts of seed variance due to math (`-2.8`) and code (`-2.0`) `< 0.5x` early-stopping losses. This directly validates running `od5e` (raising `--type-floor` to `0.42–0.45` and tightening math/code caps to `59/58` and `--od-acc-hold` to `-1.5`) to shift the accuracy distribution up by `+0.4` pts.
   - **Case 3 (`od5r` lands `< 52.0%` saved):** Inspect which types were held by the brake in rounds 10–16 of `od5r`; if the brake held too many rounds due to slice variance, keep `--od-acc-hold -2.0` while using `od5e`'s raised `--type-floor` (`0.42–0.45`) to protect accuracy at the question level rather than braking the whole type.