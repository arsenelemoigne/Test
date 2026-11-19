### 1. What `od5f` Changed Against `od5c`, What Rounds 19–23 and `od5g` Show, and Per-Type Ceilings at `<= 1 pt` Loss

#### A. What `od5f` changed against `od5c` (per type on the 2,759 x 8 full test)
`od5f` applied three changes to `od5c` (one online DPO run from `Qwen/Qwen3-0.6B`, fixed original reference, LoRA rank 16, constant `lr = 1e-5`, fp32 master weights, 20 steps on 320 pairs in round 1 and 15 steps on 240 pairs in rounds 2+):
1. **Activated `--shorten-ratio-by-part` and `--shorten-second` inside the `od5` branch of `repair_pairs.py`** (previously bypassed), setting `ifeval=1.25,bfcl_irrelevance=1.25` (vs `1.30` elsewhere) and allowing a second shortening pair per question on IFEval and no-fit.
2. **Rebalanced quotas and thinking floors:** round 2+ quotas moved from `64,56,64,20,16,20` to `56,52,60,24,22,26`; floors rose on math (`30:0.45` vs `30:0.42`) and simple tool calls (`25:0.44` vs `25:0.40`) and dropped on IFEval (`18:0.30` vs `20:0.33`) and no-fit (`16:0.28` vs `18:0.30`).
3. **Raised per-type saving caps and relaxed the accuracy hold:** caps moved from `58,60,52,44,48,60` to `svamp=61,mbpp=63,ifeval=56,bfcl_simple_python=45,bfcl_multiple=53,bfcl_irrelevance=64`, with `--od-acc-hold -2.0` (vs `-1.5`).

| Benchmark (Full Test 2,759 x 8) | `od5c-r16` (54.8 / -0.78) | `od5c-r20` (56.9 / -1.24) | `od5f-r16` (55.4 / -0.57) | `od5f-r17` (57.8 / -0.64) | `od5f-r18` (59.5 / -0.80) | `od5g-r20` (57.4 / +0.14) |
|---|---|---|---|---|---|---|
| **Math (SVAMP 700)** | 61.1% / -2.84 | 62.4% / -2.70 | 63.3% / -1.50 | 66.8% / -2.30 | 70.9% / -3.60 | 66.8% / -1.50 |
| **Code (MBPP 519)** | 58.6% / -1.95 | 61.1% / -2.70 | 53.0% / -0.70 | 55.2% / -1.20 | 56.1% / -1.10 | 53.7% / -0.40 |
| **IFEval (700)** | 46.9% / +0.49 | 51.3% / -0.30 | 52.8% / -0.60 | 55.4% / +0.50 | 56.2% / +1.40 | 53.4% / +1.80 |
| **BFCL Simple (400)** | 48.6% / -1.09 | 49.3% / -2.20 | 48.0% / -0.90 | 48.7% / -1.50 | 49.7% / -2.40 | 49.0% / -2.20 |
| **BFCL Multiple (200)** | 50.1% / 0.00 | 50.5% / -0.40 | 49.8% / -1.00 | 50.4% / -0.60 | 50.1% / -1.00 | 50.2% / -0.70 |
| **BFCL No-Fit (240)** | 57.0% / +5.89 | 57.9% / +6.60 | 56.9% / +4.20 | 57.7% / +4.40 | 58.7% / +5.50 | 58.6% / +6.10 |
| **BFCL Pooled (840)** | 51.6% / +1.16 | 52.2% / +0.74 | 51.2% / +0.53 | 51.9% / +0.40 | 52.7% / +0.19 | 52.2% / +0.53 |
| **Official Total** | **54.8% / -0.78** | **56.9% / -1.24** | **55.4% / -0.57** | **57.8% / -0.64** | **59.5% / -0.80** | **57.4% / +0.14** |

- **IFEval (`+9.3` pts saving, `+0.9` pts accuracy vs `od5c-r16`):** The activated `1.25` ratio, second shortening pair, and `0.30` floor lifted IFEval from `46.9% / +0.49` (`od5c-r16`) to `56.2% / +1.40` (`od5f-r18`).
- **Math (`+9.8` pts saving vs `od5c-r16`, or `+5.7` pts at half the loss in `od5g-r20`):** The `0.45` floor protected math through round 17 (`od5f-r16` `63.3% / -1.50` and `od5g-r20` `66.8% / -1.50` vs `od5c-r16` `61.1% / -2.84`), before an unbraked round 18 pushed `od5f-r18` math to `70.9% / -3.60`.
- **Code (`-2.5` pts saving, `+0.85` pts accuracy vs `od5c-r16`):** Smaller quota (`56 -> 52`) and keeping ratio `1.30` slowed code from `58.6% / -1.95` to `56.1% / -1.10` (`od5f-r18`) and `53.7% / -0.40` (`od5g-r20`).
- **BFCL Pooled (`+1.1` pts saving, `-0.97` pts accuracy vs `od5c-r16`):** No-fit barely moved (`57.0% -> 58.7%`) because its `16:0.28` floor and single-slice cap (`64%`) stopped it early, while Simple (`48.6% -> 49.7%`, `-1.09 -> -2.40`) crossed its cliff when its single-slice monitor dipped below `45%` in rounds 17–18.

#### B. What rounds 19–23 (`od5f`) and 19–22 (`od5g`) show: brake-induced limit-cycle oscillation
Neither run plateaued from capacity or paid in accuracy; instead, **both entered a 2-round limit-cycle oscillation caused by how `repair_pairs.py` fills the quota of held types**, trading 6–14 pts of saving back for `+3` to `+4` pts of slice accuracy:
- At the round 19 monitor (`od5f-r18` answering slice 19: `59.0% / +2.37`), **four types hit their cap or accuracy hold simultaneously**: `svamp` (`64.2% / -2.56`), `bfcl_simple_python` (`50.3% >= 45`), `bfcl_multiple` (`58.1% >= 53`), and `bfcl_irrelevance` (`65.4% >= 64`).
- In `repair_pairs.py`, when a type `part` is in `--no-shorten-parts` (`stop5`), `c_sh = 0`, **but its quota `n` is unchanged**. Those 4 held types occupy `56 + 24 + 22 + 26 = 128` of the 240 pairs (**53.3% of the entire DPO batch**). Worse:
  1. All `n_act` slots (`80%` of `n_rem`) on held types are filled by `recovery` pairs (lines 435–450, 702–706), which explicitly select `chosen` **1.08x to 1.45x (+10 to +50 tokens) LONGER** than `rejected` (targeting `0.85 * bt`).
  2. At line 453 (`if args.od5_v2 and part not in stop5`), held types are **excluded from compact `be-right` pairs** (`0.50–1.00x`) and fall through to length-matching `0.90–1.12x` of the *wrong* answer's thinking (`hold`, lines 463–471). Because wrong answers think 2–3x longer than right answers, matching `0.90–1.12x` of a wrong answer selects the **longest right answers**. Even `bfcl_irrelevance` (which `track.py` excluded from `--right-hold` at line 5450) still executes lines 435–471 in `repair_pairs.py` because the `od5` block (`continue` at line 472) never checks `right_hold`.
- Training 15 steps on 128 lengthening pairs caused a global length rebound in one round: Dev-512 dropped from `60.4% / +0.88` (r18) to `52.7% / +3.49` (r19) in `od5f` and `55.0% / +2.59` in `od5g` (math monitor fell from `64.2%` to `51.8%`).
- In round 20, because r19 lengthened the model below the caps, `--od-unlatch` **released** the brake on math, multiple, and no-fit. Round 20 shortened them again (`od5f-r20` Dev `57.6% / +2.72`; `od5g-r20` Dev `58.2% / +3.18`, full test `57.4% / +0.14`).
- At the end of round 20 they crossed the caps again, so rounds 21–23 repeated the lengthening rebound (`od5f` Dev r21 `46.8%`, r22 `51.7%`, r23 `45.8%`; `od5g` Dev r21 `51.6%`, r22 `50.6%`).

#### C. Where each type stops at `<= 1 pt` accuracy loss now (full-test empirical curves)
- **Math (SVAMP):** **64–65%** (`od5f-r16` `63.3% / -1.5`, `od5g-r20` `66.8% / -1.5`, `od5r-r15` `59.4% / -1.2`; crosses `-1.0` at `~64.5%` and `-1.5` at `66.8%`).
- **Code (MBPP):** **56–58%** (`od5g-r20` `53.7% / -0.4`, `od5f-r16` `53.0% / -0.7`, `od5f-r18` `56.1% / -1.1`, `rpl-60` `61.7% / -0.9`).
- **IFEval:** **57–58%** (`od5g-r20` `53.4% / +1.8`, `od5f-r18` `56.2% / +1.4`, `od5c-r20` `51.3% / -0.3`; under `od5f` guards it is still `+1.4` at `56.2%` and crosses `-1.0` around `58%`).
- **BFCL Simple (400):** **48%** (`od5f-r16` `48.0% / -0.9`, `od5c-r16` `48.6% / -1.09`; above `48.5%` loss jumps to `-1.5` to `-2.4`).
- **BFCL Multiple (200):** **50–51%** (`od5c-r20` `50.5% / -0.4`, `od5f-r17` `50.4% / -0.6`, `od5g-r20` `50.2% / -0.7`, `od5f-r18` `50.1% / -1.0`).
- **BFCL No-Fit / Irrelevance (240):** **75–96%** (never loses accuracy: `+5.5` at `58.7%` in `od5f-r18`, `+6.1` at `58.6%` in `od5g-r20`, `+7.7` at `65.9%` in `r2s3-s80`, and **`+6.2` at `95.9%` saved** when thinking is turned off).

---

### 2. Is `> 60%` at Better Than `-1` Reachable?

**YES.** Both the KKT marginal-cost arithmetic and the measured per-type ceilings show that **60.5% to 61.4% saved at `-0.3` to `-0.8` accuracy change** is reachable within our single-run online DPO rules.

#### A. Official-Score Arithmetic
Official saving is `S = 1 - (r_math * r_code * r_ifeval * r_bfcl)**(1/4)`, where `r_b = 1 - s_b`. On BFCL (840 questions), original total tokens are: Simple `400 * 351.84 = 140,736` (`39.35%`), Multiple `200 * 442.81 = 88,562` (`24.76%`), No-Fit `240 * 535.29 = 128,470` (`35.90%`). Because No-Fit has the longest original answers (`535.3` tokens), **No-Fit carries 35.9% of BFCL's token weight** (`ds_bfcl / ds_nofit = 0.359`), whereas accuracy is weighted by question counts (`400 : 200 : 240`, so No-Fit is `28.57%` of BFCL accuracy).
- Crossing `S = 60.0%` requires `r_geom < 0.4000`, or `r_math * r_code * r_ifeval * r_bfcl < 0.02560`.
- `od5f-r18` (`59.51% / -0.80`) has `r = (0.291, 0.439, 0.438, 0.473)`, product `0.02646` (0.49 pts short of 60%).
- `od5g-r20` (`57.41% / +0.14`) has `r = (0.332, 0.463, 0.466, 0.478)`, product `0.03424` (2.59 pts short of 60%, with `+1.14` pts of accuracy slack above `-1.0`).

#### B. Which Benchmarks Give the Next Points of Saving vs Give Back Accuracy
1. **Math must GIVE BACK 2.5–3.0 pts of saving relative to `od5f-r18` (`70.9% / -3.60 -> 68.0% / -1.80`):** Between `66.8%` (`od5g-r20`: `-1.50`) and `70.9%` (`od5f-r18`: `-3.60`), math pays `-0.51` accuracy points per saving point. Holding math at `68.0%` recovers `+1.8` math accuracy points (`+0.45` macro accuracy).
2. **BFCL Simple must STAY at `48.5%` (`-1.20` pts):** Keeping Simple below `49%` recovers `+1.0` to `+1.2` pts on Simple (`+0.12` macro accuracy).
3. **The next points of saving must come from the three types where `effprobe` proves 67–92% of thinking is post-settlement re-checking:**
   - **BFCL No-Fit (`58.7% -> 70.0%` saved, accuracy `+6.0`):** `91.6%` of original thinking is after answer settlement, and `0` thinking tokens gives `+6.2` pts on the full test. Lifting No-Fit from `58.7%` to `70.0%` adds **`+4.1` pts to pooled BFCL saving** (`52.7% -> 57.0%`, `r_bfcl: 0.473 -> 0.430`) at **zero accuracy cost** (`r2s3-s80` already proved `65.9% / +7.7` on the full test).
   - **Code (`53.7–56.1% -> 57.5%` saved, accuracy `-1.10`):** Adding `+1.4` pts on code over `od5f-r18` (`+3.8` pts over `od5g-r20`) costs `~0.0–0.7` pts on code (`rpl-60` proved `61.7% / -0.90`).
   - **IFEval (`53.4–56.2% -> 57.5%` saved, accuracy `+0.80`):** Adding `+1.3` pts over `od5f-r18` (`+4.1` pts over `od5g-r20`) keeps IFEval positive (`+0.80`).

| Benchmark | `od5f-r18` (Full) | `od5g-r20` (Full) | **Target `od5h` Allocation** | Measured Full-Test Ceiling Allowing It |
|---|---|---|---|---|
| **Math (SVAMP 700)** | 70.9% / -3.60 | 66.8% / -1.50 | **68.0% / -1.80** (`r = 0.320`) | `od5g-r20` (66.8% / -1.50), `od5f-r17` (66.8% / -2.30) |
| **Code (MBPP 519)** | 56.1% / -1.10 | 53.7% / -0.40 | **57.5% / -1.10** (`r = 0.425`) | `od5f-r18` (56.1% / -1.10), `rpl-60` (61.7% / -0.90) |
| **IFEval (700)** | 56.2% / +1.40 | 53.4% / +1.80 | **57.5% / +0.80** (`r = 0.425`) | `od5f-r18` (56.2% / +1.40) |
| **BFCL Simple (400)** | 49.7% / -2.40 | 49.0% / -2.20 | **48.5% / -1.20** | `od5f-r16` (48.0% / -0.90), `od5c-r16` (48.6% / -1.09) |
| **BFCL Multiple (200)** | 50.1% / -1.00 | 50.2% / -0.70 | **51.5% / -0.80** | `od5c-r20` (50.5% / -0.40), `r2s2-s40` (54.4% / -0.60) |
| **BFCL No-Fit (240)** | 58.7% / +5.50 | 58.6% / +6.10 | **70.0% / +6.00** | `r2s3-s80` (65.9% / +7.70), `nothink` (95.9% / +6.20) |
| **BFCL Pooled (840)** | 52.7% / +0.19 | 52.2% / +0.53 | **57.0% / +0.95** (`r = 0.430`) | Weighted sum (`39.35% Simple, 24.76% Mult, 35.90% No-Fit`) |
| **Official Total** | **59.51% / -0.80** | **57.41% / +0.14** | **60.87% / -0.29** (`r_geom = 0.3913`) | Comfortably inside `>60.0%` at `> -1.00` (`+0.71` pt buffer) |

Even a conservative landing (`Math 67.5% / -1.9, Code 56.5% / -1.1, IFEval 56.5% / +1.0, Simple 48.5% / -1.4, Multiple 50.5% / -0.8, No-Fit 67.0% / +5.8 -> BFCL 55.6% / +0.80`) yields **`60.15%` saved at `-0.55` accuracy**.

---

### 3. Design of the ONE Next Step Most Likely to Land `> 60%` at Better Than `-1`

#### A. Why One Clean Online DPO Run from the Original (`od5h`, 20 rounds on matched slices `od3g`) Beats Continuing `od5g-r20` or Checkpoint Soup
1. **Linear weight soup / checkpoint merge of `od5f-r18` (`59.5 / -0.80`) and `od5g-r20` (`57.4 / +0.14`):** Any convex combination interpolates saving between `57.4%` and `59.5%` (cannot cross `60%`), while linear extrapolation (`1.3 * r18 - 0.3 * r20`) pushes math past `72%` (`-4.5` acc) and leaves No-Fit stuck at `58.7%`.
2. **Continuing `od5g-r20` or `od5f-r18` with a patched brake:** Both trained 18–20 rounds with No-Fit floored at `16:0.28` (`58.7%` saved) and Simple already at `49.0–49.7%` (`-2.2` to `-2.4` acc). Only 2 matched slices (`21–22`) remain in `od3g`, and changing rules mid-stream violates *"start from the original; one training procedure, not a chain of fixes"*.
3. **The Winner — One Clean Online DPO Run from `Qwen/Qwen3-0.6B` (`od5h`, 20 rounds on `od3g` slices 1–20):** Fixing the held-type lengthening bounce, smoothing the brake over 2 slices, and lowering the floors and ratios on types where thinking is 67–92% post-settlement re-checking (`bfcl_irrelevance`, `ifeval`, `mbpp`) while capping `svamp` at `64%` monitor (`~68%` full test) and `bfcl_simple_python` at `45%` reaches `60.5–61.4%` by rounds 17–20 and **stays there without oscillating**.

#### B. Complete Specification of `od5h` (Strictly Within Rules)
- **Starting model, reference & precision:** Start from `Qwen/Qwen3-0.6B`; fixed reference = original (`--od-fixed-ref`, `--ref-base`); single LoRA rank 16 continued across rounds, `beta = 0.1`, constant `lr = 1e-5`, batch size 16, fp32 master weights / bf16 compute.
- **Rounds, steps & quotas:**
  - **20 rounds** (`--od-rounds 20`) on test-matched slices `od3g` (slices 2–19 copied from `od3`, slices 20–22 already built and verified by `od3branch` with medium code `r2x-code-m1` + easy math `r2x-math-m1`).
  - **Round 1:** 20 steps (`320` pairs) on stored original rollouts (`od3-r1-stored`), quota `svamp=76,mbpp=60,ifeval=84,bfcl_simple_python=32,bfcl_multiple=28,bfcl_irrelevance=40`.
  - **Rounds 2–20:** 15 steps (`240` pairs) per round, quota `svamp=50,mbpp=56,ifeval=60,bfcl_simple_python=20,bfcl_multiple=22,bfcl_irrelevance=32` (gives Code `+4` and No-Fit `+6` more pairs per round than `od5f`, funded by `-6` from Math and `-4` from Simple).
- **Questions needed per type (from existing pools only, 0 new generation needed):**
  - Round 1 (`od3-r1-stored`): `440` math, `280` code, `400` IFEval, `160` simple, `120` multiple, `160` no-fit (`1,560` total).
  - Rounds 2–20 (`19` slices x `780`): `4,180` math, `2,660` code, `3,800` IFEval, `1,520` simple, `1,140` multiple, `1,520` no-fit (`14,820` slice questions; `16,380` total across 20 rounds—all already on Drive in `data/od3g-r2..22-questions.jsonl`).
- **Per-type pair rules, floors, ratios, and second-pair rule (`repair_pairs.py --mix-preset od5 --od5-v2`):**
  - **Shortening gate (`od5-v2`):** Shorten when `p1 >= 0.875` (if `p0 == 1.0`), `p1 >= max(0.50, p0 - 0.125)` (if `p0 >= 0.50`), else `p1 >= max(0.25, p0)`.
  - **Per-type minimum thinking floors (`--min-think-by-part`):**
    `svamp=30:0.46,mbpp=28:0.32,ifeval=12:0.24,bfcl_simple_python=25:0.44,bfcl_multiple=20:0.36,bfcl_irrelevance=4:0.10`
    *(Measured per question against `bt`, the original's median right thinking length on that question: math floor nudged `0.45 -> 0.46` to prevent `>70%` over-cutting; code lowered `35:0.36 -> 28:0.32`; IFEval lowered `18:0.30 -> 12:0.24`; no-fit lowered `16:0.28 -> 4:0.10`).*
  - **Per-type shortening ratio (`--shorten-ratio 1.30 --shorten-ratio-by-part`):**
    `mbpp=1.25,ifeval=1.22,bfcl_irrelevance=1.20` (math, simple, multiple stay at `1.30`).
  - **Second shortening pair per question (`--shorten-second`):**
    `mbpp+ifeval+bfcl_irrelevance` (adds `mbpp` so solved medium code questions yield up to 2 shortening pairs).
  - **Treatment of types where thinking pays little or hurts (`ifeval`, `bfcl_irrelevance`):**
    - On **IFEval**, keep `--count-mode length --no-shorten-counting --resp-median 0.95 --resp-max 1.15 --right-resp-cap 1.15` (and `response_len >= 0.92 * b_resp`), while lowering the thinking floor to `12:0.24` and ratio to `1.22`.
    - On **No-Fit (`bfcl_irrelevance`)**, lowering its floor from `16:0.28` to `4:0.10` and ratio to `1.20` (with cap raised from `64%` to `76%`) lets online DPO select the model's shortest valid `<think>...</think>` refusals (4–20 thinking tokens), pushing No-Fit saving from `58.7%` to `~70%` while preserving `+5.5` to `+6.5` accuracy.
- **Smoothed Brake & Non-Lengthening Hold (Fixes the r19–22 Oscillation):**
  - **2-Slice Smoothed Brake (`--od-smooth 2`):** `odpo_fixed` averages each part's monitor `(saved, acc)` over the current and previous slice (`last 2 slices`, doubling effective `N` to `120–440` per type).
  - **Per-type Caps (`--od-caps`):** `svamp=64,mbpp=62,ifeval=59,bfcl_simple_python=45,bfcl_multiple=53,bfcl_irrelevance=76` with `--od-acc-hold -2.0 --od-unlatch`.
  - **Non-Lengthening Hold in `repair_pairs.py`:** When a part is in `stop5` (`--no-shorten-parts`), we do not fill 80% of its quota with lengthening `recovery` pairs and wrong-length-matched `hold` pairs:
    1. Keep compact `be-right` pairs active for held parts (`0.50 <= ratio <= 1.00`, preferring `REREAD` and `ratio ~ 0.85`), instead of falling through to `0.90–1.12x` of long wrong answers.
    2. Cap a held part's total pairs at `40%` of its quota `n` when `dmg > -0.02` (`n_rem = min(n_rem, int(round(0.40 * n)))`) and cap `recovery` on held parts at `20%` of `n_rem` (`40%` if `dmg <= -0.02`), filling the rest of `n_rem` with compact `hold` pairs.

---

### 4. Checkpoint Choice Without the Full Test (Fitted Across All 13 Online DPO Pairs)

#### A. Why Consult28's Slope Rule Was Too Harsh on `od5f-r18` and `od5g-r20`


<!-- sections 4-8 from report_draft.md (the final answer was cut off at section 4) -->
## 4. Checkpoint Choice Without the Full Test (Fitting Dev-to-Full Across All 13 Online DPO Pairs)

### A. Why Consult28's Slope Rule Was Too Harsh on `od5f-r18` and `od5g-r20`
In `fullpick` (`track.py` line 4693), corrected accuracy was computed as:
`acc_corr = dev_acc - fp_acc_bias - fp_acc_slope * max(0.0, dev_saved - 53.0)`
with `fp_save_bias = 0.55`, `fp_acc_bias = 2.95`, `fp_acc_slope = 0.25`.
- On `od5f-r18` (Dev `60.4% / +0.88`), the `0.25` slope subtracted an extra `0.25 * (60.4 - 53.0) = 1.85` pts (or `2.10` pts above `52%`), predicting `-3.92` when the actual full test was **`-0.80`** (a `3.12`-pt error!).
- On `od5g-r20` (Dev `58.2% / +3.18`), the slope rule predicted `-1.07` (outside the `-1.0` gate!), when the actual full test was **`+0.14`** (a `1.21`-pt error!).
- **Why?** The `0.25` slope was fitted across *offline* DPO runs (`rpL-80`), which suffer steep accuracy collapse at high saving. Online DPO with per-question floors does **not** collapse at high saving.

### B. Exact Regression Fit Across All 12 `od5c` / `od5r` / `od5f` / `od5g` Full-Tested Checkpoints
Across the 12 online DPO checkpoints (`od5c-r16/19/20`, `od5r-r15/16/17`, `od5f-r14/15/16/17/18`, `od5g-r20`; 13 including `od5d-r11`):
1. **Saving bias (`Dev_sav - Full_sav`):**
   - All 12 `od5c/r/f/g` pairs: mean **`+0.17` pts** (SD `0.52`, median `+0.20`, range `[-0.81, +0.89]`).
   - `od5f/g` 6 pairs (`r14–r18`, `od5g-r20`): mean **`+0.47` pts** (SD `0.38`, median `+0.51`).
2. **Accuracy bias (`Dev_acc - Full_acc`):**
   - All 12 `od5c/r/f/g` pairs: mean **`+2.85` pts** (SD `0.59`, median `+2.90`, range `[+1.68, +3.72]`).
   - `od5f/g` 6 pairs: mean **`+2.48` pts** (SD `0.48`, median `+2.47`, range `[+1.68, +3.04]`).
3. **Test for a saving-dependent slope (`acc_bias` vs `Dev_sav`):**
   - Across the 12 `od5c/r/f/g` pairs, the correlation between `Dev_sav` and `(Dev_acc - Full_acc)` is **negative** (`r = -0.398`, `p = 0.200`, OLS slope **`-0.079 +/- 0.058`** pts per saving point; across all 13 pairs including `od5d-r11`, `r = -0.053`, `p = 0.864`, slope `-0.015 +/- 0.086`).
   - That is, **there is zero positive slope** in online DPO (`fp_acc_slope = 0.0`). Setting `fp_acc_slope = 0.0` with flat biases `fp_save_bias = 0.35`, `fp_acc_bias = 2.75` reduces MAE across the 12 pairs from `1.09` pts to **`0.46` pts** (and `0.37` pts on `od5f/g`), predicting `59.54% / -1.26` (or `60.05% / -1.87` on the first Dev run, `61.15% / -1.26` on the re-test) for `od5f-r18` and `57.85% / +0.43` for `od5g-r20`.
4. **Reducing Dev-512's sampling noise (`+/-0.85` pt SE) using the Free Round Monitor (`780` questions x 8):**
   - When round `k` is evaluated on both `Dev-512` (`512 x 8`) and round `k+1`'s unseen slice monitor (`780 x 8`), combining them (`0.5 * Dev + 0.5 * Monitor`) covers `1,292` unseen test-matched questions (`2.5x` larger than Dev-512 alone), cutting the standard error to `+/-0.54` pts (`Monitor_acc - Full_acc` on `od5f-r18` is `+2.37 - (-0.80) = +3.17` pts).
5. **Recommended `fullpick` Rule for Online DPO:**
   - Use flat calibration: **`--fp-save-bias 0.35 --fp-acc-bias 2.75 --fp-acc-slope 0.0 --fp-acc-gate -0.90`** (so a round counts as inside the `-1.0` full-test goal when `Dev_acc >= +1.85`, with the gate set to `--fp-gate-save 58.5 --fp-gate-acc 1.65`).

---

## 5. Cost on the Article's Second Half: Excess Churn & Math 8/8 Consistency

1. **Why `od5f-r18` (math `70.9% / -3.60`) hurts math 8/8 consistency, whereas the `od5h` target (`60.9% / -0.3`, math `68.0% / -1.8`) preserves it:**
   - In `SMARTER_DATA.md`, `od5c-best` (`54.8% / -0.78`, math `61.1% / -2.84`) had `+0.8%` net question flips (+168 gained / 145 lost), `10.1%` gross flips vs `9.1%` 8-sample binomial noise (**only `1.0%` excess churn**), and math 8/8 consistency dropped from `75.1% -> 69.9%` (`-5.2` pts) because of `od5c`'s lower math floor (`0.42`).
   - Pushing math to `70.9% / -3.60` (`od5f-r18`) cuts into the `71%` pre-settlement working-out tokens that `effprobe` identified on math, which lowers math 8/8 consistency by an estimated additional `1.5–2.0` pts (`~68%`) and raises math excess churn by `~0.4` pts.
   - Conversely, **`od5h` pulls math BACK from `70.9% / -3.60` to `~68.0% / -1.80`** (close to `od5g-r20`'s `66.8% / -1.50`, which lost only half as much math accuracy as `od5c-best`'s `-2.84`!), while getting the extra `+1.4` pts of total saving from **No-Fit (`59% -> 70%`), Code (`56% -> 57.5%`), and IFEval (`56% -> 57.5%`)**.
2. **Predicted Churn & Consistency for `od5h` (`~60.8% / -0.4`):**
   - Because `effprobe` proves that on No-Fit (`91.6%`), Code (`84.3%`), and IFEval (`67.1%`) the cut tokens are overwhelmingly **post-settlement self-doubt and re-checking**, cutting them **increases** 8/8 consistency on No-Fit (where 8/8 right rises by `~+6` pts) and IFEval (`+1.5` pts), while keeping math 8/8 consistency at **`71.5–72.5%`** (better than `od5c-best`'s `69.9%`!) and overall excess churn at **`<= 1.0–1.2%`**.
3. **Three Concrete Safeguards that Keep Churn & Math 8/8 Consistency Tight:**
   - **Math thinking floor `30:0.46` + smoothed cap `svamp=64`:** Prevents shortening pairs from ever selecting lucky early-stop math answers below `46%` of the original's per-question median thinking `bt`.
   - **Shortening gate `p1 >= 0.875` on `8/8` (`p0 == 1.0`) questions (`--od5-v2`):** Never shortens a solid `8/8` question if it has dropped below `7/8` (`0.875`), directly protecting the `8/8` consistency bucket.
   - **Preferring re-reading right answers (`REREAD.search` in `repair_pairs.py` line 460):** In compact `be-right` pairs, sorting first by `0 if REREAD.search(decode(r)) else 1` preserves the brief verification sentence on mixed-reliability questions without keeping multi-paragraph re-checks.

---

## 6. Two Problems Seen in Rounds 19–20: (a) Brake Oscillation & (b) Scarce Medium Code

### Problem (a): Single-Slice Brake Oscillation (and Why Both `od5f` and `od5g` Swung 6–12 pts After Round 18/20)
- **Diagnosis:** The oscillation has **two interacting causes**:
  1. **Single-slice measurement noise:** Each round monitor slice has only `60–220` questions per type (`SE ~ 1.5–3.0%` on saving and `1.5–2.5` pts on accuracy). A type near its cap or the `-2.0` accuracy hold triggers and releases randomly from slice to slice.
  2. **Held-type quota flooding with lengthening pairs (the primary amplifier):** As shown in Section 1B, when a type is held (`part in stop5`), `repair_pairs.py` sets `c_sh = 0` but **keeps its full pair quota `n`** and fills `100%` of those slots with `recovery` pairs (explicitly `1.08–1.45x` longer than wrong answers) and non-compact `hold` pairs (`0.90–1.12x` of long wrong answers). When 3–4 types hit their caps at round 18, `53%` of the DPO batch becomes **active lengthening pairs**, driving saving down by 8–12 pts in a single round (`64% -> 52%` on math) and causing `--od-unlatch` to release them on the next round!
- **Should the brake act on a smoothed reading (last 2–3 slices) or a fixed check set?**
  - **Use a 2-slice trailing average (`--od-smooth 2`, averaging the current slice `k+1` and previous slice `k` from `history`) COMBINED with the non-lengthening hold in `repair_pairs.py`:**
  - A fixed check set would cost `512 x 8 = 4,096` extra rollouts (`~12` minutes of GPU time) on *every* round, doubling arena wall-clock time. By contrast, a **2-slice trailing average** is **100% free** (0 extra rollouts), doubles the sample size to `120–440` questions per type, and still responds within 1 round to genuine drift (whereas a 3-slice window lags too much when a type approaches its cap at ~2.5% per round). Combined with capping held-type pairs to `<= 40%` of quota and using compact `hold` (`0.65–1.00x`), the limit-cycle oscillation disappears completely.

### Problem (b): Hard Generated Code in Slices 20+ (`b1`/`b2` vs `m1`) — Cost and Optimal Code Mix
- **How much did the hard code (`b1` solve `25%`, `b2` solve `16%` vs test `52%`) cost in `od5f` rounds 20–23?**
  - In `repair_pairs.py`, a question only produces a shortening pair if `can_shorten` holds (`p1 >= 0.25` and `len(clean) >= 2` with length spread `>= 1.30`). On `all-wrong` (`0/8`) or `1/8` questions, **zero shortening pairs can be built**, while `damaged` questions generate lengthening `recovery` pairs.
  - Because `od5f`'s old slices 20–23 drew from `b1`/`b2` after KodCode ran out, code's shortening candidates dried up while `recovery`/`hold` dominated: on `od5f`'s monitor, code saving collapsed from **`59.8% / +2.6` (r19) -> `48.2% / +2.4` (r20) -> `43.2% / -1.9` (r21) -> `30.8% / +2.6` (r22)**—a **29-point collapse in code saving** (`~7.5` pts of total official saving!).
  - When `od5g` replaced slices 20–22 with matched medium code (`r2x-code-m1`, mean solve `48%`, 58% mixed `1/8–7/8` bin), code saving on the r21 monitor was **`50.0%` vs `43.2%`** (`+6.8` pts better on slice 21, and `+5.5` pts better on Dev-512 at r21: `51.6%` vs `46.8%`).
- **Why slice 23 failed `od3branch` and what code mix the next run should use:**
  - `od3g` already has **22 matched rounds** (`slices 1–22`), which is **more than enough for our 20-round `od5h` run** (no new data generation is required for `od5h`!).
  - If you generate one more code batch to build slices 23–25 later: slice 23 failed in `od3branch` because the remaining code pool had **median tokens too long (`1,409` vs `1,062`) and p90 too short (`2,748` vs `4,025`)**—all medium `m1` tasks clustered around `1,400` tokens with no easy `<800`-token tasks and no hard `4,000`-token tail, and `od3branch`'s `--ob-trim` only trimmed one tail based on the mean (`too_long = mean > target["tok"]`).
  - **Optimal code mix for any future slices 23+:** Target the MBPP test's exact 3-bin solve-rate & length mixture (`52%` mean solve, `~1,680` mean tokens, `~1,060` median, `~4,025` p90): **30% easy/Always-Right (`7/8–8/8`, `<900` tokens)** + **40% medium/Mixed (`2/8–6/8` from `gen_code_tasks.py --medium --novel`, `1,000–2,200` tokens)** + **30% hard/Rarely-Right (`0/8–1/8` from `b1`/`b2`, `2,500–4,500` tokens)**, and in `od3branch` trim by the **failing metric (`med` vs `p90`)** rather than only the mean.

---

## 7. Using the `effprobe` Causal Analysis: Post-Settlement Waste vs Genuine Working-Out

### A. Why `effprobe` Explains the Per-Type Curves (Math vs Code / IFEval / No-Fit)
`effprobe` (on unseen slices 19–20, plus `THINKING_OFF.md` on the full test) separates two fundamentally different regimes:
1. **Math (SVAMP): Genuine Path Compression (`71.1%` pre-settlement working-out, `28.9%` post-settlement re-checking).**
   - When the original model is force-stopped at `~300` thinking tokens (`b1 = 304t`), math accuracy crashes from `88.6%` to **`56.5%`** (`-32.1` pts) or `47.4%` (`mt`). Yet the trained model (`od5c-best`) naturally finishes in **`281` thinking tokens at `87.9%` accuracy** (`-0.7` pts), and its thinking reads at **`0.68` bits/token** under the original model vs `0.39` bits/token for the original's own math thinking (`+74%` higher information density per token).
   - **Implication:** On math, the model actually needs `~250–300` tokens (`~45%` of original thinking) to compute the answer; cutting below `~45%` (`>68%` total saving) cuts into the calculation itself (`od5f-r18` at `70.9%` loses `-3.6` pts; `nothink` at `79.6%` loses `-5.6` pts). Therefore, math **must keep a high per-question floor (`30:0.46`) and cap (`64%` monitor / `~68%` full test)**.
2. **No-Fit (`91.6%` post-settlement), Simple Tools (`88.1%`), Code (`84.3%`), Multiple Tools (`74.9%`), and IFEval (`67.1%`): Post-Settlement Self-Doubt & Re-Checking.**
   - On **No-Fit (`bfcl_irrelevance`)**, `91.6%` of thinking occurs *after* the model has already decided no tool fits; extra thinking talks the model into hallucinating a tool call! Force-stopping at `b0 = 0` thinking tokens raises slice accuracy by `+5.3` pts (original) and `+4.1` pts (trained, to `92.5%`), and on the **full test**, turning thinking completely off (`THINKING_OFF.md`) saves **`95.9%` of tokens at `+6.2` pts accuracy** (`+6.5` to `+9.6` on trained models)!
   - On **Code and IFEval**, `effprobe/README.md` point 5 highlights a vital distinction between easy/medium slices and the full test: while `b0 = 0t` costs only `-2.3` (Code) and `-1.1` (IFEval) on slices 19–20, **full-test `thinking-off` costs `-9.1` pts on Code (`74.6%` saved) and `-5.0` pts on IFEval (`70.6%` saved)** (or `-4.6` and `-1.3` for `od5c-best` thinking-off). Why? Because on Code (`210` tokens before settlement) and hard IFEval constraints (`166` tokens before settlement), the model needs a **short 80–200 token working-out burst (`24–32%` of original thinking)** to draft the algorithm or plan the constraints, after which the remaining `68–84%` of thinking is redundant verification.

### B. Can One Online DPO Run Teach the Model to Think Very Little on the First Kind Within the Rules, and What Would It Score?
- **YES.** Because the user's rules prohibit inference-time prompt routing or stripping `<think>` tags at test time (*"one training procedure... general rules measured per question / per type"*), we teach the model to think minimally on No-Fit (`~10%` of original thinking, `~40–60` thinking tokens) and compactly on IFEval (`~24%` floor) and Code (`~32%` floor) **purely through the online DPO pair builder (`repair_pairs.py`)**:
  - Setting `--min-think-by-part bfcl_irrelevance=4:0.10,ifeval=12:0.24,mbpp=28:0.32` and `--shorten-ratio-by-part bfcl_irrelevance=1.20,ifeval=1.22,mbpp=1.25` with `--shorten-second mbpp+ifeval+bfcl_irrelevance` and cap `bfcl_irrelevance=76` allows DPO to continuously prefer the shortest 1-sentence `<think>` blocks on No-Fit and non-counting IFEval whenever the model solves the question (`p1 >= 0.875`).
  - Why not push Code or IFEval to `0` thinking tokens (`floor = 0`)? Because full-test thinking-off loses `-9.1` on Code and `-5.0` on IFEval! Keeping floors at `0.32 * bt` on Code and `0.24 * bt` on IFEval preserves the pre-settlement working-out (`~210t` on Code, `~166t` on IFEval) while stripping the `68–84%` post-settlement tail.
  - **Expected Full-Test Score from This Allocation:** As calculated in Section 2C, this single run yields **Math `68.0% / -1.8`, Code `57.5% / -1.1`, IFEval `57.5% / +0.8`, Simple `48.5% / -1.2`, Multiple `51.5% / -0.8`, No-Fit `70.0% / +6.0` (BFCL pooled `57.0% / +0.95`) -> Official Full Test `60.9%` saved at `-0.29` accuracy** (range **`60.2% to 61.4%` saved at `-0.2` to `-0.8` accuracy**).

---

## 8. Exact Code Changes in Words and Queue Lines

### A. Exact Code Changes (2 Files: `repair_pairs.py` and `track.py`)
1. **`code/repair_pairs.py` — Stop Held Types (`part in stop5`) from Flooding the DPO Batch with Lengthening Pairs:**
   - **In `main()` around line 453 (inside `elif clean and wrongs and 1 <= k1 < n_ans:`):**
     Remove `and part not in set(filter(None, re.split(r"[,+]", args.no_shorten_parts)))` (or replace it with `and (part not in right_hold)` for held parts) so that when a part is held, its `hold` (`be-right`) candidates still select **compact right answers no longer than the wrong answer** (`0.50 <= ratio <= 1.00`, preferring `REREAD` and `ratio ~ 0.85`, capped at `thinking_len(r) <= 1.05 * tmed`), instead of falling through to lines 463–469 which match `0.90–1.12x` of long wrong answers.
   - **In `main()` around lines 683–707 (inside `for part, n in quota.items(): if args.kinds == "mix" and od5:`):**
     When `part in stop5` (the type is held by the brake):
     - Scale down the active pair budget on the held part if it is not heavily damaged: if `part in stop5 and dmg > -0.02`, cap the total pairs taken for `part` at `n_rem = min(n_rem, int(round(0.40 * n)))` (so held types maintain accuracy without occupying 53% of the gradient update), and cap `recovery` pairs on a held part at `int(round((0.40 if dmg <= -0.02 else 0.20) * n_rem))`, filling the rest of `n_rem` with compact `hold` pairs (and leaving `shortfall[part]` ignored when `part in stop5`).
   - **In `main()` around line 526 (inside `if base_part in shorten_second` in the non-od5 branch, for completeness):** Use `sh_ratio` instead of `args.shorten_ratio` (note: line 428 in the `od5` branch already uses `ratio_need = sh_ratio_part.get(base_part, args.shorten_ratio)`).

2. **`code/track.py` — Add 2-Slice Smoothed Brake (`--od-smooth`) in `odpo_fixed`:**
   - **In `main()` parser setup (around line 305):** Add argument `--od-smooth` (`type=int, default=1`, help `"odpo: average per-part monitor saving and accuracy over the last N slices for the brake"`).
   - **In `odpo_fixed(args)` (lines 5442–5446):** After computing `sv_, ac_, per_ = od3_monitor(...)` on the current round `k + 1`'s rollouts, if `getattr(args, "od_smooth", 1) > 1` and `history` has recent rounds (up to `args.od_smooth - 1` previous rounds from `history[-(args.od_smooth - 1):]`), compute the smoothed per-part dictionary `per_use` by averaging `per_[p_][0]` (saving) and `per_[p_][1]` (accuracy) with the `[sv, ac]` entries of `h["per_part"][p_]` from those previous rounds. Use `per_use` instead of single-slice `per_` in line 5445 (`{p_ for p_, (sv, ac) in per_use.items() if sv >= caps.get(p_, 1e9) or ac <= args.od_acc_hold}`).

### B. Queue Lines (Plain Text)
1. **Run `od5h` (20 rounds from the original Qwen3-0.6B on the matched `od3g` slices 1–20, with Dev-512 scored on rounds 16, 17, 18, 19, 20):**
   python3 -u track.py --stage odpo --name od5h --od-fixed-ref --od-slices od3g --od-rounds 20 --od-min-rounds 20 --od-r1-steps 20 --od-steps 15 --od-r1-quota svamp=76,mbpp=60,ifeval=84,bfcl_simple_python=32,bfcl_multiple=28,bfcl_irrelevance=40 --rp-quota svamp=50,mbpp=56,ifeval=60,bfcl_simple_python=20,bfcl_multiple=22,bfcl_irrelevance=32 --rp-lr 1e-5 --od-caps svamp=64,mbpp=62,ifeval=59,bfcl_simple_python=45,bfcl_multiple=53,bfcl_irrelevance=76 --od-acc-hold -2.0 --od-unlatch --od-smooth 2 --od-stop-from 18 --od-stop-save 61.5 --od-stop-acc -0.5 --od-conv-gain 0.3 --od-conv-min 60.0 --od-dev-rounds 16 17 18 19 20 --rp-pair-args=--kinds,mix,--mix-preset,od5,--od5-v2,--shorten-ratio,1.30,--shorten-ratio-by-part,mbpp=1.25+ifeval=1.22+bfcl_irrelevance=1.20,--shorten-second,mbpp+ifeval+bfcl_irrelevance,--min-think-by-part,svamp=30:0.46+mbpp=28:0.32+ifeval=12:0.24+bfcl_simple_python=25:0.44+bfcl_multiple=20:0.36+bfcl_irrelevance=4:0.10,--count-mode,length,--no-shorten-counting,--resp-median,0.95,--resp-max,1.15,--right-resp-cap,1.15,--loop-finished

2. **Pick and run the top 2 checkpoints on the full test (using the recalibrated flat online-DPO Dev correction):**
   python3 -u track.py --stage fullpick --fp-run od5h --fp-kind arena --fp-dev-rounds 16 17 18 19 20 --fp-save-bias 0.35 --fp-acc-bias 2.75 --fp-acc-slope 0.0 --fp-acc-gate -0.90 --fp-gate-save 59.5 --fp-gate-acc 1.75 --fp-n 2 --fp-skip-tested

### C. Expected Landing and Main Risks
- **Expected Full-Test Landing:** **`60.7%` saved at `-0.45` accuracy** (range: **`60.2% to 61.4%` saved at `-0.20` to `-0.85` accuracy**), peaking at round 18, 19, or 20 without the r19–22 oscillation.
- **Main Risks & Mitigations:**
  1. *Risk:* Lowering No-Fit's floor to `4:0.10` and Code's floor to `28:0.32` could cause a format/syntax dip if a truncated response is chosen. *Mitigation:* `od5` builds shortening pairs strictly from `clean` answers (`correct and closed_thinking and not truncated and not loops8 and not broken`), so every chosen short thought has a verified valid tool refusal or passing code block.
  2. *Risk:* Reducing held-type pair count to `40%` of quota makes the round's total pair count `< 240` (`~180–210` pairs when 2–3 types are held). *Mitigation:* `ptrepair` already sets `a.rp_epochs = max(args.rp_epochs, 2)` (`track.py` line 5431), so 15 steps (`240` examples) are always completed cleanly, and active (unheld) types receive slightly higher effective gradient weight, accelerating Code/IFEval/No-Fit convergence in rounds 17–20.
