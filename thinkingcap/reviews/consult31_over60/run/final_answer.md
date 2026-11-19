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
In `fullpick` (`track.py` line 4693), `acc_corr = dev_acc - 2.95 - 0.25 * max(0.0, dev_saved - 53.0)` subtracted `1.85` extra points on `od5f-r18` (Dev `60.4% / +0.88`), predicting `-3.92` when the actual full test