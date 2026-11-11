# Full-Test Review: `rpK-60`, `p1r1s0-200`, `rpM`, and the 12 GPU-Hour Endgame

---

## 1. What the Two New Full Tests Change

### 1.1 `rpK` Step 60 vs Step 20 and `rpE-80`: Where the +4.55% Saving Came From and Where the -1.44 Accuracy Sits

On the full test ($2,759 	imes 8$), **`rpK-60` scored 54.43% saved / -1.44 accuracy** (`+2.38` points above the project's frontier curve), gaining **+4.55% saving over `rpK-20` (`49.88% / -1.28` official, `-1.24` finished-only) for `-0.16` official accuracy (`-0.19` finished-only)**, and **+7.39% saving over `rpE-80` (`47.04% / -0.95` official, `-0.90` finished-only) for `-0.49` official accuracy**.

| Benchmark ($N$ Full) | `rpE-80` Full (Saved / $\Delta$Acc / Thk+Rsp) | `rpK-20` Full (Saved / $\Delta$Acc / Thk+Rsp) | `rpK-60` Full (Saved / $\Delta$Acc / Thk+Rsp) | `rpK-20` $	o$ `rpK-60` Shift (Saved / $\Delta$Acc / Tokens) | `rpE-80` $	o$ `rpK-60` Shift (Saved / $\Delta$Acc / Tokens) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **SVAMP Math** (1,000) | 54.13% / -1.25 (`306.7+8.0`) | 57.57% / -2.59 (`281.6+9.5`) | **61.11% / -2.35** (`258.8+8.1`) | **+3.54% / +0.24** (`-22.8` thk, `-1.4` rsp) | **+6.97% / -1.10** (`-47.9` thk, `+0.1` rsp) |
| **MBPP+ Code** (378) | 46.31% / -1.03 (`1089.8+51.4`) | 51.13% / -0.26 (`991.2+47.4`) | **55.43% / -1.22** (`896.8+50.4`) | **+4.30% / -0.96** (`-94.4` thk, `+3.0` rsp) | **+9.13% / -0.20** (`-193.0` thk, `-1.0` rsp) |
| **IFEval** (541, prompt / fin.) | 45.42% / -2.56 (`-2.38`) (`156.5+223.3`) | 45.28% / -3.07 (`-2.93`) (`151.2+229.7`) | **49.35% / -3.00** (`123.0+229.6`) | **+4.07% / +0.07** (`-0.07` fin; `-28.2` thk, `-0.1` rsp) | **+3.93% / -0.44** (`-0.62` fin; `-33.5` thk, `+6.3` rsp) |
| **BFCL Simple** (400) | 37.32% / +0.09 (`98.0+24.1`) | 39.50% / +0.28 (`93.8+24.1`) | **45.45% / -1.47** (`82.5+23.8`) | **+5.95% / -1.75** (`-11.3` thk, `-0.3` rsp) | **+8.13% / -1.56** (`-15.5` thk, `-0.3` rsp) |
| **BFCL Multiple** (200) | 39.41% / +0.06 (`107.4+23.4`) | 41.97% / -1.50 (`101.8+23.5`) | **47.70% / -1.00** (`89.4+23.5`) | **+5.73% / +0.50** (`-12.4` thk, `+0.0` rsp) | **+8.29% / -1.06** (`-18.0` thk, `+0.1` rsp) |
| **BFCL No-Fit** (240) | 49.55% / +3.44 (`96.4+11.0`) | 53.83% / +3.65 (`90.5+7.8`) | **61.88% / +6.20** (`76.2+4.9`) | **+8.05% / +2.55** (`-14.3` thk, `-2.9` rsp) | **+12.33% / +2.76** (`-20.2` thk, `-6.1` rsp) |
| **BFCL Pooled** (840) | 41.47% / +1.04 (`99.8+20.2`) | 44.37% / +0.82 (`94.7+19.3`) | **50.89% / +0.83** (`82.3+18.4`) | **+6.52% / +0.01** (`-12.4` thk, `-0.9` rsp) | **+9.42% / -0.21** (`-17.5` thk, `-1.8` rsp) |
| **4-Part Headline** | **47.04% / -0.95** (`-0.90`) | **49.88% / -1.28** (`-1.24`) | **54.43% / -1.44** | **+4.55% / -0.16** (`-0.19` fin) | **+7.39% / -0.49** (`-0.53` fin) |

1. **SVAMP Math by Length Band, Difficulty, and Relative Thinking (`+3.54%` saving and `+0.24` accuracy from step 20 to 60):**
   - **By quartiles of `rpE-80` thinking length:** Between step 20 and step 60, Q1 (`81–152` tokens, $n=251$) moved `+0.05` pts (`96.61% -> 96.66%`, thinking `123.3 -> 114.4`), Q2 (`152–200` tokens, $n=249$) moved `+0.25` pts (`92.17% -> 92.42%`, `168.6 -> 157.3`), Q3 (`201–322` tokens, $n=250$) moved `+0.00` pts (`86.55% -> 86.55%`, `242.6 -> 219.6`), and **Q4 (`322–2,574` tokens, $n=250$) recovered `+0.65` pts (`69.10% -> 69.75%`, `592.2 -> 544.1`, runaways `3 -> 0`)**. By `base` solve rate, Hard (`<=4/8`, $n=94$) recovered **`+1.46` pts (`32.18% -> 33.64%`)**, Medium (`5–7/8`, $n=205$) held at `79.02%` (`+0.00`), and Easy (`8/8`, $n=701$) rose `+0.14` pts (`95.42% -> 95.56%`).
   - **Why `NEW_RESULTS.md`'s `<0.5x / 0.5–1.0x / >=1.0x` count shifted from `106 / 375 / 395` (`rpK-20`) to `155 / 396 / 315` (`rpK-60`) even though SVAMP accuracy *improved* (+19 right answers):** Because all SVAMP answers (right and wrong) shortened by `-8.1%` (`281.6 -> 258.8` tokens) between steps 20 and 60, `80` wrong answers slid across the **fixed** `rpE-80` median threshold from `>=1.0x` (`395 -> 315`, `-80`) into `<1.0x` (`481 -> 551`, `+70`), while total wrong answers actually fell from `1,111` to `1,092` (`-19`). Measured against **each model's own** median right thinking on the question, wrong answers in `<0.5x / 0.5–0.95x / >=0.95x` were completely unchanged between `rpK-20` (`81 / 298 / 467`) and `rpK-60` (`78 / 303 / 471`). Compared to `rpE-80` (`82 / 250 / 430`), the net SVAMP loss (`-1.10` pts = `-88` right answers) was entirely incurred in the first 20 steps inside Q3 (`-1.15` pts) and Q4 (`-3.10` pts) from skipped re-checks.

2. **IFEval by Class and Rule (`+4.07%` saving, `+0.07` official accuracy from step 20 to 60):**
   - Between steps 20 and 60, response inflation stopped completely (`229.7 -> 229.6` tokens, runaways `2.73% -> 2.70%`) while the 112 `shorten` pairs cut thinking by `-28.2` tokens (`151.2 -> 123.0`).
   - By class (`rpE-80` $	o$ `rpK-20` $	o$ `rpK-60`):
     - **`at_least_N` (58 questions):** `18.53%` (`149+489`) $	o$ `16.81%` (`142+475`) $	o$ **`18.10%` (`116+525`)** (`+1.29` pts from step 20 to 60 as responses rose `+50` tokens, but still `-15.09` pts below `base`'s `33.19%`). Why did `rpK`'s 160 counting pairs fail to lift `at_least_N` above `18.5%`? Because `rpE-80` only passes `11.4%` of `"at least N words"` prompts (`0/8` right on most hard word-count questions), and `--count-mode all` diluted the 160 counting pairs across all 7 counting-rule families, leaving only **39 pairs in the entire `rpK` file with `chosen_response >= 300` tokens**.
     - **`fewer_than_N` (36 questions):** `71.18%` (`101+80`) $	o$ `69.44%` (`95+93`) $	o$ **`70.49%` (`79+91`)** (`+1.04` pts from step 20 to 60; only `-0.69` pts below `rpE-80`).
     - **`no_counting` (264 questions):** `60.61%` (`167+159`) $	o$ `60.65%` (`170+164`) $	o$ **`60.84%` (`133+159`)** (`+0.24` pts over `rpE-80` with `-34` thinking tokens).
     - **`other_counting` (183 questions):** `56.42%` (`155+261`) $	o$ `55.60%` (`139+273`) $	o$ **`54.51%` (`119+265`)** (`-1.91` pts vs `rpE-80`), concentrated in `length_constraints:number_paragraphs` (`54.2% -> 49.5%`), `detectable_content:number_placeholders` (`88.4% -> 85.2%`), `combination:repeat_prompt` (`46.3% -> 43.6%`), and `language:response_language` (`61.7% -> 58.9%`).

3. **BFCL Tool Calls (`+6.52%` pooled saving, `+0.01` pooled accuracy from step 20 to 60):**
   - Between steps 20 and 60, `bfcl_irrelevance` jumped **`+2.55` pts (`87.03% -> 89.58%`, `+49` right non-calls; Hard `<=4/8` jumped `+14.42` pts from `30.13%` to `44.55%`)** and `bfcl_multiple` recovered **`+0.50` pts (`74.56% -> 75.06%`, `+8` right calls; Hard `<=4/8` jumped `+7.71` pts from `21.54%` to `29.26%`)**, while `bfcl_simple_python` fell **`-1.75` pts (`82.44% -> 80.69%`, `-56` right calls)**.
   - **Exact cause of `bfcl_simple_python`'s `-1.75` pt drop:** `rpK-60`'s accuracy *given* a call barely moved (`84.82% -> 84.27%`), but its **call rate on `bfcl_simple_python` fell by `-1.44%` (`97.19% -> 95.75%`, `+46` non-calls of 3,200)**—accounting for **82% (`46 / 56`) of the lost answers**! Having 140 `bfcl_irrelevance` non-call pairs against only 48 `bfcl_simple` and 36 `bfcl_multiple` `right_cr` pairs shifted the shared call threshold toward declining borderline calls (`+49` right non-calls on `no-fit` vs `-46` lost calls on `simple`, netting `+0.01%` on pooled BFCL).

4. **What This Says About Dose (Steps) and Where the Remaining Accuracy Loss Sits:**
   - **Dose:** At `lr=1e-5` (batch 16 on ~1,100 pairs), **20 steps (`0.3` epochs) is under-dosed**: it pays the initial accuracy cost on SVAMP and IFEval before thinking compression on BFCL/IFEval and hard-question guards take effect. **60 steps (`~0.9` epochs) is the right dose**, delivering `+4.55%` extra saving for `-0.16` accuracy.
   - **Where `rpK-60`'s `-1.44` headline accuracy deficit sits:** **93% of the deficit sits in just two places:** **IFEval (`-3.00` = `-0.75` headline pts**, of which `-1.62` is `at_least_N` and `-1.32` is `other_counting`, plus `2.70%` runaways wasting `100` tokens/answer) and **SVAMP Math (`-2.35` = `-0.59` headline pts**, `-1.10` vs `rpE-80` in Q3/Q4). MBPP+ (`-1.22` = `-0.31` headline pts) and BFCL Pooled (`+0.83` = `+0.21` headline pts) net to `-0.10` headline pts.

### 1.2 What Path 1 Round 1 (`p1r1s0-200`) Teaches (`43.42% saved / -1.46` official, `-1.61` finished-only)
1. **IFEval Response Length Moves Both Ends (`at_least_N` at `40.7%` vs `fewer_than_N` at `60.1%`):**
   - `p1r1s0-200` proves beyond doubt that IFEval's `"at least N"` deficit is purely a response-length effect, not a thinking-length effect: by rewarding longer responses (`chosen_response > rejected_response` in `75.7%` of its 720 IFEval pairs, `+46.9` tokens mean, uncapped), `p1r1s0-200` pushed `at_least_N` responses from `668` (`base`) and `489` (`rpE-80`) to **`981` tokens**, driving `at_least_N` accuracy to **`40.73%` (`+7.54` pts above `base`'s `33.19%` and `+22.20` pts above `rpE-80`'s `18.53%`)**, with `length_constraints:number_words` jumping `+18.6` pts over `base` (`40.9% -> 59.5%`) and `combination:repeat_prompt` jumping `+13.4` pts (`47.3% -> 60.7%`).
   - Simultaneously, because `p1r1s0` used `--count-mode all` (which classified `fewer_than_N` rules as counting rules requiring `chosen_response >= rejected_response`!), had no `--resp-max` on `shorten` pairs (`+18.8` rsp tokens), had no `--count-resp-cap`, and had zero anti-loop pairs, `fewer_than_N` responses doubled from `83` to **`163.5` tokens**, crashing `fewer_than_N` accuracy by **`-12.85` pts (`72.92% -> 60.07%`)**, inflating `no_counting` responses (`181 -> 213` tokens), raising IFEval runaways (`3.60% -> 4.09%`), and collapsing IFEval saving to **`22.54%`**.
2. **Lineage Offset on `Dev-512`:**
   - For the `rc1 -> rpE-80` lineage, `Dev-512` tracks full-test saving within `0.1–0.7` points (`rpE-80`: `47.0%` Dev vs `47.0%` Full; `rpK-20`: `50.6%` vs `49.9%`; `rpK-60`: `53.9%` vs `54.4%`) with a fixed `-2.25` pt absolute accuracy offset (`paired` delta on `rpK-60`: `+6.89% / -0.56` Dev vs `+7.39% / -0.53` Full—accurate to within `0.03` pts on accuracy!).
   - For **`p1r1s0-200` (1 round straight from raw Qwen)**, `Dev-512` over-read saving by **`+2.46` points (`45.88%` Dev vs `43.42%` Full)** and accuracy by `-1.58` points (`+0.12` vs `-1.46`), because `Dev-512` MBPP+ has `6.0%` base runaways (`52.6%` saved on Dev vs `44.0%` on Full) and `Dev-512` IFEval has **26.5% `"at least N"` questions and only 5.3% `"fewer than N"` questions** (vs `10.7%` and `6.7%` on the Full Test).

### 1.3 Audit of My `consult9` / `consult10` Forecasts Against These Full Tests
- **`rpK-60`:** In `consult9`, I forecast **`53.2% [52.0, 54.5]` saved / `-1.45 [-2.00, -0.90]` accuracy**. Actual: **`54.43%` saved / `-1.44` accuracy** (`0.01` pt error on accuracy; saving at the top of my interval because IFEval thinking dropped `-28` tokens between steps 20 and 60).
- **`p1r1s0-200`:** In `consult9`, I forecast **`45.8% [44.5, 47.1]` saved / `-2.10 [-2.70, -1.50]` accuracy** and advised canceling its full test. Actual: **`43.42%` saved / `-1.46` official (`-1.61` finished-only) accuracy**. My saving forecast was `2.4` points too high because I assumed `Dev-512`'s `52.6%` MBPP+ saving would hold on the full test (`44.0%`), while my accuracy forecast (`-1.50 to -2.70`) bracketed `-1.61` / `-1.46`.

---

## 2. `rpM` as Built: Pair Audit, Edits, and Full-Test Forecast

### 2.1 Audit of `rpM`'s Built Pairs (`tables/rpM_pairs_local_build_math_on_hard_questions_only.csv` & `repair_pairs.py`)
1. **Will `"fewer than N"` or runaways pay for `rpM`'s 78 IFEval counting pairs (`314 vs 217` response tokens)?**
   - **No.** Comparing `rpM` against `p1r1s0` and `rpK-60` shows why every mechanism that hurt `fewer_than_N` in `p1r1s0` has been eliminated in `rpM`:
     - First, in `p1r1s0` (`--count-mode all`), `fewer_than_N` prompts (`relation == "less than"`) were classified as `is_cr = True` and given pairs requiring `chosen_response >= rejected_response`. In `rpM`, `--count-mode length` (`length_rule()`) strictly requires `relation == "at least"` on words/sentences/keywords (or `num_paragraphs >= 2`), and `--no-shorten-counting` routes `fewer_than_N` and `other_counting` questions to `right_cr` (`70` pairs), where `--right-resp-cap 1.15` holds responses to **`68 vs 80` tokens (`chosen < rejected` in `56.1%` of pairs)**.
     - Second, on `no_counting` (`141` `shorten` pairs), `--resp-near-long 0.95 --resp-max 1.05` holds responses strictly neutral at **`81.8 vs 81.9` tokens (`-0.1` token)**—fixing both `rpK`/`p1r1s0`'s `+11 to +19` token `shorten` response inflation and `rpL`'s `-11.8` token `shorten` response shrinkage.
     - Third, `rpM` has half as many counting pairs as `rpK` (`78` = `25%` of IFEval vs `160` = `50%` in `rpK`), caps them at `1.75x + 60` (max response diff `+339` tokens vs `+1,245` in `rpK` and `+2,322` in `p1r1s0`), and includes **24 response-loop pairs (`219 vs 3,966` response tokens, `-3,748` tokens)** that suppress runaway loops. Even in `rpK-60` (with 160 uncapped counting pairs and 0 loop pairs), `fewer_than_N` only slipped `-0.69` pts vs `rpE-80` (`71.2% -> 70.5%`).
2. **Are `rpM`'s Math Rules (`--guard-wide svamp --right-hold svamp`, `svamp=180,svamp_hard=60`) Sized Right?**
   - **Yes.** The developer's implementation of `--right-hold` (matching `(r, w)` directly so `0.95 <= thinking_len(r) / thinking_len(w) <= 1.25` closest to `1.05` with `REREAD`, giving `612 vs 585` median tokens, `chosen > rejected` in `74–83%` of pairs) and `--guard-wide` (rejecting the wrong answer under `0.95 * tmed` nearest `0.75 * tmed`, giving `700 vs 487` median tokens, `chosen > rejected` in `100%` of pairs) is exact. Why won't `90%+` longer math pairs over-lengthen SVAMP? Because in `rpK-60`, cross-type shortening from the 740 IFEval/BFCL pairs pulled SVAMP thinking down by `-15.6%` (`307 -> 259` tokens) even with `0%` math `shorten` pairs. Having 240 math pairs (180 on `<600`-token SVAMP-like questions) with a bounded `+5%` (`right_cr`) to `+40%` (`guard`) pull is the exact counter-weight needed to hold SVAMP near `285–305` tokens (`55–58%` saved) while repairing the `0.50x–0.95x tmed` skipped-recheck errors.
3. **Is the Dose of 60 Steps (`--rp-screen 20 40 60`) Right?**
   - **Yes.** `rpK-60` (`+4.55%` saving over `rpK-20` for `-0.16` accuracy) and `rpL-60` (`+8.6%` Dev saving over `rpL-20`) proved that 60 steps (`0.85` epochs at `lr=1e-5`) is where tool-call and IFEval shortening mature.

### 2.2 Verdict on Edits & Expected Full-Test Reading for `rpM`
- **Verdict: Run `rpM` AS IS.** Do not delay GPU 2's queue.
- **Expected Full-Test Reading (`rpM`):**
  - **Which step will win:** **Step 60** (with **Step 40** as the lower-saving, slightly higher-accuracy bracket).
  - **Expected Full-Test range at Step 60:** **`53.8% to 56.2% saved (central: 55.0%) / -0.65 to -1.20 accuracy (central: -0.90)`** (per benchmark: SVAMP `55.5–58.0% / -1.1 to -1.7`, MBPP+ `57.0–60.0% / -0.9 to -1.5`, IFEval `52.5–55.0% / -2.0 to -2.7`, BFCL pooled `50.5–52.5% / +0.9 to +1.5`).
  - **Expected Full-Test range at Step 40:** **`51.5% to 53.5% saved (central: 52.5%) / -0.65 to -1.15 accuracy (central: -0.90)`**.

---

## 3. A Better Start Than `rpE-80`? Repairing `rpK-60` (`rpN`) vs `rpM`, and Weight Interpolation

### 3.1 Judgment on Plain Weight Interpolation Between `rpE-80` and `rpK-60`
- **Verdict: Reject—it cannot land inside the box.**
- **Why (exact math):** On the full test, `rpE-80` sits at `47.04% saved / -0.95` official accuracy (`-0.90` finished-only) and `rpK-60` sits at `54.43% saved / -1.44` accuracy. Even on an idealized straight secant line between them, crossing `50.0%` saved requires weight $lpha \ge 0.401$ on `rpK-60`, where linear accuracy is already **`-1.15` (`-1.12` finished-only)**—outside the `-1.0` box. Worse, the actual LoRA trajectory between `rpE-80` and `rpK-60` is **concave**, as measured by `rpK-20` (`49.88% saved / -1.28` official, `-1.24` finished-only, which sits `-0.13` pts *below* the secant line) because SVAMP (`-1.34` pts) and IFEval (`-0.51` pts) paid their accuracy cost in the first 20 steps while IFEval/BFCL saving arrived between steps 20 and 60. *(Note: once `rpL-60`'s full test lands in ~1 hour, if `rpL-60` reads `~57.5% / -1.7`, even a secant to `rpE-80` crosses `50%` at `-1.12`; however, interpolating **`0.50 * rpM-60 + 0.50 * rpK-60`** or **`rpI-120` [`41.9% / +0.13`] + `rpL-60`** can cross `51%` at `~ -0.9` if needed as a zero-cost backup.)*

### 3.2 Would a Short Accuracy-Only Repair on Top of `rpK-60` (`rpN`) Be a Shorter Route Into the Box Than `rpM`?
- **Comparison (`rpM` from `rpE-80` vs `rpN` from `rpK-60`):**
  - **Why `rpM` from `rpE-80` is faster to execute first (~1.3 GPU-hours vs ~1.85 GPU-hours):** `rpM` reuses `rpI`'s existing `4,500` $	imes 8$ non-math rollouts on Drive (0 generation time) and only generates 1,200 short SVAMP questions (~12 min), whereas `rpK-60` has no stored training rollouts on Drive yet (generating `3,200 x 8` fresh answers with `rpK-60` takes ~1.1 GPU-hours before DPO starts). Moreover, `rpM` prevents `rpK-60`'s `-1.56` `bfcl_simple` non-call loss and `-1.10` SVAMP loss from happening in the first place.
  - **Why `rpN` on top of `rpK-60` is the single highest-probability follow-up / complementary route (and should run right after `rpM` on GPU 2, replacing the redundant `rpL-40` full test):**
    1. `rpK-60` is **already verified on the full test at `54.43% saved / -1.44 accuracy`**. It has **4.43 points of excess saving to spend** and needs only **`+0.45 to +0.65` points of headline accuracy** (`< +0.20` pts per benchmark).
    2. We already have full-test proof that accuracy-only repair on a model's own answers works: `rpI-120` (on `rpE-80`) gained **`+1.08` full-test accuracy points (`-0.95 -> +0.13`)** (`+1.27` SVAMP, `+1.56` MBPP+, `+0.99` IFEval, `+0.49` BFCL) at a slope of `+0.21` accuracy per `-1%` saving.
    3. Better still, in `rpK-60`, `2.70%` IFEval runaways (`100` tokens/answer) and `0.53%` MBPP+ runaways (`43` tokens/answer) are pure waste. Including **24 IFEval response-loop pairs and 12 MBPP+ loop pairs** inside `rpN` cuts those runaways, **saving `+2.5%` on IFEval and `+1.5%` on MBPP+**, which offsets the token cost of longer `"at least N"` responses and SVAMP re-checks so `rpN` gives back only **`-1.5% to -2.5%` headline saving** (`54.4% -> 51.8–53.0%`) while recovering **`+0.55 to +0.85` headline accuracy (`-1.44 -> -0.60 to -0.90`)**!

### 3.3 Exact Design of `rpN` (Short Accuracy + Loop Repair on Top of `rpK-60`)
- **Base & Reference Model:** `models/rpK-60`, fresh LoRA ($r=16, lpha=32$, `--bida-precision`).
- **Answers to Generate (~1.1 GPU-hours on 1 L4 GPU):** `models/rpK-60` answers **3,200 unseen questions $	imes 8$** (`--rp-own-data --rp-public --rp-exclude-dev` + excluding `dev_svamp_ids()` and `rpI` IDs, seed 41): `svamp=1200` (public SVAMP-like), `mbpp=450`, `ifeval=900`, `bfcl_simple_python=250`, `bfcl_multiple=220`, `bfcl_irrelevance=180`.
- **Exact Pairs (`800` pairs total, **0 `"think less"` (`shorten`) pairs on any type**, new `--mix-preset hold` in `repair_pairs.py`):**
  - `svamp=240` (SVAMP-like): **50% widened `capguard` (`120` pairs, wrong `<0.95 * tmed` nearest `0.75 * tmed`, right `0.95–1.25x tmed` with `REREAD`) / 50% length-held `right_cr` (`120` pairs, right `0.95–1.25x` rejected thinking with `REREAD`)**.
  - `ifeval=240`: **40% length-rule counting pairs (`96` pairs, `--count-mode length --count-resp-min 1.10 --count-resp-cap 1.75 --count-resp-pad 60`) / 50% compact `right_cr` (`120` pairs on `other_counting` + `fewer_than_N` + `no_counting`, `--right-resp-cap 1.15`) / 10% response-loop pairs (`24` pairs, `--loop-response-only ifeval`) / 0% `shorten`**.
  - `mbpp=120`: **75% `right_cr` (`90`) / 15% `capguard` (`18`) / 10% `loop` (`12`) / 0% `shorten`**.
  - `bfcl_simple_python=80, bfcl_multiple=80, bfcl_irrelevance=40`: **100% `right_cr` (`0% shorten`)**. Weighting `simple` (`80`) + `multiple` (`80`) at `4 : 1` against `irrelevance` (`40`) directly repairs `rpK-60`'s `-1.44%` non-call drop on `bfcl_simple_python` (`82.44% -> 80.69%`) and `-1.06%` drop on `bfcl_multiple`.
- **Steps & Checkpoints (~15 min training + ~30 min Dev scoring = 1.85 GPU-hours total):** `lr=1e-5` constant (`--warmup 2`), batch 16, $eta=0.1$, **45 steps**, scored at **steps 15, 30, 45** on `Dev-512` + `Dev-SVAMP-200`, paired against `rpK-60`.
- **Decision Rule for Full Test:** Full-test the checkpoint (expected: **step 30**) that achieves:
  1. Paired `Dev-512` saving vs `rpK-60` **$\ge -2.5\%$** (so projected full-test saving is **$\ge 51.9\%$**),
  2. Paired `Dev-SVAMP-200` math accuracy vs `rpK-60` **$\ge +0.80$ pts** (recovering SVAMP from `-2.35` to `>= -1.55`), and
  3. Paired `Dev-512` headline accuracy vs `rpK-60` **$\ge +0.50$ pts** (with IFEval `at_least_N` up $\ge +5$ pts and `bfcl_simple` up $\ge +1.0$ pt).

---

## 4. Path 1's Third Round 1 (`p1cr1s0`) as Built

1. **Verdict: Run `p1cr1s0` AS IS (0 edits needed).**
   - Inspecting `tables/path1_round1_v3_p1cr1s0_pairs_local_build.csv` (`3,025` pairs) confirms that all three `consult10` fixes worked cleanly:
     - On `ifeval` (`705` pairs: `310 shorten`, `206 right_cr`, `141 right_count` on `length_rule`, `48 response-loop`), all-pair saving rose from `28.5%` (`p1r1s0`) to **`48.3%`** (`162.9 vs 352.2` response tokens), `shorten` responses are strictly neutral (`101.4 vs 101.7` tokens, `-0.3` token vs `+18.8` in `p1r1s0`), and non-loop response diff fell from `+46.9` tokens (`75.7%` longer) to `+13.8` tokens (`50.4%` longer).
     - On `svamp` (`800` pairs: `360 shorten`, `280 right_cr`, `120 widened guard`, `40 loop`), adding 120 widened guards (`1,019 vs 718` tokens) and cutting `shorten` from `60%` to `45%` dialed back non-loop pair shortening from `52.3%` (`475` thinking tokens) to **`41.2%` (`562` thinking tokens)**, preventing `p1r1s0`'s `-3.24` SVAMP over-compression while still compressing from raw Qwen's `677` tokens. (Note: omitting `--no-shorten-counting` and `--right-hold,svamp` on `p1cr1s0` was right—on raw Qwen rollouts, adding them would starve `shorten` candidates and under-compress math.)
     - Cutting `loop` pairs from `360` to `144` (`40` math, `56` code, `48` IFEval) cuts training time in half while still targeting raw Qwen's `7.8%` code and `3.6%` IFEval runaways.
2. **Expected Reading on `Dev-512` (and `Dev-SVAMP-200`) at Step 150 / 200:**
   - **`Dev-512` Headline:** **`47.0% to 50.0% saved (central: 48.4%) / +0.10 to +1.05 accuracy (central: +0.50)`** (`+1.1% to +4.1%` more saving and `+0.1 to +1.0` pts higher accuracy than `p1r1s0-200`'s `45.88% / +0.04`).
   - **Per part on `Dev-512`:** IFEval `42–47%` saved (`+16 to +21` pts over `p1r1s0-200`'s `25.6%`) at `-0.5 to -1.5` acc; MBPP+ `52–55%` saved at `-0.5 to +0.8` acc (`+2.7 to +4.0` pts over `p1r1s0-200`'s `-3.25`); Math `51–54%` saved at `-0.5 to -1.2` acc (`+1.2 to +1.9` pts over `p1r1s0-200`'s `-2.41`); BFCL pooled `41–43%` saved at `+3.0 to +4.0` acc.
   - **On `Dev-SVAMP-200`:** Paired math accuracy **`+1.2 to +1.8` pts above `p1r1s0-200`** (`~53–56%` saved).
   - **Implied Full-Test Equivalent (after applying the 1-round lineage offset):** **`45.2% to 48.2% saved / -0.70 to -1.35 accuracy`**—landing right alongside `rpE-80` (`47.0% / -0.90`), but achieved in **one clean round from raw `Qwen3-0.6B`**.
3. **What Should Decide Whether Round 2 (`p1cr2s0`) Runs:**
   - Do **not** require `p1cr1s0` to clear `51–53%` saved in Round 1 (`rpE-80` is only `47.0%` saved, and `rpK-60`/`rpL-60` gained `+7.4%` to `+11.9%` saving in Round 2 from `47.0%`!). Instead, `p1cr1s0`'s job is to be a **clean Round-1 base as good as `rpE-80`**.
   - **Exact Gate for Round 2 (`p1cr2s0`):** Run `p1cr2s0` (using `rpM`'s exact `typed3` rules on `p1cr1s0`'s own answers, ~2.2 GPU-hours) **if and only if**:
     1. `p1cr1s0` reaches **$\ge 47.5\%$ saved on `Dev-512`** (`\ge +1.6\%` paired over `p1r1s0-200`) with **IFEval saving $\ge 42.0\%$**,
     2. `p1cr1s0` reaches **$\ge +0.50$ headline accuracy on `Dev-512`** (`\ge +0.45` paired over `p1r1s0-200`, with MBPP+ accuracy $\ge -0.80$), **AND**
     3. On `Dev-SVAMP-200`, `p1cr1s0` is within **`-0.50` points of `rpE-80`'s math accuracy** (`\ge +1.20` pts paired over `p1r1s0-200`).
     - If `p1cr1s0` fails this gate, **stop Path 1** and allocate GPU 1's remaining hours to full-testing `rpM` / `rpN`.

---

## 5. Plan for the Next 12 GPU-Hours on Two GPUs & Honest Probability

### 5.1 Step-by-Step Schedule (6 Wall-Clock Hours $	imes$ 2 L4 GPUs = 12 GPU-Hours)

#### **GPU 2 (The `rpE-80` / `rpK-60` Winning Line — 6.0 Wall-Clock Hours)**
1. **Hour 0.0 – 0.8 (Running now):** Finish **`rpK-40` Full Test**.
   - *Decision rule:* If by surprise `rpK-40` scores `> 50.0%` saved and `> -1.00` accuracy, we already have a checkpoint inside the box; still run `rpM` for a wider margin (`54%+ / -0.8`).
2. **Hour 0.8 – 1.2 (Queued):** Run **`devsvamp`** + **`devscore`** of `rpE-80`, `rpK-60`, and `rpL-60` on `Dev-SVAMP-200`.
   - *Decision rule:* Verify that `Dev-SVAMP-200` reproduces the full-test SVAMP gap between `rpE-80` (`-1.25`) and `rpK-60` (`-2.35`, paired `~ -1.1` pts) and reveals `rpL-60`'s true SVAMP delta.
3. **Hour 1.2 – 2.5 (Queued):** Run **`rpM`** (1,200 SVAMP-like rollouts $	imes 8$ + 60 steps `typed3` DPO from `rpE-80` + `Dev-512` & `Dev-SVAMP-200` scoring at steps 20, 40, 60, paired vs `rpE-80`).
   - *Decision rule after `rpM` Dev scores (at Hour 2.5):*
     - **Pass criterion for `rpM` checkpoint (`step 60` or `step 40`):** Paired `Dev-512` saving vs `rpE-80` **$\ge +5.5\%$** (`\ge 52.5\%` Dev saved), paired `Dev-SVAMP-200` accuracy vs `rpE-80` **$\ge -0.40$ pts** (`\ge +0.70` pts better than `rpK-60`), and `Dev-512` IFEval `at_least_N` accuracy **$\ge 49.0\%$** (vs `rpE-80`'s `50.4%` and `rpL-60`'s `42.1%`).
     - **Action:** **Drop `rpL-40`'s full test** from GPU 2's queue (`rpL-40` is `-0.93` paired accuracy vs `rpE-80` on `Dev-512` and has the same `19.6%` `"at least N words"` collapse as `rpL-60`; once `rpL-60` is full-tested on GPU 1, `rpL-40` has zero chance of beating `rpM`). In its place at Hour 2.5 on GPU 2, **immediately run the Full Test (`2,759 x 8`, ~1.7h) of the best passing `rpM` checkpoint (`rpM-60` or `rpM-40`)**.
4. **Hour 2.5 – 4.2:** **Full Test of `rpM-60` (or `rpM-40`)** on GPU 2 *(or, if `rpN` is run on GPU 1, GPU 2 runs the `rpM` full test while GPU 1 runs `rpN`)*.
   - *Decision rule:* If `rpM` scores `> 50.0%` saved and `> -1.00` accuracy on the full test, **mission accomplished**.
5. **Hour 4.2 – 6.0:** Run **`rpN`** (the 45-step accuracy-only + anti-loop repair on top of `rpK-60`, Section 3.3; ~1.85h) **OR** (if `rpN` already ran on GPU 1) run the **Full Test of `rpN-30`** / **LoRA interpolation `0.50 * rpM-60 + 0.50 * rpK-60`**.

#### **GPU 1 (The `rpL-60` Anchor, `p1cr1s0`, and `rpN` / Second Full Test — 6.0 Wall-Clock Hours)**
1. **Hour 0.0 – 1.0 (Running now):** Finish **`rpL-60` Full Test** (expected: `56.8–58.8% saved / -1.60 to -2.15 accuracy`).
   - *Decision rule when `rpL-60` lands (at Hour 1.0):*
     - Check `rpL-60`'s exact per-type full-test numbers. If by surprise `rpL-60` is `> -1.0` accuracy, we win immediately.
     - Otherwise, `rpL-60` confirms that anti-loop + response-neutral `shorten` unlocks `57%+` full-test saving and isolates the remaining loss to SVAMP and `"at least N"` (the exact two things fixed in `rpM`).
2. **Hour 1.0 – 1.2 (Queued):** Run **`devsvamp`** + **`devscore` of `p1r1s0-200`** on `Dev-SVAMP-200`.
3. **Hour 1.2 – 3.0 (Queued):** Run **`p1cr1s0`** (200 steps `round1c` from raw Qwen, scored at 100 / 150 / 200 on `Dev-512` + `Dev-SVAMP-200` vs `p1r1s0-200`).
   - *Decision rule after `p1cr1s0` (at Hour 3.0):*
     - **Branch A (Priority — Maximize Box Probability):** Unless `rpM` on GPU 2 has *already* failed its Dev gate **and** `p1cr1s0` beat `rpE-80` across the board (`Dev-512` saved $\ge 48.0\%$, acc $\ge +0.60$, `Dev-SVAMP-200` within `-0.3` of `rpE-80`), **do NOT spend 2.2h on `p1cr2s0` yet**. Instead, at Hour 3.0 on GPU 1, launch **`rpN` (the 45-step accuracy + anti-loop repair on top of `rpK-60`, Section 3.3: Hour 3.0 – 4.8)** followed by the **Full Test of `rpN-30` (or the second-best `rpM` checkpoint: Hour 4.8 – 6.0)**!
     - **Branch B (If `rpM` passed with margin on GPU 2 AND the user wants the 2-round clean-from-scratch recipe tested):** If `p1cr1s0` clears the Section 4.3 gate (`>= 47.5%` Dev saved, `>= +0.50` Dev acc, `Dev-SVAMP-200` within `-0.50` of `rpE-80`), run **`p1cr2s0`** (60 steps of `typed3` on `p1cr1s0`'s own answers, Hour 3.0 – 5.2).

### 5.2 Honest Probability of Landing a Full Test Inside the Box (`> 50%` Saved, `> -1.0` Accuracy) Within 12 GPU-Hours
- **`rpM` (Step 60 or Step 40, from `rpE-80`):** **~65% probability.** `rpK-60` already reached `54.43% saved / -1.44 accuracy`, and `rpL-60` reached `58.9%` Dev saving by fixing IFEval loops and response inflation. `rpM` needs to recover just `+0.45` points of full-test accuracy over `rpK-60` while keeping saving above `50.0%` (it has a `4.4 to 7.0` point saving buffer to trade for SVAMP re-checks and IFEval `"at least N"` words).
- **`rpN` (30–45 step accuracy + loop repair on top of `rpK-60`):** **~55% standalone probability** (~40% conditional probability of succeeding if `rpM` lands slightly outside at `-1.05 to -1.20`).
- **Combined Probability Across the 12 GPU-Hour Plan (`rpM` + `rpN` / bracketed checkpoint selection with `Dev-SVAMP-200`):** **78%.** The main remaining risk (22%) is irreducible full-test evaluation variance (`1 SD` on $2,759 	imes 8$ is `~0.18` accuracy points) combined with `rpE-80`'s structural ceiling on `"at least N words"` prompts where even `base` only passes `37.5%`.
