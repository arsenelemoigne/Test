### 1. Where Remaining Accuracy Is Lost in `r2s3-s20` / `s30`, Where Saving Is Left on the Table, and Design Implications

Starting from **`models/r2s2-s80-on-r2s1b-s180`** (`53.7%` saved / `-1.27` accuracy, `+2.36` pts above the curve), `r2s3` landed inside the objective at **`r2s3-s20` (`52.7% / -0.99`)** and **`r2s3-s30` (`51.5% / -0.84`)**, giving back `1.0%–2.2%` saving for `+0.28–0.43` accuracy. Under official benchmark weights (`0.25` Math, `0.25` Code, `0.25` IFEval, `0.1190` Simple, `0.0595` Several, `0.0714` No-Fit), `r2s3-s20` loses **`-5.553` gross macro pts** and gains **`+4.560`** (net **`-0.994`**); `r2s3-s30` loses **`-5.308` gross macro pts** and gains **`+4.470`** (net **`-0.839`**), vs. `r2s2-s80`'s **`-5.797` lost / `+4.529` gained** (net **`-1.268`**).

#### A. Remaining Accuracy Loss by Type (`r2s2-s80` $\rightarrow$ `r2s3-s20` $\rightarrow$ `r2s3-s30`)
1. **Math (`svamp`, 1,000 q, weight `0.2500`):** Net **`-3.25` $\rightarrow$ `-2.86` (`s20`) $\rightarrow$ `-2.80` (`s30`)** = **`-0.715` / `-0.700` macro pts** (`72%–83%` of total net deficit). Gross loss is `-5.46` (`232` q) / `-5.64` (`247` q), with **`57%–59%` (`3.09` / `3.30` type pts) on `orig 8/8` questions** and **`30%` (`1.61` / `1.70` pts) on `orig 6–7/8`**. On lost questions, the model's *own right answers* think **`0.47x` / `0.49x`** of the original's right median $b_{\text{think}}$.
2. **Code (`mbpp`, 378 q, weight `0.2500`):** Net **`-2.38` $\rightarrow$ `-1.92` (`s20`) $\rightarrow$ `-1.62` (`s30`)** = **`-0.480` / `-0.405` macro pts**. Gross loss is `-6.02` (`95` q) / `-5.39` (`94` q), with `62%–67%` on `orig 6–8/8` questions; right answers on lost questions think **`0.47x`** $b_{\text{think}}$.
3. **Simple Tool Calls (`bfcl_simple_python`, 400 q, weight `0.1190`):** Net **`-4.69` $\rightarrow$ `-3.50` (`s20`) $\rightarrow$ `-3.78` (`s30`)** = **`-0.417` / `-0.450` macro pts**. Gross loss is `-7.19` (`80` q) / `-7.53` (`81` q), with **`82%` (`5.91` / `6.15` pts) on `orig 6–8/8` questions**; right answers on lost questions think **`0.53x` / `0.50x`** $b_{\text{think}}$.
4. **Several Tool Calls (`bfcl_multiple`, 200 q, weight `0.0595`):** Net **`-0.69` $\rightarrow$ `-0.19` (`s20`) $\rightarrow$ `-0.56` (`s30`)** = **`-0.011` / `-0.033` macro pts** (gross loss `-7.19` / `-7.31` offset by `+7.00` / `+6.75` gained).
5. **IFEval (`ifeval`, 541 q, weight `0.2500`):** Net **`+1.20` $\rightarrow$ `+0.88` (`s20`) $\rightarrow$ `+1.39` (`s30`)** = **`+0.220` / `+0.347` macro pts**.
6. **No-Fit (`bfcl_irrelevance`, 240 q, weight `0.0714`):** Net **`+6.15` $\rightarrow$ `+5.73` (`s20`) $\rightarrow$ `+5.62` (`s30`)** = **`+0.409` / `+0.401` macro pts**.

#### B. Remaining Accuracy Loss by Cause (`s20` / `s30` vs. `r2s2-s80` and 4-vs-4 Noise Reference)
1. **Stopped Thinking Early (`< 0.5x` $b_{\text{think}}$) — `-3.197` (`s20`) / `-2.931` (`s30`) macro pts (`57.6%` / `55.2%` of gross loss; Noise `-0.157`):**
   - Per type (`s20` / `s30`): Math `3.07` / `2.89` (`56%` / `51%`), Code `3.22` / `2.99` (`53%` / `56%`), Simple `3.37` / `3.35` (`47%` / `44%`), Multiple `3.89` / `3.87` (`54%` / `53%`), IFEval `3.80` / `3.18` (`71%` / `69%`).
2. **Somewhat Short Thinking (`0.5–0.8x` $b_{\text{think}}$) — `-1.376` (`s20`) / `-1.358` (`s30`) macro pts (`24.8%` / `25.6%`; Noise `-0.557`):**
   - Per type (`s20` / `s30`): Simple `3.14` / `3.27` (`44%` / `43%`—skipped parameter/type re-check), Code `1.63` / `1.27` (`27%` / `24%`), Math `1.28` / `1.54` (`23%` / `27%`), Multiple `1.58` / `1.56` (`22%` / `21%`).
   - Together, **short thinking (`< 0.8x` $b_{\text{think}}$) still causes `82.4%` (`-4.573` macro pts at `s20`) and `80.8%` (`-4.289` macro pts at `s30`) of all gross lost accuracy** (`-3.859` / `-3.575` excess over Noise).
3. **Repetition Loops + Cut Off at Cap — `-0.510` (`s20`) / `-0.485` (`s30`) macro pts (vs. `-0.480` in `r2s2-s80`):**
   - Loops *increased* slightly from `r2s2-s80` on Math (`0.49` $\rightarrow$ `0.60` / `0.63`) and Code (`0.55` $\rightarrow$ `0.82` / `0.69`), while falling on IFEval (`0.66` $\rightarrow$ `0.46` / `0.39`).
   - **Code root cause (`repair_pairs.py` `L425–427` vs `L482–519`):** `--loop-finished` only filtered `loops8(r)` out of `loop_chosen` (`L427`), **not** out of `rights` in `right_cr` or `capguard`! Thus `be-right` and `guard` pairs on Math/Code could still pick a right answer containing a finished 8-word repetition loop as `chosen`.
4. **Unparseable / Wrong Format — `-0.218` (`s20`) / `-0.235` (`s30`) macro pts (vs. `-0.219` in `r2s2-s80`):**
   - Concentrated in **Several Tools (`1.67` / `1.83` type pts = `23%` / `25%` of Multiple loss)** and **Simple Tools (`0.68` / `0.83` pts = `9%` / `11%`)**, plus Math (`0.13` / `0.11`). `--right-hold-format` (`L497`) missed most broken-format tool answers because it required the broken wrong answer `w` to also satisfy `0.90 <= thinking_len(r)/thinking_len(w) <= 1.15`.
5. **IFEval Response Too Short (`< 0.7x` $b_{\text{resp}}$) — `-0.075` (`s20`) / `-0.060` (`s30`) macro pts** (`0.30` / `0.24` type pts, down from `0.36` in `r2s2-s80`).

#### C. Where Saving Is Left on the Table vs. the Safe Per-Type Frontier
Comparing `r2s3-s20` (`52.7%`) and `r2s3-s30` (`51.5%`) against the empirical per-type frontier (`pertype_fulltest_points.json` + `NEW_RESULTS_13nov_afternoon.md`):
- **IFEval (`48%` saved / `+0.88` at `s20`, `47%` / `+1.39` at `s30`; weight `0.2500`):** Safe frontier with counting-rule protection is **`53%–56%` saved** (`rpL-40` `55.0%`, `rpL-60` `57.9%`, `od4-r7` `54% / +3.0`, `r2s2-s80` Dev `56% / +4.6`). **`+5% to +7%` type saving (`+1.25% to +1.75%` headline saving)** is left on the table with a `+0.9` to `+1.4` accuracy buffer.
- **No-Fit (`52%` / `+5.73` at `s20`, `51%` / `+5.62` at `s30`; weight `0.0714`):** Thinking median is `0.61x` $b_{\text{think}}$ (least compressed of any type). Safe frontier reaches **`61%–66%` saved at `+4.2` to `+7.7` accuracy** (`rpM-60` `60.7% / +4.17`, `rpK-60` `61.9% / +6.20`, `rpL-80` `65.9% / +7.71`). Putting No-Fit into `--right-hold` in `r2s3` made `chosen` longer in `72%` of pairs and *lost* `2%–3%` saving. **`+8% to +11%` type saving (`+0.57% to +0.79%` headline saving)** is safely available.
- **Code (`53%` / `-1.92` at `s20`, `52%` / `-1.62` at `s30`; weight `0.2500`):** Safe frontier is **`58%–62%` saved at `-0.86` accuracy** (`rpM-60` `58.2% / -0.86`, `rpL-60` `62.2% / -0.86`). In `r2s3`, Code gave back `2%` saving because `chosen` was longer in `58%` of pairs while loops were not filtered from `rights`. Keeping Code at **`54%–56%` saved** while fixing loops and early stops adds **`+0.50% to +0.75%` headline saving** and **`+0.3` to `+0.6` type accuracy**.
- **Math (`59%` / `-2.86` at `s20`, `57%` / `-2.80` at `s30`; weight `0.2500`):** Safe frontier is **`56%–58%` saved at `-1.20` to `-1.50` accuracy** (`rpM-60` `57.0% / -1.20`). Math needs `+0.7` to `+1.1` pts better accuracy at **`57%–58%` saved**.
- **Simple (`48%` / `-3.50` at `s20`, `48%` / `-3.78` at `s30`; weight `0.1190`) & Several (`51%` / `-0.19`, `50%` / `-0.56`; weight `0.0595`):** Simple Tools is **over-shortened past its `45%` cliff** (`rpM-60` `45.1% / -0.34` vs `rpL-80` `51.4% / -5.06`). Letting Simple relax by `2%` (`48%` $\rightarrow$ `46%` saved) costs only `-0.24%` headline saving while recovering `+1.1` to `+1.5` type accuracy (`+0.13` to `+0.18` macro pts).

#### D. Why `r2s3` Under-Repaired Damaged Types While Lengthening Safe Types (Design Implications)
1. **Flaw 1 (`--right-hold` matched early-stop wrong answers):** In `repair_pairs.py` (`L491–502`), `right_hold` matched `thinking_len(r)` to `thinking_len(w)` in `[0.90, 1.15]` *without checking if `w` stopped early*. Since `80%–91%` of wrong answers on Math/Code/Simple are premature stops (`< 0.8x` $b_{\text{think}}$), `right_hold` (`60%–75%` of pairs) either rejected the question or selected a lucky ultra-short right answer that also skipped verification—fighting `--guard-wide`.
2. **Flaw 2 (`--guard-wide` used compressed `tmed` instead of original `b_think`):** On lost questions where `r2s2-s80` only got 1–3 right answers thinking `0.45x` $b_{\text{think}}$, `0.85 * tmed` was `0.38x` $b_{\text{think}}$, missing the `0.40x–0.80x` $b_{\text{think}}$ failure band. Using `--base-think data/od3-base-think.json` (already on Drive from `od3prep`) lets guards target `w < 0.85 * max(tmed, 0.70 * bt)` and pick a re-reading right answer near `min(0.90 * bt, 1.30 * tmed)`, while restricting `right_hold` to non-early wrong answers (`w >= 0.65 * tmed`).
3. **Flaw 3 (Zero shortening on safe types):** Putting `ifeval` (`0%` shorten) and `bfcl_irrelevance` (`100%` `right_hold`, `72%` chosen longer) in pure-hold mode wasted `+1.5%` to `+2.2%` of free headline saving on types that have `+1.20` and `+6.15` accuracy cushion.

---

### 2. The Plan from `r2s2-s80`: One Single-Job-Per-Type Step (`r2s3b`)

**Verdict: ONE step (`r2s3b`) from `models/r2s2-s80-on-r2s1b-s180`, reusing `r2s2-s80`'s `3,774` already-generated test-matched rollouts (`rp-r2s3`, `rp-r2s3-m1`, `rp-r2s3-m2`).**
- **Why One Step Beats Two Steps:** Under the user's single-job-per-type rule (*"never shorten and protect the SAME type in one step"*), the types that need protection (**Math, Code, Simple Tools, Several Tools**: `0%` shorten) and the types with untapped safe saving (**IFEval, No-Fit Tools**: `+1.20` and `+6.15` accuracy in `r2s2-s80`) are **disjoint sets of types**. Repairing the 4 damaged/at-edge types while safely shortening (`>= 7/8` solved) the 2 accuracy-positive types in one pass from `r2s2-s80`:
  1. Keeps $\pi_{\text{ref}} = \text{r2s2-s80}$ (one clean adapter, avoiding reference drift and a second Dev-gate error).
  2. Offsets the slight lengthening of Math/Simple repair with safe compression on IFEval (`~49%` $\rightarrow$ `53%–54%`) and No-Fit (`~54%` $\rightarrow$ `60%–62%`), holding headline saving at **`52.8%–53.6%`** while lifting accuracy to **`-0.55` to `-0.75`**.
  3. Reuses all `3,774 x 8 = 30,192` answers already generated by `r2s2-s80` in `r2s3` (**`0` new rollout generation time**; training + Dev takes ~35 min on one L4).

#### A. Questions (Test-Matched, Already Answered by `r2s2-s80` on Drive)
- **Existing `r2s2-s80` rollouts (`3,774` questions x 8, test-matched to `test_difficulty.json`):**
  - `data/r2-step3-questions.jsonl` (`2,134` q: `svamp=650`, `mbpp=339`, `ifeval=566`, `bfcl_simple_python=200`, `bfcl_multiple=159`, `bfcl_irrelevance=220`) $\rightarrow$ `runs/bon/rp-r2s3/rollouts.jsonl`.
  - `data/r2-step2-s3m1-questions.jsonl` (`1,240` q: `svamp=250`, `mbpp=260`, `bfcl_simple_python=450`, `bfcl_multiple=100`, `bfcl_irrelevance=180`, `ifeval=0`) $\rightarrow$ `runs/bon/rp-r2s3-m1/rollouts.jsonl`.
  - `data/r2-step2-s3m2-questions.jsonl` (`400` q: `bfcl_multiple=100`, `bfcl_irrelevance=300`) $\rightarrow$ `runs/bon/rp-r2s3-m2/rollouts.jsonl`.
  - **Total per type already answered by `r2s2-s80`:** `svamp=900`, `mbpp=599`, `ifeval=566`, `bfcl_simple_python=650`, `bfcl_multiple=359`, `bfcl_irrelevance=700`.
- **Why `0` new questions are needed:** In `r2s3`, the only shortfall across all `3,774` questions was `bfcl_irrelevance` (`25` of `32` `be-right` pairs) because `r2s2-s80` solves `98%` of No-Fit questions (`8/8`). In `r2s3b`, `bfcl_irrelevance` uses **`62.5%` `shorten` (`30` pairs from `>400` `>=7/8`-solved questions) + `37.5%` compact `right` (`18` pairs, well below the `25` available)**, so **every type meets its full quota** with zero new generation.

#### B. Pairs per Type and Kind (`800` Pairs Total = 50 Steps at Batch 16, Preset `r2s3b`, Fixed Shares)
| Type (`part`) | Quota (Share) | `shorten` (`>=7/8` right, `>=1.30x`) | `right` (`right_hold` or compact + format priority) | `guard` (Bounded Early-Stop Recovery) | `loop` (Finished + Runaway Loops) | `right_count` (IFEval `atleast` Rules) | Job on This Type |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`svamp`** (Math) | **`220`** (`27.5%`) | `0` (`0%`) | **`110`** (`50%`, `right_hold` `0.90–1.10x`) | **`88`** (`40%`, bounded guard) | **`22`** (`10%`) | `0` | Pure repair (early stops + loops + format) |
| **`mbpp`** (Code) | **`160`** (`20.0%`) | `0` (`0%`) | **`88`** (`55%`, `right_hold` `0.90–1.10x`) | **`40`** (`25%`, bounded guard) | **`32`** (`20%`) | `0` | Pure repair + compact hold (keep `54–56%` saved) |
| **`bfcl_simple_python`** | **`112`** (`14.0%`) | `0` (`0%`) | **`67`** (`60%`, `right_hold` `0.90–1.10x` + format) | **`45`** (`40%`, bounded guard) | `0` (`0%`) | `0` | Pure repair (`<0.8x` stops + missing-call format) |
| **`bfcl_multiple`** | **`48`** (`6.0%`) | `0` (`0%`) | **`36`** (`75%`, `right_hold` `0.90–1.10x` + format) | **`12`** (`25%`, bounded guard) | `0` (`0%`) | `0` | Pure repair (`1.7` pt format loss + `<0.5x` stops) |
| **`ifeval`** (IFEval) | **`212`** (`26.5%`) | **`95`** (`45%` total = `60%` of non-count) | **`40`** (`18.75%` total = `25%` of non-count, compact) | `0` (`0%`) | **`24`** (`11.25%` total = `15%` of non-count) | **`53`** (`25%` total) | Safe shorten (`>=7/8`, response held) + count/loop hold |
| **`bfcl_irrelevance`** | **`48`** (`6.0%`) | **`30`** (`62.5%`) | **`18`** (`37.5%`, compact `right_cr`) | `0` (`0%`) | `0` (`0%`) | `0` | Safe shorten (`0.61x` $\rightarrow$ `0.48x`) + compact hold |
| **Total (`r2s3b`)** | **`800`** (`100%`) | **`125`** (`15.6%`) | **`359`** (`44.9%`) | **`185`** (`23.1%`) | **`78`** (`9.8%`) | **`53`** (`6.6%`) | **4 types pure-repaired, 2 safe types shortened** |

#### C. Response & Thinking Rules
1. **Global Clean-Right Filter (`--loop-finished`):** Filter `loops8(r)` out of `rights` *before* building `shorten`, `right_cr`, `right_count`, and `capguard` (`rights = [r for r in rights if not loops8(r)] or rights`), so **no pair of any kind ever chooses a finished repetition loop**.
2. **Bounded Early-Stop Guard Using `--base-think` (`--guard-wide svamp+mbpp+bfcl_simple_python+bfcl_multiple --guard-wide-max 0.85 --base-think data/od3-base-think.json`):**
   - Let `ref_t = max(tmed, 0.70 * bt)` where `bt = base_think.get(str(qid), 0)` (falling back to `tmed` if `qid` is absent).
   - `early` = wrong answers with `thinking_len(w) < args.guard_wide_max * ref_t` (preferring broken-format wrong answers first, then nearest `0.70 * ref_t`).
   - `capped` = right answers with `max(0.95 * tmed, 1.08 * thinking_len(w) + 10) <= thinking_len(r) <= min(1.30 * tmed, 0.90 * bt if bt else 1.30 * tmed)`, preferring `REREAD`. This directly repairs the `0.40x–0.80x` $b_{\text{think}}$ premature-stop band (`82%` of gross loss) while capping `chosen` at `<= 0.90x` of the original's thinking (`<= 1.30x tmed`).
3. **Non-Early, Format-Aware `--right-hold` (`--right-hold svamp+mbpp+bfcl_simple_python+bfcl_multiple --right-hold-band 0.90,1.10,0.98 --right-hold-format`):**
   - Tighten the band from `[0.90, 1.15]` (center `1.00`) to **`[0.90, 1.10]` (center `0.98`)** so `right_hold` does not drift upward in length.
   - Require `thinking_len(w) >= 0.65 * tmed` in the length-matched loop so `right_hold` never matches `chosen` down to an early-stop wrong answer; if a question has a `broken(w, base_part)` wrong answer (missing tool call `(` or missing math final number or empty response), allow any clean `r` in `[0.80 * tmed, 1.15 * tmed]` to reject `w` even when `w`'s thinking length lies outside `[0.90, 1.10]`.
4. **Safe Shortening on `ifeval` and `bfcl_irrelevance` Only (`--shorten-min-right 0.875 --shorten-ratio 1.30`):**
   - Only on questions solved **`>= 7/8` (`0.875`)** by `r2s2-s80`.
   - On `ifeval`: `--count-rules --count-mode atleast --count-share 0.25 --count-resp-min 1.10 --count-resp-cap 1.75 --count-resp-pad 60 --no-shorten-counting --resp-median 0.95 --resp-max 1.05 --resp-near-long 0.92 --right-resp-cap 1.15 --loop-response-only ifeval`.
   - On `bfcl_irrelevance`: compact `right_cr` (removed from `--right-hold`) + `62.5%` `shorten` at `--floor-by-part bfcl_irrelevance=25:0.35`.

#### D. Hyperparameters, Plain `bf16` Precision, Steps, Checkpoints, and Calibrated Dev Gate
- **Training:** `--rp-base models/r2s2-s80-on-r2s1b-s180`, **`--rp-plain-precision`** (plain `bf16`, without `--bida-precision`), constant **`--rp-lr 8e-6`** (`--rp-lr-min 8e-6`, `--rp-warmup 2`), **`--rp-beta 0.1`**, LoRA rank 16, batch size 16, `--keep-order`, **`--rp-steps 50` (`--rp-epochs 1`)**, scored on Dev-512 + Dev-SVAMP-200 at **`--rp-save-at 20 30 40 50 --rp-screen 20 30 40 50`**.
- **Lineage Dev Bias (`Dev-512 minus Full Test`):**
  - `r2s1b-s180`: `+0.8%` saved / `+2.95` acc; `r2s2-s40`: `+2.1%` / `+4.09`; `r2s2-s80`: `+1.8%` / `+3.22`; `r2s3-s20`: `+0.1%` / `+3.53`; `r2s3-s30`: `+1.3%` / `+3.34`.
  - On Step-3 models from `r2s2-s80`, Dev-512 over-reads saving by `+0.1%` to `+1.3%` and accuracy by `+3.34` to `+3.53` pts.
- **Margin-Calibrated Gate (`fullpick`):** Set **`--fp-save-bias 1.3 --fp-acc-bias 3.50`** and **`--pk-save-lo 53.5 --pk-save-hi 56.5 --pk-acc 2.70 --pk-sv-save 51.0 --pk-sv-acc -0.45`**:
  - `Dev-512 saved >= 53.5%` $\rightarrow$ projects **`52.2%–53.4%` full-test saving** (`> 52.0%` even under `+1.3%` bias; `r2s3-s20`/`s30` read `52.8%`).
  - `Dev-512 acc >= +2.70` $\rightarrow$ projects **`-0.83` to `-0.64` full-test accuracy** (`> -1.00` even under `+3.53` bias; `r2s3-s20`/`s30` read `+2.54` / `+2.50`).
  - `Dev-SVAMP saved >= 51.0%` and `Dev-SVAMP acc >= -0.45` (vs `r2s3-s20` `50.1% / -0.69` and `s30` `49.4% / -0.38`).

---

### 3. Exact `track.py` Plan Lines and Small Code Changes

#### A. Small Code Changes
1. **`track.py` — `ptrepair()` (`L4586–4603`): Reuse stored `--rp-more-files` rollouts when `--rp-rollouts-from` (`reuse`) is set:**
   Currently `L4587` has `if own and not reuse and not smoke else []`, which ignores `rp-r2s3-m1` and `rp-r2s3-m2` when `--rp-rollouts-from r2s3` is passed. In `ptrepair()`, change `if own and not reuse and not smoke` to `if (own or reuse) and not smoke`, and check `roll_m_src = BON / f"rp-{reuse or name}-m{i_m}"` first (restoring from Drive and appending `roll_m_src / "rollouts.jsonl"` to `rolls` if it exists, else generating into `BON / f"rp-{name}-m{i_m}"`). This reuses all `3,774 x 8` stored answers (`rp-r2s3`, `rp-r2s3-m1`, `rp-r2s3-m2`) in 0 GPU minutes.
2. **`repair_pairs.py` — `main()`: Four small changes (`~20` lines total):**
   - **Add `"r2s3b"` to `--mix-preset` (`L129` and `L701`):**
     - `svamp`: `{"shorten": 0.0, "right": 0.50, "guard": 0.40, "loop": 0.10}`
     - `mbpp`: `{"shorten": 0.0, "right": 0.55, "guard": 0.25, "loop": 0.20}`
     - `bfcl_simple_python`: `{"shorten": 0.0, "right": 0.60, "guard": 0.40, "loop": 0.0}`
     - `bfcl_multiple`: `{"shorten": 0.0, "right": 0.75, "guard": 0.25, "loop": 0.0}`
     - `ifeval`: `{"shorten": 0.60, "right": 0.25, "guard": 0.0, "loop": 0.15}` with `count_share = 0.25`
     - `bfcl_irrelevance`: `{"shorten": 0.625, "right": 0.375, "guard": 0.0, "loop": 0.0}`.
   - **Exclude finished loops from all chosen answers when `--loop-finished` is set (`L425`):** After `if args.loop_finished:`, set `rights = [r for r in rights if not loops8(r)] or rights` so `shorten`, `right_cr`, `right_count`, and `capguard` never pick a looping right answer.
   - **Make `--right-hold` skip early-stop wrong answers and catch format errors (`L491–502`):** In `if base_part in right_hold:`, require `thinking_len(w) >= 0.65 * tmed` for the `[lo_, hi_]` ratio match; if `args.right_hold_format` is set and `best is None`, allow a fallback pair `(r, w)` where `broken(w, base_part)` is true and `0.80 * tmed <= thinking_len(r) <= 1.15 * tmed` (nearest `tmed`, preferring `REREAD`).
   - **Anchor `--guard-wide` to `max(tmed, 0.70 * bt)` when `--base-think` is provided (`L508–514`):** Set `bt = base_think.get(str(qid), 0)` and `ref_t = max(tmed, 0.70 * bt) if bt else tmed`; select `early` with `thinking_len(w) < args.guard_wide_max * ref_t` (preferring `broken(w, base_part)` first, then nearest `0.70 * ref_t`) and `capped` with `max(0.95 * tmed, 1.08 * thinking_len(early[-1]) + 10) <= thinking_len(r) <= (min(1.30 * tmed, 0.90 * bt) if bt else 1.30 * tmed)`.

#### B. Exact `track.py` Plan Lines (Plain Text)

ptrepair --name r2s3b --rp-base models/r2s2-s80-on-r2s1b-s180 --rp-own-data --rp-rollouts-from r2s3 --rp-more-files data/r2-step2-s3m1-questions.jsonl+data/r2-step2-s3m2-questions.jsonl --rp-samples 8 --rp-seed 53 --rp-kinds mix --rp-pair-args=--think-only,svamp+mbpp+ifeval+bfcl_simple_python+bfcl_multiple+bfcl_irrelevance,--prefer-reread,--count-rules,--mix-preset,r2s3b,--shorten-min-right,0.875,--shorten-ratio,1.30,--floor-by-part,bfcl_irrelevance=25:0.35,--base-think,data/od3-base-think.json,--count-mode,atleast,--count-share,0.25,--count-resp-min,1.10,--count-resp-cap,1.75,--count-resp-pad,60,--no-shorten-counting,--resp-median,0.95,--resp-max,1.05,--resp-near-long,0.92,--right-resp-cap,1.15,--loop-response-only,ifeval,--loop-finished,--guard-wide,svamp+mbpp+bfcl_simple_python+bfcl_multiple,--guard-wide-max,0.85,--right-hold,svamp+mbpp+bfcl_simple_python+bfcl_multiple,--right-hold-band,0.90+1.10+0.98,--right-hold-format --rp-quota svamp=220,mbpp=160,ifeval=212,bfcl_simple_python=112,bfcl_multiple=48,bfcl_irrelevance=48 --rp-require-full --rp-rebuild-if-short --rp-plain-precision --rp-lr 8e-6 --rp-lr-min 8e-6 --rp-warmup 2 --rp-beta 0.1 --rp-steps 50 --rp-epochs 1 --rp-save-at 20 30 40 50 --rp-screen 20 30 40 50 --rp-dev --rp-dev-svamp --rp-dev-vs rpE80 --rp-dpo-seed 0

fullpick --fp-run r2s3b --fp-kind dpo --pk-steps 20 30 40 50 --pk-save-lo 53.5 --pk-save-hi 56.5 --pk-acc 2.70 --pk-sv-save 51.0 --pk-sv-acc -0.45 --fp-save-bias 1.3 --fp-acc-bias 3.50 --fp-base models/r2s2-s80-on-r2s1b-s180

lossan --la-models fullpick:r2s3b --la-tag r2s3b

---

### 4. Expected Full-Test Landing (Per Type & Headline), Risks, and Fallback

#### A. Expected Full-Test Landing (`r2s3b-s30` / `s40`)
| Benchmark Type | Weight | `r2s2-s80` (Start) | `r2s3-s20` | `r2s3-s30` | **`r2s3b` Expected (Full Test)** | Why It Moves |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Math (`svamp`)** | `0.2500` | `~60% / -3.25` | `59% / -2.86` | `57% / -2.80` | **`57.5% / -2.15`** (`-0.54` macro) | `40%` `bt`-anchored guards + non-early `right_hold` + clean-right loop filter |
| **Code (`mbpp`)** | `0.2500` | `~54% / -2.38` | `53% / -1.92` | `52% / -1.62` | **`54.0% / -1.35`** (`-0.34` macro) | Loop filter on all `rights` + `0.98`-centered `right_hold` + `25%` guards |
| **IFEval (`ifeval`)** | `0.2500` | `~49% / +1.20` | `48% / +0.88` | `47% / +1.39` | **`53.0% / +1.05`** (`+0.26` macro) | `45%` safe shorten (`>=7/8`, response held) + `25%` `atleast` count rules |
| **Simple (`bfcl_simple`)** | `0.1190` | `~49% / -4.69` | `48% / -3.50` | `48% / -3.78` | **`46.5% / -2.40`** (`-0.29` macro) | `40%` `bt`-anchored guards + missing-call format fallback |
| **Several (`bfcl_multiple`)**| `0.0595` | `~52% / -0.69` | `51% / -0.19` | `50% / -0.56` | **`50.0% / +0.10`** (`+0.01` macro) | `25%` guards + format fallback on `1.7` pt parse-error loss |
| **No-Fit (`bfcl_irrel`)** | `0.0714` | `~54% / +6.15` | `52% / +5.73` | `51% / +5.62` | **`60.0% / +5.80`** (`+0.41` macro) | Removed from `--right-hold`; `62.5%` safe shorten (`>=7/8`) + `37.5%` compact |
| **Official Headline** | **`1.0000`** | **`53.7% / -1.27`** | **`52.7% / -0.99`** | **`51.5% / -0.84`** | **`53.1% [52.5, 53.7] / -0.58 [-0.80, -0.35]`** | **`+0.6%` to `+1.6%` more saving AND `+0.25` to `+0.40` better accuracy than `r2s3`** |

#### B. Risks and Fallbacks
1. **Risk 1 — `data/od3-base-think.json` missing ids for `s3m1` / `s3m2` top-up questions:**
   - **Mitigation (built into the code change):** `ref_t = max(tmed, 0.70 * bt) if bt else tmed` falls back to `tmed` whenever a question id is absent, and `od3prep` on Drive already merged `r2a–r2l` top-ups into `data/od3-base-think.json`.
2. **Risk 2 — Single-step `r2s3b` either gives back slightly too much saving (`< 52.2%`) or under-repairs accuracy (`~ -0.90`):**
   - **Fallback (Two-Adapter Merge on `r2s2-s80` Using Existing `r2s3-s30`):** Because `r2s3-s30` (`51.5% / -0.84`, `adapters/r2s3-s0/step-30`) is *already trained and full-tested* on `models/r2s2-s80-on-r2s1b-s180`, train a **20-step Safe-Shorten-Only adapter `r2s3s`** on the 2 safe types (`--rp-quota ifeval=200,bfcl_irrelevance=56`, `62.5%` shorten at `>=7/8` + `37.5%` compact be-right, `lr=8e-6`, 20 steps, reusing `rp-r2s3` rollouts in ~8 minutes) and merge **both adapters onto `r2s2-s80`**:
     `merge --model models/r2s2-s80-on-r2s1b-s180 --out models/r2s3-combo --adapters adapters/r2s3-s0/step-30:0.85 adapters/r2s3s-s0/step-20:0.80`
     Scaling `r2s3-s30` to `0.85` moves `51.5% / -0.84` to `51.83% / -0.90`, and adding `0.80` of the safe IFEval/No-Fit shortener adds `+1.2%` saving at neutral accuracy $\rightarrow$ **`~53.0%` saved / `-0.85` accuracy**.