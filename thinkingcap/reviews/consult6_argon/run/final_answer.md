# Second Independent Review & Hardening of `PATH1_PLAN.md` (11 Nov 2026)

## 1. Response to `FACT_CHECK.md` & `CLAUDE_REVIEW.md`: What We Accept, Where We Nuance, and How Our Estimates Change

### 1.1 Point-by-Point Verdict on `FACT_CHECK.md`
We accept **9 of the 10 corrections** in `FACT_CHECK.md` without reservation, and add one log-sourced clarification on Point 5:

1. **Path #3 (`D4 lpd-budget`) built on `michal-bida`'s recipe (ACCEPT):** `L4131–4136` confirms `D4` ran on `bidax5q` (`--bida-exact` settings, bida's pair rules, and `math1`/`math2` top-ups) with DAST's per-question budget flag `--budget`, scoring `49.0% / −1.42 pp (+1.29 vs curve)` on Screen-720 vs `bidax5q-final`'s `46.7% / −0.59 pp (+1.71 vs curve)`. Dropping Path #3 under the team's no-competitor constraint is 100% right.
2. **`rpI` drift, `rpH` IFEval, and the counterfactual (ACCEPT):**
   - `rpI` did drift longer during its first 20 steps: from `rpE-80`'s `50.4%` saved on Screen-720 (`47.0%` full test) down to `42.9%` at step 100, and only held flat between step 100 (`42.9% / −0.2 pp`) and step 120 (`43.0% / +0.9 pp` on 720; `41.9% / +0.10 pp` on full test). As `consult5` (`L5225`) noted, fresh on-policy answers *attenuated* the length drift (`+14.9%` more output tokens in `rpI` vs `+35.7%` in `rpG`) rather than eliminating it on their own.
   - On Screen-720, `rpH` (`--count-rules` on old `rc1-soup-30` answers) scored `−2.5 pp` on IFEval at step 100 and `−2.1 pp` at step 120 (`L5179–5184`), so `--count-rules` alone did not fix IFEval on the 720 screen.
   - Adding `rpI-120`'s full-test IFEval (`−1.5 pp`) and `no-fit` (`+5.8 pp`) gains measured at `41.9%` saved onto `rpE-40full` (`50.6%` saved) is a cross-saving extrapolation, not an additive proof, and our subjective `55–65%` win probability had no frequentist basis.
3. **`bidax6p` (`fp32 + fp16 autocast`) moves along the curve, not +2.8 pp vertically (ACCEPT):** `L4662–4668` shows `bidax6p` vs `bidax4` traded `−11.4 pts` of saving (`57.1% → 45.7%` at step 100) for `+3.3 pp` of accuracy (`−5.1 → −1.8 pp`), moving from slightly below the curve to `+0.3..+1.1 pp` above it. Crucially for Path 1, this means **`fp32 + fp16 autocast` shortens ~8–11 percentage points LESS aggressively than `bf16` at equal steps and pairs**—a vital fact for sizing Round 1's shortening pressure.
4. **Section 1's `zbench`, `gold1`/`teach1c`, `stopV2`, and `4+4` split attributions (ACCEPT):**
   - `zbench` `0.508` was the position-only baseline (`L1545`); pooled test AUCs were `0.736` (MLP) and `0.717` (own confidence), and `zbench2`'s within-reply timing-only test AUC (`L1621`) was strong for own confidence on SVAMP (`0.963–0.977`) and MBPP+ (`0.833–0.850`) and for the MLP on BFCL (`0.806`). What failed in `zbench` was threshold transfer from practice math (which settles at 20–40% of thinking) to SVAMP (which settles late, `L1553`).
   - `gold1` and `teach1c` failed from unanchored token-level BCE (`L1515`), `stopV2` lengthened from long-chosen guard pairs + NLL (`L2200`), and the `4+4` half-split was `lot V2` (`L1973`), whereas `benchlot` used 20 forced re-answers split into two halves of 10 (`L2266`). The case against forced-cut DPO rests on `stopB3` (`−3.2 pp`, code `−8.4 pp`), `stopB5` (`−2.7 pp`), and `stopV3` (stalling at `10–25%` saved).
5. **Screen-720 vs Full-Test citations, `rc1-rl-40` full test, `rpG`, `rpA`, Commit `965a54c`, and IFEval floor details (ACCEPT 9/10; NUANCE on IFEval full-test diagnostic at `L5095–5100`):**
   - We accept that `l2s50e-final`, `push2-final`, `margin-final`, `D4`, `rpB`, `rpC`, and `lpt6mix` are Screen-720 runs; that `rc1-rl-40` had a full test (`55.4% / −2.6 pp`, `L3622`, making 11 full tests in the log); that `rpG` lost `17.7 pts` of saving (`50.4% → 32.7%`, not 25); that `rpA` ran at `lr = 5e-6` on `pt-mid`; that commit `965a54c` only requires agent traces; and that `51% / 77% / 94%` vs `41% / 57% / 82%` were the observed re-read rates in `rpE` produced by `--prefer-reread` (`repair_pairs.py` `L168–173`).
   - **Clarification on the IFEval numbers (`265`, `145–161`, `−0.65 pp`):** Those numbers come directly from `EXPERIMENT_LOG.md` at `L5095–5100` and `L5227`. While `L5042` (written earlier on 9 Nov from the 120-question 720 screen) noted that `rpE @ 80` shortened mean IFEval response tokens (`247 → 190`) and still scored `59.4%` on the 720 screen, the subsequent **full-test (`541 × 8`) rule-by-rule breakdown (`ir-full/ifeval_rules.json`, `L5095–5100`)** found: *"every shortened model (rc1, ca1, rpE-40/60/80) loses on COUNTING rules: length_constraints:number_words −11 to −13 pts, keywords:frequency −8 to −10... Too-short responses: original 265, rc1 399, ca1 364, rpE-80 379; responses on length-rule questions median 209 → 145–161 tokens. Wrong answers with thinking under half the original's right median: original 62, rc1 815, ca1 644, rpE-80 626."* And `consult5` (`L5227`) noted: *"IFEval deficit alone costs 0.65 pt of rpE-80's aggregate."* Thus both mechanisms are real on the full test: premature thinking stops (`626` vs `62`) and shortened responses on length/counting rules (`379` vs `265`).

---

### 1.2 Weighing `CLAUDE_REVIEW.md` Against the Log (With Additional Smoking-Gun Evidence)
We evaluated every claim in `CLAUDE_REVIEW.md` against `EXPERIMENT_LOG.md`, `all_runs.json`, and the four files in `code/`:

| Claim in `CLAUDE_REVIEW.md` | Our Verdict | Quantitative Evidence from the Log & Code |
|---|---|---|
| **1. Round 1 as written in `PATH1_PLAN.md` (`39.4%` shorten, `fp32`, 200 steps) will land at ~25–35% saved, not 45–50%.** | **STRONGLY AGREE (in fact, 18–33% saved).** | Beyond Claude's citations (`lpc-runa2`: 32.5–33.4% saved at `L3553–3617`; `D6 simright`: 34.1% → 28.1% saved at `L4566–4574`; `bidax6p`: `fp32` reduces saving by ~11 pts at `L4662`), there is an even closer smoking gun in the log: **`D12 lpd-chain1` (`L4767–4773`)**, which trained from raw Qwen in `fp32+fp16` on **40% shorten + 60% guard (`shortest right > shortest finished wrong`)** and scored **21.7% saved at step 100 → 18.4% at step 200 → 17.8% at final**! Worse, in `repair_pairs.py --kinds mix --mix-preset round1`, `46.9%` of pairs are `right_long`/`right_short` (half with `chosen > rejected`) and `13.7%` are `capguard` (`chosen > 2× rejected`), so **>42% of Round-1 pairs push length UP**. |
| **2. Round 2 cannot add saving AND accuracy simultaneously (`od1` went `50.4/−0.2 → 53.3/−1.8` on 720, falling from `+2.8` to `+1.75` vs curve).** | **STRONGLY AGREE.** | Every run in the log that tried to push saving *higher* in a later round (`od1`, `Track 2f` rounds 2–4, `rr1`) moved *down* or *along* the curve. Conversely, every run that reached `+1.3 to +1.64` above the full-test curve near 50% saved (`rpe-40full` `50.6% / −1.90`, `rpe-60full` `47.8% / −1.20`, `rpe-80full` `47.0% / −0.90`) started from a **55–58% saved model (`rc1-soup-30`: `57.4% / −3.50`)** and traded 4–10 points of excess saving for `+1.6 to +2.6 pp` of accuracy. **Round 1 MUST reach ~54–58% saved so Round 2 operates in the proven `rc1 → rpE` direction.** |
| **3. Round 1 math supply in `p1src` is broken (~400 mostly always-solved GSM8K questions cannot fill 800 math pairs).** | **STRONGLY AGREE (Fatal if unfixed).** | `L3944` shows `dp13` has 2,999 math questions: `metamath_easy` (1,865) + `metamath_sv` (473) + `gsm8k_easy` (420) + `gsm8k_matched` (241). `p1src` (`L446–455`) drops `metamath_*` and `math1`/`math2`, leaving **only 661 math questions** (`63%` trivial `gsm8k_easy`), and the `60%` hash filter cuts that to **~396 math questions** (~85% solved 10/10 by raw Qwen, so only ~60 mixed-outcome questions exist to fill 520 right+guard pairs!). Similarly, dropping `nofit4` (1,871 questions, `L3747`) leaves only 783 `no-fit` questions (`~443` after Dev-512 and the 60% split, with only ~90 mixed-outcome questions to fill 208 right+guard pairs). |
| **4. Drop `rpJ` completely and run Round 1 Seed 0 first.** | **PARTIALLY DISAGREE — Fix `--kinds mix` first; `rpJ` IS a valid 1.5-hour test of Round 2 once Round 1 is redesigned to hit ~55% saved.** | Claude argues `rpJ` tests the wrong regime because `rc1-soup-30` is at 57.4% saved while Round 1 would be at 30%. **Once we redesign Round 1 to hit 54–58% saved, `rc1-soup-30` (57.4% saved) IS in the exact regime of Round 1!** And since `rpH`'s rollouts already exist on Drive, running `rpJ` (after fixing the `--kinds mix` bug below) costs only ~1.5 GPU-hours and answers the crucial Round-2 question: *does adding 20–30% shorten anchors to `rpE`'s guard recipe hold saving at ~51–53% instead of drifting to 47.0%?* However, do **not** block Round 1 data prep on `rpJ`—run them in parallel across the two GPUs. |
| **5. Code bugs in `ptrepair` seeds/flags, `repair_pairs.py`, `devset` dedupe, and `dev_score.py` paired interval.** | **STRONGLY AGREE — Plus we found 3 additional code bugs in `track_ptrepair_odpo_dev.py` and `repair_pairs.py` not in `CLAUDE_REVIEW.md` (detailed in §3).** | In addition to Claude's findings, `ptrepair` in `track_ptrepair_odpo_dev.py` **silently ignores `--rp-exclude-ids`, `--rp-warmup`, `--rp-lr-min`, and `--rp-dpo-seed`**, and `repair_pairs.py --kinds mix` replaces `rpE`'s proven `--kinds guard` pair (`shortest right with reread > shortest finished wrong`) with `right_long`/`right_short` (`max/min <= 1.2`) where `take()`'s `rng.shuffle(pool)` destroys `--prefer-reread` priority across questions! |

### 1.3 Revised Grounded Estimates
- **Path 1 as currently coded in `PATH1_PLAN.md` & `code/`:** Round 1 will land at **20–33% saved** (with math starved of mixed-outcome pairs and `lr` stuck at constant `1.5e-5`), and Round 2 will land at **32–43% saved / −0.5 to −1.5 pp**—missing the `>50%` goal by ~8–15 points of saving.
- **Path 1 with the Redesigned Shorten-Dominant Round 1 (`65% shorten / 35% rpE-guard`) + Fixed Round 2 (`25% shorten / 60% rpE-guard / 15% capguard`) + Fixed Math/No-Fit Supply & Code Bugs:** Round 1 lands at **53–57% saved / −2.4 to −3.2 pp**, and Round 2 trades 2–4 points of excess saving to land at **49.5–52.5% saved / −0.7 to −1.3 pp** on the full test (a genuine, clean shot at crossing `>50.0% / >−1.00 pp`, sitting right at the `+1.5 to +1.7 pp vs curve` frontier achieved by `rpE-80full` and `rpI-120full`).

---

## 2. Point-by-Point Review of `PATH1_PLAN.md` (Keep / Change / Drop)

| Plan Element | Verdict | Exact Change Required & Quantitative Justification |
|---|---|---|
| **1. `Dev-512` Construction (`Step 0`)** | **CHANGE (Fix 8-gram dedupe + Confirm-512)** | **Keep** the 512-question size (`140 math, 100 code, 132 IFEval, 60 simple, 35 multiple, 45 no-fit`) and exclusion of `rpE/rpG/rpH/rpI/od1` training questions.<br>**Change 1 (Dedupe):** `devset()` (`L206–214`) reconstructs `used` by re-calling `own_repair_questions` with seeds `0, 1, 7` and excludes by `r["id"]`. Because `prepare_types gsm8k_std` and `pool-dp13` GSM8K questions use different ID prefixes for identical GSM8K texts, `p1src` and `p1r2` must exclude any question sharing an **8-gram** (normalized lowercase words) with `DEV_FILE` (and `Confirm-512`).<br>**Change 2 (Confirm-512 now):** Draw **both `Dev-512` (512 q) and `Confirm-512` (512 q)** inside `devset` right now so neither Round 1 nor Round 2 ever trains on `Confirm-512` questions! Currently `Confirm-512` is mentioned in `PATH1_PLAN.md` but never created or excluded in `track_ptrepair_odpo_dev.py`. |
| **2. `Dev-512` Calibration (`Step 0`)** | **CHANGE from 2 models to 5 models** | Scoring only `rc1-soup-30` (`57.4% / −3.50`) and `rpE-80` (`47.0% / −0.90`) calibrates only two endpoints on the same lineage. Add **`rpe-40` (`50.6% / −1.90`), `rpe-60` (`47.8% / −1.20`), and `ca1-soup-10` (`49.1% / −2.40`)** (or `rpI-120` `41.9% / +0.10`). All 5 models already exist on Drive (`~20 min` each on vLLM = `~1.5 GPU-h` total) and bracket the exact `47–51% saved / −0.9 to −2.4 pp` decision zone. |
| **3. `rpJ` Cheap Test & Decision Rule (`Step 1`)** | **CHANGE `repair_pairs.py --kinds mix` before running `rpJ`; CHANGE decision rule to paired `rpJ vs rpE-80` bootstrap** | **Why `rpJ` as coded is confounded:** (1) `1,280` pairs at batch `16` = `80` steps, so `--rp-steps 60` drops `25%` of the pairs; (2) `--kinds mix` currently uses `right_long`/`right_short` (`±20%` total tokens, shuffled without `REREAD` priority) instead of `rpE`'s `--kinds guard` (`shortest right with reread > shortest wrong`); (3) `bfcl_irrelevance=140` cannot fill from `rpH`'s 300 `no-fit` questions (~60 mixed-outcome).<br>**Exact Change:** Fix `--kinds mix` in `repair_pairs.py` (see §3.2 below so the `"right"` slot uses `rpE`'s compact `checked[0] or rights[0] > wrongs[0]` pair with `--ratio 1.25`, and set `rpJ` quotas to `svamp=400,mbpp=280,ifeval=280,bfcl_simple_python=110,bfcl_multiple=90,bfcl_irrelevance=120` = `1,280` pairs for `80` steps, screened at `40, 60, 80`).<br>**Decision Rule Fix:** Add `--vs runs/bon/dev512-rpE80/rollouts.jsonl` to `dev_score.py` (10 lines, §3.5) to compute the **direct paired bootstrap 95% CI of `(rpJ − rpE-80)`** on the same 512 questions (whose accuracy CI half-width is **`±0.8 to ±1.0 pp`**, vs `±1.6 pp` against `base`). `rpJ` passes if at step 40, 60, or 80 its curve-adjusted score on Dev-512 is `>= rpE-80 - 0.3 pp` while saving **`>= +2.5%` more tokens than `rpE-80`** (`i.e. >= 49.5%` full-test-equivalent saving). |
| **4. Round 1 Data & Question Split (`p1src`)** | **CHANGE (Critical: Fix Math & No-Fit Starvation)** | **Flaw:** Dropping `metamath_*` and `math1`/`math2` leaves only 661 math questions in `P1_ANSWERS` (420 `gsm8k_easy`), and taking 60% leaves **~396 math questions** (~60 mixed-outcome) for an 800-pair math quota! And dropping `nofit4` leaves only 783 `no-fit` questions (~443 after 60% split) for 320 pairs in Round 1 and 600 questions in Round 2.<br>**Two Clean Fixes (Pick A or B):**<br>- **Option A (Zero GPU generation time — Recommended if acceptable under your rule):** In `p1src`, `math1`, `math2`, and `nofit4` were generated by *your own* `prepare_types.py` and `newtop` scripts (`L3740–3752`: `gsm8k_std` in SVAMP wording and `xlam_irrelevance` / `glaive` / `toolace`). Only the `metamath_sv` rows inside `math1`/`math2` came from MetaMath, and Line 446 (`str(r["source"]).startswith("metamath")`) *already* drops them! If you do **not** ban the `gsm8k_matched` (`gsm8k_std`) rows of `math1`/`math2` and the `xlam`/`toolace` rows of `nofit4` in `p1src`, you immediately recover **~1,800 full-difficulty `gsm8k_std` questions** and **~1,871 `no-fit` questions** with 10 stored raw-Qwen answers at **0 GPU-hours**!<br>- **Option B (If `math1`/`math2`/`nofit4` files are strictly off-limits):** Have raw Qwen answer **2,200 `gsm8k_std` questions × 8** from `own_repair_questions()` (~3 GPU-hours on vLLM) to supply Round 1 math, and use **75% of `dp13 + nofit1 + nofit2 + tools1 + code1`** (after excluding `Dev-512` and `Confirm-512`) for Round 1, while generating new `gsm8k_std` / `ifeval_built` / held-out `kodcode` & `toolace` questions for Round 2. |
| **5. Round 1 Pair Mix & Hyperparameters (`p1r1s0/s1`)** | **CHANGE to Shorten-Dominant Mix (`65% shorten / 35% rpE-guard`) + Pre-Registered `>=53%` Dev Gate** | **Flaw:** As proven by `D12 lpd-chain1` (`21.7% → 17.8%` saved at `40%` shorten in `fp32`), `lpc-runa2` (`32.5%` saved), and `D6 simright` (`28.1%` saved), `--mix-preset round1` (`39.4%` shorten, `13.7%` `capguard` where `chosen > 2× rejected`, `46.9%` `right`) will plateau at **~20–33% saved**. Round 1 must land at **53–57% saved** (`D10 lpd-l2s50e` territory: `54.4%` at step 100, `53.5%` at final) so Round 2 can repair accuracy from `~55% → 51%` saved.<br>**Exact Change:**<br>- **Drop `capguard` and `right_long` from Round 1 completely** (`guard=0.0` in `--kinds mix`, and replace `right_long`/`right_short` with `rpE`'s compact `checked[0] or rights[0] > wrongs[0]` pair).<br>- **Set Round 1 shares (`--mix-preset round1`):**<br>  * **Math (`svamp`, 800) & `No-Fit` (`bfcl_irrelevance`, 320):** **60% Shorten** (solved `>= 6/10`, `rights[-1] >= 1.35 * rights[0]`, with `1.20x` backup on `9–10/10` always-solved as in `l2s50e`) + **40% Compact Re-Read Guard** (`shortest right with reread within 1.35x > shortest finished wrong`).<br>  * **Code (`mbpp`, 800), Simple (`280`), Multiple (`200`):** **70% Shorten** (solved `>= 5/10`, ratio `>= 1.35x`, `1.20x` backup on `9–10/10`) + **30% Compact Re-Read Guard**.<br>  * **IFEval (`ifeval`, 800):** **200 Counting-Rule Right pairs** (`--count-rules`: `response_len(c) >= response_len(w)` and `thinking_len(c) <= 1.2 * thinking_len(w)`) + **400 Non-Counting Shorten pairs** (thinking ratio `>= 1.35x`, response length within ±15%) + **200 Non-Counting Compact Guard pairs** (thinking-only).<br>- **Training dose:** `3,200 pairs` (`200 steps × 16`), **1 epoch (`--rp-epochs 1`)**, `LoRA r16 alpha 32`, `lr = 2.0e-5` cosine to `5e-6` (`10` warmup steps), `fp32+fp16`.<br>- **Pre-registered Round 1 Gate:** Score `p1r1` at steps `100, 150, 200` on `Dev-512`. Advance to Round 2 **only** if calibrated Dev-512 saving is **`>= 53.0%`** (and accuracy `>= −3.5 pp`). Pick the checkpoint (`100, 150, or 200`) closest to **55% calibrated saving**. |
| **6. Round 2 Design & Reference (`p1r2s0/s1`)** | **CHANGE Question Count (`5,350 → 3,200`) & Mix (`25% shorten / 60% compact-right / 15% capguard`); KEEP Reference = Round 1** | **Keep `reference = Round 1 model`** (see Open Question 3 proof in §4.3).<br>**Change 1 (Question generation budget):** Generating `5,350 × 8 = 42,800` answers *per seed* (`85,600` answers across 2 seeds = `~22 GPU-hours`) for `1,280` pairs is a `3.0%` yield. Worse, since `p1r2s0` and `p1r2s1` both pass `--rp-seed 21`, both seeds answer the **exact same 5,350 questions**! Reduce `--rp-questions` to **`3,200` questions** (`svamp=1000,mbpp=600,ifeval=750,bfcl_simple_python=300,bfcl_multiple=220,bfcl_irrelevance=330` = `25,600` answers per seed, saving ~9 GPU-hours) AND pass `--rp-seed 21` for Seed 0 and `--rp-seed 22` for Seed 1 so the two seeds train on diverse question draws!<br>**Change 2 (Round 2 Mix):** On a `54–57%` saved Round-1 model, use **25% Shorten Anchor** (`>= 6/8` right, ratio `>= 1.30x` with `1.20x` backup) + **60% Compact Re-Read / `--count-rules` Right** (`checked[0] or rights[0] > wrongs[0]` where `thinking_len(c) <= 1.25 * thinking_len(w)`) + **15% Capped Anti-Early-Stop Guard** (`thinking_len(c) <= 1.15 * tmed` vs `thinking_len(w) < 0.50 * tmed`). Train `80 steps` (`1 epoch`), `lr = 1e-5` constant, score 2-seed soup at `40, 60, 80` on `Dev-512`. |
| **7. Checkpoint Choice & Soup** | **CHANGE to Cross-Adapter Merge onto Raw Qwen** | Because Round 1 (`A_r1`) and Round 2 (`A_r2`) are both `LoRA r16 alpha 32` linear adapters on `Qwen/Qwen3-0.6B`, merging `0.5 * (A_r1_s0 + A_r2_s0) + 0.5 * (A_r1_s1 + A_r2_s1)` directly onto `Qwen/Qwen3-0.6B` via `merge.py --model Qwen/Qwen3-0.6B --adapters adapters/p1r1s0-s0/step-200:0.5 adapters/p1r2s0-s0/step-K:0.5 adapters/p1r1s1-s1/step-200:0.5 adapters/p1r2s1-s1/step-K:0.5` is mathematically exact (`W_0 + 0.5*(dW_r1_s0 + dW_r2_s0 + dW_r1_s1 + dW_r2_s1)`) and avoids saving/restoring 1.2 GB full-model checkpoints for every screen step! |

---

## 3. Code Audit: Bugs & Mismatches in `code/` (With Exact Line Numbers & Fixes)

Even after the two 11 Nov fixes in `repair_pairs.py` (`eb712bc`: credit-based interleaving of kinds at `L254–268` and `tmed` / `early[-1]` in `capguard` at `L140–145`), **six concrete bugs remain** across `track_ptrepair_odpo_dev.py`, `repair_pairs.py`, `queued_lines.txt`, and `dev_score.py`. **The first four will silently corrupt your training runs if not fixed before execution.**

### Bug 1 (CRITICAL — `track_ptrepair_odpo_dev.py` `L69–70` & `L108–125`): `ptrepair` Silently Ignores `--rp-exclude-ids`, `--rp-warmup`, `--rp-lr-min`, and `--rp-dpo-seed`!
Look at the queued commands in `queued_lines.txt`:
```
ptrepair --name p1r1s1 ... --rp-lr 1.5e-5 --rp-lr-min 5e-6 --rp-warmup 10 --rp-steps 200 --rp-epochs 2 --rp-dpo-seed 1 ...
ptrepair --name p1r2s1 ... --rp-exclude-dev --rp-exclude-ids data/p1-round1-ids.json ... --rp-dpo-seed 1 ...
```
Now look at what `ptrepair(args)` in `track_ptrepair_odpo_dev.py` actually executes:
1. **Line 70 (`--rp-exclude-ids` ignored!):**
   ```py
   for r in own_repair_questions({p: n * n_rounds for p, n in want.items()}, seed=args.rp_seed,
                                 exclude=dev_ids() if getattr(args, "rp_exclude_dev", False) else None):
   ```
   The helper `excluded_ids(args)` was written at `L396–405` to union `dev_ids()` and `args.rp_exclude_ids` (`data/p1-round1-ids.json`), **but `ptrepair` at Line 70 never calls `excluded_ids(args)`!** As a result, Round 2 (`p1r2s0`, `p1r2s1`) completely ignores `data/p1-round1-ids.json` and trains on Round 1's questions!
   - **Fix (`L70`):** Replace `exclude=dev_ids() if getattr(args, "rp_exclude_dev", False) else None` with `exclude=excluded_ids(args)`.
2. **Line 123 (`--rp-warmup` and `--rp-lr-min` ignored!):**
   ```py
   "--lr-schedule", "cosine", "--warmup", 2, "--lr-min", args.rp_lr, "--max-steps", steps,
   ```
   `--warmup` is hardcoded to `2` (ignoring `--rp-warmup 10`) and `--lr-min` is hardcoded to `args.rp_lr` (ignoring `--rp-lr-min 5e-6`, so Round 1 trains at a **constant `1.5e-5`** instead of cosine decay to `5e-6`!).
   - **Fix (`L123`):** Use `"--warmup", getattr(args, "rp_warmup", 2), "--lr-min", getattr(args, "rp_lr_min", None) or args.rp_lr`.
3. **Lines 108, 112, 118, 125 (`--rp-dpo-seed` ignored everywhere — Seeds 0 and 1 are identical!):**
   - `L108`: `repair_pairs.py` is called without `--seed`, so `p1r1s0` and `p1r1s1` build the **exact same pairs in the exact same order**.
   - `L112`: `adapter = Path("adapters") / f"{name}-s0"` hardcodes `-s0`.
   - `L118`: `sft.py --init-only` is called without `--seed`, producing the **exact same LoRA initialization** for Seed 0 and Seed 1.
   - `L125`: `dpo.py` is called with hardcoded `"--seed", 0`, ignoring `args.rp_dpo_seed`!
   - **Fix (`L108–125`):** Pass `dpo_seed = getattr(args, "rp_dpo_seed", 0)` to `repair_pairs.py --seed {dpo_seed}`, `sft.py --init-only --seed {dpo_seed}`, and `dpo.py --seed {dpo_seed}`.

---

### Bug 2 (CRITICAL — `repair_pairs.py` `L133–138` & `L210–213`): `--kinds mix` Replaces `rpE`'s Proven Compact Correctness Pair with Unfiltered `right_long`/`right_short` and Loses `--prefer-reread`!
Look at how `--kinds mix` builds the `"right"` pairs (`55%` of Round 2 / `rpJ` and `35–50%` of Round 1) at `L134–138` and `L211–213`:
```py
close = [(r, w) for r in rights for w in wrongs if max(size(r), size(w)) <= args.ratio * max(1, min(size(r), size(w)))]
if decode:  # prefer a chosen answer that re-checks the question
    close.sort(key=lambda rw: not REREAD.search(decode(rw[0])))
for r, w in close[:4]:
    cand[part]["right_long" if size(r) > size(w) else "right_short"].append((qid, r, w))
```
And then `take()` at `L184` runs `rng.shuffle(pool)` across the entire `cand[part]["right_long"]` and `cand[part]["right_short"]` lists!
**Three things go wrong here:**
1. **`--prefer-reread` is wiped out across questions:** Sorting `close` before `close[:4]` only orders pairs *within a single question*. If Question A has 0 re-read answers (so it contributes 4 non-reread pairs to `cand`) and Question B has 2 re-read answers, `rng.shuffle(pool)` in `take()` picks Question A's non-reread pairs with equal (or higher!) probability.
2. **Long rambling right answers are selected:** Unlike `rpE` (`L169`: `thinking_len(r) <= 1.5 * thinking_len(rights[0])`), `close` pairs *any* right answer `r` with *any* wrong answer `w` within `1.2×` length. If a question has a 300-token right answer, a 1,800-token right answer, and a 1,700-token wrong answer, `close` puts `(1800-tok right, 1700-tok wrong)` into `right_long`, teaching the model to write 1,800-token traces!
3. **Half of the pairs (`right_long`) force `chosen_tokens > rejected_tokens`, and strict `--ratio 1.2` starves `no-fit` and tools:** As the docstring of `repair_pairs.py` (`L11`) notes, `right_long`/`right_short` is Lightning McQwen3's rule that was dropped on 9 Nov because `rpE`'s rule (`shortest right with reread > shortest finished wrong`) was much stronger. Moreover, in `rpJ`, `rpH` only has 300 `no-fit` questions (~60 mixed-outcome); requiring `max/min <= 1.2` leaves `bfcl_irrelevance=140` short of quota.
- **Exact Fix in `repair_pairs.py` (`L133–138` & `L210–213`):**
  In `--kinds mix`, build `cand[part]["right"]` using `rpE`'s compact re-read rule bounded by `--ratio`:
  - Restrict candidate right answers to compact ones: `compact_r = [r for r in rights if thinking_len(r) <= 1.5 * thinking_len(rights[0])]`.
  - Pair `(r, w)` where `thinking_len(r) <= args.ratio * max(1, thinking_len(w))` (so chosen thinking is never >`1.20x–1.25x` rejected thinking, and compact right answers beat longer or equal-length finished wrong answers without forcing `50%` of pairs to have `chosen > rejected`!).
  - Partition candidates into `right_reread` (where `REREAD.search(decode(r))` is true) and `right_plain`, and in `take()`, draw from `right_reread` **first** before falling back to `right_plain`!
  - Also increment `rereads` inside `if args.kinds == "mix":` (`L128–145`) so `summary.txt` logs the counting-rule and re-read statistics (currently `continue` at `L146` skips all `rereads` logging!).

---

### Bug 3 (HIGH — `queued_lines.txt` & `repair_pairs.py` `L147–153` vs `L124–126`): Missing `1.20x–1.25x` Shorten Backup in `--kinds mix` and Mismatched Steps vs Quotas in `rpJ`
1. In `repair_pairs.py`, `--kinds guard,shorten` (`L151–152`) has a backup supply `shorten_soft` (`spread >= 1.25`) if `shorten` pairs (`>= args.shorten_ratio`) run short, **but `--kinds mix` (`L124–126`) has NO backup supply!** On `rc1-soup-30` (whose answers are already compressed to ~200 tokens on math and ~90 tokens on tools) or on `10/10` easy questions, requiring `>= 1.3x–1.4x` with no `1.20x` backup causes `shorten` shortfalls that silently convert into `right` pairs (`L209: n_r = n - len(sh) - len(g)`), silently reducing shortening pressure!
   - **Fix (`L124–126` & `L208`):** Add `shorten_soft` (`spread >= 1.20` on questions solved `>= 0.85`) as a backup for `sh` in `--kinds mix`, and fail loudly (`assert len(sh) >= 0.85 * n_sh`) if shorten pairs still fall >15% short of target.
2. In `queued_lines.txt`, `rpJ` has quotas summing to **`1,280` pairs (`80` steps of `16`)** with `--rp-epochs 2`, but sets `--rp-steps 60 --rp-screen 20 40 60` (`960` pairs seen, leaving `320` pairs unseen). Set `--rp-steps 80 --rp-screen 20 40 60 80` and `--rp-epochs 1` (and likewise `--rp-epochs 1` on `p1r1` and `p1r2` once quotas are met so no pair is trained on twice).

---

### Bug 4 (HIGH — `repair_pairs.py` `L203–205` & `L210–212`): Round-1 IFEval Counting-Rule Quota Is Not Capped or Guaranteed at 50%
In `repair_pairs.py` `L204` (`--mix-preset round1`), IFEval shares are `{"shorten": 0.275, "right": 0.725, "guard": 0.0}` (`220` shorten, `580` right). Then at `L210`:
```py
r = take(part, "right", cand[part]["right_cr"], n_r)
```
In `p1src`, only ~130–160 IFEval questions produce a valid `right_cr` pair (because `right_cr` gets at most 1 pair per counting-rule question where `response_len(r) >= response_len(w)` and `thinking_len(r) <= 1.2 * thinking_len(w)`). So `r` gets only ~140 `right_cr` pairs (instead of 400), and the remaining **~440 pairs** are filled from non-counting `right_long`/`right_short` (`L211–212`)! Worse, for `--think-only ifeval`, `size` is `thinking_len`, so `right_long` (`50%` of those 440 pairs = `220` pairs) actively trains IFEval to **think longer**!
- **Fix:** Allow up to `2` `right_cr` pairs per counting-rule question (matching the `right_q[qid] < 2` cap for non-counting questions), and fill any remaining IFEval `right` quota with `rpE`-style compact guard pairs (`shortest thinking right > shortest thinking wrong` where `response_len(c) >= 0.9 * response_len(w)`), never `right_long`!

---

### Bug 5 (MEDIUM — `dev_score.py`): Missing Paired Candidate-vs-Reference Bootstrap (`--vs`)
In `PATH1_PLAN.md` Step 1, `rpJ` is judged against `rpE-80` on `Dev-512`: *"more accurate than rpE-80 by more than the paired interval's half-width"*. However, `dev_score.py` only takes `--base` (`dev512-base`) and `--cand` (`dev512-rpJ-40`), so it outputs the 95% CI of `(rpJ − base)` (half-width **`±1.5 to ±1.7 pp`**), NOT the paired 95% CI of `(rpJ − rpE-80)`! Because `rpJ` and `rpE-80` share the same base (`rc1-soup-30`) and the same 512 questions, their per-question errors are highly correlated (`r ~ 0.85`), so the true paired 95% CI half-width of `(rpJ − rpE-80)` is **`±0.8 to ±1.0 pp`**.
- **Exact Fix in `dev_score.py`:** Add an optional `--vs` argument (`--vs runs/bon/dev512-rpE80/rollouts.jsonl`). On each bootstrap draw `draw`, compute `(s_cand, a_cand, _) = headline(base, cand, draw)` and `(s_vs, a_vs, _) = headline(base, vs, draw)`, and bootstrap `(s_cand - s_vs, a_cand - a_vs)` directly.

---

### Bug 6 (MEDIUM — `track_ptrepair_odpo_dev.py` `L183–395`): Duplicate Function Definitions in `track_ptrepair_odpo_dev.py`
In `track_ptrepair_odpo_dev.py`, `DEV_FILE`, `DEV_SOURCES`, `dev_ids()`, `devset()`, `dev_generate()`, `dev_score_model()`, `devscore()`, and `ladder()` are pasted twice (`L183–305` and `L307–395`), with truncated headers at `L305` (`def mixscreen(args):`) and `L395` (`def ladder(args):`). Verify that `track.py` itself does not contain these duplicate blocks.

---

## 4. Direct Answers to the 5 Open Questions in `PATH1_PLAN.md`

### Question 1: Is the relative decision rule for `rpJ` (vs `rpE-80` on Dev-512) sound, and what margin should count?
- **The relative comparison on Dev-512 is sound in principle** because evaluating both `rpJ` and `rpE-80` on the exact same 512 questions cancels out dataset-difficulty offsets (`Dev-512` vs Full Test), **provided two conditions are met:**
  1. **Compare along the trade-off slope, not just raw dominance:** Between 45% and 55% saved, the empirical trade-off slope in `rpE` (`40 → 60 → 80` on the full test: `50.6%/−1.90 → 47.8%/−1.20 → 47.0%/−0.90`) is **~0.28 pp of accuracy per 1.0% of tokens saved** (and the quadratic yardstick slope at 50% saved is `~0.19 pp` per `1.0%` saved). Therefore, define the **frontier-adjusted gain** of `rpJ` over `rpE-80` on Dev-512 as:
     $$\Delta_{\text{frontier}} = (\text{acc}_{\text{rpJ}} - \text{acc}_{\text{rpE80}}) + 0.22 \times (\text{saved}_{\text{rpJ}} - \text{saved}_{\text{rpE80}})$$
  2. **Exact Margin That Should Count:** Using the direct paired bootstrap (`--vs dev512-rpE80`), the 95% CI half-width on `Dev-512 × 8` is `±0.85 pp` for accuracy and `±0.9%` for saving (`1-sided 90% CI` ≈ `+0.55 pp`). `rpJ` **passes** if at step 40, 60, or 80:
     - **Saving Condition:** $\text{saved}_{\text{rpJ}} - \text{saved}_{\text{rpE80}} \ge +2.5\%$ on Dev-512 (proving the shorten anchors arrested `rpE-80`'s drift and kept full-test-equivalent saving at `≥ 49.5–51.0%`), **AND**
     - **Frontier Condition:** $\Delta_{\text{frontier}} \ge +0.40\text{ pp}$ (meaning if `rpJ` saves `+3.5%` more than `rpE-80`, its Dev-512 accuracy is within `−0.37 pp` of `rpE-80`, putting it at `~50.5% saved / −1.25 pp` or better before Round 1's cleaner starting point).

### Question 2: Are 3,200 pairs and `lr 1.5e-5` cosine right for Round 1 from the original Qwen?
- **No — with the current `--mix-preset round1` (39.4% shorten), 3,200 pairs at `lr 1.5e-5` in `fp32` will land at 20–33% saved, not 50–55% saved.**
- **Why:** In your log, every single-round run from raw Qwen that had `<50%` shorten pairs or included equal-length `right>wrong` pairs (`D12 lpd-chain1`: `40%` shorten in `fp32` → **17.8–21.7% saved**; `D6 simright`: **28.1% saved**; `lpc-runa2`: **32.5–33.4% saved**) stalled far below 40% saving. Meanwhile, the runs from raw Qwen that reached `50–55%` saving in `fp32` (`D10 lpd-l2s50e`: **54.4% at step 100, 53.5% at final**; `bidax6p`: **48.9% at step 200**) used **50–67% shorten pairs** plus `shortest right > wrong` guard pairs (where chosen is also shorter than rejected on 70%+ of questions) over `200–330` steps at `lr = 2.0e-5`.
- **Exact Prescription for Round 1:** Keep `3,200 pairs` (`200 steps × 16`, `1 epoch`), raise `--rp-lr` to **`2.0e-5`** with cosine decay to **`5e-6`** (`--rp-warmup 10`, with Bug 1 fixed so `dpo.py` actually receives `--lr-min 5e-6`!), and change `--mix-preset round1` to **65% Shorten + 35% Compact Re-Read Guard** (`0% capguard`, `0% right_long`; see §5 below).

### Question 3: Should Round 2 keep the Round-1 model as reference, or the original Qwen?
- **Keep the Round-1 model (`models/p1r1s0` / `models/p1r1s1`) as Round 2's reference — do NOT use the original Qwen as Round 2's reference.**
- **Quantitative Proof from DPO Mechanics & Your Log:**
  In DPO, the implicit reward is $\beta \log(\pi_\theta(y|x) / \pi_{\text{ref}}(y|x))$. After Round 1 compresses thinking by ~55%, a compact 250-token right answer $y_c$ has much higher probability under $\pi_{\text{R1}}$ than under $\pi_{\text{orig}}$, whereas a 250-token wrong answer $y_w$ also has compressed length. If you set $\pi_{\text{ref}} = \pi_{\text{orig}}$ in Round 2 while training on equal-length `(250-token right, 250-token wrong)` pairs, the KL barrier $\beta D_{\text{KL}}(\pi_\theta \,\|\, \pi_{\text{orig}})$ pulls $\pi_\theta$ **back toward $\pi_{\text{orig}}$'s 680-token length distribution** on every gradient step! In your log, setting `reference = Round 1 (or current stage)` in `rpE`, `rpI`, and `rc1` (2 hops total from raw Qwen: `raw → R1 → R2`) achieved your highest full-test scores (`+1.45` and `+1.64 vs curve`), whereas only 4–7 hop chains drifted excessively. Two hops with `reference = Round 1` and `25%` shorten anchors in Round 2 is optimal.

### Question 4: How much can a `Dev-512` ranking be trusted given source differences (`GSM8K` vs `SVAMP`, `KodCode` vs `MBPP+`), and is calibrating on `rc1-soup-30` / `rpE-80` enough?
- **How much to trust `Dev-512`:**
  - *Within-family ranking (e.g., step 40 vs 60 vs 80, or `rpJ` vs `rpE-80`):* **High reliability (`r > 0.85`)**, provided you use paired bootstrap differences (`--vs`).
  - *Absolute level (`saved %` and `acc pp` vs Full Test):* **Systematic domain offset of ~1.5–3.0 pts**, for three reasons documented in your log: (1) `gsm8k_std` is multi-step calculation (where raw Qwen thinks ~880 tokens and answers settle at 20–40% of thinking, `L1553`), whereas SVAMP is 1–2 step arithmetic with wording traps (where raw Qwen thinks 677 tokens and answers settle at 60–80%); (2) `Dev-512` fixes IFEval counting-rule prompts at `66/132 = 50.0%` (whereas the full test has ~32% counting-rule prompts, exaggerating counting-rule gains/losses by ~1.5× on IFEval); and (3) `Dev-512` includes the hard `nofit1`/`nofit2`/`tools1`/`code1` top-ups.
- **Calibration Fix:** Calibrating on just 2 models (`rc1-soup-30` and `rpE-80`) is **not enough** because a 2-point line has zero degrees of freedom to detect nonlinearity or domain-specific tilt. Score **5 models with known full tests** on `Dev-512`: **`rc1-soup-30` (`57.4/−3.50`), `rpe-40` (`50.6/−1.90`), `rpe-60` (`47.8/−1.20`), `rpe-80` (`47.0/−0.90`), and `ca1-soup-10` (`49.1/−2.40`)**. Compute both the **4-benchmark headline offset** and the **per-part offsets** ($\Delta_p = \text{Full}_p - \text{Dev512}_p$ for $p \in \{\text{svamp, mbpp, ifeval, bfcl}\})$, and select checkpoints using the **per-part calibrated full-test predictor**.

### Question 5: Any risk in reusing the original's 10 stored answers (`P1_ANSWERS`)? Is the 60 / 40 question split right?
- **Reusing raw Qwen's 10 stored answers (`temp 0.6, top_p 0.95, top_k 20`) is 100% safe technically** (raw Qwen's weights have not changed, and 10 answers per question gives a wider `rights[0]` vs `rights[-1]` spread than 8 answers), **EXCEPT for two data-supply risks that make the blind `60 / 40` hash split across `P1_ANSWERS` fail:**
  1. **Math & `No-Fit` starvation in `P1_ANSWERS` if `math1`/`math2`/`nofit4` are dropped:** As shown in §2.4, without `math1`/`math2`, `P1_ANSWERS` has only **661 non-MetaMath math questions** (420 of them `gsm8k_easy`), and without `nofit4`, only **783 `no-fit` questions**. Taking 60% leaves ~396 math questions and ~443 `no-fit` questions—far too few to build 800 math pairs and 320 `no-fit` pairs in Round 1 AND leave 600 `no-fit` questions for Round 2!
  2. **You do NOT need a 60 / 40 split on Math or IFEval if Round 2 draws from `own_repair_questions()`:** `own_repair_questions()` has **6,598 `gsm8k_std` math questions** (`2,597` mixed / `4,001` always-right, `L5078`) and **2,998 IFEval questions** (`L5187`).
  - **Exact Fix for the Split:**
    - If `math1`/`math2` (excluding `metamath_sv` rows) and `nofit4` are allowed as own-generated top-ups (`Option A` in §2.4), keep a **70 / 30 split** (`int(md5) % 10 < 7` for Round 1, leaving 30% + all unused `gsm8k_std` and `ifeval_built` for Round 2).
    - If `math1`/`math2`/`nofit4` are strictly excluded (`Option B`), **do NOT split `dp13` by 60/40 on Math**: have raw Qwen answer **2,200 `gsm8k_std` questions × 8** (~3 GPU-h) for Round 1 Math, use **75%** of `dp13 + nofit1 + nofit2 + tools1 + code1` for Round 1 Code/IFEval/Tools (with Round 1 `no-fit` quota set to `240` instead of `320`), and use the remaining **25%** + fresh `gsm8k_std` (2,000 q) + `ifeval_built` (1,100 q) for Round 2!

---

## 5. How to Make Path 1 as Strong and Safe as Possible (Final Recommended Recipe & First Experiment)

### 5.1 Plain Warning on the One Flaw That Would Sink Path 1
Let us be blunt: **if you run Round 1 with `--mix-preset round1` as currently written (`39.4%` shorten, `46.9%` `right_long`/`right_short`, `13.7%` `capguard`) and without fixing `p1src`'s math supply, Path 1 will fail.** Your own log (`D12 lpd-chain1` at `17.8–21.7%` saved, `D6 simright` at `28.1%` saved, `lpc-runa2` at `32.5%` saved) proves that a 39%-shorten / 61%-correctness mix in `fp32` from raw Qwen lands at **~20–33% saved**. From a 30%-saved Round-1 model, Round 2 cannot reach >50% saved without cranking up shorten pairs and crashing accuracy below −2.0 pp (as `od1` just demonstrated).

**The cure is simple and already proven by `rc1 → rpE` and `D10 lpd-l2s50e`:**
- **Round 1 must be a Shorten-Dominant + Compact-Guard pass (`65% shorten / 35% rpE-guard`, `0% capguard`, `0% right_long`)** that pushes saving to **53–57% saved** on `Dev-512` (calibrated) while using `--prefer-reread` and `--count-rules` to avoid the worst math/IFEval traps up front.
- **Round 2 then samples ×8 on-policy from that 53–57% saved Round-1 model** and applies the **25% Shorten Anchor + 60% Compact Re-Read/`--count-rules` Guard + 15% Capped Anti-Early-Stop Guard** mix to trade ~3–4 points of excess saving (`55% → 51–52%`) for `+1.5 to +2.2 pp` of accuracy, landing at **49.5–52.5% saved / −0.7 to −1.3 pp** on the full test.

### 5.2 Step-by-Step Execution Checklist (Exact Code Changes & Commands)

1. **Apply the 5 Mandatory Code Patches Before Running Step 1 / Step 2 (~20 minutes of coding):**
   - **`track.py` (`ptrepair`):**
     - Line 70: `exclude=excluded_ids(args)` (so `--rp-exclude-ids` and `Confirm-512` are respected).
     - Line 108, 118, 125: pass `--seed args.rp_dpo_seed` to `repair_pairs.py`, `sft.py --init-only`, and `dpo.py`; name the adapter `f"{name}-s{args.rp_dpo_seed}"`.
     - Line 123: pass `"--warmup", args.rp_warmup, "--lr-min", args.rp_lr_min or args.rp_lr` to `dpo.py`.
   - **`track.py` (`devset` & `p1src`):**
     - In `devset`: draw both `data/dev512.jsonl` (512 q) and `data/confirm512.jsonl` (512 q), and in `dev_ids()` return the union of their IDs **plus** any pool question sharing an 8-gram with either file.
     - In `p1src`: either keep the non-`metamath` (`gsm8k_matched`) rows of `math1`/`math2` and `nofit4` (`Option A`, 0 GPU-h), or generate 2,200 `gsm8k_std` questions × 8 with raw Qwen (`Option B`, ~3 GPU-h) so Round 1 has ≥1,500 full-difficulty GSM8K questions and ≥1,000 `no-fit` questions.
   - **`repair_pairs.py` (`--kinds mix`):**
     - Replace `right_long`/`right_short` in `--kinds mix` with compact re-read right>wrong pairs: `compact_r = [r for r in rights if thinking_len(r) <= 1.5 * thinking_len(rights[0])]`, paired with `w in wrongs` where `thinking_len(r) <= args.ratio * max(1, thinking_len(w))`, drawing pairs where `REREAD.search(decode(r))` is true **first** before non-reread pairs.
     - Add `shorten_soft` (`spread >= 1.20` on `>= 8/10` right) as backup for `shorten` in `--kinds mix`.
     - Change `--mix-preset round1` to:
       ```py
       sh_share = ({"shorten": 0.60, "right": 0.40, "guard": 0.0} if part in ("svamp", "bfcl_irrelevance") else
                   {"shorten": 0.50, "right": 0.50, "guard": 0.0} if part == "ifeval" else
                   {"shorten": 0.70, "right": 0.30, "guard": 0.0})
       ```
       and change default `--mix-shares` (for Round 2) to `"shorten=0.25,right=0.60,guard=0.15"`.
   - **`dev_score.py`:** Add `--vs` to output the paired bootstrap 95% CI of `(cand − vs)` directly.

2. **Step 0 (`Dev-512` + 5-Model Calibration, ~3 GPU-hours):**
   - Run `devset`, then `devscore` on **`rc1-soup-30`, `rpe-40`, `rpe-60`, `rpe-80`, and `ca1-soup-10`**. Fit the per-part and headline `Dev-512 → Full-Test` calibration offsets.

3. **Step 1 + Step 2A in Parallel Across the Two GPUs (~3.5 GPU-hours):**
   - **GPU 2 (`rpJ` Fixed Round-2 Mix Test on `rc1-soup-30`, ~2.5 GPU-h):**
     ```
     ptrepair --name rpJ --rp-base models/rc1-soup-30 --rp-own-data --rp-rollouts-from rpH --rp-kinds mix --rp-pair-args=--think-only,ifeval,--prefer-reread,--count-rules,--ratio,1.25,--shorten-ratio,1.3,--shorten-min-right,0.75 --rp-quota svamp=400,mbpp=280,ifeval=280,bfcl_simple_python=110,bfcl_multiple=90,bfcl_irrelevance=120 --rp-lr 1e-5 --rp-steps 80 --rp-epochs 1 --rp-screen 40 60 80 --rp-dev
     ```
     Score `rpJ-{40,60,80}` with `dev_score.py --vs runs/bon/dev512-rpE80/rollouts.jsonl`.
     - **Tune Round 2's shorten share from `rpJ`:** If `rpJ-60` saves `< 50.5%` (calibrated), raise Round 2's shorten share from `0.25` to `0.35`; if `rpJ-60` saves `> 54.0%` at `< −1.5 pp`, lower Round 2's shorten share to `0.15`.
   - **GPU 1 (`p1r1s0` Shorten-Dominant Round 1 Seed 0, ~3.5 GPU-h):**
     ```
     ptrepair --name p1r1s0 --rp-base Qwen/Qwen3-0.6B --rp-own-data --rp-rollouts-from p1src --rp-kinds mix --rp-pair-args=--think-only,ifeval,--prefer-reread,--count-rules,--ratio,1.25,--shorten-ratio,1.35,--shorten-min-right,0.60,--mix-preset,round1 --rp-quota svamp=800,mbpp=800,ifeval=800,bfcl_simple_python=280,bfcl_multiple=200,bfcl_irrelevance=320 --rp-lr 2.0e-5 --rp-lr-min 5e-6 --rp-warmup 10 --rp-steps 200 --rp-epochs 1 --rp-dpo-seed 0 --rp-screen 100 150 200 --rp-dev
     ```
     - **Hard Gate on `p1r1s0`:** Check `p1r1s0` at steps `100, 150, 200` on calibrated `Dev-512`. It **must reach `≥ 53.0%` calibrated saving** (expected: `53.5–56.5% saved / −2.2 to −3.0 pp`). If saving is `< 52.0%` at step 200, run 60 additional steps at `lr = 1.5e-5` before launching Round 2.

4. **Step 2B (Round 1 Seed 1 + Round 2 On-Policy Seeds 0 & 1, ~28 GPU-hours across 2 GPUs):**
   - Train `p1r1s1` (`--rp-dpo-seed 1`), merge `models/p1r1s0` and `models/p1r1s1` at the chosen Round-1 step (`150` or `200`).
   - Run `p1r2s0` on GPU 2 (`--rp-seed 21 --rp-dpo-seed 0`) and `p1r2s1` on GPU 1 (`--rp-seed 22 --rp-dpo-seed 1`) on **3,200 unseen own questions × 8** (`svamp=1000,mbpp=600,ifeval=750,bfcl_simple_python=300,bfcl_multiple=220,bfcl_irrelevance=330`), `80 steps`, `lr = 1e-5` constant.
   - Score the 2-seed weight-averaged soups at Round-2 steps `40, 60, 80` on `Dev-512 × 8`, lock the single checkpoint with highest calibrated accuracy at `≥ 50.8%` calibrated saving, verify once on `Confirm-512 × 8`, and run the official **Full Test (2,759 × 8)** on the 2-seed soup and one earlier curve checkpoint.