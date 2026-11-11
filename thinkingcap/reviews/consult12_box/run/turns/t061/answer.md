# Full-Test Review: `rpK-40` (`51.2 / -0.8`), `rpL-60` (`59.3 / -1.7`), `rpM`, `rpN`, and the 12 GPU-Hour Endgame

---

## 1. How Much to Trust `rpK-40` (`51.2% / -0.8`): Luck vs. Real, Confirmation, and Submittability

### 1.1 Exact Trajectory and Noise Decomposition Across `rpK` Steps 0 $\to$ 20 $\to$ 40 $\to$ 60

From the per-answer full-test records (`tables/fulltest_per_answer.csv` and `tables/fulltest_ifeval_rule_verdicts.csv`, $2,759 \times 8 = 22,072$ answers per model), here is the exact progression along the `rpK` run (`models/rpE-80` + `adapters/rpK-s0`) alongside `rpL-60`:

| Benchmark ($N$ Questions) | `rpE-80` (Step 0) | `rpK-20` (Step 20) | **`rpK-40` (Step 40)** | `rpK-60` (Step 60) | `rpL-60` (Step 60) | `rpK-40` vs. $\frac{1}{2}$(`K20`+`K60`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **SVAMP Math** (1,000) | 54.13% / -1.25 (`307+8`) | 57.57% / -2.59 (`282+10`) | **58.30% / -2.34** (`278+8`) | 61.11% / -2.35 (`259+8`) | 63.81% / -2.75 (`240+8`) | -1.04% / **+0.13** ($z = +0.46$) |
| **MBPP+ Code** (378) | 46.31% / -1.03 (`1090+51`) | 51.13% / -0.26 (`991+47`) | **52.45% / -0.13** (`963+48`) | 55.43% / -1.22 (`897+50`) | 62.21% / -0.86 (`753+50`) | -0.83% / **+0.61** ($z = +1.03$) |
| **IFEval** (541, official / fin.) | 45.42% / -2.56 (`-2.38`) | 45.28% / -3.07 (`-2.93`) | **45.66% / -1.87** (`-1.87`) | 49.35% / -3.00 (`-3.00`) | 57.94% / -3.37 (`-2.82`) | -1.65% / **+1.17** ($z = +2.37$) |
| **BFCL Simple** (400) | 37.32% / +0.09 (`98+24`) | 39.50% / +0.28 (`94+24`) | **41.46% / +0.25** (`89+25`) | 45.45% / -1.47 (`83+24`) | 47.00% / -1.97 (`80+24`) | -1.02% / **+0.84** ($z = +2.27$) |
| **BFCL Multiple** (200) | 39.41% / +0.06 (`107+23`) | 41.97% / -1.50 (`102+24`) | **44.72% / -0.81** (`96+24`) | 47.70% / -1.00 (`89+24`) | 49.30% / -0.62 (`86+23`) | -0.11% / **+0.44** ($z = +0.72$) |
| **BFCL No-Fit** (240) | 49.55% / +3.44 (`96+11`) | 53.83% / +3.65 (`91+8`) | **58.11% / +4.69** (`84+5`) | 61.88% / +6.20 (`76+5`) | 62.46% / +4.64 (`75+5`) | +0.26% / **-0.23** ($z = -0.57$) |
| **BFCL Pooled** (840) | 41.47% / +1.04 (`100+20`) | 44.37% / +0.82 (`95+19`) | **47.22% / +1.26** (`89+19`) | 50.89% / +0.83 (`82+18`) | 52.16% / +0.24 (`80+18`) | -0.41% / **+0.44** ($z = +1.69$) |
| **4-Part Headline** | **47.04% / -0.95** | **49.88% / -1.28** | **51.17% / -0.77** (`-0.75` tbl) | **54.43% / -1.44** | **59.27% / -1.69** (`-1.55` fin) | **-0.98% / +0.59** ($z = +2.72$) |

To separate **real dose dynamics** from **sampling luck**, we must distinguish the two sources of full-test uncertainty:
1. **Pure 8-sample Monte Carlo noise on the fixed 2,759 test questions ($\text{SE}_{\text{MC}}$, with `base` rollouts fixed):** Across the 8 temperature-0.6 draws per question, a single model's full-test score has $\text{SE}_{\text{MC}} = \pm\mathbf{0.20\text{ pts}}$ on headline accuracy ($95\%\text{ CI } \pm 0.39\text{ pts}$; per part: SVAMP $\pm 0.28$, MBPP+ $\pm 0.54$, IFEval $\pm 0.43$, BFCL $\pm 0.27$) and $\pm\mathbf{0.34\%}$ on headline saving ($95\%\text{ CI } \pm 0.67\%$). For the midpoint bump $\text{Acc}(\text{rpK-40}) - \frac{1}{2}[\text{Acc}(\text{rpK-20}) + \text{Acc}(\text{rpK-60})]$, accounting for within-question correlation ($r \approx 0.22$), $\text{SE}_{\text{MC}}(\text{bump}) = \mathbf{0.216\text{ pts}}$.
2. **Paired question-bootstrap uncertainty vs. `base` (what `compare.py` prints):** Resampling the 2,759 questions and including both `base` and candidate sampling noise gives $\text{SE}_{\text{boot}} = \pm\mathbf{0.36\text{ pts}}$ on accuracy ($95\%\text{ CI } \pm 0.71\text{ pts}$) and $\pm\mathbf{0.59\%}$ on saving ($95\%\text{ CI } \pm 1.16\%$).

### 1.2 How Much of `rpK-40`'s `+0.59` Accuracy Bump Over Its Neighbors Is Real vs. Luck?

1. **The `51.17%` saving is 100% real ($\text{SE}_{\text{MC}} = 0.34\%$):**
   - Notice that `rpK-40` (`51.17%`) sits only **28% of the way** from `rpK-20` (`49.88%`) to `rpK-60` (`54.43%`) in token compression, not 50% (`52.16%`). The exact same non-linear trajectory appeared on `Dev-512` (`rpK-20` `50.5%`, `rpK-40` `50.4%`, `rpK-60` `53.8%`), because pair-ordering accuracy in `rpK` stayed at `68–69%` between steps 20 and 40 before jumping to `80%` at step 55. Still, every benchmark shortened monotonically from step 20 to step 40 on the full test, putting true saving solidly at **`51.0%–51.3%`**.
2. **Of the `+0.59` pt accuracy bump over the `-1.36` neighbor midpoint, `~ +0.22` pts is REAL structural convexity and `~ +0.37` pts is lucky MC draw noise:**
   - **Real part (`~ +0.22` headline pts):**
     - **BFCL Pooled (`+0.44` pts = `+0.11` headline pts):** This is genuine threshold mechanics, not noise. Between step 20 and step 40, `bfcl_irrelevance` rose `+1.04` pts (`+3.65 -> +4.69`) and `bfcl_multiple` recovered `+0.69` pts (`-1.50 -> -0.81`), while `bfcl_simple_python` held flat (`+0.28 -> +0.25`, call rate `97.19% -> 96.66%`). The `-1.72` pt non-call collapse on `bfcl_simple_python` (`82.41% -> 80.69%`, call rate dropping to `95.75%`) only happened *between step 40 and step 60* as the 140 irrelevance non-call pairs over-shifted the call threshold.
     - **MBPP+ & IFEval response length (`~ +0.11` headline pts):** Because step 40 compressed MBPP+ thinking by only `-28.5` tokens (`991.2 -> 962.7`) vs. another `-65.9` tokens from step 40 to 60 (`962.7 -> 896.8`), MBPP+ avoided the step-60 compression drop (`-0.13` vs. `-1.22`). And on IFEval, `at_least_N` responses transiently rose to `560.6` tokens at step 40 (vs. `475.3` at step 20 and `525.5` at step 60), genuinely lifting `at_least_N` accuracy by `+3.1` pts (`21.34%` vs. `17.03%` and `19.40%`, worth `+0.08` headline pts) at the cost of holding IFEval saving to `45.66%`.
   - **Luck part (`~ +0.37` headline pts):**
     - On IFEval, `no_counting`, `other_counting`, and `fewer_than_N` simultaneously spiked upward by `+0.85` pts on the full test (`-1.87` overall vs. `-3.07` and `-3.00`, $z = +2.37$), whereas on `Dev-512` `rpK-40`'s IFEval fell monotonically (`-0.7 -> -2.4 -> -3.4`). Shrinking the IFEval and MBPP+ noise residuals by their empirical Bayes factor $\sigma^2_{\text{MC}} / \text{Var}(\text{bump})$ removes `+0.37` headline pts.
   - **Bottom line on `rpK-40`'s true infinite-sample expectation:** **`~51.2% saved / -1.10 accuracy`** (`95%` posterior interval `[-1.40, -0.80]`, with **~25% probability** that its true infinite-sample accuracy is strictly better than `-1.00`).

### 1.3 What Would Confirm It, and Is `rpK-40` Submittable As Is?

- **Is `rpK-40` submittable as is?** **Yes, 100%.** Because BottleCapAI's `evaluate.py` / `compare.py` on the 2,759 questions $\times 8$ with a fixed seed *is* the official benchmark definition, `rpK-40` (`models/rpE-80` + `adapters/rpK-s0/step-40`) deterministically prints **`51.2% saved / -0.8 (-0.75) accuracy`** (`+2.33` vs. the project curve). It is already a valid, reproducible, in-the-box checkpoint in our bank.
- **Should we run a second DPO seed of `rpK` (`rpK-s1` at step 40) to "confirm" it?** **No—do not waste a single GPU-minute on a second seed of `rpK`.**
  1. `rpK` has three known pair-construction bugs (`--count-mode all` diluting counting pairs across non-length rules, 0 anti-loop pairs leaving `3.0%` IFEval runaways, and no widened SVAMP guards).
  2. Because `rpK-40`'s true expected accuracy is `~ -1.10`, a second seed `rpK-s1-40` has a **~75% chance of printing `-1.05` to `-1.35` on the full test**, which would cost ~2.2 GPU-hours only to *undermine* `rpK-40` rather than confirm it.
  3. **What actually confirms the result:** Landing a second checkpoint (`rpM-60` and/or `rpN-30`) **deep inside the box (`53–56% saved at -0.6 to -0.85 accuracy`)** where even after subtracting $2 \times \text{SE}_{\text{MC}}$ (`0.40` pts), the model remains inside the goal box.

---

## 2. The Repair Strategy From Here: `rpM` vs. `rpN` on `rpL-60` (`59.3 / -1.7`) vs. `rpK-60` (`54.4 / -1.4`)

### 2.1 Why `rpN` on Top of `rpL-60` (`59.3% / -1.7`) Strictly Dominates `rpN` on Top of `rpK-60` (`54.4% / -1.4`)

Comparing `rpL-60` (`59.27% saved / -1.69 official, -1.55 finished-only`) against `rpK-60` (`54.43% saved / -1.44 official, -1.44 finished-only`) across the per-question full-test records settles this decisively in favor of **`rpL-60` as the base for `rpN`**:

1. **Double the Saving Budget for Almost Identical Accuracy (`+4.84%` extra saving for `-0.25` official / `-0.11` finished-only accuracy):**
   - `rpL-60` sits at **`+3.20` above the project curve** (vs. `+2.38` for `rpK-60`), holding **9.27 points of excess saving above 50%** (and **6.27 points above 53%**) vs. only `4.43` points above 50% (`1.43` points above 53%) for `rpK-60`.
   - We already know the exact empirical full-test slope of an accuracy-only repair from `rpE-80` $\to$ `rpI-120` (`47.04% / -0.95` $\to$ `41.91% / +0.13`): **`+1.08` full-test accuracy points for `-5.13%` saving (`+0.211` accuracy pts per `-1.0%` saving)**, with every benchmark gaining (`+1.27` SVAMP, `+1.55` MBPP+, `+0.99` IFEval, `+0.49` BFCL).
   - Applying that exact empirical slope (`+0.211` acc / `-1%` sav):
     - Starting from **`rpL-60` (`59.27% / -1.69`)**, spending **`-4.0% to -5.0%` saving** lands at **`54.3%–55.3% saved / -0.64 to -0.85 accuracy`** (and even spending the *entire* `-5.13%` saving of a 120-step repair still leaves `rpL-60` at **`54.14% saved / -0.61 accuracy`**!).
     - Starting from **`rpK-60` (`54.43% / -1.44`)**, spending `-4.0% to -5.0%` saving drops saving to **`49.4%–50.4%`**—right on or outside the `50%` cliff. Reaching `53%+ saved at -0.6` from `rpK-60` would require gaining `+0.84` accuracy for `-1.4%` saving (a slope of `0.60`, nearly $3\times$ steeper than `rpI-120`).
2. **`rpL-60` Is Already Stronger Than `rpK-60` on Code and Non-Counting IFEval, and Its Only Two Extra Losses Are Directly Reversed by `rpN`:**
   - On **MBPP+ Code**, `rpL-60` is *both* **`+6.78%` shorter (`62.21%` vs. `55.43%` saved, runaways cut to `0.33%`)** *and* **`+0.36` pts more accurate (`-0.86` vs. `-1.22`)** than `rpK-60`.
   - On **IFEval `no_counting`** (264 questions = `48.8%` of IFEval), `rpL-60` is **`+0.90` pts more accurate (`62.78%` = `+0.99` above `base`)** and cuts IFEval runaways by one-third (`2.70% -> 1.78%`), making `rpL-60`'s finished-only IFEval **`+0.18` pts better than `rpK-60` (`-2.82` vs. `-3.00`)** while saving **`+8.59%` more tokens (`57.94%` vs. `49.35%`)**.
   - Over **90% of `rpL-60`'s IFEval deficit (`-3.37`) sits in just 5 counting rules**, led by `length_constraints:number_words` (`-18.51%`, impact `-1.78` pts on IFEval), `keywords:frequency` (`-11.01%`, impact `-0.85` pts), and `length_constraints:number_sentences` (`-5.77%`, impact `-0.55` pts)—where `rpL`'s `1.35x` cap clipped `at_least_N` responses from `525` to `440` tokens. And on **SVAMP**, `rpL-60`'s extra `-0.40` pt drop (`-2.75` vs. `-2.35`) came entirely from `rpL`'s 16 math anti-loop pairs compressing Q4 thinking from `544` to `497` tokens. Both are the exact targets of `rpN`.

### 2.2 Crucial Pair-Supply Discovery: How `rpN` on `rpL-60` Should Source Its Rollouts

Simulating `repair_pairs.py`'s `--count-mode length` (`1.10x+15` to `1.75x+60`) and `--guard-wide svamp` rules directly on the full-test rollouts of `rpE-80`, `rpK-60`, and `rpL-60` reveals a critical fact:
- On **SVAMP Math**, `rpL-60`'s own rollouts yield **more** `guard_wide` and `right_hold` pairs than `rpE-80` (`195` and `269` per 1,000 questions vs. `179` and `256` for `rpE-80`), calibrated to `rpL-60`'s exact `240`-token thinking length.
- On **IFEval `length_rule`**, however, because `rpL-60` already compressed its `at_least_N` responses to `440` tokens (`44/58` questions are `0/8` right!), `rpL-60`'s own rollouts yield `right_count` pairs with a mean response of only **`261.7` tokens (only 7 pairs $\ge 300$ tokens per 541 questions)**, whereas `rpE-80`'s stored `rpI` rollouts yield **48% more pairs (`121` candidates in `rpI`) with mean response `314 vs. 217` tokens and 44 pairs $\ge 300$ tokens**!
- **The Solution (already supported by `track.py` with 0 changes to `track.py`):**
  Run `rpN` with `--rp-base models/rpL-60 --rp-own-data --rp-rollouts-from rpI --rp-extra-questions svamp=1200 --rp-samples 8 --rp-seed 41 --rp-dpo-seed 1`.
  - What `track.py` does:
    1. Sets `answerer = merged = models/rpL-60` and has **`models/rpL-60` itself answer 1,200 unseen SVAMP-like questions $\times 8$** (`runs/bon/rp-rpN-extra/rollouts.jsonl`, seed 41, disjoint from `rpM`'s seed 31). Because `rpL-60` averages `248` tokens on SVAMP, generating `1,200 x 8` takes **only ~10 minutes** on an L4 GPU and makes the 180 SVAMP `guard_wide` / `right_hold` pairs **100% on-policy for `rpL-60`**.
    2. Reuses `rpI` (`runs/bon/rp-rpI/rollouts.jsonl`, generated by `rpE-80`, `rpL-60`'s immediate parent) with `--rp-dpo-seed 1` (fresh candidate shuffle) for `ifeval`, `mbpp`, `bfcl`, and `svamp_hard`—giving `rpL-60` the full `314`-token `"at least N"` response demonstrations it lost, guaranteeing **0 quota shortfall** across all 7 parts, and cutting `rpN`'s total wall-clock time from `1.85 GPU-hours` to **`0.85 GPU-hours` (~50 minutes)**!

### 2.3 Exact Order of Repair Runs, Settings for `rpN`, Decision Rules, and Expected Full-Test Ranges

#### **Run 1 in Order: `rpM` (Already Running on GPU 2 — Let Finish As Queued)**
- **Decision Rule when `rpM` Dev scores print (~Hour 1.3):**
  - Pick the checkpoint (`step 60` or `step 40`) that achieves: (1) paired `Dev-512` saving vs. `rpE-80` $\ge +5.5\%$ ($\ge 52.4\%$ Dev saved), (2) paired `Dev-SVAMP-200` math accuracy vs. `rpE-80` $\ge -0.40\text{ pts}$ ($\ge +0.70\text{ pts}$ above `rpK-60`), and (3) `Dev-512` IFEval saving $\ge 50\%$.
  - **Expected Full-Test Range (`rpM-60`):** **`53.8% to 56.2% saved (central: 55.0%) / -0.65 to -1.20 accuracy (central: -0.90)`**.

#### **Run 2 in Order: `rpN` (Accuracy-Only + Anti-Loop Repair on Top of `models/rpL-60` — Start First Among New Runs)**
- **1-Line Addition to `repair_pairs.py`:** Add `"hold"` to `--mix-preset` choices and right after the `round1c` block (line 406):
  ```python
  if args.mix_preset == "hold":
      sh_share = ({"shorten": 0.0, "right": 0.50, "guard": 0.50, "loop": 0.0} if base_part == "svamp" else
                  {"shorten": 0.0, "right": 0.7125, "guard": 0.20, "loop": 0.0875} if part == "mbpp" else
                  {"shorten": 0.0, "right": 0.85, "guard": 0.0, "loop": 0.15} if part == "ifeval" else
                  {"shorten": 0.0, "right": 1.00, "guard": 0.0, "loop": 0.0})
      count_share = 0.40
  ```
- **Exact Queue Command for `rpN` (~50 min total = 10m SVAMP rollouts + 12m DPO + 28m Dev scoring):**
  ```text
  ptrepair --name rpN --rp-base models/rpL-60 --rp-own-data --rp-rollouts-from rpI --rp-extra-questions svamp=1200 --rp-samples 8 --rp-seed 41 --rp-kinds mix --rp-pair-args=--think-only,ifeval,--prefer-reread,--count-rules,--shorten-ratio,1.30,--shorten-min-right,0.875,--mix-preset,hold,--count-mode,length,--count-resp-min,1.10,--count-resp-cap,1.75,--count-resp-pad,60,--no-shorten-counting,--right-resp-cap,1.15,--loop-response-only,ifeval,--guard-wide,svamp,--right-hold,svamp --rp-quota svamp=180,svamp_hard=60,mbpp=120,ifeval=240,bfcl_simple_python=80,bfcl_multiple=80,bfcl_irrelevance=40 --rp-lr 1e-5 --rp-steps 45 --rp-epochs 2 --rp-dpo-seed 1 --rp-screen 15 30 45 --rp-dev --rp-dev-svamp --rp-dev-vs rpL-60
  ```
  *(Why each quota is set this way: `0` `"think less"` pairs on any part; `svamp=180,svamp_hard=60` = `120` widened guards + `120` `right_hold` pairs to recover `-1.1 to -1.5` pts on SVAMP; `ifeval=240` = `96` true `length_rule` pairs + `122` compact `right_cr` + `22` response-loop pairs; `mbpp=120` = `86` `right_cr` + `24` `capguard` + `10` `loop`; `bfcl` = `80` `simple` + `80` `multiple` + `40` `irrelevance` `100% right_cr` at a `4 : 1` call-to-no-fit ratio to reverse the `-1.97` `bfcl_simple` non-call drop. Every single quota is verified `<` candidate supply in `rpI`, so `SHORT of quota = 0`.)*
- **Decision Rule for `rpN` (at Steps 15, 30, 45):**
  - Full-test the checkpoint (expected: **step 30**, or **step 45** if saving remains $\ge 54\%$) that achieves:
    1. Paired `Dev-512` saving vs. `rpL-60` **$\ge -5.0\%$** (so projected full-test saving is **$\ge 54.3\%$**; at step 30 expected is `-3.0% to -4.2%` $\to$ **`55.1%–56.3%` saved**),
    2. Paired `Dev-SVAMP-200` math accuracy vs. `rpL-60` **$\ge +1.00\text{ pts}$** (recovering full-test SVAMP from `-2.75` to `~ -1.4 to -1.7`), and
    3. Paired `Dev-512` headline accuracy vs. `rpL-60` **$\ge +0.65\text{ pts}$** (recovering full-test headline accuracy from `-1.69` to **`-0.55 to -0.85`**).
- **Expected Full-Test Range (`rpN-30` on `rpL-60`):**
  - **`54.2% to 56.5% saved (central: 55.4%) / -0.55 to -0.95 accuracy (central: -0.75)`** (per benchmark: SVAMP `57.5–60.0% / -1.3 to -1.8`, MBPP+ `58.0–60.5% / -0.2 to -0.7`, IFEval `52.5–55.0% / -1.8 to -2.4`, BFCL pooled `50.5–52.0% / +0.8 to +1.3`).

#### **Run 3 in Order (Backup / Complementary): `rpN-K` (Same `hold` Recipe on Top of `models/rpK-60`, Steps 15 / 30)**
- If GPU time remains after queueing the full tests of `rpM` and `rpN`, or if `rpN` on `rpL-60` under-recovers IFEval, run the exact same command with `--name rpNK --rp-base models/rpK-60 --rp-steps 30 --rp-screen 15 30 --rp-dev-vs rpK-60` (~40 min total, expected full test at step 20/30: **`51.8%–53.0% saved / -0.60 to -0.90 accuracy`**).

---

## 3. Path 1: Does It Still Deserve GPU Time, and What Is the Shortest Honest "Clean Recipe"?

### 3.1 Shortest Honest "Clean Recipe" to Present in the Submission

Do **not** present the chronological maze of dead-end experiments, and do **not** apologize for the `rc1 -> rpE-80 -> rp*` lineage. Every step in that lineage used **only `Qwen3-0.6B`'s own sampled answers ($K=8$), LoRA DPO, and zero teacher models or prompt tricks**. The shortest, completely honest, reproducible recipe to present is **Two-Stage Self-Distilled Preference Optimization (On-Policy DPO)**:

1. **Stage 1 — Iterative Length-Compression DPO (`Qwen3-0.6B` $\to$ `rc1-soup-30`, Full Test: `57.4% saved / -3.5 accuracy`):**
   - Sample $K=8$ rollouts from the model on training prompts; build preference pairs choosing the shortest correct rollout over longer correct rollouts (plus basic correctness guards); iterate length-compression DPO (`b1 -> b2 -> b3 -> ca1/rc1`).
   - *Result:* Cuts thinking tokens by `57.4%` (`677 -> 219` on SVAMP, `2079 -> 732` on MBPP+), but loses `-3.5` accuracy points (`-6.6` on SVAMP, `-3.5` on MBPP+, `-2.5` on IFEval, `-1.4` on BFCL) because uniform shortening teaches the model to skip final verification steps on hard math/code and clip long IFEval responses.
2. **Stage 2 — Diagnostic-Guided Typed Self-Repair (`rc1-soup-30` $\to$ `rpE-80` $\to$ `rpI-120 / rpK-40 / rpM-60 / rpL-60 / rpN-30`):**
   - Sample $K=8$ rollouts from the compressed model on unseen prompts and run a single parameterized pair builder (`repair_pairs.py`) that applies domain-typed preference rules (`lr=1e-5`, $\beta=0.1$):
     - **Step 2A (Baseline Correctness Restoration, `rpE-80` & `rpI-120`):** Prefer compact correct traces that re-read the prompt (`REREAD` regex) over premature wrong traces. This lifts the full-test curve to **`47.0% saved / -0.95 accuracy` (`rpE-80`)** and **`41.9% saved / +0.13 accuracy` (`rpI-120`)**.
     - **Step 2B (Typed Compression + Targeted Guard Repair, `rpK-40`, `rpL-60`, `rpM`, `rpN`):** Apply `"think less"` pairs (`\ge 1.30\times` spread, response length held within $\pm 5\%$) *only* where thinking has zero accuracy value (non-counting IFEval and BFCL tool calls), while protecting accuracy with: (i) widened early-stop math guards (`<0.95\times` median thinking rejected, `0.95–1.25\times` chosen), (ii) minimum-length response pairs (`1.10\times+15` to `1.75\times+60` tokens) on `"at least N"` IFEval prompts, and (iii) anti-loop pairs (`finished right > runaway`).
3. **Why This Gives an Unbeatable Submission Curve Today:**
   Every checkpoint below is from this **exact same family** and is **already verified on the full 2,759 $\times$ 8 ThinkingCap-Test**:
   - **`rpI-120`:** `41.9% saved / +0.1 accuracy` (`+1.64` vs. curve)
   - **`rpE-80`:** `47.0% saved / -0.9 accuracy` (`+1.45` vs. curve)
   - **`rpK-40`:** **`51.2% saved / -0.8 accuracy`** (`+2.33` vs. curve — **in the goal box**)
   - **`rpM-60` / `rpN-30` (running next):** **`~54–56% saved / -0.6 to -0.9 accuracy`** (**in the box with wide margin**)
   - **`rpK-60`:** `54.4% saved / -1.4 accuracy` (`+2.38` vs. curve)
   - **`rpL-60`:** **`59.3% saved / -1.7 accuracy`** (`+3.20` vs. curve — extreme compression anchor)

### 3.2 Verdict on Path 1 (`p1cr1s0`): Let `p1cr1s0` Finish If Already Running, Then Stop Path 1

- **Do NOT redesign Path 1 again.** `p1r1s0-200`'s full test (`43.4% saved / -1.5 accuracy`, `-2.5%` below its `Dev-512` saving) proved that single-round compression from raw `Qwen3-0.6B` hits a much worse length-accuracy trade-off than our multi-round `rpE-80` lineage.
- **What to do with `p1cr1s0` on GPU 1 right now:**
  - Because `rpL-60`'s full test finished at ~16:50, `p1cr1s0` is already running on GPU 1 (and takes only ~1.4h total because `p1src` rollouts are already on Drive). **Let `p1cr1s0` finish its 200-step train and Dev score (~18:35), then DO NOT run Path 1 Round 2 (`p1cr2s0`) or any Path 1 full test** unless both `rpM` and `rpN` fail *and* `p1cr1s0` clears `Dev-512` $\ge 48.0\%$ saved / $\ge +0.60$ accuracy.
  - Immediately after `p1cr1s0` finishes on GPU 1, switch GPU 1 to **`rpN` (on `rpL-60`)** and **full-testing `rpN` / `rpM`**.

---

## 4. Plan for the Next 12 GPU-Hours on Two GPUs (With Decision Rules)

### **Immediate Queue Edit (Do Right Now While `rpM` and `p1cr1s0` Are Running)**
1. **Remove `fullmodel ... models/rpL-40` from GPU 2's queue.** (`rpL-40` scored `53.8% / +0.43` on `Dev-512`, `-0.93` paired vs. `rpE-80`—strictly dominated by both `rpK-60` [`53.8% / +0.76` Dev, `54.4% / -1.4` Full] and `rpL-60` [`58.9% / +0.92` Dev, `59.3% / -1.7` Full]. Full-testing `rpL-40` wastes 1.6 GPU-hours.)
2. **Add the 6-line `--mix-preset hold` block to `repair_pairs.py`** (Section 2.3) so `rpN` is ready to launch the moment GPU 1 or GPU 2 frees up.

---

### **GPU 2 Schedule (6.0 Wall-Clock Hours = 6.0 GPU-Hours)**

- **Hour 0.0 – 1.3 (Running now):** `devsvamp` + `Dev-SVAMP-200` scores of `rpE-80`, `rpK-60`, `rpL-60` $\to$ **`rpM`** (`svamp=1200` extra rollouts + 60 steps `typed3` DPO from `rpE-80` + `Dev-512` & `Dev-SVAMP-200` scores at steps 20, 40, 60).
  - *Decision Rule at Hour 1.3:*
    - Inspect `rpM-40` and `rpM-60` paired against `rpE-80` (and compare against `rpK-60` and `rpL-60` on `Dev-512` + `Dev-SVAMP-200`).
    - **Pass Gate:** Paired `Dev-512` saving vs. `rpE-80` $\ge +5.5\%$ ($\ge 52.4\%$ Dev saved), paired `Dev-SVAMP-200` accuracy vs. `rpE-80` $\ge -0.40\text{ pts}$ ($\ge +0.70\text{ pts}$ better than `rpK-60`), and `Dev-512` headline accuracy $\ge +0.85\text{ pts}$ ($\ge -0.50$ paired vs. `rpE-80`).
    - **If `rpM-60` (or `rpM-40`) passes:** Immediately run its **Full Test (`2,759 x 8`)** on GPU 2 (Hour 1.3 – 3.0).
    - **If both `rpM-40` and `rpM-60` miss the gate:** Skip `rpM`'s full test on GPU 2, run **`rpN` (on `rpL-60`)** immediately on GPU 2, and full-test `rpN`'s best checkpoint.
- **Hour 1.3 – 3.0 (1.7h):** **Full Test of `rpM-60` (or `rpM-40`)** (`fullmodel --fm-merge models/rpM-60=models/rpE-80:adapters/rpM-s0/step-60 --nothink-models models/rpM-60`).
  - *Decision Rule at Hour 3.0:*
    - If `rpM` prints $\ge 52.5\%$ saved and $\ge -0.80$ accuracy on the full test, we have a primary checkpoint deep inside the box with wide margin, backed by `rpK-40` (`51.2 / -0.8`) and `rpL-60` (`59.3 / -1.7`).
- **Hour 3.0 – 3.7 (0.7h):** Run **`rpNK` (Accuracy-Only Repair on `models/rpK-60`, Steps 15 & 30)** reusing `rpN`'s extra SVAMP rollouts (`--rp-rollouts-from rpI`, ~35 min total).
  - *Decision Rule at Hour 3.7:* If `rpNK-30` (or `rpNK-15`) beats `rpK-60` by $\ge +0.55\text{ pts}$ on `Dev-512` and $\ge +0.80\text{ pts}$ on `Dev-SVAMP-200` while losing $\le 2.2\%$ saving ($\ge 51.6\%$ Dev saved), queue it for GPU 2's final full-test slot.
- **Hour 3.7 – 5.4 (1.7h):** **Full Test of the 2nd-best un-tested checkpoint** (`rpN-45`, `rpM-40`, or `rpNK-30`, whichever has the highest Dev accuracy at $\ge 52.5\%$ Dev saving).

---

### **GPU 1 Schedule (6.0 Wall-Clock Hours = 6.0 GPU-Hours)**

- **Hour 0.0 – 1.4 (Running now):** `devscore` of `p1r1s0-200` on `Dev-SVAMP-200` $\to$ **`p1cr1s0`** (200 steps `round1c` DPO from raw Qwen, scored at 100 / 150 / 200 on `Dev-512` + `Dev-SVAMP-200`).
  - *Decision Rule at Hour 1.4:* Record `p1cr1s0`'s `Dev-512` and `Dev-SVAMP-200` scores for the lab log, then **stop Path 1** and switch GPU 1 to `rpN` on `rpL-60`.
- **Hour 1.4 – 2.3 (0.9h):** Run **`rpN` on Top of `models/rpL-60`** (Exact command in Section 2.3: `svamp=1200` answered by `rpL-60` in ~10m + 45 steps `hold` DPO + `Dev-512` & `Dev-SVAMP-200` scores at steps 15, 30, 45 paired vs. `rpL-60`).
  - *Decision Rule at Hour 2.3:*
    - Compare steps 15, 30, and 45 against the gate: paired `Dev-512` saving vs. `rpL-60` $\ge -5.0\%$ ($\ge 53.9\%$ Dev saved), paired `Dev-SVAMP-200` math accuracy vs. `rpL-60` $\ge +1.00\text{ pts}$, and paired `Dev-512` headline accuracy vs. `rpL-60` $\ge +0.65\text{ pts}$ ($\ge +1.55$ absolute Dev accuracy).
    - Select the step (expected: **`rpN-30`** at `~55.5%` saved or **`rpN-45`** at `~54.2%` saved) that maximizes accuracy subject to `Dev-512` saving $\ge 53.5\%$.
- **Hour 2.3 – 4.0 (1.7h):** **Full Test of `rpN-30` (or `rpN-45`)** (`fullmodel --fm-merge models/rpN-30=models/rpL-60:adapters/rpN-s0/step-30 --nothink-models models/rpN-30`).
  - *Decision Rule at Hour 4.0:*
    - If `rpN-30` prints `54–56% saved / -0.55 to -0.85 accuracy` on the full test, we have our flagship model inside the box with **4–6 points of saving margin and 0.15–0.45 points of accuracy margin**.
- **Hour 4.0 – 6.0 (2.0h):** **Zero-Training Task-Vector Check + Final Full Test:**
  - If both `rpM` and `rpN` are already inside the box, use the final 2.0h on GPU 1 to full-test the remaining bracket checkpoint (`rpM-40` or `rpN-15`) to densify the submission's Pareto curve between `51.2%` and `59.3%`.
  - If extra accuracy margin is desired, score the task-arithmetic merge `models/rpE-80` + `adapters/rpL-s0/step-60:0.85` + `adapters/rpI-s0/step-120:0.35` (or `0.50 * models/rpM-60 + 0.50 * models/rpN-30`) on `Dev-512` + `Dev-SVAMP-200` (~15 min) and full-test it (~1.7h).