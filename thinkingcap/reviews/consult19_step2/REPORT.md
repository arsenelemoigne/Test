# Consult 19 Report: Recipe 2 (Step 2 & Step 3) and Path A vs. Path B Decisions

---

## 1. SHOULD WE RE-RUN EITHER PATH, AND CHANGE WHAT?

### Path A: Recipe 2 Step 1 (`r2s1b`, 180 offline DPO steps from the original)
- **Verdict:** **KEEP AND CONTINUE TO STEP 2. Do NOT re-run Step 1.**
- **Empirical Evidence:**
  - `r2s1b-s180` on the full test ($2,759 \times 8$) achieves **$+50.3\%$ tokens saved / $-1.7$ accuracy points**, while `r2s1b-s160` achieves **$+49.0\%$ tokens saved / $-1.7$ accuracy points**.
  - Per-type breakdown on the full test:
    - **`r2s1b-s180`**: Math `62% / -4.5`, Code `54% / -3.1`, IFEval `38% / +1.0`, Simple Tools `42% / -3.0`, Several Tools `45% / +0.9`, No-Fit `45% / +3.5`.
    - **`r2s1b-s160`**: Math `60% / -4.0`, Code `51% / -3.2`, IFEval `40% / 0.0`, Simple Tools `41% / -2.6`, Several Tools `43% / +0.6`, No-Fit `46% / +5.2`.
  - Why this is an ideal Step 1 checkpoint and must not be re-run:
    1. **Clean Separation of Jobs:** Following the user's principle (*"never shorten and protect the SAME type in one step: overshoot, then a separate type-specific repair"*), `r2s1b` did its exact job on Math (`62%` saved) and Code (`54%` saved)—it pushed their compression past the `rpM-60` target (`57%` math / `58%` code) without guard pairs (`guard=0`), creating a clean, high-saving state where the accuracy deficit is concentrated in premature truncation on Math (`-4.5`), Code (`-3.1`), and 1-function tool formatting (`-3.0`).
    2. **Untapped Compression Headroom on IFEval and Tools:** Conversely, IFEval (`38%` saved, `+1.0` accuracy) and Tool Use (`42%–45%` saved, with `+0.9` on Several and `+3.5` to `+5.2` on No-Fit) are **under-shortened** and **accuracy-positive** on the full test because offline Step 1 pairs from the original model had narrow relative length gaps on IFEval (median chosen/rejected `359 / 477` tokens) and Tools (`124 / 190`, `139 / 210`, `156 / 237`).
    3. **Why Re-running Step 1 Would Waste Time:** Re-running Step 1 with lower math/code shortening or added guards would violate the single-job-per-step rule, cost ~3 GPU-hours of training + evaluation, and merely trade away the `62% / 54%` math/code savings that Step 2/Step 3 can repair on-policy.

### Path B: The `od3` Online DPO Arena (6 rounds, fixed reference, per-type floors, no guards)
- **Verdict:** **DROP Path B (`od3`) for the current objective** (do not spend GPU time re-running it before finishing Recipe 2). If ever revisited later as a standalone research run, it requires three structural changes detailed below.
- **Empirical Evidence:**
  - `od3` ran 5 on-policy rounds (80 steps in `r1`, 30 steps in `r2–r5`, `lr=1e-5`, `beta=0.20`, fixed original reference, no guards, fixed per-type token floors `svamp=0.38`, `mbpp=0.42`, `ifeval=0.45`, `tools=0.48`).
  - Full tests on $2,759 \times 8$:
    - **`od3-r3`**: **`58.4%` saved / `-4.01` pts** (Math `74% / -8.9`, Code `64% / -4.2`, IFEval `43% / -0.9`, Simple `42% / -1.3`, Several `43% / -0.6`, No-Fit `45% / -4.6`).
    - **`od3-r4`**: **`60.5%` saved / `-4.79` pts** (Math `76% / -9.4`, Code `67% / -4.8`, IFEval `44% / -2.2`, Simple `45% / -2.0`, Several `46% / -1.1`, No-Fit `48% / -5.4`).
- **Why `od3` Failed and Why Not to Re-Run Now:**
  1. **Cross-Type Length Bleed Defeated Static Per-Type Floors:** Even though `svamp` had a `0.38` floor on its own shorten pairs, shortening every type simultaneously across 4–5 on-policy rounds compounded the global "close `</think>` early" LoRA direction. Math compressed to `74%–76%` saved (`~90` thinking tokens on SVAMP), causing a catastrophic `-8.9` to `-9.4` point collapse on math and `-4.6` to `-5.4` on no-fit.
  2. **The 780-Question Monitor Under-Read Accuracy Loss:** At `od3-r4`, the 780-question monitor read `59.3% / -2.50`, missing the full test (`-4.79`) by `2.29` points and failing to trigger the `-3.5` stop rule.
  3. **Concrete Changes Required *Only If* Revisited in a Future Study:** (a) Replace type-named floors with the **question-property continuous floor** $L_{\text{floor}}(q) = \max(60 + 0.16 L_{\text{orig}}(q),\, 0.34 L_{\text{orig}}(q))$ where $L_{\text{orig}}(q)$ is the original model's median right thinking length on question $q$; (b) cap the arena at **2 rounds** (`r1` = 120 steps at `lr=1.5e-5`, `r2` = 40 steps at `lr=1e-5`, `beta=0.10`) followed by **1 dedicated repair round** (`0%` shorten, `100%` length-matched `be-right` + `guard` on questions where solve rate dropped); and (c) lower `--shorten-min-right` on no-fit from `0.80` to `0.60` so hard abstention questions are not skipped.

---

## 2. SHOULD WE CHANGE THE PLAN FOR STEP 2 (AND STEP 3)?

**YES.** We must update Step 2's pair mix from `consult16`'s original `r2s2` preset to reflect two decisive facts established by the Step 1 full tests (`r2s1b-s160` and `r2s1b-s180`) and the user's preferences:
1. **Fact 1 (Per-Type Compression Asymmetry after Step 1):** In `r2s1b-s180`, Math (`62%` saved / `-4.5` acc) and Code (`54%` saved / `-3.1` acc) have **already overshot** the target compression (`>50%`), whereas IFEval (`38%` saved / `+1.0` acc), Several Tools (`45%` saved / `+0.9` acc), and No-Fit (`45%` saved / `+3.5` acc) are **under-shortened** (`38%–45%`) with positive accuracy margins.
2. **Fact 2 (User Rule: Never Shorten and Protect the Same Type in One Step, Avoid Over-Protecting):**
   - `consult16`'s preset `r2s2` shortens Math (`40%`) and Code (`45%`) *while simultaneously* applying `25%` and `15%` widened guards to Math and Code in the same step. That violates the user's rule and would push Math from `62%` to `>70%` saved (repeating `od3`'s `-9` pt math collapse).
   - Conversely, `rpM`'s `typed3` puts `50%` widened guards on Math (`50% be-right / 50% guard`) and `33%` guards on Code (`50% be-right / 17% guard / 33% loop`). In `rpM-60`, putting heavy guards on Math during Step 2 still allowed cross-type transfer from IFEval/tool shortening to keep Math at `57%` saved, *and* if we follow Step 2 with Step 3 repair, putting `50%` widened guards in Step 2 AND again in Step 3 risks **over-protecting** (dragging total token savings below `50%`, as seen in `rpI-120` at `41.9%` saved).

### Exact Recommendation for Step 2

#### 1. Starting Checkpoint: **`r2s1b-s180`** (over `r2s1b-s160`)
- Both full tests are now in:
  - **`r2s1b-s160` Full Test:** **`+49.0%` saved / `-1.7` acc** (Math `60 / -4.0`, Code `51 / -3.2`, IFEval `40 / 0.0`, Simple `41 / -2.6`, Several `43 / +0.6`, No-Fit `46 / +5.2`).
  - **`r2s1b-s180` Full Test:** **`+50.3%` saved / `-1.7` acc** (Math `62 / -4.5`, Code `54 / -3.1`, IFEval `38 / +1.0`, Simple `42 / -3.0`, Several `45 / +0.9`, No-Fit `45 / +3.5`).
- **Why `r2s1b-s180` wins:** Both checkpoints have the **exact same full-test accuracy (`-1.7` pts)**, but `s180` gives **`+1.3%` more overall token savings (`50.3%` vs `49.0%`)**, `+3%` more Code savings (`54%` vs `51%` at `-3.1` vs `-3.2` acc), `+2%` more Math savings (`62%` vs `60%`), and `+1.0` higher IFEval accuracy (`+1.0` vs `0.0`). That extra `+1.3%` savings buffer is critical so that Step 3's accuracy repair can spend `1.5–2.5%` token savings without ever dipping below the `50.0%` floor.

#### 2. Step 2 Job & Pair Mix: **Clean Type-Split Step 2 (Shorten IFEval & Tools Only; Pure Neutral `be-right` + `loop` on Math & Code, Zero Guards)**
- To strictly obey *"never shorten and protect the SAME type in one step"* and *"avoid over-protecting"*:
  - **IFEval and Tool Calls (`simple`, `multiple`, `irrelevance`)** are the ONLY types shortened in Step 2 (moving IFEval from `38% -> ~50%` saved and Tools from `42–45% -> ~52–56%` saved). As proven by `rpK`, `rpL`, and `rpM` (`NEW_RESULTS_13nov_morning.md` §5), shortening IFEval and Tools in Step 2 holds or slightly increases Math and Code compression by cross-type transfer without needing any `shorten` pairs on Math or Code.
  - **Math (`svamp`) and Code (`mbpp`)** receive **0% `shorten` and 0% `guard`** in Step 2—only length-neutral `be-right` (`right_long` : `right_short` = 50:50 within ratio `1.15`) and `loop` pairs. We save widened `guard` pairs (which actively lengthen thinking) for Step 3!
- **Exact Quotas & Shares per Type (`1,600` pairs total = 100 batches of 16 on `data/r2-step2-questions.jsonl` answered $\times 8$ by `r2s1b-s180`, fixed-mix rule `--no-trim-mix`):**
  - **`svamp` (320 pairs):** `0%` shorten (`0`), **`90%` `be-right`** (`288` pairs: length-matched `1.15`, half `right_long` / half `right_short`, preferring re-check of the question), **`10%` `loop`** (`32` pairs: finished right > runaway), `0%` guard.
  - **`mbpp` (320 pairs):** `0%` shorten (`0`), **`80%` `be-right`** (`256` pairs: length-matched `1.15`, 50:50 `right_long`/`right_short`), **`20%` `loop`** (`64` pairs: response-loop priority), `0%` guard. *(Note: `r2-step2-questions.jsonl` has `387` code questions—13 short of 400—which easily supplies 320 `be-right` + `loop` pairs at up to 2–3 pairs/question).*
  - **`ifeval` (480 pairs):**
    - **`20%` counting-rule `be-right`** (`96` pairs: `--count-mode length --count-share 0.20 --count-resp-cap 0.0`, true word/sentence/paragraph/postscript length constraints only, no response cap).
    - Of the remaining `384` non-counting pairs: **`62.5%` `shorten`** (`240` pairs = `50%` of 480, `--think-only` with response held within `--resp-median 0.95 --resp-max 1.05`), **`25%` `be-right`** (`96` pairs = `20%` of 480), **`12.5%` response-only `loop`** (`48` pairs = `10%` of 480, `--loop-resp-only ifeval`).
  - **Tool Calls (`480` pairs total: `bfcl_simple=180`, `bfcl_multiple=140`, `bfcl_irrelevance=160`):**
    - **`bfcl_simple` (180 pairs):** **`50%` `shorten`** (`90` pairs, `--think-only` with response held) / **`50%` `be-right`** (`90` pairs). *(Reduced from `70/30` to `50/50` because `simple` is already at `-3.0` acc on `s180` full test and needs stronger schema/argument verification).*
    - **`bfcl_multiple` (140 pairs):** **`55%` `shorten`** (`77` pairs) / **`45%` `be-right`** (`63` pairs).
    - **`bfcl_irrelevance` (160 pairs):** **`60%` `shorten`** (`96` pairs) / **`40%` `be-right`** (`64` pairs).

#### 3. Exact Thresholds & Pair Flags for Step 2
- **Shorten thresholds:** `--shorten-ratio 1.35` (backup `1.25`), `--shorten-min-right 0.75` ($\ge 6/8$ right rollouts on `s180`'s 8 answers; for `bfcl_irrelevance`, allow $\ge 5/8$ = `0.625` if needed so hard no-fit prompts are not starved), `--think-only svamp+mbpp+ifeval+bfcl_simple+bfcl_multiple+bfcl_irrelevance`, `--resp-median 0.95 --resp-max 1.05`.
- **Length-matched `be-right` ratio:** `--ratio 1.15` (strictly 50% `right_long` where chosen is longer than rejected, 50% `right_short` where chosen is shorter, eliminating hidden shortening bias in `be-right`).
- **Fixed-mix rule:** `--no-trim-mix` enabled (never refill a type's shorten/loop/guard quota with another kind; if any part is short of quota, answer extra held-out questions via `r2_extra_rows`).

#### 4. Training Hyperparameters & Checkpoints for Step 2
- **Init & Reference:** Continue LoRA from `adapters/r2s1b/step-180` (`--init-from adapters/r2s1b/step-180`), reference model = `adapters/r2s1b/step-180` (standard step-to-step DPO anchor, matching `rpE-80 -> rpM-60`).
- **Optimizer & Schedule:** **Constant learning rate `--lr 1e-5`** (`--warmup 5`, `--lr-schedule constant`), **`--beta 0.10`**, LoRA rank 16 (`alpha=32`), batch size 16 (`grad_accum=8`, microbatch 2), `--nll 0.0 --keep-order --bida-precision --dtype bfloat16`.
- **Total Steps & Dev Checkpoints:** Train **80 steps** (1,280 pair updates; 1 pass over 1,600 pairs = 100 steps max, stopping at 80). Save and score every **20 steps (`step-20, step-40, step-60, step-80`)** on **Dev-512 + Dev-SVAMP**.

#### 5. Calibrated Dev Gate for Step 2 (Accounting for Dev's Measured Over-Reading)
- **Measured Dev Bias on `r2s1b-s180`:**
  - **Overall Accuracy:** Dev-512 read `+1.25` while Full Test read `-1.70` (**Dev-512 is `+2.95` pts too kind** on overall accuracy, driven by `bfcl_simple` reading `+8.3` on Dev vs `-3.0` on Test—an `11.3` pt gap—and `ifeval`/`multiple` reading `+2.8` to `+3.0` pts kinder on Dev).
  - **Overall Token Saving:** Dev-512 read `51.1%` vs Full Test `50.3%` (**Dev-512 reads `+0.8%` higher** on overall saving, because Dev-SVAMP/GSM8K math reads `59%` vs Test `62%`, while Dev IFEval reads `41%` vs Test `38%`).
  - **Math (Dev-SVAMP vs Full Test SVAMP):** Dev-SVAMP read `54.7% saved / -1.12 acc` while Full Test SVAMP read `62.0% saved / -4.50 acc` (**Dev-SVAMP under-reads math saving by `-7.3%` and over-reads math accuracy by `+3.38` pts**).
- **Calibrated Step 2 Checkpoint Selection Gate:**
  - Define the **Bias-Corrected Full-Test Estimate** from Dev scores relative to `r2s1b-s180`'s Dev baseline (`Dev512 = 51.1% / +1.25`, `DevSVAMP = 54.7% / -1.12`):
    $$\widehat{\text{Saved}}_{\text{test}} = \text{Saved}_{\text{Dev512}} - 0.8\%, \qquad \widehat{\Delta\text{Acc}}_{\text{test}} = \Delta\text{Acc}_{\text{Dev512}} - 2.95$$
    $$\widehat{\text{MathAcc}}_{\text{test}} = \frac{1}{2}\left(\Delta\text{Acc}_{\text{DevSVAMP}} - 3.38 + \Delta\text{Acc}_{\text{Dev512-math}} - 1.60\right)$$
  - **Gate to select the winning Step 2 checkpoint $s^* \in \{20, 40, 60, 80\}$:**
    1. **Saving Target (leaving headroom for Step 3 repair):** $\text{Saved}_{\text{Dev512}} \in [54.5\%,\, 57.5\%]$ (corresponding to $\widehat{\text{Saved}}_{\text{test}} \in [53.7\%,\, 56.7\%]$, i.e., $+3.4\%$ to $+6.4\%$ more savings than `s180`, with IFEval Dev saving $\ge 49\%$ and Tools Dev saving $\ge 50\%$).
    2. **Accuracy Floor:** $\Delta\text{Acc}_{\text{Dev512}} \ge +0.80$ (so $\widehat{\Delta\text{Acc}}_{\text{test}} \ge -2.15$, i.e., losing at most $0.45$ pts during the Step 2 shortening push), with $\Delta\text{Acc}_{\text{DevSVAMP}} \ge -1.50$ and Dev-512 math accuracy no worse than `-3.5`.
    3. Pick the checkpoint in `{40, 60, 80}` that maximizes $\widehat{\Delta\text{Acc}}_{\text{test}} + 0.25\,\widehat{\text{Saved}}_{\text{test}}$ subject to $\text{Saved}_{\text{Dev512}} \ge 54.5\%$, run its Full Test to verify, and immediately feed its adapter into Step 3.

### Step 3's Repair Focus (One Paragraph)
Step 3 is a pure, single-job **on-policy accuracy repair and anti-truncation step** (`0%` shorten across all 6 types) trained for **40–60 DPO steps** at constant `lr = 1e-5`, `beta = 0.10` from the selected Step 2 checkpoint on its own $\times 8$ answers to the disjoint `data/r2-step3-questions.jsonl` split (`1,280` pairs total), specifically targeting the three full-test accuracy deficits revealed by `r2s1b-s180` (**Math `-4.5`**, **Code `-3.1`**, and **Simple Tools `-3.0`**) while spending at most `2.0%–3.5%` of Step 2's token savings buffer (`~55% -> ~52.5%` saved) to bring full-test accuracy from `~ -1.7` to **`>= -0.8` points**: specifically, **Math (`400` pairs = 31.25% of Step 3)** gets `60%` length-matched `be-right` (`--ratio 1.15`, 50:50 `right_long`/`right_short` with question re-check priority) + **`35%` widened `guard` pairs** (`--guard-widened 0.70`, shortest right answer above the 0.35 floor beating premature wrong cuts below $0.70\tilde{L}_{\text{think}}$) + `5%` `loop`; **Code (`360` pairs = 28.1%)** gets `65%` length-matched `be-right` + **`25%` widened `guard`** (`0.70`) + `10%` response `loop`; **Simple Tools (`240` pairs = 18.75%)** gets `80%` length-matched `be-right` (enforcing exact function-call argument & schema fidelity) + `20%` `guard`; and **IFEval (`160` pairs: `25%` true `length` counting + `65%` `be-right` + `10%` response `loop`), Several Tools (`60` pairs: `100%` `be-right`), and No-Fit (`60` pairs: `100%` `be-right`)** receive light maintenance quotas so their Step 2 compression is preserved without over-protecting.

---

## 3. RED FLAGS (DATA, CODE, EVALUATION, AND PLAN)

1. **RED FLAG 1 (Evaluation — Dev-512's `+11.3` pt Over-Reading on `bfcl_simple` and Dev-SVAMP's `-7.3%` Saving / `+3.4` pt Accuracy Bias):**
   - On `r2s1b-s180`, Dev-512 read `+8.3` on `bfcl_simple` while the 400-question Full Test read `-3.0` (an **`11.3`-point discrepancy** that accounts for nearly two-thirds of Dev-512's `+2.95` overall accuracy over-reading). Meanwhile, Dev-SVAMP (200 questions) read `54.7% / -1.12` where the 1,000-question Full Test SVAMP read `62.0% / -4.50`.
   - **Action:** Never trust raw `r2pick` (`acc >= -0.80` on raw Dev-512) to gate checkpoints. In `track.py`, `r2pick` currently gates on raw `dev_a >= -0.80` (`L3838`), which passed `s180` (`+1.25` raw) by `2.05` points while the full test was `-1.70`. Always apply the `-2.95` pt Dev-512 offset (requiring raw Dev-512 accuracy $\ge +1.8$ to $+2.0$ after Step 3 repair) and monitor `bfcl_simple` and `svamp` separately.

2. **RED FLAG 2 (Code — `r2pick` Automatically Launches a Full Test on the Runner-Up Checkpoint):**
   - In `code/track.py` (`L3871–3879`), `r2pick` automatically triggers `fullmodel` (a 2.25 GPU-hour Full Test) on `runner_up` whenever `abs(runner_up - best) <= 20`. That is why GPU time was spent running a full test on `r2s1b-s160` right after `s180`.
   - **Action:** Pass `--no-runner-up` or disable the automatic second full test in `r2pick` during Step 2 so GPU 2 immediately starts generating Step 3 rollouts on the winning Step 2 checkpoint rather than burning 2.25 hours testing a neighbour checkpoint.

3. **RED FLAG 3 (Data — `r2prep` Shortfalls in Code Questions and Hard No-Fit Questions for Steps 2 & 3):**
   - As logged in `NEW_RESULTS_13nov_morning.md` §5, `r2prep` has two data gaps relative to the full test:
     1. **Code (`mbpp`)** has only `387` questions in Step 2 and `339` in Step 3 (want 400 each). Because code solve rates sit near `~52%`, a 387-question pool can fall short if Step 2 asks for a high `shorten` quota on code (which requires $\ge 80\%$ solve rate). Setting Code's `shorten` share to **`0%` in Step 2** (using `be-right` + `loop`, which operate on intermediate $1\text{--}7/8$ solve-rate questions) completely avoids any Code pair shortage!
     2. **No-Fit (`bfcl_irrelevance`)** has a deficit of hard questions (`<25%` solve-rate bin is `7%` vs `12%` on the test). If `--shorten-min-right` is kept at `0.80` (`>= 7/8` right), Step 2 will only train on easy no-fit questions and ignore hard abstentions. Lowering no-fit's shorten threshold to `0.75` (or `0.625` backup) and keeping `40%` `be-right` (which specifically draws from `<80%` solve-rate questions) ensures hard abstentions are trained.

4. **RED FLAG 4 (Code/Plan — Hidden Shortening Inside `be-right` Pairs if `guard` or Unbalanced `right` Pairs Are Used):**
   - Earlier runs (`rpK`, `rpL`) suffered hidden math shortening even with `0%` nominal math shorten pairs because `guard` pairs pick the *shortest* right answer (`rights[0] > wrongs[0]`, where chosen is shorter in `~59%` of pairs) and `loop` pairs pick `rights[0] > runaway[0]` (where chosen is always much shorter).
   - **Action:** In Step 2, keep Math `loop` share small (`<= 10%`, i.e., 32 pairs) and `guard` at `0%`, so `90%` of Math pairs are strictly length-balanced `right_long` / `right_short` (`--ratio 1.15`). In Step 3, when widened `guard` pairs (`--guard-widened 0.70`) are activated on Math and Code to fix `-4.5` and `-3.1`, every widened guard has `chosen` **strictly longer** than `rejected` ($L_{\text{chosen}} \ge 0.35\tilde{L}_{\text{think}} > 0.70\tilde{L}_{\text{think}} \ge L_{\text{rejected}}$ when floored/widened properly—Wait: verify in `repair_pairs.py` L354 that `w_cut = max(floor, args.guard_widened * med)` requires ` thinking_len(w) < w_cut` AND we must also require `thinking_len(chosen) > thinking_len(rej)` so a guard pair NEVER picks a chosen answer shorter than the rejected wrong cut!). Checking `repair_pairs.py` L351–357: `w_short = [w for w in wrongs if thinking_len(w) < w_cut]` picks `chosen = rights[0]` (which only has `thinking_len(r) >= floor = 0.35 * med`), meaning if `thinking_len(rights[0])` is `0.40 * med` and `thinking_len(w_short[0])` is `0.65 * med`, **a widened guard pair can actually have `chosen` SHORTER than `rejected` (`0.40 < 0.65`)!**
   - **Critical Fix for Step 3 Widened Guards (`repair_pairs.py` L351):** In `repair_pairs.py`, constrain `w_short` or `chosen` in widened guards so that `thinking_len(chosen) > thinking_len(rej)` (e.g., `w_short = [w for w in wrongs if thinking_len(w) < min(w_cut, thinking_len(rights[0]))]`), guaranteeing that 100% of guard pairs punish premature cuts by preferring a **longer** complete reasoning trace over a **shorter** truncated wrong trace.