# LUPIN: training spec

The final ThinkingCap model: Qwen3-0.6B that thinks shorter at (nearly) the same accuracy.
Written 27 Oct 2026. Stage 2 (RL) is implemented only once Stage 1 is scored; Stage 1 can be re-run and refined first.

**Goal:** more than 50% of tokens saved at an accuracy change better than −1 point, on the fresh set ×8. Then one full
official evaluation (2,759 questions × 8).
**Reference points (fresh set ×8):**

| model | saved | accuracy |
|---|---|---|
| michal-bida | 47.0% | −0.78 |
| Lightning McQwen3 | 43.2% | −0.85 |
| rc1-rl-40 (our best so far) | 53.9% | −2.1 |

---

## 0. Rules that hold everywhere

| set | what it is | use |
|---|---|---|
| **Training questions** | never evaluation questions | training only. An 8-word contamination check runs against every evaluation question. |
| **Choosing set** (`questions_val20261023`) | 40 questions per benchmark part | the only benchmark questions allowed for *decisions*: matching the data, picking a variant, picking a checkpoint. Only aggregate statistics are compared. |
| **Fresh set** (`questions_fresh20261022`) | 40 questions per part | reported scores only. Nothing is chosen on it except the final pick among finished candidates. |
| **Full official evaluation** | 2,759 questions × 8 | once, at the end |

- **Fixed:** the evaluation settings (prompt, sampling, token cap). No test-time gate or switch, and thinking stays on.
  BottleCap's rules require the shortening to live in the weights.
- **Saved throughout:** every stage resumes after a lost runtime, and every result is logged in EXPERIMENT_LOG.md.
- **Credit:** ideas taken from others (michal-bida, Lightning McQwen3, papers) are credited in IDEA.md.

---

## Step 0: The dataset (built: `buildpool` stage, `build_pool.py` + `pool_check.py`; dp1 queued)

### 0.1 Sources per evaluation part

| evaluation part | share of the pool | training sources | format |
|---|---|---|---|
| SVAMP (math) | 3,000 | matched GSM8K 70% + MetaMathQA GSM_SV/FOBAR 30% | SVAMP's answer instruction ("Final answer: …") |
| MBPP+ (code) | 3,000 | KodCode (reference solution must pass its tests) | MBPP+'s instruction + function-name reminder |
| BFCL simple | ~1,430 | ToolACE, one function offered | BFCL's own system prompt + tool list |
| BFCL multiple | ~710 | ToolACE, 2–4 functions offered | same |
| BFCL irrelevance | ~860 | ToolACE, no function fits | same |
| IFEval | 3,000 | IFEval-like (argilla), gradable constraints only | plain user prompt |

The BFCL split follows the evaluation's own sizes (400 / 200 / 240). The score counts each of the 4 benchmarks equally,
so each gets ~3,000 questions.

### 0.2 Matching
- **Prompt length:** chat template applied, tool list included. Drawn bin by bin from the choosing set's 5 length bins,
  within its shortest-to-longest range.
- **Sub-type mix:** BFCL categories as above. For math, SVAMP's perturbation kinds are approximated by the matched-GSM8K
  corpus, which is built on SVAMP's operation count × kind.
- **Contamination:** 8-word sequences are compared with every evaluation question. Sequences found in ≥ 3 evaluation
  questions are shared templates and exempt.
- **Gradability:** only questions the evaluation's own graders can score are kept.

### 0.3 Verification: the pool must behave like the evaluation before anything trains on it
The original model answers 200 pool questions per part, 4 times each. Each part is compared with the same model on the
choosing set (v4-base):

| measure | tolerance |
|---|---|
| accuracy | ±5 points, or inside the choosing set's 95% interval |
| median reply tokens, median thinking tokens | ±15%, or inside the interval |
| share capped at the token limit | ±3 points, or inside the interval |
| *(to add)* parse rate: answers the grader can read | ±3 points |
| *(to add)* answer timing: success when answering from 20–40% of the thinking ÷ success at the full thinking (the "SVAMP-likeness" test; SVAMP 0.47) | ±0.1 |

Timing needs forced answers at cuts: about 100 questions per part, a few cuts each (label_cuts.py).

- **If a part reads CHECK:** adjust that part only (source mix, length bins, a difficulty filter such as dropping the
  easiest decile by base accuracy), rebuild it, and re-check it.
- **Gate:** all parts OK, or a written reason why a remaining gap is acceptable.

---

## Step 1: The answers (to build: an `answers` stage, `probe_curve.py --generate-only` on the whole pool)

- **1.1 Main set (decided by the user, 27 Oct night): 10 answers per question**, starting only once the dataset passes its check. The pool's 4 selection answers per question are reused, so 6 more are generated (~72,000 answers, ~13 GPU-hours on an L4 at ~5,500 answers/hour). Why 10: accuracy first. With 4 answers a question solved 70% of the time shows 4/4 about 24% of the time and gets wrongly shortened; with 10, 3%. Spread is a bonus (longest/shortest right ~1.5x measured at 4 answers; more expected at 10). Later rounds (each LUPIN round re-answers 4-8 times and retrains) are decided per round.
- (previous text) the original model answers every pool question **8 times**, graded with the evaluation's graders.
  That is ~96,000 answers, ~10–12 h on one L4 (~5–6 h split across two), resumable by chunk. Eight answers give a usable
  solve rate p per question (0, 1/8 … 1) and enough length spread for pairs. Lightning McQwen3 and michal-bida used 4;
  Kimi and DAST used 8–20.
- **1.2 Prompted set (optional, decided by ps1):** the same questions answered under the best "think briefly" prompt from
  `prompt_search.py`, with the prompt stripped afterwards. Only answers that do not mention the prompt are kept (leak
  filter). These are short answers still in Qwen's own voice; a pair can then use a prompted answer as the "chosen" side.
- **1.3 Cut labels (optional, for cut-based recipes):** on a subset (~1,500 questions), answers cut at paragraph ends and
  re-answered 4× from each cut (label_cuts.py). This gives the earliest safe stop per answer: our lots, Lightning
  McQwen3's oracle.
- **1.4 Statistics stored per question:** solve rate p, lengths of right / wrong / capped answers, the part.

Everything in Step 2 is built from these fixed answers, so comparing recipes costs only training time.

---

## Step 2: Stage 1, offline training on the answers → LUPIN0 and its variants

### 2.0 The PAIRS must match the evaluation too (user, 27 Oct night: "as important")
A representative pool is not enough: pair recipes skew it (always-failed questions give no shorten pairs, always-solved
easy ones give many), so training would be dominated by easy questions. After the 10 answers, the training pairs are
selected like the pool: per evaluation part (each benchmark 25%, BFCL 400 / 200 / 240 inside its share, as the score
weights them) and per cell of solve rate x thinking length, each cell keeps the evaluation's share (original model's
runs on the choosing set + questions_40). Hard cells are still present, with right / guard / hard pairs instead of
shorten pairs. The pair set's mix is checked against those references (same report as pool_check) before any
training; a recipe whose pairs cannot fill a cell says so instead of silently over-weighting another.

### 2.1 Pair kinds (the vocabulary)

| kind | chosen > rejected | teaches |
|---|---|---|
| **shorten** | a right answer > a longer right answer | be shorter |
| **guard** | the shortest right > a wrong or capped answer | be right, and short |
| **hard** | a longer wrong > a shorter wrong (DAST) | on questions it fails, keep thinking |
| **right** | a right answer > a wrong one of similar length (±25%) | be right; says nothing about length |
| **cut** | the answer cut at its earliest safe stop > the full answer | stop where the answer is already settled |
| **brake** | a full right answer > its early cut that fails | do not stop too early |
| **prompted** | a prompted short right answer (prompt stripped) > the normal long answer | think in the terse style |
| **first-check** | the answer cut after its first right result + one check (Chen et al.) > the longest | one check, then stop |

### 2.2 Recipes (each one is a LUPIN0.x candidate)

| id | recipe | built? | source |
|---|---|---|---|
| R1 | **long2short:** guard + shorten (≥ 1.5× longer), shorten capped at guard per part | ✅ `lupin_pairs.py --recipe long2short` | Kimi k1.5, michal-bida |
| R2 | **difficulty:** R1, but a question solved < p\* gets no length push (only one *right* pair) | ✅ `--min-solve 0.5 / 0.75` | ALP, DAST budget, Difficulty-Aware DLER |
| R3 | **DAST:** shorten + hard, widest pair per question, narrowest 15% dropped | ✅ `--recipe dast` | DAST |
| R4 | **Ada-R1:** shortest right > each of the 2 longest answers | ✅ `--recipe adar1` | Ada-R1 |
| R5 | **right only:** similar-length right > wrong (the accuracy baseline; expect little saving) | ✅ `--recipe rightwrong` | RevThink-like accuracy DPO |
| R6 | **cut oracle:** cut + brake from 1.3, mixed 40 / 40 / 20 with guard | partly (targeted_pairs.py) | Lightning McQwen3, our lots |
| R7 | **first-check cuts:** for math the answer cut after its first right result + one re-check; for tools after the first right call; for code at the first passing draft | to build (needs forced answers) | Chen et al. 2412.21187, ThinkBrake |
| R8 | **prompted:** prompted short right answers as chosen (from 1.2) | to build | Lightning McQwen3 (as pairs instead of SFT) |
| R9 | **per-part blend:** each part gets the recipe that won for it (e.g. difficulty for math, long2short for tools) | after the screen | ours |

Any recipe can also take:
- `--drop-similar 0.8`: drops near-identical pairs (likelihood displacement);
- `--add-right`: adds right pairs.

### 2.3 Training methods (beyond plain DPO)

| id | method | what changes | built? |
|---|---|---|---|
| M1 | **DPO + anchor** (default): β 0.1, lr 2e-5, LoRA r32, 1 epoch, NLL anchor on the chosen answer | — | ✅ dpo.py |
| M2 | **DPO without anchor** (Chen et al.: the anchor did not help length pairs) | `--nll 0` | ✅ |
| M3 | **checker practice** (VeriThinker): + "is this solution right? Yes/No" on our graded answers, weight 0.1, every 2nd pair | `--checker` | ✅ lupin_checker.py + dpo.py |
| M4 | **SVAMP backward questions** (RevThink/FOBAR): mask a number, give the answer, the model recovers the number; an extra NLL task on its own right answers | to build | — |
| M5 | **SimPO** (length-normalised, no reference model; compressed most in Chen et al.) | new loss option | to build |
| M6 | **KTO-style unpaired** ("good" / "bad" answers without forcing pairs: uses every answer, not one pair per question) | new loss option | to build |
| M7 | **multi-negative / listwise DPO:** one chosen against all worse answers of the question at once (ranking by right-then-short) | new loss option | to build |
| M8 | **prompt-distillation SFT first** (Lightning McQwen3's step a), then the best recipe on the distilled model's own answers | 2 rounds | SFT exists (sft.py); the round needs a stage |
| M9 | **stop-token training** at paragraph ends from the cut labels (gold1 idea) | small extra loss | stop_gold.py exists |
| M10 | **iterative round:** LUPIN0 answers the pool again (on-policy), new pairs, second DPO round (the cheap version of RL; michal-bida's next step) | 2 rounds | stages exist |
| M11 | **curriculum:** easy questions (high p) first, hard last; or two epochs where epoch 2 has only right / hard pairs | order only | to build |
| M12 | **TokenSkip densify** (arXiv 2502.12067): LUPIN0's right answers with the thinking cut to its most important words by LLMLingua-2 (reply verbatim), kept only if the model still answers right from the cut thinking; SFT on top of LUPIN0. Trims words within steps (pairs and RL trim whole steps), so the two can stack. Paper: ~50% fewer tokens costs 14B -1.7, 7B -5.4, 3B -9.3 pts (GSM8K): expect gentle ratios (0.8-0.9) at 0.6B, likely not math | SFT round | built: tokenskip.py + `tokenskip` stage (raw test on the original Qwen queued first, ratios 0.5 / 0.7 / 1.0 control) |

### 2.4 Free dials after training (no retraining)
- **Soups:** average the 3 seeds at each checkpoint (built).
- **Strength dial:** blend the trained change with the original model, e.g. ×0.5 / ×0.7 (DLER's merge: keep the top 25%
  of the change at ×0.7). One trained model then gives a whole saving/accuracy curve, and we pick the point that meets the
  goal. merge.py already supports weights.
- **Per-part adapters merged:** train one adapter per part's pairs, merge with weights; tests whether parts interfere.

### 2.5 Protocol for Stage 1
1. **Screen** (cheap): every recipe R1–R5 with M1, 1 seed, a fixed budget (e.g. 200 steps, checkpoints every 50).
   Scored on the **choosing set ×4** with the per-part breakdown.
2. **Narrow:** keep the 2–3 best on the trade-off. Add the method variants (M2, M3, then M5–M7 if built) on the best
   recipe.
3. **Confirm:** finalists with 3 seeds, soups at several steps, scored on the **fresh set ×8**. Then the strength dial
   on the best soup.
4. **LUPIN0** = the best fresh-set point within the accuracy limit. LUPIN0.1, 0.2, … are the kept variants.
5. **Refine:** if no variant meets the goal, read the per-part table. Fix the weakest part's pairs (often math accuracy
   or code saving) and re-run steps 1–3 for that part only.

---

## Rounds and what we show (user, 28 Oct)
- **Round 1 → LUPIN v1:** the original Qwen answers the checked pool 10 times per question; pairs selected to the
  evaluation's distribution (2.0); DPO.
- **Round 2 → LUPIN v2:** LUPIN v1 answers a NEW representative sample 8-10 times per question; pairs selected the same
  way; DPO from LUPIN v1. "Representative" is re-measured with LUPIN v1: its own runs on the choosing set and
  questions_40 are the reference the new sample is matched to (as the original model's were for round 1).
- **Shown after each round:** the per-type thinking-length distribution and accuracy of LUPIN v1, then v2, against the
  original Qwen, on the fresh test (240 x 8), in the ThinkingCap Pool Overlap dashboard (one curve per model per type);
  the final model also on the full evaluation (2,759 x 8).
- No training starts before the dataset check passes (user, 28 Oct).

## Step 3: Stage 2, online RL from LUPIN0 → LUPIN1.0 (implemented after Stage 1 is scored)

RL lets the model answer live and pushes it towards its own better-scoring answers, so it keeps improving where fixed
pairs run out. Every RL variant starts from LUPIN0, keeps math and BFCL multiple protectable, and has the per-type
guard (stop if a type drops ≥ 10 points in training). Checkpoints every 5 steps are scored on the fresh set ×8.

| id | RL method | reward / objective | status |
|---|---|---|---|
| RL1 | **DRPO** (current): right answers pushed up, shorter ones more (softmax of relative length / λ); wrong answers pushed down, length-blind | right-only length pressure | ✅ rl_train.py |
| RL2 | **DRPO + solve-rate scaling** (ALP): λ_q = λ / max(p, 1/K), so easy questions get full pressure and hard ones almost none | ✅ `--drpo-solve-scale` |
| RL3 | **ALP proper:** reward = right − β·length·p on *all* answers (wrong ones get shorter too), GRPO | new reward option |
| RL4 | **budget reward** (DAST budget): target length L = p·(mean right length) + (1−p)·L_max; reward for staying under L when right | new reward option |
| RL5 | **ShorterBetter** at the paper's β 0.001, with protection | ✅ (sb2 queued) |
| RL6 | **DLER recipe:** truncation penalty + dynamic sampling (only groups with a right answer) + batch-level normalisation | partly (dynamic sampling built) |
| RL7 | **progressive cap** (ThinkPrune): train under a token cap lowered in stages (e.g. 4k → 3k → 2k), the cap only in training | new option |
| RL8 | **stop reward:** a bonus when the thinking ends at a paragraph end the cut labels mark safe, a penalty when it stops where they mark unsafe | new reward, needs labels |
| RL9 | **checker reward / self-verification:** the model's own Yes/No judgement (from M3) as a confidence signal; reward a right answer with high confidence and short length | creative, new |
| RL10 | **online DPO:** each step, sample K answers, build pairs with the winning Stage-1 recipe, take one DPO step (on-policy pairs instead of rewards) | new loop, reuses lupin_pairs |
| RL11 | **per-part λ:** a separate length pressure per part, set from Stage 1's per-part trade-off (e.g. tools strong, math weak) | small option |

**Protocol:** RL2 vs RL1 first, with the same λ, steps and seed. Then the winner with 2 λ values. Then one creative
variant (RL8–RL10) at a time. LUPIN1.0 = the best fresh-set point. It must beat LUPIN0 on the goal, or LUPIN0 is the
submission.

---

## Rough compute (one L4)

| step | time |
|---|---|
| Step 0, build + check | 1–2 h |
| Step 1, 8 answers × 12,000 questions | 10–12 h (5–6 h on two GPUs); prompted set +~6 h; cut labels on 1,500 questions +~4 h |
| Step 2 screen (5 recipes × 1 seed × 200 steps + choosing-set scores) | ~10–12 h |
| Step 2 confirm (3 finalists × 3 seeds + fresh-set soups) | ~15 h |
| Step 3, per RL run (40 steps + scoring every 5) | ~10 h |

## What gets written up (IDEA.md)
- The data-matching method and its check.
- The recipe screen, with per-part results.
- The winning recipe and methods.
- The RL result.
- Negative results.
- Credits.
