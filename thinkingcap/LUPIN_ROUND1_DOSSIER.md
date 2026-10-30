# LUPIN round 1 — data and experiments dossier (state: 30 Oct 2026, ~14:00 CET)

Plain record for external analysis: context, data, methods, and every measured result. No ranking or interpretation
beyond what is needed to read a number. The full chronological lab notebook is `EXPERIMENT_LOG.md` (same repo); this
file summarises the LUPIN round-1 work from 27 Oct onward. Results still running at the time of writing are marked
"pending".

---

## 1. Task and evaluation

- **Model:** Qwen3-0.6B (thinking mode on, `enable_thinking=True`). Goal of the work: fewer generated tokens (mainly
  thinking) at minimal accuracy loss. All changes live in the weights (LoRA adapters merged); no test-time switch.
- **Official evaluation** (BottleCapAI ThinkingCap-Test, `evaluate.py` / `compare.py`): 2,759 questions, 8 samples each.
  - Benchmarks: SVAMP (math), MBPP+ (code), IFEval (instruction following), BFCL (function calling; 3 parts: one
    function "simple" 400 questions, "multiple" 200, "irrelevance"/no-fit 240).
  - Sampling: temperature 0.6, top-p 0.95, top-k 20; token cap 8,192 for MBPP+, 4,096 otherwise.
  - Score: per benchmark, accuracy change and token reduction vs the base model; the headline averages the **4**
    benchmarks (BFCL's three parts pooled into one): accuracy change = arithmetic mean; tokens saved = 1 − geometric mean
    of the per-benchmark token ratios (candidate / base).
- **Question subsets used in this work** (all drawn from the official 2,759; aggregate statistics only are used for
  decisions):

  | set | size | use |
  |---|---|---|
  | `questions_val20261023` ("choosing set") | 40 per part = 240 | screens and decisions, ×4 samples |
  | `questions_40` | 40 per part = 240 | reference statistics (×2 samples) |
  | `questions_ref20261024`, `questions_ref20261025` | 40 per part each | reference statistics (×4); the first also for cross-checks |
  | `questions_fresh20261022` ("fresh set") | 40 per part = 240 | confirm scores only, ×8 samples; never used for any choice |

- **Re-weighting:** subsets hold 40 questions per BFCL part (equal weights inside BFCL); the full evaluation holds
  400/200/240. From 30 Oct ~12:45 every score is also printed with BFCL re-pooled at 400/200/240 ("full-eval
  weighting"; tokens weighted by each part's base mean output length).
- **Noise reference (measured earlier in the project):** run-to-run noise ~0.8 pt overall on the fresh set ×8; a 240 ×4
  screen moves ~±1.5 pts overall and ~±5 pts per part (40 questions per part).
- **External reference points (public repos, full evaluation ×8):** michal-bida/thinking-compression 47.0% saved /
  −0.78 pts (one DPO round on own graded samples, long2short-style pairs, per-domain cap shorten ≤ right-vs-wrong
  pairs); basta/lightning-mcqwen3 43.2% / −0.85 (47.3% / −0.9 with a test-time gate; prompt-distillation SFT → cut-oracle
  pairs → math-only right>wrong DPO). Earlier project model rc1-rl-40 (DPO + RL): 53.9% / −2.1 on the fresh set ×8.

---

## 2. Training data ("pool")

Training questions are never evaluation questions: every candidate passes an 8-word-sequence contamination check
against all 2,759 evaluation questions (sequences found in ≥3 evaluation questions treated as shared templates); MBPP
itself is excluded.

### 2.1 Sources per part

| part | sources |
|---|---|
| SVAMP (math) | GSM8K-matched, MetaMathQA GSM_SV/FOBAR, "easy" GSM8K (1–2 calculations) and MetaMath rephrasings, all with SVAMP's answer instruction |
| MBPP+ (code) | KodCode-V1 (medium/hard, one function, assert tests from the dataset) |
| IFEval | IFEval-like (argilla) + prompts built from IFEval's own instruction library on no_robots bases (instruction mix matched to IFEval) |
| BFCL simple | Glaive function-calling v2 (one call, grounded arguments) + ToolACE (clean single-turn) + Hermes |
| BFCL multiple | ToolACE (2–4 functions offered) |
| BFCL no-fit | ToolACE "near" no-fit; Glaive question + another Glaive function at embedding ranks 15–60 (nofit_glaive_verified, 6,070) and ranks 4–15 (nofit_glaive_hard, 2,441); every no-fit label checked twice by claude-haiku-5.5 (kept only on NO + NO) |

### 2.2 Matching method

1. **Prompt length:** each part's candidates drawn bin by bin from the choosing set's 5 prompt-length bins.
2. **Behaviour:** the original Qwen answers each candidate (4 to 10 times). Each question gets a solve rate and a median
   thinking length; candidates are selected so each part reproduces the reference's share of cells (solve bin
   never / <½ / ≥½ / always × thinking quartiles; 8 thinking bins for BFCL simple from dp11 on).
3. **Proportions:** each benchmark 25% of the pool; BFCL parts 400 / 200 / 240 inside its 25%.
4. **Held-out check (`pool_check.py`):** 200 pool questions per part answered ×4 with a NEW sampling seed; 6 measures
   per part (accuracy, median reply tokens, median / p25 / p75 thinking tokens, capped share) compared with the
   reference; a measure is OK if within ±5 pts accuracy / ±15% length / ±3 pts capped, or inside the reference's 95%
   interval. 36 measures in total.
5. **Reference:** choosing set ×4 + questions_40 ×2 (80 questions per part) until 29 Oct; from 29 Oct night also the two
   refsets (160 questions per part).

### 2.3 Pool versions and held-out checks

| pool | main change | reference | measures OK | measures off |
|---|---|---|---|---|
| dp1 | prompt-length matching only | 80/part | 15/24 | math thinking +40%; IFEval-like 89 vs 49% right; simple 56 vs 91% (grading); no-fit thinking +31% |
| dp3 | behaviour-cell selection | 80/part | 17/24 | math thinking +41% (wrong answer instruction in one source); IFEval thinking −22%; no-fit 98 vs 79% right |
| dp4 | easier math, "near" no-fit | 80/part | 17/24 | IFEval thinking −26%; simple thinking −19%; no-fit 94 vs 79% |
| dp5 | IFEval built prompts | 80/part | 30/36 | math p25 +17%; code p75 −35%; simple median −16%; no-fit 96 vs 79% |
| dp6 | Haiku-verified no-fit, Hermes | 80/part | 29/36 | math accuracy 83 vs 88; code p75 −31%; simple −17% median; no-fit 91 vs 79% |
| dp7c | KodCode medium/hard, Glaive one-call, one-function no-fit | 80/part | 32/36 | simple 80 vs 88 and p75 −19%; no-fit 88 vs 79; IFEval p75 −17% |
| dp8 | 8 answers per tool/IFEval candidate, 4 thinking bins | 80/part | 33/36 | math 81 vs 88; simple 81 vs 88 and p75 −21% |
| dp9 | + Glaive no-fit (ranks 15–60) + more Glaive simple, 8 answers for math | 80/part | 34/36 | code p75 1541 vs 2220 (−31%); simple p75 166 vs 199 (−17%) |
| dp10 | 2,400 code candidates on 8 answers | 80/part | 35/36 | simple p75 160 vs 199 (−20%) |
| dp11 | next 1,200 code per half on 8 answers, 8 bins for simple | 80/part | 30/36 | math thinking +15%; simple accuracy 78 vs 88; no-fit 88 vs 79; IFEval 65 vs 51 |
| dp12 | dp10 + top-ups nofit1 (189), tools1 (724), code1 (800): hardest "sometimes right" unused candidates, 10 answers each | 160/part | 33/36 | no-fit median thinking 156 vs 128 (+22%), p75 +24%, reply +18% |
| dp13 | dp12 + nofit2 (552 hard no-fit, ranks 4–15) | 160/part | 33/36 | simple accuracy 78 vs 85 [80, 89]; math p75 578 vs 759 [591, 976]; IFEval accuracy 59 vs 52 [45, 58] |
| dp14 | dp13 without ToolACE simple | 160/part | 34/36 | simple median thinking 120 vs 142, p75 142 vs 190 |
| dp15 | dp13 minus ToolACE simple solved < 50% | 160/part | 33/36 | math accuracy 81 vs 88; simple p75 −18%; IFEval p75 −16% |

Notes recorded with the checks: with the 160-question reference, BFCL no-fit reference accuracy is 87% (the 80-question
reference gave 79%). Unchanged parts move by ±3–7 pts between checks (e.g. math 84 → 81 between dp12 and dp15 with the
same candidate set).
**dp13 was adopted as the training pool for the experiments below (user decision, 30 Oct ~08:15).**

### 2.4 Answers per question

Every pool question has exactly **10** answers by the original Qwen (newest first: the 10-answer run, then extra-answer
rounds, then the selection answers), graded with the evaluation's graders. Pool size: 12,000 questions (dp10/dp13).

### 2.5 Topic check (embedding similarity, report only)

MiniLM nearest-neighbour cosine, eval→rest-of-eval vs eval→pool (pool subsampled to the eval part's size), dp10:

| part | eval→eval | eval→pool | eval questions below the eval's own 10th percentile |
|---|---|---|---|
| svamp | 0.95 | 0.65 | 100% |
| mbpp | 0.75 | 0.66 | 41% |
| bfcl simple | 0.59 | 0.38 | 57% |
| bfcl multiple | 0.51 | 0.31 | 60% |
| bfcl no-fit | 0.49 | 0.33 | 52% |
| ifeval | 0.50 | 0.46 | 13% |

---

## 3. Pairs and training

### 3.1 Pair kinds (from each question's 10 graded answers)

| kind | chosen > rejected |
|---|---|
| shorten | a right answer > a right answer ≥ 1.5× longer (1.2× on always-solved questions with `--easy-ratio 1.2`) |
| guard | the shortest right answer > the longest wrong / capped answer |
| right | a right answer > a wrong answer of similar length (±25%) |
| hard (DAST) | a longer wrong answer > a shorter wrong answer |

Difficulty protection (`--min-solve p`): a question solved < p of the time gets no shorten / guard pair, only a right
pair. Per-part overrides (`--min-solve-part`, `--right-parts`) used in some experiments.

### 3.2 Matching the pairs to the evaluation (`lupin_select.py`)

Pairs are drawn per part at the evaluation's shares (25% per benchmark; BFCL 400/200/240) and, inside a part, per cell
(solve rate × thinking quartile) at the reference's shares; questions visited round-robin; cells with no possible pair
under a recipe have their share spread over the other cells; then every part is trimmed to the lowest fill ratio so part
shares stay exact. Default target 3,200 pairs (`--lp-total`). Options used: `--max-shorten-ratio` / `--shorten-ratio-part`
(cap of shorten pairs per be-right pair, per part), `--part-share`, `--solve-tilt`, `--think-tilt-bfcl`.

### 3.3 Training

DPO (`dpo.py`) from the original Qwen: LoRA r32, β 0.1, lr 2e-5, NLL anchor weight 1.0 on the chosen reply, 16 pairs
per step, 1 epoch, 48 pairs held out as a tripwire; checkpoints every 25 steps (every 10 in some confirms). Screens: 1
seed, each checkpoint merged and scored on the choosing set ×4. Confirms: 3 seeds averaged ("soup") per checkpoint, scored
on the fresh set ×8. Cross-checks: the same soups on `questions_ref20261024` ×4 vs the original Qwen on that set.

### 3.4 Pair inventory (pool dp10, long2short, p < 0.5, before matching)

| part | shorten (raw) | guard | right>wrong (rightwrong recipe) |
|---|---|---|---|
| svamp | 2,587 | 660 | 637 |
| mbpp | 1,744 | 898 | 1,180 |
| ifeval | 1,671 | 660 | 890 |
| bfcl simple | 1,114 | 270 | 317 |
| bfcl multiple | 514 | 195 | 253 |
| bfcl no-fit | 718 | 132 | 207 |

Questions never solved by the original Qwen: 2,155 of 12,000; always solved: 5,967.

---

## 4. Results

Format: tokens saved (%) / accuracy change (pts) vs the original Qwen on the same questions; per-part columns likewise.
"Steps" = DPO steps of 16 pairs.

### 4.1 Screens on pool dp10 (choosing set ×4, 1 seed)

| run | recipe | pairs / steps | step | overall | svamp | mbpp | ifeval | simple | multiple | no-fit |
|---|---|---|---|---|---|---|---|---|---|---|
| lpb-l2s75 | long2short, p<0.75 | 1,452 / 88 | 25 | +37.5 / −1.5 | +52 / 0.0 | +44 / −5.0 | +15 / −1.3 | +31 / −0.6 | +31 / +1.9 | +37 / −0.6 |
| | | | 50 | +43.7 / −1.5 | +55 / −5.0 | +53 / −2.5 | +25 / +3.1 | +34 / +1.9 | +37 / −3.1 | +40 / −3.8 |
| | | | 75 | +43.8 / −1.1 | +51 / −1.9 | +51 / −3.7 | +37 / +1.9 | +32 / −1.2 | +33 / +1.2 | +39 / −1.9 |
| lpb-l2s50 | long2short, p<0.5 | 2,012 / 123 | 25 | +36.3 / −0.9 | +49 / −1.2 | +34 / −2.5 | +31 / +1.3 | +26 / −0.6 | +29 / −1.2 | +34 / −1.9 |
| | | | 50 | +47.7 / −1.0 | +55 / −3.7 | +53 / −1.9 | +43 / +3.1 | +35 / −0.6 | +39 / −1.2 | +40 / −2.5 |
| | | | 75 | +47.8 / −1.9 | +51 / −2.5 | +52 / −3.1 | +52 / −0.6 | +31 / −1.9 | +34 / −3.1 | +39 / +0.6 |
| | | | 100 | +47.6 / −1.2 | +48 / −1.2 | +52 / −6.2 | +52 / +3.1 | +34 / +1.9 | +36 / −3.1 | +41 / 0.0 |
| lpb-adar1 | Ada-R1 (shortest right > 2 longest), p<0.5 | ~3,100 / 194 | 25 | +48.5 / +0.4 | +61 / −3.1 | +52 / +5.0 | +42 / +1.9 | +33 / −3.1 | +36 / −3.7 | +40 / 0.0 |
| | | | 50 | +43.2 / +0.3 | +54 / +1.9 | +48 / −0.6 | +31 / +3.8 | +33 / −4.4 | +36 / −5.6 | +43 / −1.3 |
| | | | 75 | +47.1 / −1.4 | +53 / −0.6 | +55 / −1.9 | +42 / 0.0 | +31 / −1.9 | +35 / −5.6 | +42 / −1.9 |
| | | | 100 | +45.0 / −1.0 | +52 / +1.9 | +45 / −2.5 | +46 / −1.9 | +31 / −1.9 | +35 / −3.1 | +41 / 0.0 |
| | | | 150 | +51.4 / −1.7 | +55 / −1.9 | +62 / −4.4 | +49 / +1.3 | +30 / −1.2 | +36 / −2.5 | +41 / −1.3 |
| lpb-dast | DAST (+ hard pairs), p<0.5 | — / 183 | 50 | +34.1 / −0.1 | +49 / −0.6 | +36 / −2.5 | +22 / +3.8 | +22 / +1.2 | +24 / −2.5 | +31 / −1.3 |
| | | | 75 | +30.5 / −0.7 | +49 / +1.3 | +37 / −5.6 | +2 / +3.1 | +23 / 0.0 | +25 / −6.2 | +30 / +1.2 |
| | | | 100 | +37.7 / −0.8 | +54 / 0.0 | +40 / −5.0 | +21 / +3.1 | +29 / −1.2 | +29 / −1.9 | +33 / −1.3 |
| | | | 150 | +36.3 / −1.7 | +54 / −1.2 | +30 / −3.7 | +25 / 0.0 | +31 / −1.2 | +31 / −1.9 | +34 / −1.9 |
| lpb-right | right>wrong only, no protection | — / 121 | 25 | −9.3 / −0.5 | −9 / +2.5 | −3 / −4.4 | −18 / −1.3 | −7 / +1.9 | −3 / +0.6 | −11 / +1.2 |
| | | | 50 | −14.6 / +2.5 | −10 / +3.1 | −17 / +3.8 | −23 / +1.9 | −6 / +1.2 | −14 / −0.6 | −6 / +3.1 |
| | | | 75 | −39.2 / +0.9 | −31 / +3.1 | −38 / −1.9 | −44 / +0.6 | −56 / 0.0 | −40 / +2.5 | −35 / +3.1 |
| | | | 100 | −51.0 / +2.1 | −44 / +3.1 | −63 / −0.6 | −45 / +2.5 | −55 / +0.6 | −74 / +3.7 | −28 / +6.2 |
| lpb-l2s50e | long2short p<0.5 + easy-ratio 1.2 + shorten cap 2× | 2,728 / 168 | 25 | +41.0 / −0.9 | +52 / −2.5 | +43 / −0.6 | +33 / −0.6 | +32 / +2.5 | +30 / −1.2 | +38 / −1.3 |
| | | | 50 | +46.1 / −0.1 | +54 / −1.2 | +50 / −2.5 | +41 / +3.8 | +34 / +1.2 | +37 / −1.9 | +41 / −0.6 |
| | | | 75 | +47.9 / −2.0 | +55 / −2.5 | +50 / −3.7 | +47 / −0.6 | +34 / +1.2 | +38 / −5.0 | +42 / 0.0 |
| | | | 100 | +49.3 / 0.0 | +52 / −2.5 | +56 / 0.0 | +50 / +5.0 | +32 / +0.6 | +38 / −8.1 | +42 / 0.0 |
| | | | 150 | +50.9 / −0.5 | +58 / −1.2 | +55 / −2.5 | +51 / +3.1 | +34 / −3.1 | +39 / 0.0 | +44 / −1.3 |

(Negative saving = longer than the original.)

### 4.2 Confirms on pool dp10 (fresh set ×8, 3-seed soups) and cross-checks

| run | soup step | fresh overall | svamp | mbpp | ifeval | simple | multiple | no-fit |
|---|---|---|---|---|---|---|---|---|
| lpc-adar1 | 25 | +43.0 / −1.9 | +62 / −4.4 | +45 / −1.3 | +26 / +0.6 | +30 / −2.2 | +34 / −2.5 | +30 / −3.1 |
| | 50 | +44.8 / −2.3 | +57 / −3.1 | +52 / −1.3 | +32 / −2.2 | +32 / −2.2 | +36 / −2.5 | +34 / −3.7 |
| | 75 | +44.6 / −3.5 | +55 / −4.7 | +48 / −0.9 | +38 / −5.6 | +32 / −0.6 | +36 / −4.1 | +36 / −3.1 |
| lpc-l2s50e | 50 | +44.1 / −2.7 | +54 / −5.9 | +50 / −1.3 | +37 / −0.3 | +33 / +0.9 | +35 / −4.4 | +32 / −6.6 |
| | 100 | +45.8 / −1.3 | +54 / −1.9 | +53 / +0.3 | +41 / −1.6 | +32 / +0.6 | +36 / −2.8 | +34 / −4.4 |
| | 150 | +45.4 / −3.1 | +57 / −5.6 | +50 / 0.0 | +39 / −3.4 | +31 / −0.9 | +34 / −4.1 | +34 / −4.7 |

lpc-l2s50e-soup-100 with full-eval BFCL weighting (computed from the per-part numbers): +46.1 / −1.2.

| cross-check (refs 20261024, 240 ×4) | step | overall | svamp | mbpp | ifeval | simple | multiple | no-fit |
|---|---|---|---|---|---|---|---|---|
| lpc-adar1 | 25 | +48.6 / −3.4 | +62 / −2.5 | +55 / −6.2 | +38 / −3.1 | +31 / −4.4 | +40 / −0.6 | +27 / 0.0 |
| | 50 | +46.5 / −3.4 | +57 / −4.4 | +57 / −8.8 | +32 / −1.3 | +32 / −1.9 | +42 / +3.1 | +31 / +0.6 |
| | 75 | +48.9 / −3.0 | +61 / −6.2 | +53 / −5.0 | +41 / 0.0 | +30 / −3.7 | +41 / +1.2 | +33 / 0.0 |
| lpc-l2s50e | 100 | +50.5 / −2.7 | — | — | — | — | — | — |
| | 150 | +52.5 / −1.5 | — | — | — | — | — | — |

Strength dial on lpc-adar1-soup-25 (trained change scaled; fresh ×8): ×0.5 +23.1 / +0.3; ×0.7 +31.3 / −1.6; ×1.0
+43.0 / −1.9.

### 4.3 Screens on pool dp13 (choosing set ×4, 1 seed)

Common recipe unless stated ("base recipe"): long2short, p<0.5, easy-ratio 1.2, shorten cap 2×, no-fit protection
(p<0.8 on no-fit + right pairs on no-fit); pairs matched against the 160-per-part references.

| run | change vs the base recipe | steps | step | overall | svamp | mbpp | ifeval | simple | multiple | no-fit |
|---|---|---|---|---|---|---|---|---|---|---|
| lpv-base | — (base recipe) | 86 | 25 | +46.3 / −1.7 | +58 / −5.0 | +47 / −3.7 | +43 / +3.1 | +31 / −1.2 | +33 / −1.9 | +39 / 0.0 |
| | | | 50 | +43.0 / −1.9 | +52 / −0.6 | +47 / −4.4 | +34 / 0.0 | +32 / +0.6 | +36 / −6.2 | +39 / −1.9 |
| | | | 75 | +43.2 / +0.2 | +50 / −1.2 | +51 / +0.6 | +34 / +3.1 | +32 / +2.5 | +38 / −3.7 | +39 / −3.8 |
| lpw-p1tilt | benchmark shares 15/25/35/25 (math/code/instr/tools) | 83 | 25 | +41.7 / −1.0 | +53 / −1.9 | +49 / −1.9 | +29 / +2.5 | +31 / +0.6 | +32 / −5.0 | +35 / −3.8 |
| | | | 50 | +45.6 / −2.9 | +56 / −3.1 | +51 / −7.5 | +35 / 0.0 | +35 / +1.9 | +37 / −2.5 | +42 / −1.9 |
| | | | 75 | +48.7 / −1.8 | +58 / −6.9 | +47 / −3.1 | +51 / +3.1 | +33 / +3.1 | +36 / 0.0 | +41 / −3.8 |
| lpw-2stage | starts from lpb-right step 50 | 86 | 25 | +22.4 / +1.9 | +32 / 0.0 | +5 / +3.8 | +26 / +3.8 | +23 / +1.9 | +19 / −2.5 | +32 / +0.6 |
| | | | 50 | +24.3 / +1.2 | +33 / 0.0 | +8 / +2.5 | +31 / +1.9 | +24 / −1.9 | +18 / −2.5 | +28 / +5.6 |
| | | | 75 | +29.2 / −0.8 | +34 / −1.9 | +17 / 0.0 | +37 / −0.6 | +25 / −4.4 | +26 / −5.6 | +31 / +8.1 |
| lpw-gentle | β 0.3, NLL 2, lr 1e-5 | 115 | 25 | +14.4 / +0.1 | +26 / +3.1 | +10 / −1.9 | +7 / 0.0 | +10 / +0.6 | +13 / −3.1 | +19 / −0.6 |
| | | | 50 | +29.1 / −1.7 | +38 / −2.5 | +28 / −4.4 | +23 / +0.6 | +23 / +1.9 | +23 / −1.9 | +33 / −1.3 |
| | | | 75 | +25.7 / −0.9 | +34 / +0.6 | +26 / −3.1 | +14 / −1.3 | +21 / +1.9 | +28 / −0.6 | +34 / −0.6 |
| lpw-press | shorten per be-right pair capped: math 1, code 2, instr 3, tools 2 | ~72 | 25 | +37.1 / −1.0 | +44 / −0.6 | +37 / −1.9 | +36 / −1.3 | +25 / +1.2 | +31 / +3.7 | +34 / −5.6 |
| | | | 50 | +43.0 / +0.4 | +54 / +0.6 | +48 / −2.5 | +32 / +3.8 | +32 / +1.9 | +36 / −0.6 | +38 / −1.9 |
| lpw-press6k | lpw-press with --lp-total 6400 (more distinct pairs) | 107 | 50 | +44.6 / −1.7 | +51 / −6.2 | +51 / −3.7 | +38 / +3.8 | +32 / +0.6 | +37 / 0.0 | +39 / −1.9 |
| | | | 75 | +44.7 / +0.7 | +51 / +0.6 | +51 / −0.6 | +39 / +2.5 | +30 / 0.0 | +36 / +3.7 | +39 / −2.5 |
| | | | 100 | +39.4 / −1.7 | +42 / −3.1 | +44 / −2.5 | +36 / −0.6 | +31 / −0.6 | +36 / +1.9 | +38 / −2.5 |
| lpw-math | + math protection (p<0.8 on math + right pairs on math) | — | 25 | +40.8 / −0.1 | +55 / −2.5 | +44 / −1.2 | +29 / +4.4 | +25 / +0.6 | +30 / −3.1 | +36 / −0.6 |
| | | | 50 | +50.9 / −0.6 | +61 / −3.7 | +59 / −2.5 | +42 / +3.8 | +34 / +0.6 | +37 / −1.2 | +41 / +1.2 |
| | | | 75 | +48.3 / −0.9 | +56 / −1.2 | +54 / −2.5 | +46 / +0.6 | +31 / +1.9 | +36 / −2.5 | +39 / −1.3 |
| lpw-p3medium | solve-bin tilt: medium ×2, always-solved ×0.5 | 107 | 25 | +34.7 / −1.7 | — | — | — | — | — | — |
| | | | 50 | +39.4 / −2.7 | +46 / −2.5 | +43 / −6.2 | +31 / +0.6 | +34 / +0.6 | +37 / −4.4 | +39 / −4.4 |
| | | | 75 | +41.1 / −1.9 | +44 / −1.2 | +43 / −6.2 | +41 / +1.3 | +32 / −1.2 | +36 / −1.9 | +38 / −0.6 |
| | | | 100 | +43.4 / −0.8 | +47 / −1.2 | +49 / +0.6 | +40 / +0.6 | +33 / −1.9 | +38 / −5.0 | +39 / −3.1 |

Per-type adapters (`pertype`, one LoRA per benchmark trained on its own pairs of the lpv-base pair set, merged with
per-benchmark weights; choosing ×4):

| merge weights (svamp / mbpp / ifeval / bfcl) | overall | svamp | mbpp | ifeval | simple | multiple | no-fit |
|---|---|---|---|---|---|---|---|
| 1 / 1 / 1 / 1 | +77.2 / −9.2 | +85 / −21.3 | +88 / −7.5 | +65 / +1.9 | +56 / −8.8 | +57 / −7.5 | +60 / −13.8 |
| 0.5 / 0.8 / 1 / 0.8 | +70.5 / −6.5 | +77 / −13.7 | +81 / −8.7 | +65 / +5.0 | +49 / −5.0 | +51 / −10.6 | +55 / −9.4 |
| 0.25 / 0.35 / 0.45 / 0.35 | pending | | | | | | |

### 4.4 Confirms on pool dp13 (fresh set ×8, 3-seed soups)

| run | soup step | overall | svamp | mbpp | ifeval | simple | multiple | no-fit |
|---|---|---|---|---|---|---|---|---|
| lpc-press | 40 | +45.5 / −5.2 | +59 / −8.1 | +46 / −3.8 | +40 / −5.6 | +33 / −1.9 | +36 / −4.7 | +31 / −2.8 |
| | 50 | +42.4 / −2.9 | (per part pending in this file) | | | | | |
| lpc-math | 40–70 | pending | | | | | | |

Cross-check of lpc-press (refs 20261024, 240 ×4): soup-40 +51.0 / −2.1 (full-eval weighting +50.8 / −2.2); soup-50 +48.9 / −2.8
(full-eval weighting +48.8 / −2.7). Full-eval weighting on the fresh set: soup-40 +45.5 / −5.1; soup-50 +42.3 / −3.0.

Queued: lpw-pressmath (press + math protection, screen), lpw-p4tools (BFCL
longest-thinking cells ×2).

---

## 5. SPARSE arm C (Wanda pruning of the original Qwen3-0.6B, no retraining)

Method: Wanda (score = |W| × input-feature L2 norm over calibration tokens; lowest share zeroed per output row), block by
block in order on the outputs of the already-pruned blocks; the 7 linear layers of the 28 decoder blocks (~440M
weights); calibration = 128 of the original Qwen's own right answers (≤1,024 tokens each, mixed types, pool questions).

Proxy (no generation): increase of the mean per-token NLL of 80 held-out right answers per part, vs the unpruned model:

| pruned | svamp | mbpp | ifeval | simple | multiple | no-fit |
|---|---|---|---|---|---|---|
| 10% | +2% | +1% | +1% | +2% | +2% | +2% |
| 20% | +9% | +7% | +7% | +14% | +14% | +12% |
| 30% | +33% | +23% | +27% | +50% | +55% | +44% |
| 40% | +90% | +65% | +70% | +131% | +132% | +121% |
| 50% | +217% | +149% | +156% | +306% | +321% | +281% |

Calibration-size check (30%): masks from 384 vs 128 calibration answers share 99.7% of pruned weights (q/k/v/gate/up),
99.1% (o_proj), 98.2% (down_proj).

Real evaluation (choosing set ×4 vs the unpruned model; negative saving = longer):

| pruned | overall | svamp | mbpp | ifeval | simple | multiple | no-fit |
|---|---|---|---|---|---|---|---|
| 10% | +1.8 / +0.2 | +11 / −0.6 | +10 / +1.9 | −17 / −0.6 | +3 / +1.9 | −1 / −1.9 | +1 / 0.0 |
| 20% | −7.7 / −2.0 | −5 / +1.3 | −5 / −5.6 | −21 / −2.5 | −3 / −3.1 | −10 / −0.6 | +9 / 0.0 |
| 30% | −8.9 / −1.9 | +2 / +1.3 | −10 / −5.6 | −17 / −1.9 | −5 / −2.5 | −15 / −1.2 | −14 / −0.6 |

---

## 6. Files

- `EXPERIMENT_LOG.md`: full chronological log (all runs, decisions, bugs).
- `LUPIN_SPEC.md`: plan, pair kinds, recipes, v1 gate.
- Code: `build_pool.py`, `select_pool.py`, `pool_check.py`, `topic_check.py`, `lupin_pairs.py`, `lupin_select.py`,
  `dpo.py`, `track.py` (stages: poolfix / poolmore / poolup / tooltopup / newtop / lupinanswers / lupin / pertype / dial /
  crosscheck / codecut / wanda / wandaeval), `nofit_verify.py`, `wanda_prune.py`.
- Every score is also appended to `runs/bon/results.csv` on the project Drive (from 29 Oct night).
