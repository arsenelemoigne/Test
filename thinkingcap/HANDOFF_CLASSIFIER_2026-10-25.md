# ThinkingCap — handoff: stopping classifiers, stop training, and the 24–25 Oct 2026 runs

Written 25 Oct 2026 (~10:45 CET) for another analyst (human or AI) picking up this work with no prior context.
Everything here is taken from the repository (branch `mvt-fading`, folder `thinkingcap/`), its `EXPERIMENT_LOG.md`, and the
runtime logs read during the session. Numbers are quoted as they were printed; where something is an inference, it says so.

---

## 0. One-paragraph summary

The task is to make **Qwen3-0.6B think less** (fewer "thinking" tokens before `</think>`) without losing accuracy, measured on a
fixed four-benchmark evaluation (SVAMP math, MBPP+ code, IFEval instructions, BFCL function calling). The best model so far
(`rc1-rl-40`, DPO + RL) saves **53.9 %** of tokens at **−2.1** accuracy points on a fresh 240-question sample (goal: > 55 % at
better than −2; public competitor: 43.2 % / −0.85). A second line of work trains a **stopping classifier ("judge")** that reads
the model's hidden state at each paragraph end and predicts whether stopping there keeps the answer right, with the aim of later
teaching the model to emit `</think>` itself at those points ("stop-token training"). On **held-out practice questions** the judge
looked strong (AUC 0.86–0.905, simulated ~67 % thinking saved at ~0.5 pt). On **24–25 Oct it was tested for the first time on the
real benchmark questions (zbench)**: AUC fell to **0.74**; it transfers for **code and tools**, but on **math (SVAMP)** it stops far
too early because practice math (GSM-IC/GSM-Plus) settles early while SVAMP settles late. The model's own confidence ranks SVAMP
cuts better (0.88), but a closer look shows the comparison metric (pooled AUC) mixes "question difficulty" with "timing", so a
**timing-only metric** and **combined classifiers** were added and are queued (zbench2, zB4bench). Several runs were lost overnight
to infrastructure problems (Colab runtimes, Drive mounts, three code bugs), all fixed.

---

## 1. The task and how it is scored

- **Base model:** `Qwen/Qwen3-0.6B` (thinking mode). Rules (challenge repo `BottleCapAI/ThinkingCap-Test`, README "Rules"):
  start from Qwen3-0.6B; LoRA / full fine-tune allowed; **do not train on the evaluation benchmarks**; keep the provided
  evaluation settings (prompt, sampling, token cap). The README's suggested training sets: GSM8K, Skywork-OR1 (easy band),
  CommonsenseQA, ARC Easy. It does not forbid other data.
- **Evaluation:** 4 benchmarks, 2,759 questions in total (MBPP+ 378, BFCL simple_python 400 / multiple 200 / irrelevance,
  SVAMP, IFEval 541). Score = tokens saved (geometric mean over the 4 benchmarks, BFCL subsets pooled) and the accuracy change
  vs the original model. Sampling at Qwen's recommended settings (T 0.6, top-p 0.95, top-k 20).
- **Question sets used internally:**
  - `questions_40` = the **choosing set**: 240 benchmark questions (40 per dataset/subset) used historically to pick models.
    Also used as the **benchlot / zbench test** (measure only, never trained on).
  - `questions_fresh20261022` = the **fresh set**: another 240 benchmark questions, scored ×8 samples ("fresh ×8"); all model
    results below are on it unless stated. Never used for any choice involving the judge.
  - Full 2,759 × 8 evaluation: reserved for the very end (user rule).
- **Contamination policy:** every training question goes through `contamination.py`: dropped if ≥ 30 % of its 8-grams appear in
  one of the 2,759 evaluation questions (was 50 % before an MBPP incident). **No MBPP-family data anywhere** (user, 24–25 Oct):
  MBPP+ is built on MBPP, so `mbpp_train` (MBPP minus MBPP+) is excluded from all training and from the judges.

### Reference results (fresh 240 × 8)

| model | tokens saved | accuracy | SVAMP | MBPP+ | IFEval | BFCL |
|---|---|---|---|---|---|---|
| **rc1-rl-40** (best clean) | **53.9 %** | **−2.1** | 65 % / −4.1 | 59 % / −1.9 | 46 % / −0.6 | 41–45 % / −1.2.. |
| competitor "Lightning McQwen3" (his full eval ×8, no think-gate) | 43.2 % | −0.85 | 66 % / −3.8 | 39 % / −1.9 | 32 % / +1.4 | 25 % / +0.9 |
| stopB3 soup 25 (best "stop line" model) | 34.6 % | −0.4 | 36 % / −2.2 | 50 % / −0.3 | 28 % / +2.8 | 20–22 % / 0.0..−3.7 |

Earlier rounds (×4 samples, noisier) reached SVAMP 67–69 % at −0.6..+3.1 — i.e. **math thinking can be cut by about two thirds by
training**; this matters for interpreting the "30 % oracle" below.

---

## 2. The stopping-classifier pipeline (how the judges are built)

### 2.1 Labelled data ("lots") — `track.py callot` → `probe_curve.py` + `label_cuts.py`

1. **Draw questions** from training sources (never benchmark questions), remove earlier draws, contamination-check.
2. **Generate replies:** the model answers each question N times (8 for lots B/B2, 20 for B3/B4; vLLM, Qwen sampling).
3. **Pick replies to label:** per question, the longest right reply and (with `--include-wrong`) the longest finished wrong reply.
4. **Cuts:** each reply's thinking is split at paragraph ends (`"\n\n"`); up to 10 cuts per reply (evenly spaced, first always
   included) + the full thinking (natural stop).
5. **Forced answers:** at each cut, `</think>` is closed and the model answers **20 times** (`--samples 20`, answer cap 2,048
   tokens). The cut's label is its share of right answers (k/20), graded by the type-specific grader (math exact answer, code
   tests, IFEval checks, tool-call match).
6. Outputs per lot: `runs/bon/pairs-<lot>/cal/{cuts,verdicts,replies,confidence}.jsonl, summary.txt`.

**Label used by the judges (`probe_zoo.py`):** `safe` = 20/20 right; `unsafe` = ≤ 15/20; cuts with 16–19 are left out of
training/AUC. A soft target (k/20) is used by the "real success rate" probes and for calibration.

### 2.2 Lots built

| lot | model labelled | sources (drawn) | tries/question | kept questions | notes |
|---|---|---|---|---|---|
| A | rc1-rl-40 | ifeval_like 400, gsm_ic 200, gsm_plus 200, toolace 300, mc 100, kodcode 200 | 8 | 643 | |
| A2 | rc1-rl-40 | same mix, seed 49 | 8 | ~580 (A+A2 = 1,222) | |
| B | original Qwen | ifeval_like 400, gsm_ic 200, gsm_plus 200, toolace 300, mc 100, kodcode 200 | 8 | ~640 | mc = ARC-Easy + CommonsenseQA |
| B2 | original Qwen | same mix, seed 50 | 8 | ~640 (B+B2 = 1,279 dev) | |
| B3 | original Qwen | gsm_ic 350, gsm_plus 350, ifeval_like 500 | 20 | **455** (743 new; 268 dropped by contamination) | 792 replies (422 right), 7,436 cuts |
| B4 | original Qwen | kodcode 300 (wider slice), mbpp_train 200, toolace 400, hermes_tools 250 | 20 | 647 (15 dropped) | 921 replies (460 right), 7,651 cuts; mbpp_train dropped downstream |
| bench40 | original Qwen | **the 240 choosing benchmark questions** | 8 | 240 (304 "questions" incl. r/w replies) | **measure only**: 1,812 cuts |

Source → benchmark analogue: gsm_ic / gsm_plus → SVAMP; ifeval_like → IFEval; kodcode (and mbpp_train) → MBPP+;
toolace / hermes_tools → BFCL; arc_easy / commonsense_qa (mc) → none.

### 2.3 Features — `extract_feats.py` (one forward pass per cut, no generation)

- `points.npy [n, 28 layers, 2 places, 1024]`: every layer's hidden state at the last thinking token **before** `</think>` and at
  the `</think>` token itself (**after**).
- `window.npy [n, 3, 64, 1024]`: the 64 tokens just before `</think>` at layers 14, 21, 28.
- Stored on the runtime disk only (`/content/feats`, several GB), not on Drive.
- Own-confidence readings — `own_conf.py`: the model answers **once, greedy** from the cut; stored: geometric-mean token
  probability of that answer (`own_confidence`), least likely token (`own_min`), 5 least likely (`own_low5`), first 50 tokens
  (`own_first50`), minus mean entropy (`own_certainty`), and the answer text (for "answer stability" = same answer as previous cut).

### 2.4 Classifiers — `probe_zoo.py` (+ helpers in `probe_knows.py`)

Development = the training lots; **5 folds by question** give out-of-fold (OOF) dev scores; test scores come from a fit on all
dev data. Test = either a hash-held-out 30 % of the dev questions (practice test) or a separate lot (`--test-cal`, e.g. bench40).

| classifier | input | model |
|---|---|---|
| position | cut fraction, log fraction, data type one-hot, type × fraction | logistic regression |
| own confidence (5 variants), answer stability | the readings above, used as-is (no training) | — |
| confidence (uses the right answer) | teacher-forced prob. of the full reply's answer — reference only, not usable at test time | — |
| layer sweep | each layer × place | logistic regression (standardized, L2, LBFGS) → best = **layer 20, after `</think>`** |
| linear | best layer/place | logistic regression |
| linear, real success rate | best layer/place, soft k/20 target | logistic (calibrated) |
| linear, one per type | best layer/place | one logistic per data type |
| mlp | best layer/place | MLP 1024 → 256 → 64 → 1, ReLU, dropout 0.2, AdamW (wd 1e-3), 60 epochs, early stop on 15 % |
| mlp, all 28 layers × 2 places | 57,344-d concat | MLP → 512 → 128 → 1 |
| attention, last 64 tokens (layer 14 / 21 / 28 / all three) | window | LayerNorm → 256-d proj + learned position → 4-head attention pooling (learned query) → MLP → P(safe); BCE, AdamW 1e-3, ≤ 30 epochs, early stop |
| **new (25 Oct)** own confidence, learned | 5 own readings (log) + position | logistic |
| **new** linear + own confidence | layer-20 state + 5 own readings | logistic |
| **new** mlp + own confidence | layer-20 state + 5 own readings | MLP 256 → 64 |

**Reported:** AUC (dev OOF, test; per type and ALL); calibration on test (log loss, Brier, ECE vs k/20); pooled trade-off (one
threshold for all types, chosen on dev for a loss budget, applied to test); **stopping simulation** on test: per type, stop each
reply at its earliest cut whose score ≥ a threshold chosen on dev OOF for that type's allowance (gsm 0.5 pt, kodcode 1, others 2)
and at 2 pts for all; report tokens saved / real loss vs the natural stop / coverage. "oracle 20/20" = stop at the first cut whose
20 forced answers are all right (truncation bound under the strict label). **New (25 Oct): timing-only AUC** (pairs of cuts
compared only within the same reply — see §5).

Loss accounting was corrected twice on 24 Oct: (1) loss vs the natural stop, not vs "always right"; (2) the natural stop is the
full-thinking cut's own 20 forced answers and length (it had been the selected reply's single outcome, which biased results).

---

## 3. Classifier results, chronologically

AUC: 0.5 = chance, 1 = perfect ranking of safe above unsafe cuts.

| run | model read | trained on | test set | best TEST AUC (ALL) | own confidence | position | notes |
|---|---|---|---|---|---|---|---|
| zoo A | rc1-rl-40 | lot A (643 q, 70 %) | 279 held-out practice q | MLP **0.852** | 0.790 | 0.718 | per type MLP: math 0.91 (own 0.92–0.93), instr 0.87 (own 0.74), tools 0.88 (own 0.63), code 0.80 (own 0.72) |
| zoo A2 | rc1-rl-40 | lots A+A2 (1,222 q) | 540 held-out practice q | MLP all layers **0.905** | 0.792 | 0.719 | per type: math 0.91/0.88, code 0.92, instr 0.89, tools 0.69; ≤ 2 pt pooled: 31 % / 2.4 (own 10 % / 0.4) |
| zoo B | original Qwen | lot B | 281 held-out practice q | MLP 0.826–0.828 | 0.728 | 0.668 | math 0.92–0.94 vs own 0.87–0.88 |
| zoo B2 | original Qwen | lots B+B2 (1,279 q / 11,314 cuts) | 545 held-out practice q (4,907 cuts) | MLP **0.864** | 0.740 | 0.699 | per type MLP: math 0.93–0.94, instr 0.85, code 0.78, tools 0.69, MC 0.48–0.63. Pooled ≤ 2 pt: **MLP 69 % / 1.5**, linear per type 65 % / −0.7, own 46 % / 1.1, position 51 % / 2.9; ≤ 0.5 pt: **MLP 67 % / 0.5** — this is the "−67 % at ~1 %" the user remembers |
| **zbench** (25 Oct) | original Qwen | lots B+B2+B3 (2,616 q / 23,657 cuts) | **240 benchmark questions (bench40: 1,812 cuts)** | MLP all layers **0.738** (dev 0.885) | 0.686 (least-likely-token **0.717**) | **0.508** | see §4 |
| zoo B4 | original Qwen | B+B2+B3+B4, MBPP dropped | held-out practice | — | | | crashed 25 Oct (lot B3 labels missing on runtime 1); rerun queued |
| zB4bench | original Qwen | B+B2+B3+B4, MBPP dropped | bench40 | — | | | queued 25 Oct |
| zbench2 | original Qwen | B+B2+B3 | bench40 | — | | | queued 25 Oct, with timing-only AUC + combos |

Observed: doubling the data raised the best judge by +0.04–0.06 AUC each time (data-limited); classifier architecture matters
little (all within ~0.05).

Replay of the judge on held-out practice questions (simulation, 24 Oct): 67–69 % thinking saved at −0.5 / −1.5.

---

## 4. zbench in detail (judge on the real benchmark questions)

Columns: `gsm_plus` = SVAMP, `ifeval_like` = IFEval, `kodcode` = MBPP+, `toolace` = BFCL (labels inherited from the labelling kind).

### 4.1 AUC, dev (OOF, practice) vs TEST (benchmark)

| classifier | dev ALL | TEST SVAMP | TEST IFEval | TEST MBPP+ | TEST BFCL | TEST ALL |
|---|---|---|---|---|---|---|
| position | 0.722 | 0.401 | 0.500 | 0.508 | 0.507 | 0.508 |
| own confidence | 0.731 | **0.872** | 0.613 | 0.682 | 0.569 | 0.686 |
| own conf., least likely token | 0.728 | **0.879** | 0.637 | 0.696 | 0.619 | 0.717 |
| own conf., 5 least likely | 0.732 | 0.879 | 0.623 | 0.717 | 0.619 | 0.713 |
| answer stability | 0.660 | 0.764 | 0.567 | 0.634 | 0.579 | 0.634 |
| linear (layer 20 after) | 0.875 | 0.703 | 0.607 | 0.756 | 0.632 | 0.674 |
| linear, one per type | 0.866 | 0.668 | 0.585 | 0.749 | 0.694 | 0.688 |
| mlp | **0.893** | 0.740 | 0.553 | **0.770** | **0.772** | 0.736 |
| mlp, all layers | 0.885 | 0.752 | **0.711** | 0.756 | 0.698 | **0.738** |
| attention, layer 28 | 0.835 | 0.749 | 0.652 | 0.734 | 0.649 | 0.714 |

Dev per type for reference (mlp): arc_easy 0.32, csqa 0.55, gsm_ic 0.95, gsm_plus 0.95, ifeval_like 0.87, kodcode 0.75, toolace 0.77.

### 4.2 Calibration on TEST (soft target k/20): best = mlp all layers (log loss 0.82, Brier 0.217, ECE 0.14).

### 4.3 Pooled trade-off (one threshold, chosen on dev; TEST tokens saved / real loss)

| classifier | ≤ 0.5 pt budget | ≤ 2 pt | ≤ 5 pt |
|---|---|---|---|
| position | 10 % / 2.3 | 37 % / 6.5 | 59 % / 11.7 |
| own confidence | 22 % / −0.2 | 32 % / 1.9 | 42 % / 3.8 |
| own conf., least likely token | 11 % / 0.6 | 23 % / −0.4 | 33 % / 0.4 |
| linear | 51 % / 4.0 | 55 % / 5.1 | 55 % / 5.1 |
| mlp | 55 % / 3.3 | 55 % / 3.3 | 55 % / 3.3 |
| mlp, all layers | 64 % / **6.6** | 65 % / 7.8 | 65 % / 7.8 |
| attention, layer 21 | 56 % / 5.5 | 60 % / 6.2 | 63 % / 7.1 |

**Thresholds chosen on practice data do not transfer**: a "≤ 0.5 pt" setting loses 3–8 pts on the benchmark.

### 4.4 Stopping simulation per benchmark (each type's allowance; TEST saved / loss / coverage)

| | SVAMP | IFEval | MBPP+ | BFCL |
|---|---|---|---|---|
| oracle 20/20 (bound) | 30 % / −1.3 (gain) / 59 % | 19 % / −4.2 / 34 % | 41 % / −7.7 / 58 % | 39 % / −5.6 / 54 % |
| own confidence | 15 % / 3.8 / 49 % | 68 % / 1.0 / 92 % | 94 % / 1.0 / 100 % | 23 % / −1.4 / 38 % |
| own conf., least likely | 8 % / 1.9 / 33 % | 68 % / 1.0 / 92 % | 94 % / 1.0 / 100 % | 43 % / 0.8 / 57 % |
| linear | 42 % / 14.9 / 94 % | 43 % / −0.4 / 58 % | 92 % / 2.4 / 100 % | 31 % / 0.8 / 45 % |
| mlp | 19 % / 8.0 / 76 % | 43 % / 0.9 / 58 % | 93 % / 1.0 / 100 % | 22 % / −1.3 / 35 % |
| mlp, all layers | 63 % / **23.8** / 100 % | 58 % / 4.0 / 81 % | 92 % / 1.0 / 100 % | 44 % / 0.4 / 57 % |
| attention, layer 21 | 50 % / 21.3 / 84 % | 67 % / 1.1 / 92 % | 94 % / 0.8 / 100 % | 44 % / 2.4 / 64 % |

(Negative loss = accuracy gained. "Real loss" is vs the natural stop measured by the same 20 forced answers.)

### 4.5 Why math fails — the cut-success curves (share of right forced answers by share of thinking kept)

| set | 0–20 % | 20–40 % | 40–60 % | 60–80 % | 80–100 % | full | perfect stop: saved / gained |
|---|---|---|---|---|---|---|---|
| lot B3 gsm_ic (397 replies) | 0.25 | **0.46** | 0.47 | 0.49 | 0.52 | 0.52 | 66 % / +29.5 |
| lot B3 gsm_plus (184) | 0.21 | **0.47** | 0.47 | 0.46 | 0.50 | 0.47 | 71 % / +31.8 |
| lot B3 ifeval_like (211) | **0.77** | 0.73 | 0.75 | 0.73 | 0.66 | 0.73 | 61 % / +11.4 |
| lot B4 kodcode (466) | 0.46 | 0.50 | 0.46 | 0.47 | 0.50 | 0.52 | 79 % / +22.9 |
| lot B4 hermes_tools (301) | 0.27 | 0.44 | 0.41 | 0.29 | **0.18** | 0.49 | 72 % / +9.1 |
| lot B4 toolace (136) | 0.35 | 0.46 | 0.49 | 0.41 | 0.42 | 0.54 | 46 % / +6.3 |
| **bench SVAMP** (51) | 0.26 | **0.34** | 0.48 | 0.58 | 0.55 | **0.73** | 53 % / +8.9 |
| bench IFEval (53) | 0.56 | 0.50 | 0.49 | 0.51 | 0.45 | 0.61 | 55 % / +11.0 |
| bench MBPP+ (52) | 0.46 | 0.53 | 0.44 | 0.52 | 0.49 | 0.50 | 84 % / +17.0 |
| bench BFCL (148) | 0.70 | 0.70 | 0.72 | 0.77 | 0.62 | 0.75 | 51 % / +8.7 |

("perfect stop" here = best cut vs natural stop per reply, a looser bound than "oracle 20/20".)

**Interpretation:** practice math settles by 20–40 % of the thinking; SVAMP does not (0.34 → 0.73). The judge learned the practice
timing and stops SVAMP early; position-only scores 0.40 on SVAMP (anti-correlated). Code: flat curves, almost all thinking
removable. Tools: late forced answers are bad; the safe window is early. IFEval benchmark: early cuts cost a little.

---

## 5. Is the comparison "judge vs the model's own confidence" measuring the right thing? (25 Oct analysis)

- Both see the same cut and nothing from the future → the inputs are comparable. Own confidence needs one greedy answer
  generation per cut; the judge needs one forward pass.
- **Problem:** pooled AUC rewards two different skills: (1) *which replies/questions will be right at all* (question difficulty —
  safe cuts concentrate in easy questions) and (2) *when, inside a reply, the answer has settled* (what a stop needs). Own
  confidence is naturally strong at (1); its SVAMP 0.87–0.88 may mostly mean "the model knows which easy questions it knows".
  In the decision-relevant stopping simulation on SVAMP, own confidence only "wins" by stopping rarely (15 % saved / −3.8 pts;
  least likely token 8 % / −1.9), and nothing reaches the oracle (30 % / +1.3 gained).
- **Fixes implemented (`probe_zoo.py`, commit 36a4a86, toy-tested only):**
  1. **Timing-only AUC** — `timing_auc()`: only pairs (safe, unsafe) of cuts from the *same reply* are compared; reports the
     number of replies holding both. Toy check: rising score 1.0, difficulty-only score 0.5, falling score 0.0. Shown for every
     classifier, dev and test, per type; position uses the raw cut fraction.
  2. **Own confidence as a learned classifier** (5 readings + position, same fit as the probes) — a fair, equally-trained baseline.
  3. **linear + own confidence** and **mlp + own confidence** — do hidden state and self-confidence combine?
  4. Decision metric = per-type stopping simulation; thresholds should come from **benchmark-like** practice data, never from
     benchmark questions (fitting thresholds on benchmark questions = training on the eval).
- Caveats on zbench: small (240 questions; SVAMP only 51 labelled replies); bench40 replies were sampled 8×, labels still 20 forced
  answers per cut; the "safe = 20/20" bar is strict.

Current reading: the judge knows **more than own confidence on tools (0.77 vs 0.62) and somewhat on code (0.77 vs 0.72)**; on
real math it is worse; overall about a tie (0.74 vs 0.72). Not yet measured: timing-only skill (zbench2 will).

---

## 6. Stop training ("stop line") — training the model to stop at labelled safe cuts (no judge involved)

`track.py targeted` → `targeted_pairs.py` → `dpo.py` (LoRA DPO, β 0.1, lr 2e-5, + chosen NLL), 3 seeds, adapters merged as
uniform "soups" at checkpoints, each soup scored on fresh × 8. Pairs ("shorten" kind): chosen = the **earliest safe cut** of a reply
+ a right answer from there; rejected = the full reply. **These pairs use the forced-answer labels directly, not the judge.**
`--tg-rule question` (20-try lots): a cut is safe if it beats the question's own success rate q + margin on half its forced
answers and reaches q on the other half (split-half check); strict 95 %-style rule on 8-try lots.

| run | pairs | soup → fresh × 8 (saved / acc) | per benchmark notes |
|---|---|---|---|
| stopB (lot B, incl. code; 554 pairs) | 554 | 10: +11.9 / +0.9; 20: +31.1 / −2.3; **30: +54.2 / −3.6** | soup 30: MBPP+ 82 % / −8.1 (code over-cut) |
| stopB2 (lot B, no code; 327 pairs; one epoch = 21 steps) | 327 | 20: **+34.6 / −2.1** | code 37 % shorter at 0.0 with no code pairs |
| stopB3 (lots B+B2, incl. code; 1,058 pairs) | 1,058 | 15: +18.3 / −1.0; 20: +27.1 / −2.0; **25: +34.6 / −0.4**; 30: +54.8 / −3.2; 40: +69.1 / −5.7 | soup 25: SVAMP 36 % / −2.2, MBPP+ 50 % / −0.3, IFEval 28 % / +2.8; soups 30/40: MBPP+ 85–94 % / −8 |
| stopB4 (lots B..B4, question rule, all types except MBPP) | — | queued | |
| **stopB5** (lots B..B4, **code + tools only**: kodcode, toolace, hermes_tools) | — | queued (soups 10–30) | motivated by §4: code thinking removable, tool stops safe early, math/instructions excluded |

**wronglen** (thinking length on fresh × 8 by answer outcome; "never solved" = 0/8 for the original, 14 % of answers, no pair ever
came from them). Original mean thinking tokens, all benchmarks: right 359 / wrong 1,339 / never solved 1,493 (MBPP+: 998 / 2,955 /
3,255). Shortening: rc1-rl-40 right −58 % / wrong −61 % / never −59 %; stopB2 s20 −36 / −43 / −31; stopB3 s25 −37 / −49 / −46;
stopB3 s30 −51 / −77 / −80. Training spills over onto answers it never saw, and shortens wrong answers most (free savings);
exception: SVAMP never-solved barely shrinks with stopB3 s25 (−3 %).

---

## 7. Other queued / built work

- **drpo1** (runtime 2): DRPO RL (arXiv 2510.04474, `rl_train.py --objective drpo`) from rc1-rl-40: right replies always pushed up
  with a softmax(r_l/λ) weight normalized among right replies only; wrong replies pushed down; r_l = 1 − len / group's longest;
  λ 0.2; 48 questions × 8 replies × 60 steps from a fresh MBPP-free mixed pool (gsm_ic 17 %, gsm_plus 17 %, ifeval_like 24 %,
  toolace 20 %, hermes 10 %, KodCode 16 %); checkpoints 40 and 60 on fresh × 8. Success bar (fixed beforehand): ≥ ~59 % saved at
  ≥ −2.5, or > 55 % at better than −2. Does not use the judge. Failed once on a missing package (fixed).
- **teach1** (runtime 2, written by another session): "judge as teacher" (`teach_stop.py`): judge fitted on lots B..B4 scores every
  paragraph end of fresh answers; a LoRA trains the model's own p(`</think>`) there toward the target (variants fixed:0.9, rel:0.05).
  Failed once on a parsing bug (fixed). Given §4, its math behaviour should be watched.
- **Stop-token training** (the roadmap's main step, ESTAR-like but without a test-time classifier veto): not built yet.

---

## 8. Infrastructure and what went wrong (24–25 Oct)

- Two Colab Pro runtimes (L4 GPUs) on the user's **personal** Google account `arsene@lemoigne.eu`; notebooks
  `full_eval_colab.ipynb` (runtime 1, queue `plan_line_b.txt`) and `second_colab.ipynb` (runtime 2, queue `plan_line_a.txt`), opened
  from GitHub. Each runs `track.py plan`, which executes the queue file line by line, re-reading it from the branch after each stage
  (push a line to queue a stage); finished stages are skipped; results rsynced to `MyDrive/thinkingcap-mvtf` every 5 min; the log
  copy lands in `thinkingcap-mvtf/code[-second]/track_log.txt`. The last notebook cell releases the runtime when the queue ends.
- Reading progress: download the Drive log copies in the user's Chrome (Drive ids in memory note `thinkingcap-colab-log-access`),
  or read the open Colab page; the Colab MCP connector failed (Safari blocks the localhost link; it attaches to the work account).
- Incidents: runtime 2 ended mid-labelling at ~20:10 (cause unknown); relaunch attempts failed on the Drive consent pop-up (needs a
  real user click) and a page stuck on "Resuming execution"; one relaunch spent hours in a FULL Drive restore (fixed: light restore
  for benchlot/drpo/teach); around 02:40 CEST the assistant clicked "Disconnect and delete runtime" on a half-connected page and
  runtime 1 stopped in the same window (cause not proven; rule now: never do that); runtime 2 then ended ~03:13 CEST with no error;
  after the user relaunched both at ~02:00 CET, lots B3/B4, benchlot and zbench completed; zoo B4, stopB4, drpo1, teach1 then failed
  on three bugs and both runtimes released themselves. Fixes (commit e7af185): `ensure_lots()` restores a lot made on the other
  runtime; setup installs `langdetect`; `mixed_pool` parses rows instead of a substring test. Compute units were not the cause
  (78.6 left). Note: DST ended that night (03:00 CEST → 02:00 CET), so log times mix time zones.

---

## 9. State at handoff (25 Oct ~10:45 CET)

Both runtimes relaunched by the user, scripts at commit 36a4a86+, queues:
- **Runtime 1:** zoo B4 (practice test) → **zB4bench** (bench40 test) → **stopB5** (code + tools stop) → stopB4.
- **Runtime 2:** **zbench2** (bench40, timing-only AUC + combos) → drpo1 → teach1.

---

## 10. Open questions worth analysing

1. **Timing vs difficulty:** on bench40, does the judge beat own confidence on the timing-only AUC, per type? (zbench2, zB4bench.)
2. **Do the combos help** (mlp + own confidence) on the benchmark, especially SVAMP?
3. **Does lot B4** (code/tools, 20 tries) lift BFCL/MBPP+ on the benchmark (zB4bench vs zbench2)?
4. **Math data:** practice math (GSM-IC distractors, GSM-Plus perturbations) is not SVAMP-like. Candidates for SVAMP-*like*
   training problems must be contamination-checked carefully (SVAMP was built by varying ASDiv-A problems, so ASDiv/MAWPS-derived
   sets are risky). Alternatively use own confidence for math stops, or no learned stop on math at all.
5. **Thresholds:** choose per type on benchmark-like practice data; verify on bench40 (measure only).
6. **Ceiling:** truncation bounds (oracle 20/20 30 % on SVAMP; best-cut 53 %) are far below what training already achieves on
   SVAMP (65–69 %): the judge/stop-token route may be most valuable for code and tools, while math savings come from
   training (DPO/RL).
7. **Label strictness:** safe = 20/20 vs the question-rate rule; soft targets (k/20) and their calibration.

---

## 11. Where things are

- Code: `thinkingcap/track.py` (stages: callot, benchlot, probezoo, targeted, wronglen, drpo, teach, plan), `label_cuts.py`,
  `extract_feats.py`, `own_conf.py`, `probe_zoo.py`, `probe_knows.py`, `targeted_pairs.py`, `dpo.py`, `rl_train.py`,
  `teach_stop.py`, `contamination.py`, `prepare_types.py`.
- Queues: `thinkingcap/plan_line_b.txt` (runtime 1), `thinkingcap/plan_line_a.txt` (runtime 2).
- Full chronological log with every number: `thinkingcap/EXPERIMENT_LOG.md` (entries dated 24–25 Oct at the end).
- Earlier reports: `FINAL_REPORT.md`, `FINDINGS_ALL.md`, `THINKINGCAP_FULL_DOSSIER.txt`, `REPORT_MVT_FADING.md`.
- Results on Drive (personal account): `MyDrive/thinkingcap-mvtf/runs/bon/zoo-<name>/summary.txt` (full zoo reports),
  `runs/bon/pairs-<lot>/cal/` (labelled cuts), `runs/mvtf/f8-*` (fresh × 8 scores).
