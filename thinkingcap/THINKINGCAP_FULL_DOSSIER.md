# ThinkingCap / Qwen3-0.6B shorter thinking — complete experiment dossier (state 30 Oct 2026, ~14:00 CET)

Plain record for external analysis, from the first experiment to now: context, methods, training data, evaluation, every
measured result (including failed and broken runs), bugs, and the decisions taken. No ranking or interpretation is intended;
tables copy the lab notebook's numbers. Sources: `EXPERIMENT_LOG.md` (branch mvt-fading, 3,458 lines), the per-type clone's
log (branch mvt-pertype), the specs and handoff notes in the same folder. Sections are in time order; Part F is the LUPIN
round-1 work (27–30 Oct). Values not in the sources are "—". Runs still in progress are marked pending.

## Contents

0. Task, evaluation, metrics, question sets, noise, external reference points
1. Part A — Track 1 and Tracks 2–2e (before 20 Oct to 22 Oct)
2. Part B — Track 3 per-type clone (20–21 Oct, ran in parallel with Track 2b–2e)
3. Part C — Tracks 2f and 2g, first half (22–25 Oct)
4. Part D — Track 2g, second half, and the stop-classifier work (25–26 Oct)
5. Part E — 26–30 Oct: runs outside the LUPIN pool, analyses, literature, decisions
6. Part F — LUPIN round 1: pools dp1–dp15, pairs, screens, confirms, cross-checks, Wanda pruning

---

## 0. Task and evaluation

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

---

## Part A — Track 1 and Tracks 2–2e (before 20 Oct to 22 Oct)

Source: `Test-mvt-fading/thinkingcap/EXPERIMENT_LOG.md`, lines 1-1002. All numbers copied from the log. "Tokens saved" and "accuracy change" are against the raw Qwen3-0.6B under the same evaluation setup; accuracy change is in percentage points (pp). Where the log reports an "average" saving it is reproduced as given (the log does not restate the aggregation formula in this range). "—" = not in the log.

Dates: this part of the log has few explicit dates. Explicit ones: runtime A lost "about 11:07 Paris (2026-10-21)" (Track 2d end); `questions_fresh20261022` seed; Track 2f (outside this section) "queued 22 Oct 2026". Elsewhere the log uses relative references ("yesterday", "last night", "night-1"). Track 1 and Tracks 2-2b are therefore before 2026-10-21; Track 2e is 2026-10-21 to 2026-10-22.

Project goal stated at top of log: cut tokens while losing at most 3 accuracy points. The log states 95% accuracy ranges on the 240-question subset are about +/-4 pp overall, and single-benchmark cells swing about +/-8 to +/-10 at 40 questions.

---

### Common evaluation setup

| item | value |
|---|---|
| evaluator | official `evaluate.py` / `compare.py` |
| question set (default) | `questions_40`: 40 questions per benchmark part, 240 total |
| samples per question | 2 by default; "4-sample" / "8-sample" rechecks noted per row |
| hardware | L4 GPU, bf16 (Track 1 Round 1 on T4) |
| later held-out sets | `questions_fresh20261022` (40 per part, seed 20261022, disjoint from questions_40, 8 samples); questions 41-80 of every part (scored by the separate mvt-pertype session, 8 samples) |

---

### Track 1: SFT on pruned traces, then RL (date: —; GPU: T4 for Round 1, L4 otherwise)

#### Reference: raw model (Track 1 runtime, 2 samples)

| benchmark | accuracy | mean tokens |
|---|---|---|
| SVAMP | 81.3% | 778 |
| MBPP+ | 57.5% | 1,699 |
| IFEval | 52.5% | 710 |
| BFCL irrelevance / multiple / simple | 85.0 / 76.3 / 82.5% | 205 / 223 / 190 |

#### Methods

- Rounds 1-2 (SFT): the model's own traces on 450 fixed training questions (GSM8K, ARC-Easy, CSQA, 150 each), 2 samples, 2,048-token cap, pruned; SFT 2 epochs. Round 2 adds "margin + balanced" pruning.
- RL loop 1: RL with a TokenProbe-KL anchor plus a ShorterBetter length reward, from the raw model; 12 steps x 32 fresh questions x 8 samples (384 questions, each seen once), pool = full training split (~40k questions, GSM8K/ARC-Easy/CSQA).
- RL loop 2: continuation of loop 1 settings; stopped at step 4 (lengths not falling).
- RL loop 3: continued from the loop 1 adapter with stronger length pressure; 20 x 32 x 8 (640 fresh questions, seed 1; GSM8K, Skywork, ARC-Easy, CSQA, balanced). Log shows only 2 steps, status "running".

| hyperparameter | loop 1 | loop 3 |
|---|---|---|
| start from | raw model | loop 1 adapter |
| LoRA rank / lr | 16 / 2e-5 | 16 / 2e-5 |
| ShorterBetter alpha | 0.5 | 1.0 |
| TokenProbe KL beta | 5e-3 | 5e-3 |
| lam_decay (push on redundant tokens) | 0.2 | 0.5 |
| token budget = core length x scale | 3.0 -> 1.5 | 2.0 -> 1.0 |
| core windows | 10 windows of 10 tokens | same |
| steps x questions x samples | 12 x 32 x 8 | 20 x 32 x 8 |
| token cap | 2,048 | 2,048 |

#### Results (240 q x 2 samples)

| run | method | tokens saved | accuracy change |
|---|---|---|---|
| Round 1 (T4) | SFT on pruned traces | +39.1% | -6.3 |
| Round 2 | SFT on pruned traces, margin + balanced | +59.6% | -8.0 |
| RL loop 1 | RL, TokenProbe-KL + ShorterBetter, alpha 0.5 | +5.7% | -1.2 |
| RL loop 2 | same, continued | stopped at step 4 | — |
| RL loop 3 | from loop 1, stronger pressure | running (no eval in log) | — |

Per benchmark (tokens saved / accuracy change):

| benchmark | RL loop 1 | Round 2 (SFT) |
|---|---|---|
| BFCL irrelevance | +4.3% / -5.0 | BFCL +32-39% / -3.8 to -10.0 (range over 3 parts) |
| BFCL multiple | +12.2% / +8.8 | (see above) |
| BFCL simple | +9.0% / +1.3 | (see above) |
| IFEval | +16.1% / -1.3 | +40.3% / +3.8 |
| MBPP+ | -18.1% / -3.8 | +86.4% / -6.3 |
| SVAMP | +12.6% / -1.3 | +48.8% / -23.8 |

Per-step training numbers (on that step's 256 training answers):

| RL loop 1 step | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| accuracy % | 41 | 50 | 49 | 51 | 59 | 58 | 54 | 61 | 64 | 71 | 63 | 61 |
| mean tokens | 957 | 932 | 906 | 722 | 645 | 686 | 817 | 776 | 939 | 578 | 593 | 535 |

RL loop 2 (4 steps): accuracy 57, 49, 52, 53%; tokens 946, 832, 847, 866.
RL loop 3 (2 steps): accuracy 55, 49%; tokens 977, 830.

---

### Track 2: MVT cut, fading worked examples, marginal token analysis (branch `mvt-fading`; own Colab L4 runtime, bf16; date: —)

All runs via `track.py` from `mvt_fading_colab.ipynb`.

#### Reference rerun of raw model on this runtime (2 samples)

| benchmark | accuracy | mean tokens |
|---|---|---|
| SVAMP | 82.5% | 821 |
| MBPP+ | 57.5% | 1,699 |
| IFEval | 56.3% | 833 |
| BFCL irrelevance / multiple / simple | 83.8 / 76.3 / 82.5% | 202 / 223 / 190 |

Training data for all Track 2 runs (`track.py rollouts`): same slice as Track 1 SFT, 450 questions (GSM8K, ARC-Easy, CSQA, 150 each), 2 samples, 2,048-token cap, eval sampling settings.

#### Experiment 1: Marginal Value Theorem (MVT) cut, one-shot SFT

Method: each correct rollout is cut where an MVT leaving rule fires on p(gold answer) as a function of thinking tokens, with "travel cost" c; cut traces are checked by letting the model answer naturally (`--check-samples 4`). SFT LoRA r16, lr 1e-4, 2 epochs, `--max-per-source 70` (205 rows).
- mvt256: 2-step look-ahead rule, c=256. Bug in rule: on GSM8K median p(gold) at the cut is 0.00 for every c (32 to 2,048) because p(gold) is step-shaped (near 0 until the computation ends); rule left immediately. A c=64 run with this rule was stopped after 20 questions.
- Fix (commit `d96087f`): leave only when no later point pays better per token than g(t)/(t+c). Runs named `mvth<c>`.

| run | targets | pruned / truncated / original | mean completion | per source | failed natural-answer check | settle vs MVT median thinking kept | MVT earlier / later / same |
|---|---|---|---|---|---|---|---|
| mvt256 | 302 | 236 / 5 / 61 | 576 -> 245 (-57%) | ARC -67%, CSQA -65%, GSM8K -43% | 112 of 512 | 214 / 0 | 52% / 0% / 48% |
| mvth1024 | 302 | 285 / 8 / 9 | 576 -> 129 (-78%) | ARC -78%, CSQA -78%, GSM8K -77% (1,024 -> 239) | 17 of 512 | 214 / 51 | 34% / 4% / 63% |

| benchmark (2 samples) | mvt256 saved / acc | mvth1024 saved / acc |
|---|---|---|
| BFCL irrelevance | +29% / +2.5 | -1% / -5.0 |
| BFCL multiple | +29% / -7.5 | +27% / -2.5 |
| BFCL simple | +25% / -1.3 | +24% / +2.5 |
| IFEval | +67% / +6.3 | +59% / +3.7 |
| MBPP+ | +85% / -8.8 | +64% / -8.7 |
| SVAMP | +39% / -21.3 | +54% / -23.7 |
| average | +61.6% [52.9, 70.3] / -6.5 [-10.7, -2.1] | +51.2% / -7.6 [-11.8, -3.5] |

Log observation: all one-cut SFT variants so far (settle, settle + margin, MVT 256, MVT 1024) land between -6 and -8 pp with SVAMP -21 to -24.

#### Experiment 2: fading worked examples (paragraphs + words, LLMLingua-2 word scores)

Method (`fade_traces.py`): one target per question (shortest correct rollout, 302 questions). Round k deletes lowest-gain paragraphs until k x 10% of thinking tokens are gone, then keeps the top (1 - k x 10%) of words in remaining paragraphs, scored by `microsoft/llmlingua-2-xlm-roberta-large-meetingbank`; words with digits/math symbols always kept. Current adapter answers twice from each shortened trace; on failure retry with round k-1's word share, then fall back to round k-1's trace. SFT 1 epoch per round from previous round's adapter, lr 1e-4, r16, `--max-per-source 100`.

| round | targets | mean completion | per source | SFT rows | note |
|---|---|---|---|---|---|
| 1 (10% paras, 90% words) | 277 faded, 13 fewer words cut, 12 original | 518 -> 401 (-23%) | ARC -28%, CSQA -27%, GSM8K -15% | 265 | final loss ~0.9 |
| 2 (20% paras, 80% words), from r1 | 292 faded, 6 fewer words, 4 kept r1 | 518 -> 324 (-38%) | ARC -42%, CSQA -40%, GSM8K -32% | — | stop rule ended it |

| benchmark | r1, 2 samples | r2, 2 samples | r1, 4-sample recheck |
|---|---|---|---|
| BFCL irrelevance | +5% / -1.3 | -0% / -7.5 | +9% / -4.4 |
| BFCL multiple | +9% / +3.8 | +2% / -1.2 | +3% / -5.6 |
| BFCL simple | -12% / -2.5 | +22% / -8.7 | +6% / -1.9 |
| IFEval | +27% / +1.2 | +26% / -1.2 | +17% / +2.5 |
| MBPP+ | -39% (longer) / -5.0 | -19% (longer) / -6.2 | -29% (longer) / -7.5 |
| SVAMP | +17% / -6.2 | +33% / -3.7 | +16% / -3.7 (12 lost, 6 gained; all wrong numbers) |
| average | +4.6% / -2.5 | +14.3% / -4.3 | +4.1% / -3.2 |

Log observation: training traces -38% at r2 but model answers -14%; MBPP+ answers longer in both rounds.

#### Experiment 3: marginal token analysis (words scored by own model)

Method: same fading; word scorer changed (`--word-scorer marginal`): each word (except digits/math) deleted individually, log p(gold) re-scored by teacher forcing; score = log-prob lost. Round k keeps top (1 - k x 10%) words. Scoring ~85 min for 302 traces on the L4 (GSM8K ~600 words/trace); later rounds reuse cache. Round 2 run past the stop rule with `--ignore-limit`.

| round | targets | mean completion | per source | SFT rows |
|---|---|---|---|---|
| 1 | 278 faded, 11 fewer words, 13 original | 518 -> 397 (-23%) | ARC -30%, CSQA -28%, GSM8K -15% | 265 |
| 2 | 291 faded, 6 fewer words, 5 kept r1 | 518 -> 324 (-37%) | — | — |

| benchmark | r1, 2 samples | r2, 2 samples |
|---|---|---|
| BFCL irrelevance | -12% / -3.7 | -6% / +1.3 |
| BFCL multiple | +11% / -1.2 | +6% / -7.5 |
| BFCL simple | +12% / -1.2 | +4% / -8.8 |
| IFEval | +25% / -2.5 | +14% / -7.5 |
| MBPP+ | -14% (longer) / -12.5 | -26% (longer) / -18.8 |
| SVAMP | +28% / -3.7 (7 lost, 4 gained; wrong numbers) | +2% / -3.8 |
| average | +12.1% / -5.2 | -1.3% (longer) / -8.8 |

4-sample rechecks: r1 +5.3% [-2.1, 13.2] / -2.0 [-5.1, +1.0]; r2 +1.8% / -6.2 [-9.4, -3.1] (BFCL multiple -19.4).
Log note: at 2 samples Exp 3 r1 (-5.2) scored below Exp 2 r1 (-2.5); at 4 samples the order reversed (-2.0 vs -3.2).

#### Experiment 1b: MVT cut on ARC/CSQA only, GSM8K traces kept whole

Method: `prune_traces.py --full-sources gsm8k`: GSM8K keeps each question's shortest correct rollout uncut; ARC/CSQA get MVT cut. SFT as Exp 1 (`--max-per-source 70`: 70 ARC + 70 CSQA + 65 GSM8K = 205 rows). Variants: c=256; `--min-steps 2` (MC targets keep at least 2 paragraphs); `--repeat gsm8k:2` (GSM8K rows twice, 270 rows); cap35.

Target lengths:

| run | ARC | CSQA | GSM8K |
|---|---|---|---|
| mvth1024-full-gsm8k | 445 -> 91 (-80%) | 464 -> 97 (-79%) | whole (984) |
| mvth256-full-gsm8k | 445 -> 102 (-77%) | 464 -> 96 (-79%) | whole |
| mvth1024-full-gsm8k-min2 | 445 -> 138 (-69%) | 464 -> 134 (-71%) | whole |
| mvth1024-full-gsm8k-xgsm8k2 | as mvth1024-full-gsm8k | same | whole, x2 |

Per benchmark, 2 samples (saved / acc):

| benchmark | mvth1024-full-gsm8k | mvth256-full-gsm8k | -min2 | -xgsm8k2 | mvth256-...-xgsm8k2 |
|---|---|---|---|---|---|
| BFCL irrelevance | +8% / +2.5 | +26% / -1.3 | +11% / +0.0 | +21% / +3.7 | +19% / +1.2 |
| BFCL multiple | +12% / +8.8 | +24% / +0.0 | +10% / +1.3 | +10% / +5.0 | +23% / +3.8 |
| BFCL simple | +21% / +1.3 | +21% / +1.3 | +14% / +2.5 | +17% / -2.5 | +22% / +2.5 |
| IFEval | +53% / +3.7 | +53% / +3.7 | +31% / +3.7 | +58% / +2.5 | +43% / +2.5 |
| MBPP+ | +76% / -5.0 | +85% / -6.2 | -9% (longer) / -6.2 | +64% / -2.5 | +73% / -10.0 |
| SVAMP | +22% / -8.7 | +15% / -6.2 | +14% / -10.0 | +3% / -6.2 | +13% / -5.0 |
| average | +47.6% / -1.5 | +53.5% / -2.2 | +13.1% / -2.8 | +40.7% / -1.0 | +43.0% / -2.5 |

SVAMP failure notes from the log: mvth1024-full-gsm8k 10 lost / 3 gained (5 wrong numbers after one short paragraph, 3 hit token cap, 1 trailing content "Final answer: 4 + 22 + 16 = 42"); mvth256-full-gsm8k 9 lost / 4 gained (2 trailing content); xgsm8k2 8 lost / 3 gained (wrong reasoning on tricky wording, 2 truncated). Lost answers typically restate the question in one paragraph then answer.

cap35 run (design error): intended to cut MC rows to 35/source, but `--max-per-source` also capped GSM8K, halving data to 105 rows. Result +63.4% / -11.5, SVAMP -35.0 (30 lost: one restating paragraph then a guess).

4-sample and 8-sample rechecks:

| run | samples | tokens saved | accuracy change [95% CI] | per-benchmark notes |
|---|---|---|---|---|
| mvth1024-full-gsm8k | 4 | +43.8% [36.5, 51.2] | -2.1 [-5.7, +1.2] | BFCL +27/+12/+15% / -1.2/-0.6/+0.6; IFEval +39% / +6.9 [+1.3, +12.5]; MBPP+ +77% / -10.0 [-19.4, -0.6]; SVAMP +12% / -5.0 [-13.1, +1.9] (17 lost, 9 gained) |
| mvth256-full-gsm8k | 4 | +49.7% [42.7, 57.4] | -2.7 [-6.2, +0.7] | IFEval +8.8, MBPP+ -10.0, SVAMP -8.8 |
| mvth1024-full-gsm8k-xgsm8k2 | 4 | +39.1% [31.5, 47.5] | -1.0 [-4.2, +2.1] | SVAMP -3.1, MBPP+ -8.1, IFEval +5.6 |
| mvth256-full-gsm8k-xgsm8k2 | 4 | +42.0% [33.7, 51.1] | -2.1 [-5.8, +1.4] | — |
| mvth1024-full-gsm8k-xgsm8k2 | 8 | +37.1% [32.1, 42.2] | -0.6 [-3.05, +1.74] | BFCL +21/+17/+17% / +1.6/-1.3/-0.3; IFEval +37% / +4.7 [-0.9, +10.0]; MBPP+ +65% / -2.2 [-8.4, +4.1]; SVAMP +13.5% / -5.0 [-10.3, -0.3] (34 lost, 18 gained; mostly wrong numbers) |

#### Combinations (`track.py mix`), 2 samples

| run | data | tokens saved | accuracy change | notes |
|---|---|---|---|---|
| mix-mvt1024-marginal1-x2 | ARC/CSQA MVT-cut rows + GSM8K from Exp 3 r1 (984 -> 835 tokens), GSM8K x2 | +34.6% | -3.1 | BFCL +24/+22/+24% / -3.7/+7.5/+0.0; IFEval +35% / +3.7; MBPP+ +36% / -7.5; SVAMP +42% / -10.0 (13 lost; reasoning on lost rows 827 -> 512 tokens) |
| mix-marginal1-mathwhole-x2 | ARC/CSQA from Exp 3 r1, GSM8K whole x2 | -2.6% (longer) | -1.5 | — |
| mix-marginal2-mathwhole-x2 | ARC/CSQA from Exp 3 r2, GSM8K whole x2 | -7.2% (longer) | -5.6 | SVAMP lost rows hit cap: 1,401 -> 2,529 reasoning tokens |

#### Track 2 summary table (as logged)

| run | method | samples | tokens saved | accuracy change |
|---|---|---|---|---|
| mvt256 | MVT cut, 2-step look-ahead | 2 | +61.6% | -6.5 |
| mvth1024 | MVT cut, full look-ahead | 2 | +51.2% | -7.6 |
| fade-gain-words r1 | Exp 2 round 1 | 4 | +4.1% | -3.2 |
| fade-gain-words r2 | Exp 2 round 2 | 2 | +14.3% | -4.3 |
| fade-gain-marginal r1 | Exp 3 round 1 | 4 | +5.3% | -2.0 |
| fade-gain-marginal r2 | Exp 3 round 2 | 4 | +1.8% | -6.2 |
| mvth1024-full-gsm8k | 1b | 4 | +43.8% | -2.1 |
| mvth256-full-gsm8k | 1b, c=256 | 4 | +49.7% | -2.7 |
| mvth1024-full-gsm8k-xgsm8k2 | 1b, GSM8K x2 | 8 | +37.1% | -0.6 |
| mvth256-full-gsm8k-xgsm8k2 | 1b, c=256, GSM8K x2 | 4 | +42.0% | -2.1 |
| mvth1024-full-gsm8k-min2 | 1b, 2 paragraphs min | 2 | +13.1% | -2.8 |
| mvth1024-full-gsm8k-cap35 | 1b, 35 rows/source (bug) | 2 | +63.4% | -11.5 |
| mix-mvt1024-marginal1-x2 | 1b + GSM8K compacted by Exp 3 | 2 | +34.6% | -3.1 |
| mix-marginal1-mathwhole-x2 | Exp 3 r1 on ARC/CSQA, GSM8K whole | 2 | -2.6% | -1.5 |
| mix-marginal2-mathwhole-x2 | Exp 3 r2 on ARC/CSQA, GSM8K whole | 2 | -7.2% | -5.6 |

---

### Track 2b: per-type method grid (one Colab L4 runtime; date: —)

Data (`prepare_types.py`, `typed.py`), graded by the evaluation's own checkers; 250 questions per kind, 2 replies each, 2,048-token cap. For code/tool/instructions the probe reads p(own correct answer | thinking).

| kind | source | replies | correct | capped | mean tokens |
|---|---|---|---|---|---|
| math | GSM8K | 500 | 34% | 39% | 1,465 |
| code | MBPP minus every MBPP+ task (mbpp_train) | 500 | 37% | 36% | 1,375 |
| tool | Hermes single tool calls + 25% "no tool fits" pairs | 500 | 52% | 2% | 476 |
| instructions | argilla IFEval-like | 500 | 73% | 3% | 433 |

Method: each grid cell trains on one kind only (SFT, LoRA r16, lr 1e-4, 2 epochs) and is scored on all benchmarks at 4 samples vs a 4-sample base (`track.py grid`, `gridtable`). Methods per kind: whole (own replies uncut), MVT cut by paragraphs (c=1024), MVT cut by 32-token batches, paragraph leave-one-out marginal analysis (with drop_flat group removal), fading by paragraphs only (20% per round, 3 rounds, each from the previous adapter). Later found (Track 2c): the MBPP training tasks included near copies of MBPP+ tasks under other ids (no contamination check in 2b).

Pruning stats noted in the log:

| run | checked | passed | target tokens |
|---|---|---|---|
| code-mvt1024 | 215 | 154 | 974 -> 405 (-58%); MVT cut at 0 thinking tokens for most code traces |
| math-mvt1024 | 326 | 284 | 979 -> 262 (-73%) |
| text-mvt1024 | 391 | 328 | 362 -> 251 (-31%); median thinking kept settle 73, MVT 41 |
| tool-mvt1024 | 278 | 258 | 291 -> 163 (-44%) |
| code-mvt1024-tokens32 | 225 | 162 | — |
| code-marginal / math-marginal | — | — | 974 -> 96 / 979 -> 287 |
| math-fade r1 / r2 / r3 | 79 of 122 / 93 of 122 / — | | 912 -> 769 / 622 / 376 |
| code-fade r2 | 113 of 113 | | 875 -> 511 |

Failure/bug: tool-mvt1024 first run crashed (vLLM out of memory capturing CUDA graphs next to the scoring model); fixed with `torch.cuda.empty_cache()` before the check and `enforce_eager`, requeued. Marginal by 32-token batches dropped from the plan. Colab page lost Google sign-in and output stream at ~13:50; runtime kept running, results read from `/content/track_log.txt`.

Grid table (`track.py gridtable`; tokens saved / accuracy change, 4 samples; `>` = the kind's own benchmark):

| run | SVAMP | MBPP+ | BFCL | IFEval | average |
|---|---|---|---|---|---|
| code-fade-r1 | +13% / +3.1 | >+20% / -10.6 | +9% / -2.3 | -0% / -1.2 | +11% / -2.8 |
| code-fade-r2 | +27% / +1.2 | >+33% / -3.1 | +17% / -1.7 | +18% / -5.0 | +24% / -2.1 |
| code-fade-r3 | +44% / -1.3 | >+14% / -13.1 | +16% / -2.5 | +5% / -6.9 | +21% / -5.9 |
| code-marginal | +90% / -41.2 | >+90% / -25.6 | +34% / -3.1 | +17% / -5.0 | +73% / -18.8 |
| code-mvt1024 | +9% / +2.5 | >+93% / -13.7 | +29% / +0.2 | +27% / -2.5 | +58% / -3.4 |
| code-mvt1024-tokens32 | +8% / -0.6 | >+57% / -5.6 | +14% / -1.3 | +2% / +0.0 | +24% / -1.9 |
| code-whole | -2% / +0.0 | >-9% / -6.2 | +4% / +0.2 | -10% / -8.1 | -4% / -3.5 |
| math-fade-r1 | >+24% / +5.0 | +14% / -5.6 | +5% / -1.3 | +5% / -4.4 | +13% / -1.6 |
| math-fade-r2 | >+35% / +1.9 | +9% / -9.4 | +6% / -1.9 | +9% / -7.5 | +16% / -4.2 |
| math-fade-r3 | >+50% / -5.6 | +26% / -10.6 | +12% / -2.1 | -8% / -0.6 | +23% / -4.7 |
| math-marginal | >+97% / -46.2 | +30% / -17.5 | +15% / -0.2 | -67% / -11.9 | +57% / -19.0 |
| math-mvt1024 | >+80% / -16.2 | +43% / -17.5 | +42% / -5.2 | +3% / -6.9 | +49% / -11.5 |
| math-mvt1024-tokens32 | >+55% / -14.4 | +52% / -28.1 | +25% / -3.8 | +1% / -6.9 | +37% / -13.3 |
| math-whole | >+1% / +3.7 | -2% / -5.0 | +2% / -1.3 | -5% / -6.2 | -1% / -2.2 |
| text-fade-r1 | +15% / -0.6 | +11% / -1.2 | +11% / +0.2 | >+3% / +0.6 | +10% / -0.3 |
| text-fade-r2 | +9% / +3.7 | -4% / -8.1 | +8% / -0.0 | >+11% / -4.4 | +6% / -2.2 |
| text-fade-r3 | +21% / +5.6 | -0% / -4.4 | +5% / -1.5 | >+20% / -3.1 | +12% / -0.8 |
| text-marginal | +8% / +1.2 | +2% / -5.6 | +12% / -0.8 | >+6% / +0.0 | +7% / -1.3 |
| text-mvt1024 | +20% / +3.1 | +13% / -5.0 | +9% / -1.0 | >+11% / -3.1 | +13% / -1.5 |
| text-mvt1024-tokens32 | +11% / +5.6 | -2% / -3.7 | +6% / +0.4 | >+5% / +0.0 | +5% / +0.6 |
| text-whole | +1% / +7.5 | -5% / -6.9 | +7% / -0.0 | >+9% / -2.5 | +3% / -0.5 |
| tool-fade-r1 | +4% / +4.4 | -8% / -7.5 | >-10% / -0.8 | -14% / -6.9 | -7% / -2.7 |
| tool-fade-r2 | +9% / +2.5 | -11% / -8.7 | >+32% / -0.4 | +6% / -3.1 | +10% / -2.4 |
| tool-fade-r3 | +34% / +1.9 | +57% / -10.0 | >+38% / -0.6 | +37% / -5.6 | +43% / -3.6 |
| tool-marginal | +11% / +4.4 | +34% / -10.0 | >+33% / -1.3 | +16% / -5.0 | +24% / -3.0 |
| tool-mvt1024 | +15% / +3.7 | +68% / -8.7 | >+43% / +2.1 | +26% / +0.0 | +42% / -0.7 |
| tool-mvt1024-tokens32 | -11% / -1.3 | +12% / -7.5 | >+28% / -1.5 | -20% / +3.7 | +4% / -1.6 |
| tool-whole | +11% / +3.7 | -1% / -9.4 | >+7% / -0.0 | -12% / -7.5 | +2% / -3.3 |

BFCL split for tool runs (saved irrelevance/multiple/simple / acc): tool-mvt1024 +37/+54/+38% / -1.3/+1.9/+5.6; tool-mvt1024-tokens32 +32/+15/+36% / +1.9/-3.1/-3.1; tool-marginal +35/+30/+33% / +0.6/-2.5/-1.9; tool-fade-r1 irrelevance -30% (longer), multiple -10%, simple +11%; tool-fade-r2 +40/+30/+26% / +1.2/+0.6/-3.1; tool-fade-r3 +40/+44/+32% / +1.2/-1.9/-1.3.

#### Combined models (4 samples vs raw model, unless noted)

Retrained 1b on this runtime (mvth1024-full-gsm8k, 2 samples): +42.4% / -3.2 (previous training: +47.6% / -1.5).

| model | rows | tokens saved | accuracy change | per benchmark |
|---|---|---|---|---|
| combine-final: MC MVT + math whole x2 + code MVT tokens32 + tool MVT + text MVT | 860 | +49.0% [40.2, 58.9] | -6.2 [-9.7, -3.1] | BFCL +44/+42/+28% / +8.1/-11.2/-2.5; IFEval +33% / -6.2; MBPP+ +81% / -11.9; SVAMP +12% / -5.0 |
| combine-final-mathfade: same, GSM8K faded to round 2 | 860 | +55.3% | -5.9 | SVAMP +50% / -10.0 (lost rows' reasoning 966 -> 357) |
| mvth1024-full-gsm8k-xgsm8k2 retrained | 270 | +31.1% | -2.2 | (previous training +39.1% / -1.0) |
| combine-final-light: MC MVT + tool MVT + math whole x2 (code, instructions untouched) | 502 | +48.6% | -2.3 | BFCL +40/+48/+27% / +6.2/0.0/-0.6; IFEval +49% / 0.0; SVAMP +4% / +1.2; MBPP+ +77% / -12.5 |

Weight merging (`merge.py`, W = W0 + sum w_i dW_i over per-kind adapters): queued in this range, no results logged here.

---

### Track 2c: shortest-of-8 self-distillation (date: — , before 2026-10-21; Colab L4)

Data: 1,500 questions (GSM8K train+test, MBPP minus MBPP+, ToolACE in BFCL's mix, IFEval-like, ARC/CSQA), 8 replies each at the evaluation's caps (4,096; 8,192 for code): 12,000 replies, ~4.3 h on the L4. Per question the shortest correct reply kept unedited (`select_shortest.py`); GSM8K only where that reply <= 700 tokens.

| source | questions | solved | kept | capped | mean reply | kept len | same-question mean | saved |
|---|---|---|---|---|---|---|---|---|
| arc_easy | 48 | 48 | 48 | 1% | 532 | 334 | 532 | 37% |
| commonsense_qa | 252 | 211 | 211 | 0% | 512 | 339 | 494 | 32% |
| gsm8k_all | 300 | 240 | 106 | 22% | 2,163 | 496 | 1,282 | 61% |
| ifeval_like | 300 | 272 | 272 | 2% | 418 | 285 | 409 | 30% |
| mbpp_train | 300 | 168 | 168 | 13% | 2,835 | 1,098 | 1,834 | 40% |
| toolace | 300 | 206 | 206 | 0% | 299 | 190 | 272 | 30% |

Contamination check (`contamination.py`: share of a training question's 8-grams found in one evaluation question >= 50%): 16 of 1,011 rows dropped, all MBPP training tasks that are near copies of MBPP+ tasks under other ids (51-81% overlap, e.g. mbpp_train-466 ~ Mbpp/758). The Track 2b grid used the same MBPP tasks without this check. Check applied to all later training sets.

SFT (bon-r1): 990 rows, LoRA r16, lr 5e-5, 1 epoch. Eval 240 q x 4 samples:

| benchmark | saved / acc |
|---|---|
| BFCL irrelevance | +10% / +1.2 |
| BFCL multiple | +7% / -3.1 |
| BFCL simple | +9% / -3.8 |
| IFEval | +9% / -5.0 |
| MBPP+ | -3% / -8.1 |
| SVAMP | +3% / +3.7 |
| overall | +4.4% / -2.8 |

### Track 2d: steering directions; best-of-8 DPO; stronger SFT (date: — ; ended 2026-10-21 ~11:07 Paris with loss of runtime A)

#### Steering directions (`steer_dirs.py`, `track.py steer`)

Method: residual-stream directions from the raw model's best-of-8 replies (first 64 thinking tokens): A = mean difference shortest - longest correct; B = within-question trend against log length, minus the "gives up" direction (wrong - correct); C = one learned vector (DPO, all else frozen). Stored as a v_proj bias (Qwen3 `attention_bias`), checked identical in vLLM and Transformers (per-token effect ratio 1.08, correlation 0.99). Measured by held-out pairwise accuracy at predicting which of two correct replies is shorter (0.50 = chance, about +/-0.1 at 30-45 pairs).

| reading | data | pooled direction | by kind (pooled direction) | per-kind direction notes |
|---|---|---|---|---|
| first | 200 q per kind; 2,826 replies, 351 fit pairs | layer 9: A 0.70, B 0.73; layers 7-13: 0.66-0.70; early ~0.60; late ~0.62 | (A, layer 9) tool 0.80, text 0.78, choice 0.73, math 0.61, code 0.61 | cosines between kinds -0.10 to 0.36 (math~tool 0.36); ~55-80 pairs each; pooled scores >= own on most kinds |
| bigger | all finished replies of 1,500 q; 537 fit pairs, 250 held-out pairs | 0.64-0.68 over layers 8-17; A layer 14 0.66, B layer 8 0.68 | text 0.73-0.79, tool 0.68-0.73, choice 0.63-0.68, math ~0.60-0.65, code ~0.60-0.65 | own > shared only at early layers (code B layer 0: 0.70; tool A layer 2: 0.77); cosines -0.18 to 0.39 (text~choice 0.35-0.39, math~code 0.37, math~tool 0.31-0.34) |

Steering push tests (240 q x 2 samples):

| push | tokens saved | accuracy change |
|---|---|---|
| A layer 9, x5 | +17.6% | -0.1 |
| A layer 9, x15 | +22.3% | -5.5 |
| A layer 12, x5 | +2.5% | -2.7 |
| A layer 12, x15 | +28.6% | -7.5 |
| B layer 9, x5 | +10.5% | -0.5 |
| B layer 9, x15 | +23.5% | -2.9 (4 samples: +15.7% / -5.0) |
| C layer 9, x8 | +3.3% | +3.2 |
| C layer 9, x24 | +17.6% | -8.2 |

Fresh replies on 141 held-out training questions x 4: mean tokens raw 1,156 -> A 981 / C 970 at accuracy 56-57%; B 964 at -4.9 pts. Shortest correct of 4: raw 450, A 459, B 407, C 427. Math under C (25 questions): accuracy 36% -> 47%, mean 2,375 -> 1,745 tokens.

#### Best-of-8 DPO (`bon_pairs.py`), from adapter bon-r1

Pairs: 1,267 (704 shorten, 563 guard). Shorten pairs chosen vs rejected mean tokens: GSM8K 493 vs 2,033; MBPP 747 vs 2,224; ToolACE 182 vs 514; IFEval 261 vs 493; CSQA 276 vs 723; ARC 355 vs 874. DPO: 1,125 pairs, 2 epochs, 141 steps, checkpoints every 25. Eval 240 q x 4 samples.

| checkpoint | tokens saved | accuracy | SVAMP | MBPP+ | IFEval | BFCL simple | BFCL multiple | BFCL irrelevance |
|---|---|---|---|---|---|---|---|---|
| step-25 | +47.6% | -2.6 | +62% / +0.6 | +43% / -9.4 | +48% / -1.2 | +29% / +0.6 | +34% / +0.6 | +39% / -2.5 |
| step-50 | +40.3% | -2.8 | +47% / -2.5 | +42% / -5.0 | +40% / -3.7 | +27% / +0.6 | +32% / -0.6 | +34% / +0.0 |
| step-75 | +37.3% | -2.1 | +5.0 (acc) | -10.0 (acc) | — | — | — | — |
| step-100 | +36.5% | -2.6 | — | -9.4 (acc) | -6.2 (acc) | — | — | — |
| step-125 | +31.1% | -3.2 | — | — | — | — | — | — |

Training-pair accuracy: 99% ordered right in the second pass.

#### Stronger SFT on clear-gap questions (bon-r2)

Shortest correct >= 25% under the question's mean; lr 1e-4, 3 epochs. 4 samples: +12.5% / -0.7 (SVAMP +16% / +0.6, MBPP+ +7% / -2.5, IFEval +11% / -2.5, BFCL +10-22% / -1.9 to +4.4).

#### Runtime loss

Colab recycled runtime A at about 11:07 Paris (2026-10-21): disk, all adapters (bon-dpo checkpoints, bon-r2) and the 14,400 best-of-8 rollouts lost. The worker had stopped receiving branch updates after commit 5d3f8ce, so the MBPP guard-only DPO variant never ran.

---

### Track 2e: rebuild with Drive persistence, iterated on-policy DPO rounds (2026-10-21 to 2026-10-22; Colab L4, two runtimes)

#### Round 1 data and recipe

750 of the 1,500 questions x 8 (mixed half); 466 SFT rows after 8 contamination drops; 591 DPO pairs. Kept vs same-question mean: math 59%, code 48%, CSQA 38%, ARC 34%, IFEval 34%, ToolACE 32%. Recipe: gentle SFT then DPO, pushed to 134 steps (4 passes over 533 pairs; one pass ~33 steps); every checkpoint scored in one engine (`sweep_eval.py`), 240 q x 2 samples.

| step | tokens saved | accuracy | SVAMP | MBPP+ | IFEval |
|---|---|---|---|---|---|
| 10 | +36.2% | -1.5 | +38% / -2.5 | +29% / -7.5 | +49% / +2.5 |
| 20 | +47.4% | -6.4 | +55% / -7.5 | +37% / -16.2 | +61% / -3.7 |
| 30 | +42.5% | -2.9 | +52% / -7.5 | +25% / -1.2 | +56% / -2.5 |
| 50 | +32.6% | -2.2 | +31% / 0.0 | +11% / -7.5 | +52% / -2.5 |
| 70 | +37.6% | -1.0 | +46% / -1.2 | +21% / -1.2 | +51% / -3.7 |
| 100 | +27.9% | -1.2 | +31% / -3.7 | +11% / -3.7 | +40% / 0.0 |
| 130 | +22.4% | -0.1 | +24% / -1.2 | +0% / -1.2 | +37% / 0.0 |

Pairs 100% ordered right from step ~60. Step 30 at 4 samples: +44.9% / -2.7 (SVAMP +53% / -4.4, MBPP+ +41% / -8.7, IFEval +50% / +3.1, BFCL +30-37% / -1.9 to 0.0).

#### Round 1 variants (2 samples unless noted)

| variant | step 10 | step 20 | step 30 | step 40 | step 60 | 4-sample check |
|---|---|---|---|---|---|---|
| DPO from raw model (fresh LoRA, no SFT), same pairs, 2 passes | +11.7% / -1.1 | +33.1% / -2.6 | +42.0% / -2.3 | +37.9% / -4.7 | +37.3% / -5.0 | step 30: +36.6% / +0.8 (SVAMP +39% / +0.6, MBPP+ +35% / -5.0, IFEval +43% / +6.9, BFCL +26-30% / -0.6 to +1.9) |
| MBPP guard pairs only (median-length correct reply), from SFT | +37.4% / -3.3 (MBPP+ +24% / -8.7) | +44.2% / -3.3 (MBPP+ -6.2) | +40.4% / -5.2 (MBPP+ -10.0) | +35.9% / -5.0 (MBPP+ -15.0) | — | — |
| no code pairs, from SFT ("no-code") | +35.6% / +1.1 (MBPP+ +25% / +1.3) | +50.6% / -4.4 (MBPP+ +36% / -8.7) | +44.1% / -2.8 (MBPP+ +41% / -6.2) | +35.3% / -1.0 (MBPP+ +27% / -1.2) | — | step 30: +42.4% / -0.7 (SVAMP +53% / +1.2, MBPP+ +38% / -3.7, IFEval +44% / -1.2, BFCL +29-35% / 0.0 to +2.5) |

The log compares no-code step 30 with an external reference "Lightning McQwen3" at +43.2% / -0.85 (full evaluation x 8; ours 240 q x 4).

#### Round 2 (on-policy, from no-code step 30, 750 unused questions x 8)

Round-1 model replies: math mean 661 tokens vs 2,221 raw; capped 1% vs 26%; sampling ~40 min. Shortest correct vs own mean: math 33%, code 39%, CSQA 24%, instructions 19%, ARC 18%, tools 16%. 7 contamination drops; 506 pairs (424 without code). One DPO pass.

Bug: `dpo.py` never stepped on the last partial batch, so a 317-pair pass ended at step 19 without a final checkpoint (fixed).

| run | step | samples | tokens saved | accuracy | SVAMP | MBPP+ | IFEval | BFCL |
|---|---|---|---|---|---|---|---|---|
| r2, code pairs in | 10 | 2 | +55.4% | -4.4 | +68% / -5.0 | +41% / -6.2 | +66% / -7.5 | +37-44% / 0..+3.8 |
| r2, code pairs in | 20 | 2 | +63.7% | -4.0 | +76% / -6.2 | +68% / -8.7 | +60% / -2.5 | +42-47% / -2.5..+7.5 |
| r2, code pairs in | 20 | 4 | +62.6% | -3.6 | +75% / -5.0 | +65% / -14.4 | — | — |
| r2nc (no code pairs) | 10 | 2 | +57.6% | -4.5 | +67% / -7.5 | +50% / -11.2 | +68% / -1.2 | +37-43% / -2.5..+6.2 |
| r2nc | 10 | 4 | +55.5% | -1.7 | +64% / 0.0 | +54% / -10.6 | +61% / +1.9 | +36-42% / -1.3..+5.0 |
| r2g (gentle: guard 60%, no code) | 5 | 2 | +52.1% | -5.0 | — | — | — | — |
| r2g | 10 | 2 | +55.3% | -0.9 | — | — | — | — |
| r2g | 15 | 2 | +60.6% | -6.1 | — | — | — | — |
| r2g | 10 | 4 | +54.3% | -1.1 | +63% / 0.0 | +50% / -9.4 | +61% / +2.5 | +36-44% / -0.6..+6.9 |
| r2g | 10 | 8 | +54.2% | -1.6 | +64% / 0.0 (23 lost, 23 gained) | +60% / -7.5 | +50% / -0.6 | +34-42% / -0.6..+5.3 |

r2g step 10 at 8 samples, mean tokens: SVAMP 291, MBPP+ 876, IFEval 337, BFCL 119-128.

#### Round 3 (fresh draw seed 1, from r2g step 10)

1,302 of 1,500 fresh questions unused; 750 sampled x 8. Replies: math mean 528 (0% capped), code 995, instructions 219, tools 143. Shortest correct vs own mean: math 36%, code 34%, ARC 23%, CSQA 22%, instructions 21%, tools 14%. 3 contamination drops; 420 pairs (no code); mix 0.4 shorten / 0.6 guard; one pass.

| step | samples | tokens saved | accuracy | SVAMP | MBPP+ | IFEval | BFCL |
|---|---|---|---|---|---|---|---|
| 5 | 2 | +59.8% | -0.1 | +72% / -1.2 | +47% / 0.0 | +71% / -2.5 | — |
| 10 | 2 | +61.6% | -3.4 | +77% / -10.0 | +52% / -6.2 | +65% / 0.0 | — |
| 15 | 2 | +66.6% | -6.4 | +80% / -7.5 | +57% / -16.2 | +74% / -5.0 | — |
| 20 | 2 | +69.4% | -4.2 | +82% / -7.5 | +73% / -8.7 | +67% / -3.7 | — |
| 5 | 4 | +56.8% | -2.6 | +69% / -0.6 | +53% / -12.5 | +59% / +1.3 | +38-45% / -1.3..+3.1 |
| 5 | 8 | +56.9% | -1.9 | +69% / -3.1 | +57% / -7.5 | +55% / +0.6 | +36-44% / -0.9..+6.2 |

#### Code diagnosis and code "stopped too early" pairs

Diagnosis (r2g step 10, 240 x 8 vs raw): MBPP+ lost 40 answers, gained 16; every loss a wrong program (none capped, none unreadable). Thinking on lost rows: 1,420 -> 530 tokens. Example: `next_smallest_palindrome` returns n when n is already a palindrome; the thinking checks that case and accepts it.

`code_pairs.py` (round-3 answers of r2g): 32 code questions with a correct reply, 2 replies each, cut at 25/50/75% of thinking, 4 re-answers. Wrong after cut: 24/60 (25%), 15/60 (50%), 10/60 (75%). 30 pairs added to round 3's 420 (199 shorten, 221 guard); code = 7% of pairs.

| run | step | samples | tokens saved | accuracy | per benchmark |
|---|---|---|---|---|---|
| code-r3 (r3 pairs + 30 code too-early, from r2g) | 5 / 10 / 15 / 20 | 2 | +56.0 / +57.5 / +60.1 / +64.5% | -2.0 / -4.0 / -6.4 / -6.4 | — |
| code-r3 | 5 | 4 | +53.2% | -0.1 | SVAMP +67% / +3.1, IFEval +53% / +3.7, BFCL +37-44% / -0.6..+5.6, MBPP+ +49% / -10.0 |
| code-r3 | 5 | 8 | +53.6% | -2.9 | SVAMP +66% / -1.9, IFEval +51% / -3.4, MBPP+ +54% / -8.1, BFCL +35-43% / -0.3..+5.0 |
| code-r3b (all code questions of both draws answered 4x by r2g: 300 + 133, 18 contamination drops; 167 code too-early pairs (28 + 102 + 37) + round 3's 420) | 5 / 10 / 15 / 20 | 2 | +55.2 / +55.5 / +60.3 / +65.3% | -3.6 / -3.4 / -2.2 / -4.7 | — |
| code-r3b | 15 | 4 | +60.4% | -2.7 | MBPP+ -13.7 |
| code-r3b | 15 | 8 | +60.9% | -3.0 | SVAMP +76% / -6.2, MBPP+ +61% / -7.8, IFEval +57% / +0.6, BFCL +36-44% / -0.6..+5.3 |
| soup r2g step 10 + code-r3b step 15 (`merge.py`, 0.5 + 0.5) | — | 8 | +56.7% | -2.4 | — |

Log rule adopted after code-r3 (4-sample -0.1 vs 8-sample -2.9): finalists compared at 8 samples only.

8-sample table on questions_40 (as logged):

| model (240 q x 8) | tokens saved | accuracy |
|---|---|---|
| r2g step 10 | +54.2% | -1.6 |
| bon-dpo-r3 step 5 (round 3) | +56.9% | -1.9 |
| soup r2g + code-r3b | +56.7% | -2.4 |
| code-r3 step 5 | +53.6% | -2.9 |
| code-r3b step 15 | +60.9% | -3.0 |
| Track 2 mvth1024-full-gsm8k-xgsm8k2 | +37.1% | -0.6 |

#### Rounds 4 and 5

| run | data | samples | tokens saved | accuracy | per benchmark |
|---|---|---|---|---|---|
| round 4 (seed-2 draw, from round 3 step 5) | 1,202 of 1,500 fresh; replies math 432, code 1,207, instructions 230, tools 142; 424 pairs | 2 | step 5 +59.9 / -4.0; step 10 +62.1 / -4.7; step 15 +66.1 / -2.4 | | |
| round 4 step 15 | | 4 | +66.6% | -1.1 | — |
| round 4 step 15 | | 8 | +65.4% | -2.4 | SVAMP +80% / -2.8, MBPP+ +68% / -10.0, IFEval +58% / -0.3, BFCL +41-48% / -0.3..+9.4 |
| round 5 (seed-3 draw, from round 4 step 15) | — | 2 | step 5 +69.3 / -2.2; step 10 +70.9 / -3.3; step 15 +73.5 / -4.3 | | |
| round 5 step 5 | | 4 | +71.1% | -1.1 | — |
| round 5 step 5 | | 8 | +68.6% | -2.4 | SVAMP +82% / -2.2, MBPP+ +71% / -10.3, IFEval +64% / +1.3, BFCL +43-49% |

#### Training-seed variance (runtime 2, 2 samples, questions_40)

| run | DPO seed 1 retrain | original at same step |
|---|---|---|
| round 2 (r2g step 10) | +49.7% / -3.6 | +55.3% / -0.9 |
| round 4 (step 15) | +68.0% / -7.5 | +66.1% / -2.4 |

#### Held-out evaluations (removing checkpoint-selection bias)

All earlier checkpoint/variant choices were made on questions_40. Held-out sets: `questions_fresh20261022` (`subset_questions.py --per-dataset 40 --seed 20261022 --exclude questions_40`, runs `runs/mvtf/f8-*`, 8 samples vs raw on the same set); questions 41-80 of every part (mvt-pertype session, 8 samples, read via `track.py peek`).

| model | questions_40 (8 samples unless noted) | questions 41-80 x 8 | fresh20261022 x 8 | fresh per benchmark |
|---|---|---|---|---|
| round 1 no-code step 30 | +42.4% / -0.7 (4 samples) | +43.6% / -0.9 | +40.4% / -0.9 | — |
| r2g step 10 | +54.2% / -1.6 | +53.1% / -2.7 (SVAMP -7.5, IFEval -4.1, MBPP+ -1.6) | +51.4% / -1.8 | — |
| r2nc step 10 | +55.5% / -1.7 (4 samples) | +53.4% / -2.8 | — | — |
| round 3 step 5 | +56.9% / -1.9 | — | +56.0% / -3.6 | SVAMP -7.5 |
| round 4 step 15 | +65.4% / -2.4 | — | +63.2% / -3.9 | SVAMP -9.1 |
| round 4 retrain (seed 1) | +68.0% / -7.5 (2 samples) | — | — | SVAMP -10.6 |
| round 2 retrain (seed 1) | +49.7% / -3.6 (2 samples) | — | +50.9% / -3.0 | SVAMP -9.7 |
| round 5 step 5 | +68.6% / -2.4 | — | +65.5% / -5.4 | SVAMP +79% / -10.0, IFEval +55% / -5.0, MBPP+ +73% / -2.8, BFCL +43-47% |

Log observation on fresh sets: SVAMP losses -7.5 to -10.6 for late rounds, MBPP+ -2.5 to +0.6. Round 4 pair mix: 96 math shorten pairs vs 10 tools, 15 instructions.

#### MBPP-free families (`track.py clean`)

Motivation in log: every round 1-4 model descends from the gentle SFT step (466 rows incl. ~74 MBPP tasks, near-copies dropped); "DPO from raw" used MBPP pairs. Family A: no code data at all. Family B: plus the MBPP-like slice of KodCode-Light-RL-10K as right-beats-wrong pairs only.

KodCode vs MBPP+ (Qwen3 tokenizer, medians; task tokens / solution tokens / asserts): MBPP+ 17 / 29 / 3 base asserts (plus EvalPlus extra inputs); MBPP minus MBPP+ 16 / 48 / 3; all usable KodCode (6,171) 99 / 114 / 7-8; MBPP-like slice (task <= 45, solution <= 90 tokens): 907 tasks, 28 / 63 / 6-7. Topic shares slice vs MBPP+: lists 53 vs 45%, numbers 59 vs 46%, strings 29 vs 19%. 8-gram contamination check against all 2,759 test questions applied on top of KodCode's own decontamination.

| run | data | samples / set | tokens saved | accuracy | per benchmark |
|---|---|---|---|---|---|
| A1 (DPO from raw, round-1 answers without code, 438 pairs, 28 steps) step 10 | | 2 / q40 | +15.8% | 0.0 | — |
| A1 step 20 | | 2 / q40 | +41.9% | -3.6 | — |
| A1 step 10 | | 4 / q40 | +17.8% | +1.7 | — |
| A1 step 10 | | 8 / q40 | +17.7% | +1.1 | — |
| A1 step 20 | | 4 / q40 | +37.8% | +0.1 | SVAMP +41% / +2.5, IFEval +52% / +4.4, MBPP+ +27% / -7.5, BFCL +26-29% / -0.6..+1.9 |
| A1 step 28 | | 4 / q40 | +40.5% | -1.2 | MBPP+ -8.7 |
| A1 step 28 | | 8 / fresh | +35.4% | -1.9 | — |
| A2 (from A1 step 28; 750 unused questions without code, 345 pairs) step 5 / 10 / 15 | | 2 / q40 | +45.9 / +47.0 / +53.3% | -3.1 / -5.9 / -4.6 | SVAMP -3.7 to -11.2 |
| A2 step 5 | | 4 / q40 | +44.6% | +0.3 | — |
| B1 step 20 (KodCode right-beats-wrong) | | 8 / fresh | +24.8% | -1.5 | SVAMP +36% / -0.3, MBPP+ +24% / +0.6, IFEval +19% / -4.4, BFCL +17-21% |
| B2 step 10 | | 8 / fresh | +32.7% | -0.1 | SVAMP +43% / -1.9, MBPP+ +31% / +1.9, IFEval +31% / +0.6, BFCL +24-25% / -2.5..-0.3 |

#### Balanced round (`track.py balanced`, from round 2 / r2g)

Method: savings pairs from tools, instructions and multiple choice (short-beats-long only where the question is answered right >= 6/8); math protected (at most 40 shorten pairs per math source; right-beats-wrong with a median-length correct reply); math data with SVAMP-style traps: GSM-Plus (Li et al. 2024) and GSM-IC (Shi et al. 2023). Round 2 answered 1,437 fresh questions x 8. Shortest correct vs own mean: gsm_plus 41%, gsm_ic 32%, ARC 27%, instructions 24%, CSQA 19%, tools 14%; 0 contamination drops. 736 pairs: shorten 218 (CSQA 59, gsm_plus 40, gsm_ic 40, instructions 35, ARC 24, tools 20), guard 518. Mix 30 / 70. Three DPO seeds of 43 steps each, averaged per checkpoint step (`merge.py` soup). Pairs ordered right 46-64% at steps 10-15.

| soup (3 seeds) | samples / set | tokens saved | accuracy | per benchmark |
|---|---|---|---|---|
| step 10 | 4 / q40 | +55.9% | -2.0 | — |
| step 20 | 4 / q40 | +58.9% | -0.5 | SVAMP +67% / +1.9, IFEval +65% / +4.4, BFCL +38-47% / -0.6..+3.1, MBPP+ +57% / -10.0 |
| step 30 | 4 / q40 | +57.5% | -0.5 | — |
| step 40 | 4 / q40 | +55.6% | -0.9 | — |
| step 20 | 8 / q40 | +58.8% | -2.1 | SVAMP +68% / -1.6, MBPP+ +60% / -8.4, IFEval +61% / -0.3, BFCL +36-44% |
| step 20 | 8 / fresh | +53.8% | -2.5 | SVAMP +64% / -4.7, MBPP+ +58% / -1.6, IFEval +50% / -2.8, BFCL +36-41% |

Infrastructure note: runtime 2's worker predated the `end` stage line and failed on it; `end` changed to a stage that saves, flushes and stops its worker.

---

## Part B — Track 3 per-type clone (20–21 Oct, ran in parallel with Track 2b–2e)

Source: clone `Test-mvt-pertype`, branch `mvt-pertype`, `thinkingcap/EXPERIMENT_LOG.md` lines 634-954 (content unique to this clone; lines 1-633 are shared with `Test-mvt-fading`), `thinkingcap/FINDINGS_TABLE.md` sections 3.4-3.5, `thinkingcap/REPORT_MVT_PERTYPE.txt` (method details). Dates from git log (all commits 2026-10-20 to 2026-10-21).

### Common setup

| item | value |
|---|---|
| hardware | own Colab L4 runtime, from 2026-10-20 22:08; deleted 2026-10-21 ~13:15 |
| starting model (Track 3) | untouched Qwen3-0.6B; every grid run trains a fresh adapter from it (25 separate adapters, not stacked) |
| adapter | LoRA rank 16 (~10 M parameters, 1.7% of model), base frozen |
| SFT settings (Track 3) | 2 epochs, lr 1e-4, 16 examples per step |
| MVT cut | p(model's own correct answer) measured after each piece of thinking by teacher forcing; cut at the first point where no later piece pays better per token than the average so far, g(t)/(t+c). c=1024 or c=256. Pieces = paragraphs or fixed 32-token batches ("tokens32") |
| safety check (Track 3) | cut trace kept only if the model, answering naturally from it, is right 2 times; otherwise the original uncut trace is kept. Per question the shortest accepted trace becomes the training example |
| "whole" runs | correct traces used without cutting (control for the effect of training alone) |
| token caps | 4,096; 8,192 for code (same as the evaluation) |
| evaluation set | 240-question subset: first 40 questions of each of the 6 parts (SVAMP, MBPP+, IFEval, BFCL simple/multiple/irrelevance) in the evaluation's fixed order |
| samples | 4 per question vs `s4-base`; 8-sample rechecks vs `s8-base` (8-sample rechecks reuse the first 4 answers' seeds) |
| metric | tokens saved % / accuracy change in points vs the untouched model on the same questions |
| noise (from report) | one benchmark cell moves by about +/-4 points by chance; single-benchmark intervals +/-5 to +/-9 points. Each run trained once (training randomness not measured) |

---

### B1. Data fix and raw-model reference (2026-10-20)

Method: `prepare_types.py` changed so that ToolACE prose first replies are used as BFCL-irrelevance examples only when they say no function can do it (about half of ToolACE's prose replies instead ask for a missing argument of a function that fits). Counts usable after filter (CPU check): 5,025 simple, 3,615 multiple, 276 irrelevance. The mvt-fading session's `bon` data predates this filter.

Raw Qwen3-0.6B, 240 questions x 2 samples (this runtime):

| benchmark | accuracy | mean tokens | capped |
|---|---|---|---|
| BFCL irrelevance | 82.5% | 204 | — |
| BFCL multiple | 76.3% | 223 | — |
| BFCL simple | 82.5% | 190 | — |
| IFEval | 53.8% | 734 | 3.8% |
| MBPP+ | 57.5% | 1,699 | 7.5% |
| SVAMP | 81.3% | 940 | 3.8% |

Log notes: same as the earlier runtimes (same seed).

`s4-base` (240 x 4), the reference for every grid cell:

| benchmark | accuracy | mean tokens |
|---|---|---|
| BFCL irrelevance | 85.6% | 213 |
| BFCL multiple | 79.4% | 217 |
| BFCL simple | 83.1% | 185 |
| IFEval | 51.9% | 705 |
| MBPP+ | 57.5% | 2,004 |
| SVAMP | 81.9% | 830 |

An `s8-base` (240 x 8) was also computed for 8-sample rechecks (per-benchmark values not recorded in the log).

### B2. Training data, matched set (2026-10-20)

Method: `grid_data --per-source 300`, `TRACK_SET=matched`. Usable pool after `prepare_types.py`: gsm8k_all (GSM8K train+test) 8,792; mbpp_train (MBPP minus MBPP+) 596; toolace 300 (144 simple, 72 multiple, 84 irrelevance, drawn from 5,025 / 3,615 / 276); ifeval_like (argilla IFEval-like) 56,339; multiple choice 11,592. 300 kept per source (multiple choice: 300 shared by ARC-Easy and CommonsenseQA; came out 60 ARC-Easy / 240 CommonsenseQA). 3,000 replies sampled: the untouched model's own replies, 2 per question, seed 0, ~80 min on the L4. GSM8K: only correct replies of at most 700 tokens used (`--easy-tokens 700`, SVAMP-like filter).

| source | replies | correct | capped | mean tokens |
|---|---|---|---|---|
| arc_easy | 120 | 88% | 0% | 492 |
| commonsense_qa | 480 | 51% | 0% | 506 |
| gsm8k_all | 600 | 43% | 24% | 2,167 |
| ifeval_like | 600 | 78% | 1% | 408 |
| mbpp_train | 600 | 38% | 12% | 2,842 |
| toolace | 600 | 58% | 0% | 265 |

Questions with at least one usable correct reply (training rows per kind): tool 189, text 259, choice 203 (arc_easy 55, commonsense_qa 148), code 137, math 60.

### B3. Per-type grid, matched data (2026-10-21)

Method: 5 kinds (tool, text, choice, code, math) x 5 settings (whole; MVT c=1024 paragraphs; MVT c=256 paragraphs; MVT c=1024 tokens32; MVT c=256 tokens32) = 25 runs. Each run: SFT adapter on one kind's rows only (settings in Common setup), from the untouched model. Evaluation: 240 questions x 4 samples, all benchmarks, vs `s4-base`.

#### E3.1 Cut statistics per run

| run | traces (kept / cut) | cuts checked / passed | target tokens (before -> after) |
|---|---|---|---|
| tool-whole | 189 / 0 | — | 201 -> 183 |
| tool-mvt1024 | 189 / 178 | 356 / 332 | 201 -> 142 |
| tool-mvt256 | 189 / 173 | 351 / 314 | 201 -> 121 |
| tool-mvt1024-tokens32 | 189 / 176 | 406 / 373 | 201 -> 106 |
| tool-mvt256-tokens32 | 189 / 167 | — | 201 -> 89 |
| text-whole | 259 / 0 | — | 347 -> 312 |
| text-mvt1024 | 259 / 238 | 503 / 445 | 347 -> 234 |
| text-mvt256 | 259 / 227 | 483 / 412 | 347 -> 190 |
| text-mvt1024-tokens32 | 259 / 234 | 543 / 476 | 347 -> 218 |
| text-mvt256-tokens32 | 259 / 219 | — | 347 -> 180 |
| choice-whole | 203 / 0 | — | ~455 -> ~400 (ARC 454 -> 379, CSQA 456 -> 408) |
| choice-mvt1024 | 203 / 201 | 470 / 464 | ~455 -> ~97 (ARC 454 -> 93, CSQA 456 -> 100) |
| choice-mvt256 | 203 / 194 | 443 / 419 | ~455 -> ~102 |
| choice-mvt1024-tokens32 | 203 / 199 | 496 / 487 | ~455 -> ~76 (ARC 454 -> 64, CSQA 456 -> 79) |
| choice-mvt256-tokens32 | 203 / ~190 | — | — |
| code-whole | 137 / 0 | — | 1,493 -> 1,308 |
| code-mvt1024 | 137 / 92 | 262 / 180 | 1,493 -> 790 |
| code-mvt256 | 137 / 82 | 231 / 140 | 1,493 -> 812 |
| code-mvt1024-tokens32 | 137 / 90 | 280 / 189 | 1,493 -> 824 |
| code-mvt256-tokens32 | 137 / 78 | 233 / 140 | 1,493 -> 903 |
| math-whole | 60 / 0 | — | 676 -> 509 |
| math-mvt1024 | 60 / 57 | — | 676 -> 131 |
| math-mvt256 | 60 / 57 | 194 / 184 | 676 -> 123 |
| math-mvt1024-tokens32 | 60 / ~55 | 201 / 171 | 676 -> 143 |
| math-mvt256-tokens32 | 60 / 50 | — | 676 -> 132 |

#### E3.2 Results, overall and per benchmark (4 samples, 240 q, vs `s4-base`; each cell = tokens saved / accuracy pts)

| run | overall saved | overall acc | SVAMP | MBPP+ | BFCL irr | BFCL mult | BFCL simple | IFEval |
|---|---|---|---|---|---|---|---|---|
| tool-whole | +5.2% | +0.3 | +12% / +1.9 | +6% / +0.6 | +6% / 0.0 | +5% / +0.6 | +6% / -3.1 | -4% / -0.6 |
| tool-mvt1024 | +17.6% | +1.7 | +8% / +1.3 | +2% / -3.1 | +38% / -1.2 | +55% / -3.1 | +42% / +1.9 | +6% / +9.4 |
| tool-mvt256 | +25.8% | +0.6 | +10% / -0.6 | +30% / -4.4 | +45% / +3.1 | +55% / -1.9 | +36% / +0.6 | +11% / +6.9 |
| tool-mvt1024-tokens32 | +18.1% | -2.7 | +24% / -1.9 | 0% / -6.9 | +54% / -8.7 | +24% / 0.0 | +39% / -1.3 | +3% / +1.2 |
| tool-mvt256-tokens32 | +11.5% | -4.5 | +7% / -10.6 | -5% / -6.9 | +54% / -10.0 | +24% / -1.9 | +47% / -0.6 | -6% / +3.7 |
| text-whole | +6.1% | -0.5 | +14% / -2.5 | -2% / -1.9 | +5% / -1.2 | +6% / +0.6 | +5% / 0.0 | +6% / +2.5 |
| text-mvt1024 | +15.1% | -0.9 | +17% / -1.9 | +8% / -6.2 | +11% / -5.6 | +9% / +3.1 | +11% / -1.3 | +24% / +5.6 |
| text-mvt256 | +18.2% | -2.1 | +7% / -4.4 | +29% / -7.5 | +23% / -3.7 | +12% / 0.0 | +13% / -1.3 | +19% / +5.0 |
| text-mvt1024-tokens32 | +17.1% | -1.7 | +26% / -5.0 | +5% / -8.1 | +10% / -0.6 | +13% / +3.1 | +10% / -2.5 | +25% / +6.2 |
| text-mvt256-tokens32 | +11.8% | -2.1 | +20% / -3.7 | -1% / -6.2 | +3% / -6.2 | +4% / 0.0 | +4% / -1.9 | +23% / +4.4 |
| choice-whole | +2.5% | +0.9 | +17% / +3.7 | -8% / -5.0 | +5% / 0.0 | +9% / -0.6 | 0% / +1.9 | -6% / +4.4 |
| choice-mvt1024 | +72.9% | -10.7 | +66% / -39.4 | +93% / -10.0 | +39% / -1.2 | +53% / -13.1 | +32% / -3.8 | +60% / +12.5 |
| choice-mvt256 | +69.0% | -13.3 | +61% / -38.7 | +93% / -13.7 | +37% / -3.7 | +51% / -11.2 | +28% / -0.6 | +45% / +4.4 |
| choice-mvt1024-tokens32 | +8.8% | -4.7 | +8% / -13.1 | +9% / -1.9 | +18% / -1.2 | +12% / -10.6 | +15% / -5.6 | +3% / +1.9 |
| choice-mvt256-tokens32 | +18.7% | -4.1 | +11% / -9.4 | +19% / -3.1 | +32% / +2.5 | +15% / -9.4 | +21% / -10.0 | +22% / +1.9 |
| code-whole | +2.9% | +0.1 | +8% / -3.1 | +3% / -1.9 | -3% / -1.2 | -1% / -0.6 | +1% / -1.3 | +2% / +6.2 |
| code-mvt1024 | +69.3% | -6.0 | +38% / -16.9 | +95% / -13.7 | +42% / +2.5 | +55% / -2.5 | +33% / -1.3 | +52% / +6.9 |
| code-mvt256 | +68.2% | -7.2 | +38% / -21.3 | +95% / -10.6 | +50% / -1.2 | +63% / -11.2 | +42% / -1.3 | +36% / +7.5 |
| code-mvt1024-tokens32 | +39.9% | -1.3 | +17% / -1.2 | +77% / -8.1 | +25% / +0.6 | +19% / +2.5 | +17% / -1.9 | +12% / +3.7 |
| code-mvt256-tokens32 | +46.6% | -0.5 | +10% / -3.7 | +86% / -10.6 | +43% / +3.8 | +10% / -3.7 | +23% / -2.5 | +12% / +13.1 |
| math-whole | -0.3% | -1.1 | +3% / -3.7 | -7% / -2.5 | -6% / -3.7 | +6% / -0.6 | -4% / -1.3 | +4% / +3.7 |
| math-mvt1024 | +26.3% | -2.6 | +45% / -5.6 | +14% / -12.5 | +25% / -1.9 | +22% / +1.3 | +15% / -1.9 | +22% / +8.7 |
| math-mvt256 | +32.6% | -3.6 | +53% / -6.9 | +20% / -13.7 | +25% / -2.5 | +20% / -0.6 | +17% / -3.1 | +30% / +8.1 |
| math-mvt1024-tokens32 | +22.2% | -1.6 | +36% / -1.2 | +14% / -6.9 | +9% / -1.2 | +13% / +0.6 | +10% / -3.1 | +25% / +3.1 |
| math-mvt256-tokens32 | +13.8% | -4.0 | +29% / -6.9 | +7% / -9.4 | +14% / -1.9 | +10% / 0.0 | +10% / -5.0 | +5% / +2.5 |

BFCL aggregated (from `track.py gridtable`; the "average" column equals the overall column above, rounded): tool-whole +6% / -0.8; tool-mvt1024 +45% / -0.8; tool-mvt256 +46% / +0.6; tool-mvt1024-tokens32 +39% / -3.3; tool-mvt256-tokens32 +42% / -4.2; text-whole +5% / -0.2; text-mvt1024 +10% / -1.2; text-mvt256 +16% / -1.7; text-mvt1024-tokens32 +11% / +0.0; text-mvt256-tokens32 +3% / -2.7; choice-whole +4% / +0.4; choice-mvt1024 +41% / -6.0; choice-mvt256 +39% / -5.2; choice-mvt1024-tokens32 +15% / -5.8; choice-mvt256-tokens32 +22% / -5.6; code-whole -1% / -1.0; code-mvt1024 +43% / -0.4; code-mvt256 +51% / -4.6; code-mvt1024-tokens32 +20% / +0.4; code-mvt256-tokens32 +25% / -0.8; math-whole -1% / -1.9; math-mvt1024 +21% / -0.8; math-mvt256 +21% / -2.1; math-mvt1024-tokens32 +11% / -1.2; math-mvt256-tokens32 +12% / -2.3.

#### E3.3 Observations recorded in the log (failure modes and comparisons)

- Whole (uncut) runs on matched data: overall -1.1 to +0.9. On the earlier (Track 2b, old) data the corresponding uncut runs were: tool (Hermes) -3.3 (IFEval -7.5, MBPP+ -9.4); code -3.5 (IFEval -8.1).
- tool-mvt1024 shortens BFCL 38-55% with spill-over onto other benchmarks of +2% to +8%; the earlier Hermes-data tool run had MBPP+ +68%, IFEval +26%, overall +42.0%. The log's untested hypothesis: strength of the lesson (old targets 291 -> 163, 44% removed, ~130 tokens/trace, 258 traces; new 201 -> 142, 30%, ~60 tokens, 189 traces); tool-mvt256 (40% removed) does spread to code (+30%). Prompt format excluded as cause (Hermes questions were already in BFCL format).
- Earlier Hermes tool runs all cost MBPP+ 7.5-10 points, uncut included (-9.4); matched ToolACE uncut +0.6. Untested hypothesis in log: easy Hermes questions, a quarter synthetic "no tool fits" pairs, cut to almost no thinking.
- Tool, 32-token batches: targets cut -47% to -56%, BFCL irrelevance -8.7 and -10.0 (model calls a function when none fits).
- Text: every cut setting shortens IFEval by about a quarter with IFEval +4.4 to +6.2; every one costs MBPP+ 6-8 points.
- Choice (multiple choice) paragraph cuts keep about one paragraph (-78% target tokens). Trained alone, the brevity spreads to all benchmarks (MBPP+ 93% shorter, SVAMP 66%) with SVAMP -39.4, MBPP+ -10.0, BFCL multiple -13.1. Token batches cut targets -84% yet model shortens only +8.8% and SVAMP -13.1. Log cites the night-1 configuration of MC cuts + GSM8K whole x2 (+37.1% / -0.6 at 8 samples).
- Code paragraph cuts: only 180/262 and 140/231 cuts pass the check; code thinking is one long plan, the paragraph cut often keeps no thinking at all; MBPP+ 95% shorter, SVAMP -16.9 / -21.3. Night-1 code paragraph result cited: MBPP+ +93% / -13.7. Code tokens32: MBPP+ 77-86% shorter for -8.1 / -10.6 (night 1: +57% / -5.6). code-mvt256-tokens32 overall -0.5 contains IFEval +13.1 (log: likely partly noise).
- Math: only 60 easy GSM8K rows. math-mvt1024 SVAMP +45% / -5.6 vs night-1 all-GSM8K +80% / -16.2; still MBPP+ -12.5.
- Every cut run costs MBPP+ (-2 to -14) whatever kind is trained.
- Control run "last night's Hermes tool data under tonight's settings" was queued then dropped by the user before running (commits 939d792, b07e9aa); no result.

### B4. Extras: irrelevance kept whole, and 8-sample rechecks (2026-10-21)

Method: (a) tool-mvt256 with BFCL-irrelevance-style rows kept whole (`--keep-whole`), 96 of 189 cut, targets 201 -> 159; 4 samples. (b) 8-sample rechecks (240 x 8 vs `s8-base`) of four grid adapters.

| run | samples | overall saved | overall acc | SVAMP | MBPP+ | BFCL irr | BFCL mult | BFCL simple | IFEval |
|---|---|---|---|---|---|---|---|---|---|
| tool-mvt256, irrelevance kept whole | 4 | +12.4% | -1.4 | +10% / -1.2 | -1% / -9.4 | +25% / -4.4 | +30% / +1.9 | +39% / +2.5 | +5% / +5.0 |
| tool-mvt256 | 8 | +29.3% | +0.0 | +5% / +2.2 | +42% / -4.1 | +43% / +3.4 | +54% / -4.4 | +33% / +0.9 | +19% / +1.9 |
| tool-mvt1024 | 8 | +20.5% | -0.4 | +9% / +0.3 | +16% / -0.3 | +42% / -3.4 | +49% / -5.0 | +38% / +1.2 | +8% / +0.9 |
| text-mvt1024 | 8 | +14.0% | -0.8 | +28% / 0.0 | +3% / -3.7 | +16% / -1.6 | +8% / -2.5 | +11% / -1.3 | +12% / +2.2 |
| code-mvt1024-tokens32 | 8 | +37.5% | -2.7 | +14% / -4.1 | +78% / -7.5 | +25% / -0.9 | +15% / -0.9 | +17% / -3.4 | +2% / +2.5 |

4-sample vs 8-sample overall for the rechecked runs:

| run | 4 samples | 8 samples |
|---|---|---|
| tool-mvt256 | +25.8% / +0.6 | +29.3% / +0.0 |
| tool-mvt1024 | +17.6% / +1.7 | +20.5% / -0.4 |
| text-mvt1024 | +15.1% / -0.9 | +14.0% / -0.8 |
| code-mvt1024-tokens32 | +39.9% / -1.3 | +37.5% / -2.7 |

Keeping irrelevance whole: irrelevance -4.4 and saving reduced; the variant was dropped.

### B5. Combined models (2026-10-21)

Method: one SFT adapter (same settings as grid) from the untouched model, trained on a pool of per-kind rows, capped at 150 rows per source (`--max-per-source 150`), math rows whole and shown twice (x2). Three pools:
- light: tool-mvt256 rows + text-mvt1024 rows + code whole (1,308-token targets) + math whole x2; ~500 rows (FINDINGS_TABLE says ~550).
- codecut: as light but code cut by MVT c=1024 tokens32 (824-token targets); 549 rows.
- full: codecut + multiple-choice MVT c=1024 paragraphs (55 ARC + 148 CSQA, ~95 tokens); 752 rows.

Evaluation: 240 questions x 4 samples vs `s4-base`; codecut also 240 x 8 vs `s8-base`. The codecut model was chosen among the 3 on the same 240 questions it is reported on (report caveat).

| model | rows | samples | overall saved | overall acc | SVAMP | MBPP+ | BFCL irr | BFCL mult | BFCL simple | IFEval |
|---|---|---|---|---|---|---|---|---|---|---|
| light | ~500 | 4 | +19.1% | -1.5 | +13% / -3.1 | -6% / -6.9 | +37% / -1.2 | +52% / +1.3 | +41% / +1.2 | +18% / +3.7 |
| codecut | 549 | 4 | +48.3% | -1.2 | +20% / -5.0 | +75% / -9.4 | +50% / -3.7 | +53% / +0.6 | +47% / +1.2 | +28% / +10.0 |
| full | 752 | 4 | +51.9% | -2.7 | +16% / -3.7 | +81% / -11.2 | +43% / -1.9 | +56% / 0.0 | +47% / -0.6 | +34% / +5.0 |
| codecut | 549 | 8 | +44.3% [39.4, 49.6] | -1.6 [-3.9, +0.9] | +15% / -4.1 | +73% / -7.2 | +51% / +0.6 | +48% / -1.6 | +42% / +0.6 | +20% / +5.0 |

codecut 8-sample per-benchmark intervals (from report; SVAMP and MBPP+ rows not recorded there):

| benchmark | tokens saved [95%] | accuracy [95%] |
|---|---|---|
| BFCL irrelevance | +51.2% [+38.3, +61.6] | +0.62 [-5.62, +6.25] |
| BFCL multiple | +48.3% [+41.4, +54.9] | -1.56 [-9.38, +7.50] |
| BFCL simple | +42.2% [+34.4, +49.7] | +0.62 [-2.81, +5.31] |
| IFEval | +19.6% [+9.6, +30.8] | +5.00 [+0.94, +9.69] |

Full evaluation (2,759 questions x 8) of codecut was started via `fullbest` and stopped at the user's request: at the observed pace the untouched model alone would have taken ~16 h, past the runtime's 24 h limit. No full-evaluation result.

Reference figures cited in the log for the same 240-question subset: night-1 configuration (MC MVT + math whole x2) +37.1% / -0.6 at 8 samples. External: Lightning McQwen3 (blog.basta.one/blog/thinkingcap, 14 Oct) 43.2% / -0.85 without its BFCL think gate, self-reported, question count not stated.

Artifacts: Google Drive folder `thinkingcap-mvt-pertype/` (1.3 GB): adapters grid/*, combine-light, combine-codecut, combine-full, runs/, track_log.txt. Code state tagged `mvt-pertype-v1`. Report: REPORT_MVT_PERTYPE.txt.

---

### B6. Track 3b, round 1: on-policy MVT cuts + DPO (2026-10-21)

Method: start = combine-codecut adapter (kept untouched). Fresh training questions (`prepare_types --seed 1 --exclude` the grid's questions): toolace 267, ifeval_like 300, mbpp_train 295, gsm8k_all 300 (1,162). The current model answers each 4 times. MVT cut per kind on its correct answers: tool c=256 paragraphs; text c=1024 paragraphs; code and math c=1024 32-token batches; plus "early" cuts at half the MVT cut point. Every cut checked with 4 natural answers from the cut (all 4 must be right). Pairs: shorten (safe cut > full trace), guard (safe cut > earlier cut that failed), hard (full trace > failed cut, when no cut is safe). DPO from the current adapter, reference = itself, beta 0.1, lr 2e-5, 16 pairs per step, 722 pairs, 46 steps, checkpoints at 20 and 40.

The current model's answers on the round-1 questions (4 each; untouched model on the grid's questions in brackets):

| source | correct | mean tokens |
|---|---|---|
| gsm8k_all | 62% | 1,812 |
| ifeval_like | 72% (78%) | 331 (408) |
| mbpp_train | 29% (38%) | 744 (2,842) |
| toolace | 60% (58%) | 128 (265) |

| kind | cuts checked / passed | targets (MVT / early / original) | tokens before -> after | pairs (shorten / guard / hard) |
|---|---|---|---|---|
| tool | 1,284 / 1,107 | 180 (39 / 134 / 7) | 117 -> 49 (-58%) | 183 (139 / 37 / 7) |
| text | 1,742 / 1,309 | 252 (38 / 181 / 33) | 251 -> 125 (-50%) | 276 (207 / 36 / 33) |
| code | 684 / 461 | 119 (27 / 57 / 35) | 515 -> 326 (-37%) | 131 (76 / 20 / 35) |
| math (easy, <= 700 tokens) | 876 / 726 | 119 (109 / 2 / 0) + 8 truncated | 790 -> 93 (-88%) | 135 (119 / 16 / 0) |

DPO training: chosen ranked above rejected 75% (step 20) -> 94% (step 40); margins +11 -> +20.

Evaluation: 240 questions x 4 samples vs `s4-base` (start combine-codecut on same test: +48.3% / -1.2).

| checkpoint | overall saved | overall acc | SVAMP | MBPP+ | BFCL irr | BFCL mult | BFCL simple | IFEval |
|---|---|---|---|---|---|---|---|---|
| step 20 | +44.1% | -2.5 | +31% / -3.1 | +68% / -9.4 | +43% / -6.2 | +36% / -3.1 | +40% / -1.9 | +28% / +6.2 |
| step 40 | +50.2% | -3.5 | +44% / -2.5 | +67% / -8.7 | +54% / -23.1 | +32% / -1.2 | +41% / -3.1 | +42% / +6.2 |
| final (46) | +51.2% | -6.1 | +51% / -5.0 | +65% / -11.2 | +49% / -39.4 | +32% / -2.5 | +37% / -3.1 | +46% / +6.9 |

Failure mode recorded: BFCL irrelevance falls (-23.1, then -39.4) as steps increase: the model calls a function when none fits; log attributes this to DPO pushing shorter thinking everywhere, dropping the "does any function fit?" check, with only 37 guard and 7 hard tool pairs as counterweight. SVAMP -2.5 to -5.0 at 31-51% shorter.

### B7. Track 3b, round 1 retrain (r1b) (2026-10-21)

Method: same round-1 pairs minus the 52 shorten pairs on irrelevance-type questions (670 pairs, 42 steps), NLL anchor weight 1 on the chosen side, checkpoints every 10 steps; same DPO settings, start = combine-codecut (`redpo --no-shorten irrelevance --nll 1 --save-every 10`). Evaluation: 240 x 4 vs `s4-base`; step 20 also 240 x 8 vs `s8-base` (reused from Drive).

| checkpoint | samples | overall saved | overall acc | SVAMP | MBPP+ | BFCL irr | BFCL mult | BFCL simple | IFEval |
|---|---|---|---|---|---|---|---|---|---|
| step 10 | 4 | +44.9% | -3.2 | +23% / -5.0 | +65% / -7.5 | +54% / -1.9 | +53% / -1.9 | +47% / +1.2 | +29% / +0.6 |
| step 20 | 4 | +42.8% | -0.3 | +29% / +2.5 | +60% / -11.2 | +53% / -5.0 | +42% / -1.2 | +48% / -2.5 | +29% / +10.6 |
| step 30 | 4 | +64.4% | -11.4 | +83% / -35.0 | +66% / -16.2 | +49% / -11.2 | +47% / -2.5 | +52% / -1.3 | +44% / +10.6 |
| step 40 | 4 | +88.8% | -19.3 | +98% / -50.0 | +94% / -19.4 | +76% / -25.0 | +77% / +4.4 | +76% / +2.5 | +57% / -1.9 |
| final (42) | 4 | +92.0% | -19.5 | +98% / -51.9 | +97% / -20.0 | +78% / -29.4 | +81% / +3.1 | +78% / +1.9 | +68% / +1.9 |

8-sample comparison (240 x 8 vs `s8-base`):

| model | tokens saved [95%] | accuracy [95%] | SVAMP | MBPP+ | BFCL irr | BFCL mult | BFCL simple | IFEval |
|---|---|---|---|---|---|---|---|---|
| combine-codecut (start) | +44.3% [39.4, 49.6] | -1.6 [-3.9, +0.9] | +15% / -4.1 | +73% / -7.2 | +51% / +0.6 | +48% / -1.6 | +42% / +0.6 | +20% / +5.0 |
| r1b step 20 | +47.9% [43.6, 53.1] | -2.3 [-5.0, +0.5] | +36% / -3.7 | +72% / -4.1 | +59% / -10.6 | +41% / -1.9 | +44% / +0.3 | +22% / +2.8 |

Failure modes recorded: past step 20 the model stops thinking (up to 92% shorter) and SVAMP falls to -35.0 / -50.0 / -51.9. The log notes the NLL anchor is applied to the chosen (short) side and so pulls in the same direction. Removing irrelevance shorten pairs held irrelevance at -1.9 / -5.0 through step 20, then -11.2 to -29.4. The 4-sample step-20 overall of -0.3 includes IFEval +10.6; at 8 samples step 20 is +47.9% / -2.3 vs start +44.3% / -1.6 (IFEval +2.8, irrelevance -10.6).

### B8. Queued in this clone after E7, no results in the log (2026-10-21)

From `plan_onpolicy.txt` and commits 34f9b4e through a65156f; no results for any of these are in EXPERIMENT_LOG.md or FINDINGS_TABLE.md of this clone:

| item | description |
|---|---|
| fresh-question test (`recheck --fresh --samples 8`) | questions 41-80 of every benchmark part (240 new questions), 8 samples, raw model first; on mvt-fading adapters `bon-dpo-r2g/step-10` (cited in plan as +54.2% / -1.6 at 8 answers on the usual subset) and `bon-dpo-nocode/step-30`. combine-codecut was dropped from this test by the user |
| thinkcheck | share of near-empty thinking per benchmark, raw vs trained, on fresh and usual subsets (no GPU) |
| cod (Chain-of-Draft code repair) | from `bon-dpo-r2g/step-10`; 150 MBPP-train + 300 KodCode tasks; few-shot code drafts with an explicit test step (one failing check + fix), no word cap; draft and normal replies pooled; keep the 25th-percentile passing reply with a real check and no exemplar copy; LoRA SFT 1 epoch lr 1e-4; replay rows for other tasks; MBPP challenge asserts in grading |
| codeeval | code alone on all 378 MBPP+ tasks x 4 for the raw model, the start and the repaired model |

---

## Part C — Tracks 2f and 2g, first half (22–25 Oct)

Source: thinkingcap/EXPERIMENT_LOG.md lines 1003–1650. Numbers copied as logged. Format "+S / A" = tokens saved (%) / accuracy change (pts) vs the original Qwen3-0.6B. "—" = not logged.

Evaluation sets (all 6 benchmarks, 40 questions each = 240 questions):
- usual = questions_40 (the original subset); fresh = a second, unseen 240-question subset; choosing set = questions_val20261023 (240). "x4 / x8" = samples per question. Logged run-to-run noise: same model on fresh x8 scored -2.9 and -3.8 on two runs (~1 pt); ±1.5 pts per x8 on 240 questions.
- Lines: A = MBPP-free line from clean-a2; B = MBPP-free line from clean-b2 (+ KodCode). "Soup" = average of 3 DPO seeds at a given step.

---

### C1. Track 2f setup (queued 22 Oct)

Constraint set: start only from MBPP-free families A and B. `track.py chain`: each round draws new questions (ToolACE, IFEval-like, ARC/CSQA, GSM-Plus, GSM-IC; minus earlier draws), + new KodCode tasks on line B. Contamination check: drop at >= 30% 8-gram share OR any exact 13-word run (boilerplate lines shared by >= 50 eval questions ignored). Pairs built as the earlier "balanced round"; DPO with 3 seeds; selection rule = soup with most saving at 4-sample accuracy change >= -1.5; it becomes next round's base (fresh LoRA on top). Chosen soups scored x8 on usual and fresh.

Queued variants:
- `repair` on round 4 (rep4) and round 5 (rep5): weight blends with round 2 (75/25, 50/50); length-matched right-beats-wrong DPO on new trap math (right/wrong replies within 25% length); RFT = SFT on own right replies <= typical (median) length.
- v2 from scratch (`chain --start raw`, v2r1–v2r4): no MBPP; adds verified MVT cuts (4/4 re-answers) as short-beats-long pairs for tools and instructions; KodCode from a wider slice (task <= 70, solution <= 130 tokens), tasks never repeated within a line.

#### Contamination audit (runs/audit/contamination_report.txt; every training question vs 2,759 eval questions)

| source | >= 30% 8-gram | >= 50% | exact 13-word run | note |
|---|---|---|---|---|
| GSM8K, GSM-Plus, GSM-IC, ToolACE, ARC, CSQA, Skywork | 0 | 0 | 0 (1 in Skywork, unused) | |
| MBPP (old lineage only) | — | ~9 of 300 per draw | ~39 per draw | MBPP contains MBPP+ problems under other ids (e.g. change_date_format ~ Mbpp/427, max_product ~ Mbpp/415). 50% check dropped the worst; weaker near copies may have reached gentle SFT of rounds 1–5 and the balanced model |
| IFEval-like | — | 0–2 of 300 | ~190 | shared fixed constraint wording, not questions; chain's strict check drops them |
| KodCode | — | 0 | 64 | dropped by strict check |

---

### C2. Clean line A, round 3 (ca1, from A2), 22–23 Oct

Full chain recipe, 3-seed soups. Eval questions_40 x4 per checkpoint.

| ckpt | saved | acc | notes |
|---|---|---|---|
| step 10 (chosen, floor -1.5) | +52.5% | -1.0 | |
| step 20 | +54.6% | -3.0 | |
| step 30 | +49.2% | -3.0 | |
| step 40 | +54.2% | -2.9 | svamp +60% / -2.5, ifeval +60% / +1.9, bfcl +39–47%, mbpp +51% / -10.6 |

First x8 tests lost with the runtime (half-written folder refused by evaluate.py; track.evaluate now clears it); requeued.

ca1 step 10, x8:

| set | overall | svamp | mbpp | ifeval | bfcl |
|---|---|---|---|---|---|
| usual x8 | +49.7 / -2.3 | — | — | — | — |
| fresh x8 | +48.6 / -1.8 | +58% / -4.1 | +53% / -2.8 | +46% / +0.3 | +33–36% / — |

### C3. Clean line B, round 3 (cb1, from B2), 22–23 Oct

Full recipe + KodCode right-beats-wrong; 787 pairs; 3-seed soups. questions_40 x4:

| ckpt | saved | acc | per-benchmark |
|---|---|---|---|
| step 10 | +42.9% | -1.5 | |
| step 20 | +41.6% | +0.8 | svamp +55% / +0.6, ifeval +49% / +6.9, bfcl +31–35%, mbpp +24% / -5.0 |
| step 30 | +41.9% | +1.0 | |
| step 40 (chosen) | +44.0% | +0.9 | |

cb1 step 40, x8:

| set | overall | svamp | mbpp | ifeval | bfcl |
|---|---|---|---|---|---|
| usual x8 | +42.0 / -0.6 | +55% / -3.1 | +29% / -0.9 | +45% / +0.3 | +30–38% / — |
| fresh x8 | +41.0 / -1.1 | +54% / -1.6 | +34% / +0.3 | +41% / -2.5 | +28–33% / — |

### C4. Clean line A, round 4 (ca2, from ca1 step 10; 675 pairs)

questions_40 x4:

| ckpt | saved | acc | per-benchmark |
|---|---|---|---|
| step 10 | +49.8 | -2.9 | |
| step 20 (chosen; none reached -1.5, most accurate) | +52.8 | -1.9 | svamp +60% / 0.0, mbpp +48% / -13.7, ifeval +60% / +5.0 |
| step 30 | +54.1 | -3.5 | |
| step 40 | +56.3 | -2.2 | |

ca2 step 20, x8:

| set | overall | svamp | mbpp | ifeval | bfcl |
|---|---|---|---|---|---|
| usual x8 | +53.5 / -2.4 | — | — | — | — |
| fresh x8 | +46.4 / -2.7 | +56% / -5.3 | +51% / -1.9 | +39% / -3.1 | +35–40% / — |

Observed: saving drops 7 pts from usual to fresh.

### C5. Clean line B, round 4 (cb2, from cb1 soup; KodCode right-beats-wrong; 3-seed soups)

| ckpt | set | saved | acc |
|---|---|---|---|
| step 40 | questions_40 x4 | +47.1 | +1.4 |
| step 30 (chosen) | usual x8 | +51.0 | -0.3 |
| step 30 (chosen) | fresh x8 | +46.9 | -2.6 |

Line B ended after round 4.

### C6. Repair of round 4 (rep4) and round 5 (rep5) (grey-zone lineage; rounds 1–5 had MBPP exposure in gentle SFT)

Candidates: weight blends with round 2; length-matched right-beats-wrong DPO (step 10); RFT (SFT on own right replies <= median length).

rep4 (usual x4 | fresh x8):

| candidate | usual x4 | fresh x8 | fresh per-benchmark |
|---|---|---|---|
| blend 75% r4 + 25% r2 | +66.5 / -2.5 | +61.5 / -4.0 | svamp +74% / -8.8, mbpp +67% / -4.4, ifeval +55% / -1.9, bfcl +39–43% |
| blend 50/50 | +57.8 / -2.2 | +57.3 / -2.4 | — |
| length-matched right-beats-wrong DPO, step 10 | +64.2 / -1.5 | +61.3 / -4.2 | — |
| RFT | +67.5 / -2.1 | +63.4 / -5.1 | — |

rep5 (usual x4 | fresh x8):

| candidate | usual x4 | fresh x8 | fresh per-benchmark |
|---|---|---|---|
| blend 75% r5 + 25% r2 | +67.1 / -2.3 | +63.5 / -3.9 | svamp +75% / -9.4, mbpp +69% / -4.1, ifeval +59% / -1.2, bfcl +39–44% |
| blend 50/50 | +62.7 / -3.0 | +58.8 / -3.4 | svamp +72% / -5.9, mbpp +65% / -3.1, ifeval +50% / -4.1, bfcl +38–42% |
| length-matched right-beats-wrong, step 10 | +65.0 / -1.5 | +61.9 / -4.5 | svamp -7.8, ifeval -6.2 |
| RFT | +69.4 / -0.2 | +67.1 / -5.5 | svamp +78% / -10.0, ifeval -5.6, bfcl_multiple -5.3 |

Observed: every candidate at 60%+ saved loses 4–5.5 pts on fresh; usual-set accuracy is 2–5 pts higher than fresh for these (RFT -0.2 vs -5.5). SVAMP is the largest per-benchmark loss in every row where logged.

### C7. v2 from scratch (v2r1–v2r3), 23–24 Oct

v2r1: from raw model; seed-21 draw, 1,500 questions + 300 KodCode x8; chain recipe + KodCode right-beats-wrong + verified MVT cuts (tools, instructions); 3-seed soups. Each later round starts from the previous chosen soup (v2r2 seed 22, v2r3 seed 23). ~4–5 h per round.

| model | set | overall | bfcl irr | bfcl mult | bfcl simple | ifeval | mbpp | svamp |
|---|---|---|---|---|---|---|---|---|
| v2r1 step 30 (chosen) | questions_40 x4 | +19.3 / +1.5 | | | | | | |
| v2r1 step 40 | questions_40 x4 | +18.7 / +3.1 | | | | | | |
| v2r1 step 30 | usual x8 | +16.3 / +0.8 | +13–18% / 0..+0.6 (pooled) | | | +20% / +0.6 | +8% / -0.3 | +21% / +2.5 |
| v2r1 step 30 | fresh x8 | +14.8 / -0.2 | +16% / -1.9 | +14% / -3.1 | +14% / -2.8 | +15% / +1.6 | +6% / +1.6 | +23% / -1.2 |
| v2r2 step 10 | x4 | +20.9 / +0.2 | | | | | | |
| v2r2 step 20 | x4 | +23.6 / -0.6 | | | | | | |
| v2r2 step 30 | x4 | +27.3 / +0.6 | | | | | | |
| v2r2 step 40 (chosen) | x4 | +28.6 / -0.6 | | | | | | |
| v2r2 step 40 | usual x8 | +31.2 / -0.9 | | | | | | |
| v2r2 step 40 | fresh x8 | +28.6 / -0.8 | +25% / -2.2 | +25% / -3.7 | +22% / +0.6 | +33% / +0.6 | +21% / -0.3 | +35% / -1.6 |
| v2r3 step 40 (chosen) | usual x4 | +35.6 / -0.1 | | | | | | |
| v2r3 step 40 | fresh x8 | +28.3 / -1.9 (acc CI -3.6..0.0; saving 24–32%) | | | | | | |

v2 stopped after round 3 (user); rr4 removed from queues.

---

### C8. Track 2g: rc1 cut calibration (23 Oct)

Model ca1 (models/ca1-soup-10) answered a new draw (seed 31): 1,000 questions + 300 KodCode, x8. 904 right replies kept (longest right per question). Each reply cut at 1/5/10/20/35/50/65/80% of its thinking, "</think>" closed there, 20 re-answers per cut = 144,640 answers; 2,901 of 7,232 cuts right 20/20. Confidence = per-token geometric-mean probability of the full reply's answer, teacher-forced after the cut (uses the reference answer).

Confidence bin -> real share right (n cuts):

| source | bins | ECE |
|---|---|---|
| arc_easy | 0.6–0.8 -> 0.18 (32); 0.8–0.9 -> 0.57 (16); 0.9–1.0 -> 0.99 (225) | 0.09 |
| commonsense_qa | 0.0–0.2 -> 0.41 (36); 0.2–0.6 -> 0.00–0.01 (169); 0.6–0.8 -> 0.25 (90); 0.8–0.9 -> 0.61 (102); 0.9–1.0 -> 0.99 (483) | 0.16 |
| gsm_ic | 0.0–0.6 -> 0.00 (975); 0.6–0.8 -> 0.05 (149); 0.8–0.9 -> 0.22 (87); 0.9–1.0 -> 0.80 (453) | 0.34 |
| gsm_plus | 0.0–0.6 -> 0.00 (807); 0.6–0.8 -> 0.13 (134); 0.8–0.9 -> 0.32 (27); 0.9–1.0 -> 0.86 (216) | 0.34 |
| ifeval_like | 0.2–0.4 -> 0.84 (132); 0.4–0.6 -> 0.78 (330); 0.6–0.8 -> 0.87 (48); 0.8–0.9 -> 0.84 (21); 0.9–1.0 -> 0.94 (21) | 0.31 |
| kodcode | 0.4–0.6 -> 0.58 (35); 0.6–0.8 -> 0.63 (287); 0.8–0.9 -> 0.64 (496); 0.9–1.0 -> 0.87 (926) | 0.13 |
| toolace | 0.0–0.2 -> 0.17 (69); 0.2–0.4 -> 0.57 (77); 0.4–0.6 -> 0.53 (71); 0.6–0.8 -> 0.61 (81); 0.8–0.9 -> 0.54 (120); 0.9–1.0 -> 0.92 (510) | 0.11 |

Stopping rules per source (allowance: math 0.5 pt, code 1, others 2):

| source | confidence threshold c | saved / real loss | outcome rule (earliest cut right 20/20): saved / loss |
|---|---|---|---|
| arc_easy | 0.9 | 83% / 1.3 | 80% / 0.0 |
| commonsense_qa | 0.95 | 54% / 0.2 | 56% / 0.0 |
| gsm_ic | none | — | 30% / 0.0 |
| gsm_plus | none | — | 26% / 0.0 |
| ifeval_like | 0.7 | 7% / 1.7 | 48% / 0.0 |
| kodcode | none | — | 62% / 0.0 |
| toolace | none | — | 58% / 0.0 |

Observed: math over-confident (0.98 conf -> 0.80–0.86 right); instructions flat (0.3–0.9 conf all ~0.8 right).

First rc1 pairs (confidence only: 142 cut pairs, mostly multiple choice) were dropped. Rerun with per-type rule: confidence for arc, csqa, ifeval-like; outcome rule for gsm_ic, gsm_plus, kodcode, toolace.

Cut pairs (shorten / hard): gsm_ic 110 / 140, gsm_plus 74 / 92, kodcode 169 / 100, toolace 89 / 33, arc 32 / 3, csqa 71 / 26, ifeval-like 7 / 3; + guard pairs (bon_pairs, math shorten 0) + 133 KodCode right-beats-wrong.

Process notes: serial grading of 144,640 answers > 1 h (code tests, GPU idle) -> check_vllm with 16 threads (~15 min); a killed check_vllm left an orphan vLLM engine holding 10.5 GB (RL smoke test OOM).

### C9. rc1 cal-cut DPO (from ca1), 23 Oct

1,489 training pairs (48 held out): shorten 691, guard 451, hard 397 (mix 0.45 / 0.29 / 0.26); 3 seeds, soups; seed 0 = 94 steps, pairs ordered right 71–86%.

Choosing set (questions_val20261023) x4:

| model | overall | per-benchmark |
|---|---|---|
| ca1 (start, reference) | +44.1 / -1.8 | mbpp +56% / -6.2, svamp +49% / +0.6, ifeval +30% / 0.0, bfcl +34–41% / -3.1..-0.6 |
| soup step 10 | +52.5 / -0.1 | |
| soup step 20 | +54.0 / -0.8 | |
| soup step 30 (chosen) | +55.1 / -0.6 | svamp +63% / -2.5, mbpp +66% / +1.3, ifeval +40% / -1.3, bfcl +40–50% / -1.2..+1.9 |

rc1-soup-30 x8:

| set | overall | bfcl irr | bfcl mult | bfcl simple | ifeval | mbpp | svamp |
|---|---|---|---|---|---|---|---|
| usual x8 | +58.0 / -3.1 | +48% / -4.4 | +45% / +3.1 | +38% / -0.3 | +53% / -0.6 | +61% / -6.9 | +69% / -4.4 |
| fresh x8 | +55.6 / -2.9 | +46% / -4.4 | +42% / -5.0 | +39% / +0.3 | +45% / -1.9 | +64% / -1.6 | +66% / -5.0 |

vs ca1 on fresh (+48.6 / -1.8): +7.0 saving, -1.1 accuracy; mbpp 53% / -2.8 -> 64% / -1.6; svamp -4.1 -> -5.0; bfcl ~-0.6 -> -3.0 pooled. Choosing set x4 had shown -0.6 for the same model.

### C10. rc1 RL (from rc1-soup-30), 23 Oct

Reward = right x (1 - alpha_q · sigmoid(z_thinking)), alpha 0.15 x solve rate; batch-normalized advantages; KL 1e-3; lr 1e-5; LoRA r16; 32 questions x 8 per step; pool = 782 round questions solved >= 4/8 by ca1, repeated ~every 24 steps. First steps: 80–89% right on training answers, thinking 200–340 tokens.

Choosing set x2 quick checks:

| block | overall | per-benchmark |
|---|---|---|
| block 1 (step 20) | +54.2 / -0.2 | svamp +59% / -2.5, mbpp +64% / 0.0, ifeval +44% / +2.5, bfcl +43–49% / -3.7..+1.2 |
| block 2 (step 40) | +51.3 / +0.9 | 5 of 6 parts exactly equal to base accuracy (flagged suspicious) |
| block 3 (step 60) | +48.5 / -0.6 | svamp +49% / -1.2, mbpp +56% / +1.3, ifeval +42% / -2.5, bfcl +42–50% / -3.7..+3.7 |

Training-answer thinking rose from ~200–250 tokens (block 1) to 400–500 (block 4) at 82–96% right. Block 2 hit CUDA OOM on a long reply (vocabulary-wide log-probs per position; fixed with chunked, recomputed log-probs). RL stopped after block 3.

All rc1 versions on fresh x8 (fresh set also used for selection from here, user's call):

| version | overall | svamp | mbpp | ifeval | bfcl irr | bfcl mult | bfcl simple | choosing set x4 |
|---|---|---|---|---|---|---|---|---|
| DPO soup 10 | +48.6 / -3.0 | +62% / -6.2 | +55% / -2.2 | +35% / -1.9 | +38% / -3.4 | +38% / -2.2 | +34% / +0.3 | — |
| DPO soup 20 | +54.8 / -4.0 | +65% / -5.6 | +63% / -4.7 | +46% / -3.1 | +43% / -4.7 | +40% / -3.7 | +37% / +0.6 | — |
| DPO soup 30 | +55.6 / -2.9 | +66% / -5.0 | +64% / -1.6 | +45% / -1.9 | +39–46% / -5.0..+0.3 (pooled) | | | — |
| + RL 20 steps | +54.8 / -3.7 | +65% / -6.6 | +63% / -1.3 | +45% / -4.4 | +45% / -3.7 | +41% / -3.7 | +38% / 0.0 | +57.3 / -0.2 |
| + RL 40 steps (rc1-rl-40) | +53.9 / -2.1 | +65% / -4.1 | +59% / -1.9 | +46% / -0.6 | +45% / -1.2 | +41% / -4.1 | +39% / -0.3 | +54.8 / -0.9 |
| + RL 60 steps | +49.6 / -3.4 | +58% / -3.7 | +58% / -4.7 | +38% / -2.8 | +44% / -2.2 | +42% / -4.4 | +39% / -0.3 | — |

Observed: choosing set x4 was 1.2–3.5 pts higher in accuracy than fresh x8 for every version. Spread -2.1..-4.0 across the seven versions.

### C11. MBPP audit of the gentle best-of-8 SFT rows

74 MBPP rows in runs/bon/select-r1/sft.jsonl (the only MBPP exposure of the rounds 1–5 / balanced / repair lineage; DPO rounds used --drop-sources mbpp_train), mbpp_audit.py vs 378 MBPP+ tasks (runs/audit/mbpp74.txt):

| check | count |
|---|---|
| 8-gram share >= 30% | 0 |
| exact 13-word run | 2 |
| same function name | 1 |
| difflib word similarity >= 0.8 | 18 |

Manual read: closest pairs are sibling tasks on a shared template with a different operation (add vs multiply nested tuples; n cheapest vs n most expensive; recursive gcd vs number of divisors; sum even vs sum odd in range (0.88); rhombus vs rectangle area; first even vs first odd). No row found to be an MBPP+ task under another id.

### C12. Calibration re-checks on the same 7,232 rc1 cut points (x20 re-answers)

| model | cuts 20/20 | ECE arc | csqa | gsm_ic | gsm_plus | ifeval_like | kodcode | toolace |
|---|---|---|---|---|---|---|---|---|
| ca1 | 2,901 | 0.09 | 0.16 | 0.34 | 0.34 | 0.31 | 0.13 | 0.11 |
| rc1-soup-30 | 2,954 | 0.07 | 0.17 | 0.36 | 0.37 | 0.31 | 0.13 | 0.14 |
| rc1-rl-40 | 2,860 | 0.07 | 0.17 | 0.36 | 0.37 | 0.33 | 0.12 | 0.15 |

Outcome-rule safe cuts (% of reply), rc1-rl-40 vs ca1: arc 80 / 80, csqa 57 / 56, gsm_ic 32 / 30, gsm_plus 26 / 26, ifeval 47 / 48, kodcode 58 / 62, toolace 60 / 58. gsm_ic: conf 0.97–0.98 -> 0.78 right (both trained models). Selection rule picks the same signal per type for all three models.

Confidence vs real per cut point (cal_table.py; same for ca1, rc1-soup-30, rc1-rl-40 within 0.01–0.05):

| type | real share right by cut position | confidence |
|---|---|---|
| math traps (gsm_ic, gsm_plus) | ~0 up to 10%; 0.04–0.13 at 20%; 0.2–0.33 at 50%; 0.43–0.52 at 65%; 0.64–0.76 at 80% | 0.4–0.5 at start; 0.73–0.88 at 65–80% |
| ifeval_like | 0.76–0.84 at 1% (~half of cuts 20/20), flat to 80% | flat ~0.45–0.59 |
| tools | 0.65–0.72 at 1%; 0.84–0.85 at 80% (53–75% of cuts 20/20) | — |
| code | 0.66–0.69 at 1%; 0.87–0.91 at 80% | — |
| multiple choice | arc 0.86–0.89 at 1%, csqa 0.59–0.69 -> 0.75–0.93 at 80% | — |

### C13. rr1: RL on rc1-soup-30 with fresh pool (23 Oct)

Fresh pool seed 42: 2,846 drawn, 553 dropped by contamination check -> 2,293 questions (120 steps x 32 = 3,840 draws; each question ~1.7 times); alpha 0.3; not filtered by solve rate. First steps 54–65% right, thinking 190 -> 336 tokens over steps 1–3. Stop rule: below floor - 2 -> stop.

Fresh x8:

| model | overall | bfcl irr | bfcl mult | bfcl simple | ifeval | mbpp | svamp |
|---|---|---|---|---|---|---|---|
| rr1-start (= rc1-soup-30, re-scored) | +56.4 / -3.8 | +46% / -4.7 | +41% / -4.4 | +39% / +0.6 | +48% / -5.9 | +64% / -0.9 | +67% / -5.6 |
| after RL block 1 (20 steps) | +57.5 / -5.1 | +47% / -3.7 | +43% / -5.9 | +39% / +0.3 | +47% / -3.1 | +65% / -4.1 | +68% / -10.0 |

RL stopped by rule after one block. Same model rc1-soup-30 scored -2.9 earlier and -3.8 here on the same 240 x 8 (vLLM sampling not bit-reproducible).

### C14. rc2: second calibrated-cut round (from rc1-rl-40)

Seed 43: 869 questions + KodCode x8; 675 replies; 5,400 cuts x 20; 2,019 right 20/20; math safe cuts 19–22% (ca1: 26–34%). 1,020 pairs: shorten 472, guard 319, hard 229; 61 steps.

Fresh x8:

| soup | overall | svamp | mbpp | ifeval | bfcl irr | bfcl mult | bfcl simple |
|---|---|---|---|---|---|---|---|
| 10 | +57.5 / -4.5 | +66% / -7.8 | +67% / -5.0 | +48% / -1.6 | +47% / -3.4 | +44% / -5.0 | +40% / -1.9 |
| 20 | +63.4 / -4.9 | +71% / -9.1 | +73% / -2.8 | +57% / -4.1 | +43–51% / -5.6..-0.6 (pooled) | | |
| 30 | +63.6 / -4.9 | +70% / -10.9 | +76% / -2.8 | +54% / -2.2 | +45–52% / -6.2..-0.3 (pooled) | | |
| 60 | +67.0 / -6.2 | +71% / -9.4 | +80% / -9.7 | +56% / -2.5 | +49–58% / -5.0..-0.3 (pooled) | | |

---

### C15. Stopping probes (hidden-state judges)

#### pkrl40 (rc1-rl-40 hidden states)
Development = 904 rc1 replies' 7,232 cuts re-answered by rc1-rl-40 (9% left out as 16–19/20); held-out test = rc2's 675 new questions, 5,400 cuts. Config by nested CV: layer 21, after "</think>". Out-of-fold AUC of 8 configs 0.79–0.85.

| subset | probe test (dev) | cut position test |
|---|---|---|
| ALL | 0.864 (0.842) | 0.823 (0.801) |
| gsm_ic | 0.938 (0.951) | 0.859 |
| gsm_plus | 0.916 (0.881) | 0.862 |
| instructions | 0.736 (0.753) | 0.546 |
| tools | 0.795 (0.742) | 0.585 |
| code | 0.752 (0.729) | 0.617 |
| multiple choice | 0.53–0.61 | — |

Leave-one-type-out: math 0.89–0.90, instructions 0.52, tools 0.48, MC 0.54–0.58. Reference-based "confidence" scores 0.99–1.00 within math and MC (uses the reference answer; not available to a real stopper). Stopper (earliest cut with probe >= t, t per type from dev): ~0% saved on every type at the allowance; 5% for math at a 2-pt allowance. Oracle on same test cuts: math 19–22%, instructions 49%, code 56%, tools 61%.

#### pkca1 (ca1 own cuts and hidden states, CV only)
Layer 21 after "</think>" chosen in 5/5 folds. Out-of-fold AUC: ALL 0.844 (position 0.801); math 0.937 / 0.888 (position 0.844 / 0.883); instructions 0.729 (0.481); tools 0.764 (0.553); code 0.736 (0.646); MC 0.51–0.62. Stopper ~0–17% saved, mostly over the allowance.

---

### C16. Labelled cut lots (label_cuts.py: paragraph-end cuts of right AND wrong replies, <= 10 cuts + full thinking, 20 forced re-answers each)

Columns: success by share of thinking kept 0–20 / 20–40 / 40–60 / 60–80 / 80–<100% / full; "perfect stop" = earliest cut at least as good as the natural stop: saved / accuracy gained (optimistic bound, max over ~5–10 noisy cuts).

#### Lot A (24 Oct; rc1-rl-40; seed 47; 664 clean questions; 922 replies, 563 right; 4,838 cuts)

| source (replies) | success curve | perfect stop |
|---|---|---|
| arc_easy (10) | .59/.43/.50/.50/.50/.77 | 70% / +3.5 |
| commonsense_qa (64) | .49/.58/.33/.63/.50/.57 | 54% / +18.0 |
| gsm_ic (186) | .04/.14/.29/.46/.62/.71 | 33% / +3.8 |
| gsm_plus (113) | .01/.04/.12/.31/.50/.62 | 44% / +5.0 |
| ifeval_like (180) | .81/.68/.73/.69/.86/.68 | 17% / +2.7 |
| kodcode (247) | .57/.59/.61/.61/.58/.63 | 74% / +18.3 |
| toolace (122) | .04/.37/.29/.02/.14/.41 | 20% / +1.5 |

Zoo on lot A first crashed on runtime 2 (log(0) in cut-position feature; fixed).

#### Lot B (24 Oct; original Qwen3-0.6B; seed 48; 639 clean questions; 924 replies, 522 right; 8,255 cuts)

| source (replies) | success curve | perfect stop |
|---|---|---|
| arc_easy (13) | .67/.78/.65/.74/.81/.76 | 78% / +18.8 |
| commonsense_qa (76) | .37/.43/.50/.54/.44/.50 | 76% / +27.0 |
| gsm_ic (211) | .24/.48/.55/.55/.55/.52 | 64% / +36.9 |
| gsm_plus (119) | .20/.48/.53/.54/.49/.49 | 71% / +35.1 |
| ifeval_like (170) | .62/.67/.69/.67/.56/.69 | 56% / +9.2 |
| kodcode (247) | .65/.65/.60/.61/.61/.63 | 84% / +20.7 |
| toolace (88) | .31/.34/.37/.31/.30/.48 | 63% / +2.3 |

#### Lot B3 (25 Oct ~05:40; original Qwen; contamination check dropped 268 of 743 -> 455 questions / 792 replies (422 right) / 7,436 cuts)

| source | replies (right) | cuts | success curve | perfect stop |
|---|---|---|---|---|
| gsm_ic | 397 (200) | 4,282 | 0.25 / 0.46 / 0.47 / 0.49 / 0.52 / 0.52 | 66% / +29.5 |
| gsm_plus | 184 (84) | 2,003 | 0.21 / 0.47 / 0.47 / 0.46 / 0.50 / 0.47 | 71% / +31.8 |
| ifeval_like | 211 (138) | 1,151 | 0.77 / 0.73 / 0.75 / 0.73 / 0.66 / 0.73 | 61% / +11.4 |

#### Lot B4 (25 Oct ~07:45; code + tools, original Qwen; 647 questions / 921 replies / 7,651 cuts; contamination dropped 15; only 362 new of 850 non-KodCode questions found)

| source | replies (right) | cuts | success curve | perfect stop |
|---|---|---|---|---|
| kodcode | 466 (241) | 5,116 | 0.46 / 0.50 / 0.46 / 0.47 / 0.50 / 0.52 | 79% / +22.9 |
| hermes_tools | 301 (144) | 1,768 | 0.27 / 0.44 / 0.41 / 0.29 / 0.18 / 0.49 | 72% / +9.1 |
| toolace | 136 (67) | 569 | 0.35 / 0.46 / 0.49 / 0.41 / 0.42 / 0.54 | 46% / +6.3 |
| mbpp_train (dropped downstream) | 18 (8) | 198 | 0.26 / 0.28 / 0.34 / 0.39 / 0.42 / 0.44 | 77% / +9.2 |

#### benchlot (25 Oct ~07:00; original Qwen on the 240 choosing benchmark questions x8; measure only)

| benchmark | replies (right) | cuts | success curve | perfect stop |
|---|---|---|---|---|
| SVAMP | 51 (36) | 431 | 0.26 / 0.34 / 0.48 / 0.58 / 0.55 / 0.73 | 53% / +8.9 |
| IFEval | 53 (29) | 331 | 0.56 / 0.50 / 0.49 / 0.51 / 0.45 / 0.61 | 55% / +11.0 |
| MBPP+ | 52 (26) | 572 | 0.46 / 0.53 / 0.44 / 0.52 / 0.49 / 0.50 | 84% / +17.0 |
| BFCL | 148 (109) | 478 | 0.70 / 0.70 / 0.72 / 0.77 / 0.62 / 0.75 | 51% / +8.7 |

Observed: SVAMP success at 20–40% (0.34) is below full (0.73), unlike practice GSM-IC/Plus (lot B, B3) which reach full-level by 20–40%.

---

### C17. Probe zoos (classifier comparisons on labelled lots)

#### Zoo A (lot A, rc1-rl-40; 643 questions; 70% dev / 30% test = 279 test questions)
Layer sweep: best layer 20 after "</think>" (dev AUC 0.787; early layers ~0.70). TEST AUC (ALL):

| classifier | TEST AUC |
|---|---|
| position | 0.718 |
| answer stability | 0.672 |
| own confidence (prob. of own greedy answer) | 0.790 |
| reference-based confidence | 0.805 |
| linear | 0.803 |
| linear per type | 0.794 |
| linear on real success rate | 0.782 |
| MLP | 0.852 |
| MLP all 28 layers x 2 places | 0.843 |
| attention over last 64 tokens | 0.806–0.815 |

Per type, MLP (own confidence): math 0.91/0.91 (0.92/0.93), instructions 0.87 (0.74), tools 0.88 (0.63), code 0.80 (0.72).
Accounting bug 1: first trade-off/stopper tables counted loss against "always right" (wrong replies included); fixed to loss vs natural stop.

#### Zoo B (lot B, original Qwen; 281 held-out questions)
TEST AUC: position 0.668, own confidence 0.728, MLP 0.826, MLP all layers 0.828 (math 0.92–0.94 vs own confidence 0.87–0.88).

Pooled trade-off at <= 2 pts (report v2): zoo A linear 42% saved / 1.5 vs own confidence 25% / 0.2; zoo B all rows lost 7–12 pts even at tightest budget (position 67% / 8.9).
Accounting bug 2: natural stop was the SELECTED reply's outcome (longest right / longest wrong) while cuts used 20 re-answers; fixed to the full-thinking cut's own re-answers and length.
Corrected (zooreport): zoo A at <= 2 pts: linear 35% / 1.1 vs own confidence 12% / -0.8. Zoo B still ~10-pt loss in every row (pooled loss averaged types equally; a type with ~4 held-out replies swung it) -> pooled now counts every reply once (zooreport r2).

#### Zoo A2 (rc1-rl-40, lots A + A2 = 1,222 questions; test 540 held-out)

| classifier | TEST AUC (ALL) |
|---|---|
| position | 0.719 |
| own confidence | 0.792 |
| linear | 0.845 |
| linear per type | 0.874 |
| MLP | 0.869 |
| MLP all 28 layers x 2 places | 0.905 |

Per type, MLP all layers: math 0.91/0.88, code 0.92, instructions 0.89, tools 0.69. Pooled trade-off: <= 2 pts MLP all layers 31% / 2.4; <= 0.5 pt 26% / 1.4; own confidence 10% / 0.4.
Planned lots A3/A4 (rc1-rl-40): plotA3 failed (models/rc1-rl-40 not restored); replaced by plotB3/plotB4 for the original Qwen.

#### Zoo B2 (original Qwen, lots B + B2; dev 1,279 questions / 11,314 cuts; TEST 545 questions / 4,907 cuts)

| classifier | TEST AUC (ALL) |
|---|---|
| position | 0.699 |
| own confidence | 0.740 |
| linear | 0.843 |
| MLP | 0.864 |
| MLP all layers | 0.864 |
| attention | 0.81–0.83 |

Per type (MLP): math 0.93–0.94, instructions 0.85, code 0.78, tools 0.69, MC 0.48–0.63.
Pooled one-threshold trade-off on TEST (saved / real loss vs natural stop): <= 2 pts: MLP 69% / 1.5, linear per type 65% / -0.7, own confidence 46% / 1.1, position 51% / 2.9; <= 0.5 pt: MLP 67% / 0.5 (simulation on practice types, not the benchmark).

#### zbench (25 Oct ~08:20): original-Qwen judge trained on lots B+B2+B3 (2,616 questions / 23,657 cuts), TESTED on the 240 benchmark questions (1,812 cuts)

| classifier | dev AUC | TEST AUC (benchmark) |
|---|---|---|
| MLP | 0.893 | 0.736 |
| MLP all layers | 0.885 | 0.738 |
| attention layer 28 | 0.835 | 0.714 |
| linear | 0.875 | 0.674 |
| linear on real success rate | 0.822 | 0.609 |
| own confidence (least likely token) | 0.728 | 0.717 |
| position | 0.722 | 0.508 |

Per benchmark TEST AUC: SVAMP own confidence 0.87–0.88, hidden-state MLP 0.74–0.75 (dev math 0.94–0.95); IFEval MLP all layers 0.711, others 0.55–0.65; MBPP+ judges 0.73–0.77 (own confidence 0.68–0.72); BFCL MLP 0.772. Calibration (soft k/20): MLP all layers log loss 0.82, Brier 0.217, ECE 0.14.
Trade-off with dev-chosen threshold: MLP all layers "<= 0.5 pt" -> 64% saved at real 6.6-pt loss on benchmark; own confidence (least likely token) 16–25% saved at ~0 loss. Per-type simulation: MBPP+ 92–94% saved at ~1 pt; BFCL ~44% at 0.4 (MLP all layers); IFEval 43% at -0.4 (linear) to 58% at 4.0; SVAMP every judge 8–24 pt losses.
SVAMP stopping simulation (added later): own confidence 15% / -3.8; least-likely-token 8% / -1.9; judge 63% / -23.8; oracle 30% / +1.3.

#### zbench2 (25 Oct ~13:00): timing-only AUC (pairs of cuts within the same reply) on the 240 benchmark questions
Only 83 test replies hold both a safe and an unsafe cut (SVAMP 36, IFEval 6, MBPP+ 16, BFCL 25).

| classifier | dev timing ALL (practice) | TEST timing ALL | SVAMP | IFEval | MBPP+ | BFCL |
|---|---|---|---|---|---|---|
| position | 0.603 | 0.814 | 0.836 | 0.800 | 0.829 | 0.721 |
| own confidence | 0.806 | 0.845 | 0.963 | 0.550 | 0.850 | 0.574 |
| own confidence, 5 least likely | 0.807 | 0.871 | 0.977 | 0.625 | 0.833 | 0.721 |
| linear (hidden state) | 0.828 | 0.710 | 0.768 | 0.525 | 0.666 | 0.698 |
| mlp (hidden state) | 0.847 | 0.774 | 0.815 | 0.650 | 0.721 | 0.806 |
| mlp, all layers | 0.825 | 0.690 | 0.807 | 0.625 | 0.561 | 0.651 |
| own confidence, learned (+ position) | 0.826 | 0.813 | 0.937 | 0.475 | 0.735 | 0.721 |
| mlp + own confidence | n/a | 0.844 | 0.963 | 0.400 | 0.787 | 0.752 |

Pooled AUC: mlp + own confidence 0.753, mlp all layers 0.738, mlp 0.736, own confidence least likely 0.717.

---

### C18. stop-pair DPO from the original Qwen (stopB line), 24 Oct

Pairs: chosen = earliest paragraph cut whose 20 re-answers are right >= max(0.95, full thinking's rate) + a right answer; rejected = the full reply. Shorten pairs only. Fresh LoRA, 3 seeds, soups, fresh x8.

| run | data | ckpt | overall | svamp | mbpp | ifeval | bfcl |
|---|---|---|---|---|---|---|---|
| stopB | lot B, 554 pairs (code 179, instr 84, math 202, MC 58, tools 31); 32 steps | soup 10 | +11.9 / +0.9 | +21% / +0.3 | +12% / +2.2 | +8% / +2.8 | +2–9% / -3.4..-0.3 |
| stopB | | soup 20 | +31.1 / -2.3 | +43% / -4.1 | +32% / -1.9 | +28% / -0.9 | +18–19% / -4.1..+0.9 |
| stopB | | soup 30 | +54.2 / -3.6 | +42% / -5.0 | +82% / -8.1 | +39% / +0.3 | +28–30% / -3.4..-0.6 |
| stopB2 | lot B without code: 327 train + 48 held out; 1 epoch = 21 steps | soup 20 | +34.6 / -2.1 | +47% / -3.1 | +37% / +0.0 | +30% / -2.2 | +21–22% / -3.1..-3.4 |
| stopB3 | lots B + B2, code included, 1,058 pairs; ckpts 15–40 | soup 15 | +18.3 / -1.0 | — | — | — | — |
| stopB3 | | soup 20 | +27.1 / -2.0 | — | — | — | — |
| stopB3 | | soup 25 | +34.6 / -0.4 | +36% / -2.2 | +50% / -0.3 | +28% / +2.8 | +20–22% / 0.0..-3.7 |
| stopB3 | | soup 30 | +54.8 / -3.2 | +37% / -3.7 | +85% / -8.4 | — | — |
| stopB3 | | soup 40 | +69.1 / -5.7 | +48% / -12.5 | +94% / -8.1 | — | — |

Earlier from-raw first rounds on fresh, for reference as logged: A1 35.4, B1 24.8, v2r1 14.8 (% saved). Observed: code over-shortened (82–94% at -8) at later checkpoints when code pairs are included.

#### wronglen (fresh x8; thinking-length reduction vs original Qwen, by answer class; never-solved = 0/8 for original, 14% of answers, no training pair from them)

| model | right | wrong | never solved |
|---|---|---|---|
| rc1-rl-40 | +58% | +61% | +59% |
| stopB2 soup 20 | +36 | +43 | +31 |
| stopB3 soup 25 | +37 | +49 | +46 |
| stopB3 soup 30 | +51 | +77 | +80 |

Exception noted: svamp never-solved under stopB3 soup 25 only +3%.

---

### C19. Methods implemented/queued but not run within this range

- DRPO (rl_train.py --objective drpo; stage `drpo`, drpo1): right replies pushed up with softmax(r_l / lambda) normalised among right replies only, r_l = 1 - len / group's longest reply; wrong replies pushed down (softmax(s / tau), length-blind); trust region beta0 [KL - delta]+^2. From rc1-rl-40, lambda 0.2, fresh mixed pool (GSM-IC/Plus, IFEval-like, ToolACE, Hermes, KodCode 16%; MBPP share removed), 48 questions x 8 x 60 steps (~2,900 questions once), ckpts 40/60 on fresh x8; RL cap 4,096 for all types. Pre-registered bar: >= ~59% at >= -2.5, or > 55% at better than -2. drpo_weights toy test: [0.62, 0.33, 0.05 | -1/3 x3]. First launch failed (IFEval grader: missing langdetect).
- ShorterBetter variant (sb1): r = alpha·correct - beta·|len - SOL|, alpha 2, beta 0.005, SOL = shortest correct in group else group mean, GRPO advantages; from rc1-rl-40, 48 x 8 x 60, ckpts 40/60. Queued.
- Experiment 5, judge as teacher (teach_stop.py, stage `teach`, teach1): judge (MLP layer 20) fit on lots B..B4; original Qwen answers 3,000 fresh mixed questions; up to 16 paragraph ends scored per answer; fresh LoRA trains only p("</think>" next) by BCE toward sigmoid((p - threshold)/0.03), thresholds fixed:0.9 and rel:0.05 (relative rule vs full-thinking soft success, floor 0.3). Revisions: math (gsm_ic, gsm_plus, gsm8k) target 0 at every paragraph end; later redesign: teacher = own confidence (5 least likely tokens) for math and code, soft hidden-state judge for tools, instructions and MC protected, relative rule only, margins 0.05 and 0.1. First launch failed (mixed_pool KeyError 'prompt').
- stopB4, stopB5 (code + tools only stop pairs from lots B..B4), zoo B4, zB4bench, benchA40 / zAbench: queued. zoo B4 and stopB4 first failed (runs/bon/pairs-plotB3/cal/verdicts.jsonl missing after restore race); fix ensure_lots().
- probe_zoo additions: own-confidence variants (own_min, own_low5, own_first50, own_certainty), timing-only AUC, own-confidence learned, linear/MLP + own confidence, ESTAR-style trajectory features and ESTAR label; calibration table (log loss / Brier / ECE vs k/20).

### C20. Infrastructure incidents (this range)

| when | event |
|---|---|
| 22–23 Oct | runtime lost; half-written eval folder refused by evaluate.py (ca1 x8 requeued) |
| 23 Oct | runtime 2 lost overnight, relaunched (rep5 resumed from Drive) |
| 23 Oct | rr4 started by mistake, stopped at first RL block |
| 24 Oct ~20:10 | runtime 2 ended mid lot B3 labelling (27% of 148,720 forced answers), no error |
| 25 Oct 01:45 | relaunch sat in a FULL restore; page lost it; fix 11f3e5b (light restore) |
| 25 Oct ~03:50 | both runtimes gone (runtime 1 at lot B4 grading after 153,020 forced answers; runtime 2 cell "ended unsuccessfully" after 1,606 s); cause not verified |
| 25 Oct ~09:10 | both queues ended on errors (zoo B4/stopB4 missing verdicts; drpo1 langdetect; teach1 KeyError 'prompt') |
| 25 Oct ~10:30 | both runtimes relaunched (scripts 36a4a86) |

Policy change 24 Oct ~19:50 (user): no MBPP-family data anywhere; lot B4's 200 mbpp_train questions dropped downstream (`probe_zoo.py --drop-sources`), removed from stopB4 and drpo pools.

---

## Part D — Track 2g, second half, and the stop-classifier work (25–26 Oct)

Source: thinkingcap/EXPERIMENT_LOG.md lines 1651–2254 (track 2g header at lines 1080–1100 for vocabulary) and
thinkingcap/HANDOFF_CLASSIFIER_2026-10-25.md (subsection C.2). Numbers copied as printed. "—" = not reported.
Line 1651 is the tail of a preceding entry ("answer per scored point (~30-50k short generations, ~1 GPU-hour)"); not covered here.

### D.0 Vocabulary

| term | meaning |
|---|---|
| original Qwen / f8-base | Qwen3-0.6B, thinking mode, unmodified |
| rc1-rl-40 | earlier model of this track (DPO + RL, step 40); fresh x8: 53.9% saved / -2.1 pts (SVAMP +65% / -4.1, IFEval +46% / -0.6, MBPP+ +59% / -1.9, BFCL +41-45% / -1.2..) |
| fresh x8 | questions_fresh20261022: 240 benchmark questions (40 per dataset/subset), 8 samples each, official evaluate.py |
| choosing set | questions_40: another 240 benchmark questions, historically used to pick checkpoints; used as measure-only test set "bench40" |
| x / y in tables | tokens saved % / accuracy change in points vs original Qwen (negative saved = longer) |
| lot | labelled training data: questions drawn from training sources, replies generated, thinking cut at paragraph ends ("\n\n"), "</think>" forced at each cut, N forced answers per cut graded |
| cut-success curve | share of right forced answers by share of thinking kept: 0-20 / 20-40 / 40-60 / 60-80 / 80-<100 / full |
| perfect stop | per reply, the best cut vs the natural stop: tokens saved / accuracy gained (truncation bound) |
| judge / zoo | stopping classifiers trained on lots (probe_zoo.py), see C.2 |
| own confidence | model answers once greedily from the cut; geometric-mean token probability of that answer (variants: least likely token, 5 least likely, first 50 tokens, certainty = minus mean entropy) |
| ESTAR features / trajectory | own-confidence trajectory features over successive cuts; "ESTAR label" = target "same answer as the final answer" |
| timing-only AUC | AUC computed only over (safe, unsafe) cut pairs from the same reply |
| soup N | equal-weight average of the 3 seeds' LoRA adapters at step N, merged and scored on fresh x8 |
| DPO pair kinds | shorten: earliest safe cut + right answer beats full reply; hard: full right reply beats its failing early cut; guard: question's longest right reply beats its shortest wrong; right: own right beats own wrong within +-25% length |
| sources -> benchmark analogue | gsm_ic / gsm_plus / gsm8k_matched / metamath_sv -> SVAMP; ifeval_like -> IFEval; kodcode -> MBPP+; toolace / hermes_tools -> BFCL; arc_easy / commonsense_qa (mc) -> none |
| GPU 1 / GPU 2, runtime 1 / 2 | two Colab L4 runtimes executing queue files |

### D.1 Experiments, chronological

#### C.1.1 zoo B4 — 25 Oct ~15:00 CET
Method: original Qwen's judges (classifier zoo, C.2) trained on lots B+B2+B3+B4, MBPP dropped (198 cuts). Dev 2,447 questions / 21,539 cuts. Test = 1,072 held-out practice questions / 9,571 cuts.

| classifier | TEST AUC ALL |
|---|---|
| position | 0.681 |
| own confidence | 0.721 |
| own confidence, 5 least likely | 0.740 |
| linear | 0.858 |
| MLP | 0.879 |
| MLP all layers | 0.872 |
| attention | 0.80-0.83 |
| own confidence learned | 0.812 |
| linear + own | 0.868 |
| MLP + own confidence | 0.892 |

Comparison: zoo B2 MLP 0.864. Per type (MLP): gsm 0.94-0.95, hermes 0.91, ifeval_like 0.86, kodcode 0.80, toolace 0.74, mc 0.59-0.62.
Timing-only TEST (429 replies with both labels): own confidence 0.833 overall; MLP math 0.94 / code 0.76 / toolace ~0.5 / hermes 0.81; own confidence code 0.86, toolace 0.74.
Pooled trade-off, practice test: MLP all layers 64% saved at -0.2 (gain) for a 0.5-pt budget; MLP 70% / 2.6.

#### C.1.2 zB4bench — 25 Oct ~16:30 (interim) and ~17:00 (complete)
Method: original Qwen's judges on lots B..B4, MBPP dropped; dev 3,519 questions / 31,110 cuts; TEST = bench40 (choosing-set benchmark questions, measure only); 83 replies with both labels for timing. Adds ESTAR features.

Pooled TEST AUC:

| classifier | TEST AUC ALL | timing-only TEST AUC ALL |
|---|---|---|
| position | 0.516 | 0.814 |
| linear | 0.670 | 0.645 |
| MLP | 0.722 | 0.709 |
| MLP all layers | 0.760 | 0.697 |
| own confidence learned | 0.661 | 0.807 |
| own confidence (raw) | — | 0.845 |
| own confidence, 5 least likely (raw) | 0.713 | 0.871 |
| linear + own | 0.718 | — |
| MLP + own | 0.713 | 0.775 |
| ESTAR trajectory learned | 0.741 | 0.826 |
| MLP + trajectory | 0.723 | — |
| ESTAR label, trajectory (target "same as final", true for 36% of cuts) | 0.700 | 0.834 |
| ESTAR label, MLP | 0.674 | — |
| attention | 0.70-0.71 | 0.65-0.68 |

Comparison with zbench (lots B-B3): MLP 0.736 -> 0.722; MLP all layers 0.738 -> 0.760.
Timing per benchmark: SVAMP own 0.96-0.98, ESTAR trajectory 0.930, MLP 0.820; MBPP+ own 0.83-0.85, ESTAR label 0.822, MLP 0.617; BFCL ESTAR trajectory 0.798, own 5-least 0.721, MLP 0.643 (zbench2 MLP 0.806).

Stopping simulation (2-pt allowance per type, thresholds from practice data), saved / loss:

| benchmark | results |
|---|---|
| SVAMP | every learned signal: 19-48 pt losses; raw own confidence 15% / 3.8; least likely 15% / 1.9 |
| MBPP+ | 83-91% saved at 0.8-4.4; own 5 least likely 91% / 1.3; ESTAR trajectory 91% / 1.6 |
| IFEval | own confidence 68% / 1.0 |
| BFCL | ESTAR trajectory 23% / -2.0 (gain); linear 51% / 1.8; MLP 47% / 0.7 |

#### C.1.3 drpo1 — DRPO from rc1-rl-40, 25 Oct (training to ~18:15, scoring to ~20:00)
Method: rl_train.py --objective drpo (DRPO, arXiv 2510.04474): right replies pushed up with softmax(r_l/lambda) weight normalised among right replies; wrong replies pushed down; r_l = 1 - len / group's longest. lambda 0.2, lr 1e-5, 2 updates per step, max 4,096 tokens, 60 steps, 48 new questions x 8 replies per step, LoRA from rc1-rl-40. Pool (MBPP-free mixed): gsm_ic 17%, gsm_plus 17%, ifeval_like 24%, toolace 20%, hermes 10%, KodCode 16%. Checkpoints saved at 20/40/60. Pre-fixed success bar: >= ~59% saved at >= -2.5, or > 55% at better than -2.
Later found (26 Oct ~01:00): trained with the batch-mix bug (C.1.11).

Training-batch averages:

| steps | batch accuracy | thinking tokens | total tokens |
|---|---|---|---|
| 1-10 | 0.62 | 369 | 414 |
| 11-20 | 0.60 | 289 | 338 |
| 21-30 | 0.61 | 266 | 307 |
| 31-40 | 0.69 | 218 | 270 |
| 41-50 | 0.66 | 149 | 184 |
| 51-60 | 0.59 | 129 | 187 |

Answer part (total - thinking): ~45 tokens early -> ~58 at steps 51-60 (step 57: 243 total vs 123 thinking).
Step 1: 493 thinking tokens. Batch accuracy range at steps 45-56: 0.48-0.70.

Evaluation: fresh x8.

| checkpoint | saved | accuracy | svamp | ifeval | mbpp | bfcl irr | bfcl mult | bfcl simple |
|---|---|---|---|---|---|---|---|---|
| step 20 | 61.7% | -4.9 | +74% / -10.0 | +49% / -5.0 | +69% / -1.9 | +54% / -1.6 | +46% / -6.6 | +42% / -0.3 |
| step 40 | 73.3% | -6.9 | +80% / -16.2 | +66% / -7.2 | +83% / -2.5 | +60% / +1.6 | +52% / -7.2 | +50% / +0.6 |
| step 60 | 78.7% | -6.0 | +85% / -14.1 | +73% / -5.9 | +88% / -3.1 | +64% / +2.8 | +55% / -5.0 | +53% / -0.3 |
| rc1-rl-40 (start) | 53.9% | -2.1 | +65% / -4.1 | +46% / -0.6 | +59% / -1.9 | +41-45% / -1.2.. | — | — |

Step 40 absolute: accuracy svamp 73.4%, ifeval 58.4%, mbpp 43.8%, bfcl 68.8-88.8%; mean reply tokens svamp 126, ifeval 237, mbpp 349, bfcl 78-100.
Observed: training-batch accuracy is a mixed average over new questions each step (~34% math, 24% instructions, 30% tools, 16% code; batch-to-batch swing 0.48-0.81), so the test-set SVAMP drop was not visible in it; training grades IFEval-like prompts with the project's own checker. Fix for later runs: per-type batch accuracy logged ("by_type"); every saved checkpoint scored.

#### C.1.4 drpo1 step 20 at half strength — 26 Oct ~00:00
Method: step-20 LoRA merged at weight 0.5 into rc1-rl-40 (track.py drpo --drpo-weighted 20:0.5). Fresh x8.

| model | saved | accuracy | svamp | ifeval | mbpp | bfcl irr | bfcl mult | bfcl simple |
|---|---|---|---|---|---|---|---|---|
| drpo1-20 x0.5 | 59.8% | -3.7 | +69% / -5.0 | +53% / -4.7 | +67% / -2.8 | +50% / -1.2 | +43% / -5.6 | +41% / -0.3 |

#### C.1.5 benchA40 — 25 Oct
Method: rc1-rl-40's replies to the 240 choosing questions, 8 each, cuts x 20 forced answers; measure only. 323 replies (206 right), 1,096 cuts (original Qwen: 1,812).

| benchmark | replies (right) | cuts | curve 0-20 / 20-40 / 40-60 / 60-80 / 80-100 / full | perfect stop: saved / gained |
|---|---|---|---|---|
| SVAMP | 59 (37) | 209 | 0.18 / 0.35 / 0.26 / 0.62 / 0.58 / 0.63 | 37% / +10.1 |
| IFEval | 56 (30) | 124 | 0.58 / 0.55 / 0.48 / 0.63 / 0.14 / 0.53 | 32% / +11.0 |
| MBPP+ | 57 (28) | 562 | 0.38 / 0.34 / 0.38 / 0.49 / 0.42 / 0.48 | 75% / +12.7 |
| BFCL | 151 (111) | 201 | 0.52 / 0.69 / 0.50 / 1.00 / - / 0.76 | 15% / +2.3 |

Original Qwen perfect stop for comparison: SVAMP 53%, BFCL 51%.

#### C.1.6 zAbench — rc1-rl-40's judges on benchA40, 25 Oct ~18:00 (interim) and ~20:30
Method: judges trained on lots A + A2 (rc1-rl-40's lots: 1,762 questions / 9,595 cuts); TEST = benchA40 (1,096 cuts); 53 replies with both labels for timing. ESTAR label "same as final" true for 52% of cuts.

| classifier | pooled TEST AUC | timing-only TEST AUC ALL |
|---|---|---|
| position | 0.512 | 0.827 |
| linear | 0.649 | — |
| MLP | 0.694 | 0.767 |
| MLP all layers | 0.745 | 0.728 |
| attention | 0.73 | — |
| own confidence learned | 0.661 | — |
| own confidence (raw) | — | 0.854 |
| own confidence first-50 | — | 0.860 |
| own certainty (entropy) | — | 0.870 |
| MLP + own | 0.768 | 0.824 |
| linear + own | — | 0.824 |
| MLP + trajectory | 0.749 | — |
| ESTAR trajectory | 0.696 | — |
| ESTAR label trajectory | 0.736 | 0.841 |

SVAMP timing: own confidence 0.91-0.93 (original Qwen 0.96-0.98), MLP 0.82, position 0.82.
Stopping simulation (2 pts per type; thresholds from rc1-rl-40's practice lots):

| benchmark | results (saved / accuracy as printed in the log) |
|---|---|
| SVAMP | every signal stops 0-2%; oracle bound 17% |
| IFEval | own confidence 33% saved at +6.9 (gain); MLP all layers 31% / +5.6 (gain) |
| MBPP+ | own confidence 87% / -3.2; MLP all layers 72% / +0.4 (gain); MLP 54% / +2.8 (gain) |
| BFCL | ~19% / -2.2 |

#### C.1.7 nothink nt1 / nt2 — thinking switched off, 25 Oct (nt1) to 26 Oct ~00:00 (nt2)
Method: copy of the official evaluate.py with enable_thinking=False (chat template writes an empty think block); fresh x8; measure only. nt1 (original Qwen) crashed in compare.py (refuses two runs of one model); replaced by a records-based diff (nt2), which also ran rc1-rl-40.

Original Qwen, thinking ON -> OFF (nt2 records diff; nt1 values in brackets where different):

| benchmark | accuracy ON -> OFF | change | mean reply tokens ON -> OFF | unreadable OFF |
|---|---|---|---|---|
| bfcl irrelevance | 80.9 -> 87.2 | +6.2 (nt1 +6.3) | 195 -> 8 (nt1 OFF 7.9) | — |
| bfcl multiple | 75.9 (nt1 76.0) -> 58.8 | -17.2 | 206 -> 20 (nt1 20.5) | 11.9% |
| bfcl simple | 88.1 -> 76.6 | -11.6 (nt1 -11.5) | 197 -> 24 (nt1 24.3) | 9.4% |
| ifeval | 65.6 -> 60.6 | -5.0 | 703 -> 284 | — |
| mbpp | 46.2 (nt1 46.3) -> 37.2 | -9.1 | 2,098 -> 53 | — |
| svamp | 89.7 (nt1 89.6) -> 17.5 | -72.2 (nt1 -72) | 644 -> 10 (nt1 9.7) | 24.7% (no "Final answer" line) |

rc1-rl-40, thinking ON -> OFF:

| benchmark | accuracy ON -> OFF | change | tokens ON -> OFF |
|---|---|---|---|
| bfcl irrelevance | 79.7 -> 86.6 | +6.9 | 106 -> 5 |
| bfcl multiple | 71.9 -> 63.4 | -8.4 | 121 -> 20 |
| bfcl simple | 87.8 -> 78.8 | -9.1 | 121 -> 24 |
| ifeval | 65.0 -> 55.3 | -9.7 | 379 -> 229 |
| mbpp | 44.4 -> 33.1 | -11.2 | 867 -> 45 |
| svamp | 85.6 -> 23.1 | -62.5 | 224 -> 9 |

Related cut-curve observations cited in the log: practice IFEval-like forced answer after 0-20% of thinking 0.77 vs 0.73 full, last fifth 0.66; rc1-rl-40 IFEval 0.14 at 80-100%; Hermes 0.18 at 80-100% vs 0.49 full.

#### C.1.8 Queue / design entries without results in this range (25 Oct)

| item | content |
|---|---|
| drpo0 | DRPO from raw Qwen with DLER dynamic sampling (--dynamic-sampling F: draw F x questions per step, train only on groups with >= 1 right reply, logged dropped_groups; F = 1.5), lambda 0.2. Started ~18:30 on runtime 1 (~1,300-token replies), interrupted at step 3 (~22:00), removed from the queue. |
| target (user) | > 50% saved at better than -1 pt, evaluation untouched, no test-time gate |
| bal0 | one balanced DPO round from raw Qwen (balanced_pairs.py, NLL anchor 1.0), GSM8K for math, 6,000 questions x 4, up to 200 steps x 3 seeds, soups 50-200 on fresh x8. Queued; no result in range. |
| teach1 / teach2 | experiment 5 (stop teaching: LoRA trains the model's own p("</think>") at paragraph ends toward a teacher target). teach2 = own confidence as teacher for math, code and tools (--teach-own ...,toolace,hermes_tools; --teach-judge none), instructions protected, margins 0.05 and 0.1, reusing teach1 answers. ~22:00: teach1 switched to own confidence for every covered type; teach2 removed as duplicate. |
| drpo2 | DRPO from rc1-rl-40, lambda 0.5, lr 5e-6, pool seed 64, per-type logging, every 10-step checkpoint scored. Queued; no result in range. |
| drpo3 options | rl_train.py --protect-sources (no length pressure for those groups; DRPO weights right replies equally, ShorterBetter length term 0) and --guard-drop G (stop when per-type mean batch accuracy of last 5 steps falls > G below first 5 steps') |
| stopB4 | removed from queue (superseded) |
| track.py drpo | saves every 10 steps, scores 20/40/60 by default; --drpo-soup S1 S2 ... averages checkpoint adapters with equal weights |

#### C.1.9 SVAMP check: choosing set vs fresh set — 25 Oct
Observation compiled in the log (no new run):

| model | set | SVAMP saved / acc |
|---|---|---|
| gentle round-2 model step 10 | choosing set 240 x 8 | +64% / 0.0 (23 lost, 23 gained) |
| balanced soup | choosing set x8 | +68% / -1.6 |
| several x4 checks | choosing set | +55-69% / -0.6..+3.1 |
| round 2 retrain | fresh | — / -9.7 |
| round 2 on questions 41-80 | fresh | — / -7.5 |
| rounds 4 | fresh | — / -9.1, -10.6 |
| balanced soup | fresh | +64% / -4.7 |
| rc1-rl-40 | fresh | +65% / -4.1 |

The models listed under "choosing set" had been selected on that set's 40 SVAMP questions.
Also added: teach_stop.py progress file (per source: mean p("</think>") at stop vs keep targets, thinking saved if stopped at first paragraph end with p >= 0.5, share of good stops).

#### C.1.10 sb1 — ShorterBetter reward from rc1-rl-40, 25–26 Oct (stopped ~00:30, scored ~01:45)
Method: rl_train.py with ShorterBetter length reward, beta 0.005 (5x the paper's 0.001), from rc1-rl-40, no guard, per-type batch logging, checkpoints every 10 steps. Stopped at step 33 of 60. Trained with the batch-mix bug (steps 22-30 drew only gsm_ic / ifeval_like).

Training batches: thinking 472 -> ~95 tokens; batch accuracy 0.62 -> 0.27-0.42 at steps 28-33. Per type at steps 31-33 (start in brackets): gsm_ic 0.28-0.42 (0.80-0.96), gsm_plus 0.04-0.19 (0.42-0.75), kodcode 0.16-0.26 (0.33-0.65), toolace 0.07-0.60.

Fresh x8:

| checkpoint | saved | accuracy | svamp | mbpp | ifeval | bfcl irr | bfcl mult | bfcl simple |
|---|---|---|---|---|---|---|---|---|
| step 10 | 58.6% | -3.3 | +67% / -5.3 | +70% / -2.8 | +47% / -1.9 | +50% / -1.2 | +43% / -6.9 | +42% / -1.6 |
| step 20 | 66.2% | -5.7 | +69% / -10.0 | +82% / -4.7 | +55% / -4.1 | +56% / -3.1 | +48% / -6.2 | +46% / -2.2 |
| step 30 | 78.2% | -11.2 | +83% / -28.4 | +90% / -10.0 | +68% / -4.4 | +64% / +0.9 | +54% / -3.4 | +54% / -3.1 |

sb2 (queued, no result in range): from rc1-rl-40, beta 0.001, math protected, guard 0.06, proportional mix, lr 5e-6, checkpoints every 5 steps scored at 5/10/15/20/25/30/40/50/60; math pool MetaMathQA.

#### C.1.11 RL batch-mix bug — found 26 Oct ~01:00
rl_train.py balanced_order interleaved the pool round-robin by source: each step got the same count of each source (not the pool's proportions), and after smaller sources ran out, remaining steps held only the larger ones. Affected drpo1 and sb1. Fix: each row k of a source with n rows placed at (k + u)/n, pool sorted by position. Check on a 4,368-row pool, 72 drawn per step: steps 0, 20, 40, 59 each ~12/12/17/15/8/8 for gsm_ic / gsm_plus / ifeval_like / toolace / hermes / kodcode. drpo3 and later use the fix.

#### C.1.12 stopB5 — stop-pair DPO on code + tools, 25–26 Oct (final ~03:15)
Method: original Qwen; shorten pairs from lots B..B4, code and tools sources only (kodcode, toolace, hermes_tools; lot B3 has none): 928 pairs (~660 KodCode, ~155 Hermes, ~110 ToolACE), all "stop earlier". targeted DPO (LoRA, beta 0.1, lr 2e-5, chosen NLL weight 1.0, batch 16), 3 seeds, soups 10-30. DPO seed 0 logged pairs ordered right 100% at step 40/55. Evaluation: fresh 240 x 8.

| soup | all | bfcl irrel. | bfcl multiple | bfcl simple | ifeval | mbpp | svamp |
|---|---|---|---|---|---|---|---|
| 10 | +0.2% / +0.4 | -2..+3% (range) | — | — | — | -4% / +3.4 | -1% / — |
| 15 | +5.6% / -0.9 | +4% / -1.9 | +8% / -1.6 | +5% / -0.6 | +11% / -2.2 | -1% / +0.9 | +6% / -0.9 |
| 20 | +11.9% / -1.2 | +14% / -2.5 | +14% / -1.9 | +14% / +0.6 | +15% / +0.6 | +11% / -1.3 | +8% / -2.8 |
| 25 | +40.7% / -2.2 | +27% / -0.9 | +22% / +0.9 | +24% / -0.9 | +31% / -0.6 | +69% / -5.6 | +23% / -2.2 |
| 30 | +56.3% / -2.7 | +49% / +0.0 | +39% / -1.3 | +39% / -0.6 | +31% / -1.2 | +88% / -6.9 | +20% / -2.2 |

Observed: math and instructions were not in the pairs but shortened; MBPP+ went from +11% / -1.3 (soup 20) to +69% / -5.6 (soup 25).

#### C.1.13 drpo3 -> drpo3b — DRPO from raw Qwen, math protected, 26 Oct (~03:00 to ~05:00)
Method: rl_train.py --objective drpo from raw Qwen3-0.6B, lambda 0.5, lr 5e-6, DLER dynamic sampling 1.5, math protected (gsm_ic, gsm_plus, gsm8k*: no length pressure), per-type guard 0.06, proportional batch mix, checkpoint every 5 steps (drpo3b), ~12-20 min per step.
drpo3: pool 1,592 questions for 4,320 needed (math 54%, code 2%, Hermes 0%: earlier-draw filter emptied small sources; contamination check's 13-word rule drops IFEval-like prompts). Steps 1-5: thinking 1,208 -> 974; batch accuracy 0.66-0.72; gsm_ic 0.51-0.65, gsm_plus 0.30-0.55, ifeval_like 0.85-0.95, kodcode 0.94-1.00 (8-16 replies per step), toolace 0.67-0.96. Restarted as drpo3b.
drpo3b: mixed_pool(reuse=...) — runs from the raw model no longer drop questions drawn by earlier experiments (contamination check still applies). Pool 6,224 for 4,320 needed: toolace 1,604, gsm_plus 1,371, gsm_ic 1,371, kodcode 707, ifeval_like 693, hermes_tools 478 (math 44%).

Training-batch observations:

| steps | batch accuracy | all thinking | non-math thinking | code thinking | math thinking |
|---|---|---|---|---|---|
| 1-3 | 0.67-0.68 | 1,189 / 1,061 / 1,120 | — | 1,064-1,635 | 1,800-2,900 |
| 10-12 | 0.74-0.77 | 1,076-1,218 | 397-595 | 866-1,426 | — |
| 11-16 (after resume) | 0.70-0.77 | 924-1,101 | 317-537 (step 1: 433) | 896-1,413 (step 1: 1,635) | — |

Runtime loss at step 12 (C.1.20); resumed "after step 10 from step-10 (no optimizer state saved: fresh AdamW)"; log of steps 1-10 and steps 11-12 lost; guard baseline restarted at step 11.
Guard stop at step 20: "gsm_ic 0.77 -> 0.68; ifeval_like 0.83 -> 0.76" (batch, steps 16-20 vs 11-15).

Fresh 240 x 8:

| checkpoint | saved | accuracy | svamp | mbpp | ifeval | bfcl irr | bfcl mult | bfcl simple |
|---|---|---|---|---|---|---|---|---|
| step 5 | +3.5% | -0.2 | — | — | — | — | — | — |
| step 10 | +5.2% | +0.0 | — | — | — | — | — | — |
| step 15 | +8.0% | +1.3 (CI -0.6..+3.3) | +18.4% / -0.3 | +5.5% / +4.1 | +3.7% / +2.2 | +6.5% / -0.3 | -1.4% / +0.3 | +5.7% / -2.8 |
| step 20 (guard stop) | +13.5% | -0.3 (CI -1.95..) | +21.4% / -0.6 | +10.2% / -1.6 | +12.0% / +1.3 | +13.7% / +0.3 | +8.5% / -1.3 | +8.0% / +0.3 |

#### C.1.14 mathmatch mm1 / mm2 / mm3 — 25–26 Oct
Method: (1) profile the choosing set's 40 SVAMP questions (operations, kind, length, original Qwen accuracy/thinking from benchlot x8); (2) pick 1,500 GSM8K problems (train + test splits, 1-2 calculations, no ASDiv/MAWPS/SVAMP) stratified on SVAMP's ops x kind cells and length, numeric prompt wording, earlier draws removed, contamination-checked; (3) hold out 300 as math test (data/gsm8k_matched-test.jsonl), original Qwen x8; other 1,200 = gsm8k_matched-train. SVAMP whole-set composition (described only): Subtraction 531, Addition 195, Common-Division 166, Multiplication 108; one operation 762, two 237.
Failures: mm1 IndexError (probe_curve numbered rows by line in the whole bench file, not within SVAMP); mm2 failed (benchlot rollouts carry output_ids, no thinking count; now read from "</think>" position); mm3 ran.
Earlier related source gsm8k_easy (prepare_types.py): GSM8K train problems with 1-2 calculations: 2,579 of 7,473 (404 one-step, 2,175 two-step).

| set | n | accuracy | thinking tokens | words | one-operation |
|---|---|---|---|---|---|
| choosing-set SVAMP | 40 | 0.79 | 858 | 50 | 30 of 40 |
| matched GSM8K test (0 contamination flagged) | 300 | 0.87 | 761 | 40 | 60 of 300 |
| matched GSM8K train | 1,200 | — | — | — | 199 of 1,200 |

Side note in log: gsm8k_matched draws GSM8K train and test; GSM-IC / GSM-Plus are built on GSM8K test, so models trained on gsm_ic/gsm_plus (rc1, drpo*) may have seen paraphrases of some matched-test questions.

#### C.1.15 lot B5 — matched GSM8K, 26 Oct ~00:45
Method: original Qwen, 600 gsm8k_matched-train questions x 20 tries, 20 forced answers per paragraph cut; 0 contaminated.

| source | curve 0-20 / 20-40 / 40-60 / 60-80 / 80-<100 / full | perfect stop: saved / gained |
|---|---|---|
| SVAMP (choosing set, benchlot) | 0.26 / 0.34 / 0.48 / 0.58 / 0.55 / 0.73 | 53% / +8.9 |
| matched GSM8K (lot B5) | 0.08 / 0.25 / 0.44 / 0.56 / 0.61 / 0.72 | 47% / +9.6 |
| GSM-IC (practice, lot B3) | 0.25 / 0.46 / 0.47 / 0.49 / 0.52 / 0.52 | 66% / +29.5 |
| GSM-Plus (practice, lot B3) | 0.21 / 0.47 / 0.47 / 0.46 / 0.50 / 0.47 | 71% / +31.8 |

#### C.1.16 Public competitor repos (read 25–26 Oct; numbers as reported by them / the log)

| repo | reported result | recipe as read |
|---|---|---|
| basta/lightning-mcqwen3 | 47.3% / -0.9 (43.2 / -0.85 without its test-time think gate); SVAMP 66% / -3.8 | Chain-of-Draft prompt distillation SFT -> cut oracle (4 re-answers) -> shorten/guard/hard DPO -> NLL-anchored correctness DPO -> layer-21 probe gate in evaluate.py. Math GSM8K + Skywork-OR1; 2,500 questions per source x 4 traces, cuts at 10..90% re-answered 4x (197k cuts), 9,034 pairs (40% shorten, 40% guard, 20% hard) + correctness DPO on math (right vs wrong within +-25% length, 732 pairs) |
| michal-bida/thinking-compression | 47.0% / -0.78; SVAMP 56% / -2.4 | one DPO round from raw model on own graded samples, benchmark-matched prompts, shortest right vs longest wrong/truncated + vs >= 1.5x longer right, balanced per domain; SFT only +4.5%, offline GRPO failed, long pairs alone collapse; math GSM8K 70% + MetaMathQA GSM_SV 15% + GSM_FOBAR 15%, prompts drawn evenly over SVAMP prompt-length deciles, 8-gram check; per domain shorten pairs capped at the count of right-beats-wrong pairs (SVAMP -3.9 -> -2.4 after cap) |

#### C.1.17 lot V2 — 26 Oct (started ~01:00, labelled ~09:40)
Plan evolution: first version ~3,500 questions, 6 cuts x 8 answers; replaced by ~10,300 questions (matched GSM8K rest 600 + MetaMathQA GSM_SV/FOBAR 2,500 [prepare_types metamath_sv] + KodCode 3,000 + ToolACE 2,000 + Hermes ~230 + IFEval-like 2,000), 8 replies each, 8 cuts x 4 forced answers (safe = right on both halves of the 4).
Actual: 4,868 new of 8,178 drawn (3,310 dropped as earlier draws); probe_curve per-source cap 2,000 trims MetaMathQA; contamination dropped 997 of 4,573 non-code questions (nearly all IFEval-like; 13-word run rule; 8-gram shares 6-25%, one 63%). Final: 3,554 questions, 4,996 replies, 38,939 cuts x 4 (12,156 fully safe). KodCode ~180 questions, ToolACE ~200, Hermes 0.

| source | replies | curve 0-20 / 20-40 / 40-60 / 60-80 / 80-<100 / full | perfect stop |
|---|---|---|---|
| matched GSM8K | 780 | 0.08 / 0.21 / 0.40 / 0.57 / 0.63 / 0.75 | 49% / +6.6 |
| MetaMathQA SV/FOBAR | 2,918 | 0.06 / 0.09 / 0.19 / 0.43 / 0.64 / 0.69 | 39% / +16.3 |
| IFEval-like | 672 | 0.75 / 0.76 / 0.77 / 0.74 / 0.68 / 0.75 | 67% / +11.6 |
| KodCode | 293 | 0.41 / 0.45 / 0.44 / 0.45 / 0.49 / 0.50 | 81% / +20.6 |
| ToolACE | 333 | 0.35 / 0.35 / 0.47 / 0.42 / 0.37 / 0.52 | 49% / +8.6 |

#### C.1.18 stopV2 — stop + brake pair DPO on all original-Qwen lots, 26 Oct (~09:40 to ~17:40)
Method: original Qwen; pairs from lots B..B5 + V2 via targeted_pairs. Code/tools/instructions: shorten + hard + guard. Math = matched GSM8K + MetaMathQA only (GSM-IC/Plus dropped): hard + right pairs, plus shorten pairs if the source's curve passes an SVAMP-likeness gate (success at 20-40% / success at full <= 0.6; SVAMP 0.47; measured matched GSM8K 0.32, MetaMathQA 0.13 -> both get shorten pairs). Caps per type. DPO as stopB5 (chosen-NLL anchor 1.0, batch 16, lr 2e-5), 3 seeds x 300 steps (tripwire 0.000), soups 25/50/75/100 scored (150-300 skipped). Seed 0 at step 115: pairs ordered right 62-76%.

Pairs: 6,021 after caps (5,970 trained, 48 held out).

| source | total | shorten | hard | guard | right |
|---|---|---|---|---|---|
| matched GSM8K | 1,300 | 421 | 509 | 176 | 194 |
| MetaMathQA | 1,300 | 377 | 533 | 169 | 221 |
| KodCode | 1,500 | 833 | 281 | 386 | — |
| IFEval-like | 1,258 | 836 | 123 | 299 | — |
| ToolACE | ~420 | — | — | — | — |
| Hermes | 243 | — | — | — | — |
| all kinds (later recount) | 6,021 | 2,835 (47%) | 1,564 | 1,207 | 415 |

Fresh 240 x 8:

| soup | saved | accuracy | per benchmark |
|---|---|---|---|
| 25 | -9.6% | +0.0 | every benchmark -3 to -13.5% (longer) |
| 50 | -8.6% | +1.0 | every type 2-19% longer; mbpp +2.2, bfcl multiple +1.6 |
| 75 | -9.9% | +0.9 | svamp +5.5 / +1.3; mbpp -20.3 / +2.2; ifeval -12.7 / +0.9; bfcl irr -10.9 / -0.6, mult -16.9 / 0.0, simple -14.2 / -1.6 |
| 100 | -15.8% | +0.2 | mbpp 32% longer; bfcl 13-19% longer; svamp flat |

Log notes: in stopV2, 53% of chosen replies are full-length or the question's longest right reply (hard, guard, right), so the chosen-NLL anchor imitates long replies about half the time; in stopB..B5 every chosen reply was a short cut. Soup 50 corresponds to ~376 shorten pairs seen, soup 75 to ~560.

Follow-ups queued, no results in range:
- stopV3 (first form, on hold): guard flipped to shortest right over longest wrong/unfinished on code/tools/instructions, hard pairs capped at 0.25 x shorten. Second form (queued): guard pairs dropped, hard pairs capped at 20% per type; 5,060 pairs (matched GSM8K shorten 722 / hard 254 / right 324; MetaMathQA 655 / 276 / 369; KodCode 833 / 208; IFEval-like 836 / 123; ToolACE 212 / 53; Hermes 156 / 39); seed 0 at step 65/300 (pairs ordered right 78-81%).
- gold1: stop_gold.py trains only p("</think>") at labelled paragraph ends of the lots' own replies; target 1 where the cut's forced answers are right at least as often as the full thinking (and >= 25%), else 0; lots B..B5 + V2; GSM-IC/Plus, MBPP and never-solved replies excluded; per-type caps as stopV2; LoRA r16, lr 1e-4, 8 replies per step; checkpoint every 100 steps scored on fresh x8.

Re-analysis entry (26 Oct ~17:40), figures cited: across DPO stop runs (batch 16, constant lr 2e-5) saving tracked pairs seen: ~30-35% saved at 320-400 pairs, ~55% at 480 (stopB 554 pairs, B2 327, B3 1,058, B5 928), code dropping between 400 and 480 pairs in each; B3 soup 25 34.6 / -0.4 vs B2 soup 25 34.6 / -2.1; B2 (no code pairs) cut code 37%; B5 (code + tools only) cut IFEval 31%, SVAMP 20%.

#### C.1.19 drpo3c / drpo3d — drpo3b-20 continued, 26 Oct (~08:45 to ~14:30)
Method: from models/drpo3b-20, same settings (lambda 0.5, lr 5e-6, dynamic sampling, math protected with GSM-IC/Plus), 40 more steps planned, every 5 steps scored on fresh x8, guard loosened to 10 points.
drpo3c: stopped at step 2; pool 914 for 2,880 needed (ifeval 339, gsm_ic 393, gsm_plus 120, toolace 59, KodCode 3, Hermes 0). New --pool-continues: continued run may reuse earlier draws except pools of the runs it continues.
drpo3d: pool 3,227 for 2,880 (ToolACE 816, GSM-IC 846, GSM-Plus 731, IFEval-like 476, KodCode 358; no Hermes).

Training batches: steps 1-2 accuracy 0.78 / 0.75, thinking 1,065 / 1,094; steps 1-6 accuracy 0.75-0.84, non-math thinking 401 -> 300, code 1,175 -> 767; steps 7-23 accuracy 0.74-0.83, all thinking 818-1,162, code 557-1,910. Guard stop at step 23: "toolace 0.87 -> 0.76" (batch, steps 19-23 vs 1-5; ToolACE 72-104 replies per step).

Fresh 240 x 8 (total RL steps from raw Qwen in brackets):

| checkpoint | saved | accuracy | svamp | mbpp | ifeval | bfcl irr | bfcl mult | bfcl simple |
|---|---|---|---|---|---|---|---|---|
| drpo3b-20 (20) | +13.5% | -0.3 | +21.4% / -0.6 | +10.2% / -1.6 | +12.0% / +1.3 | +13.7% / +0.3 | +8.5% / -1.3 | +8.0% / +0.3 |
| 3d-5 (25) | +12.7% | +0.9 | +22.2% / +0.9 | +4.0% / +1.3 (table: +1.2) | +13.8% / +1.9 | +12.5% / -0.9 | +7.1% / 0.0 | +9.6% / -0.6 |
| 3d-10 (30) | +14.8% | -0.6 | +24.8% / -1.9 | +11.6% / -1.3 | +10.1% / +0.9 | +13.0% / -0.3 | +11.9% / +0.6 | +9.9% / -1.3 (table: -1.2) |
| 3d-15 (35) | +18.8% | +0.3 | +30.9% / -0.3 | +17.6% / 0.0 | +13.6% / +1.6 | +13.0% / -0.3 | +12.1% / 0.0 | +10.2% / +0.6 |
| 3d-20 (40) | +20.5% | -0.8 | +29.6% / -2.5 | +14.6% / +0.6 | +22.7% / -0.6 | +16.7% / -1.6 | +13.7% / -2.2 | +12.3% / +1.3 |
| 3d-23 (43, guard stop) | +21.2% | +0.2 | +30% / -0.9 | +21% / +1.9 | +17% / +0.9 | +17% / -1.6 | +18% / -2.5 | +14% / +0.6 |

drpo3e (queued, no result in range): drpo3d-23 continued 40 steps, guard 10 pts, protects toolace/multiple (rl_train.py now logs/guards per ToolACE category and --protect-sources accepts source/category).

#### C.1.20 BFCL multiple error breakdown: drpo3d-23 vs original Qwen — 26 Oct
Fresh x8, 40 questions x 8 = 320 answers each (scratchpad bfcl_errors.py; records f8-base, f8-drpo3d-23).

| measure | original Qwen | drpo3d-23 |
|---|---|---|
| right answers | 243 | 235 |
| mean thinking tokens | 182 | 145 |
| wrong: right function, wrong arguments | 28 | 34 |
| wrong: wrong function | 7 | 10 |
| wrong: unparseable | 10 | 9 |
| never solved by either | 32 | 32 |

Loss concentrated in 7 questions (-15 answers), partly offset by 4 improved (+5); on the 7, thinking ~230 -> ~170 tokens. Error examples: allowed value not matched to the function's enum (mode "public" / "public transport" instead of "transit"; "vegetarian" instead of "vegan"); dotted function name shortened (soccer_stat(...) instead of soccer_stat.get_player_stats(...)).

#### C.1.21 teach1 / teach1b — stop teaching with own confidence, 26 Oct
teach1: trained on nothing (bug): teach_stop.py's judge loop reused the name `rows` for its mask, overwriting the answer list; step 3 found 0 teacher answers ("teachers: own confidence 0, judge 0, keep thinking 0"; p(</think>) nan). Both adapters were copies of the raw model. Setup before the bug: 19,466 paragraph ends; soft judge predicted success 0.34 at cuts vs 0.62 at full thinking.
teach1b (fixed, reusing teach1's 2,512 answers): CUDA out of memory at variant rel:0.05, 1,800/2,512 answers. Progress before crash: gsm8k_matched p(stop) 0.53-0.59 at stop targets vs 0.23-0.30 at keep targets; "good stop" 0.61-0.73. Fix: forward pass ends at the last scored paragraph end, gradient checkpointing on, teacher points cached; requeued as teach1c.

#### C.1.22 mf1 — queued 26 Oct, no result in range
Method planned: rc1-rl-40 answers lots B5 + V2 SVAMP-like math (matched GSM8K ~1,200 + MetaMathQA ~2,500) x8; pairs from its own answers only: right (right beats wrong within +-25% length) + keep-thinking (longest right beats shortest wrong when longer); no other types; targeted DPO (fresh LoRA on rc1-rl-40, NLL anchor, 3 seeds, soups 25-200 on fresh x8). Dry-tested with stand-in data.

#### C.1.23 Infrastructure events (25–26 Oct)

| time | event |
|---|---|
| 25 Oct ~21:15 | queues reordered; drpo0 interrupted at step 3 (~22:00); GPU 1 relaunched |
| 25 Oct ~23:15 | runtime 1 restart: balround missing from the light-restore list caused a 45-min full Drive restore; fixed |
| 26 Oct ~00:40-01:50 | both Colab runtimes lost (Chrome tab ids changed, pages back to "Connect"); GPU 1 in lot V2 at 750 / 4,368 questions; GPU 2 at drpo3b step 12; relaunched ~01:20 (scripts 1b57c7c); mm3's 300-question test re-sampled (fix f0ab9d7) |
| 26 Oct ~02:30 | resumable saving: rl_train.py saves adapter + optimizer + RNG + log every step (resume.pt, rl_log.jsonl); check_vllm.py appends per 2,000-trace chunk; dpo.py caches reference scores (ref.json), saves latest checkpoint + optimizer, no longer writes the final adapter early (track.py had taken a half-trained seed for a finished one). Not resumable: teach_stop.py, evaluate.py |
| 26 Oct ~07:32 | GPU 2 queue interrupted during teach1-rel0.1 scoring; restarted |
| 26 Oct ~17:00 | teach1b OOM; GPU 2 moved to drpo3e before mf1 was read; restarted so mf1 runs first |

### D.2 Classifier handoff (HANDOFF_CLASSIFIER_2026-10-25.md, written 25 Oct ~10:45 CET)

#### C.2.1 Setup stated in the handoff
- Evaluation: 2,759 questions (MBPP+ 378, BFCL simple_python 400 / multiple 200 / irrelevance, SVAMP, IFEval 541); sampling T 0.6, top-p 0.95, top-k 20. Score = tokens saved (geometric mean over 4 benchmarks, BFCL subsets pooled) and accuracy change.
- Contamination: drop if >= 30% of 8-grams appear in an evaluation question (earlier 50%). No MBPP-family data (mbpp_train excluded) from 24–25 Oct.

Reference results (fresh 240 x 8):

| model | saved | accuracy | SVAMP | MBPP+ | IFEval | BFCL |
|---|---|---|---|---|---|---|
| rc1-rl-40 | 53.9% | -2.1 | 65% / -4.1 | 59% / -1.9 | 46% / -0.6 | 41-45% / -1.2.. |
| Lightning McQwen3 (his full eval x8, no think gate) | 43.2% | -0.85 | 66% / -3.8 | 39% / -1.9 | 32% / +1.4 | 25% / +0.9 |
| stopB3 soup 25 | 34.6% | -0.4 | 36% / -2.2 | 50% / -0.3 | 28% / +2.8 | 20-22% / 0.0..-3.7 |

#### C.2.2 Labelled data pipeline (track.py callot -> probe_curve.py + label_cuts.py)
1. Draw training-source questions, remove earlier draws, contamination check. 2. Model answers each question N times (8 for lots B/B2, 20 for B3/B4; vLLM). 3. Per question, label the longest right reply and (--include-wrong) the longest finished wrong reply. 4. Cuts at paragraph ends, up to 10 per reply (evenly spaced, first always included) + full thinking. 5. At each cut "</think>" is closed and the model answers 20 times (answer cap 2,048 tokens), graded by type (math exact, code tests, IFEval checks, tool-call match); label = k/20.
Judge label: safe = 20/20; unsafe = <= 15/20; 16-19 excluded from training/AUC. Soft target k/20 for "real success rate" probes and calibration.

| lot | model labelled | sources | tries | kept questions | notes |
|---|---|---|---|---|---|
| A | rc1-rl-40 | ifeval_like 400, gsm_ic 200, gsm_plus 200, toolace 300, mc 100, kodcode 200 | 8 | 643 | |
| A2 | rc1-rl-40 | same, seed 49 | 8 | ~580 (A+A2 = 1,222) | |
| B | original Qwen | as A | 8 | ~640 | mc = ARC-Easy + CommonsenseQA |
| B2 | original Qwen | same, seed 50 | 8 | ~640 (B+B2 = 1,279) | |
| B3 | original Qwen | gsm_ic 350, gsm_plus 350, ifeval_like 500 | 20 | 455 (743 new; 268 contamination-dropped) | 792 replies (422 right), 7,436 cuts |
| B4 | original Qwen | kodcode 300, mbpp_train 200, toolace 400, hermes_tools 250 | 20 | 647 (15 dropped) | 921 replies (460 right), 7,651 cuts; mbpp_train dropped downstream |
| bench40 | original Qwen | 240 choosing-set benchmark questions | 8 | 240 (304 incl. right/wrong replies) | measure only; 1,812 cuts |

#### C.2.3 Features (extract_feats.py, one forward pass per cut) and own-confidence readings (own_conf.py)
- points.npy [n, 28 layers, 2 places, 1024]: hidden state at the last thinking token before "</think>" and at "</think>".
- window.npy [n, 3, 64, 1024]: 64 tokens before "</think>" at layers 14, 21, 28.
- own_conf.py: one greedy answer per cut; own_confidence (geometric-mean token prob.), own_min, own_low5, own_first50, own_certainty (minus mean entropy), answer text (answer stability = same as previous cut).

#### C.2.4 Classifiers (probe_zoo.py)
Dev = training lots, 5 folds by question (OOF); test = fit on all dev, scored on a hash-held-out 30% (practice test) or a separate lot (--test-cal, e.g. bench40).

| classifier | input | model |
|---|---|---|
| position | cut fraction, log fraction, type one-hot, type x fraction | logistic regression |
| own confidence (5 variants), answer stability | readings as-is | none |
| confidence using the right answer | teacher-forced prob. of the full reply's answer (reference only) | none |
| layer sweep | each layer x place | logistic (standardised, L2, LBFGS); selected layer 20, after "</think>" |
| linear | layer 20 after | logistic |
| linear, real success rate | layer 20 after, soft target | logistic (calibrated) |
| linear, one per type | layer 20 after | one logistic per type |
| mlp | layer 20 after | 1024 -> 256 -> 64 -> 1, ReLU, dropout 0.2, AdamW wd 1e-3, 60 epochs, early stop on 15% |
| mlp, all layers | 28 x 2 concat (57,344-d) | MLP -> 512 -> 128 -> 1 |
| attention (layer 14 / 21 / 28 / all) | window | LayerNorm -> 256-d proj + learned position -> 4-head attention pooling (learned query) -> MLP; BCE, AdamW 1e-3, <= 30 epochs, early stop |
| own confidence, learned (25 Oct) | 5 own readings (log) + position | logistic |
| linear + own confidence | layer-20 state + 5 readings | logistic |
| mlp + own confidence | layer-20 state + 5 readings | MLP 256 -> 64 |

Reported metrics: AUC (dev OOF, test; per type and ALL); calibration (log loss, Brier, ECE vs k/20); pooled trade-off (one threshold chosen on dev for a loss budget); per-type stopping simulation (earliest cut with score >= dev-chosen threshold; allowances gsm 0.5 pt, kodcode 1, others 2; and 2 pts for all); oracle 20/20 = first cut with all 20 forced answers right; timing-only AUC (timing_auc(): toy check rising 1.0, difficulty-only 0.5, falling 0.0). Loss accounting corrected twice on 24 Oct: (1) vs natural stop, not vs always right; (2) natural stop = full-thinking cut's own 20 forced answers and length (had been the selected reply's single outcome).

#### C.2.5 Classifier runs up to 25 Oct morning

| run | model read | trained on | test set | top TEST AUC (ALL) as printed | own confidence | position | notes |
|---|---|---|---|---|---|---|---|
| zoo A | rc1-rl-40 | lot A (643 q, 70%) | 279 held-out practice q | MLP 0.852 | 0.790 | 0.718 | per type MLP: math 0.91 (own 0.92-0.93), instr 0.87 (own 0.74), tools 0.88 (own 0.63), code 0.80 (own 0.72) |
| zoo A2 | rc1-rl-40 | A+A2 (1,222 q) | 540 held-out practice q | MLP all layers 0.905 | 0.792 | 0.719 | per type: math 0.91/0.88, code 0.92, instr 0.89, tools 0.69; <= 2 pt pooled: 31% / 2.4 (own 10% / 0.4) |
| zoo B | original Qwen | lot B | 281 held-out practice q | MLP 0.826-0.828 | 0.728 | 0.668 | math 0.92-0.94 vs own 0.87-0.88 |
| zoo B2 | original Qwen | B+B2 (1,279 q / 11,314 cuts) | 545 held-out practice q (4,907 cuts) | MLP 0.864 | 0.740 | 0.699 | per type MLP: math 0.93-0.94, instr 0.85, code 0.78, tools 0.69, MC 0.48-0.63; pooled <= 2 pt: MLP 69% / 1.5, linear per type 65% / -0.7, own 46% / 1.1, position 51% / 2.9; <= 0.5 pt: MLP 67% / 0.5 |
| zbench | original Qwen | B+B2+B3 (2,616 q / 23,657 cuts) | bench40 (1,812 cuts) | MLP all layers 0.738 (dev 0.885) | 0.686 (least likely 0.717) | 0.508 | C.2.6 |
| zoo B4 (first) | original Qwen | B..B4, MBPP dropped | practice | — | | | crashed (lot B3 labels missing on runtime 1); rerun = C.1.1 |
| zB4bench | original Qwen | B..B4 | bench40 | — | | | queued then; results C.1.2 |
| zbench2 | original Qwen | B+B2+B3 | bench40 | — | | | queued, timing-only AUC + combos |

Handoff observation: each doubling of data raised the top judge by +0.04-0.06 AUC; architectures within ~0.05 of each other. Simulated replay on held-out practice questions (24 Oct): 67-69% thinking saved at -0.5 / -1.5.

#### C.2.6 zbench detail (judges on bench40)
Column mapping: gsm_plus = SVAMP, ifeval_like = IFEval, kodcode = MBPP+, toolace = BFCL.

AUC:

| classifier | dev ALL | TEST SVAMP | TEST IFEval | TEST MBPP+ | TEST BFCL | TEST ALL |
|---|---|---|---|---|---|---|
| position | 0.722 | 0.401 | 0.500 | 0.508 | 0.507 | 0.508 |
| own confidence | 0.731 | 0.872 | 0.613 | 0.682 | 0.569 | 0.686 |
| own conf., least likely token | 0.728 | 0.879 | 0.637 | 0.696 | 0.619 | 0.717 |
| own conf., 5 least likely | 0.732 | 0.879 | 0.623 | 0.717 | 0.619 | 0.713 |
| answer stability | 0.660 | 0.764 | 0.567 | 0.634 | 0.579 | 0.634 |
| linear (layer 20 after) | 0.875 | 0.703 | 0.607 | 0.756 | 0.632 | 0.674 |
| linear, one per type | 0.866 | 0.668 | 0.585 | 0.749 | 0.694 | 0.688 |
| mlp | 0.893 | 0.740 | 0.553 | 0.770 | 0.772 | 0.736 |
| mlp, all layers | 0.885 | 0.752 | 0.711 | 0.756 | 0.698 | 0.738 |
| attention, layer 28 | 0.835 | 0.749 | 0.652 | 0.734 | 0.649 | 0.714 |

Dev per type (mlp): arc_easy 0.32, csqa 0.55, gsm_ic 0.95, gsm_plus 0.95, ifeval_like 0.87, kodcode 0.75, toolace 0.77.
Calibration on TEST (soft target): mlp all layers log loss 0.82, Brier 0.217, ECE 0.14 (lowest listed).

Pooled trade-off (threshold chosen on dev; TEST saved / real loss):

| classifier | <= 0.5 pt budget | <= 2 pt | <= 5 pt |
|---|---|---|---|
| position | 10% / 2.3 | 37% / 6.5 | 59% / 11.7 |
| own confidence | 22% / -0.2 | 32% / 1.9 | 42% / 3.8 |
| own conf., least likely | 11% / 0.6 | 23% / -0.4 | 33% / 0.4 |
| linear | 51% / 4.0 | 55% / 5.1 | 55% / 5.1 |
| mlp | 55% / 3.3 | 55% / 3.3 | 55% / 3.3 |
| mlp, all layers | 64% / 6.6 | 65% / 7.8 | 65% / 7.8 |
| attention, layer 21 | 56% / 5.5 | 60% / 6.2 | 63% / 7.1 |

Stopping simulation per benchmark (TEST saved / loss / coverage; negative loss = gain):

| | SVAMP | IFEval | MBPP+ | BFCL |
|---|---|---|---|---|
| oracle 20/20 (bound) | 30% / -1.3 / 59% | 19% / -4.2 / 34% | 41% / -7.7 / 58% | 39% / -5.6 / 54% |
| own confidence | 15% / 3.8 / 49% | 68% / 1.0 / 92% | 94% / 1.0 / 100% | 23% / -1.4 / 38% |
| own conf., least likely | 8% / 1.9 / 33% | 68% / 1.0 / 92% | 94% / 1.0 / 100% | 43% / 0.8 / 57% |
| linear | 42% / 14.9 / 94% | 43% / -0.4 / 58% | 92% / 2.4 / 100% | 31% / 0.8 / 45% |
| mlp | 19% / 8.0 / 76% | 43% / 0.9 / 58% | 93% / 1.0 / 100% | 22% / -1.3 / 35% |
| mlp, all layers | 63% / 23.8 / 100% | 58% / 4.0 / 81% | 92% / 1.0 / 100% | 44% / 0.4 / 57% |
| attention, layer 21 | 50% / 21.3 / 84% | 67% / 1.1 / 92% | 94% / 0.8 / 100% | 44% / 2.4 / 64% |

Cut-success curves:

| set | 0-20% | 20-40% | 40-60% | 60-80% | 80-100% | full | perfect stop: saved / gained |
|---|---|---|---|---|---|---|---|
| lot B3 gsm_ic (397 replies) | 0.25 | 0.46 | 0.47 | 0.49 | 0.52 | 0.52 | 66% / +29.5 |
| lot B3 gsm_plus (184) | 0.21 | 0.47 | 0.47 | 0.46 | 0.50 | 0.47 | 71% / +31.8 |
| lot B3 ifeval_like (211) | 0.77 | 0.73 | 0.75 | 0.73 | 0.66 | 0.73 | 61% / +11.4 |
| lot B4 kodcode (466) | 0.46 | 0.50 | 0.46 | 0.47 | 0.50 | 0.52 | 79% / +22.9 |
| lot B4 hermes_tools (301) | 0.27 | 0.44 | 0.41 | 0.29 | 0.18 | 0.49 | 72% / +9.1 |
| lot B4 toolace (136) | 0.35 | 0.46 | 0.49 | 0.41 | 0.42 | 0.54 | 46% / +6.3 |
| bench SVAMP (51) | 0.26 | 0.34 | 0.48 | 0.58 | 0.55 | 0.73 | 53% / +8.9 |
| bench IFEval (53) | 0.56 | 0.50 | 0.49 | 0.51 | 0.45 | 0.61 | 55% / +11.0 |
| bench MBPP+ (52) | 0.46 | 0.53 | 0.44 | 0.52 | 0.49 | 0.50 | 84% / +17.0 |
| bench BFCL (148) | 0.70 | 0.70 | 0.72 | 0.77 | 0.62 | 0.75 | 51% / +8.7 |

Observed failure mode: practice math (GSM-IC/Plus) reaches its plateau by 20-40% of thinking while SVAMP rises to 0.73 at full thinking; judges trained on practice math stop SVAMP early (e.g. mlp all layers 63% / 23.8 loss); position scores 0.40 on SVAMP. Thresholds chosen on practice data for "<= 0.5 pt" lost 3-8 pts on bench40.

#### C.2.7 Metric issue noted in the handoff (25 Oct)
Pooled AUC mixes (1) which questions/replies will be right at all and (2) when within a reply the answer settles. Own confidence on SVAMP stops rarely in the simulation (15% saved / 3.8 loss; least likely 8% / 1.9) vs oracle 30% / +1.3 gain. Added in probe_zoo.py (commit 36a4a86, toy-tested): timing-only AUC, own confidence as learned classifier, linear + own, mlp + own; thresholds to come from benchmark-like practice data. Caveats: 240 questions, SVAMP 51 labelled replies; bench40 replies sampled 8x; safe = 20/20 is strict.

#### C.2.8 Stop-pair DPO runs summarised in the handoff
targeted DPO: track.py targeted -> targeted_pairs.py -> dpo.py (LoRA DPO, beta 0.1, lr 2e-5, + chosen NLL), 3 seeds, soups scored on fresh x8. Shorten pairs: chosen = earliest safe cut + a right answer from there; rejected = full reply; labels from forced answers (no judge). --tg-rule question (20-try lots): cut safe if it beats the question's own success rate q + margin on half its forced answers and reaches q on the other half; strict 95%-style rule on 8-try lots.

| run | pairs | soup -> fresh x8 (saved / acc) | notes |
|---|---|---|---|
| stopB (lot B, incl. code) | 554 | 10: +11.9 / +0.9; 20: +31.1 / -2.3; 30: +54.2 / -3.6 | soup 30 MBPP+ 82% / -8.1 |
| stopB2 (lot B, no code; one epoch = 21 steps) | 327 | 20: +34.6 / -2.1 | code 37% shorter at 0.0 with no code pairs |
| stopB3 (B+B2, incl. code) | 1,058 | 15: +18.3 / -1.0; 20: +27.1 / -2.0; 25: +34.6 / -0.4; 30: +54.8 / -3.2; 40: +69.1 / -5.7 | soup 25: SVAMP 36% / -2.2, MBPP+ 50% / -0.3, IFEval 28% / +2.8; soups 30/40: MBPP+ 85-94% / -8 |
| stopB4 | — | queued (later removed) | |
| stopB5 | — | queued then; results C.1.12 | |

wronglen (thinking length on fresh x8 by outcome; never solved = 0/8 for the original, 14% of answers). Original mean thinking tokens all benchmarks: right 359 / wrong 1,339 / never solved 1,493 (MBPP+: 998 / 2,955 / 3,255).

| model | right | wrong | never solved |
|---|---|---|---|
| rc1-rl-40 | -58% | -61% | -59% |
| stopB2 s20 | -36% | -43% | -31% |
| stopB3 s25 | -37% | -49% | -46% (SVAMP never-solved -3%) |
| stopB3 s30 | -51% | -77% | -80% |

#### C.2.9 Other items in the handoff
- drpo1 and teach1 described as queued (results: C.1.3, C.1.21). teach1 original design: judge fitted on lots B..B4 scores every paragraph end; LoRA trains p("</think>") toward the target (variants fixed:0.9, rel:0.05).
- Infrastructure 24–25 Oct: two Colab Pro L4 runtimes; queue files plan_line_b.txt (runtime 1) and plan_line_a.txt (runtime 2); Drive sync every 5 min. Incidents: runtime 2 ended mid-labelling ~20:10; relaunches blocked by Drive consent pop-up and a stuck page; a full Drive restore took hours (fixed: light restore); runtime 1 stopped after "Disconnect and delete runtime" was clicked on a half-connected page (~02:40 CEST, cause not proven); runtime 2 ended ~03:13 CEST; zoo B4, stopB4, drpo1, teach1 then failed on three bugs (fixed in e7af185: ensure_lots() restores a lot from the other runtime; langdetect installed; mixed_pool parses rows). Compute units not the cause (78.6 left).

---

## Part E — 26–30 Oct: runs outside the LUPIN pool, analyses, literature, decisions

Scope: everything in that range not covered by LUPIN_ROUND1_DOSSIER.md (pools dp1–dp15, held-out checks, top-ups, pair kinds, lpb/lpc/lpv/lpw runs, pertype, dial, cross-checks, Wanda arm C results are there and are not repeated). Plus summaries of LUPIN_SPEC.md and SPARSE_SPEC.md.

Conventions: "saved / acc" = tokens saved (%) / accuracy change (pts) vs the original Qwen3-0.6B on the same questions. "fresh x8" = questions_fresh20261022, 40 questions per part (240), 8 samples. "choosing x4" = questions_val20261023, 240 x 4. Overall = 4 benchmarks, BFCL's three parts pooled (subset weighting 40/40/40 unless stated). "OUTSIDE" in the source = outside the project's accuracy limit. Missing values = —.

Model lineage referenced below (defined earlier in the log): rc1-rl-40 = DPO soup rc1-soup-30 + RL (rlcal), fresh x8 +53.9 / −2.1 (SVAMP +65 / −4.1). drpo3b = DRPO from the original Qwen, math protected; drpo3d = continuation from drpo3b-20; drpo3d-23 fresh x8 +21.2 / +0.2. DRPO (rl_train.py) = right answers pushed up with a softmax(relative length / λ) weight, wrong answers pushed down length-blind; larger λ = softer length preference.

---

### E1. mf1 — math repair of rc1-rl-40 (26–27 Oct)

Method: rc1-rl-40 answered the SVAMP-like math of lots B5 + V2 (3,700 questions: matched GSM8K + MetaMathQA SV) x8; contamination check 0 of 3,700 flagged. Pairs from its own answers only, math only: "right" (right > wrong within ±25% length) and "guard"/keep-thinking (longest right > shortest wrong when longer); no cut labels. 2,850 pairs: gsm8k_matched guard 397 / right 413; metamath_sv guard 995 / right 1,045. Fresh LoRA DPO on rc1-rl-40, NLL anchor on, mix guard 0.49 / right 0.51, 176 steps = one epoch per seed, 3 seeds, soups by step. Tripwire 0; pairs ordered right 79–92% late in seed 0.
Evaluation: fresh 240 x 8, soups.

| soup step | overall | svamp | mbpp | ifeval | bfcl irr | bfcl mult | bfcl simple |
|---|---|---|---|---|---|---|---|
| (start rc1-rl-40) | +53.9 / −2.1 | +65 / −4.1 | — | — | — | — | — |
| 25 | +45.6 / −1.4 | +50 / −0.3 | +47 / −2.5 | +47 / −1.9 | +42 / −0.9 | +38 / −3.1 | +35 / +0.9 |
| 50 | +32.2 / −0.5 | +30 / −0.3 | +29 / +2.5 | +35 / −2.8 | +37 / −0.3 | +34 / −3.8 (also logged −3.7) | +32 / 0.0 |
| 75 | +33.0 / −1.3 | +24 / −0.6 | — | +42 / −3.7 | — | +35 / −3.7 | — |
| 100 | +31.4 / −0.5 | +23 / +1.3 | +24 / +2.5 | +43 / −5.0 | — | +35 / −4.1 | — |
| 150 | +29.9 / −0.3 | +24 / +1.3 | — | +38 / −1.9 | — | +34 / −3.7 | — |

No soup 200 (one epoch = 176 steps). Recorded observation: saving fell on every type, not only math, as steps increased (math-only pairs with long chosen replies + NLL anchor).

### E2. stopV3 — stop pairs from the original Qwen (26–27 Oct)

Method: stopV2 rebalanced. Same cut-label lots (B..B5 + V2), gate and per-type caps as stopV2; guard pairs dropped; "keep thinking" (hard) pairs capped at 20% of each type; stop-here (shorten at a cut) and be-right pairs unchanged. 5,060 pairs (matched GSM8K shorten 722 / hard 254 / right 324; MetaMathQA 655 / 276 / 369; KodCode 833 / 208; IFEval-like 836 / 123; ToolACE 212 / 53; Hermes 156 / 39). DPO from the original Qwen, NLL anchor, 300 steps per seed, 3 seeds, soups 25–300. Pairs ordered right 78–84% during training. (Context: stopV2 soups 25/75/100 were −9.6 / 0.0, −9.9 / +0.9, −15.8 / +0.2, i.e. longer.)
Evaluation: fresh 240 x 8.

| soup step | overall | svamp | mbpp | ifeval | bfcl simple | bfcl mult | bfcl irr |
|---|---|---|---|---|---|---|---|
| 25 | +13.4 / −0.7 | +18 / −0.6 | +12 / +0.6 | +14 / −0.9 | +10 / −0.3 | +11 / −2.2 | +7 / −2.8 |
| 50 | +16.4 / −1.4 | — | — | — | — | — | — |
| 75 | +11.8 / +0.2 | — | — | — | — | — | — |
| 100 | +10.0 / −0.6 | — | — | — | — | — | — |
| 150 | +12.4 / +0.5 | — | — | — | — | — | — |
| 200 | +16.2 / −0.9 | 0 / −3.7 | +32 / 0.0 | +21 / +1.6 | 7–8% (BFCL) | — | — |
| 300 | +25.5 / −0.6 | +2 / +0.6 | +54 / −4.1 | +26 / +2.5 | +14 / −1.2 | +7 / −2.8 | +2 / 0.0 |

### E3. Analyses on existing data (26 Oct, no training)

#### 3.1 Density map (artifact "ThinkingCap Density Map")
All 71 fresh-x8 results.json files (one question-set fingerprint; recomputed scores match logged headlines). Noise: an untouched copy of the original vs the original: SVAMP −42 thinking tokens / −0.3 pts, MBPP+ −1.5 pts. Lowest thinking-token model within 1 pt of the original, per benchmark (best-of-71 on 40 questions each, selection on the same data):

| benchmark | model | thinking tokens | original |
|---|---|---|---|
| BFCL pooled | drpo1-60 | 67 | 174 |
| MBPP+ | rep4-blend50 | 742 | 2,050 |
| IFEval | bon-dpo-r2g step 10 | 193 | 454 |
| SVAMP | clean-b1 step 20 | 405 | 635 |

IFEval reply = 284 tokens with thinking off vs 703 with thinking on.

#### 3.2 "Perfect stop" recomputed strictly (benchlot / label_cuts.py data, pairs-bench40/cal)
Original rule: earliest cut whose 20 forced answers are right at least as often as the full thinking's 20, over 2 labelled replies per question (longest right + longest wrong); "accuracy gained" = best cut anywhere minus full thinking. Cells = thinking cut / forced-answer accuracy at that stop minus finishing (pts). Replies (right): SVAMP 36, IFEval 29, MBPP+ 26, BFCL 109. Oracle (cut chosen with labels).

| rule | SVAMP | IFEval | MBPP+ | BFCL |
|---|---|---|---|---|
| original rule, all replies | 53% / +5.0 | 55% / +9.5 | 84% / +13.1 | 51% / +8.2 |
| right replies only, as right as finishing (20/20) | 41% / 0.0 | 49% / +1.7 | 72% / 0.0 | 49% / +0.4 |
| right replies only, cut right ≥ 19/20 | 47% / −1.0 | 49% / +0.7 | 74% / −0.6 | 49% / +0.4 |
| right replies only, as right as finishing on both halves | 41% / 0.0 | 45% / +1.4 | 72% / 0.0 | 49% / +0.4 |

#### 3.3 Does own confidence predict a failing answer early? (teach1's saved data, CPU)
2,506 original-Qwen answers (one per question), 21,978 paragraph-end readings; own_low5 = geometric mean of the 5 least likely tokens of a greedy forced answer. AUC of "answer will end wrong" from low own confidence (0.5 = no signal):

| type (right/wrong) | 1st paragraph | 2nd | by 25% of thinking | by 50% | at end |
|---|---|---|---|---|---|
| matched GSM8K (541/79) | 0.48 | 0.50 | 0.38 | 0.46 | 0.64 |
| MetaMathQA (476/87) | 0.55 | 0.51 | 0.54 | 0.52 | 0.76 |
| KodCode (213/99) | 0.70 | 0.70 | 0.68 | 0.66 | 0.60 |
| IFEval-like (192/50) | 0.57 | 0.60 | 0.63 | 0.58 | 0.60 |
| ToolACE (260/225) | 0.67 | 0.62 | 0.64 | 0.65 | 0.50 |
| Hermes (130/154) | 0.67 | 0.80 | 0.68 | 0.69 | 0.57 |

"Give up" threshold at the 2nd paragraph stopping ≤ 5% of right answers catches 7–19% of wrong ones (code 19%, Hermes 14%, math 8–9%, IFEval 8%, ToolACE 7%). A hidden-state probe on "ends wrong" was not tried.

### E4. drpo3f — gentle DRPO continuation (27 Oct)

Method: replaced drpo3e. From drpo3d-23 (+21.2 / +0.2), DRPO λ 0.3 (drpo3d used 0.5), math and toolace/multiple protected, per-type guard 10 pts, planned 40 steps scored every 5. ~8 min per training step. Runtime restart at ~02:11 resumed after step 2. Run ended after its step-20 score; no step 25–40 scores logged, stop reason not printed in visible output.
Evaluation: fresh 240 x 8.

| step | overall | svamp | mbpp | ifeval | bfcl simple | bfcl mult | bfcl irr |
|---|---|---|---|---|---|---|---|
| 5 | +18.8 / −1.6 | — | — | — | — | — | — |
| 10 | +23.6 [CI +19.5, +27.9] / +0.23 [CI −1.80, +2.37] | +29.5 / −1.56 | +22.2 / +2.19 | +26.0 / +0.62 | +13.5 / +1.56 | +18.2 / −0.94 | +16.8 / −1.56 |
| 15 | +24.6 / +0.5 | — | — | — | — | — | — |
| 20 | +22.5 / +0.4 | — | — | — | — | — | — |

### E5. sp1 — large self-preference round (queued 27 Oct, paused, never run)

Planned: from the original Qwen, michal-bida-style own-answer pairs (shortest right > longest wrong and > much longer right, shorten capped at guard per type), 12,000 questions x 4; math + toolace/multiple protected (only ±25% right > wrong pairs, via balanced_pairs.py --protect). Pool: MetaMathQA 25%, IFEval-like 25%, KodCode 25%, ToolACE 17%, Hermes 8%. Toy-tested. Paused when the user moved to "data first" (dp1).

### E6. Budget prompt test: bp1 (failed) and bp1b (27 Oct)

bp1 (GPU 2): failed at start, "invalid choice: 'budgetprompt'" (stage-list edit lost to a concurrent edit by another session; runtime's track.py older than the plan line). Fixed with an assert and a stage-list check. Not requeued; replaced by bp1b on GPU 1.
bp1b method: original Qwen, test-time prompt "Think briefly: keep your thinking under N tokens" appended to the system message; N = svamp 400, mbpp 750, ifeval 200, bfcl 70. Measurement only (not a weights method). Thinking scanned for budget words (tokens/budget/limit/brief/quick/concise/short) vs the same scan without prompt.
Evaluation: fresh 240 x 8.

| benchmark | reply tokens without -> with | saved | acc |
|---|---|---|---|
| SVAMP | 644 -> 344 | 47% | −2.8 |
| IFEval | 703 -> 526 | 25% | −3.4 (budget words in 68% of thinkings vs 28%) |
| MBPP+ | 2,098 -> 1,922 | 8% | −4.1 |
| BFCL simple / multiple / irrelevance | ~200 -> ~200 | −1 to −3% | −1.2 / −0.6 / 0.0 |
| overall (computed, BFCL ratio ~1.02) | — | ~22% | ~−2.7 |

### E7. ps1 — prompt search on training questions (27 Oct)

Method: 11 prompts (incl. no prompt) on the original Qwen: concise, prefill, no-re-check (+prefill), token budget 100, Chain of Draft, Lightning McQwen3's examples / ex-draft (+prefill), ex-draft with one example per task type. plotV2 lot, 50 training questions per source x 4 replies (gsm8k_matched, metamath_sv, ifeval_like, toolace; no code in this lot); paired against no prompt. Leak = thinking uses a prompt word the question lacks; base (no prompt) reads 15% (IFEval-like 40%). prompt_search.py, prompt_candidates.py.

| prompt | saved | acc | leak | gsm8k | metamath | ifeval-like | toolace |
|---|---|---|---|---|---|---|---|
| concise | +1.4% | +2.0 | 11% | +10 / +1.0 | +6 / +4.5 | −9 / −0.5 | −2 / +3.0 |
| prefill | −4.3% | −1.0 | 13% | −6 / +1.5 | −3 / −0.5 | −21 / −1.5 | +10 / −3.5 |
| nocheck | +5.3% | +2.4 | 12% | +25 / +1.5 | +8 / +5.0 | −9 / +1.0 | −6 / +2.0 |
| nocheck-prefill | +0.2% | +0.9 | 10% | +20 / +2.5 | +4 / +5.5 | −25 / −6.0 | −4 / +1.5 |
| budget (100 tokens) | +21.5% | +1.9 | 22% | +41 / +3.5 | +22 / +5.0 | +4 / 0.0 | +15 / −1.0 |
| cod | +17.1% | −11.9 | 16% | +44 / −30.0 | +26 / −12.5 | −14 / −5.5 | 0 / +0.5 |
| examples | +19.3% | −0.4 | 25% | +48 / +3.0 | +18 / −1.5 | −10 / −3.0 | +10 / 0.0 |
| exdraft | +24.8% | −2.3 | 27% | +54 / −1.5 | +35 / −3.0 | −1 / −3.5 | −7 / −1.0 |
| exdraft-prefill | +24.8% | −1.9 | 24% | +55 / −1.0 | +33 / −3.5 | −14 / −4.0 | +8 / +1.0 |
| draft-typed | +20.3% | −3.5 | 30% | +60 / −1.0 | +24 / −8.5 | −22 / −6.0 | −11 / +1.5 |

200 replies per source. Later user decision (27 Oct night): prompted answers skipped for LUPIN (bp1b cost −2.7 on fresh).

Competitor re-check (27 Oct): 4 public forks of ThinkingCap-Test; work only in basta (Lightning McQwen3) and michal-bida; miroslavmyrha and horychtom no commits. Submissions go by email (git bundle). BottleCap publishes no 0.6B score; README guide: > 20% saved at ≤ 0.2 pt lost per 1% saved.

### E8. gold1 — unanchored stop-token BCE (27 Oct, broken)

Method: stop_gold.py from the original Qwen. Trains only p("</think>") at labelled paragraph ends of the lots' own replies (B..B5 + V2): target 1 where the cut's forced answers are right at least as often as full thinking (and ≥ 25%), else 0; GSM-IC/Plus, MBPP and never-solved replies excluded; per-type caps as stopV2. Binary cross-entropy, LoRA r16, lr 1e-4, 8 replies per step, 660 steps, checkpoint every 100. No anchor on other tokens. Training log: math p(stop) 0.57–0.67 at safe vs 0.15–0.23 at unsafe stops; code 0.60 vs 0.43; tools/instructions barely separated; p(stop) 0.5–0.8 even at "keep" labels (tools, IFEval, code).
Evaluation: fresh 240 x 8.

| checkpoint | saved | acc |
|---|---|---|
| 100 | 96.5% | −62.5 |
| 200 / 300 / 400 / 500 | 95.8 / 95.9 / 96.0 / 95.2% | −62.5 / −62.7 / −62.4 / −62.5 |
| 600 / 660 | 96.7 / 95.6% | −63.0 / −63.0 |

Failure mode (step 100): replies ~18–32 tokens total; 0% right and ~100% unparsable on SVAMP, MBPP+, IFEval, BFCL simple, multiple; BFCL irrelevance 99% (empty reply counts as a refusal). Hypothesis logged: global "</think>" boost with nothing anchoring the rest of the distribution. Fix noted (not run): KL/NLL anchor on other tokens and on the reply, lower lr / fewer steps, 5-reply smoke test. A later runtime restart re-ran gold1's training while walking the plan; gold1, bp1b, teach1c, stopB5 then commented out.

### E9. teach1c — judge-as-teacher stop-token BCE (27 Oct, broken)

Method: teach_stop.py (experiment 5) from the original Qwen: an MLP judge on layer-20 hidden states (fitted on label lots) scores up to 16 paragraph ends per answer of a 2,512-answer mixed pool; target = sigmoid((p − threshold) / 0.03); fresh LoRA trains only p("</think>" next) with BCE, no anchor. Variants rel:0.05 and rel:0.1 (relative rule). teach1c = rerun of teach1b after a CUDA OOM fix (forward pass ends at last scored paragraph end, gradient checkpointing, cached teacher points).
Evaluation: fresh 240 x 8.

| variant | overall | svamp | mbpp | ifeval | bfcl simple | bfcl mult | bfcl irr |
|---|---|---|---|---|---|---|---|
| rel:0.05 | +85.9 / −49.4 | +92 / −89.7 | +96 / −45.0 | +9 / −14.4 | +86 / −88.1 | +85 / −75.9 | +83 / +18.4 |
| rel:0.1 | +85.3 / −55.6 | — | — | — | — | — | — |

Same failure mode as gold1 (model stops answering).

### E10. sb2 — ShorterBetter rerun (27 Oct)

Method: RL from rc1-rl-40, ShorterBetter reward with the paper's β 0.001 (sb1 used 0.005), math protected (correctness-only reward on math), per-type guard 0.06, proportional batch mix, lr 5e-6, checkpoints every 5 steps. The guard stopped training at step 10 (ifeval_like batch accuracy 0.75 -> 0.69).
Evaluation: fresh 240 x 8.

| step | overall | svamp | mbpp | ifeval | bfcl simple | bfcl mult | bfcl irr |
|---|---|---|---|---|---|---|---|
| 5 | +53.7 / −2.9 | +63 / −3.7 | +61 / −2.2 | +46 / −3.1 | +39 / 0.0 | +41 / −5.9 | +45 / −2.2 |
| 10 | +52.6 / −3.2 | +60 / −3.1 | +59 / −4.1 | +47 / −2.8 | +39 / −0.6 | +41 / −5.0 | +45 / −2.8 |

### E11. drpo2 — DRPO from rc1-rl-40, math unprotected (27 Oct)

Method: DRPO from rc1-rl-40, λ 0.5 (drpo1 used 0.2), lr 5e-6 (drpo1 1e-5), new question pool (seed 64), math not protected, 60 steps, checkpoints every 5 (scored 5–30, 40, 50, 60). One seed.
Evaluation: fresh 240 x 8.

| step | overall | svamp | mbpp | ifeval | bfcl simple | bfcl mult | bfcl irr |
|---|---|---|---|---|---|---|---|
| 5 | +55.5 / −2.0 | +65 / −4.1 | +61 / −1.9 | +50 / −0.6 | +39 / +0.3 | +42 / −3.1 | +46 / −0.9 |
| 10 | +56.4 / −3.0 | +67 / −5.6 | +63 / −1.6 | +48 / −1.9 | +39 / −1.6 | +42 / −4.1 | +46 / −3.1 |
| 15 | +56.0 / −2.6 | +68 / −5.6 | +66 / −1.6 | +40 / 0.0 | +39 / −1.6 | +42 / −3.7 | +47 / −4.1 |
| 20 | +57.6 / −3.9 | +68 / −7.8 | — | — | — | — | — |
| 25 | +57.8 / −4.0 | +69 / −9.7 | — | — | — | — | — |
| 30 | +58.2 / −4.7 | +71 / −11.9 | +65 / −1.6 | +49 / −1.9 | +39 / −1.2 | +42 / −4.4 | +47 / −4.7 |
| 40 | +62.1 / −4.6 | +74 / −10.9 | +67 / −1.9 | +58 / −2.5 | +40 / +0.3 | +43 / −4.1 | +47 / −5.0 |
| 50 | +62.9 / −5.4 | +74 / −15.3 | +68 / −1.6 | +59 / −0.9 | +40 / −0.3 | +44 / −5.6 | +48 / −5.0 |
| 60 | +62.7 / −6.0 | +76 / −15.0 | +69 / −2.2 | +54 / −1.9 | +40 / −1.2 | +44 / −6.9 | +48 / −6.2 |

### E12. tsk1 — TokenSkip (built 27 Oct; never trained)

Paper read (arXiv 2502.12067, EMNLP 2025): LLMLingua-2 scores each word of a right chain of thought, top γ share kept in order, LoRA SFT on cut chains (~7.5k GSM8K/MATH, γ in the prompt). Reported GSM8K at ~50% fewer tokens: Qwen2.5-14B −1.7, 7B −5.4, 3B −9.3 pts; LLaMA-8B MATH-500 −30% tokens / −1.9. Cuts words, not steps; instruct-model chains of 200–500 tokens.
Built: tokenskip.py (thinking only, reply verbatim, one right answer per question; --verify keeps an example only if the original model, given the cut thinking, still answers right) + track.py stage `tokenskip` (SFT per ratio, checkpoint per epoch, fresh x8). Local test: ratio 0.6 -> actual 0.62 (247 -> 153 tokens); numbers, equations and the "double-check" step kept.
Planned: original Qwen, label-lot rollouts, ratios 0.5 / 0.7 and 1.0 (uncut control). Run status: compression worked (6,214 questions with a right answer); --verify failed at vLLM engine start (ValueError in request_memory). Fix: free LLMLingua before vLLM, gpu fraction 0.85 -> 0.7. Then paused by the user (28 Oct, no training until data check). Added to LUPIN_SPEC as M12. No results.

### E13. Literature / research passes (27 Oct)

#### 13.1 Pair-building literature (for shorter thinking)
| source | recipe | reported result |
|---|---|---|
| DAST 2503.04472 | add "longer wrong > shorter wrong" to "shorter right > longer right"; SimPO, 1 epoch, max-margin pair per kind per question, bottom ~15% margins dropped | R1-Distill-7B MATH500: correct-shorter only 90.0 / −60% tokens; both kinds 93.6 / −18% |
| Chen et al. 2412.21187 | chosen = reply cut after first right answer + one re-check, rejected = longest | QwQ-32B MATH500 92.8 at −45% (shortest reply as chosen: −22%) |
| ThinkBrake 2510.00546 | BFCL analogue: stop at first right call | — |
| Munkhbat et al. 2502.20122 | shortest right per question, never-solved dropped | ~30% shorter on GSM8K |
| Ada-R1 2504.21659 | shortest right vs the 2 longest of any outcome; DPO β 0.05–0.1, lr 5e-7, 2 epochs | −51% at −1.65 |
| DAST budget / Difficulty-Aware DLER 2510.15110 / Arora & Zanette 2502.04463 | long-vs-short right pairs only when solve rate p ≥ ~0.5; hard questions right-vs-wrong + wrong-longer only | — |
| 2410.08847 | drop near-identical pairs (likelihood displacement) | — |
Notes recorded: DPO kept most accuracy, SimPO compressed most; NLL anchor did not help length DPO in Chen et al.; DLER merge (top-25% updates x0.7) as a dial; RL beats offline pairs only at aggressive compression (DLER, AdaptThink, LC-R1). ThinkingCap-27B blog: instructions and tools cheap, code safe but small saving, math costly.

#### 13.2 RevThink (arXiv 2411.19865, NAACL 2025)
Teacher (Gemini-1.5-Pro) writes forward reasoning Rf (kept if right), backward question Qb, backward reasoning Rb; student 3-task SFT (Q->Rf, Q->Qb, Qb->Rb, weights 1/3), LoRA r32, 3 epochs math / 10 others; test time forward only. Mistral-7B / Gemma-7B, 12 datasets (no SVAMP): +13.5% avg over zero-shot, +6.8% over best distillation baseline; Mistral-7B GSM8K 54.7 -> 60.9 (distillation 56.2), MATH 10.4 -> 15.3; 10% of data beats full fine-tuning on StrategyQA; ablation: all three tasks highest, Qb->Rb alone hurts. Length: "comparable number of tokens" (no numbers). Project note: MetaMathQA GSM_FOBAR / GSM_SV rows are ready-made (Q, Qb) pairs; possible use as an extra NLL accuracy task.

#### 13.3 ALP (arXiv 2506.05256, "Just Enough Thinking")
GRPO, math, DeepScaleR-1.5B; reward = right − β·length·max(p, 1/K), penalty on all rollouts. Table 1 (64 samples, 4,096 budget): MATH-500 0.81 -> 0.80 at −72% tokens; AIME 0.22 -> 0.24 at −42%; OlympiadBench 0.47 -> 0.51 at −36%. β 1e-7, 100 steps, batch 512. No code released. Project note: solve-rate-gated shorten pairs (offline); solve-rate-scaled drpo λ (built as rl_train.py --drpo-solve-scale).

#### 13.4 "Backwards" pass (user idea: answer -> question training)
- No paper found showing answer->question training shortens reasoning. Reversal curse (Berglund et al. 2023) concerns stored facts; Reverse Training 2403.13799 fixes stored-fact reversal (0% -> 99%).
- MetaMath SV + FOBAR: GSM8K 60.6 -> 64.4 (LLaMA-2-7B, Table 3), no length data. Forward DPO vs backward DPO: GSM8K +3.5 vs +0.5 (2601.07199, Llama-3.1-8B). CoThink 2505.22017 attributes verbosity to backward/re-check training (5-question case study).
- VeriThinker 2505.17941 (NeurIPS 2025): LoRA on "is this solution right? Yes/No" only: R1-Distill-7B MATH500 3,790 -> 2,125 tokens at 94.0 -> 94.8; did not work at 1.5B (forgetting).
- Variants listed: Yes/No verdict NLL term mixed into DPO (built as lupin_checker.py + dpo.py --checker, M3); SVAMP backward questions (M4); backward check to choose the chosen answer; answer-first formats.

#### 13.5 Code source search (28 Oct, Mac only, not used)
OpenCoder-LLM/opc-sft-stage2 educational_instruct (MIT, 118,278 tasks, entry_point + asserts). Overlap with MBPP (974): 476 exact texts, 2,285 same function name, 240 share an assert, 85,175 share a 6-gram. Strict clean: 32,613; MiniLM cosine to nearest MBPP ≥ 0.9: 423, ≥ 0.8: 5,120, ≥ 0.75: 10,567 (threshold < 0.75 -> ~22,000 kept). Task tokens p10–p90 19/25/35/50/71 vs MBPP 12/13/16/18/21; solution median 62 vs 44. Not used.

#### 13.6 Competitor recipes (29 Oct, from their repos)
michal-bida: one DPO round, per-domain cap shorten ≤ right-beats-wrong pairs. Lightning McQwen3: terse-prompt distillation SFT -> cut pairs (9k) -> math-only right>wrong DPO (732 pairs). SVAMP: −2.4 / −3.8 respectively. Measured on our matched sets: shorten : be-right ratio after lupin_select matching ~2.2:1 (lpb-l2s50: 1,382 vs 630) and ~3:1 (lpb-l2s50e: 2,070 vs 658). User (29 Oct ~22:00): not to reproduce michal-bida's balance; replaced by per-type pressure caps (lpw-press, covered in LUPIN dossier).

### E14. Drive cleanups (what was removed / kept)

Policy each time: only merged model weights removed (to Drive trash; user empties trash); results.json, records, compare, adapters, lots, rollouts, data kept; every removed model rebuildable via merge.py from base + adapter(s).

| cleanup | date | removed | freed | kept / notes |
|---|---|---|---|---|
| drop1 (runtime stage, Drive mount; checks adapter exists first) | 27 Oct ~03:00 | 28 models: ca1-soup-20/30/40, ca2-soup-10..40, drpo1-20/20x0.5/40/60, rc1-rl-20/60, rc1-soup-10/20, stopB3-soup-15/20/30/40, stopB-soup-10/20/30, stopB2-soup-20, rc2-soup-10/20/30/60, rr1-rl-20 | 34.6 GB | kept sb1-10/20/30 (adapters/sb1-drpo not found); chain/queue starts kept (ca1-soup-10, rc1-soup-30, rc1-rl-40, drpo3b-20, drpo3d-23, mf1-soup-25) and all post-26 Oct 00:40 models. State before: 408.05 GB used (Drive 125.9, Photos 270.2, Gmail 11.6). Scripted Drive clicks ignored, hence a runtime stage. |
| drop2 | 28 Oct | 37 models: gold1-step-*, teach1/teach1c variants, drpo2-5..60, sb2-5/10, stopV2-soup-25..150, stopV3-soup-25..200, mf1-soup-50..150 | 44.7 GB | trigger: Drive full (499.48 / 540 GB org). drivereport: models 79.7 GB (66 x 1.21 GB), adapters 28.2, runs 11.0, data 1.1. After trash emptied: 411.7 GB used (Drive 129.5). Kept: rc1-rl-40, rc1-soup-30, ca1-soup-10, stopB3-soup-25, stopB5 soups, sb1-*, drpo3b-*, drpo3d-*, drpo3f-*, stopV3-soup-300, mf1-soup-25, drpo1-* |
| drop3 | 28 Oct ~21:15 | 29 models (~34 GB): ca1-soup-10, sb1-10/20, stopB5-soup-10..30, drpo3b-5/10/15, drpo3d-5..20, drpo3f-5, drpo1-20/40/60, stopV2/V3 soups (+ drop2 names, local copies) | ~34 GB | kept one or more per method: A stop training stopB3-soup-25, stopV3-soup-300; E rc1-soup-30; F rc1-rl-40; F+math mf1-soup-25; G gentle RL drpo3b-20, drpo3d-23, drpo3f-10/15/20; H strong RL sb1-30. B self-preference, C balanced, D clean: adapters only. |

Fresh-x8 scores recorded for drop1 models (where scored): ca2-soup-20 46.4 / −2.73; drpo1-20 61.7 / −4.92; drpo1-20x0.5 59.8 / −3.72; drpo1-40 73.3 / −6.90; drpo1-60 78.7 / −5.99; rc1-rl-20 54.8 / −3.67; rc1-rl-60 49.6 / −3.39; rc1-soup-10 48.6 / −3.02; rc1-soup-20 54.8 / −4.01; stopB3-soup-15/20/30/40 18.3 / −1.04, 27.1 / −2.03, 54.8 / −3.20, 69.1 / −5.71; stopB soups 10/20/30 11.9 / +0.9, 31.1 / −2.3, 54.2 / −3.6; stopB2-20 34.6 / −2.1; rc2 soups 20/30 63–64% / −4.9; sb1 10/20/30 58.6 / −3.3, 66.2 / −5.7, 78.2 / −11.2.

### E15. Infrastructure events affecting runs (summary)

| date | event |
|---|---|
| 27 Oct ~02:11 | both Colab runtimes restarted on fresh machines; drpo3f resumed after step 2; stopV3 soups kept |
| 27 Oct | dp2 select on GPU 1 waited up to 8 h for answers that would not come (crash elsewhere); unstuck by user in the evening |
| 27 Oct night | GPU 1 runtime disconnected during a check; GPU 2 queue ended |
| 28 Oct ~03:00 | Drive full; saves could silently fail; GPU 2 blocked until drop2 + trash emptied |
| 28 Oct ~21:20 / ~22:30 | Mac shutdown; sessions restarted; dp10 code answering restarted from zero (chunks not yet on Drive) |
| 29 Oct ~22:03 | Mac restart disconnected both runtimes; ~1.5 h lost |
| 30 Oct ~02:45–04:00 | Wanda on the Mac (8 GB, MPS fp32) stopped after 2 h (7.9 GB swap); moved to GPU stage `wanda`. MLX speed test Qwen3-0.6B: 1 answer 50 tok/s, 4 parallel 156, 8 parallel 209 (16+ OOM; KV cache ~0.11 MB/token); 240 x 4 eval ~1 h per model on the Mac |
| 30 Oct | newtop nofit2 crashed at last line (missing import) after answering; wandaeval crashed (pruned folders lacked tokenizer files); both fixed and re-queued |

### E16. User decisions on direction (chronological)

| date | decision |
|---|---|
| 27 Oct ~00:15 | drpo3e replaced by drpo3f (λ 0.3) |
| 27 Oct | PRIORITY: final model LUPIN. Step 0 representative dataset; step 1 original model answers the pool (optionally prompted); step 2 several pair strategies -> LUPIN0 and variants; step 3 LUPIN1.0 = LUPIN0 + guarded online RL. "Backward" practice (answer -> question) as a variant to test |
| 27 Oct | sp1 paused; data first (dp1) |
| 27 Oct ~12:00 | build and verify dataset before any LUPIN step; keep running stages (sb2, drpo2) |
| 27 Oct night | pair source = model's own natural answers + difficulty protection + repeated rounds; cut pairs only for code, strict labels, small dose; prompted answers skipped; 10 answers per question; pairs must match the evaluation distribution per part and per (solve rate x thinking) cell |
| 28 Oct | no training of any kind until data overlaps the evaluation (tsk1 paused); LUPIN in rounds (v1 on original Qwen's answers; v2 from v1 on v1's answers to a new matched sample) |
| 28 Oct | SPARSE add-on specified (see §18) |
| 29 Oct ~01:00–02:00 | auto-continue rules: dp11 runs only if code not within 5 pts / 10% thinking; 10-answer gate opens on near misses; "run all these steps unless I interrupt or a clear red flag"; red-flag rule: stop a screen variation only if it loses ≥ 10 pts on any one data type at every checkpoint; finalist rule: overall accuracy better than −1 on choosing set |
| 29 Oct | refsets: two more 40-per-part reference sets (seeds 20261024/20261025) -> 160 per part; crosscheck stage on questions_ref20261024 x4 (report only) |
| 29 Oct ~19:00 | optional `codecut` module (code-only cut pairs swapped for 20% of code pairs; keep only if code thinks ≥ 3% less and overall acc ≥ −1, code acc ≥ −2.5, non-code acc ≥ −1). Later superseded with the old-data runs; no results in range |
| 29 Oct ~20:30 | deployment added to roadmap (§17) |
| 29 Oct ~21:45–22:45 | build a v1 from the existing data before any round 2; not to reproduce michal-bida's balance; experiment set P1 tilt, two-stage, per-type adapters, P3, gentle, P2, P4; tools top-up; no-fit fixed at data level (no probe) |
| 29 Oct ~22:45 | data upgrade first, then iterate; decision rule per screen vs baseline (average of two best neighbouring checkpoints): WORTH 3 seeds + fresh soup if (a) acc ≥ base and saving ≥ base +2, or (b) saving ≥ base −2 and acc ≥ base +1; else drop |
| 30 Oct ~00:15 | reference = 160 per part; every score appended to runs/bon/results.csv; no early stop, v1 chosen from all results; "combine levers" = flags of ≥ 2 winning experiments + 3-seed confirm |
| 30 Oct ~08:15 | accept dp13 as pool (gate overridden) |
| 30 Oct ~12:45 | every score also printed with full-eval BFCL weighting 400/200/240 |
| 30 Oct ~13:20 | 2 epochs over same pairs replaced by 1 epoch over up to 6,400 distinct pairs (lpw-press6k) |
| 30 Oct ~13:40 | v1 gate written (§17) |

### E17. LUPIN_SPEC.md — plan summary

- Hard gate (28 Oct): no LUPIN training until the held-out pool check reads OK for all six parts vs choosing set + questions_40, IFEval mix and proportions match, and the user has seen it; enforced in code.
- Goal: > 50% tokens saved at accuracy change better than −1 pt on fresh x8; then one full official evaluation (2,759 x 8). Reference points (fresh x8): michal-bida 47.0 / −0.78; Lightning McQwen3 43.2 / −0.85; rc1-rl-40 53.9 / −2.1.
- Set rules: training questions never evaluation questions (8-word contamination check); choosing set for all decisions (aggregate stats only); fresh set for reported scores only; evaluation settings fixed, thinking on, no test-time switch.
- Step 0 dataset: sources per part, prompt-length bin matching, contamination, gradability; verification tolerances (acc ±5 pts, median reply/thinking ±15%, capped ±3 pts, or inside the 95% interval); planned additions parse rate ±3 pts and answer-timing ratio ±0.1.
- Step 1 answers: 10 per question (user decision; 4 reused + 6 new, ~72,000 answers, ~13 GPU-h); optional prompted set (leak-filtered); optional cut labels on ~1,500 questions; per-question stats stored.
- Step 2 (Stage 1 offline): pairs matched to evaluation per part and per cell (2.0). Pair kinds: shorten, guard, hard (DAST), right, cut, brake, prompted, first-check. Recipes R1 long2short, R2 difficulty (min-solve), R3 DAST, R4 Ada-R1, R5 right-only (built); R6 cut oracle (partly), R7 first-check cuts, R8 prompted (to build), R9 per-part blend. Methods M1 DPO+anchor (β 0.1, lr 2e-5, LoRA r32, 1 epoch), M2 no anchor, M3 checker practice (built), M4 SVAMP backward questions, M5 SimPO, M6 KTO, M7 listwise DPO, M8 prompt-distillation SFT first, M9 stop-token loss, M10 iterative round, M11 curriculum, M12 TokenSkip densify. Free dials: soups, strength blend (x0.5/x0.7), per-part adapters. Protocol: screen (choosing x4, 1 seed) -> narrow -> confirm (3 seeds, fresh x8) -> dial -> LUPIN0.
- Rounds: v1 (original Qwen's answers) -> v2 (v1 answers a new sample matched to v1's own behaviour). Per-round answer statistics: accuracy, thinking tokens, paragraphs, length CV, re-check words, repetition, gzip redundancy, first answer position, vocabulary diversity (MATTR, entropy, distinct tokens per 10k).
- Step 3 (Stage 2 RL from LUPIN0 -> LUPIN1.0): RL1 DRPO (built), RL2 solve-rate-scaled DRPO (built), RL3 ALP, RL4 budget reward, RL5 ShorterBetter, RL6 DLER (partly), RL7 progressive cap, RL8 stop reward, RL9 checker reward, RL10 online DPO, RL11 per-part λ. Protocol: RL2 vs RL1 first; LUPIN1.0 must improve on LUPIN0 or LUPIN0 is submitted.
- Compute estimates: step 0 1–2 h; step 1 10–12 h (8 answers); screen ~10–12 h; confirm ~15 h; ~10 h per RL run.
- Deployment (29 Oct): merge, upload as Arsenefirst/Lupin-Qwen3-0.6B (private first), serve with vLLM on a hosted L4 (HF Inference Endpoint / RunPod / Modal, ~$0.5–1/h, user approves spending), monitor /metrics, side-by-side demo vs original.
- v1 gate (30 Oct): before any v2, a v1 candidate must meet all of: (1) fresh x8, 3 seeds, full-eval weighting: acc better than −0.8 at ≥ 45% saved; (2) cross-check on refs 20261024 x4 within ~1 pt of fresh; (3) no type worse than −3 on both sets; (4) saving agrees within ~3 pts. Final model also gets the full official evaluation.

### E18. SPARSE_SPEC.md — plan summary (28 Oct; add-on, separate from LUPIN)

Question: how many weights Qwen3-0.6B needs, and whether a pruned model learns shorter thinking more or less easily. Reuses LUPIN pool/answers/pairs; no new data. Scored on fresh x8 per type; budgets v1 < 3 pts lost, v2 < 1 pt.

| arm | model | target |
|---|---|---|
| A | raw Qwen | reference |
| B | LUPIN fine-tune | < 1 pt |
| C | Wanda one-shot pruning (sweep 10–50%, calibrated on Qwen's own reasoning answers) | v1 < 3, v2 < 1 |
| D | Lottery Ticket (iterative 20% per round, rewind to original weights, retrain by self-distillation on own right answers; random-mask control; linear layers of 28 blocks only) | v1 < 3, v2 < 1 |
| E | pruned model + one LoRA round on LUPIN v1 pairs, vs B with one round | within pruned budget |
| F | pruned model, kept weights frozen, removed weights re-grown from zero and trained to think shorter (1% / 5% / 20% / all; random-set control) | same |

Order/cost: pilot on Wanda mask (C ~3 GPU-h; E + matched B ~4; F ~10; total ~15–20 GPU-h); D and E/F on its mask only if the pilot warrants (~25–45 GPU-h). Literature cited in spec: LTH on small nets 10–20% weights, BERT-base 30–60%; no published true LTH on a decoder LLM; Wanda ~50% on 7B–70B Llama, collapses near 80%, small models break sooner (SparseGPT ahead at 0.5B–3.8B); pruned reasoning models lose accuracy and think longer, own-reasoning calibration recovers much (Qwen3-8B at 50%: MATH 0.56 web-text vs 0.86 own-reasoning calibration); unstructured sparsity gives no speed/memory gain on normal GPUs. Status in range: only arm C run (results in LUPIN dossier §5); next step E or finer 10–20% levels left to the user, not queued.

---

## Part F — LUPIN round 1 (27–30 Oct)

### 2. Training data ("pool")

Training questions are never evaluation questions: every candidate passes an 8-word-sequence contamination check
against all 2,759 evaluation questions (sequences found in ≥3 evaluation questions treated as shared templates); MBPP
itself is excluded.

#### 2.1 Sources per part

| part | sources |
|---|---|
| SVAMP (math) | GSM8K-matched, MetaMathQA GSM_SV/FOBAR, "easy" GSM8K (1–2 calculations) and MetaMath rephrasings, all with SVAMP's answer instruction |
| MBPP+ (code) | KodCode-V1 (medium/hard, one function, assert tests from the dataset) |
| IFEval | IFEval-like (argilla) + prompts built from IFEval's own instruction library on no_robots bases (instruction mix matched to IFEval) |
| BFCL simple | Glaive function-calling v2 (one call, grounded arguments) + ToolACE (clean single-turn) + Hermes |
| BFCL multiple | ToolACE (2–4 functions offered) |
| BFCL no-fit | ToolACE "near" no-fit; Glaive question + another Glaive function at embedding ranks 15–60 (nofit_glaive_verified, 6,070) and ranks 4–15 (nofit_glaive_hard, 2,441); every no-fit label checked twice by claude-haiku-5.5 (kept only on NO + NO) |

#### 2.2 Matching method

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

#### 2.3 Pool versions and held-out checks

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

#### 2.4 Answers per question

Every pool question has exactly **10** answers by the original Qwen (newest first: the 10-answer run, then extra-answer
rounds, then the selection answers), graded with the evaluation's graders. Pool size: 12,000 questions (dp10/dp13).

#### 2.5 Topic check (embedding similarity, report only)

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

### 3. Pairs and training

#### 3.1 Pair kinds (from each question's 10 graded answers)

| kind | chosen > rejected |
|---|---|
| shorten | a right answer > a right answer ≥ 1.5× longer (1.2× on always-solved questions with `--easy-ratio 1.2`) |
| guard | the shortest right answer > the longest wrong / capped answer |
| right | a right answer > a wrong answer of similar length (±25%) |
| hard (DAST) | a longer wrong answer > a shorter wrong answer |

Difficulty protection (`--min-solve p`): a question solved < p of the time gets no shorten / guard pair, only a right
pair. Per-part overrides (`--min-solve-part`, `--right-parts`) used in some experiments.

#### 3.2 Matching the pairs to the evaluation (`lupin_select.py`)

Pairs are drawn per part at the evaluation's shares (25% per benchmark; BFCL 400/200/240) and, inside a part, per cell
(solve rate × thinking quartile) at the reference's shares; questions visited round-robin; cells with no possible pair
under a recipe have their share spread over the other cells; then every part is trimmed to the lowest fill ratio so part
shares stay exact. Default target 3,200 pairs (`--lp-total`). Options used: `--max-shorten-ratio` / `--shorten-ratio-part`
(cap of shorten pairs per be-right pair, per part), `--part-share`, `--solve-tilt`, `--think-tilt-bfcl`.

#### 3.3 Training

DPO (`dpo.py`) from the original Qwen: LoRA r32, β 0.1, lr 2e-5, NLL anchor weight 1.0 on the chosen reply, 16 pairs
per step, 1 epoch, 48 pairs held out as a tripwire; checkpoints every 25 steps (every 10 in some confirms). Screens: 1
seed, each checkpoint merged and scored on the choosing set ×4. Confirms: 3 seeds averaged ("soup") per checkpoint, scored
on the fresh set ×8. Cross-checks: the same soups on `questions_ref20261024` ×4 vs the original Qwen on that set.

#### 3.4 Pair inventory (pool dp10, long2short, p < 0.5, before matching)

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

### 4. Results

Format: tokens saved (%) / accuracy change (pts) vs the original Qwen on the same questions; per-part columns likewise.
"Steps" = DPO steps of 16 pairs.

#### 4.1 Screens on pool dp10 (choosing set ×4, 1 seed)

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

#### 4.2 Confirms on pool dp10 (fresh set ×8, 3-seed soups) and cross-checks

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

#### 4.3 Screens on pool dp13 (choosing set ×4, 1 seed)

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

#### 4.4 Confirms on pool dp13 (fresh set ×8, 3-seed soups)

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

### 5. SPARSE arm C (Wanda pruning of the original Qwen3-0.6B, no retraining)

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

### 6. Files

- `EXPERIMENT_LOG.md`: full chronological log (all runs, decisions, bugs).
- `LUPIN_SPEC.md`: plan, pair kinds, recipes, v1 gate.
- Code: `build_pool.py`, `select_pool.py`, `pool_check.py`, `topic_check.py`, `lupin_pairs.py`, `lupin_select.py`,
  `dpo.py`, `track.py` (stages: poolfix / poolmore / poolup / tooltopup / newtop / lupinanswers / lupin / pertype / dial /
  crosscheck / codecut / wanda / wandaeval), `nofit_verify.py`, `wanda_prune.py`.
- Every score is also appended to `runs/bon/results.csv` on the project Drive (from 29 Oct night).

