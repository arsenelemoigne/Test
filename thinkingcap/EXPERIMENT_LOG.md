# Experiment log (Qwen3-0.6B, shorter thinking)

Goal: cut tokens as much as possible while losing **at most 3 accuracy points** on the
evaluation. Everything is measured with the official `evaluate.py` / `compare.py` on the same
subset: 40 questions per benchmark (240 total), 2 samples each, L4 GPU, bf16.

## Reference: the untouched model (raw Qwen3-0.6B)

| benchmark | accuracy | mean tokens |
|---|---|---|
| SVAMP (arithmetic) | 81.3% | 778 |
| MBPP+ (code) | 57.5% | 1,699 |
| IFEval (instructions) | 52.5% | 710 |
| BFCL irrelevance / multiple / simple | 85.0 / 76.3 / 82.5% | 205 / 223 / 190 |

## Scoreboard (vs the raw model)

| run | method | tokens saved | accuracy change | within the 3-pt limit? |
|---|---|---|---|---|
| Round 1 (T4) | SFT on pruned traces | +39.1% | -6.3 pp | no |
| Round 2 | SFT on pruned traces, margin + balanced | +59.6% | -8.0 pp | no |
| RL loop 1 | RL, TokenProbe-KL + ShorterBetter, alpha 0.5 | +5.7% | -1.2 pp | yes |
| RL loop 2 | same as loop 1, continued | stopped at step 4 (lengths not falling) | - | - |
| RL loop 3 | RL continued from loop 1, stronger pressure | running | | |

Accuracy ranges (95%) are about +/-4 pp on this subset, so differences under ~3 pp are noise.

## Training data

| run | questions | sources | samples per question | token cap |
|---|---|---|---|---|
| Rounds 1-2 (SFT) | 450 fixed (150 per source), 2 epochs | GSM8K, ARC-Easy, CSQA | 2 | 2,048 |
| RL loop 1 | 384 fresh (12 steps x 32), each seen once | GSM8K, ARC-Easy, CSQA | 8 | 2,048 |
| RL loop 3 | 640 fresh (20 steps x 32), different slice (seed 1) | GSM8K, Skywork, ARC-Easy, CSQA, balanced | 8 | 2,048 |

The RL pool is the full training split (~40k questions), so no question repeats across loops.

## Hyperparameters

| | loop 1 | loop 3 |
|---|---|---|
| start from | raw model | loop 1 adapter |
| LoRA rank / lr | 16 / 2e-5 | 16 / 2e-5 |
| ShorterBetter length weight alpha | 0.5 | **1.0** |
| TokenProbe KL anchor beta | 5e-3 | 5e-3 |
| push on redundant tokens lam_decay | 0.2 | **0.5** |
| token budget = core length x scale | scale 3.0 -> 1.5 | scale **2.0 -> 1.0** |
| core windows | 10 windows of 10 tokens | same |
| steps x questions x samples | 12 x 32 x 8 | 20 x 32 x 8 |

## Per-step training numbers

`accuracy` and `tokens` are on that step's 256 training answers. Each step draws new
questions, so these jump around with question difficulty; the trend matters, not one step.

**RL loop 1**

| step | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| accuracy % | 41 | 50 | 49 | 51 | 59 | 58 | 54 | 61 | 64 | 71 | 63 | 61 |
| mean tokens | 957 | 932 | 906 | 722 | 645 | 686 | 817 | 776 | 939 | 578 | 593 | 535 |

**RL loop 2** (stopped): accuracy 57, 49, 52, 53 %; tokens 946, 832, 847, 866.

**RL loop 3**

| step | 1 | 2 |
|---|---|---|
| accuracy % | 55 | 49 |
| mean tokens | 977 | 830 |

## Evaluation details

**RL loop 1** (tokens saved / accuracy change): BFCL irrelevance +4.3% / -5.0;
BFCL multiple +12.2% / +8.8; BFCL simple +9.0% / +1.3; IFEval +16.1% / -1.3;
MBPP+ -18.1% / -3.8; SVAMP +12.6% / -1.3.

**Round 2 (SFT)**: BFCL +32-39% / -3.8 to -10.0; IFEval +40.3% / +3.8; MBPP+ +86.4% / -6.3;
SVAMP +48.8% / -23.8.

---

# Track 2: MVT cut, fading worked examples, marginal token analysis (branch `mvt-fading`)

Own Colab runtime (L4, bf16), same subset (40 questions per benchmark, 240 total, 2 samples).
Everything runs through `track.py` from `mvt_fading_colab.ipynb`.

## Reference rerun on this runtime (raw Qwen3-0.6B)

| benchmark | accuracy | mean tokens |
|---|---|---|
| SVAMP | 82.5% | 821 |
| MBPP+ | 57.5% | 1,699 |
| IFEval | 56.3% | 833 |
| BFCL irrelevance / multiple / simple | 83.8 / 76.3 / 82.5% | 202 / 223 / 190 |

## Training data

`track.py rollouts`: the same slice as SFT rounds 1-2. 450 training questions (GSM8K,
ARC-Easy, CommonsenseQA, 150 each), 2 samples, 2,048-token cap, eval sampling settings.

## Experiment 1: Marginal Value Theorem cut

### Run mvt256 (2-step look-ahead rule; flawed, see below)

`prune_traces.py --rule mvt --travel 256 --check-samples 4`, SFT `--max-per-source 70`
(LoRA r16, lr 1e-4, 2 epochs, 205 rows).

Pruning: 302 targets from 450 questions (236 pruned, 5 truncated, 61 original);
mean completion 576 -> 245 tokens (-57%): ARC -67%, CSQA -65%, GSM8K -43%.
112 of 512 correct rollouts failed the natural-answer check.

Settle vs MVT cut (500 rollouts with both): median thinking tokens kept settle 214, MVT 0;
MVT earlier 52%, later 0%, same 48%.

| benchmark | tokens saved | accuracy change |
|---|---|---|
| BFCL irrelevance / multiple / simple | +29% / +29% / +25% | +2.5 / -7.5 / -1.3 |
| IFEval | +67% | +6.3 |
| MBPP+ | +85% | -8.8 |
| SVAMP | +39% | **-21.3** |
| **average** | **+61.6% [52.9, 70.3]** | **-6.5 pp [-10.7, -2.1]**: outside the limit |

**Why it failed.** `mvt_report.py` shows the median p(gold) at the MVT cut on GSM8K is 0.00
for every travel cost (32 to 2,048). On math, p(gold) of the exact number stays near 0 until
the computation ends, then jumps. The rule compared the gain over the next 2 steps (0) with
the average rate (0) and left at once. ARC-Easy settles at 0 tokens anyway (the model knows
the answer before thinking), so MC traces are cut to one paragraph by either rule. The c=64
run with this rule was stopped after 20 questions: it would cut math the same way.

**Fix (`d96087f`)**: leave only when no later point of the trace pays better per token than
the average rate g(t) / (t + c). With a step-shaped curve, it waits for the jump. Runs with
this rule are named `mvth<c>`.

### Run mvth1024 (full look-ahead rule, travel 1024)

Same settings as mvt256. Pruning: 302 targets (285 pruned, 8 truncated, 9 original); mean
completion 576 -> 129 tokens (-78%): ARC -78%, CSQA -78%, GSM8K -77% (1,024 -> 239).
Only 17 of 512 correct rollouts failed the check. Settle vs MVT: median thinking kept settle
214, MVT 51; MVT earlier 34%, later 4%, same 63%.

| benchmark | tokens saved | accuracy change |
|---|---|---|
| BFCL irrelevance / multiple / simple | -1% / +27% / +24% | -5.0 / -2.5 / +2.5 |
| IFEval | +59% | +3.7 |
| MBPP+ | +64% | -8.7 |
| SVAMP | +54% | **-23.7** |
| **average** | **+51.2%** | **-7.6 pp [-11.8, -3.5]**: outside the limit |

The fixed rule no longer leaves math at p(gold) 0, yet SVAMP falls as much as before, and the
most patient travel cost tried still cuts the training traces by 78%. The cut point is not the
problem: one-shot SFT on traces that end right after the number appears teaches the model to
stop reasoning early, on held-out math too. Every one-cut SFT variant so far (settle, settle +
margin, MVT 256, MVT 1024) lands between -6 and -8 points with SVAMP -21 to -24.

## Experiment 2: fading worked examples, paragraphs + words (LLMLingua-2 scores)

`fade_traces.py`: one target per question (its shortest correct rollout, 302 questions).
Round k deletes the lowest-gain paragraphs until k x 10% of the thinking tokens are gone, then
keeps the best (1 - k x 10%) of the words inside the remaining paragraphs, scored by
LLMLingua-2 (`microsoft/llmlingua-2-xlm-roberta-large-meetingbank`); words with digits or
math symbols always stay. The current adapter answers naturally twice from each shortened
trace; on failure the trace retries with round k-1's word share, then falls back to round
k-1's trace. SFT: 1 epoch per round from the previous round's adapter, lr 1e-4, r16,
`--max-per-source 100`.

### Round 1 (10% of paragraphs, 90% of words kept)

Targets: 277 faded, 13 faded with fewer words cut, 12 original. Mean completion 518 -> 401
tokens (-23%): ARC -28%, CSQA -27%, GSM8K -15%. SFT on 265 rows (final loss ~0.9, higher
than the cut-trace runs: compacted text is less natural).

| benchmark | tokens saved | accuracy change |
|---|---|---|
| BFCL irrelevance / multiple / simple | +5% / +9% / -12% | -1.3 / +3.8 / -2.5 |
| IFEval | +27% | +1.2 |
| MBPP+ | -39% (longer) | -5.0 |
| SVAMP | +17% | -6.2 |
| **average** | **+4.6%** | **-2.5 pp**: within the limit |

### Round 2 (20% of paragraphs, 80% of words kept), from round 1's adapter

Targets: 292 faded, 6 with fewer words cut, 4 kept round 1's trace. Mean completion 518 -> 324
tokens (-38%): ARC -42%, CSQA -40%, GSM8K -32%.

| benchmark | tokens saved | accuracy change |
|---|---|---|
| BFCL irrelevance / multiple / simple | -0% / +2% / +22% | -7.5 / -1.2 / -8.7 |
| IFEval | +26% | -1.2 |
| MBPP+ | -19% (longer) | -6.2 |
| SVAMP | +33% | -3.7 |
| **average** | **+14.3%** | **-4.3 pp**: outside the limit, so the stop rule ends it |

**Experiment 2 answer: round 1 (+4.6%, -2.5 pp).** Fading protects math far better than one
cut (SVAMP -3.7 at round 2 vs -21 to -24 for every one-cut run), but the model takes up
only part of the shortening (training traces -38%, its own answers -14%), and accuracy
drifts down on the benchmarks it never trained on (BFCL, MBPP+). MBPP+ answers get longer
in both rounds: the telegraphic thinking may not suit code.

## Experiment 3: marginal token analysis (words scored by our own model)

Same fading as experiment 2; only the word scorer changes. `fade_traces.py --word-scorer
marginal`: for every word of the trace (except those with digits or math symbols), delete
that one word and score log p(gold) by teacher forcing; its score is the log-probability
lost. Round k keeps the best (1 - k x 10%) of the words. Scoring took ~85 min for 302
traces on the L4 (GSM8K traces have ~600 words each); later rounds reuse the cached scores.

### Round 1 (10% of paragraphs, 90% of words kept)

Targets: 278 faded, 11 with fewer words cut, 13 original. Mean completion 518 -> 397 tokens
(-23%, the same as experiment 2): ARC -30%, CSQA -28%, GSM8K -15%. SFT on 265 rows.

| benchmark | tokens saved | accuracy change |
|---|---|---|
| BFCL irrelevance / multiple / simple | -12% / +11% / +12% | -3.7 / -1.2 / -1.2 |
| IFEval | +25% | -2.5 |
| MBPP+ | -14% (longer) | **-12.5** |
| SVAMP | +28% | -3.7 (7 lost, 4 gained; the lost ones are wrong numbers, not format) |
| **average** | **+12.1%** | **-5.2 pp**: outside the limit, so the stop rule ends it |

Against experiment 2 round 1 (same 23% cut of the training traces): the model's own answers
shrink 2.6x more (+12.1% vs +4.6%), so the model takes up model-chosen deletions better than
LLMLingua's. The extra accuracy loss is almost all MBPP+ (code, never trained on). With 40
questions per benchmark the average's interval is about +/-4 pp, so -2.5 vs -5.2 is not a
reliable difference: see the 4-sample recheck below.

## Experiment 1b: MVT cut on the easy sources, math kept whole

Idea from the per-benchmark split of experiment 1: the cut saved 59-67% on IFEval with no
loss, while math and code broke. `prune_traces.py --full-sources gsm8k` keeps every GSM8K
question's shortest correct rollout uncut; ARC and CSQA get the MVT cut. SFT as before
(`--max-per-source 70`: 70 ARC + 70 CSQA + 65 GSM8K = 205 rows).

### Run mvth1024-full-gsm8k

Targets: ARC 445 -> 91 tokens (-80%), CSQA 464 -> 97 (-79%), GSM8K whole (984).

| benchmark | tokens saved | accuracy change |
|---|---|---|
| BFCL irrelevance / multiple / simple | +8% / +12% / +21% | +2.5 / +8.8 / +1.3 |
| IFEval | +53% | +3.7 |
| MBPP+ | +76% | -5.0 |
| SVAMP | +22% | -8.7 (10 lost, 3 gained) |
| **average** | **+47.6%** | **-1.5 pp**: within the limit |

The best trade-off so far. SVAMP still loses 8.7 points: of the 10 lost answers, 5 are wrong
numbers after one short paragraph ("I need the total area." then a guess), 3 hit the token
cap, 1 writes "Final answer: 4 + 22 + 16 = 42" (trailing content). SVAMP questions are short
and easy-looking, so the model gives them the brief treatment it learned on ARC/CSQA.
Next: a 4-sample recheck, and a run with fewer MC rows (35 per source) so math weighs more.

### Run mvth256-full-gsm8k

Targets: ARC 445 -> 102 (-77%), CSQA 464 -> 96 (-79%), GSM8K whole.

| benchmark | tokens saved | accuracy change |
|---|---|---|
| BFCL irrelevance / multiple / simple | +26% / +24% / +21% | -1.3 / +0.0 / +1.3 |
| IFEval | +53% | +3.7 |
| MBPP+ | +85% | -6.2 |
| SVAMP | +15% | -6.2 (9 lost, 4 gained; 2 trailing content like "4 + 22 + 16 = 42") |
| **average** | **+53.5%** | **-2.2 pp**: within the limit |

Both travel costs pass: a smaller c (leave sooner) saves more for a little more loss, as the
MVT predicts.

### 4-sample recheck of mvth1024-full-gsm8k

`track.py recheck`: base and adapter rescored with `--samples 4` on the same 240 questions
(the first 2 samples are the same generations as before), compared to a 4-sample base.

| benchmark | tokens saved | accuracy change [95% CI] |
|---|---|---|
| BFCL irrelevance / multiple / simple | +27% / +12% / +15% | -1.2 / -0.6 / +0.6 |
| IFEval | +39% | +6.9 [+1.3, +12.5] |
| MBPP+ | +77% | -10.0 [-19.4, -0.6] |
| SVAMP | +12% | -5.0 [-13.1, +1.9] (17 lost, 9 gained) |
| **average** | **+43.8% [36.5, 51.2]** | **-2.1 pp [-5.7, +1.2]**: within the limit |

Confirmed within the limit with twice the samples. The losses sit in MBPP+ (code, never
trained on) and SVAMP.

### Run mvth1024-full-gsm8k-cap35 (a design mistake)

Meant to make math weigh more by cutting MC rows to 35 per source, but `--max-per-source`
caps GSM8K too: it only halved the data (105 rows). Result +63.4%, **-11.5 pp**, SVAMP -35.0
(30 lost: one paragraph restating the question, then a guess). Not evidence about weighting.

### Why SVAMP still fails, and two fixes queued

The lost SVAMP answers restate the question in one paragraph and answer ("Jesse's room is 11
by 15 feet... Hmm, let me think." then "Final answer: 120"). That is exactly what the MC
targets look like: the MVT cut keeps only the first paragraph, the restatement. Queued:
`--min-steps 2` (keep restatement + one reasoning paragraph) and `--repeat gsm8k:2` (whole
math traces shown twice).

### 4-sample recheck of experiment 2 round 1 (fade-gain-words r1)

| benchmark | tokens saved | accuracy change |
|---|---|---|
| BFCL irrelevance / multiple / simple | +9% / +3% / +6% | -4.4 / -5.6 / -1.9 |
| IFEval | +17% | +2.5 |
| MBPP+ | -29% (longer) | -7.5 |
| SVAMP | +16% | -3.7 (12 lost, 6 gained; all wrong numbers) |
| **average** | **+4.1%** | **-3.2 pp**: just outside the limit |

The 2-sample -2.5 was optimistic: experiment 2's round 1 sits on the edge and saves almost nothing.

### Run mvth1024-full-gsm8k-min2 (`--min-steps 2`)

MC targets keep 2 paragraphs at least: ARC 445 -> 138 (-69%), CSQA 464 -> 134 (-71%).

| benchmark | tokens saved | accuracy change |
|---|---|---|
| BFCL irrelevance / multiple / simple | +11% / +10% / +14% | +0.0 / +1.3 / +2.5 |
| IFEval | +31% | +3.7 |
| MBPP+ | -9% (longer) | -6.2 |
| SVAMP | +14% | -10.0 |
| **average** | **+13.1%** | **-2.8 pp**: within, but saves little |

A second paragraph costs most of the saving and does not help SVAMP: the "restate, then
answer" habit was not the main cause.

### Run mvth1024-full-gsm8k-xgsm8k2 (`--repeat gsm8k:2`, 270 rows)

Same pruning as mvth1024-full-gsm8k; the 65 whole GSM8K traces appear twice in SFT.

| benchmark | tokens saved | accuracy change |
|---|---|---|
| BFCL irrelevance / multiple / simple | +21% / +10% / +17% | +3.7 / +5.0 / -2.5 |
| IFEval | +58% | +2.5 |
| MBPP+ | +64% | -2.5 |
| SVAMP | +3% | -6.2 (8 lost, 3 gained: wrong reasoning on tricky wording, 2 truncated) |
| **average** | **+40.7%** | **-1.0 pp**: within the limit, the smallest loss so far |

### 4-sample rechecks (continued)

| run | tokens saved | accuracy change [95% CI] |
|---|---|---|
| mvth1024-full-gsm8k-xgsm8k2 | **+39.1% [31.5, 47.5]** | **-1.0 pp [-4.2, +2.1]** (SVAMP -3.1, MBPP+ -8.1, IFEval +5.6) |
| fade-gain-marginal r1 (experiment 3) | +5.3% [-2.1, 13.2] | -2.0 pp [-5.1, +1.0] (within) |

With 4 samples experiment 3's round 1 is within the limit (-2.0) and experiment 2's is not
(-3.2): the reverse of the 2-sample ranking. The stop rule cut experiment 3 after one round on
a noisy measurement, so round 2 is run anyway (`--ignore-limit`).

### Run mvth256-full-gsm8k-xgsm8k2 (2 samples)

+43.0% tokens saved, -2.5 pp (within): BFCL +19/+23/+22% (+1.2/+3.8/+2.5), IFEval +43% (+2.5),
MBPP+ +73% (-10.0), SVAMP +13% (-5.0).

## Combination: MVT cut on the easy sources + math compacted by experiment 3

`track.py mix`: ARC/CSQA rows from the MVT cut (mvth1024-full-gsm8k), GSM8K rows from
experiment 3's faded traces (paragraphs and model-scored words removed, checked), shown twice.

### Run mix-mvt1024-marginal1-x2

ARC/CSQA MVT-cut rows + GSM8K rows from experiment 3 round 1 (984 -> 835 tokens), GSM8K twice.

| benchmark | tokens saved | accuracy change |
|---|---|---|
| BFCL irrelevance / multiple / simple | +24% / +22% / +24% | -3.7 / +7.5 / +0.0 |
| IFEval | +35% | +3.7 |
| MBPP+ | +36% | -7.5 |
| SVAMP | +42% | -10.0 (13 lost: one restating paragraph, then an answer with the sum in it) |
| **average** | **+34.6%** | **-3.1 pp**: just outside |

Compacting the math traces brings the SVAMP failure back (reasoning on lost rows 827 -> 512
tokens): math traces have to stay whole. The round-2 mix was dropped from the plan.

### Experiment 3 round 2 (run past the stop rule with `--ignore-limit`)

Targets: 291 faded, 6 fewer words, 5 kept round 1's; mean completion 518 -> 324 (-37%).

| benchmark | tokens saved | accuracy change |
|---|---|---|
| BFCL irrelevance / multiple / simple | -6% / +6% / +4% | +1.3 / -7.5 / -8.8 |
| IFEval | +14% | -7.5 |
| MBPP+ | -26% (longer) | **-18.8** |
| SVAMP | +2% | -3.8 |
| **average** | **-1.3% (longer)** | **-8.8 pp**: far outside |

A second round of word compaction breaks the model: answers get longer (it hesitates), and
code and function calling fall most. Experiment 3's answer is round 1.

### 4-sample rechecks (continued)

| run | tokens saved | accuracy change [95% CI] |
|---|---|---|
| mvth256-full-gsm8k | **+49.7% [42.7, 57.4]** | -2.7 pp [-6.2, +0.7] (within, close to the edge; IFEval +8.8, MBPP+ -10.0, SVAMP -8.8) |
| fade-gain-marginal r2 | +1.8% | -6.2 pp [-9.4, -3.1] (outside; BFCL multiple -19.4) |

### Queued: experiment 3's word scoring on the easy sources, math kept whole

`mix` with ARC/CSQA rows from experiment 3 (round 1 or 2) and GSM8K whole (twice): does
experiment 3 hurt only because it touches math?

| mvth256-full-gsm8k-xgsm8k2 | +42.0% [33.7, 51.1] | -2.1 pp [-5.8, +1.4] (within) |

### Experiment 3's word scoring on the easy sources, math kept whole (2 samples)

| run | tokens saved | accuracy change |
|---|---|---|
| mix-marginal1-mathwhole-x2 (ARC/CSQA from round 1) | -2.6% (longer) | -1.5 pp |
| mix-marginal2-mathwhole-x2 (ARC/CSQA from round 2) | -7.2% (longer) | -5.6 pp (SVAMP lost rows hit the cap: 1,401 -> 2,529 reasoning tokens) |

With math whole, light word compaction of the easy examples shortens nothing, and heavier
compaction makes answers longer and wrong. Experiment 3's round-1 saving came from the
compacted GSM8K traces (SVAMP +28%), which is also what costs math accuracy. Deleting
scattered words gives text the model does not learn to write; deleting whole paragraphs
(the MVT cut) does.

### 8-sample recheck of the winner (mvth1024-full-gsm8k-xgsm8k2)

8 samples per question (the recipe's count) on the 240-question subset, against an 8-sample base.

| benchmark | tokens saved | accuracy change [95% CI] |
|---|---|---|
| BFCL irrelevance / multiple / simple | +21% / +17% / +17% | +1.6 / -1.3 / -0.3 |
| IFEval | +37% | +4.7 [-0.9, +10.0] |
| MBPP+ | +65% | -2.2 [-8.4, +4.1] |
| SVAMP | +13.5% | -5.0 [-10.3, -0.3] |
| **average** | **+37.1% [32.1, 42.2]** | **-0.6 pp [-3.05, +1.74]**: within the limit |

The best-measured result of the track. The remaining loss is SVAMP (34 lost, 18 gained;
mostly wrong numbers).

## Scoreboard, this track (vs the raw model on this runtime)

| run | method | samples | tokens saved | accuracy change | within -3? |
|---|---|---|---|---|---|
| mvt256 | MVT cut, 2-step look-ahead | 2 | +61.6% | -6.5 | no |
| mvth1024 | MVT cut, full look-ahead | 2 | +51.2% | -7.6 | no |
| fade-gain-words r1 | exp 2 round 1 | 4 | +4.1% | -3.2 | no (edge) |
| fade-gain-words r2 | exp 2 round 2 | 2 | +14.3% | -4.3 | no |
| fade-gain-marginal r1 | exp 3 round 1 | 4 | +5.3% | -2.0 | yes |
| fade-gain-marginal r2 | exp 3 round 2 | 4 | +1.8% | -6.2 | no |
| mvth1024-full-gsm8k | 1b: MVT on ARC/CSQA, GSM8K whole | 4 | +43.8% | -2.1 | yes |
| mvth256-full-gsm8k | 1b, travel 256 | 4 | +49.7% | -2.7 | yes (edge) |
| **mvth1024-full-gsm8k-xgsm8k2** | **1b, GSM8K twice** | **8** | **+37.1%** | **-0.6** | **yes** |
| mvth256-full-gsm8k-xgsm8k2 | 1b, travel 256, GSM8K twice | 4 | +42.0% | -2.1 | yes |
| mvth1024-full-gsm8k-min2 | 1b, 2 paragraphs minimum | 2 | +13.1% | -2.8 | yes, saves little |
| mvth1024-full-gsm8k-cap35 | 1b, 35 rows per source | 2 | +63.4% | -11.5 | no |
| mix-mvt1024-marginal1-x2 | 1b + GSM8K compacted by exp 3 | 2 | +34.6% | -3.1 | no |
| mix-marginal1-mathwhole-x2 | exp 3 r1 on ARC/CSQA, GSM8K whole | 2 | -2.6% | -1.5 | yes, saves nothing |
| mix-marginal2-mathwhole-x2 | exp 3 r2 on ARC/CSQA, GSM8K whole | 2 | -7.2% | -5.6 | no |

Plain-language summary: `REPORT_MVT_FADING.md`.

---

# Track 2b: per-type method grid (one runtime; the account allows 2 GPU sessions)

New training data, graded by the evaluation's own checkers (`prepare_types.py`, `typed.py`):
GSM8K (math), MBPP minus every MBPP+ task (code), Hermes single tool calls + 25% "no tool fits"
pairs (tool), argilla IFEval-like (instructions). 250 questions per kind, 2 replies each,
2,048-token cap. For code/tool/instructions the probe reads p(own correct answer | thinking).

| kind | replies | correct | capped | mean tokens |
|---|---|---|---|---|
| math (gsm8k) | 500 | 34% | 39% | 1,465 |
| code (mbpp_train) | 500 | 37% | 36% | 1,375 |
| tool (hermes_tools) | 500 | 52% | 2% | 476 |
| instructions (ifeval_like) | 500 | 73% | 3% | 433 |

Each grid cell trains on one kind only (LoRA r16, lr 1e-4, 2 epochs) and is scored on all four
benchmarks with 4 samples against a 4-sample base (`track.py grid`, `gridtable`).

| run | tokens saved | accuracy | own benchmark (saved / acc) | notable spill-over |
|---|---|---|---|---|
| code-mvt1024 | +57.6% | -3.4 | MBPP+ +93% / -13.7 | BFCL +22-42%, IFEval +27% / -2.5 |
| math-whole | -0.9% | -2.2 | SVAMP +1% / +3.7 | IFEval -6.2 |
| code-whole | -4.0% | -3.5 | MBPP+ -9% / -6.2 | IFEval -8.1 |
| tool-whole | +1.7% | -3.3 | BFCL +2-13% / ~0 | IFEval -7.5, MBPP+ -9.4 |

Code-mvt pruning: 215 traces checked, 154 passed; targets 974 -> 405 tokens (-58%); the MVT
cut is at 0 thinking tokens for most code traces (p(own code) barely rises with thinking).

**Training on one kind of the model's own whole replies already costs 2-3.5 points**, mostly
IFEval and MBPP+: the SFT itself has a price, apart from any cut. Compare each method to the
"whole" run of the same kind.
| text-whole | +3.4% | -0.5 | IFEval +9% / -2.5 | MBPP+ -6.9, SVAMP +7.5 |
| math-mvt1024 | +49.5% | **-11.5** | SVAMP +80% / **-16.2** | MBPP+ -17.5, BFCL multiple -13.7 |

Math-mvt pruning: 326 traces checked, 284 passed; targets 979 -> 262 tokens (-73%). Trained on
math alone, the cut still breaks math: the damage is not spill-over from easy multiple-choice
traces, cutting math reasoning is harmful in itself (and spills onto code and tools).
tool-mvt1024 crashed (vLLM out of memory capturing CUDA graphs next to the scoring model):
fixed with `torch.cuda.empty_cache()` before the check and `enforce_eager`, requeued.
| text-mvt1024 | +13.2% | -1.5 | IFEval +11% / -3.1 | small and mixed: all benchmarks 8-20% shorter |
| **tool-mvt1024** | **+42.0%** | **-0.7** | BFCL +37/+54/+38% / -1.3/+1.9/+5.6 | IFEval +26% / 0, SVAMP +15% / +3.7, MBPP+ +68% / -8.7 |
| math-mvt1024-tokens32 | +37.0% | **-13.3** | SVAMP +55% / -14.4 | MBPP+ -28.1, BFCL multiple -10.6 |

Text-mvt: 391 checked, 328 passed, targets 362 -> 251 (-31%); the thinking on instructions is
short already (median 73 tokens kept by settle, 41 by MVT). Tool-mvt (after the memory fix):
278 checked, 258 passed, targets 291 -> 163 (-44%): **for tool calls the MVT cut is free**.
Math with 32-token batches is as bad as with paragraphs: what hurts math is how much of the
reasoning goes, not where exactly the cut falls.

(The Colab page lost its Google sign-in and then its output stream at ~13:50; the runtime kept
running. Results read from /content/track_log.txt through the Files panel.)
| **code-mvt1024-tokens32** | **+24.1%** | **-1.9** | MBPP+ +57% / -5.6 | everything a little shorter (BFCL 12-19%), ~no loss |

Code-mvt with 32-token batches: 225 checked, 162 passed. Paragraphs made the code cut all-or-
nothing (one long paragraph per plan, so MVT usually kept 0 tokens: MBPP+ +93% / -13.7); token
batches keep the first useful part of the plan (+57% / -5.6). **For code, cut by token batches.**
| tool-mvt1024-tokens32 | +3.9% | -1.6 | BFCL +32/+15/+36% / +1.9/-3.1/-3.1 | IFEval -20% (longer) / +3.7, SVAMP -11%, MBPP+ -7.5 |

For tool calls paragraphs win (+42.0% / -0.7 vs +3.9% / -1.6): tool thinking is short and
well paragraphed, and mid-paragraph cuts make the model write more elsewhere.
| text-mvt1024-tokens32 | +5.2% | +0.6 | IFEval +5% / 0.0 | small, mixed |

MVT part of the grid done. Best unit per kind: tool -> paragraphs (+42%, -0.7); code -> 32-token
batches (+24%, -1.9); instructions -> paragraphs (+13%, -1.5; batches +5%, +0.6); math -> no cut.
| math-marginal | +57.5% | **-19.0** | SVAMP +97% / **-46.2** | IFEval -67% (longer) / -11.9, MBPP+ -17.5 |
| code-marginal | +72.9% | **-18.8** | MBPP+ +90% / -25.6 | SVAMP +90% / -41.2 |
| tool-marginal | +24.1% | -3.0 | BFCL +35/+30/+33% / +0.6/-2.5/-1.9 | MBPP+ -10.0, IFEval -5.0 |

Paragraph marginal (leave-one-out) analysis deletes far too much where steps are redundant:
code targets 974 -> 96 tokens, math 979 -> 287. Deleting any single math or code paragraph
barely moves p(gold) because another paragraph carries the same information, so each looks
free alone, and drop_flat then removes them together while the group test, scored by teacher
forcing on a trace the model did not write, still passes. The MVT cut beats it on every kind.
| text-marginal | +7.4% | -1.3 | IFEval +6% / 0.0 | small |

Marginal by 32-token batches dropped from the plan (paragraph leave-one-out already over-
deletes; smaller units can only delete more). Fading (paragraphs only, 20% per round, 3 rounds,
each round continuing the previous adapter) follows, then the combined model.
| **math-fade-r1** | **+12.6%** | **-1.6** | SVAMP **+24% / +5.0** | MBPP+ -5.6, IFEval -4.4 |

Math fading round 1 (20% of the paragraphs, lowest probe gain first; 79 of 122 traces passed,
912 -> 769 tokens): the first run that shortens math without losing it.
| math-fade-r2 | +15.8% | -4.2 | SVAMP **+35% / +1.9** | IFEval -7.5, MBPP+ -9.4 |

Round 2 (40%; 93 of 122 passed; 912 -> 622 tokens): math itself still shortens without loss;
the drop is spill-over onto IFEval and MBPP+, the cost of training on one kind (see the whole
runs). Candidate for the math part of the combined model.
| math-fade-r3 | +23.0% | -4.7 | SVAMP +50% / **-5.6** | MBPP+ -10.6 |

Math fading sweet spot: round 2 (40% of the paragraphs). Round 3 (60%; 912 -> 376 tokens)
starts costing math. Two combined models will be scored: math whole x2, or math faded to round 2 x2.
| code-fade-r1 | +10.9% | -2.8 | MBPP+ +20% / -10.6 | small |
| code-fade-r2 | +24.1% | -2.1 | MBPP+ +33% / -3.1 | IFEval -5.0, all 16-27% shorter |

Code fading round 2 (875 -> 511 tokens, all 113 passed) ties code MVT by token batches overall
(+24%, about -2): MVT shortens code more (+57% vs +33%), fading loses less on it (-3.1 vs -5.6).
| code-fade-r3 | +21.0% | -5.9 | MBPP+ +14% / **-13.1** | BFCL -4.4, IFEval -6.9 |

Same as math: fading works up to round 2 (40% of the paragraphs) and breaks at round 3.
| tool-fade-r1 | -7.1% (longer) | -2.7 | BFCL irrelevance -30% (longer), multiple -10%, simple +11% | IFEval -6.9, MBPP+ -7.5 |

Fading does not suit tool calls: the answers get longer (the model hesitates on "no tool
fits"); the MVT cut (+42%, -0.7) stays far ahead.
| tool-fade-r2 | +10.3% | -2.4 | BFCL +40/+30/+26% / +1.2/+0.6/-3.1 | MBPP+ -8.7 |
| tool-fade-r3 | +42.6% | -3.6 | BFCL +40/+44/+32% / +1.2/-1.9/-1.3 | MBPP+ -10.0, IFEval -5.6 |

Three fading rounds reach the MVT cut's saving on tool calls, at a higher cost (spill-over).
| text-fade-r1 | +10.0% | -0.3 | IFEval +3% / +0.6 | all ~10% shorter |
| text-fade-r2 | +6.2% | -2.2 | IFEval +11% / -4.4 | MBPP+ -8.1 |
| text-fade-r3 | +12.0% | -0.8 | IFEval +20% / -3.1 | small |

### Grid table (`track.py gridtable`; tokens saved / accuracy change, 4 samples; > = own benchmark)

```
run                          svamp          mbpp           bfcl           ifeval         average
code-fade-r1             +13% / +3.1   >+20% / -10.6    +9% / -2.3     -0% / -1.2   +11% / -2.8
code-fade-r2             +27% / +1.2   >+33% / -3.1    +17% / -1.7    +18% / -5.0   +24% / -2.1
code-fade-r3             +44% / -1.3   >+14% / -13.1   +16% / -2.5     +5% / -6.9   +21% / -5.9
code-marginal            +90% / -41.2  >+90% / -25.6   +34% / -3.1    +17% / -5.0   +73% / -18.8
code-mvt1024              +9% / +2.5   >+93% / -13.7   +29% / +0.2    +27% / -2.5   +58% / -3.4
code-mvt1024-tokens32     +8% / -0.6   >+57% / -5.6    +14% / -1.3     +2% / +0.0   +24% / -1.9
code-whole                -2% / +0.0    >-9% / -6.2     +4% / +0.2    -10% / -8.1    -4% / -3.5
math-fade-r1            >+24% / +5.0    +14% / -5.6     +5% / -1.3     +5% / -4.4   +13% / -1.6
math-fade-r2            >+35% / +1.9     +9% / -9.4     +6% / -1.9     +9% / -7.5   +16% / -4.2
math-fade-r3            >+50% / -5.6    +26% / -10.6   +12% / -2.1     -8% / -0.6   +23% / -4.7
math-marginal           >+97% / -46.2   +30% / -17.5   +15% / -0.2    -67% / -11.9  +57% / -19.0
math-mvt1024            >+80% / -16.2   +43% / -17.5   +42% / -5.2     +3% / -6.9   +49% / -11.5
math-mvt1024-tokens32   >+55% / -14.4   +52% / -28.1   +25% / -3.8     +1% / -6.9   +37% / -13.3
math-whole               >+1% / +3.7     -2% / -5.0     +2% / -1.3     -5% / -6.2    -1% / -2.2
text-fade-r1             +15% / -0.6    +11% / -1.2    +11% / +0.2    >+3% / +0.6   +10% / -0.3
text-fade-r2              +9% / +3.7     -4% / -8.1     +8% / -0.0   >+11% / -4.4    +6% / -2.2
text-fade-r3             +21% / +5.6     -0% / -4.4     +5% / -1.5   >+20% / -3.1   +12% / -0.8
text-marginal             +8% / +1.2     +2% / -5.6    +12% / -0.8    >+6% / +0.0    +7% / -1.3
text-mvt1024             +20% / +3.1    +13% / -5.0     +9% / -1.0   >+11% / -3.1   +13% / -1.5
text-mvt1024-tokens32    +11% / +5.6     -2% / -3.7     +6% / +0.4    >+5% / +0.0    +5% / +0.6
text-whole                +1% / +7.5     -5% / -6.9     +7% / -0.0    >+9% / -2.5    +3% / -0.5
tool-fade-r1              +4% / +4.4     -8% / -7.5   >-10% / -0.8    -14% / -6.9    -7% / -2.7
tool-fade-r2              +9% / +2.5    -11% / -8.7   >+32% / -0.4     +6% / -3.1   +10% / -2.4
tool-fade-r3             +34% / +1.9    +57% / -10.0  >+38% / -0.6    +37% / -5.6   +43% / -3.6
tool-marginal            +11% / +4.4    +34% / -10.0  >+33% / -1.3    +16% / -5.0   +24% / -3.0
tool-mvt1024             +15% / +3.7    +68% / -8.7   >+43% / +2.1    +26% / +0.0   +42% / -0.7
tool-mvt1024-tokens32    -11% / -1.3    +12% / -7.5   >+28% / -1.5    -20% / +3.7    +4% / -1.6
tool-whole               +11% / +3.7     -1% / -9.4    >+7% / -0.0    -12% / -7.5    +2% / -3.3
```

**Best method per kind** (own benchmark, then overall): math -> fading round 1-2 (the only
math cut that keeps SVAMP; whole is the safe default); code -> MVT by 32-token batches (or
fading round 2: less saving, less loss); tool -> MVT by paragraphs (free); instructions -> any
(all cost < 1.5 pts, save 5-13%). Leave-one-out over-deletes math and code.

## Combined models (4 samples vs the raw model)

Retrained 1b on this runtime (mvth1024-full-gsm8k, 2 samples): +42.4%, -3.2 (yesterday +47.6%,
-1.5): retraining plus 2-sample noise moves a run by ~2 points.

| model | rows | tokens saved | accuracy |
|---|---|---|---|
| combine-final: MC MVT + math whole x2 + code MVT tokens32 + tool MVT + text MVT | 860 | +49.0% [40.2, 58.9] | **-6.2 [-9.7, -3.1]** |

Per benchmark: BFCL +44/+42/+28% / +8.1/-11.2/-2.5; IFEval +33% / -6.2; MBPP+ +81% / -11.9;
SVAMP +12% / -5.0. **The cuts add up**: each kind's best cut is safe alone, but pooled they
push brevity everywhere (code is cut 81%, against 57% when trained alone). Queued: the 1b winner
retrained (GSM8K x2) and rechecked with 4 samples, and a light mix (MC + tool MVT, math whole x2,
code and instructions untouched).
| combine-final-mathfade: same with GSM8K faded to round 2 | 860 | +55.3% | **-5.9** (SVAMP +50% / -10.0; lost rows' reasoning 966 -> 357) |
| mvth1024-full-gsm8k-xgsm8k2 retrained (1b winner) | 270 | +31.1% | **-2.2** (within; yesterday's training +39.1% / -1.0) |

Math fading, safe when math is trained alone, breaks math once pooled with the other cuts.
Retraining the 1b winner gives +31-39% / -1 to -2: training randomness matters at this size.
| **combine-final-light: MC MVT + tool MVT + math whole x2 (code, instructions untouched)** | 502 | **+48.6%** | **-2.3** (within) |

Light mix per benchmark: BFCL +40/+48/+27% / +6.2/0.0/-0.6; IFEval +49% / 0.0; SVAMP +4% / +1.2;
MBPP+ +77% / **-12.5** (code is cut by spill-over although no code trace is in the data).

## Weight merging (queued)

`merge.py`: W = W0 + sum w_i dW_i over per-kind adapters, so each kind's update stays separate
and can be turned down. Candidates: 1b winner + tool MVT (1.0); + tool, code tokens32 and
math-fade r2 at 0.5; all at 1.0. `fullbest` then sends the candidate saving most within -2.5
pts (4 samples) to the full evaluation.

---

# Track 2c: shortest-of-8 self-distillation, then on-policy verified MVT + DPO, then RL

## Shortest-of-8 data (matched to the evaluation)

1,500 questions (GSM8K train+test, MBPP minus MBPP+, ToolACE in BFCL's mix, IFEval-like, ARC/CSQA),
8 replies each at the evaluation's caps (4,096; 8,192 for code): 12,000 replies, ~4.3 h on the L4.
Per question the shortest correct reply is kept, nothing edited (`select_shortest.py`); GSM8K only
where that reply is <= 700 tokens (SVAMP-like).

| source | questions | solved | kept | capped | mean reply | kept len | same-question mean | saved |
|---|---|---|---|---|---|---|---|---|
| arc_easy | 48 | 48 | 48 | 1% | 532 | 334 | 532 | 37% |
| commonsense_qa | 252 | 211 | 211 | 0% | 512 | 339 | 494 | 32% |
| gsm8k_all | 300 | 240 | 106 | 22% | 2,163 | 496 | 1,282 | 61% |
| ifeval_like | 300 | 272 | 272 | 2% | 418 | 285 | 409 | 30% |
| mbpp_train | 300 | 168 | 168 | 13% | 2,835 | 1,098 | 1,834 | 40% |
| toolace | 300 | 206 | 206 | 0% | 299 | 190 | 272 | 30% |

**Contamination check** (`contamination.py`, share of a training question's 8-grams found in one
evaluation question >= 50%): 16 of 1,011 rows dropped, all MBPP training tasks that are near copies
of MBPP+ tasks under other ids (51-81% overlap, e.g. mbpp_train-466 ~ Mbpp/758). The per-type grid
(Track 2b) used the same MBPP tasks without this check, so its code results may be slightly
flattered. Every training set now goes through the check.

SFT: 990 rows, LoRA r16, lr 5e-5, 1 epoch.

**Result, 240 test questions x 4 samples vs the raw model: tokens saved +4.4%, accuracy -2.8 pts.**
Per benchmark (saved / points): bfcl_irrelevance +10% / +1.2, bfcl_multiple +7% / -3.1,
bfcl_simple_python +9% / -3.8, ifeval +9% / -5.0, mbpp -3% / -8.1, svamp +3% / +3.7.
Reading: one gentle pass (lr 5e-5, 1 epoch, 990 rows) barely moves the length (targets were
30-61% shorter than the mean reply, the model took about a tenth of that) and still costs MBPP+
8 points (within the +/-4-point noise per benchmark only partly). Shortest-of-8 SFT alone, at this
strength, is not the lever; it stays a candidate start for DPO only if nothing better appears.

### Track 2d: steering for brevity (`steer_dirs.py`, `track.py steer`)

Directions in the residual stream read from the raw model's best-of-8 replies (200 questions per
kind, first 64 thinking tokens): A mean difference shortest - longest correct; B within-question
trend against log length, minus the "gives up" direction (wrong - correct); C one learned vector
(DPO, everything else frozen). Per-kind directions and a cross-kind transfer table. Stored as a
v_proj bias (Qwen3 `attention_bias`), checked identical in vLLM and Transformers. Small push tests
on the 240-question sample (2 samples), best at 4 samples, then fresh replies on 25 held-out
questions per source x 4 vs the raw model.

**Directions (2,826 replies read, 351 fit pairs; held-out pairwise accuracy, 0.50 = chance, about
+/-0.1 at 30-45 pairs):** the pooled direction predicts which of two correct replies is shorter
from the first 64 thinking tokens alone: best at layer 9 (A 0.70, B 0.73), the band 7-13 is
0.66-0.70, early layers ~0.60, late layers ~0.62. By kind (pooled A, layer 9): tool 0.80, text 0.78,
choice 0.73, math 0.61, code 0.61: math and code decide their length less visibly at the start.
Per-kind directions are nearly unrelated to each other (cosines -0.10 to 0.36; math is closest to
tool, 0.36) and, fitted on ~55-80 pairs each, are noisier than the pooled one: on most kinds the
pooled direction scores as well as or better than the kind's own. Candidate layers: A 9, 12; B 9, 10.

**Bigger reading (every finished reply of 1,500 questions; 537 fit pairs, 250 held-out pairs):**
the first reading was partly lucky. Shared direction: 0.64-0.68 over layers 8-17 (no sharp peak;
best A layer 14 at 0.66, B layer 8 at 0.68). By kind (shared): text 0.73-0.79, tool 0.68-0.73,
choice 0.63-0.68, math ~0.60-0.65, code ~0.60-0.65. Each kind's own direction, with 1.5x more
data, still does not beat the shared one except where it reads surface words at the first layers
(code B layer 0: 0.70; tool A layer 2: 0.77). Kinds' directions stay nearly unrelated (cosines
-0.18 to 0.39; text~choice 0.35-0.39, math~code 0.37, math~tool 0.31-0.34). Reading: the start
of the thinking gives a real but moderate hint of its length (~65% vs 50%), clearest for
instructions and tools, weakest for math and code; there is no separate strong dial per kind.

**DPO pairs from the best-of-8 replies (`bon_pairs.py`):** 1,267 pairs (704 shorten, 563 guard);
shorten pairs chosen vs rejected mean tokens: gsm8k 493 vs 2,033, mbpp 747 vs 2,224, toolace
182 vs 514, ifeval 261 vs 493, csqa 276 vs 723, arc 355 vs 874. DPO from adapters/bon-r1:
1,125 pairs, 2 epochs, 141 steps, checkpoints every 25.

**Best-of-8 SFT then DPO on the same replies (240 test questions x 4 samples vs the raw model):**

| checkpoint | tokens saved | accuracy | svamp | mbpp | ifeval | bfcl simple | bfcl multiple | bfcl irrelevance |
|---|---|---|---|---|---|---|---|---|
| step-25 | **+47.6%** | **-2.6** | +62% / +0.6 | +43% / -9.4 | +48% / -1.2 | +29% / +0.6 | +34% / +0.6 | +39% / -2.5 |
| step-50 | +40.3% | -2.8 | +47% / -2.5 | +42% / -5.0 | +40% / -3.7 | +27% / +0.6 | +32% / -0.6 | +34% / +0.0 |

The contrast is the lever: the same replies that gave +4.4% as SFT give +40-48% as DPO pairs, within
the limit, after 25 steps. MBPP+ carries the loss (-5 to -9 on 40 tasks, about +/-8 noise there).

Later checkpoints of the same run (4 samples): step-75 +37.3% / -2.1 (mbpp -10.0, svamp +5.0);
step-100 +36.5% / -2.6 (mbpp -9.4, ifeval -6.2); step-125 +31.1% / -3.2 (outside). Saving falls
and IFEval worsens with more steps (the second pass memorises pairs, 99% ordered right): the
first 25-50 steps are the useful part. MBPP+ loses 5-10 points at every checkpoint.

**Steering push tests (240 test questions x 2 samples; check: the push's effect per token is the
same in vLLM and Transformers, ratio 1.08, correlation 0.99):**

| push | tokens saved | accuracy |
|---|---|---|
| A layer 9, x5 | +17.6% | -0.1 |
| A layer 9, x15 | +22.3% | -5.5 |
| A layer 12, x5 | +2.5% | -2.7 |
| A layer 12, x15 | +28.6% | -7.5 |
| B layer 9, x5 | +10.5% | -0.5 |
| B layer 9, x15 | +23.5% | -2.9 (4 samples: +15.7% / -5.0) |
| C layer 9, x8 | +3.3% | +3.2 |
| C layer 9, x24 | +17.6% | -8.2 |

Steering alone gives up to ~18% at no accuracy cost (A layer 9 x5, not confirmed at 4 samples);
stronger pushes break accuracy. Fresh replies on 141 held-out training questions x 4: mean tokens
raw 1,156 -> A 981 / C 970 at equal accuracy (56-57%), B 964 at -4.9 points; but the shortest
correct of 4 barely moves (raw 450, A 459, B 407, C 427): steering shortens the average reply,
it does not make much shorter best-of-N training data. Math under C: accuracy 36% -> 47% and
mean 2,375 -> 1,745 tokens (25 questions, noisy).

**Stronger SFT on clear-gap questions (bon r2: shortest correct >= 25% under the question's mean,
lr 1e-4, 3 epochs), 4 samples: +12.5% / -0.7** (svamp +16% / +0.6, mbpp +7% / -2.5, ifeval +11% /
-2.5, bfcl +10-22%, -1.9 to +4.4). Three times the saving of r1 at a quarter of its cost.

**Runtime lost:** Colab recycled runtime A at about 11:07 Paris (2026-10-21); the disk, every
adapter (bon-dpo checkpoints, bon-r2) and the 14,400 best-of-8 rollouts are gone. The worker had
also stopped receiving branch updates after 5d3f8ce, so the MBPP guard-only DPO variant never ran.

### Track 2e: rebuild after the lost runtime (half the best-of-8 data, Drive persistence)

750 of the 1,500 questions x 8 (a mixed half), 466 SFT rows after 8 contamination drops, 591 DPO
pairs. Kept vs same-question mean: math 59%, code 48%, CSQA 38%, ARC 34%, IFEval 34%, ToolACE 32%.
Original recipe (gentle SFT then DPO), pushed to 134 steps (4 passes over 533 pairs; one pass is
~33 steps), every checkpoint scored in one engine (`sweep_eval.py`), 240 questions x 2 samples
(about +/-4 points of noise):

| step | tokens saved | accuracy | svamp | mbpp | ifeval |
|---|---|---|---|---|---|
| 10 | +36.2% | -1.5 | +38% / -2.5 | +29% / -7.5 | +49% / +2.5 |
| 20 | +47.4% | -6.4 | +55% / -7.5 | +37% / -16.2 | +61% / -3.7 |
| 30 | +42.5% | -2.9 | +52% / -7.5 | +25% / -1.2 | +56% / -2.5 |
| 50 | +32.6% | -2.2 | +31% / 0.0 | +11% / -7.5 | +52% / -2.5 |
| 70 | +37.6% | -1.0 | +46% / -1.2 | +21% / -1.2 | +51% / -3.7 |
| 100 | +27.9% | -1.2 | +31% / -3.7 | +11% / -3.7 | +40% / 0.0 |
| 130 | +22.4% | -0.1 | +24% / -1.2 | +0% / -1.2 | +37% / 0.0 |

It does not break: past the first pass (~33 steps) the saving drains away (47% -> 22%) and the
accuracy returns to the raw model's. Peak around steps 20-30, as last night (step 25 best). With
the pairs memorised (100% ordered right from step ~60), more passes undo the brevity rather than
deepen it. Best within the limit at 2 samples: step 30, re-scored with 4 samples next.

**Step 30, 4 samples: +44.9% / -2.7** (svamp +53% / -4.4, mbpp +41% / -8.7, ifeval +50% / +3.1,
bfcl +30-37% / -1.9 to 0.0). Within the limit; reproduces last night's step 25 (+47.6% / -2.6) on half
the data. MBPP+ and SVAMP carry the loss.

**DPO straight from the raw model (fresh LoRA, no SFT), same pairs, 2 passes, 2 samples:**
step 10 +11.7% / -1.1; step 20 +33.1% / -2.6; **step 30 +42.0% / -2.3**; step 40 +37.9% / -4.7;
step 60 +37.3% / -5.0. Slower start than from the SFT adapter (step 10: 12% vs 36%), same level at
the end of the first pass (step 30: 42.0 / -2.3 vs 42.5 / -2.9); the second pass costs accuracy here.
The gentle SFT step mostly buys a head start, not a better end point.
No-SFT step 30 with 4 samples: **+36.6% / +0.8** (svamp +39% / +0.6, mbpp +35% / -5.0, ifeval +43% /
+6.9, bfcl +26-30% / -0.6 to +1.9). Less saving than the SFT start's step 30 (+44.9% / -2.7) but no
accuracy cost at all: the first model of the project that is shorter by a third and not less accurate.

**Code fix (MBPP gets guard pairs only, median-length correct reply; from the SFT start), 2 samples:**
step 10 +37.4% / -3.3 (mbpp +24% / -8.7); step 20 +44.2% / -3.3 (mbpp -6.2); step 30 +40.4% / -5.2
(mbpp -10.0); step 40 +35.9% / -5.0 (mbpp -15.0). No checkpoint within the limit. Code still gets
22-26% shorter by spill-over and loses as much as before: removing code's shorten pairs does not
protect code (as the per-type session found with uncut code examples).

**No code pairs at all (from the SFT start), 2 samples:** step 10 +35.6% / +1.1 (mbpp +25% / +1.3);
step 20 +50.6% / -4.4 (mbpp +36% / -8.7); step 30 +44.1% / -2.8 (mbpp +41% / -6.2); step 40 +35.3% /
-1.0 (mbpp +27% / -1.2). Code still gets 25-41% shorter by spill-over alone. Against the original at
the same steps, code loses a little less at steps 10 and 20 and more at step 30: within the noise of
40 tasks x 2 samples (about +/-10 on one benchmark). Code data is not the main cause of the code
loss; the shared "think less" habit is.

**No-code step 30 with 4 samples: +42.4% / -0.7** (svamp +53% / +1.2, mbpp +38% / -3.7, ifeval +44% /
-1.2, bfcl +29-35% / 0.0 to +2.5). Best model of the project so far: level with Lightning McQwen3's
+43.2% / -0.85 (his: full evaluation x 8; ours: 240 questions x 4, to be confirmed with 8). Round 2
starts from it (750 unused questions x 8).

### Round 2 (on-policy): no-code step 30 answers the 750 unused questions x 8

The round-1 model's replies are about half as long (math mean 661 tokens vs 2,221 for the raw
model; capped 1% vs 26%), so sampling took ~40 min. Its shortest correct vs its own mean: math 33%,
code 39%, CSQA 24%, instructions 19%, ARC 18%, tools 16% shorter. 7 contamination drops, 506 pairs
(424 without code). One DPO pass, 2 samples:

| run | step | tokens saved | accuracy | svamp | mbpp | ifeval | bfcl |
|---|---|---|---|---|---|---|---|
| r2, code pairs in | 10 | +55.4% | -4.4 | +68% / -5.0 | +41% / -6.2 | +66% / -7.5 | +37-44% / 0..+3.8 |
| r2, code pairs in | 20 | +63.7% | -4.0 | +76% / -6.2 | +68% / -8.7 | +60% / -2.5 | +42-47% / -2.5..+7.5 |
| r2, no code pairs | 10 | +57.6% | -4.5 | +67% / -7.5 | +50% / -11.2 | +68% / -1.2 | +37-43% / -2.5..+6.2 |

Round 2 deepens the saving (42% -> 55-64%) at about -4 points, mostly math and code. Bug found:
dpo.py never stepped on the last partial batch, so a pass of 317 pairs ended at step 19 without a
final checkpoint (fixed). Queued: 4-sample checks of r2nc step 10 and r2 step 20; a gentler round 2
(guard pairs 60%, checkpoints every 5 steps).

**4-sample checks of round 2:**
- **r2 without code pairs, step 10: +55.5% / -1.7** (within the limit): svamp +64% / 0.0, ifeval +61% /
  +1.9, bfcl +36-42% / -1.3..+5.0, mbpp +54% / **-10.6**. New best: 13 points more saving than
  round 1 for 1 point of accuracy; code carries the whole loss.
- r2 with code pairs, step 20: +62.6% / -3.6 (outside): svamp +75% / -5.0, mbpp +65% / -14.4.

**Gentle round 2 (guard pairs 60%, no code pairs), 2 samples:** step 5 +52.1% / -5.0; step 10 +55.3% /
-0.9; step 15 +60.6% / -6.1 (swings beyond what 5 steps can do: 2-sample noise). **Step 10 with 4
samples: +54.3% / -1.1** (svamp +63% / 0.0, ifeval +61% / +2.5, bfcl +36-44% / -0.6..+6.9, mbpp +50% /
-9.4). About the same as r2 without code (+55.5% / -1.7), slightly safer. Queued: 8-sample recheck of
it; round 3 on a fresh question draw (seed 1) from it.

**8-sample recheck of the gentle round-2 model (step 10), 240 questions x 8: +54.2% / -1.6** (within the
limit; 4 samples said +54.3% / -1.1: stable). svamp +64% / 0.0 (23 lost, 23 gained), ifeval +50% / -0.6,
bfcl +34-42% / -0.6..+5.3, mbpp +60% / -7.5. Mean tokens: svamp 291, mbpp 876, ifeval 337, bfcl 119-128.
Against Lightning McQwen3 (+43.2% / -0.85, full eval x 8): 11 points more saving for 0.75 point more loss.
Round 3 started: 1,302 of 1,500 fresh questions (seed 1) not used before; 750 sampled.

### Round 3 (fresh questions, seed 1): the gentle round-2 model answers 750 new questions x 8

Its replies: math mean 528 tokens (0% capped), code 995, instructions 219, tools 143. Shortest correct vs
its own mean: math 36%, code 34%, ARC 23%, CSQA 22%, instructions 21%, tools 14%. 3 contamination drops,
420 pairs (no code), guard-heavy mix (0.4 / 0.6), one pass, 2 samples:

| step | tokens saved | accuracy | svamp | mbpp | ifeval |
|---|---|---|---|---|---|
| 5 | +59.8% | -0.1 | +72% / -1.2 | +47% / 0.0 | +71% / -2.5 |
| 10 | +61.6% | -3.4 | +77% / -10.0 | +52% / -6.2 | +65% / 0.0 |
| 15 | +66.6% | -6.4 | +80% / -7.5 | +57% / -16.2 | +74% / -5.0 |
| 20 | +69.4% | -4.2 | +82% / -7.5 | +73% / -8.7 | +67% / -3.7 |

**Step 5 with 4 samples: +56.8% / -2.6** (svamp +69% / -0.6, ifeval +59% / +1.3, bfcl +38-45% / -1.3..+3.1,
mbpp +53% / **-12.5**). Rounds compound on saving (42 -> 54 -> 57-69%) while code erodes round after
round (-3.7 -> -7.5 -> -12.5 at 4-8 samples); the other benchmarks hold. Code is the bottleneck.
Queued: diagnosis of the code losses; a round with code "stopped too early" pairs (`code_pairs.py`).

**Code diagnosis (gentle round-2 model, 240 x 8 vs the raw model):** MBPP+ lost 40 answers and gained
16; every loss is a wrong program (none cut by the cap, none unreadable). Thinking on the lost rows:
1,420 -> 530 tokens. Typical loss: `next_smallest_palindrome` returns n when n is already a palindrome;
the thinking "checks" that very case and accepts it ("what if n is already a palindrome? ... return
it. That's correct."). Shorter thinking keeps a check but makes it shallow: the stricter MBPP+ tests
catch the edge cases.

**Code "stopped too early" pairs (`code_pairs.py`, round-3 answers of the gentle round-2 model):** 32
code questions with a correct reply, 2 replies each, cut at 25/50/75% of the thinking, 4 re-answers:
wrong after the cut 24/60 (25%), 15/60 (50%), 10/60 (75%). 30 hard pairs, added to round 3's 420
(199 shorten, 221 guard): code is 7% of the pairs.

**Code round (round-3 pairs + 30 code too-early pairs, from the gentle round-2 model), 2 samples:**
step 5 +56.0% / -2.0; step 10 +57.5% / -4.0; step 15 +60.1% / -6.4; step 20 +64.5% / -6.4.
**Step 5 with 4 samples: +53.2% / -0.1** (svamp +67% / +3.1, ifeval +53% / +3.7, bfcl +37-44% /
-0.6..+5.6, mbpp +49% / -10.0). Against round 3 step 5 without code pairs (+56.8% / -2.6, mbpp -12.5):
3.6 points less saving, 2.5 points more accuracy; code itself barely moves (-10.0 vs -12.5, within
noise). Queued: 8-sample recheck; a bigger code dose (code-r3b: ~600 code questions answered 4x).

**8-sample recheck of the code round, step 5: +53.6% / -2.9** (4 samples said +53.2% / -0.1). svamp +66% /
-1.9, ifeval +51% / -3.4, mbpp +54% / -8.1, bfcl +35-43% / -0.3..+5.0. The 4-sample -0.1 was luck plus
picking the sweep's best on the same questions. Best confirmed with 8 samples stays the gentle round 2
(step 10): +54.2% / -1.6. Rule from now on: compare finalists at 8 samples only.

**Bigger code dose (code-r3b):** all code questions of both draws answered 4x by the gentle round-2 model
(300 + 133; 18 contamination drops): 167 code too-early pairs (28 + 102 + 37) with round 3's 420.
2 samples: step 5 +55.2% / -3.6; step 10 +55.5% / -3.4; step 15 +60.3% / -2.2; step 20 +65.3% / -4.7.
Step 15: 4 samples +60.4% / -2.7 (mbpp -13.7); **8 samples +60.9% / -3.0** (svamp +76% / -6.2, mbpp +61%
/ -7.8, ifeval +57% / +0.6, bfcl +36-44% / -0.6..+5.3): at the limit. Code pairs, even 167 of them, do
not move code (-7.8 vs -7.5 without); this time math pays. Frontier at 8 samples:
+54.2/-1.6 (r2g step 10), +53.6/-2.9 (code-r3 step 5), +60.9/-3.0 (code-r3b step 15).

**8-sample checks (continued):** round 3 step 5 (no code pairs): **+56.9% / -1.9** (svamp +69% / -3.1,
mbpp +57% / -7.5, ifeval +55% / +0.6, bfcl +36-44% / -0.9..+6.2). Weight-space midpoint of r2g step 10
and code-r3b step 15 (`merge.py`, 0.5 + 0.5): +56.7% / -2.4: it lands between its two ends, unlike the
night-1 merges of unrelated adapters.

| model (240 q x 8) | tokens saved | accuracy |
|---|---|---|
| r2g step 10 (round 2, gentle) | +54.2% | -1.6 |
| **bon-dpo-r3 step 5 (round 3)** | **+56.9%** | **-1.9** |
| soup r2g + code-r3b | +56.7% | -2.4 |
| code-r3 step 5 | +53.6% | -2.9 |
| code-r3b step 15 | +60.9% | -3.0 |
| night-1 best (MVT on MC + math whole x2) | +37.1% | -0.6 |

### Fresh test set (to remove the selection bias)

All choices so far (which checkpoint, which variant) were made on `questions_40`, the same 240 questions
the results are reported on. `questions_fresh20261022`: 40 questions per benchmark part drawn at random
with seed **20261022** (`subset_questions.py --per-dataset 40 --seed 20261022 --exclude questions_40`),
none of them in `questions_40`, rebuilt identically from the seed. Finalists are scored there with 8
samples against the raw model on the same set (`track.py fresh`), runs `runs/mvtf/f8-*`.

**Round 4 (seed-2 draw, from round 3 step 5):** 1,202 of 1,500 fresh; replies math 432 tokens, code 1,207,
instructions 230, tools 142. 424 pairs. 2 samples: step 5 +59.9 / -4.0; step 10 +62.1 / -4.7; step 15
+66.1 / -2.4. Step 15: 4 samples +66.6% / -1.1; **8 samples +65.4% / -2.4** (svamp +80% / -2.8, mbpp +68%
/ -10.0, ifeval +58% / -0.3, bfcl +41-48% / -0.3..+9.4).

**The mvt-pertype session's held-out test of our models (questions 41-80 of every part, 8 samples; read
from its Drive folder with `track.py peek`):** round 1 no-code step 30 **+43.6% / -0.9**; r2g step 10
+53.1% / -2.7; r2nc step 10 +53.4% / -2.8. Savings hold on unseen questions; accuracy is about 1 point
worse than on questions_40, the selection effect. Per benchmark (r2g): svamp -7.5, ifeval -4.1, mbpp
-1.6: single-benchmark cells swing +/-10 at 40 questions.

### MBPP-free chains and a clean code source (queued)

Every model of rounds 1-4 descends from the gentle SFT step, whose 466 rows included ~74 MBPP tasks
(MBPP minus MBPP+, near-copies dropped); "DPO from raw" used MBPP pairs. MBPP+ is built from MBPP, so to
leave no doubt, two MBPP-free families of two rounds each (`track.py clean`): A, no code data at all;
B, plus the MBPP-like slice of KodCode-Light-RL-10K as right-beats-wrong pairs only.

KodCode against MBPP+ (CPU, Qwen3 tokenizer; medians): MBPP+ task 17 tokens, reference solution 29,
3 base asserts (plus EvalPlus's extra inputs); MBPP minus MBPP+ 16 / 48 / 3; all usable KodCode (6,171)
99 / 114 / 7-8 asserts, harder topics (graphs, matrices, LeetCode); **the MBPP-like slice (task <= 45,
solution <= 90 tokens): 907 tasks, 28 / 63 / 6-7**, topics close to MBPP+ (lists 53 vs 45%, numbers 59 vs
46%, strings 29 vs 19%). KodCode reports its own decontamination against standard benchmarks; our 8-gram
check against all 2,759 test questions runs on top.

**Runtime 2 (training randomness and fresh set):**
- Round 2 retrained with DPO seed 1 (same pairs): 2 samples +49.7% / -3.6 (original step 10 at 2 samples:
  +55.3% / -0.9).
- Round 4 retrained with DPO seed 1: 2 samples +68.0% / -7.5 (original step 15: +66.1% / -2.4).
- Fresh set (seed 20261022, 240 x 8): **round 4 step 15: +63.2% / -3.9 (outside)**; round 2 retrain:
  +50.9% / -3.0 (at the limit).
Training randomness at this size is large (several points of accuracy between seeds), and round 4 is past
the limit on unseen questions: the late rounds push too far.

### Where the accuracy goes at 60%+ (fresh questions), and the balanced round (queued)

On the fresh set (240 x 8) the loss is math, not code: SVAMP -9.1 (round 4), -10.6 (round 4 retrain),
-9.7 (round 2 retrain), -7.5 (round 2 on questions 41-80), while MBPP+ is -2.5 to +0.6. On questions_40
math looked fine and code bad: picking checkpoints on those 40 SVAMP questions selected for them. Math
is shortened most (SVAMP 76% at round 4) and SVAMP is built with traps (irrelevant numbers, rephrased
questions). Late rounds mostly push math: round 4 had 96 math shorten pairs against 10 for tools and 15
for instructions (their answers were already short), and a numeric answer can be right by a shortcut.

Balanced round (`track.py balanced`, from round 2): savings from tools, instructions and multiple choice
(short-beats-long only where the question is answered right >= 6/8), math protected (at most 40 shorten
pairs per math source, right-beats-wrong with a median-length correct reply), math data with SVAMP-style
traps: GSM-Plus (Li et al. 2024; distraction insertion, problem understanding, reversed/added operations,
numerical substitution; GSM8K-based) and GSM-IC (Shi et al. 2023; in-topic irrelevant sentences). 1,500
fresh questions x 8; DPO with 3 seeds averaged per checkpoint step (merge.py); best soup x8 usual + fresh.

**Family A, A1 (DPO from raw, round-1 answers without code, 438 pairs, 28 steps), 2 samples:** step 10
+15.8% / 0.0; step 20 +41.9% / -3.6. Step 10 with 4 samples: +17.8% / +1.7. Steps 20 and 28 rechecked
with 4 samples before A2 picks its start.
A1 step 10 with 8 samples: +17.7% / +1.1. **A1 step 20 with 4 samples: +37.8% / +0.1** (the 2-sample -3.6 was
noise): svamp +41% / +2.5, ifeval +52% / +4.4, mbpp +27% / -7.5, bfcl +26-29% / -0.6..+1.9.
A1 step 28 with 4 samples: +40.5% / -1.2 (mbpp -8.7); A2 starts from it.

**Fresh set (seed 20261022, 240 x 8), runtime 1:** round 1 no-code step 30 **+40.4% / -0.9**; round 2 (r2g step
10) **+51.4% / -1.8**; round 3 step 5 +56.0% / -3.6 (outside; svamp -7.5). Round 2 holds on both fresh sets
(-1.8 here, -2.7 on questions 41-80); round 3 does not.

**Round 5 (seed-3 draw, from round 4 step 15):** 2 samples: step 5 +69.3 / -2.2, step 10 +70.9 / -3.3, step 15
+73.5 / -4.3. Step 5: 4 samples +71.1 / -1.1; **8 samples +68.6% / -2.4** (svamp +82% / -2.2, mbpp +71% / -10.3,
ifeval +64% / +1.3, bfcl +43-49%). Not tested on fresh questions; round 4 lost 1.5 points there.

**Family A, A2 (from A1 step 28; 750 unused questions without code, 345 pairs), 2 samples:** step 5
+45.9 / -3.1; step 10 +47.0 / -5.9; step 15 +53.3 / -4.6 (none within; svamp -3.7 to -11.2). **A1 step 28
on the fresh set: +35.4% / -1.9.** A2's step 5 is rechecked with 4 samples after the balanced round.

**Balanced round data (round 2 answering 1,437 fresh questions x 8):** shortest correct vs its own mean:
gsm_plus 41%, gsm_ic 32%, ARC 27%, instructions 24%, CSQA 19%, tools 14%; 0 contamination drops. 736 pairs:
shorten 218 (CSQA 59, gsm_plus 40 and gsm_ic 40 capped, instructions 35, ARC 24, tools 20), guard 518
(math guards use a median-length correct reply). Mix 30 / 70. Three DPO seeds of 43 steps each; at step 10-15
pairs are ordered right 46-64% only (harder pairs than before). Tools and instructions give few shorten
pairs even at 6/8: round 2's answers there are already near their shortest.
**Balanced soups (3 seeds averaged), questions_40 x 4:** step 10 +55.9% / -2.0; **step 20 +58.9% / -0.5**
(svamp +67% / +1.9, ifeval +65% / +4.4, bfcl +38-47% / -0.6..+3.1, mbpp +57% / -10.0); step 30 +57.5% / -0.5.

**Balanced round, step-20 soup (3 seeds averaged):** step 40 at 4 samples +55.6% / -0.9.
- questions_40 x 8: **+58.8% / -2.1** (svamp +68% / -1.6, mbpp +60% / -8.4, ifeval +61% / -0.3, bfcl +36-44%)
- **fresh set x 8: +53.8% / -2.5** (svamp +64% / -4.7, mbpp +58% / -1.6, ifeval +50% / -2.8, bfcl +36-41%)
Against round 2 on the same fresh set (+51.4% / -1.8): 2.4 points more saving, 0.7 more loss; math loses
-4.7 on fresh questions (rounds 3-4: -7.5 to -10.6). Protecting math works; the gain is modest.
**A2 step 5 with 4 samples: +44.6% / +0.3** (the 2-sample -3.1 was noise).
Runtime 2's worker predates the `end` line and failed on it; `end` is now also a stage that saves, flushes and
stops its worker (runtime 1 reaches it later).

**Fresh set (240 x 8), last results (read from Drive compare.json):**
- Family B (MBPP-free, KodCode right-beats-wrong): **B1 step 20 +24.8% / -1.5** (svamp +36% / -0.3, mbpp +24% / +0.6,
  ifeval +19% / -4.4, bfcl +17-21%); **B2 step 10 +32.7% / -0.1** (svamp +43% / -1.9, mbpp +31% / +1.9, ifeval
  +31% / +0.6, bfcl +24-25% / -2.5..-0.3).
- **Round 5 step 5: +65.5% / -5.4** (svamp +79% / -10.0, ifeval +55% / -5.0, mbpp +73% / -2.8, bfcl +43-47%):
  outside, as rounds 3-4.

## Track 2f: MBPP-free line, full recipe, toward > 60% saved at better than -2 (queued 22 Oct 2026)

Goal set by the user: > 60% tokens saved with an accuracy change better than -2 points, from a model that never saw
any evaluation data (so only the MBPP-free families A and B as starts). `track.py chain`: each round draws new
questions (ToolACE, IFEval-like, ARC/CSQA, GSM-Plus, GSM-IC; minus every earlier draw), plus new KodCode tasks on line
B; contamination check now at 30% of 8-grams OR any exact 13-word run, boilerplate lines shared by >= 50 evaluation
questions ignored; pairs as the balanced round; DPO with 3 seeds; the soup with the most saving at a 4-sample change
>= -1.5 is chosen and becomes the next round's base model (fresh LoRA on top). Every chosen soup is scored x8 on the
usual and the fresh questions. `track.py audit` writes runs/audit/contamination_report.txt for every training
question ever prepared. Line B (runtime 1): cb1 from clean-b2 step 10, cb2. Line A (runtime 2): ca1 from clean-a2
step 5, ca2. The full 2,759 x 8 evaluation stays for the very end (user's call).

Queued after the two clean lines (22 Oct 2026):
- Runtime 2, `repair` on round 4 (rep4) and round 5 (rep5): blends with round 2 (75/25, 50/50), length-matched
  right-beats-wrong DPO on new trap math (pairs of a right and a wrong reply within 25% in length: correctness
  without a length signal), self-correction SFT on own right replies no longer than the typical one.
- Runtime 1, **v2 from scratch** (`chain --start raw`, rounds v2r1-v2r4): one recipe in order, no MBPP anywhere;
  adds verified MVT cuts (4/4 re-answers) as short-beats-long pairs for tools and instructions; KodCode from a
  wider MBPP-like slice (task <= 70, solution <= 130 tokens; the 907-task slice runs out), tasks never repeated
  within a line (checked by question text).

**Contamination audit (runs/audit/contamination_report.txt), every training question ever prepared vs the 2,759
evaluation questions:** math (GSM8K, GSM-Plus, GSM-IC), ToolACE, ARC, CSQA, Skywork: 0 at >= 30%, 0 exact 13-word
runs (1 in Skywork, unused). **MBPP (old lineage only): ~9 of 300 per draw at >= 50% and ~39 with an exact 13-word
run: MBPP holds MBPP+ problems under other ids** (e.g. change_date_format ~ Mbpp/427, max_product ~ Mbpp/415). The 50%
check dropped the worst; weaker near copies may have reached the gentle SFT of rounds 1-5 and the balanced model,
which therefore cannot be certified clean. IFEval-like: 0-2 of 300 at >= 50%, ~190 share an exact 13-word run with
IFEval: its fixed constraint wording ("answer with one of the following options: My answer is yes..."), not its
questions; the chain rounds' strict check (30% or any 13-word run) drops them all the same. KodCode: 0 at >= 50%, 64
with a 13-word run, dropped likewise. The submission should come from a clean line (A, B, v2).

**Clean line A, round 3 (ca1, from A2; full recipe; 3-seed soups), questions_40 x 4:** step 10 +52.5% / -1.0 (chosen,
floor -1.5); step 20 +54.6% / -3.0; step 30 +49.2% / -3.0; step 40 +54.2% / -2.9 (svamp +60% / -2.5, ifeval +60% /
+1.9, bfcl +39-47%, mbpp +51% / -10.6). Its x8 tests were cut by the lost runtime (a half-written folder that evaluate.py
refuses; track.evaluate now clears it) and are requeued; ca2 started from the step-10 soup.

**Clean line B, round 3 (cb1, from B2; full recipe + KodCode right-beats-wrong; 787 pairs; 3-seed soups), questions_40 x 4:**
step 10 +42.9% / -1.5; step 20 +41.6% / +0.8 (svamp +55% / +0.6, ifeval +49% / +6.9, bfcl +31-35%, mbpp +24% / -5.0);
step 30 +41.9% / +1.0. Accuracy is kept or gained; the saving is ~10 points below line A's round 3 (+52.5% / -1.0).
cb1 step 40 at 4 samples +44.0% / +0.9 (chosen). **x8 usual: +42.0% / -0.6** (svamp +55% / -3.1, mbpp +29% / -0.9,
ifeval +45% / +0.3, bfcl +30-38%); **fresh x8: +41.0% / -1.1** (svamp +54% / -1.6, mbpp +34% / +0.3, ifeval +41% /
-2.5, bfcl +28-33%). cb2 started from it.

**Clean line A, round 4 (ca2, from ca1 step 10; 675 pairs), questions_40 x 4:** step 10 +49.8 / -2.9; step 20 +52.8 /
-1.9 (svamp +60% / 0.0, mbpp +48% / -13.7, ifeval +60% / +5.0); step 30 +54.1 / -3.5; step 40 +56.3 / -2.2. Chosen
step 20 (none reached the -1.5 floor: the most accurate). Round 4 adds little saving over round 3; code carries the loss.
**ca2 step 20 x8: usual +53.5% / -2.4; fresh +46.4% / -2.7** (fresh: svamp +56% / -5.3, mbpp +51% / -1.9, ifeval +39% / -3.1,
bfcl +35-40%). Line A's 4th round is weaker on unseen questions than line B's 3rd in accuracy, and its saving drops
7 points from the usual to the fresh questions.

**ca1 step 10 (line A round 3) x8: usual +49.7% / -2.3; fresh +48.6% / -1.8** (svamp +58% / -4.1, mbpp +53% / -2.8, ifeval
+46% / +0.3, bfcl +33-36%). Line A's best on unseen questions (better than its round 4: +46.4 / -2.7).

**Repair of round 4 (rep4), 1. blend 75% round 4 + 25% round 2:** usual x4 +66.5% / -2.5; **fresh x8 +61.5% / -4.0**
(svamp +74% / -8.8, mbpp +67% / -4.4, ifeval +55% / -1.9, bfcl +39-43%). Blending in a quarter of round 2 recovers
little: math still -8.8 on unseen questions.

**Repair of round 4 (rep4), all candidates (usual x4 / fresh x8):** blend 75% r4 + 25% r2 +66.5 / -2.5 | **+61.5 / -4.0**;
blend 50/50 +57.8 / -2.2 | **+57.3 / -2.4**; length-matched right-beats-wrong DPO step 10 +64.2 / -1.5 | **+61.3 / -4.2**;
RFT (SFT on its own right replies <= median length) +67.5 / -2.1 | **+63.4 / -5.1**. None keeps 60%+ within -2 on unseen
questions: the 60%+ candidates all lose 4-5 points there (math); the 50/50 blend is the best trade-off of the repairs
(+57.3 / -2.4 fresh, ~ the balanced soup's +53.8 / -2.5 with 3.5 points more saving). rep5 started: blend75 usual x4 +67.1 / -2.3.

**Clean line B, round 4 (cb2, from cb1's soup; KodCode right-beats-wrong; 3-seed soups):** step 40 at 4 samples +47.1 / +1.4;
chosen step 30: **usual x8 +51.0 / -0.3; fresh x8 +46.9 / -2.6**. The 4th round adds ~6 points of saving over cb1 (+41.0 / -1.1
fresh) but loses 1.5 points on unseen questions. Line B finished; v2 from scratch (v2r1) started on runtime 1. Runtime 2 was
lost overnight and relaunched (rep5 resumes from Drive).

**Repair of round 5 (rep5), all candidates (usual x4 / fresh x8; fresh per type):**
blend 75% r5 + 25% r2 +67.1 / -2.3 | **+63.5 / -3.9** (svamp +75% / -9.4, mbpp +69% / -4.1, ifeval +59% / -1.2, bfcl +39-44%);
blend 50/50 +62.7 / -3.0 | **+58.8 / -3.4** (svamp +72% / -5.9, mbpp +65% / -3.1, ifeval +50% / -4.1, bfcl +38-42%);
length-matched right-beats-wrong step 10 +65.0 / -1.5 | **+61.9 / -4.5** (svamp -7.8, ifeval -6.2);
RFT +69.4 / -0.2 | **+67.1 / -5.5** (svamp +78% / -10.0, ifeval -5.6, bfcl_multiple -5.3).
Conclusion of both repairs: no technique keeps 60%+ within -2 on unseen questions; the usual-set score is optimistic by
2-5 points for every 60%+ candidate (RFT: -0.2 usual vs -5.5 fresh). Math (SVAMP) is always the main loss. Runtime 2's
queue is done (end).

## Track 2g: calibrated cuts (confidence vs real outcome) + RL (rlcal), 23 Oct 2026

**rc1 cut calibration (ca1 = models/ca1-soup-10 answering a new draw, seed 31: 1,000 questions + 300 KodCode x8; 904
right replies kept, the longest right one per question; each cut at 1/5/10/20/35/50/65/80% of its thinking, "</think>"
closed there, 20 re-answers per cut = 144,640 answers; 2,901 of 7,232 cuts right 20/20). Confidence = per-token
geometric-mean probability of the answer the full reply wrote, teacher-forced right after the cut.**

Confidence bins -> real share of right re-answers (n = cuts):
- arc_easy: 0.6-0.8 -> 0.18 (32); 0.8-0.9 -> 0.57 (16); 0.9-1.0 -> 0.99 (225). ECE 0.09
- commonsense_qa: 0.0-0.2 -> 0.41 (36); 0.2-0.6 -> 0.00-0.01 (169); 0.6-0.8 -> 0.25 (90); 0.8-0.9 -> 0.61 (102); 0.9-1.0 -> 0.99 (483). ECE 0.16
- gsm_ic: 0.0-0.6 -> 0.00 (975); 0.6-0.8 -> 0.05 (149); 0.8-0.9 -> 0.22 (87); 0.9-1.0 -> 0.80 (453). ECE 0.34
- gsm_plus: 0.0-0.6 -> 0.00 (807); 0.6-0.8 -> 0.13 (134); 0.8-0.9 -> 0.32 (27); 0.9-1.0 -> 0.86 (216). ECE 0.34
- ifeval_like: 0.2-0.4 -> 0.84 (132); 0.4-0.6 -> 0.78 (330); 0.6-0.8 -> 0.87 (48); 0.8-0.9 -> 0.84 (21); 0.9-1.0 -> 0.94 (21). ECE 0.31
- kodcode: 0.4-0.6 -> 0.58 (35); 0.6-0.8 -> 0.63 (287); 0.8-0.9 -> 0.64 (496); 0.9-1.0 -> 0.87 (926). ECE 0.13
- toolace: 0.0-0.2 -> 0.17 (69); 0.2-0.4 -> 0.57 (77); 0.4-0.6 -> 0.53 (71); 0.6-0.8 -> 0.61 (81); 0.8-0.9 -> 0.54 (120); 0.9-1.0 -> 0.92 (510). ECE 0.11

Per source (allowance: math 0.5 pt, code 1, others 2): confidence threshold c / saved / real loss | outcome rule
(earliest cut right 20/20) saved / loss:
arc_easy c 0.9: 83% / 1.3 | 80% / 0.0; commonsense_qa c 0.95: 54% / 0.2 | 56% / 0.0; gsm_ic no c | 30% / 0.0;
gsm_plus no c | 26% / 0.0; ifeval_like c 0.7: 7% / 1.7 | 48% / 0.0; kodcode no c | 62% / 0.0; toolace no c | 58% / 0.0.

Reading: the model's confidence in its own answer is informative only at the top and only for multiple choice; for
math it is overconfident (0.98 -> 0.80-0.86 right), for instructions flat (0.3-0.9 all ~0.8 right: the probability of
one wording, not of being right), for code and tools weak. No confidence threshold keeps math, code or tools within
their allowance. The real outcome, however, shows large safe cuts: code 62%, tools 58%, instructions 48%, math 26-30%
of the reply cut with every re-answer still right. First rc1 pairs (confidence only: 142 cut pairs, mostly multiple
choice) were dropped before training finished; rc1 reruns with the stopping rule per type: confidence where a
threshold keeps the allowance (arc, csqa, ifeval-like), else the real outcome (gsm_ic, gsm_plus, kodcode, toolace).
Cut pairs: gsm_ic 110 shorten / 140 hard, gsm_plus 74 / 92, kodcode 169 / 100, toolace 89 / 33, arc 32 / 3, csqa 71 / 26,
ifeval-like 7 / 3; plus guard pairs (bon_pairs, math shorten 0) and 133 KodCode right-beats-wrong.
Process notes: grading 144,640 answers serially took > 1 h (code tests; GPU idle) -> check_vllm grades with 16 threads
(~15 min); a killed check_vllm left an orphan vLLM engine holding 10.5 GB (RL smoke test went out of memory).

**v2 round 1 (v2r1, from the raw model; seed-21 draw, 1,500 questions + 300 KodCode x8; chain recipe + KodCode
right-beats-wrong + verified MVT cuts for tools and instructions; 3-seed soups), questions_40 x4:** step 30 +19.3 / +1.5
(chosen), step 40 +18.7 / +3.1. **Step 30 usual x8: +16.3 / +0.8** (bfcl +13-18% / 0..+0.6, ifeval +20% / +0.6, mbpp +8% /
-0.3, svamp +21% / +2.5); **fresh x8: +14.8 / -0.2** (bfcl irrelevance +16% / -1.9, multiple +14% / -3.1, simple +14% /
-2.8, ifeval +15% / +1.6, mbpp +6% / +1.6, svamp +23% / -1.2). A gentle first round from the raw model (as A1 step 10,
+17.7 / +1.1): almost free in accuracy, little saving. v2r2 started from its soup.
rc1 DPO with the per-type rule: 1,489 training pairs (48 held out), mix shorten 0.45 / guard 0.29 / hard 0.26; seed 0
94 steps, pairs ordered right 71-86%.

**rc1 cal-cut DPO (per-type stopping rule; 1,489 training pairs: shorten 691, guard 451, hard 397; 3 seeds, soups), choosing
set (questions_val20261023) x4:** ca1 (start, reference) +44.1 / -1.8 [mbpp +56% / -6.2, svamp +49% / +0.6, ifeval +30% / 0.0,
bfcl +34-41% / -3.1..-0.6]; soup step 10 +52.5 / -0.1; step 20 +54.0 / -0.8; **step 30 +55.1 / -0.6 (chosen)** [svamp +63% /
-2.5, mbpp +66% / +1.3, ifeval +40% / -1.3, bfcl +40-50% / -1.2..+1.9].
- **rc1-soup-30 usual x8: +58.0 / -3.1** (bfcl irrelevance +48% / -4.4, multiple +45% / +3.1, simple +38% / -0.3; ifeval +53% /
  -0.6; mbpp +61% / -6.9; svamp +69% / -4.4).
- **rc1-soup-30 fresh x8: +55.6 / -2.9** (bfcl irrelevance +46% / -4.4, multiple +42% / -5.0, simple +39% / +0.3; ifeval +45% /
  -1.9; mbpp +64% / -1.6; svamp +66% / -5.0). Against ca1 on the same fresh set (+48.6 / -1.8): +7.0 points of saving for
  -1.1 point; code better (mbpp 53% / -2.8 -> 64% / -1.6), math and tools worse (svamp -4.1 -> -5.0; bfcl ~-0.6 -> -3.0 pooled).
  On the choosing set the same model looked -0.6: x4 noise and picking the best of 3 soups. Clean (no MBPP), and on the
  same saving/accuracy frontier as the grey-zone rep4 blend 50/50 (+57.3 / -2.4) and the balanced round (+53.8 / -2.5).
- RL started from rc1-soup-30: pool of 782 questions solved >= 4/8 by ca1; 32 questions x 8 per step; first steps: accuracy
  on its training answers 80-89%, mean thinking 200-340 tokens.

**rc1 RL (from rc1-soup-30; reward right x (1 - alpha_q sigmoid(z_thinking)), alpha 0.15 x solve rate, batch-normalized
advantages, KL 1e-3, lr 1e-5, LoRA r16, 32 questions x 8 per step, pool = the 782 round questions solved >= 4/8, repeated
~every 24 steps), choosing set x2 quick checks:** block 1 (step 20) +54.2 / -0.2 [svamp +59% / -2.5, mbpp +64% / 0.0,
ifeval +44% / +2.5, bfcl +43-49% / -3.7..+1.2]; block 2 (step 40) +51.3 / +0.9 [5 of 6 parts exactly equal to the base's
accuracy: suspicious, x2]; block 3 (step 60) +48.5 / -0.6 [svamp +49% / -1.2, mbpp +56% / +1.3, ifeval +42% / -2.5, bfcl
+42-50% / -3.7..+3.7]. Mean thinking on its training answers rose from ~200-250 tokens (block 1 start) to 400-500 (block 4)
at 82-96% right: with an already short model and repeated questions, the correctness reward outweighs alpha 0.15 and RL
buys accuracy back with length. Block 2 hit CUDA out of memory on a long reply (vocabulary-wide log-probs per position;
fixed: chunked, recomputed log-probs). RL stopped after block 3 (saving falling, accuracy flat); the DPO soup and blocks 1-3
go to the x4 choice on the choosing set and the winner to x8 usual + fresh. Next RL (rr4, on the round-4 50/50 blend):
fresh contamination-checked questions (seed 41, ~900 per source + unused KodCode, each seen about once), alpha 0.3.

**rc1, every version scored on the fresh set x8 (user's call: no choosing set; so the fresh set now also chooses):**
- DPO soup 30: +55.6 / -2.9 (svamp +66% / -5.0, mbpp +64% / -1.6, ifeval +45% / -1.9, bfcl +39-46% / -5.0..+0.3)
- + RL 20 steps: **+54.8 / -3.7** (svamp +65% / -6.6, mbpp +63% / -1.3, ifeval +45% / -4.4, bfcl irr +45% / -3.7, mult +41% /
  -3.7, simple +38% / 0.0). Choosing set x4 had said +57.3 / -0.2.
- + RL 40 steps: **+53.9 / -2.1** (svamp +65% / -4.1, mbpp +59% / -1.9, ifeval +46% / -0.6, bfcl irr +45% / -1.2, mult +41% /
  -4.1, simple +39% / -0.3). Choosing set x4: +54.8 / -0.9.
- + RL 60 steps: **+49.6 / -3.4** (svamp +58% / -3.7, mbpp +58% / -4.7, ifeval +38% / -2.8, bfcl irr +44% / -2.2, mult +42% /
  -4.4, simple +39% / -0.3).
Reading: RL with alpha 0.15 on repeated questions does not move rc1 off its frontier on unseen questions: the four versions
spread over -2.1 to -3.7 at 50-56% saved, within about two noise widths (+/-1.5 pts each at x8 on 240 questions), and the
choosing set (x4) was 1.2-3.5 points more optimistic than the fresh set for every version. Best on fresh: RL 40 steps,
+53.9 / -2.1 (clean). Still to score: DPO soups 10 and 20.

**MBPP audit of the 74 MBPP rows the gentle best-of-8 SFT trained on (runs/bon/select-r1/sft.jsonl; the only MBPP exposure of
the rounds-1-5 / balanced / repair lineage: every DPO round used --drop-sources mbpp_train), mbpp_audit.py vs the 378 MBPP+
tasks (runs/audit/mbpp74.txt):** 8-gram share >= 30%: 0; exact 13-word run: 2; same function name: 1; wording similarity
(difflib on words) >= 0.8: 18. Read by hand, the closest pairs are sibling tasks built on the same template with a
different operation, not the same problem: add vs multiply nested tuples (13-word run), n cheapest vs n most expensive
items (13-word run), recursive gcd vs number of divisors (same-name flag), sum of even vs sum of odd numbers in a range
(0.88), rhombus vs rectangle area, first even vs first odd number. No training row found to be an MBPP+ task under
another id (the near copies, e.g. change_date_format ~ Mbpp/427, had been dropped by the 50% check before training).
The lineage saw MBPP-style training tasks, some of them siblings of MBPP+ tasks; it did not see MBPP+ tasks.

**v2 round 2 (v2r2, from v2r1's soup; seed 22), soups x4:** step 10 +20.9 / +0.2; 20 +23.6 / -0.6; 30 +27.3 / +0.6; 40 +28.6 /
-0.6 (chosen). **Step 40 usual x8: +31.2 / -0.9; fresh x8: +28.6 / -0.8** (bfcl irrelevance +25% / -2.2, multiple +25% /
-3.7, simple +22% / +0.6; ifeval +33% / +0.6; mbpp +21% / -0.3; svamp +35% / -1.6). v2r3 started.

**rc1 DPO soups 10 and 20 on the fresh set x8:** soup 10 **+48.6 / -3.0** (svamp +62% / -6.2, mbpp +55% / -2.2, ifeval +35% / -1.9,
bfcl irr +38% / -3.4, mult +38% / -2.2, simple +34% / +0.3); soup 20 **+54.8 / -4.0** (svamp +65% / -5.6, mbpp +63% / -4.7, ifeval
+46% / -3.1, bfcl irr +43% / -4.7, mult +40% / -3.7, simple +37% / +0.6). All seven rc1 versions on the fresh set: +48.6..+55.6%
saved at -2.1..-4.0 (soup 10 -3.0, soup 20 -4.0, soup 30 -2.9, RL 20 -3.7, RL 40 -2.1, RL 60 -3.4). Soup 10 saves exactly as much as
ca1 (+48.6) but loses 1.2 points more: the cut pairs' accuracy cost shows from the first checkpoint; the spread between neighbouring
checkpoints (1-2 points) is noise-sized. rr4 (grey zone) started by mistake before the queue change and was stopped at its first
RL block; runtime 2 now runs the calibration re-checks, then rr1 and rc2.

**Calibration re-check, rc1-soup-30 on rc1's own cuts (calcheck rc1s30; the same 7,232 cut points ca1 was measured on, 20
re-answers each, 2,954 pass with 20/20 right):** confidence vs real outcome is unchanged by the cut-pair training. ECE per source,
ca1 -> rc1-soup-30: arc_easy 0.09 -> 0.07, commonsense_qa 0.16 -> 0.17, gsm_ic 0.34 -> 0.36, gsm_plus 0.34 -> 0.37, ifeval_like
0.31 -> 0.31, kodcode 0.13 -> 0.13, toolace 0.11 -> 0.14. Same shape: multiple choice is calibrated at >= 0.9 (0.99 right);
trap math is over-confident (conf 0.98 -> 0.78 right on gsm_ic); instructions are under-confident (conf 0.3-0.5 -> ~0.8 right);
code and tools middling. The selection rule picks the same thing for both models (confidence for multiple choice and IFEval, the
outcome rule for math, code and tools). Reading: training on "stop here" pairs taught the model to stop earlier, not to know
better when it may stop; its sense of "I already have the answer" did not move. Next: the same check on rc1-rl-40 (running).
