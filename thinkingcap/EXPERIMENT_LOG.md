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

# Track 3: MVT per type, matched data (branch `mvt-pertype`)

Own Colab L4 runtime from 2026-10-20 22:08 (the RL session's runtime was stopped, with the user's
approval, to free the slot). `TRACK_SET=matched`, queue `plan_mvt.txt`.

Data fix before the run: about half of ToolACE's prose first replies ask for a missing argument of
a function that fits; BFCL's irrelevance means no function fits. `prepare_types.py` now keeps
only replies saying no function can do it (checked on CPU: 5,025 simple, 3,615 multiple, 276
irrelevance usable). The mvt-fading session's `bon` data predates this filter.

## Reference: raw Qwen3-0.6B on this runtime (240 questions, 2 samples)

| benchmark | accuracy | mean tokens |
|---|---|---|
| BFCL irrelevance | 82.5% | 204 |
| BFCL multiple | 76.3% | 223 |
| BFCL simple | 82.5% | 190 |
| IFEval | 53.8% | 734 |
| MBPP+ | 57.5% | 1,699 |
| SVAMP | 81.3% | 940 |

Same as the earlier runtimes (same seed). Capped: IFEval 3.8%, MBPP+ 7.5%, SVAMP 3.8%.

4-sample reference (`s4-base`, what every grid cell is scored against): BFCL irrelevance 85.6% /
213, multiple 79.4% / 217, simple 83.1% / 185; IFEval 51.9% / 705; MBPP+ 57.5% / 2,004; SVAMP
81.9% / 830 (accuracy / mean tokens).

## Training data (`grid_data --per-source 300`)

`prepare_types.py`: gsm8k_all 8,792 usable, mbpp_train 596, toolace 300 (144 simple, 72 multiple,
84 irrelevance, from 5,025 / 3,615 / 276), ifeval_like 56,339, multiple choice 11,592; 300 kept per
source (multiple choice: 300 shared by ARC-Easy and CommonsenseQA). 3,000 replies sampled.

The model's own replies (2 per question, caps 4,096 / 8,192 for code; ~80 min on the L4):

| source | replies | correct | capped | mean tokens |
|---|---|---|---|---|
| arc_easy | 120 | 88% | 0% | 492 |
| commonsense_qa | 480 | 51% | 0% | 506 |
| gsm8k_all | 600 | 43% | 24% | 2,167 |
| ifeval_like | 600 | 78% | 1% | 408 |
| mbpp_train | 600 | 38% | 12% | 2,842 |
| toolace | 600 | 58% | 0% | 265 |

GSM8K stays much harder than SVAMP (43% vs 82%); only its correct replies of at most 700 tokens
are used (`--easy-tokens 700`). ToolACE replies are short (265 tokens), like BFCL's (185-217).

## Grid (each kind trained alone, scored on all four benchmarks, 4 samples vs `s4-base`)

| run | traces (kept / cut) | target tokens | tokens saved | accuracy | per benchmark (saved / pts) |
|---|---|---|---|---|---|
| tool-whole | 189 / 0 | 201 -> 183 | +5.2% | +0.3 | BFCL irr +6% / 0.0, mult +5% / +0.6, simple +6% / -3.1; IFEval -4% / -0.6; MBPP+ +6% / +0.6; SVAMP +12% / +1.9 |
| tool-mvt1024 | 189 / 178 | 201 -> 142 | +17.6% | +1.7 | BFCL irr +38% / -1.2, mult +55% / -3.1, simple +42% / +1.9; IFEval +6% / +9.4; MBPP+ +2% / -3.1; SVAMP +8% / +1.3 |
| tool-mvt256 | 189 / 173 | 201 -> 121 | +25.8% | +0.6 | BFCL irr +45% / +3.1, mult +55% / -1.9, simple +36% / +0.6; IFEval +11% / +6.9; MBPP+ +30% / -4.4; SVAMP +10% / -0.6 |
| tool-mvt1024-tokens32 | 189 / 176 | 201 -> 106 | +18.1% | -2.7 | BFCL irr +54% / **-8.7**, mult +24% / 0.0, simple +39% / -1.3; IFEval +3% / +1.2; MBPP+ 0% / -6.9; SVAMP +24% / -1.9 |
| tool-mvt256-tokens32 | 189 / 167 | 201 -> 89 | +11.5% | **-4.5** | BFCL irr +54% / **-10.0**, mult +24% / -1.9, simple +47% / -0.6; IFEval -6% / +3.7; MBPP+ -5% / -6.9; SVAMP +7% / **-10.6** |
| text-whole | 259 / 0 | 347 -> 312 | +6.1% | -0.5 | BFCL +5/+6/+5% / -1.2/+0.6/0.0; IFEval +6% / +2.5; MBPP+ -2% / -1.9; SVAMP +14% / -2.5 |
| text-mvt1024 | 259 / 238 | 347 -> 234 | +15.1% | -0.9 | BFCL +11/+9/+11% / -5.6/+3.1/-1.3; IFEval +24% / +5.6; MBPP+ +8% / -6.2; SVAMP +17% / -1.9 |
| text-mvt256 | 259 / 227 | 347 -> 190 | +18.2% | -2.1 | BFCL +23/+12/+13% / -3.7/0.0/-1.3; IFEval +19% / +5.0; MBPP+ +29% / **-7.5**; SVAMP +7% / -4.4 |
| text-mvt1024-tokens32 | 259 / 234 | 347 -> 218 | +17.1% | -1.7 | BFCL +10/+13/+10% / -0.6/+3.1/-2.5; IFEval +25% / +6.2; MBPP+ +5% / -8.1; SVAMP +26% / -5.0 |
| text-mvt256-tokens32 | 259 / 219 | 347 -> 180 | +11.8% | -2.1 | BFCL +3/+4/+4% / -6.2/0.0/-1.9; IFEval +23% / +4.4; MBPP+ -1% / -6.2; SVAMP +20% / -3.7 |
| choice-whole | 203 / 0 | ~455 -> ~400 | +2.5% | +0.9 | BFCL +5/+9/0% / 0.0/-0.6/+1.9; IFEval -6% / +4.4; MBPP+ -8% / -5.0; SVAMP +17% / +3.7 |
| choice-mvt1024 | 203 / 201 | ~455 -> ~97 | +72.9% | **-10.7** | BFCL +39/+53/+32% / -1.2/**-13.1**/-3.8; IFEval +60% / +12.5; MBPP+ +93% / **-10.0**; SVAMP +66% / **-39.4** |
| choice-mvt256 | 203 / 194 | ~455 -> ~102 | +69.0% | **-13.3** | BFCL +37/+51/+28% / -3.7/**-11.2**/-0.6; IFEval +45% / +4.4; MBPP+ +93% / **-13.7**; SVAMP +61% / **-38.7** |
| choice-mvt1024-tokens32 | 203 / 199 | ~455 -> ~76 | +8.8% | **-4.7** | BFCL +18/+12/+15% / -1.2/**-10.6**/-5.6; IFEval +3% / +1.9; MBPP+ +9% / -1.9; SVAMP +8% / **-13.1** |

Matched tool data trains for free (old Hermes data: -3.3, IFEval -7.5, MBPP+ -9.4).
tool-mvt1024: 356 cut traces checked, 332 pass; 178 of 189 cut, targets 201 -> 142 (-30%).

tool-mvt1024 cuts BFCL by 38-55% at no cost (all within the +/-4-point noise) but, unlike the old
Hermes run (MBPP+ +68%, IFEval +26%), barely spills brevity onto the other benchmarks, so the
average saving is +17.6% (old +42.0%). tool-mvt256: 351 checked, 314 pass; 173 cut, 201 -> 121 (-40%).

tool-mvt256 (stops sooner): same BFCL saving, more spill-over brevity (MBPP+ +30%, -4.4, within
noise): +25.8% / +0.6. tool-mvt1024-tokens32: 406 checked, 373 pass; 176 cut, 201 -> 106 (-47%).

Token batches cut tool thinking harder (-47% vs -30%) for no extra saving at test time and a loss
on irrelevance (-8.7): as on the old data, **paragraphs win for tool calls**.

**Tool calls: best is MVT by paragraphs.** c=256 +25.8% / +0.6 (more spill-over saving), c=1024
+17.6% / +1.7 (cleaner). Token batches over-cut (-47 to -56% of the targets) and lose irrelevance
(-8.7, -10.0); c=256 by batches is outside the limit (-4.5, SVAMP -10.6).
text-whole: 259 of 300 questions have a correct reply; targets 347 -> 312 (shortest correct reply).

Why the old tool run spread brevity onto code (MBPP+ +68%) and this one does not (+2%): not the
prompt format (Hermes questions were already in BFCL's format). Likely the strength of the lesson:
old targets 291 -> 163 (44% removed, ~130 tokens per trace, 258 traces), new 201 -> 142 (30%, ~60
tokens, 189 traces); tool-mvt256 here (40% removed) does spread to code (+30%). Untested guess.

Last night's Hermes tool runs all cost MBPP+ 7.5-10 points, the uncut one included (-9.4); the
matched ToolACE uncut run costs nothing (+0.6). The Hermes data itself disturbs code; cuts on it
add the brevity spill-over. Likely cause (untested): easy Hermes questions, a quarter of them
synthetic "no tool fits" pairs, get cut to almost no thinking, as easy multiple choice did on night 1.
text-mvt1024: 503 checked, 445 pass; 238 of 259 cut, targets 347 -> 234 (-33%).

text-mvt1024: IFEval 24% shorter and +5.6 (old data: +11% / -3.1). MBPP+ -6.2 again (cf. tool cuts).
text-mvt256: 483 checked, 412 pass; 227 cut, 347 -> 190 (-45%).

text-mvt256 saves a little more than c=1024 but costs more elsewhere (MBPP+ -7.5, SVAMP -4.4);
IFEval itself gains nothing more (+19% vs +24%). text-mvt1024-tokens32: 543 checked, 476 pass; 347 -> 218 (-37%).

**Instructions: best is MVT c=1024 by paragraphs** (+15.1% / -0.9; IFEval +24% / +5.6). Every cut
setting shortens IFEval by about a quarter and raises it (+4.4 to +6.2), and every one costs MBPP+
6-8 points: code is hurt by any cut training, whatever the kind.
Night-1 rival checked (Lightning McQwen3, blog.basta.one/blog/thinkingcap, 14 Oct): 43.2% / -0.85
without its BFCL think gate; one round of verified cuts on the raw model gave them only 11.4% (SFT),
28.1% (DPO), 42.5% on their distilled model: on-policy rounds and "too early loses" pairs matter.
choice-whole: arc_easy 55 rows (454 -> 379), commonsense_qa 148 (456 -> 408).

choice-mvt1024: 470 checked, 464 pass; ARC 454 -> 93, CSQA 456 -> 100 (-78%): as on night 1, the
model knows multiple-choice answers almost before thinking, so MVT keeps about one paragraph.
(The multiple-choice pool came out 60 ARC-Easy / 240 CommonsenseQA questions.)

choice-mvt1024 alone: the "barely think" lesson spreads to everything (MBPP+ 93% shorter, SVAMP
66%) and breaks maths (-39.4), code (-10) and BFCL multiple (-13.1); IFEval gains (+12.5). This is
night 1's finding again: multiple-choice cuts are the strongest brevity signal and need maths
shown whole beside them (night-1 winner: MC cuts + GSM8K whole x2, +37.1% / -0.6 at 8 samples).
choice-mvt256: 443 checked, 419 pass; same ~78% cut.

choice-mvt1024-tokens32: 496 checked, 487 pass; ARC 454 -> 64, CSQA 456 -> 79 (-84%).

Token batches cut multiple choice even harder (-84%) yet the model barely shortens (+8.8%) and
still loses (SVAMP -13.1): thinking that stops mid-sentence is not a habit it picks up cleanly,
as with tool calls. For multiple choice, paragraphs (strong brevity signal, to pair with maths whole).
