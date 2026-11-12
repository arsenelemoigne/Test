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

**Calibration re-check, rc1-rl-40 (calcheck rc1rl40, same 7,232 cuts x 20; 2,860 pass 20/20 vs 2,954 for rc1-soup-30 and
2,901 for ca1):** ECE arc 0.07, csqa 0.17, gsm_ic 0.36, gsm_plus 0.37, ifeval_like 0.33, kodcode 0.12, toolace 0.15; outcome-rule
safe cuts arc 80%, csqa 57%, gsm_ic 32%, gsm_plus 26%, ifeval 47%, kodcode 58%, toolace 60% (ca1: 80/56/30/26/48/62/58). Same as
ca1 and the DPO soup: 40 RL steps did not change how well the model knows when it may stop either (math still 0.97 confident ->
0.78 right on gsm_ic).
**rr1 (RL done right) started:** fresh pool seed 42, 2,846 drawn, 553 dropped by the contamination check -> 2,293 questions
(120 steps x 32 = 3,840 draws: each question seen ~1.7 times, not once); alpha 0.3; first steps: 54-65% right on its training
answers (fresh questions, not filtered by solve rate), thinking 190 -> 336 tokens over steps 1-3.

**rr1 (RL done right on rc1-soup-30: fresh pool seed 42, 2,293 questions, alpha 0.3), fresh x8:** start model re-scored
(rr1-start = rc1-soup-30) **+56.4 / -3.8** (bfcl irr +46% / -4.7, mult +41% / -4.4, simple +39% / +0.6; ifeval +48% / -5.9;
mbpp +64% / -0.9; svamp +67% / -5.6); after RL block 1 (20 steps) **+57.5 / -5.1** (bfcl irr +47% / -3.7, mult +43% / -5.9,
simple +39% / +0.3; ifeval +47% / -3.1; mbpp +65% / -4.1; svamp +68% / -10.0). Below floor - 2 -> RL stopped by its rule after
one block. Readings: (1) the SAME model (rc1-soup-30) scored -2.9 this morning and -3.8 now on the same 240 x 8 questions:
run-to-run noise of about 1 point at x8 (vLLM sampling is not bit-reproducible), so single x8 differences under ~1.5 pts are
not real. (2) RL at alpha 0.3 on fresh questions buys +1 point of saving for -1.3 points, mostly math (-10 on SVAMP): on a
model already this short, the length penalty pushes the wrong place (math), as the length-band analysis predicted.
Runtime 2 moved on to rc2 (second calibrated-cut round from rc1-rl-40).

**rc2 (second calibrated-cut round, from rc1-rl-40; seed 43, 869 q + KodCode x8, 675 replies, 5,400 cuts x 20, 2,019 right
20/20; math safe cuts down to 19-22% (from 26-34% for ca1); 1,020 pairs: shorten 472, guard 319, hard 229; 61 steps), fresh x8:**
soup 10 **+57.5 / -4.5** (svamp +66% / -7.8, mbpp +67% / -5.0, ifeval +48% / -1.6, bfcl irr +47% / -3.4, mult +44% / -5.0,
simple +40% / -1.9); soup 20 +63.4 / -4.9 (svamp +71% / -9.1, mbpp +73% / -2.8, ifeval +57% / -4.1, bfcl +43-51% / -5.6..-0.6);
soup 30 +63.6 / -4.9 (svamp +70% / -10.9, mbpp +76% / -2.8, ifeval +54% / -2.2, bfcl +45-52% / -6.2..-0.3); soup 60 +67.0 / -6.2
(svamp +71% / -9.4, mbpp +80% / -9.7, ifeval +56% / -2.5, bfcl +49-58% / -5.0..-0.3). A second cut round buys 4-13 points of saving
at 2.4-4 more points of loss, mostly math: the model was already at its safe math length, as its own cut table showed.

**Confidence vs real outcome per cut point (cal_table.py; conf / real share right / share of cuts 20/20), the same for ca1,
rc1-soup-30 and rc1-rl-40 within 0.01-0.05:**
- math traps (gsm_ic, gsm_plus): real success ~0 up to 10% of the thinking, 0.04-0.13 at 20%, 0.2-0.33 at 50%, 0.43-0.52 at 65%,
  0.64-0.76 at 80%; confidence 0.4-0.5 at the start (honest there), then 0.73-0.88 at 65-80% while real is 0.47-0.76: the
  over-confidence is in the second half of the reasoning.
- instructions (ifeval_like): real 0.76-0.84 already at 1% of the thinking (about half the cuts right 20/20) and flat to 80%;
  confidence flat ~0.45-0.59: the thinking adds almost nothing on average; which questions need it is the open question.
- tools: real 0.65-0.72 at 1%, 0.84-0.85 at 80% (53-75% of cuts 20/20); code 0.66-0.69 at 1% -> 0.87-0.91 at 80%.
- multiple choice: real 0.86-0.89 at 1% (arc), 0.59-0.69 (csqa) -> 0.75-0.93 at 80%.
Reading: for instructions, tools and code, roughly half the questions are already right with almost no thinking and the rest
need it; a stopper is worth what it can tell the two apart per question. For math almost nothing is safe before 65%.
Probes running: pkca1 on runtime 1 (v2r3 finished), pkrl40 (test set = rc2's cuts) on runtime 2.

**Stopping probe, first run (pkrl40: rc1-rl-40's hidden states; development = the 904 rc1 replies' 7,232 cuts re-answered by
rc1-rl-40, 9% left out as 16-19/20; held-out test = rc2's 675 new questions, 5,400 cuts; configuration chosen by nested CV =
layer 21, AFTER "</think>").** Out-of-fold AUC of the 8 configurations 0.79-0.85. AUC, test set (dev in brackets):
- ALL: probe 0.864 (0.842) vs cut position 0.823 (0.801).
- math traps: gsm_ic 0.938 (0.951) vs position 0.859; gsm_plus 0.916 (0.881) vs 0.862.
- instructions 0.736 (0.753) vs position 0.546; tools 0.795 (0.742) vs 0.585; code 0.752 (0.729) vs 0.617.
- multiple choice: chance (0.53-0.61); the pooled probe also transfers poorly between types (leave-one-type-out: math
  0.89-0.90, instructions 0.52, tools 0.48, MC 0.54-0.58).
- 'confidence' scores 0.99-1.00 within math and MC, but it is the probability of the RIGHT answer (teacher-forced on the
  full right reply's answer): it uses the reference and is not available to a real stopper. Flagged in the report from now on.
Stopper (earliest cut with probe >= t, t per type from development data): ~0% saved on every type at the type allowance on
the test set, 5% (math, 2-pt allowance): the signal is real (clearly above the cut position on math, instructions, tools,
code) but not sharp enough at the top for a single threshold to stop safely. Oracle on the same test cuts: 19-22% math, 49%
instructions, 56% code, 61% tools.
Next in the queue: lot1 (rebalanced, ~1,500 new questions), probe on both lots with a learning curve and one probe per
data type (added now); then lot2 and the probe on all three.

**v2 round 3 (v2r3, from v2r2's soup; seed 23), soup step 40 chosen (usual x4 +35.6 / -0.1); fresh x8 +28.3 / -1.9**
(CI on accuracy -3.6..0.0, saving 24-32%). Against round 2 (fresh +28.6 / -0.8): no more saving, 1.1 points more loss (within
noise, but no progress). The v2 recipe stalls at ~28% on unseen questions after 3 rounds (~4-5 h per round).

**Stopping probe on ca1 (pkca1: ca1's own cuts and hidden states, CV only):** same picture as rc1-rl-40's. Layer 21 after
"</think>" chosen in 5/5 folds; out-of-fold AUC ALL 0.844 (position 0.801); math 0.937 / 0.888 (position 0.844 / 0.883),
instructions 0.729 (0.481), tools 0.764 (0.553), code 0.736 (0.646), multiple choice 0.51-0.62. Stopper ~0-17% saved, mostly
over the allowance. The internal signal was there before the cut-pair training and RL, at the same strength: training moved
where the model stops, not what it knows about having the answer.
v2 stopped after round 3 and rr4 removed from both queues (user). Runtime 2 had reached the end of its queue before lot1 was
added and released itself; restarted with lot1 + the probe on lot1 only (Drive mount retried once after a consent timeout).

**24 Oct, lot A (label_cuts.py: rc1-rl-40, paragraph-end cuts of right AND wrong replies, <= 10 cuts + the full thinking,
20 re-answers each; seed 47, 664 clean questions, 922 replies of which 563 right, 4,838 cuts).** Success by share of thinking
kept (0-20 / 20-40 / 40-60 / 60-80 / 80-<100% / full) and the "perfect stop" (earliest cut at least as good as the natural
stop: saved; best cut minus natural stop: accuracy gained; an optimistic bound, the max over ~5-10 noisy cuts):
- arc_easy (10 replies): .59/.43/.50/.50/.50/.77; 70% / +3.5
- commonsense_qa (64): .49/.58/.33/.63/.50/.57; 54% / +18.0
- gsm_ic (186): .04/.14/.29/.46/.62/.71; 33% / +3.8
- gsm_plus (113): .01/.04/.12/.31/.50/.62; 44% / +5.0
- ifeval_like (180): .81/.68/.73/.69/.86/.68; 17% / +2.7
- kodcode (247): .57/.59/.61/.61/.58/.63; 74% / +18.3
- toolace (122): .04/.37/.29/.02/.14/.41; 20% / +1.5
Readings: with wrong replies included, success at the natural stop is 0.41-0.77 (not 1); math climbs steadily with thinking;
code and instructions are flat (the thinking barely changes the outcome on average); tools are erratic. The "accuracy gained"
column is inflated by picking the best of several noisy cuts (code +18 with a flat curve is mostly that). Runtime 2's zoo on
lot A crashed (log(0) in the cut-position feature; fixed) and the runtime released itself; zoo A queued on runtime 1 after
lot B and zoo B.

**24 Oct, lot B (label_cuts.py on the ORIGINAL Qwen3-0.6B; seed 48, 639 clean questions, 924 replies of which 522 right,
8,255 paragraph cuts x 20).** Success by share of thinking kept (0-20 / 20-40 / 40-60 / 60-80 / 80-<100% / full); perfect stop
saved / accuracy gained (optimistic bound):
- arc_easy (13): .67/.78/.65/.74/.81/.76; 78% / +18.8
- commonsense_qa (76): .37/.43/.50/.54/.44/.50; 76% / +27.0
- gsm_ic (211): .24/.48/.55/.55/.55/.52; 64% / +36.9
- gsm_plus (119): .20/.48/.53/.54/.49/.49; 71% / +35.1
- ifeval_like (170): .62/.67/.69/.67/.56/.69; 56% / +9.2
- kodcode (247): .65/.65/.60/.61/.61/.63; 84% / +20.7
- toolace (88): .31/.34/.37/.31/.30/.48; 63% / +2.3
Reading: the untouched model's MATH is as good after 20-40% of its thinking as at its natural stop (0.48-0.55 vs 0.49-0.52):
the second half of its math reasoning adds nothing on average, whereas rc1-rl-40's math keeps climbing to the end (the
training already removed that slack). Instructions and code are flat for both. The untouched model has much more to save;
supports training the learned stop on the original model (option B).

**24 Oct, probe zoo A (lot A, rc1-rl-40; 643 questions, 70% development / 30% held out = 279 test questions).** Layer sweep:
signal grows with depth, best layer 20 after "</think>" (dev AUC 0.787; early layers ~0.70). AUC on the held-out TEST (ALL):
position 0.718, answer stability 0.672, the model's OWN confidence (probability of its own greedy answer, fair) 0.790,
reference-based confidence 0.805; probes: linear 0.803, linear per type 0.794, linear on the real success rate 0.782,
**MLP 0.852**, MLP on all 28 layers x 2 places 0.843, attention over the last 64 tokens 0.806-0.815. Per type, MLP: math
0.91/0.91 (own confidence 0.92/0.93: as good), instructions 0.87 (own 0.74), tools 0.88 (own 0.63), code 0.80 (own 0.72).
Reading: a small non-linear probe on one layer beats every simple signal, mostly on instructions, tools and code; in math
the model's own confidence in its answer already carries the same information. The trade-off/stopper tables of this first
report counted loss against "always right" (wrong replies included) and showed nothing; fixed (loss vs the natural stop) and
redone from the saved scores by the `decide` stage, which then applies the user's rule.

**24 Oct, decision (user's rule) and zoo B.** Zoo B (original Qwen, lot B, 281 held-out questions) TEST AUC: position 0.668,
own confidence 0.728, MLP 0.826, MLP all layers 0.828 (math 0.92-0.94 vs own confidence 0.87-0.88). Pooled trade-off at
<= 2 pts (report v2): zoo A linear 42% saved / 1.5 vs own confidence 25% / 0.2 -> margin +17 -> "classifier HELPS"; zoo B all
rows lost 7-12 points even at the tightest budget (position 67% / 8.9), which exposed a second accounting bias: the natural
stop was the SELECTED reply's outcome (longest right / longest wrong), while every cut is measured by 20 re-answers. Fixed:
the natural stop is now the full-thinking cut's own re-answers and length; reports redone by `zooreport` after the current
training. The decision itself stands as a direction (zoo A's classifiers clearly beat the simple signals on ranking and on
the pooled table); the stop training it launched is judged on the fresh set x8 anyway.
Running: `targeted --name stopB` = the original Qwen trained to stop at every type's real safe cut (lot B, 554 pairs:
code 179, instructions 84, math 202, MC 58, tools 31; shorten pairs only), 3 seeds, soups 10-40 on fresh x8 (runtime 2).
Runtime 1: lot A2 (rc1-rl-40) and lot B2 (original Qwen), then the judges retrained on twice the questions.

**24 Oct, stopB: the ORIGINAL Qwen trained to stop at its real safe cuts (lot B: 554 shorten pairs, every type; chosen =
the earliest paragraph cut whose 20 re-answers are right >= max(0.95, the full thinking's rate) + a right answer; rejected =
the full reply), fresh LoRA, 3 seeds, 32 steps, soups on the fresh set x8:**
- soup 10: +11.9 / +0.9 (svamp +21% / +0.3, mbpp +12% / +2.2, ifeval +8% / +2.8, bfcl +2-9% / -3.4..-0.3)
- soup 20: **+31.1 / -2.3** (svamp +43% / -4.1, mbpp +32% / -1.9, ifeval +28% / -0.9, bfcl +18-19% / -4.1..+0.9)
- soup 30: **+54.2 / -3.6** (svamp +42% / -5.0, **mbpp +82% / -8.1**, ifeval +39% / +0.3, bfcl +28-30% / -3.4..-0.6)
One DPO pass from the untouched model reaches 54% saved (vs 15-41% for every earlier from-raw first round: A1 35.4, B1 24.8,
v2r1 14.8), but code is over-cut (179 of 554 pairs; 82% shorter at -8). Corrected zoo reports (natural stop = full cut's
re-answers): zoo A at <= 2 pts: linear probe 35% / 1.1 vs own confidence 12% / -0.8 (classifier helps, +23); zoo B still
lost ~10 pts in every row because the pooled loss averaged types equally (a type with ~4 held-out replies swung it); pooled
now counts every reply once, report redone (zooreport r2).
Loop (user: pursue while promising): stopB2 = same without code pairs, checkpoints every 5 (runtime 2); stopB3 = lots B + B2
(twice the pairs, code included), checkpoints 15-40 (runtime 1, after lot B2).

**24 Oct, zoo A2: the rc1-rl-40 judge retrained on lots A + A2 (1,222 questions; test = 540 held-out questions).** TEST AUC
(ALL): position 0.719, own confidence 0.792, linear 0.845, linear per type 0.874, MLP 0.869, **MLP on all 28 layers x 2 places
0.905** (zoo A with half the data: 0.843). Per type, MLP all layers: math 0.91/0.88, code 0.92, instructions 0.89, tools 0.69.
Twice the questions lifted the best judge by +0.06 AUC: the judges were data-limited. Pooled trade-off at <= 2 pts: MLP all
layers 31% / 2.4 (26% / 1.4 at <= 0.5) vs own confidence 10% / 0.4. Runtime 2 stopped at the end of its queue before picking
up stopB2 (the line was pushed a minute late); relaunched.

**24 Oct, doubling the rc1-rl-40 judge's data again (user; units bought, 12.96 left before).** Zoo A (643 questions) 0.84-0.85
-> zoo A2 (1,222) 0.905. Queued: lot A3 (instructions 800 + MC 200 + KodCode 200 drawn, runtime 2 after stopB2) and lot A4
(GSM-IC 400 + GSM-Plus 400 + ToolACE 600 drawn, runtime 1 after stopB3); split by type so the two runtimes cannot draw the same
question. Same safeguards: training sources only, earlier draws removed, 8-gram contamination check (30%) against all 2,759
evaluation questions. Then zoo A4 on lots A + A2 + A3 + A4 (~2,400 questions), compared with zoo A and A2 (also on the
A + A2 held-out questions only, which keep their hash split). Expected +0.02-0.04 AUC; ~8 GPU-hours.

**24 Oct, stopB2 (original Qwen, lot B stop pairs WITHOUT code: 327 training pairs + 48 held out, 3 seeds, one epoch = 21
steps, so only soup 20 exists), fresh x8: +34.6% / -2.1** (svamp +47% / -3.1, mbpp +37% / +0.0, ifeval +30% / -2.2, bfcl
+21-22% / -3.1..-3.4). Code shortened 37% without a single code pair and lost nothing (stopB soup 20: 31.1 / -2.3, with code
pairs). Too few pairs to train longer: the run ends where stopB's savings were still climbing.
Lot plotA3 failed (models/rc1-rl-40 not restored: it was queued after runtime 2's restore). User: much more and more diverse
data like the evaluation, for the original Qwen. Lots A3/A4 replaced by plotB3 (GSM-IC 350, GSM-Plus 350, IFEval-like 500
drawn; runtime 2) and plotB4 (KodCode 300 from a wider slice, MBPP train minus MBPP+ 200, ToolACE 400, Hermes tools 250;
runtime 1 after stopB3), then zoo B4 on B + B2 + B3 + B4. callot now samples every drawn source (it used a fixed list).
User idea, not implemented yet: safe cut = as right as the question's own rate (not >= 95%); needs a split-sample check.

**24 Oct, zoo B2: the ORIGINAL Qwen's judge retrained on lots B + B2 (development 1,279 questions / 11,314 cuts; TEST =
545 held-out questions / 4,907 cuts).** TEST AUC (ALL): position 0.699, own confidence 0.740, linear 0.843, MLP 0.864, MLP all
layers 0.864, attention 0.81-0.83 (zoo B: MLP 0.826-0.828; +0.04 from more data, as zoo A -> A2). Per type (MLP): math
0.93-0.94, instructions 0.85, code 0.78, tools 0.69, multiple choice weak (0.48-0.63). Pooled one-threshold trade-off on TEST
(tokens saved / real loss vs the natural stop): <= 2 pt budget: MLP 69% / 1.5, linear one per type 65% / -0.7, own confidence
46% / 1.1, position 51% / 2.9; <= 0.5 pt: MLP 67% / 0.5. On the original Qwen (which thinks much longer than rc1-rl-40) a
learned stop at paragraph ends would save ~2/3 of thinking on these practice types at ~0.5-1.5 pts (simulation, not the
benchmark). Runtime 1 moves on to stopB3. Runtime 2: Drive mount timed out (no click); waiting for the user.

**24 Oct, planned for later (user): stop-token training.** Once the judges are more accurate (zoo B4+), train the model's
own `</think>` decision at every paragraph end ("\n\n") like the judge: safe cut -> next token `</think>`, unsafe -> keep
thinking (push `</think>` down), one decision per paragraph end, small LoRA, nothing else trained; must be clear-cut since
the evaluation samples. Rationale: DPO compares whole replies and teaches "shorter" more than "stop here"; the judge shows
the information is in layer ~20, and the `</think>` probability is the model's built-in stop judge. Also queued: benchlot
(the 240 choosing benchmark questions labelled with the original Qwen, measure only) + zbench (judge on lots B+B2+B3 tested
on them), runtime 2 after lot B3.

**24 Oct, deep-research summary (user-run; 22 sources; evidence mostly math, 1.5B-32B, nothing at 0.6B or on code/tools/
instructions).** Our judge (0.86-0.905 AUC) is at/above published probes (0.7-0.9+; 2504.05419, LYNX 2512.05325,
2511.14773); data, not probe size, is the bottleneck (LYNX: 1.4-2.8k problems). Closest transfer recipe = ESTAR
(2602.10004): classifier -> model writes its own stop token -> RL rewarding good stops (Qwen3-8B 4,799 -> 1,290 thinking
tokens at -0.7..-1.0). RL length rewards that hit our target on 1.5B math: ShorterBetter 2504.21370 (77-79%, +2.5..+3.2),
ThinkPrune 2504.01296 (~65%, capped = 0, tightening cap; -2.3 on AIME), AdaptThink 2505.13417 (53%, +2.4), DRPO
2510.04474 (length ranked only among right answers: -77% at -1.1 on GSM8K). Warning for our rc1 RL: batch-normalized
right x (1 - alpha*length) can push long RIGHT answers below average (DRPO's failure); any future RL ranks length only
among right answers. Stop candidates in papers: "Wait"/"Hmm" cue words, fixed delimiters, every 64 tokens; untested vs
paragraph ends. Roadmap: lots/zoos now -> stop-token training (ESTAR step 2) -> short DRPO-style RL -> separate tests
(never-solved "stop circling" with a tightening cap; cuts before "Wait"/"Hmm").

**24 Oct, 10 papers read (HTML, partly summarised pages; verify numbers before citing).** ESTAR 2602.10004: LightGBM on
logprob features -> ESTAR-FT (SFT, <stop> at <= 5 safe positions per trace, positives only, hint-injected wrong traces) ->
GRPO (16 rollouts, 39 steps; reward 1/4 format + 1/4 earliness + 1/2 correct; truncate at first verified <stop>); the 3.7x
headline still uses the classifier at test time; ESTAR-FT alone MATH500 3,962 -> 2,401 (-39%) vs ESTAR -84%. DRPO
2510.04474: length softmax-weighted only among correct answers, wrong answers pushed down regardless of length; GSM8K -77%
at -1.1 (1.5B); lr 1e-6..2e-6, 8 samples, 1,000 steps. ThinkPrune 2504.01296: reward = correct within cap L, else 0;
4k -> 3k -> 2k, each round from the best checkpoint within 10% relative accuracy; 1.5B -65% at +0.6, QwQ cliff (-3.2).
ShorterBetter 2504.21370: alpha*correct - beta*|len - shortest correct|, mean length if none correct (shortest-if-none
collapses); 1.5B -77..-79% at +2.5..+3.2; removes verification (-14% of tokens). AdaptThink 2505.13417: empty-think mode
via importance sampling, delta 0.05; GSM8K 87-99% no-think. LYNX 2512.05325: 1 forced answer per cue, MLP on 2 mid + last
layer, conformal threshold over positives only (weaker guarantee than claimed). 2504.05419: "\n\n" chunks, last-layer probe,
often linear, AUC > 0.7, 19-24% saved. DEER 2504.15895: geometric-mean confidence > 0.95 at "Wait"; code compresses 19.9% vs
math 61.5%. Dynasor 2412.20993: agreement of probes every 64 tokens. FlashThink 2505.13949: 7B yes/no verifier, GSM8K
76-91% (not 89-95%). Plan update: stop-token SFT WITH negatives -> RL with verify-at-stop + DRPO length weighting; soft
k/20 labels; report ECE/Brier; never-solved: ThinkPrune-style gentle cap test; instructions: AdaptThink-style test later.

**24 Oct, correction on ESTAR (full paper re-read by the user's research AI).** Every ESTAR result keeps the LightGBM veto at
test time (a <stop> is accepted only if the classifier scores >= 0.9), including ESTAR-FT; the paper never measures the plain
model. ESTAR-FT's job was to propose candidate stops (fewer classifier calls), not to save tokens (MATH500 2,401 vs 2,019 for
the classifier alone); weak labels (one greedy forced answer = the model's own final answer), only the first 5 safe moments
get <stop>, MATH-500 used for SFT data. RL (on-policy, rollouts cut at the first classifier-confirmed stop, earlier = more
reward) took MATH500 2,019 -> 635, always under the veto. Takeaway: stop-token SFT + RL works WITH a veto; without one (our
case) it is untested, so explicit unsafe -> keep-thinking training with our 20-answer gold-graded labels (all safe moments
labelled) is the key step; in our RL, check stops with the judge / forced answers during training only and measure the plain
model alone on fresh x8.

**24 Oct, stopB3 (original Qwen, lots B + B2 stop pairs, code included: 1,058 pairs, 3 seeds, checkpoints 15-40), fresh x8:**
- soup 15: +18.3% / -1.0; soup 20: +27.1% / -2.0; **soup 25: +34.6% / -0.4** (bfcl +20-22% / 0.0..-3.7, ifeval +28% / +2.8,
  mbpp +50% / -0.3, svamp +36% / -2.2)
- soup 30: +54.8% / -3.2 (mbpp +85% / -8.4, svamp +37% / -3.7); soup 40: +69.1% / -5.7 (mbpp +94% / -8.1, svamp +48% / -12.5)
Twice the pairs of stopB2 gives the same saving (34.6%) at a much smaller loss (-0.4 vs -2.1), with code shortened 50% at -0.3.
Past step 25 code collapses again (85-94% shorter at -8), as in stopB: code pairs over-cut once training runs longer.
Best clean point on the stop line so far = stopB3 soup 25; still below rc1-rl-40 (53.9 / -2.1).

**24 Oct, wronglen (thinking length on fresh x8, right / wrong / never-solved answers vs the original Qwen; never-solved = 0/8
for the original, 14% of answers, no training pair ever came from them).** Every model shortens wrong and never-solved answers
as much as or more than right ones: rc1-rl-40 ALL right +58% / wrong +61% / never solved +59%; stopB2 soup 20 +36 / +43 / +31;
stopB3 soup 25 +37 / +49 / +46; stopB3 soup 30 +51 / +77 / +80. The shortening spills over onto answers training never saw,
and on wrong answers most (except svamp never-solved: stopB3 soup 25 only +3%).

**24 Oct ~17:35, both runtimes alive (Colab session list: both GPU, last execution 0 min ago; logs read from Drive).**
Runtime 1: lot B4 drawing found only 362 new questions of 850 (mbpp_train/toolace/hermes_tools; earlier draws removed) +
KodCode from the wider slice (3,063 tasks); 20 answers each, first batch 79% at 17:34. Then zoo B4, stopB4.
Runtime 2: lot B3 = 743 new questions of 1,200 drawn x 20 answers (5 batches of 150), 3rd batch 95% at ~17:35 (4th 63% at 18:14, ~50 min per batch); then
labelled cuts, benchlot, zbench.

**24 Oct, DRPO implemented (user agreed; not run yet).** Full paper read (ICLR 2026). Our rc1 RL reward was exactly DRPO's
RLOO-LP baseline (Arora & Zanette) + ALP's solve-rate scaling, but our failure was the opposite of DRPO's diagnosis:
thinking grew (200-250 -> 400-500 tokens at 82-96% right; alpha 0.15 drowned by the correctness reward), not right
answers being cut. DRPO helps both ways: right replies always pushed up with a softmax(r_l / lambda) weight normalised
among the RIGHT replies only (strong, scale-free length preference), wrong replies pushed down (softmax(s / tau),
length-blind), trust region beta0 [KL - delta]+^2. Our adaptation: r_l = 1 - len / the group's longest reply (relative,
so tool calls and code feel the same pressure) instead of 1 - len / 8k. rl_train.py --objective drpo (drpo_weights
tested: Figure-1 group -> [0.62, 0.33, 0.05 | -1/3 x3]); stage `drpo`: fresh shuffled pool mixed like the evaluation
(GSM-IC/Plus, IFEval-like, ToolACE, Hermes, MBPP-train, KodCode; contamination-checked, earlier draws removed),
48 questions x 8 x 60 steps (~2,900 questions each used once), from rc1-rl-40, lambda 0.2, checkpoints 40 and 60 on fresh
x8 (~8 GPU-hours). Success (fixed before running): >= 5 pts more saving than rc1-rl-40 (>= ~59%) at accuracy >= -2.5, or
>55% at better than -2. Caveat: RL cap 4,096 for every type (code's evaluation cap is 8,192). Waiting for the GPUs to be
visible again and Drive space (arsene@lemoigne.eu nearly full).

**24 Oct, experiment 5 coded and queued (user): the judge as teacher (teach_stop.py, stage `teach`).** The judge (MLP on
layer 20 at the end of a closed cut, as probe_zoo's best) is fitted on the original Qwen's label lots (B..B4; safe =
20/20, unsafe <= 15; held-out AUC printed); the original Qwen answers 3,000 fresh mixed questions once (shared
`mixed_pool` with drpo: benchmark-like, contamination-checked, earlier draws removed); the judge scores up to 16
paragraph ends per answer (never the natural end), target = sigmoid((p - threshold) / 0.03) so the per-paragraph stop
signal is near 0 unless the judge is clearly confident (small chances would otherwise add up); a fresh LoRA trains only
p("</think>" next) at those positions (binary cross-entropy), one model per threshold (0.8, 0.9), each scored on fresh x8.
Queued on runtime 2 after drpo1. Not run yet; pure parts checked on toy data (CPU, no torch locally).

**24 Oct ~19:50, no MBPP anywhere (user).** The challenge README only forbids training on the evaluation benchmarks, and MBPP+
is built on MBPP, so MBPP-family questions are out even after the 8-gram check (its suggested sets: GSM8K, Skywork-OR1 easy
band, CommonsenseQA, ARC Easy). Lot B4 drew 200 mbpp_train questions (added 14:36; already generated, so its stage line is left
as is): zoo B4 now drops them (`probe_zoo.py --drop-sources`, before the held-out split), stopB4's --tg-shorten no longer lists
mbpp_train (targeted_pairs makes no pair for unlisted sources), and the drpo pool's 6% MBPP share moved to KodCode (16%).
Lots B/B2/B3 had no MBPP; rc1-rl-40 (drpo1's start) comes from clean line A (MBPP-free).
Lot B3: contamination check dropped 268 of 743 questions (>= 30% 8-gram share with an evaluation question); labelling
455 questions / 792 replies (422 right) / 7,436 cuts x 20 since 19:28.

**24 Oct, fairer self-confidence baselines (user question: is "own confidence" fair on long answers?).** own_confidence was
the geometric mean of the per-token probabilities of the model's greedy answer from the cut: fine for short answers (math;
on rc1-rl-40 it ties the judge there), but long answers (code, instructions) are drowned by easy tokens, so the judge's
edge there may be overstated (the judge-vs-reality numbers and the replay savings are unaffected). own_conf.py now also
keeps own_min (least likely token), own_low5 (5 least likely), own_first50 (first 50 tokens, DEER's code trick) and
own_certainty (minus the mean per-token entropy from top-5 log-probs); probe_zoo reports them as baselines; probezoo always
calls own_conf (it recomputes files without the variants). Zoo B4 (and zbench) will show them.

**24 Oct ~20:10, runtime 2 ended mid lot B3 labelling** (27% of 148,720 forced answers; log stops with no error; gone from
Colab's session list at 21:24). The 20-answer generations are on Drive; labelling has no resume, so a relaunch redoes it
(~2.5 h), then benchlot, zbench, drpo1. Runtime 1 still on lot B4 (KodCode batch 2, 35% at ~21:20).

**25 Oct, cross-entropy ideas (user, from the "Compression is intelligence part 2" video) added to the pipeline.**
(1) Soft judge: experiment 5 now also fits a judge on every cut's real success rate k/20 with cross-entropy (the yes/no
judge drops the 16-19/20 cuts), so its output is a calibrated "right this often if it stops here". (2) Relative stop
target: variant rel:0.05 trains p("</think>") toward sigmoid((soft success here - max(soft success of the full thinking
- 0.05, 0.3)) / 0.03), i.e. stop where stopping is as right as finishing (the user's per-question rule; 0.3 floor keeps
expected failures out, "stop circling" stays a separate test); it replaces the fixed-0.8 model, so teach1 still trains two
(fixed:0.9, rel:0.05). (3) Calibration grading: teach_stop prints held-out AUC, log loss, Brier and calibration error
for both judges; probe_zoo adds a TEST calibration table (log loss / Brier / ECE against k/20) for every classifier, so zoo
B4 reports it. Extra compute: one judge pass per answer on the full thinking (~5%); everything else is arithmetic.

**25 Oct ~03:00, runtime 2 relaunched (third try).** 21:34 relaunch: Drive mount failed (consent pop-up unanswered); 01:45 one
mounted but its queue sat in a FULL restore (benchlot/drpo/teach were not on the light-restore list, so every run, adapter and
model was copied from Drive) and the page lost it. Fix 11f3e5b: those stages use the light restore. Fresh runtime at ~02:45:
light restore 12 min, then lot B3 labelling restarted (455 questions / 792 replies / 7,436 cuts x 20) at ~02:58.
Runtime 1: lot B4 labelled (647 questions / 921 replies / 7,651 cuts x 20; contamination dropped 15), grading at 02:30.

**25 Oct ~03:50, both runtimes gone.** Colab's session list: no active sessions. Runtime 1's log stops at ~02:44 (lot B4 in its
grading step, after the 153,020 forced answers); runtime 2's queue cell "ended unsuccessfully" after 1,606 s (~03:13, lot B3
labelling at 9%), no error text, no out-of-memory sign. Both dying within half an hour points at an account-level stop
(compute units or a usage limit), not the code; not verified (the plan page could not be read). Nothing restarted overnight:
relaunching needs the user's Drive consent click and the unit balance checked first. On relaunch, finished generations are
reused (lot B3/B4 answers on Drive); labelling redoes its forced answers unless verdicts.jsonl was saved.

**25 Oct ~05:40 (CET), lot B3 labelled (original Qwen, 20 forced answers per paragraph cut; right AND wrong replies).**

| source | replies (right) | cuts | forced-answer success by share of thinking kept: 0-20 / 20-40 / 40-60 / 60-80 / 80-100 / full | perfect stop: saved / accuracy gained |
|---|---|---|---|---|
| gsm_ic | 397 (200) | 4,282 | 0.25 / 0.46 / 0.47 / 0.49 / 0.52 / 0.52 | 66% / +29.5 |
| gsm_plus | 184 (84) | 2,003 | 0.21 / 0.47 / 0.47 / 0.46 / 0.50 / 0.47 | 71% / +31.8 |
| ifeval_like | 211 (138) | 1,151 | 0.77 / 0.73 / 0.75 / 0.73 / 0.66 / 0.73 | 61% / +11.4 |

Math: the answer is nearly settled after 20-40% of the thinking (0.46-0.47 vs 0.47-0.52 at the full reply); only the first fifth
is too early (0.21-0.25). Instructions: stopping in the first fifth is as good as finishing (0.77 vs 0.73), and the last fifth is
the worst (0.66): extra thinking slightly hurts. A perfect per-reply stop would save 61-71% of thinking and turn many wrong
replies right (+11 to +32 pts), the ceiling a judge can approach. Runtime 2 moved on to benchlot (240 choosing questions x 8).

**25 Oct ~07:00 (CET), benchlot: the same labelled cuts on the 240 choosing BENCHMARK questions (original Qwen, 8 replies each;
measure only, nothing trains on them).** Source names are the labelling kinds: gsm_plus = SVAMP, ifeval_like = IFEval,
kodcode = MBPP+, toolace = BFCL.

| benchmark | replies (right) | cuts | success by share of thinking kept: 0-20 / 20-40 / 40-60 / 60-80 / 80-100 / full | perfect stop: saved / gained |
|---|---|---|---|---|
| SVAMP | 51 (36) | 431 | 0.26 / 0.34 / 0.48 / 0.58 / 0.55 / 0.73 | 53% / +8.9 |
| IFEval | 53 (29) | 331 | 0.56 / 0.50 / 0.49 / 0.51 / 0.45 / 0.61 | 55% / +11.0 |
| MBPP+ | 52 (26) | 572 | 0.46 / 0.53 / 0.44 / 0.52 / 0.49 / 0.50 | 84% / +17.0 |
| BFCL | 148 (109) | 478 | 0.70 / 0.70 / 0.72 / 0.77 / 0.62 / 0.75 | 51% / +8.7 |

Differs from the practice lots on math: on SVAMP the answer is NOT settled early (0.34 at 20-40% vs 0.73 at the full reply;
practice GSM-IC/Plus reach their final level by 20-40%), so a judge trained on practice math may stop SVAMP too early (zbench
measures this). Code and tools: early cuts are about as good as finishing (MBPP+ 0.46-0.53 vs 0.50; BFCL 0.70 vs 0.75).
IFEval: early cuts a little below the full reply (0.50-0.56 vs 0.61). Perfect-stop ceiling on the benchmark: 51-84% saved.
Runtime 2 now on zbench (features for lots B/B2/B3, then the judge tested on these 240 questions).

**25 Oct ~07:45 (CET), lot B4 labelled (code + tools, original Qwen, 20 forced answers per cut).**

| source | replies (right) | cuts | success: 0-20 / 20-40 / 40-60 / 60-80 / 80-100 / full | perfect stop: saved / gained |
|---|---|---|---|---|
| kodcode | 466 (241) | 5,116 | 0.46 / 0.50 / 0.46 / 0.47 / 0.50 / 0.52 | 79% / +22.9 |
| hermes_tools | 301 (144) | 1,768 | 0.27 / 0.44 / 0.41 / 0.29 / 0.18 / 0.49 | 72% / +9.1 |
| toolace | 136 (67) | 569 | 0.35 / 0.46 / 0.49 / 0.41 / 0.42 / 0.54 | 46% / +6.3 |
| mbpp_train (dropped downstream) | 18 (8) | 198 | 0.26 / 0.28 / 0.34 / 0.39 / 0.42 / 0.44 | 77% / +9.2 |

Code: success is flat (0.46-0.52) wherever the thinking is cut, so most code thinking is removable (perfect stop 79%).
Tools: a forced answer late in the thinking is worse than the full reply (Hermes 0.18 at 80-100% vs 0.49): cutting tool calls
mid-reasoning hurts, the safe window is early (20-60%). Runtime 1 now on zoo B4 (lots B+B2+B3+B4, MBPP dropped).

**zbench, first lines (judge trained on lots B+B2+B3 = 2,616 questions / 23,657 cuts; TEST = the 240 benchmark questions'
1,812 cuts):** position dev AUC 0.722 -> test 0.508; linear dev 0.875 -> test 0.674; linear on the real success rate 0.822 ->
0.609. A large drop from practice questions to the benchmark (rest of the zoo pending).

**25 Oct ~08:20 (CET), zbench complete: the original Qwen's judge trained on lots B+B2+B3 (2,616 practice questions / 23,657
cuts), TESTED on the 240 benchmark questions (1,812 cuts; svamp = gsm_plus, ifeval = ifeval_like, mbpp = kodcode, bfcl =
toolace columns).** ALL AUC dev -> TEST: MLP 0.893 -> 0.736; MLP all layers 0.885 -> 0.738; attention layer 28 0.835 ->
0.714; linear 0.875 -> 0.674; own confidence (least likely token) 0.728 -> 0.717; position 0.722 -> 0.508.
Per benchmark (TEST AUC): SVAMP: own confidence 0.87-0.88 BEATS every hidden-state judge (MLP 0.74-0.75; dev math 0.94-0.95);
IFEval: MLP all layers 0.711, others 0.55-0.65; MBPP+: judges 0.73-0.77 (own confidence 0.68-0.72); BFCL: MLP 0.772.
Calibration (soft k/20 target): MLP all layers best (log loss 0.82, Brier 0.217, ECE 0.14).
Trade-off, one threshold chosen on development: the budgets do NOT transfer: MLP all layers "<= 0.5 pt" gives 64% saved at a
real 6.6-pt loss on the benchmark; own confidence (least likely token) 16-25% saved at ~0 loss. Per-type simulation: MBPP+ 92-94%
saved at ~1 pt (judges and own confidence alike); BFCL ~44% at 0.4 (MLP all layers); IFEval 43% at -0.4 (linear) to 58% at
4.0; SVAMP: every judge over-stops (8-24 pt losses).
Reading: the judge learned practice math, whose answers settle at 20-40% of the thinking; benchmark math (SVAMP) settles late
(benchlot: 0.34 at 20-40% vs 0.73 full), so the judge stops SVAMP far too early, while the model's own confidence ranks SVAMP
cuts well. Code transfers, tools mostly, instructions weakly. Before stop-token training: math needs SVAMP-like (not
SVAMP) training problems or the own-confidence signal; thresholds must be set per type and checked on benchmark-like data.

**25 Oct ~09:10 (CET), both queues ended on errors, both runtimes released themselves (last cell).** Runtime 1: zoo B4 and
stopB4 failed (runs/bon/pairs-plotB3/cal/verdicts.jsonl missing: lot B3 finished on runtime 2 after runtime 1's start-up restore,
which had copied the half-made folder). Runtime 2: drpo1 failed in rl_train.py (IFEval grader: No module named 'langdetect'; the
mixed pool was built, 48 x 60 MBPP-free), teach1 failed in mixed_pool (KeyError 'prompt': a data file of another format
contains the word "prompt"). Fixes: ensure_lots() restores a lot from Drive when its verdicts are missing (probezoo, targeted);
setup installs langdetect; mixed_pool parses rows and skips non-chat ones. The failed lines stay in the queues and rerun on the
next launch (finished stages are skipped).

**25 Oct, queued (user): per-type stop from the benchmark findings.** stopB5 (runtime 1, after zoo B4, before stopB4): the
original Qwen trained on stop pairs from lots B..B4 for code and tools ONLY (kodcode, toolace, hermes_tools; earliest safe cut
beats the full reply, per-question rule on the 20-try lots), no math / instructions / multiple choice, 3 seeds, soups 10-30
scored on fresh x8. Rationale: on the 240 benchmark questions MBPP+ thinking is removable almost anywhere (perfect stop 84%,
judge stops 92-94% at ~1 pt) and BFCL early cuts are safe, while SVAMP settles late and IFEval early cuts cost points.
Expected: large code saving (MBPP+), some BFCL saving, svamp/ifeval near zero; compare with stopB3 soup 25 (34.6 / -0.4).
DRPO (drpo1, runtime 2) stays queued unchanged (it does not use the judge).

**25 Oct, measurement fix (user: "are we measuring the right thing?").** The zoo's AUC pools every cut, so it mixes two skills:
which questions/replies will be right at all (question difficulty) and WHEN inside a reply the answer has settled (what a stop
needs). Own confidence is strong on the first (SVAMP 0.87-0.88 is largely "this easy question is known"); in the stopping
simulation on SVAMP it only wins by stopping rarely (15% saved / -3.8; least-likely-token 8% / -1.9; judge 63% / -23.8;
oracle 30% / +1.3). probe_zoo.py now also reports a TIMING-ONLY AUC (pairs of cuts from the same reply only; toy check: rising
1.0, difficulty-only 0.5, falling 0.0) for every classifier, dev and test, per type; and three new classifiers: own confidence
LEARNED (its 5 readings + position, same fit as the probes), linear + own confidence, MLP + own confidence. zbench2 (runtime 2,
before drpo1) reruns the benchmark test with these; zoo B4 uses them too.
Context for math cuts: trained models already shorten SVAMP far beyond the 30% "oracle" (that is truncation of the original
replies with all 20 forced answers right): rc1-rl-40 svamp +65% / -4.1; earlier rounds +67-69% at -0.6..+3.1 (x4);
competitor Lightning McQwen3 SVAMP 66% / -3.8. Benchlot's best-cut ceiling on SVAMP (best cut vs natural stop) is 53% / +8.9.

**25 Oct ~10:30 (CET), both runtimes relaunched by the user (scripts 36a4a86).** User: every classifier must be tested on the
real eval sample so they can be compared. zbench2 (runtime 2: lots B+B2+B3) and the new zB4bench (runtime 1, right after zoo
B4: lots B..B4, MBPP dropped) both test on the 240 choosing benchmark questions (measure only, never trained on), with the
timing-only AUC and the own-confidence combos; zoo B4 itself keeps the held-out practice test, for the practice-vs-benchmark gap.

**25 Oct, experiment 5 protects math (user).** After zbench (the judge stops SVAMP far too early), teach_stop trains math answers
(gsm_ic, gsm_plus, gsm8k) toward "keep thinking" (target 0) at every paragraph end instead of following the judge, so the
spill-over seen in every stop model cannot teach early math stops; code, tools and instructions follow the judge (fixed:0.9)
or the relative rule (rel:0.05). Math shortening is left to length training (DRPO) and to a math-specific signal (pending
zbench2's timing-only AUC for own confidence and the combined judges).

**25 Oct, ideas from the papers (user asked why ESTAR / ShorterBetter succeed).** Reading: both use far longer-thinking models
(ESTAR Qwen3-8B, 2.4-6k tokens; ShorterBetter R1-Distill 1.5B/7B, 5-11k tokens; ours ~360 tokens when right), stay mostly in
one domain (math / medical MC), and ESTAR keeps its LightGBM classifier at TEST time (tau 0.9), which our fixed evaluation does
not allow; most of ESTAR's saving comes from RL (MATH500 3,951 -> 2,019 classifier only -> 635 with RL). ShorterBetter (pure
GRPO, no classifier): out-of-domain ~62% shorter at ~-0.5 (BBH -5.2, LiveCodeBench -40% / -1.1): same range as rc1-rl-40.
Built (toy-tested only):
- probe_zoo.py: ESTAR-style self-confidence TRAJECTORY features per cut (log own confidence and least-likely-token prob, their
  change since the previous cut, running max, slope over the last 3 cuts, answer stability, share of answer changes, fraction):
  "own-confidence trajectory, learned" (small MLP) and "mlp + trajectory"; and the ESTAR LABEL ("this cut's own answer equals
  the reply's final own answer", no correctness needed): "ESTAR label: trajectory" and "ESTAR label: mlp", fitted on that label
  and scored against the usual safe/unsafe labels. Answers compared ignoring spacing. zB4bench (runtime 1) will run them;
  zbench2 started before this change.
- rl_train.py --reward shorterbetter: r = alpha*correct - beta*|len - SOL| (alpha 2; beta 0.005 instead of the paper's 0.001
  because our replies are 5-10x shorter), SOL = shortest correct reply of the group, else the group mean; GRPO advantages.
  track.py drpo --rl-variant shorterbetter. Queued sb1 (runtime 2, after drpo1): from rc1-rl-40, same fresh MBPP-free pool size
  (48 x 8 x 60), checkpoints 40/60 on fresh x8, same success bar as drpo1.

**25 Oct ~13:00 (CET), zbench2: TIMING-ONLY AUC (cuts compared only within the same reply) on the 240 benchmark questions.**
Only 83 test replies hold both a safe and an unsafe cut (SVAMP 36, IFEval 6, MBPP+ 16, BFCL 25): small, read with care.

| classifier | dev timing ALL (practice) | TEST timing ALL | SVAMP | IFEval | MBPP+ | BFCL |
|---|---|---|---|---|---|---|
| position ("later = safer") | 0.603 | **0.814** | 0.836 | 0.800 | 0.829 | 0.721 |
| own confidence | 0.806 | 0.845 | **0.963** | 0.550 | **0.850** | 0.574 |
| own confidence, 5 least likely | 0.807 | **0.871** | **0.977** | 0.625 | 0.833 | 0.721 |
| linear (hidden state) | 0.828 | 0.710 | 0.768 | 0.525 | 0.666 | 0.698 |
| mlp (hidden state) | **0.847** | 0.774 | 0.815 | 0.650 | 0.721 | **0.806** |
| mlp, all layers | 0.825 | 0.690 | 0.807 | 0.625 | 0.561 | 0.651 |
| own confidence, learned (+ position) | 0.826 | 0.813 | 0.937 | 0.475 | 0.735 | 0.721 |
| mlp + own confidence | n/a | 0.844 | 0.963 | 0.400 | 0.787 | 0.752 |

Pooled AUC (unchanged from zbench): mlp + own confidence 0.753 (best), mlp all layers 0.738, mlp 0.736, own conf. least likely 0.717.
Reading: (1) on the benchmark, inside a reply the hidden-state judges know LESS about timing than "later is safer" (0.71-0.77
vs 0.81); on practice they knew more (0.85 vs 0.60): what they learned about timing is practice-specific. (2) The model's own
confidence carries real TIMING information on the benchmark, strongest on SVAMP (0.96-0.98) and code (0.85): the earlier worry
that its SVAMP score was only question difficulty is wrong. (3) Its poor SVAMP stopping (15% / -3.8) therefore comes from the
THRESHOLD chosen on practice data, not from the ranking: per-type thresholds set on benchmark-like data are the fix. (4) Tools are
the one type where the hidden-state MLP leads (0.81). (5) Learning own confidence on practice data (with position) lowers its
benchmark timing (0.813 vs 0.845-0.871 raw): fitting to practice hurts transfer.

**25 Oct, queued (user): the benchmark test for rc1-rl-40.** All benchmark judge tests so far read the ORIGINAL Qwen (its
replies, hidden states and own confidence); rc1-rl-40's own confidence was only tested on practice questions (0.79 vs the
original's 0.74 pooled). Runtime 1, after zB4bench: benchlot benchA40 (rc1-rl-40's replies to the 240 choosing questions, cuts
x 20 forced answers, measure only; benchlot now restores a models/ start itself), then zAbench (rc1-rl-40's judge on lots A + A2
and its own confidence, tested on benchA40, with the timing-only AUC, own-confidence combos, ESTAR trajectory and label).
Compare with zbench2 / zB4bench (original Qwen).

**25 Oct, experiment 5 redesigned from zbench2 (user).** Teacher per type: the model's own confidence (geometric mean of its 5
least likely answer tokens, read with own_conf.py at every scored paragraph end and at the full thinking of the 3,000 teacher
answers) for math and code (benchmark timing SVAMP 0.977, MBPP+ 0.833); the soft hidden-state judge for tool calls (BFCL 0.806);
instructions and multiple choice protected (keep thinking). Targets use only the relative rule, within the same answer
(teacher value here >= its value at the full thinking - margin, floor 0.3): it uses the signal's timing, which transferred to
the benchmark, and no absolute threshold, which did not. Two models: margins 0.05 and 0.1. Extra cost: one greedy forced
answer per scored point (~30-50k short generations, ~1 GPU-hour).

**25 Oct ~15:00 (CET), zoo B4: the original Qwen's judge on lots B+B2+B3+B4 (MBPP dropped: 198 cuts; dev 2,447 questions /
21,539 cuts; TEST = 1,072 held-out PRACTICE questions / 9,571 cuts).** TEST AUC ALL: position 0.681, own confidence 0.721
(5 least likely 0.740), linear 0.858, MLP 0.879, MLP all layers 0.872, attention 0.80-0.83, own confidence learned 0.812,
linear + own 0.868, **MLP + own confidence 0.892** (zoo B2 MLP 0.864: +0.015 for twice the data, smaller than earlier
doublings). Per type (MLP): gsm 0.94-0.95, hermes 0.91, ifeval_like 0.86, kodcode 0.80, toolace 0.74, mc 0.59-0.62.
Timing-only TEST (429 replies with both): own confidence 0.833 vs MLP math 0.94 / code 0.76 / toolace ~0.5 / hermes 0.81;
own confidence leads on code (0.86) and toolace (0.74). Pooled trade-off on the practice test: MLP all layers 64% saved at
-0.2 (gain) for a 0.5-pt budget; MLP 70% / 2.6. (Started before the ESTAR features were pushed: zB4bench has them.)

**25 Oct ~16:30 (CET), interim.**
zB4bench (original Qwen's judge on lots B+B2+B3+B4, MBPP dropped; dev 3,519 questions / 31,110 cuts; TEST = bench40) TEST
AUC ALL: position 0.516, linear 0.670, MLP 0.722, MLP all layers 0.760, own confidence learned 0.661, linear + own 0.718, MLP +
own 0.713, **own-confidence trajectory learned (ESTAR features) 0.741**, MLP + trajectory 0.723, ESTAR label (trained on "same
as final answer", true for only 36% of cuts): trajectory 0.700, MLP 0.674; attention 0.70-0.71. Versus zbench (lots B-B3):
MLP 0.736 -> 0.722, all layers 0.738 -> 0.760: lot B4 does not move the benchmark score (practice test rose, zoo B4 0.879).
Per-type, timing-only and stopping tables still being written (report pending).
drpo1 (DRPO from rc1-rl-40) at step 24/60, training-batch stats (new questions every step, so noisy): mean thinking tokens
steps 1-10 ~369 (493 at step 1) -> steps 14-24 ~273 (-26%); batch accuracy flat (~0.61 -> ~0.60). Checkpoint 40 is the first
scored on fresh x8.

**25 Oct, queued (user): DRPO from the raw Qwen with DLER dynamic sampling (drpo0, runtime 1, after zAbench, before stopB5).**
rl_train.py --dynamic-sampling F: draws F x --questions per step and trains only on groups with at least one right reply (an
all-wrong group carries no correctness signal), keeping --questions of them; logged as dropped_groups. track drpo --rl-dynamic
1.5 (pool sized x1.5). One clean RL run from Qwen3-0.6B (no DPO/RL chain; BottleCap's own "collapse of self-trained LMs"
paper warns about repeated self-training) vs drpo1 from rc1-rl-40. Deep research (25 Oct): the best length rewards are
correctness-gated and normalized per prompt (Arora & Zanette, L1-Max, PLP, LASER, DRPO, Kimi k1.5); all published runs start
from a model that already reasons (R1-Distill-1.5B / DeepScaleR), none at 0.6B, none outside math.

**25 Oct ~17:00 (CET), zB4bench complete (judge on lots B..B4, MBPP dropped; TEST = bench40, 83 replies with both labels for
timing).** Pooled TEST AUC: MLP all layers 0.760, ESTAR trajectory learned 0.741, MLP 0.722, MLP + own 0.713, own confidence
(5 least likely, raw) 0.713. TIMING-ONLY TEST AUC ALL: position 0.814; own confidence 0.845, 5 least likely **0.871**; ESTAR
label (trajectory trained on "same as final answer") **0.834**; ESTAR trajectory learned 0.826; own confidence learned 0.807;
MLP + own 0.775; MLP 0.709; MLP all layers 0.697; linear 0.645; attention 0.65-0.68. Per type (timing): SVAMP own 0.96-0.98,
ESTAR trajectory 0.930, MLP 0.820; MBPP+ own 0.83-0.85, ESTAR label 0.822, MLP 0.617; BFCL ESTAR trajectory **0.798** (own
5-least 0.721, MLP 0.643; zbench2's MLP 0.806). Stopping simulation (2 pts each type, thresholds from practice): SVAMP every
learned signal over-stops (19-48 pt losses; raw own confidence 15% / 3.8, least likely 15% / 1.9); MBPP+ 83-91% saved at
0.8-4.4 (own 5 least likely 91% / 1.3, ESTAR trajectory 91% / 1.6); IFEval own confidence 68% / 1.0; BFCL ESTAR trajectory
23% / -2.0 (gain), linear 51% / 1.8, MLP 47% / 0.7.
Reading: more practice data (lot B4) does not help on the benchmark; the hidden-state judges' timing stays below "later is
safer"; the ESTAR self-confidence features come close to raw own confidence (0.83 vs 0.87) and lead on tools; raw own
confidence (5 least likely tokens) remains the best timing signal overall, and practice-set thresholds still fail on SVAMP.

**benchA40 (rc1-rl-40's replies to the 240 choosing questions, 8 each, cuts x 20 forced answers; measure only):** 323 replies
(206 right), 1,096 cuts (vs 1,812 for the original: rc1-rl-40 thinks less).

| benchmark | replies (right) | cuts | success: 0-20 / 20-40 / 40-60 / 60-80 / 80-100 / full | perfect stop: saved / gained |
|---|---|---|---|---|
| SVAMP | 59 (37) | 209 | 0.18 / 0.35 / 0.26 / 0.62 / 0.58 / 0.63 | 37% / +10.1 |
| IFEval | 56 (30) | 124 | 0.58 / 0.55 / 0.48 / 0.63 / 0.14 / 0.53 | 32% / +11.0 |
| MBPP+ | 57 (28) | 562 | 0.38 / 0.34 / 0.38 / 0.49 / 0.42 / 0.48 | 75% / +12.7 |
| BFCL | 151 (111) | 201 | 0.52 / 0.69 / 0.50 / 1.00 / - / 0.76 | 15% / +2.3 |

rc1-rl-40 has much less left to cut by stopping than the original (perfect stop SVAMP 37% vs 53%, BFCL 15% vs 51%), except
code (75%). zAbench (its judge and own confidence on these cuts) is running.
Schedule change (another session, 1589f05): drpo0 = DRPO from the raw Qwen with DLER dynamic sampling, runtime 1 after zAbench,
before stopB5 (adds ~8-10 GPU-hours to runtime 1).

**25 Oct, target and clean recipe (user).** Goal now: > 50% saved at better than -1 pt, clean (evaluation untouched, no
test-time gate), short recipe. Public forks read: basta/lightning-mcqwen3 47.3% / -0.9 (43.2 / -0.85 without its test-time
think gate; recipe: Chain-of-Draft prompt distillation SFT -> cut oracle (4 re-answers) -> shorten/guard/hard DPO ->
NLL-anchored correctness DPO -> layer-21 probe gate in evaluate.py) and michal-bida thinking-compression 47.0% / -0.78 (one
DPO round from the raw model on own graded samples, benchmark-matched prompts, shortest right vs longest wrong/truncated +
vs >= 1.5x longer right, balanced per domain; SFT only +4.5%, offline GRPO failed, long pairs alone collapse). Planned
architecture (ours): raw Qwen -> experiment 5 (own-confidence stop spikes) -> DRPO; the balanced round is the baseline /
fallback stage 1, queued as balround bal0 (runtime 1, after drpo0): balanced_pairs.py (toy-tested) + dpo.py's NLL anchor
(already default 1.0), GSM8K instead of GSM-IC/Plus for math, 6,000 questions x 4, 3 seeds, soups 50-200 on fresh x8.

**25 Oct, queued (user): teach2 = experiment 5 with tool calls also taught by own confidence (runtime 2, after teach1).**
zB4bench: with lots B..B4 the hidden-state judge's BFCL timing fell to 0.64 (own confidence, 5 least likely: 0.72; ESTAR
trajectory 0.80). teach2 reuses teach1's answers and own-confidence readings (--teach-reuse), so only the teachers differ:
own confidence for math, code and tools (--teach-own ...,toolace,hermes_tools; --teach-judge none), instructions protected,
margins 0.05 and 0.1. track teach gains --teach-own / --teach-judge / --teach-reuse.

**25 Oct ~18:00 (CET), interim.** drpo1 training batches (new questions each step): mean thinking tokens ~369 at steps 1-10
-> ~273 at 14-24 -> **110-160 at steps 46-56** (-60 to -70% vs the start); batch accuracy ~0.61 at the start, 0.48-0.70 at
steps 45-56 (noisy, possibly a little lower). Checkpoints 40 and 60 are scored on fresh x8 after step 60.
zAbench first lines (rc1-rl-40's judge, lots A + A2: 1,762 questions / 9,595 cuts; TEST = benchA40, 1,096 cuts): pooled TEST
AUC position 0.512, linear 0.649, MLP 0.694, MLP all layers 0.745, attention 0.73, own confidence learned 0.661, MLP + own
**0.768**, MLP + trajectory 0.749, ESTAR trajectory 0.696, ESTAR label trajectory 0.736 (same-as-final true for 52% of cuts).
Raw own confidence, timing-only and stopping tables pending.

**25 Oct ~18:15 (CET), drpo1 training finished (60 steps; 48 new questions x 8 replies per step; rl_train.py --objective drpo
--drpo-lambda 0.2 --lr 1e-5 --updates 2 --max-tokens 4096, from rc1-rl-40).** Training-batch averages per 10 steps:

| steps | batch accuracy | thinking tokens | total tokens (thinking + answer) |
|---|---|---|---|
| 1-10 | 0.62 | 369 | 414 |
| 11-20 | 0.60 | 289 | 338 |
| 21-30 | 0.61 | 266 | 307 |
| 31-40 | 0.69 | 218 | 270 |
| 41-50 | 0.66 | 149 | 184 |
| 51-60 | 0.59 | 129 | 187 |

Thinking -65% from the first block; batch accuracy flat to slightly up until step 50, lower in the last block (0.59; noisy,
new questions each step). Watch: the answer part grows at the end (total - thinking: ~45 tokens early -> ~58 at steps 51-60;
step 57: 243 total vs 123 thinking), i.e. some reasoning may be moving out of the thinking block; the fresh x8 scoring counts
the reply's tokens, so the checkpoint scores (40 and 60, now being merged and scored) are the real test.

**25 Oct (user): more RL checkpoints tested.** "Merging" = folding a checkpoint's LoRA adapter into a model copy so the
official vLLM evaluation can score it (weight 1, no averaging): it is how a checkpoint is tested. drpo1 only saved steps 20/40/60
(40 and 60 queued for scoring); queued a second drpo1 line with --drpo-score 20 40 60 (training skipped, 40/60 already scored,
adds step 20). track.py drpo now saves every 10 steps and scores 20/40/60 by default (sb1, drpo0 and later runs).
Also seen in the queue (another session): teach2 (--teach-reuse teach1) after teach1 on runtime 2.

**25 Oct (user): RL checkpoints tested one by one, then a soup chosen from their scores.** track.py drpo --drpo-soup S1 S2 ...
averages those checkpoints' LoRA adapters with equal weights (merge.py, as the DPO soups) into models/<name>-soup-S1-S2 and
scores it on fresh x8 (training skipped when the run exists). Plan: drpo1 steps 20 / 40 / 60 scored individually first; the soup
line is queued once those scores are in (e.g. the two best neighbouring checkpoints).

**25 Oct ~19:10 (CET), drpo1 step 40 (DRPO from rc1-rl-40), fresh 240 x 8: tokens saved +73.3%, accuracy -6.9 (OUTSIDE).**
bfcl_irrelevance +60% / +1.6; bfcl_multiple +52% / -7.2; bfcl_simple_python +50% / +0.6; ifeval +66% / -7.2; mbpp +83% / -2.5;
**svamp +80% / -16.2**. Accuracy: svamp 73.4%, ifeval 58.4%, mbpp 43.8%, bfcl 68.8-88.8%. Mean reply tokens: svamp 126,
ifeval 237, mbpp 349, bfcl 78-100. Far more saving than rc1-rl-40 (53.9 / -2.1) but math collapses: DRPO with lambda 0.2
pushes too hard by step 40 (the training batches' flat accuracy hid it: they mix types and change every step). Step 60 is being
scored; step 20 next (queued) is the likely sweet spot; a soup of step 20 with rc1-rl-40-like weights (or steps 20+40) is the
candidate merged model.

**25 Oct, queued (user): thinking OFF.** How much does thinking add at this scale? `track.py nothink` (runtime 2, right after
drpo1's step-20 scoring, before sb1): the original Qwen and rc1-rl-40 answer the fresh set x8 with thinking switched off (a copy
of the official evaluate.py with enable_thinking=True -> False: the chat template writes an empty think block and the model
answers directly; measure only, never a submission setting). Reported vs the original thinking (f8-base), and rc1-rl-40 also vs
itself thinking: tokens saved and accuracy change per benchmark (= the accuracy that thinking buys, and its token price).

**25 Oct, why training accuracy hid the test drop (user) and RL monitoring fix.** drpo1's batch accuracy is a mixed average
(~34% math, 24% instructions, 30% tools, 16% code) over new questions every step (batch-to-batch swing 0.48-0.81), so a 16-pt
math drop moves it ~5 pts, inside the noise; the practice math (GSM-IC/Plus) settles early unlike SVAMP; training grades
IFEval-like prompts with our checker, not the official IFEval; and training starts from rc1-rl-40 (already -2.1), so step 40 is
-4.8 vs its own start. Fix for sb1 / drpo0 (not started yet): rl_train.py logs per-type batch accuracy and thinking every step
("by_type"); track.py drpo scores every saved checkpoint (10, 20, ..., 60) on fresh x8 by default (~7 min each for these
short models).

**25 Oct, queued (user): SVAMP-like math without cheating (lot B5, runtime 1 after zAbench).** New source gsm8k_easy
(prepare_types.py): GSM8K TRAIN problems whose reference solution makes 1-2 calculations (counted by its <<...>> annotations;
2,579 of 7,473: 404 one-step, 2,175 two-step), the difficulty SVAMP's paper describes; README-suggested set; GSM8K test (GSM-Plus
/ GSM-IC base), ASDiv and MAWPS (SVAMP's sources) not used; earlier draws removed and the usual 30% 8-gram check vs all 2,759
evaluation questions. callot plotB5: 600 drawn, original Qwen, 20 tries each, 20 forced answers per paragraph cut. Before any
use, its cut-success curve is compared with SVAMP's on the choosing set (benchlot: 0.26 / 0.34 / 0.48 / 0.58 / 0.55 / 0.73) and
with the practice math (GSM-IC 0.25 / 0.46 / 0.47 / 0.49 / 0.52 / 0.52). Also fixed: callot's earlier-draw filter parsed rows by
substring (the KeyError teach1 hit; rl-*-text.jsonl files now exist on Drive).
Also seen in the queue (another session): balround bal0 (one balanced DPO round from the raw Qwen, 6,000 questions x 4, DPO up
to 200 steps x 3 seeds) on runtime 1 after drpo0: a large addition.

**25 Oct, queued (user): a math corpus matched to SVAMP, with a held-out math test.** `track.py mathmatch` (runtime 1, after
zAbench, before lot B5): (1) profiles the CHOOSING set's 40 SVAMP questions (measure only): operations (SVAMP's Equation),
kind (its Type: Addition / Subtraction / Common-Division / Multiplication), length in words, and the original Qwen's accuracy
and thinking from benchlot's 8 replies; (2) picks 1,500 GSM8K problems (train + test splits, 1-2 calculations; never
ASDiv/MAWPS/SVAMP) stratified on SVAMP's ops x kind cells and length, written in the evaluation's numeric prompt wording,
earlier draws removed, contamination-checked; (3) holds out 300 as a MATH TEST set (data/gsm8k_matched-test.jsonl, never drawn
for training) and has the original Qwen answer it x8, printing accuracy / thinking / length per cell next to SVAMP's. The
other 1,200 (data/gsm8k_matched-train.jsonl, source gsm8k_matched) feed lot B5 (600 labelled, 20 tries) and later the RL pool
and stop teaching. Known limit: SVAMP is ~76% one-operation (whole set; 31/40 in the choosing set) while GSM8K has few
one-step problems (~330 usable), so every one-step problem is taken and the rest are two-step: the corpus is ~22% one-step.
SVAMP whole-set composition (looked up only to describe it): Subtraction 531, Addition 195, Common-Division 166,
Multiplication 108; one operation 762, two 237. Tested locally on fake model outputs (selection, split, tables).

**25 Oct ~20:00 (CET), drpo1 checkpoints on fresh x8 (DRPO from rc1-rl-40):**

| checkpoint | tokens saved | accuracy | svamp | ifeval | mbpp | bfcl irr / mult / simple |
|---|---|---|---|---|---|---|
| step 20 | 61.7% | **-4.9** | +74% / -10.0 | +49% / -5.0 | +69% / -1.9 | +54% / -1.6, +46% / -6.6, +42% / -0.3 |
| step 40 | 73.3% | -6.9 | +80% / -16.2 | +66% / -7.2 | +83% / -2.5 | +60% / +1.6, +52% / -7.2, +50% / +0.6 |
| step 60 | 78.7% | -6.0 | +85% / -14.1 | +73% / -5.9 | +88% / -3.1 | +64% / +2.8, +55% / -5.0, +53% / -0.3 |
| (rc1-rl-40, start) | 53.9% | -2.1 | +65% / -4.1 | +46% / -0.6 | +59% / -1.9 | +41-45% / -1.2.. |

Every checkpoint is outside the limit; even step 20 loses 10 points on SVAMP and ~6 on IFEval / BFCL multiple. DRPO at lambda
0.2 / lr 1e-5 overshoots within the first 20 steps. A soup of the three checkpoints cannot repair math (all three lose it);
the merged candidate worth testing is rc1-rl-40 + half of step 20 (step-20 adapter at weight 0.5, i.e. halfway back to the
start), or a gentler rerun (lambda higher / lr 5e-6) with every 10-step checkpoint scored.

**nothink nt1 (runtime 2): the original Qwen with thinking OFF, fresh x8** (it then crashed in compare.py, which refuses two
runs of one model; replaced by a records-based diff, nt2 queued on runtime 1 for rc1-rl-40). Accuracy thinking ON (from the
drpo1 deltas, f8-base) -> OFF; mean reply tokens OFF: bfcl irrelevance 80.9 -> 87.2 (+6.3; 7.9 tokens); bfcl multiple 76.0 ->
58.8 (-17.2; 20.5 tokens, 11.9% unreadable); bfcl simple 88.1 -> 76.6 (-11.5; 24.3 tokens, 9.4% unreadable); ifeval 65.6 ->
60.6 (-5.0; 284 tokens); mbpp 46.3 -> 37.2 (-9.1; 53 tokens); **svamp 89.6 -> 17.5 (-72; 9.7 tokens, 24.7% unreadable)**.
Thinking is worth ~72 points on SVAMP (part of it format: a quarter of no-thinking answers miss the "Final answer" line), 9 on
MBPP+, 5 on IFEval, 11-17 on two BFCL subsets, and nothing (it even hurts) on BFCL irrelevance.

**25 Oct, queued (user): two fixes for drpo1's overshoot.**
- drpo1 step 20 at HALF strength (runtime 1, after nt2): track.py drpo --drpo-weighted 20:0.5 merges the step-20 LoRA at weight
  0.5 into rc1-rl-40 (the delta halved: halfway between the start and step 20) and scores it on fresh x8. The drpo stage now
  restores an adapter / pool trained on the other runtime instead of retraining.
- drpo2 (runtime 2, after sb1): DRPO from rc1-rl-40 again, gentler: lambda 0.5 (was 0.2; larger = softer length preference)
  and lr 5e-6 (was 1e-5), new pool (seed 64), per-type batch accuracy logged, checkpoints every 10 steps each scored on fresh x8.

**25 Oct, where thinking HURTS (user question) and RL from the raw Qwen with math protected (queued).** Thinking makes results
worse on BFCL irrelevance (thinking OFF 87.2% vs ON 80.9%, +6.3), and cut-curve data point the same way for parts of the
instruction prompts (practice IFEval-like: forced answer after 0-20% of the thinking 0.77 vs 0.73 full; the last fifth is the
worst, 0.66; rc1-rl-40 on IFEval: 0.14 at 80-100%) and for late tool calls (Hermes 0.18 at 80-100% vs 0.49 full). Thinking is
essential on SVAMP (-72 without it). Strategy that follows: protect math, cut everything else hard.
Built (syntax-checked only): rl_train.py --protect-sources (those groups get NO length pressure: DRPO weights right replies
equally, ShorterBetter's length term is 0; they are still trained to be right) and --guard-drop G (per type: stop when the mean
batch accuracy of the last 5 steps falls more than G below the first 5 steps'; that checkpoint is kept and scored). track.py
drpo --rl-protect / --rl-guard. Queued drpo3 (runtime 2, after drpo2): from the RAW Qwen, math protected (gsm_ic, gsm_plus,
gsm8k*), lambda 0.5, lr 5e-6, DLER dynamic sampling 1.5, guard 0.06, every 10-step checkpoint scored on fresh x8.
Already queued that is related: drpo0 (runtime 1, another session: raw Qwen, DRPO lambda 0.2, dynamic sampling, no protection).

**25 Oct ~20:30 (CET), zAbench: rc1-rl-40's own confidence and judge on the benchmark (benchA40; 53 replies with both labels).**
Pooled TEST AUC: MLP + own 0.768 (best), MLP all layers 0.745, ESTAR label trajectory 0.736. TIMING-ONLY TEST AUC ALL: own
certainty (entropy) **0.870**, own confidence first-50 0.860, own confidence 0.854, ESTAR label trajectory 0.841, position 0.827,
linear + own / MLP + own 0.824, MLP 0.767, MLP all layers 0.728. SVAMP timing: own confidence 0.91-0.93 (original Qwen 0.96-0.98),
MLP 0.82, position 0.82. As for the original Qwen: the model's own confidence is the best timing signal, the hidden-state judge
is below "later is safer". Stopping simulation (2 pts per type, thresholds from rc1-rl-40's practice lots): SVAMP: every signal
stops almost nothing (0-2%; the oracle bound is 17%); IFEval: stopping early GAINS accuracy (own confidence 33% saved at +6.9,
MLP all layers 31% / +5.6); MBPP+: own confidence 87% / -3.2, MLP all layers 72% / +0.4 (gain), MLP 54% / +2.8 (gain); BFCL:
~19% / -2.2. Reading: on our best model, a learned stop would add most on code (and instructions), little on math or tools.
Queue note: runtime 1 re-read its queue at ~18:30 (before mathmatch / nt2 / drpo1 half-strength / lot B5 were added) and started
drpo0 (raw Qwen DRPO, ~8 h); those four run right after it (they sit above it in the queue file).

**25 Oct ~21:15 (CET), queues reordered (user, after another AI's review).** Removed: drpo0 (raw-Qwen DRPO with the aggressive
lambda 0.2 / lr 1e-5 that overshot, no math protection: drpo3 covers it safely) and stopB4 (superseded by stopB5 and
experiment 5). Runtime 1 from now on: (drpo0 still running, step 3 of 60, ~1,300-token replies) -> mathmatch -> nt2 -> drpo1
step 20 at half strength -> lot B5 -> stopB5 -> bal0. Runtime 2: sb1 (running, step 9; every 10-step checkpoint scored) ->
drpo3 -> teach1 -> teach2 -> drpo2. drpo0 cannot be stopped by editing the queue (a running stage finishes first); stopping it
needs "Interrupt execution" on runtime 1's queue cell and re-running that cell (saves ~9 GPU-hours).

**25 Oct ~22:00 (CET), experiment 5 without any classifier (user).** teach1 now uses the model's own confidence as the teacher for
every type it covers (math, code, tool calls: --teach-own ... --teach-judge none); instructions and multiple choice still "keep
thinking". Reason: on the benchmark, own confidence is the best timing signal overall (0.87 vs 0.77 for the hidden-state judge)
and the judge's only edge (tools, 0.81 vs 0.72 on 25 replies in zbench2) vanished with lots B..B4 (zB4bench: judge 0.64, ESTAR
own-confidence trajectory 0.80). teach2 (the same configuration reusing teach1's answers) removed as a duplicate. Runtime 2 order:
sb1 (running) -> drpo3 -> teach1 -> drpo2. GPU 1 relaunched by the user/assistant at ~22:00 on the reordered queue (drpo0
interrupted at step 3; the release cell that followed was stopped before releasing the runtime; Drive remounted).

**25 Oct, SVAMP double-check (user: "some models cut math a lot while improving it?").** Yes, but only on the CHOOSING set
(questions_40, also used to pick those checkpoints): gentle round-2 model step 10, 240 x 8: svamp +64% / 0.0 (23 lost, 23
gained); balanced soup x8: svamp +68% / -1.6; several x4 checks +55-69% at -0.6..+3.1. On FRESH questions every heavy math cut
lost: round 2 retrain -9.7, round 2 on questions 41-80 -7.5, rounds 4 -9.1 / -10.6, balanced soup +64% / -4.7, rc1-rl-40 +65% /
-4.1 (the log noted at the time: picking checkpoints on those 40 SVAMP questions selected for them). Reliable conclusion: a
~2/3 SVAMP cut has cost ~4-10 points on unseen questions; "no loss" results were selection on the choosing set.
Live progress for stop teaching (user): teach_stop.py now writes progress-<variant>.jsonl and prints every 100 answers, per
source: mean p("</think>") where the target says stop vs where it says keep thinking, thinking saved if the model stopped now
(first paragraph end with p >= 0.5), and the share of those stops at a good place; for plotting during training.

**25 Oct ~23:15 (CET), runtime 1 restarted with the light-restore fix (balround was missing from the light list: the first
relaunch sat 45 min in a full Drive restore).** mathmatch (mm1) then failed: IndexError mapping benchlot's replies to SVAMP
questions (probe_curve numbers rows by their line in the whole bench file, not within SVAMP); fixed (tested locally with the real
layout) and requeued as mm2, next after nt2.
nt2, original Qwen thinking OFF vs ON (records diff, fresh x8): bfcl irrelevance 80.9 -> 87.2 (+6.2; tokens 195 -> 8, 96% saved);
bfcl multiple 75.9 -> 58.8 (-17.2; 206 -> 20); bfcl simple 88.1 -> 76.6 (-11.6; 197 -> 24); ifeval 65.6 -> 60.6 (-5.0; 703 -> 284);
mbpp 46.2 -> 37.2 (-9.1; 2,098 -> 53); svamp 89.7 -> 17.5 (-72.2; 644 -> 10). rc1-rl-40 thinking OFF being scored.

**26 Oct ~00:00 (CET), runtime 1 results.**
nt2, rc1-rl-40 thinking OFF vs rc1-rl-40 thinking (fresh x8): bfcl irrelevance 79.7 -> 86.6 (+6.9; tokens 106 -> 5); bfcl
multiple 71.9 -> 63.4 (-8.4; 121 -> 20); bfcl simple 87.8 -> 78.8 (-9.1; 121 -> 24); ifeval 65.0 -> 55.3 (-9.7; 379 -> 229); mbpp
44.4 -> 33.1 (-11.2; 867 -> 45); svamp 85.6 -> 23.1 (-62.5; 224 -> 9). Same pattern as the original: thinking is essential for
math, worth ~9-11 points elsewhere, and hurts on BFCL irrelevance.
**drpo1 step 20 at half strength (weight 0.5 into rc1-rl-40), fresh x8: +59.8% / -3.7 (outside).** irrelevance +50% / -1.2;
multiple +43% / -5.6; simple +41% / -0.3; ifeval +53% / -4.7; mbpp +67% / -2.8; svamp +69% / -5.0. Halfway between rc1-rl-40
(53.9 / -2.1) and step 20 (61.7 / -4.9), roughly linear: no free lunch from halving.
mathmatch mm2 failed again (benchlot's generate-only rollouts carry output_ids, not a thinking count: now read from the position
of "</think>"; tested locally) and lot B5 failed for want of its file; both requeued (mm3; plotB5 with a changed line) after
stopB5, which is running.

**26 Oct ~00:30 (CET), sb1 (ShorterBetter reward, from rc1-rl-40) is collapsing (per-type batch logging, step 33 of 60).**
Thinking 472 -> ~95 tokens; batch accuracy 0.62 -> 0.27-0.42 at steps 28-33; per type at steps 31-33: gsm_ic 0.28-0.42 (0.80-0.96
at the start), gsm_plus 0.04-0.19 (0.42-0.75), kodcode 0.16-0.26 (0.33-0.65), toolace 0.07-0.60. beta 0.005 (5x the paper's,
chosen for our shorter replies) is far too strong. Also seen: steps 22-30 drew only gsm_ic / ifeval_like questions (the pool's
type mix is not even along the run). No guard was set on sb1; its 10/20/30 checkpoints are saved and are scored after step 60.

**26 Oct ~01:00 (CET), RL batch mix bug found and fixed (user: double-check the shuffling).** rl_train.py's balanced_order
interleaved the pool round-robin by source: every step got the SAME count of each source (not the evaluation-like mix the pool
was built with), and once the smaller sources ran out, the remaining steps held only the larger ones (sb1 steps 22-30: math and
instructions only). drpo1 and sb1 were trained this way (their later steps were skewed). Now each row of a source with n rows is
placed at (k + u) / n and the pool is sorted by position: every batch holds the sources in their pool proportions from the first
step to the last (checked on a 4,368-row pool, 72 drawn per step: steps 0, 20, 40, 59 each ~12/12/17/15/8/8 for gsm_ic /
gsm_plus / ifeval_like / toolace / hermes / kodcode). drpo3 and drpo2 (not started) use the fix.

**26 Oct ~01:45 (CET), sb1 (ShorterBetter reward, beta 0.005, from rc1-rl-40; stopped at step 33), fresh x8:**

| checkpoint | saved | accuracy | svamp | mbpp | ifeval | bfcl irr / mult / simple |
|---|---|---|---|---|---|---|
| step 10 | 58.6% | -3.3 | +67% / -5.3 | +70% / -2.8 | +47% / -1.9 | +50% / -1.2, +43% / -6.9, +42% / -1.6 |
| step 20 | 66.2% | -5.7 | +69% / -10.0 | +82% / -4.7 | +55% / -4.1 | +56% / -3.1, +48% / -6.2, +46% / -2.2 |
| step 30 | 78.2% | -11.2 | +83% / -28.4 | +90% / -10.0 | +68% / -4.4 | +64% / +0.9, +54% / -3.4, +54% / -3.1 |

Even step 10 is outside (-3.3): beta 0.005 overshoots within 10 steps; math and BFCL multiple pay first. Runtime 2 moved on to drpo3
(rl_train with math protected, guard 0.06, proportional batch mix). Runtime 1: stopB5 DPO at step 40/55 of seed 0 (pairs ordered
right 100%).

**26 Oct, queued (user): ShorterBetter rerun, sb2 (runtime 1, before bal0).** From rc1-rl-40 like sb1, with the paper's beta
0.001 (sb1 used 0.005, reasoning that our replies are shorter; but the penalty is per token and advantages are normalized per group,
so 5x was simply 5x the pressure), math protected (correctness-only reward on math), guard 0.06, proportional batch mix,
checkpoints every 10 steps scored. track.py drpo --rl-sb-beta passes the weight.

**26 Oct (user): more careful RL steps, evaluated more often.** RL runs started from now on save a checkpoint every 5 steps and
score steps 5, 10, 15, 20, 25, 30, 40, 50, 60 on fresh x8 (~7 min each, ~1 h per run): the first 30 steps, where drpo1 and sb1
overshot, are seen every 5 steps. sb2 also halves its learning rate (5e-6, as drpo2 / drpo3). drpo3 (already running) keeps its
10-step checkpoints, with the per-type guard.

**26 Oct ~03:00 (CET), interim.** stopB5 (original Qwen, stop pairs from lots B..B4 on code and tools only: 928 pairs; lot B3 has
no code/tool questions, 0 pairs), fresh x8: soup 10 **+0.2% / +0.4** (bfcl -2..+3%, mbpp -4% / +3.4, svamp -1%); soup 15 **+5.6% /
-0.9** (bfcl +4-8% / -0.6..-1.9, ifeval +11% / -2.2, mbpp -1% / +0.9, svamp +6% / -0.9). Barely shorter so far, and code (the
target) not shorter at all: soups 20-30 pending.
drpo3 (raw Qwen, math protected) steps 1-5: thinking 1,208 -> 974 tokens; batch accuracy 0.66-0.72; gsm_ic 0.51-0.65, gsm_plus
0.30-0.55, ifeval_like 0.85-0.95, kodcode 0.94-1.00 (8-16 replies per step), toolace 0.67-0.96; no guard stop. Its pool problem
(1,592 questions for 4,320 needed; math 54%, code 2%, Hermes 0%: the earlier-draw filter emptied the small sources, and the
contamination check's 13-word rule drops IFEval-like prompts that share IFEval's constraint wording) is reported; not fixed yet
(user decision pending).

**26 Oct ~03:15 (CET), drpo3 restarted as drpo3b with a proper pool (user: option A).** mixed_pool(reuse=...): runs from the RAW
model (drpo, teach) no longer drop questions drawn by earlier experiments (the raw model never trained on them; the 30% 8-gram /
13-word check against all 2,759 evaluation questions still applies). drpo3 had only 1,592 of the 4,320 questions it needed
(54% math, 2% code, no Hermes) and would have cycled them ~2.7x. Same settings otherwise (math protected, guard 0.06, lambda 0.5,
lr 5e-6, dynamic sampling, proportional batch mix). teach1's pool gets the same fix.

**drpo3b started (GPU 2, scripts 9c2886c).** Pool = 6,224 questions for 4,320 needed: toolace 1,604, gsm_plus 1,371, gsm_ic 1,371,
kodcode 707, ifeval_like 693, hermes_tools 478 (math 44%; the drpo3 pool had 54% math, 2% code and no Hermes). The contamination
check (a 30% 8-gram share or a shared 13-word run) dropped and listed its flagged rows. Training: raw Qwen, DRPO λ 0.5, lr 5e-6, math
protected, guard 0.06, a checkpoint every 5 steps.

**stopB5 final (GPU 1), fresh 240 x 8, cut / accuracy vs original Qwen:**

| soup | all | bfcl irrel. | bfcl multiple | bfcl simple | ifeval | mbpp | svamp |
|---|---|---|---|---|---|---|---|
| 10 | +0.2% / +0.4 | | | | | -4% / +3.4 | -1% |
| 15 | +5.6% / -0.9 | +4% / -1.9 | +8% / -1.6 | +5% / -0.6 | +11% / -2.2 | -1% / +0.9 | +6% / -0.9 |
| 20 | +11.9% / -1.2 | +14% / -2.5 | +14% / -1.9 | +14% / +0.6 | +15% / +0.6 | +11% / -1.3 | +8% / -2.8 |
| 25 | +40.7% / -2.2 | +27% / -0.9 | +22% / +0.9 | +24% / -0.9 | +31% / -0.6 | +69% / -5.6 | +23% / -2.2 |
| 30 | **+56.3% / -2.7** | +49% / +0.0 | +39% / -1.3 | +39% / -0.6 | +31% / -1.2 | +88% / -6.9 | +20% / -2.2 |

Reading: the pairs were code + tools only, yet every type got shorter (the stop habit spreads). Between steps 20 and 25 code
falls off a cliff (+11% -> +69% cut, -1.3 -> -5.6) and drives both the headline cut and most of the loss (mbpp holds most of the
tokens). Without code, soup 30 cuts tools 39-49% and IFEval 31% at 0 to -1.3. SVAMP loses about 2 points though math was never
trained. Next idea (not queued): soup 30 for everything except code, or the same run with code pairs capped. mm3 now running.

**26 Oct, queued (user): DATA V2 in place of stopB6 (GPU 1, after lot B5, before sb2).** Why: stopB5 trained on 928 pairs,
71% code (~660 KodCode, ~155 Hermes, ~110 ToolACE), all "stop earlier", no "keep thinking", no math or instructions; one epoch
ends near soup 30, where code collapsed. The labels were the bottleneck (up to 11 cuts x 20 forced answers per reply).
- **lot V2** (plotV2, original Qwen): ~3,500 NEW training questions: matched GSM8K (the 600 lot B5 left), KodCode 1,200,
  ToolACE 900, Hermes (the ~230 lot B4 left), IFEval-like 700; 8 replies each; **6 cuts x 8 forced answers** (~4.5x cheaper
  per reply; the half-split check still chooses on 4 answers and confirms on the other 4). Contamination check as always.
- **stopV2**: pairs from all original-Qwen lots (B, B2, B3, B4, B5, V2: ~5,400 questions). Code, tools and instructions get
  "stop here" where a cut is safe AND "keep thinking" pairs (full right reply beats its failing early cut; longest right beats
  shortest wrong); math gets "keep thinking" only. **Per-type caps**: kodcode 700, ifeval_like 700, toolace 400, hermes 300,
  gsm_ic 200, gsm_plus 200, gsm8k_matched 300 (max ~2,800; no type above 25%). 3 seeds, soups at 20/40/60/80/110/140/170 on
  fresh x8. Code: targeted_pairs lets one type get both kinds; track.py --lot-cuts/--lot-checks/--tg-min-samples/--tg-quota.

**mm3 (GPU 1): matched GSM8K built; original Qwen x8.** Choosing set SVAMP (40): accuracy 0.79, thinking 858 tokens, 50 words;
30 of 40 are one-operation. Matched GSM8K held-out math test (300, contamination 0 flagged): accuracy **0.87**, thinking
**761**, 40 words; but only 60 of 300 are one-operation (training part: 199 of 1,200). The match is close on thinking length,
but NOT on composition: GSM8K has few true one-step problems, so the corpus is mostly two-step, shorter in words, and easier
for the model (+8 points). Usable as a math guard/test set; it is not a copy of SVAMP's difficulty (SVAMP's twisted wording
is what costs the model). Lot B5 (600 of the training part x20) labelling started; its cut curve vs SVAMP's comes next.
drpo3b (GPU 2) steps 1-3: thinking 1,189 -> 1,061 -> 1,120; batch accuracy 0.67-0.68; math (protected) 1,800-2,900 tokens,
code 1,064-1,635, tools/instructions ~200-600; ~12 min per step.

**26 Oct, how the public competitors handle math (read from their repos).** Nobody keeps SVAMP intact: Lightning McQwen3 SVAMP
66% / -3.8, michal-bida 56% / -2.4, ours rc1-rl-40 65% / -4.1; their totals near -0.8 come from gains on the other parts.
- basta/lightning-mcqwen3: math = GSM8K + Skywork-OR1 (the challenge's suggested sets), no SVAMP matching, only answers SVAMP's
  grader parses. Scale: 2,500 questions per source x 4 traces, cuts at 10..90% re-answered 4x (197k cuts), 9,034 pairs
  (40% shorten, 40% guard, 20% hard), plus an NLL-anchored "be right" DPO on math only (own right vs own wrong trace,
  lengths within +-25%, 732 pairs).
- michal-bida/thinking-compression: math = GSM8K 70% + MetaMathQA GSM_SV 15% + GSM_FOBAR 15% (GSM8K problems rewritten
  backwards / with a verification twist, closer to SVAMP's perturbations), prompts drawn evenly from the deciles of SVAMP's
  prompt length, 8-gram check. Key fix: per domain, "shorter right" pairs capped at the number of "right beats wrong" pairs
  (math is right 85% of the time, so it got many shorten pairs and few wrong pairs: SVAMP -3.9 -> -2.4 after the cap).

**26 Oct, data v2 rescaled to the competitors' size (user) + MetaMathQA math + "be right" pairs (user).** Replaces the
3,500-question version above.
- **lot V2**: ~10,300 NEW training questions: math = matched GSM8K rest 600 + **MetaMathQA GSM_SV/FOBAR 2,500** (GSM8K-train
  rewrites: solve for x, or asked backwards from the answer; the twisted wordings closest to SVAMP; michal-bida's mix;
  prepare_types `metamath_sv`, same "Final answer:" wording), KodCode 3,000, ToolACE 2,000, Hermes rest (~230), IFEval-like
  2,000. 8 replies each; **8 cuts x 4 forced answers** (Lightning McQwen3's label budget; safe = right on both halves of the
  4). Estimated ~11-13 GPU-hours.
- **stopV2**: pairs from lots B..B5 + V2 (~12,000 questions). New pair kind **"right"** for math (targeted_pairs
  --right-sources): the question's own right reply beats its own wrong one within +-25% length, one per question
  (Lightning McQwen3's correctness DPO; our NLL anchor is on for every pair; dpo.py mixes 4 kinds). Caps: kodcode 1,800,
  ifeval_like 1,800, toolace 1,200, hermes 500, math 2,700 (gsm_ic 300, gsm_plus 300, matched 800, metamath 1,300): ~8,000
  pairs. 3 seeds, DPO stops at the last scored step (300); soups at 25/50/75/100/150/200/300 on fresh x8.
- Toy test: right pair built for close lengths, skipped when lengths differ > 25%, none for non-listed types.
Side note: data/gsm8k_matched (mm3) draws GSM8K train AND test; GSM-IC / GSM-Plus are built on GSM8K test, so models trained
on gsm_ic/gsm_plus (rc1, drpo*) may have seen paraphrases of some held-out matched-math test questions. SVAMP is unaffected.

**26 Oct ~00:45, lot B5 (matched GSM8K, original Qwen, 600 questions x 20, 0 contaminated): math TIMING now matches SVAMP.**
Success of the forced answer by share of thinking kept (0-20 / 20-40 / 40-60 / 60-80 / 80-<100 / full), perfect stop:

| source | curve | perfect stop: saved / gained |
|---|---|---|
| SVAMP (choosing set, benchlot) | 0.26 / 0.34 / 0.48 / 0.58 / 0.55 / 0.73 | 53% / +8.9 |
| **matched GSM8K (lot B5)** | 0.08 / 0.25 / 0.44 / 0.56 / 0.61 / 0.72 | 47% / +9.6 |
| GSM-IC (practice, lot B3) | 0.25 / 0.46 / 0.47 / 0.49 / 0.52 / 0.52 | 66% / +29.5 |
| GSM-Plus (practice, lot B3) | 0.21 / 0.47 / 0.47 / 0.46 / 0.50 / 0.47 | 71% / +31.8 |

Takeaway: our earlier practice math (GSM-IC/Plus) settles by 20-40% and so teaches early stops that SVAMP punishes; the
matched GSM8K builds up slowly and ends at SVAMP's level (0.72 vs 0.73), like SVAMP. It is the right math to teach stops on.
lot V2 started (scripts 639d770): 4,868 new questions of 8,178 drawn (3,310 dropped as earlier draws: lot B5's 600 matched
math plus ToolACE / IFEval-like overlaps with older pools); probe_curve's per-source cap (2,000) also trims MetaMathQA to
2,000, so ~4,370 non-code questions + KodCode 3,000 = **~7,400 new questions** (not 10,300). Still ~4x stopB5's base.
drpo3b steps 10-12: batch accuracy 0.74-0.77 (up from 0.67), thinking still flat (all 1,076-1,218; non-math 397-595;
code 866-1,426). Accuracy rises, length does not move yet.

**26 Oct, math now taught from SVAMP-like data everywhere it can be (user).**
- **stopV2**: math = matched GSM8K + MetaMathQA only (GSM-IC/Plus dropped: they settle by 20-40%, the wrong teacher). Each gets
  "keep thinking" + "be right" pairs, and ALSO "stop here" pairs on safe cuts IF its lots' curve builds up like SVAMP's:
  success at 20-40% of the thinking / success with the full thinking <= 0.6 (SVAMP 0.47; matched GSM8K 0.35 -> ON; GSM-IC
  0.88, GSM-Plus 1.0 would be OFF; MetaMathQA decided automatically from lot V2, printed in the log). track.py
  --tg-shorten-if-svamp / early_ratio() (toy-tested on the real lot B5 and B3 summary lines). Caps: gsm8k_matched 1,300,
  metamath_sv 1,300.
- **RL pools**: new --pool-math (34% math split evenly). teach1 (raw Qwen): matched GSM8K + MetaMathQA, both taught by own
  confidence. sb2 and drpo2 (from rc1-rl-40, which only draw unused questions; every matched question is now in a lot):
  MetaMathQA; sb2 protects it like the other math. drpo3b keeps its built pool (GSM-IC/Plus, protected).

**26 Oct ~00:40-01:50, both runtimes lost and relaunched (user approved; Drive access granted by the user).** Around 00:40
every Chrome tab got a new id and both Colab pages fell back to "Connect"; both Drive logs stopped (GPU 1 in lot V2 at 750 /
4,368 questions sampled; GPU 2 at drpo3b step 12). Relaunched ~01:20 on scripts 1b57c7c: queues replay and skip finished
stages from Drive. GPU 1 re-samples mm3's 300-question math test (its samples were not in the light restore; fixed in
f0ab9d7), then resumes lot V2 from its saved chunks. GPU 2 restarts drpo3b's training from step 0 (~15 h).

**26 Oct ~02:30, saving made resumable (user: "keep saving everything throughout").** Drive already got a copy every 5 min,
but the scripts restarted from zero after a lost runtime. Now:
- rl_train.py: adapter + optimizer + RNG + log saved every step (resume.pt, rl_log.jsonl); restarts continue after the last
  step; without that state, from the latest step-N checkpoint (fresh AdamW).
- check_vllm.py (cut labelling, the hours-long part of every lot): results appended per 2,000-trace chunk (.part); a restart
  skips finished chunks.
- dpo.py: reference scores cached (ref.json), latest checkpoint + optimizer saved at every checkpoint, resume skips done
  steps; it no longer writes the final adapter early (track.py took a half-trained seed for a finished one).
drpo3b restarted with it: "resuming after step 10 from step-10 (no optimizer state saved: fresh AdamW)" (steps 11-12 and the
log of steps 1-10 lost; its guard baseline restarts from step 11). Not yet resumable: teach_stop.py, evaluate.py (one scoring
run, < 1 h each).

**26 Oct ~04:30, drpo3b at step 16 (the step-15 check).** Steps 11-16: batch accuracy 0.70-0.77 (step 1: 0.67); all thinking
924-1,101 (step 1: 1,189); non-math 317-537 (step 1: 433: no drop); code 896-1,413 (step 1: 1,635, noisy). The run mostly
raises accuracy; it barely shortens anything but math's own drift. ~20 min per step. lot V2 (GPU 1): 2,400 / 4,368 non-code
questions answered.

**26 Oct ~05:00 (CET), drpo3b (raw Qwen, DRPO lambda 0.5, lr 5e-6, math protected) stopped by the guard at step 20.**
"guard: stopping at step 20, accuracy fell: gsm_ic 0.77 -> 0.68; ifeval_like 0.83 -> 0.76" (batch accuracy, steps 16-20 vs
11-15: the baseline restarted at step 11 after the resume). Fresh 240 x 8 vs original Qwen:
| checkpoint | saved | accuracy |
|---|---|---|
| step 5 | +3.5% | -0.2 |
| step 10 | +5.2% | +0.0 |
| step 15 | +8.0% | +1.3 (CI -0.6..+3.3) |
Step 15 per benchmark: svamp +18.4% / -0.3; mbpp +5.5% / +4.1; ifeval +3.7% / +2.2; bfcl irrelevance +6.5% / -0.3, multiple
-1.4% / +0.3, simple +5.7% / -2.8. Step 20 being scored. Reading: gentle DRPO from raw is safe and slightly raises accuracy
but saves very little in 15 steps (8%; rc1-rl-40 53.9%, rivals ~47%); the savings come mostly from math (wrong long loops
pushed down), not from the length pressure on the other types.
**drpo3b step 20 (the guard-stop checkpoint), fresh 240 x 8: +13.5% / -0.3 (CI -1.95..).** svamp +21.4% / -0.6; mbpp +10.2% /
-1.6; ifeval +12.0% / +1.3; bfcl irrelevance +13.7% / +0.3, multiple +8.5% / -1.3, simple +8.0% / +0.3. Steady, every type a
little shorter at no clear cost: 5 -> 20 steps went 3.5 -> 13.5% saved. The guard (noisy batch accuracy) may have stopped a run
that was still safe on the fresh set. GPU 2 moved on to teach1.
**26 Oct ~08:30, teach1 trained on nothing (bug, fixed).** teach_stop.py's judge loop reused the name `rows` for its mask
(`for kind, rows, target in ...`), overwriting the list of answers, so step 3 found 0 teacher answers ("teachers: own
confidence 0, judge 0, keep thinking 0"; p(</think>) nan): both teach1 adapters are untouched copies of the raw model and
their scores mean nothing. Fixed (loop variable renamed); queued teach1b on GPU 2 right after teach1, same settings, reusing
teach1's 2,512 answers and own-confidence readings (--teach-reuse teach1); the judge fit (~2 h of hidden states) runs again.
Before the bug the setup read: 19,466 paragraph ends; the soft judge predicts success 0.34 at cuts vs 0.62 at the full thinking.

**26 Oct ~08:45, queued (user): drpo3c = drpo3b continued** from models/drpo3b-20 (GPU 2, next), same gentle settings
(lambda 0.5, lr 5e-6, dynamic sampling, math protected, GSM-IC/Plus math as before), 40 more steps, every 5 steps scored on
fresh x8, guard loosened to 10 points (the 6-point batch guard fired on noisy per-type batches while fresh x8 still improved).
teach1's line commented (it trained on 0 answers; teach1b follows drpo3c). GPU 2's queue was interrupted at ~07:32 during
teach1-rel0.1's (meaningless) scoring and is restarted.

**26 Oct ~09:30, drpo3c stopped at step 2 (bad pool) and requeued as drpo3d.** drpo3c starts from a merged model, so its pool
removed every question any earlier experiment drew: 914 questions for 2,880 needed (ifeval 339, gsm_ic 393, gsm_plus 120,
toolace 59, KodCode 3, Hermes 0), the drpo3 problem again. New --pool-continues: a continued run may reuse earlier draws (it
never trained on lot questions), except the pools of the runs it continues (drpo3b's). Same settings otherwise.
lot V2 (GPU 1): contamination dropped 997 of 4,573 non-code questions, nearly all IFEval-like (their constraint sentences are
IFEval's own templates, so the 13-word run rule fires; the 8-gram shares listed are 6-25%, one 63%). 3,554 questions remain
(4,996 replies, 38,939 cuts x 4 forced answers) and are being labelled. Fewer instruction pairs for stopV2 as a result.

**26 Oct ~09:40, lot V2 labelled (GPU 1): 3,554 questions, 4,996 replies, 38,939 cuts x 4 (12,156 fully safe).**
Success by share of thinking kept (0-20 / 20-40 / 40-60 / 60-80 / 80-<100 / full), perfect stop:

| source | replies | curve | perfect stop |
|---|---|---|---|
| matched GSM8K | 780 | 0.08 / 0.21 / 0.40 / 0.57 / 0.63 / 0.75 | 49% / +6.6 |
| MetaMathQA SV/FOBAR | 2,918 | 0.06 / 0.09 / 0.19 / 0.43 / 0.64 / 0.69 | 39% / +16.3 |
| IFEval-like | 672 | 0.75 / 0.76 / 0.77 / 0.74 / 0.68 / 0.75 | 67% / +11.6 |
| KodCode | 293 | 0.41 / 0.45 / 0.44 / 0.45 / 0.49 / 0.50 | 81% / +20.6 |
| ToolACE | 333 | 0.35 / 0.35 / 0.47 / 0.42 / 0.37 / 0.52 | 49% / +8.6 |

Both math sets build up late like SVAMP (MetaMathQA even later); instructions are settled from the start; code barely
depends on thinking. The lot came out math-heavy: KodCode gave only ~180 questions and ToolACE ~200 (earlier-draw and
contamination filters), Hermes none (all used before) — to check before any later lot.
**stopV2 started (GPU 1)**: SVAMP-likeness gate: matched GSM8K 0.32, MetaMathQA 0.13 (bar 0.6) -> both get stop pairs. 6,021
pairs after caps (5,970 trained, 48 held out): math 2,600 (matched 1,300: shorten 421 / hard 509 / guard 176 / right 194;
MetaMathQA 1,300: shorten 377 / hard 533 / guard 169 / right 221), KodCode 1,500 (shorten 833 / hard 281 / guard 386),
IFEval-like 1,258 (shorten 836 / hard 123 / guard 299), ToolACE ~420, Hermes 243. 6.5x stopB5's 928, 40% brake/be-right pairs.
**drpo3d (GPU 2)**: pool 3,227 for 2,880 (ToolACE 816, GSM-IC 846, GSM-Plus 731, IFEval-like 476, KodCode 358; no Hermes left);
steps 1-2: batch accuracy 0.78 / 0.75, thinking 1,065 / 1,094.
~10:45: stopV2 seed 0 at DPO step 115/300 (pairs ordered right 62-76%, tripwire 0.000). drpo3d steps 1-6: batch accuracy
0.75-0.84; non-math thinking 401 -> 300; code 1,175 -> 767 (continuing from step 20, lengths now fall outside math too).
~13:15: **drpo3d stopped by the 10-point guard at step 23** ("toolace 0.87 -> 0.76", batch accuracy, steps 19-23 vs 1-5);
steps 7-23: batch accuracy 0.74-0.83, all thinking 818-1,162, code 557-1,910 (noisy). Its checkpoints 5/10/15/20 and the step-23
stop are now scored on fresh x8 (= drpo3b steps 25-43). stopV2: seed 1 done (300 steps, tripwire 0.000), seed 2 started.

**26 Oct ~14:30, drpo3d (drpo3b-20 continued, guard 10 pts) stopped by the guard at step 23:** "accuracy fell: toolace 0.87 ->
0.76" (batch; ToolACE 72-104 replies per step). Fresh 240 x 8 vs original Qwen, step 5 (= 25 RL steps in all): **+12.7% / +0.9**
(svamp +22.2% / +0.9; ifeval +13.8% / +1.9; mbpp +4.0% / +1.3; bfcl irrelevance +12.5% / -0.9, multiple +7.1% / 0.0, simple
+9.6% / -0.6). About drpo3b-20's level (+13.5 / -0.3) with better accuracy; steps 10/15/20/23 being scored.
drpo3d fresh x8: step 10 **+14.8% / -0.6** (svamp +24.8 / -1.9; mbpp +11.6 / -1.3; ifeval +10.1 / +0.9; bfcl irr +13.0 / -0.3,
mult +11.9 / +0.6, simple +9.9 / -1.3); step 15 **+18.8% / +0.3** (svamp +30.9 / -0.3; mbpp +17.6 / 0.0; ifeval +13.6 / +1.6;
bfcl irr +13.0 / -0.3, mult +12.1 / 0.0, simple +10.2 / +0.6). Savings growing again, accuracy flat, BFCL steady through step
15 (the guard's ToolACE drop is at batch steps 19-23). Steps 20/23 being scored.
**drpo3d fresh 240 x 8 (continuing drpo3b-20, which was +13.5% / -0.3):**

| checkpoint (total steps) | saved | accuracy | svamp | mbpp | ifeval | bfcl irr / mult / simple |
|---|---|---|---|---|---|---|
| 3d-5 (25) | +12.7% | +0.9 | +22% / +0.9 | +4% / +1.2 | +14% / +1.9 | +12/-0.9, +7/0.0, +10/-0.6 |
| 3d-10 (30) | +14.8% | -0.6 | +25% / -1.9 | +12% / -1.3 | +10% / +0.9 | +13/-0.3, +12/+0.6, +10/-1.2 |
| 3d-15 (35) | **+18.8%** | **+0.3** | +31% / -0.3 | +18% / 0.0 | +14% / +1.6 | +13/-0.3, +12/0.0, +10/+0.6 |

Steady: about +1% saved per step at no accuracy cost so far (35 steps: 18.8% at +0.3). Far from 50% at this pace, but the
cleanest curve of the project. 3d-20 and the step-23 guard stop are being scored.
drpo3d step 20: **+20.5% / -0.8** (svamp +29.6 / -2.5; mbpp +14.6 / +0.6; ifeval +22.7 / -0.6; bfcl irr +16.7 / -1.6, mult
+13.7 / -2.2, simple +12.3 / +1.3). BFCL pooled about -0.8 (the ToolACE batch drop shows a little on irrelevance/multiple);
SVAMP -2.5 (wide CI). Step 23 being scored.
| 3d-20 (40) | +20.5% | -0.8 | +30% / -2.5 | +15% / +0.6 | +23% / -0.6 | +17/-1.6, +14/-2.2, +12/+1.3 |
| **3d-23 (43, guard stop)** | **+21.2%** | **+0.2** | +30% / -0.9 | +21% / +1.9 | +17% / +0.9 | +17/-1.6, +18/-2.5, +14/+0.6 |

drpo3b + 3d, 43 steps from the raw Qwen: 21.2% saved at +0.2, still rising ~0.7%/step; BFCL multiple begins to cost (-2.2..-2.5).
GPU 2 moved on to teach1b. stopV2: all 3 seeds trained (300 steps each, tripwire 0.000); soups being scored.
**Queued (user): drpo3e** = drpo3d-23 continued (GPU 2, after teach1b): same settings, 40 steps, every 5 scored on fresh x8,
guard 10 points, pool may reuse earlier draws except drpo3b's and drpo3d's.

**26 Oct, BFCL multiple protected in drpo3e (user).** BFCL "multiple" = several functions offered, one call: the thinking
is where the model compares them. drpo3d: bfcl_multiple 0.0 / +0.6 / 0.0 at 3d-5..15, then -2.2 / -2.5 at 3d-20/23 (40
questions x 8, so within noise but two in a row), while bfcl_simple stayed +0.6..+1.3; the guard stopped drpo3d on ToolACE
(0.87 -> 0.76). rl_train.py: types are now logged and guarded per ToolACE category (toolace/simple, toolace/multiple,
toolace/irrelevance), and --protect-sources accepts source/category; drpo3e protects toolace/multiple (equal weight among
right replies: trained to be right, no push to be short).

**26 Oct, BFCL multiple error breakdown: drpo3d-23 vs original Qwen (fresh x8, 40 questions x 8 = 320 answers each).**
Right 243 -> 235 (-8 answers = -2.5 pts). Mean thinking 182 -> 145 tokens. Wrong answers by kind (base -> drpo3d-23): right
function + wrong arguments 28 -> 34; wrong function 7 -> 10; unparseable 10 -> 9; never solved by either 32 -> 32. The loss sits
in 7 questions (-15 answers) partly offset by 4 that improved (+5); on the 7, thinking fell from ~230 to ~170 tokens.
What the extra errors are (examples): an allowed value not checked against the function's list (mode "public" / "public
transport" instead of the listed "transit"; categories "vegetarian" instead of the listed "vegan"), and a dotted function name
shortened (soccer_stat(...) instead of soccer_stat.get_player_stats(...)). So it is not choosing the wrong tool for the job:
it is skipping the re-read of the function's exact name and allowed values, which the shorter thinking drops. Same risk on
BFCL simple (enum values), though simple did not drop. drpo3e's protection of toolace/multiple addresses the main case.
(scripts: scratchpad bfcl_errors.py; records from Drive runs/mvtf/f8-base and f8-drpo3d-23.)

**26 Oct ~17:15, stopV2 soup 25 is LONGER (-9.6% / 0.0; every benchmark -3 to -13.5%) and stopV3 is queued (user).**
Why: about 2,480 of stopV2's 6,021 pairs reward the longer reply ("hard": full right reply over a failing early cut, ~1,450;
"guard": the question's LONGEST right reply over its SHORTEST wrong one, ~1,030) against ~2,500 shorten pairs: the length
signals roughly cancel, and the easy one ("don't close </think> early") is learned first. The guard points the opposite way
from michal-bida's balanced DPO (shortest right over longest wrong) and was also applied to code and instructions, where lot
V2 shows thinking barely changes the outcome.
**stopV3** (GPU 1, right after stopV2; runs only if no stopV2 soup saved >= 35% at >= -1.0 pt, --tg-skip-if stopV2:0.35:1.0):
same lots, rule and caps as stopV2; guard flipped (targeted_pairs --guard-mode shorter: shortest right over the longest
wrong or unfinished reply, only when that one is longer) on code, tools and instructions; math keeps hard + be-right brakes
and its SVAMP-gated shorten pairs, no guard; hard pairs on code/tools/instructions capped at 0.25 x their shorten pairs
(--hard-ratio / --hard-cap-sources). Toy-tested: old default unchanged; flipped guard picks the unfinished 900-token reply
over a 100-token right one; the cap removes hard pairs above the ratio.

**stopV2 first soups (fresh 240 x 8): LONGER, not shorter.** soup 25: -9.6% saved (i.e. 9.6% more tokens) / +0.0; soup 50:
-8.6% / +1.0 (every type 2-19% longer; mbpp +2.2, bfcl multiple +1.6). The "keep thinking" side (hard: full right beats a
failing cut; guard: longest right beats shortest wrong; ~40% of the 6,021 pairs) outweighs the "stop here" side early on.
Soups 75-300 pending; if they stay longer, the brake share (or the guard pairs, which reward length itself) must be cut.

**26 Oct ~17:40, re-analysis of the stop line from scratch (user) and gold1 queued.**
- Every DPO stop run follows one curve set by pairs SEEN (batch 16, constant lr 2e-5), not by data size: ~30-35% saved at
  320-400 pairs, ~55% at 480 (stopB 554 pairs, B2 327, B3 1,058, B5 928), with code collapsing between 400 and 480 in each.
  More distinct pairs bought accuracy at equal saving (B3 soup 25 34.6/-0.4 vs B2 34.6/-2.1), not saving.
- The habit is global: B2 (no code pairs) cut code 37%; B5 (code + tools only) cut IFEval 31%, SVAMP 20%; wrong and
  never-solved answers shorten as much as right ones (wronglen).
- dpo.py anchors the CHOSEN reply with NLL weight 1.0: in stopB..B5 every chosen reply was a short cut (the anchor itself
  imitates shorter replies); in stopV2 53% of chosen replies are full-length or the question's longest right reply (hard,
  guard, right), so the anchor imitates long replies half the time. stopV2 kinds: shorten 2,835 (47%), hard 1,564, guard
  1,207, right 415. Soup 50 (~376 shorten pairs seen, where stopB3 was at ~30%) is still -8.6% / +1.0.
- Only shorten and hard pairs share a prefix up to a cut, so they are pure "stop / continue here" decisions; guard and
  right compare whole replies. The lots' perfect stop saves 39-81% per type and GAINS accuracy: the labels hold far more
  than one global dose uses.
stopV3 put on hold (it kept a whole-reply guard and the reply-imitating anchor). **gold1** (GPU 1 after stopV2):
stop_gold.py trains only p("</think>") at the lots' labelled paragraph ends of their own replies: target 1 where the cut's
forced answers are right at least as often as the reply's full thinking (and >= 25%), else 0; lots B..B5 + V2; GSM-IC/Plus,
MBPP and never-solved replies left out; per-type caps as stopV2; LoRA r16, lr 1e-4, 8 replies per step; a checkpoint every
100 steps, each scored on fresh x8 (dose chosen by score). Toy-tested with the real tokenizer: labels land on the right
paragraph-end tokens; a cut less right than finishing gets 0, cuts as right get 1; a never-solved reply is skipped.
stopV2 soup 75 (~560 shorten pairs seen): **-9.9% / +0.9** (svamp +5.5 / +1.3; mbpp -20.3 / +2.2; ifeval -12.7 / +0.9; bfcl irr
-10.9 / -0.6, mult -16.9 / 0.0, simple -14.2 / -1.6). Still longer past the dose where every shorten-only run had taken off:
the long-chosen pairs and their NLL anchor hold it; accuracy slightly up.
stopV2 soup 75: -9.9% / +0.9 (svamp +6% shorter, everything else 11-20% longer); soup 100: **-15.8% / +0.2** (mbpp 32% longer,
bfcl 13-19% longer, svamp flat). Getting longer with training: the brake pairs win. Soups 150/200/300 pending.

**26 Oct, queued (user): stopV3 = stopV2 rebalanced (GPU 1, next; stopV2's soups 150-300 skipped).** Same lots, gate and
caps; guard pairs dropped (longest right beats shortest wrong: rewards length itself); "keep thinking" (hard) pairs capped at
20% of each type; stop-here and be-right pairs unchanged. track.py --tg-drop-kinds / --tg-brake-share. 3 seeds, soups
25-300 on fresh x8.
**26 Oct ~17:30, teach1b crashed: CUDA out of memory** (GPU 2, variant rel:0.05 at 1,800/2,512 answers, on a long answer;
progress until then: gsm8k_matched p(stop) 0.53-0.59 where it should stop vs 0.23-0.30 where it should keep, "good stop"
0.61-0.73). The queue moved on to drpo3e (running). Fixes in teach_stop.py and stop_gold.py: the forward pass ends at the
last scored paragraph end (causal, nothing after it is needed) and gradient checkpointing is on; teach_stop caches its
teacher points (teacher_points.npz) so a rerun skips the ~2 h judge fit. Requeued as teach1c on GPU 1 after gold1.

**26 Oct, queued (user): mf1 = repair rc1-rl-40's math (GPU 2, after teach1b, before drpo3e).** rc1-rl-40 is still the best
model (53.9% / -2.1); its loss is mostly math (SVAMP 65% shorter / -4.1). New stage `mathfix`: rc1-rl-40 answers the
SVAMP-like math of lots B5 + V2 (matched GSM8K ~1,200 + MetaMathQA ~2,500) x8, contamination check rerun; pairs from its own
answers only: "be right" (right beats wrong within +-25% length) + "keep thinking" (longest right beats shortest wrong when
longer); no pairs for any other type; then the targeted DPO (fresh LoRA on rc1-rl-40, NLL anchor, 3 seeds, soups 25-200 on
fresh x8). No cut labelling. Goal: SVAMP back to ~-1 with the other types still short (~50% at ~-1). Dry-tested with stand-in
data (filters math rows, builds both pair kinds, hands over to targeted()).

**26 Oct ~17:00: teach1b crashed (CUDA out of memory in teach_stop.py) and GPU 2 jumped to drpo3e before mf1 was read.**
teach1b parked (needs a memory fix); GPU 2 restarted so mf1 runs first; drpo3e resumes afterwards from its saved step.
stopV3 pairs: 5,060 (no guard pairs; keep-thinking 20% per type: matched GSM8K shorten 722 / hard 254 / right 324;
MetaMathQA 655 / 276 / 369; KodCode 833 / 208; IFEval-like 836 / 123; ToolACE 212 / 53; Hermes 156 / 39); seed 0 at step
65/300 (pairs ordered right 78-81%).

### 2026-10-26 ~20:40 — mf1 pairs built, DPO under way; stopV3 seed 1 nearly done
- mf1 (GPU 2): rc1-rl-40 answered 3,700 math questions x8. Contamination check: 0 of 3,700 flagged.
  2,850 pairs: gsm8k_matched guard 397 / right 413; metamath_sv guard 995 / right 1,045 (no hard pairs: mathfix has no cut labels).
  DPO mix guard 0.49 / right 0.51, 176 steps per seed (one pass). Seed 0 done (pairs ordered right 79–92% late, tripwire 0); seed 1 started.
- stopV3 (GPU 1): seed 1 at step 245/300 (pairs ordered right ~80–84%); seed 2 and the soups next.

**26 Oct ~22:30, density map + budget-prompt test queued (user).** Density map (artifact "ThinkingCap Density Map"): all 71
fresh-x8 results.json files read from Drive (one question-set fingerprint; recomputed scores match the logged headlines).
Noise: an untouched copy of the original vs the original: SVAMP -42 thinking tokens / -0.3 pts, MBPP+ -1.5 pts. Cheapest
model within 1 pt of the original, per benchmark (thinking tokens): BFCL pooled 67 (drpo1-60) vs 174; MBPP+ 742 (rep4-blend50)
vs 2,050; IFEval 193 (bon-dpo-r2g step 10) vs 454; SVAMP 405 (clean-b1 step 20) vs 635. Best-of-71 on 40 questions each:
rough, a little optimistic. IFEval's reply is 284 tokens even with thinking off (703 with), so it cannot be cut as deep.
**bp1** (GPU 2 after mf1): original Qwen with "Think briefly: keep your thinking under N tokens" appended to the system
message (svamp 400, mbpp 750, ifeval 200, bfcl 70), fresh x8, vs the original; thinking scanned for budget words
(tokens/budget/limit/brief/quick/concise/short) against the same scan without the prompt. Measure only.

**26 Oct ~22:50, benchlot's "perfect stop" recomputed strictly (user: does it mean right once or right 95% of the time?).**
The original figure (label_cuts.py summary) is the earliest cut whose 20 forced answers are right at least as often as the
full thinking's 20, over the 2 labelled replies per question (longest right + longest wrong), averaged per reply; its "accuracy
gained" is the BEST cut anywhere minus the full thinking (hindsight). Wrong replies whose full thinking scores 0/20 count as
cuttable at the first paragraph, which inflates "saved". Recomputed from pairs-bench40/cal (original reproduced exactly:
53 / 55 / 84 / 51%):
| rule | SVAMP | IFEval | MBPP+ | BFCL |
|---|---|---|---|---|
| original rule, all replies | 53% / +5.0 | 55% / +9.5 | 84% / +13.1 | 51% / +8.2 |
| right replies only, as right as finishing (= 20/20 here) | 41% / 0.0 | 49% / +1.7 | 72% / 0.0 | 49% / +0.4 |
| right replies only, cut right >= 19/20 | 47% / -1.0 | 49% / +0.7 | 74% / -0.6 | 49% / +0.4 |
| right replies only, as right as finishing on both halves | 41% / 0.0 | 45% / +1.4 | 72% / 0.0 | 49% / +0.4 |
(thinking cut / forced-answer accuracy at that stop minus finishing, points; replies: SVAMP 36, IFEval 29, MBPP+ 26, BFCL 109
right). Oracle still: the cut is chosen with the labels. Strict stop-only ceiling on the original: ~41-47% math, ~45-49%
instructions, ~72-74% code, ~49% tools.

### 2026-10-26 ~22:25 — mf1 soup 25 (rc1-rl-40 + matched-math repair), fresh 240 x 8: **+45.6% / -1.4 (within the limit)**
- svamp +50% / -0.3 (rc1-rl-40: +65% / -4.1); mbpp +47% / -2.5; ifeval +47% / -1.9; bfcl irr +42% / -0.9, multiple +38% / -3.1, simple +35% / +0.9.
- The math repair worked on SVAMP (accuracy loss -4.1 -> -0.3), but SVAMP gave back 15 points of saving, and the overall saving fell 53.9 -> 45.6 for +0.7 pt accuracy.
  Between the two rivals (Lightning 43.2 / -0.85, michal 47.0 / -0.78) on saving, still behind on accuracy. Soups 50-175 next.
- stopV3 (GPU 1): seed 2 at step 245/300.

**26 Oct ~23:40, does the model know EARLY that its answer will fail? (user; CPU analysis of teach1's saved data.)**
teach1's 2,506 answers (original Qwen, one per question; 21,978 paragraph-end own-confidence readings, own_low5 = geometric
mean of the 5 least likely tokens of a greedy forced answer), matched to each answer's final outcome. AUC of "this answer will
END wrong" from LOW own confidence (0.5 = no signal), by where it is read:
| type (right/wrong) | 1st paragraph | 2nd | by 25% of the thinking | by 50% | at the end |
|---|---|---|---|---|---|
| matched GSM8K (541/79) | 0.48 | 0.50 | 0.38 | 0.46 | 0.64 |
| MetaMathQA (476/87) | 0.55 | 0.51 | 0.54 | 0.52 | 0.76 |
| KodCode (213/99) | 0.70 | 0.70 | 0.68 | 0.66 | 0.60 |
| IFEval-like (192/50) | 0.57 | 0.60 | 0.63 | 0.58 | 0.60 |
| ToolACE (260/225) | 0.67 | 0.62 | 0.64 | 0.65 | 0.50 |
| Hermes (130/154) | 0.67 | 0.80 | 0.68 | 0.69 | 0.57 |
Practical test: a "give up" threshold at the 2nd paragraph that stops at most 5% of RIGHT answers catches only 7-19% of the
WRONG ones (code 19%, Hermes 14%, math 8-9%, IFEval 8%, ToolACE 7%). Reading: own confidence does not reveal early that an answer
will fail (math: no signal before the end); code and tools carry a weak early signal. A hidden-state probe trained on "ends
wrong" was not tried (the earlier probes predicted "answering now is right", a different question).
**mf1 soup 50, fresh 240 x 8: +32.2% / -0.5** (svamp +30% / -0.3; mbpp +29% / +2.5; ifeval +35% / -2.8; bfcl irr +37% / -0.3,
mult +34% / -3.8, simple +32% / 0.0). From soup 25 (+45.6 / -1.4) the saving falls 13 points on every type, not only math:
the math-only "be right" + keep-thinking pairs (chosen = full right replies, NLL anchor on) lengthen every type, as the
long-chosen pairs did in stopV2. Accuracy +0.9 overall, mostly MBPP+ (+2.5); IFEval and BFCL multiple stay down.

### 2026-10-27 ~00:00 — mf1 soups 50/75 and stopV3 soup 25 (fresh 240 x 8)
- mf1-soup-50: +32.2% / -0.5 (svamp +30% / -0.3, mbpp +29% / +2.5, ifeval +35% / -2.8, bfcl irr +37% / -0.3, mult +34% / -3.7, simple +32% / 0.0).
- mf1-soup-75: +33.0% / -1.3 (svamp +24% / -0.6, ifeval +42% / -3.7, mult +35% / -3.7).
  Reading: more mf1 training only makes every type think longer (saving 53.9 -> 45.6 -> 32-33) without buying accuracy back
  (SVAMP stays at -0.3/-0.6 from soup 25 on; IFEval and BFCL multiple get worse). The "longest right beats shortest wrong"
  guard pairs teach length to all types, not just math. Best mf1 point = soup 25 (+45.6 / -1.4); not the goal.
- stopV3-soup-25 (from the original Qwen): +13.4% / -0.7 (svamp +18% / -0.6, mbpp +12% / +0.6, ifeval +14% / -0.9, bfcl irr +7% / -2.8,
  mult +11% / -2.2, simple +10% / -0.3). Unlike stopV2 (replies got longer), dropping guard pairs and capping keep-thinking pairs gives a
  real if small saving. Soups 50-300 next.
**27 Oct ~00:15, drpo3e replaced by drpo3f (user):** the same continuation of drpo3d-23 (21.2% / +0.2) but lambda 0.3 instead
of 0.5; math and toolace/multiple protected, guard 10 pts, 40 steps, every 5 scored on fresh x8 (GPU 2, after mf1's soups and
bp1). stopB3's pairs by type (from pairs-stopB3): kodcode 359, gsm_ic 248, gsm_plus 130, ifeval_like 166, commonsense_qa 80,
toolace 60, arc_easy 15 (1,058, all shorten); stopB5 ~71% code.
**27 Oct ~00:40, sp1 queued (GPU 1, after teach1c, before sb2; user):** one big self-preference round from the original
Qwen (michal-bida's recipe: own answers, shortest right > longest wrong and > much longer right, shorten capped at guard per
type) but 12,000 questions x 4 (vs our A1's 438 pairs) and math + toolace/multiple PROTECTED: balanced_pairs.py --protect gives
them only similar-length (+-25%) right > wrong pairs, no length push. Pool: MetaMathQA 25%, IFEval-like 25%, KodCode 25%,
ToolACE 17%, Hermes 8%. balround gained --bal-mix / --bal-kod / --bal-protect; toy-tested (protected math gets 300-vs-320
right > wrong, not 300-vs-900 shorten; code keeps guard + shorten; toolace/multiple protected by category).

### 2026-10-27 ~01:00 — Drive cleanup (user: free ~40 GB; record everything first)
**State before:** personal Drive 408.05 GB used (Drive 125.9 GB, Photos 270.2 GB, Gmail 11.6 GB). thinkingcap-mvtf/models/
holds ~62 merged models at ~1.13 GB each (model.safetensors 1.11 GB + tokenizer files). Every scored model's evaluation stays
on Drive untouched (runs/mvtf/f8-*/results.json, records.jsonl, compare.json): only merged weights are removed, never results,
adapters, lots, rollouts or data.
**Why safe:** a merged model = base + LoRA adapter(s) (merge.py); the adapters stay in thinkingcap-mvtf/adapters/ (a few MB
each), so each one is rebuilt in about a minute:
- `<run>-soup-<N>` (targeted / clean / balround / mathfix stages): `python merge.py --model <base> --out models/<run>-soup-<N>
  --adapters adapters/<run>-s0/step-<N>:0.333333 adapters/<run>-s1/step-<N>:0.333333 adapters/<run>-s2/step-<N>:0.333333`
  (base: Qwen/Qwen3-0.6B for stop*; the parent model for chained rounds: ca1 from the clean-a2 line, ca2 from ca1, see the
  24 Oct entries "Line A (runtime 2): ca1 from clean-a2" / "ca2").
- `drpo1-<N>`: `python merge.py --model models/rc1-rl-40 --out models/drpo1-<N> --adapters adapters/drpo1-drpo/step-<N>:1`
  (drpo1-20x0.5: weight 0.5).
- `rc1-rl-<N>`: rlcal: `--model models/rc1-soup-30` (the chosen cal soup) `--adapters adapters/rc1-rl/<checkpoint N>:1`.
**Only models made before both runtimes were replaced (26 Oct ~00:40) are deleted:** the runtimes copy their local models/ back
to Drive after every stage, so anything a running runtime still holds would be re-uploaded. Kept: every chain start and queue
start (ca1-soup-10, rc1-soup-30, rc1-rl-40, drpo3b-20, drpo3d-23, mf1-soup-25), every model made or restored since 26 Oct 00:40
(stopB5, stopV2, stopV3, drpo3b/3d, mf1, teach1), and every family whose adapters could not be confirmed on Drive (stopB,
stopB2, rc2, rr1, sb1: Drive search did not show stopB-s0 / rc2-s0 / sb1-drpo; to verify before any later deletion).
**Moved to the Drive trash (19 models, ~21.5 GB; adapters confirmed on Drive: ca1-s0..2, ca2-s0..2, drpo1-drpo, rc1-rl,
rc1-s0, stopB3-s0), with their fresh-x8 scores where scored:**
ca1-soup-20, ca1-soup-30, ca1-soup-40, ca2-soup-10, ca2-soup-30, ca2-soup-40 (choosing-set scores only, see 23-24 Oct);
ca2-soup-20 46.4% / -2.73; drpo1-20 61.7% / -4.92; drpo1-20x0.5 59.8% / -3.72; drpo1-40 73.3% / -6.90; drpo1-60 78.7% / -5.99;
rc1-rl-20 54.8% / -3.67; rc1-rl-60 49.6% / -3.39; rc1-soup-10 48.6% / -3.02; rc1-soup-20 54.8% / -4.01;
stopB3-soup-15 18.3% / -1.04; stopB3-soup-20 27.1% / -2.03; stopB3-soup-30 54.8% / -3.20; stopB3-soup-40 69.1% / -5.71.
Emptying the trash (permanent) is left to the user. Not deleted, for a later pass: 12 models of the unverified families
(~13.5 GB) and the post-00:40 models (~20 GB) once their runtimes are released.
**Cleanup done by a runtime stage instead (user):** scripted clicks are ignored by Drive and the Chrome extension was not
connected, so `dropmodels --name drop1` (GPU 2, after mf1's soups, before bp1) removes the models through the Drive mount
(to the Drive trash). It checks each model's adapter folder on Drive first and KEEPS the model if it is missing. List (31):
the 19 above, plus stopB-soup-10/20/30 (adapter stopB-s0, later found on Drive), stopB2-soup-20 (stopB2-s0), rc2-soup-10/20/30/60
(rc2-s0), rr1-rl-20 (rr1-rl), sb1-10/20/30 (sb1-drpo); the last four families are removed only if their adapters are found.
Extra recipes: sb1-<N> = merge(models/rc1-rl-40, adapters/sb1-drpo/step-<N>); rc2-soup-<N> = merge(models/rc1-rl-40,
adapters/rc2-s0..2/step-<N>, 1/3 each); rr1-rl-<N> as rc1-rl (rlcal). Fresh-x8 scores of these: stopB soups 10/20/30 11.9/+0.9,
31.1/-2.3, 54.2/-3.6; stopB2-20 34.6/-2.1; rc2 soups 10/20/30/60 in density_points (63-64% / -4.9 for 20/30); sb1 10/20/30
58.6/-3.3, 66.2/-5.7, 78.2/-11.2. Expected: up to ~35 GB freed if every adapter is found.

### 2026-10-27 ~01:30 — mf1 soups 100/150, stopV3 soups 50/75 (fresh 240 x 8)
- mf1-soup-100: +31.4% / -0.5 (svamp +23% / +1.3, ifeval +43% / -5.0, mult +35% / -4.1, mbpp +24% / +2.5).
- mf1-soup-150: +29.9% / -0.3 (svamp +24% / +1.3, ifeval +38% / -1.9, mult +34% / -3.7).
  mf1 settles at ~30% saved / ~-0.4: the math repair gives back ~24 points of rc1-rl-40's saving for ~1.7 pts accuracy. SVAMP is now
  above the original (+1.3) but BFCL multiple stays ~-4. Not the goal; soup 25 (+45.6 / -1.4) remains mf1's best trade.
- stopV3-soup-50: +16.4% / -1.4; stopV3-soup-75: +11.8% / +0.2. The stop line from the original Qwen plateaus at 12-16% saved.
**27 Oct ~01:35, bp1 failed (my bug): "invalid choice: 'budgetprompt'"** — another session had added "mathfix" to the stage
list at the same place, so my edits adding "budgetprompt" and "dropmodels" did not apply (no assert). Fixed with an assert and a
check that every stage function is in the list. drop1 moved to GPU 1 right after stopV3 (GPU 2 is on drpo3f for ~13 h); bp1
requeued as bp1b on GPU 1 after gold1.

### 2026-10-27 ~02:00 — ps1 queued: prompt search before prompt distillation (user)
- Lightning McQwen3's first step (distil a terse prompt into the weights) is the one piece of his recipe we never tried.
  Before distilling, ps1 measures 11 prompts on the ORIGINAL model: no prompt, concise, prefill, no-re-check (+prefill),
  token budget 100, Chain of Draft, his examples / ex-draft (+prefill), and ex-draft with one example per task type
  (math, code, tool call, instructions). 50 training questions per type of lot V2 x 4 replies each; tokens saved,
  accuracy change, leak rate, all paired against no prompt. Training questions only (choosing on the benchmarks would
  fit to them). GPU 1, after stopV3, before gold1. prompt_search.py, prompt_candidates.py, track.py promptsearch.
- Public competitors re-checked: 4 forks of ThinkingCap-Test; only basta (Lightning McQwen3) and michal-bida have work,
  miroslavmyrha and horychtom have no commits. Submissions go to BottleCap by email (git bundle), so most are private.
  BottleCap publishes no 0.6B score of its own; the README's guide is > 20% saved at <= 0.2 pt lost per 1% saved.

### 2026-10-27 ~02:40 — GPU 2 moved on; bp1 failed at start
- mf1 finished (one epoch = 176 steps, so no soup 200): best trade stays soup 25 (+45.6 / -1.4); later soups ~30% / -0.4.
- bp1 (budget prompt, GPU 2) failed at once: "invalid choice: 'budgetprompt'", the runtime's track.py was older than the plan
  line when the stage started. bp1b on GPU 1 measures the same thing, so it was not requeued. GPU 2 now runs drpo3f
  (from drpo3d-23, toolace/multiple protected, 40 steps scored every 5).
- GPU 1: stopV3 soups 100-300 scoring (25/50/75 = +13.4/-0.7, +16.4/-1.4, +11.8/+0.2); then drop1, ps1, gold1, bp1b, teach1c, sp1.

### 2026-10-27 — literature on building pairs for shorter thinking (research pass, for the "many pair strategies" step)
Strategies to test offline on one fixed set of graded answers, best supported first:
1. DAST (2503.04472): add "longer WRONG beats shorter wrong" pairs on top of "shorter right beats longer right"; the
   wrong-longer pairs are the accuracy guard (correct-shorter only: 90.0 / -60%; both: 93.6 / -18%; R1-Distill-7B MATH500).
   SimPO, 1 epoch, max-margin pair per kind per question, bottom ~15% margins dropped. Opposite of michal-bida's "longest wrong".
2. First-correct + one check (Chen et al. 2412.21187): chosen = the reply cut after its first right answer plus one
   re-check, rejected = longest; QwQ-32B MATH500 92.8 at -45% (shortest reply as chosen: -22%). BFCL analogue: ThinkBrake
   (2510.00546) stop at the first right call. Code: cut at the first draft that passes.
3. Shortest right PER QUESTION, drop never-solved questions (Munkhbat et al. 2502.20122): ~30% shorter on GSM8K.
4. Ada-R1 (2504.21659): shortest right vs the 2 longest of any outcome; DPO beta 0.05-0.1, lr 5e-7, 2 epochs; -51% at -1.65.
5. Difficulty by solve rate p: long-vs-short right pairs only when p >= ~0.5; hard questions get right-vs-wrong and
   wrong-longer pairs only (DAST budget, Difficulty-Aware DLER 2510.15110, Arora & Zanette 2502.04463).
6. Drop near-identical pairs (likelihood displacement, 2410.08847): similar-length right-vs-wrong pairs are at risk.
Loss: DPO kept most accuracy, SimPO compressed most; NLL anchor did not help length DPO in Chen et al. (helps right-vs-wrong
only). Free dial: merge / interpolate the trained adapter with the base (DLER: top-25% updates x0.7). RL beats offline pairs
only at aggressive compression (DLER, AdaptThink, LC-R1); offline DPO comes close otherwise (Arora & Zanette).
Per type (ThinkingCap-27B blog): instructions and tools cheap, code safe but small saving, math costly.

### 2026-10-27 — dp1 queued: build the pool, then prove it behaves like the evaluation (user)
sp1 paused; data first. buildpool (GPU 1, after ps1): build_pool.py makes ~12,000 questions (3,000 per benchmark;
BFCL split 400/200/240 like the evaluation) in each part's own prompt format, with prompt lengths drawn bin by bin from
the CHOOSING set's length spread, math 70% matched GSM8K + 30% MetaMathQA, and an 8-word contamination check against all
evaluation questions (sequences found in >= 3 evaluation questions are templates and exempt). pool_check.py then has the
original model answer 200 pool questions per part x4 and compares, part by part, accuracy, median reply/thinking tokens and
capped share with its v4-base run on the choosing set (tolerance +-5 pts, +-15%, +-3 pts, or inside the choosing set's
own interval). Only once every part reads OK do we generate answers for the whole pool and try pair strategies.
Not yet in the check: the answer-timing match (success at 20-40% of the thinking vs full), which needs cut labels.

### 2026-10-27 — PRIORITY (user): the final model, LUPIN
- Step 0: dataset proven representative (dp1: buildpool, choosing set only).
- Step 1: the original model answers the whole pool; optionally also under the best shortening prompt from ps1
  (prompt removed afterwards = prompt distillation).
- Step 2: several pair strategies from the same answers -> best = LUPIN0; variants LUPIN0.1, 0.2, ... by pair type.
- Step 3: LUPIN1.0 = LUPIN0 + our guarded online RL (drpo).
- Idea to test as a variant (user): "backward" practice, answer -> question (like reciting backwards to memorize).
  Related work: the reversal curse (Berglund et al. 2023: "A is B" training does not teach "B is A"); RevThink (Chen et
  al. 2024, "Reverse Thinking Makes LLMs Stronger Reasoners": training on forward reasoning + the backward question +
  backward reasoning raised reasoning accuracy); FOBAR / MetaMathQA GSM_FOBAR (already in our math: questions asked
  backwards from the answer). None measured thinking LENGTH; to test as an extra NLL task mixed into LUPIN0's DPO.

### 2026-10-27 — RevThink checked (arXiv 2411.19865, NAACL 2025; Chen et al., UNC + Google)
- Method: a teacher (Gemini-1.5-Pro) writes per question: forward reasoning Rf (kept only if right), a backward question Qb
  (from Q + gold answer), backward reasoning Rb; consistency-checked. Student = 3-task SFT, equal weights 1/3:
  Q -> Rf, Q -> Qb, Qb -> Rb. LoRA r32; 3 epochs (math), 10 (others). At TEST time: forward reasoning only.
- Students Mistral-7B / Gemma-7B; 12 datasets (no SVAMP). +13.5% avg over zero-shot, +6.8% over the best distillation
  baseline. Mistral-7B GSM8K 54.7 -> 60.9 (distillation 56.2), MATH 10.4 -> 15.3. 10% of the data beats plain fine-tuning
  on all of it (StrategyQA). Ablation: all three tasks best; Q -> Qb added to Q -> Rf already helps; Qb -> Rb alone hurts.
- Length: "comparable number of tokens ... at test time" (no numbers). So: an ACCURACY method, not a shortening one.
- For LUPIN: no teacher needed for math, MetaMathQA's GSM_FOBAR / GSM_SV rows ARE backward rewrites of GSM8K train
  questions (Q, Qb pairs ready-made); Rb = our model's own right answer to the rewrite. Use = an accuracy guard mixed into
  the shortening DPO (an extra NLL task), aimed at the math loss; not expected to shorten by itself.

### 2026-10-27 — ALP checked (arXiv 2506.05256, "Just Enough Thinking", Stanford/SynthLabs, preprint)
- RL (GRPO), math only, DeepScaleR-1.5B: reward = right - beta * length * max(solve rate p, 1/K), p from the question's own
  K rollouts, penalty on ALL rollouts (wrong ones too). Easy questions pay more for length, hard ones keep thinking.
- Table 1 (64 samples, 4,096 budget): MATH-500 0.81 -> 0.80 at -72% tokens; AIME 0.22 -> 0.24 at -42%; OlympiadBench
  0.47 -> 0.51 at -36%. beta 1e-7, 100 steps, batch 512. No code released; no 0.6B, no code/tools/instructions.
- For LUPIN: (1) step 2 pairs: allow "shorter right" pairs only when p is high, the offline version of the same idea
  (DAST / Difficulty-Aware DLER agree); (2) step 3 RL (LUPIN1.0): scale our drpo length penalty by the solve rate p instead of
  a flat lambda. Our drpo penalizes length only among right answers (Arora-Zanette style); ALP also pushes wrong answers
  shorter; compare both.

### 2026-10-27 — "backwards" research pass (user's idea: answer -> question for a stronger link, fewer tokens)
- No paper shows answer -> question training SHORTENS reasoning. The reversal curse is about facts stored in the weights;
  the question is in the prompt here, where models already infer "B is A" (Berglund et al.). Reverse Training (2403.13799)
  fixes stored-fact reversal (0% -> 99%) but that is a different mechanism.
- Math backward data helps ACCURACY: MetaMath SV + FOBAR rewrites GSM8K 60.6 -> 64.4 (LLaMA-2-7B, Table 3), no length data.
  Forward DPO (right vs wrong) beats backward DPO (PASS/FAIL verdicts): GSM8K +3.5 vs +0.5 (2601.07199, Llama-3.1-8B).
  CoThink (2505.22017) even blames backward / re-check training for verbosity (5-question case study).
- The one SHORTENING result nearby: VeriThinker (2505.17941, NeurIPS 2025): LoRA trained only on "is this solution right?
  Yes/No" from labelled answers, no teacher: R1-Distill-7B MATH500 3,790 -> 2,125 tokens at 94.0 -> 94.8; fewer "Wait"
  after right partial work. Did NOT work at 1.5B (forgetting).
- For LUPIN (variants, ranked): (1) a Yes/No verdict NLL term on our own graded answers mixed into LUPIN0's DPO, low LoRA rank,
  watched for forgetting; (2) SVAMP backward questions (mask a number, give the answer, recover the number) as an accuracy
  guard, measuring tokens; (3) backward check only to choose the "chosen" answer. Lowest: answer-first formats.

### 2026-10-27 — LUPIN_SPEC.md written (user): full plan from data to RL
Step 0 dataset + verification (adds parse-rate and answer-timing checks), Step 1 answers (8 per question; optional
prompted set and cut labels), Step 2 Stage 1 (pair kinds; recipes R1-R9; methods M1-M11 incl. checker practice,
backward questions, SimPO, KTO, listwise, prompt-distillation round, stop tokens, iterative round, curriculum; dials:
soups, strength blend, per-part adapters; screen on choosing x4 -> confirm on fresh x8), Step 3 Stage 2 RL (RL1-RL11:
DRPO, solve-scaled DRPO, ALP, budget reward, ShorterBetter, DLER, progressive cap, stop reward, checker reward, online DPO,
per-part lambda). Built today: lupin_pairs.py (R1-R5 + difficulty rule), lupin_checker.py + dpo.py --checker (M3),
rl_train.py --drpo-solve-scale (RL2).

### 2026-10-27 ~02:55 — both runtimes restarted ~02:11 (cause not seen here); the resume work held
- Both notebooks show setup + queue cells started 02:11-02:12 on fresh machines (setup rebuilt data/train and questions).
- GPU 2: drpo3f resumed "after step 2 (adapter, optimizer and log restored)", now sampling step 3.
- GPU 1: finished stages skipped via the light restore; stopV3 soups 25/50/75 kept; now scoring soup 100 on fresh x8.
  Next on GPU 1: drop1, ps1, dp1, gold1, bp1b, teach1c, sb2.
- stopV3-soup-100 (fresh 240 x 8): +10.0% / -0.6. The stop line from the original stays at 10-16% saved (soups 25-100); drpo3f resumed after step 2 and is at step 5.
- stopV3 soups 150 / 200: +12.4% / +0.5, +16.2% / -0.9 (soup 300 scoring). Flat at 10-16% saved through soup 200.

### 2026-10-27 ~04:00 — stopV3 soups 100/150/200 (fresh 240 x 8)
- soup 100: +10.0% / -0.6; soup 150: +12.4% / +0.5; soup 200: +16.2% / -0.9 (mbpp +32% / 0.0, ifeval +21% / +1.6,
  but svamp 0% / -3.7 and BFCL 7-8%). The stop line from the original Qwen stays at 10-16% saved at every length of
  training: cut-label stop pairs alone do not move it far. Soup 300 next, then ps1 and dp1.
- **stopV3-soup-300 (fresh 240 x 8): +25.5% / -0.6.** The last soup finally shortens more (10-16% through soup 200).
- **drop1 done (GPU 1, 27 Oct ~03:00): 28 merged models removed from Drive, 34.6 GB freed** (ca1-soup-20/30/40, ca2-soup-10..40,
  drpo1-20/20x0.5/40/60, rc1-rl-20/60, rc1-soup-10/20, stopB3-soup-15/20/30/40, stopB-soup-10/20/30, stopB2-soup-20,
  rc2-soup-10/20/30/60, rr1-rl-20); kept sb1-10/20/30 (adapters/sb1-drpo not on Drive). All rebuildable from their adapters
  (recipes above). GPU 1 now on ps1 (prompt search); drpo3f at step 10 on GPU 2.
- stopV3-soup-300 per part: mbpp +54% / -4.1, ifeval +26% / +2.5, bfcl simple +14% / -1.2, multiple +7% / -2.8,
  irrelevance +2% / 0.0, svamp +2% / +0.6. The jump is almost all code (and IFEval); math and tool calls barely move.
- ~05:10: GPU 1 on ps1, 10 of 11 prompts sampled (draft-typed last), table next, then dp1. GPU 2: drpo3f at training
  step 14/40 (~8 min a step; its fresh scores come after training, ~3.5 h).

### 2026-10-27 ~05:50 — ps1 prompt search (original Qwen, plotV2 lot, 50 questions x 4 per source, vs no prompt)
Training sources only (gsm8k_matched, metamath_sv, ifeval_like, toolace; no code in this lot). Leak = the thinking
uses a prompt word the question lacks; the base row (no prompt) already reads 15% (false hits, IFEval-like 40%),
so read leaks against 15%.
| prompt | saved | acc pp | leak | gsm8k / metamath / ifeval-like / toolace (saved/acc) |
|---|---|---|---|---|
| concise | +1.4% | +2.0 | 11% | +10/+1.0, +6/+4.5, -9/-0.5, -2/+3.0 |
| prefill | -4.3% | -1.0 | 13% | -6/+1.5, -3/-0.5, -21/-1.5, +10/-3.5 |
| nocheck | +5.3% | +2.4 | 12% | +25/+1.5, +8/+5.0, -9/+1.0, -6/+2.0 |
| nocheck-prefill | +0.2% | +0.9 | 10% | +20/+2.5, +4/+5.5, -25/-6.0, -4/+1.5 |
| **budget** (100 tokens) | **+21.5%** | **+1.9** | 22% | +41/+3.5, +22/+5.0, +4/0.0, +15/-1.0 |
| cod | +17.1% | -11.9 | 16% | +44/-30.0, +26/-12.5, -14/-5.5, 0/+0.5 |
| examples | +19.3% | -0.4 | 25% | +48/+3.0, +18/-1.5, -10/-3.0, +10/0.0 |
| exdraft | +24.8% | -2.3 | 27% | +54/-1.5, +35/-3.0, -1/-3.5, -7/-1.0 |
| exdraft-prefill | +24.8% | -1.9 | 24% | +55/-1.0, +33/-3.5, -14/-4.0, +8/+1.0 |
| draft-typed | +20.3% | -3.5 | 30% | +60/-1.0, +24/-8.5, -22/-6.0, -11/+1.5 |
Reading: the "budget" prompt is the only one that saves a lot (+21.5%) without losing accuracy, and the only one that
helps every type a little; it is the candidate for prompted answers in LUPIN step 1. Worked-example prompts save more on
math (+48-60%) but cost accuracy and leak more (the model talks about the example). Chain-of-draft breaks math (-30 pts).
No prompt shortens IFEval-like questions; several lengthen them. Small sample (200 replies per source): a screen, not a score.
- dp1 (dataset build + check) started on GPU 1. drpo3f at step 17/40 on GPU 2.

### 2026-10-27 ~06:45 — dp1 (LUPIN step 0): pool built, representativeness check FAILS (15/24 measures OK)
Pool: 9,580 questions (wanted 11,000). Prompt lengths match the choosing set in every part (by construction), but the
original Qwen does NOT behave on the pool as on the choosing set (200 pool questions x 4 per part vs choosing x 4):
| part | pool size | accuracy pool vs choosing | median thinking pool vs choosing | verdict |
|---|---|---|---|---|
| svamp | 3000 (gsm8k 819 / metamath_sv 2181, mix inverted) | 92 vs 91 OK | 512 vs 366 (+40%) | CHECK: math pool thinks 40% longer |
| mbpp | **1055 / 3000** (kodcode: 1,994 dropped by the 8-gram check) | 56 vs 51 OK | 956 vs 1101 (-13%) OK | OK, but short of questions |
| bfcl_simple | 1429 | **56 vs 91** | 138 vs 148 OK | CHECK: ToolACE simple calls fail 35 pts more |
| bfcl_multiple | 714 | **62 vs 78** | 158 vs 182 OK | CHECK: -16 pts |
| bfcl_irrelevance | **382 / 857** | 82 vs 77 OK | 164 vs 125 (+31%) | CHECK: thinks longer, short of questions |
| ifeval | 3000 | **89 vs 49** | **190 vs 332 (-43%)** | CHECK: IFEval-like is far easier and shorter |
Reading: matching prompt length is not enough; difficulty and thinking length differ by part. Only code passes.
IFEval-like is the worst (an easy, short-thinking set: training on it would teach "be short" where the real test needs
thought). Math: gsm8k_matched lacks short prompts, so metamath_sv filled the short bins (mix 27/73 instead of 70/30) and
its answers think longer (540 vs gsm8k 456 vs choosing 366). Tools: ToolACE's graded calls are much harder (or noisier)
than BFCL's. LUPIN step 1 stays blocked until a pool passes.
- drpo3f-5 (fresh 240 x 8): +18.8% / -1.6 (its start, drpo3d-23, was +21.2% / +0.2): 5 more RL steps made it worse.
- GPU 1 moved on to gold1 (old queue). GPU 2: drpo3f at step 20/40.

### 2026-10-27 ~05:46 — drpo3f-10 scored (fresh 240 x 8)
drpo3f = gentle RL (DRPO, lambda 0.3) continued from drpo3d-23 (21.2% / +0.2), protect math + toolace/multiple.
- step 10: tokens saved +23.6% CI [+19.5, +27.9], accuracy +0.23 pp CI [-1.80, +2.37] (within the limit).
  SVAMP +29.5% / -1.56; MBPP+ +22.2% / +2.19; IFEval +26.0% / +0.62; BFCL simple +13.5% / +1.56,
  multiple +18.2% / -0.94, irrelevance +16.8% / -1.56.
- vs its start drpo3d-23: +2.4 points saved at the same accuracy (noise floor ~0.8 pt / 0.1 pp). Steps 15-40 next.
- Same time, GPU 1: dp1 done, gold1 training (step 600/660); math separates safe/unsafe stops (p 0.57-0.67 vs 0.15-0.23),
  code weakly (0.60 vs 0.43), tools and instructions barely.

### 2026-10-27 ~06:20 — gold1 scored: BROKEN at every checkpoint (fresh 240 x 8)
gold1 = stop_gold.py from the original Qwen (BCE on p("</think>") at labelled paragraph ends, LoRA r16, lr 1e-4).
| checkpoint | tokens saved | accuracy |
|---|---|---|
| step 100 | 96.5% | -62.5 |
| step 200 / 300 / 400 / 500 | 95.8 / 95.9 / 96.0 / 95.2% | -62.5 / -62.7 / -62.4 / -62.5 |
| step 600 / 660 | 96.7 / 95.6% | -63.0 / -63.0 |
Step 100 detail: replies are ~18-32 tokens in total; 0% right on SVAMP, MBPP+, IFEval, BFCL simple and multiple, and
~100% "unreadable" (no parsable answer); BFCL irrelevance 99% (an empty reply counts as a correct refusal).
So this is not "stopping early": the model no longer writes an answer at all. Thinking-off would score far better.
Likely cause (hypothesis, replies not yet read): the loss only touches p("</think>") at paragraph ends and nothing
anchors the rest of the distribution; the training log shows p(stop) pushed to 0.5-0.8 even where the label says keep
(tools, IFEval, code), i.e. a global "</think>" boost, so the model emits </think> almost at once and probably keeps
emitting it (or EOS) after it. Already broken by step 100 (800 replies seen).
Fix needed before any rerun: a KL/NLL anchor on all other tokens of the same replies (and on the reply after
</think>), a much lower lr or fewer steps, and a smoke test (generate 5 replies) before scoring.
Warning: teach_stop.py (teach1c, queued on GPU 1 after bp1b) uses the same unanchored BCE on p("</think>").
- drpo3f-15: +24.6% / +0.5; drpo3f-20: +22.5% / +0.4 (start drpo3d-23 was 21.2 / +0.2; step 10 was 23.6 / +0.2).

### 2026-10-27 ~06:41 — bp1b: original Qwen + "Think briefly: keep your thinking under N tokens" (fresh 240 x 8)
Test-time prompt (NOT a clean method; it only says what a distilled prompt could buy). Budgets svamp 400, mbpp 750,
ifeval 200, bfcl 70. Reply tokens without -> with the prompt, accuracy change:
| benchmark | tokens | saved | accuracy |
|---|---|---|---|
| SVAMP | 644 -> 344 | 47% | -2.8 |
| IFEval | 703 -> 526 | 25% | -3.4 (budget words in 68% of thinkings vs 28%) |
| MBPP+ | 2098 -> 1922 | 8% | -4.1 |
| BFCL simple / multiple / irrelevance | ~200 -> ~200 | -1 to -3% | -1.2 / -0.6 / 0.0 |
Overall (my computation, geometric mean of ratios, BFCL pooled ~1.02): ~22% saved at ~-2.7 pts. Worse than ps1 on
training questions (+1.9 there); math halves but loses ~3 points; code and tools ignore the budget.
Reading: distilling this prompt alone cannot reach the goal; at best a math-only ingredient.
- GPU 1 then started teach1c (teach_stop.py, same unanchored BCE as gold1) — user asked whether to pause it.

### 2026-10-27 ~09:30 — why dp1 failed (read in its 4,800 check answers) and dp2 (LUPIN step 0, user: "a big representative dataset across all data types first")
Diagnosis, from runs/bon/probe-poolcheck-dp1/rollouts.jsonl (original Qwen x4, 200 pool questions per part):
- **BFCL simple 56% / multiple 62% is mostly GRADING, not difficulty.** Of 356 wrong simple answers, re-read with an
  independent matcher: 80 are exactly the reference call (e.g. `getAdDuration(start_date="2024-01-01", end_date=
  "2024-06-30", frequency=5)`), failed because ToolACE declares `"type": "int"` (BFCL knows "integer"); 201 differ from a
  single GPT-written reference whose values the question never states ("this month" -> "2026-02-01..26", "Tesla" vs
  "TSLA", "@fashionnova" vs "fashionnova"); 63 malformed; 12 empty. 136 of 800 simple prompts are multi-turn
  ("Inquirer: ... historical dialog"), which BFCL never is.
- **BFCL irrelevance was mislabelled**: "no-fit" = any ToolACE reply that failed to decode, so many are real requests
  with malformed parallel calls ("team-week rounds for Manchester United, Chelsea and Liverpool"); the model called a
  function in 143/800.
- **IFEval-like is easy because of its instruction mix**: dp1's prompts are mostly "at least N sentences" (93% passed),
  "include keywords" (93%), "N highlighted sections" (92%). The "filtered" config has 0 repeat-prompt, 0 response-language,
  11 two-responses, 157 no-comma, 112 quotation, 147 all-capitals rows (IFEval: 4.9%, 3.7%, 2.9%, 7.9%, 4.9%, 3.0% of its
  instructions). The "default" config (550k rows) has them.
- **Math**: SVAMP is 1-2 operations; dp1 math was 73% MetaMath SV/FOBAR ("solve for x" / backwards), median thinking 529,
  gsm8k_matched 445 (choosing 366). GSM8K train has 2,579 problems with 1-2 calculations (gsm8k_easy).
- **Code**: KodCode-Light-RL-10K is too small after the 8-gram check (1,055 / 3,000); KodCode-V1 has 484k tasks, same schema.
dp2 changes (contamination check unchanged; fresh set never read):
- prepare_types.py `--tool-clean`: BFCL type names (int->integer, number->float, ...; unknown -> dropped), single-turn
  only, reference values must be stated in the question, no-fit = a simple question offered 2 unrelated functions from
  other conversations (BFCL's construction). Local run on ToolACE: 3,061 simple, 2,319 multiple, 5,944 no-fit
  (old: 5,025 / 3,615 / 1,554).
- `--ifeval-match N`: N prompts drawn so the 1/2/3-instruction split and each instruction id's share follow IFEval's 541
  prompts; default-config kwargs converted to IFEval's per-instruction form. Local test on filtered + 1 of 10 default
  shards: 2,117 matched prompts (hard ids still short there; the full config has ~10x).
- `--kod-dataset KodCode/KodCode-V1`; math mix gsm8k_easy 0.55 / gsm8k_matched 0.15 / metamath_sv 0.30.
- select_pool.py (new): every candidate (1.6x oversampled) answered x4 by the original model; per part, cells of
  solve rate (never / <half / >=half / always) x thinking terciles matched to the original model's runs on the choosing
  set (v4-base x4) AND questions_40 (eval-base x2): 80 reference questions per part instead of 40.
- Check on NEW answers (probe_curve.py --seed 7, 200 selected questions per part), pool_check.py against both references
  and each alone, plus IFEval pass rate per instruction id.
- Queue: GPU 1 buildpool2 dp2 a then select (after teach1c), GPU 2 dp2 b (after drpo3f). ~19k candidates x4, about
  4-5 h per GPU. No training on any pool until every part reads OK.
- ~10:00 check: drpo3f ended after its step-20 score (no step 25-40 scores in the log; stop reason not printed in the
  visible output). GPU 2 then started drpo2 (scripts 9ba75c2, before the dp2 push), now at RL step 30/60; dp2 b waits
  behind it. GPU 1 still on teach1c (rel:0.05 at 1,200/2,512 answers, then rel:0.1). dp2 starts when each finishes.

### 2026-10-27 ~11:00 — teach1c broken like gold1; dp2 first try crashed (bug fixed), IFEval mix now exact
- teach1c (teach_stop.py from the original Qwen, BCE on p("</think>"), no anchor), fresh 240 x 8:
  rel:0.05 +85.9% / -49.4; rel:0.1 +85.3% / -55.6. Per part (rel:0.05): SVAMP +92% / -89.7, MBPP+ +96% / -45.0,
  BFCL simple +86% / -88.1, multiple +85% / -75.9, irrelevance +83% / +18.4, IFEval +9% / -14.4. Same failure as gold1
  (the model stops answering); confirms the unanchored stop-token loss is broken, not a dose problem.
- dp2 a and select failed on GPU 1 in prepare_types.py: some argilla default-config rows have kwargs "null"
  (TypeError on len). Fixed (rows without kwargs, or malformed JSON, are skipped).
- Second IFEval problem found while testing locally on 3 of the 10 default shards: argilla rows with 1-3 instructions
  barely hold the hard ids (no-comma 60, quotation 43, repeat-prompt 16, capital-word-frequency 14 among 39,667), so
  real rows alone reach 4,373 of 12,000 with those ids far short. Fix: ifeval_fill() builds the missing prompts from a
  Dolly question (creative_writing / brainstorming / open_qa / general_qa, no context; not an evaluation set) followed
  by IFEval's OWN instruction sentences and random arguments (its instruction library's build_description; conflicting
  pairs excluded by its registry; repeat-prompt last, repeating everything before it). Local result: 4,373 real + 7,619
  built = 11,992 prompts; every instruction id's share and the 1/2/3-instruction split equal IFEval's (e.g. no-comma
  7.9% / 7.9%, 1 instruction 56%). IFEval's official checker grades the built prompts (300 graded without error; a
  no-comma reply passes, one with a comma fails). With all 10 shards more of the 12,000 will be real rows.
- Queue: GPU 1 is now on sb2 (60 RL steps); dp2 a + select re-queued right after it (distinct lines, a failed line is
  not retried in the same session). GPU 2: dp2 b after drpo2. A runtime restart would start dp2 sooner.

### 2026-10-27 ~12:00 — user: build and verify the dataset before any LUPIN step; keep the running stages
- No restart: GPU 1 finishes sb2 (scoring its checkpoints), GPU 2 finishes drpo2 (scoring), then each runs dp2
  (GPU 2: half b, next line after drpo2; GPU 1: half a + select, re-added after sb2 with different spelling because its
  first dp2 lines failed this session). No LUPIN stage is queued; LUPIN steps 1-3 wait until dp2's check passes.
- Pass criteria for dp2 (all on held-out answers, never the fresh set): every part (SVAMP, MBPP+, IFEval, BFCL simple,
  BFCL multiple, BFCL irrelevance) matches the original model's behaviour on the choosing set and on questions_40
  (accuracy, median reply and thinking tokens, capped share); IFEval instruction-id mix equals IFEval's; proportions:
  each benchmark 25%, BFCL split 11.9 / 6.0 / 7.1% (its 400 / 200 / 240), select_pool.py now prints this check.
- All run results keep being logged here as they arrive (drpo2, sb2 checkpoints).

### 2026-10-27 ~10:55 — sb2 and drpo2 checkpoints (fresh 240 x 8), dp2 candidates built
Both runs start from rc1-rl-40.
- **sb2** (ShorterBetter, beta 0.001, math protected, guard 0.06): the guard stopped it at step 10 (ifeval_like 0.75 -> 0.69).
  - sb2-5: +53.7% / -2.9 (irrelevance +45/-2.2, multiple +41/-5.9, simple +39/0.0, ifeval +46/-3.1, mbpp +61/-2.2, svamp +63/-3.7)
  - sb2-10: +52.6% / -3.2 OUTSIDE (irrelevance +45/-2.8, multiple +41/-5.0, simple +39/-0.6, ifeval +47/-2.8, mbpp +59/-4.1, svamp +60/-3.1)
- **drpo2** (lambda 0.5, 60 steps), scored checkpoints so far:
  - drpo2-5: +55.5% / -2.0 (irrelevance +46/-0.9, multiple +42/-3.1, simple +39/+0.3, ifeval +50/-0.6, mbpp +61/-1.9, svamp +65/-4.1)
  - drpo2-10: +56.4% / -3.0 (irrelevance +46/-3.1, multiple +42/-4.1, simple +39/-1.6, ifeval +48/-1.9, mbpp +63/-1.6, svamp +67/-5.6)
  - drpo2-15: +56.0% / -2.6 (irrelevance +47/-4.1, multiple +42/-3.7, simple +39/-1.6, ifeval +40/0.0, mbpp +66/-1.6, svamp +68/-5.6)
  Both sit at ~53-56% saved for -2 to -3 pts. Math (svamp -4 to -6) costs the most; neither beats the -1 goal.
- **dp2 half a started on GPU 1** (candidates, seed 73): gsm8k_easy 2,579, gsm8k_matched 1,200, metamath_sv 80,000;
  toolace simple 3,061 / multiple 2,319 / no-fit 5,944 (11,324); ifeval_like 11,993 = **4,662 real + 7,331 built** from
  IFEval's instruction library (instruction counts 1/2/3: 6,759 / 3,969 / 1,265); KodCode-V1 (484k) loading.
  Note: 61% of the IFEval-like prompts are built, not real; the held-out check should report built vs real separately.
- ~13:00 dp2 a running on GPU 1. Its IFEval-like candidates: 4,662 real + 7,331 built of 11,993 (61% built from
  IFEval's library). select_pool.py and pool_check.py (--pool) now report IFEval real vs built separately
  (accuracy, median thinking and tokens vs the references), so a pass carried by built prompts can be seen; select
  pulls the new scripts when it starts, so dp2 uses them. gsm8k_easy supplies 2,579, just enough for 55% of the
  4,800 math candidates.
- dp2 candidate pool built on GPU 1 (build_pool.py, 1.6x oversample of 3,000 per benchmark): 18,640 questions.
  | part | kept / wanted | sources | prompt tokens p10-p90 pool vs choosing | dropped by 8-gram check |
  |---|---|---|---|---|
  | svamp | 4800/4800 | gsm8k_easy 2145, gsm8k_matched 819, metamath_sv 1836 | 60-85 vs 55-81 | 174 |
  | mbpp | 4800/4800 | kodcode (V1) | 85-152 vs 84-168 | 24,516 of 40,000 |
  | ifeval | 4800/4800 | ifeval_like (real + built) | 32-73 vs 34-76 | 3,135 |
  | bfcl_simple | **1726/2286 (short 560)** | toolace | 348-458 vs 385-482 | 632 |
  | bfcl_multiple | 1143/1143 | toolace | 564-1028 vs 591-951 | 550 |
  | bfcl_irrelevance | 1371/1371 | toolace | 383-473 vs 382-449 | 1,257 |
  BFCL simple has 1,726 candidates for the 1,429 the final pool needs (1.2x instead of 1.6x): less room to match
  behaviour cells there; select_pool reports any shortfall. GPU 1 now answers half a (9,321 questions x4).
- drpo2-15: +56.0% / -2.6 (drpo2-5 +55.5 / -2.0, -10 +56.4 / -3.0); GPU 2 scoring drpo2-20 next.

### 2026-10-27 ~11:20 — drpo2-20/25/30; dp2 half a CRASHED
- drpo2-20: +57.6% / -3.9 OUTSIDE (svamp +68/-7.8); drpo2-25: +57.8% / -4.0 OUTSIDE (svamp +69/-9.7);
  drpo2-30: +58.2% / -4.7 OUTSIDE (irrelevance +47/-4.7, multiple +42/-4.4, simple +39/-1.2, ifeval +49/-1.9,
  mbpp +65/-1.6, svamp +71/-11.9). Past step 15 drpo2 only buys ~2 pts of saving while math collapses.
- dp2 half a (GPU 1) crashed while grading its answers: an IFEval-like candidate asks for a response language ('zh')
  missing from IFEval's language table (graders.py build_description -> KeyError 'zh'). GPU 1 moved on to the select step,
  which now waits (up to 8 h) for half a's answers that will not come. dp2 b on GPU 2 would hit the same error.
  Reported to the session that owns dp2 for a candidate fix (drop rows whose instruction IFEval cannot build).

### 2026-10-27 ~14:00 — dp2 half a crashed while answering; rebuilt as dp3
- probe_curve.py -> typed.is_correct -> IFEval build_description: KeyError 'zh'. Real argilla rows use language codes
  IFEval's checker does not know (its LANGUAGE_CODES has no 'zh'); my candidate filter only checked argument names.
  GPU 1's select step then idled waiting for half a (up to 8 h).
- Fix: ifeval_buildable() tries every IFEval-like row (real and built) against IFEval's checker once at candidate
  time and drops what it rejects. Local run (filtered + 3 of 10 default shards): 37,342 eligible real rows (2,325
  rejected), 4,330 real + 7,663 built = 11,993; all 11,993 graded by the official checker without an error; mix still
  equals IFEval's.
- Renamed dp3 (fresh files; dp2's saved candidates are not reused). GPU 2: dp3 b after drpo2. GPU 1: dp3 a + select
  queued; GPU 1 is stuck in dp2's select wait until it times out (8 h) or the user stops and reruns the plan cell
  (sb2 commented out so a rerun does not resume its RL past the guard stop).
- drpo2 (strong RL from rc1-rl-40, lambda 0.5, math unprotected): -20 +57.6 / -3.9, -25 +57.8 / -4.0,
  -30 +58.2 / -4.7 (math sliding to -11.9). More saving, accuracy getting worse: not a candidate.
- drpo2-40: +62.1% / -4.6 (outside the limit); drpo2-50 scoring. GPU 1 still waiting in dp2's select (needs the user's stop + rerun of the plan cell).
- drpo2-40: +62.1% / -4.6 OUTSIDE (irrelevance +47/-5.0, multiple +43/-4.1, simple +40/+0.3, ifeval +58/-2.5, mbpp +67/-1.9,
  svamp +74/-10.9); drpo2-50: +62.9% / -5.4 OUTSIDE (irrelevance +48/-5.0, multiple +44/-5.6, simple +40/-0.3, ifeval +59/-0.9,
  mbpp +68/-1.6, svamp +74/-15.3). drpo2-60 scoring. GPU 1 still waiting in dp2's select (needs the user's stop + rerun for dp3).

### 2026-10-27 ~15:30 — drpo2 finished; dp3 candidates built on GPU 2 (half b answering)
drpo2 (rc1-rl-40 + DRPO lambda 0.5, math unprotected), fresh 240 x 8, per part saved / accuracy:
| step | overall | SVAMP | MBPP+ | IFEval | BFCL simple | BFCL multiple | BFCL irrelevance |
|---|---|---|---|---|---|---|---|
| 30 | 58.2 / -4.7 | 71 / -11.9 | 65 / -1.6 | 49 / -1.9 | 39 / -1.2 | 42 / -4.4 | 47 / -4.7 |
| 40 | 62.1 / -4.6 | 74 / -10.9 | 67 / -1.9 | 58 / -2.5 | 40 / +0.3 | 43 / -4.1 | 47 / -5.0 |
| 50 | 62.9 / -5.4 | 74 / -15.3 | 68 / -1.6 | 59 / -0.9 | 40 / -0.3 | 44 / -5.6 | 48 / -5.0 |
| 60 | 62.7 / -6.0 | 76 / -15.0 | 69 / -2.2 | 54 / -1.9 | 40 / -1.2 | 44 / -6.9 | 48 / -6.2 |
Reading: most of the loss is math (-11 to -15) and BFCL multiple / irrelevance (-4 to -7); code and IFEval lose about
1-2 points at 54-69% saved. Not a candidate as is; it says strong length pressure on rc1-rl-40 is affordable on
code/IFEval but not on math or tool choice (measured on one run, no seeds).
dp3 candidates (GPU 2): ToolACE clean 3,061 / 2,319 / 5,944; IFEval-like 4,696 real + 7,298 built (58,230 real
eligible after the checker test, all 10 shards); build_pool: same counts as dp2 (18,640; BFCL simple 1,726/2,286),
IFEval dropped 3,104 by the 8-gram check. GPU 2 answers half b (9,319 questions x4). GPU 1 still waits in dp2's select
(stop + rerun of its plan cell needed to start half a).
- ~16:00 user away from the computer, so GPU 1 cannot be unstuck by hand: GPU 2 now answers dp3 half a after
  half b. Each half first restores its answers from Drive and skips if they are complete, so GPU 1 (freed when dp2's
  select wait times out, ~21:30) joins or skips half a, then runs dp3 select.

### 2026-10-27 ~12:45 — drpo2-60; dp3 candidate pool built (GPU 2)
- drpo2-60: +62.7% / -6.0 OUTSIDE (irrelevance +48/-6.2, multiple +44/-6.9, simple +40/-1.2, ifeval +54/-1.9, mbpp +69/-2.2,
  svamp +76/-15.0). drpo2 summary: best trade at step 5-15 (+55.5-56.0% / -2.0 to -2.6); later steps add ~6 pts of saving
  and lose ~3-4 pts, mostly math (svamp -15).
- dp3 candidates (1.6 x the pool, seed 73): 18,640 questions. IFEval-like: 4,696 real + 7,298 built (all graded by IFEval's
  own checker). Per part kept/wanted: svamp 4800/4800 (gsm8k_easy 2,145, gsm8k_matched 819, metamath_sv 1,836); mbpp
  4800/4800 (KodCode-V1; 24,516 dropped by the 8-gram check, 43,212 usable); ifeval 4800/4800; bfcl_simple 1726/2286
  (SHORT 560, still 1.2 x its final 1,429); bfcl_multiple 1143/1143; bfcl_irrelevance 1371/1371. Prompt lengths match the
  choosing set in every part. Now: GPU 2 answering half b (9,319 questions x 4, ~4 h), then half a; GPU 1 frees itself
  when dp2's select wait times out (~21:30) and joins dp3 a / runs dp3 select.
- Evening: user unstuck GPU 1 (plan cell rerun); GPU 1 answers dp3 half a (9,321 x4), GPU 2 half b (9,319 x4); half a removed from GPU 2's plan.

### 2026-10-27 ~17:30 — TokenSkip (arXiv 2502.12067) read, built, queued raw (tsk1, GPU 2 after dp3)
Paper (EMNLP 2025): LLMLingua-2 (a BERT-size "keep / drop" word classifier) scores each word of a right chain of thought;
the top gamma share is kept in order; LoRA SFT on the cut chains (~7.5k GSM8K/MATH examples, gamma in the prompt). GSM8K,
~50% fewer tokens: Qwen2.5-14B -1.7, 7B -5.4, 3B -9.3 pts (smaller = worse); LLaMA-8B MATH-500 -30% tokens / -1.9. It cuts
words (connectors, articles, restatements), never steps; only instruct-model chains of 200-500 tokens, no thinking model.
Built: tokenskip.py (thinking only; reply verbatim; one right answer per question; --verify keeps an example only if the
original model, handed the cut thinking, still answers right) + track.py stage `tokenskip` (SFT per ratio, checkpoint per
epoch, fresh x8). Local test on a GSM8K-style trace: ratio 0.6 -> actual 0.62 (247 -> 153 tokens), numbers and equations
kept, filler gone, the "let me double-check" step kept (cut to "double-check 16 minus 3 is 13, 13 minus 4 is 9").
Queued: tsk1 = original Qwen, label-lot rollouts (training sources), ratios 0.5 / 0.7 and 1.0 (same SFT, uncut: control).
Also LUPIN_SPEC M12 (TokenSkip on LUPIN0's answers, gentle ratios).
- ~18:20 GPU 2: dp3 half b finished (no failure line; answers saved to Drive). tsk1 (TokenSkip, other session) then
  failed at once: vLLM could not get its GPU memory (ValueError in request_memory at engine start; root cause above it
  in the log). GPU 2's queue then ended ("everything is on Drive"), so that runtime releases itself; nothing else of
  dp3 needs it. GPU 1: dp3 half a at 48 of ~63 chunks, then select + the held-out check.
- Fix for tsk1: tokenskip.py now frees LLMLingua (del, gc, torch.cuda.empty_cache) before starting vLLM for --verify, and
  asks vLLM for 0.7 of the GPU instead of 0.85. Compression itself worked (6214 questions with a right answer); only the
  verify start failed. Re-run needs a GPU runtime (user's call); the plan line can stay as is.

### 2026-10-27 night — dp3 selection done (select_pool.py, GPU 1); held-out check running
11,998 questions selected from 18,640 answered candidates, matched to the original model on choosing set (v4-base x4)
+ questions_40 (eval-base x2). Numbers below are on the SELECTION answers (flattering by construction); the verdict is
the held-out check (new answers, seed 7, 200 per part) now running.
| part | kept | accuracy pool vs reference | median thinking pool vs reference | borrowed from other cells |
|---|---|---|---|---|
| svamp | 2998/3000 | 85 vs 87 | **500 vs 384** | 1411 |
| mbpp | 2999/3000 | 56 vs 54 | 954 vs 1051 | 231 |
| bfcl_simple | 1429/1429 | 83 vs 87 | 120 vs 145 | 278 |
| bfcl_multiple | 714/714 | 77 vs 77 | 159 vs 170 | 147 |
| bfcl_irrelevance | 857/857 | **97 vs 80** | 138 vs 131 | 242 |
| ifeval | 3001/3000 | 54 vs 52 | 249 vs 290 | 984 |
IFEval real 1,307 kept (46%, thinking 214), built 1,694 (61%, thinking 273). Proportions all OK (25 / 25 / 11.9 / 6.0 /
7.1 / 25%). Early reading: math candidates lack short-thinking easy questions (half the math pool borrowed from
neighbouring cells; thinking +30%); the built no-fit tool questions are too easy (97% vs 80%: unrelated functions are
obvious to refuse; BFCL's irrelevance functions are closer to the question). Wait for the held-out check.
- dp4 queued on GPU 1 after dp3's check (user: start fixing now, refine after the check): stage poolfix.
  Math: new sources gsm8k_easy_test (GSM8K test, 1-2 calculations) and metamath_easy (MetaMath GSM_Rephrased linked to
  a 1-2 calculation GSM8K problem by same answer + word Jaccard >= 0.4). No-fit: "near" questions (ToolACE question
  offered only the functions its reply did not call), dropped when an offered function shares a name word with the
  called one (local: 1,719 near kept of 2,319; e.g. "ad tracking data" + trackAdImpression was a false no-fit).
  Residual risk: some near rows may still be answerable; their "wrong" answers would look like difficulty to the
  selection. Selection over dp3's 18,640 answered candidates + the new ones; every part re-checked on new answers.
- Night: GPU 1's runtime DISCONNECTED during dp3's held-out check (1 of ~8 check chunks done; tab shows 'Reconnect L4'). Answers so far are on Drive; after a reconnect + Run all, dp3 select re-runs (same seed, same pool) and the check resumes, then dp4. GPU 2 is off too (its queue ended). Both need the user.

### 2026-10-27 night — user decisions for LUPIN (nothing built yet)
- Pair sources, with accuracy as the priority: backbone = the model's own natural answers (more answers per question,
  difficulty protection: no shorten pairs on questions it often fails) + repeated rounds (each round re-answers 4-8 times
  and retrains, scored after every round). Cut pairs only for code, strict labels, small dose. Prompted answers skipped
  (cost -2.7 on the real test, math hit hardest).
- LUPIN step 1: **10 answers per question** on the final pool, starting only once the dataset check passes (the pool's 4
  selection answers reused, 6 more generated: ~72,000 answers, ~13 GPU-hours). Measured with 4 answers (dp1 check): longest
  right / shortest right ~1.4-1.7x, shortest right ~15-19% shorter than a typical answer, per type.
- User (27 Oct night): the training PAIRS must match the evaluation distribution as well as the pool. After the 10 answers, pairs are selected per part and per (solve rate x thinking) cell to the evaluation's shares, and checked before training (LUPIN_SPEC 2.0).
- dp4 split over two GPUs (user): GPU 1 poolfix half a then select + check; GPU 2 poolfix half b (before tsk1). Halves hash-stamped like dp3; whichever runtime gets there first builds the dp4 candidates and saves them, the other restores them.

### 2026-10-28 ~00:00 — data bug found while drawing the overlap chart: gsm8k_easy used the wrong answer instruction
- dp3's gsm8k_easy candidates were built with prepare_data.py's MATH_INSTRUCTION ("Put the final answer on the last
  line, like: Final answer: 42"), not the evaluation's SVAMP wording ("The last line of your reply must start with
  'Final answer:' followed by the number alone.") used by gsm8k_matched and metamath_sv. Of their 8,580 answers: 2,187
  (25%) no readable final answer, 941 (11%) capped, 292 really wrong: accuracy 69%, median thinking 786 (gsm8k_matched
  93% / 410, metamath_sv 93% / 484). The selection kept 1,023 of them, so they look like hard, long math: a likely cause
  of dp3's math thinking 500 vs 384.
- Fix (before dp4 runs): gsm8k_easy, gsm8k_easy_test and metamath_easy now use the SVAMP wording (svamp_row); dp4's
  re-selection leaves out dp3's gsm8k_easy rows and their answers.
- GPU 1's restart re-ran gold1's training (stop_gold.py) while walking the plan; gold1, bp1b, teach1c and stopB5 are now
  commented out so a restart cannot redo them.
- 28 Oct: GPU 2 stuck 'Connecting' for 25 min (no second GPU granted); dp4 half b also queued on GPU 1 (a, b, select).
- 28 Oct (user): NO training of any kind until the training data is shown to overlap the evaluation and the two gaps are
  fixed (math thinking too long, mostly the gsm8k_easy format bug; no-fit tool questions too easy, 97% vs 79%). tsk1
  (TokenSkip, trains) paused in plan_line_a. Active queues: dataset stages only (dp3 check, dp4 a/b/select).
- 28 Oct (user): LUPIN in rounds: v1 = DPO on the original Qwen's answers (10/question); v2 = DPO from v1 on v1's answers to a new representative sample (8-10/question, matched to v1's own behaviour on choosing + questions_40). After each round, per-type thinking distributions vs the original on fresh x8 in the overlap dashboard (LUPIN_SPEC 'Rounds').

### 2026-10-28 — dp3 HELD-OUT CHECK (new answers, seed 7, 200 selected questions per part x4) — FAILS on 3 parts
Reference "both" = choosing set x4 + questions_40 x2 (80 questions per part); interval = the reference's 95% bootstrap.
| part | accuracy pool vs ref | median thinking pool vs ref | verdict |
|---|---|---|---|
| svamp | 81 vs 88 [83, 93] | **516 vs 366 (+41%)** | CHECK: gsm8k_easy (format bug) 65% / 987 tokens; gsm8k_matched 88% / 447, metamath_sv 90% / 465 |
| mbpp | 59 vs 53 | 902 vs 1054 (-14%) | OK |
| bfcl_simple | 84 vs 88 | 122 vs 143 (-15%) | OK vs both (CHECK vs choosing alone: -6.5 / -18%) |
| bfcl_multiple | 77 vs 78 | 161 vs 175 (-8%) | OK |
| bfcl_irrelevance | **98 vs 79** | 133 vs 129 | CHECK: built no-fit questions too easy |
| ifeval | 58 vs 51 | **246 vs 316 (-22%)** | CHECK (new): thinks too SHORT; real 53% / 232, built 63% / 258; replies 354 vs 488 tokens |
17/24 measures OK vs both references (15/24 vs choosing, q40 similar pattern). IFEval per instruction: no_comma 24%,
number_paragraphs 17%, json_format 23%, two_responses 0% (12 answers) are the hard ones; mix matches IFEval's.
Reading: math and no-fit are the two known gaps (dp4, now answering on both GPUs: 2,674 new candidates per half).
IFEval thinking too short is NEW: the selection had too few long-thinking IFEval candidates (borrowed 984); dp4 does not
touch IFEval, so it needs its own fix (dp5) before any training.
- dp5 queued after dp4 (user): IFEval only. New source ifeval_built: human-written requests from HuggingFaceH4/no_robots
  (Generation, Brainstorm, Open QA, Rewrite, Summarize; 8-150 words; e.g. "write a sad breakup song, two verses and a
  chorus") + IFEval's own instruction sentences in IFEval's mix. Local: 11,993 prompts from 6,501 requests, median 51
  words (Dolly bases were shorter), 1/2/3-instruction split 6,764 / 3,966 / 1,263 as IFEval. 2.5x oversample, answered
  on both GPUs, re-selection over dp4's answered pool + these, full held-out check. Also: dp4's leave-out of dp3's
  gsm8k_easy now keys on the old instruction text, so dp4's fixed gsm8k_easy rows stay in dp5's pool.
- dp4 selection (GPU 1, selection answers): math median thinking 422 vs 384 (dp3 507), accuracy 85 vs 87; sources
  metamath_easy 1340, gsm8k_easy (fixed wording) 521, gsm8k_matched 366, metamath_sv 700, gsm8k_easy_test 71. No-fit
  94% vs 80% (dp3 97%): still too easy. IFEval unchanged (248 vs 290; dp5 targets it). Proportions all OK. Held-out
  check running on GPU 1.
- GPU 2: dp5 b failed at once ("dp4-all's answered pool is not here": GPU 1 had not finished dp4's selection yet), its
  queue ended and the runtime released itself. poolfix now waits up to 8 h for the reused pool on Drive. GPU 1 runs
  dp5 a, b and select by itself anyway; reconnecting GPU 2 would only halve dp5's answering time.

### 2026-10-28 ~02:00 — overnight chain (user asleep): dp5 (IFEval) → dp6 (no-fit + math) → gated 10 answers
- dp6: harder no-fit questions. nofit_verify.py (runs on the Mac, user's OpenRouter key): each ToolACE question is offered
  the 1-3 functions from other conversations closest to it in meaning (all-MiniLM-L6-v2 embeddings), never its own or
  one sharing a name word; Haiku 5.5 (anthropic/claude-haiku-5.5, $0.10/M in) answers two differently worded "could one
  of these functions fulfil this request?" checks; kept only on NO + NO. Trial of 100 (after fixing empty replies):
  NO 17, UNSURE 57, mixed 22, YES 4 -> ~17% kept. First trial with word-overlap nearness gave unrelated pairs
  ("semiconductor fabrication" + legal.caseSummary): replaced by embeddings. Full run (~5,400 candidates) under way.
  dp6 also adds short math candidates (metamath_easy, gsm8k_easy_test); re-selection over dp5's answered pool.
- pool_check now also compares the thinking-length p25 and p75 (user: the whole distribution, not only the median).
- lupinanswers (LUPIN step 1): 6 new answers per question (4 reused = 10) on dp6's pool, split over both GPUs, GATED in
  code on "verdict: ALL OK" in poolcheck-dp6-both; otherwise it logs the gate and does nothing. No training queued.

### 2026-10-28 ~02:30 — dp4 held-out check + verified no-fit file + dp6 widened
dp4 check (new answers, vs choosing + questions_40; run with the older pool_check, no p25/p75):
| part | accuracy | median thinking | verdict |
|---|---|---|---|
| svamp | 84 vs 88 | 416 vs 366 (+14%) | OK (dp3 +41%: format bug fixed) |
| mbpp | 53 vs 53 | 993 vs 1054 | OK |
| bfcl_simple | 82 vs 88 | 116 vs 143 (-19%) | CHECK (dp3 -15%, borderline both times) |
| bfcl_multiple | 73 vs 78 | 157 vs 175 | OK |
| bfcl_irrelevance | 94 vs 79 | 143 vs 129 | CHECK (dp6) |
| ifeval | 52 vs 51 | 233 vs 316 (-26%) | CHECK (dp5, running) |
Math sources now: gsm8k_easy (fixed wording) 88% / 366, gsm8k_easy_test 83% / 342, metamath_easy 90% / 380.
- nofit_near_verified.jsonl (Mac, Haiku 5.5, two checks): of ~5,400 near candidates, NO+NO 926 kept; YES 241 (would have
  been wrong no-fit labels), UNSURE 3,000, mixed 1,198. Kept examples are tempting the way BFCL's are ("a recipe with
  chicken" + Search_Chicken_Dishes_by_ID; "oral health status" + getGastroenterologyReport).
- dp6 widened to BFCL simple: clean Hermes questions (BFCL types, grounded values; locally 115 simple / 161 multiple /
  92 no-fit of 478) added as candidates. Few simple ones: may not be enough to close the simple gap.
- ~03:00: Google Drive is FULL ('Org storage full', 499.48 GB used). Saves to Drive may silently fail (rsync without check), so GPU 2 waits for dp4-all forever. GPU 1 holds everything on its own disk and runs dp5 a+b+select, dp6 a+b+select and now both lupinanswers halves alone. Needs the user in the morning: free Drive space (old merged models, as drop1 did) or results stay only on GPU 1's disk.
- dp5 selection (selection answers): IFEval median thinking 298 vs 290 (dp4 248), accuracy 54 vs 52, borrowed 189 (dp4 984): the no_robots-based prompts fill the long-thinking cells. IFEval kept: real 494, Dolly-built 690, no_robots-built ~1,817. Other parts as dp4. Held-out check running on GPU 1.

### 2026-10-28 ~04:00 — dp5 held-out check (first with p25/p75): IFEval FIXED; code tail found short; dp7 queued
vs choosing + questions_40 (80 q per part), new answers, 200 pool q per part; 30/36 measures OK:
| part | accuracy | median thinking | p25 | p75 | failing |
|---|---|---|---|---|---|
| svamp | 86 vs 88 | 408 vs 366 (+11%) | 329 vs 282 (+17%) | 561 vs 585 | p25 |
| mbpp | 58 vs 53 | 885 vs 1054 (-16%, in interval) | 574 vs 586 | **1446 vs 2220 (-35%)** | p75 |
| bfcl_simple | 86 vs 88 | 120 vs 143 (-16%) | 102 vs 117 | 152 vs 199 (-24%) | median, p75 |
| bfcl_multiple | 78 vs 78 | 154 vs 175 | 124 vs 135 | 191 vs 224 | none |
| bfcl_irrelevance | **96 vs 79** | 140 vs 129 | 115 vs 111 | 183 vs 172 | accuracy |
| ifeval | 55 vs 51 | 286 vs 316 (-9%) | 210 vs 231 | 402 vs 457 | **none (fixed)** |
IFEval by origin: no_robots-built 131 q 52% / 305; Dolly-built 38 q 62% / 278; real 31 q 61% / 240.
- dp6 (running) targets svamp p25, bfcl_simple, no-fit. New dp7 targets code's missing long tail: KodCode-V1 medium+hard,
  tasks up to 160 tokens and solutions up to 400; re-selection over dp6's pool. The 10-answer step now waits for dp7's
  check (gate on poolcheck-dp7-both ALL OK).
- ~05:45: dp6's build_pool kept only 332 of 1,371 wanted no-fit candidates (from the 926 verified; 174 dropped by the
  8-gram check) and 6 of 2,286 simple ones (Hermes): BFCL's no-fit and simple prompts sit in a narrow length window
  (382-449 / 385-482 tokens, one function each), which most multi-function or long candidates miss. For dp7:
  - nofit_verify.py --singles 3: up to 3 one-function near candidates per ToolACE question, each Haiku-checked twice
    (~16,500 candidates, running on the Mac); kept ones are appended to nofit_near_verified.jsonl before dp7 starts.
  - glaive_tools (glaiveai/glaive-function-calling-v2, Apache 2.0): clean one-call questions, locally 31,080 simple /
    9,862 multiple, many compute-style like BFCL simple (calculate_loan_payment(principal=50000, interest_rate=5, ...)).
  - dp7 now covers mbpp (KodCode-V1 medium+hard) + bfcl_simple + bfcl_irrelevance.
- One-function near no-fit run (Haiku x2, ~16,500 candidates): 4,542 kept on NO+NO (report in scratch); 3,672 new appended -> nofit_near_verified.jsonl now 4,598 questions, read by dp7.
- dp6 selection (selection answers): math median 402 vs 384 (dp5 423), borrowed 90 (dp5 373); no-fit 92% vs 80% (dp5 94%); simple unchanged (121 vs 145, Hermes 6 rows); IFEval kept 298 vs 290. dp6 held-out check running; dp7 (one-function no-fit, Glaive simple, harder code) next.

### 2026-10-28 ~06:30 — dp6 held-out check: 29/36 vs both (26/36 vs choosing alone, 35/36 vs questions_40 alone)
Failing vs both: svamp accuracy 83 vs 88 (-5.6, just past the 5-pt tolerance; thinking quartiles now all OK);
mbpp p75 thinking 1,522 vs 2,220 (-31%) and capped 3% vs 8%; bfcl_simple median thinking 118 vs 143 (-17%), p75 144 vs
199 (-28%); bfcl_irrelevance accuracy 91 vs 79. IFEval and bfcl_multiple pass on every measure. The two references
disagree (choosing set: harder and longer; questions_40: the pool passes 35/36), so "both" is the fair target.
dp7 (running) targets code, simple and no-fit; math accuracy is the one gap no stage targets (borderline).
- ~07:00: dp7 too slow (hard KodCode answers ~25 min per 150-question chunk, ~12 h per half). Replaced by dp7c: same sources, new candidates capped (mbpp 1,200, bfcl_simple 1,600, no-fit all ~1,371); answers gated on poolcheck-dp7c-both. dp7's candidate build showed: mbpp 4,800 (10,327 dropped by the 8-gram check, prompts longer than the choosing set's), Glaive simple 2,286 (12,310 dropped by the 8-gram check), verified no-fit 1,371 (738 dropped).
- dp7c candidates after caps: 3,868 new (bfcl_simple 1,600 Glaive, mbpp 1,200 KodCode medium/hard, no-fit 1,068 verified near); prompt lengths of the new ones are off (Glaive simple median 391 vs 427 tokens; KodCode medium/hard 139 vs 102): the selection mixes them with dp6's candidates, the held-out check will show whether the selected pool still matches. Half a answering 1,934 questions on GPU 1.

### 2026-10-28 — Drive full: drivereport + drop2 (user: free ~40 GB; user empties the trash afterwards)
drivereport (GPU 2, read only): models 79.7 GB (66 merged models x 1.21 GB), adapters 28.2 GB, runs 11.0 GB, data 1.1 GB.
Google account: Drive 217 GB, Photos 270 GB, Gmail 11.6 GB, org limit 540 GB ("Org storage full").
drop2 removes 37 merged models (~44.8 GB) through the Drive mount (they go to the Drive TRASH; the user must empty it
to free quota). Every one is superseded or failed, and its adapter is kept. Rebuild recipes (merge.py):
- gold1-step-{100..600,660} = Qwen3-0.6B + adapters/gold1/step-N (broken run, -63 pts)
- teach1-rel0.05 / rel0.1, teach1c-rel0.05 / rel0.1 = Qwen3-0.6B + adapters/teach1(-c)-teach/<variant> (broken)
- drpo2-{5..60} = rc1-rl-40 + adapters/drpo2-drpo/step-N; sb2-{5,10} = rc1-rl-40 + adapters/sb2-shorterbetter/step-N
- stopV2-soup-{25,50,75,100,150}, stopV3-soup-{25..200} = Qwen3-0.6B + mean of adapters/stopV2|V3-s{0,1,2}/step-N
- mf1-soup-{50,75,100,150} = rc1-rl-40 + mean of adapters/mf1-s{0,1,2}/step-N
Kept models: rc1-rl-40, rc1-soup-30, ca1-soup-10, stopB3-soup-25, stopB5 soups, sb1-*, drpo3b-*, drpo3d-*, drpo3f-*,
stopV3-soup-300, mf1-soup-25, drpo1-*. All scores, adapters, lots and answers stay.
- drop2 done on GPU 2: 37 models removed, freed 44.7 GB (now in the Drive trash: the user empties it). GPU 2 then started dp7c half b (both GPUs on dp7c).

### 2026-10-28 — SUMMARY: the LUPIN training dataset, dp1 → dp7c (state at dp7c answering)
Goal (user): ~12,000 training questions that behave like the evaluation for the ORIGINAL Qwen, per part (SVAMP, MBPP+,
IFEval, BFCL simple / multiple / no-fit), each benchmark 25% (BFCL split 400 / 200 / 240), checked on NEW answers against
the choosing set (x4) + questions_40 (x2); the fresh set is never used. No LUPIN training before the check passes and
the user has seen it (hard gate, also in code for the 10-answer step).
Pool check, held-out answers, measures OK vs both references (out of 24 until dp4, 36 from dp5 with p25 / p75):
| pool | math | code | IFEval | simple | multiple | no-fit | OK |
|---|---|---|---|---|---|---|---|
| dp1 | +40% thinking | short of questions | 89 vs 49% right | 56 vs 91% (grading) | -16 pts | +31% thinking | 15/24 |
| dp3 | +41% thinking (format bug) | OK | -22% thinking | OK | OK | 98 vs 79% | 17/24 |
| dp4 | OK (+14%) | OK | -26% thinking | -19% thinking | OK | 94 vs 79% | 17/24 |
| dp5 | p25 +17% | p75 -35% | **OK** | -16% median | OK | 96 vs 79% | 30/36 |
| dp6 | accuracy 83 vs 88 | p75 -31% | OK | -17% median | OK | 91 vs 79% | 29/36 |
| dp7c | (running) | KodCode-V1 medium+hard | — | Glaive one-call | — | Haiku-verified one-function near | |
Bugs found and fixed on the way: ToolACE "int" types (BFCL rejects right calls); ungrounded ToolACE references; undecodable
replies labelled no-fit; IFEval-like instruction mix too easy; IFEval language codes the checker lacks (crash); GSM8K easy
problems with the wrong answer instruction (25% unreadable); no-fit candidates too easy (unrelated functions); BFCL's narrow
one-function prompt-length window; hard code answers too slow (dp7 -> capped dp7c).
Infra this night: Drive full (499.5 / 540 GB org) -> drop2 freed 44.7 GB, user emptied the trash -> 411.7 GB used (Drive
129.5 GB); GPU 2 was blocked until then, now on dp7c half b. Next: dp7c select + check -> if ALL OK, 6 more answers per
question (10 total) -> pairs selected to the evaluation's distribution -> user review -> LUPIN v1.
- 28 Oct: SPARSE add-on project specified (thinkingcap/SPARSE_SPEC.md): Wanda / Lottery Ticket pruning of Qwen3-0.6B, then LoRA or freed-weight training to think shorter; separate from LUPIN, reuses LUPIN data, pilot ~15-20 GPU-h on the Wanda cut. Nothing built.
- dp7c selection (selection answers): no-fit 84% vs 80% (dp6 92%), simple median thinking 132 vs 145 (dp6 121; Glaive 666 + ToolACE 759 kept), code borrowed 100 (dp6 231), math 85 vs 87 / 398 vs 384, IFEval 300 vs 290; proportions OK. Held-out check running on GPU 1; GPU 2 waits for its verdict (lupinanswers gate).

### 2026-10-28 — dp7c HELD-OUT CHECK (new answers, seed 7, 200 per part x4): 32/36 vs both, 30/36 choosing, 35/36 q40
| part | accuracy pool vs ref (both) | median thinking | verdict |
|---|---|---|---|
| svamp | 87 vs 88 | 388 vs 366 (p25 304 vs 282, p75 537 vs 585) | OK |
| mbpp | 56 vs 53 | 954 vs 1054; p75 1847 vs 2220 (-17%, was -31%) | OK (choosing alone: p75 -36% CHECK) |
| bfcl_simple | **80 vs 88** | 129 vs 143; p75 162 vs 199 (-19%) | CHECK: now slightly too HARD (toolace 76%, glaive 85%) |
| bfcl_multiple | 74 vs 78 | 159 vs 175 | OK |
| bfcl_irrelevance | **88 vs 79** | 146 vs 129 | CHECK: still too easy (dp6 91, dp5 96) |
| ifeval | 48 vs 51 | 285 vs 316; p75 379 vs 457 (-17%) | p75 CHECK |
Gate: lupinanswers a and b both logged "dp7c's pool check is not ALL OK; no answers generated". Both queues ended; both
runtimes released. No training. Waiting on the user: accept dp7c (35/36 vs q40) or a dp8 round on no-fit + simple.

### 2026-10-28 — dp8 queued (user: "yes prepare dp8")
No new candidates. Diagnosis: dp7c's selection answers said no-fit 84% vs 80%, its new answers 88%; questions picked for
a lucky 4-answer draw drift back (regression to the mean; simple drifted the other way, 80 vs 88). dp8 (stage poolmore):
every bfcl_simple / bfcl_irrelevance / IFEval candidate of dp7c-all gets 4 more answers (seed 5, half per GPU), the
selection reads all 8 answers and uses 4 thinking bins (quartiles) instead of 3, so p25 / p75 are matched, not only the
median (IFEval and simple p75 were -17/-19%). Held-out check on seed 9 (seed 7 answers informed this design). Gate
unchanged: lupinanswers dp8 runs only on "verdict: ALL OK". No training.
- dp8 started on both GPUs (user reconnected both runtimes; scripts at e47cc3b). Half a (GPU 1): 9,044 candidates x4
  (ifeval 5,299, no-fit 2,033, simple 1,712), seed 5; half b (GPU 2) the other 9,044. Estimate ~2.5 h of answering,
  then GPU 1 selects on 8 answers (4 thinking bins) and runs the seed-9 held-out check (~45 min). Checks every 30 min.
- ~14:30: dp8 both halves at 3,450/9,044 questions (~1.7 min per 150-question chunk, ~2,300 tokens/s); answering done in ~1 h.
- ~15:45: dp8 answering done on both halves (9,044 x4 each). dp8 selection (8 answers for tools / IFEval, 4 thinking bins):
  12,000 questions, proportions all OK. vs choosing: svamp 84 vs 87 / 396 vs 384; mbpp 56 vs 54 / 1028 vs 1051;
  simple 83 vs 87 / 130 vs 145 (glaive 660, toolace 766); multiple 77 vs 77 / 160 vs 170; no-fit 87 vs 80 / 140 vs 131
  (still too easy even on 8 answers: the candidates lack hard no-fit questions, selection cannot fix that);
  IFEval 56 vs 52 / 284 vs 290. Held-out check (seed 9, 1,200 questions x4) running on GPU 1; GPU 2 waits on the gate.

### 2026-10-28 ~16:30 — dp8 HELD-OUT CHECK (seed 9, 200 per part x4): 33/36 vs both, 30/36 choosing, 36/36 q40
| part | accuracy (both) | median thinking | verdict |
|---|---|---|---|
| svamp | **81 vs 88** [83,93] | 400 vs 366; p25 312/282, p75 575/585 | accuracy CHECK (dp7c 87: 4-answer selection drift, now on math) |
| mbpp | 55 vs 53 | 982 vs 1054; p75 1842 vs 2220 (-17%) | OK |
| bfcl_simple | **81 vs 88** [82,93] | 126 vs 143; **p75 157 vs 199 (-21%)** | 2 CHECK (glaive 83%/141, toolace 80%/157) |
| bfcl_multiple | 77 vs 78 | 160 vs 175 | OK |
| bfcl_irrelevance | 87 vs 79 [70,87] | 140 vs 129 | OK, but at the edge of the interval (dp7c 88 CHECK) |
| ifeval | 57 vs 51 | 292 vs 316; p75 395 vs 457 (-14%, was -17%) | OK |
Gate: closed (both not ALL OK): lupinanswers dp8 skipped. Choosing-only adds mbpp p75 -36%, simple median -15%, IFEval p75 -16%.
8 answers + quartile bins fixed IFEval p75 and moved no-fit inside its interval; simple did not move; math drifted
(it was still selected on 4 answers).

- Harder no-fit, Mac (user: "prepare the harder no-fit questions with haiku", Glaive download approved): ToolACE
  look-alikes (--hard, name words allowed) failed in trial: 200 -> 18 kept, all obviously unrelated (Haiku is UNSURE on
  close ones). Glaive (glaiveai/glaive-function-calling-v2 downloaded to the Mac scratch; 40,942 clean rows, same as Colab)
  in BFCL's calculation style: nearest functions are synonyms (rank 0-60: 4/200 kept, 112 YES); ranks 15-60 give
  same-style wrong functions (BMI -> calculate_speed, rectangle area -> calculate_emission): 123/150 kept; ranks 60-200
  140/150 but less related. Full run started: --rank 15,60 --singles 1 --max 8000 (unique questions), ~$1.

### 2026-10-28 ~17:15 — dp9 queued (user: "yes queue dp9 once haiku finishes")
- Glaive no-fit Haiku run done (report below); kept rows committed as nofit_glaive_verified.jsonl (prepare_types source
  glaive_nofit_verified, origin near-verified-glaive).
- dp9n (poolfix, both GPUs): new candidates for bfcl_simple (the Glaive simple rows not yet used) and bfcl_irrelevance
  (Glaive no-fit), oversample 3, x4 answers; union with dp7c-all only (no selection).
- dp9 (poolmore): every svamp / simple / no-fit / IFEval question brought to 8 answers (dp8's 4 extra reused, the rest
  answered with seed 5), selection on 8 answers with quartile thinking bins, held-out check on seed 13 (seeds 7 and 9 shaped
  the design). Gate unchanged.
- Glaive no-fit Haiku report: 8,000 candidates -> NO+NO 6,070 kept (76%), NO/UNSURE 636, UNSURE/NO 321, UNSURE 823, UNSURE/YES 95, YES 54 (~$1).
- ~17:40: dp9n new candidates 4,068 (bfcl_simple 2,774 Glaive, no-fit 1,294 Glaive-verified; 3,610 Glaive no-fit dropped by the 8-gram / length window, prompt lengths a bit short: no-fit median 396 vs 416, simple 383 vs 427). Both GPUs answering their 2,034 x4, fast (~25 s per chunk).
- ~19:50: dp9 answering done; selection on 8 answers (12,000, proportions OK). vs choosing: svamp 86 vs 87 / 398 vs 384
  (dp8 84); mbpp 55 vs 54 / 1034 vs 1051; simple 84 vs 87 / 132 vs 145 (glaive 880, toolace 547; dp8 83); multiple 77 vs 77;
  no-fit 84 vs 80 / 142 vs 131 (glaive 212 + toolace 646; dp8 87); IFEval 56 vs 52 / 284 vs 290. Held-out check seed 13
  running on GPU 1 (300/1,200).
- Next if dp9's check is ALL OK (explained to the user 28 Oct ~20:00): lupinanswers dp9, ORIGINAL model, 6 new answers
  per question (seed 11), both GPUs, ~4-6 h (code answers are the long ones). Each question then has >= 10 answers
  (math / tools / IFEval 14: 8 selection + 6). Data only. Then: topic-overlap check + a pair-selection proposal matched to
  the eval per type, shown to the user; LUPIN v1 training only after the user approves.
- Drive cleanup asked (user: only what is really unnecessary, log everything first): drivereport (read only) queued on
  GPU 1 right after dp9's check; the delete list goes to the user before anything is removed.

### 2026-10-28 ~20:45 — dp9 HELD-OUT CHECK (seed 13, 200 per part x4): 34/36 vs both, 30/36 choosing
| part | accuracy (both) | median thinking | verdict |
|---|---|---|---|
| svamp | 88 vs 88 (dp8 81) | 416 vs 366 (+14%); p25 308/282, p75 626/585 | OK (fixed by 8 answers) |
| mbpp | 62 vs 53 [44,64] | 898 vs 1054 (-15%); **p75 1541 vs 2220 (-31%)** | p75 CHECK (code still selected on 4 answers; dp8 p75 -17%) |
| bfcl_simple | 84 vs 88 [82,93] (dp8 81) | 133 vs 143; **p75 166 vs 199 (-17%, CI 171-221)** | p75 CHECK (glaive 87%/149, toolace 79%/176) |
| bfcl_multiple | 77 vs 78 | 155 vs 175 | OK |
| bfcl_irrelevance | 85 vs 79 [70,87] (dp8 87) | 143 vs 129 | OK (glaive 87%, toolace 85%) |
| ifeval | 58 vs 51 | 286 vs 316; p75 395 vs 457 (-14%) | OK |
Choosing-only adds: mbpp p75 -47%, simple p75 -20%, multiple median -15% / p75 -16%, IFEval reply -15% / p75 -16%.
Gate: closed, lupinanswers dp9 skipped on both GPUs; both queues ended.

drivereport drive3 (GPU 1, read only): models 47.1 GB (39 merged; drop2's stopV2/stopV3 soups are BACK: GPU 1's
runtime still held local copies and its next save re-uploaded them; dropmodels now also clears local copies when the
Drive copy is already gone, and must run on BOTH runtimes), adapters 28.2 GB, runs 23.5 GB, data 2.0 GB. Every model's
adapters are on Drive. Delete list proposed to the user (nothing removed yet).

### 2026-10-28 ~21:15 — user: option A (dp10) + drop3 (keep the strong candidates of methods A-H)
**drop3 (both runtimes, Drive trash; user empties it):** 29 merged models still on Drive, ~34 GB, all rebuildable
(merge.py, ~2 min each; scores already logged):
- ca1-soup-10 = Qwen3-0.6B + mean of adapters/ca1-s{0,1,2}/step-10
- sb1-{10,20} = rc1-rl-40 + adapters/sb1-shorterbetter/step-N (strong RL, H)
- stopB5-soup-{10..30} = Qwen3-0.6B + mean of adapters/stopB5-s{0,1,2}/step-N
- drpo3b-{5,10,15} = drpo3b's start + adapters/drpo3b-drpo/step-N; drpo3d-{5..20} = drpo3b-20 + adapters/drpo3d-drpo/step-N;
  drpo3f-5 = drpo3d-23 + adapters/drpo3f-drpo/step-5; drpo1-{20,40,60} = drpo1's start + adapters/drpo1-drpo/step-N
- stopV2-soup-{25..100}, stopV3-soup-{25..200} = Qwen3-0.6B + mean of adapters/stopV2|V3-s{0,1,2}/step-N
Plus drop2's names again (their Drive copies are gone; this clears local copies that a save would re-upload).
**Kept (strong candidates per method):** A stop training: stopB3-soup-25, stopV3-soup-300 (+25.5 / -0.6);
E rc1: rc1-soup-30; F rc1-rl-40 (53.9 / -2.1); F+math: mf1-soup-25; G gentle RL: drpo3b-20 (+13.5 / -0.3),
drpo3d-23 (+21.2 / +0.2), drpo3f-10 (+23.6 / +0.2), drpo3f-15 (+24.6 / +0.5), drpo3f-20 (+22.5 / +0.4);
H strong RL: sb1-30. (B self-preference, C balanced, D clean have adapters only.) Adapters, runs, data untouched.

**dp10 (poolmore):** every code candidate of dp9n-all brought to 8 answers (seed 5); dp8 + dp9 extra answers reused;
same data seed 81 as dp9, so only the code selection changes; held-out check seed 17. Why code drifted: dp7c/dp8/dp9 drew
from the SAME code candidates and got 56 / 55 / 62% on new answers: 4 answers per question misjudge hard code (a question
right 1/4 times is often really 2-3/4), and each new data seed reshuffles which borderline questions fill the cells,
so the code part moves by several points and its long tail (hard questions, long thinking) moves with it.
- ~20:50: dp10 code answering very slow at chunk 150 (~20 min per chunk, ~600 tokens/s: a few capped code answers keep the GPU nearly idle at each chunk's end; ~7 h per half). Restarted with --dp-chunk 600 (resumes from the saved chunks).
- Second code source search (user: "yes check this out"; Mac only, nothing on the GPUs): OpenCoder-LLM/opc-sft-stage2
  educational_instruct (MIT, 118,278 tasks, MBPP-style "Write a function to ...", entry_point + assert tests: our format).
  Overlap with MBPP (all 974, google-research-datasets/mbpp): 476 exact MBPP texts, 2,285 same function name,
  240 share an MBPP assert, 85,175 share a 6-gram (mostly the "write a python function to" phrasing). Strict clean
  (no 6-gram, no name, no assert): 32,613; then MiniLM cosine to the nearest MBPP task: >=0.9 423, >=0.8 5,120,
  >=0.75 10,567 (0.80 still lets "valid parentheses" paraphrases through: use < 0.75, ~22,000 kept). Task tokens
  p10-p90 19/25/35/50/71 vs MBPP 12/13/16/18/21 (longer; KodCode slice median 28); solutions 62 vs 44 median.
  Likely EASY tasks (MBPP-like), so it widens the pool but may not add the hard long-thinking tail; Qwen's answers
  decide. Not used yet: candidate for dp11 if dp10 still misses on code.
- ~21:20: the user's Mac shut down; both Colab runtimes kept running (Manage sessions: both active). Read through the Drive logs (code/, code-second/track_log.txt, downloaded): at 21:19 each GPU ~36-44% through its first 600-question code chunk after 33 min (~800-900 tokens/s). Bigger chunks did not help much: code answers are long. Estimate ~80-90 min per chunk, 5 chunks per half: answering done ~03:30, select + check ~04:30.
- ~22:10 (user: 'can we do 1200 questions only?'): dp10 capped at 600 code questions per half (--dp-more-cap 600) = the first 600-question chunk each GPU just finished (random among the 6,000 code candidates): 1,200 code questions on 8 answers, the other 4,800 stay on 4. Saves ~5 h; weaker than all 6,000 on 8 answers.
- Correction (user: '2400 total'): --dp-more-cap 1200 per half = 2 chunks of 600 per GPU, 2,400 of the 6,000 code candidates on 8 answers; answering done ~23:30, check ~00:15.
- ~22:30: Colab displays never reattached after the Mac shutdown; the user reconnected, then both sessions were restarted (Runtime > Restart session; the machines came back fresh: full setup + Drive mount approved by the user) and the plans relaunched with --dp-more-cap 1200. At 22:55 both still in the plan runner's first Drive restore (all pool answers, adapters: several GB on a fresh machine); dp10 resumes from the 600 code questions per half saved on Drive. New estimate: answering ~00:40, check ~01:30.
- 23:15: dp10 resumed on both GPUs but from ZERO (0 already sampled, 1,200 to go per half): the first 600-question chunks had not reached Drive before the old runtimes ended (fresh machines came back). 2 chunks x ~80 min per GPU: answering ~01:55, check ~02:45.
- dp11 staged (commented) in both plans: next 1,200 code questions per half to 8 answers (dp8-dp10 answers reused), check seed 19; runs only if dp10's check is poor.
- ~01:00 (user: "run 1,200 more on code overnight, except if we get <10% length and <5% accuracy"): dp11 now queued
  for real, decided by the runtime itself (track.py code_close_enough): after dp10's check, if code (vs both) is within
  5 accuracy points AND median and p75 thinking within 10%, dp11 is skipped (marker dp11-skipped) and dp10 goes to the
  gated 10 answers; otherwise dp11 runs (next 1,200 code questions per half to 8 answers, check seed 19; marker
  dp10-superseded, so only dp11 gets the 10 answers, still gated on its check being ALL OK). Tested on dp9's numbers:
  +9.2 pts / -15% / -31% -> runs.
- ~01:20 (user: "if it's close enough to eval data, don't wait for me and launch next steps directly"): the 10-answers
  gate (lupinanswers) now also opens when every CHECK line of the pool's check (vs both) misses its 95% interval by at
  most 1.5 points (accuracy, capped) or 3% of the bound (token counts) (track.py near_misses_only). Tested: dp9 would
  stay closed (code p75 71 tokens outside = FAR; simple p75 5 tokens = near). Training is NOT automatic: pairs (spec
  2.0) are built after the answers.

### 2026-10-29 ~01:45 — user: "run all these steps unless I interrupt or there is a clear red flag" (+ the pair variations)
Built and queued after the 10 answers (both GPUs):
- lupin_select.py (spec 2.0): pairs drawn per part (25% per benchmark, BFCL 400/200/240) and per solve x thinking cell
  (4 x 4, quartiles) at the shares of the original model's own runs on choosing + questions_40; questions visited
  round-robin; short cells reported (never filled from others); then every part trimmed to the lowest fill ratio so the
  part shares stay exact. Dry run on synthetic answers OK.
- track.py stage `lupin`: waits for the 10 answers of the pool used (auto: dp11 if dp10 superseded), merges every answer
  of the pool questions (selection + poolmore + 10-answer files), lupin_pairs.py recipe -> lupin_select.py (3,200
  wanted) -> DPO from the original model (NLL anchor) -> checkpoints 50/100/150/200 merged on local scratch (no Drive copy)
  and scored on the choosing set x4.
- Screen (the decided variations: natural answers + difficulty protection; no prompted answers):
  GPU 1: lp-l2s50 (long2short, protection p<0.5), lp-dast (DAST + protection), lp-right (right>wrong only, baseline);
  GPU 2: lp-l2s75 (long2short, protection p<0.75), lp-adar1 (Ada-R1 + protection).
  Next: the best 2 -> 3 seeds + soups on the fresh set x8; code-only cut pairs (strict labels, small dose) on the best;
  then LUPIN v2 (v1 answers a new matched sample).
- Red-flag rule (user, 29 Oct ~02:00: 'let it drift to <10% for any one data type'): a screen variation is stopped/flagged only if it loses 10 points or more on any one data type at every checkpoint. Finalist rule unchanged (overall accuracy change better than -1 on the choosing set).
- 02:10: dp10 code answering done on both GPUs (1,200 x4 per half). GPU 1 -> dp10 select + check (seed 17); GPU 2 in dp11 b waiting for dp10's check (automatic decision).
- 02:25: dp10 selection: code (mbpp) 56% vs choosing 54%, median thinking 1045 vs 1051 (dp9 55 / 1034), borrowed 68 (dp9 39); other parts as dp9 (simple 83 vs 87, glaive 881 + toolace 546). Held-out check (seed 17) running.

### 2026-10-29 ~03:15 — dp10 HELD-OUT CHECK (seed 17): 35/36 vs both
| part | accuracy (both) | median thinking | verdict |
|---|---|---|---|
| svamp | 87 vs 88 | 418 vs 366 (+14%); p75 593 vs 585 | OK |
| mbpp | 60 vs 53 [44,64] (+7.0) | 879 vs 1054 (-17%); **p75 1790 vs 2220 (-19%, inside [1612,3289])** | OK (dp9 p75 1541 CHECK): 8 answers on 2,400 code questions fixed the tail |
| bfcl_simple | 84 vs 88 | 132 vs 143; **p75 160 vs 199 [171,221] -20%** | p75 CHECK (11 tokens = 6.4% outside: not a near miss) |
| bfcl_multiple | 78 vs 78 | 157 vs 175 | OK |
| bfcl_irrelevance | 85 vs 79 | 140 vs 129 | OK |
| ifeval | 60 vs 51 [41,61] | 288 vs 316; p75 395 vs 457 | OK |
Choosing-only: also mbpp p75 -38%, simple accuracy -7.1, simple p75 -23%.
Code rule (user): accuracy +7.0 pts (> 5) and median -17% (> 10%) -> dp11 RUNS (next 1,200 code questions per half).
Simple's long tail missed in dp8/dp9/dp10 (157 / 166 / 160 vs 199): a selection blind spot (quartile bins: our
top-quartile simple questions think less than the eval's). dp11's selection now uses 8 thinking bins for bfcl_simple
only (select_pool.py --think-bins-part; tested on synthetic data), so its tail is matched more finely. Other parts unchanged.
- ~04:40: topic_check.py + stage `topiccheck` (GPU 1, before the screen; report only, never a gate): per part,
  MiniLM nearest-neighbour cosine eval->rest-of-eval vs eval->pool (pool subsampled to the evaluation part's size),
  pool->eval, and the share of evaluation questions whose closest pool question is below the evaluation's own 10th
  percentile (missed subjects); template lines (>30% of a part's questions) removed; fresh set excluded. Logic tested
  with a stub encoder (the Mac's scratch venv was wiped by the shutdown).

### 2026-10-29 ~07:00 — dp11 HELD-OUT CHECK (seed 19): 30/36 vs both, 26/36 choosing, 36/36 q40 -> dp10 chosen
dp11 CHECK lines (both): svamp median reply +15% / thinking 421 vs 366 (+15%); bfcl_simple accuracy 78 vs 88 (-9.7)
and p75 166 vs 199 (-17%); no-fit accuracy 88 vs 79 (+8.3); ifeval accuracy 65 vs 51 (+13.6). Code itself passed.
dp11 changed only code's answers and simple's bins, yet math, IFEval and no-fit moved by several points: the 8-bin change
shifted the selection's random draws for the later parts, and the check (200 questions x 4) swings run to run
(IFEval 60 -> 65, simple 84 -> 78). Iterating further would chase noise. Gate closed for dp11 (as designed).
Topic check on dp11's pool (report only; eval questions share their authors' style): eval->eval vs eval->pool median
cosine: svamp 0.95 vs 0.65 (SVAMP is near-duplicate stories: the metric is meaningless there), mbpp 0.75 vs 0.66 (41%
of MBPP questions below MBPP's own 10th percentile), simple 0.59 vs 0.37 (63%), multiple 0.51 vs 0.31 (59%), no-fit
0.49 vs 0.35 (46%), ifeval 0.50 vs 0.47 (13%, OK). BFCL's science/math-calculation questions are the farthest from ours.
DECISION (user: "if close enough to eval data, don't wait for me and launch next steps"): dp10 (35/36; every accuracy
and every code measure OK; only miss bfcl_simple p75 160 vs 199, 11 tokens outside its interval) is the training pool.
lupinanswers --la-force on dp10 (both halves), topic check on dp10, then the LUPIN screen with --lp-pool dp10.
Plan cells restarted with the finished lines commented out.
- 07:20: both plan cells restarted on the new plans (GPU 2 needed Runtime > Interrupt + a minute). GPU 1: lupinanswers dp10 a (gate forced), 750/6,000 questions x6 after ~10 min; GPU 2 starting lupinanswers dp10 b. Estimate: 10 answers done ~10:30-11:00, then topic check (GPU 1) and the LUPIN screen.
- 10:05: the 10 answers were ~3x slower than needed (6 new answers for every question although most already had 8). lupinanswers --la-target 10: tops each question up to 10 (8-answer questions get 2, 4-answer code questions 6), keeping the 6-answer run's finished chunks (GPU 1 1,800 / GPU 2 1,350 questions). Plan cells restarted.
- 10:30 (user: '14 or 10?'): lupin uses at most 10 answers per question (--lp-answers 10), newest first: the 10-answer run, then the extra-answer rounds dp11->dp8, then the 4 selection answers (the pool was chosen on the earliest answers, so they carry the lucky-draw bias; equal counts keep pair strength equal across types). Pairs and the solve-rate reading for matching use the same capped file.
- 10:35: top-up running. GPU 1 (half a): 1,950 questions already at 10+, 3,590 need 2, 460 need 6; GPU 2 (half b): 1,650 / 3,883 / 467. Estimate done ~11:45, then topic check (GPU 1) and the screen.

### 2026-10-29 ~13:15 — 10 answers done; topic check dp10; first pairs too thin -> matching fixed
- Answers: 12,000 pool questions, exactly 10 answers each (newest first: lupin-answers a 21,640 / b 20,468, poolmore
  dp11 4,932, dp10 4,516, dp9 15,004, dp8 18,144, selection 35,296).
- Topic check dp10 (report only): eval->eval vs eval->pool median cosine: svamp 0.95 vs 0.65 (meaningless: SVAMP is
  near-duplicate stories), mbpp 0.75 vs 0.66 (41% missed), simple 0.59 vs 0.38 (57%), multiple 0.51 vs 0.31 (60%),
  no-fit 0.49 vs 0.33 (52%), ifeval 0.50 vs 0.46 (13%, OK).
- lp-l2s75 pairs (long2short, protection p<0.75): 5,540 raw (never solved 2,155 questions, always solved 5,967); matched
  kept only 1,284 of 3,200 (fill 0.40 set by no-fit 92/229): cells with no possible pair (never-solved, and
  always-solved questions whose right answers are within 1.5x of each other) kept their quota, and every part was
  trimmed to the weakest part's fill. ~80 DPO steps, one scored checkpoint: too thin.
- Fix: lupin_select spreads the reference share of cells with NO pair under the recipe over the cells that have pairs
  (reported per part); part shares still exact. Screen renamed lpb-* (fresh pairs), checkpoints every 25 steps,
  scored at 25/50/75/100/150 where they exist. lp-l2s75's training (a few minutes) discarded.
- ~13:45 (user: add the 1.2x variant; pairs mixed in one epoch, not ordered; curriculum = spec M11, later on the winner): lupin_pairs.py --easy-ratio (always-solved questions: shorten from 1.2x) + --shorten-cap (shorten per part up to N x guard; always-solved questions give no guard pair, so the 1x cap would discard the extra pairs). Queued on GPU 2 after lpb-adar1: lpb-l2s50e (long2short, protection p<0.5, easy ratio 1.2, cap 2). Tested on synthetic answers. lpb-l2s75 pairs: 1,452 kept (fill 0.45, limited by no-fit's always-solved cells), ~90 steps.
- ~14:15: lpb-l2s50 pairs 2,012 kept (fill 0.63; guard 492, shorten 1,382, right 138), 123 DPO steps, training (GPU 1, step 10). lpb-l2s75 1,452 pairs, 88 steps, training (GPU 2, step 25; pairs ordered right 78-90%, tripwire 0).
- ~14:05 FIRST LUPIN SCREEN SCORES (choosing 240 x 4 vs v4-base; tokens saved / accuracy change):
  | checkpoint | overall | svamp | mbpp | ifeval | simple | multiple | no-fit |
  |---|---|---|---|---|---|---|---|
  | lpb-l2s75-25 | +37.5% / -1.5 | +52 / 0.0 | +44 / -5.0 | +15 / -1.3 | +31 / -0.6 | +31 / +1.9 | +37 / -0.6 |
  | lpb-l2s75-50 | +43.7% / -1.5 | +55 / -5.0 | +53 / -2.5 | +25 / +3.1 | +34 / +1.9 | +37 / -3.1 | +40 / -3.8 |
  (choosing set: 40 questions per part x 4, so one part moves ~2.5 pts per question; per-part noise is large.)
  lpb-l2s50 trained 123 steps, step-25 being scored.
- ~15:00 screen scores (choosing 240 x 4; saved / accuracy; per part svamp, mbpp, ifeval, simple, multiple, no-fit):
  | checkpoint | overall | svamp | mbpp | ifeval | simple | multiple | no-fit |
  |---|---|---|---|---|---|---|---|
  | lpb-l2s50-25 | +36.3% / -0.9 | +49 / -1.2 | +34 / -2.5 | +31 / +1.3 | +26 / -0.6 | +29 / -1.2 | +34 / -1.9 |
  | lpb-l2s50-50 | +47.7% / -1.0 | +55 / -3.7 | +53 / -1.9 | +43 / +3.1 | +35 / -0.6 | +39 / -1.2 | +40 / -2.5 |
  | lpb-l2s50-75 | +47.8% / -1.9 | +51 / -2.5 | +52 / -3.1 | +52 / -0.6 | +31 / -1.9 | +34 / -3.1 | +39 / +0.6 |
  | lpb-l2s75-75 | +43.8% / -1.1 | +51 / -1.9 | +51 / -3.7 | +37 / +1.9 | +32 / -1.2 | +33 / +1.2 | +39 / -1.9 |
  lpb-l2s75 done (88 steps; best 75: +43.8 / -1.1). lpb-l2s50-50 is the best so far (+47.7 / -1.0, at the -1 line).
  GPU 2 now trains lpb-adar1 (194 steps); GPU 1 scoring lpb-l2s50-100.
- ~16:00: lpb-l2s50-100: +47.6% / -1.2 (svamp +48/-1.2, mbpp +52/-6.2, ifeval +52/+3.1, simple +34/+1.9, multiple +36/-3.1, no-fit +41/0.0). lpb-l2s50 done (123 steps; best step 50: +47.7 / -1.0). GPU 1 now trains lpb-dast (183 steps, step 40, pairs ordered right 70%); GPU 2 trains lpb-adar1 (step 135/194).
- 15:27: lpb-adar1-25 (Ada-R1: shortest right > the 2 longest answers, protection p<0.5): +48.5% / +0.4 (svamp +61/-3.1, mbpp +52/+5.0, ifeval +42/+1.9, simple +33/-3.1, multiple +36/-3.7, no-fit +40/0.0). Best so far. lpb-adar1 trained 194 steps; 50-150 being scored. lpb-dast at step 130/183 (pairs ordered right ~70-80%: DAST's 'longer wrong > shorter wrong' pairs are harder to learn).
- 16:30 screen (choosing 240 x 4; saved / accuracy):
  | checkpoint | overall | svamp | mbpp | ifeval | simple | multiple | no-fit |
  |---|---|---|---|---|---|---|---|
  | lpb-adar1-50 | +43.2 / +0.3 | +54 / +1.9 | +48 / -0.6 | +31 / +3.8 | +33 / -4.4 | +36 / -5.6 | +43 / -1.3 |
  | lpb-adar1-75 | +47.1 / -1.4 | +53 / -0.6 | +55 / -1.9 | +42 / 0.0 | +31 / -1.9 | +35 / -5.6 | +42 / -1.9 |
  | lpb-adar1-100 | +45.0 / -1.0 | +52 / +1.9 | +45 / -2.5 | +46 / -1.9 | +31 / -1.9 | +35 / -3.1 | +41 / 0.0 |
  | lpb-adar1-150 | +51.4 / -1.7 | +55 / -1.9 | +62 / -4.4 | +49 / +1.3 | +30 / -1.2 | +36 / -2.5 | +41 / -1.3 |
  | lpb-dast-50 | +34.1 / -0.1 | +49 / -0.6 | +36 / -2.5 | +22 / +3.8 | +22 / +1.2 | +24 / -2.5 | +31 / -1.3 |
  | lpb-dast-75 | +30.5 / -0.7 | +49 / +1.3 | +37 / -5.6 | +2 / +3.1 | +23 / 0.0 | +25 / -6.2 | +30 / +1.2 |
  | lpb-dast-100 | +37.7 / -0.8 | +54 / 0.0 | +40 / -5.0 | +21 / +3.1 | +29 / -1.2 | +29 / -1.9 | +33 / -1.3 |
  Reading: Ada-R1 leads (step 25 +48.5 / +0.4; saving keeps rising to +51.4 at 150 for -1.7); long2short p<0.5 close
  (+47.7 / -1.0 at 50); DAST is the most accurate but saves least (30-38%); stronger protection (p<0.75) saves less.
  STEP 8 queued (rule: best saved with accuracy better than -1 -> adar1; second l2s50, on the -1 line, near-identical
  saving): lpc-adar1 (GPU 2, after lpb-l2s50e) and lpc-l2s50 (GPU 1, after lpb-right), 3 seeds, soups at 25/50/75 on
  the fresh set x8. If lpb-l2s50e beats lpb-l2s50, it replaces lpc-l2s50.
- ~17:10: lpb-dast-150: +36.3 / -1.7 (dast done; best 50: +34.1 / -0.1). GPU 1 trains lpb-right (121 steps, step 25; right>wrong pairs learn slowly, 44-62% ordered right). GPU 2: lpb-l2s50e at step 160/168, scores next.
- 17:30 lpb-l2s50e (long2short p<0.5 + always-solved shorten from 1.2x, cap 2x guard): 2,728 pairs (vs 2,012), 168 steps.
  | checkpoint | overall | svamp | mbpp | ifeval | simple | multiple | no-fit |
  |---|---|---|---|---|---|---|---|
  | lpb-l2s50e-25 | +41.0 / -0.9 | +52 / -2.5 | +43 / -0.6 | +33 / -0.6 | +32 / +2.5 | +30 / -1.2 | +38 / -1.3 |
  | lpb-l2s50e-50 | +46.1 / -0.1 | +54 / -1.2 | +50 / -2.5 | +41 / +3.8 | +34 / +1.2 | +37 / -1.9 | +41 / -0.6 |
  | lpb-l2s50e-75 | +47.9 / -2.0 | +55 / -2.5 | +50 / -3.7 | +47 / -0.6 | +34 / +1.2 | +38 / -5.0 | +42 / 0.0 |
  | lpb-l2s50e-100 | +49.3 / -0.0 | +52 / -2.5 | +56 / 0.0 | +50 / +5.0 | +32 / +0.6 | +38 / -8.1 | +42 / 0.0 |
  | lpb-l2s50e-150 | +50.9 / -0.5 | +58 / -1.2 | +55 / -2.5 | +51 / +3.1 | +34 / -3.1 | +39 / 0.0 | +44 / -1.3 |
  Best screen so far: +50.9% / -0.5 (step 150), +49.3 / 0.0 (100). More pairs on easy questions helped. It replaces
  lpb-l2s50 in STEP 8: lpc-l2s50e (GPU 1, after lpb-right), 3 seeds, fresh-set soups at 50/100/150. GPU 2 started
  lpc-adar1 (3 seeds, soups 25/50/75).
- 18:00 lpb-right (right>wrong of similar length only, no protection; the accuracy baseline): step 25 -9.3% saved (LONGER) / -0.5; step 50 -14.6% (longer) / +2.5 (svamp +3.1, mbpp +3.8, no-fit +3.1, ifeval +1.9). Pure right>wrong pairs make the model think ~15% longer and a bit more accurate: the length pressure in the other recipes is what saves tokens. GPU 2: lpc-adar1 seed 0 at step 75/194 (trains the full epoch per seed; only soups 25/50/75 are scored).
- ~18:30 (user: "yes, go to ~160 per type for v2's reference"): stage `refsets` (queued on GPU 2 after lpc-adar1):
  two more random sets of 40 questions per part (seeds 20261024 / 20261025; never from the fresh set, questions_40 or
  the choosing set), answered x4 by the original Qwen -> runs/mvtf/v4-base-20261024 / -20261025. With the choosing set
  and questions_40: 160 per part. Only aggregate statistics are read; the fresh set stays untouched. In round 2, v1
  answers the same sets (refsets --rs-model <v1> --rs-tag v1) and select_pool / pool_check / lupin_select take all four
  as reference.
- ~19:00 OPTIONAL module `codecut` (user: "add it as an optional module, make sure it doesn't spill on the others; if it
  takes accuracy down, don't pursue it"), queued on GPU 2 after refsets: code questions of the pool solved >= half the
  time (600), label_cuts.py on each one's longest right answer (6 paragraph-end cuts x 4 answers), targeted_pairs.py
  code-only with --safe 1.0 (all 4 answers from the cut pass) and >= 25% shorter; 20% of lpb-l2s50e's code pairs
  swapped for cut pairs, every other type's pairs identical. Same DPO / checkpoints, choosing set x4. Automatic verdict
  vs lpb-l2s50e at the same steps: KEEP only if code thinks >= 3% less AND overall accuracy >= -1 AND code accuracy
  >= -2.5 AND non-code accuracy >= -1 (no spill); otherwise DROPPED and not pursued.
- 18:20: lpb-right-75: -39.2% saved (thinks 39% LONGER) / +0.9 (simple -56%, ifeval -44%): pure right>wrong pairs inflate thinking more and more. Confirms the length pressure is what saves. GPU 2: lpc-adar1 seed 0 step 140/194.
- 19:27: lpb-right-100: -51.0% (thinks 51% longer) / +2.1. lpb-right done (screen complete: 6 variations). GPU 1: lpc-l2s50e seed 0 training (step 155/170); GPU 2: lpc-adar1 seed 1 (step 145/194).
- ~19:45 (user: "confirm them on another 240 eval questions to cross-check accuracy before more training"): stage
  `crosscheck`: each confirmed soup (lpc-adar1 25/50/75 on GPU 2 after refsets; lpc-l2s50e 50/100/150 on GPU 1 after
  its confirm), 3 seeds averaged, answers questions_ref20261024 (40 per part, never fresh / choosing / questions_40) x4,
  scored vs the original Qwen on the same set (refsets). Report only: the fresh set decides v1; a large disagreement
  between fresh and cross-check is a red flag before round 2.
- 20:30 (user): deployment added to the roadmap (LUPIN_SPEC.md 'Deployment'): after the final model, serve it with vLLM on a hosted L4 (HF Inference Endpoint / RunPod / Modal, ~$0.5-1/h, user approves spending), monitor /metrics, side-by-side demo vs the original Qwen.
- 21:00 FIRST FRESH CONFIRM: lpc-adar1-soup-25 (3 seeds, fresh 240 x 8): +43.0% saved / -1.9 (svamp +62/-4.4, mbpp +45/-1.3, ifeval +26/+0.6, simple +30/-2.2, multiple +34/-2.5, no-fit +30/-3.1). The screen's +48.5 / +0.4 (1 seed, choosing set) did not hold: winner's curse + set difference. Soups 50 / 75 next; lpc-l2s50e seed 1 at step 85/170 on GPU 1.
- ~21:30 Why michal-bida (47.0 / -0.78, one round, same long2short idea) keeps accuracy better: his per-domain cap
  (shorten <= right-beats-wrong pairs). Our raw pairs had it, but lupin_select's matching re-drew pairs by cell and
  always-solved cells give only shorten pairs: matched sets were shorten:be-right ~2.2:1 (lpb-l2s50 1,382 vs 630) and
  ~3:1 (lpb-l2s50e 2,070 vs 658), i.e. 2-3x his length pressure. lupin_select.py --max-shorten-ratio (lupin
  --lp-balance) keeps the balance AFTER matching, per part. Queued (user: yes): lpb-l2s50bal (long2short p<0.5,
  balance 1) on GPU 1 after the l2s50e confirm + cross-check; screened on the choosing set like the others.
  Also learned (their repos): michal-bida did it in ONE round; Lightning McQwen3 = terse-prompt distillation SFT -> cut
  pairs (9k) -> a math-only right>wrong DPO (732 pairs) for accuracy. Both lost on SVAMP (-2.4 / -3.8).
- ~21:45 (user: "build a very strong competitive v1 first, with the data we have"): v1 levers, no new data:
  (1) lpb-l2s50bal (michal-bida balance after matching; GPU 1, queued); (2) math protection: lupin_pairs.py
  --min-solve-part svamp=0.8 + --right-parts svamp (Lightning McQwen3's math-only right>wrong), lpb-l2s50math (GPU 2, after
  codecut); (3) strength dial stage `dial` (soup x0.5 / x0.7, fresh x8): dial-adar1 on lpc-adar1 soup 25 (GPU 2, after its
  cross-check); (4) best dose = the screens' checkpoints; (5) combine the winners into one recipe -> 3-seed fresh
  confirm -> LUPIN v1. Tested on synthetic answers (math: shorten 74 -> 26, right 94 -> 167).
- ~22:00 (user: 'I don't want to just reproduce bida's work'): lpb-l2s50bal replaced by OUR per-type pressure, lpb-l2s50press: after matching, shorten pairs per be-right pair capped per type from each type's measured accuracy cost (math 1, code 2, instructions 3, tools 2); lupin_select.py --shorten-ratio-part (lupin --lp-pressure). Queued on GPU 1 in the same slot.
- ~22:15 lpc-adar1 FRESH CONFIRM complete (3 seeds, fresh 240 x 8):
  | soup | overall | svamp | mbpp | ifeval | simple | multiple | no-fit |
  |---|---|---|---|---|---|---|---|
  | 25 | +43.0 / -1.9 | +62 / -4.4 | +45 / -1.3 | +26 / +0.6 | +30 / -2.2 | +34 / -2.5 | +30 / -3.1 |
  | 50 | +44.8 / -2.3 | +57 / -3.1 | +52 / -1.3 | +32 / -2.2 | +32 / -2.2 | +36 / -2.5 | +34 / -3.7 |
  | 75 | +44.6 / -3.5 (outside) | +55 / -4.7 | +48 / -0.9 | +38 / -5.6 | +32 / -0.6 | +36 / -4.1 | +36 / -3.1 |
  Ada-R1 on fresh: ~44% saved at -2 to -3.5; the screen (+48.5 / +0.4) was optimistic. Saving plateaus after step 25
  while accuracy keeps dropping: step 25 is its best point. Losses are broad (math, tools, no-fit), not one type.
  Dial (x0.5 / x0.7 on soup 25) queued. lpc-l2s50e finished its 3 trainings; its fresh soups (50/100/150) next on GPU 1.
- ~22:45 v1 experiment set built and queued (user: "build all, order P1, two-stage, per-type adapters, P3, gentle, P2, P4"),
  all on the l2s50e recipe, screened on the choosing set x4 (1 seed, checkpoints 25-150):
  GPU 1 (after lpc-l2s50e fresh + cross-check + lpb-l2s50press): lpx-p1tilt (benchmark shares math 15 / code 25 /
  instructions 35 / tools 25), lpt-l2s50e (stage `pertype`: one LoRA per benchmark on its own l2s50e pairs, merged as
  the plain sum and with per-type strengths math 0.5 / code 0.8 / instructions 1 / tools 0.8), lpx-gentle (beta 0.3,
  NLL 2, lr 1e-5), lpx-p4tools (BFCL longest-thinking cells x2).
  GPU 2 (after dial, codecut, lpb-l2s50math): lpx-2stage (starts from lpb-right step 50, the "be right" round, +2.5 acc),
  lpx-p3medium (solve bins: medium x2, always-solved x0.5), lpx-p2math (math pressure 1:1 + math right>wrong pairs).
  New options: lupin_select.py --part-share / --solve-tilt / --think-tilt-bfcl; lupin --lp-init / --lp-beta / --lp-nll /
  --lp-lr / --lp-epochs; stage pertype. Selection options tested on synthetic answers (shares exact).
  Pair inventory (12,000 questions x 10 answers, long2short p<0.5, before matching): be-shorter svamp 2,587 / mbpp 1,744
  / ifeval 1,671 / simple 1,114 / multiple 514 / no-fit 718; guard 660 / 898 / 660 / 270 / 195 / 132; right>wrong
  (rightwrong recipe) 637 / 1,180 / 890 / 317 / 253 / 207. Enough for these experiments; tools' be-right pairs are the
  tightest (top-up with more BFCL-style questions only if the winner is tools-limited).
- ~23:00 (user): tools top-up added: stage tooltopup (1,500 BFCL-split tool candidates from dp9n-all, not in the pool, answered right only sometimes, topped up to 10 answers, ~2-3 GPU-h) + lpx-toolsplus (l2s50e recipe + those questions via lupin --lp-extra-q; same eval matching), queued at the end of GPU 2.
- ~22:05 (Mac restarted ~21:57; results read from the Drive logs):
  lpc-l2s50e FRESH CONFIRM (3 seeds, fresh 240 x 8):
  | soup | overall | svamp | mbpp | ifeval | simple | multiple | no-fit |
  |---|---|---|---|---|---|---|---|
  | 50 | +44.1 / -2.7 | +54 / -5.9 | +50 / -1.3 | +37 / -0.3 | +33 / +0.9 | +35 / -4.4 | +32 / -6.6 |
  | 100 | **+45.8 / -1.3** | +54 / -1.9 | +53 / +0.3 | +41 / -1.6 | +32 / +0.6 | +36 / -2.8 | +34 / -4.4 |
  | 150 | +45.4 / -3.1 (outside) | +57 / -5.6 | +50 / 0.0 | +39 / -3.4 | +31 / -0.9 | +34 / -4.1 | +34 / -4.7 |
  Best fresh point so far: lpc-l2s50e-soup-100 +45.8 / -1.3 (michal-bida 47.0 / -0.78). Screen had said +49.3 / 0.0.
  No-fit tools (-4.4 to -6.6) and math are the weak types.
  lpc-adar1 CROSS-CHECK (new 240 x 4 vs the original Qwen): soup 25 +48.6 / -3.4; soup 50 +46.5 / -3.4 (fresh: 43.0 / -1.9,
  44.8 / -2.3): same picture, accuracy -2 to -3.5 on both independent sets.
- ~22:30 NO-FIT FIX (user: "fix no-fit at the data level, no probe"): no-fit lost -4.4 to -6.6 (l2s50e) and -3.1 to -3.7
  (Ada-R1) on every fresh result; the pool's no-fit questions were easier than BFCL's (88% vs 79%) and the farthest in
  topic. (1) tooltopup --tt-parts bfcl_irrelevance --tt-hard: up to 800 verified no-fit candidates not in the pool that
  the original model gets right only sometimes, HARDEST first (lowest solve rate), topped up to 10 answers (nofit1);
  (2) lpx-nofit: l2s50e recipe + those questions (--lp-extra-q nofit1, same eval matching, so harder no-fit fill the
  eval's harder cells) + no-fit protection like math's: --lp-min-solve-part bfcl_irrelevance=0.8 (no shortening on no-fit
  questions solved < 80%) + --lp-right-parts bfcl_irrelevance (right > wrong pairs). Queued on GPU 2 right after the dial
  (before codecut).

### 2026-10-29 ~22:45 — user: data upgrade first, then iterate (rules x proportions x pairs), soup only the strong ones
- DATA UPGRADE (parts the pool made too easy): nofit1 = up to 800 hardest sometimes-right no-fit candidates; tools1 =
  1,000 hardest sometimes-right one-function / multiple candidates (2:1); code1 = 800 hardest sometimes-right code
  candidates (pool code was 60% vs MBPP 53%); each topped up to 10 answers by the original Qwen (tooltopup, --tt-hard).
  lupin --lp-extra-q takes a list and waits for each top-up on Drive.
- NEW BASELINE lpu-base = l2s50e recipe on pool + nofit1 + tools1 + code1, with no-fit protection (p < 0.8 no shortening
  + right>wrong pairs). Every experiment after it uses the upgraded data (lpu-*): per-type adapters, P1 tilt, gentle DPO,
  P4 tools tilt (GPU 1); two-stage, per-type pressure, math+no-fit protection, P3 medium tilt, codecut (GPU 2).
  Older queued runs on the old data (lpb-l2s50press, lpx-*, lpb-l2s50math, tools-plus, codecut on l2s50e) superseded.
- DECISION RULE per screen (choosing x4, 1 seed), vs lpu-base at the same checkpoints, averaging each run's two best
  neighbouring checkpoints (less luck): WORTH 3 seeds + soup on the fresh set if (a) accuracy >= base and saving >= base
  +2 pts, or (b) saving >= base -2 pts and accuracy >= base +1; otherwise DROP and try another strategy. Winners'
  levers are then combined.
- ~22:40: lpc-adar1 cross-check soup 75: +48.9 / -3.0 (25: +48.6 / -3.4; 50: +46.5 / -3.4). GPU 2 on dial-adar1; GPU 1 on crosscheck lpc-l2s50e. Both runtimes alive (Drive logs).
- 23:45: both runtimes had disconnected at ~22:03 (Mac restart); reconnected (L4), Drive mounted (user approved), setup running, plans restart from Drive state (GPU 1: crosscheck cx-l2s50e / tooltopup tools1; GPU 2: dial-adar1 / tooltopup nofit1). ~1.5 h lost.

### 2026-10-30 ~00:15 — user: representative of the WHOLE eval (iterate the data step), log + plot every result, let all run
- Matching reference widened: references() = the original Qwen on choosing + questions_40 + refsets 20261024/20261025
  (160 questions per part, never the fresh set; aggregate statistics only); used by poolup's select/check and by
  lupin_select for every new run.
- DATA LOOP: stage `poolup dp12` (GPU 1 after tools1 + code1; waits for nofit1 from GPU 2): candidate set = pool dp10 +
  the three top-ups, each question's 10 newest answers; select_pool.py against the 160-per-part references (4 thinking
  bins, 8 for simple); held-out check on new answers (seed 29, 200 per part x4) with pool_check.py against the same
  references. All lpu-* runs now train on dp12 (--lp-pool dp12 --lp-answers-pool dp10) and only if its check passes
  (--lp-gate: ALL OK or near misses). If it fails, the data step is iterated (dp13...) before any training.
- Results table: every score line also appends a row to runs/bon/results.csv (name, run, base, saved, accuracy, per part),
  for plots (dashboard).
- No early stop: every queued experiment runs; v1 is chosen at the end from all results (user). "Combine levers" = a run
  with the flags of two or more experiments that each beat the baseline, then a 3-seed fresh confirm.
- 00:30: top-ups smaller than planned (few unused candidates are answered right only sometimes): nofit1 = 189 no-fit questions (all available; done, 2 more answers each); tools1 = 724 (simple 667 + multiple 57; running). GPU 2 on dial-adar1. If dp12's check still fails on no-fit / multiple, the next data iteration needs NEW hard candidates (e.g. another Haiku-verified Glaive no-fit batch closer to BFCL).
- ~01:20: dial lpc-adar1-soup-25 x0.5 (fresh 240 x 8): +23.1% / +0.3 (svamp +35/-0.9, mbpp +20/+1.6, ifeval +20/+2.8, simple +16/0.0, multiple +19/-3.4, no-fit +12/-3.1): half the change keeps accuracy but halves the saving (x1: +43.0 / -1.9). x0.7 next. tools1 done (724); code1 running (800: 672 need 2 answers, 128 need 6).
- ~01:40 (user: 'ready the haiku ones'): new HARD no-fit batch on the Mac: nofit_verify.py --hard --rank 4,15 (functions of the same family as the one the question needs, past the exact synonyms; trial 150 -> 48 kept on NO+NO, e.g. tip -> calculate_tax, F-to-C -> get_temperature, Fibonacci -> calculate_combinations); full run 8,000 candidates (~$1) running. Colab side ready: prepare_types source glaive_nofit_hard, stage newtop (prepare -> build_pool length bins + 8-gram check -> 10 answers, written as tooltop-<name>), poolup --pu-extras configurable.
- ~02:00 hard no-fit batch done (Mac, Haiku 5.5, ~$1): 8,000 candidates at ranks 4-15 -> NO+NO 2,441 kept (UNSURE 2,788,
  YES 1,039, mixed 1,732). nofit_glaive_hard.jsonl committed. DATA LOOP iteration 2 queued on GPU 2 after the dial:
  newtop nofit2 (up to 1,500 after the BFCL length bins + 8-gram check, 10 answers each), then poolup dp13 (dp10 + nofit1
  + tools1 + code1 + nofit2 vs the 160-per-part references; held-out check seed 31). All lpu-* runs now target dp13
  (gated). dp12's check (GPU 1) still runs, for comparison.
- ~02:20: dial lpc-adar1-soup-25 x0.7 (fresh): +31.3% / -1.6 (x0.5 +23.1 / +0.3; x1 +43.0 / -1.9): the dial trades saving for accuracy roughly linearly; Ada-R1 does not reach the target at any strength. GPU 2 restarted on the updated plan (it had started lpu-2stage on the old dp12 line): now newtop nofit2 -> poolup dp13. GPU 1: code1 nearly done, then poolup dp12.

### 2026-10-30 ~02:45 — SPARSE arm C (Wanda) on the Mac (user: "run it on CPU in the meantime", "only if done properly")
- wanda_prune.py: Wanda (|W| x input-feature norm, lowest share zeroed per output row), block by block IN ORDER on the
  outputs of the already-pruned blocks, the 7 linear layers of the 28 blocks only; calibration = 128 of Qwen's own right
  answers (dp11 check, mixed types, prompt + reply, <= 1,024 tokens); fp32 on the M3 GPU (MPS). Sanity: block-by-block
  pass == full model (max difference 0.0). Damage check (proxy, no generation): reply NLL of held-out right answers
  per type + right>wrong preference on held-out pairs; output layer computed in chunks (the full 151k-vocab logits
  swapped the 8 GB Mac). Smoke test (4 calib, 50%): 50.0% of block weights zero; right answers 2-4x less likely.
- Full run started: levels 10/20/30/40/50%, 80 right answers + 30 pairs per type, pruned models saved (bf16).
- Real eval on the Mac: MLX speed test, Qwen3-0.6B: 1 answer 50 tok/s, 4 in parallel 156, 8 in parallel 209 tok/s
  (16+ out of memory: the KV cache is ~0.11 MB per token). An eval of 240 x 4 (~700k tokens) ~ 1 h per model. Next:
  the competition's own code (ThinkingCap-Test: recipe + graders) + the choosing-set questions locally, the harness
  checked against the GPU on the original Qwen, then original + the 2-3 best Wanda levels.
- ~03:20: newtop nofit2 crashed at its last line (missing import; my bug) after answering its 552 questions x10 (saved). 552 of the 2,441 Haiku-verified hard no-fit passed BFCL's prompt-length bins + 8-gram check (1,333 dropped: Glaive prompts are shorter than BFCL's). Fixed; newtop nofit2 re-queued on GPU 1 right after code1 (resumes from the saved answers); GPU 2's poolup dp13 waits for it.

### 2026-10-30 ~04:00 — data loop results (160-per-part references) + Mac Wanda stopped
- poolcheck-dp12 (dp10 + nofit1 + tools1 + code1): 33/36. Fails all on no-fit THINKING too long: median 156 vs 128
  [122,133] +22%, p75 215 vs 173 +24%, reply +18% (accuracy fine 85 vs 87). Note: with 160 reference questions BFCL
  no-fit accuracy is 87%, not the 79% the 80-question reference suggested: part of the earlier "too easy" was reference noise.
- poolcheck-dp13 (+ nofit2: 552 hard Haiku-verified no-fit after BFCL length bins / 8-gram): 33/36, no-fit ALL OK.
  Misses: bfcl_simple accuracy 78 vs 85 [80,89] (2 pts outside; ToolACE simple 76%, Glaive 84%), svamp p75 578 vs 759
  [591,976] (2% outside, near), ifeval accuracy 59 vs 52 [45,58] (1 pt outside, near). Gate closed: the lpu-* runs skipped.
- Iteration 3: poolup dp14 = dp13 without ToolACE one-function questions (poolup --pu-drop), check seed 33; all lpu-* runs
  re-pointed to dp14 (gated). GPU 1's plan cell restarted (its queue had ended).
- Mac Wanda (full run) STOPPED after 2 h: the 8 GB Mac swapped (7.9 GB swap, ~80 MB free RAM): the fp32 model + MPS
  caches + 128 calibration hidden states did not fit with the browser running. Moved to a GPU stage `wanda` (end of
  GPU 1's queue): Wanda 10-50% + damage check + the 384-calibration coverage check (mask overlap per layer type at 30%).
- ~05:00 poolcheck-dp14 (no ToolACE simple): 34/36: simple accuracy now OK but its thinking too short (median 120 vs 142
  [136,150] -16%, p75 142 vs 190 [177,208] -25%): Glaive one-call questions are easier AND shorter than BFCL's, ToolACE's
  harder and longer. Iteration 4: dp15 = dp13's candidates keeping ToolACE simple except its hardest (solved < 50% on
  10 answers; poolup --pu-drop-hard), check seed 35, on GPU 2 (now first in its queue); every lpu-* run re-pointed to
  dp15 (gated). GPU 1 runs the Wanda stage meanwhile.
- lpc-l2s50e cross-check (new 240 x 4 vs the original Qwen): soup 100 +50.5 / -2.7; soup 150 +52.5 / -1.5 (fresh: 45.8 /
  -1.3 and 45.4 / -3.1).

### 2026-10-30 ~05:40 — SPARSE arm C (Wanda) on GPU 1: damage map + coverage check
Damage check (proxy, no generation; 128 calibration answers = Qwen's own right answers; held-out 80 right answers per
type): increase of the per-token surprise (NLL) of Qwen's own right answers vs the unpruned model:
| pruned | svamp | mbpp | ifeval | simple | multiple | no-fit |
|---|---|---|---|---|---|---|
| 10% | +2% | +1% | +1% | +2% | +2% | +2% |
| 20% | +9% | +7% | +7% | +14% | +14% | +12% |
| 30% | +33% | +23% | +27% | +50% | +55% | +44% |
| 40% | +90% | +65% | +70% | +131% | +132% | +121% |
| 50% | +217% | +149% | +156% | +306% | +321% | +281% |
Reading: 10% is almost free, 20% mild, 30% clearly felt, 40-50% heavy; the TOOLS parts are the most fragile, code the
least. The right>wrong preference proxy is uninformative (the unpruned model itself prefers the right answer only 41-60%
of the time by mean surprise: right and wrong answers of the same question are about equally "natural"); ignore it.
COVERAGE CHECK (user: is 128 calibration enough?): Wanda at 30% with 384 calibration answers vs 128: the pruned weights
are the same for 99.7% (q/k/v/gate/up), 99.1% (o_proj), 98.2% (down_proj): 128 is enough.
Next (queued at the end of GPU 1, after LUPIN): wandaeval: real eval of 10 / 20 / 30% pruned on the choosing set x4
(accuracy and thinking length per type).

### 2026-10-30 ~06:30 — data loop: dp15 33/36; loop stopped (check noise), decision left to the user
- poolcheck-dp15 (dp13 candidates minus ToolACE one-call solved < 50%): 33/36: svamp accuracy 81 vs 88 [84,92] (-6.8),
  bfcl_simple p75 156 vs 190 (-18%), ifeval p75 366 vs 435 (-16%).
- Pattern over dp12-dp15 (160-per-part references): every pool lands at 33-34/36 and the misses MOVE between parts from
  one check to the next; math was not changed at all between dp12 and dp15 yet its check accuracy went 84 -> 81. A 200-
  question x4 check swings +-3-7 pts per part: further re-picking would chase noise.
- Closest pools: dp13 (no-fit fixed; misses simple accuracy 2 pts outside, svamp p75 and ifeval accuracy near misses)
  and dp14 (34/36). Options for the user: (a) accept dp13 as the training pool (gate overridden, logged); (b) re-check
  dp13 with a bigger, less noisy check (400 questions per part) before deciding; (c) keep iterating.
- Meanwhile (GPUs otherwise idle): GPU 2 runs lpv-base = the baseline recipe on dp13, UNGATED and labelled provisional;
  GPU 1 skips its gated lpu-* lines and goes to wandaeval (Wanda 10/20/30% real eval).
- ~07:45 lpv-base (PROVISIONAL baseline on dp13: l2s50e recipe + no-fit protection; 86 DPO steps) on the choosing set x4:
  | step | overall | svamp | mbpp | ifeval | simple | multiple | no-fit |
  |---|---|---|---|---|---|---|---|
  | 25 | +46.3 / -1.7 | +58 / -5.0 | +47 / -3.7 | +43 / +3.1 | +31 / -1.2 | +33 / -1.9 | +39 / 0.0 |
  | 50 | +43.0 / -1.9 | +52 / -0.6 | +47 / -4.4 | +34 / 0.0 | +32 / +0.6 | +36 / -6.2 | +39 / -1.9 |
  | 75 | +43.2 / +0.2 | +50 / -1.2 | +51 / +0.6 | +34 / +3.1 | +32 / +2.5 | +38 / -3.7 | +39 / -3.8 |
  Same range as lpb-l2s50e's screen on dp10 (+41-51% / -2 to 0): the upgraded pool did not change the screen picture
  (screen noise +-1.5 overall). Fewer pairs (86 steps vs 168): the matching against 160 references fills less.
- wandaeval crashed: the pruned model folders had no tokenizer files (my bug). Fixed (wanda_prune.py --save also saves
  the tokenizer) and re-queued on GPU 1.

### 2026-10-30 ~08:15 — USER DECISION: accept dp13 (option A)
dp13 = pool dp10 + top-ups nofit1 / tools1 / code1 / nofit2, re-picked against the 160-per-part references; check 33/36
(no-fit fixed; simple accuracy 2 pts outside its interval; svamp p75 and ifeval accuracy near misses). Gate overridden by
the user. Baseline = lpv-base (already screened: +46.3 / -1.7 at 25, +43.0 / -1.9 at 50, +43.2 / +0.2 at 75).
Experiments queued on dp13, each vs lpv-base with the decision rule (WORTH 3 seeds + fresh soup if acc >= base and saving
>= base +2, or saving >= base -2 and acc >= base +1): GPU 2: lpw-p1tilt, lpw-2stage, lpw-press, lpw-math; GPU 1 (after
wandaeval): lpw-pertype, lpw-gentle, lpw-p3medium, lpw-p4tools.
- ~08:50 SPARSE arm C REAL EVAL (wandaeval; choosing set x4 vs the unpruned Qwen; "saved" negative = thinks LONGER):
  | pruned | overall | svamp | mbpp | ifeval | simple | multiple | no-fit |
  |---|---|---|---|---|---|---|---|
  | 10% | +1.8% / +0.2 | +11 / -0.6 | +10 / +1.9 | -17 / -0.6 | +3 / +1.9 | -1 / -1.9 | +1 / 0.0 |
  | 20% | -7.7% / -2.0 | -5 / +1.3 | -5 / -5.6 | -21 / -2.5 | -3 / -3.1 | -10 / -0.6 | +9 / 0.0 |
  | 30% | -8.9% / -1.9 | +2 / +1.3 | -10 / -5.6 | -17 / -1.9 | -5 / -2.5 | -15 / -1.2 | -14 / -0.6 |
  Reading: 10% of the block weights go for free (accuracy +0.2, thinking unchanged); at 20-30% accuracy drops ~2 pts
  (code -5.6 the most) and the model THINKS ~8-9% LONGER (instructions ~20% longer), as published for pruned reasoning
  models. Consistent with the damage map (10% ~free, 30% clearly felt). Next SPARSE step (not queued, user decides):
  E = pruned model + the LUPIN pairs (does an emptier model learn short thinking?), or finer levels 10-20%.
- GPU 1's queue had ended right after wandaeval without seeing the lpw-* lines (pushed meanwhile); cell restarted.
- ~10:20 first experiment scores (choosing x4; vs lpv-base 25 +46.3/-1.7, 50 +43.0/-1.9, 75 +43.2/+0.2):
  lpw-p1tilt (math 15 / code 25 / instructions 35 / tools 25): 25 +41.7 / -1.0; 50 +45.6 / -2.9 (mbpp -7.5): no better yet (75 pending).
  lpw-pertype-sum (4 per-type adapters, each trained on its own type's pairs, simply added): +77.2% / -9.2 (svamp -21.3,
  no-fit -13.8): adding four full-strength adapters stacks four pushes; far too strong. The tuned merge (math 0.5, code
  0.8, instructions 1, tools 0.8) is being scored; a weaker merge (e.g. all x0.3-0.5) would be the natural next try.
- ~11:20 experiment scores (choosing x4; baseline lpv-base 25 +46.3/-1.7, 50 +43.0/-1.9, 75 +43.2/+0.2):
  lpw-p1tilt: 25 +41.7/-1.0, 50 +45.6/-2.9, 75 +48.7/-1.8 (best two 50-75 avg +47.2/-2.4 vs base +43.1/-0.9: +4 saved for
  -1.5 accuracy) -> DROP by the rule (trades accuracy for saving, no better trade-off).
  lpw-pertype-tuned (math 0.5, code 0.8, instr 1, tools 0.8): +70.5 / -6.5 (svamp -13.7, multiple -10.6): still too
  strong; a soft merge (0.25 / 0.35 / 0.45 / 0.35, no retraining) queued on GPU 1.
  lpw-gentle (beta 0.3, NLL 2, lr 1e-5): 25 +14.4/+0.1, 50 +29.1/-1.7 (slower, 75 pending).
  lpw-2stage (from the "be right" round): 25 +22.4/+1.9, 50 +24.3/+1.2: more accurate than the original but saves half
  as much (75 pending).
  Reading so far: every variant sits near the same saving-vs-accuracy line (~1 pt of accuracy per ~3-4 pts of saving);
  none moves off it yet.
- ~12:30 (choosing x4; baseline lpv-base 25 +46.3/-1.7, 50 +43.0/-1.9, 75 +43.2/+0.2):
  lpw-gentle: 25 +14.4/+0.1, 50 +29.1/-1.7, 75 +25.7/-0.9 -> DROP (saves ~half as much, accuracy not better).
  lpw-2stage: 25 +22.4/+1.9, 50 +24.3/+1.2, 75 +29.2/-0.8 -> DROP by the rule (saving -20 pts); note: starting from the
  "be right" round banks ~+2 accuracy, but 86 shortening steps are not enough to reach the baseline's saving.
  lpw-press (per-type pressure: math 1:1, code/tools 2:1, instructions 3:1): 25 +37.1/-1.0, 50 **+43.0/+0.4** (same
  saving as the baseline at 50, +2.3 accuracy); 75 pending: candidate for WORTH if 50-75 holds vs the baseline's 50-75.
- ~12:45 (user): every score line now also prints the FULL-EVAL weighting: inside BFCL the parts re-pooled 400 / 200 / 240
  (tokens weighted by each part's base mean), then averaged over the 4 benchmarks as compare.py does (track.py
  full_weighted). Re-weighted key results (from the logged per-part numbers): lpc-l2s50e-soup-100 fresh +45.8 / -1.3 ->
  +46.1 / -1.2; lpw-press-50 choosing +43.0 / +0.4 -> +42.9 / +0.5; lpv-base-50 choosing +43.0 / -1.9 -> +42.5 / -1.7.
- ~13:00 lpw-press finished at 2 checkpoints (25 +37.1/-1.0, 50 +43.0/+0.4); judgment call (logged): it passes rule (b) at checkpoint 50 (same saving as the baseline at 50, +2.3 accuracy), the two-checkpoint average fails only because step 25 is early. CONFIRM queued on GPU 2 after lpw-math: lpc-press, 3 seeds, fresh soups at 50 / 75. lpw-p3medium 25: +34.7 / -1.7 (50, 75 pending).
- ~13:10 (user: more steps of per-type pressure?): lpw-press had ~72 steps (the pressure cap removes many shorten pairs), so only checkpoints 25/50. Queued: lpw-press2ep (2 epochs ~145 steps, checkpoints 25-125, GPU 1 next) to see whether saving keeps growing with accuracy held; the lpc-press confirm now saves every 10 steps and scores soups at 40/50/60/70.
- ~13:20 (user: 'won't 2 rounds over the same pairs overfit?'): lpw-press2ep replaced by lpw-press6k: 1 epoch over up to 6,400 DISTINCT matched pairs (lp-total 6400 instead of 3,200; the pool's raw pairs are 4-5x what a run keeps), checkpoints 25-125; no pair repeated. Truly new data comes with round 2 (v1's own answers on new questions).
- ~13:40 (user): v1 GATE written into LUPIN_SPEC.md (fresh: better than -0.8 at >= 45% saved, full-eval weighting; second-set cross-check within ~1 pt; no type below -3 on both; saving agrees within ~3 pts). No v2 before a v1 passes. Cross-check cx-press queued after lpc-press (GPU 2).
- ~14:40 (choosing x4; baseline 25 +46.3/-1.7, 50 +43.0/-1.9, 75 +43.2/+0.2):
  lpw-math (math + no-fit protection: no shortening below 80% solved + right>wrong pairs on both): 25 +40.8/-0.1,
  **50 +50.9/-0.6**, 75 +48.3/-0.9 -> best two (50-75) avg +49.6/-0.75 vs baseline +43.1/-0.85: +6.5 saved at equal
  accuracy -> WORTH (rule a). Confirm lpc-math queued on GPU 1 (3 seeds, fresh soups 40/50/60/70), then lpw-pressmath
  (combined: per-type pressure + math/no-fit protection) screen.
  lpw-p3medium (medium-difficulty tilt): 50 +39.4/-2.7, 75 +41.1/-1.9, 100 +43.4/-0.8 -> DROP.
- ~15:50 lpw-press6k (per-type pressure, ~1,700 distinct pairs, 107 steps; choosing x4): 50 +44.6/-1.7, 75 +44.7/+0.7,
  100 +39.4/-1.7 (full-eval weighting within 0.2): more distinct pairs did NOT raise the saving above ~45%; accuracy
  wobbles +-1.5 around -0.5 -> no better than lpw-press; per-type pressure seems to plateau at ~44-45% saved.
- FIRST FRESH CONFIRM POINT of per-type pressure: lpc-press-soup-40 (3 seeds, fresh 240 x 8): +45.5% / **-5.2** (svamp
  -8.1, ifeval -5.6, multiple -4.7, mbpp -3.8): far below its screen (+43.0 / +0.4 at step 50, 1 seed, choosing set). A
  ~5-pt screen-to-fresh gap (l2s50e's was ~1.3): the screen's +0.4 was likely a lucky single run on the choosing set.
  Waiting for soups 50/60/70 before concluding. Red flag watch: math -8.1 at this checkpoint.

### 2026-10-30 ~14:00 — lpc-press cross-check; researcher dossiers written (user)
- lpc-press cross-check (refs 20261024, 240 x 4): soup-40 +51.0% / -2.1 (full-eval weighting +50.8 / -2.2); soup-50 +48.9% / -2.8
  (+48.8 / -2.7). Same soups on the fresh set x8: -5.2 and -2.9. The same model differs by 3.1 pts between two 240-question
  sets: question-draw noise at 40 per part is larger than the ~0.8 sampling noise measured earlier.
- Timing (GPU 2 log): fresh 240 x 8 ~13 min, 240 x 4 ~5 min; the official full evaluation (2,759 x 8) is estimated ~2.5 h.
- lpc-math: seed 0 trained (86 steps), seed 1 training.
- User asked for neutral dossiers for an external researcher: LUPIN_ROUND1_DOSSIER.md (LUPIN only) and
  THINKINGCAP_FULL_DOSSIER.md (every track from the start, compiled from this log, the per-type clone, specs and handoffs).

### 2026-10-30 ~15:00 — full evaluation speed; build spec Run A built and queued (user)
- Full evaluation of the original Qwen (GPU 2) started ~14:10: 512 of 22,072 answers by 14:34 (chunks of 256 answers,
  ~10 min each). If that pace holds, ~18 h per model, not the ~2.5 h estimated; re-measured over the next hours.
- External build spec (THINKINGCAP_BUILD_SPEC.md, user's Downloads) reviewed. User decisions: build the per-type pair mix
  and the decaying learning rate, test small, run Run A with 3 seeds after lpc-math; Run C (math cut pairs) NOT built.
- Code: lupin_select.py --kind-share-part (per part, each pair kind its own share, each kind cell-matched on its own);
  dpo.py --lr-schedule cosine (--warmup 10, --lr-min 2e-6; set per step, so a resumed run keeps the schedule);
  track.py --lp-kind-share / --lp-lr-schedule. Synthetic test: part shares exact (15/30/30/11.9/5.9/7.1%), kinds per part
  as asked where pairs exist; a kind short of pairs lowers that part's fill, and the existing fill-ratio trim then
  scales every part down (real fill printed in the run's summary).
- Run A = lpc-runa (GPU 1 after lpc-math): part shares svamp 15 / mbpp 30 / ifeval 30 / bfcl 25 (400/200/240 inside);
  kinds svamp guard 40 right 60; mbpp & ifeval shorten 70 guard 15 right 15; simple shorten 60 guard 20 right 20;
  multiple shorten 30 guard 30 right 40; no-fit guard 40 right 60. min-solve for shorten: mbpp/ifeval 0.3, simple 0.5,
  multiple 0.8, svamp/no-fit 0 (guard on every solved question; the existing rule also limits guard by min-solve, so
  multiple's guard pairs are only on questions solved >= 0.8 — a small difference from the spec). Right pairs on all
  parts; easy-ratio 1.2; shorten cap off; 4,800 pairs; cosine lr; 3 seeds; checkpoints every 10; fresh soups at
  100/150/200/250/300.

### 2026-10-30 ~15:20 — lpc-math fresh soups 40/50 (3 seeds, fresh 240 x 8)
| soup | overall | svamp | mbpp | ifeval | simple | multiple | no-fit | full-eval weighting |
|---|---|---|---|---|---|---|---|---|
| 40 | +45.6 / -3.6 | +60 / -5.6 | +51 / -2.8 | +32 / -2.5 | +33 / -2.8 | +36 / -4.4 | +33 / -3.4 | +45.5 / -3.6 |
| 50 | +43.8 / -3.5 | +59 / -4.7 | +43 / -3.4 | +36 / -3.7 | +32 / -1.2 | +35 / -2.8 | +33 / -2.8 | +43.7 / -3.5 |
Screen (choosing x4, 1 seed) had shown +50.9 / -0.6 at step 50. With math protection (min-solve 0.8 + right pairs on
svamp) SVAMP still saves ~60% (shorten pairs remain on questions solved >= 8/10). Full evaluation (GPU 2): 1,280 of
22,072 answers at 15:17 (~19 answers/min, ~19 h per model at this pace).

### 2026-10-30 ~16:00 — user: train the spec's runs, compare at matched saving, check the 720 screen set's bias
- Full evaluation of a second soup (lpc-press) dropped; only the original Qwen + lpc-l2s50e-soup-100 get one.
- The 720 screen set is matched on the ORIGINAL Qwen's full run, so its match for the original is true by construction;
  the real bias test is a TRAINED model: lpc-l2s50e-soup-100's score on the 720 vs on the full evaluation. GPU 2:
  full-base -> l2s50e full -> subset check -> screenset -> subset check again with questions_screen720 included.
- Run B (lpc-runb): Run A + svamp kinds shorten 30 / guard 30 / right 40, the shorten pair = MEDIAN-length right >
  longest right (>= 1.5x) on questions solved >= 0.8 (lupin_pairs.py --median-shorten-part; synthetic test: chosen
  ~850 tokens vs rejected ~1,800, i.e. the median, not the shortest). Run D (lpc-rund): Run A's pairs, beta 0.2, lr
  1e-5. Both queued on GPU 1 after Run A, before lpw-pressmath / lpw-p4tools. Run C not built (user).
- Decision rule from now on: runs compared by accuracy at the same tokens saved (soups at 100-300 steps give each run a
  saved-vs-accuracy curve), not at the same step.

### 2026-10-30 ~16:10 — lpc-math soups 60/70; Run A stopped and restarted as lpc-runa2
- lpc-math fresh x8: soup-60 +40.1 / -3.5 (svamp +56/-5.9, multiple +35/-5.9, no-fit +33/-4.4); soup-70 +43.7 / -3.1
  (svamp +59/-5.6, multiple +35/-3.7, no-fit +33/-3.1, mbpp +46/-2.5, ifeval +33/-1.6, simple +33/-0.9).
- lpc-runa pair selection kept 2,070 of 4,800 pairs (127 steps): no-fit had 148 of its 343 (guard 78, right 70); the
  fill-ratio trim then cut every part to 43%. Other parts were 94-98% filled (svamp 678/720, mbpp 1,389/1,440, ifeval
  1,354/1,440, simple 560/571, multiple 268/286). Stopped at seed 0 step 25.
- Fix: lupin_select.py --loose-parts (track --lp-loose): a listed part takes its pairs without cell matching and does
  not set the fill ratio. Restarted as lpc-runa2 with --lp-loose bfcl_irrelevance; Runs B and D get the same flag.
- 16:20: lpc-runa2 pairs: 4,442 kept (shorten 2,265, guard 1,017, right 1,160), fill ratio 0.94; no-fit 266/343
  (guard 137, right 129), part shares 15.2 / 30.4 / 30.x / 12.0 / ~6 / ~6%. ~275 steps per seed.
  Full evaluation of the original Qwen: 2,560 / 22,072 at 16:20 (~20 answers/min; ETA ~08:45 on 31 Oct).

### 2026-10-30 ~19:15 — original Qwen full evaluation done (full-base, 2,759 x 8); our small sets vs the full evaluation
Full evaluation sizes: svamp 1,000, mbpp 378, ifeval 541, simple 400, multiple 200, no-fit 240. Original Qwen: svamp
88.7% / 677 thinking tokens, mbpp 52.4 / 2,079, ifeval 58.7 / 433, simple 82.2 / 171, multiple 76.1 / 192, no-fit 83.4 /
186 (accuracy over 8 samples, mean thinking tokens per answer). Random draws of 40 questions per part (4,000 draws),
5-95% range of the original's accuracy: svamp 82-94, mbpp 42-63, ifeval 49-68, simple 74-90, multiple 68-84, no-fit 75-91.
Each of our sets, original's accuracy (percentile among random draws) / thinking (percentile):
| set | svamp | mbpp | ifeval | simple | multiple | no-fit |
|---|---|---|---|---|---|---|
| fresh | 88 (p47) / 614 (p27) | 45 (p13) / 2112 (p57) | 69 (p97) / 447 (p63) | 88 (p91) / 169 (p47) | 75 (p41) / 180 (p17) | 79 (p17) / 168 (p16) |
| choosing | 92 (p84) / 589 (p19) | 48 (p21) / 2175 (p64) | 51 (p10) / 522 (p92) | 90 (p96) / 167 (p42) | 78 (p61) / 200 (p74) | 76 (p8) / 173 (p25) |
| questions_40 | 82 (p5) / 789 (p85) | 55 (p67) / 2163 (p63) | 53 (p19) / 468 (p75) | 82 (p50) / 160 (p19) | 81 (p83) / 194 (p59) | 85 (p65) / 180 (p38) |
| ref20261024 | 82 (p5) / 952 (p99) | 57 (p80) / 1933 (p33) | 48 (p4) / 408 (p36) | 82 (p53) / 163 (p29) | 79 (p75) / 202 (p79) | 95 (p100) / 169 (p17) |
| ref20261025 | 92 (p82) / 657 (p43) | 52 (p52) / 1938 (p33) | 60 (p59) / 374 (p15) | 79 (p28) / 176 (p68) | 68 (p5) / 186 (p35) | 86 (p70) / 199 (p78) |
The five sets do not overlap. Visual: new artboard "The full evaluation" in the Where We Shaved canvas
(https://claude.ai/artifact/GsnLfifSWasQh2DRZMY9eW). lpc-l2s50e-soup-100 full evaluation running on GPU 2 (5,376 /
22,072 at 18:58).

### 2026-10-30 ~20:25 — FULL evaluation of lpc-l2s50e-soup-100; 720 screen set built and checked
- lpc-l2s50e-soup-100, full evaluation 2,759 x 8 vs full-base: tokens saved +48.0%, accuracy -2.08 pts (full-eval
  weighting identical: +48.0 / -2.1). Same soup measured earlier: fresh 240 x 8 (own run) +45.8 / -1.3; cross-check
  (ref20261024, 240 x 4) +50.5 / -2.7.
- Recomputed from the full run's answers on each set's questions (subset_check.py): choosing +47.7 / -1.17, questions_40
  +50.7 / -2.11, ref24 +51.3 / -1.51, ref25 +47.3 / -3.80, fresh +44.7 / -2.73, screen720 +49.9 / -2.29.
  Random 240-question draws (5,000, 40 per part): saved 5-95% [+44.7, +51.3], accuracy [-4.09, -0.18], sd 1.19 pts.
  The fresh set's own run (-1.3) and the fresh questions taken from the full run (-2.73) differ by 1.4 pts: sampling
  noise of the answers on top of the question-draw noise (the evaluation's 256-batching changes generations).
- questions_screen720 (screen_set.py, fresh set excluded; multiple has 119 because one cell lacked a free question):
  original Qwen on it vs full: svamp 89.1 / 88.7, mbpp 52.7 / 52.4, ifeval 58.8 / 58.7, simple 82.4 / 82.2, multiple
  76.4 / 76.1, no-fit 83.2 / 83.4; output tokens within +-3% except mbpp -4.8%. For lpc-l2s50e: +49.9 / -2.29 on the
  720 vs +48.0 / -2.08 full (saving +1.9 high, accuracy -0.2).
- Seen in the soup's full-run output: some replies degenerate into repeated "1. 1. 1. ..." and text in another script
  (to be counted per part).
- GPU 2 queue ended. Run B moved to GPU 2 (with its 720 rescreen); GPU 1: Run A2 -> rescreen 720 -> Run D -> rescreen.
- 21:51: lpc-runa2 3 seeds trained (275 steps each). Fresh x8 soup-100: +31.0 / -0.5 (svamp +34/-0.3, mbpp +33/+0.9,
  ifeval +30/-1.6, simple +27/-1.9, multiple +28/-1.9, no-fit +25/+0.3). SVAMP saves 34% with no math shorten pairs.

### 2026-10-30 ~22:25 — michal-bida control queued (user)
- Question: is the gap to michal-bida (full eval 47.0 / -0.78 vs our lpc-l2s50e 48.0 / -2.08) the recipe or the data?
  lpc-bida = his recipe on our pool dp13 (10 answers per question): long2short (guard = shortest right > longest wrong or
  capped; shorten = shortest right > longest right if >= 1.5x), shorten capped at the guard count per part (1:1), no
  min-solve rule, no easy-ratio, NO lupin_select matching (track --lp-no-match: the raw pairs as they come). Our training
  settings (LoRA r32, beta 0.1, lr 2e-5 constant, NLL 1.0, 1 epoch), 3 seeds, fresh soups + 720 rescreen at
  100-300. Replaces lpw-pressmath / lpw-p4tools on GPU 2 (after Run B). Not reproduced from his side: his own data mix
  (GSM8K 70 / MetaMath SV 15 / FOBAR 15 for math) and his training settings.
- 22:35 (user): the control keeps the eval's difficulty mix: lpc-bida now uses lupin_select matching (160-per-part
  references, cells solve x thinking), then michal-bida's 1:1 balance re-imposed per part after matching (--lp-balance 1:
  at most one shorten pair per be-right pair), no-fit loose, target 4,800 pairs; still no min-solve rule.
- 22:59: lpc-runa2 fresh x8 soups: 100 +31.0 / -0.5; 150 +28.7 / -1.5; 200 +27.9 / -0.2; 250 +23.3 / -0.9. Saving FALLS
  with more steps (cosine lr; 26% right + 23% guard pairs). 720 rescreen started (soup-100 first).

### 2026-10-30 ~23:40 — lpc-l2s50e-soup-100 full evaluation per type (computed from both full runs' records)
| part | n | saved | acc | broke / fixed (>=6 -> <=2 of 8) | by thinking fifth (saved / acc), shortest -> longest |
|---|---|---|---|---|---|
| svamp | 1000 | 55.9 | -3.2 | 14 / 3 | 30/+0.1, 34/-0.9, 40/-4.2, 52/-6.7, 69/-4.4 |
| mbpp | 378 | 51.9 | -2.2 | 1 / 1 | 35/-2.7, 41/-2.3, 49/-3.8, 59/-4.1, 53/+2.0 |
| ifeval | 541 | 45.8 | -1.6 | 6 / 3 | 33/+1.0, 39/-0.3, 36/-4.4, 45/-2.9, 53/-1.4 |
| simple | 400 | 33.4 | -0.8 | 6 / 3 | 16/+0.5, 20/-0.9, 25/-1.2, 33/-3.9, 51/+1.9 |
| multiple | 200 | 36.6 | -0.2 | 3 / 3 | 21/+3.4, 30/+0.9, 32/-1.6, 36/-1.6, 50/-2.2 |
| no-fit | 240 | 40.9 | -3.3 | 3 / 0 | 16/0.0, 20/-0.5, 30/-1.3, 39/-3.1, 61/-11.5 |
Repeated-text replies (same short token repeated 40+ times at the end): ifeval 10 (original Qwen: 13), mbpp 1 (1), others 0,
so the degenerate replies seen in the log are not new with training. Capped replies: svamp 139 -> 1, mbpp 236 -> 28, ifeval
156 -> 52. Visual: canvas section "Trained models, measured without the noise".
- 23:16: lpc-runa2-soup-100 on questions_screen720 x 4: +33.5 / +0.5 (full-eval weighting +33.4 / +0.4); svamp +33/-1.5,
  mbpp +32/+1.9, ifeval +36/+0.4, simple +31/+1.3, multiple +34/+2.9, no-fit +33/-1.2. (Fresh x8 had said +31.0 / -0.5.)
  Canvas page rebuilt simply ("Are our tests like the real test?": the sets, one set vs the full test, training data vs
  the full test, trained models measured on the full test or the 720 only).
- 00:41 (31 Oct): lpc-runa2-soup-150 on the 720 x 4: +32.4 / -0.1 (full-eval weighting +32.3 / -0.1); svamp +24/+1.0,
  mbpp +37/-0.4, ifeval +34/-1.5, simple +33/+0.2, multiple +36/+2.7, no-fit +33/-1.5.
- ~00:45 (31 Oct): both runtimes stopped progressing. GPU 1 page shows "Reconnect L4" (it was scoring lpc-runa2-soup-200
  on the 720, 256 / 2,876 answers); GPU 2 page no longer answers scripts (Run B seed 1 at step ~205 / 275, last Drive log
  step 195). Not reconnected (standing rule: ask the user before opening or reconnecting a Colab runtime). On reconnect:
  run cells 1-6; dpo.py resumes after its last checkpoint (every 10 steps), finished evaluations are skipped.
- ~11:00-12:45 (31 Oct): runtimes reconnected by the user. Both restarted plans began RETRAINING lpc-runa2-s0 and
  lpc-runb-s0 from scratch (the lupin stage checked the local disk, empty on a fresh runtime, before restoring adapters
  from Drive). Stopped at step 75: only checkpoints step-10..70 (and the resume state) of those two seeds were
  overwritten on Drive; the final adapters and steps 80+ are intact. Fix: the lupin stage light-restores every seed's
  adapter folder before checking. GPU 1 plan: lpc-runa2 line commented (done), so it starts with the 720 rescreen.
- ~13:05: fix 155fc81 (seed adapters and the fresh start restored from Drive before the check). Both runtimes re-cloned it
  and are re-running setup; then GPU 1 starts the 720 rescreen of lpc-runa2 (soups 200/250/270 left), GPU 2 lpc-runb
  (seed 0 done, seed 1 resumes from its Drive checkpoint ~step 200).
- ~13:25: after the interrupt + cell re-run, both runners started with "drive: NOT mounted" (the mount was gone): GPU 1
  skipped the 720 rescreen (no adapters), fullmodel failed (models/rc1-rl-40 not found) and lpc-rund was starting; GPU 2
  started lpc-runb locally. Both interrupted before any training; cells 1 (drive.mount), 3-6 re-queued; the user
  approves the Drive popups.
- ~14:30: lpc-runb seed 1 resumed after step 190 with loss 8-10 (was ~0.6) and 56-62% pairs ordered right. Cause: the
  restarted stage rebuilt the pair files (raw + matched) instead of restoring them; the rebuilt file's rows came out in
  another order, and dpo.py reused the seed's saved reference scores (ref.json, matched by position) on the new rows.
  Interrupted at step ~230. Fixes: (1) the lupin stage restores pairs-<name>-raw and pairs-<name> from Drive before
  building; (2) dpo.py fingerprints the pair rows into ref.json and refuses to resume on a mismatch (old-style ref files
  with a resume point also refuse); (3) track --lp-reset-seeds: delete a damaged seed (local + Drive) once, with a marker.
  lpc-runb seed 1 is retrained from scratch (seed 0 trained before the restart, unaffected). GPU 1 (720 rescreen of
  lpc-runa2) uses finished adapters only: unaffected.
- 14:44: lpc-runa2-soup-200 on the 720 x 4: +33.0 / -0.3 (full-eval weighting +32.9 / -0.4); svamp +21/-0.2, mbpp
  +34/-0.4, ifeval +43/-1.0, simple +32/-1.2, multiple +35/+3.6, no-fit +32/-0.4. GPU 2: lpc-runb seed 1 reset (marker
  written), retraining from scratch.
- 15:44: lpc-runa2 on the 720 x 4 (full-eval weighting): soup-250 +32.3 / -1.0 (svamp +17/0.0, ifeval +45/-3.5); soup-270
  +32.5 / +0.4 (svamp +16/+1.0, mbpp +33/+1.0, ifeval +46/-1.0, simple +32/-0.2, multiple +34/+3.4, no-fit +30/-1.0).
  All five Run A2 soups sit at 32-33% saved, accuracy -1.0..+0.4: math saving falls with steps (33 -> 16%), IFEval rises
  (36 -> 46%). GPU 1 now on the rc1-rl-40 full evaluation; GPU 2 lpc-runb seed 1 step 160/275 (loss 0.56, healthy).
- 16:00 (user): Run D replaced by lpc-runac = Run A's pairs and kind mix, constant lr 2e-5 (no cosine decay), beta 0.1 (no
  stronger tie to the original); 3 seeds, fresh soups + 720 rescreen. Single change vs lpc-runa2: the learning-rate
  schedule. The michal-bida 1:1 control (lpc-bida) stays queued on GPU 2 after Run B.
- 16:44: rc1-rl-40 FULL evaluation (2,759 x 8): +55.4 / -2.6 (fresh x8 had said +53.9 / -2.1); svamp +65/-4.5, mbpp
  +60/-3.0, ifeval +49/-2.3, simple +41/-0.6, multiple +43/-1.2, no-fit +52/-0.2. Its 720 screen is next on GPU 1.
  GPU 2: lpc-runb seed 1 done (retrained), seed 2 at step 10.
- 17:44: rc1-rl-40 on the 720 x 4: +55.2 / -3.7 (full-eval weighting) vs its FULL +55.4 / -2.6: the 720 read accuracy 1.1
  pts low for this model (0.2 low for lpc-l2s50e); saving exact. So a 720 score carries about +-1 pt of accuracy error.
  GPU 1: lpc-runac seed 0 step 105/275. GPU 2: lpc-runb seed 2 step 200/275.
- 18:44: lpc-runb 3 seeds trained; fresh x8 soup-100 +23.8 / -0.4 (svamp +24/-1.9, mbpp +19/+1.2, ifeval +29/-0.9, simple
  +23/0.0, multiple +26/-2.8, no-fit +20/+2.5) — lower saving than lpc-runa2's fresh soup-100 (+31.0 / -0.5); 720
  rescreen after the fresh soups. GPU 1: lpc-runac seed 0 done, seed 1 training.
- ~19:30 (user): lpc-runb seeds are not on one pair draw (seed 0: the pre-restart lupin_select draw; seeds 1-2: the draw
  rebuilt at the restart; same counts and kind mix, different questions picked). Queued: lpc-runb --lp-reset-seeds 0
  (seed 0 retrained on the current draw; the run's fresh / 720 / cross-check soup scores are deleted and re-scored), then
  its 720 rescreen. lpc-runa2's three seeds share one (pre-restart) draw; lpc-runac's three seeds share its own draw.
  Fresh x8 lpc-runb (mixed draws) soups: 100 +23.8 / -0.4; 150 +24.5 / +0.3; 200 +26.5 / +0.2.
- ~19:45 pair-file check (user: "dig into the pairs"; files downloaded from Drive): pairs-lpc-runb vs pairs-lpc-runa2 (the
  current draws): both 4,442 pairs; per part and kind counts equal except svamp (A2: guard 285 / right 390; B: shorten 199
  median-vs-longest / guard 203 / right 273); median chosen vs rejected lengths equal on every other part; summed length
  push (sum of log(rejected/chosen) over shorten + guard) per part equal except svamp (B 375 vs A2 298, i.e. B pushes math
  MORE). 2,272 identical pairs (normal for two draws). merge.py averages per-adapter updates (B_i A_i), so different
  LoRA inits do not cancel. Pairs cannot explain B's lower saving. Queued on GPU 2 before the seed-0 retrain: single
  seeds on the 720 at step 200 (lpc-runb s0, s1; lpc-runa2 s0).
- 20:44: GPU 2 picked the lpc-runb seed-0 reset line (commit 52d4d6f) before the single-seed check lines were pushed:
  the pre-restart seed 0 is already deleted (step 185/275 of its retrain on the current draw). The single-seed lines
  stay and will score the NEW seed 0, seed 1 (both on the current draw) and lpc-runa2 seed 0 at step 200 on the 720.
  lpc-runb fresh soup-250: +23.3 / -0.8 (lpc-runa2 fresh soup-250: +23.3 / -0.9). GPU 1: lpc-runac seed 2 started.
- 21:44: clean lpc-runb (all 3 seeds on the current draw) fresh x8 soup-100: +24.0 / -0.5 (svamp +26/-0.3, mbpp +19/-0.9,
  ifeval +28/-1.6, simple +25/+1.3, multiple +27/0.0, no-fit +14/+1.6) — same as the mixed-draw soup (+23.8 / -0.4): the
  restart / mixed draws were NOT the cause of B's lower step-100 saving. GPU 1: lpc-runac seed 2 step 195/275.
- 22:44: lpc-runac (Run A pairs, constant lr) fresh x8 soup-100: +26.2 / -1.3 (svamp +28/-1.9, mbpp +26/+1.6, ifeval
  +25/-3.1, simple +26/-2.5, multiple +29/-3.4, no-fit +22/+0.9); lpc-runa2 soup-100 was +31.0 / -0.5 on the same set.
  Clean lpc-runb fresh soup-250: +24.7 / +0.1 (full-eval weighting). GPU 2 on the single-seed 720 checks.
- 23:44: single seeds on the 720 x 4 at step 200 (full-eval weighting): lpc-runb s0 +27.6 / -0.3 (mbpp +17), lpc-runb s1
  +26.6 / +1.0 (mbpp +18), lpc-runa2 s0 +33.9 / -0.3 (mbpp +33). B's seeds agree with each other and are ~6-7 pts below
  A2's seed, mostly on code, whose pairs are the same in both runs. lpc-runac fresh x8: 150 +24.9 / -0.6, 200 +22.0 /
  -0.9, 250 +19.5 / -0.8 (svamp 19 -> 4 -> 1%; mbpp 23 -> 20 -> 17%). Pattern: every run trained AFTER the 31 Oct restart
  (B, A-constant) saves less than lpc-runa2 (trained before it). Queued on GPU 1: lpc-runa2r = exact replay of lpc-runa2
  (same flags, 1 seed, current pair draw and code), scored at step 200 on the 720.
- 00:44 (1 Nov), 720 x 4 (full-eval weighting): lpc-runac 100 +29.3 / -0.7; 150 +34.3 / -0.9 (ifeval +44/-3.7); 200
  +29.2 / -0.4 (mbpp 19, svamp 19). Clean lpc-runb 100 +29.8 / -0.6; 150 +29.0 / -0.6. (lpc-runa2 on the 720: 32-33 at
  every step, accuracy -1.0..+0.4.) Added to the canvas.
- 01:01: 720 x 4 (full-eval weighting): lpc-runac 250 +26.3 / -1.2 (svamp +7, mbpp +17, ifeval +43/-3.5); clean lpc-runb
  200 +30.0 / -0.1 (ifeval +42/-1.9, mbpp +22). Proposed to the user: blends of lpc-l2s50e-soup-100 and lpc-runa2 soups.

### 2026-11-01 ~01:30 — michal-bida's repo read in full (user: "double check")
Source: github.com/michal-bida/ThinkingCap-Test, branch thinking-compression (train_dpo_offline.py, queue/a10-train-Db.sh,
reports/FINDINGS.md sections 17-20, reports/compare_Db-s8.txt). His submitted model is arm Db:
- Data: 11,327 prompts in 4 domains (~3,000 each): code KodCode 2,989; tools 2,986 = ToolACE simple 641 / multiple
  1,004 / decline 365 + xlam irrelevance 976 (45% of tools are "no function fits"); IFEval-style 2,385; math GSM8K
  2,152 + MetaMath SV 614 / FOBAR 201. Prompt length matched per decile to each benchmark; 8-gram decontamination;
  base behaviour checked against the benchmarks (accuracy code 53.8 vs 51.5, ifeval 63.7 vs 58.1, math 85.6 vs 88.5,
  tools 78.9 vs 81.2). 4 samples per prompt (45,308 samples, ~25 h); 9,337 prompts with a right sample.
- Pairs: per prompt, chosen = shortest right sample (of 4); rejected = the longest wrong/truncated sample, and the
  longest right sample if >= 1.5x the chosen (at most 2 pairs per prompt). Db: per domain, long pairs capped at the
  number of wrong pairs -> 5,862 pairs (code 941 / ifeval 724 / math 696 / tools 570 of each kind). No matching to
  benchmark cells, no difficulty rule, no similar-length right>wrong pairs.
- Training: standard DPO, beta 0.1, NO SFT/NLL term ("RPO" listed as not run), reference = base (adapter off), LoRA
  r32 alpha 64 all linear, lr 2e-5 with cosine schedule and 5% warmup, 16 pairs per step, 1 epoch (~366 steps), fp16
  autocast, max length 8,704, ONE seed, the final adapter evaluated (no checkpoint picking).
- Full eval x8: +47.0 / -0.78; per benchmark: no-fit +46% / +6.04 (the model declines more), multiple +42 / -1.12,
  simple +35 / -1.41, ifeval +55 / -0.28, mbpp +33 / -1.22, svamp +56 / -2.41. His ablations: wrong pairs only
  +23.0 / +0.81; long pairs only +83.6 / -13.0; both (D, unbalanced) +50.2 / -0.41 (2 samples). Not run by him: RPO,
  iterative second round.
Differences from all our LUPIN runs: our NLL anchor (weight 1.0), 10 answers per question (more extreme shortest
chosen), matching to eval cells, difficulty rule, similar-length right pairs, 3-seed soups, checkpoint picking, and
his tools domain being ~45% no-fit (ours ~29% of BFCL pairs). track.py: an --lp-answers other than 10 now writes its
own answers file (it would have overwritten the 10-answer one).
- ~01:45 (user): the michal-bida control is now an EXACT copy, lpc-bidax: long2short (his two pair kinds), shorten
  capped at the guard count per part (lupin_pairs default --shorten-cap 1 = his --balance), no min-solve, no right pairs,
  --lp-no-match, --lp-nll 0, --lp-answers 4 (the 4 newest of each question's 10 answers; own answers file), cosine lr 2e-5
  (warm-up 10 steps vs his 5%; floor 2e-6 vs 0), beta 0.1, LoRA r32, 16 pairs/step, 1 epoch, 1 seed. rescreen step 0 =
  the final adapter. Scored on the 720 at steps 100 / 200 / 300 / final. Our pool dp13 (12,000 questions) is the data.
- 01:44: 720 x 4 (full-eval weighting): lpc-runac 270 +23.4 / -0.1 (svamp 0%, mbpp 11%); clean lpc-runb 250 +31.0 / +0.1,
  270 +28.4 / +0.1. GPU 1: lpc-runa2r (exact one-seed replay of lpc-runa2) training; GPU 2: lpc-bidax starting.
- 02:45: lpc-bidax stopped at step 140/266: its raw pairs held "right" pairs (no-fit 28, multiple 31, simple 38, ifeval
  162, mbpp 147, ...) because track.py's --lp-min-solve defaults to 0.5. Restarted as lpc-bidax2 with --lp-min-solve 0
  (no difficulty rule, as michal-bida). bidax pairs before the fix: 4,247 (guard = shorten per part, e.g. mbpp 674/674,
  ifeval 472/472, svamp 405/...).
- 04:44: lpc-runa2r (exact replay of lpc-runa2, seed 0) on the 720 x 4 at step 200: +26.0 / -0.8 (svamp +14/-0.8, mbpp
  +17/-1.7, ifeval +38/-1.5, simple +31/-1.0, multiple +34/+1.5, no-fit +33/+3.1) vs the original lpc-runa2 seed 0 at
  step 200 +33.9 / -0.3 (scored after the restart, so scoring is not the difference). Checked identical: the answers
  file (11,999 questions x 10, same sources and counts), the pair selection (kept 4,442: shorten 2,265 / guard 1,017 /
  right 1,160; fill 0.94; mix 0.5099,0.2290,0,0.2611), the training code (dpo.py's only change adds a resume check),
  vLLM 0.23.0 before and after (transformers / peft pinned). Same seed, same flags. Remaining explanation: training
  run-to-run randomness (GPU non-determinism) large enough to move saving by ~8 pts for one seed; lpc-runa2's three
  seeds all landed high (soups 32-33), the post-restart runs (B seeds 27.6 / 26.6, A-constant, replay) at 26-30.
- 05:44: lpc-bidax2 (exact michal-bida copy, 5,026 pairs, 315 steps) on the 720 x 4 (full-eval weighting): step 100
  +54.7 / -4.6 (svamp +69/-6.9, no-fit +46/-6.5, mbpp +51/-4.8, ifeval +51/-3.5); step 200 +49.1 / -3.1 (svamp +63/-5.2,
  no-fit +43/-3.3, mbpp +35/-2.5). Steps 300 and final next. GPU 1 queue ended with lpw-pertype soft (choosing x4):
  +44.6 / -1.3 (svamp +51/+0.6, mbpp +53/-3.7, ifeval +41/-0.6, simple +29/+0.6, multiple +32/0.0, no-fit +36/-5.0).
  Queued on GPU 1: per-type merges on the 720 (pertype --pt-screen) at soft / softer / mid strengths (no training).
- 06:44, 720 x 4 (full-eval weighting): per-type merges (lpw-pertype, 4 adapters trained on lpv-base pairs): softer
  (0.2/0.25/0.35/0.25) +39.2 / -0.8; soft (0.25/0.35/0.45/0.35) +48.4 / -1.9; mid (0.3/0.45/0.6/0.45) +55.0 / -2.6. Each
  loses most on no-fit (-4.8 / -6.9 / -7.1) and math (-2.1 / -4.0 / -5.6). lpc-bidax2: step 300 +51.1 / -2.6; FINAL
  +51.8 / -2.7 (svamp +65/-5.4, no-fit +46/-4.6, ifeval +50/-2.3, simple +41/-1.7, multiple +46/-1.7, mbpp +45/-0.4) vs
  michal-bida's own full-eval +47.0 / -0.78 (no-fit +6.0, svamp -2.4): his recipe on our pool does not reproduce his
  result; the gap is concentrated on no-fit and math.
- 07:44: lpw-pressmath (1 seed, choosing x4, noisy): step 25 +34.7 / -0.2, step 50 +42.0 / -0.8 (svamp +53/0.0, mbpp
  +43/-2.5, no-fit +35/-1.9). GPU 1 idle (queue ended), waiting on the user for the no-fit data step. GPU 2: lpw-p4tools.
- 08:44: lpw-p4tools (1 seed, choosing x4): 25 +41.9 / -0.1; 50 +37.6 / -0.3; 75 +45.9 / -0.6 (svamp +55/-0.6, mbpp
  +48/-3.1, no-fit +39/-0.6). Both queues ended; 720 checks queued: p4tools 75/50 (GPU 2), pressmath 50 (GPU 1).
- 09:44: lpw-p4tools on the 720 x 4 (full-eval weighting): step 75 +48.4 / -2.6 (ifeval +43/-6.0, svamp +57/-4.0);
  step 50 +42.2 / -2.2 — the choosing set had shown -0.6 / -0.3. GPU 1's runner had not started (cwd lost after the
  queue end + restart: "can't open /content/track.py"); cells 1, 3-6 re-run.
- 10:44: GPU 1's runtime disconnected "due to inactivity" (idle after its queue ended and the cwd restart); a Drive
  access dialog is pending. Not reconnected (rule: ask the user first). GPU 2 idle (queue ended after the p4tools check).

### 2026-11-01 ~11:00 — no-fit data step (user: yes)
- GPU 2: newtop nofit3 = up to 2,000 new no-fit questions from nofit_glaive_hard + nofit_glaive_verified (Haiku-verified
  NO+NO), prompt-length bins of the choosing set, 8-gram decontamination, and (new) none whose prompt is already in a
  pool-dp*.jsonl or an earlier top-up; answered x10 by the original Qwen. Then lpc-bidax3 = lpc-bidax2's exact
  michal-bida recipe with nofit3 added to the extra questions; 720 at 100 / 200 / 300 / final.
- GPU 1: lpc-bidax2 seed 1 (same pairs file as seed 0) -> how far one seed's final score moves by training randomness.
- 14:44: newtop nofit3: 888 candidates already in the pool / top-ups dropped; 958 new no-fit questions x10: mean solve
  rate 98% (always right 886, sometimes 72, never 0) -> almost no wrong answers, so almost no guard ("right > wrong")
  pairs. AND lpc-bidax3 did not include them: with --lp-answers-pool dp10 the stage takes only the pool's own question
  ids (top-ups must already be selected into the pool), so its pairs equal lpc-bidax2's (5,026; no-fit 92 / 92). Left
  running as a SAME-SEED repeat of lpc-bidax2 (pure GPU non-determinism measure). GPU 1: lpc-bidax2 seed 1 on the 720.
  Next (proposed): michal-bida's own no-fit source (xLAM irrelevance) if the base model gets it wrong often enough.
- ~15:00 (user: "the training data must include the right data"; "better to create our own dataset"): prepare_types
  xlam_irrelevance (MadeAgents/xlam-irrelevance-7.5k -> BFCL functions; 75 of 100 sample rows convert); track
  --lp-add-q (top-up sets that join the pool even with --lp-answers-pool). Queued on GPU 1: newtop nofit4 = 3,000
  questions from xLAM irrelevance + Glaive hard + ToolACE near-verified no-fit (dedup vs pool), x10 answers; then
  lpc-bidax4 = exact michal-bida recipe + nofit4 added; 720 at 100/200/300/final.
- 15:17: lpc-bidax2 SEED 1 final on the 720 x 4: +51.0 / -3.9 (svamp +61/-6.5, no-fit +46/-5.4, mbpp +49/-3.7, ifeval
  +49/-3.3, simple +42/-0.4, multiple +47/-1.1) vs seed 0 +51.8 / -2.7: same pairs, seed-to-seed spread ~1 pt saving,
  ~1.2 pts accuracy. newtop nofit4: 1,308 candidates dropped as already in pool/top-ups; 1,871 questions x10, mean solve
  rate 94% (always 1,580, sometimes 257, never 34). lpc-bidax4 pool 13,870 questions; pairs 5,362 (no-fit 260 guard /
  260 shorten vs 92 / 92 before; tools now 39% no-fit pairs). Training on GPU 1.
- ~15:50 (user): math data step queued on GPU 2 after the bidax3 repeat: prepare_types gsm8k_std (GSM8K train+test in
  SVAMP wording, labelled gsm8k_matched); newtop now answers math sources too. newtop math1 = 2,000 questions from
  gsm8k_std + metamath_sv (dedup vs pool), x10; lpc-bidax5 = michal-bida recipe + nofit4 + math1 (--lp-add-q); 720.
- 17:44, 720 x 4 (full-eval weighting): lpc-bidax3 = SAME pairs and SAME seed as lpc-bidax2 s0 (GPU non-determinism
  only): 100 +52.4 / -3.9 (bidax2 s0: +54.7 / -4.6); 200 +56.3 / -4.2 (bidax2 s0: +49.1 / -3.1); 300 +54.8 / -3.7
  (bidax2 s0: +51.1 / -2.6). lpc-bidax4 (own no-fit added) step 100: +57.1 / -5.1 (svamp +69/-8.8, mbpp +57/-9.0,
  no-fit +49/-4.0, ifeval +53/-0.4).
- 18:04: lpc-bidax3 (same-seed repeat) FINAL +54.5 / -3.9 (no-fit +47/-7.3, svamp +66/-6.5). lpc-bidax4 (own no-fit)
  step 200 +54.6 / -3.7: no-fit +50/-1.9 (repeat at step 200: -6.2; bidax2 s0: -3.3), ifeval -0.2, but svamp +67/-8.3,
  mbpp +51/-5.0. newtop math1: 3,722 candidates already in the pool / top-ups dropped; answering 3,000 x... (in progress).
- 18:23: lpc-bidax4 step 300 on the 720: +53.2 / -3.3 (no-fit +49/-0.6, svamp +67/-6.0, simple +44/-3.5, mbpp +46/-2.9,
  ifeval +50/-2.3, multiple +47/0.0). Final being scored. GPU 2: math1 answers in progress.
- 18:40: lpc-bidax4 FINAL on the 720 (full-eval weighting): +52.7 / -3.5 (no-fit +50/-1.2, svamp +66/-5.4, mbpp +45/-4.2,
  ifeval +51/-2.7, simple +43/-2.7, multiple +47/-0.2). GPU 1 queue ended -> lpc-bidax5 seed 1 queued (waits for math1).

### 1 Nov 2026 — section 5 of the visual + a 600-step check queued
- Visual section 5 "Where the cut happens" (720 set, against the original Qwen): per type thinking, accuracy, thinking per
  right answer, wrong answers' share of thinking; 5 length bands per type. Now has bidax4 and the bida copy (bidax2 s0) at
  steps 100/200/300/final. The math 4th length band (original thinks 471-905 tokens) loses 12.5-17.7 pts at EVERY step
  of both runs; bidax4 math/code recover 100->300 then flat (overall 100: 57.1/-5.1, 200: 54.6/-3.7, 300: 53.2/-3.3,
  final 52.7/-3.5): the later steps trade saving for accuracy along the same curve.
- User asked to check more steps anyway: GPU 1 queue, after bidax5 seed 1: lpc-bidax4l = the bidax4 line unchanged but
  2 passes over the pairs (one pass = ~336 steps), stopped at 600 (cosine over the 600), then rs-bidax4l on the 720 at
  300/400/500/600.
- 21:10: math1 analysed (2,000 GSM8K/MetaMath questions x10 on the original; solve 80%; sometimes 930 / always 964 / never 106).
  By the original's mean thinking: <350 tok 60 q, 350-470 353 q, 470-905 (the band bidax4 loses -13.5 on) 934 q
  (560 always / 356 sometimes), >905 653 q. Pairs the bida rule makes from math1 with 4 answers: band 470-905 =
  485 shorter-right>longer-right + 216 right>wrong, of which 175 also point SHORTER (shortest right < longest wrong) and
  only 41 point longer. So math1 under bida's rule adds ~660 "cut more" pairs exactly where we over-cut: bidax5 is not
  expected to close the hole. Natural "too-short answer was wrong" examples are rare (88 questions in that band; the
  original seldom thinks that short: answers at 150-299 tokens are 73% right vs 92% at 450-599).
- 21:20 (user): bidax5 STOPPED on both GPUs before its first DPO step (pairs had svamp guard 1,085 / shorten 1,085).
  New math recipe, data checked before any training (user: "make sure we have the correct math training data,
  targeting exactly where the model is failing BEFORE starting; if not enough, keep answering more questions"):
  - lupin_pairs.py --band svamp:470:905: no shorten pair on math questions whose original answers think 470-905 tokens
    on average (guard pairs stay; cut pairs above 905 stay, the cut is free there: +2.1).
  - --overcut: "don't over-cut" pairs, the original's median-length right answer > a finished wrong answer thinking
    < 0.75x it; losers from bidax4's own answers (track.py onpolicy: lpc-bidax4-s0 x4 on the band questions) plus the
    original's own short wrong answers ("natural"). Kind guard (tagged), added after the 1:1 shorten cap.
  - local test on math1 (4 answers, the original's other 6 answers as a stand-in loser set): band 919 questions, 500
    shorten pairs dropped, 44 natural + 46 stand-in overcut pairs (the original rarely thinks short: real bidax4 answers
    are the point).
  - Gate before training: >= 500 overcut pairs from the trained model. GPU 2: onpolicy om1 (math1 band); GPU 1: newtop
    math2 (2,000 more questions) + onpolicy om2, in case the gate needs them; then the 600-step check.
- 22:30: onpolicy om1 (lpc-bidax4-s0 x4 on math1's 934 band questions, 470-905 by the original's 10 answers): right
  80.5%, mean thinking 291 (the original ~600), 730 wrong answers (median thinking 231) on 375 questions. Pairs from math1
  (4 answers): 319 trained + 50 natural overcut (winner median 594 tokens, loser 251), band shorten pairs dropped 574,
  shorten left 380, guard 979. Fix: a question onpolicy chose stays in the band (the 4 kept answers measure it less well;
  246 -> 319). Gate 500 not met on math1 alone -> math2 + om2 (GPU 1), then lpc-bidax5m seed 0 there; lupin
  --lp-min-overcut 500 refuses to train under the gate. GPU 2 meanwhile: the 600-step check (lpc-bidax4l), then
  bidax5m seed 1.
- 2 Nov ~00:10: GATE PASSED. lpc-bidax5m pairs (math1 + math2 + om1 + om2): svamp in band 2,955 questions; overcut
  622 trained + 121 natural = 743; band shorten dropped 1,559; svamp shorten left 1,510 (easy + hard questions only),
  svamp guard 2,387 (incl. the 743). Other parts = bidax4's exactly (no-fit 260/260, multiple 174/174, simple 229/229,
  ifeval 681/681, mbpp 886/886). 8,308 pairs (bidax4 5,362): math is now ~47% of pairs (bidax4 ~18%); one pass = ~519
  steps, so the other types get the same pairs spread over a longer cosine. Training seed 0 on GPU 1; 720 at
  100/200/300/400/final.
- 2 Nov ~03:00: lpc-bidax4l (bidax4's exact 5,362 pairs, 2 passes cut at 600 steps, cosine over 600) on the 720:
  300: +38.4 / -1.1 (no-fit +48/+12.1, multiple +37/-3.6, simple +37/-9.0, ifeval +48/-2.1, mbpp +12/-0.6, svamp +47/-1.5)
  400: +38.8 / -1.8 (no-fit +50/+11.9, multiple +39/-4.8, simple +39/-10.2, ifeval +47/-3.3, mbpp +8/-1.7, svamp +49/-1.0)
  500: +41.8 / -1.7 (no-fit +50/+10.8, multiple +39/-6.1, simple +37/-11.2, ifeval +53/-0.2, mbpp +7/-1.9, svamp +54/-2.7)
  NOT bidax4 continued: same pairs and (first pass) same order, only the slower lr decay differs, yet at step 300 it
  saves 15 pts less (mbpp barely cut: 7-12% vs 45%) and the tools behaviour flipped (no-fit +11-12, one-function -9 to
  -11: it answers "no function fits" far more often). On the curve, ~40% / -1.7 is below the usual ~40% / -0.8: no gain
  from more steps. Step 600 still scoring. Added to the visual, with bidax4 final (missing until now).
- ~03:30 (user: fix the schedule): bidax5m's 8,308 pairs at 16/step stretch its cosine over ~520 steps (bidax4 ~333),
  and the 600-step check showed the schedule alone shifts code and tools a lot. lpc-bidax5s = the same pairs at 25 per
  step (~331 steps): bidax4's schedule step for step, so the math data is the only change. GPU 2 (step-600 screen of
  bidax4l stopped, the question answered): bidax5s seed 0 -> 720 at 100/200/300/final -> seed 1. GPU 1: bidax5m
  (stretched) finishes as a reference, final scored only.
  bida's published branch (michal-bida/ThinkingCap-Test@thinking-compression, 9e997bd) read: evaluate.py = the official
  one except a --gpu-memory-utilization flag and a Triton attention fallback; compare.py and utils identical; his final
  Db = train_dpo_offline.py samples/domains.graded.jsonl --balance, scored with the official 8-sample full evaluation
  (reports/compare_Db-s8.txt: 47.0 / -0.78; svamp 56.4/-2.41, mbpp 33.0/-1.22, ifeval 54.7/-0.28, simple 35.3/-1.41,
  multiple 41.9/-1.12, no-fit 46.1/+6.04).
- ~03:20 (user): full test of the exact bida copy, lpc-bidax2 seed 0 final (720: +51.8 / -2.7), queued on GPU 1 after
  bidax5m's training (fullsoup step 0 = the final adapter, now supported).
- ~03:40 (user): michal-bida's train_dpo_offline.py read for settings. Same as ours: DPO beta 0.1, LoRA r32 alpha 64
  dropout 0 on all 7 projections, AdamW 2e-5 no weight decay, 16 pairs/step, 1 epoch, final model. Different: float32
  base + float16 autocast + GradScaler (ours: bfloat16 base and autocast, LoRA weights float32), warm-up steps//20 and
  cosine to 0 (ours: 10 steps, to 2e-6), pairs over 8,704 tokens skipped (ours 9,000), no held-out pairs (ours 48).
  His Db pairs: 5,862 (wrong pairs 941 code / 724 ifeval / 696 math / 570 tools, long pairs capped to match). His own
  caveat: Db was designed after seeing D's scores on the same questions ("slightly optimistic").
  dpo.py --bida-exact = all of those; lpc-bidax6 = bidax4's pairs with --bida-exact, GPU 2 after bidax5s seed 0's
  screen. Not tested locally (no torch/peft on the Mac): the first GPU steps are the test.
- ~03:55 status before the night: GPU 1 lpc-bidax5m (stretched schedule) step 455/520 -> full test of lpc-bidax2 s0
  final -> bidax5m final on the 720. GPU 2 lpc-bidax5s (bidax4 schedule, gate 622 trained overcut pairs passed)
  training -> 720 at 100/200/300/final -> lpc-bidax6 (bida-exact settings) -> 720 -> bidax5s seed 1.
  Visuals (user): only models trained on the current data (dp13 + top-ups) and scored on the 720 set or the full test
  are shown. Removed: l2s50e (older dp10 pool) and the Fresh 240 set from "Where We Shaved"; "ThinkingCap Model
  Tree" rebuilt as a per-type table of those models only (older rounds, measured on small tilted sets, dropped).
- ~04:20 (user): per-type spill-over study queued on GPU 1 after its current queue. track.py typeonly: 6 adapters (math,
  code, instructions, tools one function / several / no-fit), each trained ALONE on its own part of lpc-bidax5s's pairs
  (svamp 3,897, mbpp 1,772, ifeval 1,362, simple 458, multiple 348, no-fit 520), bidax4 settings, seeds 0 and 1. Each
  2-seed soup scored on the whole 720 (final; step 50 too for the three big parts): own cut + spill-over on the other
  five. Next: per-type length-band breakdowns, then dial mixes of the 6 soups. Earlier evidence of spill-over: Run A2
  (no math / no-fit shorten pairs) still cut math and no-fit 33%; the old 4-adapter sum at full strength 77% / -9.2.
- ~04:40 (user) PLANNED, not queued: on-policy round. The best v1 answers the ~14k training questions x10 (~6 h on both
  GPUs), pairs rebuilt from its own answers (bida 1:1, math band + don't-over-cut from its own mistakes), round 2 from
  v1 (v1 = reference), kept only if it beats v1 at the same saving on the 720. Decide 3 Nov against the per-type study
  and the bida runs (bidax5s, bidax6, bida copy full test).
- ~09:00: FULL TEST of lpc-bidax2 s0 final (exact michal-bida recipe on our data), 2,759 x 8: +51.2 / -3.0
  (no-fit +44/-3.5, multiple +43/-2.0, simple +40/-2.2, ifeval +53/-1.1, mbpp +42/-2.4, svamp +64/-6.0). The 720 had
  read it +51.8 / -2.7: the 720 is accurate (saving 0.6 high, accuracy 0.3 low, as with l2s50e). So the gap to his
  47.0 / -0.78 is real, not our measurement: his recipe on our data loses 2.2 pts more at 4 pts more saving; math
  (-6.0 vs his -2.4) and no-fit (-3.5 vs +6.0) carry most of it. Left to test: settings (bidax6) and data (bidax5s).
  bidax5s at step 230/333.
- ~10:30: lpc-bidax5m s0 FINAL (math fix data, stretched 520-step schedule) on the 720: +11.3 / -3.6. svamp -56% (thinks
  56% LONGER) / -19.2; mbpp +1/+4.6; ifeval +38/-0.8; simple +31/-6.0; multiple +35/-1.3; no-fit +41/+10.2. Math from
  the records: mean thinking 1,053 (original 685, bidax4 232), capped 4.6% (hardest fifth 22%), and LESS accurate at
  every length fifth (86/84/75/62/46% vs original 99/99/93/95/65): longer AND worse, not "longer and right". Tools show
  the same refusal pattern as the 600-step check (no-fit up, one-function down): the stretched schedule's signature.
  Likely cause on math: the don't-over-cut pairs mix two models (winner = original Qwen text, loser = bidax4 text), so
  DPO can learn "write like the original, not like bidax4" instead of "think enough"; with no copying term this pulled
  math back past the original into rambling. bidax5s (same data, bidax4 schedule) will separate schedule from data.
- ~11:30: lpc-bidax5s (same data as bidax5m, bidax4's 333-step schedule) on the 720: step 100 +21.6 / +1.2 (svamp -13% /
  +0.2, mbpp +25/+4.8, ifeval +33/-1.5, simple +30/-1.0, multiple +34/+2.7, no-fit +36/+1.7); step 200 +20.1 / -0.2
  (svamp -36% / -1.0, mbpp +19/+3.3, ifeval +40/-3.1, simple +34/-2.9, multiple +38/+1.3, no-fit +42/+2.1).
  => math thinks LONGER than the original with the bidax4 schedule too: the math DATA (don't-over-cut pairs, winner =
  original text vs loser = bidax4 text) drives it, not the schedule. Tools stay sane here (no refusal flip). typeonly
  math adapter stopped before training; the other five parts train first; math waits for fixed pairs.
- ~12:40: lpc-bidax5s step 300 +21.8 / +0.1 (svamp -26%/+0.6, mbpp +19/+2.9, ifeval +41/-4.2, simple +33/-2.5,
  multiple +39/+2.5, no-fit +41/+3.3); FINAL +23.6 / +1.3 (svamp -22% / +1.0, mbpp +21/+3.5, ifeval +41/-0.6, simple
  +36/-2.5, multiple +40/+2.5, no-fit +42/+3.3). Math stays ~22-36% LONGER than the original but accurate; the other
  types are healthy (no-fit +3.3 is our best no-fit yet on bidax4's schedule). On the curve overall (~24% / +1.3), not
  above it. lpc-bidax6 (bida-exact) training on GPU 2: 339 steps of 16 (no held-out). GPU 1: typeonly code done-ish.
- ~15:00 (user): queued lpc-bidax5p on GPU 2 after bidax6's screens: bidax4's plain bida recipe + math1 + math2 (GSM8K /
  MetaMath, michal-bida's math sources) added, no band rule, no don't-over-cut pairs, one pass held at ~333 steps
  (dpo.py --target-steps: pairs per step set from the pair count). The clean test of whether his harder, more numerous
  math questions are what keeps his math at -2.4. (Was the original lpc-bidax5, stopped 1 Nov before training.)
- ~15:20: lpc-bidax6 (bidax4's pairs, --bida-exact settings) on the 720: step 100 +44.4 / -2.2 (svamp +53/-3.5, mbpp
  +40/-5.0, ifeval +41/+0.8, simple +39/-4.2, multiple +42/+0.2, no-fit +44/+1.2); step 200 +40.1 / -0.9 (svamp +45/-2.3,
  mbpp +27/-0.8, ifeval +45/0.0, simple +40/-2.3, multiple +42/-1.1, no-fit +44/+2.1). vs bidax4 (same pairs, our
  settings) 100: 57.1/-5.1, 200: 54.6/-3.7: his settings cut ~13-15 pts less and lose ~2.8 pts less; math -2.3 at
  45% (bida full: 56/-2.4) and no-fit positive. On or slightly above our curve (~40% / -0.9 vs pt-softer 39.2 / -0.8,
  600-step 41.8 / -1.7). Steps 300 and final next. typeonly: code, instructions, one function, several, no-fit trained.
- ~15:40 (user): next runs switched to michal-bida's settings: lpc-bidax5p -> lpc-bidax5q (plain bida + math1/math2,
  --bida-exact, 339 steps like bidax6: isolates his math questions against bidax6); per-type study again as lpt6b with
  --bida-exact (typeonly now passes --lp-dpo), queued on GPU 1 after lpt6's screens. --lp-dpo takes several dpo flags
  comma-joined (plan lines split on spaces). If bidax6's final disappoints, revert.
- ~16:30: lpc-bidax6 step 300 on the 720: +41.7 / -1.0 (svamp +50/-2.7, mbpp +25/-0.6, ifeval +47/-1.3, simple +38/-2.3,
  multiple +41/-0.4, no-fit +45/+4.6). Then BOTH runtimes disconnected ("Reconnect L4"): bidax6 final screen and the
  lpt6 per-type screens were in progress; waiting for the user's OK to reconnect (Colab rule).

### 2 Nov 2026, ~16:30 — summary of the day: runs, settings, queues

**Results (720 set unless FULL; tokens saved / accuracy change; all lpc-* runs on pool dp13 + top-ups)**

| run | what changed vs bidax4 | result | verdict |
|---|---|---|---|
| lpc-bidax2 s0 final, FULL test | (exact michal-bida recipe on our data, our settings) | 51.2 / -3.0 (720 read 51.8 / -2.7) | the 720 measures right; the gap to his 47.0 / -0.78 is real |
| lpc-bidax4l (600-step check) | lr cosine over 600 steps, 2 passes | 300: 38.4/-1.1, 400: 38.8/-1.8, 500: 41.8/-1.7 | no gain; schedule alone flips code (barely cut) and tools (refuses calls 5-6%) |
| lpc-bidax5m (math fix, 16/step, 520 steps) | + math1/math2, no shorten on svamp 470-905, 743 don't-over-cut pairs | final 11.3 / -3.6 (svamp -56% / -19.2) | math rambles: longer AND worse |
| lpc-bidax5s (same data, 25/step, 333 steps) | as bidax5m, bidax4's schedule | 100: 21.6/+1.2, 200: 20.1/-0.2, 300: 21.8/+0.1, final 23.6/+1.3 (svamp -22%/+1.0; no-fit 42/+3.3) | math longer than the original: the mixed-model don't-over-cut pairs (winner = original text, loser = bidax4 text) teach "write like the original" |
| **lpc-bidax6** (bida-exact settings) | dpo.py --bida-exact | 100: 44.4/-2.2, 200: 40.1/-0.9, 300: 41.7/-1.0 (final pending) | his settings remove most of the accuracy gap (same pairs: bidax4 at 200 was 54.6/-3.7); vs bida full 47.0/-0.78 it cuts ~5 pts less, mostly instructions (47 vs 55) and code (25 vs 33) |

**Settings: ours vs michal-bida's (read from his train_dpo_offline.py; dpo.py --bida-exact copies his)**

| | ours (bidax4) | his (bidax6) |
|---|---|---|
| precision | bfloat16 base + bf16 autocast (LoRA weights fp32) | float32 base + float16 autocast + GradScaler |
| lr at the end | cosine to 2e-6 | cosine to 0 (transformers get_cosine_schedule_with_warmup) |
| warm-up | 10 steps | steps // 20 (~17) |
| held-out pairs | 48 | 0 |
| long pairs skipped | > 9,000 tokens | > 8,704 tokens |
| same | DPO beta 0.1, LoRA r32 alpha 64 dropout 0 all 7 projections, AdamW 2e-5 wd 0, 16 pairs/step, 1 pass, no NLL | |
Which of the four differences matters is not isolated yet (precision-only / schedule-only runs proposed, not queued).

**Data: ours vs his (his prepare_domains.py)**: math GSM8K + MetaMath (his 1,392 pairs vs our 950 SVAMP-matched);
code all of KodCode-V1 Filter/Prefill/Leetcode/Evol easy+medium, questions <= 700 chars (ours: the KodCode-Light slice,
tasks <= 45 tokens, to look like MBPP+); instructions allenai IF_multi_constraints_upto5 + RLVR-IFeval (ours: argilla
ifeval-like + built); tools ToolACE + xLAM, ~45% no-fit (ours ToolACE/Hermes/Glaive + own no-fit, ~39%). We matched
questions to the test's prompt lengths; he took them as they come (8-gram contamination filter only). His evaluation
code = the official one (+ a GPU-memory flag and a Triton fallback); his own caveat: Db was chosen after seeing earlier
variants on the same test questions ("slightly optimistic").

**Code changes today**: dpo.py --bida-exact, --target-steps (pairs per step set so one pass = N steps); track.py
typeonly (one adapter per type/subtype), --to-parts, --lp-dpo comma-joined flags (also for typeonly), fullsoup step 0 =
final adapter, onpolicy light stage fix; lupin_pairs.py --band / --overcut (+ same-question band fix); prepare_types.py
ifeval_bida (his instruction sources) and kodcode_full (his code selection); build_pool.py --no-match; newtop --nt-raw,
--nt-samples. Not tested locally (no torch / datasets on the Mac): the GPU runs are the first test.

**Queues now**
- GPU 2: bidax6 final screen -> lpc-bidax5q (bida settings + math1/math2 plain: his math questions vs bidax6) -> newtop
  instr2 (his instruction sources, raw, x4) -> newtop code2 (his KodCode selection, raw, x4, 1,500 q) -> lpc-bidax7
  (bidax6 + instr2 + code2, his settings, 339 steps) -> 720 at 100/200/300/final -> bidax5s seed 1 (proposed to drop).
- GPU 1: per-type study lpt6 (our settings, no math) 720 screens -> lpt6b (same, his settings) -> screens. Math adapter
  waits for fixed pairs (option A: same-model don't-over-cut pairs from bidax4 only), user decision pending.
- Planned for 3 Nov (memory): on-policy round (best v1 answers the training questions x10).
- Runtimes disconnected ~16:30 and were reconnected by the user.
- ~16:50 (user): bidax5s seed 1 dropped. Queued the one-change tests on bidax4's pairs: lpc-bidax6p = his precision only
  (dpo.py --bida-precision) at the end of GPU 2's queue; lpc-bidax6s = his schedule only (--bida-schedule: warm-up
  steps//20, cosine to 0) at the end of GPU 1's. Both 720 at 100/200/300/final vs bidax4 (ours) and bidax6 (all four).
- ~17:30: DATASET.md written (sources, selection, answers, pair rules, counts, settings, decoded example pairs per type).
  Found while writing it: our dp13 math is mostly EASY (of 2,999: metamath_easy 1,865 + gsm8k_easy 420 = 1-2 calculation
  problems; metamath_sv 473; gsm8k_matched 241); michal-bida's math is GSM8K + MetaMath at all difficulties. Fits our
  larger math loss; lpc-bidax5q (his math sources on his settings) tests it. Pending results: bidax6 final (re-running
  after the disconnect), lpt6 per-type screens, bidax5q. Open user decisions: math fix option A; on-policy round from
  bidax6 (when, and what gives way).
- ~18:40 (user): analysis of the "four signals" critique (checked against our files): right-beats-wrong pairs push
  shorter too (shortest right vs longest wrong; math1 band 175 of 216); the original's mistakes are not the trained
  model's (too short); mixed-model "think enough" pairs failed. Tool references hold ONE accepted value per argument
  ("km" vs "kilometers" can lose). IFEval langdetect coin-flip claim REFUTED: the official grader sets
  DetectorFactory.seed = 0. "prostring" bug CONFIRMED: 99 ToolACE rows in dp13 contain prostring / stringname /
  stringName ("file" -> "string" inside words; not from our type clean-up, which renames types only; source not yet traced).
  Built: lupin_pairs.py --recipe four (loop -> guard kind, length-balanced same-model right, shorten only on >= 80%
  solved with a per-part dose, think-enough from --trained answers -> hard kind), --drop-pattern, --tool-near (BFCL pairs
  whose answers are > R alike dropped). Local test on math1 (original x4 + bidax4 x4 as stand-in): 1,865 pairs; right
  863 (longer won 428 / 863), loop 143, shorten 680, enough 179 (winner 319 / loser 205 tokens).
  track.py polans (a trained adapter answers the whole pool, half per runtime), --lp-trained (waits for both halves),
  rescreen --cx-scale (the strength dial), fullmodel --fm-screen-only.
  Queued: GPU 1 polans pa6-a (bidax6 x8 on half of dp13 + nofit4) -> lpc-bidax8 (four signals, from bidax6, bida
  settings, 339 steps, --tool-near 0.9, prostring dropped) -> 720 at 100/200/300/final -> bidax6 dial 0.5 / 0.75 ->
  re-score on the 720: drpo3d-23, drpo3f-15, stopB3-soup-25, lpw-2stage step 50 -> lpc-bidax6s. GPU 2 adds polans
  pa6-b after bidax5q. lpt6b paused. The fair stop re-test is not built yet (after the main bet).
  YARDSTICK: reference curve = quadratic fit to the 43 Screen-720 points above 15% saved: acc = -0.388 + 0.0899 s -
  0.0028 s^2 (about 0 at 20%, -1.3 at 40%, -2.35 at 47%, -2.9 at 50%). Distance above the curve: per-type merge mid
  +1.3, bidax5s final +1.1, Run A2 +0.8/+0.9, per-type soft +0.7, bida copy +0.5, bidax6 step 300 +0.5. michal-bida's
  own 47.0 / -0.78 is +1.6 above it (about +1.3 after the 720's bias). Column "vs curve" added to the Model Tree.
- ~19:00 (user): the on-policy round (polans pa6 a/b + lpc-bidax8) PAUSED until the bida rounds report (bidax6 final,
  bidax5q, bidax7, bidax6p/6s); then choose its starting model. Everything stays built and queued as comments.
- ~19:40 (user): ThinkPrune-style RL plan adopted, smoke test first. rl_train.py --reward cap: 1 if right AND finished
  under the question's own token cap, else 0; per-question max_tokens = its cap; group advantages; KL anchor. track.py
  thinkprune: caps per part from the original's right answers in the 4-answer pool file (math q0.95, others q0.70),
  printed with the share of questions keeping a right answer under the cap; questions right > half the time, mixed like
  the test (25% each, BFCL 400/200/240), prostring rows dropped; start raw or a merged adapter; 10-step blocks; the 720
  every 40 steps. Pass/fail bar fixed BEFORE running (the yardstick): works = a checkpoint >= 1 pt above the reference
  curve at the same saving on the 720, confirmed by a second seed, then the full test; target = > 50% at better than -1;
  fails = every checkpoint on or below the curve after ~300 steps. Smoke test (tp0) queued on GPU 1 after the per-type
  screens. Notes: IFEval's langdetect is already seeded by the official grader; equivalent tool values ("km" /
  "kilometers") are NOT handled yet (references hold one value).

### 2 Nov ~14:15 — back-filled: the ps1 prompt search results (27 Oct; only its queueing had been logged)
ps1 (prompt_search.py, candidates in prompt_candidates.py) on the ORIGINAL Qwen3-0.6B, training questions, 50 per type x
4 answers (a quick screen, not a full score). The "prefill" text forced at the start of the thinking was
"Okay, I'll solve this efficiently and only check what I'm unsure about." (Lightning McQwen3's opening).
| prompt | tokens saved | accuracy |
|---|---|---|
| prefill alone | -4.3% (4% longer) | -1.0 |
| "no re-check" instruction + prefill | +0.2% | +0.9 |
| worked examples ("exdraft") | +24.8% | -2.3 |
| worked examples + prefill | +24.8% | -1.9 |
Forcing the opening did nothing for length. The opening was not chosen the cue way (searching the model's own likely
first tokens and keeping the one whose answers agree most). Test-time prefilling is now ruled out anyway (rule 3).
Thinking ON vs OFF (25-26 Oct, nt1 / nt2, fresh x8) IS in this log (~line 1822-1895): original SVAMP 89.7 -> 17.5,
MBPP+ 46.2 -> 37.2, IFEval 65.6 -> 60.6, BFCL one fn 88.1 -> 76.6, several 75.9 -> 58.8, none fits 80.9 -> 87.2;
rc1-rl-40 the same pattern. Read on the old fresh-240 set.
- ~14:10: Chrome restarted and both Colab pages reloaded; queues restarted (user OK); caffeinate restarted.
- sophielwang.com/cues read (arXiv 2610.06851, "Base Models Can Reason By Taking a Cue From Training Data"): prefilling
  an opening cue (".\n\nOkay") lifts Olmo-3-7B base on MATH-500 from 41.5% to 76.9%; RL's biggest probability change is
  at the first two tokens. About eliciting reasoning in base models, no token-saving figures; test-time prefilling is
  ruled out for us; our ps1 prefill made answers 4% longer.
- ~15:45: runners back on their queues after a ~1.5 h first copy from Drive on fresh machines.
  lpc-bidax6 FINAL on the 720: +37.8 / -1.7 (svamp +48/-3.7, mbpp +18/-2.5, ifeval +41/-1.0, simple +38/-3.1, multiple
  +40/+0.2, no-fit +44/+4.6): about 0.7 below the reference curve; step 300 (41.7 / -1.0, +0.5) stays its best point.
  Per-type study lpt6 (ONE type's pairs each, our settings, 2-seed soups, final) on the 720, own type in brackets:
  code only -3.2 / -1.0 (code -5% LONGER / +0.4; instructions -9% / -3.1); instructions only +32.6 / -2.8 (ifeval
  +43/-4.4; spill: math +27/-3.5, code +32/-1.9, tools +26-28); one function only +33.7 / -1.7 (simple +34/-1.2; spill:
  code +41/-0.8, math +34/-4.2, no-fit +35/-7.3). Spill-over is nearly total: a tools-only or instructions-only adapter
  cuts math and code about as much as its own type, and the cost lands on math and no-fit. Code-only pairs do not cut
  code at all. Multiple-only and no-fit-only screens next. GPU 2: bidax5q training (step 105/337).
- ~15:55: lpt6 several-functions only: +18.3 / -1.4 (multiple +26/0.0; spill: code +24/-2.5, simple +21/-1.5, math
  +18/-1.5, no-fit +18/-2.9, ifeval +9/0.0). Code-only checked on the records: code thinking unchanged (median 966 vs
  994, capped 9.4% vs 8.5%, accuracy 51.2 vs 50.8) although it ordered 86% of held-out code pairs right: the code pairs
  teach content, not length; in mixed runs code's cut comes from the other types' spill-over.
- ~16:00 (user): PER-TYPE DIRECTION DROPPED. Every one-type adapter sits below the reference curve (instructions only
  -2.4, several functions -1.7, one function -1.2, code only no effect): "think less" is learned as one shared habit,
  and a type trained alone loses the other types' right-beats-wrong protection, so the spill-over lands unopposed on
  math and no-fit. Dial-mixing per-type adapters would only mix general pushes. lpt6b and the step-50 screens dropped.
  Effort goes to: michal-bida's settings (bidax6), his data (bidax5q tonight, bidax7 tomorrow), then the on-policy
  four-signal round (its same-model be-right pairs for every type are the protection this showed is needed).
- ~16:40: per-type no-fit only: +11.9 / -1.9 (no-fit +36/+11.7 but multiple +17/-16.6, simple +18/-20.6: it learned to
  refuse calls; spill on math/code/ifeval small). Per-type study complete; direction dropped (see 16:00).
- RL SMOKE TEST tp0 (thinkprune, from the original, 5 steps x 16 questions x 8 answers, lr 5e-6, KL 1e-3): ran clean,
  6.6 min for the 5 steps (~80 s per 128-answer step incl. loading). Caps (original's right answers; share of questions
  right > half the time that keep a right answer under the cap): math 1,147 tokens 99%; code 1,022 84%; instructions
  528 84%; tools one fn 171 89%, several 196 91%, none fits 223 88%. Per step: batch accuracy 62-84%, thinking 302-415
  tokens, entropy 0.47-0.58 (stable), only 6-9 of 16 questions had both right and wrong answers (the rest teach
  nothing: dynamic sampling needed). "Gradients will be None" warning checked: the step-5 LoRA's B matrices are all
  non-zero (196/196, norm ~0.002), so the adapter trains.
  Sizing: the planned 64 x 16 = 1,024 answers per step would be ~8-10 min per step, ~45-50 h for 300 steps per run.
  32 x 8 with dynamic sampling (draw 2x, keep informative groups) ~3-4 min per step, ~15-20 h per run incl. the 720
  every 40 steps. Decision (size, start model, which GPU) pending with the user.
- ~17:20: bidax6 step 300 DIAL on the 720 (merge strength): 0.5 +22.5 / -0.8 (svamp +24/-2.1, ifeval +19/-0.8, no-fit
  +29/+1.7); 0.75 +30.9 / -0.7 (svamp +39/-2.5, mbpp +15/-2.1, ifeval +33/+1.5, no-fit +37/+2.7); 1.0 = +41.7 / -1.0.
  bidax6's own curve is nearly flat: from 22% to 42% saved it costs only ~0.2 points, so full strength is its best
  point (+0.5 vs the reference curve; 0.75 -0.4; 0.5 -1.0). The ~-0.7 floor is mostly math (-2.1 to -2.7 at every
  strength): math loses accuracy as soon as it is touched at all. bidax5q (his math questions) at step 310/337.
- ~17:50: old-run re-score on the 720 (yardstick): drpo3d-23 (gentle DRPO RL from raw, old RL pool, 26 Oct) +21.7 / -0.4
  (svamp +28/-0.2, mbpp +25/+0.2, ifeval +10/-2.1, simple +17/-0.4, multiple +23/+1.5, no-fit +26/+0.8): about 0.6
  BELOW the reference curve (its old fresh-240 reading was +21.2 / +0.2). Not on the visuals (older data, user rule).
  drpo3f-15, stopB3-soup-25, lpw-2stage next. bidax5q training done (335/337), its 720 screens starting.
- ~18:30 **lpc-bidax5q** (bida-exact settings + math1/math2 = michal-bida's GSM8K/MetaMath math, plain bida pairs, 339
  steps) on the 720: step 100 **+41.8 / +0.2** (svamp +47/-1.5, mbpp +36/+1.3, ifeval +43/+0.2, simple +37/-2.5, multiple
  +40/+1.7, no-fit +42/+3.1); step 200 **+48.5 / -0.9** (svamp +61/-3.5, mbpp +39/+0.6, ifeval +45/-1.0, simple +41/-1.7,
  multiple +46/+0.4, no-fit +47/+2.1). Both about **+1.7 above the reference curve**: the first run past the 1-point bar
  (needs a second seed, then the full test). Step 200 is at michal-bida's own level (47.0 / -0.78 full; the 720 reads
  saving ~0.6 high, accuracy ~0.3 low). His math data + his settings together close the gap that neither closed alone
  (bidax6 = settings only: 41.7 / -1.0; bidax5m/s = our math fix: math rambled).
  drpo3f-15 re-scored: +26.5 / +0.1 (on the curve, +0.1). GPU 1 DISCONNECTED (Reconnect) during the re-scores; waiting
  for the user's OK.
- ~18:45: lpc-bidax5q step 300 +47.9 / -1.5 (svamp +60/-4.8, mbpp +38/-1.7, ifeval +47/-0.6, simple +41/-1.2, multiple
  +44/+1.5, no-fit +47/+3.5): +1.0 vs the curve; step 200 (+48.5 / -0.9, +1.7) stays its best point. Final screening.
  GPU 1 still disconnected (waiting for the user).
- ~19:10: both runtimes were reconnected by the user (GPU 1 had hung on "Reconnecting"; GPU 2's bidax5q final screen
  was lost and re-runs). GPU 1 queue: bidax5q SEED 1 (same recipe) -> its 720 at 100/200/300/final -> stopB3 / lpw
  re-scores -> bidax6s. Full test of the best bidax5q step only after seed 1 confirms, and with the user's OK.
- ~19:45 **lpc-bidax5q FINAL** (seed 0) on the 720: **+46.7 / -0.5** (svamp +58/-1.9, mbpp +37/+0.4, ifeval +45/-0.8,
  simple +41/-2.3, multiple +44/+0.6, no-fit +47/+3.1): **+1.8 vs the reference curve**, our best point; at
  michal-bida's saving (47.0) with better accuracy than his -0.78 on the 720 (which reads accuracy ~0.3 low, saving ~0.6
  high). The final adapter is what bida scores. Seed 1 training on GPU 1 (the confirmation); full test after it, with
  the user's OK. GPU 2: newtop instr2 (his instruction sources) started.
- ~22:05: BOTH runtimes disconnected again. Before: GPU 1 bidax5q seed 1 at step 150/337 (checkpoint every 10 steps,
  dpo.py resumes from the last one); GPU 2 newtop instr2 DONE (michal-bida's instruction sources: 2,000 questions x4,
  mean solve 65%, always 1,016 / sometimes 528 / never 456: harder than our pool's instructions), then code2 had built
  33,401 usable KodCode-V1 candidates (1,747 dropped as already used) and was answering. Waiting for the user's OK.
- 3 Nov ~00:40: lpc-bidax5q SEED 1 step 100: +42.8 / -1.9 (svamp +52/-3.7, mbpp +40/-1.9, ifeval +39/-1.5, simple
  +36/-2.7, multiple +40/+0.6, no-fit +41/+1.0): on the curve (-0.2), vs seed 0 step 100 +41.8 / +0.2 (+1.7). Seed
  spread at step 100 ~2 pts of accuracy; steps 200/300/final decide. code2 answering: 300/1,500 done (~1 h per chunk).
- ~00:45: bidax5q SEED 1 step 200: +51.8 / -1.9 (svamp +62/-5.2, mbpp +48/-0.2, ifeval +51/-1.9, simple +40/-2.3,
  multiple +43/+0.4, no-fit +47/+1.0): +1.3 vs the curve (seed 0 step 200: +48.5 / -0.9, +1.7). Both seeds clear the
  1-point bar at step 200; seed 1 cuts ~3 pts more and loses ~1 more (math -5.2). Steps 300 / final next.
- 3 Nov ~01:00 (user): v1 = lpc-bidax5q. Queued: FULL TEST of both seeds' final adapters (fullsoup now names a
  single-seed run full-<name>-s<seed>-<step>, so the two seeds get separate scores), then v1 (seed 0) answers the
  training pool x8 (polans pa5q, half b on GPU 1 after the full tests, half a on GPU 2 after bidax7). Next, per the
  user's plan: (1) visualize v1 vs the original (per type, length bands, 720 + training questions), (2) its answers,
  (3) pairs aimed at where it cuts and loses, compared on training vs evaluation questions, (4) iterate with the
  yardstick, watching for over-pushing.
- ~01:25 (user): no full test until v1 is really good; full tests of bidax5q and v1's answers PAUSED. Pair combinations
  first, all on bidax5q's data (dp13 + nofit4 + math1 + math2) with michal-bida's settings, 339 steps, 720 at
  100/200/300/final, judged by the yardstick. GPU 1 after bidax5q seed 1: lpc-b5four (four signals from the original's
  answers: loop / length-balanced be-right / shorten only on >= 80% solved, dose 1x / no think-enough without trained
  answers; prostring rows and near-identical tool pairs dropped), lpc-b5diff (difficulty rule, no shorten under 50%
  solved), lpc-b5cap2 (up to 2 shorten per guard). GPU 2: lpc-bidax7 widened to bidax5q + instr2 + code2 (all of his
  sources) after the code top-up. BottleCap's own ThinkingCap method is not public (third-party pages guess RL with a
  length-aware reward); their "collapse of self-trained LMs" paper warns about repeated self-training.
- ~02:00 (user): three tasks. (1) a stronger v1 from pair techniques compared at equal accuracy (pending the user's
  list); (2) GRPO from the raw Qwen to learn how it behaves vs DPO; (3) RL on v1 with the winning reward.
  Built for (2): rl_train.py --reward lenpen (khudgins' ThinkingCap reproduction, read from github.com/khudgins/
  ornith-thinking-cap: right = 1 - lam * min(1, thinking / 2048), wrong = 0, + 0.1 format bonus; lam ramps 1e-3 ->
  lam_max), --lp-gate (GRPO-lambda: no length penalty where a group is right < G of the time), --dr-grpo (advantages not
  divided by the group spread; loss divided by replies x max tokens). His settings: G 8, 12 prompts, lr 2e-6, KL 0.04,
  temperature 0.9 (we keep the evaluation's sampling), LoRA r32 a64, 500-1000 steps. Papers: lr ~1e-6 typical at
  0.5-1.5B full fine-tune (LoRA usually higher); KL 0 (TRL default, Dr. GRPO / DAPO) to 0.04; GRPO-lambda reports a
  uniform length penalty losing accuracy after ~40 steps.
  Queued: GPU 1 after bidax5q seed 1: gr0 smoke, grA (his recipe), grB (penalty up to 0.2); GPU 2 after bidax7: grC (no
  KL), grD (gate 0.5). All from raw, 150 steps, the 720 at 50/100/150, ~4-4.5 h each.
- ~02:30 (user): GRPO from the raw Qwen as RL, 5 variations, ONE setting changed each from grA (lr 2e-6, KL 0.04,
  penalty up to 0.05, no gate): grB penalty 0.2, grC KL 0, grD gate 0.5 (GRPO-lambda), grE lr 5e-6. 150 steps each,
  12 questions x 8 answers, Dr. GRPO normalization, the 720 at 50/100/150. GPU 1: gr0 smoke -> grA -> grB -> grE;
  GPU 2 after bidax7: grC -> grD. Then adapt.
- ~02:40: GRPO screens now draw the SAME questions as v1's DPO data: pool dp13 + nofit4 + math1 + math2 (thinkprune
  --tp-extra, default nofit4,math1,math2; answers from bidax5q's 4-answer file, used for the eligibility filter only).
  The plan (user): 10 DPO pair strategies (all else = bidax5q) -> v1; 5 GRPO variations from the raw Qwen; then v1
  answers its own questions and both DPO and GRPO are tried on it. All judged on the 720 with the yardstick.
- 01:30 (real clock; earlier "~02:xx" stamps tonight were estimates): bidax5q SEED 1 final +48.5 / -2.2 (svamp
  +58/-2.9, mbpp +39/-4.6, ifeval +51/-0.4, simple +40/-2.5, multiple +44/-0.4, no-fit +45/+0.8): +0.4 vs the curve
  (seed 0 final +46.7 / -0.5, +1.8). Seed 1 step 300 also scored (line truncated in the log view). The two seeds'
  finals differ by 1.7 pts of accuracy: the recipe is ~+1.1 above the curve on average, seed luck +-0.7; the 1-point
  bar holds at step 200 for both seeds (+1.7 / +1.3), at the final only for seed 0. lpw-2stage step 50 re-scored on the
  720: +29.5 / +1.6 (+1.8 vs the curve; a dp13-era "be right first, then shorten" two-stage run, 29 Oct).
  GPU 1 had pulled the plan before the GRPO lines and started bidax6s: interrupted and restarted so gr0 / grA / grB /
  grE run first.
- ~01:45 (user): sixth GRPO variation grF = SmartThinker reward (arXiv 2603.08000). rl_train.py --reward smart: per
  group, l* = (s1^2 mu2 - s2^2 mu1)/(s1^2 - s2^2) from all lengths ~N(mu1,s1^2) and right lengths ~N(mu2,s2^2), clipped
  to the group's range; only right answers longer than l* penalized (-ReLU(l - l*)); coefficient p_err / (mean(r_len) -
  min(r_len)), the largest that keeps every right answer at or above the group mean; groups with < 2 right answers or
  all right get plain 0/1. Unit check on a toy group: right answers >= mean, wrong <= mean, the 900-token right answer
  lands exactly on the mean. Everything else = grA (the one change is the reward). Queued on GPU 2 after grD, behind a
  5-step smoke test (gs0).

### 3 Nov: grF aligned with the SmartThinker paper (read in full)
Before it ran, grF had three settings borrowed from grA that the paper does not use. Now: plain GRPO advantages
(reward minus group mean, divided by the group's spread; token-mean loss, no Dr. GRPO), no KL leash (the paper omits
it), constant lr. New `track.py --tp-max-solve`: grF keeps only questions the original Qwen misses at least once in 4,
because an all-right group gets λ = p_err/... = 0, so no signal, and the paper uses 64 questions a step against our 12
(the paper itself says difficulty-based sampling would cut the steps needed). lr stays 2e-6 like grA (LoRA r32; the
paper's 1e-6 is a full fine-tune). Paper facts worth keeping: the longest right answer always gets advantage 0 (never
pushed down); the fixed-λ ablation saved more (−63%) but lost 3.2 pts on the 1.5B model, so the dynamic λ is the
accuracy guard; SmartThinker on top of an already-pruned model (ThinkPrune-4k → SmartThinker) was its best result,
which fits the plan to try GRPO on v1.

### 3 Nov: the DPO pair-strategy screen (task 1), queued on two new runtimes
User: try many pair strategies now that the settings are fixed; more compute added. Everything = bidax5q seed 0
(--bida-exact, 339 steps, dp13 + nofit4 + math1 + math2, 4 original answers; 720 set x4 at 100/200/300/final; yardstick).
New lupin_pairs.py flags: `--add-hard P` (DAST's "keep trying": longest > shortest finished wrong, questions solved < P),
`--budget` (DAST's per-question length target replaces the fixed 1.5x), `--margin-any` (drop each kind's weakest 15%
for any recipe), recipe `simright` (similar-length right > wrong + think less, 1:1). Tested on synthetic answers.
Runtime 3 (third_colab.ipynb, plan_line_c.txt): D1 lpd-dast, D2 lpd-hard, D3 lpd-route, D4 lpd-budget, D5 lpd-margin.
Runtime 4 (fourth_colab.ipynb, plan_line_d.txt): D7 lpd-st1 -> lpd-2stage, D8 lpd-math, D6 lpd-simright, D9 lpd-adar1,
D10 lpd-l2s50e. Phase 2 (new code / answers): SimPO / LD-DPO, Ada-R1 with no-thinking answers, concise-prompt winners,
SFT warm-up, stop pairs. Seed noise: bidax5q s0 +1.8 vs s1 +0.4, so a winner needs a second seed.
**3 Nov, revised (user: some strategies will not shorten enough):** D3, D6 and D8 now keep bida's FULL pairs and only add
their protective pairs (new `--add-right-below SPEC`: similar-length right > wrong on questions solved below a share, all
parts or per part). D3 = bida + right & keep-trying pairs below 75% solved; D6 = bida + similar-length (±15%) right > wrong
everywhere; D8 = bida + right pairs on math / no-fit below 80%. D7's "be right" round cut to 50 steps (as the original),
its second round is bida's full pairs. New D11 lpd-push2 = bidax5q with up to 2x shorten pairs per guard (the 1:1 cap binds
in every part of bidax5q), the push-harder control every protective strategy must beat at the same saving. A strategy
counts only if it saves >= 40%. Unchanged: D1 DAST, D2 bida + keep trying, D4 budget, D5 margin, D9 Ada-R1, D10 l2s50e.
**3 Nov ~02:00:** only 2 GPUs on the Colab account (user), so runtimes 3/4 are dropped (plan_line_c/d and the two notebooks
removed) and the DPO screen alternates with GRPO on the two queues, scored at steps 100 / 200 / final (step 300 dropped:
too close to the final at 339). GPU 1: grA -> D11 push2 -> grB -> D1 dast -> grE -> D2 hard, D3 route, D4 budget, D5 margin,
D10 l2s50e -> old re-scores, bidax6s. GPU 2: code2 -> bidax7 -> grC -> D9 adar1 -> grD -> gs0, grF -> D7 st1 + 2stage,
D8 math, D6 simright -> bidax6p.
**3 Nov ~02:45, all-else-equal audit of the DPO screen.** Every lpd-* line differs from bidax5q's only in its pair flags
(checked token by token). The edited lupin_pairs.py builds exactly bidax5q's pairs (byte-identical to the old code at a
fixed hash seed). Found: which shorten pairs survive the 1:1 cap depended on Python's per-process set order (the parts
were walked in random order before rng.shuffle), so bidax5q's exact subset can't be rebuilt; fixed (sorted), every new
run is reproducible. That subset noise is part of the ~1.4-pt seed noise. Known differences kept: one pass in 339 steps
means more pairs = more pairs per step (bidax5q ~16); D7 trains 50 extra steps and uses stage 1 as its reference.

### 3 Nov: idea parked: weight-space editing (Astra) and a contrastive Wanda test (discussion only, nothing built)
Source: "Sparks of Weight-Space Programming" (Takehi, Teoh, Zhang, Khattab, MIT, 2 Nov 2026,
rikiyat.github.io/blog/weight-space-programming; logs github.com/RikiyaT/lm_understand-logs). A frontier LLM ("GPT-6
Astra"), with a Python sandbox and up to 50 forward-pass batches, reads and edits weights by reasoning, no gradient descent.
Closest to us, Exp 8: Qwen3.5-0.8B looped to the 32k cap on 93% of MATH-500; Astra traced the loop to one direction written
by the MLPs of layers 15 and 19 and removed it from 64 output columns each (0.016% of weights): cap hits 93 -> 33%, MATH-500
7.9 -> 48.1, IFEval x3. On Gemma 3 270M (rarely loops) no gain: "editing could remove an obstacle that hid an ability, but
it could not add an ability the model lacked". So: it removes a bad habit, it does not add skill; math is spread over the
weights (no "math weights" to pick out), but a habit (re-checking a settled answer) is a direction / a few heads.
How it relates to ours: closer to Track 2d steering (October: 'short vs long' direction, ~18% saved at -0.1, stronger pushes
broke accuracy) than to Wanda. Wanda (30 Oct: 10% pruned free, 20-30% make the model think LONGER) keeps what the model
does on its usual answers; Astra finds the part that CAUSES a behaviour and removes only that. Our per-type length
directions are nearly unrelated (cosines -0.18 to 0.39), so a math-only edit might spare the other types (DPO spills over).
Hard part: some re-checks catch real errors; a column removal acts on every question alike, so keeping accuracy needs the
model to "know" inside when it is already right (cf. "Reasoning Models Know When They're Right", 2025).
How much waste is there to remove (our 25-26 Oct cut measurements, original Qwen, benchmark questions, small samples:
51-148 replies): earliest cut whose 20 forced answers are as right as finishing, right replies only: SVAMP 41%, IFEval
49%, MBPP+ 72%, BFCL 49% of the thinking. But on SVAMP the answer builds up LATE (success 0.34 at 20-40% kept vs 0.73
full), so math has the least "re-check after a settled answer" and code the most; we never labelled re-checks directly.
Cheap tests first, no API cost, ~1 GPU-h each, after the GRPO / DPO screens: (1) label re-checks in our math answers
(did the answer change?) and fit a probe on hidden states: can the 0.6B tell useful from wasted re-checks? (2)
contrastive Wanda: score weights on wasted vs useful stretches (wanda_prune.py), dampen those used only by wasted ones.
Only if (1) shows a signal: an Astra harness (OpenRouter key; cost unknown, user's call), judged on the 720 / full test.
**3 Nov, Astra repo read in full (later project, nothing built).** Correction: run.json token usage accumulates within a
conversation (the repo README says not to sum it over turns); the real p50c total is ~5.4M input (5.2M cached) + 0.15M
output over 4 conversations (turns 16-40 may be under-reported), not 86.8M. Time: 1.8 h of model calls, 9.1 h of generation
passes (34 of 50 passes were generations, 16 teacher-forced reads; 19 requests failed and did not count).
How p50c found the edit: 4 hand-written matched pairs (same correct start, then a productive check vs a 3-4x repeated
phrase), teacher-forced; direction = mean residual difference over the last 36 tokens; per-layer attribution picked the MLP
writers of layers 15 and 19; edit W' = W0 - a u(u^T W0) on down_proj, kept to the top-64 columns per layer by
(u.W[:,j]) x activation difference; dose a = 1 / 2 / 3 / 6; controls: random direction +1.6 +-3.5 pp, premature-close
log-prob unchanged; frozen edit confirmed on 1,152 fresh problems: +49.8 +-1.5 pp, 579 rescues vs 5 losses, accuracy on
answers that had finished unchanged (93.9 -> 93.9). Answer length: mean drops only because fewer answers are cut; FINISHED
answers got LONGER (MATH-500 5,631 -> 16,740 tokens, GSM8K 4,954 -> 10,372). The task prompt forbade "making the model skip
its thinking or close it early regardless of the problem". p10c / p25c failed: wrong components (copying heads, one
neuron) and too small a dose. Gemma follow-up (no looping): every budget within ~1 SE, MATH-500 6.8 -> 7.0-7.7. Released:
prompts, every turn, eval code, edits; missing: the kernel prelude, the forward-pass server, the driver loop.
**3 Nov ~03:25, where the thinking tokens go (720 set x4, records.jsonl of s4-base and s4-lpc-bidax5q-s0-final).**
| type | raw: thinking / capped / repeated 8-grams | v1: thinking / capped / repeated 8-grams |
|---|---|---|
| svamp | 685 / 2.1% / 2.3% | 281 / 0.0% / 0.6% |
| mbpp | 2,077 / 8.5% / 7.2% | 1,300 / 2.9% / 3.5% |
| ifeval | 420 / 2.9% / 2.2% | 176 / 1.5% / 0.2% |
| bfcl (3 parts) | 175-206 / 0% / 0.1-0.3% | 94-104 / 0% / 0% |
Capped answers: raw 65 (2.3%) holding 22% of all thinking tokens; v1 21 (0.7%: 14 code, 7 instructions, 1 right)
holding **12% of all v1 thinking**, median 18% repeated 8-grams (14% of them > 30%). Code is 63% of v1's thinking.
Re-check words (wait / let me check / verify / alternatively / hmm) per 1,000 words barely move (raw 10-17, v1 8-15):
v1 thinks less but hesitates at the same rate. Right SVAMP answers: the final number first appears with 53% (raw) / 39%
(v1) of the thinking still to come (a rough "after the answer" proxy: the number may first appear as an intermediate).
Decision (user): Astra weight editing runs ON TOP of the final v1, Astra doing the diagnosis and the fix.

### 3 Nov ~04:00: grA step 50 (720 x4) + grA crash fixed
**grA-50** (GRPO from the raw Qwen, khudgins/Ornith length penalty, λ ramping to 0.05 over 100 steps, so ~0.025 at step 50;
leash 0.04, lr 2e-6): **+1.5% / -1.0** (vs curve about -0.7; saving under 15%, below the yardstick's range). svamp +1.7 /
-0.2, mbpp +3.3 / 0.0, ifeval +0.4 / -2.1, one function -4.1 / -1.2, several +1.6 / -2.9, no fit +2.8 / -1.0. Training: batch
accuracy 83-95%, thinking 400-730 tokens per step with no trend, every question with signal (12/12): at this penalty
strength the reward is almost all "be right", so 50 steps change nearly nothing. As expected for this recipe (khudgins
trains 500-1,000 steps); the stronger half of the ramp (steps 50-100) is where saving should show.
**Crash:** at step 51 rl_train's IFEval grader hit a language constraint and `import langdetect` failed (the package
was missing on this runtime). Fixed: rl_train.py installs langdetect if missing (same fix protects grC / grD / grF).
grA re-queued on GPU 1 right after D11 lpd-push2 (now training), resuming from its saved step 50. GPU 2: code2 answers
done (1,500 questions), bidax7 training.
- ~07:00 **D11 lpd-push2 step 100** (bidax5q + up to 2x 'think less' pairs): **+49.2% / -3.7** (full-eval weighting; the
  720 print: +49.3 / -3.5), **-1.0 vs the curve** (curve -2.7 at 49%). svamp +62 / -5.4, mbpp +49 / -5.2, ifeval +41 / -2.5,
  one function +40 / -4.4, several +42 / +2.9, no fit +43 / -1.2. Against bidax5q s0 at the same step (41.8 / +0.2, +1.7):
  doubling the push buys ~7 points of saving for ~4 points of accuracy, so the 1:1 cap is protecting accuracy, as
  michal-bida intended. Steps 200 / final scoring.
- ~08:00 **D11 lpd-push2** step 200 **+49.7 / -2.9 (on the curve, -0.1)**; final **+52.4 / -3.0 (+0.4)**: svamp +64 / -3.7,
  mbpp +44 / -4.4, ifeval +53 / -3.3, one function +44 / -2.5, several +48 / +1.1, no fit +48 / +1.2. The push-harder control
  reaches the highest saving of the screen so far but only rides the reference curve: extra 'think less' pairs move the
  model ALONG the curve, they do not lift it (bidax5q s0: +1.0 to +1.8). This is the bar for the protective strategies:
  beat ~-3.0 at ~52%, or -2.9 at ~50%.
- ~08:00 **bidax7 step 100** (bidax5q + instr2 + code2, michal-bida's own instruction and code sources): **+41.3 / -1.9,
  -0.5 vs the curve**: svamp +49 / -4.6, mbpp +38 / -1.7, ifeval +40 / -0.8, one function +36 / -1.9, several +39 / +0.8, no
  fit +41 / +0.2. Below bidax5q s0 at step 100 (41.8 / +0.2, +1.7) at the same saving; steps 200 / 300 / final scoring.
  GPU 1 moved on to grB (GRPO, 4x penalty).
- ~09:00 **bidax7** (bidax5q + instr2 + code2): step 200 **+43.1 / -0.6 (+1.1 vs curve)**, step 300 **+45.3 / -0.8 (+1.3)**,
  final +43.5 / -2.5 (-0.7). Step 300: svamp +49 / -1.3, mbpp +39 / +0.8, ifeval +51 / -2.9, one function +38 / -1.0,
  several +42 / +0.6, no fit +46 / +1.5. Steps 200-300 clear the 1-point bar (like bidax5q s0: +1.7 / +1.0 at 200 / 300) but at
  ~3 points less saving, and the final checkpoint drops (mbpp +30 / -4.4, svamp -3.5). His code / instruction questions do not
  improve on bidax5q; one seed, so within seed noise. GPU 2 moved on to grC (GRPO, no leash).

### 3 Nov: "Chirurgical" edits (built, tested on CPU, queued after the DPO / GRPO screens)
User: replicate "Sparks of Weight-Space Programming" Exp 8 ourselves (no Astra, no API) on the raw Qwen, then v1, then a
subtler habit than looping. Technique, from the run notes (logs github.com/RikiyaT/lm_understand-logs):
- **Loop direction (p50c, the 50-round run):** 4 matched contexts (same correct start, then a productive check or a repeated
  phrase), teacher-forced; u = mean(loop - productive) of the hidden state after the layer; the MLPs of layers 15 and 19
  write it most; edit W = W0 - a u (mask64 * u^T W0) on down_proj, mask = the 64 columns writing the most loop-specific
  signal (u.W[:,j] x activation difference), chosen once on the first probe; a = 6 (10.6% / 8.1% of those matrices).
  Checks: loop text -0.103 vs productive -0.009 nats/token, transfer cosine 0.84 on 6 new contexts, premature close stays at
  parts per billion, random direction +1.6 +-3.5 pp. Confirmed on 1,152 fresh problems: +49.8 +-1.5 pp.
- **Copying head (p25c, the 25-round run):** found by attention, not weights: per head, the attention a token pays to what
  followed its previous occurrence ("previous-matching-continuation"); L15H7 0.887 on loops vs 0.082 productive. Edit:
  zero its 256 o_proj columns (also x0.5, x-0.5). Proof: dose-response, x1.5 makes loops MORE likely, a same-layer
  low-copy head does much less. 768 fresh problems: +8.3 +-1.4 pp (95 rescued, 31 broken). "do NOT claim perfect
  selectivity": some productive likelihood loss. p50c found head edits less selective than the direction.
- **Correction:** the "loop removal shortens finished answers" figure was a small first-sample side computation
  (13 / 23 problems); the run notes say rescued answers stayed long (median ~8,100 tokens) and finished answers got longer
  on average. Whether our edit trims repetition inside normal answers is ours to measure.
**chirurgical.py** (phases mine / probe / mech / gen, each resumable): mines capped answers with >= 30% repeated 8-grams
from the original's own pool answers (lupin-all-...-a4: dp13 + top-ups, prostring rows dropped), cuts each before its
first repeated 32-token stretch (loop branch = the real next 48 tokens; productive branch = the most novel of 8 sampled
continuations of the same prefix, novelty >= 0.8), one pair per question, half to find / half to check; 300 finished right
answers; fixed dev 256 / holdout 512 questions never used for pairs. Probe: direction per layer + split-half cosine,
MLP writers, copy score for every head (Qwen3-0.6B: 28 layers x 16 heads of 128). Candidates: top-3 copy heads x {0, 0.5},
direction on the top-2 writer layers x {1, 2, 4, 6}, a random orthogonal direction (same mask, a = 6). Mech gate: loop
log-prob < -0.01, productive and normal > -0.02, premature-close odds < 2x. Gen: paired dev answers (same seeds, eval
sampling, 8,192 tokens, typed.py grading): accuracy +- SE, thinking saved, cap share, saving on answers both finished,
per type; pick = most saved at >= -1 point, confirmed on holdout, then the 720 x4 (s4-chir-NAME). float32 reads and edits,
bfloat16 for generation (as the paper's evaluation). Tests: chirurgical_test.py (helpers + probe / candidates / mech on a
tiny random Qwen3, CPU) pass; chirurgical_cpu_check.py runs the real 0.6B on CPU (bfloat16) with 6 hand-written matched pairs.
Queued at the end of GPU 1 (after bidax6s): chirurgical chs0 --ch-smoke, then ch1.
**3 Nov ~11:40, Chirurgical CPU check on the real Qwen3-0.6B (Mac, bfloat16, 6 hand-written matched pairs, 3 to find /
3 to check; no Astra, no API; plumbing + sanity only):** copy heads L16H14 0.60 vs 0.18, L18H5 0.39 vs 0.05, L6H11, L21H8,
L21H9; loop-direction writers L22, L26, L19, L21; split-half direction cosine <= 0.55 (3 pairs per half: too few to trust).
Mech (loop / productive / normal log-prob per token, close): L16H14 x0 -0.035 / -0.001 / -0.000 PASS, x0.5 -0.019 PASS;
L18H5 x0 -0.064 / -0.005 / +0.011 PASS, x0.5 -0.028 PASS; direction L22+L26 a2 -0.019 / -0.001 PASS, a6 -0.059 / -0.007
PASS; random direction a6 +0.008 (loop MORE likely) -> dropped. Dose-response and the random control behave as in the
paper. No accuracy or length evidence yet (needs the GPU generation phase).
**3 Nov ~12:30, Chirurgical revised (user): generate from scratch, like the paper, but ask the questions we already know
loop.** mine now fixes dev 256 / holdout 512 first, then the ORIGINAL model answers 1,500 fresh practice questions (75% of
them questions where a stored answer was capped with >= 30% repeated 8-grams, 25% random for the normal answers), graded;
loops and pairs come from those fresh answers. Inserted on GPU 1 between grB and D1 (chs0 smoke, then ch1). Tests:
helpers, probe / candidates / mech on a tiny Qwen3, and mine's selection with a stand-in generator all pass.
**~12:40 (user: "it's ok if it takes more time"):** sizes raised toward the paper's: 3,000 fresh practice answers, dev 512
(paired SE ~+-1.8 pts), holdout 1,024 (the paper confirmed on 1,152), 200 held-out pairs and 400 normal answers for the checks.
- ~11:25 **grB step 50** (GRPO, 4x length penalty: λ ramping to 0.2 over 100 steps, ~0.1 at step 50): **-2.0% / -0.5**
  (thinks 2% LONGER; below the yardstick's range): svamp -2 / -1.0, mbpp 0 / +2.9, ifeval -8 / -2.5, one function +1 / -1.7,
  several +4 / -0.6, no fit 0 / -1.5. Like grA at step 50 (+1.5 / -1.0): 50 GRPO steps at 12 questions x 8 answers move
  nothing yet, even at 4x the penalty. Steps 100 / 150 next. GPU 2: grC at step 50, scoring.
**3 Nov ~13:30, Chirurgical CPU check with 20 hand-written pairs (10 find / 10 check; real Qwen3-0.6B, bfloat16; no
Astra):** go (top copy-head gap 0.43; direction split-half cosine up to 0.77, from 0.55 with 3 pairs per half). Same top
copy heads as the 6-pair batch: L16H14 0.58 vs 0.15, L18H5 0.36 vs 0.04. Writers L19, L22, L26, L27. Mech (loop /
productive / normal): L16H14 x0 -0.025 / +0.006 / -0.000, x0.5 -0.013; L18H5 x0 -0.044 / +0.005 / +0.011; direction
L19+L22 a2 -0.058 / +0.006 / +0.008, **a6 -0.196 / +0.009 / +0.025** (the paper's a6: -0.103 vs -0.009); random a6 +0.001
-> dropped. The paper's mechanism (copy head + MLP loop direction, selective, dose-dependent, random control null) shows up
in our model, weaker on heads (0.58 vs 0.15 here, 0.887 vs 0.082 in the paper). Likelihood only; no length / accuracy yet.

### 3 Nov: the Astra harness (the paper as it was run), built and tested without spending
astra_harness.py: kernel (persistent CPU Python over Qwen3-0.6B's weights: NAMES / load / load_original / save_edit /
drop_edit / edited / tok; save_edit refuses the embedding, lm_head, the final norm, non-finite tensors and whole-tensor
scaling; scrubbed environment), forward tool (fresh GPU process per pass: generate original / edited / both with the
eval sampling and per-type caps, graded, paired diff +- SE and thinking saved; teacher_forced log-probs / top-k / hidden
states / MLP inputs / attention), driver (OpenRouter, openai/gpt-6-astra, reasoning effort high; one block per turn;
blocks touching training, internet, processes or the environment refused; conversation restarted with notes above 60k
tokens). Prompt astra_prompt.txt = the paper's Exp 8 prompt, rewritten for our model and goal (fewest thinking tokens at
<= ~1 point, four test types, no "close early regardless").
**HARD spending stop (user: EUR 8):** cap $8 (below EUR 8); real cost per call from OpenRouter's usage accounting (worst
case charged if absent); before each call the worst case of the next one (whole context uncached at $10/M + 8,000 reply
tokens at $50/M = at most $1.20) must fit; prompts kept far below the 272k-token price tier. Price on 3 Nov: $10/M in, $1/M
cached, $50/M out. Key only from OPENROUTER_API_KEY (Colab Secrets, loaded in cell 5, removed from the environment at once).
Tests (astra_harness_test.py, scripted Astra, real kernel on a tiny Qwen3): session loop, refusals, pass / code-run
counting, failed requests not counted, final answer; the stop never passes the cap; the key never reaches the kernel,
a log or a reply. Track stage `astra` (--as-smoke: GPU tools only, no API call); afterwards OUR checks, never shown to Astra:
paired holdout (512) and the 720 x4. Queued on GPU 1 after ch1: as0 smoke; a1 (10 passes, $8) commented until the key is set.
**3 Nov ~14:30, Astra through the user's Delos worker "Ivan" (user: up to EUR 20).** Ivan's raw endpoint
(/workers/{id}/chat/completions: stateless, OpenAI-compatible, no tools, billed to the worker) lists gpt-6-astra; 3 tiny
test calls answered (OK; 391 with reasoning_effort high and with reasoning.effort; reasoning tokens counted inside
completion_tokens). It returns token counts only (no cost, no cache info), so the driver (--provider delos) charges every
token at the public list price with NO cache discount ($10/M in, $50/M out): a conservative count. Cap $20 (< EUR 20).
Key only via Colab Secrets DELOS_API_KEY (cell 5), removed from the environment at once (both keys). Plan line a1 updated
(commented until the secret is set). Delos's own billing / markup is not visible to us: check the worker's usage page
after the first turns.

### 3 Nov: the Astra DATA experiments (idea + sandbox built; run once v1 is chosen)
User idea: give Astra what we already know about both models and let it find, by itself, an edit that beats our DPO v1.
Astra never reads the raw data in its context (weights 600M numbers, all hidden states ~700 GB): as in the paper it
writes code over files on its disk and reads short printouts; hidden states are read on demand (teacher-forced passes
over EXISTING answers, seconds to minutes, nothing new generated). New stage `astradata`: raw.jsonl (the raw Qwen's
training answers) + v1.jsonl (v1's answers, from the polans job) with id, type, sample, right, cut, thinking_tokens,
tokens and full_text (ready for a teacher-forced read), summary.json per model and type; our holdout questions removed and
fixed there. NOT the 720 set or the full test: they stay the judges (giving Astra the eval runs would let it tune to the
judge). astra --as-data / --as-goal: "vs-dpo" (start from the raw weights, beat v1) or "on-top" (--as-start v1: edit v1's
own weights to think shorter still). Not Colab itself: Astra gets the GPU only through the forward-pass tool (no Drive,
keys or other runs). Queued commented on GPU 1: astradata d1 (needs v1's answers), ax1 vs-dpo, ax2 on-top, 25 passes each,
Delos worker Ivan, $20 counted cap each. Judged afterwards on our holdout and the 720 / full test.
**Mac test session (CPU, Ivan, 2 passes):** running; so far Astra measured on 8 easy math answers (8/8 right, 394 thinking
tokens) that 227 tokens (58%) come after the right number first appears, and ranked MLP neurons against checking /
hesitation words: L22 n899, L26 n2934, L23 n716 (L22 and L26 = our script's top loop-direction writers, found independently).
Bug found and fixed: backticks inside Astra's code strings split its block (2 turns refused); fences now only count at line starts.
**3 Nov ~15:30, Mac Astra test FINISHED (Delos worker Ivan, gpt-6-astra, CPU, 12 easy math questions, 2 passes, 11 turns,
$1.10 counted at list price without cache).** Astra's own method: diagnosis (8/8 right, 394 thinking tokens, 227 after the
right number first appears), then a static scan of MLP output columns against "check / verify / Wait" token embeddings
vs arithmetic / conclusion / thinking-boundary directions; edit = remove the checking projection from 4 columns
(L22:899, L23:716, L26:2934, L23:1777 "Wait"), protecting the other directions; a precommitted rule (keep only if all right
and > 10% saved). Paired check on 8 new questions: 8/8 -> 8/8, thinking 468 -> 454 (**-3.0%**). Astra's verdict: "not a
validated repair": below its own 10% bar, too few questions, no dose-response. It left the edit submitted (it could have
dropped it with a code run). The harness works end to end with the real Astra; the result itself says nothing (8 easy
questions). The real test is the Colab run (12,000 practice questions of every type, 512 answers a round, our judges).
**3 Nov ~16:30, experiment 3 will run FIRST with Gemini 4 Argon (user's own Google AI Studio key: exact spending in AI Studio).**
The key lists 60 Google models (no "Carbon", no Astra): the strongest text model is gemini-4-argon ($2/M in, $10/M out on
OpenRouter's list: 5x cheaper than Astra). Not a strict reproduction (the paper used Astra); Astra via Ivan follows with the
identical harness, for the comparison. Driver: --provider google (Google's OpenAI-compatible endpoint, reasoning_effort high);
one test call answered 391; Argon's thinking tokens are only in total_tokens (223 vs 58 + 7) and billed as output, so the
count takes output = total - prompt. Cap $15 counted. Key only via Colab Secrets GOOGLE_API_KEY (cell 5). Queued commented
at the end of GPU 1: astra g1 (Argon, 10 passes), then a1 (Astra via Ivan).
- ~13:40 **grC step 50** (GRPO, no leash: KL 0): **-0.3% / +0.4** (svamp +2 / -0.2, mbpp +3 / +3.5, ifeval -8 / -1.3, one
  function -1 / +0.6, several +4 / -2.1, no fit -1 / -0.2). Like grA / grB at step 50: unchanged from the raw Qwen. Both
  runtimes resumed after the ~12:40 reconnect (grA from step 50 on GPU 1, grC from its last block on GPU 2).
- ~15:30 **grA step 100: -2.3% / -1.2** (svamp -1 / -2.5, mbpp -4 / +2.5, ifeval -3 / -2.5, one function -4 / -0.8, several 0 / -3.6,
  no fit 0 / -2.3); **grC step 100: -3.0% / -1.0** (svamp +1 / +0.4, mbpp -5 / -1.2, ifeval -4 / -1.7, one function -3 / -1.9,
  several -1 / -4.2, no fit -7 / +1.2). At the full penalty (λ 0.05 reached at step 100) both think slightly LONGER and lose
  ~1 point. Reading: the khudgins penalty is tiny next to the right / wrong reward (at most 0.05 x min(1, thinking / 2048):
  ~0.01 for a typical 400-token answer, against 1.0 for being right), so 12 questions x 8 answers a step give almost no
  length signal; the reward is nearly pure "be right" and GRPO drifts. grB (4x penalty) and grE (lr x2.5) test exactly
  this lever; step 150 of grA / grC next.
- ~15:45 (user: don't run useless GRPO): queued GRPO changed. grD (gate) -> **grS** = standard GRPO scaling (no Dr. GRPO),
  penalty 0.05 (rescaling alone); grE (lr x2.5) -> **grP** = standard scaling + penalty 0.5 reached at step 50. grB (0.2,
  running), grF (SmartThinker, standard scaling) unchanged; grA / grC finish their 150 steps as the confirmation.
- ~17:00 **grA step 150 (final): -5.8% / -0.9** (svamp -5 / -1.0, mbpp -1 / +1.0, ifeval -13 / -3.3, one function -3 / 0.0,
  several -5 / -1.3, no fit -4 / +1.0); **grC step 150 (final): -6.7% / -0.3** (svamp -5 / -1.5, mbpp -14 / +1.5, ifeval -7 / -1.7,
  one function -2 / 0.0, several +2 / 0.0, no fit -5 / +0.8). Confirmed: with Dr. GRPO and the 0.05 penalty, GRPO from the
  raw Qwen makes thinking LONGER over training (step 50 / 100 / 150: grA +1.5 / -2.3 / -5.8%, grC -0.3 / -3.0 / -6.7%) at
  about -0.5 to -1 point: the "be right" reward dominates and longer thinking is what it rewards. The leash (grA KL 0.04 vs
  grC 0) changes little. grS / grP (standard scaling; 10x penalty) test the fix. GPU 1: grB resumed; GPU 2: D9 Ada-R1 DPO.
- ~18:00 **grB step 100** (4x penalty, full at step 100): **+1.2% / 0.0** (svamp +2 / +0.2, mbpp +1 / +2.3, ifeval +3 / -1.5, one
  function +1 / +0.4, several -2 / -3.4, no fit -3 / +0.2). The 4x penalty stops the drift to longer thinking seen in grA / grC
  (-2.3 / -3.0% at step 100) but does not shorten either. D9 lpd-adar1 (GPU 2): 30,049 pairs (17,870 questions; 2,844 never
  solved, 11,152 always solved), ~89 pairs a step over 339 steps, training.
- ~18:15 (user): **GRPO stopped.** No GRPO run shortened thinking (grA / grC finals 6-7% longer, grB +1.2% at step 100).
  grB finishes its 150 steps; grS, grP, gs0 and grF (SmartThinker) are taken out of the queues (kept as comments). The
  GPUs now run the DPO pair-strategy screen only.
- ~19:30 **grB step 150 (final): -0.6% / -0.7** (svamp 0 / -1.9, mbpp -4 / +0.6, ifeval +1 / -1.0, one function -2 / -0.6,
  several +2 / -0.8, no fit -1 / -0.4). GRPO screen closed: no run shortened thinking. **Chirurgical smoke chs0 failed** in its
  gen phase: save_candidate built the edited copy on the GPU and kept ~3.7 GB, so vLLM found 18.28 < 18.73 GB free. Before
  that, every phase worked (200 fresh answers: 102 capped, 63 right; 10 loops -> 5 pairs; probe; mech; the edited head
  L21H8 survives bfloat16 100%). Fixed (copies built on CPU, memory freed). ch1 started with the old script (its mine phase:
  617 loop-prone questions + 2,383 others); a resume line with the fixed script follows it in the queue.
- ~20:30 **Chirurgical ch1 on real data (raw Qwen), reads and checks done:** 3,000 fresh answers (617 loop-prone questions +
  2,383 others): 121 capped (4.0%), 2,027 right; 50 looping questions (34 code, 9 instructions, 7 math) -> only 27 pairs
  (13 to find, 14 to check) after the novelty filter; 396 normal answers; dev 512, holdout 1,024. Probe GO on the copy heads
  (direction split-half cosine up to 0.71). **The same copy heads as the hand-written Mac pairs:** L16H14 0.65 vs 0.39, L18H5
  0.42 vs 0.18, L21H8 0.43 vs 0.19 (weaker contrast than the 0.58 vs 0.15 of the clean pairs). Writers L26, L27, L25, L19.
  **Mech: almost nothing is selective on real loops.** Heads x0 / x0.5: loop text only -0.002 to -0.009 nats/token (gate
  -0.01) -> dropped. Direction L26+L27: a1 drop, **a2 PASS** (loop -0.014 vs productive -0.008, normal -0.011: barely
  selective), a4 / a6 drop (productive and normal text fall as much as loops), random a6 drop. Reading: the paper's mechanism
  is there, but on our model's real loops it is weak and tangled with normal text; only dir a2 goes to the paired
  generation test (the resume line runs it with the fixed script).
- ~21:30 **Chirurgical ch1 FINISHED (raw Qwen, the paper's loop recipe, scripted): no edit kept.** ch1 failed in gen as
  expected (old script); the resume line ran gen with the fix. Paired dev test (512 questions, same seeds): **dir L26+L27 a2:
  accuracy 72.7 -> 68.9 (-3.7 +- 1.6), thinking saved +3.0% (-0.1% on answers both finished), capped 1.2% -> 2.7%** (worse);
  the random control a6: -0.8 +- 1.6, +3.7% saved, capped 1.8%. pick: none (rule: saves at no worse than -1 point). The
  edits survive bfloat16 (98-99% of entries, 100% of their size). **Conclusion:** the paper's loop fix does not transfer to
  the raw Qwen3-0.6B: its real loops are rare (4% of answers even on loop-prone questions), the loop signal is weak and
  tangled with normal thinking, and the one edit that passed the likelihood checks lost accuracy and created MORE cut
  answers. The random direction did better than the "loop" direction. Astra may still find something the fixed recipe
  cannot (experiment 3 stays queued as a reminder).
- **Astra smoke as0 (GPU tools, no API): all three work on Colab** (paired generation on 6 practice questions with per-type
  summaries; a teacher-forced read with log-probs, top-k, hidden states, MLP inputs and attention; the kernel over 311
  tensors). GPU 1 now on D1 lpd-dast; GPU 2: D9 Ada-R1 at step 150 / 338.
- ~24:00 (user): **Astra-style experiment A on v1 NOW, with Gemini 4 Argon on the user's Google key (EUR 15 -> cap $16),
  50 passes.** GPU 1: finish D1 DAST, then polans pv1 (v1 = lpc-bidax5q-s0 answers every training question x4, both halves),
  astradata dv1 (raw: the 10-answer file lupin-all-dp13-nofit1-tools1-code1-nofit2 first, the 4-answer top-ups after, each
  question from the first file that has it; + v1's answers; holdout removed), astra gv1 (on-top: Argon edits v1's own
  weights; objective in the prompt: > 50% fewer thinking tokens than the RAW model at < 1 point lost; "raw + DPO = v1,
  ~47% / -0.5"). GPU 1's other DPO runs (D2, D3, D4, D5, D10, re-scores, bidax6s) moved to the end of GPU 2's queue.
- ~00:30 Astra harness, monitoring (user: keep what Astra sees and thinks; spot red flags): Argon's thought summaries come
  back inside <thought> tags when thinking_config include_thoughts is set (Google refuses it together with
  reasoning_effort; tested): saved per turn to thoughts.md, kept out of the conversation (as the paper). Logs are written
  straight to Drive (MyDrive/thinkingcap-mvtf/runs/astra/<name>/: turns/tNNN/prompt.txt = what it saw, thoughts.md, answer.md,
  run.json = cost; messages.json = the whole context; notes.md = its notebook; final_answer.md), so nothing is lost with a
  runtime. Red-flag counters per turn: no_block, refused, forward_failed, kernel_restarts, repeated_block (possibly stuck),
  code_errors; printed on the runner's status line.
- ~01:15 Astra harness for the Argon run (user): Google's OpenAI-compatible endpoint reports no cached tokens; the NATIVE
  generateContent API does (test: 27,160 of a repeated 29,214-token prompt cached = 93%, at $0.10/M instead of $2/M) and
  returns thought parts separately. --provider google now uses the native API with the exact cost (uncached / cached /
  thinking, doubled above 200k). Conversation restart at 872k tokens (user: the paper's model_context_window), results cut
  at 10,000 characters (user: as the paper). The stop's worst case for the next call counts the part the last call had
  cached at the cached price (520k context, 500k cached: $0.34).
- ~01:40 (user): Argon also gets HOW v1 WAS MADE. astradata --ad-v1-train lpc-bidax5q: data/v1_training/ = recipe.md (the
  DPO settings, pair rules, DATASET.md sections 4-5), pairs.jsonl (every v1 training pair decoded: question, chosen,
  rejected, kind, token counts; holdout removed), checkpoints/step-10 ... step-330 + final (the LoRA adapters every 10
  steps, loadable with safetensors to see what DPO changed layer by layer). Forward passes accept
  "model": "checkpoint:step-N" (merged on the raw Qwen), for generate and teacher_forced.
- ~02:00 (user): **Astra on v1 waits for the end of the DPO screen** (v1's answers, the data folder and the Argon run are
  parked at the end of GPU 1's queue). DPO screen rebalanced: GPU 1 = D1 dast (running), D2 hard, D3 route, D4 budget,
  D5 margin, D10 l2s50e (~26 h); GPU 2 = D9 adar1 (running), D7 st1 + 2stage, D8 math, D6 simright, bidax6p, re-scores,
  bidax6s (~27 h).
- ~02:30 **Experiment B (Astra as training researcher) built but NOT scheduled** (user: keep A, decide B later): the
  TRAIN tool (forward type "train": recipe + every pair option + DPO options from a validated menu -> track.py lupin; start
  raw or v1; GPU-hour budget --train-hours, 15 h agreed; the result is "trained:<name>[:step-N]"), "compare" generation
  (any two models paired: original / edited / checkpoint:step-N / trained:...), research prompt
  (astra_prompt_research.txt). Menu validation tested (bad name, recipe, option, range refused). The run history for B
  (every run's settings + results) is still to be built when B is chosen. User's honest question: B overlaps with asking
  Claude to plan the strategy; its value is a second independent mind + a closed autonomous loop + a measurable result.
- **Test card for Astra with numbers** (user: describe the test in detail, never show it): astradata now writes
  test_card.md = astra_test_card.md (composition, protocol, scoring, rules, source -> part map, invented examples) + a
  table per part, test (the 720 screen, raw Qwen x4) vs training data (raw answers): question length p50 / p90, accuracy,
  % always / sometimes / never right, thinking p25 / p50 / p75 / p90, % cut. Aggregates only.
- **Bug caught before any run:** the runner refreshed only *.py and plan*.txt each stage, so astra_prompt*.txt and the card
  would have been missing on Colab (the Argon run would have crashed at start). It now also copies astra_*.txt and astra_*.md.
- ~04:00 **Astra sandbox test (Delos worker Ivan = gpt-6-astra, Mac CPU, small local data folder, 4 passes, $3 cap):**
  22 turns, $2.59 COUNTED (token counts at Astra's list price with no cache discount: an upper bound; the real Delos bill
  is on Ivan's credits page). Worked: the data folder (it read raw.jsonl, summary, test_card, recipe, pairs and flagged the
  empty local v1.jsonl), the kernel (an edit of layers.25.mlp.down_proj), two teacher-forced reads (log-probs, top-k, MLP
  inputs at layers 18 / 22 / 25), the generation pass (all answers cut at the 256-token CPU limit: a test limit), the
  spending stop (stopped at $2.59 / $3 keeping the reserve). Not testable on the Mac: checkpoint:step-N (no peft). Harness
  bugs found and fixed: the refusal filter matched "pty" inside "empty" (6 harmless blocks refused; now only real imports
  and calls), two blocks in one reply wasted the turn (now the first runs and the rest are reported ignored), a closing and
  an opening fence fused on one line broke the code (now split); the budget-end final message now carries the last
  forward results and the edit list (the final report had only the notes and could not state what the edit did).

### 2026-11-04 — consult1: independent strategy analysis (Ivan / gpt-6-astra, Mac CPU, raw-Qwen data only)
- Given only the raw Qwen's 71,480 training answers (numbers + 2,806 texts), the test profile and thinking on/off; nothing of our work. 13 turns, 12 code runs, $2.83 (cap $4). Report: consult1_report.md.
- Findings we had not isolated: 80% of IFEval cuts had ALREADY closed </think> (~71% of their tokens are answer-side loops, e.g. repeated quotes); code cuts 24% repeated 8-word windows vs 5% in right code; BFCL irrelevance is 56% of training tool questions vs 29% on the test.
- Limits it computed: deleting every cut answer = ~15% saving; halving all thinking = ~44%; shortest-of-4 right = 17-25% per part. So >50% needs denser reasoning on ordinary right answers, not only tail cuts.
- Ranking: A compact verified SFT (rewritten dense traces + 20-30% original replay; est. 46-67% / -1.2..+1.5), B correctness-constrained DPO/GRPO after A (42-68%), C learned readiness head for </think> (32-57%). A+B est. 52-72%.
- consult1 follow-up 1 (GRPO recipe, our runs not shown; $5.09 total): consult1_grpo_recipe.md. Start from a modest compact-SFT adapter;
  reward = right x formed x not-cut x (1 - 0.30 L/(L+B_type)), B = median right length per type (svamp 512, mbpp 768, ifeval 416,
  bfcl 160/192/192), wrong/cut/empty = 0, NO format bonus; group 8 x 32 questions, 252 steps; mean-centred advantages (no std),
  Dr. GRPO constant denominator; lr 1e-5 warm-up 12 + cosine to 1e-6; KL 0.01; clip 0.8-1.2, grad clip 0.5; LoRA r16 a32;
  mbpp rollouts 8,192; keep all-right groups (they carry the length signal). Ladder: penalty 0.45 / 0.15, lr 5e-6, KL 0.003, temp 0.8.
- Against our grA (khudgins): per-step length pressure ~10x weaker (0.05 x thinking/2048 vs 0.30 x L/(L+B)), and smaller than the
  0.1 format bonus that wrong answers also get; lr 5x lower (2e-6 constant) with KL 4x higher (0.04); 96 vs 256 completions per
  step and 150 vs 252 steps (14k vs 64k completions); code rollouts cut at 4,096; fresh AdamW every 10-step block. Sent our runs
  (consult1_our_grpo_runs.md) + the Strategy A question as follow-up 2 (cap $11).
- consult1 follow-up 2 (our GRPO runs shown + Strategy A detail; session total $10.52): consult1_grpo_diagnosis_and_sft.md.
  Diagnosis: the length contrast in our lenpen reward was 115-1,035x smaller than the right/wrong contrast (halving a tool
  trace's thinking = +0.0016 reward in grA); ramp = only ~100 full-strength-equivalent steps; thinking-only penalty (IFEval waste
  is answer-side) saturating at 2,048; grC shows KL was not the blocker; grS / grP / grF untested, not failures. Confirmed two
  trainer issues in rl_train.py: never-solved questions excluded (min_solve < rate), and the Dr. GRPO denominator counts ACTIVE
  replies only (changes with filtering). Plus fresh AdamW every 10-step block and 4,096 code rollouts. Revised first run =
  compact-SFT start, its reward on ALL tokens, KL 0.005, persistent optimizer, step-84 go/no-go (>= 5% shorter than the SFT start).
  Strategy A: rewrite verified right answers with a $2/$10 teacher (prompts per type in the file), verify (re-grade, check the
  computation/plan/argument mapping, student NLL triage), 6,048 examples (75% compact, 25% original replay, ~15% hard repairs),
  LoRA r16 a32, lr 8e-5 cosine, 2 epochs, loss on the whole reply; rewriting cost ~$45-65; go to B at >= 20% saved, >= -0.3 pt.
- ~03:30 GRPO v2 plumbing CPU smoke (Mac, hf backend, 6 easy math questions, 2 blocks x 2 steps, 2 questions x 4): passes.
  Gated reward gives a signal in all-right groups (2/2 groups every step; grA's reward gave ~0 there); code question got its
  own rollout limit; lr warm-up + cosine follows the global step across blocks (1e-5, 1e-5, 5.5e-6, 1e-6); block 2 loaded
  block 1's AdamW ("optimizer state continued"). Untested on CPU: the vLLM path with --full-support (top_k -1): smoke on GPU first.

### 4 Nov ~05:20 — D9 lpd-adar1 step 100 (720 x4)
**lpd-adar1-s0-100: +68.3% / -6.6** (full-eval weighting +68.2 / -6.7). svamp +79 / -11.7, mbpp +74 / -7.9, ifeval +58 / -1.0,
one function +52 / -6.9, several +56 / -4.6, no fit +59 / -5.6. Curve at 68.3% = -7.3, so +0.7 above it (on the curve).
Beyond D11 push2's range (max 52%); far outside the 1-point limit. Ada-R1 pairs cut very hard by step 100 (svamp, mbpp worst).
- ~06:20 **D9 lpd-adar1 step 200: +69.4% / -7.6** (svamp +80 / -13.1, mbpp +74 / -9.4, ifeval +58 / -3.3, one function +56 / -9.0,
  several +58 / -4.6, no fit +64 / -0.2), on the curve (0.0). **Final (338): +65.3% / -6.8** (svamp +79 / -10.8, mbpp +63 / -9.4,
  ifeval +55 / -1.7, one function +56 / -9.4, several +58 / -5.3, no fit +63 / -1.2), -0.3 vs curve. Ada-R1 pairs = far too much
  shortening; svamp / mbpp / one-function lose 9-13 points. Not a v1 candidate.
- **D1 lpd-dast step 100: +39.5% / -2.6** (svamp +57 / -1.9, mbpp +33 / -3.5, ifeval +22 / -3.3, one function +33 / -5.4, several
  +38 / -3.2, no fit +47 / +4.2), -1.4 vs curve. **Step 200: +30.8% / -2.0** (svamp +48 / -3.7, mbpp +22 / -2.1, ifeval +8 / -1.0,
  one function +34 / -6.0, several +37 / -2.1, no fit +45 / +5.2), -1.7 vs curve: DAST gets LONGER with training and stays below
  the curve; worse than bidax5q (46.7 / -0.5). Final pending.
- Bug found: my --nt-full code (3 Nov) was inserted into drpo() instead of nothink(), so "ntfull" re-printed the old fresh-240
  numbers. Moved into nothink(); re-queued as ntfull2 on GPU 2 (next after lpd-st1).
- ~07:20 **D1 lpd-dast final (335): +33.1% / -2.3** (svamp +48 / -3.1, mbpp +24 / -2.5, ifeval +16 / -2.3, one function +33 / -6.9,
  several +37 / -2.9, no fit +47 / +5.6), -1.8 vs curve. DAST closes below the curve at every checkpoint: dropped.
- **ntfull2: the original Qwen with thinking OFF on the FULL test (2,759 x 8)** (measurement for the blog curves; not allowed as a
  solution): svamp 88.7 -> 20.6 (-68.1), mbpp 52.4 -> 40.8 (-11.6), ifeval 58.7 -> 57.5 (-1.3), one function 82.2 -> 71.4
  (-10.7), several 76.1 -> 55.3 (-20.8), no fit 83.4 -> 90.3 (+6.9); tokens saved 89-98% (ifeval 62%: its answers stay long).
  Data in runs/mvtf/full-base-nothink (Drive).
- ~08:20 **D7 lpd-st1 final (stage 1: right-beats-wrong pairs only, 50 steps): -4.6% / +0.8** (svamp +7 / +1.0, mbpp -15 / 0.0,
  ifeval -7 / +1.5, one function -9 / 0.0, several -5 / -1.5, no fit 0 / +3.3); +1.7 vs curve. As intended: accuracy up a little,
  answers slightly longer (code 15% longer); it is the start point for lpd-2stage (bida's 1:1 pairs on top), now training.
- ~13:20 **D2 lpd-hard step 100: +9.4% / +0.1** (svamp +20 / 0.0, mbpp +7 / +0.2, ifeval -1 / +0.8, one function +9 / -0.6, several
  +9 / -2.3, no fit +12 / +1.5), -0.1 vs curve. The hard "keep trying" pairs hold length back a lot (bidax5q step 100: 41.8 / +0.2).
- **D7 lpd-2stage step 100: +16.2% / +1.7** (svamp +24 / +0.8, mbpp -5 / +2.3, ifeval +23 / +0.4, one function +13 / -1.7, several
  +9 / +0.8, no fit +36 / +10.6), +1.4 vs curve: accuracy up (no-fit +10.6) but shortening slow; steps 200 / final decide.
- ~15:20 **D2 lpd-hard step 200: +13.4% / -0.2** (-0.5 vs curve); **final: +13.8% / -0.1** (svamp +29 / -0.6, mbpp +4 / +1.0,
  ifeval +2 / -1.7, one function +14 / +0.4, several +13 / -0.2, no fit +26 / +2.3), -0.4 vs curve. The "keep trying on hard
  questions" pairs almost cancel the shortening: accuracy kept, saving tiny. Dropped.
- **D7 lpd-2stage step 200: +19.4% / +1.0** (+0.7); **final: +16.5% / +1.4** (svamp +20 / +2.3, mbpp -19 / +4.0, ifeval +35 /
  -1.7, one function +14 / -7.5, several +9 / -1.7, no fit +42 / +12.3), +1.1 vs curve. Starting from the right-beats-wrong
  stage makes the same bida pairs shorten 3x less (bidax5q 46.7%); one-function -7.5, code longer. Not a v1 candidate.
- D3 lpd-route started: 13,430 pairs (the hard + right-below pairs add many).
- ~19:20 **D8 lpd-math step 100: +40.9% / -1.6** (svamp +47 / -3.1, mbpp +39 / -2.1, ifeval +40 / -2.1, one function +34 / -2.5,
  several +38 / +2.3, no fit +41 / +2.7), -0.2 vs curve. The extra right>wrong pairs on hard math did not protect math
  (bidax5q step 100: 41.8 / +0.2, svamp -1.5). Steps 200 / final next. GPU 1: D3 route step 130/336.
- ~20:00 **D8 lpd-math step 200: +43.3% / -1.8** (-0.1 vs curve; svamp +50 / -4.6); **final: +41.7% / -1.0** (svamp +44 / -3.3,
  mbpp +29 / +1.3, ifeval +48 / -2.1, one function +41 / -2.9, several +44 / -1.5, no fit +44 / +5.4), +0.5 vs curve. On the
  curve, below bidax5q (46.7 / -0.5, +1.8): the math right>wrong pairs did not protect math. Dropped.

### 5 Nov
- ~00:30 **D3 lpd-route step 100: +7.5% / -1.0** (svamp +17 / -0.6, mbpp +1 / -1.0, ifeval -4 / -2.5, one function +10 / -1.9,
  several +12 / -0.4, no fit +19 / +2.7), -1.1 vs curve. 13,430 pairs, most of them hard / right>wrong: they cancel the shortening
  (like D2 hard). Steps 200 / final next. GPU 2: D6 simright step 250/334 after the restarts (fast restore now: f1580c2).
- ~01:30 **D3 lpd-route step 200: +6.2% / -0.6** (-0.7 vs curve); **final: +9.4% / -0.7** (svamp +20 / -0.8, mbpp +5 / -0.4,
  ifeval 0 / -3.1, one function +5 / +0.4, several +10 / +0.6, no fit +21 / +3.1), -0.9 vs curve. Hard + right pairs cancel the
  shortening (as D2). Not a v1 candidate. GPU 1 now D4 budget (7,748 pairs); GPU 2 scoring D6 simright.
- Consultant DPO session consult2 (Ivan, fresh session, best run / bida family / GRPO withheld): restarted once after the user
  asked to hide the GRPO runs ($1.98 spent on the aborted start); running, cap $18.
- **consult2 done** (Ivan, $4.17 + $1.98 aborted start): consult2_dpo_strategy.md. Recommends: six per-type DPO adapters from the
  original (4,800 pairs: math 10/40/50 shorten/guard/right, code & ifeval 65/20/15, simple 45/25/30, multiple 20/35/45, no-fit
  10/50/40), merged at per-type strengths (start 0.35/0.80/0.90/0.65/0.50/0.85), then ONE gated on-policy round (2,400 pairs,
  no math / no-fit shorten). Key findings: our "longest wrong" guards are hidden length pressure (median rejected/chosen 1.3-1.9x);
  wrong>wrong "hard" pairs cancel compression; the per-type merges (pt-*) are the strongest evidence (pt-mid 55.0 / -2.6:
  code and ifeval tolerate strong doses, math and no-fit carry the loss). Forecast 51 / -0.5, P(both thresholds) ~1/3-1/2.
  Fast alternative: repair pt-mid with a short correctness-heavy on-policy round.
- consult2 follow-up (game plan grounding, session total $8.99 + $1.98 aborted): consult2_game_plan_review.md. GAME_PLAN.md v2
  applies it: Track A repairs math + CODE + no-fit (code is -0.88 of pt-mid's loss), 1,600 pairs, NLL 0, lr 1e-5, 100 steps
  max; Track B = 4 adapters (not 6) at the historical pt settings (bf16, NLL 1.0, constant 2e-5, batch 16), quotas verified
  feasible from the existing store, merge search from the 3 historical vectors +-0.05/0.10; round-2 gate tightened to
  48-52% and >= -0.5. Awaiting the user's approval; nothing queued.
- consult2 objections reply ($10.38 total): consult2_objections_reply.md. Agrees with both user objections: fp32 + fp16
  autocast for both tracks (switch only on a clear bidax6p result); short rounds at lr 5e-6 (2.5e-6 first step), pools as
  reservoirs, early screens (A: 5/10/20, then 30/40; round 2: 4/8, then 16/24), save every 5 / 2 steps, one strength rescue
  (0.25 / 0.5) on a promising checkpoint; batches kept (16 / 64). GAME_PLAN.md -> v3.
- consult2 lpt6 reply ($15.81 total): consult2_lpt6_reply.md. lpt6 weakens Track B: the spill-over matrix is ~rank one
  (first component 95% of the log-compression matrix; simple-only vs multiple-only profiles cosine 0.991), so separate
  training isolates the gradient source, not the behaviour; Run A (joint) at 33.4 / +0.4 vs single-type adapters at
  ~33 / -1.7..-2.8. New ranking: Track A first, then a joint adapter on the balanced recipe, per-type only if a bounded
  lpt6 completion shows complementarity (13 screens: math adapter in historical bf16 settings, irrelevance solo, all six
  at 0.5, five predeclared merges C / P / S6 / P+ / T, additivity residuals). 4 vs 6: neither by default; use six to
  finish lpt6. Most informative cheap data: pt-mid's 4 adapters alone at their deployed strengths -> queued (4 screens)
  after the 4 full-strength solos on GPU 2.
- ~05:30 **D6 lpd-simright step 100: +34.1% / +0.7** (svamp +52 / -0.2, mbpp +16 / 0.0, ifeval +29 / +1.5, one function +30 /
  -2.9, several +32 / +1.5, no fit +40 / +6.7), +1.3 vs curve; **step 200: +30.3% / +0.6** (svamp +40 / +0.6, mbpp +6 / +1.0,
  ifeval +37 / -0.6, one function +31 / -4.0, several +30 / -1.5, no fit +40 / +9.4), +0.8. Accuracy kept or gained (no-fit
  strongly) but the right pairs halve the shortening vs bidax5q (41.8 / +0.2 at step 100). Final screening next.
  GPU 1: D4 budget step 55/337. Both tabs show "Connecting" while still progressing (cosmetic).
- ~06:30 **D6 lpd-simright final: +28.1% / +1.7** (svamp +35 / +0.2, mbpp +10 / +2.7, ifeval +32 / +3.1, one function +30 / -5.8,
  several +29 / -1.9, no fit +41 / +10.2), +1.8 vs curve (as high above the curve as v1, but at 28% saved): the similar-length
  right pairs buy accuracy (no-fit +10) and cost shortening; one-function is hurt (-5.8). Not a v1 candidate on its own; a
  candidate ingredient for the per-type design (Jasmin).

### 5 Nov — decisions of the day (user), state before the night
- DPO screen so far (720 x4, finals): D9 adar1 65.3 / -6.8, D11 push2 52.4 / -3.0, D8 math 41.7 / -1.0, D1 dast 33.1 / -2.3,
  D6 simright 28.1 / +1.7, D7 2stage 16.5 / +1.4, D2 hard 13.8 / -0.1, D3 route 9.4 / -0.7. None beats v1 (46.7 / -0.5).
  Still to run: D4 budget (training, step 90/337), D5 margin, D10 l2s50e, D12 chain1 rounds 1-2 (GPU 1); bidax6p / 6s (GPU 2).
- Consultant (Ivan / gpt-6-astra, consult2, ~$16 today): DPO strategy (consult2_dpo_strategy.md), game-plan grounding
  (consult2_game_plan_review.md), the user's objections accepted (consult2_objections_reply.md: fp32 both tracks, lr 5e-6 short
  rounds, early screens), lpt6 reply (consult2_lpt6_reply.md: per-type spill-over is ~one shared habit; Track A first).
- GAME_PLAN.md v3 (not committed to GPUs beyond the solos): Track A = repair pt-mid (pt-mid answers 5,400 q x4 -> 1,600 "be
  right" pairs on math / code / no-fit + maintenance, fresh LoRA on merged pt-mid, fp32, lr 5e-6, 40 steps, screens 5/10/20/40
  first, fill gaps where interesting). Track B per-type adapters + merge + mixed-step merges, to be redesigned as "Jasmin".
- Order agreed: (1) pt-mid's 4 adapters scored ALONE (8 screens: full strength + pt-mid strength), queued GPU 2, solo-svamp
  started; (2) Track A (code to build and test on the Mac, then GPU 2); (3) one sandbox with EVERYTHING (raw Qwen, thinking off,
  bida / v1 INCLUDED this time, all DPO rounds, all per-type LoRA runs, Track A) -> the consultant designs "Jasmin", the best
  per-type LoRA from scratch.
- GRPO: grA2 (grA + all trainer fixes, same data) and gr2 (consultant's recipe from v1) prepared, commented in plan_line_b;
  gr2 now screened at steps 10 20 40 60 80 120 160 200 250 (step sensitivity, --tp-score-steps).
- Ops: Chrome Memory Saver exception for Colab (user); finished plan stages commented out and old-line adapters no longer
  restored at plan start (restarts now fast); "Connecting" on the tabs is cosmetic while steps advance.
- ~07:30 **Track A built and queued** (GPU 2, after the 8 pt-mid solo screens): track.py ptrepair + repair_pairs.py + dpo.py
  --keep-order. (1) pt-mid (lpw-pertype merged 0.30/0.45/0.60/0.45) answers 5,400 training questions x4 (math 2,400, code 900,
  no-fit 1,500, ifeval 300, simple 150, multiple 150; the original's mixed-outcome questions first); (2) 1,600 be-right pairs
  (half shortest right > shortest finished wrong, half similar-length right > wrong with exactly half the right one longer;
  <= 3 per question, <= 2 right; no cut rejections) in a stratified order; (3) fresh LoRA on merged pt-mid, fp32 + fp16
  autocast, beta 0.1, no NLL, batch 16, lr 2.5e-6 then 5e-6, 40 steps, saved every 5; (4) pt-mid + repair adapter on the 720
  at 5 / 10 / 20 / 40. Tested on the Mac: pair builder (quotas, half longer, order) and 3 CPU DPO steps (keep-order, fp32).
  GPU smoke line first (rpA-smoke: 5 questions per part, 4 steps).
- ~08:30 **pt-mid adapters scored ALONE at full strength** (720 x4, no training):
  - MATH adapter: **+41.7% / -2.1** (svamp +59 / -4.8, mbpp +51 / -1.0, ifeval +26 / +0.6, simple +21 / -1.7, multiple +24 / -1.3,
    no-fit +25 / -6.0), -0.6 vs curve: trained on math pairs only, it shortens code almost as much as math (51 vs 59%) and
    hurts no-fit most: the same shared-habit spill-over as lpt6.
  - CODE adapter: **+16.5% / -1.3** (svamp +20 / -1.3, mbpp +25 / -0.2, ifeval +7 / -1.3, simple +14 / -1.5, multiple +14 /
    -2.3, no-fit +11 / -3.5), -1.6 vs curve: more own-type than the others (code 25% vs 7-20% elsewhere) and keeps code
    accuracy, unlike lpt6's code-only adapter (which did not shorten code).
  IFEval and tools solos next, then the 4 at pt-mid strength.
- ~09:30 pt-mid solos (720 x4, no training), continued:
  - IFEVAL adapter (1.0): **+36.1% / -1.2** (svamp +34 / -2.7, mbpp +42 / +0.2, ifeval +40 / +0.6, simple +28 / -1.9, multiple
    +29 / -1.3, no-fit +29 / -5.6), -0.4 vs curve: shortens code more than ifeval itself.
  - TOOLS adapter (1.0): **+50.1% / -2.9** (svamp +53 / -4.0, mbpp +59 / -4.6, ifeval +44 / -3.7, simple +38 / +1.5, multiple
    +43 / +2.5, no-fit +45 / -2.3), 0.0 vs curve: alone it already saves 50%; its own tools GAIN accuracy (simple +1.5,
    multiple +2.5) while math, code and ifeval pay: the strongest "general shorten" push of the four.
  - MATH adapter at its pt-mid strength 0.30: **+11.9% / -1.0** (svamp +20 / -1.5, mbpp +13 / +0.8, ifeval +4 / -1.5, simple
    +9 / -1.5, multiple +10 / -2.3, no-fit +11 / -1.9), -1.3 vs curve.
  Reading so far: every pt-mid adapter spills over (math -> code 51%, ifeval -> code 42%, tools -> code 59%); the merge's 55%
  is mostly stacked general pushes, as lpt6 suggested; own types tend to KEEP accuracy (tools +1.5/+2.5, ifeval +0.6, code
  -0.2), the loss lands on the OTHER types. Remaining: code / ifeval / tools at pt-mid strength.
- ~10:30 **D4 lpd-budget step 100: +43.8% / -1.8** (svamp +58 / -4.6, mbpp +44 / -1.7, ifeval +31 / -0.8, simple +35 / -0.8, multiple
  +39 / +0.6, no-fit +39 / -0.6), 0.0 vs curve; **step 200: +48.6% / -1.5** (svamp +61 / -3.1, mbpp +40 / -1.2, ifeval +46 / -0.8,
  simple +41 / -2.3, multiple +45 / +0.2, no-fit +47 / +0.2), +1.1 vs curve. At bidax5q's step-200 saving (48.5 / -0.9, +1.7) but
  0.6 less accurate; math carries it (-3.1). Final next.
- pt-mid solos at pt-mid strength: CODE 0.45 **+10.0% / -2.5** (-2.7 vs curve; ifeval -3.7, math -2.5: likely partly screen noise,
  since the same adapter at 1.0 read -1.3); IFEVAL 0.60 **+23.1% / -1.2** (-1.4; no-fit -3.3, multiple -2.7). Tools at 0.45 next,
  then the additivity check vs pt-mid (55.0 / -2.6), then Track A.
- ~11:30 **D4 lpd-budget final: +49.0% / -1.3** (svamp +60 / -2.9, mbpp +43 / -0.2, ifeval +47 / -1.9, simple +40 / -2.5, multiple
  +45 / +1.7, no-fit +46 / +0.4), **+1.4 vs curve**: the best D-variant so far, close to v1 (46.7 / -0.5, +1.8) but not above it;
  math carries the loss.
- pt-mid solo TOOLS at 0.45: **+28.6% / -1.1** (svamp +27 / -2.9, mbpp +32 / +0.4, ifeval +28 / -0.8, simple +25 / -2.5, multiple
  +28 / -0.2, no-fit +28 / -0.8), -1.0 vs curve.
- **Additivity check (pt-mid's 4 solos at their strengths vs the measured pt-mid merge 55.0 / -2.6):** token ratios multiply
  (0.881 x 0.900 x 0.769 x 0.714 = 0.435 -> predicted 56.5% saved, measured 55.0: lengths ADD almost exactly as stacked
  pushes); accuracy losses do NOT add (sum -5.8 predicted, -2.6 measured: +3.2 better than additive). Merging protects more
  than the sum of its parts (each adapter's own-type "be right" pairs offset the others' spill-over).
- **Track A smoke passed on the GPU** (answers, pairs, 4 training steps); the real run rpA started: 4,372 questions (svamp
  2,400 of which all 1,644 mixed; mbpp 900; no-fit 472 = all 260 mixed + the rest; ifeval / simple / multiple 300 / 150 / 150),
  pt-mid answering x4 (2,400 / 4,372 sampled at 11:30).
- ~12:30 **Track A rpA**: pairs exactly as specified (1,600 from 1,233 questions; every quota filled; chosen longer in 43-52%
  per part, median chosen ~= rejected length: no length signal, as intended; first 80 pairs span all six parts). Training
  (40 steps, lr 5e-6, fp32): DPO loss 0.693 -> 0.667, ordered right 54-62% until step 35, 71% at 40: a very gentle dose.
  **rpA-5 (pt-mid + repair at step 5): +53.9% / -3.4** (svamp +61 / -3.7, mbpp +64 / -6.0, ifeval +46 / -1.9, simple +37 / -0.2,
  multiple +41 / +0.4, no-fit +42 / -5.8), +0.3 vs curve: at step 5 the repair has barely moved the model, so this is ~pt-mid
  re-measured (55.0 / -2.6 on 1 Nov; -0.8 apart = screen noise and the merge path). Steps 10 / 20 / 40 next.
- ~13:30 **Track A rpA curve** (pt-mid + be-right repair, lr 5e-6): step 10 **+53.6 / -2.7** (+0.9 vs curve), step 20 **+54.1 /
  -3.6** (+0.1), step 40 **+52.3 / -2.8** (+0.5); step 5 was +53.9 / -3.4. Flat within screen noise (+-0.7): the repair at this
  dose did not move accuracy overall. Per type: no-fit improves steadily (-5.8 -> -5.2 -> -3.7 -> -1.7, the one clear signal),
  code -6.0 -> -2.5 (noisy), math stays -3.7..-5.4, ifeval worsens (-1.9 -> -4.2). Training barely moved (loss 0.693 -> 0.667).
  Verdict: lr 5e-6 x 40 steps is too gentle to repair; not a candidate. Next candidate (user to decide): the same pairs at a
  stronger dose (lr 2e-5, as D7 stage 1; or 1e-5 x 80 steps), screens 10 / 20 / 40; the pairs and pt-mid's answers are reused.
  GPU 2 moved on to bidax6p (the precision-only test); GPU 1 D5 margin trained, screening.
- ~13:45 (user) Track A stronger doses queued, same pt-mid answers and pairs as rpA (ptrepair --rp-from): rpB lr 2e-5 x 40
  steps (screens 10 / 20 / 40) on GPU 1 right after the D5 margin screens; rpC lr 1e-5 x 80 steps (screens 10 / 20 / 40 / 80)
  on GPU 2 right after the bidax6p screens.
- ~14:30 **D5 lpd-margin** (narrowest 15% of each pair kind dropped): step 100 **+55.0 / -3.1** (+0.8 vs curve), step 200 **+56.9 /
  -3.8** (+0.5), **final +53.3 / -2.6** (svamp +65 / -4.2, mbpp +50 / -2.7, ifeval +50 / -2.7, simple +42 / -3.5, multiple +46 /
  -0.8, no-fit +49 / +2.3), +1.0. Keeping only wide-margin pairs = more shortening at more accuracy cost (on the curve, below
  v1). Not a v1 candidate.
- Queue note: GPU 1 had already re-read its plan before rpB was pushed, so it started D10 l2s50e first; rpB runs right after
  D10 (~5 h later than intended). rpC stays after bidax6p on GPU 2 (bidax6p at step 190/336).
- ~15:30 **bidax6p (bidax4's pairs, our settings, ONLY the precision switched to fp32 + fp16 autocast)**: step 100 **+45.7 / -1.8**
  (svamp +55 / -2.3, mbpp +43 / -2.3, ifeval +40 / -2.9, simple +39 / -2.7, multiple +44 / -0.4, no-fit +45 / +3.7), +0.3 vs curve;
  step 200 **+48.9 / -2.0** (svamp +52 / -3.5, mbpp +45 / -0.6, ifeval +53 / -3.1, simple +43 / -5.0, multiple +46 / -0.8, no-fit
  +49 / +3.1), +0.7. Same pairs in bf16 (bidax4): 100 +57.1 / -5.1, 200 +54.6 / -3.7; with ALL of his settings (bidax6): 100
  +44.4 / -2.2, 200 +40.1 / -0.9. Reading: the precision alone accounts for most of the change his settings made (much gentler,
  -11 pts saving and +3 pts accuracy at step 100): float32 + fp16 autocast trains less aggressively than our bf16 path. It moves
  the model along the curve (from below to slightly above) rather than giving a big jump above it. Final next; then rpC.
- ~16:30 bidax6p step 300: **+47.8 / -1.4** (svamp +57 / -1.7, mbpp +39 / -0.4, ifeval +48 / -1.9, simple +42 / -6.5, multiple
  +46 / -0.8, no-fit +48 / +2.5), +1.1 vs curve. Final next, then rpC on GPU 2.
- ~17:30 (user) **Finish lpt6 queued** (GPU 2, after rpC; the consultant's 13-screen design): (a) train the missing math adapter
  (typeonly lpt6 --to-parts svamp, bidax5s pairs, 2 seeds, lpt6's exact historical settings); (b) solos at 1.0: math + no-fit;
  (c) all six solos at 0.5 (rescreen --cx-scale 0.5); (d) five merges via the new mixscreen stage (math / code / ifeval / simple /
  multiple / no-fit): C 0/0/.5/.5/.5/0, P .5/0/.5/.5/.5/.5, S6 all .5, P+ 1/0/.5/.5/.5/1, T .5/0/.5/0/1/.5. Purpose: a second,
  differently-trained per-type family (6 adapters, rule-B pairs, no NLL, cosine) to compare with pt-mid's (4 adapters, LUPIN
  pairs, NLL 1.0, constant lr): solos, dose and merges, as data for the Jasmin battle plan.
- Deep research for Jasmin (107 agents, 24 sources, 23 of 25 claims confirmed): research_jasmin_merging.md. No source tests
  0.6B, per-type DPO LoRA merging, or non-math tasks. Supported: separate-then-merge beats data mixing for 2-5 sources at 3-13B
  (DEM, Cohere Aya, LoRA Soups), DPO merges more consistent than SFT merges; the advantage fades with more skills (3+: data mix
  won; TIES collapses with merge count; Task Arithmetic the only reliable method); plain shortest-vs-longest DPO cuts ~17-30%
  only; -50% at < 1 pt at 1.5B came from on-policy RL with correctness-aware length rewards (DRPO; AdaptThink uses skipping
  thinking = not allowed here) or SEARCHED / layer-wise merge coefficients (Evo-L2S -52% at +4.6 pts, FIM-TIES); uniform merges
  degrade at 1.5B ("smaller models struggle to learn from model merging"); DARE needs drop < 0.5.
- ~19:30 **bidax6p final: +48.8 / -2.3** (svamp +57 / -3.1, mbpp +39 / -0.6, ifeval +52 / -5.2, simple +42 / -5.2, multiple +46 /
  +0.2, no-fit +48 / +3.7), +0.4 vs curve. Precision alone: best at step 300 (+1.1); his schedule (bidax6s) still to come.
- **rpB-10 (pt-mid + repair at lr 2e-5, step 10): +52.4 / -2.2** (svamp +58 / -5.8, mbpp +63 / -0.8, ifeval +46 / -0.8, simple +37 /
  -2.5, multiple +40 / +0.6, no-fit +42 / -1.5), +1.2 vs curve: code and no-fit repaired (mbpp -6.0 -> -0.8, no-fit -5.8 -> -1.5
  vs rpA-5), math still -5.8. Training moved this time (loss 0.693 -> 0.578, 75% ordered right at step 40). Steps 20 / 40 next.
  rpC (lr 1e-5 x 80) training on GPU 2, step 25/80.
- ~21:30 **Track A rpB (lr 2e-5) curve**: step 10 +52.4 / -2.2 (+1.2), **step 20 +48.7 / -1.1** (svamp +53 / -1.9, mbpp +56 / -1.9,
  ifeval +45 / -1.5, simple +35 / -1.0, multiple +38 / +1.5, no-fit +41 / +2.1), **+1.55 vs curve**; **step 40 +43.4 / +0.3** (svamp
  +45 / -3.1, mbpp +48 / +1.0, ifeval +44 / +1.7, simple +31 / -3.1, multiple +36 / +0.8, no-fit +41 / +7.3), **+2.1 vs curve: the
  highest above the curve of any model on the current data** (v1 +1.8). The repair works at this dose: accuracy comes back
  (code, ifeval, no-fit fully), but so does length (55 -> 43% saved); the points move ALONG a frontier above v1's. Not yet
  >= 50 / -1. Natural next probes: strength rescue (rpB-40 at 0.5, rpB-20 at 0.5 / 0.75) to land between 20 and 40 at ~50 / -1,
  and a second seed. rpC-10 (lr 1e-5): +53.6 / -3.0 (+0.6); rpC 20 / 40 / 80 next.
- ~22:30 rpC (lr 1e-5 x 80): **step 20 +50.5 / -1.8** (svamp +57 / -4.0, mbpp +61 / -2.5, ifeval +41 / -0.6, simple +36 / -0.6, multiple
  +40 / +2.1, no-fit +42 / -1.5), +1.2 vs curve; **step 40 +51.2 / -2.1** (svamp +58 / -4.0, mbpp +61 / -4.4, ifeval +43 / -1.3, simple
  +36 / -0.2, multiple +39 / +1.1, no-fit +41 / +2.7), +1.0. Gentler than rpB: keeps ~51% saved but math (-4.0) and code stay
  damaged; step 80 next. GPU 1: D10 l2s50e step 250/332.
- 5 Nov ~13:50 Claude session restart: both Colab tabs lost their output; the user reconnected, plans re-run from a47f4cf
  (rs-bidax6p, rpB commented as done). GPU 1 resumed D10 l2s50e after step 310; GPU 2 skipped the finished rpC screens.
- **rpC-80 (lr 1e-5, step 80): +43.4 / +0.6** (svamp +44 / -1.7, mbpp +47 / +2.3, ifeval +46 / +0.6, simple +31 / -2.7, multiple
  +35 / +1.1, no-fit +39 / +5.4), **+2.4 vs curve: new highest above the curve** (rpB-40 +2.1 at the same 43.4% saved, +0.3 pts).
  rpC's four points (53.6/-3.0, 50.5/-1.8, 51.2/-2.1, 43.4/+0.6) and rpB's lie on one frontier: the repair trades length for
  accuracy, the gentler lr lands slightly higher at the end. Still no point at >= 50 / -1. Track A done; finish-lpt6 started
  (svamp adapter seed 1 training, 241 steps).
- ~15:20 **D10 lpd-l2s50e** (bidax5q s0 settings, l2s50e pair rule: shorten only where solve >= 50%, easy ratio 1.2, shorten cap 2):
  **step 100 +54.4 / -3.6** (svamp +71 / -7.9, mbpp +59 / -3.7, ifeval +37 / -0.2, simple +41 / -2.5, multiple +45 / +0.4, no-fit +44 /
  -5.4), +0.2 vs curve; **step 200 +49.7 / -3.0** (svamp +64 / -6.0, mbpp +46 / -1.5, ifeval +40 / -4.0, simple +42 / -2.5, multiple +45
  / +0.0, no-fit +47 / +1.2), -0.2 vs curve. On the curve, no better than D11 push2 at the same saving (push2-200 49.7 / -2.9); math
  hit hardest. Final screen next. GPU 2: lpt6 svamp seed 1 step 200/241.
- ~15:50 **D10 lpd-l2s50e final: +53.5 / -2.7** (svamp +68 / -5.4, mbpp +50 / -3.1, ifeval +46 / -1.0, simple +43 / -3.3, multiple +46
  / -2.1, no-fit +48 / +1.2), +0.8 vs curve; beats D11 push2 final (52.4 / -3.0) slightly at a bit more saving. Math still the cost.
  GPU 1 now trains D12 lpd-chain1 (333 steps x 18 pairs). GPU 2: lpt6 svamp merged and screening.
- ~16:15 **Finish lpt6, missing solos at full strength** (2 seeds souped, bfloat16, our Nov 2 settings, bidax5s's pair set):
  - **math only (lpt6-svamp): -38.1 / -3.1** (answers LONGER: svamp -78% / -16.2; mbpp -17 / +2.3, ifeval -39 / +1.5, simple -23 /
    -1.7, multiple -23 / 0.0, no-fit -29 / +2.5). bidax5s's math pairs (shorten 1510, guard 2387) lengthen math, as bidax5s itself
    did (svamp -13 to -36%); the spill-over lengthens every type. Off the curve's range (fitted above 15% saved).
  - **no-fit only (lpt6-bfcl_irrelevance): +11.9 / -1.9** (no-fit +36 / +11.7 but simple +18 / -20.6, multiple +17 / -16.6; ifeval
    +3 / -0.4, mbpp +9 / +1.0, svamp +11 / +0.2), -2.2 vs curve: it teaches "no function fits" and the model then refuses calls on
    the other tool types.
  Both are poor merge ingredients at 1.0; the half-strength solos (rs-lpt6h) are screening now, merges after.
- ~17:10 (user: let the consultant redesign the lpt6 merges) **consult3** (new session; consult2's state was lost with the
  scratchpad; data-only pack: all_runs.json 132 runs, per_type.md, test card, thinking-off, his lpt6 reply; $3.17, 11 code
  runs): consult3_merge_design.md. Cancel P / S6 / Pplus / T (lpt6 math and no-fit are not protectors), keep C as a
  diagnostic; 10 ranked screens mostly on the pt family (drop code; shift strength from math/code to instructions/tools;
  cross-family swaps of tools and instructions; a weak no-fit 0.25; pt-math in C; a repair on the redistributed base), with
  decision rules on the half-strength lpt6 solos. Jasmin default: 4 groups (pooled tools) + the 3 tool slices as
  challengers (7 adapters), pt pair rule, NLL 1.0, constant lr 2e-5, float32, ~N/16 steps with checkpoints every 10,
  task arithmetic starting at math .20 / code .15 / instr .60 / tools .55. Supports the user's candidate with 2 changes.
  Awaiting the user's decision; the merges stay paused.
- ~17:20 lpt6 math solo at 0.5: **-12.2 / -0.5** (svamp -26% / -2.9, ifeval -15 / +1.5, mbpp -3 / -0.4, simple -6 / -1.2,
  multiple -6 / -0.8, no-fit -6 / +1.9): still longer everywhere and worse on math -> by consult3's rule A, out of all merges.
  GPU 1: D12 chain1 step 250/333.
- ~17:40 (user OK) queued on GPU 2 after rs-lpt6h: consult3 screens #1 pt-nocode (math .30 / code 0 / instr .60 / tools .45)
  and #2 pt-shift (.15 / 0 / .65 / .65), no training; then the Jasmin pair sets, October's lpv-base rule (l2s50e: shortest
  right > right >= 1.5x longer on questions solved >= 50%, easy 1.2x; shortest right > longest wrong; no-fit protection:
  solve >= 0.8 + right pairs) on the NEW data (dp13 + nofit1, tools1, code1, nofit2, nofit4, math1, math2; 10 answers per
  question, as in October): jpa = matched like October (lupin_select.py against the original model's runs on the choosing
  set + questions_40, both drawn from the evaluation questions), jpb = unmatched, cut at random to jpa's count per part x
  kind. Pairs only (lupin --lp-pairs-only), counts and median lengths reported, no training yet. User's agreed Jasmin
  changes: math adapter + a few "be right" pairs; pooled tools adapter (no-fit protection) + the 3 slices as challengers;
  code adapter possibly dropped (screen #1). Next (user): variations of each rule (the 1.5x ratio, the 50% solve floor and
  1.2x easy ratio, per-type tweaks) decided by the data.
- ~18:00 lpt6 code solo at 0.5: **-0.8 / +0.3** (mbpp -2% / +2.3, ifeval -10 / +0.2, multiple +6 / -3.2, rest ~0): no
  shortening at all, as at 1.0. GPU 1: D12 chain1 trained (333 steps), screens at 100 / 200 / final starting.
- ~18:05 **consult3 follow-up on the Jasmin pair sets** (code + jasmin_pairs.md given; $9.40, total $12.57; 26 code runs):
  consult3_jasmin_pairs.md. Build jpa as a frozen replica (no math supplement), but October's rule does not mean October's
  behaviour on a new pool: the shorten cap is 2x GUARD pairs only (per part, before matching); an empty matching cell gets
  its share redistributed but a nearly empty one stays short (one example can cut the total 39%); every part is trimmed
  to the weakest part's fill (how 3,200 requested became ~1,400; nofit4 may release it and give every adapter more steps);
  no-fit protection switches at 8/10 while the matcher pools 5-9/10; the 1.2 easy ratio applies only to 10/10; right
  band 0.25; lengths are total output tokens. jpb (random pairs at jpa's counts) tests the whole selector, not the
  matching alone: add jpb-RR (question-first, same counts and questions per pair) and jpc-trainref (references from
  held-out TRAINING questions, at pool AND pair stages, for a clean rules path). Variants in order: matching policy;
  1.5 -> 2.0 / 1.25; floor .5 -> .6 / .4; easy 1.2 -> 1.5 / 1.1; math supplement 0 / 5 / 10%; no-fit floor .8 vs .9; with
  pre-cap / post-cap / post-match / final reports (k/10 solve, sources, unique questions, lengths, steps at batch 16).
  Math supplement: 10% of the math budget as similar-length right > finished wrong (within 15%, half right-longer) on
  5-7/10 questions, 1 per question. Train only the affected adapter (fp32, NLL 1, constant 2e-5, ckpt every 10), solo
  screen own x4 / others x2, compare at matched saving and matched exposure.
- ~18:30 lpt6 instructions solo at 0.5: **+17.8 / -0.2** (ifeval +19 / -0.6, svamp +19 / +0.4, mbpp +18 / +0.8, simple +14 /
  -2.1, multiple +18 / 0.0, no-fit +12 / -1.7), -0.5 vs curve: at half strength a mild shared shortening
  (at 1.0: 32.6 / -2.8). GPU 1: D12 chain1 step-100 screen grading.
- ~18:40 (user) bidax6s (bida's full schedule) DROPPED from GPU 2's queue.
- ~18:45 (user) GRPO stays off. Let every per-type DPO LoRA run finish (rs-lpt6h, consult3 screens #1/#2), then gather all per-type data -> battle plan -> try it (Jasmin).
- **D12 lpd-chain1 step 100: +21.7 / +0.4** (svamp +36 / -0.6, mbpp +11 / +0.2, ifeval +10 / +1.7, simple +25 / -1.0, multiple
  +27 / -0.4, no-fit +29 / +2.3), +0.2 vs curve: the chain-1 rule (shortest right > longest right >= 1.5x; guard =
  shortest right > shortest finished wrong; balanced 40:60) trains slowly, far below 40% at step 100. Steps 200 / final next.
- ~19:20 **D12 lpd-chain1 step 200 +18.4 / +0.4** (svamp +26 / -0.2, mbpp +9 / -0.4, ifeval +8 / +1.0, simple +27 / -2.1, multiple
  +26 / +0.6, no-fit +33 / +4.8), +0.1 vs curve; **final +17.8 / +0.8** (svamp +28 / -0.2, mbpp +5 / +2.5, ifeval +6 / 0.0, simple
  +24 / -2.5, multiple +25 / -0.6, no-fit +36 / +5.6), +0.5 vs curve. Chain-1 keeps accuracy but saves under 20%: far from
  the 40% bar, below D11 push2 in saving. Round 2 (polans pc1 a + b, then lpd-chain1r2) running.
- lpt6 half-strength solos: **one function @0.5 +15.3 / -0.7** (simple +18 / +0.2; no-fit +18 / -3.3, ifeval +4 / -1.0), -1.0 vs
  curve; **several functions @0.5 +8.3 / -0.2** (multiple +14 / +0.6; no-fit +10 / -2.1, ifeval +1 / -1.9), -0.4 vs curve.
  Mild shared shortening; the cost lands on no-fit again. No-fit @0.5 next, then consult3 screens #1 / #2.
- ~20:20 GPU 2's queue ENDED (screens ran ~15-25 min each, faster than planned):
  - lpt6 no-fit @0.5: **+4.5 / -0.6** (no-fit +23 / +4.8; simple +10 / -4.0, multiple +10 / -3.4, ifeval -3 / -1.7), -0.6 vs
    curve: half strength keeps the "no call" bias (callable -4.0 / -3.4) -> consult3 rule B: cancel screen #6.
  - **consult3 #1 pt-nocode (math .30 / code 0 / instr .60 / tools .45): +50.8 / -2.7** (svamp +58 / -4.0, mbpp +60 / -2.5, ifeval
    +44 / -2.5, simple +36 / -1.7, multiple +40 / +1.3, no-fit +40 / -4.8), +0.35 vs curve (pt-mid +1.3). Dropping code loses
    4.2 points of saving at the SAME accuracy (pt-mid 55.0 / -2.6): code still saves 60% without its own adapter (spill-over),
    and the code adapter costs nothing in the merge. Prediction was 51 / -1.5: wrong on accuracy -> keep a code adapter.
  - **consult3 #2 pt-shift (.15 / 0 / .65 / .65): +54.4 / -3.9** (svamp +59 / -5.4, mbpp +63 / -5.0, ifeval +49 / -2.7, simple +39 /
    -1.9, multiple +44 / +1.7, no-fit +45 / -7.3), -0.1 vs curve. Prediction 55 / -2.0: wrong. Less math strength does NOT
    protect math (-5.4 with math at .15): math's loss comes from the instructions / tools spill-over; more tools strength hurts
    no-fit (-7.3) and code (-5.0). pt-mid's mix stays the best of the pt merges (+1.3).
  - rescore-old (old line, reference only): drpo3d-23 +21.7 / -0.4, drpo3f-15 +26.5 / +0.1, stopB3-soup-25 +39.6 / -3.2.
  - lpw-2stage step 50 (dp13, "be right" round then l2s50e): **+29.5 / +1.6** (no-fit +37 / +6.5, multiple +32 / +2.9, mbpp +16 /
    +3.7, simple +27 / -4.0), +1.8 vs curve.
  GPU 2 is IDLE. GPU 1: polans pc1-a (chain1 answers 10,068 questions x4), then pc1-b, chain1r2.
- ~22:15 (user) **DPO story page** https://claude.ai/artifact/AVfDkpzQ62tuXuwwgciC21 (Dashboard type): every scored checkpoint
  (144) on one chart with our curve, the README bar (> 20% shorter, <= 0.2 pt per 1% saved), the target box and the references
  (raw, thinking off on the FULL test ~90% / -22.3 [saved approximate: 62-98% per part], michal-bida, Lightning McQwen3);
  one-row-per-run tables for the joint runs grouped by idea and for the per-type families + repair; lessons. README bar also
  added to FullEval (dotted line, "vs README bar" in tooltips) and ModelTree ("vs README" column): 109 of 144 runs pass, none at
  >= 20% saved falls below; the 35 that fail are all under 20% saved.
- ~22:35 (user) public-submission search: BottleCapAI/ThinkingCap-Test has 5 forks: michal-bida, basta/lightning-mcqwen3 (known),
  horychtom and miroslavmyrha (untouched copies), marek-krcal/MakeThinkingAffordableAgain (in progress: tooling rewrite on
  branches, SimPO/DPO planned, no scores yet). New on Hugging Face: **mrtineu/thinkingcap-test-submission (Martin Senkyr,
  3 Nov): 28.7% / -1.71 full test** (math 40 / -2.3, code 16 / -2.0, ifeval 31 / -1.1, simple 24 / -1.2, multiple 28 / -4.3,
  no-fit 26 / +0.2), -1.6 vs our curve, +4.0 vs the README bar: one extra MLP layer after layer 13 (198k trained params) on
  short answers the model wrote with Qwen3-8B hints. Added as a reference to FullEval and the DPO story page.
- ~22:50 (user) **consult4** started: the consultant FROM SCRATCH (no earlier reports, none of our decisions) on all per-type data: raw Qwen per full-test question, thinking off, 144 checkpoints, run summaries, references, per-type families, every run command + code; question = best strategy for > 50% / better than -1 with our compute. Cap $25, 60 code runs.
- ~23:05 **consult4 report** (from scratch, $8.56, 20 code runs): consult4_strategy.md. Audit: our screen totals weight the
  three BFCL parts equally, not 400/200/240 (rpC-80 +0.6 -> +0.43, rpB-40 +0.3 -> +0.10, bidax5q -0.5 -> -0.59; the
  "full-eval weighting" line is the official one); --bida-exact sets held-out pairs to 0 (no tripwire). Strategy: NOT six
  specialists; J = a joint DPO adapter on all six types (5,424 pairs = 339 steps x 16: shorten = shortest right > longest
  right >= 1.5x, solve >= .75 for math / multiple / no-fit and >= .5 elsewhere, no easy ratio; guard = shortest right >
  longest wrong incl. capped; ~7% length-balanced right pairs; <= 2 pairs per question; fp32 + fp16 autocast, beta 0.1,
  NLL 0, lr 2e-5 warm-up 16 + cosine to 0, 2 seeds averaged in weight space) + a bida-style control; I = an
  instruction-only adapter (80 steps) merged at 0.25-0.5 as a compression booster; R = one balanced repair on the merged
  model (1,280 pairs: 50% length-balanced right, 25% anti-overcut, 25% shortening replay; lr 1e-5, NLL 0.1, <= 80 steps,
  strength 0 / .5 / 1). Selection on a NON-benchmark dev set (600) + a locked confirmation set (600), at most 12 screens,
  one frozen candidate, one full test. Expected ~50-52% / -0.8, P(both thresholds) 25-40%. Alternatives: (1) joint + one
  on-policy round; (2) October 4-adapter reproduction + balanced repair.
- ~23:15 **Official weighting adopted** (consult4 audit, checked: recomputing each 720 score with BFCL pooled 400/200/240
  reproduces the printed "full-eval weighting" lines within 0.1). All 117 screen scores in FullEval / ModelTree / the DPO
  story page now carry the official overall accuracy (old equal-BFCL value kept as acc_screen_eq); shifts -0.5 to +0.1.
  Key runs: rpC-80 +0.43 (+2.2 vs curve), rpB-40 +0.10 (+1.9), bidax5q final -0.59 (+1.7), pt-mid -2.58 (+1.3). Still 0
  checkpoints in the target box. The yardstick formula is unchanged (fixed 2 Nov); FullEval's own live fit refits itself.
  From now on the log uses the full-eval-weighting line as the overall accuracy.
- ~23:45 (user) queued on GPU 2 after chain1 answers half b: FULL TEST of pt-mid (pertype --pt-full, new option: merge + official 2,759 x 8 against full-base).
- ~00:00 (user) queued on GPU 1 after chain1 round 2: five lpt6 merges (C, C+code, C+weak no-fit .25, tools-only 3 slices, C at .7), 2-seed averages, no math.
- ~00:10 (user) chain1 round 2 POSTPONED on both GPUs (pc1 answers so far kept on Drive: ~4,000 of 17,000, resumable); GPU 2 -> pt-mid full test now, GPU 1 -> the five lpt6 merges now.
- ~00:30 6 Nov (user) **October per-type run redone with only two changes** queued on GPU 1 after the lpt6 merges: the new
  training data (jpa = October's lpv-base rule and matching on dp13 + nofit1, tools1, code1, nofit2, nofit4, math1, math2,
  10 answers) and float32 (dpo.py --dtype float32: pure fp32); same 4 groups, NLL 1, constant lr 2e-5, 16 pairs/step, one
  pass, 48 held out, seed 0; each group x kind cut at random to lpv-base's count (pertype --pt-cap-like, new option) so each
  adapter takes October's number of steps; scored like October: 4 solos at 1.0, 4 at pt-mid strength, softer / soft / mid.
- ~00:50 (user) jpa built and its capped counts CHECKED before training (pertype --pt-check, new: prints available / October / kept pairs and steps per group, no training), right after lpt6mix-C on GPU 1.
- ~00:20 6 Nov **lpt6mix-C (instructions + one function + several functions, each 0.5, 2-seed averages): +41.3 / -1.3** official
  (screen-equal -1.3; svamp +41 / -1.7, mbpp +45 / -1.0, ifeval +41 / -0.6, simple +36 / -0.2, multiple +39 / +3.6, no-fit +37 / -9.4),
  +0.15 vs curve. The merge saves MORE than its solos predict (token ratios multiplied: 36.2%; consult3 guessed 40 / -2.5); accuracy
  close to the solos' sum (about -1.1); the whole cost is no-fit -9.4 (no no-fit adapter, two call-making adapters).
- ~00:45 (user) the October rerun's adapters save a checkpoint every 5 steps (pertype --pt-save 5, new; adapters/lpj-<group>/step-N, synced to Drive) so any step can be scored or restored later.
- ~01:10 BUG (mine): lupin --lp-pairs-only had been inserted into balround(), not lupin(); jpa built its pairs (2,812 matched: math 802, code 780, instr 799, simple 335, multiple 182, no-fit 210; vs ~1,400 in October) then started an unwanted joint training (jpa-s0). Fixed (moved into lupin).
- ~01:35 6 Nov **FULL TEST of pt-mid** (October 4-adapter merge .30/.45/.60/.45, rebuilt from the saved adapters, no repair; 2,759 x 8):
  **+53.6 / -3.3** official (svamp +61 / -6.0, mbpp +62 / -3.9, ifeval +50 / -1.7, simple +35 / +0.2, multiple +38 / +0.4, no-fit
  +40 / -5.6), +0.3 vs curve. Its 720 screen re-ran in the same stage: 55.1 / -2.6 (= Oct 31's 55.0 / -2.6). The 720 read 1.4 high on
  saving and 0.7 favourable on accuracy: the October strengths were picked on screens. On the full test pt-mid sits just above
  the curve, below michal-bida (47.0 / -0.78, +1.6); the loss is math and no-fit, as on the screen, plus code.
- jpa cap CHECK (lpj, before training): every group reaches October's exact counts, no shortfall: math shorten 299 / guard 47 /
  right 7 (available 599 / 87 / 17), code 251 / 79 / 23 (497 / 156 / 50), instructions 218 / 86 / 49 (454 / 146 / 103), tools 284 /
  44 / 25 (574 / 89 / 40); 353 pairs per adapter = 19 steps at 16 pairs/step after 48 held out, as in October.
- ~01:40 (user) the October rerun is EXTENDED instead of stopped: per adapter the October-count pairs first (shuffled, seed 0),
  then the remaining matched jpa pairs (~703 in all), trained in that order (--mix all --keep-order), constant lr, float32,
  checkpoints every 5 steps + step 19 (dpo.py --save-at, new) = the capped run's last full step. GPU 2 (idle) trains it now and
  scores the October comparison at step 19 (pertype --pt-step 19: solos at 1.0 and pt-mid strength, softer / soft / mid);
  GPU 1 after its merges scores each adapter alone at steps 25 and 40 (pertype --pt-no-train --pt-wait 3). New pertype
  options: --pt-extend, --pt-step, --pt-wait.
- ~01:50 (user) dose scores changed to steps 10 and 30 (+ the full October comparison at step 19 on GPU 2); every 5th step stays stored on Drive.
- ~02:00 (user) drivereport extended (sub-folders > 100 MB and files > 300 MB with dates, read only) and queued on GPU 1 between two merges: Drive clean-up list first, nothing deleted.
- ~02:30 lpt6 merges: **C + code .5: +39.5 / -2.8** (svamp +42 / -1.9, mbpp +40 / -3.1, ifeval +37 / -3.3, simple +37 / -2.1, multiple +40
  / +1.7, no-fit +39 / -8.1), -1.6 vs curve: the lpt6 code adapter (useless alone) hurts in the merge, unlike pt's code adapter.
  **C + no-fit .25: +43.3 / -1.2** (svamp +42 / -1.3, mbpp +48 / -0.4, ifeval +42 / -2.7, simple +37 / 0.0, multiple +41 / +1.1, no-fit +41
  / -2.1), +0.5 vs curve: the weak no-fit adapter repairs no-fit (-9.4 -> -2.1) WITHOUT breaking the call types and adds saving:
  the best lpt6 merge so far. drivereport: Drive project folder ~197 GB = adapters 120 (never-scored checkpoints), runs 61,
  models 13.3, data 2.7; drivetidy (new, dry run) queued to list exact clean-up items.
- ~02:30 lpt6mix-tools (one fn + several + no-fit, each 0.5): **+31.0 / -2.3** official (svamp +31 / -5.0, mbpp +31 / -2.1, ifeval +25 / -1.7,
  simple +32 / -4.0, multiple +37 / +2.9, no-fit +39 / +2.1): the three tool slices merged shorten everything ~31% but lose
  one-function and math; weaker than pt's single pooled tools adapter at a similar dose (tools@.45 28.6 / -1.1).
- The extended October rerun in PURE fp32 ran out of GPU memory on the code adapter (8,192-token answers on an L4); the math
  adapter (lpj-svamp) trained. Rerun as lpk with --bida-precision (fp32 weights + fp16 autocast, as bidax6p / Track A), new
  pertype --pt-dpo pass-through; everything else unchanged.
- drivetidy first dry run: 108 GB listed, but two flaws found in review BEFORE anything moved: scored GRPO checkpoints
  (grA/B/C-rl steps 50 / 100) were listed (adapter "grA-rl" vs results "grA-*"), and GRPO resume files were counted twice.
  Fixed (name suffixes -rl / -drpo / -teach / ... stripped; D skips files inside C items); a second dry run is queued.
- ~02:45 (user found it) new competitor **matousek-martin/qwen3-self-replay** (non-fork, created 5 Nov; "An independent submission to
  BottleCapAI's ThinkingCap Test"): LoRA on Qwen3-0.6B, correctness-gated length reward + supervised replay of its own short
  verified solutions, no teacher. Full test 2,759 x 8: Self-Replay-256 **31.23% / +0.21** (CI -0.48..+0.88; +0.5 vs our curve),
  Self-Replay-850 **41.60% / -1.17** (+0.3), checkpoint 1280 46.58% / -2.35 (-0.1). No-fit -4.17 / -6.88 pp; one seed;
  checkpoints chosen on the benchmark (stated). Added to FullEval as references (per-part numbers not copied).
- ~03:00 **lpt6mix-C7 (instructions + one fn + several, each 0.7): +51.3 / -3.6** official (svamp +55 / -4.2, mbpp +55 / -2.7, ifeval +50 /
  -3.7, simple +42 / -2.1, multiple +46 / +1.7, no-fit +45 / -11.7), -0.5 vs curve: the higher dose saves more but no-fit
  collapses without a no-fit adapter. lpt6 merge summary: C 41.3 / -1.3, C + code 39.5 / -2.8, C + weak no-fit 43.3 / -1.2 (best),
  tools only 31.0 / -2.3, C at .7 51.3 / -3.6.
- drivetidy second dry run (fixed): **99.1 GB** (A 4.8, B 2.6, C 51.7 = 1,009 never-scored checkpoints, D 7.5, E 16.9 + 6.6,
  F 2.6, G 6.3); reviewed: scored GRPO steps 50 / 100 now kept, no results folder, nothing from 5-6 Nov, no input of a
  queued run. User approved A-G: drivetidy --dt-apply queued on GPU 1 (moves to Drive trash; the user empties it).
- ~03:20 (user) GRPO v2 option after Herz1c's write-up: rl_train.py --gr-form linear (R = max(0, 1 - c L/B) for a right, formed,
  finished answer, else 0) and --gr-norms-from FILE (B = the original model's mean generated tokens per test part on TRAINING
  answers, never test questions; the official metric is a ratio of summed tokens per benchmark, so long code answers must
  not dominate the penalty). Prepared as the commented plan line gr2n next to gr2; GRPO stays off.
- ~03:35 (user) gr2n set to start from the ORIGINAL Qwen (not v1), 40 steps x 32 questions x 8 answers, scored at 10 / 20 / 30 / 40, per-type mean-length normalisation from the 4-answer training file thinkprune already restores; still OFF.
- ~03:45 (user) gr2n selects questions and length means from the original's 10 answers per question (thinkprune --tp-answers10, new), not 4; GRPO itself samples 8 fresh answers per question per step (group 8).
- ~04:00 (user) GRPO launched: gr2n (from the original Qwen, 40 steps x 32 questions x 8 answers, scored at 10/20/30/40, gated linear per-type reward) queued on GPU 2 after the October rerun's step-19 scores, preceded by a 5-step smoke test (gr2nsmoke).
- (user) just one GRPO run: the smoke test removed; gr2n alone.
- ~04:10 (user) the single GRPO run is now gr2e: gr2n restricted to questions the original solves at least half the time (10 stored answers, --tp-min-solve 0.5 with --tp-include-unsolved = rate >= 0.5); everything else unchanged.
- ~04:20 (user) thinkprune --tp-check (new: eligible questions per part vs 40 x 32 needed, no training) queued on GPU 2 as the next stage, before gr2e.
- ~04:20 6 Nov **gr2e selection check OK**: 14,001 questions solved >= 5/10 for 1,280 needed (10.9x): svamp 6,141, mbpp 1,835,
  ifeval 1,758, simple 1,216, multiple 540, no-fit 2,511 (each >= 5x its share); gr2e stays queued as is.
- **Drive clean-up applied** (drivetidy --dt-apply, user-approved A-G): 99.1 GB moved to the Drive trash; the user empties it.
- **October rerun (lpk = October rule + matching on the new data, fp32 weights + fp16 compute), adapters alone at 1.0:**
  - math @ step 19 (= October's count): **+40.9 / -2.3** (svamp +60 / -2.3, mbpp +44 / -3.3, ifeval +29 / -1.0, simple +23 / -2.3,
    multiple +25 / -1.9, no-fit +24 / -4.0), -0.9 vs curve. October's math adapter: 41.7 / -2.1 with svamp 59 / -4.8 and no-fit
    -6.0: same overall, but math itself loses half as much (-2.3 vs -4.8); code pays more (-3.3 vs -1.0).
  - math @ step 10: **+26.6 / -1.1** (svamp +48 / -0.8; no-fit -3.1, multiple -2.5), -1.1 vs curve.
  - code @ step 19: **+16.6 / -1.9** (mbpp +26 / +1.3; ifeval -4.6, svamp -3.5), -2.2 vs curve. October's: 16.5 / -1.3 (mbpp 25 / -0.2).
- ~05:20 lpk adapters alone at 1.0: **instructions @19 +27.7 / -1.2** (ifeval +34 / -2.3; no-fit -2.7, simple -2.3), -1.2 vs curve
  (October: 36.1 / -1.2, ifeval 40 / +0.6: the rerun shortens instructions LESS and costs ifeval more); **tools @19 +42.8 / -1.6**
  official (simple +38 / -4.0, multiple +44 / +1.1, no-fit +42 / 0.0; svamp -2.1, ifeval -1.3), +0.1 vs curve (October: 50.1 / -2.9 with
  math -4.0, code -4.6: less saving, much less spill-over cost); code @10 +10.9 / -1.3 (mbpp +16 / +2.1); instructions @10 +14.0 /
  -1.4. So far the rerun's adapters are gentler than October's (less saving, less damage); merges at step 19 next.
- ~06:20 lpk scores. Alone at pt-mid strength (step 19): math @.30 **+12.1 / -1.1** (Oct 11.9 / -1.0), code @.45 **+8.8 / -0.8** (Oct 10.0 / -2.5),
  instructions @.60 **+17.7 / -0.7** (Oct 23.1 / -1.2). Dose curve at 1.0: math 10 / 19 / 30 = 26.6 / -1.1, 40.9 / -2.3, **33.9 / -1.4**
  (saving FALLS after 19: the extra matched pairs past October's count pull length back; svamp 60 -> 50% while svamp acc -2.3 -> -3.5);
  code 10.9 / -1.3, 16.6 / -1.9, **16.3 / -2.5** (saturates, accuracy worsens); tools @10 **+22.0 / -1.1** (@19 42.8 / -1.6).
  Reading: step 19 (= October's count) is the sweet spot for math and tools; code past 10 only loses accuracy.
- ~07:20 lpk: tools @.45 (step 19) **+20.2 / -1.2** (Oct 28.6 / -1.1); **softer merge @19 (.20/.25/.35/.25): +32.0 / -1.4** (svamp +38 / -1.9,
  no-fit +28 / -4.2, multiple -2.1), -1.0 vs curve: below October's softer (39.2 / -0.8); instructions @30 **+36.5 / -0.6**, +0.2 vs curve
  (better than @19, 27.7 / -1.2: instructions keeps improving with the extra pairs); tools @30 **+45.3 / -2.4** (ifeval -4.8, svamp -3.3).
  GPU 1's queue ended -> GRPO gr2e moved to GPU 1 to start now (soft / mid merges still on GPU 2).
- ~08:00 lpk merges @19 (GPU 2's queue now ended): **soft +40.3 / -2.3** (-1.0 vs curve; October soft 48.4 / -1.9),
  **mid +48.5 / -2.1** (+0.5 vs curve; October pt-mid 55.0 / -2.6 on the 720). **Verdict on the October rerun: the new data + fp32
  does NOT beat October**: every merge saves 6-8 points less; mid sits a little better on the curve (+0.5 vs pt-mid's +0.3 full-test),
  but at less saving. Best single change seen: instructions at step 30 (36.5 / -0.6).
- ~08:30 **BUG in GRPO gr2e, run stopped after 10 steps**: rl_train's --gr-norms-from re-derived each answer's type with
  build_pool.part_of and set the svamp length scale B to **82 tokens** (other parts plausible: irrelevance 231, multiple 208,
  simple 180, ifeval 669, mbpp 1,818), while the stage's own caps show svamp answers up to 1,423 tokens. With the linear reward
  (max(0, 1 - 0.3 L/B)) every right math answer above ~273 tokens earned 0, i.e. math was trained to be cut, not to be right
  (training log: mean tokens 620 -> 464 in 10 steps). Fix: thinkprune --tp-norm-means computes B per part inside the stage from the
  pool's own labels over the original's 10 answers, asserts every B >= 100 tokens, and passes it as --gr-norms. Restarted as
  **gr2f** on GPU 1 (same settings otherwise: original Qwen, solved >= 5/10, 40 steps x 32 x 8, scored at 10/20/30/40);
  gr2e's step-10 adapter is discarded (its 720 score was interrupted).
- ~08:45 gr2f started on GPU 1 with the corrected scale: B = irrelevance 231, multiple 208, simple 180, ifeval 669, mbpp 1,818,
  **svamp 824** (was 82). Scores due at steps 10 / 20 / 30 / 40 (~35 min of training per 10 steps + a 720 screen each).
- ~09:30 (user) Ivan's joint plan J rejected: its core is bida's pair rules + settings; nothing built on another candidate's recipe will be run or submitted.
- ~10:00 **gr2f @ step 10: +4.0 / -0.4** official (-0.3 vs curve; svamp +7 / -0.4, multiple +6 / -1.1, ifeval +3 / -1.7, mbpp +3 / +0.8):
  barely moved after 10 steps (training answers 600 -> 450 tokens, but that is on the training questions); step 20 training done, its score next.
- ~11:00 **gr2f @ step 20: +11.4 / -0.6** official (-0.9 vs curve): math and code shorten with accuracy UP (svamp +18 / +0.6,
  mbpp +16 / +0.8), but instructions barely shorten and lose (ifeval +2 / -2.5) and tool calls lose (multiple +10 / -2.3,
  simple +7 / -1.9; no-fit +9 / 0.0). Step 30 trained (29.7 min), its score running.
- ~12:00 **gr2f @ step 30: +6.8 / -0.4** official (-0.5 vs curve): saving FELL from step 20 (11.4): ifeval now LONGER than the
  original (-7% / -1.7), svamp +12 / 0.0, mbpp +11 / +0.8, tools +9-11 / -0.4 to -0.8. Step 40 trained, its score running.
- ~13:00 **gr2f @ step 40 (final): +11.3 / +0.3** official (0.0 vs curve; multiple +12 / -2.7, ifeval +3 / +2.5, svamp +17 / -0.4).
  **GRPO verdict:** 40 steps x 32 x 8 from the raw model save 4 -> 11 -> 7 -> 11% at about -0.5 to +0.3: accuracy holds, but the
  saving is small and does not grow steadily; on the curve, not above it. Both GPUs' queues have now ended.

### 2026-11-09 — repair stage opened to any model and to own data only (code, no run)
Plan (user): full-test round 3 (ca1-soup-10) and round 4 before RL (rc1-soup-30), then a "be right" repair on the one closest
to > 50% / better than -1, with nothing from michal-bida (no MetaMath, math1, math2, instr2, code2, nofit4).
- `track.py ptrepair --rp-base models/<X>`: repair any merged model instead of pt-mid (it answers, is DPO's reference, and
  every screen merges onto it).
- `track.py ptrepair --rp-own-data` (new `own_repair_questions()`): math = GSM8K train+test at every difficulty in SVAMP wording
  (prepare_types gsm8k_std), minus questions already trained on in rounds 1-4; code / instructions / tools = pool-dp13 rows
  minus MetaMath + own top-ups nofit1, nofit2, tools1, code1; 8-gram check against every evaluation question.
- `repair_pairs.py --kinds guard` / `ptrepair --rp-kinds guard`: only the 21 Oct pair (shortest right > shortest finished
  wrong); the similar-length right>wrong kind (Lightning McQwen3's) left out.
- Checked here without a GPU: guard-only pairs on synthetic answers; own_repair_questions on stand-in files (MetaMath and easy
  pool math excluded, used and test-overlapping GSM8K dropped); a mocked ptrepair smoke run (base model passed to answering,
  DPO and screens; --kinds guard; smoke quota). Not run: tests/test_pipeline.py (needs the challenge's utils package).
Plan lines, in order:
  fullmodel --name full-ca1 --nothink-models models/ca1-soup-10            (GPU 1)
  fullmodel --name full-rc1soup --nothink-models models/rc1-soup-30         (GPU 2)
  ptrepair --name rpD-smoke --rp-base models/<B> --rp-own-data --rp-kinds guard --rp-smoke
  ptrepair --name rpD --rp-base models/<B> --rp-own-data --rp-kinds guard --rp-questions svamp=2400,mbpp=900,ifeval=600,bfcl_simple_python=300,bfcl_multiple=300,bfcl_irrelevance=300 --rp-quota svamp=640,mbpp=400,ifeval=320,bfcl_simple_python=80,bfcl_multiple=80,bfcl_irrelevance=80 --rp-lr 1e-5 --rp-steps 80 --rp-screen 10 20 40 80
- 9 Nov ~09:05: Step 1 started on both runtimes from 9c4ac5b. GPU 2 (second_colab): fullmodel rc1-soup-30 running (vLLM up 09:02).
  GPU 1 (full_eval_colab): fullmodel ca1-soup-10 FAILED at once: models/ca1-soup-10 not on Drive (a clean-up removed the merged
  model). New `fullmodel --fm-soup NAME:STEP` rebuilds models/NAME-soup-STEP from the original Qwen + adapters/NAME-s0..2/step-STEP
  at 1/3 each (first chain round: the seeds started from clean-a2's adapter, so they hold the whole lineage); dry-run checked here.
  plan_line_b now: fullmodel --name full-ca1 --nothink-models models/ca1-soup-10 --fm-soup ca1:10 (needs full_eval_colab re-run).

### 9 Nov: COPD (contrastive on-policy distillation) built and queued (user)
- rl_train.py --objective copd (Ruan et al., arXiv 2607.19046): no reward; a frozen teacher (Qwen3-4B, or 'self') re-reads each
  sampled reply under a light- and a heavy-effort instruction; token advantage = log p(light) - log p(heavy), clipped +-10;
  loss = -mean over replies of (1/T) sum A_t log pi; one update per step. Refuses a teacher with a different tokenizer. Logs the
  teacher's verdict each step (mean advantage, share of tokens pushed down, advantage at </think>). Toy-model tests pass.
- Settings researched: lr 1e-5 = the paper's 1e-6 (full fine-tuning) x 10 for LoRA ("LoRA Without Regret": optimal LoRA lr ~10x
  full FT for SFT and RL); one answer per question in the paper (no group needed), so 64 questions x 2 answers per step.
- Questions (user): Run A = solved at least once in the original's 10 answers, always-solved kept, never-solved out (the
  never-solved arm, free savings vs the risk of giving up early on hard-but-solvable questions, is a later test).
- Queued on GPU 2 after the rc1-soup-30 full test: cp1smoke (5 steps, memory check) then cp1a (80 x 64 x 2 = 10,240 answers,
  gr2f's budget; 720 at 20/40/60/80). After it: token distribution per data type + accuracy vs the original (user).
- ~10:20 **rc1-soup-30 FULL TEST (2,759 x 8): +57.4 / -3.5** official (BFCL 400/200/240), **+0.95 vs curve** (pt-mid's full test:
  53.6 / -3.3, +0.3). Per type: svamp +67 / -6.6, mbpp +63 / -3.5, ifeval +51 / -2.5, irrelevance +51 / -4.1, multiple +44 / -0.8,
  simple +41 / -0.1. Earlier x8 runs of the same model: usual-set 58.0 / -3.1, fresh set 55.6 / -2.9 (Oct). Math carries the loss.
  Its 720 screen running next, then COPD on GPU 2.
- 9 Nov ~12:10 COPD sanity check (copd_demo.py): one SVAMP-style trick question, a hand-written answer in six parts (setup, a
  useful re-read, the calculation, a needless double-check, the stop, the answer). Qwen3-0.6B as its own coach (COPSD, on the
  Mac): mean push per part +0.01 / -0.01 / -0.06 / +0.01 / -0.01 / 0.00, biggest single pushes on spaces and digits, 0.00 at
  </think>: the 0.6B barely reacts to the effort instructions, so the self-coach version would teach nothing. The 4B check runs on
  GPU 2 first (copddemo) before the smoke test and Run A (the Mac's 8.6 GB RAM cannot hold the 4B).
- (user) Run B, only if Run A loses accuracy: COPD pushes applied to RIGHT answers only (wrong answers never teach "be brief").
- 9 Nov ~13:00 **ca1-soup-10 FULL TEST: +49.1 / -2.4** official (+0.4 vs curve; its 720 screen said 52.4 / -0.9: flattered).
  svamp +58 / -4.3, mbpp +53 / -2.2, ifeval +45 / -2.4, irrelevance +41 / -3.2, multiple +38 / +0.6, simple +36 / +0.4.
  Same pattern as rc1-soup-30, milder: math loss from always-solved questions (100 -> 94.6); no-fit 7/8 questions -23 pts.
  160 regressed questions (rc1-soup-30: 212). Breakdown + struggle in runs/mvtf/bd-ca1-soup-10, st-ca1-soup-10.
- 9 Nov COPD smoke test (cp1smoke2): step 1 ran (coach verdict on 128 answers: mean push -0.012, 38% of tokens pushed down,
  </think> 0.00), step 2 out of memory moving the teacher back to the GPU (~7 GB still held after step 1); Run A (cp1a) the same
  at step 2. copddemo, 20 hand-written questions: 0.6B as coach ~0 (direction 10/20); **4B as coach reacts (needless check
  pushed down in 19/20) but pushes the USEFUL check down twice as hard (-0.20 vs -0.10; needless below useful only 7/20), stop
  +0.01**: on this evidence COPD with these instructions would remove the re-read step. User: run Run A anyway for a few steps,
  it may show something useful. Memory fix: no KV cache in batch_logps (use_cache=False), explicit frees between the read and
  training phases, GPU memory logged per step (gpu_gb). cp1a resumes after its step 1.
- 9 Nov **ca1-soup-10 why-data** (st-ca1-soup-10/summary.json, read from Drive): wrong answers out of 8 per question, original ->
  ca1. Math: cut off 139 -> 0, stopped early 27 -> 252, plain wrong 644 -> 922; on its 62 regressed questions 30% stopped early,
  68% wrong after normal-length thinking, thinking 0.43 of the original's right answers, 271 of 307 wrong numbers far off (the
  question misread, as rc1). Code: cut off 236 -> 34, stopped early 23 -> 157, wrong 1,179 -> 1,311. IFEval: stopped early
  24 -> 247, cut off 156 -> 94, wrong 1,648 -> 1,575. No-fit: stopped early 3 -> 29, wrong 316 -> 351. Calls: about unchanged.
  Same mechanism as rc1-soup-30, milder: fewer early stops (it shortens less), the misread-question errors remain.
- 9 Nov COPD Run A (cp1a, short version) stopped by the user after step 4 (resumes from there): memory fix holds (gpu_gb 1.33
  after every step), but ~15-20 min per step (the 8 GB teacher moved CPU<->GPU twice a step). Steps 2-3: practice accuracy
  0.867 / 0.828, tokens 566 / 636, coach 36-38% of tokens pushed down, mean -0.018, </think> 0.00.
- 9 Nov ~14:45 **COPD Run A (cp1a) @ step 10: +14.6 / -0.2** official (720 x 4; -0.5 vs curve): svamp +23 / +0.2, mbpp +15 / +0.8,
  simple +11 / -1.0, irrelevance +10 / 0.0, multiple +8 / -1.1, ifeval +8 / -1.3. Against GRPO gr2f at the same step 10:
  4.0 / -0.4 (and 11.4 / -0.6 at step 20): COPD shortens ~3.5x faster per step with no math loss, despite the demo's warning.
  Run A continues to step 20.
- 9 Nov ~15:30 **COPD Run A (cp1a) @ step 20 (end of the short run): +20.8 / -0.4** official (720 x 4; -0.7 vs curve): svamp
  +35 / +0.2, mbpp +18 / +0.6, irrelevance +19 / -0.6, multiple +20 / -1.1, simple +14 / -0.4, ifeval +11 / -1.9. Step 10 -> 20:
  saving 14.6 -> 20.8, accuracy -0.2 -> -0.4: still growing, math still free. GPU 2's queue ended.
- 9 Nov **rpE (repair of rc1-soup-30) @ step 10: +57.4 / -2.9** (compare's average; official 57.2 / -3.0). rc1-soup-30's own
  720 screen (s4-rc1-soup-30/compare.json, from Drive): +58.7 / -3.8 (svamp -5.0, ifeval -4.0, simple -4.0, irrelevance -4.6,
  mbpp -3.5, multiple +0.6). After 10 repair steps: svamp -2.7 (+2.3), ifeval -2.5 (+1.5), simple -2.7 (+1.3), irrelevance
  -3.7 (+0.9), multiple +1.5, mbpp -4.6 (-1.1); +0.9 accuracy for -1.3 saving. Repair pairs: chosen re-checks the question
  51 / 77 / 94% (math / code / no-fit) vs rejected 41 / 57 / 82%; chosen and rejected about equal length.
- 9 Nov **rpE (repair of rc1-soup-30) @ 20: +54.8 / -2.6** (irrelevance -1.9, simple -1.0, mbpp -2.5, ifeval -2.9, svamp -4.2,
  multiple +1.3); **@ 40: +50.5 / -1.7** official (irrelevance -0.2, multiple +1.5, simple -1.9, ifeval -1.7, mbpp -2.7, svamp -1.9;
  saved: svamp 60, mbpp 54, irrelevance 50, multiple 43, ifeval 42, simple 40). Path from rc1's 720 (58.7 / -3.8): each 10 steps
  buys ~0.5 pt of accuracy for ~2 pts of saving after step 20; vs curve rc1 +0.6, @10 +0.6?, see below. Step 80 scoring.
- 9 Nov **rpE @ 80: +50.4 / -0.2** official (720 x 4; compare's average 50.6 / -0.1; **+2.8 vs curve**): irrelevance +51 / +3.1,
  multiple +43 / +1.3, simple +39 / -2.1, ifeval +50 / +0.4, mbpp +52 / +1.0, svamp +56 / -2.5. First model to meet >50% / better
  than -1 on the 720 screen. CAVEAT: the 720 flattered ca1-soup-10 by 1.5 pts (720 -0.9 vs full test -2.4); rc1-soup-30's 720
  (-3.8) was close to its full test (-3.5). A full test is needed before any claim. Screens at 45 / 50 running.
- 9 Nov ~19:45 (after Colab's provisioning outage; both runtimes restarted) **rpE @ 45: +52.9 / -1.7** official (irrelevance +0.4,
  multiple +0.6, simple -2.1, ifeval -3.1, mbpp -1.7, svamp -1.3). Path @40 50.5 / -1.7, @45 52.9 / -1.7, @80 50.4 / -0.2: between
  40 and 45 the screens differ by 2.4 pts of saving at equal accuracy (screen noise ~ +-2). COPD steps 21-25 trained (465-542 tokens).
- 9 Nov **rpE @ 50: +49.5 / -1.6** (irrelevance +2.3, multiple +1.5, simple -1.7, ifeval -1.9, mbpp -3.3, svamp -1.3). Full tests
  of rpE-80 then rpE-40 started on GPU 1 (both models built by fullmodel --fm-merge).
- 9 Nov **COPD cp1a @ 25: +25.6 / -1.5** (-1.6 vs curve): svamp +40 / -0.2, mbpp +23 / -1.7, ifeval +16 / -4.0, irrelevance +20 /
  -1.7, multiple +23 / +0.6, simple +19 / 0.0. Saving up from 20.8 (@20) but accuracy down from -0.4: the lr jumped back to
  ~5.5e-6 at step 21 (cosine stretched to 40 steps); IFEval carries the loss (the coach shortens the RESPONSE too).
- 9 Nov **diagnostic (user): thinking vs response tokens and accuracy per type, 720 x 4, from compare.json** (base -> model):
  | model | ifeval acc | ifeval think | ifeval response | svamp acc / think | mbpp acc / think |
  | rc1-soup-30 | 59.0 -> 55.0 | 420 -> 129 | 247 -> 217 | 90.0 -> 85.0 / 685 -> 205 | 50.8 -> 47.3 / 2077 -> 685 |
  | rpE @ 50 | 59.0 -> 57.1 | 420 -> 153 | 247 -> 245 | 90.0 -> 88.8 / 685 -> 274 | 50.8 -> 47.5 / 2077 -> 968 |
  | rpE @ 80 | 59.0 -> 59.4 | 420 -> 146 | 247 -> 190 | 90.0 -> 87.5 / 685 -> 297 | 50.8 -> 51.9 / 2077 -> 981 |
  | COPD @ 25 | 59.0 -> 55.0 | 420 -> 317 | 247 -> 245 | 90.0 -> 89.8 / 685 -> 406 | 50.8 -> 49.2 / 2077 -> 1586 |
  (1) The repair's lr was CONSTANT (cosine from 1e-5 to lr-min = 1e-5): 50 -> 80 the thinking lengths barely move (ifeval 153 ->
  146, mbpp 968 -> 981, svamp 274 -> 297) while accuracy rises (ifeval +2.3, mbpp +4.4): its pairs are equal-length (medians
  chosen / rejected 271/278, 440/440, 185/183), so they teach correctness inside the same length; the saving fell only over
  steps 0-40. (2) IFEval's loss is NOT a shortened response: COPD and rpE @50 keep the response (247 -> 245); rpE @80 even shortens
  it (-> 190) yet beats the original. The losses go with THINKING cuts (rc1 -69%, COPD -25%). IFEval is the noisiest part
  (original 59%, 720 half-width +-3.6 pts vs svamp +-1.9). Which instructions fail needs records + questions (next).
- 9 Nov **COPD cp1a @ 30: +29.2 / -1.3** (-1.2 vs curve): svamp +45 / -1.0, mbpp +33 / -0.8, ifeval +12 / -3.5, irrelevance +26 /
  -0.6, multiple +24 / 0.0, simple +22 / +0.6. Saving still rising (20.8 -> 25.6 -> 29.2), accuracy steady near -1.4, IFEval the
  loss. ifeval_rules.py + ifevalrules stage queued on GPU 1 after the full tests (full-rc1, full-ca1, full-rpE-80/40, s4-cp1a-20/25).
- 9 Nov **COPD cp1a @ 35: +29.8 / -1.7** (svamp +48 / -1.9, mbpp +27 / -1.0, ifeval +15 / -2.5, irrelevance +26 / -0.8, multiple
  +25 / -0.6, simple +23 / -2.1): the saving has flattened (29.2 -> 29.8) while accuracy drifts down; math now loses too (-1.9).
- 9 Nov **COPD diagnostic (user: what is happening at step 35?)**. (a) The lr: cosine stretched to 40 steps, so 6.1e-6 at step 21,
  4.5e-6 at 25, 2.7e-6 at 30, 1.4e-6 at 35, ~1e-6 at 40: the saving flattens as the lr reaches its floor. (b) The coach's signal
  shrinks as the student becomes "brief": mean push -0.015 (steps 21-25) -> -0.008..-0.012 (31-38), tokens pushed down 34-37% ->
  31-33%; entropy 0.29 -> 0.25-0.28; practice answers ~500 -> ~400 tokens, practice accuracy flat 0.83-0.91. (c) 720 @35 vs @25
  (compare.json): svamp thinking 406 -> 353, acc 89.8 -> 88.1; mbpp 1,586 -> 1,501, 49.2 -> 49.8, still 4.4% cut at 8,192
  (original 8.5%); ifeval thinking 317 -> 290, RESPONSE 245 -> 278 (longer), cut 3.1 -> 3.8%, acc 55.0 -> 56.5. The accuracy drift
  is within the per-type noise except a slow math slide (-0.2 / -1.0 / -1.9 at 25 / 30 / 35); COPD has no correctness term to
  stop it. Data-backed fix: the rpE repair (equal-length right>wrong pairs, constant lr) took rc1 from -3.8 to -0.2 at the same
  length, so "COPD to compress, then the same repair" on a COPD checkpoint; and a constant lr for any further COPD.
- 9 Nov (user) **COPD Run C (cp1c) queued on GPU 2 after Run A**: rl_train --copd-outcome w adds w x (right - the question's mean
  right) to every token of each reply (GRPO-style group comparison, no std) on top of the coach's per-token push: correctness
  trained in the same step; w = 1.0, 32 questions x 4 answers, CONSTANT lr 1e-5 after 4 warm-up steps, 40 steps, 720 at 10/20/30/40.
  No per-type brake (user). Toy-model test passes.
- 9 Nov (user: option 3, and keep data on the mix) Run C now uses --tp-mixed-share 0.5: per part, the questions alternate
  "sometimes wrong" (the original solves 1-9 of its 10 stored answers) and "always right" (10/10); the counts available per
  part are printed at the start, each pool row carries solve10, and every step logs per kind: questions, how many are MIXED NOW
  (right and wrong among the 4 fresh answers: where the correctness push acts) and the share right.
- 9 Nov ~21:30 **rpE @ 80 FULL TEST (2,759 x 8): +47.0 / -0.9** official (**+1.45 vs curve: our best full-test trade-off**):
  irrelevance +50 / +3.4, multiple +39 / +0.1, simple +37 / +0.1, ifeval +45 / -2.6, mbpp +46 / -1.0, svamp +54 / -1.2.
  Accuracy meets the goal (better than -1), the saving does not (47 < 50). Its 720 screen said 50.4 / -0.2: flattered by 3.4 pts
  of saving and 0.7 pt of accuracy. Reference (not ours): michal-bida 47.0 / -0.78, Herz1c 43.4 / -0.97. rpE @ 40 full test
  running. Run C rescheduled to Run A's checkpoints (10/20/25/30/35/40, 5-step blocks).
- 9 Nov **COPD Run A @ 40 (final): +34.1 / -2.4** (svamp +51 / -2.7, mbpp +31 / -4.4, ifeval +22 / -1.9, irrelevance +30 / -1.5,
  multiple +27 / -0.2, simple +25 / 0.0): saving 29.8 -> 34.1 at the lr floor, accuracy down to -2.4.
- 9 Nov Run C (cp1c) stopped at step 3: its mix was not 50/50 (steps 1-3: 8/16, 2/16, 3/16 sometimes-wrong) because rl_train
  re-orders the pool by source (balanced_order). Fix: with --tp-mixed-share, thinkprune keeps exactly steps x questions
  questions in the mixed order (319 of 640 sometimes-wrong on the real pool sizes). Relaunched as cp1c2. Early mix data
  (cp1c steps 1-3): sometimes-wrong questions mixed-now 6/8, 2/2, 3/3; always-right ones 2/8, 5/14, 2/13 (right 0.91-0.98).
  Available per part (sometimes / always): svamp 2,597 / 4,001, mbpp 1,274 / 898, ifeval 1,015 / 1,098, simple 316 / 955,
  multiple 239 / 360, irrelevance 401 / 2,253.
- 9-10 Nov **Run C (cp1c2) started** (both runtimes restarted after the outage; plan start now light): 640 questions, 319
  sometimes-wrong. Steps 1-6 (practice): accuracy 0.84 / 0.76 / 0.89 / 0.87 / 0.57 / 0.66 swinging with the draw's share of
  sometimes-wrong questions (7 / 12 / 7 / 8 / 13 / 10 of 16); sometimes-wrong mixed-now 4/7, 8/12, 2/7, 4/8, 12/13, 7/10 (right
  47-79%); always-right mixed-now 0-2 (right 96-100%). 24 min per 5-step block. GPU 1: rpE @40 full test running.
- 10 Nov **rpE @ 40 FULL TEST: +50.6 / -1.9** (720 said 50.5 / -1.7). Step 80 full: 47.0 / -0.9. Step 60 full test next.
- 10 Nov **COPD Run C (cp1c2) @ 10: +12.7 / -1.1** (720; Run A @10: 14.6 / -0.2). Training continues (scores at 20/25/30/35/40).
- 10 Nov **rpE @ 60 FULL TEST: +47.8 / -1.2** (+1.3 vs curve; irrelevance +49 / +1.4, multiple +39 / -0.2, simple +38 / +0.1,
  ifeval +46 / -3.0, mbpp +48 / -0.7, svamp +55 / -1.5); its 720: 48.6 / -0.7. **rpE @ 40 FULL: +50.6 / -1.9** (+1.1; irrelevance
  -0.4, multiple -0.6, simple +0.4, ifeval -2.8, mbpp -1.7, svamp -3.0). Repair path on the full test: 40 -> 60 -> 80 = 50.6 / -1.9,
  47.8 / -1.2, 47.0 / -0.9: each 20 steps buys ~0.5 pt of accuracy for ~1-3 pts of saving.
- 10 Nov **COPD Run C (cp1c2, + correctness push, constant lr), 720**: @10 12.6 / -1.1, @20 28.0 / -2.0, @25 35.8 / -3.6, @30
  43.2 / -3.2, @35 49.7 / -3.0, @40 57.1 / -4.7 (svamp +74 / -5.2, mbpp +60 / -7.5, irrelevance +46 / -6.5). With a constant lr
  it shortens much further than Run A (57 vs 34%) but lands ON our DPO curve (-0.2 to -0.5 vs curve): the correctness push did
  not keep accuracy. A shortener comparable to rc1-soup-30 (57.4 / -3.5 full), i.e. a candidate base for the same repair.
- 10 Nov **IFEval rule-by-rule diagnostic (full tests, 541 questions x 8; ir-full/ifeval_rules.json)**: every shortened model
  (rc1, ca1, rpE-40/60/80) loses on COUNTING rules: length_constraints:number_words -11 to -13 pts, keywords:frequency -8 to -10,
  number_sentences -6 to -7, number_highlighted_sections -6 to -9, number_placeholders -5 to -11; gains on quotation, language,
  capitals (+3 to +6). On length rules the wrong responses are mostly TOO SHORT: original 265 below the asked number, rc1 387,
  ca1 364, rpE-80 379; responses on length-rule questions median 209 -> 145-161 tokens. Wrong answers with thinking under half
  the original's right-answer median: original 62 / 916, shortened models 626-815 / ~1,070. So IFEval's loss = counting /
  length requirements broken by shorter thinking AND shorter responses on exactly those questions.
- 10 Nov (user) **repair improvements for rpF (repair of Run C @40)**: repair_pairs.py --count-rules: IFEval questions with a
  counting rule (word / sentence / paragraph counts, keyword frequency, number of sections / bullets / placeholders) take as chosen
  a right answer whose RESPONSE is at least the rejected's and whose THINKING is at most 1.2x the rejected's (else no pair): the
  pair teaches "write the asked length", not "think more"; the summary prints counting-rule questions / pairs / skipped.
  IFEval questions 600 -> 1,500 (rpE found only 154 IFEval pairs for a quota of 320). Fake-answer test: the 3 cases behave.
  Queued: plan B = breakdown of rpE-80 full, then rpF; plan A = rpE 80 -> 120 (720 at 100 / 120).
- 10 Nov (user) **three continuations of repair step 80 (40 steps each = 80 -> 120), 8 answers per question**, all starting from
  rpE step 80's adapter (also DPO's reference), 720 at their steps 20 / 40: **rpG** = rc1-soup-30 answers the repair's questions
  x8, rpE's pair rules; **rpH** = rpG + IFEval counting-rule pairs (--count-rules) and 1,500 IFEval questions; **rpI** = a new
  round: the repaired model (models/rpE-80) answers a new question draw (seed 1) x8, pairs from its current mistakes + the IFEval
  fix. New ptrepair options: --rp-samples, --rp-seed, --rp-answers-model, --rp-init-from. GPU 1: breakdown rpE-80, rpI, rpF;
  GPU 2: rpG, rpH (the plain resume 80 -> 120 replaced by rpG).
- 10 Nov (user) queued: **copdanalysis** (GPU 1, after the rpE-80 breakdown): copd_analysis.py = right vs wrong lengths on Runs A / C
  training questions (the original's 10 answers: is "right up / wrong down" also "short up / long down"?), the 720 screens of A
  and C per part (right / wrong lengths, cut at the limit, wrong answers that stopped early), the training logs side by side.
  **cp1d** (GPU 2, after rpH) = the ONE-CHANGE version of Run C: Run A's exact settings (64 x 2, lr 1e-5 -> 1e-6, solved >= 1/10)
  + only the correctness push (--copd-outcome 1.0); 720 at 10/20/25/30/35/40.

## 10 Nov — COPD Runs A and C, read from the data (copdanalysis stage, GPU 1)

1. **Wrong answers are longer than right ones** on the training questions (the original's 10 answers, questions with both):
   svamp right 880 / wrong 1773 tokens, wrong longer on 72% of questions; mbpp 1449/2271 (62%); ifeval 643/1057 (57%);
   irrelevance 330/448 (75%). So "push right up, wrong down" inside a group is also "push short up, long down".
2. **720 screens.** Run C cuts much harder: at step 40, right-answer tokens are svamp 583->159 (A: 312), mbpp 1209->420 (A: 843),
   ifeval 519->256 (A: 372). Its extra errors are answers that **stopped early** (thinking < half the original's median right-answer
   thinking): svamp 54 of 73 wrong (A: 19/61), mbpp 68/272 (A: 26/257), ifeval 54/212 (A: 14/210), simple 25/96 (A: 9/83).
   Cut-offs are not the issue (mbpp 8.5% -> 0.2%). At equal savings C is worse: A-40 34.1/-2.4 vs C-25 35.8/-3.6.
3. **Training logs.** Run A's coach push stays at about -0.01 to -0.02 for all 40 steps. Run C's push fades from about -0.015 to 0 by
   step 24, then turns positive (+0.015 at step 40), yet tokens keep falling (700 -> 250-300). From step 24 the correctness term
   alone drives the shortening; the coach no longer does. Practice accuracy is noisy (0.57-0.92) because half the questions are
   sometimes-wrong.
**Reading:** the correctness push carries a hidden length reward, because wrong answers are longer, and it removes thinking that was
needed. cp1d (Run A settings + the correctness push only) tests whether the 16x8 / constant-lr / 50%-mixed settings or the push
itself causes it.

## 10 Nov — rpG (rpE-80 continued 80->120, normal method, 8 answers/question, 1484 pairs; IFEval 208/320 pairs)
- rpG-20 (= repair step 100), 720 screen: **40.6% saved, +1.0 pts** (BFCL-weighted 40.4 / +0.9). Per part: irrelevance +51%/+6.9,
  multiple +39%/-1.1, simple +37%/-2.1, ifeval +38%/+0.2, mbpp +40%/+1.7, svamp +42%/+1.0.
- vs rpE-80 on the same 720 (50.4 / -0.2): about 10 pts less saved for +1.2 pts accuracy. Distance above the curve is about the same
  (+2.35 vs +2.77 on the 720), so the model slid along the curve toward longer answers rather than improving.

## 10 Nov — online DPO (odpo stage, queued on GPU 1 after rpF)
od1: 4 rounds from rpE step 80 (base rc1-soup-30). Each round the CURRENT model answers a new slice of own questions x8
(1,290 per round, no question twice), repair_pairs.py --kinds guard,shorten builds 160 pairs: half guard (shortest right >
shortest wrong, + re-read / IFEval counting rules) and half shorten (shortest right > longest right, only on questions right
>= 6/8 with the longest >= 1.5x the shortest; never on IFEval counting-rule questions); DPO 10 steps continuing the adapter,
the round's start as the reference; 720 x4 after rounds 2 and 4. Tested on fake answers (pairs) and mocked rounds (loop).

## 10 Nov — correction: the October rerun (lpk) verdict was too strong
Re-read of the 6 Nov data (all 720 x4, same yardstick): adapters alone are a tie with October (tools +0.07 vs +0.01, math -0.91
vs -0.59, instructions @19 -1.15 vs -0.41 but @30 +0.24). The merges lose (mid +0.5 vs October pt-mid +1.3; soft -1.0 vs +0.7)
because they reused October's strengths (.30/.45/.60/.45), which were picked ON this 720 for October's own adapters: the rerun was
never tuned, and its gentler adapters at the same strengths save 6-8 points less. Also: two changes at once (new data AND fp32,
so neither effect is known), one seed, one run; gaps of ~1 pt are within 720 x4 noise. The earlier "+0.5 vs +0.3" compared a
720 score with a full-test score. Conclusion: the rerun does not show that better settings fail; it was not a fair test.
od2 (queued after od1, user): the same online loop from the ORIGINAL Qwen (fresh LoRA, reference = the original in round 1),
6 rounds x 10 steps, fp32 weights + fp16 compute (dpo.py --bida-precision, as every repair), 720 x4 after rounds 2, 4, 6.
- rpG-40 (= repair step 120), 720 x4, official weighting: **32.7% saved, +0.5** (svamp 31% / +0.4, mbpp 23% / +2.5, ifeval 37% / -1.9,
  simple 34% / -2.9, multiple 36% / -0.4, irrelevance 50% / +8.5). Step 100 official: 40.4 / +0.9. vs curve: step 80 +2.8,
  step 100 +2.2, step 120 +0.9. **More steps of the same repair only make the model longer (code 46% -> 23% saved) and fall back
  toward the curve: rpE step 80 stays the best stopping point.**

## 10 Nov — public competitor scan (read-only)
No new or better 0.6B score anywhere: BottleCapAI/ThinkingCap-Test still has the same 5 forks. michal-bida (last push 20 Oct),
basta (14 Oct), horychtom, miroslavmyrha, Herz1c (15 Oct), mrtineu (3 Nov), khudgins: unchanged. marek-krcal active (SimPO /
DPO branches 6-9 Nov), no published score. matousek-martin: ablations + release v0.1.2, headline unchanged (256 = 31.23 / +0.21,
850 = 41.60 / -1.17, 1280 = 46.58 / -2.35); per type @850: svamp 53.2 / -1.29, mbpp 42.3 / -1.26, ifeval 38.1 / +0.18, simple
28.4 / -2.00, multiple 30.4 / +2.44, irrelevance 33.4 / -6.87. Herz1c/DoesLearningToStopScale: same recipe on Qwen3-1.7B
(47.4 / -0.14), not a 0.6B entry. RikiyaT is not a ThinkingCap entry. **Upstream commit 965a54c (6 Nov): a submission must now
include the coding agent's traces and the conversation with the harness, as a text document.**

## 10 Nov — rpI (new round: rpE-80 answers 5,700 new questions x8 itself; 1,579 guard pairs, IFEval 320/320 incl. 152 counting-rule; no-fit 59/80)
720 x4, official weighting: **step 20 (=100): 42.9% / -0.2; step 40 (=120): 43.0% / +0.9** (svamp 47 / -0.4, mbpp 39 / +3.5,
ifeval 42 / -0.4, simple 39 / -2.5, multiple 42 / +0.2, no-fit 52 / +6.7). vs curve: +1.5 then **+2.6** (rpE-80 +2.8, rpG-100 +2.2,
rpG-120 +0.9). Unlike rpG (same steps, old model's answers), rpI kept its length (43% at both steps) while gaining accuracy:
answers from the CURRENT model stop the drift toward longer answers. IFEval -0.4 at 42% saved (rpG-120: -1.9 at 37%).

## 10 Nov — rpH (= rpG + IFEval counting rule, 1,500 IFEval questions; 1,598 pairs, 175 counting-rule)
- rpH-20 (= step 100), 720 x4 official: **42.0% / -0.6** (svamp 44 / -0.2, mbpp 40 / +0.2, ifeval 43 / -2.5, simple 37 / -1.2,
  multiple 40 / -0.6, no-fit 50 / +3.3); vs curve +1.0 (rpG-100 +2.2, rpI-100 +1.5). IFEval did NOT improve at step 100 (rpG-100:
  38% / +0.2), but one part of the 720 is 120 questions x4 (CI about +/-4.4 pts): per-part gaps of 2-3 points are noise.
- rpH-40 (= step 120): **37.0% / -0.3** (svamp 34 / -0.6, mbpp 37 / +0.6, ifeval 36 / -2.1, simple 35 / -1.7, multiple 39 / -1.1,
  no-fit 49 / +6.5); vs curve +0.6. **The three continuations at step 120 (vs curve): rpG +0.9, rpH +0.6, rpI +2.6.** Only rpI
  (answers from the CURRENT model on new questions) keeps its length; both runs on the old model's answers drift longer.
  The IFEval counting rule alone did not fix IFEval on the 720.
- rpF pairs (Run C step 40 answers x4, all 2,998 IFEval questions): 1,446 pairs; IFEval 320/320 (258 counting-rule); mbpp 287/400
  (short 113: Run C is right on code too rarely for guard pairs at 4 answers), tools 60-71/80. Training started.
- rpF-10 (Run C step 40 + 10 repair steps), 720 x4: **54.9% / -4.0** (svamp 73 / -4.8, mbpp 57 / -4.8, ifeval 36 / -3.3, simple 43 / -2.9,
  multiple 45 / -0.2, no-fit 44 / -5.6); vs curve -0.1 (Run C-40: 57.1 / -4.7, -0.3).
- rpF-20: **54.8% / -3.8** (ifeval 42 / -0.8, svamp 72 / -6.2, mbpp 55 / -4.4, simple -4.0, multiple -1.5, no-fit -5.4), vs curve +0.1.
- rpF-40: **52.4% / -4.5** (svamp 71 / -5.0, mbpp 52 / -6.2, ifeval 32 / -4.0, simple -3.1, multiple +1.1, no-fit -5.6), vs curve -1.1.
  The repair does not lift Run C: it stays on or below the curve (rpE lifted rc1-soup-30 from +0.95 to +1.45 full test). Run C's
  loss (stopped-early math and code) is not fixed by right-vs-wrong pairs at equal length.
- rpF-80: **48.1% / -2.6** (svamp 63 / -3.5, mbpp 50 / -4.6, ifeval 31 / -0.8, simple -2.7, multiple +2.3, no-fit -2.5), vs curve -0.1:
  80 steps moved Run C ALONG the curve (57 -> 48% saved, -4.7 -> -2.6), not above it.
- od1 started on GPU 1 (round 1: rpE-80 answers 1,290 new questions x8).

## 10 Nov — rpI step 120, FULL TEST (2,759 x 8): **41.9% saved / +0.1** (official weighting)
svamp 47 / +0.0, mbpp 38 / +0.5, ifeval 41 / -1.5, simple 37 / +0.2, multiple 38 / -0.9, no-fit 50 / +5.8. vs curve **+1.64**: the
highest of any full test so far (rpE-80 +1.45, rc1-soup-30 +0.95; michal-bida 47.0 / -0.78, about +1.6). No accuracy loss at all,
but 42% saved, below the 50% goal. Its 720 screen said 43.0 / +0.9: the screen read 1.1 pts high on saving and 0.8 on accuracy.

## Items decided 10 Nov that were committed but not logged above (added for the record)
- **cp1e** (queued on GPU 2 after cp1d): cp1d + `rl_train.py --copd-early 1.0` (new, "fix 2", user): every token of a WRONG reply whose
  thinking is under 0.5 x the original model's median thinking on its right answers to that question (row field base_think, from the
  original's 10 stored answers; track.py thinkprune prints how many questions carry it) gets -1.0 more. Logged per step as
  copd.early_penalised = [punished, wrong]. Tested on mocked replies (only the too-short wrong reply is punished).
- **od2 changed to 10 rounds** (user: "add more rounds of 160"): 10 rounds x 10 steps, IFEval 290 questions per round (2,900 of the
  2,998 usable, no question reused), 720 x4 after rounds 2, 4, 6, 8, 10.
- **repair_pairs.py --kinds guard,shorten backup supply:** if "shorten" pairs (longest right >= 1.5x shortest) run short, questions
  with a 1.25-1.5x spread fill in; the summary prints both candidate counts per part.
- Full test of rpI-120 queued on GPU 2 after rpH (user); done above.
- cp1d started (GPU 2): its log printed **0/15407 questions carry base_think**: the original's 10-answer file stores no
  reasoning_tokens, so cp1e's --copd-early would have done nothing. Fixed before cp1e starts: base_think falls back to the
  answer's thinking ids, else its whole length (output_tokens; slightly over-estimates thinking, the 0.5x threshold stays a proxy).
  cp1d does not use it. od1 round 1: 160 pairs (guard 80 / shorten 80, no shortfall: svamp 299 shorten candidates >= 1.5x),
  10 steps, pairs ordered right 75 -> 82%; round 2 answering.

## 10 Nov — consult5: independent review by the Ivan worker (gpt-6-astra), cap $15 then $30 (user)
27 analysis turns (26 code runs) on CURRENT_STATE.md, the dossier, the log, all_runs.json, plans and code; stopped at $12.86 (no
notes kept, so the harness's fallback would have produced an empty report); one final call ($0.76) on the conversation with tool
outputs trimmed wrote the report: consult5_review.md. Total about $13.6-14.6 (one in-flight turn may have been billed).
Main points: objective not met; the biggest problem is reading moves on a repeatedly-used screen as frontier progress; the curve
is descriptive (at 50% it predicts -2.9, so +1 above it still misses); rpI-120 vs rpE-80 is only +0.19 vs curve; fresh answers
ATTENUATE the drift (rpI +14.9% more output vs rpG +35.7%) but do not stop it, and are confounded with pair counts; screen/full
gaps are model-dependent (savings +1.2 mean, accuracy +0.4 mean, ranges -1.1..+1.5); 6/8 is not a safety certificate; IFEval
deficit alone costs 0.65 pt of rpE-80's aggregate. Code issues found: (1) thinking_len() treats an empty thinking as the whole
answer, FIXED; (2) dpo.py reference-cache fingerprint uses prefixes + lengths only (open); (3) a round with fewer than 160 pairs
cannot reach step 10 in one epoch, FIXED in odpo (2 epochs allowed); (4) log the identities of rollout policy / init / reference
separately (open); (5) od2 = 90,400 completions for 1,600 pairs: measure pair yield. Recommendation: ONE controlled experiment
from the original Qwen: fresh-rollout vs frozen-rollout arms, identical mixed pairs (256 per round, half guard half shorten at
>= 6/8 and 1.25x), 3 rounds x 32 updates, 2 seeds, a NEW 512-question dev set for selection (never the 720), predetermined final
checkpoint, paired bootstrap on the full test. Its forecast: 35-50% / -2..0. Advice: do not spend GPU days on more COPD variants.

## 10-11 Nov night — runtimes reconnected by the user; queues resumed from Drive
- **od1 round 2** (rpE-80 + 2 online rounds = 20 steps; 160 pairs per round, guard 80 / shorten 80, no shortfall), 720 x4 official:
  **51.5% / -1.2** (svamp 58 / -2.7, mbpp 54 / -2.5, ifeval 47 / +0.4, simple 41 / -3.1, multiple 44 / +0.8, no-fit 54 / +4.2);
  vs curve +2.0 (rpE-80 on the 720: 50.4 / -0.2, +2.8). Rounds 3 done, round 4 answering.
- **cp1d** (Run A settings + the right/wrong reward only), 720 x4: step 10 14.2 / -1.4; 20 30.4 / -2.1; 25 34.0 / -1.9.
  vs curve: 20 -1.9, 25 -1.3 (Run A: 20 -0.7, 25 -1.6; Run C: 25 -2.8). The reward alone cuts faster than Run A, but cp1d does
  not collapse like Run C: Run C's other settings (16x8, constant lr, 50% sometimes-wrong questions) explain part of its fall.
- consult6 (Gemini 4 Argon via OpenRouter, cap $25, $3 report reserve): relaunched after an empty first turn (Gemini emitted a
  native function call, MALFORMED_FUNCTION_CALL; fixed by a system note + provider errors retried); 27 turns, $0.34 so far.

## 11 Nov — od1 done, cp1d step 30, plan changes
- **od1 round 4** (rpE-80 + 4 online rounds, 40 steps), 720 x4 official: **53.3% / -1.8** (svamp 63 / -4.4, mbpp 55 / +0.8, ifeval
  44 / -3.7, simple 43 / -1.2, multiple 47 / -1.9, no-fit 58 / +4.4); vs curve +1.75 (round 2: +2.0; rpE-80 on the 720: +2.8).
  The online loop moved rpE-80 ALONG the curve (more saving, less accuracy), not above it; differences within the 720's noise.
- **cp1d step 30:** 39.5% / -2.0 (vs curve -0.8). Steps 35 / 40 next. cp1e dropped (consult5 + consult6: no more COPD variants).
- consult6 (Gemini 4 Argon) review + independent fact-check (reviews/consult6_argon/REPORT.md, FACT_CHECK.md): transcription good,
  ~10 interpretation errors. User decision: Path 1 only (2 rounds, compact-and-verify, clean from the original Qwen); Paths 2 and 3
  dropped (Path 3 rested on a bida-pair run). PATH1_PLAN.md. Cheap test rpJ (rc1-soup-30 stand-in) queued on GPU 2 after cp1d,
  after Dev-512 (own held-out questions) + calibration scores of rc1-soup-30 and rpE-80. The full clean run (p1src -> p1r1 -> p1r2,
  seeds 0 / 1) is written and commented in both plans until rpJ passes. consult6's second analysis (of the fact-check and the plan)
  is running.
- Caveat: od2 started on GPU 1 before Dev-512 existed, so its question draws do NOT exclude Dev-512 (its --rp-exclude-dev found no
  file). od2 is screened on the 720, not judged on Dev-512; if it is ever scored on Dev-512, its overlap must be removed first.
- **cp1d final:** step 35 39.5% / -2.8; **step 40 42.6% / -3.3** (svamp 63 / -4.0, mbpp 40 / -5.4, ifeval 27 / -1.7, simple 32 / -3.1,
  multiple 32 / -0.2, no-fit 38 / -2.1); vs curve -1.8. COPD + right/wrong reward ends below the curve (Run A at 40: 34.1 / -2.4,
  -1.4; Run C at 40: 57.1 / -4.7, -0.3). COPD line closed.
- **Dev-512 built (11 Nov ~01:00):** 512 questions (svamp 140, mbpp 100, IFEval 66 counting + 66 other, simple 60, multiple 45 no-fit
  / 35 several) outside the 11,360 questions drawn for training by rpE/rpG/rpH, rpI and od1; the original answers it x8.
- Studies added (reference only, no copying): reviews/competitor_methods_study.md, reviews/rl_length_methods_study.md;
  reviews/claude_path1_review.md (independent review of PATH1_PLAN.md: round 1 likely ~25-35% saved; rpJ dropped).
- **Path 1 final queued (11 Nov ~02:00; PATH1_FINAL.md):** user decisions: public datasets allowed (MetaMath, math1/math2/nofit4;
  recipes stay off-limits), shorten bar 8/10 in round 1, rpJ dropped. Round 1 from the original (stored 10 answers, 75% of the
  pool, Dev-512 near-dupes dropped), 3,200 pairs shorten-dominant (60-70% shorter, rest compact re-read be-right), lr 2e-5 cosine
  to 5e-6, 200 steps; automatic gate (Dev-512 saving >= rpE-80's + 6) before round 2 (on-policy, 25/60/15, 80 steps). Seed 0 on GPU 2
  after the Dev-512 calibration (rc1-soup-30, rpE-40/60/80, rpI-120); seed 1 on GPU 1 after od2.
- **consult6 second analysis** (reviews/consult6_argon/run/final_answer.md) and the Claude review agreed: the first round-1 design
  (39% shorter) would land at 20-33% saved; round 1 must reach ~53-57%.
- **consult7 (Gemini): Path 2 RL design** (reviews/consult7_path2/run/final_answer.md; PATH2_PLAN.md with my caveats): budget-gated
  GRPO from Path 1's final model, per-question target = P35 of its own correct lengths, batch-wise advantage normalisation,
  clip-higher, dynamic sampling, lr 1.5e-5 cosine to 5e-6, 50 steps, 2 seeds; runs only after Path 1. Gemini spend so far ~$8.6.
- 11 Nov ~02:35 **Dev-512 calibration, first 2 of 5** (GPU 2): rc1-soup-30 Dev +58.7% [56.3, 61.0] / -1.83 [-3.70, -0.20]
  (full test 57.4 / -3.5); rpE-80 Dev +46.9% [44.2, 49.8] / +1.36 [-0.28, +2.99] (full test 47.0 / -0.9). Saving reads
  ~0 to +1.3 high; accuracy reads ~+1.7 to +2.3 high on Dev (ranking kept). Gate bar = 46.9 + 6 = 52.9% Dev saving.
  rpE-40 / rpE-60 / rpI-120 next. GPU 1: od2 round 2/10. consult8 (Gemini final Path 1 review) running, cap raised to $15 (user).
- 11 Nov ~03:30 **Independent review by Claude Fable 5.2** (user switched models; reviews/fable_path1_review.md): Path 1 = a clean
  rewrite of rc1 -> rpE with no new ingredient, expected on the existing frontier (47 / -0.9 .. 50.6 / -1.9), ~1 in 6 to land in
  the box; round 1 still worth running (cheap, gated). Unexploited lever: IFEval and no-fit (raw Qwen thinking OFF on the full
  test: IFEval 62% saved / -1.3, no-fit 96% / +6.9; every trained full test is worse on IFEval at 41-53% saved); on paper IFEval
  at 57-62% saved lifts rpE-80 to 49.9-51.5 / about -0.6. Proposal: the user's budget ladder, minimal version (measure budgets
  0 / 30 / 60 first, then "short think" pairs per question; short real think, never empty). Red flags: round 2 shorter bar 6/8;
  Dev-512 too noisy to pick among round-2 steps; seed spread vs the gate; od2 low value; counting-rule shortfall refilled in
  repair_pairs.py L215; the README's own bar is long passed. Nothing changed in the queues (user decides).

## 11 Nov ~03:00-03:45 — user asleep: "queue path 1 directly", "independently plan and queue the repair on ifeval and tool calls", "make the changes you feel needed"
- **Dev-512 calibration, 3 of 5:** rpE-40 Dev +50.5% [47.9, 53.0] / +0.55 [-1.14, +2.11] (full test 50.6 / -1.9). With rc1-soup-30
  (58.7 / -1.83 vs 57.4 / -3.5) and rpE-80 (46.9 / +1.36 vs 47.0 / -0.9): saving within ~1 point of the full test, accuracy
  +1.7 to +2.5 high, ranking kept. rpE-60 and rpI-120 running.
- **consult8 (Gemini 4 Argon) done, $3.64: GO for round 1.** Plan-vs-code table: everything verified except IFEval's counting
  half (refill) and round 2's guard / be-right collision; 4 bugs (round 2 draws Dev-512 near-duplicates under other ids; guard
  and be-right pairs collide and the shortfall is under-reported; counting half refilled; tool top-up source names = false
  alarm, checked in prepare_types.py). reviews/consult8_path1_final/run/final_answer.md.
- **Real data pulled from Drive to the Mac for testing** (Chrome downloads, read-only): the five full-test record files
  (original, thinking off, rc1-soup-30, rpE-80, rpI-120), the original's 10 stored answers (1.7 GB, 17,870 questions), rc1-soup-30's
  and rpE-80's own answers (rp-rpH, rp-rpI); the 2,759 test questions rebuilt with utils/eval/benchmarks.py.
- **Why IFEval and no-fit fail (REPAIR_IFEVAL_TOOLS.md, official checkers re-run rule by rule):** IFEval's loss in every trained
  model = shortened RESPONSES on 58 "at least N words / sentences" questions (-1.4 to -1.7 of the -1.5 to -2.4; words / N 0.89 ->
  0.57-0.65), not missing thinking (thinking off: -0.16 on that class, +0.65 on the 264 questions without a counting rule); thinking
  only matters on the other counting rules (thinking off -1.69). No-fit: 28 near-fit questions hold the errors (original 2%,
  thinking off 59%, rpI-120 27%); the other 212 are at 94-97% whatever the thinking. One / several functions: without thinking
  the calls are as good (83.4 vs 84.7%, 76.9 vs 80.1% among calls); the loss is not calling. Every trained model has a thinking
  FLOOR (5th percentile 66-75 tokens on IFEval and tools, < 40 tokens in 0-0.2% of answers): own-answer pairs cannot go below it.
- **Path 1 final changes (PATH1_FINAL.md "Final changes before launch"):** round 2 "shorter" bar 7/8; hard mix rule (cut back to
  the scarcest side; guard and be-right never the same pair; real shortfall); IFEval counting half fixed 1:1 with up to 2 counting
  pairs per question; reserve for round 2 per type (25%, IFEval 33%, several functions 35%); --resp-guard 0.9 on IFEval "shorter"
  pairs; round 2 skips Dev-512 near-duplicates (8-grams); Dev-512 scores print the paired difference (--rp-dev-vs) and the thinking /
  response split; token ids stored compactly (2 GB instead of ~10 GB on round 1's file). Round 1 built locally on the real stored
  answers, both seeds: 13,082 questions, 3,040 pairs, exact mix per type, IFEval 720 / 800 (counting half), 56% "shorter" in every
  prefix, longest pair 8,339 tokens. Round 2 dry-run on rc1-soup-30's real answers: every quota met at the exact mix.
- **rpK (new, independent of Path 1): the IFEval + tool-call repair** from rpE-80 on its own answers (rpI's rollouts, no generation):
  repair_pairs.py --mix-preset typed = "think less" pairs only on IFEval without a counting rule and on tool calls (7/8 bar),
  math and code "be right" + "don't stop early" only; 1,060 pairs (37% "think less"), 60 steps lr 1e-5, Dev-512 at 20 / 40 / 60
  paired against rpE-80. Tests whether IFEval / tools can be shortened while math and code hold (lpt6 said one-type adapters
  spill over; those had no pairs on the other types).
- **Queue:** GPU 2 was about to reach `end` (runtime stop) after the calibration: new `hold` stage + wait lines pushed first
  (ccff1c2), then the real lines: devset, p1src, p1r1s0 (Dev at 100 / 150 / 200), rpK, p1r2s0 (gate rpE-80 + 6), end. GPU 1:
  od2 (round 2 of 10, ~1.5-2 h per round) then p1src, p1r1s1, p1r2s1.

## 11 Nov morning — results of the night (GPU 2), read at 09:05
- **Dev-512 calibration, 5 of 5:** rpE-60 Dev 46.5 [43.9, 49.0] / +1.05 (full 47.8 / -1.2); rpI-120 Dev 42.6 [39.9, 45.3] / +2.49
  (full 41.9 / +0.1). All five (Dev - full): saving +1.3, -0.1, -1.3, -0.1, +0.7 (mean +0.1, within ~1.3); accuracy +1.7, +2.5,
  +2.3, +2.3, +2.4 (mean +2.2). Dev-512 cannot order two models 1 point apart (rpE-60 vs rpE-80 swap).
- **Path 1, round 1, seed 0 (p1r1s0; from the original, 12,668 questions after 318 Dev-512 questions and 289 near-duplicates were
  dropped; 2,964 pairs, IFEval 644 / 800; 200 steps; pairs ordered right 82% at 100, 84% at 150, 91% at 200), Dev-512 x8:**
  | step | saved | accuracy | paired vs rpE-80: saved | accuracy |
  | 100 | 47.5 [44.8, 50.1] | -0.27 | +0.5 [-2.0, +3.3] | -1.63 [-3.15, -0.18] |
  | 150 | 43.7 [41.1, 46.3] | -0.28 | -3.2 [-5.9, -0.4] | -1.65 [-3.28, -0.17] |
  | 200 | 45.9 [43.2, 48.7] | +0.12 | -1.0 [-3.0, +1.2] | -1.25 [-2.92, +0.33] |
  Step 200 per type: math +58% / -2.4, code +53 / -3.2, IFEval +26 / +1.8, one function +43 / +7.5, several +47 / +2.5, no fit +39 /
  +0.6. Thinking + response tokens, original -> model: math 1115+20 -> 470+9, code 1934+80 -> 888+67, IFEval 436+273 -> 197+330,
  one function 159+21 -> 82+21, several 186+18 -> 90+18, no fit 155+28 -> 92+20. IFEval's thinking halved but its RESPONSES grew
  (273 -> 330 tokens), so IFEval saves only 26% (and gains accuracy): the response protections over-corrected.
  **GATE CLOSED** (45.9 < rpE-80's 46.93 + 6): round 2 skipped, as pre-registered. Round 1 saves no more than rpE-80 and is 1.3-1.6
  points less accurate (paired): it does not reach the shorten-hard point the plan needed (Gemini forecast 53-57%; the Claude and
  Fable reviews expected less). The queue then reached `end`: GPU 2 stopped at ~07:40.
- **rpK, the IFEval + tool-call repair (from rpE-80, 1,060 pairs, 60 steps), Dev-512 x8:**
  | step | saved | accuracy | paired vs rpE-80: saved | accuracy |
  | 20 | 50.5 [48.2, 52.8] | +1.51 | +3.6 [+1.5, +5.6] | +0.15 [-1.05, +1.35] |
  | 40 | 50.4 [47.6, 53.5] | +0.95 | +3.5 [+1.4, +5.5] | -0.42 [-1.63, +0.80] |
  | 60 | 53.8 [51.2, 56.5] | +0.76 | +6.8 [+4.6, +9.1] | -0.60 [-2.05, +0.82] |
  Per type, saved / accuracy (rpE-80 on Dev: math 49 / +2.1, code 49 / +3.1, IFEval 45 / -2.8, one function 42 / +3.8, several 46 /
  +3.9, no fit 45 / +1.4): step 20: math 50 / +1.2, code 57 / +2.8, IFEval 48 / -0.7, one function 44 / +3.3, several 49 / +1.4, no
  fit 50 / +2.8; step 60: math 54 / +0.8, code 61 / +3.3, IFEval 45 / -3.4, one function 51 / +2.1, several 55 / -1.8, no fit 58 /
  +6.4. Tokens at step 60 (thinking + response): no fit 72+5, one function 68+20, several 74+18, IFEval 113+275, math 517+9, code
  723+66. Reading: tool calls shortened as intended (+9 to +13 points); IFEval's thinking fell (to 113) and its response came back
  to the original's length (275 vs 273), so IFEval's saving is unchanged; **math and code did NOT hold in tokens** (code 49 -> 61%
  saved, math 49 -> 54: the shared habit spills over even with holding pairs in every batch), but their accuracy held on Dev.
  Net: +6.8 points of saving for -0.6 of accuracy (paired, interval -2.05 to +0.82), a far better trade than our curve (~-1.7 for
  +6.8). Applying the paired differences to rpE-80's full test (47.0 / -0.9): rpK-20 ~ 50.6 / -0.75, rpK-40 ~ 50.5 / -1.3, rpK-60 ~
  53.8 / -1.5 (each +-1.3 saving, +-1.3 accuracy; three steps looked at). **rpK-20 is the first checkpoint whose estimate sits in
  the goal box**; only a full test can say. Not run: the user's rule is no full test unless asked.
- GPU 1: od2 round 6 of 10 (720: round 2 11.0 / -0.8, round 4 29.3 / -1.0); Path 1 seed 1 is queued after it.
- 11 Nov ~09:20-09:50 (user): **four FULL tests queued** (2,759 x 8, then the 720): rpK steps 20 / 60 / 40 on GPU 2 (stopped since ~07:40:
  it needs the user's restart) and the same list plus Path 1 round 1 step 200 on GPU 1 after od2, before Path 1's seed 1 (a finished
  test is skipped). fullmodel --fm-merge now accepts the original as base (round 1's adapter). **consult9 (Gemini 4 Argon, cap $10):**
  analysis of Path 1's round 1 and of rpK with "as much detail as possible" (user): per-question Dev-512 tables for 8 models
  (downloaded from Drive: the original, rpE-80, p1r1s0-100 / 150 / 200, rpK-20 / 40 / 60), the full test per question and IFEval rule
  verdicts for 5 models, every pair of round 1 and of rpK (csv + decoded examples), the raw GPU logs, the whole notebook, the dossier,
  every plan and every earlier review. reviews/consult9_path1_results/. Dev-512 split, rpE-80 -> rpK-60 (thinking + response): math
  564+9 -> 517+9, code 954+66 -> 723+66, IFEval 146+245 -> 113+275, one function 83+21 -> 68+20, several 92+18 -> 74+18, no fit
  90+11 -> 72+5: code gives the largest part of rpK's extra saving (-24% tokens at flat Dev accuracy, 49.2 -> 49.4%).
- 11 Nov ~10:25 **Runaway answers (found while following consult9's look at truncation; full-test records):** answers that loop
  until the token cap (always graded wrong) hold a large share of the tokens. IFEval: original 3.60% of answers = 21.2% of its
  tokens; rc1-soup-30 2.24% = 26.8%; **rpE-80 2.43% = 26.2% (mean 380 tokens, 287 without them)**; rpI-120 3.12% = 31.1%. Code (cap
  8,192): original 7.80% = 30.1%; rpE-80 1.12% = 8.1%; rpI-120 1.69% = 10.5%. Math: original 1.74% = 10.4%, trained ~0.
  In rpE-80's IFEval the loops are in the RESPONSE (98 of 105: repeated quotes, placeholders, "o*o*o*", a word repeated), on 51
  questions, only 2 of them always (49 questions also have finished answers). Dev-512 shows the same: rpK's IFEval median response
  is unchanged (81 tokens) while its mean rises 245 -> 275 because cut answers go 3.6% -> 4.6%; Path 1's round 1 raises both (median
  96 -> 117, cut 3.9 -> 4.8%). On paper, rpE-80 without its IFEval runaways = IFEval 58.8% saved instead of 45.4% -> headline 50.4%
  at the same or better accuracy; without the code runaways too: ~51.3%. **Our pair builder never uses an unfinished answer** (the
  rejected answer must be finished), so no pair ever says "do not loop". Supply in rpE-80's own answers (rpI rollouts): 145 IFEval
  questions with both an unfinished and a finished answer (94 with a right one), code 32, math 62. Caveat for the user's rule:
  michal-bida's rejected side includes cut-off answers; a dedicated "a finished answer beats a runaway one" pair is generic, but
  the overlap is flagged. Not built, not queued.
- 11 Nov ~11:05 **consult9 (Gemini 4 Argon) done, $6.92, 201 turns: reviews/consult9_path1_results/REPORT.md.** Main points, with my
  check against the pair tables (figures confirmed unless noted):
  (1) Path 1's round 1 missed its forecast (53-57%) by 7-9 points for two reasons. IFEval: 25.6% saved vs rpE-80's 44.7% (math 57.7
  vs 49.5 and code 52.6 vs 49.3 shortened MORE than rpE-80); in round 1's IFEval pairs the chosen answer's response is longer than
  the rejected one's in 76% of pairs (+47 tokens; counting-rule pairs 201 vs 145 tokens, 81% longer; even the "shorter" pairs 118 vs
  99, 61% longer, an effect of --resp-guard), only 25% of IFEval pairs were "shorter" pairs, and runaway answers rose. Structure:
  the pairs' own chosen / rejected token ratio, weighted as the headline, is 46% saved, and the model landed at 45.9%: one offline
  round on the original's answers under fp32 moves the model to its chosen answers' length (other fp32 single rounds in the log:
  45-49%). It admits three errors of consult8: it never computed that ratio, it compared with bf16 runs and with rc1-soup-30
  (4 stages), and it endorsed the count-rule and response-guard rules without checking their effect on response length.
  (2) Path 1: stop as queued (drop seed 1; the fallback and round 2 as designed cannot reach the box); stop od2; drop the full test
  of p1r1s0-200 (the user asked for it: kept unless told otherwise).
  (3) rpK: the saving gain is real (paired +3.6 / +3.5 / +6.8, intervals of about +-1.6); in rpK's 320 IFEval pairs the chosen answer
  thinks 46 tokens less but WRITES 42 tokens more (76% longer), net -4 tokens, which is why IFEval's saving did not move; rpK's
  finished IFEval responses did not lengthen, the rise is runaways (36 -> 43 -> 49 of 1,056). Its full-test forecasts: rpK-20
  50.1 / -0.80 (45% chance in the box), rpK-40 50.3 / -1.25 (25%), rpK-60 53.2 / -1.45 (15%). (Not confirmed: it says code's
  be-right pairs carried a shortening signal; over ALL of rpK's code pairs the chosen answer is LONGER (776 vs 693 tokens), so
  code's -24% is spill-over and fewer runaways, not its own pairs.)
  (4) Next: NOT cut answers below the floor; **rpL** = rpK with three IFEval fixes (counting-rule pairs only on true "at least N"
  rules with the chosen response capped at 1.35x + 25; "shorter" pairs guarded on the question's median response and never a
  longer response than the rejected one; a finished answer beats a runaway one), IFEval 20% counting / 50% shorter / 30% be-right,
  tools 50 / 50, 60 steps, Dev-512 at 25 / 40 / 55, pass = saved >= 52.5 and accuracy >= +1.25; then an interpolation of rpK-20 and
  rpK-60 if the full tests bracket the box.
  (5) Clean recipe from the original: two rounds: round 1 = 90% "shorter" / 10% be-right on the stored answers (bar 7/10, 1.40x,
  lr 5e-5, 150 steps; forecast 51.5-53.5% on Dev at about -3 to -3.7 full-test), round 2 = the typed repair on its own answers;
  round 1 alone costs ~1 GPU-hour (no generation). My caveat: 90% "shorter" pairs at lr 5e-5 is the regime where our log and the
  competitors' notes show accuracy collapsing; its forecasts have been 7-9 points off once already.
- 11 Nov ~11:45 (user: "1. Yes 2. Yes 3. Yes keep it") **rpL built and queued; Path 1 seed 1 dropped; the full test of Path 1's round 1
  kept.** repair_pairs.py: --mix-preset typed2, --count-mode atleast (new at_least()), --count-share, --count-resp-cap,
  --resp-median, --resp-max, --right-resp-cap, and the anti-loop candidate list (a finished right answer > a runaway answer;
  before, unfinished answers were never used). Built on rpE-80's real answers (rpI's rollouts): 1,060 pairs = 440 "think less",
  478 be-right (64 counting-rule), 64 guards, 78 anti-loop (IFEval 48, math 16, code 14); IFEval without the anti-loop pairs:
  thinking 105 vs 166, response 110 vs 113, chosen response longer in 35% (rpK: 76%), total 214 vs 279 (rpK: 270 vs 274). The stage
  dry-run builds the identical file. dev_score.py also prints the share of runaway answers per type (rpK-60 on Dev: IFEval
  3.9% -> 4.6%, math 7.8% -> 0.4%, code 6.0% -> 0.4%). Queues: GPU 2 = rpK-20 full test (running) -> rpL -> rpK-60 -> rpK-40 full
  tests; GPU 1 = od2 (running) -> the full test of p1r1s0-200 only. REPAIR_IFEVAL_TOOLS.md has the rpL section.
- 11 Nov ~12:10 (user: "where does Path 1 redesigned go?") **Path 1's round 1 redesigned = p1br1s0, queued on GPU 1 after od2** (before the
  full test of p1r1s0-200). Not consult9's version (90% "shorter" pairs, lr 5e-5: the regime where the log shows accuracy
  collapsing) but ONE change at a time: last night's round 1 (same stored answers, 60 / 40 and 70 / 30 mixes, lr 2e-5 cosine,
  200 steps) with (a) IFEval's pairs as in rpL and (b) anti-loop pairs. repair_pairs.py --mix-preset round1b. Built on the real
  stored answers: 3,120 pairs, every quota full: math 480 shorter / 200 be-right / 120 anti-loop, code 560 / 120 / 120, IFEval 400
  shorter / 120 be-right / 160 counting ("at least N" only) / 120 anti-loop, tool calls as before. IFEval without the anti-loop pairs:
  thinking 239 vs 527, response 127 vs 135, chosen response longer in 36% (last night: 76%). Anti-loop supply in the original's
  answers: math 539 questions, code 195, IFEval 146. Longest pair 8,386 tokens; pairs are ~40% longer on average (slower steps).
  Dev-512 at 100 / 150 / 200, paired against p1r1s0-200. Round 2 (rpL-style, on its own answers) only if it clears ~51% saved.

## 11 Nov ~12:50 — rpK step 20, FULL TEST (2,759 x 8): **49.9% saved / -1.3** (official weighting)
Per type (rpE-80's full test in brackets): math 58 / -2.6 (54 / -1.2), code 51 / -0.3 (46 / -1.0), IFEval 45 / -3.1 (45 / -2.6), one
function 40 / +0.3 (37 / +0.1), several 42 / -1.5 (39 / +0.1), no fit 54 / +3.6 (50 / +3.4). vs curve +1.57 (rpE-80 +1.45, rpI-120 +1.64).
- Outside the box on both sides, narrowly: 0.1 point of saving and 0.3 point of accuracy short. +2.9 points of saving over rpE-80 for
  -0.4 of accuracy: a little better than our frontier's slope, the same distance above the curve as rpE-80 and rpI-120.
- The forecasts were right on saving (mine 50.6, consult9's 50.1) and too kind on accuracy (mine -0.75, consult9's -0.80): the paired
  Dev-512 difference said +0.15, the full test says -0.4.
- **Where the accuracy went: math.** SVAMP lost 1.4 points against rpE-80 (-0.35 on the headline) while shortening 4 more points;
  several functions -1.6 (-0.1), IFEval -0.5 (-0.13); code gained 0.7 (+0.18). Dev-512 did not see the math loss (its math read
  50% / +1.2 for rpK-20): Dev-512's math is full-difficulty GSM8K (the original solves 75%), the test's is SVAMP (89%). rpK's math
  "hold" pairs were built on the same hard GSM8K questions (rpI's draw predates the public-dataset decision), so easy SVAMP-like
  math was not protected. **Dev-512 cannot be trusted for math accuracy; rpL (same math pairs) will carry the same risk.**

## 11 Nov ~13:45 — rpL (rpK with the IFEval pairs fixed + anti-loop pairs), Dev-512 x8
| step | saved [95%] | accuracy | paired vs rpE-80: saved | accuracy |
| 20 | 50.3 [47.6, 53.0] | +1.30 | +3.4 [+1.6, +5.3] | -0.07 [-1.29, +1.22] |
| 40 | 53.8 [50.8, 56.9] | +0.43 | +6.9 [+4.7, +9.4] | -0.93 [-2.34, +0.36] |
| 60 | **58.9 [56.0, 61.8]** | **+0.92** | **+11.9 [+9.6, +14.4]** | **-0.44 [-1.95, +1.04]** |
Per type at step 60 (saved / accuracy; thinking + response): math 56 / +1.4 (489+9), code 66 / +2.6 (620+65, runaways 0.1%), IFEval
57 / -4.3 (104+199, runaways 2.9% vs rpE-80's 3.6%), one function 52 / +5.2 (66+21), several 57 / -0.7 (70+18), no fit 59 / +5.6 (70+5).
Step 20: math 51 / +1.7, code 52 / +3.5, IFEval 50 / -2.7 (132+221), tools 45-51. Step 40: math 50 / +2.3, code 59 / +1.4, IFEval
54 / -4.5 (118+209), tools 48-55. Training 75 / 74 / 79% pairs ordered right; ~4x slower than rpK (4,000-token rejected answers).
- The two fixes work on tokens: IFEval's saving 44.7 (rpE-80) -> 50 / 54 / 57%, where rpK stayed at 45-48%; loops down.
- **IFEval accuracy on Dev falls on the "at least N" questions only** (35 questions): rpE-80 50.4 -> rpL 47.5 / 44.3 / 42.1 (rpK: 53.9 at
  step 20, 50.7 at step 60); no counting rule 65.3 -> 67.8 / 66.3 / 67.4; other counting rule 47.2 -> 45.6 / 44.8 / 46.0. The counting
  pairs (20% of IFEval, response capped) no longer protect those questions enough against 160 "think less" pairs.
- Code saves much more (954 -> 620 thinking tokens) at flat Dev accuracy; math reads fine on Dev, which is blind on SVAMP-style math.
- Applying the paired differences to rpE-80's full test (47.0 / -0.9) gives rpL-60 ~58.9 / -1.3; rpK-20's full test came 0.55 point
  below such an estimate, so a fair guess is 57-59% at -1.5 to -2.2: outside the box on accuracy, but far above our old frontier
  (rc1-soup-30: 57.4 / -3.5) if it holds. No full test of rpL is queued (the user decides).
- GPU 2 moved on to the full test of rpK-60. consult10 (Gemini, cap $8) relaunched with the rpK-20 full test per question and rpL's
  per-question Dev-512 results; free to change approach; reviews/consult10_after_rpK20_full/.
- 11 Nov ~14:00 (user: "run rpL full eval tests steps 60 and 40", "queue them for after"): queued without displacing a queued test:
  GPU 1 = full test of p1r1s0-200 (running) -> **full test of rpL-60** -> p1br1s0; GPU 2 = full test of rpK-60 (running) -> rpK-40 ->
  **full test of rpL-40**. od2 finished (GPU 1 moved on at ~13:20).

## 11 Nov ~15:10 — rpK step 60, FULL TEST (2,759 x 8): **54.4% saved / -1.4** (official weighting)
Per type (rpK-20's full test, then rpE-80's, in brackets): math 61 / -2.3 (58 / -2.6; 54 / -1.2), code 55 / -1.2 (51 / -0.3; 46 / -1.0),
IFEval 49 / -3.0 (45 / -3.1; 45 / -2.6), one function 45 / -1.5 (40 / +0.3; 37 / +0.1), several 48 / -1.0 (42 / -1.5; 39 / +0.1), no fit
62 / +6.2 (54 / +3.6; 50 / +3.4). **vs curve +2.38: the highest of any full test** (rpI-120 +1.64, rpK-20 +1.57, rpE-80 +1.45).
- 4.5 points more saving than rpK-20 for 0.1 point of accuracy; 7.4 points more than rpE-80 for 0.5. Outside the box on accuracy
  only (-1.4 for -1.0), with 4.4 points of saving to spare.
- Dev-512 predicted it well this time: paired +6.8 / -0.60 against rpE-80 -> 53.8 / -1.5; measured 54.4 / -1.4.
- Where the accuracy sits: IFEval -3.0 (0.75 point of the headline), math -2.3 (0.58), code -1.2 (0.30), one function -1.5; no
  fit +6.2 lifts the tool-call benchmark to +0.8. A straight line from rpE-80 to rpK-60 passes 50% saved at about -1.1.

## 11 Nov ~15:20 — Path 1's round 1 (p1r1s0 step 200), FULL TEST (2,759 x 8): **43.4% saved / -1.5** (official weighting)
Per type: math 59 / -3.2, code 44 / -3.3, IFEval 23 / +0.4, one function 40 / -0.2, several 42 / +1.2, no fit 45 / +0.4. vs curve +0.26: on
the curve, dominated by rpE-80 (47.0 / -0.9) and rpI-120 (41.9 / +0.1). Dev-512 had read 45.9 / +0.12: for this model built straight
from the original, Dev-512 reads the saving 2.5 points too HIGH (code: 52.6% on Dev, 44% on the test) and the accuracy 1.6 points too
kind (the rc1 lineage: saving within ~1.3, accuracy +2.2): the Dev-512 offsets are not the same for every lineage.
IFEval saves only 23% (responses inflated by the pair rules, as found); math and code carry the loss (-3.2, -3.3).
- 11 Nov ~15:25 **consult10 (Gemini 4 Argon) done, $2.64: reviews/consult10_after_rpK20_full/REPORT.md** (written before the rpK-60 and
  p1r1s0-200 full tests arrived). Main points:
  (1) rpK-20's math loss (-107 right answers of 8,000 vs rpE-80): all of it is NEW wrong answers whose thinking is 0.5-1.0x rpE-80's
  median right thinking on the question (+115; under 0.5x only +24): a one-pass calculation that skips the final re-check, on the
  long-thinking half of the questions. Our "don't stop early" guard only catches wrong answers under 0.5x, and the math "be right"
  pair picks the SHORTEST right answer (chosen shorter than rejected in 59% of rpK's math pairs). Dev-512 hid it because its math
  is hard GSM8K; on the 105 Dev math questions where rpE-80 thinks under 600 tokens, rpK-20 loses 1.43 points (full test: 1.34):
  a free SVAMP-like check. It also proposes a new Dev-SVAMP-200 slice.
  (2) rpL: real on saving and loops. The "at least N" collapse is on the 14 true word / sentence-count questions (37.5% -> 19.6%):
  my at_least() rule let placeholders / highlighted sections / letter frequency take two thirds of the 64 counting pairs, the 1.35x
  cap rejected valid long answers, the "shorter" pairs still shrink the response (chosen shorter in 67%), and steps 40-60 of loop
  pairs compress normal long answers. rpL's 16 math anti-loop pairs turned its math pairs into a shortening signal (Dev math thinking
  -17%): it forecasts rpL-60's full test at 56.8-58.8% / -1.60 to -2.15 (math 60-62% / -3.4 to -4.2) and advises to full-test rpL-60 only.
  (3) **rpM** (typed3): from rpE-80; math protected on SVAMP-like questions (rpE-80 answers 1,200 new ones), 240 math pairs = half
  widened guards (wrong thinking under 0.95x the median right thinking, chosen within 0.95-1.25x), half length-held be-right, no
  "shorter", no loop; IFEval: counting pairs only on true length / keyword-count rules with the chosen response 1.10x+15 to 1.75x+60
  of the rejected one's (25%), "shorter" pairs only on questions with no counting rule and with the response within 5% (45%), be-right
  22.5%, response loops 7.5%; several functions 50 / 50; code as rpL; 60 steps, Dev at 30 / 45 / 60 on Dev-512 + Dev-SVAMP-200. Its
  forecast: 53.5-55.0% / -0.55 to -0.85. Also a zero-training check: rpE-80 + 0.55 x rpL-60's adapter.
  (4) p1br1s0: three edits before it starts (math 45% shorter / 35% be-right / 15% widened guards / 5% loop; fewer loop pairs; the
  IFEval rules of rpM; several functions 50 / 50; gate margin 4).
  (5) It judges the box reachable with margin (53-55% at -0.6 to -0.8) through rpM.
- 11 Nov ~16:40 **rpM built, tested and queued (GPU 2, after the rpK-40 full test, before rpL-40's); the redesigned round 1 becomes
  p1cr1s0 (GPU 1, after the rpL-60 full test).** Code: repair_pairs.py --mix-preset typed3 / round1c, --count-mode length
  (length_rule()), --count-resp-min / --count-resp-pad, --no-shorten-counting, --resp-near-long, --loop-response-only, --guard-wide,
  --right-hold (length-matched right / wrong pair), a "svamp_hard" part beside "svamp" (--easy-math-prefix); the older pair files
  (round 1, rpK, rpL) rebuild byte-identical. track.py: devsvamp stage (Dev-SVAMP-200), devscore --ds-dev / --ds-vs, ptrepair
  --rp-extra-questions (new questions answered by the base model beside reused answers) and --rp-dev-svamp; every generated
  question now carries its own id (before, questions without a pool id were named "source-line number": two answer files of one
  run could share ids for different questions; this only threatened the fill-until-full path, never run).
  Checked on rpE-80's real answers (math on the hard GSM8K answers only: the SVAMP-like ones are generated on the GPU): math guards
  700 vs 487 thinking tokens (chosen longer in 100%), math be-right 612 vs 585; IFEval counting pairs 314 vs 217 response tokens,
  "shorter" pairs thinking 99 vs 185 with the response 82 vs 82; IFEval 313 of 320 (141 "shorter" candidates for 144).
  p1cr1s0 on the original's real stored answers: 3,025 pairs: math 360 shorter / 280 be-right / 120 widened guards / 40 anti-loop,
  code 560 / 104 / 80 guards / 56, IFEval 310 shorter / 206 be-right / 141 counting / 48 anti-loop (705 of 800), several functions
  100 / 100; 144 anti-loop pairs (p1br1s0 had 360). Whole flow dry-run with the queue lines (fake GPU answers).
  Queues: GPU 2 = rpK-40 full test (running) -> devsvamp -> Dev-SVAMP scores of rpE-80, rpK-60, rpL-60 -> rpM -> rpL-40 full test.
  GPU 1 = rpL-60 full test (running) -> devsvamp -> Dev-SVAMP score of p1r1s0-200 -> p1cr1s0.
- 11 Nov ~16:05 (user: "make a new argon review with the full scores") **consult11 launched** (Gemini 4 Argon, cap $6): the full tests of
  rpK-60 and of Path 1's round 1 per question (records downloaded; tables rebuilt for 8 models), rpM and p1cr1s0 as built, and five
  questions (what the full tests change; edits to rpM before it starts; a repair on top of rpK-60 instead of rpE-80, or an
  interpolation; p1cr1s0; the next 12 GPU-hours). New facts from the records: IFEval "at least N words / sentences" (58 questions):
  original 33.2% right, rpE-80 18.5, rpK-20 16.8, rpK-60 18.1, **Path 1's round 1 40.7** (its responses there are 981 tokens against the
  original's 668), while its "fewer than N" falls to 60.1 from 72.9: response length moves both classes. SVAMP wrong answers by
  thinking band (< 0.5x / 0.5-1.0x / >= 1.0x of rpE-80's median right thinking): rpE-80 82 / 284 / 396, rpK-20 106 / 375 / 395, rpK-60
  155 / 396 / 315.
- 11 Nov ~16:45 **consult11 (Gemini 4 Argon) done, $1.77: reviews/consult11_full_scores/REPORT.md** (its report draft: the final call
  returned only a planning note). Main points:
  (1) rpK-60 vs rpK-20 (+4.55 saved for -0.16): the math loss against rpE-80 was paid in the first 20 steps (questions where rpE-80
  thinks longest); from 20 to 60 math held (+0.24) while IFEval's thinking fell 28 tokens and tool calls shortened; measured against
  each model's OWN median right thinking the wrong-answer bands do not move between steps 20 and 60. One function's -1.75 from
  step 20 to 60 is 82% NON-calls (call rate 97.19% -> 95.75%): 140 "no fit" pairs against 48 + 36 be-right pairs on one / several
  functions pushed the model to decline. 93% of rpK-60's -1.44 sits in IFEval (-0.75 headline: "at least N" and other counting
  rules, 2.70% runaways) and math (-0.59). 60 steps is the right dose.
  (2) Path 1's round 1 proves IFEval's "at least N" deficit is a response-length effect (40.7% right with 981-token responses), and
  that rewarding length everywhere costs "fewer than N" (-12.85 points). Dev-512 over-reads a from-the-original model (its code has
  6% runaways in the original; its IFEval is 26.5% "at least N").
  (3) **rpM: run as is** (the length-rule pairs cannot hurt "fewer than N": those questions get be-right pairs with a capped
  response; math rules sized right). Its forecast: step 60 53.8-56.2% / -0.65 to -1.20 (central 55.0 / -0.90).
  (4) Interpolating rpE-80 and rpK-60: rejected (the line crosses 50% at about -1.15, and the path is concave).
  (5) **rpN** = an accuracy-only repair on top of rpK-60: rpK-60 answers 3,200 new questions x8 (~1.1 GPU-h), 800 pairs with NO "think
  less" pair: math 240 (widened guards + length-held be-right, SVAMP-like), IFEval 240 (96 length-rule pairs, 120 be-right, 24 response
  loops), code 120 (90 / 18 guards / 12 loops), tool calls be-right only (one function 80, several 80, no fit 40); 45 steps, Dev at
  15 / 30 / 45 paired against rpK-60; expected 51.8-53.0% / -0.60 to -0.90.
  (6) **p1cr1s0: run as is**; its round 2 only if Dev-512 >= 47.5% saved (IFEval >= 42%), accuracy >= +0.50 and Dev-SVAMP within 0.5
  of rpE-80. It advises dropping rpL-40's full test for a full test of rpM's best step, and gives 65% for rpM, 78% overall (its
  forecasts have run about half a point too kind on accuracy).

## 11 Nov ~16:50 — rpL step 60, FULL TEST (2,759 x 8): **59.3% saved / -1.7** (official weighting)
Per type (rpK-60's full test, then rpE-80's, in brackets): math 64 / -2.7 (61 / -2.3; 54 / -1.2), code 62 / -0.9 (55 / -1.2; 46 / -1.0),
IFEval 58 / -3.4 (49 / -3.0; 45 / -2.6), one function 47 / -2.0 (45 / -1.5; 37 / +0.1), several 49 / -0.6 (48 / -1.0; 39 / +0.1), no fit
62 / +4.6 (62 / +6.2; 50 / +3.4). **vs curve +3.20: by far the highest of any full test** (rpK-60 +2.38, rpI-120 +1.64). Our previous
model at this saving, rc1-soup-30, read 57.4 / -3.5.
- 12.3 points more saving than rpE-80 for 0.8 point of accuracy; 4.9 more than rpK-60 for 0.3. The forecasts held: mine 57-59% / -1.5 to
  -2.2, consult10's 56.8-58.8 / -1.60 to -2.15; math came out better than consult10 feared (-2.7, not -3.4 to -4.2).
- The IFEval fix and the anti-loop pairs are confirmed on the full test: IFEval 58% saved (rpK-60 49%, rpE-80 45%) for -0.4 more;
  code 62% saved with better accuracy than rpK-60.
- Outside the box on accuracy only, with 9.3 points of saving to spare. Where the accuracy sits: IFEval -3.4 (0.85 point of the
  headline), math -2.7 (0.68), code -0.9 (0.22), one function -2.0. A straight line from rpE-80 to rpL-60 passes 50% saved at about
  -1.1: mixing weights is not enough; an accuracy repair that spends part of the spare saving is the open route.

## 11 Nov ~17:05 — rpK step 40, FULL TEST (2,759 x 8): **51.2% saved / -0.8: THE FIRST FULL TEST INSIDE THE GOAL BOX** (> 50% at better than -1)
Per type: math 58 / -2.3, code 52 / -0.1, IFEval 46 / -1.9, one function 41 / +0.3, several 45 / -0.8, no fit 58 / +4.7 (headline accuracy
-0.75 before rounding). vs curve +2.33. rpK = rpE-80 + 40 steps of the typed repair ("think less" only on IFEval and tool calls).
- Caution: its neighbours on the same run read 49.9 / -1.3 (step 20) and 54.4 / -1.4 (step 60); step 40's IFEval is -1.9 where
  both neighbours are near -3.0. One full test carries about +-0.7 accuracy and +-1.2 saving: part of this is luck, and the
  margin is thin (1.2 points of saving, 0.2 of accuracy). Dev-512 had ranked step 40 below step 20 on accuracy.
- consult9 and consult10 both advised dropping this test as the least informative; the user asked for it.
- 11 Nov ~18:05 **Dev-SVAMP-200 (200 SVAMP-like math questions) does NOT see the math loss either:** rpE-80 49.4% / +3.12; rpK-60 55.8 / +3.12
  (paired vs rpE-80: +6.4 / +0.00, full test said -1.1); rpL-60 59.3 / +2.38 (paired -0.75; full test -1.5); p1r1s0-200 54.7 / -1.88 (scored
  on GPU 1, on its own copy of the original's reference answers). Saving reads right; 200 questions are too few for 1-point accuracy gaps.
  The "hold" stage fails on GPU 2's runtime (KeyError: its track.py copy predates it); the 15-minute wait before rpM was skipped.
- **consult12 (Gemini 4 Argon) done, $1.55: reviews/consult12_box/run/final_answer.md.** (1) rpK-40: the saving is real; of its accuracy
  bump over its neighbours ~+0.22 is real (tool-call threshold, code, "at least N" responses) and ~+0.37 luck: true value ~51.2 / -1.10,
  ~25% chance its true accuracy is better than -1; submittable as is (the evaluation is fixed); do NOT run a second seed of rpK. (2) rpN
  on rpL-60 dominates rpN on rpK-60 (twice the saving to spend at -0.25 accuracy); the accuracy-only repair rpE-80 -> rpI-120 bought
  +0.21 accuracy per point of saving; rpN from rpL-60 forecast 54.2-56.5 / -0.55 to -0.95; rpM forecast unchanged 53.8-56.2 / -0.65 to -1.20.
  IFEval length-rule pairs should come from rpE-80's answers (rpL-60 writes too short to give many). (3) Path 1: let p1cr1s0 finish, then
  stop Path 1 (no round 2, no full test unless rpM and rpN both fail). The clean recipe to present = two stages: iterative self-DPO
  compression from the original (-> rc1-soup-30), then typed self-repair (rpE -> rpK / rpL / rpM / rpN), all on the model's own answers.
  (4) Drop rpL-40's full test (dominated); full-test the best of rpM and rpN.
- **rpN built and queued on GPU 1 after p1cr1s0** (repair_pairs.py --mix-preset hold; on rpE-80's answers: IFEval 96 length-rule + 122
  be-right + 22 loops, code 86 / 24 guards / 10 loops, tool calls be-right only, every quota full; older pair files still identical).
- 11 Nov ~18:30 (user) rpL extended to steps 80 / 100 / 120 (resumed from step 60; Dev-512 at 80 / 100 / 120), queued on GPU 2 after rpM, before rpL-40's full test.
- 11 Nov ~19:00 (user) **Next phase decided:** once rpM, rpN, rpL 80/100/120, p1cr1s0 and the rpL-40 full test are finished, ONE Argon call with all the data (newest included) to design a CLEAN recipe from scratch, simple, not combining many methods (the user likes 'overshoot the saving, then correct accuracy'; Argon free to propose anything); then build and run one or two clean routes.
- 11 Nov (user): at most THREE clean routes.

## 11 Nov evening — rpM (consult10's design: math protected on SVAMP-like questions, IFEval length rules), from rpE-80
- **rpM step 20, Dev-512: 49.3% [46.7, 51.9] / +0.49**; paired vs rpE-80 **+2.3 saved / -0.87 accuracy**. Per type: math 49 / +1.7 (569+9 tokens),
  code 55 / +2.0, IFEval 46 / -3.9 (138+247, runaways 3.9%), one function 44 / +2.5, several 49 / +1.4, no fit 51 / +2.2. Dev-SVAMP-200: 51.8 /
  +3.44, paired vs rpE-80 +2.4 / +0.31 (math held, as designed). Worse than rpK-20 (+3.6 / +0.15) and rpL-20 (+3.4 / -0.07) at the same
  step: IFEval's accuracy fell (-3.9) and its runaways did NOT fall (3.9%, rpL-20: 2.9%) with only 24 response-loop pairs.
- **Path 1 round 1, third version (p1cr1s0) step 100, Dev-512: 34.6% [31.9, 37.5] / +2.03**; paired vs last night's round 1 (p1r1s0-200) -11.3
  saved / +1.92 accuracy. Per type: math 37 / +1.2 (703+9 thinking + response), code 36 / +1.1, IFEval 29 / +0.8 (262+242), one function 35 /
  +6.2, several 37 / +2.1, no fit 35 / +5.3. Far less shortening than the first round 1 at step 100 (47.5%): the anti-loop pairs and widened
  guards (chosen LONGER) hold length; accuracy is better. (Dev-512 over-reads this lineage's saving by ~2.5 points.)
- **rpM step 40, Dev-512: 52.6% [49.8, 55.6] / +0.91**; paired vs rpE-80 **+5.7 / -0.46** (rpK-40 +3.5 / -0.42, rpL-40 +6.9 / -0.93). Per type:
  math 53 / +2.4, code 58 / +1.5, IFEval 48 / -2.9 (125+246, runaways 4.0%), one function 48 / +2.7, several 53 / +1.1, no fit 55 / +3.9.
  Dev-SVAMP-200 53.8 / +2.94 (paired +4.4 / -0.19). Better trade than rpL-40 at a similar saving; IFEval's runaways still not reduced.
- 11 Nov ~19:45 (user) full tests of rpM steps 20 / 40 / 60 queued: step 40 on GPU 2 right after rpM (before rpL 80-120), steps 60 then 20 on GPU 1 after p1cr1s0 (before rpN).
- **rpM step 60, Dev-512: 54.9% [52.3, 57.3] / +2.24 — paired vs rpE-80 +8.0 saved / +0.87 ACCURACY (the first repair step on Dev-512 that is
  MORE accurate than rpE-80 while saving 8 points more).** rpK-60 +6.8 / -0.60, rpL-60 +11.9 / -0.44. Per type: math 48 / +1.8 (581+9), code
  63 / +4.9 (runaways 0.1%), IFEval 54 / -1.8 (114+209, runaways 2.8%), one function 50 / +5.0, several 54 / +2.1, no fit 57 / +4.2. Dev-512
  read rpK-60's paired difference right (+6.8 / -0.60 -> full 54.4 / -1.4): applied to rpE-80's full test (47.0 / -0.9) this reads
  ~55 / ~0.0; rpK-20's case was 0.5 too kind, so a fair guess for the full test is 54-55% at -0.1 to -0.6.
- p1cr1s0 step 100, Dev-SVAMP-200: 37.7 / +2.06 (paired vs p1r1s0-200 -17.2 / +3.44).
- rpM step 60, Dev-SVAMP-200: 51.4 / +3.12 (paired vs rpE-80 ~0.0 accuracy, as rpK-60's). GPU 2 started the full test of rpM-40 (~20:15).
- **p1cr1s0 step 150, Dev-512: 32.4% / +2.51** (step 100: 34.6 / +2.03): the saving SHRINKS with more steps (math thinking 703 -> 802 tokens,
  code 1218 -> 1302); paired vs the first round 1 -13.5 / +2.39. Dev-SVAMP 28.1 / +2.62. The extra be-right / guard / anti-loop pairs in
  round 1 hold the model long: it moves along the curve toward accuracy, not toward 50%.
- 11 Nov ~20:40 (user: "how much does this transfer ... or is it benchmark overfitting?") **Transfer check built and queued (GPU 2, after rpL
  80-120, before rpL-40's full test):** track.py stage `transfer` + transfer_score.py + prepare_types.py builders tr_math500 (MATH-500,
  whole-number answers, 200), tr_skywork (Skywork-OR1 easy band, 200), tr_arc_c (ARC-Challenge test, 200), tr_csqa (CommonsenseQA
  validation, 200), tr_mmlu (MMLU test, 250), tr_humaneval (HumanEval, 164); none used or imitated in training. The original, rpE-80,
  rpK-40, rpL-60 and rpM-60 answer them x4; per set: saved %, accuracy change with a paired 95% interval, runaways. Not run locally
  (no `datasets` on the Mac): first run on the GPU.
- 11 Nov ~21:00 (user: compare the competitors' generalisation) transfer check extended to the competitors' PUBLIC weights: basta/lightning-mcqwen3 (Hub LoRA, gate not used), matousek-martin Self-Replay 850 and 256 (GitHub release v0.1.2 zips), mrtineu/thinkingcap-test-submission (full model on the Hub). michal-bida and Herz1c published no weights. Measurement only (track.py competitor_model, kept off Drive).
- 11 Nov ~21:15 (user) transfer models: ours rpK-40, rpK-60, rpL-40, rpL-60; competitors basta, matousek 850 / 256 (mrtineu dropped). michal-bida and Herz1c: no public weights anywhere (Hub search, their GitHub repos incl. bida's thinking-compression branch, releases).
- 11 Nov ~21:25 (user) transfer models now: original, rpK-40, rpL-40, rpL-60, bidaREPLICA (our replica of michal-bida's recipe: lpc-bidax5q seed 0 final adapter on the original; 720 46.7 / -0.5; not his weights), basta, matousek 850 (rpK-60 and matousek 256 dropped).
- 11 Nov ~21:30 (user) transfer check at 8 answers per question (as the test), not 4: ~1,200 questions x 8 x 7 models, ~6-7 GPU-hours on GPU 2.
- 11 Nov ~21:40 (user) transfer check split: GPU 2 = original + rpK-40, rpL-40, rpL-60 (after rpL 80-120); GPU 1 = bidaREPLICA, basta, matousek850 (after rpN). Then ALL results go to Argon for the clean-recipe review.
- **p1cr1s0 step 200, Dev-512: 34.4% / +2.90** (paired vs the first round 1 -11.5 / +2.78); Dev-SVAMP 31.7 / +2.44. Steps 100 / 150 / 200: 34.6 / 32.4 / 34.4 saved at +2.0 / +2.5 / +2.9. Path 1's third round 1 = accurate but short of saving (Dev-512 over-reads this lineage's saving by ~2.5): far from 50%. GPU 1 started the full test of rpM-60 (~22:05).

## 11 Nov ~22:40 — rpM step 40, FULL TEST (2,759 x 8): **52.9% saved / -1.1** (about -1.10 before rounding)
Per type (rpK-40's full test in brackets): math 58 / -1.5 (58 / -2.3), code 56 / -1.7 (52 / -0.1), IFEval 50 / -2.5 (46 / -1.9), one function
41 / +0.3 (41 / +0.3), several 44 / 0.0 (45 / -0.8), no fit 57 / +4.1 (58 / +4.7). vs curve +2.36 (rpK-40 +2.33).
- Just outside the box on accuracy (-1.1 for -1.0), 1.7 points more saving than rpK-40 (51.2 / -0.8) for 0.3 less accuracy: on the
  same frontier, not above it.
- **Math protection worked on the full test**: math -1.5 (rpK-40 -2.3, rpE-80 -1.2) at the same 58% saving. The loss moved to code
  (-1.7 vs -0.1; code 4 points shorter) and IFEval (-2.5 vs -1.9).
- Dev-512 had read it +5.7 / -0.46 vs rpE-80 -> estimate 52.7 / -1.36; measured 52.9 / -1.1: right.
- **rpL step 80 (resumed from 60), Dev-512: 62.1% / -0.30**; paired vs rpE-80 +15.2 / -1.66 (step 60: +11.9 / -0.44). Per type: math 60 / -0.4,
  code 67 / +1.5, IFEval 62 / -2.5 (91+181), one function 56 / -0.2, several 60 / **-6.4**, no fit 63 / +6.1. More saving, accuracy now
  falling (several functions collapses). Training: pairs ordered right 88% at 80 and 100, 94% at 120.
- **GPU 1 disconnected (~21:40; its Drive log last saved 21:38, checked at 22:08) during the full test of rpM-60, at 16,896 of 22,072 answers** (Colab shows "Reconnect L4"). The test restarts
  from zero on reconnection (evaluate() discards a run without results.json). Waiting for the user to reconnect.
- **rpL step 100, Dev-512: 66.2% / -3.95** (paired vs rpE-80 +19.3 / -5.31): **one function collapses (-31.9; response 21 -> 12 tokens: it
  stops calling)**, several functions -12.9; math 59 / 0.0, code 72 / +0.8, IFEval 69 / -0.3 (81+141), no fit 65 / +6.9. Past step 60 the typed
  "think less" pairs over-train the tool-call threshold: the model declines real calls. rpL's useful range ends at step 60-80.
- GPU 1 relaunched (~22:20, user reconnected): code at 657adbc, queue = rpM-60 full test (from zero) -> rpM-20 full test -> rpN -> transfer
  part 2.
- 11 Nov ~22:35 (user) full test of rpL-80 queued on GPU 2 after rpL-40's.
- 11 Nov ~22:45 (user note, verified in the READMEs) Herz1c's own held-out checks: 0.6B HumanEval (164 x4) 65.0% saved [62.0, 67.9] / -1.22 [-5.03, +2.74]; 1.7B HumanEval 58.4% / -0.76, MATH-500 (313 x4, graded format-independently) 11.4% / +4.8; both sizes save least and gain accuracy on the hardest MATH-500 levels. Our transfer check (HumanEval 164 x8, MATH-500 whole-number 200 x8) compares directly; our MATH-500 grading needs the 'Final answer:' line.
- 11 Nov ~23:00 transfer check part 1 FAILED on GPU 2 (NameError: prepare_types.py defined the tr_* builders after its __main__ call); fixed (block moved before the guard) and re-queued on GPU 2 after the rpL-80 full test. GPU 1's part 2 will use the fixed code. rpL step 120, Dev-512: 66.4% / -4.59 (one function -40.8, several -13.9): collapse confirmed.
- 11 Nov ~23:10 (user) transfer check: MATH-500 now ALL its whole-number problems (up to 500, was 200) and scored by difficulty level 1-5, as Herz1c reports it.

## 11 Nov ~23:50 — state before the clean-recipe review (log update, user)
Full tests (2,759 x 8) of the typed repair line (all from rpE-80 = 47.0 / -0.9):
| model | saved / accuracy | math | code | IFEval | one function | several | no fit |
| rpK-20 | 49.9 / -1.3 | 58 / -2.6 | 51 / -0.3 | 45 / -3.1 | 40 / +0.3 | 42 / -1.5 | 54 / +3.6 |
| **rpK-40** | **51.2 / -0.8 (IN THE BOX)** | 58 / -2.3 | 52 / -0.1 | 46 / -1.9 | 41 / +0.3 | 45 / -0.8 | 58 / +4.7 |
| rpK-60 | 54.4 / -1.4 | 61 / -2.3 | 55 / -1.2 | 49 / -3.0 | 45 / -1.5 | 48 / -1.0 | 62 / +6.2 |
| rpM-40 | 52.9 / -1.1 | 58 / -1.5 | 56 / -1.7 | 50 / -2.5 | 41 / +0.3 | 44 / 0.0 | 57 / +4.1 |
| rpL-60 | 59.3 / -1.7 | 64 / -2.7 | 62 / -0.9 | 58 / -3.4 | 47 / -2.0 | 49 / -0.6 | 62 / +4.6 |
| Path 1 round 1 (from the original) | 43.4 / -1.5 | 59 / -3.2 | 44 / -3.3 | 23 / +0.4 | 40 / -0.2 | 42 / +1.2 | 45 / +0.4 |
Dev-512 only so far: rpM-20 49.3 / +0.49, rpM-40 52.6 / +0.91, **rpM-60 54.9 / +2.24 (paired vs rpE-80 +8.0 / +0.87)**; rpL-80 62.1 / -0.30,
rpL-100 66.2 / -3.95, rpL-120 66.4 / -4.59 (one function -31.9 / -40.8: the model stops calling tools); p1cr1s0 (Path 1 round 1, third
version) 34.6 / 32.4 / 34.4 saved at +2.0 / +2.5 / +2.9 (steps 100 / 150 / 200): accurate but barely shortens.
Running: full tests of rpL-40, rpL-80 (GPU 2), rpM-60 (restarted after GPU 1 disconnected), rpM-20 (GPU 1); then rpN (accuracy-only repair
of rpL-60) and the transfer check (MATH-500 by level, Skywork, ARC-C, CSQA, MMLU, HumanEval x8: original, rpK-40, rpL-40, rpL-60, our
michal-bida replica, basta, matousek 850).
Method note: every stage of the line (A1, A2, ca1, rc1, rpE, rpK / rpL / rpM) is offline DPO on the model's own graded answers (no SFT,
no RL); close to iterative DPO / Iterative RPO (Pang et al. 2024) but rpK / rpL / rpM each ran ONE round on rpE-80's older answers.
- 11 Nov ~23:50 (user) **clean-round strategy asked from the user's Delos worker (Gemini 4 Argon through its chat/completions endpoint,
  key in the Keychain as delos-worker-key-thinkingcap)**, with all the data so far; reviews/consult13_worker_clean/.
- 11 Nov ~00:05 (user) consult13's spending cap removed (cap set to $100,000; report reserve $3 kept so the report is always written).

### 11 Nov ~23:55 — rpL-40 full test (2,759 x 8, GPU 2)
rpL-40 (rpE-80 + rpL step 40): **tokens saved +55.1% [54.0, 56.2], accuracy -1.37 pts [-2.09, -0.68]**.
Per type: bfcl_irrelevance +58% / +3.9; bfcl_multiple +45% / -0.6; bfcl_simple_python +43% / -0.3; ifeval +55% / -3.0;
mbpp +57% / -1.8; svamp +60% / -1.5. Same accuracy as rpK-60 (54.4 / -1.4) at +0.7 pts more saving; rpL-60 59.3 / -1.7.
Outside the box (accuracy). IFEval remains the main loss.
Note: the consult13 worker report (Gemini via the Delos worker endpoint, $0.78) was judged weak by the user and is discarded;
the clean-recipe review goes back to the OpenRouter Argon harness (as consult6-12).

### 12 Nov ~00:05 — rpM-60 full test (2,759 x 8, GPU 1): IN THE BOX
rpM-60 (rpE-80 + rpM step 60): **tokens saved +54.1% [53.0, 55.3], accuracy -0.93 pts [-1.67, -0.17]** -> inside 50 / -1
(point estimate; the CI still reaches -1.67). Best in-box result so far (rpK-40 51.2 / -0.8: +2.9 pts more saving at ~same accuracy).
Per type: bfcl_irrelevance +61% / +4.2; bfcl_multiple +47% / -1.6; bfcl_simple_python +45% / -0.3; ifeval +50% / -2.3;
mbpp +58% / -0.9; svamp +57% / -1.2. vs rpK-60 (54.4 / -1.4): same saving, +0.5 accuracy (math and code protected).
Dev-512 read 54.9 / +2.24 (saving within 0.8; accuracy ~3.2 too kind).
Decision (user, 11 Nov ~23:59): launch the clean-recipe Argon review (OpenRouter harness, as consult6-12) automatically once all runs finish.
- 12 Nov ~00:20 (user): rpM-60 added to the transfer (generalization) check, GPU 2 after the rpK40/rpL40/rpL60 transfer line
  (base answers reused from Drive). Charts: rpK / rpL / rpM full tests now drawn as three branches from rpE-80 (they were one
  zigzag line); Frontier page regained the README bar and a "no thinking" toggle.

### 12 Nov ~01:25 — rpL-80 full test (2,759 x 8, GPU 2)
rpL-80: **tokens saved +62.8% [61.7, 63.8], accuracy -2.06 pts [-2.86, -1.26]**. Per type: bfcl_irrelevance +66% / +7.7;
bfcl_multiple +53% / -2.9; bfcl_simple_python +51% / -5.1; ifeval +62% / -2.6; mbpp +66% / -1.8; svamp +66% / -2.9.
Most saving ever on a full test; out of the box on accuracy. One-function tool calls start to break (-5.1; rpL-100/120 collapse on Dev).
Dev-512 had read 62.1 / -0.30 (saving right, accuracy ~1.8 too kind). rpL line: 40 55.1/-1.4, 60 59.3/-1.7, 80 62.8/-2.1.
- 12 Nov ~01:45 (user): AIME 2025 check of rpM-60 vs the original, as "The Art of Efficient Reasoning" (arXiv 2602.20945,
  Table 2: Qwen3-0.6B 14.9k -> 8.9k tokens, Mean@8 13.3 -> 24.6, Pass@8 26.7 -> 36.7, 32k budget). transfer stage gained
  --tr-sets / --tr-max-tokens / --tr-tag; prepare_types.py tr_aime25 (math-ai/aime25, 30 problems). Queued on GPU 2 after the
  rpM-60 transfer line: 30 x 8 answers per model at a 32,768-token cap, our "Final answer:" format (the paper uses \boxed{}).

### 12 Nov ~02:10 — rpM-20 full test (2,759 x 8, GPU 1): in the box, just
rpM-20: **tokens saved +50.1% [48.9, 51.3], accuracy -0.94 pts [-1.64, -0.27]**. Per type: bfcl_irrelevance +54% / +3.5;
bfcl_multiple +42% / -0.6; bfcl_simple_python +39% / +0.3; ifeval +47% / -2.6; mbpp +52% / -0.7; svamp +56% / -1.5.
rpM line: step 20 50.1 / -0.94, step 60 54.1 / -0.93 -> 40 more steps bought +4 pts saving at no accuracy cost. rpM-40 52.9 / -1.10
sits between (noise ~0.35 pts). Dev-512 had read 49.3 / +0.49 (accuracy ~1.4 too kind).
- 12 Nov ~03:40, balance: GPU 2's transfer of the original takes ~5.5 h (chunks of 300 q x8 at ~950 tok/s). GPU 1's
  competitor transfer would have regenerated the same base answers; new --tr-wait-base: it answers with its models first,
  then waits for GPU 2's base answers on Drive and scores. rpM-60's transfer and the AIME 2025 run moved to GPU 1.
- rpN (hold repair on rpL-60) Dev scores: step 15 58.5 / +1.16 (Dev-SVAMP 58.9 / +2.44), step 30 59.7 / +0.71 (Dev-SVAMP
  57.3 / +2.81); paired vs rpL-60 ~0 on both saving and accuracy -> rpN changes nothing measurable; IFEval responses still
  shrink (273 -> 186 -> 157 tokens on Dev). No full test planned.
- rpN-45: Dev-512 59.9 / +1.28, Dev-SVAMP-200 56.1 / +2.31 -> same as rpL-60 (Dev 58.9 / +0.92). rpN closed: no effect.
  GPU 1 picked up the new code (6e00d4b): competitor transfer started with bidaREPLICA, base answers to come from GPU 2.
- 12 Nov ~06:40: transfer generation is slow on L4 (~950 output tok/s; ~60-75 min per 300-question x8 chunk, 5 chunks per
  model; KV-cache bound at the 4k/8k caps). Estimate: GPU 2 base done ~08:00, then rpK40/rpL40/rpL60 (~4 h each) -> ~20:00;
  GPU 1 bidaREPLICA/basta/matousek850 (~5 h each) -> ~21:00, then rpM60 transfer (~4 h) + AIME (~1.5 h). Argon waits for all.
- 12 Nov ~09:25 (user: "Launch argon"): consult14 = Gemini 4 Argon (OpenRouter), the clean-recipe review, cap $20 (reserve $4),
  replies up to 60k tokens, context handoff at 190k (the harness keeps prompts below the 272k long-context price tier). Pack:
  reviews/consult14_clean/pack (consult13's pack + the current log + NEW_RESULTS.md of 12 Nov + rpM-60 per-question table and
  answer-by-answer analysis). Prompt: >50% / no worse than -1, at most 3 recipes of 1-3 learning steps as one clear pathway.
  Transfer + AIME results go in a follow-up message when they land.
- 12 Nov ~11:00, research (Claude agent): one online RL run with reward "correct AND length <= cap" (cap = generation cap).
  Proven at 1.5B full FT on long math traces (ThinkPrune 2504.01296: 10.4k -> 4.7k tokens at +0.1; LASER truncation 2505.15612:
  -71% at -1.6; DLER 2510.15110: -77% with batch-wise advantage normalisation). Unproven at 0.6B + LoRA + short mixed tasks; no
  IFEval/BFCL data; small models compress less (2601.02972). LoRA RL: lr ~10x full FT (~1e-5), all linear layers, small batch
  ("LoRA Without Regret"). Agent's odds for >50% / better than -1 in one run: ~30-45%. Cheapest test: 150 steps, easy prompts
  of all 4 types, per-question cap 0.5x the original's median right length as generation cap, LoRA r16, lr 1e-5 constant, N=8,
  beta 0, batch-normalised advantages, Dev checks at 50/100/150; go on only if >=35% shorter at <=-2 by step 100.
- 12 Nov ~09:40: GPU 2's runtime disconnected during the transfer of rpK-40 (Drive log stops at chunk ~3, 60% of 2,400). The
  original's transfer answers are complete on Drive. plan_line_a: every finished line commented out so a fresh runtime only
  replays the transfer of rpK-40 / rpL-40 / rpL-60, then holds. Waiting for the user to reconnect.
- 12 Nov ~10:30: consult14 crashed at turn 179 ($5.29) with HTTP 400: a code run printed nothing and the empty user message is
  refused by OpenRouter. Fixed the message, added a guard in astra_harness.py (empty results become "(the code ran and printed
  nothing)"), resumed from its saved conversation; running again.
- Charts: new design canvas "ThinkingCap Frontier Canvas" (interactive explorer + full tests by type + loss-by-type heat map);
  rpM-60 canvas gained an interactive board comparing any two of 10 models (original, thinking off, rpE-80, rpK-40/60,
  rpL-40/60/80, rpM-20/60), full-test records of rpM-20, rpL-40, rpL-80 fetched from Drive.
- 12 Nov ~11:20: GPU 2 reconnected; its restarted transfer line found neither data/transfer.jsonl nor the original's answers
  on Drive (the Drive mount was not visible yet at start) and began regenerating the original's answers (~5.5 h). Stopped;
  the line now uses --tr-wait-base (answers rpK-40 / rpL-40 / rpL-60 first, then restores the original's answers from Drive).
- 12 Nov ~11:25: **consult14 (Gemini 4 Argon, $10.91) report: reviews/consult14_clean/REPORT.md.** Forced into its final
  report at turn 362 (it never wrote notes and re-read the pack after 6 context restarts). Three clean recipes:
  (1, run first) 2-step DPO: step 1 from the original = prefix-cut "stop when sure" pairs 35% + shortest-vs-longest correct
  (thinking only, response >= 0.95x median) 50% + anti-loop 15%, 3,000 pairs, LoRA r32, lr 5e-5, beta 0.1, 100 steps,
  target 56-61% Dev; step 2 on fresh step-1 answers = length-matched right-beats-wrong on 0<p<=0.75 (50%), floor guards (20%),
  anti-loop + easy maintenance (30%), lr 2e-5, beta 0.15, 60 steps. Forecast 53-56 / -0.5 to -0.95. (2) DPO bootstrap to
  ~46% then difficulty-gated GRPO (brevity bonus only when pass rate >= 0.5). (3) prefix-cut SFT -> compression DPO ->
  calibration DPO. Gate: 1,200-prompt pilot (step 1 must hit 56-61% Dev with IFEval responses >= 94%; step 2 +1.8 accuracy).
