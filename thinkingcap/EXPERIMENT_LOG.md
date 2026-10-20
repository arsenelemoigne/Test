# Experiment log (Qwen3-0.6B, shorter thinking)

## Morning summary (20 Oct 2026)

**Best result:** ablation A, plain RL with a "short and correct" reward, 20 steps from the raw
model on four training subjects. It uses **19.6% fewer tokens with no accuracy loss (+0.3 points)**
on the 240-question benchmark subset. Adapter: `adapters/ablA` on the Colab VM.

**What the night showed, each idea tested on its own against the raw model:**

| idea | tokens saved | accuracy | verdict |
|---|---|---|---|
| A: RL, short-and-correct reward | +19.6% | +0.3 | works best |
| B: A + TokenProbe (anchor key words, push filler) | +6.1% | -0.3 | hurts: shortening stays on the training subjects |
| C: A + higher lr with decay and restarts, 2 updates per step | lost (see incident) | | at step 5 held-out 58.6% / 846 tokens vs A's 53.1% / 1,031: learns ~2x faster |
| D: C + AdaptThink (learn to skip thinking) | lost (see incident) | | not measured |
| steering vector alone (x2 / x4) | ~0% / some | -4.0 / -15.1 | does not transfer to the benchmarks |
| mixing loop 1 and loop 3 adapters (25% / 50%) | +15% / +20% | +0.5 / -2.8 | a dial, not a method |

**Main insight:** making the model think shorter on maths and quiz questions is easy (held-out
answers 28% shorter and 15 points more accurate). Carrying that over to code, function calling
and instructions is the hard part, and anything that anchors the model to its old behaviour
(TokenProbe's KL term, steering learned on maths) makes the carry-over worse. Code (MBPP+)
barely shortens in any run.

**Next:** train with questions that look like the test but are not the test: coding questions
with unit tests (`code_tasks.py`, HumanEval, ready) and possibly instruction-following data
(open question for you: RLVR-IFeval is a separate dataset but built to mimic IFEval's
constraint types; worth deciding whether that counts as too close to the benchmark).

**Incident:** around 4:30 the browser pane closed and reopening Colab detached the running
session. Ablation B finished; the first try at C was lost and restarted on top of A. While
recovering, an empty T4 runtime was briefly allocated and deleted. Compute used overnight is
about 3 units an hour while two sessions run (this one and the MVT/fading one); about 78 units
were left at 5:20. The browser pane closed again around 6:30; by 9:10 Colab had reclaimed
the L4 VM, so C and D's results **and every adapter on it, including the best one (A)**, are
gone. Only what is in this log survives. Lesson: copy adapters off the VM (Google Drive mount, or
a download) as soon as each run finishes. Rebuilding A takes ~1h45 on an L4 (seed 0 reproduces
the same questions).


## Findings so far (updated as results come in)

1. **Imitation (SFT on cut traces) teaches underthinking.** +39-60% shorter but -6 to -8 points;
   SVAMP -24. The model learns "stop early", not "stop when done".
2. **RL keeps accuracy.** Rewarding short *correct* answers never pays a wrong one, so accuracy
   holds (loop 1: -1.2 points). On its home subjects it is even more accurate than the raw
   model (held-out 58.8% vs 45.7%), because the raw model often rambles past the token cap.
3. **Transfer is the bottleneck, not overfitting.** Training answers got 44% shorter, test answers
   6%. Each question is seen once, so it is not memorisation: training is maths and multiple
   choice, the test adds code, function calling and instructions. Code (MBPP+) never got shorter.
4. **More pressure buys length with accuracy.** Loop 3 (twice the length weight) reached +24% but
   -6 points, losses concentrated on code and SVAMP.
5. **Length vs accuracy is a dial.** Mixing two adapters' weight changes interpolates between them:
   25% loop 3 gives +15% at +0.5 points, 50% gives +20% at -2.8, 75% gives +20% at -7.7.
6. **A steering vector works a little, for free.** Pushing the residual stream along the
   concise-minus-verbose direction (one bias, no training) gives up to -9% tokens with higher
   accuracy at x2 on maths and quiz questions; at x4 accuracy drops, at x8 the model breaks.
7. **...but steering does not transfer.** On the benchmarks, steer x2 saves ~nothing and loses
   4 points (code -11, SVAMP -9, code replies longer). A direction learned on maths and quiz
   answers is not "concise" in general. Same root cause as finding 3. At x4: -15 points
   (MBPP+ 57.5% -> 27.5%). Steering is a dead end for this benchmark mix.
8. **Plain RL, run longer on more subjects, is the best so far.** Ablation A (length reward only,
   20 steps, GSM8K + Skywork + ARC + CSQA, from the raw model): +19.6% tokens saved at +0.3 points.
   Held-out guard: 45.3% -> 60.2% accuracy, 1,065 -> 769 tokens. SVAMP -30% tokens at equal
   accuracy, BFCL -12 to -22% with higher accuracy; code still barely shorter (-3%).
9. **TokenProbe hurts transfer.** B = A + TokenProbe-KL, otherwise identical: on the held-out
   maths/quiz check B matches A (60.5% / 796 tokens vs 60.2% / 769), but on the benchmarks it
   saves only 6% (A: 20%). The KL anchor to the base model keeps behaviour near the original
   outside the training subjects, so the shortening stays local. Per benchmark B: BFCL +5 to +7%,
   IFEval +14%, MBPP+ -8% (longer), SVAMP +11% / -6 pp.
10. **Stability tricks double the shortening, and overshoot.** C (A + lr 4e-5 cosine with restarts,
   2 updates per step): +40% tokens saved, the first run that shortens code (+31%), but -6.9 points,
   concentrated where reasoning matters (MBPP+ -13.8, SVAMP -10). Same direction as loop 3: more
   pressure buys length at the cost of code and maths.
11. **The other track (branch mvt-fading) agrees: keep maths whole.** Its winner (SFT on traces cut
   by the Marginal Value Theorem on easy multiple-choice questions only, maths traces untouched and
   shown twice) gives +37.1% at -0.6 pp with 8 samples per question; MBPP+ 65% shorter at -2.2.
   Every run there that shortened maths lost 6-8 points (maths -21 to -24). Long C is changed
   accordingly: maths and code are protected, only ARC/CSQA carry length pressure.
12. **Protecting code and maths as done here does not help.** P1 (code protected) is A again
   (+18.7%, -0.7 pp) with code no better (MBPP+ longer, -3.8). P2 (code and maths protected) saves
   less (+12.6%) and SVAMP still loses 7.5 points, so the SVAMP loss is not caused by the GSM8K
   length pressure; the shortening learned on quiz questions reaches SVAMP anyway. The code
   protection is weak because the 2,048-token training cap truncates most HumanEval replies
   (protected accuracy 14-30%, ~1,700 tokens), so the correctness reward rarely fires. Code needs
   a larger cap (the evaluation allows 8,192 on MBPP+).
13. **Next lever: training data that looks like the test.** `code_tasks.py` adds HumanEval (164
   problems with unit tests, not an evaluation benchmark) with a reward that runs the tests, in
   MBPP+'s prompt format. Ready to run as experiment E after the ablation.


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
| **Ablation A** | RL from raw, length reward only, 20 steps, 4 sources | **+19.6%** | **+0.3 pp** | **yes (best)** |
| Round 1 (T4) | SFT on pruned traces | +39.1% | -6.3 pp | no |
| Round 2 | SFT on pruned traces, margin + balanced | +59.6% | -8.0 pp | no |
| RL loop 1 | RL, TokenProbe-KL + ShorterBetter, alpha 0.5 | +5.7% | -1.2 pp | yes |
| RL loop 2 | same as loop 1, continued | stopped at step 4 (lengths not falling) | - | - |
| RL loop 3 | RL continued from loop 1, stronger pressure | +23.6% | -6.0 pp | no |
| **Blend 25/75** | 25% loop 3 + 75% loop 1 | **+15.4%** | **+0.5 pp** | **yes (safest)** |
| **Blend 50/50** | loop 3 and loop 1 add-ons mixed exactly (no training) | **+19.5%** | **-2.8 pp** | **yes (best so far)** |
| Blend 75/25 | 75% loop 3 + 25% loop 1 | +19.6% | -7.7 pp | no |

Accuracy ranges (95%) are about +/-4 pp on this subset, so differences under ~3 pp are noise.
**The baseline itself moves.** Re-running the raw model on a new L4 VM (20 Oct, same questions,
same eval seed) gave SVAMP 940 tokens instead of 778 and IFEval 734 instead of 710 (accuracy:
SVAMP 81.3% both times, IFEval 53.8% vs 52.5%); vLLM sampling is not bit-reproducible across
machines. Only compare a candidate against the base measured on the same VM.

## Queued on 20 Oct (results copied to Drive: MyDrive/thinkingcap_runs)

| run | what | why |
|---|---|---|
| A (rebuild) | same as ablation A, seed 0 | **done: +22.1% [14.7, 29.0], -1.2 pp [-5.4, +3.1]**. BFCL irr/mult/simple +4.7/+19.3/+11.5% (+1.3/+3.8/0 pp); IFEval +28.5% (+1.3); MBPP+ +0.1% (0); SVAMP +41.4% (-7.5). Guard 49.6% -> 59.0%, 1,073 -> 774 tok. Saved to Drive |
| C | A + lr 4e-5 cosine to 4e-6, 2 restarts, 2 updates per step | **done: +40.2% [32.8, 47.1], -6.9 pp [-11.5, -2.6]: over the limit.** BFCL irr/mult/simple +22/+32/+31% (-3.8/-2.5/-5.0 pp); IFEval +38% (0); MBPP+ +31% (-13.8); SVAMP +59% (-10.0). Guard 46.5% -> 58.2%, 1,079 -> 687 tok. Saved to Drive |
| steer mix | steering vector from 640 questions incl. coding (HumanEval, unit tests) | **done: 270 pairs (concise 436 / verbose 1,008 tok). Held-out x1/x2/x3: 51.2% / 1,008, 50.0% / 960, 46.1% / 935. Benchmarks x2: +5.9%, -3.1 pp (MBPP+ -18.6% longer, -10 pp). Still no transfer.** |
| P1 | A's recipe + HumanEval in the mix as *protected* (correctness only, KL 0.01) | **done: +18.7% [11.3, 26.1], -0.7 pp [-4.8, +3.2]. MBPP+ -9.5% (longer), -3.8 pp; SVAMP +39.6% / -2.5; IFEval +22.9% / +3.8; BFCL +10 to +19%. Protected HumanEval solved only 14% at step 20 (1,807 tok, most replies hit the 2,048 cap)** |
| P2 | as P1, maths (GSM8K, Skywork) also protected | **done: +12.6% [5.6, 19.2], -2.1 pp [-5.7, +1.5]. SVAMP +24.3% / -7.5; MBPP+ -10.9% / -2.5; IFEval +25.6% / +2.5; BFCL +2 to +10%** |
| C step 10, C/A 50-50 blend | C's halfway checkpoint, and an exact blend of C and A | salvage C's extra shortening inside the limit (queued after P2) |
| **C long** | C's recipe for 60 steps; HumanEval in the mix (124 train / 40 held out), code protected (correctness only, KL 0.01) with a **4,096-token cap for code**; maths length weight 0.25 (P2 showed full maths protection does not save SVAMP); guard on held-out maths/quiz/code (40 HumanEval held out) that rolls back and softens on a >3 pt drop; benchmark tests at steps 20/40/60 | papers (Art of Efficient Reasoning) see length drop first and accuracy recover with more training; 20 steps may stop in phase 1 |

## Ablation: what each ingredient adds (running overnight)

Every run starts from the raw model with the same seed (same questions in the same order),
20 steps x 32 questions x 8 samples, alpha 0.5, LoRA r16, and is scored on the same 240-question
subset. Each run adds one ingredient to the previous one, so the difference between two
neighbours is that ingredient's marginal effect. The guard only measures (no rollbacks).

| run | ingredients | tokens saved | accuracy change |
|---|---|---|---|
| S x2 | steering vector only (no training), strength 2 | ~0% (code and BFCL longer) | -4.0 pp [-8.0, -0.1] |
| S x4 | steering vector only, strength 4 | (code shorter, BFCL longer) | -15.1 pp [-19.9, -10.4] |
| **A** | RL with the ShorterBetter length reward, lr 2e-5 | **+19.6% [8.8, 29.5]** | **+0.3 pp [-3.0, +3.7]** |
| B | A + TokenProbe-KL (core tokens anchored, filler pushed down over budget) | +6.1% [-6.5, 17.8] | -0.3 pp [-4.6, +4.0] |
| C | **A** + lr 4e-5 cosine to 4e-6 with 2 restarts, 2 optimizer updates per step (built on A, since B < A) | lost (VM reclaimed); held-out at step 5: 58.6% / 846 tok | |
| D | C + AdaptThink (2 of 8 samples start with an empty thinking block) | lost | |

Loop 4 (stacked on loop 1, lr 5e-5, entropy bonus 1e-3) was stopped after 3 steps for this:
its entropy rose 0.80 -> 1.01 -> 1.16, a sign the entropy bonus with that lr destabilises it.

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

| step | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| accuracy % | 55 | 49 | 53 | 52 | 56 | 60 | 57 | 46 | 47 | 56 | 64 | 57 | 71 | 54 | 60 | 50 | 67 | 42 | 48 | - |
| mean tokens | 977 | 830 | 859 | 817 | 809 | 879 | 863 | 898 | 853 | 805 | 729 | 820 | 672 | 811 | 775 | 743 | 736 | 845 | 964 | - |

## Held-out check: same questions for every model

128 questions from the validation split (32 per source, never trained on), 4 answers each,
2,048-token cap. This compares the models on their home subjects with identical questions.

| model | all | ARC | CSQA | GSM8K | Skywork |
|---|---|---|---|---|---|
| raw | 45.7% / 1,092 tok | 85.2% / 495 | 57.0% / 502 | 38.3% / 1,475 | 2.3% / 1,898 |
| RL loop 1 | **58.8% / 859** | 85.2% / 425 | 57.8% / 349 | 78.1% / 854 | 14.1% / 1,807 |
| RL loop 3 | 55.3% / 772 | 77.3% / 319 | 57.0% / 235 | 63.3% / 855 | 23.4% / 1,681 |

On these subjects RL makes the model both shorter and more accurate: the raw model often runs
past the 2,048-token cap on math and never answers. The weak point is transfer to the test
benchmarks (code, function calling, instructions), which training does not cover.

## Steering vector (no training)

Concise-minus-verbose direction from 97 pairs of correct replies to the same question
(concise 369 tokens, verbose 913 on average), residual stream after layer 14 of 28, stored
as a v_proj bias. Same held-out check as above (128 questions x 4).

| model | all | ARC | CSQA | GSM8K | Skywork |
|---|---|---|---|---|---|
| raw | 45.7% / 1,092 | 85.2% / 495 | 57.0% / 502 | 38.3% / 1,475 | 2.3% / 1,898 |
| raw + steer x0.5 | 45.7% / 1,069 | 82.0% / 486 | 52.3% / 520 | 44.5% / 1,369 | 3.9% / 1,902 |
| raw + steer x1 | 48.6% / 1,029 | 83.6% / 473 | 61.7% / 441 | 45.3% / 1,303 | 3.9% / 1,899 |
| raw + steer x2 | 52.0% / 991 | 82.8% / 493 | 59.4% / 462 | 60.2% / 1,143 | 5.5% / 1,867 |
| raw + steer x4 | 43.4% / 959 | 75.8% / 424 | 55.5% / 406 | 34.4% / 1,112 | 7.8% / 1,894 |
| raw + steer x8 | 22.9% / 1,116 | 46.9% / 474 | 42.2% / 434 | 1.6% / 1,594 | 0.8% / 1,962 |

Up to x2 the push makes replies shorter and more accurate (fewer replies cut off at the cap); at x4 it
keeps shortening but accuracy falls below the raw model; at x8 the model breaks (maths 1.6%, and
replies get longer again). x2 is the sweet spot on this check.

## Evaluation details

**Ablation A** (accuracy / tokens; raw in brackets): BFCL irrelevance 83.8% / 180 (85.0 / 205);
BFCL multiple 82.5% / 173 (76.3 / 223); BFCL simple 85.0% / 158 (82.5 / 190); IFEval 55.0% / 525
(52.5 / 710); MBPP+ 53.8% / 1,655 (57.5 / 1,699); SVAMP 81.3% / 545 (81.3 / 778).
Guard (held-out, 128 x 2): step 5 53.1% / 1,031; step 10 55.9% / 953; step 15 57.0% / 848;
step 20 60.2% / 769 (raw 45.3% / 1,065). Entropy 0.60 -> 0.81.

**Blend 50/50** (accuracy / mean tokens, raw in brackets): BFCL irrelevance 80.0% / 193 (85.0 / 205);
BFCL multiple 75.0% / 187 (76.3 / 223); BFCL simple 77.5% / 157 (82.5 / 190);
IFEval 57.5% / 478 (52.5 / 710); MBPP+ 50.0% / 1,725 (57.5 / 1,699); SVAMP 76.3% / 551 (81.3 / 778).
Accuracy CI [-7.1, +1.2] pp: inside the limit, but only just.

**RL loop 3**: BFCL irrelevance +0.6% / -8.8; BFCL multiple +18.6% / -3.8; BFCL simple
+19.9% / -3.8; IFEval +33.8% / 0.0; MBPP+ +7.7% / -8.8; SVAMP +35.8% / -10.0.

**RL loop 1** (tokens saved / accuracy change): BFCL irrelevance +4.3% / -5.0;
BFCL multiple +12.2% / +8.8; BFCL simple +9.0% / +1.3; IFEval +16.1% / -1.3;
MBPP+ -18.1% / -3.8; SVAMP +12.6% / -1.3.

**Round 2 (SFT)**: BFCL +32-39% / -3.8 to -10.0; IFEval +40.3% / +3.8; MBPP+ +86.4% / -6.3;
SVAMP +48.8% / -23.8.
