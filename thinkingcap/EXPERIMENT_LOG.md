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
| RL loop 3 | RL continued from loop 1, stronger pressure | +23.6% | -6.0 pp | no |
| **Blend 25/75** | 25% loop 3 + 75% loop 1 | **+15.4%** | **+0.5 pp** | **yes (safest)** |
| **Blend 50/50** | loop 3 and loop 1 add-ons mixed exactly (no training) | **+19.5%** | **-2.8 pp** | **yes (best so far)** |

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

## Evaluation details

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
