# New results, 14 Nov 2026 (read this first)

## od4 = your consult20 variant A, first stage, with your 3 tweaks (latch braked types, no-fit out of --right-hold, Dev at
## rounds 3 / 4), extended by the user to 8 rounds (slices 6-8; a 9th test-matched slice did not exist)
From the ORIGINAL, fixed reference (the original), one LoRA (rank 16) continued over rounds, constant lr 1.5e-5, beta 0.1,
round 1 = 20 steps on the original's stored answers, rounds 2-8 = 15 steps each on the previous round's merged model's own
answers (x8) to a new 780-question test-matched slice; preset od3_arena; per-type floors; 7/8-style bar --shorten-min-right
0.85; per-type data-derived saving caps (math 65, code 66, IFEval 60, simple 47, several 50, no-fit 65), no accuracy hold.
A type whose monitor saving reaches its cap gets NO shortening pairs from then on (latched): length-held be-right + loops.

| model | monitor (next slice) overall | math | code | IFEval | simple | several | no-fit | held after |
|---|---|---|---|---|---|---|---|---|
| r1 | 22.7 / +0.58 | 30 / -0.1 | 23 / +1.1 | 17 / -0.1 | 20 / +2.6 | 16 / +0.1 | 25 / +0.7 | none |
| r2 | 41.8 / -2.02 | 55 / -3.8 | 46 / -3.9 | 29 / -0.1 | 31 / +0.8 | 44 / -1.2 | 41 / -1.8 | none |
| r3 | 52.7 / -2.37 | 68 / -4.5 | 62 / -4.1 | 41 / -1.4 | 37 / +0.9 | 40 / +1.8 | 43 / -1.2 | math |
| r4 | 52.1 / -1.08 | 66 / -5.1 | 63 / -0.2 | 36 / +0.1 | 41 / +2.2 | 37 / +1.9 | 50 / -1.9 | math |
| r5 | 56.4 / -1.66 | 68 / -4.7 | 70 / -4.2 | 42 / +1.3 | 41 / +1.9 | 50 / +2.4 | 51 / -1.8 | math, code, several |
| r6 | 58.6 / -1.69 | 67 / -5.4 | 70 / -2.5 | 50 / +0.9 | 46 / +2.6 | 53 / -1.7 | 48 / -2.0 | same |
| r7 | 56.1 / -0.14 | 58 / -3.4 | 63 / -0.4 | 54 / +3.0 | 47 / +1.9 | 49 / +1.0 | 51 / -3.2 | + simple |
| r8 | (last round, no monitor) | | | | | | | |

Training per round (loss / pairs ordered right at steps 5 / 10 / 15): r2 0.435/85 0.394/88 0.366/91; r3 0.598/68 0.542/79
0.338/91; r4 0.787/60 0.632/74 0.511/79; r5 0.850/72 0.689/59 0.406/84; r6 0.750/56 0.837/50 0.510/82; r7 0.714/56 0.659/61
0.483/78; r8 0.779/60 0.666/60 0.516/78 (the held rounds learn their pairs much less well).

Dev-512 (x8) / Dev-SVAMP-200:
| model | Dev-512 | math | code | IFEval | simple | several | no-fit | Dev-SVAMP |
|---|---|---|---|---|---|---|---|---|
| od4-r3 | 55.8 / -1.80 | 72 / -6.4 | 63 / -1.8 | 38 / -0.9 | 42 / +3.5 | 47 / +3.2 | 36 / -2.2 | 67.6 / -4.44 |
| od4-r4 | 54.5 / -1.86 | 68 / -6.3 | 60 / -2.3 | 41 / -1.1 | 44 / +4.6 | 48 / +1.4 | 38 / -0.8 | 64.7 / -2.12 |
| od4-r5 | 59.8 / -1.17 | 68 / -6.6 | 69 / -2.0 | 49 / +0.3 | 47 / +5.8 | 52 / +3.9 | 41 / -0.3 | 64.7 / -2.50 |
| od4-r6 | 60.1 / -1.18 | 66 / -6.0 | 70 / -1.9 | 52 / +0.1 | 49 / +7.9 | 53 / 0.0 | 45 / -2.5 | 63.4 / -2.06 |
| od4-r8 | 52.3 / +1.61 | 49 / -0.9 | 58 / +2.4 | 52 / +0.1 | 49 / +9.2 | 53 / +1.8 | 44 / +0.3 | 49.8 / +1.62 |

FULL TESTS (2,759 x 8; curve acc = -0.388 + 0.0899 s - 0.0028 s^2):
| model | full test | above the curve |
|---|---|---|
| **od4-r8** | **51.6 / -1.3** | +1.90 |
| **od4-r7** | **55.4 / -2.0** | +2.00 |
| od3-r3 / od3-r4 (no brakes) | 58.4 / -4.01, 60.5 / -4.79 | +0.68, +0.41 |
| recipe 2 step 1 r2s1b-s180 (offline) | 50.3 / -1.7 | +1.25 |
| best offline + repair lineage (rpK-40 / rpM-60 / rpL-60) | | +2.3 / +2.8 / +3.2 |
Dev over-read od4-r8 by -0.7 saving / -2.9 accuracy. The last round (4 of 6 types held) gave back ~8 pts of Dev saving for
~+2.8 accuracy (math 66 -> 49 saved, -6.0 -> -0.9): the stabilising rounds recover accuracy by LENGTHENING.

## Loss analyses (loss_analysis/*.txt; loss_analysis.py in code/): per question vs the original's full test
Net accuracy change per type, then the main cause of the LOST accuracy (on questions the model solves less often than the
original). Noise reference (the original's answers 4-7 vs 0-3): net ~0 per type, its own losses are "full length but wrong",
loops and cut-offs.
| model | math | code | IFEval | simple | several | no-fit |
|---|---|---|---|---|---|---|
| od2 r10 (120 q/type) | -10.9 (stopped early 86%) | -5.4 (stopped early 88%) | -3.5 (stopped early 68%, resp. too short 14%) | -3.7 | -0.7 | -3.9 |
| od3-r4 | -9.4 (stopped early 86%; 60% on q the original solves 8/8) | -4.8 (74%) | -2.2 (59%; resp. too short 9%) | -2.0 | -1.1 | -5.4 |
| od4-r7 | -3.1 (stopped early 62%; 54% on 8/8 q) | -2.1 (65%; loops 11%) | -1.3 (66%) | -0.8 | -1.3 | -3.3 (stopped early 62%) |
| od4-r8 | -1.7 (stopped early 51%; 57% on 8/8 q) | -0.9 (55%; loops 12%) | -1.2 (71%) | -1.2 | -0.9 | -2.0 (53% early, 40% short) |
"stopped early" = thinking < 0.5x the original's median RIGHT thinking on that question. Four loss kinds overall:
(1) stopping too early, mostly on questions the original solves reliably (the big one), (2) repetition loops (code, IFEval),
(3) IFEval responses too short for length rules, (4) broken answer format (tool calls 6-10%, some math).

## The user's goal and the od5 idea (EXPERIMENT_LOG 13 Nov ~23:55 design notes)
The user wants ONE clean online algorithm that shortens and then stops by itself before accuracy falls (no repair), general
(rules from per-question measurements, not type names), later run once heavily (24 answers per question, "within 1 wrong").
od5 idea (not built): per question vs the ORIGINAL's solve rate p0 (stored): shorten only where the current model solves it
as often as the original (within 1 of 8); on a question that lost accuracy a RECOVERY pair (chosen = a right answer about
the original's median right length, rejected = a wrong, usually short one); shares follow the rule (not fixed), no caps;
+ dedicated pairs for loops / IFEval response length / format. Differs from today's 7/8 bar: 7/8 looks only at the current
model, and a damaged question gets at most a compact (shortest-right) be-right pair at a fixed share.

## Path 1 (offline), for context: recipe 2 step 2 (r2s2, your consult19 plan) trained (1,600 pairs incl. Haiku-generated
## extras where our sources ran out) — Dev-512: s20 51.1 / +1.03, s40 55.0 / +2.35, s60 55.1 / +1.54, s80 55.5 / +1.95;
## full tests of s40 and s80 running.
