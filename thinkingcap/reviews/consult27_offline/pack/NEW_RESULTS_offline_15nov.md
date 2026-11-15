# Offline DPO: all full tests (15 Nov), for consult27

Full test = 2,759 questions x 8 answers, official weighting (4 benchmarks equal, BFCL pooled 400/200/240; saved = 1 - geomean
of token ratios). Per type: saved % / accuracy change (points). Curve: acc = -0.388 + 0.0899 s - 0.0028 s^2.
Goal: > 50% saved at better than -1. User's margin target: > 52% at better than -1.

NEW since consult25: r2s3b (your consult25 redo of Path 1's step 3, from r2s2-s80, 799 pairs, plain bf16, lr 8e-6) full tests:
step 30 51.7 / -0.86 and step 40 51.3 / -0.78 (you expected 53.1 / -0.58). Both inside the goal, under the 52% margin; the same
point as r2s3-s30 (51.5 / -0.84): math and simple tool calls lose less than r2s3, IFEval gives back accuracy, the saving did not
rise. Loss analysis: loss_analysis/r2s3b.txt (s30 and s40 vs the original, with a noise reference). Dev-512 had read s40 at 51.9 / +2.39
(accuracy over-read ~3.2).

Chains: rpE / rpI / rpK / rpL / rpM are repair runs on top of earlier models (rpK / rpL / rpM from rpE-80, which itself comes
from clean line A); Path 1 = step 1 r2s1b (from the ORIGINAL) -> step 2 r2s2 (from r2s1b-s180) -> step 3 r2s3 / r2s3b (from
r2s2-s80). The online DPO run od5c (one algorithm from the original, 16 rounds) is shown for reference only.

| model | full saved / acc | above curve | math | code | IFEval | one fn | several | no-fit | what |
|---|---|---|---|---|---|---|---|---|---|
| Per-type merge · mid · FULL TEST | 53.6 / -3.3 | +0.31 | 61 / -6.0 | 62 / -3.9 | 50 / -1.7 | 35 / 0.2 | 38 / 0.4 | 40 / -5.6 | pt-mid (October's 4 per-type adapters merged at .30/.45/.60/.45, no repair) on the full test 2,759 x 8 |
| ca1-soup-10 (clean line A, round 3) | 49.1 / -2.4 | +0.32 | 58 / -4.3 | 53 / -2.2 | 45 / -2.4 | 36 / 0.4 | 38 / 0.6 | 41 / -3.2 | clean line A DPO round 3 · 3 seeds averaged · full test 2,759 x 8 |
| rc1-soup-30 | 57.4 / -3.5 | +0.95 | 67 / -6.6 | 63 / -3.5 | 51 / -2.5 | 41 / -0.1 | 44 / -0.8 | 51 / -4.1 | clean line A: ca1 (DPO round 3) + one 'stop as soon as you're sure' round (calibrated cuts) · 3 seeds averaged · full test 2,759 x 8 |
| Repair rpE · step 80 | 47.0 / -0.9 | +1.45 | 54 / -1.2 | 46 / -1.0 | 45 / -2.6 | 37 / 0.1 | 39 / 0.1 | 50 / 3.4 | repair of rc1-soup-30: equal-length 'be right' pairs (shortest right > shortest wrong, own questions only), chosen answer prefers a re-check of the question, if |
| Repair rpE · step 40 | 50.6 / -1.9 | +1.11 | 59 / -3.0 | 53 / -1.7 | 47 / -2.8 | 39 / 0.4 | 41 / -0.6 | 50 / -0.4 | repair of rc1-soup-30: equal-length 'be right' pairs (shortest right > shortest wrong, own questions only), chosen answer prefers a re-check of the question, if |
| Repair rpE · step 60 | 47.8 / -1.2 | +1.29 | 55 / -1.5 | 48 / -0.7 | 46 / -3.0 | 38 / 0.1 | 39 / -0.2 | 49 / 1.4 | repair of rc1-soup-30: equal-length 'be right' pairs (shortest right > shortest wrong, own questions only), chosen answer prefers a re-check of the question, if |
| Repair rpI · step 120 | 41.9 / 0.1 | +1.64 | 47 / 0.0 | 38 / 0.5 | 41 / -1.5 | 37 / 0.2 | 38 / -0.9 | 50 / 5.8 | rpE-80 + 40 'be right' DPO steps on its own answers to 5,700 new questions · full test 2,759 x 8 |
| Typed repair rpK · step 20 | 49.9 / -1.3 | +1.57 | 58 / -2.6 | 51 / -0.3 | 45 / -3.1 | 40 / 0.3 | 42 / -1.5 | 54 / 3.6 | rpE-80 + 'think less' pairs only on IFEval and tool calls (11 Nov) · full test 2,759 x 8 |
| Typed repair rpK · step 40 | 51.2 / -0.8 | +2.33 | 58 / -2.3 | 52 / -0.1 | 46 / -1.9 | 41 / 0.3 | 45 / -0.8 | 58 / 4.7 | rpE-80 + 'think less' pairs only on IFEval and tool calls (11 Nov) · FIRST FULL TEST IN THE BOX · full test 2,759 x 8 |
| Typed repair rpK · step 60 | 54.4 / -1.4 | +2.38 | 61 / -2.3 | 55 / -1.2 | 49 / -3.0 | 45 / -1.5 | 48 / -1.0 | 62 / 6.2 | rpE-80 + 'think less' pairs only on IFEval and tool calls (11 Nov) · full test 2,759 x 8 |
| Typed repair rpL · step 60 | 59.3 / -1.7 | +3.20 | 64 / -2.7 | 62 / -0.9 | 58 / -3.4 | 47 / -2.0 | 49 / -0.6 | 62 / 4.6 | rpK + IFEval pairs that no longer reward longer answers + anti-loop pairs (11 Nov) · full test 2,759 x 8 |
| Path 1 round 1 · step 200 | 43.4 / -1.5 | +0.26 | 59 / -3.2 | 44 / -3.3 | 23 / 0.4 | 40 / -0.2 | 42 / 1.2 | 45 / 0.4 | from the ORIGINAL Qwen: one shorten-dominant DPO round on its stored answers (10-11 Nov) · full test 2,759 x 8 |
| Typed repair rpL · step 40 | 55.1 / -1.37 | +2.57 | 60 / -1.5 | 57 / -1.8 | 55 / -3.0 | 43 / -0.3 | 45 / -0.6 | 58 / 3.9 | rpK + IFEval pairs that no longer reward longer answers + anti-loop pairs (11 Nov) · full test 2,759 x 8 |
| Typed repair rpM · step 60 | 54.1 / -0.93 | +2.79 | 57 / -1.2 | 58 / -0.9 | 50 / -2.3 | 45 / -0.3 | 47 / -1.6 | 61 / 4.2 | rpE-80 + typed repair with math protected (wide guards, length-matched be-right) and IFEval counting pairs only on true length rules (11 Nov) · IN THE BOX, best |
| Typed repair rpL · step 80 | 62.8 / -2.06 | +3.73 | 66 / -2.9 | 66 / -1.8 | 62 / -2.6 | 51 / -5.1 | 53 / -2.9 | 66 / 7.7 | rpK + IFEval pairs that no longer reward longer answers + anti-loop pairs (11 Nov) · full test 2,759 x 8 |
| Typed repair rpM · step 20 | 50.1 / -0.94 | +1.97 | 56 / -1.5 | 52 / -0.7 | 47 / -2.6 | 39 / 0.3 | 42 / -0.6 | 54 / 3.5 | rpE-80 + typed repair with math protected (wide guards, length-matched be-right) and IFEval counting pairs only on true length rules (11 Nov) · in the box (just |
| Typed repair rpM · step 40 | 52.9 / -1.1 | +2.37 | 58 / -1.5 | 56 / -1.7 | 50 / -2.5 | 41 / 0.3 | 44 / 0.0 | 57 / 4.1 | rpE-80 + typed repair with math protected · full test 2,759 x 8 |
| Path 1 · recipe 2 step 1 · step 160 | 49.0 / -1.7 | +1.01 | 60 / -4.0 | 51 / -3.2 | 40 / 0.0 | 41 / -2.6 | 43 / 0.6 | 46 / 5.2 | Offline DPO from the original (2,880 pairs, test-matched questions, constant 2e-5) · full test 2,759 x 8 · next: shorten IFEval + tools, then repair |
| Path 1 · recipe 2 step 1 · step 180 | 50.3 / -1.7 | +1.25 | 62 / -4.5 | 54 / -3.1 | 38 / 1.0 | 42 / -3.0 | 45 / 0.9 | 45 / 3.5 | Offline DPO from the original (2,880 pairs, test-matched questions, constant 2e-5) · full test 2,759 x 8 · starting point of step 2 |
| Path 1 · recipe 2 step 2 · step 40 | 52.9 / -1.74 | +1.73 |  |  |  |  |  |  | Step 2 from step 1 (r2s1b-180): 1,600 test-matched pairs, shorten IFEval + tool calls only, constant 1e-5 · full test 2,759 x 8 (per-type split not available) |
| Path 1 · recipe 2 step 2 · step 80 | 53.7 / -1.27 | +2.36 |  |  |  |  |  |  | Step 2 from step 1 (r2s1b-180): 1,600 test-matched pairs, shorten IFEval + tool calls only, constant 1e-5 · full test 2,759 x 8 (per-type split not available) |
| Path 1 · recipe 2 step 3 · step 20 | 52.7 / -0.99 | +2.44 | 59 / -2.86 | 53 / -1.92 | 48 / 0.88 | 48 / -3.5 | 51 / -0.19 | 52 / 5.73 | Step 3 = repair from step 2 (r2s2-80): 633 'be right' / anti-loop / format pairs on its own answers, no shortening, constant 8e-6 · full test 2,759 x 8 (exact a |
| Path 1 · recipe 2 step 3 · step 30 | 51.5 / -0.84 | +2.34 | 57 / -2.8 | 52 / -1.62 | 47 / 1.39 | 48 / -3.78 | 50 / -0.56 | 51 / 5.62 | Step 3 = repair from step 2 (r2s2-80): 633 'be right' / anti-loop / format pairs on its own answers, no shortening, constant 8e-6 · full test 2,759 x 8 · INSIDE |
| Path 2 · online DPO od5c · round 15 | 54.0 / -0.56 | +3.14 | 62 / -2.8 | 54 / -0.9 | 48 / 0.5 | 48 / -1.6 | 50 / -1.4 | 57 / 6.9 | Same od5c run, round 15 (checks round 16 was not luck) · full test 2,759 x 8 · INSIDE THE GOAL and above the 52% margin (its free monitor had read 52.9 / +2.41) |
| Path 2 · online DPO od5c · round 16 (final) | 54.8 / -0.78 | +3.09 | 61 / -2.8 | 59 / -2.0 | 47 / 0.5 | 49 / -1.1 | 50 / 0.0 | 57 / 5.9 | ONE online DPO run from the original (od5 v2: per-question rule vs the original, repair pairs, per-type brake), 16 rounds of its own answers on test-matched sli |
| Path 1 · step 3b (redo) · step 30 | 51.7 / -0.86 | +2.36 | 57 / -2.8 | 52 / -1.8 | 47 / 1.2 | 49 / -3.4 | 51 / -0.6 | 52 / 6.1 | Step 3b = consult25's redo of the repair from r2s2-80 (799 pairs, plain bf16, lr 8e-6), step 30 · full test 2,759 x 8 · inside the goal (same point as step 40) |
| Path 1 · step 3b (redo) · step 40 | 51.3 / -0.78 | +2.36 | 55 / -2.1 | 51 / -1.6 | 48 / 0.3 | 49 / -2.9 | 51 / -0.2 | 52 / 6.0 | Step 3b = consult25's redo of the repair from r2s2-80: 799 pairs (math / code / tools pure repair with guards anchored to the original's thinking, IFEval + no-f |
