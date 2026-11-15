# Path 1 step 3 — results (15 Nov, read this first; NEW_RESULTS_path1.md has steps 1-2)

Step 3 (your consult22 design) = a repair from r2s2-s80 (step 2's 80-step model, full test 53.7% / -1.27): ptrepair r2s3,
preset r2s3 (no shortening; be-right pairs length-held 0.90-1.15x around 1.00, anti-loop incl. finished loops, format pairs,
IFEval counting-rule pairs), 633 pairs (user approved: no-fit 25 instead of 32, since few no-fit questions are half-right;
every other type at quota: math 200, code 140, IFEval 120, simple 100, several 48), constant lr 8e-6, beta 0.1, rank 16,
batch 16, 40 steps (1 pass), checkpoints 10/20/30/40. Pairs: chosen longer in 62% (math), 58% (code), 45% (IFEval), 75%
(simple), 54% (several), 72% (no-fit). Questions: data/r2-step3-questions.jsonl (2,134) + extras s3m1 (1,240) + s3m2 (400:
several 100, no-fit 300 incl. Haiku-written no-fit, which turned out easy: the original solves 98%). Training loss 0.692 ->
0.639, ordered right 52-69% (slow, as step 2).

Dev-512 / Dev-SVAMP-200 (Dev over-reads accuracy on this lineage by ~3.2-4.1 pts):
| step | Dev-512 | Dev-SVAMP |
|---|---|---|
| 10 | 53.5 / +1.60 | 52.8 / -0.62 |
| 20 | 52.8 / +2.54 | 50.1 / -0.69 |
| 30 | 52.8 / +2.50 | 49.4 / -0.38 |
| 40 | 51.1 / +2.63 | (not read) |

FULL TESTS (2,759 x 8, official headline: 4 benchmarks equal, BFCL pooled 400/200/240; exact accuracy from the loss analysis):
| model | saved | accuracy | math | code | IFEval | simple | several | no-fit |
|---|---|---|---|---|---|---|---|---|
| r2s2-s80 (step 2, start) | 53.7 | -1.27 | | | | | | |
| **r2s3-s20** | **52.7** | **-0.99** | 59 / -2.86 | 53 / -1.92 | 48 / +0.88 | 48 / -3.50 | 51 / -0.19 | 52 / +5.73 |
| **r2s3-s30** | **51.5** | **-0.84** | 57 / -2.80 | 52 / -1.62 | 47 / +1.39 | 48 / -3.78 | 50 / -0.56 | 51 / +5.62 |
Both are inside the objective (> 50% at better than -1); the user's margin target is > 52% saved at better than -1.
Loss analysis per type and cause: loss_analysis/r2s3.txt (s20 and s30 vs the original, noise reference included).
Remaining losses: math (-2.8, lost questions think ~0.48x the original's right median), simple tool calls (-3.5 to -3.8),
code (-1.6 to -1.9); IFEval +0.9 to +1.4 and no-fit +5.6 are gains. Step 3 gave back ~1-2 pts of saving for +0.3-0.4 accuracy.

For context (online path, not for you to redesign): the online run od5c reads 52.9% / +0.84 on its free monitor at round
10 (780 test-matched questions x8; a full test is running).
