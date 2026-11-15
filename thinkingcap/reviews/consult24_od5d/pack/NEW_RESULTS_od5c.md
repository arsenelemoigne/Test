# od5c (your consult23 od5 v2, run from the ORIGINAL Qwen) — read this first (15 Nov, ~01:30)

Settings: plan_line_b.txt (odpo --name od5c): rank 16, constant lr 1.4e-5, beta 0.1, 15 DPO steps per round (round 1: 20 steps,
320 pairs), quotas per round math 58 / code 52 / IFEval 58 / simple 26 / several 22 / no-fit 24 (= 240), mix preset od5 +
--od5-v2, fixed reference = the original, LoRA continued across rounds, brake (--od-caps svamp=62,mbpp=63,ifeval=56,
bfcl_simple_python=46,bfcl_multiple=49,bfcl_irrelevance=56 and --od-acc-hold -2.0, --od-unlatch): a type over its saving cap
or at <= -2.0 accuracy on this round's slice gets NO shortening pairs that round (its quota is filled with recovery, length-held
be-right, loop, format pairs only).

## Round-by-round monitor (model after round k answering slice k+1, 780 test-matched questions x8, vs the original's stored answers)
Headline = official weighting: 4 benchmarks equal (BFCL's three parts pooled), saved = 1 - geometric mean of the 4 token ratios.
| model | overall | math | code | IFEval | simple | several | no-fit | od4 same round |
|---|---|---|---|---|---|---|---|---|
| od5c-r1 | 24.5 / +0.41 | 31 / -0.0 | 29 / +1.6 | 18 / +0.0 | 21 / +0.7 | 17 / -1.6 | 22 / +0.5 | 22.7 / +0.58 |
| od5c-r2 | 36.9 / -1.28 | 46 / -1.9 | 38 / -1.4 | 31 / -0.4 | 26 / -1.7 | 41 / -1.5 | 38 / -0.8 | 41.8 / -2.02 |
| od5c-r3 | 44.2 / -0.82 | 57 / -1.7 | 50 / -3.8 | 36 / -0.3 | 33 / +2.8 | 34 / +4.7 | 37 / +0.4 | 52.7 / -2.37 |
| od5c-r4 | 49.6 / -1.19 | 62 / -5.9 | 59 / +1.7 | 36 / -0.9 | 39 / +0.2 | 40 / +3.8 | 47 / -2.2 | 52.1 / -1.08 |
| od5c-r5 | 47.0 / +0.39 | 57 / -2.4 | 53 / -0.6 | 39 / +1.9 | 35 / +3.2 | 44 / +3.0 | 45 / +1.3 | 56.4 / -1.66 |
| od5c-r6 | 46.9 / -0.33 | 54 / -2.2 | 51 / -2.3 | 43 / +2.1 | 38 / +0.6 | 44 / -0.9 | 40 / +3.5 | 58.6 / -1.69 |
Curve (acc = -0.388 + 0.0899 s - 0.0028 s^2): od5c-r5 +2.74 above, r6 +2.0 (od4's best full test +2.0). od4's monitor
over-read the full test: od4-r7 monitor 56.1 / -0.14 vs FULL 55.4 / -2.0; od4-r8 full 51.6 / -1.3. Round 7 is training now
(rounds 8-9 follow on existing slices, then 7 more slices are built with Haiku-written no-fit questions -> rounds 10-16).

## Pairs per round, per type: shorten / be-right (compact or length-held) / recovery / loop / format, and candidate counts
(candidates shorten / recovery / hold / loop / format / count-rule; "(held)" = braked that round)
| round (pairs) | held | math | code | IFEval | simple | several | no-fit |
|---|---|---|---|---|---|---|---|
| r3 (240) | none | 37/11/6/4/0 (cand 146 / 12 / 27 / 4 / 0 / 0) | 30/9/7/6/0 (cand 63 / 10 / 35 / 22 / 0 / 0) | 24/15/12/7/0 (cand 24 / 15 / 49 / 12 / 0 / 18) | 15/5/4/0/2 (cand 43 / 5 / 12 / 0 / 2 / 0) | 14/4/2/0/2 (cand 34 / 2 / 13 / 0 / 7 / 0) | 18/5/1/0/0 (cand 50 / 1 / 8 / 0 / 0 / 0) |
| r4 (238) | ['mbpp'] | 32/11/10/5/0 (cand 101 / 15 / 19 / 5 / 0 / 0) | 0/31/14/6/0 (cand 58 (held) / 14 / 31 / 24 / 0 / 0) | 27/15/9/7/0 (cand 27 / 19 / 53 / 16 / 0 / 14) | 19/3/1/0/2 (cand 52 / 1 / 4 / 0 / 2 / 0) | 16/4/0/0/2 (cand 38 / 0 / 12 / 0 / 2 / 0) | 16/5/3/0/0 (cand 46 / 3 / 9 / 0 / 0 / 0) |
| r5 (204) | ['bfcl_irrelevance', 'svamp'] | 0/11/24/4/0 (cand 59 (held) / 24 / 11 / 4 / 0 / 0) | 30/9/7/6/0 (cand 58 / 8 / 29 / 11 / 0 / 0) | 19/16/16/7/0 (cand 19 / 16 / 38 / 19 / 0 / 10) | 17/5/2/0/2 (cand 52 / 2 / 11 / 0 / 2 / 0) | 13/4/3/0/2 (cand 28 / 4 / 10 / 0 / 5 / 0) | 0/4/3/0/0 (cand 44 (held) / 3 / 4 / 0 / 0 / 0) |
| r6 (224) | ['svamp'] | 0/18/17/7/0 (cand 97 (held) / 17 / 18 / 7 / 0 / 0) | 30/9/7/6/0 (cand 59 / 8 / 41 / 17 / 0 / 0) | 20/17/14/7/0 (cand 20 / 14 / 41 / 16 / 0 / 12) | 18/5/2/0/1 (cand 56 / 2 / 8 / 0 / 1 / 0) | 15/4/2/0/1 (cand 35 / 2 / 6 / 0 / 1 / 0) | 15/5/4/0/0 (cand 47 / 4 / 12 / 0 / 0 / 0) |
| r7 (213) | ['mbpp', 'svamp'] | 0/17/16/4/0 (cand 118 (held) / 16 / 17 / 4 / 0 / 0) | 0/25/15/6/0 (cand 56 (held) / 15 / 25 / 18 / 0 / 0) | 13/27/11/7/0 (cand 13 / 11 / 49 / 11 / 0 / 7) | 18/5/2/0/1 (cand 47 / 2 / 11 / 0 / 1 / 0) | 14/4/3/0/1 (cand 32 / 3 / 7 / 0 / 1 / 0) | 17/5/1/1/0 (cand 50 / 1 / 10 / 1 / 0 / 0) |
Rounds 1-2: r1 320 = shorten 209, be-right 62, loop 34, format 15, recovery 0; r2 240 = shorten 137, be-right 50, recovery 29,
loop 21, format 3. Training loss (steps 5/10/15): r1 0.653/0.538/0.444/0.363 (20 steps, 94% ordered right), r2 ->0.44 (81%),
r3 0.520/0.560/0.490, r4 0.578/0.624/0.552, r5 0.626/0.622/0.523, r6 0.665/0.591/0.587, r7 0.622/0.616/0.584.

## Our reading (please check it)
1. Saving plateaus at ~47% from round 5: the brake holds math (r5-r7) and code (r4, r7); a held type cannot fill its quota
   with protective pairs alone, so 16-36 slots per round stay EMPTY (204 / 224 / 213 pairs trained in rounds 5 / 6 / 7)
   and are not given to another type; "think less" pairs fell from 137 (r2) to 62 (r7).
2. IFEval is CANDIDATE-limited (every shorten candidate is used: 24, 27, 19, 20, 13): about 60% of its questions carry a
   counting / at-least rule (no shortening there), the rest needs p1 >= 0.875 / p1 >= max(0.5, p0 - 0.125) vs the original.
   The tool types are QUOTA-limited (simple 47-56 candidates for ~18 pairs, no-fit 44-50 for ~17, several 28-38 for ~14).
3. With the official weighting, IFEval = 1/4 of the headline: IFEval 43 -> 56% alone would move round 6's 46.9 to ~50.6;
   all four healthy types at their caps with math / code holding ~57 / 53 -> ~54%.

## Our od5d proposal (for a NEW run from the original; od5c continues unchanged as the reference)
A. Budget follows health: a held type keeps its protective pairs (recovery, length-held be-right up to 20%, loop, format);
   its remaining slots go to SHORTEN pairs on the types that are not held and whose slice accuracy is >= the original's,
   in proportion to their unused shorten candidates (no healthy type -> no transfer).
B. On a healthy type (slice accuracy >= the original's), a question is shorten-eligible within one answer in 8 of the
   original: p0 = 1 -> p1 >= 0.75; p0 >= 0.5 -> p1 >= max(0.5, p0 - 0.25); else p1 >= max(0.25, p0 - 0.125).
C. On a healthy IFEval, counting / at-least questions may get a shorten pair only if the chosen answer's RESPONSE is at
   least as long as the rejected answer's (think less, write no less) and >= 0.92x the original's median right response.
