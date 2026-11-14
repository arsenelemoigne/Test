# od5b so far (14 Nov afternoon) — read this first

od5b = your consult21 od5 design, as built (code/repair_pairs.py --mix-preset od5, code/track.py odpo_fixed), with these
changes made while running (all in EXPERIMENT_LOG.md, 14 Nov ~11:00-16:00):
1. IFEval response floor: "original's response = mean tokens - median right thinking" counted the original's runaway answers
   and blocked IFEval shortening (6 shorten pairs of 88 in round 1). Fixed with data/od3-base-resp.json = the original's
   median RIGHT response per question (od3prep --od3-resp); repair_pairs --base-resp.
2. Shares proportional to candidate counts let the NEUTRAL be-right pairs crowd out shortening (IFEval round 1: 129 neutral vs
   33 shorten candidates -> 12 shorten pairs of 88). From round 2: neutral pairs capped at 25% of a type's slots; shortening
   vs recovery still proportional to their candidate counts.
3. User: keep small rounds (15 steps / 240 pairs, lr 1.2e-5, beta 0.1) but MORE rounds, up to 16: 9 test-matched slices
   exist; 458 Haiku-written hard no-fit questions are queued to build slices 10-16 after round 9. No per-round Dev checks.
4. The Colab runtime has generated ~3x slower since ~11:00 (550 vs 1,600-1,800 tokens/s; memory and engine normal):
   rounds take ~1.5-2 h. Not a model effect: implied answer length ~590 tokens = od4-r1's on the same slice.

## Round by round (free monitor = the model after round k answering round k+1's 780 new test-matched questions x8)
| model | overall | math | code | IFEval | simple | several | no-fit |
|---|---|---|---|---|---|---|---|
| od5b-r1 | 11.2 / +0.43 | 19 / -0.5 | 9 / +1.0 | 8 / +0.6 | 13 / +0.5 | 6 / -0.8 | 9 / +1.9 |
| od5b-r2 | 16.4 / -0.52 | 32 / -1.8 | 6 / -0.8 | 11 / +0.2 | 12 / +0.8 | 23 / -1.7 | 20 / +0.6 |
| od4-r1 (same slices) | 22.7 / +0.58 | 30 / -0.1 | 23 / +1.1 | 17 / -0.1 | 20 / +2.6 | 16 / +0.1 | 25 / +0.7 |
| od4-r2 | 41.8 / -2.02 | 55 / -3.8 | 46 / -3.9 | 29 / -0.1 | 31 / +0.8 | 44 / -1.2 | 41 / -1.8 |

Training (loss / pairs ordered right at steps 5 / 10 / 15 or 20): r1 0.682/65 0.622/84 0.550/81 0.575/74 (20 steps);
r2 0.557/74 0.550/75 0.559/69; r3 0.596/72 0.519/72 0.520/71 (od4's rounds ended at ~0.32-0.37 / 91%).

Pairs per round (od5 kinds; "right" = neutral 0.90-1.12x be-right + IFEval length-rule pairs):
| round | math sh/rec/right/loop/fmt | code | IFEval | simple | several | no-fit |
|---|---|---|---|---|---|---|
| 1 (stored original answers, 320) | 52/0/10/10/8 | 25/0/24/7/0 | 12/0/65/11/0 | 24/0/8/0/4 | 15/0/8/2/3 | 16/0/12/4/0 |
| 2 (od5b-r1 answers, 240, cap on) | 38/4/10/7/1 | 22/6/9/5/0 | 10/15/33/8/0 | 18/1/6/1/1 | 12/2/5/0/2 | 14/1/6/3/0 |
| 3 (od5b-r2 answers, 240) | 35/10/7/?/? | 22/6/9/?/? | 12/18/28/?/? | 17/1/6/?/? | 13/1/4/?/? | 9/2/7/?/? |
Round 1 pairs file: od5b/pairs-od5b-r1.jsonl (chosen / rejected tokens 0.59 overall; chosen longer in 22%).
Candidate counts per round are in od5b/od5b_gpu1_log_today.txt (round 3) and EXPERIMENT_LOG.md (rounds 1-2), e.g. round 2:
math shorten 153 / recovery 17 / neutral 10, code 31 / 8 / 11, IFEval 10 / 15 / 29 (+29 loop, 7 length-rule), simple 43 / 3 / 7.
od5b/od4-r2.jsonl: od4's round-1 model's 6,240 answers to slice 2 (the same questions od5b-r1 answered in its round 2).

## Our first reading (check it)
In od4, a question the model solves only sometimes got a "compact" be-right pair (chosen = the SHORTEST right answer, over
the shortest finished wrong one): those pairs ALSO taught "shorter" on code / IFEval, whose questions are mostly mixed. od5
made them neutral (0.90-1.12x), and shortening needs p1 >= max(6/8, p0 - 1/8) (8/8 if p0 = 8/8), so most code and IFEval
questions (original solve rate ~52% / ~59%) are never shorten-eligible: code saved only 6-9% after 2 rounds.

## Path 1 context: recipe 2 step 2 (r2s2) FULL tests: s40 52.9 / -1.74, s80 53.7 / -1.27 (+2.36 above the curve).
## Best online: od4-r8 51.6 / -1.3, od4-r7 55.4 / -2.0 (full tests).
