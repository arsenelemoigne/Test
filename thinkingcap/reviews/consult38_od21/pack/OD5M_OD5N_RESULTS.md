# od5m / od5n round by round (MONITOR level unless stated). Read with README levels: monitor over-reads accuracy ~2.5-3 pts.

Each MONITOR row j = model of round j-1 answering round j's fresh slice. loss / pairs = last DPO step (15/15) of round j's training.

| run | round | loss (last step) | pairs ordered right | monitor saved | monitor acc | od5f same-round monitor | per type saved/acc (math=svamp, code=mbpp, IFEval, one fn=bfcl_simple, several=bfcl_multiple, no-fit=bfcl_irrelevance) |
|---|---|---|---|---|---|---|---|
| od5n | 2 | 0.4316 | 75% | 27.3 | -0.13 | 28.5 / +0.37 | math 31/-0.6, code 32/0.0, IFEval 24/+0.3, one fn 22/+0.5, several 19/+0.5, no-fit 23/-2.4 |
| od5m | 2 | 0.4937 | 76% | 26.6 | -0.13 | 28.5 / +0.37 | math 32/-1.2, code 29/+0.2, IFEval 23/-0.2, one fn 23/+0.1, several 18/+1.3, no-fit 27/+1.2 |
| od5n | 3 | 0.5548 | 79% | 40.8 | -1.94 | 34.2 / -1.66 | math 49/-3.3, code 47/-4.0, IFEval 32/+0.6, one fn 29/-0.7, several 43/-0.6, no-fit 39/-2.1 |
| od5m | 3 | 0.4732 | 78% | 33.8 | -0.97 | 34.2 / -1.66 | math 42/-1.2, code 32/-1.2, IFEval 29/-1.2, one fn 26/+1.0, several 36/-0.4, no-fit 37/-1.9 |
| od5n | 4 | 0.6353 | 66% | 37.6 | -0.39 | 34.0 / -0.11 | math 40/-0.5, code 47/-3.0, IFEval 33/0.0, one fn 29/+2.3, several 29/+1.0, no-fit 35/+2.2 |
| od5m | 4 | 0.7015 | 65% | 45.7 | -1.19 | 34.0 / -0.11 | math 54/-1.6, code 55/-3.6, IFEval 39/-0.8, one fn 34/+2.2, several 33/+1.8, no-fit 38/-0.7 |
| od5n | 5 | 0.5024 | 78% | 40.3 | -0.77 | 39.4 / -0.26 | math 44/-1.7, code 46/-0.6, IFEval 35/-0.8, one fn 33/+1.2, several 35/-1.7, no-fit 42/-0.5 |
| od5m | 5 | 0.4846 | 79% | 44.7 | -1.21 | 39.4 / -0.26 | math 57/-3.6, code 51/+0.5, IFEval 30/-1.9, one fn 38/+0.6, several 38/+1.9, no-fit 47/-2.1 |
| od5n | 6 | 0.7133 | 56% | 49.4 | -1.47 | 41.9 / -0.65 | math 58/-5.8, code 56/-3.1, IFEval 41/+1.0, one fn 37/+1.9, several 46/+1.8, no-fit 48/+2.3 |
| od5m | 6 | 0.5571 | 70% | 45.8 | -0.14 | 41.9 / -0.65 | math 53/-1.9, code 55/-1.3, IFEval 36/+1.0, one fn 33/+1.7, several 45/+1.8, no-fit 46/+0.9 |
| od5n | 7 | 0.5539 | 68% | 46.9 | -0.66 | 45.2 / -0.91 | math 50/-3.8, code 53/-0.9, IFEval 43/+0.9, one fn 39/+2.6, several 46/-2.1, no-fit 43/+1.6 |
| od5m | 7 | 0.3977 | 85% | 55.5 | -0.60 | 45.2 / -0.91 | math 64/-5.1, code 66/-0.5, IFEval 45/+1.8, one fn 45/+2.4, several 51/-1.9, no-fit 46/+2.4 |
| od5n | 8 | 0.5493 | 71% | 44.1 | +0.26 | 46.1 / +0.30 | math 42/-2.9, code 47/-0.9, IFEval 43/+2.2, one fn 43/+2.8, several 43/+0.4, no-fit 49/+4.3 |
| od5m | 8 | 0.5253 | 78% | 51.9 | +0.39 | 46.1 / +0.30 | math 55/-2.9, code 56/-1.0, IFEval 48/+2.5, one fn 47/+3.8, several 47/-0.3, no-fit 53/+4.2 |
| od5n | 9 | 0.6028 | 68% | 48.5 | -0.54 | 54.1 / +0.46 | math 41/-3.3, code 54/-2.7, IFEval 49/+0.4, one fn 46/+3.5, several 51/+0.2, no-fit 55/+5.9 |
| od5m | 9 | 0.5633 | 74% | 51.6 | -0.22 | 54.1 / +0.46 | math 50/-2.6, code 57/-0.6, IFEval 46/+0.9, one fn 49/-0.1, several 54/-2.8, no-fit 57/+7.3 |
| od5m | 10 | 0.4330 | 80% | 47.5 | +2.37 | 52.7 / +1.67 | math 42/+1.0, code 55/+1.5, IFEval 46/+2.5, one fn 48/+4.3, several 48/+2.5, no-fit 47/+6.5 |
| od5n | 10 | 0.4444 | 80% | 40.1 | +2.25 | 52.7 / +1.67 | math 29/+0.7, code 42/+1.2, IFEval 44/+1.6, one fn 46/+5.8, several 45/+3.4, no-fit 45/+7.0 |
| od5m | 11 | 0.4624 | 81% | 53.0 | -0.22 | 60.6 / -0.37 | math 53/-1.9, code 61/-0.4, IFEval 46/-1.0, one fn 50/+3.6, several 53/-2.5, no-fit 55/+4.2 |
| od5n | 11 | 0.4428 | 78% | 44.5 | +0.69 | 60.6 / -0.37 | math 35/-0.6, code 50/+1.0, IFEval 45/0.0, one fn 46/+2.5, several 48/-0.2, no-fit 52/+4.5 |
| od5m | 12 | 0.6876 | 66% | 57.6 | +0.49 | 55.2 / -0.15 | math 55/-1.1, code 67/-2.6, IFEval 57/+0.5, one fn 51/+4.3, several 53/+3.5, no-fit 54/+7.9 |
| od5n | 12 | 0.5020 | 80% | 49.5 | +1.51 | 55.2 / -0.15 | math 36/-1.4, code 59/-1.6, IFEval 51/+2.6, one fn 51/+6.1, several 52/+3.0, no-fit 54/+9.8 |
| od5m | 13 | 0.4422 | 79% | 53.7 | +1.72 | 48.3 / +1.89 | math 54/-0.2, code 53/+2.0, IFEval 55/+2.5, one fn 49/+3.7, several 56/-3.5, no-fit 56/+5.7 |
| od5n | 13 | 0.4492 | 79% | 50.9 | +0.93 | 48.3 / +1.89 | math 42/-1.1, code 56/+0.5, IFEval 52/+2.1, one fn 50/+3.2, several 56/-3.9, no-fit 57/+5.8 |
| od5m | 14 | 0.4621 | 79% | 57.8 | +0.74 | 51.2 / +0.81 | math 57/-1.0, code 62/+1.3, IFEval 55/-0.2, one fn 53/+2.8, several 57/-1.7, no-fit 64/+7.0 |
| od5n | 14 | 0.5937 | 71% | 56.5 | +0.51 | 51.2 / +0.81 | math 47/-2.9, code 65/+0.7, IFEval 56/+1.3, one fn 54/+4.0, several 58/-4.2, no-fit 65/+7.2 |
| od5m | 15 | 0.4248 | 82% | 57.5 | +1.69 | 55.5 / +1.72 | math 56/-1.9, code 60/+2.1, IFEval 58/+4.2, one fn 53/-0.3, several 58/+1.2, no-fit 60/+8.0 |
| od5n | 15 | 0.3998 | 80% | 50.4 | +2.39 | 55.5 / +1.72 | math 37/+0.1, code 55/+2.1, IFEval 55/+5.3, one fn 51/-1.2, several 56/+0.2, no-fit 59/+8.8 |
| od5m | 16 | 0.5313 | 74% | 59.4 | +2.73 | 56.5 / +2.70 | math 59/-1.5, code 64/+2.7, IFEval 59/+5.4, one fn 48/+3.4, several 61/-1.0, no-fit 63/+10.5 |
| od5n | 16 | 0.5109 | 66% | 49.3 | +2.77 | 56.5 / +2.70 | math 34/+0.7, code 52/+1.2, IFEval 57/+4.9, one fn 46/+3.1, several 59/-1.0, no-fit 61/+11.1 |
| od5m | 17 | 0.5329 | 72% | 53.6 | +2.96 | 53.9 / +1.85 | math 51/+0.5, code 59/+5.1, IFEval 50/+2.2, one fn 50/+4.0, several 56/-3.8, no-fit 61/+10.6 |
| od5n | 17 | 0.6220 | 72% | 51.9 | +2.51 | 53.9 / +1.85 | math 44/+0.4, code 60/+3.1, IFEval 49/+2.1, one fn 50/+3.4, several 56/-2.3, no-fit 62/+11.7 |
| od5m | 18 | 0.4817 | 80% | 51.9 | +2.21 | 54.6 / +1.91 | math 46/-0.9, code 57/+3.5, IFEval 49/+2.9, one fn 52/+1, several 58/-2.8, no-fit 61/+11.2 |
| od5n | 18 | 0.4210 | 78% | 53.2 | +2.62 | 54.6 / +1.91 | math 49/-1.0, code 59/+4.6, IFEval 47/+4.4, one fn 53/+1.2, several 60/-0.3, no-fit 64/+7.0 |
| od5n | 19 | 0.5173 | 68% | 58.1 | +2.59 | 59.0 / +2.37 | math 52/-1.1, code 63/+3.1, IFEval 57/+5.8, one fn 55/+1.2, several 62/-2.4, no-fit 72/+9.0 |
| od5m | 19 | 0.5523 | 74% | 53.9 | +2.30 | 59.0 / +2.37 | math 47/-0.7, code 58/+2.0, IFEval 52/+4.3, one fn 52/+3.5, several 60/-1.3, no-fit 67/+8.1 |
| od5m | 20 | 0.6712 | 70% | 52.8 | +1.86 | 47.8 / +2.90 | math 51/-0.7, code 48/-1.5, IFEval 56/+4.9, one fn 53/+3.5, several 56/+1.4, no-fit 63/+9.8 |
| od5n | 20 | 0.5550 | 80% | 51.8 | +2.38 | 47.8 / +2.90 | math 48/-0.4, code 46/+0.6, IFEval 57/+5.0, one fn 53/+3.0, several 56/+0.5, no-fit 64/+9.4 |
| od5n | 21 | 0.6527 | 66% | 55.3 | +0.80 | 55.1 / +1.99 | math 50/-0.9, code 51/-5.3, IFEval 61/+8.2, one fn 56/-3.4, several 61/-3.7, no-fit 63/+12.7 |
| od5m | 21 | 0.5664 | 71% | 52.4 | +1.75 | 55.1 / +1.99 | math 46/+0.1, code 49/-4.3, IFEval 58/+7.3, one fn 54/+0.8, several 60/+0.3, no-fit 62/+12.0 |
| od5m | 22 | 0.5458 | 69% | 52.2 | +2.22 | 42.1 / +4.38 | math 48/-0.5, code 43/+2.1, IFEval 59/+5.2, one fn 55/+0.2, several 58/-2.1, no-fit 67/+8.4 |
| od5n | 22 | 0.4560 | 79% | 51.9 | +3.45 | 42.1 / +4.38 | math 53/-1.1, code 36/+2.1, IFEval 59/+9.6, one fn 55/+0.3, several 58/+1.7, no-fit 69/+9.5 |

Round 1 (both): step 20/20 loss 0.342, pairs ordered right 95%; monitor r1 = raw Qwen vs itself (0/0).

## DEV-512 x8 (official weighting, 95% paired bootstrap)

- od5m-r18: saved 55.6 [53.1, 58.1] / acc +2.30 [+0.60, +4.08]
- od5n-r18: saved 57.5 [54.8, 59.9] / acc +2.83 [+0.92, +4.70]
- od5m-r22: saved 49.9 [47.2, 52.5] / acc +3.45 [+1.71, +5.04]
- od5n-r22: saved 58.3 [55.8, 60.6] / acc +4.27 [+2.50, +6.05]
- Reference od5f Dev: r16 55.3/+1.72 (full 55.4/-0.57), r17 58.0/+1.83, r18 60.4/+0.88 (full 59.5/-0.80), r19 52.7/+3.49.
- fullpick od5n preselected by monitor (>=55 saved, >=0 acc): r13, r18, r20, r22. More Dev results (fullpick of both runs) are in DEV_FULLPICK_LATEST.md if present.

## FULL TESTS

None yet for od5m / od5n (fullpick gate 60.5 saved and +1.75 Dev acc not passed by the 4 Dev scores above).

## Infrastructure note
The first odpo line (18 rounds) ended with a Dev of <name>-best (= r18); the continuation line skipped rounds 1-18 and trained 19-22 on od3g slices; <name>-best was then r22.