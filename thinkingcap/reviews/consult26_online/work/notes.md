# Consult 26 Notes: Online DPO Reliable >52% at >-1

## Core Facts Loaded
- `od5c` (v2 rules, `--bida-precision` = fp32 master + fp16 autocast, 16 rounds):
  - Round 10: 54.1% saved / -1.68 acc (+2.04 above curve). Math 65 / -4.6, Code 54 / -1.3, IFEval 46 / -0.7, Simple 46 / -1.1, Several 48 / -1.1, No-fit 53 / +2.3.
  - Round 15: 54.0% saved / -0.56 acc (+3.14 above curve). Math 62 / -2.8, Code 54 / -0.9, IFEval 48 / +0.5, Simple 48 / -1.6, Several 50 / -1.4, No-fit 57 / +6.9.
  - Round 16 (`od5c-best`): 54.8% saved / -0.78 acc (+3.09 above curve). Math 61 / -2.8, Code 59 / -2.0, IFEval 47 / +0.5, Simple 49 / -1.1, Several 50 / 0.0, No-fit 57 / +5.9.
- `od5d` (v3 rules A/B/C, plain bf16, 11 rounds? Let's check EXPERIMENT_LOG.md and raw logs):
  - Round 10: 51.6% saved / -1.95 acc (+1.25 above curve). Math 54 / -3.5, Code 52 / -2.3, IFEval 49 / -1.3, Simple 49 / -2.4, Several 50 / -2.2, No-fit 56 / +3.4.
  - Round 11 (`od5d-best`): 49.5% saved / -1.21 acc (+1.59 above curve). Math 50 / -1.8, Code 49 / -2.1, IFEval 47 / -0.3, Simple 50 / -1.7, Several 51 / -3.6, No-fit 57 / +3.6.
- Cheap estimate biases:
  - Monitor over-reads accuracy by ~2.5-3.0 pts, under-reads saving by ~1.1-1.2 pts.
  - Dev-512 over-reads accuracy by ~2.9-3.0 pts.

## EXPERIMENT_LOG & od5d Details
- `od5d` only ran 11 rounds because only 11 slices existed when `od5d` started (slices 12-16 were built later on GPU 1 for `od5c`).
- `od5d` monitor trajectory (rounds 1 to 10):
  - r1: 22.0 / +0.60 (math 29/-1.0, code 23/+1.6, IFEval 16/+0.5, simple 21/+1.6, several 16/-1.2, no-fit 24/+3.0)
  - r2: 34.6 / -1.18 (math 43/-1.8, code 35/-1.9, IFEval 28/-0.6, simple 27/-0.4, several 39/-3.1, no-fit 36/+1.8)
  - r3: 39.2 / +0.27 (math 49/-0.2, code 44/-0.7, IFEval 30/-1.0, simple 32/+1.5, several 31/+3.9, no-fit 36/+4.8)
  - r4: 43.3 / -0.13 (math 50/-2.7, code 47/+0.7, IFEval 37/+0.6, simple 36/+0.6, several 37/+4.0, no-fit 45/-1.3)
  - r5: 44.2 / +0.39 (math 48/-2.1, code 50/-2.2, IFEval 38/+2.9, simple 37/+3.6, several 46/+2.2, no-fit 46/+2.7)
  - r6: 43.5 / +0.02 (math 43/-1.4, code 46/+0.6, IFEval 42/-0.4, simple 42/+0.9, several 45/-2.1, no-fit 43/+4.9)
  - r7: 45.4 / +0.44 (math 46/-2.8, code 45/-0.0, IFEval 44/+2.9, simple 45/+1.6, several 44/-0.5, no-fit 50/+3.7)
  - r8: 49.2 / -1.10 (math 48/-2.8, code 51/-2.6, IFEval 46/-0.1, simple 49/+1.1, several 53/-1.3, no-fit 56/+3.1)
  - r9: 44.3 / +1.82 (math 38/+0.3, code 47/-0.1, IFEval 45/+3.5, simple 48/+4.8, several 47/+1.3, no-fit 47/+3.7)
  - r10: 50.7 / -0.00 -> FULL TEST 51.6 / -1.95
  - r11 (Dev-512 49.6 / +1.69) -> FULL TEST 49.5 / -1.21
- Supply of test-matchable candidates left AFTER slice 16:
  - All 16 slices `data/od3-r1..r16-questions.jsonl` ALREADY EXIST on Drive!
  - Plus leftover unused test-matchable candidates after slice 16: math 1,195, code 983, IFEval 5,242, simple 2,033, several 1,038, no-fit 3,046 (though remaining no-fit are easier, 91% vs 83%).
