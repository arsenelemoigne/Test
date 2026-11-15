# Online DPO: all full tests so far (15 Nov), with the cheap estimate each one had

Full test = 2,759 questions x 8 answers, official weighting (4 benchmarks equal, BFCL pooled 400/200/240; saved = 1 - geomean
of token ratios). Per type: saved % / accuracy change (points). Curve: acc = -0.388 + 0.0899 s - 0.0028 s^2.
Goal: > 50% saved at better than -1. User's margin target: > 52% at better than -1.

| model | full saved / acc | above curve | cheap estimate (kind) | math | code | IFEval | simple | several | no-fit |
|---|---|---|---|---|---|---|---|---|---|
| od4-r8 (od4-best) | 51.6 / -1.3 | +1.90 | Dev-512 52.3 / +1.61 | see log | | | | | |
| od5c-r10 | 54.1 / -1.68 | +2.04 | monitor 52.9 / +0.84 | 65 / -4.6 | 54 / -1.3 | 46 / -0.7 | 46 / -1.1 | 48 / -1.1 | 53 / +2.3 |
| **od5c-r15** | **54.0 / -0.56** | **+3.14** | monitor 52.9 / +2.41 | 62 / -2.8 | 54 / -0.9 | 48 / +0.5 | 48 / -1.6 | 50 / -1.4 | 57 / +6.9 |
| **od5c-r16 (od5c-best)** | **54.8 / -0.78** | **+3.09** | Dev-512 55.0 / +2.22 | 61 / -2.8 | 59 / -2.0 | 47 / +0.5 | 49 / -1.1 | 50 / 0.0 | 57 / +5.9 |
| od5d-r10 (v3 rules, plain bf16) | 51.6 / -1.95 | +1.25 | monitor 50.7 / -0.00 | 54 / -3.5 | 52 / -2.3 | 49 / -1.3 | 49 / -2.4 | 50 / -2.2 | 56 / +3.4 |
| od5d-r11 (od5d-best) | 49.5 / -1.21 | +1.59 | Dev-512 49.6 / +1.69 | 50 / -1.8 | 49 / -2.1 | 47 / -0.3 | 50 / -1.7 | 51 / -3.6 | 57 / +3.6 |

Readings (ours, check them):
- od5c is in the goal at TWO consecutive rounds (15 and 16), both above 52%: the end of the run is not a single lucky checkpoint.
  Round 10 had the same saving but math at -4.6; rounds 11-16 repaired math/code accuracy while keeping the saving.
- Cheap estimates over-read accuracy by 1.9-3.0 points (monitor ~2.5-3.0, Dev-512 ~2.9-3.0) and under-read saving by ~1.
- Subset check (r15): random 240-question draws of the full test give saved [51.0, 57.1], accuracy [-2.53, +1.54] (sd 1.26).
- od5d (od5 v3 = rules A/B/C of your consult24 + plain bf16) did worse than od5c (v2 + --bida-precision). Two things changed at
  once (rules AND precision), so we cannot tell which one hurt.
- loss_analysis/od5c3.txt (r10 / r15 / r16 vs the original, with a noise reference): losses are mostly "stopped thinking early
  (<0.5x the original's right median)"; on math 60% of the lost accuracy is on questions the original solved 8/8.
- od5c needed refills: slices 12-16 needed Haiku-written several-tool and hard no-fit questions (see REPRODUCE_od5c.md).
- Running now (GPU 2): od5r = od5c's exact line with only the seeds changed (--rp-dpo-seed 1, --rp-gen-seed 54), 16 rounds,
  then its 2 full tests. Its results are not in this pack.

Full reproduction record of od5c: od5c/REPRODUCE_od5c.md (settings, every round's monitor per type, pairs per kind/type/round,
data refills, biases). Raw round log: od5c/raw_gpu1_log_rounds3to16.txt. Per-type pairs table: od5c/pairs_r3_r7_per_type.md.
Best full-test saving per type at better than -1 across ALL our runs: pertype_best_full.md.
