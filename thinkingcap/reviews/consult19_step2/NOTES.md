# Consult 19 Working Notes — Recipe 2 Step 2 & Step 3, Path A vs Path B

## 1. Files Read & Key Numbers So Far
- `NEW_RESULTS_13nov_morning.md`:
  - **Path A (`r2s1b`, Recipe 2 Step 1)**: 2,880 pairs, 180 offline DPO steps from `Qwen/Qwen3-0.6B`, constant lr 2e-5 (10 warmup), beta 0.1, rank 16, one pass, preset `r2s1`, `--think-only` all types, shorten ratio 1.40, solved >= 8/10 (0.80), no widened guards.
    - Pairs used: math (396 shorten, 288 be-right, 36 loop = 720); code (504 shorten, 162 be-right, 54 loop = 720); IFEval (288 shorten, 240 be-right, 144 counting, 48 loop = 720); simple (182 shorten, 78 be-right = 260); several (110 shorten, 110 be-right = 220); no-fit (144 shorten, 96 be-right = 240).
    - Median chosen / rejected tokens: math 396/674, code 591/1213, IFEval 359/477, simple 124/190, several 139/210, no-fit 156/237.
    - Dev-512 & Dev-SVAMP progression:
      - s40: Dev-512 34.7 / -0.09 (math 39/-0.4, code 39/-1.1, IFEval 33/-0.1, simple 29/+4.2, several 31/+1.8, no-fit 23/-3.9) | Dev-SVAMP 39.7 / +1.69
      - s80: Dev-512 40.9 / +0.27 (math 48/-0.8, code 42/-1.6, IFEval 37/+0.9, simple 37/+6.9, several 39/0.0, no-fit 31/-2.2) | Dev-SVAMP 43.4 / +1.19
      - s100: Dev-512 45.9 / +0.43 (math 53/-3.6, code 49/-0.8, IFEval 42/+1.0, simple 38/+9.4, several 42/+2.5, no-fit 35/-0.3) | Dev-SVAMP 52.2 / -0.75
      - s120: Dev-512 45.1 / +0.98 (math 50/-2.5, code 47/-1.5, IFEval 44/+2.7, simple 37/+7.3, several 42/+3.6, no-fit 37/+3.3) | Dev-SVAMP 49.0 / -1.25
      - s140: Dev-512 47.8 / +0.73 (math 54/-2.7, code 51/-3.6, IFEval 42/+2.9, simple 42/+8.8, several 46/+3.9, no-fit 38/+4.2) | Dev-SVAMP 51.9 / -2.00
      - s160: Dev-512 50.1 / +1.19 (math 55/-4.1, code 57/-0.9, IFEval 44/+3.6, simple 44/+7.7, several 45/+5.0, no-fit 37/+4.4) | Dev-SVAMP 53.5 / -1.25
      - s180: Dev-512 51.1 / +1.25 (math 59/-2.9, code 58/-1.9, IFEval 41/+3.8, simple 45/+8.3, several 48/+3.9, no-fit 37/+3.6) | Dev-SVAMP 54.7 / -1.12
    - Full Test `r2s1b-s180` (2,759 x 8): **+50.3% saved / -1.7 acc** (math 62 / -4.5, code 54 / -3.1, IFEval 38 / +1.0, simple 42 / -3.0, several 45 / +0.9, no-fit 45 / +3.5).
    - Dev-512 vs Full Test discrepancy on s180: Dev-512 read 51.1 / +1.25 (+2.95 pts too kind on accuracy, +0.8 pts higher on saving). Dev-SVAMP read 54.7 / -1.12 vs Full Test SVAMP 62 / -4.5 (+3.38 pts too kind on math accuracy, -7.3 pts lower on math saving!).
  - **Path B (`od3` arena)**: Online DPO from original, fixed reference, per-type floors, no guards, 6 rounds (r1..r5 completed, r6 adapter missing).
    - Monitor (780 x 8): r1 21.6/+0.21, r2 43.9/-1.81, r3 55.9/-3.38, r4 59.3/-2.50, r5 65.4/-4.87.
    - Dev-512 of od3-r4: 62.2 / -3.57 (math 76 / -9.4), Dev-SVAMP 72.8 / -6.00.
    - Full Test `od3-r3`: **58.4% / -4.01** (math 74 / -8.9, code 64 / -4.2, IFEval 43 / -0.9, simple 42 / -1.3, several 43 / -0.6, no-fit 45 / -4.6).
    - Full Test `od3-r4`: **60.5% / -4.79** (math 76 / -9.4, code 67 / -4.8, IFEval 44 / -2.2, simple 45 / -2.0, several 46 / -1.1, no-fit 48 / -5.4).
- `NEW_RESULTS_13nov.md` & `EXPERIMENT_LOG.md` & `YOUR_PLAN.md` & `YOUR_CONSULT18_PLAN.md` & `YOUR_R2S1_REVIEW.md` & `code/repair_pairs.py` (L1-137).

## 2. Open Questions to Check in Code & Tables
1. Inspect `code/repair_pairs.py` (L138-600) and `code/track.py` to see exact implementations of `r2s2`, `typed3`, `r2pick`, `ptrepair`, `rpM`, `rpE`, and Step 2 / Step 3 pipelines.
2. Check how `rpK`, `rpL`, `rpM` worked starting from `rpE-80`: what were `rpE-80`'s full-test per-type numbers vs `r2s1b-s180`'s full-test per-type numbers?
3. Check if `r2s1b-s160` full test has arrived or if per-answer / per-question data for `r2s1b-s180` and `od3` are in `tables_full/` or `raw_logs/`.
4. Analyze why math (-4.5), code (-3.1), and simple tools (-3.0) dropped on `r2s1b-s180` full test while IFEval (38% saved / +1.0 acc), several (45% / +0.9), and no-fit (45% / +3.5) have room to shorten AND gained accuracy!
5. Formulate exact answers to Questions 1, 2, and 3 respecting all user constraints (constant lr, never shorten and protect the SAME type in one step, avoid over-protecting, fixed-mix rule, no michal-bida data).

## 3. Current Candidate Strategy
- **Q1 (Path A vs Path B)**:
  - Keep Path A (`r2s1b`) and continue to Step 2 — do NOT re-run Step 1 (`r2s1b-s180` is already at +50.3% / -1.7 on full test, very close to `rpE-80`'s 48.2% / -1.6 full test which fed `rpK-40` 51.2/-0.8 and `rpM-60` 54.1/-0.93!).
  - Drop Path B (`od3`) or explain exact root cause why multi-round online DPO collapsed math (-8.9 to -9.4) and no-fit (-4.6 to -5.4) and what would be required if ever re-run.
- **Q2 (Step 2 & Step 3 Plan)**:
  - In `r2s1b-s180`, math is ALREADY at 62% saved (-4.5 acc) and code is at 54% saved (-3.1 acc), whereas IFEval is only at 38% saved (+1.0 acc) and BFCL tools are at 42-45% saved.
  - Therefore, `consult16`'s original `r2s2` mix (which shortens math 40% and code 45% *and* mixes 25%/15% widened guards into math/code in the same step!) violates BOTH the empirical state of `r2s1b` (math is already at 62% saved!) AND the user's rule ("never shorten and protect the SAME type in one step").
  - Conversely, a type-split Step 2 (like `rpM`'s `typed3`, or even cleaner: math & code get 0% shorten and pure repair/verification, while IFEval & tools get shortened) matches the user's rule and the exact state of `r2s1b`! Wait—let's check the exact distinction between Step 2 and Step 3 in Recipe 2 under the user's rule!

- Read `code/repair_pairs.py` L136-636.

## New Full Test Data Received (Turn 17)
- `r2s1b-s160` FULL TEST (2,759 x 8): **+49.0% saved / -1.7 pts** (math 60 / -4.0, code 51 / -3.2, IFEval 40 / 0.0, simple 41 / -2.6, several 43 / +0.6, no-fit 46 / +5.2).
- Compare `r2s1b-s180` FULL TEST: **+50.3% saved / -1.7 pts** (math 62 / -4.5, code 54 / -3.1, IFEval 38 / +1.0, simple 42 / -3.0, several 45 / +0.9, no-fit 45 / +3.5).
- Compare `rpE-80` FULL TEST: **+47.0% saved / -0.9 pts** (math 54 / -1.2, code 46 / -1.0, IFEval 45 / -2.6, simple 37 / +0.1, several 39 / +0.1, no-fit 50 / +3.4).
- Compare `rpM-60` FULL TEST (from `rpE-80`): **+54.1% saved / -0.93 pts** (math 57 / -1.2, code 58 / -0.9, IFEval 50 / -2.3, simple 45 / -0.3, several 47 / -1.6, no-fit 61 / +4.2).
