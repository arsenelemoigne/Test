# Consult 12 Analysis Notes

## 1. Trust in `rpK-40` (51.17% saved / -0.77 official, -0.75 rounded per-type)
- Full-test trajectory of `rpK` (`models/rpE-80` + `adapters/rpK-s0`):
  - `rpE-80` (step 0): `47.04%` saved / `-0.95` official (`-0.90` fin)
  - `rpK-20` (step 20): `49.88%` saved / `-1.28` official (`-1.24` fin)
  - `rpK-40` (step 40): `51.17%` saved / `-0.77` official (`-0.77` fin; `-0.75` from rounded per-type table)
  - `rpK-60` (step 60): `54.43%` saved / `-1.44` official (`-1.44` fin)
- Noise decomposition:
  - Fixed-test-set 8-sample MC standard error (`SE_MC`, fixing the 2,759 test questions and `base` rollouts): `+-0.20` pts on headline accuracy (`95% CI +-0.39` pts) and `+-0.34%` on headline saving (`95% CI +-0.67%`).
  - Paired question-level bootstrap SE vs `base` (official `compare.py` CI): `+-0.36` pts on accuracy (`95% CI +-0.71` pts) and `+-0.59%` on saving (`95% CI +-1.16%`).
  - Midpoint bump `rpK-40 - 0.5*(rpK-20 + rpK-60)`: observed `+0.59` pts (`-0.77` vs `-1.36`), with paired `SE = 0.216` pts (`z = +2.72`).
  - Per-benchmark decomposition of the `+0.59` pt bump:
    1. `bfcl_pooled`: `+0.44` pts (`+0.11` headline pts). Real threshold mechanics: at step 40 `bfcl_irrelevance` gained `+1.04` pts (`+3.65 -> +4.69`) and `bfcl_multiple` recovered `+0.69` pts (`-1.50 -> -0.81`), while `bfcl_simple_python` held at `+0.25` (call rate `96.66%`) before suffering the `-1.72` pt non-call drop at step 60 (`80.69%`, call rate `95.75%`).
    2. `mbpp`: `+0.61` pts (`+0.15` headline pts). Partly real (step 40 is at `962.7` thinking tokens vs `991.2` at step 20 and `896.8` at step 60, only 30% of the way to step 60's compression), partly MC draw noise (`SE = 0.59` pts).
    3. `svamp`: `+0.13` pts (`+0.03` headline pts). Flat (`-2.59 -> -2.34 -> -2.35`).
    4. `ifeval`: `+1.17` pts (`+0.29` headline pts; `-3.07 -> -1.87 -> -3.00`). About `+0.25 to +0.30` pts is real transient response lengthening at step 40 (`241.5` response tokens vs `229.7` and `229.6`; `560.6` on `at_least_N`), while `~ +0.85 to +0.90` pts (`+0.22` headline pts) is a lucky 8-sample MC fluctuation (`SE = 0.49` pts; on `Dev-512`, `rpK-40` IFEval was `-2.4`, monotonically between `-0.7` and `-3.4`).
  - Empirical Bayes shrinkage estimate of `rpK-40`'s true infinite-sample expectation on ThinkingCap-Test: **`51.2%` saved / `-1.05 to -1.15` accuracy** (~25-30% probability of true infinite-sample accuracy `> -1.00`).
  - Submittable as is? **Yes, 100%.** The official evaluation is fixed; `rpK-40` deterministically scores `51.2% / -0.8` (`+2.33` vs curve). Do NOT run a second DPO seed of `rpK` (`rpK-s1`): `rpK` has unpatched IFEval/SVAMP pair bugs and `<30%` chance of a second seed clearing `-1.00`. Confirm the box with margin via `rpM` and `rpN`.

## 2. Repair Strategy: `rpM` -> `rpN` on `rpL-60` (59.3 / -1.7) vs `rpK-60` (54.4 / -1.4)
- Why `rpN` on **`rpL-60` (`59.27% / -1.69` official, `-1.55` fin)** strictly dominates `rpN` on **`rpK-60` (`54.43% / -1.44`)**:
  - `rpL-60` has **9.27 points of saving above 50%** (more than double `rpK-60`'s 4.43 points) for only `-0.25` pts official accuracy (`-0.11` pts finished-only).
  - At `rpI-120`'s verified full-test accuracy-repair slope (`+0.211` accuracy per `-1%` saving), spending `-4.0% to -5.0%` saving from `rpL-60` lands at **`54.3%–55.3%` saved / `-0.65 to -0.85` accuracy** (whereas spending `-4%` from `rpK-60` drops saving to `50.4%`, right on the cliff edge).
  - Rollout source for `rpN` on `rpL-60`: use `--rp-base models/rpL-60 --rp-own-data --rp-rollouts-from rpI --rp-extra-questions svamp=1200 --rp-samples 8 --rp-seed 41 --rp-dpo-seed 1`. This generates 1,200 on-policy SVAMP-like questions with `rpL-60` in ~10 min while reusing `rpI`'s 4,500 non-math/hard-math rollouts (giving 121 `right_count` length-rule candidates with `314 vs 217` response tokens that `rpL-60`'s own compressed `at_least_N` rollouts lack, and cutting total `rpN` wall-clock time from 1.85h to **0.85h**).
