# ThinkingCap — current state, 10 Nov 2026 (evening)

One-page entry point. Details: `THINKINGCAP_FULL_DOSSIER.md` (Parts 0–G, plain record) and `EXPERIMENT_LOG.md` (lab notebook).

## Objective

Qwen3-0.6B (thinking on) must generate **more than 50% fewer tokens** (thinking included) with an accuracy change **better than
−1 point**, on BottleCapAI's ThinkingCap-Test full evaluation: 2,759 questions × 8 samples (SVAMP math, MBPP+ code, IFEval,
BFCL simple / multiple / irrelevance; BFCL pooled 400/200/240). All changes in the weights. The user also wants the result to come
from **one clean, reproducible recipe** (a single training line), not a stack of corrective stages, and refuses any recipe copied
from a competitor (michal-bida or others), even credited.

Yardstick ("our curve", fitted 2 Nov on 720-screen points): acc = −0.388 + 0.0899·s − 0.0028·s² (s = % saved). "vs curve" = the
accuracy above that line at the same saving. The 720 screen (720 questions × 4) reads optimistic: about +1 to +3 points of saving
and +0.7 to +1 point of accuracy above the full test for the same model.

## Best full-test results (2,759 × 8, official weighting)

| model | how it was made | saved | accuracy | vs curve |
|---|---|---|---|---|
| rpI step 120 | rpE-80 + 40 steps of "be right" DPO on its OWN answers to 5,700 new questions | 41.9% | +0.1 | **+1.64** |
| rpE step 80 | rc1-soup-30 + 80 steps "be right" DPO (guard pairs, re-read preference) on rc1's answers | 47.0% | −0.9 | +1.45 |
| rpE step 60 | same, 60 steps | 47.8% | −1.2 | +1.3 |
| rpE step 40 | same, 40 steps | 50.6% | −1.9 | +1.1 |
| rc1-soup-30 | 3 on-policy DPO rounds (A1, A2, ca1) + 1 "stop when sure" cut-calibration round (oracle re-answers) | 57.4% | −3.5 | +0.95 |
| pt-mid | October: 4 per-type LoRAs merged at .30/.45/.60/.45 | 53.6% | −3.3 | +0.3 |
| michal-bida (public, not ours) | one offline DPO run | 47.0% | −0.78 | ≈ +1.6 |

No model has reached the goal box (> 50% at better than −1). rpE-80 is 3 points of saving short; rpI-120 has no accuracy loss but
saves 42%.

## What 31 Oct – 10 Nov showed

- Almost every method lands ON the curve (moves along it, not above it): GRPO variants, COPD (Qwen3-4B teacher, Runs A and C),
  merges, more DPO steps on old answers (rpG, rpH), the repair of COPD Run C (rpF: 57.1/−4.7 → 48.1/−2.6, vs curve −0.1).
- Only the repairs ("shortest right answer beats shortest wrong answer", equal length, prefer answers that re-read the question)
  lifted a model above the curve, by +0.5 to +0.7 on the full test.
- **Learning from the CURRENT model's own answers stops the drift toward longer answers**: rpG and rpH (old model's answers)
  drifted from 50% to 33–37% saved in 40 steps; rpI (current model's answers on new questions), same steps, held 43% and gained
  accuracy (720: +2.6 vs curve; full test +1.64).
- COPD Run C (with a right/wrong reward added) over-cut: its new errors are answers that stopped thinking too early (svamp 54 of 73
  wrong answers vs 19/61 for Run A); wrong answers in the training data are ~2× longer than right ones, so "reward right" also
  rewarded "short".
- IFEval losses concentrate on counting rules (number of words / sentences / keyword frequency): responses too short. A pair rule
  for them (`--count-rules`) did not clearly help on the 720 (noisy: one part = 120 questions × 4).

## Running / queued (10 Nov evening)

- GPU 1: **od1** — online DPO from rpE-80: 4 rounds; each round the current model answers 1,290 new questions × 8, 160 pairs
  (half "be right", half "shortest right beats longest right when right ≥ 6/8"), 10 DPO steps; 720 after rounds 2 and 4.
  Then **od2** — the same loop from the ORIGINAL Qwen, 10 rounds × 10 steps: the candidate "single straight line" recipe.
- GPU 2: **cp1d** (COPD Run A + right/wrong reward only), then **cp1e** (+ penalty for wrong answers that stopped thinking early).

## Open questions for a reviewer

1. Is online DPO (od1/od2) the right bet for a single reproducible line that reaches > 50% / better than −1? What would you change?
2. Why does nearly everything land on the same curve? Is the curve a property of the 0.6B model, of our data, or of our pair rules?
3. What in our data or evaluation could be misleading us (720-screen optimism, noise, selection on the screen, contamination)?
4. Which single experiment would be most informative next, with our compute (two Colab L4 GPUs)?
