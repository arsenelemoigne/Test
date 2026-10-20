# Research brief: learning when a short think is enough (final stage, not started)

Sources read in full: AdaptThink (arXiv 2505.13417, code github.com/THU-KEG/AdaptThink) and
Thinkless (arXiv 2505.13379, code github.com/VainF/Thinkless). Also AutoThink (2505.10832), ARM
(2505.20258), LHRM/HGPO (2505.14631), ThinkSwitcher (2505.14183), AdaCoT (2505.11896), AutoL2S
(2505.22662), Qwen3 report (2505.09388), mode leakage (2510.12680), ALP (2506.05256), small-model
CoT learnability (2502.12143), zero-step mode selection (2510.19176).

## Key facts

- **AdaptThink**: advantage A = delta * 1[no-think] + R - R_ref(x), R_ref = base model's pass rate
  pre-sampled (K = 16). PPO clip 0.2, no KL. Importance sampling forces 50% of samples to start
  with `</think>` (no-think) and 50% with a normal opener, which fixes the cold start and prevents
  collapse. delta = 0.05, lr 2e-6, batch 128, 314 steps, DeepScaleR maths. 1.5B: -53% tokens,
  +2.4 pts (GSM8K 978 -> 480 tokens, 79.0 -> 83.1, 87% no-think). delta sweep: 0 gives -33% /
  +5.5, 0.1 gives -64.5% / -0.5. No-think share falls with difficulty (MATH L1 98% -> L5 62%).
- **Thinkless**: control token `<short>` / `<think>`; reward +1 short-correct, 1-gamma
  think-correct (repo 0.5 then 0.8), -1 wrong. Plain GRPO collapses to always-short within ~120
  steps; fix (DeGRPO) normalises the control token separately (alpha = 1/1000). GSM8K 1919 -> 624
  tokens at 83.5 -> 84.2; MATH-500 -4.2 pts.
- **ARM** has an explicit non-empty short-CoT format (~33 tokens on CSQA) among four formats;
  plain GRPO collapses to long CoT in ~10 steps, fixed by up-weighting rare formats early.
- **ALP** (closest to "difficulty from group pass rate"): r = 1[correct] - beta * N *
  max(p_solved, 1/K); ~50% fewer tokens at 1.5B, spends 5.35x more tokens on the hardest 30% than
  the easiest 30%. No mode token at all.
- **No paper trains adaptive thinking at 1B or below.** Qwen3-0.6B's no-think mode is much
  weaker than thinking (MATH-500 55.2 vs 77.6), so skipping thinking is costly at this size;
  small models learn short CoT better than long CoT (Qwen2.5-0.5B 19.5 vs 14.8).
- Failure modes reported: collapse to always-short (Thinkless), always-think (AutoThink naive,
  AdaCoT, LHRM at 1.5B), reasoning leaking after `</think>` (AutoThink, Qwen3 hybrids), accuracy
  loss concentrated on the hardest items (ThinkSwitcher, AutoL2S).

## Proposed design for us (suggestion, not from the sources)

- Short mode = a model-written think of at least ~10-15 and at most ~48-64 content tokens. Never
  the empty template block.
- Rollouts: of 8, force 4 to open with a short-mode opener during training only, cut at the
  budget, mask the forced/inserted tokens from the loss (or importance-sample the first decision
  as AdaptThink); evaluation applies no prefix, so track the free-mode short-think rate.
- Reward r = R + delta * 1[short and correct] - R_ref(x), used directly as the advantage (a group
  mean would cancel R_ref and drop AdaptThink's accuracy constraint); delta ~0.1-0.2, swept small
  first; R_ref pre-sampled from the base model (K = 16).
- Keep the 50/50 forcing all run, sum (not per-sequence) normalisation, low lr.
- Monitor per dataset and per difficulty bin: free-mode short rate, short vs long accuracy, think
  and answer lengths, reasoning words after `</think>`, truncations, accuracy on the hardest bin,
  empty-think rate (must be 0).
- Simpler alternative to compare: ALP's difficulty-scaled length penalty.
- Rule risk: a short non-empty think satisfies "thinks before it answers" literally; the
  organisers' reading is unverified. Keep a floor on think length and report the distribution.
- Expected: gains mostly on the easiest GSM8K/ARC/CSQA items; code probably needs real thinking.
