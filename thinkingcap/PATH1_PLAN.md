# Path 1 plan — "Compact-and-Verify" DPO (11 Nov 2026)

Built from Gemini 4 Argon's review (reviews/consult6_argon/REPORT.md, its Rank 1) after an independent fact-check
(reviews/consult6_argon/FACT_CHECK.md). Goal unchanged: > 50% tokens saved at better than −1 point on the full test (2,759 × 8),
one reproducible recipe from the original Qwen3-0.6B, the model's own answers only, no teacher text, no prompt tricks, no
competitor recipe.

## What we take from the review, and what we do not

- **Taken:** a 2-round recipe from the original Qwen. Round 1 shortens on the original's own answers; round 2 re-samples from the
  round-1 model and mixes three pair kinds: ~30% "shorter right beats longer right" (only on questions right ≥ 6/8), ~55% "right
  beats wrong at equal thinking length (±20%), preferring answers that re-read the question" (IFEval counting-rule questions:
  the right answer writes at least as much and thinks at most 1.2×), ~15% "don't stop too early" (a right answer at ≤ 1.15× the
  question's median right thinking beats a wrong answer under 0.5× of it). Every choice on our own held-out questions (Dev-512),
  a separate Confirm-512, 2 seeds, one full test at the end, paired bootstrap.
- **Not taken:** its win probabilities (no measured basis); its Path 3 (built on michal-bida's pair recipe + DAST: excluded by our
  rule); its claim that fp32 alone recovers ~2.8 points (wrong: fp32 moved the model along the curve); its reading that rpI
  "stopped the drift" (rpI still drifted longer for 20 steps; single seed).

## Step 0 — Dev-512 and calibration (queued on GPU 2, ~2 GPU-hours)

- **Dev-512:** 512 own questions drawn OUTSIDE every question already used for training by rpE/rpG/rpH (seed 0), rpI (seed 1)
  and od1 (seed 7): math 140, code 100, IFEval 132 (66 with a counting rule, 66 without), tools 140 (60 one function, 35
  several, 45 no fit). Never the test, never the 720. Later runs exclude it (`--rp-exclude-dev`; od2 already does).
- The original Qwen answers it × 8 (the reference). Score = the official headline (4 benchmarks equal, BFCL pooled 400/200/240,
  tokens saved = 1 − geometric mean of the token ratios) with a paired bootstrap 95% interval (dev_score.py).
- **Calibration:** rc1-soup-30 and rpE-80 are scored on Dev-512 too. Their full tests are known (57.4 / −3.5 and 47.0 / −0.9),
  so the Dev-512 → full-test offset is measured instead of assumed.

## Step 1 — the cheap test, rpJ (queued on GPU 2 after Step 0, ~2.5 GPU-hours)

- rc1-soup-30 (a round-1-like shortened model) + 60 DPO steps of the round-2 pair mix, built from rc1-soup-30's OWN existing
  answers (rpH's 5,700 questions × 8; no new generation). Quotas: math 320, code 320, IFEval 320, tools 320 (100 / 80 / 140).
  Fresh LoRA r16, β 0.1, lr 1e-5 constant, batch 16, fp32 weights + fp16 compute, reference = rc1-soup-30. Scored on Dev-512 at
  steps 20, 40, 60.
- **Decision rule (relative, so the Dev offset cancels):** rpJ passes if, at some step, it saves at least as much as rpE-80 on
  Dev-512 AND is more accurate than rpE-80 by more than the paired interval's half-width, OR it saves ≥ 3 points more than rpE-80
  at the same accuracy. It fails if it lands on rpE-80's line (same trade, just a different point) or below. (rpE-80 = the repair
  with 0% shorten pairs; rpJ adds the 30% shorten anchor and the capped guard: this isolates the mix.)

## Step 2 — the full recipe, only if Step 1 passes (~40 GPU-hours, 2 GPUs, ~1 day)

- **Round 1 (from the original Qwen):** ~8,000 own questions × 8 own answers (never-solved dropped; ~half reliably solved
  6–8/8, ~half sometimes solved 2–5/8) → ~3,200 pairs (200 steps of 16): per type the three kinds (math and no-fit: 35% shorten /
  50% equal-length right / 15% guard; code and one / several functions: 50 / 35 / 15; IFEval: counting-rule questions only
  counting-rule right pairs, others 55% shorten / 45% right). LoRA r16 α32, β 0.1, lr 1.5e-5 with 10 warm-up steps and cosine to
  5e-6, reference = the original, seeds 0 and 1; merge each seed at step 200.
- **Round 2 (on-policy):** each seed's round-1 model answers ~6,000 NEW own questions × 8 → 1,280 pairs with the rpJ mix (30 /
  55 / 15), fresh LoRA, lr 1e-5 constant, 80 steps, reference = that round-1 model; checkpoints 40 / 60 / 80.
- **Choice:** the 2-seed weight average (0.5 / 0.5) at 40 / 60 / 80, scored on Dev-512 × 8; the most accurate among those saving
  ≥ the target on Dev (after Step 0's offset); verified once on Confirm-512; then one full test with the paired bootstrap, both
  seeds reported. Submit several checkpoints along the line (the challenge README asks for points on the length/accuracy curve).

## GPU plan

- GPU 2: Step 0 → Step 1 (queued now, after cp1d; cp1e dropped).
- GPU 1: od1 finishing (online DPO from rpE-80); od2 queued (the same loop from the original Qwen, 10 rounds, ~9 GPU-hours).
  If Step 1 passes, od2 should probably give way to Step 2 (open decision).

## Open questions for the next analysis

1. Is the relative decision rule for rpJ (vs rpE-80 on Dev-512) sound, and what margin should count?
2. Round-1 design: are 3,200 pairs and lr 1.5e-5 cosine right for a 0.6B LoRA from the original model, given our history (one
   offline round from the original typically lands at ~50% / −2.5 to −3 on the full test)?
3. Should round 2 keep the round-1 model as reference, or the original?
4. Dev-512 is drawn from our own sources, which differ from the test's (GSM8K-style math vs SVAMP, KodCode vs MBPP+): how much can
   a Dev-512 ranking be trusted, and is calibrating on rc1-soup-30 / rpE-80 enough?
5. Given the fact-check, does Path 1 remain the best bet, or would you now rank something else first?
