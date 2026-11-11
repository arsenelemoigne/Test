# Path 1 — final run (queued 11 Nov 2026, ~01:45)

One clean, reproducible 2-round DPO recipe **from the original Qwen3-0.6B**, the model's own answers only, our own data sources
only (nothing from michal-bida's sources), every choice on our own held-out questions (Dev-512), never on the test or the 720.
Designed from Gemini 4 Argon's two reviews (reviews/consult6_argon/) and an independent Claude review
(reviews/claude_path1_review.md); both agreed on the changes below.

## Why it changed from the first plan

Both reviewers showed from our log that the first round-1 design (~39% "shorter" pairs, the rest "be right" pairs at equal
length) would land around 20–33% saved: runa2 (31–33%), D6 (28%), D12 (18–22%). Round 2 cannot add both saving and accuracy
(od1: 50.4 / −0.2 → 53.3 / −1.8 on the 720, falling vs the curve). Our only above-the-curve full tests came from a model
shortened hard first (rc1-soup-30, 57.4 / −3.5) then repaired (rpE-80 47.0 / −0.9, rpI-120 41.9 / +0.1). So: **round 1 shortens
hard to ~53–57%, round 2 repairs accuracy**, the rc1 → rpE structure collapsed into two clean rounds.

## Step 0 — Dev-512 and calibration (GPU 2, running)

- Dev-512: 512 own questions outside every question used for training recently (math 140, code 100, IFEval 66 counting-rule +
  66 other, one function 60, several 35, no fit 45); the original answers it × 8.
- Calibration: rc1-soup-30, rpE-40, rpE-60, rpE-80, rpI-120 scored on Dev-512 (their full tests are known), to read the Dev →
  full-test offset.

## Round 1 — shorten-dominant, from the original (GPU 2 seed 0, GPU 1 seed 1)

- **Material (p1src):** the original's 10 stored answers per pool question, own sources only (no MetaMath, none of the
  nofit4 / math1 / math2 top-ups), never a Dev-512 question or a near-duplicate of one (≥ 30% shared rare 8-grams), 75% of
  the pool's questions (a fixed hash; the other 25% stay unseen for round 2), plus 2,200 NEW full-difficulty GSM8K questions
  answered now by the original × 8 (the pool's own math is ~660 mostly easy questions).
- **Pairs (repair_pairs.py --kinds mix --mix-preset round1), 3,200:** quotas math 800, code 800, IFEval 800, one function 280,
  several 200, no fit 240. Per type:
  - math and no fit: 60% "shorter right beats longer right" / 40% "compact be-right";
  - code, one / several functions: 70% / 30%;
  - IFEval: 50% / 50%, and counting-rule questions only get counting-rule pairs (the right answer writes at least as much,
    thinks at most 1.2×), never a "shorter" pair.
  - "Shorter": on questions right ≥ 6 of 10, longest right ≥ 1.35× the shortest (backup 1.20–1.35× if short of supply).
  - "Compact be-right" (rpE's proven pair): the shortest right answer that re-reads the question (within 1.5× the shortest),
    else the shortest right, beats the shortest finished wrong answer.
  - Kinds interleaved by share through the file (fixed bug: "shorter" pairs used to come last).
- **Training:** fresh LoRA r16 on the original, reference = the original, β 0.1, no NLL, batch 16, lr 2e-5 with 10 warm-up
  steps and cosine to 5e-6, 200 steps, fp32 weights + fp16 compute, seed 0 / 1 (LoRA init, pair order and pair choice
  differ per seed). Scored on Dev-512 at steps 100, 150, 200.

## Gate (pre-registered, automatic)

Round 2 runs only if round 1's Dev-512 saving at step 200 is **at least rpE-80's Dev-512 saving + 6 points** (rpE-80 = 47.0%
on the full test; round 1 must reach ~53%+ for round 2 to repair down to ~50%). Otherwise round 2 is skipped and reported; the
reviewers' fallback is 60 more round-1 steps at lr 1.5e-5.

## Round 2 — on-policy repair (after the gate)

- The round-1 model answers ~3,200 NEW own questions × 8 (math 1,000, code 600, IFEval 750, one function 300, several 220, no
  fit 330; never a round-1 or Dev-512 question; seed 0 and seed 1 draw different questions).
- **Pairs, 1,280:** 25% "shorter" (right ≥ 6/8, ≥ 1.30×) / 60% compact be-right (+ IFEval counting rule) / 15% "don't stop too
  early" (a right answer at ≤ 1.15× the question's median right thinking beats the longest wrong answer under 0.5× of it).
- **Training:** fresh LoRA on the round-1 model, reference = the round-1 model, lr 1e-5 constant, 80 steps; Dev-512 at 40,
  60, 80.

## Choice and final test

The 2-seed weight average at the best round-2 step on calibrated Dev-512 (highest accuracy among checkpoints with enough
saving), then one full test (2,759 × 8) with a paired bootstrap, both seeds reported; checkpoints along the line kept for the
submission's length/accuracy curve.

## Expected (honest)

Gemini: round 1 ~53–57% / −2.2 to −3.2; final 49.5–52.5% / −0.7 to −1.3. The Claude review was more cautious (the log has
never shown a clean 2-round climb above the curve). The gate tells us after round 1 which of the two is right.

## Open decisions for the user

- Precision: fp32 weights + fp16 compute and a pair-length cap of 8,704 tokens are labelled in our code as michal-bida's
  settings (where we first saw them); every repair since 5 Nov used them. Kept for now; switch back to bf16 if you prefer.
- Seed 1 starts on GPU 1 only after od2 finishes (~10 rounds of the original's long answers, many hours); stop od2 to start it
  sooner?
