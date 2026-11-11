# Path 1 — final run (queued 11 Nov 2026, ~01:45)

One clean, reproducible 2-round DPO recipe **from the original Qwen3-0.6B**, the model's own answers only, public datasets
allowed (user, 11 Nov: using the same public datasets as michal-bida is not copying; his recipe stays off-limits), every choice on our own held-out questions (Dev-512), never on the test or the 720.
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

- **Material (p1src):** the original's 10 stored answers per pool question, including the public datasets (MetaMath, the
  math1 / math2 / nofit4 top-ups: SVAMP-like math, much closer to the test than full-difficulty GSM8K), never a Dev-512
  question or a near-duplicate of one (≥ 30% shared rare 8-grams), 75% of the questions (a fixed hash; the other 25% stay
  unseen for round 2). No new generation.
- **Pairs (repair_pairs.py --kinds mix --mix-preset round1), 3,200:** quotas math 800, code 800, IFEval 800, one function 280,
  several 200, no fit 240. Per type:
  - math and no fit: 60% "shorter right beats longer right" / 40% "compact be-right";
  - code, one / several functions: 70% / 30%;
  - IFEval: 50% / 50%, and counting-rule questions only get counting-rule pairs (the right answer writes at least as much,
    thinks at most 1.2×), never a "shorter" pair.
  - "Shorter": on questions right ≥ 8 of 10 (user, 11 Nov: 6 of 10 lets a 50%-solved question through 38% of the time;
    8 of 10, 5%), longest right ≥ 1.35× the shortest (backup 1.20–1.35× if short of supply).
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

- The round-1 model answers ~3,200 NEW questions × 8 (public datasets allowed; math SVAMP-like) (math 1,000, code 600, IFEval 750, one function 300, several 220, no
  fit 330; never a round-1 or Dev-512 question; seed 0 and seed 1 draw different questions).
- **Pairs, 1,280:** 25% "shorter" (right ≥ 7/8 since 11 Nov, ≥ 1.30×) / 60% compact be-right (+ IFEval counting rule) / 15% "don't stop too
  early" (a right answer at ≤ 1.15× the question's median right thinking beats the longest wrong answer under 0.5× of it).
- **Training:** fresh LoRA on the round-1 model, reference = the round-1 model, lr 1e-5 constant, 80 steps; Dev-512 at 40,
  60, 80.

## Choice and final test

The 2-seed weight average at the best round-2 step on calibrated Dev-512 (highest accuracy among checkpoints with enough
saving), then one full test (2,759 × 8) with a paired bootstrap, both seeds reported; checkpoints along the line kept for the
submission's length/accuracy curve.

## Rules added on 11 Nov (user)

- **Fixed proportion, never silently refilled:** in every type the ratio "think less" : "be right" is fixed (round 1: 60/40 for
  math and no fit, 70/30 for code and function calls, 50/50 for IFEval's non-counting half; round 2: 25 / 75 with the
  early-stop guards counted inside the "be right" side). If one kind runs short, the whole type shrinks with it (fewer pairs,
  same mix) instead of filling the gap with the other kind. The shortfall is written to shortfall.json.
- **IFEval:** counting-rule questions never get "shorter" pairs; they hold their own fixed half of IFEval's quota (counting-rule
  be-right pairs); the mix applies to the other half.
- **Fill until full (round 2):** if a type is short of pairs by more than 10%, the round-1 model answers more new questions for
  that type (3x the shortfall, at least 100; up to 2 extra batches) and the pairs are rebuilt. Round 1 uses a fixed pool of
  stored answers, so there it keeps the proportion with a smaller total.
- **Hold:** lifted by the user on 11 Nov ~03:20 ("queue it directly, I will check tomorrow"); calibration so far: saving reads within ~1 point of the full test, accuracy ~2 points high.

## Final changes before launch (11 Nov ~03:40; user: "update the path 1 accordingly and queue it directly")

After the Fable 5.2 review (reviews/fable_path1_review.md) and consult8 (Gemini 4 Argon, verdict **GO for round 1**,
reviews/consult8_path1_final/run/final_answer.md, $3.64). Every change was tested on the REAL answer files downloaded from
Drive (the original's 10 stored answers, rc1-soup-30's and rpE-80's own answers), not on mock data.

- **Round 2's "shorter" bar raised to 7/8** (was 6/8: a half-solved question passed 14.5% of the time; 7/8: 3.5%; round 1 stays
  8/10). Measured on rc1-soup-30's real answers (a round-1-like model): the "shorter" supply barely drops (-7 to -15%); the
  scarce kind is "be right", not "shorter".
- **The mix is a hard rule** (repair_pairs.py): pairs are taken, then every side is cut back to the scarcest one, so the final
  file holds exactly the set mix whatever ran short. Fixes found by consult8 and the Fable review: the "don't stop early" pair
  and the "be right" pair of one question could be the same pair (the supply was counted twice and the mix drifted in round 2);
  a guarded question now gives no second be-right pair; counting-rule questions never give a guard; shortfall.json holds the
  REAL gap (fill-until-full was reading an under-count).
- **IFEval's counting half is fixed 1 : 1 with the other half** (before: a short counting half let the other half grow = the
  silent refill). Counting-rule pairs are the scarce kind (244 for a half-quota of 400 on the real answers), so a counting-rule
  question may now give up to 2 pairs on different answers (411 available).
- **Share kept unseen for round 2, per type:** 25%, IFEval 33%, "several functions" 35% (round 2 needs ~1,000 IFEval questions
  for its counting half and ~250 "several functions" questions; with a flat 25% it had 792 and 187).
- **IFEval "shorter" pairs never prefer a shorter response** (--resp-guard 0.9: the chosen answer's response is at least 0.9x
  the rejected one's). Reason: the full-test records show IFEval's loss in every trained model is shortened RESPONSES on "at
  least N words / sentences" questions, not shortened thinking (REPAIR_IFEVAL_TOOLS.md).
- **Round 2 never draws a near-duplicate of a Dev-512 question** (consult8's bug 1: Dev-512's math has gsm8k-std ids, round 2's
  public math has pool ids for the same GSM8K problems; ids alone missed them; now the 8-gram rule of round 1 applies).
- **Dev-512 scores** now also print the paired difference against a reference (round 1 vs rpE-80; round 2 vs round 1) and the
  thinking / response token split per type.
- **Memory:** repair_pairs.py stores token ids compactly (round 1's 1.3 GB answer file: 2 GB of RAM instead of ~10 GB).
- consult8's "bug 4" (tool top-up source names) is a false alarm: those rows carry source glaive_tools / toolace + a category.
  Not taken: its idea to prefer a non-early wrong answer in the be-right pair (it would change rpE's proven pair).

**Round 1 on the real stored answers (both seeds built locally, identical to what the GPU will build):** 13,082 questions, 3,040
pairs: math 480 shorter / 320 be-right, code 560 / 240, one function 196 / 84, several 140 / 60, no fit 144 / 96, IFEval 180
shorter / 180 be-right / 360 counting-rule (720 of 800: the counting half is the limit). "Shorter" pairs = 56% of the file, in
every prefix. The quotas sum to 3,120, not 3,200 as written above; 200 steps = 3,200 pairs, so the first 160 pairs are seen
twice. Longest pair 8,339 tokens (cap 8,704). Seeds 0 and 1 share 1,237 of 3,040 pairs.

**Round 2 draw:** math 1,000, code 600, IFEval 960, one function 340, several 250, no fit 330 new questions x 8.

**Queue (HOLD lifted by the user):** GPU 2: calibration (rpI-120 last) -> p1src -> round 1 seed 0 (Dev-512 at 100 / 150 / 200)
-> rpK (the independent IFEval + tool-call repair, ~1.2 h) -> gate -> round 2 seed 0. GPU 1: od2 (running, ~12 h left) -> seed 1.

**What to watch (consult8):** pairs file: any non-IFEval type under 85% of its quota = stop; "chosen longer" must stay under
~25% (math / code / tools) and ~35% (IFEval). Training: pairs ordered right 55 -> 72% by step 20, 82-88% at 150-200; over 96%
before step 150 = over-fitting; under 65% at step 100 = stalled. Gate nuance: if step 150 passes rpE-80 + 6 and step 200 does
not, the gate closes; point round 2 at step 150 by hand.

## Expected (honest)

Gemini: round 1 ~53–57% / −2.2 to −3.2; final 49.5–52.5% / −0.7 to −1.3. The Claude review was more cautious (the log has
never shown a clean 2-round climb above the curve). The gate tells us after round 1 which of the two is right.

## Open decisions for the user

- Precision: fp32 weights + fp16 compute and a pair-length cap of 8,704 tokens are labelled in our code as michal-bida's
  settings (where we first saw them); every repair since 5 Nov used them. Kept for now; switch back to bf16 if you prefer.
- Seed 1 starts on GPU 1 only after od2 finishes (~10 rounds of the original's long answers, many hours); stop od2 to start it
  sooner?
