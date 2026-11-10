# The budget ladder — what we are trying to build (for review, 10 Nov 2026)

This document is self-contained. The attached `EXPERIMENT_LOG.md` is the full lab notebook (oldest first, newest at the end;
the last ~700 lines cover 6–10 Nov). Please read this document first, then use the log to check claims and numbers.

---

## 1. The task

- **Model:** Qwen3-0.6B, thinking mode on (it reasons between `<think>` and `</think>`, then answers).
- **Goal:** make it generate **more than 50% fewer tokens** (thinking included) with an accuracy change **better than −1
  point** versus the original model. All changes must be in the weights (LoRA merged); no prompt tricks, no test-time switch.
- **Official evaluation (BottleCapAI ThinkingCap-Test):** 2,759 questions × 8 samples (temperature 0.6, top-p 0.95, top-k 20;
  token cap 8,192 for code, 4,096 otherwise): SVAMP math 1,000, MBPP+ code 378, IFEval 541, BFCL function calling 840 (one
  function 400, several 200, none fits 240). Headline = the 4 benchmarks equally weighted (BFCL's three parts pooled
  400/200/240): accuracy change = mean of the 4 changes; tokens saved = 1 − geometric mean of the 4 token ratios.
- **Team constraints:**
  1. One clean, reproducible recipe (a single training line), not a stack of corrective stages.
  2. No copying of any competitor's recipe (michal-bida, Lightning McQwen3, others), even credited.
  3. Training only on the model's own answers (no teacher-written answers).
  4. Compute: two Colab L4 GPUs (24 GB each).
  5. No decision may use the test or the 720-question screen (a subset of the test) any more; see section 3.

## 2. Where we stand (full test, official weighting)

| model | how it was made | saved | accuracy |
|---|---|---|---|
| rpI step 120 | rpE-80 + 40 DPO steps of "be right" pairs on its own answers to 5,700 new questions | 41.9% | +0.1 |
| rpE step 80 | rc1-soup-30 + 80 DPO steps of "be right" pairs (shortest right > shortest wrong, prefer answers that re-read the question) | 47.0% | −0.9 |
| rc1-soup-30 | 3 on-policy DPO rounds (shorter right > longer right, right > wrong) + 1 cut-calibration round | 57.4% | −3.5 |
| michal-bida (public, not ours) | one offline DPO round on ~5,900 own-answer pairs | 47.0% | −0.78 |

Nobody, us included, has reached > 50% at better than −1. Almost every method we tried (DPO rounds, GRPO variants, per-type
LoRA merges, a 4B-teacher token-level distillation called COPD, repairs) lands on the same accuracy-vs-saving trade-off curve;
only the "be right" repairs moved a model slightly above it. Learning from the CURRENT model's own answers slowed the drift
toward longer answers (rpI), but did not stop it.

## 3. What an independent review (10 Nov) told us

- We used the 720-question screen (part of the test) over and over to choose models; its best results are optimistic
  (screen vs full test: saving +1.2 points on average, accuracy +0.4, model-dependent).
- Our "curve" is a description, not a yardstick: at 50% saved it predicts −2.9, so "+1 above the curve" still misses.
- "Right 6 of 8 times" is not a safety certificate (a 50% question passes it 15% of the time).
- IFEval alone costs our best model 0.65 point overall (counting rules: responses too short).
- Recommended protocol: fresh vs frozen answer arms, 2 seeds, a NEW held-out dev set (never the test) for every choice, a
  checkpoint fixed in advance, question-level paired bootstrap on the full test.

## 4. The idea: build thinking from the bottom up

Accuracy as a function of the thinking budget, A(B), per data type: A(0) is the no-thinking accuracy; A rises steeply, then
flattens (log-like) toward the original's natural accuracy A_nat; each extra token buys less. The goal is the fewest tokens at
which accuracy reaches A_nat − 1, i.e. minimise expected tokens subject to accuracy ≥ A_nat − 1, equivalently stop each question
where its next tokens add less accuracy than a threshold λ (spend tokens where the marginal gain is highest).

Instead of shortening a long-thinking model (what everyone does), we start from **no thinking** and **climb a ladder of thinking
budgets**: at each budget, train the model to get as many questions right as possible within that budget, make sure questions
already handled at lower budgets do not start thinking more, then raise the budget. Each stage should LIFT A(B) at its budget
without lowering it at smaller budgets, so the point where A reaches A_nat − 1 moves to fewer tokens.

What we know at budget 0 (the original, full test, thinking OFF vs ON): math 20.6 vs 88.7, code 40.8 vs 52.4, IFEval 57.5 vs
58.7, one function 71.4 vs 82.2, several 55.3 vs 76.1, none fits 90.3 vs 83.4. So IFEval and "none fits" need almost no thinking;
math needs a lot. A trained model (rc1-rl-40) with thinking off, on 240 fresh questions × 8: math 23.1 (original off 17.5),
several functions 63.4 (58.8), one function 78.8 (76.6), IFEval 55.3 (60.6), code 33.1 (37.2).

## 5. Definitions

- **Per-type budget rungs:** the types think very differently (the original's right answers: tools ~200 tokens, IFEval ~520,
  math ~580, code ~1,200). Rungs per type = 0, then the 5/10/20/30/40/50/65/80/90th percentiles of that type's natural
  right-answer thinking lengths, then "natural" (no cap). Final-answer tokens are outside the budget but count in the score.
- **An answer at budget B:** generate the thinking, cut it at the last sentence end within B tokens, close `</think>`, and let
  the model write the final answer. B = 0 = empty thinking.
- **The original's rate p0(q):** share of its 8 natural (uncapped) answers that are right.
- **A question's level (needed budget) b\*(q):** the lowest rung at which it is right at least p0(q) − 1/8 of the time (as right
  as the original, one answer of slack). Never-solved questions (p0 = 0) get no "stop early" pairs; but if a cut of a wrong
  answer is right (the model had the answer, then talked itself out of it), that gives a "stop earlier and be right" pair.

## 6. Phase 0 — measurement only, no training (~2–3 GPU-hours; code written and tested, not yet run)

1. ~1,500 own training questions (never evaluation questions; contamination-checked): math 450, code 300, IFEval 450, tools
   300.
2. The original answers each 8 times uncapped → p0 per question and the per-type rungs.
3. Each natural answer is cut at each rung and re-answered (adaptive: a question stops climbing at its level).
4. Report per type: the distribution of levels; the **oracle curve** (tokens and accuracy if every question used exactly its
   level), the ceiling of the method; how often a shorter cut is MORE right than the natural answer; whether the level is
   predictable from the question (rank correlation with question length and with p0).
5. **Go / no-go:** continue only if the oracle saves ≥ 60% at accuracy ≥ original − 0.5 (official weighting) on these
   questions; the trained model will land below its oracle.

## 7. Phase 1 — the ladder training (designed, not built)

**Stages:** one per rung, per type, in order. Each stage:
1. **Fresh questions, never reused** (~800 per stage in the test's type shares; ~7 stages ≈ 5,600 of our ~17,000 usable own
   questions; IFEval, ~3,000 usable, gets ~400 per stage; more IFEval-style prompts can be generated with our builder).
2. **The CURRENT model answers** each question 8 times at the stage's budget B_k and 8 times at each lower budget (cheap).
   This gives each question's level on the spot; nothing is carried over between stages.
3. **Question mix at the stage:** ~60% "edge" questions (right 2–6 of 8 at B_k: between this rung and the next; they carry
   the accuracy gain), ~30% already solved at a lower level (they give the length-anchor pairs), ~10% just out of reach
   (right 0–1 of 8 at B_k but right at a higher rung). Never-solved questions are not used. Starting shares, to be checked on
   Phase 0. Between stages, types further from the original's accuracy get more questions (in proportion to their gap, with a
   floor of half their test share).
4. **Pairs (DPO), one rule set every stage:**
   - **Get there within the budget** (compaction): at B_k, a right answer beats a wrong answer of SIMILAR thinking length
     (±25%), so the pair carries correctness, not shortness.
   - **Don't drift up** (the anchor): for a question already solved at a lower level, its right answer at that level beats a
     right answer that thought ≥ 1.5× more.
   - **Don't stop too early** (the guard): a right answer at the question's level beats a wrong answer one rung below.
   - IFEval: budgets count thinking only; on counting-rule questions no pair may shorten the final response.
5. **Training:** ~300–600 pairs = 20–40 DPO steps of 16 pairs, one pass, LoRA r16 continued from the previous stage, lr 1e-5
   constant (2 warm-up steps), β 0.1, no SFT/NLL term, fp32 weights + fp16 compute, **reference = the previous stage**.
6. **Tracking panel (the guard between stages):** a FIXED held-out set (~300 questions, 50 per type, never trained on, not the
   test) re-answered after every stage, 8 times at EVERY rung. It reports per type: each panel question's level and how levels
   moved since the last stage (down = progress, up = an earlier stage being undone); mean thinking length per level with its
   spread; accuracy per level and overall vs the original; the A(B) curve.
7. **Exit check:** a stage is done only if accuracy at B_k rose on the panel AND lower-level questions did not get longer
   (thinking within +10%). Otherwise re-run the stage on another batch of new questions (≤ 2 extra batches), with more of the
   failing pair kind.
8. **Rollback:** every stage's adapter is kept; if the panel shows a stage undid earlier ones, go back one stage and redo it.
9. **Near the end:** when a type is within ~1 point of the original's accuracy, its stages get more questions and more "don't
   stop too early" pairs (not a higher lr). A type stops climbing once back at the original's accuracy on the dev set.

**Reference settings we based this on:** our best repair rpE (equal-length right > wrong pairs, lr 1e-5 constant, 80 steps)
lifted accuracy +1 to +3 points without a length push. On identical pairs, fp32/fp16 + cosine-to-0 (michal-bida's published
settings, used here only as a measurement) cut 13–15 points less and lost ~2.8 points less than our older bf16 settings.
michal-bida's own run: 5,862 pairs, one pass, lr 2e-5 cosine, LoRA r32, β 0.1.

**Later option — RL stages:** reward +1 if right AND thinking within B_k, else 0 (+ a small bonus for staying at a lower
level). GRPO on the current model, same panel and rollback. For a stage DPO cannot make solid, or as a comparison arm. Our past
GRPO runs were slow and sensitive (2–3× the GPU time of a DPO stage).

**Control arm (from the review):** the same stages with answers from the ORIGINAL model (offline), same counts; 2 seeds each.

**Evaluation:** a new 512-question dev set from our own sources (never trained on, not the test) is the only basis for choices;
the final checkpoint (end of the last stage) is fixed in advance; both seeds go to the full test once; paired bootstrap;
success = both seeds > 50% saved and better than −1.

## 8. Risks we see

1. **Can the model predict its budget?** It must decide how much to think from the question alone. On math its confidence is
   badly calibrated (says 0.98, right 0.80–0.86). Phase 0 checks predictability first.
2. **Forced cuts vs natural stops:** an answer cut by force reads differently from one that stopped on its own; DPO on cut
   answers may teach abrupt stops, or only "stop when told". Natural short answers should be preferred as chosen answers where
   they exist.
3. **Noise in levels:** 8 answers per question; one answer of slack.
4. **IFEval responses** must not shrink (our known failure).
5. **Seven stages of DPO may accumulate drift** (reference = previous stage); the panel and rollback are the defence.

## 9. What we would like from you

1. **Analysis:** is the bottom-up ladder a sound way to reach > 50% saved at better than −1 with one reproducible recipe on
   Qwen3-0.6B? Where is it most likely to fail? What does our log say for or against it?
2. **Design review:** the per-type rungs, the level definition, the three pair kinds and their shares, the question mix,
   hyperparameters (steps per stage, lr, LoRA rank, β, reference = previous stage), the panel and exit check, the
   offline-vs-online choice. What would you change, concretely, and why?
3. **Strategy:** given two L4 GPUs over the next week and everything in the log, what is the best sequence of experiments
   (Phase 0 first?), with decision rules at each point? Is there a better route to the goal we are missing?
4. **Traps:** anything in our evaluation, data or method (from the log) that could make the ladder look better or worse than it
   is.
Please be specific, quantitative, and blunt.
