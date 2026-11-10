# Budget ladder — plan (draft, 10 Nov 2026, user's idea)

## Idea

Start from no thinking and raise the thinking budget step by step. For every question, find the smallest budget at which
the model is **as right as the original Qwen is on that question**. Train the model to think about that much: no thinking
where none is needed ("freeze"), enough where it is needed, and never more than the original needed. Questions settled at a
low budget stay in the training mix as anchors while the ladder climbs.

One recipe, from the original Qwen, built to be reproducible (2 seeds, selection on a held-out dev set, one full test at
the end), following the consult5 review's protocol.

## What we already know (original Qwen, full test, thinking OFF vs ON)

| part | thinking on | thinking off | tokens saved off |
|---|---|---|---|
| math (svamp) | 88.7 | 20.6 | ~90%+ |
| code (mbpp) | 52.4 | 40.8 | ~90%+ |
| IFEval | 58.7 | 57.5 | 62% |
| one function | 82.2 | 71.4 | ~90%+ |
| several functions | 76.1 | 55.3 | ~90%+ |
| no function fits | 83.4 | 90.3 | ~90%+ |

So budget 0 already matches the original on IFEval and "no fit" and loses everywhere else, most on math.

## Definitions

- **Rungs, per data type (user, 10 Nov):** each type gets its own ladder, because the types think very differently (the
  original's right answers on the 720: tools ~190–210 tokens, IFEval ~520, math ~580, code ~1,210). Rungs = 0, then the 5th,
  10th, 20th, 30th, 40th, 50th, 65th, 80th and 90th percentiles of that type's natural right-answer thinking (measured in
  Phase 0 from the 8 natural answers), then natural. So every type gets the same number of rungs, packed where its answers
  actually are; a tool question is never tested at "1,024 tokens" and a code question never stops at "32" only. Final-answer
  tokens are not part of B but count in the score, as in the benchmark. Everything (b\*, the oracle curve, the decision
  rule, stages, quotas) is reported and decided per type; a type stops climbing once it is back at the original's accuracy.
- **How an answer at budget B is made — the cut, not a new sample:** the original answers each question 8 times with no cap
  (its natural answers). For each natural answer, keep its first B thinking tokens, close `</think>` there and let the model
  write the final answer (as rc1's cut calibration and label_cuts.py / check_vllm.py already do). B = 0 is an empty thinking.
  This gives 8 answers per question per rung for the price of short continuations, and every rung is "the same thinking,
  stopped earlier".
- **The original's rate p0(q):** the share of its 8 natural answers that are right (graded as the benchmark does).
- **Needed budget b\*(q):** the smallest rung B where the share right at B is at least p0(q) − tolerance.
  Tolerance (user, 10 Nov): one answer of eight (0.125), so a question at 6/8 natural needs 5/8 at B.
- **Classes of questions:**
  - p0 = 0 (the original's 8 natural answers all wrong): no explicit "stop early" pair (it could teach giving up on questions
    it can solve); the general shortening is expected to carry over to them anyway. **Exception:** if some cut of a wrong
    answer is RIGHT (the thinking had the answer and then talked itself out of it), that gives a "stop earlier and be right"
    pair: chosen = the right cut answer, rejected = the wrong natural answer. Phase 0 counts how many such questions exist.
  - b\* = 0: the "freeze" class (no thinking needed).
  - b\* = natural: thinking cannot be shortened on this question.

## Phase 0 — measurement only, no training (~2–3 GPU-hours, one L4)

1. Draw ~1,500 training questions from our own sources, never an evaluation question (the existing own_repair_questions
   draw, contamination-checked): about math 450, code 300, IFEval 450, tools 300 (100 per BFCL kind).
2. The original answers each 8 times with no cap (≈ 12,000 answers): p0 per question.
3. The ladder: re-answer from every cut at every rung (≈ 8 rungs × 12,000 short continuations; adaptive: a question stops
   climbing at its first qualifying rung, which cuts the cost to roughly a third).
4. Report, per part:
   - the distribution of b\* (share of questions at each rung) and the share of "freeze" questions;
   - **the oracle curve:** tokens and accuracy if every question used exactly its b\* — the ceiling this method can reach;
   - how often a shorter rung is MORE right than the natural answer (thinking that talks itself out of a right answer);
   - whether b\* can be predicted from the question (correlation with question length, type, the original's p0), which is
     the real risk: the model must guess its budget before it thinks.

**Decision rule for Phase 0:** go on only if the oracle saves at least 60% of tokens at an accuracy within 0.5 point of the
original on these training questions (the learned model will do worse than its oracle; 50% / −1 needs headroom). If the
oracle is near 45–50%, the method cannot reach the goal and we stop before training.

## Phase 1 — training, bottom-up (draft v2, designed with the user 10 Nov; built only after Phase 0)

**Shape.** Stages climb the per-type budget ladder. At every stage the thinking allowed goes up a little and accuracy should
go up a little; a stage ends only when it is solid; questions handled at a lower budget must not start thinking more.

**One stage = fresh questions, never reused** (against memorising): each stage draws its OWN new questions (about 800, the
types in the test's shares; ~7 stages use ~5,600 of our ~17,000 usable own questions; IFEval, the scarcest at ~3,000, gets
~400 per stage). No question appears in two stages; no pair is used twice.

**Sampling in a stage (online: the CURRENT model answers).** Each new question is answered 8 times with the thinking capped
at the stage's budget B_k (cut at the last sentence end, then the answer), AND 8 times at the budgets below (cheap: they are
short). So, on the spot, every question gets its level = the lowest budget at which the current model is about as right as
the original (one answer of slack). Nothing is carried over from earlier stages, so there is no separate "anchor set".

**Pairs in a stage (one rule set, the same every stage):**
1. **Get there within the budget** (the compaction lesson): at budget B_k, a right answer beats a wrong answer.
2. **Don't drift up** (the anchor, built from the same new questions): for a question already solved at a LOWER level, its
   right answer at that level beats a right answer that used much more thinking (>= 1.5x).
3. **Don't stop too early** (the guard): a right answer at the question's level beats a wrong answer one budget below.
4. IFEval: budgets count thinking only; on counting-rule questions no pair may shorten the final response.

**Training per stage:** about 300-500 pairs = 20-30 DPO steps of 16 pairs, one pass, LoRA continued from the previous stage,
lr 1e-5 constant, beta 0.1, the original Qwen as the reference throughout (so drift is measured against the start).

**"Each stage strong" — the exit check (on a small held-out slice, never the 720 or the test):**
- accuracy at B_k went UP vs the stage's start, and
- the questions at lower levels did NOT get longer (their thinking within +10%).
If either fails, the stage runs again on another batch of new questions (at most 2 extra batches), with a larger share of the
failing kind of pair (more "get there" pairs if accuracy did not rise, more "don't drift up" pairs if old levels grew).

**Strengthen near the end:** when a type is within ~1 point of the original's accuracy, its stages get more questions and more
"don't stop too early" pairs (not a higher learning rate: that is what made earlier runs unstable). A type stops climbing once
it is back at the original's accuracy on the dev set.

**Control arm (review):** the same stages with answers from the ORIGINAL model instead of the current one (offline), same
counts. 2 seeds each.

## Evaluation (consult5 protocol)

- A **new dev set** of 512 questions from our own sources (not in training, not the 720, not the test), scored ×4, is the
  only thing used to choose; the checkpoint is fixed in advance (the end of the last stage).
- Both seeds of the chosen arm go to the full test; question-level paired bootstrap for the uncertainty; seeds reported
  separately.
- Success = both seeds above 50% saved and better than −1 on the full test.

## Risks

1. **Predicting the budget:** on math the model's own confidence is badly calibrated (says 0.98, right 0.80–0.86), so it
   may not learn which math questions need long thinking. Phase 0's predictability check measures this first.
2. **Cut artifacts:** an answer cut mid-sentence and closed by force reads differently from a natural stop; DPO on it may
   teach abrupt stops. Mitigation: cut at the nearest sentence or paragraph end below B.
3. **Noise in b\*:** 8 answers per question; the tolerance absorbs one answer; Phase 0 reports how often b\* changes when
   the 8 answers are split 4 + 4.
4. **IFEval responses** must not shrink (the known failure on counting rules).

## Open choices for the user

1. Tolerance: one answer in eight — DECIDED (user, 10 Nov).
2. Never-solved questions: no "give up" pairs; only the "a shorter cut was right" pairs — DECIDED (user, 10 Nov).
3. Phase 0 placement: GPU 1 after od1 (replacing od2), or on GPU 2 replacing cp1e (consult5 advised no more COPD)?
