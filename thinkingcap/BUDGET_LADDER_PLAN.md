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

## Phase 1 — training (if Phase 0 passes)

**Pairs** (from the ladder answers themselves; all model-written, no teacher):
- **freeze / over-thinking:** chosen = a right answer at b\*(q); rejected = a right natural answer whose thinking is at
  least 1.5× longer (and ≥ b\* + 64 tokens). For b\* = 0 questions, chosen = a right no-thinking answer.
- **under-thinking guard:** chosen = a right answer at b\*(q); rejected = a wrong answer cut one rung lower (the model
  stopped too early). This is the guard COPD Run C lacked.
- **IFEval:** budgets count thinking only; on counting-rule questions the chosen answer's final response must be at least
  as long as the rejected one's (the existing --count-rules logic); no pair shortens a final response.
- Shares per part follow the test (BFCL 400/200/240); half freeze, half guard; at most one pair of each kind per question.

**The ladder as a curriculum (the user's "gradually"):** stage k trains on questions with b\* ≤ rung k, plus an anchor share
(~30%) of questions from the earlier stages' budget classes; one stage per rung that holds enough questions (expected 4–5
stages).

**Against memorising (user, 10 Nov):** every pair is used once (one pass, no epochs); the anchor share uses NEW questions of the
earlier budget classes, never the same pairs again; each budget class holds many different questions, so the lesson is "this
kind of question needs about this much", not "this question = 256". Check: training-set gains without dev-set gains = memorising;
the dev set's questions are never trained on.
Control arm (the review's advice to change one thing at a time): the same pairs all at once, same total steps.

**Training:** LoRA r16 / alpha 32, DPO beta 0.1, no SFT term, lr 1e-5 constant after 2 warm-up steps, batch 16, fp32
weights + fp16 compute, reference = the original Qwen throughout, ~20 steps per stage (~100 steps total), 2 seeds (0, 1).
Optional later: re-run the ladder with the current model between stages (fresh rungs, the review's "fresh" arm).

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
