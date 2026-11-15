# r2mid: next single-step offline DPO trial (draft for the user's approval, 15 Nov night)

## Goal of this trial, honestly stated

Find out whether ONE offline DPO step from the original can land between the two single steps we have measured, and
above our length-accuracy curve. Reaching the full goal (> 52% saved at better than -1) in one offline step is
unlikely. Every single offline step so far sits close to the curve (+0.3 to +1.25 above it). Only chains and online
DPO rose clearly above it (+2.4 to +3.1), because they train on the shortened model's OWN mistakes. A single offline
step only ever sees the original's answers.

| single offline step from the original | result | above curve |
|---|---|---|
| p1r1s0 (11 Nov) | full 43.4 / -1.5 | +0.26 |
| r2s1b step 160 / 180 (Path 1 step 1, shorten-heavy, cosine lr 2e-5) | full 49.0 / -1.7, 50.3 / -1.7 | +0.9 / +1.25 |
| r2one step 140 (protective, constant lr 1.8e-5) | Dev 35.7 / +2.42 (about 35 / -0.6 on a full test) | about 0 |
| (for scale) Path 1, 3 steps | full 51.5 / -0.84, 51.7 / -0.86 | +2.34 / +2.36 |

**Success criterion (written before training):**
- the full test lands >= +1.5 above the curve, OR
- it lands inside the goal box (> 50% at better than -1).

Otherwise the write-up reports: "a single offline step lands on the curve".

## What changes vs r2one (few things, each with a reason)

1. **Shorten share halfway between r2s1b and r2one.** Share of "think less" pairs per type (the rest are be-right,
   guards and loops, as in r2one):

   | type | r2s1b | r2one | **r2mid** |
   |---|---|---|---|
   | math | 55% | 45% | **50%** |
   | code | 70% | 58% | **64%** |
   | IFEval (of the non-counting part) | 50% | 62.5% | **56%** |
   | one function | 70% | 45% | **57%** |
   | several | 50% | 55% | **52%** |
   | no fit | 60% | 65% | **62%** |

   The counting-rule share (20%), loop pairs and "no chosen answer ever loops" stay as in r2one.
2. **Lower minimum thinking lengths.** r2one's absolute floors (math 100 tokens, one function 90) excluded about a
   thousand short right answers on tool calls alone. The floor becomes a share of the question's own median right
   thinking, with a small absolute minimum: math 40:0.42, code 60:0.38, IFEval 30:0.30, one function 30:0.40,
   several 30:0.38, no fit 20:0.30.
3. **Per-difficulty treatment inside each type** (the user's idea, 15 Nov). The original's solve rate on the question
   (8 answers) decides the pair kind:
   - **8/8, "easy":** "think less" allowed, but the chosen answer must keep at least 0.50x that question's median
     right thinking (stricter than the type floor). This is where most lost accuracy appears.
   - **4-7/8, "medium":** no "think less"; length-held be-right pairs only (a right answer near the median length
     beats a wrong one).
   - **0-3/8, "hard":** no "think less", no guards; be-right pairs only where they exist. Few questions, little to gain.
4. **Constant lr 1.8e-5, 200 steps, plain bf16, LoRA r16, beta 0.1** (as r2one). r2s1b's cosine schedule is not
   allowed under the user's rules.
5. **Data:** the same 12,100 test-matched questions and the original's stored answers that r2one used (no
   generation); a dry run on the Mac first shows the pairs per type, kind and difficulty before any GPU time.
6. **Scoring:** Dev-512 at steps 140 and 200 only (no Dev-SVAMP). Full tests only if Dev reaches >= 47% saved.

## A caveat on "the easy questions"

Our loss analyses show 45-67% of the lost accuracy on questions the original solved 8/8. But the noise reference
(the original against its own second set of 8 answers) shows the same pattern (67% on 8/8 for math), because
questions picked as "8/8" from 8 samples tend to score lower when re-sampled. Only the EXCESS over that noise is
real damage. For r2s3b on math: lost 4.90 against 3.20 for the noise reference, so about 1.7 points are real.
Item 3 targets that excess. It is a reasonable bet, not a sure fix.

## Predicted result (to compare after the run)

- Dev-512 at step 200: 42-47% saved, about +1.0 to +1.8.
- Full test, if it runs: 43-47% / -0.9 to -1.4, so +0.8 to +1.5 above the curve.
- Most likely outcome: better placed than r2one, still short of the goal.

## Other ways to give a single offline step the missing "easy-question" signal (not proposed here)

- **Truncate-and-re-answer negatives.** Cut the original's right answer early, let it answer from the cut, and use
  the wrong answers as rejected examples. This creates "stopped too early" negatives offline. It is close to the
  oracle-cut method of another entrant (basta) and to ThoughtFold, so it falls under the user's no-competitor rule
  unless the user decides otherwise.
- **Prompt-induced short answers.** The original answers under a "be brief" instruction; its short wrong answers
  become rejected examples and its short right ones chosen. This is close to basta's prompt-distillation step:
  same caveat.
- **Two offline steps instead of one** (Path 1 already shows 2-3 steps work): a cleaner story than six stages, but
  not one step.
