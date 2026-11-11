# Path 2 — RL on top of Path 1 (design, 11 Nov 2026; runs only after Path 1)

Designed by Gemini 4 Argon (reviews/consult7_path2/run/final_answer.md, $2.28) from Path 1's final plan, our verified RL study
(reviews/rl_length_methods_study.md), the competitor study (insight only, nothing copied), our log and rl_train.py. Not yet built
or checked in code. My caveats are at the end.

## Starting point

Path 1's final model (the 2-seed average of round 2 at its chosen step, merged), expected ~49–51% saved at ~−1. Path 2's job is
not to compress from scratch but to shave the remaining waste (+2.5 to +4.5 points of saving) on questions the model already
masters, without losing round 2's repaired accuracy. Never from round 1.

## Why our past RL barely moved (Gemini's diagnosis from the log and the code)

1. grA / grB / grC: the length signal was tiny next to the right/wrong reward (≈0.004 vs 1.0), wrong answers got a +0.1 format
   bonus, lr 2e-6 (a full-fine-tuning rate) on LoRA, and before 4 Nov the optimizer state was reset every 10 steps.
2. Dr. GRPO settings (no std division, fixed token denominator) made all-right groups — where the length signal lives — almost
   silent next to mixed groups; with --max-tokens-code 8192 the fixed denominator was very large.
3. gr2f: the cosine schedule fell to ~3e-6 by step 28, and no filtering of uninformative groups; starting from the original, a
   mild length penalty cannot compress 50% in 40 steps.
4. RL on already-shortened models (rc1-rl, drpo1, sb1/sb2) either lengthened back (correctness dominates) or collapsed to early
   stops (group-minimum targets with no floor).

## The reward (per question, from Path 1's own answers)

Path 1's model first answers ~2,400 unseen questions × 8; for each question:
- **Target B(q)** = 35th percentile of its correct answers' length (thinking only for IFEval; total tokens otherwise). A correct
  answer at or below B(q) gets the full 1.0 — no pressure to cut further.
- **Tail cap C(q)** = max(90th percentile, 1.35 × median) of its correct lengths, at most the type's 90th percentile.
- **Correct answer, group solve rate ≥ 7/8 (user, 11 Nov: raised from Gemini's 4/8; a question solved half the time passes
  7/8 by luck only ~4% of the time, vs 64% at 4/8):** 1.0 if ≤ B(q); 1.0 − α·min(1, (length − B(q)) / type mean length) between B and C;
  1.0 − α − 0.25 above C. α = 0.20 for math and no-fit, 0.35 for code, IFEval and function calls.
- **Correct answer, group solve rate < 7/8:** 1.0 (no length pressure: "just be right").
- **Wrong or cut-off:** 0; **wrong and stopped early** (thinking < half the question's median right thinking): −0.20.
- IFEval counting-rule questions: a correct answer whose response is < 90% of the usual response length loses its length bonus.
- No format bonus for wrong answers.

## Algorithm and settings

- Clipped policy gradient, on-policy (1 update per batch), clip 0.20 / 0.28 (clip-higher).
- **Advantages normalised over the whole batch** (not per group, no Dr. GRPO), loss divided by the actual tokens.
- Group 8, 32 informative questions per step (48 drawn; groups with no reward spread dropped).
- lr 1.5e-5, 4 warm-up steps, cosine to 5e-6 over 50 steps; KL 0.002 to Path 1's model; grad clip 0.5; LoRA r16 α32 on all
  linear layers; rollouts at the test's caps (4,096 / 8,192).
- 50 steps, adapters every 10; 2 seeds.
- Questions: solved ≥ 4/8 by Path 1's model (reference answers), in two bands: "mastered" (7–8/8, length spread ≥ 1.25×: they
  get the length pressure) and "shaky" (4–6/8: "just be right"). **Fixed proportion every step (user, 11 Nov):** each step's
  32 kept groups are 16 from each band (refilled from the same band when a group is dropped as uninformative), so the
  balance between "think less" and "be right" never drifts as the model changes; type shares
  as the test (math 600, code 600, IFEval 600, one function 285, several 145, no fit 170); never Dev-512, the 720 or the test;
  each question used once.

## Choice, cost, expected

- Score the 2-seed average at steps 20 / 30 / 40 / 50 on Dev-512, paired against Path 1's model. Eligible only if it saves
  ≥ 1.5 points more, its calibrated accuracy is ≥ −0.85, and it loses ≤ 0.35 points vs Path 1 (paired). Half-strength merge as
  a fallback if a step overshoots. One full test on the chosen checkpoint.
- ~4 hours wall-clock on the two GPUs (~8 GPU-hours) + one full test.
- Gemini's forecast: Path 1 + Path 2 at 52–55% saved / −0.6 to −0.95 (treat as optimistic, like its earlier forecasts).
- Cheap first test (~2 GPU-hours, defined in its §4): run the reward on Path 1's model for a short block and check the gate.

## Code changes needed (its §3)

rl_train.py: a new `--reward p2budget` (per-question B(q) / C(q) from a reference file, the toggle, the early-stop −0.20, the
IFEval response guard), batch-wise advantage normalisation, actual-token loss denominator, dynamic sampling that drops groups
with no reward spread, clip-higher; thinkprune stage: build the per-question reference file from Path 1's answers, the 50/50
difficulty mix, Dev-512 screens. To be built and dry-run tested while Path 1 runs.

## My caveats (Claude)

1. Its question sources say "no MetaMath / no math1 / math2 / nofit4": outdated — you allowed the public datasets on 11 Nov;
   Path 2 should use them like Path 1 (SVAMP-like math).
2. Part of its diagnosis is weak: with the Adam optimizer, a constant scale on the loss (the "fixed denominator dilution") mostly
   cancels out; the stronger points are the tiny length signal, the cosine decay to near zero, and starting from the original.
3. Its targets are relative to Path 1's own lengths (35th percentile), so Path 2 can only shave a few points: it cannot rescue a
   Path 1 that lands far below 50%. It is a finishing stage, not a second chance.
4. Nothing in the RL study shows LoRA RL at 0.6B reaching these numbers; the closest evidence (Qwen3-0.6B, full fine-tuning)
   cut 40% from a long-thinking start. Expect the +2.5 to +4.5 points to be the optimistic end.
