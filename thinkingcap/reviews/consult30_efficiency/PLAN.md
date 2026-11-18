# Did it get smarter, or only shorter? Analysis plan (18 Nov 2026)

## 0. Where we stand

The best model (online DPO od5c, round 16) writes 54.8% fewer tokens than the original Qwen3-0.6B for −0.78 points of accuracy
on the full test (2,759 questions × 8 answers, six task types: math, code, instructions, one function, several functions,
"no function fits"). The average barely moves. The question this plan answers is the one the article's last sections ask:

**We got the model to think less on average. Did it get smarter?**

"Smarter" is ambiguous, so the plan splits it into four questions, each with its own measure. The order of the article
section follows them.

| # | Question | Measure | What we already have |
|---|---|---|---|
| 1 | Does it still solve every question the original solved? | Questions lost beyond chance | Yes: 48 questions lost (best model), 24 / 76 for the gentlest / most aggressive runs, 758 for thinking off; no trained model passes on 5 of 6 tasks |
| 2 | Where did the saved tokens come from: whole steps dropped (trimming) or the same steps written shorter (condensing)? | Step matching | Yes (CPU, 1,600 answer pairs): ~98% trimming, 1–2% condensing, every task |
| 3 | How far past the point where the answer is settled does each model keep thinking? | Overthinking after the answer | Partly: on math, 51% of the original's thinking comes after the correct number first appears, 34% for the trained model |
| 4 | What were the cut tokens worth, and is the cut in the right place? | Marginal value of a thinking token, per task | Partly: on average 0.21 accuracy points per 100 cut tokens; math 0.7, code 0.16, instructions and "no function fits" below 0 |

The working hypothesis (the user's): the model learned **when to stop**, not how to reason better per token. That predicts:
mostly trimming (2), less thinking after the answer (3), cut tokens of low value except where steps were still needed (4),
and no per-question gain (1). It is consistent with the known result that letting a model think longer does not make it
smarter.

## 1. Questions lost beyond chance (renamed from "intelligence density")

**Why rename.** The old name suggested tokens; the measure has none. It asks only: does the new model still solve what the
original solved? "Intelligence density" is used elsewhere for intelligence per token or per unit of compute, which is our
questions 2 and 4, not this one.

**How it works.** Each model answers each question 8 times. For each question, the 16 answers (8 + 8) are split; if the two
models were the same, every split into two groups of 8 would be equally likely (exact hypergeometric null). That gives the
drop chance alone would produce; only the excess counts. Summed over 2,759 questions it reads "it lost the equivalent of N
questions". A task passes if its loss beyond chance stays under 0.5 points even at the top of its 95% interval.

**Why it matters.** The average can stay flat while answers swap: of every 100 answers, 70 stay right, 5 become wrong, 4
become right, 21 stay wrong. The average sees −1; this test sees which questions broke.

**To extend.** Per task and per model for 4 models (original, best, second seed, most aggressive), plus a control: the
original against a second run of itself (should be ~0). Check whether losses concentrate on hard questions (by the
original's solve rate) or on long ones.

## 2. Trimming vs condensing

**Reasoning.** Thinking length = number of steps × tokens per step. Trimming lowers the first, condensing the second. If the
model learned to reason better per token, condensing should be large; if it learned to stop, trimming should be almost all.

**Method (trim_condense.py).** Split each thinking trace into steps (sentences; fragments under 3 words join the next). Embed
every step (all-MiniLM-L6-v2) and match new steps to original steps one-to-one by cosine similarity (best pairs first,
threshold 0.75, sensitivity 0.65 / 0.85). Then exactly:

    original tokens − new tokens = trimmed + condensed − added
    trimmed   = tokens of original steps with no match
    condensed = Σ over matched pairs (original step tokens − new step tokens)
    added     = tokens of new steps with no match

Reworded steps sometimes fall below the threshold and count once as trimmed and once as added, so the stable figure is
**net trimming = trimmed − added**.

**Result so far (1,600 pairs).** Net trimming 98–99%, condensing 1–2% at every threshold and in every task (max 4%). Steps per
answer 54.9 → 21.5; tokens per matched step 17.0 → 16.4.

**To extend.** All 22,072 answers, the 4 models; matching with a second embedding model (bge-small) to show the result does
not depend on the matcher; a manual check of 50 matched and 50 unmatched steps.

**Information per token (the user's "higher carrying value tokens").** If condensing happened, each remaining token should
carry more information. Measure: mean surprisal (bits per token) of thinking under a fixed reader, the ORIGINAL model
(GPU only). Also surprisal of the steps that were cut vs those kept: were the cut steps predictable (repetition, re-checks)?
The gzip proxy rises (8.3 → 12.0 bits/token) but mainly because repeated passages compress well, so it is not used as
evidence.

**Step types.** Label each step: compute (numbers, code, a function call), check ("let me verify", re-doing a step), restate
(repeating the question or an earlier step), hesitate ("wait", "hmm", "alternatively"). Which kinds were trimmed? The
hypothesis predicts checks and restatements go, compute stays.

## 3. Overthinking: thinking after the answer is settled

**Reasoning.** If the model learned when to stop, the share of thinking written after the answer is already settled should
fall. This is the most direct test of "learned to stop".

**Method A (no GPU, approximate).** For math, find the first point where the final answer's number appears in the thinking;
for tool calls, the first point where the final call's function name and arguments appear; for code, the first point where
the final function's signature and body appear. Share of thinking after that point, original vs trained. Already done for
math: 51% → 34%.

**Method B (GPU, exact; the "early-stop probe").** For a sample of questions per task, cut the thinking at 10%, 20%, …, 100%
of its length, close the thinking and let the model answer. Accuracy at each cut gives each answer's settle point (the
earliest cut from which it stays right). Report: share of thinking after the settle point, per task, original vs trained.

## 4. Marginal value of a thinking token, and the equation

**Reasoning.** The challenge is an optimization: choose how much to cut on each task to save the most tokens while losing at
most 1 point in total. The standard condition for the optimum (Lagrange) is that **every task's last cut tokens have the
same marginal value**: keep cutting a task while its next 100 tokens cost less than one common price, and set that price so
the total loss is exactly 1 point. The target "more than 50% fewer tokens at better than −1" corresponds to roughly
**0.27 points per 100 cut tokens** on average.

**What we have.** Average value of the cut tokens 0.21 per 100 (inside the bar). Per task: math 0.7 (over-cut), code 0.16,
instructions and "no function fits" below 0 (under-cut: removing more raises accuracy). So the model is not at the optimum.

**To extend.** The early-stop probe (3B) gives each task's accuracy-against-thinking-tokens curve for both models; its slope
is the marginal value at every budget. Find, per task, the budget where the marginal value crosses the bar, and the saving
and loss that cut would give: a recipe target per task. Also the weighting: the challenge averages tokens geometrically over
four benchmarks, which changes each task's weight in the equation.

## 5. Data for the review

- Full-test records, one line per answer (2,759 × 8 = 22,072 per model): right, thinking tokens, response tokens; for the
  original (base), the original with thinking off (nothink), the best model (od5c16) and 12 offline models.
- Full-text thinking traces (the answers themselves) for base, nothink and od5c16.
- trim_condense.py and its report on 1,600 pairs.
- The article as published (text), the measures we use, the lab notebook.

## 6. Open questions for the reviewer

1. Is the four-question frame right, and are the measures sound? What would a careful reviewer attack?
2. Is the trimming/condensing method biased (sentence splitting, the embedding matcher, the threshold, splitting tokens by
   characters)? A better method?
3. The marginal-value equation: is the Lagrange framing right for this score (geometric mean of token ratios, arithmetic
   mean of accuracy changes)? Compute the per-task optimum from the data we have.
4. Overthinking: is Method A good enough without the GPU probe? What are its biases per task?
5. Which GPU analyses (surprisal, early-stop probe, other models) are worth the time, in what order?
6. What should the article claim, and what should it not claim?
