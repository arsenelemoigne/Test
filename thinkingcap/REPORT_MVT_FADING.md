# Night report: making Qwen3-0.6B think shorter (MVT, fading, marginal words)

Branch `mvt-fading`, own Colab L4 runtime, night of 19-20 October 2026.
Full numbers, settings and per-run notes: `EXPERIMENT_LOG.md` (section "Track 2").

## The goal, in one line

Make the small model write fewer tokens (thinking included) while losing **at most 3 points**
of accuracy against the untouched model, measured on the same 240 test questions.

## The answer

**Best result: "short on easy questions, full on math" (experiment 1b, math examples shown twice).**

| | tokens saved | accuracy change |
|---|---|---|
| 2 answers per question | 40.7% | -1.0 pts |
| **4 answers per question (more reliable)** | **39.1% [31.5, 47.5]** | **-1.0 pts [-4.2, +2.1]** |

The model writes about **40% fewer tokens** and stays well inside the 3-point limit.
On instruction following (IFEval) it is 39% shorter **and** better (+5.6 pts); function
calling (BFCL) is 14-25% shorter with no loss; math (SVAMP) is 18% shorter for -3.1 pts;
code (MBPP+) is 67% shorter but loses 8 pts, the weak spot.

A slightly bolder version (same recipe without showing math twice) saves **43.8%** for
**-2.1 pts** (4 answers per question), also within the limit. RECHECK256

## How we got there: the three experiments, simply

All three follow the same recipe: take the model's own correct answers to 450 practice
questions, shorten them, teach the model on the shortened versions, then test it.
They differ only in **how** the answers are shortened.

### Experiment 1: the Marginal Value Theorem cut

*The idea.* A bird leaves a berry bush when the berries come slower than its average rate,
counting the trip to the next bush. Here, the "berries" are the model's certainty that its
answer is right, measured after every paragraph of thinking, and the "trip" (c) is the
answer it still has to write. Stop thinking when one more paragraph adds less certainty
than it costs.

*What happened.* Cutting every example this way made answers 50-60% shorter but cost 6.5 to
7.6 points, with math collapsing (-21 to -24). Two lessons:

1. On math, the certainty of the exact number stays near 0% until the very last step, then
   jumps. My first rule only looked two paragraphs ahead and stopped at 0%; I fixed it to
   look at the whole remaining thinking. It still failed, so the cut point was not the issue.
2. On easy multiple-choice questions the model knows the answer **before** thinking, so any
   rule cuts them to one paragraph. Taught on those, the model stops thinking on math too.

### Experiment 1b: short on easy questions, full on math (your idea about task types)

You noticed that the cut worked on language tasks and broke only math and code. So: cut the
easy multiple-choice examples with the MVT, and keep the math examples **whole**. This is the
winner (table above). Showing the math examples twice made it safer still (-1.0 pts).

What did **not** help: keeping 2 paragraphs minimum in the easy examples (only 13% shorter),
and compacting the math examples (math fell again, -10). **Math examples must stay untouched.**

### Experiment 2: fading worked examples (paragraphs + filler words)

*The idea.* Like a teacher who gives fewer hints each lesson: cut a little (10% of the
paragraphs and 10% of the words, the filler ones by a standard filler detector), teach, then
cut a bit more, and so on.

*What happened.* Round 1: only 4.1% shorter, -3.2 pts (4 answers per question), on the edge.
Round 2: 14.3% shorter, -4.3 pts, outside. Gentle fading protects math much better than one
big cut, but the model takes up only a fraction of the shortening and slowly loses accuracy
elsewhere.

### Experiment 3: marginal token analysis (your idea: the model judges its own words)

*The idea.* For every word of the thinking, remove only that word and measure how much the
model's certainty in the right answer drops. Words that do not matter get removed first.

*What happened.* Round 1: the model's answers shrank **2.6x more** than with the generic
filler detector for the same amount of trimming (12.1% vs 4.6% shorter), and it stays within
the limit with 4 answers per question (5.3% shorter, -2.0 pts). So the model learns its own
deletions better. But round 2 broke it (-8.8 pts, answers even longer): too much telegraphic
thinking makes it hesitate, especially on code. It costs about 85 minutes of GPU to score
every word once.

## What we learned

1. **Where the waste is depends on the task.** On easy and language questions most thinking
   is waste; on math every step carries the answer. A single model can learn both habits if
   the training data shows both.
2. **One big cut teaches laziness; the fix is not a better cut point but keeping math whole.**
   Every run that cut the math examples in one go (4 runs, across two sessions) lost 6-8
   points, with math -21 to -24.
3. **Measure with enough samples.** With 2 answers per question the average moves by about
   +/-4 points; two conclusions flipped once rechecked with 4 answers.
4. **Code is the open problem.** We never train on code, and every method loses 5-10 points
   there even when the average holds.

## Suggested next steps

- Run the winner on the full evaluation (2,759 questions x 8 answers) before trusting it.
- Add code examples (kept whole, like math) to the training mix to protect MBPP+.
- Try experiment 3's word scoring on the easy examples only, with math kept whole.

## Honest caveats

- 240 questions, 2-4 answers each: a development signal, not the leaderboard score.
- The 4-answer rechecks share their first 2 answers with the first tests (same seeds).
- Training used 450 practice questions from 3 sources (GSM8K, ARC-Easy, CommonsenseQA).
