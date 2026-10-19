# IDEA: stop thinking once the answer is settled

## Motivation

Qwen3-0.6B often keeps thinking after it already knows its answer. If we can find, inside the
model's own reasoning, the point from which its answer no longer changes, everything after
that point is waste. Training the model on its own traces cut at that point should teach it to
stop there, without the blunt length penalties that also cut useful reasoning.

## Method: settle-point pruning + self-distillation

1. **Sample.** Generate the model's own replies to training questions (GSM8K, ARC-Easy,
   CommonsenseQA), with the evaluation's sampling settings.
2. **Probe.** Cut a reply's thinking after k tokens, close it with `</think>` and
   `Final answer:`, and read the model's probability of the correct answer by teacher
   forcing. Doing this every 16 tokens gives a curve p(correct | first k thinking tokens).
   All the cuts of one reply are packed into a single forward pass: each cut's closing tokens
   attend only to their own prefix, at the positions they would have there.
3. **Prune.** For each correct reply, split the thinking into paragraphs. Cut after the
   paragraph from which p(correct) stays above 0.9. Before that point, try deleting
   paragraphs that did not raise p(correct), keeping a deletion only if the whole edited trace
   still reaches 0.9. Let the model answer naturally from the edited thinking twice and keep
   the trace only if both answers are graded correct. Per question, keep the shortest accepted
   trace; questions that never settle keep their shortest correct original reply, so the model
   is not taught to always stop early.
4. **Self-distil.** LoRA fine-tune (rank 16, 2 epochs) on those traces, loss on the completion
   only.

The shortening is learned by the model; no prompt or decoding change is used at evaluation.

## Measurements

All on one free Colab T4. The full evaluation (2,759 questions x 8 samples) does not fit,
so both models are scored on the same subset: the first 40 questions of every benchmark in
`benchmarks.py`'s fixed shuffle, 2 samples each, with the evaluation's prompts, sampling and
token caps (`subset_questions.py`). Treat the numbers as a development signal.

**Where the thinking goes (probe, 100 validation questions x 4 samples).** Among correct
rollouts, 94% settle (p(correct) reaches 0.9 and stays) and 32% of their thinking comes
after that point. A quarter of all rollouts hit the 4,096-token cap, 86% of them on
Skywork math, which the model answers correctly only 4% of the time.

**Round 1.** 450 training questions (GSM8K, ARC-Easy, CommonsenseQA; 150 each, 2 samples,
2,048-token cap) gave 311 pruned traces, 75% shorter than the originals (578 -> 145 tokens).
LoRA SFT on all of them:

| | tokens saved | accuracy change |
|---|---|---|
| BFCL (3 subsets) | +20% to +34% | -11.3 to +1.3 pp |
| IFEval | +17% | -1.3 pp |
| MBPP+ | +61% | -5.0 pp |
| SVAMP | +43% | -15.0 pp |
| **average (geometric mean of tokens)** | **+39.1% [29.1, 50.2]** | **-6.25 pp [-10.7, -1.5]** |

The shortening transfers to every benchmark, including code and function calling that were
never trained on, but accuracy drops. Reading the SVAMP failures shows why: the model
learned to **underthink**. It now often closes its thinking right after restating the
question and writes the arithmetic in the answer line ("Final answer: 52 + 38 = 90"), which
the grader rejects, or skips the step that mattered. 244 of the 311 traces were
multiple-choice questions the model knows almost at once, so the data taught "stop
immediately".

**Round 2** (same rollouts): keep one paragraph after the settle point (`--margin 1`),
require 4 correct natural answers instead of 2 (`--check-samples 4`), and cap training at
70 traces per dataset (`--max-per-source 70`).

ROUND2

## What did not work / limits

LIMITS
