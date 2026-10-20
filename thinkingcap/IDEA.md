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

Rerun on an L4 in bf16, with its own base run on the same subset:

| | tokens saved | accuracy change |
|---|---|---|
| BFCL (3 subsets) | +32% to +39% | -10.0 to -3.8 pp |
| IFEval | +40% | +3.8 pp |
| MBPP+ | +86% | -6.3 pp |
| SVAMP | +49% | -23.8 pp |
| **average** | **+59.6% [51.2, 67.7]** | **-8.0 pp [-12.2, -4.0]** |

More shortening, no accuracy recovery: SVAMP gets worse. Imitating pruned traces teaches
"short" faster than it teaches "correct". The TokenProbe paper reports the same thing (SFT on
compressed traces degenerates), and both it and ShorterBetter train with RL from the base
model, where a wrong answer always loses reward. That is the next stage.

**Round 3: RL from the base model** (`rl_train.py`). Per question, 8 sampled replies.
Reply-level reward from ShorterBetter: correct replies are rewarded for being close to the
group's shortest correct length L* (the group mean when none is correct), wrong ones never
beat a correct one. Token-level pressure from TokenProbe: each reply's log-probs are z-scored,
the best 10-token windows are "core", and in replies over a budget (core length x a scale
annealed from 3 to 1.5) the other tokens lose their KL anchor to the base model and are pushed
down. A variant marks redundant tokens with our own probe instead (everything after p(correct)
settles).

RL loop 1 (12 steps x 32 questions x 8 samples from the base model, GSM8K/ARC/CSQA, KL
variant, alpha 0.5, lr 2e-5, LoRA r16, ~1h on an L4):

| | tokens saved | accuracy change |
|---|---|---|
| BFCL (3 subsets) | +4% to +12% | -5.0 to +8.8 pp |
| IFEval | +16% | -1.3 pp |
| MBPP+ | -18% (longer) | -3.8 pp |
| SVAMP | +13% | -1.3 pp |
| **average** | **+5.7% [-4.0, 14.9]** | **-1.2 pp [-4.8, 2.7]** |

Accuracy is preserved (within noise), unlike both SFT rounds, but 12 steps barely move the
length: the published runs use hundreds of steps.

**Ablation (20 steps each, from the raw model, same seed and questions; GSM8K, Skywork, ARC,
CommonsenseQA balanced).** Each run adds one ingredient to the previous one:

| | tokens saved | accuracy change |
|---|---|---|
| A: ShorterBetter length reward only | **+19.6%** [8.8, 29.5] | **+0.3 pp** [-3.0, 3.7] |
| B: A + TokenProbe-KL | +6.1% [-6.5, 17.8] | -0.3 pp [-4.6, 4.0] |

On a held-out check of the training subjects (128 validation questions), A and B are the same:
45% -> 60% accuracy and 1,065 -> ~780 tokens. TokenProbe's KL anchor to the base model keeps the
model's behaviour unchanged where it was not trained (code, function calling, instructions), so
the length gain does not transfer. The plain reward transfers: SVAMP -30% tokens at equal
accuracy, BFCL -12 to -22% with higher accuracy, IFEval -26%.

A training-free steering vector (concise-minus-verbose direction from 97 pairs of correct
replies, stored as a v_proj bias so the model still loads with no custom code) shortens maths
replies a little at strength 2 but costs 4 points on the benchmarks, 15 at strength 4.

## What did not work / limits

- **Code does not shorten.** MBPP+ replies stay ~1,650-1,900 tokens in every run: no coding
  question is in the training mix. `code_tasks.py` adds HumanEval with a unit-test reward.
- **Imitation underthinks** (both SFT rounds, finding above).
- **The subset is small.** 240 questions x 2 samples gives roughly +/-4 points on accuracy and
  +/-10 points on tokens saved; the full evaluation (2,759 x 8) is the real test.
- **More pressure costs accuracy where it was not trained** (loop 3: +24% at -6 points, mostly
  code and SVAMP).
