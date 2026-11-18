# Draft: the second half of the article (for review, 18 Nov)

Layout (no "Part I / II" labels in the article): two big headlines, larger than section titles.
- Before section 1: **Making a small model think less** (sections 1–11, unchanged).
- Before section 12: **Did it get smarter?** (the text below; replaces the current sections 12–13; section 14 on legal AI
  becomes the last section, renumbered 19).

---

# Did it get smarter?

*(Replaces the current sections 12–13. Numbers: original Qwen3-0.6B vs the best model of Part I (online DPO, round 16), full test, 2,759 questions ×
8 answers each. [GPU] = result arriving from tonight's run; the sentence is written for it and the number goes in.)*

---

The first half ends with a model that writes 54.8% fewer tokens for 0.78 points of accuracy. On average, almost nothing was lost.
That raises the obvious question. A model that reaches the same answers with half the thinking sounds like a model that
thinks better. Did it get smarter, or did it only get shorter?

BottleCap states the goal of this kind of work as spending "fewer tokens getting to the answer, and change nothing else"
[23]. The challenge's score checks the first half. This part checks the second, with seven measurements.

## 12. Question by question, the answers move

The average hides movement. Of every 100 answers, 70 are right for both models, 21 wrong for both, 5 go from right to wrong
and 4 from wrong to right. The average moves by less than one point; 9 answers in 100 swap.

Some of that swapping is chance: run the original twice and it disagrees with itself. So I measure two things per question,
from its 8 answers with each model:

- **Net change**: how many questions' worth of accuracy was gained or lost. Over the whole test: **−23 questions (−0.85
  points)**.
- **Excess churn**: how much the answers moved in either direction, beyond what rerunning the same model produces (computed
  exactly: if both models were the same, any split of a question's 16 answers into 8 and 8 would be equally likely).
  **+73 questions [61, 85]**. The check that the measure is fair: the original against a second run of itself gives +2
  [−9, +11], zero within noise.

| Model | Tokens saved | Net change | Excess churn |
|---|---|---|---|
| Gentlest run (offline) | 42% | +7 questions | +54 |
| Best model | 55% | −23 | +73 |
| Most aggressive run (offline) | 63% | −58 | +95 |
| Original, thinking off | 94% | −800 | +717 |

The more a model saves, the more its answers move. Per task, the movement is real everywhere (math +21, code +6,
instructions +12, one function +13, several functions +12, "no function fits" +9 questions beyond chance), including where
the average did not change at all: on "several functions" the net change is exactly zero, yet 12 questions' worth of answers
moved each way.

*(This replaces the "loss beyond chance" of the earlier version, which corrected losses and gains separately and so
mis-counted questions that moved only one way.)*

## 13. It did not forget how. It became less consistent

If the model had lost skills, it would fail questions it used to solve at all. It does not. The share of questions it gets
right **at least once in 8 tries** is unchanged: 88.2% → 88.1% overall, 96.8% → 96.4% on math. What changed is the share it
gets right **every time**: on math, 70.1% → 61.7%.

So the trained model still knows how to solve the questions it lost. It is less reliable on them. Section 9 shows where: on
multi-step math questions, where one skipped check is enough to change the number.

## 14. It cut steps, not words

Thinking length is the number of steps times the tokens per step. A model can get shorter in two ways: drop whole steps
(trimming), or write the same steps in fewer tokens (condensing). Only the second would mean it reasons more compactly.

To tell them apart, I split each answer's thinking into steps (sentences), and matched each step of the trained model to the
step of the original that says the same thing, on the same question [method note].

- **At least 90% of the saving is trimming.** Main estimate 98%, at every matching threshold, in every task; four other
  matchers that use no AI model at all give 91–99%.
- **The steps it keeps are unchanged**: 17.0 → 16.4 tokens per matched step. Steps per answer: 55 → 22.
- **Could condensing hide in the trimming number?** A step rewritten both shorter and in different words would not match.
  I checked those unmatched steps: they are as long as the original step they most resemble (0.95×), and carry *less*
  content per token, not more. Even crediting all of them as condensed, it adds at most 2% of the saving. Several steps
  folded into one: 6%.

## 15. What it cut was re-checking an answer it already had

If the model learned when to stop, the thinking it dropped should come after the answer was already decided. For each
answer, I located the first point where its final answer appears in its own thinking (the final number on math; the
function and every argument on tool calls; the first draft of the function on code).

| Task | Thinking after the answer is there: original → trained | Share of the cut that was post-answer thinking |
|---|---|---|
| Math | 54% → 37% | 66% |
| One function | 49% → 31% | 64% |
| Several functions | 52% → 33% | 67% |
| Code | 51% → 29% | 74% |

The original spends about half of its thinking after it already has the answer; two-thirds of what training removed is that
post-answer thinking. On code, the original writes the whole function inside its thinking and then copies it out (99.6% of
answers); the trained model does it in 69%. Hesitation words ("wait", "let me verify") fall from 2.95 to 2.45 per 100 words.

[GPU] Text is a proxy: a number can appear before the model has actually committed to it. The causal test cuts each answer at
5%, 10%, … 95% of its thinking, forces the answer, and finds the earliest cut from which it stays right: on [X]% of answers
the original was already right [Y]% of the way in.

## 16. The steps it kept are not denser

Same length could still mean more content. On the matched steps, per 100 tokens, original → trained: numbers 6.7 → 6.8,
operators 2.4 → 2.4, code 3.0 → 3.1, function arguments 1.1 → 1.0. No change, in any task. One real difference: on math the
trained model gets to the numbers sooner (numbers in the first 100 thinking tokens 4.7 → 7.1 per 100): it spends less of its
opening restating the question.

[GPU] Information per token: both models' thinking read by the original model as a fixed reader. Predictable tokens carry
little new information. [Result: bits per thinking token X → Y; first 100 tokens X → Y.]

## 17. The tokens it cut were worth very little

The first 259 thinking tokens of the trained model buy 28.2 points of accuracy over thinking off; the 411 tokens it no longer
writes bought 0.85. A cut token was worth about **0.2 points per 100**, twenty times less than the average thinking token. Per
task: math 0.7, code 0.16, one function 1.2, and on instructions and "no function fits" less than nothing: removing them
raised accuracy.

[GPU] That comparison sets two different models side by side. The direct test: cut the **original** model's own thinking where
the trained model stops (per question), force the answer. [Result: it loses X points], so the original's last tokens were
[worth almost nothing / worth something the trained model learned to do without].

## 18. One surprise: the model itself improved a little

With thinking switched off in both, the trained model answers 2.1 points better than the original. Hint or effect? The same
check on other trained models: the second online run is also better without thinking (+1.9); the two offline runs are not
(−0.3 and −3.7). Online training, where the model learns from its own fresh answers each round, made the model itself a
little better, beyond its thinking.

## So: shorter, not smarter

Training taught the model when to stop. It drops re-checks of answers it already had, keeps every step it writes the same,
and is about half as expensive to get an answer from. Question by question it is not more capable, and slightly less
consistent. The honest claim is answer efficiency, not reasoning efficiency.

For evaluating any method that shortens reasoning, this suggests reporting three numbers next to the saving: excess churn,
the share of questions solved every time, and how much of the cut was post-answer thinking.

---
### Method note (footnote or appendix)
Steps = sentences (fragments under 3 words joined). Matching: all-MiniLM-L6-v2 embeddings, one-to-one, best pairs first,
cosine ≥ 0.75 (robust from 0.65 to 0.85). Saved tokens = trimmed + condensed − added, exactly. Reported on 1,600 answer pairs
(200 math, 200 code, 100 per other task, two answers per question); the model-free re-check covers all 22,072 pairs.
