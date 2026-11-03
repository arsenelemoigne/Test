# The test (what you are scored on; you never see its questions)

Source: the official ThinkingCap benchmark (github.com/BottleCapAI/ThinkingCap-Test). Its goal: "generate fewer tokens
(thinking included) than the original Qwen3-0.6B while keeping the same accuracy".

## Composition: 2,759 questions, each answered 8 times
| part | questions | what it asks |
|---|---|---|
| SVAMP (type "svamp") | 1,000 | grade-school arithmetic word problems with a single numeric answer |
| MBPP+ (type "mbpp") | 378 | write a Python function from a one-sentence description, keeping the required name; hidden unit tests |
| IFEval (type "ifeval") | 541 | a response that obeys one to three checkable constraints (25 kinds in 9 groups: keywords, length, formatting, casing, language, ...) |
| BFCL simple (type "bfcl_simple_python") | 400 | one function is offered; the request needs exactly one call to it |
| BFCL multiple (type "bfcl_multiple") | 200 | two to four functions are offered; pick the right one and fill in its arguments |
| BFCL irrelevance (type "bfcl_irrelevance") | 240 | the offered function does not fit the request: the right answer is no call at all |

Invented examples of the style (NOT test items):
- svamp: "A baker made 48 muffins and sold 3 boxes of 9. How many muffins are left?"
- mbpp: "Write a function `count_upper(s)` that returns the number of uppercase letters in a string."
- ifeval: "Describe a lighthouse in exactly 3 sentences, without using any commas."
- bfcl: a JSON list of offered functions with parameters, then "What's the weather in Lyon tomorrow?"

## Protocol (fixed)
Chat template with thinking on; sampling temperature 0.6, top_p 0.95, top_k 20; at most 8,192 new tokens for MBPP+
and 4,096 for the others (thinking + answer); an answer cut at the limit counts as wrong. Weights load in bfloat16.

## Scoring
Per part: tokens saved (all generated tokens, thinking included) against the original Qwen3-0.6B, and the accuracy change
in points. The average counts the four benchmarks equally, the three BFCL parts pooled as one: accuracy change with the
arithmetic mean, tokens saved with the geometric mean of the per-benchmark token ratios.

## Rules
Start from Qwen3-0.6B (fine-tuning, LoRA, minor changes allowed; still a ~0.6B model that thinks before it answers).
Never train on the evaluation benchmarks. Keep the evaluation settings. No prompt tricks, no turning thinking off.

## What the practice / training data stand for
| training source | imitates |
|---|---|
| gsm8k_matched, gsm8k_easy, metamath_easy, metamath_sv | svamp |
| kodcode | mbpp |
| ifeval_like, ifeval_built | ifeval |
| toolace, hermes_tools, glaive_tools (simple / multiple / irrelevance categories) | the three BFCL parts |
