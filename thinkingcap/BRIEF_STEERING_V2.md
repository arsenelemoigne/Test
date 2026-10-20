# Brief: activation steering for shorter thinking, version 2 (paste-ready)

## Context

- **Challenge:** BottleCapAI ThinkingCap. Make Qwen3-0.6B write fewer tokens, thinking included,
  while losing at most 3 accuracy points against the untouched model.
- **Benchmarks** (fixed `evaluate.py`; prompt, sampling and token caps cannot change): SVAMP
  (math), MBPP+ (code), IFEval (instructions), BFCL (function calling).
- **Rules:**
  - Still a ~0.6B model that thinks before answering. An empty think is not allowed.
  - Loads in standard Hugging Face Transformers and vLLM with no custom code, so any change must
    live in standard weights or config.
  - No training on the evaluation benchmarks.
  - No prompt engineering at evaluation time.
  - Own, general ideas.
- **Setup used:** a 240-question subset (40 per benchmark) with 2 samples each. Accuracy is good to
  about ±4 points; use 4 or 8 samples for final comparisons. One Colab L4.

## What was tried (v1) and what happened

**Method**
- Took 97 questions (270 in a second try, which added HumanEval code questions) where the model
  had two correct answers, one short and one long.
- Took the residual stream after layer 14 (of 28), averaged over the thinking tokens of each
  answer.
- Direction = mean(concise) − mean(verbose).
- Stored as a v_proj bias in layer 14. Since attention weights sum to 1, a v_proj bias adds a
  constant W_o·R·b to the attention output. Solving for b injects exactly strength × v at every
  position, and the model still loads with no custom code. Unit test: the attention output shifts
  by exactly v.
- Strengths ×0.5 to ×8, with no layer search.

**Results**

| check | strength | effect |
|---|---|---|
| held-out maths/quiz | ×2 | 9% shorter and +6 points (fewer replies cut off at the cap) |
| held-out maths/quiz | ×4 | accuracy below the raw model |
| held-out maths/quiz | ×8 | model broken (maths 1.6%) |
| benchmarks | ×2 | about 0% tokens saved, −4 points (code −10, SVAMP −9, code replies longer) |
| benchmarks | ×4 | −15 points (MBPP+ 57.5% → 27.5%) |
| benchmarks, math + quiz + code direction | ×2 | +5.9% tokens saved, −3.1 points (MBPP+ 18.6% longer, −10 points) |

**Diagnosis: the mechanics were right, the methodology was weak.**

1. **The direction was not "concise".** Two natural samples to the same question differ in
   content (approach, mistakes, topic), not only in style. Averaging over whole answers captured
   "how these answers differ", not "be brief". Proper versions (e.g. ASC, activation-steered
   compression) contrast *the same reasoning* written verbose vs concise.
2. **It was applied everywhere: prompt, thinking and final answer.** That most likely broke code:
   the push disturbed the code being written, not just the thinking.
3. **One layer, and the strength was a guess.** No sweep over layers. The push size (norm 2.7–2.9)
   was never compared with the typical residual norm at that layer.
4. **No sanity check of the direction.** Nobody verified that projecting onto it separates concise
   from verbose thinking on held-out examples before using it.

## Proposal (v2)

**1. Clean contrast pairs**
- Same reasoning, two styles: a teacher (e.g. Qwen3-4B) rewrites each correct thinking trace of
  the 0.6B model into a concise version that keeps every step. An alternative is two traces
  aligned step by step.
- Covers maths, quiz, code and tool-like data that is not from the benchmarks (e.g. GSM8K, ARC,
  CSQA, HumanEval, synthetic function-calling).
- Keep only pairs where the concise version, fed back as the thinking, still gives the correct
  answer.
- Read activations position by position on the aligned steps, or on matched windows, rather than
  averaging whole answers.

**2. Validate the direction before using it**
- On held-out pairs, the projection onto the direction should separate concise from verbose (AUC
  well above 0.5), for every domain separately.
- Check that it does not simply encode the topic or answer correctness.

**3. Search the layer and the strength**
- For each layer, compute the direction and record its norm relative to the mean residual norm.
- Sweep a few layers (early, middle, late) and strengths expressed as a fraction of the residual
  norm (e.g. 2%, 5%, 10%).

**4. Act only on the thinking, within the rules**
- A weight bias acts at every position, including the answer.
- *Option 1:* choose a layer and direction whose effect on code and answer tokens is measured as
  small, and accept a global push.
- *Option 2 (complementary, "Option B"):* nudge only thinking-specific tokens.
  - Concretely: raise the logit of `</think>` and lower the logits of "Wait", "Hmm" and
    "Alternatively", through a last-layer bias pointed at those tokens' output directions.
  - About 1,000 extra numbers.
  - It is approximate because of the final RMSNorm, so measure the real logit shift.
  - It leaves code and answer tokens almost untouched.
- *Do not* untie the embeddings to get a separate output table. That adds about 0.16B parameters
  (about 25% of the model).

**5. Combine with training if it helps**
- A small steering push stored as a bias can be added on top of any trained adapter (LoRA stays
  separate).
- Test whether it adds to the RL or MVT gains.

## Testing protocol (method-agnostic; reuse for any candidate)

1. **Held-out check first** (cheap, about 10 min per candidate):
   - Use about 128 validation questions per source plus 40 held-out coding problems. None are
     benchmark questions.
   - Report accuracy and mean tokens per source, and the share of replies that hit the cap.
   - Report the share of empty or near-empty thinks; it must be 0.
2. **Benchmark subset**, only for candidates that pass step 1:
   - 240 questions with 2 samples to screen, then 4 or 8 samples to confirm.
   - Always compare against the raw model run on the same VM. The baseline drifts across
     machines: SVAMP tokens went 778 vs 940 between two runs.
   - Report the official average (tokens saved as a geometric mean, accuracy change in points,
     95% intervals) and the per-benchmark split.
3. **Per-benchmark guard:**
   - Reject if the average accuracy drops more than 3 points.
   - Flag any single benchmark down more than 5 points; code and SVAMP were the usual victims.
4. **Thinking vs answer split:** measure both, so that savings come from thinking and not from
   broken answers.
5. **Log every candidate** (settings, held-out numbers, benchmark numbers) in one table, keep the
   models in Drive, and keep one checkpoint per point on the length/accuracy curve. The organisers
   explicitly want several points.

## Reference numbers to beat (240-question subset, vs the raw model)

| method | tokens saved | accuracy change |
|---|---|---|
| SFT on MVT-cut easy questions, maths kept whole (other track) | +37.1% | −0.6 (8 samples) |
| RL: blend of a strong and a gentle run | +33.5% | −0.5 |
| RL, plain short-and-correct reward | +22.1% | −1.2 |
| steering v1, ×2 | ~0% to +5.9% | −3.1 to −4.0 |
