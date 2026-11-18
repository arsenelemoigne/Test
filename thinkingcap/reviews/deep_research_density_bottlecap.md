# Deep research: intelligence density vs BottleCap evaluation (18 Nov 2026)

## Summary

BottleCap's own wording sets a stricter goal than the one they score. The ThinkingCap challenge (github.com/BottleCapAI/ThinkingCap-Test) scores two things: tokens saved, as 1 − (r1·r2·r3·r4)^(1/4), and the change in average accuracy, each with a 95% interval that resamples whole questions paired against the baseline. As a rough guide, a good result shortens by more than 20% and gives up no more than about 0.2 points per 1% saved. Their blog posts state the aim more strictly: "spend fewer tokens getting to the answer, and change nothing else" (Gemma-4-26B post, 23 Oct 2026). They also concede that "A raw average can't separate 'thinks more briefly' from 'bailed on the tough ones'" (Qwen3.6-27B post, 13 Jul 2026). Yet every check they run is at the average level. They pair questions only for token counts, and accuracy is judged by benchmark means across seeds. In their own Gemma result, a −0.15pp average covers an AA-LCR loss of −4.00pp, which they call "the one real cost in the suite".

None of the existing metrics combines all four of your properties: scored per question, corrected exactly for chance using repeated samples, a pass/fail verdict per task, and losses counted without netting against gains. Outcome efficiency (Chen et al.), ThoughtTerminator's overthinking scores and LLMThinkBench's O-Score are token or aggregate measures with no per-question correction for chance. Negative flip rate (Positive-Congruent Training, MUSCLE, Sheng & Lu 2026) counts losses only, question by question, but from one sample per model and with no subtraction of chance. Yet PCT itself shows that two models differing only in random seed still produce about 5.2% NFR. The closest published analogue is Cacioli's Reliable Change Index for LLMs (arXiv 2604.27405, Apr 2026). It samples each item 10 times per model and flags changes beyond ±1.96. But it uses a pooled normal-approximation standard error rather than an exact exchangeability null, and it reports churn rates rather than a pass/fail verdict per task.

This verification round did not confirm anything on Trajectory's "intelligence density" (Yap, Sept 2026), G-Pass@k, accuracy-at-fixed-budget curves, or formal equivalence tests in ML evaluation. Treat those parts of the question as unanswered.

Recommended pitch: "BottleCap's stated goal for ThinkingCap is to spend fewer tokens getting to the answer and change nothing else, and your own posts note that a raw average cannot tell 'thinks more briefly' from 'bailed on the tough ones'. Matched Δ% already closes that gap for tokens. Intelligence density closes it for accuracy. Each question is answered 8 times by both models, the way the evaluation already runs. Under the null that the two models are the same, the 16 answers are exchangeable, which gives the exact drop each question would show by chance. Only drops beyond chance are summed, and gains elsewhere are not allowed to offset them. A benchmark passes when the upper end of the 95% interval of this 'loss beyond chance' stays under half a question. It needs no extra generations: it reuses the 22,072 responses a full run already produces. It turns 'unchanged accuracy' from a statement about averages into a statement about the questions themselves, the 'change nothing else' you set as the target. On my submissions it shows what the average hides: loss beyond chance grows with saving, from about 24 questions at 42% saved to 76 at 63%, so it is the measure that would tell a team which checkpoint actually thinks more efficiently rather than just answering more cheaply." Add one honest line on its limits: with 8 samples per model, a single question can only be flagged when the drop is large; small tasks give wide intervals; and the 0.5-question threshold is a choice that should be calibrated by running the baseline against itself with a different seed.

## Findings

- **ThinkingCap challenge scoring is set at the average level. Tokens saved = 1 − (r1·r2·r3·r4)^(1/4) over four equally weighted benchmarks: MBPP+ 378, BFCL 840 (its 3 subsets pooled), SVAMP 1,000 and IFEval 541, for 2,759 questions. Accuracy change = the arithmetic mean of the per-benchmark accuracy differences, in points. Truncated answers count as wrong. Each question is answered 8 times (22,072 responses) at temperature 0.6, top-p 0.95, top-k 20. Both numbers carry 95% intervals from resampling whole questions, paired against the baseline. Shortening must live in the model: no reliance on prompting, no thinking-off.** (high, vote 3-0)
  - Sources: https://github.com/BottleCapAI/ThinkingCap-Test
  - Evidence: All of this comes from the official README and code (recipe.py: SAMPLES = 8, SAMPLING; evaluate.py prints 'Intervals are 95%, resampling whole prompts'). Votes 3-0 across four claims. The README states no per-question reliability or flip criterion.

- **The stated bar is a trade-off rate on average accuracy, not a reliability criterion per question. As a rough guide, a good result shortens by more than 20% and gives up no more than about 0.2 accuracy points per 1% of tokens saved. They also ask for several checkpoints to see how the method trades length for accuracy.** (high, vote 3-0)
  - Sources: https://github.com/BottleCapAI/ThinkingCap-Test
  - Evidence: README lines 261-263 and 273. It is worded as a 'rough guide', not a hard threshold. Applied to your 54.8% saving, the guide would tolerate about 11 points of loss, so −0.78 sits well inside it.

- **BottleCap's stated intent is stricter than 'same average'. Qwen3.6-27B post (13 Jul 2026): 'One of our core aims at BottleCap is efficiency in AI'; 'do not try to make the model smarter or teach it new capabilities'; 'The only thing we wanted to change was the amount of computation spent reaching an answer'; and they aim to preserve 'knowledge, reasoning ability, answer quality'. Gemma-4-26B post (23 Oct 2026): 'The goal is to spend fewer tokens getting to the answer, and change nothing else.' Read literally, 'change nothing else' rules out swaps of questions, which supports a non-inferiority check per question. That reading is your inference, not something BottleCap says.** (high, vote 3-0 / 2-1)
  - Sources: https://bottlecapai.com/post/thinkingcap-qwen3-6-27b/, https://bottlecapai.com/post/thinkingcap-gemma-4-26b/
  - Evidence: Verbatim first-party quotes. One claim voted 3-0, the 'supports non-inferiority' claim 2-1, because that step is an interpretation. Both posts are about BottleCap's own large models, not the scoring of the 0.6B challenge. 'Answer efficiency vs reasoning efficiency' is your framing: BottleCap says it preserves reasoning, not that it improves it.

- **BottleCap already knows that averages can mislead, and that sampling is noisy. They fixed the token side by pairing and left the accuracy side as averages. They write 'A raw average can't separate "thinks more briefly" from "bailed on the tough ones"' and introduce matched Δ%, the paired per-question difference in thinking tokens. They also note 'At the recommended sampling temperature of 1.0, output quality and length can vary substantially between runs', so they use 5 seeds, significance tests, and mean ± 95% CI across seeds. Accuracy 'unchanged' is judged by benchmark means against the spread between seeds ('how far the answer moves when only the sampling seed changes'), with no per-question accuracy analysis.** (high, vote 3-0)
  - Sources: https://bottlecapai.com/post/thinkingcap-qwen3-6-27b/, https://bottlecapai.com/post/thinkingcap-gemma-4-26b/
  - Evidence: First-party posts, three claims each voted 3-0. Δ Acc is a plain difference of means. Matched Δ% applies no correctness filter. This is the strongest hook for your pitch: you are carrying their own matched-pairing idea over from tokens to correctness.

- **BottleCap's own headline result shows an average hiding a real loss. ThinkingCap Gemma-4-26B-A4B claims 'up to 69% and on average 39% less thinking, at unchanged accuracy', with a −0.15pp average over 12 benchmarks. Four gaps exceed their intervals: IFBench +1.83, GPQA-Diamond −0.96, MMMLU −0.66 and AA-LCR −4.00, which they call 'the one real cost in the suite'. Without AA-LCR the other 11 average about +0.20pp, so gains elsewhere offset the named loss.** (high, vote 3-0)
  - Sources: https://bottlecapai.com/post/thinkingcap-gemma-4-26b/
  - Evidence: The figures come from the post. The offset arithmetic is the verifier's (the deltas sum to −1.83). Note that the post gives a CI for each model, not a CI on each difference.

- **Outcome efficiency (Chen et al. 2024, 'Do NOT Think That Much for 2+3=?') is ξ_O = (1/N) Σ σ_i (T̂_i / T_i): the share of tokens spent before the first correct answer, averaged over the dataset. It is a ratio of tokens, has no correction for repeated-sample noise and no equivalence test. The paper judges its length-reduction method by aggregates: QwQ-32B on MATH500 goes from 2,407.9 to 1,330.7 tokens, with accuracy 93.0 to 92.8. Its headline is a '48.6%' cut 'while maintaining accuracy'. It reports no flip analysis per question.** (high, vote 3-0 / 2-1)
  - Sources: https://arxiv.org/abs/2412.21187
  - Evidence: The formula claim voted 3-0 and the evaluation-framing claim 2-1. The table rows give a 44.7% cut, not 48.6%, which apparently comes from another comparison. The paper also reports process efficiency, which is likewise an aggregate.

- **ThoughtTerminator (Pu et al. 2025) uses repeated samples, but only to estimate difficulty: difficulty = the 'simple inaccuracy rate of that model over that pair over n samples', which the paper calls approximate and does not correct for noise. Its overthinking scores are in tokens. O_g is the mean gap between a model's expected spend and the shortest correct answer from any model. O_env is the mean gap between a model's maximum and minimum spend per question. Neither measures how correctness changes per question.** (high, vote 3-0)
  - Sources: https://arxiv.org/html/2504.13367v1
  - Evidence: Eqs. 1, 3 and 4 of the primary paper. Two claims, each voted 3-0.

- **LLMThinkBench's Overthinking Score is a task-level harmonic mean, O_i = 2·A_i·E_t,i/(A_i+E_t,i), with tokens min-max normalised across the pool of models. Variability is handled with 3-fold cross-validation (mean ± SD) and paired t-tests across tasks. Each question is sampled once at near-deterministic temperature, so there is no chance correction per question, and losses can offset gains.** (high, vote 3-0)
  - Sources: https://arxiv.org/pdf/2507.04023v3
  - Evidence: Definitions 3-4 and Section 4.4 of the primary paper. Two claims, each voted 3-0.

- **Negative flip rate is the established per-question metric that counts losses only, but it is uncorrected. Positive-Congruent Training (Yan et al., CVPR 2021) defines negative flips as samples the old model got right and the new one gets wrong, and NFR as their fraction. Gains do not offset losses. Going from ResNet-18 to ResNet-50 cuts error from 30.24% to 25.85%, yet NFR is still 4.89%. Two ResNet152 models with the same 78.3% accuracy, differing only in initialisation, disagree on 10.4% of samples, about 5.2% NFR from randomness alone. The paper reports this floor but does not subtract it. MUSCLE (Apple, 2024) applies NFR to LLM updates and notes that the new model can regress on many samples 'even when it has a better overall performance'. Sheng & Lu (Aug 2026) report NFR = Pr[c_new=0 | c_old=1] for each update pair, e.g. Qwen2.5 to Qwen3 is about 29% on MMLU-Pro and 2.6% on GSM8K. None of these is used to judge length-reduction methods.** (high, vote 3-0 (one 2-1))
  - Sources: https://openaccess.thecvf.com/content/CVPR2021/papers/Yan_Positive-Congruent_Training_Towards_Regression-Free_Model_Updates_CVPR_2021_paper.pdf, https://arxiv.org/abs/2011.09161, https://arxiv.org/html/2407.09435v2, https://www.alphaxiv.org/abs/2608.13607.md
  - Evidence: Six claims, voted 3-0 except one at 2-1 (the PCT definition, over 'raw count' vs rate). The 5.2% noise floor is exactly what your exact null removes. That is the cleanest way to say how your metric differs from NFR.

- **The closest published analogue to 'loss beyond chance' is Cacioli's within-model Reliable Change Index (arXiv 2604.27405, 30 Apr 2026). It adapts the clinical Jacobson & Truax 1991 RCI to compare two LLM versions item by item, with K=10 samples per item per model at T=0.7, and counts a change as reliable when |RCI| > 1.96. It differs from your metric in four ways. It uses a normal approximation with one pooled S_diff per model pair, from split-half reliability, rather than an exact exchangeability null per question. It excludes items at 0% or 100% accuracy for both models. It reports churn (improved and deteriorated shares) rather than summing losses into question-equivalents. It gives no equivalence verdict per task. Its MMLU-Pro results support the swap argument: Llama-3 to 3.1 had 33.7% of items improved and 28.4% deteriorated for a +1.6pt net gain; Qwen2.5 to Qwen3 had 46.9% improved and 39.0% deteriorated for +2.8pt. It recommends 'reporting churn rate alongside aggregate accuracy'. It does not address length reduction.** (medium, vote 3-0 / 2-1)
  - Sources: https://arxiv.org/html/2604.27405v1
  - Evidence: Single-author preprint, not peer reviewed. Votes 3-0 and 2-1. 'Closest' is a judgement made after the search, not something the source states.

- **Answer to Q3: no verified metric matches yours on all four properties. Yours is question-level, takes its chance correction from an exact exchangeability null over 16 pooled samples, counts only losses (no netting), and gives a 95% bootstrap equivalence verdict per task with a 0.5-question margin. NFR has the question-level, losses-only part but no noise correction. RCI has repeated samples and noise correction but uses a normal approximation, nets the two directions only as churn shares, and gives no verdict. BottleCap's matched Δ% pairs per question only for tokens. Your contribution is the combination, and its application to judging length reduction.** (medium, vote synthesis)
  - Sources: https://arxiv.org/html/2604.27405v1, https://arxiv.org/html/2407.09435v2, https://bottlecapai.com/post/thinkingcap-qwen3-6-27b/
  - Evidence: Synthesis across the verified claims. It is a finding of absence, limited to what this search covered, which did not verify G-Pass@k or literature on equivalence or non-inferiority testing.


## Other
{
 "caveats": "Not answered in this round. No claim about Trajectory's \"intelligence density\" (Yap, Field Notes, Sept 2026) survived verification, so whether it has a formal definition is still open. Do not cite it as a formal metric until you have checked it yourself. G-Pass@k (Liu et al. 2024), accuracy-at-fixed-budget curves, and equivalence/TOST-style testing in ML evaluation were not verified either. G-Pass@k in particular, which measures how consistent a model is across k samples, is a plausible neighbour you should check before claiming novelty. No direct quotes from Tomas Mikolov (talks, interviews) were verified; all BottleCap wording comes from the challenge README and two unsigned company blog posts (13 Jul and 23 Oct 2026). Those posts describe BottleCap's own large models (Qwen3.6-27B, Gemma-4-26B), not the scoring of the 0.6B challenge. Do not present their 5-seed or matched Δ% methods as challenge rules. The RCI paper is a single-author preprint, and Sheng & Lu is a recent v1. The labels \"answer efficiency vs reasoning efficiency\" are your own terms. Weaknesses to state in the pitch: with 8 samples per model the exact test per question has low power, so only large drops are flagged and many small real losses go undetected. Small tasks and subsets (e.g. MBPP+ with 378 questions, the BFCL subsets) give wide bootstrap intervals, so the verdict depends on task size. The 0.5-question margin is arbitrary unless calibrated on a baseline-vs-baseline null run. An absolute margin in question-equivalents gets stricter as tasks grow, a fairness question for SVAMP's 1,000 against MBPP+'s 378. It costs nothing in generation because it reuses the existing 8 samples, but it is new evaluation code that BottleCap would have to trust.",
 "openQuestions": [
  "Is Trajectory's 'intelligence density' (Yap, Field Notes, Sept 2026) defined formally anywhere, and does it overlap with this metric or only share the name? If it already means something else, keep the name or rename yours?",
  "How does G-Pass@k (Liu et al. 2024) relate to this metric, and has anyone used it or another consistency measure across repeated samples to judge methods that shorten reasoning?",
  "What does a run of the baseline against itself with a different seed give for loss beyond chance on each ThinkingCap benchmark? That calibrates the 0.5-question margin and shows the false-fail rate.",
  "Would a margin relative to task size (e.g. a fraction of the questions) or a per-benchmark threshold be fairer across SVAMP (1,000), IFEval (541) and MBPP+ (378)?"
 ],
 "refuted": [],
 "unverified": [],
 "sources": [
  {
   "url": "https://github.com/BottleCapAI/ThinkingCap-Test",
   "quality": "primary",
   "angle": "BottleCap primary sources",
   "claimCount": 5
  },
  {
   "url": "https://bottlecapai.com/post/thinkingcap-qwen3-6-27b/",
   "quality": "primary",
   "angle": "BottleCap primary sources",
   "claimCount": 5
  },
  {
   "url": "https://bottlecapai.com/post/thinkingcap-gemma-4-26b/",
   "quality": "primary",
   "angle": "BottleCap primary sources",
   "claimCount": 5
  },
  {
   "url": "https://arxiv.org/abs/2412.21187",
   "quality": "primary",
   "angle": "Efficiency metrics for length-reduction methods",
   "claimCount": 5
  },
  {
   "url": "https://arxiv.org/html/2504.13367v1",
   "quality": "primary",
   "angle": "Efficiency metrics for length-reduction methods",
   "claimCount": 5
  },
  {
   "url": "https://arxiv.org/pdf/2507.04023v3",
   "quality": "primary",
   "angle": "Efficiency metrics for length-reduction methods",
   "claimCount": 5
  },
  {
   "url": "https://arxiv.org/pdf/2503.21614",
   "quality": "secondary",
   "angle": "Efficiency metrics for length-reduction methods",
   "claimCount": 5
  },
  {
   "url": "https://arxiv.org/html/2507.02076v1",
   "quality": "secondary",
   "angle": "Efficiency metrics for length-reduction methods",
   "claimCount": 5
  },
  {
   "url": "https://themoonlight.io/en/review/efficient-reasoning-models-a-survey",
   "quality": "secondary",
   "angle": "Efficiency metrics for length-reduction methods",
   "claimCount": 3
  },
  {
   "url": "https://arxiv.org/html/2604.27405v1",
   "quality": "primary",
   "angle": "Flip rates and per-question regression",
   "claimCount": 5
  },
  {
   "url": "https://openaccess.thecvf.com/content/CVPR2021/papers/Yan_Positive-Congruent_Training_Towards_Regression-Free_Model_Updates_CVPR_2021_paper.pdf",
   "quality": "primary",
   "angle": "Flip rates and per-question regression",
   "claimCount": 5
  },
  {
   "url": "https://arxiv.org/html/2407.09435v2",
   "quality": "primary",
   "angle": "Flip rates and per-question regression",
   "claimCount": 5
  },
  {
   "url": "https://www.alphaxiv.org/abs/2608.13607.md",
   "quality": "primary",
   "angle": "Flip rates and per-question regression",
   "claimCount": 5
  },
  {
   "url": "https://arxiv.org/pdf/2105.03048",
   "quality": "primary",
   "angle": "Flip rates and per-question regression",
   "claimCount": 5
  },
  {
   "url": "https://oneuptime.com/blog/post/2026-08-31-measure-llm-eval-variance-thresholds/markdown",
   "quality": "blog",
   "angle": "Flip rates and per-question regression",
   "claimCount": 5
  },
  {
   "url": "https://arxiv.org/abs/2604.27405",
   "quality": "primary",
   "angle": "Repeated-sample reliability and equivalence tests",
   "claimCount": 5
  },
  {
   "url": "https://arxiv.org/abs/2412.13147",
   "quality": "primary",
   "angle": "Repeated-sample reliability and equivalence tests",
   "claimCount": 5
  },
  {
   "url": "https://arxiv.org/abs/2407.09435",
   "quality": "primary",
   "angle": "Repeated-sample reliability and equivalence tests",
   "claimCount": 5
  },
  {
   "url": "https://arxiv.org/pdf/2603.17449",
   "quality": "primary",
   "angle": "Trajectory intelligence density",
   "claimCount": 5
  },
  {
   "url": "https://arxiv.org/pdf/2605.07501",
   "quality": "primary",
   "angle": "Trajectory intelligence density",
   "claimCount": 5
  },
  {
   "url": "https://fin.ai/research/we-dont-need-higher-peak-intelligence-only-more-intelligence-density/",
   "quality": "blog",
   "angle": "Trajectory intelligence density",
   "claimCount": 4
  }
 ],
 "stats": {
  "angles": 5,
  "sourcesFetched": 21,
  "claimsExtracted": 102,
  "claimsVerified": 25,
  "confirmed": 25,
  "killed": 0,
  "unverified": 0,
  "afterSynthesis": 11,
  "urlDupes": 0,
  "budgetDropped": 9,
  "agentCalls": 103
 }
}