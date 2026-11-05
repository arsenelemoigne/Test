# Deep research for Jasmin (5 Nov 2026): per-type LoRA DPO + merging, -50% tokens at < 1 pt on a 0.6B model

No verified source tests the exact Jasmin setup: per-type LoRA DPO adapters on a 0.6B model, merged after training, aimed at about -50% tokens for less than 1 point of accuracy loss. Your own result, a single joint DPO run with 1:1 shorter-right and right-beats-wrong pairs that saved about 47% for -0.5 points, is already close to the best published small-model results.

What the literature does support:
- **Training separately then merging can beat a data mix.** This holds with 2 objectives or skills at 3B-8B: DEM at 3-13B, Cohere's Aya merge at 8B, and LoRA Soups at 7B. Merged DPO checkpoints gave more consistent gains than merged SFT checkpoints.
- **The advantage fades as more skills are added.** With 3 skills, data mixing won. One study merging 2-12 checkpoints at 3-8B found Task Arithmetic was the only reliable method and TIES collapsed.
- **Plain shortest-vs-longest DPO pairs are too weak for -50%.** They gave about -17.5% at 1.5B; self-training SFT gave about -30%.
- **The methods that reach -50% at under 1 point of loss at 1.5B are of two kinds.** One is on-policy RL with correctness-aware length rewards or a learned skip-thinking choice: AdaptThink gave -53% at +2.4 and DRPO gave -77% on GSM8K at -1.1. The other is a searched or layer-wise merge of a short model with a long reasoning model: Evo-L2S gave -52% to -55% at +4 points and FIM-TIES gave -92% at higher accuracy. Plain uniform merges degrade at 1.5B and below.

Suggested plan for Jasmin, based on these findings:
1. Keep the joint-DPO pair mix as the baseline.
2. Train 4 per-type adapters with the same mix in each, so every adapter keeps its own right-beats-wrong anchor.
3. Merge with Task Arithmetic or a layer-wise, searched set of coefficients, not TIES, and not DARE at its default drop rate of 0.9.
4. Score the merge on a two-objective front of accuracy against length, measured on held-out questions from every type, and compare it against the joint run.
5. If the merge cannot pass -50% / -1 point, add a short AdaptThink- or DRPO-style on-policy stage.

## Findings (verified 3-vote)
### Training one model per data source or objective and adding the weighted deltas (task arithmetic) can beat a single run on mixed data, with 2-5 sources at 3B-13B. The evidence is full fine-tunes or LoRA SFT, plus one DPO study at 8B. None of it is below 1B.
- confidence: high (3-0 (all four claims))
- evidence: DEM, OpenLLaMA-7B: MMLU 50.74 vs 47.77, BBH 40.56 vs 36.38, DROP 37.96 vs 32.71, MathQA 32.16 vs 30.35 against a well-tuned data-mix baseline; the same direction holds at 3B and 13B, with weights w_i=0.25. Aya 23 8B: merging per objective beat the data mix by up to 8% general and 10% safety. LoRA Soups CAT (learned per-layer LoRA concatenation, Llama-7B) beat data mixing by 12% and the TIES/DARE/LoraHub merges by 43% on average. DEM itself uses full fine-tunes only, with no LoRA and no DPO; it mentions PEFT only as a fix for storage cost.
- sources: https://www.arxiv.org/pdf/2406.15570v2, https://arxiv.org/pdf/2410.10801, https://arxiv.org/html/2410.13025v2

### Merging DPO checkpoints gives more consistent gains over a data mix than merging SFT checkpoints, but the outcome depends on the merge method. SFT merges in the same study raised general performance but also raised harmful generations.
- confidence: medium (3-0)
- evidence: Aya 23 8B: DPO merges averaged +2.8% general and +2.2% safety over the 15% safety-mix baseline across four merge methods. SFT merges averaged about +6% general but +4.6% more harmful generations. Not every method helped: TIES on DPO checkpoints fell to 63.6 from 71.0 general. This is a single source, at 8B.
- sources: https://arxiv.org/pdf/2410.10801

### Merged adapters can be superadditive, with the merge beating the sum of the solo adapter gains. This matches your observation that merges preserve accuracy better than the sum of solo effects.
- confidence: medium (3-0)
- evidence: GSM-Hard, Llama-7B LoRA rank 32: base 5.91, code-only 8.04 (+36%), math-only 14.18 (+140%), DATA-MIX LoRA 18.8, CAT math+code 21.11 (+257%, more than the 176% sum). CAT over 3 seeds: 21.63 +/- 1.07, so the gap over DATA-MIX is about two standard deviations. CAT's coefficients are trained on a small mixed set, so it is not a training-free merge.
- sources: https://arxiv.org/html/2410.13025v2

### The merging advantage shrinks or reverses as the number of merged skills or checkpoints grows. With 3 skills, joint data mixing beat every LoRA merge. TIES collapses as the merge count grows, and Task Arithmetic is the only method that reliably improves on the base. With 4 per-type adapters, Jasmin sits in the risky range, so the joint run must be kept as a control.
- confidence: medium (3-0 (all three claims))
- evidence: LoRA Soups: with 3 prompt formats, all LoRA merging baselines did worse than DATA-MIX ('still the best strategy'). Systematic merging study (Llama 3.2 3B, Llama 3.1 8B, Qwen3 4B, Qwen3 8B; 2-12 checkpoints): Task Arithmetic beat the base in 100% of merges at n=12 (only 4 merges), mean gain about +1.02. TIES success fell from 31% at n=2 to 0% at n=12 (mean -4.32), which the authors hypothesise comes from weights drifting too far from the base. Nothing was tested below 3B.
- sources: https://arxiv.org/html/2410.13025v2, https://arxiv.org/html/2511.21437v1

### Plain 'shortest correct beats longest' DPO pairs, or SFT on the model's own shortest correct answers, keep accuracy but cut only about 17-30% of tokens. That is well short of -50%, so the pair rule alone will not reach the target.
- confidence: high (3-0)
- evidence: DPO Shortest on R1-Distill-Qwen-1.5B: -17.5% average length at +0.8 accuracy (GSM8K 78.3/804, MATH500 82.4/3708, AIME 30.7/10794); on 7B it gave -29.5% at -0.6. Self-training SFT on the shortest correct samples from best-of-N with few-shot conditioning: about -30% tokens averaged over 1.5B-7B models with average accuracy held, but on GSM8K -36% length came with about a 3% relative accuracy drop. The DPO Shortest baseline was run by the AdaptThink authors themselves and may be under-tuned.
- sources: https://arxiv.org/pdf/2505.13417, https://arxiv.org/html/2502.20122v2

### DAST's pair rule uses same-correctness pairs only: the shorter correct answer beats the longer correct one, and the longer wrong answer beats the shorter wrong one. Pairs are scored against a per-question token budget and trained with SimPO. Adding correct-beats-wrong pairs gave no gain in DAST, which conflicts with your finding that a 1:1 mix with right-beats-wrong pairs protects accuracy. The settings differ (7B math, with DICP pairs present), and DAST compressed much less at 7B than at 32B.
- confidence: medium (3-0 (method, compression); 2-1 (CICP conflict))
- evidence: Budget: L_budget = p*L_mean_correct + (1-p)*L_max, where p is the sampling accuracy. At most one DCP and one DICP are kept per question (largest margin), and the lowest-margin 15-18% of pairs are dropped. Ablation on 7B MATH-500: DAST 93.6 accuracy at 18.0% compression vs DAST+CICP 93.2 at 18.3%. Compression: 7B MATH-500 -18.1% (93.2 to 93.6), 7B AIME24 +1.9% length; 32B MATH-500 -46.0%. AdaptThink's table reports DAST at 1.5B as -42.1% at -0.6.
- sources: https://arxiv.org/html/2503.04472v2

### On-policy RL that separates the length reward for correct and incorrect answers, or that teaches the model when to skip thinking, reaches -50% or better at 1.5B while keeping or improving accuracy on math. This is the strongest published route to the target, beyond pair-based DPO.
- confidence: medium (3-0; DRPO 7B claim 2-1)
- evidence: AdaptThink, R1-Distill-Qwen-1.5B: -53% average length at +2.4 points (GSM8K 978 to 480 tokens, accuracy 79.0 to 83.1; MATH500 4887 to 1782; AIME 12073 to 6679); 7B: -40.1% at +2.3. DRPO, 1.5B: GSM8K -77% at -1.1%. On 7B, over four math sets, DRPO went from 3053 to 1502 tokens (-51%) at -2.6%, so harder sets lost more than 1 point at -50%. Both are math-only, both are authors' own runs, and neither tests a 0.6B model or code, instruction-following or function-calling tasks.
- sources: https://arxiv.org/pdf/2505.13417, https://arxiv.org/html/2510.04474v2

### Merging a short-CoT model with a long-CoT model can cut length by about 50-55% at under 1 point of loss at 7B with TIES, AIM-TIES or Task Arithmetic. DARE fails at its default drop rate of 0.9. At 1.5B, plain uniform merges degrade: harder benchmarks drop and the merged models show 'false reflections'.
- confidence: high (3-0 (all three claims))
- evidence: 7B, against R1-7B's 54.5 average: TIES -53.0% at +0.3, AIM-TIES -55.3% at +0.5, Sens-Merging -49.8% at +1.9, Task Arithmetic -48.7% at -1.0; DARE variants lost 2.1-7.2 points, and the authors recommend a drop rate below 0.5 (they used 0.3). 1.5B: Task Arithmetic and TIES held GSM8K at 74.5 and 75.9 vs 76.8, but MATH500 fell 5-7 points and AIME24 went from 20 to 10; average merging collapsed (GSM8K 26.6). The authors' section title: 'Smaller models struggle to learn from model merging'.
- sources: https://arxiv.org/abs/2503.20641v2

### Searched or layer-wise merge coefficients rescue long-to-short merging at 1.5B, reaching -50% to -92% length at equal or better accuracy. This suggests that if Jasmin merges per-type adapters, the merge weights should be searched per layer and scored on accuracy against length, not set uniformly.
- confidence: medium (3-0 (all three claims))
- evidence: Evo-L2S (NSGA-II search over Task Arithmetic merges, COLM 2026). At 1.5B, against a 45.5% six-benchmark average: L2S-01 scores 50.1% at -52.5% length, L2S-02 49.7% at -55.3%, L2S-03 -1.0pp at -60.1%, L2S-04 -4.7pp at -69.3%. At 7B, -58% at -0.8pp. This used a single search seed and the Pareto points were picked on the test benchmarks. FIM-TIES (Fisher-weighted per-layer coefficients, single-author preprint, 1.5B): 47.3 average vs 43.3 for ACM-TIES and about 41 for the long-CoT model, with length 5671 to 411 tokens. Both merge full models, not LoRA adapters, and both are math-only.
- sources: https://arxiv.org/html/2604.06465, https://arxiv.org/pdf/2603.21705

## caveats
- **No evidence at 0.6B.** No verified source tests a 0.6B model, per-type LoRA adapters trained with DPO for length, or merging such adapters. Everything is extrapolated from 1.5B-13B.
- **Merging may work worse at small scale.** Several sources say merging gets harder at small scale and with more experts, so the 3-8B merging results may overstate what to expect at 0.6B.
- **Math only.** All length-reduction evidence is on math benchmarks. Nothing covers code, instruction following or function calling, including 'no function fits' irrelevance cases, which is exactly where cross-task spill-over matters.
- **Spill-over and protective data are uncovered.** Question 4 (spill-over and how protective data reduces it) has only indirect support. Question 5 (concrete hyperparameters for sub-1B DPO length training: LoRA rank, learning rate, beta, float32 vs bf16, batch size) is not covered by any verified claim. The only incidental settings are LoRA rank 32, alpha 64, learning rate 3e-4 for skill SFT at 7B; learning rate 1e-5, batch 16, 1 epoch for self-training SFT; DARE drop rate below 0.5; DEM weight 0.25.
- **Fragile evidence base.** Many results are single preprints run by their own authors with no reported seed variance, and some baselines (DPO Shortest) may be under-tuned. The Evo-L2S Pareto points were selected on the test benchmarks.
- **Ambiguous percentages.** It is unclear whether some figures are absolute points or relative (DRPO's 1.1% and 2.6%; DEM's gains are relative).
- **Two claims were refuted.** One said best-of-16 shortest-correct gives only about 12%; the other said ModelMerging at 1.5B loses 9.4 points in AdaptThink's table. Neither is used.
- **Time-sensitivity.** The field moves fast; sources date from 2024 to Aug 2026.

## openQuestions
['At 0.6B with 4 adapters, does a searched per-layer Task Arithmetic merge of per-type DPO adapters (each trained on a 1:1 mix of shorter-right and right-beats-wrong pairs) beat the joint 1:1 DPO run that saved about 47% for -0.5 points, or does the finding that data mixing wins when there are many skills dominate?', "Do right-beats-wrong pairs genuinely protect accuracy at 0.6B across task types, given that DAST found no gain from them at 7B math? Would adding DAST's 'longer wrong beats shorter wrong' pairs or a per-question token budget help with hard items?", "How much of the cross-type spill-over (one type's adapter shortening every type) is useful versus harmful on function-calling irrelevance and instruction following, and does it change with merge coefficients or replay data?", 'What DPO/LoRA hyperparameters (beta, learning rate, rank, steps, float32 vs bf16) are stable for length-reduction DPO on sub-1B models? No verified source covers this.']

## refuted
[{'claim': 'Naive best-of-N with N=16 that keeps only the shortest correct sample gives a small length reduction (about 12% on average) because it is sample-inefficient. Picking the shortest correct path alone is a weak pair or data rule unless the sampling distribution is first pushed toward concise outputs.', 'vote': '0-3', 'source': 'https://arxiv.org/html/2502.20122v2'}, {'claim': 'On the 1.5B model, the long-to-short ModelMerging baseline cut length by 32.3% but lost 9.4 average accuracy points (MATH500 80.6 to 63.0, AIME 29.4 to 18.1). This is evidence that weight-merging long2short methods degrade badly at small scale.', 'vote': '0-3', 'source': 'https://arxiv.org/pdf/2505.13417'}]

## sources
- https://www.arxiv.org/pdf/2406.15570v2 (primary, Merge vs data-mix (small-scale, DPO))
- https://arxiv.org/pdf/2410.10801 (primary, Merge vs data-mix (small-scale, DPO))
- https://arxiv.org/html/2410.13025v2 (primary, Merge vs data-mix (small-scale, DPO))
- https://arxiv.org/html/2511.21437v1 (primary, Merge vs data-mix (small-scale, DPO))
- https://arxiv.org/html/2510.04474v2 (primary, DPO length-reduction pair rules)
- https://arxiv.org/html/2503.04472v2 (primary, DPO length-reduction pair rules)
- https://arxiv.org/html/2502.20122v2 (primary, DPO length-reduction pair rules)
- https://arxiv.org/pdf/2505.13417 (primary, DPO length-reduction pair rules)
- https://arxiv.org/abs/2503.20641v2 (primary, Long-to-short via model merging)
- https://arxiv.org/html/2604.06465 (primary, Long-to-short via model merging)
- https://arxiv.org/pdf/2603.21705 (primary, Long-to-short via model merging)
- https://arxiv.org/html/2501.12599v1 (primary, Long-to-short via model merging)
- https://liner.com/review/unlocking-efficient-longtoshort-llm-reasoning-with-model-merging (secondary, Long-to-short via model merging)
- https://arxiv.org/html/2601.06052v2 (primary, Cross-task spill-over and protection)
- https://arxiv.org/pdf/2504.07448 (primary, Cross-task spill-over and protection)
- https://preview.aclanthology.org/setup/2025.acl-long.1284 (primary, Cross-task spill-over and protection)
- https://arxiv.org/html/2411.05209v1 (primary, Cross-task spill-over and protection)
- https://ar5iv.labs.arxiv.org/html/2410.04587 (primary, Cross-task spill-over and protection)
- https://arxiv.org/pdf/2504.21370 (primary, Cross-task spill-over and protection)
- https://arxiv.org/html/2508.10164 (primary, Practitioner hyperparameters)
- https://arxiv.org/pdf/2607.00862 (primary, Practitioner hyperparameters)
- https://arxiv.org/pdf/2502.04463 (primary, Practitioner hyperparameters)
- https://arxiv.org/html/2602.20945v1 (primary, Skeptical / limits)
- https://arxiv.org/html/2601.02972v1 (primary, Skeptical / limits)