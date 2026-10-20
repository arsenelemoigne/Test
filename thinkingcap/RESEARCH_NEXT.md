# Deep research: new ways to cut Qwen3-0.6B's thinking (21 Oct 2026)

Automated multi-source research (21 sources fetched, 25 claims checked by 3 independent verifiers each; 23 confirmed, 2 refuted).

## Summary

For Qwen3-0.6B under ThinkingCap's rules, three lines of work have the strongest evidence. Each also targets the transfer gap you've seen. (1) On-policy self-distillation from the model itself conditioned on a concise prompt (CRISP-style reverse KL), or from a larger Qwen3 teacher that rewrites the student's own traces (MPD). MPD is the only method found with out-of-domain evidence: a model trained only on math still got shorter on code-understanding questions (CodeMMLU 3.6k to 2.3k tokens, accuracy up). (2) Iterated on-policy SFT, i.e. sampling 8, keeping correct-and-short, and resampling often. It roughly matches tuned length-penalty RL (about 80% shorter at 1.5B with no accuracy loss) at much lower cost, and its ablation says fresh on-policy data is what makes it work, not the loss function. This argues for replacing your one-shot MVT SFT set with repeated rounds. (3) A multi-objective search over interpolation coefficients between endpoints that share the same base (Evo-L2S / model-interpolation style). It is training-free and fits an L4 budget, but at 1.5B merging models of different origin is fragile. Adaptive think / no-think RL (AdaptThink, Thinkless) gives the largest small-model cuts on easy math (53-67%). It mostly works by skipping thinking, which probably breaks the must-think rule, so you would need to adapt it to a short-think mode. Token-level pruning (TokenSkip) loses more accuracy the smaller the model and is the weakest bet. No method has been shown below 1.5B, and none reports results on code generation (MBPP+) or function calling (BFCL).

## Findings (ranked)

### 1. (high confidence, vote 3-0)

Adaptive mode selection (AdaptThink, Thinkless) gives the largest token cuts reported at 1.5B. It learns per problem when to think, based on difficulty. AdaptThink on R1-Distill-Qwen-1.5B: -53% length, +2.4 pts accuracy averaged over GSM8K, MATH500 and AIME24. Thinkless on GSM8K: 1919 to 624 tokens (-67%), Pass@1 0.8347 to 0.8418, using long thinking on only 13% of queries. Think usage rises with difficulty: 13% on GSM8K, 26% on Minerva, 52% on MATH-500, 100% on AIME. Accuracy costs on harder sets: MATH-500 -4.2 pts (5675 to 2555 tokens), Minerva -1.2 pts. AIME tokens still fell 61% while thinking 100% of the time, so part of the gain comes from shorter thinking, not only from skipping it.

*Evidence and caveats:* Primary papers, 3-0 votes on all three claims. Caveats: the base model is R1-Distill-1.5B, which overthinks heavily; evaluation is math only; the savings rely on an empty <think></think> block, which probably breaks ThinkingCap's must-think rule. To comply, replace NoThinking with a minimal-think mode, e.g. reward a short think trace on easy items. SVAMP is close to GSM8K, so math would likely benefit most.

Sources: https://arxiv.org/abs/2505.13417, https://arxiv.org/abs/2505.13379

### 2. (medium confidence, vote 3-0)

On-policy self-distillation toward a concise teacher (CRISP) needs no labels, no token budgets and no difficulty estimator. It fits the rules because the be-concise instruction is used only at training time. The teacher is the same model prompted to be concise, and the student is trained with per-token reverse KL on its own rollouts. On Qwen3-14B: MATH-500 up to 56% shorter with up to +3.3 pts; AIME24 38% shorter within about 1 pt. A follow-up (BIRD) on Qwen3-1.7B: MATH-500 2673 to 1092 tokens (-59%) at 79.4 to 79.6; AIME24 -35% at +2.5 pts; AIME25 only -8%. At 1.7B the gain is compression, not accuracy.

*Evidence and caveats:* Primary 2026 preprints, 3-0 votes. The smallest model tested is 1.7B. Nobody has shown that a 0.6B model follows a concise instruction well enough to act as its own teacher. BIRD's checkpoint selection used the test benchmark. At 1.7B a plain concise prompt beat BIRD on accuracy. Both methods were trained and evaluated on math only. BIRD's detailed two-stage recipe description was refuted 0-3, so treat its exact pipeline as unverified.

Sources: https://arxiv.org/abs/2603.05433, https://arxiv.org/pdf/2607.15736

### 3. (medium confidence, vote 3-0)

Mixed-Policy Distillation (MPD) from a larger Qwen3 teacher is the only approach found with evidence of transfer outside its training domain, which is your main bottleneck. The student samples a trace, Qwen3-8B rewrites it to be concise, and the student is trained with KL against the teacher's token distributions on the rewritten trace. It uses no ground truth, reward model or length penalty. On Qwen3-1.7B trained only on OpenMathInstruct-2: average Pass@4 68.2 to 71.3, tokens 8.2k to 6.1k (-25.6%; up to 27.1%). Out of domain: CodeMMLU 81.2 to 82.8 with 3.6k to 2.3k tokens; LongBench-v2 48.7 to 50.3 with 0.9k to 0.7k tokens.

*Evidence and caveats:* A single primary preprint (May 2026), 3-0 on all three claims. Caveats: CodeMMLU is multiple-choice code understanding, not code generation like MBPP+; Pass@4 is a lenient metric; there is one student model; correctness is not part of the objective, so the student can learn compressed traces that are wrong. For 0.6B: Qwen3-1.7B or Qwen3-4B as rewriter and logit teacher should fit on an L4, since Qwen3 models share a tokenizer. The idea of using the teacher's logits on a compressed version of the student's own trace explains the transfer, and you can apply it to code and tool-call traces from non-benchmark data.

Sources: https://arxiv.org/pdf/2605.08776

### 4. (high confidence, vote 3-0)

Simplified length RL turns into iterated on-policy SFT, and the fresh on-policy data does most of the work. Removing the KL term and group std normalization and using a truncation reward (correct and under the limit) reduces GRPO to SFT on self-sampled, filtered, correct and concise outputs. On R1-Distill-Qwen-1.5B: 10,178 to 2,186 tokens (-79%) at 59.0 to 59.9% accuracy; efficiency 2.74 vs 2.55 for the best RL baseline (L1). The off-policy ablation used the same filter, with data refreshed every 50 steps instead of per step. It got only modest shortening and lost substantial accuracy on 7B MATH-500. DLER makes the same point from the RL side: a plain truncation penalty (zero reward over 4000 tokens) works once the optimization is fixed (batch-wise advantage normalization, clip 0.2/0.28, dynamic sampling).

*Evidence and caveats:* Two primary papers, all claims 3-0. Implication: your MVT SFT was one-shot and static, and your ShorterBetter GRPO used std normalization. Two concrete next steps: (a) several rounds of resample-filter-SFT with a cutoff derived from MVT; (b) re-run DLER with batch-wise normalization and asymmetric clipping, not just the truncation reward. That fix may be why your high-learning-rate RL lost accuracy on code and math. Evidence is math-only, at 1.5B and 7B.

Sources: https://www.arxiv.org/pdf/2602.13407, https://arxiv.org/pdf/2510.15110

### 5. (medium confidence, vote 3-0)

Weight interpolation and multi-objective merge search are a cheap, training-free knob, but at small scale they need endpoints from the same base. Interpolating between Qwen3 Thinking-2507 and Instruct-2507 at 4B and 30B gives efficient reasoners. Behaviour shifts in three stages as lambda changes. At 4B, lambda in [0.4, 0.6] is the sweet spot where thinking appears and tokens drop (IFEval 6492 to 2932 tokens from lambda 0.4 to 0.5). Above that, outputs overthink. Evo-L2S runs NSGA-II over merge coefficients with accuracy and length as objectives. It reports over 50% shorter reasoning at 1.5B (+4.2 pts) and 7B (-0.8 pts) at about 7.5 A100-hours.

*Evidence and caveats:* Primary papers, 3-0. Qwen3-0.6B has no Instruct/Thinking pair, so you would need to build the endpoints yourself (base, RL adapter, MVT-SFT adapter). Your exact LoRA blend (+33.5% at -0.5 pts) already amounts to a 1-D lambda sweep. Additions that are actually new: searching 3-way or per-module blend weights, or TIES/DARE over several adapters, with tokens and accuracy on a non-benchmark proxy set that covers code and tool calls as the two objectives. Warning from the 1.5B merging study: merging models of different origin is fragile at small scale (Task Arithmetic/TIES -5 to -7 pts MATH500, -10 pts AIME24; plain averaging collapses GSM8K from 76.8 to 26.6). Sens-Merging was the exception. Low lambda values remove thinking entirely, which breaks the rule. The broader claim that merging is training-free and gives up to 55% shorter outputs was refuted 0-3.

Sources: https://arxiv.org/html/2510.10977v2, https://www.alphaxiv.org/abs/2604.06465, https://arxiv.org/abs/2503.20641v2

### 6. (medium confidence, vote 3-0)

Difficulty-aware trace rewriting by a teacher (DAP / LiteCoT) is an offline SFT data recipe. A large teacher rates each problem's difficulty and rewrites the trace to a matching length (about 720 tokens on average). A student trained on 100K of these traces beat R1-Distill trained on 800K long-CoT traces: 1.5B 52.4 vs 46.6 average over AIME24, MATH500, GPQA, AMC23 and OlympiadBench, at lower training and inference cost.

*Evidence and caveats:* Primary paper, later accepted at ACL Findings 2026, 3-0. The comparison is confounded (different teacher and recipe from the R1-Distill baseline). The students were Qwen2.5 models trained from scratch, not an existing thinking model being shortened. Evaluation is math and science only. For you: a Qwen3-4B or 8B rewriter could produce difficulty-scaled traces for code and tool-use data that is not from the benchmarks. That attacks the domain gap directly, but offline data underperformed on-policy data (see finding 4).

Sources: https://arxiv.org/abs/2505.19716v1

### 7. (medium confidence, vote 3-0)

Token-level pruning of CoT (TokenSkip: LLMLingua-2 importance scores, then LoRA SFT on pruned traces) is cheap but loses more accuracy the smaller the model. At ratio 0.5 on GSM8K: Qwen2.5-3B 83.7 to 74.4 (-9.3), 7B 91.4 to 86.0 (-5.4), 14B -1.7. Mild ratios cost less: at 0.9 the 3B model loses 0.5 pts for about 17% fewer tokens.

*Evidence and caveats:* Primary paper, 3-0. No model at 1.5B or below was tested, and only short-CoT instruct models were used. The method puts the compression ratio in the prompt, which you would have to drop to comply with the fixed evaluation prompt. Token-level deletion is close to your failed TokenProbe idea. This is the lowest priority for a 0.6B model.

Sources: https://arxiv.org/abs/2502.12067

## Caveats

No verified result covers a model below 1.5B. Every number is from 1.5B-14B models, usually R1-Distill models that overthink far more than Qwen3-0.6B, so less redundancy may be left to remove. Almost every evaluation is math. The only out-of-domain evidence is MPD's CodeMMLU, which is multiple-choice, so nothing directly covers MBPP+ code generation or BFCL function calling. Most sources are 2025-2026 arXiv preprints without peer review or independent replication, and some picked checkpoints on test benchmarks (BIRD). The best-performing adaptive methods depend on empty think blocks, which probably breaks ThinkingCap's must-think rule. Two claims were refuted: training-free merging giving up to 55% shorter outputs, and BIRD's detailed two-stage recipe. Any search, merge or selection step needs a proxy set that is not from the benchmarks, or it risks contamination. The ranking (on-policy self/teacher distillation and iterated on-policy SFT first, a merge search over same-base adapters as a cheap addition, adaptive short-think RL only once modified, token pruning last) is synthesis judgment, not measured on 0.6B.

## Open questions

- Can Qwen3-0.6B prompted to be concise act as a useful CRISP-style self-teacher, or does it need a 1.7B/4B Qwen3 logit teacher (MPD-style) to give a clean signal?
- Does MPD-style compression trained on code or tool-call traces that are not from the benchmarks (e.g. self-generated traces on non-MBPP programming tasks and synthetic function-calling data) actually shorten MBPP+ and BFCL outputs without dropping pass rates?
- Would AdaptThink/Thinkless rewarding a minimal non-empty think block (rather than NoThinking) keep the 50-67% easy-math savings while satisfying the must-think rule?
- Does a DLER-style re-run (batch-wise advantage normalization, asymmetric clipping, dynamic sampling) explain and remove the code and math losses of your high-learning-rate GRPO run?

## Refuted

- (0-3) Merging a long-CoT reasoning model (DeepSeek-R1-Distill-Qwen) with its short-CoT base model (Qwen2.5-Math) using task-vector, SVD-based or activation-informed methods cuts average response length by up to 55% while keeping or slightly improving accuracy. This is a training-free, weight-only way to shorten outputs. (https://arxiv.org/abs/2503.20641v2)
- (0-3) BIRD uses a two-stage recipe for shortening reasoning. Stage 1 samples concise self-generated traces under a brevity instruction, keeps only the correct ones, and runs one epoch of LoRA SFT on them under the original prompt, with the brevity instruction removed. Stage 2 runs on-policy reverse-KL self-distillation from a concise self-teacher, starting from the Stage 1 model. It is cheap enough to fit inside the ThinkingCap rules, and the result is a standard model with no inference-time changes. (https://arxiv.org/pdf/2607.15736)
