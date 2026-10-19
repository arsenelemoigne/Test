# Post-training for shorter thinking: research survey and a method proposal for the ThinkingCap Test

*Compiled 2026-10-19. Sources were read through web search summaries and abstracts. The container's network policy blocks direct arXiv and Hugging Face fetches, so the numbers below are the authors' own claims and have not been checked against the full papers.*

---

## 0. What the test rewards, and what that implies

The metric shapes the method, so start there:

- **Tokens saved is a geometric mean over 4 benchmarks with equal weight.** A 30% cut on BFCL is worth exactly as much as a 30% cut on SVAMP. A math-only method that cuts SVAMP by 60% and leaves BFCL, IFEval and MBPP+ alone scores worse than one that cuts all four by 25%. **Every domain must shrink.**
- **Cut-off answers count as wrong, and wrong answers' tokens still count.** At 0.6B, a large share of tokens is probably in **degenerate repetition loops that hit the 4096/8192 cap**. Removing those loops saves tokens *and* gains accuracy. Measure this first.
- **The evaluation is out of domain with respect to the suggested training data** (math and QA in training; arithmetic, function calling, code and instruction following in the evaluation). The literature shows math-only shortening transfers to code and science ([ETR](https://arxiv.org/pdf/2604.05355), [ATTNPO](https://arxiv.org/pdf/2602.09953), [ShorterBetter](https://arxiv.org/pdf/2504.21370)). **I found no evidence about transfer to function calling or instruction following.** That gap is both a risk and room to stand out.
- **Prompting and turning thinking off are excluded, and the method must be "general".** Reviewers want an *algorithm*, ideally with a Pareto curve (several checkpoints).

---

## 1. Map of the field (2025 to Oct 2026)

### A. RL with length-aware rewards (the main family)
| Idea | Paper | Key point |
|---|---|---|
| Per-prompt adaptive penalty, scaled by solve rate (easy prompts pay more per token) | [Just Enough Thinking / ALP, 2506.05256](https://arxiv.org/abs/2506.05256) | Changing one reward term reportedly cuts tokens by about 50% with accuracy roughly flat |
| Difficulty-calibrated length reward plus attention-based step pruning | [TRAAC "Think Right", 2510.01581](https://arxiv.org/pdf/2510.01581) | Bins problems into easy/medium/hard; prunes more on easy |
| Difficulty-aware advantage reweighting plus length regularisation | [GRPO-LEAD, EMNLP'25](https://aclanthology.org/2025.emnlp-main.287.pdf) | Also explicitly penalises wrong answers |
| Budget estimated online from the GRPO group (no extra compute) | [Dual-process TLB, 2505.16315](https://arxiv.org/pdf/2505.16315) | The group itself supplies the target length |
| Model predicts its own budget, then follows it | [SelfBudgeter, 2505.11274](https://arxiv.org/pdf/2505.11274) | |
| Target = shortest correct length in the group | [ShorterBetter, 2504.21370](https://arxiv.org/pdf/2504.21370) | 50 to 80% reduction, including out of domain |
| **Shortest correct rollouts form a "virtual shorter distribution", used as a regulariser with no length reward** | [Implicit Compression Regularization (ICR), 2605.07316](https://arxiv.org/html/2605.07316v1) | Claims length penalties *inevitably* cost accuracy; ICR keeps accuracy, generalises beyond math, and stacks with penalties |
| Step-level (not trajectory-level) penalties | [SWAP, 2603.00296](https://arxiv.org/pdf/2603.00296) | Trajectory penalties can't tell essential steps from redundant ones |
| Token-significance-weighted penalty | [Not All Tokens Matter, 2506.08125](https://arxiv.org/html/2506.08125v2) | |
| **Split redundancy into internal (before the first correct answer) and external (after it)** | [Reconsidering Overthinking, 2508.02178](https://arxiv.org/pdf/2508.02178) | **Cutting after the answer is safe; cutting before it hurts** |
| Entropy-trend reward | [ETR, 2604.05355](https://arxiv.org/pdf/2604.05355) | Math-only training; HumanEval on Qwen3-4B 41→54% while length halved |
| Adaptive truncation during online RL | [TRiMS, 2603.17449](https://arxiv.org/html/2603.17449) | Over 80% reduction on 1.5 to 14B models; schedules need tuning per model |
| Drop outlier-length rollouts | [DOT, 2601.03969](https://arxiv.org/html/2601.03969) | Transfers to code |
| Fix GRPO's suppression of correct long chains | [LAGR, 2602.22556](https://arxiv.org/pdf/2602.22556) | Stability under length pressure |

**Known pitfalls.** Applying length rewards early hurts exploration ([Short-RL](https://www.researchgate.net/publication/391878319_Efficient_RL_Training_for_Reasoning_Models_via_Length-Aware_Optimization)). GRPO's sequence-level normalisation under-penalises *long wrong* answers, and Dr. GRPO swaps that for a different bias ([2607.23364](https://arxiv.org/abs/2607.23364) proves you can't have both unbiasedness and length invariance). In TRL, choose `loss_type` deliberately: `dapo` (token-level) is the safe default.

### B. Self-distillation, no reward model needed
- **[CRISP, 2603.05433](https://arxiv.org/abs/2603.05433):** the teacher is *the same model with a conciseness instruction*; the student is trained with on-policy reverse KL on its own rollouts. It reports up to 56% shorter on MATH-500 with Qwen3-14B and +3.3 pp accuracy. The prompt is used only to make the teacher, so the shortening lives in the weights, which is within the rules. **Mixed replication:** [2605.13255](https://arxiv.org/html/2605.13255) saw no gains on 4B/8B, and [2607.05184](https://arxiv.org/html/2607.05184v1) found conciseness alone doesn't preserve long-budget gains. Very cheap with LoRA: the teacher is just the adapter switched off.
- **[SSOPD, 2605.17497](https://arxiv.org/html/2605.17497v1):** a teacher conditioned on the shortest *correct* completion is distilled into prefixes of the longest *wrong* one.
- **[SRCL, 2609.30652](https://arxiv.org/html/2609.30652v1):** keeps only self-rewrites that are shorter, terminate naturally and are correct, then uses them as cross-entropy targets. The effect was modest at 8B.
- **[Mixed-policy distillation, 2605.08776](https://arxiv.org/pdf/2605.08776), [ConMax, 2601.04973](https://arxiv.org/pdf/2601.04973)** (confidence-maximising compression: −43% at −0.7 pp on Qwen2.5-7B), [CoT-Valve](https://arxiv.org/pdf/2502.09601) (a length-controllable direction in parameter space), and [ConPress, 2602.01472](https://arxiv.org/html/2602.01472v2): when several questions are packed into one prompt, the model compresses each one by itself. That self-compressed reasoning becomes training data.

### C. Step-level pruning of traces, then SFT
- [Step Entropy, 2508.03346 (ICLR'26)](https://arxiv.org/html/2508.03346): prunes low-entropy steps; **dropping whole steps keeps accuracy at 40% removal, while token-level pruning (TokenSkip) breaks at 20%.**
- [Graph-based CoT pruning, 2604.05643](https://arxiv.org/html/2604.05643): redundant-reflection graph, then SFT → DPO → GRPO, −42%. [DRP, 2505.13975](https://arxiv.org/html/2505.13975): keep/delete/rewrite/merge actions per step. [OS-Pruner, 2607.11089](https://arxiv.org/html/2607.11089): pruning as an optimal-stopping problem. [Supertokens, 2604.26355](https://arxiv.org/html/2604.26355).

### D. Early exit and "the model already knows"
- [DEER, 2504.15895](https://arxiv.org/pdf/2504.15895): at "Wait" cues, try an answer and exit if confident. Dynasor-CoT: exit after 3 consecutive matching answers.
- [LYNX, 2512.05325](https://arxiv.org/html/2512.05325): a hidden-state probe at cue tokens, supervised by forced exits, plus conformal control. [Conformal Thinking, 2602.03814](https://arxiv.org/html/2602.03814v2).
- [Reasoning Theater (Goodfire), 2603.05488](https://arxiv.org/html/2603.05488v4): probes show the model's belief settles long before its CoT ends; exiting at 95% probe confidence saved 68% of tokens on MMLU.
- **Caveat ([2609.03633](https://arxiv.org/html/2609.03633v1)):** forcing `</think>` doesn't make the model *stop* reasoning; it keeps going in the answer. **Any exit has to be learned by the policy, not injected.**
- **Gap:** I found **no paper that trains the reasoning model itself to end at the point where its probed answer has converged.** Section 2 targets exactly this.

### E. Weight-space and activation-space methods
- Steering: [Manifold Steering, 2505.22411](https://arxiv.org/abs/2505.22411) (one "overthinking direction", up to −71% on R1-distill), [ASC, 2507.04742](https://arxiv.org/pdf/2507.04742) (−67% on GSM8K/MATH500 with no training), [Reasoning-strength planning, 2506.08390](https://arxiv.org/html/2506.08390v1) (activations at `<think>` encode the length the model is planning), [Reflection Steering, 2608.25542](https://arxiv.org/html/2608.25542), [PID-steering, 2506.18831](https://arxiv.org/html/2506.18831).
- Merging: [Long-to-Short via merging](https://github.com/hahahawu/Long-to-Short-via-Model-Merging) (about −50% at 7B), [Evo-L2S, 2604.06465](https://arxiv.org/html/2604.06465), [Fisher layer-adaptive merging, 2603.21705](https://arxiv.org/pdf/2603.21705). There's no short-thinking 0.6B checkpoint to merge with, but **merging RL checkpoints along the trajectory** is a cheap way to trace the Pareto curve.

### F. Latent reasoning (probably out of scope)
[Abstract-CoT, 2604.22709](https://arxiv.org/html/2604.22709v2) (a reserved discrete vocabulary), [Latent Thought Flow, 2606.16222](https://arxiv.org/pdf/2606.16222), CODI and Coconut. These change *what* the thinking is made of. They fight the rule "still thinks before it answers", the requirement to load in vLLM, and the fixed tokeniser and prompt. Mention them in IDEA.md as future work; don't build on them.

### G. Small models and agentic tasks
- Under 1B: pure SFT produces long, repetitive outputs; **EOS-temperature tricks and RL fix a lot** ([2505.07961](https://arxiv.org/html/2505.07961v1)); loop detection alone saves about 20% ([TrimR](https://arxiv.org/pdf/2505.17155)); [LFM-1B-Math](https://liquid.ai/research/lfm-1b-math-can-small-models-be-concise-reasoners).
- Agents: overthinking hurts tool use ([2502.08235](https://arxiv.org/pdf/2502.08235)); models know *when* to call tools without reasoning ([2605.09252](https://arxiv.org/html/2605.09252v2)). That suggests large, safe savings on BFCL, especially the *irrelevance* subset.

### H. Reference point: BottleCapAI's own ThinkingCap
ThinkingCap-Qwen3.6-27B: −46% (OOD) / −58% (in-domain); GSM8K thinking 3175→648 tokens with accuracy *up*. ThinkingCap-Qwen3.8-27B: −37.2% at −0.86 pp over 12 benchmarks ([blog](https://bottlecapai.com/post/thinkingcap-qwen3-8-27b/), [MarkTechPost](https://www.marktechpost.com/2026/09/24/bottlecap-ai-releases-thinkingcap-qwen3-8-27b-37-2-fewer-thinking-tokens-at-a-0-86pp-accuracy-cost/)). The recipe is unpublished ("curated multi-domain problems, minimally invasive"). Agentic reductions were smaller (about 11%). **The takeaway: they value multi-domain, minimally invasive and accuracy-neutral, which is the bar to beat.**

---

## 2. Proposed method: **Sufficiency-Gated Self-Distillation (SGSD)**

*One idea, three parts: find the point where the model's own answer becomes settled, teach the model to stop there, and never penalise reasoning before that point.*

### Core insight
Combine three findings that no single paper puts together:
1. Cutting **after** the answer is decided is safe; cutting **before** it causes underthinking ([2508.02178](https://arxiv.org/pdf/2508.02178)).
2. The model's belief **settles long before its CoT ends** and can be read cheaply ([Reasoning Theater](https://arxiv.org/html/2603.05488v4), [LYNX](https://arxiv.org/html/2512.05325), [DEER](https://arxiv.org/pdf/2504.15895)).
3. Exits must be **learned by the policy**, not injected ([2609.03633](https://arxiv.org/html/2609.03633v1)), and blunt length rewards cost accuracy ([ICR](https://arxiv.org/html/2605.07316v1)).

So: **label each rollout's minimal sufficient prefix (MSP)**, the earliest step boundary after which a forced answer is reliably correct, and **put length pressure only on the tokens after the MSP.** This is a per-token, per-sample, self-calibrating budget. It needs no difficulty bins or λ schedules, and it works for any task with a grader. The eval repo already ships graders for **all four domains**.

### Stage 0: profile the baseline (half a day, essential)
- Run the full baseline. For each benchmark, report: mean and median tokens, **% of responses hitting the cap**, % of responses with n-gram loops, and tokens of wrong versus right answers.
- This tells you where the tokens are. For a 0.6B model, I'd bet loops and cap hits dominate on MBPP+/IFEval.

### Stage 1: multi-domain training pool (no eval contamination)
| Domain | Train sources (not the evaluation sets) | Grader |
|---|---|---|
| Math | GSM8K, Skywork-OR1 easy band (provided) | `rewards.py` |
| QA | CommonsenseQA, ARC-Easy (provided) | `rewards.py` |
| Code | e.g. KodCode / APPS-intro / TACO-easy, with unit tests | reuse `utils/eval/sandbox.py` |
| Function calling | e.g. xLAM-function-calling-60k, ToolACE, plus **synthetic irrelevance** items (offer a function that doesn't fit) | AST-match, like BFCL's grader |
| Instruction following | synthetic verifiable constraints (e.g. Tulu-3 IF-RLVR style), deduplicated against IFEval prompts | reuse IFEval-style verifiers |

Run an n-gram deduplication against `questions/questions.jsonl` and document it in IDEA.md (rule 2).

### Stage 2: MSP labelling, offline with vLLM
For each training prompt, sample K=8 rollouts from the base model, using the evaluation's sampling settings.
For each **correct** rollout:
1. Split the `<think>` block into steps at `\n\n` and at reflection cues ("Wait", "Alternatively", "Let me check", "Hmm").
2. At each boundary *b*, append `\n</think>\n\n` and sample *m*=4 short answers (max about 256 tokens, or whatever the task needs). With vLLM prefix caching this is cheap: the prefix is shared, and 0.6B is fast.
3. **MSP** = the earliest *b* where at least 3 of 4 forced answers are correct **and** stay correct at every later boundary (stability, as in Dynasor's consensus).
4. For **wrong** rollouts and cap hits, mark the whole trace as "no sufficient prefix".

Outputs: per-token labels (*pre-MSP* or *post-MSP*), and statistics that are interesting in their own right: **how much of each domain's thinking is post-sufficiency**. That plot alone is a strong IDEA.md figure.

### Stage 3: warm start with sufficiency-truncated self-distillation (SFT, LoRA)
- Target = the model's own trace up to MSP + a small margin (one extra step), then the model's **own** natural transition and answer. Regenerate the answer from the truncated prefix and keep it only if it's correct, so the targets stay on-policy and keep Qwen's style.
- Among the correct rollouts of a prompt, choose the shortest MSP target (the ICR/ShorterBetter idea, but applied after truncation).
- Mix in about 20% untruncated shortest-correct traces from hard prompts (pass rate ≤ 2/8) so the model doesn't learn "always stop early".
- This alone is a solid submission point. Save the checkpoint.

### Stage 4: RL with sufficiency-gated advantages (GRPO, LoRA, TRL + vLLM colocate)
Reward per rollout *i* in a group:
```
r_i = correct_i                                # 0/1 from the grader
A_i = group-normalised(r_i)                     # standard GRPO
```
**Token-level shaping** (the new part). For *correct* rollouts, use an online MSP estimate: cheap probes at every few boundaries, or a learned probe (below). Then:
```
advantage_t = A_i                     for t ≤ MSP_i
advantage_t = A_i − β · p_i           for t > MSP_i     (p_i = group pass rate → easy prompts pay more, as in ALP)
```
For *wrong* rollouts, use the standard negative advantage with **token-level (DAPO) normalisation**, so long wrong answers aren't under-penalised. For cap hits, add an extra penalty, since loops are pure waste.

Optional: add ICR's "virtual shorter distribution" term (a KL toward the group's shortest correct rollout) as a stabiliser.

Why this should beat a plain length penalty:
- Tokens before sufficiency are **never** penalised → no underthinking on hard prompts.
- The budget is per sample and comes from the model's own beliefs → it works the same for math, code, function calling and instruction following.
- β·p_i scales pressure with difficulty, with no tuned bins.

### Stage 5 (stretch, the novel bit): a learned sufficiency head *inside* the RL loop
Probing every boundary during RL costs extra generations. Instead, **train a tiny linear probe** on hidden states at step boundaries to predict "forced answer is correct", using the Stage-2 labels as supervision (the LYNX approach, but used for *training-time credit assignment*, not inference-time exits). Use it to place MSP online at almost no cost. Throw the probe away after training; the final model is plain Qwen3 + LoRA (rule 1, and it loads in vLLM).

### Zero-training baselines worth one table row each
1. **Folded steering vector:** compute the "long versus short thinking" direction (difference of mean residuals at `<think>` between long and short correct traces) and bake it into the weights. Qwen3 supports `attention_bias=True` in its config, so the vector can be written as an `o_proj` bias at chosen layers. That loads in standard HF and vLLM. Check this on one layer before relying on it.
2. **Checkpoint interpolation:** linear merges between base and the SFT/RL checkpoints give a cheap Pareto curve (the README asks for several points).

### Ablations for IDEA.md
| Ablation | Question it answers |
|---|---|
| Uniform length penalty (Kimi/ALP-style) vs. sufficiency-gated | Does gating protect accuracy? |
| Math+QA only vs. 5-domain pool | Does shortening transfer to function calling and instruction following? (The literature has no answer, so either result is new.) |
| SFT only vs. SFT→RL | What does RL add? |
| Probe MSP vs. forced-answer MSP | Is the cheap probe good enough? |
| β sweep → Pareto curve | The length/accuracy trade-off the reviewers explicitly ask for |
| Cap-hit rate before/after | How much of the gain is loop removal? |

---

## 3. Compute budget (single GPU)
- MSP labelling: about 20k prompts × 8 rollouts plus probes on a 0.6B model in vLLM. That's hours on a 4090/L4, not days.
- SFT LoRA: under 1 h. GRPO LoRA, 4k context, gradient checkpointing, Liger cross-entropy (see README): about 300 to 600 steps, roughly 6 to 12 h on a 24 GB card.
- Develop with `--smoke` and a single benchmark at 2 samples. Do one full 8-sample run per final checkpoint (about 2 h on Ampere).

## 4. Risks
- **Forced-answer probes on IFEval are noisy:** a constraint like "write 300 words" can't be met in a 256-token probe. Use a domain-specific probe length, or skip probing for length-constrained instructions and use only the shortest-correct target there.
- **Probing changes the distribution:** the answer after a forced `</think>` isn't the policy's natural answer. That's why Stage 3 regenerates answers and Stage 4 lets RL learn the exit.
- **0.6B RL is noisy:** use larger groups (8 to 16), a small learning rate, and keep a KL anchor to base, or switch the LoRA off as the reference (README tip).

## 5. Where this method is new
- Prior work has the pieces separately: early exit at inference (DEER/LYNX), shortest-correct regularisers (ICR/ShorterBetter), step-level penalties (SWAP), and the internal/external split (2508.02178).
- **Using the model's own answer sufficiency as a per-token RL credit boundary, labelled by forced-answer probes and amortised by a hidden-state probe, then measured on function calling and instruction following**, did not appear in anything I found.
