# RL methods that shorten reasoning: verification of the Kimi summary and a methods catalogue for a 0.6B model

Written 11 Nov 2026. Every number below comes from the paper's PDF, downloaded from arxiv.org/pdf/<ID> and read as text. Page numbers are the PDF page counter. A claim I could not confirm is marked **[unverified]**. My own reasoning is marked **[inference]**.

Context: Qwen3-0.6B in thinking mode. The goal is more than 50% fewer tokens at better than -1 point accuracy on ThinkingCap-Test (SVAMP, MBPP+, IFEval, BFCL), scored at a fixed 4,096 / 8,192 token cap with 8 samples. We already have a GRPO / Dr. GRPO + LoRA trainer (rl_train.py).

---

## 1. Verification of the pasted Kimi summary

All five arXiv IDs exist and are the papers the summary names:
- 2501.12599: *Kimi k1.5: Scaling RL with LLMs*, v4 of 3 Jun 2025.
- 2507.20534: *Kimi K2: Open Agentic Intelligence*.
- 2602.02276: *Kimi K2.5: Visual Agentic Intelligence*.
- 2607.24653: *Kimi K3: Open Frontier Intelligence*.
- 2510.15110: *DLER: Doing Length pEnalty Right*, NVIDIA, v1 of 16 Oct 2025.

### Claim 1: Kimi k1.5 (2501.12599)

| Sub-claim | Verdict | Exact text / source |
|---|---|---|
| Min-max relative length reward inside the group: λ = 0.5 − (len(i) − min_len)/(max_len − min_len); λ if correct, min(0, λ) if wrong; zero when max_len = min_len | **VERIFIED** (one location detail) | §2.3.3 "Length Penalty". The section starts on p.5 but the formula is on **p.6**: "If max_len = min_len, we set length reward zero for all responses … len_reward(i) = λ If r(x, y_i, y*) = 1; min(0, λ) If r(x, y_i, y*) = 0, where λ = 0.5 − (len(i) − min_len)/(max_len − min_len)." |
| Added to the task reward with a weight | **VERIFIED** | p.6: "This length-based reward is then added to the original reward with a weighting parameter." |
| "standard policy optimization without length penalty, followed by a constant length penalty for the rest of training" | **VERIFIED** (exact quote) | p.6: "length penalty may slow down training during the initial phases … we propose to gradually warm up the length penalty … Specifically, we employ standard policy optimization without length penalty, followed by a constant length penalty for the rest of training." |
| Separate long2short RL phase (§2.4) with the length penalty and a shorter max rollout | **VERIFIED** | §2.4, "Long2short RL", p.8: "we select a model that offers the best balance between performance and token efficiency to serve as the base model, and conduct a separate long2short RL training phase. In this second phase, we apply the length penalty introduced in Section 2.3.3, and significantly reduce the maximum rollout length to further penalize responses that exceed the desired length while possibly correct." |
| 60.8 on AIME2024 at an average of 3,272 tokens (§3.4) | **VERIFIED** | §3.4, "Long2short", p.13: "k1.5-short w/ rl achieves a Pass@1 score of 60.8 on AIME2024 (averaged over 8 runs) while utilizing only 3,272 tokens on average." |
| Optimizer = mean-reward baseline + squared log-ratio penalty (eq. 3), not GRPO | **VERIFIED** | §2.3.2, p.5. This is online mirror descent with surrogate loss (r − τ log Z − τ log π_θ/π_θi)². τ log Z is approximated by the mean sampled reward. Eq. 3 gradient: (1/k) Σ_j [∇log π_θ(y_j)(r(x,y_j) − r̄) − (τ/2) ∇(log π_θ/π_θi)²]. Details the summary leaves out: samples come from the reference policy π_θi (off-policy); there is no std normalisation and no value network; "we also reset the optimizer at the start of each iteration." |
| Weight, switch time and reduced rollout length not published | **VERIFIED (absence)** | I found no numeric value for the penalty weight, the warm-up switch point or the long2short max length anywhere in the paper. |
| "the larger model generally shows better token efficiency than the smaller model" (§3.5 p.14) | **VERIFIED** (exact quote) | §3.5, "Scaling of model size and context length", p.14. Next sentence: "if test-time compute has a budget, training smaller models with a larger context length may be viable solutions." The two model sizes are not given. |

### Claim 2: Kimi K2 (2507.20534)

| Sub-claim | Verdict | Exact text / source |
|---|---|---|
| §3.2.3 p.13, per-sample max token budget set by task type; over-budget responses are truncated and penalised; values not given | **VERIFIED** | §3.2.3 "RL Algorithm", paragraph "Budget Control", p.13: "we enforce a per-sample maximum token budget throughout RL training, where the budget is determined based on the type of task. Responses that exceed this token budget are truncated and assigned a penalty". No budgets or penalty values are given. The same section confirms K2 reuses the k1.5 objective, with the Muon optimizer, a PTX loss and temperature decay. |

### Claim 3: Kimi K2.5 (2602.02276)

| Sub-claim | Verdict | Exact text / source |
|---|---|---|
| §4.4.2 p.9, "Toggle": phases alternate every m iterations | **VERIFIED** | §4.4.2 "Reinforcement Learning", paragraph "Token Efficient Reinforcement Learning", p.9: "the algorithm alternates between two optimization phases every m iterations". Phase 0 applies when ⌊t/m⌋ mod 2 = 0. |
| Phase 0: r̃ = r · 1[mean_i r(x,y_i) < λ or \|y\| ≤ budget(x)]; Phase 1: r̃ = r | **VERIFIED** | Same page. The budget constraint "is only enforced when the model's mean accuracy for a given problem exceeds the threshold λ." |
| budget(x) = ρ-th percentile of correct-answer lengths, estimated once at the start | **VERIFIED** | Eq. 2, p.9: "This budget is estimated once at the beginning of training and remains fixed thereafter." |
| "Toggle decreases output tokens by 25~30% with a negligible impact on performance" on K2 Thinking | **VERIFIED** (exact quote) | p.9: "We evaluate the effectiveness of Toggle on K2 Thinking … On average, Toggle decreases output tokens by 25∼30% with a negligible impact on performance." |
| Transfer to GPQA / MMLU-Pro when trained on math + code | **VERIFIED** | p.9: "when trained exclusively on mathematics and programming tasks, the model still achieves consistent token reductions on GPQA and MMLU-Pro with only marginal degradation in performance (Figure 5)." |
| Token-level gradient masking for log-ratios outside [α, β] | **VERIFIED** (one nuance) | Eq. 1 and text, p.8: "policy gradients are computed normally for tokens with log-ratios within the interval [α,β], while gradients for tokens falling outside this range are zeroed out … regardless of the sign of the advantages." Nuance: eq. 1 is printed as Clip(ratio, α, β), while the prose says log-ratio. The loss is also normalised by N, the total number of tokens in the batch. |
| λ, m and ρ not published | **VERIFIED (absence)** | No values appear in the paper. |
| Motivation (not in the summary but useful) | — | p.9: "models trained under rigid budget constraints often fail to generalize to higher compute scales … defaulting to truncated reasoning patterns" ("length-overfitting"). |

### Claim 4: Kimi K3 (2607.24653)

| Sub-claim | Verdict | Exact text / source |
|---|---|---|
| §4.1.2 p.13: reward overridden with −1 above τ·b0(x), where b0 is estimated from the cold-start model | **VERIFIED** | §4.1.2, "Reasoning Effort RL", p.13: "We associate each problem x with an initial token budget b0(x) estimated from the cold-start model, and override the task reward with −1 for trajectories whose total token budget T(y) exceeds a scaled threshold τ · b0(x)." T(y) counts thinking tokens for general tasks. For agentic tasks it counts all output tokens, including tool arguments. |
| τ annealed per domain to produce low / high / max experts | **VERIFIED** | Same page: "We first train a max-budget variant with a relatively large τ … We then anneal τ to smaller values to obtain the high- and low-effort expert models. The adjustment of τ is configured per domain under human-in-the-loop guidance." |
| 9 experts distilled into one model | **VERIFIED** | p.12: "Crossing these three domain experts with three reasoning effort levels in {low, high, max} yields a total of nine expert models." Consolidation uses SFT plus Multi-Teacher On-Policy Distillation (§4.1.3, p.13). |
| Effort given as a plain-language option, not a token count (App. F p.47) | **VERIFIED** | p.47: "Instead of modifying the generation prefix or exposing a token budget, the message states the requested level in natural language". The schema reserves four levels (low, medium, high, max), and K3 supports a subset. |
| Extra (not in the summary) | — | p.13: a verbosity control for the generative reward model: "a candidate whose output length exceeds σ · ℓ0 automatically loses the binary comparison." |

### Claim 5: DLER (2510.15110)

| Sub-claim | Verdict | Exact text / source |
|---|---|---|
| Truncation-based length penalty works … only with batch-wise reward normalisation, higher clipping and dynamic sampling | **PARTLY WRONG (overstated / mis-scoped)** | The recipe is the four parts named (abstract p.1): "batch-wise reward normalization, higher clipping, dynamic sampling, and simple truncation length penalty". Two corrections. (a) The paper is not about "DeepSeek-R1-1.5B only". It trains DeepSeek-R1-**1.5B and 7B** (§4.1, p.8), plus Llama-3.1-Nemotron-Nano-8B (§4.6). (b) The ablations that justify each ingredient (Figs. 2, 5, 6) were run on **DeepSeek-R1-7B**, not on 1.5B. The paper never states that truncation works *only* with these ingredients. It says plain GRPO + truncation keeps losing accuracy (Fig. 2, 7B), and that each ingredient fixes one identified problem. |
| Quote "It is not the sophisticated design of the length penalty that determines performance, but rather the choice of RL optimization algorithm." | **VERIFIED** (exact) | Key Insight 1, p.2. |

**Bottom line:** claims 1–4 hold, with small page/location fixes for claim 1. Claim 5's quote is exact, but "on DeepSeek-R1-1.5B only" is wrong: the paper covers 1.5B and 7B, and the ablations are on 7B.

---

## 2. Methods catalogue

Unless stated otherwise, "1.5B" means DeepSeek-R1-Distill-Qwen-1.5B: a heavy over-thinker, roughly 5–16k tokens on math. That is a different starting point from Qwen3-0.6B.

### 2.1 DLER: truncation + optimizer fixes (2510.15110, NVIDIA, Oct 2025)
- **Models:** DeepSeek-R1-1.5B and 7B (also Nemotron-Nano-8B). **Shown at ≤2B: yes (1.5B).**
- **Reward:** truncation. A response over the target length gets reward 0. Target 4,000 tokens (§4.1, p.8).
- **Algorithm:** a GRPO-style clipped objective with three changes:
  - **Batch-wise advantage normalisation:** A_i = R_i − mean_group(R); A_norm = (A − mean_batch)/std_batch (eq. 3, p.4). This replaces GRPO's per-group std.
  - **Clip-higher:** ε_low 0.2, ε_high 0.28.
  - **Dynamic sampling:** "discard prompts whose rollouts all yield zero reward or all yield positive reward and resample until the target batch size is reached" (p.6).
- **Hyperparameters (Table 4, p.21, veRL):**

  | Setting | Value |
  |---|---|
  | train batch | 512 prompts |
  | PPO mini-batch | 64 |
  | PPO epochs | 1 |
  | lr | 1e-6 |
  | rollout temperature | 1 |
  | rollouts per prompt | 16 |
  | clip | 0.2 / 0.28 |
  | filter_groups | on (seq_reward) |
  | KL loss | coef 0.0005, type mse |
  | max prompt | 1024 |
  | training | 450 steps on 7B (§3.4) |
  | data | DeepScaleR-Preview, 40K competition math problems |

  For Nemotron-8B: target 6,000 tokens and ε_high 0.36.
- **Why each piece is there (paper's findings):**
  - Truncation raises per-prompt advantage variance: {0.4, 0.32, 0.3, 0.29} for caps {4k, 8k, 12k, 16k} (p.4).
  - Clipped tokens are low-probability, high-entropy "transitional" words. Clip-higher stops entropy collapse (Key Finding 2, p.6).
  - About half the prompts start out all-zero (all 16 rollouts too long). Late in training about 40% are all-one, and the model "overfits to over-shorter responses", plateauing at about 2k under a 4k cap (p.6).
- **Results (Table 1, p.9; average over MATH, AIME-24, AMC, Minerva, Olympiad; eval at 32k, 16 samples):**

  | Model | Avg length | Accuracy | AIME-24 | Olympiad |
  |---|---|---|---|---|
  | 1.5B base | 10,499 | — | 29.79 | 44.07 |
  | DLER-1.5B | 2,466 (**−77%**) | MATH 84.31→86.95 | 34.38 | 48.31 |
  | DA-DLER-1.5B | 2,106 (−80%) | — | — | — |
  | DLER-7B | 2,405 (−69%) | ≥ base on every benchmark | — | — |
  | DA-DLER-7B | 2,167 (−73%) | — | — | — |

  - DA-DLER (difficulty-aware): correctness ratio threshold 0.5, cap 2,000 for easy and 4,000 for hard, run for 150 more steps.
- **Other findings:**
  - Run under the DLER recipe, penalty shape (truncation, cosine, L1-Max, Laser) "no longer push[es] the accuracy–efficiency frontier" (§4.5). Truncation is also cheapest, because rollouts stop at the cap.
  - Update-selective merge, for weak public data: keep the top 25% largest-update deltas, scale them by 0.7 and add them to the base. This recovered accuracy at −46% length (§4.6).

### 2.2 ThinkPrune: iterative length-limit RL (2504.01296)
- **Models:** R1-Distill-Qwen-1.5B, DeepScaleR-1.5B-Preview, QwQ-32B. **≤2B: yes.**
- **Reward:** DeepSeek-R1 correctness reward computed on clip(Y, L). Any answer past the limit L is discarded, so reward = 0 (p.4).
- **Algorithm:** GRPO in veRL; batch 128; 16 rollouts per question. lr and KL are not given in the main text **[unverified]**.
- **Data:** 2,470 AIME+AMC problems (the PRIME subset).
- **Schedule:** one-shot runs at 4k, 3k or 2k, or iterative runs 4k→3k→2k. Checkpoint rule: "the shortest average output length while maintaining a relative 10% accuracy drop."
- **Results (Table 1; average of MATH500, AIME24, AMC, Olympiad; eval at 32k):**

  | Model | Before | After |
  |---|---|---|
  | R1-Distill-1.5B, 4k→3k→2k | 10,355 tokens, 56.8 | 3,574 tokens (**−65%**), 57.4 (**+0.6**) |
  | DeepScaleR-1.5B, 4k→3k→2k | 5,914 tokens, 65.7 | 3,370 (−43%), 63.2 (**−2.5**) |
  | DeepScaleR-1.5B, one-shot 2k | — | 3,429, 61.7 |
  | QwQ-32B | — | −2.7 accuracy; iterating did not help |

  - Iterating beats one-shot at the same final cap for the 1.5B models. "Saturated" models (already RL-trained) lose more accuracy.

### 2.3 L1 / LCPO: length control through the prompt (2503.04697, COLM 2025)
- **Model:** DeepScaleR-1.5B-Preview; a 7B variant in App. A.9. **≤2B: yes.**
- **Reward:**
  - L1-Exact: r = 1[correct] − α·|n_gold − n_y|, with α = 0.0003. The prompt adds "Think for n_gold tokens." with n_gold ~ U(100, 4000).
  - L1-Max: r = 1[correct] · clip(α·(n_gold − n_y) + δ, 0, 1), with δ = 0.5. The L1-Max α is not stated in the text I read **[unverified]**.
- **Algorithm:** GRPO in veRL; lr 1e-6; batch 128; training context 4K (eval 8K). L1-Exact ran 700 steps; L1-Max ran 120 more steps. Data: DeepScaleR.
- **Result:** controllable length. About 1% below DeepScaleR-4K at matched budgets. Its "Short Reasoning Models" beat GPT-4o by 2% at equal length. This is a controllability method rather than a fixed reduction figure.

### 2.4 ShorterBetter: Sample Optimal Length (2504.21370)
- **Models:** R1-Distill-1.5B and 7B. **≤2B: yes.**
- **Reward:** r = α·1[correct] − β·|ℓ(y) − ℓ_SOL|. The absolute-value bars are lost in the PDF text, but the "deviation" reading is consistent with the text. ℓ_SOL is the length of the *shortest correct* response in the group, or the group mean length if none is correct (eq. 1, p.5).
- **Algorithm:** GRPO in veRL.

  | Setting | Value |
  |---|---|
  | batch | 64 |
  | group size | n = 8 |
  | lr | 1e-6 |
  | steps | 300 (1.5B), 200 (7B) |
  | max train length | 6K (1.5B), 5K (7B) |
  | α | 1 or 2 |
  | β | 0.001 |
  | KL γ | 0.001 |
  | ε | 0.2 |
  | data | DeepScaleR |

- **Results (§5.1):**
  - 1.5B in-domain math: −77.6% / −79.2% length, accuracy +2.5 / +3.2 (α = 1 / 2).
  - 1.5B out-of-domain: −77.0% / −75.5% length, accuracy −0.8 / −1.6.
  - 7B: −62% length, +7.1 in-domain and −0.48 out-of-domain.
  - Caveat: evaluation uses **one sample per problem**, so these numbers are noisy.

### 2.5 "Concise Reasoning via RL" (Fatemi et al., 2504.05185)
- **Models:** R1-Distill-1.5B and 7B; Qwen2.5-Math-1.5B-Instruct. **≤2B: yes.**
- **Reward:** no explicit length term.
  - PPO: +1 correct and boxed, −0.5 boxed but wrong, −1 no box.
  - GRPO: 1/0.
- **Claim:** RL with a per-token-averaged loss makes responses *longer* when reward is negative and *shorter* when positive (Theorems 1–2). So a short second RL phase on problems the model can **occasionally solve** shortens outputs with no penalty at all.
- **Settings (App. A.7):**
  - PPO: γ = 1, λ_GAE = 0.95; 8 samples, temperature 0.6, max length 20,000; actor lr 5e-7, critic lr 9e-6.
  - GRPO: 8 samples, max length 24,576, lr 1e-6.
  - Second stage: 8 MATH problems, "fewer than 60 training steps".
- **Results:** "over 54% and 40% reduction in response length for the R1 1.5B and R1 7B models, respectively, while preserving accuracy" (§8).
- **Failure modes they documented on 1.5B:**
  - GRPO on 4 *unsolvable* problems "quickly converged to extremely short outputs (fewer than 80 tokens)", because the advantages are zero and only the KL term acts.
  - On all-solvable problems, KL dominates again and progress stalls (p.12).
  - They recommend PPO over GRPO for this second stage.

### 2.6 Arora & Zanette, "Training Language Models to Reason Efficiently" (2502.04463)
- **Models:** R1-Distill-1.5B and 7B. **≤2B: yes.**
- **Reward:** r = 1[correct] · (1 − α·σ((LEN − MEAN(x))/STD(x))). MEAN and STD are the per-prompt mean and std of *correct* response lengths, estimated online. α ∈ {0.05, 0.1, 0.2, 0.4} (eq. 5).
- **Algorithm:** PPO with the RLOO advantage, in OpenRLHF.

  | Setting | Value |
  |---|---|
  | samples per prompt | 8 |
  | prompts per iteration | 32 |
  | global batch | 128 (2 gradient steps per iteration) |
  | lr | 5e-6 (1.5B), 2e-6 (7B) |
  | KL | 1e-3 |
  | generation | temperature 1, 32K context |
  | training | about 100 RL steps (~200 gradient updates) |
  | data | 3.2k Numina prompts |

- **Results, 1.5B, MATH500 (App. Tables 3–5, base 5,278 tokens / 0.833):**

  | α | Tokens | Accuracy |
  |---|---|---|
  | 0 | 3,077 | 0.840 |
  | 0.05 | 2,606 (**−51%**) | 0.835 (**+0.2**) |
  | 0.1 | 2,536 | 0.819 |
  | 0.4 | 1,395 | 0.749 |

  - 7B: "a reduction of 50% in tokens with less than 5% reduction in accuracy."
- **Key side finding (p.9):** at α = 0, length still dropped. This is "because of the recently discovered biased nature of RLOO's implementation … which normalizes the loss by the length of the response. We find that removing the length bias from the implementation leads to no change in the response length".

### 2.7 AdaptThink: learn to skip thinking (2505.13417)
- **Models:** R1-Distill-1.5B and 7B. **≤2B: yes.**
- **Objective:** advantage A = 1[first token = `</think>`]·δ + R(x,y) − R̄_ref(x). R̄_ref is the mean reward of K reference samples drawn before training. The loss is PPO-style with no KL (eq. 7).
- **Importance sampling:** the first token is forced 50/50 between `</think>` (NoThinking) and a thinking opener ("Alright"). This solves the cold start.
- **Settings:** veRL; context 16K; batch 128; lr 2e-6; K = 16; δ = 0.05; ε = 0.2; 1 epoch (314 steps); DeepScaleR data. Checkpoint at step 300 (1.5B).
- **Results, 1.5B:** length −50.9% / −63.5% / −44.7% on GSM8K / MATH500 / AIME24, with accuracy +4.1 / +1.4 / +1.6. Even at δ = 0 the model picks NoThinking on more than 50% of GSM8K/MATH500.

### 2.8 Dr. GRPO: length-bias analysis (2503.20783, COLM 2025)
- **Not a length-reduction method.** It removes GRPO's 1/|o_i| and 1/std(R) terms, normalising instead by a constant (the generation budget, MAX_TOKENS).
- **Mechanism (p.6), for GRPO's response-level bias:**
  - Positive advantage: "greater gradient updates for shorter responses" (favours short correct answers).
  - Negative advantage: "longer responses are penalized less" (favours long wrong answers).
  - Separately, std normalisation up-weights very easy and very hard prompts.
  - The authors also note that many PPO implementations carry the same length bias.
- **Shown on:** Qwen2.5-Math-1.5B/7B and Qwen2.5-1.5B base (Oat, MATH data). Result: better token efficiency because long *incorrect* responses stop growing.
- **[inference]** Combined with Arora's finding in §2.6, moving to Dr. GRPO also removes the implicit "shorter correct gets bigger per-token update" push. Under Dr. GRPO, any shortening must come from an explicit length term or a cap.

### 2.9 DAPO (2503.14476, ByteDance Seed)
- **Model:** Qwen2.5-32B base (not small).
- **Ingredients:** clip-higher (ε_low 0.2, ε_high 0.28); dynamic sampling (drop all-0 and all-1 groups); token-level loss (normalise over all tokens in the batch); overlong reward shaping; no KL.
- **Soft Overlong Punishment (eq. 13):** 0 if |y| ≤ L_max − L_cache; ((L_max − L_cache) − |y|)/L_cache inside the cache window; −1 beyond L_max. Used with L_max − L_cache = 16,384, L_cache = 4,096, L_max = 20,480.
- **Settings:** lr 1e-6 constant with 20-step warm-up; 512 prompts × 16 responses; mini-batch 512 (16 gradient updates per rollout step).
- **Relevance:** the clip-higher and dynamic-sampling settings that DLER and "Art" reuse come from here.

### 2.10 Laser / Laser-D / Laser-DE (2505.15612)
- **Models:** R1-Distill-1.5B, 7B, 32B. **≤2B: yes.**
- **Reward (step bonus):**
  - LASER: R = correct + α·1[correct and L ≤ L_T], with α = 0.5.
  - Laser-D: L_T is set per difficulty bucket (3 classes) and adapted during training.
  - Laser-DE: also rewards *incorrect long* responses (an exploration bonus).
- **Their truncation baseline on 1.5B (Table 3, avg of 4 benchmarks):**

  | Run | Avg length | Accuracy |
  |---|---|---|
  | Base | 10,177 | 56.9 |
  | T8192 | 2,915 | 55.3 |
  | T4096 | 1,646 | **49.4** |
  | Arora-style "Group" α = 0.2 | 3,085 | 54.7 |
  | LASER L_T = 4096 | 3,636 (−64%) | 58.7 (+1.8) |
  | LASER-D L_T = 4096 | 3,520 | 60.3 |

  - Truncation at 4k with plain GRPO loses badly. DLER (§2.1) argues this is an optimizer problem.

### 2.11 ALP: Adaptive Length Penalty (2506.05256, "Just Enough Thinking")
- **Model:** DeepScaleR-1.5B. **≤2B: yes.**
- **Reward:** r = 1[correct] − β·N·max(p_solved(q), 1/K). p_solved is the online solve rate over K rollouts, so easy prompts pay more per token. Read with 2602.09591's restatement, the penalty is −β·len(y)·max(acc(x), 1/G), i.e. a per-token cost.
- **Settings:** GRPO in veRL; 100 steps; batch 512; context 16,384; β = 1e-7 (1e-8 also tried).
- **Result:** "cuts average token usage by 50% without significantly dropping performance". It moves tokens from easy to hard prompts.

### 2.12 DRPO: Decoupled Reward Policy Optimization (2510.04474)
- **Models:** R1-Distill-1.5B, 7B, R1-Distill-Llama-8B. **≤2B: yes.**
- **Diagnosis:** "when rewards for correct but long rollouts are penalized, GRPO's group-relative advantage function can assign them negative advantages".
- **Method:** builds on DisCO, a discriminative objective with a KL trust-region constraint. Correct rollouts are re-weighted by exp(r_l(o)/λ), normalised **only within the correct group**, with r_l = 1 − |o|/C. λ = +∞ recovers DisCO with no length pressure.
- **Settings:** AdamW, weight decay 0.01; lr 2e-6 (1.5B), 1e-6 (7B), 5e-7 (8B); batch 128; mini-batch 32; 8 responses per question; 8k generation budget for training *and* eval; 1000 steps; λ ∈ {0.5, 0.2, 0.1}.
- **Results:**
  - 1.5B on GSM8K: "77% length reduction with only 1.1% performance loss", versus 68% / −4.3% for the next baseline.
  - 7B: 3,053 → 1,502 tokens (−51%) at −2.6%.

### 2.13 "The Art of Efficient Reasoning: Data, Reward, and Optimization" (2602.20945, HKU/Tencent, 2026)
The only paper I found with a **Qwen3-0.6B** run. **≤2B: yes (0.6B, 1.7B, plus 1.5B ablations).**
- **Reward:** truncation, R = 1[correct]·1[L ≤ L_T]. In their re-implementation, Kimi-style and Laser rewards use α = 0.4.
- **Main protocol:** GRPO; lr 1e-6; clip-high 0.28; batch 128; N = 8; L_R = 16k; L_T = 4k; R1-Distill-1.5B; DeepScaleR data; about 0.2M GPU-hours of ablations.
- **Qwen3 settings (App. E):** L_R = L_T (rollout cap equals target); **8k for Qwen3-0.6B**, 10k for 1.7B; **N = 24**; prompts = **DeepScaleR-Easy (pass rate > 0.5 over 8 rollouts)**; on-policy; batch 128 for 0.6B and 1.7B.
- **Results (Table 2, AIME'25):**

  | Model | Mean@8 | Pass@8 | Length |
  |---|---|---|---|
  | Qwen3-0.6B | 13.33 → 24.58 | 26.67 → 36.67 | 14.9k → 8.9k (−40%), step 640 |
  | Qwen3-1.7B | 35.00 → 38.75 | — | 17.7k → 11.2k (−37%), step 560 |

  - Across Qwen3 0.6B–30B, length falls 15–47%.
- **Findings (with evidence):**
  - **Two stages.** First length adaptation: length falls roughly exponentially, ~6k→2k, and entropy drops. Then reasoning refinement: length is flat while accuracy and entropy recover.
  - **Hard prompts only:** "catastrophic failure", with entropy spiking and length collapsing.
  - **Larger N** (8→24) speeds adaptation and improves refinement on math, with little effect on code.
  - **Masking all incorrect rollouts** (penalising only long-correct ones) is the "short-is-correct trap": entropy explodes after ~400 steps and length collapses.
  - **Masking long-correct rollouts** (the "-L&C" variant) gives better accuracy at longer length.
  - **L_R = L_T = 4k gave the best Pareto front.** Their explanation: negative samples stay short, which avoids the trap.
  - **Kimi-style reward:** best at a 2k eval budget, "suffers from a reasoning collapse issue" at 32k.
  - **Off-policy staleness up to 16** is faster but brings entropy explosion and length rebound. They recommend on-policy.
  - **Math-only training transfers the length bias to LiveCodeBench code.**
  - **Adaptive L_T at the 90th percentile of correct lengths:** shorter, and better at a 4k budget (App. G, Qwen3-0.6B, Fig. 8).

### 2.14 Other 2025–2026 entries (shorter notes)

| Paper | What it does | Size ≤2B? | Key numbers / settings |
|---|---|---|---|
| **GFPO** (2508.09726, Microsoft) | Sample G = 24, train only on the k = 8 shortest (or best reward/token). Adaptive-k: 4/6/8 by difficulty | No (Phi-4-reasoning 14B) | 46–71% less length inflation vs GRPO; 71–85% with reward/token. lr 1e-7, 100 steps |
| **Thinkless** (2505.13379) | `<short>`/`<think>` control token; Decoupled GRPO weights the mode-token loss by α = 1/1000 to prevent mode collapse; distillation warm-up first | Yes (R1-Distill-1.5B) | 50–90% less long-mode use on Minerva Algebra / MATH-500 / GSM8K |
| **LAPO** (2507.15758) | Two stages. Stage 1 rewards length inside the [P30, P70] band of correct lengths (α = 0.7). Stage 2 puts the per-problem median length into the prompt (β = 0.7) | Yes (R1-1.5B, DeepScaleR-1.5B) | Up to −40.9% tokens, +2.3 accuracy; GRPO; 4,096 train cap; 10k problems; 3 episodes per stage |
| **S-GRPO** (2505.07686) | One path, several forced early-exit points; reward decays with exit position | No (Qwen3-8B/14B, R1-7B/14B) | −35.4% to −61.1% length with an accuracy gain |
| **Optimal Reasoning Length** (2602.09591) | Sweeps RLOO-LP, ALP, DRPO; DAPO config with no overlong shaping, fully on-policy, 16K cap, FP16 | Yes (R1-1.5B, **Qwen3-1.7B**) | Accuracy is **non-monotonic** in length. On Qwen3-1.7B the unconstrained token-average baseline "matches or slightly exceeds the within-sweep peak", so length penalties bought little accuracy there. 180 steps on Qwen3-1.7B |
| **CARE** (2609.29664) | Per-question contrastive length reward inside GRPO, no extra hyperparameters | Yes (Qwen3-1.7B) | lr 1e-6; 256 questions × 16; temperature 1.0; **train cap 6,144**; KL 0.001; ε 0.2; MATH data. Qwen3-1.7B average: base 40.06 / 11,586 tokens; **plain GRPO 41.18 / 9,049 (−22%)**; CARE 43.82 / 8,154. Explicit short rewards on Qwen3 caused incorrect responses to grow later (p.6, 4B example) |
| **Correct, Concise and Complete** (2601.02972) | SFT then GRPO; penalty = fraction of tokens after the *first correct answer*, λ = 1 | Yes (Qwen3-1.7B) | GRPO group 8; batch 256; 50 iterations; lr 1e-6; KL 0.001; 16,384 cap. Qwen3-1.7B −22% length with < 2.5 pt drop; the reduction grows with model size |
| **Length penalties and monitorability** (2607.09786) | Length-penalised RL at 60%→30% of baseline CoT length | No (Qwen3-4B, 14B) | Accuracy near baseline at 30%, but hint-verbalising faithfulness −35% (4B) |
| **NAT** (2603.06619) | Trains on a subset of tokens to save *training* compute | n/a | Not about output length; not relevant |

---

## 3. What the evidence says for a 0.6B model

### 3.1 Facts (from the papers above)

1. **Small-scale evidence exists, but almost all of it is on R1-Distill-Qwen-1.5B, which overthinks heavily** (5–16k tokens on math, 32k evaluation). These papers report ≥50% fewer tokens with ≥ −1 pt accuracy at ≤2B:
   - DLER (−77%, accuracy up)
   - ShorterBetter (−77%, +2.5 in-domain / −0.8 out-of-domain)
   - ThinkPrune on the distill model (−65%, +0.6)
   - AdaptThink (−51% to −64%, accuracy up)
   - Arora α = 0.05 on MATH500 (−51%, +0.2)
   - ALP (−50%)
   - DRPO on GSM8K (−77%, −1.1)

   On **Qwen3** small models the reported cuts are smaller:
   - Qwen3-0.6B: −40% (Art, AIME'25)
   - Qwen3-1.7B: −37% (Art), −22% (Correct-Concise-Complete, plain GRPO in CARE)
   - Qwen3-1.7B: penalties beat the unconstrained baseline only marginally (2602.09591)
2. **Reward forms with ≤2B evidence:**
   - truncation / hard cap (ThinkPrune, DLER, Art, Laser baseline)
   - step bonus (LASER)
   - per-prompt z-scored sigmoid (Arora)
   - deviation from the shortest correct answer (ShorterBetter)
   - solve-rate-weighted per-token cost (ALP)
   - target length in the prompt (L1)
   - think/no-think mode bonus (AdaptThink, Thinkless)
   - correct-only normalised weighting (DRPO)
   - post-first-answer fraction (2601.02972)
   - none at all, just training on occasionally solvable problems (Fatemi)
3. **Kimi-style min-max group reward at small scale.** Only the Art paper re-ran it, on 1.5B with α = 0.4. It was strongest under a strict 2k evaluation budget and collapsed at a 32k budget. Kimi never published its weight.
4. **Optimizer choices with small-model evidence:**
   - **Batch-level instead of group-level std normalisation:** DLER, 7B ablation used on 1.5B; Dr. GRPO removes the std entirely.
   - **Clip-higher ε_high = 0.28:** DLER, Art.
   - **Dynamic sampling** (drop all-0 / all-1 groups): DLER.
   - **No or very small KL:** 5e-4 MSE in DLER; 1e-3 in Arora, ShorterBetter, CARE, 2601.02972. With zero-advantage groups, KL can drive the 1.5B model to under 80 tokens (Fatemi).
   - **Length normalisation:** per-response 1/|o| (GRPO, OpenRLHF RLOO) is itself a hidden shortening force for correct answers. Removing it removed the shortening at α = 0 (Arora).
5. **Typical hyperparameters at 1.5B-ish scale** (all full fine-tuning, none LoRA):

   | Setting | Range seen |
   |---|---|
   | lr | 1e-6 to 2e-6 (Arora 5e-6) |
   | group size | 8–16 (Art used 24 for Qwen3) |
   | prompts per step | 64–512 |
   | steps | 100–700; length usually drops in the first ~50–150 steps (Art Stage I; Fatemi < 60 steps; Arora ~100 steps) |
   | training cap | 4k–8k (L1 4k, ThinkPrune 2–4k, DLER 4k, ShorterBetter 6k, Art 8k for 0.6B, CARE 6,144) |
   | sampling temperature | 0.6–1.0 |

6. **Documented failure modes:**
   - **Collapse to short wrong answers:** hard-only prompts (Art); GRPO on unsolvable problems (Fatemi, <80 tokens); masking incorrect rollouts ("short-is-correct trap", Art).
   - **Over-shortening and a plateau below the cap** from all-correct groups dominating the batch (DLER, ~2k under a 4k cap).
   - **Entropy collapse** without clip-higher (DLER); entropy explosion with high staleness (Art).
   - **Incorrect answers growing later in training** (CARE, Qwen3).
   - **Length-overfitting:** the model cannot use more budget later (K2.5).
   - **Accuracy loss on hard problems** with tight one-shot caps (Laser T4096: −7.5 pts; ThinkPrune on saturated models: −2.5).
   - **Less faithful CoT** (2607.09786).
7. **Cross-domain transfer of length habits:**
   - math+code → GPQA/MMLU-Pro (K2.5 Toggle)
   - math → LiveCodeBench (Art)
   - MATH → MMLU-STEM (Fatemi)
   - Kimi K2 instead uses a **per-task-type** budget.

### 3.2 Inferences for our setting (not tested by anyone on our benchmark)

- **[inference] Our scoring rule makes truncation the natural reward.** We score at a fixed 4,096 / 8,192 cap, so a rollout over the cap is already wrong at evaluation time. Truncation with **rollout cap = target = our eval cap or a bit below** (Art's L_R = L_T) aligns the training reward with the metric. It is the cheapest option, because rollouts stop at the cap. DLER's claim that penalty shape matters little once the optimizer is fixed argues against spending effort on fancy penalties.
- **[inference] Optimizer settings for rl_train.py**, in order of evidence:
  1. Clip-higher 0.28.
  2. Dynamic sampling, dropping groups that are all-correct or all-wrong.
  3. Batch-level advantage std (DLER) instead of group std. Dr. GRPO's no-std is also defensible; DLER's argument is that group std is unstable under truncation noise.
  4. KL ≤ 1e-3 or off.
  5. With Dr. GRPO-style constant normalisation, expect **no** implicit shortening. The cap or an explicit term must do all the work. This may partly explain why our past GRPO runs "barely moved" **[inference, unverified against our logs]**.
- **[inference] Prompt selection matters more than the penalty at 0.6B.** Train on prompts the model passes more than half the time (Art: pass rate > 0.5, N = 24). Keep some partially solvable ones (Fatemi: occasionally solvable problems shorten outputs on their own). Keep hard prompts out of the length phase. For us, per-type pools from SVAMP, MBPP+, IFEval and BFCL with measured pass rates at the cap.
- **[inference] Tighten the cap in stages** (ThinkPrune 4k→3k→2k beat one-shot on 1.5B) rather than going straight to the target. Select checkpoints by "shortest length within an accuracy tolerance", as ThinkPrune does, but with our −1 pt tolerance instead of their 10%.
- **[inference] Difficulty awareness is cheap to add** once a group has K samples: DA-DLER thresholds, ALP's solve-rate weight, or K2.5's "only enforce the budget when mean accuracy > λ". It targets exactly the easy-prompt waste that dominates SVAMP-style data.
- **[inference] LoRA is untested.** No paper here uses LoRA. All lr values (1e-6 to 2e-6) are for full fine-tuning. A LoRA run needs a higher lr, typically 10× or more, and that value is not established for this task.
- **[inference] Per-type budgets.** Our four task types have very different natural lengths (BFCL/IFEval answers are short, MBPP+ code is longer). A single cap may over-penalise code. Kimi K2's per-task-type budget, or a per-prompt budget like K2.5's ρ-percentile of correct lengths or K3's τ·b0(x), fits better.
- **[inference] Small models lose less than large ones but gain less too.** k1.5 found larger models more token-efficient, and 2601.02972 found smaller relative reductions on Qwen3-1.7B (−22%) than on 32B (−40%). Reaching ">50% at −1 pt" on Qwen3-0.6B by RL alone is **not yet demonstrated anywhere**. The closest result is −40% on AIME'25 with accuracy *up* (Art), at a 32k budget, not at a fixed 4k/8k cap.

### Sources (all fetched 11 Nov 2026)
arXiv: 2501.12599, 2507.20534, 2602.02276, 2607.24653, 2510.15110, 2504.01296, 2503.04697, 2504.21370, 2504.05185, 2502.04463, 2505.13417, 2503.20783, 2503.14476, 2505.15612, 2506.05256, 2510.04474, 2602.20945, 2508.09726, 2505.13379, 2507.15758, 2505.07686, 2602.09591, 2609.29664, 2601.02972, 2607.09786, 2603.06619. Found via web search: [The Art of Efficient Reasoning](https://www.alphaxiv.org/abs/2602.20945), [Optimal Reasoning Length](https://arxiv.org/html/2602.09591v3), [To Think or Not to Think](https://arxiv.org/html/2609.29664v1), [DRPO](https://arxiv.org/pdf/2510.04474), [Length Penalties and Monitorability](https://arxiv.org/pdf/2607.09786), [2601.02972](https://arxiv.org/pdf/2601.02972v1), [NAT](https://arxiv.org/pdf/2603.06619).
