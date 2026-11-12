# Clean Recipe Design & Strategic Analysis Report: General Reasoning Compression for Qwen3-0.6B

---

## 1. Empirical Analysis: What Works, What Fails, and Why

### 1.1 Stage-by-Stage Trajectory of the Experimental Campaign

The central objective is to compress total generated tokens (thinking plus final response) on **Qwen3-0.6B (thinking mode)** by **more than 50.0%** while maintaining an accuracy change **no worse than -1.0 percentage point** across the 2,759-question ThinkingCap-Test ($\times 8$ rollouts per question; equal 25% weight across SVAMP Math, MBPP+ Code, IFEval, and pooled BFCL [400 one-function / 200 several-functions / 240 none-fits]; token savings defined as $1 - \text{geomean}(r_{\text{math}}, r_{\text{code}}, r_{\text{ifeval}}, r_{\text{bfcl}})$).

Across all full-test evaluations and Dev-512 runs, a clear structural law has emerged: **simultaneously asking a 0.6B model to compress its chain-of-thought and protect task accuracy inside a single undifferentiated optimization stage fails, whereas separating aggressive thinking compression from length-invariant calibration succeeds.**

| Model / Run | Lineage & Method | Full-Test Saved (%) | Full-Test $\Delta$Acc (pts) | Status vs. Box ($>50\%$ / $\ge -1.0$) | Primary Takeaway |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **p1r1s0-200** | 1-step DPO from Original (shorten-dominant + protective pairs) | 43.4% | -1.50 | Fails both | Gradient conflict: protective pairs dilute shortening; shortening pairs still hurt fragile items. |
| **p1cr1s0** | 1-step DPO from Original (heavier protective pair ratio) | *34.0%* *(Dev)* | *+2.90* *(Dev)* | Fails saving | Increasing protection in a 1-step mix stalls compression below 35%. |
| **bidax2 full** | 1-step 1:1 shortest-correct DPO from Original (competitor replica) | 51.2% | -3.00 | Fails accuracy | Pure 1-step compression crosses 50% saving easily but breaks math re-checks and IFEval responses. |
| **rc1-soup-30** | Multi-round DPO (`A1/A2/ca1`) + prefix-cut ("stop when sure") pairs | 57.4% | -3.50 | Fails accuracy | **Overshoot state**: establishes a strong short-thinking prior, but introduces hasty errors. |
| **rpE-40 / 60 / 80** | `rc1` + untyped "right-beats-wrong" DPO repair | 50.6 / 47.8 / 47.0% | -1.9 / -1.2 / -0.9 | Misses box | Untyped repair trades away 10.4% token savings to buy +2.6 pts accuracy because right answers are longer than hasty wrong answers. |
| **rpI-120** | `rpE-80` + further untyped repair on new questions | 41.9% | +0.10 | Fails saving | Confirms linear trade-off of untyped repair: restores full accuracy by undoing compression. |
| **rpK-20 / 40 / 60** | `rpE-80` + typed repair (shorten only IFEval + tools; protect math/code) | 49.9 / **51.2** / 54.4% | -1.3 / **-0.80** / -1.4 | **rpK-40 PASSES** | First entry into the target box, but narrow window: step 60 degrades accuracy to -1.4. |
| **rpL-40 / 60 / 80** | `rpE-80` + typed2 (decouple IFEval response length + anti-loop pairs) | 55.1 / 59.3 / 62.8% | -1.37 / -1.70 / -2.06 | Fails accuracy | Massive saving jump (+8–12%) from killing loops & protecting response length, but math (-2.9) and tools (-5.1) over-compress. |
| **rpM-20 / 40 / 60** | `rpE-80` + typed3 (wide math guards, length-matched be-right, true length rules) | **50.1** / 52.9 / **54.1%** | **-0.94** / -1.10 / **-0.93** | **rpM-20 & rpM-60 PASS** | **Best result (`rpM-60`: 54.1% / -0.93)**. Step 20 $\rightarrow$ 60 adds +4.0% saving at *zero* accuracy cost. |
| **rpN-15 / 30 / 45** | `rpL-60` + "hold" repair (be-right + guards + loops, zero shorten, lr 1e-5) | *58.5 / 59.7%* *(Dev)* | *+1.16 / +0.71* *(Dev)* | Closed (no net gain) | Passive hold without active length-matched contrast cannot fix collapsed tool-call habits once broken. |

### 1.2 Why Single-Round Mixed DPO Fails (`p1r1s0`, `p1cr1s0`, `bidax2`)

When starting directly from the original Qwen3-0.6B, one-step DPO faces a fundamental structural obstacle: in the base model's rollout distribution, **length and correctness are positively correlated on multi-step problems and negatively correlated on simple/looping problems**.
1. **Unprotected 1-step shortening (`bidax2 full`: 51.2% / -3.0)**: Pairing shortest correct against longest correct (or wrong) rollouts across the board hits 51.2% token reduction in one step, proving that multi-round `A1 -> A2 -> ca1` chaining is *not* required just to get 50%+ compression. However, it loses 3.0 accuracy points because: (a) on math, it rewards skipping verification steps; and (b) on IFEval, it rewards shorter total completions, shrinking the final user-facing response.
2. **Mixed shorten + protect in 1 step (`p1r1s0-200`: 43.4% / -1.5; `p1cr1s0`: 34.0% / +2.9)**: Adding "protective" pairs (where a longer right answer beats a shorter wrong answer) into the same DPO batch as shortening pairs (where a shorter right answer beats a longer right answer) creates directly opposing gradients on the `<think>` termination token (`</think>`) and transition markers ("Wait", "Let me double check"). When shortening dominates (`p1r1s0`), compression stalls at 43.4% while accuracy still drops -1.5 pts. When protective pairs are increased (`p1cr1s0`), compression collapses to 34%.

### 1.3 Why Untyped Repair Leaks Savings (`rpE-80`, `rpI-120`) vs. Why `rpM-60` Broke the Trade-Off

After `rc1-soup-30` reached **57.4% saved / -3.5 acc**, the model had acquired a strong short-thinking prior, but its wrong answers were predominantly **premature stops** (thinking traces $0.5\times$ to $0.9\times$ the length of a normal correct trace).
* **The Flaw in Untyped Repair (`rpE` / `rpI`)**: When `rpE` formed standard "right-beats-wrong" DPO pairs ($y^+ = \text{right}$, $y^- = \text{wrong}$) without controlling for length, $\text{len}(y^+)$ was systematically greater than $\text{len}(y^-)$. Consequently, the DPO gradient attributed correctness to *length*, teaching the model to think longer everywhere. From `rc1` to `rpE-80` to `rpI-120`, token savings eroded monotonically from **57.4% $\rightarrow$ 47.0% $\rightarrow$ 41.9%**.
* **The `rpM` Breakthrough (Length-Matched Calibration)**: In `rpM`, right-beats-wrong repair pairs were **length-matched** (selecting $y^+$ and $y^-$ with matched thinking lengths, or requiring $\text{len}_{\text{think}}(y^+) \le 1.1\times \text{len}_{\text{think}}(y^-)$) alongside wide guards that only intervened on catastrophic truncation, while decoupling thinking length from response length on instruction tasks. Because correctness gradients were made orthogonal to sequence length, training from step 20 (`rpM-20`: 50.1% / -0.94) to step 60 (`rpM-60`: 54.1% / -0.93, 95% CI `[53.0, 55.3]` / `[-1.67, -0.17]`) increased token savings by **+4.0 percentage points at zero accuracy cost**.

### 1.4 Per-Task Mechanics and Failure Modes (with Numbers)

Comparing `rpM-60` (54.1% saved / -0.93 acc) against `rpL-80` (62.8% saved / -2.06 acc) reveals the exact operating boundary for each reasoning regime:

| Task / Split | Weight | `rpM-60` Saved | `rpM-60` $\Delta$Acc | `rpL-80` Saved | `rpL-80` $\Delta$Acc | Governing Failure Mechanism & Sweet Spot |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **SVAMP (Math)** | 25.0% | 57% | -1.2 pts | 66% | -2.9 pts | **Skipped verification**: Pushing math savings past ~58% eliminates the single arithmetic self-check needed on multi-step word problems. Sweet spot: **55–58% saved**. |
| **MBPP+ (Code)** | 25.0% | 58% | -0.9 pts | 66% | -1.8 pts | **Edge-case omission & loops**: Base model wastes 20–30% tokens in repetitive edge-case loops; cutting loops gives free savings up to ~58%, beyond which boundary conditions are missed. Sweet spot: **56–60% saved**. |
| **IFEval** | 25.0% | 50% | -2.3 pts | 62% | -2.6 pts | **Response-length coupling & cap loops**: Naive shortening shrinks the *final answer* on length/count constraints ("at least $N$ words/sentences"). Killing cap-hitting loops saves 20–30% tokens; preserving response length keeps $\Delta$Acc near -1.5 to -2.0. Sweet spot: **50–55% saved**. |
| **BFCL: One Function** | 11.9% | 45% | -0.3 pts | 51% | -5.1 pts | **Schema/call collapse**: Below 47% saving, accuracy is virtually lossless (-0.3 pts). Above 50% saving (`rpL-80`: -5.1; `rpL-100/120`: -32 / -41 pts), the model skips parameter extraction and stops emitting tool calls. Sweet spot: **44–47% saved**. |
| **BFCL: Several** | 6.0% | 47% | -1.6 pts | 53% | -2.9 pts | **Multi-call planning**: Requires sequential argument mapping; tolerates up to ~47% compression before dropping secondary calls. Sweet spot: **45–48% saved**. |
| **BFCL: None Fits** | 7.1% | 61% | **+4.2 pts** | 66% | **+7.7 pts** | **Overthinking hallucination**: When the original model thinks too long on an unsolvable tool request, it talks itself into calling an invalid tool. Shortening thinking produces a massive **+4.2 to +7.7 pt accuracy gain**. Sweet spot: **60–65% saved**. |

### 1.5 General Behaviors vs. Benchmark-Tuned Artifacts

To satisfy the requirement that the model learns a **general "think less" capability** that transfers to unseen benchmarks (MATH-500, Skywork, ARC-C, CSQA, MMLU, HumanEval, AIME 2025) without rules keyed to dataset names:

* **What is Genuinely General (Keep and Formalize)**:
  1. **Strict Thinking/Response Decoupling**: Measuring and penalizing length *exclusively* inside `<think>...</think>` while requiring $\text{len}(\text{response of } y^+) \ge 0.95 \times \text{len}(\text{response of } y^-)$. A model should think concisely, never answer curtly. This solves the IFEval length-constraint failure generally across all open-ended generation tasks without inspecting the prompt for "at least $N$ words".
  2. **Early-Exit Prefix Cutting ("Stop When Sure")**: Truncating a correct thinking trace right after the first sentence/paragraph that derives the final answer (stripping redundant "Wait, let me re-verify..." cycles) teaches the model to recognize internal convergence.
  3. **Anti-Loop / Cap-Hit Elimination**: Preferring any clean, terminated rollout over a rollout that hits the token limit or exhibits $n$-gram repetition. Runaway repetition is a universal 0.6B pathology, not a benchmark quirk.
  4. **Uncertainty-Gated Length-Matched Calibration**: Using the empirical pass rate $\hat{p}_q = \frac{1}{K}\sum_{k=1}^K \mathbb{I}[\text{correct}(y_{q,k})]$ across the $K=8$ rollouts of question $q$ as a universal proxy for question difficulty. Easy questions ($\hat{p}_q \ge 0.75$) receive aggressive shortening; uncertain/hard questions ($0 < \hat{p}_q < 0.75$) receive **length-matched right-beats-wrong** pairs and floor guards so the model never learns to skip necessary computation on hard problems.

* **What Looked Benchmark-Tuned in Earlier Runs (Eliminate)**:
  1. **Dataset-Name Branches**: Rules of the form `if task == 'svamp'` vs. `if task == 'ifeval'` in `repair_pairs.py`. In our clean recipes below, every rule is driven strictly by **measured per-question statistics** ($\hat{p}_q$, thinking-length coefficient of variation $CV_q$, and response-length variance).
  2. **Prompt Regexes for Counting/Length**: Scanning prompts for literal strings like `"words"` or `"sentences"` to switch pair rules. Decoupling `<think>` compression from response length makes prompt regexes completely unnecessary: when $y^+$ is required to have a response at least as long as the median correct rollout for question $q$, any prompt whose verifier requires a long response automatically preserves response length.

### 1.6 Connections to the Research Literature (26-Paper Pack)

The empirical findings in our lab notebook directly mirror theoretical and empirical laws established in recent efficient-reasoning literature:
* **Overthinking on Simple & Unanswerable Tasks (`arXiv:2412.21187`, `arXiv:2502.03373`)**: Chen et al. (`arXiv:2412.21187`, *"Do NOT Think That Much for 2+3=?"*) show that reasoning models generate their first valid solution inside the first 30–40% of the thinking trace and spend the remaining 60–70% in self-doubt loops ("Wait, let me check again"). Furthermore, Yeo et al. (`arXiv:2502.03373`) show that excessive CoT exploration increases hallucination when no valid action exists—explaining why BFCL `none fits` jumps by **+4.2 to +7.7 accuracy points** when thinking is compressed by 61–66%.
* **Length-Harmonized Pruning & Prefix Cutting (`arXiv:2501.12570`, `arXiv:2501.12599`)**: O1-Pruner (`arXiv:2501.12570`) and Kimi k1.5 (`arXiv:2501.12599`) demonstrate that naive length penalties degrade accuracy because they penalize necessary reasoning steps on hard problems equally with redundant verification on easy problems. Normalizing length relative to per-question reference distributions and bootstrapping with shortest-correct / prefix-truncated trajectories prevents reasoning collapse.
* **Difficulty-Adaptive Thinking & Hard-Math Transfer (`arXiv:2602.20945`, `arXiv:2412.18547`, `arXiv:2503.04697`)**: As measured in `arXiv:2602.20945` (which evaluates compressed reasoning models on AIME 2025 under a 32k token cap) and TALE (`arXiv:2412.18547`), a well-compressed model must exhibit *elastic* token allocation: compressing by 55–70% on routine problems (SVAMP, GSM8K, ARC-C, easy MATH) while dynamically expanding its thinking trace to thousands of tokens when problem complexity demands multi-step search (MATH Level 5, AIME 2025). By conditioning shortening pressure on empirical pass rate $\hat{p}_q$ and using length-matched repair on low-$\hat{p}_q$ items, our recipes preserve deep thinking on low-pass-rate problems rather than imposing a rigid token ceiling (`arXiv:2503.04697`).

---

## 2. Three Clean Recipes from Original Qwen3-0.6B (Ranked)

All three recipes start from the **original Qwen3-0.6B** weights, use **only the model's own generated answers** on a fixed pool of **4,000 public training prompts** (1,000 math word problems [GSM8K/MathQA], 1,000 Python coding tasks [MBPP-train/KodCode], 1,000 verifiable instruction-following prompts [tulu-3-if/IFEval-like synthetic constraints], and 1,000 tool-use prompts [glaive-function-calling/xlam-function-calling], with zero overlap with ThinkingCap-Test), require **no benchmark-name rules** during pair construction or reward calculation, and run comfortably on **two Colab L4 GPUs (24 GB VRAM each)** using LoRA.

---

### Recipe 1 (RANK 1 — Primary Recommendation): Two-Stage "Overshoot-Then-Calibrate" Decoupled DPO

#### The Idea in Two Sentences
Step 1 aggressively compresses `<think>` tokens into the 57–60% savings zone in a single DPO pass from the original model by combining self-truncated prefix-cut thoughts with shortest-thinking correct rollouts while strictly holding final response length neutral. Step 2 takes fresh on-policy rollouts from Step 1 and recovers accuracy to $\ge -0.7$ pts without losing savings by applying **length-matched** right-beats-wrong DPO pairs on uncertain questions alongside anti-loop and truncation-floor guards.

#### One-Line Pathway
`Original Qwen3-0.6B` $\longrightarrow$ **[Step 1: Offline DPO for Thinking-Only Compression** *(Job: cut `<think>` length by ~58% and eliminate loops without shrinking final responses)* **]** $\longrightarrow$ **[Step 2: Offline DPO for Length-Matched Calibration** *(Job: correct hasty premature-stop errors on uncertain items while holding token savings fixed at 53–55%)* **]** $\longrightarrow$ `Final Model`

#### Exact Procedure from the Original Model

**Preparation (On-Policy Rollouts for Step 1)**:
* Sample $K = 8$ rollouts per prompt on the 4,000 training prompts from the **Original Qwen3-0.6B** ($T = 0.7, \text{top\_p} = 0.95$, max new tokens $= 4096$). Grade all rollouts with deterministic programmatic verifiers ($r \in \{0, 1\}$).
* For every rollout $y$, parse $L_{\text{think}}(y)$ (tokens inside `<think>...</think>`) and $L_{\text{resp}}(y)$ (tokens after `</think>`).
* **Self-Generated Prefix-Cut Augmentation ("Stop When Sure")**: On prompts with pass rate $\hat{p}_q \ge 0.5$, take the longest correct rollout, truncate its `<think>` block at the first sentence boundary after the first internal self-verification phrase (`"\nWait,"`, `"\nAlternatively,"`, `"\nLet me double check"`, `"\nLet's verify"`) that occurs at $\ge 35\%$ of $L_{\text{think}}$, append `</think>\n`, and let Original Qwen3-0.6B greedy-decode the response. Keep only cut completions that pass the verifier ($r = 1$).

**Step 1: Aggressive Thinking-Only Compression DPO**
* **Single Job**: Shift the thinking length distribution down by ~58% while keeping response length invariant and eradicating cap-hitting loops.
* **Measured Per-Question Pair Rules (No Benchmark Names)**:
  1. *Prefix-Cut Shortening Pairs (35% of pool, ~1,050 pairs)*: On prompts where a valid prefix-cut rollout $y_{\text{cut}}^+$ passed verification and $L_{\text{think}}(y_{\text{cut}}^+) \le 0.60 \times L_{\text{think}}(y_{\text{orig}}^+)$, pair $y^+ = y_{\text{cut}}^+$ against $y^- = y_{\text{orig}}^+$, subject to response-length neutrality: $L_{\text{resp}}(y^+) \ge 0.95 \times L_{\text{resp}}(y^-)$.
  2. *Natural Decoupled Shortening Pairs (50% of pool, ~1,500 pairs)*: On prompts with at least two natural correct rollouts, select $y^+ = \arg\min_{y \in \mathcal{C}_q} L_{\text{think}}(y)$ and $y^- = \arg\max_{y \in \mathcal{C}_q} L_{\text{think}}(y)$ such that:
     * $L_{\text{think}}(y^+) \le 0.65 \times L_{\text{think}}(y^-)$ (at least 35% thinking reduction),
     * $L_{\text{think}}(y^+) \ge 32$ tokens (non-empty thinking floor so `<think>` never collapses),
     * **$L_{\text{resp}}(y^+) \ge 0.95 \times \text{median}_{y \in \mathcal{C}_q} L_{\text{resp}}(y)$** (response completeness lock: prevents shrinking responses on constraint/long-form questions).
  3. *Anti-Loop / Cap-Hit Pairs (15% of pool, ~450 pairs)*: On any prompt where at least one rollout hit the 4,096 token cap or contained $\ge 4$ consecutive repeated 20-gram blocks ($y_{\text{loop}}^-$) and at least one rollout passed verification cleanly ($y^+$), pair $(y^+, y_{\text{loop}}^-)$.
* **Dataset Size & Training Settings (Step 1)**:
  * Total pairs: **3,000 pairs** shuffled uniformly.
  * Reference model: Original Qwen3-0.6B (implicit via LoRA adapter disable).
  * LoRA settings: `r = 32`, `lora_alpha = 64`, `lora_dropout = 0.05`, target modules `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`.
  * Optimizer & Schedule: AdamW, `lr = 5e-5`, cosine decay, warmup ratio `0.10`, DPO $\beta = 0.10$, effective batch size `32` (micro-batch 4 $\times$ grad accum 4 $\times$ 2 GPUs), max length `4096`.
  * Duration & Checkpoints: **100 steps** (~1.06 epochs). Save checkpoints at steps `60, 80, 100`; evaluate on Dev-512 and pick the checkpoint in the **56%–61% saved** overshoot window (expected at step 80). Merge LoRA weights into base before Step 2 so Step 2 uses Step 1 as its exact reference policy $\pi_{\text{ref}}$.

**Step 2: Length-Matched Accuracy Calibration DPO**
* **Single Job**: Repair premature-stop errors and protect multi-step reasoning on uncertain prompts without re-inflating thinking length.
* **Fresh On-Policy Rollouts**: Using the merged Step 1 checkpoint, generate $K = 8$ rollouts per prompt across the 4,000 training prompts ($T = 0.7$). Compute per-question pass rate $\hat{p}_q \in \{0, \frac{1}{8}, \dots, 1\}$ and median correct thinking length $M_q = \text{median}_{y \in \mathcal{C}_q} L_{\text{think}}(y)$.
* **Measured Per-Question Pair Rules (No Benchmark Names)**:
  1. *Length-Matched Right-Beats-Wrong Pairs (50% of pool, ~1,200 pairs)*: On uncertain prompts where $0 < \hat{p}_q \le 0.75$ (at least one right and one wrong rollout), pair a correct rollout $y^+$ against a wrong rollout $y^-$ that failed via normal termination (not a cap-hit), subject to the **strict length-matching constraint**:
     $$0.75 \times M_q \le L_{\text{think}}(y^+) \le 1.10 \times L_{\text{think}}(y^-) \quad \text{and} \quad L_{\text{resp}}(y^+) \ge 0.95 \times L_{\text{resp}}(y^-)$$
     *Why this works*: Because $y^+$ is no more than 10% longer in `<think>` than $y^-$ (and often slightly shorter), the DPO gradient rewards *valid reasoning structure, tool-call syntax, and arithmetic verification* rather than token volume.
  2. *Anti-Collapse Floor Guards (20% of pool, ~480 pairs)*: On prompts where a wrong rollout $y_{\text{hasty}}^-$ prematurely stopped at $L_{\text{think}}(y_{\text{hasty}}^-) < 0.45 \times M_q$ (skipped re-check or skipped tool call), pair it against the *shortest* correct rollout $y^+$ that satisfies $L_{\text{think}}(y^+) \in [0.75 M_q, 1.05 M_q]$.
  3. *Residual Anti-Loop & Easy-Item Maintenance Pairs (30% of pool, ~720 pairs)*: Half (~360 pairs) are $(y^+, y_{\text{loop}}^-)$ on any remaining cap-hitting/repetitive failures; half (~360 pairs) are decoupled shortest-beats-longest correct pairs restricted strictly to easy, saturated prompts ($\hat{p}_q \ge 0.875$ and $M_q > 180$ tokens) with $L_{\text{resp}}(y^+) \ge L_{\text{resp}}(y^-)$.
* **Dataset Size & Training Settings (Step 2)**:
  * Total pairs: **2,400 pairs**.
  * Reference model: Merged Step 1 model. Fresh LoRA adapter (`r = 32, alpha = 64`).
  * Optimizer & Schedule: AdamW, `lr = 2e-5` (lower LR for surgical calibration), cosine schedule, warmup `0.10`, DPO $\beta = 0.15$ (tighter KL anchor to Step 1), effective batch size `32`.
  * Duration & Checkpoints: **60 steps** (~0.8 epochs). Save and evaluate on Dev-512 at steps `20, 40, 60`.

#### How It Stays General
Not a single line of code inspects benchmark names or prompt keywords.
* A hard math word problem or multi-function tool call naturally exhibits $0.125 \le \hat{p}_q \le 0.75$ and premature-stop wrong rollouts ($L_{\text{think}}(y^-) < 0.45 M_q$), automatically triggering Length-Matched Right-Beats-Wrong and Floor Guard pairs.
* An IFEval prompt with a length rule naturally fails whenever $L_{\text{resp}}$ drops, so $\mathcal{C}_q$ only contains full-length responses, and the invariant $L_{\text{resp}}(y^+) \ge 0.95 \times \text{median}_{y \in \mathcal{C}_q} L_{\text{resp}}(y)$ guarantees that Step 1 and Step 2 never compress the final response.
* An unanswerable tool prompt (`none fits`) or trivial question has short valid derivations and high $\hat{p}_q$, receiving full thinking compression.

#### Expected Full-Test Range & Evidence
* **Expected Full-Test Score**: **53.0% to 56.0% tokens saved** at **-0.50 to -0.95 accuracy change** (**CLEARS $>50\%$ / $> -1.0$**).
* **Empirical Evidence**:
  * Step 1 combines the exact two mechanisms that drove `bidax2` (51.2% in 1 step) and `rc1-soup-30` (57.4% via prefix-cut pairs), while adding `rpL`'s response-length decoupling from day one (which worth +4–5% savings and +0.6 pts IFEval accuracy over `rc1`).
  * Step 2 directly applies the `rpM` length-matched calibration principle that achieved **54.1% / -0.93** on full test (`rpM-60`) and gained +4.0% savings between steps 20 and 60 at zero accuracy cost, without first losing 10.4% savings in an untyped `rpE` stage.

#### Compute Budget, Decision Rules, and Main Risk
* **GPU-Hours (2x Colab L4)**: Step 1 rollouts + cut decoding: `2.0 h` | Step 1 DPO (100 steps): `0.6 h` | Step 2 rollouts: `1.2 h` | Step 2 DPO (60 steps): `0.4 h` | Dev-512 + Full Test evals: `2.8 h` $\rightarrow$ **Total: ~7.0 L4-hours (3.5 wall-clock hours on 2x L4)**.
* **Decision Rule After Step 1**: On Dev-512, Step 1 must reach **56.0%–61.0% tokens saved** with $\Delta\text{Acc} \ge -3.5$ pts and zero increase in cap-hit rate. If savings are $<55\%$, pick step 100 (or train 20 more steps); if savings are $>62\%$ and tool calls drop by $>4$ pts, fall back to the step 60 checkpoint.
* **Decision Rule After Step 2**: On Dev-512, pick the checkpoint (`20, 40, 60`) that maximizes token savings subject to Dev-512 $\Delta\text{Acc} \ge +0.5$ pts (since Dev-512 runs ~1.2–1.5 pts more optimistic on accuracy than the 2,759-question full test, as seen in `rpM-60`).
* **Main Risk & Mitigation**: Prefix-cut thoughts in Step 1 can sound abrupt if cut mid-calculation, causing tool-call syntax errors. *Mitigation*: Only accept prefix-cut candidates where greedy continuation from `</think>\n` passes the exact task verifier ($r=1$) and cap prefix-cut pairs at 35% of Step 1's pool.

---

### Recipe 2 (RANK 2 — On-Policy RL Alternative): Two-Stage "DPO Bootstrap $\rightarrow$ Difficulty-Gated GRPO"

#### The Idea in Two Sentences
Step 1 runs a single moderate DPO round on shortest-correct and prefix-cut pairs from the original model to pull thinking length down by ~45–48% so subsequent on-policy RL rollouts already sample inside the concise regime. Step 2 runs on-policy Group Relative Policy Optimization (GRPO) with a piecewise verifiable reward that gives a group-normalized `<think>` brevity bonus *only* on high-pass-rate prompts ($\hat{p}_q \ge 0.5$) while rewarding pure accuracy with zero length pressure on difficult prompts ($\hat{p}_q < 0.5$).

#### One-Line Pathway
`Original Qwen3-0.6B` $\longrightarrow$ **[Step 1: Offline DPO Bootstrap** *(Job: shift base thinking distribution down by ~46% and remove cap loops)* **]** $\longrightarrow$ **[Step 2: On-Policy Difficulty-Gated GRPO** *(Job: push token savings to 52–55% while directly maximizing exact verifier pass rate)* **]** $\longrightarrow$ `Final Model`

#### Exact Procedure from the Original Model

**Step 1: Offline DPO Bootstrap**
* **Single Job**: Move the policy out of the verbose base regime into the ~46% savings zone cleanly so on-policy GRPO does not waste 80% of its rollouts exploring 2,000-token traces.
* **Data & Rules**: Follow the exact Step 1 rollout and pair construction of Recipe 1 (natural decoupled shortest-correct + prefix-cut + anti-loop pairs with $L_{\text{resp}}(y^+) \ge 0.95 \times L_{\text{resp}}(y^-)$ on the first 2,500 training prompts, yielding **2,000 pairs**), but stop earlier at the moderate-compression checkpoint.
* **Settings**: LoRA `r = 32, alpha = 64`, `lr = 4e-5`, $\beta = 0.12$, batch size `32`, **60 steps**. Merge the step-60 checkpoint (~45–48% saved, ~ -1.5 acc on Dev-512).

**Step 2: On-Policy Difficulty-Gated GRPO**
* **Single Job**: Jointly optimize verifier accuracy and adaptive thinking brevity online using group-relative advantages (`arXiv:2501.12599`, `arXiv:2501.12570`).
* **On-Policy Sampling**: On the remaining 1,500 training prompts (plus cycling through the full 4,000 pool), sample a group of $G = 8$ rollouts per prompt $\{y_1, \dots, y_8\}$ from the current policy ($T = 0.8$, max tokens $= 3072$).
* **General Per-Question Reward Function (No Benchmark Names)**:
  Let $c_i \in \{0, 1\}$ be the programmatic verifier score for rollout $y_i$, $\hat{p}_q = \frac{1}{8}\sum_{j=1}^8 c_j$ be the group pass rate, $\text{cap}(y_i) \in \{0,1\}$ indicate hitting the token limit or repeating a 20-gram $\ge 4$ times, and $\mu_{\text{think}}^+, \sigma_{\text{think}}^+$ be the mean and standard deviation of $L_{\text{think}}(y_j)$ among *correct* rollouts in the group. Define the scalar reward $R(q, y_i)$:
  1. *If $c_i = 0$ (Incorrect)*:
     $$R(q, y_i) = -0.50 \cdot \text{cap}(y_i) - 0.25 \cdot \mathbb{I}\big[L_{\text{think}}(y_i) < 0.50 \cdot \mu_{\text{think}}^+\big]$$
     *(Penalizes both runaway loops and hasty premature stops that cut thinking below half of the group's working length).*
  2. *If $c_i = 1$ (Correct)*:
     $$R(q, y_i) = 1.0 + \lambda(\hat{p}_q) \cdot \text{clip}\!\left(\frac{\mu_{\text{think}}^+ - L_{\text{think}}(y_i)}{\sigma_{\text{think}}^+ + 10},\; -1.0,\; +1.0\right) - 0.30 \cdot \mathbb{I}\!\left[L_{\text{resp}}(y_i) < 0.90 \cdot \text{median}_{j: c_j=1} L_{\text{resp}}(y_j)\right]$$
     where the difficulty-gating coefficient $\lambda(\hat{p}_q)$ is:
     $$\lambda(\hat{p}_q) = \begin{cases} 0.35 & \text{if } \hat{p}_q \ge 0.75 \quad (\text{easy/certain: strong brevity bonus}) \\ 0.15 & \text{if } 0.50 \le \hat{p}_q < 0.75 \quad (\text{moderate: mild brevity bonus}) \\ 0.00 & \text{if } \hat{p}_q < 0.50 \quad (\text{hard/uncertain: pure accuracy, zero length penalty}) \end{cases}$$
* **Training Settings (Step 2)**:
  * Algorithm: GRPO with group-normalized advantages $A_i = \frac{R_i - \text{mean}(R)}{\text{std}(R) + 10^{-4}}$, clipping $\epsilon = 0.20$, explicit KL penalty against merged Step 1 model $\beta_{\text{KL}} = 0.04$.
  * LoRA `r = 32, alpha = 64`, `lr = 1e-5`, cosine schedule, batch size `16 prompts x 8 rollouts = 128 sequences per step`.
  * Duration & Checkpoints: **120 GRPO steps** (evaluating on Dev-512 at steps `40, 80, 120`).

#### How It Stays General
The reward function $R(q, y_i)$ depends exclusively on $(c_i, L_{\text{think}}(y_i), L_{\text{resp}}(y_i), \text{cap}(y_i))$ measured within the 8-rollout group of prompt $q$. Hard math or complex multi-tool prompts naturally have $\hat{p}_q < 0.50$, setting $\lambda(\hat{p}_q) = 0$ so GRPO optimizes pure accuracy and penalizes hasty guesses ($-0.25$). Easy questions and `none fits` prompts have $\hat{p}_q \ge 0.75$, activating $\lambda = 0.35$ to compress thinking aggressively while the response-length guard ($-0.30$) protects instruction-following responses.

#### Expected Full-Test Range & Evidence
* **Expected Full-Test Score**: **51.5% to 54.5% tokens saved** at **-0.40 to -0.90 accuracy change** (**CLEARS $>50\%$ / $> -1.0$**).
* **Evidence**: Matches the Long2Short RL formulation of Kimi k1.5 (`arXiv:2501.12599`) and O1-Pruner (`arXiv:2501.12570`), which show that on-policy group-relative updates after an offline length-shortening bootstrap achieve higher accuracy retention than pure offline preference optimization because the policy directly observes when its own shortened traces fail the verifier.

#### Compute Budget, Decision Rules, and Main Risk
* **GPU-Hours (2x Colab L4)**: Step 1 rollouts + DPO: `2.4 h` | Step 2 on-policy vLLM/fast generation + GRPO (120 steps $\times$ 128 seqs): `5.5 h` | Evaluations: `2.6 h` $\rightarrow$ **Total: ~10.5 L4-hours**.
* **Decision Rules**: After Step 1, verify Dev-512 is in `[44%, 49%]` saved and $\Delta\text{Acc} \ge -1.6$. During Step 2, stop at the first checkpoint (`40, 80, 120`) where Dev-512 savings cross **53.5%** with Dev-512 $\Delta\text{Acc} \ge +0.6$.
* **Main Risk**: On-policy RL on a 0.6B model can find reward hacks (e.g., collapsing `<think>` to a single boilerplate sentence on high-$\hat{p}_q$ batches). *Mitigation*: Enforce a hard reward mask setting $R(q, y_i) = -0.50$ whenever $L_{\text{think}}(y_i) < 25$ tokens, and keep $\beta_{\text{KL}} = 0.04$ anchored to Step 1.

---

### Recipe 3 (RANK 3 — Three-Step Surgical Separation): "Prefix-Cut SFT $\rightarrow$ Compression DPO $\rightarrow$ Calibration DPO"

#### The Idea in Two Sentences
By decomposing the historical multi-stage chain into three strictly orthogonal learning steps—Step 1 (Rejection SFT on self-generated prefix-cut thoughts to teach clean early-exit syntax), Step 2 (pure thinking-shortening DPO to overshoot savings to ~59%), and Step 3 (pure length-matched right-beats-wrong DPO to calibrate accuracy)—no step ever mixes artificial cut strings with natural rollouts or mixes shortening with repair. This achieves the cleanest gradient separation and highest savings ceiling within the 3-step limit.

#### One-Line Pathway
`Original Qwen3-0.6B` $\longrightarrow$ **[Step 1: Self-Cut Rejection SFT** *(Job: teach clean early `</think>` transition after first valid derivation)* **]** $\longrightarrow$ **[Step 2: Thinking-Only Compression DPO** *(Job: overshoot savings to 58–61% on natural rollouts)* **]** $\longrightarrow$ **[Step 3: Length-Matched Calibration DPO** *(Job: restore accuracy to $> -0.8$ pts at fixed length)* **]** $\longrightarrow$ `Final Model`

#### Exact Procedure from the Original Model
* **Step 1 (Self-Cut Rejection SFT)**:
  * From the $4,000 \times 8$ rollouts of the Original Qwen3-0.6B, select **1,600 verified self-completions**: 800 verified prefix-cut traces (where truncating `<think>` after the first answer derivation and greedy-continuing passed the verifier) and 800 natural shortest-thinking correct traces on prompts where $\hat{p}_q \ge 0.5$ and $L_{\text{resp}}$ is at or above the prompt's median correct response length.
  * Train SFT for **1 epoch (50 steps at batch size 32)**, `lr = 2e-5`, LoRA `r = 32, alpha = 64`, loss masked to completion tokens (`<think>` + response). Merge adapter. *(Dev-512 after Step 1: ~35–40% saved, ~ -0.5 acc).*
* **Step 2 (Thinking-Only Compression DPO)**:
  * Generate $K = 8$ on-policy rollouts per prompt from the merged Step 1 model. Because Step 1 already taught early-exit transitions, natural rollouts now include clean short thoughts without needing synthetic prefix cuts inside DPO.
  * Build **2,500 natural shortening + anti-loop pairs** using Recipe 1's exact decoupled shortening rule ($L_{\text{think}}(y^+) \le 0.65 L_{\text{think}}(y^-)$, $L_{\text{resp}}(y^+) \ge 0.95 \times \text{median } L_{\text{resp}}$) and anti-loop rule.
  * Train DPO for **80 steps**, `lr = 4e-5`, $\beta = 0.10$, batch size `32`, LoRA `r = 32, alpha = 64`. Merge best overshoot checkpoint (~58–60% saved, ~ -2.2 acc).
* **Step 3 (Length-Matched Calibration DPO)**:
  * Generate $K = 8$ on-policy rollouts from the merged Step 2 model and run Recipe 1's exact Step 2 (**2,400 pairs**: 50% length-matched right-beats-wrong on $0 < \hat{p}_q \le 0.75$, 20% anti-collapse floor guards, 30% anti-loop & saturated-item maintenance; **60 steps**, `lr = 2e-5`, $\beta = 0.15$).

#### How It Stays General, Expected Full-Test Range & Trade-Offs
* **Generality**: Uses the exact same task-agnostic per-question rollout statistics ($\hat{p}_q, M_q, L_{\text{think}}, L_{\text{resp}}$) as Recipe 1.
* **Expected Full-Test Score**: **53.5% to 57.0% tokens saved** at **-0.60 to -0.95 accuracy change** (**CLEARS $>50\%$ / $> -1.0$**).
* **GPU-Hours (2x Colab L4)**: **~8.5 L4-hours** (three rollout passes + 50 SFT steps + 80 DPO steps + 60 DPO steps + evaluations).
* **Decision Rules**: Step 1 must reach 35–40% saved with $\Delta\text{Acc} \ge -0.8$ on Dev-512; Step 2 must reach 57–61% saved; Step 3 must bring Dev-512 $\Delta\text{Acc}$ to $\ge +0.5$ while retaining $\ge 54\%$ savings.
* **Main Risk**: SFT on a 0.6B model can narrow rollout diversity before Step 2 DPO if over-trained. *Mitigation*: Strictly cap Step 1 SFT at 1 epoch (50 steps) with `lr = 2e-5` and sample Step 2 rollouts at $T = 0.8$.

---

## 3. Which Recipe to Run First & The Cheapest Gate Experiment

### 3.1 Why Run Recipe 1 First
**Run Recipe 1 (Two-Stage "Overshoot-Then-Calibrate" Decoupled DPO) first.**
1. **Direct Distillation of Proven Physics**: Every component of Recipe 1 has already been validated in the lab notebook (`rc1` proved prefix-cut + shortening DPO hits 57.4% savings; `rpL` proved response-length decoupling and anti-loop pairs unlock +5–8% savings on IFEval/Code; `rpM-60` proved length-matched calibration recovers accuracy to -0.93 without giving back savings). Recipe 1 collapses the 6-stage historical chain (`A1 -> A2 -> ca1 -> rc1 -> rpE -> rpM`) into **2 clean, non-overlapping DPO steps from the original model** by eliminating the wasteful untyped repair stage (`rpE`) and imposing response-length invariance from Step 1.
2. **Lowest Engineering & Optimization Variance**: Unlike Recipe 2 (which requires tuning online GRPO generation and KL schedules on a 0.6B model) and Recipe 3 (which requires an extra rollout generation pass between SFT and DPO), Recipe 1 uses the exact offline DPO training pipeline (`dpo.py`) already verified in the codebase.

### 3.2 The Cheapest Gate Experiment (~2.5 L4 GPU-Hours Before Full Spend)
Before running all 4,000 prompts and full-test evaluations (7.0 L4-hours), run a **1,200-Prompt Mini-Pilot Gate** (~2.5 L4 GPU-hours total; ~1.25 wall-clock hours on 2x L4) to verify the two critical transitions of Recipe 1:

1. **Gate 1 — Can Clean Step 1 Hit the 56–60% Overshoot Zone Directly from Original without Breaking IFEval Responses? (~1.3 L4-hours)**
   * Sample $K = 8$ rollouts from Original Qwen3-0.6B on a stratified 1,200-prompt subset of the training pool (300 math, 300 code, 300 instruction, 300 tool).
   * Build **1,200 Step-1 pairs** (35% verified prefix-cut, 50% natural decoupled shortest-correct with $L_{\text{resp}}(y^+) \ge 0.95 \times \text{median } L_{\text{resp}}$, 15% anti-loop).
   * Train Step 1 DPO for **60 steps** (`batch_size = 32, lr = 5e-5, beta = 0.10`) and evaluate on **Dev-512** at step 60.
   * **Pass Criterion**: Dev-512 total token savings $\in [56.0\%, 61.0\%]$, IFEval average response length (excluding `<think>`) remains within $\ge 94\%$ of the original model's response length, and Dev-512 $\Delta\text{Acc} \ge -3.0$ pts. *(Note: `p1r1s0` stalled at 43.4% because it mixed protective pairs into Step 1, and `bidax2` hit -3.0 on full test because it shrank responses and lacked Step 2 calibration. Passing Gate 1 proves Step 1 reaches the `rc1` overshoot state cleanly in one pass from the original model).*

2. **Gate 2 — Does Length-Matched Step 2 Recover $\ge +1.8$ Accuracy Points on Dev-512 While Losing $< 3.0\%$ Savings? (~1.2 L4-hours)**
   * Merge Gate 1's step-60 adapter, sample $K = 8$ rollouts on the same 1,200 training prompts, and build **960 Step-2 pairs** using Recipe 1's exact Step 2 rules (50% length-matched right-beats-wrong where $L_{\text{think}}(y^+) \le 1.10 L_{\text{think}}(y^-)$ on $0 < \hat{p}_q \le 0.75$, 20% floor guards, 30% anti-loop/easy maintenance).
   * Train Step 2 DPO for **40 steps** (`lr = 2e-5, beta = 0.15`) and evaluate on **Dev-512** at steps 20 and 40.
   * **Pass Criterion**: Dev-512 accuracy improves by **$\ge +1.8$ points** over Gate 1 (reaching Dev-512 $\Delta\text{Acc} \ge +0.5$ pts relative to Original) while token savings stay **$\ge 53.5\%$** (losing $< 3.0\%$ savings from Gate 1).
   * **Go/No-Go Action**: If Gate 1 and Gate 2 pass, immediately scale to the full 4,000-prompt Recipe 1 run and run the 2,759 $\times 8$ Full Test. If Gate 2 loses $> 4\%$ savings, tighten the length-matching cap in Step 2 from $1.10\times$ to $1.00\times$ ($L_{\text{think}}(y^+) \le L_{\text{think}}(y^-)$).

---

## 4. What to Report in the Final Submission

To make the challenge submission authoritative, transparent, and scientifically compelling, structure the technical report around four empirical exhibits:

### 4.1 The Length vs. Accuracy Pareto Trajectory
Plot total token savings (%) on the $x$-axis against full-test $\Delta\text{Accuracy}$ (pts) on the $y$-axis, showing the target box ($x > 50\%, y \ge -1.0$) shaded in green:
* **The Single-Step Ceiling Curve**: Plot `p1cr1s0` (34.0% / +2.9 Dev), `p1r1s0-200` (43.4% / -1.5), and `bidax2` (51.2% / -3.0) to demonstrate visually that 1-step DPO from the original model defines a steep downward diagonal that misses the target box.
* **The Untyped Repair Leak Curve**: Plot `rc1-soup-30` (57.4% / -3.5) $\rightarrow$ `rpE-40` (50.6% / -1.9) $\rightarrow$ `rpE-60` (47.8% / -1.2) $\rightarrow$ `rpE-80` (47.0% / -0.9) $\rightarrow$ `rpI-120` (41.9% / +0.1) to show how naive right-beats-wrong repair climbs back up the *exact same diagonal*, trading away 10.4–15.5% token savings to recover accuracy.
* **The Clean 2-Step "Overshoot-Then-Calibrate" Trajectory**: Plot Original (0% / 0.0) $\rightarrow$ Recipe 1 Step 1 (~58.5% / -2.6) $\rightarrow$ Recipe 1 Step 2 Checkpoints 20, 40, 60 (moving **vertically upward** into the **53.5–55.5% saved / -0.5 to -0.9 acc** box), alongside `rpM-20 / 40 / 60` (50.1% / -0.94 $\rightarrow$ 54.1% / -0.93). Emphasize the orthogonality of Step 1 (horizontal shift in length) and Step 2 (vertical shift in accuracy).
* Include a breakdown table separated into **Thinking Tokens Saved (%)** vs. **Response Tokens Saved (%)** per task, proving that the model achieves $>60\%$ compression inside `<think>` while changing final user-facing response length by $<3\%$.

### 4.2 Out-of-Distribution Transfer & Difficulty-Elasticity Check
Report the zero-shot transfer suite ($\times 8$ rollouts, never trained on) comparing **Original Qwen3-0.6B**, the **Clean Recipe Final Model**, **`rpM-60`**, and public competitor weights (**`bidaREPLICA`**, **`basta`**, **`matousek 850`**):
* **Difficulty-Stratified Math Transfer (Skywork Easy Math $\rightarrow$ MATH-500 Levels 1 to 5 $\rightarrow$ AIME 2025 at 32k Cap)**:
  * Show average `<think>` tokens and accuracy at each difficulty tier. Following `arXiv:2602.20945`, highlight **elastic token scaling**: on Skywork and MATH Levels 1–2, the model compresses thinking by 60–70% with zero accuracy loss; on MATH Levels 4–5 and AIME 2025 (30 problems $\times 8$, 32k cap), the model dynamically scales its thinking budget up to thousands of tokens rather than prematurely truncating, outperforming uniform length-penalty baselines (`bidaREPLICA`, `basta`).
* **Broad Domain Transfer (HumanEval Code, ARC-C, CSQA, MMLU)**:
  * Report token savings and $\Delta\text{Acc}$ on HumanEval (comparing against `Herz1c`'s reported 65% / -1.2) and multiple-choice reasoning (ARC-C, CSQA, MMLU), confirming that removing self-doubt loops and preserving response completeness transfers cleanly to unseen domains and formats.

### 4.3 Ablation & Negative Results Table (What Failed and Why)
Explicitly document the five key negative results discovered during the project, as they provide high scientific value to the community:
1. **Why Mixing Shortening and Protection in One Step Fails (`p1r1s0-200`: 43.4% / -1.5; `p1cr1s0`: 34% / +2.9)**: Opposing gradients on `</think>` transition probability cancel each other out.
2. **Why Untyped Right-Beats-Wrong Repair Destroys Compression (`rpE-80`: 47.0% / -0.9; `rpI-120`: 41.9% / +0.1)**: Because hasty errors are shorter than verified solutions after an aggressive compression stage, uncontrolled repair pairs have $\text{len}(y^+) > \text{len}(y^-)$ and teach the model that longer thinking equals higher reward.
3. **Why Total-Sequence Shortening Breaks Instruction Following (`rpK` vs. `rpL`/`rpM` on IFEval)**: Penalizing total tokens shrinks final answers, violating word/sentence/paragraph minimums. Decoupling `<think>` length from response length solved this immediately.
4. **Why Over-Compressing Tool Calls Beyond ~48% Collapses Function Calling (`rpL-80`: -5.1 pts; `rpL-100/120`: -32 / -41 pts on `one function`)**: Below a critical thinking floor (~45–48% saving), the 0.6B model skips argument-to-schema verification and stops emitting tool call blocks, whereas on `none fits`, 61–66% compression *improves* accuracy by **+4.2 to +7.7 points** by preventing overthinking hallucinations (`arXiv:2502.03373`).
5. **Why Passive "Hold" Repair Fails (`rpN` on `rpL-60`)**: Removing shortening pairs without enforcing strict length-matched contrast on active failure modes neither recovers collapsed tool calls nor shifts the Pareto frontier.