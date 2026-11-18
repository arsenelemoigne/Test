# Consult 30 Efficiency - Working Notes

## Initial File Review (Turns 1-6)
- Read README_PACK.md, PLAN.md, MEASURES_WE_USE.md, NEW_RESULTS_online_17nov.md, density_strict.py, trim_condense.py, article_metrics.py, trim_condense_report_1600pairs.json.
- Currently reading ARTICLE_TEXT.md.
- Key datasets: svamp (math), mbpp (code), ifeval (instructions), bfcl_simple_python, bfcl_multiple, bfcl_irrelevance.
- Official score: 4 benchmarks (svamp, mbpp, ifeval, bfcl pooled 400/200/240 = 840 questions).
  - Saving = 1 - geom_mean_{b in 4}(T1_b / T0_b)
  - Accuracy change = arith_mean_{b in 4}(Acc1_b - Acc0_b)

## Turn 12 Update: Question 1 (Frame & Questions Lost Beyond Chance)
- Read all of ARTICLE_TEXT.md, density_strict.py, trim_condense.py, article_metrics.py, trim_condense_report_1600pairs.json, density_strict.json.
- Key insights on Question 1 (The Frame & "Questions lost beyond chance"):
  1. Algebraic decomposition of `density_strict.py`:
     - Per question $q$, let $k_0, k_1 \in \{0..8\}$ be right counts for base and trained, $m = k_0 + k_1$.
     - Observed loss $O_L(q) = \max(0, k_0 - k_1)/8$, observed gain $O_G(q) = \max(0, k_1 - k_0)/8$.
     - Notice $O_G(q) - O_L(q) = (k_1 - k_0)/8$ (net accuracy change) and $O_G(q) + O_L(q) = |k_1 - k_0|/8$ (absolute churn / L1 distance).
     - Since the hypergeometric null $c(m) = E_{H(16,m,8)}[\max(0, K_0 - (m-K_0))/8] = rac{1}{2} E_{H(16,m,8)}[|2K_0 - m|/8]$ is identical for loss and gain:
       - Excess loss $L(q) = O_L(q) - c(m) = rac{1}{2}\left(rac{|k_0 - k_1|}{8} - E_{	ext{null}}rac{|K_0 - K_1|}{8}ight) - rac{1}{2}rac{k_1 - k_0}{8}$.
       - Excess gain $G(q) = O_G(q) - c(m) = rac{1}{2}\left(rac{|k_0 - k_1|}{8} - E_{	ext{null}}rac{|K_0 - K_1|}{8}ight) + rac{1}{2}rac{k_1 - k_0}{8}$.
       - Therefore: $G(q) - L(q) = (k_1 - k_0)/8$ (exact net accuracy change!) and $G(q) + L(q) = rac{|k_0 - k_1|}{8} - 2c(m)$ (excess absolute churn / bidirectional dispersion!).
  2. Reviewer attack on $L(q)$ ("Questions lost beyond chance"):
     - Because $c(m)$ is subtracted symmetrically from both $O_L$ and $O_G$ when $k_0 
eq k_1$, any pure unidirectional shift distorts the other direction!
     - Specifically: if a question moves from $8/8 	o 7/8$, $O_L = 1/8 = 0.125$, $O_G = 0$, $c(15) = 0.0625$. Then $L = +0.0625$, and $G = -0.0625$!
     - Thus on SVAMP (math), where `od5c16` has $L = +24.81$ questions lost, it reports $G = -3.57$ "negative questions gained beyond chance"! Conversely, on `bfcl_irrelevance` where accuracy rises $+5.89$ pts (+14.12 net questions), it reports $L = -2.82$ ("negative 2.82 questions lost") and $G = +11.30$ (instead of 14.12)!
     - Why? Because under the null $p_0 = p_1$, $\hat{p} = (k_0+k_1)/16$ is pooled. When the alternative $p_0 
eq p_1$ holds (e.g., $p_0 = 1, p_1 = 0.875$), pooling $m=15$ treats the 1 failure as equally likely to have come from base or trained under the null, attributing half of the real change to "chance" and subtracting $c(m)$ from the zero side too!
     - Also, clipping $L(q)$ at 0 in `cost = t1 + np.clip(L, 0, None) * t0` introduces a positive bias even under the null (since $E[\max(0, O_L - c(m))] > 0$).
  3. Split-half control (base samples 0..3 vs 4..7, $K=4$):
     - Confirmed ~0: total $L = -3.33$ questions ($-0.12$ pts), all tasks within noise.
  4. Concentration by base solve rate $k_0$:
     - Cannot stratify `L(q)` directly by $k_0$ without accounting for regression to the mean! At $k_0 = 8$, $O_G$ is forced to $0$, so any question with $k_1 < 8$ has $m < 16$ and $c(m) > 0$, making $G = -c(m) < 0$ even under the null!

## Turn 27 Update: Question 2 (Trimming vs Condensing & Information per Token)
1. Audit of `code/trim_condense.py`:
   - Re-ran on the exact 1,600 pairs (200 questions x 8 samples for svamp+mbpp) and extended to all 6 tasks (5,518 pairs = 2,759 questions x 2 samples).
   - Identified 5 potential biases in `trim_condense.py` and tested each one by ablation:
     a. **Sentence splitting (`SENT_SPLIT`) bug on decimal numbers**: `(?<=[.!?])\s+` splits after periods followed by spaces, and `.\n` splits on single newlines, breaking code blocks and decimal/numbered lists. Replacing with paragraph/double-newline splitting or fixed 15-word windows changes matched step length ratio slightly.
     b. **Token weighting by character length (`len(step)/total_chars * thinking_tokens`)**: In math/code, digits, operators, and punctuation are 1-2 chars per token while words are 4-5 chars per token. Using a regex subword/token proxy changes condensing by <0.3%.
     c. **One-to-one greedy matching vs Hungarian (optimal bipartite) vs Many-to-One merging**:
        - Hungarian optimal matching gives virtually identical results to greedy matching (~1.7% vs ~1.4% condensing).
        - **CRITICAL BIAS**: One-to-one step matching *cannot detect cross-sentence fusion* (where 2-3 base sentences are condensed into 1 trained sentence). When we allow many-to-one step merging (matching additional base steps to an already-matched trained step if similarity >= threshold), the condensing share rises from ~1.5% to **12.5% - 14.9%** (and trimming drops from ~98.5% to ~85-87%)! Even under many-to-one merging, however, **trimming is still 85-88% of the token cut**.
     d. **Pairing sample $k$ with sample $k$ (arbitrary independent seed pairing)**: Since sample $k$ of base and sample $k$ of `od5c16` are independent stochastic rollouts, pairing $k$ with $k$ compares different solution paths on multi-path questions, inflating "novel" and "unmatched" steps. Pairing each `od5c16` sample with its **best-matching base sample** ($k^* \in \{0..7\}$) or restricting to **both-correct pairs** increases step match rate from 68% to 78% and raises measured 1-to-1 condensing from ~1.4% to **6.0% - 7.1%** (leaving **93% - 94% net trimming**).
     e. **Embedding matcher vs Model-Free matchers**: Tested 3 model-free matchers (Word 1-2 gram TF-IDF cosine, Character 3-gram Jaccard, Normalized Word Recall) across thresholds on both the 1,600 svamp+mbpp pairs and all 5,518 pairs across all 6 tasks.
        - Across all 6 tasks (5,518 pairs, Word 1-2g cosine >= 0.35): **97.3% of the 60.5% thinking cut is net trimming** (58.8% saved from net trimming vs 1.6% from condensing; gross trimming 72.1%, novel steps -13.2%).
        - Per task net trimming share of cut: `svamp` 94.3% (condensing 5.7%), `mbpp` 99.6% (condensing 0.4%), `ifeval` 99.1%, `bfcl_simple_python` 93.7%, `bfcl_multiple` 96.3%, `bfcl_irrelevance` 92.8%.
        - **Bottom line**: Even after fixing sentence splitting, subword token counting, best-of-8 sample pairing, and many-to-one step fusion, **85% to 97% of the token reduction is step removal (trimming), not wording compression (condensing)**. The qualitative conclusion holds robustly.

2. Model-free "Information per token" (CPU) & GPU Surprisal design:
   - **Trap to warn about**: Whole-trace `gzip` bits/token (+26.3%) and whole-trace bigram diversity (+16.5%) are **severely confounded by sequence length**! Longer texts have more cross-step phrase repetition, so Lempel-Ziv (`gzip`) compresses longer traces much more efficiently (and whole-trace TTR falls with length by Heaps' law).
   - **Best CPU model-free measures (length-controlled)**:
     1. **Matched-prefix gzip payload bits/char** (comparing `base[:L]` vs `od5c16[:L]` at $L = \min(L_{	ext{base}}, L_{	ext{od5c}})$, stripping the 18-byte gzip header): rises by only **+2.37%** overall (+4.2% on SVAMP, +5.4% on MBPP, ~0% on BFCL).
     2. **Fixed-window (80-word prefix) bigram diversity**: rises by **+2.76%** (+3.31% on matched prefixes).
     3. **Corpus unigram surprisal** ($-rac{1}{N}\sum \log_2 P_{	ext{corpus}}(w_i)$): rises by **+1.66%** (9.24 to 9.39 bits/token).
     4. **Per-step gzip bits/token**: changes by **-0.80%** (virtually zero).
   - Conclusion from CPU metrics: Once length bias is removed, information density per token changes by only **+1% to +3%**, confirming almost zero local lexical/syntactic compression.
   - **What the GPU surprisal measurement must be**:
     - Naive self-surprisal ($-\log p_{	heta}(t_i | t_{<i})$ under each model's own weights) is confounded both by **weight shift** (DPO changes entropy) and **position/length** (later tokens in a reasoning chain have lower conditional entropy because context is established).
     - The right GPU protocol:
       1. Evaluate **both** `base` and `od5c16` traces under a **single frozen reference evaluator** (specifically `base` Qwen3-0.6B, and optionally a larger judge like Qwen3-4B/8B) to measure cross-entropy $H(t_i | q, t_{<i}; 	heta_{	ext{base}})$.
       2. Compare at **matched token positions** $i \le \min(T_{	ext{base}}, T_{	ext{od5c}})$ (or stratify by token position $i \in [1..50], [51..150], \dots$) so position-dependent entropy decay does not masquerade as higher per-token information.
       3. Measure **answer-information gain per token**: the reduction in negative log-likelihood of the ground-truth answer $y^*$ per thinking token, $\Delta \log p_{	heta_{	ext{base}}}(y^* | q, t_{\le k}) / k$.

## Turn 36 Update: Question 3 (Overthinking / Thinking After the Answer is Settled)
1. Exact definitions & measurements across all non-truncated traces (and verified on correct-only traces):
   - **SVAMP (math, final number settled)**:
     - Using standalone number match (`after_first`) vs result-context match (`after_res`, preceded by `=`, `equals`, `is`, `total`, `answer`, `so`, etc., which filters out ~4.7% false early matches where the target number coincides with a problem-statement number):
     - Detection rate: 99.6% (base) and 98.9% (od5c16) for standalone; 99.0% (base) and 97.3% (od5c16) for result-context.
     - Share of thinking after the answer is settled (`after_res_all`):
       - **Base**: **47.3%** of thinking per trace (**53.3%** of thinking tokens, **328.5 / 616.2 tokens**; on first-match: 51.4% per trace, 57.4% of tokens = 353.9t).
       - **od5c16**: **30.4%** of thinking per trace (**35.9%** of thinking tokens, **93.9 / 261.7 tokens**; on first-match: 36.0% per trace, 42.5% of tokens = 111.2t).
     - Decomposition of the 354.5-token thinking cut on SVAMP:
       - **66.2% of the cut (234.7 tokens)** comes from cutting post-answer verification/re-checking (`328.5 -> 93.9` tokens, a **-71.4%** reduction)!
       - **33.8% of the cut (119.8 tokens)** comes from pre-answer derivation (`287.7 -> 167.8` tokens, a **-41.7%** reduction, mostly cutting problem-restatement preamble).
   - **BFCL Simple Python & Multiple (tool calls: function name + all argument values settled)**:
     - Detection rate: Function name is mentioned in **99.6% (base) / 97.9% (od5c16)** of `bfcl_simple_python` traces and **99.4% / 98.2%** of `bfcl_multiple` traces. All argument values are also mentioned in **93.6% / 81.4%** (`simple_python`) and **93.7% / 82.9%** (`multiple`).
     - Share of thinking after function name + argument values are settled (`after_settled_a_all`):
       - **`bfcl_simple_python`**: Base writes **58.7%** of thinking after the call is settled (**59.1%** of tokens, **100.3 / 169.7 tokens**; after function name alone: 69.4% / 119.5t). `od5c16` drops to **43.1%** (**42.8%** of tokens, **32.6 / 76.0 tokens**). **72.3% of the total thinking cut (67.7 / 93.7 tokens)** is post-settled overthinking!
       - **`bfcl_multiple`**: Base writes **63.2%** of thinking after the call is settled (**63.4%** of tokens, **120.4 / 189.9 tokens**; after function name alone: 72.6% / 140.0t). `od5c16` drops to **44.5%** (**44.7%** of tokens, **37.3 / 83.5 tokens**). **78.0% of the total thinking cut (83.0 / 106.4 tokens)** is post-settled overthinking!
     - **`bfcl_irrelevance`** (abstention phrase settled): Base writes **21.3%** (28.8% conditional on detection, 38.1 / 185.8t) after first stating no function applies, vs **16.4%** (21.8% conditional, 11.5 / 72.5t) for `od5c16`. Here **76.5% of the cut (86.8 / 113.4t)** is *pre-decision* (faster elimination of irrelevant candidate tools rather than enumerating every parameter of every tool).
   - **MBPP (code: `def func_name(` or final `return` statement drafted inside `<think>`)**:
     - Major behavioral shift discovered: **Base drafts the literal `def func_name(` inside `<think>` on 99.6% of traces**, and then spends **50.5% of its thinking (51.1% of tokens = 796.0 / 1559.1t; or 49.7% = 774.1t after the drafted return/code block)** dry-running examples and re-verifying the code inside `<think>` before outputting `</think>`!
     - **`od5c16`** drops literal `def` drafting inside `<think>` from **99.6% to 68.7%** (76.4% including return expressions), and when it does draft, cuts post-draft thinking from **49.6% to 33.2%**. Unconditionally across all traces, post-draft thinking falls from **49.7% of tokens (774.1t) to 31.4% (249.0t)** — accounting for **68.4% of the 767.2-token cut** (73.8% using `after_def_all`)!
   - **IFEval (instructions)**: Cannot be reliably defined by text matching because the final answer is an open-ended text (essay, poem, Kannada translation, formatting constraints) rather than a discrete symbol, number, or function signature.

2. Reliability per task & what the GPU early-stop probe adds:
   - **Reliability of text-trace detection**:
     - High on **SVAMP** (99% coverage; main caveat is ~4-5% coincidental early number matches with problem inputs, mitigated by result-context regex) and **BFCL tool calls** (98-99% function name, 82-94% all args; caveat: mentioning all parameter values in `<think>` does not prove the model has finished checking default parameters or type constraints).
     - Moderate/structural on **MBPP**: detects literal code drafting inside `<think>` (99.6% in base, 68.7% in `od5c16`), but when `od5c16` plans the algorithm in English without writing `def func(` inside `<think>`, text matching cannot pinpoint the exact sentence where the algorithm is settled.
     - Unusable on **IFEval** (and only keyword-approximate on **BFCL irrelevance**).
   - **What the GPU early-stop probe (cutting thinking at 10%, 20%, ..., 100% of tokens and appending `</think>\n` to force an answer) adds**:
     1. **Causal sufficiency vs surface mention**: Proves whether the model *would actually output the right final answer* if forced to stop at token $k$, distinguishing a lucky early mention of a number/argument from a committed internal state.
     2. **Task-universal coverage (including IFEval and non-drafting MBPP)**: Works identically on all 6 tasks without task-specific regexes.
     3. **Separates two hypotheses for why `od5c16` stops earlier**:
        - *Same accuracy-vs-token curve, earlier stopping threshold*: At matched token budgets $k$ (e.g., 50, 100, 150, 200 tokens), `base` truncated at $k$tokens already achieves the same accuracy as `od5c16`, meaning `base` was simply continuing on a flat plateau.
        - *Steeper accuracy-vs-token curve*: `od5c16` reaches the plateau at fewer tokens than `base` truncated at $k$ (e.g., because `od5c16` skips problem-restatement preambles in the first 30% of thinking).

## Turn 42 Update: Question 4 (Marginal Value of a Thinking Token & Constrained Optimization) and Question 5 (Priorities & Claims)
1. Exact Challenge Score Optimization & KKT Conditions:
   - Let $B = \{\text{svamp}, \text{mbpp}, \text{ifeval}, \text{bfcl}\}$ be the 4 equally weighted benchmarks, where $\text{bfcl}$ pools $t \in \{\text{simple}, \text{mult}, \text{irrel}\}$ with question weights $w_t = N_t / 840 = (10/21, 5/21, 6/21)$.
   - Output tokens $T_b = T_b^{\text{ans}} + \tau_b$ (where $\tau_b \ge 0$ is thinking tokens and $T_b^{\text{ans}}$ is the incompressible answer floor), output ratio $r_b = T_b / T_{b,0}$.
   - Saving score: $S(T) = 1 - \left(\prod_{b \in B} \frac{T_b}{T_{b,0}}\right)^{1/4} = 1 - \exp\left(\frac{1}{4} \sum_{b \in B} \ln \frac{T_b}{T_{b,0}}\right)$.
   - Constrained optimization:
     $$\min_{\tau_t \ge 0} \frac{1}{4} \sum_{b \in B} \ln\left(\frac{T_b(\tau)}{T_{b,0}}\right) \quad \text{s.t.} \quad \frac{1}{4} \sum_{b \in B} (A_{b,0} - A_b(\tau)) \le L$$
     (with $L = 1.00$ pt for the challenge target, or $L = 0.79$ pts for the same loss as `od5c16`).
   - **Optimality (KKT) Condition**:
     - For the 3 standalone benchmarks $b \in \{\text{svamp}, \text{mbpp}, \text{ifeval}\}$ at an interior solution ($\tau_b > 0$):
       $$\frac{\partial \ln T_b}{\partial \tau_b} = \lambda \frac{d A_b}{d \tau_b} \implies \kappa_b(T_b) \equiv T_b \frac{d A_b}{d T_b} = \frac{d A_b}{d \ln r_b} = \frac{1}{\lambda} = \text{const}.$$
       In terms of raw tokens, the marginal accuracy per token must be **inversely proportional to current output length $T_b$**:
       $$\frac{d A_b}{d T_b} = \frac{1}{\lambda T_b}.$$
       Because `od5c16` writes $T_{\text{mbpp}} = 873.0$ output tokens on MBPP vs $T_{\text{svamp}} = 269.7$ on SVAMP and $T_{\text{bfcl}} = 99.0$ on BFCL, 1 token saved on BFCL is worth **$8.82\times$** as much in the geometric mean as 1 token saved on MBPP, and 1 token saved on SVAMP is worth **$3.24\times$** as much as 1 token on MBPP!
     - Inside the pooled BFCL benchmark ($T_{\text{bfcl}} = \sum_t w_t T_t$, $A_{\text{bfcl}} = \sum_t w_t A_t(T_t)$), the weights $w_t$ **cancel out** in the KKT ratio $(\partial \ln T_{\text{bfcl}} / \partial \tau_t) / (\partial A_{\text{bfcl}} / \partial \tau_t)$! Thus all active BFCL sub-tasks must have equal **raw per-token slope**:
       $$\frac{d A_{\text{simple}}}{d T_{\text{simple}}} = \frac{d A_{\text{mult}}}{d T_{\text{mult}}} = \frac{1}{\lambda T_{\text{bfcl}}},$$
       while any sub-task with $\frac{d A_t}{d T_t} \le 0$ (specifically `bfcl_irrelevance`, where thinking *hurts* accuracy) is driven to the **boundary corner $\tau_{\text{irrel}}^* = 0$**!
     - Answer-token floor effect: Since $\frac{d A_b}{d \ln T_b} = \frac{T_b^{\text{ans}} + \tau_b}{\tau_b} \frac{d A_b}{d \ln \tau_b}$, when $\tau_b \to 0$ on a task with long answers like IFEval ($T_{\text{ifeval}}^{\text{ans}} \approx 230\text{--}262$ tokens, so $r_{\text{ifeval}}^{\min} = 0.377$), cutting thinking tokens yields diminishing percentage reductions in total output tokens $T_b$.

2. Empirical Marginal Values & How the Cut Should Be Redistributed:
   - **Secant marginal value of tokens cut (`base` $\to$ `od5c16`) and total thinking value (`base` $\to$ `nothink`)**:
     - `svamp`: Cut $416.4$ output tokens ($r=0.393$), $\Delta A = -2.84$ pts $\implies$ **$+0.681$ pts lost / 100 tokens** (secant $\kappa = +3.04$; local slope across 11 online/RP models: **$+2.34$ pts / 100t, $\kappa_{\text{loc}} = +6.30$**; `nothink` loses $-68.14$ pts!).
     - `bfcl_simple_python`: Cut $94.9$t ($r=0.513$), $\Delta A = -1.09$ pts $\implies$ **$+1.153$ pts lost / 100t** (secant $\kappa = +1.64$; local slope across 11 models: **$+14.32$ pts / 100t, $\kappa_{\text{loc}} = +14.32$** — steepest cliff near 100 tokens!).
     - `mbpp`: Cut $1252.4$t ($r=0.411$), $\Delta A = -1.95$ pts $\implies$ **$+0.156$ pts lost / 100t** (secant $\kappa = +2.19$; local slope: **$+0.35$ pts / 100t, $\kappa_{\text{loc}} = +3.05$**).
     - `bfcl_multiple`: Cut $108.9$t ($r=0.495$), $\Delta A = 0.00$ pts $\implies$ **$0.000$ pts / 100t** (local slope: **$+3.35$ pts / 100t, $\kappa_{\text{loc}} = +3.58$**; `nothink` loses $-20.75$ pts).
     - `ifeval`: Cut $324.8$t ($r=0.533$), $\Delta A = +0.49$ pts $\implies$ **$-0.149$ pts lost / 100t** (even `nothink` at $\tau=0$ loses only $-1.27$ pts vs `base`, or $1.76$ pts vs `od5c16` for $141$ thinking tokens = $+1.25$ pts / 100t, $\kappa = +5.06$).
     - `bfcl_irrelevance`: Cut $122.1$t ($r=0.426$), $\Delta A = +5.89$ pts $\implies$ **$-4.821$ pts lost / 100t** (local slope: **$-6.98$ pts / 100t, $\kappa_{\text{loc}} = -6.33$**; `rpL80` gains $+7.71$ pts at $72.6$t and `nothink` gains $+6.93$ pts at $7.9$t!).
   - **Redistribution & Estimated Gain**:
     - `od5c16` over-cuts `bfcl_simple_python` ($\kappa_{\text{loc}} = +14.3$) and `svamp` ($\kappa_{\text{loc}} = +6.3$) while under-cutting `mbpp` ($\kappa_{\text{loc}} = +3.05$), `ifeval` (where $\tau=0$ loses only $1.27$ pts vs base), and especially `bfcl_irrelevance` ($\kappa_{\text{loc}} = -6.33$, where cutting tokens *increases* accuracy!).
     - **Give back** ~10–23% tokens to `bfcl_simple_python` ($100 \to 107\text{--}123$t, recovering $+0.75\text{--}1.35$ pts) and ~3% to `svamp` ($270 \to 277$t, recovering $+1.34$ pts), and **cut deeper** on `bfcl_irrelevance` ($90.7 \to 72.6$t or $7.9$t), `ifeval` ($371.2 \to 265.7$t or $262.1$t), and `mbpp` ($873.0 \to 803.2$t or $722.7$t).
     - **Exact estimated gains**:
       1. *Within thinking models only (using achieved Pareto checkpoints or smooth log-linear reallocation within the observed online/RP range)*:
          - At the **same accuracy loss as `od5c16` ($-0.78$ pts)**: saving rises from **$54.84\%$ to $59.23\%\text{--}60.24\%$** (**$+4.4$ to $+5.4$ percentage points gain**).
          - At the **$-1.00$ pt accuracy budget**: saving rises to **$60.29\%\text{--}61.46\%$** (**$+5.5$ to $+6.6$ pts gain**, crossing the $60\%$ threshold!).
       2. *Allowing the corner solution $\tau=0$ (`nothink`) on `bfcl_irrelevance` and `ifeval`*:
          - Saving rises to **$62.87\%$ at $-0.78$ pts** (**$+8.0$ pts gain**) and **$63.74\%$ at $-0.99$ pts** (**$+8.9$ pts gain**).

## Final Check
All sections verified and saved.
Turn 86 check.
