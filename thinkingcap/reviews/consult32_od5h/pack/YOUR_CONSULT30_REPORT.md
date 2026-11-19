# Efficiency & Reasoning Audit: Did Online DPO Make Qwen3-0.6B Smarter or Teach It When to Stop?

## 1. The Frame: Evaluating the Four-Question Split and "Questions Lost Beyond Chance"

### Is the four-question split the right architecture?
The four-question structure in `PLAN.md`—(1) *where accuracy moves beyond sampling noise*, (2) *trimming vs. condensing*, (3) *overthinking after the answer is settled*, and (4) *the marginal value of a thinking token and optimal allocation*—is a strong narrative and analytical arc. It moves logically from **outcome** (what happened to reliability?) to **mechanism** (were steps deleted or compressed, and at what stage of reasoning?) to **economics** (how should token cuts be allocated across tasks?).

However, a careful reviewer will target three specific gaps and one redundancy:
1. **What is missing:**
   - **No-think baseline transfer (`nothink` on base vs. trained):** The strongest direct test of whether weight updates improved *zero-CoT representation/skill* versus *stopping policy* is evaluating `od5c16` with thinking turned off (`<think>\n</think>`). In `NEW_RESULTS_online_17nov.md`, `od5c16` with thinking off scores $-20.16$ points vs. base's $-22.28$ ($+2.12$ points better at $94.4\%$ token saving: math $20.6\% \to 24.3\%$, multiple-tool $55.3\% \to 67.1\%$). That proves DPO induced a small direct decision/formatting improvement on top of stopping earlier, while still confirming that $\sim 90\%$ of the gap to full thinking comes from having a scratchpad.
   - **Pass@$k$ vs. G-Pass@$k$ (coverage vs. consistency):** The article reports G-Pass@$8$ (all 8 draws right, measuring stability), which drops on SVAMP from $70.1\%$ to $61.7\%$. A reviewer will also ask for standard **Pass@$8$** (at least 1 of 8 draws right, measuring upper-bound capability/coverage). On our traces, Pass@$8$ on SVAMP drops by only $-0.50$ points ($97.2\% \to 96.7\%$) and on MBPP by $-1.37$ points ($69.2\% \to 67.8\%$), while G-Pass@$8$ drops $-8.40$ points on SVAMP. Thus, the model did not "forget how to solve" those math problems; cutting self-verification loops reduced per-sample reliability ($k_q/8$) on questions it still solves in most of 8 draws.
   - **Truncation recovery ($4{,}096$-token cap):** On MBPP, base hits the token cap on $22.30\%$ of traces ($3{,}582$ of $22{,}072$ total traces across all tasks), whereas `od5c16` truncates on only $6.15\%$ of MBPP traces ($211$ total). On truncated base traces, accuracy is near zero because `</think>` is never reached; avoiding cap-outs gives `od5c16` a structural accuracy rescue that partially masks losses on non-truncated problems.
2. **What is redundant:**
   - The article and plan currently track four overlapping reliability metrics: matched-draw flips (`article_metrics.py`), split-half count shifts (`NEW_RESULTS_online_17nov.md`), G-Pass@$k$, and hypergeometric `L(q)` (`density_strict.py`). Keep two: **net accuracy + Pass@8/G-Pass@8** for capability vs. consistency, and one clean **excess-churn test** for question-level shifts.

### Is "Questions Lost Beyond Chance" (`density_strict.py`) the right name and test for Question 1?
**No—a statistical reviewer will immediately attack both the formula and the name.**

In `density_strict.py`, for each question $q$ with base right-count $k_0 \in \{0..8\}$, trained right-count $k_1 \in \{0..8\}$, and pooled total $m = k_0 + k_1$, the code defines observed loss $O_L(q) = \max(0, k_0 - k_1)/8$, observed gain $O_G(q) = \max(0, k_1 - k_0)/8$, and subtracts the symmetric hypergeometric null expectation $c(m) = E[\max(0, 2K_0 - m)/8]$ from **both** $O_L(q)$ and $O_G(q)$:
- $L(q) = O_L(q) - c(m)$ and $G(q) = O_G(q) - c(m)$.

**Why this breaks under the alternative hypothesis ($p_0 \neq p_1$):**
Algebraically, $O_G(q) - O_L(q) = (k_1 - k_0)/8$ (exact net accuracy change) and $O_G(q) + O_L(q) = |k_0 - k_1|/8$ (L1 churn). Because $c(m) = \frac{1}{2}E_{\text{null}}[|K_0 - K_1|/8]$ is subtracted from *both* sides simultaneously, $L(q)$ and $G(q)$ are actually linear combinations of net drift and excess L1 dispersion:
- $G(q) - L(q) = (k_1 - k_0)/8$ (net accuracy change, independent of $c(m)$),
- $G(q) + L(q) = |k_0 - k_1|/8 - E_{\text{null}}[|K_0 - K_1|/8]$ (excess bidirectional churn).

Whenever a question has a real unidirectional shift, one of $O_L, O_G$ is zero, but $c(m) > 0$ is still subtracted from the zero side. For example, if a question drops from $k_0=8$ to $k_1=7$, $O_L = 0.125$, $O_G = 0$, and $c(15) = 0.0625$, yielding $L(q) = +0.0625$ and **$G(q) = -0.0625$ (a negative gain!)**. Aggregated over datasets (`density_strict.json`), this produces nonsensical numbers:
- On **SVAMP**, `od5c16` loses $-28.38$ net question-equivalents ($-2.84$ pts), which `density_strict.py` decomposes into **$L = +24.81$ questions lost** and **$G = -3.57$ "negative questions gained beyond chance."**
- On **`bfcl_irrelevance`**, where `od5c16` *gains* $+14.12$ net question-equivalents ($+5.89$ pts), `density_strict.py` reports **$L = -2.82$ ("negative 2.82 questions lost")** and $G = +11.30$!
- Furthermore, in `strict_density`, clipping `np.clip(L, 0, None)` before multiplying by $T_0(q)$ introduces a **positive upward bias under the null**: even on the split-half control (base draws $0..3$ vs. $4..7$, where unclipped $\sum L(q) = -3.33$ questions / $-0.12$ pts across $2{,}759$ questions, confirming the null expectation is $\approx 0$), clipping `L(q)` at zero inflates the penalty on every question where random noise happens to exceed $c(m)$.
- Finally, **stratifying $L(q)$ by base solve rate $k_0$** (e.g., claiming losses concentrate on $k_0 \in \{7,8\}$) is confounded by **regression to the mean**: at $k_0 = 8$, accuracy cannot go up ($O_G = 0$ identically), so any $k_1 < 8$ yields $G(q) = -c(m) < 0$ and $L(q) > 0$ even under pure binomial noise. To stratify by base difficulty without regression-to-the-mean bias, you must split base into 4 selection draws (to define difficulty bin) and 4 evaluation draws (compared against 4 `od5c16` draws); doing so shows that out-of-fold net loss on SVAMP is $-2.65$ pts on easy ($4/4$) questions vs. $-6.15$ pts on medium ($2/4\text{--}3/4$) questions.

**Recommended fix for Question 1:** Rename the metric **"Net Question Drift and Excess Churn"** (or **"Noise-Corrected Question Flips"**). Report two clean, non-paradoxical numbers per task:
1. **Net shift** $\Delta Q = \sum_q (k_1 - k_0)/8$ (exact net question-equivalents gained/lost), and
2. **Excess churn** $\sum_q (|k_0 - k_1|/8 - 2c(m))$ (or the split-half count of questions moving down/up minus the base $4+4$ split-half noise floor: $+106$ excess questions down and $+18$ excess questions up across all $2{,}759$ questions).

---

## 2. Trimming vs. Condensing: Method Audit, Model-Free Re-Check, and Information Density

### Audit of `code/trim_condense.py`
`trim_condense.py` attributes $98.0\%$ of the net thinking-token reduction on `od5c16` ($386.6$ of $394.6$ tokens cut per trace across $22{,}072$ pairs) to **net trimming** (unmatched base sentences minus unmatched trained sentences) and only $2.0\%$ ($8.0$ tokens) to **condensing** (matched base sentences being longer than their matched trained counterparts). Reviewing the five pipeline design choices reveals two substantial structural biases and three minor distortions:

1. **Sentence splitting (`(?<=[.!?])\s+|\n+`) — High distortion on code and equations:** Splitting on whitespace after periods breaks decimals, method calls (`math.sqrt(...)`), and numbered lists (`1. `), while `\n+` turns every line of an MBPP Python draft inside `<think>` into a separate "sentence."
2. **Embedding matcher (`all-MiniLM-L6-v2`) at cosine threshold $0.60$ — Over-matches boilerplate:** `MiniLM` has a $256$-word-piece limit and assigns cosine similarity $>0.60$ to generic reasoning connectives (*"Let's check the constraints," "Wait, let's recalculate"*) even when their mathematical content differs, while missing multi-step algebraic condensations.
3. **Sharing tokens proportionally by character length (`len(s) / total_chars`) — Minor distortion ($<3\%$):** Code, digits, and punctuation have $\sim 2.0\text{--}2.5$ characters/token vs. $\sim 4.2$ for English prose, slightly under-weighting short equations relative to verbose prose transitions.
4. **One-to-one greedy matching — Structural bias against multi-sentence compression:** If `od5c16` merges $3$ base sentences ($60$ tokens total) into $1$ combined sentence ($25$ tokens), greedy 1-to-1 matching pairs the trained sentence with the single closest base sentence (say, $22$ tokens $\to -3$ tokens "condensing") and classifies the other two base sentences ($38$ tokens) as **trimmed**. Thus 1-to-1 matching structurally forces multi-sentence synthesis to register as trimming.
5. **Pairing sample $k$ with sample $k$ — Trajectory mismatch noise:** Draws $k=0..7$ in `base` and `od5c16` are independent stochastic samples ($T=0.6$), not aligned counterfactual trajectories. If draw $k$ in base explores approach A while draw $k$ in `od5c16` uses approach B (even though draw $j$ in base used approach B), sample-$k$-to-$k$ matching fails to align them and misclassifies the divergence as gross trimming plus unmatched additions.

### Model-Free Re-Check on the Traces (All $22{,}072$ Pairs)
To test whether the "~98% trimming" conclusion is an artifact of `MiniLM`, greedy 1-to-1 matching, or $k$-to-$k$ pairing, we re-evaluated all $22{,}072$ trace pairs using **four model-free matchers** across all 6 tasks (allocating tokens proportionally to word+punctuation token counts):
- **M1 (Normalized Word Jaccard $\ge 0.35$, 1-to-1 greedy, sample $k$-to-$k$):** Direct model-free analogue of `trim_condense.py`.
- **M2 (Character 4-gram Dice $\ge 0.40$, 1-to-1 greedy, sample $k$-to-$k$):** Captures sub-word morphology, code identifiers, and numbers.
- **M3 (Many-to-One Subsumption, Recall $\ge 0.50$):** Eliminates the 1-to-1 greedy bias by allowing every base sentence whose vocabulary is $\ge 50\%$ covered by *any* trained sentence to map into that trained sentence, directly measuring whether multiple base steps were folded into fewer trained sentences.
- **M4 (Best-of-8 Trajectory Pairing):** Pairs each `od5c16` draw $k$ with whichever of the 8 `base` draws $j \in \{0..7\}$ shares the highest Jaccard similarity, eliminating random sample-$k$ mismatch.

| Task ($N$ pairs) | `trim_condense` Net Trim % | M1 (Word Jaccard 1-to-1) Net Trim % (Gross Trim / Cond / Add %) | M2 (4-gram Dice 1-to-1) Net Trim % | M3 (Many-to-1 Subsumption) Net Trim % (Cond %) | M4 (Best-of-8 Pairing) Net Trim % (Cond %) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **SVAMP** ($8{,}000$) | $98.1\%$ | **$98.8\%$** ($110.2 / 1.2 / -11.4\%$) | $98.8\%$ | $90.4\%$ ($9.6\%$) | $97.6\%$ ($2.4\%$) |
| **MBPP** ($2{,}056$) | $98.8\%$ | **$99.1\%$** ($110.0 / 0.9 / -10.9\%$) | $98.9\%$ | $93.4\%$ ($6.6\%$) | $98.0\%$ ($2.0\%$) |
| **IFEval** ($3{,}296$) | $97.5\%$ | **$99.2\%$** ($116.5 / 0.8 / -17.3\%$) | $98.9\%$ | $93.1\%$ ($6.9\%$) | $97.8\%$ ($2.2\%$) |
| **BFCL Simple** ($3{,}200$) | $93.0\%$ | **$96.8\%$** ($126.9 / 3.2 / -30.0\%$) | $96.6\%$ | $84.7\%$ ($15.3\%$) | $94.9\%$ ($5.1\%$) |
| **BFCL Multiple** ($1{,}600$) | $94.5\%$ | **$96.8\%$** ($123.7 / 3.2 / -26.8\%$) | $96.8\%$ | $84.4\%$ ($15.6\%$) | $94.9\%$ ($5.1\%$) |
| **BFCL Irrel.** ($1{,}920$) | $97.2\%$ | **$98.0\%$** ($115.6 / 2.0 / -17.6\%$) | $97.8\%$ | $87.9\%$ ($12.1\%$) | $96.3\%$ ($3.7\%$) |
| **All Tasks** ($22{,}072$) | **$98.0\%$** | **$98.8\%$** ($112.4 / 1.2 / -13.6\%$) | **$98.6\%$** | **$91.4\%$** ($8.6\%$) | **$97.5\%$** ($2.5\%$) |

**Does the conclusion hold?**
**Yes—qualitatively and decisively.** Even when we fix the $k$-to-$k$ pairing bias (M4: $97.5\%$ net trimming) and allow unrestricted many-to-one sentence merging (M3: $91.4\%$ net trimming, $8.6\%$ upper-bound multi-sentence condensing), **over $91\%$ of the token reduction comes from dropping sentences altogether**—primarily self-doubt loops (*"Wait, let me double check..."* drops $70\text{--}86\%$), redundant verification passes, and full code drafts inside `<think>`. Matched sentences barely change length ($16.5 \to 15.0$ tokens in M1). Note, however, that reporting only *net* trimming ($98\%$) masks $\sim 13.6\%$ gross unmatched new sentences in `od5c16` ($112.4\%$ gross base trimmed minus $13.6\%$ unmatched trained sentences); the article should report both gross and net figures.

### Measuring "Information per Token" Without a GPU vs. With a GPU
- **Best CPU (GPU-free) measurement — Prefix-Controlled & Prompt-Conditioned Compression Rate:**
  Raw `zlib` bytes/token (`info_density.py`, where `od5c16` is $+20.7\%$ higher overall: $1.800 \to 2.173$ bytes/token) is heavily confounded by `zlib`'s sliding-window dictionary advantage on longer texts. We separated length artifacts from true information density across all $22{,}072$ traces using two controls:
  1. **Prefix-matched compression (`zlib` bytes/token on the first $K=100$ tokens of both models on traces $\ge 100$ tokens):** The density difference drops from $+20.7\%$ to **$+2.3\%$ overall** ($+1.2\%$ on SVAMP, $+5.5\%$ on MBPP, $+0.4\%$ on IFEval, $+1.2\text{--}1.8\%$ on BFCL). Thus, at matched prefix length, `od5c16`'s phrasing is only $\sim 1\text{--}5\%$ denser; `base`'s $20.7\%$ full-trace penalty comes from **repeating itself in the tail after token 100**.
  2. **Prompt-conditioned new-information rate** `(zlib(prompt + think) - zlib(prompt)) / think_tokens` and **distinct 3-gram ratio**: Prompt-conditioned novel bytes/token increases by **$+33.0\%$ overall** ($1.384 \to 1.841$ novel bytes/token; $+25.8\%$ on SVAMP, $+74.2\%$ on MBPP, $+37.7\%$ on IFEval), and unique 3-gram density rises from $0.752 \to 0.849$ ($+12.8\%$), because `od5c16` eliminates prompt restatements and repetitive tail verification loops.
- **What the GPU surprisal measurement should be:**
  1. **Cross-model vs. self-surprisal (bits/token):** Run teacher-forced forward passes of **both** `base` and `od5c16` on both sets of traces to compute $H_{\text{base}}(y_{\text{think}} \mid x)$ and $H_{\text{od5c16}}(y_{\text{think}} \mid x)$. Reporting only self-surprisal is circular (policy sharpening lowers self-entropy even on identical text); evaluating under `base` as a fixed prior tests whether `od5c16`'s retained tokens carry higher semantic surprise or simply omit low-entropy filler transitions (*"Wait, let me recalculate..."*).
  2. **Position-aligned surprisal curve $H(t)$ vs. token index $t$:** Plot token surprisal from $t=1..T$ to verify whether `base` exhibits decaying surprisal in its tail ($t > t^*$, low-information verification loops) that `od5c16` truncates, versus higher per-token surprisal in tokens $1..100$.
  3. **Predictive Mutual Information with the Final Answer ($I(Y_{1:k}; A^* \mid X)$):** Measure the log-likelihood gain of the ground-truth answer $A^*$ per thinking token consumed, $\Delta \log P(A^* \mid X, Y_{1:k}) / k$, which quantifies *task-relevant* information per token rather than surface lexical unpredictability.

---

## 3. Overthinking: Post-Settlement Thinking Share and What the GPU Early-Stop Probe Adds

### Exact Post-Settlement Thinking Share on the Traces
We parsed all $22{,}072$ traces per model to locate the exact token position inside `<think>` where the final answer $A$ is first settled, computing both `post_share_of_detected` (among traces where $A$ appears inside `<think>`) and `post_share_of_all` (setting post-settlement tokens to $0$ when $A$ is emitted only after `</think>`):

1. **SVAMP ($8{,}000$ traces/model — Final number inside `<think>`):**
   - *Unconstrained number match:* Detected in **$95.7\%$** of `base` traces vs. **$94.8\%$** of `od5c16` traces (`base` solves $0.9\%$ in the answer block after `</think>`, `od5c16` solves $5.1\%$ after `</think>`). First appearance occurs at **$28.0\%$** of thinking tokens in `base` ($187$ of $667$ tokens) vs. **$49.7\%$** in `od5c16` ($125$ of $252$ tokens). Thus, **post-settlement thinking drops from $72.0\%$ (`base`, $480$ tokens) to $50.3\%$ (`od5c16`, $127$ tokens)** among detected traces (**$70.7\% \to 48.0\%$** across all traces). Of the $416.6$ thinking tokens cut on SVAMP, **$352.9$ tokens ($84.7\%$ of the entire cut)** come from post-settlement verification, versus $63.7$ tokens ($15.3\%$) from pre-settlement derivation!
   - *Answer-context match (`= N`, `is N`, etc., removing $\sim 4.5\%$ coincidental early input-number hits):* First settlement occurs at $198$ tokens in `base` vs. $132$ tokens in `od5c16`; post-settlement share among detected traces drops from **$69.7\%$ (`base`) to $46.8\%$ (`od5c16`)**.
2. **BFCL Tool Calls (`simple_python` $3{,}200$ traces, `multiple` $1{,}600$ traces — Function name + all argument values):**
   - *Function name settled:* Detected in $98.9\%$ (`base`) vs. $97.6\%$ (`od5c16`) on `simple_python` (and $98.6\%$ vs. $97.6\%$ on `multiple`). Post-settlement thinking after naming the function drops from **$77.9\% \to 63.3\%$** on `simple_python` ($134 \to 50$ tokens) and **$80.0\% \to 64.9\%$** on `multiple` ($157 \to 55$ tokens).
   - *Full call settled (function name + all non-trivial argument values mentioned):* Detected in $85.6\%$ (`base`) vs. $81.6\%$ (`od5c16`) on `simple_python`, and $94.1\%$ vs. $89.9\%$ on `multiple`. Post-settlement thinking after all arguments are identified drops from **$52.2\% \to 36.4\%$** on `simple_python` ($89 \to 29$ tokens among detected; **$45.4\% \to 30.1\%$** of all thinking tokens) and from **$54.3\% \to 38.7\%$** on `multiple` ($106 \to 33$ tokens among detected; **$51.4\% \to 35.1\%$** of all thinking tokens).
3. **MBPP Code ($2{,}056$ traces/model — `def <func_name>` and draft completion inside `<think>`):**
   - Among completed (non-truncated) traces, `base` writes a literal `def <func_name>(` code block inside `<think>` on **$99.6\%$** of traces, whereas `od5c16` does so on only **$68.7\%$** of traces—on the remaining **$31.3\%$**, `od5c16` plans in English and emits the Python code only *after* `</think>`!
   - When `def <func_name>(` appears inside `<think>`, it starts at $34.1\%$ of thinking tokens in `base` vs. $51.4\%$ in `od5c16` (**$65.9\%$ vs. $48.6\%$** of thinking spent after starting the first code draft; **$64.9\% \to 34.4\%$** across all completed traces). After the *end* of the first full code block inside `<think>`, `base` spends **$36.4\%$** of its thinking ($468$ tokens) re-drafting and double-checking, vs. **$15.1\%$** ($91$ tokens) in `od5c16`.

### Reliability of Text-Based Settlement Detection and Value of the GPU Early-Stop Probe
- **Reliability per task:**
  - *High on SVAMP:* Numeric answers are discrete and easy to match ($95\%$ coverage); only $\sim 4.5\%$ of problems suffer coincidental early matches with prompt numbers, which answer-context matching resolves.
  - *Moderate-to-high on BFCL (`simple_python`, `multiple`):* Matching function name + all argument values ($82\text{--}94\%$ coverage) bounds when the ingredients are present, though mentioning an argument value in `<think>` does not prove the model has finished checking default parameters or type formatting.
  - *Structural only on MBPP:* Text matching detects literal code drafting inside `<think>` ($99.6\% \to 68.7\%$), proving `od5c16` learned to stop drafting code twice (once in `<think>`, once in the answer), but cannot pinpoint when an English-only algorithm plan ($31.3\%$ of `od5c16` traces) is conceptually complete.
  - *Unusable on IFEval (and approximate on BFCL Irrelevance):* Long-form constrained essays are not written verbatim inside `<think>` before `</think>`.
- **What the GPU Early-Stop Probe (truncating `<think>` at $10\%, 20\%, \dots, 100\%$ and appending `</think>\n` to force an answer) adds:**
  1. **Causal commit point vs. surface mention:** Proves whether the model *actually outputs the correct final answer* when forced to stop at token $k$, distinguishing a lucky early mention of a number/argument from a committed internal state.
  2. **Universal coverage across all 6 tasks:** Works identically on IFEval, BFCL Irrelevance, and non-drafting MBPP traces where regex detection fails.
  3. **Decisive test between the two core hypotheses:** Comparing `base` truncated at $k$ tokens against `od5c16` at $k$ tokens separates: (a) **Pure stopping-threshold shift** (`base` truncated at $150\text{--}250$ tokens already matches `od5c16`'s accuracy, proving `base` was merely spinning on a flat accuracy plateau), vs. (b) **Faster prefix convergence** (`od5c16` reaches the accuracy plateau at fewer tokens than `base` truncated at $k$, because `od5c16` also skips pre-settlement prompt restatements).

---

## 4. The Marginal Value of a Thinking Token and Optimal Cross-Task Allocation

### Constrained Optimization and Exact KKT Optimality Conditions for the Challenge Score
Let $B = \{\text{svamp}, \text{mbpp}, \text{ifeval}, \text{bfcl}\}$ be the four equally weighted benchmarks, where $\text{bfcl}$ pools $t \in \{\text{simple}, \text{mult}, \text{irrel}\}$ with fixed question weights $w_t = N_t / 840 = (10/21, 5/21, 6/21)$. For each task, total output tokens are $T_t(\tau_t) = T_t^{\text{ans}} + \tau_t$ (with incompressible answer floor $T_t^{\text{ans}}$ and thinking budget $\tau_t \ge 0$), output token ratio is $r_b = T_b / T_{b,0}$, and accuracy is $A_b(\tau)$.

Maximizing the geometric-mean saving score $S(\tau) = 1 - \left(\prod_{b \in B} r_b(\tau)\right)^{1/4}$ subject to arithmetic-mean accuracy loss $\frac{1}{4} \sum_{b \in B} (A_{b,0} - A_b(\tau)) \le L$ is equivalent to minimizing $\frac{1}{4} \sum_{b \in B} \ln(T_b(\tau)/T_{b,0})$. Forming the Lagrangian with multiplier $\lambda > 0$ yields the exact **Karush-Kuhn-Tucker (KKT) optimality conditions**:

1. **Equal Logarithmic Slope Across the Four Benchmarks:**
   For every interior benchmark ($\tau_b > 0$),
   $$\frac{\partial \ln T_b}{\partial \tau_b} = \lambda \frac{d A_b}{d \tau_b} \implies \kappa_b(T_b) \equiv T_b \frac{d A_b}{d T_b} = \frac{d A_b}{d \ln r_b} = \frac{1}{\lambda} = \text{constant across } b \in B.$$
   Equivalently, in terms of **raw marginal accuracy per token**,
   $$\frac{d A_b}{d T_b} = \frac{1}{\lambda T_b}.$$
   Because the saving score uses the **geometric mean of output ratios**, a $1\%$ relative cut in total output tokens $T_b$ buys the exact same saving score on every benchmark. At `od5c16`'s operating point ($T_{\text{mbpp}} = 873.0$, $T_{\text{ifeval}} = 371.2$, $T_{\text{svamp}} = 269.7$, $T_{\text{bfcl}} = 99.0$ tokens), **saving $1$ token on BFCL is worth $8.82\times$ as much in the geometric mean as saving $1$ token on MBPP**, and **saving $1$ token on SVAMP is worth $3.24\times$ as much as $1$ token on MBPP**. Conversely, an optimal allocation requires the raw per-token accuracy slope $dA_b/dT_b$ on BFCL to be $8.82\times$ steeper than on MBPP!
2. **Equal Raw Per-Token Slope Inside Pooled BFCL (and Corner Solution for Negative-Slope Sub-Tasks):**
   Inside BFCL ($T_{\text{bfcl}} = \sum_t w_t T_t$ and $A_{\text{bfcl}} = \sum_t w_t A_t(T_t)$), the sub-task weights $w_t$ **cancel out** in the KKT derivative ratio $\frac{\partial \ln T_{\text{bfcl}} / \partial \tau_t}{\partial A_{\text{bfcl}} / \partial \tau_t} = \frac{1}{T_{\text{bfcl}} (dA_t / dT_t)}$. Thus all active BFCL sub-tasks must satisfy
   $$\frac{d A_{\text{simple}}}{d T_{\text{simple}}} = \frac{d A_{\text{mult}}}{d T_{\text{mult}}} = \frac{1}{\lambda T_{\text{bfcl}}},$$
   while any sub-task with $\frac{dA_t}{dT_t} \le 0$ (where thinking hurts accuracy, as on `bfcl_irrelevance`) violates interior positivity and is driven to the **boundary corner $\tau_{\text{irrel}}^* = 0$**!
3. **Answer-Token Floor Penalty:**
   Writing thinking elasticity as $\frac{dA_b}{d\ln \tau_b}$, the KKT condition is $\frac{T_b^{\text{ans}} + \tau_b}{\tau_b} \frac{dA_b}{d\ln \tau_b} = \frac{1}{\lambda}$. When $\tau_b$ approaches $0$ on a long-answer benchmark like IFEval ($T_{\text{ifeval}}^{\text{ans}} \approx 230\text{--}262$ tokens, floor ratio $r_{\text{ifeval}}^{\min} = 0.377$), cutting thinking tokens yields diminishing percentage reductions in total output $T_b$.

### Empirical Marginal Value of Cut Tokens and Optimal Redistribution
From the per-answer records and the 18 model checkpoints in `metrics.json` (fitting local slopes across the 11 online/RP checkpoints around `od5c16`), each task's marginal value of thinking tokens is:

| Task (Weight in Score) | `base` $\to$ `od5c16` Output Tokens ($r_t$) | $\Delta A$ (`od5c16` vs `base`) | Secant Slope $\Delta A / 100\Delta T$ | Secant Log-Slope $\kappa = \Delta A / \Delta \ln r$ | Local Slope Near `od5c16` ($\text{pts}/100\text{t}$) | Local KKT Log-Slope $\kappa_{\text{loc}} = T_1 (dA/dT)$ | `nothink` ($\tau=0$) $\Delta A$ vs `base` ($r_{\min}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SVAMP** ($0.250$) | $686.1 \to 269.7$ ($0.393$) | $-2.84$ pts | **$+0.681$** | $+3.04$ | $+2.34$ | **$+6.30$** | $-68.14$ pts ($0.028$) |
| **MBPP** ($0.250$) | $2125.4 \to 873.0$ ($0.411$) | $-1.95$ pts | **$+0.156$** | $+2.19$ | $+0.35$ | **$+3.05$** | $-18.34$ pts ($0.068$) |
| **IFEval** ($0.250$) | $696.0 \to 371.2$ ($0.533$) | $+0.49$ pts | **$-0.149$** | $-0.77$ | $-0.05$ ($+1.25$ to $\tau=0$) | **$-0.19$** ($+3.55$ to $\tau=0$) | $-1.27$ pts ($0.377$) |
| **BFCL Simple** ($0.119$) | $194.9 \to 100.0$ ($0.513$) | $-1.09$ pts | **$+1.153$** | $+1.64$ | $+14.32$ | **$+14.32$** | $-4.38$ pts ($0.123$) |
| **BFCL Multiple** ($0.060$) | $215.9 \to 106.9$ ($0.495$) | $0.00$ pts | **$0.000$** | $0.00$ | $+3.35$ | **$+3.58$** | $-20.75$ pts ($0.127$) |
| **BFCL Irrel.** ($0.071$) | $212.8 \to 90.7$ ($0.426$) | $+5.89$ pts | **$-4.821$** | $-6.91$ | $-6.98$ | **$-6.33$** | $+6.93$ pts ($0.037$) |

**Why `od5c16`'s uniform $\sim 50\text{--}60\%$ cut is sub-optimal and how to redistribute it:**
Online DPO with a task-agnostic length margin cut every task to $r_t \in [0.39, 0.53]$. At that point, the KKT log-slopes $\kappa_{\text{loc}}$ are severely imbalanced:
1. **`bfcl_simple_python` ($\kappa_{\text{loc}} = +14.32$) and `svamp` ($\kappa_{\text{loc}} = +6.30$) are over-cut:** Near $100$ tokens on `bfcl_simple_python` and $270$ tokens on `svamp`, accuracy is on a steep cliff. Giving back just $+7$ to $+23$ tokens on `bfcl_simple_python` (e.g., to `od5c08`'s $123.5$ tokens) and $+8$ tokens on `svamp` (to `od5c12`'s $277.3$ tokens) recovers $+1.35$ pts on `bfcl_simple_python` and $+1.34$ pts on `svamp` at minimal saving cost.
2. **`bfcl_irrelevance` ($\kappa_{\text{loc}} = -6.33$) and `ifeval` ($\kappa_{\text{loc}} \approx 0$) are under-cut:** On `bfcl_irrelevance`, thinking actively induces false-positive tool calls; cutting further from $90.7 \to 72.6$ tokens (`rpL80`) or $7.9$ tokens (`nothink`, $\tau=0$) **gains** $+1.82$ to $+1.04$ accuracy points while slashing BFCL's mean tokens! On `ifeval`, cutting thinking from $141$ tokens (`od5c16`) down to $33$ tokens (`rpL80`) or $0$ tokens (`nothink`) loses only $0.67$ to $1.76$ pts while shrinking $r_{\text{ifeval}}$ from $0.533 \to 0.382\text{--}0.377$.
3. **`mbpp` ($\kappa_{\text{loc}} = +3.05$) can be cut deeper:** Because $T_{\text{mbpp}} = 873$ tokens is large, trimming MBPP from $873 \to 723\text{--}803$ tokens (`rpL80`/`od2c08`) trades a small accuracy dip for a substantial log-ratio reduction.

**Estimated Gain from Optimal Redistribution:**
- **Within achieved thinking checkpoints / smooth KKT reallocation (no $\tau=0$ routing):**
  - At the **exact same accuracy loss as `od5c16` ($-0.78$ pts)**: redistributing cuts across tasks (giving back tokens to `svamp` and `bfcl_simple_python` while cutting `ifeval`, `mbpp`, and `bfcl_irrelevance` to the `rpL80` regime) increases saving from **$54.84\%$ to $59.23\%\text{--}60.24\%$** (**$+4.4$ to $+5.4$ percentage points more saving at identical accuracy**).
  - At the **$-1.00$ point challenge threshold**: optimal redistribution reaches **$60.29\%\text{--}61.46\%$ saving** (**$+5.5$ to $+6.6$ percentage points above `od5c16`**).
- **Allowing the KKT boundary corner $\tau=0$ (`nothink`) on `bfcl_irrelevance` (and/or `ifeval`):**
  - Routing `bfcl_irrelevance` to $\tau=0$ ($7.9$ output tokens, $+6.93$ pts vs. base) and reallocating the saved accuracy budget to `svamp`/`mbpp`/`ifeval` raises saving to **$62.87\%$ at $-0.78$ pts loss** (**$+8.0$ pts gain**) and **$63.74\%$ at $-0.99$ pts loss** (**$+8.9$ pts gain**).
- **What the GPU probe must measure for exact single-model $A_t(\tau_t)$ curves:**
  Cross-checkpoint slopes conflate token budget $\tau_t$ with training round. To measure pure inference-time $A_t(\tau_t)$ curves and plug exact derivatives $dA_t/d\tau_t$ into the KKT solver, run `base` and `od5c16` at **10 fixed absolute thinking-token budgets** per task (e.g., $\tau \in \{0, 25, 50, 75, 100, 150, 200, 300, 500, \infty\}$ for SVAMP/BFCL/IFEval; $\{0, 50, 100, 200, 400, 600, 800, 1200, 2000, \infty\}$ for MBPP) by forcing `</think>\n` when the thinking token count hits $\tau_t$ and recording both accuracy $A_t(\tau_t)$ and total output length $T_t(\tau_t)$.

---

## 5. Priorities and Claims

### Priority-Ordered GPU Analyses (With Sample Sizes)
1. **Priority 1 — Forced Early-Stop Probe & Absolute Budget Curves $A_t(\tau)$ (`base` vs. `od5c16`):**
   - *Why first:* Simultaneously answers Question 3 (causal commit point across all 6 tasks, including IFEval and MBPP) and Question 4 (exact $A_t(\tau_t)$ derivatives for KKT optimal token allocation), while testing whether `base` truncated at $k$ tokens matches `od5c16` at $k$ tokens.
   - *Sample size:* Stratified subset of **$600$ questions $\times 4$ draws = $2{,}400$ traces per model** ($100$ questions per task $\times 6$ tasks, stratified by base solve rate $k_0$), evaluated across **$10$ budget cutoffs** ($0\%, 10\%, \dots, 100\%$ of trace length, plus $5$ fixed absolute token budgets). Only the short post-`</think>` answer ($30\text{--}250$ tokens) is generated at each cutoff ($\sim 24{,}000$ short answer continuations/model, $\sim 35\text{--}45$ minutes on one A100/H100 with vLLM prefix caching).
2. **Priority 2 — Cross-Model Teacher-Forced Surprisal & Position-Aligned Entropy Curve:**
   - *Why second:* Cheap single-forward-pass test (no generation) that resolves Question 2's information-density question by measuring `base` and `od5c16` log-probabilities on the exact same traces as a function of token index $t$.
   - *Sample size:* **$1{,}200$ questions $\times 2$ draws = $2{,}400$ trace pairs** ($200$ questions/task), running teacher-forced forward passes under both `base` and `od5c16` ($9{,}600$ forward passes total, $\sim 10$ minutes on one GPU).
3. **Priority 3 — `od5c16` No-Think & Task-Aware Budget Routing Verification on Full Test Set:**
   - *Why third:* Directly verifies the KKT redistribution gain ($+4.4$ to $+8.9$ percentage points additional saving) by running `od5c16` with task-specific `max_thinking_tokens` caps (and $\tau=0$ on `bfcl_irrelevance` / `ifeval`) across all $2{,}759 \times 8 = 22{,}072$ test instances ($\sim 25$ minutes on one GPU).

### Two Short Paragraphs for the Article (What to Claim and What NOT to Claim)

**What to claim (adaptable draft):**
> Online DPO did not make Qwen3-0.6B a denser step-by-step composer; it taught the model **when its answer is already settled and when to stop verifying it**. Across all $22{,}072$ test traces, model-free text alignment confirms that over $91\text{--}98\%$ of the $54.8\%$ token reduction comes from deleting whole sentences rather than compressing phrasing—and when we compare the first $100$ thinking tokens head-to-head, `od5c16`'s local compression density is within $2.3\%$ of the original model. What disappears is the repetitive tail: on math, the original model reaches its final number $28\%$ of the way through `<think>` and spends the remaining $72\%$ ($480$ tokens) second-guessing itself, whereas `od5c16` cuts $85\%$ of its tokens from that post-settlement verification loop (`Wait, let me double check...` drops by $83\%$). On code, the original model drafts the full Python function inside `<think>` $99.6\%$ of the time before copying it into the final answer, while `od5c16` skips in-think code drafting nearly a third of the time and cuts post-draft re-checking from $468$ tokens to $91$.

**What NOT to claim—and the allocation takeaway (adaptable draft):**
> We should not claim that every cut token was useless filler, nor that a uniform $55\%$ cut across benchmarks is optimal. Cutting verification loops leaves single-attempt solution coverage (Pass@$8$) virtually intact on math ($97.2\% \to 96.7\%$), but reduces per-draw consistency (8-of-8 stability drops $8.4$ points, costing $-2.84$ points in mean math accuracy), while on irrelevant-tool detection (`bfcl_irrelevance`), cutting $57\%$ of thinking tokens actually *improves* accuracy by $+5.89$ points (and turning thinking off completely gains $+6.93$ points) by preventing the model from talking itself into false tool calls. Because our challenge score averages token ratios geometrically across benchmarks of very different lengths, saving one token on a $99$-token tool call is worth $8.8\times$ as much as saving one token on an $873$-token code problem. Redistributing the budget via the score's exact optimality condition—giving $10\text{--}20$ thinking tokens back to math and simple tool calls where the accuracy slope is steep, while cutting deeper on instruction following, code, and irrelevant-tool checks—unlocks an extra $4.4$ to $8.0$ percentage points of token saving ($59.2\%\text{--}62.9\%$ total saving) at the exact same $-0.78$-point accuracy cost.
