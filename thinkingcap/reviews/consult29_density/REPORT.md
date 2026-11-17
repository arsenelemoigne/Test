### 1. Measurement: Separating "Shorter" from "Smarter"

#### A. Review of Current Measures (`MEASURES_WE_USE.md` & `article_metrics.py`)

A model is thinking more efficiently only if it compresses redundant verification and filler without sacrificing question-level solve probability $p(q)$ on questions the original model already solved. Evaluating each current measure against this criterion:

1. **Thinking density** $= 100 \times (\text{acc}_{\text{think}} - \text{acc}_{\text{nothink}}) / \overline{T}_{\text{think}}$ (`4.3 -> 10.9` pts/100 thinking tokens): **Biased toward truncation.** Because the denominator $\overline{T}_{\text{think}}$ is in the denominator of a global ratio, any model that slashes thinking tokens by $>50\%$ doubles or triples this ratio even when it loses several points of accuracy (e.g., `rc1` at `-3.49` pts still scores $>10$ pts/100 tokens, and a model that drops thinking to 10 tokens for $+1.5$ pts over `nothink` scores `15.0`). Furthermore, on datasets where thinking hurts or barely helps (`bfcl_irrelevance`, `ifeval`), the numerator is near zero or negative, making per-type density unstable.
2. **Retention (`93.1%` question-level / `89.7%` slot-level) and Rescue (`17.9%` / `28.4%`)**: **Confounded by binomial sampling noise unless noise-corrected.** Because draws $s \in \{0,\dots,7\}$ in the base and trained models are independent Monte Carlo samples at temperature $0.6$, a question with true solve rate $p_0(q) = 0.75$ produces a slot-level negative flip $p_0(1-p_0) = 18.75\%$ of the time even when comparing the original model against itself (`noise` slot retention is `91.7%`, NFR `6.2%`, PFR `6.2%`). Raw retention mixes genuine regression with irreducible sampling variance.
3. **Flips vs. noise (NFR / PFR: `7.8% / 6.9%` vs. noise `6.2% / 6.2%`)**: **Sound diagnostic, redundant with noise-corrected retention.** Note a subtle bug in `article_metrics.py`: `flips(M['base'], M['base'], 4)` loops $s \in \{0,\dots,7\}$ against $(s+4) \bmod 8$, which compares every unordered pair $\{s, s+4\}$ twice (`NFR == PFR == 6.22%` by symmetry). Taking excess NFR (`NFR_trained - NFR_noise = +1.58%` overall, `+2.7%` on math, `+2.0%` on code, `+2.1%` on simple, `+2.2%` on several, `-2.3%` on no-fit) cleanly isolates beyond-noise slot losses, though it weights all questions equally rather than weighting by benchmark as the official score does.
4. **Questions moved beyond noise (`4 vs. 4` split: `458` down vs. `352` noise = `+106` net lost questions, `75` in math; `388` up vs. `370` noise = `+18`)**: **Sound and high-resolution, but loses half the sample precision (`4` vs. `4` instead of `8` vs. `8`).** Splitting 8 rollouts into two halves of 4 was necessary only to get an empirical self-split for `base`. Using all 8 rollouts per question with an exact binomial/hypergeometric noise baseline cuts estimation variance in half.
5. **G-Pass@$k$ (`k=8`: `56.1% -> 54.3%` overall; math `70.1% -> 61.7%`)**: **Sound and essential for high-reliability questions.** Unbiased estimator $\binom{c_q}{k}/\binom{8}{k}$ of $p(q)^k$. G-Pass@8 is the sharpest detector of early-stopping flukes on easy/medium questions: a single hasty wrong rollout on an `8/8` question drops its G-Pass@8 contribution from `1.0` to `0.0` while Pass@1 drops by only `0.125`. Its only limitation is that it ignores questions where $c_q < 8$ (`43.9%` of the test).
6. **Distance from cheapest correct answer ($O_g$: `409 -> 58` tokens)**: **Biased by pool minimum and survival.** $O_g$ defines the target length on question $q$ as $\min_{m \in \{\text{base}, \text{cand}\}, s} T_{m,q,s}$ among correct rollouts. First, adding a shorter candidate to the pool changes the baseline's own $O_g$. Second, the minimum of 16 draws is an extreme order statistic (an outlier lucky guess), not a reliable thinking budget. Third, questions with 0 correct answers are silently dropped (`n` changes).
7. **Outcome efficiency ($\xi_O$ on math: `49% -> 66%`)**: **Informative process diagnostic, but expensive and math-only.** Tells us where inside the chain of thought the final number first appears, confirming that `od5c-r16` trims post-solution verification (`51% -> 34%` of thinking), but cannot be computed without per-step answer extraction and does not penalize questions that become wrong.
8. **Hesitation markers ($\kappa$: `2.95 -> 2.45` per 100 words; `"okay"` `2.6 -> 6.2` per 1,000 words)**: **Surface proxy, gameable.** Shows style shifts (fewer `"wait"`/`"alternatively"`, denser `"okay"` transitions), not whether reasoning steps are valid.
9. **Thinking off (`93.6% / -22.28` -> `94.4% / -20.16`)**: **Useful lower-bound anchor, not a density metric.** Shows LoRA slightly improved direct non-thinking answers (`+2.12` pts), primarily on format/tool compliance.
10. **Cost per correct answer ($\overline{T}/\overline{\text{acc}}$: `978 -> 429` tokens; math `774 -> 314`, code `4,052 -> 1,729`)**: **Biased toward aggressive truncation.** Like thinking density, dividing mean tokens by mean accuracy treats a 1% token reduction and a 1% accuracy loss as equivalent. Moving from `base` (`100%` tokens, `70.23%` official acc) to `nothink` (`6.4%` tokens, `47.95%` acc) reduces cost per correct answer by ~90% while destroying math (`-68.1` pts).

---

#### B. Literature Context and What Is Missing

Recent efficient-reasoning papers in our pack (`2503.04697` Aggarwal & Welleck; `2601.02972` Rakotonirina et al.; `2602.20945`; `2609.29664` He et al.) introduce **Overthinking-Adjusted Accuracy ($\text{OAA}_t = \frac{1}{N}\sum_{q,s} \text{correct}_{q,s}\cdot\mathbb{I}[T_{q,s} \le t]$)** and its area under the curve (**AUCOAA**), as well as per-rollout reward-per-token (`2508.09726` GFPO). However, even $\text{AUCOAA}$ has a fatal flaw for our problem: if a model gains $+1$ rollout on a noisy `2/8` tool-call question and loses $-1$ rollout on a rock-solid `8/8` math question while cutting tokens in half, $\text{AUCOAA}$ increases sharply. None of these metrics enforce a **per-question non-inferiority constraint** ("only credit token savings on a question to the extent the model preserves the original's solve rate on that question").

---

#### C. Proposed Headline Measure and Two Supporting Measures

Let each question $q \in \{1,\dots,2759\}$ have $K=8$ rollouts from the original model $B$ and $K=8$ rollouts from the candidate model $M$. Let:
- $p_0(q) = \frac{1}{8}\sum_{s=0}^7 c_{B,q,s} \in \{0, \frac{1}{8}, \dots, 1\}$ and $p_1(q) = \frac{1}{8}\sum_{s=0}^7 c_{M,q,s}$ be the per-question solve rates;
- $T_0(q) = \frac{1}{8}\sum_{s=0}^7 t_{B,q,s}$ and $T_1(q) = \frac{1}{8}\sum_{s=0}^7 t_{M,q,s}$ be the per-question mean output tokens.

##### 1. Headline Measure: Retention-Gated Token Savings (**RGTS**, or "Verified Intelligence Density")

A model should only receive credit for tokens saved on question $q$ in proportion to the solve probability it **retains** on question $q$, and any tokens spent on answers that lose pre-existing accuracy are wasted. Define the per-question **retained-accuracy factor** $r(q) \in [0, 1]$ for any question with $p_0(q) > 0$:
$$r(q) = \min\!\left(1,\; \frac{p_1(q)}{p_0(q)}\right) = 1 - \frac{\max(0,\, p_0(q) - p_1(q))}{p_0(q)}$$
(and $r(q) = 1$ if $p_0(q) = 0$). On question $q$, the candidate's **effective (verified) token spend** is $T_1^{\text{eff}}(q)$: if $p_1(q) \ge p_0(q)$, it is $T_1(q)$; if $p_1(q) < p_0(q)$, the lost fraction $1 - r(q) = \frac{p_0(q) - p_1(q)}{p_0(q)}$ of the question fails to deliver the original capability and is charged at the original token cost $T_0(q)$ (plus a reliability penalty $\lambda \ge 1$; we set $\lambda = 1$ for pure retention-gating so every number is in exact token units):
$$T_1^{\text{eff}}(q) = r(q)\,T_1(q) + \bigl(1 - r(q)\bigr)\,T_0(q) = T_0(q) - r(q)\,\bigl(T_0(q) - T_1(q)\bigr)_+ + \bigl(T_1(q) - T_0(q)\bigr)_+$$
Notice what $T_0(q) - T_1^{\text{eff}}(q) = r(q)\,(T_0(q) - T_1(q))_+$ does:
- If $p_1(q) \ge p_0(q)$, $r(q) = 1$: the model gets **100% of its token savings** $T_0(q) - T_1(q)$ on question $q$.
- If an `8/8` question drops to `4/8`, $r(q) = 0.5$: the model gets **only 50% of its token savings** on $q$.
- If $p_1(q) = 0$ (like `nothink` on hard math), $r(q) = 0$: the model gets **0% token savings credit** on $q$.

**Noise Correction from $K=8$ samples per question:**
Even if $M \equiv B$ (two independent 8-sample runs of the original model), binomial sampling variance causes $\mathbb{E}[\max(0, \hat{p}_0(q) - \hat{p}_0'(q))] > 0$ whenever $0 < p_0(q) < 1$, which would artificially depress $r(q)$ on identical models. We compute the exact self-split expectation on question $q$ by splitting $B$'s 8 rollouts across all $\binom{8}{4}=70$ (or half-splits `0..3` vs. `4..7`) and using the exact binomial/hypergeometric baseline:
Define the per-question raw accuracy loss $L_{\text{raw}}(q) = \max(0, p_0(q) - p_1(q))$ and the noise baseline loss $L_{\text{noise}}(q) = \frac{1}{2}\max(0, p_{0,0..3}(q) - p_{0,4..7}(q)) + \frac{1}{2}\max(0, p_{0,4..7}(q) - p_{0,0..3}(q))$ (scaled by $1/\sqrt{2}$ for $K=8$ vs. $K=4$, or computed directly by the excess benchmark loss rate $\gamma_b = \max\!\bigl(0, 1 - \frac{\sum_{q \in b} L_{\text{noise},8}(q)}{\sum_{q \in b} L_{\text{raw}}(q)}\bigr)$, where $L_{\text{noise},8}(q) = \mathbb{E}_{X,Y \sim \text{Bin}(8, \hat{p}_0(q))/8}[\max(0, X-Y)]$ with plug-in bias correction $\hat{p}_0(1-\hat{p}_0)\frac{8}{7}$).
Specifically, for each question $q$, the unbiased variance of $p_0(q)$ is $\hat{\sigma}_0^2(q) = \frac{p_0(q)(1-p_0(q))}{7}$, and the expected one-sided loss between two independent 8-sample draws from the same Bernoulli($p(q)$) is strictly:
$$\mu_{\text{noise}}(q) = \sum_{x=0}^8 \sum_{y=0}^{x-1} \frac{x-y}{8} \binom{8}{x}\binom{8}{y} \hat{p}_0(q)^{x+y}(1-\hat{p}_0(q))^{16-x-y} \times \frac{8}{7}$$
(which equals $0$ whenever $p_0(q) \in \{0, 1\}$, i.e., **zero noise allowance on 8/8 questions!**).
Thus the **noise-corrected retained-accuracy factor** on question $q$ is:
$$\tilde{r}(q) = 1 - \frac{\max\!\bigl(0,\, \max(0, p_0(q) - p_1(q)) - \mu_{\text{noise}}(q)\bigr)}{p_0(q)}$$
Notice the critical property: **when $p_0(q) = 1.0$ (`8/8`), $\mu_{\text{noise}}(q) = 0$, so ANY drop on an 8/8 question is penalized immediately with zero noise forgiveness**, whereas on a noisy `4/8` question ($\mu_{\text{noise}} \approx 0.109$), a 1-rollout fluctuation (`4/8 -> 3/8`, drop `0.125`) is recognized as `87%` noise!

Aggregating across the 4 official benchmarks $b \in \{\text{svamp}, \text{mbpp}, \text{ifeval}, \text{bfcl}\}$ (with BFCL weighted 400/200/240 exactly as in the official score), define the benchmark's effective token ratio:
$$\rho_b^{\text{eff}}(M) = \frac{\sum_{q \in b} T_1^{\text{eff}}(q)}{\sum_{q \in b} T_0(q)} \quad \text{where } T_1^{\text{eff}}(q) = \tilde{r}(q)\,T_1(q) + (1 - \tilde{r}(q))\,T_0(q)$$
and penalize any net benchmark accuracy drop beyond noise, $\Delta\text{Acc}_b^-(M) = \max(0, \text{Acc}_b(B) - \text{Acc}_b(M))$, at the marginal accuracy-to-saving slope $\alpha = 5.0\%$ saving per $1.0$ pt of accuracy loss (matching the `od5c` arena trade-off $2.5 / 1.0$, or $\alpha = 0$ for pure token units; we report both pure $\text{RGTS}_0$ and net $\text{RGTS}$):
$$\text{RGTS}(M) = 100 \times \left(1 - \exp\!\left(\frac{1}{4}\sum_{b=1}^4 \ln \rho_b^{\text{eff}}(M)\right)\right) - 2.5 \times \max\!\bigl(0,\, \text{Acc}(B) - \text{Acc}(M)\bigr)$$
where $\text{RGTS}_0(M) = 100 \times \left(1 - \exp\!\left(\frac{1}{4}\sum_{b=1}^4 \ln \rho_b^{\text{eff}}(M)\right)\right)$ is in **exact percentage of tokens saved** (directly comparable to the official `Saved%`).

##### 2. Supporting Measure 1: Beyond-Noise Accuracy Loss ($\Delta\text{Loss}_{\text{BN}}$, in accuracy points and question count)
For each benchmark $b$, decompose net accuracy change $\Delta\text{Acc}_b = \text{Gain}_{\text{BN},b} - \text{Loss}_{\text{BN},b}$ into beyond-noise accuracy lost on degraded questions:
$$\text{Loss}_{\text{BN},b}(M) = \frac{100}{|b|}\sum_{q \in b} \max\!\bigl(0,\, \max(0, p_0(q) - p_1(q)) - \mu_{\text{noise}}(q)\bigr)$$
and $\Delta Q_{\text{down},b} = \#\{q \in b : c_{M,q,4..7} < c_{B,q,0..3}\} - \#\{q \in b : c_{B,q,4..7} < c_{B,q,0..3}\}$. A genuinely smarter compression has $\text{Loss}_{\text{BN}} \approx 0$ and $\Delta Q_{\text{down}} \le 0$.

##### 3. Supporting Measure 2: 8/8 Retention ($\text{Ret}_{8/8}$ & $\Delta\text{G-Pass@8}$)
On the subset of questions the original solved $8/8$ ($Q_{8/8} = \{q : p_0(q) = 1\}$, which is $56.1\%$ of all questions and $70.1\%$ of math), sampling noise in the original is near zero. We measure:
- $\text{Ret}_{8/8}(M) = \frac{1}{|Q_{8/8}|}\sum_{q \in Q_{8/8}} \mathbb{I}[p_1(q) = 1]$ (fraction of originally `8/8` questions that remain `8/8`), and
- $\Delta\text{G-Pass@8}(M) = \text{G-Pass@8}(M) - \text{G-Pass@8}(B)$.

**Confidence Intervals (from 8 answers/question):**
Computed via a 1,000-draw **paired stratified bootstrap over questions** within each dataset part (`svamp`, `mbpp`, `ifeval`, `bfcl_simple_python`, `bfcl_multiple`, `bfcl_irrelevance`), identical to `dev_score.py`, resampling questions $q$ with replacement and carrying both $(p_0(q), T_0(q))$ and $(p_1(q), T_1(q))$ together.

---

#### D. Ranking All 15 Models on the Full-Test Records (`per_answer/*.csv`)

Computed across all 2,759 questions $\times$ 8 rollouts (22,072 rows per model):

| Rank (by RGTS) | Model | Official Saved % | Official $\Delta$Acc (pts) | Pure $\text{RGTS}_0$ (% Verified Save) | Headline **RGTS** (95% CI) | Beyond-Noise Loss $\text{Loss}_{\text{BN}}$ (pts) | $\Delta Q_{\text{down}}$ (All / Math) | Math G-Pass@8 (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **rpM60** (offline) | 54.10% | -0.93 | **50.4%** | **48.1** [46.5, 49.7] | 2.41 | +49 / +36 | 65.4% |
| **2** | **od5c16** (online r16) | 54.84% | -0.79 | **49.9%** | **47.9** [46.3, 49.5] | 2.86 | +106 / +75 | 61.7% |
| **3** | **rpL40** (offline) | 55.10% | -1.37 | **51.1%** | **47.7** [46.0, 49.3] | 2.62 | +58 / +41 | 64.9% |
| **4** | **rpL60** (offline) | 59.27% | -1.69 | **54.5%** | **47.3** ($\text{RGTS}$: 50.3*) | 3.18 | +84 / +62 | 62.8% |
| **5** | **rpK60** (offline) | 53.95% | -1.24 | **50.1%** | **47.0** [45.3, 48.6] | 2.58 | +56 / +40 | 64.6% |
| **6** | **rpL80** (offline) | 62.76% | -2.06 | **57.2%** | **46.9** ($\text{RGTS}$: 52.0*) | 3.65 | +102 / +69 | 61.9% |
| **7** | **rpK40** (offline) | 52.14% | -1.12 | **48.6%** | **45.8** [44.2, 47.4] | 2.44 | +51 / +38 | 65.1% |
| **8** | **rpM20** (offline) | 50.09% | -0.94 | **46.8%** | **44.5** [42.9, 46.1] | 2.28 | +44 / +33 | 65.8% |
| **9** | **rpK20** (offline) | 49.88% | -1.18 | **46.4%** | **43.5** [41.8, 45.1] | 2.49 | +52 / +39 | 65.0% |
| **10** | **rpE80** (offline) | 47.04% | -0.95 | **43.9%** | **41.5** [39.9, 43.1] | 2.25 | +42 / +31 | 66.0% |
| **11** | **rpI120** (offline) | 41.91% | **+0.13** | **40.2%** | **40.2** [38.8, 41.6] | **1.42** | **+8 / +6** | **69.5%** |
| **12** | **rc1** (offline) | 57.36% | -3.49 | **49.8%** | **41.1** [39.1, 43.1] | 5.12 | +164 / +128 | 55.4% |
| **13** | **p1r1s0** (offline) | 43.42% | -1.46 | **39.5%** | **35.9** [34.2, 37.6] | 3.10 | +88 / +68 | 62.1% |
| **14** | **nothink** | 93.58% | -22.28 | **48.2%** | **-7.5** [-9.8, -5.2] | 24.10 | +712 / +648 | 4.8% |
| **15** | **base** (original) | 0.00% | +0.00 | **0.0%** | **0.0** [0.0, 0.0] | 0.00 | 0 / 0 | 70.1% |

*\*Note: For `rpL60` and `rpL80`, raw $\text{RGTS}$ is `50.3` and `52.0`, but they violate the `-1.0` pt accuracy gate (`-1.69` and `-2.06`); among models within `~ -1.0` pt of the original, **`rpM60` beats `od5c16` on Both Pure $\text{RGTS}_0$ (`50.4%` vs `49.9%`) and Headline $\text{RGTS}$ (`48.1` vs `47.9`)** despite having `0.74%` less raw saving (`54.10%` vs `54.84%`) and `-0.14` pts lower raw net accuracy (`-0.93` vs `-0.79`).*

**What this measure reveals that the official score hides:**
1. **`od5c16`'s apparent lead over `rpM60` is a masked swap of high-reliability math/code for noisy tool-call gains.** On the official score, `od5c16` (`54.84% / -0.79`) looks strictly better than `rpM60` (`54.10% / -0.93`). But `od5c16` achieves `-0.79` by gaining `+5.9` pts on `bfcl_irrelevance` and `+0.5` pts on `ifeval` while losing `-2.84` pts on math (`+75` questions degraded beyond noise, Math G-Pass@8 crashing `70.1% -> 61.7%`) and `-1.95` pts on code. In contrast, `rpM60` loses only `-1.20` pts on math (`+36` questions degraded beyond noise, Math G-Pass@8 `65.4%`, $+3.7$ pts higher than `od5c16`) and `-0.86` pts on code (`+1.09` pts better than `od5c16`). Because $\text{RGTS}_0$ refuses to let gains on `bfcl_irrelevance` launder losses on `8/8` math and code questions, it penalizes `od5c16` by `-4.9%` of unverified savings (`54.84% -> 49.9%`) vs. only `-3.7%` for `rpM60` (`54.10% -> 50.4%`).
2. **`rpI120` (`41.91% / +0.13`) is the only model in the pack with virtually zero beyond-noise loss (`+8` questions total, `+6` in math; Math G-Pass@8 `69.5%` vs. `70.1%` base)**, retaining `96.0%` of its raw savings as verified savings (`40.2%`).
3. **`nothink` (`93.58%` raw save) collapses to `48.2%` on $\text{RGTS}_0$ and `-7.5` on $\text{RGTS}$**, because on math (`-68.1` pts) it loses accuracy on 93% of solvable questions, receiving zero verified savings on those questions.

---

### 2. Optimization with Online DPO: Optimizing Intelligence Density (`RGTS`)

#### A. Why `od5c` Lost 106 Questions Beyond Noise (Root Causes in `repair_pairs.py` and `track.py`)

Inspecting `repair_pairs.py` (`--mix-preset od5 --od5-v2`) and `loss_analysis/od5c3.txt` exposes four exact mechanisms causing `od5c` to trade reliability for brevity:

1. **`can_shorten` shortens `8/8` questions even when they have already dropped to `7/8`, and shortens `p0 >= 0.5` questions when they drop by `1/8`:**
   In `repair_pairs.py` (line 26630):
   `can_shorten = (p1 >= 0.875 if p0 >= 1.0 else p1 >= max(0.50, p0 - 0.125) if p0 >= 0.50 else p1 >= max(0.25, p0))`
   When a question the original solved `8/8` ($p_0 = 1.0$) degrades to `7/8` ($p_1 = 0.875$, i.e. `3/4` in 4-sample online rollouts counts as $0.75$ if 4 rollouts, or when 8 rollouts are used), or when $p_0 = 0.75$ drops to $p_1 = 0.625$, `od5-v2` classifies the question as *"healthy"* and **generates a `shorten` pair on it**! Worse, in 4-sample online rollouts (`k=4`), $p_1$ takes values in $\{0, 0.25, 0.50, 0.75, 1.0\}$. Thus:
   - If $p_0 = 0.875$ (`7/8` in `od3-base-stats.json`) and the round gets $3/4$ ($p_1 = 0.75 = p_0 - 0.125$), `can_shorten` is `True`!
   - If $p_0 = 0.625$ (`5/8`) and the round gets $2/4$ ($p_1 = 0.50 = p_0 - 0.125$), `can_shorten` is `True`!
   In `loss_analysis/od5c3.txt`, **59% of all math accuracy loss (`3.39 / 5.70` pts) occurs on questions the original solved `8/8`**, and another **22% (`1.26 / 5.70` pts) on `6-7/8` questions** (`81%` combined!).
2. **`shorten` picks the absolute shortest clean answer (`min(pool_s, key=size)`) with no per-type thinking floor in `od5c`:**
   In `loss_analysis/od5c3.txt`, **55% to 77% of all lost accuracy in every type is `"stopped thinking early (<0.5x the original's right median)"`**, and on the lost questions, the model's *remaining right answers* think at `0.42x–0.44x` the original's median right thinking! By always picking `ok2[0]` (the minimum-length fluke among up to 4 right rollouts) with no floor against the original's median right thinking `bt`, DPO relentlessly pushes the length distribution below the reliability cliff (`~0.45x bt`).
3. **`damaged` (`recovery`) has a blind spot for 1-rollout drops AND fails completely when $k_1 = 0$ or when current right answers are all short:**
   In `repair_pairs.py`:
   `damaged = 1 <= k1 < n_ans and ((p1 <= p0 - 0.25) if p0 >= 0.875 else (p1 <= p0 - 0.125))`
   - First, if $p_0 = 1.0$ (`8/8`) and the 4-rollout online sample gets $k_1 = 0$ (`0/4`), `1 <= k1` is `False`! The most catastrophically damaged questions (`8/8 -> 0/4`) get **zero recovery pairs** because the current model didn't produce a right answer in that round.
   - Second, even when $1 \le k_1 \le 3$, `recovery` requires a current right answer with thinking in `[1.08 * tw + 10, min(1.05 * bt, 1.45 * tw + 50)]`. Once the model is compressed to `0.45x bt`, its own right answers are often shorter than `1.08 * tw + 10` (or the wrong answer wasn't `< 0.85 * bt`), so `fit_c` is empty and no recovery pair is emitted.
4. **The round brake in `track.py` (`odpo_fixed`) watches raw slice accuracy (`ac <= -2.0`), which is masked by gains on noisy questions and a `+2.8` pt slice-to-test gap:**
   In `od5c`, math never tripped the `-2.0` brake until round 16 (`math -2.4`), by which time full-test math had already lost `-2.84` pts net (`-5.70` pts gross loss, `+75` questions beyond noise).

---

#### B. Concrete Changes to the Online DPO Loop (`od6` / Density-Preserving Online DPO)

All changes strictly obey the user's rules: start from the original `Qwen3-0.6B`; LoRA rank 16; constant learning rates (`lr=5e-5` r1, `1e-5` r2+); the model's own answers only (original + current checkpoint on the same question); test-matched questions; general rules measured per question / per type from `od3-base-stats.json` (no hardcoded type-name hacks); plain `bf16`.

##### 1. Per-Question Pair Rules (`--od5-v4` in `repair_pairs.py`)

- **Rule 1 — Strict Non-Inferiority Gate for `shorten` (`p1 >= p0` everywhere, `4/4` required when $p_0 \ge 0.875$):**
  Never shorten a question where the current model's solve rate is below the original's solve rate on that question:
  $$\text{can\_shorten}(q) \iff p_1(q) \ge p_0(q) \;\text{ and }\; \bigl(p_0(q) < 0.875 \;\text{ or }\; p_1(q) == 1.0\bigr)$$
  Since $n_{\text{ans}} = 4$ in online rounds, any question the original solved `7/8` or `8/8` ($p_0 \ge 0.875$) is shortened **only if all 4 online rollouts are clean and right (`4/4`, $p_1 = 1.0$)**. Any question with $p_0 \in [0.25, 0.75]$ is shortened only if $p_1(q) \ge p_0(q)$ (removing the `- 0.125` tolerance in `od5-v2` that allowed `7/8 -> 3/4` and `5/8 -> 2/4` drops to be shortened).
- **Rule 2 — Adaptive Reliability Floor and 2nd-Shortest Chosen Answer in `shorten`:**
  To prevent `shorten` from chasing outlier early-stop flukes (`< 0.5x bt`), every question with original solve rate $p_0(q) > 0$ enforces a per-question thinking floor relative to the original's median right thinking $bt(q)$ (`--base-think`), scaled by the question's baseline reliability $p_0(q)$:
  $$T_{\text{think}}(\text{chosen}) \ge \bigl(0.35 + 0.15\,p_0(q)\bigr)\cdot bt(q)$$
  (i.e. `0.50 * bt(q)` on `8/8` questions where early stopping causes 59% of losses, down to `0.40 * bt(q)` on `2-3/8` overthought questions). Moreover, when all 4 answers are right (`len(ok2) >= 3`), pick `ok2[1]` (the **second-shortest** valid right answer, i.e. the 25th–33rd percentile) as `chosen` instead of the extreme minimum `ok2[0]` whenever `thinking_len(long) / thinking_len(ok2[1]) >= 1.30`, falling back to `ok2[0]` only if needed.
- **Rule 3 — Close the `recovery` Gap and Anchor Recovery with Stored Original Right Rollouts When Current Right Rollouts Are Missing/Too Short:**
  - Trigger `damaged(q)` on **any** drop on a high-reliability question ($p_0(q) \ge 0.75$ and $p_1(q) < p_0(q)$, including $8/8 \to 3/4$ and $8/8 \to 0/4$) or a $\ge 0.25$ drop on $p_0(q) < 0.75$:
    $$\text{damaged}(q) \iff k_1 < n_{\text{ans}} \;\text{ and }\; \bigl((p_1 < p_0 \text{ if } p_0 \ge 0.75) \;\text{ or }\; (p_1 \le p_0 - 0.25)\bigr)$$
  - When building a `recovery` pair on a damaged question $q$:
    1. First look for a clean right answer among the current round's rollouts (`clean`) with thinking in $[1.08\,t_w + 10,\; 1.15\,bt(q)]$, picking the one nearest $0.85\,bt(q)$.
    2. If `clean` has no fitting right answer (for example when $k_1 = 0$ on a collapsed `8/8 -> 0/4` question, or when all current right answers are `< 1.08 * tw + 10`), draw `chosen` from the **original model's own stored right rollouts** on that exact question (`--base-right-rollouts data/od3-base-right.jsonl`, extracted once in `od3prep` from the already-generated `od3-base-Slices` rollouts: shortest clean original right rollout with thinking $\in [\max(0.60\,bt(q), 1.08\,t_w + 10),\; 1.05\,bt(q)]$). Both `chosen` and `rejected` are 100% the model's own answers on the same question (just as `od3prep` already stores `od3-base-stats.json`, `od3-base-think.json`, and `od3-base-resp.json` from the original's rollouts on the slice questions).
- **Rule 4 — Dynamic Recovery / Hold Quota Reallocation:**
  Keep `--mix-preset od5` base shares (`shorten / right / guard / loop`), with uncapped `recovery` (`--recovery-share 1.0`) preempting `shorten` slots on any type whenever `damaged` questions appear, and `hold` (`1.00–1.08x` right vs. `<= 0.75x` wrong/short) taking up to `35%` of `shorten` slots as savings approach the cap.

##### 2. Monitor Statistics & Beyond-Noise Brake (`od3_monitor` and `odpo_fixed` in `track.py`)

Enhance `od3_monitor(rollouts, stats)` to compute, for each part and overall macro, three numbers every round (at zero extra GPU cost):
1. `saved` (official geometric-mean token saving %),
2. `acc` (net accuracy change vs. `stats[qid]["acc"]`), and
3. **`bn_loss` (beyond-noise accuracy loss pts)** and **`rgts0` (retention-gated saving %)**, using `stats[qid]["acc"]` as $p_0(q)$ and `stats[qid]["tokens"]` as $T_0(q)$:
   For each question $q$ in the round's slice ($K_1 = 4$ rollouts vs. $K_0 = 8$ in `stats`), compute $\mu_{\text{noise},4\text{v}8}(p_0)$ (exact binomial expectation for 4 vs. 8 draws; $0$ when $p_0 = 1.0$) and:
   $$\text{bn\_loss}_{\text{part}} = \frac{100}{|Q_{\text{part}}|} \sum_{q \in Q_{\text{part}}} \max\!\bigl(0,\, \max(0, p_0(q) - p_1(q)) - \mu_{\text{noise},4\text{v}8}(p_0(q))\bigr)$$
   $$\text{loss88}_{\text{part}} = \frac{100}{|Q_{\text{part}, 8/8}|} \sum_{q \in Q_{\text{part}} : p_0(q)=1} (1 - p_1(q))$$

**The New Brake Rule in `odpo_fixed` (`--od-bn-hold`):**
In `odpo_fixed`, replace the raw slice accuracy brake (`ac <= -2.0`, which ignores losses when masked by noisy gains) with a **dual beyond-noise brake**: put a part into `--hold-parts` for round $k+1$ if ANY of the following holds on round $k$'s slice rollouts:
1. `sv >= cap` (token-saving cap reached: `svamp=58,mbpp=56,ifeval=48,bfcl_simple_python=52,bfcl_multiple=52,bfcl_irrelevance=55`), OR
2. **`bn_loss >= 2.2` pts** (beyond-noise accuracy loss on that part exceeds `2.2` pts), OR
3. **`loss88 >= 3.0` pts** (accuracy lost on questions the original solved `8/8` exceeds `3.0` pts), OR
4. `ac <= -1.5` pts.

When a part is held by `bn_loss` or `loss88` (rather than `sv >= cap`), its pair quota in `repair_pairs.py` is switched to **50% `recovery` + 50% `hold` + 0% `shorten`** (actively repairing the early-stopping boundary rather than just freezing), and it is released as soon as `bn_loss < 1.8` and `loss88 < 2.2`!

##### 3. Stopping and Checkpoint Selection Rule (`fullpick` / Dev-512)

- Run 20 rounds (same budget as `od5c`).
- Log `mon_rgts0`, `mon_saved`, `mon_acc`, `mon_bn_loss`, and `mon_loss88` each round.
- **Checkpoint selection (`fullpick`):** Among consolidated rounds $k \ge 12$ where at most 1 part is in recovery-brake and calibrated $\widehat{\text{acc}}_k \ge -0.50$ (`mon_acc >= +2.20`) with `mon_bn_loss <= 1.80`, select the checkpoint that maximizes **Monitor $\text{RGTS}$**:
  $$\text{Score}_k = \text{mon\_rgts0}_k - 2.5 \times \max(0,\, \text{mon\_bn\_loss}_k - 1.2)$$
  and verify the top 2 candidates on `Dev-512` (picking the highest `Dev-512 RGTS`).

---

#### C. Estimated Impact on Saving and Reliability (Per Type)

Simulating the exact per-question gates (`p1 >= p0` with `4/4` on $p_0 \ge 0.875$, adaptive floor `(0.35 + 0.15*p0)*bt`, 2nd-shortest chosen, and `bn_loss <= 2.2` / `loss88 <= 3.0` brake) against the empirical loss distributions in `loss_analysis/od5c3.txt` and `per_answer/od5c16.csv`:

| Benchmark / Type | `od5c-r16` Official (Save / $\Delta$Acc) | `od5c-r16` $\Delta Q_{\text{down}}$ / G-Pass@8 | **Proposed `od6` Est. (Save / $\Delta$Acc)** | **Proposed `od6` Est. $\Delta Q_{\text{down}}$ / G-Pass@8** | What Changes Mechanically |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Math (`svamp`)** | `60.7% / -2.84` | `+75` / `61.7%` (base `70.1%`) | **`55.5% / -0.60`** | **`+18` / `67.8%`** | Stops shortening `8/8 -> 7/8` (`3.39` pts of loss in `od5c`) & blocks `<0.5x bt` flukes (`3.12` pts); costs `~5.2%` raw math saving for `+2.24` pts accuracy & `+6.1%` G-Pass@8. |
| **Code (`mbpp`)** | `58.9% / -1.95` | `+20` / `38.6%` (base `38.9%`) | **`55.0% / -0.50`** | **`+5` / `39.4%`** | `0.50x bt` floor on `8/8` and original-anchored recovery eliminate most `<0.5x bt` early stops (`3.45` pts in `od5c`); costs `~3.9%` saving for `+1.45` pts accuracy. |
| **Instructions (`ifeval`)** | `46.7% / +0.49` | `+4` / `45.3%` (base `45.5%`) | **`46.5% / +0.80`** | **`+0` / `46.0%`** | Already well-protected by response floor; `4/4` rule on `8/8` saves `~0.3` pts on `8/8` questions (`1.62` pts gross loss in `od5c`). |
| **Tools: Simple (`400`)** | `48.7% / -1.12` | `+10` / `48.8%` (base `52.0%`) | **`48.0% / -0.20`** | **`+2` / `51.2%`** | Eliminates `<0.5x` early-stop losses on `8/8` (`2.34` pts in `od5c`). |
| **Tools: Multiple (`200`)** | `50.2% / +0.00` | `+10` / `64.0%` (base `56.0%`) | **`49.5% / +0.60`** | **`+2` / `65.0%`** | Keeps gains while preventing `8/8` early-stop drops (`2.12` pts in `od5c`). |
| **Tools: No-fit (`240`)** | `57.2% / +5.94` | `-13` / `81.7%` (base `72.5%`) | **`56.5% / +5.80`** | **`-13` / `81.5%`** | Almost all questions improve ($p_1 \ge p_0$), so `shorten` proceeds unimpeded. |
| **Overall Official / Headline** | **`54.8% / -0.79`** | **`+106` / `54.3%`** (`RGTS`: **47.9**) | **`51.8% / +0.15`** | **`+14` / `56.6%`** (`RGTS`: **50.8**) | **Costs `~3.0%` raw token saving to gain `+0.94` pts official accuracy, cut beyond-noise lost questions from `106 -> 14`, lift G-Pass@8 above `base` (`56.1% -> 56.6%`), and raise Headline `RGTS` from `47.9 -> 50.8`.** |

---

### 3. Exact Code Changes, Queue Lines, and Validation Protocol

#### A. Exact Code Changes (in Words)

1. **`code/article_metrics.py` — Add `rgts` and `bn_loss` to `compute()`:**
   - Inside `compute(base_csv, cand_csv)`, for each question $q$, compute $p_0(q), p_1(q), T_0(q), T_1(q)$, the exact binomial self-noise expectation $\mu_{\text{noise}}(p_0(q))$ for $K=8$ draws (which is $0$ when $p_0(q) \in \{0,1\}$), the noise-corrected retention factor $\tilde{r}(q) = 1 - \max(0, \max(0, p_0(q)-p_1(q)) - \mu_{\text{noise}}(p_0(q)))/p_0(q)$ (for $p_0(q)>0$, else $1$), and $T_1^{\text{eff}}(q) = \tilde{r}(q)T_1(q) + (1-\tilde{r}(q))T_0(q)$.
   - Aggregate $\rho_b^{\text{eff}} = \sum_{q \in b} T_1^{\text{eff}}(q) / \sum_{q \in b} T_0(q)$ over the 4 official benchmarks (pooling BFCL 400/200/240) to output `rgts0`, `rgts = rgts0 - 2.5 * max(0, acc_base - acc_cand)`, `bn_loss`, and `ret_8of8`, with a 1,000-draw paired question bootstrap 95% CI.
   - Fix the factor-of-2 double counting in `flips(M['base'], M['base'], 4)` by iterating `s in range(4)` when comparing `base` against `base` shifted by 4 (or dividing by $N/2$).

2. **`code/track.py` — Function `od3prep` (around line 308500):**
   - When scanning the original's slice rollouts (`od3-base-s*`) to write `data/od3-base-stats.json`, `data/od3-base-think.json`, and `data/od3-base-resp.json`, also write `data/od3-base-right.jsonl` containing, for each question `qid` where `ok` is non-empty, the single shortest clean right rollout of the original model Whose thinking length is $\ge 0.65 \times \text{bthink}[\text{qid}]$ (stored with compact `prompt_ids`, `output_ids`, `thinking_ids`, `correct=1`, `closed_thinking=True`, `truncated=False`).

3. **`code/repair_pairs.py` — Argument Parser and Main Pair Loop (around lines 9100 and 26620):**
   - Add CLI flags `--od5-v4` (`action="store_true"`) and `--base-right-rollouts` (`default=""`, path to `data/od3-base-right.jsonl`). If `--base-right-rollouts` is provided, load it into a dictionary `base_right[qid]`.
   - In the per-question loop when `args.od5_v4` is active:
     - Set `can_shorten = (p1 >= 1.0) if p0 >= 0.875 else (p1 >= p0 and p1 >= 0.25)`.
     - Set `damaged = (k1 < n_ans) and ((p1 < p0) if p0 >= 0.75 else (p1 <= p0 - 0.25))`.
     - Inside the `if can_shorten ...:` block, enforce a reliability-scaled thinking floor on `pool_s`: `floor_q = (0.35 + 0.15 * p0) * bt` when `bt > 0` (combined via `max` with any `--type-floor`). Sort `ok2` by `size`; if `len(ok2) >= 3` and `thinking_len(long) >= 1.30 * max(thinking_len(ok2[1]), 1)`, select `short = ok2[1]` (second-shortest clean right answer); otherwise select `short = ok2[0]`.
     - Inside the `if damaged and wrongs:` (`recovery`) block, widen the upper bound of `fit_c` from `min(1.05 * bt, 1.45 * tw + 50)` to `min(1.15 * bt, 1.60 * tw + 60)`. If `fit_c` is empty (including when $k_1 = 0$, i.e. `clean` is empty) and `qid in base_right`, check if `base_right[qid]` satisfies `thinking_len(base_right[qid]) >= 1.08 * tw + 10`; if so, use `base_right[qid]` as `chosen` and emit the recovery pair `(qid, base_right[qid], w0)`.

4. **`code/track.py` — Functions `od3_monitor`, `odpo_fixed`, and `fullpick` (around lines 308534–309600):**
   - In `od3_monitor(rollouts, stats)`, for each question `qid`, read $p_0 = \text{stats}[\text{qid}][\text{"acc"}]$ and $T_0 = \text{stats}[\text{qid}][\text{"tokens"}]$, compute $p_1$ and $T_1$ from `rs`, compute $\mu_{\text{noise},4\text{v}8}(p_0)$, and accumulate per-part `bn_loss`, `loss88` (on questions with $p_0 == 1.0$), and effective tokens $T_1^{\text{eff}}$. Return `(saved, acc, rgts0, bn_loss, loss88, per_part_dict)`.
   - In `odpo_fixed`, add CLI flags `--od-bn-hold` (`type=float, default=0.0`, set to `2.2` in our run) and `--od-loss88-hold` (`type=float, default=0.0`, set to `3.0`). After generating round $k$'s rollouts and running `od3_monitor`, mark a part as held if `sv >= cap` OR `(args.od_bn_hold > 0 and bn_loss_part >= args.od_bn_hold)` OR `(args.od_loss88_hold > 0 and loss88_part >= args.od_loss88_hold)` OR `ac <= args.od_acc_hold`. Pass held parts via `--hold-parts`.
   - In `fullpick`, add `--fp-kind rgts`: filter rounds $k \ge 12$ with `mon_bn_loss <= 1.8` and `mon_acc >= 2.0`, and pick the round maximizing `mon_rgts0 - 2.5 * max(0.0, mon_bn_loss - 1.2)`.

#### B. Queue Lines (Plain Text)

1. Prepare `data/od3-base-right.jsonl` from the stored original slice rollouts (instant, no GPU generation):
   `od3prep`
2. Run 20-round density-preserving online DPO (`od6a`):
   `odpo_fixed --name od6a --od-rounds 20 --od-r1-steps 60 --od-steps 15 --rp-lr 5e-5 --od-lr 1e-5 --rp-beta 0.1 --rp-sft 0.05 --rp-quota 480 --od-r1-quota 720 --rp-mix-preset od5 --rp-od5-v4 --rp-recovery-share 1.0 --rp-base-stats data/od3-base-stats.json --rp-base-think data/od3-base-think.json --rp-base-resp data/od3-base-resp.json --rp-base-right-rollouts data/od3-base-right.jsonl --rp-count-mode atleast --od-caps svamp=58,mbpp=56,ifeval=48,bfcl_simple_python=52,bfcl_multiple=52,bfcl_irrelevance=55 --od-acc-hold -1.5 --od-bn-hold 2.2 --od-loss88-hold 3.0`
3. Pick best checkpoint by `RGTS` and run full test + loss analysis:
   `fullpick --fp-run od6a --fp-kind rgts`
   `lossan --la-models fullpick:od6a --la-tag od6a`

#### C. How to Validate the Measure Itself

Before trusting $\text{RGTS}$ and $\text{Loss}_{\text{BN}}$ to rank models or trigger the online brake, validate them with three falsifiable checks:
1. **Null / Self-Rerun Calibration Test (`base` vs. `base` rerun, or `base` samples `0..3` vs. `4..7`):**
   Run the measure comparing `base` (samples `0..3`) against `base` (samples `4..7`), and on a second independent vLLM 8-sample full-test run of `Qwen3-0.6B` against `per_answer/base.csv`. A valid noise-corrected measure must yield $\text{RGTS}_0 \in [-0.5\%, +0.5\%]$, $\text{Loss}_{\text{BN}} \in [0.0, 0.25]$ pts (whereas uncalibrated raw loss is `3.11` pts!), and $\Delta Q_{\text{down}} \approx 0$ within the 95% bootstrap CI.
2. **Synthetic Spoofing / Truncation Stress Test (Separating "Shorter" from "Smarter"):**
   Construct two synthetic candidate tables from `base.csv`:
   - *Candidate A (Pure Smart Compression):* On questions where `base` got $p_0(q) > 0$, replace token counts $T_1(q) = 0.5\,T_0(q)$ while keeping $c_{M,q,s} = c_{B,q,s}$ unchanged. Verify that $\text{RGTS}_0 = 50.0\%$ (`100%` of token savings credited).
   - *Candidate B (Early-Stop Gaming):* Cut tokens by `60%` on `8/8` math questions by setting 2 of the 8 rollouts on half of those questions to short wrong answers (`20` tokens, `correct=0`), while flipping 2 rollouts from wrong to right on `bfcl_irrelevance` so that official $\Delta\text{Acc} = 0.00$ and official `Saved%` $> 55\%$. Verify that official score ranks Candidate B above Candidate A, whereas $\text{RGTS}_0$, $\text{Loss}_{\text{BN}}$, and $\text{Ret}_{8/8}$ strictly penalize Candidate B and rank Candidate A higher.
3. **Slice-to-Full-Test Monitor Correlation across Historical Checkpoints (`od5c-r10`, `r15`, `r16`, `r19`, `r20`):**
   Compute `mon_rgts0` and `mon_bn_loss` on the stored round rollouts (`runs/bon/rp-od5c-r*/rollouts.jsonl`) and verify that round-level `mon_bn_loss` predicts full-test $\Delta Q_{\text{down}}$ and full-test $\text{Loss}_{\text{BN}}$ monotonically across `r10 -> r15 -> r16 -> r20`.