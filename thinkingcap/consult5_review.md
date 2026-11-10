## 1. Verdict — five lines

- **The objective has not been met on the full test.**
- rpE-80 achieves **47.0% savings / −0.9 points**: insufficient compression and essentially no accuracy margin.
- The completed rpI-120 achieves **41.9% / +0.1**: accuracy recovery, not accuracy-preserving additional compression.
- rpE-80’s apparent screen success, **50.4% / −0.2**, did not survive full evaluation.
- **The biggest problem is treating movement on a repeatedly selected screen as evidence of a reproducible improvement in the accuracy–length frontier.**

## 2. What the data actually shows

### Full-test results

I recomputed accuracy using weights \(1/4\) for SVAMP, MBPP+, IFEval and pooled BFCL, with BFCL internally weighted 400/200/240.

| Model | Official savings | Reported accuracy change | Recomputed from rounded parts |
|---|---:|---:|---:|
| rc1 | 57.4% | −3.5 | −3.502 |
| rpE-40 | 50.6% | −1.9 | −1.892 |
| rpE-60 | 47.8% | −1.2 | −1.200 |
| rpE-80 | 47.0% | −0.9 | −0.939 |
| rpI-120 | 41.9% | +0.1 | +0.135 |

Rounding explains these small discrepancies. It does not explain the missed target.

From rpE-80 to rpI-120, the geometric-mean token ratio increases from 0.530 to 0.581: **9.6% more output**, for approximately one point of accuracy recovery. rpI must now remove **more than 13.9% of its remaining output** while losing less than approximately 1.1 accuracy points.

Its aggregate recovery also hides uneven behavior:

- No-fit’s +5.8 contributes **+0.414 overall points**.
- IFEval’s −1.5 contributes **−0.375 overall points**.
- MBPP+ contributes +0.125.

Thus “original accuracy restored” is true only in the aggregate—not across capabilities.

**Savings cannot be reconstructed exactly from rounded per-part percentages.** The evaluator uses a geometric mean of benchmark token ratios, and pooling BFCL requires baseline token means. Averaging the six savings percentages would be wrong.

### The screen is optimistic—but not by a universal correction

For seven identifiable JSON screen/full matches:

- Screen savings exceed full savings by **1.21 points on average**, median 1.10; range **−0.10 to +3.40**.
- Screen accuracy changes exceed full changes by **0.42 points on average**, median 0.50; range **−0.30 to +0.80**.

Adding three matched comparisons requiring log entries gives mean gaps of **+1.35 savings / +0.32 accuracy**, with accuracy gaps ranging **−1.10 to +1.50**.

Consequently, neither “subtract one accuracy point” nor a fixed savings correction is justified. The gap is model-dependent. Also, `all_runs.json` is not exhaustive: the log contains full evaluations absent from its eight full records.

### “Our curve” is descriptive, not a yardstick of scientific progress

The quoted quadratic reproduces the historical screen fit under the old screen aggregation. It is fitted to correlated checkpoints and strengths—not independent draws from a model’s attainable frontier.

At 50% savings it predicts **−2.893 accuracy points**. Being one point “above the curve” would still miss the objective badly.

Using that same curve:

- rpE-80 full residual: **+1.448**.
- rpI-120 full residual: **+1.637**.
- Apparent frontier improvement: **only +0.189 points**.

That is not compelling evidence of a new regime. Moreover, approximately **65%** of screen points above 15% savings lie within one accuracy point of this curve. That is compatible with a shared trade-off plus substantial measurement noise; it does not identify a capacity limit.

### Fresh answers help—but “stop the drift” is false

Relative to rpE-80’s screen result:

| Continuation | Savings at step 120 | Increase in remaining output |
|---|---:|---:|
| rpG | 32.7% | +35.7% |
| rpH | 37.0% | +27.0% |
| rpI | 43.0% | +14.9% |

rpI **attenuates** length drift. It does not eliminate it, and its full result confirms the drift.

Freshness is not isolated: pair counts, IFEval coverage and counting-rule composition differ. For example, rpG has 208 IFEval pairs, while rpI has 320, including 152 counting-rule pairs. These are substantive training differences.

### Noise is large enough to change the conclusions

As an illustration—not a valid confidence interval for these runs—an independent-Bernoulli calculation gives accuracy-difference standard errors of approximately:

- **1.30 points on the screen**;
- **0.47 points on full evaluation**.

For individual screen parts, illustrative 95% difference half-widths are approximately **4 points for math and 6 points for code/IFEval**.

Actual uncertainty requires paired, question-level resampling of saved responses. Eight answers from one question are not eight independent questions. Training-seed variation is another uncertainty entirely.

Checkpoint selection, strength selection and repeated screen inspection compound this. rpE-80’s 0.1-point full-test margin above −1 is not credible evidence of non-inferiority.

## 3. Mistakes and risks

### Evaluation and decision-making

- **The screen has become training feedback.** Selecting checkpoints, mixtures, strengths and new methods against it makes its best results optimistic. Full testing selected candidates does not erase that history.
- **“Above curve” is being confused with success.** Its baseline, aggregation and fitted population differ from the actual objective. Report the two objective coordinates first; retire curve residuals as the decision criterion.
- **Small changes are overinterpreted.** A single-seed +0.5 accuracy improvement, especially per part, is not a mechanism discovery.
- **Too many moving parts obscure attribution.** Freshness, question difficulty, counting coverage, initial weights and pair supply have changed together.

### Data

- **Correct-answer filtering removes precisely the difficult population needing protection.** Questions with no correct rollout provide no preference pair; easy questions dominate shortening. Report rejected-question coverage, not merely accepted-pair counts.
- **Benchmark-shaped is not benchmark-equivalent.** Hard GSM8K-derived questions, KodCode and synthetic instruction/tool questions have different failure distributions. Matching prompt length or original solve rate does not establish transfer.
- **Contamination is not resolved by naming an “own-data” mode.** Earlier stopping diagnostics explicitly trained on benchmark questions. Those results cannot establish generalization; descendants require provenance tracing. Conversely, using GSM8K’s test split is not automatically leakage into this different evaluation.
- **Clean provenance remains unfinished.** `build_pool.py` documents a competitor-associated default math mixture; `repair_pairs.py` documents the provenance concern around its similar-length correctness option. Removing one source is not enough. Freeze an independently specified source manifest and completion hashes.
- The corruption filters are scattered across orchestrator paths. Make validation a single mandatory data-entry operation.

### Pair rules and DPO

- **“Be right” is not length-neutral.** Shortest-correct versus shortest-wrong can prefer a correct answer four times longer—or four times shorter. That is legitimate correctness training, but it explains movement in length.
- **6/8 is not a safety certificate.** A question with true success probability 0.5 passes this gate **14.5%** of the time; at 0.6 it passes **31.5%**. For 6/8 observations, the exact two-sided 95% interval has a lower endpoint around **0.35**.
- **Shortest-right versus longest-right selects extremes.** It exploits observed length variation without proving that the discarded reasoning was unnecessary. It also risks compressing the final answer rather than thinking.
- **IFEval needs special care.** Its output-length constraints directly oppose indiscriminate total-token minimization. The recorded shift toward too-short final responses is a concrete failure mechanism. Restoring rpE-80’s IFEval deficit alone would improve its aggregate accuracy by **0.65 points**.
- Sequence-summed DPO is not a token-budget constraint. Pair selection carries much of the length objective; fixed beta and learning rate do not make heterogeneous length gaps equivalent.

### Concrete code issues

1. **Incorrect zero-thinking fallback.** `thinking_len()` returns total output length when `thinking_ids` is an empty list. I reproduced an empty-thinking response being assigned 100 thinking tokens. Distinguish missing information from a valid zero.
2. **Incomplete reference-cache fingerprint.** `dpo.py` hashes short token prefixes and sequence lengths, not full chosen/rejected contents and reference identity. Different pairs can therefore retain the same fingerprint. The previous cache mismatch causing a loss jump from approximately 0.6 to 8 was not made impossible by this fix.
3. **Quota shortages need hard handling.** At batch 16, 144 usable pairs provide only nine one-epoch updates, not ten. The online loop must not assume its requested checkpoint exists after accepting fewer pairs. Refill deterministically or fail explicitly.
4. **Separate policy identities.** Log and hash rollout policy, initialization, reference policy and evaluation merge independently. “Current answers” alone does not establish which of these changed.
5. **Online sampling is expensive relative to learning.** Queued od2 requests **90,400 completions for at most 1,600 training pairs**. That may be justified for filtering, but it demands measured pair yield and gradient coverage—not more rounds by default.

### COPD

The teacher’s light-minus-heavy log-probability difference is an **effort-style preference**, not evidence that a token is dispensable. Your own demonstration found useful checking pushed down more strongly than needless checking.

Adding correctness can rescue accuracy, but also creates an implicit shortening pressure when wrong responses are longer. The logged analysis explicitly observes this. Neither that nor the early-stop penalty establishes selective removal of redundant reasoning.

**Do not spend the next scarce GPU days expanding COPD variants.** The mechanism currently lacks the evidence needed for this objective.

## 4. Answers to the four open questions

**1. Can od1/od2 reach the objective? What should change?**  
Plausibly, but it is not the central expectation from these data. Freshness plus shortening pressure addresses a real failure mode; freshness alone has not shifted the full frontier convincingly. od1 is a diagnostic continuation, not the requested clean recipe. Start the actual candidate from original Qwen, preserve one objective throughout, fix the implementation issues, and protect IFEval final-answer requirements.

**2. Why the same curve—model, data or pair rules?**  
You have demonstrated a recurring empirical trade-off under related data and objectives. You have not separated those causes. Pair rules that reward short successful traces and correctness repair that restores longer traces naturally produce such a curve. There is no evidence that this quadratic is a 0.6B capacity law.

**3. What could mislead you?**  
Adaptive screen selection, model-dependent screen/full gaps, question clustering, seed variation, historical contamination, completion truncation, source mismatch, and final-answer shortening masquerading as shorter thinking.

**4. Most informative next experiment?**  
A controlled **fresh-versus-frozen rollout comparison from original Qwen**, with identical mixed pair rules, reference policy, training dose and two seeds. Not another continuation with several simultaneous changes.

## 5. What to do next: one experiment, ranked first—and stop there initially

### Experiment 1: Does refreshing rollouts improve a clean mixed-DPO recipe?

**Arms**

- **Fresh:** each round samples the current policy.
- **Frozen:** every round samples original Qwen.
- Both initialize from original Qwen and use an immutable original-Qwen DPO reference.
- Two complete seeds, **0 and 1**, controlling sampling, adapter initialization and training order.

This isolates refresh while giving the proposed clean recipe an honest test.

**Data**

Freeze a deduplicated, approved-source manifest before training. Exclude the entire official evaluation by normalized text and source-family checks; add code-level duplicate checks. Use only model-generated completions—no answer rewriting or prompt-budget instructions.

Three disjoint rounds, each initially:

- 128 math questions;
- 128 code;
- 128 IFEval;
- 61 simple, 30 multiple, 37 irrelevance.

Generate eight answers per question using official sampling settings and token caps. Allow at most **256 preordered reserve questions per round**, shared across arms where applicable. If quotas remain unattainable, record a feasibility failure; do not silently change the mixture.

**Pairs**

Exactly **256 pairs per round**, quotas:

- math 64, code 64, IFEval 64;
- simple 30, multiple 16, irrelevance 18.

Within each part, half:

- shortest valid correct versus shortest finished wrong;
- shortest valid correct versus longest valid correct, requiring **at least 6/8 valid correct** and a **1.25 length ratio**.

For IFEval, shortening comparisons use **thinking tokens only**, with every required final-response constraint verified. Fix zero-thinking handling. Maximum one pair of each kind per question; no duplicate pairs.

Keep the existing gate for this experiment: changing both freshness and the gate would spoil attribution.

**Training**

- LoRA rank 16, alpha 32, dropout 0.
- DPO beta 0.1; no auxiliary SFT loss.
- AdamW, learning rate **1e−5**, betas 0.9/0.999, weight decay 0, gradient clipping 1.
- Effective batch 16, microbatch 1, BF16 and gradient checkpointing.
- Two epochs over each round’s 256 pairs: **32 updates per round, 96 total**.
- Preserve adapter and optimizer state across rounds. No reference reset, merging-strength sweep or corrective stage.
- Preserve complete responses; never silently truncate training pairs.

**Selection and evaluation**

Use one newly reserved **512-question, four-sample development set**, with the same benchmark allocation proportions. Never use the existing screen for selection.

Evaluate only the predetermined final checkpoint. Select the arm with greatest mean development savings across both seeds among arms with mean accuracy change at least −0.5. If neither qualifies, label the experiment unsuccessful rather than picking a lucky checkpoint.

Run the selected arm’s **two seed checkpoints** on the full test. Bootstrap paired question-level statistics within parts, recomputing BFCL pooling and geometric token savings each time. Report seeds separately; do not average away a failed seed.

**Expected result and decision**

My planning range—not a confidence interval—is approximately **35–50% full savings and −2 to 0 accuracy points**. Success beyond 50%/−1 is possible, but not the central forecast.

- Consistent fresh-arm gains across seeds support refreshing; uncertainty overlapping no improvement means **inconclusive**, not “freshness proven.”
- Claim objective attainment only when the full point estimates satisfy both strict inequalities.
- Claim robust attainment only with uncertainty bounds supporting both constraints and replication across seeds.
- If neither arm moves the frontier, stop adding rounds: investigate pair coverage and the IFEval failure mechanism instead.

Shared first-round rollouts bound training generation at roughly **62,000 completions**, plus development scoring and two full evaluations. That is materially more informative than spending comparable compute on od2’s 90,400 rollout completions alone, and is a reasonable bounded program for two L4s over the next few days.