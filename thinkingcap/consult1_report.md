# Recommendation

**I would start with verifier-filtered, compact-reasoning fine-tuning, then add a small amount of correctness-constrained preference/RL training.** I would keep learned early termination as a bounded, secondary experiment.

The target looks **plausible, but not established by these data**. There is substantial redundant reasoning, and expensive failures create additional opportunity. However, simply selecting existing short answers, eliminating cutoffs, or shortening every thinking block by half is insufficient.

My central engineering target would be **about 60% fewer total tokens**, leaving a margin over 50%, while aiming for **no accuracy loss**. The uncertainty around the one-point accuracy requirement is considerably greater than the uncertainty around token savings.

All projections below are estimates—not model experiments. I ran no forward passes.

---

# 1. What the data says

## 1.1 The scoring changes the optimization priorities

Let \(r_b\) be the new/original **mean total generated-token ratio** for benchmark \(b\). The token score is

\[
S=1-(r_{\rm SVAMP}r_{\rm MBPP}r_{\rm IFEval}r_{\rm BFCL})^{1/4}.
\]

BFCL must first pool its three parts, with **400 simple, 200 multiple, and 240 irrelevance questions**. Accuracy changes are averaged across the four benchmarks, with BFCL likewise pooled.

The reported baseline accuracy is approximately:

- SVAMP: **88.7%**
- MBPP+: **52.4%**
- IFEval: **58.7%**
- Pooled BFCL: **81.09%**
- Equal-benchmark average: **70.22%**

Thus the accuracy requirement is an average **above approximately 69.22%**, not a one-point allowance on each part.

**Do not optimize raw token totals alone.** Math and code account for **79.1% of full-test thinking tokens**, but BFCL still receives one-quarter of each aggregate score.

For example, saving 60% on math and code and 30% on IFEval still requires **more than approximately 44% savings on pooled BFCL** to cross 50% overall.

## 1.2 Token accounting

I analyzed **71,480 training completions from 17,870 questions**, with four completions per question.

| Part | Test accuracy | Test mean thinking tokens | Test cut rate | Training mean thinking / total tokens | Training token share spent on wrong answers, including cuts |
|---|---:|---:|---:|---:|---:|
| SVAMP | 88.7% | 677 | 1.7% | 807 / 827 | **37.1%** |
| MBPP+ | 52.4% | 2,079 | 7.8% | 1,745 / 1,802 | **67.0%** |
| IFEval | 58.7% | 433 | 3.6% | 422 / 670 | **55.7%** |
| BFCL simple | 82.2% | 171 | 0% | 158 / 180 | **24.2%** |
| BFCL multiple | 76.1% | 192 | 0% | 190 / 210 | **33.8%** |
| BFCL irrelevance | 83.4% | 186 | 0% | 206 / 233 | **14.3%** |

The provided full-test table does **not** contain mean total-output lengths. Consequently, absolute test token-budget estimates below use **test thinking means plus training mean non-thinking lengths**. That approximation matters especially for IFEval; it should be replaced with exact means when the team evaluates candidates.

### SVAMP: preserve calculation; remove narration and repeated confirmation

- Thinking is **97.6%** of training output tokens.
- The longest approximately 10% of training responses consume **37.7%** of tokens and have only **32.2% accuracy**.
- Correct training responses average **613 thinking tokens**; wrong, non-cut responses average **1,227**.
- Even questions answered correctly on all four attempts average **545 thinking tokens**.

The text shows substantial repetition after reaching a result. One correct height problem derives 56, derives it again with equations, confirms it, and rehearses the final-answer format. Among 173 sampled correct math traces containing the final numeric answer in the thinking, the number appeared **4.1 times on average**.

That is evidence of compressibility, **not** a safe rule to stop at the first appearance of a number.

Most importantly, thinking OFF loses **72.2 accuracy points** in the small diagnostic study. This model needs its arithmetic scratchpad. I would train **equations, variable bindings, and one useful check**, not answer-only behavior.

### MBPP+: the largest tail opportunity, but also genuine difficult reasoning

- Thinking is **96.8%** of training output.
- The longest 10% consume **39.8%** of tokens, with **10.7% accuracy**.
- Training cuts alone consume **22.2%** of code tokens.
- Questions with zero successes in four attempts are **31.7% of questions but consume 50.6% of tokens**.
- Correct responses nevertheless average **1,022 thinking tokens**.

The long traces contain both useful edge-case analysis and pathological reconsideration. A cut example repeatedly becomes confused about removing consecutive repeated characters. A successful permutation solution repeatedly debates a three-line implementation.

In the outcome-stratified text sample, code cuts had approximately **23.8% repeated eight-word windows**, versus **5.2%** for correct code traces. Approximately **98% of sampled code cuts never closed the thinking block**.

The opportunity is therefore twofold:

1. Compact the ordinary successful plan.
2. Teach the model to stop re-litigating an implementation and emit its best completed solution.

But hidden tests make aggressive removal of edge-case reasoning dangerous. Thinking OFF loses **9.1 points**.

### IFEval: this is partly an answer-generation problem

- Thinking is only **62.9%** of training output.
- Non-cut responses average **156 non-thinking tokens**.
- Cut responses average **2,888 non-thinking tokens**.
- Cuts consume **20.6%** of all training IFEval tokens.

Crucially, **80.2% of sampled IFEval cuts had already closed `</think>`**; approximately **71% of their tokens were answer-side**. One example simply repeats quotation marks until the limit.

A thinking-only intervention will miss much of this waste.

Other failures are not caused by insufficient deliberation: a long trace explicitly checks that an answer contains no commas, then emits an answer full of commas. Train concise **constraint identification plus executable-style checks**, and train clean completion/EOS behavior.

Do not shorten required content indiscriminately: minimum word counts, exact sentence counts, and formatting constraints impose real answer-length floors. Thinking OFF loses **5.0 points** in the diagnostic.

### BFCL simple and multiple: routine reasoning is verbose, but selection still matters

These parts have almost no cutoff problem. Their savings must come from the **body of ordinary successful reasoning**, not just tails.

Typical traces repeatedly:

- restate the request;
- restate the function description;
- explain each parameter;
- repeat the intended call.

A compact equivalent could retain: **selected capability → required arguments and types → call**.

However, thinking OFF loses **11.6 points on simple** and **17.2 on multiple**. Tool choice and argument mapping cannot simply be removed.

Also, an untrained cap is particularly unsafe here. A **64-thinking-token cap would affect essentially every training response**, including virtually every correct one. The model must learn to reach the decision earlier.

### BFCL irrelevance: attractive savings, but a distribution-shift trap

The OFF diagnostic improves accuracy by **6.2 points**, suggesting that extended reasoning sometimes rationalizes an inappropriate call. A short, explicit applicability check is promising.

Nevertheless:

- Training accuracy is **91.8%**, versus **83.4% on the test**.
- Irrelevance makes up **56.0% of training tool questions**, versus **28.6% of test tool questions**.

Naively training on the available tool distribution risks an overly conservative “no call” policy. Include hard positive matches, partial matches, missing-argument cases, and genuinely irrelevant functions.

## 1.3 Existing short trajectories are useful—but insufficient

Using metadata, I calculated the shortest correct response among the four original attempts, conditional on at least one being correct.

| Part | Questions with at least one correct attempt | Shortest-correct savings versus the question’s mean correct response |
|---|---:|---:|
| SVAMP | 92.4% | 25.0% |
| MBPP+ | 68.3% | 24.6% |
| IFEval | 66.8% | 23.8% |
| BFCL simple | 88.9% | 16.9% |
| BFCL multiple | 86.0% | 16.6% |
| BFCL irrelevance | 96.1% | 21.3% |

Therefore, **“fine-tune on the shortest existing correct response” is a good baseline, not a sufficient plan for 50% savings**.

Two other useful limits:

- Magically deleting every training cut response’s entire token cost yields only about **15% aggregate savings** under an evaluation-weighted proxy.
- Halving all thinking lengths, preserving estimated answer lengths, yields only about **44% overall savings**.

The successful intervention must learn **substantially denser reasoning on normal successful cases**.

These are descriptive comparisons, not causal intervention results. Long responses can be long because the problem is hard.

## 1.4 Distribution and data-access limitations

Important train/test shifts:

- **Math:** training is harder and longer—82.9% versus 88.7% accuracy; 807 versus 677 thinking tokens. Training savings may overstate test savings.
- **Code:** test is harder and more heavy-tailed—52.4% versus 55.3% accuracy; 2,079 versus 1,745 thinking tokens; 7.8% versus 4.9% cuts.
- **Irrelevance:** the 8.4-point accuracy gap is substantial.
- “Always/sometimes/never right” proportions cannot be compared directly: training has **four draws**, test **eight**. Zero of four is not proof a question is unsolvable.

The full-text file is also much smaller than the metadata:

- **2,806 records**
- **2,386 distinct questions**
- Correct full traces for only **1,063 distinct questions**

It is deliberately outcome-stratified. Its cut prevalence is not representative.

**The 71,480 metadata rows are not 71,480 usable training trajectories.** Additional training requires access to approved source questions/validators or generation of new, independently verifiable tasks. I would not assume those missing assets exist.

---

# 2. Strategies and quantitative potential

### How to read the estimates

Each table entry is:

**total-token savings % / accuracy change in percentage points**

Positive accuracy changes are improvements. These are engineering scenario ranges, **not confidence intervals**. Savings and accuracy endpoints are not independently selectable.

Aggregate estimates use the exact benchmark weighting and approximate BFCL baseline total lengths described above.

## Strategy A — Verified compact-reasoning distillation

### Idea

Train the model to perform the same necessary reasoning in a denser form:

- **Math:** variables, equations, computation, one relevant check.
- **Code:** contract, algorithm, important edge cases, then implementation.
- **IFEval:** constraint checklist, economical construction, clean termination.
- **Tools:** capability match, typed argument mapping, explicit no-match reason where appropriate.

Use supervised fine-tuning on **verified compact trajectories**, mixed with original correct trajectories.

This is related to verified self-training/distillation approaches such as STaR, but the key intervention here is the construction and validation of shorter reasoning.

### Implementation

1. Begin with the available correct traces; split by question before editing.
2. Construct compact alternatives by removing restatement and redundant verification while retaining actual computation and decisions.
3. Bootstrap with fresh completions from the compact model under the **unchanged deployment template**.
4. Accept targets only when the final answer passes a trusted checker.
5. Keep approximately **20–30% uncompressed correct replay**, including difficult cases.

For new data, use tasks with independently known answers: procedural arithmetic, code tasks with reference implementations and withheld tests, constraint-generated instructions, and synthetic tool schemas with reference calls.

For code, one visible assertion is not an adequate validator. If the original training harness is unavailable, use independently constructed tests—not tests authored only by the same model that wrote the candidate solution.

Use all-linear **rank-16 LoRA**, initially one epoch, then selective second-epoch training. With the supplied architecture this is approximately **10.1 million trainable adapter parameters**. Merge the adapter for deployment; preserve tied embeddings and the base tokenizer.

### Potential

| Part | Savings / accuracy change |
|---|---:|
| SVAMP | **55–75% / −0.7 to +0.6** |
| MBPP+ | **50–72% / −1.5 to +2.5** |
| IFEval | **25–45% / −1.2 to +1.5** |
| BFCL simple | **45–65% / −1.5 to +0.8** |
| BFCL multiple | **40–60% / −2.0 to +0.8** |
| BFCL irrelevance | **60–80% / −0.7 to +2.5** |
| **Overall** | **46–67% / −1.2 to +1.5** |

**Confidence:** medium in substantial token savings; low-to-medium in satisfying the accuracy bound. Accuracy retention is least certain on code and multiple-tool selection; irrelevance has the greatest distribution-shift concern.

**Compute:** approximately **18–32 GPU-hours**, including a modest data-generation round, several short LoRA runs, and screening. Shared final validation is additional.

**Main failure:** compressing away genuine computation or learning the formatting of reasoning without its substance. Training only on easy successes would worsen this.

**My ranking: first.**

---

## Strategy B — Correctness-constrained length optimization

### Idea

Optimize the model on its own sampled completions, using verified correctness and length rather than prescribing every short trajectory.

This can use GRPO-style outcome optimization. If the RL implementation is not ready quickly, use iterative **DPO-style preference training**:

- verified correct over incorrect;
- shorter correct over longer correct;
- no preference for shorter incorrect over longer correct.

This can discover concise trajectories that an editor misses and adapt the stopping behavior to problem difficulty.

### Implementation

Sample groups of **4–8 completions** on fresh, approved training questions. Use a bounded length penalty and a reference-model/KL constraint. For example,

\[
R=V-\lambda\min(L/B,1),
\]

where \(V\) is verified correctness and \(B\) normalizes length across task families.

A small \(\lambda\) does **not** guarantee accuracy preservation in expectation. Enforce accuracy constraints through held-out evaluation and checkpoint selection.

Additional safeguards:

- Do not turn all-wrong groups into a race to fail quickly; suppress their length-only update or send them to a repair curriculum.
- Balance benchmark families rather than raw example counts.
- Protect constraint-mandated answer lengths.
- Include nonempty, useful reasoning in accepted training targets.
- Mix original correct replay to prevent drift.

### Potential

| Part | Savings / accuracy change |
|---|---:|
| SVAMP | **50–75% / −1.0 to +1.0** |
| MBPP+ | **50–75% / −1.5 to +3.0** |
| IFEval | **20–45% / −1.5 to +2.0** |
| BFCL simple | **40–65% / −2.0 to +1.0** |
| BFCL multiple | **35–60% / −3.0 to +1.0** |
| BFCL irrelevance | **55–80% / −1.0 to +3.0** |
| **Overall** | **42–68% / −1.5 to +1.9** |

**Confidence:** low-to-medium on savings, low on sub-point accuracy retention when started directly from the original model. Confidence improves after Strategy A.

**Compute:** approximately **24–40 GPU-hours** for a small rollout/optimization sweep with working validators. An offline preference version is cheaper. I would not build a complicated distributed RL system for this project.

**Main failure:** exploiting imperfect verifiers, sacrificing difficult questions for shorter outputs, or unstable updates on a small model. Code checker quality is particularly important.

**My ranking: second, preferably after A.**

---

## Strategy C — Learned, adaptive termination

### Idea

Teach the model **when it has enough reasoning**, rather than imposing a fixed budget.

On training questions, take existing reasoning prefixes at sentence boundaries, close the thinking block, and sample answers. Find prefixes where the answer remains reliably correct.

Use those counterfactual results to train:

- a small readiness head that adjusts the `</think>` logit from the current hidden state; and
- optionally a separate answer-completion signal for EOS, particularly for IFEval loops.

At deployment, these are model-internal learned logits under the same sampling settings—not a prompt instruction, external cutoff, or forced early closure. No test-part label is supplied to the model.

This is a genuinely different intervention: a small architectural stopping mechanism rather than primarily changing the reasoning style.

### Implementation

- Test several prefixes per question, with repeated continuations to reduce lucky-stop labels.
- Use short-prefix failure followed by longer-prefix success as especially useful negative/positive evidence.
- Do not label an arbitrary prefix “ready” merely because all continuations fail.
- Train the readiness head and a small adapter; distill into ordinary token prediction if the head provides no incremental value.

A modest 1,024→128→1 head is negligible relative to 0.6B parameters.

The offline prefix intervention is for creating training labels only. The submitted model must still generate meaningful reasoning and terminate it itself.

### Potential

| Part | Savings / accuracy change |
|---|---:|
| SVAMP | **40–65% / −2.0 to +0.5** |
| MBPP+ | **45–70% / −3.0 to +2.0** |
| IFEval | **10–30% / −2.0 to +1.0** |
| BFCL simple | **25–50% / −3.0 to +0.5** |
| BFCL multiple | **20–45% / −4.0 to +0.5** |
| BFCL irrelevance | **45–70% / −1.0 to +2.0** |
| **Overall** | **32–57% / −2.4 to +1.1** |

**Confidence:** low. Its optimistic region could meet the target, but I would not make it the primary bet. Tool selection is especially vulnerable to premature closure.

**Compute:** approximately **12–24 GPU-hours**, mostly counterfactual continuations and validation rather than head training.

**Main failure:** the model has not yet computed the answer when the head decides to stop. Readiness can also be poorly calibrated off distribution.

**My ranking: third, with a strict time limit.**

---

## Combining the strategies

**A + B is the main combination.** A supplies a reliable compact reasoning style; B adjusts the accuracy–length frontier on fresh rollouts.

Add C only if it improves that frontier. Its savings overlap strongly with A and B; **do not multiply their independent savings estimates**.

A plausible combined envelope is:

| Part | Combined total-token savings |
|---|---:|
| SVAMP | **60–78%** |
| MBPP+ | **60–78%** |
| IFEval | **30–50%** |
| BFCL simple | **50–70%** |
| BFCL multiple | **45–65%** |
| BFCL irrelevance | **65–85%** |
| **Overall** | **52–72%** |

The associated broad accuracy-change envelope is approximately **−1.2 to +1.9 points**. It includes failure of the accuracy requirement; static analysis cannot justify a tighter guarantee.

A concrete central budget is:

| Part | Target mean thinking tokens |
|---|---:|
| SVAMP | 220 |
| MBPP+ | 650 |
| IFEval | 170 |
| BFCL simple | 65 |
| BFCL multiple | 85 |
| BFCL irrelevance | 45 |

Preserving training-estimated answer lengths, these budgets imply approximately **58.4% overall savings**. They are **mean training objectives, not hard per-question caps**. Difficult questions must retain the ability to think longer.

I would not spend this project on layer pruning, quantization, speculative decoding, or a new vocabulary: those primarily change latency, memory, or accounting—not the requested number of ordinary generated tokens—and introduce unnecessary accuracy risk.

---

# 3. Concrete three-day plan

The budget is **144 GPU-hours**. I would run independent generation/training jobs on the two L4s, not distributed training.

## Before training: establish trustworthy data and measurement

### Data controls

- Split by **question, near-duplicate, source/template, and tool-schema family**, never by completion.
- Audit provenance. Source labels alone do not prove exclusion from evaluation benchmarks.
- Use only approved non-evaluation data; do not retrieve or reconstruct hidden questions.
- Request training validators/source manifests if available—not evaluation questions or hidden tests.
- Reserve held-out questions before editing or generating descendants.

Sampling quotas for the main training mixture should approximately follow the scoring:

- 25% math
- 25% code
- 25% instruction following
- 11.9% simple tools
- 6.0% multiple tools
- 7.1% irrelevance

Within these, preserve difficult cases and constraint coverage. Do not fit primarily to the outcome-stratified text sample.

### Validation

Use two levels:

1. **Development screen:** roughly 60–100 questions per part, initially four draws.
2. **Frozen lockbox:** approximately 1,000 independent questions, eight draws for the baseline and final candidate, with evaluation-aligned weighting.

The lockbox should combine source-held-out approved questions and independently generated, reference-checked questions. The supplied text sample alone is failure-enriched and cannot simply be treated as a representative test proxy.

Use the exact deployment template, temperature **0.6**, top-p **0.95**, top-k **20**, and original **4,096/8,192** limits. Count all generated tokens and every cutoff failure. No best-of-eight selection or voting.

For uncertainty, use **paired question-level bootstrap**, retaining each question’s eight completions together. Track both accuracy difference and log token ratio. One thousand questions may still be insufficient to establish a sub-point loss; report that uncertainty rather than pretending eight draws are eight independent questions.

## Schedule

| Period | GPU 1 | GPU 2 | Decision |
|---|---|---|---|
| **Day 1** | Compact-data generation/editing ~8 h; SFT ablations ~8 h; screens ~8 h | Fresh verified-data generation ~14 h; frozen baseline evaluation ~10 h | Does compact SFT beat shortest-original SFT without clear accuracy regression? |
| **Day 2** | Preference/RL branch ~16 h; screening ~8 h | Readiness branch ~8 h; targeted data repair ~8 h; combination SFT ~8 h | Does policy optimization improve the frontier? Does readiness add anything beyond SFT? |
| **Day 3** | Final combination/repair ~8 h; lockbox evaluation ~12 h; reserve ~4 h | Parallel lockbox evaluation ~12 h; reserve ~12 h | Select one candidate; quantify whether the target is supported |

These are allocations, not promises of throughput. Code-heavy generation should be timed on day one and the remaining schedule adjusted.

## First experiments

### Experiment 1: shortest-original SFT versus compact SFT

Train:

- **Control:** shortest available verified original trajectories.
- **Moderate compression:** approximately half-length reasoning.
- **Aggressive but structured compression:** the central budgets above, with original replay.

This decides whether the model can internalize denser reasoning rather than merely shortening familiar traces.

Inspect separately:

- math calculation accuracy;
- code hidden/reference-test performance;
- IFEval constraint families and answer-side loops;
- BFCL call/no-call confusion;
- empty or malformed thinking blocks.

### Experiment 2: repair the largest failure mechanisms

Before applying more length pressure:

- fix IFEval answer repetition and missing EOS;
- strengthen code contract/edge-case examples;
- rebalance BFCL applicability decisions;
- include long-but-correct examples where additional reasoning is demonstrably useful.

This may recover accuracy headroom that can subsequently pay for compression.

### Experiment 3: one small policy-optimization sweep

Use two or three conservative length-penalty strengths, not a large hyperparameter search.

If an RL trainer is not operational quickly, switch to verified preference pairs. The project should optimize the model, not spend a day building infrastructure.

### Experiment 4: bounded readiness ablation

Compare:

- compact model alone;
- compact model plus learned readiness;
- ordinary termination-token fine-tuning without the extra head.

Keep the head only if it demonstrably improves the frontier.

## Stopping rules

1. **Kill dominated candidates.** More tokens and worse accuracy is an immediate rejection.
2. **Do not increase compression on a clearly accuracy-damaged model.** Repair or restore replay first.
3. **Cap the readiness branch at about eight initial GPU-hours** unless it shows a clear incremental benefit.
4. **Stop RL after roughly 4–6 initial GPU-hours of optimization without frontier improvement**, or immediately on verifier exploitation or collapse toward empty reasoning.
5. **Do not spend the last day chasing another few percent of savings.** Freeze the best plausible candidate and measure it.
6. For a strong validation claim, require:
   - estimated savings comfortably above 50%, preferably **55–60% or more**;
   - a one-sided confidence bound supporting **less than one point of accuracy loss**;
   - no serious collapse hidden inside the benchmark average.
7. If those confidence bounds are inconclusive, expand validation with the reserve budget. If they remain inconclusive, report **“promising but not demonstrated,”** not success.

## Bottom line

The strongest evidence is not merely that failures are long. It is that **ordinary successful answers contain large amounts of removable narration**, while a few identifiable mechanisms—especially code reconsideration and IFEval answer loops—inflate the tail.

The best bet is therefore:

> **Teach compact, task-appropriate reasoning; verify the answers; then optimize length conservatively on fresh rollouts.**

I would aim for roughly **60% total-token savings**, preserve the ability to reason longer on genuinely difficult questions, and treat the **one-point accuracy condition as the principal experimental risk**.