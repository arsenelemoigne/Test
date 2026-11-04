## Recommendation

I would use **a modestly compressed SFT model as the starting point, then conservative, on-policy GRPO with a correctness-gated length reward, group-mean centering without standard-deviation division, and one optimizer update per rollout batch**.

Three cautions matter:

1. **Keep all-correct groups.** They contain much of the useful shortening signal.
2. **Do not reward shorter wrong answers.** Otherwise the cheapest failure can become the preferred behavior.
3. **A scalar correctness-plus-length reward cannot guarantee less than one point of accuracy loss.** Accuracy-based checkpoint acceptance is part of the recipe, not an optional final check.

Below is the configuration I would actually start with. It is a defensible first run, not a claim that untested hyperparameters are optimal.

# 1. Exact first-run recipe

## A. Starting model and data

### Starting point

Start from **Qwen3-0.6B plus a modest compact-reasoning SFT adapter**, not an aggressively shortened model.

Required SFT acceptance criteria on held-out development questions:

- **20–35% total-token savings** against the original model.
- Aggregate accuracy point estimate **no worse than −0.3 points**.
- No part showing a repeated loss exceeding **2 points**.
- At least **99.5% well-formed, nonempty thinking blocks**.
- No evidence of answer-checker exploitation.

These are screening criteria, not statistical proof of equivalence. If no SFT checkpoint passes, **repair the SFT stage first**; I would not silently substitute the original model and retain the same performance forecast.

Continue training the adapter. Freeze a copy of the complete SFT-start policy as the KL reference.

### Training pool

Use **4,032 distinct approved training questions**:

| Type | Questions | Sampling share |
|---|---:|---:|
| SVAMP-like | 1,008 | 25.00% |
| MBPP-like | 1,008 | 25.00% |
| IFEval-like | 1,008 | 25.00% |
| BFCL simple | 480 | 11.90% |
| BFCL multiple | 240 | 5.95% |
| BFCL irrelevance | 288 | 7.14% |

This matches the scoring weights, including BFCL’s internal proportions.

Within each type:

- Stratify approximately by source and original four-draw success count.
- **Retain zero-of-four questions.** Do not declare them unsolvable.
- Deduplicate questions and near-duplicate/template families before splitting.
- Do not oversample irrelevance because it is abundant in the raw data.
- Require working, independently trustworthy training checkers.

**Data-access prerequisite:** the metadata alone does not provide these prompts. The supplied text sample has only **2,386 distinct questions**. Obtain the approved source questions and their checkers, or construct new reference-checked training questions. Do not pretend that the metadata provides 17,870 usable text examples.

### Validation sets

Separate from SFT and RL training:

- **Development:** 336 questions: 84 math, 84 code, 84 IFEval, 40 simple, 20 multiple, 24 irrelevance.
- **Frozen lockbox:** preferably 1,008 questions with the same proportions as the training-pool table, scaled down by four.
- Split by question/template/schema family, not completion.
- Development: four draws per question.
- Final lockbox: eight draws per question.
- All validation uses the **exact test sampling and token limits**.

## B. Rollouts and batching

| Setting | Value |
|---|---|
| Group size | **8 completions per question** |
| Questions per optimizer step | **32** |
| Completions per step | **256** |
| Steps | **252 maximum** |
| Total training completions | **64,512** |
| Pool traversal | Two shuffled passes, without dynamic replacement |
| Policy synchronization | After every optimizer step; no stale-policy rollout queue |
| Training temperature | **0.6** |
| Training top-p | **1.0** |
| Training top-k | **Disabled** |
| MBPP rollout limit | **8,192 new tokens** |
| All other rollout limits | **4,096 new tokens** |
| Evaluation sampling | Temperature 0.6, top-p 0.95, top-k 20 |
| Evaluation limits | Unchanged from the test |

Each step has eight math, eight code, eight IFEval, and eight tool questions. Across a shuffled **21-step cycle**, the 168 tool questions comprise:

- 80 simple;
- 40 multiple;
- 48 irrelevance.

### Why disable training top-p/top-k?

This is a deliberate small training/evaluation difference. It makes the behavior policy a full-support categorical distribution and removes a common custom-trainer bug: using unfiltered model log-probabilities as though they were the probabilities of top-p/top-k sampling.

For training, define

\[
\pi_\theta(a\mid h)=\operatorname{softmax}(z_\theta(h)/0.6)_a.
\]

Use **these same temperature-adjusted probabilities** for rollout log-probabilities, policy ratios, and KL calculations.

Do not generate at temperature 0.6 and compute training ratios from temperature-1 probabilities.

Keeping the original rollout limits avoids teaching the model to fail under an artificial training deadline. The reward—not an external cap—must induce earlier completion.

## C. Reward: exact formula

Let:

- \(L\): all generated tokens, including thinking, answer, and generated special tokens;
- \(V\in\{0,1\}\): the training checker’s correctness verdict;
- \(F\in\{0,1\}\): valid reasoning/output format;
- \(C\in\{0,1\}\): completion was cut at the token limit.

Use fixed per-type normalizers:

| Type | \(B_t\) |
|---|---:|
| SVAMP | **512** |
| MBPP | **768** |
| IFEval | **416** |
| BFCL simple | **160** |
| BFCL multiple | **192** |
| BFCL irrelevance | **192** |

The reward is

\[
\boxed{
R=V\,F\,(1-C)
\left(1-0.30\frac{L}{L+B_t}\right)
}
\]

Thus:

- Correct, well-formed, complete answer: reward between **0.70 and 1.00**.
- Wrong answer: **0**.
- Cut answer: **0**, regardless of partial correctness.
- Empty/malformed thinking: **0**, even if the final answer is correct.
- No separate length penalty on wrong answers.
- No format bonus that an incorrect answer can collect.
- No partial-credit reward in the first run.

### Exact format gate

A completion passes \(F\) if:

1. It has the expected single thinking section and closing delimiter, accounting for an opening delimiter already supplied by the chat template.
2. Thinking contains at least **eight non-whitespace content tokens**, excluding delimiters.
3. There is a nonempty final answer.
4. It does not reopen or nest thinking sections.
5. It finishes normally rather than being cut.

Eight tokens is a **collapse guard**, not a claim that eight tokens constitute useful reasoning. Audit actual reasoning separately. Do not give increasing reward for passing this threshold; that would invite padding.

Check correctness on the final answer through the intended parser, not by searching the thinking for a correct number or call.

### Why this length function?

The normalizers are close to the original model’s **median total length among correct training responses**:

| Type | Measured correct median |
|---|---:|
| SVAMP | 487 |
| MBPP | 774 |
| IFEval | 411 |
| BFCL simple | 151 |
| BFCL multiple | 171 |
| BFCL irrelevance | 187 |

The bounded function:

- prevents extremely long responses from producing arbitrarily large penalties;
- still differentiates lengths above \(B_t\), unlike a clipped linear penalty;
- penalizes **all output tokens**, preventing migration of verbose thinking into the final answer;
- gives comparable pressure across types.

At \(L=B_t\), reward is **0.85**. At \(L=B_t/2\), it is **0.90**.

**Important limitation:** that reward improvement could theoretically compensate for a substantial accuracy loss. For example, 90% accuracy at reward 0.85 has the same expected reward as 85% accuracy at reward 0.90. Therefore, the reward alone does **not** enforce the one-point constraint. Held-out accuracy must constrain checkpoint selection.

## D. Advantages, objective, and optimizer

### Advantages

For the eight completions of question \(q\):

\[
A_{qi}=R_{qi}-\frac{1}{8}\sum_{j=1}^{8}R_{qj}.
\]

Use:

- Group-mean subtraction: **yes**.
- Group-standard-deviation division: **no**.
- Batch-standard-deviation division: **no**.
- Additional advantage whitening: **no**.
- Advantage clipping: **none**.

### Loss aggregation

Use a Dr. GRPO-style **constant-denominator token sum**:

\[
\mathcal L_{\rm PG}
=
-\frac{1}{256\cdot8192}
\sum_{q,i}\sum_{j=1}^{L_{qi}}
\min\left[
\rho_{qij}A_{qi},
\operatorname{clip}(\rho_{qij},0.8,1.2)A_{qi}
\right].
\]

- Include thinking, final-answer, and termination-token actions.
- Mask prompt and padding tokens.
- Do **not** divide each response’s loss by its own length.
- Do **not** normalize by the batch’s realized total token count.
- Keep the denominator unchanged for groups with zero advantages.

Use a KL term with the **same token-sum denominator**:

\[
\mathcal L=\mathcal L_{\rm PG}+0.01\,\mathcal L_{\rm KL}.
\]

Prefer exact categorical \(D_{\mathrm{KL}}(\pi_\theta\|\pi_{\rm ref})\) at sampled histories, computed in vocabulary/sequence chunks. Both distributions use temperature 0.6. The reference remains the SFT-start policy.

### Optimizer settings

| Setting | Value |
|---|---|
| Optimizer | AdamW |
| Peak learning rate | **\(1\times10^{-5}\)** |
| Betas | **(0.9, 0.999)** |
| Epsilon | **\(10^{-8}\)** |
| Weight decay | **0** |
| Warm-up | **12 steps**, linear |
| Schedule | Cosine decay to **\(1\times10^{-6}\)** by step 252 |
| Restarts | **None** |
| Gradient norm clipping | **0.5** |
| Optimizer updates per rollout batch | **One** |
| PPO ratio clipping | **[0.8, 1.2]** |
| Entropy bonus | **0** |
| KL coefficient | **0.01**, fixed |
| Precision | BF16 model compute; FP32 reductions/log-softmax where practical |

With one optimizer update on each fresh batch, the importance ratios are initially one. **PPO clipping is not the primary safety mechanism here.** Small updates, KL, fresh rollouts, and validation are. Do not claim clipping protects against arbitrary damage from the single update.

### LoRA

- Targets: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`, in all 28 layers.
- Rank: **16**.
- Alpha: **32**.
- Dropout: **0**.
- No trainable biases.
- Embeddings and tied output head: frozen.
- Approximately **10.1 million trainable parameters**.
- Gradient checkpointing and memory-efficient attention enabled.
- Start with microbatch size **one response**, accumulating to the full 256-response batch; increase only after memory testing.

Dropout is zero partly to make policy-log-probability accounting reproducible.

### Dynamic sampling

**Off for the first run.**

- Do not drop all-correct groups.
- All-wrong groups have zero policy-gradient advantage.
- Such groups can skip policy-gradient backward work, but retain their place in the fixed denominator and sampling statistics.
- Do not repeatedly resample until a desired success pattern appears.

This preserves the intended task distribution and avoids unpredictable rollout expense.

# 2. Why these choices—and what will make or break the run

## Group size eight is a reasonable compromise

I fitted a beta-binomial model to the original four-draw correctness counts. Its eight-draw predictions are:

| Type | All correct | All wrong | Mixed correctness |
|---|---:|---:|---:|
| SVAMP | 62.7% | 4.2% | 33.1% |
| MBPP | 33.7% | 25.6% | 40.6% |
| IFEval | 39.7% | 28.8% | 31.5% |
| BFCL simple | 69.0% | 8.3% | 22.7% |
| BFCL multiple | 56.2% | 9.9% | 33.9% |
| BFCL irrelevance | 84.0% | 2.7% | 13.2% |

These are **model-based extrapolations**, not measurements of eight training draws. They also precede SFT and use the original filtered sampling.

Under the proposed mixture, approximately:

- **51.6%** all-correct groups;
- **16.4%** all-wrong;
- **32.0%** mixed.

Group size four would supply less correctness contrast; sixteen doubles rollout cost. Eight is a sensible first allocation.

## Do not standardize each group’s advantages

With the proposed reward, the measured median reward standard deviation in original all-correct four-draw groups is only:

- **0.0085–0.0158**, depending on type.

For mixed-correctness groups it is approximately:

- **0.365–0.376**.

Dividing each group by its own standard deviation would amplify tiny correct-vs-correct length differences by roughly **23–44 times relative to correctness differences**.

That is precisely the wrong default for a strict accuracy budget. Mean centering preserves the intended reward scale.

This choice is consistent with the criticism of question-dependent scaling in the *Understanding R1-Zero-Like Training* / Dr. GRPO line of work. The recommendation is not “all normalization is bad”; it is that normalization must not silently undo your chosen correctness/length trade-off.

## Keep all-correct groups

Every observed all-correct four-draw group had some length variation. Median shortest-to-longest spreads were:

- Math: **181 tokens**
- Code: **409**
- IFEval: **169**
- Simple tools: **44**
- Multiple tools: **57**
- Irrelevance: **65**

These groups are valuable shortening examples. DAPO-style dynamic sampling is useful in some settings, but dropping groups based on uniform **correctness** would discard most tool-training opportunities here.

## The three most consequential choices

1. **Reward gating and advantage scaling.**  
   Rewarding shorter wrong answers, or standardizing away the correctness/length scale, can fundamentally change what is learned.

2. **Starting checkpoint and checker integrity.**  
   GRPO cannot reliably recover missing reasoning from a damaged SFT checkpoint. Nor can it distinguish genuine correctness from exploitable checkers.

3. **Update size: learning rate, KL, and update reuse.**  
   One fresh-batch update is intentionally conservative. Multiple passes through the same rollout batch are not a free compute optimization.

The literature supports these design principles, not these exact numerical constants on Qwen3-0.6B. The constants remain hypotheses to test.

# 3. Monitoring and explicit intervention thresholds

Save checkpoints every **21 steps**. Run the development evaluation at steps **0, 42, 84, 126, 168, 210, and 252**.

Always log **raw correctness separately from shaped reward**.

## Training health dashboard

| Metric | Healthy / expected | Investigate or stop |
|---|---|---|
| Raw accuracy | Approximately stable or improving | Pause if aggregate development accuracy is down **>2 points on two screens**; inspect individual parts |
| Total-token ratio vs SFT start | By step 84, roughly **5–15% lower** is encouraging | Less than **5% improvement** by step 84, with stable accuracy, triggers a length-pressure diagnosis |
| Format validity | **≥99.5%** | Roll back if invalid/empty thinking exceeds **1%** on repeated measurements |
| Thinking near eight-token floor | Rare; explicit short tool reasoning is acceptable | More than **5% overall** near the floor triggers manual audit for padding or reasoning collapse |
| Cut rate | No higher than SFT baseline; preferably falling | Increase of **>1 absolute point**, especially in code/IFEval, triggers investigation |
| Entropy by type | Broadly stable; modest reduction is normal | Below **50% of initial entropy** early in training, together with diversity/accuracy deterioration, is a collapse warning |
| Mean KL per generated token | Early values below roughly **0.03 nats**; later below **0.08** are comfortable | **>0.1** warrants inspection; **>0.2** repeatedly plus accuracy deterioration warrants rollback |
| Post-update ratio clipping fraction | Preferably **<5%** outside [0.8, 1.2] | **>10% for three steps** suggests an oversized update |
| All-wrong groups | Establish actual SFT-start baseline | Increase of **>5 absolute points** over a rolling window, with falling correctness, is concerning |

These KL/entropy thresholds are engineering alarms, not universal laws.

### Implementation checks before committing the run

Use a small pilot batch to verify:

- Before updating, teacher-forced old/new policy ratios equal one.
- Temperature adjustment is identical in sampling and log-probabilities.
- No dropout or stale adapter is active in one path but not the other.
- Prompt/padding tokens receive no policy loss.
- Rewards and advantages remain attached to the correct question/completion.
- The denominator does not change when zero-advantage groups are skipped.
- vLLM and trainer log-probabilities agree reasonably; investigate material systematic discrepancies.
- Code execution is sandboxed and cannot inspect grading files or manipulate the checker.

## What the failures look like

### Length does not fall

Distinguish three cases:

- **Length advantages are essentially zero:** lengths have become identical, rewards are accidentally rounded, or all-correct groups are being discarded.
- **Length advantages exist but the policy hardly moves:** KL or learning rate is too restrictive.
- **Length falls only on successful questions, while failures remain long:** expected with correctness-gated rewards. Repair those failures; do not immediately introduce a reward for failing quickly.

### Accuracy collapses while reward improves

This is the scalar-objective trade-off described above. Look for:

- rapidly shorter math reasoning;
- omitted code edge cases;
- premature no-call decisions;
- IFEval answers falling below required word/sentence counts.

Reject the checkpoint even if training reward rises.

### Thinking is emptied or padded

The model may emit eight useless tokens, then answer. Minimum length alone will not prevent this.

Audit at least **20 random traces per type** at major checkpoints, including difficult questions. Check for actual arithmetic, algorithm selection, constraint planning, or argument mapping—not stock phrases.

### Reward hacking

Warning signs include:

- training-checker accuracy improving while source-held-out accuracy declines;
- code that special-cases visible assertions;
- multiple candidate answers exploiting a permissive parser;
- emitting expected call syntax without correctly matching the request;
- final answers hidden in, or parsed from, the thinking section.

Use expanded code tests and small semantic perturbations of training-style questions as a separate audit.

## Acceptance rule

Evaluate **one development-selected candidate** on the frozen lockbox.

Use paired, question-level bootstrap intervals, retaining all eight draws of each question together. Select candidates for:

- total-token savings comfortably above 50%, preferably **55–60% or more**;
- accuracy loss whose one-sided confidence bound supports **less than one point**;
- no major part-specific collapse concealed by average gains.

If the lockbox interval is too wide, the result is **inconclusive**, not successful. Repeatedly selecting checkpoints on the lockbox would invalidate its purpose.

# 4. Small tuning ladder

Do not run all of these automatically. Diagnose the first run, then change **one setting**. Compare branches from the same checkpoint, with the same reference policy and evaluation set.

| Order | Single change | When to try it | What would count as success |
|---|---|---|---|
| **1** | Length coefficient **0.30 → 0.45** | By step 84, accuracy is stable, implementation is correct, but shortening is <5% versus SFT | Clear additional shortening without a meaningful accuracy decline |
| **2** | Length coefficient **0.30 → 0.15** | Length falls well, but accuracy deteriorates while KL/update size remain modest | Most accuracy recovers while retaining useful savings |
| **3** | Peak LR **\(10^{-5}\to5\times10^{-6}\)** | Post-update ratios move too far, KL jumps, or accuracy oscillates | Smoother learning and better held-out accuracy at comparable shortening |
| **4** | KL coefficient **0.01 → 0.003** | Length gradients exist, accuracy is stable, and both KL and policy movement remain extremely small | More shortening without unstable drift |
| **5** | Training temperature **0.6 → 0.8**, still unfiltered | Groups lack useful diversity and difficult questions rarely produce a correct candidate | More mixed/useful groups and better accuracy progress without excessive cuts |

For variation 5, recompute log-probabilities and reference KL at **temperature 0.8**. Evaluation remains unchanged.

I would screen a branch over **42–84 steps**, not immediately repeat all 252. Allocate the budget to at most **one or two justified branches**, not five full runs.

I would not initially tune entropy bonuses, restart schedules, multiple PPO epochs, and LoRA rank simultaneously. They add degrees of freedom before the main mechanism is understood.

# 5. Compute and expected outcome

## Compute

The main run generates **64,512 completions**. Under the original model’s training distribution and the proposed mixture, that is approximately **56.5 million generated tokens**. The SFT start should reduce this, although unfiltered sampling may alter the tail.

A reasonable planning envelope is:

- Training rollouts: **25–45 GPU-hours**
- GRPO/reference forward-backward work: **20–35 GPU-hours**
- Development and final validation: **15–25 GPU-hours**
- SFT preparation, pilots, and one short branch: remaining budget

That is approximately **60–105 GPU-hours for the main RL/evaluation workload**, before substantial data preparation.

Run vLLM on one L4 and optimization on the other, synchronizing each batch. Measure the first 21 steps: synchronization means two GPUs do not automatically give perfect utilization. If throughput is poor, stop at an earlier saved checkpoint rather than lowering rollout limits and changing the task.

## Expected result after the first good run

These estimates are **against the original model**, including the SFT warm start. They are not incremental gains from RL alone.

| Test part | Total-token savings | Accuracy change, points | Confidence |
|---|---:|---:|---|
| SVAMP | **55–72%** | **−0.7 to +0.5** | Medium on savings; low–medium on accuracy |
| MBPP+ | **55–72%** | **−1.5 to +2.0** | Medium on savings; low on accuracy |
| IFEval | **25–42%** | **−1.0 to +1.5** | Low–medium |
| BFCL simple | **45–65%** | **−1.2 to +0.7** | Medium on savings; low–medium on accuracy |
| BFCL multiple | **40–60%** | **−1.8 to +0.7** | Low–medium |
| BFCL irrelevance | **60–80%** | **−0.5 to +2.0** | Medium on savings; low on transfer accuracy |

Under the test’s scoring, these correspond approximately to:

- **47–65% overall token savings**
- **−1.1 to +1.3 points overall accuracy change**

The token calculation uses approximate BFCL baseline total lengths because the full-test profile reports thinking means but not mean total lengths.

My central scenario is approximately **57% savings with essentially unchanged aggregate accuracy**. That is an engineering target, not a measured expectation with a calibrated probability.

Relative to an SFT start saving 25–35%, reaching that central scenario requires roughly **34–43% additional reduction in the SFT model’s remaining tokens**. That is substantial: this recipe is not guaranteed to achieve it in 252 updates.

**Bottom line:** the run is most likely to succeed if it learns from the plentiful all-correct length contrasts **without allowing those contrasts to outweigh correctness**. Preserve the reward scale, keep rollout accounting exact, and make non-inferiority—not shaped reward—the checkpoint-selection criterion.