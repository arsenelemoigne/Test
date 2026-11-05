## Executive conclusion

**Revise the draft before running it.** The new information changes three recommendations materially:

1. **Run Track A first, but repair code as well as math and irrelevance.** Code contributes **−0.875 points** to pt-mid’s overall accuracy loss; it is not a minor residual.
2. **Use four adapters for Track B initially, with batch 16 and the historical per-type training settings.** My previous six-adapter strengths and batch-8 schedule were insufficiently grounded. They should not be treated as established hyperparameters.
3. **Keep the proposed Track B pair quotas. They are feasible from the existing four-answer store.** I verified this jointly, including overlapping pair kinds, direction balancing, and per-question caps—not merely by counting each kind separately.

I distinguish below:

- **High confidence:** arithmetic, direct experimental evidence, or verified feasibility.
- **Medium:** a defensible extrapolation from the runs.
- **Low:** an engineering starting point that needs an experiment.

“Grounded” does not mean every value has been experimentally optimized. Many have not.

---

# 1. Corrections to the goal, shared settings, and comparison rules

| Draft item | Recommendation | Evidence and reason | Confidence |
|---|---|---|---|
| Full-test target: >50% saving, accuracy >−1 point | **Keep exactly.** | Official objective. Both inequalities are strict. | High |
| Screen target: ≥52%, ≥−0.5 | **Keep as an engineering acceptance margin, not a statistical guarantee.** | Screen saving was approximately 0.6 points optimistic on two models; seed variability is reported around ±0.7 accuracy points. A 52/−0.5 screen result leaves useful, but unproven, margin. | Medium |
| Screen/full corrections +0.6 saving, −0.3 accuracy | **Report them; do not mechanically correct every candidate.** | Two models are insufficient to establish a universal calibration, particularly for a new pair distribution. | High |
| Beat the withheld model by exceeding its “vs curve” | **Change. Select on actual saving and accuracy, then use Pareto dominance.** | The curve is a historical fit, not the scoring rule. A candidate can improve its curve residual while missing the target badly. The withheld approximately 47/−0.5 model is a useful baseline, not a substitute objective. | High |
| One change at a time | **Keep for causal comparisons. Describe Track B itself as a bundled candidate, not a one-factor experiment.** | Changing grouping, pairs, anchor, batch and schedule together cannot isolate pair-rule effects. | High |
| Start/reference | **Keep:** original for B; exact merged pt-mid for A. | DPO preferences must be measured relative to the appropriate starting policy. | High |
| LoRA rank 32, alpha 64, dropout 0, seven projections | **Keep.** Freeze embeddings and tied output weights. | Matches the existing family; no evidence justifies spending this budget on rank or architecture search. | Medium |
| DPO beta 0.1 | **Keep.** | Common to the disclosed D-runs. No controlled beta comparison supports changing it. | Medium |
| “No NLL because NLL did not help before” | **Delete that justification. Use track-specific settings below.** | The successful pt merges used an NLL anchor of 1.0. The supplied evidence does not identify its marginal benefit, but clearly does not establish that it was useless. | High |
| Precision: FP32 weights, FP16 autocast | **Track A:** retain this for the first repair experiment to match the D-run correctness-training family. **Track B:** use the historical BF16 path, subject to the already-running precision test. | Precision is another confound. BF16 versus FP16 can affect numerical stability; it is not an irrelevant implementation detail. | Medium |
| AdamW; clipping at 1.0 | **Keep optimizer and clip 1.0. Copy and log all omitted AdamW settings.** | Clipping is a conventional safety choice, not an experimentally established optimum here. Weight decay, epsilon and optimizer betas must not be left implicit. | Medium/low |
| Batch 16 is “the common D-run setup” | **Correct the attribution.** | D-runs used pairs/339 per step; that was not universally 16. Batch 16 is directly supported by the newly specified historical per-type runs. | High |
| Skip pairs “over 8,704 tokens” | **Define this per branch:** prompt + chosen, and prompt + rejected must each fit. Do not apply 8,704 to their sum. | In `raw_meta`, the maximum prompt-plus-completion length is **8,391**; **zero answers exceed 8,704**. Verify actual tokenized template lengths. Never silently truncate. | High |
| One screening seed; two for finalists | **Keep.** Treat two seeds as a stability check, not a reliable variance estimate. | Reported seed spread motivates replication, but two observations cannot characterize its distribution. | High |

For DPO, retain the existing sequence-level log-probability calculation. Do not quietly switch to average-token log-probabilities; that changes the objective.

For the NLL anchor, **reuse the historical implementation and record its reduction**. Weight 1.0 multiplying a token-mean NLL is not equivalent to weight 1.0 multiplying a sequence sum.

---

# 2. Track A: corrected repair experiment

## 2.1 The draft’s diagnosis omits an important failure

Using the reported per-part pt-mid accuracy changes, the contributions to overall accuracy are:

| Part | Accuracy change | Contribution to overall change |
|---|---:|---:|
| Math | −5.6 | **−1.400** |
| Code | −3.5 | **−0.875** |
| IFEval | +0.6 | +0.150 |
| Simple tools | −0.4 | −0.048 |
| Multiple tools | +1.7 | +0.101 |
| Irrelevance | −7.1 | **−0.507** |
| **Total from rounded entries** | | **−2.579** |

Thus, **code is responsible for more overall loss than irrelevance**.

Repairing math to −2 and irrelevance to +3, while leaving everything else unchanged, would yield approximately **−0.96 overall**. That is barely inside the accuracy requirement before accounting for length growth, uncertainty, or screen selection.

The hypothesis should instead be:

> Repair math, code and irrelevance while retaining the compression and protecting the other three parts.

## 2.2 Exact Track A specification

### Data and pairs

Use the merged pt-mid model at its historical strengths:

**math/code/IFEval/tools = 0.30/0.45/0.60/0.45.**

Generate four answers per question on this initial **5,400-question slice**:

| Part | Questions | Reason |
|---|---:|---|
| Math | 2,400 | Largest accuracy deficit; ample mixed-outcome supply |
| Code | 900 | Second-largest contribution to accuracy loss |
| Irrelevance | 1,500 | Need difficult disagreements in an otherwise easy pool |
| IFEval | 300 | Detect and protect instruction-following behavior |
| Simple tools | 150 | Protect valid calls against over-abstention |
| Multiple tools | 150 | Protect function selection |
| **Total** | **5,400** | **21,600 new answers** |

These are **sampling allocations, not estimated optima**. Prioritize previously mixed-outcome questions, then include both previously all-right and all-wrong questions. For irrelevance, include the existing mixed-outcome slice before filling from easy examples.

The draft’s “math + no-fit training pool ≈5,400 questions” is incorrect: the metadata contains **9,729** such questions. Here, 5,400 is an explicitly selected all-part slice.

Build **1,600 pairs**:

| Part | Total | Shortest-right > shortest-finished-wrong | Similar-length right > wrong |
|---|---:|---:|---:|
| Math | 640 | 320 | 320 |
| Code | 400 | 200 | 200 |
| Irrelevance | 240 | 120 | 120 |
| IFEval | 160 | 80 | 80 |
| Simple tools | 96 | 48 | 48 |
| Multiple tools | 64 | 32 | 32 |
| **Total** | **1,600** | **800** | **800** |

This assigns **80% to repair and 20% to maintenance**. Call the latter maintenance, not historical replay: these are also new pt-mid answers.

Pair details:

- No shorten pairs.
- No wrong-versus-wrong pairs.
- Correct chosen answers must be finished and have a nonempty, closed thinking block.
- Similar-length means  
  \[
  \max(L_c/L_r,L_r/L_c)\le1.25.
  \]
- Exactly half of the similar-length quota should have the **correct answer strictly longer**; the remainder can be shorter or equal.
- Maximum **three pairs per question**, at most **two similar-length pairs**.
- Deduplicate across kinds.
- No absolute chosen-length windows on correctness pairs.
- Initially exclude cut rejections: this experiment targets correctness rather than additional tail-shortening.

**Why these values?**

- The 50/50 mix is a controlled compromise between D7-style correctness preferences and the short-wrong guard used in the older iterative line. It is **not** established as optimal.
- Similar-length preferences have direct positive evidence: D7 stage 1 produced **+0.8 overall**, **+1.0 math**, **+1.5 IFEval**, and **+3.3 irrelevance**, although with increased tokens.
- The shortest-wrong guard prevents us from presenting only long rejected answers and calling that “correctness training.”
- The 1,600-pair size is a convenient bounded experiment with adequate task coverage, not a sample-complexity result.

**Confidence:** medium in the pair direction and task allocation; low in the exact 50/50 and 1,600 values.

### Training and checkpoints

| Parameter | Recommended value | Reason/confidence |
|---|---|---|
| Start and reference | Exact merged pt-mid | Essential; high |
| New adapter | Rank 32, alpha 64, dropout 0 | Known configuration; medium |
| DPO beta | 0.1 | Known configuration; medium |
| NLL anchor | **0 for the first repair run** | Keeps the first correction test close to D7; avoids directly imitating longer chosen answers through an added term. Whether this helps is uncertain; medium/low |
| LR | **1e−5 peak** | Half the original training LR for a short repair; conservative engineering choice, not a controlled finding; low |
| Schedule | 5-step warm-up, cosine to 0 | 5% of the maximum run; low |
| Effective batch | **16 pairs** | Known per-type training batch; medium |
| Maximum training | **100 steps, one pass over 1,600 pairs** | Arithmetic consistency; high |
| Checkpoints/screens | **20, 40, 60, 100** | Covers 20–60-step early repair and completion; medium |
| Continuation past 60 | Only if the run is still approaching a useful frontier | Avoids automatically training through degradation |

The draft’s **1,500 pairs, batch 16, 60 steps, one pass** is internally inconsistent: 60 steps expose only 960 pair presentations. It must either be called a partial pass or changed.

I recommend the 100-step maximum above, **not a requirement to complete it**.

## 2.3 What would count as success?

Use three different labels:

1. **Repair signal:** statistically credible accuracy improvement versus pt-mid, with tolerable length growth.
2. **Useful checkpoint:** improves the observed saving/accuracy frontier.
3. **Finalist:** **≥52% saving and ≥−0.5 accuracy on the screen**, followed by validation.

To reach the finalist accuracy threshold from −2.6 requires approximately **+2.1 points**, not the draft’s +1.5.

A plausible *working-point budget*, not a prediction, is:

| Part | Saving | Accuracy change |
|---|---:|---:|
| Math | 55% | −2.0 |
| Code | 63% | −1.0 |
| IFEval | 50% | +0.3 |
| Simple | 37% | −0.5 |
| Multiple | 41% | +1.2 |
| Irrelevance | 43% | +3.0 |

This computes to **52.68% saved / −0.45 accuracy**.

If the same accuracy repair reduces math saving to 45%, aggregate saving falls to **50.25%**. There is little room for unbounded reasoning-length recovery.

### Replace the draft’s stopping rule

- **Saving <48% at step 40:** terminate this repair track unless evaluation uncertainty makes the crossing ambiguous. It has sacrificed more than seven saving points from pt-mid and is no longer a plausible fast repair.
- **Accuracy gain <0.5 at step 40:** a **warning, not an automatic stop**. That threshold is smaller than the reported seed variation. Stop for futility when paired uncertainty indicates little repair benefit and saving has also deteriorated.
- **At step 60:** continue only if it is on an improving frontier or close enough that the remaining pass could plausibly cross the acceptance bar.
- A checkpoint at 50/−1 is **not a finalist**; retain it only if it improves the useful frontier.

### How A could fail

- pt-mid produces too few correct answers on damaged questions to construct the intended pairs.
- Correctness recovery comes mainly from restoring long thought.
- Shortest-right guards still carry substantial hidden shortening pressure.
- Irrelevance repair becomes indiscriminate refusal, damaging valid calls.
- DPO raises relative preference by lowering rejected likelihood while also lowering chosen likelihood.
- Pair graders reward formatting artifacts rather than task correctness.

Those failure modes require different interventions; “train longer” is not a general remedy.

---

# 3. Track B: four adapters, verified pair supply, and a smaller-strength search

## 3.1 Four or six?

**Four initially: math, code, IFEval, tools.**

Reasons:

1. The only demonstrated per-type merge results use **four**, not six.
2. The proposed tool subsets contain only **480/240/480 pairs**. At batch 16, six-way training would give only **30/15/30 steps** for their adapters.
3. A shared tools adapter can learn relevance discrimination jointly, rather than summing a call-specialist update and a potentially conflicting refusal-specialist update.
4. Four dimensions make the merge search more credible under limited validation.

This does **not** establish that four is intrinsically better. If the joint tools adapter repeatedly trades irrelevant-call suppression against valid-call accuracy, split tools in a follow-up. A useful first split would be **valid-call versus irrelevance**, rather than immediately introducing all six adapters.

**Confidence: medium.**

## 3.2 Keep the six-part pair quotas inside those four adapters

| Part | Total | Shorten | Guard | Similar-length right |
|---|---:|---:|---:|---:|
| Math | 720 | 72 — 10% | 288 — 40% | 360 — 50% |
| Code | 1,440 | 936 — 65% | 288 — 20% | 216 — 15% |
| IFEval | 1,440 | 936 — 65% | 288 — 20% | 216 — 15% |
| Simple | 480 | 216 — 45% | 120 — 25% | 144 — 30% |
| Multiple | 240 | 48 — 20% | 84 — 35% | 108 — 45% |
| Irrelevance | 480 | 48 — 10% | 240 — 50% | 192 — 40% |
| **Total** | **4,800** | **2,256** | **1,308** | **1,236** |

The tools adapter therefore receives **1,200 pairs: 312 shorten, 444 guard, 444 right**.

### Grounding for the shares

- **Math 10/40/50:** D9’s aggressive compression caused **−11.7 to −13.1 math points**. D7 showed correctness recovery; Run B’s gentle shortening did not clearly improve the global frontier. Ten percent is a cautious probe, not an optimized share.
- **Code 65/20/15:** target the demonstrated **57/+0.4** region of pt-soft, while avoiding pt-mid’s **65/−3.5**. Run A’s 70% shortening is precedent, but not proof that 65% is optimal.
- **IFEval 65/20/15:** D9 step 100 reached **58/−1.0** and pt-mid **50/+0.6**. It merits compression pressure, but constraint compliance and answer runaway require protection.
- **Simple 45/25/30:** lower shortening pressure than the earlier 60% recipe because D9 caused **−6.9** points at 52% saving.
- **Multiple 20/35/45:** protect the strongest thinking dependence among tools; full-test thinking-off loss was **−20.8**. Moderate merge doses already achieved **37–41% saving with +1.7**.
- **Irrelevance 10/50/40:** prioritize correcting behavior rather than mining plentiful easy shorten pairs. Its training accuracy is **91.8%**, versus **83.4%** on the test; pt-mid lost **7.1 points**, whereas D7’s two-stage line recovered strongly.

**Confidence:** medium in the ordering of compression pressure; low in the exact percentages.

The total 4,800 and 720/1,440/1,440/1,200 allocation retain the earlier **15/30/30/25** benchmark training shares. They provide a familiar experiment size and adequate quotas—not a claim that pair counts should equal scoring weights.

## 3.3 Exact shortening rules

All lengths below are generated-token counts unless explicitly called thinking length.

| Part | Eligibility and chosen answer | Required ratio | Chosen thinking window |
|---|---|---|---:|
| Math | 4/4 right; second-shortest right | Rejected/chosen total length 1.3–4 | 128–1,024 |
| Code | ≥3/4 right; shortest right | Total 1.5–6 | 128–2,048 |
| IFEval | ≥3/4 right; shortest right | Thinking 1.5–6; total ≥1.15 | 32–1,400 |
| Simple | ≥3/4 right; shortest right | Total 1.5–6 | 48–384 |
| Multiple | 4/4 right; second-shortest right | Total 1.3–4 | 64–384 |
| Irrelevance | ≥3/4 right; shortest right | Total 1.5–6 | 48–384 |

Additional exact conditions:

- Rejection is another finished correct answer.
- At least **50% of the total token difference** must be thinking-token difference; **80% for IFEval**.
- At most **two shorten pairs per question**.
- For four correct samples, “median” means the observed **second-shortest answer**, never an interpolated or rewritten answer.
- Windows apply only to shorten pairs—not correctness pairs or inference.

### What grounds those eligibility numbers?

- **1.5×** has direct precedent in B and the iterative line.
- **1.3×** is a deliberately gentler median-based signal for protected tasks; it is not established as superior to 1.5.
- **≥3/4 and 4/4** reduce shortening on fragile questions. Four samples remain a noisy solve-rate estimate; these are selection strata, not reliable estimates of true success probability.
- Upper ratios **4 and 6** prevent extreme examples from dominating the shortening pool. Their exact values are engineering choices.
- The IFEval **80% thinking-difference rule** avoids learning to remove required answer content. Its exact value is untested.

The absolute windows are especially weakly established. The computation shows they are mostly inactive:

| Part | Ratio/gap-qualified candidate pairs excluded by window |
|---|---:|
| Math | 4.14% |
| Code | 1.49% |
| IFEval | 0% |
| Simple | 0.23% |
| Multiple | 0.97% |
| Irrelevance | 1.14% |

**None of the lower bounds excluded any candidate in this store.** Keep them as documented safety filters, but do not attribute experimental performance to them. They could become much more consequential in round 2.

## 3.4 Exact guards and right pairs

**Guards:**

- Half shortest right > shortest finished wrong.
- Half shortest right > longest wrong/cut.
- Cut rejections ≤25% of guards.
- Rejections longer than four times chosen ≤25% of guards.
- No chosen-length window.

**Right pairs:**

- Correct > finished wrong.
- Symmetric length ratio ≤1.25.
- Exactly half have chosen strictly longer; remainder chosen shorter/equal.
- Prefer lower solve-count disagreements where feasible.

**Global:**

- Maximum three pairs per question.
- Maximum two right pairs per question.
- No duplicate chosen/rejected sample pair across kinds.
- No wrong/wrong hard pairs.

The 25% caps are conservative policy choices, not experimental optima. In the feasible allocation I constructed, they were not binding: selected guard cut shares were at most **11.8%**, and extreme-ratio shares at most **14.6%**.

---

## 3.5 Recomputed availability under these rules

I enumerated candidates from all **71,480 metadata records**, then solved a constrained allocation problem enforcing the quotas jointly.

### Candidate capacity

“Right capacity” includes the two-per-question cap. Guard columns overlap with one another and with right pairs, so they must not simply be added.

| Part | Shorten capacity | Shortest-wrong guards | Longest-wrong/cut guards | Unique guards, union | Right capacity | Questions supplying right pairs |
|---|---:|---:|---:|---:|---:|---:|
| Math | 2,732 | 1,478 | 1,644 | 2,327 | 1,183 | 808 |
| Code | 1,490 | 848 | 886 | 1,344 | 865 | 563 |
| IFEval | 1,293 | 607 | 681 | 1,055 | 714 | 459 |
| Simple | 417 | 229 | 229 | 344 | 316 | 183 |
| Multiple | 204 | 174 | 174 | 264 | 250 | 144 |
| Irrelevance | 1,297 | 259 | 260 | 400 | 218 | 145 |

**All six sets of quotas are simultaneously feasible without generating additional answers.** The constructed allocation contains exactly 4,800 pairs, no duplicate sample pairs, and no question exceeding three pairs.

Important tight spots:

- IFEval needs **936 shorten pairs from 896 eligible questions**, so allowing second pairs is necessary.
- Irrelevance needs **192 right pairs from 145 questions**, so the two-right-pair allowance is necessary.
- Irrelevance right capacity is only **218** against a quota of 192: label or content audits could consume that slack.

### One feasible allocation’s length signals

These are **an existence witness**, not a claim that this particular random tie-breaking is optimal.

| Part | Shorten chosen/rejected median | Guard chosen/rejected median | Right chosen/rejected median | Guards with chosen longer |
|---|---:|---:|---:|---:|
| Math | 477 / 751 | 602 / 1,007 | 600 / 589 | 25.3% |
| Code | 517 / 1,074 | 887 / 1,152 | 1,100 / 1,086 | 29.9% |
| IFEval | 298 / 515 | 411 / 507 | 420 / 405 | 33.3% |
| Simple | 133 / 229 | 147 / 180 | 161 / 160 | 19.2% |
| Multiple | 156 / 227 | 170 / 210 | 188 / 184 | 26.2% |
| Irrelevance | 142 / 257 | 219 / 371 | 239 / 250 | 16.7% |

Every right-pair quota has exactly **50% chosen-longer examples**.

These counts establish **metadata feasibility**. The full store must still validate thinking closure, grader consistency, provenance and exact-token duplicate answers. All metadata-labelled right answers have positive thinking length and none is cut, but metadata alone does not certify syntax or content.

---

# 4. Track B training and merge search

## 4.1 Replace B2 with a historically grounded four-adapter schedule

| Setting | Recommendation | Grounding/confidence |
|---|---|---|
| Precision | BF16 historical path | Matches the demonstrated pt family; medium |
| NLL anchor | **1.0, historical implementation** | Present in every disclosed pt merge. Not proven necessary; medium |
| LR | **Constant 2e−5** | Matches that family; medium |
| Warm-up | Match historical code; do not add 5% silently | Preserve comparability |
| Batch | **16**, not 8 | Direct historical precedent; medium |
| Training | One pass | Matches historical dose; medium |
| Math | 720 pairs → **45 steps**, save 15/30/45 | Exact arithmetic |
| Code | 1,440 → **90**, save 30/60/90 | Exact arithmetic |
| IFEval | 1,440 → **90**, save 30/60/90 | Exact arithmetic |
| Tools | 1,200 → **75**, save 25/50/75 | Exact arithmetic |

My previous batch-8 recommendation doubled the number of optimizer updates without evidence that this was beneficial. **Withdraw it as the default.**

Start the merge search with the **final one-pass adapters**, because that is how the historical strengths were obtained. Save earlier adapters for a documented fallback if training diagnostics or merged evaluation show overtraining.

### NLL uncertainty

DPO’s preference objective can improve margins while reducing chosen-answer likelihood; an NLL term can oppose that. Conversely, it can preserve verbosity by imitating long correct answers.

Therefore:

- Start B with **1.0** to preserve the demonstrated per-type family.
- If an anchor ablation is needed, compare **1.0 versus 0** with identical pairs, batch, schedule and seed.
- Do not infer the answer by comparing unrelated D-runs to pt merges.

## 4.2 Replace the six-dimensional strengths

My earlier starting strengths—particularly code 0.80, IFEval 0.90 and irrelevance 0.85—were not justified by the now-known merge scales.

Use these **three actual historical vectors** first:

| Candidate | Math | Code | IFEval | Tools |
|---|---:|---:|---:|---:|
| Softer | 0.20 | 0.25 | 0.35 | 0.25 |
| Soft | 0.25 | 0.35 | 0.45 | 0.35 |
| Mid | 0.30 | 0.45 | 0.60 | 0.45 |

The new adapters have different pair distributions and potentially different delta norms, so these remain **starting points, not transferable optima**.

Around the best vector, test at most eight one-coordinate variants:

- Math **±0.05**
- Code **±0.05**
- IFEval **±0.10**
- Tools **±0.05**

This replaces the arbitrary global ×0.6/0.8/1.0 followed by ±0.15 search.

Why these increments?

- They are comparable to the historical adjacent strength changes.
- ±0.15 is three times the entire softer-to-soft math increment and risks jumping over the useful region.
- The historical results show steep code/math accuracy cliffs, warranting smaller moves.

**Budget: three initial screens plus at most eight coordinate screens.** That is a search cap, not an instruction to run all eleven. Stop when remaining candidates cannot plausibly improve the observed frontier.

Merge weight deltas, not LoRA factor matrices independently. Keep one merged model at inference.

## 4.3 Joint control

**Keep the control, but match B’s actual settings.**

- Same 4,800 pairs.
- One joint rank-32 adapter.
- BF16, NLL 1.0, constant LR 2e−5, batch 16.
- **300 steps**, screens at **100/200/300**.

The four separate runs also total **300 optimizer steps**. This is a cleaner grouping comparison than the draft’s batch-8 adapters versus batch-16 joint run.

It still is not a pure parameter-count-controlled comparison—the separate training has four adapters before merging—but it answers the practical question: does separate training plus merging improve the deployable model?

---

# 5. Track B round 2: gate, exact pool and dose

## Gate

Change **≥46% and >−0.7** to:

> Run one compression-oriented on-policy round only if saving is **48–52%** and accuracy is **≥−0.5**, with no unresolved task-specific collapse.

- If already ≥52/−0.5: validate; do not automatically add a round.
- If accuracy is worse: use a repair experiment, not more compression.
- If saving is below 48%: first fix the off-policy operating point.

The arithmetic explains the change:

| Starting saving | Reduction in remaining tokens needed to reach 52% |
|---|---:|
| 46% | 11.11% |
| 48% | 7.69% |
| 50% | 4.00% |
| 51% | 2.04% |

The older iterative line lost approximately **0.9 accuracy points between rounds 1 and 2**. Starting at −0.7 is not an attractive accuracy budget for another shortening round.

**Confidence:** medium in tightening the gate; low in the exact 48 threshold.

## Keep the 2,400-pair pool, explicitly

| Part | Shorten | Guard | Right | Total |
|---|---:|---:|---:|---:|
| Math | 0 | 200 | 200 | 400 |
| Code | 352 | 160 | 128 | 640 |
| IFEval | 384 | 160 | 96 | 640 |
| Simple | 72 | 84 | 84 | 240 |
| Multiple | 16 | 64 | 80 | 160 |
| Irrelevance | 0 | 192 | 128 | 320 |
| **Total** | **824** | **860** | **716** | **2,400** |

- Generate approximately **3,000 questions ×8** from the selected merged model.
- A concrete allocation is math **600**, code **800**, IFEval **800**, simple **300**, multiple **200**, irrelevance **300**.
- Prioritize disagreement, long failures and difficult relevance decisions; none may come from either evaluation set.
- **75% new on-policy pairs, 25% original-store replay.**
- Shorten eligibility: **≥6/8 correct**, except multiple **≥7/8**.
- Keep the same length and pair-integrity rules; inspect whether the lower windows now remove substantial supply.
- No math or irrelevance shorten pairs.

The round-2 count and 75/25 replay are **low-confidence trust-region-like engineering choices**, not findings from the data. Their purpose is to limit distribution drift and retain correctness examples.

### Training

- Start/reference: selected merged model.
- LR **1e−5**, beta **0.1**.
- Keep B’s **NLL 1.0** initially.
- Batch **64**.
- One pass: **38 steps**, the last batch containing **32 pairs**, not duplicated padding examples.
- Warm-up **2 steps**, cosine to zero.
- Save at **10/20/30/38**.

The approximately 20–45-step older iterative experiments motivate a short round, but do not establish equivalence: their data and batches differed. Batch 64 and 38 steps are therefore **low-confidence dose choices**.

---

# 6. Section 9: direct answers

### BF16 versus FP32/FP16: wait?

**Read the running precision experiment before choosing the final path if its result is imminent. Do not stall CPU pair construction or data audits.**

A precision change can affect loss scaling, clipping and update norms even if final accuracy differences are small. Compare:

- Nonfinite/overflow events.
- Skipped optimizer steps.
- Gradient and adapter-update norms.
- Chosen likelihood and preference margins.
- Saving and accuracy at matched pair exposure.

If the two paths agree within noise and show no stability differences, use BF16 for B to match historical training and simplify execution. Otherwise, adopt the stable path and stop presenting the run as a precision-matched reproduction.

### Cosine to zero versus 2e−6?

For **B**, use the historical **constant 2e−5**, not either cosine schedule, initially.

For **A and the short on-policy round**, use cosine to zero as specified. There is no controlled evidence that a 2e−6 floor improves this task. My earlier suggestion of that floor was an unverified optimizer preference, not a result.

### Four or six?

**Four now.** Revisit the tools grouping only when diagnostics show a reproducible simple/multiple versus irrelevance conflict.

### Is 720×4 plus 240×8 enough?

**Enough for screening and detecting sizeable regressions; not enough to certify a sub-one-point loss for a borderline model.**

The 240 set has only roughly 40 questions per part if balanced. Repeated answers reduce sampling noise but do not replace independent questions. For illustration, even treating 320 answers as independent gives a worst-case single-part accuracy interval of approximately **±5.5 points**; question clustering can make it wider. Paired model differences may be much more precise, but their precision must be measured.

Use the available sets as follows:

1. **720×4:** development and checkpoint selection.
2. **240×8:** locked confirmation, not another repeatedly queried tuning set.
3. **Two seeds:** stability check for the frozen recipe.
4. **Full test:** final authorized measurement.

Confirm whether the “fresh” set has already been repeatedly used historically. If so, call it a secondary validation set, not a pristine holdout.

Do not claim that seed averaging or bootstrap resampling creates the missing independent questions.

---

# 7. Replace the training tripwires with informative measurements

## The draft’s 60%/50%/95% rules are not valid stopping laws

At initialization, policy equals reference, so DPO reward margins are zero. Depending on tie handling, “ordered right” can appear as 0%, 50%, or another implementation-specific value.

- **Above 60% by mid-run:** not a necessary condition for downstream improvement.
- **Approximately 50%:** could indicate weak learning, but also implementation or tie-handling issues.
- **Above 95% early:** could indicate saturation or easy training pairs; it does not by itself imply a bad model.

Keep these as dashboard observations, not acceptance criteria.

## What to save

### Before training

Save the immutable:

- Base model, adapter and tokenizer hashes.
- Full pair manifest: question ID, source, sample IDs, token hashes, kind, solve count, correctness, cut status.
- Prompt, chosen, rejected, thinking and final-answer token lengths.
- Ratio/gap filters and reasons for exclusion.
- Cross-kind deduplication results.
- Per-question reuse counts.
- Quota allocation seed and pair ordering.
- Grader version and all validation/audit exclusions.
- Exact DPO and NLL reductions, optimizer settings, precision and batch accumulation.

This is necessary to distinguish “pair rule helped” from “a different subset of questions helped.”

### Every optimizer step; summarize every five

Keep:

- DPO loss and NLL loss **separately**.
- Chosen/rejected policy and reference log-probabilities.
- Preference margins and ordered-pair fraction by **part and kind**.
- Chosen likelihood change, both sequence-level and token-normalized.
- Gradient norm before clipping; clipping frequency.
- Learning rate, loss scale, overflow/skipped-step events.
- Adapter-delta norm by layer and projection.
- Pairs and completion tokens processed.

Stop immediately for persistent nonfinite values or corrupted updates. A finite loss spike deserves diagnosis; it is not automatically a scientific failure.

### Every screened checkpoint or merge

Save per question and sample:

- Correctness and checker failure category.
- Total, thinking and final-answer tokens.
- Cut status and whether thinking closed.
- Empty/malformed thinking and malformed final answers.
- Tool action class: correct call, wrong call, false refusal, correct refusal.
- Code failures separated into naming/parsing, ordinary test failure and cutoff.
- IFEval failure by constraint family, including answer runaway.
- Generation settings and RNG seed.

Report:

- Official aggregate score and all six parts.
- Means and tail quantiles for token lengths.
- Always/sometimes/never solved transitions.
- Accuracy changes by baseline difficulty and baseline length.
- Paired confidence intervals using **question-level resampling**, stratified by part, keeping repeated answers together.
- BFCL pooling and geometric token scoring inside every bootstrap replicate.

## Replace the blanket “any part below −3” alarm

An absolute −3 alarm fires immediately on pt-mid’s starting math and code, so it is not useful for A.

Instead:

- Flag **new degradation relative to the track’s starting model**, especially on maintenance tasks.
- Keep the **cut-rate increase >1 percentage point** as a warning, but inspect counts and uncertainty.
- Require manual review for disappearance of thinking, systematic false refusals or grader exploitation regardless of aggregate score.

---

# 8. What results would change my recommendation?

| Observation | Consequence |
|---|---|
| A gains accuracy mainly by lengthening math/code, and falls below 48% saving | Stop A; do not force a repair that destroys the useful operating point |
| A’s chosen likelihood collapses despite improving DPO margins | Run a matched NLL-anchor repair ablation |
| B’s joint control matches or beats separate adapters | Prefer the simpler joint model; abandon modularity as an assumption |
| Shared tools adapter improves no-fit but damages valid calls reproducibly | Split irrelevance from valid-call tools |
| Code is accurate near 57% saving but fails sharply beyond it | Stop increasing code dose; seek additional IFEval/tool savings or accept that the target is not yet reached |
| IFEval token savings come from violating required answer length | Reject the model; strengthen constraint-stratified pairing, not just token filters |
| No-fit label audit removes enough right pairs to miss the quota | Reallocate only after documenting the shortage, or generate targeted no-fit answers; do not replace correctness pairs with easy shortening |
| Round 2 reproduces the older approximately one-point accuracy debt | Stop after the earliest useful checkpoint; no automatic third round |
| Strong screen improvement disappears on the fresh set or second seed | Treat it as selection noise/domain mismatch, not proof that more strength tuning is needed |

---

# 9. Execution order and cost correction

**Revised order:**

1. CPU-only pair construction, feasibility checks, provenance and grader audits.
2. Track A on one GPU.
3. Track B’s four adapters on the other GPU, then use both as available.
4. Three historical-strength merge screens.
5. Joint control in parallel with the limited coordinate search.
6. Only then the gated round 2.
7. Freeze the recipe; second seed and fresh validation; authorized full test.

The draft’s GPU-hour estimates should be treated as **reservations**, not predictions:

- **A:** approximately 3 hours generation remains plausible, but measure it. Training may take roughly 0.5–1.5 hours; four screens add approximately 4 hours. Budget **about 8–9 GPU-hours**, rather than promising 7.
- **B adapters:** retain a conservative **up-to-14 GPU-hour reservation**, but four batch-16 runs total only 300 updates; actual cost depends heavily on sequence lengths, reference caching and the NLL path.
- **Merge search:** **3–11 GPU-hours** at the stated one-hour screen cost.
- **Joint control:** approximately **6–7 GPU-hours**, including three screens.
- **Round 2:** reserve **up to approximately 13 GPU-hours**, conditional on passing the gate.

Measure throughput on the first representative batches and revise these estimates. If both tracks clearly fail their gates, stop rather than consuming the remaining three-day budget automatically.

**Bottom line:** the pair quotas are buildable now. The main corrections are to **repair code in A, return B to four historically scaled adapters, remove unsupported claims about NLL and optimizer settings, and distinguish exploratory thresholds from evidence that the strict target has actually been met.**