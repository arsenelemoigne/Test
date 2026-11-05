## Bottom line

**Yes—`lpt6` materially weakens my original rationale for Track B.** Separate training does not produce task-local compression controls. I would now rank the immediate options:

1. **Track A: repair the already-useful pt-mid model**, with the early checkpoints we agreed.
2. **A joint adapter using the revised, balanced pair recipe.**
3. **New per-type training only after a bounded completion of `lpt6` demonstrates useful complementarity.**

I nevertheless agree with **finishing `lpt6` as a diagnostic experiment**. Two important directions—math and irrelevance—are missing from its reported matrix, and those are exactly the directions hypothesized to supply protection. What I would not do is launch a large six-dimensional merge search and interpret its best noisy result as proof of modularity.

---

# 1. What the spill-over matrix says

I read both files and extracted all four reported `lpt6` adapters from the 43 shared runs.

Each entry below is **tokens saved % / accuracy-point change**. An adapter is evaluated alone at its reported full strength.

| Training pairs | Math | Code | IFEval | Simple | Multiple | Irrelevance |
|---|---:|---:|---:|---:|---:|---:|
| **Code only** | −3/−1.0 | **−5/+0.4** | −9/−3.1 | 4/−0.4 | 3/−2.3 | 2/+1.5 |
| **IFEval only** | 27/−3.5 | 32/−1.9 | **43/−4.4** | 26/0.0 | 26/−1.1 | 28/−3.5 |
| **Simple only** | 34/−4.2 | 41/−0.8 | 23/+1.0 | **34/−1.2** | 38/0.0 | 35/−7.3 |
| **Multiple only** | 18/−1.5 | 24/−2.5 | 9/0.0 | 21/−1.5 | **26/0.0** | 18/−2.9 |

### Compression is largely shared, but not completely task-independent

Comparing each adapter’s own-part saving with the unweighted mean saving on its other five parts:

| Adapter | Own-part saving | Other-parts mean | Own-part advantage |
|---|---:|---:|---:|
| Code | −5% | −0.6% | −4.4 points |
| IFEval | 43% | 27.8% | +15.2 |
| Simple | 34% | 34.2% | −0.2 |
| Multiple | 26% | 18.0% | +8.0 |

These are descriptive averages, not official benchmark-weighted scores.

The simple-only adapter is almost a textbook example of a **global compression update**: it shortens code more than simple calls, and it supplies the greatest saving on five of the six parts among these four adapters.

I also transformed each entry into log compression,

\[
d_{jp}=-\log(T_{jp}/T_{0p}),
\]

and examined the resulting four-by-six matrix:

- Its first singular component accounts for **95.0% of squared matrix magnitude**.
- Restricting to the three shortening adapters gives **95.6%**.
- The cosine similarity between the simple-only and multiple-only compression profiles is **0.991**.

This strongly supports a dominant shared behavioral effect. But it is an **uncentered, small descriptive matrix**. It does not prove that parameter updates are collinear, that there is literally one internal “thinking knob,” or that smaller residual directions cannot improve accuracy.

### The damage is not confined to math and irrelevance

Those are important vulnerabilities, particularly for simple-only training:

- Math contributes **−1.05 overall accuracy points**.
- Irrelevance contributes another **−0.52**.
- Together they explain almost all its reported **−1.7 overall** loss.

But IFEval-only training loses **4.4 points on IFEval itself**, contributing **−1.10 overall**. Multiple-only training’s largest weighted loss is actually code: **−0.625 overall**.

Thus the correct conclusion is:

> An isolated adapter can alter reasoning length globally while omitting correctness supervision on the tasks that absorb the resulting damage.

It is not simply “all costs always land on math and no-fit.”

### Code-only deserves careful interpretation

The code adapter produces:

- **5% more code tokens**;
- a slightly shorter thinking median, **966 versus 994**;
- a higher cutoff rate, **9.4% versus 8.5%**;
- **+0.4 code accuracy points**.

The median and mean move differently because the tail worsens. “It did not shorten code’s mean output” is established; “it learned no length behavior” is too strong.

Similarly, ordering **86% of held-out pairs correctly** does not establish shorter free-running generations. Pair discrimination, completion likelihood and sampled generation behavior are different measurements.

---

# 2. Why this does not contradict the successful four-adapter merges

The historical merges achieved:

| Merge | Saving | Accuracy change | Sum of four merge strengths |
|---|---:|---:|---:|
| Softer | 39.2% | −0.8 | 1.05 |
| Soft | 48.4% | −1.9 | 1.40 |
| Mid | 55.0% | −2.6 | 1.80 |

There are three compatible explanations.

## A. Adding several global compression updates can produce stronger global compression

Modularity is not necessary to explain 55% saving. Several broadly acting updates can reinforce one another.

For illustration—not as a forecast—I combined the four known `lpt6` solo effects using additive log token ratios and additive accuracy changes. Giving each a coefficient of 0.5 predicts approximately:

- **38.5% saving**
- **−3.4 accuracy points**

At coefficient 0.75 each, the same crude model predicts approximately **51.8%/−5.1**.

This shows that “large saving from a merge” is compatible with shared compression. It also shows that naïvely combining these particular solo effects does **not** predict an attractive accuracy frontier.

These calculations omit the unknown math and irrelevance effects and assume unverified dose linearity. They are diagnostic illustrations only.

## B. The recipes differ substantially

The historical four-adapter family used:

- Different pair construction, including its no-fit protection;
- **NLL weight 1.0**;
- Constant LR **2e−5**;
- A **joint tools adapter**;
- Different pair counts and resulting training exposure.

`lpt6` used:

- Rule B plus its additional math-protection examples;
- **No NLL**;
- Cosine **2e−5 → 2e−6**;
- Separate tool-subtype adapters.

Those changes can alter both compression and correctness. The results do not identify which change mattered.

## C. The historical code result need not have come from the code adapter

The strongest historical code result was **57% saving/+0.4 accuracy** at pt-soft. Without solo evaluations, we cannot attribute its compression to the code adapter.

The `lpt6` simple-only adapter already demonstrates **41% code saving**. It is entirely plausible that, in the historical merge:

- Other adapters supplied global compression;
- The code adapter supplied some correctness or content protection.

That is a hypothesis worth testing, not something the aggregate result proves.

### Consequence for my recommendation

My earlier language about “isolating compression by task” was too strong. **Separate training isolates the source of the gradient, not the affected behavior.**

The joint-training evidence is now more compelling:

- Run A at step 100: **33.4%/+0.4**.
- `lpt6` simple-only: **33.7%/−1.7**.
- `lpt6` IFEval-only: **32.6%/−2.8**.

At almost identical saving, the joint run is **2.1–3.2 accuracy points better**. These are not controlled comparisons, but they directly support maintaining cross-task protection during training.

Track A remains attractive because it starts from an actual **55% saving** operating point. The new evidence reinforces the need to retain maintenance pairs from all parts during repair; it does not establish that a math/no-fit update will stay local.

---

# 3. Four versus six adapters

## Why I recommended four

The defensible reasons were:

1. **Four is the grouping with demonstrated successful merges.**
2. The proposed subtype pools are small: **480 simple, 240 multiple, 480 irrelevance pairs**.
3. Four reduces merge-search dimensionality.
4. Joint tools training can oppose false calls and false refusals within the same optimization process.

What was **not** justified was interpreting the four adapters as four largely independent behavioral controls.

## Evidence for six

- Separate tool directions may have different accuracy effects even when their length effects are similar.
- Simple-only and multiple-only training show different accuracy profiles.
- Irrelevance is qualitatively different from valid-call selection.
- A subtype update can be assigned zero strength without discarding all tool training.
- Crucially, **irrelevance-only has not yet been reported**. It could provide a useful protective direction.

## Evidence against six

- Simple and multiple have compression-profile cosine **0.991**: two nominal knobs may mostly duplicate one behavioral direction.
- Separate training removes the other tool subtypes’ correctness supervision.
- The simple-only update causes **−7.3 irrelevance points**.
- Small subtype datasets and additional strengths increase variance and selection opportunities.
- Six-way merging has not yet demonstrated any advantage here.

### Revised position

**Neither four nor six is now my default assumption for a new compression run. Joint training is the simpler default challenger; per-type training is conditional.**

For completing this historical study, however, **use all six archived subtype adapters**. Changing to four midway would prevent the experiment from answering its original question.

If six works, compare it with a **pooled-tools four-adapter control** before claiming that subtype separation was necessary.

---

# 4. Should the team finish `lpt6` first?

**Yes, as a bounded mechanistic study.** Do not simultaneously change its pairs, precision, anchor or schedule.

This is an exception to our current FP32/FP16 default: **a historical completion should retain historical BF16**. Otherwise a newly trained math adapter becomes another confound.

## Phase 1: finish the solo matrix

### Training

Train the missing math adapter using:

- The **archived math slice of the exact `lpt6` pair set**, including its original protection examples;
- Original base model;
- BF16;
- Rank **32**, alpha **64**;
- No NLL anchor;
- Batch **16 pairs per optimizer step**;
- LR **2e−5 → 2e−6 cosine**, with **10 warm-up steps**;
- One pass;
- The same **two seeds and seed-averaging procedure** used historically.

The precise math-pair count and optimizer-step count are not in either new file. Recover them from the archived pair manifest and use the original handling of the final batch. Do not reconstruct a merely similar set from `raw_meta`.

Preserve both individual seed adapters as well as the averaged result. Record whether the historical averaging operated on weight deltas or LoRA factors; these are not generally equivalent.

### Screens

1. Math-only, strength **1.0**.
2. Existing irrelevance-only, strength **1.0**.

Use the existing full-strength screens for code, IFEval, simple and multiple, provided the evaluation path and baseline are unchanged.

Then score **all six adapters separately at strength 0.5**.

That is **eight new solo screens** total.

Why spend six screens at 0.5? Because a full-strength solo result is not a reliable derivative at half strength. These measurements distinguish solo dose nonlinearity from merge interaction.

---

## Phase 2: five predeclared merge screens

Use coefficient order:

**math / code / IFEval / simple / multiple / irrelevance.**

| Merge | Strengths | Question answered |
|---|---|---|
| **C: compression core** | **0 / 0 / 0.5 / 0.5 / 0.5 / 0** | What do the known shortening directions do without proposed protection? |
| **P: add protection** | **0.5 / 0 / 0.5 / 0.5 / 0.5 / 0.5** | Do math and irrelevance add useful protection? |
| **S6: all six** | **0.5 / 0.5 / 0.5 / 0.5 / 0.5 / 0.5** | Does the code adapter help in combination despite poor solo compression? |
| **P+: stronger protection** | **1.0 / 0 / 0.5 / 0.5 / 0.5 / 1.0** | Can protection improve the frontier, or does it only restore verbosity? |
| **T: change tool direction** | **0.5 / 0 / 0.5 / 0 / 1.0 / 0.5** | Is replacing simple pressure with multiple pressure useful? |

**Total core experiment: 13 new screens—eight solos and five merges.**

These strengths are experimental interventions, not predicted optima:

- 0.5 is a moderate, easily interpreted dose.
- 1.0 is already anchored by the original solo measurements.
- Zero gives genuine removal controls.
- T preserves the sum of simple/multiple coefficients, although it does not guarantee equal update norm.

Every coefficient used has a corresponding measured solo effect. That makes the additivity test substantially cleaner.

If a merge produces malformed behavior or gross collapse, stop that branch. Do not spend additional screens tuning around it.

### Why not immediately search many low strengths?

Because the purpose is initially **identification**, not maximizing a selected screen result. These contrasts isolate:

- Adding protection;
- Adding code;
- Increasing protection;
- Changing the tool-update direction.

A large unconstrained search could find a lucky checkpoint without explaining any of those questions.

---

# 5. How to test additivity correctly

For adapter \(j\), part \(p\), and its actual measured strength \(\lambda\), define:

\[
e_{jp}(\lambda)=
\log\left(\frac{\bar T_{jp}(\lambda)}{\bar T_{0p}}\right),
\qquad
a_{jp}(\lambda)=A_{jp}(\lambda)-A_{0p}.
\]

The simple additive predictions are:

\[
\widehat e_p=\sum_j e_{jp}(\lambda_j),
\]

\[
\widehat{\Delta A_p}=\sum_j a_{jp}(\lambda_j),
\qquad
\widehat S_p=100(1-\exp(\widehat e_p)).
\]

Use the **actual screen baseline token means**, not the full-test profile means.

Then compute:

\[
R^{\rm length}_p=e^{\rm observed}_p-\widehat e_p,
\qquad
R^{\rm accuracy}_p=
\Delta A^{\rm observed}_p-\widehat{\Delta A_p}.
\]

Interpretation:

- Negative length residual: more compression than predicted.
- Positive accuracy residual: better accuracy preservation than predicted.
- A useful merge may have some length sacrifice but a large positive accuracy residual.
- Near-additive broad compression with additive failures supports the “several general pushes” explanation.

These are **null models**, not laws. Shared failures, saturation and nonlinear generation make accuracy especially unlikely to add perfectly.

### The most informative contrasts

- **P − C:** protection in the presence of compression.
- **S6 − P:** code’s contribution in context.
- **P+ − P:** the marginal price and benefit of stronger protection.
- **T − P:** whether independent tool-subtype strengths have practical value.

Bootstrap questions jointly across the baseline, solos and merges. This preserves their correlation and gives uncertainty on the interaction residuals.

Also inspect per-question overlap: two adapters may appear non-additive simply because both already fail the same questions.

---

# 6. What result means “go Track B,” “go joint,” or “stop per-type”?

These are operational decisions under the remaining compute budget, not universal scientific verdicts.

## “Go Track B”

Proceed if the study demonstrates **useful complementarity**, not merely higher saving:

- A merge reaches approximately **≥48% saving and ≥−0.5 accuracy**, or the actual **≥52%/−0.5** finalist bar; **and**
- Adding protective or code directions improves the observed frontier, rather than just undoing compression.

Alternatively, a strong improvement at lower saving can justify one limited follow-up, but not the full previous Track B program.

Before committing substantial new per-type training, compare against a matched joint run. A practically meaningful advantage would be approximately:

- **≥0.5 accuracy points** at saving within **1 point**, or
- **≥2 saving points** with accuracy within **0.2 points**,

with paired uncertainty supporting the direction. These are predeclared engineering margins, not claims about statistical power.

## “Go joint”

Prefer joint training if:

- The merged directions mainly accumulate global compression and corresponding damage;
- Protection does not improve the frontier;
- A matched joint run supplies at least the above practical advantage.

For that comparison, train one joint adapter on the **union of the exact same `lpt6` pairs**, using the same historical optimizer, precision, anchor and two seeds. Screen at the halfway checkpoint and final checkpoint.

This adds **two screens**, plus training. It is a much stronger comparison than using Run A alone.

If no matched control is run, the conclusion is only **“joint is the next better-supported experiment,”** not “joint has been causally proven superior.”

## “Stop this per-type recipe”

Stop further `lpt6`-style investment if:

- Every tested merge is dominated by available joint/baseline candidates, allowing for uncertainty;
- Higher saving still entails unacceptable accuracy loss;
- Math/irrelevance/code additions fail to buy useful accuracy;
- Tool-subtype reweighting gives no meaningful frontier improvement.

Do **not** call all per-type training “dead.” Five merges cannot exclude every coefficient vector or every pair recipe. The historical pt family already establishes that another per-type recipe behaves differently.

The defensible conclusion would be:

> This particular separate-training recipe offers no demonstrated benefit worth pursuing within the current budget.

---

# 7. How to settle four versus six if the pilot succeeds

Only if the six-way pilot is promising, train **one pooled-tools adapter** on the exact union of the three tool slices, using the same historical settings.

Compare:

- Six-way S6: **0.5/0.5/0.5/0.5/0.5/0.5**.
- Four-way: math/code/IFEval **0.5/0.5/0.5**, pooled-tools strength **0.25 and 0.5**.

That is **two further screens**.

The two tool strengths provide a small dose bracket because a pooled adapter’s scale need not match the sum of three separately optimized updates. Even then, interpret differences alongside update norms and token saving—not nominal coefficients alone.

If four matches six, choose four for simplicity. If T or another subtype contrast provides a reproducible advantage the pooled adapter cannot match, retain subtype separation.

---

# 8. The highest-value additional data

## A. Score the actual four pt-family adapters alone

This is arguably the most informative missing experiment for reconciling the history.

Score each historical adapter alone at its **pt-mid deployed strength**:

- Math **0.30**
- Code **0.45**
- IFEval **0.60**
- Tools **0.45**

**Four screens, no training.**

Their summed log-token and accuracy predictions can be compared with the already measured pt-mid merge. This directly answers whether its success reflects:

- Strong standalone components;
- Global compression plus specialized protection;
- Or beneficial interactions.

If GPU time is tight, I would prioritize these four screens over an expansive merge search.

## B. Per-question screen records

For the original model, solo adapters, historical pt merges and new merges, obtain:

- Correctness for every sample;
- Total, thinking and final-answer token counts;
- Cut status and thinking closure;
- Tool call/refusal category;
- IFEval constraint failures;
- Code failure categories.

These enable:

- Paired uncertainty;
- Failure-overlap analysis;
- Easy/hard and short/long question slices;
- Detection of “protection” that merely trades one class of mistakes for another;
- Proper decomposition of mean-length changes into ordinary outputs and cut tails.

## C. Exact pair and training manifests

Especially important here:

- Pair counts per part and kind;
- Chosen/rejected lengths and direction;
- Provenance of the trained-model “too-short wrong” examples;
- Actual optimizer steps, batches and token exposure;
- NLL reduction and coefficient;
- Seed-averaging implementation;
- Checkpoint and model hashes.

Without these, “code pairs teach content” and “the NLL anchor explains pt success” remain hypotheses.

## D. Actual update geometry

Compute layerwise norms and cosine similarities of the **merged weight deltas**, not raw LoRA factors.

The output compression matrix is nearly rank one; the updates might or might not be. Combining behavioral profiles with update geometry will distinguish redundant updates from different parameter directions that happen to produce similar length effects.

---

## Final recommendation

**Finish `lpt6`, but cap its initial completion at the 13-screen diagnostic design above.** Keep Track A as the leading target-reaching experiment. Do not launch the full new per-type Track B program merely because a six-way merge shortens more.

The decisive evidence would be that a separately trained direction **preserves or restores correctness in the presence of the shared compression update**, and that this advantage survives comparison with joint training. That is the modularity worth paying for—not whether each adapter bears a different task label.