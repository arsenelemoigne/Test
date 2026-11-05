# Recommendation

**Keep the per-type approach, but change what it is supposed to achieve.** These adapters are not independent “compression knobs” for their training types. They mostly supply a shared length-control effect, plus type-specific correctness biases. Jasmin should optimize that combination—not maximize each adapter’s own-type shortening and then add the winners.

The two new solos overturn the premise behind the paused “protection” merges:

- **lpt6-math is not a protector:** it makes math **78% longer and 16.2 points less accurate**.
- **lpt6-no-fit is not a general tools protector:** its **+11.7** on no-fit accompanies **−20.6/−16.6** on callable tools.

I would **cancel P, S6, Pplus and T as currently specified**, retain C as a diagnostic, and spend most screens testing pt component necessity, dose-matched substitutions, and whether a better compression base improves repair.

My current Jasmin default is **four deployed groups, pt-style pairs, NLL=1, float32 master weights, early checkpoints, and conservative task arithmetic**. Train the three tool slices as challengers if affordable, but also retain a genuinely pooled-tools training run.

---

# 1. Updated spill-over analysis

## Scoring and uncertainty

All numbers below are screen measurements unless labeled predictions. An entry is:

**tokens saved % / accuracy change in points**.

Two qualifications matter:

1. Screen noise is approximately **±1 point overall and ±3–4 per part**. The matrices support large effects and broad patterns, not fine rankings between nearby cells.
2. The recorded aggregate scores fit **equal weighting of the three BFCL screen parts** more closely than the full-test BFCL proportions. For example, lpt6-no-fit’s recorded accuracy is **−1.9**; applying the full-test BFCL weights, **400:200:240**, to its displayed part deltas gives **−2.40**.

This could reflect balanced screen sampling rather than a scoring error. Confirm the screen counts and aggregation before selection. I retain recorded totals for historical comparisons and explicitly use full-test weights where appropriate. Exact token-score reconstruction also requires baseline token totals, which were not supplied.

## Complete matrices

### lpt6: full-strength two-seed soups

| Adapter | Math | Code | Instructions | One function | Several functions | No-fit |
|---|---:|---:|---:|---:|---:|---:|
| Math | **−78 / −16.2** | −17 / +2.3 | −39 / +1.5 | −23 / −1.7 | −23 / 0.0 | −29 / +2.5 |
| Code | −3 / −1.0 | **−5 / +0.4** | −9 / −3.1 | 4 / −0.4 | 3 / −2.3 | 2 / +1.5 |
| Instructions | 27 / −3.5 | 32 / −1.9 | **43 / −4.4** | 26 / 0.0 | 26 / −1.1 | 28 / −3.5 |
| One function | 34 / −4.2 | 41 / −0.8 | 23 / +1.0 | **34 / −1.2** | 38 / 0.0 | 35 / −7.3 |
| Several functions | 18 / −1.5 | 24 / −2.5 | 9 / 0.0 | 21 / −1.5 | **26 / 0.0** | 18 / −2.9 |
| No-fit | 11 / +0.2 | 9 / +1.0 | 3 / −0.4 | 18 / −20.6 | 17 / −16.6 | **36 / +11.7** |

### pt: strength 1.0

| Adapter | Math | Code | Instructions | One function | Several functions | No-fit |
|---|---:|---:|---:|---:|---:|---:|
| Math | **59 / −4.8** | 51 / −1.0 | 26 / +0.6 | 21 / −1.7 | 24 / −1.3 | 25 / −6.0 |
| Code | 20 / −1.3 | **25 / −0.2** | 7 / −1.3 | 14 / −1.5 | 14 / −2.3 | 11 / −3.5 |
| Instructions | 34 / −2.7 | 42 / +0.2 | **40 / +0.6** | 28 / −1.9 | 29 / −1.3 | 29 / −5.6 |
| Pooled tools | 53 / −4.0 | 59 / −4.6 | 44 / −3.7 | **38 / +1.5** | **43 / +2.5** | **45 / −2.3** |

### pt: actual pt-mid strengths

These are measured solos at **math .30, code .45, instructions .60, tools .45**, not interpolations.

| Adapter | Math | Code | Instructions | One function | Several functions | No-fit |
|---|---:|---:|---:|---:|---:|---:|
| Math @ .30 | **20 / −1.5** | 13 / +0.8 | 4 / −1.5 | 9 / −1.5 | 10 / −2.3 | 11 / −1.9 |
| Code @ .45 | 9 / −2.5 | **14 / −2.3** | 11 / −3.7 | 4 / −0.8 | 7 / −2.1 | 5 / −1.2 |
| Instructions @ .60 | 24 / −0.8 | 20 / −0.2 | **27 / −1.5** | 20 / −0.6 | 20 / −2.7 | 22 / −3.3 |
| Pooled tools @ .45 | 27 / −2.9 | 32 / +0.4 | 28 / −0.8 | **25 / −2.5** | **28 / −0.2** | **28 / −0.8** |

There are no corresponding solo measurements at every pt-soft/pt-softer coefficient; those should not be invented from linear scaling.

## Shared behavior: strongly low-dimensional, but not exclusively “think less”

I transformed savings into log compression,

\[
d_{jp}=-\log(1-S_{jp}/100),
\]

then performed an SVD.

| Matrix | First component’s share of squared magnitude | First two components |
|---|---:|---:|
| All six lpt6 solos | **89.4%** | **95.0%** |
| Four pt solos, 1.0 | **96.7%** | **99.6%** |
| Four pt solos, deployed strengths | **97.4%** | **98.9%** |

For lpt6, the first component has similarly signed, similarly sized loadings across all six evaluation parts. The three shortening adapters—instructions, one-function, several-functions—alone are **95.6% rank-one**.

Thus:

- Much of compression is shared across types.
- lpt6-math largely moves in the **opposite direction of the shared length factor**.
- Adding math and no-fit reduces the earlier four-solo rank-one result from about **95% to 89%**, revealing meaningful additional structure.

These matrices are nevertheless algebraically full rank. The SVD is descriptive, uncentered, and based on a tiny behavioral matrix; it does **not** establish collinear parameter updates or a literal one-dimensional internal mechanism. Column-centering leaves first-component shares of **90.7%, 80.5%, and 91.5%**, respectively, so the shared effect is not merely an artifact of using uncentered data.

Also, the measurements concern **all generated tokens**. Without separate thinking/answer token counts, “shared think-less habit” is a useful interpretation, not a directly isolated measurement.

## Type specificity: more evident in correctness than in compression

Own-type saving minus the mean saving on other types:

| Adapter | lpt6 @ 1 | pt @ 1 | pt @ deployed strength |
|---|---:|---:|---:|
| Math | **−51.8** | +29.6 | +10.6 |
| Code | −4.4 | +11.8 | +6.8 |
| Instructions | +15.2 | +7.6 | +5.8 |
| One function | −0.2 | — | — |
| Several functions | +8.0 | — | — |
| No-fit | +24.4 | — | — |
| Pooled tools | — | −10.0 | −2.0 |

These are descriptive, unweighted comparisons; “own” for pooled tools averages its three parts.

Important interpretations:

- **pt has some local compression advantage**, especially math, but not independent task controls.
- **lpt6 one-function is a global compressor**, not a one-function-specific compressor.
- **lpt6 math specializes in the wrong direction:** own-type accuracy is **17.1 points worse** than its mean accuracy effect elsewhere.
- **lpt6 no-fit has a strong label/action bias:** own-type accuracy advantage is **19.0 points**, bought by suppressing legitimate calls.
- **pt pooled tools supplies task-specific correctness despite nonlocal compression:** at 1.0, its full-test-weighted BFCL accuracy change is **+0.65**, versus **−4.10** averaged over math/code/instructions.

Accuracy is less rank-one than length: its first component explains **66.9% for lpt6**, versus 89.4% for log compression. That is consistent with a shared length response plus several correctness-specific effects.

The no-fit trade-off is especially unfavorable under official weighting. Its callable losses imply **−10.42 points pooled BFCL accuracy**, despite the no-fit gain. Holding those callable losses fixed, BFCL neutrality would require **+48.2 points on no-fit**, not +11.7.

## Measured merge protection: real non-additivity, not universal protection

Use two diagnostic nulls:

- **Length:** add log token ratios, equivalently multiply token ratios.
- **Accuracy:** add solo accuracy changes.

For pt-mid:

| Part | Predicted saving from deployed solos | Measured saving | Sum of solo accuracy changes | Measured accuracy | Accuracy residual |
|---|---:|---:|---:|---:|---:|
| Math | 59.6 | 62 | −7.7 | −5.6 | **+2.1** |
| Code | 59.3 | 65 | −1.3 | −3.5 | **−2.2** |
| Instructions | 55.1 | 50 | −7.5 | +0.6 | **+8.1** |
| One function | 47.6 | 37 | −5.4 | −0.4 | **+5.0** |
| Several functions | 51.8 | 41 | −7.3 | +1.7 | **+9.0** |
| No-fit | 52.5 | 42 | −7.2 | −7.1 | **+0.1** |

Aggregate shorthand:

- Multiplying the recorded solo aggregate token ratios predicts **56.46% saved**, versus **55.0% measured**.
- Solo accuracy changes sum to **−5.8**, versus **−2.6 measured**: approximately **+3.2 points of apparent protection**.
- Full-test-weighted part residuals give **+3.14 points**.

The aggregate length agreement hides substantial redistribution: tools retain **about 10.5–10.8 percentage points more tokens** than predicted, while code is compressed more.

Therefore, **do not transfer a blanket +3.2-point protection bonus to new merges**. Protection is strongest on instructions and callable tools, absent on no-fit, and reversed on code. Overlapping failures, nonlinear decoding, and retained reasoning can all contribute. Per-question records are needed to separate these explanations.

---

# 2. Why lpt6 is worse: ranked causes

“Worse” is not uniform: lpt6 one-function and several-functions remain potentially useful directions. The serious failures are math, code’s lack of compression, instructions’ accuracy, and no-fit’s action bias.

| Rank | Candidate cause | Evidence and limitation | What screens can establish |
|---|---|---|---|
| **1** | **Pair composition, selection and training exposure** | Directly consistent with the learned behaviors; pair-only joint experiments produce very large changes. Counts and objective composition remain confounded. | Whether the resulting directions are useful at lower dose or in context—not which training ingredient caused them. |
| **2** | **NLL anchor 1 versus 0** | Plausible protection against drifting away from chosen correct sequences, especially with narrow slices. No matched per-type ablation exists. | Cross-family substitution tests the complete recipe bundle. NLL causality requires training. |
| **3** | **Pooled versus split tools** | Strongly relevant to no-call bias: pooled tools sees both “call” and “do not call.” Cannot explain math/code/instructions, whose grouping did not change. | Dose-matched pooled/split substitutions test practical usefulness. Clean grouping attribution needs matched training. |
| **4** | **Constant versus cosine schedule** | Plausible through exposure and checkpoint timing, but cosine is demonstrably capable of good results. Final checkpoints are not matched optimization doses. | Half-strength and checkpoint screens diagnose overdose; they cannot identify schedule causality. |
| **5** | **Seed and soup effects** | Possible, particularly for narrow datasets, but currently less compelling than the systematic behavioral evidence. | Individual s0/s1 versus their weight soup can be tested without training. |

### Why pair composition/exposure ranks first

- lpt6 math alone has **3,897 pairs**, versus approximately **1,400 across the entire pt family**.
- Its actual mix is **61.3% guard / 38.7% shorten**, not 1:1.
- At 16 pairs/step, math alone receives approximately **244 steps**. The same pt pair pool took approximately **86 joint steps**.
- The underlying bidax5s recipe already makes math longer. Its final result is **23.6 / +1.3**, with math **22% longer**.
- The related 16-pair/step bidax5m run reaches **11.3 / −3.6**, with math **56% longer / −19.2**—strikingly similar to the lpt6-math failure.

This implicates the pair/exposure recipe, but does **not** prove that “more pairs are bad.” Additional pairs change selection, chosen/rejected length distributions, source mix, and optimizer exposure simultaneously.

The joint pair-rule screens also show that objectives can move the operating point dramatically:

- Plain bida, bidax5q-final: **46.7 / −0.5**
- l2s50e-final: **53.5 / −2.7**
- simright-final: **28.1 / +1.7**
- route-final: **9.4 / −0.7**

Conversely, **l2s50e is not intrinsically superior**: its joint result loses substantially more accuracy than plain bida. The successful pt system is a recipe bundle, not proof of one winning pair rule.

### Precision deserves a separate recommendation

Both existing families used bf16, so precision **cannot explain their difference**.

It is nevertheless worth changing for Jasmin. The closest precision-only comparison, bidax4-final versus bidax6p-final, changes **52.7 / −3.5 → 48.8 / −2.3**. Some accuracy recovery is associated with reduced compression; this is not evidence of a free +1.2 points at matched saving.

Also, the structured run description specifies **float32 weights plus fp16 autocast**, not necessarily full-float32 computation everywhere. Record those distinctions explicitly.

### Training ablations eventually needed

A small controlled follow-up should hold data, grouping, seed and exposure fixed while varying:

1. NLL **0 versus 1**.
2. Constant versus cosine schedule, recording both matched-step and approximately matched-integrated-LR checkpoints.
3. pt-style versus bida-style pair construction with **pair count/exposure controlled**.
4. Pooled tools versus independently trained slices on the same union of examples.

Existing-adapter merges cannot answer those causal questions.

---

# 3. Ranked screen plan: at most ten scored configurations

## Notation and interpretation

- **P = (math, code, instructions, pooled tools)**.
- **L = (math, code, instructions, one-function, several-functions, no-fit)**.
- Every L coefficient refers to the **two-seed weight soup**: coefficient .5 means .25 on each seed’s delta.
- Unlisted adapters have zero strength.

Predictions below are **planning estimates, not fitted performance guarantees**. Before the half-strength results, allow roughly **±4 saving points and ±1.5 accuracy points**, sometimes more for cross-family interactions. The “worst parts” column identifies expected failure locations.

The aim is not ten chances to find a noisy winner. Each contrast should change a Jasmin decision.

| Priority | Exact configuration | Hypothesis and Jasmin decision | Predicted saved / accuracy; worst parts |
|---|---|---|---|
| **1** | **P=(.30,0,.60,.45); L=0** | **Is pt-code needed?** Its deployed solo loses 2.5 points overall. Compare directly with pt-mid, while allowing that its merge interactions may be beneficial. Determines whether code needs a smaller dose/earlier checkpoint or can be omitted. | **51 / −1.5**. Math roughly −3 to −6; no-fit −5 to −8. |
| **2** | **P=(.15,0,.65,.65); L=0** | **Can compression be redistributed away from math/code?** Tests whether instructions/tools can deliver the shared compression with better correctness. Establishes a more compressed repair base. | **55 / −2.0**. Math and no-fit remain likely bottlenecks; stronger tools may also hurt code. |
| **3** | **P=0; L=(0,0,.50,.50,.50,0)** | Retain **C**, now explicitly a **compression-interaction test**, not a six-type prototype. Determines whether lpt6 supplies useful complementarity at all. | **40 / −2.5**. Math approximately −4 to −7; no-fit −5 to −9. |
| **4** | **P=(.30,0,.60,0); L=(0,0,0,.50,.50,0)** | Replace pooled pt-tools in #1 with split callable-tools directions. Approximately dose-matched: the split pair’s aggregate saving null is **26.4%**, versus pt-tools@.45’s **28.6%**. Tests practical value of splitting. | **50 / −2.5**. No-fit likely worst, approximately −7 to −12; math also exposed. |
| **5** | **P=(.30,0,0,.45); L=(0,0,.65,0,0,0)** | Replace pt-instructions in #1. Approximate matched solo compression: **22.6% predicted** for lpt6-instructions@.65 versus **23.1% measured** for pt-instructions@.60. Tests whether the lpt instruction direction is intrinsically worse at similar compression. | **51 / −2.8**. Math and instructions likely worse than #1. |
| **6** | **P=(.30,0,.60,.45); L=(0,0,0,0,0,.25)** | **Can weak no-fit correction be useful inside a call-capable base?** Tests contextual suppression of the no-call bias. Determines whether a separate no-fit adapter deserves any deployment strength. | **52 / −2.1**. No-fit may gain about 3 points versus #1, but simple/multiple may lose about 5/4 points. Run only under the gate below. |
| **7** | **P=(.30,0,0,0); L=(0,0,.50,.50,.50,0)** | Add **pt-math**, not lpt6-math, to C. Tests whether a genuinely shortening math-trained direction complements the lpt core. Informs math pair/checkpoint selection. | **47 / −3.2**. Math and no-fit remain worst; improvement over the accuracy-additive null would be informative. |
| **8** | **P=0; L=(0,.50,.50,.50,.50,0)** | Add lpt6-code to C. **Does an apparently useless solo become useful in context?** Prevents prematurely concluding that all adapters must shorten their own type to be retained. | **39 / −2.8**. Little saving benefit expected; instructions may worsen. |
| **9** | **Base #2, plus new repair LoRA @1.0, step 10** | Test whether a redistributed compression base has a better repair trajectory than pt-mid. | **52 / −1.6**. Math likely remains limiting; no-fit should improve. |
| **10** | **Same repair run, repair @1.0, step 20** | Estimate the early repair slope and whether it approaches the actual target rather than merely restoring length. | **49 / −0.5**. Math/code likely residual bottlenecks. |

### Why #1 is unusually informative

Two equally simple counterfactuals disagree sharply:

- Adding the remaining solos independently predicts **51.6 / approximately −3.3**.
- Removing code’s measured effect while retaining pt-mid’s interactions predicts **50.0 / −0.1**.

That disagreement is precisely the information the ablation supplies. I would not assume either endpoint is correct.

### Exact repair experiment for #9–10

Use #2 as the merged base and reference, with:

- A fresh LoRA, DPO β=.1, NLL=1.
- Float32 master weights and optimizer state; record autocast separately.
- LR **2e−5**, 5% warm-up, cosine over a **40-step** pilot horizon.
- Batch **16 pairs**.
- Training-only answers sampled from that base.
- A **1:1 guard/right** mixture: shortest-right versus shortest finished-wrong; and right versus wrong of similar length, balancing whether the correct response is longer.
- Score steps **10 and 20**; saving a checkpoint is not an additional evaluation screen.

This is an explicitly specified pilot, not a claim that every undocumented historical repair setting was identical.

Gate repair on #2 being a reasonable base: approximately **≥53% saved, accuracy ≥−3**, and no catastrophic part collapse. Otherwise do not spend these two screens automatically.

Historical repair results justify caution: rpC-20 reaches **50.5 / −1.8**, whereas rpC-80 reaches **43.4 / +0.6**. Repair often buys accuracy by returning tokens. It has not yet demonstrated the required operating point.

## How the half-strength solos change the list

First replace every extrapolated lpt response with its measured @.5 response. For each merge, recompute the partwise log-length null and accuracy-additive null.

### A. Math remains longer and less accurate

If math@.5 is still clearly Pareto-inferior—longer and worse on math—**keep it out of all merges**. Do not revive P/Pplus.

If instead math@.5 retains substantially more accuracy than expected, particularly math loss no worse than about **−2 points**, replace #8 with:

- **P=0; L=(.25,0,.50,.50,.50,0).**

That tests a *small* contextual protector. Under current full-dose extrapolation it would be only **34.5 / −3.7**; the new half-dose evidence must materially overturn that pessimistic prior.

### B. No-fit’s callable damage persists

At .5, linear extrapolation implies approximately **−10.3/−8.3** on callable parts. If anything close to that remains, **cancel #6**. It is not worth spending a screen to reconfirm obvious bias.

Run #6 only if .5 shows a pronounced safe region—for example:

- Neither callable part loses more than about **3 points**;
- no-fit improves by roughly **4 points or more**;
- weighted BFCL accuracy is near-neutral or positive.

These are screening gates, not claims of statistical significance. Inspect paired uncertainty.

Replace cancelled #6 with a lower-dose pt-code contrast:

- **P=(.30,.15,.60,.45); L=0**.

Expect approximately **52% saved**, with accuracy plausibly between #1 and pt-mid. This tests whether code has a useful low-dose window rather than only an on/off choice.

### C. The shortening core has a safe half-dose region

If C performs near **40% saved with accuracy ≥−1**, replace a low-value remaining slot with:

- **P=0; L=(0,0,.70,.70,.70,0)**.

Its full-solo log-additive saving forecast is **50.6%**. Accuracy is the uncertainty: the current additive forecast is about **−4.1**, so escalation is justified only by measured protection at .5.

If C is already worse than approximately **−3 at ≤40% saving**, deprioritize lpt-based combinations; focus on pt and repair.

### D. Code has no half-dose benefit

If code@.5 neither shortens nor improves code accuracy, and harms other parts, cancel #8 unless C itself shows unusually strong complementarity. “No solo compression” alone is not enough to cancel it; **no accuracy benefit plus spill-over harm** is.

### E. Evidence points to soup instability

If the team suspects incompatible seed updates, replace #7 and #8 with **C-s0 and C-s1**, respectively:

- Instructions, one-function and several-functions each **.5**, using only the named seed.

Compare both with soup-C. This separates seed-specific behavior from weight-averaging interactions without training. One seed alone would not settle it.

These are **replacements within ten slots**, not additional screens.

## Execution and selection discipline

- Approximately **8 merge GPU-hours**, plus repair training and two checkpoint screens: roughly **11.5 GPU-hours** before any cancellations.
- Preserve per-question outputs, correctness, total/thinking/answer tokens, truncation, and call/no-call decisions.
- Bootstrap **questions**, preserving pairing across baseline/solos/merges; generations from the same question are not independent questions.
- Use the real four-benchmark score, with BFCL correctly pooled.
- Confirm a finalist on untouched evaluation data after freezing the recipe. Do not declare success from one noisy **50.x / −0.x** screen.
- None of the evaluation responses may enter preference pairs or repair training.

---

# 4. Current Jasmin recipe

## Groups: four by default; six must earn their deployment

My default deployed structure is:

1. Math
2. Code
3. Instructions
4. **Pooled tools**

For the best informative training campaign, train **seven adapters from scratch**: the three non-tool adapters, one pooled-tools adapter, and three tool-slice challengers.

That permits a clean four-versus-six deployment comparison. **Summing three independently trained tool slices does not recreate a pooled-tools adapter.** If the team trains only six slices, it loses the clean pooled control.

Move to six deployed groups only if #4/#6 or the matched Jasmin challengers demonstrate useful independent control without the no-call/call trade-off.

## Pairs: preserve the successful pt baseline before adding fixes

Use original-Qwen3 training-pool answers, never evaluation answers.

| Group | Initial rule |
|---|---|
| Math | Exact pt l2s50e eligibility and length-ratio rule: shortest correct versus longest wrong; and shortest correct versus sufficiently longer correct. Preserve the existing ≥50%-solved filter and easy-question exception. **Do not copy the blanket medium-math shortening suppression.** |
| Code | Same pt rule, with executable correctness and completion checks. Audit whether pair selection disproportionately rewards long, marginally correct solutions or penalizes truncation artifacts. |
| Instructions | Same pt rule initially. This is a strong candidate source of shared compression, but own-type constraints must remain checked. |
| Pooled tools | Exact pt rule **including no-fit protection**, retaining both callable and non-callable examples. |
| Split-tool challengers | First use the corresponding slices of that same pool/rule for a controlled comparison. A production no-fit specialist should not be deployed if its correctness comes from suppressing calls globally. |

If splitting proves useful but no-fit still biases action selection, the next revision should include **shared callable calibration pairs or anchors** in the no-fit training stream. For example, a 75% target-slice / 25% shared-BFCL mixture is a reasonable pilot—not an established optimum.

Audit, per group:

- Unique questions and pairs;
- solve-rate distribution;
- chosen/rejected correctness and length;
- shorten/guard proportions;
- source mixture;
- truncated or malformed outputs.

“Same pair-rule name” is not sufficient. The pt and joint l2s50e runs already show why.

## Objective, optimization and precision

Initial controlled Jasmin settings:

- Qwen3-0.6B base; same LoRA targets/rank as the existing runs.
- **DPO β=.1, NLL anchor=1.0**.
- **Constant LR 2e−5** for the first pt replication.
- Batch **16 pairs**, initially **one pass**.
- Float32 master parameters, adapter updates and optimizer state; stable log-probability computation.
- Explicitly record whether autocast is disabled or fp16.
- Merge in float32; evaluate with the prescribed **bf16 loading**.

I would not simultaneously switch to cosine just because the successful joint reference used it. First preserve pt’s optimization recipe while testing precision and checkpoints. If checkpoint curves show overshoot, test **1e−5 or warm-up/cosine** in a controlled challenger.

NLL=1 is a defensible default, not a proven causal explanation for pt’s superiority.

## Steps and checkpoint selection

Do **not** give each small pt slice 244 or 339 steps simply because another recipe used that many.

For group \(g\), start with approximately

\[
T_g=\lceil N_g/16\rceil
\]

steps, saving **0, 10, 20, …, final**, including final checkpoints below ten steps.

Choose checkpoints using a **vector of own-type benefit and spill-over**, not a single overall solo score:

1. Own-type saving and accuracy.
2. Full-test-weighted accuracy spill-over.
3. Worst-part damage, especially math and callable/no-fit confusion.
4. Contribution at a plausible deployment dose.
5. Marginal value when merged with the other selected directions.

Retain two or three Pareto-efficient checkpoints per adapter. Prefer the earliest checkpoint that delivers the useful effect; do not select “final” by convention.

Use own-type ×4 / other-types ×2 for inexpensive initial assessment, but maintain score weights independent of sampling replication. Approximate own-type losses beyond **2 points**, or another-part losses beyond **3 points**, should trigger scrutiny—not automatic rejection from one noisy screen.

For code in particular, #1 and #8 determine whether weak solo results indicate an unnecessary adapter or a useful interaction partner.

## Merge method and starting strengths

Use ordinary task arithmetic on **effective weight deltas**:

\[
W=W_0+\sum_g \lambda_g\Delta W_g.
\]

Do not separately average LoRA factors in a way that changes the intended delta. No evidence yet justifies adding a more elaborate merge method as another confound.

A provisional four-group starting vector is:

- **Math .20, code .15, instructions .60, tools .55.**

It is a starting point, not a forecast for newly trained adapters. Float32 training and checkpoint choice may change scale substantially.

Screen-driven changes:

- #1 succeeds → code strength **0 or very small**, or an earlier code checkpoint.
- #2 succeeds → favor instructions/tools over math.
- #4 succeeds at matched compression → retain split callable-tools controls.
- #6 succeeds without callable harm → permit a small independent no-fit coefficient.
- #5 favors pt → retain the pt instruction recipe rather than importing lpt settings.
- Strong checkpoint effects → choose training steps before expanding the merge-strength search.

## Repair: optional, not the default route to the target

Train repair only after selecting the best compression base. Use that base’s **training-pool** responses and early checkpoints.

Adopt it only if it improves the frontier **relative to simply reducing merge strength**. The relevant win is not “accuracy rose”; it is “accuracy rose while maintaining the required compression.”

At present:

- pt-mid: **55.0 / −2.6**
- rpC-20: **50.5 / −1.8**
- rpC-80: **43.4 / +0.6**
- best joint reference: **46.7 / −0.5**

No supplied result establishes **>50% saving with <1 point lost**. The gap is modest enough to justify Jasmin, but large enough that success should not be assumed.

## Assessment of the team’s candidate

**I support it, with two modifications.**

Redoing pt in float32, preserving its data/pair rule, and saving checkpoints is the most defensible next training family. It tests a promising recipe rather than trying to rescue clearly pathological lpt6 directions.

But:

1. **Train a pooled-tools adapter as well as the three slices.** Six slices alone cannot answer four versus six fairly.
2. **Select checkpoints for merge usefulness, including spill-over—not maximum own-type compression.**

The updated evidence favors **shared compression with targeted correctness calibration**. Jasmin’s opportunity is to learn that combination more cleanly, not to make six stronger independent “think less” adapters.