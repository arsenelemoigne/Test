# Recommendation

**I would train a strong joint DPO adapter, add a small instruction-trained compression adapter only if its cross-type effects are benign, then apply one short, balanced correctness-repair round.** I would average two training seeds in weight space and select from a small, predefined set of strengths.

I would **not** deploy six independent specialists, assume that adapters affect only their training type, or repeat the previous unrestricted repair recipe.

The target is plausible but unproved. **None of the 144 scored records achieves both >50% saving and >−1 accuracy point, even on the screen.** A credible plan must therefore test a new hypothesis, not present an existing checkpoint as a solution.

---

# 1. What the evidence actually says

Throughout, a result such as **55.0 / −2.6** means percentage of generated tokens saved / accuracy-point change. Unless explicitly stated otherwise, these are the supplied screen measurements.

## 1.1 First fix the measurement

The official accuracy change is

\[
\Delta A=\frac{\Delta A_{\rm math}+\Delta A_{\rm code}+\Delta A_{\rm instr}
+(400\Delta A_{\rm onefn}+200\Delta A_{\rm several}+240\Delta A_{\rm nofit})/840}{4}.
\]

Saving uses the geometric mean of the **four benchmark total-output token ratios**. Within BFCL, pool token totals, not the three saving percentages.

Several important audit findings:

- The raw baseline contains **2,759 questions**, but only **719** have `in_screen720=True`: multiple-function BFCL has 119, the other parts 120. Resolve this before another screen.
- Some reported aggregate accuracies do not reproduce the prescribed BFCL weighting. From rounded part scores:
  - `lpt6-nofit`: reported **−1.9**, officially weighted approximately **−2.40**.
  - `rpB-40`: **+0.3 → +0.10**.
  - `rpC-80`: **+0.6 → +0.43**.
  - `bidax5q-final`: **−0.5 → −0.59**.
  - `budget-final`: **−1.3 → −1.42**.
- The largest discrepancy is **0.50 point**, too large to explain by one-decimal rounding. Equal averaging of BFCL parts explains much, but not all, of the discrepancy. The aggregation implementation is not provided.
- The raw file contains **thinking-token summaries**, not total generated tokens or individual answer lengths. It cannot independently verify official savings, truncation rates, or within-question length–correctness relationships.
- The stated screen/full discrepancy is approximately **+2 saving points and +0.2 accuracy point**. The identifiable `bidax2-final` versus full record differs by only **+0.6 / +0.3**; the exact checkpoint correspondence is unspecified. Treat the two-point saving correction as a planning allowance, not a calibrated universal offset.
- The fitted reference curve describes the historical trade-off, **not the objective or an attainable frontier**. Selecting the largest distance above it rewards some candidates that remain far from 50% saving.
- Commands include queued but unscored experiments. **No lpt6 merge was measured.** Its merge commands are not results.
- `track.py`, the GRPO implementation, and the evaluator are absent. Consequently, some stage-level defaults cannot be recovered from the supplied standalone trainer defaults. New runs should specify everything explicitly.

The original officially weighted accuracy is approximately **70.23%**. The goal therefore requires accuracy **strictly above about 69.23%**, alongside a total-token ratio strictly below 0.5.

## 1.2 Raw model: where there is room, and where thinking matters

| Part | Baseline accuracy | Mean / median thinking tokens | 8/8 solved | 0/8 solved | Thinking tokens in longest decile |
|---|---:|---:|---:|---:|---:|
| Math | 88.70% | 677 / 425 | 70.1% | 3.2% | 34.9% |
| Code | 52.45% | 2,079 / 1,193 | 29.1% | 30.2% | 33.1% |
| Instructions | 58.73% | 433 / 320 | 34.0% | 20.0% | 30.4% |
| One function | 82.16% | 171 / 145 | 66.5% | 7.5% | 21.0% |
| Multiple functions | 76.06% | 192 / 171 | 56.0% | 8.5% | 20.4% |
| No fitting function | 83.39% | 186 / 140 | 72.5% | 10.0% | 26.4% |

Important implications:

- **Code is the biggest wasted-thinking opportunity**, but not an easy source of correctness pairs. Questions solved 0/8 consume **50.4% of code thinking tokens**; the longest decile has only **5.6% accuracy**. Resample selected difficult training questions to find valid winners; do not label long failures as desirable.
- **Math cannot simply stop thinking.** Its always-solved questions constitute 70.1% of questions but only 47.3% of thinking tokens. Removing only easy-example verbosity will not capture the entire opportunity.
- The historical math exclusion band, **470–905 thinking tokens**, covers **24.9% of math questions and 22.7% of thinking tokens**, with **87.9% accuracy**. That is a substantial, largely successful region—not an obvious region requiring blanket lengthening.
- Long thinking and low solve rate correlate, especially in code and math. This is **not causal evidence that cutting a long answer preserves correctness**.

Thinking-off reinforces the need for differentiated training:

| Part | Full-test accuracy change with thinking off |
|---|---:|
| Math | −68.1 |
| Code | −11.6 |
| Instructions | −1.2 |
| One function | −10.8 |
| Multiple functions | −20.8 |
| No fitting function | +6.9 |

Overall, thinking-off loses approximately **22.3 points**. It is forbidden anyway. Its useful lesson is narrower: preserve calculation and tool selection; remove redundant instruction planning and avoid overthinking no-fit decisions **without teaching universal abstention**.

## 1.3 What each per-type adapter does

These matrices are the most useful evidence in the pack. Every cell is **saving / accuracy change**.

### October `pt`: four adapters

| Adapter, strength | Math | Code | Instructions | One function | Multiple | No-fit | Overall |
|---|---|---|---|---|---|---|---|
| Math, 1.0 | 59/−4.8 | 51/−1.0 | 26/+0.6 | 21/−1.7 | 24/−1.3 | 25/−6.0 | 41.7/−2.1 |
| Code, 1.0 | 20/−1.3 | 25/−0.2 | 7/−1.3 | 14/−1.5 | 14/−2.3 | 11/−3.5 | 16.5/−1.3 |
| Instructions, 1.0 | 34/−2.7 | 42/+0.2 | 40/+0.6 | 28/−1.9 | 29/−1.3 | 29/−5.6 | 36.1/−1.2 |
| Tools, 1.0 | 53/−4.0 | 59/−4.6 | 44/−3.7 | 38/+1.5 | 43/+2.5 | 45/−2.3 | 50.1/−2.9 |
| Math, 0.30 | 20/−1.5 | 13/+0.8 | 4/−1.5 | 9/−1.5 | 10/−2.3 | 11/−1.9 | 11.9/−1.0 |
| Code, 0.45 | 9/−2.5 | 14/−2.3 | 11/−3.7 | 4/−0.8 | 7/−2.1 | 5/−1.2 | 10.0/−2.5 |
| Instructions, 0.60 | 24/−0.8 | 20/−0.2 | 27/−1.5 | 20/−0.6 | 20/−2.7 | 22/−3.3 | 23.1/−1.2 |
| Tools, 0.45 | 27/−2.9 | 32/+0.4 | 28/−0.8 | 25/−2.5 | 28/−0.2 | 28/−0.8 | 28.6/−1.1 |

Training used roughly **1,400 filled pairs out of 3,200 requested**, matched by solve-rate × thinking-length cells; rank 16, alpha 32, all seven attention/MLP projections, NLL 1, constant learning rate \(2\times10^{-5}\), BF16, one pass.

**Interpretation:** these are not task-isolated adapters. Tools training compresses code more than tools; instruction training compresses code almost as much as instructions. Math has a stronger own-type effect, but large spillover.

### November `lpt6`: six adapters

| Adapter, strength | Math | Code | Instructions | One function | Multiple | No-fit | Overall |
|---|---|---|---|---|---|---|---|
| Math, 1.0 | −78/−16.2 | −17/+2.3 | −39/+1.5 | −23/−1.7 | −23/0 | −29/+2.5 | −38.1/−3.1 |
| Math, 0.5 | −26/−2.9 | −3/−0.4 | −15/+1.5 | −6/−1.2 | −6/−0.8 | −6/+1.9 | −12.2/−0.5 |
| Code, 1.0 | −3/−1.0 | −5/+0.4 | −9/−3.1 | 4/−0.4 | 3/−2.3 | 2/+1.5 | −3.2/−1.0 |
| Code, 0.5 | 6/−0.4 | −2/+2.3 | −10/+0.2 | 1/+0.2 | 6/−3.2 | 0/+0.2 | −0.8/+0.3 |
| Instructions, 1.0 | 27/−3.5 | 32/−1.9 | 43/−4.4 | 26/0 | 26/−1.1 | 28/−3.5 | 32.6/−2.8 |
| Instructions, 0.5 | 19/+0.4 | 18/+0.8 | 19/−0.6 | 14/−2.1 | 18/0 | 12/−1.7 | 17.8/−0.2 |
| One function, 1.0 | 34/−4.2 | 41/−0.8 | 23/+1.0 | 34/−1.2 | 38/0 | 35/−7.3 | 33.7/−1.7 |
| One function, 0.5 | 20/−0.8 | 17/+0.2 | 4/−1.0 | 18/+0.2 | 20/−0.2 | 18/−3.3 | 15.3/−0.7 |
| Multiple, 1.0 | 18/−1.5 | 24/−2.5 | 9/0 | 21/−1.5 | 26/0 | 18/−2.9 | 18.3/−1.4 |
| Multiple, 0.5 | 13/+0.8 | 8/+0.6 | 1/−1.9 | 8/0 | 14/+0.6 | 10/−2.1 | 8.3/−0.2 |
| No-fit, 1.0 | 11/+0.2 | 9/+1.0 | 3/−0.4 | 18/−20.6 | 17/−16.6 | 36/+11.7 | 11.9/−1.9 |
| No-fit, 0.5 | 4/−0.8 | 2/+0.8 | −3/−1.7 | 10/−4.0 | 10/−3.4 | 23/+4.8 | 4.5/−0.6 |

This family used unmatched, more numerous `bidax5s` pairs, NLL 0, cosine scheduling, BF16, and two seeds averaged. Math alone had **3,897 pairs: 1,510 shorten and 2,387 guard**.

The large effects are clear:

1. **The math “protection” adapter is harmful.** It increases math tokens 78% while losing 16.2 points. More thinking is not necessarily better reasoning.
2. **The no-fit adapter learns an abstention bias.** At strength 1 it gains 11.7 on no-fit but loses 20.6/16.6 on relevant tools. Officially pooled BFCL accuracy falls **10.42 points**. Even half strength loses **1.34 BFCL points**.
3. **Code-only training is not a reliable compression direction.**
4. **Half-strength instruction training is the most promising inexpensive supplementary direction**, but its −0.2 overall score is within noise; full strength is clearly less attractive.

A descriptive SVD of the six-part log token-ratio matrices puts **96.7% of the October family’s squared magnitude in one direction**, and **89.4% for November**. This is not a mechanistic proof, but it quantifies the dominant **shared length-control effect**, rather than six independent skills.

## 1.4 Merging: useful, nonlinear, and not a substitute for evaluation

| October merge | Math/code/instruction/tools strengths | Overall |
|---|---|---:|
| Softer | .20/.25/.35/.25 | 39.2/−0.8 |
| Soft | .25/.35/.45/.35 | 48.4/−1.9 |
| Mid | .30/.45/.60/.45 | 55.0/−2.6 |
| No code | .30/0/.60/.45 | 50.8/−2.7 |
| Shift | .15/0/.65/.65 | 54.4/−3.9 |

For `pt-mid`:

- Multiplying the four measured solo token ratios predicts **56.5% saving**, close to the observed **55.0%**.
- Adding solo accuracy changes predicts **−5.8 points**, versus observed **−2.6**.
- But this apparent compensation is uneven:
  - Instruction accuracy: summed **−7.5**, merged **+0.6**.
  - Multiple-function accuracy: summed **−7.3**, merged **+1.7**.
  - Code: summed **−1.3**, merged **−3.5**.
  - No-fit: summed **−7.2**, merged **−7.1**.

Thus, **neither accuracy addition nor per-part token-ratio multiplication is reliable**. The aggregate token approximation hides BFCL discrepancies of roughly ten saving points.

Removing code costs **4.2 saving points** without a clear overall accuracy benefit. Increasing instruction/tools weight while reducing math does not protect math: `pt-shift` still loses **5.4 math points**.

Earlier stronger merges on a different 240-question set reached **70.5–77.2% saving**, but lost **6.5–9.2 points**. They are evidence of controllable compression, not of target attainment.

For the unmeasured lpt6 merges, even crude solo-ratio multiplication gives only approximately:

- instruction/one-function/multiple at half strength: **36% saving**;
- adding half-strength math and no-fit: **32%**;
- all six at half strength: **31%**.

These are **not forecasts**. They merely show why spending the budget on a large six-way search is poorly motivated.

## 1.5 Repairs recover accuracy—but usually buy back tokens

| Repair | Saving | Reported accuracy | Approx. official pooled accuracy |
|---|---:|---:|---:|
| Starting `pt-mid` | 55.0 | −2.6 | −2.58 |
| `rpA-40`, low LR | 52.3 | −2.8 | approximately −2.8 |
| `rpB-20` | 48.7 | −1.1 | −1.20 |
| `rpB-40` | 43.4 | +0.3 | +0.10 |
| `rpC-20` | 50.5 | −1.8 | −1.83 |
| `rpC-40` | 51.2 | −2.1 | −2.19 |
| `rpC-80` | 43.4 | +0.6 | +0.43 |

The successful endpoints recover **2.7–3.0 accuracy points while surrendering 11.6 saving points**.

Repair is not uniformly restoring skill. From `pt-mid` to `rpC-80`, no-fit improves **12.5 points**, while one-function accuracy deteriorates **2.3 points**. Some recovery is a change in calling/abstention behaviour.

The original repair builder:

- uses the compressed model’s own answers;
- mixes shortest-right > shortest-finished-wrong guards with approximately length-matched correctness pairs;
- intends half the correctness winners to be longer;
- permits up to three pairs per question;
- interleaves types.

However, shortages can change the intended balance, and unrestricted guards can strongly favour longer answers. **Repair needs shortening replay, explicit length-direction balance, and relevant-tool protection.**

## 1.6 Joint runs and pair rules

### Joint training settings

| Setting or data change | Main evidence |
|---|---|
| Early Run A, protected math/no-fit, NLL anchor | Roughly 32–33% saving near neutral accuracy; final 32.5/+0.4 |
| Run A, constant LR | Saving falls from 34.3 at step 150 to 23.4 at final |
| Gentle median-right math shortening, Run B | Approximately 28–31% saving near neutral accuracy |
| Extra long-tool weighting | 48.4/−2.6; instruction accuracy −6.0 |
| Correctness stage followed by October recipe | 29.5/+1.6, well below token target |
| Initial bida copy | Final 51.8/−2.7; second seed 51.0/−3.9 |
| Repeated initial copy | 54.5/−3.9; metadata calls it a repeat, but command/pool differences prevent a clean “GPU noise only” attribution |
| Added hard no-fit data | 52.7/−3.5; no-fit improves relative to earlier copies, other damage remains |
| FP32-base precision change only | 48.8/−2.3 |
| Exact precision and schedule, without added math | Best middle checkpoint 41.7/−1.0; final 37.8/−1.7 |
| Exact settings plus added math, `bidax5q` | Final 46.7/−0.5 |
| Same recipe, second seed | 48.5/−2.2 |
| Additional code/instruction data | Middle checkpoints 43–45% saving, −0.6 to −0.8; final 43.5/−2.5 |
| Math-band exclusion plus overcut pairs | 23.6/+1.3; math **22% longer** |
| Stretched math-fix schedule | 11.3/−3.6; math **56% longer and −19.2 points** |
| Two-pass/longer training | Approximately 38–42% saving; strong relevant-tool/no-fit imbalance |

The strongest single-adapter recipe is therefore **bida-style DPO with the improved math pool and exact numerical/scheduling settings**, but its two endpoints average only **47.6% saving and about −1.42 officially weighted points**. The favourable seed alone is not a robust baseline.

### Pair-rule experiments

The implemented pair meanings matter:

- **Shorten:** correct > correct, usually shortest versus longest.
- **Bida guard:** shortest correct > longest incorrect, capped or unclosed included.
- **Right:** correct > finished incorrect of similar length.
- **DAST hard:** longest finished incorrect > shortest finished incorrect.
- **Chain guard:** shortest correct > shortest finished incorrect.

| Pair-rule variant | Final reported result | Interpretation |
|---|---:|---|
| DAST | 33.1/−2.3 | Wrong-over-wrong “keep trying” is unhelpful here |
| Add hard pairs | 13.8/−0.1 | Preserves accuracy largely by abandoning compression |
| Routing plus hard/correctness pairs | 9.4/−0.7 | Same problem |
| Difficulty budget | 49.0/−1.3 | Closest alternative joint rule, not a demonstrated gain over seed noise |
| Drop narrowest 15% of pairs | 53.3/−2.6 | More compression, more loss |
| Add similar-length correctness | 28.1/+1.7 | Strong accuracy/length trade-off, not free repair |
| Correctness only | −4.6/+0.8 | Length increases |
| Correctness then compression | 16.5/+1.4 | Poor compression |
| Extra correctness on math/no-fit | 41.7/−1.0 | Safer but too long |
| Ada-R1-style top-two losers | 65.3/−6.8 | Excessive shortening pressure |
| October threshold/easy-ratio rule, large unmatched set | 53.5/−2.7 | Matching/anchor differences matter; rule alone does not solve accuracy |
| Up to 2× shorten pairs | 52.4/−3.0 | More pressure is not the missing ingredient |
| Chain-1, 40:60 shorten/short-wrong guard | 17.8/+0.8 | Too much length restoration |

A critical command detail: the scored `simright` run’s newest command uses **long2short plus additional right pairs**, not merely the standalone `simright` recipe. Likewise, the scored routing variant retains the original shortening rule and adds hard/correctness pairs. Names alone would misdescribe these experiments.

The three GRPO settings—length penalty, fourfold penalty, and no KL—produce approximately **−7% to +2% saving**, with no useful positive trajectory. I would not spend this budget continuing them.

Public full-test references reach **47.0/−0.78** and **43.2/−0.85**. They support the difficulty of the target, not its impossibility. The teacher-hint approach and test-time gate are outside this project’s permitted route.

### Signal versus noise

**Strong signal:** global adapter transfer; math/no-fit specialist failure; excess shortening harming accuracy; strong correctness pressure restoring length; numerical settings and training duration matter.

**Weak signal:** sub-point improvements, the best isolated checkpoint, half-strength instruction accuracy gains, and apparent merge accuracy “synergy.” Screen noise is about ±1 overall/±3–4 per part, and the two `bidax5q` seeds differ by **1.7 accuracy points**. The 144 measurements are correlated and heavily selected.

---

# 2. Concrete strategy

## 2.1 Model structure

Use **three functional components**, not six deployed specialists:

1. **J: joint compression/correctness adapter**, trained on all six types.
2. **I: instruction-only supplementary adapter**, intended as a weak, broadly transferring compression direction.
3. **R: balanced repair adapter**, trained on the selected J+I model.

Train **two seeds per component**. The instruction component is optional: it must earn inclusion through measured cross-type performance.

All use rank 16, alpha 32, dropout 0, no trainable bias, targeting `q/k/v/o/gate/up/down` in all 28 layers. Freeze embeddings, tied output head, norms, and base parameters. No architecture change, routing, prompt modification, or inference-time stopping intervention.

## 2.2 Data acquisition and validation

Start with approximately **16,000 non-benchmark training questions**:

- Math: 4,000, initially 70% GSM8K / 30% relevant MetaMath.
- Code: 4,500 KodCode.
- Instructions: 3,500 externally sourced, mechanically checkable instruction tasks.
- One function: 1,600.
- Multiple functions: 800.
- No-fit: 1,600, deliberately overgenerated because trustworthy mixed-success examples are scarce.

Generate **four original-model answers per question** with thinking on and normal sampling. Reserve up to **4,000 question top-ups of four more answers**, concentrating on:

- solve-rate thresholds;
- code questions with plausible but inconsistent solutions;
- relevant/no-fit boundary cases;
- underfilled pair quotas.

Do not equate 4/4 with genuinely easy. For math and no-fit shortening candidates, require eight samples where feasible; final thresholds below then mean 6/8 rather than an uncertain 3/4.

Before generation, reserve **1,200 additional non-benchmark questions**: 600 development and 600 locked confirmation, 100 per type in each. Split by normalized question/source family, not individual completions.

Contamination checks must cover all 2,759 benchmark questions, tool schemas, and near-duplicates. The supplied builder ignores eight-word sequences shared by at least three evaluation questions; I would audit that exemption and separately check substantive text and schemas. No benchmark answers enter training.

Use dataset labels, execution tests, schema/argument checks, and deterministic instruction validators to grade **the model’s own complete answers**. Do not substitute dataset-written solutions as chosen completions.

## 2.3 Joint pair rules and counts

The following is the **planned filled dataset**, not a claim about available pairs:

| Type | Total | Shorten S | Guard G | Length-balanced right R | Minimum solve rate for S |
|---|---:|---:|---:|---:|---:|
| Math | 1,344 | 576 | 640 | 128 | .75 |
| Code | 1,536 | 736 | 736 | 64 | .50 |
| Instructions | 1,280 | 608 | 608 | 64 | .50 |
| One function | 608 | 288 | 288 | 32 | .50 |
| Multiple functions | 304 | 128 | 144 | 32 | .75 |
| No-fit | 352 | 144 | 160 | 48 | .75 |
| **Total** | **5,424** | **2,480** | **2,576** | **368** | |

This gives exactly **339 updates at 16 pairs/update**.

Rules:

- **S:** shortest verified correct > longest verified correct; loser/winner total-output length ratio **≥1.5**. For math with at least three correct samples, use the lower-median correct winner rather than the shortest, retaining the same ratio requirement.
- Do not use the October **1.2 easy ratio** initially; marginal correct/correct differences are weak evidence.
- **G:** shortest verified correct > longest incorrect, including capped/unclosed failures. Permit these at any nonzero solve rate. Do not remove them merely because a math question is medium difficulty.
- **R:** correct > finished incorrect with **max/min total length ≤1.25**. Enforce approximately 50:50 longer/shorter correct winners within each type.
- No chosen incorrect responses; **zero wrong-over-wrong hard pairs**.
- At most **two distinct pairs per question**, at most one of each kind.
- All chosen responses must contain a nonempty, closed thinking block and a complete answer.
- Skip an entire overlength pair; never truncate a chosen answer to fit training.
- Cap shorten/correctness ratios after selection: approximately **.75 math, .92 code, .91 instructions, .90 one-function, .73 multiple, .69 no-fit**.

Use coarse solve bins \(0,\;(0,.5),\;[.5,1),\;1\) and four thinking-length bins as **coverage diagnostics and soft sampling targets**. Borrow within the same solve bin before adjacent solve bins. Do not trim every type to the weakest cell’s fill rate.

For code, up to 25% of its guard quota may deliberately target long/capped failures. For tools, balance call/no-call decisions across schema families; prevent one source or function template from dominating. Do not match the exact test questions’ fine-grained cells.

If a quota is short, top up generation first. If it remains short, report the shortage and reduce the dataset rather than duplicate pairs or silently relax correctness. Keep batch 16 and recompute the one-pass schedule.

### Necessary control

Train a **bida-style control**, two seeds, from the same clean source pool and numerical settings: shortest-right/longest-right, ratio 1.5, guard as above, per-type shorten:guard at most 1:1, no added R, no solve threshold or math-band exclusion.

This tests whether the new safeguards improve the actual trade-off rather than recreating the overly conservative Run A failure.

## 2.4 Explicit training settings

| Setting | J and bida-style control | I | R |
|---|---|---|---|
| Seeds | 0, 1 | 0, 1 | 0, 1 |
| Reference | Original model | Original model | Frozen selected J+I |
| Loss | Sequence-summed DPO | Same | Same |
| DPO beta | 0.1 | 0.1 | 0.1 |
| Chosen mean-token NLL coefficient | 0 | 0 | 0.1 |
| Base precision | FP32 | BF16 | FP32 |
| Forward precision | FP16 autocast, GradScaler | BF16 | FP32 initially |
| LoRA/optimizer parameters | FP32 | FP32 | FP32 |
| Peak LR | \(2\times10^{-5}\) | \(2\times10^{-5}\) | \(10^{-5}\) |
| Schedule | 16-update warmup, cosine to zero | 10-update warmup, cosine to \(2\times10^{-6}\) | 4-update warmup, cosine to zero |
| Effective batch | 16 pairs | 16 pairs | 16 pairs |
| Microbatch | One pair | One pair | One pair |
| Epochs / updates | One / 339 | One / 80 | One / 80 maximum |
| Training pairs | 5,424 target | Same 1,280 instruction pairs used by J | 1,280 |
| Saved checkpoints | 100, 200, 300, 339 | 40, 80 | 20, 40, 80 |
| Optimizer | AdamW, betas .9/.999, epsilon \(10^{-8}\), weight decay 0 | Same | Same |
| Gradient clipping | Global norm 1.0 | Same | Same |
| Maximum prompt+completion | 8,704 tokens | 8,704 | 8,704 |

Additional implementation requirements:

- Gradient checkpointing; training cache disabled.
- FP32 log-softmax/reductions; no length normalization of the DPO log-probability margin.
- Use `--mix all` and an explicitly stratified pair file. The trainer’s default three-kind mix would discard the new R pairs.
- Do not use `--target-steps` to silently change effective batch size.
- Separate diagnostic pairs from training. `--bida-exact` currently sets held-out size to zero and thereby disables the built-in tripwire; restore an external diagnostic set.
- Audit immediate ending after `</think>`, empty thinking, malformed answers, truncation, and call/no-call confusion on generated development answers.
- Cache reference scores against hashes of **both pair file and reference weights**.
- Record actual optimizer-update count, scaler skips, pair-token distributions, and CPU/CUDA RNG states.
- If FP32 repair does not fit 24 GB at long contexts, the predefined fallback is FP32 weights with FP16 autocast/scaling—not shortening the data cap. Record this deviation.

The modest repair NLL and replay mix are **new hypotheses**, not historically established improvements.

## 2.5 Merge selection

Average **weight deltas**, not LoRA factors:

\[
\bar\Delta_J=(\Delta_{J,0}+\Delta_{J,1})/2,\qquad
\bar\Delta_I=(\Delta_{I,0}+\Delta_{I,1})/2.
\]

Test:

\[
W_{\rm pre}=W_0+a\bar\Delta_J+b\bar\Delta_I
\]

with the predefined candidates:

- \((a,b)=(1,0)\): joint-only control;
- \((1,.25),(1,.5),(1.1,.25),(1.1,.5)\).

**Default hypothesis: \(a=1,b=.5\)**, but not a predetermined winner.

Why this combination? Multiplying `bidax5q-final` and half-strength lpt6-instruction token ratios gives a crude **56% screen-saving hypothesis**, leaving several saving points for repair. This is motivation to measure the merge, not a prediction of its accuracy.

Use ordinary task arithmetic in FP32. Do not introduce TIES/DARE, negative strengths, or layerwise searches without evidence. Save the final merged model in BF16 with the stock tokenizer.

## 2.6 One constrained repair round

Generate approximately **8,000 training questions ×4 answers from the frozen pre-repair model**, targeting observed training/development failure categories, never benchmark questions. Top up difficult cells where a correct winner exists.

Repair quotas:

| Type | Pairs |
|---|---:|
| Math | 512 |
| Code | 256 |
| Instructions | 192 |
| One function | 144 |
| Multiple functions | 80 |
| No-fit | 96 |
| **Total** | **1,280** |

Within each type:

- **50%** length-balanced R pairs, max/min length ≤1.25, half longer-correct winners;
- **25%** anti-overcut pairs: shortest suitable correct > too-short finished wrong;
- **25%** correct/correct shortening replay, preferably from the current model; original-model replay is allowed when necessary.

For anti-overcut pairs, require wrong thinking length below 0.75× correct thinking, but cap the correct/incorrect total-length ratio at 2. Avoid teaching “write much more” as the universal fix.

Train two seeds; select the same checkpoint for both and average their deltas. Test repair strength **\(c=0,.5,1\)**:

\[
W_{\rm final}=W_{\rm pre}+c\bar\Delta_R.
\]

Freeze \(a,b\) before generating repair data. Do not keep altering the underlying merge after repair.

## 2.7 Selection without another screen-fitting exercise

- Development data selects checkpoints and eliminates bad merges.
- Compare **recipes averaged across both seeds**, not the best seed.
- Prefer the modified J over the control only for a meaningful dev trade-off: approximately ≥0.5 accuracy point gained at ≤3 saving points spent, or ≥2 saving points gained at ≤0.3 accuracy point spent.
- Use at most **12 new official-screen evaluations** in total, including controls and confirmation.
- Screen only the best two development-qualified pre-repair candidates and best two final candidates—not every stored checkpoint.
- Preferred promotion region: **≥54% screen saving and ≥0 officially weighted accuracy change**. This provides some allowance for transfer and selection noise; it is not a confidence guarantee.
- Use question-cluster bootstrap intervals, maintaining benchmark weights. Do not treat repeated answers as independent questions.
- Evaluate the actual seed-averaged model; averaging weights is not an ensemble and is not guaranteed to help.
- Select one candidate before opening the locked non-benchmark confirmation set. If confirmation fails the predeclared gate, report failure rather than tuning on it.

---

# 3. Expected result and confidence

A successful engineering outcome would look approximately like this on the full test:

| Part | Target saving | Target accuracy change |
|---|---:|---:|
| Math | 59% | −1.5 |
| Code | 50% | 0 |
| Instructions | 50% | −0.5 |
| One function | 44% | −1.0 |
| Multiple functions | 47% | 0 |
| No-fit | 52% | +3.0 |

This corresponds to approximately **51.7–51.8% overall saving and −0.40 accuracy point**. Exact saving requires the true BFCL baseline total-output lengths, which are absent from the pack.

**That is a target, not a validated forecast.** My more cautious planning expectation is **about 50–52% saving and −0.8 point**, with substantial risk on either side.

Reasonable broad per-type planning ranges:

| Part | Saving range | Accuracy-change range |
|---|---:|---:|
| Math | 52–64% | −4 to +0.5 |
| Code | 42–57% | −2 to +2 |
| Instructions | 43–56% | −2.5 to +1.5 |
| One function | 37–49% | −3 to +1 |
| Multiple functions | 40–53% | −3 to +2 |
| No-fit | 44–59% | −1 to +6 |

Correlated unfavourable/favourable scenarios aggregate to approximately **44–58% saving and −2.7 to +1.7 points**. These are engineering uncertainty ranges, **not calibrated statistical confidence intervals**.

My subjective probability of meeting both strict thresholds within this budget is **roughly 25–40%**. Confidence is high in the failure modes identified above, moderate in the joint-adapter foundation, and low-to-moderate in the supplementary merge plus constrained repair delivering the remaining gain.

---

# 4. Four-day plan for both L4s

I would budget approximately **70–100 GPU-hours**, including regeneration, validation, and overhead, with the rest of the available 144–192 GPU-hours reserved for failures or a predefined switch. More checkpoint searching is not automatically a good use of spare compute.

| Day | GPU 0 | GPU 1 | Decision |
|---|---|---|---|
| **1: establish the foundation** | Generate half the original-model training pool; train control seed 0 and modified J seed 0 | Other half of pool; train control seed 1 and modified J seed 1 | Audit aggregation, contamination, grader validity, numerical stability and pair fill. Compare step 200/final on development. Stop broken recipes before merging. |
| **2: test complementarity** | Train I seed 0; evaluate joint/control averages and half the predefined merges | Train I seed 1; evaluate remaining merges and seed stability | Does I add ≥4 saving points at ≤~0.5 accuracy point cost on development, without a relevant-tool or math collapse? Screen the top two only. Freeze one pre-repair model; generate its repair rollouts. |
| **3: repair without undoing compression** | Repair seed 0; development evaluations at 20/40/80 | Repair seed 1; same evaluations | Select a common checkpoint by mean dev performance; evaluate averaged repair at strengths 0/.5/1. Screen at most two finalists. If repair spends >4 saving points for <1 accuracy point recovered, stop increasing it. |
| **4: freeze and check** | Load frozen BF16 stock model; run official full test, all 2,759 questions ×8, approximately 3–4 hours | Locked non-benchmark confirmation and packaging/stock-loading checks; independent screen resampling if useful, with no candidate changes | Publish the full-test result, uncertainty, token/cap statistics, and exact artifact hashes. No post-full-test strength tuning. |

Additional stop/switch rules:

- **Data:** if fewer than 80% of planned pairs can be filled after targeted top-ups, reduce the experiment count, not pair quality.
- **Numerics:** any persistent NaNs, reference mismatch, or large scaler-skip rate stops the run.
- **Behaviour:** reproducible math loss >4 points, code/instruction loss >3 points, or relevant-tool loss >3 points on development triggers rejection—not reliance on no-fit gains to conceal it.
- **Booster:** if all I strengths damage accuracy disproportionately, switch to Alternative 1 immediately.
- **Repair:** do not continue past 80 steps merely because accuracy is rising. Previous repairs show that accuracy can improve while the token objective disappears.
- **Final gate:** if no candidate is convincingly near the target, freeze the best valid trade-off and explicitly report that the target was not reached. Do not select a screen outlier to manufacture success.

The final full test must retain temperature 0.6, top-p .95, top-k 20, thinking on, and the official 8,192/4,096 token caps. Report the already-screened and remaining question strata separately as a diagnostic, without using either to change the frozen model.

---

# 5. Alternatives, ranked

## Alternative 1: joint adapter plus one short on-policy DPO round, no specialist booster

**Switch if:** instruction merging fails the cross-type cost test or behaves inconsistently across seeds.

Start from the two-seed-averaged joint winner. Generate its own answers and train a **100-step, 1,600-pair** continuation:

- 50% verified correct/correct shortening;
- 25% capped/incorrect guards;
- 25% length-balanced correctness;
- same type shares and solve thresholds as J;
- rank 16/alpha 32; beta .1; NLL .1;
- FP32 weights/FP16 autocast;
- LR \(10^{-5}\), five-step warmup, cosine to zero;
- batch 16, one pass, seeds 0/1;
- checkpoints 50/100; continuation strength .5 or 1.

This avoids cross-adapter interaction and attacks failures on the current policy. However, no scored second chain round establishes that iterative self-training is beneficial.

**Expected:** roughly **47–53% saving, −1.7 to +0.3 points**. It is simpler and safer operationally, but may remain below 50%.

## Alternative 2: reproduce the October four-adapter merge, then balanced repair

**Switch if:** the joint foundation repeatedly remains below roughly 45% saving, while inexpensive four-group reproductions show substantially better compression on development.

Reproduce the four groups with:

- original-model answers, ideally ten per selected question;
- long2short, minimum solve .5, no-fit .8;
- easy ratio 1.2, normal ratio 1.5, shorten cap 2;
- additional no-fit correctness pairs;
- rank 16/alpha 32;
- BF16, beta .1, NLL 1;
- constant LR \(2\times10^{-5}\), batch 16, one pass;
- two seeds per group.

Test only the predefined softer/soft/mid strengths, then apply the **new balanced repair**, not unrestricted historical repair.

This route has direct evidence of **55% screen saving**, but also direct evidence that ordinary repair reduces it to **43%**. Its likely outcome is around **45–53% saving with −2 to +0.5 points**, with lower confidence of satisfying both thresholds than the primary strategy.

I would not switch to all-six lpt6 merging, stronger no-fit adapters, wrong-over-wrong “keep trying,” or another broad GRPO sweep.

---

## Bottom line

The central lesson is not “specialize each type and add the adapters.” It is:

**Length control transfers broadly; correctness errors remain type-dependent; tool abstention and verbosity can dominate apparent improvements.**

The best use of these GPUs is to obtain a reproducible joint foundation, test **one** modest supplementary compression direction, and spend a tightly limited repair budget on verified current-policy mistakes—while enforcing the official aggregation and refusing to optimize another hundred noisy screen points.