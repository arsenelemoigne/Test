# Recommendation

**Use task-specific DPO adapters, merge them into one model, then permit one small, accuracy-gated on-policy round.** Allocate compression mainly to code and instruction following; protect arithmetic and tool selection; train irrelevance primarily for correctness rather than brevity.

I would **not** run another unrestricted shortest-right versus longest-answer experiment, add wrong-versus-wrong “hard” pairs, or automatically submit the final checkpoint.

This is a plausible route to the target, not an established solution. None of the disclosed, directly comparable runs achieves it. My planning estimate is **about 51–52% saved with 0.3–0.6 accuracy points lost**, but uncertainty is substantial.

---

## 1. What the evidence actually says

### First, use the correct objective

For each benchmark, let \(r_b\) be generated tokens divided by original-model tokens. The score is

\[
S=100\left[1-(r_{\rm math}r_{\rm code}r_{\rm IFEval}r_{\rm BFCL})^{1/4}\right].
\]

Accuracy changes are averaged across the four benchmarks. BFCL must first be pooled: accuracy by question counts, and tokens by baseline token expenditure.

Consequences:

- Code accounts for approximately **39.4% of raw test tokens**, but it does **not** receive 39.4% of the score’s weight.
- Arithmetic accuracy is expensive: losing four math points costs one overall point.
- Recovering ten irrelevance points gains about **0.71 overall points**, not 2.5.
- A model cannot compensate for weak compression everywhere else merely by eliminating code’s longest tails.

The historical trade-off curve is a diagnostic, not a selection objective. At 50% saving it predicts −2.89 accuracy points; meeting −1 requires approximately **+1.89 points above that curve**.

### Pair-kind and training-dose evidence

Numbers below are **saving % / accuracy-point change**.

| Experiment | Important observations | Interpretation |
|---|---|---|
| **D1: unfiltered shorten + wrong/wrong hard, no guards** | Step 100: **39.5/−2.6** overall. Math **57/−1.9**, but code **33/−3.5**, IFEval **22/−3.3**, simple tools **33/−5.4**. Final falls to **33.1/−2.3**. | Compression without correctness preferences is unsafe. Later training does not monotonically shorten. Irrelevance improves, but this does not justify the mixture elsewhere. |
| **D9: many shortening preferences** | Step 100: **68.3/−6.6**. Math **79/−11.7**, code **74/−7.9**, simple tools **52/−6.9**, multiple **56/−4.6**. IFEval is the exception: **58/−1.0**. | There is ample compression capacity, but aggressive global shortening destroys necessary computation. IFEval is the strongest candidate for an aggressive *isolated* compression dose. |
| **D9, later steps** | Step 200: **69.4/−7.6**; final **65.3/−6.8**. Irrelevance changes from **59/−5.6** to **64/−0.2**, while other parts worsen. | Different parts want different checkpoints. Irrelevance’s late recovery is real evidence against a single universal stopping time. |
| **D2: B + hard pairs on <50%-solved questions** | Only **9.4/ +0.1** at step 100 and **13.8/−0.1** final. Final code saving **4%**, IFEval **2%**. | Rewarding longer wrong answers largely neutralizes useful compression. “Longer wrong” is not evidence of better reasoning. |
| **D3: more hard + similar-length correctness** | **7.5/−1.0** at step 100; IFEval **−4/−2.5**. Uses 13,430 pairs. | No support for this routing recipe. Pair composition, pair exposure and optimization are confounded by the much larger dataset. |
| **Correctness-only stage 1** | **−4.6/+0.8** overall after 50 steps; math **7/+1.0**, IFEval **−7/+1.5**, irrelevance **0/+3.3**. | Similar-length right/wrong preferences can buy accuracy, usually at a token cost. Useful as protection, not as the whole solution. |
| **Two-stage correctness → B** | Step 100: reported **16.2/+1.7**. Final code **−19/+4.0**, math **20/+2.3**, irrelevance **42/+12.3**, but simple tools **14/−7.5**. | Strong evidence that accuracy repair is possible, particularly for irrelevance. Also strong evidence of cross-task interference and unacceptable verbosity if applied globally. |
| **D8: B + math/irrelevance correctness pairs** | Step 200 **43.3/−1.8**; final **41.7/−1.0**. Code improves from **37/−2.5** to **29/+1.3**; math remains **44/−3.3** final. | Protection is not necessarily localized to the intended task. Late correctness recovery can sacrifice substantial compression. |

The earlier fixed-share run is a useful safer baseline:

- **Run A, step 100:** **33.4/+0.4**, including code **32/+1.9**, IFEval **36/+0.4**, simple **31/+1.3**, multiple **34/+2.9**.
- By step 250: **32.3/−1.0**. More training bought essentially no aggregate saving.
- Adding gentle math shortening did not clearly improve the overall frontier.
- Constant learning rate also did not solve the plateau.

### The most useful evidence: independently trained adapters

| Merge | Overall | Math | Code | IFEval | Simple | Multiple | Irrelevance |
|---|---:|---:|---:|---:|---:|---:|---:|
| Softer | 39.2/−0.8 | 44/−2.1 | 44/−0.8 | 37/+0.4 | 28/+0.6 | 32/+1.3 | 30/−4.8 |
| Soft | 48.4/−1.9 | 53/−4.0 | **57/+0.4** | 46/−2.1 | 33/−1.0 | 37/+1.7 | 38/−6.9 |
| Mid | 55.0/−2.6 | 62/−5.6 | 65/−3.5 | **50/+0.6** | 37/−0.4 | **41/+1.7** | 42/−7.1 |

This suggests **different compression strengths, not simply a weaker global adapter**:

- Code’s soft-to-mid increment buys eight saving points but loses **3.9 accuracy points**.
- IFEval tolerates the stronger dose much better.
- Multiple-tool selection is already good around **37–41% saving**.
- Math and irrelevance are the major accuracy liabilities.

A useful budget calculation: holding the mid merge’s lengths fixed, repairing irrelevance from −7.1 to +3 and math from −5.6 to −2 would move overall accuracy from approximately **−2.58 to −0.96**. Repairing code to −1 would give approximately **−0.33**.

That is **not a prediction that these repairs preserve length**. It identifies the most valuable repairs.

### What the raw data adds

I found **17,870 questions and 71,480 answers**, exactly four answers per question—not merely the initial approximately 12,000-question pool.

#### Pair availability is very uneven

| Part | Questions with both right and wrong answers | Questions with similar-length right/wrong pair¹ | ≥3/4 right and shortest/longest right ratio ≥1.5 |
|---|---:|---:|---:|
| Math | 1,644 | 808 | 2,862 |
| Code | 886 | 563 | 1,050 |
| IFEval | 681 | 459 | 869 |
| Simple tools | 229 | 183 | 329 |
| Multiple tools | 174 | 144 | 187 |
| Irrelevance | 260 | 145 | 1,003 |

¹ Length ratio at most 1.25.

**Irrelevance has abundant easy shortening pairs and scarce correctness pairs.** A naïve pair miner therefore emphasizes exactly the wrong signal. Its training accuracy is 91.8%, versus 83.4% on the test.

Also, choosing the longest wrong rejection is not correctness-neutral. Median rejected/chosen length ratios for B-style guards are:

- Math **1.87**
- Code **1.50**
- IFEval **1.35**
- Simple **1.27**
- Multiple **1.27**
- Irrelevance **1.73**

Thus, “50% guards” can still mean nearly every update encourages shorter output. In contrast, approximately **45–53% of all similar-length right/wrong candidate pairs have the correct answer longer**. These are valuable anti-collapse examples.

#### Long failures matter—but do not establish a safe shortening policy

Share of training tokens spent on cut answers:

- Math **15.3%**
- Code **22.2%**
- IFEval **20.6%**

Questions solved **0/4** consume **50.6% of code tokens** and **38.6% of IFEval tokens**. They deserve additional sampling, not automatic “give up sooner” preferences. Four failures do not establish impossibility.

Likewise, selecting the shortest correct answer retrospectively saves only about **17–32%** on solvable questions, depending on part. Achieving >50% requires generalization beyond selecting among the original four outputs—one reason a carefully controlled on-policy round is attractive.

#### Text inspection changes the interpretation of “thinking too long”

- Code and math cuts are usually unclosed thinking, often with repeated reconsideration.
- **80% of sampled IFEval cuts already contain `</think>`**: many are final-answer runaway, not excessive thinking alone.
- Some sampled instruction combinations are apparently inconsistent and need a training-data audit.
- Correct irrelevance samples frequently contain explanatory prose rather than `[]`; training and evaluation grader semantics must agree.

The text sample is outcome-balanced, so these observations are diagnostic—not population-rate estimates.

Finally, the full-test thinking-off measurement supports selective protection:

- Math: **−68.1 points**
- Code: **−11.6**
- IFEval: **−1.3**
- Simple: **−10.7**
- Multiple: **−20.8**
- Irrelevance: **+6.9**

This is not permission to remove thinking. It shows where compression is most dangerous.

### Important uncertainty

These experiments are not clean marginal-effect estimates. B alone is absent; most D-runs have one seed; adapter strengths are unspecified; and global models create cross-task effects. Some rounded part numbers do not exactly reproduce reported aggregates. I would use original question-level screen results for selection, not reconstruct close decisions from the tables.

---

## 2. The recommended training recipe

### A. Prepare a 4,800-pair pool with explicit quotas

Train **six adapters separately**, all starting from the original model.

| Part | Pairs | Shorten | Guard | Similar-length right |
|---|---:|---:|---:|---:|
| Math | 720 | 72 — 10% | 288 — 40% | 360 — 50% |
| Code | 1,440 | 936 — 65% | 288 — 20% | 216 — 15% |
| IFEval | 1,440 | 936 — 65% | 288 — 20% | 216 — 15% |
| Simple tools | 480 | 216 — 45% | 120 — 25% | 144 — 30% |
| Multiple tools | 240 | 48 — 20% | 84 — 35% | 108 — 45% |
| Irrelevance | 480 | 48 — 10% | 240 — 50% | 192 — 40% |
| **Total** | **4,800** | **2,256** | **1,308** | **1,236** |

The **adapter separation is important**. A joint mixture resembling these shares is a control experiment, not my main bet: Run A already shows that reasonable shares alone can plateau.

### B. Exact pair construction

All chosen answers must be naturally generated, correct, finished, and contain a nonempty, closed thinking block. No cuts, rewritten conclusions, or teacher completions become chosen answers.

#### Shorten pairs

Use total generated length except for the additional IFEval thinking-specific test.

| Part | Eligibility and chosen answer | Rejected/chosen ratio | Chosen thinking-length selection window |
|---|---|---|---|
| Math | **4/4 correct**; choose second-shortest correct—the lower observed median | Total length **1.3–4×** | **128–1,024** |
| Code | **≥3/4 correct**; shortest correct | Total length **1.5–6×** | **128–2,048** |
| IFEval | **≥3/4 correct**; shortest correct | Thinking **1.5–6×**, total **≥1.15×**; ≥80% of saved tokens must come from thinking | **32–1,400** |
| Simple | **≥3/4 correct**; shortest correct | Total **1.5–6×** | **48–384** |
| Multiple | **4/4 correct**; lower observed median correct | Total **1.3–4×** | **64–384** |
| Irrelevance | **≥3/4 correct**; shortest correct | Total **1.5–6×** | **48–384** |

These are **training-selection bounds**, not inference budgets or forced stopping rules.

Additional rules:

- At most **two distinct shorten pairs per question**.
- Prefer one per question before adding second pairs.
- For non-IFEval shorten pairs, require at least half the length difference to come from thinking.
- Stratify IFEval by constraint family and required response length. Do not fill its quota almost entirely with short-final-answer questions.
- Do not reward cutting mandatory final-answer content.

The metadata supports the scale: with these approximate filters and a two-pair cap, I counted **1,490 code**, **417 simple-tool**, and **204 multiple-tool** shortening candidates. The refined IFEval rule supplies **1,293** candidates.

#### Guard pairs

Use only questions with both successes and failures:

- **Half:** shortest correct > shortest finished wrong.
- **Half:** shortest correct > longest wrong/cut.
- Prioritize cases where the first rule chooses a **longer correct** answer over a short failure.
- Cap cut rejections at **25% of guards**.
- Also cap extreme long-tail rejections—rejected more than four times chosen length—at **25% of guards**.
- Never apply the shortening-window caps to correctness pairs: occasionally necessary long correct reasoning must remain learnable.

This retains anti-loop pressure without making every “correctness” preference a disguised length preference.

#### Similar-length right pairs

- Correct > finished wrong from the same question.
- \(\max(L_c/L_r,L_r/L_c)\le1.25\).
- Select approximately **half with the correct answer longer**.
- Prefer 1/4- and 2/4-solved questions, but retain 3/4 disagreement cases.
- At most two such pairs per question; maximum three total pairs per question across all kinds.
- Deduplicate pairs across kinds.

**No wrong-versus-wrong hard pairs.**

If a quota is short, generate additional answers on relevant training questions. Do not silently replace scarce correctness pairs with easy shortening pairs.

### C. Match difficulty and audit labels

- Downweight raw math volume: its 6,999 questions must not dominate.
- Within math, use approximately **35% GSM8K-matched, 25% MetaMath-SV, 35% MetaMath-easy, 5% GSM8K-easy**, subject to provenance checks.
- For simple tools, avoid allowing the easier Glaive slice to dominate; start approximately balanced with ToolACE.
- For irrelevance, the 90% correctness-pair allocation naturally emphasizes mixed-outcome questions. Add near-fit/missing-parameter examples from permitted training sources.
- Audit suspicious constraints and tool-grading discrepancies. Do not discard a question merely because it was solved 0/4.
- Check provenance of the source named `gsm8k_easy_test`; a source label alone neither establishes contamination nor establishes safety.

**Practical prerequisite:** the supplied full-text sample is not sufficient to instantiate these pairs. It contains only 56 questions with two or more right full-text records and 32 with both right and wrong/cut records. Recover the team’s original prompt/completion store, or regenerate answers to permitted training prompts. Metadata candidate counts are not trainable completions.

### D. Training settings and checkpoint selection

Use the known setup rather than adding architectural novelty:

- Rank-32 LoRA, alpha 64, dropout 0.
- Decoder attention and MLP projections; freeze embeddings and tied output weights.
- DPO beta **0.1**, original model as reference.
- **No SFT/NLL term** in the first comparison.
- AdamW, peak LR **\(2\times10^{-5}\)**, 5% warm-up, cosine decay to **\(2\times10^{-6}\)**.
- BF16 computation with appropriate optimizer precision.
- Full completion log-probabilities, including thinking and final answer; mask prompts.
- Preserve complete sequences, including original cut rejections. Do not introduce training truncation that turns valid completions into malformed examples.
- Precompute reference log-probabilities.

For the separate adapters, use effective batch size **8**, one pass:

| Adapter | Steps | Checkpoints |
|---|---:|---|
| Math | 90 | 30 / 60 / 90 |
| Code | 180 | 60 / 120 / 180 |
| IFEval | 180 | 60 / 120 / 180 |
| Simple | 60 | 20 / 40 / 60 |
| Multiple | 30 | 10 / 20 / 30 |
| Irrelevance | 60 | 20 / 40 / 60 |

These step counts are **not directly comparable** with the common 339-step runs because effective batch sizes differ.

Initial merge candidate:

- Checkpoints: **60 / 120 / 180 / 40 / 20 / 60**
- Strengths: **0.35 / 0.80 / 0.90 / 0.65 / 0.50 / 0.85**

Those strengths are an initialization for measurement, not an inferred optimum. First screen global multipliers **0.6, 0.8, 1.0**, then make a small coordinate search over math, code, IFEval and irrelevance strengths.

Merge actual weight deltas:

\[
W=W_0+\sum_t\lambda_t\Delta W_t.
\]

The submitted model is one ordinary approximately 0.6B model—**no task router, no benchmark identifier, no ensemble at inference**. Every merged candidate must be evaluated on every part. One cannot simply combine the best per-part table entries.

### E. One conditional on-policy round

Run this only if the merged first round is near the useful frontier: approximately **46–50% saving with accuracy better than −0.7 points** on the screen.

Generate eight answers each on about **3,000 training questions**, prioritizing:

1. Previous mixed-outcome questions.
2. Code/IFEval questions with long failures.
3. Near-fit irrelevant tools and confused multiple-tool choices.
4. A substantial fresh, source-matched slice.

Use the **actual merged candidate** to generate these answers. Then train one small joint adapter, with that candidate as reference.

Proposed second-round pool:

| Part | Total | Shorten | Guard | Right |
|---|---:|---:|---:|---:|
| Math | 400 | 0 | 200 | 200 |
| Code | 640 | 352 | 160 | 128 |
| IFEval | 640 | 384 | 160 | 96 |
| Simple | 240 | 72 | 84 | 84 |
| Multiple | 160 | 16 | 64 | 80 |
| Irrelevance | 320 | 0 | 192 | 128 |
| **Total** | **2,400** | **824** | **860** | **716** |

- **75% fresh on-policy pairs, 25% original-policy replay**, within the type/kind quotas where available.
- Fresh shortening requires **≥6/8 correct**, and **≥7/8 for multiple tools**.
- Keep the same ratio and completion-integrity rules.
- No new math or irrelevance shortening pressure.
- LR **\(10^{-5}\)**, beta 0.1, effective batch 64, approximately **38 steps**.
- Checkpoints **10 / 20 / 30 / 38**.

The goal is only another **roughly 7–10% reduction in remaining tokens**, not another aggressive absolute 10–15-point saving jump.

Why stop at one round? The older iterative line moved **40.4/−0.9 → 51.4/−1.8 → 56.0/−3.6**. Iteration clearly finds shorter modes, but its accuracy debt compounds. The historical “balanced” repeat also failed the accuracy target.

---

## 3. Expected results and confidence

Here is my **full-test planning forecast**, conditional on successful data recovery, label auditing and checkpoint selection. These are research estimates, not statistical confidence intervals.

| Part | Central saving | Central accuracy change | Plausible saving range | Plausible accuracy range |
|---|---:|---:|---:|---:|
| Math | 43% | −1.3 | 35–48% | −2.0 to 0 |
| Code | 62% | −0.7 | 55–66% | −2.5 to +0.5 |
| IFEval | 56% | −0.7 | 50–59% | −1.7 to +0.5 |
| Simple tools | 37% | −0.5 | 32–42% | −1.5 to +0.8 |
| Multiple tools | 37% | +0.3 | 30–41% | −1.5 to +1.8 |
| Irrelevance | 48% | +3.5 | 40–57% | 0 to +6 |

The central allocation gives:

- **BFCL pooled:** 40.3% saved, +0.83 accuracy points.
- **Overall:** **51.15% saved, −0.47 accuracy points**.

A somewhat better code/IFEval outcome moves saving toward 52–53%. A disappointing outcome is approximately **44–47% saved**, or an accuracy loss exceeding one point.

Two especially uncertain improvements are required:

- Code must move beyond the demonstrated **57/+0.4** operating point without reproducing the **65/−3.5** failure.
- IFEval needs roughly **56% total saving**. If its final-answer length stays near the original mean, that leaves only about 43 thinking tokens on average—an approximately 90% thinking reduction. This is aggressive despite its small thinking-off accuracy gap.

I would call the probability of meeting **both** strict thresholds within this budget **roughly one-third to one-half**, as a subjective research judgment. It is not defensible to promise success from these observations.

---

## 4. Ranked alternatives and the three-day execution plan

### Alternative 1: Repair the existing mid-strength merge

**Best fast alternative if its weights remain available.**

Start from **55.0/−2.6**. Generate on-policy answers and apply a 20–40-step correctness-heavy round focused on math, code and irrelevance, with replay protecting IFEval and valid tool calls.

Why promising: the accuracy-budget calculation shows that targeted repairs could suffice.

Why second choice: repair can lengthen outputs or undo the compression. The two-stage experiment demonstrates both effects. Do not assume 55% saving survives.

### Alternative 2: Joint fixed-share DPO, then the same gated on-policy round

Use the proposed 4,800-pair table in one common-setup run, screening at **100 and 200**, not merely final.

This is simpler and a useful control. However, Run A and D8 suggest a likely off-policy plateau around **33–43% saving**. I rank it below adapter separation because cross-task interference is visible throughout the results.

### Alternative 3: Softer merged model, no second round

Choose the best conservative merge with approximately **40–48% saving** and minimal accuracy change.

This is the fallback submission if the target cannot be reached honestly. It is preferable to claiming that a nominal 51%/−0.9 screen result is securely within bounds.

### GPU plan

Budget approximately **120 GPU-hours**, leaving about **24 GPU-hours** for failed jobs, regeneration or confirmation.

| Work | GPU-hours budget | Decision |
|---|---:|---|
| Recover/regenerate full rollouts; targeted top-ups | 10 | Can the intended quotas and graders be trusted? |
| Six first-round adapters | 14 | Which tasks tolerate isolated compression? |
| Joint control, including two screens | 6 | Does separation actually help? |
| Small merge search and full 720-question screens | 18 | Find a feasible first-round operating point |
| On-policy generation, two small candidate updates and screens | 13 | Modest shortening round versus accuracy-repair round |
| Independent larger validation, including baseline | 23 | Does the result survive checkpoint selection? |
| Second-seed confirmation | 8 | Is the apparent gain seed-specific? |
| Final authorized full evaluation | 8 | Final measurement, not training feedback |
| Operational reserve | 20 | Failures and additional checks |
| **Total** | **120** | |

**Day 1:** recover data, audit scorers, launch isolated adapters on GPU 1; run the joint control and, if available, the existing-mid repair probe on GPU 2.

**Day 2:** select merges, generate on-policy answers, run the two short second-round branches. Keep the search deliberately small.

**Day 3:** evaluate at most two finalists on an independent **approximately 2,000-question ×8** validation set, confirm the leading recipe’s seed sensitivity, freeze the model, then perform the final authorized evaluation.

### Stopping and acceptance rules

1. **Do not advance an accuracy-broken compression model.** If a round loses more than one additional overall point, or shows a clear multi-point math/code regression, return to the preceding checkpoint.
2. **Stop on-policy training once accuracy worsens without a useful saving increment.** Do not finish an epoch for administrative convenience.
3. **Aim for margin:** approximately **≥52% screen saving and ≥−0.5 accuracy points**, not 50.1/−0.99. The observed screen/full offsets—+0.6 saving and −0.3 accuracy—are based on only two models and are not guarantees.
4. Bootstrap **paired questions**, retaining all repeated answers within each question. Do not treat the eight samples as eight independent questions.
5. Freeze selection before the full test. Never use test failures to construct training pairs.
6. Audit that every candidate still naturally thinks, closes `</think>`, and answers under the unchanged generation protocol.

**Bottom line:** the best opportunity is not “more shortening pairs.” It is **isolating the tasks that tolerate compression, supplying correctness preferences that genuinely oppose premature stopping, repairing irrelevance, and using one modest on-policy round to cross the remaining gap**.