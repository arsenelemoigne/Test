# Jasmin data review

## Bottom line

**Build jpa first as a frozen replication baseline—but do not treat “October’s rule, unchanged” as meaning unchanged training behavior.** The code contains several mechanisms through which the new pool can change the selected data substantially:

1. New source names may be unmapped or excluded.
2. Adding examples to a previously empty matching cell can **reduce**, rather than increase, the final dataset.
3. The weakest-filled part controls the size of **every** part.
4. The raw shorten cap is tied to **guard pairs only**; matching can subsequently violate that ratio.
5. No-fit protection operates differently below and above **8/10 correct**, while the matcher puts 5/10 through 9/10 in one cell.
6. Random pair sampling and question-round-robin sampling are not equivalent controls.

I would build three principal sets:

- **jpa-replica:** exact historical selection behavior, with validated inputs and better reporting.
- **jpb-RR:** unmatched by solve/length, but retaining question-level coverage controls and jpa’s final part×kind counts.
- **jpc-trainref:** matched against a disjoint training-reference set, with an accompanying unmatched control at its counts.

For deployment, I prefer the training-reference route. However, **changing only the pair-matching references does not remove evaluation-statistics dependence from an already evaluation-matched question pool**.

A correction to my previous report: October’s exact command used **10 answers per question**, not four. Thus jpa preserves October’s answer count. Four answers belong to the relevant bida/D10 comparisons.

---

# 1. Is jpa the right first build?

**Yes, as the historical-rule baseline. No, as an automatically approved final training set.**

Keep the first build free of the proposed extra math correctness pairs. Otherwise there is no clean “new pool, old rule” baseline. Add those as a separately named variant.

The pack contains source code and summary statistics, but not the underlying pool/rollout manifests. Consequently, the issues below are either **verified code behavior** or **input checks that must be performed**—not claims that the actual new pool has already failed them.

## 1.1 Validate source routing before doing anything else

`lupin_pairs.py` calls `build_pool.part_of(row)`; it does **not** use an existing `row["part"]`.

The recognized math sources are:

- `gsm8k_matched`
- `gsm8k_easy`
- `gsm8k_easy_test`
- `metamath_easy`
- `metamath_sv`

Tools must have source `toolace`, `hermes_tools`, or `glaive_tools`, plus an appropriate category.

**Literal source values `math1`, `math2`, and `nofit4` are not recognized.** They may only be pool/file names, in which case this is fine. But verify the actual rollout rows. In an in-memory test of the supplied code, mixing a literal `source="math1"` with recognized sources caused a sorting `TypeError` after pairs acquired `part=None`.

There is a second, separate restriction in `build_pool.py`:

- Default math mix allows only `gsm8k_matched` and `metamath_sv`.
- Default tool source allows only `toolace`.

A source can therefore be recognized by `part_of` yet excluded by the pool-builder defaults.

**Required assertions:**

- Every input row maps to exactly one intended part.
- Mapped part agrees with the manifest’s declared part.
- Every eligible source has an explicit allowlist/mix entry.
- Question IDs are globally unique across top-ups.
- Each question has exactly ten distinct original-model rollouts, not duplicated files or accidental cross-source ID collisions.
- Record source provenance separately from the part-routing label; do not relabel arbitrary data merely to make it pass.

## 1.2 The exact October rule is more specific than its shorthand

With ten answers, the default behavior is:

| Observed correct answers | Math/code/instructions | No-fit |
|---|---|---|
| 0/10 | No pair | No pair |
| 1–4/10 | One eligible similar-length `right` pair | One eligible similar-length `right` pair |
| 5–7/10 | `guard` and eligible `shorten` | One eligible similar-length `right` pair |
| 8–9/10 | `guard` and eligible `shorten` | `guard`, eligible `shorten`, and eligible `right` |
| 10/10 | Eligible `shorten`, using easy ratio | Same |

Important details:

- **`guard` means shortest correct > longest wrong**, including truncated or unclosed wrong answers. It does not necessarily mean shortest finished wrong.
- Below the solve floor, the code replaces that signal with a similar-length correct-versus-**finished**-wrong pair, then exits the question’s processing.
- The default `right-band` is **0.25**, not 0.15:
  \[
  |L_r-L_w|\le .25\max(L_r,L_w).
  \]
  This allows a longer/shorter ratio up to **1.333**.
- The **1.2 easy ratio applies only to 10/10-correct questions**. A 9/10 question still uses 1.5.
- Pair length comparisons use **total output tokens**, not thinking tokens. Matching uses median thinking length, with an output-length fallback.

Thus changing the “50% solve floor” changes both **whether shortening is allowed** and **what correctness signal is generated**. It is not a pure shortening filter.

## 1.3 Raw caps can change sharply with the new pool

In `lupin_pairs.py:314–323`, the cap is:

\[
N_{\text{shorten},p}\le 2N_{\text{guard},p}.
\]

It is:

- Per part;
- Applied before matching;
- Based on **guards only**, not `guard + right`;
- Implemented by random subsampling of shorten pairs.

Consequences:

- A set of always-correct questions can generate many shortening candidates but **zero retained pairs if that part has no guards**. I verified this with the actual rule body: twenty always-correct synthetic questions produced twenty shortening candidates and zero final pairs.
- Adding hard no-fit questions with solve rate below .8 can add `right` pairs without increasing the shorten allowance.
- Increasing a solve floor reduces the guard count and can suppress shortening even on untouched 10/10 questions.

For example, in a synthetic pool with thirty always-correct questions and one question at each of 5/10–9/10, raising the floor:

| Floor | Guards | Retained shortening pairs |
|---|---:|---:|
| .5 | 5 | 10 |
| .6 | 4 | 8 |
| .8 | 2 | 4 |

That is why floor changes require reporting **pre-cap and post-cap counts**.

**Do not change this behavior silently in jpa.** But monitor it before concluding that a ratio or source variant “did nothing.”

## 1.4 Matching has an empty-cell discontinuity

The prose says short cells are not filled from other cells. The implementation makes an important distinction:

- **Completely empty cells:** their reference share is redistributed over live cells.
- **Nonempty but underfilled cells:** their shortfall remains unfilled.

This produces a discontinuity. With two cells having target shares 40%/60% and a 100-pair budget:

| Availability | Allocated quotas | Retained |
|---|---|---:|
| 0 / 1,000 | 0 / 100 | **100** |
| 1 / 1,000 | 40 / 60 | **61** |
| 100 / 1,000 | 40 / 60 | **100** |

**Adding one example to an empty cell can reduce the selected total by 39%.** With the subsequent global fill rule, it can shrink other parts too.

The new math/no-fit sources make this particularly relevant.

For jpa-replica, preserve this behavior and report it. For a revised selector, use an explicit capacity-aware allocation policy: distinguish structurally unsupported cells from scarce cells, record unmet target mass, and avoid an abrupt zero-to-one transition. That would be a **new selection variant**, not an invisible bug fix.

## 1.5 One underfilled part shrinks everything

At `lupin_select.py:173–184`, every part is trimmed to the smallest fill fraction.

The 3,200 requested pairs imply:

| Part | Requested |
|---|---:|
| Math | 800 |
| Code | 800 |
| Instructions | 800 |
| One function | 381 |
| Several functions | 190 |
| No-fit | 229 |

If no-fit fills only 100/229 while every other part fills completely, the final dataset becomes approximately:

**349 / 349 / 349 / 166 / 83 / 100 = 1,396 pairs.**

That illustrates how October could request 3,200 yet deliver roughly 1,400. It is not evidence that no-fit was necessarily October’s actual bottleneck.

Adding nofit4 may release this bottleneck and increase **all adapters’ training counts**, including code and instructions. Conversely, activating scarce cells could tighten it.

Therefore “only the data changes” may also mean **more optimizer steps at one epoch**. Record and control that distinction.

For later per-type production training, I would favor **independent part budgets** over discarding useful math/code/instruction data because a tool slice is short. Preserve the intended callable/no-fit proportions inside pooled tools. Keep global trimming only in the historical baseline and matched controls.

Also: `--loose-parts` is not a reliable escape hatch. It excludes a part from determining the fill ratio, but the final loop still trims that part to the common `keep` count, contrary to its help text.

## 1.6 The cap and round-robin do not guarantee the final intended mix

Two more implementation details matter:

- **Matching can violate the raw 2:1 shorten/guard cap.** For example, selecting 80 shortening pairs and ten guards from a raw 100/100 pool produces 8:1.
- The optional post-matching cap is **off by default** in October’s command. When enabled, it counts all non-shortening kinds, including `hard`, as the denominator.

Do not enable that cap without naming it as a variant.

Round-robin is:

- Within each **cell**, not across the whole part;
- Over questions with pairs;
- Random within a question’s pair list.

It does not prioritize correctness pairs over shortening pairs. With `kind-share-part`, each kind gets a separate selection pass, so question coverage is not coordinated across kinds.

Finally, the report’s `kinds` and `kept_cells` are collected **before final trimming**, and some before later balancing. Recompute statistics from the actual final pair file.

## 1.7 Reproducibility and reference hygiene

Before building, fix or explicitly audit:

- **Shared RNG coupling:** adding no-fit data can change which math pairs survive, even with identical math inputs. I verified this using the actual pair-building body. Prefer independent stable RNG streams per part/stage for new controls; retain the historical stream for jpa-replica.
- **Reference overlap:** reference questions are keyed by `(reference-file index, id)`. The same question in two reference sets is counted twice rather than consolidated.
- **Answer-count mismatch:** equal-looking solve rates from four, eight, and ten answers do not have equal binning behavior. At true success probability .8, “always correct” occurs with probability **41.0% at four samples versus 10.7% at ten**.
- **Correctness semantics:** pair construction requires correct, closed, nontruncated outputs. `select_pool.py` uses raw `correct`; reference loading also trusts it. Confirm that upstream grading already applies the completion rule.
- **Zero thinking lengths:** `or output_tokens` treats zero as missing. Report zero/missing thinking explicitly rather than silently conflating them.

---

# 2. Is jpb the right matching control?

## Keep it—but change the claim, or strengthen the control

As proposed, jpb answers:

> Does October’s entire pair-selection procedure outperform random pair sampling at the same final part×kind counts?

That is useful. It does **not** isolate solve×length matching because random pair sampling also changes:

- Unique-question coverage;
- Number of pairs contributed per question;
- Source composition;
- Round-robin behavior.

A question with three eligible pairs is three times as likely to contribute to an initial random pair draw as a question with one. Question-first selection does not have that bias.

## Recommended controls

### A. jpa-replica

Exact October selection on the new raw pairs, with the audits above.

### B. jpb-RR: preferred primary unmatched control

Use the **same capped raw pair universe**, then:

- Remove solve×thinking cell quotas;
- Match jpa’s **final part×kind counts**;
- Use question-first selection without replacement;
- Coordinate selection across kinds so a question’s second pair is not preferred over an eligible new question merely because kinds were processed separately;
- Match or closely track unique-question counts and the pairs-per-question histogram.

If exact constraints are infeasible, report the discrepancy rather than hiding it. Keep the original random-pair jpb as a cheap pairs-only diagnostic.

For “the whole matching policy,” let source composition vary naturally. To isolate matching beyond source reweighting, add a secondary control also matched on **part×kind×source**. It need not be trained unless source shifts are substantial.

### C. jpc-trainref: the cleaner matching reference

Reserve roughly **160–320 questions per part** from allowed training sources, disjoint from:

- Pair-producing questions;
- Other evaluation/development questions used to choose checkpoints.

Generate or reuse ten original-model answers per reference question and construct the same behavioral cells.

Use a corresponding unmatched control at jpc’s counts; otherwise jpc versus jpa may confound reference choice with dataset size and kind mix.

### The rules issue is not fully removed by changing pair references

The current pool was itself selected using evaluation-derived:

1. Prompt-length ranges/bins;
2. Solve-rate×thinking distributions.

Changing only `lupin_select.py --reference` leaves both dependencies in place.

For a genuinely evaluation-statistics-free path, construct the candidate pool using training references—or a predeclared source/sample policy—at **both pool-selection stages and pair selection**. Using evaluation text solely for exclusion/decontamination is a different purpose and should be documented separately.

The benchmark’s silence is not affirmative authorization. Disclose the historical evaluation-statistics use and seek clarification if submitting it; prefer the clean route rather than claiming it is unquestionably allowed.

---

# 3. One-rule-at-a-time variants

## First: establish the actual selection effects without training

For every variant, retain the same question rollouts, reference set, random-priority keys and all other settings.

Report, **before caps, after caps, after matching, and after final trimming**:

- Pairs and unique questions by part×kind×source;
- Solve counts **k=0,…,10**, not just four bins;
- Thinking and total-output lengths: p10/p50/p90/p99;
- Chosen/rejected length ratio and signed log-length difference;
- Fraction where the correct answer is longer;
- Finished-wrong versus truncated/unclosed rejected answers;
- Exact duplicate pairs, including duplicates tagged as different kinds;
- Cell availability, quota, exclusions, shortfalls and final fill;
- Question coverage and pair overlap/Jaccard with baseline;
- Expected steps at batch 16.

A variant that produces essentially the same final pairs should not get a GPU run merely because its flag changed.

Build several selection-seed realizations cheaply. Train one initially; use a second seed for promising comparisons.

## Recommended order and values

| Priority | Variable | Values, changing one setting at a time | First adapter |
|---|---|---|---|
| **1** | Matching/reference policy | jpa, jpb-RR, clean training-reference matching | Math and pooled tools |
| **2** | Longer-right threshold | **1.5 baseline; 2.0 conservative; 1.25 permissive** | Math first for 2.0; instructions for 1.25 |
| **3** | Solve floor | **.5 baseline; .6 first; .4 permissive**. Use .8 only as a stress test if supported | Math for .6; instructions/code for .4 |
| **4** | Always-correct easy ratio | **1.2 baseline; 1.5 conservative; 1.1 permissive** | Math and instructions |
| **5** | Small math correctness supplement | **0%, 5%, 10%**, as specified below | Math only |
| **6** | Targeted tool/data corrections | No-fit floor **.8 baseline versus .9**; retain pooled-tools calibration | Pooled tools, then no-fit challenger |

If pre-build statistics show that the raw cap is binding heavily, a **cap=1 versus 2 versus 3** diagnostic deserves priority over fine-grained ratio searches. Otherwise several ratio variants will merely reshuffle which shortening pairs fit under an unchanged limit.

### What each variable actually tests

**Long ratio 1.5 → 2.0 or 1.25**

This changes eligibility, not the chosen/rejected endpoints of a surviving pair. Under the default easy override, it primarily affects partially solved questions:

- 5–9/10 for math/code/instructions;
- 8–9/10 for no-fit.

It does not change 10/10 pairs while `easy-ratio=1.2` remains active.

**Solve floor .5 → .6**

This reroutes exactly the 5/10 questions from guard/shorten to eligible similar-length `right`. It also reduces the guard-linked shortening budget.

The matcher still puts 5/10 through 9/10 together, so its coarse cells cannot expose this change. Report exact k/10 distributions.

**Easy ratio 1.2 → 1.5 or 1.1**

This isolates always-correct questions. Set it explicitly to 1.5 for the conservative test. Setting it to zero disables the override and ties it to whatever the long ratio currently is—a less clean experimental definition.

## What I expect to matter most

1. **Pool/source coverage, matching support and effective training exposure.**
2. **Solve-floor/cap interactions and the correctness-versus-shortening mix.**
3. **The easy ratio**, if the new math pool contributes many 10/10 questions.
4. **The ordinary long ratio**, conditional on how much of its change survives capping and matching.

This is a conditional ranking, not a claim that matching has already been proven beneficial.

For math1, the supplied counts are:

- **48.2% always solved**
- **46.5% sometimes solved**
- **5.3% never solved**

If its reported 80% mean solve rate uses the same answers, the sometimes-solved group averages approximately **68.4% correctness**. That puts substantial mass exactly where floor and guard rules matter. Math2’s distribution is not supplied; do not assume it matches math1.

## Cheap training and checkpoint protocol

For each selected comparison:

- Train **only the affected adapter**, not all seven.
- Hold float32 policy, NLL=1, β=.1, LR, batch, LoRA configuration and training seed fixed.
- Start with constant LR **2e−5**, batch **16**, one pass.
- Save **step 0, every 10 steps, and final**. If fewer than twenty steps, also save step 5.
- Initially score two checkpoints: approximately **half-pass and final**. If final is worse, score the saved early checkpoint nearest step 10.

With fully filled jpa quotas, this means approximately:

- Math/code/instructions: **50 steps each**;
- Pooled tools: **50 steps**;
- Simple/multiple/no-fit challengers: **24/12/15 steps**.

Actual counts may be much smaller.

For variants with different sizes, show two comparisons:

1. **Operational one-pass outcome**;
2. **Matched-exposure outcome**, using a common feasible count or common step budget.

Do not confuse a pair-rule benefit with training longer.

Score solos with **own type ×4, others ×2**. For pooled tools, all three BFCL parts are “own.” Compute benchmark scores using the official weights, not the number of generated samples.

Advance a variant only if it improves the own-type accuracy/length frontier without unacceptable spill-over. Practical first-pass signals are:

- Roughly **≥5 percentage points more own-type saving** without a clear accuracy loss; or
- Approximately **≥2 points better own-type accuracy at matched saving**, followed by confirmation because per-part noise is larger than that.

Compare at matched compression using checkpoint choice or one additional strength, not only at strength 1.0. Then substitute the winning adapter into a **fixed merge context**.

### Updated interpretation of the two half-strength results

- **Math@.5:** −12.2% saved / −0.5 overall; math **26% longer / −2.9**. It remains unsuitable as a protective merge component.
- **Code@.5:** −0.8% saved / +0.3 overall; code **2% longer / +2.3**. This is not evidence of useful compression, but its code-accuracy improvement means it should not be described as having no benefit whatsoever.

The pending pt-minus-code screen remains the right deployment test.

---

# 4. Exact proposal for the math adapter’s “few be-right pairs”

## First supplement: 10% additional correctness bucket, not a wholesale recipe change

My preferred first variant is:

- **90%** sampled from the unchanged jpa math pairs;
- **10%** supplemental pairs of kind **`right`**;
- **No additional `hard` pairs**;
- **No additional repair-style guards in this first variant**.

The final pair count stays equal to jpa math’s count.

### Supplemental eligibility

Use original-model answers to training questions with:

\[
0.5\le\hat p<0.8,
\]

i.e. **5/10, 6/10 or 7/10 correct**.

For each supplemental pair:

- Correct answer must be finished and have closed thinking.
- Wrong answer must also be finished and have closed thinking.
- Require
  \[
  |L_r-L_w|\le .15\max(L_r,L_w),
  \]
  allowing a maximum length ratio of approximately **1.176**.
- Choose the closest-length eligible pair within the requested orientation.
- Balance supplemental pairs approximately **50% correct-longer / 50% correct-shorter**.
- At most one supplemental pair per question.
- Exclude exact pair duplicates already present in the retained 90%.
- Preserve source and 5/6/7-correct coverage as far as support permits.

Why target 5–7/10? October already creates `right` pairs below .5. This adds correctness supervision exactly where the baseline switches to guard/shorten. It avoids spending the small budget duplicating an existing low-solve signal.

At an 800-pair math budget, this is **80 supplemental pairs: about forty of each length orientation**.

If there are too few eligible, balanced pairs, start at **5%**. Do not duplicate scarce examples or silently widen the length band.

### Be precise about the percentage

This is a **10% supplemental bucket**, not necessarily 10% total `right` pairs. If the baseline right fraction is \(r\), the resulting total right fraction is approximately:

\[
0.9r+0.1.
\]

For example, a baseline with 20% right pairs becomes approximately **28% right**, not 10%.

### Existing flags are only a candidate generator

`--add-right-below svamp=0.8` generates the relevant additional candidates at 5–7/10; below .5 the early branch already emits the right pair.

But that flag alone does **not** enforce:

- A 10% quota;
- A 15% length band;
- Balanced length direction;
- Post-selection preservation of the intended share.

Similarly, `--right-parts svamp` adds right candidates at 5–9/10, not just the intended middle range.

Construct and audit the supplemental bucket explicitly.

## Why not copy repair pairs immediately?

The repair results are encouraging but not a clean endorsement of more correctness pairs everywhere:

- pt-mid math: **−5.6 accuracy**, 62% saved;
- rpB-20 math: **−1.9**, 53% saved;
- rpC-80 math: **−1.7**, 44% saved.

Some protection came with substantially restored length. Moreover, repair used the compressed model’s own failures; Jasmin’s initial pairs come from the original model.

The joint `math-final` variant reached **41.7 / −1.0**, but math accuracy was **−3.3**, versus **−1.9** for bidax5q-final. It does not prove that broad extra right-pair mining protects math.

If the 10% right supplement helps, the next single-change challenger can replace half of that bucket:

- **5% balanced similar-length `right`;**
- **5% repair-style `guard`: shortest correct > shortest finished wrong**, on the same 5–7/10 questions.

Do not simultaneously add a medium-length exclusion band or DAST wrong-over-wrong pairs.

## How to tell whether it helped

Require:

1. Better math accuracy at **matched math token saving**, not merely longer reasoning;
2. No new length inflation resembling lpt6-math;
3. No substantial negative spill-over;
4. Improvement when substituted into the fixed merge, not just solo;
5. Confirmation on independent questions or a second training seed.

A result that restores two math accuracy points by giving back twenty saving points is protection, but not necessarily progress toward Jasmin’s target.

---

# 5. Changes to data selection I would make

## 5.1 Keep the useful new sources—but control where they enter

The new pool has a credible positive reference: bidax5q-final reaches **46.7 / −0.5**. That supports testing its math sources, not copying every associated recipe.

At the same time:

- D10 on this pool, unmatched and with different training settings, reaches **53.5 / −2.7**, math **−5.4**.
- The modified math/protection recipe behind bidax5s and lpt6 can produce severe length inflation.
- Adding further instruction/code sources in bidax7 did not establish improvement: final **43.5 / −2.5**, despite better intermediate checkpoints.

Thus **more data is not itself the intervention to optimize**. Retained source mix, eligibility and checkpoint exposure are.

After jpa/jpb, the most useful source ablations are:

- Math adapter: old sources versus old+math1/math2, at matched pair count and kind mix;
- Pooled tools: with versus without nofit4, maintaining callable/no-fit part quotas.

Do not infer a specific optimal GSM8K/MetaMath ratio from the available summaries.

## 5.2 Target coverage deficits, not just raw pair volume

Add questions where the retained set has inadequate support:

- Math: reliable 5–7/10 correctness contrasts and safe easy-question shortening;
- Tools: genuine ambiguity and callable counterexamples, not just more “no call” labels;
- Instructions: constraint combinations the current source mix lacks;
- Code: executable, correctly graded tasks with representative output lengths.

Never-solved examples produce no October preference pair. Adding more of them will not repair a pair-count shortage unless the supervision source or objective changes.

## 5.3 Keep pooled tools as the calibration anchor

The no-fit specialist’s full-strength **+11.7** on no-fit and **−20.6/−16.6** on callable tools show why nofit4 must not dominate merely because it supplies many pairs.

Preserve pooled-tools subpart proportions close to **400:200:240** for the initial controlled run. Inspect the **final kind mix within each subpart**, since correct proportions alone do not guarantee balanced gradients.

For split challengers, retain the pooled adapter as a genuine independent training control. If no-fit needs callable calibration examples, introduce them as a named later variant.

## 5.4 Relax brittle pool matching only in a separately named clean variant

`build_pool.py` excludes prompts outside the choosing set’s observed minimum/maximum length before binning. With a small reference set, that is a brittle hard cutoff.

For the training-reference pipeline, I would replace this with:

- Training-derived quantile bins;
- Explicit lower/upper overflow bins;
- A documented tail budget rather than automatic rejection.

Do not change jpa-replica’s upstream behavior retrospectively.

Also, `select_pool.py` greedily borrows from neighboring cells. Borrowing can consume examples another cell needs later. Report actual retained cells and borrowing, not merely “matched pool.”

## 5.5 Strengthen decontamination and provenance checks

The current eight-word check:

- Ignores n-grams appearing in at least three evaluation questions as presumed templates;
- Deduplicates exact serialized prompts;
- Does not establish semantic independence or source-split validity.

Retain it, but add provenance/split checks, normalized exact duplicates, and targeted near-duplicate review across top-ups. Ensure function schemas are covered when stored outside prompt text.

The recognized name `gsm8k_easy_test` deserves a provenance audit; its name alone does not establish contamination with the four target benchmarks, but it should not pass without explanation.

## 5.6 Validate selection on independent generations

Selection and pair mining use the same ten rollouts, so apparent solve-rate/length matching can be optimistic. On a modest training-only audit sample, generate fresh answers—or use existing independent answers if available—to check regression after selection.

Keep this separate from benchmark evaluation and from the pair-training set.

---

## Build authorization checklist

I would authorize the first builds after these conditions are met:

1. **Source routing, IDs, rollout counts and provenance validated.**
2. **jpa-replica remains free of new math supplements.**
3. **Final-file statistics include caps, cells, source mix and global trimming.**
4. **jpb’s scope is explicit; preferably use the strengthened question-first control.**
5. **A clean training-reference pathway is defined at both pool and pair selection.**
6. **Rule variants use stable random priorities and matched-exposure comparisons.**
7. **Extra math correctness starts small, length-balanced and separately named.**

The most important lesson from the code is that the effective intervention is not a flag such as “ratio=1.5.” It is **which questions and pair kinds survive the entire pipeline, and how many optimization steps they receive**. Jasmin should measure and control that object first.