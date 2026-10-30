# Build spec: LUPIN v1 candidates from the existing dp13 data

For the model that builds and runs the experiments. Everything here uses data and code that already exist (pool dp13,
10 graded original-Qwen answers per question, `lupin_pairs.py`, `lupin_select.py`, `dpo.py`, `merge.py`, `label_cuts.py`).
Nothing new is sampled from the model for v1. Every choice has a "why" that points at a logged result.

## 0. Principles (read first)

1. **One shortening adapter, per-type pressure controlled by the pair mix, not by per-type adapters.** Per-type adapters
   summed were far too strong (+77% / −9.2; tuned merge +70% / −6.5): the brevity lesson from any one type spills onto
   all types, so four adapters stack four pushes. In one adapter the spill-over is a feature: math gets 21–30% shorter
   with zero math shorten pairs (drpo3b/d/f, lpw-math).
2. **Cut hard where the cost curve is flat (IFEval, MBPP+, BFCL simple), gently or not at all where it is steep
   (SVAMP, BFCL multiple, no-fit).** The score is a geometric mean over the 4 benchmarks, so a point of saving on IFEval
   is worth exactly a point on SVAMP, but costs ~0 instead of ~0.1 pt of accuracy.
3. **Train 3 seeds from the start and soup them.** Training-seed variance is 2–5 pts at the same step (round 2 retrain
   −3.6 vs −0.9; round 4 −7.5 vs −2.4). A 1-seed screen is a coin flip; skip it.
4. **Compare runs at matched saving, not matched step.** lpw-gentle was dropped at step 75 for +25.7 / −0.9 when it
   was simply slower; compare its accuracy to other runs at 25% saved, not to their step 75.
5. **Decide on the full official eval.** 240 × 8 has ±2 pts of noise and the same model moves 3 pts between two
   240-question draws. The full 2,759 × 8 is 2.5 h on one L4 and has ±0.6.

## 1. Common settings for every run

| item | value | why |
|---|---|---|
| start model | original Qwen3-0.6B, thinking on | all v1 recipes start from raw; stacking on rc1-rl-40 inherits its SVAMP −4 |
| adapter | LoRA r32, alpha 64, all 7 linear layers of the 28 blocks, dropout 0 | what dpo.py already does for LUPIN; r16 vs r32 made no logged difference, keep the current |
| objective | DPO, β 0.1, NLL anchor weight 1.0 on the chosen reply | the known-working setting (l2s50e 45.8 / −1.3). Keep the anchor: in these recipes the chosen is always the shorter or the right reply, so it pulls the right way (it only hurt in stopV2 where 53% of chosen were full-length) |
| learning rate | 2e-5, linear warm-up 10 steps, cosine decay to 2e-6 at the last step | 2e-5 constant is what produced every good result; the decay is new and is there to stop the end-of-run wobble (lpv-base: −1.7, −1.9, +0.2 at 25/50/75 is not learning, it is a constant-lr random walk) |
| batch | 16 pairs per step, 1 epoch, no pair repeated | user rule (no second epoch over the same pairs); more distinct pairs did not raise saving (press6k) but they do not hurt |
| max lengths | prompt + reply 4,096; code 8,192; gradient checkpointing on | matches the evaluation caps; capped replies appear as rejected in guard pairs |
| seeds | 3 seeds (data order + LoRA init), identical pair file | soups need identical data |
| checkpoints | every 10 steps, each seed | soups at any step |
| soups | equal-weight average of the 3 adapters at the same step (`merge.py`), merged into the base | the only variance reducer that has worked every time |
| held-out tripwire | 48 pairs, as now | sanity only |
| pool | dp13, the 10 newest graded answers per question | user decision 30 Oct; do not iterate the pool again (dp10 → dp13 changed nothing measurable) |
| excluded sources | no MBPP family, no GSM-IC / GSM-Plus, no prompted answers | user policy; GSM-IC/Plus plateau at 20–40% of thinking unlike SVAMP, so pairs from them teach cuts that are too early for SVAMP |

## 2. Pair kinds (from each question's 10 graded answers)

| kind | chosen > rejected | length signal | use |
|---|---|---|---|
| shorten | a right answer > a right answer ≥ 1.5× longer (1.2× on always-solved) | strong shorten | the engine of saving |
| shorten-median (new, math only) | the **median-length** right answer > the longest right answer (≥ 1.5×) | mild shorten | teaches "drop the runaway tail", not "skip the re-read" |
| guard | the shortest right answer > the longest wrong or capped answer | shorten + be right | protects against the runaway/capped failure (26% of raw math replies were capped) |
| right | a right answer > a wrong answer of similar length (±25%) | none | pure accuracy; the only pair kind that moved accuracy up (lpb-right +2.5, mf1 SVAMP −4.1 → −0.3) |
| hard (DAST) | longer wrong > shorter wrong | lengthen | **do not use**: lpb-dast landed on the same line at lower saving (34–38% / −0.1..−0.8) |
| cut (first-check) | see Run C | mild shorten, same prefix | math only, Run C |

Filters on every pair: chosen never capped; chosen and rejected from different samples; questions never solved (0/10)
give no pair at all (14% of the pool; they only teach "give up"); the 8-word contamination check as now.

## 3. Per-part pair mix (the actual lever)

Shares are shares of the total pair budget; inside BFCL the parts keep 400 / 200 / 240.
`--min-solve` = a question solved below this fraction gets no shorten/guard pair, only right pairs.

| part | share of pairs | shorten | guard | right | min-solve for shorten | why |
|---|---|---|---|---|---|---|
| SVAMP (math) | 15% | **0%** (Run A) | 40% | 60% | — | SVAMP's cut curve rises to the very end (0.26 → 0.73); every logged model with ≥ 50% math saving lost 3–5 pts. Spill-over alone gives 21–30% at −0.3 to −1.6. Right pairs are the repair (mf1) |
| MBPP+ (code) | 30% | 70% | 15% | 15% | 0.3 | cut curve flat (0.46 at 0–20% kept vs 0.50 full); 50–73% saved at −0.3 to −2.8 in five different models; only breaks above 85%. Low min-solve because harder code questions are as safe to cut as easy ones |
| IFEval | 30% | 70% | 15% | 15% | 0.3 | flat curve (0.77 at 0–20% vs 0.73 full); +50–64% at −0.6 to +1.3 in four models; floor is the answer itself (~280 of 703 tokens) so ~60% is the ceiling |
| BFCL simple | 12% | 60% | 20% | 20% | 0.5 | robust (+39–42% at ~0 in every run); 48% of BFCL under full-eval weighting |
| BFCL multiple | 6% | 30% | 30% | 40% | 0.8 | loses 3–6 pts at 40% (wrong enum value, dotted name shortened); shorten only on questions it almost always gets right |
| BFCL no-fit | 7% | **0%** | 40% | 60% | — | the "does any function fit" check is the first thing dropped (−4 to −7, −23 to −39 in Track 3b); lpw-math's protection is exactly this |

Budget: **4,800 pairs → 300 steps.** Raw inventory check against dp10 counts (dp13 is larger): IFEval needs 1,008
shorten of 1,671+ available; code 1,008 of 1,744+; SVAMP 720 right+guard of 1,297; no-fit 336 right+guard of 339
(tight: if short, let `lupin_select` fill no-fit from the top-up nofit1/nofit2 questions first, which are the hard ones).
Cell matching (solve bin × thinking quartile) stays on inside each part; the part shares above override the default
25/25/25/25 (`--part-share`); make sure the "trim every part to the lowest fill ratio" step does not undo the shares.

Why 4,800 and not 3,200: IFEval and code saving kept rising to step 100–150 in l2s50e (IFEval 33 → 52% from step 25
to 150); with 60% of pairs on those two types, 300 steps gives them ~1,900 pairs each, enough to saturate.

## 4. The runs

Four shortening runs, each 3 seeds. A is the main bet; B and C are the two ways to add gentle math saving; D is the
hyper-parameter control. Run A first on both GPUs (3 seeds → ~3 trainings), then B and C, then D.

### Run A — "allocation" (main bet)
Exactly section 3. Expected per type: IFEval 55–60, MBPP+ 60–65, BFCL 38–42 pooled, SVAMP 25–35 (spill-over).
Computed overall at (30 / 62 / 57 / 40): **~50% saved**, expected accuracy about −0.6 to −1.0.

### Run B — A + gentle math shorten
Same as A, but SVAMP pairs become shorten-median 30% / guard 30% / right 40%, min-solve 0.8 for the shorten-median pairs
(only on questions solved ≥ 8/10). Expected SVAMP 38–45% at −0.5 to −1 extra on SVAMP (−0.15 to −0.25 overall);
overall ~52–53%.
Why median not shortest: the shortest-of-10 right math reply is the one that happened to skip the re-read; SVAMP's
traps (irrelevant numbers, changed wording) are exactly what the re-read catches ("restated the question in one
paragraph then answered wrong" is the logged SVAMP failure).

### Run C — A + first-correct-plus-one-check math cuts (recipe R7, built, never run)
Build math shorten pairs from the model's own right replies:
1. For each SVAMP-part question solved ≥ 5/10, take the longest right reply.
2. Find the first paragraph that states a final numeric answer equal to the gold, then keep through the end of the
   **next** paragraph (one verification). Close with `</think>` and the original answer text.
3. Verify with `label_cuts.py` machinery: 20 forced answers from the cut; keep the pair only if ≥ 19/20 right **and**
   the cut keeps ≥ 40% of the thinking (a safety floor against lucky early guesses).
4. chosen = the cut reply; rejected = the same reply full length.
Use these as SVAMP's shorten pairs: 40% of math pairs, guard 20%, right 40%. Budget ~290 cut pairs, so label ~800 math
questions (~1 GPU-h at 20 forced answers each).
Why: Chen et al. (2412.21187) got −45% at no loss with this chosen, versus −22% with "shortest reply as chosen" on the
same model. Shared prefix means the DPO gradient is entirely about "stop after one check", the cleanest math signal
available. Your rc1 cut pairs failed on SVAMP because they cut at the *earliest* safe point (26–30% of the reply);
this cuts at 50–65%.

### Run D — hyper-parameter control
Same pairs as A. β 0.2, lr 1e-5, NLL 1.0, 300 steps. Score the soups at the same **saving** as A's soups (e.g. A at
45% vs D at 45%), not the same step. If D's accuracy is ≥ 0.7 higher at matched saving on the 480-question prefilter
(section 6), rerun B or C with D's settings. Why: slower runs were never allowed to finish; lpw-gentle at step 75 was
on or above the line and was dropped for being slow.

### Not in v1 (deliberately)
- Hard/DAST pairs, prompted answers, cut pairs for code/tools (natural pairs beat them in every comparison).
- Per-type adapters, RL, Wanda, pool re-matching.
- 2 epochs. The second round of saving comes from **v2** (v1's own answers on new matched questions, same recipe),
  which is where rounds 2–5 of Track 2e got from 42% to 65%.

## 5. Stage 2 — repair (run on the chosen soup of A/B/C)

| item | value | why |
|---|---|---|
| start | the chosen v1 soup, merged; new LoRA r32 on top | separate adapter so it can be dialled (×0.5..×1) |
| pairs | **right** pairs only (±25% length), all parts at the eval shares 25/25/25/25 (BFCL 400/200/240), from pool questions **not** used in stage 1 | zero length signal by construction; mf1 re-lengthened everything because its guard pairs had the longest right reply as chosen |
| count / steps | 800 pairs, 50 steps, lr 1e-5, β 0.1, NLL 1.0, 3 seeds | small dose: lpb-right gained +2.5 by step 50 and started lengthening by step 75 |
| score | soups at 20 / 30 / 40 / 50 | expect +0.5 to +1.0 accuracy at −1 to −3 saving; stop at the first soup where saving dropped > 3 |

## 6. Evaluation protocol (this is what makes the result real)

1. **Prefilter** every soup at steps 100 / 150 / 200 / 250 / 300 on the two existing held-out sets together:
   `questions_fresh20261022` × 8 and `questions_ref20261024` × 4 (480 questions, ~18 min). Report both weightings; use
   the full-eval weighting (BFCL 400/200/240) for decisions.
2. **Red flag** (stop the run's remaining soups): SVAMP or MBPP+ below −4 on both sets, or any BFCL part below −6.
3. **Per run, pick two soups**: the highest saving with accuracy ≥ −1.0, and the highest accuracy with saving ≥ 45.
4. **Full official eval** (2,759 × 8, ~2.5 h) on each picked soup: 8 full evals for A–D, one night on two L4s.
5. **v1 = highest saving on the full eval with accuracy > −1.0.** Also report the best accuracy at ≥ 47% saved.
   The bar to beat: michal-bida 47.0 / −0.78, Lightning McQwen3 43.2 / −0.85 (both full eval × 8).
6. Repair (section 5) on v1 only if its full-eval accuracy is between −1.0 and −1.6: repair, then full eval again.
7. **Strength dial** for a last 0.3 pt: if the chosen soup is at −1.1, score ×0.9 and ×0.8 on the prefilter (dial
   traded ~linearly: ×0.5 halved saving and gave +2 pts) and full-eval the one that crosses −1.0.

## 7. What each result would mean

| observation | meaning | next |
|---|---|---|
| A reaches ≥ 50 at ≥ −1.0 on the full eval | the allocation thesis holds | submit A; start v2 from A's own answers |
| A at 46–49 / −0.5 | cheap types saturated below hope | B or C adds the missing 3–4 pts of math saving |
| A at ≥ 50 but −1.3 to −1.6 | losses in BFCL multiple/no-fit or spill-over on math | repair stage; if the loss is on no-fit, raise its right-pair share to 80% |
| IFEval saving < 50 at step 300 | pressure too low | raise IFEval shorten share to 80%, min-solve 0.2; or lower the ratio threshold to 1.3× |
| MBPP+ below −3 | code over-cut | min-solve 0.5 for code, shorten share 55% |
| D beats A by ≥ 0.7 at matched saving | the lr was too hot all along | rerun the best of A/B/C with D's settings |

## 8. Compute estimate

300 DPO steps ≈ 2–3 h per seed on an L4 (including reference scoring). Four runs × 3 seeds ≈ 30 GPU-h; prefilters
~20 soups × 18 min ≈ 6 GPU-h; 8 full evals ≈ 20 GPU-h. Total ≈ 56 GPU-h ≈ 28 h on two L4s, so about two days with
the full evals overnight. Run order: A (both GPUs) → C's labelling + B → C → D.
