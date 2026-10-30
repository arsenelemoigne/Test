# ThinkingCap / Qwen3-0.6B: why the project is stuck, and a strategy to get unstuck

Analysis of `THINKINGCAP_FULL_DOSSIER.md` and `EXPERIMENT_LOG.md` (state 30 Oct 2026, 14:00). All numbers below are
copied from those files or computed from them with the official scoring formula.

## 1. Where you actually are

The frontier has not moved since 23 October. Every clean candidate sits on one line: roughly 1 point of accuracy per
3 to 4 points of tokens saved, once past ~40% saved.

| model (fresh 240 x 8) | date | saved | acc |
|---|---|---|---|
| ca1 soup 10 | 23 Oct | 48.6 | -1.8 |
| rc1-rl-40 | 23 Oct | 53.9 | -2.1 |
| cb2 step 30 | 23 Oct | 46.9 | -2.6 |
| balanced soup 20 | ~22 Oct | 53.8 | -2.5 |
| mf1 soup 25 | 26 Oct | 45.6 | -1.4 |
| lpc-l2s50e soup 100 | 29 Oct | 45.8 | -1.3 |
| lpc-adar1 soup 25 | 29 Oct | 43.0 | -1.9 |
| lpc-press soup 40 / 50 | 30 Oct | 45.5 / 42.4 | -5.2 / -2.9 |

Competitors (full official eval x 8): michal-bida 47.0 / -0.78, Lightning McQwen3 43.2 / -0.85. Your target: >50% at
better than -1. You are about 1 point of accuracy and 5 points of saving away, and that gap has been constant for a week
while the method count kept growing (MVT cuts, fading, steering, probes, stop-token losses, DRPO, ShorterBetter,
Wanda, 15 pool versions).

## 2. Three things the data says are wrong with the process

### 2.1 You cannot measure the difference you are trying to make

- Sampling noise, same model, same 240 questions, two runs: 0.9 pts (rc1-soup-30: -2.9 then -3.8).
- Question-draw noise, same model, two 240-question sets: 3.1 pts (lpc-press soup 40: -5.2 fresh vs -2.1 refs).
- Training-seed noise at the same step: 2 to 5 pts (round 2 retrain -3.6 vs -0.9; round 4 retrain -7.5 vs -2.4).
- Screen (1 seed, choosing x4) to confirm (3 seeds, fresh x8) gaps: 1.3 to 5 pts; lpw-press went +0.4 -> -5.2.
- Checkpoint trajectories are non-monotonic (lpv-base: -1.7, -1.9, +0.2), which is noise, not learning.

The target is "-0.8 rather than -1.3". The 95% interval on a 240 x 8 accuracy change is about +/-2 pts. The decision
rule (compare at the same checkpoint, pick the best of 3 to 7 checkpoints) then adds a winner's-curse bias of roughly
+1.5 to +2 pts on the selection set, which is exactly the size of the screen-to-fresh gaps you keep seeing. Most of what
looks like "every variant lands on the same line" is that the variants differ by less than the noise.

The fix is cheap: the log says fresh 240 x 8 takes ~13 min and the full official 2,759 x 8 takes ~2.5 h on one L4.
The full eval has 11.5x more questions, so its interval is about +/-0.6. That is the only instrument that can resolve
-0.8 from -1.3.

### 2.2 The cut is allocated backwards against the scoring formula

Tokens saved = 1 - geometric mean of the four per-benchmark token ratios. Each benchmark counts the same whatever its
absolute length. Accuracy change = plain mean of the four. So the optimum cuts hardest where the accuracy cost curve is
flattest and gently where it is steep. Your data on the cost per benchmark:

| benchmark | evidence from your runs | verdict |
|---|---|---|
| IFEval | thinking off: 703 -> 284 tokens at -5.0; cut curve flat (0.77 at 0-20% kept vs 0.73 full); trained models hit +50-64% at -0.6 to +1.3 (r2g, balanced, round 5, Track 2 MVT +53-67% / +3.7) | cheap up to ~60% (floor is the answer itself, ~280 tokens) |
| MBPP+ | cut curve flat (0.46 at 0-20% vs 0.50 full); +50-73% at -0.3 to -2.8 (stopB3 s25, l2s50e, drpo1-20, round 5); only breaks at 85-95% (-7 to -10) or with thinking off (-9.1) | cheap up to ~65-70% |
| BFCL | simple robust (+39-42% at ~0); multiple -3 to -6 at +40%; no-fit -3 to -7 when the "does any function fit" check is dropped | moderate: ~40-45% pooled at -1 to -2; with full-eval weighting simple is 48% of BFCL, which helps |
| SVAMP | thinking off: 89.7 -> 17.5; cut curve rises to the end (0.26/0.34/0.48/0.58/0.55/0.73); every model: +50% -> -3 to -5, +65% -> -4 to -6, +75% -> -8 to -10; lost answers "restate the question in one paragraph then answer" | expensive: the computation and one re-read are needed; SVAMP alone is 25% of the accuracy score, so -4 on SVAMP is -1.0 overall |

Now look at how the best clean model spends its cut (lpc-l2s50e soup 100, fresh): SVAMP 54%, MBPP+ 53%, IFEval 41%,
BFCL 34%. It cuts math the most and the two cheap benchmarks the least. rc1-rl-40 is the same shape (65 / 59 / 46 / 43).

Same formula, different allocations (computed; the first two lines reproduce the logged full-eval-weighted numbers):

| allocation svamp / mbpp / ifeval / bfcl | overall saved | expected acc (from your per-type history) |
|---|---|---|
| 54 / 53 / 41 / 34 (lpc-l2s50e, actual) | 46.1 | -1.2 (measured) |
| 65 / 59 / 46 / 43 (rc1-rl-40, actual) | 54.2 | -2.1 (measured) |
| 54 / 65 / 58 / 34 (same math, cheap types pushed) | 54.0 | about -1.5 |
| 45 / 65 / 58 / 40 | 53.1 | about -1.0 |
| 40 / 65 / 58 / 40 | 52.0 | about -0.8 |
| 30 / 70 / 60 / 45 | 53.6 | about -0.7 |

Pushing IFEval and MBPP+ to where your own history says they are still nearly free buys ~8 points of saving. Giving
part of that back on SVAMP buys the accuracy. The target region (>50% at about -1) is reachable with the levers you
already have, by moving pressure between types rather than by finding a new method.

Two runs already point this way and are the most important things in the queue:
- lpw-math (math and no-fit protected: no shortening below 80% solved, right>wrong pairs): screen +50.9 / -0.6, the best
  screen of the project. Its confirm lpc-math is running.
- drpo3b/d/f (math protected): SVAMP still saved 21-30% from spill-over at -0.3 to -1.6. Math does not need its own
  shorten pairs to get a quarter shorter; the other types' lesson carries over.

### 2.3 Effort is going to lines with no path to the score

- SPARSE / Wanda: unstructured sparsity saves no tokens; the pruned models think 8-9% longer. It is interesting and
  irrelevant to the submission.
- Pool representativeness (dp1 to dp15, three days): lpv-base on dp13 scored the same as lpb-l2s50e on dp10 ("the
  upgraded pool did not change the screen picture"); the 33-34/36 misses move between parts with the check's own noise.
- Stop classifiers, probes, stop-token losses (gold1, teach1): the test-time gate is ruled out by your own rule; baked
  into weights it collapsed or underperformed natural pairs (stopB3 s25 34.6 / -0.4 at best).
- RL on top of DPO: rc1 RL gained 0.8 pt at -1.7 saving; rr1, sb1, sb2, drpo1, drpo2 all lost; drpo3 from raw is too
  slow for the guard. The literature you read says the same: RL wins only at aggressive compression.

What did move the needle: natural-answer DPO pairs (shortest right vs longer right, guards, difficulty protection),
3-seed soups, and anything that protects or repairs math (lpw-math screen, mf1: SVAMP -4.1 -> -0.3 at 50% math saving).

## 3. Strategy

### Step 0 (today, no GPU): read your own results properly
`runs/bon/results.csv` has every score. Plot accuracy change against tokens saved for every checkpoint of every run on
the fresh set (and separately for choosing). Compare runs at matched saving, not at matched step: a slower recipe
(lpw-gentle at step 75: +25.7 / -0.9) was dropped for being slow, not for being on a worse line. Fit the line; anything
above it by more than 1.5 pts on more than one checkpoint is a real lever. Also tabulate per-type saving for the top
five models so the allocation in 2.2 is visible.

### Step 1: change the measurement protocol before training anything else
1. Screens (240 x 4, 1 seed) only to catch disasters (any type below -5) and to read saving. Never rank accuracy on them.
2. Confirms: 3-seed soups, scored on at least 200 questions per part (1,200 x 8, about 65 min), disjoint from the
   choosing set and from the fresh set, which has also been used for selection since 23 Oct.
3. Anything within 3 pts of target gets the full official eval (2,759 x 8, 2.5 h). The two L4s can full-eval six to
   eight models overnight. The v1 gate should be stated on the full eval, because that is the number competitors report
   and the only one with a +/-0.6 interval.
4. Pre-register the pick: the first soup that meets the bar on the full eval is v1. No best-of-seven.

### Step 2: one run that implements the allocation (the main bet)
Recipe = lpw-math (math and no-fit protected) plus harder pressure on the cheap types:
- IFEval and code shorten pairs: ratio >= 1.5x, shorten:be-right cap 3-4:1, drawn from all solve bins except never-solved.
- BFCL simple shorten pairs at the current cap; multiple and no-fit protected like math (p < 0.8, right pairs).
- Math: no shorten pairs at all in the first version (rely on spill-over for ~25-30%); right>wrong and guard pairs only.
- Train longer than 86 steps: 150 to 200 steps with 6,000+ distinct pairs so IFEval and code reach saturation; save every
  10 steps; 3 seeds from the start, no 1-seed screen.
- Target per type: SVAMP 30-40, MBPP+ 65, IFEval 58, BFCL 40. Expected overall 52 +/- 2 at about -1.
If lpc-math confirms at >= 48 and better than -1.2 on fresh, send its soup 50 and 60 to the full eval in parallel with
this run; it may already be v1.

### Step 3: fix how math shorten pairs are built (second bet, math only)
The shortest-of-10 right reply on a SVAMP-like question is often the one that skipped the re-read, so long2short teaches
"skip verification", which is exactly what SVAMP punishes. Two alternatives you have already built but never ran:
- First-correct-answer + one check (Chen et al. 2412.21187, your recipe R7): chosen = the model's own right reply
  truncated after the first final answer and one verification, closed with </think>, verified by 20 forced re-answers
  (>= 19/20, your label_cuts machinery); rejected = the same reply full length. Shared prefix, computation intact,
  typically 50-65% of the reply. Chen et al. report this at -45% with no loss where "shortest reply as chosen" gave -22%.
- TokenSkip densification (M12, tsk1 built, never trained) for math: compress words, keep every computation step.
Swap these in for math in the Step 2 recipe once Step 2 has a confirmed number, so the effect is attributable.

### Step 4: a repair stage after shortening
mf1 showed accuracy can be bought back after the fact (SVAMP -4.1 -> -0.3) but its guard pairs (longest right > shortest
wrong) and the NLL anchor on long chosen replies re-lengthened every type (overall 53.9 -> 45.6). Redo it with
length-matched right>wrong pairs only (+/-25%), all types, no guard pairs, no NLL anchor, 20-40 steps, 3 seeds. Expected:
+0.5 to +1 pt at a cost of 1-3 pts of saving. Apply to the Step 2 soup.

### Step 5: park
Wanda/SPARSE, pool iterations, probe zoos, stop-token losses, and RL until an offline v1 has passed the full-eval gate.
If a v1 passes with margin, RL is the one thing to try after, with the guard and per-type λ you already built.

## 4. What to expect
Honest ceiling with current data and methods: 50-54% saved at -0.7 to -1.2 on the full eval. That beats both public
repos on saving and matches them on accuracy within the measurable interval. "-0.8 on 240 questions" was never a
measurable goal; "beat 47.0 / -0.78 on the full official eval" is.

## 5. Checklist for the next 48 hours
1. results.csv: acc-vs-saving plot per run, iso-saving comparison, per-type saving table of top five models.
2. lpc-math soups 40-70 on fresh; if >= 48 / better than -1.2, full eval on soups 50 and 60 tonight.
3. Launch the Step 2 run, 3 seeds, checkpoints every 10, confirm on 1,200 x 8, finalists to full eval.
4. Build first-correct + one-check math pairs (CPU/GPU labelling, ~1 GPU-h) ready for Step 3.
5. Write the v1 gate as a full-eval number and stop the 240-question gate.
