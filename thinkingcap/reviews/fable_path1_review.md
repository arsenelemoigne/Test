# Independent review of Path 1 + the log (Claude Fable 5.2, 11 Nov 2026, while consult8 runs)

Read: PATH1_FINAL.md, EXPERIMENT_LOG.md (state, 25 Oct thinking-off, 9-11 Nov), all_runs.json (197 screens + 8 full tests,
per part), repair_pairs.py, the earlier reviews, the competitor study, the challenge README. Nothing was run on a GPU.

## Verdict
Path 1 is a clean, well-guarded rewrite of what already gave our best full tests (shorten hard, then repair). It holds no new
ingredient, so the expected result is our existing frontier: about 47 / -0.9 to 50.6 / -1.9. My estimate of landing in the box
(> 50% at better than -1) on the full test: roughly 1 in 6 (a judgment, not a measurement). Round 1 is still worth running:
cheap (no generation), gated, and the base of any variant.

## Is the objective achievable?
Plausibly, but not by more of the same. Full-test frontier: rpI-120 41.9 / +0.1, rpE-80 47.0 / -0.9, rpE-60 47.8 / -1.2,
rpE-40 50.6 / -1.9, rc1-soup-30 57.4 / -3.5. At 50% saved we sit near -1.75; the gap is ~3 points of saving or ~0.75 point
of accuracy. A single full test carries about +-1.2 saving / +-0.7 accuracy, and identical recipes re-run differed by 2.7 to
7.9 points of saving (bidax2 vs bidax3, runa2 vs runa2r): a robust pass needs a true ~52-53 / -0.5.

## The unexploited lever: where thinking is worth nothing
Original Qwen, thinking fully off, FULL test (ntfull2) vs our best model (rpE-80 full test):

| type | thinking off: accuracy change | rpE-80: saved / accuracy |
|---|---|---|
| math (SVAMP) | -68.1 | 54% / -1.2 |
| code (MBPP+) | -11.6 | 46% / -1.0 |
| IFEval | -1.3 (62% saved) | 45% / -2.6 |
| one function | -10.7 | 37% / +0.1 |
| several functions | -20.8 | 39% / +0.1 |
| no function fits | +6.9 (96% saved) | 50% / +3.4 |

- On IFEval every trained full test (7 models: 41-53% saved at -1.1 to -3.0) is beaten by the raw model with NO thinking
  (62% / -1.3). Across the 197 screens IFEval's loss is flat (-1.3 to -2.5 in every saving bin, even under 10% saved): it is a
  side effect of training, not the price of cutting. No run ever pushed IFEval past 58% or no-fit past 64%.
- Arithmetic with the official weighting, other types kept at rpE-80: IFEval at 62% -> headline 51.5 / about -0.6; IFEval at
  57% (a ~30-token think) -> 49.9; IFEval 57% + no-fit 75% -> 51.5; IFEval 62% + no-fit 85% -> 53.6 / about -0.5. On paper
  only: no model of ours has learned this in the weights.
- Path 1 goes the other way: math gets the strongest shortening (60% "shorter" pairs, where thinking is worth 68 points);
  IFEval the weakest (counting-rule questions, about half, get no "think less" pair at all).

## Alternative / complement: the user's own budget ladder (BUDGET_LADDER_PLAN.md, 10 Nov), minimal version
1. Measurement only (~2-3 GPU-hours, no training): raw Qwen on our own training questions at thinking budgets 0 / ~30 / ~60
   tokens vs natural, per type; which questions are as right with a short think.
2. If confirmed, add "short think" pairs for exactly those questions (chosen = the short-think right answer with a response at
   least as long; rejected = the long-think answer) to Path 1's rounds; eligibility per QUESTION (measured), not per benchmark.
3. Keep a short REAL think (10-60 tokens), never empty: README rule 1 wants a model "that thinks before it answers" and lists
   "turn thinking off" under not expected.
4. Risk: the stop habit spreading to math / code (stopB5: "stop habit spreads to every type"; basta's gate trained into the
   weights: -3.3). Watch SVAMP thinking length and accuracy on Dev-512; keep the be-right pairs on math / code.

## Red flags
1. No new ingredient (above). Round 2 = rpE's repair; its new parts (25% shorter, count rules, capped guards) each read as
   noise or as a move along the curve in the log (rpH, od1).
2. Allocation: cuts math hardest, IFEval least (above).
3. Round 2's "shorter" bar is 6/8: a half-solved question passes 14.5% of the time (8/10: 5.5%; 7/8: 3.5%). Fixed in round 1
   and in Path 2, not here. Suggest --shorten-min-right 0.875.
4. Dev-512 cannot separate round-2 steps: +-1.7 accuracy per score, ~2 points optimistic (2 models so far). "Most accurate of
   steps 40 / 60 / 80" mostly picks luck: fix the step in advance or use the paired --vs comparison.
5. Seeds: with 2.7-7.9 points of run-to-run spread, two seeds can straddle the gate (52.9% on Dev); say now what happens then.
6. od2 (GPU 1): od1 already showed the online loop slides along the curve; od2 has ~8 rounds left, overlaps Dev-512 and is
   read on the 720. Hours of GPU for an answered question.
7. Code vs plan (repair_pairs.py L215-216): when IFEval has fewer counting-rule pairs than half the quota, the other half
   grows to fill it (n - len(cnt)): a refill the "never refilled" rule forbids; "shorter" share on IFEval can reach ~40%
   instead of 25%. Also the --mix-preset help text still shows the old 35/50/15 shares.
8. Design-level test exposure: the per-type shares were shaped by ~200 looks at the 720 and 8 full tests; Dev-512 only guards
   the step choice. Say so in IDEA.md.
9. Strategy: the README's own bar for "a good result" is > 20% saved at <= 0.2 accuracy point per 1% saved; rpE-80 (47.0 /
   -0.9) and rpI-120 (41.9 / +0.1) pass it ~10x over and match the best public entries (bida 47.0 / -0.78). It asks for the
   idea, several checkpoints along the curve, negative results and the agent traces, and warns against ideas "hardcoded to a
   very specific" setup. The 50 / -1 box is ours, not theirs.

## Not verified
p1src's real pair supply (stored answers are on Drive), dpo.py / track.py line by line (left to consult8), the full-test
per-part token counts behind the arithmetic (fresh-set means used for the three BFCL parts).
