# Path 1 (recipe 2) — read this first (14 Nov afternoon)

Your consult19 plan: step 1 offline DPO from the original (r2s1b), step 2 = shorten ONLY IFEval + tool calls (math / code:
length-held be-right + loops), step 3 = a pure repair. Steps 1 and 2 are done.

## Step 1: r2s1b-s180 (2,880 pairs, constant lr 2e-5, rank 16, beta 0.1, 180 steps): FULL 50.3 / -1.7
(math 62 / -4.5, code 54 / -3.1, IFEval 38 / +1.0, simple 42 / -3.0, several 45 / +0.9, no-fit 45 / +3.5).
## Step 2: r2s2 from r2s1b-s180, preset r2s2a, 1,600 pairs, constant lr 1e-5, beta 0.1, 80 steps, 1 pass.
Data: step-2 questions (test-matched bins) + extra test-matched questions where pairs ran short, incl. Haiku-generated where
our sources were used up (IFEval requests written by Haiku, gen_ifeval_requests.py; simple tool questions written by Haiku,
gen_tool_questions.py): code/, EXPERIMENT_LOG 13-14 Nov. Pairs per type all at quota (math 320, code 320, IFEval 480,
simple 180, several 140, no-fit 160).
Training: loss 0.693 -> 0.59-0.62, ordered right 54-74% (slow, one pass, every batch new pairs).
Dev-512 / Dev-SVAMP: s20 51.1 / +1.03, 55.4 / -0.75; s40 55.0 / +2.35, 55.4 / -1.00; s60 55.1 / +1.54, 54.5 / -1.25;
s80 55.5 / +1.95 (Dev-SVAMP pending in the log).
FULL TESTS (2,759 x 8; merged onto r2s1b-s180 — an earlier test merged onto the original by mistake, 5.3 / +0.3, invalid):
| checkpoint | full test | above the curve |
|---|---|---|
| **r2s2-s80** | **53.7 / -1.27** | +2.36 |
| r2s2-s40 | 52.9 / -1.74 | +1.73 |
Dev over-read accuracy by 3.2-4.1 pts on this lineage (s80 Dev +1.95 vs full -1.27).
Loss analysis (loss_analysis/r2s2.txt), net accuracy per type vs the original, s80: math -3.25, code -2.38, simple -4.69,
several -0.69, IFEval +1.20, no-fit +6.15 (causes per type inside: mostly stopping early / short thinking; see the file).

## Questions from step 3's pairs supply so far: data/r2-step3-questions.jsonl (the step-3 set, test-matched, never used:
math 650, code 339, IFEval 566, simple 200, several 159, no-fit 220); more can be generated with Haiku where our sources
are used up (user's rule), matched to the test by the original's solve-rate bins.
