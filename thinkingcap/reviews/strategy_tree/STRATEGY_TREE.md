# ThinkingCap strategy tree (20 Oct - 15 Nov 2026)

Every method family we tried to make Qwen3-0.6B think shorter, with its best measured result and why it stopped.
Sources: `EXPERIMENT_LOG.md` (the "why"), `reviews/strategy_tree/frontier_points.json` (numbers, preferred),
`reviews/consult26_online/pack/all_runs.json`, `reviews/pertype_best_full.md`, `reviews/od5c/REPRODUCE_od5c.md`.
`FINAL_REPORT.md` (22 Oct) is outdated: its "balanced" model was never full-tested.

**How to read the numbers.** "saved / acc" = tokens saved (%) / accuracy change (points) against the original model.
- **Full** = the official test, 2,759 questions x 8 answers, BFCL pooled 400/200/240, 4 benchmarks weighted equally. The only number that counts.
- **720** = our 720-question screen x 4. Its error vs the full test depends on the model (accuracy 1.5 too kind for ca1, 1.1 too harsh for rc1-rl-40; saving up to 3.4 too high).
- **Dev-512** = 512 own held-out questions x 8. It over-reads accuracy by ~2-3 points (rc1 lineage +2.2, od5c r16 +3.0, r2s2-s40 +4.05).
- **240** = 240 real test questions x2 / x4 / x8 ("usual", "choosing", "fresh" sets, October): noisy, and the sets used to pick checkpoints flattered by 1-3.5 points.
- **vs curve** = acc minus acc = -0.388 + 0.0899 s - 0.0028 s^2 (s = % saved), fitted on 2 Nov to 43 points of the 720 screen. Given only for full tests.
- Targets: > 50% saved at better than -1 (the "box"), margin > 52%. BottleCap's bar (> 20% shorter, <= 0.2 pt per 1% saved) was passed by almost every run above 20% saved.

---

## (a) Summary of the top-level families

| # | Family | When | Best official full test (2,759 x 8) | vs curve | Verdict |
|---|---|---|---|---|---|
| 1 | Prompt-only / test-time | 25 Oct - 4 Nov | Thinking off: ~90% / -22.3 (not allowed as a solution) | -7.3 | dead end |
| 2 | SFT on the model's own short answers | 20-22 Oct | no full test (best: 240 x4, +12.5 / -0.7) | - | dead end |
| 3 | Truncation and early-stop rules | 20-27 Oct | rc1-soup-30 (calibrated-cut DPO round) 57.4 / -3.5 | +0.95 | superseded (became the base of the repair line) |
| 4 | Activation steering and weight edits | 21 Oct, 3 Nov | no full test | - | dead end (Astra: not finished) |
| 5 | Pruning | 27-30 Oct | no full test | - | dead end |
| 6 | Reinforcement learning (RL) | 20 Oct - 11 Nov | rc1-rl-40 (RL on top of a DPO model) 55.4 / -2.6 | +1.40 | dead end |
| 7 | Offline DPO (preference training on stored answers) | 21 Oct - 15 Nov | rpM-60 54.1 / -0.93 (6-stage chain); clean 3-step recipe r2s3b-s40 51.3 / -0.78 | +2.79 / +2.36 | superseded; r2one not finished |
| 8 | Online DPO | 10-15 Nov | **od5c r16 54.8 / -0.78**, r15 54.0 / -0.56 | +3.09 / +3.14 | **kept** |
| 9 | Built on a competitor's recipe (excluded) | 27 Oct - 6 Nov | bida copy 51.2 / -3.0 | +0.13 | banned by the user |

DPO = training on pairs "this answer is better than that one" (chosen vs rejected). LoRA = a small trainable add-on,
merged into the weights at the end. A "soup" = the average of several seeds' adapters.

## (b) Why online DPO (from the measured numbers only)

1. One offline DPO round from the original lands about where its pairs point: Path 1 round 1 full 43.4 / -1.5, recipe 2 step 1 (r2s1b-s180) 50.3 / -1.7.
2. Offline reached the box only by chaining: rpK-40 51.2 / -0.8 and rpM-60 54.1 / -0.93 sit on six stages (A1 -> A2 -> ca1 -> rc1 -> rpE -> rpK/rpM).
3. The clean 3-step offline recipe from the original reached the box but not the margin: r2s3-s30 51.5 / -0.84, r2s3b-s40 51.3 / -0.78, r2s3b-s30 51.7 / -0.86.
4. RL from the original barely shortened: GRPO grA/grB/grC <= +1.5% saved (720), gr2f 11.3 / +0.3 (720); the best RL full test, rc1-rl-40 55.4 / -2.6, is RL on top of DPO.
5. Online DPO without a brake overshot: od3 r4 full 60.5 / -4.79 (math -9.4); with a per-type brake (od4) 51.6 / -1.3.
6. od5c (per-question rule vs the original, recovery pairs, brake) is one run from the original with two rounds in the box above 52%: r16 54.8 / -0.78, r15 54.0 / -0.56, about +3.1 above the curve (clean offline: +2.36).
7. Weakness: math is still -2.8 at r15 and r16; the headline is helped by no-fit (+5.9 / +6.9) and IFEval (+0.5).
8. Weakness: the -1 goal is not statistically secure. r16's 95% interval is -1.55 to -0.07 (from `reviews/reproduction/FRESH_RUN_BRIEF.md`; not found in the log), and round 10 of the same run was 54.1 / -1.68.
9. Weakness: slices 10-16 contain questions written by Claude Haiku (no-fit from slice 10, several-function in slice 12, hard no-fit in 13-16); and od5c trained with `--bida-precision`, a precision setting first seen in michal-bida's code. The plain-bf16 rerun of the exact recipe (od5p) is only queued.
10. Weakness: one seed. The seed replication od5r is 8 points behind on saving at round 4 (monitor 41.5 / -0.88 vs 49.6 / -1.19); the log calls it "too early to call".

---

## The tree

### 1. Prompt-only and test-time approaches

**1a. Prompt search (ps1).** 11 system prompts on the original (concise, no re-check, token budget, chain of draft, worked examples). *27 Oct.*
- Best: "budget" prompt +21.5% / +1.9 on 200 training answers per type (training questions only, not the test).
- Why stopped: a screen only; chain of draft "breaks math (-30 pts)"; the planned prompt distillation (Lightning McQwen3's first step) was never run.
- Verdict: not finished (never distilled).

**1b. Budget prompt on test questions (bp1b).** "Think briefly: keep your thinking under N tokens". *27 Oct.*
- Best: ~22% / ~-2.7 (fresh 240 x 8; math 47% / -2.8, code and tools ignore it).
- Why: "distilling this prompt alone cannot reach the goal; at best a math-only ingredient."
- Verdict: dead end.

**1c. Thinking off (nt1, ntfull2).** Reference measurement, not allowed. *25 Oct, 4 Nov.*
- Full: SVAMP -68.1, MBPP+ -11.6, IFEval -1.2, one function -10.7, several -20.8, no-fit +6.9; saved 89-98% per part (IFEval 62%). Official ~-22.3. vs curve -7.3.
- Lesson: thinking is worth ~68 points on math, almost nothing on IFEval and no-fit.
- Verdict: reference only.

**1d. Test-time think gate** (Lightning McQwen3's): excluded by our 25 Oct rule ("no test-time gate"). Never run.

**1e. Budget ladder:** proposed 11 Nov by the Fable review; not found as a run in the log. Not finished.

### 2. SFT on the model's own short answers

SFT = plain imitation training on chosen texts.

**2a. Settle-point pruning SFT (IDEA.md).** Cut each correct trace where p(correct) settles, then SFT. *~20 Oct.*
- Best: round 1 +39.1% / -6.3; round 2 +59.6% / -8.0 (240 x 2).
- Why: the model "learned to underthink" (SVAMP -15 to -24).
- Verdict: dead end.

**2b. Shortest-of-8 SFT (bon-r1, bon r2).** Keep each question's shortest correct answer of 8, SFT. *21 Oct.*
- Best: r2 (clear-gap questions, stronger) +12.5% / -0.7 (240 x 4); r1 +4.4% / -2.8.
- Why: "the same replies that gave +4.4% as SFT give +40-48% as DPO pairs". The contrast, not imitation, is the lever.
- Verdict: superseded by DPO (family 7).

**2c. RFT (SFT on own right answers <= median length), as a repair of rounds 4 and 5.** *22 Oct.*
- Best: rep5 RFT fresh 240 x 8 67.1 / -5.5 (usual set had read -0.2).
- Why: "no technique keeps 60%+ within -2 on unseen questions".
- Verdict: dead end.

### 3. Truncation and early-stop rules

**3a. MVT cut (Marginal Value Theorem).** Cut each trace where the expected gain per extra token falls below the average; SFT on cut traces. Runs mvt256, mvth1024, the "1b" variants (cut ARC/CSQA, keep math whole). *20-21 Oct.*
- Best: mvth1024-full-gsm8k-xgsm8k2 +37.1% / -0.6 (240 x 8, on the questions also used to choose); retrained +31-39% / -1 to -2.
- Why: on math "one-shot SFT on traces that end right after the number appears teaches the model to stop reasoning early" (SVAMP -21 to -24 when math is cut).
- Verdict: superseded by DPO.

**3b. Fading worked examples and marginal-token scoring.** Delete low-value paragraphs/words a little more each round (LLMLingua-2 or our own scores). *20-21 Oct.*
- Best: math-fade-r1 +12.6% / -1.6, SVAMP +24% / +5.0 (240 x 4, per-type grid); fade-gain-marginal r1 +5.3% / -2.0.
- Why: scattered word deletion "gives text the model does not learn to write"; leave-one-out "deletes far too much" (math -46).
- Verdict: dead end.

**3c. Per-type cut grid and combined models.** Best cut per type, then pooled. *21 Oct.*
- Best: combine-final-light +48.6% / -2.3 (240 x 4).
- Why: "The cuts add up": pooled, they push brevity everywhere (code -12.5 without any code trace).
- Verdict: superseded.

**3d. Calibrated cuts, "stop as soon as you're sure" (rc1, rc2).** Cut points labelled by 20 re-answers, then DPO with "stop here" pairs. *23-24 Oct.*
- Best: rc1-soup-30 **Full 57.4 / -3.5, vs curve +0.95** (fresh 240 x 8 had read 55.6 / -2.9). rc2 fresh 63.4-67.0 / -4.9 to -6.2.
- Why: "training on 'stop here' pairs taught the model to stop earlier, not to know better when it may stop"; rc2 "buys 4-13 points of saving at 2.4-4 more points of loss, mostly math".
- Lesson: an overshoot model that a later repair can pull back. rc1-soup-30 is the start of the repair line (7e).
- Verdict: superseded.

**3e. Stop pairs from the original (stopB, B2, B3, B5, stopV2, stopV3).** Original's own "safe cut" vs full reply as DPO pairs. *24-27 Oct.*
- Best: stopB3 soup 25 +34.6 / -0.4 (fresh 240 x 8); re-scored on the 720 later at 39.6 / -3.2. stopV3 plateaued at 10-25%.
- Why: "Every DPO stop run follows one curve set by pairs SEEN"; code collapses past ~480 pairs; stopV2 got longer because its "keep thinking" pairs rewarded length.
- Verdict: dead end.

**3f. Confidence-based stopping and stop-token training (judges, experiment 5, gold1, teach1/teach1c).** Hidden-state probes or the model's own confidence decide where `</think>` should come; a loss on p(`</think>`). *24-27 Oct.*
- Measured: own confidence was the best timing signal (timing-only AUC 0.87); strict "perfect stop" ceiling on the original: math 41-47%, code 72-74% of the thinking.
- Training: gold1 96% / -62.5; teach1c +85.9 / -49.4 (fresh 240 x 8).
- Why: "the model no longer writes an answer at all"; the unanchored stop-token loss "is broken, not a dose problem".
- Verdict: dead end.

### 4. Activation steering and weight edits

**4a. Steering for brevity (Track 2d).** A "short vs long" direction added to the residual stream. *21 Oct.*
- Best: direction A, layer 9, x5: +17.6% / -0.1 (240 x 2, not confirmed at x4); stronger pushes -5 to -8.
- Why: "steering shortens the average reply, it does not make much shorter best-of-N training data"; per-type directions nearly unrelated.
- Verdict: dead end.

**4b. Loop-direction and copy-head edits ("Chirurgical", ch1).** Replicate the weight-editing paper's loop fix on the raw Qwen. *3 Nov.*
- Best: direction L26+L27 a2 on dev 512 (paired): -3.7 +- 1.6 accuracy, +3.0% saved, capped answers 1.2 -> 2.7%. No edit kept.
- Why: "the paper's loop fix does not transfer ... real loops are rare (4% of answers) ... The random direction did better than the 'loop' direction."
- Verdict: dead end.

**4c. Astra / Argon weight editing by a frontier model.** Harness built and tested. *3-4 Nov.*
- Only result: a Mac test, 8 easy questions, thinking -3.0% (Astra's own verdict: "not a validated repair").
- Why stopped: the real run was planned "on top of v1" (bidax5q, family 9); parked, then v1 was abandoned.
- Verdict: not finished.

### 5. Pruning

**5a. Wanda weight pruning (SPARSE arm C).** Zero the least important weights. *30 Oct.*
- Best: 10% pruned +1.8% / +0.2 (choosing 240 x 4); 20-30% pruned think ~8-9% LONGER at ~-2.
- Why: pruning keeps behaviour or makes thinking longer; contrastive Wanda parked (3 Nov), never built.
- Verdict: dead end.

**5b. TokenSkip (word-level trace compression, tsk1).** Built 27 Oct; the first run crashed on GPU memory, then paused for the dataset work. Not found as a trained result. Verdict: not finished.

### 6. Reinforcement learning

**6a. Early RL loops (TokenProbe-KL + ShorterBetter).** *~20 Oct.* Best loop 1 +5.7% / -1.2 (240 x 2). Loop 2 "lengths not falling". Dead end.

**6b. RL on top of a DPO model (rlcal on rc1, rr1).** Reward = right x (1 - length penalty within the group). *23 Oct.*
- Best: rc1-rl-40 **Full 55.4 / -2.6, vs curve +1.40** (fresh 240 x 8 had read 53.9 / -2.1). Not in frontier_points.json.
- Why: "RL with alpha 0.15 on repeated questions does not move rc1 off its frontier"; rr1 at alpha 0.3 "buys +1 point of saving for -1.3 points, mostly math".
- Verdict: superseded (used as a base for 6c and mf1, then dropped).

**6c. DRPO (drpo1, drpo2 from rc1-rl-40; drpo3b/d/f from the raw model).** *25-27 Oct.*
- Best: drpo2-5 55.5 / -2.0; drpo1 step 40 73.3 / -6.9 ("math collapses"); gentle from raw drpo3f-15 +24.6 / +0.5 (all fresh 240 x 8).
- Why: strong pressure "affordable on code/IFEval but not on math or tool choice"; gentle versions save ~25%.
- Verdict: dead end.

**6d. ShorterBetter reward (sb1, sb2).** *26-27 Oct.* sb1 collapsed by step 33; sb2-5 53.7 / -2.9 (fresh 240 x 8), stopped by its guard at step 10. Dead end.

**6e. GRPO from the original (grA, grB, grC; tp0 smoke; gr2f).** Group-relative RL with a length term. *3 Nov, 6 Nov.*
- Best: gr2f step 40 11.3 / +0.3 (720); grA/grC finals thought 6-7% LONGER.
- Why: the length term was "tiny next to the right / wrong reward" (consult1: 115-1,035x smaller); 3 Nov: "GRPO stopped. No GRPO run shortened thinking"; gr2f "saving is small and does not grow steadily".
- Verdict: dead end.

**6f. COPD, on-policy distillation from a Qwen3-4B coach (cp1a, cp1c2, cp1d) + repair rpF.** *9-11 Nov.*
- Best: cp1a step 20 +20.8 / -0.4 (720); cp1c2 step 40 57.1 / -4.7 (on the curve); rpF-80 on cp1c2 48.1 / -2.6 (on the curve).
- Why: the 4B coach "pushes the USEFUL check down twice as hard"; the correctness push "carries a hidden length reward"; consult5: "do not spend GPU days on more COPD variants"; "COPD line closed" (11 Nov).
- Verdict: dead end.

**"Trajectory RL"** under that name: not found in the log.

### 7. Offline DPO (pairs built from stored answers)

**7a. Best-of-8 DPO rounds 1-5, balanced round, blends (Track 2c/2f).** Pairs "shortest right > longest right" and "right > wrong", one short pass per round, then the improved model answers new questions. *21-22 Oct.*
- Best: round 2 (r2g step 10) fresh 240 x 8 51.4 / -1.8; balanced soup fresh 53.8 / -2.5; round 4 fresh 63.2 / -3.9.
- Why: late rounds push math ("SVAMP -7.5 to -10.6") and the lineage saw ~74 MBPP training tasks ("cannot be certified clean").
- Verdict: superseded by the MBPP-free lines (7b).

**7b. MBPP-free clean lines A and B, v2 from scratch, mf1.** Same rounds without MBPP data. *22-27 Oct.*
- Best: ca1-soup-10 (line A round 3) **Full 49.1 / -2.4, vs curve +0.32** (fresh had read 48.6 / -1.8). Line B cb1 fresh 41.0 / -1.1; v2r2 fresh 28.6 / -0.8; mf1 soup 25 fresh 45.6 / -1.4.
- Why: line A's 4th round "is weaker on unseen questions"; v2 stopped after round 3; mf1's guard pairs "teach length to all types".
- Verdict: superseded (ca1 -> rc1 -> repair line).

**7c. LUPIN: a representative dataset first (dp1-dp13), then pair-rule screens (lpb/lpc/lpw, Runs A2 / A-constant / B).** *27 Oct - 1 Nov.*
- Best: lpc-l2s50e-soup-100 **Full 48.0 / -2.08, vs curve +0.44** (not in frontier_points.json). Run A2 sat at 32-33% / -1.0 to +0.4 on the 720.
- Why: "every variant sits near the same saving-vs-accuracy line"; the gap to michal-bida's 47.0 / -0.78 led to the bida control (family 9).
- Verdict: superseded (the dp13 data was kept).

**7d. Per-type adapters and merges (pt-*, ptsolo, lpt6, lpt6mix, lpk), and the pt-mid repair (rpA, rpB, rpC).** One adapter per task type, merged at chosen strengths. *30 Oct - 6 Nov.*
- Best: pt-mid **Full 53.6 / -3.3, vs curve +0.31** (720 had read 55.0 / -2.58). Repair rpC-80 43.4 / +0.43 (720, +2.19 vs curve on that screen); rpB-40 43.4 / +0.10.
- Why: the spill-over is "~rank one" (each adapter shortens every type); "lengths ADD almost exactly as stacked pushes"; the repair "trades length for accuracy". The lpk rerun verdict was later called "not a fair test". "Jasmin" was planned, never trained.
- Verdict: dead end (its lesson, shared habit plus repair, fed 7e).

**7e. The repair line on rc1-soup-30 (rpE, rpG, rpH, rpI, rpK, rpL, rpM, rpN).** Equal-length "be right" pairs on the model's own answers, then "typed" rounds: think less only on IFEval and tool calls, protect math. *9-12 Nov.*
- rpE-80 Full 47.0 / -0.9 (+1.45); rpI-120 Full 41.9 / +0.1 (+1.64).
- **rpK-40 Full 51.2 / -0.8 (+2.33): first full test in the box.**
- **rpM-60 Full 54.1 / -0.93 (+2.79): in the box and above 52%** (CI -1.67 to -0.17, per the log).
- rpL-60 Full 59.3 / -1.7 (+3.20); rpL-80 Full 62.8 / -2.06 (+3.73, the highest above the curve of any full test).
- Why superseded: rpG/rpH on old answers "drift longer"; rpN "changes nothing measurable"; rpL past step 80 "stops calling" tools. Above all it is a 6-stage chain, and the user asked for one clean recipe (11 Nov: "ONE Argon call ... to design a CLEAN recipe from scratch").
- Verdict: superseded.

**7f. Path 1, round 1 from the original (p1r1s0, p1cr1s0).** One big offline round on the original's stored answers. *11 Nov.*
- Best: p1r1s0-200 **Full 43.4 / -1.5, vs curve +0.26**. p1cr1s0 Dev-512 34.6 / +2.0 ("accurate but barely shortens").
- Why: Dev gate closed (45.9 < 52.9), round 2 skipped "as pre-registered"; IFEval saved only 23% because the response guard picked longer responses.
- Verdict: dead end.

**7g. Path 1, recipe 2 (three steps from the original: r2s1b -> r2s2 -> r2s3 / r2s3b).** Step 1 shorten, step 2 shorten IFEval and tools only, step 3 pure repair. All questions matched to the test's difficulty profile. *12-15 Nov.*
- Step 1 r2s1b-s180 Full 50.3 / -1.7 (+1.25); step 2 r2s2-s80 Full 53.7 / -1.27 (+2.36).
- **Step 3 r2s3-s30 Full 51.5 / -0.84 (+2.34); r2s3-s20 52.7 / -0.99 (+2.44, at the edge).**
- **Step 3b r2s3b-s40 Full 51.3 / -0.78 (+2.36); s30 51.7 / -0.86 (+2.36).**
- Why superseded: in the box but under the 52% margin; the repair gives saving back; "consult25 had expected 53.1 / -0.58: the redo did not raise the saving."
- Verdict: superseded (best clean offline result).

**7h. r2one: one offline step from the original (consult27).** *15 Nov.* 3,200 pairs built; training started (step 120 of 200 at the last entry). Result not found in the log. Verdict: not finished.

### 8. Online DPO (the current model answers fresh questions every round)

**8a. od1, from rpE-80.** 4 rounds of 160 pairs, the round's start as reference. *10-11 Nov.*
- Best: round 2 51.5 / -1.2 (720); round 4 53.3 / -1.8 (720).
- Why: "moved rpE-80 ALONG the curve ... not above it".
- Verdict: superseded.

**8b. od2, from the original, moving reference.** 10 rounds. *10-11 Nov.*
- Best: round 4 29.3 / -1.0; round 10 67.1 / -5.6, math -11.9 (720).
- Why: "never stops shortening and accuracy falls steadily"; failure analysis: "the dominant failure is PREMATURE STOPPING"; the reference reset each round, so "nothing holds the model near the original".
- Verdict: dead end.

**8c. od3, "arena overshoot" with a fixed reference and per-type floors.** *13 Nov.*
- Best: r3 Full 58.4 / -4.01 (+0.68); r4 Full 60.5 / -4.79 (+0.41); math -8.9 / -9.4.
- Why: "the fixed anchor and floors slowed it, did not stop it"; the planned type-split repair (rpO) was never queued.
- Verdict: superseded.

**8d. od4 = od3 + a per-type brake.** A type over its saving cap or under -2 accuracy gets no "think less" pairs that round. *13-14 Nov.*
- Best: r8 Full 51.6 / -1.3 (+1.90); r7 Full 55.4 / -2.0 (+2.00).
- Why: consult21: early stopping still "~80% of od4's lost accuracy"; the 7/8 bar "is blind to the original"; no-fit never braked.
- Verdict: superseded by od5.

**8e. od5b (od5 v1 rules: per-question comparison with the original).** *14 Nov.* Rounds 1-3 only (monitor r2 16.4 / -0.52, "half od4's saving"). consult23 found 5 causes; restarted clean as od5c. Verdict: superseded.

**8f. od5c (od5 v2), the result.** 16 rounds of 780 test-matched questions x 8; pairs compare the current model with the ORIGINAL per question (shorten only if as good as the original, recovery pairs where it got worse, loop/format/IFEval-length pairs); fixed reference = the original; brake per type; one adapter, constant lr 1.4e-5, `--bida-precision`. *14 Nov 17:45 - 15 Nov ~11:00.*
- r10 Full 54.1 / -1.68 (+2.04).
- **r15 Full 54.0 / -0.56 (+3.14).**
- **r16 Full 54.8 / -0.78 (+3.09).**
- Per type at r16: math 61 / -2.8, code 59 / -2.0, IFEval 47 / +0.5, one function 49 / -1.1, several 50 / 0.0, no-fit 57 / +5.9.
- Biases: monitor and Dev-512 over-read accuracy by 2.5-3.0.
- Open: the seed replication od5r is running; the plain-bf16 replicate od5p and the extension to round 24 are queued.
- Verdict: **kept** (main submission candidate).

**8g. od5d (od5 v3: freed slots to healthy types, looser rules, plain bf16).** *15 Nov.*
- Best: r11 Full 49.5 / -1.21 (+1.59); r10 Full 51.6 / -1.95 (+1.25).
- Why: "od5c's recipe (v2) beat its v3 rules"; the monitor's over-read "made rules A/B/C call IFEval / simple / several 'healthy' and shorten them".
- Verdict: dead end. It changed the rules and the precision together, so it does not test precision alone.

---

## 9. Runs built on a competitor's recipe (excluded from the clean method; the user later banned this)

These copy or imitate michal-bida's public recipe (pair rules, 1:1 balance, training settings, data sources) or run
paper pair rules on his settings. They are not our method. The ban: 6 Nov ~09:30, "nothing built on another
candidate's recipe will be run or submitted"; 12 Nov, his data sources removed from recipe 2 ("we should not do this").

- **sp1** (27 Oct): "one big self-preference round ... (michal-bida's recipe ...)". Paused for the dataset work, never run.
- **lpc-bida / lpc-bidax** (30 Oct - 1 Nov): his recipe on our dp13 pool. lpc-bidax was stopped (a default rule leaked in) and restarted as bidax2.
- **bidax2, the "bida copy"** (1 Nov): **Full 51.2 / -3.0, vs curve +0.13** (720 read 51.8 / -2.7). "his recipe on our pool does not reproduce his result" (his own 47.0 / -0.78).
- **bidax2s1, bidax3, bidax4, bidax4l, bidax5m, bidax5s** (1-2 Nov): seed, repeat, own no-fit, longer schedule, math fixes. Best ~+0.3 vs curve on the 720; bidax5m "math rambles: longer AND worse".
- **bidax6 / bidax6p / b6dial** (2-5 Nov): his exact settings; precision only; strength dial. His fp32 + fp16 precision "accounts for most of the change his settings made". This is how `--bida-precision` entered the repair line, od2 and od5c.
- **bidax5q, named "v1"** (2-3 Nov): his settings + his math questions. 720 46.7 / -0.59. The Astra run planned on it never happened. **bidax7** (+ his code and instruction sources): 720 best 45.3 / -0.8.
- **D1-D12 pair-strategy screen (lpd-*), on bidax5q's data and settings** (3-5 Nov), 720 finals: DAST 33.1 / -2.3, hard 13.8 / -0.1, route 9.4 / -0.7, budget 49.0 / -1.3, margin 53.3 / -2.6, similar-length right 28.1 / +1.7, two-stage 16.5 / +1.4, math protection 41.7 / -1.0, Ada-R1 65.3 / -6.8, l2s50e 53.5 / -2.7, push harder 52.4 / -3.0, chain-1 17.8 / +0.8. "None beats v1."
- Not run: lpc-bidax8 ("four signals", paused 2 Nov); consult4 plan J ("its core is bida's pair rules + settings", rejected 6 Nov); Path 3 (dropped 11 Nov: it "rested on a bida-pair run").
- Lesson: copying reproduced his saving but not his accuracy on our data. The one piece that stayed in our runs is the precision setting, still to be removed (od5p).

---

## Inconsistencies found

1. **"Best point of any run."** The log (15 Nov 14:40) calls od5c r16's +3.09 above the curve "the best point of any run". By frontier_points.json, rpL-80 (62.8 / -2.06, +3.73), rpL-60 (+3.20) and od5c r15 (+3.14) sit higher. r16 is the best point inside the box above 52%, not the highest above the curve.
2. **Thinking-off reference.** frontier_points.json `off` = 94 / -23, with per-type parts from the October fresh-240 run (SVAMP -72.2, IFEval -5.0). The full test (ntfull2, 4 Nov) gives SVAMP -68.1, IFEval -1.2, official ~-22.3. I trust the full test.
3. **od1 key.** `od1-120` holds 51.5 / -1.2, which the log calls round 2 (rpE-80 + 20 steps = step 100). Round 4 (step 120) is 53.3 / -1.8 on the 720. Either the key or the numbers are off; I trust the log.
4. **Full tests missing from frontier_points.json:** lpc-l2s50e-soup-100 (48.0 / -2.08, 30 Oct), rc1-rl-40 (55.4 / -2.6, 31 Oct), thinking off (4 Nov).
5. **Haiku questions.** The brief says slices 12-16. REPRODUCE_od5c.md says Haiku near-miss no-fit questions (r2k-nofit) already enter slices 10-11, then several-function (12) and hard no-fit (13-16).
6. **r16 interval -1.55 to -0.07:** not found in EXPERIMENT_LOG.md; it is in `reviews/reproduction/FRESH_RUN_BRIEF.md`.
7. **Curve definition.** Described as "fitted over all runs". The log says it was fitted on 2 Nov to 43 points of the 720 screen above 15% saved, and kept fixed since.
8. **Family flag.** In frontier_points.json `fams`, bidax2 (which contains bidax2-full) is flagged `"full": false`. Cosmetic.
9. **FINAL_REPORT.md** (22 Oct) presents a "recommended" balanced model at ~54-59% / -2 to -2.5. Those numbers are from 240-question sets only; the model was never full-tested, and its lineage saw MBPP training tasks.
