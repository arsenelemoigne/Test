# "It thinks about half as much. Did it get smarter?" — the data (18 Nov 2026)

Every number below is for the original Qwen3-0.6B ("original") against the best model (online DPO od5c, round 16, "trained"),
on the FULL TEST (2,759 questions × 8 answers = 22,072 answers per model), unless marked otherwise. Status: ✅ measured,
⏳ running / queued on the GPU. The second project (deletion tuning) is NOT part of this document.

Data: per-answer records `reviews/consult30_efficiency/pack/per_answer/*.csv`; answer texts `~/Downloads/tc_full/{base,od5c16,nothink}.jsonl`
(also in the consult30 pack, `traces/`); article data `public/drafts/tc-7k2q/stats.json` on the site.

---

## Point 1 — On average, almost nothing was lost ✅
| | Saved | Accuracy |
|---|---|---|
| Best model (od5c r16) | 54.8% | −0.78 |
| Second seed (od5r r16) | 52.1% | −0.58 |
| Thinking off (original) | 94% | −22.3 |
Official scoring (geometric mean of token ratios over math / code / instructions / tool calls; mean of accuracy changes).

## Point 2 — Question by question, answers move: it is not "the same model, shorter" ✅
Script: `churn_metrics.py` → `runs/churn/summary.txt`. Net drift = Σ(k1−k0)/8 (exact change, question-equivalents);
excess churn = Σ(|k0−k1|/8 − E0) with E0 the exact hypergeometric null (movement a rerun of the same model would produce);
95% bootstrap over questions. **Noise control (original vs itself, answers 0–3 vs 4–7): excess churn +1.6 q [−8.6, +10.9] ≈ 0.**

| Task | n | Net drift (q / pts) | Excess churn (q / pts) [95% CI] |
|---|---|---|---|
| Math | 1000 | −28.4 / −2.84 | +21.2 / +2.12 [+15.2, +27.0] |
| Code | 378 | −7.4 / −1.95 | +6.4 / +1.70 [+2.8, +10.5] |
| Instructions | 541 | +2.6 / +0.49 | +11.5 / +2.12 [+5.8, +17.4] |
| One function | 400 | −4.4 / −1.09 | +13.4 / +3.34 [+8.9, +18.1] |
| Several functions | 200 | 0.0 / 0.00 | +12.1 / +6.07 [+8.1, +16.6] |
| No function fits | 240 | +14.1 / +5.89 | +8.5 / +3.53 [+4.9, +12.1] |
| **All** | 2759 | **−23.4 / −0.85** [−40.1, −4.4] | **+73.1 / +2.65** [+61.2, +85.2] |

Other models for comparison (all tasks): rpM-60 (offline, 48% saved) drift −22.2, excess churn +67.6; rpL-80 (offline, 63%)
drift −57.9, churn +94.7; thinking off drift −800, churn +717. More saving → more churn.
Replaces the old "questions lost beyond chance" (biased when losses and gains are corrected separately; consult30).
Per 100 answers (matched slots): 70 right in both, 5 right→wrong, 4 wrong→right, 21 wrong in both.

## Point 3 — It did not forget how; it became less consistent ✅
Same script. Pass@8 = solved at least once in 8; 8/8 = solved every time.
| Task | Pass@8 | 8/8 |
|---|---|---|
| Math | 96.8 → 96.4 | **70.1 → 61.7** |
| Code | 69.8 → 69.8 | 29.1 → 29.1 |
| Instructions | 80.0 → 79.9 | 34.0 → 34.9 |
| One function | 92.5 → 92.0 | 66.5 → 64.2 |
| Several functions | 91.5 → 89.0 | 56.0 → 64.0 |
| No function fits | 90.0 → 93.8 | 72.5 → 81.7 |
| All | 88.2 → 88.1 | 56.1 → 54.3 |
(Argon's math Pass@8 97.2 → 96.7 used a different record set; use these.)

## Point 4 — The saving is whole steps cut, not steps written shorter ✅
- `trim_condense.py` (MiniLM matching, 1,600 pairs: 200 math + 200 code + 100 × 4 others, samples 0–1) → `runs/trimcond/report.json`:
  net trimming 98–99%, condensing 1–2% in every task, at match thresholds 0.65 / 0.75 / 0.85. Steps per answer 54.9 → 21.5;
  tokens per matched step 17.0 → 16.4.
- Argon (consult30) on ALL 22,072 pairs, four matchers with no AI model: net trimming 98.8% (word overlap), 98.6% (4-grams),
  97.5% (best-of-8 pairing), 91.4% (many-to-one, an upper bound on merging 8.6%). Gross trimmed 112%, new unmatched steps 14%.
- Check for HIDDEN condensing (a shorter AND reworded step falls below the threshold and is booked as trimmed + added):
  `hidden_condense.py` → `runs/hidden_condense/`. 48% of the trained thinking is unmatched one-to-one; those steps are
  0.95× the length of their closest original step (not shorter rewrites); crediting every close one (similarity ≥ 0.55) with
  its closest unmatched original step gives an UPPER BOUND of 2.2% of the saving (math 1.5, code 2.3, instructions 2.5,
  tools 2.5–4.3); and they carry LESS content per token than unmatched original steps (numbers 6.2 vs 7.5). So condensing +
  merging ≤ ~8% of the saving; trimming ≥ ~90%.
- `content_density.py` (1,600 pairs) → `runs/density_content/`: merging (a trained step covering several DISTINCT original
  steps; repeats count as trimming) = 6.2% of the saving (math 8.2, code 6.1, instructions 1.8, tools 3.6–7.6).

## Point 5 — What was cut is mostly re-checking an answer already found ✅ (text) / ⏳ (causal)
Script: `settle_text.py` → `runs/settle/summary.txt` (all non-truncated answers; share of thinking written after the final
answer first appears; "part of the cut" = how much of the original→trained cut comes from that post-answer thinking).

| Task (settle point) | Right answers: share after | All answers: share after | Part of the cut |
|---|---|---|---|
| Math (final number, answer context) | 53.2% → 33.6% | 53.8% → 37.3% | 66–67% |
| One function (name + all argument values) | 51.2% → 32.7% | 49.1% → 30.7% | 64–67% |
| Several functions (name + all arguments) | 56.7% → 36.1% | 52.2% → 33.1% | 67–74% |
| Code (first draft of the final function starts) | 51.9% → 26.7% | 51.4% → 29.4% | 71–74% |
Code: the original drafts the final function inside its thinking in 99.6% of answers, the trained model in 68.7%.
Hesitation words ("wait", "hmm", "let me verify"…): 2.95 → 2.45 per 100 thinking words (article, section 8).
(Argon reported math 72% → 50% by another detection; these numbers are the ones to use: same definition for both models,
truncated answers excluded.)
⏳ CAUSAL version (effprobe on GPU 1, unseen slices 19–20, ~1,560 questions): cut each answer at 0–95% of its own thinking
(5% steps), force the answer, find the earliest cut from which it stays right.

## Point 6 — The steps it keeps are not denser ✅ (text) / ⏳ (surprisal)
`content_density.py`, matched steps (same step in both models), per 100 thinking tokens, original → trained:
numbers 6.7 → 6.8, operators 2.4 → 2.4, code 3.0 → 3.1, arguments 1.1 → 1.0, hesitation 1.4 → 1.6 (every task within noise).
Only change: math reaches the numbers sooner (numbers in the first 100 thinking tokens 4.7 → 7.1, operators 0.2 → 0.6).
Argon: compression of the first 100 thinking tokens only +2.3% denser; the full-trace gain (+20.7%) is the repetitive tail cut.
⏳ surprisal (effprobe): bits per thinking token of both models' traces read by both models; first 100 tokens; profile by position.

## Point 7 — The cut tokens were worth very little ✅ (comparison of two models) / ⏳ (causal)
From the article data (full test): accuracy with thinking, trained thinking tokens, thinking off.
| Task | Thinking orig → trained | Value of the kept part (trained − thinking off) | Cut tokens | Value of the cut (orig − trained) | Points per 100 cut tokens |
|---|---|---|---|---|---|
| Math | 677 → 262 | +65.3 | 415 | +2.84 | 0.68 |
| Code | 2079 → 823 | +9.7 | 1255 | +1.95 | 0.16 |
| Instructions | 433 → 141 | +1.8 | 292 | −0.49 | −0.17 |
| One function | 171 → 76 | +9.7 | 94 | +1.09 | 1.16 |
| Several functions | 192 → 84 | +20.8 | 109 | 0.00 | 0.00 |
| No function fits | 186 → 72 | −1.0 | 113 | −5.89 | −5.19 |
| **All** | 670 → 259 | **+28.2** | 411 | **+0.85** | **0.21** |
Thinking density (accuracy thinking adds per 100 thinking tokens): 4.3 → 10.9 overall.
⏳ CAUSAL version (effprobe): the ORIGINAL cut where the trained model stops — per task (mt) and per question (mq) — with paired
95% CIs: were the original's own extra tokens worth anything to itself?
⏳ Accuracy-vs-thinking curves per task (budgets at the 10th–90th percentiles of each task's thinking) → `eff_allocate.py`:
the best per-task budgets for the official score (Argon's estimate 59–64% saved at the same accuracy). For the next recipe,
not an argument about intelligence.

## Point 8 — The model itself improved a little (thinking off) ✅
Each trained model with thinking OFF against the original with thinking OFF (full test, official-style mean):
| Model | Change |
|---|---|
| od5c r16 (online, best) | **+2.1** (math 20.6 → 24.3, several 55.3 → 67.1, no-fit 90.3 → 92.6) |
| od5r r16 (online, second seed) | **+1.9** (math +5.1, several +6.6, no-fit +3.6) |
| rpM-60 (offline) | −0.3 |
| rpL-80 (offline) | −3.7 (tool calls −21, −22) |
Both online runs improved the model beyond its thinking; the offline runs did not.

## Generalisation (context) ✅
Four benchmarks never trained on (ARC-Challenge, CommonsenseQA, MMLU, HumanEval): 52–63% fewer tokens, accuracy within a
few points; only ARC-Challenge (−3.4) beyond noise. Not yet tested question by question (excess churn) on these.

---
## Pending (GPU 1, `effprobe`, after the od5f round-15 full test; results tonight)
1. Causal settle points (Point 5).
2. Matched cuts of the original (Point 7, causal).
3. Surprisal under both models (Point 6).
4. Accuracy-vs-thinking curves and the per-task optimum (Point 7, recipe).
