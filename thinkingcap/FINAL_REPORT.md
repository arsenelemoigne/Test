# ThinkingCap: final simple report (22 Oct 2026, morning)

**Goal:** make Qwen3-0.6B write far fewer tokens (thinking included) on the 4 test benchmarks (math SVAMP, code MBPP+,
tool calls BFCL, instructions IFEval) while losing at most 3 accuracy points against the untouched model.
**Competitor to beat:** Lightning McQwen3, **43.2% fewer tokens / −0.85 points** (his own full evaluation, 8 answers).

## 1. The answer in three lines
- **Recommended model: the "balanced" model** (`models/bal-r2-soup-20`): **~54–59% fewer tokens at about −2 to −2.5 points**,
  within the limit on every test we ran, including questions it never saw. That is **10–15 points more saving than the
  competitor**, for about 1.5 points more accuracy loss.
- **Safe fallback: round 2** (`adapters/bon-dpo-r2g/step-10`): ~51–54% at −1.6 to −2.7.
- **Cleanest model** (never saw any MBPP data, zero doubt on the rules): family B2 (`adapters/clean-b2/step-10`), 33% at −0.1,
  or family A2 (45% at +0.3, usual questions only).

## 2. How we test (and why some old numbers were too rosy)
- The full official test is 2,759 questions × 8 answers (many GPU hours). We use samples of 240 real test questions
  (40 per benchmark part), scored with the official grader against the untouched model on the same questions:
  - **"usual"**: the first 40 of each part (we also used these to choose checkpoints → slightly flattering);
  - **"fresh"**: 40 random others per part (seed 20261022), never used for any choice → **the honest test**;
  - the other session's fresh set (questions 41–80).
- Lesson of the night: numbers measured with 2 or 4 answers, or on the questions used to choose, were up to 3 points
  too optimistic. **Only 8-answer, fresh-question numbers are trusted below.**

## 3. The method that works (in plain words)
1. The model answers each training question **8 times**.
2. We build **pairs** from its own answers: "your shortest correct answer beats your longest correct one" and
   "a correct answer beats a wrong one".
3. **DPO** (a preference training) on those pairs, with a small add-on (LoRA, +1.7% parameters, merged away at the end),
   **one short pass** (10–30 steps). More passes do not help: the model memorises the pairs and forgets the lesson.
4. **Rounds**: the improved model answers *new* questions, new pairs, train again. Each round raises the bar.
- What did *not* work: cutting the model's thinking with rules (MVT, fading, word scoring: math breaks), plain imitation of
  the shortest answers (only 4–12% shorter), a "be brief" steering dial in the activations (≤18%, not combinable),
  training on code (it hurts code), extra code pairs to fix code.

## 4. All finalists, honest numbers (tokens saved / accuracy change in points)
| Model | Usual 240 × 8 | **Fresh 240 × 8** | Questions 41–80 × 8 | Verdict |
|---|---|---|---|---|
| Round 1 (no code pairs) | 42.4 / −0.7 *(×4)* | **40.4 / −0.9** | 43.6 / −0.9 | ✅ = competitor |
| Round 2 | 54.2 / −1.6 | **51.4 / −1.8** | 53.1 / −2.7 | ✅ safe |
| Round 2, retrained (other seed) | – | 50.9 / −3.0 | – | ⚠️ edge: training luck matters |
| **Balanced round (3 seeds averaged)** | **58.8 / −2.1** | **53.8 / −2.5** | – | ✅ **best** |
| Round 3 | 56.9 / −1.9 | 56.0 / −3.6 | – | ❌ |
| Round 4 | 65.4 / −2.4 | 63.2 / −3.9 | – | ❌ |
| Round 5 | 68.6 / −2.4 | 65.5 / −5.4 | – | ❌ |
| Family A1, MBPP-free, no code | 40.5 / −1.2 *(×4)* | 35.4 / −1.9 | – | ✅ clean |
| Family A2, MBPP-free, no code | 44.6 / +0.3 *(×4)* | – | – | ✅ clean (to confirm) |
| Family B1, MBPP-free + clean code data | – | 24.8 / −1.5 | – | ✅ clean, modest |
| Family B2, MBPP-free + clean code data | – | **32.7 / −0.1** | – | ✅ clean, code +1.9 |
| *Competitor (his full eval)* | | *43.2 / −0.85* | | |

## 5. What we learned about 60%+
- **Rounds 3–5 reach 56–69% but fail on unseen questions.** The loss is **math (SVAMP −7.5 to −10.6)**: math gets 76–80%
  shorter, and SVAMP is built with traps (irrelevant numbers, reworded questions) that very short thinking misses.
  Late rounds mostly push math, because tool and instruction answers are already near their shortest.
- **The balanced round** (savings from tools/instructions/multiple choice, math protected, SVAMP-style trap data
  GSM-Plus and GSM-IC, 3 training seeds averaged) keeps math at −4.7 on unseen questions and stays within the limit:
  the best trade-off we have. It does not reach 60% on unseen questions (53.8%).
- **Code**: with no code training it loses ~2 points on unseen questions (the −8/−10 seen on the usual questions was
  partly luck); clean code data (KodCode) keeps code at or above the untouched model but saves less overall.
- **Training luck** moves accuracy by 1–3 points between seeds: averaging seeds (as the balanced model does) is safer.

## 6. Rules check
- Never trained on the 4 test benchmarks; every training question checked against all 2,759 test questions (8-word overlap).
- **Grey zone:** rounds 1–5 and the balanced model descend from a first light step that saw ~74 MBPP problems
  (MBPP+ is built from MBPP; we removed every MBPP+ task and near copies). Families A and B never saw MBPP at all.
  If you want zero doubt, submit B2 or A2, or rebuild the balanced recipe from A1 (a few GPU hours).
- The model still thinks; no prompt or evaluation change; it loads in plain Transformers/vLLM (LoRA merges into the weights).

## 7. Suggested next steps
1. Run the **official full evaluation (2,759 × 8)** on the chosen model only (the final number).
2. If the MBPP grey zone matters: **balanced recipe on the MBPP-free line** (start from A1 or B2).
3. To push past 60% honestly: more balanced rounds with more fresh questions and seed averaging, choosing checkpoints
   only on fresh questions.

## 8. Where everything is
- **Google Drive:** `MyDrive/thinkingcap-mvtf/`. `adapters/` (every model), `models/` (merged, incl. `bal-r2-soup-20`), `runs/`
  (every score), `data/`, `code/` (scripts and logs).
- **GitHub:** branch `mvt-fading`, folder `thinkingcap/`: `EXPERIMENT_LOG.md` (every run), `FINDINGS_ALL.md`, this report.
