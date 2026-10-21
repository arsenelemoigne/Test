# ThinkingCap: every finding so far (branch mvt-fading, 20–21 Oct 2026)

## 0. Context (read first)
- **Challenge:** BottleCapAI ThinkingCap Test. Make Qwen3-0.6B (a 0.6B "thinking" model) write fewer tokens, thinking included, while keeping accuracy.
- **Rules:** start from Qwen3-0.6B; LoRA or changed weights allowed; must load in plain Transformers and vLLM (no custom code); must still think; no training on the evaluation benchmarks; evaluation settings fixed (prompt, sampling, token caps 4,096, 8,192 for MBPP+); answers cut by the cap count as wrong.
- **Benchmarks** (each 25% of the score):
  - SVAMP: math word problems, 1,000 questions
  - MBPP+: Python functions with strict extra tests, 378 questions
  - BFCL: function calling, 840 questions in 3 parts (one function offered; several offered; "irrelevance", where no function fits and the right answer is to call nothing)
  - IFEval: formatting instructions, 541 questions
- **Score:**
  - tokens saved = 1 − geometric mean of the per-benchmark token ratios (BFCL pooled)
  - accuracy change = mean of the per-benchmark changes, in points
  - our own hard limit: accuracy change ≥ −3 points vs the raw model
- **Our test sample:** 240 questions (40 per benchmark part, fixed), with 2, 4 or 8 answers per question.
  - Noise: about ±4 points on the average at 2 answers, ±2–3 at 4 answers; one benchmark cell at 2 answers is ±8–10 points.
  - The full evaluation (2,759 × 8) was dropped as too slow and expensive.
- **Notation:** "X% / Y" = X% fewer tokens / accuracy change of Y points vs the raw model on the same questions and answers per question.
- **Hardware:** Colab L4 (24 GB), 2 GPU sessions per account, ~3.1 units/h. LoRA rank 16 on all attention and MLP projections.
- **Training data** (never the test benchmarks):
  - GSM8K train+test (math, filtered to "easy": the shortest correct answer is at most 700 tokens, to match SVAMP)
  - MBPP minus every MBPP+ task (code)
  - ToolACE (tools, in BFCL's format)
  - argilla ifeval-like (instructions)
  - ARC-Easy and CommonsenseQA (multiple choice, not tested; a brevity signal)
  - Night 1 used older data: all of GSM8K, Hermes tool data, a 2,048-token cap.
- **Contamination check:** 8-gram overlap against every test question. 16 MBPP near-copies of MBPP+ were found and dropped (8 in the half-size rebuild).

## 1. Raw model (reference)
| benchmark | accuracy | mean tokens |
|---|---|---|
| SVAMP | 81–83% | 780–940 |
| MBPP+ | 57.5% | 1,700–2,000 |
| IFEval | 52–56% | 700–830 |
| BFCL irrelevance / several / one | 83–86 / 76–79 / 82–83% | 185–223 |

Training-question difficulty for the raw model:
- GSM8K: 43% correct, 24% hit the cap (much harder than SVAMP's 82%)
- MBPP train: 38%
- ToolACE: 58%
- ifeval-like: 78%
- CSQA: 51%
- ARC: 88%

## 2. Night 1: cutting the model's own correct answers (MVT, fading, word scoring) → SFT
**MVT** = Marginal Value Theorem (foraging theory):
1. Measure the model's confidence in its own correct answer after each paragraph of thinking (teacher forcing).
2. Stop at the first point where no later part pays more per token than the average so far, g(t)/(t+c).
3. c is the "travel cost": 1024 is patient, 256 is impatient.
4. A cut is kept only if the model still answers correctly from it (2 natural re-answers).

| run | method | answers/q | tokens saved | accuracy | within −3? | note |
|---|---|---|---|---|---|---|
| mvt256 | MVT, 2-step look-ahead (flawed) | 2 | +61.6% | −6.5 | no | SVAMP −21.3: cut where p(correct)=0 |
| mvth1024 | MVT, full look-ahead | 2 | +51.2% | −7.6 | no | SVAMP −23.7: cutting math breaks it |
| fade r1 (exp 2) | fading: drop 10% of paragraphs + 10% of words (LLMLingua-2) | 4 | +4.1% | −3.2 | edge | |
| fade r2 | 20% / 20% | 2 | +14.3% | −4.3 | no | |
| marginal r1 (exp 3) | words scored by our own model (leave-one-out) | 4 | +5.3% | −2.0 | yes | saves little |
| marginal r2 | | 4 | +1.8% | −6.2 | no | |
| 1b mvth1024-full-gsm8k | MVT on multiple choice only, math kept whole | 4 | +43.8% | −2.1 | yes | |
| 1b, c=256 | | 4 | +49.7% | −2.7 | edge | |
| **1b, math shown twice** | | **8** | **+37.1%** | **−0.6** | **yes** | **night-1 best**; SVAMP −5.0, MBPP+ −2.2 |
| 1b, c=256, math twice | | 4 | +42.0% | −2.1 | yes | |
| 1b, min 2 paragraphs | | 2 | +13.1% | −2.8 | yes | saves little |
| 1b, 35 rows/source | design mistake (also capped math) | 2 | +63.4% | −11.5 | no | |
| mixes of 1b + word scoring | | 2 | −7% to +35% | −1.5 to −5.6 | mostly no | |

**Lesson:** easy multiple-choice cuts teach "barely think", which spreads everywhere and breaks math unless whole math examples sit beside them.

## 3. Night 1 per-type grid (old data, one type trained at a time, 4 answers)
| type | best method | tokens saved / accuracy | its own benchmark | main spill-over |
|---|---|---|---|---|
| tools (Hermes) | MVT paragraphs | +42.0% / −0.7 | BFCL +37–54%, ~0 | MBPP+ +68% / −8.7 |
| code | MVT 32-token pieces | +24.1% / −1.9 | MBPP+ +57% / −5.6 | small |
| math | every cut breaks it | MVT −11.5; word scoring −19.0 | SVAMP −16 to −46 | |
| math | fading round 1 | +12.6% / −1.6 | SVAMP +24% / +5.0 | the only safe math cut |
| instructions | any | +5 to +13% / −0.3 to −1.5 | | |
| any type, uncut ("whole") | | ~0% / −2 to −3.5 | | old data alone cost points |

Night 1 combined models (4 answers):

| model | tokens saved / accuracy |
|---|---|
| all best cuts pooled | +49.0% / −6.2 (cuts add up) |
| same, math faded | +55.3% / −5.9 |
| **"light mix"** (MC cut + tool cut + whole math ×2) | **+48.6% / −2.3** |
| retrain of the night-1 winner | +31.1% / −2.2 (training randomness ~2 pts) |
| weight merging of per-type adapters | −13.7 to −29.9 (catastrophic) |

## 4. Other session (branch mvt-pertype): MVT per type on matched data, 25 runs (4 answers; their report)
| type | best setting | tokens saved / accuracy |
|---|---|---|
| tools | MVT paragraphs, c=256 | +25.8% / +0.6 (BFCL 36–55% shorter; 8 answers +29.3 / 0.0) |
| instructions | MVT paragraphs, c=1024 | +15.1% / −0.9 (IFEval +24% / +5.6) |
| code | MVT 32-token, c=1024 | +39.9% / −1.3 (MBPP+ +77% / −8.1; 8 answers −2.7) |
| math | keep whole; gentlest cut 32-token c=1024 | +22.2% / −1.6 (SVAMP +36% / −1.2) |
| multiple choice | none safe alone | −4 to −13 (breaks math) |
| any type, uncut | | −1.1 to +0.9: matched data makes training free |

Their combined models:

| model | 4 answers | 8 answers |
|---|---|---|
| light | +19.1% / −1.5 | |
| **codecut** | +48.3% / −1.2 | **+44.3% / −1.6** (MBPP+ −7.2, SVAMP −4.1) |
| full | +51.9% / −2.7 | |

**Their key findings:**
- Every cut, whatever the type trained, costs code (−2 to −14).
- Instructions improve with cuts.
- 8 answers settle 1–4 points lower than 4.
- Half of ToolACE's "no function fits" replies were really "missing argument" replies, so they added a filter. Our data predates it.

## 5. Competitor: Lightning McQwen3 (public; self-reported, 8 answers, full evaluation)
**+43.2% / −0.85** without his "think gate". The gate gives 47.3 / −0.88, but it forces empty thinking from the eval script, likely against the rules.

| benchmark | his result |
|---|---|
| SVAMP | 66% / −3.8 |
| MBPP+ | 39% / −1.9 |
| IFEval | 32% / +1.4 |
| BFCL | 25% / +0.9 |

His recipe:
1. Chain-of-Draft distillation (≤5 words per thinking step): 26.5% / −1.27.
2. On-policy cut oracle (9 cut points, 4/4 re-answers) → DPO, stopped at step 100 of 561: 42.5 / −1.45.
3. Math correctness DPO with an NLL anchor: 43.2 / −0.85.

He trained on **no code data**. Naive shortest-vs-longest DPO broke his model (90% / −27).

## 6. Other checks
- **RL session** (branch ccr-f7e14615): 33.5% / −0.5 claimed.
  - It's a 50/50 weight blend chosen on the test sample, 2 answers, interval [−4.2, +3.0], not rebuilt.
  - Real but fragile.
  - Its loop 1 alone: +5.7% / −1.2.
- **Early SFT rounds on pruned traces:** +39.1% / −6.3 and +59.6% / −8.0 (outside).

## 7. Tonight: best-of-8 ("shortest of 8"): the model answers 8×, keep its shortest correct answer
Data: 1,500 questions × 8 (round 1 full), 750 × 8 (rebuild). Kept answer vs the same question's average:

| source | full data | rebuild |
|---|---|---|
| math | 61% shorter | 59% |
| code | 40% | 48% |
| ARC | 37% | 34% |
| CSQA | 32% | 38% |
| instructions | 30% | 34% |
| tools | 30% | 32% |

| run | answers/q | tokens saved | accuracy | note |
|---|---|---|---|---|
| SFT on shortest, gentle (1 pass, lr 5e-5) | 4 | +4.4% | −2.8 | model barely learns length (loss 0.27: its own answers are already likely) |
| SFT, stronger (questions with a clear gap, 3 passes, lr 1e-4) | 4 | +12.5% | −0.7 | |

## 8. Tonight: best-of-8 → DPO (the big jump)
**Pairs**, built from the same answers:
- "shorten": shortest correct beats the longest correct, when the longest is ≥1.5× longer
- "guard": shortest correct beats the shortest wrong (finished) answer

**Settings:**
- mix 60% shorten / 40% guard
- β 0.1, lr 2e-5 constant, 16 pairs per step
- NLL anchor on the chosen answer (RPO)
- collapse tripwire (P(end right after `</think>`)), never fired
- checkpoints every 10–25 steps

**Night run** (full data, 1,267 pairs, start = gentle-SFT adapter; runtime lost afterwards):

| step | answers/q | tokens saved | accuracy | SVAMP | MBPP+ | IFEval | BFCL |
|---|---|---|---|---|---|---|---|
| 25 | 4 | **+47.6%** | −2.6 | +62% / +0.6 | +43% / −9.4 | +48% / −1.2 | +29–39% / −2.5..+0.6 |
| 50 | 4 | +40.3% | −2.8 | +47% / −2.5 | +42% / −5.0 | +40% / −3.7 | |
| 75 | 4 | +37.3% | −2.1 | | −10.0 | | |
| 100 | 4 | +36.5% | −2.6 | | −9.4 | −6.2 | |
| 125 | 4 | +31.1% | −3.2 | | | | |

**Rebuild** (half data, 591 pairs; one pass ≈ 33 steps). Original recipe (gentle SFT → DPO), pushed to 134 steps:

| step | answers/q | tokens saved | accuracy | SVAMP | MBPP+ | IFEval |
|---|---|---|---|---|---|---|
| 10 | 2 | +36.2% | −1.5 | +38% / −2.5 | +29% / −7.5 | +49% / +2.5 |
| 20 | 2 | +47.4% | −6.4 | +55% / −7.5 | +37% / −16.2 | +61% / −3.7 |
| 30 | 2 | +42.5% | −2.9 | | | |
| 30 | **4** | **+44.9%** | **−2.7** | +53% / −4.4 | +41% / −8.7 | +50% / +3.1 |
| 50 | 2 | +32.6% | −2.2 | | | |
| 70 | 2 | +37.6% | −1.0 | | | |
| 100 | 2 | +27.9% | −1.2 | | | |
| 130 | 2 | +22.4% | −0.1 | | | |

**Variants, same pairs:**

| variant | step | answers/q | tokens saved | accuracy | code (MBPP+) | math (SVAMP) |
|---|---|---|---|---|---|---|
| DPO from the raw model (no SFT) | 10 / 20 / 30 / 40 / 60 | 2 | 11.7 / 33.1 / 42.0 / 37.9 / 37.3 | −1.1 / −2.6 / −2.3 / −4.7 / −5.0 | | |
| DPO from raw, step 30 | 30 | 4 | +36.6% | **+0.8** | +35% / −5.0 | +39% / +0.6 |
| code fix: code gets guard pairs only (median-length correct) | 10 / 20 / 30 / 40 | 2 | 37.4 / 44.2 / 40.4 / 35.9 | −3.3 / −3.3 / −5.2 / −5.0 | −8.7 / −6.2 / −10.0 / −15.0 | |
| **no code pairs at all** | 10 / 20 / 30 / 40 | 2 | 35.6 / **50.6** / 44.1 / 35.3 | +1.1 / −4.4 / −2.8 / −1.0 | +1.3 / −8.7 / −6.2 / −1.2 | |
| **no code pairs, step 30** | 30 | **4** | **+42.4%** | **−0.7** | +38% / −3.7 | +53% / +1.2 |

No-code step 30, other benchmarks (4 answers): IFEval +44% / −1.2; BFCL +29–35% / 0.0 to +2.5.

**DPO findings:**
1. **The contrast is the lever.** The same data gives 4% as SFT and 45–48% as DPO.
2. **The best point is the end of the first pass over the pairs** (step 20–30). More passes don't break the model; they undo the brevity (47% → 22%) or cost accuracy. The pairs get memorized (100% ordered right), the DPO signal vanishes, and what's left (the NLL anchor ≈ plain SFT) pulls back. Progress needs **new pairs**, not more passes.
3. **The SFT step is optional:** it gives a head start; DPO from raw reaches the same point by the end of the pass.
4. **Code loses accuracy even without any code training.** Code gets 25–41% shorter by spill-over of the shared "think less" habit. Removing code pairs helps (−3.7 vs −8.7 at 4 answers) but doesn't fully fix it. The code-fix variant (guard only) didn't help.
5. **Math is our strength** (53–62% shorter, ~0 loss), unlike MVT cuts and unlike the competitor (−3.8).

## 9. Tonight: steering ("a brevity dial inside the model")
**Method:**
1. Read the residual stream over the first 64 thinking tokens of short vs long correct answers to the same question.
2. Build three directions:
   - A: mean difference
   - B: within-question trend vs log length, minus a "gives up" direction
   - C: one learned vector (DPO on the vector alone)
3. The push is stored as a v_proj bias (Qwen3 `attention_bias`), which loads in plain vLLM and Transformers. Verified: the push's effect per token matches in both engines (ratio 1.08, correlation 0.99).

**Reading the signal** (held-out "which of two correct answers will be shorter?" from the first 64 tokens; 0.50 = chance):
- **Shared direction:** 0.64–0.68 at layers 8–17 (a first, smaller reading said 0.70–0.73 at layer 9, partly luck).
- **By type:** instructions 0.73–0.79, tools 0.68–0.73, multiple choice 0.63–0.68, math 0.60–0.65, code 0.60–0.65.
- **Each type's own direction is not better than the shared one,** and the types' directions are nearly unrelated (cosines −0.18 to 0.39).

**Push tests** (strength in units of |A| at the layer, about 1% of the residual norm):

| push | answers/q | tokens saved | accuracy |
|---|---|---|---|
| A, layer 9, ×5 | 2 | +17.6% | −0.1 |
| A, layer 9, ×15 | 2 | +22.3% | −5.5 |
| A, layer 12, ×5 | 2 | +2.5% | −2.7 |
| A, layer 12, ×15 | 2 | +28.6% | −7.5 |
| B, layer 9, ×5 | 2 | +10.5% | −0.5 |
| B, layer 9, ×15 | 2 / 4 | +23.5% / +15.7% | −2.9 / −5.0 |
| C, layer 9, ×8 | 2 | +3.3% | +3.2 |
| C, layer 9, ×24 | 2 | +17.6% | −8.2 |

Fresh answers on 141 held-out training questions × 4:
- steering cuts the **average** answer 15–16% at equal accuracy (A, C)
- the **shortest correct of 4** barely moves (raw 450 tokens vs 407–459)

**Verdict:** a mild free saving (~15–18%) at most; not a better data source; not yet stacked on DPO.

## 10. Best results side by side
| model | evaluation | tokens saved | accuracy | SVAMP | MBPP+ | IFEval | BFCL |
|---|---|---|---|---|---|---|---|
| **ours: best-of-8 → DPO, no code pairs, step 30** | 240 q × 4 | **42.4%** | **−0.7** | 53% / +1.2 | 38% / −3.7 | 44% / −1.2 | 29–35% / 0..+2.5 |
| ours: best-of-8 → SFT → DPO, step 25 (night) | 240 q × 4 | 47.6% | −2.6 | 62% / +0.6 | 43% / −9.4 | 48% / −1.2 | 29–39% |
| ours: same recipe, rebuild, step 30 | 240 q × 4 | 44.9% | −2.7 | 53% / −4.4 | 41% / −8.7 | 50% / +3.1 | 30–37% |
| ours: DPO from raw, step 30 | 240 q × 4 | 36.6% | +0.8 | 39% / +0.6 | 35% / −5.0 | 43% / +6.9 | 26–30% |
| other session: MVT per type, "codecut" | 240 q × 8 | 44.3% | −1.6 | 15% / −4.1 | 73% / −7.2 | 20% / +5.0 | 42–51% / ~0 |
| night 1 best: MVT on MC + math whole ×2 | 240 q × 8 | 37.1% | −0.6 | 13.5% / −5.0 | 65% / −2.2 | 37% / +4.7 | 17–21% |
| competitor Lightning McQwen3 (no gate) | full × 8 | 43.2% | −0.85 | 66% / −3.8 | 39% / −1.9 | 32% / +1.4 | 25% / +0.9 |

**Caveats:**
- Ours are on 240 questions; the 4→8-answer recheck usually lowers the saving by 1–4 points.
- Best checkpoints are picked on the same questions they're reported on (mild selection bias).
- One training seed per run (retraining moved a result ~2 points).

## 11. Running now / next
- **Round 2 (on-policy):** the no-code step-30 model answers the 750 unused questions × 8, giving new pairs.
  - DPO for one pass, with code pairs (queued before no-code won)…
  - …and the same without code pairs (queued on the same answers)
  - Results ~20:30–21:00 Paris.
- **Then:**
  - 8-answer confirmation of the winner
  - more rounds while they improve
  - optional: stack the gentle steering dial
  - adopt the other session's ToolACE "no function fits" filter
  - a hybrid round with their verified tool MVT cuts as extra pairs
  - finally, RL with a length penalty from the best model
- **Open problem:** code. It loses accuracy through the shared "think less" habit, not its own data. A candidate fix: code-specific "stopped too early → wrong" pairs (on-policy cut oracle, as the competitor did).
- **Operations:** results, models and data go to Google Drive (MyDrive/thinkingcap-mvtf) after every job and every 20 min, and the code and log to GitHub, after the runtime was lost once (Colab recycled it while the worker sat idle).
