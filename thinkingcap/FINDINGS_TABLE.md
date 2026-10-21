# ThinkingCap: every finding so far, in one place (19–21 Oct 2026)

Self-contained briefing for discussing these results with another AI or person.
Source logs: `thinkingcap/EXPERIMENT_LOG.md` (all tracks), `REPORT_MVT_FADING.md` (night 1),
`REPORT_MVT_PERTYPE.txt` (track 3, very detailed). Repo github.com/arsenelemoigne/Test, branches
`mvt-fading` (night 1), `mvt-pertype` (track 3 and 3b), `ccr-f7e14615-o4qu8r` (RL track).

---

## 1. Context

**Challenge.** BottleCapAI ThinkingCap Test (github.com/BottleCapAI/ThinkingCap-Test): make
**Qwen3-0.6B** (a small "thinking" model that writes a `<think>…</think>` block before answering)
use **fewer output tokens, thinking included, while keeping accuracy**.

**Rules that matter.** Start from Qwen3-0.6B; LoRA or changed weights allowed; must load in plain
Transformers and vLLM (no custom code); the model must still think before answering (thinking may
not be turned off); no training on the evaluation benchmarks; evaluation prompt, sampling
(temperature 0.6, top-p 0.95, top-k 20) and token caps (4,096; 8,192 for code) unchanged; capped
answers count as wrong; organisers value original ideas.

**Benchmarks (25% each).** SVAMP (math word problems, 1,000 q), MBPP+ (Python functions, 378 q),
BFCL function calling (840 q: *simple* = one function offered, *multiple* = several offered,
*irrelevance* = no offered function fits, the right answer is to call nothing), IFEval (format
instructions, 541 q).

**Score.** Tokens saved = 1 − geometric mean of per-benchmark token ratios (BFCL's three parts
pooled). Accuracy change = mean of the four per-benchmark changes (points). Our own limit:
**accuracy change ≥ −3 points**.

**How we measure.** A fixed 240-question subset (first 40 of each of the 6 benchmark parts, in the
evaluation's order), 2, 4 or 8 sampled answers per question, against the untouched model on the
same questions and seeds. Noise: about ±4 points per benchmark cell at 4 answers; overall
accuracy interval about ±2.5 points at 8 answers. The full evaluation (2,759 × 8) was never run.

**Hardware.** One Colab L4 (24 GB) per track; ~3.1 Colab units per hour.

**Untouched model on the subset (4 answers).** BFCL irrelevance 85.6% / 213 tokens, multiple 79.4% /
217, simple 83.1% / 185; IFEval 51.9% / 705; MBPP+ 57.5% / 2,004; SVAMP 81.9% / 830.
(2 answers, night 1: SVAMP 81.3% / 778, MBPP+ 57.5% / 1,699, IFEval 52.5% / 710.)

---

## 2. Vocabulary used in the tables

| term | meaning |
|---|---|
| trace / reply | the model's full answer: thinking + final answer |
| SFT | supervised fine-tuning: train the model to reproduce chosen traces (LoRA rank 16, lr 1e-4, 2 epochs unless stated) |
| LoRA / adapter | small trainable add-on (~10 M parameters, 1.7% of the model); base weights frozen |
| MVT cut | Marginal Value Theorem rule: measure p(model's own correct answer) after each piece of thinking (teacher forcing); cut at the first point where no later piece pays better per token than the average so far, g(t)/(t+c) |
| c (travel) | patience of the MVT rule: c=1024 "patient" (cuts less), c=256 "impatient" (cuts more) |
| paragraphs / tokens32 | the pieces the cut works on: paragraph breaks, or fixed 32-token batches (can cut mid-sentence) |
| check | a cut trace is kept only if the model, answering naturally from it, is right k times (k=2 in tracks 2–3, k=4 in track 3b); otherwise the original is kept |
| whole / uncut | trained on correct traces without cutting (control: what training alone does) |
| spill-over | effect of training on one task type on the other benchmarks |
| x2 | that source's training rows shown twice |
| fading | cut a growing share (10–20% per round) of the least useful paragraphs (and words), round after round |
| marginal (exp 3) | delete words/paragraphs whose removal barely lowers p(correct) |
| DPO | preference training on pairs (chosen > rejected) relative to a reference model |
| on-policy | the traces come from the *current* (already trained) model, not the untouched one |
| saved / acc | tokens saved % / accuracy change in points, vs the untouched model on the same questions |

Benchmark column order in the tables: **SVAMP · MBPP+ · BFCL irr / mult / simple · IFEval**, each
"saved / acc". "—" = not recorded.

---

## 3. Master table: every scored run

### 3.1 Track 1 (early session, branch ccr-…; 2 answers; old data GSM8K / ARC-Easy / CSQA, 2,048 cap)

| # | run | method | ans | saved | acc | per benchmark | verdict |
|---|---|---|---|---|---|---|---|
| 1 | SFT round 1 (T4) | SFT on pruned own traces (settle rule) | 2 | +39.1% | −6.3 | — | fails limit |
| 2 | SFT round 2 | same, margin + balanced sources | 2 | +59.6% | −8.0 | SVAMP +48.8%/−23.8 · MBPP+ +86.4%/−6.3 · BFCL +32–39%/−3.8 to −10.0 · IFEval +40.3%/+3.8 | math collapses |
| 3 | RL loop 1 | RL, TokenProbe-KL + ShorterBetter length reward | 2 | +5.7% | −1.2 | SVAMP +12.6%/−1.3 · MBPP+ −18.1%/−3.8 · BFCL irr +4.3%/−5.0, mult +12.2%/+8.8, simple +9.0%/+1.3 · IFEval +16.1%/−1.3 | safe, saves little |
| 4 | RL loop 3 | RL continued, stronger pressure | 2 | — | — | SVAMP +35.8%/−10.0 · MBPP+ +7.7%/−8.8 · BFCL +0.6/+18.6/+19.9% / −8.8/−3.8/−3.8 · IFEval +33.8%/0.0 | math hurt |
| 5 | RL session (other branch) best | GRPO "short and correct" + exact LoRA blend | 2 | +33.5% | −0.5 | — (about 0.5–1 pt optimistic at 2 answers) | reported by the RL session |

### 3.2 Track 2 (night 1, branch mvt-fading; MVT, fading, marginal; old data, 2,048 cap)

| # | run | method | ans | saved | acc | verdict / notes |
|---|---|---|---|---|---|---|
| 6 | mvt256 | MVT cut all sources, 2-step look-ahead (flawed) | 2 | +61.6% | −6.5 | math −21 to −24 |
| 7 | mvth1024 | MVT full look-ahead, c=1024, all sources | 2 | +51.2% | −7.6 | math collapses |
| 8 | fade-gain-words r1 | fading 10% paragraphs + 10% words (LLMLingua-2) | 4 | +4.1% | −3.2 | edge |
| 9 | fade-gain-words r2 | round 2 | 2 | +14.3% | −4.3 | fails |
| 10 | fade-gain-marginal r1 | words scored by own model (exp 3) | 4 | +5.3% | −2.0 | safe, saves little |
| 11 | fade-gain-marginal r2 | round 2 | 4 | +1.8% | −6.2 | breaks (telegraphic text) |
| 12 | mvth1024-full-gsm8k | **MVT on ARC/CSQA, GSM8K kept whole** | 4 | +43.8% | −2.1 | within |
| 13 | mvth256-full-gsm8k | same, c=256 | 4 | +49.7% | −2.7 | edge |
| 14 | **mvth1024-full-gsm8k-xgsm8k2** | **same + GSM8K shown twice** | **8** | **+37.1% [32.1, 42.2]** | **−0.6 [−3.05, +1.74]** | **night-1 winner**; SVAMP +13.5%/−5.0, MBPP+ +65%/−2.2, BFCL +17–21%/−1.3 to +1.6, IFEval +37%/+4.7 |
| 15 | mvth256-full-gsm8k-xgsm8k2 | c=256, GSM8K x2 | 4 | +42.0% | −2.1 | within |
| 16 | …-min2 | keep ≥ 2 paragraphs | 2 | +13.1% | −2.8 | saves little |
| 17 | …-cap35 | 35 rows per source (design mistake) | 2 | +63.4% | −11.5 | fails |
| 18 | mix-mvt1024-marginal1-x2 | 1b + GSM8K compacted by exp 3 | 2 | +34.6% | −3.1 | fails (math) |
| 19 | mix-marginal1-mathwhole-x2 | exp 3 words on ARC/CSQA, math whole | 2 | −2.6% | −1.5 | saves nothing |
| 20 | mix-marginal2-mathwhole-x2 | exp 3 round 2 | 2 | −7.2% | −5.6 | fails |

### 3.3 Track 2b (night 1, per-type grid on OLD data: GSM8K all, Hermes tools, MBPP minus MBPP+, IFEval-like; 4 answers; 2,048 cap)

| # | run | trained on | cut | saved | acc | SVAMP | MBPP+ | BFCL (avg) | IFEval |
|---|---|---|---|---|---|---|---|---|---|
| 21 | math-whole | GSM8K | none | −1% | −2.2 | +1/+3.7 | −2/−5.0 | +2/−1.3 | −5/−6.2 |
| 22 | math-mvt1024 | GSM8K | MVT par. | +49% | −11.5 | +80/−16.2 | +43/−17.5 | +42/−5.2 | +3/−6.9 |
| 23 | math-mvt1024-tokens32 | GSM8K | MVT t32 | +37% | −13.3 | +55/−14.4 | +52/−28.1 | +25/−3.8 | +1/−6.9 |
| 24 | math-marginal | GSM8K | leave-one-out | +57% | −19.0 | +97/−46.2 | +30/−17.5 | +15/−0.2 | −67/−11.9 |
| 25 | math-fade-r1 / r2 / r3 | GSM8K | fading 20/40/60% | +13 / +16 / +23% | −1.6 / −4.2 / −4.7 | +24/+5.0 · +35/+1.9 · +50/−5.6 | −5.6 · −9.4 · −10.6 | | −4.4 · −7.5 · −0.6 |
| 26 | code-whole | MBPP | none | −4% | −3.5 | −2/0.0 | −9/−6.2 | +4/+0.2 | −10/−8.1 |
| 27 | code-mvt1024 | MBPP | MVT par. | +58% | −3.4 | +9/+2.5 | +93/−13.7 | +29/+0.2 | +27/−2.5 |
| 28 | code-mvt1024-tokens32 | MBPP | MVT t32 | +24% | −1.9 | +8/−0.6 | +57/−5.6 | +14/−1.3 | +2/0.0 |
| 29 | code-marginal | MBPP | leave-one-out | +73% | −18.8 | +90/−41.2 | +90/−25.6 | +34/−3.1 | +17/−5.0 |
| 30 | code-fade-r1 / r2 / r3 | MBPP | fading | +11 / +24 / +21% | −2.8 / −2.1 / −5.9 | | +20/−10.6 · +33/−3.1 · +14/−13.1 | | |
| 31 | tool-whole | Hermes | none | +2% | −3.3 | +11/+3.7 | −1/−9.4 | +7/0.0 | −12/−7.5 |
| 32 | **tool-mvt1024** | Hermes | MVT par. | **+42%** | **−0.7** | +15/+3.7 | **+68/−8.7** | +43/+2.1 | +26/0.0 |
| 33 | tool-mvt1024-tokens32 | Hermes | MVT t32 | +4% | −1.6 | −11/−1.3 | +12/−7.5 | +28/−1.5 | −20/+3.7 |
| 34 | tool-marginal | Hermes | leave-one-out | +24% | −3.0 | +11/+4.4 | +34/−10.0 | +33/−1.3 | +16/−5.0 |
| 35 | tool-fade-r1 / r2 / r3 | Hermes | fading | −7 / +10 / +43% | −2.7 / −2.4 / −3.6 | | −7.5 · −8.7 · +57/−10.0 | | |
| 36 | text-whole | IFEval-like | none | +3% | −0.5 | +1/+7.5 | −5/−6.9 | +7/0.0 | +9/−2.5 |
| 37 | text-mvt1024 | IFEval-like | MVT par. | +13% | −1.5 | +20/+3.1 | +13/−5.0 | +9/−1.0 | +11/−3.1 |
| 38 | text-mvt1024-tokens32 | IFEval-like | MVT t32 | +5% | +0.6 | +11/+5.6 | −2/−3.7 | +6/+0.4 | +5/0.0 |
| 39 | text-marginal | IFEval-like | leave-one-out | +7% | −1.3 | +8/+1.2 | +2/−5.6 | +12/−0.8 | +6/0.0 |
| 40 | text-fade-r1 / r2 / r3 | IFEval-like | fading | +10 / +6 / +12% | −0.3 / −2.2 / −0.8 | | | | +3/+0.6 · +11/−4.4 · +20/−3.1 |

Night-1 combined models (4 answers):

| # | model | rows | saved | acc | notes |
|---|---|---|---|---|---|
| 41 | combine-final: MC MVT + math whole x2 + code MVT t32 + tool MVT + text MVT | 860 | +49.0% | −6.2 | cuts add up; MBPP+ +81%/−11.9, SVAMP −5.0, IFEval −6.2 |
| 42 | combine-final-mathfade: same, GSM8K faded r2 | 860 | +55.3% | −5.9 | SVAMP −10.0 |
| 43 | 1b winner retrained (#14 recipe) | 270 | +31.1% | −2.2 | retraining moves a result 2–8 points |
| 44 | combine-final-light: MC MVT + tool MVT + math whole x2 | 502 | +48.6% | −2.3 | MBPP+ +77%/−12.5 by spill-over |

### 3.4 Track 3 (this work, branch mvt-pertype): per-type grid on MATCHED data, 4 answers

Matched data: math = GSM8K train+test filtered to questions with a correct reply ≤ 700 tokens
(SVAMP-like; 60 questions survived); code = MBPP minus every MBPP+ task (137 with a correct reply);
tool = ToolACE in BFCL's format and mix, with genuine "no function fits" only (189); text =
argilla IFEval-like (259); choice = ARC-Easy + CommonsenseQA (203). Caps 4,096 / 8,192 code.
"cut" = traces cut / kept, training-target tokens before → after.

| # | run | cut (n, tokens) | saved | acc | SVAMP | MBPP+ | BFCL irr / mult / simple | IFEval | verdict |
|---|---|---|---|---|---|---|---|---|---|
| 45 | tool-whole | 0/189, 201→183 | +5.2% | +0.3 | +12/+1.9 | +6/+0.6 | +6/0.0 · +5/+0.6 · +6/−3.1 | −4/−0.6 | training is free (old data −3.3) |
| 46 | tool-mvt1024 | 178/189, 201→142 | +17.6% | +1.7 | +8/+1.3 | +2/−3.1 | +38/−1.2 · +55/−3.1 · +42/+1.9 | +6/+9.4 | free; little spill-over |
| 47 | **tool-mvt256** | 173/189, 201→121 | **+25.8%** | **+0.6** | +10/−0.6 | +30/−4.4 | +45/+3.1 · +55/−1.9 · +36/+0.6 | +11/+6.9 | **tool winner** |
| 48 | tool-mvt1024-tokens32 | 176/189, 201→106 | +18.1% | −2.7 | +24/−1.9 | 0/−6.9 | +54/−8.7 · +24/0.0 · +39/−1.3 | +3/+1.2 | breaks irrelevance |
| 49 | tool-mvt256-tokens32 | 167/189, 201→89 | +11.5% | −4.5 | +7/−10.6 | −5/−6.9 | +54/−10.0 · +24/−1.9 · +47/−0.6 | −6/+3.7 | fails |
| 50 | text-whole | 0/259, 347→312 | +6.1% | −0.5 | +14/−2.5 | −2/−1.9 | +5/−1.2 · +6/+0.6 · +5/0.0 | +6/+2.5 | free |
| 51 | **text-mvt1024** | 238/259, 347→234 | **+15.1%** | **−0.9** | +17/−1.9 | +8/−6.2 | +11/−5.6 · +9/+3.1 · +11/−1.3 | **+24/+5.6** | **text winner** |
| 52 | text-mvt256 | 227/259, 347→190 | +18.2% | −2.1 | +7/−4.4 | +29/−7.5 | +23/−3.7 · +12/0.0 · +13/−1.3 | +19/+5.0 | more spill-over |
| 53 | text-mvt1024-tokens32 | 234/259, 347→218 | +17.1% | −1.7 | +26/−5.0 | +5/−8.1 | +10/−0.6 · +13/+3.1 · +10/−2.5 | +25/+6.2 | |
| 54 | text-mvt256-tokens32 | 219/259, 347→180 | +11.8% | −2.1 | +20/−3.7 | −1/−6.2 | +3/−6.2 · +4/0.0 · +4/−1.9 | +23/+4.4 | |
| 55 | choice-whole | 0/203, ~455→~400 | +2.5% | +0.9 | +17/+3.7 | −8/−5.0 | +5/0.0 · +9/−0.6 · 0/+1.9 | −6/+4.4 | free |
| 56 | choice-mvt1024 | 201/203, ~455→~97 | +72.9% | −10.7 | +66/−39.4 | +93/−10.0 | +39/−1.2 · +53/−13.1 · +32/−3.8 | +60/+12.5 | "barely think" everywhere; math collapses |
| 57 | choice-mvt256 | 194/203, ~455→~102 | +69.0% | −13.3 | +61/−38.7 | +93/−13.7 | +37/−3.7 · +51/−11.2 · +28/−0.6 | +45/+4.4 | fails |
| 58 | choice-mvt1024-tokens32 | 199/203, ~455→~76 | +8.8% | −4.7 | +8/−13.1 | +9/−1.9 | +18/−1.2 · +12/−10.6 · +15/−5.6 | +3/+1.9 | fails |
| 59 | choice-mvt256-tokens32 | ~190/203 | +18.7% | −4.1 | +11/−9.4 | +19/−3.1 | +32/+2.5 · +15/−9.4 · +21/−10.0 | +22/+1.9 | fails |
| 60 | code-whole | 0/137, 1,493→1,308 | +2.9% | +0.1 | +8/−3.1 | +3/−1.9 | −3/−1.2 · −1/−0.6 · +1/−1.3 | +2/+6.2 | free (old −3.5) |
| 61 | code-mvt1024 | 92/137, 1,493→790 | +69.3% | −6.0 | +38/−16.9 | +95/−13.7 | +42/+2.5 · +55/−2.5 · +33/−1.3 | +52/+6.9 | all-or-nothing cut kills the plan |
| 62 | code-mvt256 | 82/137, 1,493→812 | +68.2% | −7.2 | +38/−21.3 | +95/−10.6 | +50/−1.2 · +63/−11.2 · +42/−1.3 | +36/+7.5 | fails |
| 63 | **code-mvt1024-tokens32** | 90/137, 1,493→824 | **+39.9%** | **−1.3** | +17/−1.2 | +77/−8.1 | +25/+0.6 · +19/+2.5 · +17/−1.9 | +12/+3.7 | **code winner** (code itself −8) |
| 64 | code-mvt256-tokens32 | 78/137, 1,493→903 | +46.6% | −0.5 | +10/−3.7 | +86/−10.6 | +43/+3.8 · +10/−3.7 · +23/−2.5 | +12/+13.1 | best average, but code −10.6 |
| 65 | **math-whole** | 0/60, 676→509 | −0.3% | −1.1 | +3/−3.7 | −7/−2.5 | −6/−3.7 · +6/−0.6 · −4/−1.3 | +4/+3.7 | **math winner (keep whole)** |
| 66 | math-mvt1024 | 57/60, 676→131 | +26.3% | −2.6 | +45/−5.6 | +14/−12.5 | +25/−1.9 · +22/+1.3 · +15/−1.9 | +22/+8.7 | far milder than night 1 (−16) |
| 67 | math-mvt256 | 57/60, 676→123 | +32.6% | −3.6 | +53/−6.9 | +20/−13.7 | +25/−2.5 · +20/−0.6 · +17/−3.1 | +30/+8.1 | fails |
| 68 | math-mvt1024-tokens32 | ~55/60, 676→143 | +22.2% | −1.6 | +36/−1.2 | +14/−6.9 | +9/−1.2 · +13/+0.6 · +10/−3.1 | +25/+3.1 | gentlest math cut |
| 69 | math-mvt256-tokens32 | 50/60, 676→132 | +13.8% | −4.0 | +29/−6.9 | +7/−9.4 | +14/−1.9 · +10/0.0 · +10/−5.0 | +5/+2.5 | fails |
| 70 | tool-mvt256, irrelevance kept whole | 96/189, 201→159 | +12.4% | −1.4 | +10/−1.2 | −1/−9.4 | +25/−4.4 · +30/+1.9 · +39/+2.5 | +5/+5.0 | did not protect irrelevance; dropped |

8-answer rechecks (track 3):

| # | run | 4 answers | 8 answers | 8-answer per benchmark |
|---|---|---|---|---|
| 71 | tool-mvt256 | +25.8% / +0.6 | +29.3% / +0.0 | SVAMP +5/+2.2 · MBPP+ +42/−4.1 · BFCL +43/+3.4 · +54/−4.4 · +33/+0.9 · IFEval +19/+1.9 |
| 72 | tool-mvt1024 | +17.6% / +1.7 | +20.5% / −0.4 | SVAMP +9/+0.3 · MBPP+ +16/−0.3 · BFCL +42/−3.4 · +49/−5.0 · +38/+1.2 · IFEval +8/+0.9 |
| 73 | text-mvt1024 | +15.1% / −0.9 | +14.0% / −0.8 | SVAMP +28/0.0 · MBPP+ +3/−3.7 · BFCL +16/−1.6 · +8/−2.5 · +11/−1.3 · IFEval +12/+2.2 |
| 74 | code-mvt1024-tokens32 | +39.9% / −1.3 | +37.5% / −2.7 | SVAMP +14/−4.1 · MBPP+ +78/−7.5 · BFCL +25/−0.9 · +15/−0.9 · +17/−3.4 · IFEval +2/+2.5 |

Track 3 combined models (one adapter on pooled winner rows, ≤150 rows per source, math whole x2):

| # | model | rows | ans | saved | acc | SVAMP | MBPP+ | BFCL irr / mult / simple | IFEval |
|---|---|---|---|---|---|---|---|---|---|
| 75 | light: tool-mvt256 + text-mvt1024 + code WHOLE + math whole x2 | ~550 | 4 | +19.1% | −1.5 | +13/−3.1 | −6/−6.9 | +37/−1.2 · +52/+1.3 · +41/+1.2 | +18/+3.7 |
| 76 | **codecut: same, code mvt1024-tokens32** | 549 | 4 | **+48.3%** | **−1.2** | +20/−5.0 | +75/−9.4 | +50/−3.7 · +53/+0.6 · +47/+1.2 | +28/+10.0 |
| 77 | full: codecut + choice-mvt1024 | 752 | 4 | +51.9% | −2.7 | +16/−3.7 | +81/−11.2 | +43/−1.9 · +56/0.0 · +47/−0.6 | +34/+5.0 |
| 78 | **codecut, 8 answers (CURRENT BEST, safe)** | 549 | 8 | **+44.3% [39.4, 49.6]** | **−1.6 [−3.9, +0.9]** | +15/−4.1 | +73/−7.2 | +51/+0.6 · +48/−1.6 · +42/+0.6 | +20/+5.0 |

### 3.5 Track 3b: on-policy rounds from codecut (MVT cuts of the current model verified 4/4, DPO)

Round 1 data (fresh questions, seed 1, none used before): toolace 267, ifeval_like 300, mbpp_train 295,
gsm8k_all 300. The current model (codecut) answered 4×: tool 60% correct / 128 tokens; text 72% / 331;
code 29% / 744; math 62% / 1,812. Cuts: tool c=256 paragraphs, text c=1024 paragraphs, code and math
c=1024 tokens32; plus "early" cuts at half the MVT point; every cut checked 4/4.

| kind | cuts checked / passed | targets (MVT / early / original) | tokens before → after | pairs (shorten / guard / hard) |
|---|---|---|---|---|
| tool | 1,284 / 1,107 | 180 (39 / 134 / 7) | 117 → 49 (−58%) | 183 (139 / 37 / 7) |
| text | 1,742 / 1,309 | 252 (38 / 181 / 33) | 251 → 125 (−50%) | 276 (207 / 36 / 33) |
| code | 684 / 461 | 119 (27 / 57 / 35) | 515 → 326 (−37%) | 131 (76 / 20 / 35) |
| math (≤700-token correct reply) | 876 / 726 | 119 (109 / 2 / 0) | 790 → 93 (−88%) | 135 (119 / 16 / 0) |

Pairs: *shorten* = safe cut > full trace; *guard* = safe cut > earlier cut that failed; *hard* = full
trace > failed cut (no safe cut exists). DPO: β 0.1, lr 2e-5, 16 pairs per step, reference = start.

| # | run | pairs | ans | saved | acc | SVAMP | MBPP+ | BFCL irr / mult / simple | IFEval | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| 79 | r1 step 20 | 722 | 4 | +44.1% | −2.5 | +31/−3.1 | +68/−9.4 | +43/−6.2 · +36/−3.1 · +40/−1.9 | +28/+6.2 | below the start |
| 80 | r1 step 40 | 722 | 4 | +50.2% | −3.5 | +44/−2.5 | +67/−8.7 | +54/**−23.1** · +32/−1.2 · +41/−3.1 | +42/+6.2 | irrelevance collapses |
| 81 | r1 final (46) | 722 | 4 | +51.2% | −6.1 | +51/−5.0 | +65/−11.2 | +49/**−39.4** · +32/−2.5 · +37/−3.1 | +46/+6.9 | fails |
| 82 | r1b step 10 (no shorten pairs on irrelevance, NLL anchor 1) | 670 | 4 | +44.9% | −3.2 | +23/−5.0 | +65/−7.5 | +54/−1.9 · +53/−1.9 · +47/+1.2 | +29/+0.6 | |
| 83 | **r1b step 20** | 670 | 4 | +42.8% | −0.3 | +29/+2.5 | +60/−11.2 | +53/−5.0 · +42/−1.2 · +48/−2.5 | +29/+10.6 | looked best (luck) |
| 84 | r1b step 30 | 670 | 4 | +64.4% | −11.4 | +83/**−35.0** | +66/−16.2 | +49/−11.2 · +47/−2.5 · +52/−1.3 | +44/+10.6 | collapse |
| 85 | r1b step 40 | 670 | 4 | +88.8% | −19.3 | +98/−50.0 | +94/−19.4 | +76/−25.0 · +77/+4.4 · +76/+2.5 | +57/−1.9 | stops thinking |
| 86 | r1b final (42) | 670 | 4 | +92.0% | −19.5 | +98/−51.9 | +97/−20.0 | +78/−29.4 · +81/+3.1 · +78/+1.9 | +68/+1.9 | stops thinking |
| 87 | **r1b step 20, 8 answers** | 670 | 8 | **+47.9% [43.6, 53.1]** | **−2.3 [−5.0, +0.5]** | +36/−3.7 | +72/−4.1 | +59/**−10.6** · +41/−1.9 · +44/+0.3 | +22/+2.8 | a trade vs #78, not a gain |

---

## 4. Headline comparison (8 answers where available)

| model | saved | acc | SVAMP | MBPP+ | BFCL | IFEval | measured on |
|---|---|---|---|---|---|---|---|
| Night-1 winner (#14): MC MVT + math whole x2 | 37.1% | −0.6 | 13.5% / −5.0 | 65% / −2.2 | 17–21% / −1.3 to +1.6 | 37% / +4.7 | 240 q × 8 |
| **Track 3 best (#78): combine-codecut** | **44.3%** | **−1.6** | 15% / −4.1 | 73% / −7.2 | 42–51% / −1.6 to +0.6 | 20% / +5.0 | 240 q × 8 |
| Track 3b (#87): r1b step 20 | 47.9% | −2.3 | 36% / −3.7 | 72% / −4.1 | 41–59% / −10.6 to +0.3 | 22% / +2.8 | 240 q × 8 |
| RL session best | 33.5% | −0.5 | — | — | — | — | 240 q × 2 (optimistic) |
| **Lightning McQwen3 (public rival), clean** | **43.2%** | **−0.85** | 66% / −3.8 | 39% / −1.9 | 25% / +0.9 | 32% / +1.4 | 8 answers, question count not stated, self-reported |
| Lightning McQwen3 with "think gate" | 47.3% | −0.88 | | | 45% / +0.8 | | gate likely against the rules |

Rival recipe (Ondřej Baštař, CTU Prague, blog.basta.one/blog/thinkingcap, 14 Oct 2026):
(1) Chain-of-Draft distillation — traces generated under "≤5 words per thinking step", prompt removed,
SFT on 11,247 → 26.5% / −1.27; (2+3) on-policy truncation oracle on the distilled model — cuts at
10…90% of the thinking, 4 re-answers each (197k cuts), DPO pairs 40% shorten / 40% guard / 20% hard,
β 0.1, lr 2e-5, stopped at step 100 of 561 → 42.5% / −1.45; (4) math correctness DPO with an NLL anchor
→ 43.2% / −0.85. Same oracle on the untouched model: 11.4% (SFT) / 28.1% (DPO). His gate forces an
empty think block on ~30% of BFCL prompts from the eval script (probably against the rules).

---

## 5. Findings (what the table shows)

1. **Matched practice data makes training itself free.** All five track-3 uncut runs: −1.1 to +0.9;
   night 1's data cost 2–3.5 points before any cut (#21, #26, #31, #36 vs #45, #50, #55, #60, #65).
2. **Each task type needs its own cut unit.** Tool and instructions: paragraphs. Code: 32-token pieces
   (paragraph cuts remove the whole plan: #61–62). Math: keep whole, or at most the gentlest cut.
3. **Finer cuts are not better cuts.** 32-token pieces cut deeper in training but shorten less at test
   time and break things (tool irrelevance #48–49, multiple choice #58–59): mid-sentence thinking is
   not imitated cleanly, and late checks get cut.
4. **Patient (c=1024) vs impatient (c=256):** same effect on the trained task; impatient spreads
   brevity and damage further.
5. **Code (MBPP+) is hurt by every cut, whatever is trained** (−2 to −14 across ~20 cut runs).
6. **Instructions (IFEval) improve with almost every cut** (+2 to +12; partly noise).
7. **Easy examples (multiple choice) teach "barely think" everywhere** and break math (#56–57; night 1 #6–7);
   they are only usable beside whole math (#14).
8. **Cuts add up when combined** (#41, #77): each safe alone, together they cost more.
9. **The strength of the lesson drives spill-over.** Night-1 tool data (Hermes, easier, longer replies,
   synthetic "no fit" pairs) gave code 68% shorter at −8.7 (#32); matched ToolACE gives 2% shorter
   (#46). Same code, same test; the data is the only difference (verified by diff).
10. **On-policy DPO (track 3b) shortens a lot but collapses fast.** Within ~20 steps the model drops
   the "does any function fit?" check (BFCL irrelevance −10 to −39) and, later, stops thinking
   altogether (math −35 to −52 at 64–92% saved). An NLL anchor on the chosen side does not prevent it
   (the chosen side is itself short). Early checkpoints help math (36% shorter at −3.7) and code
   (−4.1 instead of −7.2) but hurt irrelevance.
11. **4-answer results are optimistic for selected runs.** Best runs lose 1–4 points of saving and
   0.4–2 points of accuracy at 8 answers (#76→#78, #83→#87).

---

## 6. Open questions worth discussing

- How to keep the BFCL irrelevance check under on-policy DPO: add irrelevance questions as
  "full > cut" pairs; weight guard/hard pairs up; lower β or learning rate; stop at step 10–20.
- How to stop code losing 4–8 points: fewer/softer code cuts, whole code rows beside cut ones,
  or code-specific "too early" pairs.
- Whether math can be shortened safely in a different way (terse rewriting of each step, as the rival
  does with Chain-of-Draft, rather than removing steps).
- Whether the per-type map transfers to the full evaluation (2,759 × 8), never run here.
- Training randomness was not measured for track 3 (night 1: a retrain moved results 2–8 points).
- Behaviour outside the benchmarks (agreeing with wrong user statements, inventing details) never checked.

## 7. Where the models are

Google Drive (Colab account), folder `thinkingcap-mvt-pertype/`: `adapters/combine-codecut`
(current best, untouched), `adapters/combine-light`, `adapters/combine-full`, `adapters/grid/*`
(25 per-type adapters + extra), and, as round 1 finishes copying, `adapters/onpolicy-r1*` with every
checkpoint; `runs/` with all evaluations. Code state of the best model: git tag `mvt-pertype-v1`.
