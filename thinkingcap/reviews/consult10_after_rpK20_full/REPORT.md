# Strategy Review After `rpK-20` Full Test and `rpL` Dev-512

---

## 1. What the `rpK-20` Full Test Changes & Where Accuracy Went

On the full test ($2,759 \times 8$), **`rpK-20` scored 49.88% saved / -1.24 accuracy** (rounded: **49.9% / -1.3**) vs `rpE-80`'s **47.04% saved / -0.90 accuracy**—a paired full-test shift of **+2.84% saved / -0.34 accuracy**, compared to `Dev-512`'s paired reading of **+3.59% saved / +0.15 accuracy**.

| Part ($N$ Full / $N$ Dev) | `rpE-80` Full (Saved / $\Delta$Acc) | `rpK-20` Full (Saved / $\Delta$Acc) | Paired Full (`rpK-20` $-$ `rpE-80`) | Paired `Dev-512` (`rpK-20` $-$ `rpE-80`) | Full $-$ Dev Paired Gap (Saved / Acc) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **SVAMP Math** (1,000 / 140) | 54.1% / -1.25 | 57.6% / -2.59 | **+3.44% / -1.34** (`-25.0` thk, `+1.5` rsp) | +0.66% / -0.80 (`-7.5` thk, `+0.0` rsp) | **+2.77% / -0.53** |
| **MBPP+ Code** (378 / 100) | 46.3% / -1.03 | 51.1% / -0.26 | **+4.82% / +0.76** (`-98.6` thk, `-4.0` rsp) | +7.24% / -0.38 (`-146.0` thk, `+0.2` rsp) | -2.41% / **+1.14** |
| **IFEval** (541 / 132) | 45.4% / -2.38 | 45.3% / -2.93 | **-0.15% / -0.55** (`-5.3` thk, `+6.3` rsp) | +2.95% / +2.18 (`-11.1` thk, `-9.8` rsp) | **-3.10% / -2.73** |
| **BFCL Simple** (400 / 60) | 37.3% / +0.09 | 39.5% / +0.28 | **+2.18% / +0.19** (`-4.2` thk, `-0.0` rsp) | +1.85% / -0.42 (`-3.1` thk, `-0.2` rsp) | +0.34% / +0.60 |
| **BFCL Multiple** (200 / 35) | 39.4% / +0.06 | 42.0% / -1.50 | **+2.56% / -1.56** (`-5.6` thk, `+0.0` rsp) | +2.77% / -2.50 (`-5.8` thk, `+0.1` rsp) | -0.21% / +0.94 |
| **BFCL No-Fit** (240 / 45) | 49.5% / +3.44 | 53.8% / +3.65 | **+4.28% / +0.21** (`-5.9` thk, `-3.2` rsp) | +5.61% / +1.39 (`-6.1` thk, `-4.2` rsp) | -1.33% / -1.18 |
| **BFCL Pooled** (840 / 140) | 41.5% / +1.04 | 44.4% / +0.82 | **+2.89% / -0.22** (`-5.0` thk, `-0.9` rsp) | +3.16% / -0.40 (`-4.6` thk, `-1.3` rsp) | -0.27% / +0.18 |
| **4-Part Headline** | **47.04% / -0.90** | **49.88% / -1.24** | **+2.84% / -0.34** | **+3.59% / +0.15** | **-0.75% / -0.49** |

### 1.1 Exact Per-Question Diagnosis of Where `rpK-20` Lost Accuracy Against `rpE-80`

1. **SVAMP Math (`-1.34` pts vs `rpE-80`, `-107` right answers of 8,000; `-0.34` pts on the headline):**
   - **Which questions & which lengths:** Dividing SVAMP's 1,000 questions into quartiles of `rpE-80`'s thinking length (`fulltest_per_question.csv`), Q1 (`81–152` tokens, mean `128`) changed by `-0.60%` (`-4.7` thinking tokens), and Q2 (`152–200` tokens, mean `174`) changed by `+0.15%` (`-5.7` tokens). Virtually the entire loss sits in **Q3 (`201–322` tokens, mean `251`: `-1.15%` acc, `-8.7` tokens)** and **Q4 (`322–2,574` tokens, mean `673`: `-3.75%` acc, `-81.0` tokens, `673 -> 592`)**, which together hold **`-98` of the `-107` net lost right answers**. By original solve rate, `8/8` questions ($n=701$) slipped `-0.80%` (`-3.78% -> -4.58%`), `5–7/8` ($n=205$) lost `-2.50%` (`+1.16% -> -1.34%`), and `<=4/8` ($n=94$) lost `-2.79%` (`+12.37% -> +9.57%`). On the 61 questions that lost $\ge 2$ of 8 answers, `rpE-80` thought `487.3` tokens (`80.9%` right) and `rpK-20` cut thinking by `-55.7` tokens to `431.6` (`50.2%` right).
   - **What kind of error (and what `NEW_RESULTS.md`'s `<0.5x` check missed):** Runaways did not change (`3` vs `3` of 8,000 = `0.04%`). Checking `< 0.50x` of `rpE-80`'s median right thinking on that question (`e80_med_right_thk`) shows only `+24` new wrong answers (`82 -> 106`), which made it look like *"not mainly an early-stop problem."* **Look instead at the `0.50x to 1.00x` band of `e80_med_right_thk` (`fulltest_per_answer.csv`):**
     - Wrong finished answers with `thinking < 0.50x e80_med_right_thk`: `82` (`rpE-80`) $\to$ `106` (`rpK-20`) (**`+24` wrongs**).
     - Wrong finished answers in **`0.50x to 1.00x e80_med_right_thk`**: `284` (`rpE-80`) $\to$ `375` (`rpK-20`) (**`+91` wrongs**).
     - Total wrong finished answers **`< 1.00x e80_med_right_thk`**: `366` (`rpE-80`) $\to$ `481` (`rpK-20`) (**`+115` wrongs = 107.5% of the entire `+107` net SVAMP loss!** Meanwhile, wrong answers with `thinking >= 1.00x e80_med_right_thk` fell from `635` to `627`, `-8`).
     - Conversely, in `rpI-120` (which *gained* `+102` right answers = `+1.27%` over `rpE-80` on SVAMP), wrong answers with `thinking < 1.00x e80_med_right_thk` dropped from `366` to `249` (**`-117` wrongs**).
   - **Why `rpK`'s math pairs caused this:** Because `rpE-80` is already compressed to a `190`-token median on SVAMP, `0.60x–0.95x` of median right thinking (`120–180` tokens on easy questions, `240–420` tokens on Q3/Q4 questions) is not an aborted stub—it is a **1-pass calculation that skips the final 40–80 token re-read/unit-check step**. In `repair_pairs.py`, `capguard` only catches `thinking_len(w) < 0.5 * tmed` (completely blind to the `0.50x–1.00x tmed` failure band), while `right_cr` selects `rights[0]` (the **shortest** right rollout, typically `0.70x–0.85x tmed`), which made `chosen_thinking < rejected_thinking` in **59.4% of `rpK`'s SVAMP `right_cr` pairs** (`405.4` vs `523.0` tokens). Combined with shared-LoRA shortening spillover from IFEval/BFCL, `rpK-20` pushed SVAMP along the exact full-test SVAMP curve (`rpI-120` `47.0% / +0.02` $\to$ `rpE-80` `54.1% / -1.25` $\to$ `rpE-60` `55.0% / -1.50` $\to$ `rpK-20` `57.6% / -2.59` $\to$ `rpE-40` `59.0% / -3.00`, ~`-0.38` pts accuracy per `+1%` SVAMP saving).

2. **Why `Dev-512` Hid the Math Loss (Two Distinct Causes):**
   - **Absolute level offset (`+3.3` to `+3.8` pts across all models):** `devset()` called `own_repair_questions(..., public=False)`, drawing 140 full-difficulty `gsm8k-std` questions where `base` solves `74.73%`, has `7.77%` runaways (`87/1,120`), and has `25.0%` hard (`<=4/8`) questions and only `44.3%` easy (`8/8`) questions. Cutting `7.77%` runaways to `0.6%` gives every trained model a `+12.8%` boost on the `25%` hard bucket (`+3.2` pts overall). Full-test SVAMP has `88.70%` base accuracy, `1.74%` runaways, `70.1%` `8/8` questions (where accuracy can only go down), and only `9.4%` `<=4/8` questions.
   - **Paired delta distortion (`-0.80%` on Dev vs `-1.34%` on Full, and `+0.66%` vs `+3.44%` saved):** Because `rpK`'s math pairs were built on `rpI`'s `gsm8k-std` rollouts (`capguard` mean `916.5` vs `498.6` tokens), they protected the **`>600`-token GSM8K tail** (on Dev-512 Q4, where `rpE-80` thinks `1,319` tokens, `rpK-20` thought `1,309` tokens and gained `+1.07%` accuracy), while leaving the `<600`-token SVAMP regime unprotected. **Crucial verification:** On the 105 `Dev-512` math questions where `rpE-80` thinks `<600` tokens (Dev Q1–Q3: mean `312.3` tokens, base accuracy `86.4%`—matching SVAMP's `306.7` tokens and `88.7%`), `rpK-20` lost **`-1.43%` against `rpE-80` (`87.14% -> 85.71%`)**, matching Full-Test SVAMP's **`-1.34%`** to within `0.09` points!

3. **BFCL Multiple (`-1.56` pts vs `rpE-80`, `-25` right answers of 1,600):**
   - Concentrated on the 47 hard (`<=4/8`) questions, where `rpE-80`'s `+10.11%` gain over `base` fell to `+2.66%` (`-7.45%` vs `rpE-80`). Of the 25 lost answers, 17 were wrong function/argument calls (`327 -> 344`) and 8 were non-calls (`55 -> 63`, call rate `96.56% -> 96.06%`).
   - `Dev-512` **did** show this (`-2.50%` paired on `Dev-512` `bfcl_multiple`): `--mix-preset typed` gave `bfcl_multiple` `70% shorten / 30% right_cr` (`84 / 36`), over-shortening multi-tool selection.

4. **IFEval (`-0.55` pts vs `rpE-80`, `-24` right answers of 4,328; `-0.15%` saved):**
   - On the full test, `rpK-20`'s response length rose `223.3 -> 229.7` tokens (`+6.3`) and runaways rose `2.43% -> 2.73%` (`105 -> 118` answers), with losses on `at_least_N` (`-1.72%`), `fewer_than_N` (`-1.74%`), and `other_counting` (`-0.82%`), led by `number_paragraphs` (`-4.6%`), `english_lowercase` (`-4.2%`), and `startend:quotation` (`-3.7%`, runaways `11.3% -> 12.8%`).
   - Why `Dev-512` missed it at step 20: on `Dev-512`'s 132 IFEval questions, `rpK-20` had a 1-checkpoint noise spike of `+2.18%` (`38 -> 36` runaways), whereas `rpK-40` (`43` runaways) and `rpK-60` (`49` runaways, `-0.57%` acc) showed the exact runaway inflation that appeared immediately at step 20 on the 541-question full test.

### 1.2 What Was Wrong in My `consult9` Forecast (`50.1% / -0.80` vs `49.9% / -1.24`)
My saving forecast (`50.1%`) was within `0.2` points of `49.9%`, but my accuracy forecast (`-0.80`) was **`0.44` points too optimistic** (`0.49` points on the paired delta: I used `Dev-512`'s `+0.15` paired delta instead of `-0.34`). Specifically:
1. **I trusted `rpK-20`'s single-step `+2.18%` IFEval spike on `Dev-512`** instead of shrinking it toward the step-40/60 runaway trend; that `2.73`-point Full-vs-Dev IFEval gap alone cost **`-0.68` points on the 4-part headline**.
2. **I missed the GSM8K-vs-SVAMP domain shift inside `Dev-512` Math:** I assumed `rpK`'s 160 `gsm8k-std` pairs would hold SVAMP length and accuracy as they held `Dev-512`'s `>600`-token GSM8K tail, missing an extra **`-0.13` points of paired headline accuracy** (`-0.34` points total vs `rpE-80`).

---

## 2. `rpL`: Reading Its Results, What Is Broken, and Which Checkpoint to Full-Test

### 2.1 Is `rpL`'s Gain Real?
**Yes—on saving and loop suppression, `rpL` is a major breakthrough.** Paired against `rpE-80` on `Dev-512`, `rpL` gains **`+3.4%` (step 20), `+6.9%` (step 40), and `+11.9%` (`[+9.6, +14.4]`, step 60)** of headline saving (`50.3% / 53.8% / 58.9%`), out-saving `rpK` by **`+3.4` points at step 40** and **`+5.1` points at step 60**:
- **IFEval saving jumped by `+12.5` points over `rpE-80` (`44.7% -> 57.2%` at step 60; `+11.9` points over `rpK-60`'s `45.3%`):** Total IFEval tokens dropped from `146+245 = 392` (`rpE-80`) and `113+275 = 388` (`rpK-60`) to **`104+199 = 303` tokens** at `rpL-60`. Runaways dropped from `3.60%` (`rpE-80`) and `4.64%` (`rpK-60`) to **`2.94% / 2.84% / 2.94%`**. On `no_counting` (66 questions = 50% of IFEval), accuracy **rose `+2.1` to `+2.5` points over `rpE-80`** (`65.3% -> 67.8% / 66.3% / 67.4%`) as runaways fell from `5.68%` to `3.79%`.
- **MBPP+ Code saving jumped to `66.0%` at step 60 (`+2.62` Dev acc):** The 14 code `anti-loop` pairs (`2,047` vs `8,192` tokens) cut 8,192-token code runaways from `1.12%` (`9/800` in `rpE-80`) to **`0.12%` (`1/800` at `rpL-60`)**, and rollouts $\ge 3,500$ tokens from `6.50%` to `2.75%`.
- **BFCL Pooled saving reached `55.2%` at `+3.90` Dev accuracy at step 60** (`bfcl_simple` `51.9% / +5.21`, `bfcl_irrelevance` `59.1% / +5.56`).

### 2.2 What Is Wrong or Missing in `rpL` (Verified Per Question and Per Pair)

1. **Why IFEval `"at least N"` Accuracy Collapsed (`50.4% -> 47.5% -> 44.3% -> 42.1%` on `Dev-512`):**
   - Splitting `Dev-512`'s 35 `"at least N"` questions by rule (`dev512_per_question.csv`) reveals that on the **21 non-length questions** (`keywords:frequency`, `letter_frequency`, `number_placeholders`, `number_highlighted_sections`), accuracy **did not drop at all** (`58.9%` in `rpE-80` $\to$ `58.3% / 60.7% / 57.1%` in `rpL`).
   - **100% of the collapse occurred on the 14 true `length_constraints:number_words / number_sentences` ("at least N") questions**, where accuracy fell from **`37.5%` (`rpE-80`) and `40.2%` (`rpK-20`) to `31.2%` (`rpL-20`) $\to$ `19.6%` (`rpL-40`) $\to$ `19.6%` (`rpL-60`)** as mean responses shrank from `367.2` tokens (`rpE-80`) to `260.9 -> 235.9 -> 291.2` (non-runaway mean `151.1` tokens at step 60; e.g., `ifeval-3846` `932 -> 583` tokens, right `7/8 -> 1/8`; `dp5-ifeval-3994` `747 -> 524` tokens, right `6/8 -> 2/8`; `ifeval-2716` `104 -> 82` tokens, right `7/8 -> 1/8`).
   - **Four interacting causes in `rpL_pairs.csv`:**
     1. **`at_least(r)` diluted the 64-pair `right_count` quota:** Including `number_placeholders`, `number_highlighted_sections`, and `letter_frequency` (short-response rules with `60–100` token responses) inside `at_least(r)` consumed two-thirds of the 64 `right_count` pairs, leaving **only 14 pairs in the entire 1,060-pair file with `chosen_response >= 300` tokens** (down from 39 in `rpK`)! Meanwhile, `length_constraints:number_paragraphs` was omitted from `at_least(r)`.
     2. **`--count-resp-cap 1.35` dropped valid long-response repairs:** On `"at least 400 words"` questions where `rpE-80`'s wrong answer wrote ~250 words (`1.6x` short), capping `chosen_response <= 1.35 * rejected_response + 25` rejected the valid 420-word right answer (`cand["right_count"]` fell from `273` in `rpK` to `169` in `rpL`).
     3. **`--resp-median 0.85 --resp-max 1.05` made `shorten` pairs shrink responses:** Because `long` (`rights[-1]`) usually had an above-median response, allowing `short` down to `0.85 * rmed` while capping at `1.05 * r_long` caused **`chosen_response < rejected_response` in 66.9% of `rpL`'s 160 `shorten` pairs** (`81.8` vs `93.6` tokens, `-11.8` tokens).
     4. **48 IFEval `loop` pairs (`189.8` vs `3,627.1` response tokens) over-weighted EOS in responses:** Step 20 already captured the full runaway reduction (`3.60% -> 2.94%`); steps 40 and 60 did not reduce runaways further (`2.84% / 2.94%`) and instead compressed normal long responses by `-25%`.

2. **Hidden Math Shortening in `rpL` (Invisible on `Dev-512`'s Aggregate Score):**
   - Replacing 16 math `right_cr` pairs with 16 `loop` pairs (`chosen_thinking = 898` vs `rejected_thinking = 4,096`) shifted `rpL`'s 160 math pairs from length-neutral (`chosen < rejected` in `49.4%`, mean `-10.5` tokens in `rpK`) to **strongly shortening (`chosen < rejected` in `60.0%` of math pairs, median `342 vs 390`, mean `-352.7` tokens)**!
   - On `Dev-512` Math, median finished thinking fell from `323.0` (`rpE-80`) and `289.0` (`rpK-60`) to **`268.0` tokens at `rpL-60` (`-17.0%`)**, and on the SVAMP-like `Dev-512` Q1–Q3 slice (`rpE80:thinking < 600`, $n=105$), thinking fell **`-12.7%` at `rpL-60`** (vs `-2.2%` at `rpK-20` and `-3.2%` at `rpK-60`). On full-test SVAMP, `rpL-60` will push SVAMP saving to **`60–62%`** and SVAMP accuracy down to **`-3.4 to -4.2`** (`-2.1 to -2.9` pts worse than `rpE-80`).

### 2.3 Which `rpL` Checkpoint Deserves a Full Test & Expected Reading
- **Do NOT full-test `rpL-20` or `rpL-40`.** `rpL-20` (`50.3% / +1.30` Dev) will read **`50.0–51.2% saved / -1.30 to -1.60 accuracy`** (strictly outside the `-1.0` accuracy box), and `rpL-40` (`53.8% / +0.43` Dev, `-0.93` paired vs `rpE-80`) will read **`53.0–54.5% saved / -1.85 to -2.35 accuracy`**.
- **Full-test `rpL-60` ONLY—by replacing `rpK-40` in GPU 2's queue** (zero net extra GPU hours; `rpK-40` at `50.4% / +0.95` Dev is strictly dominated once `rpK-20` and `rpK-60` are tested).
- **Expected Full-Test reading for `rpL-60`:** **`56.8% to 58.8% saved (central: 57.8%) / -1.60 to -2.15 accuracy (central: -1.85)`** (per part: SVAMP `60–62% / -3.4 to -4.2`, MBPP+ `59–62% / -0.5 to -1.3`, IFEval `55–57% / -3.5 to -4.1`, BFCL pooled `51–53% / +0.8 to +1.4`). While `rpL-60` will miss `-1.0` accuracy due to SVAMP and `"at least N"`, it establishes a landmark high-saving anchor (~`1.6` points more accurate than `rc1-soup-30` at the same `57.5–58%` saving) and can be LoRA-interpolated with `rpE-80` or `rpM`.

---

## 3. The Run After `rpL` (`rpM`): Exact Settings, Math Protection, and the New SVAMP Dev Slice

Why did `rpL-60` lose ~`0.95` points of full-test headline accuracy against `rpE-80` (`-0.90 -> -1.85`) while gaining `+10.8` points of full-test saving? Only three places: **SVAMP Math (`~ -0.55` headline pts)**, **IFEval `"at least N"` (`~ -0.32` headline pts)**, and **`bfcl_multiple` (`~ -0.08` headline pts)**. Fixing those three in `rpM` gives back ~`2.0` points of headline saving (holding SVAMP at `54%` instead of `61%`) and recovers `+0.9 to +1.1` points of headline accuracy, placing `rpM` at **53.5–55.5% saved / -0.65 to -0.90 accuracy on the full test**.

### 3.1 Which Dev Slice Must Be Added So the Next Choice Can Be Trusted on Math
1. **Immediate zero-cost diagnostic on existing `Dev-512` (`Dev-Math-Sub600`, $n=105$):** Score every checkpoint on the 105 `Dev-512` math questions where `rpE80:thinking < 600` tokens (Q1–Q3 of `Dev-512` math). On this slice, `base` solves `86.43%` (Full SVAMP: `88.70%`), `rpE-80` thinks `312.3` tokens (Full SVAMP: `306.7`), and `rpK-20` vs `rpE-80` reads `-1.43%` (Full SVAMP: `-1.34%`).
2. **Add `Dev-SVAMP-200` (200 unseen SVAMP-like questions $\times 8$, ~5 min per model on L4):**
   - `Dev-512` was built with `public=False` in `own_repair_questions()`, which forced 100% of math to come from `gsm8k-std-all.jsonl` and excluded all 6,999 SVAMP-like questions in `pool-dp13` (`metamath_easy`, `metamath_sv`, `gsm8k_easy`, `gsm8k_matched`) + `math1` + `math2`.
   - Meanwhile, `p1src` held out 25% of `pool-dp13` + `math1` + `math2` (`hashlib.md5(str(qid)) % 100 < 25`)—giving **~1,750 unseen SVAMP-like math questions** that were never used in `rpE`, `rpI`, `rpK`, `rpL`, `od1`, `od2`, or `p1src` (and already have `base` $\times 10$ rollouts in `runs/bon/P1_ANSWERS`!).
   - Draw **200 SVAMP-like questions** from `pool-dp13` inside that held-out 25% slice (`public=True`, `id` not in `data/p1-round1-ids.json`, no 8-gram overlap with eval or training) as `data/dev-svamp200.jsonl`. Generate $\times 8$ rollouts for `base`, `rpE-80`, `rpK-20`, `rpL-60`, and `rpM` (~5 min/model on L4), and replace `Dev-512`'s 140 `gsm8k-std` questions with `Dev-SVAMP-200` in `dev_score.py` for all checkpoint selection.

### 3.2 How Math Must Be Protected in `rpM`
1. **Which questions:** Generate `rpE-80` $\times 8$ rollouts on **1,200 SVAMP-like math questions** drawn via `own_repair_questions({"svamp": 1200}, seed=31, public=True, exclude=dev_ids() | dev_svamp200_ids)` (~12 minutes on 1 L4 GPU). Combine these with `rpI`'s existing rollouts so math pairs come **75% from SVAMP-like questions (180 pairs) and 25% from GSM8K (60 pairs)** = **240 math pairs** (raising math's share of the pair file from 15% to 21%).
2. **Which pairs (fixing the `0.50x–1.00x tmed` blind spot and removing the shortening bias):**
   - **0% `shorten` and 0% `loop` on math:** `rpE-80` has only `0.04%` runaways (`3/8,000`) on SVAMP Full Test; math `loop` pairs (`898 vs 4,096` tokens) have zero upside on SVAMP and caused `rpL`'s `-353`-token math shortening push.
   - **50% widened `capguard` (120 pairs):** In `repair_pairs.py`, change `capguard` on `svamp` from `thinking_len(w) < 0.5 * tmed` to **`thinking_len(w) < 0.95 * tmed`** and `capped = [r for r in rights if 0.95 * tmed <= thinking_len(r) <= 1.25 * tmed]` (preferring `REREAD`). This directly targets the `0.50x–0.95x tmed` band where **85% (`+91 / +107`) of `rpK-20`'s SVAMP losses occurred**, ensuring `chosen_thinking > rejected_thinking` by a clean, bounded `+20% to +45%` (`~ +50 to +90` tokens).
   - **50% length-held `right_cr` (120 pairs):** On `svamp`, instead of picking `rights[0]` (which was shorter than `wrongs[0]` in 60–67% of `rpK`/`rpL` math pairs), restrict `pool_r` on `svamp` to **`0.95 * thinking_len(wrongs[0]) <= thinking_len(r) <= 1.25 * thinking_len(wrongs[0])`** (fallback: the right answer closest in thinking length to `tmed`). Across all 240 math pairs, `chosen_thinking >= rejected_thinking` in **~85% of pairs** (median `+35 to +55` tokens), counteracting cross-type shortening from IFEval/BFCL.

### 3.3 Exact Code Edits in `repair_pairs.py` for `rpM` (`--mix-preset typed3`)
1. **Fix `at_least(r)` so `right_count` goes 100% to rules where short responses fail:**
   - Add `"length_constraints:number_paragraphs"` (`relation == "at least"` or `num_paragraphs >= 2`) to `at_least(r)`, and **remove** `"detectable_content:number_placeholders"`, `"detectable_format:number_highlighted_sections"`, and `"keywords:letter_frequency"` from `at_least(r)` (keep them in `counting(r)`).
   - Separate `is_cr = part == "ifeval" and at_least(q[0])` (gets `right_count`) from `no_shorten_if = part == "ifeval" and counting(q[0])` (so `other_counting` questions get `right_cr` rather than `shorten` pairs, preserving thinking on `other_counting`).
   - In `right_count`, require **`1.10 * response_len(rej) + 15 <= response_len(r) <= 1.75 * response_len(rej) + 60`** (`--count-resp-cap 1.75`) so every `right_count` pair genuinely teaches writing enough words to pass `"at least N"`.
2. **Make IFEval `shorten` pairs strictly response-length-neutral:**
   - For `part in think_only`, require **`0.95 * response_len(long) <= response_len(r) <= 1.05 * response_len(long)` and `response_len(r) >= 0.95 * rmed`** (`--resp-median 0.95 --resp-max 1.05`), so `chosen_response` matches `rejected_response` within $\pm 5\%$ (0% response shrinkage on `"at least N"`).
3. **Restrict IFEval `loop` pairs to response-loops (`r["closed_thinking"] and r["truncated"]`) and cut share from 15% to 7.5% (24 pairs).**

### 3.4 Exact `rpM` Run Specification
- **Start model & reference:** `models/rpE-80`, fresh LoRA ($r=16, \alpha=32$, `--bida-precision`).
- **Answers used:** `runs/bon/rp-rpI/rollouts.jsonl` + new `rpE-80` $\times 8$ rollouts on 1,200 SVAMP-like questions (`public=True`, ~12 min generation).
- **Quotas & Pair Mix (1,140 pairs total, `--mix-preset typed3`):**
  - `svamp=240` (180 SVAMP-like + 60 GSM8K): **0% `shorten` / 50% length-held `right_cr` / 50% widened `capguard` (`<0.95*tmed`) / 0% `loop`**.
  - `mbpp=160`: **0% `shorten` / 71.25% `right_cr` (114) / 20% `capguard` (32) / 8.75% `loop` (14)**.
  - `ifeval=320`: **25% `right_count` (80 pairs, on true `number_words` / `number_sentences` / `number_paragraphs` / `keywords:frequency` `"at least N"`, response in `[1.10x+15, 1.75x+60]`) / 45% `shorten` (144 pairs, on `no_counting` only, response within $\pm 5\%$ of `long`) / 22.5% `right_cr` (72 pairs, on `other_counting` + `no_counting`, `--right-resp-cap 1.15`) / 7.5% `loop` (24 response-loop pairs)**.
  - `bfcl_simple_python=160`: **70% `shorten` (112) / 30% `right_cr` (48)**.
  - `bfcl_multiple=120`: **50% `shorten` (60) / 50% `right_cr` (60)** (fixing `rpK`'s `-1.56` full-test loss on `bfcl_multiple`).
  - `bfcl_irrelevance=140`: **60% `shorten` (84) / 40% `right_cr` (56)**.
- **Training & Checkpoints:** `lr=1e-5` constant (`--warmup 4 --lr-min 1e-5`), batch 16, $\beta=0.1$, **60 steps**, scored on `Dev-512` + `Dev-SVAMP-200` at **steps 30, 45, 60**.
- **Decision Rule for Full Test:** Pick the highest-saving checkpoint where:
  1. Paired headline saving vs `rpE-80` is **$\ge +6.0\%$** (`Dev-512` saved $\ge 53.0\%$),
  2. On `Dev-SVAMP-200` (and `Dev-Math-Sub600`), paired math accuracy vs `rpE-80` is **$\ge -0.30$ pts** and math thinking is within **`-4% to +5%` of `rpE-80`** (`295–320` tokens on SVAMP),
  3. On `Dev-512` IFEval, `"at least N"` accuracy is **$\ge 50.0\%$** (no drop vs `rpE-80`'s `50.4%`) and total IFEval tokens $\le 325$.

---

## 4. Redesigned Round 1 of Path 1 (`p1br1s0`, Starting in ~2 Hours)

### 4.1 Assessment of `p1br1s0` as Queued
Keeping the `60/40` and `70/30` mixes at `lr=2e-5` (instead of my `consult9` `90/10` at `5e-5`) and adding `rpL`'s IFEval rules + anti-loop pairs fixed `p1r1s0`'s core bottleneck: in `path1_redesigned_round1_pairs_local_build.csv`, the non-loop IFEval saving ratio jumped from **`28.5%` (`p1r1s0`) to `44.6%` (`p1br1s0`)** (`65.7%` including loops).

However, `p1br1s0` as built has **three concrete flaws** revealed by `rpK-20`'s full test and `rpL`:
1. **Math over-compression (`64.8%` pair saving on `svamp`, `92%` chosen shorter):** Last night's `p1r1s0` (`60% shorten / 40% right_cr` on `svamp`) *already* compressed `Dev-512` math to `57.7–60.2%` saved (`-2.4 to -3.7` Dev accuracy = `-5.5 to -6.5` Full-Test SVAMP accuracy). Replacing 120 math `right_cr` pairs with 120 `loop` pairs (`863 vs 4,090` tokens) in `p1br1s0` pushes `svamp` even shorter, when raw Qwen only has `1.74%` runaways on SVAMP Full Test.
2. **Same IFEval `"at least N"` response shrinkage as `rpL`:** In `p1br1s0`'s 400 IFEval `shorten` pairs, `chosen_response < rejected_response` in **`66.5%` of pairs** (`91.1 vs 108.1` tokens, `-17.0` tokens), and `at_least(r)` + `--count-resp-cap 1.35` starves `"at least N words/sentences"` pairs.
3. **360 `loop` pairs (including 120 8,192-token MBPP+ pairs) will make 200 DPO steps ~4x slower (~2+ hours) and dominate gradients**, and the Round 2 gate (`rpE80 + 6 = 52.9%`) is still set at `6` even though `EXPERIMENT_LOG.md` states Round 2 should run if Round 1 clears `~51%`.

### 4.2 Three Exact Changes to Make Before `p1br1s0` Starts

1. **Edit 1 (`repair_pairs.py`, `round1b` preset, lines 281–286): Protect Math with `capguard`, cut `loop` shares from 360 to 144 pairs (`svamp` 40, `mbpp` 56, `ifeval` 48) to halve training time, and balance `bfcl_multiple` to `50/50`:**
   - In `if args.mix_preset == "round1b":`, set `sh_share` to:
     - `svamp`: `{"shorten": 0.45, "right": 0.35, "guard": 0.15, "loop": 0.05}` (with `capguard` threshold `< 0.95 * tmed`)
     - `mbpp`: `{"shorten": 0.70, "right": 0.13, "guard": 0.10, "loop": 0.07}`
     - `ifeval` (non-counting `n_mix` portion): `{"shorten": 0.55, "right": 0.375, "guard": 0.0, "loop": 0.075}`, `count_share = 0.20`
     - `bfcl_multiple`: `{"shorten": 0.50, "right": 0.50, "guard": 0.0}`
     - `bfcl_irrelevance`: `{"shorten": 0.60, "right": 0.40, "guard": 0.0}`
     - `bfcl_simple_python`: `{"shorten": 0.70, "right": 0.30, "guard": 0.0}`
2. **Edit 2 (`repair_pairs.py`, `at_least(r)` at lines 45–57 & `p1br1s0` CLI flags): Stop IFEval response shrinkage on `"at least N"`:**
   - In `at_least(r)`, restrict true return values to `rule in ("length_constraints:number_words", "length_constraints:number_sentences", "length_constraints:number_paragraphs", "keywords:frequency")` where `kw.get("relation") == "at least"` (or `rule == "length_constraints:number_paragraphs"` with `(kw.get("num_paragraphs") or 0) >= 2`), dropping `number_placeholders`, `number_highlighted_sections`, and `letter_frequency` from `at_least(r)`.
   - In `p1br1s0`'s `--rp-pair-args`, change `--resp-median,0.85,--resp-max,1.05,--count-resp-cap,1.35` to **`--resp-median,0.98,--resp-max,1.05,--count-resp-cap,1.65`**.
3. **Edit 3 (`plan_line_b.txt` / Round 2 gate): Lower `--rp-gate` margin from `6` (`52.9%`) to `4.0` (`50.9%` on `Dev-512`) and pass `typed3` repair rules to Round 2 (`p1br2s0`):**
   - Change `--rp-gate p1br1s0-200,rpE80,6` to **`--rp-gate p1br1s0-200,rpE80,4.0`** so Round 2 automatically runs if `p1br1s0-200` reaches $\ge 50.9\%$ on `Dev-512`.

---

## 5. Plainly: Is the Goal Box (`> 50%` Saved, `> -1.0` Accuracy) Reachable With Margin, and What Is the Shortest Route?

**Yes—the box is reachable with a `3 to 5` point saving margin (`53–55%` saved) and `0.2 to 0.4` points of accuracy margin (`-0.60 to -0.80` on the full test) using LoRA DPO on the model's own answers.**

Why I am confident despite `rpK-20` landing at `49.9% / -1.24`:
1. **`rpK-20` failed on IFEval saving (`45.3%` vs `rpE-80`'s `45.4%`) and lost `-1.34` points on SVAMP—yet still missed the box by only `0.12%` saving and `0.24` points of accuracy.**
2. **`rpL` already proved on `Dev-512` that fixing IFEval's response inflation and adding anti-loop pairs unlocks `+12.5` points of IFEval saving (`44.7% -> 57.2%`), `+11.2` points of BFCL saving (`44.0% -> 55.2%` at `+3.9` acc), and `+16.7` points of MBPP+ saving (`49.3% -> 66.0%` at `0.12%` runaways).**
3. With IFEval at `54–56%` saved, MBPP+ at `55–58%` saved, and BFCL pooled at `50–52%` saved, **we no longer need SVAMP to compress past `rpE-80`'s `54.1%` (`307` tokens)!** Keeping SVAMP at `rpE-80`'s `54.1%` saved / `-1.0` accuracy while taking `rpL`'s gains on the other three benchmarks yields a 4-part geometric mean of **`53.8%` saved at `-0.5 to -0.7` accuracy** on the full test.

### The Shortest Route From Here (In Exact Execution Order)
1. **Right now (0 training cost, on GPU 2):** Cancel `rpK-40`'s full test and replace it with **`rpL-60`'s full test** after `rpK-60` finishes.
   - **Zero-training insurance (`rpE80-rpL60-soup`):** Once `rpL-60` is scored, merging `rpL-60`'s LoRA adapter on `rpE-80` at weight **`0.55` (`merge.py --model models/rpE-80 --adapters adapters/rpL-s0/step-60:0.55`)** interpolates between `rpE-80` (`47.0% / -0.90`) and `rpL-60` (`~57.8% / -1.85`), landing at ~`53.0%` saved on `Dev-512`. Because `rpL`'s `"at least N"` and SVAMP losses accelerated nonlinearly between steps 20 and 60 while anti-loop suppression already saturated by step 20, also check **`0.50 * adapters/rpL-s0/step-20 + 0.50 * adapters/rpL-s0/step-60`** on `Dev-512` + `Dev-SVAMP-200` (~20 min).
2. **Primary winning run (`rpM`, ~1.5 L4 GPU-hours):** Build `Dev-SVAMP-200`, generate `rpE-80` $\times 8$ rollouts on 1,200 SVAMP-like questions (~12 min), and run **`rpM`** (Section 3.4: 60 steps from `rpE-80` with 240 SVAMP-protected math pairs, fixed `at_least(r)` + response-neutral IFEval `shorten` pairs, and 50/50 `bfcl_multiple`). Score on `Dev-512` + `Dev-SVAMP-200` at steps 30, 45, 60, and run one full test on the passing checkpoint (**expected full test: `53.5–55.0% saved / -0.55 to -0.85 accuracy`**).
3. **Clean 2-round recipe from raw `Qwen3-0.6B` (`p1br1s0` $\to$ `p1br2s0` on GPU 1):** Apply the 3 edits in Section 4.2 before `p1br1s0` starts in two hours, then run `p1br2s0` using `rpM`'s exact `typed3` repair rules on `p1br1s0-200`'s own answers.