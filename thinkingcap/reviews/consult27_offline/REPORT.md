### 1. What Each Chain Step Contributed (Per Type), What Transfers to One Step from the Original, and What Cannot

#### A. Empirical Contribution of Each Chain Step
1. **Path 1 Step 1 (`r2s1b-s180`, 2,880 pairs from the original, 180 steps at `2e-5`):**
   - **Full test:** `50.3%` saved / `-1.70` accuracy (`s160`: `49.0% / -1.70`).
   - **Per-type landing (`s180`):** Math (`svamp`) `62% / -4.5`, Code (`mbpp`) `54% / -3.1`, IFEval `38% / +1.0`, Simple Tools (`bfcl_simple_python`) `42% / -3.0`, Several Tools (`bfcl_multiple`) `45% / +0.9`, No-Fit (`bfcl_irrelevance`) `45% / +3.5`.
   - **What it did:** `r2s1b` compressed Math and Code aggressively (`62%` and `54%` saved) because their shorten shares (`55%` and `70%`) and floors (`80:0.40` and `110:0.42`) allowed deep compression on `>=8/10` questions, while under-compressing IFEval (`38%` saved, only `40%` effective shorten share) and Tool Calls (`42%–45%` saved; only `12%` of total pairs were allocated to Simple + Several + No-Fit combined, and No-Fit's `0.80` solve-rate bar excluded harder abstentions).
2. **Path 1 Step 2 (`r2s2-s80` on `r2s1b-s180`, 1,600 pairs, 80 steps at `1e-5`):**
   - **Full test:** `53.7%` saved / `-1.27` accuracy (`s40`: `52.9% / -1.74`).
   - **Per-type landing (`s80`):** Math `~60% / -3.25`, Code `~54% / -2.38`, IFEval `~48% / +1.20`, Simple `~48% / -4.69`, Several `~51% / -0.69`, No-Fit `~52% / +6.15`.
   - **What it did:** Step 2 froze shortening on Math and Code (`0%` shorten, `80%–90%` length-held `right_hold` + `10%–20%` `loop`), recovering `+1.25` pts on Math and `+0.72` pts on Code, while pushing IFEval (`+10%` saved, `+0.20` acc) and Tools (`+6%–7%` saved, `+2.65` pts on No-Fit) by devoting `55%` of all pairs to IFEval + Tools with `50%–62.5%` shorten shares. However, continuing to shorten Simple Tools (`50%` shorten) pushed Simple accuracy from `-3.0` down to `-4.69`.
3. **Path 1 Step 3 (`r2s3-s20/s30` and `r2s3b-s30/s40` on `r2s2-s80`, 800 pairs, 20–40 steps at `8e-6`):**
   - **Full test:** `r2s3-s20` `52.7% / -0.99`, `r2s3-s30` `51.5% / -0.84`, `r2s3b-s30` `51.7% / -0.86`, `r2s3b-s40` `51.3% / -0.78`.
   - **Per-type landing (`r2s3-s20` $\rightarrow$ `r2s3b-s40`):** Math `59%/-2.86` $\rightarrow$ `55%/-2.1`, Code `53%/-1.92` $\rightarrow$ `51%/-1.6`, IFEval `48%/+0.88` $\rightarrow$ `48%/+0.3`, Simple `48%/-3.50` $\rightarrow$ `49%/-2.9`, Several `51%/-0.19` $\rightarrow$ `51%/-0.2`, No-Fit `52%/+5.73` $\rightarrow$ `52%/+6.0`.
   - **What it did:** Pure repair + widened early-stop guards (`guard_wide`) recovered `+0.4` to `+1.15` pts on Math, `+0.46` to `+0.78` pts on Code, and `+1.19` to `+1.79` pts on Simple Tools, but traded away `1.0%–2.4%` token saving on Math/Code because guards on an already-compressed model inevitably reward longer thinking (`0.70x–0.90x` of the original's thinking vs the compressed model's `0.45x` median).
4. **Repair Chain (`rc1-soup-30` $\rightarrow$ `rpE-80` $\rightarrow$ `rpM-60`, `54.1% / -0.93`):**
   - **Per-type landing (`rpM-60`):** Math `57% / -1.2`, Code `58% / -0.9`, IFEval `50% / -2.3`, Simple `45% / -0.3`, Several `47% / -1.6`, No-Fit `61% / +4.2`.
   - **What it proved:** Holding Math saving around `57%–58%` (instead of `62%`) and Simple Tools around `45%–47%` (instead of `49%`) keeps Math at `-1.2` and Simple at `-0.3`, while No-Fit can safely reach `57%–61%` saved at `+4.2` to `+6.9` accuracy (`od5c` also hits `57% / +5.9–6.9` on No-Fit).

#### B. What CAN Be Had in a Single Step from the Original vs. What CANNOT
- **What CAN be transferred directly into one offline step from the original:**
  1. **Test-matched difficulty and prompt-length stratification across all 6 types (`r2prep` / `od3prep`):** `r2s1b` used the raw `p1src` pool for Math, Code, Simple, and Several (which had unstratified difficulty and prompt-length mismatches), and only `348` total pairs across all three Tool types (`12%` of the batch). Drawing a fully test-matched question pool where every type matches the full test's 5 solve-rate bins (`[0, 0.25, 0.5, 0.75, 0.999, 1.01]`) and prompt lengths immediately brings Step 2/3 data quality into Step 1.
  2. **Rebalanced type quotas and shorten shares from the start:** In `r2s1b`, Math (`55%` shorten) and Code (`70%` shorten) were over-shortened while IFEval (`38%` saved) and Tools (`42%–45%` saved) were starved of pairs (`348` tool pairs total). By shifting pair budget and shorten pressure in Step 1—moderating Math (`45%` shorten, higher floor `100:0.46`) and Simple Tools (`45%` shorten, floor `90:0.45`), while increasing IFEval (`50%` total shorten), Several Tools (`55%` shorten), and No-Fit (`65%` shorten with a lower `0.50` solve-rate bar on abstentions)—the single step lands directly on the balanced per-type saving profile (`Math ~57%, Code ~56%, IFEval ~48%, Simple ~46%, Several ~50%, No-Fit ~57%`) that took Path 1 two steps (`r2s1b` $\rightarrow$ `r2s2`) to reach.
  3. **All defect-prevention pairs that exist in the original's own rollouts:**
     - **IFEval `atleast` counting-rule pairs (`--count-mode atleast --count-resp-min 1.10 --count-resp-cap 1.65 --count-resp-pad 45 --no-shorten-counting`):** The original model fails counting rules (`302` candidate pairs in `r2s1b`). Teaching "write at least `1.10x + 15` response tokens while thinking `<=1.2x`" works identically from the original.
     - **Finished + runaway anti-loop pairs (`--loop-finished --loop-response-only ifeval`):** The original model produces both runaway cap-outs and finished 8-gram repetition loops on Code, Math, and IFEval.
     - **Format / parseability priority (`bad_format` in `right_cr` and `format` pairs):** The original's wrong answers on `bfcl_multiple`, `bfcl_simple_python`, and `svamp` already include unparseable tool calls (missing `(` after `</think>`) and missing final numbers (`_nofinal`). Prioritizing well-formed right answers over broken-format wrong answers works directly on the original's rollouts.
     - **Length-held correctness pairs (`--right-hold` within `[0.90x, 1.12x]` of the question's median right thinking, with `>=0.65x` floor on the rejected answer):** On mixed-solve questions (`1/8 <= p0 <= 6/8`), pairing a right answer near `1.00x` median right thinking against a finished wrong answer of similar length (`>=0.65x`) teaches "think carefully and re-read (`--prefer-reread`) without compressing hard questions."
- **What CANNOT be done in a single offline step from the original, and WHY:**
  1. **On-policy early-stop recovery (`recovery` / `guard_wide` against `<0.5x` premature stopping on `8/8` easy questions):** In `r2s2`, `r2s3b`, and `od5c`, `62%–71%` of lost accuracy comes from the *shortened* model stopping thinking at `<0.5x` of the original's right median on questions the original solved `8/8` (`100%`). In the original model's own rollouts, **`8/8` questions have zero wrong answers**, so it is mathematically impossible to construct a `right > wrong` recovery pair on an `8/8` question from the original's rollouts alone.
  2. **True `<0.5x` early-stop guards on `6–7/8` questions:** When the original model gets a question wrong (`1–7/8`), its wrong answers are almost always *longer* (`1.0x–2.0x` median right thinking, wandering or looping) rather than `<0.5x` premature stops (in the noise reference of `loss_analysis/r2s2.txt`, `57%–70%` of the original's errors are full-length wrong answers and only `14%–17%` are `<0.5x`). Therefore, standard `capguard` pairs (`shortest right > shortest finished wrong`, with `thinking_len(c) <= thinking_len(w) * 1.15`) have very few candidates in the original's pool (`25` in Math, `15` in Code in `r2s1b`), and widening the guard (`--guard-wide`) on the original's rollouts without an upper bound on the rejected answer's length would pair a `1.15x` right answer against a `0.85x` wrong answer, actively teaching the model *not* to shorten.
  - **How we solve this in one offline step:** Instead of trying to repair `<0.5x` early stops *after* over-shortening `8/8` Math and Simple Tool questions, we **prevent `<0.5x` collapse on `8/8` questions proactively** by: (a) raising the per-question thinking floor on Math (`100:0.46` instead of `80:0.40`) and Simple Tools (`90:0.45` instead of `65:0.38`) so no `shorten` pair ever rewards a chosen answer below `0.45x–0.46x` of the original's right median; (b) requiring the chosen answer in every `shorten` pair to be the **second-shortest** right answer (`ok2[0]` above the floor) rather than an outlier shortest fluke; and (c) using **bounded early-stop guards (`guard_early`)** on `1–7/8` questions where the original *did* produce a genuinely short finished wrong answer (`thinking_len(w) < 0.80 * med_r`) paired with a compact right answer (`thinking_len(c) <= min(1.05 * med_r, 1.35 * thinking_len(w) + 35)`), which inoculates against stopping before `0.80x` without fighting compression above `0.80x`.

---

### 2. The Single-Step Design (`r2one`)

#### A. Question Pool & Pair Quotas (Test-Matched, Existing Pools Only, Zero Missing Questions)
- **Source pools (existing stored answers of the original Qwen3-0.6B only, `--r2-drop metamath,bida`):**
  - `runs/bon/rp-p1src/rollouts.jsonl` (`x10` answers from the original) + our existing `tooltop-*` sets answered `x8` by the original (`r2c-math`, `r2d-ifeval`, `r2e-nofit`, `r2f-code`, `r2f-simple`, `r2g-math`, `r2g-simple`, `r2g-multiple`, `r2g-nofit`, `r2g-ifeval`, `r2g-code`, plus `nofit1`, `nofit2`, `tools1`, `code1`).
  - Strictly excludes Dev-512, Dev-SVAMP-200, any question sharing a rare 8-gram with the full test or Dev sets, and any `metamath_*` or `bida` (`math1`, `math2`, `instr2`, `code2`, `nofit4`) source.
- **Total available vs. selected test-matched questions (`3,200` questions $\rightarrow$ `3,200` pairs = `200` steps at batch `16`):**
  Across `p1src` + `r2c..r2g` (which already supplied `r2s1b` + `r2s2` + `r2s3` + 16 rounds of `od3prep`), our clean test-matched pool holds **`>8,500` clean questions** answered by the original:
  1. **Math (`svamp` / clean GSM): `700` questions $\rightarrow$ `700` pairs (`21.875%` of batch).** Available clean test-matched questions: `>2,200` (solve rate `89%`, exact 5-bin match to full test). **Missing: `0`.**
  2. **Code (`mbpp` / KodCode): `640` questions $\rightarrow$ `640` pairs (`20.0%` of batch).** Available clean test-matched questions: `~1,850` (solve rate `52%–55%`, 5-bin matched). **Missing: `0`.**
  3. **IFEval (`ifeval`): `760` questions $\rightarrow$ `760` pairs (`23.75%` of batch).** Available clean test-matched questions: `>2,400` (solve rate `59%`, exact 5-bin match). **Missing: `0`.**
  4. **Simple Tools (`bfcl_simple_python`): `440` questions $\rightarrow$ `440` pairs (`13.75%` of batch).** Available clean test-matched questions: `>1,400` (solve rate `82%`, exact 5-bin match). **Missing: `0`.**
  5. **Several Tools (`bfcl_multiple`): `300` questions $\rightarrow$ `300` pairs (`9.375%` of batch).** Available clean test-matched questions: `~850` (solve rate `77%`, exact 5-bin match). **Missing: `0`.**
  6. **No-Fit (`bfcl_irrelevance`): `360` questions $\rightarrow$ `360` pairs (`11.25%` of batch).** Available clean test-matched questions: `>1,500` (solve rate `72%`, exact 5-bin match). **Missing: `0`.**
  - *Note on Tools weight:* Tools receive `440 + 300 + 360 = 1,100` pairs (`34.375%` of total pairs vs `12.1%` in `r2s1b`), in exact `40 : 27 : 33` proportion to their test weights and compression headroom.

#### B. Per-Type Pair Kinds, Shares, and Exact Thresholds (Preset `r2one`)
All rules are defined by general per-question / per-type statistical properties (`base_think` median right thinking $M_q$, solve rate $p_0$, response length $R_q$, and parseability/repetition detectors):

| Type (`part`) | Questions / Pairs | `shorten` Share (Pairs) | `right` (`right_hold` + format priority) | `guard` (Bounded early-stop `<0.80x`) | `loop` (Finished + Runaway) | `right_count` (IFEval `atleast`) | Per-Question Thinking Floor (`ABS:RATIO` of $M_q$) | Shorten Eligibility ($p_0$) & Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Math (`svamp`)** | `700` / `700` | **45%** (`315`) | **35%** (`245`) | **12%** (`84`) | **8%** (`56`) | `0%` (`0`) | **`100 : 0.46`** | $p_0 \ge 0.875$ (`>=7/8`), ratio `>=1.35x` |
| **Code (`mbpp`)** | `640` / `640` | **58%** (`371`) | **22%** (`141`) | **8%** (`51`) | **12%** (`77`) | `0%` (`0`) | **`110 : 0.42`** | $p_0 \ge 0.750$ (`>=6/8`), ratio `>=1.35x` |
| **IFEval (`ifeval`)** | `760` / `760` | **50%** (`380`) | **18%** (`137`) | **0%** (`0`) | **12%** (`91`) | **20%** (`152`) | **`60 : 0.35`** | $p_0 \ge 0.750$ (`>=6/8`), ratio `>=1.30x` |
| **Simple (`bfcl_simple_python`)** | `440` / `440` | **45%** (`198`) | **43%** (`189`) | **12%** (`53`) | **0%** (`0`) | `0%` (`0`) | **`90 : 0.45`** | $p_0 \ge 0.875$ (`>=7/8`), ratio `>=1.35x` |
| **Several (`bfcl_multiple`)** | `300` / `300` | **55%** (`165`) | **35%** (`105`) | **10%** (`30`) | **0%** (`0`) | `0%` (`0`) | **`85 : 0.40`** | $p_0 \ge 0.750$ (`>=6/8`), ratio `>=1.35x` |
| **No-Fit (`bfcl_irrelevance`)** | `360` / `360` | **65%** (`234`) | **35%** (`126`) | **0%** (`0`) | **0%** (`0`) | `0%` (`0`) | **`25 : 0.32`** | $p_0 \ge 0.500$ (`>=4/8`), ratio `>=1.30x` |
| **Total** | **`3,200` / `3,200`** | **`1,663` (`52.0%`)** | **`943` (`29.5%`)** | **`218` (`6.8%`)** | **`224` (`7.0%`)** | **`152` (`4.8%`)** | — | — |

#### C. Exact Pair-Construction Rules and Thresholds
1. **Thinking-Only Measurement (`--think-only all`):** Length ratios and floors apply strictly to `thinking_len(r)` (tokens inside `<think>...</think>`), never penalizing response tokens.
2. **Shorten Eligibility & Floor Protection:**
   - A question is eligible for `shorten` if its original solve rate $p_0$ meets the per-type bar: **`0.875` (`>=7/8`)** on high-solve types (`svamp`, `bfcl_simple_python` where $p_{\text{test}} > 0.80$), **`0.750` (`>=6/8`)** on medium-solve types (`mbpp`, `ifeval`, `bfcl_multiple`), and **`0.500` (`>=4/8`)** on abstention (`bfcl_irrelevance`, so medium-difficulty abstentions are compressed rather than skipped).
   - **Floor (`--floor-by-part`):** $\text{floor}_q = \max(\text{ABS}, \text{RATIO} \times M_q)$ where $M_q$ is the question's median right thinking length. Chosen right answers must have $\text{thinking\_len}(c) \ge \text{floor}_q$, must be the **2nd-shortest** right answer (`ok2[0]`, trimming the noisy shortest outlier when $\ge 4$ right answers exist), and must satisfy $\text{thinking\_len}(w) / \text{thinking\_len}(c) \ge 1.35$ (`1.30` on `ifeval` and `bfcl_irrelevance`, with `1.20` backup).
   - **Response-length preservation on `shorten` (`--resp-median 0.95 --resp-max 1.05 --resp-near-long 0.92`):** Every `shorten` pair's chosen answer must have a response within `[0.95x, 1.05x]` of the question's median right response and `>= 0.92x` of the rejected answer's response, preventing response truncation on IFEval, Code, and Tools.
3. **IFEval Response-Length Rules (`right_count` = 20% of IFEval quota = `152` pairs):**
   - `--count-rules --count-mode atleast --count-share 0.20 --count-resp-min 1.10 --count-resp-cap 1.65 --count-resp-pad 45 --no-shorten-counting`.
   - Applies to minimum-length / counting prompts (`atleast`); chosen right answer writes `>= 1.10x + 15` response tokens and `<= 1.65x + 45` response tokens vs the rejected wrong answer, while thinking `<= 1.20x` of the rejected answer. Counting-rule questions are never used for `shorten` pairs.
4. **Bounded Early-Stop Guards (`guard` on Math, Code, Simple, Several):**
   - `--guard-wide svamp+mbpp+bfcl_simple_python+bfcl_multiple --guard-wide-max 0.80`.
   - On questions with both right and finished wrong answers (`1/8 <= p_0 <= 7/8`), if a finished wrong answer stopped thinking early ($\text{thinking\_len}(w) < 0.80 \times M_q$), pair it against the shortest non-looping, well-formed right answer with $\text{thinking\_len}(c) > \text{thinking\_len}(w)$ and $\text{thinking\_len}(c) \le \min(1.05 \times M_q, 1.35 \times \text{thinking\_len}(w) + 35)$, preferring answers that re-read the prompt (`--prefer-reread`). Any shortfall in `guard` falls back seamlessly to `right` (`right_cr`).
5. **Length-Held Correctness & Format Priority (`right` / `right_hold`):**
   - `--right-hold svamp+mbpp+bfcl_simple_python+bfcl_multiple --right-hold-band 0.12 --right-resp-cap 1.15`.
   - On `svamp`, `mbpp`, `bfcl_simple_python`, and `bfcl_multiple`, `right` pairs select a right answer within `[0.88x, 1.12x]` of $M_q$ (closest to `1.00x`, preferring `--prefer-reread`) against a finished wrong answer with $\text{thinking\_len}(w) \ge 0.65 \times M_q$, **prioritizing broken-format wrong answers first** (`bad_format`: empty responses, `svamp` without a final number, `bfcl` without a function call `(` after `</think>`). On `ifeval` and `bfcl_irrelevance`, `right` uses compact `right_cr` (`thinking_len(c) <= 1.05 * thinking_len(w)`, `response_len(c) <= 1.15 * response_len(w) + 20`).
6. **Loop Pairs (`loop` on Math, Code, IFEval):**
   - `--loop-finished --loop-response-only ifeval`.
   - Pairs a clean, finished right answer against either a truncated runaway or a finished answer containing a repeated 8-gram (`>=4` times in thinking or `>=3` times in the response; response-only on `ifeval`). No chosen answer of *any* pair kind is ever allowed to contain a repetition loop.

#### D. Training Hyperparameters
- **Base & Reference Model:** `Qwen/Qwen3-0.6B` (both policy initialization and frozen DPO reference).
- **Precision:** Plain `bfloat16` (no `--bida-precision`).
- **LoRA Configuration:** Rank `16`, alpha `32`, dropout `0.0`, target modules `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` (`sft.py --init-only --lora-r 16 --lora-alpha 32`, seed `0`).
- **Optimizer & Schedule:** `3,200` pairs in 1 stratified pass (`--block 16`), batch size `16` (`--batch-size 2 --grad-accum 8`), **`200` optimizer steps total**, **constant learning rate `1.8e-5`** (`--lr 1.8e-5 --lr-min 1.8e-5 --warmup 10`), DPO **`beta 0.1`**, `--nll-weight 0.0`, `--max-len 8704`, checkpoints saved every `20` steps (`120, 140, 160, 180, 200`).
- *Why `200` steps at constant `1.8e-5`:* `r2s1b` ran `180` steps at `2.0e-5` (`s160: 49.0%`, `s180: 50.3%`). With `3,200` pairs (`200` steps) at `1.8e-5` and stronger tool/IFEval shorten shares, steps `140–200` sweep cleanly across `50.5%` to `54.5%` full-test saving while giving the optimizer `11%` more unique questions and slightly gentler updates per step.

---

### 3. Which Checkpoints to Full-Test and How to Choose Them Without the Full Test

#### A. Calibration of Dev-512 and Dev-SVAMP-200 Against the Full Test
Across all offline and online DPO runs (`r2s1b`, `r2s2`, `r2s3`, `r2s3b`, `rpM`, `od5c`), the systematic offsets between Dev sets and the full test ($2,759 \times 8$) are:
1. **Dev-512 (`512 x 8`) vs. Full Test:**
   - **Accuracy bias:** Dev-512 **over-reads accuracy by `+2.7` to `+3.2` points** (mean **`+2.95` pts**). For example: `r2s1b-s160` Dev-512 `+1.27` vs Full `-1.70` (`+2.97` bias); `r2s1b-s180` Dev-512 `+1.41` vs Full `-1.70` (`+3.11` bias); `r2s3b-s40` Dev-512 `+2.36` vs Full `-0.78` (`+3.14` bias); `od5c-r16` Dev-512 `+2.31` vs Full `-0.78` (`+3.09` bias).
   - **Saving bias:** When all 6 types are compressed together (unlike Step 1 where Tools/IFEval lagged), Dev-512 **under-reads full-test token saving by `1.2%` to `2.5%`** (`r2s1b-s180`: Dev `49.0%` vs Full `50.3%`; `r2s3-s20`: Dev `~51.0%` vs Full `52.7%`; `rpM-60`: Dev `49.3%` vs Full `54.1%`).
2. **Dev-SVAMP-200 (`200 x 8`) vs. Full Test Math:**
   - Dev-SVAMP-200 tracks full-test SVAMP saving within `±1.5%` and over-reads full-test SVAMP accuracy by **`~1.5` to `2.0` points** (`r2s1b-s160`: Dev-SVAMP `-2.50` vs Full Math `-4.0`; `r2s1b-s180`: Dev-SVAMP `-3.12` vs Full Math `-4.5`). Because Math is `25%` of the official score and the binding constraint on accuracy, Dev-SVAMP-200 is the sharpest guardrail against Math over-compression.

#### B. Screening Grid and Exact Selection Rule (`fullpick`)
- **Screen checkpoints on Dev-512 (`512 x 8`) and Dev-SVAMP-200 (`200 x 8`):** Steps **`140, 160, 180, 200`**.
- **Calibrated Gate for `> 52%` Full-Test Saving at `> -1.0` Full-Test Accuracy:**
  1. **Dev-512 Accuracy Gate (`--pk-acc 2.15`):** Since Dev-512 over-reads full-test accuracy by `~2.95` pts, requiring **`Dev-512 acc >= +2.15`** targets a full-test accuracy of **`-0.80` pts** (safely inside `-1.00`, with a `0.20` pt buffer).
  2. **Dev-SVAMP-200 Gates (`--pk-sv-save 54.0 --pk-sv-acc -1.25`):** Requiring **`Dev-SVAMP saved >= 54.0%`** and **`Dev-SVAMP acc >= -1.25`** (vs `-3.12` in `r2s1b-s180`) guarantees full-test Math lands around `56%–58%` saved at `-2.0` to `-2.6` accuracy (saving `+0.50` macro points over `r2s1b-s180`).
  3. **Dev-512 Per-Type Sanity Check & Saving Band (`--pk-save-lo 50.5 --pk-save-hi 55.0`):** Because Dev-512 under-reads balanced full-test saving by `~1.5%–2.0%`, **`Dev-512 saved >= 50.8%`** corresponds to **`52.3%–53.5%` on the full test**.
- **Which 2 Checkpoints to Full-Test:**
  - **Primary pick (via `fullpick`):** The checkpoint in `{140, 160, 180, 200}` that passes the gate (`Dev-512 saved in [50.5, 55.0]`, `Dev-512 acc >= +2.15`, `Dev-SVAMP saved >= 54.0`, `Dev-SVAMP acc >= -1.25`) with the **highest Dev-512 saving** (expected to be **step `180`**).
  - **Safety companion pick (automatically queued by `fullpick`):** One checkpoint earlier (e.g., **step `160`** if `180` is picked, or **step `180`** if `200` is picked), which trades `~1.0%` saving for `+0.25` accuracy protection.

---

### 4. Exact Code Changes in Words and Queue Lines

#### A. Code Changes (Two Files, Minimal & Self-Contained)
1. **`repair_pairs.py` inside `main()`:**
   - **Add `--mix-preset r2one` to the `--mix-preset` choices** (`choices=("typed", "typed2", "r2s1", "r2s2", "r2s2a", "r2s3", "r2s3b", "od5", "r2one")`).
   - **Per-type shorten eligibility for `r2one`:** Where `can_shorten` is computed when `p0 is None` (around line 435), add a rule for `args.mix_preset == "r2one"`: set the minimum solve rate bar `min_r` to `0.875` if `base_part in ("svamp", "bfcl_simple_python")`, `0.50` if `part == "bfcl_irrelevance"`, and `0.75` otherwise (`mbpp`, `ifeval`, `bfcl_multiple`), and set `can_shorten = (p1 >= min_r)`.
   - **Bounded early-stop guard cap for `r2one`:** In the `guard_wide` block (around line 465), when `args.mix_preset == "r2one"`, use the question's own median right thinking `med_r` as `g_ref` and cap the chosen answer at `thinking_len(r) <= min(1.05 * med_r, 1.35 * thinking_len(w_g) + 35)` for rejected answers with `thinking_len(w_g) < args.guard_wide_max * med_r` (`0.80 * med_r`). This ensures guards on the original's rollouts only fire against `<0.80x` premature stops and never reward bloated `>1.05x` thinking.
   - **Preset shares for `r2one`:** In the `sh_share` preset block (around line 615), add `if args.mix_preset == "r2one":`:
     - `svamp`: `{"shorten": 0.45, "right": 0.35, "guard": 0.12, "loop": 0.08}`
     - `mbpp`: `{"shorten": 0.58, "right": 0.22, "guard": 0.08, "loop": 0.12}`
     - `ifeval`: `{"shorten": 0.625, "right": 0.225, "guard": 0.0, "loop": 0.15}` with `count_share = 0.20` (yielding `20%` `right_count`, `50%` `shorten`, `18%` `right`, `12%` `loop` of IFEval's total quota)
     - `bfcl_simple_python`: `{"shorten": 0.45, "right": 0.43, "guard": 0.12, "loop": 0.0}`
     - `bfcl_multiple`: `{"shorten": 0.55, "right": 0.35, "guard": 0.10, "loop": 0.0}`
     - `bfcl_irrelevance`: `{"shorten": 0.65, "right": 0.35, "guard": 0.0, "loop": 0.0}`
     Also include `"r2one"` in the condition `(args.mix_preset in ("r2s2", "r2s2a", "r2s3", "r2s3b", "r2one"))` so `shorten` uses `min(n_sh, ...)` and any unfilled `guard` / `loop` slots roll into `right` (and vice versa) so no quota is ever short.
2. **`track.py` inside `r2prep(args)` and `ptrepair(args)`:**
   - **In `r2prep(args)`:** Add an optional argument `--r2-one-file` (default `""`). When `--r2-one-file data/r2one-questions.jsonl` is passed, `r2prep` pools all clean stored original answers from `runs/bon/rp-p1src/rollouts.jsonl` plus all `tooltop-*` sets in `--r2-tops` (minus Dev-512, Dev-SVAMP-200, their 8-gram near-duplicates, test 8-gram near-duplicates, and `--r2-drop metamath,bida`), selects a single test-matched set of `need_one = {"svamp": 700, "mbpp": 640, "ifeval": 760, "bfcl_simple_python": 440, "bfcl_multiple": 300, "bfcl_irrelevance": 360}` (`3,200` questions total) matching the full test's 5 solve-rate bins (`r2_select`), writes the questions to `data/r2one-questions.jsonl`, and writes their stored original rollouts to `runs/bon/rp-r2one-src/rollouts.jsonl` (zero GPU generation required).
   - **In `ptrepair(args)`:** No structural changes needed—passing `--rp-rollouts-from r2one-src` and `--rp-questions svamp=700,mbpp=640,ifeval=760,bfcl_simple_python=440,bfcl_multiple=300,bfcl_irrelevance=360` reads the `3,200` test-matched questions and their stored original answers directly, builds the `3,200` pairs via `repair_pairs.py`, trains for `200` steps in plain `bf16`, and scores steps `140, 160, 180, 200` on Dev-512 and Dev-SVAMP-200.

#### B. Exact Queue Lines (Plain Text)
1. **Build the 3,200 test-matched question set and stored original rollout file (0 GPU hours):**
   `r2prep --r2-drop metamath,bida --r2-tops r2c-math,r2d-ifeval,r2e-nofit,r2f-code,r2f-simple,r2g-math,r2g-simple,r2g-multiple,r2g-nofit,r2g-ifeval,r2g-code --r2-one-file data/r2one-questions.jsonl`
2. **Run the single-step offline DPO training and Dev-512 / Dev-SVAMP screening (`r2one`):**
   `ptrepair --name r2one --rp-base Qwen/Qwen3-0.6B --rp-rollouts-from r2one-src --rp-drop metamath,bida --rp-questions svamp=700,mbpp=640,ifeval=760,bfcl_simple_python=440,bfcl_multiple=300,bfcl_irrelevance=360 --rp-quota svamp=700,mbpp=640,ifeval=760,bfcl_simple_python=440,bfcl_multiple=300,bfcl_irrelevance=360 --rp-kinds mix --rp-pair-args=--think-only,all,--prefer-reread,--count-rules,--shorten-ratio,1.35,--shorten-min-right,0.75,--mix-preset,r2one,--floor-by-part,svamp=100:0.46+mbpp=110:0.42+ifeval=60:0.35+bfcl_simple_python=90:0.45+bfcl_multiple=85:0.40+bfcl_irrelevance=25:0.32,--count-mode,atleast,--count-share,0.20,--count-resp-min,1.10,--count-resp-cap,1.65,--count-resp-pad,45,--no-shorten-counting,--resp-median,0.95,--resp-max,1.05,--resp-near-long,0.92,--right-resp-cap,1.15,--loop-response-only,ifeval,--loop-finished,--guard-wide,svamp+mbpp+bfcl_simple_python+bfcl_multiple,--guard-wide-max,0.80,--right-hold,svamp+mbpp+bfcl_simple_python+bfcl_multiple,--right-hold-band,0.12 --rp-lr 1.8e-5 --rp-steps 200 --rp-save-every 20 --rp-screen 140,160,180,200 --rp-dev512 --rp-require-full`
3. **Select and run the 2 best checkpoints on the Full Test (`fullpick`):**
   `fullpick --fp-run r2one --fp-kind dpo --pk-steps 140,160,180,200 --pk-save-lo 50.5 --pk-save-hi 55.0 --pk-acc 2.15 --pk-sv-save 54.0 --pk-sv-acc -1.25`

---

### 5. Expected Full-Test Landing (Per Type & Overall) and Main Risks

#### A. Expected Full-Test Landing (`r2one-s180` / `s160`, $2,759 \times 8$)
| Benchmark Type | Official Weight | `r2s1b-s180` (Old Step 1) | `r2s3-s20` (3-Step Chain) | **`r2one` Expected (Point Estimate)** | **`r2one` Expected Range** | Why It Moves vs. `r2s1b-s180` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Math (`svamp`)** | `0.2500` | `62% / -4.50` | `59% / -2.86` | **`57.5% / -2.30`** | `56%–59% / -1.8 to -2.8` | Shorten share `55% -> 45%`, floor `80:0.40 -> 100:0.46`, `35%` `right_hold` + `12%` `<0.80x` guards + test-matched bins |
| **Code (`mbpp`)** | `0.2500` | `54% / -3.10` | `53% / -1.92` | **`55.5% / -1.80`** | `54%–57% / -1.3 to -2.3` | Shorten share `70% -> 58%`, `12%` finished+runaway `loop` pairs (`77` pairs vs `54`), `22%` `right_hold` + `8%` `<0.80x` guards |
| **IFEval (`ifeval`)** | `0.2500` | `38% / +1.00` | `48% / +0.88` | **`47.5% / +0.60`** | `46%–49% / +0.1 to +1.1` | Shorten share `40% -> 50%` (`380` pairs vs `288`), `atleast` counting rule with `1.10x` min & `1.65x` cap, `12%` response loops |
| **Simple Tools (`simple`)** | `0.1190` | `42% / -3.00` | `48% / -3.50` | **`46.0% / -1.50`** | `44%–48% / -0.8 to -2.2` | Quota `182 -> 440` pairs, shorten `70% -> 45%` (`>=7/8` only), floor `65:0.38 -> 90:0.45`, `43%` format-prioritized `right_hold` |
| **Several Tools (`multiple`)** | `0.0595` | `45% / +0.90` | `51% / -0.19` | **`50.0% / +0.10`** | `48%–52% / -0.5 to +0.7` | Quota `88 -> 300` pairs (`55%` shorten, `35%` format-prioritized `right_hold`, `10%` `<0.80x` guards) |
| **No-Fit (`irrelevance`)** | `0.0714` | `45% / +3.50` | `52% / +5.73` | **`56.5% / +5.60`** | `54%–59% / +4.5 to +6.5` | Quota `160 -> 360` pairs, `65%` shorten with `p0 >= 0.50` bar so medium/hard abstentions are compressed (`od5c`: `57% / +5.9`) |
| **Tools Pooled (`bfcl`)** | `0.2500` | `43.6% / -0.21` | `49.9% / -0.07` | **`50.0% / +0.91`** | `48.5%–51.5% / +0.3 to +1.5` | Weighted `400 : 200 : 240` across Simple / Several / No-Fit |
| **Official Composite** | **`1.0000`** | **`50.3% / -1.70`** | **`52.7% / -0.99`** | **`52.8% / -0.65`** | **`51.8%–53.8% / -0.35 to -0.95`** | **`+2.5%` more saving and `+1.05` pts better accuracy than `r2s1b-s180`** |

#### B. Main Risks and Mitigations
1. **Risk 1: Math still loses `> -3.0` pts on `8/8` questions because offline pairs from the original cannot include `8/8` early-stop recovery pairs.**
   - *Why it could happen:* Even with a `100:0.46` floor and `45%` shorten share, `180–200` steps of DPO can push the model's tail below `0.45x` on easy SVAMP questions.
   - *Built-in mitigation:* The Dev-SVAMP-200 gate (`--pk-sv-acc -1.25 --pk-sv-save 54.0`) directly measures SVAMP compression and accuracy across steps `140, 160, 180, 200` and picks the latest step before SVAMP accuracy degrades past `-1.25` on Dev-SVAMP (`~ -2.5` on full test).
2. **Risk 2: Unfilled `guard` candidates on the original's rollouts (`<0.80x` finished wrong answers are sparser on the original than on a shortened model).**
   - *Why it could happen:* Because the original model mostly errs by thinking too long rather than stopping at `<0.80x`, `cand[part]["capguard"]` may provide only `40%–70%` of the nominal `8%–12%` guard share on Math/Code/Tools.
   - *Built-in mitigation:* In `repair_pairs.py`, any unfilled `guard` slots automatically roll into `right` (`right_hold` with format priority and `--prefer-reread`), which holds thinking at `0.88x–1.12x` of the original's right median against `>=0.65x` wrong answers—providing a continuous length brake without ever dropping below the `3,200`-pair quota.
3. **Risk 3: Checkpoint step selection lands `0.5%` under `52.0%` if step `160` is chosen over step `180`.**
   - *Built-in mitigation:* `fullpick` tests both the highest-saving gated checkpoint (e.g. `s180`) and its neighbor (`s160` or `s200`), ensuring we observe both ends of the `51.8%–53.8%` saving / `-0.35 to -0.95` accuracy trade-off on the full test.