Here is the complete pre-launch review of `od5h` answering all five questions, correcting the `consult31` errors, integrating `effsame.txt` and `od5g-r20` CIs, and giving the exact code/flag changes before launch.

---

### 0. Corrections to `consult31` Errors & Integration of New Data

1. **Math 8/8 Consistency and Excess Churn (`SMARTER_DATA.md`):**
   - In `consult31` Section 5, I misquoted `od5c-r16` (`54.8%` saved, `-0.78` acc) as having math 8/8 consistency `75.1% -> 69.9%` and `+168 / -145` question flips. The actual full-test numbers in `SMARTER_DATA.md` are:
     - **Math Pass@8:** `96.8% -> 96.4%` (`-0.4` pts); **Math 8/8:** **`70.1% -> 61.7%`** (`-8.4` pts).
     - **All-task Pass@8:** `88.2% -> 88.1%`; **All-task 8/8:** `56.1% -> 54.3%` (`-1.8` pts; No-fit `72.5% -> 81.7%`, Multiple `56.0% -> 64.0%`, IFEval `34.0% -> 34.9%`, Simple `66.5% -> 64.2%`, Code `29.1% -> 29.1%`).
     - **Net drift (questions / pts):** Math `-28.4 q / -2.84 pts`, Code `-7.4 q / -1.95 pts`, IFEval `+2.6 q / +0.49 pts`, Simple `-4.4 q / -1.09 pts`, Multiple `0.0 q / 0.00 pts`, No-fit `+14.1 q / +5.89 pts`, All `-23.4 q / -0.85 pts`.
     - **Excess churn above the exact hypergeometric sampling-noise null ($E_0$):** **All tasks `+73.1` questions / `+2.65` pts `[+61.2, +85.2]`** (Math `+21.2 q / +2.12 pts [15.2, 27.0]`, Code `+6.4 q / +1.70 pts`, IFEval `+11.5 q / +2.12 pts`, Simple `+13.4 q / +3.34 pts`, Multiple `+12.1 q / +6.07 pts`, No-fit `+8.5 q / +3.53 pts`).
   - **How this corrects the reasoning:** Math's `-2.84` pt net accuracy drop in `od5c-r16` did **not** come from losing problem-solving capability (`Pass@8` moved only `-0.4` pts); it came from an `-8.4` pt drop in 8/8 per-rollout consistency (`70.1% -> 61.7%`) when math was shortened to ~62% saved. That means we must **not** push math above `60%` saved to reach our `>60%` macro target; keeping math smoothly capped at `~59-60%` with `--od5-held` compact `be-right` pairs and getting the extra `+3.5%` macro saving from Code, IFEval, and No-Fit (where 8/8 consistency stayed flat or rose in `SMARTER_DATA.md`) is essential.
2. **`r2s3-s80` vs `rpL-80`:**
   - The `65.9% saved / +7.7 pts` full-test score on No-Fit is `rpL-80` (offline DPO run) in `fulltests.json` (`r2s3-s80` scored `64.0% / +6.0` on No-Fit). Both prove that No-Fit sustains `64-66%` saving with `+6.0 to +7.7` pts accuracy gain on the full test.
3. **Learning rate (`1.4e-5`) and retained flags from `od5g`:**
   - Every `od5` run (`od5c`, `od5r`, `od5f`, `od5g`) used `--rp-lr 1.4e-5`. Keeping `1.4e-5` in `od5h` is right.
   - Keeping `od5g`'s exact pair flags (`--count-mode atleast`, `--base-stats data/od3-base-stats.json`, `--base-resp data/od3-base-resp.json`, `--base-think data/od3-base-think.json`, `--think-only ifeval`, `--prefer-reread`, `--count-rules`, `--resp-median 0.95 --resp-max 1.05`) and dropping the monitor early-stop (`--od-stop-from`) are **all right**:
     - `--base-stats`, `--base-resp`, and `--base-think` are required by `if args.kinds == "mix" and od5:` in `repair_pairs.py` (lines 285-307, 333, 391, 439) to compute question-level damage (`k0 - k1`), the original's thinking floor `bt`, and IFEval response-length caps `bresp`. Dropping them would silently disable `damaged` detection and `right_resp_cap`!
     - Dropping monitor-based early stop (`--od-stop-from`) is right because the round monitor is a single 150-question slice with `+2.5` to `+3.0` pts accuracy optimism and `±4%` slice-to-slice saving noise.
4. **New `effsame.txt` Data (Equal Thinking Caps on Unseen Slices 19-20):**
   - **Math (`svamp`):** At equal token caps, `od5c-best` is far more accurate than the original (`130t`: `38.8%` vs `12.0%`; `205t`: `67.8%` vs `30.2%`; `322t`: `80.8%` vs `59.5%`).
   - **Code (`mbpp`) & Simple (`bfcl_simple_python`):** Even the original model achieves `50.0-51.1%` on Code at `183-817t` (vs `52.3%` at `1813t`) and `80.6-82.8%` on Simple at `47-92t` (vs `81.2%` at `150t`), confirming there is no capability cliff when shortening Code to ~63% saved.
   - **IFEval:** Trained model is ~4 pts higher than the original at equal thinking caps (`~62%` vs `57.2-58.4%`).
   - **No-Fit (`bfcl_irrelevance`):** The original model is **most accurate with ZERO thinking (`b0 = 0t`: `87.5%`, `+5.3 pts [+0.3, +10.5]` vs natural `82.2%` at `210t`)**, whereas partial thinking (`b1=53t..b9=115t`) drops accuracy to `77.8-80.9%`. When the model thinks at length on a no-fit prompt, it talks itself into calling a tangential tool; when it immediately recognizes no tool fits (`0-40t`), it abstains cleanly.

---

### Question 1: Line-by-Line Review of `--od5-held` (`repair_pairs.py`)

Tracing every branch of `repair_pairs.py` when `--od5-held` is active (`args.od5_v2` and `args.od5_held` are both True):

1. **`right_count` / `loop` / `format` pairs taken before `n_rem` (`lines 683-687`):**
   - `got5` takes `right_count` (`10%` of `n` on IFEval), `loop` (`12%` of `n`), and `format` (`10%` of `n`) **before** `n_rem = n - len(got5)` and before `if args.od5_held and c_sh == 0:` (line 689).
   - **Verdict:** Right. These are pure defect repairs (`loop` replaces a truncated/looping wrong answer with the shortest clean right answer; `format` replaces an empty-response / missing-final-number / missing-tool-call wrong answer with a right answer within `0.85x-1.25x` thinking; `right_count` fixes IFEval response length with thinking `<= 1.15x`). Taking them before `n_rem` ensures held parts never lose protection against loops or format errors.

2. **Held + undamaged (`dmg > -0.02`) vs Held + damaged (`dmg <= -0.02`) (`lines 689-698`):**
   - **Held + undamaged (`dmg > -0.02`):** `n_rem = min(n_rem, int(round(0.40 * n)))`. Then `rec_h` takes at most `0.20 * n_rem` recovery pairs (`~4` pairs on math `n=48`, `~2` on simple/multiple `n=28`), and `hold_h` takes up to `0.80 * n_rem` compact `be-right` pairs (`~15` on math, `~9` on simple/multiple).
   - **Crucial fix over `od5f`/`od5g`:** When `cand[part]["hold"]` has fewer candidates than `n_rem - len(rec_h)`, line 694 takes whatever `hold` pairs exist and **stops**—it does **not** backfill the remaining slots with `recovery` (whereas line 720 in the old code dumped all remaining slots into `recovery`, putting up to 38 lengthening pairs onto math and erasing 18-25 pts of saving).
   - **Held + damaged (`dmg <= -0.02`):** `n_rem` is not reduced (`n_rem = n - len(got5)`), and `rec_h` gets `0.40 * n_rem` (up to `19` recovery pairs + `29` compact hold pairs on math).
   - **Can it starve a held part of protection or still lengthen it?**
     - It **does not starve** a held part: defect pairs are taken in full, undamaged slices get up to 80% compact `be-right` + 20% recovery, and any slice losing `> 2.0` pts (`dmg <= -0.02`) immediately gets 100% of its quota with 40% recovery.
     - However, there is **one subtle edge case in candidate construction (`lines 457-473`) when no compact pair is available on a question**:
       - Inside `elif clean and wrongs and 1 <= k1 < n_ans:`, line 457 searches for a compact `be-right` pair (`0.50 <= ratio <= 1.00`, with `right_resp_cap(r)` enforced on IFEval). **If no compact pair exists on that question (`best is None`), execution falls through to line 467 (`for r in (clean if best is None else []):`)**, which searches `0.90 <= ratio <= 1.12` **without** checking `right_resp_cap(r)`! On non-IFEval parts, since `0.90..1.00` was already empty, any fallback pair has `1.00 < ratio <= 1.12` (chosen thinks up to `+12%` longer than rejected); on IFEval, a question whose right answers all failed `right_resp_cap` can get picked by line 467 anyway.
       - **Exact 1-line fix (`repair_pairs.py` line 467):** Guard the fallback so `--od5-v2` + `--od5-held` never falls through to `1.00 < ratio <= 1.12` or bypasses `right_resp_cap`:
         `for r in (clean if best is None and not (args.od5_v2 and (args.od5_held or part not in set(filter(None, re.split(r"[,+]", args.no_shorten_parts))))) else []):`

3. **Is the 40% / 20% split right, and should the freed slots stay unused or go to shortening on the unheld parts?**
   - **Yes, 40% / 20% (and 100% / 40% when damaged) is right, and leaving the freed slots unused in `repair_pairs.py` is BETTER than explicitly filling them with extra shorten candidates.**
   - **Why (`track.py` line 5303 & `dpo.py` line 127):** `ptrepair` runs `dpo.py` with `--max-steps 15 --batch 16 --epochs 2 --keep-order`. As long as total pairs across all 6 parts is `>= 120` (even with 3 parts held at 40%, total pairs is `~165-195 >= 120`), **`dpo.py` still runs all 15 optimizer steps (240 pair-updates)** by wrapping into epoch 2 of the stratified pair file (`order = [i for _ in range(args.epochs) for i in range(len(rows))][:240]`).
   - Because `pairs.jsonl` is interleaved in stratified blocks (`--block 80`), shrinking held parts to 40% of their quota **automatically gives the unheld parts (Code, IFEval, No-Fit) ~25-35% more of the 240 gradient updates** by repeating their **top-ranked** shorten pairs in the wrap-around steps, rather than scraping bottom-of-the-barrel shorten candidates (`od5d` tried `--od5-v3` explicit slot reallocation by digging deeper into the candidate pool and dropped to `49.5% saved / -1.21 acc`).

---

### Question 2: Review of `--od-smooth 2` (`track.py`)

Tracing lines 5446-5458 of `track.py`:
```python
if getattr(args, "od_smooth", 1) > 1:  # (19 Nov, consult31) the brake reads the mean of the last N slices
    import statistics
    prev_h = [h for h in history if k + 1 - args.od_smooth < h["round"] <= k][-(args.od_smooth - 1):]
```
1. **First rounds (`k = 0` and `k = 1`) — ONE BUG FOUND IN ROUND 2 (`k = 1`):**
   - In Round 1 (`k = 0`), the brake does not run (`k > 0` is False), and after Round 1 trains, lines 5471-5476 log Round 1's monitor to `history`: because Round 1 is answered by the **original model (`od3-r1-stored`)**, `h["round"] == 1` has **`0.0%` saved and `0.00` accuracy change on every part**.
   - In **Round 2 (`k = 1`)**, `k + 1 - args.od_smooth < h["round"] <= k` is `0 < h["round"] <= 1`, which **matches Round 1 (`h["round"] == 1`, `0.0 / 0.0`)**!
   - Thus, in Round 2, `per_` averages Round 2's monitor with `0.0 / 0.0`, **cutting Round 2's measured saving and accuracy drop in half** (so a `-3.2` pt drop in Round 2 becomes `-1.6` pts and misses `--od-acc-hold -2.0`).
   - **Exact 1-line fix (`track.py` line 5448):** Exclude Round 1's `0.0/0.0` baseline monitor (`h["round"] > 1`):
     ```python
     prev_h = [h for h in history if max(1, k + 1 - args.od_smooth) < h["round"] <= k][-(args.od_smooth - 1):]
     ```
2. **A part missing from a monitor:**
   - Handled safely: `[v[0]] + [h["per_part"][p_][0] for h in prev_h if p_ in h.get("per_part", {})]` skips any history entry lacking `p_` and always has at least `v[0]` from the current slice.
3. **Interaction with `--od-unlatch` and `--od-acc-hold -2.0`:**
   - With `--od-unlatch`, `held` is recomputed each round from the 2-slice smoothed `per_`. This stops single-slice saving noise (`±4-6%`) from toggling parts in and out of `held` every round.
   - Meanwhile, inside `repair_pairs.py`, `part_dmg` (`dmg <= -0.02`) is computed on the **current slice's unsmoothed rollouts (`k1 - k0`)**. So if an already-held part suffers a sudden `-3%` drop on the current slice, `repair_pairs.py` immediately sees `dmg <= -0.02` and gives it full quota + 40% recovery.

---

### Question 3: Risk Analysis on No-Fit (`4:0.10`, cap `76`, ratio `1.20`, 2nd pair) & Code (`28:0.32`, 2nd pair)

1. **No-Fit (`bfcl_irrelevance`: floor `4:0.10`, cap `76`, ratio `1.20`, 2nd pair):**
   - **Risk of calling tools:** **Near zero (shortening reduces tool calls).** In BFCL irrelevance, `correct = True` means abstaining from any tool call. Every pair in `repair_pairs.py` picks its chosen answer `r` from `clean = [r for r in rights if not looping(r)]`, where `right(r)` requires `bool(r["correct"]) and r["closed_thinking"] and not r["truncated"]`. A rollout that calls a tool is graded `correct = False` and can **never** be chosen. Moreover, `effsame.txt` proves that on No-Fit, **0 thinking tokens (`b0 = 0t`) achieves `87.5%` (`+5.3 pts` over natural `82.2%`)**, whereas partial thinking (`53-115t`) drops accuracy to `77.8-80.9%`.
   - **Risk of format breaking:** `right(r)` requires `r["closed_thinking"] and not r["truncated"]` (`</think>` properly closed), and `cand[part]["format"]` (up to `10%` of quota) explicitly penalizes `response_len(w) == 0`. On the full test, `rpL-80` at `65.9%` saved gained `+7.7` pts on No-Fit (`r2s3-s80` at `64.0%` saved gained `+6.0` pts) with zero format degradation.
2. **Code (`mbpp`: floor `28:0.32`, cap `68`, ratio `1.22`, 2nd pair):**
   - **What prevents format or accuracy collapse on Code:**
     - `correct = True` on `mbpp` requires the generated post-`</think>` Python function to execute and pass the unit-test assertions; syntax errors, missing code blocks, or unclosed `</think>` are all `correct = False`.
     - `shorten` on `mbpp` requires `>= 5/8` passing rollouts (`--shorten-min-by-part mbpp=0.60`).
   - **CRITICAL BUG FOUND IN `shorten2` WHEN `--shorten-skip-shortest` IS ENABLED (`repair_pairs.py` lines 407 & 424-428):**
     Look at lines 407 and 425 of `repair_pairs.py`:
     ```python
     idx0 = 1 if (args.shorten_skip_shortest and len(clean) >= 4) else 0
     c, w = clean[idx0], clean[-1]
     ...
     if (part in set(filter(None, re.split(r"[,+]", args.shorten_second_parts)))
             and len(clean) >= 4 and k1 >= need_r):
         c2, w2 = clean[idx0 + 1], clean[-2]
     ```
     Because `--shorten-skip-shortest` is enabled, whenever `len(clean) >= 4`, `idx0 = 1`, so **`c2 = clean[idx0 + 1] = clean[2]` (the 3rd-shortest clean right answer) and `w2 = clean[-2]` (the 2nd-longest clean right answer)!**
     - When `len(clean) == 4` (`clean[0..3]`): `c2` is `clean[2]` and `w2` is `clean[-2] = clean[2]` — **the exact same rollout!** `thinking_len(w2) >= ratio_need * thinking_len(c2)` is `False`, so **zero 2nd pairs can ever be formed when `len(clean) == 4`!**
     - When `len(clean) == 5` (`clean[0..4]`): `c2 = clean[2]` and `w2 = clean[3]` (adjacent middle ranks), which almost never differ by `>= 1.22x`!
     - Even when `len(clean) >= 6`, pairing the 3rd-shortest against the 2nd-longest loses over half the valid 2nd shorten pairs on Code, IFEval, and No-Fit!
     - **Why:** Every rollout in `clean` has **already passed the per-type thinking floor** (`thinking_len(r) >= max(ABS, RATIO * bt)`). When `idx0 == 1`, Pair 1 uses `clean[1]` vs `clean[-1]`, leaving `clean[0]` (or `clean[1]` when `len(clean) == 4` if `clean[0]` is used) unused. If we only advance to `clean[2]` when `len(clean) >= 6` (or use `c2 = clean[2 if (args.shorten_skip_shortest and len(clean) >= 6) else (0 if idx0 == 1 else 1)]`), then `len(clean) == 4` or `5` uses `clean[0]` vs `clean[-2]` (two distinct rollouts separated by 2-3 ranks, both above the thinking floor) and `len(clean) >= 6` skips `clean[0]` and uses `clean[2]` vs `clean[-2]`.

---

### Question 4: 20 vs 22 Rounds on `od3g`, `fullpick` Dev Rounds, and the Dev Gate

1. **Should `od5h` run 20 rounds or 22 rounds (`od3g` has 22 slices)?**
   - **Run 22 rounds (`--od-rounds 22`) and score `--od-dev-rounds 17,18,19,20,21,22` (`--fp-dev-rounds 17,18,19,20,21,22`).**
   - **Evidence:**
     - `od3g` already has all 22 test-matched slices on disk (`od5g` ran 22 rounds on `od3g`).
     - `od5f` peaked at Round 18 (`57.2%` full test) and regressed at Round 20 (`53.5%`) **only because** the old held-branch (`80%` recovery) dumped 38 lengthening recovery pairs per round onto math after math hit its cap.
     - `od5g` improved from Round 19 (`56.7% / -0.45`) to Round 20 (`57.4% / +0.14`), and Dev saving kept climbing through Round 21 (`60.2%`).
     - With `--od5-held` eliminating post-cap lengthening, saving will plateau smoothly near the caps in rounds 20-22. Running 22 rounds costs only ~1.3 extra GPU hours and lets `fullpick` choose the best 2 checkpoints across the entire `r17..r22` window.
2. **Is the `fullpick` gate (`--fp-gate-save 58.5 --fp-gate-acc 1.65`, 2 full tests) right for a `> 60% / > -1.0` goal?**
   - In `fullpick` (`track.py` lines 4683-4698), `--fp-gate-save` and `--fp-gate-acc` act as a **hard abort filter** (`if not good: print("NO FULL TEST"); return`), while `--fp-save-bias 0.35 --fp-acc-bias 2.75` inside `dist(c)` ranks all passing candidates by `(0, -ps)` when `pa = c["acc"] - 2.75 > -1.0` (i.e. `c["acc"] > +1.75`).
   - Across the 12 `od5` Dev/Full pairs, `Dev - Full` accuracy bias has mean `+2.75` pts with SD `~0.45` pts (`od5g-r19` had bias `+2.32` pts: Dev `+1.87` -> Full `-0.45`). If `od5h` lands at Dev `62.0% / +1.55` (which corresponds to `~60.5% / -0.80` on the full test if bias is `+2.35`), a hard gate of `--fp-gate-acc 1.65` would abort without running any full test after already spending 15 GPU hours!
   - **Adjustment:** Keep `--fp-save-bias 0.35 --fp-acc-bias 2.75` (so `dist(c)` still treats `Dev acc >= +1.75` as the `-1.0` boundary when ranking rounds), keep `--fp-gate-save 58.5`, and lower the hard abort gate `--fp-gate-acc` from `1.65` to **`1.45`**.

---

### Question 5: Revised Per-Type Projections, Most Likely Failure Mode, and Final Verdict

#### A. Revised Per-Type Full-Test Projections for `od5h` (with honest ranges)

Comparing against `od5g-r20` (`57.4% [56.2, 58.4]` saved, `+0.14 [-0.63, +0.88]` accuracy):

| Task Family (Weight) | `od5g-r20` Full Test (Save / Acc) | `od5h` Projected Full-Test Saving [Range] | `od5h` Projected Full-Test Acc $\Delta$ [Range] | Why It Moves vs `od5g-r20` |
| :--- | :---: | :---: | :---: | :--- |
| **Math (`svamp`, 25%)** | `58.9%` / `-2.1` | **`60.0%` `[58.0, 61.5]`** | **`-2.3 pts` `[-3.2, -1.4]`** | Cap `60` + `--od5-held` stops post-cap recovery flood; keeps 8/8 reliability near `~62%`. |
| **Code (`mbpp`, 25%)** | `57.9%` / `-1.6` | **`62.5%` `[60.0, 64.5]`** | **`-1.8 pts` `[-2.8, -0.8]`** | Quota `52`, ratio `1.22`, floor `28:0.32`, cap `68`, + 2nd pair/q (with `c2` fix) adds `+4.5%` saving; `effsame.txt` shows flat accuracy down to `183t`. |
| **IFEval (`ifeval`, 25%)** | `52.5%` / `+2.2` | **`57.5%` `[55.0, 59.5]`** | **`+1.2 pts` `[+0.0, +2.3]`** | Quota `48`, ratio `1.22`, floor `25:0.30`, cap `64`, + 2nd pair/q adds `+5.0%` saving while `right_resp_cap` & `right_count` protect accuracy. |
| **Tools — Simple (11.9%)** | `56.7%` / `-1.3` | **`57.5%` `[55.5, 59.5]`** | **`-1.2 pts` `[-2.2, -0.2]`** | Cap `58` + `--od5-held` holds saving steady without post-cap lengthening. |
| **Tools — Multiple (6.0%)** | `69.0%` / `+0.6` | **`68.0%` `[66.0, 70.0]`** | **`+0.5 pts` `[-0.8, +1.8]`** | Cap `64` + `--od5-held` maintains `~68%` saving at positive accuracy. |
| **Tools — No-Fit (7.1%)** | `58.9%` / `+6.3` | **`66.5%` `[63.5, 69.5]`** | **`+6.8 pts` `[+5.0, +8.2]`** | Floor `4:0.10`, cap `76`, ratio `1.20`, quota `36`, + 2nd pair/q pushes saving to `rpL-80` level (`65.9% / +7.7`); `effsame.txt` confirms `0t` is `+5.3` pts. |
| **Tools Composite (25%)** | `60.3%` / `+1.3` | **`62.6%` `[60.5, 64.5]`** | **`+1.5 pts` `[+0.5, +2.5]`** | Weighted `400 / 200 / 240` across Simple / Multiple / No-Fit. |
| **Full Test Macro (100%)** | **`57.4%` `[56.2, 58.4]`** / **`+0.14` `[-0.63, +0.88]`** | **`60.7%` `[59.2%, 62.1%]`** | **`-0.35 pts` `[-0.95, +0.25]`** | **+3.3% more saving than `od5g-r20` with ~0.65 pt accuracy margin above `-1.0`.** |

#### B. Single Most Likely Way `od5h` Fails
- **The single most likely failure mode is falling slightly short of `60.0%` full-test saving (landing at `58.8% - 59.6%` saved at `-0.2 to +0.2` accuracy) if Code and IFEval only gain `+2-3%` saving instead of `+4.5-5.0%`.**
- Why: In `od5g-r20`, `Dev - Full` saving bias on `od3g` was `+2.0%` (`59.4%` Dev -> `57.4%` Full). If `shorten2` (`c2 = clean[idx0 + 1]`) is left unfixed, questions with `4` or `5` clean right answers produce almost no 2nd shorten pairs when `--shorten-skip-shortest` is active, starving Code and IFEval of the extra shorten pairs needed to cross `62%` on Dev (`>60%` on Full). Fixing `c2` in `repair_pairs.py` and running 22 rounds directly mitigates this failure mode.

---

#### C. Verdict: **GO WITH CHANGES** (3 small, exact changes)

1. **`code/repair_pairs.py` line 425 (fix `shorten2` starvation when `--shorten-skip-shortest` is True):**
   - Replace line 425:
     ```python
     c2, w2 = clean[idx0 + 1], clean[-2]
     ```
     with:
     ```python
     c2, w2 = clean[2 if (args.shorten_skip_shortest and len(clean) >= 6) else (0 if idx0 == 1 else 1)], clean[-2]
     ```
   - And on line 467 (prevent `--od5-held` from falling back to `1.00 < ratio <= 1.12` or bypassing `right_resp_cap` when `best is None`):
     Replace line 467:
     ```python
     for r in (clean if best is None else []):
     ```
     with:
     ```python
     for r in (clean if best is None and not (args.od5_v2 and (args.od5_held or part not in set(filter(None, re.split(r"[,+]", args.no_shorten_parts))))) else []):
     ```

2. **`code/track.py` line 5448 (prevent `--od-smooth 2` in Round 2 from averaging with Round 1's `0.0 / 0.