Here is the strict pre-launch review of the `od5h` implementation, code changes, run line, errors from `consult31`, and new data (`effsame.txt` and `od5g-r20` confidence intervals).

---

### 0. Corrections to `consult31` Errors & Integration of New Data

1. **Math 8/8 Consistency and Excess Churn (`SMARTER_DATA.md`):**
   - In `consult31` Section 5, I misquoted `od5c-r16` (54.8% saved, `-0.78` acc) as having math 8/8 consistency `75.1% -> 69.9%` and `+168 / -145` question flips. The actual full-test numbers in `SMARTER_DATA.md` are:
     - **Math Pass@8:** `96.8% -> 96.4%` (`-0.4` pts); **Math 8/8:** **`70.1% -> 61.7%`** (`-8.4` pts).
     - **All-task Pass@8:** `88.2% -> 88.1%`; **All-task 8/8:** `56.1% -> 54.3%` (`-1.8` pts; No-fit `72.5% -> 81.7%`, Multiple `56.0% -> 64.0%`, IFEval `34.0% -> 34.9%`, Simple `66.5% -> 64.2%`, Code `29.1% -> 29.1%`).
     - **Net drift (questions / pts):** Math `-28.4 q / -2.84 pts`, Code `-7.4 q / -1.95 pts`, IFEval `+2.6 q / +0.49 pts`, Simple `-4.4 q / -1.09 pts`, Multiple `0.0 q / 0.00 pts`, No-fit `+14.1 q / +5.89 pts`, All `-23.4 q / -0.85 pts`.
     - **Excess churn above the exact hypergeometric sampling-noise null ($E_0$):** **All tasks `+73.1` questions / `+2.65` pts `[+61.2, +85.2]`** (Math `+21.2 q / +2.12 pts [15.2, 27.0]`, Code `+6.4 q / +1.70 pts`, IFEval `+11.5 q / +2.12 pts`, Simple `+13.4 q / +3.34 pts`, Multiple `+12.1 q / +6.07 pts`, No-fit `+8.5 q / +3.53 pts`).
   - **Where this changes the reasoning:** Math's `-2.84` pt net accuracy drop in `od5c-r16` did **not** come from losing ability to solve problems (Pass@8 barely moved: `-0.4` pts); it came from an `-8.4` pt drop in 8/8 per-rollout reliability (`70.1% -> 61.7%`) when math was shortened to ~62% saved. That makes protecting math with the `60%` saving cap + `--od5-held` compact `be-right` pairs + `80:0.35` floor even more critical: we cannot afford to push math past `60-61%` saved to make up overall saving, because math's 8/8 reliability degrades steeply beyond 60% saved. Instead, the extra saving over `od5g-r20` (`57.4% [56.2, 58.4]`, `+0.14 [-0.63, +0.88]`) **must** come from Code, IFEval, and No-Fit—where 8/8 consistency either held flat (`29.1% -> 29.1%` on Code) or improved (`34.0% -> 34.9%` on IFEval, `72.5% -> 81.7%` on No-Fit).
2. **`r2s3-s80` vs `rpL-80`:**
   - The `65.9% saved / +7.7 pts` full-test result on No-Fit is `rpL-80` (offline DPO run) in `fulltests.json`, not `r2s3-s80` (`r2s3-s80` achieved `64.0% / +6.0` on No-Fit). Both confirm that No-Fit sustains `64-66%` saving with `+6 to +7.7` pts accuracy gain on the full test.
3. **Learning rate (`1.4e-5` vs `1e-5`) and retained flags from `od5g`:**
   - Every `od5` run (`od5c`, `od5r`, `od5f`, `od5g`) used `--rp-lr 1.4e-5`. Keeping `--rp-lr 1.4e-5` in `od5h` is completely right—dropping to `1e-5` would under-step by ~29% per round and fail to reach `>60%` saved in 20 rounds.
   - Likewise, keeping `od5g`'s exact pair flags (`--count-mode atleast`, `--base-stats data/od3-base-stats.json`, `--base-resp data/od3-base-resp.json`, `--base-think data/od3-base-think.json`, `--think-only ifeval`, `--prefer-reread`, `--count-rules`, `--resp-median 0.95 --resp-max 1.05`) and dropping the monitor early-stop (`--od-stop-from`) are **all right**:
     - `--base-stats`, `--base-resp`, and `--base-think` are required by `if args.kinds == "mix" and od5:` in `repair_pairs.py` (lines 285-307, 333, 391, 439) to compute question-level damage (`k0 - k1`), the original's thinking floor `bt`, and IFEval response-length caps `bresp`. Dropping `--base-stats` or `--base-resp` would silently disable `damaged` detection (`k0 = None`) and `right_resp_cap` (`bresp = 0`)!
     - `--count-mode atleast` was used in `od5c`, `od5f`, and `od5g` (where IFEval gained `+1.5` to `+2.2` pts on the full test) and covers `number_placeholders` and `number_highlighted_sections` in addition to word/sentence counts.
     - Dropping monitor-based early stop (`--od-stop-from`) is right because the round monitor is a single 150-question slice with `+2.5` to `+3.0` pts accuracy optimism and `±4%` slice-to-slice saving noise; stopping early on a single noisy slice risks cutting off the run 2 rounds too early, whereas `fullpick` evaluates the finished round models on the 512-question Dev set (`x8`).
4. **New `effsame.txt` Data (Equal Thinking Caps on Unseen Slices 19-20):**
   - `effsame.txt` is a major validation of both the training mechanism and the `od5h` design:
     - **Math (`svamp`):** At equal token caps, `od5c-best` is dramatically more accurate than the original (`130t`: `38.8%` vs `12.0%`; `205t`: `67.8%` vs `30.2%`; `322t`: `80.8%` vs `59.5%`). Online DPO genuinely teaches the model to reason more compactly on math, but natural math accuracy (`88.6%` at `640t`) still requires ~250t average thinking (`~60%` saved) to stay near `86%`.
     - **Code (`mbpp`) & Simple (`bfcl_simple_python`):** Even the original model achieves `50.0-51.1%` on Code at `183-817t` (vs `52.3%` at `1813t`) and `80.6-82.8%` on Simple at `47-92t` (vs `81.2%` at `150t`). Thus shortening Code to ~63% saved (`~670t`) does not cross any intrinsic capability cliff.
     - **IFEval:** Trained model is ~4 pts higher than the original at equal thinking caps (`~62%` vs `57.2-58.4%`).
     - **No-Fit (`bfcl_irrelevance`):** The original model is **most accurate with ZERO thinking (`b0 = 0t`: `87.5%`, `+5.3 pts [+0.3, +10.5]` vs natural `82.2%` at `210t`)**, whereas forcing partial thinking (`b1=53t..b9=115t`) drops accuracy to `77.8-80.9%` (`-1.3` to `-4.4` pts)! Why? Because when the model thinks at length on a no-fit prompt, it talks itself intorationalizing a tangential tool call; when it immediately recognizes no tool fits (`0-40t`), it refuses cleanly. This directly supports lowering the No-Fit floor from `15:0.20` to `4:0.10` and raising its cap to `76`.

---

### Question 1: Line-by-Line Review of `--od5-held` (`repair_pairs.py`)

Let us trace every branch of `repair_pairs.py` with `--od5-held` enabled (`args.od5_v2` is True and `args.od5_held` is True):

1. **Defect pairs taken BEFORE `n_rem` (`lines 683-687`):**
   ```python
   got5 = take(part, "right_count", int(round(0.10 * n)) if part == "ifeval" else 0)
   got5 += take(part, "loop", int(round(0.12 * n)))
   got5 += take(part, "format", int(round(0.10 * n)))
   n_rem = n - len(got5)
   ```
   - `right_count` (up to `10%` of `n` on IFEval), `loop` (up to `12%` of `n`), and `format` (up to `10%` of `n`) are taken **before** the `if args.od5_held and c_sh == 0:` branch at line 689.
   - **Effect:** None of these three defect-repair types lengthens healthy answers (`loop` replaces a truncated/looping wrong answer with the shortest clean right answer; `format` replaces an empty-response / missing-final-number / missing-tool-call wrong answer with a right answer within `0.85x-1.25x` thinking; `right_count` fixes IFEval response length with thinking `<= 1.15x`). Taking them before `n_rem` ensures held parts still get full protection against loops and format breaks.

2. **Candidate construction (`lines 439-475`) — TWO SUBTLE EDGE CASES FOUND:**
   - **Edge Case 1a (Held with no compact pair available, `lines 457-473`):**
     ```python
     elif clean and wrongs and 1 <= k1 < n_ans:
         best = None
         if args.od5_v2 and (args.od5_held or part not in set(filter(None, re.split(r"[,+]", args.no_shorten_parts)))):
             for r in clean:
                 if not right_resp_cap(r):
                     continue
                 for w in wrongs:
                     ratio = (thinking_len(r) + 1) / (thinking_len(w) + 1)
                     if 0.50 <= ratio <= 1.00:
                         ...
         for r in (clean if best is None else []):
             for w in wrongs:
                 ratio = (thinking_len(r) + 1) / (thinking_len(w) + 1)
                 if 0.90 <= ratio <= 1.12:
                     ...
     ```
     Look at line 467: `for r in (clean if best is None else []):`.
     When a part is held (`part in no_shorten_parts`) and `--od5-held` is active, it first searches for a compact `be-right` pair (`0.50 <= ratio <= 1.00`). **However, if a question has NO compact pair in `0.50 <= ratio <= 1.00` (`best is None`)**—because every right answer on that question has `ratio > 1.00` (or `< 0.50`, or failed `right_resp_cap` on IFEval)—execution **falls through** to line 467 (`0.90 <= ratio <= 1.12`)!
     - On non-IFEval parts, since `0.90 <= ratio <= 1.00` was already searched and empty, any pair found by the line 467 fallback has **`1.00 < ratio <= 1.12`** (the chosen answer thinks up to `+12%` longer than the wrong answer).
     - On IFEval, line 467 also **skips `if not right_resp_cap(r): continue`**, so an IFEval question whose right answers all exceeded `right_resp_cap` (`1.20 * bresp`) will fall through to line 467 and get selected anyway!
     - **Impact:** Unlike `recovery` (which chooses `1.08x + 10` to `1.45x + 50` and prioritizes `tw < 0.65 * bt`), a `1.00..1.12` fallback is nearly length-neutral (`+0%` to `+12%`), so on non-IFEval parts it will not cause the 20-pt saving collapse that `recovery` caused. However, if you want `--od5-held` to be strictly non-lengthening (`ratio <= 1.00`) and always respect `right_resp_cap`, line 467 should only run when `not (args.od5_v2 and (args.od5_held or part not in no_shorten_parts))`.
   - **Edge Case 1b (`if damaged` vs `elif` mutual exclusion, `lines 439 vs 455`):**
     Line 439 is `if damaged and src_rec and wrongs:` and line 455 is `elif clean and wrongs and 1 <= k1 < n_ans:`.
     If a question is marked `damaged` (`k0 - k1 >= 2/8` on `7-8/8` or `>= 1/8` on `<7/8`), it enters line 439 to look for a lengthening `recovery` pair (`1.08 * tw + 10 <= tr <= hi`). Because of `elif`, a `damaged` question **never** generates a compact `hold` candidate—even if no lengthening recovery pair fits (or if `rec_h` is capped at `20%` of `n_rem` when `dmg > -0.02`)!
     - **Impact:** On a typical 25-question math slice, ~4-6 questions are `damaged` (going into `cand[part]["recovery"]`) and ~6-9 questions are undamaged mixed (`1 <= k1 < 8` with `k1 >= k0 - 1`, going into `cand[part]["hold"]`). Because `rec_h` takes up to `20%` of `n_rem` (~4 pairs on math) from `recovery` and `hold_h` takes the rest from `hold`, having `damaged` questions in `recovery` and undamaged mixed questions in `hold` usually partitions cleanly.

3. **Held + undamaged (`dmg > -0.02`) vs Held + damaged (`dmg <= -0.02`) (`lines 689-698`):**
   ```python
   if args.od5_held and c_sh == 0:
       dmg = part_dmg.get(part, 0.0)
       if dmg > -0.02:
           n_rem = min(n_rem, int(round(0.40 * n)))
       rec_h = take(part, "recovery", int(round((0.40 if dmg <= -0.02 else 0.20) * n_rem)))
       hold_h = take(part, "hold", n_rem - len(rec_h))
       by_part[part] = got5 + rec_h + hold_h
   ```
   - **Held + undamaged (`dmg > -0.02`):** `n_rem` is capped at `0.40 * n` (e.g., on math `n=48`, `n_rem <= 19`; on simple/multiple `n=28`, `n_rem <= 11`). `rec_h` gets at most `0.20 * n_rem` (`4` pairs on math, `2` on simple/multiple), and `hold_h` gets up to `0.80 * n_rem` (`15` compact hold pairs on math, `9` on simple/multiple). Plus `got5` (`loop` + `format`) sits outside `n_rem`.
   - **Crucial fix over `od5f`/`od5g`:** When `cand[part]["hold"]` has fewer candidates than `n_rem - len(rec_h)` (say only 7 compact hold candidates exist on math), line 694 takes those 7 and **stops**—it does **not** backfill the remaining slots with `recovery` (whereas line 720 in the old code did `rec5 += take(part, "recovery", n - len(by_part[part]))`, dumping up to 38 lengthening recovery pairs onto math and erasing 18-25 pts of saving!).
   - **Held + damaged (`dmg <= -0.02`):** `n_rem` is **not** reduced (`n_rem = n - len(got5)`), and `rec_h` gets up to `0.40 * n_rem` (up to `19` recovery pairs on math + `29` compact hold pairs).
   - **Can it starve a held part of protection or still lengthen it?**
     - It **cannot starve** a held part: defect pairs (`loop`, `format`, `right_count`) are taken in full before `n_rem`; compact `be-right` pairs get 80% of `n_rem` when undamaged; and if slice accuracy drops by `> 2.0` pts (`dmg <= -0.02`), the full quota `n` is restored with 40% recovery.
     - It **will not lengthen** an undamaged held part: at most 4 recovery pairs (`0.20 * 0.40 * 48`) against up to 15 compact (`ratio <= 1.00`) hold pairs will keep math/simple/multiple within `±2-3%` of their cap instead of oscillating by `15-25%`.

4. **Is the 40% / 20% split right, and should the freed slots stay unused or go to shortening on the unheld parts?**
   - **Yes, the 40% / 20% split (and 100% / 40% when damaged) is right, and leaving the freed slots unused in `repair_pairs.py` is BETTER than explicitly filling them with extra shorten pairs from unheld parts.** Here is why from the code:
     - In `track.py` (lines 5303-5305, 5432-5433), `ptrepair` always runs `dpo.py` with `--max-steps 15 --batch 16 --epochs 2 --keep-order`.
     - In `dpo.py`, `steps = min(math.ceil(len(rows) * 2 / 16), 15)`. As long as total pairs across all 6 parts is `>= 120` (and even with math + simple + multiple all held at 40%, total pairs is `~160-190 >= 120`), **`dpo.py` still runs all 15 optimizer steps (240 pair-updates)** by wrapping around into epoch 2 of the stratified pair file!
     - Because `repair_pairs.py` interleaves parts in stratified blocks (`--block 80`), shrinking held parts to 40% of their quota in `pairs.jsonl` **automatically increases the share of the 240 gradient updates going to the unheld parts (Code, IFEval, No-Fit) by ~25-35%**, repeating their **top-ranked** pairs in the wrap-around steps rather than scraping bottom-of-the-barrel candidate pairs (`od5d` tried `--od5-v3` explicit slot reallocation by digging deeper into the shorten candidate list and crashed to `49.5% saved / -1.21 acc`).

---

### Question 2: Review of `--od-smooth 2` (`track.py`)

Let us trace lines 5446-5458 of `track.py`:
```python
if getattr(args, "od_smooth", 1) > 1:  # (19 Nov, consult31) the brake reads the mean of the last N slices
    import statistics
    prev_h = [h for h in history if k + 1 - args.od_smooth < h["round"] <= k][-(args.od_smooth - 1):]
    per_ = {p_: (statistics.fmean([v[0]] + [h["per_part"][p_][0] for h in prev_h if p_ in h.get("per_part", {})]),
                 statistics.fmean([v[1]] + [h["per_part"][p_][1] for h in prev_h if p_ in h.get("per_part", {})]))
            for p_, v in per_.items()}
```
1. **Round 1 (`k = 0`) and Round 2 (`k = 1`, first round where brake runs):**
   - In Round 1 (`k = 0`), `if args.od_caps and k > 0:` is `False`, so the brake does not run; Round 1 trains on the original's stored answers, and then lines 5471-5476 append Round 1's monitor record (`"round": 1`, `"saved": 0.0`, `"acc": 0.0`, `"per_part": {p: [0.0, 0.0], ...}`) to `history`.
   - **SUBTLE BUG IN ROUND 2 (`k = 1`):**
     In Round 2 (`k = 1`), `k + 1 - args.od_smooth < h["round"] <= k` evaluates to `0 < h["round"] <= 1`, which matches **Round 1's monitor (`h["round"] == 1`)**!
     Remember what Round 1's monitor is: Round 1 is answered by the **original model (`od3-r1-stored`)**, so `h["round"] == 1` has **`0.0%` saved and `0.00` accuracy change on every part**!
     Consequently, in Round 2 (`k = 1`), `per_` averages Round 2's actual metrics with Round 1's `(0.0, 0.0)`, **halving Round 2's measured saving and accuracy drop by 50%**!
     - Specifically, if a part lost `-3.2` pts in Round 2 (which should trigger `--od-acc-hold -2.0`), averaging with Round 1's `0.0` dilutes it to `-1.6` pts, so `--od-acc-hold -2.0` fails to fire in Round 2!
     - **Fix (1-line change on line 5448):** Exclude Round 1's `0.0/0.0` baseline monitor (`h["round"] > 1`) when smoothing trained-model monitors:
       ```python
       prev_h = [h for h in history if max(1, k + 1 - args.od_smooth) < h["round"] <= k][-(args.od_smooth - 1):]
       ```
       With `max(1, ...)`, Round 2 (`k = 1`) uses its own trained monitor (`1 slice`), and Round `k + 1 >= 3` averages Round `k + 1` with Round `k` (`2 trained slices`).
2. **A part missing from a monitor:**
   - If a part `p_` is present in `per_` (current round) but missing from `h["per_part"]` in a previous round, the list comprehension `[v[0]] + [h["per_part"][p_][0] for h in prev_h if p_ in h.get("per_part", {})]` safely skips the missing history entry and takes `fmean([v[0]]) = v[0]` without raising a `KeyError` or `StatisticsError`.
   - If `p_` is missing from the current round's slice entirely (never happens on `od3g`, where every slice has all 6 parts), `p_` is not in `per_.items()`.
3. **Interaction with `--od-unlatch` and `--od-acc-hold -2.0`:**
   - Line 5454: `held = sorted((set() if getattr(args, "od_unlatch", False) else latched) | {p_ for p_, (sv, ac) in per_.items() if sv >= caps.get(p_, 1e9) or ac <= args.od_acc_hold})`.
   - Because `--od-unlatch` is passed, `held` is recomputed purely from the 2-slice smoothed `per_` each round.
   - This works very well for **saving caps (`sv >= caps[p_]`)**: a single slice's random `+5%` saving spike won't prematurely hold a part, and once a part is genuinely at cap (`~60%`), a single slice's `-4%` dip won't unhold it unless the 2-slice mean drops below cap.
   - What about **`--od-acc-hold -2.0`**? Because `ac` is also smoothed over 2 slices, a sudden single-slice accuracy drop of `-3.5` pts after a `+0.5` pt slice averages to `-1.5` pts (above `-2.0`). And Note: inside `repair_pairs.py`, `part_dmg` (`dmg <= -0.02`) is computed on the **current slice's rollouts (`k1 - k0`)**, NOT smoothed! That is actually a great complementary design: if a held part suffers a sudden `-3%` drop on the current slice, `repair_pairs.py` immediately gives it `dmg <= -0.02` (full quota + 40% recovery).

---

### Question 3: Risk Analysis for No-Fit (`4:0.10`, cap `76`, ratio `1.20`, 2nd pair) & Code (`28:0.32`, 2nd pair)

#### A. No-Fit (`bfcl_irrelevance`: floor `4:0.10`, cap `76`, ratio `1.20`, `--shorten-second-parts ...+bfcl_irrelevance`)
1. **What is the risk that No-Fit answers start calling tools or breaking format?**
   - **Why tool-calling will NOT increase (and accuracy actually rises):**
     In BFCL irrelevance, an answer is graded `correct = True` **if and only if** the model abstains from calling any function. Every pair in `repair_pairs.py` (`shorten`, `hold`, `recovery`, `loop`) chooses `r` from `clean = [r for r in rights if not looping(r)]`, where `right(r)` requires `bool(r["correct"]) and r["closed_thinking"] and not r["truncated"]`. Thus, a tool-calling rollout can **never** be the chosen answer in any pair—it can only be the rejected answer `w`.
     Furthermore, the new `effsame.txt` data proves that on `bfcl_irrelevance`, **shorter thinking directly reduces false-positive tool calls**: at `0` thinking tokens (`b0 = 0t`), the original model achieves **`87.5%` (`+5.3 pts` over natural `82.2%`)**, whereas at `53-115t` (`b1..b9`) accuracy is `77.8-80.9%`. And on the full test, `rpL-80` at `65.9%` saved gained **`+7.7` pts** on No-Fit (`r2s3-s80` at `64.0%` saved gained `+6.0` pts; `od5g-r20` at `58.9%` saved gained `+6.3` pts).
   - **What about empty post-`</think>` response (`response_len(r) == 0`) or format break?**
     Check how `repair_pairs.py` protects format:
     - First, `right(r)` (line 102) requires `r["closed_thinking"] and not r["truncated"]` (so `</think>` is always properly closed).
     - Second, `cand[part]["format"]` (lines 366-380) checks `bad = [w for w in wrongs if response_len(w) == 0 or ...]` and explicitly trains against empty responses.
     - Third, wait: in `cand[part]["shorten"]` (lines 406-433), does `clean` filter out `response_len(r) == 0` if BFCL's evaluator grades an empty post-`</think>` refusal as `correct = True`? In `repair_pairs.py`, `clean` only checks `right(r)` and `not looping(r)`. On `bfcl_irrelevance`, a rollout that closes `</think>` and outputs a short refusal has `response_len(r) > 0`, and in `rpL-80` (`65.9%` saved on No-Fit) and `r2s3-s80` (`64.0%` saved), shortening No-Fit to ~70t thinking never broke format.
   - **Is `4:0.10` + cap `76` + ratio `1.20` + 2nd pair safe, or slightly too aggressive?**
     In `od5g-r20`, No-Fit reached `58.9%` saved (`+6.3` pts) with floor `15:0.20`, cap `70`, ratio `1.25`, and 1 pair/question, and only hit its `70%` monitor cap in 2 out of 20 rounds because 25 No-Fit questions per slice with 1 pair/question could only yield ~18-22 shorten pairs against a quota of 32. Adding the 2nd shorten pair per question (`--shorten-second-parts`) is what unlocks filling the 36-pair quota from ~20 clean questions. With 2 pairs per question + quota 36 + ratio `1.20`, No-Fit will shorten much faster than in `od5g`. Keeping the floor at `8:0.12` (or `4:0.10`) and cap at `75-76` is safe on accuracy (because `0t` thinking is `+5.3` pts on `effsame.txt`), and will take No-Fit full-test saving from `58.9%` to **`~65-68%`**.

#### B. Code (`mbpp`: floor `28:0.32`, cap `68`, ratio `1.22`, 2nd pair)
1. **What is the risk that Code breaks format or loses accuracy at `28:0.32` + 2nd pair?**
   - **What in the pair rules prevents format breaks on Code:**
     - On `mbpp`, `correct = True` requires the extracted Python code after `</think>` to **execute and pass the test assertions**. Any rollout with missing code, broken syntax, unclosed `</think>`, or truncation is graded `correct = False` and excluded from `rights` / `clean`.
     - In addition, `shorten` on `mbpp` requires:
       1. `k1 >= ceil(0.60 * 8) = 5` of 8 rollouts pass unit tests (`--shorten-min-by-part mbpp=0.60`);
       2. At least 4 passing rollouts exist above the floor (`len(clean) >= 4`, lines 407 & 414) before a 2nd pair (`clean[1]` vs `clean[-2]`) can be formed;
       3. `--shorten-skip-shortest` (line 407: `idx0 = 1 if (args.shorten_skip_shortest and len(clean) >= 4) else 0`): when `>= 4` clean answers exist, **the single shortest right answer `clean[0]` is skipped as an outlier!** So the 1st shorten pair uses `clean[1]` (the 2nd-shortest passing answer) vs `clean[-1]`, and the 2nd shorten pair (line 425: `c2, w2 = clean[idx0 + 1], clean[-2]`) uses **`clean[2]` (the 3rd-shortest passing answer)** vs `clean[-2]`!
   - **Wait—verify `clean[idx0 + 1]` vs `clean[-2]` when `len(clean) == 4` and `--shorten-skip-shortest` is True (`lines 407, 424-425`)!**
     Look closely at lines 40