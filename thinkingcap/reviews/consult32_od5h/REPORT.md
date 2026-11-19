# Consult 32 — Pre-Launch Technical Review of `od5h`

---

## 0. Corrections to Consult 31 Errors & Impact of New Data (`effsame` & `od5g-r20` CIs)

### 0.1 Resolving the Three Consult 31 Errors
1. **Math consistency & question churn (`SMARTER_DATA.md` vs. Consult 31 Section 5):**
   - *Correction:* In `SMARTER_DATA.md` (math, $n=500$), the original model has `8/8` consistency on **70.1%** of questions (`0/8` on 7.9%, `1..7/8` on 22.0%). In `od5c-best` (`r20`, math **73.1% saved / `-1.6` pts**), `8/8` consistency drops from **70.1% $\to$ 61.7%** (`-8.4` pts, not `75.1% -> 69.9%`), while `1..7/8` rises from **22.0% $\to$ 29.7%** (`+7.7` pts). At the single-sample (`s0`) level, `od5c-best` has **111** questions flip `0 -> 1` and **119** flip `1 -> 0` (230 total flips vs. 157 expected from original pass-rate variance = **+73 excess churned questions [61, 85]**, not `+168 / -145`).
   - *Where this changes our reasoning:* The `-1.6` pt math drop on `od5c-best` was driven by solid `8/8` questions becoming noisy `1..7/8` questions (`-8.4` pts on `8/8`) when math was pushed above **73% saved** (`p10` thinking fell to ~143 tokens) and then hit by aggressive lengthening/shortening oscillations. By contrast, `od5f-r18` (math **69.1% saved / `-0.7` pts**) and `od5g-r20` (math **66.7% saved / `-0.1` pts**) show that keeping math in the **68–71% saved** band avoids the `8/8 -> 1..7/8` cliff. This validates keeping math’s save cap at **72%**, floor at **`80:0.20`**, and quota at **48** (down from 60).
2. **Offline `rpL-80` vs. `r2s3-s80` on no-fit (`fulltests.json`):**
   - *Correction:* The `65.9% saved / +7.7 pts` no-fit result in `fulltests.json` (`full_r2s3_s80.json`) is **`rpL-80`** (an offline length-shortening run; overall `43.2% / -2.84`), whereas **`r2s3-s80`** (`full_r2p1s3_s80.json`) scored `45.2% saved / +3.0 pts` on no-fit (overall `54.7% / -0.94`).
   - *Where this changes our reasoning:* Within the online `od5` family, no-fit has reached **47.8% / +4.2** (`od5c-r20`), **39.5% / +4.4** (`od5f-r18`), and **44.1% / +5.5** (`od5g-r20`). We should not project 65%+ saved on no-fit from an offline length-only run; rather, as shown in Section 5 below, doubling no-fit’s pair supply (`20 -> 36` quota + 2nd pair + floor `4:0.10`) realistically moves no-fit from `44.1%` (`od5g-r20`) to **`53–58%` saved** while preserving **`+4.0 to +6.0` pts** accuracy.
3. **Learning rate across `od5c` / `od5f` / `od5g`:**
   - *Correction:* Every `od5` run (`od5`, `od5b`, `od5c`, `od5d`, `od5e`, `od5f`, `od5g`) used **`--rp-lr 1.4e-5`** (`track.py` default on line 4926), never `1e-5`.
   - *Impact:* Keep **`1.4e-5`**. Dropping to `1e-5` would cut step size by 29% without precedent and risk falling short of 60% saved in 20 rounds.

### 0.2 Integrating `effsame.txt` and `od5g-r20` Full-Test CIs
- **`od5g-r20` full-test CIs (`57.4% [56.2, 58.4]`, `+0.14 [-0.63, +0.88]`):** `od5g-r20` has **+1.14 pts** of accuracy headroom above the `-1.0` bar (`-0.63` lower 95% CI bound), proving that accuracy is well-protected and our remaining gap is purely **`+2.6` to `+3.6` pts of saving** on the under-compressed parts (`code` at 38.6%, `ifeval` at 57.6%, `no_fit` at 44.1%) plus stopping `math` (`66.7%`) and `multiple` (`66.7%`) from losing saving during held rounds.
- **`effsame.txt` (equal thinking token caps):**
  - **Math:** Trained model (`od5c-r20`) is vastly more accurate at equal thinking caps (`@130t`: **38.8% vs. 12.0%** original; `@205t`: **67.8% vs. 30.2%**; `@322t`: **80.8% vs. 59.5%**), confirming genuine reasoning density compression. However, accuracy still drops steeply below 205 tokens (`67.8% -> 38.8%` at `130t`), confirming the `80:0.20` floor.
  - **Code:** Trained and original models have identical accuracy at every cap (`@443t`: 18.8% vs. 18.6%; `@644t`: 33.8% vs. 35.1%; `@902t`: 46.8% vs. 46.6%), with a steep accuracy-vs-length slope. Code cannot be crudely truncated below 450 tokens; saving must come from trimming redundant verification on solved problems while respecting a high relative floor (`28:0.32`).
  - **IFEval:** Accuracy is flat across all caps (`@103t`: **62.0%** trained vs. **58.3%** original; `@456t`: **61.4%** vs. **57.2%**). Thinking beyond ~100 tokens adds zero accuracy on IFEval, supporting aggressive shortening (`16:0.15`, ratio `1.22`, cap `72`, 2nd pair).
  - **No-fit:** The original model is **most accurate with NO thinking at all** (`0t`: **75.0%** vs. full thinking **69.7%**, **+5.3 pts `[+0.3, +10.5]`**), and the trained model hits **74.8%** at `27t`. Thinking on no-fit actively causes false-positive tool calls; compressing no-fit down toward `4:0.10` improves both saving and accuracy.

---

## 1. Line-by-Line Audit of `--od5-held` (`repair_pairs.py`)

### 1.1 Trace Through Every Branch (`lines 439–475` and `lines 681–735`)
1. **Candidate construction (`lines 439–475`):**
   - **Damaged questions (`line 439: if damaged and src_rec and wrongs:`):** Enters the `recovery` candidate builder (`1.08 * tw + 10 <= thinking_len(r) <= min(1.05 * bt, 1.45 * tw + 50)`). Because `line 455` is an `elif clean and wrongs and 1 <= k1 < n_ans:`, a `damaged` question (`drop >= 2/8` on original `7–8/8`, or `drop >= 1/8` on original `<7/8`) only ever populates `cand[part]["recovery"]`, never `cand[part]["hold"]`.
   - **Undamaged mixed questions (`line 455: elif clean and wrongs and 1 <= k1 < n_ans:`):**
     - `line 457`: `if args.od5_v2 and (args.od5_held or part not in set(filter(None, re.split(r"[,+]", args.no_shorten_parts)))):`
       With `--od5-held`, a held part now enters the **compact `be-right` search** (`0.50 <= ratio <= 1.00`, enforcing `right_resp_cap` on IFEval) and ranks candidates by `(0 if REREAD else 1, thinking_len(r), abs(ratio - 0.75))`. In `od5f`/`od5g`, held parts skipped this block (`best` stayed `None`) and went exclusively to the lengthening `0.90 <= ratio <= 1.12` loop.
     - **Held with no compact pair available (`lines 467–473: for r in (clean if best is None else []):`):**
       If an undamaged mixed question has *no* pair in `0.50 <= ratio <= 1.00` (`best is None`), execution falls through to the `0.90 <= ratio <= 1.12` loop. Because `[0.90, 1.00]` was already searched (except on IFEval if rejected by `right_resp_cap`—and note `line 470` does not check `right_resp_cap`), any fallback candidate found on non-IFEval has **`1.00 < ratio <= 1.12`** (chosen is 0–12% longer than rejected), and on IFEval it bypasses `right_resp_cap`.
       - *Severity & fix:* This only occurs on questions where every right trace is longer than every wrong trace (or `< 0.50x`), and `+0% to +12%` is small compared to `recovery` (`1.08x + 10` to `1.45x + 50`). However, to make `--od5-held` strictly non-lengthening and respect `right_resp_cap` on IFEval, we can either block the fallback when `args.od5_held` is set on a held part or leave it as a mild tie-breaker. (See Section 5 for exact recommendation.)
2. **Defect pairs taken before `n_rem` (`lines 683–687`):**
   - `got5` takes `right_count` (up to `10%` of `n` on IFEval), `loop` (up to `12%` of `n`), and `format` (up to `10%` of `n`). Then `n_rem = n - len(got5)`.
   - *Can `got5` starve a held part of protection or lengthen it?* **No.** First, `loop` pairs replace runaway traces (`> 1.35 * bt`) with clean traces, so they *shorten* while fixing loops; `format` and `right_count` fix broken tags/counts using clean right answers. Second, at `line 692`, `n_rem = min(n_rem, int(round(0.40 * n)))` is applied **after** `got5` is taken. Because `len(got5) <= 0.32 * n`, `n - len(got5)` is always at least `0.68 * n >= 0.40 * n`, so `got5` never reduces `n_rem` for an undamaged held part (it sits alongside `n_rem`, giving `len(got5) + 0.40 * n` total pairs).
3. **Held + undamaged (`dmg > -0.02`, `lines 691–695`):**
   - `n_rem = min(n_rem, int(round(0.40 * n)))` (`19` pairs on math `n=48`, `11` pairs on simple/multiple `n=28`).
   - `rec_h = take(part, "recovery", int(round(0.20 * n_rem)))` (at most **`4` recovery pairs** on math, **`2`** on simple/multiple).
   - `hold_h = take(part, "hold", n_rem - len(rec_h))` (up to **`15` compact `be-right` pairs** on math, **`9`** on simple/multiple).
   - **Crucial fix vs. `od5f`/`od5g`:** In `od5f`/`od5g`, when a part was held, `c_sh = 0` caused `line 720` (`rec5 += take(part, "recovery", n_rem - len(sh5) - len(rec5) - len(hold5))`) to dump **80%+ of the full quota (`~38` pairs on math) into lengthening `recovery` pairs**, causing math saving to crash by 15–25 pts in 2 rounds (`74% -> 49%` in `od5f` rounds 12–14). In `--od5-held`, `line 694` **never backfills unfilled `hold_h` slots with `recovery`**. If `cand[part]["hold"]` only has 10 pairs, `hold_h` takes 10 and stops.
4. **Held + damaged (`dmg <= -0.02`, `lines 691–695`):**
   - `n_rem` keeps the full `n - len(got5)` quota (`~45` on math, `~26` on simple/multiple).
   - `rec_h` takes up to `0.40 * n_rem` (`~18` recovery pairs on math, `~10` on simple/multiple), and `hold_h` takes the remaining `0.60 * n_rem` (`~27` compact `be-right` pairs on math).
   - *Can this starve a damaged held part of protection?* **No.** Up to 18 recovery pairs + 27 compact `be-right` pairs + 3 defect pairs = 48 accuracy-enforcing pairs in a single round, which is full-quota protection without the 38-recovery-pair lengthening flood.

### 1.2 Is the 40% / 20% Split Right, and Should Freed Slots Go to Unheld Parts?
- **The 40% / 20% split (and 100% / 40% when damaged) is right.** On an undamaged held slice, `~4` recovery pairs (`1.08x..1.45x`) balanced against `~15` compact `be-right` pairs (`0.50x..1.00x`, median `~0.75x`) yields a net length change near zero or slightly negative, holding saving steady at the cap while continuously reinforcing accuracy on mixed questions.
- **Do NOT explicitly reallocate freed pair slots to unheld parts (`od5_v3` style) — `dpo.py` already reallocates gradient updates automatically and with higher pair quality:**
  - In `track.py` (`line 4928`), `ptrepair` runs `dpo.py` with `--epochs 2 --batch 16 --max-steps 15 --keep-order` (`240` pair updates per round).
  - `repair_pairs.py` outputs a **stratified** pair file interleaving the 6 parts (`lines 752–758`). As long as the pair file has $\ge 120$ pairs (and even if math, simple, and multiple are all held at 40% simultaneously, total pairs $\approx 19 + 52 + 48 + 36 + 11 + 11 = 177 \ge 120$), `dpo.py` (`lines 124–128`) **still executes all 15 steps (240 pair updates)** by wrapping the first `240 - 177 = 63` pairs into epoch 2.
  - Because the held parts occupy fewer rows in the stratified file (`41/177 = 23.2%` vs. `104/240 = 43.3%`), the **unheld parts (`code`, `ifeval`, `no_fit`) automatically rise from 56.7% to 76.8% of the 240 gradient updates** across the 15 steps—and they do so by repeating their **top-ranked** shorten pairs in epoch 2 rather than scraping low-margin pairs deep in `cand[part]["shorten"]` (which is why `--od5-v3` in `od5d` degraded to `49.5% / -1.21`). Leaving unused slots empty in `repair_pairs.py` while keeping `15` steps in `dpo.py` is strictly better.

---

## 2. Review of `--od-smooth 2` (`track.py` lines 5368–5394)

1. **First rounds (no history / 1 monitor):**
   - **Round 1:** `r == 1`, so `if r > 1 and ...` (`line 5364`) does not run; no parts are held in Round 1.
   - **Round 2 (`len(mon_hist) == 1`):** `mon_hist[-2:]` has 1 element (`[r1]`). Both `cur` and the smoothed `da` equal `r1`’s delta. If any part drops $\le -2.0$ pts in Round 1 (which trains on 300 pairs), the brake fires immediately in Round 2 without waiting for a second bad round.
2. **Missing part in a monitor (`lines 5376–5377`):**
   - `vals = [h[p].get("da") for h in mon_hist[-n_sm:] if p in h and h[p].get("da") is not None]`. If a part were ever missing in one monitor, `vals` averages the remaining valid monitors in the window without throwing a `KeyError` or dividing by zero (`if not vals: continue`). Every `od3g` slice (`s01..s22`) has all 6 parts (`50/50/50/30/30/30`).
3. **Interaction with `--od-unlatch`, `--od-acc-hold -2.0`, save caps, and `--damage-from`:**
   - **Save cap stays instantaneous (`line 5379`):** `sv = cur.get("sv")` reads `mon_hist[-1]` (the latest slice only), *not* the 2-round mean. As soon as a part reaches its save cap (`math=72`, `simple=76`, `multiple=74`, `ifeval=72`, `no_fit=76`), it is held immediately on the next round.
   - **Accuracy brake uses the 2-round mean (`lines 5377–5382`):** On a 50-question slice (1 question = `2.0` pts) or 30-question tool slice (1 question = `3.3` pts), a single-slice sampling dip of `-2.5` pts after a `+1.0` pt slice averages to `-0.75 > -2.0` and no longer triggers a false hold. A sustained drop (`-2.0` and `-2.2`) or a severe single-round drop (`-4.5` after `0.0`) averages $\le -2.0$ and triggers the hold.
   - **Interaction between `--od-unlatch` and `--od5-held` (`damage` check in `repair_pairs.py`):** Notice that `repair_pairs.py` reads `--damage-from` from the **latest** monitor file (`mon_r{r-1}.json`, `line 5266`). Suppose math drops `-3.5` pts in `r-2` (held as damaged in `r-1`, receiving 40% recovery / 60% compact hold) and rebounds to `-0.8` pts in `r-1`. Entering round `r`, the 2-round mean in `track.py` is `(-3.5 + -0.8) / 2 = -2.15 <= -2.0`, so math remains held for one extra round (`r`), **but** `repair_pairs.py` reads `dmg = -0.8 > -0.02` from `mon_r{r-1}.json` and automatically switches math to the **undamaged hold branch** (`40%` quota, `20%` recovery / `80%` compact hold) before unlatching in `r+1`! This provides a smooth two-stage recovery (heavy repair on the crash round $\to$ compact stabilization on the rebound round $\to$ unlatch).

---

## 3. Audit of the 4 Queue Flags You Did Not Take

1. **`--rp-lr 1.4e-5` (kept) vs. `1e-5`:** **Keep `1.4e-5`.** Every `od5` run used `1.4e-5`.
2. **`--count-mode atleast` (kept) vs. `length`:** **Keep `atleast`.** In `repair_pairs.py` (`line 206`), `--count-mode length` activates `if mode in ("exact", "length") and rel == "at_least" and actual > n + 2: return True`, which falsely rejects valid IFEval responses that satisfy `"at least N words/sentences/paragraphs"` by exceeding $N+2$! All `od5c`/`od5f`/`od5g` runs used `atleast`.
3. **Keeping `--base-stats`, `--base-resp`, `--base-think`, `--think-only`, `--prefer-reread`, `--count-rules`, `--response`:** **Keep all of them.** In `repair_pairs.py` (`lines 324–333`, `369`), `--base-think` and `--base-resp` supply the original model’s per-question thinking length `bt` and response length `base_rlen`. Dropping `--base-think` causes `bt` to fall back to `max(t_lens)` of the current round’s 8 already-shortened rollouts (`line 333`), which shrinks every round and breaks the shorten floor `min_r * bt`, recovery cap `1.05 * bt`, and loop detector `1.35 * bt`.
4. **Dropping monitor-based early stop (`--od-stop-from 18 ...`):** **Keep it dropped.** The 240-question monitor is `+2.5 to +3.0` pts optimistic on accuracy and has `±2.5` pt slice-to-slice saving noise; stopping early on a monitor spike risks cutting off the true Dev/Test peak.

---

## 4. Risk Analysis: No-Fit (`4:0.10`, ratio `1.20`, cap `76`, 2nd pair) & Code (`28:0.32`, ratio `1.04`, quota `52`, 2nd pair)

### 4.1 No-Fit: Risk of Tool Calls or Format Breakage
- **Why no-fit does not start calling tools when shortened:** On BFCL `no_fit`, calling a tool gives score `s = 0.0`; abstaining gives `s = 1.0`. As `effsame.txt` proves (`0t`: **75.0%** vs. full thinking **69.7%**, `+5.3` pts), *longer* thinking is what causes the model to talk itself into hallucinating a tool call.
- **What in `repair_pairs.py` prevents format breakage at `4:0.10` + 2nd pair:**
  1. `clean` filter (`line 366`): Every chosen trace `r` must have `r["s"] >= 1.0` (verified zero tool calls), `not has_format_defect(r)` (`lines 130–145`: requires intact `<think>...</think>` tags, no duplicate `<think>` tags, response body $\ge 2$ chars, and `finish_reason != "length"`), and `not has_loop(r, bt)`.
  2. Floor `4:0.10`: Because original `no_fit` thinking averages ~390 tokens (`p10 ≈ 180t`), `max(4, int(0.10 * bt))` sets the actual floor to **18–39 tokens** on normal questions (only allowing 4 tokens if the original model already thought $< 40$ tokens). A 20–35 token `<think>` block is a clean 1-sentence relevance check.
  3. Second shorten pair (`lines 426–438`): Only emits a second pair when a question has $\ge 2$ distinct clean (`s >= 1.0`, format-valid) traces and $\ge 2$ distinct longer anchors (`tw2 >= max(tl + 15, tl / 0.80)`).

### 4.2 Code: Why `28:0.32` + Ratio `1.04` + 2nd Pair Works (and Fixes a Hidden Bug in Quota `52`)
- **Why code is protected against the `od5c`/`od5f` accuracy drop:** In `od5c` (`-3.1` pts on code) and `od5f` (`-2.8` pts on code), the code floor was `24:0.22`, allowing 1,200-token traces to be cut to 260 tokens (where `effsame.txt` shows code accuracy collapses). Raising the code floor to **`28:0.32`** in `od5g` recovered code accuracy by **+2.3 pts** (to **`-0.5` pts** on full test, and `+2.2` pts on `s20`) with zero held rounds. Keeping `28:0.32` ensures no shorten pair on code ever chooses a trace shorter than 32% of that problem’s original thinking length.
- **Why `code` MUST stay in `--od5-second-parts` when its quota is `52`:**
  In `od5g` (where code had 1 pair per question out of 50 slice questions), `cand["code"]["shorten"]` only had **32–45 candidates per round** (mean `37.6`). Look at `repair_pairs.py` lines `712–720`: on an unheld part with quota `n = 52` (`n_rem ≈ 50`, `n_sh5 = 25`), if `hold5` (`cand["code"]["hold"]`, which averages only `8–12` candidates on code) and `sh5` (`25` pairs) do not fill `n_rem = 50`, **`line 720` dumps all remaining unfilled slots into `recovery` (lengthening) pairs!** Putting `code` in `--od5-second-parts` expands `cand["code"]["shorten"]` from ~38 to ~60 candidates, ensuring `sh5` gets high-margin shorten pairs from the top of the pool.

---

## 5. 20 vs. 22 Rounds, Dev Scoring Window, and `fullpick` Gate

### 5.1 Should `od5h` Run 20 Rounds or 22 Rounds?
- **Run 20 rounds (do NOT extend to 22 rounds).**
  1. Look at the slice difficulty in `od3g` (`s01..s22`): in `od5c` (which ran 22 rounds), saving peaked at **`r20` (62.8% Dev / 61.3% Test)** and **regressed in `r21–r22`** (`r21` monitor fell to `58.9% / +1.4`, `r22` Dev fell to `61.4%`). In `od5f`, Dev peaked at **`r18` (59.2% Dev)**. In `od5g`, Dev peaked at **`r20` (58.0% Dev)**. No `od5` run has ever improved in rounds 21–22; late rounds after `r20` over-compress easy questions and trigger holds without adding net saving.
  2. Running 20 rounds saves ~1.3 GPU hours of rollout/training and avoids testing noisy `r21–r22` checkpoints in `fullpick`.

### 5.2 Which Dev Rounds Should `fullpick` Score, and Is the Gate Right?
- **Dev rounds to score (`--dev-rounds`):** Change `16,17,18,19,20` to **`15,16,17,18,19,20`** (adding `r15` costs only ~6 minutes for 600 $\times$ 1 rollouts and gives a 6-checkpoint window covering both early peaks like `od5f-r16/r18` and late peaks like `od5c-r20` / `od5g-r20`).
- **Evaluating the `fullpick` gate (`--save- floor 58.5 --acc-floor -1.0 --save-bias 0.35 --acc-bias 2.75 --max-full 2`):**
  - Under the flat calibration (`save_bias = 0.35`, `acc_bias = 2.75`, slopes `0`), a checkpoint passes the gate iff:
    $$\text{Dev save} \ge 58.5 + 0.35 = \mathbf{58.85\%} \quad \text{and} \quad \text{Dev }\Delta\text{acc} \ge -1.0 + 2.75 = \mathbf{+1.75\text{ pts}}.$$
  - Let’s check this against historical runs on `dev600`:
    - `od5c-r20`: Dev `62.8% / +2.67` $\to$ Test `61.3% / -1.09` (passed gate).
    - `od5f-r18`: Dev `59.2% / +1.50` $\to$ Test `58.7% / -1.21` (rejected by `+1.75` acc gate, which is *correct* because Test was `-1.21 < -1.0`!).
    - `od5g-r20`: Dev `58.0% / +2.50` $\to$ Test `57.4% / +0.14` (rejected by `58.85%` save gate, which is *correct* because Test was `57.4% < 60.0%`!).
  - **One small adjustment to `--save-floor` (`58.5 -> 58.3`) and `--acc-bias` (`2.75 -> 2.65`):**
    Across the 7 paired `(Dev, Test)` checkpoints in `od5c/f/g`, the actual `Dev - Test` saving gap had a mean of **`+0.35` pts** (and in `od5c-r16`, Test was actually `+0.2` pts *higher* than Dev: `58.5%` Dev $\to$ `58.7%` Test), while `od5g-r20` had a `Dev - Test` accuracy gap of **`+2.36` pts** (`+2.50` Dev $\to$ `+0.14` Test). If `od5h` hits **`59.8%` Test / `-0.7` Test** on one checkpoint (`Dev ≈ 59.5% / +1.65`) and **`60.4%` Test / `-0.8` Test** on another (`Dev ≈ 60.7% / +1.65`), a rigid `+1.75` Dev accuracy floor (`-1.0 + 2.75`) would block `fullpick` from running the 2 full tests and default to the fallback! Lowering `--acc-bias` from `2.75` to **`2.55`** (so the Dev accuracy gate is **`+1.55` pts**, corresponding to expected Test **`-1.0` pts** given the `+2.36` to `+2.71` observed gap in `od5f`/`od5g`) and `--save-floor` to **`58.3`** (Dev gate **`58.65%`**) ensures `fullpick` always tests the top 2 candidates whenever `od5h` is within striking distance of `60% / -1.0`, while still ranking candidates by `pred_save - 20 * max(0, -0.5 - pred_acc)`.

---

## 6. Revised Per-Type Projections, Most Likely Failure Mode, and Final Decision

### 6.1 Revised Per-Type Projections for `od5h` (Full Test, 2,759 $\times$ 8)

| Part (Test $N$, Weight) | `od5c-r20` (Test) | `od5f-r18` (Test) | `od5g-r2