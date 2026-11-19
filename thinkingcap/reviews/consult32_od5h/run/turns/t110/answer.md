# Consult 32 — Pre-Launch Technical Review of `od5h`

---

## 0. Corrections to Consult 31 Errors & Impact of New Data (`effsame` & `od5g-r20` CIs)

### 0.1 Resolving the Three Consult 31 Errors
1. **Math consistency & question churn (`SMARTER_DATA.md` vs. Consult 31 Section 5):**
   - *Correction:* In `SMARTER_DATA.md`, on math (`n=500`), the original model has `8/8` consistency on **70.1%** of questions (`0/8` on 7.9%, `1..7/8` on 22.0%), and `od5c-best` (`r20`, math 73.1% saved / `-1.6` pts) drops `8/8` consistency from **70.1% $\to$ 61.7%** (`-8.4` pts, not `75.1% -> 69.9%`), while `1..7/8` rises from **22.0% $\to$ 29.7%** (`+7.7` pts). At the single-sample (`s0`) level, `od5c-best` has **111** questions flip `0 -> 1` and **119** flip `1 -> 0` (230 total flips vs. 157 expected from original pass-rate variance = **+73 excess churned questions [61, 85]**, not `+168 / -145`).
   - *Where this matters for `od5h`:* The `-1.6` pt math drop on `od5c-best` was driven almost entirely by solid `8/8` questions becoming noisy `1..7/8` questions (`-8.4` pts on `8/8`) when math was pushed to **73.1% saved** (where `p10` thinking falls to ~143 tokens) and then oscillated under old recovery bursts. In `od5f-r18` (math **69.1% saved, `-0.7` pts**) and `od5g-r20` (math **66.7% saved, `-0.1` pts**), holding math around **67–70% saved** avoids the steep `8/8 -> 1..7/8` cliff. This reinforces keeping math’s cap at **72%** and quota at **48** (not pushing math past 72%).
2. **Offline `rpL-80` vs. `r2s3-s80` on no-fit (`fulltests.json`):**
   - *Correction:* The `65.9% saved / +7.7 pts` no-fit result in `fulltests.json` (`full_r2s3_s80.json`) belongs to **`rpL-80`** (offline length-shortening run, overall `43.2% / -2.84`), whereas **`r2s3-s80`** is in `full_r2p1s3_s80.json` (`45.2% saved / +3.0 pts` on no-fit, overall `54.7% / -0.94`).
   - *Where this matters for `od5h`:* Online DPO runs in the `od5` family achieved **47.8% / +4.2** (`od5c-r20`), **39.5% / +4.4** (`od5f-r18`), and **44.1% / +5.5** (`od5g-r20`) on no-fit. However, `rpL-80` (`65.9% / +7.7`) and `effsame.txt` still confirm the underlying mechanism: shortening no-fit thinking all the way down to 65%+ saved improves no-fit accuracy rather than hurting it. What limited `od5f`/`od5g` on no-fit was **pair starvation** (`16–20` pairs/round with `no_fit=1.35`, yielding only `5–14` shorten pairs/round in late rounds), not an accuracy wall.
3. **Learning rate across `od5c` / `od5f` / `od5g`:**
   - *Correction:* Every `od5` run (`od5`, `od5b`, `od5c`, `od5d`, `od5e`, `od5f`, `od5g`) used **`--rp-lr 1.4e-5`** (`track.py` default on line 4926 is `1.4e-5`), never `1e-5`.
   - *Where this matters for `od5h`:* Lowering lr to `1e-5` would be an untested 29% reduction in step size that would slow compression across 20 rounds. **Keep `--rp-lr 1.4e-5`**.

### 0.2 What `effsame.txt` and `od5g-r20` CIs Tell Us
- **Math (`effsame.txt`):** At equal thinking budgets, the trained model (`od5c-r20`) is dramatically more accurate than the original model (`@130t`: **38.8% vs. 12.0%**, `+26.9` pts; `@205t`: **67.8% vs. 30.2%**, `+37.6` pts; `@322t`: **80.8% vs. 59.5%**, `+21.3` pts), and still shows a steep accuracy slope with tokens (`38.8% -> 67.8% -> 80.8% -> 87.3% -> 90.0%`). Therefore, math has a real token floor below ~200 tokens (`@130t` drops to 38.8%), confirming `math=80:0.20` and a `72%` save cap.
- **Code (`effsame.txt`):** Trained and original accuracy are identical at every cap (`@443t`: 18.8% vs. 18.6%; `@644t`: 33.8% vs. 35.1%; `@902t`: 46.8% vs. 46.6%), and accuracy climbs steeply with thinking length (`18.8% -> 33.8% -> 46.8% -> 53.7% -> 57.6%`). Code cannot be crudely truncated at 400 tokens without losing accuracy; its compression must come from selectively shortening bloated/redundant traces on solved problems while respecting a safe floor (`28:0.32`).
- **IFEval (`effsame.txt`):** Accuracy is completely flat wrt thinking budget for both original (`58.3%` at `103t` vs. `57.2%` at `456t`) and trained (`62.0%` at `103t` vs. `61.4%` at `456t`). Extra thinking above 100 tokens buys zero accuracy on IFEval, fully supporting `ifeval=16:0.15`, ratio `1.22`, cap `72`, and 2 shorten pairs per question.
- **No-fit (`effsame.txt`):** The original model is **most accurate with NO thinking at all** (`0t`: **75.0%** vs. full thinking **69.7%**, `+5.3` pts `[+0.3, +10.5]`), and at `27t` the trained model hits **74.8%** (`+5.1` pts over full-thinking original). Thinking on no-fit actively causes false-positive tool calls; compressing no-fit toward `4:0.10` improves both saving and accuracy.

---

## 1. Line-by-Line Review of `--od5-held` (`repair_pairs.py`)

### 1.1 Trace Through Every Branch in `repair_pairs.py`
1. **Candidate construction (`lines 439–475`):**
   - **Damaged questions (`line 439: if damaged and src_rec and wrongs:`):** Enters the `recovery` candidate builder (`1.08 * tw + 10 <= thinking_len(r) <= min(1.05 * bt, 1.45 * tw + 50)`). Note that because `line 455` is an `elif clean and wrongs and 1 <= k1 < n_ans:`, any question flagged as `damaged` (`drop >= 2/8` from original `7–8/8`, or `drop >= 1/8` from original `<7/8`) only ever produces a `"recovery"` candidate, never a `"hold"` candidate.
   - **Undamaged mixed questions (`line 455: elif clean and wrongs and 1 <= k1 < n_ans:`):**
     - `line 457`: `if args.od5_v2 and (args.od5_held or part not in set(filter(None, re.split(r"[,+]", args.no_shorten_parts)))):`
       With `--od5-held`, a held part now enters the **compact `be-right` loop** (`0.50 <= ratio <= 1.00`, enforcing `right_resp_cap` on IFEval), picking the candidate with `(0 if REREAD else 1, thinking_len(r), abs(ratio - 0.75))`. In `od5f`/`od5g`, held parts skipped this loop (`best` stayed `None`) and went straight to the lengthening fallback (`0.90 <= ratio <= 1.12`).
     - **Fallback when no compact pair exists (`lines 467–473: for r in (clean if best is None else []):`):**
       If an undamaged mixed question has **no** pair with `0.50 <= ratio <= 1.00` (either all clean answers are `< 0.50 * tw` or all are `> 1.00 * tw`), `best` is `None`, so execution falls through to the `0.90 <= ratio <= 1.12` loop. Because `[0.90, 1.00]` was already searched (except on IFEval if `right_resp_cap` rejected it—and note `line 470` does not check `right_resp_cap`), any fallback candidate found on non-IFEval has `1.00 < ratio <= 1.12` (chosen is 0–12% longer than rejected). While much milder than `recovery` (`1.08x + 10` to `1.45x + 50`), it is still slightly lengthening (`ratio > 1.00`) and bypasses `right_resp_cap` on IFEval.
2. **Quota and branch allocation when held (`lines 681–698`):**
   - **Defect pairs taken before `n_rem` (`lines 683–687`):**
     `got5` takes `right_count` (10% of `n` on IFEval), `loop` (12% of `n`), and `format` (10% of `n`). Then `n_rem = n - len(got5)`.
     - *Can `got5` starve a held part or lengthen it?* **No.** First, `right_count`, `loop`, and `format` require a clean chosen answer vs. a defective rejected answer; for `loop`, the rejected answer is a runaway/long trace (`> 1.35 * bt`), so `loop` pairs actively **shorten** while fixing defects. Second, in `lines 691–692`, `n_rem = min(n_rem, int(round(0.40 * n)))` is applied **after** `got5` is subtracted. Thus `got5` never reduces `n_rem` unless `len(got5) > 0.60 * n`, which is impossible since `0.10 + 0.12 + 0.10 = 0.32 * n` max. (Total pairs for an undamaged held part is `len(got5) + min(n - len(got5), 0.40 * n)`, typically `2–5` defect pairs + `0.40 * n` hold/recovery pairs.)
   - **Held + undamaged (`dmg > -0.02`, `lines 691–695`):**
     `n_rem = min(n_rem, int(round(0.40 * n)))`, `rec_h = take(part, "recovery", int(round(0.20 * n_rem)))`, `hold_h = take(part, "hold", n_rem - len(rec_h))`.
     - For math (`n=48`): `n_rem = 19`, `rec_h <= 4` pairs (`8.3%` of `n`), `hold_h <= 15` compact pairs.
     - For simple/multiple (`n=28`): `n_rem = 11`, `rec_h <= 2` pairs (`7.1%` of `n`), `hold_h <= 9` compact pairs.
     - **Crucial difference vs. `od5f`/`od5g`:** In `od5f`/`od5g`, when a part was held, `c_sh = 0` caused `rec5` at `line 720` (`rec5 += take(part, "recovery", n_rem - len(sh5) - len(rec5) - len(hold5))`) to take **80% of `n_rem`** (`~38` lengthening recovery pairs on math in a single round!), which caused math saving to crash by `15–25` pts whenever the brake fired. In `--od5-held`, `line 694` **never backfills missing `hold_h` slots with `recovery`**. If `cand[part]["hold"]` only has 8 candidates, `hold_h` takes 8 and stops.
   - **Held + damaged (`dmg <= -0.02`, `lines 691–695`):**
     `n_rem` keeps the full `n - len(got5)` quota (`~45` on math, `~26` on simple/multiple), `rec_h` takes up to `0.40 * n_rem` (`~18` recovery pairs on math, `~10` on simple/multiple), and `hold_h` takes the remaining `0.60 * n_rem` compact `be-right` pairs. This provides strong accuracy repair when a slice genuinely lost `> 2.0` pts without the 38-pair recovery overshoot.

### 1.2 Is the 40% / 20% Split Right, and Should Freed Slots Go to Unheld Parts?
- **The 40% / 20% split is right.** At `4` max recovery pairs + `15` compact (`ratio <= 1.00`) hold pairs on an undamaged held math slice, net length pressure is slightly negative to neutral (eliminating the `74% -> 49%` math saving oscillation seen in `od5f` rounds 12–14), while `40%` of full quota (`~18` recovery pairs + `~27` compact hold pairs) when `dmg <= -0.02` is enough to recover a damaged slice within 1–2 rounds.
- **Do NOT explicitly reallocate freed slots (`extra = take(p_, "shorten", ...)` as in `od5_v3`) — the current code already reallocates gradient steps automatically and more safely:**
  - Look at how `ptrepair` trains in `dpo.py` (`lines 124–128`): `--rp-epochs 2 --batch 16 --max-steps 15 --keep-order`.
  - `repair_pairs.py` outputs a **stratified** pair list (`lines 752–758`) interleaving the 6 parts. Even if math (`48 -> 19`), simple (`28 -> 11`), and multiple (`28 -> 11`) were all held simultaneously, total pairs = `19 + 52 + 48 + 36 + 11 + 11 = 177` pairs. With `epochs=2`, `177 * 2 = 354 >= 240`, so `dpo.py` **still executes the full 15 steps = 240 pair-updates** by wrapping 63 pairs into epoch 2.
  - Because the held parts take fewer slots in the stratified file (`41/177 = 23%` instead of `104/240 = 43%`), the **unheld parts (code, IFEval, no-fit) automatically rise from 56.7% to 76.8% of the 240 gradient updates** across the 15 steps—repeating their **top-ranked** pairs in epoch 2 rather than scraping bottom-of-the-barrel shorten candidates (which is why `od5d` with `--od5-v3` crashed to `49.5% / -1.21`).
- **One optional 1-line cleanup in `repair_pairs.py` (`line 467`):** When `args.od5_held` is active and `part` is in `no_shorten_parts`, skipping the `0.90 <= ratio <= 1.12` fallback (`for r in (clean if best is None and not (args.od5_held and part in ...) else []):`) guarantees strictly zero lengthening (`ratio <= 1.00`) on held `hold_h` pairs and never bypasses `right_resp_cap` on IFEval. Even as written, the impact of `line 467` is very small (`<= 12%` on the rare mixed question where all clean traces are `> 1.00 * tw`), so this is minor.

---

## 2. Review of `--od-smooth 2` (`track.py` lines 5368–5394)

Tracing `track.py` lines `5368–5394` with `--od-smooth 2`, `--od-acc-hold -2.0`, `--od-unlatch`, and `--od-save-cap math=72,simple=76,multiple=74,ifeval=72,no_fit=76`:

1. **Round 1 (before any training):** `r == 1`, so the `if r > 1 and ...` block (`line 5364`) is skipped. Nothing is held in Round 1.
2. **Round 2 (1 monitor in history: `[r1]`):**
   - `mon_hist[-2:]` has length 1 (`[r1]`). Both `cur` and the 1-element mean `da` equal `r1`'s delta.
   - This is desirable: Round 1 uses `300` pairs (`--od5-r1-quotas`), and if any part drops `<= -2.0` pts right out of the gate, Round 2 brakes immediately without waiting.
3. **Round $r \ge 3$ (2 monitors in history: `[r-2, r-1]`):**
   - **Accuracy brake (`da <= -2.0`):** `da` is the 2-round mean `(mon[r-2].da + mon[r-1].da) / 2.0`. A single-round noisy dip of `-2.5` pts on a 50-question slice (1 question = `2.0` pts on code/math/ifeval, `3.3` pts on tool parts) after a `+0.5` pt round averages to `-1.0` pts and **does not** falsely trip the `-2.0` brake. A genuine 2-round drop (`-2.0` and `-2.5`, or a single catastrophic `-4.5` drop after `0.0`) averages `<= -2.0` and trips the brake.
   - **Save cap (`sv >= cap`):** Uses `cur` (`mon_hist[-1]` only, `line 5379`), NOT the 2-round mean. As soon as a part reaches its save cap in `r-1`, it is held in round `r` (and passed to `repair_pairs.py` via `--damage-Hold`, where `dmg` is the current slice’s `da`, so an undamaged cap-held part gets `n_rem <= 0.40 * n` and `rec_h <= 0.20 * n_rem`).
   - **Unlatch (`--od-unlatch`):** Because `held_parts` is cleared at `line 5367` (`if args.od_unlatch: held_parts = set()`) at the start of every round $r > 1$, a part unlatches in round $r$ iff **both** `mean(da[r-2], da[r-1]) > -2.0` **and** `sv[r-1] < cap`. Note a subtlety: if a part dropped `-3.5` pts in `r-2` (triggering hold in `r-1`), and in `r-1` recovers to `-0.8` pts, the 2-round mean entering round `r` is `(-3.5 + -0.8)/2 = -2.15 <= -2.0`, so it stays held for one extra stabilizing round (`r`) before unlatching in `r+1`. Under `--od5-held`, that extra round uses `dmg = -0.8 > -0.02` (`40%` quota, compact `be-right` pairs), which stabilizes accuracy without lengthening!
   - **Missing part in a monitor (`vals` check at `lines 5376–5377`):** `vals = [h[p].get("da") for h in mon_hist[-n_sm:] if p in h and h[p].get("da") is not None]`. Since every `od3g` slice (`s01..s22`) contains all 6 parts (`50/50/50/30/30/30`), `vals` always has length `min(r-1, 2)`, and if a part were ever missing in one monitor, `vals` gracefully averages the available one.

**Verdict on `--od-smooth 2`:** Clean, bug-free, and interacts well with `--od-unlatch`, `--od-acc-hold -2.0`, and `--od5-held`.

---

## 3. Audit of the 4 Queue Flags You Did Not Take

1. **`--rp-lr 1.4e-5` (kept) vs. `1e-5` (rejected):** **Keep `1.4e-5`.** As noted in Section 0.1, all `od5` runs used `1.4e-5`.
2. **`--count-mode atleast` (kept) vs. `length` (rejected):** **Keep `atleast`.** In `repair_pairs.py` (`lines 193–224`), `count_violation(..., mode="atleast")` only flags a response when it falls *below* a minimum count requirement (`actual < n`) or violates `less_than`/`exact` when `mode != "atleast"`. Switching to `length` activates `line 206` (`if mode in ("exact", "length") and rel == "at_least" and actual > n + 2: return True`), which rejects valid IFEval answers that satisfy "at least $N$ words/sentences/paragraphs/bullets" by writing more than $N+2$ units! All `od5c`/`od5f`/`od5g` runs used `atleast`; keep `atleast`.
3. **Keeping `--base-stats`, `--base-resp`, `--base-think`, `--think-only`, `--prefer-reread`, `--count-rules`, `--response` in `track.py`:** **Keep them.** In `track.py` (`lines 5255–5293`), `--od5-v2` passes those flags to `repair_pairs.py`, where `base_think` and `base_resp` set per-question original baselines `bt` (`lines 324–333`) and `right_resp_cap` (`line 369`). Dropping `--base-think` falls back to `max(t_lens)` of the current round’s 8 rollouts (`line 333`), which shrinks every round and corrupts `bt` (breaking `recovery` bounds `1.05 * bt`, `loop` detection `1.35 * bt`, and shorten target `min_r * bt`)! Keeping the exact `od5g` base flags was 100% the right call.
4. **Dropping monitor-based `--od-stop-from 18 --od-stop-save 61.5`:** **Keep it dropped.** Across `od5c`, `od5f`, and `od5g`, the 240-question slice monitor is noisy (`±2.5` pts SD on saving, `+2.5 to +3.0` pts optimistic on accuracy). Only Dev (`dev600 x 1`) via `fullpick` should gate stopping/selection.

---

## 4. Risk Analysis: No-Fit (`4:0.10`, ratio `1.20`, cap `76`, 2nd pair) & Code (`28:0.32`, ratio `1.04`, 2nd pair)

### 4.1 No-Fit: Risk of Spurious Tool Calls or Broken Format
- **What is the failure mode on no-fit?** On BFCL `no_fit` (relevance detection where none of the provided tools fit the user’s request), a wrong answer (`s == 0`) happens when the model hallucinates a `<tool_call>` instead of declining/answering in plain text; a right answer (`s >= 1.0`) abstains from calling irrelevant tools.
- **Why does shortening no-fit *reduce* tool-call hallucinations (`effsame.txt`)?** In `effsame.txt`, the original model’s accuracy on `no_fit` is **75.0% with 0 thinking tokens** vs. **69.7% with full thinking** (`390` tokens), and `od5g-r20` scored **+5.5 pts** (`75.2%` vs. `69.7%`) at `44.1%` saved. Longer thinking on `no_fit` causes the model to talk itself into force-fitting a tool parameter; cutting thinking to a brief check (`4–40` tokens: *"None of the tools match X; I will state that directly"*) prevents overthinking into a false tool call.
- **What in `repair_pairs.py` prevents format breakage or bad pairs when pushing `4:0.10`, ratio `1.20`, 2nd pair?**
  1. `clean` filter (`line 366`): Every chosen candidate `r` must have `r["s"] >= 1.0` (verified no false tool call), `not has_format_defect(r)` (`lines 130–145`: must have valid `<think>...</think>` tags, no extra `<think>` tags, non-empty response body `>= 2` chars, and not `finish_reason == "length"`), and `not has_loop(r, bt)` (no repeated blocks or n-grams).
  2. Floor `4:0.10`: Because `no_fit` original thinking averages ~390 tokens (`p10 ≈ 180t`), `max(4, 0.10 * bt)` sets the per-question floor to **~18–39 tokens** (not 4 tokens, except on questions where the original already thought `< 40` tokens). That leaves enough room for a 1-sentence tool-relevance rejection inside `<think>...</think>`.
  3. Second shorten pair (`lines 426–438`): Only triggers when a question has **at least 2 distinct clean answers** and **at least 2 distinct longer anchors** (`len(clean) >= 2 and len(longer_pool) >= 2`), pairing the second-shortest clean answer against a distinct longer trace with `tw2 >= max(tl + 15, tl / 0.80)`. If a question does not have two clean short traces that both pass `s >= 1.0` and format checks, no second pair is emitted.

### 4.2 Code: Risk at Floor `28:0.32`, Ratio `1.04`, Quota `52`, and 2nd Pair
- **Why did code lose accuracy in `od5c-r20` (`48.6% saved / -3.1 pts`) and `od5f-r18` (`42.9% saved / -2.8 pts`), whereas `od5g-r20` got `38.6% saved / -0.5 pts`?**
  - In `od5c` and `od5f`, code had floor **`24:0.22`** and ratio **`1.08`** (`od5c`) / **`1.05`** (`od5f`). Because code original thinking averages ~1,440 tokens, `0.22 * bt` allowed pairs that truncated 1,000-token reasoning chains down to ~220 tokens in a single jump (`effsame.txt` shows code accuracy plunges from `46.8%` at `902t` to `18.8%` at `443t`).
  - In `od5g`, raising the code floor to **`28:0.32`** and lowering the ratio to **`1.02`** improved code full-test accuracy by **+2.3 pts** (`-0.5` pts vs. `-2.8` pts in `od5f`), and on the `s20` monitor code was `+2.2` pts (`58.5%` vs. `56.2%` baseline) with **0 rounds held** across all 20 rounds.
- **Is `28:0.32` + ratio `1.04` + 2nd pair safe on code?**
  - **Floor `28:0.32` is the primary safety guard:** No shorten pair on code can ever pick a chosen trace shorter than `32%` of that question’s original thinking length `bt` (roughly `450+` tokens on a 1,400-token problem), and `tl <= 0.80 * tw` requires the chosen trace to be at least 20% shorter than the anchor while still passing all unit tests (`s >= 1.0`).
  - **What is the subtle risk of putting `code` in `--od5-second-parts` alongside quota `52`?**
    Look at `repair_pairs.py` lines `436–438`: when a code question emits a 2nd shorten pair (`sh2`), both `sh1` and `sh2` go into `cand["code"]["shorten"]`, which is sorted globally by `key=lambda x: x[0]` (`tl / max(bt, 1)`). If 20–25 easy code questions (out of 50 in the slice) each contribute **2** shorten pairs with low `tl / bt`, then filling quota `52` will draw those pairs from **only ~26–30 distinct code questions** (`~2` pairs per easy question) instead of spreading across 35–40 questions! Furthermore, in `od5g` (without 2nd pair on code), code already had plenty of single-pair shorten candidates (`34–45` per round in `s01–s20`), unlike `ifeval` and `no_fit` which only have 30 questions/slice on `no_fit` and fewer long anchors.
  - **Wait—let's check the numbers:** On `od3g` slices, `code` has **50 questions per slice**. In `od5g`, single-pair `cand["code"]["shorten"]` yielded `32–45` candidates per round (average `37.6` candidates across rounds 2–20). Because `37.6 < 52`, without a 2nd pair on code, a quota of `52` would only get ~38 shorten pairs per round and (`line 720`) **dump the remaining ~10–14 slots into `recovery` (lengthening) pairs on code!** That’s why `code` *needs* `--od5-second-parts` if its quota is `52`!
  -Wait: verify `line 720` in `repair_pairs.py`! On an **unheld** part:
    - `n_sh5 = int(round(n_rem * c_sh / tot_c))`
    - `sh5 = take(part, "shorten", n_sh5)`
    - `rec5 = take(part, "recovery", min(n_rec5, n_rem - len(sh5)))`
    - `hold5 = take(part, "hold", n_rem - len(sh5) - len(rec5))`
    - `line 719-720`: `if len(sh5) + len(rec5) + len(hold5) < n_rem: rec5 += take(part, "recovery", n_rem - len(sh5) - len(rec5) - len(hold5))`
    Look at what happens in `od5g` when `cand[part]["shorten"]