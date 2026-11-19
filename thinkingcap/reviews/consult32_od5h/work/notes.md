# Consult32 Review Notes (od5h Pre-Launch Review)

## 1. Files Read So Far
- `README_PACK.md`
- `code_changes.diff`
- `od5h_plan_lines.txt`
- `effprobe/effsame.txt`
- `SMARTER_DATA.md`
- `NEW_RESULTS_od5f.md`
- `YOUR_CONSULT31_REPORT.md`
- `code/repair_pairs.py` (lines 180-338; need lines 338-750 and `code/track.py` around `odpo_fixed` & `fullpick`)

## 2. Corrections to Consult31 Errors Noted by User
1. **Math 8/8 & Churn (`SMARTER_DATA.md`):**
   - `od5c-r16` (54.8% saved, -0.78 acc):
     - Math Pass@8: `96.8 -> 96.4`; Math 8/8: **`70.1 -> 61.7`** (`-8.4` pts, NOT `75.1 -> 69.9`).
     - All tasks Pass@8: `88.2 -> 88.1`; All tasks 8/8: `56.1 -> 54.3` (`-1.8` pts; No-fit `72.5 -> 81.7`, Several `56.0 -> 64.0`, IFEval `34.0 -> 34.9`, Simple `66.5 -> 64.2`, Code `29.1 -> 29.1`).
     - Net drift (q / pts): Math `-28.4 / -2.84`, Code `-7.4 / -1.95`, IFEval `+2.6 / +0.49`, Simple `-4.4 / -1.09`, Multiple `0.0 / 0.00`, No-fit `+14.1 / +5.89`, All `-23.4 / -0.85`.
     - Excess churn (above exact hypergeometric null E0): Math `+21.2 q / +2.12 pts [15.2, 27.0]`, Code `+6.4 / +1.70`, IFEval `+11.5 / +2.12`, Simple `+13.4 / +3.34`, Multiple `+12.1 / +6.07`, No-fit `+8.5 / +3.53`, **All `+73.1 q / +2.65 pts [+61.2, +85.2]`** (vs `rpI-120` `+54.4`, `rpM-60` `+67.6`, `rpL-80` `+94.7`, `nothink` `+717`).
2. **`r2s3-s80` vs `rpL-80`:**
   - The `65.9% / +7.7` full-test score on No-Fit is `rpL-80` (offline DPO run) in `fulltests.json`.
3. **Learning rate:**
   - `od5c`, `od5r`, `od5f`, `od5g` all used `--rp-lr 1.4e-5` (not `1e-5`). The user's `od5h_plan_lines.txt` rightly keeps `--rp-lr 1.4e-5`.

## 3. New Data Since Consult31
- **`effsame.txt` (equal thinking caps on unseen slices 19-20):**
  - Base forced at `od5c-best`'s per-type caps (`b1..b9` percentiles of trained thinking):
    - `svamp`: original natural `88.6%` at `640t`; at `b1=130t` `12.0%` (trained `38.8%`), at `b5=205t` `30.2%` (trained `67.8%`), at `b8=322t` `59.5%` (trained `80.8%`).
    - `mbpp`: original natural `52.3%` at `1813t`; at `b0=0t` `50.0%`, `b1=183t` `49.6%` .. `b8=817t` `51.1%` (equal to trained).
    - `ifeval`: original natural `59.4%` at `333t`; at `b0=0t` `61.2%`, `b1=71t..b8=148t` `57.2-58.4%` (vs trained `~62%`).
    - `bfcl_simple_python`: original natural `81.2%` at `150t`; at `b1=47t..b9=92t` `80.6-82.8%`.
    - `bfcl_multiple`: original natural `73.3%` at `207t`; at `b0=0t` `56.7%`, `b1=44t` `65.8%`, `b4=61t` `71.5%`.
    - `bfcl_irrelevance`: original natural `82.2%` at `210t`; at `b0=0t` **`87.5% (+5.3 [+0.3, +10.5])`**, while at `b1=53t..b9=115t` it drops to `77.8-80.9%`!
- **`od5g-r20` full test CIs:** `57.4% [56.2, 58.4]`, `+0.14 [-0.63, +0.88]`.

## 4. Open Questions to Check in Code Next
1. Inspect `repair_pairs.py` lines 338-750 to check all branches of `--od5-held` (held + damaged, held + undamaged, held with no compact pair available, `right_count`/`loop`/`format` pairs taken before `n_rem`, shortfall handling, and whether unused slots should stay unused or go to unheld shortening).
2. Inspect `track.py` around `odpo_fixed` and `fullpick` to verify `--od-smooth 2` edge cases (round 1 / first rounds, missing part, interaction with `--od-unlatch` and `--od-acc-hold -2.0`, and how `ptrepair` handles fewer than 240 pairs for 15 steps).
3. Check the flags the user kept from `od5g` vs my consult31 line (`lr 1.4e-5`, `--count-mode atleast`, `--base-stats`, `--base-resp`, `--base-think`, `--think-only`, `--prefer-reread`, `--count-rules`, `--resp-median 0.95 --resp-max 1.05`, no monitor early-stop).
4. Evaluate 20 vs 22 rounds on `od3g`, `fullpick` Dev rounds (`16..20` vs `17..22`), and the Dev gate (`>= 58.5` saved, `>= +1.65` acc).

## 5. `repair_pairs.py` Lines 335-560 Findings
- In `od5` block (`if args.kinds == "mix" and od5:`, lines 339-476):
  - `floors` filter `all_rights` into `rights` (lines 331-338).
  - `clean = [r for r in rights if not looping(r)]` (line 362): chosen answers never loop, are finished, right, and >= floor.
  - `cand[part]["loop"]` (lines 364-365): `clean[0]` vs longest looping/unfinished wrong.
  - `cand[part]["format"]` (lines 375-380): `near[0]` (within 0.85-1.25x thinking of `bad[0]`) vs `bad[0]` (broken format: `response_len == 0`, math `_nofinal`, or BFCL `(` not in post-`</think>` response for `bfcl_simple_python` / `bfcl_multiple`).
  - `cand[part]["right_count"]` (lines 381-387): on IFEval `is_cr` (`at_least` when `--count-mode atleast`).
  - `cand[part]["shorten"]` (lines 406-433): requires `can_shorten` (`p1 >= 0.875` if `p0 == 1.0`, `>= max(0.50, p0 - 0.125)` if `p0 >= 0.50`, else `>= max(0.25, p0)`), `not no_shorten`, `len(clean) >= 2`. On IFEval, enforces `response_len(r) >= max(args.resp_median * rmed, 0.92 * b_resp)` and `<= args.resp_max * response_len(long)`.
  - `cand[part]["recovery"]` (lines 439-454): when `damaged` (`1 <= k1 < n_ans` and drop `>= 2/8` on `7-8/8` or `>= 1/8` on `<7/8`), chooses `r` with `1.08 * tw + 10 <= thinking_len(r) <= min(1.05 * bt, 1.45 * tw + 50)`.
  - `cand[part]["hold"]` (lines 455-475): `elif clean and wrongs and 1 <= k1 < n_ans:` (undamaged mixed question):
    - Line 457: `if args.od5_v2 and (args.od5_held or part not in set(filter(None, re.split(r"[,+]", args.no_shorten_parts)))):`
      searches for compact `be-right` pair (`0.50 <= ratio <= 1.00`), sorted by `(0 if REREAD else 1, thinking_len(r), abs(ratio - 0.75))`.
    - Lines 467-473: `for r in (clean if best is None else []):` **FALLBACK WHEN NO COMPACT PAIR EXISTS (`best is None`)!** It searches `0.90 <= ratio <= 1.12`! Wait: if `best is None` (e.g. every right answer thinks longer than the wrong answer, or `< 0.50x` of the wrong answer), a held part falls through to `0.90 <= ratio <= 1.12` (which can pick a right answer up to `1.12x` as long as a long wrong answer, or if wrong answers are short early stops, `0.90-1.12x` of the wrong answer). Let's check how lines 675-735 use `cand[part]["hold"]` and `cand[part]["recovery"]`!

## 6. Detailed Code Audit Findings (Questions 1-4)

### Q1: `--od5-held` Line-by-Line Audit (`repair_pairs.py` lines 455-475 & 680-698)
1. **Candidate construction (`lines 455-475`):**
   - `elif clean and wrongs and 1 <= k1 < n_ans:` (Wait: `elif` after `if damaged and src_rec and wrongs:`! Notice: if a question IS `damaged`, it enters the `if damaged and src_rec and wrongs:` block at line 439, even if `best` is `None` (no recovery pair fits `1.08 * tw + 10 <= thinking_len(r) <= hi`). Because of `elif`, a damaged question NEVER enters `cand[part]["hold"]`!).
   - Inside `elif clean and wrongs and 1 <= k1 < n_ans:` (undamaged mixed question):
     - Line 457: `if args.od5_v2 and (args.od5_held or part not in set(filter(None, re.split(r"[,+]", args.no_shorten_parts)))):`
       Searches for a compact pair `0.50 <= ratio <= 1.00`, preferring `REREAD`, then shortest `thinking_len(r)`, then `abs(ratio - 0.75)`.
     - Line 467: `for r in (clean if best is None else []):`
       If NO compact pair exists (`best is None`, i.e., every right answer is either `< 0.50x` or `> 1.00x` of every wrong answer), it falls through to the `0.90 <= ratio <= 1.12` loop! Since `0.90 <= ratio <= 1.00` was already checked and empty (except possibly on IFEval when `right_resp_cap` rejected a pair in `0.90..1.00`—and note line 470 does NOT check `right_resp_cap`!), any pair found in the fallback on non-IFEval has `1.00 < ratio <= 1.12` (chosen is LONGER than rejected by up to +12%), and on IFEval it bypasses `right_resp_cap`!
       However, how big is `+0% to +12%` when `best is None`? It is bounded by `1.12x` of a wrong answer (unlike `recovery`, which requires `1.08x + 10` to `1.45x + 50` and prioritizes `tw < 0.65 * bt`). Still, if the intent is strictly non-lengthening (`ratio <= 1.00`), falling through to `1.00 < ratio <= 1.12` when no compact pair exists slightly lengthens on those fallback questions (and on IFEval skips `right_resp_cap`).
2. **Defect pairs taken BEFORE `n_rem` (`lines 683-687`):**
   - `got5` takes `right_count` (10% of `n` on IFEval), `loop` (12% of `n`), `format` (10% of `n`).
   - Then `n_rem = n - len(got5)`.
3. **Held + undamaged (`dmg > -0.02`, lines 691-695):**
   - `if dmg > -0.02: n_rem = min(n_rem, int(round(0.40 * n)))`
   - Total pairs for the part = `len(got5) + n_rem`, which is up to `0.32n + 0.40n = 0.72n` on IFEval (or `0.22n + 0.40n = 0.62n` on other parts) if defect pairs are plentiful, or `0.40n` if `got5` is empty. (Wait: the prompt says "take at most 40% of their quota unless their slice lost > 2 pts, recovery at most 20% of that, 40% if damaged". Because `n_rem` is capped at `0.40 * n` AFTER `got5` is taken, `got5` sits outside the `0.40 * n` cap—which is actually good because `loop`/`format`/`right_count` are defect repairs, not lengthening!).
   - Inside `n_rem` (e.g. `0.40 * n` = 19 pairs for math `n=48`, or 11 pairs for simple `n=28`):
     - `rec_h` takes at most `int(round(0.20 * n_rem))` recovery pairs (`~4` pairs for math, `~2` for simple).
     - `hold_h` takes `n_rem - len(rec_h)` compact hold pairs (`~15` pairs for math, `~9` for simple).
   - **Can it starve a held part of protection or still lengthen it?**
     - Unlike old `od5f`/`od5g` (where a held part had `c_sh = 0`, so `n_hold5 = 0.20 * n_rem` and `rec5` took **80% of `n_rem`** = 38 recovery pairs on math, massively lengthening it by 25 pts of saving!), `--od5-held` flips the ratio to **20% recovery / 80% compact hold** when undamaged (`dmg > -0.02`) and **40% recovery / 60% compact hold** when damaged (`dmg <= -0.02`).
     - What if `cand[part]["hold"]` has fewer candidates than `n_rem - len(rec_h)`? Notice line 694 does NOT backfill missing `hold_h` slots with `recovery` (unlike line 720 in the old code: `rec5 += take(part, "recovery", ...)`!). That is crucial: in old `od5`, whenever `hold` ran short, line 720 dumped all remaining slots into `recovery` (lengthening). Here, if `hold` has fewer candidates, the remaining slots are simply left empty—it NEVER backfills with `recovery`.
     - Could it starve a held part of protection when `dmg <= -0.02` (held + damaged)? When `dmg <= -0.02`, `n_rem` is NOT capped at `0.40 * n` (it keeps the full `n - len(got5)` quota!), and `rec_h` gets `0.40 * n_rem` (e.g., on math `n=48`, up to 19 recovery pairs + 29 compact hold pairs). That is plenty of protection without the 38-pair recovery flood that caused the 15-25 pt saving oscillation in `od5f`/`od5g`.
     - What if a held part has `dmg` between `-0.015` and `-0.02` (e.g. `-1.8` pts)? It is treated as undamaged (`n_rem <= 0.40 * n`, `rec_h <= 0.20 * n_rem`). That matches `--od-acc-hold -2.0`.
4. **Unused slots (`shortfall` and `dpo.py` step count):**
   - Line 698 `continue` skips `if len(by_part[part]) < n: shortfall[part] = ...` AND skips the summary lines 725-730 (`od5 candidates...`, `od5 shorten/recovery/right/loop/format`, `SHORT of quota`)! (Minor logging omission: for held parts it logs lines 696-697 instead of 725-730, which is fine and clean).
   - Because `ptrepair` runs with `--rp-epochs 2 --batch 16 --max-steps 15` (`track.py` line 4928, 5303-5305, 5433), `dpo.py` computes `total = math.ceil(len(rows) * 2 / 16)` and `steps = min(total, 15)`. As long as `len(rows) >= 120` pairs across all 6 parts (and even if math + simple + multiple were all held at 40%, `240 - 0.6*(48+28+28) = 178 >= 120`), `dpo.py` STILL runs all **15 optimizer steps of 16 pairs = 240 pair-updates** by wrapping into epoch 2 of the stratified pair file!
   - Wait: Look at what happens when `len(rows)` is e.g. 190 pairs instead of 240 pairs, and `dpo.py` runs 15 steps of 16 pairs = 240 examples with `--keep-order` and `epochs=2` (`dpo.py` line 127: `order = [i for _ in range(args.epochs) for i in range(len(rows))][:steps * args.batch]`):
     Because `repair_pairs.py` interleaves parts in stratified order, the first 190 examples are 1.0x of the file, and the remaining `240 - 190 = 50` examples are the first `50/190 = 26%` of the file repeated a second time!
     Even more importantly: in that stratified file of 190 pairs, the unheld parts (code, IFEval, no-fit) ALREADY make up a larger fraction of the 190 pairs (`(52+48+36)/190 = 71.6%` vs `136/240 = 56.7%`), so across the 240 gradient updates in 15 steps, the unheld parts AUTOMATICALLY receive ~26% more gradient updates (`172` pair updates out of `240` instead of `136`), WITHOUT lowering pair quality by digging deeper into the candidate pool (`extra = take(p_, "shorten", ...)` as `od5_v3` did in `od5d`, which hurt accuracy: `od5d` dropped to `49.5% / -1.21`)!
     This is a great insight: keeping `ptrepair` at 15 steps with `epochs=2` on the stratified file naturally scales up the step share of the unheld parts using their top-ranked pairs, rather than Scraping bottom-of-barrel pairs like `od5_v3`.
