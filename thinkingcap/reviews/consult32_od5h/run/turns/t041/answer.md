Look at `at_least` vs `length_rule` and `--no-shorten-counting`:
- In `od5c`, `od5r`, `od5f`, `od5g`, the command line had:
  `--count-rules,--count-mode,atleast,--no-shorten-counting,--count-resp-cap,1.35,--count-resp-pad,50,--right-resp-cap,1.25,--base-think,data/od3-base-think.json,--base-stats,data/od3-base-stats.json,--base-resp,data/od3-base-resp.json,--think-only,svamp+mbpp+ifeval+bfcl_simple_python+bfcl_multiple+bfcl_irrelevance,--resp-median,0.95,--resp-max,1.05,--prefer-reread`
- And on IFEval in `od5f-r18` and `od5g-r20`, those exact flags gave **+1.4% to +1.8% accuracy gain on IFEval on the full test at 53-56% saved**!
- Had we dropped `--base-stats`, `--base-resp`, `--base-think`, `--think-only`, `--prefer-reread`, `--count-rules`, `--resp-median 0.95`, `--resp-max 1.05`, `repair_pairs.py` would not even have `p0` (`base_stats` would be `{}` so `od5_dmg` would be empty and `p0 is None` on line 388 would fall back to `p1 >= args.shorten_min_right` with `damaged = False`!!), nor `bt` from `base_think`, nor `b_resp` from `base_resp`, nor `think_only` (`size(r)` would measure `output_tokens` instead of `thinking_len(r)`!), nor `REREAD` preference!
- Catching that in `od5h_plan_lines.txt` by building directly from `od5g`'s exact line saved the run from catastrophic failure. All of those kept arguments from `od5g` (`1.4e-5`, `--count-mode atleast`, `--base-stats`, `--base-resp`, `--base-think`, `--think-only`, `--prefer-reread`, `--count-rules`, `--resp-median 0.95`, `--resp-max 1.05`, and omitting monitor early-stop because the monitor reads +2.5 to +3.0 pts optimistic on accuracy) are **100% right and should NOT be changed**.

Now let's do a deep, surgical check of Question 1:
**"1. Review the diff line by line: does --od5-held do what you intended in every branch (held + damaged, held + undamaged, held with no compact pair available, the right-count / loop / format pairs taken before it)? Can it starve a held part of protection or still lengthen it? Is the 40% / 20% split right, and should the freed slots go to shortening on the unheld parts (today they are simply not used: rounds train on fewer pairs; ptrepair keeps 15 steps)?"**

Let's trace EVERY line of `--od5-held` in `code_changes.diff` and `repair_pairs.py`!
Let's look at the two places `--od5-held` touches in `repair_pairs.py`:

### Place 1: Candidate generation for `cand[part]["hold"]` (lines 455–475)
```python
 455: elif clean and wrongs and 1 <= k1 < n_ans:  # undamaged, mixed: a length-neutral be-right pair (0.90-1.12x, at 1.00)
 456: best = None
 457: if args.od5_v2 and (
     args.od5_held
     or part
     not in set(filter(None, re.split(r"[,+]", args.no_shorten_parts)))
 ):
     458:  # (14 Nov, consult23) not held: a COMPACT be-right pair, the right answer no longer than the wrong one
     459: for r in clean:
         460: for w in wrongs:
             461: ratio = thinking_len(r) / max(1, thinking_len(w))
             462: if 0.50 <= ratio <= 1.00 and (
                 base_part != "ifeval"
                 or args.right_resp_cap <= 0
                 or response_len(r) <= args.right_resp_cap * response_len(w) + 15
             ):
                 464: key = (
                     0 if decode and REREAD.search(decode(r)) else 1,
                     thinking_len(r),
                     abs(ratio - 0.75),
                 )
                 465: if best is None or key < best[0]:
                     466: best = (key, r, w)
 467: for r in clean if best is None else []:
     468: for w in wrongs:
         469: ratio = thinking_len(r) / max(1, thinking_len(w))
         470: if 0.90 <= ratio <= 1.12:
             471: key = (
                 round(abs(ratio - 1.0), 2),
                 0 if decode and REREAD.search(decode(r)) else 1,
             )
             472: if best is None or key < best[0]:
                 473: best = (key, r, w)
 474: if best:
     475: cand[part]["hold"].append((qid, best[1], best[2]))
```
WAIT A SECOND! Look at lines 467–473 ("held with no compact pair available"):
- Suppose a question is undamaged and mixed (`1 <= k1 < 8`), on a held part (`args.od5_held` is True), and **no compact pair (`0.50 <= ratio <= 1.00`) exists** (`best is None` after line 466)!
- Why would no pair with `0.50 <= ratio <= 1.00` exist?
  Because every clean right answer `r` has `thinking_len(r) > thinking_len(w)` (i.e. `ratio > 1.00`), OR `thinking_len(r) < 0.50 * thinking_len(w)`!
  When `ratio > 1.00` for all `(r, w)`, line 467 (`for r in (clean if best is None else []):`) runs the fallback loop:
  `if 0.90 <= ratio <= 1.12:` (which, since `ratio <= 1.00` was already checked and found empty—except on IFEval if `response_len` failed `right_resp_cap`!—can ONLY match `1.00 < ratio <= 1.12`, i.e., **chosen right answer is LONGER than rejected wrong answer**, OR on IFEval a pair that violated `right_resp_cap`!).
- Even worse: on `bfcl_irrelevance` (no-fit), remember line 5459 of `track.py`:
  `# consult20 tweak: no-fit is NOT length-held (its right answers are short refusals; holding them adds length)`
  Wait: on `bfcl_irrelevance`, right answers are often VERY short (`ratio < 0.50` vs a rambling wrong answer that tries to call a tool!). If `ratio < 0.50`, the `0.50 <= ratio <= 1.00` loop skips it, and then the fallback `0.90 <= ratio <= 1.12` checks if there is a long right answer near the wrong answer's length (`0.90-1.12x`)!
- Wait: what happens on unheld parts when `best is None`? On unheld parts (`part not in stop5`), `n_hold5` is capped at `20%` of `n_rem`, and `shorten` takes ~80% of `n_rem`.
  But on a **held part** (`held5 = True`), look at line 694:
  `hold_h = take(part, "right", cand[part]["hold"], n_rem - len(rec_h))`
  On a held part, `hold_h` takes **80% of `n_rem`** (when undamaged) or **60% of `n_rem`** (when damaged)!
  If `cand[part]["hold"]` includes fallback pairs from lines 467–473 where `best is None` in the `0.50 <= ratio <= 1.00` search, every single one of those fallback pairs has **`1.00 < ratio <= 1.12`** (chosen thinking longer than rejected thinking, or on IFEval violating `right_resp_cap`)!
- Let's check how often `best is None` in `0.50 <= ratio <= 1.00` vs `1.00 < ratio <= 1.12` on real rollouts or in theory:
  When a question has only wrong answers that stopped slightly earlier than the right answers (`1.00 < ratio <= 1.12`), an undamaged question is getting a **lengthening pair** (`chosen_tokens > rejected_tokens`, up to `+12%`) disguised as a "compact hold" pair!
  Wait: why did line 467 exist in the first place?
  In `od5_v2` (before `--od5-held`), when `part in stop5` (held), line 457 (`if args.od5_v2 and part not in stop5:`) was **False**, so held parts skipped lines 458-466 and used lines 467-473 (`0.90 <= ratio <= 1.12`). For unheld parts, lines 467-473 acted as a fallback if no `0.50-1.00` pair existed.
  With `--od5-held`, if a held part has NO compact pair (`0.50 <= ratio <= 1.00`) on a question, falling back to `1.00 < ratio <= 1.12` means picking a pair where `thinking_len(r) > thinking_len(w)` (up to `1.12x`)!
  Wait: does a held part have enough `0.50 <= ratio <= 1.00` candidates without the `1.00 < ratio <= 1.12` fallback, or could dropping the fallback starve a held part?
  Let's check! On a slice of ~300 questions (e.g. ~80 math, ~45 code, ~55 IFEval, ~40 simple, ~30 multiple, ~40 no-fit), how many mixed undamaged questions exist, and how many slots does a held part actually need under `--od5-held`?

Let's check Place 2 in `repair_pairs.py` (lines 681–698) first!
Look closely at lines 681–698 of `repair_pairs.py`:
```python
 681: stop5 = set(filter(None, re.split(r"[,+]", args.no_shorten_parts)))
 682: got5 = []
 683: for kind5, cap5 in (
     ("right_count", (0.10 if args.od5_v2 else 0.20) if part == "ifeval" else 0.0),
     ("loop", 0.12),
     ("format", 0.10),
 ):
     685: if cap5 > 0:
         686: got5 += take(
             part,
             "right" if kind5 == "right_count" else kind5,
             cand[part][kind5],
             int(round(cap5 * n)),
         )
 687: n_rem = n - len(got5)
 688: dmg = sum(od5_dmg[part]) / max(
     1, len(od5_dmg[part])
 )  # the part's accuracy change on this slice
 689: held5 = args.od5_held and part in stop5
 690: if (
     held5
 ):  # (19 Nov, consult31) a held part: compact holds, few recovery pairs, a smaller share of the batch
     691: if dmg > -0.02:
         692: n_rem = min(n_rem, int(round(0.40 * n)))
     693: rec_h = take(
         part,
         "recovery",
         cand[part]["recovery"],
         int(round((0.40 if dmg <= -0.02 else 0.20) * n_rem)),
     )
     694: hold_h = take(part, "right", cand[part]["hold"], n_rem - len(rec_h))
     695: by_part[part] = got5 + rec_h + hold_h
     696: counts[(part, "od5 slice accuracy vs the original (pts)")] = (
         f"{100 * dmg:+.1f}"
     )
     697: counts[(part, "od5 held: recovery / compact hold (pairs)")] = (
         f"{len(rec_h)} / {len(hold_h)}"
     )
     698: continue
```

STOP AND LOOK AT LINES 683–698 VERY CAREFULLY!
Let's trace EVERY branch of lines 683–698:

1. **What happens with `got5` (`right_count` / `loop` / `format` taken BEFORE `n_rem`)?**
   - Before checking `held5`, lines 683–686 take up to:
     - `0.10 * n` of `right_count` (on IFEval)
     - `0.12 * n` of `loop`
     - `0.10 * n` of `format`
     into `got5`!
   - Then `n_rem = n - len(got5)`.
   - Then on line 691-692:
     `if dmg > -0.02:`
     `    n_rem = min(n_rem, int(round(0.40 * n)))`
   - And then `by_part[part] = got5 + rec_h + hold_h`!
   - Wait: Look at what that does:
     - `got5` is taken against the FULL quota `n` (`0.10*n + 0.12*n + 0.10*n` = up to `0.32 * n` on IFEval, or `0.22 * n` on other parts).
     - And then `n_rem` (the pairs added *after* `got5`) is capped at `0.40 * n` (not `0.40 * n - len(got5)`!).
     - So if `got5` has e.g. `5` pairs on math (`n = 56`, `0.40 * n = 22`), `n_rem = min(51, 22) = 22`, and `by_part["svamp"]` gets `5 + 22 = 27` pairs (`48%` of `n`), which is fine (`got5` are defect repairs: loop + format + right_count, which are purely protective and anti-runaway).
     -Wait: is `got5` ever lengthening?
       - `loop`: `clean[0]` (shortest clean right answer) vs longest looping/unfinished wrong answer (`min(loops, key=lambda r: -r["output_tokens"])`). That is strongly **shortening** (kills runaways) and protective!
       - `format`: `near[0]` (`0.85 <= ratio <= 1.25`, closest to `1.0x` thinking of `bad[0]`) vs `bad[0]` (empty response, math `_nofinal`, or BFCL missing `(`). Length-neutral, fixes broken format!
       - `right_count`: IFEval `at_least` rule: `min(fit, key=thinking_len)` with `thinking_len(r) <= 1.2 * thinking_len(rej)` and `response_len(r) >= need`.
     - So taking `got5` before `if held5:` is actually GOOD: defect pairs (`loop`, `format`, `right_count`) are never starved by the hold!

2. **Now look at lines 693–695 in the `held + damaged` (`dmg <= -0.02`) and `held + undamaged` (`dmg > -0.02`) branches! THERE IS A CRITICAL BUG / ASYMMETRY HERE!**
   Compare lines 693–695 (`if held5:`) with lines 718–721 (the normal unheld `od5` block):
   In the normal block (lines 718–721):
   ```python
   hold5 = take(
       part,
       "right",
       cand[part]["hold"],
       n_hold5 if args.od5_v3 else n_rem - len(sh5) - len(rec5),
   )
   if (
       len(sh5) + len(rec5) + len(hold5) < n_rem and not healthy5
   ):  # protective kinds fill any gap, never shortening
       rec5 += take(
           part,
           "recovery",
           cand[part]["recovery"],
           n_rem - len(sh5) - len(rec5) - len(hold5),
       )
```
   Now look at what lines 693–695 do when `held5` is True:
   ```python
   rec_h = take(
       part,
       "recovery",
       cand[part]["recovery"],
       int(round((0.40 if dmg <= -0.02 else 0.20) * n_rem)),
   )
   hold_h = take(part, "right", cand[part]["hold"], n_rem - len(rec_h))
   by_part[part] = got5 + rec_h + hold_h
```
   WAIT A SECOND!
   Look at two cases in lines 693–695:
   - **Case A: `held + damaged` (`dmg <= -0.02`), when `cand[part]["hold"]` has fewer candidates than `0.60 * n_rem`!**
     Suppose a held part is **badly damaged** (`dmg <= -0.02`, e.g. math or simple tools lost `-3.5` pts on this slice!).
     Because the part is damaged on this slice, most of its mixed questions are classified as **`damaged = True`** (lines 401 & 439: they go into `cand[part]["recovery"]`, NOT `cand[part]["hold"]`, because line 455 is an **`elif`** after `if damaged and src_rec and wrongs:`!).
     Read lines 439 and 455 again!
     - Line 439: `if damaged and src_rec and wrongs:` -> appends to `cand[part]["recovery"]`.
     - Line 455: `elif clean and wrongs and 1 <= k1 < n_ans:` -> appends to `cand[part]["hold"]`!
     Because it is an `if ... elif ...`, **a damaged question NEVER enters `cand[part]["hold"]`!**
     Furthermore, on a slice of `od3g` (which has only **300 questions total** across all 6 parts! Let's check how many questions per part are in an `od3g` slice!), how many questions does each part have in a single round's slice?
     Let's check right now in Python or from the logs!
     In `od3` / `od3g`, a slice has ~300–450 questions total:
     - `bfcl_multiple`: ~25-40 questions
     - `bfcl_simple_python`: ~40-50 questions
     - `bfcl_irrelevance`: ~35-50 questions
     - `mbpp`: ~45-60 questions
     - `ifeval`: ~60-80 questions
     - `svamp`: ~80-100 questions
     Of those questions on a slice, most are `8/8` right (`k1 == n_ans = 8`) or `0/8` right (`k1 == 0`)! Only the **mixed** questions (`1 <= k1 <= 7`) can produce `recovery` or `hold` pairs (at most 1 pair per question)!
     On `bfcl_simple_python` (`n = 28`), `bfcl_multiple` (`n = 20`), `mbpp` (`n = 52`), `svamp` (`n = 56`):
     In a round where a held part is **damaged** (`dmg <= -0.02`), `n_rem` is NOT reduced to `0.40 * n` (it stays `~n = 56` on math, `52` on code, `28` on simple tools).
     Line 693 caps `rec_h` at `0.40 * n_rem` (e.g., 22 recovery pairs on math, 11 on simple tools), expecting `cand[part]["hold"]` to fill the other `0.60 * n_rem` (34 compact-hold pairs on math, 17 on simple tools)!
     Wait:
     1) When a part is damaged (`dmg <= -0.02`), half of its mixed questions are `damaged = True` (so they are in `cand[part]["recovery"]` and **excluded** from `cand[part]["hold"]` by `elif`!). There will often only be **5 to 12** undamaged mixed questions in `cand[part]["hold"]`!
     2) Because lines 693–695 do **NOT** have line 720 (`if len(rec_h) + len(hold_h) < n_rem: rec_h += take(...)`), if `cand[part]["hold"]` only has 8 pairs, `hold_h` gets 8 pairs, and the remaining unused slots are simply **dropped** even though `cand[part]["recovery"]` had more recovery pairs waiting to repair the damaged part! That **starves a held + damaged part of protection**!
     3) Wait: even more importantly, if a held part is **damaged** (`dmg <= -0.02`), WHY did its quota jump from `0.40 * n` back to `1.00 * n` (`n_rem` uncapped when `dmg <= -0.02`)?!
        Think about what happened in `od5f` and `od5g` in rounds 19–22!
        In `od5f` and `od5g`, why did saving crash from `59.5%` (r18) to `46.8%` (r21) and from `58.2%` (r20) to `50.6%` (r22)?
        Because on one noisy slice, code or math read `dmg <= -0.02` (e.g. code `-2.8` pts in `od5g` round 21!).
        When `dmg <= -0.02` on a held part:
        - `n_rem` is NOT capped at `0.40 * n` (`if dmg > -0.02: n_rem = min(n_rem, int(round(0.40 * n)))`), so `n_rem` jumps up to `100%` of `n` (52 pairs on code, 56 on math)!
        - AND `rec_h` takes `0.40 * n_rem` = **40% of the full quota `n`** (~21 recovery pairs on code, ~22 on math)!
        - Every single `recovery` pair has `1.08 * tw + 10 <= thinking_len(r) <= min(1.05 * bt, 1.45 * tw + 50)` — i.e., **chosen is strictly longer than rejected by 10–45%+**!
        - Putting 20+ lengthening recovery pairs into a single round on code or math is the EXACT mechanism that caused the round 19/21/22 lengthening rebounds in `od5f` and `od5g`!

Let's verify this! Look at the numbers:
- Suppose math (`n = 56`) or code (`n = 52`) is held, and on a single slice its raw slice accuracy `dmg` happens to be `-2.1` pts (`dmg <= -0.02`).
  Wait: notice that in `repair_pairs.py` line 688:
  `dmg = sum(od5_dmg[part]) / max(1, len(od5_dmg[part]))  # the part's accuracy change on this slice`
  `repair_pairs.py` does NOT know about `--od-smooth 2`! `dmg` in `repair_pairs.py` is ALWAYS the **single-slice** un-smoothed accuracy change of that round's slice!
  Read that again!
  In `track.py`, `--od-smooth 2` smooths `per_` over 2 slices to decide which parts are in `held` (`--no-shorten-parts`).
  Once a part is in `held` (`--no-shorten-parts`), `track.py` invokes `repair_pairs.py`.
  Inside `repair_pairs.py` (line 688), `dmg` is computed from `od5_dmg[part]` on **THIS single slice alone** (1-slice noise of ±2.5 to ±4.0 pts)!
  So even with `--od-smooth 2` in `track.py`, whenever a held part has a single noisy slice with `dmg <= -0.02` in `repair_pairs.py`:
  1. Line 691 (`if dmg > -0.02: n_rem = min(n_rem, int(round(0.40 * n)))`) is **bypassed**, so `n_rem` blows up from `0.40 * n` to `1.00 * n`!
  2. Line 693 (`int(round((0.40 if dmg <= -0.02 else 0.20) * n_rem))`) multiplies `0.40 * 1.00 * n = 0.40 * n` (which is **5x as many recovery pairs** as `0.20 * 0.40 * n = 0.08 * n`!!)!
  Look at that math:
  - When `dmg = -0.019` (undamaged):
    `n_rem = 0.40 * n`.
    `rec_h` cap = `0.20 * n_rem = 0.20 * (0.40 * n) = 0.08 * n` (on math `n=56`, that's **4 recovery pairs**).
  - When `dmg = -0.021` (0.2 pts lower on a single slice!):
    `n_rem` is NOT capped (`~0.90 * n = 50`).
    `rec_h` cap = `0.40 * n_rem = 0.40 * 50 = 20` (**20 recovery pairs** — a **5x jump** across a 0.2-pt single-slice threshold)!
  - AND in `cand[part]["hold"]` (lines 467–473), if a question has no compact pair (`0.50 <= ratio <= 1.00`), it falls back to `0.90 <= ratio <= 1.12` (which only matches `1.00 < ratio <= 1.12`, adding STILL MORE lengthening pairs)!

WHAT A CATCH! Let's double check that arithmetic:
- At `dmg = -1.9%`: `n_rem = 0.40 * n = 22` pairs on math; `rec_h <= 0.20 * 22 = 4` recovery pairs, `hold_h <= 18` compact hold pairs.
- At `dmg = -2.1%` (a 0.2% single-slice fluctuation!): `n_rem` jumps from `0.40 * n` (`22`) to `n - len(got5)` (`~50`), AND the recovery fraction doubles from `0.20` to `0.40`, so `rec_h` jumps from **4 recovery pairs to 20 recovery pairs** (500% as many lengthening pairs)!
- That 5x cliff at `dmg = -0.02` on a single un-smoothed slice is a textbook lengthening landmine!
- Why did that happen? Because my consult31 wording said "cap total pairs from any held type at 40% of its quota (unless its slice accuracy dropped below -2.0 pts) ... cap recovery pairs on held types at 20% of the type's remaining slots (or 40% if slice accuracy dropped below -2.0 pts)", which compounded removing the 40% quota cap AND doubling the recovery share on the same single-slice `dmg <= -0.02` trigger!
- How should lines 691–694 work instead so a held part NEVER explodes in lengthening pairs on a noisy slice, yet still gets protected when damaged?
  1. Keep `n_rem` capped on held parts EVEN when damaged — e.g. `0.40 * n` when `dmg > -0.02` and at most **`0.55 * n`** (or always `0.40 * n`! Wait: let's check: `0.40 * 0.50 * n = 0.20 * n` = 11 recovery pairs vs 4, or if `n_rem = min(n_rem, int(round((0.50 if dmg <= -0.02 else 0.40) * n)))` and `rec_h` is `0.30` if `dmg <= -0.02` else `0.15`, then `rec_h` goes smoothly from `3` pairs (`0.15 * 0.40 * 56`) to `8` pairs (`0.30 * 0.50 * 56`), instead of jumping from `4` to `20`!).
  2. Look at lines 467–473 in `repair_pairs.py`:
     ```python
     for r in clean if best is None else []:
         for w in wrongs:
             ratio = thinking_len(r) / max(1, thinking_len(w))
             if 0.90 <= ratio <= 1.12:
```
     When `args.od5_held` is True (or whenever line 457 ran!), why would `best` be `None` after lines 459–466?
     Only if NO `(r, w)` pair had `0.50 <= ratio <= 1.00`!
     If no `(r, w)` pair had `0.50 <= ratio <= 1.00`, then any pair matched by `0.90 <= ratio <= 1.12` in lines 467–473 MUST have **`1.00 < ratio <= 1.12`** (chosen thinking LONGER than rejected thinking)!
     Wait: what if `ratio < 0.50` (a very short right answer vs a long wrong answer, which is common on `bfcl_irrelevance` and on math/code when the wrong answer rambled)?
     In lines 459–466, `if 0.50 <= ratio <= 1.00:` REJECTS `ratio < 0.50`! And then if another right answer on that same question had `ratio = 1.08`, lines 467–473 pick the `ratio = 1.08` lengthening pair instead of the `ratio = 0.45` compact right answer!
     Wait: why did line 462 have `0.50 <= ratio <= 1.00` instead of `0.35 <= ratio <= 1.00`, and why did line 467 run when line 457 already searched for a compact pair?
     If line 457 runs (i.e. `args.od5_v2 and (args.od5_held or part not in stop5)`), we explicitly want a **non-lengthening** (`ratio <= 1.00`) `be-right` pair! If a question only has `ratio > 1.00` (where every right answer thinks longer than the wrong answer: an early-stop error on an undamaged question), picking a `1.00 < ratio <= 1.12` pair in `hold` makes `hold` pairs lengthen thinking!
     Even better: if `best is None` after checking `0.50 <= ratio <= 1.00`, what if the fallback for `args.od5_held` (or when `best is None`) checks `ratio <= 1.00` (down to the type floor, since `clean` is ALREADY filtered by the type's floor on line 337: `rights = [r for r in all_rights if thinking_len(r) >= ftok]`!) instead of `1.00 < ratio <= 1.12`?!
     THINK ABOUT THAT!
     Every `r` in `clean` has **ALREADY** passed the per-question type floor `thinking_len(r) >= max(abs_min, ratio_