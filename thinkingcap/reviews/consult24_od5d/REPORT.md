# Review of `od5c` Plateau Diagnosis and `od5d` Design

## 1. Is the Diagnosis Right? What Else Limits the Saving?

**Yes, all four parts of your diagnosis are exact and verified against the code and the `od5c` pair logs:**

1. **Held types leave slots empty and collapse the batch's net shortening ratio:**
   In `od5c`, once math (`svamp`) and code (`mbpp`) reach 54–62% and 51–59% saved (rounds 4–7), the `-2.0` accuracy hold triggers (`mbpp` held in r4 and r7; `svamp` in r5–r7; `bfcl_irrelevance` in r5). Total shortening pairs per round drop from **209 / 320 (65.3%)** in r1 and **138 / 240 (57.5%)** in r3 to **110 / 238 (46.2%)** in r4, **79 / 204 (38.7%)** in r5, **98 / 224 (43.8%)** in r6, and **62 / 213 (29.1%)** in r7. Meanwhile, `recovery` pairs (which reward longer thinking, `1.08 * tw + 10` up to `1.05 * bt`) rise to **46–55 pairs/round (20.5–27.0%)**, nearly cancelling out the 62–98 shortening pairs across the shared LoRA adapter.
2. **IFEval is strictly candidate-limited, not quota-limited:**
   IFEval's quota is 58 pairs/round (`n_sh5` target ~31 after 6 `right_count`, 7 `loop`, and 20% `hold`), yet `cand["ifeval"]["shorten"]` yields only **24, 27, 19, 20, and 13 candidates** across rounds 3–7. Every single shorten candidate is taken (`24/24`, `27/27`, `19/19`, `20/20`, `13/13`), leaving 0 unused shorten candidates in every round. Two rules choke IFEval's candidate pool:
   - First, `--count-mode atleast` sets `is_cr = True` on ~59/200 IFEval questions per slice (30%) and unconditionally skips `shorten` on all of them (`if not is_cr:`).
   - Second, on the remaining ~141 questions, `od5c`'s eligibility rule (`p1 >= max(0.75, p0 - 0.125)`, and `8/8` if `p0 == 1.0`) excludes both mixed-solve questions (`p0` in `4/8..5/8`, where IFEval's 58.6% base solve rate concentrates) and any `8/8 -> 7/8` question (`p1 = 0.875 < 1.0`).
3. **The three tool types are strictly quota-limited, with 73–90 unused shorten candidates sitting idle every round:**
   With quotas of `26 / 22 / 24` (= 72 total, 30% of the batch), `bfcl_simple_python`, `bfcl_multiple`, and `bfcl_irrelevance` take only `17–19`, `13–16`, and `15–18` shorten pairs per round (`46–51` total), while leaving **28–38**, **15–22**, and **30–33** valid shorten candidates unused on the table in every unheld round.
4. **The official 4-benchmark weighting makes IFEval and Tools equal to Math and Code:**
   Both `od3_monitor` (arithmetic mean of the 4 benchmark savings, weighting BFCL `400 : 200 : 240`) and the full-test headline (`1 - geometric mean` of the 4 benchmark token ratios) give 25% weight each to Math, Code, IFEval, and BFCL. In `od5c` round 6, Math (54%) and Code (51%) average **52.5%**, whereas IFEval (43%) and BFCL (`(400*38 + 200*44 + 240*40)/840 = 40.0%`) average **41.5%**, pulling the headline to **46.9%** (47.3% geometric). Moving IFEval `43% -> 56%` alone lifts round 6 to **50.3%** (50.6% geometric); bringing IFEval and the three tool types to their caps (`56%` and `46 / 49 / 56%` -> BFCL `49.6%`) while Math and Code merely hold `57% / 53%` yields **53.9% saved** (54.0% geometric).

### Three Additional Code-Level Bottlenecks Limiting Saving in `repair_pairs.py`

Checking `repair_pairs.py` against the `od5c` logs and `od4-r2.jsonl` reveals three subtler bottlenecks that `od5d` must also fix so Rules A, B, and C work at full strength:

- **Bottleneck 5 (Held types and shorten-starved IFEval bypass the 20% `hold` cap on line 575, hiding empty slots and flooding the batch with neutral pairs):**
  Look at line 575 of `repair_pairs.py`:
  `hold5 = take(part, "right", cand[part]["hold"], n_rem - len(sh5) - len(rec5))`
  Instead of capping `hold5` at `n_hold5` (`20%` of `n_rem`), line 575 takes `n_rem - len(sh5) - len(rec5)`. Consequently, when `mbpp` is held (`sh5 = 0`), it does **not** leave its 30 shorten slots empty for Rule A to transfer—it dumps **31 `hold` pairs** in r4 (leaving only 1 empty slot out of 52!) and **25 `hold` pairs** in r7 (leaving only 6 empty slots!). Similarly, on `ifeval` in r7, when `sh5` drops to 13, line 574 first expands `rec5` past its 20% cap (`n_rec_cap = 7`) to take all **11 `recovery` pairs** (even though IFEval accuracy is `+2.1`!), and line 575 then takes **21 `hold` pairs** (plus 6 `right_count` = 27 `right` pairs) so IFEval reports 0 empty slots while training against shortening.
  - *Fix:* Cap `hold5` at `n_hold5` (`20%` of `n_rem`) on every part (including held parts), and on healthy parts (`dmg >= 0` and not held) cap `rec5` at `n_rec_cap` (`20%` of `n_act`) even when `len(sh5) < n_act - n_rec_cap`. That frees the true unused shorten slots on both held types and IFEval so Rule A transfers all of them to healthy shorten pairs, and stops healthy IFEval from taking excess lengthening `recovery` pairs.
- **Bottleneck 6 (`--resp-max 1.05` on line 344 kills 77% of Rule C's IFEval candidates if left active on counting/at-least questions):**
  Line 344 filters `pool_s` by `response_len(r) <= args.resp_max * response_len(long)` (`1.05x`). Rule C requires `response_len(r) >= response_len(long)` on counting/at-least questions (`is_cr`). If `--resp-max 1.05` remains active on `is_cr` questions, `response_len(r)` is squeezed into `[1.00, 1.05] * response_len(long)`. Testing on `od4-r2.jsonl` shows that on the 59 `at_least` IFEval questions, Rule C unlocks **30** shorten candidates without the `1.05x` upper cap (and **27** when requiring `r["output_tokens"] < long["output_tokens"]`), but only **7** if squeezed by `<= 1.05 * response_len(long)`.
  - *Fix:* On `is_cr` questions under Rule C, do not apply `args.resp_max * response_len(long)`; instead require `response_len(r) >= max(response_len(long), 0.92 * b_resp)` and `r["output_tokens"] < long["output_tokens"]` (so total tokens are strictly reduced while the response is at least as long).
- **Bottleneck 7 (`od5_dmg` ordering in `repair_pairs.py`):**
  In `repair_pairs.py`, `od5_dmg[part]` is currently accumulated inside the `for qid, q in groups.items():` loop (line 287) and only averaged at line 552. Because Rules B and C depend on whether `part` is healthy (`part_dmg >= 0.0` and `part not in stop5`) **during** candidate construction (lines 285–349), `part_dmg` must be pre-computed in a 4-line pass over `groups` before the `for qid, q in groups.items():` loop begins.

---

## 2. Decisions on Rules A, B, C and Hyperparameters

### Rule A (Slot Transfer from Unfilled Quotas to Healthy Types' Shorten Pairs): **KEEP and Strengthen**
- **Exact rule and threshold:** After every part takes its own pairs (with `hold5` strictly capped at `n_hold5 = round(0.20 * n_rem)` on all parts, including held parts, and `rec5` capped at `n_rec_cap = round(0.20 * n_act)` on healthy parts where `part_dmg >= 0.0`), compute `free = sum(quota.values()) - sum(len(v) for v in by_part.values())`. While `free > 0`, distribute those unfilled slots across **healthy** parts (`part not in stop5` and `part_dmg >= 0.0`) that still have unused `cand[part]["shorten"]` candidates, in proportion to their base quotas `quota[part]` (relaxing the per-part question cap `q_cap` from 1 to 2 only if needed after 1-per-question shorten candidates are exhausted). If no healthy part has remaining `shorten` candidates, leave the remaining slots empty.
- **Why:** In `od5c` rounds 4–7, capping `hold5` at 20% on held Math and Code frees **30–60 slots per round** whenever one or both are braked. Routing those slots into `shorten` pairs on healthy types (`part_dmg >= 0.0`) turns braked rounds from plateau rounds (`29–44%` shorten share) into targeted IFEval + Tool compression rounds (`~55–60%` shorten share, zero extra shorten pressure on braked Math/Code).

### Rule B (Shorten Eligibility Within 1 Answer in 8 of the Original): **KEEP, Applied ONLY to Healthy Types (`part_dmg >= 0.0` and `part not in stop5`), Plus the `8/8 -> 7/8` Fix on All Unheld Types**
- **Direct answer to the user's question (All types vs. only healthy types with slice accuracy `>=` original's):**
  Apply the lower-bin expansion (`p0 >= 0.5 -> p1 >= max(0.5, p0 - 0.25)`; `p0 < 0.5 -> p1 >= max(0.25, p0 - 0.125)`) **ONLY to healthy types (`part_dmg >= 0.0` and `part not in stop5`)**, while applying the top-bin fix (`p0 == 1.0 -> p1 >= 0.75`, i.e. `>= 6/8` instead of requiring `8/8` when `p0 == 8/8`) to **all unheld types**.
- **Why not apply full Rule B to damaged types (`part_dmg < 0.0`):**
  1. **Math and Code are never shorten-candidate-limited:** Even under the stricter `od5c` bar, Math has **101–146** shorten candidates for 32–37 slots (`3–4x` oversupply) and Code has **58–63** candidates for 30 slots (`2x` oversupply).
  2. **Protecting `od5c`'s `+2.7` accuracy advantage on Math and Code:** Loss analysis (`loss_analysis/od4.txt`) shows that when Math or Code is already slightly below the original (`-0.5` to `-1.9` pts, before the `-2.0` brake fires), shortening `2/8–5/8` fragile questions (`p1 >= 0.25` or `0.50`) causes early-stop calculation/logic drops. Keeping the `p1 >= max(0.75, p0 - 0.125)` floor on lower bins whenever `part_dmg < 0.0` preserves `od5c`'s high-reliability filter on Math and Code whenever their slice accuracy dips below `0.0`, while immediately unlocking `4/8–7/8` questions on healthy IFEval, Code (e.g. r1 and r4 where Code is `+1.6` and `+1.7`), and Tools whenever `part_dmg >= 0.0`.
- **Exact threshold formula:**
  - If `part not in stop5` and `part_dmg >= 0.0` (**healthy type**):
    - `p0 == 1.0`: `p1 >= 0.75` (`6/8`)
    - `p0 >= 0.5`: `p1 >= max(0.50, p0 - 0.25)` (`4/8` for `p0 in [4/8, 6/8]`, `5/8` for `p0 == 7/8`)
    - `p0 < 0.5`: `p1 >= max(0.25, p0 - 0.125)` (`2/8` for `p0 in [0, 3/8]`)
  - If `part not in stop5` and `part_dmg < 0.0` (**unheld but slightly negative slice accuracy, `-2.0 < dmg < 0.0`**):
    - `p0 == 1.0`: `p1 >= 0.75` (`6/8`, fixing the `8/8` cliff)
    - `p0 < 1.0`: `p1 >= max(0.75, p0 - 0.125)` (existing `od5c` rule).

### Rule C (Healthy IFEval Counting / At-Least Questions Shorten-Eligible When Thinking Less and Writing No Less): **KEEP with Total-Token Guard**
- **Exact rule and thresholds:** On a healthy IFEval (`part == "ifeval"`, `part not in stop5`, `part_dmg >= 0.0`), allow `is_cr` questions (`--count-mode atleast`, or `counting(q[0])`) to enter the `shorten` block (still keeping `right_count` at 10% of IFEval's quota as an explicit response-length booster on `is_cr` questions with wrong answers) if and only if for the chosen short-thinking answer `r` and rejected long-thinking answer `long`:
  1. `response_len(r) >= response_len(long)` (chosen response is at least as long as rejected response),
  2. `response_len(r) >= 0.92 * b_resp` (where `b_resp = base_resp.get(qid, rmed)` is the original's median right response length),
  3. `r["output_tokens"] < long["output_tokens"]` (and do **not** apply `response_len(r) <= args.resp_max * response_len(long)` on `is_cr` questions, so `1.05x` does not filter out valid longer-response answers that still save total tokens via much shorter thinking), and
  4. `thinking_len(long) / max(1, thinking_len(short)) >= args.shorten_ratio` (`1.30`).
- **Quantitative impact verified on `od4-r2.jsonl`:** Combined with Rule B, Rule C increases healthy IFEval's shorten candidates from **19–27 per slice** to **52–68 per slice**, completely eliminating IFEval's candidate starvation and allowing IFEval to fill its full ~35–42 shorten quota (plus transferred Rule A slots).

### Brake Threshold, Saving Caps, Quotas, LR, and Steps per Round
- **Brake threshold (`--od-acc-hold`):** **Keep `-2.0` with `--od-unlatch`.** At `-2.0`, single-slice noise (`~0.8–1.1` pts SD per type) does not falsely freeze Math or Code in rounds 2–3 (`-1.7` / `-1.9`), yet real accuracy drops (`-3.8` Code in r3, `-5.9` Math in r4) immediately trigger a 1-round recovery hold that rebounds Code by `+5.5` pts (`-3.8 -> +1.7`) and Math by `+3.5` pts (`-5.9 -> -2.4`). Moreover, gating Rules A, B, and C on `part_dmg >= 0.0` introduces a smooth **two-tier governor**:
  - `dmg >= 0.0`: healthy — receives Rule A transferred shorten slots + Rule B/C expanded shorten eligibility.
  - `-2.0 < dmg < 0.0`: caution — keeps its own base shorten quota under strict `p1 >= 0.75` eligibility, receives **no** transferred shorten slots from Rule A, and lets `recovery` expand up to 35% if `dmg <= -0.015`.
  - `dmg <= -2.0` (or saved `>= cap`): held — `0` shorten pairs; takes `recovery` + capped 20% `hold` + `loop`/`format`, and transfers all remaining slots via Rule A to healthy types.
- **Per-type saving caps (`--od-caps`):** **Raise Tool caps by +2 pts (`bfcl_simple_python=48,bfcl_multiple=51,bfcl_irrelevance=58`) and keep `svamp=62,mbpp=63,ifeval=56`.**
  - *Why:* Tools have `+1.1` to `+2.6` net accuracy in `od5c` and short thinking (`128–141` median tokens) where Rule A slot transfers will push tool savings faster; raising their caps from `46/49/56` to `48/51/58` prevents premature cap-braking on tools at 46% while remaining well inside the safe full-test tool frontier (`50–66%` saved at positive accuracy).
- **Quotas (`--od-r1-quota` and `--rp-quota`):** **Rebalance 12 pairs from Math/Code to Tools in R2+ (`svamp=52,mbpp=46,ifeval=58,bfcl_simple_python=30,bfcl_multiple=26,bfcl_irrelevance=28` = 240; keep R1 at 320: `svamp=76,mbpp=54,ifeval=88,bfcl_simple_python=40,bfcl_multiple=30,bfcl_irrelevance=32`).**
  - *Why:* In `od5c`, Math + Code had `110 / 240` (`45.8%`) of the base quota while all three tool types combined had only `72 / 240` (`30.0%`), even though Tools have equal 25% benchmark weight and 73–90 unused shorten candidates per round. Shifting 6 pairs from Math and 6 from Code directly to the three tool types (`72 -> 84`, `+17%` base tool quota) accelerates BFCL from Round 2 onward even before any type is held, and works synergistically with Rule A once Math/Code brake in Round 4+.
- **LR and steps per round:** **Keep constant `lr = 1.4e-5`, warmup 2, `beta = 0.1`, Round 1 = 20 steps (320 pairs), Rounds 2+ = 15 steps (240 pairs).** Now that Rule A prevents batch shrinkage (keeping rounds 4–8 at ~235–240 pairs instead of dropping to 204–213), 15 steps at batch 16 (`240` pair views = exactly 1.0 epoch) with `1.4e-5` delivers the ideal step size without over-stepping on a single slice.

---

## 3. Exact Code Changes and `od5d` Queue Line

### Code Changes in `reviews/consult24_od5d/pack/code/repair_pairs.py`

1. **CLI flag (`main`, around line 193):**
   Add `--od5-v3` (`action="store_true"`, help: `"od5d: Rules A (slot transfer to healthy shorten), B (1-in-8 shorten eligibility on healthy types), C (healthy IFEval counting/at-least shorten with response >= rejected and >= 0.92x base)"`). When `--od5-v3` is passed, also set `args.od5_v2 = True` so all `od5_v2` defaults remain active.
2. **Pre-compute per-part slice accuracy change `part_dmg` before the question loop (`main`, around lines 238–255):**
   Before `for qid, q in groups.items():`, compute `od5_dmg` across all questions in `groups` that exist in `base_stats`:
   For each `qid, q in groups.items()`, let `part = part_of(q[0])`, `p1 = sum(1 for r in q if right(r)) / max(1, len(q))`, and if `qid in base_stats`, append `p1 - base_stats[qid]["acc"]` to `od5_dmg[part]`. Then compute `part_dmg_map = {p: (sum(v) / len(v) if v else 0.0) for p, v in od5_dmg.items()}` and `stop_sh_set = set(filter(None, re.split(r"[,+]", args.no_shorten_parts)))`. Define a part as healthy via `is_healthy = (part not in stop_sh_set) and (part_dmg_map.get(part, 0.0) >= 0.0)`. (Remove the duplicate `od5_dmg[part].append(p1 - p0)` at line 287.)
3. **Rule B in candidate construction (`main`, inside `if od5:` around lines 288–291):**
   When `args.od5_v3` is true:
   - If `is_healthy` (`part not in stop_sh_set and part_dmg_map.get(part, 0.0) >= 0.0`), set `sh_ok = (p1 >= 0.75) if p0 >= 1.0 else ((p1 >= max(0.50, p0 - 0.25)) if p0 >= 0.50 else (p1 >= max(0.25, p0 - 0.125)))`.
   - Else (when `part_dmg < 0.0`), set `sh_ok = (p1 >= max(0.75, p0 - 0.125))` (which also allows `7/8` when `p0 == 1.0`, since `0.875 >= 0.875`).
4. **Rule C in candidate construction (`main`, inside `if od5:` around lines 331–349):**
   - At line 339, change `if not is_cr:` to `if (not is_cr) or (args.od5_v3 and part == "ifeval" and is_healthy):`.
   - Inside that block (lines 340–346), when `args.od5_v3 and part == "ifeval" and is_cr`: filter `pool_s` by requiring `response_len(r) >= max(response_len(long), 0.92 * b_resp)` and `r["output_tokens"] < long["output_tokens"]` (bypassing `response_len(r) <= args.resp_max * response_len(long)` for `is_cr`). For non-`is_cr` questions, keep the existing `od5_v2` filter unchanged (`response_len(r) >= floor_r and (args.resp_max <= 0 or response_len(r) <= args.resp_max * response_len(long))`).
   - At line 331 (`hold` pairs on undamaged mixed questions), keep `if not is_cr and not sh_ok and p1 >= p0 and clean and wrongs:` so questions newly made shorten-eligible by Rule B are not diverted into `hold` unless they fail to produce a shorten pair.
5. **Rule A and strict `hold`/`recovery` caps in the `od5` allocation block (`main`, inside `if args.kinds == "mix" and od5:` around lines 550–588):**
   - On line 573 (`n_rec_cap`), when `args.od5_v3` is true and `part not in stop5`: if `dmg >= 0.0`, strictly cap `rec5` at `n_rec_cap = int(round(0.20 * n_act))` (i.e. `rec5 = take(part, "recovery", cand[part]["recovery"], n_rec_cap if (args.od5_v3 and part not in stop5 and dmg >= 0.0) else (n_act - len(sh5)))`).
   - On line 575 (`hold5`), when `args.od5_v3` is true: cap `hold5` at `n_hold5 = min(c_hold, int(round(0.20 * n_rem)))` on **all** parts (including held parts in `stop5` and candidate-limited parts), instead of `n_rem - len(sh5) - len(rec5)` (and remove the `if (c_sh + c_rec) else n_rem` fallback on line 566 when `args.od5_v3` is true).
   - After the `for part, n in quota.items():` loop finishes (after line 588), add the Rule A transfer pass when `args.od5_v3` is true:
     Compute `free = sum(quota.values()) - sum(len(v) for v in by_part.values())`. Identify healthy parts `h_parts = [p for p in quota if p not in stop5 and part_dmg_map.get(p, 0.0) >= 0.0]`. While `free > 0`: in a round-robin weighted by `quota[p]` over `h_parts`, call `extra = take(p, "shorten", cand[p]["shorten"], 1)` (first with the existing `q_cap = 1`, and if a full pass over `h_parts` yields 0 pairs while `free > 0`, allow `q_cap = 2` for transferred shorten pairs), appending each taken pair to `by_part[p]` and decrementing `free` by 1, breaking when no healthy part can supply another `shorten` pair.

### Plain-Text Queue Line for `plan_line_b.txt` (`od5d`)

odpo --name od5d --od-fixed-ref --rp-base Qwen/Qwen3-0.6B --od-rounds 12 --od-steps 15 --od-r1-steps 20 --od-r1-quota svamp=76,mbpp=54,ifeval=88,bfcl_simple_python=40,bfcl_multiple=30,bfcl_irrelevance=32 --rp-quota svamp=52,mbpp=46,ifeval=58,bfcl_simple_python=30,bfcl_multiple=26,bfcl_irrelevance=28 --rp-questions svamp=220,mbpp=140,ifeval=200,bfcl_simple_python=80,bfcl_multiple=60,bfcl_irrelevance=80 --rp-own-data --rp-kinds mix --rp-samples 8 --rp-seed 18 --rp-pair-args=--think-only,svamp+mbpp+ifeval+bfcl_simple_python+bfcl_multiple+bfcl_irrelevance,--prefer-reread,--count-rules,--shorten-ratio,1.30,--shorten-min-right,0.85,--mix-preset,od5,--od5-v2,--od5-v3,--base-stats,data/od3-base-stats.json,--base-resp,data/od3-base-resp.json,--base-think,data/od3-base-think.json,--floor-by-part,svamp=25:0.44+mbpp=35:0.36+ifeval=20:0.32+bfcl_simple_python=20:0.42+bfcl_multiple=25:0.42+bfcl_irrelevance=20:0.35,--count-mode,atleast,--count-resp-min,1.10,--count-resp-cap,1.75,--count-resp-pad,60,--resp-median,0.95,--resp-max,1.05,--right-resp-cap,1.15 --dpo-lr 1.4e-5 --dpo-warmup 2 --dpo-beta 0.1 --od-caps svamp=62,mbpp=63,ifeval=56,bfcl_simple_python=48,bfcl_multiple=51,bfcl_irrelevance=58 --od-acc-hold -2.0 --od-unlatch --od-stop-save 53.0 --od-stop-acc -0.3 --od-conv-min 51.0 --od-conv-gain 0.4 --od-min-rounds 8

### How Many Rounds and When to Stop
- **Run up to 12 rounds** (`--od-rounds 12`, `--od-min-rounds 8`), with automatic stopping enabled after Round 8 if either:
  1. Monitor saving reaches `>= 53.0%` at monitor accuracy `>= -0.30` (`--od-stop-save 53.0 --od-stop-acc -0.3`), or
  2. Monitor saving is `>= 51.0%` and gains `< 0.4%` since the previous `>= 51.0%` round (`--od-conv-min 51.0 --od-conv-gain 0.4`).
- **Full-test selection (`fullpick`):** Score the top 2 rounds from rounds 6–10 that achieve monitor saving **`50.5–53.0%`** with monitor accuracy **`>= -0.20`** (and where neither Math nor Code is below `-2.5%` on that slice, i.e., pick a post-recovery round like `od5c` r5 rather than a pre-brake dip round like `od5c` r4).

---

## 4. Expected Trajectory, Full-Test Landing, and Main Risks

### Expected Round-by-Round Monitor Trajectory (`od5d`)
- **Round 1 (after 20 steps on 320 R1 pairs; Rules B & C active on all 6 types since R1 answers = original so `dmg = 0.0`):**
  **`26.5% saved / +0.3 to +0.5 acc`** (Math `31%`, Code `31%`, IFEval `23%` vs `18%` in `od5c` thanks to full 53/88 shorten fill from Rules B+C, Tools `22–25%`).
- **Round 2 (after 15 steps on 240 pairs):**
  **`39.5% saved / -0.8 to -0.3 acc`** (Math `45%`, Code `40%`, IFEval `36%` vs `31%` in `od5c`, Tools `33–41%`).
- **Round 3 (after 15 steps on 240 pairs):**
  **`46.5% saved / -0.6 to -0.1 acc`** (Math `55%`, Code `50%`, IFEval `43%` vs `36%` in `od5c`, Tools `39–43%`).
- **Round 4 (first Math or Code brake likely triggers, transferring ~25–30 slots to healthy IFEval & Tools via Rule A):**
  **`50.0% saved / -0.8 to -0.2 acc`** (Math `58%`, Code `55%`, IFEval `46%`, Tools `42–46%`).
- **Round 5 (held Math/Code rebounds in accuracy while Rule A drives IFEval + Tools upward instead of stalling at 47%):**
  **`50.5–51.5% saved / +0.0 to +0.5 acc`** (Math `56% / -1.8`, Code `53% / -0.5`, IFEval `49% / +1.0`, BFCL `45% / +1.8`).
- **Rounds 6–8 (steady convergence as IFEval reaches 51–54% and BFCL reaches 46–48% while Math/Code cycle between hold and light shortening at 55–58% / 53–56%):**
  **`51.8–53.2% saved / -0.2 to +0.3 acc`**.

### Expected Full-Test Landing and Why `od4`'s Monitor Over-Read Its Full Test (`-0.14` vs `-2.0`)
- **Why `od4-r7` read `-0.14` on slice 8 but landed at `-2.0` on the full test (and `-1.3` at `51.6%` saved for `od4-r8`):**
  Comparison of `NEW_RESULTS_14nov.md` and `loss_analysis/od4.txt` identifies three specific causes of `od4-r7`'s `-1.86` pt monitor-to-full-test gap:
  1. **Unbraked accuracy collapse on Math (`-3.8` full test vs `-1.8` slice) and Code (`-4.5` full test vs `-2.7` slice):** `od4` had **no accuracy brake** (`--od-acc-hold` was off, only saving caps at 65/66%) and **no `recovery` pairs**. It pushed Math to 65.5% saved and Code to 62.2% saved on the full test (`0.61x` and `0.59x` median right thinking), causing severe early-stopping losses (`-3.81` pts on SVAMP, `-4.50` pts on MBPP).
  2. **Winner's curse across 8 slices:** Selecting the single highest-accuracy monitor slice (`r7` at `-0.14`, whereas `r6` was `-1.19` and `r8` was `-0.98`) picked a favorable 780-question draw (`~0.6` pts positive sampling fluctuation), especially on MBPP (140 questions/slice vs 378 full test) and BFCL-Multiple (60 questions/slice vs 200 full test, where slice 8 read `+2.7` vs `-1.7` on the full test).
  3. **Monitor arithmetic vs. full-test geometric saving (+1.0 to +1.5% extra saving on full test):** Full-test saving is consistently **`+0.8%` to `+1.5%` higher** than the monitor headline because full-test `compare()` uses the geometric mean of the 4 benchmark token ratios (`od4-r7`: monitor `54.5%` -> full test `55.4%`; `od4-r8`: monitor `50.2%` -> full test `51.6%`).
- **Expected `od5d` full-test landing:**
  Because `od5d` keeps Math and Code in the safe **`54–57%` / `52–55%`** zone via `--od-acc-hold -2.0` + `recovery` pairs (avoiding `od4`'s `-3.8 / -4.5` Math/Code cliff) and gets its extra 4–5% headline saving from healthy IFEval (`50–53%`) and Tools (`46–48%`), a checkpoint reading **`51.5–52.5% saved / 0.0 to +0.3 acc`** on the monitor (with 2-round moving-average accuracy `>= -0.2` to avoid single-slice winner's curse) is projected to land at **`52.3–53.5% tokens saved / -0.6 to -0.1 accuracy`** on the full test — comfortably beating both `> 50%` saved and `> -1.0` accuracy.

### Main Risks and Mitigations Built Into `od5d`
1. **Risk: Rule C shorten pairs on IFEval `at_least` questions slightly erode IFEval's `+2.0` accuracy cushion if responses shrink toward the threshold.**
   - *Mitigation:* Rule C requires **both** `response_len(chosen) >= response_len(rejected)` (chosen writes no less than rejected on that exact prompt) **and** `response_len(chosen) >= 0.92 * b_resp`, while keeping `10%` of IFEval's quota dedicated to explicit `right_count` response-lengthening pairs (`>= 1.10x` rejected response). Furthermore, if IFEval slice accuracy ever dips below `0.0` (`part_dmg < 0.0`), Rules A, B, and C automatically turn off for IFEval on that slice.
2. **Risk: Rule A transfers too many shorten pairs to a single tool type in a round where both Math and Code are held, causing a 1-round tool accuracy dip.**
   - *Mitigation:* Proportional allocation across all healthy types, the `q_cap` per-question limit (80 questions/slice on Simple and No-Fit, 60 on Multiple, 200 on IFEval), and the per-type saving caps (`48 / 51 / 58%`) bound how many shorten pairs any single tool type can absorb in one round.
3. **Risk: Picking a full-test checkpoint on a single lucky slice (winner's curse) or immediately after an unbraked Math/Code spike.**
   - *Mitigation:* Only pick a round for full testing where (a) monitor saving is `>= 50.5%`, (b) both the current round and the 2-round average monitor accuracy are `>= -0.3`, and (c) Math and Code were either held in the preceding round (post-recovery) or sit better than `-2.2` on the current monitor slice.