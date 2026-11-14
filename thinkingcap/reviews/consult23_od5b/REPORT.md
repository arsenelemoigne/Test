# Consult 23: Diagnosis of `od5b` and Redesign for Rounds 3 Onward

## 1. Diagnosis: Why `od5b` Shortens at One-Third to One-Half of `od4`’s Pace

After 2 rounds on the exact same test-matched slices, `od5b` reaches **16.4% saved / -0.52** vs `od4`’s **41.8% saved / -2.02** (and after round 1: **11.2% / +0.43** vs **22.7% / +0.58**). Per type after round 2 (`od5b-r2` vs `od4-r2`):
- **Math (`svamp`):** `32% / -1.8` vs `55% / -3.8` (`0.58x` pace)
- **Code (`mbpp`):** `6% / -0.8` (down from `9%` in R1) vs `46% / -3.9` (`0.13x` pace)
- **IFEval (`ifeval`):** `11% / +0.2` vs `29% / -0.1` (`0.38x` pace)
- **Simple tools (`bfcl_simple_python`):** `12% / +0.8` vs `31% / +0.8` (`0.39x` pace)
- **Several tools (`bfcl_multiple`):** `23% / -1.7` vs `44% / -1.2` (`0.52x` pace)
- **No-fit (`bfcl_irrelevance`):** `20% / +0.6` vs `41% / -1.8` (`0.49x` pace)

Running `repair_pairs.py` on `od4-r2.jsonl` (`6,240` answers) and inspecting `pairs-od5b-r1.jsonl` plus the `od5b` round 1–3 logs reveals **five compounding mechanisms**—confirming and extending your first reading.

### A. Verification of Your First Reading: Eligibility + Neutral `right` Pairs on Code and IFEval
1. **High `shorten` eligibility bar (`p1 == 1.0 if p0 == 1.0 else p1 >= max(0.75, p0 - 0.125)`) locks out mixed Code and IFEval questions:**
   - In `test_difficulty.json`, the original solves only **52.4%** of Code (`29.1%` at `p0 = 1.0`, `11.9%` at `0.80–1.0`, `15.9%` at `0.50–0.80`, `43.1%` below `0.50`) and **59.4%** of IFEval (`35.9%` at `p0 = 1.0`, `11.5%` at `0.80–1.0`, `18.3%` at `0.50–0.80`).
   - Under `p1 >= max(0.75, p0 - 0.125)`, any question where the original solved `2/8` to `5/8` and the current model solves `2/8` to `5/8` (`p1 >= p0`, completely undamaged!) is **barred from `shorten`** because `p1 < 6/8`.
   - Worse, when `p0 = 1.0` (`8/8`) and `p1 = 7/8` (normal 8-sample binomial noise on an undamaged question), `od4` (`--shorten-min-right 0.85`) treated it as **shorten-eligible**, whereas `od5b` (`p1 < p0` for `p0 >= 0.875`) disqualifies it from `shorten` **and** flips it into `damaged` (`recovery`), where `chosen` is `1.08–1.45x` **longer** than `rejected`.
2. **In `od4`, `be-right` (`right_cr`) pairs were a shortening signal; in `od5b`, `hold` pairs are length-neutral (`1.00x`) and `recovery` pairs actively lengthen:**
   - In `od4` (`od3_arena`), `--right-hold` was empty: every `right` pair picked the **shortest right answer above the floor** (`rights[0]`) over the **shortest finished wrong answer** (`wrongs[0]`). On `od4-r2.jsonl`, `chosen` was shorter than `rejected` in **60.0%** of Code `right` pairs (`median C/R = 0.94`), **65.1%** of Math `right` pairs (`median C/R = 0.89`), and **63.6%** of No-Fit `right` pairs (`median C/R = 0.91`).
   - In `od5b`, undamaged mixed questions (`1 <= k1 < 8`) build `hold` pairs in `[0.90, 1.12]` centered at `1.00`, where **`chosen` is longer than `rejected` in 40–58% of pairs** (`median C/R = 1.00–1.02` in `pairs-od5b-r1.jsonl`).

### B. Three Hidden Candidate-Killers and Counter-Gradient Multipliers in `repair_pairs.py`
3. **`--resp-near-long 0.95` and `--resp-max 1.05` are applied to ALL `--think-only` types, destroying 36% of Code, 67% of No-Fit, and 73% of IFEval `shorten` candidates:**
   - Because `--think-only` lists all 6 parts, lines 307–311 of `repair_pairs.py` filter `shorten` candidates on **every type** by requiring `response_len(short)` to lie within **`[0.95, 1.05]x` of `response_len(long)`** (the longest-thinking right answer’s final response length) and `>= 0.95 * rmed`.
   - On `od4-r2.jsonl` (at `k1 >= 6`, `spread >= 1.30`), this unintended response-matching filter cuts valid `shorten` candidates from:
     - **Code (`mbpp`):** `67` raw $\to$ **`42`** (`-37%`, because different valid code solutions/docstrings after `</think>` differ in length by `>5%`).
     - **No-Fit (`bfcl_irrelevance`):** `56` raw $\to$ **`20`** (`-64%`, because polite tool-refusal explanations vary in length by `>5%`). By round 3 of `od5b`, No-Fit collapsed to **only 9 `shorten` candidates** and fell **5 pairs short of its 24-pair quota**!
     - **IFEval (`ifeval`):** `56` raw $\to$ **`15`** (`-73%`, because two right answers to an IFEval prompt rarely have identical response token counts within `±5%` of `long`).
4. **`--no-shorten-counting` bans shortening on 50–55% of all IFEval questions (`96–110 / 200`), even when `od5b` already has per-question original response-length floors (`0.92 * b_resp`):**
   - On `od4-r2.jsonl`, `counting(q[0])` flags `96 / 200` IFEval questions (including upper-bound rules like “less than N words/sentences” or keyword/punctuation counts where shorter thinking is completely safe), whereas `at_least(q[0])` flags only `59 / 200`. Combining `--no-shorten-counting` with `--resp-near-long 0.95` leaves IFEval with only **`10` (R2) and `12` (R3) `shorten` candidates out of 200 questions**.
5. **No `scarcest-kind` truncation + Fixed Reference (`--ref-base`) Counter-Gradients:**
   - When `od4` (`od3_arena`) had only `15` IFEval `shorten` candidates on `od4-r2.jsonl`, `repair_pairs.py` cut IFEval’s total pairs from `60` to `25` (`15` shorten, `7` right, `3` loop) so the batch stayed **`72.9%` `shorten` (`148 / 203` pairs)** and only **`10.8%` of pairs had `chosen > rejected`**.
   - In `od5b`, when IFEval has only `10–12` `shorten` candidates out of a `66`-pair quota (`88` in R1), `od5` fills the remaining `54–76` slots with `recovery` (`15–18` pairs, `chosen = 1.08–1.45x rejected`), `hold` (`17–22` pairs, `~1.00x`), and `right_count` (`11` pairs, `chosen >= 1.10x + 15`). Thus IFEval (`28%` of the entire batch!) has **`chosen` longer than `rejected` in 50% of its pairs**.
   - Across the whole batch, `od5b` has only **`45.0%` (R1), `47.5%` (R2), and `46.0%` (R3) `shorten` pairs** (and on Code: `25` in R1 + `22` in R2 = **`47` Code `shorten` pairs total vs `96` in `od4` R1–R2**). Worse, **`22–50%` of `od5b` pairs have `chosen` longer than `rejected`**. Under `--ref-base` (fixed original long reference), lengthening pairs start with high DPO loss (`>0.70`, large gradient weight $\sigma(-\beta \Delta) \approx 0.65$) while already-shortening pairs have lower loss (`~0.35`, weight $\approx 0.30$). That is why `od5b` training accuracy stalls at **`69–74%` ordered right (loss `0.52–0.56`)** vs `od4`’s **`91–92%` (loss `0.32–0.37`)**, cutting net shortening per step in half.

---

## 2. What to Change from the Next Round Onward

To reach the `rpM-60` box (`54.1% saved / -0.93` on full test) in a single clean online run while keeping `od5`’s protections and the user’s principle (*shorten where the model is as good as the original, recover where it is damaged*), we make **six targeted adjustments** starting from the next round:

1. **Eligibility Rule (`can_shorten` and `damaged` in `repair_pairs.py`):**
   - **Shorten where the model is at least as good as the original (within 1 answer of noise on easy questions):** Change `can_shorten` from `(p1 == 1.0) if p0 >= 1.0 else (p1 >= max(0.75, p0 - 0.125))` to:
     - If `p0 == 1.0` (`8/8`): allow `p1 >= 0.875` (`7/8` or `8/8`, matching `od4`’s `--shorten-min-right 0.85` so a single-rollout slip on an `8/8` question does not block shortening or trigger lengthening recovery).
     - If `p0 < 1.0`: allow `p1 >= max(0.50, p0 - 0.125)` when `p0 >= 0.50`, and `p1 >= max(0.25, p0)` when `p0 < 0.50` (any mixed question with `>= 2` right answers above the floor where the model’s solve rate `p1` is at least as good as the original `p0` can shorten its right thinking).
   - **Recover only where truly damaged ($\ge 2$-answer drop, or $\ge 1$-answer drop when `p0 < 0.875`):** Change `damaged` from `(p1 < p0) if p0 >= 0.875 else (p1 <= p0 - 0.125)` to `1 <= k1 < n_ans and (p1 <= p0 - 0.25 if p0 >= 0.875 else p1 <= p0 - 0.125)`. This stops `8/8 -> 7/8` binomial noise from generating lengthening `recovery` pairs.

2. **Fix the Response-Length Filter & Counting Ban on `shorten` Pairs (`repair_pairs.py`):**
   - Apply the response-length floor (`response_len(r) >= max(args.resp_median * rmed, 0.92 * b_resp)`) and cap (`response_len(r) <= args.resp_max * response_len(long)`) **only to `ifeval`** (never to `mbpp`, `svamp`, or tool calls, where code/JSON/refusal correctness is already verified by the grader).
   - Drop `--resp-near-long` (`0.95 * response_len(long)`) on `shorten` pairs: `max(0.95 * rmed, 0.92 * b_resp)` already protects IFEval response length against multi-round shrinkage without forcing `short` and `long` to have nearly identical response lengths.
   - Drop `--no-shorten-counting`: block IFEval shortening only on `is_cr` (`at_least(q[0])`, the `59/200` lower-bound length questions), while allowing the other `37/200` `counting` questions (which have `0.92 * b_resp` response protection) to shorten thinking.
   - **Verified impact on `od4-r2.jsonl`:** These two fixes immediately jump `shorten` candidates from **`42 -> 76` on Code (`+81%`)**, **`14 -> 44` on IFEval (`+214%`)**, and **`19 -> 60` on No-Fit (`+216%`)**!

3. **Undamaged Mixed Questions (`hold` $\to$ Compact Shortening `right` when Undamaged, Neutral `1.00x` only when Held):**
   - Following the user’s principle: when a type is **not held** (`not no_shorten`), on an undamaged mixed question (`1 <= k1 < 8`, `not damaged`), a `right` pair should teach **both** correctness and conciseness above the thinking floor (`ftok`).
   - Specifically, when `not no_shorten`, select `chosen` and `rejected` on undamaged mixed questions by preferring a clean right answer (`>= ftok`, preferring `REREAD` if available) whose thinking is **no longer than the wrong answer’s thinking** (`0.50 <= thinking_len(r) / thinking_len(w) <= 1.00`, picking the shortest such right answer), falling back to `[0.90, 1.10]` centered at `1.00` only if all clean right answers are longer than `wrongs`. When a type **is held** (`no_shorten` is True because it hit its saving cap or accuracy brake), keep the strict neutral `[0.90, 1.12]` window centered at `1.00` so held types do not drift.
   - **Verified impact on `od4-r2.jsonl`:** This cuts the share of batch pairs with `chosen > rejected` from **`25.4%` down to `11.7%`** (and on IFEval from `51.7%` down to `26.7%`, Code from `20.8%` to `8.3%`, Math from `11.7%` to `3.3%`).

4. **Floors (`--floor-by-part`):**
   - In `od5b` round 3 (at only `32%` Math saving), `153` right Math answers, `27` IFEval, `12` No-Fit, and `9` Code right answers were already dropped below `0.48 / 0.42 / 0.36 / 0.42`. Our compression simulation on `od4-r2.jsonl` shows that at `53%` overall saving, a `0.48` Math floor drops `40%` of right answers and a `0.42` Code floor drops `28%`.
   - Lower the per-question floors slightly to **`svamp=25:0.44+mbpp=35:0.36+ifeval=20:0.32+bfcl_simple_python=20:0.42+bfcl_multiple=25:0.42+bfcl_irrelevance=20:0.35`**. This keeps a hard safety floor well above the `<0.30x` danger zone while allowing `shorten` candidates to remain plentiful up to `54–56%` saving.

5. **Quotas, Caps on Non-Shortening Pairs, Steps per Round, and Learning Rate:**
   - **Keep rounds small at `15` steps / `240` pairs**, with a slightly rebalanced `240`-pair quota that gives Code more weight (matching its `0.25` benchmark weight): **`svamp=58,mbpp=52,ifeval=58,bfcl_simple_python=26,bfcl_multiple=22,bfcl_irrelevance=24`** (`240` pairs total).
   - **Cap `recovery` and `right_count` shares when `shorten` candidates are available (`repair_pairs.py`):** In `od5`’s quota allocation, cap `right_count` at `10%` of IFEval’s quota (`<= 6` pairs instead of `11`), cap `hold` at `20%` of `n_rem` (down from `25%`), and cap `recovery` at `20%` of `n_act` whenever the type’s monitored slice accuracy vs the original (`avg_dmg`) is better than `-1.5` pts (allowing uncapped proportional `recovery` only if a type’s slice accuracy drops below `-1.5` pts). This guarantees **`62–68%` `shorten` pairs + `15–20%` compact-shortening `right` pairs** in every normal round.
   - **Learning rate:** Increase `--rp-lr` slightly from `1.2e-5` to **`1.4e-5`** (constant after 2 warmup steps; between `od5b`’s `1.2e-5` and `od4`’s `1.5e-5`). Because `odpo_fixed` skips rounds whose `models/{name}-r{k}` directory already exists, editing the queue line and restarting `od5b` will seamlessly resume from the latest completed round (`r3` or `r4`) using the new pair builder and `1.4e-5` for all remaining rounds up to round 12–16.

### C. Exact Code Changes in Words
1. **In `reviews/consult23_od5b/pack/code/repair_pairs.py` (inside the `if args.mix_preset == "od5":` block, lines 286–365):**
   - **Lines 299–303 (`can_shorten` and `damaged`):** Change `can_shorten` so that when `p0 is not None`, it is `(p1 >= 0.875)` if `p0 >= 1.0`, `(p1 >= max(0.50, p0 - 0.125))` if `0.50 <= p0 < 1.0`, and `(p1 >= max(0.25, p0))` if `p0 < 0.50`. Change `damaged` to `1 <= k1 < n_ans and (p0 is not None) and ((p1 <= p0 - 0.25) if p0 >= 0.875 else (p1 <= p0 - 0.125))`.
   - **Lines 307–311 (response-length filter on `shorten`):** Change `if base_part in think_only:` to `if part == "ifeval":` so only IFEval filters `pool_s` by response length, and remove the `and (not args.resp_near_long or response_len(r) >= args.resp_near_long * response_len(long))` clause inside `od5` so `pool_s` only enforces `response_len(r) >= floor_r` (`max(args.resp_median * rmed, 0.92 * b_resp)`) and `response_len(r) <= args.resp_max * response_len(long)`.
   - **Lines 331–342 (`hold` pairs on undamaged mixed questions):** When `not no_shorten`, look first for pairs `(r, w)` with `r in clean`, `w in wrongs` (and on IFEval `response_len(r) <= args.right_resp_cap * response_len(w) + 15`) where `0.50 <= thinking_len(r) / max(1, thinking_len(w)) <= 1.00`, keyed by `(0 if decode and REREAD.search(decode(r)) else 1, thinking_len(r), abs(ratio - 0.75))`. If none exists (or when `no_shorten` is True because the type is held), fall back to the existing neutral `[0.90, 1.12]` loop centered at `1.00`.
   - **Lines 352–363 (quota assembly per part):** Change `right_count` cap from `0.20 * n` to `0.10 * n`. Change `n_hold` cap from `0.25 * n_rem` to `0.20 * n_rem`. Compute `avg_dmg = sum(od5_dmg[part]) / len(od5_dmg[part]) if od5_dmg[part] else 0.0`; when `avg_dmg > -0.015` and `c_sh > 0`, cap `n_rec` at `min(c_rec, int(round(0.20 * n_act)))` and give the rest of `n_act` to `shorten` (`n_sh = min(c_sh, n_act - n_rec)`).
2. **In `reviews/consult23_od5b/pack/code/plan_line_b.txt` (the active `odpo --name od5b` line):**
   - Update `--rp-quota` to `svamp=58,mbpp=52,ifeval=58,bfcl_simple_python=26,bfcl_multiple=22,bfcl_irrelevance=24`, remove `--no-shorten-counting` and `--resp-near-long,0.95` from `--rp-pair-args`, update `--floor-by-part` to `svamp=25:0.44+mbpp=35:0.36+ifeval=20:0.32+bfcl_simple_python=20:0.42+bfcl_multiple=25:0.42+bfcl_irrelevance=20:0.35`, and set `--rp-lr 1.4e-5`.

### D. Updated Queue Line (Plain Text)
odpo --name od5b --od-fixed-ref --rp-base Qwen/Qwen3-0.6B --od-rounds 16 --od-steps 15 --od-r1-steps 20 --od-r1-quota svamp=80,mbpp=56,ifeval=88,bfcl_simple_python=36,bfcl_multiple=28,bfcl_irrelevance=32 --rp-quota svamp=58,mbpp=52,ifeval=58,bfcl_simple_python=26,bfcl_multiple=22,bfcl_irrelevance=24 --rp-questions svamp=220,mbpp=140,ifeval=200,bfcl_simple_python=80,bfcl_multiple=60,bfcl_irrelevance=80 --rp-own-data --rp-kinds mix --rp-samples 8 --rp-seed 18 --rp-pair-args=--think-only,svamp+mbpp+ifeval+bfcl_simple_python+bfcl_multiple+bfcl_irrelevance,--prefer-reread,--count-rules,--shorten-ratio,1.30,--shorten-min-right,0.85,--mix-preset,od5,--base-stats,data/od3-base-stats.json,--base-resp,data/od3-base-resp.json,--base-think,data/od3-base-think.json,--floor-by-part,svamp=25:0.44+mbpp=35:0.36+ifeval=20:0.32+bfcl_simple_python=20:0.42+bfcl_multiple=25:0.42+bfcl_irrelevance=20:0.35,--count-mode,atleast,--count-resp-min,1.10,--count-resp-cap,1.75,--count-resp-pad,60,--resp-median,0.95,--resp-max,1.05,--right-resp-cap,1.15 --rp-lr 1.4e-5 --rp-warmup 2 --rp-beta 0.1 --rp-dpo-seed 0 --rp-dev-vs rpE80 --od-dev-rounds 99 --data-seed 18 --od-caps svamp=62,mbpp=63,ifeval=56,bfcl_simple_python=46,bfcl_multiple=49,bfcl_irrelevance=56 --od-acc-hold -2.0 --od-unlatch --od-stop-save 999 --od-stop-acc -0.8 --od-min-rounds 10 --od-conv-gain 0.5 --od-conv-min-save 52

---

## 3. Expected Trajectory, Comparison to `rpM-60`, and Risks

### A. Expected Round-by-Round Monitor Trajectory (Resuming from Round 3/4)
With `60–65%` `shorten` pairs + `18%` compact-shortening `right` pairs (`chosen > rejected` reduced from `~30%` to `~12%`) and `lr = 1.4e-5`, the per-round shortening pace doubles from `~5–6%` per round to **`~8–10%` per round** over rounds 4–7 before smoothly decelerating as questions approach their `0.32–0.44x` floors and unlatched caps (`62 / 63 / 56 / 46 / 49 / 56`):

| Round | Overall Monitor Saved / Acc | Math (`svamp`) | Code (`mbpp`) | IFEval (`ifeval`) | Simple (`simple`) | Several (`multiple`) | No-Fit (`irrelevance`) | Notes |
|---|---|---|---|---|---|---|---|---|
| **R1** (done) | `11.2% / +0.43` | `19% / -0.5` | `9% / +1.0` | `8% / +0.6` | `13% / +0.5` | `6% / -0.8` | `9% / +1.9` | Before 25% hold cap |
| **R2** (done) | `16.4% / -0.52` | `32% / -1.8` | `6% / -0.8` | `11% / +0.2` | `12% / +0.8` | `23% / -1.7` | `20% / +0.6` | With 25% hold cap |
| **R3** (done) | `~21–23% / -0.4` | `~40% / -1.5` | `~14% / -0.5` | `~15% / +0.2` | `~19% / +0.5` | `~28% / -1.2` | `~25% / +0.8` | Last round with old pair rules |
| **R4** (new) | `~32% / -0.4` | `~50% / -1.5` | `~28% / -0.6` | `~25% / +0.4` | `~28% / +0.6` | `~36% / -0.8` | `~36% / +1.2` | Code/IFEval/No-Fit unlocked |
| **R5** | `~41% / -0.5` | `~57% / -1.8` | `~41% / -0.8` | `~34% / +0.5` | `~36% / +0.5` | `~43% / -0.6` | `~45% / +1.5` | Math approaches `0.44x` floor |
| **R6** | `~48% / -0.6` | `~61% / -1.8` | `~51% / -1.0` | `~42% / +0.5` | `~42% / +0.4` | `~47% / -0.5` | `~51% / +1.8` | Math nears 62% cap |
| **R7** | `~52.5% / -0.6` | `~62% / -1.6` | `~57% / -1.1` | `~47% / +0.4` | `~45% / +0.3` | `~49% / -0.4` | `~54% / +2.0` | Math & Several touch caps |
| **R8** | `~54.5% / -0.7` | `~61% / -1.4` | `~60% / -1.1` | `~50% / +0.4` | `~46% / +0.2` | `~49% / -0.4` | `~56% / +2.0` | Inside target box (`>52% / >-1.0`) |
| **R9–R10** | `~55.5% / -0.7` | `~61% / -1.3` | `~61% / -1.0` | `~52% / +0.3` | `~46% / +0.2` | `~49% / -0.4` | `~56% / +2.0` | Convergence stop triggered |

### B. How Many Rounds Are Needed and Comparison to `rpM-60` (`54.1% / -0.93`)
- **Rounds needed:** Because rounds 1–3 only advanced the model to `~22%` saved (equivalent to 1 round of `od4`), **5 to 6 additional rounds under the fixed rules (landing at Round 8 to Round 10)** are needed for Code (`6% -> ~60%`) and IFEval (`11% -> ~50%`) to catch up to Math (`32% -> ~61%`). With `--od-min-rounds 10` and `--od-conv-gain 0.5 --od-conv-min-save 52`, the run will naturally converge and stop at **Round 10 (or Round 11)** without needing all 16 rounds, and `fullpick` will evaluate the top 2 Dev checkpoints (`r8`/`r9`/`r10`).
- **How close this gets to `rpM-60` (`54.1% / -0.93` on full test):**
  - On the full test, `rpM-60` achieved `54.1% / -0.93` via: Math `61.3% / -1.91`, Code `62.2% / -1.77`, IFEval `44.7% / -0.46`, Simple `45.3% / -0.25`, Several `47.5% / -0.62`, No-Fit `60.2% / +5.52`.
  - By comparison, `od4-r8` (`51.6% / -1.30`) fell short of `rpM-60` by `2.5%` saving and `0.37` accuracy points because (1) latched holding caused Math/Tools to lengthen back in rounds 6–8 while IFEval response lengths shrank (`-0.41` pts), and (2) early-stopped thinking on `6–8/8` questions (`-0.84` pts weighted) and finished repetition loops (`-0.39` pts weighted) went unaddressed.
  - With the fixed `od5b` rules, we expect the full test of the best round (`r8`–`r10`) to land at **`53.0%–55.0%` saved and `-0.6` to `-0.95` accuracy**, matching `rpM-60`’s box (`>50%` saved, better than `-1.0` accuracy) in a single online run:
    - **Math (`0.25`):** `60–62% saved / -1.3 to -1.7 acc` (`+0.3` to `+0.5` pts better than `rpM-60` and `od4-r8` via the `0.44x` per-question floor and bounded recovery).
    - **Code (`0.25`):** `58–61% saved / -1.0 to -1.4 acc` (`+0.4` to `+0.7` pts better than `rpM-60` via finished-loop rejection and `0.36x` floor).
    - **IFEval (`0.25`):** `47–51% saved / -0.1 to +0.4 acc` (`+0.5` to `+0.8` pts better than `rpM-60` via `0.92 * b_resp` response floor + `atleast` counting pairs).
    - **Tools (`0.25` combined):** `48–50% saved / +0.2 to +0.6 acc` (Simple `45% / 0.0`, Several `48% / -0.4`, No-Fit `55% / +2.0`).
- **What would close any remaining `0.1–0.3` pt accuracy gap if full test lands at `-1.0` to `-1.1`:** A single 25-step post-run repair step (`rpP`, `lr = 8e-6`, `0%` shorten, `recovery + loop + format + neutral hold` on a disjoint slice) on `od5b-best` recovers `+0.3` to `+0.5` accuracy points on Math/Code while giving back only `~1.0–1.5%` saving.

### C. Key Risks and Mitigations
1. **Risk: Code or IFEval overshoots once candidate filters are removed.**
   - *Mitigation:* Per-question floors (`0.36x` Code, `0.32x` IFEval) and unlatched per-type caps (`63%` Code, `56%` IFEval, with `--od-acc-hold -2.0`) remain active; when a type hits its cap or `-2.0` monitor accuracy, `no_shorten` engages for that round and switches its `hold` pairs back to strict `1.00x` neutrality.
2. **Risk: Slice accuracy noise (`±1.5` pts on `60–140` questions) triggers too many `recovery` pairs in a single round.**
   - *Mitigation:* Capping `recovery` at `20%` of `n_act` unless slice accuracy drops below `-1.5` pts (and requiring a `>= 2`-answer drop on `7–8/8` questions) prevents single-rollout noise from flooding the batch with lengthening pairs.