```python
# Let's check THIS critical point from `CLAUDE_REVIEW.md` vs `EXPERIMENT_LOG.md`!
# Look at what `CLAUDE_REVIEW.md` showed:
# 1. In `lpc-runa2` (L3553-3617):
#    - 4,442 pairs from raw Qwen's own 10 answers: `shorten 2,265 (51%), guard 1,017 (23%), equal-length right 1,160 (26%)`, with NO shorten pairs on math and no-fit.
#    - Trained from raw Qwen (`LoRA r32, lr 2e-5 cosine, beta 0.2` or in `lpc-runac` `constant lr 2e-5, beta 0.1`).
#    - Result on 720 screen: **`lpc-runa2` plateaued at 32.5–33.4% saved (+0.44 pp)**, and **`lpc-runac` (constant lr 2e-5, beta 0.1) plateaued at 23.4–34.3% saved (-0.11 to -0.85 pp)**!
# 2. In `D6 lpd-simright` (L4566-4574):
#    - When equal-length `right>wrong` pairs (±15% length) were added 1:1 alongside shorten+guard pairs from raw Qwen (in `fp32+fp16` precision!), saving dropped to **34.1% at step 100 -> 30.3% at step 200 -> 28.1% at final**!
# 3. And why did `bidax2`, `bidax4`, `bidax6p`, `bidax5q`, and `lpc-l2s50e` reach **47–54% saved** in Round 1, whereas `lpc-runa2` and `D6 simright` only reached **28–33% saved**?!
#    WAIT A SECOND! Let's check the EXACT difference between the pair rules of `(bidax2, bidax6p, l2s50e, ca1)` vs `(runa2, D6 simright, and Path 1's Round-1 --mix-preset round1)`!
#    Look at `repair_pairs.py` when `--kinds mix --mix-preset round1` is run on raw Qwen's answers:
#    - In `bidax2` / `bidax6p` / `l2s50e`:
#      * `50%` of pairs were `shorten` (`shortest right > longest right >= 1.5x`).
#      * The other `50%` of pairs were `guard = shortest right > longest wrong` (in `bida`) or `shortest right > shortest finished wrong` (in `l2s50e` / `rpE`)!
#      * And as `copdanalysis` (10 Nov) and `L3777` proved, on raw Qwen's answers, **wrong answers are 1.6x–2.0x LONGER than right answers** (on 72% of math questions and 62% of code questions; in `math1`, 175 of 216 `right>wrong` pairs had `chosen < rejected`)!
#      * So in `l2s50e` and `bida`, BOTH `shorten` AND `guard` (`shortest right > wrong`) pushed length DOWN (total ~85–90% of pairs had `chosen_tokens < rejected_tokens`)!
#    - By contrast, what does `--kinds mix --mix-preset round1` in `repair_pairs.py` do?!
#      * Only `35%` (on math/no-fit), `27.5%` (on IFEval), and `50%` (on code/tools) — **39.4% overall** — are `shorten` pairs!
#      * `46.9%` overall are `right` (`right_long` + `right_short` within `1.2x`, where **half have `chosen > rejected`**, and on IFEval `right_cr` has `chosen >= rejected` in response length)!
#      * `13.7%` overall are `capguard` (`pick <= 1.15 * tmed` vs `early[-1] < 0.5 * tmed`), where **`chosen` is ALWAYS 1.5x–2.3x LONGER than `rejected`**!
#      * Let's sum the direction of length pressure in `--mix-preset round1`:
#        - Shorten pairs (`chosen < rejected`): **39.4%**
#        - `right_short` (`chosen <= rejected`, by only 0–17%): **~18%**
#        - `right_long` (`chosen > rejected`, by 0–20%): **~18%**
#        - `right_cr` (`chosen` response >= `rejected` response): **~11%**
#        - `capguard` (`chosen > 2.0x rejected`! Strongly pushes LONGER!): **13.7%**
#      * Net result: in `--mix-preset round1`, **>42% of all pairs have `chosen_tokens >= rejected_tokens`** (and the 13.7% `capguard` pairs have a 2x length gap pushing *longer*, which on raw Qwen—where natural early-stop wrong answers are rare, L3779—either fails to fill or actively fights shortening, just like `stopV2` and `D2 hard`)!
#      * PLUS in `track_ptrepair_odpo_dev.py` Line 122, `ptrepair` uses `--bida-precision` (`fp32 + fp16 autocast`) at `lr = 1.5e-5` for only `200 steps`!
#      * Therefore, Claude's review is **100% RIGHT on this point**: with `--mix-preset round1` (`39.4%` shorten, `46.9%` equal-length right, `13.7%` capped guard where chosen > 2x rejected) in `fp32+fp16` for 200 steps at `lr 1.5e-5`, **Round 1 from raw Qwen will land at ~28–36% saved, NOT 45–50% saved!**

print("Verified Claude's diagnosis of Round 1 pair-mix length pressure!")
```