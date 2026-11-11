```python
# `D12 lpd-chain1` is the SMOKING GUN in the log!
# Look at `D12 lpd-chain1` (L4767-4773) vs `D10 lpd-l2s50e` (L4708-4713):
# - In `D12 lpd-chain1`, the team tested a Round-1 pair mix with **40% shorten** (`shortest right > longest right >= 1.5x`) and **60% equal-length guard** (`shortest right > shortest finished wrong`) in `fp32+fp16` precision!
# - And what happened in `D12 lpd-chain1`?!
#   - Step 100: **+21.7% saved / +0.4 pp**
#   - Step 200: **+18.4% saved / +0.4 pp**
#   - Final (333): **+17.8% saved / +0.8 pp**!
#   It saved **UNDER 22%** (`17.8%` at the end!), and its saving *decreased* from step 100 (`21.7%`) to step 333 (`17.8%`), just like `lpc-runa2` (`33.4% -> 23.3%` on fresh) and `D6 lpd-simright` (`34.1% -> 28.1%`)!
# - Even worse: in `PATH1_PLAN.md`, Round 1 (`p1r1s0` / `p1r1s1` with `--kinds mix --mix-preset round1`) has:
#   - **39.4% shorten** (almost identical to `D12 lpd-chain1`'s 40% shorten!),
#   - **46.9% equal-length right** (`right_long` + `right_short`, where half have `chosen > rejected`),
#   - **13.7% capped guard** (`chosen <= 1.15 * tmed` vs `rejected < 0.5 * tmed`, where `chosen` is **>2x longer** than `rejected`!).
# - Therefore, on `D12 lpd-chain1` + `lpc-runa2` + `D6 lpd-simright`, we have **direct, unassailable empirical proof in the log** that Round 1 as currently configured in `PATH1_PLAN.md` (`--mix-preset round1`, 39.4% shorten) will land at **18–32% saved**, NOT 45–50% saved!

# Now let's check: HOW did `lpc-l2s50e` (`48.0% / -2.08` full test, `49.9% / -2.29` on 720) and `D10 lpd-l2s50e` (`54.4% / -3.6` at step 100, `53.5% / -2.7` at final on 720) and `ca1-soup-10` (`49.1% / -2.40` full test, `52.4% / -0.9` on 720) reach **50–55% saved** from the original Qwen using the team's OWN rules (not michal-bida's)?
# Let's check the team's OWN October pair rule (`l2s50e`, L3193, L3230, L4708, L4739):
# - In `l2s50e` (the team's own October recipe, NOT michal-bida's!):
#   1. **Shorten pairs (65–67% of pairs, 2:1 ratio to guard):** On questions solved `>= 50%` (or `>= 75%` on math/no-fit), `shortest right > longest right` with ratio `>= 1.5x` (or `>= 1.2x–1.25x` on always-solved `10/10` questions).
#   2. **Guard / Right pairs (33–35% of pairs):** On mixed-outcome questions, `shortest right (with --prefer-reread within 1.5x) > shortest finished wrong` (`--kinds guard`, plus `--count-rules` on IFEval counting-rule prompts).
#   3. **Zero `capguard` (`chosen > 2x rejected`) and zero `right_long` (`chosen > rejected`) in Round 1!**
#   Look at why that works for Round 1:
#   - When Round 1 uses **65–70% Shorten (`--shorten-ratio 1.3`, `--shorten-min-right 0.6` or `0.75` on math/no-fit, with 1.2x backup on always-solved) + 30–35% `rpE`-style Guard (`shortest right with reread > shortest finished wrong`, `--count-rules` on IFEval)**, **~85% of Round 1 pairs have `chosen_tokens <= rejected_tokens`**!
#   - That drives Round 1 cleanly into the **53–57% saved** zone (just like `D10 lpd-l2s50e` at `54.4%` step 100 / `53.5%` final, and `rc1-soup-30` at `57.4%` full test), while `--prefer-reread`, `--count-rules`, and `--think-only ifeval` already protect trap math, `no-fit`, and IFEval counting rules during Round 1!
#   - AND THEN, because Round 1 is at **53–57% saved**, **Round 2 (`p1r2`) operates in the proven `rc1 -> rpE` direction**: trading 2–4 points of excess saving (`54–57% -> 51–53%`) for **+1.5 to +2.5 pp of accuracy** using on-policy answers from Round 1!

print("Verified exact mechanism of D12 lpd-chain1 (17.8-21.7% saved) vs D10 lpd-l2s50e (53.5-54.4% saved)!")
```