```python
# Let's check what is currently running on GPU 1 and GPU 2, and how long each step takes!
# From `NEW_RESULTS.md`:
# - GPU 2: Dev-SVAMP-200 being built, then the Dev-SVAMP scores of rpE-80, rpK-60, rpL-60, then rpM (your design; consult11: run as is), then rpL-40's full test.
# - GPU 1: after the rpL-60 full test: Dev-SVAMP-200, then p1cr1s0 (Path 1's round 1, third version; consult11: run as is).
# - Not built yet: rpN (your accuracy-only repair on top of a shortened model, designed on rpK-60).

# Wait: since BOTH `rpK-40` (on GPU 2) and `rpL-60` (on GPU 1) have ALREADY finished their full tests right before this prompt (`NEW_RESULTS.md`: "Since then two more FULL tests printed: rpK step 40 = 51.2% saved / -0.8... and rpL step 60 = 59.3% / -1.7"), what are GPU 1 and GPU 2 doing AT THIS EXACT MOMENT (`~17:15`)?
# - GPU 2 just finished `rpK-40` full test (~17:05) and is now running:
#   1. `devsvamp` + `devscore` of `rpE-80`, `rpK-60`, `rpL-60` on `Dev-SVAMP-200` (~20 min total, finishing ~17:25).
#   2. `rpM` (~1.25h total: 12 min for `svamp=1200` extra rollouts + 15 min for 60 steps DPO + 45 min for `Dev-512` & `Dev-SVAMP-200` at steps 20/40/60; finishing ~18:40).
#   3. Queued after `rpM` on GPU 2: `fullmodel` of `rpL-40` (~1.6h)—which we can replace before `rpM` finishes!
# - GPU 1 just finished `rpL-60` full test (~16:50) and is now running:
#   1. `devsvamp` (skipped if GPU 2 already saved it, or ~2 min) + `devscore` of `p1r1s0-200` on `Dev-SVAMP-200` (~7 min).
#   2. `p1cr1s0` (already started around ~17:00! 0 generation time since it reuses `p1src`; ~50 min for 200 DPO steps + ~50 min for `Dev-512` and `Dev-SVAMP-200` at steps 100/150/200; finishing ~18:40).

# Let's check this timing!
# At ~17:15 (right now), `rpM` on GPU 2 and `p1cr1s0` on GPU 1 are BOTH either starting or already in flight, and BOTH will finish their training + Dev-512/Dev-SVAMP-200 scoring in **~1.3 to 1.5 hours**!
# And look at what is queued AFTER `rpM` (on GPU 2) and AFTER `p1cr1s0` (on GPU 1):
# - On GPU 2: `fullmodel` of `rpL-40` is queued right after `rpM`.
# - On GPU 1: `hold` is queued right after `p1cr1s0`.
# That means the user has ~1.2 hours while `rpM` and `p1cr1s0` run to:
# 1. Remove `rpL-40`'s full test from GPU 2's queue (`rpL-40` is strictly dominated: on `Dev-512` it scored `53.81% saved / +0.43 accuracy`, `-0.93` paired vs `rpE-80`, which is `-0.49` pts *worse* than `rpL-60` and `-0.33` pts *worse* than `rpK-60`!).
# 2. Code the 5-line `--mix-preset hold` in `repair_pairs.py` and queue **`rpN` on top of `rpL-60`** (and/or `rpN-K` on top of `rpK-60`), plus the full tests of `rpM` and `rpN`!

# Let's double check Question 3 ("Path 1: does it still deserve GPU time? ... Given that the repair line works from rpE-80 (itself 3 DPO rounds + a cut-calibration round + a repair from the original), what is the shortest honest "clean recipe" to present, and should Path 1 be redesigned, run as queued, or dropped?"):
# Let's look carefully at every part of Question 3:
# 1. What is the true lineage of `rpE-80` -> `rpK-40` / `rpL-60` / `rpM` / `rpN`?
#    - Round 1–3 (`b1 -> b2 -> b3`): 3 iterative best-of-8 length-preference DPO rounds on the model's own graded answers (`dp13` pool).
#    - Round 4 (`ca1 -> rc1-soup-30`): cut-calibration + DPO round (`57.4% saved / -3.5 accuracy` on the full test).
#    - Round 5 (`rpE-80`): accuracy-only "be-right" DPO repair on `rc1-soup-30`'s own mistakes (`47.0% saved / -0.9 accuracy` on the full test).
#    - Round 6 (`rpK` / `rpL` / `rpM`): typed DPO round on `rpE-80`'s own answers (`"think less"` only on non-counting IFEval and tool calls; `"don't stop early"` + `"be right"` + anti-loop + counting-rule guards on math/code/IFEval), producing:
#      - `rpK-40` (`51.2% / -0.8`),
#      - `rpK-60` (`54.4% / -1.4`),
#      - `rpL-60` (`59.3% / -1.7`),
#      - `rpM-60` (running now, targeting `54–56% / -0.7 to -0.9`), and
#      - Round 7 (`rpN` on `rpL-60`): accuracy-only repair spending ~4–5 points of `rpL-60`'s 9.3-point saving surplus to land at `54–56% / -0.5 to -0.8`.
# 2. What is the **shortest honest "clean recipe" to present** if Path 1 does not replace the `rpE-80` lineage?
#    - Don't pretend `rpE-80` was a 1-round model, and don't present the messy chronological lab notebook of dead ends (MVT, GRPO, COPD, Wanda, etc.)!
#    - Present the winning lineage as a **single unified principle—Iterative Self-Distilled Preference Optimization (On-Policy DPO) in Two Phases**:
#      - **Phase 1 — Iterative Length Compression (`Qwen3-0.6B -> rc1-soup-30`, `57.4% saved / -3.5 accuracy`):** Multi-round DPO on the model's own $K=8$ rollouts preferring shorter correct reasoning traces over longer/wrong traces, which compresses thinking by `57.4%` (`677 -> 219` tokens on math, `2079 -> 732` on code) at the cost of `-3.5` accuracy points from dropped verification checks and shortened IFEval responses.
#      - **Phase 2 — Diagnostic-Guided Typed Self-Repair (`rc1-soup-30 -> rpE-80 -> rpK-40 / rpM-60 / rpL-60 / rpN`, `41.9%` to `59.3%` saved along the Pareto curve):** Using the model's own $K=8$ rollouts on unseen questions, apply a single parameterized pair-building script (`repair_pairs.py`) with four domain-typed pair rules derived from error analysis (no teacher model, no prompt changes):
#        1. *Re-check & Early-Stop Guards (`svamp`, `mbpp`):* reject premature wrong traces (`<0.95x` median right thinking) in favor of length-matched right traces (`0.95–1.25x`) that re-read the problem.
#        2. *Response-Neutral Thinking Compression (`ifeval` non-counting, `bfcl`):* shorten thinking (`>=1.30x` spread) only where thinking has zero accuracy payoff, while holding response length within `+-5%`.
#        3. *Length-Constraint Pairs (`ifeval` minimum-length rules):* on `"at least N words/sentences/keywords"` prompts, prefer correct responses that are `1.10x+15` to `1.75x+60` tokens longer with no extra thinking.
#        4. *Anti-Loop Pairs (`ifeval`, `mbpp`):* prefer finished correct answers over runaway/unclosed generation loops (`301 vs 4,014` tokens).
#    - **And look at the submission's length / accuracy Pareto curve that this exact lineage ALREADY has on the Full Test (`2,759 x 8`)—every single point is a checkpoint from the same `rc1 -> rpE-80 -> rp*` family:**
#      1. **`rpI-120`:** **`41.9% saved / +0.1 accuracy`** (`+1.64` vs curve — net accuracy gain over raw Qwen3-0.6B!)
#      2. **`rpE-80`:** **`47.0% saved / -0.9 accuracy`** (`+1.45` vs curve)
#      3. **`rpK-40`:** **`51.2% saved / -0.8 accuracy`** (`+2.33` vs curve — **inside the goal box!**)
#      4. *(Coming in ~3 hours)* **`rpM-60` and/or `rpN-30`:** **`~54–56% saved / -0.6 to -0.9 accuracy`** (`~ +2.6 to +3.0` vs curve — **deep inside the goal box with margin!**)
#      5. **`rpK-60`:** **`54.4% saved / -1.4 accuracy`** (`+2.38` vs curve)
#      6. **`rpL-60`:** **`59.3% saved / -1.7 accuracy`** (`+3.20` vs curve — extreme 60%-compression anchor!)
#      Look at that curve: `41.9% / +0.1` $\to$ `47.0% / -0.9` $\to$ `51.2% / -0.8` $\to$ `[54-56% / -0.7]` $\to$ `54.4% / -1.4` $\to$ `59.3% / -1.7`. Every checkpoint is fully evaluated on all 2,759 questions $\times 8$, monotonically spanning `42%` to `59%` token savings!
# 3. Given that, **does Path 1 still deserve GPU time? Should it be redesigned, run as queued, or dropped?**
#    - Let's be 100% blunt:
#      - **Do NOT redesign Path 1 again** (no `round1d`).
#      - **`p1cr1s0` (Round 1 v3, ~1.5 GPU-hours on GPU 1) is ALREADY running/queued right now on GPU 1 using existing `p1src` rollouts (0 generation cost). Let `p1cr1s0` finish its 200-step train + Dev-512/Dev-SVAMP score (~1.5h), and then STOP Path 1 unless `p1cr1s0` clears a strict pre-registered gate.**
#      - Wait: what if `p1cr1s0` has NOT started training yet when the user reads this, or what after `p1cr1s0` finishes?
#        - Even if `p1cr1s0` finishes (it only takes ~1.5h because `p1src` rollouts are already on Drive), **do NOT spend 3.9 GPU-hours on Path 1 Round 2 (`p1cr2s0`: 2.2h generation+training + 1.7h full test)** unless `p1cr1s0` unambiguously beats `rpE-80` on BOTH `Dev-512` (`saved >= 48.0%`, `IFEval saved >= 43.0%`, `headline d_acc >= +0.60`) AND `Dev-SVAMP-200` (`math d_acc >= rpE-80 - 0.30`).
#        - Why? Because `p1r1s0-200` scored `43.4% / -1.5` on the full test (with a `-2.5%` Dev-to-Full saving penalty and `-1.6` accuracy penalty on 1-round models from raw Qwen), proving that compressing raw `Qwen3-0.6B` from `0%` to `53%+` in just 1–2 rounds hits a much steeper accuracy wall than our multi-round `rc1 -> rpE-80 -> rpL/rpM/rpN` curriculum. Meanwhile, GPU 1's remaining ~4.5 hours are needed to run **`rpN` (on top of `rpL-60`)** and **full-test the winning `rpM` and `rpN` checkpoints**!

# Let's double check Question 2 in detail:
# "2. The repair strategy from here, in order: rpM as queued; an accuracy-only repair (your rpN) on top of rpL-60 (59.3 / -1.7, 9.3 points of saving to spend) versus on top of rpK-60 (54.4 / -1.4); anything else. Exact settings for the run you would start first, its decision rule, and expected full-test range. Aim: a model inside the box WITH margin (for example 53%+ at -0.6 or better)."

# Let's carefully compare the two options for `rpN` on top of `rpL-60`:
# Wait: earlier we found a critical fact when we simulated `repair_pairs.py` on `rpL-60`'s own rollouts vs `rpE-80`'s rollouts (`rpI`):
# Let's check BOTH ways to build `rpN` on top of `rpL-60`:
# - **Way 1 (`rpN` using `--rp-rollouts-from rpI --rp-extra-questions svamp=1200 --rp-seed 41`):**
#   - Base & Reference Model: `models/rpL-60` (`59.27% saved / -1.69 official, -1.55 finished-only`).
#   - Rollouts:
#     - For `svamp` (`180` pairs): `models/rpL-60` itself answers `1,200` fresh SVAMP-like questions $\times 8$ (`--rp-extra-questions svamp=1200 --rp-samples 8 --rp-seed 41`, ~10 minutes on L4!). Why? Because on SVAMP, our full-test simulation showed `rpL-60` has **195 `guard_wide` and 269 `right_hold` questions per 1,000 SVAMP questions** (even more than `rpE-80`!), and using `rpL-60`'s own fresh math rollouts targets the exact `240`-token skipped-recheck errors that `rpL-60` makes!
#     - For `ifeval` (`240` pairs), `mbpp` (`120` pairs), `bfcl` (`200` pairs), and `svamp_hard` (`60` pairs): reuses `rpI` (`runs/bon/rp-rpI/rollouts.jsonl`, generated by `rpE-80`, the immediate parent of `rpL-60`), with `--rp-dpo-seed 1` (fresh pair shuffle). Why? Because:
#       1. On IFEval `length_rule` (`at_least_N`), `rpL-60` already compressed its responses to `440.5` tokens (`13.58%` accuracy, only 7 pairs $\ge 300$ tokens in 541 questions), whereas `rpI` (`rpE-80`'s rollouts) has **121 `right_count` candidates with mean response `314 vs 217` tokens and 44 pairs $\ge 300$ tokens**—giving `rpL-60` the exact longer-response `"at least N"` demonstrations it needs to recover from `13.6%` to `25–30%+`!
#       2. It saves **~55 minutes of GPU rollout generation time** (`0.85 GPU-hours` total run time instead of `1.85 GPU-hours`!), so `rpN` can run AND be full-tested well within the 12 GPU-hour budget!
# - **Wait: what if `--rp-rollouts-from rpM-extra` or reusing `rpM`'s 1,200 SVAMP rollouts vs `rpL-60` answering `svamp=1200`?**
#   In `track.py` (`code/track_excerpts.py` line 158):
#   When `--rp-base models/rpL-60 --rp-rollouts-from rpI --rp-extra-questions svamp=1200 --rp-seed 41` is run, `track.py` automatically sets `answerer = merged = models/rpL-60` and creates `runs/bon/rp-rpN-extra/rollouts.jsonl` answered by `models/rpL-60`! It works out-of-the-box with ZERO changes to `track.py`!
# - **And what if the user prefers `rpN` where `models/rpL-60` generates ALL 3,200 questions from scratch (`--rp-own-data --rp-public --rp-exclude-dev`, no `--rp-rollouts-from rpI`)?**
#   Let's check! If `models/rpL-60` generates all 3,200 questions from scratch:
#   1. It takes ~55 minutes longer (`1.75 GPU-hours` instead of `0.85 GPU-hours`).
#   2. On `ifeval`, because `rpL-60`'s own `at_least_N` responses are shorter (`261.7` tokens in `right_count` vs `343.9` in `rpE-80`), you would need `ifeval=1100` questions to get enough `right_count` pairs, and their responses would be ~80 tokens shorter than `rpI`'s.
#   Therefore, `--rp-base models/rpL-60 --rp-own-data --rp-rollouts-from rpI --rp-extra-questions svamp=1200 --rp-samples 8 --rp-seed 41` (where `rpL-60` answers the 1,200 SVAMP-like questions on-policy in 10 minutes and reuses `rpI` for IFEval/MBPP+/BFCL/hard-math) is **strictly better on both pair quality and GPU wall-clock time**!

# Let's test building `rpN`'s exact pair mix locally on `rpI`'s summary / `rpM`'s pair table or verify the exact quotas and `--mix-preset hold` code!
# Let's check the candidate counts in `rpM_pairs_local_build_summary.txt` to be 100% sure that NO part in `rpN` will fall short of its quota (`SHORT of quota: 0` on every part)!
# In `rpM_pairs_local_build_summary.txt`:
# - `bfcl_simple_python`: `candidates right_cr = 83`. If we set `bfcl_simple_python=80` (`100% right_cr`), `80 <= 83` -> **0 shortfall!**
# - `bfcl_multiple`: `candidates right_cr = 99`. If we set `bfcl_multiple=80` (`100% right_cr`), `80 <= 99` -> **0 shortfall!**
# - `bfcl_irrelevance`: `candidates right_cr = 59`. If we set `bfcl_irrelevance=40` (`100% right_cr`), `40 <= 59` -> **0 shortfall!**
# - `mbpp`: `candidates right_cr = 430, capguard = 64, loop = 14`. If we set `mbpp=120` with `right=0.75 (90), guard=0.15 (18), loop=0.10 (12), shorten=0.0`, `90 <= 430`, `18 <= 64`, `12 <= 14` -> **0 shortfall!**
# - `ifeval`: `candidates right_count = 121, right_cr = 304, loop = 90`.
#   Wait: look closely at how `repair_pairs.py` enforces `count_share` and `sh_share` on `ifeval`!
#   Let's check lines 290-315 of `repair_pairs.py`!
```
```python
with open(os.path.join(pack_dir, "code/repair_pairs.py"), "r") as f:
    rp_lines = f.read().splitlines()
for i, line in enumerate(rp_lines[210:285], start=211):
    print(f"{i:3d}: {line}")
```