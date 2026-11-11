```python
# LOOK AT THOSE TWO COMPARISONS (`rpL vs rpM IFEval 'right' pairs` and `rpL vs rpM Math pairs`):
# 1. On Math (`svamp`):
#    In `rpL`, mean `chosen_thinking` was **`519.2` vs `rejected_thinking` `871.9` tokens (`chosen > rejected` in only `39.4%` of math pairs—meaning `60.6%` of `rpL`'s math pairs STILL pushed math shorter!)**.
#    In `rpM` (and `rpN`), mean `chosen_thinking` is **`679.4` vs `542.1` tokens (`chosen > rejected` in `91.7%` of math pairs!)** because of `--guard-wide svamp --right-hold svamp` and 0 math `loop` pairs!
# 2. On IFEval (`right` = `right_count` + `right_cr`):
#    In `rpL`, mean `chosen_response` was `149.7` vs `141.0` tokens (`+8.7` tokens), with only **14 pairs >= 300 tokens**!
#    In `rpM` (and `rpN` using `rpI` rollouts), mean `chosen_response` is **`183.5` vs `136.7` tokens (`+46.8` tokens; `314 vs 217` on the 78 `right_count` pairs), with 31 pairs >= 300 tokens**, and 93 of the 136 questions (`68%`) were NEVER in `rpL`!
# 3. And what if `rpN` on top of `rpL-60` uses `--rp-rollouts-from rpI` (with `--rp-extra-questions svamp=1200 --rp-seed 41` answered by `rpL-60` in 10 min, OR reusing `rpM`'s 1,200 SVAMP rollouts) AND `--rp-dpo-seed 1` (so `take()` draws a fresh shuffle of `right_cr` / `capguard` / `loop` candidates from `rpI`'s 4,500 questions!)?
#    Let's check! In `rpI`, there are:
#    - `304` IFEval `right_cr` candidates (only 48 were used in `rpL`!)
#    - `121` IFEval `right_count` `length_rule` candidates (78–96 used in `rpN`!)
#    - `90` IFEval response-loop candidates (24 used in `rpN`!)
#    - `430` MBPP+ `right_cr` candidates and `64` `capguard` candidates!
#    - `83` `bfcl_simple_python`, `99` `bfcl_multiple`, `59` `bfcl_irrelevance` `right_cr` candidates!
#    - Plus 1,200 brand-new SVAMP-like questions (`--rp-extra-questions svamp=1200 --rp-seed 41`) answered directly by `rpL-60` in ~10 minutes!
#    Look at what that achieves:
#    - **Zero rollout-generation wait on MBPP+, IFEval, BFCL** (reusing `rpI`'s 4,500 questions on Drive) + **only ~10 minutes on SVAMP-like math** (`svamp=1200` answered by `rpL-60`).
#    - **Total runtime of `rpN`: ~50 minutes (0.85 GPU-hours)** instead of 1.85 GPU-hours!
#    - **Starts from `rpL-60` (`59.27% saved / -1.69 official, -1.55 finished-only`)**, which has **9.27 points of saving to spend** (more than DOUBLE `rpK-60`'s 4.43 points!).

# Wait: let's also check if we can test weight interpolation (`merge.py`) between our existing full-tested checkpoints right now on our data, OR check what linear / adapter combinations exist!
# Wait: earlier we checked `rpE-80` (`47.04% / -0.95`) + `rpL-60` (`59.27% / -1.69`):
# On a straight secant line between `rpE-80` (`47.04% / -0.95`) and `rpL-60` (`59.27% / -1.69`), crossing `50.0%` saved requires weight $\alpha = (50.0 - 47.04) / (59.27 - 47.04) = 0.242$ on `rpL-60`, where linear accuracy is `-0.95 + 0.242 * (-1.69 - (-0.95)) = -1.13` (outside the box).
# WAIT: what about interpolating **`rpK-40` (`51.17% / -0.77`) + `rpL-60` (`59.27% / -1.69`)** or **`rpI-120` (`41.91% / +0.13`) + `rpL-60` (`59.27% / -1.69`)**?
# Let's check the exact secant math for:
# 1) `rpI-120` (`41.91% / +0.13`) and `rpL-60` (`59.27% / -1.69`):
#    At $\alpha = 0.55$ on `rpL-60` + $0.45$ on `rpI-120`:
#    Linear saving = `0.45 * 41.91 + 0.55 * 59.27 = 51.46%`
#    Linear accuracy = `0.45 * (+0.13) + 0.55 * (-1.69) = -0.87`!
#    At $\alpha = 0.60$ on `rpL-60` + $0.40$ on `rpI-120`:
#    Linear saving = `52.33%`, Linear accuracy = `-0.96`!
#    Wait: both `adapters/rpI-s0/step-120` and `adapters/rpL-s0/step-60` are LoRA adapters trained directly on top of the **EXACT SAME base model (`models/rpE-80`)**!
#    Because both `rpI-120` and `rpL-60` are linear LoRA deltas $\Delta W_{\text{rpI}}$ and $\Delta W_{\text{rpL}}$ on top of `models/rpE-80`:
#    What happens if you merge `models/rpE-80` with `adapters/rpL-s0/step-60:0.75` + `adapters/rpI-s0/step-120:0.40` (a task-arithmetic sum of the shortening adapter $\Delta W_{\text{rpL}}$ and the accuracy-repair adapter $\Delta W_{\text{rpI}}$ on their shared base `rpE-80`), OR train `rpN` directly on top of `rpL-60`?
#    Let's check why training `rpN` directly on top of `rpL-60` is much stronger than blind task-vector addition (because `rpI-120` had no `length_rule` counting pairs and no widened SVAMP guards), while task-vector addition `rpE-80 + 0.80 * rpL-60 + 0.35 * rpI-120` is a 12-minute zero-training Dev-512 check!

# Now let's examine Question 3 in detail:
# "3. Path 1: does it still deserve GPU time? It was meant to be the clean recipe from the original model; its round 1 reached 43.4 / -1.5 on the full test, and its third round 1 (p1cr1s0) is queued. Given that the repair line works from rpE-80 (itself 3 DPO rounds + a cut-calibration round + a repair from the original), what is the shortest honest "clean recipe" to present, and should Path 1 be redesigned, run as queued, or dropped?"

# Let's check the history of `rpE-80` and `Path 1` in `EXPERIMENT_LOG.md`, `plans_and_docs/FINAL_REPORT.md`, `plans_and_docs/THINKINGCAP_FULL_DOSSIER.md`, and `PATH1_FINAL.md`!
print("=== THINKINGCAP_FULL_DOSSIER.md (head) ===")
print(read_file("plans_and_docs/THINKINGCAP_FULL_DOSSIER.md", 6000))
```