### 0. Three-Level Calibration & Exact Diagnosis of Every Previous Run

To hit **more than 60.0% token saving** at **better than -1.00 point of accuracy** on the **Full Test (2,759 questions x 8 answers, official score)** from scratch (`Qwen/Qwen3-0.6B`), we first separate all measurements by level and establish the empirical offsets between them.

#### A. Offsets Across the Three Levels (Monitor vs Dev-512 vs Full Test)
1. **Monitor (`<run>-monitor.jsonl`, model of round `j-1` answering round `j`'s fresh ~780-question slice x 8)**:
   - Across all full-tested rounds of `od5f` (`od5f-monitor.jsonl` lines 15–19 vs `NEW_RESULTS_od5f.md` lines 25–30), `od5k-r19` (`od5k-monitor.jsonl` line 2 vs `CONSULT36_RESULTS.md` line 40), and `od5g-r20` (`od5g-monitor.jsonl` line 3 vs `FINAL_MODEL_od5g_r20.md` line 2):
     - `od5f-r14`: Monitor (slice 15) **55.5% / +1.72** vs Full Test **54.6% / -0.94** (Monitor offset vs Full Test: **+0.9% saving, +2.66 pp accuracy**).
     - `od5f-r15`: Monitor (slice 16) **56.5% / +2.70** vs Full Test **56.1% / -1.09** (Monitor offset vs Full Test: **+0.4% saving, +3.79 pp accuracy**).
     - `od5f-r16`: Monitor (slice 17) **53.9% / +1.85** vs Full Test **55.4% / -0.57** (Monitor offset vs Full Test: **-1.5% saving, +2.42 pp accuracy**).
     - `od5f-r17`: Monitor (slice 18) **54.6% / +1.91** vs Full Test **57.8% / -0.64** (Monitor offset vs Full Test: **-3.2% saving, +2.55 pp accuracy**).
     - `od5f-r18`: Monitor (slice 19) **59.0% / +2.37** (and **58.8% / +1.99** in `od5k-monitor.jsonl` line 1) vs Full Test **59.5% / -0.80** (Monitor offset vs Full Test: **-0.5% to -0.7% saving, +2.79 to +3.17 pp accuracy**).
     - `od5k-r19`: Monitor (slice 20) **57.9% / +1.75** vs Full Test **60.6% / -1.00** (Monitor offset vs Full Test: **-2.7% saving, +2.75 pp accuracy**).
     - `od5g-r20`: Monitor (slice 21) **55.0% / +2.07** vs Full Test **57.4% / +0.14** (Monitor offset vs Full Test: **-2.4% saving, +1.93 pp accuracy**).
   - **Rule for Monitor -> Full Test forecast**: On slices 18, 19, and 20, the Monitor under-reads Full-Test saving by **-0.5% to -3.2%** (because slices 18 and 20 have slightly shorter baseline completions) and over-reads Full-Test accuracy by **+2.55 to +3.17 points** (mean **+2.75 points**).

2. **Dev-512 (`dev512-*_dev_score.json`, fixed 512 questions x 8, official weights)**:
   - For clean 1-round-braked runs (`od5f`, `od5k`, `od5g`, where tool-call caps hold `bfcl_simple_python` and `bfcl_multiple` around 50% saving):
     - `od5f-r14`: Dev-512 **55.4% / +1.67** vs Full Test **54.6% / -0.94** (Dev offset: **+0.8% saving, +2.61 pp accuracy**).
     - `od5f-r15`: Dev-512 **56.3% / +1.72** vs Full Test **56.1% / -1.09** (Dev offset: **+0.2% saving, +2.81 pp accuracy**).
     - `od5f-r16`: Dev-512 **55.3% / +1.72** vs Full Test **55.4% / -0.57** (Dev offset: **-0.1% saving, +2.29 pp accuracy**).
     - `od5f-r17`: Dev-512 **58.0% / +1.83** vs Full Test **57.8% / -0.64** (Dev offset: **+0.2% saving, +2.47 pp accuracy**).
     - `od5f-r18`: Dev-512 **61.5% / +1.49** (training eval, `NEW_RESULTS_od5f.md` line 16) and **60.4% / +0.88** (standalone rescore, `dev512-od5f-r18_dev_score.json`) vs Full Test **59.5% [58.5, 60.4] / -0.80 [-1.53, -0.02]** (Dev offset: **+0.9% to +2.0% saving, +1.68 to +2.29 pp accuracy**).
     - `od5k-r19`: Dev-512 **60.5% [58.2, 62.9] / +1.69 [-0.18, +3.50]** (`CONSULT36_RESULTS.md` line 18, `dev512-od5k-r19_dev_score.json`) vs Full Test **60.6% [59.6, 61.6] / -1.00 [-1.76, -0.28]** (`CONSULT36_RESULTS.md` line 40) (Dev offset: **-0.1% saving, +2.69 pp accuracy**).
     - `od5g-r20`: Dev-512 **58.2% / +3.18** (`FINAL_MODEL_od5g_r20.md` line 25, `dev512-od5g-r20_dev_score.json`) vs Full Test **57.4% [56.2, 58.4] / +0.14 [-0.63, +0.88]** (Dev offset: **+0.8% saving, +3.04 pp accuracy**).
   - **Rule for Dev-512 -> Full Test forecast (on clean `od5f`-style runs)**: Dev-512 reads **-0.1% to +1.0% higher on saving** (mean **+0.4%**) and **+2.29 to +2.81 points higher on accuracy** (mean **+2.55 points**). Therefore, to land at **>60.0% saving** and **better than -1.00 accuracy (i.e. -0.85 to -0.35)** on the Full Test, a round's Dev-512 score must reach **>= 60.5% saving** (ideally **60.8% to 62.2%**) at **>= +1.75 Dev-512 accuracy** (ideally **+1.80 to +2.35**).

---

#### B. What `od5k-r19` vs `od5f-r18` Proved Per Task (And Why `od5k-r19` Landed at 60.6% / -1.00)
Comparing `od5f-r18` (`full-od5f-r18_compare.json` & `CONSULT36_RESULTS.md` lines 47–54) directly against `od5k-r19` (`full-od5k-r19_compare.json` & `CONSULT36_RESULTS.md` lines 40–68) on the **Full Test (2,759 x 8)** shows the exact per-task trade-off produced by `od5k`'s consult36 settings (`--od-caps mbpp=65,ifeval=60,bfcl_irrelevance=72`, `--floor-by-part ifeval=14:0.25+bfcl_irrelevance=10:0.20`, `--rp-quota svamp=48,mbpp=52,ifeval=64,bfcl_simple_python=24,bfcl_multiple=22,bfcl_irrelevance=30`):

- **`svamp` (math, weight ~24% of total accuracy)**:
  - `od5f-r18` Full Test: **71.0% saved / -3.60 pp** (`70.6%` exact in `full-od5f-r18_compare.json`).
  - `od5k-r19` Full Test: **69.7% saved [68.2, 71.1] / -2.94 pp [-4.15, -1.78]** (`CONSULT36_RESULTS.md` line 49).
  - **What happened**: In round 19, `od5f-r18` monitored on slice 19 at `svamp 65.2% / -2.60` (`od5k-monitor.jsonl` line 1), which exceeded `svamp=61` and `-2.0`. Thus `svamp` was held for 1 round in `od5k-r19` (`PAIRS_SUMMARY_all_runs.jsonl`: 19 `be_right_hold` + 25 `recovery` + 4 `loop` = 48 pairs). Holding `svamp` once recovered **+0.66 pp** of Full-Test math accuracy (`-3.60 -> -2.94`) while losing only **0.9 to 1.3 points** of math saving (`70.6% -> 69.7%`)!
  - **Why `svamp` had reached `-3.60` in `od5f-r18` in the first place**: At round 17 (`od5f-r17` Full Test `57.8% / -0.64`), `svamp` was at **67.0% saved / -2.30 pp** (`fulltests.json`). On the round 18 monitor (`od5f-monitor.jsonl` line 18, `models/od5f-r17` on slice 18), `svamp` read **59.5% / -1.26**—just 1.5 points below `od5f`'s `svamp=61` cap! Because `59.5 < 61`, `svamp` was **not** held in round 18 and trained on 56 aggressive shortening pairs (`pairs_od5f_r18.jsonl`), jumping from `67.0% / -2.30` to `71.0% / -3.60` on the Full Test.

- **`mbpp` (code, weight ~29% of total accuracy — the single largest accuracy weight)**:
  - `od5f-r18` Full Test: **56.0% saved / -1.10 pp**.
  - `od5k-r19` Full Test: **58.6% saved [57.3, 59.8] / -1.98 pp [-3.41, -0.57]** (`CONSULT36_RESULTS.md` line 50).
  - **What happened**: `od5k-r19` pushed `mbpp` saving up by **+2.6% (`56.0% -> 58.6%`)**, and `mbpp` **paid -0.88 pp of Full-Test accuracy (`-1.10 -> -1.98`)**, which reduced the weighted overall Full-Test accuracy by **-0.25 pp**—the exact difference between `-0.75` (well inside goal) and `-1.00` (on the boundary)!
  - **Why `mbpp` paid `-0.88 pp` in `od5k`**: On slice 19 (`od5k-monitor.jsonl` line 1), `mbpp` was the **only** part besides `ifeval` and `bfcl_irrelevance` not held by the brake (`mbpp` monitor saving was `59.2% < 65`). It trained on 52 full shortening pairs (`PAIRS_SUMMARY_all_runs.jsonl`: 43 `shorten` + 4 `be_right` + 5 `loop` with median chosen/rejected ratio `0.663` and floor `35:0.36`). Even worse, as `CONSULT36_RESULTS.md` line 59 notes: in `od5k` round 20 (`od5k-monitor.jsonl` line 2), `mbpp` monitor accuracy crashed to **-2.25** (and **-5.36** in round 21), triggering the `-2.0` accuracy brake and collapsing `mbpp` saving to `48.8%` (r21) and `32.5%` (r22). Raising `mbpp` cap from `63` to `65` (and keeping `mbpp` quota at `52` with floor `35:0.36` past `56%` Full-Test saving) pushed code past its safe boundary.

- **`ifeval` (instructions) and `bfcl_irrelevance` (no function fits)**:
  - `ifeval` Full Test: `od5f-r18` **56.0% / +1.40 pp** -> `od5k-r19` **58.2% [55.7, 60.7] / +0.67 pp [-0.95, +2.34]** (+2.2% saving, accuracy still **+0.67 pp above the original base model**).
  - `bfcl_irrelevance` Full Test: `od5f-r18` **59.0% / +5.50 pp** -> `od5k-r19` **61.0% [58.3, 63.7] / +5.52 pp [+3.05, +7.94]** (+2.0% saving, **+0.02 pp accuracy gain — 100% free token saving**).
  - **Why these two tasks bought the saving cleanly**: In `od5f`, `ifeval` was capped at `56` (with floor `18:0.30`) and `bfcl_irrelevance` at `64` (with floor `16:0.28`). Inspecting `PAIRS_SUMMARY_all_runs.jsonl` for `od5f` across all 18 rounds shows that `ifeval` and `bfcl_irrelevance` were held by their caps in round 11 (`od5f` r11 monitor: `ifeval 56.6% >= 56`, `bfcl_irrelevance 66.5% >= 64`), where `ifeval` received 60 `be_right_hold`/`recovery` pairs (`chosen_over_rejected_median = 0.99 to 1.07`), pulling `ifeval` saving back down from `56.6%` to `48.2%` by round 13, and `ifeval` was held again in round 16 (`55.7%`). Simply raising `--od-caps` on `ifeval` (`56 -> 60`) and `bfcl_irrelevance` (`64 -> 72`) and lowering `--floor-by-part` on `ifeval` (`18:0.30 -> 14:0.25`) and `bfcl_irrelevance` (`16:0.28 -> 10:0.20`) from round 1 prevents those mid-run hold pullbacks on `ifeval` and `bfcl_irrelevance` and unlocks `+2.5% to +4.0%` more saving on those two high-padding tasks (`66%` and `91%` post-settlement padding, `EXPERIMENT_LOG.md` line 6763).

- **`bfcl_simple_python` (one function) and `bfcl_multiple` (several functions)**:
  - `bfcl_simple_python` Full Test: `od5f-r18` **50.0% / -2.40 pp** -> `od5k-r19` **50.8% [48.1, 53.6] / -2.19 pp [-4.25, -0.09]** (+0.8% saving, **+0.21 pp accuracy gain**).
  - `bfcl_multiple` Full Test: `od5f-r18` **50.0% / -1.00 pp** -> `od5k-r19` **51.8% [49.4, 54.1] / -1.25 pp [-3.09, +0.47]** (+1.8% saving, -0.25 pp accuracy).
  - Both tool-call tasks stayed rock-solid around `50–52%` saving because `od5f` and `od5k` kept `--od-caps bfcl_simple_python=45,bfcl_multiple=53` and did **not** use `--od5-held` or `--od-smooth 2`.

---

#### C. Verification of Training Slices (`od3` vs `od3g`) and Round Schedule From Scratch
- By direct file comparison in `training_data/consult37_data/`, `od3-rK-questions.jsonl` and `od3g-rK-questions.jsonl` are **byte-for-byte identical for every round `K = 2, 3, ..., 19`** (identical file sizes and contents; round 1 uses `od3-r1-stored` in both). Only rounds 20, 21, and 22 differ (`od3g` has the test-matched medium code and math slices for rounds 20–22, `NEW_RESULTS_od5f.md` lines 69–72).
- Furthermore, why did `od5f`, `od5g`, and `od5k` all lose saving after round 18 or 19 (`od5f-r19` monitor `47.8%`, `od5g-r20` monitor `52.1%`, `od5k-r20` monitor `53.6%`)? Because on slice 19 (`od3-r19-questions.jsonl`), 4 to 5 tasks simultaneously cross their saving caps (`svamp`, `bfcl_simple_python`, `bfcl_multiple`, `bfcl_irrelevance`, and in `od5f`/`od5g` `ifeval`), so round 19 becomes a mass-hold round that pulls saving down, and in round 20 `mbpp` crosses `-2.0` monitor accuracy and gets pulled back to `32–48%` saving.
- **Conclusion on Schedule and Slices**: An **18-round schedule on `od3` slices (rounds 1..18, identical to `od5f`)** is the proven optimal horizon from scratch: it peaks cleanly at round 18 (with round 17 as a secondary checkpoint) right before slice 19 triggers multi-task cap holds.

---

### 1. Exact Runner Lines for Variations A and B (From Scratch)

Both variations start **from scratch** from `Qwen/Qwen3-0.6B`, run **18 rounds** on `od5f`'s exact `od3` slices (`data/od3-rK-questions.jsonl`), keep `--od-dev-rounds 14,15,16,17,18` so Dev-512 and Dev-SVAMP are automatically scored after rounds 14, 15, 16, 17, and 18, and use `fullpick` to gate and run the Full Test. Only flags that already exist in `OD5F_BEST_LINE.txt` are used; nothing is invented.

#### Variation A (`od5m`): Minimal Change From `od5f` — Release `ifeval` and `bfcl_irrelevance` Caps & Floors While Slightly Tightening `svamp` Cap (`61 -> 59`)
Variation A is `od5f`'s exact recipe (`OD5F_BEST_LINE.txt` line 1) with **only `--od-caps` and `--floor-by-part` modified** (all quotas, steps, ratios, seeds, and slices 100% identical to `od5f`):
- It unlocks `ifeval` (cap `56 -> 60`, floor `18:0.30 -> 14:0.25`) and `bfcl_irrelevance` (cap `64 -> 72`, floor `16:0.28 -> 10:0.20`)—the exact two changes that gave `od5k-r19` `+2.2%` on `ifeval` and `+2.0%` on `bfcl_irrelevance` at zero accuracy cost.
- It **keeps `mbpp` untouched** at `od5f`'s cap (`63`) and floor (`35:0.36`) (unlike `od5k` which raised `mbpp` cap to `65` and paid `-0.88 pp` on code).
- It lowers `svamp` cap by 2 points (`61 -> 59`) so that on round 18's slice (where `od5f-r17` monitored at `svamp 59.5% / -1.26` and slipped past `61` in `od5f`, jumping to `71% / -3.60`), `svamp` is held at round 18 and settles at `~68–69% / -2.3 to -2.8` (saving `+0.8 to +1.2 pp` of math accuracy, just as holding `svamp` in `od5k-r19` recovered `+0.66 pp`).

**Runner lines for GPU 1 (Variation A — `od5m`)**:

odpo --name od5m --od-fixed-ref --rp-base Qwen/Qwen3-0.6B --od-rounds 18 --od-steps 15 --od-r1-steps 20 --od-r1-quota svamp=80,mbpp=56,ifeval=88,bfcl_simple_python=36,bfcl_multiple=28,bfcl_irrelevance=32 --rp-quota svamp=56,mbpp=52,ifeval=60,bfcl_simple_python=24,bfcl_multiple=22,bfcl_irrelevance=26 --rp-questions svamp=220,mbpp=140,ifeval=200,bfcl_simple_python=80,bfcl_multiple=60,bfcl_irrelevance=80 --rp-own-data --rp-kinds mix --rp-samples 8 --rp-seed 18 --rp-pair-args=--think-only,svamp+mbpp+ifeval+bfcl_simple_python+bfcl_multiple+bfcl_irrelevance,--prefer-reread,--count-rules,--shorten-ratio,1.30,--shorten-ratio-by-part,ifeval=1.25+bfcl_irrelevance=1.25,--shorten-second,ifeval+bfcl_irrelevance,--shorten-min-right,0.85,--mix-preset,od5,--od5-v2,--base-stats,data/od3-base-stats.json,--base-resp,data/od3-base-resp.json,--base-think,data/od3-base-think.json,--floor-by-part,svamp=30:0.45+mbpp=35:0.36+ifeval=14:0.25+bfcl_simple_python=25:0.44+bfcl_multiple=22:0.38+bfcl_irrelevance=10:0.20,--count-mode,atleast,--count-resp-min,1.10,--count-resp-cap,1.75,--count-resp-pad,60,--resp-median,0.95,--resp-max,1.05,--right-resp-cap,1.15 --rp-lr 1.4e-5 --rp-warmup 2 --rp-beta 0.15 --rp-sft 0.0 --rp-lora-r 16 --rp-lora-alpha 32 --rp-max-len 4096 --rp-grad-accum 8 --rp-epochs 3 --od-caps svamp=59,mbpp=63,ifeval=60,bfcl_simple_python=45,bfcl_multiple=53,bfcl_irrelevance=72 --od-acc-hold -2.0 --od-unlatch --od-stop-save 70.0 --od-stop-acc 0.0 --od-conv-min 65.0 --od-conv-gain 0.1 --od-dev-rounds 14,15,16,17,18

fullpick --name od5m --fp-candidates models/od5m-r18,models/od5m-r17,models/od5m-r16,models/od5m-r15,models/od5m-r14,models/od5m-best --fp-min-save 59.5 --fp-min-acc -1.0 --fp-max-full 2

---

#### Variation B (`od5n`): Rebalance Per-Round Quotas Away From `svamp` & `mbpp` Into `ifeval` & `bfcl_irrelevance` + Protect `mbpp` Floor (`35:0.36 -> 36:0.38`) and Cap (`63 -> 60`)
Variation B attacks the problem through a **second, orthogonal mechanism** across all 18 rounds: **per-round pair quota reallocation (`--rp-quota`) combined with explicit `mbpp` (code) shortening protection**:
- Why a different mechanism from Variation A: In Variation A, per-round quotas (`svamp=56,mbpp=52,ifeval=60,bfcl_simple_python=24,bfcl_multiple=22,bfcl_irrelevance=26`) are unchanged, relying on the `svamp=59` cap brake in later rounds to stop math from over-shortening. However, whenever a task hits its cap under `--od-unlatch`, standard `od5` hold feeds `be_right_hold` + `recovery` pairs on 100% of its quota for that round, creating discrete sawtooth oscillations (`od5f-monitor.jsonl`: r11 `60.6%` -> r13 `48.3%` -> r18 `59.0%`).
- Instead of relying on a tighter `svamp` cap (`svamp` cap stays at `od5f`'s `61`), Variation B **smoothly slows `svamp` and `mbpp` shortening across all rounds 2..18** by shifting 4 pairs/round from `svamp` (`56 -> 52`) and 4 pairs/round from `mbpp` (`52 -> 48`) directly into `ifeval` (`60 -> 64`) and `bfcl_irrelevance` (`26 -> 30`) (total remains `240` pairs/round), while protecting `mbpp` with cap `63 -> 60` and floor `35:0.36 -> 36:0.38` so code stops shortening at `~54–55%` Full-Test saving (`-0.6 to -0.9 pp` accuracy, vs `-1.10` in `od5f-r18` and `-1.98` in `od5k-r19`).

**Runner lines for GPU 2 (Variation B — `od5n`)**:

odpo --name od5n --od-fixed-ref --rp-base Qwen/Qwen3-0.6B --od-rounds 18 --od-steps 15 --od-r1-steps 20 --od-r1-quota svamp=80,mbpp=56,ifeval=88,bfcl_simple_python=36,bfcl_multiple=28,bfcl_irrelevance=32 --rp-quota svamp=52,mbpp=48,ifeval=64,bfcl_simple_python=24,bfcl_multiple=22,bfcl_irrelevance=30 --rp-questions svamp=220,mbpp=140,ifeval=200,bfcl_simple_python=80,bfcl_multiple=60,bfcl_irrelevance=80 --rp-own-data --rp-kinds mix --rp-samples 8 --rp-seed 18 --rp-pair-args=--think-only,svamp+mbpp+ifeval+bfcl_simple_python+bfcl_multiple+bfcl_irrelevance,--prefer-reread,--count-rules,--shorten-ratio,1.30,--shorten-ratio-by-part,ifeval=1.25+bfcl_irrelevance=1.25,--shorten-second,ifeval+bfcl_irrelevance,--shorten-min-right,0.85,--mix-preset,od5,--od5-v2,--base-stats,data/od3-base-stats.json,--base-resp,data/od3-base-resp.json,--base-think,data/od3-base-think.json,--floor-by-part,svamp=30:0.45+mbpp=36:0.38+ifeval=14:0.25+bfcl_simple_python=25:0.44+bfcl_multiple=22:0.38+bfcl_irrelevance=10:0.20,--count-mode,atleast,--count-resp-min,1.10,--count-resp-cap,1.75,--count-resp-pad,60,--resp-median,0.95,--resp-max,1.05,--right-resp-cap,1.15 --rp-lr 1.4e-5 --rp-warmup 2 --rp-beta 0.15 --rp-sft 0.0 --rp-lora-r 16 --rp-lora-alpha 32 --rp-max-len 4096 --rp-grad-accum 8 --rp-epochs 3 --od-caps svamp=61,mbpp=60,ifeval=60,bfcl_simple_python=45,bfcl_multiple=53,bfcl_irrelevance=72 --od-acc-hold -2.0 --od-unlatch --od-stop-save 70.0 --od-stop-acc 0.0 --od-conv-min 65.0 --od-conv-gain 0.1 --od-dev-rounds 14,15,16,17,18

fullpick --name od5n --fp-candidates models/od5n-r18,models/od5n-r17,models/od5n-r16,models/od5n-r15,models/od5n-r14,models/od5n-best --fp-min-save 59.5 --fp-min-acc -1.0 --fp-max-full 2

---

### 2. Every Setting That Differs From `od5f`'s Line (`OD5F_BEST_LINE.txt`)

Below is the item-by-item comparison of every modified parameter relative to `OD5F_BEST_LINE.txt` line 1, with exact file, line, and quoted numbers.

#### A. Differences in Variation A (`od5m`) vs `od5f` (`OD5F_BEST_LINE.txt` line 1)

1. **`--od-caps`**:
   - **Old value (`od5f`, `OD5F_BEST_LINE.txt` line 1)**: `svamp=61,mbpp=63,ifeval=56,bfcl_simple_python=45,bfcl_multiple=53,bfcl_irrelevance=64`
   - **New value (`od5m`)**: `svamp=59,mbpp=63,ifeval=60,bfcl_simple_python=45,bfcl_multiple=53,bfcl_irrelevance=72`
   - **Sub-change 1a (`ifeval` cap `56 -> 60` and `bfcl_irrelevance` cap `64 -> 72`)**:
     - **Data & Quoted Numbers**:
       - `NEW_RESULTS_od5f.md`