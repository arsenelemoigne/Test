## 3. Final Plan (`od5j` + `fullpick`)

*(Correction to Sections 1–2: All `od5f` baseline values below are copied strictly from `OD5F_BEST_LINE.txt` line 1 and `WHAT_WE_RAN.md` lines 5–16: `--od-r1-quota svamp=80,mbpp=56,ifeval=88,bfcl_simple_python=36,bfcl_multiple=28,bfcl_irrelevance=32`; `--rp-quota svamp=56,mbpp=52,ifeval=60,bfcl_simple_python=24,bfcl_multiple=22,bfcl_irrelevance=26`; `--od-caps svamp=61,mbpp=63,ifeval=56,bfcl_simple_python=45,bfcl_multiple=53,bfcl_irrelevance=64`; `--floor-by-part svamp=30:0.45+mbpp=35:0.36+ifeval=18:0.30+bfcl_simple_python=25:0.44+bfcl_multiple=22:0.38+bfcl_irrelevance=16:0.28`; `--shorten-ratio 1.30`; `--shorten-ratio-by-part ifeval=1.25+bfcl_irrelevance=1.25`; `--shorten-second ifeval+bfcl_irrelevance`.)*

---

### 1. Exact `odpo` Line for ONE New Run (`od5j`) from `Qwen/Qwen3-0.6B`

```text
odpo --name od5j --od-fixed-ref --rp-base Qwen/Qwen3-0.6B --od-rounds 18 --od-steps 15 --od-r1-steps 20 --od-r1-quota svamp=80,mbpp=56,ifeval=88,bfcl_simple_python=36,bfcl_multiple=28,bfcl_irrelevance=32 --rp-quota svamp=56,mbpp=52,ifeval=60,bfcl_simple_python=24,bfcl_multiple=22,bfcl_irrelevance=26 --rp-questions svamp=220,mbpp=140,ifeval=200,bfcl_simple_python=80,bfcl_multiple=60,bfcl_irrelevance=80 --rp-own-data --rp-kinds mix --rp-samples 8 --rp-seed 18 --rp-pair-args=--think-only,svamp+mbpp+ifeval+bfcl_simple_python+bfcl_multiple+bfcl_irrelevance,--prefer-reread,--count-rules,--shorten-ratio,1.30,--shorten-ratio-by-part,ifeval=1.22+bfcl_irrelevance=1.22,--shorten-second,ifeval+bfcl_irrelevance,--shorten-min-right,0.85,--mix-preset,od5,--od5-v2,--od5-held,--base-stats,data/od3-base-stats.json,--base-resp,data/od3-base-resp.json,--base-think,data/od3-base-think.json,--floor-by-part,svamp=32:0.47+mbpp=35:0.36+ifeval=16:0.28+bfcl_simple_python=25:0.44+bfcl_multiple=22:0.38+bfcl_irrelevance=12:0.22,--count-mode,atleast,--count-resp-min,1.10,--count-resp-cap,1.75,--count-resp-pad,60,--resp-median,0.95,--resp-max,1.05,--right-resp-cap,1.15 --rp-lr 1.4e-5 --rp-warmup 2 --rp-beta 0.1 --rp-dpo-seed 0 --rp-dev-vs rpE80 --od-dev-rounds 99 --data-seed 18 --od-caps svamp=60,mbpp=62,ifeval=59,bfcl_simple_python=45,bfcl_multiple=53,bfcl_irrelevance=70 --od-acc-hold -2.0 --od-unlatch --od-stop-save 999 --od-conv-gain -999 --od-min-rounds 18 --od-best-name od5j-best --note consult35
```

#### Design Decisions Required by Task
1. **Brake**: **Keep `od5f`'s 1-round brake (`--od-acc-hold -2.0`, NO `--od-smooth 2`)**. In `od5f`, the 1-round brake held math and code immediately after `r2` (`34.2 / -1.66`), recovering `r3` to `34.0 / -0.11` (`OD5F_ROUNDS_1_19_VS_AFTER.md` lines 11–12). In `od5h`, `--od-smooth 2` delayed holding math until `r4`, letting math accuracy drop to `-3.4` (`r11`) and `-3.0` (`r12`).
2. **`--od5-held` vs `--od5-v2`**: **Keep `--od5-held` WITH the Consult 33 fix** (`WHAT_WE_RAN.md` line 8: undamaged hold window `0.90–1.05 @1.00`, damaged `0.95–1.10 @1.03`, median loop answer, recovery `55%/25%`). Why: With the Consult 33 patch, `--od5-held` is strictly length-neutral on held parts (preventing the `0.50–1.00` shortening bug of `od5h`) while increasing accuracy recovery pairs to `55%` when a slice is damaged (`WHAT_WE_RAN.md` lines 8, 19–20), directly helping `svamp` and `mbpp` recover accuracy during held rounds without losing saving.
3. **Question Slices & Number of Rounds**: **Original `od3` question sets (no `--od3-branch od3g`) for `18` rounds (`--od-rounds 18 --od-min-rounds 18`)**. Why: `OD5F_ROUNDS_1_19_VS_AFTER.md` lines 4, 29, 39 show `od5f` achieved its peak `59.5 / -0.80` at round 18 on the original `od3` sets (`r19` monitor `59.0 / +2.37`), whereas continuing to `r20` on old slices dropped monitor saving to `47.8 / +2.90` (line 36), and switching to `od3g` (`od5g-r20`) dropped full-test saving by `2.1` points to `57.4 / +0.14` (line 38).
4. **Unchanged Core Flags from `OD5F_BEST_LINE.txt`**: Online DPO, LoRA r16, constant `--rp-lr 1.4e-5`, `--rp-beta 0.1`, `--od-steps 15`, `--od-r1-steps 20`, `--od-r1-quota svamp=80,mbpp=56,ifeval=88,bfcl_simple_python=36,bfcl_multiple=28,bfcl_irrelevance=32`, `--rp-quota svamp=56,mbpp=52,ifeval=60,bfcl_simple_python=24,bfcl_multiple=22,bfcl_irrelevance=26`, `--shorten-second ifeval+bfcl_irrelevance`.

#### Exact Flag-by-Flag Changes vs `OD5F_BEST_LINE.txt`

| Flag | Old Value (`od5f`) | New Value (`od5j`) | Why & Evidence | Expected Effect on Saving / Accuracy |
|---|---|---|---|---|
| `--name` / `--od-best-name` / `--note` | `od5f` / `od5f-best` / `consult28` | `od5j` / `od5j-best` / `consult35` | Run identifiers. | None. |
| `--rp-pair-args` (`--od5-held`) | `--od5-v2` only (no `--od5-held`) | `--od5-v2,--od5-held` (with Consult 33 patch) | `WHAT_WE_RAN.md` lines 8, 19–20: patched `--od5-held` keeps held parts length-neutral (`0.90–1.05 @1.00` / `0.95–1.10 @1.03`), uses median loop answers, and boosts recovery share to `55%` when damaged (`25%` undamaged). | **+0.2 to +0.4 pts accuracy** on held parts (`svamp`, `mbpp`) with **0% saving loss** during holds. |
| `--shorten-ratio-by-part` | `ifeval=1.25+bfcl_irrelevance=1.25` | `ifeval=1.22+bfcl_irrelevance=1.22` | `OD5F_ROUNDS_1_19_VS_AFTER.md` line 30 & `WHAT_WE_RAN.md` line 14: on `od5f-r18`, `IFEval` (`56 / +1.4`) and `no-fit` (`59 / +5.5`) have strong positive accuracy margins; `od5h` (`r12` monitor: `no-fit 60/+5.0, IFEval 59/+3.5`) showed `1.22` on `ifeval` raises saving to `59%` while keeping `+3.5` monitor accuracy. Keeps `mbpp` and `svamp` at `1.30`. | **+0.5 to +0.8% overall saving** from `ifeval` and `bfcl_irrelevance` with negligible accuracy impact. |
| `--floor-by-part` | `svamp=30:0.45+mbpp=35:0.36+ifeval=18:0.30+bfcl_simple_python=25:0.44+bfcl_multiple=22:0.38+bfcl_irrelevance=16:0.28` | `svamp=32:0.47+mbpp=35:0.36+ifeval=16:0.28+bfcl_simple_python=25:0.44+bfcl_multiple=22:0.38+bfcl_irrelevance=12:0.22` | `OD5F_ROUNDS_1_19_VS_AFTER.md` lines 30, 32: `od5f-r18` math was `71 / -3.6` on full test (`63.9 / -2.56` on Dev-SVAMP-200) because math shortened too far. Raising `svamp` floor slightly (`30:0.45 -> 32:0.47`) protects math accuracy while keeping `mbpp=35:0.36`, `simple=25:0.44`, and `multi=22:0.38` untouched and trimming `ifeval` (`18:0.30 -> 16:0.28`) and `bfcl_irrelevance` (`16:0.28 -> 12:0.22`) where accuracy is `+1.4` and `+5.5`. | **+0.3 to +0.5 pts overall accuracy** (by curbing math tail-shortening) and **+0.3% saving** on `ifeval`/`no-fit`. |
| `--od-caps` | `svamp=61,mbpp=63,ifeval=56,bfcl_simple_python=45,bfcl_multiple=53,bfcl_irrelevance=64` | `svamp=60,mbpp=62,ifeval=59,bfcl_simple_python=45,bfcl_multiple=53,bfcl_irrelevance=70` | `OD5F_ROUNDS_1_19_VS_AFTER.md` line 30 & `WHAT_WE_RAN.md` line 9: `od5f-r18` lost accuracy on math (`71 / -3.6`) and code (`56 / -1.1`) while gaining accuracy on `IFEval` (`56 / +1.4`) and `no-fit` (`59 / +5.5`). Tightening `svamp` (`61 -> 60`) and `mbpp` (`63 -> 62`) caps prevents late-round over-shortening on math/code, while raising `ifeval` (`56 -> 59`) and `bfcl_irrelevance` (`64 -> 70`) shifts the saving budget onto the two positive-accuracy parts. | **Net +0.5 to +1.0% overall saving** (targeting `>= 60.0%` full test) while improving full-test accuracy from `-0.80` toward **`-0.3 to +0.0`**. |

---

### 2. Exact `fullpick` Line and Gating Rules

*(Note: Because the exact CLI flag syntax of the `fullpick` command in `OD5I_NOW_LINES.txt` was not included in the prompt pack, the exact `fullpick` CLI flag names are `UNVERIFIED`; use the exact `fullpick` wrapper syntax from `OD5I_NOW_LINES.txt` with the parameters below.)*

```text
fullpick --name od5j --rounds 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18 --min-mon-save 50.0 --min-mon-acc 0.0 --max-dev 8 --min-dev-save 58.0 --min-dev-acc 1.25 --max-full 2
```

- **Stage 1 (Monitor Filter -> Dev-512, at most 8 checkpoints)**:
  - Select up to **8** rounds from `od5j` whose next-slice monitor satisfies **`saved >= 50.0%` AND `accuracy >= +0.00`** (ranked by monitor saving descending, or latest rounds first among ties).
  - Run `Dev-512` evaluation on those `<= 8` checkpoints.
- **Stage 2 (Dev-512 Gate -> Full Test `2,759 x 8`, at most 2 checkpoints)**:
  - Gate strictly at **`Dev-512 saved >= 58.0%` AND `Dev-512 accuracy >= +1.25`**.
  - Why these exact gates: On `od5f` (`OD5F_ROUNDS_1_19_VS_AFTER.md` lines 25–31), `r17` had Dev-512 `58.0 / +1.83` (-> Full `57.8 / -0.64`) and `r18` had Dev-512 `61.5 / +1.49` (-> Full `59.5 / -0.80`). Since Dev-512 reads `~2.0–2.3` points optimistic on accuracy and `~0.2–2.0` points optimistic on saving, gating at `Dev >= 58.0%` and `Dev >= +1.25` guarantees we only spend full-test compute on checkpoints projected at `>= 57.5–60.5%` full saving and `>= -1.0` full accuracy.
  - Pick at most **2** passing checkpoints (the highest-saving checkpoint passing `+1.25`, and the highest-accuracy checkpoint above `60.0%` Dev saving) for the full `2,759 x 8` test.

---

### 3. Does `od5h` at Round 12 (`60.2 / +1.04`) Change Our Plan?

**No — it does NOT change our plan to run `od5j`, though we should let the already-running `od5h` finish or evaluate `od5h-r12` on Dev-512 only if GPU time is free.**

Why `od5h-r12` (`60.2 / +1.04` on the `r13` slice monitor) is still inferior to `od5f-r18` (`59.0 / +2.37` monitor -> `59.5 / -0.80` full test):
1. **Monitor Optimism (`+1.04` monitor implies `~ -1.5 to -2.0` full-test accuracy)**: `OD5F_ROUNDS_1_19_VS_AFTER.md` line 6 states: *"Monitor = model of round k-1 on fresh slice k; it reads about 2.5-3 points optimistic on accuracy vs the full test."* Indeed, `od5f-r18` had a monitor of `59.0 / +2.37` (`r19` monitor, line 21) and scored `59.5 / -0.80` on the full test (`-3.17` pts lower than its monitor). With a monitor of only `+1.04` (`1.33` pts below `od5f-r18`'s `+2.37`), `od5h-r12` is projected to land around **`-1.5 to -2.1` accuracy** on the full test.
2. **Math Is Already Heavily Damaged on `od5h`**: On the `r11` and `r12` monitors, `od5h` math is at `56 / -3.4` and `62 / -3.0` (on optimistic slice monitors), because `od5h` runs the unpatched `--od5-held` (`0.50–1.00` shortest-thinking selection) and looser floors.
3. **Rounds 19–22 of `od5h` Use Depleted `od3g` Slices**: As shown by `od5g-r20` (`OD5F_ROUNDS_1_19_VS_AFTER.md` line 38), running into rounds 19–22 on `od3g` loses saving (`57.4%`).
- **Action on `od5h` / `od5i`**: Cancel `od5i` (if not started) and replace it with `od5j`. If `od5h-r12` is already saved on disk, you may include `od5h-r12` in a Dev-512 check, but **do not run a full test on any `od5h` checkpoint unless it passes `Dev-512 >= 58.0%` saved and `>= +1.50` accuracy**.

---

### 4. The 3 Biggest Risks

1. **Monitor-to-Full-Test Accuracy Gap (`~2.5–3.2` pts) and Dev-512-to-Full-Test Gap (`~2.2–2.3` pts)**:
   - *Evidence*: `od5f-r18` dropped from `+2.37` (r19 monitor) and `+1.49` (Dev-512) to `-0.80` on the full test (`OD5F_ROUNDS_1_19_VS_AFTER.md` lines 21, 29, 30).
   - *Mitigation*: Require `monitor >= +0.0` for Dev-512 and `Dev-512 >= +1.25` for full test, and tighten `svamp` cap (`61 -> 60`) and floor (`30:0.45 -> 32:0.47`) where `od5f-r18` lost `-3.6` pts.
2. **Round-to-Round Oscillation in Online DPO Rounds 10–18**:
   - *Evidence*: In `od5f`, monitor saving oscillated between `48.3%` (`r12`) and `60.6%` (`r10`) across rounds 10–19 (`OD5F_ROUNDS_1_19_VS_AFTER.md` lines 16–21). A single final round can land in a trough.
   - *Mitigation*: Evaluate up to 8 checkpoints on `Dev-512` across rounds 10–18 via `fullpick` rather than relying solely on round 18.
3. **Math (`svamp`) and Tool-Calling (`bfcl_simple_python`) Accuracy Sensitivity**:
   - *Evidence*: On `od5f-r18` full test, `math` was `71 / -3.6` and `one function` was `50 / -2.4`.
   - *Mitigation*: Keep `bfcl_simple_python` floor at `25:0.44` and cap at `45`, raise `svamp` floor to `32:0.47`, lower `svamp` cap to `60`, and use the patched `--od5-held` (`55%` recovery pairs when damaged) with the immediate 1-round `-2.0` brake.