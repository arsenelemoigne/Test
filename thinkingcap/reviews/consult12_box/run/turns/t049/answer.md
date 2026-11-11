```python
# LOOK AT `rpM_pairs_local_build_summary.txt`:
# Look at the candidate counts inside `rpI` (`runs/bon/rp-rpI/rollouts.jsonl`, already on Drive!):
# - `bfcl_simple_python`: candidates `right_cr = 83`!
# - `bfcl_multiple`: candidates `right_cr = 99`!
# - `bfcl_irrelevance`: candidates `right_cr = 59`!
# - `ifeval`: candidates `right_count = 121`, `right_cr = 304`, `loop = 90`!
# - `mbpp`: candidates `right_cr = 430`, `capguard = 64`, `loop = 14`!
# - `svamp_hard`: candidates `capguard = 598`, `right_cr = 739`!
# AND when `rpM` finishes its 1,200 SVAMP-like rollouts (`runs/bon/rp-rpM-extra/rollouts.jsonl`), or when `rpN` generates its 1,200 SVAMP-like rollouts (`--rp-extra-questions svamp=1200`), there are hundreds of `svamp` `capguard` and `right_cr` candidates too!

# WAIT A SECOND!
# Look at those candidate counts in `rpI`:
# In `rpI`, we ALREADY have:
# - `83` `bfcl_simple_python` `right_cr` candidates (enough for a quota of `80` `right_cr` pairs!)
# - `99` `bfcl_multiple` `right_cr` candidates (enough for a quota of `80` `right_cr` pairs!)
# - `59` `bfcl_irrelevance` `right_cr` candidates (enough for a quota of `40` `right_cr` pairs!)
# - `121` `ifeval` `right_count` candidates, `304` `right_cr` candidates, `90` response-loop candidates (enough for `240` `ifeval` pairs: `96` `right_count`, `120` `right_cr`, `24` `loop`, `0` `shorten`!)
# - `430` `mbpp` `right_cr` candidates, `64` `capguard` candidates, `14` `loop` candidates (enough for `120` `mbpp` pairs: `90` `right_cr`, `18` `capguard`, `12` `loop`, `0` `shorten`!)
# - Plus `--rp-extra-questions svamp=1200` (answered in ~10 minutes by `models/rpL-60`, plus `svamp_hard=60` from `rpI`, giving `svamp=180,svamp_hard=60` = `240` math pairs: `120` widened `capguard` + `120` `right_hold`!).
#
# HOLD EVERYTHING! Let's check how fast `rpN` runs if it uses `--rp-rollouts-from rpI --rp-extra-questions svamp=1200`!
# - Generating `svamp=1200` with `models/rpL-60` (`240` tokens/answer): **~10 minutes**!
# - Building `840` pairs (`svamp=180,svamp_hard=60,mbpp=120,ifeval=240,bfcl_simple_python=80,bfcl_multiple=80,bfcl_irrelevance=40`): **~1 minute**!
# - Training `45 steps` (`lr=1e-5`, batch 16, `0.86` epochs on 840 pairs): **~12 minutes**!
# - Scoring `steps 15, 30, 45` on `Dev-512` + `Dev-SVAMP-200` (paired vs `rpL-60`): **~28 minutes**!
# **TOTAL WALL-CLOCK TIME FOR `rpN`: ~50 MINUTES (~0.85 GPU-HOURS)!**
# Compare that to `1.85 GPU-hours` if `rpL-60` had to generate all 3,200 questions across MBPP+ and IFEval from scratch!
# Not only does `--rp-rollouts-from rpI --rp-extra-questions svamp=1200` cut `rpN`'s runtime in HALF (from `1.85h` to **`0.85h`**), it ALSO solves `rpL-60`'s `at_least_N` rollout-starvation problem (because `rpI` has 121 `right_count` length-rule candidates with `314 vs 217` response tokens, whereas `rpL-60`'s own rollouts have compressed `at_least_N` responses)!
# AND on SVAMP (`svamp=180`), because `--rp-extra-questions svamp=1200` is answered by `answerer = models/rpL-60` itself (with `--rp-seed 41` so it draws 1,200 unseen SVAMP-like questions distinct from `rpM`'s seed 31!), the 180 SVAMP-like math pairs are 100% on-policy for `rpL-60`!

# Wait: what if someone asks whether reusing `rpI`'s non-math rollouts (which `rpK` and `rpL` also trained on from `rpE-80`) for `rpN` on top of `rpL-60` means some of the `right_cr` / `right_count` pairs overlap with pairs `rpL-60` already saw during its 60 steps of `rpL`?
# LET'S CHECK THIS! This is a crucial scientific question!
# How many pairs in `rpL` (`tables/rpL_pairs.csv`) were `right_cr` or `right_count` vs `shorten`?
# Let's inspect `tables/rpL_pairs.csv` and `tables/rpM_pairs_local_build_math_on_hard_questions_only.csv` right now!

rpl_df = pd.read_csv(os.path.join(pack_dir, "tables/rpL_pairs.csv"))
rpm_df = pd.read_csv(os.path.join(pack_dir, "tables/rpM_pairs_local_build_math_on_hard_questions_only.csv"))
print("rpL_pairs columns:", rpl_df.columns.tolist())
print("\nrpL_pairs breakdown by part and kind:")
print(pd.crosstab(rpl_df['part'], rpl_df['kind'], margins=True))
print("\nrpM_pairs breakdown by part and kind:")
print(pd.crosstab(rpm_df['part'], rpm_df['kind'], margins=True))
```