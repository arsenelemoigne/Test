```python
# Let's check this! Why did `rpE` work so well on `rc1-soup-30`, and what happens in `rpJ` vs `rpE`?
# Wait: let's look at `rpJ` vs `rpE`:
# In `rpE` (which scored 50.6% / -1.90 pp at step 40, 47.8% / -1.20 pp at step 60, and 47.0% / -0.90 pp at step 80 on the full test):
# - What pair rule did `rpE` use?
#   `--kinds guard --think-only ifeval --prefer-reread`!
#   Wait: in `repair_pairs.py`, what is `--kinds guard`?
#   `chosen` = shortest right answer within 1.5x of `rights[0]` that re-reads the question (else `rights[0]`).
#   `rejected` = `wrongs[0]` (the shortest finished wrong answer!).
#   And why did `rpE` drift from 57.4% saved (`rc1-soup-30`) down to 50.6% (@40), 47.8% (@60), 47.0% (@80)?
#   Look at `repair_pairs.py` Lines 156-174:
#   On a already-shortened model (`rc1-soup-30`, 57.4% saved), `wrongs[0]` (the shortest finished wrong answer) includes the **30% early-stop wrong answers** (`thinking < 0.5x` median right), while `chosen` (`checked[0]` within `1.5x` of `rights[0]`) is a full compact right answer!
#   Wait: even so, in `rpE`, the medians of `chosen` and `rejected` were `271/278, 440/440, 185/183` (L5040) because on the other 70% of questions `rights[0]` was slightly shorter than `wrongs[0]`, balancing the 30% early-stop questions where `chosen > wrongs[0]`!
#   Now look at what `--kinds mix` in `repair_pairs.py` does in `rpJ` (and why `CLAUDE_REVIEW.md` noted `rpJ` differs from `rpE-80` in 5 things at once):
#   In `rpJ` (`queued_lines.txt`):
#   1) Instead of `rpE`'s `--kinds guard` (`shortest right with reread > wrongs[0]`), `rpJ` replaces `rpE`'s pairs with:
#      - `30% shorten` (`rights[0] > rights[-1]` on `>= 6/8` solved, ratio `>= 1.3x`)
#      - `15% capguard` (`pick <= 1.15 * tmed` > `early[-1] < 0.5 * tmed`, where `chosen` is **>2x longer** than `rejected`!)
#      - `55% right` (`right_long` + `right_short` within `--ratio 1.2`, where `take()` calls `rng.shuffle(pool)` and **loses `--prefer-reread` across questions**, AND picks arbitrary-length right answers instead of `rights[0]`!)
#   2) Look at `rpJ`'s quotas vs `rpE`'s quotas:
#      - `rpE` quotas (`L4957`): `svamp=640, mbpp=400, ifeval=320, bfcl_simple_python=80, bfcl_multiple=80, bfcl_irrelevance=80` (total `1,600` pairs; `rpE` found `1,484` in `rpG`, `1,598` in `rpH`).
#      - `rpJ` quotas (`queued_lines.txt`): `svamp=320, mbpp=320, ifeval=320, bfcl_simple_python=100, bfcl_multiple=80, bfcl_irrelevance=140` (total `1,280` pairs).
#      - Look at `bfcl_irrelevance=140` in `rpJ`: `rpJ` reuses `rpH`'s rollouts (`--rp-rollouts-from rpH`), which only answered **300 `bfcl_irrelevance` questions**!
#      - In those 300 `bfcl_irrelevance` questions, `rc1-soup-30` only has ~60–70 mixed-outcome questions!
#      - How many `right` pairs within `--ratio 1.2` (`±20%` total tokens) + `capguard` pairs (`thinking_len(w) < 0.5 * tmed`) can 60–70 mixed-outcome `no-fit` questions produce? Far fewer than `140 - 42 = 98`!
#   3) Even more importantly: what happens when `rpJ` IS run (since it is already queued on GPU 2 after `devset` and only takes ~1.5–2 GPU-hours since `rpH` rollouts already exist)?
#      Let's check! `devset` + `devscore rc1s30` + `devscore rpE80` + `rpJ` is ALREADY running or queued at the top of `plan_line_a.txt` on GPU 2!
#      Wait: should the team kill `rpJ` if it's already running, or let `rpJ` finish (it's only ~1.5h after Step 0) while we fix the bugs in `repair_pairs.py` and `track.py` and redesign Round 1 so Round 1 hits **55–58% saved**?

# Let's check all the numbers on Round 1 runs from raw Qwen in `all_runs.json` to see which Round-1 recipe from raw Qwen actually reaches **55–58% saved** on clean own data!
df_runs = pd.DataFrame(all_runs)
print("--- ALL RUNS IN ALL_RUNS.JSON WITH SAVED BETWEEN 51% AND 60% ---")
sub50 = df_runs[(df_runs['saved'] >= 51.0) & (df_runs['saved'] <= 60.0)].sort_values('acc', ascending=False)
for _, r in sub50.iterrows():
    print(f"{r['key']:18s} ({r['kind']:4s}) | saved={r['saved']:5.1f}% | acc={r['acc']:+5.2f} | {r['what'][:75]}")
```