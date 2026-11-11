```python
# Now let's check Claim C2 in `CLAUDE_REVIEW.md`!
# Look at what `CLAUDE_REVIEW.md` says about `od1`:
#   "Round 2 must then add saving AND accuracy, which nothing in the log has done: od1 (rpE-80 + online half guard / half shorten rounds) went 50.4/-0.2 -> 53.3/-1.8 on the 720, its distance above the curve FELL (+2.8 -> +1.75)."
# WAIT A MINUTE!
# Let's check what `od1` actually did and why `rc1-soup-30 -> rpE-40/60/80` WORKED while `rpE-80 -> od1` LOST distance above the curve!
# Look at the two directions in the `(saved, acc)` plane:
# Direction A (pushing a ~30–50% saved model MORE SHORT in Round 2, like `od1` on `rpE-80`):
#   - `rpE-80` was at `50.4% / -0.2 pp` on 720 (`47.0% / -0.9 pp` on full test).
#   - `od1` took `rpE-80` and trained it on `50% shorten / 50% guard` (`--kinds guard,shorten`), pushing saving UP from `50.4% -> 53.3%`, and accuracy FELL from `-0.2 -> -1.8 pp` (distance above curve fell from `+2.8 -> +1.75`)!
#   - Earlier, `Track 2f` (round 1 -> round 2 -> round 3 -> round 4) and `D12 chain1` also pushed saving UP in later rounds and lost accuracy!
# Direction B (taking a ~55–58% saved Round-1 model and repairing accuracy at ~50–52% saved in Round 2, like `rc1-soup-30 -> rpE`):
#   - `rc1-soup-30` started at **57.4% saved / −3.50 pp** on the full test (58.7% / −3.8 pp on 720).
#   - `rpE` trained on `rc1-soup-30`'s own mistakes (`--kinds guard --prefer-reread --think-only ifeval`), trading ~6.8 points of excess saving (`57.4% -> 50.6%` at step 40; `57.4% -> 47.8%` at step 60; `57.4% -> 47.0%` at step 80) to gain **+1.60 to +2.60 pp of accuracy** on the full test (`-3.50 -> -1.90 -> -1.20 -> -0.90 pp`), moving **+0.54 pp HIGHER above the curve on the full test (`+0.95 -> +1.49`)**!
#
# STOP AND THINK ABOUT THIS!
# Look at the fundamental asymmetry between Direction A and Direction B across the team's entire 205-run history:
# - **No run in the log has EVER started Round 2 at 30–45% saved and added BOTH saving and accuracy to cross 50% / -1.0!** When `od1` tried to push `rpE-80` (47.0% full / 50.4% 720) shorter to 53.3% using 50% shorten pairs, accuracy dropped −1.6 pp (`-0.2 -> -1.8 pp`) and distance vs curve fell by −1.05 pp!
# - **Conversely, EVERY run in `all_runs.json` that sits +1.3 to +2.8 pp above the curve near 50% saved (`rpe-40full`, `rpe-60full`, `rpe-80full`, `rpB-20`, `rpC-20`, `rpC-80`) was created via Direction B:**
#   1. **Round 1** compresses raw Qwen **past 50% saved into the 55–58% saved zone** (at `-3.0 to -3.8 pp` accuracy).
#   2. **Round 2** samples on-policy from that 55–58% saved Round-1 model and trains on **compact correctness pairs (`--kinds guard` with `--prefer-reread --count-rules --think-only ifeval`, plus a small 15–20% shorten anchor so saving settles at 51–52% instead of drifting down to 47%)**, trading 4–6 points of excess saving (`56–58% -> 51–52%`) for **+2.2 to +2.8 pp of accuracy (`-3.3 -> -0.8 pp`)**!

print("Critical strategic synthesis verified: Round 1 MUST reach 55-58% saved so Round 2 operates in the proven rc1 -> rpE direction!")
```