```python
# Look closely at the table above:
# 1. `cleanA-rc1-full` -> `rpe-40full` -> `rpe-60full` -> `rpe-80full` -> `rpi-120full`:
#    Wait: why is `rc1 -> rpE -> rpI` NOT an acceptable final submission on its own?
#    Look at Constraint 1 in LADDER_DESIGN_FOR_REVIEW.md:
#    "1. One clean, reproducible recipe (a single training line), not a stack of corrective stages."
#    Remember how `rc1-soup-30` was made!
#    `clean-a1` (DPO round 1, no code) -> `clean-a2` (DPO round 2, no code) -> `ca1-soup-10` (DPO round 3, 3 seeds souped) -> `rc1-soup-30` (cut-calibration DPO round 4, 3 seeds souped) -> `rpE-80` (repair DPO round 5) -> `rpI-120` (repair DPO round 6)!
#    That was a 6-round stack of ad-hoc corrective stages where checkpoints were hand-picked on screens at every round!
#
# 2. Now let's ask: WHY did `rpE` and `rpI` lift the model above the standard DPO trade-off curve (from +0.95 to +1.49 and +1.64 vs curve on the full test), and WHY did `rpE` lose saving from 57.4% -> 47.0% -> 41.9%?
#    Let's check the pairs in `rpE` and `rpI`:
#    In `rpE`, the pairs were 100% "guard/be-right" pairs (shortest right > shortest finished wrong, preferring answers that re-read the question, equal median lengths 271/278, 440/440, 185/183), with ZERO shorten pairs!
#    Because there were ZERO shorten pairs in `rpE` (and `rpI`), training for 80+40 steps allowed length to drift back up from 57.4% saved to 47.0% (`rpE-80`) and 41.9% (`rpI-120`)!
#    Wait: what happened when they tried adding shorten pairs or running from the original model?
#    Let's check ALL single-recipe or clean-line runs from the original model in `all_runs.json`!

# Let's filter `df` for runs that started from the raw model (single round or clean 2-stage) and see how they performed!
print("--- SINGLE-ROUND / CLEAN RUNS FROM RAW MODEL IN ALL_RUNS.JSON ---")
raw_runs = df[~df['key'].str.startswith(('rp', 'pt', 'lpt6', 'lpk', 'cleanA'))].sort_values('vs_curve', ascending=False)
for _, r in raw_runs.head(25).iterrows():
    print(f"{r['key']:15s} ({r['kind']:4s}) | saved={r['saved']:5.1f}% | acc={r['acc']:+5.2f} | vs_curve={r['vs_curve']:+5.2f} | v:{r['v_saved']:.0f}/{r['v_acc']:+.1f} c:{r['c_saved']:.0f}/{r['c_acc']:+.1f} f:{r['f_saved']:.0f}/{r['f_acc']:+.1f} bfcl:{r['bfcl_saved']:.0f}/{r['bfcl_acc']:+.2f} (i:{r['i_acc']:+.1f})")
```