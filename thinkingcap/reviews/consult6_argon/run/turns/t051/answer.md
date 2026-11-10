```python
# Let's do a deep dive into WHY earlier runs failed or succeeded, and what IS the closest path to >50% saved at > -1.0 pp on the full test!
# Let's check the numbers on `rpE-40full`, `rpE-60full`, `rpE-80full`, `rc1-soup-30`, `ca1-soup-10`, `pt-mid-full`, `bidax5q`, `D4 lpd-budget`, `rpB`, `rpC`, `rpI`!
# Look at `rpe-40full`, `rpe-60full`, `rpe-80full` on the FULL 2,759 x 8 test:
# Let's print the entire trajectory from `cleanA-rc1-full` (step 0 of rpE) -> `rpe-40full` -> `rpe-60full` -> `rpe-80full` -> `rpi-120full`!

traj_keys = ['cleanA-rc1-full', 'rpe-40full', 'rpe-60full', 'rpe-80full', 'rpi-120full']
print("--- FULL-TEST TRAJECTORY OF RC1 -> RPE -> RPI ---")
for k in traj_keys:
    r = df[df['key'] == k].iloc[0]
    # Let's also check what the overall accuracy and saving would be if IFEval (f) were at -0.5 pp instead of -2.6 to -3.0 pp!
    acc_if_f_fixed = r['acc'] + (-0.5 - r['f_acc']) / 4.0
    print(f"{k:16s} | saved={r['saved']:5.1f}% | acc={r['acc']:+5.2f} (if f=-0.5: {acc_if_f_fixed:+5.2f}) | v:{r['v_saved']:.0f}%/{r['v_acc']:+.1f} | c:{r['c_saved']:.0f}%/{r['c_acc']:+.1f} | f:{r['f_saved']:.0f}%/{r['f_acc']:+.1f} | bfcl:{r['bfcl_saved']:.1f}%/{r['bfcl_acc']:+.2f} (s:{r['s_acc']:+.1f}, m:{r['m_acc']:+.1f}, i:{r['i_acc']:+.1f})")
```