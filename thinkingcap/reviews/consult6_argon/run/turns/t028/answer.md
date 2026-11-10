```python
# Let's check the pairs of 720 vs full runs in all_runs.json (plus ca1 and l2s50e from the log)!
pairs_720_full = [
    ("bidax2", "bidax2-final", "bidax2-full"),
    ("pt-mid", "pt-mid", "pt-mid-full"),
    ("rc1-soup-30", "cleanA-rc1-720", "cleanA-rc1-full"),
    ("rpE-40", "rpe-40", "rpe-40full"),
    ("rpE-60", "rpe-60", "rpe-60full"),
    ("rpE-80", "rpe-80", "rpe-80full"),
    ("rpI-120", "rpi-120", "rpi-120full"),
]

print(f"{'Model':14s} | {'720 saved':>9s} {'full saved':>10s} {'diff':>6s} | {'720 acc':>8s} {'full acc':>8s} {'diff':>6s}")
print("-" * 72)
diffs_s, diffs_a = [], []
for label, k720, kfull in pairs_720_full:
    r720 = df_runs[df_runs['key'] == k720].iloc[0]
    rfull = df_runs[df_runs['key'] == kfull].iloc[0]
    ds = r720['saved'] - rfull['saved']
    da = r720['acc'] - rfull['acc']
    diffs_s.append(ds)
    diffs_a.append(da)
    print(f"{label:14s} | {r720['saved']:9.1f} {rfull['saved']:10.1f} {ds:+6.1f} | {r720['acc']:+8.2f} {rfull['acc']:+8.2f} {da:+6.2f}")

# Add ca1-soup-10 and lpc-l2s50e-soup-100 from log:
# ca1-soup-10: 720 said 52.4 / -0.9, full test 49.1 / -2.4
# lpc-l2s50e-soup-100: 720 said 49.9 / -2.29, full test 48.0 / -2.08
print(f"{'ca1-soup-10*':14s} | {52.4:9.1f} {49.1:10.1f} {52.4-49.1:+6.1f} | {-0.90:+8.2f} {-2.40:+8.2f} {-0.90 - (-2.40):+6.2f}")
print(f"{'l2s50e-100*':14s} | {49.9:9.1f} {48.0:10.1f} {49.9-48.0:+6.1f} | {-2.29:+8.2f} {-2.08:+8.2f} {-2.29 - (-2.08):+6.2f}")
```