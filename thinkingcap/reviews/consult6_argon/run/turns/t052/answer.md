```python
# Rebuild df after kernel restart
df_runs = pd.DataFrame(all_runs)
records = []
for idx, r in df_runs.iterrows():
    p = r['parts']
    row = {
        'key': r['key'],
        'name': r['name'],
        'kind': r['kind'],
        'what': r['what'],
        'saved': r['saved'],
        'acc': r['acc'],
        'yardstick': -0.388 + 0.0899 * r['saved'] - 0.0028 * (r['saved']**2),
    }
    row['vs_curve'] = row['acc'] - row['yardstick']
    for k in ['v', 'c', 'f', 's', 'm', 'i']:
        if isinstance(p, dict) and k in p:
            row[f'{k}_saved'] = p[k]['saved']
            row[f'{k}_acc'] = p[k]['acc']
    row['bfcl_acc'] = (400*row['s_acc'] + 200*row['m_acc'] + 240*row['i_acc'])/840.0
    row['bfcl_saved'] = 100.0 * (1.0 - (400*(1-row['s_saved']/100.0) + 200*(1-row['m_saved']/100.0) + 240*(1-row['i_saved']/100.0))/840.0)
    records.append(row)
df = pd.DataFrame(records)

traj_keys = ['cleanA-rc1-full', 'rpe-40full', 'rpe-60full', 'rpe-80full', 'rpi-120full']
print("--- FULL-TEST TRAJECTORY OF RC1 -> RPE -> RPI ---")
for k in traj_keys:
    r = df[df['key'] == k].iloc[0]
    acc_if_f_fixed = r['acc'] + (-0.5 - r['f_acc']) / 4.0
    print(f"{k:16s} | saved={r['saved']:5.1f}% | acc={r['acc']:+5.2f} (if f=-0.5: {acc_if_f_fixed:+5.2f}) | v:{r['v_saved']:.0f}%/{r['v_acc']:+.1f} | c:{r['c_saved']:.0f}%/{r['c_acc']:+.1f} | f:{r['f_saved']:.0f}%/{r['f_acc']:+.1f} | bfcl:{r['bfcl_saved']:.1f}%/{r['bfcl_acc']:+.2f} (s:{r['s_acc']:+.1f}, m:{r['m_acc']:+.1f}, i:{r['i_acc']:+.1f})")
```