```python
# Let's analyze all_runs.json quantitatively:
# 1. Per-part accuracy vs saving across all 197 screen-720 runs and 8 full runs (plus the 2 extra full runs in log: ca1 and l2s50e).
# 2. Extract part-level data into columns and examine slopes, correlations, spill-over, etc.

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

# Let's check top runs by vs_curve (for saved >= 35%) on 720 and on full
print("--- TOP 15 SCREEN-720 RUNS (saved >= 35%) BY VS_CURVE ---")
sub720 = df[(df['kind'] == '720') & (df['saved'] >= 35.0)].sort_values('vs_curve', ascending=False)
for _, r in sub720.head(15).iterrows():
    print(f"{r['key']:15s} | saved={r['saved']:5.1f}% | acc={r['acc']:+5.2f} | vs_curve={r['vs_curve']:+5.2f} | v:{r['v_saved']:.0f}/{r['v_acc']:+.1f} c:{r['c_saved']:.0f}/{r['c_acc']:+.1f} f:{r['f_saved']:.0f}/{r['f_acc']:+.1f} bfcl:{r['bfcl_saved']:.0f}/{r['bfcl_acc']:+.2f} (i:{r['i_acc']:+.1f})")

print("\n--- ALL RUNS WITH SAVED >= 48% AND ACC >= -2.5 ---")
sub_near = df[(df['saved'] >= 48.0) & (df['acc'] >= -2.5)].sort_values('acc', ascending=False)
for _, r in sub_near.iterrows():
    print(f"{r['key']:15s} ({r['kind']:4s}) | saved={r['saved']:5.1f}% | acc={r['acc']:+5.2f} | vs_curve={r['vs_curve']:+5.2f} | v:{r['v_saved']:.0f}/{r['v_acc']:+.1f} c:{r['c_saved']:.0f}/{r['c_acc']:+.1f} f:{r['f_saved']:.0f}/{r['f_acc']:+.1f} s:{r['s_saved']:.0f}/{r['s_acc']:+.1f} m:{r['m_saved']:.0f}/{r['m_acc']:+.1f} i:{r['i_saved']:.0f}/{r['i_acc']:+.1f}")
```