```python
# Let's check the FULL TEST runs (all 8 in all_runs.json + ca1, l2s50e, michal-bida from the log) to see the EXACT per-part breakdown on the full 2,759 x 8 test!
# Let's check if l2s50e and michal-bida per-part numbers are in EXPERIMENT_LOG.md:
# Earlier we saw:
# lpc-l2s50e-soup-100 full test: saved +48.0%, acc -2.08
#   svamp 55.9 / -3.2, mbpp 51.9 / -2.2, ifeval 45.8 / -1.6, simple 33.4 / -0.8, multiple 36.6 / -0.2, no-fit 40.9 / -3.3
# michal-bida full test: saved +47.0%, acc -0.78
#   svamp 56 / -2.41, mbpp 33 / -1.22, ifeval 55 / -0.28, simple 35 / -1.41, multiple 42 / -1.12, no-fit 46 / +6.04

full_table = []
for _, r in df[df['kind'] == 'full'].iterrows():
    full_table.append({
        'model': r['key'],
        'saved': r['saved'],
        'acc': r['acc'],
        'v_s': r['v_saved'], 'v_a': r['v_acc'],
        'c_s': r['c_saved'], 'c_a': r['c_acc'],
        'f_s': r['f_saved'], 'f_a': r['f_acc'],
        'bfcl_s': r['bfcl_saved'], 'bfcl_a': r['bfcl_acc'],
        's_s': r['s_saved'], 's_a': r['s_acc'],
        'm_s': r['m_saved'], 'm_a': r['m_acc'],
        'i_s': r['i_saved'], 'i_a': r['i_acc'],
    })

# Add lpc-l2s50e-soup-100 and michal-bida
full_table.append({
    'model': 'l2s50e-100(full)', 'saved': 48.0, 'acc': -2.08,
    'v_s': 55.9, 'v_a': -3.2, 'c_s': 51.9, 'c_a': -2.2, 'f_s': 45.8, 'f_a': -1.6,
    'bfcl_s': 36.3, 'bfcl_a': (400*(-0.8)+200*(-0.2)+240*(-3.3))/840,
    's_s': 33.4, 's_a': -0.8, 'm_s': 36.6, 'm_a': -0.2, 'i_s': 40.9, 'i_a': -3.3
})
full_table.append({
    'model': 'michal-bida(ref)', 'saved': 47.0, 'acc': -0.78,
    'v_s': 56.0, 'v_a': -2.41, 'c_s': 33.0, 'c_a': -1.22, 'f_s': 55.0, 'f_a': -0.28,
    'bfcl_s': 39.8, 'bfcl_a': (400*(-1.41)+200*(-1.12)+240*(+6.04))/840,
    's_s': 35.0, 's_a': -1.41, 'm_s': 42.0, 'm_a': -1.12, 'i_s': 46.0, 'i_a': +6.04
})

df_full_all = pd.DataFrame(full_table).sort_values('saved', ascending=False)
print(df_full_all[['model', 'saved', 'acc', 'v_s', 'v_a', 'c_s', 'c_a', 'f_s', 'f_a', 'bfcl_s', 'bfcl_a', 'i_a']].to_string(index=False))
```