```python
# Let's see what else is in all_runs.json for repairs (rpA, rpB, rpC, rpE, rpF, rpG, rpH, rpI)
rep_runs = df_runs[df_runs['key'].str.startswith('rp')].copy()
for idx, r in rep_runs.iterrows():
    p = r['parts']
    p_str = " | ".join([f"{k}:{v['saved']}%/{v['acc']:+.1f}" for k, v in p.items()]) if isinstance(p, dict) else ""
    print(f"{r['key']:15s} | {r['kind']:4s} | saved={r['saved']:5.1f}% | acc={r['acc']:+5.2f} | {r['what'][:65]} | {p_str}")
```