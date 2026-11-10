```python
# Let's check what other full runs exist in EXPERIMENT_LOG.md that might or might not be in all_runs.json!
# Earlier we saw lpc-l2s50e-soup-100 full evaluation (+48.0 / -2.08)! Why is it not in all_runs.json as kind='full' or is it under another name?
for idx, r in df_runs.iterrows():
    if "l2s50e" in r['key'] or "full" in r['key'] or "bida" in r['key']:
        print(f"{r['key']:20s} | {r['kind']:4s} | saved={r['saved']:5.1f} | acc={r['acc']:+5.2f} | {r['name']}")
```