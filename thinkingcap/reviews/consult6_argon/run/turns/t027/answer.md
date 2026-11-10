```python
# Let's see all 205 keys in all_runs.json to understand the families of runs
for i in range(0, len(df_runs), 5):
    chunk = df_runs.iloc[i:i+5]
    print(" | ".join([f"{r['key']}({r['saved']:.1f}/{r['acc']:+.2f})" for _, r in chunk.iterrows()]))
```