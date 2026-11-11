```python
for r in all_runs[140:]:
    print(f"{r['key']:25s} | {r['kind']:10s} | saved: {r['saved']:5.1f} | acc: {r['acc']:+5.2f} | {r['name']} | {r['what'][:80]}")
```