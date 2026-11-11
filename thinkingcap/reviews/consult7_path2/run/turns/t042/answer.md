```python
for r in all_runs:
    key = r.get("key", "")
    if any(k in key.lower() or k in r.get("name", "").lower() for k in ["gr", "tp", "drpo", "sb", "cp", "copd", "rl", "full", "rpe", "rpi", "rc1", "ca1", "od"]):
        print(f"{key:25s} | {r['kind']:10s} | saved: {r['saved']:5.1f} | acc: {r['acc']:+5.2f} | {r['name']} | {r['what'][:80]}")
```