```python
for r in all_runs[:140]:
    if any(k in r['key'].lower() or k in r['what'].lower() for k in ["gr", "tp", "drpo", "sb", "rl", "arora", "shorterbetter", "lenpen", "gated"]):
        print(f"{r['key']:25s} | {r['kind']:10s} | saved: {r['saved']:5.1f} | acc: {r['acc']:+5.2f} | {r['name']} | {r['what']}")
```