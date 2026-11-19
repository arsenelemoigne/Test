```python
for m in ft["models"]:
    print(f"{m['key']:15s} ({m['saved']:5.1f}% / {m['acc']:+5.2f}) -> keys: {list(m.keys())}")
    p = m.get("parts") or m.get("per_part") or m.get("by_type") or {}
    if p:
        s = ", ".join(f"{k}: {v['saved']:.1f}%/{v['acc']:+.1f}" for k, v in p.items())
        print("   ", s)
```