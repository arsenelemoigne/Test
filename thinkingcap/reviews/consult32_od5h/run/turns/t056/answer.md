```python
for m in ft["models"]:
    if any(k in m["key"] for k in ("rpL", "r2s3", "od5", "od4", "od3")):
        p = m["parts"]
        s = ", ".join(f"{k}: {v['saved']:.1f}%/{v['acc']:+.1f}" for k, v in p.items())
        print(f"{m['key']:15s} ({m['saved']:5.1f}% / {m['acc']:+5.2f}): {s}")
```