```python
print(ft["models"][0].keys())
for m in ft["models"]:
    name = m.get("name") or m.get("label") or str(m)
    print(f"{name:22s} saved={m.get('saved'):5.1f} acc={m.get('acc'):+6.2f} dev=({m.get('dev_saved')}, {m.get('dev_acc')})")
    if any(name.startswith(p) for p in ("od5", "rpL", "r2s3", "od4", "rpM", "rpI", "r2one")):
        print("  ", m.get("per_type") or m.get("by_type") or {k: v for k, v in m.items() if k not in ("name", "saved", "acc", "dev_saved", "dev_acc")})
```