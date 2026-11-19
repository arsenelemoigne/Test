```python
for m in ft["models"]:
    print(f"{m['model']:25s} saved={m.get('saved')} acc={m.get('acc')} dev_saved={m.get('dev_saved')} dev_acc={m.get('dev_acc')}")
    if m["model"].startswith(("od5", "rpL", "r2s3", "od4")):
        print("   per_type:", m.get("per_type") or m.get("by_type") or {k: v for k, v in m.items() if k not in ("model", "saved", "acc", "dev_saved", "dev_acc")})
```