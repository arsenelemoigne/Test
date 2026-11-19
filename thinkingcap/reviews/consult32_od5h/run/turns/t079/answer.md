```python
print(type(ft), len(ft))
if isinstance(ft, list):
    for row in ft:
        name = str(row.get("name") or row.get("model") or row.get("run") or row)
        if any(x in name for x in ["od5", "rpL", "r2s"]):
            print(row)
elif isinstance(ft, dict):
    print(list(ft.keys())[:20])
```