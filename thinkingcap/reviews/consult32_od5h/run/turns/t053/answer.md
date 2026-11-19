```python
import json

ft = json.loads((pack / "fulltests.json").read_text())
print("fulltests keys:", list(ft.keys()) if isinstance(ft, dict) else f"list of {len(ft)}")
if isinstance(ft, list):
    for row in ft:
        print(row.get("name") or row.get("model") or row.get("label"), row.get("saved"), row.get("acc"))
elif isinstance(ft, dict):
    for k, v in ft.items():
        if isinstance(v, dict) and ("saved" in v or "all" in v):
            print(k, v.get("saved"), v.get("acc"))
        else:
            print(k, type(v))
```