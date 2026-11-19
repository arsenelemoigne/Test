Let's see the structure of `ft["models"]` in `fulltests.json` (since `ft["models"]` appears to be a list!).
```python
import json

ft = json.load(
    open(
        "/Users/arsenelemoigne/Claude Build/Test-mvt-fading/thinkingcap/reviews/consult32_od5h/pack/fulltests.json"
    )
)["models"]
print(type(ft), len(ft))
for m in ft:
    name = m.get("name") or m.get("model") or m.get("id")
    if any(k in str(name) for k in ("rpL", "r2s3", "od5")):
        print(name, m.get("saved_pct"), m.get("acc_delta"), m.keys())
```