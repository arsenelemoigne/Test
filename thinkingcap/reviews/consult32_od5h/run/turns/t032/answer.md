Let's also check lines 450-480 of `repair_pairs.py`, and read `EXPERIMENT_LOG.md` and `fulltests.json` around `od5c`, `od5r`, `od5f`, `od5g`, and `rpL-80`.
```python
with open(
    "/Users/arsenelemoigne/Claude Build/Test-mvt-fading/thinkingcap/reviews/consult32_od5h/pack/code/repair_pairs.py"
) as f:
    lines = f.readlines()
for i, line in enumerate(lines[448:485], start=449):
    print(f"{i:4d}: {line}", end="")

import json

ft = json.load(
    open(
        "/Users/arsenelemoigne/Claude Build/Test-mvt-fading/thinkingcap/reviews/consult32_od5h/pack/fulltests.json"
    )
)
for k in [
    "od5c-best",
    "od5r-r19",
    "od5r-r20",
    "od5f-r18",
    "od5g-r20",
    "rpL-80",
]:
    if k in ft:
        print(k, ft[k]["saved"], ft[k]["acc"], ft[k].get("by_part"))
```