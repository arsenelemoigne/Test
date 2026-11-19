Let's check the structure of `fulltests.json` and `EXPERIMENT_LOG.md` (specifically around `od5f` and `od5g`), and also update `/Users/arsenelemoigne/Claude Build/Test-mvt-fading/thinkingcap/reviews/consult32_od5h/work/notes.md`.
```python
import json

ft = json.load(
    open(
        "/Users/arsenelemoigne/Claude Build/Test-mvt-fading/thinkingcap/reviews/consult32_od5h/pack/fulltests.json"
    )
)
print("Type of ft:", type(ft), "len:", len(ft))
if isinstance(ft, list):
    for item in ft[-15:]:
        print(
            item.get("name") or item.get("label") or item.get("model"),
            {
                k: v
                for k, v in item.items()
                if k not in ("by_part", "parts", "sources")
            },
        )
elif isinstance(ft, dict):
    print(list(ft.keys())[-15:])
```