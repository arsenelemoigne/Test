Let's print the keys of `ft["models"]` and inspect `rpL-80`, `od5c-best`, `od5r-r19`, `od5r-r20`, `od5f-r18`, `od5g-r20`, etc., and read the end of `EXPERIMENT_LOG.md`.
```python
import json

ft = json.load(
    open(
        "/Users/arsenelemoigne/Claude Build/Test-mvt-fading/thinkingcap/reviews/consult32_od5h/pack/fulltests.json"
    )
)["models"]
for k in [
    "rpL-80",
    "r2s3-s80",
    "od5c-best",
    "od5r-r19",
    "od5r-r20",
    "od5f-r18",
    "od5g-r20",
]:
    if k in ft:
        print(
            f"=== {k} === saved={ft[k]['saved']} acc={ft[k]['acc']} ci={ft[k].get('ci_saved')}/{ft[k].get('ci_acc')}"
        )
        for p, v in ft[k].get("parts", ft[k].get("by_part", {})).items():
            print("  ", p, v)

with open(
    "/Users/arsenelemoigne/Claude Build/Test-mvt-fading/thinkingcap/reviews/consult32_od5h/pack/EXPERIMENT_LOG.md"
) as f:
    el = f.read()
print("=== EXPERIMENT_LOG tail ===")
print(el[-4500:])
```