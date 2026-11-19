Use `.get('parts')`! Let's run that and read `pertype_best_full.md` and `THINKING_OFF.md`.
```python
import json

ft = json.load(
    open(
        "/Users/arsenelemoigne/Claude Build/Test-mvt-fading/thinkingcap/reviews/consult32_od5h/pack/fulltests.json"
    )
)["models"]
for m in ft:
    if any(k in m.get("key", "") for k in ("rpL", "r2s", "od5")):
        print(
            f"{m.get('key'):15s}: saved={m.get('saved')} acc={m.get('acc')} | parts={m.get('parts')}"
        )

for fn in ["pertype_best_full.md", "THINKING_OFF.md"]:
    print(f"=== {fn} ===")
    print(
        open(
            f"/Users/arsenelemoigne/Claude Build/Test-mvt-fading/thinkingcap/reviews/consult32_od5h/pack/{fn}"
        ).read()
    )
```