Let's print the `key`, `saved`, `acc`, and `parts` for `rpL`, `r2s3`, and `od5*` from `fulltests.json`, and also check `THINKING_OFF.md` and `pertype_best_full.md`.
```python
import json

ft = json.load(
    open(
        "/Users/arsenelemoigne/Claude Build/Test-mvt-fading/thinkingcap/reviews/consult32_od5h/pack/fulltests.json"
    )
)["models"]
for m in ft:
    if any(k in m["key"] for k in ("rpL", "r2s", "od5")):
        print(
            f"{m['key']:15s}: saved={m['saved']} acc={m['acc']} | parts={ {p: (v['saved'], v['acc']) for p, v in m['parts'].items()} }"
        )
```