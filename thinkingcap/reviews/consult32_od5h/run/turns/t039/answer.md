Let's check `rpL-80` in `fulltests.json` (`key` had `rpl` lowercase probably) and also read the rest of `THINKING_OFF.md`.
```python
import json

ft = json.load(
    open(
        "/Users/arsenelemoigne/Claude Build/Test-mvt-fading/thinkingcap/reviews/consult32_od5h/pack/fulltests.json"
    )
)
print("original_thinking_off:", ft["original_thinking_off"])
for m in ft["models"]:
    if "rpl" in m["key"].lower():
        print(m["key"], m["saved"], m["acc"], m["parts"])
print("=== THINKING_OFF tail ===")
print(
    open(
        "/Users/arsenelemoigne/Claude Build/Test-mvt-fading/thinkingcap/reviews/consult32_od5h/pack/THINKING_OFF.md"
    ).read()[-1500:]
)
```