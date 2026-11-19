Let's print `track.py` from 4620 to 4720 (`fullpick`), 4920 to 5060 (`ptrepair`), and 5370 to 5620 (`odpo_fixed`).
```python
with open(
    "/Users/arsenelemoigne/Claude Build/Test-mvt-fading/thinkingcap/reviews/consult32_od5h/pack/code/track.py"
) as f:
    tlines = f.readlines()

for a, b in [(4620, 4720), (4920, 5040), (5370, 5600)]:
    print(f"=== LINES {a+1}..{b} ===")
    for i, line in enumerate(tlines[a:b], start=a + 1):
        print(f"{i:4d}: {line}", end="")
```