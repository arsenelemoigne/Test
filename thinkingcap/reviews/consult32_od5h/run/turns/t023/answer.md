Now let's read `track.py` around `odpo_fixed`, `fullpick`, and `ptrepair` (or wherever `pairs.jsonl` is trained on in `track.py`).
```python
with open(
    "/Users/arsenelemoigne/Claude Build/Test-mvt-fading/thinkingcap/reviews/consult32_od5h/pack/code/track.py"
) as f:
    tlines = f.readlines()

for i, line in enumerate(tlines, start=1):
    if any(
        k in line
        for k in (
            "def odpo",
            "def fullpick",
            "def ptrepair",
            "od_smooth",
            "od5_held",
            "shortfall",
        )
    ):
        print(f"{i:4d}: {line}", end="")
```