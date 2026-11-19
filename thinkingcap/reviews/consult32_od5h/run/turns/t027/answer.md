Let's print `track.py` from 5425 to 5620.
```python
with open(
    "/Users/arsenelemoigne/Claude Build/Test-mvt-fading/thinkingcap/reviews/consult32_od5h/pack/code/track.py"
) as f:
    tlines = f.readlines()

for i, line in enumerate(tlines[5425:5620], start=5426):
    print(f"{i:4d}: {line}", end="")
```