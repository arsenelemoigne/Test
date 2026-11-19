```python
with open("/Users/arsenelemoigne/Claude Build/Test-mvt-fading/thinkingcap/reviews/consult32_od5h/pack/code/track.py") as f:
    tr_lines = f.readlines()
for idx, l in enumerate(tr_lines, 1):
    if "od_smooth" in l or "def odpo_fixed" in l or "def fullpick" in l or "def ptrepair" in l:
        print(f"Match at {idx}: {l.strip()}")
```