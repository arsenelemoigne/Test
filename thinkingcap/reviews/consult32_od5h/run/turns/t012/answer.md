```python
from pathlib import Path

pack = Path(
    "/Users/arsenelemoigne/Claude Build/Test-mvt-fading/thinkingcap/reviews/consult32_od5h/pack"
)

rp = (pack / "code/repair_pairs.py").read_text().splitlines()
for i in range(200, len(rp)):
    print(f"{i+1:4d}: {rp[i]}")
```