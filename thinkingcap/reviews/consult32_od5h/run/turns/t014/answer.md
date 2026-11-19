```python
from pathlib import Path

pack = Path(
    "/Users/arsenelemoigne/Claude Build/Test-mvt-fading/thinkingcap/reviews/consult32_od5h/pack"
)

rp = (pack / "code/repair_pairs.py").read_text().splitlines()
print("================ repair_pairs.py lines 335-755 ================")
for i in range(334, len(rp)):
    print(f"{i+1:4d}: {rp[i]}")
```