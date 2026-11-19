```python
from pathlib import Path

pack = Path(
    "/Users/arsenelemoigne/Claude Build/Test-mvt-fading/thinkingcap/reviews/consult32_od5h/pack"
)

for name in [
    "od5h_plan_lines.txt",
    "effprobe/effsame.txt",
    "SMARTER_DATA.md",
    "NEW_RESULTS_od5f.md",
]:
    print(f"================ {name} ================")
    print((pack / name).read_text())
```