```python
from pathlib import Path

pack = Path(
    "/Users/arsenelemoigne/Claude Build/Test-mvt-fading/thinkingcap/reviews/consult32_od5h/pack"
)

for name in [
    "SMARTER_DATA.md",
    "NEW_RESULTS_od5f.md",
    "YOUR_CONSULT31_REPORT.md",
]:
    print(f"================ {name} ================")
    print((pack / name).read_text())

print("================ effsame full ================")
print((pack / "effprobe/effsame.txt").read_text())
```