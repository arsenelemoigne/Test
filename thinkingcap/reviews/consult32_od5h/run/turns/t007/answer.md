```python
from pathlib import Path

pack = Path(
    "/Users/arsenelemoigne/Claude Build/Test-mvt-fading/thinkingcap/reviews/consult32_od5h/pack"
)

txt = (pack / "YOUR_CONSULT31_REPORT.md").read_text()
print(txt[1200:])

print("================ effsame full ================")
eff = (pack / "effprobe/effsame.txt").read_text()
print("len effsame:", len(eff))
print(eff)
```