```python
from pathlib import Path

pack = Path(
    "/Users/arsenelemoigne/Claude Build/Test-mvt-fading/thinkingcap/reviews/consult32_od5h/pack"
)

txt = (pack / "YOUR_CONSULT31_REPORT.md").read_text()
print(txt[10500:])

print("================ effsame tail ================")
eff = (pack / "effprobe/effsame.txt").read_text()
print(eff[:5000])
print("----")
print(eff[5000:])
```