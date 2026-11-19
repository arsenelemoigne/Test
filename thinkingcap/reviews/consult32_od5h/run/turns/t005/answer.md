```python
from pathlib import Path

pack = Path(
    "/Users/arsenelemoigne/Claude Build/Test-mvt-fading/thinkingcap/reviews/consult32_od5h/pack"
)

txt = (pack / "NEW_RESULTS_od5f.md").read_text()
print("NEW_RESULTS_od5f tail:")
print(txt[9000:])

print("================ YOUR_CONSULT31_REPORT.md ================")
print((pack / "YOUR_CONSULT31_REPORT.md").read_text())
```