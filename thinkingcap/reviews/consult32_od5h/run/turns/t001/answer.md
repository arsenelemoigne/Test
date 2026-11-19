```python
from pathlib import Path

pack = Path(
    "/Users/arsenelemoigne/Claude Build/Test-mvt-fading/thinkingcap/reviews/consult32_od5h/pack"
)

for name in [
    "README_PACK.md",
    "code_changes.diff",
    "od5h_plan_lines.txt",
    "effprobe/effsame.txt",
    "SMARTER_DATA.md",
]:
    p = pack / name
    print(f"================ {name} ================")
    if p.exists():
        print(p.read_text())
    else:
        print("NOT FOUND")

print("================ FILES IN pack ================")
for p in sorted(pack.rglob("*")):
    if p.is_file():
        print(p.relative_to(pack), p.stat().st_size)
```