```python
from pathlib import Path

pack = Path("/Users/arsenelemoigne/Claude Build/Test-mvt-fading/thinkingcap/reviews/consult32_od5h/pack")

rp_lines = (pack / "code" / "repair_pairs.py").read_text().splitlines()
print("=== repair_pairs.py lines 560..760 ===")
for i, line in enumerate(rp_lines[559:760], start=560):
    print(f"{i:4d}: {line}")

tr_lines = (pack / "code" / "track.py").read_text().splitlines()
print("\n=== track.py around od_smooth / odpo_fixed / fullpick ===")
for i, line in enumerate(tr_lines, start=1):
    if any(k in line for k in ["od_smooth", "od-smooth", "no_shorten", "od_unlatch", "od_acc_hold", "fullpick", "od5_held", "od5-held"]):
        start = max(1, i - 15)
        end = min(len(tr_lines), i + 25)
        print(f"\n--- match at line {i} ({start}..{end}) ---")
        for j in range(start, end + 1):
            print(f"{j:4d}: {tr_lines[j-1]}")
```