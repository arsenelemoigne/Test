```python
rp_lines = (pack / "code" / "repair_pairs.py").read_text().splitlines()
print("=== repair_pairs.py lines 680..760 ===")
for i, line in enumerate(rp_lines[679:760], start=680):
    print(f"{i:4d}: {line}")

tr_lines = (pack / "code" / "track.py").read_text().splitlines()
print(f"\n=== track.py total lines: {len(tr_lines)} ===")
for i, line in enumerate(tr_lines, start=1):
    if any(k in line for k in ["od_smooth", "od-smooth", "no_shorten", "od_unlatch", "od_acc_hold", "fullpick", "od5_held", "od5-held", "ptrepair", "shortfall"]):
        print(f"line {i:4d}: {line}")
```