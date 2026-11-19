```python
tr_lines = (pack / "code" / "track.py").read_text().splitlines()
for i, line in enumerate(tr_lines, start=1):
    if any(k in line for k in ["od_smooth", "od-smooth", "no_shorten", "od_unlatch", "od_acc_hold", "def odpo", "def fullpick", "od5_held", "od5-held", "shortfall.json"]):
        print(f"line {i:4d}: {line}")
```