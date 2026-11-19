```python
# Find fullpick and od3_monitor in track.py
for i, l in enumerate(tr_lines):
    if "def fullpick" in l or "def od3_monitor" in l:
        print(f"Found at line {i+1}: {l}")
        for j in range(i, min(len(tr_lines), i+110)):
            print(f"{j+1}: {tr_lines[j]}")
```