```python
tr_lines = (pack / "code" / "track.py").read_text().splitlines()
print("=== track.py lines 4620..4710 (fullpick) ===")
for i, line in enumerate(tr_lines[4619:4710], start=4620):
    print(f"{i:4d}: {line}")

print("\n=== track.py lines 5180..5510 (ptrepair / odpo / odpo_fixed) ===")
for i, line in enumerate(tr_lines[5179:5510], start=5180):
    print(f"{i:4d}: {line}")
```