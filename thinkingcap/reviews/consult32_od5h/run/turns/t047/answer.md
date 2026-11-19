```python
tr_lines = (pack / "code" / "track.py").read_text().splitlines()
print("=== track.py lines 5370..5535 ===")
for i, line in enumerate(tr_lines[5369:5535], start=5370):
    print(f"{i:4d}: {line}")

print("\n=== code directory files ===")
for p in sorted((pack / "code").iterdir()):
    print(p.name)
```