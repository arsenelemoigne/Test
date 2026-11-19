```python
dp_lines = (pack / "code" / "dpo.py").read_text().splitlines()
print("=== dpo.py ===")
for i, line in enumerate(dp_lines, start=1):
    print(f"{i:4d}: {line}")

print("\n=== track.py lines 5485..5560 ===")
for i, line in enumerate(tr_lines[5484:5560], start=5485):
    print(f"{i:4d}: {line}")
```