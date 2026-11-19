```python
tr_lines = (pack / "code" / "track.py").read_text().splitlines()
print("=== track.py lines 5208..5520 ===")
for i, line in enumerate(tr_lines[5207:5520], start=5208):
    print(f"{i:4d}: {line}")

td_lines = (pack / "code" / "train_dpo.py").read_text().splitlines()
print("\n=== train_dpo.py ===")
for i, line in enumerate(td_lines, start=1):
    print(f"{i:4d}: {line}")
```