```python
with open(f"{pack_dir}/PATH1_FINAL.md") as f:
    lines = f.readlines()
print(f"Total lines in PATH1_FINAL.md: {len(lines)}")
print("".join(lines[:100]))
```