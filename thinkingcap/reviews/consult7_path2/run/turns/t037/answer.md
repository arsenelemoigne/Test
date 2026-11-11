```python
with open(f"{pack_dir}/competitor_methods_study.md") as f:
    comp_lines = f.readlines()
print(f"Total lines in competitor_methods_study.md: {len(comp_lines)}")
for i, line in enumerate(comp_lines):
    if line.startswith("#"):
        print(f"L{i+1}: {line.strip()}")
```