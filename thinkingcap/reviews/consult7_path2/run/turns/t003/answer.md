```python
with open(f"{pack_dir}/rl_length_methods_study.md") as f:
    rl_lines = f.readlines()
print(f"Total lines in rl_length_methods_study.md: {len(rl_lines)}")
for i, line in enumerate(rl_lines):
    if line.startswith("#"):
        print(f"L{i+1}: {line.strip()}")
```