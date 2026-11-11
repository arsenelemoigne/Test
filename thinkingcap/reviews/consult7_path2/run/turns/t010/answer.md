```python
with open(f"{pack_dir}/code/rl_train.py") as f:
    rl_code_lines = f.readlines()
print(f"Total lines in rl_train.py: {len(rl_code_lines)}")
for i, line in enumerate(rl_code_lines):
    if line.startswith("def ") or line.startswith("class ") or "add_argument" in line:
        print(f"L{i+1}: {line.strip()}")
```