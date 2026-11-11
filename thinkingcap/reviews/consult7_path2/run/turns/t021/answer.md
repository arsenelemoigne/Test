```python
for i in range(3660, len(exp_lines)):
    line = exp_lines[i]
    if line.startswith("#"):
        print(f"L{i+1}: {line.strip()[:120]}")
```