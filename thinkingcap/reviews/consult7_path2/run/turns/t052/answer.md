```python
for i, line in enumerate(exp_lines):
    if "gr2" in line or "grA2" in line or "tp_persist" in line or "tp_rl" in line:
        print(f"L{i+1}: {line.strip()[:140]}")
```