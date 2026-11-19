```python
exp_lines = (pack / "EXPERIMENT_LOG.md").read_text().splitlines()
print(f"EXPERIMENT_LOG total lines: {len(exp_lines)}")
for i, line in enumerate(exp_lines[-320:], start=len(exp_lines) - 319):
    print(f"{i:4d}: {line}")
```