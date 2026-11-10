```python
# Let's check where rpD or rpE is first defined in EXPERIMENT_LOG.md
for i in range(4940, 5040):
    if "rpD" in log_lines[i] or "rpE" in log_lines[i] or "repair" in log_lines[i]:
        print(f"L{i+1}: {log_lines[i].strip()}")
```