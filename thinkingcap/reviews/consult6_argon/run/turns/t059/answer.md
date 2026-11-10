```python
# Let's check what happened to tsk1 (TokenSkip) and what else happened in L2770-3100!
for i in range(2770, 3100):
    line = log_lines[i].strip()
    if any(k in line.lower() for k in ["tsk", "tokenskip", "drpo", "v2r", "saved", "fresh"]):
        print(f"L{i+1}: {line[:120]}")
```