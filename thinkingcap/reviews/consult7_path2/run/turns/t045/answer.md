```python
for i in range(1080, 2750):
    line = exp_lines[i]
    if line.startswith("#") or "drpo" in line.lower() or "sb" in line.lower() or "arora" in line.lower():
        if line.startswith("#") or "scored" in line or "step" in line or "fresh" in line or "final" in line:
            print(f"L{i+1}: {line.strip()[:130]}")
```