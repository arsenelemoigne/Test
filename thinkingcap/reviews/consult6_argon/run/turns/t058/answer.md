```python
# Let's inspect the results of drpo2, drpo3b, drpo3d, drpo3f, sb2, tsk1 (TokenSkip), and any other runs around L2650-2850 and L4000-4150!
for i in range(2640, 2790):
    line = log_lines[i].strip()
    if any(k in line.lower() for k in ["drpo", "sb2", "tsk", "saved", "step", "fresh"]):
        print(f"L{i+1}: {line[:120]}")
```