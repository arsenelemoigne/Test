```python
for i in range(3970, len(exp_lines)):
    line = exp_lines[i]
    if re.search(r"\b(grA|grB|grC|grD|grE|grF|grS|grP|gr2|tp0|tp1|thinkprune|drpo)\b", line):
        print(f"L{i+1}: {line.strip()[:140]}")
```