```python
import re

for i, line in enumerate(exp_lines):
    if re.search(r"\b(grA|grB|grC|grD|grE|grF|grS|grP|gr2|tp0|tp1|thinkprune|drpo|sb1|sb2)\b", line):
        print(f"L{i+1}: {line.strip()[:140]}")
```