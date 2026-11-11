```python
import re

with open(f"{pack_dir}/EXPERIMENT_LOG.md") as f:
    exp_lines = f.readlines()
print(f"Total lines in EXPERIMENT_LOG.md: {len(exp_lines)}")

patterns = re.compile(r"\b(grA|grB|grC|gr2e|gr2f|thinkprune|tp\d|cp\d|copd|drpo|sb\d|arora|gated|lenpen)\b", re.I)
matches = []
for i, line in enumerate(exp_lines):
    if line.startswith("#") or patterns.search(line):
        if line.startswith("#"):
            print(f"L{i+1}: {line.strip()[:120]}")
```