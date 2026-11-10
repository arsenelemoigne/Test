```python
# Let's search EXPERIMENT_LOG.md for key sections:
# 1. Track 2g: calibrated cuts, rc1, stopV3, bp1, budget, cut experiments
# 2. Thinking-off experiments (nothink, ntfull)
# 3. Full evaluation baseline numbers (full-base)
# 4. Repairs rpA, rpB, rpC, rpD, rpE, rpF, rpG, rpH, rpI
# 5. D4 lpd-budget and how it was defined! Wait: there WAS a run called D4 lpd-budget! Let's check what D4 lpd-budget and bp1b and Track 2g did!

def search_log(pattern, context=5):
    matches = []
    regex = re.compile(pattern, re.IGNORECASE)
    for i, line in enumerate(log_lines):
        if regex.search(line):
            matches.append((i+1, line.strip()))
    return matches

for pat in ["full-base", "nothink", "lpd-budget", "D4", "rpE", "rpD", "calibrated cut", "rc1", "oracle", "talked itself out"]:
    m = search_log(pat)
    print(f"Pattern '{pat}': {len(m)} matches. First 10:", [x[0] for x in m[:10]], "Last 10:", [x[0] for x in m[-10:]])
```