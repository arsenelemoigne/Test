```python
# Let's check what happened to od1 and od2 or if there are any other runs in EXPERIMENT_LOG.md!
# Wait: earlier we saw at the end of EXPERIMENT_LOG.md that od1 round 1 had started! Were there any more entries about od1, od2, cp1d, cp1e?
# Let's check if there are any other files or references in the log to od1, od2, lpd-budget (D4), etc.

for pat in ["od1", "od2", "lpd-budget", "D4", "DAST", "adar1", "D9"]:
    m = search_log(pat)
    print(f"--- {pat} ({len(m)} matches) ---")
    for ln, txt in m:
        print(f"L{ln}: {txt}")
```