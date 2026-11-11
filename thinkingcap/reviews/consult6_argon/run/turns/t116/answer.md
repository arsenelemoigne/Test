```python
# Look at `l2s50e-100` (54.4% / -3.61 on 720) and `l2s50e-final` (53.5% / -2.81 on 720) in the table above!
# Let's check what `l2s50e` (D10 `lpd-l2s50e`) and `lpc-l2s50e-soup-100` (48.0% / -2.08 full test, 49.9% / -2.29 on 720) did!
# Let's search `EXPERIMENT_LOG.md` for `lpd-l2s50e` and `l2s50e`!
for i, l in enumerate(log_lines):
    if "l2s50e" in l:
        if i >= 4500 or (3100 <= i <= 3550):
            print(f"L{i+1}: {l.strip()[:130]}")
```