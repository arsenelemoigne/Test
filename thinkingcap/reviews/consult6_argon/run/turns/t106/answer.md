```python
# Let's check WHY `math1`, `math2`, and `nofit4` were called "michal-bida top-ups" in `p1src`!
# Look at `EXPERIMENT_LOG.md` Lines 3742-3752 and 4942:
# Wait: did michal-bida write or provide `nofit4`, `math1`, `math2`?
# Let's check how `newtop` built `nofit4`, `math1`, `math2` in `EXPERIMENT_LOG.md`!
for i in range(3735, 3810):
    print(f"L{i+1}: {log_lines[i].strip()}")
```