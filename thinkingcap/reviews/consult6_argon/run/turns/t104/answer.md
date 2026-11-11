```python
# Let's check `own_repair_questions()` in the log (around L4945-4951) to see how `own_repair_questions()` is implemented in `track.py`!
# Let's search `EXPERIMENT_LOG.md` for `own_repair_questions`!
for i, l in enumerate(log_lines):
    if "own_repair_questions" in l:
        print(f"L{i+1}: {l.strip()}")
```