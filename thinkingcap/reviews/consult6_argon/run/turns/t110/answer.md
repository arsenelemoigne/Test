```python
# Let's also check if `track_ptrepair_odpo_dev.py` or `queued_lines.txt` or `dev_score.py` were updated at the same time!
for fname in ["dev_score.py", "queued_lines.txt", "track_ptrepair_odpo_dev.py"]:
    with open(os.path.join(code_dir, fname), "r") as f:
        txt = f.read()
    print(f"=== {fname}: {len(txt)} chars ===")
```