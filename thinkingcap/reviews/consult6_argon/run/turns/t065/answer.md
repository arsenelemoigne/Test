```python
# Let's read all 4 files in code/ completely!
for fname in ['queued_lines.txt', 'dev_score.py', 'repair_pairs.py', 'track_ptrepair_odpo_dev.py']:
    fpath = os.path.join(code_dir, fname)
    with open(fpath, "r") as f:
        txt = f.read()
    print(f"=== {fname} ({len(txt)} chars) ===")
    print(txt[:9000])
    if len(txt) > 9000:
        print(f"--- [REST OF {fname}] ---")
        print(txt[9000:])
```