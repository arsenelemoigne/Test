```python
# Let's check the rest of track_ptrepair_odpo_dev.py or any other lines in it, and let's check how `ptrepair` uses `excluded_ids(args)`!
# WAIT A MINUTE! Look at `ptrepair` in `track_ptrepair_odpo_dev.py` (Part 1, around line 65 of the file):
# Let's check how `own_repair_questions` is called inside `ptrepair`!
tr_lines = tr_text.splitlines()
for i, l in enumerate(tr_lines):
    if "exclude" in l or "own_repair_questions" in l or "rp_lr" in l or "rp_dpo_seed" in l or "rp_warmup" in l or "seed" in l:
        print(f"L{i+1}: {l}")
```