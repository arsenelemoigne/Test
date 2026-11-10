```python
# Let's print the rest of repair_pairs.py and track_ptrepair_odpo_dev.py!
with open(os.path.join(code_dir, "repair_pairs.py"), "r") as f:
    rp_txt = f.read()
print("=== repair_pairs.py (second half) ===")
print(rp_txt[4200:])

with open(os.path.join(code_dir, "track_ptrepair_odpo_dev.py"), "r") as f:
    tr_txt = f.read()
print("\n=== track_ptrepair_odpo_dev.py ===")
print(tr_txt[:9000])
if len(tr_txt) > 9000:
    print(tr_txt[9000:])
```