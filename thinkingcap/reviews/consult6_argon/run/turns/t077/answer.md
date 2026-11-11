```python
# Now let's read track_ptrepair_odpo_dev.py completely!
with open(os.path.join(code_dir, "track_ptrepair_odpo_dev.py"), "r") as f:
    tr_text = f.read()

print(f"=== track_ptrepair_odpo_dev.py ({len(tr_text)} chars) ===")
print(tr_text[:10000])
```