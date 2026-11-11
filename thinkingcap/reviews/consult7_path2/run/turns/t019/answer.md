```python
with open(f"{pack_dir}/code/repair_pairs.py") as f:
    rp_lines = f.readlines()
print(f"Total lines in repair_pairs.py: {len(rp_lines)}")
print("".join(rp_lines[:120]))
```