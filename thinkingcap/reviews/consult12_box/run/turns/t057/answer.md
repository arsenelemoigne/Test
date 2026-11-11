```python
with open(os.path.join(pack_dir, "code/repair_pairs.py"), "r") as f:
    rp_lines = f.read().splitlines()
for i, line in enumerate(rp_lines[210:285], start=211):
    print(f"{i:3d}: {line}")
```