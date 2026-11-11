```python
with open(os.path.join(pack_dir, "code/queued_lines.txt"), "r") as f:
    ql = f.read()
print(ql[:3500])
```