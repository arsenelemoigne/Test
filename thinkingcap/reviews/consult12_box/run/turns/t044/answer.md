```python
with open(os.path.join(pack_dir, "code/track_excerpts.py"), "r") as f:
    te = f.read()
print(te[9000:])
```