```python
for root, dirs, files in os.walk(pack_dir):
    rel = os.path.relpath(root, pack_dir)
    print(f"Dir: {rel} -> {files}")
```