```python
print("Files in run:", os.listdir(os.path.join(parent_dir, "run")))
for fn in os.listdir(os.path.join(parent_dir, "run")):
    fp = os.path.join(parent_dir, "run", fn)
    if os.path.isfile(fp):
        print(fn, os.path.getsize(fp))
```