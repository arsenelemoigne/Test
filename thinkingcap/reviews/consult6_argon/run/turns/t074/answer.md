```python
# Let's read all 4 files in pack/code/ completely!
for fname in sorted(os.listdir(code_dir)):
    fpath = os.path.join(code_dir, fname)
    with open(fpath, "r") as f:
        content = f.read()
    print(f"================ {fname} ({len(content)} chars) ================")
    print(content[:9000])
    if len(content) > 9000:
        print(f"... [TRUNCATED {len(content)-9000} chars] ...")
```