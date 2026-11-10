```python
with open(os.path.join(pack_dir, "PATH1_PLAN.md"), "r") as f:
    path1_plan = f.read()

print(f"PATH1_PLAN.md length: {len(path1_plan)} chars")
print(path1_plan[:9500])
```