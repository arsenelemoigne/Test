```python
for p in sorted(pack.rglob("*")):
    if p.is_file():
        print(p.relative_to(pack), f"({p.stat().st_size} bytes)")
```