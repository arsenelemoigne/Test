```python
for m, v in ft["models"].items():
    if any(x in m for x in ["od5", "rpL", "r2s"]):
        print(m, v)
```