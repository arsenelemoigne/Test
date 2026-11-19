```python
for row in ft["models"]:
    if any(x in str(row) for x in ["od5", "rpL", "r2s"]):
        print(row)
```