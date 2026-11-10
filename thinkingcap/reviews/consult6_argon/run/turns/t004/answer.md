```python
for ln, h in headers:
    if ln >= 3480 or (1080 <= ln <= 2255):
        print(f"L{ln}: {h}")
```