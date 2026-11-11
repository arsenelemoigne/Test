```python
# Let's check what variables are in the kernel right now!
print([k for k in globals().keys() if not k.startswith("_")])
```