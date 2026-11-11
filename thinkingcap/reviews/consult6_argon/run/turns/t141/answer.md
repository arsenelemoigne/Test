```python
# Let's see what `session.log` has!
with open(os.path.join(parent_dir, "session.log")) as f:
    print(f.read())
```