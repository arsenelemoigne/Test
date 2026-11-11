```python
# AHA! Look at t126 and t127! Why did t126 and t127 not end the run? Let's read `t127/prompt.txt`!
with open(os.path.join(turns_dir, "t127", "prompt.txt")) as f:
    print("t127 prompt.txt:\n", f.read())
```