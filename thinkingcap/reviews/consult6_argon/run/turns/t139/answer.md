```python
# AHA! Look at `session_second.log`, `session_third.log`, `REPORT.md`!
# Let's read `session_second.log` and `session_third.log` to see the exact prompt of the second session and why it restarted!
with open(os.path.join(parent_dir, "session_second.log")) as f:
    print("SESSION_SECOND.LOG:\n", f.read())
with open(os.path.join(parent_dir, "session_third.log")) as f:
    print("\nSESSION_THIRD.LOG:\n", f.read())
```