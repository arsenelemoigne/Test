```python
# And let's check `messages.json` in `run/` to see what happened in the current session!
with open(os.path.join(parent_dir, "run", "messages.json")) as f:
    msgs_curr = json.load(f)
print("Num msgs_curr:", len(msgs_curr))
for idx, m in enumerate(msgs_curr[:6]):
    print(idx, m["role"], len(m["content"]), m["content"][:500])
```