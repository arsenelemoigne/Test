```python
# AHA! Look at `messages_before_second_analysis.json`! Let's check the last user message in `messages_before_second_analysis.json`!
with open(os.path.join(parent_dir, "run", "messages_before_second_analysis.json")) as f:
    msgs_prev = json.load(f)
print("Num msgs_prev:", len(msgs_prev))
for m in msgs_prev[-4:]:
    print("ROLE:", m["role"], "LEN:", len(m["content"]))
    if m["role"] == "user":
        print(m["content"][:3000])
```