```python
# Let's also check session_part1.log, session_part2.log, session_part3.log, session_part4.log!
for fn in ['session_part1.log', 'session_part2.log', 'session_part3.log', 'session_part4.log']:
    with open(os.path.join(parent_dir, fn)) as f:
        print("===", fn)
        print(f.read())
```