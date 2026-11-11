```python
# Let's check `t127/answer.md`! Why did `t127` happen right before `t128` (the handoff restart)?
with open(os.path.join(turns_dir, "t127", "answer.md")) as f:
    t127_ans = f.read()
print("t127_ans matches of OPEN:", OPEN.findall(t127_ans))
print("t127_ans head:", t127_ans[:400])
print("t127_ans tail:", t127_ans[-400:])
```