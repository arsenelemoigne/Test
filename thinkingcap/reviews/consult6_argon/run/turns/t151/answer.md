```python
# Look at `t139` through `t147` too! What happened in `t139` through `t147`? Let's check!
for td in ['t138', 't139', 't140', 't141']:
    td_path = os.path.join(turns_dir, td)
    if os.path.isdir(td_path):
        with open(os.path.join(td_path, "prompt.txt")) as f:
            p_txt = f.read()
        with open(os.path.join(td_path, "answer.md")) as f:
            a_txt = f.read()
        print(f"=== {td} === prompt len={len(p_txt)}, ans len={len(a_txt)}")
        print("PROMPT:", p_txt[:400])
        print("ANS:", a_txt[:400])
```