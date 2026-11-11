```python
# Let's see what is inside t125, t126, t127, t128, t139!
for td in ['t125', 't126', 't127', 't128', 't138', 't139']:
    td_path = os.path.join(turns_dir, td)
    if os.path.isdir(td_path):
        print(td, os.listdir(td_path))
        for fn in os.listdir(td_path):
            print("  ", fn, os.path.getsize(os.path.join(td_path, fn)))
```