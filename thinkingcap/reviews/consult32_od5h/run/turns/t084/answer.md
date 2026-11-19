```python
# Let's check THINKING_OFF.md and pertype_best_full.md as well to make sure we didn't miss any data
with open(os.path.join(pack, "THINKING_OFF.md")) as f:
    print("--- THINKING_OFF.md ---")
    print(f.read()[:3000])
with open(os.path.join(pack, "pertype_best_full.md")) as f:
    print("--- pertype_best_full.md ---")
    print(f.read()[:3000])
```