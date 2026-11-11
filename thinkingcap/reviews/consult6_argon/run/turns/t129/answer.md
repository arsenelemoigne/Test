```python
with open(os.path.join(work_dir, "notes.md")) as f:
    notes_text = f.read()
print("NOTES HEAD:\n", notes_text[:2500])
print("\n---\nNOTES TAIL:\n", notes_text[-1500:])
```