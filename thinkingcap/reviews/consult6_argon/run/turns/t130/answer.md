```python
with open(os.path.join(work_dir, "report_draft.md")) as f:
    draft_text = f.read()
print("DRAFT LENGTH:", len(draft_text))
print("DRAFT HEAD:\n", draft_text[:2500])
print("\n---\nDRAFT TAIL:\n", draft_text[-2500:])
```