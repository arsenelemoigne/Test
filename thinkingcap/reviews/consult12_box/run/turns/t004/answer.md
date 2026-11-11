```python
with open(os.path.join(pack_dir, "earlier_reviews/consult11_gemini_REPORT.md"), "r") as f:
    text = f.read()
print("=== consult11_gemini_REPORT.md (part 2) ===")
print(text[9000:])
```