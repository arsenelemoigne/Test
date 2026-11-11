```python
def read_file(rel_path, max_chars=9500, from_end=False):
    path = os.path.join(pack_dir, rel_path)
    with open(path, "r") as f:
        text = f.read()
    if from_end:
        return text[-max_chars:]
    return text[:max_chars]

print("=== consult11_gemini_REPORT.md ===")
print(read_file("earlier_reviews/consult11_gemini_REPORT.md", 9500))
```