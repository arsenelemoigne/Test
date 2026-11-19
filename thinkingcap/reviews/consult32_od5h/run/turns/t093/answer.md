```python
import glob, os
print([k for k in globals().keys() if not k.startswith("__")])
for p in glob.glob("**/notes.md", recursive=True):
    print("FOUND:", p)
```