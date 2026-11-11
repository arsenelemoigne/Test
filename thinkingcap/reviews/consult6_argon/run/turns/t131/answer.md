```python
# Check what all sections of report_draft.md are and check if there were earlier messages or if we also have the original ladder review or if report_draft.md needs to satisfy both or what!
import re
headers = [line for line in draft_text.splitlines() if line.startswith("#")]
for h in headers:
    print(h)
```