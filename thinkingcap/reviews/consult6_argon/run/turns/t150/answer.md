```python
# OH!! Look at why t126 failed to terminate the run:
# Because inside our final report (`second_report`), we had ```python code blocks (showing the bug fixes in Python)!
# Wait: `astra_harness.py` checks `find_blocks(text)` for ANY ```python block!
# If the final report contains a ```python block, `astra_harness.py` thinks it's a code execution request instead of a final answer WITHOUT a code block!
# Look at `find_blocks(text)` in `astra_harness.py`:
# `OPEN = re.compile(r"^```(python|forward)\s*$")`
# Any markdown block that starts with ```python (or ```forward) is intercepted by `astra_harness.py` as a tool call to run in the kernel!
# Whereas if we use ```diff or ```text or ```bash or just ``` (or indent by 4 spaces) in our final report, `find_blocks(text)` returns `[]` and `astra_harness.py` treats the reply as the FINAL REPORT and finishes cleanly!
print("WHAT A CATCH! Checking `find_blocks` in `second_report`:")
OPEN = re.compile(r"^```(python|forward)\s*$", re.M)
print("Matches in draft_text:", OPEN.findall(draft_text))
```