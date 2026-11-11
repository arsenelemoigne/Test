```python
# Let's understand why the conversation restarted!
# When a conversation restarts in this harness, the initial prompt is re-sent with HANDOFF appended, OR is this a continuation of the first or second turn?
# Wait: look at `notes.md` and `report_draft.md` and `PATH1_PLAN.md`! Let's read `PATH1_PLAN.md`, `FACT_CHECK.md`, `CLAUDE_REVIEW.md` and `LADDER_DESIGN_FOR_REVIEW.md`!
for fn in ['PATH1_PLAN.md', 'FACT_CHECK.md', 'CLAUDE_REVIEW.md']:
    print("="*60, fn)
    with open(os.path.join(pack_dir, fn)) as f:
        print(f.read())
```