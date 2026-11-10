```python
# Let's start recording notes in work/notes.md as instructed!
# Also let's do quantitative calculations on all_runs.json and numbers from the log.

notes_path = os.path.join(work_dir, "notes.md")
draft_path = os.path.join(work_dir, "report_draft.md")

# First let's check all_runs.json statistics, curve fit, screen vs full test, per-part trade-offs, etc.
# Wait: how is the official headline calculated from the 6 parts?
# Let's verify the official headline formula:
# "Headline = the 4 benchmarks equally weighted (BFCL's three parts pooled 400/200/240):
#  accuracy change = mean of the 4 changes; tokens saved = 1 - geometric mean of the 4 token ratios."
# Wait: how is BFCL's token ratio pooled? Is it weighted average of token counts or weighted average of ratios (400/840, 200/840, 240/840)?
# Let's check against the 8 full runs in all_runs.json!

for idx, r in full_runs.iterrows():
    p = r['parts']
    acc_v = p['v']['acc']
    acc_c = p['c']['acc']
    acc_f = p['f']['acc']
    acc_bfcl = (400 * p['s']['acc'] + 200 * p['m']['acc'] + 240 * p['i']['acc']) / 840.0
    acc_4 = (acc_v + acc_c + acc_f + acc_bfcl) / 4.0
    acc_6eq = (acc_v + acc_c + acc_f + p['s']['acc'] + p['m']['acc'] + p['i']['acc']) / 6.0
    
    # For saved: part 'saved' is rounded to integer in parts dict, let's check both arithmetic/geometric
    rv = 1 - p['v']['saved']/100.0
    rc = 1 - p['c']['saved']/100.0
    rf = 1 - p['f']['saved']/100.0
    rs = 1 - p['s']['saved']/100.0
    rm = 1 - p['m']['saved']/100.0
    ri = 1 - p['i']['saved']/100.0
    rbfcl_arith = (400*rs + 200*rm + 240*ri)/840.0
    saved_4 = (1 - (rv * rc * rf * rbfcl_arith)**0.25) * 100
    print(f"{r['key']:15s}: logged saved={r['saved']:5.1f} vs calc4={saved_4:5.1f} | logged acc={r['acc']:+5.2f} vs calc4={acc_4:+5.2f} (6eq={acc_6eq:+5.2f})")
```