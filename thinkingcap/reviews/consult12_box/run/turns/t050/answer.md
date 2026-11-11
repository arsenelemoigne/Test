```python
# Look at `rpl_df` vs `rpm_df`:
# How many questions in `rpL_pairs.csv` overlap with `rpM_pairs` on IFEval `right_count` (the 78 true length-rule pairs in `rpM`) or `right_cr`?
# Let's check!
rpl_if_right = set(rpl_df[(rpl_df['part'] == 'ifeval') & (rpl_df['kind'] == 'right')]['id'])
rpm_if_right = set(rpm_df[(rpm_df['part'] == 'ifeval') & (rpm_df['kind'] == 'right')]['id'])
print("IFEval right questions in rpL:", len(rpl_if_right))
print("IFEval right questions in rpM:", len(rpm_if_right))
print("Overlap:", len(rpl_if_right & rpm_if_right))

# More importantly: why did `rpL` only see `60 steps * 16 = 960` pairs (`0.90` epochs of `rpL_pairs.csv`) where 440 of those pairs were `shorten` pairs, and on IFEval `rpL` used `--count-mode atleast --count-resp-cap 1.35` (where 2/3 of counting pairs were placeholders/highlights/letters and `chosen_response` was capped at `1.35x + 25`!) and on math `rpL` used `guard < 0.5x` + shortest `right_cr` + 16 math `loop` pairs?
# Let's check the token statistics of `rpl_df` vs `rpm_df` on IFEval `right` and math `guard`/`right`!
print("\n=== rpL vs rpM IFEval 'right' pairs (counting + be-right) ===")
print("rpL IFEval right: mean chosen_rsp =", rpl_df[(rpl_df['part']=='ifeval') & (rpl_df['kind']=='right')]['chosen_response'].mean(),
      "vs rej_rsp =", rpl_df[(rpl_df['part']=='ifeval') & (rpl_df['kind']=='right')]['rejected_response'].mean(),
      "| >=300 tok:", (rpl_df[(rpl_df['part']=='ifeval') & (rpl_df['kind']=='right')]['chosen_response'] >= 300).sum())
print("rpM IFEval right: mean chosen_rsp =", rpm_df[(rpm_df['part']=='ifeval') & (rpm_df['kind']=='right')]['chosen_response'].mean(),
      "vs rej_rsp =", rpm_df[(rpm_df['part']=='ifeval') & (rpm_df['kind']=='right')]['rejected_response'].mean(),
      "| >=300 tok:", (rpm_df[(rpm_df['part']=='ifeval') & (rpm_df['kind']=='right')]['chosen_response'] >= 300).sum())

print("\n=== rpL vs rpM Math pairs ===")
print("rpL svamp (all): mean chosen_thk =", rpl_df[rpl_df['part']=='svamp']['chosen_thinking'].mean(),
      "vs rej_thk =", rpl_df[rpl_df['part']=='svamp']['rejected_thinking'].mean(),
      "| chosen > rej:", (rpl_df[rpl_df['part']=='svamp']['chosen_thinking'] > rpl_df[rpl_df['part']=='svamp']['rejected_thinking']).mean()*100)
print("rpM svamp_hard (all): mean chosen_thk =", rpm_df[rpm_df['part']=='svamp_hard']['chosen_thinking'].mean(),
      "vs rej_thk =", rpm_df[rpm_df['part']=='svamp_hard']['rejected_thinking'].mean(),
      "| chosen > rej:", (rpm_df[rpm_df['part']=='svamp_hard']['chosen_thinking'] > rpm_df[rpm_df['part']=='svamp_hard']['rejected_thinking']).mean()*100)
```