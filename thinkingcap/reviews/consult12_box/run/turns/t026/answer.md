```python
# Let's also check paired question-bootstrap SE between rpK-40 and rpK-20, and between rpK-40 and rpK-60, and between rpK-40 and the average of (rpK-20, rpK-60)!
hs_40_20, ha_40_20, pq_40_20 = paired_question_se('rpK40', 'rpK20')
hs_40_60, ha_40_60, pq_40_60 = paired_question_se('rpK40', 'rpK60')
print(f"rpK40 vs rpK20 | Paired SE: Saved +- {hs_40_20:.3f}% (95% CI +- {1.96*hs_40_20:.2f}%), Acc +- {ha_40_20:.3f} (95% CI +- {1.96*ha_40_20:.2f})")
print("   Per part SE (rpK40 vs rpK20):", {k: (round(v[0], 3), round(v[1], 3)) for k, v in pq_40_20.items()})
print(f"rpK40 vs rpK60 | Paired SE: Saved +- {hs_40_60:.3f}% (95% CI +- {1.96*hs_40_60:.2f}%), Acc +- {ha_40_60:.3f} (95% CI +- {1.96*ha_40_60:.2f})")
print("   Per part SE (rpK40 vs rpK60):", {k: (round(v[0], 3), round(v[1], 3)) for k, v in pq_40_60.items()})

# Let's check the exact difference between rpK-40 and 0.5*(rpK-20 + rpK-60) on every benchmark and test its z-score!
sub_20 = ft_ans[ft_ans['model'] == 'rpK20'].copy()
sub_40 = ft_ans[ft_ans['model'] == 'rpK40'].copy()
sub_60 = ft_ans[ft_ans['model'] == 'rpK60'].copy()

q20 = sub_20.groupby(['dataset', 'id']).agg(acc20=('right_off', 'mean'), tok20=('total_tokens', 'mean'), thk20=('thinking', 'mean'), rsp20=('response_tokens', 'mean'), call20=('right_off', 'count')).reset_index()
q40 = sub_40.groupby(['dataset', 'id']).agg(acc40=('right_off', 'mean'), tok40=('total_tokens', 'mean'), thk40=('thinking', 'mean'), rsp40=('response_tokens', 'mean')).reset_index()
q60 = sub_60.groupby(['dataset', 'id']).agg(acc60=('right_off', 'mean'), tok60=('total_tokens', 'mean'), thk60=('thinking', 'mean'), rsp60=('response_tokens', 'mean')).reset_index()

qall = q20.merge(q40, on=['dataset', 'id']).merge(q60, on=['dataset', 'id'])
qall['bump_acc'] = (qall['acc40'] - 0.5 * (qall['acc20'] + qall['acc60'])) * 100.0
qall['part4'] = np.where(qall['dataset'].str.startswith('bfcl'), 'bfcl_pooled', qall['dataset'])

print("\n=== Bump of rpK-40 over 0.5*(rpK-20 + rpK-60) ===")
for ds, g in qall.groupby('dataset'):
    mean_bump = g['bump_acc'].mean()
    se_bump = g['bump_acc'].std(ddof=1) / np.sqrt(len(g))
    z = mean_bump / se_bump
    print(f"  {ds:18s} (N={len(g):4d}): bump = {mean_bump:+6.2f} pts | SE = {se_bump:5.3f} | z = {z:+5.2f} | thk: {g['thk20'].mean():.1f} -> {g['thk40'].mean():.1f} -> {g['thk60'].mean():.1f} | rsp: {g['rsp20'].mean():.1f} -> {g['rsp40'].mean():.1f} -> {g['rsp60'].mean():.1f}")

part4_means = []
part4_ses = []
for p, g in qall.groupby('part4'):
    mb = g['bump_acc'].mean()
    sb = g['bump_acc'].std(ddof=1) / np.sqrt(len(g))
    part4_means.append(mb)
    part4_ses.append(sb)
    print(f"  [Part4] {p:12s} (N={len(g):4d}): bump = {mb:+6.2f} pts | SE = {sb:5.3f} | z = {mb/sb:+5.2f}")

h_bump = np.mean(part4_means)
h_bump_se = 0.25 * np.sqrt(sum(s**2 for s in part4_ses))
print(f"  [HEADLINE] bump = {h_bump:+6.2f} pts | SE = {h_bump_se:5.3f} | z = {h_bump/h_bump_se:+5.2f}")
```