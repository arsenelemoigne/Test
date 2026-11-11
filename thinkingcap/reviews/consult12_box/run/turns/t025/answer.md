```python
# Look at those numbers! That is a super important distinction:
# 1) When the official script prints `+-0.7 accuracy and +-1.2 saving` (95% CI), how does it compute that?
# It computes a paired bootstrap across the 2,759 QUESTIONS (treating the 2,759 questions as a random sample from a hypothetical infinite question population AND including the 8-sample variance of BOTH `base` and `cand`).
# Let's verify how the official script's paired question-level bootstrap CI vs pure 8-sample MC variance works!
def paired_question_se(model, ref='base'):
    # Question-level means for model and ref
    sub_m = ft_ans[ft_ans['model'] == model].copy()
    sub_r = ft_ans[ft_ans['model'] == ref].copy()
    sub_m['part4'] = np.where(sub_m['dataset'].str.startswith('bfcl'), 'bfcl_pooled', sub_m['dataset'])
    sub_r['part4'] = np.where(sub_r['dataset'].str.startswith('bfcl'), 'bfcl_pooled', sub_r['dataset'])
    
    qm = sub_m.groupby(['part4', 'id']).agg(acc_m=('right_off', 'mean'), tok_m=('total_tokens', 'mean')).reset_index()
    qr = sub_r.groupby(['part4', 'id']).agg(acc_r=('right_off', 'mean'), tok_r=('total_tokens', 'mean')).reset_index()
    q = qm.merge(qr, on=['part4', 'id'])
    q['d_acc'] = (q['acc_m'] - q['acc_r']) * 100.0
    
    part_q_se = {}
    for p, g in q.groupby('part4'):
        N = len(g)
        se_d_acc = g['d_acc'].std(ddof=1) / np.sqrt(N)
        # Delta method for ratio = mean(tok_m) / mean(tok_r)
        mu_m = g['tok_m'].mean()
        mu_r = g['tok_r'].mean()
        ratio = mu_m / mu_r
        # linearized variable for ratio: (tok_m - ratio * tok_r) / mu_r
        lin_r = (g['tok_m'] - ratio * g['tok_r']) / mu_r * 100.0
        se_saved = lin_r.std(ddof=1) / np.sqrt(N)
        part_q_se[p] = (se_saved, se_d_acc)
    
    h_se_acc = 0.25 * np.sqrt(sum(part_q_se[p][1]**2 for p in ['svamp', 'mbpp', 'ifeval', 'bfcl_pooled']))
    gmean_r = (100.0 - all_stats[model]['headline']['saved']) / 100.0
    h_var_saved = 0.0
    for p in ['svamp', 'mbpp', 'ifeval', 'bfcl_pooled']:
        r_p = all_stats[model][p]['ratio']
        se_r_p = part_q_se[p][0]
        h_var_saved += (0.25 * (gmean_r / r_p) * se_r_p) ** 2
    h_se_saved = np.sqrt(h_var_saved)
    return h_se_saved, h_se_acc, part_q_se

for m in ['rpE80', 'rpK20', 'rpK40', 'rpK60', 'rpL60']:
    hs_b, ha_b, pq_b = paired_question_se(m, 'base')
    hs_e, ha_e, pq_e = paired_question_se(m, 'rpE80')
    print(f"{m:6s} vs base  | Question-bootstrap SE: Saved +- {hs_b:.3f}% (95% CI +- {1.96*hs_b:.2f}%), Acc +- {ha_b:.3f} (95% CI +- {1.96*ha_b:.2f})")
    print(f"{m:6s} vs rpE80 | Question-bootstrap SE: Saved +- {hs_e:.3f}% (95% CI +- {1.96*hs_e:.2f}%), Acc +- {ha_e:.3f} (95% CI +- {1.96*ha_e:.2f})")
```