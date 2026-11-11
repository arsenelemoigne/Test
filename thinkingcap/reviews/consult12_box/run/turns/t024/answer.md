```python
# Let's compute BOTH:
# 1) Pure MC sampling noise (within-question variance across the 8 samples per question, fixing the 2,759 test questions):
#    For each question i, the 8 samples s=0..7 are i.i.d. draws from the model's temperature-0.6 distribution on question i.
#    So the sample variance of the mean of 8 draws on question i is s_i^2 / 8, where s_i^2 = var(y_{i,0..7}, ddof=1).
#    Across N_b questions in benchmark b, the variance of the benchmark mean due to finite (K=8) sampling on the fixed test set is:
#    Var_MC(mean_b) = (1 / N_b^2) * sum_{i=1}^{N_b} (s_i^2 / 8).
# 2) Question- + sampling-level paired bootstrap SE (resampling questions with replacement, or resampling both questions and samples) between two models (e.g. rpK40 vs base, or rpK40 vs rpK20/60).
# Wait: are the 8 samples across different models coupled by random seed in vLLM, or independent?
# Let's check the within-question sample correlation between rpK20, rpK40, rpK60!

piv_r = ft_ans.pivot(index=['dataset', 'id', 'sample'], columns='model', values='right_off')
piv_t = ft_ans.pivot(index=['dataset', 'id', 'sample'], columns='model', values='total_tokens')

# Let's check within-question sample correlation (after subtracting question mean):
q_mean_r = piv_r.groupby(level=['dataset', 'id']).transform('mean')
resid_r = piv_r - q_mean_r
print("Within-question sample correlation of accuracy (resid_r):")
print(resid_r[['base', 'rpE80', 'rpK20', 'rpK40', 'rpK60', 'rpL60']].corr().round(3))

# Now let's compute the exact within-question MC standard error (fixing the 2,759 questions, varying only the 8 MC draws per question) for a single model AND for the difference between two models!
def mc_se_model(model):
    sub = ft_ans[ft_ans['model'] == model].copy()
    sub['part4'] = np.where(sub['dataset'].str.startswith('bfcl'), 'bfcl_pooled', sub['dataset'])
    # For each question, compute sample variance across the 8 samples (ddof=1)
    q_stats = sub.groupby(['part4', 'id']).agg(
        acc_var=('right_off', lambda x: np.var(x * 100.0, ddof=1) / 8.0),
        tok_var=('total_tokens', lambda x: np.var(x, ddof=1) / 8.0),
        tok_mean=('total_tokens', 'mean')
    ).reset_index()
    
    part_se = {}
    for p, g in q_stats.groupby('part4'):
        N = len(g)
        se_acc = np.sqrt(g['acc_var'].sum()) / N
        se_tok = np.sqrt(g['tok_var'].sum()) / N
        mean_tok = g['tok_mean'].mean()
        # base mean tok
        b_mean_tok = ft_ans[(ft_ans['model'] == 'base') & (np.where(ft_ans['dataset'].str.startswith('bfcl'), 'bfcl_pooled', ft_ans['dataset']) == p)]['total_tokens'].mean()
        se_saved = (se_tok / b_mean_tok) * 100.0
        part_se[p] = (se_saved, se_acc)
    
    # Headline SE (pure MC sampling noise on the fixed 2759 questions for 1 model):
    # Headline acc = 0.25 * sum(acc_p) -> var = (1/16) * sum(se_acc_p^2)
    h_se_acc = 0.25 * np.sqrt(sum(part_se[p][1]**2 for p in ['svamp', 'mbpp', 'ifeval', 'bfcl_pooled']))
    # Headline saved = 1 - (prod r_p)^0.25 -> d(saved)/d(r_p) = -0.25 * (gmean_r / r_p)
    gmean_r = (100.0 - all_stats[model]['headline']['saved']) / 100.0
    h_var_saved = 0.0
    for p in ['svamp', 'mbpp', 'ifeval', 'bfcl_pooled']:
        r_p = all_stats[model][p]['ratio']
        se_r_p = part_se[p][0] # in percentage points of ratio
        h_var_saved += (0.25 * (gmean_r / r_p) * se_r_p) ** 2
    h_se_saved = np.sqrt(h_var_saved)
    return h_se_saved, h_se_acc, part_se

for m in ['base', 'rpE80', 'rpK20', 'rpK40', 'rpK60', 'rpL60']:
    h_s, h_a, pse = mc_se_model(m)
    print(f"{m:6s} | Pure 8-sample MC SE (single model, fixed 2759 Qs): Saved +- {h_s:.3f}% (95% CI +- {1.96*h_s:.2f}%), Acc +- {h_a:.3f} pts (95% CI +- {1.96*h_a:.2f} pts)")
    print("   Per part SE (saved%, acc_pts):", {k: (round(v[0], 3), round(v[1], 3)) for k, v in pse.items()})
```