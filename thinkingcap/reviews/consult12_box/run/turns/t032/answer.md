```python
# Let's check Dev-512 vs Full Test across all models in the rpE80 -> rpK / rpL lineage!
# Let's compute Dev-512 metrics from dev512_per_question.csv for base, rpE80, rpK-20, rpK-40, rpK-60, rpL-20, rpL-40, rpL-60, p1r1s0-100/150/200!
dev_models = ['base', 'rpE80', 'rpK-20', 'rpK-40', 'rpK-60', 'rpL-20', 'rpL-40', 'rpL-60', 'p1r1s0-100', 'p1r1s0-150', 'p1r1s0-200']

def compute_dev_metrics(df, model):
    res = {}
    for ds, g in df.groupby('part'):
        b_tok = (g['base:thinking'] + g['base:response']).mean()
        m_tok = (g[f'{model}:thinking'] + g[f'{model}:response']).mean()
        b_acc = (g['base:right_of_8'] / 8.0 * 100.0).mean()
        m_acc = (g[f'{model}:right_of_8'] / 8.0 * 100.0).mean()
        ratio = m_tok / b_tok
        saved = (1.0 - ratio) * 100.0
        d_acc = m_acc - b_acc
        m_thk = g[f'{model}:thinking'].mean()
        m_rsp = g[f'{model}:response'].mean()
        m_run = (g[f'{model}:runaway_of_8'] / 8.0 * 100.0).mean()
        res[ds] = {
            'saved': saved, 'd_acc': d_acc, 'acc': m_acc, 'ratio': ratio,
            'thk': m_thk, 'rsp': m_rsp, 'tot': m_tok, 'run': m_run, 'n': len(g)
        }
    # Official BFCL weighting on Dev-512: simple 400/840, multiple 200/840, irrelevance 240/840
    w_bfcl = {'bfcl_simple_python': 400/840, 'bfcl_multiple': 200/840, 'bfcl_irrelevance': 240/840}
    b_tok_bfcl = sum(w_bfcl[k] * (df[df['part']==k]['base:thinking'] + df[df['part']==k]['base:response']).mean() for k in w_bfcl)
    m_tok_bfcl = sum(w_bfcl[k] * res[k]['tot'] for k in w_bfcl)
    d_acc_bfcl = sum(w_bfcl[k] * res[k]['d_acc'] for k in w_bfcl)
    ratio_bfcl = m_tok_bfcl / b_tok_bfcl
    res['bfcl_pooled'] = {
        'saved': (1.0 - ratio_bfcl) * 100.0,
        'd_acc': d_acc_bfcl,
        'ratio': ratio_bfcl
    }
    parts4 = ['svamp', 'mbpp', 'ifeval', 'bfcl_pooled']
    gmean_ratio = np.prod([res[p]['ratio'] for p in parts4]) ** 0.25
    res['headline'] = {
        'saved': (1.0 - gmean_ratio) * 100.0,
        'd_acc': np.mean([res[p]['d_acc'] for p in parts4])
    }
    return res

dev_stats = {m: compute_dev_metrics(dev_q, m) for m in dev_models}

print("=== Dev-512 vs Full Test Comparison ===")
pair_map = [
    ('rpE80', 'rpE80'),
    ('rpK-20', 'rpK20'),
    ('rpK-40', 'rpK40'),
    ('rpK-60', 'rpK60'),
    ('rpL-20', None),
    ('rpL-40', None),
    ('rpL-60', 'rpL60'),
    ('p1r1s0-200', 'p1r1s0'),
]
for dm, fm in pair_map:
    ds = dev_stats[dm]['headline']
    d_vs_e_sav = ds['saved'] - dev_stats['rpE80']['headline']['saved']
    d_vs_e_acc = ds['d_acc'] - dev_stats['rpE80']['headline']['d_acc']
    if fm is not None:
        fs_sav = all_stats[fm]['headline']['saved']
        # official accuracy
        g = ft_if_full[ft_if_full['model'] == fm]
        off_if_acc = g.groupby(['id', 'sample'])['passed'].all().mean() * 100.0 - 58.73382624768946
        fs_acc = (all_stats[fm]['svamp']['d_acc'] + all_stats[fm]['mbpp']['d_acc'] + off_if_acc + all_stats[fm]['bfcl_pooled']['d_acc']) / 4.0
        f_vs_e_sav = fs_sav - 47.043
        f_vs_e_acc = fs_acc - (-0.950)
        print(f"{dm:10s} | Dev: {ds['saved']:5.2f}% / {ds['d_acc']:+5.2f} (vs rpE80: {d_vs_e_sav:+5.2f}% / {d_vs_e_acc:+5.2f}) | Full: {fs_sav:5.2f}% / {fs_acc:+5.2f} (vs rpE80: {f_vs_e_sav:+5.2f}% / {f_vs_e_acc:+5.2f}) | Full-Dev offset: {fs_sav-ds['saved']:+5.2f}% / {fs_acc-ds['d_acc']:+5.2f}")
    else:
        print(f"{dm:10s} | Dev: {ds['saved']:5.2f}% / {ds['d_acc']:+5.2f} (vs rpE80: {d_vs_e_sav:+5.2f}% / {d_vs_e_acc:+5.2f}) | Full: [not run]")
```