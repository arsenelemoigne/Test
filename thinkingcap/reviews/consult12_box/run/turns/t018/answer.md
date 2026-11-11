```python
# Let's compute exact full-test statistics for all models in fulltest_per_question.csv!
print("Datasets in fulltest_per_question:", ft_q['dataset'].value_counts().to_dict())

models = ['base', 'rc1', 'rpE80', 'rpI120', 'rpK20', 'rpK40', 'rpK60', 'rpL60', 'p1r1s0']

def compute_full_metrics(df, model):
    res = {}
    for ds, g in df.groupby('dataset'):
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
        m_call = (g[f'{model}:calls_of_8'] / 8.0 * 100.0).mean()
        res[ds] = {
            'saved': saved, 'd_acc': d_acc, 'acc': m_acc, 'ratio': ratio,
            'thk': m_thk, 'rsp': m_rsp, 'tot': m_tok, 'run': m_run, 'call': m_call, 'n': len(g)
        }
    # Pooled BFCL (400 simple, 200 multiple, 240 irrelevance = 840 questions)
    bfcl_g = df[df['dataset'].str.startswith('bfcl')]
    b_tok_bfcl = (bfcl_g['base:thinking'] + bfcl_g['base:response']).mean()
    m_tok_bfcl = (bfcl_g[f'{model}:thinking'] + bfcl_g[f'{model}:response']).mean()
    b_acc_bfcl = (bfcl_g['base:right_of_8'] / 8.0 * 100.0).mean()
    m_acc_bfcl = (bfcl_g[f'{model}:right_of_8'] / 8.0 * 100.0).mean()
    ratio_bfcl = m_tok_bfcl / b_tok_bfcl
    res['bfcl_pooled'] = {
        'saved': (1.0 - ratio_bfcl) * 100.0,
        'd_acc': m_acc_bfcl - b_acc_bfcl,
        'acc': m_acc_bfcl,
        'ratio': ratio_bfcl,
        'thk': bfcl_g[f'{model}:thinking'].mean(),
        'rsp': bfcl_g[f'{model}:response'].mean(),
        'tot': m_tok_bfcl,
        'run': (bfcl_g[f'{model}:runaway_of_8'] / 8.0 * 100.0).mean(),
        'n': len(bfcl_g)
    }
    # 4-part headline: svamp, mbpp, ifeval, bfcl_pooled
    parts4 = ['svamp', 'mbpp', 'ifeval', 'bfcl_pooled']
    gmean_ratio = np.prod([res[p]['ratio'] for p in parts4]) ** 0.25
    headline_saved = (1.0 - gmean_ratio) * 100.0
    headline_d_acc = np.mean([res[p]['d_acc'] for p in parts4])
    headline_acc = np.mean([res[p]['acc'] for p in parts4])
    res['headline'] = {'saved': headline_saved, 'd_acc': headline_d_acc, 'acc': headline_acc}
    return res

all_stats = {m: compute_full_metrics(ft_q, m) for m in models}

for m in models:
    h = all_stats[m]['headline']
    print(f"{m:8s} | Headline: {h['saved']:6.2f}% / {h['d_acc']:+6.2f} (acc {h['acc']:5.2f}%)")
    for p in ['svamp', 'mbpp', 'ifeval', 'bfcl_simple_python', 'bfcl_multiple', 'bfcl_irrelevance', 'bfcl_pooled']:
        s = all_stats[m][p]
        print(f"   {p:18s}: saved {s['saved']:5.2f}% | d_acc {s['d_acc']:+6.2f} (acc {s['acc']:5.2f}%) | thk {s['thk']:6.1f} + rsp {s['rsp']:5.1f} | run {s['run']:4.2f}% | call {s.get('call', 0):5.2f}%")
```