```python
# AHA! Look at prompt_pass (all) vs base (58.73%):
# rpE80: 56.17 - 58.73 = -2.56% (rounds to -2.6!)
# rpK20: 55.66 - 58.73 = -3.07% (rounds to -3.1!)
# rpK40: 56.86 - 58.73 = -1.87% (rounds to -1.9!)
# rpK60: 55.73 - 58.73 = -3.00% (rounds to -3.0!)
# rpL60: 55.36 - 58.73 = -3.37% (rounds to -3.4!)
# p1r1s0: 59.15 - 58.73 = +0.42% (rounds to +0.4!)
# Why? Because in official evaluation, even if an answer doesn't have </think> (so runaway=1, meaning the whole output is treated as response by official IFEval if </think> is missing, whereas our local table parser sets response="" when </think> is missing!).
# Let's verify this! That explains the exact difference between official IFEval and finished-only IFEval!
for m in ['base', 'rc1', 'rpE80', 'rpI120', 'rpK20', 'rpK40', 'rpK60', 'rpL60', 'p1r1s0']:
    g = ft_if_full[ft_if_full['model'] == m]
    off_if_acc = g.groupby(['id', 'sample'])['passed'].all().mean() * 100.0
    base_if_acc = 58.73382624768946
    d_if_off = off_if_acc - base_if_acc
    d_sv = all_stats[m]['svamp']['d_acc']
    d_mb = all_stats[m]['mbpp']['d_acc']
    d_bf = all_stats[m]['bfcl_pooled']['d_acc']
    h_off = (d_sv + d_mb + d_if_off + d_bf) / 4.0
    h_fin = all_stats[m]['headline']['d_acc']
    sav = all_stats[m]['headline']['saved']
    print(f"{m:8s} | Saved: {sav:6.2f}% | Official d_acc: {h_off:+6.2f} (fin: {h_fin:+6.2f}) | SVAMP {d_sv:+5.2f}, MBPP {d_mb:+5.2f}, IFEval_off {d_if_off:+5.2f} (fin {all_stats[m]['ifeval']['d_acc']:+5.2f}), BFCL {d_bf:+5.2f}")
```