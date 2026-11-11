```python
# Look at SVAMP Q4_long in rpL-60 vs rpK-60:
# Why did Q4_long drop another -1.30 pts in rpL-60 (-4.25 -> -5.55) while thinking fell another -47.5 tokens (544.1 -> 496.6)?
# Because rpL had 16 math anti-loop pairs (301 vs 4,014 tokens)! Even though SVAMP only had 0.04% runaways in rpE-80, those 16 math anti-loop pairs penalized long math thinking on Q4_long (`544.1 -> 496.6` tokens)!
# And in `rpM`, those 16 math anti-loop pairs were REMOVED (`--loop-response-only ifeval` so `svamp` has 0 anti-loop pairs!) AND replaced by 120 widened guards + 120 length-held `right_cr` pairs on SVAMP-like questions!

# Now let's check BFCL in rpL-60 vs rpK-60, rpK-40, rpE-80!
for ds in ['bfcl_simple_python', 'bfcl_multiple', 'bfcl_irrelevance']:
    g = ft_q[ft_q['dataset'] == ds]
    print(f"\n=== {ds} (N={len(g)}) ===")
    for m in ['base', 'rpE80', 'rpI120', 'rpK20', 'rpK40', 'rpK60', 'rpL60', 'p1r1s0']:
        acc = (g[f'{m}:right_of_8'] / 8.0 * 100.0).mean()
        d_acc = acc - (g['base:right_of_8'] / 8.0 * 100.0).mean()
        call = (g[f'{m}:calls_of_8'] / 8.0 * 100.0).mean()
        d_call = call - (g['base:calls_of_8'] / 8.0 * 100.0).mean()
        acc_given_call = (g[f'{m}:right_of_8'].sum() / g[f'{m}:calls_of_8'].sum() * 100.0) if ds != 'bfcl_irrelevance' else np.nan
        thk = g[f'{m}:thinking'].mean()
        rsp = g[f'{m}:response'].mean()
        print(f"  {m:7s}: acc {acc:5.2f}% ({d_acc:+6.2f}) | call {call:5.2f}% ({d_call:+6.2f}) | acc|call {acc_given_call:5.2f}% | thk {thk:5.1f} + rsp {rsp:4.1f}")
```