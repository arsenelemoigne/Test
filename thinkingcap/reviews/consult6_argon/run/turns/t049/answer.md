```python
# Let's check what happens EVEN IF you have 0 slack (kr >= k0) across 4 early rungs!
for p_nat in [0.90, 0.80, 0.60]:
    for p_early in [p_nat - 0.30, p_nat - 0.20, p_nat - 0.10]:
        prob_pass_one_0slack = 0.0
        prob_pass_any_4_0slack = 0.0
        for k0 in range(1, 9):
            pk0 = binom.pmf(k0, 8, p_nat)
            p_ge = 1.0 - binom.cdf(k0 - 1, 8, p_early)
            prob_pass_one_0slack += pk0 * p_ge
            prob_pass_any_4_0slack += pk0 * (1.0 - (1.0 - p_ge)**4)
        print(f"[0 slack: kr >= k0] p_nat={p_nat:.2f}, p_early={p_early:.2f} (loss={100*(p_early-p_nat):+5.1f} pp): 1 rung={prob_pass_one_0slack*100:5.1f}%, >=1 of 4 rungs={prob_pass_any_4_0slack*100:5.1f}%")
```