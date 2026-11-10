```python
# Let's examine even more carefully what happens in the level definition b*(q)!
# Look at the simulation result above:
# Even for a single question where true accuracy at rung r is p_r = 0.50 and true natural accuracy is p_nat = 0.85:
# With 8 samples at each rung, what is the probability that at LEAST ONE early rung (say among the first 5 rungs where true accuracy is only 0.50!) gets >= k0 - 1?
# Let's compute the exact probabilities for specific (p_early, p_nat) pairs!

from scipy.stats import binom

for p_nat in [0.90, 0.80, 0.75, 0.60]:
    for p_early in [p_nat - 0.30, p_nat - 0.20, p_nat - 0.15, p_nat - 0.10]:
        if p_early <= 0: continue
        # Suppose there are M=4 early rungs where true accuracy is p_early
        # k0 ~ Binom(8, p_nat)
        prob_pass_one_rung = 0.0
        prob_pass_any_4_rungs = 0.0
        for k0 in range(1, 9):
            pk0 = binom.pmf(k0, 8, p_nat)
            # Need kr >= k0 - 1
            p_ge = 1.0 - binom.cdf(k0 - 2, 8, p_early) if k0 >= 2 else 1.0
            prob_pass_one_rung += pk0 * p_ge
            prob_pass_any_4_rungs += pk0 * (1.0 - (1.0 - p_ge)**4)
        print(f"p_nat={p_nat:.2f}, p_early={p_early:.2f} (true loss={100*(p_early-p_nat):+5.1f} pp): P(pass 1 rung)={prob_pass_one_rung*100:5.1f}%, P(pass >=1 of 4 early rungs)={prob_pass_any_4_rungs*100:5.1f}%")
```