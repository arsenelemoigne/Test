```python
# Look at the pure 8-sample MC SE of IFEval:
# For a single model on the fixed 541 test questions, the 8-sample MC SE is 0.433 pts!
# And for the bump `acc(rpK40) - 0.5*(acc(rpK20) + acc(rpK60))`, accounting for the within-question sample correlation (r ~ 0.22), the pure 8-sample MC SE of `acc40 - 0.5*(acc20 + acc60)` on IFEval is:
# sqrt(1 + 0.25 + 0.25 + 0.5*0.09 - 2*0.22) * 0.434 = 1.05 * 0.434 = 0.456 pts (or question-bootstrap SE = 0.519 pts)!
# Wait: also notice that at step 40, IFEval response tokens rose to `241.5` (`+11.8` tokens over step 20 and step 60, and `+85` tokens on `at_least_N`: `560.6` vs `475.3` and `525.5`), which lifted `at_least_N` by `+3.1` pts (`21.34%` vs `17.03%` and `19.40%`, contributing `+0.33` pts to IFEval), while `no_counting` jumped `+0.62` pts (`62.41%` vs `61.70%` and `61.88%`, contributing `+0.32` pts) and `other_counting` jumped `+1.22` pts (`57.17%` vs `56.49%` and `55.40%`, contributing `+0.41` pts) and `fewer_than_N` jumped `+1.74` pts (`71.88%` vs `69.44%` and `70.83%`, contributing `+0.12` pts).
# Notice that on Dev-512, `rpK-40`'s IFEval was `-2.4` (between `-0.7` at step 20 and `-3.4` at step 60)!
# Let's compute the Empirical Bayes (James-Stein / Gaussian process / local regression) shrinkage estimate of `rpK-40`'s TRUE infinite-sample accuracy and saving on the 2,759 test questions!
# Let's do this rigorously in Python!

# Wait: what is the true underlying curve E[acc(step)] and E[saved(step)] across step = 0 (rpE80), 20, 40, 60?
# For each benchmark b in {svamp, mbpp, ifeval, bfcl_pooled}:
# We know the exact within-question 8-sample MC noise covariance Sigma_MC across (step 0, 20, 40, 60)!
# Specifically, for a fixed model m on the 2,759 test questions, the observed score Y_m = mu_m + eps_m, where mu_m is the exact infinite-sample expectation E_{samples}[score | 2759 test questions] (plus any difference relative to base's 8-sample realization, which is fixed across all models because base's rollouts are fixed!).
# Wait: `base` rollouts are FIXED across all evaluations! Every candidate model is compared against the SAME stored `base` rollouts!
# Let's verify that! Yes! `dev_score.py` and the full test compare every model against the SAME `base` rollout file!
# That means When comparing `rpK-20`, `rpK-40`, `rpK-60`, `rpE-80`, `base`'s sampling noise is a constant offset across all models, and the ONLY random variation in a model's full-test score (given the fixed evaluation protocol and fixed `base` rollouts) is the candidate model's own 8 MC samples per question (`SE_MC = +-0.20` pts on headline accuracy, `+-0.34%` on headline saving)!
# Wait AND if we consider question-sampling uncertainty (if a reviewer asks "what if the test set had different questions or what is the paired bootstrap CI against base?"), the paired question-bootstrap SE vs base is `+-0.36` pts (`95% CI +-0.71` pts) on accuracy and `+-0.59%` (`95% CI +-1.16%`) on saving!

# Let's check both! Let's write a script to decompose the variance of `rpK-40` vs its neighbors under BOTH:
# 1) Fixed-test-set 8-sample MC noise (`SE_MC = 0.198` pts on headline accuracy, `0.340%` on headline saving; difference `rpK40 - 0.5*(rpK20 + rpK60)` has `SE_MC_bump`), and
# 2) Paired question-bootstrap noise (`SE_boot = 0.361` pts vs base, `0.216` pts for the midpoint bump `rpK40 - 0.5*(rpK20 + rpK60)`).

for p in ['svamp', 'mbpp', 'ifeval', 'bfcl_pooled']:
    # Compute within-question sample variance of bump = y40 - 0.5*(y20 + y60) across s=0..7
    sub = ft_ans[ft_ans['model'].isin(['rpK20', 'rpK40', 'rpK60'])].copy()
    sub['part4'] = np.where(sub['dataset'].str.startswith('bfcl'), 'bfcl_pooled', sub['dataset'])
    sub_p = sub[sub['part4'] == p]
    piv = sub_p.pivot(index=['id', 'sample'], columns='model', values='right_off') * 100.0
    piv['bump'] = piv['rpK40'] - 0.5 * (piv['rpK20'] + piv['rpK60'])
    q_var = piv.groupby('id')['bump'].apply(lambda x: np.var(x, ddof=1) / 8.0)
    mc_se_bump = np.sqrt(q_var.sum()) / len(q_var)
    obs_bump = piv['bump'].mean()
    print(f"{p:12s} | Obs bump: {obs_bump:+6.2f} pts | Pure 8-sample MC SE of bump: {mc_se_bump:5.3f} pts | z_MC = {obs_bump/mc_se_bump:+5.2f}")
```