```python
# LOOK AT `rpE-80 -> rpI-120` ON THE FULL TEST (`Exact Full-Test Transition of rpE-80 -> rpI-120`):
# That table is pure gold!
# Look at what `rpI-120` (accuracy-only repair on `rpE-80`'s own answers, 0 `shorten` pairs) did on the FULL TEST:
# - SVAMP: `-1.25 -> +0.02` (`+1.27` pts accuracy, `-7.13%` saving as thinking rose `306.7 -> 355.5` tokens, `+15.9%`)
# - MBPP+: `-1.03 -> +0.53` (`+1.55` pts accuracy, `-8.35%` saving as thinking rose `1089.8 -> 1269.0` tokens, `+16.4%`)
# - IFEval: `-2.56 -> -1.57` (`+0.99` pts accuracy, `-4.43%` saving as thinking rose `156.5 -> 163.2` and response rose `223.3 -> 247.4` tokens)
# - BFCL Pooled: `+1.04 -> +1.53` (`+0.49` pts accuracy, `-0.16%` saving!)
# - Headline: **`47.04% / -0.95 -> 41.91% / +0.13`** (`+1.08` pts accuracy for `-5.13%` saving, a slope of **`+0.211` accuracy per `-1%` saving**!).

# NOW APPLY THAT EXACT EMPIRICAL TRANSITION TO `rpL-60` (`59.27% saved / -1.69 official, -1.55 finished-only`) VS `rpK-60` (`54.43% saved / -1.44`):
# Let's compute what happens when you apply `rpN` (the targeted accuracy-only repair) to `rpL-60` vs `rpK-60`!
# 1. Look at the starting point and the budget of saving to spend:
#    - **`rpL-60` starts at `59.27% saved / -1.69 accuracy` (`+3.20` above the project curve—by far the highest point ever achieved!).**
#      It has **9.27 percentage points of excess saving above 50.0%** (and **6.27 points above 53.0%**!), and needs **`+0.70` pts of accuracy** to cross `-1.00` (or **`+1.10` pts** to hit `-0.60`).
#      At `rpI-120`'s empirical full-test repair slope (`+0.211` accuracy per `-1%` saving), spending `-4.0% to -5.0%` saving from `rpL-60` (`59.27% -> 54.3% to 55.3%` saved) recovers **`+0.85 to +1.10` accuracy points (`-1.69 -> -0.60 to -0.85`)**!
#      Even if `rpN` on `rpL-60` spends a full `-5.13%` saving (the entire 120-step `rpI-120` shift!), `rpL-60` STILL lands at **`54.14% saved / -0.61 accuracy`**—deep inside the goal box with **`+4.1%` saving margin and `+0.39` accuracy margin**!
#    - By contrast, **`rpK-60` starts at `54.43% saved / -1.44 accuracy` (`+2.38` above the curve).**
#      It has only **4.43 points of excess saving above 50.0%** (and only **1.43 points above 53.0%**!), while needing **`+0.45` pts** just to cross `-1.00` and **`+0.85` pts** to hit `-0.60`.
#      If `rpN` on `rpK-60` spends `-3.5% to -4.5%` saving, `rpK-60` drops to `49.9%–50.9%` saved—right back onto the `50%` cliff edge (and it *cannot* physically reach `53%+ saved at -0.6` because `-0.6` from `54.43% / -1.44` requires a slope of `+0.59` accuracy per `-1%` saving, nearly 3x steeper than `rpI-120`!).
# 2. Even better: WHY is `rpL-60`'s `-0.25` pt accuracy difference vs `rpK-60` (`-1.69` vs `-1.44`) the *easiest* part to repair?
#    Let's check where `rpL-60` differs from `rpK-60`:
#    - On **MBPP+ Code**, `rpL-60` is ALREADY **+0.36 pts more accurate (`-0.86` vs `-1.22`) AND +6.78% shorter (`62.21%` vs `55.43%` saved, runaways `0.33%` vs `0.53%`)**!
#    - On **IFEval `no_counting`** (48.8% of IFEval), `rpL-60` is ALREADY **+0.90 pts more accurate (`+0.99` above base vs `+0.09` in `rpK-60`)** and has **one-third fewer runaways (`1.78%` vs `2.70%`)**!
#    - The ONLY two places where `rpL-60` trails `rpK-60` are:
#      (a) **IFEval `length_rule` questions** (`number_words` at `-18.51%` vs `-12.50%`, `keywords:frequency` at `-11.01%` vs `-9.52%`), where `rpL`'s `1.35x` cap clipped `at_least_N` responses from `525` to `440` tokens (`13.58%` vs `19.40%`).
#      (b) **SVAMP Q4_long**, where `rpL`'s 16 math anti-loop pairs cut Q4 thinking from `544` to `497` tokens (`-2.75` vs `-2.35`).
#    - And because `rpL-60` ALREADY suppressed runaways (`1.78%` on IFEval, `0.33%` on MBPP+, `0.01%` on SVAMP) and ALREADY compressed `no_counting` IFEval thinking/responses (`112+180` tokens vs `123+230` in `rpK-60`), an accuracy-only repair (`rpN`) on top of `rpL-60` starts from a base whose non-counting IFEval and MBPP+ are already clean and compact!

# WAIT A SECOND! Let's check Question B for `rpN` on top of `rpL-60`:
# If we build `rpN` on top of `rpL-60`, what rollout source should we use to build `rpN`'s pairs?
# Let's compare TWO rollout sources for `rpN` on top of `rpL-60`:
# Source 1: **`rpL-60`'s own fresh rollouts** (`3,200 questions x 8`) vs
# Source 2: **Reusing `rpI` + `rpM-extra` rollouts** (`rpE-80`'s `4,500 + 1,200 = 5,700 questions x 8`, already on Drive!) vs
# Source 3: **Wait: what if `rpL-60` generates fresh rollouts on `svamp=1200, bfcl_simple_python=300, bfcl_multiple=220, bfcl_irrelevance=180, mbpp=450, ifeval=900` (`3,250 questions x 8`), OR what if on `at_least_N` `rpL-60` has fewer long right answers than `rpE-80`?**
# Let's check! Earlier we saw that on the 118 `length_rule` questions in the Full Test:
# - `rpE-80` had `51/118 (43.2%)` mixed questions (`mean rsp = 352.9` tokens).
# - `rpL-60` had `47/118 (39.8%)` mixed questions (`mean rsp = 320.6` tokens).
# Wait: on `fulltest_per_answer.csv` and `fulltest_ifeval_rule_verdicts.csv`, let's simulate running `repair_pairs.py`'s exact `length_rule` pair builder (`--count-mode length --count-resp-min 1.10 --count-resp-cap 1.75 --count-resp-pad 60`) on `rpL-60`'s full-test rollouts vs `rpK-60`'s full-test rollouts vs `rpE-80`'s full-test rollouts!
# We have the actual per-answer rollouts of `rpE80`, `rpK60`, and `rpL60` in `fulltest_per_answer.csv`! Let's test this right now!

def sim_ifeval_pairs_on_ft(model):
    sub_a = ft_ans[(ft_ans['model'] == model) & (ft_ans['dataset'] == 'ifeval')].copy()
    # merge ifeval_class and ifeval_rules from ft_q
    meta = ft_q[ft_q['dataset'] == 'ifeval'][['id', 'ifeval_class', 'ifeval_rules']].copy()
    meta['id'] = meta['id'].astype(str)
    sub_a = sub_a.merge(meta, on='id')
    
    n_count_pairs = 0
    count_chosen_rsp = []
    count_rej_rsp = []
    n_right_cr = 0
    n_loop = 0
    
    for qid, g in sub_a.groupby('id'):
        cls = g['ifeval_class'].iloc[0]
        rules = str(g['ifeval_rules'].iloc[0])
        is_len = (cls == 'at_least_N') or ('keywords:frequency' in rules) or ('number_paragraphs' in rules)
        
        rights = g[(g['right_off'] == 1.0) & (g['runaway'] == 0)].sort_values('thinking')
        wrongs = g[(g['right_off'] == 0.0) & (g['runaway'] == 0)].sort_values('thinking')
        loops = g[(g['runaway'] == 1)]
        
        if len(rights) > 0 and len(loops) > 0:
            n_loop += 1
            
        if len(rights) > 0 and len(wrongs) > 0:
            if is_len:
                taken = set()
                for _, rej in wrongs.iloc[:2].iterrows():
                    need = 1.10 * rej['response_tokens'] + 15
                    cap = 1.75 * rej['response_tokens'] + 60
                    fit = rights[(~rights['sample'].isin(taken)) &
                                 (rights['response_tokens'] >= need) &
                                 (rights['thinking'] <= 1.2 * rej['thinking']) &
                                 (rights['response_tokens'] <= cap)]
                    if len(fit) > 0:
                        pick = fit.iloc[0]
                        taken.add(pick['sample'])
                        n_count_pairs += 1
                        count_chosen_rsp.append(pick['response_tokens'])
                        count_rej_rsp.append(rej['response_tokens'])
            else:
                n_right_cr += 1
    print(f"{model:6s} (on 541 FT IFEval Qs): right_count pairs = {n_count_pairs:3d} (chosen rsp {np.mean(count_chosen_rsp):.1f} vs rej {np.mean(count_rej_rsp):.1f}, >=300 tok: {sum(x>=300 for x in count_chosen_rsp):2d}) | right_cr Qs = {n_right_cr:3d} | loop Qs = {n_loop:2d}")

for m in ['rpE80', 'rpK60', 'rpL60']:
    sim_ifeval_pairs_on_ft(m)
```