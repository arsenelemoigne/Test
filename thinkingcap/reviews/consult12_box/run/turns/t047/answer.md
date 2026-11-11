```python
# STOP AND LOOK AT THAT SIMULATION OUTPUT (`sim_ifeval_pairs_on_ft`):
# `rpE80 (on 541 FT IFEval Qs): right_count pairs = 40 (chosen rsp 343.9 vs rej 245.3, >=300 tok: 19)`
# `rpK60 (on 541 FT IFEval Qs): right_count pairs = 27 (chosen rsp 413.1 vs rej 264.8, >=300 tok: 12)`
# `rpL60 (on 541 FT IFEval Qs): right_count pairs = 27 (chosen rsp 261.7 vs rej 191.1, >=300 tok: 7)`
#
# READ THAT AGAIN!
# On `rpL-60`'s own rollouts, because `rpL-60` already compressed its `at_least_N` responses down to `440.5` tokens (`13.58%` accuracy), `rpL-60`'s own `right_count` pairs only have a mean chosen response of **`261.7` tokens** (and only **7 pairs >= 300 tokens** out of 541 questions, vs **19 pairs >= 300 tokens** and **`343.9` mean tokens** in `rpE-80`'s rollouts)!
# AND what about SVAMP math? Let's check `sim_svamp_pairs_on_ft` on `rpE80` vs `rpK60` vs `rpL60`!

def sim_svamp_pairs_on_ft(model):
    sub_a = ft_ans[(ft_ans['model'] == model) & (ft_ans['dataset'] == 'svamp')].copy()
    n_guard = 0
    guard_c_thk, guard_w_thk = [], []
    n_right = 0
    right_c_thk, right_w_thk = [], []
    for qid, g in sub_a.groupby('id'):
        rights = g[(g['right_off'] == 1.0) & (g['runaway'] == 0)].sort_values('thinking')
        wrongs = g[(g['right_off'] == 0.0) & (g['runaway'] == 0)].sort_values('thinking')
        if len(rights) > 0 and len(wrongs) > 0:
            tmed = rights['thinking'].median()
            w_under = wrongs[wrongs['thinking'] < 0.95 * tmed]
            if len(w_under) > 0:
                # nearest 0.75 * tmed
                w_pick = w_under.iloc[(w_under['thinking'] - 0.75 * tmed).abs().argsort().iloc[0]]
                c_band = rights[(rights['thinking'] >= 0.95 * tmed) & (rights['thinking'] <= 1.25 * tmed)]
                c_pick = c_band.iloc[0] if len(c_band) > 0 else rights.iloc[(rights['thinking'] - tmed).abs().argsort().iloc[0]]
                n_guard += 1
                guard_c_thk.append(c_pick['thinking'])
                guard_w_thk.append(w_pick['thinking'])
            # right_hold: 0.95 <= c/w <= 1.25 nearest 1.05
            best_pair = None
            best_dist = 1e9
            for _, r_row in rights.iterrows():
                for _, w_row in wrongs.iterrows():
                    rat = r_row['thinking'] / max(1, w_row['thinking'])
                    if 0.95 <= rat <= 1.25:
                        d = abs(rat - 1.05)
                        if d < best_dist:
                            best_dist = d
                            best_pair = (r_row['thinking'], w_row['thinking'])
            if best_pair is not None:
                n_right += 1
                right_c_thk.append(best_pair[0])
                right_w_thk.append(best_pair[1])
    print(f"{model:6s} (on 1000 FT SVAMP Qs): guard_wide = {n_guard:3d} (chosen {np.mean(guard_c_thk):.1f} vs rej {np.mean(guard_w_thk):.1f}) | right_hold = {n_right:3d} (chosen {np.mean(right_c_thk):.1f} vs rej {np.mean(right_w_thk):.1f})")

for m in ['rpE80', 'rpK60', 'rpL60']:
    sim_svamp_pairs_on_ft(m)
```