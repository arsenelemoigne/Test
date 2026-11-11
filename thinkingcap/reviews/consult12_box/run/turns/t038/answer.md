```python
# LOOK AT THAT TABLE (`HEAD-TO-HEAD: rpK-60 vs rpL-60 on the Full Test`)!
# This is an eye-opening comparison. Let's analyze every single line:
# 1. `rpL-60` has **+4.84% MORE full-test saving** (`59.27%` vs `54.43%`—a massive **9.27 points of excess saving above 50%**, vs 4.43 points for `rpK-60`) for only **`-0.25` pts official accuracy (`-1.69` vs `-1.44`)** and **`-0.11` pts finished-only accuracy (`-1.55` vs `-1.44`)**!
# 2. Look at why `rpL-60`'s official IFEval (`-3.37`) vs `rpK-60` (`-3.00`) is `-0.37` pts while finished-only IFEval is **`+0.18` pts BETTER (`-2.82` vs `-3.00`)**:
#    - In `rpK-60`, `2.70%` of IFEval answers are runaways (`9.7` right runaways by accident out of `117` runaways when official IFEval grades the unclosed thinking trace as response!), whereas in `rpL-60`, runaways fell by one-third to **`1.78%`**, and on finished answers `rpL-60` is actually `+0.18` pts *more* accurate than `rpK-60` (`54.94%` vs `54.76%`) while saving **`+8.59%` MORE tokens on IFEval (`57.94%` vs `49.35%`)**!
# 3. Look at MBPP+ Code:
#    - `rpL-60` is BOTH **+6.78% shorter (`62.21%` vs `55.43%` saved; `753.4` vs `896.8` thinking tokens; runaways cut from `0.53%` to `0.33%`)** AND **+0.36 pts MORE accurate (`-0.86` vs `-1.22`)** than `rpK-60`!
# 4. Look at where `rpL-60`'s `-0.25` pt official accuracy difference vs `rpK-60` sits:
#    - **IFEval `at_least_N` (58 questions = 10.7% of IFEval):** Because `rpL` capped counting responses at `1.35x + 25` (`--count-mode atleast` where 2/3 of pairs were non-length rules) and clipped `shorten` responses (`--resp-median 0.85`), `rpL-60`'s `at_least_N` responses dropped from `525.5` (`rpK-60`) to `440.5` tokens, costing `-5.82` pts on `at_least_N` (`13.58%` vs `19.40%` = `-0.62` pts on IFEval = `-0.16` headline pts), while on `no_counting` (`48.8%` of IFEval) `rpL-60` is **`+0.90` pts above `rpK-60` (`+0.99` above `base`)**!
#    - **SVAMP Q4_long (250 questions):** Because `rpL` had 16 math anti-loop pairs (`301 vs 4,014` tokens), `rpL-60` compressed Q4 thinking by another `-47.5` tokens (`544.1 -> 496.6`), costing `-0.40` pts on SVAMP (`-2.75` vs `-2.35` = `-0.10` headline pts) while gaining `+2.71%` SVAMP saving (`63.81%` vs `61.11%`).
#    - **BFCL Pooled:** `rpL-60` saves `+1.27%` more (`52.16%` vs `50.89%`), is `+0.38` pts better on `bfcl_multiple` (`-0.62` vs `-1.00`), `-0.50` pts on `bfcl_simple` (`-1.97` vs `-1.47`), and `-1.56` pts on `bfcl_irrelevance` (`+4.64` vs `+6.20`).
# 5. Now ask the key question for `rpN` (the accuracy-only repair):
#    **Wait: does `rpN` on top of `rpL-60` beat `rpN` on top of `rpK-60`?**
#    Let's check TWO things:
#    **(A) First, can `rpL-60` still generate valid long `"at least N"` responses and `REREAD` math solutions when we sample $K=8$ rollouts from `rpL-60` to build `rpN`'s own-data pairs, OR has `rpL-60` forgotten how to write long `"at least N"` responses?**
#    We can check this RIGHT NOW on `fulltest_per_question.csv` and `dev512_per_question.csv`!
#    Let's check how many questions on the Full Test and `Dev-512` have `>=1` right answer and `>=1` wrong answer (`1 <= right_of_8 <= 7`, i.e., pair-forming questions!) in `rpL-60` vs `rpK-60` vs `rpE-80`!

print("=== Pair-Forming Potential (Questions with 1 <= right_of_8 <= 7 and >=1 right answer) on Full Test ===")
for ds in ['svamp', 'mbpp', 'ifeval', 'bfcl_simple_python', 'bfcl_multiple', 'bfcl_irrelevance']:
    g = ft_q[ft_q['dataset'] == ds]
    print(f"\nDataset: {ds} (N={len(g)})")
    for m in ['rpE80', 'rpK60', 'rpL60']:
        r = g[f'{m}:right_of_8']
        mixed = ((r >= 1) & (r <= 7)).sum()
        all_right = (r == 8).sum()
        all_wrong = (r == 0).sum()
        print(f"  {m:6s}: mixed (1..7 right, builds right_cr/guard) = {mixed:4d} ({mixed/len(g)*100:4.1f}%) | 8/8 right = {all_right:4d} | 0/8 right = {all_wrong:4d}")

print("\n=== Specifically on IFEval `at_least_N` (N=58) on Full Test ===")
g_al = ft_q[(ft_q['dataset'] == 'ifeval') & (ft_q['ifeval_class'] == 'at_least_N')]
for m in ['rpE80', 'rpK60', 'rpL60']:
    r = g_al[f'{m}:right_of_8']
    mixed = ((r >= 1) & (r <= 7)).sum()
    all_right = (r == 8).sum()
    all_wrong = (r == 0).sum()
    print(f"  {m:6s}: mixed (1..7 right) = {mixed:2d}/58 ({mixed/58*100:4.1f}%) | 8/8 right = {all_right:2d} | 0/8 right = {all_wrong:2d} | mean rsp = {g_al[f'{m}:response'].mean():.1f}")
```