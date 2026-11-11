```python
# Let's analyze the sampling noise of a single full test (8 samples per question) vs the true infinite-sample expectation for a fixed model, AND the question-level bootstrap SE!
# Wait: since the evaluation question set (2,759 questions) IS the test set (ThinkingCap-Test is fixed), the only randomness in evaluating a given model checkpoint with a fixed evaluation seed is the 8 MC samples per question!
# We have fulltest_per_answer.csv! Let's load it and compute the exact within-question sampling variance (8 samples per question) for every model and benchmark, as well as the paired difference covariance between rpK20, rpK40, rpK60, rpE80, rpL60!

ft_ans = pd.read_csv(os.path.join(pack_dir, "tables/fulltest_per_answer.csv"))
print("ft_ans shape:", ft_ans.shape)

# Wait: for official IFEval, let's merge the official prompt-level `passed` from ft_if_full into ft_ans for dataset == 'ifeval'!
if_prompt_pass = ft_if_full.groupby(['model', 'id', 'sample'])['passed'].all().reset_index()
if_prompt_pass.rename(columns={'passed': 'if_right_off'}, inplace=True)

ft_ans = ft_ans.merge(if_prompt_pass, on=['model', 'id', 'sample'], how='left')
ft_ans['right_off'] = np.where(ft_ans['dataset'] == 'ifeval', ft_ans['if_right_off'].astype(float), ft_ans['right'].astype(float))
ft_ans['total_tokens'] = ft_ans['thinking'] + ft_ans['response_tokens']

# Let's verify that ft_ans['right_off'] reproduces the exact official numbers!
for m in ['base', 'rpE80', 'rpK20', 'rpK40', 'rpK60', 'rpL60']:
    sub = ft_ans[ft_ans['model'] == m]
    accs = sub.groupby('dataset')['right_off'].mean() * 100.0
    print(m, {k: round(v, 2) for k, v in accs.to_dict().items()})
```