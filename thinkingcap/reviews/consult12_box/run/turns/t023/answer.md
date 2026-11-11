```python
if_prompt_pass['id'] = if_prompt_pass['id'].astype(str)
ft_ans['id'] = ft_ans['id'].astype(str)
ft_ans = ft_ans.merge(if_prompt_pass, on=['model', 'id', 'sample'], how='left')
ft_ans['right_off'] = np.where(ft_ans['dataset'] == 'ifeval', ft_ans['if_right_off'].astype(float), ft_ans['right'].astype(float))
ft_ans['total_tokens'] = ft_ans['thinking'] + ft_ans['response_tokens']

for m in ['base', 'rpE80', 'rpK20', 'rpK40', 'rpK60', 'rpL60']:
    sub = ft_ans[ft_ans['model'] == m]
    accs = sub.groupby('dataset')['right_off'].mean() * 100.0
    print(m, {k: round(v, 2) for k, v in accs.to_dict().items()})
```