```python
# Let's check fulltest_per_question.csv and dev512_per_question.csv to see what models are in there and compute exact standard errors and noise of full tests!
ft_q = pd.read_csv(os.path.join(pack_dir, "tables/fulltest_per_question.csv"))
dev_q = pd.read_csv(os.path.join(pack_dir, "tables/dev512_per_question.csv"))
print("Fulltest models:", ft_q["model"].unique())
print("Fulltest columns:", ft_q.columns.tolist())
print("Dev512 models:", dev_q["model"].unique())
print("Dev512 columns:", dev_q.columns.tolist())
```