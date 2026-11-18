# What is in this pack (18 Nov 2026)

Read first: PLAN.md (the analysis plan and its reasoning), then ARTICLE_TEXT.md (the article as published), then
MEASURES_WE_USE.md (every measure we use, with results).

- per_answer/<model>.csv: full test, one line per answer (2,759 questions x 8 = 22,072 lines): id, dataset (svamp = math,
  mbpp = code, ifeval = instructions, bfcl_simple_python = one function, bfcl_multiple = several functions,
  bfcl_irrelevance = no function fits), sample (0-7), correct (0/1), output_tokens, thinking_tokens, truncated.
  Models: base = the original Qwen3-0.6B; nothink = the original with thinking switched off; od5c16 = the best model (online
  DPO round 16, 54.8% saved / -0.78); 12 offline models (rpM60, rpL80, ... rc1; earlier recipes, for comparison).
- traces/<model>.jsonl: the same answers WITH their full text (thinking between <think> and </think>, then the response),
  for base, nothink and od5c16. ~110 MB in total: read them line by line, never all at once into one data frame.
- code/trim_condense.py: the trimming-vs-condensing method (PLAN.md section 2); results/trim_condense_report_1600pairs.json is
  its output on 1,600 answer pairs (200 math + 200 code + 100 per other task, samples 0 and 1).
- code/density_strict.py + results/density_strict.json: the questions-lost-beyond-chance test (PLAN.md section 1).
- code/article_metrics.py: how the article's other measures are computed.
- results/article_stats.json: the data behind the article's charts (per type: tokens, thinking, accuracy, loops, paragraphs...).
- NEW_RESULTS_online_17nov.md: the latest results summary; EXPERIMENT_LOG.md: the full lab notebook.
