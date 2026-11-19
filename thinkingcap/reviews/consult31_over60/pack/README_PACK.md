# Pack for consult31: > 60% saved at better than -1 (built from the lab notebook as of its last line)

- NEW_RESULTS_od5f.md: od5f (your consult28 plan) line by line: monitors per round (slice = fresh test-matched questions; the monitor reads
  ~2.5 pts optimistic on accuracy), Dev-512 scores, full tests per type, and the extension to round 21.
- fulltests.json: 44 full tests of our models, per type, plus the original with thinking off.
- THINKING_OFF.md: thinking-off full tests (original; od5c-best +2.1, od5r-r16 +1.9, rpM-60 -0.3, rpL-80 -3.7 vs the original off).
- YOUR_CONSULT28_REPORT.md (od5f's design), YOUR_CONSULT30_REPORT.md (efficiency, KKT redistribution estimate).
- SMARTER_DATA.md: question-level movement (net drift, excess churn), Pass@8 and 8/8, trimming vs condensing, post-answer thinking,
  marginal token value, for od5c-r16 (54.8 / -0.78).
- ARTICLE_TEXT.md: the user's article as published (charts removed), for context.
- pertype_best_full.md, REPRODUCE_od5c.md, code/ (as run), EXPERIMENT_LOG.md (the whole notebook).
- effprobe/ (if present): forced early-stop curves and budget allocation on unseen slices 19-20, original and od5c-best.
