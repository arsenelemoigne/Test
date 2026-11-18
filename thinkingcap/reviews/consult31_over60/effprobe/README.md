# effprobe (18-19 Nov, GPU 1): what shortening did, measured causally

Questions: unseen training slices 19-20 (1,560 test-matched questions, never trained on by od5c-best). Models: the original
Qwen3-0.6B ("base") and od5c-best (online DPO od5c round 16, full test 54.8 / -0.78). Each answered every question 4 times.

- curves_summary.txt: per type, the accuracy when an answer is cut and the same model must answer at once:
  fNN = cut at NN% of its own thinking (sentence-aligned; samples 0-1), bK=Tt = a fixed thinking budget of T tokens (the
  model's own per-type length deciles; samples 0-3; answers already shorter keep their natural outcome), mt / mq = the
  ORIGINAL cut at the trained model's per-type / per-question thinking length. "post-settlement share" = thinking after the
  settle point (the smallest cut from which every larger cut stays right), over right answers.
- surprisal.txt: bits per thinking token of each model's traces read by each model (all / first 100 tokens / by decile).
- allocate.txt: best per-task budgets for the official score from the trained model's curves.
- effprobe_math.png, effprobe_cut_by_task.png: the two charts shown to the user.

Key readings (see EXPERIMENT_LOG.md, 19 Nov 01:00-01:20):
1. Math at equal tokens: trained 87.9% at ~290 tokens; the original stopped at ~300 tokens 47-57%.
2. Share of the cut that was working-out (before the answer settles) vs re-checking: math 68/32, code 7/93, IFEval 34/66,
   one function 12/88, several 28/72, no-fit 10/90.
3. Surprisal: the original reads the trained thinking at 0.81 bits/token vs its own 0.49 (math 0.39 -> 0.68): repetition gone.
4. No thinking at all: trained 50.2% vs original 47.2% (all types): the model itself improved.
5. Per-task budgets (math full, code / IFEval / no-fit none, tools small): 82% saved at the same average accuracy ON THESE SLICES,
   but NOT reliable for the full test: on the full test the original with thinking off loses 9.1 pts on code and 5.0 on IFEval
   (here only -2.3 / +2.7), and the trained model reads ~3 pts optimistic on these slices (+2.26 here vs -0.78 full). With the
   full-test thinking-off costs the same allocation gives ~82% / -2.2 vs the original. Check budgets on Dev / the full test.
6. Caveat: a forced cut interrupts the original's planned long trace; the equal-cap run (effsame, GPU 2) adds each model cut
   at the OTHER model's caps (curves-<model>-at-<other>/summary.txt, add when it lands).
