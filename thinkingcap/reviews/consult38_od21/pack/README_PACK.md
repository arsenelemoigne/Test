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

## consult32 additions
- YOUR_CONSULT31_REPORT.md: your last report (sections 4-8 come from your draft: the final answer was cut at section 4).
- od5h_plan_lines.txt: od5g's exact line, then the proposed od5h line and its fullpick line.
- code_changes.diff: the two code changes (--od5-held in repair_pairs.py, --od-smooth in track.py); code/ has the full files.
- effprobe/effsame.txt: forced early-stop curves at the OTHER model's per-type caps (both directions).

## consult36 additions (20 Nov 2026, afternoon)
- YOUR_CONSULT35_REPORT.md: your consult35 report and its plan (section 3), which became od5j.
- OD5J_LINES_AS_RUN.txt: the exact GPU 1 plan running now (od5j to 18, fullpick, od5jx 19-22, fullpick).
- OD5J_ROUNDS.md: od5j monitors rounds 2-16, each next to od5f / od5c / od5h at the same round.
- OD5H_FULLTESTS.md: od5h Dev-512 vs full tests (r19 61.7/-1.68, r21 62.7/-1.69, r22 69.6/-3.64; Dev over-read accuracy 3.8-5.8 pts).
- FINAL_MODEL_od5g_r20.md: od5g-r20 57.4/+0.14 and od5f-r18 59.5/-0.80 per type.
- EXPERIMENT_LOG.md: updated to 20 Nov 13:45. ARTICLE_TEXT.md: the current article (best 2) text.

## consult37 additions (20 Nov 2026, evening)
- YOUR_CONSULT36_REPORT.md: your last report (cut off in section 3); CONSULT36_RUNS_AS_RUN.txt: its two runs as run; CONSULT36_RESULTS.md: what they gave.
- EXPERIMENT_LOG.md: updated to 20 Nov ~18:40.
- Raw training files (question slices data/od3-rK, data/od3g-rK, the per-round pairs runs/bon/pairs-*/pairs.jsonl, the rollouts) live on the lab's Google Drive, not in this pack; every slice's statistics (question counts per task, the original's solve rate, output and prompt lengths) and every round's pair counts per task are in EXPERIMENT_LOG.md and in the run logs (OD5H_COLAB_LOG.txt) and code/.

## training_data/consult37_data (exported from the lab's Drive on 20 Nov 19:40, 80 MB): READ THIS
- PAIRS_SUMMARY_all_runs.jsonl (2,402 rows): for EVERY online DPO run (od5c, od5r, od5f, od5g, od5h, od5j, od5k, od5l) and EVERY
  round: pairs per task ("part") and per kind (shorten, right, recovery, hold, loop, format, guard...), median chosen/rejected
  length ratio, median chosen length. This is the exact training signal each run got.
- pairs_od5f_rK.jsonl (rounds 1-23): od5f's pairs one by one (id, task, kind, chosen/rejected token counts, the start of each
  answer decoded, 600 chars).
- answers_od5f_rK.jsonl: od5f's own training answers per round, ONE ROW PER ANSWER (8 per question; "right" is 0/1, group by
  "id"); first row lists the source record's fields.
- <run>-monitor.jsonl and <run>-latch.json for every run: each round's monitor per task and the brake state.
- od3-rK-questions.jsonl / od3g-rK-questions.jsonl: the question slices themselves (round K trains on slice K); od3-base-stats.json:
  the original's per-question stats used for floors/caps.
- dev512-*_dev_score.json and full-od5*_*.json: Dev and full-test score files.
Use your code runs to aggregate them (pandas/json); do not read them whole into the conversation.

- YOUR_NOTES_FROM_A_CRASHED_FIRST_ATTEMPT.md: your own notes from 16 turns of this same consult (the session died on a network error); reuse them, check them.

## consult38 additions (21 Nov 2026): od5m + od5n, the two from-scratch variations of YOUR consult37 plan
- YOUR_CONSULT37_REPORT.md: your consult37 report (the plan for variations A and B).
- OD5M_OD5N_LINES_AS_RUN.txt: the exact lines as run. NOTE: only the caps/floors/quotas of your plan were applied; your extra flags
  (--rp-beta 0.15, --rp-epochs 3, --rp-lora-*, stop rules) were NOT in od5f's line and were dropped. od5m = A, od5n = B.
  Both trained rounds 1-18 on od3 slices, then the same line continued to round 22 on od3g slices.
- OD5M_OD5N_RESULTS.md: every round 2-22 of both runs: last-step DPO loss, pairs ordered right, MONITOR (total and per task) next
  to od5f's same-round monitor; Dev-512 scores so far; no full test yet.
- DEV_FULLPICK_LATEST.md (if present): the Dev scores of the rounds each run's fullpick selected, as they arrived.
- training_data/consult38_data (if present, exported from Drive on 21 Nov): same format as consult37_data, now with od5m and od5n
  pairs and training answers one by one (pairs_od5m_rK, pairs_od5n_rK, answers_od5m_rK, answers_od5n_rK), their monitors and
  latches, all Dev files (dev512-od5m-*, dev512-od5n-*), and PAIRS_SUMMARY_all_runs.jsonl including od5m/od5n.
- training_data/consult37_data: unchanged (od5f in full).
- EXPERIMENT_LOG.md: updated to 21 Nov afternoon.
