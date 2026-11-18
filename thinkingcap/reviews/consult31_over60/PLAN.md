# Consult31: > 60% saved at better than -1 (user, 18 Nov ~21:30 CET)

Trigger: once od5f rounds 19-21 have their monitors AND the fullpick od5f 19-21 full tests (GPU 2, plan_line_a) are logged.
Then: (1) log them in EXPERIMENT_LOG.md, (2) add the full tests to the Frontier canvas DATA (reviews/consult31_over60/canvas/Main.dc.html is
a copy; the live one is the artifact), (3) if GPU 1's effprobe has finished, download its summaries (Drive) to pass with --effprobe,
(4) build_pack.py, (5) launch.sh (Argon, Google key from the Keychain, $3 cap, 120 turns), (6) check after ~2 min that it reads the pack
and writes notes.md, (7) report REPORT.md to the user.

Pack = od5f data (all rounds, per type, incl. 19-21) + whole lab notebook + thinking-off data + the current article (context) + consult28
and consult30 reports + question-level data + code. Competitor and BottleCap-method lines stripped. Prompt: astra_prompt_consult_over60.txt.
