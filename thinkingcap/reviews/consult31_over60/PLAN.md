# Consult31: > 60% saved at better than -1 (user, 18 Nov ~21:30 CET)

Trigger (user, 19 Nov): once ALL the data is in and GPU 2 has finished: od5f to round 23 + its gated full tests, effsame (equal caps), and the od5g branch (GPU 1) with its gated full tests.
Then: (1) log them in EXPERIMENT_LOG.md, (2) add the full tests to the Frontier canvas DATA (reviews/consult31_over60/canvas/Main.dc.html is
a copy; the live one is the artifact), (3) add effsame's two summaries (GPU 2 log: '== base at od5c-best's caps' etc.) to reviews/consult31_over60/effprobe/ and pass --effprobe reviews/consult31_over60/effprobe,
(4) build_pack.py, (5) launch.sh (Argon, Google key from the Keychain, $3 cap, 120 turns), (6) check after ~2 min that it reads the pack
and writes notes.md, (7) report REPORT.md to the user.

Pack = od5f data (all rounds, per type, incl. 19-21) + whole lab notebook + thinking-off data + the current article (context) + consult28
and consult30 reports + question-level data + code. Competitor and BottleCap-method lines stripped. Prompt: astra_prompt_consult_over60.txt.
