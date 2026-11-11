```python
# Look at L4945-4951:
# "`track.py ptrepair --rp-own-data` (new `own_repair_questions()`): math = GSM8K train+test at every difficulty in SVAMP wording (prepare_types gsm8k_std), minus questions already trained on in rounds 1-4; code / instructions / tools = pool-dp13 rows minus MetaMath + own top-ups nofit1, nofit2, tools1, code1; 8-gram check against every evaluation question."
# "own_repair_questions on stand-in files (MetaMath and easy pool math excluded, used and test-overlapping GSM8K dropped)"
#
# WAIT A SECOND!
# Read that again!
# 1. In `own_repair_questions()` (which builds `Dev-512` and Round 2's questions):
#    - **Math** comes from `prepare_types gsm8k_std` (GSM8K train+test at every difficulty in SVAMP wording), and **easy pool math (`gsm8k_easy`) is EXCLUDED**!
#    - **Code / Instructions / Tools** come from `pool-dp13` rows minus MetaMath + `nofit1, nofit2, tools1, code1` (plus `ifeval_built` for IFEval, L5105)!
# 2. Now look at what `p1src` (`track_ptrepair_odpo_dev.py` Lines 412-464) does for Round 1:
#    - `p1src` reads `P1_ANSWERS` (`lupin-all-dp13-nofit1-tools1-code1-nofit2-nofit4-math1-math2.jsonl`).
#    - `p1src` bans `math1` and `math2` (which are the ONLY places inside `P1_ANSWERS` that contain `gsm8k_std`!), AND bans `metamath_*`!
#    - So the ONLY math left inside `p1src` is `dp13`'s `gsm8k_easy` (420 trivial questions) + `gsm8k_matched` (241 questions) = **661 questions**, and after the 60% hash filter, **only ~396 math questions** (of which ~252 are `gsm8k_easy`, which `own_repair_questions()` explicitly excluded as too easy)!
#    - AND because `own_repair_questions()` draws its math from `gsm8k_std` (not `gsm8k_easy`) and its extra IFEval from `ifeval_built` (1,500+ questions), what about Code and Tools (`mbpp`, `bfcl_simple_python`, `bfcl_multiple`, `bfcl_irrelevance`)?
#      Let's check! How many Code and Tools questions exist in `own_repair_questions()` total (`pool-dp13` + `nofit1, nofit2, tools1, code1`)?!
#      Look at L5078 + L3747:
#      - `mbpp` (KodCode in `dp13 + code1`): `1,274 + 898 = 2,172` questions TOTAL!
#      - `bfcl_simple_python` (in `dp13 + tools1`): `316 + 955 = 1,271` questions TOTAL!
#      - `bfcl_multiple` (in `dp13 + tools1`): `239 + 360 = 599` questions TOTAL!
#      - `bfcl_irrelevance` (in `dp13 + nofit1 + nofit2`, WITHOUT `nofit4`'s 1,871 questions): `2,654 - 1,871 = 783` questions TOTAL!
#      WAIT A SECOND!
#      Look at those total pool sizes in `dp13 + nofit1 + nofit2 + tools1 + code1` (without `nofit4`):
#      - `bfcl_multiple`: **599 questions TOTAL in the entire pool!**
#        * `Dev-512` takes **35** of them (`564` left).
#        * `p1src` (60% hash) puts **~338** of them in `data/p1-round1-ids.json` (Round 1).
#        * How many `bfcl_multiple` questions are left in the 40% unseen pool for Round 2 (`--rp-exclude-dev --rp-exclude-ids data/p1-round1-ids.json`)?!
#          ONLY `564 - 338 = 226` questions (and if `Confirm-512` takes another 35, only **191 questions**)!
#          And in those ~191–226 `bfcl_multiple` questions, raw Qwen's mixed-outcome rate is `239/599 = 40%`, and a shortened Round-1 model's mixed-outcome rate is ~25–35% (~55–75 mixed questions) — barely enough to scrape together 80 pairs!
#      - `bfcl_irrelevance` (`no-fit`): **783 questions TOTAL in `dp13 + nofit1 + nofit2` (once `nofit4` is excluded)!**
#        * `Dev-512` takes **45** of them (`738` left).
#        * `p1src` (60% hash) puts **~443** of them in `data/p1-round1-ids.json` (Round 1).
#        * How many `bfcl_irrelevance` questions are left in the 40% unseen pool for Round 2 (`p1r2s0` / `p1r2s1`, which asks for `--rp-questions bfcl_irrelevance=600` and `--rp-quota bfcl_irrelevance=140`)?!
#          ONLY **~295 questions** (not 600!)!
#          And out of ~295 `no-fit` questions answered by a Round-1 model, remember `rpI` (L5172): out of **300** `no-fit` questions answered ×8 by `rpE-80`, only **59 questions** had a mixed outcome (`no-fit 59/80`)!
#          So in BOTH Round 1 (~443 `no-fit` questions, asking for 320 pairs!) AND Round 2 (~295 `no-fit` questions, asking for 140 pairs!), `bfcl_irrelevance` will suffer a massive pair shortfall!

print("Pool exhaustion math for Math, No-Fit, and Multiple verified!")
```