```python
# HOLD EVERYTHING! Look at what we just discovered by comparing `EXPERIMENT_LOG.md` Lines 4948-4950 and 5015 with `repair_pairs.py` Lines 122-178!
# Let's read this very carefully:
# 1. In `rpE`, `rpG`, `rpH`, `rpI`:
#    What was `--kinds` set to?
#    `--rp-kinds guard`!
#    Wait: what did `--kinds guard` in `repair_pairs.py` actually build?!
#    Look at `repair_pairs.py` Lines 155-173:
#      `chosen = rights[0]`  # the SHORTEST right answer!
#      `if decode:`
#          `checked = [r for r in rights if thinking_len(r) <= 1.5 * thinking_len(rights[0]) and REREAD.search(decode(r))]`
#          `chosen = checked[0] if checked else rights[0]`
#      `cand[part]["guard"].append((qid, chosen, wrongs[0]))`  # `wrongs[0]` = the SHORTEST finished wrong answer!
#    AND look at the docstring of `repair_pairs.py` (Lines 10-12):
#      `--kinds guard (9 Nov): guard pairs only, the whole quota (the user's own 21 Oct rule, bon_pairs' guard; the similar-length "right" kind matches Lightning McQwen3's correctness DPO and is left out).`
#    WAIT A SECOND!
#    Look at why `rpE`'s pairs were described in `all_runs.json` as "equal-length 'be right' pairs (shortest right > shortest wrong, own questions only)":
#    Because in `rpE`, they ran `--kinds guard` (**shortest right > shortest finished wrong**, with `--prefer-reread` picking the shortest right answer within `1.5x` of `rights[0]` that matches `REREAD`), and the medians of `shortest right` vs `shortest wrong` *happened* to be nearly equal (`271/278, 440/440, 185/183`, L5040)!
#    AND notice why the docstring of `repair_pairs.py` says:
#    `the similar-length "right" kind matches Lightning McQwen3's correctness DPO and is left out`!
#    Wait: when the coder implemented `--kinds mix` in `repair_pairs.py` (Lines 122-145 and 200-221) to follow our first report:
#    Let's trace what `repair_pairs.py --kinds mix` ACTUALLY does right now:
#    1) For `55%` (or `50%/35%/72.5%` in Round 1) of the quota (`"right"`), `--kinds mix` does NOT use `rpE`'s pair (`shortest right with reread within 1.5x > shortest finished wrong`)! Instead, it uses `right_long` and `right_short` (`max(size(r), size(w)) <= 1.2 * min(size(r), size(w))`, half strictly longer, half strictly shorter — which the docstring at Line 11 explicitly noted was Lightning McQwen3's correctness pair that was previously excluded!).
#    2) Even worse, look at Line 136-138 of `repair_pairs.py` inside `if args.kinds == "mix":`:
#       `close = [(r, w) for r in rights for w in wrongs if max(size(r), size(w)) <= args.ratio * max(1, min(size(r), size(w)))]`
#       `if decode:  # prefer a chosen answer that re-checks the question`
#       `    close.sort(key=lambda rw: not REREAD.search(decode(rw[0])))`
#       `for r, w in close[:4]:`
#       `    cand[part]["right_long" if size(r) > size(w) else "right_short"].append((qid, r, w))`
#       `...`
#       `def take(part, kind, pool, n):`
#       `    rng.shuffle(pool)`
#       WAIT A MINUTE! Look at Lines 136-138 + Line 183 (`rng.shuffle(pool)`):
#       In Line 136, `close` is sorted so `REREAD` pairs come first in `close[:4]`, and up to 4 pairs per question are appended to `cand[part]["right_long"]` and `cand[part]["right_short"]`.
#       Even if a question has 0 `REREAD` answers (or 1 `REREAD` answer and 3 non-`REREAD` answers), all 4 get appended into `cand[part]["right_long"]` / `right_short`, and then `take()` calls `rng.shuffle(pool)` across the whole pool, throwing away the `REREAD` priority when sampling across questions, AND doesn't pick the *shortest* right answer (`rights[0]`), it pairs *any* right answer (including long ones!) with *any* wrong answer of similar length!
#    3) AND look at what happens in `repair_pairs.py` Lines 263-266 (`stratified order: interleave the parts by their shares, kinds alternating inside each part`):
#       `queues = {part: sorted(ps, key=lambda p: (p["kind"], rng.random())) for part, ps in by_part.items()}`
#       `for part in queues:`
#       `    g = [p for p in queues[part] if p["kind"] == "guard"]`
#       `    r = [p for p in queues[part] if p["kind"] != "guard"]`
#       `    queues[part] = [x for pair in zip(g, r) for x in pair] + g[len(r):] + r[len(g):]`
#       WAIT A SECOND! Look at Lines 263-266 of `repair_pairs.py`!
#       Let's trace what Lines 263-266 do when `args.kinds == "mix"`:
#       In `by_part[part]`, there are THREE kinds of pairs: `"shorten"`, `"guard"`, and `"right"`!
#       First, Line 263 sorts `ps` alphabetically by `p["kind"]`:
#       Alphabetical order of kinds:
#       1. `"guard"` (15% of pairs)
#       2. `"right"` (55% of pairs)
#       3. `"shorten"` (30% of pairs)
#       Next, Line 264 splits `queues[part]` into:
#       - `g`: `p["kind"] == "guard"` (15% of pairs)
#       - `r`: `p["kind"] != "guard"` (which is ALL `"right"` pairs FIRST, followed by ALL `"shorten"` pairs LAST, because Line 263 sorted by `p["kind"]`!)
#       Next, Line 266 zips `g` (15% of pairs) with the first 15% of `r` (which are all `"right"` pairs!), and then appends `r[len(g):]`!
#       WHAT IS IN `r[len(g):]`?!
#       First the remaining 40% `"right"` pairs, and ONLY IN THE LAST 30% OF THE QUEUE (at the very end of the epoch!) do ANY `"shorten"` pairs appear!
#       HOLD ON! Let's verify this in Python right now!

ps_demo = [{"kind": "shorten", "i": i} for i in range(96)] + [{"kind": "guard", "i": i} for i in range(48)] + [{"kind": "right", "i": i} for i in range(176)]
rng = random.Random(0)
q_sorted = sorted(ps_demo, key=lambda p: (p["kind"], rng.random()))
g = [p for p in q_sorted if p["kind"] == "guard"]
r = [p for p in q_sorted if p["kind"] != "guard"]
q_final = [x for pair in zip(g, r) for x in pair] + g[len(r):] + r[len(g):]

# In rpJ, total pairs = 1280 (320 per part), steps = 60, batch = 16 -> 60 * 16 = 960 pairs seen in 60 steps (75% of 1280)!
# Wait: at step 20 (320 pairs = 25% of 1280), step 40 (640 pairs = 50% of 1280), step 60 (960 pairs = 75% of 1280):
# How many "shorten" pairs has the model seen at step 20, step 40, and step 60 in `q_final`?!
for step, frac in [(20, 0.25), (40, 0.50), (60, 0.75), (80, 1.00)]:
    n_seen = int(len(q_final) * frac)
    c_seen = Counter(p["kind"] for p in q_final[:n_seen])
    print(f"At step {step:2d} ({frac*100:3.0f}% through pairs): {dict(c_seen)}")
```