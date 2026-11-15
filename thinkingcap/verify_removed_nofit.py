"""(15 Nov, user: "Yes, run the Haiku checks") Label check of the free function-removed no-fit questions (prepare_types.py
haiku_multiple_removed): the same two Haiku checks as every earlier near-miss no-fit source (gen_nofit_questions.py CHECK1 +
CHECK2), kept only on NO + NO. Writes tools_multiple_removed_verified.jsonl (question, functions, k = row index in
tools_multiple_haiku.jsonl). Runs on this Mac (key from the environment, never written).

    OPENROUTER_API_KEY=... python verify_removed_nofit.py [--limit 20]
"""
import argparse, json, os, sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import prepare_types as pt
from gen_nofit_questions import call, verdict, CHECK1, CHECK2


def check(item, key, model):
    # Haiku 5.5 reasons before answering: with gen_nofit_questions.py's 10-token limit its reply is cut off empty (UNSURE),
    # so this check allows 800 tokens; the rule is unchanged (two wordings, kept only on NO + NO)
    fs = json.dumps(item["functions"], indent=1)[:12000]
    x = verdict(call(CHECK1.format(question=item["question"], functions=fs), key, model, 800, 0.0))
    y = verdict(call(CHECK2.format(question=item["question"], functions=fs), key, model, 800, 0.0)) if x == "NO" else "skip"
    return x, y

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--model", default="anthropic/claude-haiku-5.5")
ap.add_argument("--out", default=str(Path(__file__).with_name("tools_multiple_removed_verified.jsonl")))
a = ap.parse_args()
key = os.environ["OPENROUTER_API_KEY"]
items = []
for k, r in enumerate(json.loads(l) for l in open(Path(__file__).with_name("tools_multiple_haiku.jsonl")) if l.strip()):
    try:
        fs = [pt.bfcl_function(f) for f in r["functions"]]
        right = r["call"]["name"]
    except Exception:
        continue
    others = [f for f in fs if f["name"] != right]
    if not others or len(others) == len(fs) or not pt.bfcl_types(others) or any(
            pt.name_words(f["name"]) & pt.name_words(right) for f in others):
        continue
    raw = [f for f in r["functions"] if f.get("name") != right]
    items.append({"k": k, "question": r["question"], "functions": raw})
if a.limit:
    items = items[:a.limit]
with ThreadPoolExecutor(10) as ex:
    verdicts = list(ex.map(lambda it: check(it, key, a.model), items))
kept = [it for it, (x, y) in zip(items, verdicts) if x == "NO" and y == "NO"]
if not a.limit:
    Path(a.out).write_text("".join(json.dumps(r) + "\n" for r in kept))
print(f"{len(items)} checked, {len(kept)} kept on NO + NO; first {dict(Counter(x for x, _ in verdicts))}, "
      f"second {dict(Counter(y for _, y in verdicts))}")
