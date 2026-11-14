"""Haiku-written HARD no-fit tool questions (14 Nov, user: "yes" to generating the missing no-fit data with Haiku). od5b
can only run 9 test-matched rounds: the no-fit questions we have left are too easy (the original solves 94% vs the test's
83%). Each call asks Claude Haiku (via OpenRouter) for 10 near-miss items for one domain: 1-3 functions (JSON schema) and a
realistic user question that looks related to them (same domain, shared words) but that NONE of them can fulfil (another
action, another entity or data source, a capability no parameter covers). Each item is then checked twice by Haiku with two
different wordings ("is there a tool the assistant should call?") and kept only on NO + NO, as for our earlier verified
no-fit sets. prepare_types.py (source haiku_nofit) converts the functions to BFCL's format; the original's answers x8 and
the test-matched slice builder decide which ones are used.
Usage: OPENROUTER_API_KEY=... python gen_nofit_questions.py --calls 200 --out nofit_haiku.jsonl
"""
import argparse
import json
import os
import random
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from gen_tool_questions import DOMAINS

PROMPT = """Create 10 test items for a function-calling benchmark, all of the kind "no function fits". Domain: {domain}.
Each item: 1 to 3 Python-style functions (name in snake_case, a one-sentence description, parameters as a JSON schema object
with "type": "object", "properties" (each with "type" among string, integer, number, boolean, array, object and a
"description") and "required"), and a realistic user question that looks closely related to these functions (same domain,
overlapping words) but that NONE of them can fulfil: it needs a different action, a different kind of entity or data, or a
capability that no parameter covers. Make them hard: a careless assistant should be tempted to call one of the functions.
Vary the questions (short and long, casual and formal). English only.
Return ONLY a JSON array of 10 objects: {{"functions": [{{...}}], "question": "..."}}"""
CHECK1 = """A user asked an assistant:
{question}

The assistant may only use these tools:
{functions}

Is there a tool here that the assistant should call for this request (even if it covers only part of it, or needs
reasonable default arguments)? Answer with exactly one word: YES, NO, or UNSURE."""
CHECK2 = """Tools available:
{functions}

Request: {question}

Could calling one of these tools (with sensible arguments) actually fulfil this request, fully or in a meaningful part?
Answer with exactly one word: YES, NO, or UNSURE."""


def call(content, key, model, max_tokens, temperature, seed=0):
    body = json.dumps({"model": model, "max_tokens": max_tokens, "temperature": temperature, "seed": seed,
                       "messages": [{"role": "user", "content": content}]}).encode()
    request = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=body, headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                return (json.load(response)["choices"][0]["message"].get("content") or "").strip()
        except Exception:
            if attempt == 3:
                return ""
            time.sleep(2 * 2 ** attempt)


def generate(domain, key, model, seed):
    text = call(PROMPT.format(domain=domain), key, model, 9000, 1.0, seed)
    try:
        out = json.loads(text[text.find("["): text.rfind("]") + 1])
    except ValueError:
        return []
    keep = []
    for it in out if isinstance(out, list) else []:
        fs, q = it.get("functions") if isinstance(it, dict) else None, it.get("question") if isinstance(it, dict) else None
        if isinstance(fs, list) and 1 <= len(fs) <= 3 and isinstance(q, str) and q.strip() and all(
                isinstance(f, dict) and f.get("name") and isinstance(f.get("parameters"), dict) for f in fs):
            keep.append({"domain": domain, "functions": fs, "question": q.strip()})
    return keep


def verdict(text):
    t = text.upper()
    return "NO" if t.startswith("NO") else "YES" if t.startswith("YES") else "UNSURE"


def check(item, key, model):
    fs = json.dumps(item["functions"], indent=1)[:12000]
    a = verdict(call(CHECK1.format(question=item["question"], functions=fs), key, model, 10, 0.0))
    b = verdict(call(CHECK2.format(question=item["question"], functions=fs), key, model, 10, 0.0)) if a == "NO" else "skip"
    return a, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", type=int, default=200)
    ap.add_argument("--model", default="anthropic/claude-haiku-5.5")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(Path(__file__).with_name("nofit_haiku.jsonl")))
    a = ap.parse_args()
    key = os.environ["OPENROUTER_API_KEY"]
    rng = random.Random(a.seed)
    seeds = [(rng.choice(DOMAINS).strip(), a.seed * 100000 + k) for k in range(a.calls)]
    with ThreadPoolExecutor(a.workers) as ex:
        items = [it for batch in ex.map(lambda s: generate(s[0], key, a.model, s[1]), seeds) for it in batch]
    seen, uniq = set(), []
    for it in items:
        k = it["question"].lower()
        if k not in seen:
            seen.add(k)
            uniq.append(it)
    with ThreadPoolExecutor(a.workers) as ex:
        verdicts = list(ex.map(lambda it: check(it, key, a.model), uniq))
    kept = [it for it, (x, y) in zip(uniq, verdicts) if x == "NO" and y == "NO"]
    Path(a.out).write_text("".join(json.dumps(r) + "\n" for r in kept))
    from collections import Counter
    print(f"{len(items)} generated, {len(uniq)} unique, {len(kept)} kept on NO + NO; first-check verdicts "
          f"{dict(Counter(x for x, _ in verdicts))}, second {dict(Counter(y for _, y in verdicts))}")


if __name__ == "__main__":
    main()
