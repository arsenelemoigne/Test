"""(16 Nov, user: "use Haiku / Argon to generate more hard tool questions, but so that we don't throw so many away, and not
minimal edits of used questions") NEW one-function and no-function-fits questions aimed at what the ORIGINAL Qwen3-0.6B
actually gets wrong, written by Gemini 4 Argon (user's Google key) or Claude Haiku 5.5 (OpenRouter), same prompt, so the
two generators can be compared on hard questions per dollar.

The trap patterns come from the original's stored answers (pattern analysis 16 Nov, reviews/tool_patterns/PATTERNS.md):
no-fit: it calls a sibling function (same domain, other entity / capability); one function: it adds ungrounded optional
arguments, splits or merges list values, appends a second call for an unsupported follow-up, or returns no call on chatty
requests. Only abstract pattern descriptions are given to the generator, never our real questions.

No-fit items are kept only on Haiku's NO + NO label checks (the same two wordings as every earlier no-fit source). One-function
items must pass the BFCL clean-up of prepare_types.py (BFCL types, every argument value literally in the question).
Keys come from the environment (never written): OPENROUTER_API_KEY, GOOGLE_API_KEY.

    python gen_targeted_tools.py --gen argon --kind nofit --calls 6 --out tools_tgt_nofit_argon.jsonl
"""
import argparse
import json
import os
import random
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DOMAINS = """aviation maintenance logs, museum collections, beekeeping, maritime port schedules, wine cellars, orchestra
rehearsals, dental clinics, kindergarten enrolment, bicycle repair shops, ski resorts, library interlibrary loans, solar
farm monitoring, veterinary pharmacies, film festival programming, hiking trail conditions, laundromat machines, chess
tournaments, recycling centres, tattoo studios, observatory bookings, tide tables, escape rooms, coffee roasting, vintage
car auctions, wedding planning, translation agencies, municipal parking permits, bowling alleys, aquarium feeding,
textbook rentals, drone flight permits, plant nurseries, podcast hosting, piano tuning, glacier research stations,
co-working spaces, blood donation drives, fishing licences, theatre costume rentals, quarry operations""".replace("\n", " ").split(", ")

NOFIT = """Create 10 test items for a function-calling benchmark, all of the kind "no function fits". Domain: {domain}.
A small model must be STRONGLY tempted to make a call. Use these traps, which a 0.6B model falls for most often:
(a) sibling entity: the functions handle a closely related entity of the same domain, but not the one asked about
(e.g. the request is about one kind of record, the function returns a neighbouring kind with similar fields);
(b) sibling capability: a calculator or lookup that computes or returns something close to, but different from, what
is asked (the numbers in the request would fit its parameters, the result would be wrong);
(c) keyword bait: the request contains a word that also appears in a function name or description, with another meaning;
(d) two-part request where neither part can be done by the functions.
Mix the traps. Each item: 1 to 3 Python-style functions (snake_case names, a one-sentence description, parameters as a
JSON schema object with "type": "object", "properties" (each with "type" among string, integer, number, boolean, array,
object, and a "description") and "required"), and a realistic user question that NONE of them can fulfil, even in part.
Vary length and tone. English only.
Return ONLY a JSON array of 10 objects: {{"functions": [{{...}}], "question": "...", "trap": "a|b|c|d"}}"""

SIMPLE = """Create 10 test items for a function-calling benchmark of the kind "one function, call it correctly". Domain: {domain}.
The call must be easy to get WRONG for a 0.6B model, using these traps, which it falls for most often:
(a) optional-parameter bait: the question mentions extra facts that look like arguments but are not parameters, or that
correspond to an optional parameter which must be LEFT OUT because the question does not ask for it;
(b) list arguments: a parameter takes an array, and the question gives several values in one sentence;
(c) follow-up bait: the question asks for the call AND something the function cannot do; only the one call is correct;
(d) chatty request: a long, casual, story-like question where the actual request is short;
(e) distractor numbers: the question contains numbers that are NOT arguments.
Mix the traps. Each item: one Python-style function (snake_case name, a one-sentence description, parameters as a JSON
schema object with "type": "object", "properties" (each with "type" among string, integer, number, boolean, array,
object, and a "description" that states units and formats exactly) and "required"), a realistic user question, and the ONE
correct call. Every argument value in the call must appear literally in the question (same number or words). English only.
Return ONLY a JSON array of 10 objects: {{"function": {{...}}, "question": "...", "call": {{"name": "...", "arguments": {{...}}}}, "trap": "a|b|c|d|e"}}"""

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

COST = {"usd": 0.0}


def openrouter(content, model, max_tokens, temperature, seed=0):
    body = json.dumps({"model": model, "max_tokens": max_tokens, "temperature": temperature, "seed": seed,
                       "usage": {"include": True}, "messages": [{"role": "user", "content": content}]}).encode()
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=body, headers={
        "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}", "Content-Type": "application/json"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                j = json.load(r)
            COST["usd"] += float((j.get("usage") or {}).get("cost") or 0)
            return (j["choices"][0]["message"].get("content") or "").strip()
        except Exception:
            if attempt == 3:
                return ""
            time.sleep(2 * 2 ** attempt)


def argon(content, max_tokens):
    body = {"contents": [{"role": "user", "parts": [{"text": content}]}],
            "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 1.0, "thinkingConfig": {"thinkingLevel": "low"}}}
    req = urllib.request.Request("https://generativelanguage.googleapis.com/v1beta/models/gemini-4-argon:generateContent",
                                 data=json.dumps(body).encode(),
                                 headers={"x-goog-api-key": os.environ["GOOGLE_API_KEY"], "Content-Type": "application/json"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                d = json.loads(r.read())
            m = d.get("usageMetadata", {})
            COST["usd"] += ((m.get("promptTokenCount", 0) - m.get("cachedContentTokenCount", 0)) * 2e-6
                            + (m.get("candidatesTokenCount", 0) + m.get("thoughtsTokenCount", 0)) * 10e-6)
            parts = (d.get("candidates") or [{}])[0].get("content", {}).get("parts", [])
            return "".join(p.get("text", "") for p in parts if not p.get("thought")).strip()
        except Exception:
            if attempt == 3:
                return ""
            time.sleep(10 * (attempt + 1))


def parse(text):
    try:
        out = json.loads(text[text.find("["): text.rfind("]") + 1])
        return [x for x in out if isinstance(x, dict)]
    except ValueError:
        return []


def ok_function(f):
    if not isinstance(f, dict) or not re.fullmatch(r"[A-Za-z_]\w*", str(f.get("name", ""))):
        return False
    p = f.get("parameters")
    return isinstance(p, dict) and isinstance(p.get("properties"), dict)


def verdict(t):
    t = (t or "").upper()
    return "NO" if t.startswith("NO") else "YES" if t.startswith("YES") else "UNSURE"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", choices=("argon", "haiku"), required=True)
    ap.add_argument("--kind", choices=("nofit", "simple"), required=True)
    ap.add_argument("--calls", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--haiku", default="anthropic/claude-haiku-5.5")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    doms = [rng.choice(DOMAINS) for _ in range(a.calls)]
    prompt = NOFIT if a.kind == "nofit" else SIMPLE

    def gen(i):
        text = (argon(prompt.format(domain=doms[i]), 16000) if a.gen == "argon"
                else openrouter(prompt.format(domain=doms[i]), a.haiku, 12000, 1.0, a.seed * 1000 + i))
        return [dict(x, domain=doms[i], gen=a.gen) for x in parse(text)]
    with ThreadPoolExecutor(6) as ex:
        items = [x for b in ex.map(gen, range(a.calls)) for x in b]
    gen_cost = COST["usd"]
    seen, uniq = set(), []
    for it in items:
        q = str(it.get("question", "")).strip()
        if not q or q.lower() in seen:
            continue
        if a.kind == "nofit":
            fs = it.get("functions")
            if not (isinstance(fs, list) and 1 <= len(fs) <= 3 and all(ok_function(f) for f in fs)):
                continue
        else:
            f, call = it.get("function"), it.get("call")
            if not (ok_function(f) and isinstance(call, dict) and call.get("name") == f.get("name")
                    and isinstance(call.get("arguments"), dict)
                    and all(k in f["parameters"]["properties"] for k in call["arguments"])):
                continue
        seen.add(q.lower())
        uniq.append(it)
    kept = uniq
    if a.kind == "nofit":  # Haiku label checks, NO + NO (800 tokens: Haiku 5.5 reasons before answering)
        def check(it):
            fs = json.dumps(it["functions"], indent=1)[:12000]
            x = verdict(openrouter(CHECK1.format(question=it["question"], functions=fs), a.haiku, 800, 0.0))
            y = verdict(openrouter(CHECK2.format(question=it["question"], functions=fs), a.haiku, 800, 0.0)) if x == "NO" else "skip"
            return x, y
        with ThreadPoolExecutor(10) as ex:
            v = list(ex.map(check, uniq))
        kept = [it for it, (x, y) in zip(uniq, v) if x == "NO" and y == "NO"]
    Path(a.out).write_text("".join(json.dumps(r) + "\n" for r in kept))
    print(f"{a.gen} {a.kind}: {len(items)} generated, {len(uniq)} valid and unique, {len(kept)} kept; generation "
          f"${gen_cost:.3f}, total ${COST['usd']:.3f}", flush=True)


if __name__ == "__main__":
    main()
