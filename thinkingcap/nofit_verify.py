"""Harder no-fit tool questions for dp6, with labels checked by a cheap model (user, 28 Oct: Haiku via OpenRouter).

dp3/dp4's no-fit questions were too easy (original Qwen right 94-98% vs BFCL irrelevance ~80%): a question offered a
function from an unrelated conversation is obvious to refuse. Here each ToolACE question (clean: single-turn, BFCL type
names, grounded reference) is offered the 1-3 functions from OTHER conversations whose names and descriptions share the
most words with it (TF-IDF over words), never its own called function nor one sharing a name word with it. Some of those
could answer the question, which would turn a "wrong" call into fake difficulty, so every candidate goes to a judge
model ("could any of these functions fulfil this request? YES / NO / UNSURE") and only NO is kept.

Runs on this Mac (the API key never leaves it); writes nofit_near_verified.jsonl next to this file, which the Colab
stage reads from the repo clone (prepare_types.py source toolace_near_verified).

    OPENROUTER_API_KEY=... python nofit_verify.py --toolace toolace.parquet --out nofit_near_verified.jsonl
"""

import argparse
import json
import math
import os
import random
import re
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROMPT = """You are checking labels in a function-calling dataset.

User request:
{question}

Available functions (JSON schemas):
{functions}

Could this request be fulfilled, fully or in part, by calling one of these functions with arguments taken from or
reasonably implied by the request? Answer with exactly one word: YES, NO, or UNSURE."""

WORD = re.compile(r"[a-z]{3,}")
STOP = {"the", "and", "for", "with", "can", "you", "please", "this", "that", "from", "are", "get", "what", "which", "have",
        "will", "would", "could", "about", "provide", "information", "details", "data", "need", "want", "like", "also",
        "some", "any", "all", "list", "find", "given", "specified", "specific", "returns", "return", "used", "using"}


def words(text):
    return [w for w in WORD.findall(text.lower()) if w not in STOP]


def name_words(name):
    parts = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])", name.replace(".", "_"))
    return {w.lower() for w in parts if len(w) >= 4}


PROMPT2 = """A user asked an assistant:
{question}

The assistant may only use these tools:
{functions}

Is there a tool here that the assistant should call for this request (even if it covers only part of it, or needs
reasonable default arguments)? Answer with exactly one word: YES, NO, or UNSURE."""


def judge(question, functions, key, model, template=PROMPT):
    import time
    body = json.dumps({"model": model, "max_tokens": 10, "temperature": 0,
                       "messages": [{"role": "user", "content": template.format(
                           question=question, functions=json.dumps(functions, indent=1)[:12000])}]}).encode()
    request = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=body, headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                text = (json.load(response)["choices"][0]["message"].get("content") or "").strip().upper()
            return "NO" if text.startswith("NO") else "YES" if text.startswith("YES") else "UNSURE"
        except Exception:
            if attempt == 3:
                return "ERROR"
            time.sleep(2 * 2 ** attempt)  # rate limits: back off and retry


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--rows", required=True, help="clean ToolACE rows (JSONL: prompt, category, function, reference)")
    parser.add_argument("--out", default=str(Path(__file__).with_name("nofit_near_verified.jsonl")))
    parser.add_argument("--max", type=int, default=6000, help="candidates sent to the judge")
    parser.add_argument("--model", default="anthropic/claude-haiku-5.5")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--singles", type=int, default=0, help="one-function candidates per question (dp7: 3)")
    parser.add_argument("--hard", action="store_true",
                        help="dp9: same-domain look-alikes: functions MAY share name words with the called one (only its own "
                             "functions are left out), the closest ones by meaning first; Haiku's NO + NO still decides")
    parser.add_argument("--rank", default="0,60", help="nearness window FROM,TO in the meaning ranking (Glaive: 20,120, "
                        "past the near-synonyms of the called function)")
    parser.add_argument("--skip", default="", help="JSONL of earlier kept candidates: (question, function) pairs not sent again")
    parser.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    key = os.environ["OPENROUTER_API_KEY"]
    rng = random.Random(args.seed)
    rows = [json.loads(l) for l in Path(args.rows).read_text().splitlines() if l.strip()]
    asked = [r for r in rows if r["category"] in ("simple", "multiple")]
    # every distinct function, indexed by the words of its name and description
    functions, owner = {}, {}
    for k, r in enumerate(asked):
        for f in r["function"]:
            functions.setdefault(f["name"], f)
            owner.setdefault(f["name"], set()).add(k)
    # nearness by meaning: sentence embeddings of each function (name + description) and of each question
    from sentence_transformers import SentenceTransformer
    import numpy as np
    encoder = SentenceTransformer(args.encoder)
    names = list(functions)
    fvec = encoder.encode([n.replace("_", " ") + ": " + str(functions[n].get("description", "")) for n in names],
                          batch_size=128, normalize_embeddings=True, show_progress_bar=False)
    skip = set()
    if args.skip:
        for l in Path(args.skip).read_text().splitlines():
            if l.strip():
                c = json.loads(l)
                skip |= {(c["question"], f["name"]) for f in c["functions"]}
    candidates, asked_once = [], set()
    order = list(range(len(asked)))
    rng.shuffle(order)
    for k in order:
        r = asked[k]
        question = r["prompt"][-1]["content"]
        if question in asked_once:  # Glaive repeats questions (the same BMI request many times)
            continue
        asked_once.add(question)
        called = {next(iter(c)) for c in (r.get("reference") or [])} | {f["name"] for f in r["function"]}
        avoid = set().union(*(name_words(c) for c in called)) if called else set()
        qvec = encoder.encode([question], normalize_embeddings=True, show_progress_bar=False)[0]
        sims = fvec @ qvec
        lo, hi = map(int, args.rank.split(","))
        top = [names[i] for i in np.argsort(-sims)[lo:hi]]
        if lo:
            rng.shuffle(top)
        same = {c.lower().replace(".", "_") for c in called}
        pick = [n for n in top if n not in called and k not in owner[n] and n.lower().replace(".", "_") not in same
                and (args.hard or not (name_words(n) & avoid)) and (question, n) not in skip]
        if not pick:
            continue
        if args.singles:  # BFCL's no-fit questions offer ONE function: up to --singles one-function candidates per question
            for n in pick[:args.singles]:
                candidates.append({"question": question, "functions": [functions[n]], "source_row": k})
        else:
            size = rng.choices((1, 2, 3), weights=(0.6, 0.25, 0.15))[0]
            candidates.append({"question": question, "functions": [functions[n] for n in pick[:size]], "source_row": k})
        if len(candidates) >= args.max:
            break
    print(f"{len(candidates)} near candidates from {len(asked)} ToolACE questions; judging with {args.model}", flush=True)
    with ThreadPoolExecutor(args.workers) as pool:  # two differently worded checks; a question stays no-fit only on NO + NO
        first = list(pool.map(lambda c: judge(c["question"], c["functions"], key, args.model), candidates))
        second = list(pool.map(lambda c: judge(c["question"], c["functions"], key, args.model, PROMPT2), candidates))
    verdicts = [a if a == b else f"{a}/{b}" for a, b in zip(first, second)]
    tally = Counter(verdicts)
    kept = [c | {"verdict": v, "hard": args.hard} for c, v in zip(candidates, verdicts) if v == "NO"]
    Path(args.out).write_text("".join(json.dumps(c) + "\n" for c in kept))
    print(f"verdicts {dict(tally)}; kept {len(kept)} (NO only) -> {args.out}", flush=True)
    Path(args.out).with_suffix(".report.json").write_text(json.dumps({"candidates": len(candidates), "verdicts": tally,
                                                                       "kept": len(kept), "model": args.model, "hard": args.hard}, indent=1))


if __name__ == "__main__":
    main()
