"""Haiku-written ONE-function tool-call questions in BFCL simple_python's style (14 Nov, user: "for any missing data, you
can generate it with haiku"). Step 2 lacks "be right" pairs on simple tool calls: our ToolACE / Glaive sources have run out
of the harder questions. Each call asks Claude Haiku (via OpenRouter) for 10 items for one domain seed: a Python-style
function (JSON schema: name, description, parameters with types, required), a user question, and the one correct call.
Items vary in difficulty on purpose (several arguments, optional ones to include or leave out, lists, distractor numbers).
prepare_types.py (source haiku_tools) then applies the Glaive clean-up: BFCL type names, every argument value stated in
the question; the original's answers x8 and r2prep's test-matched bins decide which ones are used.
Usage: OPENROUTER_API_KEY=... python gen_tool_questions.py --calls 150 --out tools_simple_haiku.jsonl
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

DOMAINS = """geometry, algebra, statistics, probability, physics kinematics, electricity, thermodynamics, chemistry
stoichiometry, biology genetics, ecology, astronomy, geography distances, weather data, unit conversion, currency exchange,
loan and mortgage maths, investment returns, taxes, retail discounts, payroll, inventory, shipping costs, travel bookings,
hotel search, flight search, restaurant search, recipe nutrition, fitness and calories, medical dosage, sports statistics,
music playlists, movie lookup, book lookup, calendar and dates, time zones, text processing, string manipulation, list
sorting, matrix operations, number theory, cryptography basics, image resizing, file sizes, network bandwidth, database
queries, real estate, car fuel economy, energy usage, agriculture yields, population growth, elections data, stock prices,
company financials, sales forecasting, project scheduling, translation, dictionary lookup, map routing, public transport,
event tickets, pet care, gardening, construction materials, painting area, chemistry pH, optics, sound and acoustics,
fluid dynamics, genetics probability, epidemiology, sports scheduling, game scores""".replace("\n", " ").split(", ")
PROMPT = """Create 10 test items for a function-calling benchmark. Domain: {domain}.
Each item: one Python-style function (name in snake_case, a one-sentence description, parameters as a JSON schema object
with "type": "object", "properties" (each with "type" among string, integer, number, boolean, array, object, and a
"description"), and "required"), a realistic user question, and the ONE correct call.
Rules: every argument value in the call must appear literally in the question (same number or words). Vary difficulty:
some items with 1-2 arguments, some with 4-6 arguments, some with an optional parameter the question sets and others where
it must be left out, some with a list argument, some questions containing an extra number that is NOT an argument.
The question must be fully answerable by calling the function once. English only.
Return ONLY a JSON array of 10 objects: {{"function": {{...}}, "question": "...", "call": {{"name": "...", "arguments": {{...}}}}}}"""


def ask(domain, key, model, seed):
    body = json.dumps({"model": model, "max_tokens": 8000, "temperature": 1.0, "seed": seed,
                       "messages": [{"role": "user", "content": PROMPT.format(domain=domain)}]}).encode()
    request = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=body, headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                j = json.load(response)
            text = (j["choices"][0]["message"].get("content") or "").strip()
            out = json.loads(text[text.find("["): text.rfind("]") + 1])
            return [x for x in out if isinstance(x, dict)], j.get("usage", {})
        except Exception:
            if attempt == 3:
                return [], {}
            time.sleep(2 * 2 ** attempt)


def valid(item):
    f, call, q = item.get("function"), item.get("call"), item.get("question")
    if not (isinstance(f, dict) and isinstance(call, dict) and isinstance(q, str) and q.strip()):
        return False
    if not re.fullmatch(r"[A-Za-z_]\w*", str(f.get("name", ""))) or call.get("name") != f.get("name"):
        return False
    params = f.get("parameters") or {}
    props, args = params.get("properties") or {}, call.get("arguments")
    if not isinstance(props, dict) or not props or not isinstance(args, dict):
        return False
    return all(k in props for k in args) and all(k in args for k in params.get("required", []) if k in props)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", type=int, default=150)
    ap.add_argument("--model", default="anthropic/claude-haiku-5.5")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(Path(__file__).with_name("tools_simple_haiku.jsonl")))
    a = ap.parse_args()
    key = os.environ["OPENROUTER_API_KEY"]
    rng = random.Random(a.seed)
    seeds = [(rng.choice(DOMAINS).strip(), a.seed * 100000 + k) for k in range(a.calls)]
    with ThreadPoolExecutor(a.workers) as ex:
        results = list(ex.map(lambda s: (s, ask(s[0], key, a.model, s[1])), seeds))
    kept, seen, bad, tin, tout = [], set(), 0, 0, 0
    for (domain, _), (items, usage) in results:
        tin += usage.get("prompt_tokens", 0)
        tout += usage.get("completion_tokens", 0)
        for it in items:
            if not valid(it) or it["question"].strip().lower() in seen:
                bad += 1
                continue
            seen.add(it["question"].strip().lower())
            kept.append({"domain": domain, "function": it["function"], "question": it["question"].strip(), "call": it["call"]})
    Path(a.out).write_text("".join(json.dumps(r) + "\n" for r in kept))
    print(f"{len(kept)} items kept from {a.calls} calls ({bad} invalid or duplicate); tokens in {tin} / out {tout}; "
          f"failed calls {sum(1 for _, (i, _) in results if not i)}")


if __name__ == "__main__":
    main()
