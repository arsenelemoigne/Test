"""Haiku-written writing requests for built IFEval questions (14 Nov, user: "for any missing data, you can generate it
with haiku"). Our human request sources (No Robots, Dolly, OpenAssistant) are used up; ifeval_built needs plain requests
(no format rules: IFEval's instructions are added afterwards by ifeval_fill, matched to IFEval's instruction mix).

Each call asks Claude Haiku (via OpenRouter, the route of our earlier Haiku batches) for 20 different requests a real person
might type, for one (topic, kind) seed: generation / brainstorm / open question / advice / rewrite (text included) /
summarize (text included). Kept: English, 12-150 words, no coding words, no format or length demands, exact and near
duplicates (8-word overlap) removed. Written to ifeval_requests_haiku.jsonl (one {"text", "kind", "topic"} per line).
The 8-gram check against every evaluation question and the request-level check run later (newtop / build_pool).
Usage: OPENROUTER_API_KEY=... python gen_ifeval_requests.py --calls 350 --out ifeval_requests_haiku.jsonl
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

TOPICS = """travel planning, cooking at home, personal finance, job interviews, career change, parenting toddlers, teenagers,
school homework, college applications, pets and dogs, cats, gardening, house plants, home repair, moving house, renting,
buying a car, cycling, running, yoga, sleep, healthy eating, mental wellbeing, friendship, dating, weddings, birthdays,
holidays and festivals, small business, starting a bakery, a coffee shop, marketing a local shop, customer complaints,
writing a cover letter, a resignation letter, a thank-you note, a speech for a friend, a eulogy, a toast, poetry about
nature, a short story, a fairy tale for children, science fiction, a mystery, history of ancient Rome, medieval castles,
the industrial revolution, world war two, space exploration, the solar system, oceans and marine life, climate and
weather, volcanoes, dinosaurs, birds, insects, forests, mountains, rivers, cities, architecture, museums, art and painting,
photography, music and instruments, learning a language, reading habits, book club, films, theatre, video games, board
games, chess, football, basketball, tennis, the olympics, fitness at home, nutrition myths, first aid, visiting a doctor,
elderly parents, volunteering, charity events, neighbourhood events, recycling, saving energy, electric cars, public
transport, smartphones for beginners, online safety for grandparents, social media habits, remote work, office etiquette,
meetings, time management, studying for exams, public speaking, negotiation, budgeting a trip, camping, hiking, fishing,
the beach, winter sports, a road trip, a museum visit, a restaurant review, a product review, a hotel complaint, a lost
package, a landlord dispute, philosophy questions, ethics dilemmas, economics basics, how elections work, local news, a
community newsletter, a school newsletter, a recipe blog, a travel blog, a podcast episode, a product launch, a charity
appeal, an apology, a job advertisement, a menu description, a tour guide script""".replace("\n", " ").split(", ")
KINDS = {
    "generation": "ask the assistant to WRITE something (a story, poem, letter, post, description, script, etc.)",
    "brainstorm": "ask the assistant for IDEAS or options",
    "open_qa": "ask a factual or explanatory QUESTION",
    "advice": "describe a personal situation and ask for ADVICE",
    "rewrite": "ask the assistant to REWRITE, rephrase or improve a short text, and INCLUDE that text (2-5 sentences) in "
               "the request",
    "summarize": "ask the assistant to SUMMARIZE or explain a short passage, and INCLUDE the passage (3-6 sentences) in "
                 "the request",
}
PROMPT = """Write 20 different requests that real people might type to an AI assistant. Topic: {topic}. Kind: each request
should {kind}.
Rules: plain English, as a real person would write (varied tone, some casual, some formal), each between 15 and 120 words.
Do NOT include any instruction about format or length (no word counts, no number of paragraphs or bullet points, no
JSON, no titles, no language or capitalisation demands, no "in N words"). Nothing about programming or code.
The 20 requests must be clearly different from each other.
Return ONLY a JSON array of 20 strings."""
CODING = re.compile(r"\b(python|code|coding|script|function|program|programming|sql|javascript|java|c\+\+|html|css|regex|"
                    r"bash|api|json|excel formula)\b", re.I)
FORMAT = re.compile(r"\b(\d+\s+(words|sentences|paragraphs|bullet|points|lines)|bullet points?|in json|all caps|"
                    r"capital letters|title in|markdown)\b", re.I)


def ask(topic, kind, key, model, seed):
    body = json.dumps({"model": model, "max_tokens": 6000, "temperature": 1.0, "seed": seed,
                       "messages": [{"role": "user", "content": PROMPT.format(topic=topic, kind=KINDS[kind])}]}).encode()
    request = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=body, headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                j = json.load(response)
            text = (j["choices"][0]["message"].get("content") or "").strip()
            text = text[text.find("["): text.rfind("]") + 1]
            out = json.loads(text)
            return [str(x) for x in out if isinstance(x, str)], j.get("usage", {})
        except Exception:
            if attempt == 3:
                return [], {}
            time.sleep(2 * 2 ** attempt)


def grams(t):
    w = re.findall(r"\w+", t.lower())
    return {tuple(w[i:i + 8]) for i in range(max(0, len(w) - 7))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", type=int, default=350)
    ap.add_argument("--model", default="anthropic/claude-haiku-5.5")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(Path(__file__).with_name("ifeval_requests_haiku.jsonl")))
    a = ap.parse_args()
    key = os.environ["OPENROUTER_API_KEY"]
    rng = random.Random(a.seed)
    seeds = [(rng.choice(TOPICS).strip(), rng.choice(list(KINDS)), a.seed * 100000 + k) for k in range(a.calls)]
    with ThreadPoolExecutor(a.workers) as ex:
        results = list(ex.map(lambda s: (s, ask(s[0], s[1], key, a.model, s[2])), seeds))
    kept, seen, drop, tok_in, tok_out = [], set(), {"length": 0, "coding": 0, "format": 0, "duplicate": 0}, 0, 0
    for (topic, kind, _), (texts, usage) in results:
        tok_in += usage.get("prompt_tokens", 0)
        tok_out += usage.get("completion_tokens", 0)
        for t in texts:
            t = " ".join(t.split())
            if not 12 <= len(t.split()) <= 150:
                drop["length"] += 1
                continue
            if CODING.search(t):
                drop["coding"] += 1
                continue
            if FORMAT.search(t):
                drop["format"] += 1
                continue
            g = grams(t)
            if t.lower() in seen or (g and len(g & seen) * 10 >= 3 * len(g)):
                drop["duplicate"] += 1
                continue
            seen.add(t.lower())
            seen |= g
            kept.append({"text": t, "kind": kind, "topic": topic})
    Path(a.out).write_text("".join(json.dumps(r) + "\n" for r in kept))
    print(f"{len(kept)} requests kept from {a.calls} calls; dropped {drop}; tokens in {tok_in} / out {tok_out}; "
          f"calls that failed {sum(1 for _, (t, _) in results if not t)}")


if __name__ == "__main__":
    main()
