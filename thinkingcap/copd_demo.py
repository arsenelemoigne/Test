"""Do a COPD coach's two hats differ, and in the right direction? (user, 9 Nov) 20 hand-written questions (5 math, 4 code,
4 instructions, 3 tool calls, 4 no-fit tool calls; only the tomato one comes from the test, and nothing here is trained on),
each with an answer in six parts: setup, a USEFUL check (re-read what is asked, the example, the rules, whether a tool fits),
the work, a NEEDLESS double-check, the stop (</think>) and the answer. Each --teachers model reads every answer under the
light and the heavy effort instruction (COPD's prompts); push per token = log p(light) - log p(heavy), clipped to +-10.
Printed: per question the mean push on the useful check, the needless check, the work and the stop; then per part over
all 20, and how often the needless check is pushed below the useful one. What COPD needs: needless check < 0 and below
the useful check, stop > 0. On the Mac the 0.6B as its own coach gave ~0 on the tomato question.

    python copd_demo.py --teachers Qwen/Qwen3-0.6B Qwen/Qwen3-4B
"""
import argparse
import statistics
import warnings

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore")
LIGHT = "Reasoning Effort: Minimum viable, shortcuts strongly encouraged."
HEAVY = "Reasoning Effort: Absolute maximum with no shortcuts permitted."
PARTS = ("setup", "useful check", "work", "needless check", "stop", "answer")
FINAL = " The last line of your reply must start with 'Final answer:' followed by the number alone."
CHECK_MATH = " Hmm, let me double-check that again. {v}. Yes, that seems right. Let me verify once more: {v}."
CHECK_CODE = " Hmm, let me double-check the code again. Let me verify once more that the syntax is right. Yes, that seems right."
CHECK_TEXT = " Hmm, let me double-check everything again. Let me verify once more. Yes, that seems right."
CHECK_TOOL = " Hmm, let me double-check that again. Let me verify once more. Yes, that seems right."


def case(kind, question, setup, useful, work, needless, answer):
    return {"kind": kind, "question": question,
            "parts": [("setup", "<think>\n" + setup), ("useful check", " " + useful), ("work", " " + work),
                      ("needless check", needless), ("stop", "\n</think>\n\n"), ("answer", answer)]}


def tools(spec, q):
    return f"You can call these functions: {spec}. Reply with the calls in brackets, or [] if none fits.\nQuestion: {q}"


CASES = [
    case("math", "A farmer had 160 tomatoes in his garden. If he picked 56 of them yesterday and 41 today. How many did he have "
         "left after yesterday's picking?" + FINAL, "He had 160 tomatoes and picked 56 yesterday and 41 today.",
         "Wait, the question asks only about after yesterday's picking, so today's 41 does not count.", "160 minus 56 is 104.",
         CHECK_MATH.format(v="160 - 56 = 104"), "Final answer: 104"),
    case("math", "Jane has 3 boxes with 12 crayons in each box. She also has 5 pencils. How many crayons does she have?" + FINAL,
         "Jane has 3 boxes of 12 crayons and 5 pencils.", "The question asks about crayons only, so the 5 pencils do not matter.",
         "3 times 12 is 36.", CHECK_MATH.format(v="3 x 12 = 36"), "Final answer: 36"),
    case("math", "After giving away 15 stickers, Sam has 27 stickers left. How many stickers did he have at first?" + FINAL,
         "Sam gave away 15 stickers and has 27 left.", "Wait, it asks how many he had at first, so I add back, I do not subtract.",
         "27 plus 15 is 42.", CHECK_MATH.format(v="27 + 15 = 42"), "Final answer: 42"),
    case("math", "A shop sold 48 apples in the morning and 3 times as many in the afternoon. How many more apples were sold in "
         "the afternoon than in the morning?" + FINAL, "Morning 48 apples, afternoon 3 times as many.",
         "The question asks how many more, so I need the difference, not the total.",
         "Afternoon is 3 times 48, which is 144. 144 minus 48 is 96.", CHECK_MATH.format(v="144 - 48 = 96"), "Final answer: 96"),
    case("math", "A bus had 30 people. At the first stop 12 people got off and 7 got on. How many people got off the bus at the "
         "first stop?" + FINAL, "The bus had 30 people; 12 got off and 7 got on.",
         "The question only asks how many got off, which is given directly; no need to compute who is left.",
         "12 people got off.", CHECK_MATH.format(v="12 got off"), "Final answer: 12"),
    case("code", "Write a Python function to merge three lists into a single sorted list.\nassert merge_sorted([3, 1], [2], [0]) == "
         "[0, 1, 2, 3]", "I need merge_sorted taking three lists.",
         "The example shows the result must be sorted, so joining the lists is not enough; I must sort the combined list.",
         "Join them with a + b + c and return sorted(...).", CHECK_CODE,
         "```python\ndef merge_sorted(a, b, c):\n    return sorted(a + b + c)\n```"),
    case("code", "Write a Python function that returns the common elements of nested lists.\nassert set(common([[1, 2, 3], [2, 3, 4], "
         "[3, 2]])) == {2, 3}", "I need common taking a list of lists.",
         "Common means present in every list, so this is an intersection, not a union of all elements.",
         "Start from the first list as a set and intersect it with each other list.", CHECK_CODE,
         "```python\ndef common(lists):\n    result = set(lists[0])\n    for l in lists[1:]:\n        result &= set(l)\n    return "
         "list(result)\n```"),
    case("code", "Write a Python function to convert a tuple of positive integers into a single integer.\nassert tuple_to_int((1, 2, "
         "3)) == 123", "I need tuple_to_int taking a tuple of digits.",
         "int() cannot take a tuple directly; I have to join the digits as strings first.",
         "Use int(''.join(map(str, t))).", CHECK_CODE,
         "```python\ndef tuple_to_int(t):\n    return int(''.join(map(str, t)))\n```"),
    case("code", "Write a Python function that returns the n-th triangular number.\nassert triangle(4) == 10",
         "I need triangle(n).", "Check against the example: for 4, 1 + 2 + 3 + 4 is 10, so n * (n + 1) // 2 matches.",
         "Return n * (n + 1) // 2.", CHECK_CODE, "```python\ndef triangle(n):\n    return n * (n + 1) // 2\n```"),
    case("ifeval", "Write a tweet about coffee without using any capital letters. Include at least 3 hashtags.",
         "A short tweet about coffee.", "The rules: no capital letters at all, even in hashtags, and at least three hashtags.",
         "Draft: morning coffee is my favorite ritual, with three lowercase hashtags.", CHECK_TEXT,
         "morning coffee is my favorite ritual, warm and slow. #coffee #morning #slowliving"),
    case("ifeval", "Write a poem about the sea of at least 60 words.", "A poem about the sea.",
         "It must be at least 60 words, so I need several stanzas, not a short verse.",
         "Four stanzas: waves, light, sailors, night.", CHECK_TEXT,
         "The waves roll in with silver light,\nthey fold and break from morning to night.\n\nThe sun lies low on the restless blue,\n"
         "and every crest is shining new.\n\nThe sailors sing of distant shores,\nof rocking decks and creaking oars.\n\n"
         "And when the dark comes soft and deep,\nthe sea will hum the world to sleep,\nits endless song of wind and foam,\n"
         "that calls each wandering sailor home."),
    case("ifeval", "Give two different answers to: why do cats purr? Separate the two answers with six asterisks: ******",
         "Two answers about why cats purr.",
         "It wants exactly two separate responses with ****** between them, so I must not merge them into one.",
         "First: they are content. Second: the vibration may help them heal.", CHECK_TEXT,
         "Cats often purr when they feel safe and content.\n******\nPurring may also help cats heal, since the vibrations can "
         "support bones and tissue."),
    case("ifeval", "Explain photosynthesis in exactly 3 bullet points. Use markdown bullet points such as:\n* This is a point.",
         "Explain photosynthesis briefly.", "Exactly three bullets, no more and no less, each starting with an asterisk.",
         "Points: light absorption, sugar making, oxygen release.", CHECK_TEXT,
         "* Plants absorb sunlight with chlorophyll.\n* They turn water and carbon dioxide into sugar.\n* They release oxygen."),
    case("tool", tools("calculate_return_on_equity(net_income, shareholder_equity, dividends_paid=0)",
                       "Calculate the return on equity with net income 2000000, equity 10000000 and dividends paid 200000."),
         "The user wants the return on equity.", "They also gave the dividends, so I must pass dividends_paid, not keep the default.",
         "Call calculate_return_on_equity with all three values.", CHECK_TOOL,
         "[calculate_return_on_equity(net_income=2000000, shareholder_equity=10000000, dividends_paid=200000)]"),
    case("tool", tools("get_weather(city, unit='celsius')", "What is the weather in Paris in fahrenheit?"),
         "The user wants the weather in Paris.", "They asked for fahrenheit, so I must set unit and not keep the celsius default.",
         "Call get_weather for Paris in fahrenheit.", CHECK_TOOL, "[get_weather(city='Paris', unit='fahrenheit')]"),
    case("tool", tools("book_flight(origin, destination, date), get_hotels(city)", "Book a flight from Rome to Oslo on 2026-12-03."),
         "The user wants a flight.", "Only the flight is requested, so I call book_flight and not get_hotels.",
         "Call book_flight with Rome, Oslo and the date.", CHECK_TOOL,
         "[book_flight(origin='Rome', destination='Oslo', date='2026-12-03')]"),
    case("nofit", tools("search_music_instrument_players(instrument, genre)", "Who is the best jazz singer?"),
         "The user asks about jazz singers.", "The tool searches instrument players; a singer is not an instrument, so no tool fits.",
         "None of the functions can answer this.", CHECK_TOOL, "[]"),
    case("nofit", tools("calculate_vector_angle(X_component, Y_component)", "What is a holy book?"),
         "The user asks what a holy book is.", "The only tool computes vector angles, which has nothing to do with this question.",
         "So I call nothing.", CHECK_TOOL, "[]"),
    case("nofit", tools("get_top_scorer(league, season)", "Who has the most assists in the Premier League?"),
         "The user asks about assists.", "The tool returns top scorers, not assist leaders, so it does not answer this.",
         "So no call fits.", CHECK_TOOL, "[]"),
    case("nofit", tools("convert_currency(amount, from_currency, to_currency)", "When was the Treaty of Lisbon signed?"),
         "The user asks for a historical date.", "Converting currency cannot give a historical date, so no function fits.",
         "So I call nothing.", CHECK_TOOL, "[]"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--teachers", nargs="+", default=["Qwen/Qwen3-0.6B", "Qwen/Qwen3-4B"])
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    for c in CASES:
        c["ids"], c["owner"] = [], []
        for name, text in c["parts"]:
            ids = tok.encode(text, add_special_tokens=False)
            c["ids"] += ids
            c["owner"] += [name] * len(ids)
    for teacher in args.teachers:
        model = AutoModelForCausalLM.from_pretrained(
            teacher, dtype=torch.bfloat16 if device == "cuda" else torch.float32).to(device).eval()

        def logps(question, instruction, reply):
            p = tok.apply_chat_template([{"role": "user", "content": instruction + "\n\n" + question}], tokenize=True,
                                        add_generation_prompt=True, enable_thinking=True)
            p = list(p["input_ids"]) if hasattr(p, "keys") else list(p)
            ids = torch.tensor([p + reply], device=device)
            with torch.no_grad():
                lp = torch.log_softmax(model(ids).logits[0, len(p) - 1:-1].float(), -1)
            return lp.gather(1, torch.tensor(reply, device=device)[:, None])[:, 0].tolist()

        print(f"\n=== coach {teacher}: mean push per token (+ kept/encouraged, - discouraged)")
        print(f"  {'question':<14}" + "".join(f"{p:>16}" for p in PARTS))
        per_part, wins = {p: [] for p in PARTS}, 0
        for i, c in enumerate(CASES):
            adv = [max(-10.0, min(10.0, a - b)) for a, b in
                   zip(logps(c["question"], LIGHT, c["ids"]), logps(c["question"], HEAVY, c["ids"]))]
            means = {p: statistics.fmean(a for a, o in zip(adv, c["owner"]) if o == p) for p in PARTS}
            for p in PARTS:
                per_part[p].append(means[p])
            wins += means["needless check"] < means["useful check"]
            print(f"  {i + 1:>2} {c['kind']:<11}" + "".join(f"{means[p]:>+16.2f}" for p in PARTS))
        print(f"  {'MEAN (20)':<14}" + "".join(f"{statistics.fmean(per_part[p]):>+16.2f}" for p in PARTS))
        print(f"  needless check pushed below the useful check in {wins}/{len(CASES)} questions; "
              f"needless check < 0 in {sum(x < 0 for x in per_part['needless check'])}/{len(CASES)}; "
              f"stop > 0 in {sum(x > 0 for x in per_part['stop'])}/{len(CASES)}")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
