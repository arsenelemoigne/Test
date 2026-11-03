"""CPU check of chirurgical.py on the REAL Qwen3-0.6B (user, 3 Nov: "test it a little on my computer"): hand-written
matched pairs as in the paper's Exp 8 (same correct start, then a productive check or the same phrase repeated), plus
short finished answers; runs probe -> candidates -> mech. Generation (vLLM) is left to the Colab smoke run.

    python chirurgical_cpu_check.py            # ~10-20 min on an 8 GB Mac, bfloat16
"""
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import chirurgical as ch

CASES = [  # (question, shared start, productive continuation, sentence that gets repeated)
    ("Tom has 12 apples, gives 5 to Ana, then buys 8 more. How many apples does he have?",
     "Tom starts with 12 apples. He gives 5 to Ana, so 12 - 5 = 7. Then he buys 8 more, so 7 + 8 = 15.",
     " Let me check with the other order: 12 + 8 = 20, and 20 - 5 = 15. Both ways give 15, so the answer is 15.",
     " Wait, let me check again: 12 - 5 = 7 and 7 + 8 = 15."),
    ("A shop sells 3 pens for $4. How much do 12 pens cost?",
     "Twelve pens is 12 / 3 = 4 groups of three pens. Each group costs $4, so 4 x 4 = 16 dollars.",
     " As a check, one pen costs 4/3 dollars, and 12 x 4/3 = 16. So 12 pens cost $16.",
     " So 4 groups times $4 is $16."),
    ("Write a Python function that returns the largest element of a list.",
     "I need a function max_of(lst). The simplest way is to use the built-in max, but an empty list would raise.",
     " I will return None for an empty list, otherwise loop and keep the biggest value seen so far, which also works for negative numbers.",
     " The function should return the largest element of the list."),
    ("Write a short poem about rain with exactly three lines.",
     "Three lines, about rain. First line could set the scene: grey clouds over the town.",
     " Second line: the sound of drops on the roof. Third line: the smell of wet earth afterwards. That makes three lines.",
     " I need exactly three lines about rain."),
    ("Mia reads 15 pages a day. How many pages does she read in 6 days?",
     "She reads 15 pages each day for 6 days, so 15 x 6 = 90 pages.",
     " Checking: 10 x 6 = 60 and 5 x 6 = 30, and 60 + 30 = 90. The answer is 90.",
     " Let me verify: 15 times 6 is 90."),
    ("A train travels 60 km per hour for 3 hours. How far does it go?",
     "Distance is speed times time: 60 x 3 = 180 km.",
     " In one hour it goes 60 km, in two hours 120 km, in three hours 180 km. So 180 km.",
     " Wait, 60 times 3 equals 180."),
]


CASES_B = [  # (3 Nov, user: same result on a different batch?) other questions, other repeated phrases
    ("Sara buys 4 boxes of 6 eggs and breaks 3. How many eggs are left?",
     "Four boxes of six eggs is 4 x 6 = 24 eggs. She breaks 3, so 24 - 3 = 21.",
     " To double check, 6 + 6 + 6 + 6 = 24, minus the 3 broken ones gives 21 eggs left.",
     " Hmm, so the total is 21 eggs."),
    ("A rectangle is 8 cm long and 5 cm wide. What is its area?",
     "Area of a rectangle is length times width: 8 x 5 = 40 square centimetres.",
     " Splitting it, 8 x 5 is the same as 4 x 10 = 40, so the area is 40 cm^2.",
     " Let me recompute 8 times 5, which is 40."),
    ("Write a Python function that counts the vowels in a string.",
     "I need count_vowels(s). Vowels are a, e, i, o, u, and capital letters should count too.",
     " I will lower-case the string once and sum 1 for each character found in the set of vowels; an empty string gives 0.",
     " The function counts the vowels in the string."),
    ("Answer in exactly two sentences: why is the sky blue?",
     "Two sentences only. The first should say sunlight is scattered by air molecules.",
     " The second should say blue light scatters more than red because its wavelength is shorter, which is why we see blue.",
     " I must answer in exactly two sentences."),
    ("Leo has 45 marbles and shares them equally among 5 friends. How many does each get?",
     "Sharing 45 equally among 5 means 45 / 5 = 9 marbles each.",
     " Checking by multiplying back, 9 x 5 = 45, which matches, so each friend gets 9.",
     " Wait, 45 divided by 5 is 9."),
    ("Call the weather tool for Paris tomorrow. Available: get_weather(city, day).",
     "The user wants tomorrow's weather in Paris, and get_weather takes a city and a day.",
     " So the call is get_weather(city=\"Paris\", day=\"tomorrow\"); no other function is needed.",
     " The function is get_weather with city and day."),
]


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    enc = lambda t: tok.encode(t, add_special_tokens=False)
    pairs, normal = [], []
    import sys
    cases = CASES_B if "--batch-b" in sys.argv else CASES
    for k, (q, start, prod, rep) in enumerate(cases):
        prompt = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=True, add_generation_prompt=True)
        prompt = list(prompt["input_ids"]) if hasattr(prompt, "keys") else list(prompt)
        prefix = enc("<think>\n" + start)
        pairs.append({"id": f"c{k}", "part": "x", "prompt_ids": prompt, "prefix": prefix, "loop": enc(rep * 4),
                      "productive": enc(prod), "split": "find" if k % 2 == 0 else "check"})
        normal.append({"id": f"n{k}", "part": "x", "prompt_ids": prompt,
                       "output_ids": enc("<think>\n" + start + prod + "\n</think>\n\nThe answer follows from the work above.")})
    out = Path(tempfile.mkdtemp()) / "cpu"
    out.mkdir()
    ch.write_jsonl(out / "pairs.jsonl", pairs)
    ch.write_jsonl(out / "normal.jsonl", normal)
    ch.DTYPE = "bfloat16"
    args = SimpleNamespace(out=str(out), model="Qwen/Qwen3-0.6B", attn_pairs=3, attn_len=512, heads=2, layers=2,
                           columns=64, doses=[2, 6], seed=0, check_pairs=3, check_normal=6, normal_len=128,
                           min_stability=0.8, min_copy_gap=0.2)
    ch.probe(args)
    ch.mech(args, tok)
    print(json.dumps(json.loads((out / "mech.json").read_text()), indent=1))


if __name__ == "__main__":
    main()
