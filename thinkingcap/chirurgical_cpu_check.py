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


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    enc = lambda t: tok.encode(t, add_special_tokens=False)
    pairs, normal = [], []
    for k, (q, start, prod, rep) in enumerate(CASES):
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
                           columns=64, doses=[2, 6], seed=0, check_pairs=3, check_normal=6, normal_len=128)
    ch.probe(args)
    ch.mech(args, tok)
    print(json.dumps(json.loads((out / "mech.json").read_text()), indent=1))


if __name__ == "__main__":
    main()
