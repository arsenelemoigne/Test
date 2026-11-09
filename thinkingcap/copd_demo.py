"""Does a COPD coach's two hats differ? (user, 9 Nov) One trick math question and a hand-written answer in six parts
(setup, a useful re-read of the question, the calculation, a needless double-check, the stop, the answer), read by each
--teachers model under the light and heavy effort instructions; prints the mean push per part, the biggest single pushes
and the push at </think>. Before Run A: the 0.6B as its own coach gave ~0 everywhere (pushes on spaces and digits).

    python copd_demo.py --teachers Qwen/Qwen3-0.6B Qwen/Qwen3-4B
"""
import argparse
import torch, warnings
warnings.filterwarnings("ignore")
from transformers import AutoTokenizer, AutoModelForCausalLM
ap = argparse.ArgumentParser()
ap.add_argument("--teachers", nargs="+", default=["Qwen/Qwen3-0.6B", "Qwen/Qwen3-4B"])
args = ap.parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
Q = ("A farmer had 160 tomatoes in his garden. If he picked 56 of them yesterday and 41 today. How many did he have left "
     "after yesterday's picking? The last line of your reply must start with 'Final answer:' followed by the number alone.")
LIGHT = "Reasoning Effort: Minimum viable, shortcuts strongly encouraged."
HEAVY = "Reasoning Effort: Absolute maximum with no shortcuts permitted."
# a student answer with: setup, a useful re-read, the calculation, a needless double-check, the stop
segments = [
    ("setup", "<think>\nThe farmer had 160 tomatoes. He picked 56 yesterday and 41 today."),
    ("re-read the question", " Wait, the question asks only about after yesterday's picking, so today's 41 does not count."),
    ("calculation", " 160 minus 56 is 104."),
    ("needless double-check", " Hmm, let me double-check that again. 56 plus 104 is 160. Yes, that seems right. Let me verify once more: 160 - 56 = 104."),
    ("stop thinking", "\n</think>\n\n"),
    ("answer", "Final answer: 104"),
]
reply_ids, owner = [], []
for name, text in segments:
    ids = tok.encode(text, add_special_tokens=False)
    reply_ids += ids; owner += [name] * len(ids)

def logps(instruction):
    msgs = [{"role": "user", "content": (instruction + "\n\n" if instruction else "") + Q}]
    p = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, enable_thinking=True)
    p = list(p["input_ids"]) if hasattr(p, "keys") else list(p)
    ids = torch.tensor([p + reply_ids], device=device)
    with torch.no_grad():
        lp = torch.log_softmax(model(ids).logits[0, len(p) - 1:-1].float(), -1)
    return lp.gather(1, torch.tensor(reply_ids, device=device)[:, None])[:, 0].tolist()

for name in args.teachers:
  model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16 if device == "cuda" else torch.float32).to(device).eval()
  print(f"\n=== coach {name}")
  light, heavy = logps(LIGHT), logps(HEAVY)
  adv = [max(-10, min(10, a - b)) for a, b in zip(light, heavy)]
  print("PER PART (mean push per token; + = keep / encourage, - = discourage):")
  for name, _ in segments:
      xs = [a for a, o in zip(adv, owner) if o == name]
      print(f"  {name:<24} {len(xs):>3} tokens   mean push {sum(xs)/len(xs):+.2f}")
  print("\nBIGGEST SINGLE-WORD PUSHES:")
  words = [(tok.decode([t]), a, o) for t, a, o in zip(reply_ids, adv, owner)]
  for w, a, o in sorted(words, key=lambda x: x[1])[:6]:
      print(f"  {a:+.2f}  {w!r:<14} ({o})")
  for w, a, o in sorted(words, key=lambda x: -x[1])[:6]:
      print(f"  {a:+.2f}  {w!r:<14} ({o})")
  close = tok.convert_tokens_to_ids("</think>")
  print("\nAT THE STOP TOKEN </think>:", [f"{a:+.2f}" for t, a in zip(reply_ids, adv) if t == close])
  del model
  torch.cuda.empty_cache() if device == 'cuda' else None
