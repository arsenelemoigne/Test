"""Candidate prompts for prompt_search.py (a .py file: the Colab runtime copies only *.py and plan*.txt)."""

CANDIDATES = [
 {
  "name": "base",
  "note": "no prompt: the reference every candidate is paired against"
 },
 {
  "name": "concise",
  "system": "Solve the problem concisely and correctly.",
  "note": "Lightning McQwen3's simplest prompt"
 },
 {
  "name": "prefill",
  "prefill": "Okay, I'll solve this efficiently and only check what I'm unsure about.",
  "note": "Lightning McQwen3: no system prompt, the thinking starts with this sentence"
 },
 {
  "name": "nocheck",
  "system": "Think efficiently: work out each step once and stop thinking as soon as you know the answer. Do not re-read the question or re-check results you are already sure of.",
  "leak": [
   "re-check",
   "re-read",
   "efficiently"
  ],
  "note": "names the two habits the cut lots show as waste (re-reading, re-checking)"
 },
 {
  "name": "nocheck-prefill",
  "system": "Think efficiently: work out each step once and stop thinking as soon as you know the answer. Do not re-read the question or re-check results you are already sure of.",
  "prefill": "Okay, I'll solve this efficiently and only check what I'm unsure about.",
  "leak": [
   "re-check",
   "re-read"
  ]
 },
 {
  "name": "budget",
  "system": "Think briefly: keep your thinking under 100 tokens.",
  "leak": [
   "token",
   "100",
   "budget",
   "under 100"
  ],
  "note": "token-budget prompt (TALE, Han et al. 2024), one budget for every type"
 },
 {
  "name": "cod",
  "system": "Think step by step, but only keep a minimum draft for each thinking step, with 5 words at most.",
  "leak": [
   "draft",
   "5 words",
   "five words"
  ],
  "note": "Chain of Draft (Xu et al. 2025) without examples"
 },
 {
  "name": "examples",
  "system": "Think briefly, like in these two examples.\n\nExample 1\nQuestion: A shop sells pens at $3 each. Tom buys 4 pens and pays with a $20 bill. How much change does he get?\nThinking: 4 pens cost 4 x 3 = 12 dollars. Change: 20 - 12 = 8.\nFinal answer: 8\n\nExample 2\nQuestion: Where would you keep milk to stop it from spoiling?\nA. cupboard\nB. refrigerator\nC. oven\nD. table\nThinking: Cold slows spoiling, so the refrigerator.\nFinal answer: B",
  "leak": [
   "pen",
   "milk",
   "refrigerator",
   "spoil",
   "tom",
   "$20",
   "20 bill"
  ],
  "note": "Lightning McQwen3's 'examples' prompt"
 },
 {
  "name": "exdraft",
  "system": "Think step by step, but only keep a minimum draft for each thinking step, with 5 words at most, like in these two examples.\n\nExample 1\nQuestion: A shop sells pens at $3 each. Tom buys 4 pens and pays with a $20 bill. How much change does he get?\nThinking: 4 x 3 = 12; 20 - 12 = 8.\nFinal answer: 8\n\nExample 2\nQuestion: Where would you keep milk to stop it from spoiling?\nA. cupboard\nB. refrigerator\nC. oven\nD. table\nThinking: cold stops spoiling: refrigerator.\nFinal answer: B",
  "leak": [
   "pen",
   "milk",
   "refrigerator",
   "spoil",
   "tom",
   "$20",
   "20 bill",
   "draft",
   "5 words",
   "five words"
  ],
  "note": "Lightning McQwen3's final 'ex-draft' prompt (the one he distilled)"
 },
 {
  "name": "exdraft-prefill",
  "system": "Think step by step, but only keep a minimum draft for each thinking step, with 5 words at most, like in these two examples.\n\nExample 1\nQuestion: A shop sells pens at $3 each. Tom buys 4 pens and pays with a $20 bill. How much change does he get?\nThinking: 4 x 3 = 12; 20 - 12 = 8.\nFinal answer: 8\n\nExample 2\nQuestion: Where would you keep milk to stop it from spoiling?\nA. cupboard\nB. refrigerator\nC. oven\nD. table\nThinking: cold stops spoiling: refrigerator.\nFinal answer: B",
  "prefill": "Okay, I'll solve this efficiently and only check what I'm unsure about.",
  "leak": [
   "pen",
   "milk",
   "refrigerator",
   "spoil",
   "tom",
   "$20",
   "20 bill",
   "draft",
   "5 words",
   "five words"
  ]
 },
 {
  "name": "draft-typed",
  "system": "Think step by step, but only keep a minimum draft for each thinking step, with 5 words at most, like in these examples. Then give the reply in the format the task asks for.\n\nExample 1 (arithmetic)\nQuestion: A shop sells pens at $3 each. Tom buys 4 pens and pays with a $20 bill. How much change does he get?\nThinking: 4 x 3 = 12; 20 - 12 = 8.\nReply: Final answer: 8\n\nExample 2 (code)\nQuestion: Write a function last_word(s) that returns the last word of a sentence.\nThinking: split on spaces; return last.\nReply: def last_word(s):\n    return s.split()[-1]\n\nExample 3 (tool call)\nQuestion: Functions offered: get_weather(city: str, unit: str = \"celsius\"). User: What's the weather in Lyon in fahrenheit?\nThinking: get_weather; city Lyon; unit fahrenheit.\nReply: [get_weather(city=\"Lyon\", unit=\"fahrenheit\")]\n\nExample 4 (instructions)\nQuestion: Describe a lighthouse in exactly two sentences, all in lowercase.\nThinking: two sentences; lowercase only.\nReply: a lighthouse stands on the rocky shore. its light guides ships at night.",
  "leak": [
   "pen",
   "tom",
   "$20",
   "last_word",
   "lyon",
   "weather",
   "lighthouse",
   "draft",
   "5 words",
   "five words"
  ],
  "note": "ex-draft with one example per evaluated task type (math, code, tool call, instructions), written for this test; matches Lightning McQwen3's 'draft-mixed' idea for our weak types"
 }
]
