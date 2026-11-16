# What the original Qwen3-0.6B gets wrong on tool calls (16 Nov, its stored answers: 2,730 no-fit and 1,426 one-function questions x10)

- No-fit: mean solve 0.917, only 10% of questions solved < 80% (Glaive 12%, ToolACE 8%). Its failure is always the same:
  it CALLS a function. Typical traps among the 170 questions it solves <= 35%:
  (a) sibling entity: the function handles a related entity of the same domain ("Basic Latin" Unicode block -> a language
  lookup; Bollywood movies -> an anime list; Italian restaurants -> a desserts menu; African elephant -> African country
  resources); (b) sibling capability: a calculator close to the request (discount + tax -> net salary, discount percent,
  commission); (c) keyword bait (cricket -> Pakistan news); (d) two-part requests where neither part fits.
- One function: mean solve 0.82, 23.5% of questions solved < 80% (ToolACE 34%, Glaive 17%). Failures: ungrounded extra
  optional arguments (country, category), list vs string (keywords split or merged), a second call for an unsupported
  follow-up, malformed calls, no call on chatty requests, percentage conventions (0.2 vs 20).
- Use: gen_targeted_tools.py gives these traps to the generator as ABSTRACT descriptions (never our real questions), in 40
  domains never used before.
