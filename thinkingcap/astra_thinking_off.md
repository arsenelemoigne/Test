# Thinking ON vs OFF, the original Qwen3-0.6B (a measurement, NOT an allowed solution)

The original model answered 240 test-style questions (40 per part, from the same four benchmarks) 8 times each, once with
thinking on and once with thinking switched off (an empty thinking block). Turning thinking off is forbidden by the
rules; these numbers only show how much each part depends on its thinking. Small sample: about +-3 to 5 points per part.

| part | accuracy, thinking ON | accuracy, thinking OFF | change (pts) | tokens ON | tokens OFF |
|---|---|---|---|---|---|
| bfcl_irrelevance (no function fits) | 80.9 | 87.2 | +6.2 | 195 | 8 |
| bfcl_multiple (pick one of 2-4) | 75.9 | 58.8 | -17.2 | 206 | 20 |
| bfcl_simple_python (one function) | 88.1 | 76.6 | -11.6 | 197 | 24 |
| ifeval (instruction constraints) | 65.6 | 60.6 | -5.0 | 703 | 284 |
| mbpp (Python functions) | 46.2 | 37.2 | -9.1 | 2,098 | 53 |
| svamp (math word problems) | 89.7 | 17.5 | -72.2 | 644 | 10 |
