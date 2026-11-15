# Consult 27 Notes: Single-Step Offline DPO from Original Qwen3-0.6B

## Task Summary
Design ONE offline DPO run from the ORIGINAL Qwen3-0.6B (one round of pairs built from the original's own stored answers, any number of optimizer steps) that reaches > 52% tokens saved at better than -1 accuracy on the full test (objective: > 50% at better than -1; margin target: 52%).

## Key Full-Test Benchmarks
- Path 1:
  - r2s1b-s160: 49.0 / -1.7 (math 60/-4.0, code 51/-3.2, IFEval 40/0.0, simple 41/-2.6, several 43/+0.6, no-fit 46/+5.2)
  - r2s1b-s180: 50.3 / -1.7 (math 62/-4.5, code 54/-3.1, IFEval 38/+1.0, simple 42/-3.0, several 45/+0.9, no-fit 45/+3.5)
  - r2s2-s40 (from r2s1b-s180): 52.9 / -1.74
  - r2s2-s80 (from r2s1b-s180): 53.7 / -1.27 (math ~-3.25, code -2.38, IFEval +1.20, simple -4.69, several -0.69, no-fit +6.15)
  - r2s3-s20 (from r2s2-s80): 52.7 / -0.99 (math 59/-2.86, code 53/-1.92, IFEval 48/+0.88, simple 48/-3.50, several 51/-0.19, no-fit 52/+5.73)
  - r2s3-s30 (from r2s2-s80): 51.5 / -0.84 (math 57/-2.80, code 52/-1.62, IFEval 47/+1.39, simple 48/-3.78, several 50/-0.56, no-fit 51/+5.62)
  - r2s3b-s30 (from r2s2-s80): 51.7 / -0.86 (math 57/-2.8, code 52/-1.8, IFEval 47/+1.2, simple 49/-3.4, several 51/-0.6, no-fit 52/+6.1)
  - r2s3b-s40 (from r2s2-s80): 51.3 / -0.78 (math 55/-2.1, code 51/-1.6, IFEval 48/+0.3, simple 49/-2.9, several 51/-0.2, no-fit 52/+6.0)
- Repair chain (clean line A -> rc1-soup-30 -> rpE-80 -> rpK / rpL / rpM):
  - rpM-60: 54.1 / -0.93 (math 57/-1.2, code 58/-0.9, IFEval 50/-2.3, simple 45/-0.3, several 47/-1.6, no-fit 61/+4.2)
  - rpK-40: 51.2 / -0.8 (math 58/-2.3, code 52/-0.1, IFEval 46/-1.9, simple 41/+0.3, several 45/-0.8, no-fit 58/+4.7)
- Online reference (od5c):
  - od5c-r15: 54.0 / -0.56 (math 62/-2.8, code 54/-0.9, IFEval 48/+0.5, simple 48/-1.6, several 50/-1.4, no-fit 57/+6.9)
  - od5c-r16: 54.8 / -0.78 (math 61/-2.8, code 59/-2.0, IFEval 47/+0.5, simple 49/-1.1, several 50/0.0, no-fit 57/+5.9)
