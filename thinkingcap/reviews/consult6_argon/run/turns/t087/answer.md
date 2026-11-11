```python
# Let's now do a deep, line-by-line code review of all 4 files in pack/code/:
# 1. `repair_pairs.py`
# 2. `dev_score.py`
# 3. `track_ptrepair_odpo_dev.py`
# 4. `queued_lines.txt`
#
# Wait: let's look VERY carefully at the bugs and mismatches we already spotted and check if there are any more!
# Let's list and verify every bug in Python!

# BUG 1 (CRITICAL — silently drops CLI arguments in `ptrepair`!):
# Look at `queued_lines.txt`:
#   ptrepair --name p1r1s0 ... --rp-lr 1.5e-5 --rp-lr-min 5e-6 --rp-warmup 10 --rp-steps 200 --rp-epochs 2 --rp-dpo-seed 0 ...
#   ptrepair --name p1r2s0 ... --rp-exclude-dev --rp-exclude-ids data/p1-round1-ids.json ... --rp-dpo-seed 0 ...
#   ptrepair --name p1r1s1 ... --rp-dpo-seed 1 ...
#   ptrepair --name p1r2s1 ... --rp-dpo-seed 1 ...
# Now look at `ptrepair(args)` in `track_ptrepair_odpo_dev.py`:
#   1) Line 70:
#      `for r in own_repair_questions({p: n * n_rounds for p, n in want.items()}, seed=args.rp_seed,`
#      `                              exclude=dev_ids() if getattr(args, "rp_exclude_dev", False) else None):`
#      WAIT A SECOND!
#      The developer wrote `def excluded_ids(args):` at Line 396 to combine `dev_ids()` (`--rp-exclude-dev`) AND `args.rp_exclude_ids` (`data/p1-round1-ids.json`)!
#      Wait: did they actually call `excluded_ids(args)` in `ptrepair`?!
#      In `track_ptrepair_odpo_dev.py` Line 70, `ptrepair` calls `exclude=dev_ids() if getattr(args, "rp_exclude_dev", False) else None` instead of `exclude=excluded_ids(args)`!
#      Wait: if `ptrepair` in `track.py` calls `dev_ids()` instead of `excluded_ids(args)`, then `--rp-exclude-ids data/p1-round1-ids.json` is COMPLETELY IGNORED in Round 2 (`p1r2s0` and `p1r2s1`), meaning Round 2 will draw the SAME questions that Round 1 trained on!
#   2) Look at `ptrepair` Lines 112 and 121-125 in `track_ptrepair_odpo_dev.py`:
#      Line 112: `adapter = Path("adapters") / f"{name}-s0"` (hardcodes `-s0` even when `--rp-dpo-seed 1`!)
#      Line 122-125:
#        `py("dpo.py", "--model", merged, "--pairs", pairs / "pairs.jsonl", "--init-adapter", start, "--out", adapter,`
#        `   "--mix", "all", "--keep-order", "--nll", 0.0, "--beta", 0.1, "--batch", 16, "--lr", args.rp_lr,`
#        `   "--lr-schedule", "cosine", "--warmup", 2, "--lr-min", args.rp_lr, "--max-steps", steps, "--epochs", args.rp_epochs,`
#        `   "--save-every",`
#        `   1 if smoke else 5, "--held-out", 0, "--max-len", 8704, "--bida-precision", "--dtype", args.dtype, "--seed", 0)`
#      WAIT A MINUTE! Look at Line 123 and Line 125:
#      - `"--warmup", 2` is HARDCODED to `2` (ignores `args.rp_warmup`, which is passed as `--rp-warmup 10` in `p1r1s0` and `p1r1s1`!)
#      - `"--lr-min", args.rp_lr` is HARDCODED to `args.rp_lr` (ignores `args.rp_lr_min`, which is passed as `--rp-lr-min 5e-6` in `p1r1s0` and `p1r1s1`! So Round 1 runs at a CONSTANT `1.5e-5` LR instead of cosine decay to `5e-6`!)
#      - `"--seed", 0` is HARDCODED to `0` (ignores `args.rp_dpo_seed`, which is passed as `--rp-dpo-seed 1` in `p1r1s1` and `p1r2s1`! AND `repair_pairs.py` is called at Line 108 WITHOUT passing `--seed args.rp_dpo_seed`, AND `sft.py --init-only` at Line 118 is called with default seed! That means `p1r1s0` and `p1r1s1` have the EXACT SAME pair draw, the EXACT SAME pair order, the EXACT SAME LoRA init, and the EXACT SAME DPO seed `0`—so Seed 0 and Seed 1 are identical except for GPU non-determinism!).

print("Checked ptrepair args wiring in track_ptrepair_odpo_dev.py!")
```