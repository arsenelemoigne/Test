# Consult 33 Working Notes (`od5h` vs `od5i`)

## 1. Files Read
- `README_PACK.md`
- `YOUR_CONSULT31_REPORT.md` (Consult 31 plan for `od5h`)
- `YOUR_CONSULT32_REPORT.md` (Consult 32 pre-launch audit)
- `OD5H_ROUNDS.md` & `OD5H_COLAB_LOG.txt` (actual `od5ht` and `od5h` rounds 1..7 logs)
- `RUN_LINES.txt` & `od5h_plan_lines.txt`
- `code_changes.diff` & `code_changes_since_consult32.diff`
- `code/repair_pairs.py` (lines 340-770)

## 2. Empirical Trajectory of `od5h` (Rounds 1..7)
Per-round check (model r(k-1) on slice k; reads ~2.5-3 pts optimistic on accuracy vs full test):
- **r2 (r1 on slice 2):** `30.1% / -0.06` (`no-fit 28/-0.1, code 33/+0.0, math 36/-1.0, ifeval 26/+0.5, mult 21/+0.1, simple 25/+0.5`). Brake reads 1 slice; held = none.
- **r3 (r2 on slice 3):** `42.1% / -2.25` (`no-fit 42/-2.2, code 47/-4.1, math 48/-2.8, ifeval 35/-1.5, simple 32/+0.2, mult 44/-0.2`).
  - 2-slice brake mean (r2,r3): `code 40/-2.1` -> **HELD (`mbpp`)**. (`math 42/-1.9`, just above -2.0).
  - Round 3 training pairs on held `mbpp` (damaged `-4.1 <= -2.0`): `14 recovery + 29 compact hold` (43 pairs + loop/format).
- **r4 (r3 on slice 4):** `45.6% / -1.44` (`mult 37/-0.7, code 55/-4.3, math 55/-2.0, ifeval 36/+0.1, no-fit 40/+1.2, simple 35/+0.4`).
  - Even though `mbpp` was HELD in round 3, `mbpp` shortened `47% -> 55%` and stayed at `-4.3`!
  - 2-slice brake mean (r3,r4): `mbpp 51/-4.2`, `svamp 51/-2.4` -> **HELD (`mbpp`, `svamp`)**.
  - Round 4 training pairs: `mbpp` (slice dmg `-4.3 <= -2.0`): `20 recovery + 27 compact hold`, trained on `211 pairs` total (15 steps = 240 updates, wrapping 29 pairs!). `svamp` (slice dmg `-2.0` -> `dmg = -0.0200` check whether `<= -0.02` or `> -0.02`: got `4 recovery + 13 compact hold` -> so `dmg > -0.02` triggered 40% quota!).
- **r5 (r4 on slice 5):** `51.5% / -1.71` (`simple 42/+0.5, code 62/-1.2, math 60/-5.2, ifeval 41/-0.6, no-fit 49/-1.5, mult 42/+1.7`).
  - Both held parts (`mbpp` and `svamp`) shortened FURTHER: `mbpp 55% -> 62%` (`-1.2`), `svamp 55% -> 60%` (`-5.2`)!
  - 2-slice brake mean (r4,r5): `mbpp 58/-2.8`, `svamp 57/-3.6` -> **HELD (`mbpp`, `svamp`)**.
  - Round 5 training pairs: `mbpp` (slice dmg `-1.2 > -2.0`, undamaged branch!): `4 recovery + 18 compact hold`; `svamp` (slice dmg `-5.2 <= -2.0`, damaged branch!): `13 recovery + 14 compact hold` (only 13 recovery candidates existed on svamp!). Total pairs = `194` (15 steps = 240 updates, wrapping 46 pairs!).
- **r6 (r5 on slice 6):** `52.0% / -1.91` (`simple 40/-0.1, code 64/-4.6, math 53/-3.0, ifeval 47/-0.6, mult 49/+0.5, no-fit 51/+1.5`).
  - Notice: `svamp` had `13 rec + 14 compact hold` in r5 and lengthened back (`60% -> 53%`, acc `-5.2 -> -3.0`), whereas `mbpp` had `4 rec + 18 compact hold` (plus 15 loop pairs!) in r5 and shortened (`62% -> 64%`, acc `-1.2 -> -4.6`).
  - Round 6 training pairs: `mbpp` (dmg `-4.6`): `15 rec + 29 compact hold`; `svamp` (dmg `-3.0`): `18 rec + 19 compact hold`. Total pairs = `228`.
- **r7 (r6 on slice 7):** `55.6% / -1.15` (`no-fit 49/+1.7, code 68/-1.6, math 58/-3.3, ifeval 48/-0.7, mult 52/-1.1, simple 46/+1.7`).
  - `mbpp` shortened again `64% -> 68%` (`-1.6`), `svamp` `53% -> 58%` (`-3.3`).

## 3. Initial Findings on the 5 Questions
1. **Why held parts kept shortening in `od5h`:**
   - Primary direct cause in `cand[part]["hold"]`: `0.50 <= ratio <= 1.00` sorted by `(REREAD, thinking_len(r), abs(ratio - 0.75))` explicitly picks the **shortest right answer** (`thinking_len(r)` first!) down to `0.50x` of the wrong answer. Even when a wrong answer is short, `thinking_len(r)` chooses the minimum-length right trace on that question!
   - Four additional mechanisms:
     (a) **Defect (`loop`) pairs on held parts:** Before `n_rem`, `got5` takes up to `12%` of `n` (`7` pairs on `mbpp`, `6` on `svamp`) from `cand[part]["loop"]`. Look at line 365: `cand[part]["loop"].append((qid, clean[0], min(loops, key=lambda r: -r["output_tokens"])))`. And what is `clean[0]`? Look at how `rights` / `clean` is ordered or selected, and `min(loops, key=lambda r: -r["output_tokens"])` is the **longest** looping trace (`output_tokens` up to max_tokens)! Wait: `loop` pairs pair `clean[0]` against the longest runaway trace—a huge shortening gradient!
     (b) **Cross-task LoRA generalisation from unheld parts (`ifeval`, `bfcl_irrelevance`, `bfcl_multiple`, `bfcl_simple_python`)** + **15-step epoch wrap over fewer pairs (`194-211` pairs -> `29-46` wrapped pairs):** When `mbpp` and `svamp` are capped at 40% quota (`194` total pairs), `dpo.py` still runs `15 steps * 16 = 240` examples, repeating the first 46 stratified pairs (which are mostly unheld shortening pairs!).
     (c) **Candidate starvation in `recovery`:** On damaged slices, `rec_h` asks for `0.40 * n_rem` (`~20` on `mbpp`, `~18` on `svamp`), creative recovery filter `tw < 0.85 * bt` and `1.08 * tw + 10 <= tl <= min(1.05 * bt, 1.45 * tw + 50)` often finds fewer candidates (`13` on svamp r5, `8` on svamp r7, `14` on mbpp r3), and `hold_h = take(..., n_rem - len(rec_h))` backfills all missing recovery slots with `hold` pairs!
     (d) ** Why `mbpp` crashed at r3 (`47% / -4.1`) in the first place:** In `od5h`, `mbpp` was given BOTH a lower floor (`35:0.36 -> 28:0.32`), a lower ratio (`1.30 -> 1.25`), a higher quota (`52 -> 56`), AND `--shorten-second mbpp`. In rounds 1-2, `mbpp` jumped to `47%` saved by r2 (vs `39%` in `od5f`), immediately hitting `-4.1` accuracy!
