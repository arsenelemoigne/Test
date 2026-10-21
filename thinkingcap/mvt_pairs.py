"""On-policy, verified MVT cuts turned into DPO pairs.

For each correct reply of the CURRENT model (its own rollouts, sampled fresh each round):

  1. Split the thinking into units (paragraphs, or 32-token batches for code) and score the
     probe curve: p(correct answer | first t thinking tokens), as prune_traces.py does.
  2. Candidate cuts: the Marginal Value Theorem cut, one earlier (half as much thinking kept)
     and one later (halfway to the end). One rule-chosen cut and two neighbours instead of
     trying every 10%.
  3. Verify every candidate: the model answers --check-samples times from the cut thinking
     (check_vllm.py, one batch). A cut is safe only if every answer is right.
  4. Pairs, as in on-policy truncation oracles (Lightning McQwen3, ThoughtFold):
       shorten  the shortest safe cut (+ its own answer)  beats  the full reply
       guard    the shortest safe cut                     beats  a shorter unsafe cut + its wrong answer
       hard     no safe cut: the full reply               beats  the earliest cut + its wrong answer
     Maths (--strict-sources) only accepts cuts keeping at least --strict-keep of the thinking.

    python mvt_pairs.py --rollouts runs/r1/probe/rollouts.jsonl --adapter adapters/start --out runs/r1/pairs
    python dpo.py --pairs runs/r1/pairs/pairs.jsonl --init-adapter adapters/start --out adapters/r1
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

from probe_curve import read_jsonl, write_jsonl
from prune_traces import Pruner, mvt_index, split_units, step_positions, thinking_text


def main():
    args = parse_arguments()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pruner = Pruner(SimpleNamespace(model=args.model, adapter=args.adapter, dtype=args.dtype, checker="vllm",
                                    check_samples=args.check_samples, answer_tokens=512,
                                    batch_tokens=args.batch_tokens))
    units = dict(spec.split("=") for spec in args.units)
    strict = set(args.strict_sources.split(",")) if args.strict_sources else set()

    by_question = defaultdict(list)
    for r in read_jsonl(Path(args.rollouts)):
        by_question[r["id"]].append(r)

    plans, jobs = [], []
    for number, (qid, group) in enumerate(by_question.items(), 1):
        correct = [r for r in group if r["correct"] and r["closed_thinking"] and not r["truncated"]]
        for rollout in correct[:args.per_question]:
            unit = units.get(rollout["source"], "paragraph")
            steps = split_units(pruner.decode_thinking(rollout["thinking_ids"]), unit, pruner.tokenizer)
            if len(steps) < 2:
                continue
            curve = pruner.p_gold(rollout, [steps[:n] for n in range(len(steps) + 1)],
                                  pruner.suffixes(rollout, group))
            positions = step_positions([len(pruner.encode(s)) for s in steps])
            mvt = max(1, mvt_index(curve, positions, args.travel))
            cuts = sorted({max(1, mvt // 2), mvt, mvt + max(1, (len(steps) - mvt) // 2)} - {len(steps)})
            if rollout["source"] in strict:
                floor = args.strict_keep * positions[-1]
                cuts = [c for c in cuts if positions[c] >= floor]
            entry = {"rollout": rollout, "steps": steps, "positions": positions, "mvt": mvt, "cuts": {}}
            for c in cuts:
                entry["cuts"][c] = len(jobs)
                jobs.append((rollout, steps[:c], "cut"))
            plans.append(entry)
        if number % 50 == 0 or number == len(by_question):
            print(f"probed {number}/{len(by_question)} questions, {len(jobs)} cuts to verify", flush=True)

    verdicts = verify(pruner, jobs, out)
    pairs, kinds = [], Counter()
    for entry in plans:
        rollout, steps = entry["rollout"], entry["steps"]
        full = rollout["output_ids"]
        results = {c: (verdicts[j], pruner.encode(thinking_text(steps[:c]))) for c, j in entry["cuts"].items()}
        safe = [c for c, (v, _) in sorted(results.items()) if v["answer_ids"]]
        base = {"id": rollout["id"], "source": rollout["source"], "prompt_ids": rollout["prompt_ids"]}
        if safe:
            best = safe[0]
            verdict, thinking = results[best]
            chosen = thinking + verdict["answer_ids"]
            if len(chosen) < len(full):
                pairs.append(base | {"kind": "shorten", "chosen_ids": chosen, "rejected_ids": full})
                kinds["shorten"] += 1
            early = [c for c, (v, _) in sorted(results.items()) if c < best and v["wrong_ids"]]
            if early:
                v, t = results[early[-1]]
                pairs.append(base | {"kind": "guard", "chosen_ids": chosen, "rejected_ids": t + v["wrong_ids"]})
                kinds["guard"] += 1
        else:
            wrong = [c for c, (v, _) in sorted(results.items()) if v["wrong_ids"]]
            if wrong:
                v, t = results[wrong[0]]
                pairs.append(base | {"kind": "hard", "chosen_ids": full, "rejected_ids": t + v["wrong_ids"]})
                kinds["hard"] += 1
    write_jsonl(out / "pairs.jsonl", pairs)
    report(pairs, plans, verdicts, out)


def verify(pruner, jobs, out):
    """Every candidate cut answered --check-samples times in one vLLM batch."""
    import subprocess
    import sys

    from prune_traces import GRADE_FIELDS

    inputs, results = out / "cuts.jsonl", out / "verdicts.jsonl"
    write_jsonl(inputs, [{"key": i, "prompt_ids": r["prompt_ids"], "thinking_ids": pruner.encode(thinking_text(steps)),
                          "max_tokens": pruner.answer_cap(r), "row": {f: r[f] for f in GRADE_FIELDS if f in r}}
                         for i, (r, steps, _) in enumerate(jobs)])
    command = [sys.executable, "check_vllm.py", "--model", pruner.args.model, "--inputs", str(inputs),
               "--out", str(results), "--samples", str(pruner.args.check_samples), "--dtype", pruner.args.dtype]
    if pruner.args.adapter:
        command += ["--adapter", pruner.args.adapter]
    subprocess.run(command, check=True)
    by_key = {v["key"]: v for v in read_jsonl(results)}
    return [by_key[i] for i in range(len(jobs))]


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--adapter", help="the current model's LoRA: it scores, re-answers and wrote the rollouts")
    parser.add_argument("--rollouts", required=True, help="the current model's own rollouts (probe_curve.py)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--travel", type=float, default=1024)
    parser.add_argument("--units", nargs="*", default=["mbpp_train=tokens32"],
                        help="source=unit; paragraph for the rest")
    parser.add_argument("--check-samples", type=int, default=4, help="re-answers per cut; all must be right")
    parser.add_argument("--per-question", type=int, default=2, help="correct rollouts used per question")
    parser.add_argument("--strict-sources", default="gsm8k_all", help="sources whose cuts keep --strict-keep")
    parser.add_argument("--strict-keep", type=float, default=0.5, help="share of the thinking a strict cut keeps")
    parser.add_argument("--batch-tokens", type=int, default=32768)
    parser.add_argument("--dtype", default="bfloat16")
    return parser.parse_args()


def report(pairs, plans, verdicts, out):
    lines = [f"{'source':<16}{'replies':>8}{'shorten':>9}{'guard':>7}{'hard':>6}{'full len':>10}{'chosen len':>12}{'saved':>7}"]
    sources = sorted({p["rollout"]["source"] for p in plans})
    for source in sources + ["ALL"]:
        mine = [p for p in pairs if source in ("ALL", p["source"])]
        short = [p for p in mine if p["kind"] == "shorten"]
        replies = sum(1 for e in plans if source in ("ALL", e["rollout"]["source"]))
        full = sum(len(p["rejected_ids"]) for p in short) / max(1, len(short))
        chosen = sum(len(p["chosen_ids"]) for p in short) / max(1, len(short))
        kinds = Counter(p["kind"] for p in mine)
        lines.append(f"{source:<16}{replies:>8}{kinds['shorten']:>9}{kinds['guard']:>7}{kinds['hard']:>6}"
                     f"{full:>10.0f}{chosen:>12.0f}{1 - chosen / full if full else float('nan'):>7.0%}")
    safe = sum(v["answer_ids"] is not None for v in verdicts)
    lines += ["", f"cuts verified: {len(verdicts)}, safe: {safe} ({safe / max(1, len(verdicts)):.0%})",
              "shorten: shortest safe cut over the full reply; guard: that cut over a shorter unsafe cut's wrong "
              "answer; hard: full reply over an early wrong cut; lengths over shorten pairs"]
    text = "\n".join(lines)
    print(text)
    (out / "summary.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
