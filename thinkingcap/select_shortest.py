"""Shortest-of-N self-distillation data: for each question, the shortest of the model's own correct replies.

Nothing is cut or edited: every target is a reply the model actually wrote and that the
evaluation's graders accept, so training only shifts the model toward its own brief, correct
behaviour. With N samples per question the shortest correct reply is typically well under the
mean length; with N = 2 (our "whole" runs) it barely differs.

    python select_shortest.py --rollouts runs/bon/probe/rollouts.jsonl --out runs/bon/select \\
        --easy-tokens gsm8k_all=700

Writes sft.jsonl (sft.py's format) and summary.txt: per source, how many questions have a
correct reply, the mean length of all replies, of the correct ones and of the kept ones.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--rollouts", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--easy-tokens", nargs="*", default=[],
                        help="source=N: keep that source's question only if its shortest correct reply is at most N "
                             "tokens (e.g. GSM8K down to SVAMP's difficulty)")
    args = parser.parse_args()
    limits = {s: int(n) for s, n in (spec.split("=") for spec in args.easy_tokens)}

    by_question = defaultdict(list)
    for line in Path(args.rollouts).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            by_question[r["id"]].append(r)

    targets, stats = [], defaultdict(lambda: defaultdict(list))
    for qid, group in by_question.items():
        source = group[0]["source"]
        s = stats[source]
        s["questions"].append(1)
        s["all"] += [r["output_tokens"] for r in group]
        s["capped"] += [r["truncated"] for r in group]
        correct = [r for r in group if r["correct"] and not r["truncated"] and r["closed_thinking"]]
        if not correct:
            continue
        s["solved"].append(1)
        s["correct"] += [r["output_tokens"] for r in correct]
        best = min(correct, key=lambda r: r["output_tokens"])
        if source in limits and best["output_tokens"] > limits[source]:
            continue
        s["kept"].append(best["output_tokens"])
        s["kept_mean_of_question"].append(sum(r["output_tokens"] for r in group) / len(group))
        targets.append({"id": qid, "source": source, "task": best["task"], "answer": best["answer"],
                        "kind": "shortest", "steps": None, "prompt_ids": best["prompt_ids"],
                        "completion_ids": best["output_ids"], "original_tokens": best["output_tokens"],
                        "thinking_tokens": len(best["thinking_ids"]), "samples": len(group),
                        "correct_samples": len(correct)})

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "sft.jsonl").write_text("".join(json.dumps(t) + "\n" for t in targets))
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    lines = [f"{'source':<16}{'questions':>10}{'solved':>8}{'kept':>6}{'capped':>8}{'all':>8}{'correct':>9}"
             f"{'kept len':>10}{'same q mean':>13}{'saved':>7}"]
    for source, s in sorted(stats.items()):
        saved = 1 - mean(s["kept"]) / mean(s["kept_mean_of_question"]) if s["kept"] else float("nan")
        lines.append(f"{source:<16}{len(s['questions']):>10}{len(s['solved']):>8}{len(s['kept']):>6}"
                     f"{mean(s['capped']):>8.0%}{mean(s['all']):>8.0f}{mean(s['correct']):>9.0f}"
                     f"{mean(s['kept']):>10.0f}{mean(s['kept_mean_of_question']):>13.0f}{saved:>7.0%}")
    lines += ["", "all / correct: mean length of every reply / of the correct ones; kept len: the shortest correct "
              "reply per kept question; same q mean: those questions' mean reply length; saved: kept vs that mean"]
    text = "\n".join(lines)
    print(text)
    (out / "summary.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
