"""Benchmark questions in the layout probe_curve.py / label_cuts.py read, to MEASURE the stopping judge on real
benchmark questions (never to train on them).

Each row keeps the evaluation's own prompt and grading fields; "source" is the practice type the judge knows for that
benchmark (SVAMP -> gsm_plus, MBPP+ -> kodcode, BFCL -> toolace, IFEval -> ifeval_like) so per-type thresholds carry
over, and "dataset" keeps the real name. MBPP+ is graded here with the original MBPP asserts (MBPP+'s extra tests run
only in the evaluation's sandbox): a close stand-in for labelling cuts.

    python bench_rows.py --questions questions_40 --out data/bench40.jsonl
"""

import argparse
import json
import re
from pathlib import Path

SOURCE = {"svamp": "gsm_plus", "mbpp": "kodcode", "bfcl_simple_python": "toolace", "bfcl_multiple": "toolace",
          "bfcl_irrelevance": "toolace", "ifeval": "ifeval_like"}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--questions", default="questions_40")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rows = [json.loads(l) for l in (Path(args.questions) / "questions.jsonl").read_text().splitlines() if l.strip()]
    mbpp = {}
    if any(r["dataset"] == "mbpp" for r in rows):
        from datasets import load_dataset
        for split in ("train", "validation", "prompt", "test"):
            for t in load_dataset("google-research-datasets/mbpp", "full", split=split):
                mbpp[t["task_id"]] = t
    out = []
    for r in rows:
        base = {"id": r["id"], "dataset": r["dataset"], "source": SOURCE[r["dataset"]], "prompt": r["messages"], "answer": ""}
        if r["grader"] == "numeric":
            out.append(base | {"task": "math", "answer": r["answer"]})
        elif r["grader"] == "bfcl":
            out.append(base | {"task": "tool", "grader": "bfcl", "category": r["category"], "function": r["function"],
                               "reference": r["reference"]})
        elif r["grader"] == "ifeval":
            out.append(base | {"task": "instructions", "grader": "ifeval", "instruction_id_list": r["instruction_id_list"],
                               "kwargs": r["kwargs"], "original_prompt": r["original_prompt"]})
        elif r["grader"] == "mbpp":
            task = mbpp.get(int(re.sub(r"\D", "", r["id"])))
            if task is None:
                continue
            out.append(base | {"task": "code", "grader": "code", "entry_point": r["entry_point"], "tests": task["test_list"],
                               "setup": task["test_setup_code"]})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in out))
    counts = {}
    for r in out:
        counts[r["dataset"]] = counts.get(r["dataset"], 0) + 1
    print(f"wrote {len(out)} benchmark questions to {args.out}: {counts}", flush=True)


if __name__ == "__main__":
    main()
