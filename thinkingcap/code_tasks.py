"""Coding questions with unit tests, to train length on the kind of task MBPP+ evaluates.

Training only on maths and multiple choice did not shorten code replies at all (MBPP+ even got
longer), so this adds HumanEval (164 problems, not one of the evaluation benchmarks) in the
training format: the same "write a complete solution in one code block" instruction MBPP+ is
asked with, and a reward that runs the problem's own tests on the extracted code.

    python code_tasks.py --out data/code.jsonl          # writes rows with task "code"
    python rl_train.py --data data/train.jsonl,data/code.jsonl --sources gsm8k,...,humaneval ...

The tests run model-written code in a subprocess with CPU, memory and time limits and nothing
else (no filesystem or network isolation), as the evaluation itself does on Colab when
bubblewrap is unavailable: only on a disposable VM.
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

INSTRUCTION = "Write a complete Python solution to the following task. Return the solution in one Python code block."
CODE_BLOCK = re.compile(r"```(?:python|py)?[ \t]*\r?\n(.*?)```", re.S | re.I)
TIMEOUT = 10
MEMORY = 1024 ** 3


def rows():
    from datasets import load_dataset
    for task in load_dataset("openai/openai_humaneval", split="test"):
        question = (f"{INSTRUCTION}\n\n{task['prompt'].rstrip()}\n\n"
                    f"The tests call `{task['entry_point']}`; keep that name.")
        yield {"prompt": [{"role": "user", "content": question}],
               "answer": json.dumps({"prompt": task["prompt"], "test": task["test"],
                                     "entry_point": task["entry_point"]}),
               "task": "code", "source": "humaneval"}


def extract_code(reply):
    """The last fenced block after the thinking, or None."""
    blocks = CODE_BLOCK.findall(reply.split("</think>")[-1])
    return blocks[-1] if blocks else None


def limits():
    import resource
    for limit, value in ((resource.RLIMIT_AS, MEMORY), (resource.RLIMIT_CPU, TIMEOUT)):
        try:
            resource.setrlimit(limit, (value, value))
        except (ValueError, OSError):  # macOS refuses RLIMIT_AS; Linux (Colab) applies both
            pass


def is_code_correct(reply, answer):
    """Run the problem's tests against the reply's code; True only if they all pass in time."""
    code = extract_code(reply)
    if not code:
        return False
    spec = json.loads(answer)
    # the prompt's imports (typing etc.) first, so a solution that relies on them still runs
    imports = "\n".join(line for line in spec["prompt"].splitlines() if line.startswith(("import ", "from ")))
    program = f"{imports}\n{code}\n\n{spec['test']}\n\ncheck({spec['entry_point']})\n"
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "solution.py"
        path.write_text(program)
        try:
            done = subprocess.run([sys.executable, str(path)], cwd=directory, capture_output=True,
                                  timeout=TIMEOUT, preexec_fn=limits)
        except subprocess.TimeoutExpired:
            return False
    return done.returncode == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="data/code.jsonl")
    args = parser.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    written = list(rows())
    out.write_text("".join(json.dumps(r) + "\n" for r in written))
    print(f"{len(written)} coding questions -> {out}")


if __name__ == "__main__":
    main()
