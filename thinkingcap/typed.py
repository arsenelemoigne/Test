"""Graders for every kind of training question: math, multiple choice, code, tool calls, instructions.

Math and multiple choice keep utils/train/rewards.py. The other three reuse the evaluation's own
checkers from the questions directory (EvalPlus's sanitizer, BFCL's AST checker, Google's IFEval
verifiers), so a training trace counts as correct by the same rule the benchmark uses:

    from typed import is_correct
    is_correct(question_row, full_reply_text)

A row carries "task" and, for the new kinds, "grader" plus that grader's fields (see
prepare_types.py).

Code, tool and instruction answers have no short gold string to put a probability on, so the
probe reads the model's own correct answer instead: OWN_ANSWER tasks score p(that answer | the
thinking so far), as a per-token geometric mean so long answers stay comparable.
"""

import os
import re
import resource
import subprocess
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

OWN_ANSWER = {"code", "tool", "instructions"}
QUESTIONS = Path(os.environ.get("TYPED_QUESTIONS", "questions"))


def full_text(tokenizer, ids):
    """Decode a reply keeping <think>...</think> (graders split on it) but not end-of-turn tokens."""
    text = tokenizer.decode(ids, skip_special_tokens=False)
    for token in ("<|im_end|>", "<|endoftext|>"):
        text = text.replace(token, "")
    return text


def response_of(text):
    """The reply after the thinking, as the evaluation reads it."""
    return text.split("</think>")[-1].strip() if "</think>" in text else text.strip()


def is_correct(row, text, truncated=False):
    if truncated:
        return False
    task = row["task"]
    if task in ("math", "multiple_choice"):
        from utils.train.rewards import is_task_correct
        return is_task_correct(text, row["answer"], task)
    reply = response_of(text)
    if task == "code":
        return code_passes(row, reply)
    if task == "tool":
        from utils.eval.graders import grade_call
        checker, language = bfcl_checker()
        try:
            return bool(grade_call(row, reply, checker, language)[0])
        except Exception:  # an unusual schema the checker cannot type: not a usable trace
            return False
    if task == "instructions":
        return bool(ifeval_checker()(row, reply)[0])
    raise ValueError(f"unknown task {task!r}")


@lru_cache(maxsize=1)
def bfcl_checker():
    from utils.eval.benchmarks import paths
    from utils.eval.graders import load_bfcl
    return load_bfcl(paths(QUESTIONS)["bfcl"])


@lru_cache(maxsize=1)
def ifeval_checker():
    from utils.eval.benchmarks import paths
    from utils.eval.graders import load_ifeval
    return load_ifeval(paths(QUESTIONS)["ifeval"])


@lru_cache(maxsize=1)
def sanitizer():
    from utils.eval.benchmarks import paths
    from utils.eval.graders import load_sanitize
    return load_sanitize(paths(QUESTIONS)["evalplus"])


def code_passes(row, reply, timeout=10):
    """Run the sanitized solution against the task's asserts in a child process with limits."""
    try:
        code = sanitizer()(reply, row["entry_point"])
    except Exception:
        return False
    if not code.strip():
        return False
    program = "\n".join([row.get("setup") or "", code, *row["tests"]])
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "solution.py"
        path.write_text(program)
        try:
            result = subprocess.run([sys.executable, "-I", str(path)], cwd=folder, capture_output=True,
                                    timeout=timeout, preexec_fn=limit_resources)
        except subprocess.TimeoutExpired:
            return False
    return result.returncode == 0


def limit_resources():
    resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
    resource.setrlimit(resource.RLIMIT_CPU, (10, 10))


ENTRY = re.compile(r"assert\s+(?:not\s+)?\(?\s*(?:set\(|math\.isclose\()?\s*([A-Za-z_]\w*)\s*\(")


def entry_point(test):
    """The function an MBPP assert calls: 'assert sum_list([1, 2]) == 3' -> 'sum_list'."""
    match = ENTRY.search(test)
    return match.group(1) if match else None


BFCL_TYPES = {"object": "dict", "number": "float", "integer": "integer", "string": "string",
              "boolean": "boolean", "array": "array", "dict": "dict", "float": "float"}


def bfcl_schema(schema):
    """A JSON-schema parameter block -> BFCL's dialect ("object" -> "dict", "number" -> "float")."""
    if not isinstance(schema, dict):
        return schema
    out = dict(schema)
    if "type" in out:
        kind = out["type"][0] if isinstance(out["type"], list) else out["type"]
        if kind not in BFCL_TYPES:
            raise ValueError(f"unsupported type {kind!r}")
        out["type"] = BFCL_TYPES[kind]
    if "properties" in out:
        out["properties"] = {name: bfcl_schema(value) for name, value in out["properties"].items()}
    if "items" in out:
        out["items"] = bfcl_schema(out["items"])
    return out


def bfcl_function(tool):
    """A Hermes/OpenAI tool -> a BFCL function entry."""
    function = tool.get("function", tool)
    parameters = bfcl_schema(function.get("parameters") or {"type": "object", "properties": {}})
    parameters.setdefault("properties", {})
    parameters.setdefault("required", [])
    parameters["type"] = "dict"
    return {"name": function["name"], "description": function.get("description", ""), "parameters": parameters}


def bfcl_reference(function, arguments):
    """A gold call -> BFCL's possible-answer form: every given argument must match, others may be omitted."""
    reference = {name: [value] for name, value in arguments.items()}
    for name in function["parameters"].get("properties", {}):
        if name not in reference:
            reference[name] = [""]
    return [{function["name"]: reference}]


def main():
    print(entry_point(sys.argv[1]) if len(sys.argv) > 1 else __doc__)


if __name__ == "__main__":
    main()
