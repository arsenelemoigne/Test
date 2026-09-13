"""
STEP 5 -- scoring, same method Harvey LAB uses.

Each of the 29 criteria is graded pass/fail by an LLM judge that sees only the
deliverables and the criterion. No gold answer; the criterion text is the standard.

LAB's headline metric is all-pass (1.0 only if all 29 pass), which is too sparse
to compare arms on a handful of tasks. The metric here is criterion pass rate,
with all-pass reported alongside.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

TASK_JSON = Path(__file__).resolve().parent / "task" / "task.json"

PROMPT = """You are grading one criterion of a legal work-product evaluation.

TASK
{task}

CRITERION: {title}
STANDARD: {match}

AGENT OUTPUT
{output}

Does the agent output satisfy the standard? Judge substance, not wording or
formatting. Answer with JSON only: {{"verdict": "pass" | "fail", "reasoning": "..."}}"""


def criteria() -> list[dict]:
    return json.loads(TASK_JSON.read_text())["criteria"]


def task_description() -> str:
    d = json.loads(TASK_JSON.read_text())
    return f"{d['title']}\n\n{d['instructions']}"


def score(deliverables: dict[str, str], judge, workers: int = 8) -> dict:
    """
    deliverables: {filename: text}. judge: callable(prompt, max_tokens) -> str.

    Criteria are graded concurrently. Each judge call is independent by design
    (one criterion, no shared state), so this changes wall-clock only, not the
    result. 29 criteria x 30 runs = 870 calls; serially that is hours.
    """
    output = "\n\n".join(f"===== {k} =====\n{v}" for k, v in deliverables.items())
    task = task_description()
    crit = criteria()

    def grade_one(c: dict) -> dict:
        # 4000, not 1500: a reasoning judge spends most of its budget thinking and
        # returns an empty string if the cap is tight.
        try:
            raw = judge(
                PROMPT.format(task=task, title=c["title"],
                              match=c["match_criteria"], output=output),
                max_tokens=4000,
            ) or ""
        except Exception as e:                       # noqa: BLE001
            raw = ""
            print(f"    {c['id']} judge error: {str(e)[:110]}", flush=True)
        m = re.search(r"\{.*\}", raw, re.S)
        try:
            v = json.loads(m.group(0)) if m else None
        except json.JSONDecodeError:
            v = None
        bad = v is None
        if bad:
            v = {"verdict": "fail", "reasoning": f"JUDGE RETURNED NO JSON (len={len(raw)})"}
        return {"id": c["id"], "title": c["title"],
                "verdict": v.get("verdict", "fail"),
                "reasoning": v.get("reasoning", ""), "_bad": bad}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(grade_one, crit))

    n_unparseable = sum(1 for r in results if r.pop("_bad"))
    if n_unparseable:
        print(f"  WARNING: {n_unparseable}/{len(results)} judge replies were unparseable. "
              f"Scores are NOT trustworthy - switch WM_JUDGE to a non-reasoning model "
              f"(e.g. google/gemini-2.5-flash) and re-run.")
    n_pass = sum(1 for r in results if r["verdict"] == "pass")
    return {
        "n_criteria": len(results),
        "n_passed": n_pass,
        "criterion_pass_rate": n_pass / len(results) if results else 0.0,
        "all_pass": n_pass == len(results),
        "n_unparseable": n_unparseable,
        "results": results,
    }
