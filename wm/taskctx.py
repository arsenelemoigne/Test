"""
Which task are we running, and what are its issues?

    export WM_TASK_DIR=~/Test/wm/tasks/license-s04

Defaults to wm/task (the DSA this project started on), whose issues are the
hand-written list in abstraction.py. Any other directory must carry an
issues.json produced by `python -m wm.run issues`.
"""

from __future__ import annotations

import os
from pathlib import Path

_DEFAULT = Path(__file__).resolve().parent / "task"


def task_dir() -> Path:
    v = os.environ.get("WM_TASK_DIR", "").strip()
    return Path(v).expanduser().resolve() if v else _DEFAULT


def is_default() -> bool:
    return task_dir() == _DEFAULT


def issues_file() -> Path:
    return task_dir() / "issues.json"


def issues():
    """The hand-written list for the original task, the generated one otherwise."""
    if is_default() and not issues_file().exists():
        from .abstraction import ISSUES
        return ISSUES
    f = issues_file()
    if not f.exists():
        raise RuntimeError(
            f"{f} does not exist.\n"
            f"Run `python -m wm.run issues` first - it reads the client's memo "
            f"and the counterparty markup\nin {task_dir()} and writes the issue "
            f"list and its hard limits.")
    from .issuegen import load
    return [g.as_issue() for g in load(f.read_text())]


def gen_issues():
    """The generated issues WITH their machine-checkable limits, or None."""
    f = issues_file()
    if not f.exists():
        return None
    from .issuegen import load
    return load(f.read_text())


def check(decisions):
    """Authority check: generated limits where we have them, the hand-written
    rules for the original task."""
    g = gen_issues()
    if g is not None:
        from .issuegen import check_generic
        return check_generic(decisions, g)
    from .abstraction import check as hand_check
    return hand_check(decisions)


def issues_by_id():
    return {i.id: i for i in issues()}


def runs_dir():
    """Where this task's runs live.

    The original task keeps wm/runs so its existing runs stay put; every other
    task gets its own subdirectory. Sharing one directory meant `report` listed
    runs from two different CONTRACTS in one table, and `gradeall` graded them
    together.
    """
    base = Path(__file__).resolve().parent / "runs"
    return base if is_default() else base / task_dir().name
