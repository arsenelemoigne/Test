"""
STEP 3 -- the experimental conditions.

Only the INPUT changes between arms. The task, the required JSON output, the
renderer and the judge are identical everywhere.

  A0  raw          all task documents, verbatim (~63k tokens)
  A2  prose        frontier-written prose analysis: same facts as A4, flowing
                   prose, no fields, no IDs, length-matched        <-- CONTROL
  A4  worldmodel   the typed Issue list                            <-- TREATMENT
  A4G worldmodel+  A4 plus the gravity table (tier per issue, read
                   from the client's authority memo)                <-- TREATMENT 2
  A6  both         A4 alongside the raw documents (the deployable shape)

A4G is separated from A4 deliberately. Gravity is extra INFORMATION, not just
extra form -- it tells the model which issues are walk-aways. Folding it into A4
would confound "structure helps" with "being told what matters helps", which are
different claims. A4 vs A4G isolates the second one.

A2 is the condition that makes the experiment mean anything. Without it, a win
for A4 over A0 is explained just as well by "a frontier model read the contract
for you" or "the context got shorter" as by "structure helps". A2 holds the
information, the author and the length constant so that only FORM varies.
"""

from __future__ import annotations

import os

from pathlib import Path

from . import taskctx
from .abstraction import CONFIDENTIAL as _CONF, FACTS as _FACTS


def _issues():
    return taskctx.issues()


def _facts():
    """FACTS and _confidential() were written for the original DSA. A ported task
    has neither, and carrying them over would leak another deal's numbers into
    the prompt."""
    return _FACTS if taskctx.is_default() else {}


def _confidential():
    return _CONF if taskctx.is_default() else []

def TASK_DIR():
    return taskctx.task_dir()

OUTPUT_SPEC = """
Return ONLY a JSON array. One object per issue, using these exact issue ids:
{ids}

Each object:
  {{"issue_id": "...",
    "disposition": "ACCEPT" | "REJECT" | "MODIFY",
    "counter": "the operative position Carden takes on this issue, stated concretely -
                include specific numbers, durations and named jurisdictions where relevant",
    "rationale": "why, in language suitable to send to opposing counsel"}}

Think first, then emit the JSON last. Do not wrap it in prose.
IMPORTANT: the rationale is sent to the counterparty. Never disclose internal
negotiation authority: {confidential}
"""


def _ids() -> str:
    return ", ".join(f"{i.id} ({i.name})" for i in _issues())


def output_spec() -> str:
    return OUTPUT_SPEC.format(ids=_ids(), confidential="; ".join(_confidential()))


def TASK() -> str:
    """What the agent is being asked to do.

    The original DSA statement names the parties and dates from FACTS. A ported
    task has no FACTS, so the instruction comes from its own task.json - which
    is also what the judge grades against, so the two cannot drift apart.
    """
    if taskctx.is_default():
        f = _facts()
        return (f"You act for {f['us']} in negotiating a Data Sharing Agreement with "
                f"{f['them']}. Luminos returned a first markup on {f['markup_received']} "
                f"against Carden's initial draft of {f['initial_draft_date']}.\n\n"
                "Decide Carden's counter-turn position on every substantive change: "
                "accept it, reject it and revert, or modify it with a counter-proposal. "
                "Where you reject or modify, state the specific position Carden proposes.")
    import json
    tj = taskctx.task_dir() / "task.json"
    instr = json.loads(tj.read_text()).get("instructions", "") if tj.exists() else ""
    return (instr.strip() + "\n\n" if instr else "") + (
        "Decide your client's position on every substantive change the counterparty "
        "made: accept it, reject it and revert, or modify it with a counter-proposal. "
        "Where you reject or modify, state the specific position your client proposes.")


# --- A0 -------------------------------------------------------------------

def raw() -> str:
    parts = []
    for f in sorted(TASK_DIR().glob("*.txt")):
        parts.append(f"===== {f.name} =====\n{f.read_text()}")
    return "\n\n".join(parts)


# --- A4 -------------------------------------------------------------------

def worldmodel() -> str:
    f = _facts()
    L = ["NEGOTIATION STATE", "=" * 70, ""]
    if f:
        L += [f"We are {f['us']}. Counterparty: {f['them']}.",
              f"Markup received {f['markup_received']} against our draft of "
              f"{f['initial_draft_date']}."]
    L += [f"{len(_issues())} substantive issues are open. Each needs a disposition.", ""]
    for i in _issues():
        L += [
            f"{i.id}  {i.name}   [Section {i.section}]",
            f"    our draft   : {i.ours}",
            f"    they propose: {i.theirs}",
            f'    their words : "{i.theirs_quote}"',
            f"    authority   : {i.authority}",
        ]
        if i.hard_limit:
            L.append(f"    hard limit  : {i.hard_limit}")
        L.append("")
    L += ["INTERNAL ONLY - must not appear in anything sent to Luminos:"]
    L += [f"  - {c}" for c in _confidential()]
    return "\n".join(L)


# --- A2 (the control) -----------------------------------------------------

PROSE_TWIN_PROMPT = """Rewrite the negotiation state below as flowing narrative prose.

Rules:
- Preserve every fact: every position, every number, every authority limit, every
  quoted phrase, and the confidentiality warning.
- Remove ALL structure: no issue IDs, no field labels, no bullets, no headings,
  no tables. Continuous paragraphs only.
- Do not add analysis, do not prioritise, do not recommend. Same information,
  different form.
- Aim for a similar length to the input.

{wm}"""


def prose_twin(cache: Path, llm) -> str:
    """Generated once by the frontier model, then cached and reused for every run."""
    if cache.exists():
        return cache.read_text()
    text = llm(PROSE_TWIN_PROMPT.format(wm=worldmodel()), max_tokens=8000)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(text)
    return text


# --- assembly -------------------------------------------------------------

def build(condition: str, prose: str | None = None) -> str:
    if condition == "A0":
        body = f"SOURCE DOCUMENTS\n\n{raw()}"
    elif condition == "A2":
        if prose is None:
            raise ValueError("A2 needs the prose twin")
        body = f"NEGOTIATION BACKGROUND\n\n{prose}"
    elif condition == "A4":
        body = worldmodel()
    elif condition == "A4G":
        from . import gravity
        body = (f"{worldmodel()}\n\n"
                f"{gravity.report({i.id: i.name for i in _issues()})}")
    elif condition in ("A5", "A5N"):
        # Same prompt as A4. The difference is not what the model is told, it is
        # that the code evaluates the answer and the model gets to revise.
        body = worldmodel()
    elif condition == "A6":
        body = f"{worldmodel()}\n\n\nSOURCE DOCUMENTS\n\n{raw()}"
    else:
        raise ValueError(condition)
    return f"{TASK()}\n\n{body}\n\n{output_spec()}"


_ALL = ["A0", "A2", "A4", "A4G", "A5", "A5N", "A6"]

# Narrow the run without editing code:
#     export WM_CONDITIONS=A2,A4,A4G
# A2 vs A4 is the primary comparison and both prompts are tiny. A0 and A6 carry
# the whole 63k-token contract and cost roughly forty times as much per run, and
# A0 vs A4 is confounded by preprocessing and context length anyway - it was
# never going to be a result. Drop them first when money is short.
CONDITIONS = [c.strip() for c in os.environ.get("WM_CONDITIONS", "").split(",")
              if c.strip() in _ALL] or _ALL
