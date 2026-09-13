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

from pathlib import Path

from .abstraction import CONFIDENTIAL, FACTS, ISSUES

TASK_DIR = Path(__file__).resolve().parent / "task"

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
    return ", ".join(f"{i.id} ({i.name})" for i in ISSUES)


def output_spec() -> str:
    return OUTPUT_SPEC.format(ids=_ids(), confidential="; ".join(CONFIDENTIAL))


TASK = (
    f"You act for {FACTS['us']} in negotiating a Data Sharing Agreement with "
    f"{FACTS['them']}. Luminos returned a first markup on {FACTS['markup_received']} "
    f"against Carden's initial draft of {FACTS['initial_draft_date']}.\n\n"
    "Decide Carden's counter-turn position on every substantive change: accept it, "
    "reject it and revert, or modify it with a counter-proposal. Where you reject or "
    "modify, state the specific position Carden proposes."
)


# --- A0 -------------------------------------------------------------------

def raw() -> str:
    parts = []
    for f in sorted(TASK_DIR.glob("*.txt")):
        parts.append(f"===== {f.name} =====\n{f.read_text()}")
    return "\n\n".join(parts)


# --- A4 -------------------------------------------------------------------

def worldmodel() -> str:
    L = ["NEGOTIATION STATE", "=" * 70, "",
         f"We are {FACTS['us']}. Counterparty: {FACTS['them']}.",
         f"Luminos markup received {FACTS['markup_received']} against our draft of "
         f"{FACTS['initial_draft_date']}.",
         f"{len(ISSUES)} substantive issues are open. Each needs a disposition.", ""]
    for i in ISSUES:
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
    L += [f"  - {c}" for c in CONFIDENTIAL]
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
    text = llm(PROSE_TWIN_PROMPT.format(wm=worldmodel()), max_tokens=16000)
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
        from .abstraction import ISSUES
        body = (f"{worldmodel()}\n\n"
                f"{gravity.report({i.id: i.name for i in ISSUES})}")
    elif condition == "A6":
        body = f"{worldmodel()}\n\n\nSOURCE DOCUMENTS\n\n{raw()}"
    else:
        raise ValueError(condition)
    return f"{TASK}\n\n{body}\n\n{output_spec()}"


CONDITIONS = ["A0", "A2", "A4", "A4G", "A6"]
