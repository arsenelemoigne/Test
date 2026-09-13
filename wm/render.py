"""
STEP 4 -- render decisions back to text.

EVERY condition goes through this same function. The model's job, in all arms,
is to produce the JSON decision list; the text the judge reads is generated
deterministically from that JSON.

Why this matters: it removes format from the experiment. LLM judges reward
enumerated, explicit, longer answers, and a world-model arm would naturally
produce exactly that. If the arms differed in how text is produced, a win could
just be the judge liking the layout. Here the only thing that varies between
arms is what the model was shown.
"""

from __future__ import annotations

from . import taskctx


def _nous() -> str:
    """Notre client. "Carden" etait ecrit en dur et apparaissait dans le redline
    de toute autre tache, ce qui nomme une partie etrangere au contrat."""
    import json
    if taskctx.is_default():
        return FACTS["us"]
    f = taskctx.task_dir() / "_parties.json"
    if f.exists():
        d = json.loads(f.read_text())
        return d.get("us_org") or "our client"
    return "our client"
from .abstraction import FACTS, Decision, Disposition

VERB = {
    Disposition.ACCEPT: "Accepted",
    Disposition.REJECT: "Rejected",
    Disposition.MODIFY: "Modified",
}


def redline(decisions: list[Decision]) -> str:
    """counter-turn-redline-dsa.docx"""
    L = [
        "DATA SHARING AGREEMENT - CARDEN COUNTER-TURN REDLINE",
        (f"{FACTS['us']} and {FACTS['them']}" if taskctx.is_default()
         else "the parties to the agreement"),
        (f"Counter-turn against the first markup received {FACTS['markup_received']}, "
         f"marked against our initial draft of {FACTS['initial_draft_date']}."
         if taskctx.is_default()
         else "Counter-turn against the counterparty's first markup."),
        "Prepared in Microsoft Word Track Changes.",
        "",
        "=" * 76,
        "DISPOSITION OF EACH TRACKED CHANGE",
        "=" * 76,
        "",
    ]
    for d in decisions:
        iss = taskctx.issues_by_id().get(d.issue_id)
        if iss is None:
            continue
        L.append(f"Section {iss.section} - {iss.name}")
        L.append(f"  Our initial draft: {iss.ours}")
        L.append(f"  Luminos proposed:     {iss.theirs}")
        L.append(f"  DISPOSITION:          {VERB[d.disposition].upper()}")
        L.append(f"  Our counter-position (operative language for this turn):")
        for line in _wrap(d.counter, 70):
            L.append(f"      {line}")
        L.append("")
    return "\n".join(L)


def _partie(cote: str) -> str:
    """Le nom reel du conseil, ecrit au pack depuis l'e-mail de renvoi."""
    import json
    f = taskctx.task_dir() / "_parties.json"
    if not f.exists():
        return "Counsel to the counterparty" if cote == "them" else "Counsel to our client"
    d = json.loads(f.read_text())
    nom, org = d.get(cote, ""), d.get(f"{cote}_org", "")
    if not nom:
        return "Counsel to the counterparty" if cote == "them" else "Counsel to our client"
    return f"{nom}, {org}" if org else nom


def _opening() -> str:
    """First paragraph of the cover note. Names the deal only where we have one."""
    if taskctx.is_default():
        return (f"Thank you for the markup received {FACTS['markup_received']}. Attached "
                f"is Carden's counter-turn redline, prepared in Microsoft Word Track "
                f"Changes against the Carden initial draft of "
                f"{FACTS['initial_draft_date']}. Our position on each substantive change "
                f"is set out below.")
    return ("Thank you for the markup. Attached is our counter-turn redline, prepared "
            "in Microsoft Word Track Changes against our initial draft. Our position "
            "on each substantive change is set out below.")


def cover_note(decisions: list[Decision]) -> str:
    """cover-note-to-calyx.docx"""
    L = [
        (f"To:      {FACTS['their_counsel']}, counsel to {FACTS['them']}"
         if taskctx.is_default() else f"To:      {_partie('them')}"),
        (f"From:    {FACTS['our_counsel']}, for {FACTS['us']}"
         if taskctx.is_default() else f"From:    {_partie('us')}"),
        ("Re:      Data Sharing Agreement - Carden counter-turn"
         if taskctx.is_default() else f"Re:      Counter-turn redline - {_nous()}"),
        "",
        _opening(),
        "",
    ]
    for d in decisions:
        iss = taskctx.issues_by_id().get(d.issue_id)
        if iss is None:
            continue
        L.append(f"{iss.name} (Section {iss.section}) - {VERB[d.disposition]}")
        L.append(f"  {_nous()}'s position:" if not taskctx.is_default()
                 else "  Carden's position:")
        for line in _wrap(d.counter, 70):
            L.append(f"      {line}")
        L.append("  Rationale:")
        for line in _wrap(d.rationale, 70):
            L.append(f"      {line}")
        L.append("")
    L.append("We are keen to finalise the Agreement promptly and are happy to schedule a "
             "call to work through any remaining points.")
    return "\n".join(L)


def _wrap(s: str, w: int) -> list[str]:
    out, line = [], ""
    for word in (s or "").split():
        if len(line) + len(word) + 1 > w:
            out.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        out.append(line)
    return out or [""]
