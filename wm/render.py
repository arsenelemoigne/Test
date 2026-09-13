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

from .abstraction import FACTS, ISSUES_BY_ID, Decision, Disposition

VERB = {
    Disposition.ACCEPT: "Accepted",
    Disposition.REJECT: "Rejected",
    Disposition.MODIFY: "Modified",
}


def redline(decisions: list[Decision]) -> str:
    """counter-turn-redline-dsa.docx"""
    L = [
        "DATA SHARING AGREEMENT - CARDEN COUNTER-TURN REDLINE",
        f"{FACTS['us']} and {FACTS['them']}",
        f"Counter-turn against the Luminos first markup received {FACTS['markup_received']}, "
        f"marked against the Carden initial draft of {FACTS['initial_draft_date']}.",
        "Prepared in Microsoft Word Track Changes.",
        "",
        "=" * 76,
        "DISPOSITION OF EACH TRACKED CHANGE",
        "=" * 76,
        "",
    ]
    for d in decisions:
        iss = ISSUES_BY_ID.get(d.issue_id)
        if iss is None:
            continue
        L.append(f"Section {iss.section} - {iss.name}")
        L.append(f"  Carden initial draft: {iss.ours}")
        L.append(f"  Luminos proposed:     {iss.theirs}")
        L.append(f"  DISPOSITION:          {VERB[d.disposition].upper()}")
        L.append(f"  Carden counter-position (operative language for this turn):")
        for line in _wrap(d.counter, 70):
            L.append(f"      {line}")
        L.append("")
    return "\n".join(L)


def cover_note(decisions: list[Decision]) -> str:
    """cover-note-to-calyx.docx"""
    L = [
        f"To:      {FACTS['their_counsel']}, counsel to {FACTS['them']}",
        f"From:    {FACTS['our_counsel']}, for {FACTS['us']}",
        "Re:      Data Sharing Agreement - Carden counter-turn",
        "",
        f"Thank you for the markup received {FACTS['markup_received']}. Attached is "
        f"Carden's counter-turn redline, prepared in Microsoft Word Track Changes against "
        f"the Carden initial draft of {FACTS['initial_draft_date']}. Our position on each "
        "substantive change is set out below.",
        "",
    ]
    for d in decisions:
        iss = ISSUES_BY_ID.get(d.issue_id)
        if iss is None:
            continue
        L.append(f"{iss.name} (Section {iss.section}) - {VERB[d.disposition]}")
        L.append("  Carden's position:")
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
