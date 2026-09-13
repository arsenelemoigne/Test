"""
The tied encoder -- the JEPA-shaped part.

JEPA's core move is that the same (or EMA-tied) encoder maps both views into a
shared latent space, and the loss lives in that space rather than in the input
space. Two things follow that matter here:

  * the two documents MUST go through the identical function, or their latents
    are not comparable;
  * the interesting quantity is the DIFFERENCE between latents, not either
    latent on its own.

What is different from a real JEPA: our encoder is symbolic and hand-designed
rather than learned. That is deliberate. A learned encoder needs a training
corpus of paired (draft, markup) contracts, which does not exist publicly; and
JEPA's defining property -- discarding detail that reconstruction would keep --
destroys exactly what a contract deliverable is made of ($3,500,000, 18 months,
Delaware). So the encoder is discrete and interpretable, and the "latent space"
is a fixed set of typed slots.

Two channels:

  SLOT CHANNEL (symbolic)  -- encode each document into the same 14 slots, then
                              diff. High precision, but blind to anything the
                              schema has no slot for.
  SECTION CHANNEL (lexical)-- align the two documents section by section and
                              measure how much each section moved. Low precision,
                              but it sees everything. Its job is RECALL: flagging
                              material changes that fall outside the schema.

The second channel is the honest use of a latent/embedding space in this problem:
detection, not reasoning. It is what would have caught the deleted "No License"
clause in the earlier NDA task without anyone thinking to model it.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parent / "task"


# ===========================================================================
# Slot channel
# ===========================================================================

@dataclass(frozen=True)
class Slot:
    """A dimension of the latent space. Carries NO position -- positions are
    what the encoder fills in, per document."""
    id: str
    name: str
    section: str
    question: str          # what the encoder must read out of a document


SLOTS: list[Slot] = [
    Slot("I01", "Liability cap", "14.1",
         "What is the aggregate liability cap? State the formulation exactly: fixed "
         "dollar amount, fee multiple, or a combination."),
    Slot("I02", "Data retention period", "6.1",
         "How long may the recipient retain each delivery of the Shared Data Set?"),
    Slot("I03", "Deletion obligation", "6.2",
         "At the end of the retention period, what must happen to the data? Is deletion "
         "or return mandatory, or is anonymisation-in-place permitted?"),
    Slot("I04", "Deletion certification", "6.3",
         "Is a written certification of destruction required, by whom and within what period?"),
    Slot("I05", "Re-identification prohibition", "7.1-7.3",
         "Is the re-identification prohibition absolute, or are there exceptions, "
         "qualifications or intent/materiality thresholds?"),
    Slot("I06", "Breach notification window and trigger", "8.1, 8.2, 1.8",
         "How many hours to notify a Security Incident, and what event starts the clock "
         "(first suspicion/Discovery, or confirmation after investigation)?"),
    Slot("I07", "Subprocessor approval", "9.1, 9.2",
         "May any subprocessor access the data without completing a Vendor Security "
         "Assessment? Is there a pre-approved list?"),
    Slot("I08", "Audit: regulator disclosure", "10.6",
         "May the discloser share audit findings with regulators, and is that right "
         "restricted or conditioned on consent?"),
    Slot("I09", "Audit: modality and frequency", "10.1-10.4",
         "Are on-site audits available or is auditing limited to remote means? How often?"),
    Slot("I10", "Residuals", "4.5, 1.19",
         "Is there a residuals clause permitting use of unaided-memory knowledge, and are "
         "residuals excluded from Confidential Information?"),
    Slot("I11", "Cross-border transfer", "11.1, 11.2",
         "What must be satisfied before a cross-border transfer: prior written approval, "
         "a Transfer Impact Assessment, a transfer mechanism, or some subset?"),
    Slot("I12", "Data use scope", "4.1, 4.1A, 1.17",
         "What may the recipient use the data for? Are derivative works or machine-learning "
         "models permitted, and who owns them?"),
    Slot("I13", "Governing law", "15.1", "Which state's law governs?"),
    Slot("I14", "Venue", "15.2", "Where must proceedings be brought?"),
    Slot("I15", "Term, renewal and non-renewal notice", "12.1, 12.2, 12.3",
         "How long is the initial term, how long is each automatic renewal, and how much "
         "notice is needed to stop renewal?"),
]

SLOTS_BY_ID = {s.id: s for s in SLOTS}


ENCODE_PROMPT = """Read the agreement below and answer each question about it.

Answer ONLY from this document. If the document does not address a question, answer
exactly "NOT ADDRESSED". Quote the operative words where you can. Be specific about
numbers, durations and jurisdictions. Do not evaluate, recommend or compare to
anything -- just report what this document says.

{questions}

Return ONLY a JSON object mapping each id to {{"value": "...", "quote": "..."}}.

=== AGREEMENT ===
{document}"""


def encode(document: str, llm) -> dict[str, dict]:
    """
    THE TIED ENCODER. Run this, unchanged, on each document.

    Note what it does NOT receive: the other document, the negotiation memo, the
    task instructions. It is a pure readout of one document into the shared slot
    space. That is what makes two encodings comparable -- and it is also what
    makes the representation reusable across tasks rather than a precomputed
    answer to this one.
    """
    qs = "\n".join(f"{s.id} [{s.name}, Section {s.section}]: {s.question}" for s in SLOTS)
    raw = llm(ENCODE_PROMPT.format(questions=qs, document=document), max_tokens=8000)
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        raise ValueError("encoder returned no JSON")
    return json.loads(m.group(0))


@dataclass
class Delta:
    slot_id: str
    name: str
    section: str
    ours: str
    theirs: str
    ours_quote: str
    theirs_quote: str
    changed: bool


def latent_diff(s_x: dict[str, dict], s_y: dict[str, dict]) -> list[Delta]:
    """s_x = our draft's latent, s_y = their markup's latent. The diff is the signal."""
    out = []
    for s in SLOTS:
        a = s_x.get(s.id, {})
        b = s_y.get(s.id, {})
        av, bv = a.get("value", "NOT ADDRESSED"), b.get("value", "NOT ADDRESSED")
        out.append(Delta(
            slot_id=s.id, name=s.name, section=s.section,
            ours=av, theirs=bv,
            ours_quote=a.get("quote", ""), theirs_quote=b.get("quote", ""),
            changed=_normalise(av) != _normalise(bv),
        ))
    return out


def _normalise(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


# ===========================================================================
# Section channel -- the recall guard
# ===========================================================================

SECTION_RE = re.compile(r'^\s*\**(\d{1,2}(?:\.\d{1,2}[A-Z]?)?)\**\s+["\u201c*]*([A-Za-z][^.]{2,80})[.:"\u201d]')


def sections(text: str) -> dict[str, str]:
    """Split a contract into {section number: body}. Crude and deliberately so --
    this channel trades precision for coverage."""
    out: dict[str, list[str]] = {}
    current = "0"
    for line in text.splitlines():
        m = SECTION_RE.match(line)
        if m:
            current = m.group(1)
            out.setdefault(current, [])
        out.setdefault(current, []).append(line)
    return {k: "\n".join(v) for k, v in out.items()}


def _shingles(s: str, n: int = 5) -> Counter:
    s = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", "", s.lower()))
    return Counter(s[i:i + n] for i in range(max(0, len(s) - n + 1)))


def _cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    num = sum(a[k] * b[k] for k in common)
    da = sum(v * v for v in a.values()) ** 0.5
    db = sum(v * v for v in b.values()) ** 0.5
    return num / (da * db) if da and db else 0.0


@dataclass
class SectionDelta:
    section: str
    similarity: float
    present_in_draft: bool
    present_in_markup: bool
    covered_by_slot: str | None


def section_diff(draft: str, markup: str, threshold: float = 0.995) -> list[SectionDelta]:
    """
    Align the two documents on their shared frame (section numbering) and measure
    how far each section moved. Sections that moved materially but map to no slot
    are the blind spots in the symbolic channel.

    Similarity is character-5-gram cosine -- a lexical latent. A sentence embedder
    drops in here unchanged.

    MEASURED LIMITATION, and it is the central finding about latent-space methods
    for contracts. Legal materiality is ANTI-CORRELATED with textual magnitude.
    On this task, at a conventional threshold of 0.93 the channel catches every
    structural move (deleted sections, inserted clauses, rewrites) and misses the
    three highest-impact edits in the whole markup:

        24 hours -> 72 hours          (breach notification)   similarity ~0.99
        Delaware -> Maryland          (governing law)         similarity ~0.98
        2 years  -> 3 years           (term)                  similarity ~0.99

    Each is one word in a long paragraph. Only at 0.995 do they surface, and then
    34 sections are flagged instead of 17. No threshold separates them cleanly,
    because the separation is not lexical -- it is legal.

    This is why a learned latent space cannot be the PRIMARY mechanism here, and
    why the slot channel carries the load: a slot asks "how many hours?" and reads
    72, regardless of how small the edit was.
    """
    a, b = sections(draft), sections(markup)
    slot_sections: dict[str, str] = {}
    for s in SLOTS:
        for part in re.split(r"[,\s]+", s.section):
            part = part.strip()
            if part:
                slot_sections[part] = s.id
                slot_sections[part.split(".")[0]] = slot_sections.get(part.split(".")[0], s.id)

    out = []
    for key in sorted(set(a) | set(b), key=_sortkey):
        sim = _cosine(_shingles(a.get(key, "")), _shingles(b.get(key, "")))
        if key in a and key in b and sim >= threshold:
            continue
        out.append(SectionDelta(
            section=key, similarity=sim,
            present_in_draft=key in a, present_in_markup=key in b,
            covered_by_slot=slot_sections.get(key) or slot_sections.get(key.split(".")[0]),
        ))
    return out


def _sortkey(s: str):
    try:
        return tuple(int(re.sub(r"\D", "", p) or 0) for p in s.split("."))
    except ValueError:
        return (999,)


def blind_spots(draft: str, markup: str) -> list[SectionDelta]:
    """Material section changes the slot schema cannot see. These are the ones a
    playbook-derived model misses by construction."""
    return [d for d in section_diff(draft, markup) if d.covered_by_slot is None]


# ===========================================================================

def render_latent(deltas: list[Delta], blind: list[SectionDelta]) -> str:
    """The latent diff, as text for the drafting model."""
    L = ["LATENT DIFF -- our draft vs their markup", "=" * 70, "",
         f"{sum(1 for d in deltas if d.changed)} of {len(deltas)} slots moved.", ""]
    for d in deltas:
        if not d.changed:
            continue
        L += [f"{d.slot_id}  {d.name}   [Section {d.section}]",
              f"    ours   : {d.ours}",
              f"    theirs : {d.theirs}"]
        if d.theirs_quote:
            L.append(f'    quote  : "{d.theirs_quote[:240]}"')
        L.append("")
    unchanged = [d.slot_id for d in deltas if not d.changed]
    if unchanged:
        L += [f"Unchanged slots: {', '.join(unchanged)}", ""]
    if blind:
        L += ["OUTSIDE THE SCHEMA -- sections that moved with no slot to hold them:", ""]
        for b in blind:
            state = ("deleted" if not b.present_in_markup else
                     "inserted" if not b.present_in_draft else
                     f"rewritten (similarity {b.similarity:.2f})")
            L.append(f"    Section {b.section}: {state}")
        L.append("")
    return "\n".join(L)
