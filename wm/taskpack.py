"""
Turn any Harvey LAB task folder into plain text this pipeline can read.

The documents arrive as .docx (often with tracked changes), .eml and .xlsx.
Tracked changes are the point of a markup task, so they are preserved inline as
{+inserted+} and {-deleted-} rather than flattened to either version.

    python -m wm.run pack ip-licensing/license-agreement-first-turn-redline/scenario-04
"""

from __future__ import annotations

import email
import email.policy
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def _walk(node) -> str:
    """Paragraph text in document order, insertions and deletions marked.

    Walks the tree rather than collecting w:ins text and substituting it back:
    a paragraph with several separate insertions concatenates into a string
    that appears nowhere in the text, so all but the first were lost.
    """
    parts = []
    for child in node:
        tag = child.tag
        if tag == W + "ins":
            inner = _walk(child)
            if inner:
                parts.append("{+" + inner + "+}")
        elif tag == W + "del":
            inner = _walk(child)
            if inner:
                parts.append("{-" + inner + "-}")
        elif tag in (W + "t", W + "delText"):
            parts.append(child.text or "")
        elif tag == W + "tab":
            parts.append("\t")
        elif tag in (W + "br", W + "cr"):
            parts.append("\n")
        else:
            parts.append(_walk(child))
    return "".join(parts)


def _para_text(p) -> str:
    return _walk(p)


def extract_docx(path: Path) -> str:
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml")
    root = ET.fromstring(xml)
    lines = []
    for p in root.iter(W + "p"):
        t = _para_text(p).strip()
        if t:
            lines.append(t)
    return "\n".join(lines)


def extract_eml(path: Path) -> str:
    m = email.message_from_bytes(path.read_bytes(), policy=email.policy.default)
    head = [f"{k}: {m[k]}" for k in ("From", "To", "Cc", "Date", "Subject") if m[k]]
    try:
        body = m.get_body(preferencelist=("plain", "html"))
        text = body.get_content() if body else ""
    except Exception:                                   # noqa: BLE001
        text = m.get_payload(decode=True).decode("utf-8", "replace")
    return "\n".join(head) + "\n\n" + text


S = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def _col(ref: str) -> int:
    """"BC12" -> 54. Column letters to a zero-based index."""
    n = 0
    for ch in ref:
        if not ch.isalpha():
            break
        n = n * 26 + (ord(ch.upper()) - 64)
    return n - 1


def extract_xlsx(path: Path) -> str:
    """Read a workbook with the standard library only.

    An .xlsx is a zip of XML. openpyxl is nicer but is not installed
    everywhere, and a missing pricing model is a missing negotiation input -
    it should not be silently skipped because of a dependency.
    """
    with zipfile.ZipFile(path) as z:
        names = set(z.namelist())
        shared = []
        if "xl/sharedStrings.xml" in names:
            root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in root.iter(S + "si"):
                shared.append("".join(t.text or "" for t in si.iter(S + "t")))

        titles = {}
        if "xl/workbook.xml" in names:
            wb = ET.fromstring(z.read("xl/workbook.xml"))
            for idx, sh in enumerate(wb.iter(S + "sheet"), start=1):
                titles[idx] = sh.get("name", f"sheet{idx}")

        out = []
        sheets = sorted(n for n in names
                        if n.startswith("xl/worksheets/sheet") and n.endswith(".xml"))
        for i, sn in enumerate(sheets, start=1):
            out.append(f"--- sheet: {titles.get(i, sn.rsplit('/', 1)[-1])}")
            root = ET.fromstring(z.read(sn))
            for row in root.iter(S + "row"):
                cells = {}
                for c in row.iter(S + "c"):
                    ref, typ = c.get("r", ""), c.get("t")
                    v = c.find(S + "v")
                    if typ == "s" and v is not None and v.text is not None:
                        try:
                            val = shared[int(v.text)]
                        except (ValueError, IndexError):
                            val = ""
                    elif typ == "inlineStr":
                        val = "".join(t.text or "" for t in c.iter(S + "t"))
                    else:
                        val = (v.text or "") if v is not None else ""
                    if val.strip():
                        cells[_col(ref)] = val.strip()
                if cells:
                    width = max(cells) + 1
                    out.append(" | ".join(cells.get(j, "") for j in range(width)).rstrip(" |"))
        return "\n".join(out)


EXTRACTORS = {".docx": extract_docx, ".eml": extract_eml, ".xlsx": extract_xlsx,
              ".txt": lambda p: p.read_text(errors="replace"),
              ".md": lambda p: p.read_text(errors="replace")}


def pack(task_dir: Path, out_dir: Path) -> dict[str, int]:
    """Extract every document to <out_dir>/<stem>.txt. Returns name -> chars."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sizes = {}
    docs = task_dir / "documents"
    for f in sorted(docs.iterdir() if docs.exists() else []):
        fn = EXTRACTORS.get(f.suffix.lower())
        if fn is None:
            sizes[f.name] = -1                 # unreadable, recorded not hidden
            continue
        try:
            text = fn(f)
        except Exception as e:                          # noqa: BLE001
            sizes[f.name] = -1
            print(f"  {f.name}: FAILED {str(e)[:70]}")
            continue
        (out_dir / (f.stem + ".txt")).write_text(text)
        sizes[f.name] = len(text)
    tj = task_dir / "task.json"
    if tj.exists():
        (out_dir / "task.json").write_text(tj.read_text())
    return sizes


# --- role detection --------------------------------------------------------
# Which file is the template, which is their markup, which is our mandate.
# Named by pattern because the benchmark names files descriptively and this is
# more honest than asking a model to guess and then trusting it silently.
ROLE_PATTERNS = [
    ("markup",   r"(first[- ]markup|markup|redline|marked)"),
    ("memo",     r"(negotiation|playbook|parameters|authority|instruction|mandate)"),
    ("template", r"(template|form|initial[- ]draft|standard|master)"),
    ("policy",   r"(policy|guidelines)"),
    ("pricing",  r"(fee|rate[- ]card|pricing|schedule[- ]b)"),
    ("email",    r"(email|transmittal|cover|\.eml)"),
]


def roles(names: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for n in names:
        low = n.lower()
        for role, pat in ROLE_PATTERNS:
            if re.search(pat, low):
                out.setdefault(role, []).append(n)
                break
        else:
            out.setdefault("other", []).append(n)
    return out
