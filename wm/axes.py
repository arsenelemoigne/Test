"""
Les AXES : chaque modification du markup, classee par fonction juridique,
direction et beneficiaire. Le "vecteur" de la moitie du contrat que le moteur
de scenarios ne voit pas.

Le vecteur declare de ce matin demandait a un modele COMBIEN valait une
redaction, sur des axes comme tail_risk - une valorisation, que rien
n'ancrait (rho = 0,22). Ici les axes sont des FONCTIONS juridiques et une
modification les deplace par REGLE, pas par opinion : inserer "material"
devant "breach" releve le seuil du recours et alourdit la charge du
demandeur. Tout juriste s'accorde sur la direction ; personne sur
l'ampleur ; on ne demande que la direction. Le modele de langage CLASSE
(quelle clause, quel axe, quel sens, au profit de qui) et une classification
se verifie contre un jeu etiquete. Une valorisation ne se verifiait pas.

Ce que le profil PEUT dire : quel camp chaque modification favorise et sur
quelle fonction ; ou un markup concentre sa pression ; l'asymetrie ("vous
avez accepte six deplacements de charge et rien n'est revenu") ; les
modifications dont le sens est reellement ambigu, signalees au lieu d'etre
cachees. Ce qu'il NE PEUT PAS dire : que trois deplacements de charge valent
une hausse de plafond. Les echanges entre axes demandent des dollars
(claim.py, pour le coeur parametrique) ou une fonction de preference. Les
deux se composent ; ni l'un ni l'autre ne suffit seul.

Marotta-Wurgler (4 JELS 677, 2007) a fait cela sur un axe (pro-vendeur /
pro-acheteur) pour 647 contrats ; c'est le meme indice, sur huit fonctions,
avec des directions determinees par regle.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict

AXES = {
    "obligation":   "force de l'obligation : resultat <-> moyens ; 'shall' <-> 'reasonable efforts'",
    "threshold":    "seuil de declenchement : material, substantial, knowing, willful, de minimis",
    "burden":       "charge de la preuve : qui doit etablir le fait ; quels registres font foi",
    "scope":        "perimetre : definitions, affilies, territoire, carve-outs, exclusions",
    "timing":       "delais : preavis, cure, survie, prescription, 'promptly' vs 'N days'",
    "remedy":       "recours : exclusivite, plafonds, credits, resiliation, acces au juge",
    "control":      "discretion : consentement, 'sole discretion', approbation, cession",
    "verification": "verification : audit, reporting, certification, escrow",
    # Les termes de PRIX n'ont pas de fonction juridique : un escalateur qui
    # passe de 4 % a CPI plafonne a 2 % ne deplace ni le recours ni la charge,
    # il deplace de l'argent. Sans cet axe le classifieur les rangeait sous
    # "remedy" et gonflait cet axe d'un cinquieme. Ils sont chiffres par
    # claim.py ; ici on les compte, on ne les valorise pas.
    "price":        "prix : montants, taux, escalateurs, echeances de paiement (chiffre ailleurs)",
}
PARTIES = ("licensor", "licensee")
MAGNITUDES = ("minor", "moderate", "major")


# --- 1. les modifications, extraites du markup -------------------------------

@dataclass
class Hunk:
    id: str
    section: str
    heading: str
    kind: str          # inline | added | deleted | new_section | removed_section
    text: str          # avec les marqueurs {+ +} {- -}
    inserted: int = 0
    deleted: int = 0
    before: str = ""   # la version du template pour une section reecrite


# Un en-tete est une LIGNE : "Section 2.1 — Grant", "Article 3 — Fees",
# "A.1 — Annual License Fee", "SCHEDULE A — PRICING". Chercher "Section X" au
# milieu d'un corps de texte rattachait un alinea a la section qu'il CITE, pas
# a celle ou il se trouve ; et les annexes, qui n'ecrivent pas "Section",
# n'avaient aucune adresse - vingt-quatre modifications d'un bareme se sont
# retrouvees sous la derniere clause du corps.
_HEAD_LINE = re.compile(
    r"^\s*(?:\*\*)?(?:\{[+-])?\s*(?:(?:Section|Article|SECTION|ARTICLE)\s+)?"
    r"((?:[A-Z]\.)?\d+(?:\.\d+)*(?:\([a-z]\))?|(?:SCHEDULE|EXHIBIT|Schedule|Exhibit)\s+[A-Z])"
    r"\s*[—–-]\s*([^\n]{1,80})")
_SEC = re.compile(r"Section\s+([0-9A-Z]+(?:\.[0-9]+)*(?:\([a-z]\))?)\s*[—–-]\s*([^.\n]{0,80})")
_HEAD = re.compile(r"^(?:\*\*)?(?:\{[+-])?\s*(?:Section\s+)?((?:[A-Z]\.)?\d+(?:\.\d+)*|(?:SCHEDULE|EXHIBIT|Schedule|Exhibit)\s+[A-Z])\s*[—–-]")


_MARQ = re.compile(r"^\s*\[(?:ADDED|DELETED):\s*")


def _nu(ln: str) -> str:
    """La ligne sans son marqueur d'ouverture de bloc.

    "[DELETED: Section 12.1 - Audit..." porte bien un en-tete de section, mais
    _HEAD_LINE ancre en debut de ligne et le crochet le masquait. La PREMIERE
    ligne d'un bloc perdait donc son adresse et heritait de la section
    precedente - l'article 12 supprime se retrouvait sous la 9.1.
    """
    return _MARQ.sub("", ln)


def _section_of(text: str) -> tuple[str, str]:
    for ln in text.split("\n"):
        m = _HEAD_LINE.match(_nu(ln))
        if m:
            head = m.group(2).replace("**", "").strip().strip('"').rstrip("+}-").strip()
            return m.group(1).strip(), head.split(". ")[0][:80]
    return "", ""


def _blocks(lines: list[str]) -> list[tuple[str, list[str]]]:
    """Regroupe les lignes : un bloc [ADDED: ... ] ou [DELETED: ... ] peut
    courir sur plusieurs paragraphes, jusqu'au crochet fermant."""
    out, i = [], 0
    while i < len(lines):
        ln = lines[i]
        m = re.search(r"\[(ADDED|DELETED):", ln)
        if m:
            kind = m.group(1).lower()
            buf, depth = [], 0
            while i < len(lines):
                buf.append(lines[i])
                depth += lines[i].count("[") - lines[i].count("]")
                i += 1
                if depth <= 0:
                    break
            out.append((kind, buf))
            continue
        out.append(("plain", [ln]))
        i += 1
    return out


def _template_index(template: str) -> dict[str, str]:
    """Le texte du template par numero de section (l'en-tete et ses alineas)."""
    idx, cur = {}, ""
    for ln in template.split("\n"):
        if not ln.strip():
            continue
        sec, _ = _section_of(ln)
        if sec and _HEAD.match(ln.strip()):
            cur = sec
            idx[cur] = ln.strip()
        elif cur:
            idx[cur] += "\n" + ln.strip()
    return idx


def parties(template: str) -> tuple[str, str]:
    """Les noms des parties, lus dans le preambule : 'X, a ... ("Licensor")'.
    Le fichier _parties.json porte les CONSEILS ; un profil qui titre
    'Nakamura & Oakes' pour le licencie a confondu l'avocat et le client."""
    def find(role):
        m = re.search(r"\n([^\n,]{3,80}),[^\n]{0,200}\(\"" + role + r"\"\)", template)
        return m.group(1).strip() if m else role.capitalize()
    return find("Licensor"), find("Licensee")


def _fend(buf: list[str], sec0: str, head0: str) -> list[tuple[str, str, list[str]]]:
    """Fend un bloc [ADDED:] / [DELETED:] a chaque nouvelle SECTION qu'il contient.

    Un seul bloc peut ajouter plusieurs sections d'un coup : dans ce markup,
    un [ADDED: ...] court de 14.12 a 14.14 et le crochet ne se referme qu'a la
    fin. Attribue en entier a sa PREMIERE section, il faisait disparaitre la
    non-sollicitation ET la clause du client le plus favorise - deux clauses
    entieres, dont une que le memo du client classe en walk-away. Elles
    n'apparaissaient nulle part dans le profil, et leur point sortait avec un
    score de zero qu'on aurait pu lire comme "le markup n'y touche pas".

    Les alineas restent avec leur section : 14.12(a) a (e) ne fendent rien,
    seul le passage a 14.13 fend.
    """
    pieces, cur = [], []
    sec, head = sec0, head0          # l'adresse de la piece EN COURS
    for ln in buf:
        m = _HEAD_LINE.match(_nu(ln))
        if m:
            base = m.group(1).split("(")[0]
            if cur and base != (sec or "").split("(")[0]:
                pieces.append((sec, head, cur))
                cur, sec, head = [], "", ""
            if not sec:
                # la premiere adresse rencontree dans la piece la nomme ; les
                # suivantes sont ses alineas et ne la renomment pas, sinon
                # 14.12(a)-(e) ressortait sous "14.12(e)"
                sec, head = _section_of(_nu(ln))
                if not sec:
                    sec, head = m.group(1).strip(), m.group(2)[:80]
        cur.append(ln)
    if cur:
        pieces.append((sec, head, cur))
    return pieces


def hunks(markup: str, template: str = "") -> list[Hunk]:
    """Chaque modification du markup, dans l'ordre du document.

    Trois notations coexistent dans un markup extrait : {+ +} / {- -} en ligne,
    des blocs [ADDED: ...] / [DELETED: ...] qui peuvent courir sur plusieurs
    paragraphes, et des sections entieres ajoutees SANS marqueur (14.12,
    Schedule C). Les dernieres ne se voient qu'en comparant les numeros de
    section au template - d'ou le second argument.
    """
    lines = [l for l in markup.split("\n") if l.strip()]
    out: list[Hunk] = []
    n = 0
    last_sec, last_head = "", ""
    for kind, buf in _blocks(lines):
        text = "\n".join(buf)
        sec, head = _section_of(text)
        if sec:
            last_sec, last_head = sec, head
        else:
            # un alinea "(b) ..." ou une ligne de bareme appartient a la
            # section qui le precede ; sans cela un tiers des modifications
            # n'avait pas d'adresse
            sec, head = last_sec, (last_head + " (suite)") if last_head else ""
        if kind in ("added", "deleted"):
            for sec_i, head_i, sub in _fend(buf, sec, head):
                n += 1
                body = re.sub(r"\[(ADDED|DELETED):\s*", "", "\n".join(sub)).rstrip("]")
                out.append(Hunk(
                    id=f"h{n:02d}", section=sec_i or sec, heading=head_i or head, kind=kind,
                    text=("{+" + body + "+}") if kind == "added" else ("{-" + body + "-}"),
                    inserted=len(body) if kind == "added" else 0,
                    deleted=len(body) if kind == "deleted" else 0))
                if sec_i:
                    last_sec, last_head = sec_i, head_i
        elif "{+" in text or "{-" in text:
            n += 1
            ins = sum(len(x) for x in re.findall(r"\{\+(.*?)\+\}", text, re.S))
            dele = sum(len(x) for x in re.findall(r"\{-(.*?)-\}", text, re.S))
            k = "new_section" if text.lstrip("*").startswith("{+Section") else (
                "removed_section" if text.lstrip("*").startswith("{-Section") else "inline")
            out.append(Hunk(id=f"h{n:02d}", section=sec, heading=head, kind=k, text=text,
                            inserted=ins, deleted=dele))

    if template:
        # une section reecrite d'un bloc arrive comme ADDED : sans la version du
        # template en face, le modele classe comme mouvement du texte qui n'a
        # pas bouge ("sole control of the defence" etait deja la)
        idx = _template_index(template)
        for h in out:
            base = h.section.split("(")[0]
            if h.kind in ("added", "deleted") or h.deleted > 300:
                h.before = idx.get(h.section, "") or idx.get(base, "")
        tmpl_secs = {_section_of(ln)[0] for ln in template.split("\n") if _HEAD.match(ln)}
        # seules les sections portant LEUR propre en-tete comptent comme deja
        # vues : un alinea rattache par heritage a 14.12 ne prouve pas que
        # 14.12 a ete marquee
        marked = {h.section for h in out if "(suite)" not in h.heading}
        # sections presentes dans le markup, absentes du template, sans marqueur
        for kind, buf in _blocks(lines):
            if kind != "plain":
                continue
            text = buf[0]
            sec, head = _section_of(text)
            if sec and sec not in tmpl_secs and sec not in marked and _HEAD.match(text.strip()):
                n += 1
                marked.add(sec)
                out.append(Hunk(id=f"h{n:02d}", section=sec, heading=head, kind="new_section",
                                text="{+" + text + "+}", inserted=len(text)))
    return out


# --- 2. la classification ------------------------------------------------------

@dataclass
class Move:
    hunk: str
    section: str
    axis: str
    favours: str        # licensor | licensee | ambiguous | neutral
    magnitude: str      # minor | moderate | major
    what: str           # une phrase
    quote: str = ""     # le passage decisif


CLASSIFY_PROMPT = """You are classifying tracked changes in a contract markup by LEGAL FUNCTION.
The contract is between {licensor} ("Licensor", the vendor) and {licensee} ("Licensee",
the customer). The markup was written by counsel to the Licensee.

For each change, list every legal-function MOVE it makes. One change can make several
moves (a rewritten section may move scope AND remedy). Each move has:
  axis      one of: {axes}
  favours   "licensor" | "licensee" | "ambiguous" | "neutral"
            - the party whose legal position the move strengthens, on that axis
            - "ambiguous" when the direction genuinely depends on who ends up the
              claimant (e.g. adding precision to a mutual clause); do NOT guess
            - "neutral" for renumbering, cross-references, drafting clean-up
  magnitude "minor" | "moderate" | "major" - how much the FUNCTION moves, not its
            dollar value (you are not valuing anything)
  what      one sentence: what the change does to that function
  quote     the decisive words of the change (short)

Rules of thumb (the direction is a matter of legal function, not opinion):
  - a qualifier inserted before an obligation or breach (material, substantial,
    reasonable, knowing, willful, gross) RAISES the threshold and favours the obligor
  - "shall" -> "shall use (commercially) reasonable efforts" weakens the obligation
    (result -> means) and favours the obligor
  - "sole and exclusive remedy" inserted narrows the remedy and favours the obligor;
    deleted, it widens the remedy and favours the obligee
  - a cure period inserted or lengthened favours the breaching party (timing)
  - "prior written consent of X" inserted favours X (control); deleted, the other party
  - a definition widened (Affiliate >50% -> 50% or more) widens scope for the party
    that benefits from the defined term
  - a cap raised, an exception to a cap added, a carve-out from a waiver added: remedy
    widened, favours the likely claimant on that clause
  - audit rights deleted: verification down, favours the audited party
  - a notice period lengthened favours the party RECEIVING the notice
  - fees, rates, escalators, amounts: axis "price", favours the party that pays less
    or receives more. Do NOT file price changes under "remedy": they move money,
    not legal function

AXES
{axis_help}

CHANGES
{changes}

Return ONLY a JSON array of moves, e.g.
[{{"hunk": "h03", "axis": "threshold", "favours": "licensor", "magnitude": "moderate",
  "what": "...", "quote": "..."}}, ...]
Every hunk id listed above must appear at least once (use axis "scope", favours
"neutral", magnitude "minor" for pure drafting changes)."""


def _render_hunk(h: Hunk, limit: int = 1400) -> str:
    t = h.text if len(h.text) <= limit else h.text[:limit] + " [...]"
    out = f"--- {h.id}  Section {h.section or '?'}  {h.heading}  ({h.kind})\n{t}"
    if h.before:
        b = h.before if len(h.before) <= limit else h.before[:limit] + " [...]"
        out += (f"\n    BEFORE (template text of this section - classify ONLY what changed; "
                f"text present in both is NOT a move):\n    {b}")
    return out


def parse_moves(raw: str, valid_hunks: set[str]) -> tuple[list[Move], list[str]]:
    """Les mouvements lus, et les raisons de rejet pour ceux qui ne le sont pas."""
    spans, pile, dans, esc = [], [], False, False
    for i, ch in enumerate(raw or ""):
        if dans:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                dans = False
            continue
        if ch == '"':
            dans = True
        elif ch == "[":
            pile.append(i)
        elif ch == "]" and pile:
            spans.append(raw[pile.pop():i + 1])
    rows = None
    for s in sorted(spans, key=len, reverse=True):
        try:
            cand = json.loads(s)
        except ValueError:
            continue
        if isinstance(cand, list) and any(isinstance(r, dict) and "axis" in r for r in cand):
            rows = cand
            break
    if rows is None:
        return [], ["aucun tableau JSON de mouvements"]
    out, why = [], []
    for r in rows:
        if not isinstance(r, dict):
            continue
        hid, ax, fav, mag = (str(r.get("hunk", "")), str(r.get("axis", "")).lower(),
                             str(r.get("favours", "")).lower(), str(r.get("magnitude", "")).lower())
        if hid not in valid_hunks:
            why.append(f"{hid}: id inconnu")
            continue
        if ax not in AXES:
            why.append(f"{hid}: axe inconnu {ax!r}")
            continue
        if fav not in PARTIES + ("ambiguous", "neutral"):
            why.append(f"{hid}: beneficiaire inconnu {fav!r}")
            continue
        if mag not in MAGNITUDES:
            mag = "moderate"
        out.append(Move(hunk=hid, section="", axis=ax, favours=fav, magnitude=mag,
                        what=str(r.get("what", ""))[:240], quote=str(r.get("quote", ""))[:160]))
    return out, why


def classify(hs: list[Hunk], llm, licensor: str, licensee: str, batch: int = 6,
             raw_out=None) -> tuple[list[Move], list[str]]:
    """Un appel par lot de `batch` modifications. Rend les mouvements et les
    rejets. Chaque reponse brute est conservee si raw_out est fourni."""
    moves, why = [], []
    by_id = {h.id: h for h in hs}
    for i in range(0, len(hs), batch):
        lot = hs[i:i + batch]
        prompt = CLASSIFY_PROMPT.format(
            licensor=licensor, licensee=licensee, axes=", ".join(AXES),
            axis_help="\n".join(f"  {k:<13} {v}" for k, v in AXES.items()),
            changes="\n\n".join(_render_hunk(h) for h in lot))
        raw = llm(prompt, max_tokens=14000) or ""
        if raw_out is not None:
            raw_out.append(raw)
        ms, w = parse_moves(raw, {h.id for h in lot})
        if not ms and len(lot) > 1:
            # un lot entier perdu (reponse coupee ou sans JSON) : on le
            # rejoue en deux moities plutot que de perdre six modifications
            ms, w = [], []
            for half in (lot[:len(lot) // 2], lot[len(lot) // 2:]):
                sub, w2 = classify(half, llm, licensor, licensee, batch=len(half), raw_out=raw_out)
                ms += sub
                w += w2
            moves += ms
            why += w
            continue
        for m in ms:
            m.section = by_id[m.hunk].section
        moves += ms
        why += w
        missing = {h.id for h in lot} - {m.hunk for m in ms}
        for hid in sorted(missing):
            why.append(f"{hid}: aucun mouvement rendu")
    return moves, why


# --- 3. l'oracle par regles ----------------------------------------------------
# Ne juge que les formes canoniques, et seulement quand l'obligé de la phrase
# est identifiable. Sert de second annotateur : l'accord regles / modele est
# la premiere mesure de fiabilite disponible sans juriste.

_QUALIFIERS = r"(?:material(?:ly)?|substantial(?:ly)?|knowing(?:ly)?|willful(?:ly)?|gross(?:ly)?)"
_EFFORTS = r"(?:commercially\s+)?reasonable\s+efforts"


def _obligor(sentence: str) -> str | None:
    """'Licensor shall' -> licensor. Clauses mutuelles -> None."""
    s = sentence.lower()
    if re.search(r"\b(each|either)\s+party\b|\bthe\s+parties\b", s):
        return None
    m = re.search(r"\b(licensor|licensee)(?:'s)?\s+(?:shall|must|will|may|agrees)", s)
    return m.group(1) if m else None


def _other(p: str) -> str:
    return "licensee" if p == "licensor" else "licensor"


def rule_label(text: str, max_chars: int = 500) -> tuple[str, str] | None:
    """(axe, beneficiaire) pour une modification canonique, sinon None.

    Formes COURTES seulement : une forme canonique ne dit quelque chose que
    quand elle EST la modification. Sur un bloc de 3 000 caracteres, "cure"
    ou "reasonable efforts" apparaissent quelque part sans etre le point, et
    l'oracle contredisait le modele a tort - deux des quatre desaccords de la
    premiere passe venaient de la."""
    if len(text) > max_chars:
        return None
    ins = " ".join(re.findall(r"\{\+(.*?)\+\}", text, re.S)).lower()
    dele = " ".join(re.findall(r"\{-(.*?)-\}", text, re.S)).lower()
    plain = re.sub(r"\{[+-](.*?)[+-]\}", r"\1", text, flags=re.S)
    ob = _obligor(plain)

    if re.search(_EFFORTS, ins) and not re.search(_EFFORTS, dele):
        return ("obligation", ob) if ob else ("obligation", "ambiguous")
    if re.search(_EFFORTS, dele) and not re.search(_EFFORTS, ins):
        return ("obligation", _other(ob)) if ob else ("obligation", "ambiguous")
    if re.fullmatch(rf"\s*{_QUALIFIERS}\s*", ins) and not dele:
        return ("threshold", ob) if ob else ("threshold", "ambiguous")
    if re.fullmatch(rf"\s*{_QUALIFIERS}\s*", dele) and not ins:
        return ("threshold", _other(ob)) if ob else ("threshold", "ambiguous")
    if "sole and exclusive remedy" in ins:
        return ("remedy", ob) if ob else ("remedy", "ambiguous")
    if "sole and exclusive remedy" in dele:
        return ("remedy", _other(ob)) if ob else ("remedy", "ambiguous")
    m = re.search(r"prior written consent of (?:the )?(licensor|licensee)", ins)
    if m:
        return ("control", m.group(1))
    m = re.search(r"prior written consent of (?:the )?(licensor|licensee)", dele)
    if m:
        return ("control", _other(m.group(1)))
    day = r"(\d+)\)?\s*[- ]?day"
    mi, md = re.search(day, ins), re.search(day, dele)
    if "cure" in plain.lower() and (mi or md):
        # la partie que le delai de cure protege est celle qui doit guerir
        # la partie ADJACENTE a "fails to cure", pas la premiere de la phrase :
        # "Licensee may terminate if Licensor fails to cure" designe Licensor
        b = re.search(r"\b(licensor|licensee)\s+fails?\s+to\s+cure", plain.lower())
        breaching = b.group(1) if b else ob
        if not breaching:
            return ("timing", "ambiguous")
        if mi and md:
            longer = int(mi.group(1)) > int(md.group(1))
            return ("timing", breaching if longer else _other(breaching))
        return ("timing", breaching if mi else _other(breaching))
    return None


def agreement(hs: list[Hunk], moves: list[Move]) -> tuple[int, int, list[str]]:
    """Sur les modifications que les regles savent juger : le modele a-t-il
    rendu le meme axe ET le meme beneficiaire ? (juges, accords, desaccords)"""
    n = ok = 0
    bad = []
    by_hunk: dict[str, list[Move]] = {}
    for m in moves:
        by_hunk.setdefault(m.hunk, []).append(m)
    for h in hs:
        lab = rule_label(h.text)
        if lab is None:
            continue
        n += 1
        got = by_hunk.get(h.id, [])
        if any(m.axis == lab[0] and m.favours == lab[1] for m in got):
            ok += 1
        else:
            bad.append(f"{h.id} s.{h.section}: regles {lab[0]}/{lab[1]}, modele "
                       + (", ".join(f"{m.axis}/{m.favours}" for m in got) or "rien"))
    return n, ok, bad


# --- 4. le profil ----------------------------------------------------------------

_W = {"minor": 1, "moderate": 2, "major": 4}


def profile(moves: list[Move]) -> dict:
    """Par axe : mouvements et poids par beneficiaire, ambigus, neutres."""
    P = {a: {"licensor": [0, 0], "licensee": [0, 0], "ambiguous": 0, "neutral": 0}
         for a in AXES}
    for m in moves:
        if m.favours in PARTIES:
            P[m.axis][m.favours][0] += 1
            P[m.axis][m.favours][1] += _W[m.magnitude]
        else:
            P[m.axis][m.favours] += 1
    return P


def report(hs: list[Hunk], moves: list[Move], why: list[str], licensor: str, licensee: str,
           parametric_sections: set[str] | None = None) -> str:
    P = profile(moves)
    L = ["PROFIL DU MARKUP PAR FONCTION JURIDIQUE", "=" * 84,
         f"{len(hs)} modifications, {len(moves)} mouvements classes, "
         f"{sum(1 for m in moves if m.favours == 'ambiguous')} ambigus, "
         f"{sum(1 for m in moves if m.favours == 'neutral')} neutres", "",
         f"{'axe':<14}{'-> ' + licensor[:14]:>22}{'-> ' + licensee[:14]:>22}{'ambigu':>9}{'net':>9}",
         "-" * 84]
    tot_l = tot_r = 0
    for a in AXES:
        l, r = P[a]["licensor"], P[a]["licensee"]
        tot_l += l[1]
        tot_r += r[1]
        net = r[1] - l[1]
        L.append(f"{a:<14}{f'{l[0]} ({l[1]})':>22}{f'{r[0]} ({r[1]})':>22}{P[a]['ambiguous']:>9}"
                 f"{net:>+9}")
    L += ["-" * 84, f"{'TOTAL (poids)':<14}{tot_l:>22}{tot_r:>22}{'':>9}{tot_r - tot_l:>+9}",
          "", "poids : minor 1, moderate 2, major 4. Le net est ORDINAL : il dit de quel",
          "cote et sur quelle fonction le markup appuie, pas ce que cela vaut."]
    # concentration
    tot = tot_l + tot_r
    if tot:
        top = sorted(AXES, key=lambda a: -(P[a]["licensor"][1] + P[a]["licensee"][1]))[:3]
        share = sum(P[a]["licensor"][1] + P[a]["licensee"][1] for a in top) / tot
        L += ["", f"Concentration : {', '.join(top)} portent {share:.0%} du poids."]
    if tot_l + tot_r and tot_r > 3 * max(tot_l, 1):
        L += [f"Asymetrie : {tot_r} contre {tot_l}. Presque rien ne revient a {licensor} - "
              f"c'est un premier markup, c'est attendu, et c'est ce qu'une contre-proposition",
              "doit renverser axe par axe, pas seulement sur les chiffres."]

    # par section
    L += ["", "PAR MODIFICATION", "-" * 84]
    by_hunk: dict[str, list[Move]] = {}
    for m in moves:
        by_hunk.setdefault(m.hunk, []).append(m)
    for h in hs:
        ms = by_hunk.get(h.id, [])
        tag = ""
        if parametric_sections and h.section and h.section.split("(")[0] in parametric_sections:
            tag = "  [chiffre par claim.py]"
        L.append(f"{h.id}  s.{h.section or '?':<8} {h.heading[:40]:<42}{h.kind:<16}{tag}")
        for m in ms:
            L.append(f"        {m.axis:<13} -> {m.favours:<10} {m.magnitude:<9} {m.what[:70]}")
        if not ms:
            L.append("        (non classe)")
    amb = [m for m in moves if m.favours == "ambiguous"]
    if amb:
        L += ["", f"{len(amb)} MOUVEMENTS AMBIGUS - le sens depend de qui sera demandeur :"]
        for m in amb:
            L.append(f"  {m.hunk} s.{m.section:<7} {m.axis:<13} {m.what[:80]}")
    if why:
        L += ["", f"{len(why)} rejets ou manques :"] + [f"  {w}" for w in why[:12]]
    return "\n".join(L)
