"""A QUI appartient le chiffre qu'on vient de lire dans une contre-proposition ?

Le controle de mandat lisait tous les nombres du champ `counter` et prenait le
plus grand. Une contre-proposition ecrit pourtant les deux positions dans la
meme phrase - c'est meme sa forme normale :

    "Veridian offers a twelve (12) month warranty; Halcyon's demand for
     eighteen (18) months is rejected."

Douze est notre position, dix-huit est la leur, et prendre le maximum revient a
nous imputer la position que nous venons de refuser. Sur quatre formulations
courantes, trois declenchaient une violation imaginaire.

Ce module decoupe le texte en propositions et ecarte celles qui attribuent
visiblement leur chiffre a l'autre partie. C'est une HEURISTIQUE, et elle est
dangereuse dans un seul sens : ecarter une proposition ou nous ACCEPTONS leur
chiffre masquerait un vrai depassement. D'ou la seconde regle - une marque
d'acceptation a la premiere personne annule l'ecartement et la proposition est
gardee.

Ce qui est ecarte reste consultable (`split`), parce qu'un controle dont on ne
peut pas lire les rejets est un controle qu'on ne peut pas croire.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

# Marques d'attribution a l'autre partie, ou de refus explicite.
LEURS = (
    "reject", "rejects", "rejected", "rejecting", "decline", "declines", "declined",
    "refuse", "refused", "we do not accept", "is not acceptable", "unacceptable",
    "they propose", "they proposed", "they request", "they requested", "they demand",
    "their proposal", "their proposed", "their request", "their demand",
    "as proposed by", "proposed by the", "requested by the", "demanded by",
    "counterparty", "licensee's proposed", "licensor's proposed",
    "rather than", "instead of", "as opposed to", "in place of", "in lieu of",
    "not the", "down from", "up from", "originally", "previously",
    "the markup", "markup's", "struck", "delete", "deleted", "deleting",
    "revert", "reverted", "reverting", "restore", "restored", "restoring",
)

# Marques d'adoption a la premiere personne. Elles ANNULENT l'ecartement :
# "we accept the 24-month warranty they demanded" reste un depassement.
NOTRES = (
    "we accept", "we agree", "we are prepared to accept", "accepts the",
    "our position is", "we propose", "we offer", "we will", "we shall",
    "agreed at", "agree to", "concede", "conceded",
)

_COUPE = re.compile(r"(?<=[.;:])\s+|\s+(?:but|whereas|while|although|though)\s+|\n+",
                    re.IGNORECASE)

# Marqueurs CONTRASTIFS : ils ne qualifient pas la proposition entiere, ils
# coupent en son milieu. "twelve (12) months rather than the eighteen (18)
# months requested" est une seule proposition sans ponctuation, dont la tete
# est notre position et la queue la leur. Sans cette coupure, le chiffre de la
# queue restait impute a notre client.
_CONTRASTE = re.compile(
    r"\s+(?:rather than|instead of|as opposed to|in lieu of|in place of|"
    r"and not|not the|down from|up from|reduced from|increased from|"
    r"as against|versus|vs\.?)\s+", re.IGNORECASE)


def parties(task_dir: Path | None = None) -> list[str]:
    """Les noms propres des deux camps, s'ils ont ete extraits.

    "Halcyon's demand for eighteen (18) months" ne porte aucune des marques
    ci-dessus : c'est le nom de la partie adverse qui attribue le chiffre.
    """
    if task_dir is None:
        try:
            from . import taskctx
            task_dir = taskctx.task_dir()
        except Exception:                                   # noqa: BLE001
            return []
    f = Path(task_dir) / "_parties.json"
    if not f.exists():
        return []
    try:
        d = json.loads(f.read_text())
    except Exception:                                       # noqa: BLE001
        return []
    out = []
    for k in ("them_org", "them"):
        v = str(d.get(k, "")).strip()
        if v:
            # le premier mot suffit et evite les virgules du nom de cabinet
            out.append(v.split(",")[0].split()[0].lower())
    return [w for w in out if len(w) > 3]


def split(text: str, leurs_noms: list[str] | None = None) -> tuple[list[str], list[str]]:
    """(propositions gardees, propositions ecartees)."""
    noms = [n.lower() for n in (leurs_noms if leurs_noms is not None else parties())]
    gardees, ecartees = [], []
    morceaux = []
    for part in _COUPE.split(text or ""):
        tete, *queue = _CONTRASTE.split(part, maxsplit=1)
        morceaux.append((tete, False))
        # Ce qui suit le marqueur decrit leur position, sauf si la tete elle-meme
        # est une acceptation ("we accept X rather than Y" reste a verifier).
        for q in queue:
            morceaux.append((q, not any(m in tete.lower() for m in NOTRES)))

    for part, force_ecart in morceaux:
        p = part.strip()
        if not p:
            continue
        if force_ecart:
            ecartees.append(p)
            continue
        low = p.lower()
        marque = (any(m in low for m in LEURS)
                  or any(f"{n}'s" in low or f"{n} propose" in low
                         or f"{n} request" in low or f"{n} demand" in low
                         for n in noms))
        if marque and not any(m in low for m in NOTRES):
            ecartees.append(p)
        else:
            gardees.append(p)
    # Tout ecarter voudrait dire que la contre-proposition n'enonce aucune
    # position, ce qui est un defaut de redaction, pas une exoneration : on
    # retombe alors sur le texte entier plutot que de ne rien controler.
    if not gardees:
        return ([text] if (text or "").strip() else []), ecartees
    return gardees, ecartees


def ours(text: str, leurs_noms: list[str] | None = None) -> str:
    return " ".join(split(text, leurs_noms)[0])
