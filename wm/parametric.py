"""
Le contrat comme OBJET PARAMETRIQUE, pas comme liste de clauses annotees.

Ce que ce module fait et que rien d'autre dans ce depot ne faisait :

  1. le contrat entier est un ensemble de VARIABLES, chacune avec un DOMAINE -
     les redactions possibles, pas seulement la notre et la leur ;
  2. chaque valeur possible porte un VECTEUR, pas un scalaire. Un plafond de
     responsabilite ne vaut pas "-3". Il coute du revenu, ajoute du risque de
     queue, retire du controle, et chacune de ces dimensions se lit separement ;
  3. le contrat a donc une valeur VECTORIELLE V(assignation), et le markup du
     client est un autre point dans le meme espace. La difference n'est pas un
     nombre : c'est une direction. Elle dit SUR QUELLE DIMENSION ils ont pris ;
  4. on peut donc CHERCHER. Chaque assignation du produit des domaines est un
     contrat possible ; on les evalue toutes et on garde celles qui reequilibrent.

Et le point qui rend la chose utile plutot que decorative : chaque option porte
AUSSI une estimation de sa valeur POUR EUX. Une negociation n'est pas un jeu a
somme nulle clause par clause, et les seules concessions qui valent la peine
d'etre offertes sont celles ou leur gain depasse notre perte. C'est le
logrolling de Lax & Sebenius, et il exige exactement cette representation
multidimensionnelle : sur un score unique il est invisible.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field, asdict


# Les axes. Un contrat ne se reduit pas a un prix : il alloue du revenu, du
# risque, du controle, de la flexibilite, de la charge administrative et de la
# capacite a faire executer. Ces six-la sont un choix de modelisation, pas une
# verite - mais ils sont lisibles, ce qu'un embedding appris n'est pas.
DIMENSIONS = ("revenue", "tail_risk", "control", "flexibility", "admin", "enforceability")

# Les axes ne sont PAS dans la meme unite. revenue, tail_risk et admin sont en
# milliers d'euros par an ; control, flexibility et enforceability sont des
# echelles relatives. Les additionner sans le dire est exactement l'erreur que
# ce module existe pour eviter, donc la conversion est declaree ici et non
# enfouie dans une somme.
DEFAULT_WEIGHTS = {"revenue": 1.0, "tail_risk": 1.0, "admin": 1.0,
                   "control": 25.0, "flexibility": 15.0, "enforceability": 20.0}


@dataclass
class Option:
    """Une redaction possible d'un parametre."""
    id: str
    text: str
    ours: dict[str, float] = field(default_factory=dict)    # effet sur NOTRE vecteur
    theirs: dict[str, float] = field(default_factory=dict)  # effet sur LE LEUR

    def vec(self, whose: str = "ours") -> dict[str, float]:
        src = self.ours if whose == "ours" else self.theirs
        return {d: float(src.get(d, 0.0)) for d in DIMENSIONS}


@dataclass
class Parameter:
    """Une variable du contrat, avec son domaine."""
    id: str
    name: str
    section: str
    options: list[Option]
    ours: str              # id de l'option dans NOTRE modele
    theirs: str            # id de l'option dans LEUR markup

    def option(self, oid: str) -> Option:
        for o in self.options:
            if o.id == oid:
                return o
        raise KeyError(f"{self.id}: option inconnue {oid!r}")


@dataclass
class Coupling:
    """Deux parametres dont l'effet conjoint n'est pas la somme des effets."""
    a: str
    a_option: str
    b: str
    b_option: str
    joint: dict[str, float]
    note: str = ""


def zero() -> dict[str, float]:
    return {d: 0.0 for d in DIMENSIONS}


def add(x: dict[str, float], y: dict[str, float]) -> dict[str, float]:
    return {d: x.get(d, 0.0) + y.get(d, 0.0) for d in DIMENSIONS}


def sub(x: dict[str, float], y: dict[str, float]) -> dict[str, float]:
    return {d: x.get(d, 0.0) - y.get(d, 0.0) for d in DIMENSIONS}


def norm(x: dict[str, float], weights: dict[str, float] | None = None) -> float:
    w = weights or DEFAULT_WEIGHTS
    return sum(w.get(d, 1.0) * x.get(d, 0.0) ** 2 for d in DIMENSIONS) ** 0.5


def total(x: dict[str, float], weights: dict[str, float] | None = None) -> float:
    """Scalarisation. A n'utiliser que pour trier, jamais pour conclure."""
    w = weights or DEFAULT_WEIGHTS
    return sum(w.get(d, 1.0) * x.get(d, 0.0) for d in DIMENSIONS)


class Contract:
    def __init__(self, params: list[Parameter], couplings: list[Coupling] | None = None):
        self.params = {p.id: p for p in params}
        self.couplings = couplings or []

    # --- evaluation -------------------------------------------------------
    def vector(self, assignment: dict[str, str], whose: str = "ours") -> dict[str, float]:
        v = zero()
        for pid, oid in assignment.items():
            v = add(v, self.params[pid].option(oid).vec(whose))
        for c in self.couplings:
            if assignment.get(c.a) == c.a_option and assignment.get(c.b) == c.b_option:
                v = add(v, c.joint if whose == "ours" else zero())
        return v

    @property
    def template(self) -> dict[str, str]:
        return {p.id: p.ours for p in self.params.values()}

    @property
    def markup(self) -> dict[str, str]:
        return {p.id: p.theirs for p in self.params.values()}

    def drift(self) -> dict[str, float]:
        """V(leur markup) - V(notre modele). Une DIRECTION, pas un nombre."""
        return sub(self.vector(self.markup), self.vector(self.template))

    def conceded(self, assignment: dict[str, str]) -> list[str]:
        return [p for p in assignment if assignment[p] == self.params[p].theirs
                and self.params[p].theirs != self.params[p].ours]

    # --- recherche --------------------------------------------------------
    def enumerate_all(self):
        ids = list(self.params)
        domains = [[o.id for o in self.params[i].options] for i in ids]
        for combo in itertools.product(*domains):
            yield dict(zip(ids, combo))

    def counters(self, min_concessions: int = 1, weights=None, limit: int = 12):
        """Les contre-propositions credibles, sur la frontiere efficace.

        Credible = on leur concede au moins `min_concessions` parametres sur
        lesquels ils avaient bouge. Efficace = aucune autre assignation ne fait
        mieux POUR NOUS sans faire moins bien POUR EUX.
        """
        vt, wt = self.vector(self.template), self.vector(self.template, "theirs")
        rows = []
        for a in self.enumerate_all():
            conc = self.conceded(a)
            if len(conc) < min_concessions:
                continue
            ours = sub(self.vector(a), vt)              # notre perte vs notre modele
            theirs = sub(self.vector(a, "theirs"), wt)  # leur gain vs notre modele
            rows.append((a, ours, theirs,
                         total(ours, weights), total(theirs, weights)))

        # frontiere de Pareto sur (notre total, leur total)
        front = []
        for r in rows:
            if not any(o[3] >= r[3] and o[4] >= r[4] and (o[3] > r[3] or o[4] > r[4])
                       for o in rows):
                front.append(r)
        front.sort(key=lambda r: -((r[4] / abs(r[3])) if r[3] else 0))
        return front[:limit]


# --- rendu -----------------------------------------------------------------

def show_vector(v: dict[str, float], width: int = 9) -> str:
    return "".join(f"{v.get(d, 0.0):>{width}.1f}" for d in DIMENSIONS)


def header(width: int = 9) -> str:
    return "".join(f"{d[:width-1]:>{width}}" for d in DIMENSIONS)


def single_trades(c: Contract, weights=None):
    """Chaque concession possible, prise seule : notre perte, leur gain, ratio.

    C'est la liste qu'un negociateur veut vraiment. Un ratio eleve est une
    monnaie d'echange : ils y tiennent plus que cela ne nous coute. Un ratio
    bas est une concession pure - on la garde pour la fin ou on la refuse.
    """
    vt, wt = c.vector(c.template), c.vector(c.template, "theirs")
    rows = []
    for pid, p in c.params.items():
        for o in p.options:
            if o.id == p.ours:
                continue
            a = dict(c.template)
            a[pid] = o.id
            lose = total(sub(c.vector(a), vt), weights)
            gain = total(sub(c.vector(a, "theirs"), wt), weights)
            ratio = gain / abs(lose) if lose else float("inf")
            rows.append((p.name, o.text, lose, gain, ratio, o.id == p.theirs))
    return sorted(rows, key=lambda r: -r[4])


def report(c: Contract, weights=None, min_concessions: int = 2) -> str:
    vt, vm = c.vector(c.template), c.vector(c.markup)
    d = c.drift()
    n_assign = 1
    for p in c.params.values():
        n_assign *= len(p.options)

    L = ["LE CONTRAT COMME OBJET PARAMETRIQUE", "=" * 78, "",
         f"{len(c.params)} variables, {n_assign:,} redactions possibles, "
         f"{len(DIMENSIONS)} dimensions.", "",
         f"{'':<26}{header()}", "-" * 78,
         f"{'notre modele':<26}{show_vector(vt)}",
         f"{'leur markup':<26}{show_vector(vm)}",
         f"{'DERIVE':<26}{show_vector(d)}", ""]

    worst = sorted(DIMENSIONS, key=lambda x: d[x])[:2]
    L += [f"  Ils n'ont pas pris 'de la valeur'. Ils ont pris sur "
          f"{' et '.join(worst)}.",
          "  Un score unique aurait additionne ces colonnes et perdu cela.", ""]

    L += ["PAR VARIABLE - ce que leur redaction change, dimension par dimension",
          "-" * 78, f"{'':<26}{header()}"]
    for p in c.params.values():
        if p.ours == p.theirs:
            continue
        diff = sub(p.option(p.theirs).vec(), p.option(p.ours).vec())
        L.append(f"  {p.name[:22]:<24}{show_vector(diff)}")

    singles = single_trades(c, weights)
    L += ["", "CHAQUE CONCESSION, PRISE SEULE", "-" * 78,
          f"  {'variable':<24}{'redaction':<34}{'nous':>8}{'eux':>7}{'ratio':>7}"]
    for name, text, lose, gain, ratio, is_theirs in singles:
        mark = " <-- leur demande" if is_theirs else ""
        r = f"{ratio:.2f}" if ratio != float("inf") else "inf"
        L.append(f"  {name[:22]:<24}{text[:32]:<34}{lose:>8.0f}{gain:>7.0f}"
                 f"{r:>7}{mark}")
    refuse = [r for r in singles if r[5] and r[4] < 0.6]
    if refuse:
        L += ["", "  A REFUSER - ils demandent, cela nous coute bien plus que cela ne",
              "  leur rapporte. Ce sont des prises de valeur, pas des echanges :"]
        for name, text, lose, gain, ratio, _ in refuse:
            L.append(f"    {name[:26]:<28} nous {lose:>7.0f}  eux {gain:>6.0f}  "
                     f"ratio {ratio:.2f}")

    L += ["", "COMBINAISONS SUR LA FRONTIERE EFFICACE", "-" * 78,
          "  Les echanges ou LEUR gain depasse NOTRE perte. Invisible sur un",
          "  score unique : il faut les deux vecteurs pour les voir.", "",
          f"  {'concessions':<34}{'nous':>8}{'eux':>8}{'ratio':>8}"]
    for a, ours, theirs, to, tt in c.counters(min_concessions, weights):
        names = ", ".join(c.params[p].name[:16] for p in c.conceded(a))[:32]
        ratio = (tt / abs(to)) if to else float("inf")
        L.append(f"  {names:<34}{to:>8.1f}{tt:>8.1f}"
                 f"{(f'{ratio:.2f}' if ratio != float('inf') else '  inf'):>8}")
    L += ["", "  ratio > 1 : ils gagnent plus que nous ne perdons. A offrir en premier.",
          "  ratio < 1 : concession pure. A garder pour la fin, ou a refuser."]
    return "\n".join(L)


def dump(c: Contract) -> str:
    return json.dumps({"params": [asdict(p) for p in c.params.values()],
                       "couplings": [asdict(x) for x in c.couplings]}, indent=2)
