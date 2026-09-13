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

    def size(self) -> int:
        n = 1
        for p in self.params.values():
            n *= len(p.options)
        return n

    def counters(self, min_concessions: int = 1, weights=None, limit: int = 12,
                 max_enum: int = 200_000, seed: int = 0):
        """Les contre-propositions credibles, sur la frontiere efficace.

        Credible = on leur concede au moins `min_concessions` parametres sur
        lesquels ils avaient bouge. Efficace = aucune autre assignation ne fait
        mieux pour nous sans faire moins bien pour eux.

        La frontiere est calculee par tri et balayage, en O(n log n). La version
        naive comparait chaque paire : sur douze variables a trois redactions,
        cela fait 5 x 10^11 comparaisons et ne rend jamais la main.
        """
        vt, wt = self.vector(self.template), self.vector(self.template, "theirs")
        n = self.size()
        if n <= max_enum:
            space = self.enumerate_all()
        else:
            space = self._sample(max_enum, seed)

        rows = []
        for a in space:
            conc = self.conceded(a)
            if len(conc) < min_concessions:
                continue
            ours = sub(self.vector(a), vt)
            theirs = sub(self.vector(a, "theirs"), wt)
            rows.append((a, ours, theirs,
                         total(ours, weights), total(theirs, weights)))
        if not rows:
            return []

        # balayage : trie sur notre total decroissant, garde ce qui ameliore leur total
        rows.sort(key=lambda r: (-r[3], -r[4]))
        front, best_theirs = [], float("-inf")
        for r in rows:
            if r[4] > best_theirs:
                front.append(r)
                best_theirs = r[4]
        front.sort(key=lambda r: -((r[4] / abs(r[3])) if r[3] else 0))
        return front[:limit]

    def _sample(self, k: int, seed: int):
        """Echantillon uniforme de l'espace quand il est trop grand pour etre
        parcouru. Une frontiere echantillonnee est une borne inferieure, pas la
        frontiere - le rapport le dit."""
        import random
        rng = random.Random(seed)
        ids = list(self.params)
        for _ in range(k):
            yield {i: rng.choice([o.id for o in self.params[i].options]) for i in ids}

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
    n_assign = c.size()

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
          "  score unique : il faut les deux vecteurs pour les voir.",
          ("  (espace echantillonne : borne inferieure de la frontiere)"
           if c.size() > 200_000 else ""), "",
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


# --- elicitation -----------------------------------------------------------

PARAM_PROMPT = """Voici notre modele de contrat et la version que la partie adverse
nous a renvoyee, marquee. Construis le contrat comme un OBJET PARAMETRIQUE.

Pour chaque point que leur markup a change de facon substantielle, donne une
VARIABLE. Entre huit et douze variables : prends les plus materielles, ignore le
reste. Pour chaque variable, donne un DOMAINE de deux a quatre redactions :

  - la notre, telle qu'elle figure dans notre modele        (id "ours")
  - la leur, telle qu'elle figure dans leur markup          (id "theirs")
  - une ou deux redactions INTERMEDIAIRES qu'un negociateur envisagerait et que
    ni l'un ni l'autre n'a encore proposees. Ce sont elles qui rendent la
    recherche utile : sans elles il n'y a que capituler ou refuser.

ATTENTION AU SCHEMA. Au niveau de la VARIABLE, "ours_option" et "theirs_option"
sont des identifiants d'option, donc des chaines. Au niveau de l'OPTION, "ours"
et "theirs" sont des vecteurs, donc des objets. Ne confonds pas les deux
niveaux : chaque option doit avoir un "id", un "text", et deux objets vecteurs.

Chaque redaction porte DEUX vecteurs sur ces six axes, jamais un score unique :

CONVENTION DE SIGNE, SANS EXCEPTION : un nombre POSITIF est MEILLEUR pour la
partie concernee, sur les six axes. Une charge administrative qui augmente est
donc NEGATIVE, pas positive.

  revenue        effet sur le revenu annuel, en milliers
  tail_risk      effet sur la perte extreme esperee, en milliers (plus de risque
                 = negatif)
  admin          effet sur le cout d'administration annuel, en milliers (plus de
                 cout = negatif)
  control        -5 a +5, qui decide
  flexibility    -5 a +5, marge de manoeuvre future
  enforceability -5 a +5, capacite a faire executer

  "ours"    l'effet POUR NOUS ({party})
  "theirs"  l'effet POUR EUX, autant qu'on puisse l'inferer de leur markup et
            de ce qu'il revele de leurs priorites

Les deux vecteurs sont le coeur de l'exercice. Deux facons de les rater :

  - remplir "theirs" comme l'oppose de "ours" : tout devient a somme nulle et il
    n'y a plus d'echange a trouver ;
  - appliquer un facteur constant, par exemple "theirs" = 2 x "ours" : les
    ratios sortent tous identiques et le classement ne reflete plus que la
    taille des clauses.

Les deux detruisent exactement l'information recherchee. Pour chaque redaction,
demande-toi ce que CETTE partie-la y gagne concretement, et pourquoi elle l'a
demandee plutot qu'une autre. Les valeurs doivent varier fortement d'une clause
a l'autre.

ILS NE PEUVENT PAS AVOIR RAISON SUR TOUT. Une partie qui negocie prend toujours
quelque chose : au moins un tiers de leurs demandes doivent leur rapporter MOINS
qu'elles ne nous coutent. Ce sont des prises de valeur pure, et les identifier
est la moitie de l'exercice. Si chacune de leurs demandes leur vaut plus qu'elle
ne nous coute, tu dis qu'il faut accepter leur markup en entier.

LE SIGNE, SUR UN EXEMPLE. Ils portent le plafond de responsabilite de 12 mois de
redevances a 2x le terme. Notre exposition AUGMENTE, donc c'est MAUVAIS pour
nous, donc "ours" porte {{"tail_risk": -410}} - un nombre NEGATIF. Leur
protection augmente, donc "theirs" porte {{"tail_risk": 150}} - un nombre
POSITIF. N'ecris jamais la QUANTITE de risque ; ecris ce qu'elle VAUT a la
partie concernee. Plus de risque, plus de cout, plus de contrainte : nombre
negatif, toujours.

REGLE DE COHERENCE, a verifier avant de repondre : ils ont marque ce contrat
pour se l'approprier. La quasi-totalite de leurs redactions doit donc etre
NEGATIVE pour nous et POSITIVE pour eux. Si tu obtiens l'inverse sur une
clause, tu as inverse un signe. Relis-la.

ORDRES DE GRANDEUR : ancre-toi sur la valeur annuelle du contrat telle qu'elle
ressort des documents. Aucun effet annuel ne devrait depasser cette valeur, et
le risque de queue est une perte ESPEREE - donc deja ponderee par sa
probabilite - pas le pire cas.

La redaction de reference ("ours") porte des zeros partout : tout est mesure
par rapport a notre modele.

Donne aussi les COUPLAGES : les paires de redactions dont l'effet conjoint n'est
pas la somme des effets separes - un plafond et son exception, une garantie et
la charge de la preuve qui l'accompagne. Cinq vrais couplages valent mieux que
vingt supposes.

Contexte : {context}

Reponds UNIQUEMENT par ce JSON :
{{"params": [{{"id":"cap","name":"Plafond de responsabilite","section":"11.1",
   "ours_option":"ours","theirs_option":"theirs",
   "options":[{{"id":"ours","text":"...","ours":{{}},"theirs":{{"revenue":0}}}},
              {{"id":"mid","text":"...","ours":{{"tail_risk":-150}},"theirs":{{"tail_risk":120}}}},
              {{"id":"theirs","text":"...","ours":{{"tail_risk":-410}},"theirs":{{"tail_risk":150}}}}]}}],
 "couplings": [{{"a":"cap","a_option":"theirs","b":"remedy","b_option":"theirs",
   "joint":{{"tail_risk":-95}},"note":"une phrase"}}]}}

=== NOTRE MODELE ===
{template}

=== LEUR MARKUP ===
{markup}"""


def _obj(raw: str) -> dict:
    import re as _re
    m = _re.search(r"\{.*\}", raw or "", _re.S)
    if not m:
        raise ValueError("pas d'objet JSON dans la reponse")
    return json.loads(m.group(0))


def _pick(r: dict, *keys):
    for k in keys:
        v = r.get(k)
        if isinstance(v, str):
            return v
    return None


def parse_model(raw: str) -> tuple[Contract, list[str]]:
    """Lit la reponse d'elicitation. Renvoie le contrat ET le motif de chaque rejet.

    Les rejets etaient comptes sans etre expliques : "10 variables mal formees"
    ne dit pas s'il manque un champ, si un identifiant ne correspond pas, ou si
    le modele a repondu dans une autre forme. Sans le motif on ne peut que
    deviner, et une elicitation coute un appel a chaque essai.
    """
    d = _obj(raw)
    rows = d.get("params") or d.get("variables") or []
    params, why = [], []
    for r in rows:
        pid = _pick(r, "id", "key") or "?"
        try:
            raw_opts = r.get("options") or r.get("domain") or []
            opts = [Option(id=str(o["id"]), text=o.get("text", ""),
                           ours=o.get("ours") or o.get("us") or {},
                           theirs=o.get("theirs") or o.get("them") or {})
                    for o in raw_opts if isinstance(o, dict) and "id" in o]
        except (KeyError, TypeError) as e:
            why.append(f"{pid}: options illisibles ({type(e).__name__})")
            continue
        if len(opts) < 2:
            why.append(f"{pid}: {len(opts)} option(s), il en faut au moins 2")
            continue
        ids = {o.id for o in opts}

        # les pointeurs. "ours"/"theirs" au niveau variable entraient en
        # collision avec "ours"/"theirs" au niveau option, qui sont des
        # vecteurs : on accepte les deux formes mais on privilegie les noms
        # non ambigus, et a defaut on deduit.
        a = _pick(r, "ours_option", "our_option", "baseline", "ours", "current")
        b = _pick(r, "theirs_option", "their_option", "theirs", "proposed")
        if a not in ids:
            a = "ours" if "ours" in ids else (opts[0].id if opts else None)
        if b not in ids:
            b = "theirs" if "theirs" in ids else (opts[-1].id if opts else None)
        if a not in ids or b not in ids or a == b:
            why.append(f"{pid}: impossible d'identifier notre redaction et la "
                       f"leur parmi {sorted(ids)}")
            continue
        params.append(Parameter(id=str(pid), name=r.get("name", pid),
                                section=str(r.get("section", "")),
                                options=opts, ours=a, theirs=b))

    pids = {p.id for p in params}
    cps = []
    for r in d.get("couplings") or []:
        try:
            if r["a"] in pids and r["b"] in pids:
                cps.append(Coupling(a=r["a"], a_option=r["a_option"], b=r["b"],
                                    b_option=r["b_option"],
                                    joint=r.get("joint") or {}, note=r.get("note", "")))
            else:
                why.append(f"couplage {r.get('a')}/{r.get('b')}: variable inconnue")
        except (KeyError, TypeError):
            why.append(f"couplage illisible: {str(r)[:60]}")
    return Contract(params, cps), why


def build_model(template: str, markup: str, llm, party: str, context: str,
                max_tokens: int = 24000, raw_out=None) -> Contract:
    raw = llm(PARAM_PROMPT.format(party=party, context=context,
                                  template=template, markup=markup),
              max_tokens=max_tokens)
    if raw_out is not None:
        raw_out.write_text(raw)          # toujours, pour pouvoir diagnostiquer
    c, why = parse_model(raw)
    if why:
        print(f"  {len(why)} entrees ecartees :", flush=True)
        for w in why[:10]:
            print(f"    {w}", flush=True)
    return c


def load(raw: str) -> Contract:
    d = json.loads(raw)
    params = [Parameter(id=p["id"], name=p["name"], section=p.get("section", ""),
                        ours=p["ours"], theirs=p["theirs"],
                        options=[Option(**o) for o in p["options"]])
              for p in d["params"]]
    return Contract(params, [Coupling(**c) for c in d.get("couplings", [])])


def model_sanity(c: Contract, weights=None) -> list[str]:
    """Les facons dont une elicitation produit un modele qui ne dit rien.

    Aucune ne se voit dans les chiffres pris un par un ; toutes se voient dans
    leur distribution. Ce sont des tests sur le MODELE, pas sur le contrat.
    """
    import statistics as st
    out = []
    rows = single_trades(c, weights)
    theirs = [r for r in rows if r[5]]                  # leurs demandes seulement
    ratios = [r[4] for r in theirs if r[4] != float("inf")]

    # 1. un multiplicateur constant. Une symetrie donne des ratios tous a 1, un
    #    multiplicateur uniforme les donne tous a k. Ce qui compte est l'absence
    #    de dispersion, pas la valeur autour de laquelle ils se serrent.
    if len(ratios) >= 4:
        cv = st.pstdev(ratios) / abs(st.mean(ratios)) if st.mean(ratios) else 0
        if cv < 0.25:
            out.append(
                f"RATIOS UNIFORMES : leurs {len(ratios)} demandes sortent a un "
                f"ratio median de {st.median(ratios):.2f} avec un coefficient de "
                f"variation de {cv:.3f}.\n  Le modele a applique un facteur "
                f"constant au lieu d'inferer leurs priorites clause par clause. "
                f"Sans\n  dispersion il n'y a aucun echange a trouver : le "
                f"classement ne reflete que les tailles.")

    # 2. tout concede. Si chacune de leurs demandes leur rapporte plus qu'elle
    #    ne nous coute, le modele dit d'accepter le markup en entier.
    if ratios and all(x > 1 for x in ratios):
        out.append(
            f"AUCUNE PRISE DE VALEUR : les {len(ratios)} demandes ont toutes un "
            f"ratio > 1.\n  Le modele conclut qu'il faut tout accepter. Une "
            f"partie adverse qui negocie prend\n  toujours quelque chose : une "
            f"elicitation credible doit trouver des ratios < 1.")

    # 3. coherence des signes. Une partie qui marque un contrat prend quelque
    #    chose : si son markup ameliore NOTRE position, un axe a ete rempli
    #    comme une QUANTITE (plus de risque = nombre positif) au lieu d'une
    #    VALEUR (plus de risque = nombre negatif). C'est le defaut le plus
    #    frequent, et il inverse la liste des concessions a refuser.
    vt = c.vector(c.template)
    dr = total(sub(c.vector(c.markup), vt), weights)
    gagnantes = []
    for pid, p in c.params.items():
        if p.ours == p.theirs:
            continue
        a = dict(c.template)
        a[pid] = p.theirs
        if total(sub(c.vector(a), vt), weights) > 0:
            gagnantes.append(p.name)
    n_moved = sum(1 for p in c.params.values() if p.ours != p.theirs)
    if n_moved and (dr > 0 or len(gagnantes) > n_moved * 0.4):
        out.append(
            f"SIGNES INCOHERENTS : leur markup ameliore notre position de "
            f"{dr:+.0f} au total,\n  et {len(gagnantes)}/{n_moved} de leurs "
            f"demandes nous seraient profitables"
            + (f" ({', '.join(x[:24] for x in gagnantes[:4])})" if gagnantes else "")
            + ".\n  Une partie adverse ne demande pas des clauses qui nous "
              f"avantagent. Un axe a ete\n  rempli comme une quantite - plus de "
              f"risque = nombre positif - au lieu d'une\n  valeur. La liste des "
              f"concessions a refuser est alors inversee.")
    return out


def model_notes(c: Contract, weights=None) -> list[str]:
    """Observations sur le contrat lui-meme. A lire, mais pas bloquantes.

    La dominance d'un axe en est une : un plafond de responsabilite ECRASE
    reellement le reste dans beaucoup de contrats, et ce n'est pas un defaut
    d'elicitation. Les echanges se trouvent dans les RATIOS clause par clause,
    qui restent informatifs meme quand une seule dimension porte la derive.
    """
    out = []
    d = c.drift()
    w = weights or DEFAULT_WEIGHTS
    mag = {k: abs(d[k]) * w.get(k, 1.0) for k in DIMENSIONS}
    tot = sum(mag.values())
    if tot:
        top = max(mag.items(), key=lambda kv: kv[1])[0]
        share = max(mag.values()) / tot
        if share > 0.7:
            out.append(f"{top} porte {share:.0%} de la derive ponderee. Les six "
                       f"axes se comportent ici presque comme un seul ; les "
                       f"echanges restent dans les ratios.")
    used = [k for k in DIMENSIONS
            if any(abs(o.vec().get(k, 0)) > 1e-9 or abs(o.vec("theirs").get(k, 0)) > 1e-9
                   for p in c.params.values() for o in p.options)]
    if len(used) < 3:
        out.append(f"seuls {len(used)} axes sur {len(DIMENSIONS)} sont utilises "
                   f"({', '.join(used)}). Le reste du vecteur est inerte.")
    n_mid = sum(1 for p in c.params.values()
                if any(o.id not in (p.ours, p.theirs) for o in p.options))
    if n_mid < len(c.params) / 2:
        out.append(f"{n_mid}/{len(c.params)} variables ont une redaction "
                   f"intermediaire. Sans elles il n'y a que ceder ou refuser, "
                   f"et la recherche n'a rien a chercher.")
    return out


def sanity_report(c: Contract, weights=None) -> str:
    w = model_sanity(c, weights)
    notes = model_notes(c, weights)
    tail = ("\n\nA NOTER (non bloquant) :\n  " + "\n  ".join(notes)) if notes else ""
    if not w:
        return "controles du modele : aucun signal d'alerte." + tail
    return ("CE MODELE N'EST PROBABLEMENT PAS EXPLOITABLE\n" + "=" * 78 + "\n\n"
            + "\n\n".join(w) + tail)


# retro-compatibilite
def zero_sum_warning(c: Contract, weights=None) -> str:
    w = model_sanity(c, weights)
    return "\n\n".join(w)
