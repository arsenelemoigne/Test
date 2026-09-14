"""
Le simulateur de negociation : deux camps, un contrat parametrique, des offres
qui sont des ASSIGNATIONS (parametre -> redaction), jamais du texte libre.

Pourquoi ce banc d'essai et pas Harvey LAB : le rubric de LAB est le mandat
reecrit en cases a cocher - la bonne reponse y est donnee, il n'y a rien a
negocier. Ici la bonne reponse n'existe pas d'avance. Chaque camp a une utilite
(le vecteur elicite, avec des poids qui lui sont propres), un seuil de rupture,
un temperament, et l'issue se mesure exactement : ce que nous gardons, ce
qu'ils obtiennent, si l'accord est efficace, en combien de tours.

Ce que le simulateur tient pour acquis, et qu'il faut garder en tete :

  - Les utilites sont DECLAREES (elicitees), pas mesurees. En simulation cela
    suffit : les deux camps jouent sur des preferences connues du code, donc
    l'issue est calculable. Le transfert au reel est une autre question.
  - Notre croyance sur eux est le vecteur elicite ; leur vraie utilite en est
    une perturbation (poids bruites, parfois une fixation sur un point). Notre
    modele d'eux est donc faux quelque part, comme dans la vie.
  - Les offres sont des identifiants de redaction. Une recompense lue dans du
    texte se contourne linguistiquement - on l'a vu, un modele a reecrit "50%"
    en "50 percent" pour satisfaire un regex. Une recompense calculee sur des
    identifiants ne s'obtient qu'en choisissant de meilleures redactions.

Trois politiques pour notre camp, comparees sur les MEMES adversaires :

  algo       aucun modele de langage. Concede dans l'ordre des ratios (leur
             gain cru / notre perte) jusqu'a son seuil du tour. 200 lignes.
  llm_raw    le modele voit les redactions, l'historique, un mandat en prose.
             Aucun chiffre.
  llm_value  la meme chose, plus la couche parametrique : couts, gains crus,
             ratios, valeur des offres. C'est l'hypothese du projet.

Si algo bat llm_raw, un programme bat un negociateur LLM. Si llm_value bat
llm_raw, la couche de valeur aide. Si rien ne bat llm_raw, l'idee ne tient pas
- au moins sous cette forme, et c'est ce qu'on voulait savoir.
"""

from __future__ import annotations

import json
import math
import random
import statistics
from dataclasses import dataclass, field, asdict

from . import parametric as pm

DEFAULT_ROUNDS = 6

# Les poids de scalarisation. None = ceux du modele elicite ; le pont
# claim_bridge en fournit d'autres, tous en dollars.
WEIGHTS: dict | None = None


def _w() -> dict:
    return WEIGHTS or pm.DEFAULT_WEIGHTS


# --- les camps -----------------------------------------------------------

@dataclass
class Side:
    name: str                       # "us" | "them"
    whose: str                      # "ours" | "theirs" : quel vecteur des redactions
    weights: dict                   # poids par dimension (le temperament les bruite)
    ideal: dict                     # l'assignation qu'il signerait demain
    aspiration: float               # u(ideal)
    reservation: float              # en dessous, il prefere ne pas signer
    beta: float = 1.0               # Faratin : <1 dur (boulware), 1 lineaire, >1 conciliant
    persona: str = ""               # prose, pour un adversaire joue par un LLM
    label: str = ""                 # "dur" | "neutre" | "conciliant"
    fixation: str = ""              # un parametre qui compte bien plus que prevu

    def target(self, t: int, T: int) -> float:
        """Le seuil d'acceptation au tour t : part de l'aspiration, atteint la
        reservation au dernier tour. C'est la tactique temporelle classique."""
        if T <= 1:
            return self.reservation
        x = min(1.0, max(0.0, t / (T - 1)))
        return self.aspiration - (self.aspiration - self.reservation) * (x ** (1.0 / self.beta))


def utility(c: pm.Contract, side: Side, a: dict) -> float:
    ref = pm.total(c.vector(c.template, side.whose), side.weights)
    return pm.total(c.vector(a, side.whose), side.weights) - ref


def belief(c: pm.Contract, whose: str, a: dict) -> float:
    """Ce que le modele elicite dit de la valeur d'une assignation pour un camp.
    C'est ce que l'AUTRE camp croit ; la verite est dans Side.weights."""
    ref = pm.total(c.vector(c.template, whose), _w())
    return pm.total(c.vector(a, whose), _w()) - ref


def make_us(c: pm.Contract, budget: float = 0.35, beta: float = 1.0) -> Side:
    """Notre camp : l'utilite elicitee telle quelle, un budget de concession
    exprime en part de ce que leur markup nous prend."""
    loss = belief(c, "ours", c.markup)           # negatif
    return Side(name="us", whose="ours", weights=dict(_w()),
                ideal=dict(c.template), aspiration=0.0,
                reservation=budget * loss, beta=beta, label="nous")


PERSONAS = {
    "dur": ("Tu es un negociateur dur. Tu cedes tard et peu, tu rappelles que ton "
            "client a d'autres options, tu ne justifies pas tes demandes.", 0.45),
    "neutre": ("Tu negocies de facon professionnelle et previsible : tu cedes "
               "progressivement, tu expliques brievement tes priorites.", 1.0),
    "conciliant": ("Tu tiens a conclure. Tu cedes volontiers sur ce qui compte "
                   "peu pour ton client et tu cherches les echanges gagnant-gagnant.", 2.5),
}


def make_them(c: pm.Contract, seed: int) -> Side:
    """Un adversaire tire au sort : temperament, seuil, poids bruites, et une
    fois sur deux une fixation - un point qui vaut trois fois ce que notre
    modele croyait. C'est la que notre croyance est fausse."""
    rng = random.Random(1000 + seed)
    label = rng.choice(list(PERSONAS))
    persona, beta = PERSONAS[label]
    weights = {d: w * math.exp(rng.gauss(0.0, 0.4)) for d, w in _w().items()}
    fixation = ""
    if rng.random() < 0.5:
        moved = [p for p in c.params.values() if p.theirs != p.ours]
        if moved:
            fixation = rng.choice(moved).id
    side = Side(name="them", whose="theirs", weights=weights, ideal=dict(c.markup),
                aspiration=0.0, reservation=0.0, beta=beta, persona=persona,
                label=label, fixation=fixation)
    # la fixation s'exprime dans utility_fix, en triplant l'effet de CE
    # parametre - c'est un point qui compte, pas un axe
    side.aspiration = utility_fix(c, side, c.markup)
    side.reservation = rng.uniform(0.25, 0.7) * side.aspiration
    return side


def utility_fix(c: pm.Contract, side: Side, a: dict) -> float:
    """utility(), plus la fixation : l'effet du parametre fixe compte triple."""
    u = utility(c, side, a)
    if side.fixation and side.fixation in c.params:
        p = c.params[side.fixation]
        seul = {side.fixation: a.get(side.fixation, p.ours)}
        ref = {side.fixation: p.ours}
        extra = (pm.total(c.vector(seul, side.whose), side.weights)
                 - pm.total(c.vector(ref, side.whose), side.weights))
        u += 2.0 * extra
    return u


def true_u(c: pm.Contract, side: Side, a: dict) -> float:
    return utility_fix(c, side, a)


# --- la politique algorithmique -------------------------------------------

def concede_greedy(c: pm.Contract, side: Side, other_whose: str, start: dict,
                   floor: float) -> dict:
    """Depuis `start`, applique les changements de redaction dans l'ordre du
    ratio (gain cru de l'autre / notre perte) tant que notre utilite reste au
    dessus de `floor`. Les changements qui nous rapportent aussi passent en
    premier : ce ne sont pas des concessions, ce sont des ameliorations."""
    a = dict(start)
    u = true_u(c, side, a)
    while True:
        best, best_key = None, None
        ob = belief(c, other_whose, a)
        for pid, p in c.params.items():
            for o in p.options:
                if o.id == a[pid]:
                    continue
                b = dict(a)
                b[pid] = o.id
                du = true_u(c, side, b) - u
                dother = belief(c, other_whose, b) - ob
                if dother <= 1e-9:
                    continue
                if u + du < floor - 1e-9:
                    continue
                key = (float("inf") if du >= -1e-9 else dother / abs(du))
                if best_key is None or key > best_key:
                    best, best_key = (pid, o.id, du), key
        if best is None:
            break
        pid, oid, du = best
        a[pid] = oid
        u += du
    return a


class AlgoPolicy:
    name = "algo"
    calls = 0

    def __init__(self, c: pm.Contract, side: Side, other: Side):
        self.c, self.side, self.other = c, side, other

    def accepts(self, offer: dict, t: int, T: int) -> bool:
        return true_u(self.c, self.side, offer) >= self.side.target(t, T) - 1e-9

    def propose(self, t: int, T: int, history: list) -> dict:
        floor = self.side.target(t, T)
        # partir de notre ideal, mais prendre gratuitement ce que leur derniere
        # offre nous donne deja (un point ou ils ont bouge vers nous)
        start = dict(self.side.ideal)
        last = next((h["offer"] for h in reversed(history) if h["by"] != self.side.name), None)
        if last:
            for pid in start:
                b = dict(start)
                b[pid] = last[pid]
                if true_u(self.c, self.side, b) >= true_u(self.c, self.side, start) - 1e-9:
                    start[pid] = last[pid]
        return concede_greedy(self.c, self.side, self.other.whose, start, floor)


# --- les politiques LLM ---------------------------------------------------

def _menu(c: pm.Contract, side: Side, with_value: bool, other_whose: str) -> str:
    L = []
    trades = {(r[0], r[1]): r for r in pm.single_trades(c)} if with_value else {}
    for p in c.params.values():
        L.append(f"{p.id}  {p.name}  [section {p.section}]")
        for o in p.options:
            tag = ""
            if o.id == p.ours:
                tag = "  (notre modele)" if side.name == "us" else "  (leur modele)"
            elif o.id == p.theirs:
                tag = "  (leur markup)" if side.name == "us" else "  (notre markup)"
            line = f"    {o.id}: {o.text}{tag}"
            if with_value and o.id != p.ours:
                r = trades.get((p.name, o.text))
                if r:
                    mine = r[2] if side.name == "us" else r[3]
                    theirs = r[3] if side.name == "us" else r[2]
                    line += f"   [nous {mine:+.0f}, eux (estime) {theirs:+.0f}]"
            L.append(line)
        L.append("")
    return "\n".join(L)


def _hist(c: pm.Contract, side: Side, history: list) -> str:
    if not history:
        return "(aucune offre encore)"
    L = []
    for h in history:
        who = "NOUS" if h["by"] == side.name else "EUX"
        diff = [f"{pid}={oid}" for pid, oid in h["offer"].items()
                if oid != c.params[pid].ours]
        L.append(f"tour {h['t']}  {who} : " + (", ".join(diff) if diff else "(le modele tel quel)"))
    return "\n".join(L)


NEG_PROMPT = """Tu negocies un contrat pour {who}. L'autre partie est {other}.
{persona}

Le contrat est decrit comme une liste de points. Chaque point a plusieurs
redactions possibles, identifiees par un code. Une offre est un choix de
redaction pour CHAQUE point.

{mandate}

POINTS ET REDACTIONS
{menu}

HISTORIQUE DES OFFRES (seuls les points qui s'ecartent du modele sont listes)
{hist}

Tour {t} sur {T}. Sans accord au dernier tour, il n'y a pas de contrat.
{value}
Reponds par UN objet JSON, sans autre texte, de l'une de ces deux formes :
  {{"accept": true}}                       pour accepter la derniere offre adverse
  {{"offer": {{"<point>": "<code>", ...}}}}    pour faire une contre-offre COMPLETE
Un code inconnu est ignore et le point garde sa redaction precedente.
"""


def _mandate_prose(c: pm.Contract, side: Side) -> str:
    """Le mandat SANS chiffres : les priorites en mots, le budget en proportion.
    C'est ce qu'un juriste recoit d'ordinaire."""
    mag = []
    for p in c.params.values():
        a, b = {p.id: p.ours}, {p.id: p.theirs}
        mag.append((abs(pm.total(c.vector(b, side.whose), side.weights)
                        - pm.total(c.vector(a, side.whose), side.weights)), p.name))
    top = [n for _, n in sorted(mag, reverse=True)[:5]]
    if side.name == "us":
        part = abs(side.reservation / belief(c, "ours", c.markup)) if belief(c, "ours", c.markup) else 0
        return (f"MANDAT : leur markup nous prend beaucoup. Tu peux ceder au plus "
                f"environ {part:.0%} de ce qu'il nous prend, pas davantage. Les points "
                f"qui comptent le plus pour ton client : {', '.join(top)}. Prefere les "
                f"echanges ou l'autre partie gagne plus que ce que tu perds.")
    part = side.reservation / side.aspiration if side.aspiration else 0
    return (f"MANDAT : ton markup est ta position d'ouverture. Tu dois obtenir au "
            f"moins environ {part:.0%} de ce qu'il vaut pour ton client. Les points "
            f"qui comptent le plus : {', '.join(top)}.")


def _value_block(c: pm.Contract, side: Side, other: Side, history: list) -> str:
    """La couche parametrique : ce que valent les offres sur la table, pour nous
    et pour eux (cru), et la ligne a ne pas franchir."""
    L = ["VALEUR (calculee par le modele parametrique ; 0 = notre modele)"]
    L.append(f"  notre seuil de rupture : {side.reservation:+.0f}")
    for h in history[-4:]:
        who = "nous" if h["by"] == side.name else "eux"
        mine = true_u(c, side, h["offer"])
        theirs = belief(c, other.whose, h["offer"])
        L.append(f"  tour {h['t']} {who:<4} : pour nous {mine:+.0f}   pour eux (estime) {theirs:+.0f}")
    L.append("  Une bonne contre-offre leur rend beaucoup (estime) pour peu de nous.")
    return "\n".join(L) + "\n"


def _last_object(text: str) -> dict | None:
    """Le dernier objet JSON equilibre du texte qui porte accept ou offer."""
    out, debut, prof = None, None, 0
    dans, esc = False, False
    for i, ch in enumerate(text or ""):
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
        elif ch == "{":
            if prof == 0:
                debut = i
            prof += 1
        elif ch == "}" and prof:
            prof -= 1
            if prof == 0 and debut is not None:
                try:
                    o = json.loads(text[debut:i + 1])
                except ValueError:
                    o = None
                if isinstance(o, dict) and ("accept" in o or "offer" in o):
                    out = o
                debut = None
    return out


class LLMPolicy:
    def __init__(self, c: pm.Contract, side: Side, other: Side, call, with_value: bool,
                 who: str = "", other_name: str = ""):
        self.c, self.side, self.other, self.call = c, side, other, call
        self.with_value = with_value
        self.name = "llm_value" if with_value else "llm_raw"
        self.who, self.other_name = who or side.name, other_name or other.name
        self.calls, self.invalid, self.unparsed = 0, 0, 0
        self._rappel = ""
        self._pending_accept = False
        self._last_offer: dict | None = None

    def _ask(self, t: int, T: int, history: list) -> dict | None:
        prompt = NEG_PROMPT.format(
            who=self.who, other=self.other_name, persona=self.side.persona,
            mandate=_mandate_prose(self.c, self.side),
            menu=_menu(self.c, self.side, self.with_value, self.other.whose),
            hist=_hist(self.c, self.side, history), t=t + 1, T=T,
            value=((_value_block(self.c, self.side, self.other, history)
                    if self.with_value else "")
                   + (getattr(self, "_rappel", "") + "\n" if getattr(self, "_rappel", "") else "")))
        self.calls += 1
        raw = self.call(prompt, max_tokens=6000) or ""
        self._rappel = ""          # le rappel vaut pour UN appel, pas pour la suite
        o = _last_object(raw)
        if o is None:
            self.unparsed += 1
        return o

    def accepts(self, offer: dict, t: int, T: int) -> bool:
        # Un seul appel par tour : la reponse du modele est soit "accept" soit
        # une contre-offre, que propose() rendra ensuite sans rappeler.
        history = self._history_ref
        o = self._ask(t, T, history)
        if o and o.get("accept") is True:
            self._last_offer = None
            return True
        self._last_offer = o.get("offer") if o else None
        return False

    def propose(self, t: int, T: int, history: list) -> dict:
        prev = next((h["offer"] for h in reversed(history) if h["by"] == self.side.name),
                    dict(self.side.ideal))
        a = dict(prev)
        o = self._last_offer
        if not isinstance(o, dict):
            # rien d'exploitable : on repete notre offre precedente
            return a
        for pid, oid in o.items():
            if pid in self.c.params and any(x.id == oid for x in self.c.params[pid].options):
                a[pid] = oid
            else:
                self.invalid += 1
        return a



# --- la couche de faisabilite ------------------------------------------------

class Guarded:
    """Le modele propose, un controle DETERMINISTE dispose.

    C'est le remede que la litterature documente pour le defaut qu'on a mesure.
    TERMS-Bench (arXiv:2605.13909, 2026) fait de l'environnement le verificateur
    et compte par episode les violations de prix plancher et de rationalite
    individuelle ; il rapporte que des agents "capitulent devant une offre
    defavorable des les premiers tours, divulguent leur limite, ou violent leur
    propre budget pour forcer la transaction". Nous mesurons exactement cela :
    llm_raw conclut sous son seuil de reserve 3 a 4 fois sur 6 a 12. La reponse
    rapportee n'est pas un meilleur prompt ni un juge LLM - "LLM-as-a-Judge Is
    Not an Oracle" (arXiv:2609.02246) - mais un verificateur exterieur.

    Deux regles, et elles ne se discutent pas :
      - ACCEPTER une offre sous le seuil de reserve est refuse. Le modele
        voulait signer ; on ne signe pas.
      - PROPOSER une assignation sous le seuil du tour est refuse, et le
        modele est rappele avec la contrainte violee. Apres `retries` essais,
        on prend l'offre faisable la plus proche de ce qu'il voulait : pour
        chaque point ou il s'ecartait, on rend sa redaction tant que le seuil
        tient, dans l'ordre du ratio - donc en gardant ce qui leur rapporte
        le plus par unite de ce que cela nous coute.

    La couche ne rend jamais l'agent plus genereux : elle ne fait que refuser
    l'infaisable. Si elle ameliore le resultat, c'est que le modele franchissait
    sa propre ligne.
    """

    def __init__(self, inner, c: pm.Contract, side: Side, other: Side, retries: int = 0):
        # retries = 0 par defaut : sur un modele qui bradait tout le markup, le
        # rattrapage deterministe donne EXACTEMENT le meme resultat (u = -513,
        # mandat tenu) que zero, un ou deux rappels au modele - pour 4 appels
        # au lieu de 8 ou 12. Rappeler le modele ne sert que si l'on veut lui
        # laisser choisir QUELS points reprendre ; par defaut, le code choisit.
        self.inner, self.c, self.side, self.other = inner, c, side, other
        self.retries = retries
        self.name = inner.name + "_guard"
        self.refus_accept = 0        # acceptations sous le seuil, refusees
        self.refus_offre = 0         # offres infaisables, corrigees
        self._T = None

    # les compteurs du modele interieur restent lisibles
    @property
    def calls(self):
        return self.inner.calls

    @property
    def invalid(self):
        return self.inner.invalid

    @property
    def unparsed(self):
        return self.inner.unparsed

    @property
    def _history_ref(self):
        return self.inner._history_ref

    @_history_ref.setter
    def _history_ref(self, v):
        self.inner._history_ref = v

    def accepts(self, offer: dict, t: int, T: int) -> bool:
        self._T = T
        veut = self.inner.accepts(offer, t, T)
        if veut and true_u(self.c, self.side, offer) < self.side.reservation - 1e-9:
            self.refus_accept += 1
            # il voulait signer sous le seuil : on refuse et on contre-propose
            self.inner._last_offer = None
            return False
        return veut

    def propose(self, t: int, T: int, history: list) -> dict:
        floor = self.side.target(t, T)
        a = self.inner.propose(t, T, history)
        if true_u(self.c, self.side, a) < self.side.reservation - 1e-9:
            # compte l'offre infaisable, qu'elle soit corrigee par un rappel au
            # modele ou par le rattrapage deterministe : c'est le meme defaut
            self.refus_offre += 1
        for _ in range(self.retries):
            if true_u(self.c, self.side, a) >= self.side.reservation - 1e-9:
                break
            manque = self.side.reservation - true_u(self.c, self.side, a)
            self.inner._rappel = (
                f"REFUS DU CONTROLE : cette offre vous place {manque:.0f} sous votre "
                f"seuil de rupture ({self.side.reservation:+.0f}). Reprenez des points "
                f"et proposez une offre qui reste au-dessus.")
            self.inner.accepts(offer_bidon(self.c), t, T)   # un appel, pour re-demander
            a = self.inner.propose(t, T, history)
        if true_u(self.c, self.side, a) < self.side.reservation - 1e-9:
            a = self._plus_proche_faisable(a, max(floor, self.side.reservation))
        return a

    def _plus_proche_faisable(self, voulu: dict, floor: float) -> dict:
        """Depuis notre modele, rendre les points ou il s'ecartait, par ratio
        decroissant, tant que le seuil tient."""
        a = dict(self.side.ideal)
        ecarts = [pid for pid in voulu if voulu[pid] != a.get(pid)]
        ob = belief(self.c, self.other.whose, a)

        def cle(pid):
            b = dict(a)
            b[pid] = voulu[pid]
            du = true_u(self.c, self.side, b) - true_u(self.c, self.side, a)
            dv = belief(self.c, self.other.whose, b) - ob
            return float("inf") if du >= -1e-9 else dv / abs(du)

        for pid in sorted(ecarts, key=cle, reverse=True):
            b = dict(a)
            b[pid] = voulu[pid]
            if true_u(self.c, self.side, b) >= floor - 1e-9:
                a = b
        return a


def offer_bidon(c: pm.Contract) -> dict:
    return dict(c.template)


# --- une negociation ------------------------------------------------------

def pareto_local(c: pm.Contract, us: Side, them: Side, a: dict) -> int:
    """Nombre de changements d'UN point qui amelioreraient les deux camps a la
    fois. 0 = localement efficace. Une valeur > 0 signifie qu'on a laisse de la
    valeur commune sur la table."""
    n = 0
    uu, ut = true_u(c, us, a), true_u(c, them, a)
    for pid, p in c.params.items():
        for o in p.options:
            if o.id == a[pid]:
                continue
            b = dict(a)
            b[pid] = o.id
            if true_u(c, us, b) > uu + 1e-9 and true_u(c, them, b) > ut + 1e-9:
                n += 1
    return n


def negotiate(c: pm.Contract, us: Side, them: Side, pol_us, pol_them, T: int) -> dict:
    history = []
    offer = dict(them.ideal)                     # ils ouvrent avec leur markup
    history.append({"t": 0, "by": "them", "offer": offer})
    for pol in (pol_us, pol_them):
        pol._history_ref = history               # les politiques LLM lisent l'historique
    agreed, by, t = False, "", 0
    for t in range(T):
        if pol_us.accepts(offer, t, T):
            agreed, by = True, "us"
            break
        ours = pol_us.propose(t, T, history)
        history.append({"t": t, "by": "us", "offer": ours})
        if pol_them.accepts(ours, t, T):
            agreed, by, offer = True, "them", ours
            break
        offer = pol_them.propose(t, T, history)
        history.append({"t": t + 1, "by": "them", "offer": offer})
    final = offer if agreed else None
    loss_markup = belief(c, "ours", c.markup)
    res = {
        "agreed": agreed, "accepted_by": by, "rounds": t + 1,
        "u_us": true_u(c, us, final) if agreed else None,
        "u_them": true_u(c, them, final) if agreed else None,
        # part de ce que le markup nous prenait que nous avons GARDEE
        "kept_us": (1 - true_u(c, us, final) / loss_markup) if agreed and loss_markup else None,
        "share_them": (true_u(c, them, final) / them.aspiration) if agreed and them.aspiration else None,
        "pareto_gap": pareto_local(c, us, them, final) if agreed else None,
        "n_changed": sum(1 for pid in final if final[pid] != c.params[pid].ours) if agreed else None,
        "mandate_breach": (true_u(c, us, final) < us.reservation - 1e-9) if agreed else None,
        "calls": getattr(pol_us, "calls", 0) + getattr(pol_them, "calls", 0),
        "invalid_ids": getattr(pol_us, "invalid", 0),
        "refus_accept": getattr(pol_us, "refus_accept", 0),
        "refus_offre": getattr(pol_us, "refus_offre", 0),
        "unparsed": getattr(pol_us, "unparsed", 0),
        "them": {"label": them.label, "beta": them.beta, "fixation": them.fixation,
                 "reservation": them.reservation, "aspiration": them.aspiration},
        "us": {"reservation": us.reservation},
        "final": final, "history": history,
    }
    return res


# --- la campagne ------------------------------------------------------------

def run_campaign(c: pm.Contract, policies: list[str], seeds: list[int], T: int,
                 call=None, them_mode: str = "algo", budget: float = 0.35,
                 us_name: str = "notre client", them_name: str = "la partie adverse",
                 on_result=None, skip: set | None = None, workers: int = 1) -> list[dict]:
    """skip : (politique, adversaire) deja faits, sautes - une campagne
    interrompue reprend ou elle s'est arretee. workers : negociations menees
    en parallele ; chacune est independante, seul le journal est partage."""
    from concurrent.futures import ThreadPoolExecutor
    import threading
    lock = threading.Lock()
    jobs = [(pol, seed) for seed in seeds for pol in policies
            if not (skip and (pol, seed) in skip)]

    def one(job):
        pol, seed = job
        them = make_them(c, seed)
        r = _one(c, pol, seed, them, T, call, them_mode, budget, us_name, them_name)
        if on_result:
            with lock:
                on_result(r)
        return r

    if workers > 1 and call is not None:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            out = list(ex.map(one, jobs))
    else:
        out = [one(j) for j in jobs]
    return out


def _one(c, pol, seed, them, T, call, them_mode, budget, us_name, them_name) -> dict:
    if True:
        if True:
            us = make_us(c, budget=budget)
            if them_mode == "llm":
                if call is None:
                    raise RuntimeError("them=llm demande un modele")
                p_them = LLMPolicy(c, them, us, call, with_value=True,
                                   who=them_name, other_name=us_name)
            else:
                p_them = AlgoPolicy(c, them, us)
            if pol == "algo":
                p_us = AlgoPolicy(c, us, them)
            elif pol in ("llm_raw", "llm_value", "llm_raw_guard", "llm_value_guard"):
                if call is None:
                    raise RuntimeError(f"{pol} demande un modele")
                base = LLMPolicy(c, us, them, call, with_value=("value" in pol),
                                 who=us_name, other_name=them_name)
                p_us = Guarded(base, c, us, them) if pol.endswith("_guard") else base
            else:
                raise ValueError(pol)
            r = negotiate(c, us, them, p_us, p_them, T)
            r.update({"policy": pol, "seed": seed, "them_mode": them_mode, "T": T})
            return r


def summary(results: list[dict]) -> str:
    by = {}
    for r in results:
        by.setdefault(r["policy"], []).append(r)

    def ms(xs):
        xs = [x for x in xs if x is not None]
        if not xs:
            return "   -   "
        m = statistics.mean(xs)
        se = (statistics.pstdev(xs) / math.sqrt(len(xs))) if len(xs) > 1 else 0.0
        return f"{m:6.2f}±{se:.2f}"

    L = [f"{'politique':<11}{'n':>3}{'accord':>8}{'tours':>7}{'garde':>13}{'attendu':>9}{'eux':>13}"
         f"{'pareto':>9}{'chang.':>8}{'hors mandat':>12}{'appels':>8}"]
    for pol, rs in by.items():
        ag = [r for r in rs if r["agreed"]]
        # VALEUR ATTENDUE : la rupture compte pour zero. Le tableau 'garde' ne
        # porte que sur les accords, donc une politique qui ne conclut qu'une
        # fois sur deux n'y est jugee que sur ses reussites - un biais de
        # survie qui flatte exactement la politique qui echoue le plus.
        att = sum(r["kept_us"] for r in ag if r["kept_us"] is not None) / len(rs)
        L.append(f"{pol:<11}{len(rs):>3}{len(ag)/len(rs):>8.0%}"
                 f"{statistics.mean(r['rounds'] for r in rs):>7.1f}"
                 f"{ms([r['kept_us'] for r in ag]):>13}"
                 f"{att:>9.2f}"
                 f"{ms([r['share_them'] for r in ag]):>13}"
                 f"{ms([r['pareto_gap'] for r in ag]):>9}"
                 f"{ms([r['n_changed'] for r in ag]):>8}"
                 f"{sum(1 for r in ag if r['mandate_breach']):>12}"
                 f"{sum(r['calls'] for r in rs):>8}")
    L += ["",
          "garde   : part de ce que leur markup nous prenait que nous avons gardee (1 = tout),",
          "          ACCORDS SEULEMENT - ne juge une politique que sur ses reussites",
          "attendu : la meme chose, rupture comptee zero. C'est le chiffre a lire :",
          "          un accord manque n'est pas un demi-succes, c'est pas de contrat.",
          "eux     : part de la valeur de leur markup qu'ils obtiennent (selon LEUR utilite)",
          "pareto  : changements d'un point qui amelioreraient les deux camps (0 = efficace)",
          "chang.  : points qui s'ecartent de notre modele dans l'accord (petit = marginal)",
          "hors mandat : accords conclus sous notre seuil de rupture (l'algo ne peut pas)"]
    # apparie : meme adversaire, deux politiques
    pols = list(by)
    if len(pols) >= 2:
        L.append("")
        L.append("COMPARAISONS APPARIEES (meme adversaire), rupture comptee zero :")
        for i in range(len(pols)):
            for j in range(i + 1, len(pols)):
                A = {r["seed"]: r for r in by[pols[i]]}
                B = {r["seed"]: r for r in by[pols[j]]}
                v = lambda r: (r["kept_us"] or 0.0) if r["agreed"] else 0.0
                d = [v(A[s]) - v(B[s]) for s in A if s in B]
                if not d:
                    L.append(f"  {pols[i]} vs {pols[j]} : aucune paire d'accords")
                    continue
                m = statistics.mean(d)
                se = (statistics.pstdev(d) / math.sqrt(len(d))) if len(d) > 1 else 0.0
                wins = sum(1 for x in d if x > 1e-9)
                L.append(f"  {pols[i]} - {pols[j]} : {m:+.3f} ± {se:.3f}  "
                         f"(n={len(d)}, {pols[i]} gagne {wins}/{len(d)})")
    return "\n".join(L)
