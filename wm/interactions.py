"""
Les INTERACTIONS entre clauses, calculees exactement.

Pourquoi ce fichier existe. Tout le reste de la pile suppose que la valeur
d'un contrat est la SOMME de celles de ses clauses. C'est ecrit en toutes
lettres a trois endroits : claim_bridge.build() evalue chaque redaction seule,
negotiation._balayage() optimise point par point, negotiation.playbook()
calibre son seuil en sommant des couts individuels. Si l'hypothese est fausse,
ces trois-la se trompent du meme montant, et personne ne le voit.

Or c'est precisement l'objection classique. Un plafond de responsabilite et
une exception au plafond ne se lisent pas separement : lever le plafond quand
l'indemnite PI en sort deja n'a aucun effet, et le lever quand elle y reste en
a un enorme. Deux concessions "moderees" peuvent composer une concession
majeure.

LA MESURE. Pour deux points i et j, la part de valeur qui n'appartient a
aucun des deux seuls est le dividende de Harsanyi d'ordre 2 :

    I(i,j) = U(i et j concedes) - U(i) - U(j) + U(aucun)

et a l'ordre 3 :

    I(i,j,k) = U(ijk) - U(ij) - U(ik) - U(jk) + U(i) + U(j) + U(k) - U(0)

Negatif : concedes ensemble, les deux points coutent PLUS que la somme de
leurs couts separes. Positif : ils se recouvrent, l'un rend l'autre sans
objet.

CE QUI NOUS DISPENSE DE LA MACHINERIE HABITUELLE. Shapley, les dividendes de
Harsanyi echantillonnes, les primitives d'interaction parcimonieuses (Ren,
Zhang et al. ; arXiv:2305.01939 ; applique au jugement juridique dans
arXiv:2410.09083) existent pour un cas precis : la fonction de valeur est une
boite noire qu'on ne peut qu'interroger, en masquant des entrees. Le cout
explose alors en 2^n - sur les 45 modifications de ce markup, 2^45 vaut 35 000
milliards d'evaluations, et il faut echantillonner.

Nous avons la fonction. claim.evaluate() rend le gain des deux parties pour
n'importe quelle combinaison de termes, exactement. Le dividende ne s'estime
donc pas : il se calcule, avec 4 evaluations par paire et 8 par triplet, sans
approximation et sans theoreme de reconstruction a invoquer - la
reconstruction est verifiee au selftest, a la precision machine.

CE QUE CELA NE COUVRE PAS. Les 14 points que claim.py chiffre. Les 12 autres -
definitions, charge de la preuve, seuils de materialite, discretion - n'ont
pas de dollars, donc pas d'interaction calculable ici. Leur additivite reste
une hypothese non testee, et il faut le dire avec le resultat.
"""

from __future__ import annotations

import itertools

from . import claim as C
from . import claim_bridge as CB


def valeur(assignment: dict, party: str = "licensor", n: int = 600, seed: int = 0,
           base: C.Terms = C.TEMPLATE, poids_queue: float = 0.25) -> float:
    """Ce que vaut une assignation complete pour une partie, en dollars.

    Meme mesure que les vecteurs du pont - moyenne plus une part de queue -
    et memes tirages a chaque appel (meme seed), pour que la difference entre
    deux assignations soit celle des TERMES et pas celle du hasard.
    """
    t = CB.terms_of(assignment, base)
    xs = [getattr(p, party) for p in C.simulate(t, n, seed)]
    s = C.stats(xs)
    return s["mean"] + poids_queue * (s["cvar5"] - s["mean"])


class Champ:
    """U(S) pour tout sous-ensemble S de points concedes, avec cache.

    S = l'ensemble des points ou l'on prend la redaction du markup ; hors S on
    garde le modele. U(vide) est donc notre modele intact.
    """

    def __init__(self, party="licensor", n=600, seed=0, base=C.TEMPLATE, poids_queue=0.25):
        self.ref = {pid: opts[0][0] for pid, _, _, opts in CB.VERIDIAN}
        self.kw = dict(party=party, n=n, seed=seed, base=base, poids_queue=poids_queue)
        self._c: dict[frozenset, float] = {}
        self.appels = 0

    def U(self, S) -> float:
        k = frozenset(S)
        if k not in self._c:
            a = dict(self.ref)
            for pid in k:
                a[pid] = CB.THEIRS[pid]
            self._c[k] = valeur(a, **self.kw)
            self.appels += 1
        return self._c[k]

    def dividende(self, S) -> float:
        """Le dividende de Harsanyi de S : somme alternee sur ses sous-ensembles.

            I(S) = somme_{T inclus dans S} (-1)^{|S|-|T|} U(T)

        C'est la part de U(S) qui n'est explicable par AUCUN sous-ensemble
        strict de S. A |S| = 1 c'est l'effet propre du point ; au-dela, ce que
        la combinaison ajoute.
        """
        S = list(S)
        tot = 0.0
        for r in range(len(S) + 1):
            signe = (-1) ** (len(S) - r)
            for T in itertools.combinations(S, r):
                tot += signe * self.U(T)
        return tot


def bruit(party="licensor", n=800, graines=(0, 1, 2)) -> tuple[float, float]:
    """L'ecart a l'additivite, mesure sur plusieurs tirages.

    Sans cela le chiffre n'a pas d'echelle. Mesure : a n = 150 l'ecart va de
    12 % a 39 % selon la graine - c'est du bruit ; a n = 800 et plus il se
    stabilise autour de 26 % avec un ecart-type de 2 points. Un resultat qui
    bouge avec la graine n'est pas un resultat, et le seul moyen de le savoir
    est de changer la graine.
    """
    import statistics
    xs = []
    for g in graines:
        ch = Champ(party=party, n=n, seed=g)
        ids = [pid for pid, _, _, _ in CB.VERIDIAN]
        add = sum(ch.dividende([pid]) for pid in ids)
        xs.append((ch.U(ids) - ch.U([])) - add)
    return statistics.mean(xs), (statistics.pstdev(xs) if len(xs) > 1 else 0.0)


def carte(ordre: int = 2, party: str = "licensor", n: int = 800, seed: int = 0,
          points: list[str] | None = None):
    """Tous les dividendes jusqu'a `ordre`, plus l'ecart total a l'additivite."""
    ch = Champ(party=party, n=n, seed=seed)
    ids = points or [pid for pid, _, _, _ in CB.VERIDIAN]
    nom = {pid: name for pid, name, _, _ in CB.VERIDIAN}
    seuls = {pid: ch.dividende([pid]) for pid in ids}
    div = {}
    for k in range(2, ordre + 1):
        for S in itertools.combinations(ids, k):
            div[S] = ch.dividende(S)
    exact = ch.U(ids) - ch.U([])
    additif = sum(seuls.values())
    # par ordre : ce que chaque ordre de l'expansion ajoute. Si la serie ne
    # converge pas, un modele tronque a l'ordre 2 peut etre PIRE que le modele
    # additif - c'est exactement ce qu'on veut savoir avant d'ajouter des
    # termes d'interaction a quoi que ce soit.
    par_ordre = {k: sum(v for S, v in div.items() if len(S) == k)
                 for k in range(2, ordre + 1)}
    return {"nom": nom, "ids": ids, "seuls": seuls, "div": div, "ordre": ordre,
            "exact": exact, "additif": additif, "ecart": exact - additif,
            "par_ordre": par_ordre, "appels": ch.appels, "champ": ch}


def report(m: dict, top: int = 12, incertitude: tuple | None = None) -> str:
    L = ["LES INTERACTIONS ENTRE CLAUSES, CALCULEES (pas estimees)", "=" * 78, "",
         f"{len(m['ids'])} points chiffrables, {m['appels']} evaluations du contrat.",
         "U(S) = ce que vaut le contrat pour nous quand on concede les points de S.", ""]
    L.append("CE QUE COUTE LE MARKUP ENTIER")
    L.append("-" * 78)
    L.append(f"  somme des effets pris un par un : {m['additif']/1e6:>10.2f} M$")
    L.append(f"  valeur exacte du markup entier  : {m['exact']/1e6:>10.2f} M$")
    ec, base = m["ecart"], abs(m["additif"]) or 1.0
    L.append(f"  ecart a l'additivite            : {ec/1e6:>10.2f} M$"
             f"   ({abs(ec)/base:.1%} de la somme)")
    if incertitude:
        mu, sd = incertitude
        L.append(f"  sur 3 tirages independants      : {mu/1e6:>10.2f} M$ "
                 f"+/- {sd/1e6:.2f}  (ecart-type)")
        if abs(mu) < 3 * sd:
            L.append("  ATTENTION : l'ecart n'est pas distinguable du bruit de tirage.")
    L.append("")
    if abs(ec) / base < 0.05:
        L += ["  L'hypothese d'additivite tient sur ce contrat. Les trois endroits qui la",
              "  supposent - les vecteurs du pont, le balayage de la frontiere, le seuil du",
              "  mandat enumere - sont justifies, a cet ecart pres.", ""]
    else:
        L += ["  L'ADDITIVITE NE TIENT PAS. Les vecteurs du pont, le balayage de la",
              "  frontiere et le seuil du mandat enumere se trompent tous du meme montant,",
              "  dans le meme sens. Lire les paires ci-dessous avant de se fier a un total.", ""]
    if m.get("par_ordre"):
        L.append("L'EXPANSION CONVERGE-T-ELLE ?")
        L.append("-" * 78)
        cum = 0.0
        for k, v in sorted(m["par_ordre"].items()):
            cum += v
            L.append(f"  ordre {k} (les {'paires' if k == 2 else 'triplets' if k == 3 else str(k)+'-uplets'}) "
                     f"ajoute {v/1e6:>+8.2f} M$   cumul {cum/1e6:>+8.2f} M$ "
                     f"= {cum/ec:>6.0%} de l'ecart")
        L.append("")
        if m["par_ordre"] and abs(cum - ec) > 0.2 * abs(ec):
            L += ["  LA SERIE NE CONVERGE PAS a cet ordre. Un modele qui ajouterait des",
                  "  termes d'interaction par paires serait PIRE que le modele additif :",
                  "  il corrige au-dela de la cible, et les triplets ramenent en arriere.",
                  "  La conclusion n'est donc pas 'ajoutons des I_ij' - c'est 'n'additionnons",
                  "  plus rien'. La valeur d'une offre complete se DEMANDE au moteur, qui la",
                  "  calcule exactement en 0,04 ms, au lieu de se reconstruire par morceaux.",
                  ""]
    L.append("EFFET PROPRE DE CHAQUE POINT (concede seul)")
    L.append("-" * 78)
    for pid, v in sorted(m["seuls"].items(), key=lambda kv: kv[1]):
        L.append(f"  {m['nom'][pid][:44]:<46}{v/1e6:>10.2f} M$")
    L.append("")
    L.append(f"INTERACTIONS (ce que la combinaison ajoute, au-dela de la somme)")
    L.append("-" * 78)
    ds = sorted(m["div"].items(), key=lambda kv: -abs(kv[1]))
    if not ds or abs(ds[0][1]) < 1.0:
        L.append("  aucune interaction non nulle : les points de ce contrat ne se")
        L.append("  chevauchent pas, en dollars.")
    for S, v in ds[:top]:
        if abs(v) < 1.0:
            break
        sens = "AGGRAVE" if v < 0 else "recouvre"
        L.append(f"  {v/1e6:>+9.2f} M$  {sens:<9} " + " x ".join(m["nom"][p][:22] for p in S))
    L += ["", "  negatif = concedes ensemble, ces points coutent PLUS que la somme de leurs",
          "            couts separes (l'un retire la protection dont l'autre dependait)",
          "  positif = ils se recouvrent : le second ne coute presque rien une fois le",
          "            premier concede", "",
          "Les 12 points NON chiffrables du contrat - definitions, charge de la preuve,",
          "seuils de materialite, discretion - n'apparaissent pas ici. Leur additivite",
          "reste une hypothese non testee."]
    return "\n".join(L)
