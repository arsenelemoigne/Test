"""
Le contrat comme CREANCE CONDITIONNELLE : des termes types, un moteur de
scenarios, une distribution de gains par partie.

Trois couches, et c'est la separation qui compte :

  1. STRUCTURE   - qui doit quoi, a qui, quand, sous quelle condition. Des
                   champs types (Terms). Verifiable contre le texte. Aucun
                   adjectif, aucun vecteur : la valeur du plafond est "12 mois
                   de redevances glissantes", pas "tail_risk = -500".
  2. CONSEQUENCE - ce que la structure PRODUIT dans un etat du monde : flux,
                   expositions, pour chaque partie. Calcule, pas declare. Un
                   plafond de 34 M$ ne "vaut" rien en soi ; face a une
                   reclamation PI de 5 M$ il vaut 5 M$ de plus qu'un plafond
                   de 3 M$, et face a aucune reclamation il vaut zero.
  3. PREFERENCE  - ce que cette distribution vaut POUR MOI : aversion au
                   risque, poids. C'est la seule couche propre a l'utilisateur,
                   et elle n'entre qu'a la fin.

Le meme moteur repond de deux facons, et la premiere ne demande AUCUNE
probabilite :

  scenarios()  - des etats du monde nommes, deterministes. "Une reclamation PI
                 de 5 M$ en annee 3." Chaque partie y a un gain calculable, et
                 deux contrats s'y comparent par DOMINANCE : si B fait mieux
                 que A pour moi dans chaque scenario, je n'ai pas besoin de
                 savoir lequel est probable. Si l'ordre depend du scenario, le
                 tableau dit exactement ou, et le seuil de bascule se lit.
  simulate()   - tirage Monte Carlo sur les memes evenements avec des taux de
                 base (BASE, chacun etiquete comme hypothese). Rend une
                 distribution : moyenne, P5, P95, CVaR. C'est la ou les
                 probabilites entrent, et elles entrent en un seul endroit,
                 lisibles et remplacables.

Le droit entre comme MODULES (LAW) : ce qu'une loi applicable fait d'un
plafond face a une faute lourde, d'une exclusion des dommages indirects, et ce
qu'un contentieux y coute. Changer de loi = charger un autre module et
recalculer. C'est exactement la question que le memo de Veridian dit que ses
juristes ne pouvaient pas trancher sans "a comprehensive legal review".

Ce que ce moteur NE fait PAS, deliberement : les standards vagues ("material
breach", "commercially reasonable") ne compilent pas. Ils entrent ici comme
une probabilite et une charge de la preuve (voir LAW[...]["p_cap_defeated"]),
jamais comme un point. Scott & Triantis, 115 Yale L.J. 814.
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass, replace, asdict, field


# --- 1. STRUCTURE ----------------------------------------------------------

@dataclass(frozen=True)
class Terms:
    label: str = ""
    # duree et prix
    term_years: int = 5
    users: int = 1500                  # 0 = illimite
    fee_y1: float = 2_640_000.0
    escalator: str = "fixed"           # "fixed" | "cpi"
    esc_rate: float = 0.04             # taux fixe
    esc_floor: float = 0.0             # plancher (cpi)
    esc_cap: float = 1.0               # plafond (cpi)
    billing: str = "quarterly_advance" # | "semiannual_arrears"
    payment_days: int = 30
    # responsabilite
    cap_basis: str = "trailing"        # "trailing" (n mois de redevances) | "total_term" (multiple du total)
    cap_months: int = 12
    cap_mult: float = 1.0
    sole_remedy_ip: bool = True        # art. 10.2 : modifier / remplacer / licence / resilier-rembourser
    ip_uncapped: bool = False          # indemnite PI hors plafond ET hors 10.2 : monetaire illimitee
    data_uncapped: bool = False        # obligations de securite des donnees hors plafond
    conseq_waiver: bool = True
    carve_confidentiality: bool = False
    carve_ip: bool = False
    carve_data: bool = False
    carve_downtime_profits: bool = False
    # niveaux de service
    sla: bool = False
    sla_target: float = 0.999
    credit_per_01: float = 0.02        # part de la redevance mensuelle par 0,1 % de manque
    credit_cap: float | None = 0.10    # part de la redevance mensuelle ; None = sans plafond
    chronic: bool = False
    chronic_threshold: float = 0.98
    chronic_months: int = 3
    chronic_window: int = 12
    # garantie
    warranty_days: int = 90
    # sortie
    tfc_licensee: bool = False
    tfc_after_months: int = 0
    tfc_notice_days: int = 90
    tfc_fee_frac: float = 0.0          # part des redevances restantes due a la sortie
    # droit
    law: str = "DE"                    # DE | NY | TX | FR
    forum: str = "court"               # court | arbitration
    # divers
    audit: bool = True
    audit_interval_months: int = 12
    escrow: bool = False
    escrow_breach_trigger: bool = False
    mfc: bool = False
    non_solicit_years: int = 0

    def with_(self, **kw) -> "Terms":
        return replace(self, **kw)


# Les trois positions du dossier Veridian / Halcyon, lues dans le template
# v1.0, le markup HIH v2.0 et le memo interne. Chaque valeur est citee.
TEMPLATE = Terms(
    label="template v1.0",
    users=1500, fee_y1=2_640_000, escalator="fixed", esc_rate=0.04,     # s.1.15, 4.1, 4.2
    billing="quarterly_advance", payment_days=30,                        # s.4.3
    cap_basis="trailing", cap_months=12,                                 # s.11.1
    sole_remedy_ip=True, conseq_waiver=True,                             # s.10.2, 11.2
    sla=False, warranty_days=90,                                         # sched. B.4, s.7.2
    tfc_licensee=False, law="DE", forum="court",                         # s.9.3, 13.1, 13.2
    audit=True, audit_interval_months=12,                                # s.12.1
)

MARKUP = TEMPLATE.with_(
    label="markup HIH v2.0",
    users=0, fee_y1=2_640_000, escalator="cpi", esc_floor=0.0, esc_cap=0.02,   # email s.1, s.2
    billing="semiannual_arrears", payment_days=60,
    cap_basis="total_term", cap_mult=2.0,                                # email s.6
    sole_remedy_ip=False, ip_uncapped=True, data_uncapped=True,          # email s.5, s.6
    carve_confidentiality=True, carve_ip=True, carve_data=True, carve_downtime_profits=True,
    sla=True, sla_target=0.9995, credit_per_01=0.05, credit_cap=None,    # email s.3 ; memo s.7
    chronic=True, chronic_threshold=0.99, chronic_months=1, chronic_window=1,
    warranty_days=548,                                                   # 18 mois, email s.7
    tfc_licensee=True, tfc_after_months=0, tfc_notice_days=90, tfc_fee_frac=0.0,   # email s.8
    law="TX", forum="arbitration",                                       # email s.9
    audit=False,                                                         # email s.10
    escrow=True, escrow_breach_trigger=True, mfc=True, non_solicit_years=2,   # email s.11
)

COUNTER = TEMPLATE.with_(
    label="contre-position Veridian (memo)",
    users=2500, fee_y1=3_168_000, escalator="cpi", esc_floor=0.03, esc_cap=0.05,   # memo s.2, s.3
    billing="quarterly_advance", payment_days=45,
    cap_basis="trailing", cap_months=18,                                 # memo s.5.1
    sole_remedy_ip=True, ip_uncapped=False, data_uncapped=True,          # memo s.5.3, s.6
    carve_confidentiality=True, carve_data=True,                         # memo s.5.2
    sla=True, sla_target=0.999, credit_per_01=0.02, credit_cap=0.10,     # memo s.7
    chronic=True, chronic_threshold=0.98, chronic_months=3, chronic_window=12,
    warranty_days=365,                                                   # memo s.8
    tfc_licensee=True, tfc_after_months=24, tfc_notice_days=180, tfc_fee_frac=0.5,   # memo s.9
    law="DE", forum="court",                                             # memo s.1
    audit=True, audit_interval_months=18,                                # memo s.10
    escrow=True, escrow_breach_trigger=False, mfc=False, non_solicit_years=2,
)


# --- Le droit comme modules --------------------------------------------------
# Chaque entree est une regle par defaut ou imperative, reduite a ce que le
# moteur peut en faire : une probabilite qu'une clause ne tienne pas, et un
# cout de contentieux. Les valeurs sont des HYPOTHESES a remplacer par les
# sources (voir le brief juridique) ; leur place ici est le point important.
LAW = {
    # Sources (brief juridique du 13/09/2026) :
    # DE  - ABRY Partners v. F&W (Del. Ch. 2006), Express Scripts v. Bracket
    #       (Del. 2021) : le plafond tient sauf fraude intentionnelle ; la faute
    #       lourde PEUT etre plafonnee. Le plus favorable au fournisseur.
    # NY  - Kalisch-Jarcho (1983), Sommer (1992), Abacus v. ADT (2012) : le
    #       plafond tombe pour gross negligence "smacking of intentional
    #       wrongdoing" ; barre haute. Metropolitan Life v. Noble Lowndes (1994).
    # TX  - Bombardier v. SPEP (Tex. 2019) : plafond opposable meme au demandeur
    #       en fraude ; gross negligence : cours d'appel partagees ; exigence de
    #       fair notice / conspicuousness (Dresser v. Page, 1993) - un plafond
    #       peu visible peut tomber. Tex. Bus. & Com. Code s.2.719.
    # FR  - art. 1231-3 (dommage previsible ; faute lourde / dolosive) ; Ch.
    #       mixte 22 avr. 2005 et Faurecia II Com. 29 juin 2010 : la faute lourde
    #       ne se deduit pas du seul manquement, meme essentiel - barre haute ;
    #       art. 1170 / Chronopost 1996 : plafond repute non ecrit s'il vide
    #       l'obligation essentielle, mais Faurecia II VALIDE un plafond egal
    #       aux redevances, negocie et tarife ; art. 1231-5 : la redevance de
    #       sortie requalifiee en clause penale peut etre moderee (Civ. 3e 8
    #       janv. 2026 sur le dedit) ; L442-1 C. com. desequilibre significatif
    #       ~0 pour un contrat d'entreprise negocie.
    # Les probabilites restent des ESTIMATIONS : Judilibre (CA depuis 2022)
    # permet de les mesurer pour la France ; c'est le travail suivant.
    "DE": dict(p_cap_defeated=0.03, p_essential=0.0, conseq_waiver_holds=0.97,
               tfc_fee_penalty=0.05, lit_cost=450_000, lit_months=18, review_cost=0),
    "NY": dict(p_cap_defeated=0.08, p_essential=0.0, conseq_waiver_holds=0.95,
               tfc_fee_penalty=0.05, lit_cost=500_000, lit_months=20, review_cost=40_000),
    "TX": dict(p_cap_defeated=0.12, p_essential=0.0, conseq_waiver_holds=0.92,
               tfc_fee_penalty=0.10, lit_cost=450_000, lit_months=18, review_cost=150_000),
    "FR": dict(p_cap_defeated=0.12, p_essential=0.05, conseq_waiver_holds=0.88,
               tfc_fee_penalty=0.30, lit_cost=120_000, lit_months=14, review_cost=120_000),
}
FORUM = {"court": dict(cost_mult=1.0, months_mult=1.0),
         "arbitration": dict(cost_mult=0.7, months_mult=0.6)}


# --- 2. CONSEQUENCE : les taux de base ----------------------------------------
# Chaque nombre est une hypothese. Ils n'entrent QUE dans simulate() ; les
# scenarios nommes n'en utilisent aucun.
BASE = dict(
    cpi_mean=0.03, cpi_sd=0.015,
    rate=0.06,                        # cout du capital annuel (pour le calendrier de facturation)
    # disponibilite mensuelle : normale, mois degrade, mois catastrophique.
    # Hyperscalers 2025 : 99,97-99,98 % (IncidentHub) ; Uptime Institute 2025 :
    # panne significative mediane 53 min, 18 % > 4 h, 6 % > 24 h. Une
    # application SaaS fait moins bien que son hebergeur.
    p_bad_month=0.05, p_awful_month=0.006,
    uptime_normal=(0.9990, 0.9999), uptime_bad=(0.990, 0.998), uptime_awful=(0.95, 0.99),
    downtime_cost_per_hour=8_000,     # pour le licencie (ERP a l'echelle de l'entreprise)
    # reclamation PI contre le fournisseur. Frequence : pas de source par
    # fournisseur (RPX : +30 % de defendeurs NPE en 2025, logiciel ~30 % des
    # actions ; Chien : 20 % des startups financees ont recu une demande) ;
    # cout de defense : AIPLA 2025, mediane 0,6 M$ (<1 M$ en jeu) a 3,6 M$.
    p_ip_claim_year=0.03, ip_claim_median=2_500_000, ip_claim_sigma=1.0, ip_defense_cost=1_200_000,
    ip_remedy_frac=0.35,              # cout de modifier/remplacer sous 10.2, en part de la reclamation
    ip_conseq_frac=0.6,               # dommages indirects en sus si la carve-out joue
    # violation de donnees imputable au fournisseur. Cout : IBM 2025, moyenne
    # mondiale 4,44 M$, Etats-Unis 10,2 M$ ; frequence par entreprise : non
    # publiee (DBIR / assureurs cyber a consulter).
    p_breach_year=0.03, breach_median=4_000_000, breach_sigma=0.9, breach_conseq_frac=0.5,
    # manquement du fournisseur -> litige
    p_dispute_year=0.02, dispute_median=1_500_000, dispute_sigma=0.8,
    # sortie du licencie : son besoin disparait. Churn logo entreprise (ACV >
    # 100 k$) : 1-2 %/an (KeyBanc / Sapphire 2025) ; on prend 3 % pour inclure
    # les reorganisations qui ne se voient pas dans le churn.
    p_need_lapses_year=0.03, switching_cost=1_500_000, residual_value_frac=0.30,
    # insolvabilite du fournisseur
    p_insolvency_year=0.01, escrow_rescue_frac=0.6, escrow_leak_cost=500_000,
    # depassement d'utilisateurs et audit
    p_overage_year=0.12, overage_frac=(0.05, 0.25), per_user_fee=528, expansion_users=(0, 1000),
    # defauts et garantie
    p_defect_month=0.06, defect_cost=60_000, release_age_months=24,
    # clause du client le plus favorise
    p_mfc_trigger_year=0.15, mfc_discount=(0.05, 0.15),
    # debauchage
    p_poach_year=0.05, poach_cost=200_000, non_solicit_effect=0.7,
    # valeur d'usage : ce que la plateforme rapporte au licencie, PAR SIEGE et
    # par mois, independamment du prix paye. Base : 1,4 x le prix du template
    # (2 640 000 $ / 1 500 sieges / 12). Un escalateur plus bas ne peut pas
    # reduire la valeur du service ; une version anterieure le faisait.
    use_value_seat_month=1.40 * 2_640_000 / 1500 / 12,
    seat_demand=1500,                 # sieges dont le licencie a besoin, hors expansion
    use_value_frac=1.40,              # conserve pour les valeurs residuelles
)


@dataclass
class Events:
    """Un etat du monde. Vide = calme. simulate() en tire ; scenarios() en ecrit."""
    cpi: list[float] = field(default_factory=list)            # par annee
    uptime: list[float] = field(default_factory=list)         # par mois
    ip_claims: list[tuple[int, float]] = field(default_factory=list)      # (mois, montant)
    breaches: list[tuple[int, float]] = field(default_factory=list)
    disputes: list[tuple[int, float, bool]] = field(default_factory=list)  # (mois, montant, faute lourde ?)
    need_lapses_month: int | None = None
    insolvency_month: int | None = None
    overage: list[tuple[int, float]] = field(default_factory=list)        # (annee, part)
    expansion_users: int = 0
    defects: list[tuple[int, int]] = field(default_factory=list)          # (mois, age de la version en mois)
    mfc_trigger: tuple[int, float] | None = None                          # (annee, remise)
    poach: list[int] = field(default_factory=list)                        # mois
    cap_defeated: bool = False                                            # tire une fois par monde
    essential_defeated: bool = False
    conseq_waiver_fails: bool = False


def draw_events(t: Terms, rng: random.Random, B=BASE) -> Events:
    e = Events()
    months = t.term_years * 12
    e.cpi = [max(0.0, rng.gauss(B["cpi_mean"], B["cpi_sd"])) for _ in range(t.term_years)]
    for m in range(months):
        u = rng.random()
        lo, hi = (B["uptime_awful"] if u < B["p_awful_month"] else
                  B["uptime_bad"] if u < B["p_awful_month"] + B["p_bad_month"] else
                  B["uptime_normal"])
        e.uptime.append(rng.uniform(lo, hi))
    for y in range(t.term_years):
        if rng.random() < B["p_ip_claim_year"]:
            e.ip_claims.append((y * 12 + rng.randrange(12),
                                B["ip_claim_median"] * math.exp(rng.gauss(0, B["ip_claim_sigma"]))))
        if rng.random() < B["p_breach_year"]:
            e.breaches.append((y * 12 + rng.randrange(12),
                               B["breach_median"] * math.exp(rng.gauss(0, B["breach_sigma"]))))
        if rng.random() < B["p_dispute_year"]:
            e.disputes.append((y * 12 + rng.randrange(12),
                               B["dispute_median"] * math.exp(rng.gauss(0, B["dispute_sigma"])),
                               rng.random() < LAW[t.law]["p_cap_defeated"]))
        if e.need_lapses_month is None and rng.random() < B["p_need_lapses_year"]:
            e.need_lapses_month = y * 12 + rng.randrange(12)
        if e.insolvency_month is None and rng.random() < B["p_insolvency_year"]:
            e.insolvency_month = y * 12 + rng.randrange(12)
        if rng.random() < B["p_overage_year"]:
            e.overage.append((y, rng.uniform(*B["overage_frac"])))
        if rng.random() < B["p_poach_year"]:
            e.poach.append(y * 12 + rng.randrange(12))
    e.expansion_users = rng.randint(*B["expansion_users"])
    for m in range(months):
        if rng.random() < B["p_defect_month"]:
            e.defects.append((m, rng.randrange(B["release_age_months"] + 1)))
    if rng.random() < B["p_mfc_trigger_year"] * t.term_years:
        e.mfc_trigger = (rng.randrange(t.term_years), rng.uniform(*B["mfc_discount"]))
    e.essential_defeated = rng.random() < LAW[t.law]["p_essential"]
    e.conseq_waiver_fails = rng.random() > LAW[t.law]["conseq_waiver_holds"]
    return e


# --- 2. CONSEQUENCE : le moteur ----------------------------------------------

@dataclass
class Payoff:
    licensor: float = 0.0
    licensee: float = 0.0
    detail: dict = field(default_factory=dict)

    def add(self, key: str, licensor: float = 0.0, licensee: float = 0.0):
        self.licensor += licensor
        self.licensee += licensee
        d = self.detail.setdefault(key, [0.0, 0.0])
        d[0] += licensor
        d[1] += licensee


def fee_schedule(t: Terms, cpi: list[float]) -> list[float]:
    out, f = [], t.fee_y1
    for y in range(t.term_years):
        if y > 0:
            r = t.esc_rate if t.escalator == "fixed" else min(t.esc_cap, max(t.esc_floor, cpi[y]))
            f *= 1 + r
        out.append(f)
    return out


def cap_amount(t: Terms, fees: list[float], month: int) -> float:
    """Le plafond en vigueur au mois donne. 'trailing' = redevances des n
    derniers mois ; 'total_term' = multiple du total du terme initial (la
    formulation 'paid and payable' du markup : au maximum des le premier jour)."""
    if t.cap_basis == "total_term":
        return t.cap_mult * sum(fees)
    paid = 0.0
    for m in range(max(0, month - t.cap_months), month):
        paid += fees[m // 12] / 12
    if month < t.cap_months:                       # s.11.1 : annualise si moins de n mois
        paid = fees[0] * t.cap_months / 12
    return paid


def _capped(t: Terms, fees, month, amount, uncapped: bool, e: Events) -> float:
    if uncapped or e.cap_defeated or e.essential_defeated:
        return amount
    return min(amount, cap_amount(t, fees, month))


def evaluate(t: Terms, e: Events, B=BASE) -> Payoff:
    """Le contrat, execute dans un etat du monde. Chaque ligne est une clause."""
    p = Payoff()
    law, forum = LAW[t.law], FORUM[t.forum]
    months = t.term_years * 12
    cpi = e.cpi or [B["cpi_mean"]] * t.term_years
    fees = fee_schedule(t, cpi)
    monthly = [fees[m // 12] / 12 for m in range(months)]
    end = months                                   # mois de fin effectif du contrat

    # sortie anticipee du licencie ---------------------------------------
    if e.need_lapses_month is not None:
        m0 = e.need_lapses_month
        if t.tfc_licensee and m0 >= t.tfc_after_months:
            exit_m = min(months, m0 + math.ceil(t.tfc_notice_days / 30))
            remaining = sum(monthly[exit_m:])
            fee_due = t.tfc_fee_frac * remaining
            if law["tfc_fee_penalty"] and fee_due:
                fee_due *= (1 - law["tfc_fee_penalty"])   # reduction judiciaire, en esperance
            p.add("sortie: redevance de resiliation", licensor=fee_due, licensee=-fee_due)
            p.add("sortie: cout de migration", licensee=-B["switching_cost"])
            end = exit_m
        else:
            # pas de porte de sortie : il paie jusqu'au bout pour une valeur residuelle
            p.add("sortie impossible: usage residuel",
                  licensee=-(1 - B["residual_value_frac"]) * B["use_value_frac"] * sum(monthly[m0:]))

    if e.insolvency_month is not None and e.insolvency_month < end:
        end = min(end, e.insolvency_month)
        rescue = B["escrow_rescue_frac"] if t.escrow else 0.0
        p.add("insolvabilite du fournisseur",
              licensee=-B["switching_cost"] * (1 - rescue))

    # redevances et valeur d'usage ---------------------------------------
    paid = sum(monthly[:end])
    p.add("redevances", licensor=paid, licensee=-paid)
    # sieges servis : la demande, bornee par le plafond nominatif
    demand = B["seat_demand"] + e.expansion_users
    seats = demand if t.users == 0 else min(t.users, demand)
    use = sum(B["use_value_seat_month"] * seats * (1 + cpi[m // 12]) ** (m // 12)
              for m in range(end))
    p.add("valeur d'usage", licensee=use)
    # calendrier de facturation : trimestriel d'avance vs semestriel echu
    shift_months = {"quarterly_advance": +1.5, "semiannual_arrears": -3.0}[t.billing]
    shift_months -= (t.payment_days - 30) / 30
    carry = paid * shift_months / 12 * B["rate"]
    p.add("calendrier de facturation", licensor=carry, licensee=-carry)
    if law["review_cost"]:
        p.add("droit applicable: revue juridique", licensor=-law["review_cost"])

    # utilisateurs -------------------------------------------------------
    if t.users == 0:
        # les sieges au-dela de 1 500 sont servis sans redevance : ce que le
        # fournisseur aurait facture au tarif marginal
        gain = e.expansion_users * B["per_user_fee"] * (end / 12)
        p.add("utilisateurs illimites: sieges non factures", licensor=-gain)
    else:
        for y, frac in e.overage:
            if y * 12 >= end:
                continue
            due = frac * t.users * B["per_user_fee"]
            if t.audit:
                # un audit tous les n mois : la fraction de l'annee ou il est detecte
                caught = min(1.0, 12 / t.audit_interval_months)
                p.add("audit: depassement recouvre", licensor=due * caught, licensee=-due * caught)
            else:
                p.add("pas d'audit: depassement non recouvre", licensee=due)   # il use sans payer

    # niveaux de service -------------------------------------------------
    bad_run = []
    for m in range(end):
        u = e.uptime[m] if m < len(e.uptime) else 1.0
        hours_down = (1 - u) * 730
        p.add("indisponibilite: cout d'exploitation", licensee=-hours_down * B["downtime_cost_per_hour"])
        if t.sla and u < t.sla_target:
            short = (t.sla_target - u) / 0.001
            credit = short * t.credit_per_01 * monthly[m]
            if t.credit_cap is not None:
                credit = min(credit, t.credit_cap * monthly[m])
            p.add("credits SLA", licensor=-credit, licensee=credit)
            if t.carve_downtime_profits and not e.conseq_waiver_fails:
                lost = hours_down * B["downtime_cost_per_hour"]
                rec = _capped(t, fees, m, lost, False, e)
                p.add("perte d'exploitation recouvree (carve-out)", licensor=-rec, licensee=rec)
        if t.chronic:
            bad_run.append(1 if u < t.chronic_threshold else 0)
            window = bad_run[-t.chronic_window:]
            if sum(window) >= t.chronic_months and m + 1 < end:
                # resiliation pour defaillance chronique : le licencie part sans redevance
                rem = sum(monthly[m + 1:end])
                p.add("resiliation chronique: redevances perdues", licensor=-rem, licensee=rem)
                p.add("resiliation chronique: valeur d'usage perdue",
                      licensee=-B["use_value_frac"] * rem * B["residual_value_frac"])
                p.add("resiliation chronique: migration", licensee=-B["switching_cost"] * 0.5)
                end = m + 1
                break

    # reclamations PI -----------------------------------------------------
    for m, amount in e.ip_claims:
        if m >= end:
            continue
        if t.sole_remedy_ip:
            cost = B["ip_remedy_frac"] * amount            # modifier, remplacer, licencier
            p.add("reclamation PI: remede operationnel (10.2)", licensor=-cost)
        else:
            cost = _capped(t, fees, m, amount, t.ip_uncapped, e)
            p.add("reclamation PI: indemnite monetaire", licensor=-cost, licensee=cost - amount)
            if t.carve_ip and not e.conseq_waiver_fails:
                extra = B["ip_conseq_frac"] * amount
                p.add("reclamation PI: dommages indirects (carve-out)", licensor=-extra, licensee=extra)
        p.add("reclamation PI: cout de defense", licensor=-B["ip_defense_cost"] * forum["cost_mult"])

    # violations de donnees ------------------------------------------------
    for m, amount in e.breaches:
        if m >= end:
            continue
        rec = _capped(t, fees, m, amount, t.data_uncapped, e)
        p.add("violation de donnees: dommage direct", licensor=-rec, licensee=rec - amount)
        if t.carve_data and not e.conseq_waiver_fails:
            extra = B["breach_conseq_frac"] * amount
            p.add("violation de donnees: indirects (carve-out)", licensor=-extra, licensee=extra)

    # litiges pour manquement ---------------------------------------------
    for m, amount, gross in e.disputes:
        if m >= end:
            continue
        ev = replace(e, cap_defeated=e.cap_defeated or gross)
        rec = _capped(t, fees, m, amount, False, ev)
        p.add("litige: dommages recouvres", licensor=-rec, licensee=rec - amount)
        cost = law["lit_cost"] * forum["cost_mult"]
        p.add("litige: frais de procedure", licensor=-cost, licensee=-cost)

    # garantie -----------------------------------------------------------
    for m, age in e.defects:
        if m >= end:
            continue
        covered = age * 30 <= t.warranty_days
        if covered:
            p.add("garantie: correction a la charge du fournisseur", licensor=-B["defect_cost"])
        else:
            p.add("garantie expiree: correction payante", licensor=B["defect_cost"] * 0.5,
                  licensee=-B["defect_cost"])

    # clause du client le plus favorise -----------------------------------
    if t.mfc and e.mfc_trigger is not None:
        y, disc = e.mfc_trigger
        if y * 12 < end:
            rebate = disc * sum(monthly[y * 12:end])
            p.add("MFC: alignement tarifaire", licensor=-rebate, licensee=rebate)

    # escrow -------------------------------------------------------------
    if t.escrow and t.escrow_breach_trigger and e.disputes:
        p.add("escrow: liberation sur manquement", licensor=-B["escrow_leak_cost"])

    # debauchage -----------------------------------------------------------
    for m in e.poach:
        if m >= end:
            continue
        eff = B["non_solicit_effect"] if t.non_solicit_years else 0.0
        p.add("debauchage", licensor=-B["poach_cost"] * (1 - eff) * 0.5,
              licensee=-B["poach_cost"] * (1 - eff) * 0.5)

    return p


# --- 2a. Scenarios nommes : aucune probabilite ------------------------------

def scenarios(t: Terms) -> dict[str, Events]:
    """Des etats du monde ecrits a la main. Deterministes. C'est la vue qui ne
    demande AUCUN taux de base : elle dit ce que chaque contrat FAIT si ceci
    arrive, et laisse a l'utilisateur le soin de juger si ceci arrivera."""
    months = t.term_years * 12
    calm = Events(cpi=[0.03] * t.term_years, uptime=[0.9995] * months)
    S = {"calme": calm}
    S["3 mois degrades (99,2 %)"] = replace(calm, uptime=[0.9995] * months)
    for m in (14, 15, 16):
        S["3 mois degrades (99,2 %)"].uptime[m] = 0.992
    S["1 mois catastrophique (96 %)"] = replace(calm, uptime=[0.9995] * months)
    S["1 mois catastrophique (96 %)"].uptime[20] = 0.96
    S["reclamation PI 5 M$ (an 3)"] = replace(calm, ip_claims=[(30, 5_000_000)])
    S["reclamation PI 20 M$ (an 3)"] = replace(calm, ip_claims=[(30, 20_000_000)])
    S["violation de donnees 3 M$ (an 2)"] = replace(calm, breaches=[(18, 3_000_000)])
    # 6 M$ et non 2 : au-dessus du plafond glissant, pour que la faute lourde
    # (qui fait tomber le plafond) se distingue de la faute simple
    S["litige 6 M$, faute simple (an 4)"] = replace(calm, disputes=[(40, 6_000_000, False)])
    S["litige 6 M$, faute lourde (an 4)"] = replace(calm, disputes=[(40, 6_000_000, True)])
    S["le besoin disparait (an 3)"] = replace(calm, need_lapses_month=30)
    S["insolvabilite du fournisseur (an 4)"] = replace(calm, insolvency_month=42)
    S["depassement de sieges 20 % (an 2)"] = replace(calm, overage=[(1, 0.20)], expansion_users=500)
    S["concurrent moins cher (an 2)"] = replace(calm, mfc_trigger=(1, 0.10))
    S["defaut sur version de 8 mois"] = replace(calm, defects=[(20, 8)])
    return S


# --- 2b. Monte Carlo ------------------------------------------------------------

def simulate(t: Terms, n: int = 2000, seed: int = 0, B=BASE) -> list[Payoff]:
    rng = random.Random(seed)
    return [evaluate(t, draw_events(t, rng, B), B) for _ in range(n)]


def stats(xs: list[float]) -> dict:
    xs = sorted(xs)
    n = len(xs)
    k = max(1, n // 20)
    return {"mean": statistics.mean(xs), "p5": xs[k], "p50": xs[n // 2], "p95": xs[-k],
            "cvar5": statistics.mean(xs[:k])}


# --- 3. PREFERENCE ----------------------------------------------------------

def certainty_equivalent(xs: list[float], risk_aversion: float = 0.0, scale: float = 1e6) -> float:
    """CARA : la valeur certaine qu'on echangerait contre la distribution.
    risk_aversion = 0 rend la moyenne. C'est la SEULE entree propre a l'utilisateur."""
    if risk_aversion <= 0:
        return statistics.mean(xs)
    a = risk_aversion / scale
    m = max(xs)
    return m - math.log(statistics.mean(math.exp(-a * (x - m)) for x in xs)) / a


# --- rendu -----------------------------------------------------------------

def _money(x: float) -> str:
    return f"{x/1e6:+.2f} M$" if abs(x) >= 1e5 else f"{x/1e3:+.0f} k$"


def scenario_table(terms: list[Terms]) -> str:
    """Le tableau probabilite-libre : gain de chaque partie dans chaque etat du
    monde, pour chaque contrat, et la dominance qui s'en deduit."""
    names = list(scenarios(terms[0]))
    L = ["GAINS PAR SCENARIO (aucune probabilite ; ecart au scenario calme du template)",
         "=" * 100]
    base = evaluate(terms[0], scenarios(terms[0])["calme"])
    for party, idx in (("FOURNISSEUR (Veridian)", 0), ("LICENCIE (Halcyon)", 1)):
        L += ["", party, f"{'scenario':<38}" + "".join(f"{t.label[:18]:>20}" for t in terms)]
        L.append("-" * 100)
        dom = {t.label: [0, 0] for t in terms}
        for s in names:
            row = []
            for t in terms:
                pay = evaluate(t, scenarios(t)[s])
                v = (pay.licensor, pay.licensee)[idx] - (base.licensor, base.licensee)[idx]
                row.append(v)
            L.append(f"{s:<38}" + "".join(f"{_money(v):>20}" for v in row))
            best = max(row)
            for t, v in zip(terms, row):
                dom[t.label][0] += (v >= best - 1e-6)
                dom[t.label][1] += 1
        L.append("")
        L.append("  meilleur ou ex aequo dans : " + ", ".join(
            f"{k} {a}/{b}" for k, (a, b) in dom.items()))
    L += ["", "Lire : si un contrat est meilleur pour vous dans TOUS les scenarios, vous n'avez",
          "besoin d'aucune probabilite. Sinon, les lignes ou l'ordre s'inverse sont",
          "exactement ce qu'il faut negocier - et le prix de la clause se lit en face."]
    return "\n".join(L)


def distribution_table(terms: list[Terms], n: int = 2000, seed: int = 0,
                       risk_aversion: float = 0.0) -> str:
    L = [f"DISTRIBUTION DES GAINS (Monte Carlo, n={n}, taux de base BASE ; ecart au template)",
         "=" * 100]
    ref = simulate(terms[0], n, seed)
    ref_s = {k: statistics.mean(getattr(p, k) for p in ref) for k in ("licensor", "licensee")}
    for party in ("licensor", "licensee"):
        L += ["", {"licensor": "FOURNISSEUR (Veridian)", "licensee": "LICENCIE (Halcyon)"}[party],
              f"{'contrat':<34}{'moyenne':>12}{'P5':>12}{'P50':>12}{'P95':>12}{'CVaR5':>12}{'eq. certain':>13}"]
        L.append("-" * 100)
        for t in terms:
            xs = [getattr(p, party) - ref_s[party] for p in simulate(t, n, seed)]
            s = stats(xs)
            ce = certainty_equivalent(xs, risk_aversion)
            L.append(f"{t.label[:33]:<34}{_money(s['mean']):>12}{_money(s['p5']):>12}"
                     f"{_money(s['p50']):>12}{_money(s['p95']):>12}{_money(s['cvar5']):>12}"
                     f"{_money(ce):>13}")
    L += ["", f"eq. certain : equivalent certain CARA, aversion au risque = {risk_aversion} "
              f"(0 = la moyenne). C'est la couche 3, la seule qui soit propre a vous."]
    return "\n".join(L)


def decomposition(a: Terms, b: Terms, n: int = 2000, seed: int = 0, top: int = 12) -> str:
    """D'ou vient l'ecart entre deux contrats, clause par clause, en esperance."""
    A, Bp = simulate(a, n, seed), simulate(b, n, seed)
    keys = set()
    for p in A + Bp:
        keys |= set(p.detail)
    rows = []
    for k in keys:
        ma = statistics.mean(p.detail.get(k, [0, 0])[0] for p in A)
        mb = statistics.mean(p.detail.get(k, [0, 0])[0] for p in Bp)
        la = statistics.mean(p.detail.get(k, [0, 0])[1] for p in A)
        lb = statistics.mean(p.detail.get(k, [0, 0])[1] for p in Bp)
        rows.append((k, mb - ma, lb - la))
    rows.sort(key=lambda r: -abs(r[1]) - abs(r[2]))
    L = [f"D'OU VIENT L'ECART  {a.label}  ->  {b.label}   (esperance, par clause)",
         "=" * 100, f"{'clause':<50}{'fournisseur':>16}{'licencie':>16}", "-" * 100]
    for k, dl, dr in rows[:top]:
        L.append(f"{k[:49]:<50}{_money(dl):>16}{_money(dr):>16}")
    tot_l = sum(r[1] for r in rows)
    tot_r = sum(r[2] for r in rows)
    L += ["-" * 100, f"{'TOTAL':<50}{_money(tot_l):>16}{_money(tot_r):>16}"]
    return "\n".join(L)


def breakeven(a: Terms, b: Terms, event: str, sizes: list[float], month: int = 30) -> str:
    """A partir de quelle taille d'evenement le contrat b coute plus que a au
    fournisseur - lu directement, sans probabilite."""
    L = [f"SEUIL DE BASCULE  {a.label} -> {b.label}  sur '{event}'", "-" * 70,
         f"{'taille':>14}{'ecart fournisseur':>22}{'ecart licencie':>20}"]
    calm_a, calm_b = scenarios(a)["calme"], scenarios(b)["calme"]
    for s in sizes:
        ea = replace(calm_a, **{event: [(month, s)]})
        eb = replace(calm_b, **{event: [(month, s)]})
        pa, pb = evaluate(a, ea), evaluate(b, eb)
        L.append(f"{_money(s):>14}{_money(pb.licensor - pa.licensor):>22}"
                 f"{_money(pb.licensee - pa.licensee):>20}")
    return "\n".join(L)
