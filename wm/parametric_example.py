"""
La licence Veridian / Halcyon comme objet parametrique.

Sept variables, chacune avec son domaine reel : ce que nous avions ecrit, ce
qu'ils ont propose, et les redactions intermediaires qu'un negociateur
envisagerait. LES CHIFFRES SONT POSES A LA MAIN et illustratifs ; les clauses,
les options et les couplages viennent du contrat et du markup.

Unites : milliers d'euros par an pour revenue et tail_risk ; les quatre autres
axes sont des echelles relatives ou zero = notre redaction.
"""

from .parametric import Option as O, Parameter as P, Coupling as C, Contract

PARAMS = [
    P("users", "Modele d'utilisateurs", "1.15", ours="named", theirs="authorized",
      options=[
        O("named", "Named User, nominatif, non transferable",
          ours={"revenue": 0, "control": 0, "admin": 0},
          theirs={"revenue": 0, "flexibility": 0}),
        O("concurrent", "Utilisateurs simultanes, plafond a 1,400",
          ours={"revenue": -280, "control": -1, "admin": -1},
          theirs={"revenue": 275, "flexibility": 1}),
        O("authorized", "Authorized User, poste partageable",
          ours={"revenue": -612, "control": -2, "admin": -1},
          theirs={"revenue": 600, "flexibility": 1}),
      ]),
    P("escalator", "Escalateur tarifaire", "4.2", ours="cpi_floor3", theirs="lesser_cpi2",
      options=[
        O("cpi_floor3", "CPI-U, plancher 3%, plafond 5%",
          ours={"revenue": 0}, theirs={"revenue": 0, "flexibility": 0}),
        O("cpi_floor2", "CPI-U, plancher 2%, plafond 5%",
          ours={"revenue": -74}, theirs={"revenue": 72}),
        O("lesser_cpi2", "Le moindre de CPI-U ou 2%",
          ours={"revenue": -148}, theirs={"revenue": 145}),
      ]),
    P("cap", "Plafond de responsabilite", "11.1", ours="m12", theirs="x2_term",
      options=[
        O("m12", "12 mois de redevances glissantes",
          ours={"tail_risk": 0}, theirs={"tail_risk": 0}),
        O("m18", "18 mois de redevances glissantes",
          ours={"tail_risk": -150}, theirs={"tail_risk": 120, "enforceability": 2}),
        O("x2_term", "2x les redevances du terme initial",
          ours={"tail_risk": -410, "control": -1},
          theirs={"tail_risk": 150, "enforceability": 3}),
      ]),
    P("warranty", "Periode de garantie", "7.2", ours="d90", theirs="m18",
      options=[
        O("d90", "90 jours", ours={"admin": 0}, theirs={"enforceability": 0}),
        O("m12", "12 mois", ours={"admin": -12, "tail_risk": -40},
          theirs={"tail_risk": 130, "enforceability": 3}),
        O("m18", "18 mois", ours={"admin": -24, "tail_risk": -96},
          theirs={"tail_risk": 180, "enforceability": 5}),
      ]),
    P("sole_remedy", "Clause de recours exclusif (indemnite PI)", "12.3",
      ours="keep", theirs="delete",
      options=[
        O("keep", "Recours exclusif maintenu, election a notre main",
          ours={"tail_risk": 0, "control": 0}, theirs={"enforceability": 0}),
        O("narrow", "Maintenu, mais sans les reclamations de contrefacon deliberee",
          ours={"tail_risk": -70, "control": -1},
          theirs={"tail_risk": 95, "enforceability": 2}),
        O("delete", "Supprime",
          ours={"tail_risk": -205, "control": -2},
          theirs={"tail_risk": 120, "enforceability": 3}),
      ]),
    P("burden", "Charge de la preuve sur les credits SLA", "9.4",
      ours="them", theirs="us",
      options=[
        O("them", "Au client de demontrer l'indisponibilite",
          ours={"admin": 0, "enforceability": 0}, theirs={"admin": 0}),
        O("shared", "Releves conjoints, presomption neutre",
          ours={"admin": -60, "enforceability": -1}, theirs={"admin": 20}),
        O("us", "A nous de demontrer que l'incident n'a pas eu lieu",
          ours={"admin": -140, "enforceability": -3}, theirs={"admin": 40}),
      ]),
    P("nonsolicit", "Non-sollicitation", "14.13", ours="y1", theirs="y2",
      options=[
        O("y1", "1 an, mutuelle", ours={"control": 0}, theirs={"control": 0}),
        O("y2", "2 ans, mutuelle",
          ours={"control": 1, "flexibility": -1}, theirs={"control": 3}),
      ]),
]

COUPLINGS = [
    C("cap", "x2_term", "sole_remedy", "delete",
      {"tail_risk": -95, "control": -1},
      "Le plafond et le recours exclusif etaient une seule protection. "
      "Perdre les deux compose au lieu de s'additionner."),
    C("users", "authorized", "escalator", "lesser_cpi2",
      {"revenue": -40},
      "Plus de sieges sous un escalateur plus faible : la perte se compose."),
    C("burden", "us", "warranty", "m18",
      {"admin": -35},
      "Une garantie longue dont nous portons la preuve coute plus que les deux "
      "separement."),
]


def contract() -> Contract:
    return Contract(PARAMS, COUPLINGS)
