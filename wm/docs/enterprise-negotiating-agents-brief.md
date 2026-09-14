# Les agents juridiques negociateurs en entreprise : ce qui existe, ce qui est mesure

AVERTISSEMENT SUR LES SOURCES. Cette note a ete assemblee alors que tout acces
direct aux pages (arxiv.org, huggingface.co, aclanthology.org, les sites des
editeurs) etait refuse par le proxy. **Rien n'a ete lu en texte integral** : tout
vient d'extraits de resultats de recherche. Les affirmations corroborees par au
moins deux extraits independants sont marquees `[S+]`, celles qui reposent sur un
seul extrait `[S]`. A verifier avant toute citation dans un document qui sort.

## 1. Les produits

| Produit | Mecanisme documente | Evaluation publiee | Modele de valeur | Modele de l'adversaire | Multi-tours autonome |
|---|---|---|---|---|---|
| **Luminance** Autonomous Negotiation | LLM maison + ML sur ~150M documents ; lit le markup, remedie, envoie, reagit a l'IA d'en face ; "playbook parameters" | aucune ; la demo de nov. 2023 etait un NDA simule entre deux entites imaginaires | non publie | non | **oui - le seul a le revendiquer** `[S+]` |
| **Pactum AI** (achats) | **Contract Space** (termes negociables) + **Value Function** (cout/valeur par terme), ancre et limites par negociation | chiffres clients : Walmart +3 % de gain moyen, +35 jours de delai de paiement, 68 % de taux de conclusion, >1 Md$ automatises | **oui, utilite numerique sur les termes** | appariement/ordonnancement des fournisseurs | oui `[S+]` |
| **Ironclad** Jurist + AI Playbooks | LLM executant une logique par "play" ; positions **preferee / repli / non standard** ; routage des ecarts | aucune | ordinal, textuel | non | non `[S+]` |
| **LexCheck** | appariement au playbook -> accepter / refuser motive / modifier par repli pre-approuve ; 5-6 a 20-30 exemples d'entrainement | revendications editeur seules | echelle de replis, non numerique | non | revendique `[S+]` |
| **Harvey** | LLM + contexte de la firme ; agent conversationnel de construction de playbook | publie des benchmarks (BLB, LAB) mais **pas de son propre produit de redline** | non | non | non `[S+]` |
| **Sirion / Icertis / DocuSign IAM / ContractPodAi / Evisort-Workday** | decomposition agentique (extraction, redaction, detection, redline, obligations) ; scores de risque et matrices d'approbation sur la **valeur du deal** | aucune publique | seuils de routage, **pas** un modele de valeur des termes | non | non `[S+]` |
| **Spellbook / Robin AI / Genie / Legartis / Juro / Della / Definely** | LLM dans Word ou le navigateur ; bibliotheques de clauses | aucune | non | non | non `[S]`-`[S+]` |

Conclusion : **aucun produit juridique n'expose une valeur de reserve, un taux
d'echange entre points, ni un etat de croyance sur l'adversaire.** Le seul a
disposer d'une fonction de valeur numerique declaree est Pactum, en achats, sur
du prix, et sa methode n'est pas publiee.

## 2. Les benchmarks

- **RedlineBench** (Crosby x micro1, juin 2026) - negociation multi-tours sur un
  MSA SaaS, 4 tours alternes, `.docx` de reference multi-avocats, rubriques
  ponderees -10..+10, vote majoritaire pondere par tour, **pas de juge LLM au
  niveau comportemental**. Cinq dimensions dont **prediction d'acceptation par
  l'adversaire** et orientation conclusion. Tetes de classement : Claude Opus 5
  56,9 ; GPT-5.6-Sol 56,2 ; Gemini 3.6 Flash 51. **Aucun modele au-dessus de
  51,5 % sur la dimension commerciale** `[S+]`.
- **TERMS-Bench** (arXiv 2605.13909) - jeu bayesien, **l'environnement est le
  verificateur** (type, politique et paiement de l'adversaire sont latents et
  connus du banc). Metriques : SE+ (efficacite du surplus), CSE+, **FAGR-**
  (faux accords sur episodes infaisables), erreur de croyance, **taux de
  violation critique**. Les accords perdants ne sont pas tronques a zero. Taux
  de violation de 0 a >2 % selon les modeles ; GPT-4o-mini perd contre des
  heuristiques codees a la main `[S+]`.
- **Harvey LAB** - 1 200+ taches, 24 domaines, 75 000+ criteres experts ;
  meilleur score au 10 sept. 2026 : 25,4 % `[S+]`. Qualite du travail rendu, pas
  resultat de negociation.
- **Vals Legal AI Report** (fev. 2025) - sur le **redline**, les avocats battent
  tous les outils (79,7 %) ; c'est l'une des deux seules taches ou l'humain
  gagne `[S+]`.
- **LawGeex 2018** (94 % contre 85 %) - etude editeur, non revue par les pairs,
  sur des NDA avec une liste de 30 points pre-etablie ; Ken Adams : "les
  conditions sont un scenario du pire" pour les avocats `[S+]`.

**Aucun benchmark juridique ne mesure le respect du mandat.** RedlineBench note
le jugement contre des avocats ; TERMS-Bench mesure le surplus et les violations
mais sur du prix, hors droit. La combinaison markup juridique + valeur captee +
nombre de franchissements de la valeur de reserve n'a pas d'equivalent publie.

## 3. Les modes d'echec documentes, avec des chiffres

- **Franchissement de la valeur de reserve.** TERMS-Bench les compte
  explicitement : 0 a >2 % selon les modeles `[S+]`. Nos 3/6 et 4/12 en
  texte seul sont bien plus eleves - parce que le cadre juridique n'a pas de
  scalaire prix pour servir de garde-fou.
- **Le surplus laisse sur la table.** Les modeles "saturent le taux d'accord
  mais laissent l'essentiel du surplus" `[S+]`.
- **Modeliser l'adversaire n'est pas une strategie.** arXiv 2605.16575 : les
  agents estiment les preferences adverses avec precision et tot, et cela
  "n'ameliore pas de facon fiable le resultat du cote informe" `[S+]`. C'est un
  appui direct a notre resultat : ce qui corrige les franchissements est le
  **verificateur**, pas le modele d'adversaire.
- **Ancrage.** Forte correlation entre premiere offre et accord final ; les
  agents "n'exploitent pas la structure d'utilite sous-jacente" `[S]`.
- **Hallucination, ligne de base.** Stanford RegLab (mai 2024, preenregistre,
  202 requetes) : Lexis+ AI >17 %, Westlaw AI-AR >34 % `[S+]`.
- **Fuite de sa propre limite : aucune mesure publiee nulle part.** Champ libre.

## 4. Comment les mandats sont encodes en pratique

1. echelles ordinales de clauses (preferee / repli / non standard), stockees en
   **texte** ; l'execution est un **routage vers un humain**, pas un refus ;
2. matrices d'approbation numeriques, mais sur la **valeur du contrat**, jamais
   sur la valeur du vecteur de termes negocies ;
3. contexte au niveau du prompt (Harvey, "playbook parameters" de Luminance) ;
4. utilite explicite : **Pactum seul**.

**Aucun produit juridique n'applique mecaniquement une valeur de reserve.**

## 5. L'enveloppe reglementaire

- **ABA Formal Op. 512** (29 juil. 2024) : competence, confidentialite,
  **consentement eclaire avant de confier des confidences client a une IA
  generative - une clause type dans la lettre de mission ne suffit pas** `[S+]`.
- **Barreau de Californie, mai 2026** : guidance etendue aux agents ;
  l'autonomie "ne dispense pas l'avocat d'exercer son propre jugement" `[S]`.
  **Oregon Formal Op. 2026-208** (fev. 2026) sur les agents autonomes `[S]`.
- **France/UE** : plan d'action du CNB (mars 2024), **guide pratique sur l'IA
  generative, 1re edition, sept. 2024** (le secret professionnel au centre),
  guide CCBE oct. 2025 `[S+]`. AI Act : la negociation contractuelle commerciale
  tombe en principe hors de l'annexe III ; le calendrier cite dans les extraits
  est incoherent (aout 2026 contre aout 2027) - **a verifier** `[S]`.

Consequence : un agent qui **engage** le client est hors du cadre ; un agent qui
calcule une position et rend une recommandation **bornee et journalisee** a un
humain est dans le cadre - et notre verificateur deterministe est precisement
l'artefact qui prouve la "supervision humaine effective".

## 6. Ou nous sommes neufs, ou nous sommes en retard

Neufs : (a) une valorisation en **flux de tresorerie** des termes numeriques -
rien d'equivalent en IA juridique ; (b) un **verificateur deterministe qui ramene
mecaniquement a zero les franchissements de mandat sans perdre d'accords** -
exactement le defaut que TERMS-Bench quantifie et que tous les CLM traitent par
un routage humain ; (c) un profil ordinal pour les termes **structurels**, la ou
RedlineBench plafonne a 51,5 % en commercial ; (d) la preuve que le gain vient du
verificateur et non du modele d'adversaire.

En retard : la taille d'echantillon (6 et 12 episodes sur 2 contrats, contre
1 200 taches et 75 000 criteres pour LAB) ; aucune mesure de la qualite du
redline **dans le document** (nous negocions sur des termes, pas sur un `.docx`) ;
aucune validation tierce ; aucune distribution Word/CLM.

## 7. Les cinq choses a faire, dans l'ordre

1. **Prendre RedlineBench comme cadre** et publier les deux nombres que personne
   ne publie - taux de franchissement et valeur captee - avec une ablation
   propre : le verificateur greffe sur Opus 5 / GPT-5.6.
2. **Reprendre le vocabulaire de TERMS-Bench tel quel** (SE+, CSE+, FAGR-,
   erreur de croyance, taux de violation critique) et publier la correspondance
   avec nos "+0,19 d'esperance" et "franchissements -> 0".
3. **Changer d'echelle** : >=50 contrats x >=100 episodes randomises, avec
   intervalles de confiance. "3/6" et "4/12" seront ecartes comme anecdotes.
4. **Faire du verificateur le produit** : un compilateur de mandat qui ingere du
   preferee/repli/walk-away textuel, en sort des contraintes numeriques, et
   borne toute position sortante avec un journal.
5. **Mesurer la fuite de la limite** - l'erreur de croyance de l'adversaire sur
   notre valeur de reserve, tour par tour. Personne n'a publie ce nombre.
