# Discussion — Analyse des actes RCS de DELOS INTELLIGENCE (SIREN 977575612)

Date : 22/10/2026

## Demande initiale

Objectif : télécharger et analyser les actes déposés au RCS par DELOS INTELLIGENCE (SIREN 977575612), en particulier les dépôts du 17/09/2026, pour vérifier le montant réel de sa levée de fonds (série A annoncée à 10 M€).

Contexte vérifié :
- BODACC B n°191 (5-6 oct. 2026), annonce n°7167 : capital porté à 9 183,41 €.
- Annonce légale du 31/08/2026 : PV du président du 31/07/2026 décidant une augmentation de capital de 2 451,41 € de nominal (6 732 € → 9 183,41 €) ; PV du 07/09/2026 constatant la réalisation définitive au 31/08/2026.
- data.inpi.fr liste 10 actes, dont au moins 5 dépôts du 17/09/2026.

Étapes demandées :
1. Récupérer les PDF via l'API RNE de l'INPI (identifiants dans `INPI_USER` / `INPI_PASS`), ou via Playwright sur data.inpi.fr en secours (s'arrêter en cas de captcha). Enregistrer dans `./actes/` avec la date de dépôt et le type d'acte dans le nom de fichier.
2. Extraire le texte : `pdftotext`, puis OCR `ocrmypdf -l fra` si le PDF est scanné.
3. Analyser chaque acte (actes du 17/09/2026 en priorité), avec citation exacte et numéro de page : type d'acte, actions nouvelles, nominal, catégorie, prix de souscription, prime d'émission, montant total souscrit, souscripteurs, conversion de BSA AIR / OC / BSPCE, certificat du dépositaire (art. L225-146 C. com.), délégations ou tranches futures.
4. Rapport `./rapport.md` : tableau des actes, equity souscrit et libéré (calcul détaillé), écart avec les 10 M€ et explications possibles, valorisations pre-money / post-money implicites, liste de ce qui est absent ou illisible. N'inventer aucun chiffre (« non mentionné »).

## Résultat : bloqué avant l'étape 1

Aucun PDF n'a été téléchargé, aucun rapport n'a été produit, rien n'a été inventé.

1. **Réseau** : la politique réseau de l'environnement cloud refuse l'accès (403 du proxy) à `registre-national-entreprises.inpi.fr`, `data.inpi.fr` et `www.bodacc.fr`. L'API et Playwright sont donc impossibles, et la documentation officielle de l'API RNE n'a pas pu être vérifiée.
   → Correctif : menu de l'environnement (barre de titre de la session) › Edit › Network access › ajouter ces domaines aux domaines autorisés. Doc : https://code.claude.com/docs/en/cloud-environments#network-access
2. **Identifiants** : `INPI_USER` et `INPI_PASS` ne sont pas définies dans la session.
   → Correctif : les ajouter comme variables d'environnement / secrets dans les réglages de l'environnement (jamais dans le chat). Une nouvelle session les prendra en compte.
3. **OCR** : `pdftotext` est disponible ; `ocrmypdf` s'installe via pip, mais Tesseract et le pack français ne sont pas installés.

## Question : partager l'écran de la machine virtuelle pour se connecter ?

Impossible :
- la machine virtuelle n'a pas d'écran interactif partageable (seulement des captures d'écran) ;
- même avec un navigateur, le réseau bloque data.inpi.fr.

## Options pour continuer

1. **Recommandé** : télécharger soi-même les PDF depuis data.inpi.fr (fiche 977575612 › Documents associés › Actes, au minimum les dépôts du 17/09/2026), puis les envoyer en pièces jointes dans le chat ou les ajouter dans `actes/` sur la branche `ccr-64759a0a-7l5s7y`. Claude fait ensuite l'extraction, l'analyse et `rapport.md`. Les identifiants restent sur ton ordinateur.
2. Lancer Claude Code sur ton propre ordinateur (application de bureau ou terminal), qui utilise ton réseau.
3. Débloquer l'environnement cloud (domaines INPI autorisés + `INPI_USER` / `INPI_PASS`), puis ouvrir une nouvelle session.
