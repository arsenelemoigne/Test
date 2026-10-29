# Registre PI — démo Projet LAB (EFB)

Suivi d'un portefeuille de marques et dessins : registre des titres, échéances de renouvellement, carte des pays couverts, journal de veille avec dossier de preuve, assistant, export Excel.

Démo pédagogique : le cabinet est fictif. Les clients affichés sont soit fictifs (signalés comme tels), soit importés depuis des registres publics.

## Lancer en local

```bash
npm install
DEMO_PASSWORD=choisir-un-mot-de-passe npm run dev
```

## Variables d'environnement

Voir `.env.example`. `/api/health` indique lesquelles sont définies (jamais leur valeur).

## Fonctionnement

- **Email → registre** : un email reçu par l'assistant (webhook Resend sur `/api/inbound`) est classé (nouveau client, mise à jour, ajout de marque, recherche, question, note), traité, résumé dans l'historique, puis l'assistant répond par email. Seuls `LAWYER_EMAIL` et `ALLOWED_SENDERS` sont servis.
- **Nouveau client** : société identifiée dans l'annuaire officiel, marques et dessins importés de l'INPI (réponses brutes conservées dans `raw_imports`).
- **Assistant** : outils Légifrance, Judilibre, INPI, annuaire, BODACC, droit de l'UE, recherche web en repli ; les liens non renvoyés par un outil sont signalés.
- **Recherche de contrefaçons** : à la demande (bouton ou assistant) ; résultats en direct sur le tableau de bord, ajoutés au journal de la marque (Excel ou CSV).

## Déploiement (Railway)

Service Railway depuis ce dépôt, dossier racine `ip-watch`. Railway lance `npm run build` puis `npm start` ; Next.js écoute sur le port fourni par `PORT`.
