# Registre PI — démo Projet LAB (EFB)

Suivi d'un portefeuille de marques et dessins : registre des titres, échéances de renouvellement, carte des pays couverts, journal de veille avec dossier de preuve, assistant, export Excel.

Démo pédagogique : le cabinet est fictif. Les clients affichés sont soit fictifs (signalés comme tels), soit importés depuis des registres publics.

## Lancer en local

```bash
npm install
DEMO_PASSWORD=choisir-un-mot-de-passe npm run dev
```

## Variables d'environnement

| Variable | Obligatoire | Rôle |
|---|---|---|
| `DEMO_PASSWORD` | oui | Mot de passe partagé de la démo (choix Avocat ou Client à la connexion) |
| `OPENROUTER_API_KEY` | non | Active l'assistant |
| `OPENROUTER_MODEL` | non | Modèle OpenRouter, `google/gemini-2.5-flash` par défaut |
| `LAWYER_EMAIL` | non | Destinataire des rappels d'échéance |
| `ASSISTANT_EMAIL` | non | Adresse à laquelle les clients écrivent l'assistant |

## Données

Chaque client est un fichier JSON dans `data/clients/`, déclaré dans `data/clients/index.ts`. Les modifications faites dans l'interface sont, à ce stade, enregistrées dans le navigateur de chaque utilisateur.

## Déploiement (Railway)

Service Railway depuis ce dépôt, dossier racine `ip-watch`. Railway lance `npm run build` puis `npm start` ; Next.js écoute sur le port fourni par `PORT`.
