import type { FindingKind, FindingStatus, RightType } from "./types";

export const TYPE_LABELS: Record<RightType, string> = {
  marque: "Marque",
  dessin: "Dessin & modèle",
};

export const KIND_LABELS: Record<FindingKind, string> = {
  marketplace: "Place de marché",
  site: "Site marchand",
  domaine: "Nom de domaine",
  depot: "Dépôt concurrent",
};

export const STATUS_LABELS: Record<FindingStatus, string> = {
  nouveau: "Nouveau",
  a_verifier: "À vérifier",
  constat_demande: "Constat demandé",
  action_engagee: "Action engagée",
  ecarte: "Écarté",
};

export const OPEN_STATUSES: FindingStatus[] = ["nouveau", "a_verifier"];

export const FIELD_LABELS: Record<string, string> = {
  name: "Nom",
  number: "Numéro",
  owner: "Titulaire",
  classes: "Classes",
  filingDate: "Date de dépôt",
  registrationDate: "Date d'enregistrement",
  expiryDate: "Date d'expiration",
  status: "Statut",
  notes: "Notes",
  relevance: "Pertinence",
  comment: "Commentaire",
  constatRecommended: "Constat recommandé",
};
