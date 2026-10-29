import type { FindingKind, RightType } from "./types";

export const TYPE_LABELS: Record<RightType, string> = {
  marque: "Marque",
  dessin: "Dessin & modèle",
};

export const KIND_LABELS: Record<FindingKind, string> = {
  annonce: "Annonce",
  image: "Image similaire",
  domaine: "Nom de domaine",
  web: "Page web",
};

export const FIELD_LABELS: Record<string, string> = {
  name: "nom",
  number: "numéro",
  owner: "titulaire",
  classes: "classes",
  filingDate: "date de dépôt",
  registrationDate: "date d'enregistrement",
  expiryDate: "date d'expiration",
  status: "statut",
  notes: "notes",
};
