import type { FindingKind, RightType } from "./types";

export const TYPE_LABELS: Record<RightType, string> = {
  marque: "Marque",
  dessin: "Dessin & modèle",
  brevet: "Brevet",
  certificat_utilite: "Certificat d'utilité",
  domaine: "Nom de domaine",
  droit_auteur: "Droit d'auteur (preuve)",
  cov: "Obtention végétale",
  ig: "Indication géographique",
  autre: "Autre",
};

// Usual term and renewal, used to suggest an expiry date for titles entered by hand.
export const TYPE_HINTS: Record<RightType, string> = {
  marque: "10 ans, renouvelable indéfiniment",
  dessin: "5 ans, renouvelable jusqu'à 25 ans",
  brevet: "20 ans maximum, annuités chaque année",
  certificat_utilite: "10 ans maximum, annuités chaque année",
  domaine: "renouvellement auprès du bureau d'enregistrement (souvent annuel)",
  droit_auteur: "e-Soleau : 5 ans, renouvelable une fois ; le droit dure 70 ans après la mort de l'auteur",
  cov: "25 ans (30 pour certaines espèces)",
  ig: "sans limite de durée",
  autre: "selon le titre",
};

export const OFFICE_LABELS_SHORT = { FR: "INPI", EU: "EUIPO", WO: "OMPI", EP: "OEB", AUTRE: "autre" } as const;

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
  imageUrl: "photo de référence",
};
