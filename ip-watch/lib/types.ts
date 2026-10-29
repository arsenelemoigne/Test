export type RightType = "marque" | "dessin";

// Office that granted the right: INPI (FR), EUIPO (EU) or WIPO international registration (WO).
export type Office = "FR" | "EU" | "WO";

export type Territory = {
  // ISO 3166-1 alpha-2 code, or "EU" for the whole European Union.
  code: string;
  status?: string;
};

export type Right = {
  id: string;
  type: RightType;
  office: Office;
  number: string;
  name: string;
  owner: string;
  classes: number[];
  filingDate: string | null;
  registrationDate: string | null;
  expiryDate: string | null;
  status: string;
  territories: Territory[];
  imageUrl?: string | null;
  sourceUrl?: string | null;
  notes?: string;
};

export type FindingKind = "marketplace" | "site" | "domaine" | "depot";

export type FindingStatus = "nouveau" | "a_verifier" | "constat_demande" | "action_engagee" | "ecarte";

export type Evidence = {
  capturedAt: string;
  screenshotUrl: string | null;
  sha256: string | null;
  scanUrl?: string | null;
};

export type Finding = {
  id: string;
  foundAt: string;
  kind: FindingKind;
  platform: string;
  url: string;
  title: string;
  price: number | null;
  currency: string | null;
  country: string | null;
  seller: string | null;
  company: { name: string; id?: string; address?: string; source?: string } | null;
  matchedRightIds: string[];
  relevance: 1 | 2 | 3 | 4 | 5;
  status: FindingStatus;
  constatRecommended: boolean;
  comment: string;
  evidence: Evidence | null;
};

export type ClientData = {
  slug: string;
  name: string;
  // True for the fictitious test company, false for data imported from public registries.
  fictitious: boolean;
  disclaimer: string;
  sources: { name: string; updatedAt: string | null; note?: string }[];
  importedAt: string;
  rights: Right[];
  findings: Finding[];
  lastWatchAt: string | null;
};

export type Role = "avocat" | "client";
