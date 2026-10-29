export type Role = "avocat" | "client";

export type RightType = "marque" | "dessin" | "brevet" | "certificat_utilite" | "domaine" | "droit_auteur" | "cov" | "ig" | "autre";

// Office or register: INPI (FR), EUIPO (EU), WIPO (WO), European Patent Office (EP), or any other register.
export type Office = "FR" | "EU" | "WO" | "EP" | "AUTRE";

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
  // False for titles held by the client's group (holding, subsidiaries) rather than the client itself.
  clientOwned?: boolean;
  // "inpi" for imported titles, "manuel" for titles entered by the lawyer.
  origin?: "inpi" | "manuel";
};

export type FieldEdit = { value: unknown; prev: unknown; by: Role; at: string };

export type RunSummary = { id: string; startedAt: string; status: RunStatus; total: number; high: number };

export type Mark = Right & { edits: Record<string, FieldEdit>; lastRun: RunSummary | null; favorite: boolean };

export type Company = {
  siren: string;
  name: string;
  legalName?: string;
  address?: string;
  createdOn?: string;
  activity?: string;
  category?: string;
  officers?: string[];
  url: string;
};

export type Client = {
  id: string;
  slug: string;
  name: string;
  status: "pending" | "importing" | "ready" | "error";
  company: Company | null;
  holderNames: string[];
  dataUpdatedAt: string | null;
  importedAt: string | null;
  error: string | null;
};

export type FindingKind = "annonce" | "image" | "domaine" | "web";

export type Finding = {
  id: number;
  runId: string;
  markId: string;
  foundAt: string;
  kind: FindingKind;
  source: string;
  platform: string | null;
  url: string;
  title: string | null;
  price: number | null;
  currency: string | null;
  country: string | null;
  seller: string | null;
  company: { name: string; source?: string; id?: string } | null;
  confidence: number | null;
  reason: string | null;
  screenshotPath: string | null;
  sha256: string | null;
  scanUrl: string | null;
  capturedAt: string | null;
  status: string;
  comment: string | null;
};

export type RunStatus = "running" | "done" | "error";

export type RunStep = { name: string; status: "running" | "done" | "skipped" | "error"; detail?: string };

export type RunStats = { steps: RunStep[]; total: number; high: number; byCountry: Record<string, number>; traceId?: string };

export type WatchRun = {
  id: string;
  clientId: string;
  markId: string;
  markName: string;
  trigger: "bouton" | "assistant" | "email";
  status: RunStatus;
  startedAt: string;
  finishedAt: string | null;
  stats: RunStats | null;
  error: string | null;
};

export type HistoryEvent = {
  id: number;
  createdAt: string;
  kind: "email" | "import" | "recherche" | "modification" | "note" | "question";
  title: string;
  summary: string | null;
  details?: Record<string, unknown> | null;
};

// A saved assistant conversation; each one is answered on its own, never from the others.
export type ConversationMessage = {
  role: "user" | "assistant";
  content: string;
  marks?: string[];
  sources?: { name: string; count: number }[];
  traceId?: string | null;
  runId?: string | null;
  at: string;
};

export type Conversation = { id: string; title: string; messages: ConversationMessage[]; createdAt: string; updatedAt: string };
