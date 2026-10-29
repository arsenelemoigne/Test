// Working name, to be replaced once the group picks the project name.
export const APP_NAME = "Registre PI";

export const FIRM_NAME = "Cabinet fictif — Projet LAB EFB";

export const PRODUCTION_HOSTING_NOTE =
  "En production : hébergement des données et inférence IA dans l'Union européenne (Scaleway). La démo actuelle n'est pas encore hébergée ainsi.";

export const REMINDER_OFFSETS_DAYS = [180, 90, 30];

export function lawyerEmail(): string | null {
  return process.env.LAWYER_EMAIL || null;
}

export function assistantEmail(): string | null {
  return process.env.ASSISTANT_EMAIL || null;
}
