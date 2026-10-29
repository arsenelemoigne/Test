import { cookies } from "next/headers";
import { CLIENTS } from "@/data/clients";
import type { ClientData } from "./types";

export const CLIENT_COOKIE = "ipw_client";

export function listClients(): { slug: string; name: string; fictitious: boolean }[] {
  return CLIENTS.map((c) => ({ slug: c.slug, name: c.name, fictitious: c.fictitious }));
}

// Real clients come first so the demo opens on them once imported.
export function currentClient(): ClientData {
  const wanted = cookies().get(CLIENT_COOKIE)?.value;
  const sorted = [...CLIENTS].sort((a, b) => Number(a.fictitious) - Number(b.fictitious));
  return sorted.find((c) => c.slug === wanted) ?? sorted[0];
}
