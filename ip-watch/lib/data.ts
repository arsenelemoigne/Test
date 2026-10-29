import { cookies } from "next/headers";
import { listClients } from "./store";
import type { Client } from "./types";

export const CLIENT_COOKIE = "ipw_client";

// The client chosen in the header, or the most recently added ready one.
export async function currentClient(): Promise<Client | null> {
  const clients = await listClients();
  const wanted = cookies().get(CLIENT_COOKIE)?.value;
  return clients.find((c) => c.slug === wanted) ?? [...clients].reverse().find((c) => c.status === "ready") ?? clients[clients.length - 1] ?? null;
}
