import { createHash, createHmac, timingSafeEqual } from "crypto";
import { cookies } from "next/headers";
import type { Role } from "./types";

export const SESSION_COOKIE = "ipw_session";

function secret(): string | null {
  const password = process.env.DEMO_PASSWORD;
  if (!password) return null;
  return createHash("sha256").update(`ip-watch:${password}`).digest("hex");
}

export function isConfigured(): boolean {
  return Boolean(process.env.DEMO_PASSWORD);
}

export function checkPassword(candidate: string): boolean {
  const expected = process.env.DEMO_PASSWORD;
  if (!expected) return false;
  const a = Buffer.from(candidate);
  const b = Buffer.from(expected);
  return a.length === b.length && timingSafeEqual(a, b);
}

export function signSession(role: Role): string {
  const key = secret();
  if (!key) throw new Error("DEMO_PASSWORD is not set");
  const mac = createHmac("sha256", key).update(role).digest("hex");
  return `${role}.${mac}`;
}

export function verifySession(value: string | undefined): Role | null {
  const key = secret();
  if (!key || !value) return null;
  const [role, mac] = value.split(".");
  if (role !== "avocat" && role !== "client") return null;
  const expected = createHmac("sha256", key).update(role).digest("hex");
  const a = Buffer.from(mac ?? "");
  const b = Buffer.from(expected);
  return a.length === b.length && timingSafeEqual(a, b) ? role : null;
}

export function currentRole(): Role | null {
  return verifySession(cookies().get(SESSION_COOKIE)?.value);
}
