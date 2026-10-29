// RDAP: registrar, holder (when not hidden) and creation date of a domain name.
function endpoint(domain: string): string {
  if (domain.endsWith(".fr")) return `https://rdap.nic.fr/domain/${domain}`;
  if (domain.endsWith(".com")) return `https://rdap.verisign.com/com/v1/domain/${domain}`;
  if (domain.endsWith(".net")) return `https://rdap.verisign.com/net/v1/domain/${domain}`;
  return `https://rdap.org/domain/${domain}`;
}

type Entity = { roles?: string[]; vcardArray?: [string, [string, unknown, string, unknown][]]; entities?: Entity[] };

function fullName(e: Entity): string | null {
  const fn = e.vcardArray?.[1]?.find((f) => f[0] === "fn");
  const v = fn?.[3];
  return typeof v === "string" && v.trim() ? v.trim() : null;
}

function find(entities: Entity[] | undefined, role: string): Entity | undefined {
  for (const e of entities ?? []) {
    if (e.roles?.includes(role)) return e;
    const nested = find(e.entities, role);
    if (nested) return nested;
  }
  return undefined;
}

export function registrableDomain(host: string): string {
  const parts = host.replace(/^www\./, "").split(".");
  const twoLevel = ["co.uk", "com.cn", "com.au", "co.jp", "com.br", "com.hk"];
  const last2 = parts.slice(-2).join(".");
  return twoLevel.includes(last2) ? parts.slice(-3).join(".") : last2;
}

export async function lookupDomain(domain: string): Promise<{ holder: string | null; registrar: string | null; createdAt: string | null }> {
  const res = await fetch(endpoint(domain), { headers: { Accept: "application/rdap+json, application/json" } });
  if (!res.ok) throw new Error(`RDAP ${res.status}`);
  const r = (await res.json()) as { entities?: Entity[]; events?: { eventAction?: string; eventDate?: string }[] };
  const holder = find(r.entities, "registrant");
  const registrar = find(r.entities, "registrar");
  const name = holder ? fullName(holder) : null;
  return {
    holder: name && !/redacted|privacy|data protected|non publié/i.test(name) ? name : null,
    registrar: registrar ? fullName(registrar) : null,
    createdAt: r.events?.find((e) => e.eventAction === "registration")?.eventDate ?? null,
  };
}
