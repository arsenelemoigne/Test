export function formatDate(value: string | Date | null | undefined): string {
  if (!value) return "—";
  const d = typeof value === "string" ? new Date(value) : value;
  if (Number.isNaN(d.getTime())) return "—";
  return d.toLocaleDateString("fr-FR", { day: "2-digit", month: "2-digit", year: "numeric", timeZone: "UTC" });
}

export function formatDateTime(value: string | Date | null | undefined): string {
  if (!value) return "—";
  const d = typeof value === "string" ? new Date(value) : value;
  return d.toLocaleString("fr-FR", { dateStyle: "short", timeStyle: "short", timeZone: "Europe/Paris" });
}

export function formatPrice(price: number | null, currency: string | null): string {
  if (price === null) return "—";
  return new Intl.NumberFormat("fr-FR", { style: "currency", currency: currency ?? "EUR", maximumFractionDigits: 0 }).format(price);
}

export function toIsoDate(value: string): string | null {
  const m = value.trim().match(/^(\d{2})\/(\d{2})\/(\d{4})$/);
  if (m) return `${m[3]}-${m[2]}-${m[1]}`;
  if (/^\d{4}-\d{2}-\d{2}$/.test(value.trim())) return value.trim();
  return null;
}
