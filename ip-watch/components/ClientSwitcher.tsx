"use client";

type Props = { clients: { slug: string; name: string }[]; current: string | null };

export function ClientSwitcher({ clients, current }: Props) {
  if (clients.length === 0) return null;
  if (clients.length === 1) return <span className="hand text-[14px]">{clients[0].name}</span>;
  return (
    <select
      defaultValue={current ?? undefined}
      onChange={async (e) => {
        await fetch("/api/client", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ slug: e.target.value }),
        }).catch(() => null);
        window.location.reload();
      }}
      className="field hand py-1 text-[13px]"
      aria-label="Client"
    >
      {clients.map((c) => (
        <option key={c.slug} value={c.slug}>
          {c.name}
        </option>
      ))}
    </select>
  );
}
