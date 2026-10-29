"use client";

type Props = { clients: { slug: string; name: string; fictitious: boolean }[]; current: string };

export function ClientSwitcher({ clients, current }: Props) {
  if (clients.length < 2) {
    const only = clients[0];
    return <span className="font-mono text-xs">{only?.name}</span>;
  }
  return (
    <select
      defaultValue={current}
      onChange={async (e) => {
        await fetch("/api/client", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ slug: e.target.value }),
        }).catch(() => null);
        window.location.reload();
      }}
      className="border border-black bg-white px-2 py-1.5 font-mono text-xs"
      aria-label="Client"
    >
      {clients.map((c) => (
        <option key={c.slug} value={c.slug}>
          {c.name}
          {c.fictitious ? " (fictif)" : ""}
        </option>
      ))}
    </select>
  );
}
