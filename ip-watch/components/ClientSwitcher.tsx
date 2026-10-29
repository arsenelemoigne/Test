"use client";

import { usePathname } from "next/navigation";

type Props = { clients: { slug: string; name: string; fictitious: boolean }[]; current: string };

export function ClientSwitcher({ clients, current }: Props) {
  const pathname = usePathname();
  if (clients.length < 2) {
    const only = clients[0];
    return <span className="font-mono text-xs">{only?.name}</span>;
  }
  return (
    <form method="POST" action="/api/client">
      <input type="hidden" name="back" value={pathname} />
      <select
        name="slug"
        defaultValue={current}
        onChange={(e) => e.currentTarget.form?.submit()}
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
    </form>
  );
}
