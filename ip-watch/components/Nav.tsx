"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const LINKS = [
  { href: "/", label: "Vue d'ensemble" },
  { href: "/portefeuille", label: "Portefeuille" },
  { href: "/echeances", label: "Échéances" },
  { href: "/carte", label: "Carte" },
  { href: "/veille", label: "Veille" },
  { href: "/assistant", label: "Assistant" },
];

export function Nav() {
  const pathname = usePathname();
  return (
    <nav className="flex gap-1 overflow-x-auto border-b border-neutral-200 py-2">
      {LINKS.map((l) => {
        const active = l.href === "/" ? pathname === "/" : pathname.startsWith(l.href);
        return (
          <Link
            key={l.href}
            href={l.href}
            className={`whitespace-nowrap px-3 py-1.5 font-mono text-xs uppercase tracking-wider ${active ? "bg-black text-white" : "hover:bg-neutral-100"}`}
          >
            {l.label}
          </Link>
        );
      })}
    </nav>
  );
}
