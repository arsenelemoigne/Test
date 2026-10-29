"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Scribble } from "./RoughBox";

const LINKS = [
  { href: "/", label: "Tableau de bord" },
  { href: "/marques", label: "Marques" },
];

export function Nav() {
  const pathname = usePathname();
  return (
    <nav className="flex gap-6">
      {LINKS.map((l) => {
        const active = l.href === "/" ? pathname === "/" : pathname.startsWith(l.href);
        return (
          <Link key={l.href} href={l.href} className={`hand flex flex-col items-center text-[15px] ${active ? "" : "text-[color:var(--pencil)] hover:text-[color:var(--ink)]"}`}>
            {l.label}
            <Scribble width={active ? 70 : 0} className={active ? "" : "invisible"} />
          </Link>
        );
      })}
    </nav>
  );
}
