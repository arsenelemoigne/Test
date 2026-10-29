import Link from "next/link";
import { redirect } from "next/navigation";
import { currentRole } from "@/lib/auth";
import { currentClient } from "@/lib/data";
import { APP_NAME, assistantEmail, FIRM_NAME, PRODUCTION_HOSTING_NOTE } from "@/lib/config";
import { listClients, storeReady } from "@/lib/store";
import { AppProvider } from "@/components/AppProvider";
import { Nav } from "@/components/Nav";
import { ClientSwitcher } from "@/components/ClientSwitcher";
import { LogoutButton } from "@/components/LogoutButton";
import { Scribble } from "@/components/RoughBox";
import { formatDate } from "@/lib/format";

export const dynamic = "force-dynamic";

export default async function AppLayout({ children }: { children: React.ReactNode }) {
  const role = currentRole();
  if (!role) redirect("/login");
  if (!storeReady()) {
    return <p className="hand p-8">Base de données non configurée : ajoutez SUPABASE_URL et SUPABASE_SERVICE_ROLE_KEY dans Railway.</p>;
  }
  const [clients, client] = await Promise.all([listClients(), currentClient()]);

  return (
    <AppProvider value={{ role, client, assistantEmail: assistantEmail() }}>
      <div className="mx-auto flex min-h-screen max-w-[1400px] flex-col px-4 sm:px-6">
        <header className="flex flex-wrap items-center justify-between gap-3 pb-2 pt-5">
          <Link href="/" className="flex flex-col">
            <span className="hand text-[22px] font-semibold leading-none tracking-tight">{APP_NAME}</span>
            <Scribble width={104} className="mt-0.5" />
          </Link>
          <Nav />
          <div className="flex flex-wrap items-center gap-2">
            <ClientSwitcher clients={clients.map((c) => ({ slug: c.slug, name: c.name }))} current={client?.slug ?? null} />
            <LogoutButton />
          </div>
        </header>

        <main className="flex-1 py-5">{children}</main>

        <footer className="flex flex-wrap items-center justify-between gap-2 border-t border-dashed border-[color:var(--line)] py-4 text-[11px] text-[color:var(--pencil)]">
          <span>
            {FIRM_NAME} · {role === "avocat" ? "avocat" : "client"}
            {client?.importedAt && ` · registre INPI importé le ${formatDate(client.importedAt)}`}
            {client?.dataUpdatedAt && ` (données INPI du ${client.dataUpdatedAt})`}
          </span>
          <span className="flex items-center gap-4">
            <Link href="/architecture" className="btn-quiet hidden text-[12px] md:inline">
              {PRODUCTION_HOSTING_NOTE}
            </Link>
            <Link href="/historique" className="btn-quiet text-[12px]">
              historique
            </Link>
          </span>
        </footer>
      </div>
    </AppProvider>
  );
}
