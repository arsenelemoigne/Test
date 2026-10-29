import { redirect } from "next/navigation";
import { currentRole } from "@/lib/auth";
import { currentClient, listClients } from "@/lib/data";
import { APP_NAME, assistantEmail, FIRM_NAME, lawyerEmail, PRODUCTION_HOSTING_NOTE } from "@/lib/config";
import { EditsProvider } from "@/components/EditsProvider";
import { Nav } from "@/components/Nav";
import { ClientSwitcher } from "@/components/ClientSwitcher";
import { ExportButton } from "@/components/ExportButton";
import { formatDate } from "@/lib/format";

export default function AppLayout({ children }: { children: React.ReactNode }) {
  const role = currentRole();
  if (!role) redirect("/login");
  const client = currentClient();
  const settings = {
    lawyerEmail: lawyerEmail(),
    assistantEmail: assistantEmail(),
    assistantEnabled: Boolean(process.env.OPENROUTER_API_KEY),
  };

  return (
    <EditsProvider client={client} role={role} settings={settings}>
      <div className="mx-auto flex min-h-screen max-w-7xl flex-col px-4 sm:px-6">
        <header className="flex flex-wrap items-center justify-between gap-3 border-b border-black py-4">
          <div className="flex items-baseline gap-3">
            <span className="text-lg font-semibold tracking-tight">{APP_NAME}</span>
            <span className="label hidden sm:inline">{FIRM_NAME}</span>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <ClientSwitcher clients={listClients()} current={client.slug} />
            <span className="chip">{role === "avocat" ? "Avocat" : "Client"}</span>
            <ExportButton />
            <form method="POST" action="/api/logout">
              <button className="btn">Quitter</button>
            </form>
          </div>
        </header>

        <Nav />

        {client.fictitious && (
          <div className="mt-4 border border-dashed border-black px-3 py-2 font-mono text-xs">
            DONNÉES FICTIVES — {client.disclaimer}
          </div>
        )}

        <main className="flex-1 py-6">{children}</main>

        <footer className="space-y-1 border-t border-black py-4 text-[11px] leading-relaxed text-neutral-500">
          <p>
            Démo pédagogique — Projet LAB, École de formation professionnelle des barreaux de Paris. {client.disclaimer}
          </p>
          <p>
            Sources :{" "}
            {client.sources.map((s, i) => (
              <span key={s.name}>
                {i > 0 && " · "}
                {s.name}
                {s.updatedAt && ` (mise à jour du ${formatDate(s.updatedAt)})`}
              </span>
            ))}
            . Import du {formatDate(client.importedAt)}.
          </p>
          <p>{PRODUCTION_HOSTING_NOTE}</p>
        </footer>
      </div>
    </EditsProvider>
  );
}
