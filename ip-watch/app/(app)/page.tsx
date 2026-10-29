import Link from "next/link";
import { currentRole } from "@/lib/auth";
import { assistantEmail } from "@/lib/config";
import { currentClient } from "@/lib/data";
import * as store from "@/lib/store";
import { Dashboard } from "@/components/Dashboard";
import { RoughBox } from "@/components/RoughBox";

export const dynamic = "force-dynamic";

export default async function DashboardPage({ searchParams }: { searchParams: { run?: string } }) {
  const client = await currentClient();
  if (!client) {
    const email = assistantEmail();
    return (
      <RoughBox seed={3} className="mx-auto mt-10 max-w-lg">
        <div className="space-y-3 p-8 text-center">
          <h1 className="hand text-[24px] font-semibold">Aucun client pour l&apos;instant</h1>
          <p className="text-[14px] text-[color:var(--pencil)]">
            {email ? (
              <>
                Écrivez ou transférez un email à <span className="hand text-[color:var(--ink)]">{email}</span> : le registre se remplit tout seul.
              </>
            ) : (
              "Ajoutez un client : nom, SIREN ou lien vers sa fiche suffisent."
            )}
          </p>
          {currentRole() === "avocat" && (
            <Link href="/nouveau" className="btn-ink">
              + nouveau client
            </Link>
          )}
        </div>
      </RoughBox>
    );
  }
  const [marks, runs] = await Promise.all([store.listMarks(client.id), store.latestRuns(client.id, 10)]);
  return <Dashboard key={client.id} client={client} marks={marks} runs={runs} initialRunId={searchParams.run ?? null} />;
}
