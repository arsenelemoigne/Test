import { DiagnosticPanel } from "@/components/DiagnosticPanel";

export const dynamic = "force-dynamic";

export default function DiagnosticPage() {
  return (
    <div className="mx-auto max-w-4xl space-y-4">
      <div>
        <h1 className="hand text-[28px] font-semibold leading-tight">Diagnostic</h1>
        <p className="text-[13px] text-[color:var(--pencil)]">
          Un appel simple vers chaque source. Aucun crédit SearchAPI, urlscan ni IA n&apos;est consommé ; l&apos;INPI compte deux appels de quota.
        </p>
      </div>
      <DiagnosticPanel />
    </div>
  );
}
