import ExcelJS from "exceljs";
import { currentRole } from "@/lib/auth";
import { currentClient } from "@/lib/data";
import { appUrl } from "@/lib/email";
import { countryName } from "@/lib/territories";
import * as store from "@/lib/store";

export const dynamic = "force-dynamic";

const COLUMNS: { header: string; width: number }[] = [
  { header: "Recherche du", width: 18 },
  { header: "Déclenchée par", width: 14 },
  { header: "Détecté le", width: 18 },
  { header: "Type", width: 12 },
  { header: "Source", width: 16 },
  { header: "Plateforme / vendeur", width: 24 },
  { header: "Titre", width: 44 },
  { header: "URL", width: 50 },
  { header: "Prix", width: 10 },
  { header: "Devise", width: 8 },
  { header: "Pays", width: 16 },
  { header: "Titulaire du site", width: 30 },
  { header: "Source titulaire", width: 18 },
  { header: "Confiance (%)", width: 13 },
  { header: "Motif", width: 40 },
  { header: "Statut", width: 12 },
  { header: "Commentaire", width: 30 },
  { header: "Capture", width: 40 },
  { header: "Empreinte SHA-256", width: 66 },
  { header: "Capturé le", width: 18 },
  { header: "Analyse urlscan", width: 40 },
];

const TRIGGERS = { bouton: "bouton", assistant: "assistant", email: "email" } as const;

export async function GET(request: Request, { params }: { params: { id: string } }) {
  if (!currentRole()) return new Response("Non connecté.", { status: 401 });
  const client = await currentClient();
  if (!client) return new Response("Aucun client.", { status: 404 });
  const markId = decodeURIComponent(params.id);
  const mark = (await store.listMarks(client.id)).find((m) => m.id === markId);
  const rows = await store.markFindings(client.id, markId);
  const format = new URL(request.url).searchParams.get("format") === "csv" ? "csv" : "xlsx";

  const values = rows.map(({ finding: f, run }) => [
    run ? new Date(run.startedAt) : null,
    run ? TRIGGERS[run.trigger] : "",
    new Date(f.foundAt),
    f.kind,
    f.source,
    f.platform ?? "",
    f.title ?? "",
    f.url,
    f.price,
    f.currency ?? "",
    f.country ? countryName(f.country) : "",
    f.company?.name ?? "",
    f.company?.source ?? "",
    f.confidence,
    f.reason ?? "",
    f.status,
    f.comment ?? "",
    f.screenshotPath ? `${appUrl()}/api/evidence/${f.screenshotPath}` : "",
    f.sha256 ?? "",
    f.capturedAt ? new Date(f.capturedAt) : null,
    f.scanUrl ?? "",
  ]);

  const base = `journal-${client.slug}-${(mark?.name ?? markId).toLowerCase().replace(/[^a-z0-9]+/g, "-")}-${new Date().toISOString().slice(0, 10)}`;

  if (format === "csv") {
    // Semicolons and a BOM so French Excel opens it correctly.
    const cell = (v: unknown) => {
      if (v === null || v === undefined) return "";
      const s = v instanceof Date ? v.toLocaleString("fr-FR", { timeZone: "Europe/Paris" }) : String(v);
      return /[;"\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
    };
    const csv = [COLUMNS.map((c) => c.header), ...values].map((r) => r.map(cell).join(";")).join("\r\n");
    return new Response(`﻿${csv}`, { headers: { "Content-Type": "text/csv; charset=utf-8", "Content-Disposition": `attachment; filename="${base}.csv"` } });
  }

  const wb = new ExcelJS.Workbook();
  wb.creator = "Registre PI";
  const ws = wb.addWorksheet((mark?.name ?? markId).slice(0, 31).replace(/[\\/*?:[\]]/g, " "));
  ws.columns = COLUMNS.map((c) => ({ header: c.header, width: c.width }));
  values.forEach((v) => ws.addRow(v));
  ws.getRow(1).font = { bold: true };
  ws.views = [{ state: "frozen", ySplit: 1 }];
  for (const col of [1, 3, 20]) ws.getColumn(col).numFmt = "dd/mm/yyyy hh:mm";
  const buffer = await wb.xlsx.writeBuffer();
  return new Response(buffer, {
    headers: { "Content-Type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "Content-Disposition": `attachment; filename="${base}.xlsx"` },
  });
}
