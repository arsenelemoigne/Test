"use client";

import { useState } from "react";
import { useData } from "./EditsProvider";
import { deadlineFor, DEADLINE_LABELS } from "@/lib/deadlines";
import { countryName, OFFICE_LABELS, rightCountries } from "@/lib/territories";
import { FIELD_LABELS, KIND_LABELS, STATUS_LABELS, TYPE_LABELS } from "@/lib/labels";

function day(d: Date | string | null | undefined): Date | null {
  if (!d) return null;
  const out = typeof d === "string" ? new Date(d) : d;
  return Number.isNaN(out.getTime()) ? null : out;
}

export function ExportButton() {
  const { client, rights, findings, history } = useData();
  const [busy, setBusy] = useState(false);

  const run = async () => {
    setBusy(true);
    try {
      const ExcelJS = (await import("exceljs")).default;
      const wb = new ExcelJS.Workbook();
      wb.creator = "Registre PI (démo)";

      const header = (ws: import("exceljs").Worksheet) => {
        ws.getRow(1).font = { bold: true };
        ws.views = [{ state: "frozen", ySplit: 1 }];
      };

      const titres = wb.addWorksheet("Titres");
      titres.columns = [
        { header: "Identifiant", key: "id", width: 18 },
        { header: "Type", key: "type", width: 16 },
        { header: "Office", key: "office", width: 24 },
        { header: "Nom", key: "name", width: 32 },
        { header: "Titulaire", key: "owner", width: 28 },
        { header: "Classes", key: "classes", width: 14 },
        { header: "Dépôt", key: "filing", width: 12, style: { numFmt: "dd/mm/yyyy" } },
        { header: "Enregistrement", key: "registration", width: 14, style: { numFmt: "dd/mm/yyyy" } },
        { header: "Expiration", key: "expiry", width: 12, style: { numFmt: "dd/mm/yyyy" } },
        { header: "Statut", key: "status", width: 16 },
        { header: "Pays", key: "countries", width: 40 },
        { header: "Notes", key: "notes", width: 30 },
      ];
      for (const r of rights) {
        titres.addRow({
          id: r.id,
          type: TYPE_LABELS[r.type],
          office: OFFICE_LABELS[r.office],
          name: r.name,
          owner: r.owner,
          classes: r.classes.join(", "),
          filing: day(r.filingDate),
          registration: day(r.registrationDate),
          expiry: day(r.expiryDate),
          status: r.status,
          countries: rightCountries(r).map(countryName).join(", "),
          notes: r.notes ?? "",
        });
      }
      header(titres);

      const echeances = wb.addWorksheet("Échéances");
      echeances.columns = [
        { header: "Identifiant", key: "id", width: 18 },
        { header: "Nom", key: "name", width: 32 },
        { header: "Expiration", key: "expiry", width: 12, style: { numFmt: "dd/mm/yyyy" } },
        { header: "Date calculée", key: "computed", width: 13 },
        { header: "Ouverture du renouvellement", key: "opens", width: 24, style: { numFmt: "dd/mm/yyyy" } },
        { header: "Fin du délai de grâce", key: "grace", width: 20, style: { numFmt: "dd/mm/yyyy" } },
        { header: "État", key: "state", width: 22 },
        { header: "Jours restants", key: "days", width: 14 },
      ];
      for (const r of rights) {
        const d = deadlineFor(r);
        echeances.addRow({
          id: r.id,
          name: r.name,
          expiry: d.expiry,
          computed: d.computed ? "oui" : "non",
          opens: d.windowOpens,
          grace: d.graceEnds,
          state: DEADLINE_LABELS[d.state],
          days: d.daysLeft,
        });
      }
      header(echeances);

      const veille = wb.addWorksheet("Veille");
      veille.columns = [
        { header: "Identifiant", key: "id", width: 10 },
        { header: "Détecté le", key: "foundAt", width: 18, style: { numFmt: "dd/mm/yyyy hh:mm" } },
        { header: "Type", key: "kind", width: 18 },
        { header: "Plateforme", key: "platform", width: 20 },
        { header: "Titre de l'annonce", key: "title", width: 40 },
        { header: "URL", key: "url", width: 50 },
        { header: "Prix", key: "price", width: 10 },
        { header: "Devise", key: "currency", width: 8 },
        { header: "Pays", key: "country", width: 16 },
        { header: "Vendeur", key: "seller", width: 24 },
        { header: "Société identifiée", key: "company", width: 30 },
        { header: "Identifiant société", key: "companyId", width: 18 },
        { header: "Source société", key: "companySource", width: 20 },
        { header: "Droits concernés", key: "rights", width: 30 },
        { header: "Pertinence (1-5)", key: "relevance", width: 15 },
        { header: "Statut", key: "status", width: 16 },
        { header: "Constat recommandé", key: "constat", width: 18 },
        { header: "Commentaire", key: "comment", width: 40 },
        { header: "Capture le", key: "capturedAt", width: 18, style: { numFmt: "dd/mm/yyyy hh:mm" } },
        { header: "Empreinte SHA-256", key: "sha", width: 66 },
        { header: "Capture d'écran", key: "screenshot", width: 40 },
      ];
      for (const f of findings) {
        veille.addRow({
          id: f.id,
          foundAt: day(f.foundAt),
          kind: KIND_LABELS[f.kind],
          platform: f.platform,
          title: f.title,
          url: f.url,
          price: f.price,
          currency: f.currency,
          country: f.country ? countryName(f.country) : "",
          seller: f.seller ?? "",
          company: f.company?.name ?? "",
          companyId: f.company?.id ?? "",
          companySource: f.company?.source ?? "",
          rights: f.matchedRightIds.join(", "),
          relevance: f.relevance,
          status: STATUS_LABELS[f.status],
          constat: f.constatRecommended ? "oui" : "non",
          comment: f.comment,
          capturedAt: day(f.evidence?.capturedAt),
          sha: f.evidence?.sha256 ?? "",
          screenshot: f.evidence?.screenshotUrl ?? "",
        });
      }
      header(veille);

      const journal = wb.addWorksheet("Modifications");
      journal.columns = [
        { header: "Date", key: "at", width: 18, style: { numFmt: "dd/mm/yyyy hh:mm" } },
        { header: "Par", key: "by", width: 10 },
        { header: "Élément", key: "kind", width: 12 },
        { header: "Identifiant", key: "id", width: 18 },
        { header: "Champ", key: "field", width: 22 },
        { header: "Avant", key: "prev", width: 30 },
        { header: "Après", key: "value", width: 30 },
      ];
      for (const h of history) {
        journal.addRow({
          at: day(h.at),
          by: h.by === "avocat" ? "Avocat" : "Client",
          kind: h.kind,
          id: h.id,
          field: FIELD_LABELS[h.field] ?? h.field,
          prev: String(h.prev ?? ""),
          value: String(h.value ?? ""),
        });
      }
      header(journal);

      const buffer = await wb.xlsx.writeBuffer();
      const blob = new Blob([buffer], { type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `registre-${client.slug}-${new Date().toISOString().slice(0, 10)}.xlsx`;
      a.click();
      URL.revokeObjectURL(url);
    } finally {
      setBusy(false);
    }
  };

  return (
    <button type="button" className="btn" onClick={run} disabled={busy}>
      {busy ? "Export…" : "Export Excel"}
    </button>
  );
}
