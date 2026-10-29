import { ArchitectureDiagram } from "@/components/ArchitectureDiagram";
import { RoughBox } from "@/components/RoughBox";

const PROCESSORS = [
  { role: "Hébergement applicatif", who: "Scaleway — Serverless Containers", where: "Paris, Île-de-France (FR)", data: "traitements, tâches de fond" },
  { role: "Base de données", who: "Scaleway — Managed PostgreSQL", where: "Paris, Île-de-France (FR)", data: "registre, journaux de veille, historique" },
  { role: "Stockage des preuves", who: "Scaleway — Object Storage", where: "Paris, Île-de-France (FR)", data: "captures d'écran, empreintes" },
  { role: "Inférence IA", who: "Mistral AI", where: "Union européenne (siège Paris)", data: "questions, extraits du registre, emails à trier" },
  { role: "Emails entrants et sortants", who: "Brevo", where: "France / UE", data: "emails adressés à l'assistant, réponses" },
];

const SOURCES = [
  { name: "Légifrance", who: "DILA (FR), via PISTE", use: "textes en vigueur", access: "OAuth PISTE" },
  { name: "Judilibre", who: "Cour de cassation (FR), via PISTE", use: "jurisprudence Cass. et cours d'appel", access: "OAuth PISTE" },
  { name: "INPI — API diffusion", who: "INPI (FR)", use: "marques FR, UE, internationales ; dessins", access: "compte INPI" },
  { name: "Annuaire des entreprises", who: "DINUM (FR)", use: "SIREN, dirigeants, siège", access: "libre" },
  { name: "BODACC", who: "DILA (FR)", use: "annonces légales", access: "libre" },
  { name: "EUR-Lex · CURIA", who: "Office des publications de l'UE (LU) · CJUE", use: "droit et jurisprudence de l'UE", access: "libre" },
  { name: "RDAP", who: "AFNIC (FR), Verisign", use: "titulaire d'un nom de domaine", access: "libre" },
  { name: "urlscan.io", who: "urlscan GmbH, Aachen (DE)", use: "captures horodatées, domaines similaires", access: "clé API" },
  { name: "Recherche d'annonces et visuelle", who: "API de résultats (Shopping, Lens, Amazon, eBay)", use: "veille des contrefaçons", access: "clé API" },
];

export default function ArchitecturePage() {
  return (
    <div className="mx-auto max-w-5xl space-y-8">
      <div>
        <h1 className="hand text-[28px] font-semibold leading-tight">Architecture cible</h1>
        <p className="max-w-3xl text-[14px] text-[color:var(--pencil)]">
          En production, l&apos;application et l&apos;inférence sont hébergées dans l&apos;Union européenne. Les données des clients restent chez des sous-traitants européens liés par contrat ; seules des
          données publiques partent vers les sources.
        </p>
      </div>

      <RoughBox seed={31}>
        <div className="p-4">
          <ArchitectureDiagram />
        </div>
      </RoughBox>

      <section>
        <h2 className="hand text-[18px]">Sous-traitants (art. 28 RGPD)</h2>
        <p className="text-[13px] text-[color:var(--pencil)]">Ils traitent les données du cabinet et de ses clients pour son compte : contrat de sous-traitance, hébergement dans l&apos;UE, aucun transfert hors UE.</p>
        <div className="mt-2 overflow-x-auto">
          <table className="table-base min-w-[720px]">
            <thead>
              <tr>
                <th>Rôle</th>
                <th>Prestataire</th>
                <th>Localisation</th>
                <th>Données traitées</th>
              </tr>
            </thead>
            <tbody>
              {PROCESSORS.map((p) => (
                <tr key={p.role}>
                  <td className="hand">{p.role}</td>
                  <td>{p.who}</td>
                  <td>{p.where}</td>
                  <td className="text-[color:var(--pencil)]">{p.data}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section>
        <h2 className="hand text-[18px]">Sources et API interrogées</h2>
        <p className="text-[13px] text-[color:var(--pencil)]">
          Ce ne sont pas des sous-traitants : l&apos;application leur adresse des requêtes portant sur des informations publiques (nom d&apos;une marque, d&apos;une société, URL publique), sans
          transmettre de données confidentielles du client.
        </p>
        <div className="mt-2 overflow-x-auto">
          <table className="table-base min-w-[720px]">
            <thead>
              <tr>
                <th>Source</th>
                <th>Éditeur</th>
                <th>Usage</th>
                <th>Accès</th>
              </tr>
            </thead>
            <tbody>
              {SOURCES.map((s) => (
                <tr key={s.name}>
                  <td className="hand">{s.name}</td>
                  <td>{s.who}</td>
                  <td className="text-[color:var(--pencil)]">{s.use}</td>
                  <td className="whitespace-nowrap">{s.access}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <RoughBox seed={32}>
        <div className="space-y-2 p-5 text-[13.5px]">
          <h2 className="hand text-[16px]">Règles</h2>
          <ul className="list-disc space-y-1 pl-5">
            <li>Secret professionnel : les échanges avec le client et le registre restent dans l&apos;application et chez ses sous-traitants européens.</li>
            <li>Ce qui sort vers les sources : uniquement le nom d&apos;une marque, d&apos;une société ou une URL publique.</li>
            <li>Chaque réponse de l&apos;assistant cite ses sources ; un lien qui n&apos;a pas été renvoyé par une source est signalé comme non vérifié.</li>
            <li>Les captures sont horodatées et identifiées par leur empreinte SHA-256 ; elles ne remplacent pas un constat de commissaire de justice.</li>
          </ul>
          <p className="pt-1 text-[12px] text-[color:var(--pencil)]">Démo : Railway, Supabase, OpenRouter et Resend tiennent provisoirement ces rôles, en attendant la mise en production.</p>
        </div>
      </RoughBox>
    </div>
  );
}
