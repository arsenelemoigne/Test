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
  { name: "RDAP", who: "AFNIC (FR), Verisign, rdap.org", use: "titulaire et bureau d'enregistrement d'un domaine ; domaines détenus par la marque écartés", access: "libre" },
  { name: "Recherche web", who: "connecteur de recherche de l'inférence", use: "repli de l'assistant ; requêtes ciblées sur les copies (dupe, réplique, places de marché)", access: "inclus" },
  { name: "Google Lens · Google Shopping", who: "SearchApi LLC, New York (US)", use: "produits visuellement proches de la photo ; annonces avec prix et vendeur", access: "clé API" },
  { name: "Amazon · eBay", who: "SearchApi LLC, New York (US)", use: "annonces sur les places de marché", access: "clé API" },
  { name: "Google Cloud Vision", who: "Google, point d'accès UE (détection web)", use: "pages qui reprennent la photo du produit, à l'identique ou en partie", access: "clé API" },
  { name: "Pages officielles du titulaire", who: "site de la marque", use: "photo de référence du produit authentique", access: "libre" },
  { name: "urlscan.io", who: "urlscan GmbH, Aachen (DE)", use: "captures horodatées non publiques, domaines similaires", access: "clé API" },
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
          Ce ne sont pas des sous-traitants : l&apos;application leur adresse des requêtes portant sur des informations publiques (nom d&apos;une marque ou d&apos;une société, description et
          photo publiques d&apos;un produit, URL publique), sans transmettre de données confidentielles du client.
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
            <li>Ce qui sort vers les sources : uniquement le nom d&apos;une marque ou d&apos;une société, la description publique d&apos;un produit, une URL ou une photo publique (site officiel, registre).</li>
            <li>
              Recherche de contrefaçons : le modèle identifie d&apos;abord le produit protégé, puis interroge le web, les places de marché et la recherche par image ; les sites officiels et les
              domaines du titulaire sont écartés, et chaque page retenue est comparée visuellement au produit authentique. Les résultats restent « à vérifier » par l&apos;avocat.
            </li>
            <li>Chaque réponse de l&apos;assistant cite ses sources ; un lien qui n&apos;a pas été renvoyé par une source est signalé comme non vérifié.</li>
            <li>Les captures sont horodatées et identifiées par leur empreinte SHA-256 ; elles ne remplacent pas un constat de commissaire de justice.</li>
          </ul>
          <p className="pt-1 text-[12px] text-[color:var(--pencil)]">Démo : Railway, Supabase, OpenRouter et Resend tiennent provisoirement ces rôles, en attendant la mise en production.</p>
        </div>
      </RoughBox>
    </div>
  );
}
