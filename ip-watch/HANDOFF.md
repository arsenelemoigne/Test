# Handoff — Registre PI (state on 29 Oct 2026)

Context: EFB "LAB – Atelier Nouveaux Services" project. A fictitious law firm offers recurring, lawyer-led IP portfolio management. This app is the showcase. Showcase client: Boucheron (Quatre ring); Boucheron is not a client and the UI says so.

Deadlines: final deliverables 31 Oct 2026; jury pitch 13–20 Nov 2026.

## Where things live

- Code: `arsenelemoigne/Test`, branch `claude/friendly-tesla-mz1yz2`, folder `ip-watch/` (the repo root is an unrelated app, "meshly"; the root `tsconfig.json` excludes `ip-watch`).
- Hosting: Railway, project `fantastic-endurance`, service `friendly-art`, root directory `/ip-watch`, auto-deploys from the branch above. URL: https://friendly-art-production-e161.up.railway.app. Planned custom domain: `pi.getmanda.eu` (CNAME at the getmanda.eu DNS provider). Do not touch the `zimi-geo` / `fabulous-success` services or `wa.getmanda.eu`.
- Supabase: project `registre-pi` (ref `qkigxbbkpynydgttqncg`, eu-west-3 Paris, free plan, org "arsenelemoigne's Org"). Created, not used yet.
- Railway variables set: `DEMO_PASSWORD`, `OPENROUTER_API_KEY`, `OPENROUTER_MODEL`, `LAWYER_EMAIL`, `INPI_USERNAME`, `INPI_PASSWORD`. Missing: `SEARCHAPI_API_KEY`, `URLSCAN_API_KEY`, `RESEND_API_KEY`. `/api/health` shows which are set.

## Decisions agreed with the user

- AI via OpenRouter for the demo. The site only states EU hosting (Scaleway) as the production target, never as current fact.
- Evidence pack = screenshot + page copy + URL + timestamp + SHA-256; it does not replace a commissaire de justice internet constat (NF Z67-147). The UI says so.
- Demo behind one shared password (Avocat / Client roles), noindex.
- Never invent Boucheron records. Until real data is imported, only the clearly fictitious "Maison Exemple" client is shown.
- Keys never go in the repo or the chat; they live in Railway variables.

## Done (step 1)

Login, overview, editable portfolio with edit history (browser storage), deadlines (renewal window −6 months, grace +6 months, reminders D-180/90/30), black-and-white map (world-atlas 50m, EU expanded, lapsed designations excluded), watch log with evidence pack, Excel export (exceljs), assistant grounded in the client JSON (`/api/ask`).

## Next steps

1. INPI import for Boucheron (marks FR/EU/WO, designs FR/WO) → `data/clients/boucheron.json` + register in `data/clients/index.ts`. API notes (from the user's earlier pilot):
   - Base `https://api-gateway.inpi.fr`. `GET /services/uaa/api/authenticate` (401 expected, sets `XSRF-TOKEN` cookie), then `POST /auth/login {username, password, rememberMe:true}` with header `X-XSRF-TOKEN`. Session ≈ 5 min; quota ≈ 100 calls (429).
   - Marks: `POST /services/apidiffusion/api/marques/search {collections:['FR','EU','WO'], query:'[DEPOSANT=BOUCHERON]', fields:['*'], position:0, size:200}`. Also `[Mark_Exp=...]`, `[ClassNumber=(14 OU 3)]`, combinable with ET / OU.
   - Designs: `POST /services/apidiffusion/api/modeles/search {collections:['FR','WO'], query:'[DEPOSANT=...]'}` (a query without `[FIELD=...]` returns 500). No EU designs, no patents.
   - Notice: `GET /services/apidiffusion/api/marques/notice/FR1234567` (classes, dates, opposition deadline, designated countries with decisions).
   - Data date (licence condition, must be cited): `GET /services/apidiffusion/api/marques/metadata`.
   - The holder may appear under several entity names; check variants.
2. Shared edits in Supabase instead of browser storage.
3. Resend: deadline reminder emails to `LAWYER_EMAIL`; inbound assistant address (`.resend.app`), allow-listed senders only.
4. Watch (monthly + on demand): SearchAPI (`google_lens` visual matches on the ring photo, `google_shopping`, `amazon_search`, `ebay`), urlscan (scan + screenshot + SHA-256, search for lookalike domains), RDAP (`rdap.nic.fr`, `rdap.verisign.com`), `recherche-entreprises.api.gouv.fr` for French sellers.
