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
- Never invent Boucheron records. The fictitious "Maison Exemple" client exists only in local development (IPW_DEV_MEMORY=1), never in production.
- Keys never go in the repo or the chat; they live in Railway variables.

## Done

- Step 1 UI, then rebuilt (30 Oct): dashboard (key figures, coverage / search map, live search results, deadlines, assistant), Marques (DB-backed edits, per-mark search button, per-mark log xlsx/csv), discreet Historique, Architecture cible page, hand-drawn look (roughjs, Shantell Sans).
- Supabase tables: clients, marks, events, watch_runs, findings, raw_imports, assistant_logs; private storage bucket `evidence`.
- Email assistant: Resend webhook `/api/inbound` → intent → onboard / add marks / start search / answer / note → history + reply.
- Assistant agent (`lib/agent.ts`) with tools; link verification; every tool call logged to `assistant_logs`.

## To check on first real use

- INPI field mapping (`lib/sources/inpi.ts` normalize) against the raw answers stored in `raw_imports`.
- Légifrance search body and Judilibre parameters against `assistant_logs`.
- SearchAPI result keys for each engine.
