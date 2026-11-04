# llm-ads: a privacy-first ad marketplace for LLM apps

Prototype. Any app that runs an LLM (a free chat app, an open-source client, an inference provider's free tier)
adds the SDK. Advertisers bid on the marketplace. The twist: **the catalog comes to the app, the conversation never leaves it.**

```
Advertisers ──► MARKETPLACE  (offers, bids, budgets)
                    │  signed catalog, bids included   (GET /catalog)
                    ▼
Publisher app ──► SDK: local relevance x bid auction ──► "Sponsored" card in the UI
                    │  only a signed click + aggregated view counts come back
                    ▼
MARKETPLACE bills the advertiser per click, pays the publisher 70%
```

## Run it

Node 22.18 or newer, no dependencies to install.

```bash
cd llm-ads
npm run marketplace          # http://localhost:4000
npm run demo                 # http://localhost:3001  (in a second terminal)
npm test                     # 16 tests
```

The demo works offline with a stub model. To use a real small model, point it at any OpenAI-compatible endpoint:

```bash
LLM_BASE_URL=http://localhost:11434/v1 LLM_MODEL=qwen2.5:3b npm run demo   # Ollama
```

With a real model the demo uses **tool mode**: the model decides when to call `get_sponsored_offers`.
Without one (or with `AD_MODE=card`) every user message is matched locally and the card appears beside the answer.

Try `I need running shoes, I have flat feet`, `which vpn should I use on public wifi`, then `What is the capital of France?` (no ad).
`GET http://localhost:4000/stats` shows spend, publisher earnings and platform revenue.

## How it works

| Step | Where | File |
|---|---|---|
| Advertisers set offers with approved claims, bid per click, budget | marketplace | `marketplace/store.ts`, `marketplace/seed.ts` |
| Marketplace publishes the whole catalog, signed with Ed25519 so bids can't be tampered with | marketplace | `store.ts` `buildCatalog`, `shared/crypto.ts` |
| App downloads and verifies the catalog (fails open: no ads, never a broken app) | publisher | `sdk/catalog.ts` |
| Relevance of each offer to the message, TF-IDF by default, embeddings pluggable | publisher | `sdk/relevance.ts` |
| Auction: score = relevance x bid, minimum relevance 0.25 so irrelevant high bidders never show | publisher | `sdk/auction.ts` |
| Optional LLM tool: the model asks for offers only when it sees purchase intent | publisher | `sdk/index.ts` |
| Click link signed by the publisher (HMAC, timestamp, nonce). Replays and forgeries redirect but aren't billed | both | `sdk/tracking.ts`, `store.ts` `recordClick` |
| Only per-offer view totals are reported | publisher | `sdk/tracking.ts` `ImpressionCounter` |

## How this differs from existing networks

| | ZeroClick, Lulu Ads, Koah... | llm-ads |
|---|---|---|
| What travels | The user's query goes to the ad server | The catalog goes to the app |
| Who picks the ad | Their server | The app, locally |
| What the ad network learns | What users ask | Which offer got clicked, and view totals |

## Not built yet

- **Fraud is the open problem.** A dishonest publisher can mint valid click links. Next steps: CPA billing
  (pay on purchase through an affiliate postback), per-publisher anomaly detection, holding payouts.
- Advertiser accounts, authentication and creative review (`POST /offers` is open).
- Persistence: everything is in memory and resets on restart.
- Second-price auction (today the winner pays its own bid).
- Category-sharded catalogs once there are too many offers to ship whole.
- Embedding scorer wired to a local model (the class exists, the demo uses TF-IDF).
- Legal: ads must stay clearly labeled (EU Unfair Commercial Practices Directive, DSA, FTC in the US).
  The publisher still sees conversations because it runs the model. The promise is that advertisers and the marketplace don't.
