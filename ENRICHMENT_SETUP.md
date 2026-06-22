# Enrichment Pipeline — Setup Guide

This guide walks you through everything needed to run the rebuilt OSU scholar
enrichment pipeline: installing dependencies, obtaining API keys, configuring
the `.env` file, and running the pipeline.

---

## 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### Install the Playwright browser (needed for Tier 3 Cloudflare bypass)

```bash
playwright install chromium
```

> The bypass ladder is graceful-optional: if `curl_cffi` or the Playwright
> browser is missing, the pipeline falls back to plain `httpx` and just logs
> a warning. But without them the currently-blocked sources (Google Scholar,
> Cloudflare-walled pages) won't be recovered.

---

## 2. Environment variables

Add the following to your `.env` file at the repo root. **Only `MONGODB_URI`
and `OPENAI_API_KEY` are strictly required.** Everything else is optional —
the corresponding collector will skip gracefully if its key is absent, but
adding a key increases rate limits or unlocks a source.

```bash
# -- CORE INFRASTRUCTURE (required) -----------------------------------
MONGODB_URI=mongodb+srv://...                    # see §3.1
OPENAI_API_KEY=sk-...                            # see §3.2

# -- PINECONE (required for RAG sync) ---------------------------------
PINECONE_API_KEY=...                             # see §3.3
PINECONE_ENVIRONMENT=us-east-1
PINECONE_INDEX_NAME=osu-scholars

# -- ENRICHMENT SOURCE KEYS (all optional) ----------------------------
OPENALEX_EMAIL=you@osu.edu                       # §3.4  (polite pool; free)
UNPAYWALL_EMAIL=you@osu.edu                      # §3.5  (free, email only)
NCBI_API_KEY=...                                 # §3.6  (raises PMC limits)
GITHUB_TOKEN=ghp_...                             # §3.7  (60→5000 req/hr)
HF_TOKEN=hf_...                                  # §3.8  (Hugging Face)
CORE_API_KEY=...                                 # §3.9  (OA full-text)
ALTMETRIC_API_KEY=...                            # §3.10 (impact metrics)
PATENTSVIEW_API_KEY=...                          # §3.11 (USPTO patents)
ZENODO_ACCESS_TOKEN=...                          # §3.12 (datasets)
YOUTUBE_API_KEY=AIza...                          # §3.13 (lecture metadata)
SEMANTIC_SCHOLAR_API_KEY=...                     # §3.14 (raises S2 rate limit 10x)

# -- TUNABLES (defaults shown) -----------------------------------------
ENRICHMENT_OUTPUT_DIR=output/osu_faculty_run
UNPAYWALL_MAX_DOIS=60
ALTMETRIC_MAX_DOIS=40
OPENCITATIONS_MAX_DOIS=40
YOUTUBE_TRANSCRIPT_MAX=20
YOUTUBE_TRANSCRIPT_CHAR_CAP=15000
PMC_MAX_ARTICLES=20
PMC_FULLTEXT_CAP=8000
```

---

## 3. Getting each API key (step-by-step)

### 3.1 MongoDB Atlas — required
1. Go to https://cloud.mongodb.com → sign up (free tier works).
2. Create a cluster → Database Access → add a user with password.
3. Network Access → add your IP (or `0.0.0.0/0` for dev).
4. Connect → "Drivers" → copy the connection string.
5. Paste into `MONGODB_URI`. Database name in use: `ngo_profiles`.

### 3.2 OpenAI — required (for LLM summarization)
1. https://platform.openai.com/api-keys → **Create new secret key**.
2. Add at least $5 credits under **Billing**.
3. Paste into `OPENAI_API_KEY`.

### 3.3 Pinecone — required for vector DB
1. https://app.pinecone.io → sign up (free tier: 1 index, 100K vectors).
2. Create an index named `osu-scholars` with dimension matching your
   embedder (1536 for `text-embedding-3-small`, 3072 for `-large`).
3. API Keys → copy → paste into `PINECONE_API_KEY`.

### 3.4 OpenAlex — free, email-only "polite pool"
- No signup required. Just put your email in `OPENALEX_EMAIL`.
- This moves you into the polite pool (10 req/s) and gives priority support.
- Docs: https://docs.openalex.org/how-to-use-the-api/rate-limits-and-authentication

### 3.5 Unpaywall — free, email-only
- No signup. Put your email in `UNPAYWALL_EMAIL` (reuses `OPENALEX_EMAIL`
  automatically if you don't set it).
- Docs: https://unpaywall.org/products/api

### 3.6 NCBI API Key (PubMed Central OA) — free
1. Log into https://www.ncbi.nlm.nih.gov/account/
2. **Account Settings** → **API Key Management** → **Create an API Key**.
3. Paste into `NCBI_API_KEY`. Rate limit goes from 3/s → 10/s.

### 3.7 GitHub Personal Access Token — free
1. https://github.com/settings/tokens → **Generate new token (classic)**.
2. No scopes needed for public data; select nothing.
3. Copy → paste into `GITHUB_TOKEN`. Limit goes from 60/hr → 5000/hr.

### 3.8 Hugging Face token — free
1. https://huggingface.co/settings/tokens → **New token** → Read.
2. Copy → paste into `HF_TOKEN`. Optional; public data works without it.

### 3.9 CORE API (OA full-text aggregator) — free
1. https://core.ac.uk/services/api → **Register for an API key**.
2. Free tier: 1,000 requests/day, 10 req/min.
3. Paste into `CORE_API_KEY`. **If absent, the `core_api` collector
   skips gracefully** and the pipeline runs the other 32 sources.

### 3.10 Altmetric API — free for academic use
1. Public per-DOI endpoint works without a key (rate-limited).
2. For higher limits: email `support@altmetric.com` from your `.edu` address
   requesting non-commercial academic access.
3. Paste into `ALTMETRIC_API_KEY` when you receive it.

### 3.11 PatentsView API — free
1. https://patentsview.org/apis/keyrequest → submit the short form.
2. Key emailed within a few minutes.
3. Paste into `PATENTSVIEW_API_KEY`. Works without a key but with stricter
   limits.

### 3.12 Zenodo access token — free
1. https://zenodo.org/account/settings/applications/tokens/new/
2. Check the `deposit:actions` scope is **not** needed for reads; just
   create a token with default (read) scope.
3. Paste into `ZENODO_ACCESS_TOKEN`. Optional.

### 3.13 YouTube Data API — free
1. https://console.cloud.google.com → create a project.
2. **APIs & Services** → **Library** → enable **YouTube Data API v3**.
3. **Credentials** → **Create Credentials** → **API key**.
4. Paste into `YOUTUBE_API_KEY`. Free quota: 10,000 units/day.
5. Without this, `youtube_lectures` is disabled; `youtube_transcripts`
   depends on it, so it's also disabled. This is the only "free but
   required" key if you want YouTube coverage.

### 3.14 Semantic Scholar API Key — highly recommended
The `semantic_scholar` collector is our most rate-limit-sensitive source. Without a key, S2 shares one bucket across all anonymous clients and will 429 you after ~3 quick calls.

**Steps:**
1. Go to https://www.semanticscholar.org/product/api
2. Click **Request API Key** → short form (name, email, institution, use case).
3. Key arrives by email within minutes to a day.
4. Paste into `.env`:
   ```
   SEMANTIC_SCHOLAR_API_KEY=...
   ```
5. The collector auto-detects the key, sends `x-api-key` header, and cuts `rate_limit_delay` from 3s → 1.2s per call.

### Sources that need NO key at all
The following collectors run out of the box with no signup:
`web_search`, `semantic_scholar`, `openalex` (but email recommended),
`orcid`, `crossref`, `nsf_grants`, `nih_grants`, `osu_courses`,
`rate_my_professor`, `osu_news`, `google_news`, `osu_expertise`,
`arxiv`, `biorxiv`, `wikidata`, `wikipedia`, `clinicaltrials`,
`usaspending`, `opencitations`, `paperswithcode`, `figshare`, `osf`,
`gdelt`, `unpaywall` (email only), `google_scholar` (but frequently
blocked; relies on bypass).

---

## 4. Run the pipeline

```bash
# List every available source (33 total) and exit
python run_enrichment_pipeline.py --list-sources

# Enrich a small batch first (sanity check)
python run_enrichment_pipeline.py --limit 5

# Re-run ONLY the new collectors on already-enriched profiles, merging
# with existing enrichment.json so you don't lose prior data:
python run_enrichment_pipeline.py --limit 20 \
  --re-enrich-sources youtube_transcripts,unpaywall,altmetric,opencitations,\
wikidata,wikipedia,github,patentsview,clinicaltrials,arxiv,biorxiv,pmc_oa,\
zenodo,figshare,osf,usaspending,gdelt,huggingface,paperswithcode,core_api

# Re-process profiles where >50% of collectors previously failed
python run_enrichment_pipeline.py --re-enrich-failed

# Target one specific professor by name or profile_id
python run_enrichment_pipeline.py --name "Smith"
python run_enrichment_pipeline.py --profile-id 01409eae-9cd1-4efc-bd7e-038818d41a36

# Enrich only (no chunking / Pinecone / Mongo sync) — fastest feedback loop
python run_enrichment_pipeline.py --limit 3 --skip-chunking \
  --sources youtube_transcripts,unpaywall,wikidata,github,patentsview
```

Each profile writes:
- `output/osu_faculty_run/profiles/{id}/enrichment.json`
- `output/osu_faculty_run/profiles/{id}/enrichment_text.txt`
- MongoDB `enrichment_raw` collection document

---

## 5. Sync enriched profiles into `scholars` (Mongo + Pinecone)

```bash
python sync_profiles_to_mongodb.py
```

This LLM-processes the merged enrichment text, chunks it, embeds it into
Pinecone, and writes the structured `scholars` document that the frontend
admin app consumes.

---

## 6. Verify the rebuild worked

Expected results after rebuild on a typical profile:
- `summary.successful_sources` rises from ~4 (old) to **15+** (new).
- `confidence.overall_confidence` rises from ~0.2 → **0.5–0.8**.
- `confidence.differentiator_hits` ≥ 3 for most profiles
  (YouTube transcript, Wikidata, patents, GitHub, or trials).
- `enrichment_text.txt` grows from ~5K → **30K–150K** chars.

Sanity-check one profile:
```bash
cat output/osu_faculty_run/profiles/<profile_id>/enrichment.json | \
  python -c "import sys, json; d = json.load(sys.stdin); \
             print('sources ok:', d['summary']['successful_sources'], \
                   '/', d['summary']['total_sources_queried']); \
             print('confidence:', d['confidence']['overall_confidence']); \
             print('differentiator hits:', d['confidence'].get('differentiator_hits'))"
```

---

## 7. Troubleshooting

**`ImportError: curl_cffi`** — run `pip install curl_cffi`. The bypass ladder
will still work at Tier 1 (httpx direct) without it, but Cloudflare-walled
pages will continue to fail.

**`playwright._impl._errors.Error: Executable doesn't exist`** — run
`playwright install chromium`.

**`403 Forbidden` from Google Scholar** — expected; `google_scholar.py` now
escalates to `curl_cffi`. If you still see 403, the Cloudflare challenge is
stricter; set `bypass_tier="playwright"` explicitly for that collector.

**`core_api` always fails** — you didn't set `CORE_API_KEY`. That's fine;
it's disabled automatically via `API_KEY_REQUIREMENTS`.

**`youtube_transcripts` always fails** — check that `youtube_lectures` ran
first and produced videos. Without upstream video IDs it has nothing to fetch.

**Pipeline very slow** — lower `max_concurrent` in the orchestrator
constructor, or disable heavy collectors with
`--sources=<comma-list>` for testing.

---

## 8. Minimum viable key set (if you want to get started fast)

If you want to validate the pipeline end-to-end with the smallest setup:

1. `MONGODB_URI` (required)
2. `OPENAI_API_KEY` (required)
3. `PINECONE_API_KEY` (required)
4. `OPENALEX_EMAIL` = your email (free, 30 sec)
5. `GITHUB_TOKEN` (free, 1 min) — 5000x rate limit increase
6. `NCBI_API_KEY` (free, 2 min) — unlocks PMC full text
7. `YOUTUBE_API_KEY` (free, 5 min via Google Cloud Console) — unlocks
   the single biggest differentiator (YouTube transcripts)

Everything else can wait. The pipeline will run with just these and cover
~25 of the 33 sources.
