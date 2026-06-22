## Daily Story Worker: Start Here

### 1. Prerequisites

- `MONGODB_URI` is set in `.env`
- `LLM_PROVIDER=vertex`
- `GCP_PROJECT_ID` and `GCP_LOCATION` are set
- `GOOGLE_APPLICATION_CREDENTIALS` points to a valid service-account JSON key
- `STORY_LLM_MODEL` (or `LLM_MODEL`) is set
- Source scholar data exists in Mongo collection `legend_scholars`
- (Optional) `STORY_ML_ENABLED=1` (default on)
- (Optional) `STORY_EMBEDDING_MODEL=all-MiniLM-L6-v2`
- (Optional) `STORY_EMBEDDING_LOCAL_ONLY=1` (default on; avoids model downloads)
- (Optional) `STORY_CROSS_ENCODER_ENABLED=1` (default on; reranks retrieval with a cross-encoder when available)
- (Optional) `STORY_CROSS_ENCODER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2`
- (Optional) `STORY_CROSS_ENCODER_LOCAL_ONLY=1` (default on; avoids downloading reranker weights at runtime)
- (Optional) `STORY_CROSS_ENCODER_TOP_K=24` (rerank pool size before final context selection)
- (Optional) `STORY_CROSS_ENCODER_WEIGHT=0.35` (blend weight against base dense+lexical score)
- (Optional) `STORY_TRENDS_ENABLED=1` (default on)
- (Optional) `STORY_TREND_PROVIDER=rss` (`rss`, `newsapi`, `gdelt`, `auto`)
- (Optional) `STORY_TREND_REGION=us` (used by NewsAPI)
- (Optional) `NEWSAPI_KEY=...` (required only if using `newsapi` provider)
- (Optional) `STORY_TREND_RSS_URLS=https://...,...` (comma-separated RSS feeds)
- (Optional) `STORY_TREND_USE_CACHE=1` (default on)
- (Optional) `STORY_TREND_CACHE_COLLECTION=daily_story_trend_issues`
- (Optional) `STORY_TREND_CACHE_TTL_HOURS=72`
- (Optional) `STORY_ENFORCE_PROFILE_QUALITY=1` (off by default)
- (Optional) `STORY_PROFILE_MIN_QUALITY_SCORE=60`
- (Optional) `STORY_STRICT_RELIABILITY=1` (default on)
- (Optional) `STORY_REQUIRE_VERIFIED_TREND_URL=1` (default on)
- (Optional) `STORY_ALLOW_CORPUS_ONLY_WHEN_NO_TREND=1` (default on; do not hard-fail when trend feeds are temporarily empty)
- (Optional) `STORY_MAX_OUTPUT_TOKENS=6200` (raise/limit model output budget for JSON stability)
- (Optional) `STORY_VERTEX_SCHEMA_ENFORCED=1` (default on; requests schema-constrained JSON from Vertex when SDK supports it)
- (Optional) `STORY_MIN_CLAIM_EVIDENCE_ITEMS=4`
- (Optional) `STORY_MIN_PARAGRAPH_OVERLAP=0.02`

Optional (recommended if your machine has broken proxy vars):

- `VERTEX_DISABLE_SYSTEM_PROXY=1`

### 2. First local test (no LLM)

```bash
python daily_story_worker.py --scholar-id 7e3a28d4-ff81-4085-aff1-d16426bce921 --dry-run
```

What this does:

- Reads scholar context from `legend_scholars`
- Normalizes scholar display name from noisy scrape titles when needed
- Runs topic modeling (NMF + concept-pair fallback) on scholar corpus
- Selects one daily topic using seeded weighted random choice
- Ranks evidence chunks with hybrid retrieval (dense similarity + lexical + section prior)
- Optionally ingests global trending issues and aligns one issue to the selected topic
- Builds a commentary-style prompt (Medium/Substack structure + scholar style profile)
- Enforces strict reliability gates:
  - verified trend URL in output metadata
  - byline anti-impersonation (`By Scholar Name` blocked for deceased scholars)
  - mandatory `claim_evidence_map`
  - paragraph-to-citation grounding checks
- Writes one story doc into `legend_scholar_daily_stories`
- Writes a run log into `daily_story_jobs`
- Marks story as `pending_review`

### 3. Real generation test (LLM)

```bash
python daily_story_worker.py --scholar-id 7e3a28d4-ff81-4085-aff1-d16426bce921 --topic "Civic responsibility in a polarized democracy"
```

### 4. Daily batch run

```bash
python daily_story_worker.py --max-scholars 25
```

Disable trends for offline runs:

```bash
python daily_story_worker.py --max-scholars 25 --disable-trends
```

### 5. Schedule it (Windows Task Scheduler)

Use this as the action command:

```powershell
powershell -NoProfile -Command "cd D:\NGO-Automation; python daily_story_worker.py --max-scholars 25"
```

Recommended schedule:

- Trigger: Daily
- Time: low-traffic hour (for example, 01:30 AM local time)
- Retry: every 30 minutes, up to 2 attempts

### 5.1 Suggested cron suite (production)

- `00:45` `profile_quality_refresh`:
  - Validate `legend_scholars` docs, normalize names, and flag low-quality profiles.
- `01:30` `daily_story_worker`:
  - Generate one story per scholar for that date.
- `02:15` `daily_story_retry_failed`:
  - Retry only `failed_generation` / `failed_validation` with stricter context filtering.
- `08:00` `daily_story_eval`:
  - Compute quality metrics (citation coverage, topic relevance, validation error rates).

Job commands:

```bash
python profile_quality_refresh.py --max-scholars 1000 --update-scholar-docs
python trend_ingestion_worker.py --provider auto --max-items 60
python daily_story_worker.py --max-scholars 25 --enforce-profile-quality
python daily_story_retry_failed.py --date 2026-03-04
python daily_story_eval.py --date 2026-03-04 --window-days 1
```

Single-command suite run:

```bash
python daily_story_suite.py --date 2026-03-04 --max-scholars 25 --trend-provider auto --enforce-profile-quality
```

### 6. Collections and statuses

- Input: `legend_scholars`
- Outputs:
  - `legend_scholar_daily_stories`
  - `daily_story_jobs`
  - `daily_story_trend_issues`
  - `daily_story_profile_quality`
  - `daily_story_quality_events`

Story status values:

- `pending_review`: passed safety checks, waiting human approval
- `generated`: passed checks and auto-publish mode was enabled
- `failed_validation`: generated but blocked by safety checks

### 7. Safety defaults

Current defaults are conservative:

- Posthumous anti-impersonation policy is ON
- Human review requirement is ON
- Every story includes explicit disclosure text
- Citation chunk IDs are required and validated
