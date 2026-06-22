## Daily Story Feature Architecture (ML + Vertex AI)

### Goal
Generate one daily scholar-inspired article per profile from `legend_scholars`, grounded in scraped evidence, with strong anti-impersonation safety and human review.

### Scope and policy
- Input data source: `legend_scholars`
- Generation model: Vertex AI (Gemini), not OpenAI
- Output is synthetic perspective, never a real first-person post from deceased scholars
- Every story must cite source chunks and include disclosure

### End-to-end pipeline
1. Scholar quality and normalization (ML pre-step)
- Read each profile from `legend_scholars`
- Normalize scholar name from noisy page titles using pattern extraction from biography chunks
- Compute profile quality signals:
  - context chunk count
  - biography/legacy section presence
  - text cleanliness ratio
- Mark low-quality profiles as `needs_data_repair` and skip generation for that run

2. Topic mining (ML)
- Build scholar-local corpus from `rag_context.section_chunks` and `section_text`
- Primary: NMF topic modeling on TF-IDF sentence matrix
- Fallbacks: concept co-occurrence pairs + phrase mining
- Optional embedding rerank using sentence-transformers (local/offline-first)
- Filter topics with concept-aware quality rules to remove institutional/noisy phrases
- Daily topic selection uses seeded weighted random (random per day, deterministic replay)

3. World issue ingestion (ML + data integration)
- Fetch current global issues from configurable providers:
  - RSS (no key): BBC, NYT World, Al Jazeera (default)
  - NewsAPI (keyed): `NEWSAPI_KEY`
  - GDELT (no key)
- Normalize title/summary/source/publish time
- Match scholar topic to one issue using hybrid similarity:
  - lexical overlap
  - optional embedding similarity
  - recency weighting

4. Context retrieval (ML)
- Candidate chunks from `section_chunks`
- Hybrid ranking:
  - dense similarity(topic, chunk)
  - lexical overlap(topic tokens, chunk terms)
  - section prior (biography/background preferred)
- Cross-encoder rerank on top-K candidates (when model is available locally)
- Keep top N chunks with dedupe and citation IDs

5. Draft generation (AI)
- Use Vertex AI with strict JSON schema output (response schema when SDK supports it)
- Prompt requires:
  - commentary structure for Medium/Substack readability
  - style profile inferred from scholar corpus (cadence, connectors, concept anchors)
  - current issue hook + historical bridge + argument progression
  - historically grounded analytical voice
  - no living-person impersonation
  - no fabricated facts
  - inline citations `[chunk:ID]`

6. Safety and validation gate
- Validate title/article length
- Require >=4 unique inline chunk citations (or validated fallback to used chunk IDs)
- Reject unknown chunk IDs
- Persona checks for first-person posthumous claims
- Block byline impersonation (e.g., `By John Hope Franklin`)
- Require verified trend source URL when live issue mode is enabled
- Require `claim_evidence_map` JSON linking claims to chunk IDs
- Validate paragraph-level grounding between cited paragraphs and cited evidence chunks
- Add disclosure banner automatically
- Store as `pending_review` by default

7. Review and publish
- Editor UI reviews story text + citations + validation notes
- Approved stories move to published state
- Rejected stories feed back into data-quality queue

### Collections
- Input: `legend_scholars`
- Output stories: `legend_scholar_daily_stories`
- Job runs: `daily_story_jobs`
- Trend cache: `daily_story_trend_issues`
- Profile readiness: `daily_story_profile_quality`
- Quality events: `daily_story_quality_events`
- Operational job logs:
  - `daily_story_trend_jobs`
  - `daily_story_profile_quality_jobs`
  - `daily_story_retry_jobs`
  - `daily_story_eval_jobs`

### Recommended cron suite
1. `profile_quality_refresh` (00:45)
- Normalize names and profile-quality flags

2. `daily_story_generate` (01:30)
- Run `daily_story_worker.py --max-scholars N`

3. `daily_story_retry_failed` (02:15)
- Retry only failed stories with stricter context filters

4. `daily_story_eval` (08:00)
- Compute and store quality KPIs

Optional combined job:
- `daily_story_suite.py` can run all four phases in one command.

### Production reliability rules
- Idempotency key: `story_key = profile_id:story_date`
- Unique index on `story_key`
- Retry with exponential backoff for Vertex calls
- Record `topic_selection` and `ml_retrieval` metadata in each story doc
- Require human review in initial rollout

### Accuracy metrics to monitor
- Citation validity rate
- Validation pass rate
- Topic relevance score (LLM judge + human feedback)
- Editor rejection rate
- Percentage of profiles skipped due to low data quality

### Rollout plan (recommended)
1. Week 1: dry-run only (`--dry-run --no-llm`) and measure topic/retrieval quality
2. Week 2: LLM generation on 5-10 scholars/day with mandatory review
3. Week 3: full scholar set with retry/eval jobs enabled

### Repository strategy
- Start in this repository (faster integration with existing scraping/chunking data)
- Expose worker commands as deployable jobs (Task Scheduler / Cloud Run jobs)
- Split to a separate repo only when team ownership, deployment lifecycle, or scale clearly diverges
