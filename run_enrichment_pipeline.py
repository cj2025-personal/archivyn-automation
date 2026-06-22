"""
Run the enrichment pipeline on existing OSU professor profiles.

For EACH professor the pipeline does (in order):
  1. Collect data from up to 13 public sources (Semantic Scholar, OpenAlex, etc.)
  2. Save enrichment.json + enrichment_text.txt
  3. Merge original profile text + enrichment text → re-chunk via LLM
  4. Upload new chunks to Pinecone as embeddings
  5. Sync the professor to MongoDB with LLM-generated summaries

Steps 3-5 happen IMMEDIATELY per professor so progress is never lost.
If the pipeline crashes at professor 2000, professors 1-1999 are fully
persisted in Pinecone + MongoDB already.

Usage:
    # Install one extra dependency
    pip install scholarly

    # List all 13 data sources
    python run_enrichment_pipeline.py --list-sources

    # Dry-run — see what will happen
    python run_enrichment_pipeline.py --dry-run --limit 10

    # Test a single professor (full pipeline: enrich → chunk → Pinecone → Mongo)
    python run_enrichment_pipeline.py --name "Amber Bruney"

    # Run a batch
    python run_enrichment_pipeline.py --start 0 --limit 100

    # Run all 4077 (auto-skips already-enriched)
    python run_enrichment_pipeline.py

    # Skip Pinecone/MongoDB (collect enrichment data only)
    python run_enrichment_pipeline.py --skip-pinecone --skip-mongodb

    # Skip only MongoDB
    python run_enrichment_pipeline.py --skip-mongodb

    # Disable slow sources
    python run_enrichment_pipeline.py --disable google_scholar

    # Force re-enrich already-done profiles
    python run_enrichment_pipeline.py --force --limit 50
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from enrichment.orchestrator import (
    EnrichmentOrchestrator,
    load_professor_queries_from_profiles,
    ALL_COLLECTORS,
    API_KEY_REQUIREMENTS,
)
from enrichment.base_collector import ProfessorQuery


_mongo_client = None
_mongo_collection = None


def _get_enrichment_collection():
    """Lazy-init MongoDB connection for raw enrichment storage."""
    global _mongo_client, _mongo_collection
    if _mongo_collection is not None:
        return _mongo_collection

    from config.mongodb_utils import create_mongo_client, resolve_mongo_db_name

    mongodb_uri = os.getenv("MONGODB_URI")
    if not mongodb_uri:
        raise ValueError("MONGODB_URI not found in environment variables")

    _mongo_client = create_mongo_client(mongodb_uri)
    db_name = resolve_mongo_db_name(mongodb_uri)
    db = _mongo_client[db_name]
    _mongo_collection = db.enrichment_raw
    # Create index on profile_id for fast lookups/upserts
    _mongo_collection.create_index("profile_id", unique=True)
    return _mongo_collection


def _store_raw_enrichment_to_mongo(query, enrichment_path: Path, text_path: Path):
    """
    Store raw enrichment data directly in MongoDB.
    No LLM processing — just the structured source data + raw text.
    """
    collection = _get_enrichment_collection()

    enrichment_data = json.loads(enrichment_path.read_text(encoding="utf-8"))
    raw_text = text_path.read_text(encoding="utf-8") if text_path else ""

    doc = {
        "profile_id": query.profile_id,
        "name": query.name,
        "university": query.university,
        "department": query.department,
        "profile_url": query.profile_url,
        "enriched_at": enrichment_data.get("enriched_at"),
        "confidence": enrichment_data.get("confidence", {}),
        "summary": enrichment_data.get("summary", {}),
        "sources": enrichment_data.get("sources", {}),
        "raw_text": raw_text,
        "raw_text_length": len(raw_text),
    }

    collection.update_one(
        {"profile_id": query.profile_id},
        {"$set": doc},
        upsert=True,
    )


def _configure_logger() -> logging.Logger:
    level_name = os.getenv("ENRICHMENT_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("enrichment_pipeline")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enrich OSU professor profiles with data from 13 public sources."
    )
    parser.add_argument(
        "--profiles-dir",
        type=Path,
        default=Path("output/osu_faculty_run/profiles"),
        help="Path to existing profiles directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/osu_faculty_run"),
        help="Base output directory",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="0-based index to start from in the profiles list",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of profiles to enrich",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Enrich a single professor by name (partial match)",
    )
    parser.add_argument(
        "--profile-id",
        type=str,
        default=None,
        help="Enrich a single professor by profile_id",
    )
    parser.add_argument(
        "--sources",
        type=str,
        default=None,
        help=(
            "Comma-separated list of sources to enable. Available: "
            + ", ".join(sorted(ALL_COLLECTORS.keys()))
        ),
    )
    parser.add_argument(
        "--disable",
        type=str,
        default=None,
        help="Comma-separated list of sources to disable",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-enrichment even if enrichment.json already exists",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=4,
        help="Max concurrent API requests per professor (default: 4)",
    )
    parser.add_argument(
        "--skip-pinecone",
        action="store_true",
        help="Skip Pinecone upload (collect + chunk only)",
    )
    parser.add_argument(
        "--skip-mongodb",
        action="store_true",
        help="Skip MongoDB sync",
    )
    parser.add_argument(
        "--skip-chunking",
        action="store_true",
        help="Skip chunking + Pinecone + MongoDB (collect enrichment data only)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Skip chunking/upload for profiles below this confidence score (0.0-1.0)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually running collectors",
    )
    parser.add_argument(
        "--list-sources",
        action="store_true",
        help="List all available data sources and exit",
    )
    parser.add_argument(
        "--re-enrich-failed",
        action="store_true",
        help="Only re-process profiles where >50%% of collectors failed",
    )
    parser.add_argument(
        "--re-enrich-sources",
        type=str,
        default=None,
        help="Comma-separated list of sources to re-run for already-enriched profiles (merges results)",
    )
    parser.add_argument(
        "--clear-cache-for",
        type=str,
        default=None,
        help="Comma-separated list of sources whose cached results should be deleted before running",
    )
    parser.add_argument(
        "--enable-cleaning",
        action="store_true",
        help="Enable GPT-4o-mini cleaning of enrichment text before chunking",
    )
    parser.add_argument(
        "--skip-cleaning",
        action="store_true",
        help="Skip GPT-4o-mini cleaning even if --enable-cleaning is set",
    )
    return parser.parse_args()


async def enrich_and_sync_single_professor(
    orchestrator: EnrichmentOrchestrator,
    query: ProfessorQuery,
    output_dir: Path,
    logger: logging.Logger,
    skip_chunking: bool = False,
    skip_pinecone: bool = False,
    skip_mongodb: bool = False,
    min_confidence: float = 0.0,
    enable_cleaning: bool = False,
) -> dict:
    """
    Full pipeline for one professor:
    1. Collect enrichment from all sources
    2. Save enrichment.json + enrichment_text.txt
    3. Chunk merged text
    4. Upload to Pinecone
    5. Sync to MongoDB
    """
    summary = {
        "name": query.name,
        "profile_id": query.profile_id,
        "successful_sources": [],
        "failed_sources": [],
        "text_saved": False,
        "chunked": False,
        "chunks_count": 0,
        "pinecone_uploaded": 0,
        "mongodb_synced": False,
        "confidence": 0.0,
    }

    # ── Step 1: Collect enrichment data from all sources ──
    print(f"  [Step 1/5] Collecting enrichment data from public sources...")
    results = await orchestrator.enrich_professor(query)

    # ── Step 2: Save enrichment files ──
    print(f"  [Step 2/5] Saving enrichment files...")
    enrichment_path = orchestrator.save_enrichment(query, results)
    text_path = orchestrator.save_enrichment_text(query, results)

    summary["successful_sources"] = [n for n, r in results.items() if r.success]
    summary["failed_sources"] = [n for n, r in results.items() if not r.success]
    summary["text_saved"] = text_path is not None

    # Read confidence score
    try:
        enr_data = json.loads(enrichment_path.read_text(encoding="utf-8"))
        summary["confidence"] = enr_data.get("confidence", {}).get("overall_confidence", 0.0)
    except Exception:
        pass

    # If no enrichment data was collected or no text saved, stop here
    if not summary["successful_sources"] or not summary["text_saved"]:
        print(f"  [Pipeline] ⚠️ No enrichment data collected, skipping remaining steps")
        return summary

    # ── Step 2.5: Clean enrichment text with GPT-4o-mini ──
    if enable_cleaning and text_path:
        try:
            from enrichment.enrichment_cleaner import EnrichmentCleaner
            raw_size = text_path.stat().st_size
            print(f"  [Step 2.5] Cleaning enrichment text with GPT-4o-mini ({raw_size:,} bytes raw)...")
            t0 = time.perf_counter()
            cleaner = EnrichmentCleaner()
            cleaned_path = cleaner.clean_file(text_path)
            clean_elapsed = time.perf_counter() - t0
            cleaned_size = cleaned_path.stat().st_size
            reduction = (1 - cleaned_size / max(raw_size, 1)) * 100
            print(f"  [Step 2.5] ✅ Cleaned: {raw_size:,} -> {cleaned_size:,} bytes ({reduction:+.0f}%) in {clean_elapsed:.1f}s")
        except Exception as e:
            print(f"  [Step 2.5] ⚠️ Cleaning failed (continuing with raw text): {e}")
    elif enable_cleaning and not text_path:
        print(f"  [Step 2.5] ⏭️ No text to clean")
    else:
        print(f"  [Step 2.5] ⏭️ Cleaning disabled")

    # ── Store raw enrichment to MongoDB (no LLM needed) ──
    if not skip_mongodb:
        try:
            print(f"  [Step 3/5] Storing raw enrichment data to MongoDB...")
            _store_raw_enrichment_to_mongo(query, enrichment_path, text_path)
            summary["mongodb_synced"] = True
            print(f"  [Step 3/5] ✅ Raw enrichment stored in MongoDB")
        except Exception as e:
            print(f"  [Step 3/5] ❌ MongoDB store failed: {e}")
            summary["mongodb_synced"] = False

    # If only collecting data (no chunking/upload), stop here
    if skip_chunking:
        print(f"  [Pipeline] ⏭️ Skipping chunk/upload (--skip-chunking)")
        return summary

    # Skip below confidence threshold
    if min_confidence > 0 and summary["confidence"] < min_confidence:
        print(f"  [Pipeline] ⏭️ Skipping chunk/upload: confidence {summary['confidence']:.2f} < threshold {min_confidence:.2f}")
        return summary

    # ── Steps 4-5: Chunk → Pinecone (incremental) ──
    from enrichment.incremental_sync import incremental_sync_professor

    profile_dir = output_dir / "profiles" / query.profile_id
    chunking_output_dir = output_dir / "chunked_profiles"

    sync_result = incremental_sync_professor(
        profile_dir=profile_dir,
        chunking_output_dir=chunking_output_dir,
        professor_name=query.name,
        skip_pinecone=skip_pinecone,
        skip_mongodb=True,  # Already stored raw above
    )

    summary["chunked"] = sync_result.get("chunked", False)
    summary["chunks_count"] = sync_result.get("chunks_count", 0)
    summary["pinecone_uploaded"] = sync_result.get("pinecone_uploaded", 0)

    return summary


def _print_sources():
    """Print all available data sources with descriptions."""
    source_descriptions = {
        # -- Original 13 --
        "semantic_scholar": "Publications, citations, abstracts, h-index, co-authors (API, free)",
        "openalex": "Publications, concepts, institutions, citation networks (API, free)",
        "google_scholar": "h-index, citation metrics, top papers (scraping, free, slow)",
        "crossref": "Full publication metadata, abstracts, funding info, DOIs (API, free)",
        "orcid": "Career history, education, employment, grants, peer reviews (API, free)",
        "nsf_grants": "NSF federal grants, amounts, abstracts (API, free)",
        "nih_grants": "NIH-funded grants, project details, funding (API, free)",
        "rate_my_professor": "Student ratings, difficulty, reviews, course tags (GraphQL, free)",
        "osu_courses": "Courses taught at OSU, descriptions, schedules (API, free)",
        "osu_news": "OSU press releases and news articles (scraping, free)",
        "google_news": "News articles and media mentions (RSS, free)",
        "youtube_lectures": "Lectures, talks, interviews on YouTube (API, needs YOUTUBE_API_KEY)",
        "osu_expertise": "OSU Experts directory + Knowledge Bank repository (scraping, free)",
        "web_search": "DDG search + recursive scraping of faculty / lab / CV pages (free)",
        # -- New differentiator sources --
        "unpaywall": "Legal OA copies of paywalled DOIs (dependent on openalex/crossref)",
        "arxiv": "Preprint full-text metadata by author name (API, free)",
        "biorxiv": "bioRxiv/medRxiv life-sci preprints by author (API, free)",
        "pmc_oa": "PubMed Central OA full-text excerpts (NCBI E-utils, NCBI_API_KEY raises limits)",
        "core_api": "CORE OA aggregator full-text search (needs CORE_API_KEY, free tier 1k/day)",
        "github": "Repos, READMEs, topics, stars (GITHUB_TOKEN raises limits)",
        "huggingface": "HF models & datasets by author (HF_TOKEN raises limits)",
        "paperswithcode": "Code implementations linked to papers (API, free)",
        "zenodo": "Datasets / software / posters (ZENODO_ACCESS_TOKEN raises limits)",
        "figshare": "Datasets / figures / posters (API, free)",
        "osf": "OSF projects / pre-registrations (API, free)",
        "wikidata": "Structured researcher facts via SPARQL (awards, positions, lineage, free)",
        "wikipedia": "Biographical Wikipedia extracts for notable profs (MediaWiki API, free)",
        "altmetric": "Per-DOI impact: policy/patents/news/Wikipedia (public endpoint, key raises limits)",
        "opencitations": "Supplementary citation graph per DOI (API, free)",
        "usaspending": "Federal grants/contracts (DOE/DOD/USDA/EPA beyond NIH/NSF, API, free)",
        "patentsview": "USPTO patents by inventor (PATENTSVIEW_API_KEY raises limits)",
        "clinicaltrials": "ClinicalTrials.gov studies with this researcher as PI (API, free)",
        "gdelt": "Global news index — mentions beyond Google News (API, free)",
        "youtube_transcripts": "Full captions for lecture videos — depends on youtube_lectures",
    }

    # Dynamic: rely on ALL_COLLECTORS as the source of truth
    all_names = set(ALL_COLLECTORS.keys())
    # Warn about any collectors missing a description entry
    undocumented = all_names - set(source_descriptions.keys())
    for name in undocumented:
        source_descriptions[name] = "(no description)"

    print(f"\n  Available data sources ({len(all_names)} total):\n")
    print(f"  {'Source':<22} {'Needs Key?':<12} Description")
    print(f"  {'─'*22} {'─'*12} {'─'*65}")
    for name in sorted(all_names):
        key_name = API_KEY_REQUIREMENTS.get(name)
        needs_key = f"Yes ({key_name})" if key_name else "No"
        desc = source_descriptions.get(name, "")
        print(f"  {name:<22} {needs_key:<12} {desc}")
    print()


async def _main(logger: logging.Logger, args: argparse.Namespace) -> None:
    if args.list_sources:
        _print_sources()
        return

    profiles_dir = args.profiles_dir
    output_dir = args.output_dir

    if not profiles_dir.exists():
        raise FileNotFoundError(f"Profiles directory not found: {profiles_dir}")

    # Parse source filters
    enabled = args.sources.split(",") if args.sources else None
    disabled = args.disable.split(",") if args.disable else None

    # If --re-enrich-sources is set, override enabled sources
    if args.re_enrich_sources:
        enabled = [s.strip() for s in args.re_enrich_sources.split(",")]
        args.force = True  # Must force to re-process existing profiles
        print(f"[Setup] Re-enrichment mode: only running sources: {', '.join(enabled)}")

    # Clear cached failure results for specified sources
    if args.clear_cache_for:
        sources_to_clear = [s.strip() for s in args.clear_cache_for.split(",")]
        cache_dir = output_dir / "enrichment_cache"
        if cache_dir.exists():
            cleared = 0
            for cache_file in cache_dir.iterdir():
                fname = cache_file.name
                if any(fname.startswith(src + "_") for src in sources_to_clear):
                    cache_file.unlink()
                    cleared += 1
            print(f"[Setup] Cleared {cleared} cached results for: {', '.join(sources_to_clear)}")
        else:
            print(f"[Setup] No cache directory found at {cache_dir}")

    # Load professor queries from existing profiles
    print(f"\n[Setup] Loading professor profiles from: {profiles_dir}")
    if args.profile_id:
        profile_json = profiles_dir / args.profile_id / f"{args.profile_id}.json"
        if not profile_json.exists():
            raise FileNotFoundError(f"Profile not found: {profile_json}")
        data = json.loads(profile_json.read_text(encoding="utf-8"))
        queries = [ProfessorQuery(
            profile_id=args.profile_id,
            name=data.get("name", ""),
            university="Ohio State University",
            profile_url=data.get("profile_url", ""),
        )]
        print(f"[Setup] ✅ Single profile: {data.get('name', '')} ({args.profile_id})")
    elif args.name:
        all_queries = load_professor_queries_from_profiles(str(profiles_dir))
        name_lower = args.name.lower()
        queries = [q for q in all_queries if name_lower in q.name.lower()]
        if not queries:
            raise ValueError(f"No professor found matching name: {args.name}")
        print(f"[Setup] ✅ Found {len(queries)} professors matching '{args.name}'")
    else:
        queries = load_professor_queries_from_profiles(
            str(profiles_dir),
            limit=args.limit,
            start_from=args.start,
            filter_no_enrichment=not args.force and not args.re_enrich_failed,
        )
        print(f"[Setup] ✅ Loaded {len(queries)} profiles (start={args.start}, limit={args.limit or 'all'}, force={args.force})")

    # --re-enrich-failed: filter to only profiles with >50% source failures
    if args.re_enrich_failed:
        filtered = []
        for q in queries:
            enr_path = profiles_dir / q.profile_id / "enrichment.json"
            if not enr_path.exists():
                continue  # not enriched yet, skip
            try:
                enr_data = json.loads(enr_path.read_text(encoding="utf-8"))
                summary = enr_data.get("summary", {})
                total_queried = summary.get("total_sources_queried", 0)
                failed = summary.get("failed_sources", 0)
                if total_queried > 0 and failed / total_queried > 0.5:
                    filtered.append(q)
            except Exception:
                continue
        queries = filtered
        print(f"[Setup] --re-enrich-failed: filtered to {len(queries)} profiles with >50% failed sources")

    if args.dry_run:
        print(f"\n{'='*80}")
        print(f"DRY RUN — would enrich {len(queries)} professors:")
        print(f"{'='*80}")
        for i, q in enumerate(queries[:30], 1):
            print(f"  {i:4d}. {q.name:<35} id={q.profile_id[:12]}  dept={q.department or '?'}")
        if len(queries) > 30:
            print(f"  ... and {len(queries) - 30} more")

        active_sources = enabled or sorted(ALL_COLLECTORS.keys())
        if disabled:
            active_sources = [s for s in active_sources if s not in disabled]
        print(f"\nActive sources ({len(active_sources)}): {', '.join(active_sources)}")

        mode_parts = ["enrich"]
        if not args.skip_chunking:
            mode_parts.append("chunk")
        if not args.skip_pinecone and not args.skip_chunking:
            mode_parts.append("pinecone")
        if not args.skip_mongodb and not args.skip_chunking:
            mode_parts.append("mongodb")
        print(f"Pipeline mode: {' → '.join(mode_parts)} (per professor, incremental)")
        return

    if not queries:
        print("[Setup] ⚠️ No profiles to enrich. Use --force to re-enrich already-processed profiles.")
        return

    # Determine pipeline mode
    mode_parts = ["enrich"]
    if not args.skip_chunking:
        mode_parts.append("chunk")
    if not args.skip_pinecone and not args.skip_chunking:
        mode_parts.append("pinecone")
    if not args.skip_mongodb and not args.skip_chunking:
        mode_parts.append("mongodb")

    # ── Print banner ──
    print(f"\n{'='*80}")
    print(f"ENRICHMENT PIPELINE")
    print(f"  Profiles → Enrich (13 sources) → Chunk → Pinecone → MongoDB")
    print(f"{'='*80}")
    print(f"[Pipeline] Profiles to process: {len(queries)}")
    print(f"[Pipeline] Output directory: {output_dir}")
    print(f"[Pipeline] Pipeline mode: {' → '.join(mode_parts)} (incremental per professor)")
    print(f"[Pipeline] Max concurrent API requests: {args.max_concurrent}")
    if args.min_confidence > 0:
        print(f"[Pipeline] Min confidence for chunk/upload: {args.min_confidence}")
    print(f"[Pipeline] Env: OPENAI_API_KEY={'✅ set' if os.getenv('OPENAI_API_KEY') else '❌ missing'}, "
          f"PINECONE_API_KEY={'✅ set' if os.getenv('PINECONE_API_KEY') else '❌ missing'}, "
          f"MONGODB_URI={'✅ set' if os.getenv('MONGODB_URI') else '❌ missing'}")

    # Initialize orchestrator
    print(f"\n[Setup] Initializing enrichment orchestrator...")
    orchestrator = EnrichmentOrchestrator(
        output_dir=str(output_dir),
        enabled_sources=enabled,
        disabled_sources=disabled,
        max_concurrent=args.max_concurrent,
    )

    # Process each professor
    start_time = time.perf_counter()
    total = len(queries)
    success_count = 0
    fail_count = 0
    pinecone_total = 0
    mongo_total = 0
    chunks_total = 0
    source_stats: dict = {}

    try:
        for i, query in enumerate(queries, 1):
            prof_start = time.perf_counter()
            print(f"\n{'━'*80}")
            print(f"[{i}/{total}] Processing: {query.name}")
            print(f"  Profile ID: {query.profile_id}")
            print(f"  Profile URL: {query.profile_url}")
            if query.department:
                print(f"  Department: {query.department}")
            print(f"{'━'*80}")

            try:
                summary = await enrich_and_sync_single_professor(
                    orchestrator=orchestrator,
                    query=query,
                    output_dir=output_dir,
                    logger=logger,
                    skip_chunking=args.skip_chunking,
                    skip_pinecone=args.skip_pinecone,
                    skip_mongodb=args.skip_mongodb,
                    min_confidence=args.min_confidence,
                    enable_cleaning=args.enable_cleaning and not args.skip_cleaning,
                )

                n_success = len(summary["successful_sources"])
                n_fail = len(summary["failed_sources"])

                # Track per-source stats
                for s in summary["successful_sources"]:
                    source_stats[s] = source_stats.get(s, 0) + 1
                for s in summary["failed_sources"]:
                    source_stats.setdefault(s, 0)

                # Track sync stats
                chunks_total += summary.get("chunks_count", 0)
                pinecone_total += summary.get("pinecone_uploaded", 0)
                if summary.get("mongodb_synced"):
                    mongo_total += 1

                elapsed = time.perf_counter() - prof_start

                # Build status line
                status_parts = [f"{n_success}/{n_success + n_fail} sources"]
                if summary.get("chunks_count"):
                    status_parts.append(f"{summary['chunks_count']} chunks")
                if summary.get("pinecone_uploaded"):
                    status_parts.append(f"{summary['pinecone_uploaded']} vectors→Pinecone")
                if summary.get("mongodb_synced"):
                    status_parts.append("MongoDB ✅")
                elif not args.skip_mongodb and not args.skip_chunking and summary.get("chunks_count"):
                    status_parts.append("MongoDB ❌")
                if summary.get("confidence"):
                    status_parts.append(f"confidence={summary['confidence']:.2f}")

                # Print per-professor summary
                print(f"\n  {'─'*60}")
                if n_success > 0:
                    print(f"  ✅ [{i}/{total}] DONE: {query.name}")
                    success_count += 1
                else:
                    print(f"  ❌ [{i}/{total}] NO DATA: {query.name}")
                    fail_count += 1
                print(f"     {' | '.join(status_parts)}")
                if summary.get("successful_sources"):
                    print(f"     Sources OK: {', '.join(sorted(summary['successful_sources']))}")
                if summary.get("failed_sources"):
                    print(f"     Sources failed: {', '.join(sorted(summary['failed_sources']))}")
                print(f"     Time: {elapsed:.1f}s | Running total: {success_count} success, {fail_count} fail")

            except Exception as e:
                fail_count += 1
                elapsed = time.perf_counter() - prof_start
                print(f"\n  ❌ [{i}/{total}] EXCEPTION: {query.name} — {e} ({elapsed:.1f}s)")
                import traceback
                traceback.print_exc()

            # Delay between professors to let API rate limits reset
            if i < total:
                await asyncio.sleep(5.0)

    finally:
        await orchestrator.close()

    elapsed_total = time.perf_counter() - start_time

    # ── Final summary ──
    success_rate = (success_count / max(total, 1)) * 100
    print(f"\n{'='*80}")
    print(f"[SUMMARY] Enrichment Pipeline Complete!")
    print(f"{'='*80}")
    print(f"  Total professors processed: {total}")
    print(f"  Successful (≥1 source):     {success_count}")
    print(f"  Failed (0 sources):         {fail_count}")
    print(f"  Success rate:               {success_rate:.1f}%")
    print(f"  Total chunks generated:     {chunks_total:,}")
    print(f"  Total Pinecone vectors:     {pinecone_total:,}")
    print(f"  MongoDB profiles synced:    {mongo_total}")
    print(f"  Total time:                 {elapsed_total:.1f}s ({elapsed_total/max(total,1):.1f}s avg/professor)")
    print(f"{'─'*80}")
    print(f"  Per-source success rates:")
    for source_name in sorted(source_stats.keys()):
        count = source_stats[source_name]
        pct = count / max(total, 1) * 100
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"    {source_name:<22} {count:>4}/{total} ({pct:5.1f}%) {bar}")
    print(f"{'='*80}")


def _update_excel_enrichment_status(profiles_dir: Path, logger: logging.Logger):
    """Update OSU.xlsx with enrichment status columns."""
    import pandas as pd

    excel_path = Path("OSU.xlsx")
    if not excel_path.exists():
        logger.warning("OSU.xlsx not found, skipping Excel update")
        return

    print(f"\n[Excel] Updating enrichment status in {excel_path}...")
    df = pd.read_excel(excel_path)

    # Build profile_id -> enrichment info lookup
    enrichment_info = {}
    for pdir in profiles_dir.iterdir():
        if not pdir.is_dir():
            continue
        enr = pdir / "enrichment.json"
        if not enr.exists():
            continue
        try:
            data = json.loads(enr.read_text(encoding="utf-8"))
            sources_ok = data.get("summary", {}).get("successful_source_names", [])
            confidence = data.get("confidence", {}).get("overall_confidence", 0.0)
            enrichment_info[pdir.name] = {
                "enriched": "Yes" if sources_ok else "No data",
                "sources": ", ".join(sorted(sources_ok)) if sources_ok else "",
                "confidence": round(confidence, 2),
            }
        except Exception:
            enrichment_info[pdir.name] = {
                "enriched": "Error",
                "sources": "",
                "confidence": 0.0,
            }

    # Also build name -> profile_id lookup from profile JSONs
    name_to_pid = {}
    for pdir in profiles_dir.iterdir():
        if not pdir.is_dir():
            continue
        pjson = pdir / f"{pdir.name}.json"
        if pjson.exists():
            try:
                pdata = json.loads(pjson.read_text(encoding="utf-8"))
                name = pdata.get("name", "").strip().lower()
                if name:
                    name_to_pid[name] = pdir.name
            except Exception:
                pass

    # Map rows to enrichment info via Scholar Profile ID or Name
    enriched_col = []
    sources_col = []
    confidence_col = []

    for _, row in df.iterrows():
        pid = str(row.get("Scholar Profile ID", "")).strip()
        name = str(row.get("Name", "")).strip().lower()
        info = None

        if pid and pid != "nan" and pid in enrichment_info:
            info = enrichment_info[pid]
        elif name and name in name_to_pid:
            mapped_pid = name_to_pid[name]
            if mapped_pid in enrichment_info:
                info = enrichment_info[mapped_pid]

        if info:
            enriched_col.append(info["enriched"])
            sources_col.append(info["sources"])
            confidence_col.append(info["confidence"])
        else:
            enriched_col.append("")
            sources_col.append("")
            confidence_col.append("")

    df["Enriched"] = enriched_col
    df["Enrichment Sources"] = sources_col
    df["Enrichment Confidence"] = confidence_col

    df.to_excel(excel_path, index=False)
    enriched_yes = sum(1 for v in enriched_col if v == "Yes")
    enriched_no = sum(1 for v in enriched_col if v == "No data")
    print(f"[Excel] ✅ Updated: {enriched_yes} enriched, {enriched_no} no data, {len(df) - enriched_yes - enriched_no} not processed")


if __name__ == "__main__":
    args = _parse_args()
    logger = _configure_logger()

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(_main(logger, args))

    # Update Excel after pipeline completes
    profiles_dir = args.profiles_dir
    if not args.dry_run and not args.list_sources:
        try:
            _update_excel_enrichment_status(profiles_dir, logger)
        except Exception as e:
            print(f"[Excel] ❌ Failed to update Excel: {e}")
