"""
Full Automation Pipeline

End-to-end script that:
1) Reads an Excel file (or a URL list) with profile URLs
2) Scrapes each profile (page, CVs, linked personal sites)
3) Cleans text and creates section-aware chunks using the existing OpenAI chunking pipeline
4) Uploads chunks to Pinecone as embeddings
5) Syncs profiles from Pinecone into MongoDB with LLM-generated summaries

Usage:
    python full_automation_pipeline.py path/to/profiles.xlsx
    python full_automation_pipeline.py --urls-file path/to/urls.txt

You must have:
    - OPENAI_API_KEY set (for chunking + summaries + embeddings)
    - PINECONE_API_KEY set (or configured in config/pinecone_config.py)
    - MONGODB_URI set (Mongo connection string)
"""

import asyncio
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple, List, Optional

from dotenv import load_dotenv

# Load environment variables early
load_dotenv()

# Local imports â€“ reuse existing pipeline pieces
from unified_pipeline import UnifiedPipeline, load_urls_file
from upload_chunks_to_pinecone import load_all_chunks, upload_chunks_to_pinecone
from sync_profiles_to_mongodb import MongoDBScholarSync
from backfill_chunk_names import backfill_chunk_names


def append_error_log(log_path: Path, message: str) -> None:
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().isoformat()
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {message}\n")
    except Exception:
        pass


def _configure_utf8_stdio() -> None:
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("PYTHONUTF8", "1")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


async def run_full_pipeline(
    excel_path: Optional[str] = None,
    urls: Optional[List[str]] = None,
    profile_name: Optional[str] = None,
    profile_id: Optional[str] = None,
    profile_url: Optional[str] = None,
    output_dir: str = "output",
    chunking_output_dir: str = "output/chunked_profiles",
    use_llm_chunking: bool = True,
    llm_provider: str = "openai",
    llm_model: str = "gpt-4o-mini",
    limit: int | None = None,
    start_from: int = 0,
    pinecone_batch_size: int = 50,
    skip_pinecone: bool = False,
    skip_mongodb: bool = False,
    skip_indexes: bool = False,
    incremental_sync: bool = True,
    incremental_sync_batch_size: int = 100,
    incremental_sync_pinecone_batch_size: int = 50,
) -> Dict[str, Any]:
    """
    Run the entire pipeline: Excel/URL list -> scrape -> clean + chunk -> Pinecone -> MongoDB.
    """
    _configure_utf8_stdio()
    print("=" * 80)
    print("FULL AUTOMATION PIPELINE")
    print("Excel/URL List -> Scraping -> Cleaning -> OpenAI Chunking -> Pinecone -> MongoDB")
    print("=" * 80)
    error_log_path = Path(output_dir) / "errors.log"

    # ------------------------------------------------------------------
    # Stage 1: Unified scraping / cleaning / chunking to JSON + chunks
    # ------------------------------------------------------------------
    pipeline = UnifiedPipeline(
        output_dir=output_dir,
        chunking_output_dir=chunking_output_dir,
        use_llm_chunking=use_llm_chunking,
        llm_provider=llm_provider,
        llm_model=llm_model,
        incremental_sync_enabled=incremental_sync,
        incremental_sync_batch_size=incremental_sync_batch_size,
        incremental_pinecone_batch_size=incremental_sync_pinecone_batch_size,
        incremental_skip_pinecone=skip_pinecone,
        incremental_skip_mongo=skip_mongodb,
        incremental_skip_indexes=skip_indexes,
    )

    if urls:
        summary = await pipeline.run_from_urls(
            urls=urls,
            profile_name=profile_name,
            profile_id=profile_id,
            profile_url=profile_url,
        )
    else:
        if not excel_path:
            raise ValueError("excel_path is required when urls are not provided")
        summary = await pipeline.run(
            excel_path=excel_path,
            limit=limit,
            start_from=start_from,
        )

    if summary.get("successful", 0) == 0:
        print("\n[FullPipeline] No successful profiles from unified pipeline; skipping Pinecone & MongoDB steps.")
        return summary

    if incremental_sync:
        print("\n[FullPipeline] Incremental sync is enabled in UnifiedPipeline.")
        print("[FullPipeline] Pinecone/Mongo stages were flushed batchwise during profile processing.")
        return summary

    chunk_dir = Path(summary["chunking_output_dir"])
    if not use_llm_chunking:
        print("\n[FullPipeline] LLM chunking disabled; no chunks.json will be created.")
        print("[FullPipeline] Skipping Pinecone and MongoDB stages.")
        return summary

    if not chunk_dir.exists():
        print(f"\n[FullPipeline] Chunking output directory does not exist: {chunk_dir}")
        print("[FullPipeline] Skipping Pinecone and MongoDB stages.")
        return summary

    # Ensure each chunk has professor_name populated based on profile JSON
    print("\n[FullPipeline] Backfilling professor_name into chunks where missing...")
    backfill_chunk_names(
        profiles_root=Path(summary["output_dir"]) / "profiles",
        chunks_root=Path(summary["chunking_output_dir"]),
    )

    # ------------------------------------------------------------------
    # Stage 2: Load chunks and upload to Pinecone
    # ------------------------------------------------------------------
    if skip_pinecone:
        print("\n[FullPipeline] Skipping Pinecone upload (skip_pinecone=True).")
    else:
        try:
            print("\n" + "=" * 80)
            print("[FullPipeline] Stage 2: Uploading chunks to Pinecone")
            print("=" * 80)
            print(f"[FullPipeline] Chunk directory: {chunk_dir}")

            chunks = load_all_chunks(str(chunk_dir))
            if not chunks:
                print("[FullPipeline] No chunks loaded; skipping Pinecone and MongoDB stages.")
                return summary

            # Upload embeddings to Pinecone
            upload_chunks_to_pinecone(chunks, batch_size=pinecone_batch_size)
        except Exception as e:
            append_error_log(error_log_path, f"[pinecone] {e}\n{traceback.format_exc()}")
            raise

    # ------------------------------------------------------------------
    # Stage 3: Sync profiles from Pinecone into MongoDB
    # ------------------------------------------------------------------
    if skip_mongodb:
        print("\n[FullPipeline] Skipping MongoDB sync (skip_mongodb=True).")
        return summary

    print("\n" + "=" * 80)
    print("[FullPipeline] Stage 3: Syncing profiles to MongoDB")
    print("=" * 80)

    # Build a unique map of professor_id -> professor_name from the chunks we just uploaded
    # This lets us sync only the profiles from this run instead of all profiles in Pinecone.
    profiles: Dict[str, str] = {}
    # If we skipped Pinecone, we may not have `chunks` in scope; reload if needed.
    if "chunks" not in locals():
        chunks = load_all_chunks(str(chunk_dir))

    for chunk in chunks:
        prof_id = chunk.get("professor_id") or chunk.get("profile_id")
        prof_name = chunk.get("professor_name") or "Unknown"
        if prof_id and prof_id not in profiles:
            profiles[prof_id] = prof_name

    if not profiles:
        print("[FullPipeline] No professor IDs found in chunks; skipping MongoDB sync.")
        return summary

    try:
        sync = MongoDBScholarSync()
        if not skip_indexes:
            sync.create_indexes()
    except Exception as e:
        append_error_log(error_log_path, f"[mongodb_init] {e}\n{traceback.format_exc()}")
        raise

    total_profiles = len(profiles)
    successful = 0
    failed = 0

    print(f"[FullPipeline] Syncing {total_profiles} profiles from Pinecone to MongoDB...")
    for idx, (prof_id, prof_name) in enumerate(profiles.items(), start=1):
        print(f"\n[FullPipeline][MongoDB] {idx}/{total_profiles}: {prof_name} ({prof_id})")
        try:
            if sync.sync_profile(prof_id, prof_name):
                successful += 1
                print("[FullPipeline][MongoDB] âœ“ Synced successfully")
            else:
                failed += 1
                print("[FullPipeline][MongoDB] âœ— Failed to sync")
        except Exception as e:
            failed += 1
            append_error_log(error_log_path, f"[mongodb_sync] {prof_id} {prof_name} :: {e}\n{traceback.format_exc()}")
            print("[FullPipeline][MongoDB] âœ— Failed to sync (exception)")

        # Small delay to avoid potential rate limiting
        time.sleep(0.5)

    print("\n" + "=" * 80)
    print("[FullPipeline] MongoDB Sync Summary")
    print(f"[FullPipeline] Successful: {successful}")
    print(f"[FullPipeline] Failed: {failed}")
    print(f"[FullPipeline] Total attempted: {total_profiles}")
    print("=" * 80)
    return summary


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the full Excel -> Scrape -> Clean+Chunk (OpenAI) -> Pinecone -> MongoDB pipeline"
    )
    parser.add_argument(
        "excel_path",
        type=str,
        nargs="?",
        default="profile.xlsx",
        help="Path to Excel file with profile URLs (ignored if --urls-file is set)",
    )
    parser.add_argument(
        "--urls-file",
        type=str,
        default=None,
        help="Path to a .txt file with one URL per line (treat as a single profile)",
    )
    parser.add_argument(
        "--profile-name",
        type=str,
        default=None,
        help="Optional profile name to use in URLs-file mode",
    )
    parser.add_argument(
        "--profile-url",
        type=str,
        default=None,
        help="Primary profile URL to store in output JSON (defaults to first URL in file)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Base output directory for profiles JSON (default: output, or output/url_list_runs/<timestamp> in URLs-file mode)",
    )
    parser.add_argument(
        "--chunking-output-dir",
        type=str,
        default=None,
        help="Output directory for chunked profiles (default: <output_dir>/chunked_profiles)",
    )
    parser.add_argument(
        "--no-llm-chunking",
        action="store_true",
        help="Disable LLM-based section-aware chunking (will also skip Pinecone + MongoDB stages)",
    )
    parser.add_argument(
        "--llm-provider",
        type=str,
        default="openai",
        choices=["ollama", "openai"],
        help="LLM provider for chunking (default: openai)",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="gpt-4o-mini",
        help="LLM model name (forced to gpt-4o-mini)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of profiles to process from Excel",
    )
    parser.add_argument(
        "--start-from",
        type=int,
        default=0,
        help="Start processing from this index (for resuming large runs)",
    )
    parser.add_argument(
        "--pinecone-batch-size",
        type=int,
        default=50,
        help="Batch size for uploading vectors to Pinecone (default: 50)",
    )
    parser.add_argument(
        "--skip-pinecone",
        action="store_true",
        help="Skip Pinecone upload stage (useful if already uploaded)",
    )
    parser.add_argument(
        "--skip-mongodb",
        action="store_true",
        help="Skip MongoDB sync stage",
    )
    parser.add_argument(
        "--skip-indexes",
        action="store_true",
        help="Skip creating MongoDB indexes (faster for repeated runs)",
    )
    parser.add_argument(
        "--no-incremental-sync",
        action="store_true",
        help="Disable incremental batchwise Pinecone/Mongo sync in UnifiedPipeline and use end-of-run sync stages",
    )
    parser.add_argument(
        "--sync-batch-size",
        type=int,
        default=100,
        help="Profiles per incremental sync batch in UnifiedPipeline (default: 100)",
    )
    parser.add_argument(
        "--sync-pinecone-batch-size",
        type=int,
        default=50,
        help="Chunk batch size per incremental Pinecone flush (default: 50)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Fix Windows event loop if needed (same pattern as main.py / unified_pipeline)
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # Resolve output directories (match unified_pipeline behavior)
    is_urls_mode = bool(args.urls_file)
    if args.output_dir:
        resolved_output_dir = args.output_dir
    else:
        if is_urls_mode:
            run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            resolved_output_dir = os.path.join("output", "url_list_runs", run_stamp)
        else:
            resolved_output_dir = "output"

    if args.chunking_output_dir:
        resolved_chunking_output_dir = args.chunking_output_dir
    else:
        resolved_chunking_output_dir = os.path.join(resolved_output_dir, "chunked_profiles")

    urls = load_urls_file(args.urls_file) if args.urls_file else None

    error_log_path = Path(resolved_output_dir) / "errors.log"
    try:
        asyncio.run(
            run_full_pipeline(
                excel_path=None if is_urls_mode else args.excel_path,
                urls=urls,
                profile_name=args.profile_name,
                profile_url=args.profile_url,
                output_dir=resolved_output_dir,
                chunking_output_dir=resolved_chunking_output_dir,
                use_llm_chunking=not args.no_llm_chunking,
                llm_provider=args.llm_provider,
            llm_model="gpt-4o-mini",
                limit=args.limit,
                start_from=args.start_from,
                pinecone_batch_size=args.pinecone_batch_size,
                skip_pinecone=args.skip_pinecone,
                skip_mongodb=args.skip_mongodb,
                skip_indexes=args.skip_indexes,
                incremental_sync=not args.no_incremental_sync,
                incremental_sync_batch_size=args.sync_batch_size,
                incremental_sync_pinecone_batch_size=args.sync_pinecone_batch_size,
            )
        )
    except Exception as e:
        append_error_log(error_log_path, f"[fatal] {e}\n{traceback.format_exc()}")
        raise


if __name__ == "__main__":
    main()

