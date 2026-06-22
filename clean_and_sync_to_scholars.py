"""
Clean existing enrichment data with GPT-4o-mini and push through the full
pipeline: clean -> chunk -> Pinecone -> MongoDB scholars collection.

This handles ALREADY-ENRICHED profiles that have enrichment_text.txt on disk
but haven't been cleaned/chunked/synced yet (or need to be re-synced).

Pipeline per professor:
  1. Clean enrichment_text.txt with GPT-4o-mini -> enrichment_text_cleaned.txt
  2. Merge original profile + cleaned enrichment text
  3. Chunk via ProfileChunkingPipeline (LLM section assignment)
  4. Upload chunks to Pinecone (embeddings + vector upsert)
  5. Fetch chunks from Pinecone -> generate LLM summaries -> upsert to MongoDB scholars

Uses the same incremental_sync_professor() flow as the main enrichment pipeline.

Usage:
  python clean_and_sync_to_scholars.py --limit 5
  python clean_and_sync_to_scholars.py --only-with-data
  python clean_and_sync_to_scholars.py --only-with-data --skip-cleaning
  python clean_and_sync_to_scholars.py --name "Claudia Turro"
  python clean_and_sync_to_scholars.py --dry-run --only-with-data
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Clean enrichment data and sync through Pinecone + MongoDB scholars."
    )
    parser.add_argument(
        "--profiles-dir", type=Path,
        default=Path("output/osu_faculty_run/profiles"),
        help="Path to profiles directory",
    )
    parser.add_argument(
        "--chunks-dir", type=Path,
        default=Path("output/osu_faculty_run/chunked_profiles"),
        help="Path to chunked_profiles output directory",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--name", type=str, default=None, help="Filter by professor name (partial match)")
    parser.add_argument("--profile-id", type=str, default=None, help="Process a single profile by ID")
    parser.add_argument(
        "--only-with-data", action="store_true",
        help="Only process profiles with at least 1 successful enrichment source",
    )
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--skip-cleaning", action="store_true", help="Skip GPT-4o-mini cleaning step")
    parser.add_argument("--skip-pinecone", action="store_true", help="Skip Pinecone upload")
    parser.add_argument("--skip-mongodb", action="store_true", help="Skip MongoDB sync")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Re-process even if chunks.json already exists")
    return parser.parse_args()


def discover_profiles(args) -> List[Dict]:
    """Find enriched profiles to process."""
    profiles_dir = args.profiles_dir
    profiles = []

    print(f"\n[Discovery] Scanning {profiles_dir} ...")

    if args.profile_id:
        pdir = profiles_dir / args.profile_id
        if pdir.exists():
            info = _load_profile_info(pdir)
            if info:
                print(f"[Discovery] Single profile: {info['name']} ({args.profile_id})")
                return [info]
        print(f"[Discovery] Profile not found: {pdir}")
        return []

    total_dirs = 0
    skipped_no_json = 0
    skipped_no_enrichment = 0

    for pdir in sorted(profiles_dir.iterdir()):
        if not pdir.is_dir():
            continue
        total_dirs += 1
        info = _load_profile_info(pdir)
        if not info:
            skipped_no_json += 1
            continue
        if not info["has_enrichment"]:
            skipped_no_enrichment += 1
            continue
        profiles.append(info)

    print(f"[Discovery] Total profile dirs:       {total_dirs}")
    print(f"[Discovery] Skipped (no profile.json): {skipped_no_json}")
    print(f"[Discovery] Skipped (no enrichment):   {skipped_no_enrichment}")
    print(f"[Discovery] With enrichment data:      {len(profiles)}")

    if args.name:
        name_lower = args.name.lower()
        before = len(profiles)
        profiles = [p for p in profiles if name_lower in p["name"].lower()]
        print(f"[Filter] Name match '{args.name}': {before} -> {len(profiles)}")

    if args.only_with_data:
        before = len(profiles)
        profiles = [p for p in profiles if p["successful_sources"] > 0]
        print(f"[Filter] Only with data (1+ source): {before} -> {len(profiles)}")

    if args.min_confidence > 0:
        before = len(profiles)
        profiles = [p for p in profiles if p["confidence"] >= args.min_confidence]
        print(f"[Filter] Min confidence {args.min_confidence}: {before} -> {len(profiles)}")

    # Show source distribution
    source_dist = {}
    for p in profiles:
        s = p["successful_sources"]
        source_dist[s] = source_dist.get(s, 0) + 1
    if source_dist:
        print(f"[Discovery] Source distribution:")
        for k in sorted(source_dist.keys()):
            print(f"  {k} sources: {source_dist[k]} profiles")

    profiles = profiles[args.start:]
    if args.limit:
        profiles = profiles[:args.limit]

    print(f"[Discovery] Final queue (start={args.start}, limit={args.limit or 'all'}): {len(profiles)} profiles")
    return profiles


def _load_profile_info(pdir: Path) -> Optional[Dict]:
    profile_json = pdir / f"{pdir.name}.json"
    if not profile_json.exists():
        return None

    try:
        pdata = json.loads(profile_json.read_text(encoding="utf-8"))
    except Exception:
        return None

    enrichment_text = pdir / "enrichment_text.txt"
    enrichment_json = pdir / "enrichment.json"
    successful_sources = 0
    confidence = 0.0
    source_names = []

    if enrichment_json.exists():
        try:
            edata = json.loads(enrichment_json.read_text(encoding="utf-8"))
            successful_sources = edata.get("summary", {}).get("successful_sources", 0)
            confidence = edata.get("confidence", {}).get("overall_confidence", 0.0)
            source_names = edata.get("summary", {}).get("successful_source_names", [])
        except Exception:
            pass

    # Check file sizes and modification times
    enrichment_size = enrichment_text.stat().st_size if enrichment_text.exists() else 0
    enrichment_mtime = enrichment_text.stat().st_mtime if enrichment_text.exists() else 0
    cleaned_path = pdir / "enrichment_text_cleaned.txt"
    cleaned_size = cleaned_path.stat().st_size if cleaned_path.exists() else 0
    chunks_path = Path("output/osu_faculty_run/chunked_profiles") / pdir.name / "chunks.json"
    has_chunks = chunks_path.exists()
    chunks_mtime = chunks_path.stat().st_mtime if has_chunks else 0

    # Detect stale: enrichment data is newer than chunks (e.g. re-enriched)
    is_stale = has_chunks and enrichment_mtime > chunks_mtime

    return {
        "profile_id": pdata.get("profile_id", pdir.name),
        "name": pdata.get("name", ""),
        "profile_url": pdata.get("profile_url", ""),
        "department": pdata.get("department", ""),
        "dir": pdir,
        "has_enrichment": enrichment_text.exists(),
        "successful_sources": successful_sources,
        "confidence": confidence,
        "source_names": source_names,
        "enrichment_size": enrichment_size,
        "cleaned_size": cleaned_size,
        "has_chunks": has_chunks,
        "is_stale": is_stale,
    }


def clean_single(profile: Dict) -> bool:
    """Clean enrichment_text.txt with GPT-4o-mini."""
    pdir = profile["dir"]
    text_path = pdir / "enrichment_text.txt"
    cleaned_path = pdir / "enrichment_text_cleaned.txt"

    if not text_path.exists():
        print(f"  [Clean] No enrichment_text.txt found, skipping")
        return False

    raw_size = text_path.stat().st_size
    print(f"  [Clean] Raw enrichment text: {raw_size:,} bytes")

    # Skip if already cleaned and newer than raw
    if cleaned_path.exists() and cleaned_path.stat().st_mtime > text_path.stat().st_mtime:
        cleaned_size = cleaned_path.stat().st_size
        print(f"  [Clean] Already cleaned ({cleaned_size:,} bytes), skipping")
        return True

    from enrichment.enrichment_cleaner import EnrichmentCleaner
    cleaner = EnrichmentCleaner()
    t0 = time.perf_counter()
    try:
        result_path = cleaner.clean_file(text_path)
        elapsed = time.perf_counter() - t0
        cleaned_size = result_path.stat().st_size
        print(f"  [Clean] Done: {raw_size:,} -> {cleaned_size:,} bytes ({elapsed:.1f}s)")
        return True
    except Exception as e:
        elapsed = time.perf_counter() - t0
        print(f"  [Clean] FAILED after {elapsed:.1f}s: {e}")
        return False


def main():
    args = parse_args()

    # ── Banner ──
    print()
    print("=" * 70)
    print("  CLEAN + FULL SYNC (Pinecone + MongoDB scholars)")
    print("=" * 70)
    print(f"  Profiles dir:  {args.profiles_dir}")
    print(f"  Chunks dir:    {args.chunks_dir}")
    print(f"  Cleaning:      {'SKIP' if args.skip_cleaning else 'GPT-4o-mini'}")
    print(f"  Pinecone:      {'SKIP' if args.skip_pinecone else 'ENABLED'}")
    print(f"  MongoDB:       {'SKIP' if args.skip_mongodb else 'ENABLED'}")
    print()

    # ── Env check ──
    print("[Env] Checking environment variables...")
    openai_ok = bool(os.getenv("OPENAI_API_KEY"))
    pinecone_ok = bool(os.getenv("PINECONE_API_KEY"))
    mongodb_ok = bool(os.getenv("MONGODB_URI"))
    anthropic_ok = bool(os.getenv("ANTHROPIC_API_KEY"))
    print(f"  OPENAI_API_KEY:    {'SET' if openai_ok else 'MISSING'}")
    print(f"  PINECONE_API_KEY:  {'SET' if pinecone_ok else 'MISSING'}")
    print(f"  MONGODB_URI:       {'SET' if mongodb_ok else 'MISSING'}")
    print(f"  ANTHROPIC_API_KEY: {'SET' if anthropic_ok else 'MISSING'} (optional)")

    if not args.skip_cleaning and not openai_ok:
        print("\nERROR: OPENAI_API_KEY required for cleaning. Use --skip-cleaning to skip.")
        return
    if not args.skip_pinecone and not pinecone_ok:
        print("\nERROR: PINECONE_API_KEY required. Use --skip-pinecone to skip.")
        return
    if not args.skip_mongodb and not mongodb_ok:
        print("\nERROR: MONGODB_URI required. Use --skip-mongodb to skip.")
        return

    # ── Discover profiles ──
    profiles = discover_profiles(args)

    if not profiles:
        print("\nNo profiles match the filters.")
        return

    if args.dry_run:
        print(f"\n{'='*70}")
        print(f"DRY RUN - would process {len(profiles)} profiles:")
        print(f"{'='*70}")
        for i, p in enumerate(profiles[:50], 1):
            sources_str = ", ".join(p.get("source_names", [])) or "(none)"
            cleaned_str = "cleaned" if p["cleaned_size"] > 0 else "raw"
            chunks_str = "has_chunks" if p["has_chunks"] else "no_chunks"
            print(f"  {i:4d}. {p['name']:<30s} src={p['successful_sources']} conf={p['confidence']:.2f} [{cleaned_str}] [{chunks_str}] [{sources_str}]")
        if len(profiles) > 50:
            print(f"  ... and {len(profiles) - 50} more")
        return

    # ── Import sync function ──
    print("\n[Setup] Importing incremental sync pipeline...")
    from enrichment.incremental_sync import incremental_sync_professor
    print("[Setup] Ready")

    # ── Process loop ──
    pipeline_start = time.perf_counter()
    cleaned_count = 0
    chunked_count = 0
    pinecone_count = 0
    pinecone_vectors_total = 0
    mongo_count = 0
    failed_count = 0
    skipped_count = 0
    total_clean_time = 0.0
    total_sync_time = 0.0

    for i, profile in enumerate(profiles, 1):
        prof_start = time.perf_counter()
        pid = profile["profile_id"]
        name = profile["name"]
        pdir = profile["dir"]

        # ── Resume logic: skip fully-processed profiles ──
        is_cleaned = profile["cleaned_size"] > 0
        has_chunks = profile["has_chunks"]
        is_stale = profile.get("is_stale", False)
        fully_done = is_cleaned and has_chunks and not is_stale

        if fully_done and not args.force:
            skipped_count += 1
            if skipped_count <= 5:
                print(f"  [{i}/{len(profiles)}] SKIP (already done): {name}")
            elif skipped_count == 6:
                print(f"  ... skipping more already-processed profiles (use --force to re-process)")
            continue

        if is_stale and not args.force:
            print(f"  [{i}/{len(profiles)}] RE-PROCESSING (enrichment updated): {name}")

        print(f"\n{'='*70}")
        print(f"[{i}/{len(profiles)}] {name}")
        print(f"{'='*70}")
        print(f"  Profile ID:   {pid}")
        print(f"  Department:   {profile.get('department') or '(unknown)'}")
        print(f"  Sources:      {profile['successful_sources']} ({', '.join(profile.get('source_names', [])) or 'none'})")
        print(f"  Confidence:   {profile['confidence']:.3f}")
        print(f"  Enrichment:   {profile['enrichment_size']:,} bytes")
        print(f"  Already cleaned: {'Yes (%s bytes)' % '{:,}'.format(profile['cleaned_size']) if profile['cleaned_size'] > 0 else 'No'}")
        print(f"  Has chunks:   {'Yes' if has_chunks else 'No'}")

        # ── Step 1: Clean enrichment text ──
        if not args.skip_cleaning:
            print(f"\n  --- Step 1/4: GPT-4o-mini Cleaning ---")
            t0 = time.perf_counter()
            ok = clean_single(profile)
            clean_elapsed = time.perf_counter() - t0
            total_clean_time += clean_elapsed
            if ok:
                cleaned_count += 1
            else:
                print(f"  [Clean] Failed, will use raw enrichment text")
        else:
            print(f"\n  --- Step 1/4: Cleaning SKIPPED ---")

        # ── Steps 2-4: Chunk -> Pinecone -> MongoDB ──
        print(f"\n  --- Steps 2-4: Chunk -> Pinecone -> MongoDB ---")
        t0 = time.perf_counter()
        try:
            sync_result = incremental_sync_professor(
                profile_dir=pdir,
                chunking_output_dir=args.chunks_dir,
                professor_name=name,
                skip_pinecone=args.skip_pinecone,
                skip_mongodb=args.skip_mongodb,
            )
            sync_elapsed = time.perf_counter() - t0
            total_sync_time += sync_elapsed

            # Track results
            n_chunks = sync_result.get("chunks_count", 0)
            n_vectors = sync_result.get("pinecone_uploaded", 0)
            did_chunk = sync_result.get("chunked", False)
            did_mongo = sync_result.get("mongodb_synced", False)

            if did_chunk:
                chunked_count += 1
            if n_vectors > 0:
                pinecone_count += 1
                pinecone_vectors_total += n_vectors
            if did_mongo:
                mongo_count += 1

            if not did_chunk and not did_mongo:
                failed_count += 1

            # Per-professor summary
            prof_elapsed = time.perf_counter() - prof_start
            print(f"\n  --- Result for {name} ---")
            print(f"  Chunked:     {'Yes (%d chunks)' % n_chunks if did_chunk else 'No'}")
            print(f"  Pinecone:    {'%d vectors uploaded' % n_vectors if n_vectors > 0 else ('SKIPPED' if args.skip_pinecone else 'No vectors')}")
            print(f"  MongoDB:     {'Synced' if did_mongo else ('SKIPPED' if args.skip_mongodb else 'Failed')}")
            print(f"  Time:        {prof_elapsed:.1f}s (clean={clean_elapsed if not args.skip_cleaning else 0:.1f}s, sync={sync_elapsed:.1f}s)")

        except Exception as e:
            failed_count += 1
            prof_elapsed = time.perf_counter() - prof_start
            print(f"\n  EXCEPTION after {prof_elapsed:.1f}s: {e}")
            import traceback
            traceback.print_exc()

        # Running totals every 10 profiles
        if i % 10 == 0 or i == len(profiles):
            elapsed_so_far = time.perf_counter() - pipeline_start
            rate = elapsed_so_far / i
            eta = rate * (len(profiles) - i)
            print(f"\n  [Progress] {i}/{len(profiles)} done | "
                  f"cleaned={cleaned_count} chunked={chunked_count} pinecone={pinecone_count} mongo={mongo_count} failed={failed_count} | "
                  f"elapsed={elapsed_so_far:.0f}s | ETA={eta:.0f}s ({eta/60:.1f}min)")

    # ── Final summary ──
    total_elapsed = time.perf_counter() - pipeline_start

    print(f"\n{'='*70}")
    print(f"  PIPELINE COMPLETE")
    print(f"{'='*70}")
    print(f"  Total profiles processed:   {len(profiles)}")
    print(f"  Cleaned (GPT-4o-mini):      {cleaned_count}")
    print(f"  Chunked:                    {chunked_count}")
    print(f"  Pinecone uploads:           {pinecone_count} ({pinecone_vectors_total:,} total vectors)")
    print(f"  MongoDB scholars synced:    {mongo_count}")
    print(f"  Failed:                     {failed_count}")
    print(f"  Skipped:                    {skipped_count}")
    print(f"")
    print(f"  Total time:                 {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
    print(f"  Avg per profile:            {total_elapsed/max(len(profiles),1):.1f}s")
    print(f"  Time in cleaning:           {total_clean_time:.1f}s")
    print(f"  Time in chunk+sync:         {total_sync_time:.1f}s")
    print(f"")
    print(f"  Success rate:               {(chunked_count + mongo_count) / max(len(profiles)*2, 1) * 100:.1f}%")
    if mongo_count > 0:
        print(f"\n  >>> {mongo_count} profiles now available in MongoDB 'scholars' collection <<<")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
