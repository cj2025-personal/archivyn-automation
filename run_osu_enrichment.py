"""
End-to-end OSU enrichment runner.

Stages, in order:
  1. collect_osu_scholar_urls.py  -> DDG -> osu_scholars/final/*.txt
     (also writes the ``source`` field to each Mongo doc).
  2. run_legend_scholar_pipeline.py over the collected .txt files
     -> scrape -> license-aware chunk -> local chunked_profiles JSON.
     Files are processed in batches because Windows caps a command
     line at ~32K chars and there are thousands of .txt files.
  3. sync successful chunked runs into MongoDB (collection ``scholars``,
     tagged ``scholar_type='osu'``) via LocalChunkedMongoSync.

MongoDB is the default storage destination. The legend pipeline's in-run
Mongo sync stays disabled (it needs Pinecone); the post-pipeline sync in
stage 3 handles Mongo directly from the local chunked JSON.

Usage:
    python run_osu_enrichment.py                          # all scholars
    python run_osu_enrichment.py --limit 25               # first 25
    python run_osu_enrichment.py --start-from 500 --limit 100
    python run_osu_enrichment.py --skip-collection        # only run pipeline + sync
    python run_osu_enrichment.py --skip-pipeline          # only collect URLs
    python run_osu_enrichment.py --skip-existing          # resume mode
    python run_osu_enrichment.py --skip-pinecone          # skip vector upload
    python run_osu_enrichment.py --skip-mongodb           # don't write Mongo (debug)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _build_collect_cmd(args: argparse.Namespace) -> list:
    cmd = [sys.executable, "collect_osu_scholar_urls.py",
           "--excel-path", args.excel_path,
           "--max-urls", str(args.max_urls),
           "--output-dir", args.output_dir,
           "--checkpoint-file", args.checkpoint_file,
           "--log-file", args.log_file]
    if args.limit is not None:
        cmd += ["--limit", str(args.limit)]
    if args.start_from:
        cmd += ["--start-from", str(args.start_from)]
    if args.filter_field:
        cmd += ["--filter-field", args.filter_field]
    if args.filter_department:
        cmd += ["--filter-department", args.filter_department]
    if args.skip_existing:
        cmd += ["--skip-existing"]
    if args.sleep_seconds is not None:
        cmd += ["--sleep-seconds", str(args.sleep_seconds)]
    return cmd


def _list_url_files(output_dir: str) -> list:
    out = Path(output_dir)
    if not out.exists():
        return []
    return sorted(str(p) for p in out.glob("*.txt") if p.stat().st_size > 0)


def _build_pipeline_cmd(args: argparse.Namespace, url_files: list) -> list:
    cmd = [
        sys.executable,
        "run_legend_scholar_pipeline.py",
        *url_files,
        "--output-root",
        args.runs_root,
    ]
    if args.skip_pinecone:
        cmd.append("--skip-pinecone")
    # The legend pipeline's in-run Mongo sync requires Pinecone and writes the
    # richer MongoDBScholarSync document. We instead sync to Mongo in stage 3
    # from the local chunked JSON (LocalChunkedMongoSync), so always disable the
    # in-run Mongo sync here.
    cmd.append("--skip-mongodb")
    return cmd


def _slug_of(txt_path: str) -> str:
    """The .txt stem, which the legend pipeline uses as the output-dir suffix."""
    return Path(txt_path).stem


def _already_completed(slug: str, runs_root: Path) -> bool:
    """True if a prior pipeline run already produced a non-empty chunks.json
    for this scholar (output dir ``<timestamp>-<slug>/chunked_profiles``)."""
    if not runs_root.exists():
        return False
    for d in runs_root.glob(f"*-{slug}"):
        chunks_root = d / "chunked_profiles"
        if not chunks_root.exists():
            continue
        for chunk_dir in chunks_root.iterdir():
            if not chunk_dir.is_dir():
                continue
            chunks_json = chunk_dir / "chunks.json"
            if chunks_json.exists() and chunks_json.stat().st_size > 0:
                return True
    return False


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--start-from", type=int, default=0)
    p.add_argument("--excel-path", default="excel/OSU.xlsx")
    p.add_argument("--max-urls", type=int, default=50)
    p.add_argument("--sleep-seconds", type=float, default=2.5)
    p.add_argument("--output-dir", default="osu_scholars/final")
    p.add_argument("--checkpoint-file", default="osu_scholars/url_collection_checkpoint.json")
    p.add_argument("--log-file", default="osu_scholars/url_collection_log.jsonl")
    p.add_argument("--filter-field", default=None)
    p.add_argument("--filter-department", default=None)
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip scholars already collected (URL collection stage)")
    p.add_argument("--skip-collection", action="store_true",
                   help="Skip URL collection; only run the pipeline on existing .txt files")
    p.add_argument("--skip-pipeline", action="store_true",
                   help="Skip the legend pipeline; only collect URLs")
    p.add_argument("--skip-pinecone", action="store_true",
                   help="Pass through to the legend pipeline (skip vector upload)")
    p.add_argument("--skip-mongodb", action="store_true",
                   help="Skip the post-pipeline Mongo sync (stage 3). MongoDB is "
                        "the default destination; use this only for debugging.")
    p.add_argument("--batch-size", type=int, default=120,
                   help="URL files per legend-pipeline invocation. Windows "
                        "caps a command line at ~32K chars, so all 3,457 "
                        "paths cannot be passed at once — they are batched.")
    p.add_argument("--resume", action="store_true",
                   help="Stage 2: skip scholars whose pipeline output dir "
                        "already contains a non-empty chunks.json (re-run safe).")
    p.add_argument("--runs-root", default="output/osu_url_list_runs",
                   help="Where the OSU enrichment pipeline writes per-scholar output.")
    p.add_argument("--mongo-collection", default="scholars",
                   help="MongoDB collection to upsert OSU scholars into "
                        "(default: scholars; tagged scholar_type='osu').")
    p.add_argument("--no-mongo-llm", action="store_true",
                   help="Disable LLM summaries during the Mongo sync "
                        "(faster, less curated output).")
    args = p.parse_args()

    if not args.skip_collection:
        print("=" * 80)
        print("[OSU] Stage 1: collect URLs via DuckDuckGo")
        print("=" * 80)
        rc = subprocess.call(_build_collect_cmd(args))
        if rc != 0:
            print(f"[OSU] URL collection exited with code {rc}; aborting.")
            return rc
    else:
        print("[OSU] Skipping URL collection (--skip-collection).")

    if args.skip_pipeline:
        print("[OSU] Skipping legend pipeline (--skip-pipeline). Done.")
        return 0

    url_files = _list_url_files(args.output_dir)
    if not url_files:
        print(f"[OSU] No URL files in {args.output_dir}; nothing to scrape.")
        return 1

    runs_root = Path(args.runs_root)
    sync_slugs = [_slug_of(f) for f in url_files]
    if args.resume:
        before = len(url_files)
        url_files = [f for f in url_files if not _already_completed(_slug_of(f), runs_root)]
        print(f"[OSU] --resume: {before - len(url_files)} already completed, "
              f"{len(url_files)} remaining.")
    failed_batches = 0
    if not url_files:
        print("[OSU] Nothing left to scrape; proceeding to sync existing runs.")
    else:
        batch_size = max(1, args.batch_size)
        batches = [url_files[i:i + batch_size] for i in range(0, len(url_files), batch_size)]

        print()
        print("=" * 80)
        print(f"[OSU] Stage 2: legend pipeline on {len(url_files)} URL files "
              f"in {len(batches)} batch(es) of up to {batch_size}")
        print("=" * 80)

        for idx, batch in enumerate(batches, 1):
            print(f"\n[OSU] --- Batch {idx}/{len(batches)} ({len(batch)} scholars) ---")
            rc = subprocess.call(_build_pipeline_cmd(args, batch))
            if rc != 0:
                failed_batches += 1
                print(f"[OSU] Batch {idx} exited with code {rc} — continuing with next batch.")

        print()
        print("=" * 80)
        if failed_batches:
            print(f"[OSU] Stage 2 complete with {failed_batches}/{len(batches)} batch(es) reporting errors.")
            print("[OSU] Re-run with --resume to retry only the unfinished scholars.")

    if not args.skip_mongodb:
        print()
        print("=" * 80)
        print(f"[OSU] Stage 3: sync chunked profiles → MongoDB ({args.mongo_collection})")
        print("=" * 80)
        try:
            _sync_runs_to_mongo(
                runs_root=runs_root,
                slugs=sync_slugs,
                collection_name=args.mongo_collection,
                use_llm=not args.no_mongo_llm,
            )
        except Exception as e:
            print(f"[OSU] Mongo sync failed: {e}")
            return 1

    if failed_batches:
        return 1
    print("[OSU] Done — scraping, chunking, and storage complete.")
    return 0


def _sync_runs_to_mongo(
    runs_root: Path,
    slugs: list,
    collection_name: str,
    use_llm: bool,
) -> None:
    """Walk every <timestamp>-<slug> run dir for the given slugs and upsert
    each scholar into MongoDB (tagged scholar_type='osu'). Idempotent."""
    from sync_local_chunked_to_mongodb import LocalChunkedMongoSync

    sync = LocalChunkedMongoSync(
        collection_name=collection_name,
        use_llm=use_llm,
        scholar_type="osu",
    )
    try:
        if not runs_root.exists():
            print(f"[Mongo] runs_root does not exist: {runs_root}")
            return
        run_dirs: list = []
        for slug in slugs:
            run_dirs.extend(sorted(runs_root.glob(f"*-{slug}")))
        print(f"[Mongo] Found {len(run_dirs)} run dir(s) to sync.")
        for run_dir in run_dirs:
            chunks_root = run_dir / "chunked_profiles"
            profiles_root = run_dir / "profiles"
            sync.sync_from_roots(
                chunks_root,
                profiles_root if profiles_root.exists() else None,
            )
        print(f"[Mongo] Done. Upserted OSU scholars into '{collection_name}'.")
    finally:
        try:
            sync.mongo_client.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
