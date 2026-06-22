"""
End-to-end legendary scholar enrichment runner.

Stages:
  1. collect_legendary_scholar_urls.py -> DDG -> legendary_scholars/final/*.txt
  2. run_legend_scholar_pipeline.py over the collected .txt files
  3. sync successful chunked runs into Mongo collection legend_scholars (default)
  4. (optional, --use-sqlite) mirror the same successful runs into SQLite

MongoDB is the primary store. Mongo sync is intentionally done with
LocalChunkedMongoSync so legendary scholars stay out of the main scholars
collection. SQLite is opt-in via --use-sqlite.
"""
from __future__ import annotations

import argparse
import sqlite3
import subprocess
import sys
from pathlib import Path


def _build_collect_cmd(args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        "collect_legendary_scholar_urls.py",
        "--excel-path",
        args.excel_path,
        "--max-urls",
        str(args.max_urls),
        "--output-dir",
        args.output_dir,
        "--checkpoint-file",
        args.checkpoint_file,
        "--log-file",
        args.log_file,
        "--per-query-results",
        str(args.per_query_results),
    ]
    if args.limit is not None:
        cmd += ["--limit", str(args.limit)]
    if args.start_from:
        cmd += ["--start-from", str(args.start_from)]
    if args.slugs_file:
        cmd += ["--slugs-file", args.slugs_file]
    if args.skip_existing:
        cmd.append("--skip-existing")
    if args.sleep_seconds is not None:
        cmd += ["--sleep-seconds", str(args.sleep_seconds)]
    return cmd


def _list_url_files(output_dir: str) -> list[str]:
    root = Path(output_dir)
    if not root.exists():
        return []
    return sorted(str(path) for path in root.glob("*.txt") if path.stat().st_size > 0)


def _load_slugs_file(path: str | None) -> set[str]:
    if not path:
        return set()
    slugs = set()
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        slug = raw_line.strip()
        if not slug or slug.startswith("#"):
            continue
        slugs.add(slug)
    return slugs


def _build_pipeline_cmd(args: argparse.Namespace, url_files: list[str]) -> list[str]:
    cmd = [
        sys.executable,
        "run_legend_scholar_pipeline.py",
        *url_files,
        "--output-root",
        args.runs_root,
        "--skip-mongodb",
    ]
    if args.skip_pinecone:
        cmd.append("--skip-pinecone")
    return cmd


def _slug_of(txt_path: str) -> str:
    return Path(txt_path).stem


def _already_completed(slug: str, runs_root: Path) -> bool:
    if not runs_root.exists():
        return False
    for run_dir in runs_root.glob(f"*-{slug}"):
        chunks_root = run_dir / "chunked_profiles"
        if not chunks_root.exists():
            continue
        for chunk_dir in chunks_root.iterdir():
            if not chunk_dir.is_dir():
                continue
            chunks_json = chunk_dir / "chunks.json"
            if chunks_json.exists() and chunks_json.stat().st_size > 0:
                return True
    return False


def _sync_runs_to_mongo(
    *,
    runs_root: Path,
    slugs: list[str],
    collection_name: str,
    use_llm: bool,
) -> None:
    from sync_local_chunked_to_mongodb import LocalChunkedMongoSync

    sync = LocalChunkedMongoSync(collection_name=collection_name, use_llm=use_llm)
    try:
        run_dirs: list[Path] = []
        for slug in slugs:
            run_dirs.extend(sorted(runs_root.glob(f"*-{slug}")))
        print(f"[LegendMongo] Found {len(run_dirs)} run dir(s) to sync.")
        for run_dir in run_dirs:
            chunks_root = run_dir / "chunked_profiles"
            profiles_root = run_dir / "profiles"
            sync.sync_from_roots(
                chunks_root,
                profiles_root if profiles_root.exists() else None,
            )
    finally:
        sync.close()


def _sync_runs_to_sqlite(
    *,
    runs_root: Path,
    slugs: list[str],
    db_path: str,
    table: str,
    use_llm: bool,
) -> None:
    from sync_local_chunked_to_sqlite import LocalChunkedSqliteSync
    from config import sqlite_utils

    sync = LocalChunkedSqliteSync(db_path=db_path, table=table, use_llm=use_llm)
    try:
        run_dirs: list[Path] = []
        for slug in slugs:
            run_dirs.extend(sorted(runs_root.glob(f"*-{slug}")))
        print(f"[LegendSQLite] Found {len(run_dirs)} run dir(s) to sync.")
        for run_dir in run_dirs:
            chunks_root = run_dir / "chunked_profiles"
            profiles_root = run_dir / "profiles"
            sync.sync_from_roots(
                chunks_root,
                profiles_root if profiles_root.exists() else None,
            )
        total = sqlite_utils.count_scholars(sync.conn, table=table)
        print(f"[LegendSQLite] Done. Total rows in '{table}': {total}  (db={db_path})")
    finally:
        sync.close()


def _reset_mongo_collection(collection_name: str) -> None:
    import os
    from dotenv import load_dotenv
    from config.mongodb_utils import create_mongo_client, resolve_mongo_db_name

    load_dotenv(".env")
    uri = os.getenv("MONGODB_URI")
    if not uri:
        raise ValueError("MONGODB_URI not set")
    client = create_mongo_client(uri)
    try:
        db = client[resolve_mongo_db_name(uri)]
        db[collection_name].delete_many({})
        print(f"[Legendary] Cleared Mongo collection: {collection_name}")
    finally:
        client.close()


def _reset_sqlite_table(db_path: str, table: str) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(f"DROP TABLE IF EXISTS {table}")
        conn.commit()
        print(f"[Legendary] Dropped SQLite table if present: {table}  (db={db_path})")
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--excel-path", default="excel/legendary.xlsx")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-from", type=int, default=0)
    parser.add_argument("--max-urls", type=int, default=50)
    parser.add_argument("--sleep-seconds", type=float, default=2.0)
    parser.add_argument("--per-query-results", type=int, default=15)
    parser.add_argument("--output-dir", default="legendary_scholars/final")
    parser.add_argument("--checkpoint-file", default="legendary_scholars/url_collection_checkpoint.json")
    parser.add_argument("--log-file", default="legendary_scholars/url_collection_log.jsonl")
    parser.add_argument(
        "--slugs-file",
        default=None,
        help="Optional newline-delimited slug file to target only specific legendary scholars.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--skip-collection", action="store_true")
    parser.add_argument("--skip-pipeline", action="store_true")
    parser.add_argument("--skip-pinecone", action="store_true")
    parser.add_argument("--skip-mongodb", action="store_true",
                       help="Skip the post-pipeline Mongo sync into legend_scholars.")
    parser.add_argument("--batch-size", type=int, default=60)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--runs-root", default="output/legend_url_list_runs")
    parser.add_argument("--mongo-collection", default="legend_scholars")
    parser.add_argument("--reset-mongo-collection", action="store_true")
    parser.add_argument("--no-mongo-llm", action="store_true",
                       help="Disable LLM summaries during Mongo sync.")
    parser.add_argument("--use-sqlite", action="store_true",
                       help="Also mirror chunked runs into SQLite (off by default).")
    parser.add_argument("--sqlite-path", default="data/scholars.db")
    parser.add_argument("--sqlite-table", default="legend_scholars")
    parser.add_argument("--reset-sqlite-table", action="store_true")
    parser.add_argument("--no-sqlite-llm", action="store_true",
                       help="Disable LLM summaries during SQLite sync.")
    args = parser.parse_args()

    if args.reset_mongo_collection and args.skip_mongodb:
        raise ValueError("--reset-mongo-collection cannot be combined with --skip-mongodb")
    if args.reset_sqlite_table and not args.use_sqlite:
        raise ValueError("--reset-sqlite-table requires --use-sqlite")

    if args.reset_mongo_collection:
        _reset_mongo_collection(args.mongo_collection)
    if args.reset_sqlite_table:
        _reset_sqlite_table(args.sqlite_path, args.sqlite_table)

    if not args.skip_collection:
        print("=" * 80)
        print("[Legendary] Stage 1: collect URLs via DuckDuckGo")
        print("=" * 80)
        rc = subprocess.call(_build_collect_cmd(args))
        if rc != 0:
            print(f"[Legendary] URL collection exited with code {rc}; aborting.")
            return rc
    else:
        print("[Legendary] Skipping URL collection (--skip-collection).")

    if args.skip_pipeline:
        print("[Legendary] Skipping legend pipeline (--skip-pipeline).")
        return 0

    url_files = _list_url_files(args.output_dir)
    target_slugs = _load_slugs_file(args.slugs_file)
    if target_slugs:
        url_files = [path for path in url_files if _slug_of(path) in target_slugs]
    if not url_files:
        print(f"[Legendary] No URL files in {args.output_dir}; nothing to scrape.")
        return 1

    sync_slugs = [_slug_of(path) for path in url_files]
    runs_root = Path(args.runs_root)
    if args.resume:
        before = len(url_files)
        url_files = [path for path in url_files if not _already_completed(_slug_of(path), runs_root)]
        print(f"[Legendary] --resume: {before - len(url_files)} already completed, {len(url_files)} remaining.")
    if not url_files:
        print("[Legendary] Nothing left to scrape.")
    else:
        batch_size = max(1, args.batch_size)
        batches = [url_files[i:i + batch_size] for i in range(0, len(url_files), batch_size)]
        print()
        print("=" * 80)
        print(
            f"[Legendary] Stage 2: legend pipeline on {len(url_files)} URL files "
            f"in {len(batches)} batch(es) of up to {batch_size}"
        )
        print("=" * 80)
        failed_batches = 0
        for idx, batch in enumerate(batches, 1):
            print(f"\n[Legendary] --- Batch {idx}/{len(batches)} ({len(batch)} scholars) ---")
            rc = subprocess.call(_build_pipeline_cmd(args, batch))
            if rc != 0:
                failed_batches += 1
                print(f"[Legendary] Batch {idx} exited with code {rc}; continuing with next batch.")
        if failed_batches:
            print(
                f"[Legendary] Stage 2 complete with {failed_batches}/{len(batches)} batch(es) reporting errors."
            )
            print("[Legendary] Re-run with --resume to retry only the unfinished scholars.")
            return_code = 1
        else:
            return_code = 0

    if not args.skip_mongodb:
        print()
        print("=" * 80)
        print(f"[Legendary] Stage 3: sync chunked runs -> Mongo ({args.mongo_collection})")
        print("=" * 80)
        _sync_runs_to_mongo(
            runs_root=runs_root,
            slugs=sync_slugs,
            collection_name=args.mongo_collection,
            use_llm=not args.no_mongo_llm,
        )

    if args.use_sqlite:
        print()
        print("=" * 80)
        print(f"[Legendary] Stage 4: mirror chunked runs -> SQLite ({args.sqlite_path})")
        print("=" * 80)
        _sync_runs_to_sqlite(
            runs_root=runs_root,
            slugs=sync_slugs,
            db_path=args.sqlite_path,
            table=args.sqlite_table,
            use_llm=not args.no_sqlite_llm,
        )

    return return_code if "return_code" in locals() else 0


if __name__ == "__main__":
    raise SystemExit(main())
