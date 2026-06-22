"""
Migrate collections from one MongoDB cluster to another.

Source defaults to MONGODB_SOURCE_URI (or --source-uri).
Dest   defaults to MONGODB_URI in .env (the new cluster) (or --dest-uri).

Copies are idempotent and resumable: documents are upserted by _id with
ReplaceOne, so re-running continues/repairs a partial migration.

Named scopes (--scope):
  core     scholars, legend_scholars, enrichment_raw
  scholar  core + scholar-derived (features/vectors/neighbors/clusters/lexicon/
           review/daily-story)            <- default
  all      every non-system collection present in the source DB

Examples:
  # Compare both clusters (read-only):
  python migrate_cluster.py --list

  # Dry-run the default scholar scope:
  python migrate_cluster.py

  # Actually migrate the scholar scope:
  python migrate_cluster.py --yes

  # Migrate only the core collections:
  python migrate_cluster.py --scope core --yes

  # Migrate specific collections, dropping dest first:
  python migrate_cluster.py --collections scholars --drop-dest --yes
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

from config.mongodb_utils import create_mongo_client, resolve_mongo_db_name

CORE = ["scholars", "legend_scholars", "enrichment_raw"]
SCHOLAR_DERIVED = [
    "profile_features",
    "profile_vectors",
    "profile_neighbors",
    "cluster_assignments",
    "domain_lexicon",
    "professorchunkexposures",
    "scholar_review_items",
    "scholar_review_batches",
    "legend_scholar_daily_stories",
    "daily_story_jobs",
    "daily_story_profile_quality",
    "daily_story_profile_quality_jobs",
    "daily_story_trend_issues",
    "daily_story_trend_jobs",
]


def _read_commented_old_uri() -> Optional[str]:
    """Best-effort: pull a commented '#mongodb+srv://...' line from .env."""
    env_path = Path(".env")
    if not env_path.exists():
        return None
    for line in env_path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s.startswith("#") and "mongodb+srv://" in s:
            return s.lstrip("#").strip()
    return None


def _resolve_source_uri(arg: Optional[str]) -> str:
    uri = arg or os.getenv("MONGODB_SOURCE_URI") or _read_commented_old_uri()
    if not uri:
        raise SystemExit(
            "No source URI. Pass --source-uri, set MONGODB_SOURCE_URI, "
            "or keep the old URI as a commented line in .env."
        )
    return uri


def _select_collections(args, src_db) -> List[str]:
    if args.collections:
        return list(args.collections)
    if args.scope == "core":
        wanted = CORE
    elif args.scope == "all":
        return sorted(
            n for n in src_db.list_collection_names() if not n.startswith("system.")
        )
    else:  # scholar
        wanted = CORE + SCHOLAR_DERIVED
    present = set(src_db.list_collection_names())
    return [c for c in wanted if c in present]


def _copy_collection(src_db, dst_db, name: str, batch_size: int, drop_dest: bool) -> tuple[int, int]:
    from pymongo import ReplaceOne

    src = src_db[name]
    dst = dst_db[name]
    total = src.estimated_document_count()
    if drop_dest:
        dst_db.drop_collection(name)
        print(f"    dropped dest collection: {name}")

    # Paginate by _id so we never hold a long-lived cursor (Atlas shared tiers
    # disallow noTimeout cursors and idle cursors expire). Fully resumable.
    copied = 0
    last_id = None
    while True:
        query = {} if last_id is None else {"_id": {"$gt": last_id}}
        docs = list(src.find(query).sort("_id", 1).limit(batch_size))
        if not docs:
            break
        ops = [ReplaceOne({"_id": d["_id"]}, d, upsert=True) for d in docs]
        dst.bulk_write(ops, ordered=False)
        copied += len(docs)
        last_id = docs[-1]["_id"]
        print(f"    {name}: {copied}/{total}", end="\r", flush=True)
    dest_count = dst.estimated_document_count()
    print(f"    {name}: copied {copied} (source~{total}, dest now~{dest_count})        ")
    return copied, dest_count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-uri", default=None)
    parser.add_argument("--dest-uri", default=None)
    parser.add_argument("--scope", choices=["core", "scholar", "all"], default="scholar")
    parser.add_argument("--collections", nargs="*", default=None,
                        help="Explicit collection names (overrides --scope).")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--drop-dest", action="store_true",
                        help="Drop each dest collection before copying (clean import).")
    parser.add_argument("--list", action="store_true",
                        help="Show collections + counts on both clusters and exit.")
    parser.add_argument("--yes", action="store_true", help="Actually perform the copy.")
    args = parser.parse_args()

    load_dotenv(".env")
    source_uri = _resolve_source_uri(args.source_uri)
    dest_uri = args.dest_uri or os.getenv("MONGODB_URI")
    if not dest_uri:
        raise SystemExit("No dest URI. Set MONGODB_URI in .env or pass --dest-uri.")

    src_db_name = resolve_mongo_db_name(source_uri)
    dst_db_name = resolve_mongo_db_name(dest_uri)

    def _host(uri: str) -> str:
        try:
            return uri.split("@", 1)[1].split("/", 1)[0]
        except Exception:
            return "?"

    print(f"SOURCE: {_host(source_uri)}  db={src_db_name}")
    print(f"DEST:   {_host(dest_uri)}  db={dst_db_name}")
    if _host(source_uri) == _host(dest_uri):
        raise SystemExit("Source and dest hosts are identical; aborting to avoid a no-op/self-copy.")

    src_client = create_mongo_client(source_uri)
    dst_client = create_mongo_client(dest_uri)
    try:
        src_db = src_client[src_db_name]
        dst_db = dst_client[dst_db_name]

        if args.list:
            src_names = sorted(src_db.list_collection_names())
            dst_names = set(dst_db.list_collection_names())
            print(f"\nSOURCE has {len(src_names)} collections:")
            for n in src_names:
                sc = src_db[n].estimated_document_count()
                dc = dst_db[n].estimated_document_count() if n in dst_names else "-"
                print(f"  src={sc:>8}  dest={str(dc):>8}  {n}")
            return 0

        collections = _select_collections(args, src_db)
        if not collections:
            print("No matching collections found in source.")
            return 0

        print(f"\nScope='{args.scope}' -> {len(collections)} collection(s) to migrate:")
        grand = 0
        for n in collections:
            c = src_db[n].estimated_document_count()
            grand += c
            print(f"  {c:>8}  {n}")
        print(f"Total ~{grand} documents.")

        if not args.yes:
            print("\nDRY RUN. Re-run with --yes to perform the migration.")
            return 0

        print("\nMigrating...")
        for n in collections:
            print(f"  -> {n}")
            _copy_collection(src_db, dst_db, n, args.batch_size, args.drop_dest)
        print("\nDone. Verify with: python migrate_cluster.py --list")
        return 0
    finally:
        src_client.close()
        dst_client.close()


if __name__ == "__main__":
    sys.exit(main())
