"""
Safe MongoDB cleanup helper.

By design this NEVER deletes anything unless you explicitly:
  1. name the collections to drop (or pass --empty-only / --from-file), AND
  2. pass --yes to confirm.

Without --yes it runs in dry-run mode and only prints what *would* happen.

Examples:
  # See what's there (counts), do nothing:
  python cleanup_mongo_collections.py --list

  # Dry-run dropping two collections (prints plan, deletes nothing):
  python cleanup_mongo_collections.py --drop foo bar

  # Actually drop them:
  python cleanup_mongo_collections.py --drop foo bar --yes

  # Drop every EMPTY (0-doc) collection (dry-run first, then --yes):
  python cleanup_mongo_collections.py --empty-only
  python cleanup_mongo_collections.py --empty-only --yes

  # Drop a curated list from a text file (one collection name per line):
  python cleanup_mongo_collections.py --from-file drop_list.txt --yes
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from config.mongodb_utils import create_mongo_client, resolve_mongo_db_name

# Collections that must never be dropped by this tool, regardless of input.
# These hold the scraping project's primary data.
PROTECTED = {
    "scholars",
    "legend_scholars",
    "enrichment_raw",
}


def _connect():
    load_dotenv(".env")
    uri = os.getenv("MONGODB_URI")
    if not uri:
        raise SystemExit("MONGODB_URI not set in .env")
    client = create_mongo_client(uri)
    db = client[resolve_mongo_db_name(uri)]
    return client, db


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true", help="List all collections with counts and exit.")
    parser.add_argument("--drop", nargs="*", default=[], help="Explicit collection names to drop.")
    parser.add_argument("--from-file", default=None, help="Path to a file with one collection name per line to drop.")
    parser.add_argument("--empty-only", action="store_true", help="Target every collection whose document count is 0.")
    parser.add_argument("--allow-protected", action="store_true",
                        help="Permit dropping protected collections (scholars/legend_scholars/enrichment_raw).")
    parser.add_argument("--yes", action="store_true", help="Actually perform the drops. Without this it's a dry run.")
    args = parser.parse_args()

    client, db = _connect()
    try:
        names = sorted(db.list_collection_names())
        counts = {n: db[n].estimated_document_count() for n in names}

        if args.list:
            print(f"DB has {len(names)} collections:")
            for n in names:
                print(f"  {counts[n]:>8}  {n}")
            return 0

        targets: list[str] = []
        if args.empty_only:
            targets += [n for n in names if counts[n] == 0]
        if args.from_file:
            file_names = [
                line.strip()
                for line in Path(args.from_file).read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.strip().startswith("#")
            ]
            targets += file_names
        targets += list(args.drop)

        # De-dupe, keep only real collections.
        seen = set()
        resolved: list[str] = []
        missing: list[str] = []
        for t in targets:
            if t in seen:
                continue
            seen.add(t)
            if t in names:
                resolved.append(t)
            else:
                missing.append(t)

        if not args.allow_protected:
            blocked = [t for t in resolved if t in PROTECTED]
            resolved = [t for t in resolved if t not in PROTECTED]
            for b in blocked:
                print(f"  [protected] refusing to drop: {b} (use --allow-protected to override)")

        if missing:
            for m in missing:
                print(f"  [skip] not found: {m}")

        if not resolved:
            print("Nothing to drop.")
            return 0

        print("\nPlan — collections to DROP:")
        total_docs = 0
        for t in resolved:
            total_docs += counts[t]
            print(f"  {counts[t]:>8}  {t}")
        print(f"Total: {len(resolved)} collection(s), ~{total_docs} document(s).")

        if not args.yes:
            print("\nDRY RUN. Re-run with --yes to actually drop these collections.")
            return 0

        print("\nDropping...")
        for t in resolved:
            db.drop_collection(t)
            print(f"  dropped: {t}")
        print(f"Done. Dropped {len(resolved)} collection(s).")
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())
