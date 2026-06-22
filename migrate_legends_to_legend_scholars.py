"""
Migrate the 5 new legend docs from the regular ``scholars`` collection
to ``legend_scholars`` (where they belong) and delete duplicates.

Background:
- ``legend_scholars`` already holds 4 correctly-named docs (Jackson,
  Lewis, Franklin, Hoxby).
- The 5 new legends (Woodson, Diop, Owens, Van Sertima, Du Bois) were
  pushed to ``scholars`` by the default sync target across multiple
  pipeline runs, producing 4 duplicates of each.

This script:
  1. For each of the 5 names, finds the most-recent doc in ``scholars``
     using ``updatedAt`` (falls back to ``updated_at`` / ``created_at``).
  2. Upserts that doc into ``legend_scholars`` keyed by ``_id``.
  3. Deletes ALL ``scholars`` docs whose ``professor_name`` or
     ``name.full`` matches the legend name (cleans up duplicates).

Usage:
    python migrate_legends_to_legend_scholars.py            # dry run
    python migrate_legends_to_legend_scholars.py --apply    # write
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))

from config.mongodb_utils import create_mongo_client, resolve_mongo_db_name


LEGEND_NAMES = [
    "Carter G. Woodson",
    "Cheikh Anta Diop",
    "Ida Stephens Owens",
    "Ivan Van Sertima",
    "W. E. B. Du Bois",
]

# Canonical profile_ids for the 5 new legends, taken from the latest
# local run (output/url_list_runs/20260507_171544-*) which used the
# fully-fixed pipeline. Mongo docs predate the timestamp field so we
# can't reliably pick by ``updatedAt`` — these IDs are authoritative.
CANONICAL_IDS_BY_NAME = {
    "Carter G. Woodson":  "bfe8d1c3-03e1-41c2-aa51-89ec4e831637",
    "Cheikh Anta Diop":   "bf2fd888-138e-45bb-af56-c587a48abe2a",
    "Ida Stephens Owens": "d6e66dcb-50e8-42a2-a62a-20e4acd4f105",
    "Ivan Van Sertima":   "751bafe8-7910-4246-b5ba-086a3ab0e8b4",
    "W. E. B. Du Bois":   "1aee7898-6a22-472e-9cb2-834dd441e4a6",
}


def _name_filter(name: str) -> dict:
    return {
        "$or": [
            {"professor_name": name},
            {"name.full": name},
        ]
    }


def _doc_timestamp(doc: dict):
    return (
        doc.get("updatedAt")
        or doc.get("updated_at")
        or doc.get("createdAt")
        or doc.get("created_at")
    )


def migrate(apply_changes: bool) -> int:
    load_dotenv()
    uri = os.getenv("MONGODB_URI")
    if not uri:
        print("MONGODB_URI not set", file=sys.stderr)
        return 2
    client = create_mongo_client(uri)
    db = client[resolve_mongo_db_name(uri)]
    scholars = db.scholars
    legend_scholars = db.legend_scholars

    summary = []
    for name in LEGEND_NAMES:
        candidates = list(scholars.find(_name_filter(name)))
        if not candidates:
            print(f"  [SKIP] {name}: no docs in scholars collection")
            continue

        # Prefer the explicitly-known canonical id from the latest run.
        canonical_id = CANONICAL_IDS_BY_NAME.get(name)
        canonical = next((d for d in candidates if d["_id"] == canonical_id), None)
        if not canonical:
            # Fallback: pick most-recent by timestamp; final fallback is
            # the first doc returned.
            candidates.sort(key=lambda d: (_doc_timestamp(d) or ""), reverse=True)
            canonical = candidates[0]
            canonical_id = canonical["_id"]
            print(f"  [WARN] {name}: canonical id from latest run not found in scholars; "
                  f"falling back to {canonical_id}")

        already_in_legends = legend_scholars.count_documents({"_id": canonical_id})

        action = []
        if already_in_legends:
            action.append(f"already in legend_scholars: keep id={canonical_id}")
        else:
            action.append(f"copy id={canonical_id} -> legend_scholars")
        del_ids = [d["_id"] for d in candidates]  # all copies (incl. canonical)
        action.append(f"delete {len(del_ids)} doc(s) from scholars")

        print(f"  {name}: {' | '.join(action)}")
        for d in candidates:
            ts = _doc_timestamp(d) or "?"
            marker = "**" if d["_id"] == canonical_id else "  "
            print(f"    {marker} _id={d['_id']}  updated={ts}")

        summary.append((name, canonical, del_ids))

    if not summary:
        print("\nNothing to do.")
        return 0

    if not apply_changes:
        print("\nDry run — pass --apply to perform the migration.")
        return 0

    print()
    moved = 0
    deleted = 0
    for name, canonical, del_ids in summary:
        canonical_id = canonical["_id"]
        # 1. Upsert into legend_scholars by _id (preserve original _id)
        legend_scholars.replace_one({"_id": canonical_id}, canonical, upsert=True)
        moved += 1
        # 2. Remove ALL copies (incl. the canonical) from scholars
        res = scholars.delete_many({"_id": {"$in": del_ids}})
        deleted += res.deleted_count
        print(f"  {name}: upserted to legend_scholars, deleted {res.deleted_count} from scholars")

    print()
    print(f"Migration complete. moved={moved}, deleted_from_scholars={deleted}")

    print()
    print("Final state:")
    print(f"  legend_scholars total: {legend_scholars.estimated_document_count()}")
    for d in legend_scholars.find({}, {"professor_name": 1, "name.full": 1, "_id": 1}):
        print(f"   - {d.get('professor_name') or (d.get('name') or {}).get('full')}  ({d['_id']})")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Perform writes (default: dry run)")
    args = parser.parse_args()
    return migrate(apply_changes=args.apply)


if __name__ == "__main__":
    raise SystemExit(main())
