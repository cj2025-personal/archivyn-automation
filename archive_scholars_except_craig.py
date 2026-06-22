"""
Archive all scholars except one target record.

Behavior:
- Reads MONGODB_URI from environment/.env
- Resolves DB name from URI (same helper as other repo scripts)
- Keeps exactly one matching scholar record (first by _id)
- Copies all other scholars into `archived_scholars` (upsert by _id)
- Deletes copied records from `scholars`
- Verifies `scholars` ends with exactly one record

Usage:
    python archive_scholars_except_craig.py --yes
    python archive_scholars_except_craig.py --name "Craig Johnson" --yes
"""

from __future__ import annotations

import argparse
import os
import re
from typing import Any, Dict, Iterable, List

from dotenv import load_dotenv
from pymongo import ReplaceOne

from config.mongodb_utils import create_mongo_client, resolve_mongo_db_name


def build_name_filter(full_name: str) -> Dict[str, Any]:
    parts = [re.escape(p) for p in full_name.strip().split() if p.strip()]
    if not parts:
        raise ValueError("Target name is empty.")

    # Exact name match, tolerant of extra whitespace/casing and optional title prefix.
    base_name = r"\s+".join(parts)
    pattern = r"^\s*(?:(?:Dr\.?|Prof\.?|Professor)\s+)?" + base_name + r"\s*$"

    return {
        "$or": [
            {"name.full": {"$regex": pattern, "$options": "i"}},
            {"name.display": {"$regex": pattern, "$options": "i"}},
            {"professor_name": {"$regex": pattern, "$options": "i"}},
            {"name": {"$regex": pattern, "$options": "i"}},
        ]
    }


def chunked(values: List[Any], size: int) -> Iterable[List[Any]]:
    for i in range(0, len(values), size):
        yield values[i : i + size]


def archive_scholars_except_one(
    *,
    target_name: str,
    source_collection: str,
    archive_collection: str,
    batch_size: int,
    execute: bool,
) -> None:
    load_dotenv()

    mongodb_uri = os.getenv("MONGODB_URI")
    if not mongodb_uri:
        raise ValueError("MONGODB_URI not found in environment variables/.env")

    db_name = resolve_mongo_db_name(mongodb_uri)
    client = create_mongo_client(mongodb_uri)
    db = client[db_name]

    scholars = db[source_collection]
    archived = db[archive_collection]
    target_filter = build_name_filter(target_name)

    total_before = scholars.count_documents({})
    if total_before == 0:
        raise RuntimeError(f"Collection '{source_collection}' is empty.")

    matches = list(
        scholars.find(
            target_filter,
            {
                "_id": 1,
            },
        ).sort("_id", 1)
    )
    if not matches:
        raise RuntimeError(
            f"No record found for '{target_name}' in '{source_collection}'. Aborting."
        )

    keep_doc = matches[0]
    keep_id = keep_doc["_id"]

    ids_to_archive = [
        doc["_id"]
        for doc in scholars.find({"_id": {"$ne": keep_id}}, {"_id": 1})
    ]
    to_archive_count = len(ids_to_archive)

    print(f"[MongoDB] Database: {db_name}")
    print(f"[MongoDB] Source: {source_collection}")
    print(f"[MongoDB] Archive: {archive_collection}")
    print(f"[State] Total in source before: {total_before}")
    print(f"[State] Matching '{target_name}' records: {len(matches)}")
    print(f"[State] Keep _id: {keep_id}")
    print(f"[Plan] Archive + remove from source: {to_archive_count}")

    if not execute:
        print("[Dry Run] No data changed. Re-run with --yes to execute.")
        return

    copied = 0
    for id_batch in chunked(ids_to_archive, batch_size):
        docs = list(scholars.find({"_id": {"$in": id_batch}}))
        if not docs:
            continue
        ops = [ReplaceOne({"_id": doc["_id"]}, doc, upsert=True) for doc in docs]
        archived.bulk_write(ops, ordered=False)
        copied += len(docs)

    if ids_to_archive:
        delete_result = scholars.delete_many({"_id": {"$in": ids_to_archive}})
        deleted = delete_result.deleted_count
    else:
        deleted = 0

    archived_present = (
        archived.count_documents({"_id": {"$in": ids_to_archive}})
        if ids_to_archive
        else 0
    )
    remaining_total = scholars.count_documents({})
    remaining_target = scholars.count_documents(target_filter)
    keep_exists = scholars.count_documents({"_id": keep_id})

    if ids_to_archive and archived_present != len(ids_to_archive):
        raise RuntimeError(
            "Verification failed: not all expected records are present in archive collection."
        )
    if remaining_total != 1 or keep_exists != 1 or remaining_target != 1:
        raise RuntimeError(
            "Verification failed: source collection does not contain exactly one target record."
        )

    print(f"[Result] Copied to archive: {copied}")
    print(f"[Result] Deleted from source: {deleted}")
    print(f"[Result] Remaining in source: {remaining_total}")
    print(f"[Result] Remaining '{target_name}' records: {remaining_target}")
    print("[Done] Migration completed successfully.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Archive all scholars except one target record."
    )
    parser.add_argument(
        "--name",
        default="Craig Johnson",
        help="Full name to keep in the source collection (default: Craig Johnson).",
    )
    parser.add_argument(
        "--source-collection",
        default="scholars",
        help="Source collection name (default: scholars).",
    )
    parser.add_argument(
        "--archive-collection",
        default="archived_scholars",
        help="Archive collection name (default: archived_scholars).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Batch size for copy operations (default: 500).",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Execute the migration. Without this flag the script runs in dry-run mode.",
    )

    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than 0.")

    archive_scholars_except_one(
        target_name=args.name,
        source_collection=args.source_collection,
        archive_collection=args.archive_collection,
        batch_size=args.batch_size,
        execute=args.yes,
    )


if __name__ == "__main__":
    main()
