"""
Propagate the on-disk false-positive cleanup into MongoDB + Pinecone.

After `clean_false_positive_enrichments.py` stripped bad sources from
enrichment.json files, the MongoDB `enrichment_raw` docs and Pinecone
vectors for those profiles still contain the old polluted data. This
script fixes that — but only for profiles actually affected (identified
by the "Stripped by cleanup" marker in their enrichment.json).

Actions (idempotent):
  1. MongoDB: upsert the cleaned enrichment.json content into
     `enrichment_raw`, replacing sources/summary/confidence.
  2. Pinecone: delete all vectors with metadata.profile_id matching an
     affected profile. Next enrichment pipeline run will rebuild chunks
     from the cleaned enrichment_text.txt.

Only bad profiles are touched. Good profiles are untouched.

Usage:
    python sync_cleanup_to_db.py              # dry-run
    python sync_cleanup_to_db.py --apply      # actually modify DB / Pinecone
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

STRIPPED_MARK = "Stripped by cleanup"
ROOT = Path("output/osu_faculty_run/profiles")


def find_affected_profiles():
    """Return list of (profile_id, enrichment_json_path, raw_text_path,
    stripped_sources)."""
    affected = []
    for pdir in sorted(ROOT.iterdir()):
        if not pdir.is_dir():
            continue
        enr = pdir / "enrichment.json"
        if not enr.exists():
            continue
        try:
            doc = json.loads(enr.read_text(encoding="utf-8"))
        except Exception:
            continue
        stripped = [
            src for src, info in (doc.get("sources") or {}).items()
            if STRIPPED_MARK in (info.get("error") or "")
        ]
        if stripped:
            affected.append({
                "profile_id": doc.get("profile_id", pdir.name),
                "name": doc.get("professor_name", ""),
                "enrichment_path": enr,
                "text_path": pdir / "enrichment_text.txt",
                "stripped_sources": stripped,
                "doc": doc,
            })
    return affected


def update_mongo(affected, apply: bool):
    from config.mongodb_utils import create_mongo_client, resolve_mongo_db_name

    mongodb_uri = os.getenv("MONGODB_URI")
    if not mongodb_uri:
        print("MONGODB_URI not set — skipping Mongo update")
        return 0

    print(f"\n[Mongo] Connecting ...")
    client = create_mongo_client(mongodb_uri)
    db = client[resolve_mongo_db_name(mongodb_uri)]
    coll = db.enrichment_raw

    updated = 0
    not_in_mongo = 0
    for item in affected:
        pid = item["profile_id"]
        existing = coll.find_one({"profile_id": pid})
        if not existing:
            not_in_mongo += 1
            continue
        doc = item["doc"]
        raw_text = ""
        if item["text_path"].exists():
            raw_text = item["text_path"].read_text(encoding="utf-8")
        patch = {
            "confidence": doc.get("confidence", {}),
            "summary": doc.get("summary", {}),
            "sources": doc.get("sources", {}),
            "raw_text": raw_text,
            "raw_text_length": len(raw_text),
            "cleanup_applied": True,
            "cleanup_stripped": item["stripped_sources"],
        }
        if apply:
            coll.update_one({"profile_id": pid}, {"$set": patch})
        updated += 1

    print(f"[Mongo] {'Updated' if apply else 'Would update'}: {updated} docs")
    print(f"[Mongo] Skipped (not yet in Mongo): {not_in_mongo}")
    client.close()
    return updated


def delete_pinecone_vectors(affected, apply: bool):
    from api.services.vector_db import VectorDBService

    index_name = os.getenv("PINECONE_INDEX_NAME", "ngo-profiles")
    # Pipeline uses text-embedding-3-small (1536); VectorDBService default is
    # 384. Override to match what the pipeline actually writes.
    dimension = int(os.getenv("PINECONE_DIMENSION", "1536"))

    print(f"\n[Pinecone] Connecting to index={index_name} dim={dimension} ...")
    try:
        vdb = VectorDBService(index_name=index_name, dimension=dimension)
    except Exception as e:
        print(f"[Pinecone] Connection failed: {e}")
        return 0

    total_ids = 0
    per_profile_counts = []

    import random

    for item in affected:
        pid = item["profile_id"]
        # Query to find vectors for this profile (up to 2000 — more than a
        # profile would realistically have)
        random_vec = [random.uniform(-0.01, 0.01) for _ in range(dimension)]
        try:
            res = vdb.index.query(
                vector=random_vec,
                top_k=2000,
                include_metadata=False,
                filter={"profile_id": {"$eq": pid}},
            )
            ids = [m.id for m in res.matches]
        except Exception as e:
            print(f"[Pinecone] Query failed for {pid}: {e}")
            ids = []

        if ids:
            total_ids += len(ids)
            per_profile_counts.append((pid, item["name"], len(ids)))
            if apply:
                try:
                    # Pinecone delete by ID list, chunked
                    for chunk in [ids[i:i+1000] for i in range(0, len(ids), 1000)]:
                        vdb.index.delete(ids=chunk)
                except Exception as e:
                    print(f"[Pinecone] Delete failed for {pid}: {e}")

    print(f"[Pinecone] {'Deleted' if apply else 'Would delete'}: {total_ids} vectors "
          f"across {len(per_profile_counts)} profiles")
    # Show top 10 affected
    per_profile_counts.sort(key=lambda x: -x[2])
    if per_profile_counts:
        print("[Pinecone] Top 10 profiles by vector count:")
        for pid, name, cnt in per_profile_counts[:10]:
            print(f"  {name:<30}  {cnt:>4} vectors  id={pid[:8]}")
    return total_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--skip-mongo", action="store_true")
    parser.add_argument("--skip-pinecone", action="store_true")
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")

    print("[1/3] Scanning disk for cleaned-but-not-synced profiles ...")
    affected = find_affected_profiles()
    print(f"      Found {len(affected)} profiles with stripped sources")
    if not affected:
        print("Nothing to do.")
        return

    # Summary by source
    from collections import Counter
    by_src = Counter()
    for a in affected:
        for s in a["stripped_sources"]:
            by_src[s] += 1
    print("      Breakdown by source:")
    for s, c in by_src.most_common():
        print(f"        {s:<15} {c:>4} profiles")

    if not args.skip_mongo:
        print(f"\n[2/3] MongoDB sync ({'APPLY' if args.apply else 'dry-run'}) ...")
        update_mongo(affected, apply=args.apply)

    if not args.skip_pinecone:
        print(f"\n[3/3] Pinecone vector cleanup ({'APPLY' if args.apply else 'dry-run'}) ...")
        delete_pinecone_vectors(affected, apply=args.apply)

    print(f"\n{'✅ APPLIED' if args.apply else '🔍 DRY RUN COMPLETE — re-run with --apply'}")


if __name__ == "__main__":
    main()
