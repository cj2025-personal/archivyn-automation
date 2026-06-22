"""
Repair the top-level ``professor_name`` field on docs in the
``legend_scholars`` collection where it disagrees with the LLM-extracted
``name.full`` sub-field.

These are docs that the original (pre-fix) pipeline created with broken
``professor_name`` values like ``"All Over Albany"`` (a blog name),
``"academic.oup.com"`` (a domain), or a video title — while the LLM
summary step downstream still managed to capture the correct
``name.full``. We trust ``name.full`` here.

Usage:
    python fix_legend_scholar_names.py             # dry run, no writes
    python fix_legend_scholar_names.py --apply     # actually update Mongo

Optionally also propagate the corrected name into Pinecone vectors:
    python fix_legend_scholar_names.py --apply --update-pinecone
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))

from config.mongodb_utils import create_mongo_client, resolve_mongo_db_name


def repair(apply_changes: bool, update_pinecone: bool) -> int:
    load_dotenv()
    uri = os.getenv("MONGODB_URI")
    if not uri:
        print("MONGODB_URI not set in environment.", file=sys.stderr)
        return 2

    client = create_mongo_client(uri)
    db = client[resolve_mongo_db_name(uri)]
    coll = db.legend_scholars

    to_fix = []
    for d in coll.find({}, {"_id": 1, "professor_name": 1, "name": 1}):
        full = ((d.get("name") or {}).get("full") or "").strip()
        pn = (d.get("professor_name") or "").strip()
        if full and full != pn:
            to_fix.append((d["_id"], pn, full))

    if not to_fix:
        print("Nothing to fix — every doc's professor_name already matches name.full.")
        return 0

    print(f"Will repair {len(to_fix)} doc(s):")
    for _id, old, new in to_fix:
        print(f"  {_id}  {old!r}  ->  {new!r}")

    if not apply_changes:
        print("\nDry run — pass --apply to perform writes.")
        return 0

    for _id, _old, new in to_fix:
        coll.update_one({"_id": _id}, {"$set": {"professor_name": new}})
    print(f"\nUpdated {len(to_fix)} legend_scholars docs.")

    if update_pinecone:
        try:
            from api.services.vector_db import get_vector_db
            from config.pinecone_config import INDEX_NAME, INDEX_DIMENSION
        except Exception as exc:
            print(f"Pinecone update skipped (import failed): {exc}")
            return 0
        index = get_vector_db(index_name=INDEX_NAME, dimension=INDEX_DIMENSION)
        for _id, _old, new in to_fix:
            try:
                # Update metadata on every vector belonging to this profile.
                # We use the profile_id == _id convention used elsewhere
                # in the codebase.
                index.update_metadata_for_profile(
                    profile_id=str(_id),
                    metadata_patch={"professor_name": new},
                )
                print(f"  pinecone meta updated for {_id}: professor_name -> {new!r}")
            except AttributeError:
                # Fallback: some VectorDB wrappers don't expose this helper.
                print(
                    f"  pinecone meta update method not available; "
                    f"re-run upload for {_id} if vectors must reflect the new name."
                )
                break
            except Exception as exc:
                print(f"  pinecone meta update failed for {_id}: {exc}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Perform writes (default: dry run).")
    parser.add_argument(
        "--update-pinecone",
        action="store_true",
        help="Also update professor_name in Pinecone vector metadata.",
    )
    args = parser.parse_args()
    return repair(apply_changes=args.apply, update_pinecone=args.update_pinecone)


if __name__ == "__main__":
    raise SystemExit(main())
