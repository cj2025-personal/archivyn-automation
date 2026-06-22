"""
Sync data *from* the vector DB (Pinecone) *into* MongoDB.

This script is intentionally separate from `sync_profiles_to_mongodb.py`:
- `sync_profiles_to_mongodb.py` builds a curated `scholars` document with LLM summaries.
- This script mirrors raw Pinecone chunk metadata/text into MongoDB for auditing, backups,
  and downstream processing without re-querying Pinecone.

Default behavior:
- Reads Pinecone vectors (preferring IDs with prefixes `chunk_` and `profile_`)
- Fetches metadata (and drops embedding values)
- Upserts each chunk into MongoDB collection `vector_chunks`

Usage:
    python sync_vectordb_to_mongodb.py
    python sync_vectordb_to_mongodb.py --profile-id <uuid>
    python sync_vectordb_to_mongodb.py --namespace "" --prefix chunk_ --prefix profile_
    python sync_vectordb_to_mongodb.py --dry-run
"""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from dotenv import load_dotenv

from api.services.vector_db import get_vector_db
from config.mongodb_utils import create_mongo_client
from config.pinecone_config import INDEX_DIMENSION, INDEX_NAME


try:
    from pymongo import MongoClient, UpdateOne
    from pymongo.errors import ConnectionFailure
except ImportError as e:  # pragma: no cover
    raise ImportError("pymongo not installed. Install with: pip install pymongo") from e


@dataclass(frozen=True)
class MongoTarget:
    db_name: str
    collection_name: str


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_mongo_db_name(mongodb_uri: str, override_db: Optional[str] = None) -> str:
    if override_db:
        return override_db
    try:
        from urllib.parse import urlparse

        parsed = urlparse(mongodb_uri)
        db_name = parsed.path.lstrip("/").split("?")[0] if parsed.path else "ngo_profiles"
        return db_name or "ngo_profiles"
    except Exception:
        return "ngo_profiles"


def _iter_pinecone_ids(index: Any, *, namespace: str, prefix: Optional[str], page_limit: int) -> Iterable[str]:
    token: Optional[str] = None
    while True:
        resp = index.list_paginated(
            prefix=prefix,
            limit=page_limit,
            pagination_token=token,
            namespace=namespace if namespace else None,
        )
        for item in resp.vectors or []:
            if item and getattr(item, "id", None):
                yield item.id
        token = resp.pagination.next if resp.pagination else None
        if not token:
            break


def _batched(items: Iterable[str], batch_size: int) -> Iterable[List[str]]:
    batch: List[str] = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _get_profile_chunks_via_query(index: Any, *, profile_id: str, namespace: str) -> List[Tuple[str, Dict[str, Any]]]:
    import random

    random_vector = [random.uniform(-0.01, 0.01) for _ in range(INDEX_DIMENSION)]
    resp = index.query(
        vector=random_vector,
        top_k=1000,
        include_metadata=True,
        filter={"professor_id": profile_id},
        namespace=namespace if namespace else None,
    )
    out: List[Tuple[str, Dict[str, Any]]] = []
    for match in resp.matches or []:
        metadata = match.metadata or {}
        out.append((match.id, dict(metadata)))
    return out


def _normalize_chunk_doc(vector_id: str, metadata: Dict[str, Any], *, namespace: str) -> Dict[str, Any]:
    profile_id = metadata.get("profile_id") or metadata.get("professor_id") or ""
    professor_id = metadata.get("professor_id") or metadata.get("profile_id") or ""
    professor_name = metadata.get("professor_name") or ""
    section = metadata.get("section") or "Unknown"
    order = metadata.get("order") if metadata.get("order") is not None else 0
    chunk_id = metadata.get("chunk_id") or ""
    text = metadata.get("text") or ""
    content_type = metadata.get("content_type") or ""

    # Keep the original metadata for traceability (excluding large fields we already lift).
    metadata_copy = dict(metadata)

    return {
        "_id": vector_id,
        "profile_id": profile_id,
        "professor_id": professor_id,
        "professor_name": professor_name,
        "section": section,
        "order": order,
        "chunk_id": chunk_id,
        "text": text,
        "content_type": content_type,
        "pinecone": {
            "index": INDEX_NAME,
            "namespace": namespace,
        },
        "metadata": metadata_copy,
        "synced_at": _utc_now_iso(),
    }


def _select_best_professor_name(chunk_docs: List[Dict[str, Any]]) -> str:
    names = [d.get("professor_name") for d in chunk_docs if d.get("professor_name")]
    if not names:
        return "Unknown"
    return Counter(names).most_common(1)[0][0]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Sync Pinecone vectors into MongoDB")
    parser.add_argument("--profile-id", type=str, default=None, help="Sync only one profile (professor_id/profile_id)")
    parser.add_argument("--namespace", type=str, default="", help="Pinecone namespace (default: empty)")
    parser.add_argument(
        "--prefix",
        action="append",
        default=[],
        help="ID prefix to list (repeatable). Default: chunk_ and profile_",
    )
    parser.add_argument(
        "--page-limit",
        type=int,
        default=100,
        help="Pinecone list page size (1-100, default: 100)",
    )
    parser.add_argument("--fetch-batch-size", type=int, default=200, help="Pinecone fetch batch size (default: 200)")
    parser.add_argument("--mongo-db", type=str, default=None, help="Override MongoDB database name")
    parser.add_argument("--mongo-collection", type=str, default="vector_chunks", help="Target collection (default: vector_chunks)")
    parser.add_argument("--dry-run", action="store_true", help="Do not write to MongoDB; only report counts")
    parser.add_argument(
        "--only-content-type",
        type=str,
        default="profile_chunk",
        help="Only sync vectors with this metadata.content_type (default: profile_chunk). Use empty to disable filtering.",
    )
    args = parser.parse_args()

    load_dotenv()

    if args.page_limit <= 0:
        raise ValueError("--page-limit must be > 0")
    if args.page_limit > 100:
        print(f"[Args] page-limit {args.page_limit} > 100; clamping to 100 (Pinecone API limit).")
        args.page_limit = 100

    mongodb_uri = os.getenv("MONGODB_URI")
    if not mongodb_uri:
        raise ValueError("MONGODB_URI not found in environment variables")

    db_name = _resolve_mongo_db_name(mongodb_uri, override_db=args.mongo_db)
    target = MongoTarget(db_name=db_name, collection_name=args.mongo_collection)

    try:
        mongo_client = create_mongo_client(mongodb_uri)
    except Exception as e:
        raise ConnectionError(f"Failed to connect to MongoDB: {e}") from e

    db = mongo_client[target.db_name]
    chunks_collection = db[target.collection_name]

    vector_db = get_vector_db(index_name=INDEX_NAME, dimension=INDEX_DIMENSION)
    index = vector_db.index

    namespace = args.namespace or ""

    if not args.prefix:
        prefixes = ["chunk_", "profile_"]
    else:
        prefixes = list(args.prefix)

    print("=" * 60)
    print("VectorDB -> MongoDB Sync")
    print("=" * 60)
    print(f"[Pinecone] Index: {INDEX_NAME} (dimension {INDEX_DIMENSION})")
    print(f"[Pinecone] Namespace: '{namespace or 'default'}'")
    print(f"[MongoDB] Database: {target.db_name}")
    print(f"[MongoDB] Collection: {target.collection_name}")
    if args.profile_id:
        print(f"[Mode] Single profile: {args.profile_id}")
    else:
        print(f"[Mode] Full scan (prefixes: {', '.join(prefixes)})")
    if args.dry_run:
        print("[Mode] Dry run (no writes)")

    # Ensure minimal indexes for common queries.
    if not args.dry_run:
        chunks_collection.create_index("profile_id")
        chunks_collection.create_index("professor_id")
        chunks_collection.create_index([("profile_id", 1), ("section", 1), ("order", 1)])

    docs_to_write: List[Dict[str, Any]] = []
    total_vectors_seen = 0
    total_docs_kept = 0

    if args.profile_id:
        items = _get_profile_chunks_via_query(index, profile_id=args.profile_id, namespace=namespace)
        total_vectors_seen = len(items)
        for vector_id, metadata in items:
            if args.only_content_type and metadata.get("content_type") != args.only_content_type:
                continue
            docs_to_write.append(_normalize_chunk_doc(vector_id, metadata, namespace=namespace))
        total_docs_kept = len(docs_to_write)
    else:
        seen_ids: Set[str] = set()
        all_ids: List[str] = []

        print("[Pinecone] Listing vector IDs...")
        for prefix in prefixes:
            for vector_id in _iter_pinecone_ids(index, namespace=namespace, prefix=prefix, page_limit=args.page_limit):
                if vector_id in seen_ids:
                    continue
                seen_ids.add(vector_id)
                all_ids.append(vector_id)

        print(f"[Pinecone] Found {len(all_ids)} candidate vector IDs")

        for id_batch in _batched(all_ids, args.fetch_batch_size):
            fetched = index.fetch(ids=id_batch, namespace=namespace if namespace else None)
            vectors = fetched.vectors or {}
            total_vectors_seen += len(id_batch)

            for vector_id, vec in vectors.items():
                metadata = getattr(vec, "metadata", None) or {}
                if not isinstance(metadata, dict):
                    continue
                if args.only_content_type and metadata.get("content_type") != args.only_content_type:
                    continue
                docs_to_write.append(_normalize_chunk_doc(vector_id, metadata, namespace=namespace))

            if len(docs_to_write) >= 2000:
                if not args.dry_run:
                    _bulk_upsert(chunks_collection, docs_to_write)
                total_docs_kept += len(docs_to_write)
                docs_to_write = []

        if docs_to_write:
            if not args.dry_run:
                _bulk_upsert(chunks_collection, docs_to_write)
            total_docs_kept += len(docs_to_write)

    if args.profile_id and docs_to_write and not args.dry_run:
        _bulk_upsert(chunks_collection, docs_to_write)

    print("-" * 60)
    print(f"[Summary] Pinecone vectors seen: {total_vectors_seen}")
    print(f"[Summary] MongoDB docs upserted: {total_docs_kept if not args.profile_id else len(docs_to_write)}")
    if args.profile_id and docs_to_write:
        print(f"[Summary] professor_name (most common): {_select_best_professor_name(docs_to_write)}")
    if not args.dry_run:
        count = chunks_collection.count_documents({})
        print(f"[MongoDB] Total docs in {target.db_name}.{target.collection_name}: {count}")


def _bulk_upsert(collection: Any, docs: List[Dict[str, Any]]) -> None:
    ops = [UpdateOne({"_id": d["_id"]}, {"$set": d}, upsert=True) for d in docs]
    if not ops:
        return
    collection.bulk_write(ops, ordered=False)


if __name__ == "__main__":
    main()
