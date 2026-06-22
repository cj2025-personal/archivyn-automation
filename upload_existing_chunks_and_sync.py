"""
One-off script to:
1) Upload existing chunked profiles under output/chunked_profiles to Pinecone
2) Sync just those profiles into MongoDB.

This version:
- Reads professor_name from each chunk (as written by backfill_chunk_names.py),
  or falls back to profile JSON under output/profiles.
- Uses only ASCII output for Windows terminal compatibility.

Usage:
    python upload_existing_chunks_and_sync.py
"""

from pathlib import Path
from typing import Dict, Iterable, List, Any, Set

from dotenv import load_dotenv

load_dotenv()

from sync_profiles_to_mongodb import MongoDBScholarSync


def load_chunks_with_names(
    chunks_root: Path = Path("output/chunked_profiles"),
    profiles_root: Path = Path("output/profiles"),
) -> List[Dict[str, Any]]:
    """Load chunks from disk and ensure professor_name is present on each."""
    if not chunks_root.exists():
        print(f"[Setup] Chunks root not found: {chunks_root}")
        return []

    # Map profile_id -> name from profile JSON
    profile_names: Dict[str, str] = {}
    if profiles_root.exists():
        for profile_dir in profiles_root.iterdir():
            if not profile_dir.is_dir():
                continue
            profile_id = profile_dir.name
            profile_json = profile_dir / f"{profile_id}.json"
            if not profile_json.exists():
                continue
            try:
                import json

                with profile_json.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                name = data.get("name") or ""
                if name:
                    profile_names[profile_id] = name
            except Exception as e:
                print(f"[Warn] Failed to read profile JSON for {profile_id}: {e}")

    all_chunks: List[Dict[str, Any]] = []

    for chunk_file in chunks_root.glob("*/chunks.json"):
        try:
            import json

            with chunk_file.open("r", encoding="utf-8") as f:
                data = json.load(f)

            profile_id = data.get("profile_id") or chunk_file.parent.name
            sections = data.get("sections") or {}

            for section_name, section_chunks in sections.items():
                if not isinstance(section_chunks, list):
                    continue
                for chunk in section_chunks:
                    if not isinstance(chunk, dict):
                        continue

                    prof_name = (
                        chunk.get("professor_name")
                        or profile_names.get(profile_id, "")
                    )

                    all_chunks.append(
                        {
                            "profile_id": profile_id,
                            "professor_id": profile_id,
                            "professor_name": prof_name,
                            "section": chunk.get("section", section_name),
                            "chunk_id": chunk.get("chunk_id", ""),
                            "order": chunk.get("order", 0),
                            "text": chunk.get("text", ""),
                        }
                    )
        except Exception as e:
            print(f"[Warn] Failed to load chunks from {chunk_file}: {e}")

    print(
        f"[Loading] Loaded {len(all_chunks)} chunks from {len(list(chunks_root.glob('*/chunks.json')))} files"
    )
    return all_chunks


def _chunked(items: List[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(items), size):
        yield items[i:i + size]


def _get_existing_profile_ids(
    scholars_collection,
    profile_ids: List[str],
    batch_size: int = 500,
) -> Set[str]:
    existing: Set[str] = set()
    if not profile_ids:
        return existing

    for batch in _chunked(profile_ids, batch_size):
        cursor = scholars_collection.find(
            {"profile_id": {"$in": batch}},
            {"profile_id": 1, "_id": 1},
        )
        for doc in cursor:
            profile_id = doc.get("profile_id") or doc.get("_id")
            if profile_id:
                existing.add(str(profile_id))

    return existing


def upload_chunks_to_pinecone(chunks: List[Dict[str, Any]], batch_size: int = 100) -> None:
    """Upload chunks with professor_name to Pinecone using embeddings service."""
    from api.services.vector_db import get_vector_db
    from api.services.embeddings import get_embeddings_service
    from config.pinecone_config import INDEX_NAME, INDEX_DIMENSION

    print("\n[Setup] Initializing services...")
    print(f"[Setup] Index: {INDEX_NAME}")
    print(f"[Setup] Dimension: {INDEX_DIMENSION}")
    print("[Setup] Model: text-embedding-3-small")

    try:
        vector_db = get_vector_db(index_name=INDEX_NAME, dimension=INDEX_DIMENSION)
        embeddings_service = get_embeddings_service()
        print(f"[Setup] Connected to Pinecone index: {INDEX_NAME}")
        print(f"[Setup] Initialized embeddings service: {embeddings_service.model_name}")
    except Exception as e:
        print(f"[Setup] Error initializing services: {e}")
        raise

    valid_chunks = [c for c in chunks if (c.get("text") or "").strip()]
    print(
        f"\n[Processing] Processing {len(valid_chunks)} valid chunks "
        f"(skipped {len(chunks) - len(valid_chunks)} empty chunks)"
    )
    if not valid_chunks:
        print("[Processing] No valid chunks to process!")
        return

    from config.pinecone_config import INDEX_DIMENSION as DIM

    total_batches = (len(valid_chunks) + batch_size - 1) // batch_size
    successful = 0
    failed = 0

    print(f"\n[Processing] Processing {len(valid_chunks)} chunks in {total_batches} batches...")

    for batch_idx in range(0, len(valid_chunks), batch_size):
        batch = valid_chunks[batch_idx : batch_idx + batch_size]
        batch_num = batch_idx // batch_size + 1
        print(f"\n[Batch {batch_num}/{total_batches}] Processing {len(batch)} chunks...")

        try:
            texts = [c["text"] for c in batch]
            print(f"[Batch {batch_num}] Generating embeddings...")
            embeddings = embeddings_service.embed_batch(texts, batch_size=len(texts))

            if len(embeddings) != len(batch):
                print(
                    f"[Batch {batch_num}] Warning: embeddings {len(embeddings)} "
                    f"!= chunks {len(batch)}"
                )

            vectors = []
            for i, chunk in enumerate(batch):
                if i >= len(embeddings):
                    failed += 1
                    continue
                emb = embeddings[i]
                if len(emb) != DIM or all(v == 0.0 for v in emb):
                    failed += 1
                    continue

                profile_id = chunk["profile_id"]
                chunk_id = chunk.get("chunk_id") or ""
                vector_id = f"chunk_{chunk_id}" if chunk_id else f"profile_{profile_id}_{i}"

                metadata = {
                    "profile_id": profile_id,
                    "professor_id": chunk.get("professor_id", profile_id),
                    "section": chunk.get("section"),
                    "chunk_id": chunk_id,
                    "order": chunk.get("order", 0),
                    "text": chunk.get("text", ""),
                    "content_type": "profile_chunk",
                }
                if chunk.get("professor_name"):
                    metadata["professor_name"] = chunk["professor_name"]

                vectors.append({"id": vector_id, "values": emb, "metadata": metadata})

            if vectors:
                print(
                    f"[Batch {batch_num}] Uploading {len(vectors)} vectors to Pinecone..."
                )
                pinecone_batch_size = 100
                for j in range(0, len(vectors), pinecone_batch_size):
                    sub = vectors[j : j + pinecone_batch_size]
                    vector_db.index.upsert(vectors=sub)
                successful += len(vectors)
            else:
                print(f"[Batch {batch_num}] No valid vectors to upload")

        except Exception as e:
            print(f"[Batch {batch_num}] Error: {e}")
            failed += len(batch)

    print("\n" + "=" * 60)
    print("[Summary] Upload Complete")
    print(f"[Summary] Successful uploads: {successful}")
    print(f"[Summary] Failed uploads: {failed}")
    print(f"[Summary] Total chunks processed: {len(valid_chunks)}")
    print("=" * 60)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Upload existing chunks to Pinecone and/or sync them to MongoDB"
    )
    parser.add_argument(
        "--mongo-only",
        action="store_true",
        help="Skip uploading chunks to Pinecone and only run MongoDB sync",
    )
    parser.add_argument(
        "--start-from",
        type=int,
        default=1,
        help=(
            "Start MongoDB sync from this 1-based index "
            "as shown in the [MongoDB] idx/total log (e.g., 599)"
        ),
    )
    parser.add_argument(
        "--include-existing",
        action="store_true",
        help="Sync profiles even if they already exist in MongoDB",
    )
    args = parser.parse_args()

    chunks_dir = Path("output/chunked_profiles")
    profiles_dir = Path("output/profiles")

    print("=" * 60)
    print("Upload Existing Chunks -> Pinecone -> MongoDB")
    print("=" * 60)
    print(f"[Setup] Chunks directory: {chunks_dir}")
    print(f"[Setup] Profiles directory: {profiles_dir}")

    chunks = load_chunks_with_names(chunks_dir, profiles_dir)
    if not chunks:
        print("[Setup] No chunks found; nothing to upload.")
        return

    # Build map of professor_id -> professor_name from the chunks we just processed
    profiles: Dict[str, str] = {}
    for chunk in chunks:
        prof_id = chunk.get("professor_id") or chunk.get("profile_id")
        prof_name = chunk.get("professor_name") or "Unknown"
        if prof_id and prof_id not in profiles:
            profiles[prof_id] = prof_name

    sync = None
    existing_ids: Set[str] = set()
    if not args.include_existing:
        sync = MongoDBScholarSync()
        existing_ids = _get_existing_profile_ids(sync.scholars_collection, list(profiles.keys()))
        if existing_ids:
            print(
                f"[Skip] Found {len(existing_ids)} existing profiles in MongoDB; "
                "skipping for Pinecone upload and MongoDB sync."
            )

    chunks_to_upload = chunks
    if existing_ids:
        chunks_to_upload = [c for c in chunks if c.get("profile_id") not in existing_ids]

    if args.mongo_only:
        print("\n[Stage 1] Skipping Pinecone upload (mongo-only mode).")
    else:
        if not chunks_to_upload:
            print("\n[Stage 1] Skipping Pinecone upload (no new profiles to upload).")
        else:
            print("\n[Stage 1] Uploading chunks to Pinecone...")
            upload_chunks_to_pinecone(chunks_to_upload, batch_size=50)

    if not profiles:
        print("\n[Stage 2] No professor IDs found in chunks; skipping MongoDB sync.")
        return

    print("\n[Stage 2] Syncing profiles to MongoDB...")
    if sync is None:
        sync = MongoDBScholarSync()
    sync.create_indexes()

    profile_items = list(profiles.items())

    if existing_ids:
        profile_items = [(pid, name) for pid, name in profile_items if pid not in existing_ids]

    total = len(profile_items)
    success = 0
    failed = 0
    failed_profiles: List[Dict[str, str]] = []

    start_from = max(args.start_from, 1)

    for idx, (prof_id, prof_name) in enumerate(profile_items, start=1):
        if idx < start_from:
            continue

        print(f"\n[MongoDB] {idx}/{total}: {prof_name} ({prof_id})")
        if sync.sync_profile(prof_id, prof_name):
            success += 1
            print("  [MongoDB] Synced successfully")
        else:
            failed += 1
            print("  [MongoDB] Failed to sync")
            failed_profiles.append(
                {"professor_id": prof_id, "professor_name": prof_name}
            )

    print("\n" + "=" * 60)
    print("[Summary] MongoDB Sync (Existing Chunks)")
    print(f"[Summary] Successful: {success}")
    print(f"[Summary] Failed: {failed}")
    print(f"[Summary] Total attempted: {total}")
    if failed_profiles:
        print("\n[Summary] Failed profile IDs:")
        for item in failed_profiles:
            print(f"  - {item['professor_name']} ({item['professor_id']})")

        # Optionally write failures to a small JSON file for later inspection
        try:
            import json

            output_path = Path("output") / "mongo_sync_failed_profiles.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(failed_profiles, f, ensure_ascii=False, indent=2)
            print(f"\n[Summary] Wrote failed profile list to: {output_path}")
        except Exception as e:
            print(f"[Summary] Warning: Could not write failed profile list: {e}")

    print("=" * 60)


if __name__ == "__main__":
    main()
