"""
Incremental sync — chunk + upload to Pinecone + sync to MongoDB
for a single professor immediately after enrichment.

This ensures progress is never lost: each professor is fully persisted
before moving on to the next one.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Lazy-init singletons so we don't re-connect per professor
_vector_db = None
_embeddings_service = None
_mongo_sync = None
_connections_printed = False


def _get_vector_db():
    global _vector_db, _connections_printed
    if _vector_db is None:
        from api.services.vector_db import get_vector_db
        from config.pinecone_config import INDEX_NAME, INDEX_DIMENSION
        print(f"  [Pinecone] Connecting to index: {INDEX_NAME} (dim={INDEX_DIMENSION})...")
        _vector_db = get_vector_db(index_name=INDEX_NAME, dimension=INDEX_DIMENSION)
        print(f"  [Pinecone] ✅ Connected")
    return _vector_db


def _get_embeddings_service():
    global _embeddings_service
    if _embeddings_service is None:
        from api.services.embeddings import get_embeddings_service
        print(f"  [Embeddings] Initializing embedding model...")
        _embeddings_service = get_embeddings_service()
        print(f"  [Embeddings] ✅ Model: {_embeddings_service.model_name}")
    return _embeddings_service


def _get_mongo_sync():
    global _mongo_sync
    if _mongo_sync is None:
        from sync_profiles_to_mongodb import MongoDBScholarSync
        print(f"  [MongoDB] Connecting...")
        _mongo_sync = MongoDBScholarSync()
        print(f"  [MongoDB] ✅ Connected")
    return _mongo_sync


def chunk_single_profile(
    profile_dir: Path,
    chunking_output_dir: Path,
    llm_provider: str = "openai",
    llm_model: str = "gpt-4o-mini",
) -> Optional[Dict]:
    """
    Chunk a single enriched profile. Merges original text + enrichment text,
    runs the chunking pipeline, and returns the chunks data dict.

    Returns None on failure.
    """
    from enrichment.rechunk_enriched import merge_profile_with_enrichment

    merged = merge_profile_with_enrichment(profile_dir)
    if not merged:
        print(f"  [Chunking] ⚠️ No enrichment data to merge, skipping")
        return None

    profile_id = merged["profile_id"]
    print(f"  [Chunking] Merging original ({merged['original_text_len']:,} chars) + enrichment ({merged['enrichment_text_len']:,} chars) = {merged['total_text_len']:,} chars")

    try:
        from profile_chunking_pipeline import ProfileChunkingPipeline

        pipeline = ProfileChunkingPipeline(
            output_dir=str(chunking_output_dir),
            llm_provider=llm_provider,
            llm_model=llm_model,
        )

        print(f"  [Chunking] Running LLM chunking ({llm_provider}/{llm_model})...")
        result = pipeline.process_profile(
            profile_id=profile_id,
            cleaned_text=merged["merged_text"],
        )

        # Read back the chunks.json that was just written
        chunks_file = chunking_output_dir / profile_id / "chunks.json"
        if chunks_file.exists():
            chunks_data = json.loads(chunks_file.read_text(encoding="utf-8"))
            sections = chunks_data.get("sections", {})
            total_chunks = sum(len(v) for v in sections.values())
            section_names = list(sections.keys())
            print(f"  [Chunking] ✅ {total_chunks} chunks across {len(section_names)} sections: {', '.join(section_names)}")
            return chunks_data

        print(f"  [Chunking] ⚠️ chunks.json not found after processing")
        return None

    except Exception as e:
        print(f"  [Chunking] ❌ Failed: {e}")
        logger.error("[Chunking] Failed for %s: %s", profile_id, e)
        return None


def upload_chunks_to_pinecone_single(
    chunks_data: Dict,
    professor_name: str,
    batch_size: int = 50,
) -> int:
    """
    Upload chunks for a single professor to Pinecone.
    Returns number of vectors uploaded.
    """
    from config.pinecone_config import INDEX_DIMENSION

    vector_db = _get_vector_db()
    embeddings_service = _get_embeddings_service()

    profile_id = chunks_data.get("profile_id", "")
    sections = chunks_data.get("sections", {})

    # Flatten chunks
    all_chunks = []
    for section_name, section_chunks in sections.items():
        for chunk in section_chunks:
            text = chunk.get("text", "").strip()
            if not text:
                continue
            all_chunks.append({
                "profile_id": profile_id,
                "professor_id": profile_id,
                "professor_name": professor_name,
                "section": chunk.get("section", section_name),
                "chunk_id": chunk.get("chunk_id", ""),
                "order": chunk.get("order", 0),
                "text": text,
            })

    if not all_chunks:
        print(f"  [Pinecone] ⚠️ No valid chunks to upload")
        return 0

    print(f"  [Pinecone] Generating embeddings for {len(all_chunks)} chunks...")
    uploaded = 0
    skipped = 0

    for batch_start in range(0, len(all_chunks), batch_size):
        batch = all_chunks[batch_start:batch_start + batch_size]
        batch_num = (batch_start // batch_size) + 1
        total_batches = (len(all_chunks) + batch_size - 1) // batch_size
        texts = [c["text"] for c in batch]

        try:
            embeddings = embeddings_service.embed_batch(texts, batch_size=len(texts))
        except Exception as e:
            print(f"  [Pinecone] ❌ Embedding batch {batch_num} failed: {e}")
            skipped += len(batch)
            continue

        vectors = []
        for i, chunk in enumerate(batch):
            if i >= len(embeddings):
                skipped += 1
                continue
            emb = embeddings[i]
            if len(emb) != INDEX_DIMENSION:
                skipped += 1
                continue
            if all(v == 0.0 for v in emb):
                skipped += 1
                continue

            vector_id = f"chunk_{chunk['chunk_id']}" if chunk["chunk_id"] else f"profile_{profile_id}_{batch_start + i}"
            vectors.append({
                "id": vector_id,
                "values": emb,
                "metadata": {
                    "profile_id": profile_id,
                    "professor_id": profile_id,
                    "professor_name": professor_name,
                    "section": chunk["section"],
                    "chunk_id": chunk["chunk_id"],
                    "order": chunk["order"],
                    "text": chunk["text"],
                    "content_type": "profile_chunk",
                    "enriched": True,
                },
            })

        if vectors:
            # Pinecone recommends max 100 per upsert
            for sub_start in range(0, len(vectors), 100):
                sub_batch = vectors[sub_start:sub_start + 100]
                vector_db.index.upsert(vectors=sub_batch)
            uploaded += len(vectors)

        if total_batches > 1:
            print(f"  [Pinecone] Batch {batch_num}/{total_batches}: {len(vectors)} vectors uploaded")

    if skipped:
        print(f"  [Pinecone] ✅ {uploaded} vectors uploaded ({skipped} skipped)")
    else:
        print(f"  [Pinecone] ✅ {uploaded} vectors uploaded")
    return uploaded


def sync_to_mongodb_single(
    profile_id: str,
    professor_name: str,
) -> bool:
    """
    Sync a single professor from Pinecone to MongoDB.
    Calls the existing MongoDBScholarSync.sync_profile() which:
    1. Fetches chunks from Pinecone for this professor_id
    2. Generates LLM summaries per section
    3. Upserts the scholar document to MongoDB
    """
    try:
        print(f"  [MongoDB] Syncing profile (fetching chunks from Pinecone → LLM summaries → MongoDB)...")
        mongo_sync = _get_mongo_sync()
        success = mongo_sync.sync_profile(profile_id, professor_name)
        if success:
            print(f"  [MongoDB] ✅ Profile synced to MongoDB")
        else:
            print(f"  [MongoDB] ❌ Sync failed (no chunks found in Pinecone?)")
        return success
    except Exception as e:
        print(f"  [MongoDB] ❌ Sync error: {e}")
        logger.error("[MongoDB] Sync failed for %s: %s", professor_name, e)
        return False


def incremental_sync_professor(
    profile_dir: Path,
    chunking_output_dir: Path,
    professor_name: str,
    skip_pinecone: bool = False,
    skip_mongodb: bool = False,
    llm_provider: str = "openai",
    llm_model: str = "gpt-4o-mini",
) -> Dict[str, Any]:
    """
    Full incremental sync for one professor:
    enrichment_text.txt → chunk → Pinecone → MongoDB

    Returns summary dict.
    """
    profile_id = profile_dir.name
    result = {
        "profile_id": profile_id,
        "professor_name": professor_name,
        "chunked": False,
        "chunks_count": 0,
        "pinecone_uploaded": 0,
        "mongodb_synced": False,
    }

    # Skip if no enrichment data
    if not (profile_dir / "enrichment_text.txt").exists():
        print(f"  [Sync] ⚠️ No enrichment_text.txt found, skipping chunk/upload")
        return result

    # ── Step 1: Chunk ──
    print(f"  [Step 3/5] Chunking merged text...")
    t0 = time.perf_counter()
    chunks_data = chunk_single_profile(
        profile_dir, chunking_output_dir,
        llm_provider=llm_provider, llm_model=llm_model,
    )

    if not chunks_data:
        print(f"  [Sync] ⚠️ Chunking produced no output, skipping Pinecone/MongoDB")
        return result

    total_chunks = sum(len(v) for v in chunks_data.get("sections", {}).values())
    result["chunked"] = True
    result["chunks_count"] = total_chunks
    chunk_time = time.perf_counter() - t0
    print(f"  [Step 3/5] ✅ Chunking complete: {total_chunks} chunks ({chunk_time:.1f}s)")

    # ── Step 2: Upload to Pinecone ──
    if not skip_pinecone:
        print(f"  [Step 4/5] Uploading to Pinecone...")
        t0 = time.perf_counter()
        n_uploaded = upload_chunks_to_pinecone_single(chunks_data, professor_name)
        result["pinecone_uploaded"] = n_uploaded
        pine_time = time.perf_counter() - t0
        print(f"  [Step 4/5] ✅ Pinecone upload: {n_uploaded} vectors ({pine_time:.1f}s)")
    else:
        print(f"  [Step 4/5] ⏭️ Pinecone upload skipped (--skip-pinecone)")

    # ── Step 3: Sync to MongoDB ──
    if not skip_mongodb and not skip_pinecone:
        print(f"  [Step 5/5] Syncing to MongoDB...")
        t0 = time.perf_counter()
        synced = sync_to_mongodb_single(profile_id, professor_name)
        result["mongodb_synced"] = synced
        mongo_time = time.perf_counter() - t0
        if synced:
            print(f"  [Step 5/5] ✅ MongoDB sync complete ({mongo_time:.1f}s)")
        else:
            print(f"  [Step 5/5] ❌ MongoDB sync failed ({mongo_time:.1f}s)")
    elif skip_mongodb:
        print(f"  [Step 5/5] ⏭️ MongoDB sync skipped (--skip-mongodb)")
    elif skip_pinecone:
        print(f"  [Step 5/5] ⏭️ MongoDB sync skipped (requires Pinecone)")

    return result
