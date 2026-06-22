"""
Clear all embeddings from the configured Pinecone index (ASCII-only output).

This is a small helper for Windows terminals that choke on non-ASCII characters
in print statements used by other scripts.

It reads the index name and dimension from config/pinecone_config.py and uses
api.services.vector_db.get_vector_db to get a handle to the index, then calls
delete_all=True on the default namespace.

Usage:
    python clear_pinecone_index_ascii.py
"""

from dotenv import load_dotenv

load_dotenv()


def clear_pinecone_index() -> None:
    from config.pinecone_config import INDEX_NAME, INDEX_DIMENSION
    from api.services.vector_db import get_vector_db

    print("=" * 60)
    print("Pinecone Index Clear (ASCII)")
    print("=" * 60)
    print(f"Index: {INDEX_NAME}")
    print("=" * 60)

    try:
        vector_db = get_vector_db(index_name=INDEX_NAME, dimension=INDEX_DIMENSION)
        stats_before = vector_db.index.describe_index_stats()
        total_before = stats_before.total_vector_count
        print(f"[Before] Total vectors: {total_before}")

        if total_before == 0:
            print("[Info] Index is already empty.")
            return

        print("[Deleting] Deleting all vectors (default namespace)...")
        vector_db.index.delete(delete_all=True)

        stats_after = vector_db.index.describe_index_stats()
        total_after = stats_after.total_vector_count
        print(f"[After] Total vectors: {total_after}")

        if total_after == 0:
            print("[Result] Successfully cleared index.")
        else:
            print("[Warning] Some vectors may remain; check index stats.")
    except Exception as e:
        print(f"[Error] Failed to clear Pinecone index: {e}")


if __name__ == "__main__":
    clear_pinecone_index()

