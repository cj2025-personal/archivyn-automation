"""
Clear MongoDB 'scholars' collection used by the RAG pipeline.

Uses the same database resolution logic as MongoDBScholarSync:
- Reads MONGODB_URI from environment/.env
- Infers database name from the URI path, or falls back to 'ngo_profiles'
- Drops or empties the 'scholars' collection.

Usage:
    python clear_mongodb_scholars.py
    python clear_mongodb_scholars.py --collection legend_scholars
"""

import os
from dotenv import load_dotenv

load_dotenv()

from config.mongodb_utils import create_mongo_client, resolve_mongo_db_name


def clear_scholars_collection(collection_name: str = "scholars", drop: bool = False) -> None:
    try:
        import pymongo
    except ImportError:
        raise ImportError("pymongo not installed. Install with: pip install pymongo")

    mongodb_uri = os.getenv("MONGODB_URI")
    if not mongodb_uri:
        raise ValueError("MONGODB_URI not found in environment variables")

    db_name = resolve_mongo_db_name(mongodb_uri)
    client = create_mongo_client(mongodb_uri)

    db = client[db_name]
    coll = db[collection_name]

    if drop:
        print(f"[MongoDB] Dropping collection '{db_name}.{collection_name}'...")
        coll.drop()
    else:
        print(f"[MongoDB] Deleting all documents from '{db_name}.{collection_name}'...")
        coll.delete_many({})

    count = coll.count_documents({})
    print(f"[MongoDB] Remaining documents in '{collection_name}': {count}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Clear a MongoDB collection (default: scholars)")
    parser.add_argument("--collection", type=str, default="scholars", help="Collection name to clear")
    parser.add_argument("--drop", action="store_true", help="Drop the collection instead of deleting documents")
    args = parser.parse_args()

    clear_scholars_collection(collection_name=args.collection, drop=args.drop)
