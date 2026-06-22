"""
Clear all documents from legend_scholar_daily_stories.

Usage:
    python clear_legend_scholar_daily_stories.py
"""

import os

from dotenv import load_dotenv

from config.mongodb_utils import create_mongo_client


# Environment is loaded in-script so the console command stays one line.
load_dotenv(dotenv_path=".env")

MONGODB_URI = os.getenv("MONGODB_URI", "")
DB_NAME = os.getenv("MONGODB_DB", "ngo_profiles")
COLLECTION_NAME = "legend_scholar_daily_stories"


def main() -> int:
    if not MONGODB_URI:
        raise ValueError("MONGODB_URI not found. Set it in .env")

    client = create_mongo_client(MONGODB_URI)
    db = client[DB_NAME]
    coll = db[COLLECTION_NAME]

    before = coll.count_documents({})
    result = coll.delete_many({})
    after = coll.count_documents({})

    print(f"collection={COLLECTION_NAME}")
    print(f"before={before}")
    print(f"deleted={result.deleted_count}")
    print(f"after={after}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
