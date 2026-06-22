"""
Build scholar-origins.json for the Archivyn homepage world map.

Aggregates ngo_profiles.scholars by ancestral origin (about.race) and writes
a JSON file the React component consumes:

    {
      "destination": { "lat": 39.9612, "lng": -82.9988, "label": "Ohio State University" },
      "origins": [
        { "key": "white",            "label": "Europe",         "lat": ..., "lng": ..., "count": 391 },
        { "key": "east_asian",       "label": "East Asia",      ...                                  },
        ...
      ],
      "totals": { "with_race": 564, "total_scholars": 3020, "generated_at": "..." }
    }

Usage:
    python build_scholar_origins.py
    python build_scholar_origins.py --out d:/ngo-web/frontend/public/scholar-origins.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

load_dotenv()

# Representative geographic centroids used for arc origins.
# These are not exact homelands - they are visual anchors for the
# "ancestral origin" narrative. Tune freely.
RACE_GEO = {
    "White":           {"key": "white",            "label": "Europe",          "lat": 50.45, "lng": 10.45},
    "Black":           {"key": "black",            "label": "West Africa",     "lat":  9.08, "lng":  8.67},
    "East Asian":      {"key": "east_asian",       "label": "East Asia",       "lat": 35.86, "lng": 104.19},
    "Latino_Hispanic": {"key": "latino_hispanic",  "label": "Latin America",   "lat": 23.63, "lng": -102.55},
    "Indian":          {"key": "indian",           "label": "South Asia",      "lat": 20.59, "lng":  78.96},
    "Middle Eastern":  {"key": "middle_eastern",   "label": "Middle East",     "lat": 33.22, "lng":  43.68},
    "Southeast Asian": {"key": "southeast_asian",  "label": "Southeast Asia",  "lat": 13.41, "lng": 103.86},
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("d:/ngo-web/frontend/public/scholar-origins.json"),
    )
    parser.add_argument(
        "--collection",
        default="scholars",
        help="Mongo collection to read from (default: scholars)",
    )
    args = parser.parse_args()

    try:
        from config.mongodb_utils import create_mongo_client, resolve_mongo_db_name
    except Exception as e:
        print(f"[ERROR] mongo import failed: {e}", file=sys.stderr)
        return 1

    uri = os.getenv("MONGODB_URI")
    if not uri:
        print("[ERROR] MONGODB_URI not set", file=sys.stderr)
        return 1

    client = create_mongo_client(uri)
    coll = client[resolve_mongo_db_name(uri)][args.collection]

    total_scholars = coll.count_documents({})
    pipeline = [
        {"$match": {"about.race": {"$nin": [None, ""]}}},
        {"$group": {"_id": "$about.race", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
    ]
    rows = list(coll.aggregate(pipeline))
    client.close()

    origins = []
    matched = 0
    unmatched = []
    for row in rows:
        race = row["_id"]
        count = int(row["count"])
        matched += count
        geo = RACE_GEO.get(race)
        if not geo:
            unmatched.append({"race": race, "count": count})
            continue
        origins.append(
            {
                "key": geo["key"],
                "label": geo["label"],
                "race": race,
                "lat": geo["lat"],
                "lng": geo["lng"],
                "count": count,
            }
        )

    payload = {
        "destination": {
            "lat": 39.9612,
            "lng": -82.9988,
            "label": "Ohio State University",
            "short": "OSU",
            "city": "Columbus, Ohio",
        },
        "origins": origins,
        "totals": {
            "scholars_total": total_scholars,
            "scholars_with_race": matched,
            "race_categories": len(origins),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "unmatched_races": unmatched,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[OK] Wrote {args.out}")
    print(f"     scholars_total={total_scholars}  with_race={matched}  origins={len(origins)}")
    if unmatched:
        print(f"     unmatched races (no geo): {unmatched}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
