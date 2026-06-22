"""
Backfill professor_name into existing chunks.json files.

For each profile:
- Read profile JSON from output/profiles/{profile_id}/{profile_id}.json
- Read chunks JSON from output/chunked_profiles/{profile_id}/chunks.json
- Add "professor_name": <name> to every chunk entry (if missing or different)
- Overwrite chunks.json in place.

Run once after generating chunks:

    python backfill_chunk_names.py
"""

import json
from pathlib import Path


def backfill_chunk_names(
    profiles_root: Path = Path("output/profiles"),
    chunks_root: Path = Path("output/chunked_profiles"),
) -> None:
    if not profiles_root.exists() or not chunks_root.exists():
        print(f"profiles_root={profiles_root} or chunks_root={chunks_root} does not exist")
        return

    updated = 0
    skipped = 0

    for profile_dir in profiles_root.iterdir():
        if not profile_dir.is_dir():
            continue

        profile_id = profile_dir.name
        profile_json_path = profile_dir / f"{profile_id}.json"
        chunks_dir = chunks_root / profile_id
        chunks_json_path = chunks_dir / "chunks.json"

        if not profile_json_path.exists() or not chunks_json_path.exists():
            skipped += 1
            continue

        try:
            with profile_json_path.open("r", encoding="utf-8") as f:
                profile_data = json.load(f)
            name = profile_data.get("name") or ""
        except Exception as e:
            print(f"[WARN] Failed to read profile JSON for {profile_id}: {e}")
            skipped += 1
            continue

        if not name:
            # Nothing to add; skip but count
            skipped += 1
            continue

        try:
            with chunks_json_path.open("r", encoding="utf-8") as f:
                chunks_data = json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to read chunks JSON for {profile_id}: {e}")
            skipped += 1
            continue

        sections = chunks_data.get("sections") or {}
        changed = False

        for section_name, section_chunks in sections.items():
            if not isinstance(section_chunks, list):
                continue
            for chunk in section_chunks:
                if not isinstance(chunk, dict):
                    continue
                if chunk.get("professor_name") != name:
                    chunk["professor_name"] = name
                    changed = True

        if not changed:
            skipped += 1
            continue

        try:
            with chunks_json_path.open("w", encoding="utf-8") as f:
                json.dump(chunks_data, f, indent=2, ensure_ascii=False)
            updated += 1
            print(f"[OK] Updated chunks for {profile_id} with professor_name='{name}'")
        except Exception as e:
            print(f"[WARN] Failed to write chunks JSON for {profile_id}: {e}")
            skipped += 1

    print(f"\nDone. Updated {updated} profiles; skipped {skipped}.")


if __name__ == "__main__":
    backfill_chunk_names()

