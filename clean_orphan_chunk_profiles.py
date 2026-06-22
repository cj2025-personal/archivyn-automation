"""
Remove chunked_profiles entries that do not have a matching profile JSON.

We only want chunked profiles for profile IDs that exist under:
    output/profiles/{profile_id}/{profile_id}.json

This script:
- Lists all profile IDs present in output/profiles
- Walks output/chunked_profiles
- Deletes any chunked_profiles/{profile_id} directory whose ID is not in profiles

Usage:
    python clean_orphan_chunk_profiles.py
"""

import shutil
from pathlib import Path


def clean_orphan_chunk_profiles(
    profiles_root: Path = Path("output/profiles"),
    chunks_root: Path = Path("output/chunked_profiles"),
) -> None:
    if not chunks_root.exists():
        print(f"[Setup] Chunks root does not exist: {chunks_root}")
        return

    if not profiles_root.exists():
        print(f"[Setup] Profiles root does not exist: {profiles_root}")
        return

    # Collect valid profile IDs (directories in output/profiles)
    valid_ids = {
        d.name
        for d in profiles_root.iterdir()
        if d.is_dir() and (d / f"{d.name}.json").exists()
    }
    print(f"[Info] Found {len(valid_ids)} profile IDs with JSON in {profiles_root}")

    removed = 0
    skipped = 0

    for chunk_dir in chunks_root.iterdir():
        if not chunk_dir.is_dir():
            continue

        profile_id = chunk_dir.name
        if profile_id not in valid_ids:
            try:
                print(f"[Remove] Deleting orphan chunk directory: {chunk_dir}")
                shutil.rmtree(chunk_dir)
                removed += 1
            except Exception as e:
                print(f"[Warn] Failed to delete {chunk_dir}: {e}")
        else:
            skipped += 1

    print(f"\n[Summary] Removed {removed} orphan chunk directories; kept {skipped} valid ones.")


if __name__ == "__main__":
    clean_orphan_chunk_profiles()

