"""
SQLite mirror of sync_local_chunked_to_mongodb.py.

Reuses LocalChunkedMongoSync's document-building logic (LLM section summaries,
name parsing, chunk aggregation, etc.) and only swaps the final upsert
destination from MongoDB to SQLite.

Usage:
  python sync_local_chunked_to_sqlite.py --runs output/url_list_runs/<run_id> [...]
  python sync_local_chunked_to_sqlite.py --chunks-root <dir> --profiles-root <dir>

Flags:
  --db-path data/scholars.db   target SQLite file (auto-created)
  --table legend_scholars      target table (default: legend_scholars)
  --no-llm                     disable LLM summaries (faster, lower quality)
"""
from __future__ import annotations

import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from config import sqlite_utils
from sync_local_chunked_to_mongodb import LocalChunkedMongoSync


class LocalChunkedSqliteSync(LocalChunkedMongoSync):
    """Same document construction as the Mongo sync; SQLite as destination."""

    def __init__(
        self,
        db_path: str = sqlite_utils.DEFAULT_SQLITE_PATH,
        table: str = sqlite_utils.DEFAULT_TABLE,
        use_llm: bool = True,
    ):
        # Skip parent __init__ (don't open Mongo client). Replicate only the
        # bits we need: load .env, optional OpenAI client for LLM summaries.
        load_dotenv(dotenv_path=".env")

        self.collection_name = table
        self.table = table
        self.use_llm = use_llm

        self.openai_client = None
        self.chunk_summary_max_chars = 320
        self.section_summary_max_chars = 900
        self.long_bio_max_chars = 900
        if self.use_llm:
            openai_key = os.getenv("OPENAI_API_KEY")
            if not openai_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables")
            try:
                from openai import OpenAI
                import httpx

                http_client = httpx.Client(timeout=120.0)
                self.openai_client = OpenAI(api_key=openai_key, http_client=http_client)
            except Exception:
                from openai import OpenAI

                self.openai_client = OpenAI(api_key=openai_key)

        self.conn: sqlite3.Connection = sqlite_utils.connect(db_path, table=table)
        self.db_path = db_path

    def sync_chunks_file(self, chunks_path: Path, profiles_root: Optional[Path]) -> bool:
        try:
            data = self._load_json(chunks_path)
            profile_id = data.get("profile_id") or chunks_path.parent.name
            sections = data.get("sections", {})
            if not sections:
                print(f"[Skip] No sections in {chunks_path}")
                return False

            professor_name = ""
            if profiles_root:
                profile_json = profiles_root / profile_id / f"{profile_id}.json"
                if profile_json.exists():
                    prof_data = self._load_json(profile_json)
                    professor_name = (
                        prof_data.get("name")
                        or prof_data.get("professor_name")
                        or ""
                    ).strip()

            if not professor_name:
                professor_name = self._extract_name_from_chunks_data(data)

            if not professor_name:
                professor_name = "Unknown"

            source_payload = self._load_source_chunks_payload(profiles_root, profile_id)
            document = self._create_scholar_document(
                profile_id,
                professor_name,
                sections,
                source_payload=source_payload,
            )
            sqlite_utils.upsert_scholar(
                self.conn,
                profile_id=profile_id,
                professor_name=professor_name,
                document=document,
                updated_at=datetime.now(timezone.utc).isoformat(),
                table=self.table,
            )
            print(f"[SQLite] Upserted {professor_name} ({profile_id}) into {self.table}")
            return True
        except Exception as e:
            print(f"[Error] Failed to sync {chunks_path}: {e}")
            return False

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Sync local chunked_profiles into SQLite")
    parser.add_argument("--chunks-root", type=str, default=None, help="Path to chunked_profiles directory")
    parser.add_argument("--profiles-root", type=str, default=None, help="Path to profiles directory (for names)")
    parser.add_argument("--runs", nargs="*", default=None, help="One or more output/url_list_runs/<run_id> directories")
    parser.add_argument("--db-path", type=str, default=sqlite_utils.DEFAULT_SQLITE_PATH, help="SQLite file path")
    parser.add_argument("--table", type=str, default=sqlite_utils.DEFAULT_TABLE, help="Target table name")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM summaries")
    args = parser.parse_args()

    sync = LocalChunkedSqliteSync(db_path=args.db_path, table=args.table, use_llm=not args.no_llm)

    try:
        if args.runs:
            for run_dir in args.runs:
                run_path = Path(run_dir)
                chunks_root = run_path / "chunked_profiles"
                profiles_root = run_path / "profiles"
                sync.sync_from_roots(chunks_root, profiles_root if profiles_root.exists() else None)
        else:
            if not args.chunks_root:
                raise SystemExit("Provide --chunks-root or --runs")
            chunks_root = Path(args.chunks_root)
            profiles_root = Path(args.profiles_root) if args.profiles_root else None
            sync.sync_from_roots(chunks_root, profiles_root)

        total = sqlite_utils.count_scholars(sync.conn, table=args.table)
        print(f"[SQLite] Done. Total rows in '{args.table}': {total}  (db={args.db_path})")
    finally:
        sync.close()


if __name__ == "__main__":
    main()
