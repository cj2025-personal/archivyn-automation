from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_SQLITE_PATH = "data/scholars.db"
DEFAULT_TABLE = "legend_scholars"


SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS {table} (
    profile_id      TEXT PRIMARY KEY,
    professor_name  TEXT,
    doc             TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_{table}_name ON {table}(professor_name);
"""


def connect(db_path: str = DEFAULT_SQLITE_PATH, table: str = DEFAULT_TABLE) -> sqlite3.Connection:
    p = Path(db_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    for stmt in SCHEMA_DDL.format(table=table).strip().split(";"):
        s = stmt.strip()
        if s:
            conn.execute(s)
    conn.commit()
    return conn


def upsert_scholar(
    conn: sqlite3.Connection,
    profile_id: str,
    professor_name: str,
    document: Dict[str, Any],
    updated_at: str,
    table: str = DEFAULT_TABLE,
) -> None:
    conn.execute(
        f"""
        INSERT INTO {table} (profile_id, professor_name, doc, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(profile_id) DO UPDATE SET
            professor_name = excluded.professor_name,
            doc            = excluded.doc,
            updated_at     = excluded.updated_at
        """,
        (profile_id, professor_name, json.dumps(document, ensure_ascii=False), updated_at),
    )
    conn.commit()


def count_scholars(conn: sqlite3.Connection, table: str = DEFAULT_TABLE) -> int:
    cur = conn.execute(f"SELECT COUNT(*) FROM {table}")
    return int(cur.fetchone()[0])


def get_scholar(
    conn: sqlite3.Connection, profile_id: str, table: str = DEFAULT_TABLE
) -> Optional[Dict[str, Any]]:
    cur = conn.execute(
        f"SELECT doc FROM {table} WHERE profile_id = ?", (profile_id,)
    )
    row = cur.fetchone()
    if not row:
        return None
    return json.loads(row[0])
