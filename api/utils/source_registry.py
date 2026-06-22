"""
Source registry for auditable ingestion.
Stores records in JSONL with in-memory index for the latest version.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class SourceRecord:
    source_id: str
    url: str
    accessed_at: str
    domain: str
    license_type: str
    license_url: str
    rights_holder: str
    allowed_use: str
    paywalled: bool
    robots_allowed: Optional[bool]
    copyright_notes: str
    status: str
    status_reason: str = ""
    final_url: str = ""
    fetch: Optional[Dict] = None
    content: Optional[Dict] = None
    search: Optional[Dict] = None
    profile_context: Optional[Dict] = None
    record_version: int = 1
    recorded_at: str = ""
    supersedes_version: Optional[int] = None


class SourceRegistry:
    def __init__(self, registry_path: str) -> None:
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self._latest: Dict[str, SourceRecord] = {}
        self._load_existing()

    def _load_existing(self) -> None:
        if not self.registry_path.exists():
            return
        try:
            with open(self.registry_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        src_id = data.get("source_id")
                        if not src_id:
                            continue
                        base = {
                            "source_id": "",
                            "url": "",
                            "accessed_at": "",
                            "domain": "",
                            "license_type": "unknown",
                            "license_url": "",
                            "rights_holder": "",
                            "allowed_use": "facts_only",
                            "paywalled": False,
                            "robots_allowed": None,
                            "copyright_notes": "",
                            "status": "review",
                            "status_reason": "",
                            "final_url": "",
                            "fetch": None,
                            "content": None,
                            "search": None,
                            "profile_context": None,
                            "record_version": 1,
                            "recorded_at": "",
                            "supersedes_version": None,
                        }
                        base.update(data)
                        self._latest[src_id] = SourceRecord(**base)
                    except Exception:
                        continue
        except Exception:
            # Fail silently: registry is best-effort
            pass

    def upsert(self, record: SourceRecord) -> SourceRecord:
        existing = self._latest.get(record.source_id)
        if existing:
            record.record_version = existing.record_version + 1
            record.supersedes_version = existing.record_version
        else:
            record.record_version = 1
        record.recorded_at = _utc_now()
        self._latest[record.source_id] = record
        try:
            with open(self.registry_path, "a", encoding="utf-8") as f:
                json.dump(asdict(record), f, ensure_ascii=False)
                f.write("\n")
        except Exception:
            pass
        return record

    def get_latest(self, source_id: str) -> Optional[SourceRecord]:
        return self._latest.get(source_id)

    def iter_latest(self):
        return self._latest.values()

    def snapshot(self, snapshot_path: Optional[str] = None) -> Optional[Path]:
        path = Path(snapshot_path) if snapshot_path else self.registry_path.with_suffix(".latest.json")
        try:
            payload = {k: asdict(v) for k, v in self._latest.items()}
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            return path
        except Exception:
            return None
