"""
Zenodo collector — datasets, software, posters uploaded by researcher.

API docs: https://developers.zenodo.org/
Free, no auth for public records. ZENODO_ACCESS_TOKEN raises rate limits.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from .base_collector import BaseCollector, CollectorResult, ProfessorQuery
from .validation import normalize

BASE_URL = "https://zenodo.org/api/records"


class ZenodoCollector(BaseCollector):
    def __init__(self, **kwargs):
        kwargs.setdefault("rate_limit_delay", 1.0)
        super().__init__(**kwargs)
        self.token = os.getenv("ZENODO_ACCESS_TOKEN", "")

    @property
    def source_name(self) -> str:
        return "zenodo"

    async def collect(self, query: ProfessorQuery) -> CollectorResult:
        params = {
            "q": f'creators.name:"{query.name}"',
            "size": 30,
            "sort": "mostrecent",
        }
        if self.token:
            params["access_token"] = self.token

        resp = await self._get_json(BASE_URL, params=params)
        if not resp:
            return self._make_result(query, success=False, error="Zenodo query returned empty")

        hits = (resp.get("hits") or {}).get("hits") or []
        q_last = normalize(query.last_name)
        records: List[Dict[str, Any]] = []

        for h in hits:
            meta = h.get("metadata") or {}
            creators = meta.get("creators") or []
            creator_names = normalize(" ".join(c.get("name", "") for c in creators))
            if q_last and q_last not in creator_names:
                continue
            records.append({
                "id": h.get("id"),
                "doi": meta.get("doi"),
                "title": meta.get("title"),
                "description": (meta.get("description") or "")[:1500],
                "resource_type": (meta.get("resource_type") or {}).get("title"),
                "publication_date": meta.get("publication_date"),
                "keywords": meta.get("keywords"),
                "license": (meta.get("license") or {}).get("id"),
                "url": h.get("links", {}).get("html"),
                "creators": [c.get("name") for c in creators],
            })

        if not records:
            return self._make_result(query, success=False, error="No Zenodo records matched")

        data = {"total_records": len(records), "records": records}
        return self._make_result(query, success=True, data=data, raw_text=self._to_text(data, query))

    def _to_text(self, data: Dict, query: ProfessorQuery) -> str:
        lines = [
            f"=== Zenodo deposits by {query.name} ===",
            f"Total: {data['total_records']}",
            "",
        ]
        for r in data["records"]:
            lines.append(f"• [{r.get('resource_type')}] {r['title']} ({r.get('publication_date')})")
            lines.append(f"  DOI: {r.get('doi')} | License: {r.get('license')}")
            if r.get("url"):
                lines.append(f"  URL: {r['url']}")
            if r.get("description"):
                lines.append(f"  Description: {r['description']}")
            lines.append("")
        return "\n".join(lines)
