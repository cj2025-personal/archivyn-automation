"""
CORE API collector — 300M+ OA full-text aggregator.

API docs: https://api.core.ac.uk/docs/v3
Requires CORE_API_KEY (free tier: 1k reqs/day, register at core.ac.uk).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from .base_collector import BaseCollector, CollectorResult, ProfessorQuery

BASE = "https://api.core.ac.uk/v3"


class CoreAPICollector(BaseCollector):
    def __init__(self, **kwargs):
        kwargs.setdefault("rate_limit_delay", 1.5)
        super().__init__(**kwargs)
        self.api_key = os.getenv("CORE_API_KEY", "")

    @property
    def source_name(self) -> str:
        return "core_api"

    async def collect(self, query: ProfessorQuery) -> CollectorResult:
        if not self.api_key:
            return self._make_result(
                query, success=False,
                error="CORE_API_KEY not set; register at https://core.ac.uk",
            )

        params = {
            "q": f'authors:"{query.name}"',
            "limit": 30,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        await self._rate_limit()
        try:
            client = await self.get_client()
            resp = await client.get(f"{BASE}/search/works", params=params, headers=headers)
            if resp.status_code in (401, 403):
                return self._make_result(query, success=False, error=f"CORE auth failed ({resp.status_code})")
            if resp.status_code != 200:
                return self._make_result(query, success=False, error=f"CORE HTTP {resp.status_code}")
            payload = resp.json()
        except Exception as e:
            return self._make_result(query, success=False, error=f"CORE fetch error: {e}")

        results = payload.get("results") or []
        q_last = query.last_name.lower()
        works: List[Dict[str, Any]] = []
        for r in results:
            authors = " ".join(a.get("name", "") for a in (r.get("authors") or [])).lower()
            if q_last and q_last not in authors:
                continue
            works.append({
                "id": r.get("id"),
                "title": r.get("title"),
                "abstract": (r.get("abstract") or "")[:1500],
                "year": r.get("yearPublished"),
                "doi": r.get("doi"),
                "download_url": r.get("downloadUrl"),
                "publisher": r.get("publisher"),
                "language": (r.get("language") or {}).get("name"),
                "repositories": [d.get("name") for d in (r.get("dataProviders") or [])],
            })

        if not works:
            return self._make_result(query, success=False, error="No CORE works matched")

        data = {"total_works": len(works), "works": works}
        return self._make_result(query, success=True, data=data, raw_text=self._to_text(data, query))

    def _to_text(self, data: Dict, query: ProfessorQuery) -> str:
        lines = [f"=== CORE OA works by {query.name} ===", f"Total: {data['total_works']}", ""]
        for w in data["works"]:
            lines.append(f"• {w['title']} ({w.get('year')})")
            if w.get("repositories"):
                lines.append(f"  Repositories: {', '.join(filter(None, w['repositories']))}")
            if w.get("download_url"):
                lines.append(f"  Download: {w['download_url']}")
            if w.get("abstract"):
                lines.append(f"  Abstract: {w['abstract']}")
            lines.append("")
        return "\n".join(lines)
