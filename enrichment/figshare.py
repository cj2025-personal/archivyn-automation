"""
Figshare collector — datasets, figures, code, posters.

API docs: https://docs.figshare.com/
Free public endpoints, no auth needed for searching published articles.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base_collector import BaseCollector, CollectorResult, ProfessorQuery
from .validation import normalize

SEARCH_URL = "https://api.figshare.com/v2/articles/search"


class FigshareCollector(BaseCollector):
    def __init__(self, **kwargs):
        kwargs.setdefault("rate_limit_delay", 1.0)
        super().__init__(**kwargs)

    @property
    def source_name(self) -> str:
        return "figshare"

    async def collect(self, query: ProfessorQuery) -> CollectorResult:
        body = {
            "search_for": f'"{query.name}"',
            "page_size": 30,
        }
        resp = await self._post_json(SEARCH_URL, json_body=body)
        if not resp:
            return self._make_result(query, success=False, error="Figshare search empty")

        q_last = normalize(query.last_name)
        records: List[Dict[str, Any]] = []
        for r in resp:
            authors = r.get("authors") or []
            authors_str = normalize(" ".join(a.get("full_name", "") for a in authors))
            if q_last and q_last not in authors_str:
                continue
            records.append({
                "id": r.get("id"),
                "title": r.get("title"),
                "doi": r.get("doi"),
                "url": r.get("url_public_html") or r.get("url"),
                "defined_type": r.get("defined_type_name") or r.get("defined_type"),
                "published_date": r.get("published_date"),
                "authors": [a.get("full_name") for a in authors],
            })

        if not records:
            return self._make_result(query, success=False, error="No Figshare items matched")

        data = {"total": len(records), "items": records}
        return self._make_result(query, success=True, data=data, raw_text=self._to_text(data, query))

    def _to_text(self, data: Dict, query: ProfessorQuery) -> str:
        lines = [
            f"=== Figshare items by {query.name} ===",
            f"Total: {data['total']}",
            "",
        ]
        for r in data["items"]:
            lines.append(f"• [{r.get('defined_type')}] {r['title']} ({r.get('published_date')})")
            lines.append(f"  DOI: {r.get('doi')} | URL: {r.get('url')}")
        return "\n".join(lines)
