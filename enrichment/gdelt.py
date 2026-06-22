"""
GDELT DOC 2.0 collector — global news index.

API docs: https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/
Free, no auth. Returns news articles mentioning the researcher worldwide,
deeper than Google News.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base_collector import BaseCollector, CollectorResult, ProfessorQuery

API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"


class GDELTCollector(BaseCollector):
    def __init__(self, **kwargs):
        kwargs.setdefault("rate_limit_delay", 2.0)
        super().__init__(**kwargs)

    @property
    def source_name(self) -> str:
        return "gdelt"

    async def collect(self, query: ProfessorQuery) -> CollectorResult:
        params = {
            "query": f'"{query.name}" "Ohio State"',
            "mode": "ArtList",
            "format": "json",
            "maxrecords": 50,
            "sort": "DateDesc",
        }
        resp = await self._get_json(API_URL, params=params)
        if not resp:
            return self._make_result(query, success=False, error="GDELT returned empty")

        articles_raw = resp.get("articles") or []
        articles: List[Dict[str, Any]] = []
        for a in articles_raw:
            articles.append({
                "title": a.get("title"),
                "url": a.get("url"),
                "domain": a.get("domain"),
                "language": a.get("language"),
                "country": a.get("sourcecountry"),
                "seen_date": a.get("seendate"),
            })

        if not articles:
            return self._make_result(query, success=False, error="No GDELT articles matched")

        # By-country / by-domain aggregates
        from collections import Counter
        domains = Counter(a["domain"] for a in articles if a.get("domain"))
        countries = Counter(a["country"] for a in articles if a.get("country"))

        data = {
            "total_articles": len(articles),
            "top_domains": domains.most_common(10),
            "top_countries": countries.most_common(10),
            "articles": articles,
        }
        return self._make_result(query, success=True, data=data, raw_text=self._to_text(data, query))

    def _to_text(self, data: Dict, query: ProfessorQuery) -> str:
        lines = [
            f"=== GDELT global news mentions of {query.name} ===",
            f"Total: {data['total_articles']}",
        ]
        if data["top_domains"]:
            lines.append(f"Top domains: {', '.join(f'{d}({c})' for d, c in data['top_domains'])}")
        if data["top_countries"]:
            lines.append(f"Top countries: {', '.join(f'{c}({n})' for c, n in data['top_countries'])}")
        lines.append("")
        for a in data["articles"][:30]:
            lines.append(f"• {a['title']}")
            lines.append(f"  {a['url']} [{a.get('domain')} / {a.get('country')} / {a.get('seen_date')}]")
        return "\n".join(lines)
