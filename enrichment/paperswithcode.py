"""
Papers With Code collector — code implementations linked to papers.

API docs: https://paperswithcode.com/api/v1/docs/
Free, no auth. Useful for CS / ML researchers to surface code repos and
benchmark results tied to their publications.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base_collector import BaseCollector, CollectorResult, ProfessorQuery

BASE = "https://paperswithcode.com/api/v1"


class PapersWithCodeCollector(BaseCollector):
    def __init__(self, **kwargs):
        kwargs.setdefault("rate_limit_delay", 1.2)
        super().__init__(**kwargs)

    @property
    def source_name(self) -> str:
        return "paperswithcode"

    async def collect(self, query: ProfessorQuery) -> CollectorResult:
        # paperswithcode.com rejects some names with malformed responses
        # (returns HTML 500 pages that break JSON parsing), so we fetch raw
        # and decode defensively.
        await self._rate_limit()
        try:
            client = await self.get_client()
            http = await client.get(f"{BASE}/papers/", params={"q": query.name, "items_per_page": 50})
            if http.status_code in (400, 401, 403, 404, 410, 422, 500, 502, 503):
                return self._make_result(query, success=False, error=f"paperswithcode HTTP {http.status_code}")
            try:
                resp = http.json()
            except Exception as e:
                return self._make_result(query, success=False, error=f"paperswithcode non-JSON response: {e}")
        except Exception as e:
            return self._make_result(query, success=False, error=f"paperswithcode fetch error: {e}")

        if not resp:
            return self._make_result(query, success=False, error="paperswithcode query empty")

        results = resp.get("results") or []
        q_last = query.last_name.lower()

        matched: List[Dict[str, Any]] = []
        for p in results:
            authors = (p.get("authors") or [])
            authors_lower = " ".join(authors).lower() if isinstance(authors, list) else str(authors).lower()
            if q_last and q_last not in authors_lower:
                continue
            paper_id = p.get("id")
            repos = await self._get_json(f"{BASE}/papers/{paper_id}/repositories/") if paper_id else None
            repo_list = []
            if repos:
                for r in (repos.get("results") or []):
                    repo_list.append({
                        "url": r.get("url"),
                        "framework": r.get("framework"),
                        "stars": r.get("stars"),
                        "is_official": r.get("is_official"),
                    })
            matched.append({
                "id": paper_id,
                "title": p.get("title"),
                "abstract": (p.get("abstract") or "")[:1500],
                "authors": authors,
                "published": p.get("published"),
                "url_abs": p.get("url_abs"),
                "url_pdf": p.get("url_pdf"),
                "repositories": repo_list,
            })

        if not matched:
            return self._make_result(query, success=False, error="No papers-with-code entries matched")

        data = {
            "total_papers": len(matched),
            "papers": matched,
        }
        return self._make_result(query, success=True, data=data, raw_text=self._to_text(data, query))

    def _to_text(self, data: Dict, query: ProfessorQuery) -> str:
        lines = [
            f"=== Papers With Code entries for {query.name} ===",
            f"Total: {data['total_papers']}",
            "",
        ]
        for p in data["papers"]:
            lines.append(f"• {p['title']} — {p.get('published')}")
            if p["abstract"]:
                lines.append(f"  Abstract: {p['abstract']}")
            for r in p["repositories"]:
                lines.append(
                    f"  Code: {r['url']} [{r.get('framework')}] "
                    f"stars={r.get('stars')} official={r.get('is_official')}"
                )
            lines.append("")
        return "\n".join(lines)
