"""
bioRxiv / medRxiv collector — life-sci preprints by author.

API docs: https://api.biorxiv.org/
Free, no auth. Searches by author name across bioRxiv and medRxiv.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base_collector import BaseCollector, CollectorResult, ProfessorQuery
from .validation import normalize

BASE = "https://api.biorxiv.org/details"


class BiorxivCollector(BaseCollector):
    def __init__(self, **kwargs):
        kwargs.setdefault("rate_limit_delay", 1.5)
        super().__init__(**kwargs)

    @property
    def source_name(self) -> str:
        return "biorxiv"

    async def collect(self, query: ProfessorQuery) -> CollectorResult:
        matched: List[Dict[str, Any]] = []
        q_last = normalize(query.last_name)
        q_first = normalize(query.first_name)

        for server in ("biorxiv", "medrxiv"):
            # The endpoint supports /details/{server}/{DOI} and
            # /details/{server}/{interval}. We use a broad recent date window
            # and filter client-side by author name.
            # For scalability, prefer the /pub/{server}/{interval} endpoint
            # which lists DOIs; but detailed filter by author isn't offered, so
            # we leverage Europe PMC search-style fallback if available.
            # Here we query the /pubs endpoint for the last 24 months.
            cursor = 0
            for _ in range(3):  # limit pagination
                url = f"{BASE}/{server}/2023-01-01/2030-12-31/{cursor}"
                resp = await self._get_json(url)
                if not resp:
                    break
                collection = resp.get("collection") or []
                if not collection:
                    break
                for paper in collection:
                    authors = normalize(paper.get("authors") or "")
                    if q_last and q_last in authors and (not q_first or q_first[:3] in authors):
                        matched.append({
                            "server": server,
                            "doi": paper.get("doi"),
                            "title": paper.get("title"),
                            "abstract": paper.get("abstract"),
                            "authors": paper.get("authors"),
                            "date": paper.get("date"),
                            "category": paper.get("category"),
                            "version": paper.get("version"),
                        })
                messages = resp.get("messages") or []
                total = (messages[0].get("total") if messages else 0) or 0
                cursor += len(collection)
                if cursor >= int(total):
                    break

        if not matched:
            return self._make_result(query, success=False, error="No bioRxiv/medRxiv preprints matched")

        # Dedupe by DOI
        seen = set()
        dedup = []
        for m in matched:
            if m["doi"] in seen:
                continue
            seen.add(m["doi"])
            dedup.append(m)

        data = {"total_preprints": len(dedup), "preprints": dedup}
        return self._make_result(query, success=True, data=data, raw_text=self._to_text(data, query))

    def _to_text(self, data: Dict, query: ProfessorQuery) -> str:
        lines = [
            f"=== bioRxiv/medRxiv preprints by {query.name} ===",
            f"Total: {data['total_preprints']}",
            "",
        ]
        for p in data["preprints"]:
            lines.append(f"• {p['title']} [{p['server']} v{p.get('version')}, {p.get('date')}]")
            lines.append(f"  DOI: {p['doi']} | Category: {p.get('category')}")
            if p.get("abstract"):
                lines.append(f"  Abstract: {p['abstract'][:1500]}")
            lines.append("")
        return "\n".join(lines)
