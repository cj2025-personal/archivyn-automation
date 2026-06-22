"""
arXiv collector — preprint full-text metadata by author name.

API docs: https://info.arxiv.org/help/api/user-manual.html
Free, no auth. Returns Atom XML; parsed via stdlib.
Covers physics, math, CS, quantitative bio, economics, stats, EE/SS.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional

from .base_collector import BaseCollector, CollectorResult, ProfessorQuery
from .validation import names_match_from_query

BASE_URL = "http://export.arxiv.org/api/query"
NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


class ArxivCollector(BaseCollector):
    def __init__(self, **kwargs):
        # arXiv requests no more than 1 req per 3s per IP
        kwargs.setdefault("rate_limit_delay", 3.0)
        super().__init__(**kwargs)

    @property
    def source_name(self) -> str:
        return "arxiv"

    async def collect(self, query: ProfessorQuery) -> CollectorResult:
        # Query by author name; arXiv handles name variants loosely
        params = {
            "search_query": f'au:"{query.name}"',
            "start": 0,
            "max_results": 50,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }

        text = await self._fetch_html(BASE_URL, params=params)
        if not text:
            return self._make_result(query, success=False, error="arXiv query returned empty")

        try:
            root = ET.fromstring(text)
        except ET.ParseError as e:
            return self._make_result(query, success=False, error=f"arXiv XML parse error: {e}")

        entries = root.findall("a:entry", NS)
        papers: List[Dict[str, Any]] = []

        for entry in entries:
            authors = [
                (a.findtext("a:name", default="", namespaces=NS) or "").strip()
                for a in entry.findall("a:author", NS)
            ]
            # Require that at least one author matches the queried professor
            if not any(names_match_from_query(query, a, require_first_name=False) for a in authors):
                continue

            title = (entry.findtext("a:title", default="", namespaces=NS) or "").strip()
            summary = (entry.findtext("a:summary", default="", namespaces=NS) or "").strip()
            published = entry.findtext("a:published", default="", namespaces=NS) or ""
            arxiv_id = entry.findtext("a:id", default="", namespaces=NS) or ""
            primary_cat_el = entry.find("arxiv:primary_category", NS)
            primary_category = primary_cat_el.get("term") if primary_cat_el is not None else ""

            # Extract PDF link
            pdf_url = ""
            for link in entry.findall("a:link", NS):
                if link.get("title") == "pdf":
                    pdf_url = link.get("href", "")
                    break

            papers.append({
                "arxiv_id": arxiv_id.rsplit("/", 1)[-1],
                "title": title,
                "abstract": summary,
                "authors": authors,
                "published": published,
                "primary_category": primary_category,
                "pdf_url": pdf_url,
                "arxiv_url": arxiv_id,
            })

        if not papers:
            return self._make_result(query, success=False, error="No arXiv preprints matched")

        data = {"total_preprints": len(papers), "preprints": papers}
        return self._make_result(query, success=True, data=data, raw_text=self._to_text(data, query))

    def _to_text(self, data: Dict, query: ProfessorQuery) -> str:
        lines = [
            f"=== arXiv preprints by {query.name} ===",
            f"Total matched: {data['total_preprints']}",
            "",
        ]
        for p in data["preprints"]:
            lines.append(f"• {p['title']}")
            lines.append(f"  arXiv: {p['arxiv_id']} | Category: {p['primary_category']} | Submitted: {p['published'][:10]}")
            if p["abstract"]:
                lines.append(f"  Abstract: {p['abstract'][:1500]}")
            if p["pdf_url"]:
                lines.append(f"  PDF: {p['pdf_url']}")
            lines.append("")
        return "\n".join(lines)
