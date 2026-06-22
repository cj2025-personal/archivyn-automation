"""
PubMed Central OA collector — biomedical full-text (legal OA subset).

Uses NCBI E-utilities ESearch to find PMC articles by author, then EFetch
for XML. Licensed for text/data mining. Free, no key required (NCBI_API_KEY
recommended for higher limits).
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional

from .base_collector import BaseCollector, CollectorResult, ProfessorQuery

ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
ESUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


class PMCOpenAccessCollector(BaseCollector):
    def __init__(self, **kwargs):
        kwargs.setdefault("rate_limit_delay", 0.4)  # NCBI: 3/s without key, 10/s with
        super().__init__(**kwargs)
        self.api_key = os.getenv("NCBI_API_KEY", "")
        self.max_articles = int(os.getenv("PMC_MAX_ARTICLES", "20"))
        self.full_text_char_cap = int(os.getenv("PMC_FULLTEXT_CAP", "8000"))

    @property
    def source_name(self) -> str:
        return "pmc_oa"

    def _common_params(self) -> Dict[str, str]:
        p: Dict[str, str] = {"tool": "osu-scholar-enrichment", "email": os.getenv("OPENALEX_EMAIL") or "research@example.com"}
        if self.api_key:
            p["api_key"] = self.api_key
        return p

    async def collect(self, query: ProfessorQuery) -> CollectorResult:
        # Search the PMC open-access subset
        term = f'("{query.name}"[Author]) AND open access[filter]'
        params = {**self._common_params(), "db": "pmc", "term": term, "retmode": "json", "retmax": str(self.max_articles)}
        search_resp = await self._get_json(ESEARCH, params=params)
        if not search_resp:
            return self._make_result(query, success=False, error="PMC esearch failed")

        idlist = (search_resp.get("esearchresult") or {}).get("idlist") or []
        if not idlist:
            return self._make_result(query, success=False, error="No PMC articles matched")

        articles: List[Dict[str, Any]] = []
        for pmcid in idlist:
            params = {**self._common_params(), "db": "pmc", "id": pmcid, "rettype": "full", "retmode": "xml"}
            await self._rate_limit()
            try:
                client = await self.get_client()
                resp = await client.get(EFETCH, params=params)
                if resp.status_code != 200:
                    continue
                xml_text = resp.text
            except Exception:
                continue

            try:
                root = ET.fromstring(xml_text)
            except ET.ParseError:
                continue

            # Extract title, abstract, first N chars of body
            title = "".join(root.itertext()).split("\n", 1)[0][:300]
            title_el = root.find(".//article-title")
            if title_el is not None:
                title = "".join(title_el.itertext()).strip()

            abstract_parts = []
            for abs_el in root.findall(".//abstract"):
                abstract_parts.append("".join(abs_el.itertext()).strip())
            abstract = "\n".join(abstract_parts)[:4000]

            body_parts = []
            for body in root.findall(".//body"):
                body_parts.append("".join(body.itertext()).strip())
            body_text = "\n".join(body_parts)[: self.full_text_char_cap]

            articles.append({
                "pmcid": f"PMC{pmcid}",
                "url": f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmcid}/",
                "title": title,
                "abstract": abstract,
                "body_excerpt": body_text,
            })

        if not articles:
            return self._make_result(query, success=False, error="Search matched but no articles retrievable")

        data = {"total_articles": len(articles), "articles": articles}
        return self._make_result(query, success=True, data=data, raw_text=self._to_text(data, query))

    def _to_text(self, data: Dict, query: ProfessorQuery) -> str:
        lines = [
            f"=== PubMed Central OA full-text excerpts for {query.name} ===",
            f"Total articles: {data['total_articles']}",
            "",
        ]
        for a in data["articles"]:
            lines.append(f"── {a['title']} ({a['pmcid']}) ──")
            lines.append(f"URL: {a['url']}")
            if a["abstract"]:
                lines.append(f"Abstract: {a['abstract']}")
            if a["body_excerpt"]:
                lines.append(f"Excerpt: {a['body_excerpt']}")
            lines.append("")
        return "\n".join(lines)
