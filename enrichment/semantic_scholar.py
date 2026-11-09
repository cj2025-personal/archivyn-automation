"""
Semantic Scholar collector.
API docs: https://api.semanticscholar.org/api-docs/graph
License: https://www.semanticscholar.org/product/api/license

Who the scholar is on Semantic Scholar is read from their own papers, not
guessed from their name: OpenAlex runs first and has already matched the
scholar to works with DOIs, and S2 lists the authors of each of those papers.
The author id that appears on several of them is the scholar. (S2 rarely
records affiliations and splits one person across name variants - "Jeffrey
S. Volek", "J. Volek" - so a name search matched nobody reliably: on the
2026 run every one of 681 scholars came back "not found".)

What is kept is facts - metrics, titles, years, venues, citation counts, ids.
Abstract text is not: it belongs to the authors and publishers, and S2
passes it on under their terms, not its own. The S2 license asks for credit:
wherever this data is shown, say "Semantic Scholar".

Without an API key every anonymous client shares one bucket that 429s
easily; a free key gives a dedicated 1 request/second.
Register: https://www.semanticscholar.org/product/api  ->  SEMANTIC_SCHOLAR_API_KEY=...
"""

import json
import logging
import os
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import httpx

from .base_collector import BaseCollector, CollectorResult, ProfessorQuery
from .crawl_policy import USER_AGENT
from .validation import names_match_from_query

logger = logging.getLogger(__name__)

BASE_URL = "https://api.semanticscholar.org/graph/v1"
ATTRIBUTION = "Semantic Scholar"

PAPER_FIELDS = (
    "paperId,title,year,citationCount,influentialCitationCount,"
    "venue,publicationTypes,journal,isOpenAccess,openAccessPdf,"
    "fieldsOfStudy,s2FieldsOfStudy,externalIds,url"
)

AUTHOR_FIELDS = (
    "authorId,name,aliases,affiliations,homepage,paperCount,"
    "citationCount,hIndex,url,externalIds"
)

# How many of the scholar's DOIs to look up, and how many of them must name
# the same S2 author before that author is taken to be the scholar.
MAX_ANCHOR_DOIS = 30
MIN_ANCHOR_PAPERS = 2


def openalex_dois(profile_id: str) -> List[str]:
    """DOIs of the works OpenAlex matched to this scholar earlier in the run."""
    base = Path(os.getenv("ENRICHMENT_OUTPUT_DIR", "output/osu_faculty_run"))
    path = base / "profiles" / profile_id / "enrichment.json"
    if not path.exists():
        return []
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    works = (doc.get("sources", {}).get("openalex", {}).get("data") or {}).get("works") or []
    works = sorted(works, key=lambda w: w.get("cited_by_count") or 0, reverse=True)
    seen, out = set(), []
    for w in works:
        doi = str(w.get("doi") or "").replace("https://doi.org/", "").strip().lower()
        if doi and doi not in seen:
            seen.add(doi)
            out.append(doi)
    return out[:MAX_ANCHOR_DOIS]


def anchored_author_id(query: ProfessorQuery, papers: List[Optional[Dict]], dois_looked_up: int) -> Optional[str]:
    """The S2 author named on enough of the scholar's own papers, if one is."""
    counts: Counter = Counter()
    for paper in papers:
        if not paper:
            continue
        ids = {a.get("authorId") for a in (paper.get("authors") or [])
               if a.get("authorId") and names_match_from_query(query, a.get("name") or "")}
        counts.update(ids)
    if not counts:
        return None
    (best, n), *rest = counts.most_common()
    needed = MIN_ANCHOR_PAPERS if dois_looked_up >= MIN_ANCHOR_PAPERS else 1
    if n < needed:
        return None
    # Two different authors equally anchored: not enough to tell them apart.
    if rest and rest[0][1] == n:
        return None
    return best


class SemanticScholarCollector(BaseCollector):
    """Citation data for the scholar's S2 author record, found through their papers."""

    def __init__(self, **kwargs):
        self.api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip()
        kwargs.setdefault("rate_limit_delay", 1.2 if self.api_key else 3.0)
        super().__init__(**kwargs)

    @property
    def source_name(self) -> str:
        return "semantic_scholar"

    async def get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            headers = {"User-Agent": USER_AGENT}
            if self.api_key:
                headers["x-api-key"] = self.api_key
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                follow_redirects=True,
                headers=headers,
            )
        return self._client

    async def collect(self, query: ProfessorQuery) -> CollectorResult:
        dois = openalex_dois(query.profile_id)
        if not dois:
            return self._make_result(
                query, success=False,
                error="No OpenAlex works to identify the scholar by (OpenAlex runs first)",
            )

        batch = await self._post_json(
            f"{BASE_URL}/paper/batch",
            {"ids": [f"DOI:{d}" for d in dois]},
            params={"fields": "authors.authorId,authors.name"},
        )
        if batch is None:
            reason = ("Rate-limited by Semantic Scholar (set SEMANTIC_SCHOLAR_API_KEY)"
                      if self.rate_limited else "Semantic Scholar paper lookup failed")
            return self._make_result(query, success=False, error=reason)

        author_id = anchored_author_id(query, batch if isinstance(batch, list) else [], len(dois))
        if not author_id:
            return self._make_result(
                query, success=False,
                error="No Semantic Scholar author is named on enough of the scholar's papers",
            )

        detail = await self._get_json(f"{BASE_URL}/author/{author_id}", params={"fields": AUTHOR_FIELDS}) or {}
        papers = await self._get_author_papers(author_id, limit=100)
        if not detail and not papers and self.rate_limited:
            return self._make_result(query, success=False, error="Rate-limited by Semantic Scholar (set SEMANTIC_SCHOLAR_API_KEY)")

        data = {
            "author_id": author_id,
            "matched_by": f"authorship of {len(dois)} OpenAlex-matched DOIs",
            "attribution": ATTRIBUTION,
            "name": detail.get("name", query.name),
            "aliases": detail.get("aliases", []),
            "affiliations": detail.get("affiliations", []),
            "homepage": detail.get("homepage"),
            "paper_count": detail.get("paperCount", 0),
            "citation_count": detail.get("citationCount", 0),
            "h_index": detail.get("hIndex"),
            "semantic_scholar_url": detail.get("url", ""),
            "external_ids": detail.get("externalIds", {}),
            "papers": papers,
        }
        return self._make_result(query, success=True, data=data, raw_text=self._to_text(data))

    async def _get_author_papers(self, author_id: str, limit: int = 100) -> List[Dict]:
        """The author's papers, most cited first."""
        papers: List[Dict] = []
        offset, batch = 0, min(limit, 100)
        while offset < limit:
            resp = await self._get_json(
                f"{BASE_URL}/author/{author_id}/papers",
                params={"fields": PAPER_FIELDS, "limit": batch, "offset": offset},
            )
            if not resp or not resp.get("data"):
                break
            papers.extend(resp["data"])
            if len(resp["data"]) < batch:
                break
            offset += batch
        papers.sort(key=lambda p: p.get("citationCount") or 0, reverse=True)
        return papers

    def _to_text(self, data: Dict) -> str:
        """Facts as text for chunking - no abstracts."""
        lines = [
            f"=== Semantic Scholar Profile: {data['name']} ===",
            f"Source: {ATTRIBUTION}",
            f"Affiliations: {', '.join(data['affiliations']) if data['affiliations'] else 'N/A'}",
            f"Total papers: {data['paper_count']}",
            f"Total citations: {data['citation_count']}",
        ]
        if data["h_index"]:
            lines.append(f"h-index: {data['h_index']}")
        if data["homepage"]:
            lines.append(f"Homepage: {data['homepage']}")
        lines.append("")
        lines.append("--- Publications (sorted by citation count) ---")
        for i, paper in enumerate(data["papers"][:50], 1):
            journal = paper.get("journal")
            venue = paper.get("venue") or (journal.get("name") if isinstance(journal, dict) else "")
            fos = ", ".join(paper.get("fieldsOfStudy") or [])
            lines.append(f"\n{i}. {paper.get('title', 'Untitled')} ({paper.get('year', 'N/A')})")
            lines.append(f"   Citations: {paper.get('citationCount', 0)}")
            if venue:
                lines.append(f"   Venue: {venue}")
            if fos:
                lines.append(f"   Fields: {fos}")
        return "\n".join(lines)
