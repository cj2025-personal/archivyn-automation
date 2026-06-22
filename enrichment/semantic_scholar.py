"""
Semantic Scholar collector.
API docs: https://api.semanticscholar.org/api-docs/graph

Free tier (no key): shared bucket across all unauthenticated clients — gets
429'd aggressively. With an API key: 1 req/sec dedicated per-key limit plus
fewer spurious 429s.

Register for a free key: https://www.semanticscholar.org/product/api → "API Key"
Add to .env as:  SEMANTIC_SCHOLAR_API_KEY=...

Returns: publications, abstracts, citation counts, influential citations,
         co-authors, fields of study, external IDs.
"""

import logging
import os
from typing import Any, Dict, List, Optional

import httpx

from .base_collector import BaseCollector, CollectorResult, ProfessorQuery
from .validation import names_match_from_query, has_osu_affiliation

logger = logging.getLogger(__name__)

BASE_URL = "https://api.semanticscholar.org/graph/v1"

# Fields to request for each paper
PAPER_FIELDS = (
    "paperId,title,abstract,year,citationCount,influentialCitationCount,"
    "venue,publicationTypes,journal,isOpenAccess,openAccessPdf,"
    "fieldsOfStudy,s2FieldsOfStudy,externalIds,url"
)

AUTHOR_FIELDS = (
    "authorId,name,aliases,affiliations,homepage,paperCount,"
    "citationCount,hIndex,url,externalIds"
)


class SemanticScholarCollector(BaseCollector):
    """Collect publication and citation data from Semantic Scholar."""

    def __init__(self, **kwargs):
        # With an API key we get a dedicated 1 req/s bucket — can reduce delay.
        # Without a key, S2 shares one bucket across all anonymous clients and
        # 429s aggressively after ~3 quick calls, so we keep a generous delay.
        self.api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip()
        if self.api_key:
            kwargs.setdefault("rate_limit_delay", 1.2)
        else:
            kwargs.setdefault("rate_limit_delay", 3.0)
        super().__init__(**kwargs)

    @property
    def source_name(self) -> str:
        return "semantic_scholar"

    async def get_client(self) -> httpx.AsyncClient:
        # Override base client so the S2 API key is sent on every request.
        if self._client is None or self._client.is_closed:
            headers = {
                "User-Agent": "Mozilla/5.0 (compatible; AcademicProfileBot/1.0; research-project)",
            }
            if self.api_key:
                headers["x-api-key"] = self.api_key
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                follow_redirects=True,
                headers=headers,
            )
        return self._client

    async def collect(self, query: ProfessorQuery) -> CollectorResult:
        # Step 1: Search for the author
        author = await self._find_author(query)
        if not author:
            return self._make_result(query, success=False, error="Author not found on Semantic Scholar")

        author_id = author["authorId"]

        # Step 2: Get full author details
        author_detail = await self._get_author_detail(author_id)

        # Step 3: Get their papers (up to 100 most cited)
        papers = await self._get_author_papers(author_id, limit=100)

        # Build structured data
        data = {
            "author_id": author_id,
            "name": author_detail.get("name", query.name) if author_detail else query.name,
            "aliases": (author_detail or {}).get("aliases", []),
            "affiliations": (author_detail or {}).get("affiliations", []),
            "homepage": (author_detail or {}).get("homepage"),
            "paper_count": (author_detail or {}).get("paperCount", 0),
            "citation_count": (author_detail or {}).get("citationCount", 0),
            "h_index": (author_detail or {}).get("hIndex"),
            "semantic_scholar_url": (author_detail or {}).get("url", ""),
            "external_ids": (author_detail or {}).get("externalIds", {}),
            "papers": papers,
        }

        # Flatten to text for chunking
        raw_text = self._to_text(data, query)
        return self._make_result(query, success=True, data=data, raw_text=raw_text)

    async def _find_author(self, query: ProfessorQuery) -> Optional[Dict]:
        """Search for author by name, disambiguate by affiliation.

        Matching strategy:
        1. Name must match (first+last)
        2. Prefer authors with Ohio State affiliation (early return)
        3. Fallback: accept best name-only match by paperCount if
           unambiguous or if the top candidate has significantly more papers
        """
        name_matched_no_affil = []

        # Cap to 2 variants max: primary name + last-name fallback.
        # S2 429s aggressively, and each query costs a full rate_limit_delay.
        search_names = [query.name]
        if query.last_name and len(query.last_name) >= 4:
            search_names.append(query.last_name)

        for variant in search_names:
            # S2 API rejects queries shorter than ~4 chars
            if len(variant.strip()) < 4:
                continue
            # Last-name-only searches can return hundreds of candidates;
            # request more for that case so we can disambiguate downstream.
            limit = 20 if variant == query.last_name else 10
            params = {"query": variant, "fields": AUTHOR_FIELDS, "limit": limit}
            resp = await self._get_json(f"{BASE_URL}/author/search", params=params)
            if not resp or not resp.get("data"):
                continue

            candidates = resp["data"]
            for author in candidates:
                found_name = author.get("name") or ""
                if not names_match_from_query(query, found_name):
                    continue

                # Best case: name + OSU affiliation
                affiliations = [a.lower() for a in (author.get("affiliations") or [])]
                if any("ohio state" in a for a in affiliations):
                    logger.info("[semantic_scholar] Matched %s via name + OSU affiliation", query.name)
                    return author

                # Track name-only match as fallback (only if has papers)
                if (author.get("paperCount") or 0) > 0:
                    # Deduplicate by authorId
                    aid = author.get("authorId")
                    if aid and aid not in {a.get("authorId") for a in name_matched_no_affil}:
                        name_matched_no_affil.append(author)

        if not name_matched_no_affil:
            return None

        # Fallback: if exactly one name-matched candidate, accept it
        if len(name_matched_no_affil) == 1:
            author = name_matched_no_affil[0]
            logger.info(
                "[semantic_scholar] Matched %s via name-only fallback (1 candidate, %d papers)",
                query.name, author.get("paperCount", 0),
            )
            return author

        # Multiple candidates: sort by paper count
        sorted_candidates = sorted(
            name_matched_no_affil,
            key=lambda a: a.get("paperCount") or 0,
            reverse=True,
        )
        top = sorted_candidates[0]
        top_count = top.get("paperCount") or 0

        # Heuristic A: if top candidate has an EXACT first-name match to the
        # query (not just initial), trust it. S2 often fragments one person
        # across variants like "J. Volek" (44p), "Jeffrey S Volek" (23p),
        # "Jeff S. Volek" (9p) — they're all the same person, and the initials
        # split the count. Prior "3× dominance" rule failed here.
        from .validation import _split_name, normalize
        q_first_n = normalize(query.first_name)
        for cand in sorted_candidates[:5]:
            cand_first, cand_last = _split_name(cand.get("name") or "")
            if cand_first and cand_first == q_first_n and (cand.get("paperCount") or 0) >= 5:
                logger.info(
                    "[semantic_scholar] Matched %s via exact-first-name fallback (%s, %d papers)",
                    query.name, cand.get("name"), cand.get("paperCount"),
                )
                return cand

        # Heuristic B: 3× dominance kept as a last resort for when first name
        # isn't exact (e.g. query is "Matt" but S2 has "Matthew").
        second = sorted_candidates[1] if len(sorted_candidates) > 1 else {}
        second_count = second.get("paperCount") or 0
        if top_count >= 10 and (second_count == 0 or top_count >= second_count * 3):
            logger.info(
                "[semantic_scholar] Matched %s via best-candidate fallback "
                "(%d papers vs %d for runner-up)",
                query.name, top_count, second_count,
            )
            return top

        # Heuristic C: if the top candidate has a large paper count (>=50)
        # and the queried query.name is strongly contained in the label, accept.
        top_name_norm = normalize(top.get("name") or "")
        q_last_n = normalize(query.last_name)
        if top_count >= 50 and q_last_n in top_name_norm.split():
            logger.info(
                "[semantic_scholar] Matched %s via high-productivity last-name fallback (%s, %d papers)",
                query.name, top.get("name"), top_count,
            )
            return top

        # If department info available, try to disambiguate by field of study
        if query.department:
            dept_lower = query.department.lower()
            for candidate in sorted_candidates[:3]:
                # Check if candidate's external IDs or papers suggest the right field
                papers = await self._get_author_papers(candidate["authorId"], limit=5)
                if papers:
                    fields = set()
                    for p in papers:
                        for f in (p.get("fieldsOfStudy") or []):
                            fields.add(f.lower())
                        for sf in (p.get("s2FieldsOfStudy") or []):
                            fields.add((sf.get("category") or "").lower())
                    # Check if any field overlaps with department keywords
                    dept_words = set(dept_lower.split())
                    if any(w in " ".join(fields) for w in dept_words if len(w) > 3):
                        logger.info(
                            "[semantic_scholar] Matched %s via department disambiguation (%s)",
                            query.name, query.department,
                        )
                        return candidate

        logger.info(
            "[semantic_scholar] %s: %d ambiguous name matches, skipping",
            query.name, len(name_matched_no_affil),
        )
        return None

    async def _get_author_detail(self, author_id: str) -> Optional[Dict]:
        return await self._get_json(
            f"{BASE_URL}/author/{author_id}",
            params={"fields": AUTHOR_FIELDS},
        )

    async def _get_author_papers(self, author_id: str, limit: int = 100) -> List[Dict]:
        """Get papers sorted by citation count."""
        papers = []
        offset = 0
        batch = min(limit, 100)

        while offset < limit:
            resp = await self._get_json(
                f"{BASE_URL}/author/{author_id}/papers",
                params={
                    "fields": PAPER_FIELDS,
                    "limit": batch,
                    "offset": offset,
                },
            )
            if not resp or not resp.get("data"):
                break
            papers.extend(resp["data"])
            if len(resp["data"]) < batch:
                break
            offset += batch

        # Sort by citation count descending
        papers.sort(key=lambda p: p.get("citationCount") or 0, reverse=True)
        return papers

    def _to_text(self, data: Dict, query: ProfessorQuery) -> str:
        """Flatten structured data into readable text for chunking."""
        lines = []
        lines.append(f"=== Semantic Scholar Profile: {data['name']} ===")
        lines.append(f"Affiliations: {', '.join(data['affiliations']) if data['affiliations'] else 'N/A'}")
        lines.append(f"Total papers: {data['paper_count']}")
        lines.append(f"Total citations: {data['citation_count']}")
        if data['h_index']:
            lines.append(f"h-index: {data['h_index']}")
        if data['homepage']:
            lines.append(f"Homepage: {data['homepage']}")
        lines.append("")

        lines.append("--- Publications (sorted by citation count) ---")
        for i, paper in enumerate(data["papers"][:50], 1):
            title = paper.get("title", "Untitled")
            year = paper.get("year", "N/A")
            cites = paper.get("citationCount", 0)
            venue = paper.get("venue") or paper.get("journal", {}).get("name") if isinstance(paper.get("journal"), dict) else paper.get("venue", "")
            abstract = paper.get("abstract", "")
            fos = ", ".join(paper.get("fieldsOfStudy") or [])

            lines.append(f"\n{i}. {title} ({year})")
            lines.append(f"   Citations: {cites}")
            if venue:
                lines.append(f"   Venue: {venue}")
            if fos:
                lines.append(f"   Fields: {fos}")
            if abstract:
                lines.append(f"   Abstract: {abstract[:500]}")

        return "\n".join(lines)
