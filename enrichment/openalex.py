"""
OpenAlex collector.
API docs: https://docs.openalex.org/

Completely free, no API key required. Polite pool: include email in params.
Returns: publications, concepts, institutions, citation networks, co-authors.
"""

import logging
import os
from typing import Any, Dict, List, Optional

from .base_collector import BaseCollector, CollectorResult, ProfessorQuery
from .validation import names_match_from_query

logger = logging.getLogger(__name__)

BASE_URL = "https://api.openalex.org"


class OpenAlexCollector(BaseCollector):
    """Collect structured academic data from OpenAlex."""

    def __init__(self, **kwargs):
        # OpenAlex polite pool (with email): 10 req/s — 1s is fine
        kwargs.setdefault("rate_limit_delay", 1.0)
        super().__init__(**kwargs)
        # Adding email moves us to the polite pool (faster, no rate limits)
        self.mailto = os.getenv("OPENALEX_EMAIL", "")

    @property
    def source_name(self) -> str:
        return "openalex"

    def _polite_params(self, params: Optional[Dict] = None) -> Dict:
        p = dict(params or {})
        if self.mailto:
            p["mailto"] = self.mailto
        return p

    async def collect(self, query: ProfessorQuery) -> CollectorResult:
        # Step 1: Find the author
        author = await self._find_author(query)
        if not author:
            return self._make_result(query, success=False, error="Author not found on OpenAlex")

        author_id = author["id"]

        # Step 2: Get top works (up to 100, sorted by cited_by_count)
        works = await self._get_works(author_id, limit=100)

        # Step 3: Get co-authors
        coauthors = self._extract_coauthors(works, query.name)

        # Step 4: Extract concepts/topics
        concepts = author.get("x_concepts") or author.get("topics") or []

        data = {
            "openalex_id": author_id,
            "display_name": author.get("display_name", query.name),
            "orcid": author.get("orcid"),
            "works_count": author.get("works_count", 0),
            "cited_by_count": author.get("cited_by_count", 0),
            "h_index": (author.get("summary_stats") or {}).get("h_index"),
            "i10_index": (author.get("summary_stats") or {}).get("i10_index"),
            "2yr_mean_citedness": (author.get("summary_stats") or {}).get("2yr_mean_citedness"),
            "institutions": self._extract_institutions(author),
            "concepts": [
                {"name": c.get("display_name", ""), "score": c.get("score", 0)}
                for c in concepts[:30]
            ],
            "works": works,
            "coauthors": coauthors[:30],
        }

        raw_text = self._to_text(data, query)
        return self._make_result(query, success=True, data=data, raw_text=raw_text)

    async def _find_author(self, query: ProfessorQuery) -> Optional[Dict]:
        """Search for author, disambiguate by institution.

        Strict matching:
        1. Search with OSU institution filter first — verify name matches
        2. Fallback: search by name only, require OSU in institutions
        Never return a result without OSU affiliation verification.
        """
        # Try name + institution filter
        params = self._polite_params({
            "search": query.name,
            "filter": "last_known_institutions.display_name:Ohio State University",
            "select": "id,display_name,orcid,works_count,cited_by_count,summary_stats,"
                      "affiliations,last_known_institutions,x_concepts,topics",
        })
        resp = await self._get_json(f"{BASE_URL}/authors", params=params)
        if resp and resp.get("results"):
            # Verify the top result's name actually matches (first+last)
            for author in resp["results"]:
                found_name = author.get("display_name", "")
                if names_match_from_query(query, found_name):
                    return author

        # Fallback: search by name only, but require OSU affiliation
        params = self._polite_params({
            "search": query.name,
            "select": "id,display_name,orcid,works_count,cited_by_count,summary_stats,"
                      "affiliations,last_known_institutions,x_concepts,topics",
            "per_page": 10,
        })
        resp = await self._get_json(f"{BASE_URL}/authors", params=params)
        if not resp or not resp.get("results"):
            return None

        # Require name match + OSU affiliation
        for author in resp["results"]:
            found_name = author.get("display_name", "")
            if not names_match_from_query(query, found_name):
                continue
            insts = author.get("last_known_institutions") or author.get("affiliations") or []
            for inst in insts:
                inst_name = (inst.get("display_name") or "").lower() if isinstance(inst, dict) else ""
                if "ohio state" in inst_name:
                    return author

        # No match with verified affiliation — return nothing
        return None

    async def _get_works(self, author_id: str, limit: int = 100) -> List[Dict]:
        """Get works by this author, sorted by citation count."""
        # Extract the OpenAlex ID part
        oa_id = author_id.split("/")[-1] if "/" in author_id else author_id
        params = self._polite_params({
            "filter": f"author.id:{oa_id}",
            "sort": "cited_by_count:desc",
            "per_page": min(limit, 100),
            "select": "id,title,publication_year,cited_by_count,type,doi,"
                      "primary_location,abstract_inverted_index,concepts,open_access,"
                      "authorships,biblio",
        })
        resp = await self._get_json(f"{BASE_URL}/works", params=params)
        if not resp or not resp.get("results"):
            return []

        works = []
        for w in resp["results"]:
            # Reconstruct abstract from inverted index
            abstract = self._reconstruct_abstract(w.get("abstract_inverted_index"))
            location = w.get("primary_location") or {}
            source = location.get("source") or {}

            works.append({
                "title": w.get("title", ""),
                "year": w.get("publication_year"),
                "cited_by_count": w.get("cited_by_count", 0),
                "type": w.get("type", ""),
                "doi": w.get("doi", ""),
                "venue": source.get("display_name", ""),
                "is_open_access": (w.get("open_access") or {}).get("is_oa", False),
                "abstract": abstract,
                "concepts": [
                    c.get("display_name", "")
                    for c in (w.get("concepts") or [])[:5]
                ],
            })

        return works

    @staticmethod
    def _extract_institutions(author: Dict) -> List[Dict]:
        """Extract institution info from OpenAlex author data.

        The 'affiliations' field nests institution data under an 'institution' key:
            {"institution": {"display_name": "...", "type": "...", ...}, "years": [...]}
        The 'last_known_institutions' field has it flat:
            {"display_name": "...", "type": "...", "country_code": "..."}
        Handle both formats and deduplicate by name.
        """
        seen_names = set()
        institutions = []

        # Prefer affiliations (more complete history) then last_known_institutions
        raw_affiliations = author.get("affiliations") or []
        last_known = author.get("last_known_institutions") or []

        for item in raw_affiliations:
            if not isinstance(item, dict):
                continue
            # affiliations format: nested under "institution" key
            inst = item.get("institution", item)
            if not isinstance(inst, dict):
                continue
            name = inst.get("display_name") or ""
            if not name or name in seen_names:
                continue
            seen_names.add(name)
            institutions.append({
                "name": name,
                "type": inst.get("type") or "",
                "country": inst.get("country_code") or "",
            })

        for inst in last_known:
            if not isinstance(inst, dict):
                continue
            name = inst.get("display_name") or ""
            if not name or name in seen_names:
                continue
            seen_names.add(name)
            institutions.append({
                "name": name,
                "type": inst.get("type") or "",
                "country": inst.get("country_code") or "",
            })

        return institutions

    @staticmethod
    def _reconstruct_abstract(inverted_index: Optional[Dict]) -> str:
        """Reconstruct abstract from OpenAlex inverted index format."""
        if not inverted_index:
            return ""
        word_positions = []
        for word, positions in inverted_index.items():
            for pos in positions:
                word_positions.append((pos, word))
        word_positions.sort()
        return " ".join(w for _, w in word_positions)

    @staticmethod
    def _extract_coauthors(works: List[Dict], self_name: str) -> List[Dict]:
        """Extract frequent co-authors from works list."""
        # We don't have authorship info in the trimmed works, so this is a stub
        # that can be extended if we fetch full work records
        return []

    def _to_text(self, data: Dict, query: ProfessorQuery) -> str:
        lines = []
        lines.append(f"=== OpenAlex Profile: {data['display_name']} ===")
        if data.get("orcid"):
            lines.append(f"ORCID: {data['orcid']}")
        lines.append(f"Total works: {data['works_count']}")
        lines.append(f"Total citations: {data['cited_by_count']}")
        if data.get("h_index"):
            lines.append(f"h-index: {data['h_index']}")
        if data.get("i10_index"):
            lines.append(f"i10-index: {data['i10_index']}")

        if data["institutions"]:
            lines.append(f"Institutions: {', '.join(i['name'] for i in data['institutions'] if i['name'])}")

        if data["concepts"]:
            lines.append("\n--- Research Concepts/Topics ---")
            for c in data["concepts"][:20]:
                lines.append(f"  - {c['name']} (relevance: {c['score']:.2f})" if isinstance(c['score'], float) else f"  - {c['name']}")

        lines.append("\n--- Publications (sorted by citation count) ---")
        for i, w in enumerate(data["works"][:50], 1):
            lines.append(f"\n{i}. {w['title']} ({w.get('year', 'N/A')})")
            lines.append(f"   Citations: {w['cited_by_count']}")
            if w.get("venue"):
                lines.append(f"   Venue: {w['venue']}")
            if w.get("doi"):
                lines.append(f"   DOI: {w['doi']}")
            if w.get("concepts"):
                lines.append(f"   Topics: {', '.join(w['concepts'])}")
            if w.get("abstract"):
                lines.append(f"   Abstract: {w['abstract'][:500]}")

        return "\n".join(lines)
