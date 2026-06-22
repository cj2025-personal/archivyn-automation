"""
CrossRef collector.
API docs: https://api.crossref.org/swagger-ui/index.html

Completely free, no key required (polite pool with email).
Returns: full publication metadata, abstracts, references, funding info,
         license info, citation counts — the authoritative DOI registry.
"""

import logging
import os
from typing import Any, Dict, List, Optional

from .base_collector import BaseCollector, CollectorResult, ProfessorQuery
from .validation import normalize, names_match, has_osu_affiliation

logger = logging.getLogger(__name__)

BASE_URL = "https://api.crossref.org"


class CrossRefCollector(BaseCollector):
    """Collect detailed publication metadata from CrossRef DOI registry."""

    def __init__(self, **kwargs):
        # CrossRef polite pool: ~50 req/s with email, but be conservative
        kwargs.setdefault("rate_limit_delay", 2.0)
        super().__init__(**kwargs)
        self.mailto = os.getenv("CROSSREF_EMAIL", os.getenv("OPENALEX_EMAIL", ""))

    @property
    def source_name(self) -> str:
        return "crossref"

    async def collect(self, query: ProfessorQuery) -> CollectorResult:
        works = await self._search_works(query)
        if not works:
            return self._make_result(query, success=False, error="No CrossRef publications found")

        # Extract funding info across all works
        funders = self._extract_funders(works)

        # Extract subject areas
        subjects = self._extract_subjects(works)

        data = {
            "total_works": len(works),
            "works": works,
            "funders": funders,
            "subjects": subjects,
        }

        raw_text = self._to_text(data, query)
        return self._make_result(query, success=True, data=data, raw_text=raw_text)

    async def _search_works(self, query: ProfessorQuery, rows: int = 100) -> List[Dict]:
        """Search CrossRef for works by this author.

        Validation: Each paper must have an author whose first+last name
        matches the professor. OSU affiliation is preferred but not required
        at query time — post-query validation handles disambiguation.
        """
        params = {
            "query.author": query.name,
            "rows": rows,
            "sort": "is-referenced-by-count",
            "order": "desc",
            "select": (
                "DOI,title,author,abstract,published-print,published-online,"
                "container-title,type,is-referenced-by-count,references-count,"
                "subject,funder,license,publisher,volume,issue,page,"
                "short-container-title"
            ),
        }
        if self.mailto:
            params["mailto"] = self.mailto

        resp = await self._get_json(f"{BASE_URL}/works", params=params)
        if not resp or not resp.get("message", {}).get("items"):
            return []

        works = []
        for item in resp["message"]["items"]:
            # Strict validation: require first+last name AND OSU affiliation
            authors = item.get("author", [])
            author_match = False
            for a in authors:
                given = (a.get("given") or "").strip()
                family = (a.get("family") or "").strip()
                full = f"{given} {family}"
                if not names_match(query.first_name, query.last_name, full):
                    continue
                # Check affiliation on this specific author
                affiliations = [
                    af.get("name", "") for af in (a.get("affiliation") or [])
                ]
                if affiliations and has_osu_affiliation(" ".join(affiliations)):
                    author_match = True
                    break
                # If no affiliation data on this author, still accept if name
                # is a strong match (exact first+last) — some papers lack affiliation
                if not affiliations and normalize(query.first_name) == normalize(given) and normalize(query.last_name) == normalize(family):
                    author_match = True
                    break
            if not author_match:
                continue

            # Extract published date
            pub_date = item.get("published-print", {}) or item.get("published-online", {})
            date_parts = pub_date.get("date-parts", [[]])[0] if pub_date else []
            year = date_parts[0] if date_parts else None

            # Extract abstract (CrossRef uses JATS XML sometimes)
            abstract = (item.get("abstract") or "").strip()
            # Strip JATS XML tags
            if abstract.startswith("<"):
                import re
                abstract = re.sub(r"<[^>]+>", "", abstract).strip()

            # Author list
            author_list = []
            for a in authors[:20]:
                name = f"{a.get('given', '')} {a.get('family', '')}".strip()
                affil = ", ".join(
                    af.get("name", "") for af in (a.get("affiliation") or [])
                )
                author_list.append({"name": name, "affiliation": affil})

            # Funders
            funders = []
            for f in (item.get("funder") or []):
                funders.append({
                    "name": f.get("name", ""),
                    "doi": f.get("DOI", ""),
                    "award": f.get("award", []),
                })

            works.append({
                "doi": item.get("DOI", ""),
                "title": (item.get("title") or [""])[0],
                "year": year,
                "journal": (item.get("container-title") or [""])[0],
                "journal_abbrev": (item.get("short-container-title") or [""])[0],
                "type": item.get("type", ""),
                "citation_count": item.get("is-referenced-by-count", 0),
                "references_count": item.get("references-count", 0),
                "abstract": abstract[:800],
                "authors": author_list,
                "funders": funders,
                "publisher": item.get("publisher", ""),
                "volume": item.get("volume", ""),
                "issue": item.get("issue", ""),
                "page": item.get("page", ""),
                "subjects": item.get("subject", []),
                "license": [
                    lic.get("URL", "") for lic in (item.get("license") or [])
                ],
            })

        return works

    @staticmethod
    def _extract_funders(works: List[Dict]) -> List[Dict]:
        """Aggregate unique funders across all works."""
        funder_map: Dict[str, Dict] = {}
        for w in works:
            for f in w.get("funders", []):
                name = f.get("name", "")
                if not name:
                    continue
                key = name.lower()
                if key not in funder_map:
                    funder_map[key] = {"name": name, "works_count": 0, "awards": set()}
                funder_map[key]["works_count"] += 1
                for award in f.get("award", []):
                    funder_map[key]["awards"].add(award)

        funders = []
        for f in sorted(funder_map.values(), key=lambda x: x["works_count"], reverse=True):
            funders.append({
                "name": f["name"],
                "works_funded": f["works_count"],
                "award_numbers": sorted(f["awards"]),
            })
        return funders

    @staticmethod
    def _extract_subjects(works: List[Dict]) -> List[Dict]:
        """Aggregate subject areas across all works."""
        subject_counts: Dict[str, int] = {}
        for w in works:
            for subj in w.get("subjects", []):
                subject_counts[subj] = subject_counts.get(subj, 0) + 1
        return [
            {"name": name, "count": count}
            for name, count in sorted(subject_counts.items(), key=lambda x: x[1], reverse=True)
        ]

    def _to_text(self, data: Dict, query: ProfessorQuery) -> str:
        lines = []
        lines.append(f"=== CrossRef Publications: {query.name} ===")
        lines.append(f"Total publications found: {data['total_works']}")

        if data["subjects"]:
            lines.append("\n--- Subject Areas ---")
            for s in data["subjects"][:20]:
                lines.append(f"  - {s['name']} ({s['count']} publications)")

        if data["funders"]:
            lines.append("\n--- Funding Sources (from publication records) ---")
            for f in data["funders"][:15]:
                awards = f", awards: {', '.join(f['award_numbers'][:5])}" if f["award_numbers"] else ""
                lines.append(f"  - {f['name']} ({f['works_funded']} publications{awards})")

        lines.append("\n--- Publications (sorted by citation count) ---")
        for i, w in enumerate(data["works"][:50], 1):
            year = w.get("year") or "N/A"
            lines.append(f"\n{i}. {w['title']} ({year})")
            lines.append(f"   DOI: {w['doi']}")
            lines.append(f"   Journal: {w['journal']}")
            lines.append(f"   Citations: {w['citation_count']}, References: {w['references_count']}")
            if w.get("authors"):
                author_names = [a["name"] for a in w["authors"][:6]]
                if len(w["authors"]) > 6:
                    author_names.append(f"... +{len(w['authors'])-6} more")
                lines.append(f"   Authors: {', '.join(author_names)}")
            if w.get("abstract"):
                lines.append(f"   Abstract: {w['abstract'][:500]}")
            if w.get("funders"):
                funder_names = [f["name"] for f in w["funders"]]
                lines.append(f"   Funded by: {', '.join(funder_names)}")

        return "\n".join(lines)
