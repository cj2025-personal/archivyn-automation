"""
Unpaywall collector — legal OA version lookup for DOIs.

API docs: https://unpaywall.org/products/api
Free with email identification. Returns OA locations (repository / publisher /
preprint) for each DOI, so we can fetch the author's own self-archived copy
instead of paywalled PDFs.

Strategy: feed DOIs we already have (from openalex / crossref / semantic_scholar)
and resolve the best OA URL for each. Raw text is a list of
"title → OA pdf url" lines; actual PDF fetching is left to downstream chunker.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base_collector import BaseCollector, CollectorResult, ProfessorQuery

BASE_URL = "https://api.unpaywall.org/v2"


class UnpaywallCollector(BaseCollector):
    def __init__(self, **kwargs):
        kwargs.setdefault("rate_limit_delay", 0.2)
        super().__init__(**kwargs)
        self.email = os.getenv("UNPAYWALL_EMAIL") or os.getenv("OPENALEX_EMAIL") or "research@example.com"
        # Cap DOI lookups per professor to avoid thousands of calls
        self.max_dois = int(os.getenv("UNPAYWALL_MAX_DOIS", "60"))

    @property
    def source_name(self) -> str:
        return "unpaywall"

    async def collect(self, query: ProfessorQuery) -> CollectorResult:
        dois = self._gather_dois(query)
        if not dois:
            return self._make_result(
                query, success=False,
                error="No DOIs available from prior enrichment; run openalex/crossref first",
            )

        dois = dois[: self.max_dois]
        oa_records: List[Dict[str, Any]] = []

        for doi in dois:
            info = await self._get_json(f"{BASE_URL}/{doi}", params={"email": self.email})
            if not info:
                continue
            best_loc = info.get("best_oa_location") or {}
            if not best_loc:
                continue
            oa_records.append({
                "doi": doi,
                "title": info.get("title", ""),
                "year": info.get("year"),
                "is_oa": info.get("is_oa", False),
                "oa_status": info.get("oa_status"),
                "journal": info.get("journal_name"),
                "genre": info.get("genre"),
                "best_oa_url": best_loc.get("url"),
                "best_oa_pdf": best_loc.get("url_for_pdf"),
                "host_type": best_loc.get("host_type"),
                "license": best_loc.get("license"),
                "version": best_loc.get("version"),
                "repository": best_loc.get("repository_institution"),
            })

        if not oa_records:
            return self._make_result(query, success=False, error="No OA locations found for any DOI")

        data = {
            "total_dois_checked": len(dois),
            "total_oa_found": len(oa_records),
            "oa_papers": oa_records,
        }
        return self._make_result(query, success=True, data=data, raw_text=self._to_text(data, query))

    def _gather_dois(self, query: ProfessorQuery) -> List[str]:
        """Pull DOIs from prior enrichment sources (openalex/crossref/semantic_scholar)."""
        base = Path(os.getenv("ENRICHMENT_OUTPUT_DIR", "output/osu_faculty_run"))
        enrichment_path = base / "profiles" / query.profile_id / "enrichment.json"
        if not enrichment_path.exists():
            return []
        try:
            doc = json.loads(enrichment_path.read_text(encoding="utf-8"))
        except Exception:
            return []

        dois: List[str] = []
        sources = doc.get("sources", {})

        # openalex works
        for w in (sources.get("openalex", {}).get("data", {}).get("works") or []):
            doi = w.get("doi")
            if doi:
                dois.append(str(doi).replace("https://doi.org/", ""))

        # crossref works
        for w in (sources.get("crossref", {}).get("data", {}).get("works") or []):
            doi = w.get("doi")
            if doi:
                dois.append(str(doi))

        # semantic_scholar papers
        for p in (sources.get("semantic_scholar", {}).get("data", {}).get("papers") or []):
            ext = p.get("externalIds") or p.get("external_ids") or {}
            if isinstance(ext, dict) and ext.get("DOI"):
                dois.append(str(ext["DOI"]))

        # Dedupe, preserve order, lowercase
        seen = set()
        out: List[str] = []
        for d in dois:
            d = d.strip().lower()
            if d and d not in seen:
                seen.add(d)
                out.append(d)
        return out

    def _to_text(self, data: Dict, query: ProfessorQuery) -> str:
        lines = [
            f"=== Unpaywall OA lookup for {query.name} ===",
            f"DOIs checked: {data['total_dois_checked']}, OA versions found: {data['total_oa_found']}",
            "",
        ]
        for r in data["oa_papers"]:
            lines.append(f"• {r['title']} ({r.get('year') or 'n/a'})")
            lines.append(f"  DOI: {r['doi']} | OA status: {r.get('oa_status')}")
            if r.get("best_oa_pdf"):
                lines.append(f"  OA PDF: {r['best_oa_pdf']}")
            elif r.get("best_oa_url"):
                lines.append(f"  OA page: {r['best_oa_url']}")
            if r.get("repository"):
                lines.append(f"  Repository: {r['repository']}")
            lines.append("")
        return "\n".join(lines)
