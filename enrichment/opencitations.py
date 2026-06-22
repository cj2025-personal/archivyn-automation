"""
OpenCitations collector — open citation graph per DOI.

Docs: https://opencitations.net/index/coci/api/v1
Free, no auth. Returns who cites each of the researcher's papers.
Useful supplement to OpenAlex for citation context.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base_collector import BaseCollector, CollectorResult, ProfessorQuery

COUNT_URL = "https://opencitations.net/index/coci/api/v1/citation-count"
REF_URL = "https://opencitations.net/index/coci/api/v1/references"


class OpenCitationsCollector(BaseCollector):
    def __init__(self, **kwargs):
        kwargs.setdefault("rate_limit_delay", 1.0)
        super().__init__(**kwargs)
        self.max_dois = int(os.getenv("OPENCITATIONS_MAX_DOIS", "40"))

    @property
    def source_name(self) -> str:
        return "opencitations"

    async def collect(self, query: ProfessorQuery) -> CollectorResult:
        dois = self._gather_dois(query)[: self.max_dois]
        if not dois:
            return self._make_result(query, success=False, error="No DOIs available")

        records: List[Dict[str, Any]] = []
        total_citations = 0
        for doi in dois:
            cnt = await self._get_json(f"{COUNT_URL}/{doi}")
            count = 0
            if isinstance(cnt, list) and cnt:
                try:
                    count = int(cnt[0].get("count") or 0)
                except (ValueError, TypeError):
                    count = 0
            if count > 0:
                records.append({"doi": doi, "citation_count": count})
                total_citations += count

        if not records:
            return self._make_result(query, success=False, error="No citation data for any DOI")

        records.sort(key=lambda r: r["citation_count"], reverse=True)
        data = {
            "total_dois_checked": len(dois),
            "total_citations": total_citations,
            "top_cited": records[:30],
        }
        return self._make_result(query, success=True, data=data, raw_text=self._to_text(data, query))

    def _gather_dois(self, query: ProfessorQuery) -> List[str]:
        base = Path(os.getenv("ENRICHMENT_OUTPUT_DIR", "output/osu_faculty_run"))
        p = base / "profiles" / query.profile_id / "enrichment.json"
        if not p.exists():
            return []
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return []
        dois: List[str] = []
        srcs = doc.get("sources", {})
        for w in (srcs.get("openalex", {}).get("data", {}).get("works") or []):
            d = w.get("doi")
            if d:
                dois.append(str(d).replace("https://doi.org/", "").lower())
        for w in (srcs.get("crossref", {}).get("data", {}).get("works") or []):
            d = w.get("doi")
            if d:
                dois.append(str(d).lower())
        seen = set()
        return [d for d in dois if not (d in seen or seen.add(d))]

    def _to_text(self, data: Dict, query: ProfessorQuery) -> str:
        lines = [
            f"=== OpenCitations counts for {query.name} ===",
            f"DOIs checked: {data['total_dois_checked']} | Total OCI citations: {data['total_citations']:,}",
            "",
            "── Top-cited (by OpenCitations) ──",
        ]
        for r in data["top_cited"]:
            lines.append(f"• {r['doi']} — {r['citation_count']} citations")
        return "\n".join(lines)
