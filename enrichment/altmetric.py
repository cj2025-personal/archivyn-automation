"""
Altmetric collector — news/policy/social mentions per paper.

API docs: https://api.altmetric.com/
Free for non-commercial use; public endpoint does not require a key for
per-DOI lookups at /v1/doi/{doi}. Include API key via ALTMETRIC_API_KEY
for higher rate limits.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base_collector import BaseCollector, CollectorResult, ProfessorQuery

BASE_URL = "https://api.altmetric.com/v1/doi"


class AltmetricCollector(BaseCollector):
    def __init__(self, **kwargs):
        kwargs.setdefault("rate_limit_delay", 1.0)
        super().__init__(**kwargs)
        self.api_key = os.getenv("ALTMETRIC_API_KEY", "")
        self.max_dois = int(os.getenv("ALTMETRIC_MAX_DOIS", "40"))

    @property
    def source_name(self) -> str:
        return "altmetric"

    async def collect(self, query: ProfessorQuery) -> CollectorResult:
        dois = self._gather_dois(query)[: self.max_dois]
        if not dois:
            return self._make_result(query, success=False, error="No DOIs available")

        records: List[Dict[str, Any]] = []
        for doi in dois:
            params = {"key": self.api_key} if self.api_key else None
            info = await self._get_json(f"{BASE_URL}/{doi}", params=params)
            if not info:
                continue
            records.append({
                "doi": doi,
                "title": info.get("title"),
                "altmetric_score": info.get("score"),
                "cited_by_policies_count": info.get("cited_by_policies_count", 0),
                "cited_by_patents_count": info.get("cited_by_patents_count", 0),
                "cited_by_wikipedia_count": info.get("cited_by_wikipedia_count", 0),
                "cited_by_tweeters_count": info.get("cited_by_tweeters_count", 0),
                "cited_by_fbwalls_count": info.get("cited_by_fbwalls_count", 0),
                "cited_by_msm_count": info.get("cited_by_msm_count", 0),  # mainstream media
                "cited_by_rdts_count": info.get("cited_by_rdts_count", 0),  # reddit
                "cited_by_feeds_count": info.get("cited_by_feeds_count", 0),
                "readers_count": (info.get("readers") or {}).get("mendeley", 0),
                "altmetric_url": info.get("details_url"),
            })

        if not records:
            return self._make_result(query, success=False, error="No Altmetric data for any DOI")

        # Aggregate
        totals = {
            "policy": sum(r["cited_by_policies_count"] for r in records),
            "patents": sum(r["cited_by_patents_count"] for r in records),
            "news": sum(r["cited_by_msm_count"] for r in records),
            "wikipedia": sum(r["cited_by_wikipedia_count"] for r in records),
            "tweets": sum(r["cited_by_tweeters_count"] for r in records),
            "reddit": sum(r["cited_by_rdts_count"] for r in records),
        }
        # Top papers by altmetric score
        records.sort(key=lambda r: r.get("altmetric_score") or 0, reverse=True)

        data = {
            "total_papers_with_altmetric": len(records),
            "totals": totals,
            "papers": records,
        }
        return self._make_result(query, success=True, data=data, raw_text=self._to_text(data, query))

    def _gather_dois(self, query: ProfessorQuery) -> List[str]:
        base = Path(os.getenv("ENRICHMENT_OUTPUT_DIR", "output/osu_faculty_run"))
        enrichment_path = base / "profiles" / query.profile_id / "enrichment.json"
        if not enrichment_path.exists():
            return []
        try:
            doc = json.loads(enrichment_path.read_text(encoding="utf-8"))
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
        t = data["totals"]
        lines = [
            f"=== Altmetric impact summary for {query.name} ===",
            f"Papers with Altmetric presence: {data['total_papers_with_altmetric']}",
            f"Aggregate mentions — policy: {t['policy']}, patents: {t['patents']}, "
            f"news: {t['news']}, Wikipedia: {t['wikipedia']}, tweets: {t['tweets']}, reddit: {t['reddit']}",
            "",
            "── Top-scoring papers ──",
        ]
        for p in data["papers"][:20]:
            lines.append(f"• {p['title']} [score: {p.get('altmetric_score')}]")
            lines.append(
                f"  policy={p['cited_by_policies_count']} patents={p['cited_by_patents_count']} "
                f"news={p['cited_by_msm_count']} wiki={p['cited_by_wikipedia_count']}"
            )
            if p.get("altmetric_url"):
                lines.append(f"  Details: {p['altmetric_url']}")
        return "\n".join(lines)
