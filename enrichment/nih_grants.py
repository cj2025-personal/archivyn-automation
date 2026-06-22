"""
NIH RePORTER collector.
API docs: https://api.reporter.nih.gov/

Completely free, no API key. Returns NIH-funded grants including:
project details, abstracts, funding amounts, co-investigators.
"""

import logging
from typing import Any, Dict, List, Optional

from .base_collector import BaseCollector, CollectorResult, ProfessorQuery
from .validation import names_match

logger = logging.getLogger(__name__)

BASE_URL = "https://api.reporter.nih.gov/v2/projects/search"


class NIHGrantsCollector(BaseCollector):
    """Collect NIH grant data from the RePORTER API."""

    def __init__(self, **kwargs):
        # NIH RePORTER: government API, conservative rate limit
        kwargs.setdefault("rate_limit_delay", 1.5)
        super().__init__(**kwargs)

    @property
    def source_name(self) -> str:
        return "nih_grants"

    async def collect(self, query: ProfessorQuery) -> CollectorResult:
        grants = await self._search_grants(query)
        if not grants:
            return self._make_result(query, success=False, error="No NIH grants found")

        data = {
            "total_grants": len(grants),
            "total_funding": sum(g.get("amount", 0) for g in grants),
            "grants": grants,
        }

        raw_text = self._to_text(data, query)
        return self._make_result(query, success=True, data=data, raw_text=raw_text)

    async def _search_grants(self, query: ProfessorQuery) -> List[Dict]:
        """Search NIH RePORTER by PI name and organization."""
        body = {
            "criteria": {
                "pi_names": [
                    {
                        "first_name": query.first_name,
                        "last_name": query.last_name,
                        "any_name": "",
                    }
                ],
                "org_names": ["Ohio State University"],
            },
            "offset": 0,
            "limit": 100,
            "sort_field": "award_amount",
            "sort_order": "desc",
        }

        resp = await self._post_json(BASE_URL, json_body=body)
        if not resp:
            return []

        results = resp.get("results", [])
        if not results:
            return []

        grants = []
        for r in results:
            pi_names = []
            for pi in (r.get("principal_investigators") or []):
                pi_name = f"{pi.get('first_name', '')} {pi.get('last_name', '')}".strip()
                if pi_name:
                    pi_names.append(pi_name)

            # Verify professor is actually a PI on this grant
            is_pi = any(
                names_match(query.first_name, query.last_name, pn)
                for pn in pi_names
            )
            if not is_pi:
                continue

            org = r.get("organization") or {}
            grants.append({
                "project_num": r.get("project_num", ""),
                "title": (r.get("project_title") or "").strip(),
                "abstract": (r.get("abstract_text") or "").strip(),
                "fiscal_year": r.get("fiscal_year"),
                "award_amount": r.get("award_amount") or 0,
                "amount": r.get("award_amount") or 0,
                "start_date": r.get("project_start_date", ""),
                "end_date": r.get("project_end_date", ""),
                "principal_investigators": pi_names,
                "organization": org.get("org_name", ""),
                "department": org.get("dept_name", ""),
                "activity_code": r.get("activity_code", ""),
                "funding_mechanism": r.get("full_foa", ""),
                "nih_institute": (r.get("agency_ic_fundings") or [{}])[0].get("name", "") if r.get("agency_ic_fundings") else "",
            })

        grants.sort(key=lambda x: x["award_amount"], reverse=True)
        return grants

    def _to_text(self, data: Dict, query: ProfessorQuery) -> str:
        lines = []
        lines.append(f"=== NIH Grants: {query.name} ===")
        lines.append(f"Total NIH grants: {data['total_grants']}")
        lines.append(f"Total NIH funding: ${data['total_funding']:,}")
        lines.append("")

        for i, grant in enumerate(data["grants"], 1):
            lines.append(f"\n{i}. {grant['title']}")
            lines.append(f"   Project #: {grant['project_num']}")
            if grant["award_amount"]:
                lines.append(f"   Amount: ${grant['award_amount']:,}")
            lines.append(f"   Fiscal year: {grant.get('fiscal_year', 'N/A')}")
            if grant.get("start_date"):
                lines.append(f"   Period: {grant['start_date']} to {grant.get('end_date', 'ongoing')}")
            if grant.get("principal_investigators"):
                lines.append(f"   PIs: {', '.join(grant['principal_investigators'])}")
            if grant.get("department"):
                lines.append(f"   Department: {grant['department']}")
            if grant.get("nih_institute"):
                lines.append(f"   NIH Institute: {grant['nih_institute']}")
            if grant.get("abstract"):
                lines.append(f"   Abstract: {grant['abstract'][:600]}")

        return "\n".join(lines)
