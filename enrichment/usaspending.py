"""
USAspending.gov collector — federal grants/contracts beyond NIH/NSF.

Covers DOE, DOD, USDA, EPA, NASA, NEH, IMLS, etc. Free, no auth.
API docs: https://api.usaspending.gov/

Strategy: search awards by recipient name + place of performance (OSU in Ohio).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base_collector import BaseCollector, CollectorResult, ProfessorQuery

SEARCH_URL = "https://api.usaspending.gov/api/v2/search/spending_by_award/"


class USASpendingCollector(BaseCollector):
    def __init__(self, **kwargs):
        kwargs.setdefault("rate_limit_delay", 1.0)
        super().__init__(**kwargs)

    @property
    def source_name(self) -> str:
        return "usaspending"

    async def collect(self, query: ProfessorQuery) -> CollectorResult:
        # USAspending.gov awards are listed to institutions, not PIs,
        # so we search for grants to OSU and filter by PI name in the award
        # description / PI keywords.
        body = {
            "filters": {
                "keywords": [query.name],
                "recipient_search_text": ["Ohio State University"],
                "award_type_codes": ["02", "03", "04", "05"],  # Grants
            },
            "fields": [
                "Award ID", "Recipient Name", "Awarding Agency",
                "Awarding Sub Agency", "Award Amount", "Description",
                "Period of Performance Start Date", "Period of Performance Current End Date",
            ],
            "page": 1,
            "limit": 50,
            "sort": "Award Amount",
            "order": "desc",
        }
        resp = await self._post_json(SEARCH_URL, json_body=body)
        if not resp:
            return self._make_result(query, success=False, error="USAspending returned empty")

        awards: List[Dict[str, Any]] = []
        total_amount = 0.0

        for r in resp.get("results", []):
            desc = (r.get("Description") or "").lower()
            # Keep only awards where the professor's last name appears in description
            if query.last_name and query.last_name.lower() not in desc:
                continue
            amount = r.get("Award Amount") or 0
            try:
                amount = float(amount)
            except (TypeError, ValueError):
                amount = 0.0
            total_amount += amount
            awards.append({
                "award_id": r.get("Award ID"),
                "recipient": r.get("Recipient Name"),
                "agency": r.get("Awarding Agency"),
                "sub_agency": r.get("Awarding Sub Agency"),
                "amount": amount,
                "description": r.get("Description"),
                "start_date": r.get("Period of Performance Start Date"),
                "end_date": r.get("Period of Performance Current End Date"),
            })

        if not awards:
            return self._make_result(query, success=False, error="No matching USAspending awards")

        data = {
            "total_awards": len(awards),
            "total_funding_usd": total_amount,
            "awards": awards,
        }
        return self._make_result(query, success=True, data=data, raw_text=self._to_text(data, query))

    def _to_text(self, data: Dict, query: ProfessorQuery) -> str:
        lines = [
            f"=== USAspending.gov federal awards for {query.name} ===",
            f"Total awards: {data['total_awards']} | Total funding: ${data['total_funding_usd']:,.0f}",
            "",
        ]
        for a in data["awards"]:
            lines.append(f"• {a['agency']} / {a.get('sub_agency') or ''} — ${a['amount']:,.0f}")
            lines.append(f"  Award ID: {a['award_id']} | {a.get('start_date', '')} → {a.get('end_date', '')}")
            if a.get("description"):
                lines.append(f"  Description: {a['description'][:600]}")
            lines.append("")
        return "\n".join(lines)
