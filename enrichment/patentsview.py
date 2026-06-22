"""
PatentsView collector — USPTO patents by inventor name.

API docs: https://patentsview.org/apis/api-endpoints
Free, no auth required (key optional for higher limits).

Uses the v1 /inventors endpoint. Pulls patents where the inventor's
name matches and location is Ohio (to narrow down to OSU-affiliated).
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from .base_collector import BaseCollector, CollectorResult, ProfessorQuery

# PatentsView v1 endpoint (search.patentsview.org/api/v1/patent/) returned
# 410 Gone as of 2024-2025 — they now require POST against the new base.
# The currently-live endpoint requires an API key in the X-Api-Key header.
# Without a key the endpoint returns 410; we degrade gracefully.
PATENTS_URL = "https://search.patentsview.org/api/v1/patent/"


class PatentsViewCollector(BaseCollector):
    def __init__(self, **kwargs):
        kwargs.setdefault("rate_limit_delay", 1.5)
        super().__init__(**kwargs)
        self.api_key = os.getenv("PATENTSVIEW_API_KEY", "")

    @property
    def source_name(self) -> str:
        return "patentsview"

    def _headers(self) -> Dict[str, str]:
        h = {"Accept": "application/json", "Content-Type": "application/json"}
        if self.api_key:
            h["X-Api-Key"] = self.api_key
        return h

    async def collect(self, query: ProfessorQuery) -> CollectorResult:
        first = query.first_name
        last = query.last_name
        if not last:
            return self._make_result(query, success=False, error="Need last name for patent search")

        # PatentsView now requires an API key — without one the endpoint 410s.
        if not self.api_key:
            return self._make_result(
                query, success=False,
                error="PATENTSVIEW_API_KEY not set (register at patentsview.org/apis/keyrequest)",
            )

        body = {
            "q": {
                "_and": [
                    {"inventors.inventor_name_first": first},
                    {"inventors.inventor_name_last": last},
                ]
            },
            "f": [
                "patent_id", "patent_title", "patent_abstract", "patent_date",
                "assignees.assignee_organization",
                "inventors.inventor_name_first", "inventors.inventor_name_last",
                "inventors.location_state", "inventors.location_country",
                "cpc_current.cpc_class_id", "cpc_current.cpc_section_id",
            ],
            "o": {"size": 30, "sort": [{"patent_date": "desc"}]},
        }

        await self._rate_limit()
        try:
            client = await self.get_client()
            resp = await client.post(PATENTS_URL, headers=self._headers(), json=body)
            if resp.status_code in (401, 403, 404, 410, 429):
                return self._make_result(
                    query, success=False,
                    error=f"PatentsView HTTP {resp.status_code} — endpoint may have moved or key invalid",
                )
            resp.raise_for_status()
            try:
                data_raw = resp.json()
            except Exception as e:
                return self._make_result(query, success=False, error=f"PatentsView non-JSON response: {e}")
        except Exception as e:
            return self._make_result(query, success=False, error=f"PatentsView error: {e}")

        patents_list = data_raw.get("patents") or []
        if not patents_list:
            return self._make_result(query, success=False, error="No patents found for inventor name")

        # Filter to Ohio-based inventors (likely OSU)
        patents: List[Dict[str, Any]] = []
        for p in patents_list:
            invs = p.get("inventors") or []
            ohio_match = any(
                (i.get("location_state") or "").lower() in ("oh", "ohio")
                for i in invs
            )
            assignees = p.get("assignees") or []
            assignee_str = " ".join((a.get("assignee_organization") or "") for a in assignees).lower()
            osu_assignee = "ohio state" in assignee_str
            if not (ohio_match or osu_assignee):
                continue
            patents.append({
                "patent_id": p.get("patent_id"),
                "title": p.get("patent_title"),
                "abstract": p.get("patent_abstract"),
                "date": p.get("patent_date"),
                "assignees": [a.get("assignee_organization") for a in assignees if a.get("assignee_organization")],
                "cpc_classes": list({
                    (c.get("cpc_class_id") or "") for c in (p.get("cpc_current") or [])
                    if c.get("cpc_class_id")
                }),
            })

        if not patents:
            return self._make_result(query, success=False, error="No Ohio / OSU-affiliated patents matched")

        data = {"total_patents": len(patents), "patents": patents}
        return self._make_result(query, success=True, data=data, raw_text=self._to_text(data, query))

    def _to_text(self, data: Dict, query: ProfessorQuery) -> str:
        lines = [
            f"=== USPTO patents for {query.name} ===",
            f"Total: {data['total_patents']}",
            "",
        ]
        for p in data["patents"]:
            lines.append(f"• {p['title']} — patent {p['patent_id']} ({p.get('date')})")
            if p["assignees"]:
                lines.append(f"  Assignees: {', '.join(p['assignees'])}")
            if p["cpc_classes"]:
                lines.append(f"  CPC classes: {', '.join(p['cpc_classes'])}")
            if p["abstract"]:
                lines.append(f"  Abstract: {p['abstract'][:1000]}")
            lines.append("")
        return "\n".join(lines)
