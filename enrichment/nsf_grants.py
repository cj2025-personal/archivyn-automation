"""
NSF Award Search collector.
API docs: https://www.research.gov/common/webapi/awardapisearch-v1.htm

Completely free, no API key. Returns federal grant data including:
award amounts, abstracts, dates, co-PIs, program info.
"""

import logging
from typing import Any, Dict, List, Optional

from .base_collector import BaseCollector, CollectorResult, ProfessorQuery
from .validation import names_match

logger = logging.getLogger(__name__)

BASE_URL = "https://api.nsf.gov/services/v1/awards.json"


class NSFGrantsCollector(BaseCollector):
    """Collect NSF grant/award data."""

    def __init__(self, **kwargs):
        # NSF API: government API, conservative rate limit
        kwargs.setdefault("rate_limit_delay", 1.5)
        super().__init__(**kwargs)

    @property
    def source_name(self) -> str:
        return "nsf_grants"

    async def collect(self, query: ProfessorQuery) -> CollectorResult:
        awards = await self._search_awards(query)
        if not awards:
            return self._make_result(query, success=False, error="No NSF awards found")

        data = {
            "total_awards": len(awards),
            "total_funding": sum(self._parse_amount(a.get("fundsObligatedAmt", "0")) for a in awards),
            "awards": awards,
        }

        raw_text = self._to_text(data, query)
        return self._make_result(query, success=True, data=data, raw_text=raw_text)

    async def _search_awards(self, query: ProfessorQuery) -> List[Dict]:
        """Search NSF awards by PI name and institution."""
        params = {
            "pdPIName": query.name,
            "awardeeName": "Ohio State",
            "printFields": (
                "id,title,abstractText,startDate,expDate,fundsObligatedAmt,"
                "piFirstName,piLastName,piEmail,coPDPI,awardeeName,"
                "awardeeCity,awardeeStateCode,fundProgramName,primaryProgram,"
                "poName,publicAccessMandate"
            ),
            "offset": 1,
            "rpp": 100,  # results per page
        }

        resp = await self._get_json(BASE_URL, params=params)
        if not resp:
            return []

        response_data = resp.get("response", {})
        awards_raw = response_data.get("award", [])
        if not awards_raw:
            return []

        awards = []
        for a in awards_raw:
            # Verify PI name matches (first+last)
            pi_first = (a.get("piFirstName") or "").strip()
            pi_last = (a.get("piLastName") or "").strip()
            pi_full = f"{pi_first} {pi_last}"
            if not names_match(query.first_name, query.last_name, pi_full):
                # Also check co-PIs
                co_pis = a.get("coPDPI", []) if isinstance(a.get("coPDPI"), list) else []
                is_co_pi = any(
                    names_match(query.first_name, query.last_name, cp)
                    for cp in co_pis if isinstance(cp, str)
                )
                if not is_co_pi:
                    continue
            awards.append({
                "award_id": a.get("id", ""),
                "title": a.get("title", ""),
                "abstract": a.get("abstractText", ""),
                "start_date": a.get("startDate", ""),
                "end_date": a.get("expDate", ""),
                "amount": self._parse_amount(a.get("fundsObligatedAmt", "0")),
                "pi_name": f"{a.get('piFirstName', '')} {a.get('piLastName', '')}".strip(),
                "pi_email": a.get("piEmail", ""),
                "co_pis": a.get("coPDPI", []) if isinstance(a.get("coPDPI"), list) else [],
                "institution": a.get("awardeeName", ""),
                "program": a.get("fundProgramName", "") or a.get("primaryProgram", ""),
                "program_officer": a.get("poName", ""),
            })

        # Sort by amount descending
        awards.sort(key=lambda x: x["amount"], reverse=True)
        return awards

    @staticmethod
    def _parse_amount(val: str) -> int:
        try:
            return int(str(val).replace(",", "").replace("$", "").strip())
        except (ValueError, TypeError):
            return 0

    def _to_text(self, data: Dict, query: ProfessorQuery) -> str:
        lines = []
        lines.append(f"=== NSF Grants: {query.name} ===")
        lines.append(f"Total NSF awards: {data['total_awards']}")
        lines.append(f"Total NSF funding: ${data['total_funding']:,}")
        lines.append("")

        for i, award in enumerate(data["awards"], 1):
            lines.append(f"\n{i}. {award['title']}")
            lines.append(f"   Award ID: {award['award_id']}")
            lines.append(f"   Amount: ${award['amount']:,}")
            lines.append(f"   Period: {award['start_date']} to {award['end_date']}")
            if award.get("program"):
                lines.append(f"   Program: {award['program']}")
            if award.get("co_pis"):
                co_pi_str = ", ".join(award["co_pis"]) if isinstance(award["co_pis"], list) else str(award["co_pis"])
                lines.append(f"   Co-PIs: {co_pi_str}")
            if award.get("abstract"):
                lines.append(f"   Abstract: {award['abstract'][:600]}")

        return "\n".join(lines)
