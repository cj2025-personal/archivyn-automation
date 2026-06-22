"""
ClinicalTrials.gov collector — trials where the researcher is PI / investigator.

API docs: https://clinicaltrials.gov/data-api/api (v2).
Free, no auth. Returns structured JSON with trial protocols, outcomes,
sponsors, collaborators.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base_collector import BaseCollector, CollectorResult, ProfessorQuery

BASE_URL = "https://clinicaltrials.gov/api/v2/studies"


class ClinicalTrialsCollector(BaseCollector):
    def __init__(self, **kwargs):
        kwargs.setdefault("rate_limit_delay", 1.0)
        super().__init__(**kwargs)

    @property
    def source_name(self) -> str:
        return "clinicaltrials"

    async def collect(self, query: ProfessorQuery) -> CollectorResult:
        params = {
            "query.term": f'"{query.name}" AND "Ohio State"',
            "pageSize": 50,
            "fields": (
                "NCTId,BriefTitle,OfficialTitle,OverallStatus,StartDate,CompletionDate,"
                "LeadSponsorName,Phase,StudyType,BriefSummary,Condition,"
                "OverallOfficialName,OverallOfficialAffiliation,OverallOfficialRole,"
                "Keyword"
            ),
            "format": "json",
        }
        resp = await self._get_json(BASE_URL, params=params)
        if not resp:
            return self._make_result(query, success=False, error="ClinicalTrials returned empty")

        studies = resp.get("studies") or []
        trials: List[Dict[str, Any]] = []
        last = query.last_name.lower()

        for s in studies:
            proto = s.get("protocolSection", {})
            id_mod = proto.get("identificationModule", {})
            status_mod = proto.get("statusModule", {})
            sponsor_mod = proto.get("sponsorCollaboratorsModule", {})
            desc_mod = proto.get("descriptionModule", {})
            cond_mod = proto.get("conditionsModule", {})
            design_mod = proto.get("designModule", {})
            contacts = proto.get("contactsLocationsModule", {}).get("overallOfficials") or []

            # Verify the professor appears as an official
            officials_matched = [
                o for o in contacts
                if last and last in (o.get("name") or "").lower()
            ]
            if not officials_matched:
                continue

            trials.append({
                "nct_id": id_mod.get("nctId"),
                "brief_title": id_mod.get("briefTitle"),
                "official_title": id_mod.get("officialTitle"),
                "status": status_mod.get("overallStatus"),
                "start_date": (status_mod.get("startDateStruct") or {}).get("date"),
                "completion_date": (status_mod.get("completionDateStruct") or {}).get("date"),
                "lead_sponsor": (sponsor_mod.get("leadSponsor") or {}).get("name"),
                "phase": design_mod.get("phases"),
                "study_type": design_mod.get("studyType"),
                "brief_summary": desc_mod.get("briefSummary"),
                "conditions": cond_mod.get("conditions"),
                "overall_officials": [
                    {
                        "name": o.get("name"),
                        "affiliation": o.get("affiliation"),
                        "role": o.get("role"),
                    } for o in officials_matched
                ],
            })

        if not trials:
            return self._make_result(query, success=False, error="No trials matched this investigator")

        data = {"total_trials": len(trials), "trials": trials}
        return self._make_result(query, success=True, data=data, raw_text=self._to_text(data, query))

    def _to_text(self, data: Dict, query: ProfessorQuery) -> str:
        lines = [
            f"=== ClinicalTrials.gov studies with {query.name} as investigator ===",
            f"Total: {data['total_trials']}",
            "",
        ]
        for t in data["trials"]:
            lines.append(f"• {t['brief_title']} ({t['nct_id']})")
            lines.append(f"  Status: {t['status']} | Phase: {t.get('phase')} | Type: {t.get('study_type')}")
            lines.append(f"  Sponsor: {t.get('lead_sponsor')} | Start: {t.get('start_date')}")
            if t.get("conditions"):
                lines.append(f"  Conditions: {', '.join(t['conditions']) if isinstance(t['conditions'], list) else t['conditions']}")
            if t.get("brief_summary"):
                lines.append(f"  Summary: {t['brief_summary'][:1200]}")
            lines.append("")
        return "\n".join(lines)
