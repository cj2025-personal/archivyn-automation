"""
ORCID collector.
API docs: https://info.orcid.org/documentation/api-tutorials/

Public API — no key required for reading public profiles.
Returns: canonical career history, education, employment, works, grants, peer reviews.
ORCID is the most authoritative source for structured career data.
"""

import logging
from typing import Any, Dict, List, Optional

from .base_collector import BaseCollector, CollectorResult, ProfessorQuery
from .validation import names_match, has_osu_affiliation

logger = logging.getLogger(__name__)

PUBLIC_API = "https://pub.orcid.org/v3.0"


class ORCIDCollector(BaseCollector):
    """Collect structured career data from ORCID public profiles."""

    def __init__(self, **kwargs):
        # ORCID public API: 24 req/s but multiple calls per professor
        kwargs.setdefault("rate_limit_delay", 1.5)
        super().__init__(**kwargs)

    @property
    def source_name(self) -> str:
        return "orcid"

    async def collect(self, query: ProfessorQuery) -> CollectorResult:
        # Step 1: Search for the ORCID iD
        orcid_id = await self._find_orcid(query)
        if not orcid_id:
            return self._make_result(query, success=False, error="ORCID profile not found")

        # Step 2: Fetch the full record
        record = await self._get_full_record(orcid_id)
        if not record:
            return self._make_result(query, success=False, error="Could not fetch ORCID record")

        # Step 3: Parse into structured data
        data = self._parse_record(orcid_id, record, query)

        # Step 4: Post-validate — reject empty profiles and profiles without OSU employment
        employment = data.get("employment", [])
        has_osu = any(
            has_osu_affiliation(emp.get("institution", ""))
            for emp in employment
        )
        has_content = (
            data.get("publications_count", 0) > 0
            or data.get("biography")
            or len(employment) > 0
        )
        if not has_content:
            return self._make_result(query, success=False, error="ORCID profile is empty — likely wrong person")
        if employment and not has_osu:
            return self._make_result(query, success=False, error="ORCID profile has no Ohio State employment — likely wrong person")

        raw_text = self._to_text(data, query)
        return self._make_result(query, success=True, data=data, raw_text=raw_text)

    async def _find_orcid(self, query: ProfessorQuery) -> Optional[str]:
        """Search ORCID for this professor.

        Strategy:
        1. Search with name + OSU affiliation (most reliable)
        2. Search with name only, but require OSU in institution list
        Never return a result that can't be verified as OSU-affiliated.
        """
        search_url = f"{PUBLIC_API}/expanded-search/"

        for search_query in [
            f'family-name:{query.last_name} AND given-names:{query.first_name} AND affiliation-org-name:"Ohio State"',
            f'family-name:{query.last_name} AND given-names:{query.first_name}',
        ]:
            try:
                client = await self.get_client()
                await self._rate_limit()
                resp = await client.get(
                    search_url,
                    params={"q": search_query, "rows": 10},
                    headers={"Accept": "application/json"},
                )
                if resp.status_code != 200:
                    continue
                data = resp.json()
            except Exception as e:
                logger.warning("[orcid] Search error: %s", e)
                continue

            results = data.get("expanded-result", [])
            if not results:
                continue

            # Strict: require both name match AND OSU affiliation
            for r in results:
                given = (r.get("given-names") or "").strip()
                family = (r.get("family-names") or "").strip()
                found_name = f"{given} {family}"

                if not names_match(query.first_name, query.last_name, found_name):
                    continue

                institutions = r.get("institution-name", [])
                if any("ohio state" in (inst or "").lower() for inst in institutions):
                    return r.get("orcid-id")

        return None

    async def _get_full_record(self, orcid_id: str) -> Optional[Dict]:
        """Fetch the complete ORCID public record."""
        try:
            client = await self.get_client()
            await self._rate_limit()
            resp = await client.get(
                f"{PUBLIC_API}/{orcid_id}/record",
                headers={"Accept": "application/json"},
            )
            if resp.status_code != 200:
                return None
            return resp.json()
        except Exception as e:
            logger.warning("[orcid] Record fetch error for %s: %s", orcid_id, e)
            return None

    def _parse_record(self, orcid_id: str, record: Dict, query: ProfessorQuery) -> Dict:
        """Parse the full ORCID record into structured data."""

        # --- Biography ---
        person = record.get("person", {})
        bio_data = person.get("biography", {})
        biography = (bio_data.get("content") or "") if bio_data else ""

        # --- Education ---
        edu_section = record.get("activities-summary", {}).get("educations", {})
        education = []
        for group in edu_section.get("affiliation-group", []):
            for summary in group.get("summaries", []):
                edu = summary.get("education-summary", {})
                org = edu.get("organization", {})
                education.append({
                    "institution": org.get("name", ""),
                    "department": edu.get("department-name", ""),
                    "role": edu.get("role-title", ""),
                    "start_year": self._extract_year(edu.get("start-date")),
                    "end_year": self._extract_year(edu.get("end-date")),
                })

        # --- Employment ---
        emp_section = record.get("activities-summary", {}).get("employments", {})
        employment = []
        for group in emp_section.get("affiliation-group", []):
            for summary in group.get("summaries", []):
                emp = summary.get("employment-summary", {})
                org = emp.get("organization", {})
                employment.append({
                    "institution": org.get("name", ""),
                    "department": emp.get("department-name", ""),
                    "role": emp.get("role-title", ""),
                    "start_year": self._extract_year(emp.get("start-date")),
                    "end_year": self._extract_year(emp.get("end-date")),
                })

        # --- Works (publications) ---
        works_section = record.get("activities-summary", {}).get("works", {})
        publications = []
        for group in works_section.get("group", [])[:100]:
            summaries = group.get("work-summary", [])
            if not summaries:
                continue
            w = summaries[0]
            title_obj = w.get("title", {})
            title_val = title_obj.get("title", {}).get("value", "") if title_obj else ""
            journal = (w.get("journal-title") or {}).get("value", "") if isinstance(w.get("journal-title"), dict) else ""

            ext_ids = {}
            for eid in (w.get("external-ids", {}).get("external-id", []) or []):
                ext_ids[eid.get("external-id-type", "")] = eid.get("external-id-value", "")

            publications.append({
                "title": title_val,
                "type": w.get("type", ""),
                "year": self._extract_year(w.get("publication-date")),
                "journal": journal,
                "doi": ext_ids.get("doi", ""),
                "put_code": w.get("put-code"),
            })

        # --- Fundings ---
        fund_section = record.get("activities-summary", {}).get("fundings", {})
        grants = []
        for group in fund_section.get("group", []):
            for summary in group.get("funding-summary", []):
                org = summary.get("organization", {})
                title_obj = summary.get("title", {})
                title_val = title_obj.get("title", {}).get("value", "") if title_obj else ""
                grants.append({
                    "title": title_val,
                    "type": summary.get("type", ""),
                    "funder": org.get("name", ""),
                    "start_year": self._extract_year(summary.get("start-date")),
                    "end_year": self._extract_year(summary.get("end-date")),
                })

        # --- Peer Reviews ---
        pr_section = record.get("activities-summary", {}).get("peer-reviews", {})
        peer_review_count = 0
        peer_review_orgs = set()
        for group in pr_section.get("group", []):
            for inner_group in group.get("peer-review-group", []):
                for summary in inner_group.get("peer-review-summary", []):
                    peer_review_count += 1
                    org = (summary.get("convening-organization") or {}).get("name", "")
                    if org:
                        peer_review_orgs.add(org)

        return {
            "orcid_id": orcid_id,
            "orcid_url": f"https://orcid.org/{orcid_id}",
            "biography": biography,
            "education": education,
            "employment": employment,
            "publications_count": len(publications),
            "publications": publications[:80],
            "grants": grants,
            "peer_review_count": peer_review_count,
            "peer_review_organizations": sorted(peer_review_orgs),
        }

    @staticmethod
    def _extract_year(date_obj: Optional[Dict]) -> Optional[str]:
        if not date_obj or not isinstance(date_obj, dict):
            return None
        year = date_obj.get("year", {})
        if isinstance(year, dict):
            return year.get("value")
        return str(year) if year else None

    def _to_text(self, data: Dict, query: ProfessorQuery) -> str:
        lines = []
        lines.append(f"=== ORCID Profile: {query.name} ({data['orcid_id']}) ===")
        lines.append(f"ORCID URL: {data['orcid_url']}")

        if data.get("biography"):
            lines.append(f"\nBiography: {data['biography']}")

        if data["education"]:
            lines.append("\n--- Education ---")
            for edu in data["education"]:
                period = ""
                if edu.get("start_year"):
                    period = f" ({edu['start_year']}-{edu.get('end_year', 'present')})"
                lines.append(f"  - {edu.get('role', 'Degree')}, {edu['institution']}{period}")
                if edu.get("department"):
                    lines.append(f"    Department: {edu['department']}")

        if data["employment"]:
            lines.append("\n--- Employment History ---")
            for emp in data["employment"]:
                period = ""
                if emp.get("start_year"):
                    period = f" ({emp['start_year']}-{emp.get('end_year', 'present')})"
                lines.append(f"  - {emp.get('role', 'Position')}, {emp['institution']}{period}")
                if emp.get("department"):
                    lines.append(f"    Department: {emp['department']}")

        if data["grants"]:
            lines.append(f"\n--- Grants ({len(data['grants'])} total) ---")
            for g in data["grants"][:20]:
                period = ""
                if g.get("start_year"):
                    period = f" ({g['start_year']}-{g.get('end_year', '')})"
                lines.append(f"  - {g['title']}{period}")
                lines.append(f"    Funder: {g.get('funder', 'N/A')}, Type: {g.get('type', 'N/A')}")

        if data["peer_review_count"] > 0:
            lines.append(f"\n--- Peer Review Activity ---")
            lines.append(f"Total reviews: {data['peer_review_count']}")
            if data["peer_review_organizations"]:
                lines.append(f"Organizations: {', '.join(data['peer_review_organizations'][:15])}")

        lines.append(f"\n--- Publications ({data['publications_count']} total) ---")
        for i, pub in enumerate(data["publications"][:40], 1):
            year = pub.get("year") or "N/A"
            lines.append(f"  {i}. {pub['title']} ({year})")
            if pub.get("journal"):
                lines.append(f"     Journal: {pub['journal']}")
            if pub.get("doi"):
                lines.append(f"     DOI: {pub['doi']}")

        return "\n".join(lines)
