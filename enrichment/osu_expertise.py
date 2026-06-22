"""
OSU Research Expertise / Knowledge Bank collector.

Scrapes:
1. OSU Experts Directory (https://experts.osu.edu) — structured research expertise profiles
2. OSU Knowledge Bank (https://kb.osu.edu) — institutional repository of publications,
   theses, dissertations, and reports authored by OSU faculty.
"""

import logging
import re
from typing import Any, Dict, List, Optional

from .base_collector import BaseCollector, CollectorResult, ProfessorQuery
from .validation import names_match_from_query

logger = logging.getLogger(__name__)

EXPERTS_SEARCH = "https://experts.osu.edu"
KNOWLEDGE_BANK_API = "https://kb.osu.edu/server/api"


class OSUExpertiseCollector(BaseCollector):
    """Collect data from OSU Experts directory and Knowledge Bank."""

    def __init__(self, **kwargs):
        # OSU Experts + Knowledge Bank: be respectful to university servers.
        # Both sites sporadically 403 non-browser UAs / return cf-challenge
        # pages under load, so wire through curl_cffi bypass ladder.
        kwargs.setdefault("rate_limit_delay", 2.5)
        kwargs.setdefault("timeout", 45.0)
        kwargs.setdefault("bypass_tier", "curl_cffi")
        super().__init__(**kwargs)

    @property
    def source_name(self) -> str:
        return "osu_expertise"

    async def collect(self, query: ProfessorQuery) -> CollectorResult:
        # Collect from both sources in parallel
        import asyncio
        experts_task = self._search_experts(query)
        kb_task = self._search_knowledge_bank(query)
        experts_data, kb_data = await asyncio.gather(experts_task, kb_task)

        if not experts_data and not kb_data:
            return self._make_result(query, success=False, error="Not found in OSU Experts or Knowledge Bank")

        data = {
            "experts_profile": experts_data,
            "knowledge_bank": kb_data,
        }

        raw_text = self._to_text(data, query)
        return self._make_result(query, success=True, data=data, raw_text=raw_text)

    async def _search_experts(self, query: ProfessorQuery) -> Optional[Dict]:
        """Search OSU Experts directory for the professor's profile.

        Strategy:
        1. Try the Pure REST API first (structured JSON, most reliable)
        2. Fall back to HTML scraping of /en/persons/ search page
        """
        # Strategy 1: Pure REST API (available on many Pure/Elsevier portals)
        profile = await self._search_experts_api(query)

        # Strategy 2: HTML scraping fallback (bypass-aware)
        if not profile:
            try:
                html = await self._fetch_html(
                    f"{EXPERTS_SEARCH}/en/persons/",
                    params={"search": query.name},
                )
                if html:
                    profile = self._parse_experts_search(html, query)
            except Exception as e:
                logger.warning("[osu_expertise] HTML search failed: %s", e)

        if not profile:
            return None

        # If we found a profile URL, fetch the full profile
        if profile.get("profile_url"):
            full_profile = await self._fetch_experts_profile(profile["profile_url"])
            if full_profile:
                profile.update(full_profile)

        return profile

    async def _search_experts_api(self, query: ProfessorQuery) -> Optional[Dict]:
        """Search via Pure REST API (returns structured JSON)."""
        try:
            resp = await self._get_json(
                f"{EXPERTS_SEARCH}/ws/api/persons",
                params={"q": query.name, "size": 5},
            )
            if not resp or not resp.get("items"):
                return None

            for item in resp["items"]:
                found_name = item.get("name", {})
                full_name = f"{found_name.get('firstName', '')} {found_name.get('lastName', '')}".strip()
                if not full_name:
                    full_name = found_name.get("text", "")

                if not names_match_from_query(query, full_name):
                    continue

                profile_url = ""
                for link in (item.get("links") or item.get("link") or []):
                    href = link.get("href", "") if isinstance(link, dict) else str(link)
                    if "/persons/" in href:
                        profile_url = href
                        break
                if not profile_url and item.get("uuid"):
                    profile_url = f"{EXPERTS_SEARCH}/en/persons/{item['uuid']}"

                department = ""
                for assoc in (item.get("staffOrganisationAssociations") or []):
                    org = assoc.get("organisationalUnit", {})
                    dept_name = org.get("name", {})
                    department = dept_name.get("text", "") if isinstance(dept_name, dict) else str(dept_name)
                    if department:
                        break

                return {
                    "name": full_name,
                    "profile_url": profile_url,
                    "department": department,
                }

            return None

        except Exception as e:
            logger.debug("[osu_expertise] REST API not available: %s", e)
            return None

    def _parse_experts_search(self, html: str, query: ProfessorQuery) -> Optional[Dict]:
        """Parse search results from OSU Experts HTML page."""
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            return None

        soup = BeautifulSoup(html, "html.parser")

        # Strategy 1: Find <a> tags linking to /persons/ (most resilient)
        for link in soup.find_all("a", href=re.compile(r"/en/persons/")):
            name = link.get_text(strip=True)
            if not name or len(name) < 3:
                continue
            if not names_match_from_query(query, name):
                continue

            profile_url = link.get("href", "")
            if profile_url and not profile_url.startswith("http"):
                profile_url = f"{EXPERTS_SEARCH}{profile_url}"

            # Try to find department from the surrounding container
            department = ""
            parent = link.find_parent(["div", "li", "tr"])
            if parent:
                # Look for org/dept text in sibling elements
                for el in parent.find_all(class_=re.compile(r"department|affiliation|org")):
                    department = el.get_text(strip=True)
                    if department:
                        break
                # Fallback: look for any span/div with organization-like content
                if not department:
                    for el in parent.find_all(["span", "div"]):
                        text = el.get_text(strip=True)
                        if text and text != name and len(text) > 5:
                            department = text
                            break

            return {
                "name": name,
                "profile_url": profile_url,
                "department": department,
            }

        # Strategy 2: Original class-based matching as final fallback
        for result in soup.find_all("div", class_=re.compile(r"result-container|rendering|list-result")):
            name_el = result.find("a", class_=re.compile(r"link person"))
            if not name_el:
                name_el = result.find("a", href=re.compile(r"/persons/"))
            if not name_el:
                continue

            name = name_el.get_text(strip=True)
            if not names_match_from_query(query, name):
                continue

            profile_url = name_el.get("href", "")
            if profile_url and not profile_url.startswith("http"):
                profile_url = f"{EXPERTS_SEARCH}{profile_url}"

            org_el = result.find(class_=re.compile(r"department|affiliation"))
            department = org_el.get_text(strip=True) if org_el else ""

            return {
                "name": name,
                "profile_url": profile_url,
                "department": department,
            }

        return None

    async def _fetch_experts_profile(self, url: str) -> Optional[Dict]:
        """Fetch and parse a full OSU Experts profile page (bypass-aware)."""
        try:
            html = await self._fetch_html(url)
            if not html:
                return None
            return self._parse_experts_profile(html)
        except Exception as e:
            logger.warning("[osu_expertise] Profile fetch failed: %s", e)
            return None

    def _parse_experts_profile(self, html: str) -> Optional[Dict]:
        """Parse a full OSU Experts profile page."""
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            return None

        soup = BeautifulSoup(html, "html.parser")
        data = {}

        # Research interests / fingerprint
        fingerprint_section = soup.find(id=re.compile(r"fingerprint", re.I))
        if fingerprint_section:
            concepts = []
            for item in fingerprint_section.find_all(class_=re.compile(r"concept")):
                concepts.append(item.get_text(strip=True))
            data["research_fingerprint"] = concepts[:30]

        # Research output summary
        output_section = soup.find(id=re.compile(r"research-output|activities", re.I))
        if output_section:
            outputs = []
            for item in output_section.find_all("li")[:30]:
                text = item.get_text(strip=True)
                if text and len(text) > 10:
                    outputs.append(text[:300])
            data["research_outputs"] = outputs

        # Personal profile / biography
        bio_section = soup.find(class_=re.compile(r"personal-profile|biography"))
        if bio_section:
            data["biography"] = bio_section.get_text(strip=True)[:2000]

        # Network / collaborators
        network_section = soup.find(id=re.compile(r"network|collaborat", re.I))
        if network_section:
            collabs = []
            for link in network_section.find_all("a", href=re.compile(r"/persons/"))[:20]:
                collabs.append(link.get_text(strip=True))
            data["collaborators"] = collabs

        return data

    async def _search_knowledge_bank(self, query: ProfessorQuery) -> Optional[Dict]:
        """Search OSU Knowledge Bank (DSpace) for the professor's works."""
        # DSpace REST API
        params = {
            "query": query.name,
            "scope": "",
            "dsoType": "item",
            "size": 30,
        }

        try:
            resp = await self._get_json(
                f"{KNOWLEDGE_BANK_API}/discover/search/objects",
                params=params,
            )
            if not resp:
                return None
        except Exception as e:
            logger.warning("[osu_expertise] Knowledge Bank search failed: %s", e)
            return None

        embedded = resp.get("_embedded", {})
        objects = embedded.get("searchResult", {}).get("_embedded", {}).get("objects", [])
        if not objects:
            return None

        items = []
        for obj in objects:
            item_data = obj.get("_embedded", {}).get("indexableObject", {})
            if not item_data:
                continue

            metadata = item_data.get("metadata", {})
            title = self._get_metadata_value(metadata, "dc.title")
            abstract = self._get_metadata_value(metadata, "dc.description.abstract")
            date = self._get_metadata_value(metadata, "dc.date.issued")
            doc_type = self._get_metadata_value(metadata, "dc.type")
            authors = [
                v.get("value", "")
                for v in metadata.get("dc.contributor.author", [])
            ]

            author_match = any(names_match_from_query(query, a) for a in authors)
            if title and author_match:
                items.append({
                    "title": title,
                    "abstract": (abstract or "")[:500],
                    "date": date or "",
                    "type": doc_type or "",
                    "authors": authors[:10],
                    "handle": item_data.get("handle", ""),
                })

        if not items:
            return None

        return {
            "total_items": len(items),
            "items": items,
        }

    @staticmethod
    def _get_metadata_value(metadata: Dict, field: str) -> Optional[str]:
        values = metadata.get(field, [])
        if values and isinstance(values, list):
            return values[0].get("value")
        return None

    def _to_text(self, data: Dict, query: ProfessorQuery) -> str:
        lines = []
        lines.append(f"=== OSU Research Profiles: {query.name} ===")

        experts = data.get("experts_profile")
        if experts:
            lines.append("\n--- OSU Experts Directory ---")
            if experts.get("department"):
                lines.append(f"Department: {experts['department']}")
            if experts.get("profile_url"):
                lines.append(f"Profile: {experts['profile_url']}")
            if experts.get("biography"):
                lines.append(f"Biography: {experts['biography'][:800]}")
            if experts.get("research_fingerprint"):
                lines.append(f"Research fingerprint: {', '.join(experts['research_fingerprint'][:20])}")
            if experts.get("collaborators"):
                lines.append(f"Key collaborators: {', '.join(experts['collaborators'][:15])}")
            if experts.get("research_outputs"):
                lines.append("\nRecent research outputs:")
                for output in experts["research_outputs"][:15]:
                    lines.append(f"  - {output}")

        kb = data.get("knowledge_bank")
        if kb:
            lines.append(f"\n--- OSU Knowledge Bank ({kb['total_items']} items) ---")
            for i, item in enumerate(kb["items"][:20], 1):
                lines.append(f"\n{i}. {item['title']}")
                if item.get("type"):
                    lines.append(f"   Type: {item['type']}")
                if item.get("date"):
                    lines.append(f"   Date: {item['date']}")
                if item.get("authors"):
                    lines.append(f"   Authors: {', '.join(item['authors'][:5])}")
                if item.get("abstract"):
                    lines.append(f"   Abstract: {item['abstract']}")
                if item.get("handle"):
                    lines.append(f"   Handle: https://kb.osu.edu/handle/{item['handle']}")

        return "\n".join(lines)
