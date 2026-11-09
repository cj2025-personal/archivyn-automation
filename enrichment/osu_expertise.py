"""
Ohio State Knowledge Bank collector (kb.osu.edu, DSpace 7 REST API).

What a scholar wrote or advised in Ohio State's institutional repository:
titles, dates, types and handles. Two corrections from the 2026 run, where it
found nobody (0 of 460):
  - The OSU Experts directory this collector also read (experts.osu.edu) no
    longer exists - the domain does not resolve - so that half is gone.
  - The Knowledge Bank records authors in `dc.creator` and faculty advisors in
    `dc.contributor.advisor`; the collector read only `dc.contributor.author`,
    which these items leave empty.

What is kept: for works the scholar authored, the record and its authors; for
theses they advised, the record only - the student is not who we collect
about. Abstracts are kept only when the item carries an open license
(dc.rights.uri to Creative Commons or a public-domain mark); the repository's
default rights statement says items "may be protected by copyright".

The source key stays `osu_expertise` so earlier records and the viewers that
read them keep working.
"""

import logging
from typing import Dict, List, Optional

from .base_collector import BaseCollector, CollectorResult, ProfessorQuery
from .validation import names_match_from_query

logger = logging.getLogger(__name__)

KNOWLEDGE_BANK_API = "https://kb.osu.edu/server/api"
OPEN_LICENSE_MARKERS = ("creativecommons.org", "publicdomain")


def _values(metadata: Dict, field: str) -> List[str]:
    return [v.get("value", "") for v in (metadata.get(field) or []) if isinstance(v, dict) and v.get("value")]


def is_openly_licensed(metadata: Dict) -> bool:
    uris = " ".join(_values(metadata, "dc.rights.uri")).lower()
    return any(marker in uris for marker in OPEN_LICENSE_MARKERS)


def item_for_scholar(query: ProfessorQuery, item: Dict) -> Optional[Dict]:
    """The scholar's part in one repository item, or None if they have none."""
    metadata = item.get("metadata") or {}
    title = (_values(metadata, "dc.title") or [""])[0]
    if not title:
        return None
    authors = _values(metadata, "dc.creator") + _values(metadata, "dc.contributor.author")
    advisors = _values(metadata, "dc.contributor.advisor")
    if any(names_match_from_query(query, a) for a in authors):
        role = "author"
    elif any(names_match_from_query(query, a) for a in advisors):
        role = "advisor"
    else:
        return None
    record = {
        "title": title,
        "role": role,
        "date": (_values(metadata, "dc.date.issued") or [""])[0],
        "type": (_values(metadata, "dc.type") or [""])[0],
        "handle": item.get("handle", ""),
        "open_license": is_openly_licensed(metadata),
    }
    if role == "author":
        record["authors"] = authors[:10]
    if record["open_license"]:
        record["abstract"] = ((_values(metadata, "dc.description.abstract") or [""])[0])[:500]
    return record


class OSUExpertiseCollector(BaseCollector):
    """What the scholar authored or advised in OSU's Knowledge Bank."""

    def __init__(self, **kwargs):
        kwargs.setdefault("rate_limit_delay", 2.5)
        kwargs.setdefault("timeout", 45.0)
        super().__init__(**kwargs)

    @property
    def source_name(self) -> str:
        return "osu_expertise"

    async def collect(self, query: ProfessorQuery) -> CollectorResult:
        try:
            resp = await self._get_json(
                f"{KNOWLEDGE_BANK_API}/discover/search/objects",
                params={"query": query.name, "dsoType": "item", "size": 40},
            )
        except Exception as e:
            logger.warning("[osu_expertise] Knowledge Bank search failed: %s", e)
            resp = None
        if not resp:
            return self._make_result(query, success=False, error="Knowledge Bank search failed")

        objects = (((resp.get("_embedded") or {}).get("searchResult") or {}).get("_embedded") or {}).get("objects") or []
        items = []
        for obj in objects:
            record = item_for_scholar(query, (obj.get("_embedded") or {}).get("indexableObject") or {})
            if record:
                items.append(record)
        if not items:
            return self._make_result(query, success=False, error="Not found in the OSU Knowledge Bank")

        data = {"knowledge_bank": {"total_items": len(items), "items": items}}
        return self._make_result(query, success=True, data=data, raw_text=self._to_text(items, query))

    def _to_text(self, items: List[Dict], query: ProfessorQuery) -> str:
        authored = [i for i in items if i["role"] == "author"]
        advised = [i for i in items if i["role"] == "advisor"]
        lines = [f"=== OSU Knowledge Bank: {query.name} ==="]
        if authored:
            lines.append(f"\n--- Works ({len(authored)}) ---")
            for i, item in enumerate(authored[:20], 1):
                lines.append(f"\n{i}. {item['title']}")
                for label, key in (("Type", "type"), ("Date", "date")):
                    if item.get(key):
                        lines.append(f"   {label}: {item[key]}")
                if item.get("authors"):
                    lines.append(f"   Authors: {', '.join(item['authors'][:5])}")
                if item.get("abstract"):
                    lines.append(f"   Abstract (openly licensed): {item['abstract']}")
                if item.get("handle"):
                    lines.append(f"   Handle: https://kb.osu.edu/handle/{item['handle']}")
        if advised:
            lines.append(f"\n--- Theses and projects advised ({len(advised)}) ---")
            for item in advised[:20]:
                when = f" ({item['date'][:4]})" if item.get("date") else ""
                kind = f" [{item['type']}]" if item.get("type") else ""
                lines.append(f"  - {item['title']}{when}{kind}")
        return "\n".join(lines)
