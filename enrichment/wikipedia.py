"""
Wikipedia collector — biographical page extract if the researcher has one.

Uses MediaWiki action=query&prop=extracts plus infobox via pageprops.
Free, no auth.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base_collector import BaseCollector, CollectorResult, ProfessorQuery
from .validation import normalize, strict_identity_match

API = "https://en.wikipedia.org/w/api.php"


class WikipediaCollector(BaseCollector):
    def __init__(self, **kwargs):
        kwargs.setdefault("rate_limit_delay", 0.5)
        super().__init__(**kwargs)

    @property
    def source_name(self) -> str:
        return "wikipedia"

    async def get_client(self):
        # Wikimedia policy requires a contact-bearing User-Agent; generic UAs
        # get 403 on wikipedia.org/w/api.php.
        import httpx, os
        if self._client is None or self._client.is_closed:
            email = os.getenv("OPENALEX_EMAIL") or os.getenv("UNPAYWALL_EMAIL") or "research@example.com"
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                follow_redirects=True,
                headers={
                    "User-Agent": f"OSU-scholar-enrichment/1.0 (research; {email})",
                    "Accept": "application/json",
                },
            )
        return self._client

    async def collect(self, query: ProfessorQuery) -> CollectorResult:
        # Prior version hardcoded "Ohio State" into the search query, which
        # filtered out researchers whose Wikipedia article doesn't mention OSU
        # in the intro (many legitimate pages don't). Now: search by name
        # alone, then verify OSU appears in the full extract.
        q_last = normalize(query.last_name)
        q_first = normalize(query.first_name)

        page_title: Optional[str] = None

        # Try two queries: name, then name + "researcher" (disambiguates pages
        # with common names, e.g. musicians vs academics).
        for srsearch in (query.name, f"{query.name} researcher", f"{query.name} professor"):
            search = await self._get_json(API, params={
                "action": "query", "list": "search",
                "srsearch": srsearch,
                "srlimit": 8, "format": "json",
            })
            if not search:
                continue

            candidates = (search.get("query") or {}).get("search") or []
            # Prefer candidates whose snippet/title has both first & last name.
            for c in candidates:
                title_norm = normalize(c.get("title", ""))
                snippet = (c.get("snippet") or "").lower()
                if q_last and q_last in title_norm and (not q_first or q_first in title_norm or q_first in snippet):
                    page_title = c.get("title")
                    break
            if page_title:
                break
            # Fallback: last-name match with no first-name confirmation.
            for c in candidates:
                if q_last and q_last in normalize(c.get("title", "")):
                    page_title = c.get("title")
                    break
            if page_title:
                break

        if not page_title:
            return self._make_result(query, success=False, error="No Wikipedia page matched")

        # Get extract + pageprops
        detail = await self._get_json(API, params={
            "action": "query",
            "prop": "extracts|pageprops|info",
            "titles": page_title,
            "explaintext": "1",
            "exintro": "0",  # full text
            "exsectionformat": "plain",
            "inprop": "url",
            "format": "json",
        })
        if not detail:
            return self._make_result(query, success=False, error="Wikipedia detail fetch failed")

        pages = (detail.get("query") or {}).get("pages") or {}
        if not pages:
            return self._make_result(query, success=False, error="No page data")

        page = next(iter(pages.values()))
        extract = page.get("extract") or ""
        if len(extract.strip()) < 80:
            return self._make_result(
                query, success=False,
                error="Wikipedia extract too short to be a real bio",
            )

        # CRITICAL: strict identity gate. This is where the Summer-Lee-vs-L-Lee
        # false positive used to land. Now requires:
        #   - Full name (first + last) anchored in extract
        #   - Either "Ohio State" OR department mentioned
        # If we don't have both, we assume it's a different person with the
        # same surname and reject rather than poison RAG with wrong content.
        if not strict_identity_match(
            query, extract,
            require_full_name=True,
            require_affiliation=True,
            department_hint=query.department,
            min_name_density=2,
        ):
            return self._make_result(
                query, success=False,
                error=(
                    "Wikipedia page failed strict identity gate "
                    "(missing full-name + OSU/department signal); "
                    "likely a same-named but different person"
                ),
            )

        data = {
            "title": page.get("title"),
            "page_id": page.get("pageid"),
            "url": page.get("fullurl"),
            "extract": extract[:8000],
        }
        return self._make_result(query, success=True, data=data, raw_text=self._to_text(data, query))

    def _to_text(self, data: Dict, query: ProfessorQuery) -> str:
        return (
            f"=== Wikipedia article: {data['title']} ===\n"
            f"URL: {data['url']}\n\n"
            f"{data['extract']}"
        )
