"""
Wikidata collector — structured researcher facts via SPARQL.

Pulls: awards, academic positions, doctoral advisor, doctoral students,
employer history, notable works, and affiliated Wikipedia article.

Free, no auth. Endpoint: https://query.wikidata.org/sparql
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base_collector import BaseCollector, CollectorResult, ProfessorQuery
from .validation import normalize

SPARQL_URL = "https://query.wikidata.org/sparql"
ENTITY_SEARCH = "https://www.wikidata.org/w/api.php"


class WikidataCollector(BaseCollector):
    def __init__(self, **kwargs):
        kwargs.setdefault("rate_limit_delay", 1.5)
        super().__init__(**kwargs)

    @property
    def source_name(self) -> str:
        return "wikidata"

    async def get_client(self):
        # Wikimedia policy requires a User-Agent with a real contact.
        # Generic "Mozilla" UAs get 403'd. Override the parent's client with
        # a compliant one on first use.
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
        qid = await self._find_entity(query)
        if not qid:
            return self._make_result(query, success=False, error="No Wikidata entity found")

        sparql = f"""
        SELECT ?propLabel ?valueLabel ?qualifierLabel WHERE {{
          VALUES ?prop {{
            wdt:P166 wdt:P108 wdt:P69 wdt:P184 wdt:P185
            wdt:P106 wdt:P800 wdt:P735 wdt:P734 wdt:P463
            wdt:P570 wdt:P569 wdt:P19 wdt:P551
          }}
          wd:{qid} ?prop ?value .
          OPTIONAL {{ ?value rdfs:label ?valueLabel . FILTER(LANG(?valueLabel) = "en") }}
          OPTIONAL {{ ?prop rdfs:label ?propLabel . FILTER(LANG(?propLabel) = "en") }}
        }}
        """
        headers = {"Accept": "application/sparql-results+json"}
        sparql_resp = await self._get_json(
            SPARQL_URL,
            params={"query": sparql, "format": "json"},
        )
        if not sparql_resp:
            # Try raw fetch with headers as fallback
            client = await self.get_client()
            resp = await client.get(SPARQL_URL, params={"query": sparql, "format": "json"}, headers=headers)
            if resp.status_code == 200:
                sparql_resp = resp.json()

        facts: List[Dict[str, str]] = []
        if sparql_resp:
            for binding in sparql_resp.get("results", {}).get("bindings", []):
                prop = binding.get("propLabel", {}).get("value", "")
                value = binding.get("valueLabel", {}).get("value", "")
                if prop and value:
                    facts.append({"property": prop, "value": value})

        # Also fetch Wikipedia sitelink if any
        wikipedia_url = await self._get_wikipedia_url(qid)

        data = {
            "wikidata_qid": qid,
            "wikidata_url": f"https://www.wikidata.org/wiki/{qid}",
            "wikipedia_url": wikipedia_url,
            "facts": facts,
        }

        if not facts and not wikipedia_url:
            return self._make_result(query, success=False, error="Entity exists but has no relevant facts")

        return self._make_result(query, success=True, data=data, raw_text=self._to_text(data, query))

    async def _find_entity(self, query: ProfessorQuery) -> Optional[str]:
        """Search Wikidata for a human entity matching the professor's name."""
        params = {
            "action": "wbsearchentities",
            "search": query.name,
            "language": "en",
            "format": "json",
            "type": "item",
            "limit": 15,
        }
        resp = await self._get_json(ENTITY_SEARCH, params=params)
        if not resp:
            return None

        q_name = normalize(query.name)
        q_last = normalize(query.last_name)
        q_first = normalize(query.first_name)

        # Researcher-ish descriptions. Widened from the original set to include
        # field-specific labels ("chemist", "civil engineer", "nutritionist"…).
        researcher_kw = (
            "professor", "researcher", "scientist", "academic", "scholar",
            "engineer", "chemist", "biologist", "physicist", "mathematician",
            "economist", "sociologist", "psychologist", "historian",
            "philosopher", "physician", "surgeon", "pharmacologist",
            "nutritionist", "geologist", "ecologist", "astronomer",
            "author", "writer", "lecturer", "dean", "faculty",
            "biochemist", "neuroscientist", "epidemiologist", "pathologist",
            "cardiologist", "oncologist", "immunologist",
        )

        # 1) Strong match: full-name match + researcher-like description
        for item in resp.get("search", []):
            label = normalize(item.get("label", ""))
            description = (item.get("description") or "").lower()
            if (q_name in label or label in q_name) and any(kw in description for kw in researcher_kw):
                return item.get("id")

        # 2) Medium match: exact name match regardless of description
        for item in resp.get("search", []):
            label = normalize(item.get("label", ""))
            if label == q_name:
                return item.get("id")

        # 3) Weak match: first + last both present in label (handles middle
        # initials like "Jeffrey S. Volek" vs "Jeff Volek")
        if q_first and q_last:
            for item in resp.get("search", []):
                label = normalize(item.get("label", ""))
                desc = (item.get("description") or "").lower()
                if q_last in label.split() and any(
                    label.split()[0].startswith(q_first[:3])
                    or q_first.startswith(label.split()[0])
                    for _ in [None]
                ):
                    # Require description to hint at a researcher/academic
                    if any(kw in desc for kw in researcher_kw):
                        return item.get("id")
        return None

    async def _get_wikipedia_url(self, qid: str) -> str:
        params = {
            "action": "wbgetentities",
            "ids": qid,
            "props": "sitelinks/urls",
            "sitefilter": "enwiki",
            "format": "json",
        }
        resp = await self._get_json(ENTITY_SEARCH, params=params)
        if not resp:
            return ""
        entity = (resp.get("entities") or {}).get(qid, {})
        sitelinks = entity.get("sitelinks", {})
        enwiki = sitelinks.get("enwiki", {})
        return enwiki.get("url", "")

    def _to_text(self, data: Dict, query: ProfessorQuery) -> str:
        lines = [
            f"=== Wikidata facts for {query.name} ===",
            f"Wikidata: {data['wikidata_url']}",
        ]
        if data.get("wikipedia_url"):
            lines.append(f"Wikipedia: {data['wikipedia_url']}")
        lines.append("")
        for f in data["facts"]:
            lines.append(f"- {f['property']}: {f['value']}")
        return "\n".join(lines)
