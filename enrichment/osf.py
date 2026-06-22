"""
OSF (Open Science Framework) collector — projects, preprints, registrations.

API docs: https://developer.osf.io/
Free, no auth for public data. OSF hosts pre-registrations, study materials,
code, and data — strong signal about research methodology.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base_collector import BaseCollector, CollectorResult, ProfessorQuery
from .validation import normalize

USERS_URL = "https://api.osf.io/v2/users/"
NODES_URL = "https://api.osf.io/v2/nodes/"


class OSFCollector(BaseCollector):
    def __init__(self, **kwargs):
        kwargs.setdefault("rate_limit_delay", 1.0)
        super().__init__(**kwargs)

    @property
    def source_name(self) -> str:
        return "osf"

    async def collect(self, query: ProfessorQuery) -> CollectorResult:
        # OSF user search via filter[full_name]=...
        resp = await self._get_json(
            USERS_URL,
            params={"filter[full_name]": query.name, "page[size]": 10},
        )
        if not resp:
            return self._make_result(query, success=False, error="OSF user search empty")

        q_last = normalize(query.last_name)
        user_id = None
        for u in resp.get("data") or []:
            attrs = u.get("attributes") or {}
            full_name = normalize(attrs.get("full_name") or "")
            if q_last and q_last in full_name:
                user_id = u.get("id")
                break
        if not user_id:
            return self._make_result(query, success=False, error="No matching OSF user")

        # Fetch this user's public nodes (projects, preprints, registrations)
        nodes_resp = await self._get_json(
            f"{USERS_URL}{user_id}/nodes/",
            params={"page[size]": 30},
        )
        if not nodes_resp:
            return self._make_result(query, success=False, error=f"OSF nodes for {user_id} empty")

        items: List[Dict[str, Any]] = []
        for n in nodes_resp.get("data") or []:
            attrs = n.get("attributes") or {}
            items.append({
                "id": n.get("id"),
                "title": attrs.get("title"),
                "description": (attrs.get("description") or "")[:1500],
                "category": attrs.get("category"),
                "public": attrs.get("public"),
                "tags": attrs.get("tags"),
                "date_created": attrs.get("date_created"),
                "date_modified": attrs.get("date_modified"),
                "url": f"https://osf.io/{n.get('id')}/",
            })

        if not items:
            return self._make_result(query, success=False, error="User matched but no nodes")

        data = {"osf_user_id": user_id, "total_nodes": len(items), "nodes": items}
        return self._make_result(query, success=True, data=data, raw_text=self._to_text(data, query))

    def _to_text(self, data: Dict, query: ProfessorQuery) -> str:
        lines = [
            f"=== OSF projects/registrations by {query.name} ===",
            f"OSF user: {data['osf_user_id']} | Total nodes: {data['total_nodes']}",
            "",
        ]
        for n in data["nodes"]:
            lines.append(f"• [{n.get('category')}] {n['title']}")
            lines.append(f"  URL: {n.get('url')} | Created: {n.get('date_created')}")
            if n.get("description"):
                lines.append(f"  Description: {n['description']}")
            lines.append("")
        return "\n".join(lines)
