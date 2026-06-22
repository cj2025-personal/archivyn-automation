"""
GitHub collector — public repos, READMEs, topics, stars for a researcher.

API docs: https://docs.github.com/en/rest
Free, no auth needed for public data, but GITHUB_TOKEN greatly raises rate
limits (5000/hr vs 60/hr). Set GITHUB_TOKEN env var.

Strategy:
  1. Search users by name + affiliation ("Ohio State", "OSU").
  2. Also search commits/repos by the full name for non-self-described users.
  3. Pull top repos (by stars) with description + README excerpt.
"""

from __future__ import annotations

import base64
import os
from typing import Any, Dict, List, Optional

from .base_collector import BaseCollector, CollectorResult, ProfessorQuery

BASE = "https://api.github.com"


class GitHubCollector(BaseCollector):
    def __init__(self, **kwargs):
        kwargs.setdefault("rate_limit_delay", 1.0)
        super().__init__(**kwargs)
        self.token = os.getenv("GITHUB_TOKEN", "")

    @property
    def source_name(self) -> str:
        return "github"

    def _headers(self) -> Dict[str, str]:
        h = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    async def _gh_get(self, url: str, params: Optional[Dict] = None) -> Optional[Any]:
        await self._rate_limit()
        try:
            client = await self.get_client()
            resp = await client.get(url, params=params, headers=self._headers())
            if resp.status_code in (401, 403, 404):
                return None
            resp.raise_for_status()
            return resp.json()
        except Exception:
            return None

    async def collect(self, query: ProfessorQuery) -> CollectorResult:
        user = await self._find_user(query)
        if not user:
            return self._make_result(query, success=False, error="No matching GitHub user found")

        login = user.get("login")
        repos = await self._gh_get(f"{BASE}/users/{login}/repos", params={"per_page": 50, "sort": "updated"}) or []
        # Keep only non-forks with >1 star OR description (to prune throwaway)
        repos = [
            r for r in repos
            if not r.get("fork") and (r.get("stargazers_count", 0) >= 1 or r.get("description"))
        ]
        repos.sort(key=lambda r: r.get("stargazers_count", 0), reverse=True)
        top = repos[:15]

        enriched: List[Dict[str, Any]] = []
        for r in top:
            readme_excerpt = await self._fetch_readme(login, r["name"])
            enriched.append({
                "name": r.get("name"),
                "full_name": r.get("full_name"),
                "description": r.get("description") or "",
                "language": r.get("language"),
                "stars": r.get("stargazers_count", 0),
                "forks": r.get("forks_count", 0),
                "topics": r.get("topics") or [],
                "created_at": r.get("created_at"),
                "updated_at": r.get("updated_at"),
                "homepage": r.get("homepage"),
                "html_url": r.get("html_url"),
                "readme_excerpt": readme_excerpt,
            })

        data = {
            "login": login,
            "profile_url": user.get("html_url"),
            "name": user.get("name") or query.name,
            "bio": user.get("bio") or "",
            "company": user.get("company") or "",
            "location": user.get("location") or "",
            "blog": user.get("blog") or "",
            "public_repos": user.get("public_repos", 0),
            "followers": user.get("followers", 0),
            "top_repos": enriched,
        }
        return self._make_result(query, success=True, data=data, raw_text=self._to_text(data, query))

    async def _find_user(self, query: ProfessorQuery) -> Optional[Dict]:
        # Try: name + "Ohio State"
        q_first = (query.first_name or "").lower()
        q_last = (query.last_name or "").lower()

        for q in (
            f'{query.name} "Ohio State"',
            f'{query.name} OSU',
            f'"{query.name}"',
        ):
            resp = await self._gh_get(f"{BASE}/search/users", params={"q": q, "per_page": 5})
            if not resp or not resp.get("items"):
                continue
            for item in resp["items"]:
                user = await self._gh_get(f"{BASE}/users/{item['login']}")
                if not user:
                    continue
                hay = " ".join([
                    (user.get("name") or ""),
                    (user.get("bio") or ""),
                    (user.get("company") or ""),
                    (user.get("location") or ""),
                    (user.get("email") or ""),
                    (user.get("blog") or ""),
                ]).lower()
                if not q_last or q_last not in hay:
                    continue

                # STRICT GATE: require first-name signal AND OSU affiliation.
                # Prior version accepted any GitHub user with matching last
                # name, letting unrelated users pollute the RAG. Now we demand:
                #   (a) first name / first initial present, AND
                #   (b) "ohio state" OR ".osu.edu" OR "osu" context in profile.
                first_ok = (q_first and q_first in hay) or (
                    q_first and f" {q_first[0]}." in hay
                ) or (q_first and hay.startswith(f"{q_first[0]}. "))
                aff_ok = (
                    "ohio state" in hay
                    or ".osu.edu" in hay
                    or " osu " in f" {hay} "
                    or "osu." in hay
                )
                if first_ok and aff_ok:
                    return user
        return None

    async def _fetch_readme(self, login: str, repo: str) -> str:
        resp = await self._gh_get(f"{BASE}/repos/{login}/{repo}/readme")
        if not resp:
            return ""
        content = resp.get("content") or ""
        try:
            raw = base64.b64decode(content).decode("utf-8", errors="ignore")
            return raw[:3000]
        except Exception:
            return ""

    def _to_text(self, data: Dict, query: ProfessorQuery) -> str:
        lines = [
            f"=== GitHub profile for {query.name} ===",
            f"Login: @{data['login']} | Profile: {data['profile_url']}",
            f"Bio: {data['bio']}",
            f"Company: {data['company']} | Location: {data['location']} | Blog: {data['blog']}",
            f"Public repos: {data['public_repos']} | Followers: {data['followers']}",
            "",
            "── Top repositories ──",
        ]
        for r in data["top_repos"]:
            lines.append(f"\n• {r['name']} — ⭐ {r['stars']} | {r.get('language') or 'n/a'}")
            if r["description"]:
                lines.append(f"  {r['description']}")
            if r["topics"]:
                lines.append(f"  Topics: {', '.join(r['topics'])}")
            if r["readme_excerpt"]:
                lines.append(f"  README: {r['readme_excerpt'][:800]}")
        return "\n".join(lines)
