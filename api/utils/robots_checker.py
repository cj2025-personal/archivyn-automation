"""
Robots.txt checker with simple in-memory caching.
"""
from __future__ import annotations

import time
from typing import Dict, Optional
from urllib.parse import urlparse, urlunparse
from urllib import robotparser
import urllib.request


class RobotsCache:
    def __init__(self, user_agent: str = "*", timeout: int = 10, ttl_seconds: int = 3600) -> None:
        self.user_agent = user_agent
        self.timeout = timeout
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Dict[str, Optional[bool]]] = {}

    def _robots_url(self, url: str) -> str:
        parsed = urlparse(url)
        scheme = parsed.scheme or "https"
        netloc = parsed.netloc
        return urlunparse((scheme, netloc, "/robots.txt", "", "", ""))

    def allowed(self, url: str) -> Optional[bool]:
        if not url:
            return None
        parsed = urlparse(url)
        if not parsed.netloc:
            return None
        key = f"{parsed.scheme or 'https'}://{parsed.netloc}"
        entry = self._cache.get(key)
        now = time.time()
        if entry and (now - entry.get("fetched_at", 0) < self.ttl_seconds):
            return entry.get("allowed")

        robots_url = self._robots_url(url)
        rp = robotparser.RobotFileParser()
        try:
            req = urllib.request.Request(robots_url, headers={"User-Agent": self.user_agent})
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                content = resp.read().decode("utf-8", errors="replace")
            rp.parse(content.splitlines())
            allowed = rp.can_fetch(self.user_agent, url)
        except Exception:
            allowed = None

        self._cache[key] = {"allowed": allowed, "fetched_at": now}
        return allowed
