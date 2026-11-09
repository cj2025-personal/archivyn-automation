"""
How this repo fetches other people's web pages: under one honest name, only
where the site's robots.txt allows it, and never past a refusal.

Why it is the only way: collecting what a university publishes about its own
faculty is lawful; getting around a site's defences is what turns it into a
contract or trespass claim (hiQ lost to LinkedIn on breach of contract after
scraping past its blocks). Measured 2026-11-08 on the 40 university sites that
hold most of our scholars' pages: robots.txt allowed the scholars' pages on 39,
and a bot that said who it was reached pages for 70% of scholars - so honesty
costs little. Where a site refuses, the answer is to ask its web team, not to
look like a browser.

Documented APIs (OpenAlex, ORCID, Crossref, NIH, arXiv's API...) are governed
by their own terms and are not "crawled"; robots.txt is for page fetches.
"""

from __future__ import annotations

import asyncio
import ssl
import threading
import time
import urllib.error
import urllib.request
from typing import Awaitable, Callable, Dict, Optional, Tuple
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import httpx

#: The only User-Agent this repo sends. A real contact, so a site that wants
#: us to slow down or stop can say so.
USER_AGENT = "ProjArchBot/1.0 (educational research crawler; contact: hello@projarch.org)"
#: The token a site's robots.txt names us by.
AGENT_TOKEN = "ProjArchBot"

#: Text that marks a challenge or block page served with a 200. Such a page
#: is a refusal, recorded as one - not something to get past.
_CHALLENGE_SIGNATURES = (
    "just a moment",
    "checking your browser",
    "cf-browser-verification",
    "enable javascript and cookies",
    "attention required",
    "please verify you are a human",
    "verify you are human",
    "cf-challenge",
)
# Not the bare word "captcha": a contact form with reCAPTCHA is an ordinary page.

# A robots.txt read stays good this long; one that could not be read (5xx,
# timeout) is retried sooner, and until then the site's pages are skipped.
_ROBOTS_TTL_S = 6 * 3600
_UNREADABLE_TTL_S = 10 * 60

RobotsFetcher = Callable[[str], Awaitable[Tuple[Optional[int], str]]]


def is_challenge_page(html: str) -> bool:
    """True for a bot-check or block page served in place of the content."""
    if not html:
        return False
    low = html[:8000].lower()
    return any(sig in low for sig in _CHALLENGE_SIGNATURES)


def parser_for(status: Optional[int], body: str) -> Tuple[RobotFileParser, bool]:
    """
    How a robots.txt response is read, as crawlers commonly read it:
    200 - its rules; 401/403 - the whole site is off limits; other 4xx (no
    file) - everything is allowed; 5xx or no answer - nothing is fetched for
    now. Returns the parser and whether the read counts as settled.
    """
    rp = RobotFileParser()
    if status == 200:
        rp.parse(body.splitlines())
        return rp, True
    if status in (401, 403):
        rp.disallow_all = True
        return rp, True
    if status is not None and 400 <= status < 500:
        rp.allow_all = True
        return rp, True
    rp.disallow_all = True
    return rp, False


async def _default_fetch(url: str) -> Tuple[Optional[int], str]:
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(15.0), follow_redirects=True, headers={"User-Agent": USER_AGENT}
        ) as client:
            resp = await client.get(url)
            return resp.status_code, resp.text if resp.status_code == 200 else ""
    except Exception:
        return None, ""


class RobotsPolicy:
    """Per-host robots.txt rules, read once and cached."""

    def __init__(self, fetch: Optional[RobotsFetcher] = None, clock: Callable[[], float] = time.monotonic):
        self._fetch = fetch or _default_fetch
        self._clock = clock
        self._cache: Dict[str, Tuple[RobotFileParser, float]] = {}
        self._locks: Dict[str, asyncio.Lock] = {}

    async def _rules(self, url: str) -> RobotFileParser:
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        lock = self._locks.setdefault(origin, asyncio.Lock())
        async with lock:
            cached = self._cache.get(origin)
            if cached and cached[1] > self._clock():
                return cached[0]
            status, body = await self._fetch(f"{origin}/robots.txt")
            rp, settled = parser_for(status, body)
            self._cache[origin] = (rp, self._clock() + (_ROBOTS_TTL_S if settled else _UNREADABLE_TTL_S))
            return rp

    async def allows(self, url: str) -> bool:
        return (await self._rules(url)).can_fetch(AGENT_TOKEN, url)

    async def crawl_delay(self, url: str) -> Optional[float]:
        rp = await self._rules(url)
        try:
            delay = rp.crawl_delay(AGENT_TOKEN)
        except Exception:
            delay = None
        return float(delay) if delay else None


#: One policy for the process, so every collector shares the cache.
ROBOTS = RobotsPolicy()


# ── The same rules for synchronous code (scripts, the page worker) ─────────

SyncRobotsFetcher = Callable[[str], Tuple[Optional[int], str]]

# Crawlers commonly read at most 500 KiB of a robots.txt; the rest is ignored.
_ROBOTS_MAX_BYTES = 512 * 1024


def fetch_robots_txt(url: str, *, context: Optional[ssl.SSLContext] = None) -> Tuple[Optional[int], str]:
    """(status, body) for a robots.txt URL, fetched under our name; (None, "")
    when there is no answer. `context` is for callers that fetch the site's
    pages with their own TLS settings, so robots.txt is read the same way."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=15, context=context) as resp:
            status = getattr(resp, "status", None) or resp.getcode()
            body = resp.read(_ROBOTS_MAX_BYTES).decode("utf-8", errors="replace")
            return status, body if status == 200 else ""
    except urllib.error.HTTPError as exc:
        return exc.code, ""
    except Exception:
        return None, ""


class SyncRobotsPolicy:
    """RobotsPolicy for synchronous callers: per-host rules, read once and
    cached, read as parser_for reads them. Safe to share between threads."""

    def __init__(self, fetch: Optional[SyncRobotsFetcher] = None, clock: Callable[[], float] = time.monotonic):
        self._fetch = fetch or fetch_robots_txt
        self._clock = clock
        self._cache: Dict[str, Tuple[RobotFileParser, float]] = {}
        self._locks: Dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    def _rules(self, url: str) -> Optional[RobotFileParser]:
        parts = urlsplit(url)
        if parts.scheme.lower() not in ("http", "https") or not parts.netloc:
            return None
        origin = f"{parts.scheme.lower()}://{parts.netloc.lower()}"
        with self._locks_guard:
            lock = self._locks.setdefault(origin, threading.Lock())
        with lock:
            cached = self._cache.get(origin)
            if cached and cached[1] > self._clock():
                return cached[0]
            status, body = self._fetch(f"{origin}/robots.txt")
            rp, settled = parser_for(status, body)
            self._cache[origin] = (rp, self._clock() + (_ROBOTS_TTL_S if settled else _UNREADABLE_TTL_S))
            return rp

    def allows(self, url: str) -> bool:
        """Whether robots.txt lets us fetch url. Only http(s) URLs are ever allowed."""
        rp = self._rules(url)
        return rp is not None and rp.can_fetch(AGENT_TOKEN, url)

    def crawl_delay(self, url: str) -> Optional[float]:
        rp = self._rules(url)
        if rp is None:
            return None
        try:
            delay = rp.crawl_delay(AGENT_TOKEN)
        except Exception:
            delay = None
        return float(delay) if delay else None


#: One synchronous policy for the process, shared like ROBOTS.
SYNC_ROBOTS = SyncRobotsPolicy()


def robots_allows(url: str) -> bool:
    """Whether robots.txt lets us fetch url - for synchronous code."""
    return SYNC_ROBOTS.allows(url)


def crawl_delay(url: str) -> Optional[float]:
    """The Crawl-delay robots.txt asks of us for url's site, in seconds, or None."""
    return SYNC_ROBOTS.crawl_delay(url)
