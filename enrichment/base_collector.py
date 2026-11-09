"""
Base class for all enrichment data collectors.
Each collector queries a single public data source for professor information.
"""

import asyncio
import hashlib
import json
import logging
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit

import httpx

from .crawl_policy import ROBOTS, USER_AGENT, is_challenge_page

logger = logging.getLogger(__name__)


@dataclass
class ProfessorQuery:
    """Input query for looking up a professor across data sources."""
    profile_id: str
    name: str
    university: str = "Ohio State University"
    department: str = ""
    profile_url: str = ""
    email: str = ""

    @property
    def first_name(self) -> str:
        parts = self.name.strip().split()
        return parts[0] if parts else ""

    @property
    def last_name(self) -> str:
        parts = self.name.strip().split()
        return parts[-1] if len(parts) > 1 else ""

    @property
    def name_variants(self) -> List[str]:
        """Generate common name variants for fuzzy matching."""
        variants = [self.name]
        parts = self.name.strip().split()
        if len(parts) >= 2:
            # "John Smith"
            variants.append(f"{parts[0]} {parts[-1]}")
            # "J Smith"
            variants.append(f"{parts[0][0]} {parts[-1]}")
            # "J. Smith"
            variants.append(f"{parts[0][0]}. {parts[-1]}")
            # "Smith, John"
            variants.append(f"{parts[-1]}, {parts[0]}")
            # Handle middle names: "John A Smith" -> "John Smith"
            if len(parts) > 2:
                variants.append(f"{parts[0]} {parts[-1]}")
        return list(dict.fromkeys(variants))  # dedupe, preserve order


@dataclass
class CollectorResult:
    """Standardized result from any collector."""
    source: str  # e.g. "semantic_scholar", "openalex"
    professor_query: str  # professor name used in query
    profile_id: str
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    raw_text: str = ""  # Flattened text for chunking
    error: Optional[str] = None
    cached: bool = False
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BaseCollector(ABC):
    """
    Base class for all data source collectors.

    Subclasses must implement:
        - source_name: property returning the source identifier
        - collect(): async method that queries the source and returns CollectorResult
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        rate_limit_delay: float = 1.0,
        timeout: float = 30.0,
        max_retries: int = 5,
    ):
        self.cache_dir = cache_dir
        self.rate_limit_delay = rate_limit_delay
        self.timeout = timeout
        self.max_retries = max_retries
        self._last_request_time = 0.0
        # True while the last API answer was a 429, so a collector can say
        # "rate-limited" instead of "not found" when its retries run out.
        self.rate_limited = False
        self._client: Optional[httpx.AsyncClient] = None

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Unique identifier for this data source (e.g. 'semantic_scholar')."""
        ...

    @abstractmethod
    async def collect(self, query: ProfessorQuery) -> CollectorResult:
        """Query the data source and return structured results."""
        ...

    async def get_client(self) -> httpx.AsyncClient:
        """Lazy-init a shared async HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                follow_redirects=True,
                headers={"User-Agent": USER_AGENT},
            )
        return self._client

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    # ── Rate Limiting ──────────────────────────────────────────────

    async def _rate_limit(self):
        """Enforce minimum delay between requests."""
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - elapsed)
        self._last_request_time = time.monotonic()

    # ── HTTP helpers ───────────────────────────────────────────────

    async def _fetch_html(
        self,
        url: str,
        params: Optional[Dict] = None,
        headers: Optional[Dict] = None,
    ) -> Optional[str]:
        """
        Fetch a web page's text, as crawl_policy says pages are fetched:
        under our own name, only where robots.txt allows, and a 401/403/429
        or a bot-check page is the site's answer - None, not a retry in
        disguise. For documented APIs use _get_json / _get_api_text.
        """
        page = await self._fetch_page(url, params=params, headers=headers, accept_pdf=False)
        return page["text"] if page else None

    async def _fetch_any(
        self,
        url: str,
        *,
        headers: Optional[Dict] = None,
        accept_pdf: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch a web page or PDF under crawl_policy's rules.

        Returns {status, content_type, text, content, url} or None when
        robots.txt disallows the page, the site refuses us, or it fails.
        """
        return await self._fetch_page(url, headers=headers, accept_pdf=accept_pdf)

    async def _fetch_page(
        self,
        url: str,
        *,
        params: Optional[Dict] = None,
        headers: Optional[Dict] = None,
        accept_pdf: bool = True,
    ) -> Optional[Dict[str, Any]]:
        full_url = str(httpx.URL(url, params=params)) if params else url
        if not await ROBOTS.allows(full_url):
            print(f"    [{self.source_name}] robots.txt disallows {full_url} - skipped")
            return None
        await self._rate_limit()
        delay = await ROBOTS.crawl_delay(full_url)
        if delay and delay > self.rate_limit_delay:
            await asyncio.sleep(delay - self.rate_limit_delay)
        # Our name is the only one we send, whatever a caller passes.
        headers = {k: v for k, v in (headers or {}).items() if k.lower() != "user-agent"}
        try:
            client = await self.get_client()
            resp = await client.get(full_url, headers=headers)
        except Exception as e:
            logger.debug("[%s] fetch failed for %s: %s", self.source_name, full_url, e)
            return None
        if resp.status_code in (401, 403, 429):
            print(f"    [{self.source_name}] {urlsplit(full_url).netloc} refused the bot ({resp.status_code}) - skipped")
            return None
        if resp.status_code != 200:
            return None
        ct = (resp.headers.get("content-type") or "").lower()
        if accept_pdf and ("application/pdf" in ct or full_url.lower().endswith(".pdf")):
            if not resp.content:
                return None
            return {"status": 200, "content_type": "pdf", "text": "", "content": resp.content, "url": str(resp.url)}
        if is_challenge_page(resp.text):
            print(f"    [{self.source_name}] {urlsplit(full_url).netloc} answered with a bot check - skipped")
            return None
        return {
            "status": 200,
            "content_type": "html" if "text/html" in ct or "text/plain" in ct else ct,
            "text": resp.text,
            "content": resp.content,
            "url": str(resp.url),
        }

    async def _get_api_text(self, url: str, params: Optional[Dict] = None) -> Optional[str]:
        """GET a documented API that answers in text or XML (e.g. arXiv's Atom
        API). Governed by that API's terms, not robots.txt."""
        await self._rate_limit()
        try:
            client = await self.get_client()
            resp = await client.get(url, params=params)
        except Exception as e:
            logger.debug("[%s] API request failed: %s", self.source_name, e)
            return None
        return resp.text if resp.status_code == 200 else None

    async def _get_json(self, url: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """GET request returning parsed JSON, with retry + rate limit + jitter."""
        for attempt in range(1, self.max_retries + 1):
            await self._rate_limit()
            try:
                client = await self.get_client()
                resp = await client.get(url, params=params)
                if resp.status_code == 429:
                    self.rate_limited = True
                    # Exponential backoff: 5s, 15s, 35s, 75s, 120s + jitter
                    wait = min(5 * (2 ** attempt) + random.uniform(0, 5), 120)
                    print(f"    [{self.source_name}] ⏳ Rate limited (429), retrying in {wait:.0f}s (attempt {attempt}/{self.max_retries})...")
                    await asyncio.sleep(wait)
                    continue
                # Non-retryable client errors — return None immediately
                if resp.status_code in (400, 401, 403, 404, 405, 410, 422):
                    return None
                resp.raise_for_status()
                self.rate_limited = False
                return resp.json()
            except httpx.HTTPStatusError as e:
                code = e.response.status_code
                # Don't retry client errors (4xx) — they won't succeed on retry
                if 400 <= code < 500:
                    return None
                wait = min(5 * (2 ** attempt) + random.uniform(0, 5), 120)
                print(f"    [{self.source_name}] ⚠️ HTTP {code} (attempt {attempt}/{self.max_retries}), retrying in {wait:.0f}s...")
                if attempt == self.max_retries:
                    raise
                await asyncio.sleep(wait)
            except (httpx.ConnectError, httpx.ReadTimeout) as e:
                wait = min(5 * (2 ** attempt) + random.uniform(0, 3), 120)
                print(f"    [{self.source_name}] ⚠️ {type(e).__name__} (attempt {attempt}/{self.max_retries}), retrying in {wait:.0f}s...")
                if attempt == self.max_retries:
                    raise
                await asyncio.sleep(wait)
        return None

    async def _post_json(self, url: str, json_body: Dict, params: Optional[Dict] = None) -> Optional[Dict]:
        """POST request returning parsed JSON, with retry + rate limit + jitter."""
        for attempt in range(1, self.max_retries + 1):
            await self._rate_limit()
            try:
                client = await self.get_client()
                resp = await client.post(url, json=json_body, params=params)
                if resp.status_code == 429:
                    self.rate_limited = True
                    wait = min(5 * (2 ** attempt) + random.uniform(0, 5), 120)
                    print(f"    [{self.source_name}] ⏳ Rate limited (429), retrying in {wait:.0f}s (attempt {attempt}/{self.max_retries})...")
                    await asyncio.sleep(wait)
                    continue
                if resp.status_code in (400, 401, 403, 404, 405, 410, 422):
                    return None
                resp.raise_for_status()
                self.rate_limited = False
                return resp.json()
            except httpx.HTTPStatusError as e:
                code = e.response.status_code
                if 400 <= code < 500:
                    return None
                wait = min(5 * (2 ** attempt) + random.uniform(0, 5), 120)
                print(f"    [{self.source_name}] ⚠️ HTTP {code} (attempt {attempt}/{self.max_retries}), retrying in {wait:.0f}s...")
                if attempt == self.max_retries:
                    raise
                await asyncio.sleep(wait)
            except (httpx.ConnectError, httpx.ReadTimeout) as e:
                wait = min(5 * (2 ** attempt) + random.uniform(0, 3), 120)
                print(f"    [{self.source_name}] ⚠️ {type(e).__name__} (attempt {attempt}/{self.max_retries}), retrying in {wait:.0f}s...")
                if attempt == self.max_retries:
                    raise
                await asyncio.sleep(wait)
        return None

    # ── Caching ────────────────────────────────────────────────────

    def _cache_key(self, query: ProfessorQuery) -> str:
        raw = f"{self.source_name}:{query.name}:{query.university}:{query.department}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _cache_path(self, query: ProfessorQuery) -> Optional[Path]:
        if not self.cache_dir:
            return None
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self.cache_dir / f"{self.source_name}_{self._cache_key(query)}.json"

    def _load_cache(self, query: ProfessorQuery) -> Optional[CollectorResult]:
        path = self._cache_path(query)
        if path and path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                data["cached"] = True
                return CollectorResult(**data)
            except Exception:
                logger.debug("[%s] Cache miss (corrupt): %s", self.source_name, path)
        return None

    def _save_cache(self, query: ProfessorQuery, result: CollectorResult):
        path = self._cache_path(query)
        if path:
            try:
                path.write_text(json.dumps(result.to_dict(), default=str), encoding="utf-8")
            except Exception as e:
                logger.debug("[%s] Cache save failed: %s", self.source_name, e)

    # ── Convenience ────────────────────────────────────────────────

    def _make_result(
        self,
        query: ProfessorQuery,
        success: bool,
        data: Optional[Dict] = None,
        raw_text: str = "",
        error: Optional[str] = None,
    ) -> CollectorResult:
        from datetime import datetime, timezone
        return CollectorResult(
            source=self.source_name,
            professor_query=query.name,
            profile_id=query.profile_id,
            success=success,
            data=data or {},
            raw_text=raw_text,
            error=error,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    async def safe_collect(self, query: ProfessorQuery) -> CollectorResult:
        """Collect with caching and error handling."""
        # Check cache first
        cached = self._load_cache(query)
        if cached:
            print(f"    [{self.source_name}] ⚡ Cache hit")
            return cached

        print(f"    [{self.source_name}] Querying...")
        try:
            result = await self.collect(query)
            if result.success:
                self._save_cache(query, result)
                text_len = len(result.raw_text) if result.raw_text else 0
                data_keys = len(result.data) if result.data else 0
                print(f"    [{self.source_name}] ✅ Success — {text_len:,} chars, {data_keys} data fields")
            else:
                print(f"    [{self.source_name}] ⚠️ No data: {result.error}")
            return result
        except Exception as e:
            print(f"    [{self.source_name}] ❌ Error: {e}")
            logger.error("[%s] Unexpected error for %s: %s", self.source_name, query.name, e)
            return self._make_result(query, success=False, error=str(e))
