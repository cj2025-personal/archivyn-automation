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

import httpx

from . import bypass as _bypass

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
        bypass_tier: str = "direct",
    ):
        self.cache_dir = cache_dir
        self.rate_limit_delay = rate_limit_delay
        self.timeout = timeout
        self.max_retries = max_retries
        # "direct" | "curl_cffi" | "playwright"
        # Collectors targeting Cloudflare-walled sites set this to "curl_cffi"
        # so escalation to playwright is automatic on block detection.
        self.bypass_tier = bypass_tier
        self._last_request_time = 0.0
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
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; AcademicProfileBot/1.0; research-project)"
                },
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
        Fetch raw HTML/text with automatic Cloudflare bypass when available.
        Collectors scraping anti-bot-walled sites should use this instead of
        raw httpx.

        Ladder:
          1. If bypass_tier == "direct" → httpx; on 403/block, try curl_cffi.
          2. If bypass_tier == "curl_cffi" → start with curl_cffi, escalate.
          3. If bypass_tier == "playwright" → go straight to playwright.
        """
        await self._rate_limit()

        # Tier 1: httpx direct (always try first unless explicitly skipped)
        if self.bypass_tier == "direct":
            try:
                client = await self.get_client()
                resp = await client.get(url, params=params, headers=headers)
                if resp.status_code == 200 and not _bypass.is_blocked_html(resp.text):
                    return resp.text
                if resp.status_code in (401, 404, 410):
                    return None
                # Block detected — fall through to bypass ladder
            except Exception as e:
                logger.debug("[%s] direct fetch failed: %s; trying bypass", self.source_name, e)

        # Tier 2/3: bypass ladder
        start = "curl_cffi" if self.bypass_tier in ("direct", "curl_cffi") else "playwright"
        resp = await _bypass.fetch_with_ladder(
            url,
            params=params,
            headers=headers,
            timeout=self.timeout,
            start_tier=start,
        )
        if resp is None:
            return None
        print(f"    [{self.source_name}] 🛡️  Bypass tier '{resp.tier}' used for {url}")
        if resp.status >= 400 or _bypass.is_blocked_html(resp.text):
            return None
        return resp.text

    async def _fetch_any(
        self,
        url: str,
        *,
        headers: Optional[Dict] = None,
        accept_pdf: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch a URL and return both text and binary content, transparently
        escalating through the Cloudflare bypass ladder when the direct
        request 403s or returns a Cloudflare challenge page.

        Returns a dict {status, content_type, text, content, url, tier} or
        None on total failure.

        Use this for collectors that scrape generic web pages (can be HTML
        or PDF) — e.g. web_search_collector's seed fetch, osu_expertise's
        Knowledge Bank scrape.
        """
        await self._rate_limit()

        # Tier 1: httpx direct (attempt first for cheap pages)
        if self.bypass_tier in ("direct", "curl_cffi"):
            try:
                client = await self.get_client()
                resp = await client.get(url, headers=headers)
                ct = (resp.headers.get("content-type") or "").lower()
                # PDF path — binary; no block-check needed
                if accept_pdf and ("application/pdf" in ct or url.lower().endswith(".pdf")):
                    if resp.status_code == 200 and resp.content:
                        return {
                            "status": resp.status_code,
                            "content_type": "pdf",
                            "text": "",
                            "content": resp.content,
                            "url": str(resp.url),
                            "tier": "direct",
                        }
                # HTML / text path — check for block
                if resp.status_code == 200 and not _bypass.is_blocked_html(resp.text):
                    return {
                        "status": resp.status_code,
                        "content_type": "html" if "text/html" in ct or "text/plain" in ct else ct,
                        "text": resp.text,
                        "content": resp.content,
                        "url": str(resp.url),
                        "tier": "direct",
                    }
                if resp.status_code in (401, 404, 410):
                    return None
                # 403 / cf-challenge signature → escalate
            except Exception as e:
                logger.debug("[%s] direct fetch failed: %s; escalating", self.source_name, e)

        # Tier 2/3: bypass ladder (curl_cffi → playwright). These only return
        # text (HTML) — can't get raw binary through Playwright cheaply — so
        # for PDFs we're limited to Tier 1.
        if url.lower().endswith(".pdf"):
            return None
        start = "curl_cffi" if self.bypass_tier in ("direct", "curl_cffi") else "playwright"
        resp2 = await _bypass.fetch_with_ladder(
            url,
            headers=headers,
            timeout=self.timeout,
            start_tier=start,
        )
        if resp2 is None:
            return None
        if resp2.status >= 400 or _bypass.is_blocked_html(resp2.text):
            return None
        print(f"    [{self.source_name}] 🛡️  Bypass tier '{resp2.tier}' used for {url}")
        return {
            "status": resp2.status,
            "content_type": "html",
            "text": resp2.text,
            "content": resp2.text.encode("utf-8", errors="ignore"),
            "url": resp2.url,
            "tier": resp2.tier,
        }

    async def _get_json(self, url: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """GET request returning parsed JSON, with retry + rate limit + jitter."""
        for attempt in range(1, self.max_retries + 1):
            await self._rate_limit()
            try:
                client = await self.get_client()
                resp = await client.get(url, params=params)
                if resp.status_code == 429:
                    # Exponential backoff: 5s, 15s, 35s, 75s, 120s + jitter
                    wait = min(5 * (2 ** attempt) + random.uniform(0, 5), 120)
                    print(f"    [{self.source_name}] ⏳ Rate limited (429), retrying in {wait:.0f}s (attempt {attempt}/{self.max_retries})...")
                    await asyncio.sleep(wait)
                    continue
                # Non-retryable client errors — return None immediately
                if resp.status_code in (400, 401, 403, 404, 405, 410, 422):
                    return None
                resp.raise_for_status()
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
                    wait = min(5 * (2 ** attempt) + random.uniform(0, 5), 120)
                    print(f"    [{self.source_name}] ⏳ Rate limited (429), retrying in {wait:.0f}s (attempt {attempt}/{self.max_retries})...")
                    await asyncio.sleep(wait)
                    continue
                if resp.status_code in (400, 401, 403, 404, 405, 410, 422):
                    return None
                resp.raise_for_status()
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
