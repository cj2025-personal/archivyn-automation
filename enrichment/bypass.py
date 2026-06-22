"""
Three-tier HTTP fetch ladder for bypassing anti-bot walls.

Tier 1 (direct)    : httpx async — default for any well-behaved API.
Tier 2 (curl_cffi) : TLS fingerprint impersonation (real Chrome). Handles most
                     Cloudflare "Just a Moment" challenges without a browser.
Tier 3 (playwright): Headless browser with stealth patches. Slow; only used
                     when JS execution is required.

Collectors opt in by setting `bypass_tier` on BaseCollector. The ladder
auto-escalates on 403/Cloudflare-signature responses.

All tiers are graceful-optional: if curl_cffi or playwright aren't installed,
the code falls back to httpx and logs a warning instead of crashing.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# ── Cookie jar (Cloudflare clearance persistence) ─────────────────────────
# Cloudflare hands out a `cf_clearance` cookie after a successful challenge
# solve. We persist it per-domain on disk so that:
#   1. A later run starts with the cookie already attached (skipping the
#      "Just a Moment" challenge entirely)
#   2. Multiple collectors hitting the same domain share the cookie
#      (amortising one Playwright solve across many requests)
# Cookies are treated as valid for 50 minutes (Cloudflare default is 30-60).

_COOKIE_DIR = Path(os.getenv("BYPASS_COOKIE_DIR", "output/bypass_cookies"))
_COOKIE_TTL_SECONDS = 50 * 60
_cookie_cache_lock = asyncio.Lock()


def _cookie_file(domain: str) -> Path:
    _COOKIE_DIR.mkdir(parents=True, exist_ok=True)
    safe = domain.replace(":", "_").replace("/", "_")
    return _COOKIE_DIR / f"{safe}.json"


def _load_domain_cookies(domain: str) -> Dict[str, str]:
    path = _cookie_file(domain)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    saved_at = data.get("saved_at", 0)
    if time.time() - saved_at > _COOKIE_TTL_SECONDS:
        return {}
    return data.get("cookies") or {}


def _save_domain_cookies(domain: str, cookies: Dict[str, str]) -> None:
    if not cookies:
        return
    try:
        _cookie_file(domain).write_text(
            json.dumps({"saved_at": time.time(), "cookies": cookies}),
            encoding="utf-8",
        )
    except Exception as e:
        logger.debug("cookie save failed for %s: %s", domain, e)


def _domain_of(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def _merge_cookies(existing: Dict[str, str], new_cookies) -> Dict[str, str]:
    """Merge new cookies (from response) into existing dict."""
    out = dict(existing)
    try:
        # new_cookies can be httpx.Cookies, dict, or list of (name, value)
        if hasattr(new_cookies, "items"):
            for k, v in new_cookies.items():
                out[k] = v
        else:
            for k, v in new_cookies:
                out[k] = v
    except Exception:
        pass
    return out


# ── Block detection ───────────────────────────────────────────────────────

CLOUDFLARE_SIGNATURES = (
    "just a moment",
    "checking your browser",
    "cf-browser-verification",
    "enable javascript and cookies",
    "captcha",
    "recaptcha",
    "hcaptcha",
    "access denied",
    "403 forbidden",
    "ray id",
    "attention required",
    "please verify you are a human",
)


def is_blocked_html(html: str) -> bool:
    if not html:
        return False
    low = html[:8000].lower()
    return any(sig in low for sig in CLOUDFLARE_SIGNATURES)


# ── Tier availability ─────────────────────────────────────────────────────

try:
    from curl_cffi.requests import AsyncSession as _CurlAsyncSession  # type: ignore
    _CURL_CFFI_AVAILABLE = True
except Exception:
    _CurlAsyncSession = None  # type: ignore
    _CURL_CFFI_AVAILABLE = False

try:
    from playwright.async_api import async_playwright  # type: ignore
    _PLAYWRIGHT_AVAILABLE = True
except Exception:
    async_playwright = None  # type: ignore
    _PLAYWRIGHT_AVAILABLE = False


CHROME_IMPERSONATES = ["chrome120", "chrome119", "chrome116", "chrome110"]


@dataclass
class FetchResponse:
    status: int
    text: str
    headers: Dict[str, str]
    url: str
    tier: str  # "direct" | "curl_cffi" | "playwright"


# ── Tier 2: curl_cffi ─────────────────────────────────────────────────────

_curl_session: Optional["_CurlAsyncSession"] = None


async def _curl_cffi_fetch(
    url: str,
    params: Optional[Dict] = None,
    headers: Optional[Dict] = None,
    timeout: float = 30.0,
) -> Optional[FetchResponse]:
    if not _CURL_CFFI_AVAILABLE:
        return None
    global _curl_session
    if _curl_session is None:
        _curl_session = _CurlAsyncSession()
    try:
        impersonate = random.choice(CHROME_IMPERSONATES)
        domain = _domain_of(url)
        # Load any stored cf_clearance / __cf_bm cookies for this domain
        stored = _load_domain_cookies(domain) if domain else {}
        resp = await _curl_session.get(
            url,
            params=params,
            headers=headers or {},
            cookies=stored or None,
            impersonate=impersonate,
            timeout=timeout,
        )
        # Persist any new cookies the server set (especially cf_clearance)
        if domain:
            try:
                merged = _merge_cookies(stored, resp.cookies)
                # Only save if at least one cf_* cookie is present — avoids
                # cluttering disk with session cookies from non-CF sites.
                if any(k.startswith("cf_") or k == "__cf_bm" for k in merged):
                    _save_domain_cookies(domain, merged)
            except Exception:
                pass
        return FetchResponse(
            status=resp.status_code,
            text=resp.text,
            headers=dict(resp.headers),
            url=str(resp.url),
            tier="curl_cffi",
        )
    except Exception as e:
        logger.debug("curl_cffi fetch failed for %s: %s", url, e)
        return None


# ── Tier 3: playwright ────────────────────────────────────────────────────

_playwright_browser = None
_playwright_lock = asyncio.Lock()


async def _get_playwright_browser():
    global _playwright_browser
    if not _PLAYWRIGHT_AVAILABLE:
        return None
    async with _playwright_lock:
        if _playwright_browser is None:
            try:
                pw = await async_playwright().start()
                _playwright_browser = await pw.chromium.launch(headless=True)
            except Exception as e:
                logger.warning("Playwright launch failed: %s", e)
                return None
        return _playwright_browser


async def _playwright_fetch(
    url: str,
    timeout: float = 45.0,
) -> Optional[FetchResponse]:
    if not _PLAYWRIGHT_AVAILABLE:
        return None
    browser = await _get_playwright_browser()
    if browser is None:
        return None
    context = None
    page = None
    domain = _domain_of(url)
    try:
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1366, "height": 768},
            locale="en-US",
        )
        # Inject any previously-solved cf_clearance cookies for this domain
        if domain:
            stored = _load_domain_cookies(domain)
            if stored:
                try:
                    await context.add_cookies([
                        {
                            "name": k, "value": v,
                            "domain": domain, "path": "/",
                        } for k, v in stored.items()
                    ])
                except Exception:
                    pass
        page = await context.new_page()
        # Minimal stealth: hide webdriver property
        await page.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        )
        resp = await page.goto(url, wait_until="domcontentloaded", timeout=timeout * 1000)
        # Wait a bit for Cloudflare JS challenge to resolve
        await page.wait_for_timeout(2500)
        html = await page.content()
        status = resp.status if resp else 200

        # Capture cookies after potential CF solve and persist them
        if domain:
            try:
                all_cookies = await context.cookies(url)
                cf_cookies = {
                    c["name"]: c["value"] for c in all_cookies
                    if c["name"].startswith("cf_") or c["name"] == "__cf_bm"
                }
                if cf_cookies:
                    existing = _load_domain_cookies(domain)
                    _save_domain_cookies(domain, {**existing, **cf_cookies})
                    logger.info("bypass: saved %d CF cookies for %s", len(cf_cookies), domain)
            except Exception as e:
                logger.debug("cookie capture failed: %s", e)

        return FetchResponse(
            status=status,
            text=html,
            headers=dict(resp.headers) if resp else {},
            url=page.url,
            tier="playwright",
        )
    except Exception as e:
        logger.debug("playwright fetch failed for %s: %s", url, e)
        return None
    finally:
        if page is not None:
            try:
                await page.close()
            except Exception:
                pass
        if context is not None:
            try:
                await context.close()
            except Exception:
                pass


# ── Ladder ────────────────────────────────────────────────────────────────

async def fetch_with_ladder(
    url: str,
    *,
    params: Optional[Dict] = None,
    headers: Optional[Dict] = None,
    timeout: float = 30.0,
    start_tier: str = "curl_cffi",
) -> Optional[FetchResponse]:
    """
    Try curl_cffi, then playwright. Escalate if response looks blocked.
    Returns None if all tiers fail.

    Use this from collectors whose target sites gate with Cloudflare / anti-bot
    (Google Scholar, ResearchGate, some faculty pages).
    """
    order = []
    if start_tier == "curl_cffi":
        order = ["curl_cffi", "playwright"]
    elif start_tier == "playwright":
        order = ["playwright"]

    last: Optional[FetchResponse] = None
    for tier in order:
        if tier == "curl_cffi":
            resp = await _curl_cffi_fetch(url, params=params, headers=headers, timeout=timeout)
        else:
            resp = await _playwright_fetch(url, timeout=timeout)
        if resp is None:
            continue
        last = resp
        # Detect block: 403 or Cloudflare signatures in body
        if resp.status == 403 or is_blocked_html(resp.text):
            logger.info("bypass: %s tier hit block on %s, escalating", tier, url)
            continue
        return resp
    return last


async def shutdown_bypass():
    """Close shared sessions — call from orchestrator teardown."""
    global _curl_session, _playwright_browser
    if _curl_session is not None:
        try:
            await _curl_session.close()
        except Exception:
            pass
        _curl_session = None
    if _playwright_browser is not None:
        try:
            await _playwright_browser.close()
        except Exception:
            pass
        _playwright_browser = None


def bypass_available() -> Dict[str, bool]:
    """Runtime probe for which tiers are installed."""
    return {
        "curl_cffi": _CURL_CFFI_AVAILABLE,
        "playwright": _PLAYWRIGHT_AVAILABLE,
    }
