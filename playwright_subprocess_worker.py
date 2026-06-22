"""
Playwright subprocess worker.

Responsibilities:
- Fetch main page HTML, clean it, and extract profile text.
- Detect CV / Resume / Curriculum Vitae links (including cloud storage links).
- Download CV documents (HTML or binary) and extract text when possible.
- Detect personal website links, recursively crawl a few pages, and extract text.
- Return structured JSON to the controller with defensive defaults.
"""

import io
import json
import os
import re
import shutil
import ssl
import sys
import tempfile
import time
import threading
from datetime import datetime, timezone
from queue import Queue
import urllib.error
import urllib.parse
import urllib.request
from typing import Dict, List, Optional, Set, Tuple

from bs4 import BeautifulSoup, Tag

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
}

SSL_CONTEXT = ssl.create_default_context()
SSL_CONTEXT.check_hostname = False
SSL_CONTEXT.verify_mode = ssl.CERT_NONE

MEDIA_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp", ".svg", ".ico",
    ".mp4", ".mov", ".avi", ".wmv", ".mkv", ".webm",
    ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg",
    ".zip", ".rar", ".7z", ".tar", ".gz",
}

BLOCKED_TEXT_PATTERNS = (
    "request unsuccessful",
    "incapsula incident id",
    "access denied",
    "verify you are human",
    "captcha",
    "attention required",
    "please enable javascript to continue using this application",
    "javascript is disabled",
    "the system can't perform the operation now",
    "checking your browser",
    "security check",
    "not a robot",
)


def log(message: str) -> None:
    """Lightweight stderr logger."""
    sys.stderr.write(f"[Worker] {message}\n")
    sys.stderr.flush()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_url(url: str) -> str:
    return url.strip().split("#")[0]


def is_media_url(url: str) -> bool:
    try:
        path = urllib.parse.urlparse(url).path.lower()
    except Exception:
        return False
    return any(path.endswith(ext) for ext in MEDIA_EXTENSIONS)


def is_document_url(url: str) -> bool:
    try:
        path = urllib.parse.urlparse(url).path.lower()
    except Exception:
        return False
    return path.endswith((".pdf", ".doc", ".docx"))


class ScraperWorker:
    MAX_OLLAMA_CALLS = 3
    OLLAMA_TIMEOUT = 2.0

    def __init__(self) -> None:
        self.visited: Set[str] = set()
        self.cv_links_seen: Set[str] = set()
        self.playwright = None
        self.browser = None
        self.ollama_available: Optional[bool] = None
        self.ollama_calls = 0
        self.personal_site_cache: Dict[str, bool] = {}
        self._tesseract_available = shutil.which("tesseract") is not None

    @staticmethod
    def is_blocked_or_placeholder_text(text: str) -> bool:
        sample = (text or "").strip().lower()
        if not sample:
            return False
        return any(pattern in sample for pattern in BLOCKED_TEXT_PATTERNS)

    def storage_state_path(self, url: str) -> str:
        host = urllib.parse.urlparse(url).netloc.lower().replace(":", "_")
        host = re.sub(r"[^a-z0-9._-]+", "_", host) or "default"
        root = os.path.join(tempfile.gettempdir(), "ngo_automation_playwright_state")
        os.makedirs(root, exist_ok=True)
        return os.path.join(root, f"{host}.json")

    def extract_urls_from_text(self, text: str) -> List[str]:
        if not text:
            return []
        candidates: List[str] = []
        seen: Set[str] = set()
        patterns = re.findall(r"https?://[^\s<>\]\)\"']+|www\.[^\s<>\]\)\"']+", text, flags=re.I)
        for item in patterns:
            candidate = item.strip().rstrip(".,;:)]}>")
            if candidate.lower().startswith("www."):
                candidate = "https://" + candidate
            candidate = normalize_url(candidate)
            if not candidate or candidate in seen or is_media_url(candidate) or is_document_url(candidate):
                continue
            seen.add(candidate)
            candidates.append(candidate)
        return candidates

    def _extract_text_from_docx(self, data: bytes) -> str:
        try:
            from docx import Document  # type: ignore

            doc = Document(io.BytesIO(data))
            parts = [p.text.strip() for p in doc.paragraphs if (p.text or "").strip()]
            return "\n".join(parts).strip()
        except Exception:
            return ""

    # ------------------------------------------------------------------ fetch
    def fetch(self, url: str, timeout: int = 20) -> Tuple[str, Optional[str], Optional[str], Optional[bytes], Dict]:
        """
        Fetch a URL using urllib.
        Returns tuple: (kind, html_text, final_url, binary_content, meta)
        kind: "html", "binary", or "error".
        """
        start = time.time()
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, context=SSL_CONTEXT, timeout=timeout) as resp:
                final_url = resp.geturl()
                content_type = (resp.headers.get("Content-Type") or "").lower()
                raw = resp.read()
                meta = {
                    "status_code": getattr(resp, "status", None),
                    "content_type": content_type,
                    "content_length": len(raw) if raw else 0,
                    "etag": resp.headers.get("ETag"),
                    "last_modified": resp.headers.get("Last-Modified"),
                    "final_url": final_url,
                    "fetcher": "urllib",
                    "fetched_at": utc_now_iso(),
                    "duration_ms": int((time.time() - start) * 1000),
                }

            if content_type.startswith(("image/", "video/", "audio/")):
                return "error", None, final_url, None, meta

            if "pdf" in content_type or "word" in content_type or "octet" in content_type or raw[:4] == b"%PDF":
                return "binary", None, final_url, raw, meta

            text = self.decode_bytes(raw, content_type)
            if text is None:
                # could be binary masquerading as html
                return "binary", None, final_url, raw, meta

            if len(text) < 50:
                return "error", None, final_url, None, meta

            if self.is_blocked_or_placeholder_text(text):
                meta["error"] = "blocked_or_placeholder_content"
                return "error", text, final_url, None, meta

            return "html", text, final_url, None, meta
        except urllib.error.HTTPError as e:
            # Handle 403 Forbidden and other HTTP errors
            if e.code == 403:
                log(f"HTTP 403 Forbidden for {url} - trying Playwright fallback")
                # Try Playwright as fallback for 403 errors
                playwright_result = self.fetch_playwright(url, timeout=timeout)
                if playwright_result[0] != "error":
                    return playwright_result
                return "error", None, None, None, {
                    "status_code": 403,
                    "error": "HTTP 403 Forbidden",
                    "fetcher": "urllib",
                    "fetched_at": utc_now_iso(),
                    "duration_ms": int((time.time() - start) * 1000),
                }
            log(f"HTTP Error {e.code} for {url}: {e}")
            return "error", None, None, None, {
                "status_code": e.code,
                "error": str(e),
                "fetcher": "urllib",
                "fetched_at": utc_now_iso(),
                "duration_ms": int((time.time() - start) * 1000),
            }
        except Exception as e:
            log(f"Fetch error for {url}: {e}")
            return "error", None, None, None, {
                "status_code": None,
                "error": str(e),
                "fetcher": "urllib",
                "fetched_at": utc_now_iso(),
                "duration_ms": int((time.time() - start) * 1000),
            }

    def fetch_playwright(self, url: str, timeout: int = 30) -> Tuple[str, Optional[str], Optional[str], Optional[bytes], Dict]:
        """
        Optional Playwright fetch when urllib yields poor HTML (short/blocked).
        Returns same tuple schema.
        """
        start = time.time()
        try:
            from playwright.sync_api import sync_playwright  # type: ignore
        except Exception:
            return "error", None, None, None, {
                "status_code": None,
                "error": "playwright_unavailable",
                "fetcher": "playwright",
                "fetched_at": utc_now_iso(),
                "duration_ms": int((time.time() - start) * 1000),
            }

        if not self.browser:
            try:
                if not self.playwright:
                    self.playwright = sync_playwright().start()
                self.browser = self.playwright.chromium.launch(
                    headless=True,
                    args=["--disable-blink-features=AutomationControlled"],
                )
            except Exception as exc:
                self.browser = None
                return "error", None, None, None, {
                    "status_code": None,
                    "error": f"playwright_start_failed: {exc}",
                    "fetcher": "playwright",
                    "fetched_at": utc_now_iso(),
                    "duration_ms": int((time.time() - start) * 1000),
                }

        # Guard: a prior call may have started ``self.playwright`` but failed to
        # launch the browser (e.g. Chromium binary missing). Without this check
        # the next call would skip init and dereference a None browser, crashing
        # the whole worker subprocess and discarding all already-scraped seeds.
        if not self.browser:
            return "error", None, None, None, {
                "status_code": None,
                "error": "playwright_browser_unavailable",
                "fetcher": "playwright",
                "fetched_at": utc_now_iso(),
                "duration_ms": int((time.time() - start) * 1000),
            }

        state_path = self.storage_state_path(url)
        context = None
        page = None
        try:
            context_kwargs = {
                "accept_downloads": True,
                "java_script_enabled": True,
                "locale": "en-US",
                "timezone_id": "America/Chicago",
                "user_agent": HEADERS["User-Agent"],
                "extra_http_headers": {
                    "Accept-Language": "en-US,en;q=0.9",
                    "DNT": "1",
                    "Upgrade-Insecure-Requests": "1",
                },
                "service_workers": "block",
                "viewport": {"width": 1366, "height": 1900},
            }
            if os.path.exists(state_path):
                context_kwargs["storage_state"] = state_path
            context = self.browser.new_context(**context_kwargs)
            context.add_init_script(
                """
                Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
                window.chrome = window.chrome || { runtime: {} };
                Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});
                Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
                """
            )

            def _route_handler(route):
                try:
                    if route.request.resource_type in {"image", "media", "font"}:
                        route.abort()
                    else:
                        route.continue_()
                except Exception:
                    try:
                        route.continue_()
                    except Exception:
                        pass

            context.route("**/*", _route_handler)
            page = context.new_page()
            downloads = []
            page.on("download", lambda download: downloads.append(download))

            response = page.goto(url, wait_until="domcontentloaded", timeout=timeout * 1000)
            try:
                page.wait_for_load_state("networkidle", timeout=min(timeout * 1000, 10000))
            except Exception:
                pass
            page.wait_for_timeout(1200)
            try:
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                page.wait_for_timeout(400)
            except Exception:
                pass
            final_url = page.url
            meta = {
                "status_code": getattr(response, "status", None) if response else None,
                "content_type": (response.headers.get("content-type") if response else None),
                "content_length": None,
                "etag": (response.headers.get("etag") if response else None),
                "last_modified": (response.headers.get("last-modified") if response else None),
                "final_url": final_url,
                "fetcher": "playwright",
                "fetched_at": utc_now_iso(),
                "duration_ms": int((time.time() - start) * 1000),
            }
            if downloads:
                tmp_dir = tempfile.mkdtemp(prefix="pw_dl_")
                tmp_path = os.path.join(tmp_dir, downloads[0].suggested_filename or "download.bin")
                downloads[0].save_as(tmp_path)
                with open(tmp_path, "rb") as handle:
                    body = handle.read()
                meta["content_length"] = len(body)
                meta["download_url"] = downloads[0].url
                meta["content_type"] = meta.get("content_type") or "application/octet-stream"
                return "binary", None, final_url or downloads[0].url, body, meta
            if final_url.lower().endswith((".pdf", ".doc", ".docx")):
                body = b""
                try:
                    if response is not None:
                        body = response.body()
                except Exception:
                    body = b""
                return "binary", None, final_url, body or None, meta
            html = page.content()
            if meta is not None:
                meta["content_length"] = len(html) if html else 0
            try:
                context.storage_state(path=state_path)
            except Exception:
                pass
            return "html", html, final_url, None, meta
        except Exception as exc:
            try:
                if page is not None:
                    page.close()
            except Exception:
                pass
            return "error", None, None, None, {
                "status_code": None,
                "error": f"playwright_fetch_failed:{exc}",
                "fetcher": "playwright",
                "fetched_at": utc_now_iso(),
                "duration_ms": int((time.time() - start) * 1000),
            }
        finally:
            try:
                if page is not None:
                    page.close()
            except Exception:
                pass
            try:
                if context is not None:
                    context.close()
            except Exception:
                pass

    def fetch_playwright_cv(self, url: str, timeout: int = 60) -> Tuple[str, Optional[str], Optional[str], Optional[bytes], Dict]:
        """
        Playwright fetch tuned for CV/doc links. Attempts to get binary body if possible.
        """
        start = time.time()
        try:
            from playwright.sync_api import sync_playwright  # type: ignore
        except Exception:
            return "error", None, None, None, {
                "status_code": None,
                "error": "playwright_unavailable",
                "fetcher": "playwright",
                "fetched_at": utc_now_iso(),
                "duration_ms": int((time.time() - start) * 1000),
            }

        if not self.browser:
            try:
                if not self.playwright:
                    self.playwright = sync_playwright().start()
                self.browser = self.playwright.chromium.launch(
                    headless=True,
                    args=["--disable-blink-features=AutomationControlled"],
                )
            except Exception as exc:
                self.browser = None
                return "error", None, None, None, {
                    "status_code": None,
                    "error": f"playwright_start_failed: {exc}",
                    "fetcher": "playwright",
                    "fetched_at": utc_now_iso(),
                    "duration_ms": int((time.time() - start) * 1000),
                }

        # Guard: a prior call may have started ``self.playwright`` but failed to
        # launch the browser (e.g. Chromium binary missing). Without this check
        # the next call would skip init and dereference a None browser, crashing
        # the whole worker subprocess and discarding all already-scraped seeds.
        if not self.browser:
            return "error", None, None, None, {
                "status_code": None,
                "error": "playwright_browser_unavailable",
                "fetcher": "playwright",
                "fetched_at": utc_now_iso(),
                "duration_ms": int((time.time() - start) * 1000),
            }

        state_path = self.storage_state_path(url)
        context = None
        page = None
        try:
            context_kwargs = {
                "accept_downloads": True,
                "java_script_enabled": True,
                "locale": "en-US",
                "timezone_id": "America/Chicago",
                "user_agent": HEADERS["User-Agent"],
                "extra_http_headers": {
                    "Accept-Language": "en-US,en;q=0.9",
                    "DNT": "1",
                    "Upgrade-Insecure-Requests": "1",
                },
                "service_workers": "block",
                "viewport": {"width": 1366, "height": 1900},
            }
            if os.path.exists(state_path):
                context_kwargs["storage_state"] = state_path
            context = self.browser.new_context(**context_kwargs)
            context.add_init_script(
                """
                Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
                window.chrome = window.chrome || { runtime: {} };
                Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});
                Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
                """
            )

            def _route_handler(route):
                try:
                    if route.request.resource_type in {"image", "media", "font"}:
                        route.abort()
                    else:
                        route.continue_()
                except Exception:
                    try:
                        route.continue_()
                    except Exception:
                        pass

            context.route("**/*", _route_handler)
            page = context.new_page()
            downloads = []
            page.on("download", lambda download: downloads.append(download))
            response = page.goto(url, wait_until="domcontentloaded", timeout=timeout * 1000)
            try:
                page.wait_for_load_state("networkidle", timeout=min(timeout * 1000, 12000))
            except Exception:
                pass
            page.wait_for_timeout(1200)
            if response:
                final_url = response.url
                ctype = (response.headers.get("content-type") or "").lower()
                body = b""
                try:
                    body = response.body()
                except Exception:
                    body = b""

                # If we received PDF/Office binary
                meta = {
                    "status_code": getattr(response, "status", None),
                    "content_type": ctype,
                    "content_length": len(body) if body else 0,
                    "etag": response.headers.get("etag"),
                    "last_modified": response.headers.get("last-modified"),
                    "final_url": final_url,
                    "fetcher": "playwright",
                    "fetched_at": utc_now_iso(),
                    "duration_ms": int((time.time() - start) * 1000),
                }

                if any(x in ctype for x in ["pdf", "msword", "officedocument"]) or final_url.lower().endswith((".pdf", ".doc", ".docx")):
                    return "binary", None, final_url, body or None, meta

                if downloads:
                    tmp_dir = tempfile.mkdtemp(prefix="pw_dl_")
                    tmp_path = os.path.join(tmp_dir, downloads[0].suggested_filename or "download.bin")
                    downloads[0].save_as(tmp_path)
                    with open(tmp_path, "rb") as handle:
                        data = handle.read()
                    meta["content_length"] = len(data)
                    meta["download_url"] = downloads[0].url
                    meta["content_type"] = meta.get("content_type") or "application/octet-stream"
                    return "binary", None, downloads[0].url or final_url, data, meta

                # Try explicit download on SharePoint/OneDrive if viewer page
                if ("sharepoint.com" in final_url.lower() or "onedrive" in final_url.lower()):
                    download_url = url
                    if "download=1" not in download_url:
                        download_url = f"{download_url}{'&' if '?' in download_url else '?'}download=1"
                    try:
                        api_resp = page.request.get(download_url, timeout=timeout * 1000)
                        api_ct = (api_resp.headers.get("content-type") or "").lower()
                        api_body = api_resp.body()
                        if api_body and (api_body.startswith(b"%PDF") or "pdf" in api_ct or download_url.lower().endswith(".pdf")):
                            page.close()
                            meta.update({
                                "content_type": api_ct or meta.get("content_type"),
                                "content_length": len(api_body),
                                "final_url": download_url,
                            })
                            return "binary", None, download_url, api_body, meta
                    except Exception:
                        pass

                for handle in page.query_selector_all("a[href], button"):
                    try:
                        href = handle.get_attribute("href") or ""
                        text = (handle.inner_text() or "").strip().lower()
                    except Exception:
                        continue
                    candidate_url = urllib.parse.urljoin(final_url or url, href) if href else ""
                    likely_doc = is_document_url(candidate_url) or any(tok in text for tok in ["download", "curriculum vitae", "cv", "resume"])
                    if not likely_doc:
                        continue
                    if candidate_url and is_document_url(candidate_url):
                        ck, _, cf, cb, cm = self.fetch(candidate_url, timeout=timeout)
                        if ck == "binary" and cb:
                            return "binary", None, cf or candidate_url, cb, cm
                    try:
                        with page.expect_download(timeout=10000) as download_info:
                            handle.click()
                        download = download_info.value
                        tmp_dir = tempfile.mkdtemp(prefix="pw_dl_")
                        tmp_path = os.path.join(tmp_dir, download.suggested_filename or "download.bin")
                        download.save_as(tmp_path)
                        with open(tmp_path, "rb") as handle_fp:
                            data = handle_fp.read()
                        meta["content_length"] = len(data)
                        meta["download_url"] = download.url
                        meta["content_type"] = meta.get("content_type") or "application/octet-stream"
                        return "binary", None, download.url or final_url, data, meta
                    except Exception:
                        continue

                # Else treat as html
                html = page.content()
                try:
                    context.storage_state(path=state_path)
                except Exception:
                    pass
                return "html", html, final_url, None, meta

            html = page.content()
            final_url = page.url
            try:
                context.storage_state(path=state_path)
            except Exception:
                pass
            return "html", html, final_url, None, {
                "status_code": None,
                "content_type": None,
                "content_length": len(html) if html else 0,
                "etag": None,
                "last_modified": None,
                "final_url": final_url,
                "fetcher": "playwright",
                "fetched_at": utc_now_iso(),
                "duration_ms": int((time.time() - start) * 1000),
            }
        except Exception as exc:
            try:
                if page is not None:
                    page.close()
            except Exception:
                pass
            return "error", None, None, None, {
                "status_code": None,
                "error": f"playwright_fetch_failed:{exc}",
                "fetcher": "playwright",
                "fetched_at": utc_now_iso(),
                "duration_ms": int((time.time() - start) * 1000),
            }
        finally:
            try:
                if page is not None:
                    page.close()
            except Exception:
                pass
            try:
                if context is not None:
                    context.close()
            except Exception:
                pass

    def stop_playwright(self) -> None:
        try:
            if self.browser:
                self.browser.close()
        except Exception:
            pass
        try:
            if self.playwright:
                self.playwright.stop()
        except Exception:
            pass
        self.browser = None
        self.playwright = None

    # --------------------------------------------------------------- cleaning
    def clean_html(self, html: str) -> BeautifulSoup:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript", "svg", "iframe"]):
            tag.decompose()
        for tag in soup(["nav", "footer", "aside"]):
            tag.decompose()

        # Heuristically remove navigation/header/utility blocks that leak menu text
        nav_keywords = ["nav", "menu", "breadcrumb", "header", "footer", "toolbar", "utility"]
        candidates = soup.find_all(["div", "ul", "ol", "section", "header", "footer"])
        for tag in candidates:
            if not isinstance(tag, Tag):
                continue
            # Defensive attr extraction
            attrs_raw = getattr(tag, "attrs", {}) or {}
            tag_id = attrs_raw.get("id") or ""
            tag_classes = attrs_raw.get("class") or []
            attrs = f"{tag_id} {' '.join(tag_classes)}".strip().lower()
            role_attr = (attrs_raw.get("role") or "").lower()
            # Profile pages often place the person's core identity block in a
            # "hero" container. Keep those unless they are clearly navigation.
            if "profile-hero" in attrs or "hero-container" in attrs:
                continue
            if any(k in attrs for k in nav_keywords) or role_attr == "navigation":
                tag.decompose()
                continue
            # Remove massive link lists typical of nav bars
            links = tag.find_all("a")
            if len(links) > 25 and len(self.text_from_soup(tag)) < 400:
                tag.decompose()
        return soup
        return soup

    @staticmethod
    def text_from_soup(node: Optional[Tag]) -> str:
        if not node:
            return ""
        text = node.get_text(separator=" ")
        return " ".join(text.split())

    # ------------------------------------------------------- main content pick
    def pick_main_content(self, soup: BeautifulSoup, fallback_html: str) -> Tag:
        """
        Try to isolate the body content of the profile: prefer <main>, then
        article/section containing an <h1>/<h2>, else largest div with text.
        """
        candidates = []
        for selector in ("main", "article", "section"):
            tag = soup.find(selector)
            if tag and len(self.text_from_soup(tag)) > 300:
                return tag

        # try container with h1/h2
        headings = soup.find_all(["h1", "h2"])
        for h in headings:
            if not isinstance(h, Tag):
                continue
            if len(self.text_from_soup(h)) < 2:
                continue
            parent = h.find_parent(["section", "article", "div"])
            if parent:
                candidates.append(parent)
        if candidates:
            best = max(candidates, key=lambda n: len(self.text_from_soup(n)))
            if len(self.text_from_soup(best)) > 200:
                return best

        # fallback: largest div (skip obvious nav-like blocks)
        best_div = None
        best_len = 0
        for div in soup.find_all("div"):
            if not isinstance(div, Tag):
                continue
            attrs_raw = getattr(div, "attrs", {}) or {}
            div_id = attrs_raw.get("id") or ""
            div_classes = attrs_raw.get("class") or []
            attrs = f"{div_id} {' '.join(div_classes)}".strip().lower()
            if any(k in attrs for k in ["nav", "menu", "breadcrumb", "footer", "header"]):
                continue
            txt_len = len(self.text_from_soup(div))
            if txt_len > best_len:
                best_len = txt_len
                best_div = div
        return best_div or (soup.body or soup)

    # --------------------------------------------------------- profile fields
    def extract_profile_data(self, soup: BeautifulSoup, html: str, url: str) -> Dict:
        data = {"name": "", "email": "", "position": "", "department": "", "full_text": ""}

        # Name heuristics
        for tag_name in ("h1", "h2"):
            for tag in soup.find_all(tag_name):
                name = tag.get_text(strip=True)
                if name and 2 < len(name) < 120 and not any(x in name.lower() for x in ("university", "college", "school of", "menu", "navigation")):
                    data["name"] = name
                    break
            if data["name"]:
                break

        if not data["name"] and soup.title and soup.title.string:
            title = soup.title.string
            parts = re.split(r"[-|:]", title)
            if parts:
                candidate = parts[0].strip()
                if 2 < len(candidate) < 120:
                    data["name"] = candidate

        # Email
        emails = re.findall(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b", html)
        if emails:
            data["email"] = emails[0]

        # Position
        for selector in (".title", ".position", ".rank"):
            el = soup.select_one(selector)
            if el:
                txt = el.get_text(strip=True)
                if txt and len(txt) < 120:
                    data["position"] = txt
                    break

        # Department
        for selector in (".department", "[class*='department']", "[id*='department']"):
            el = soup.select_one(selector)
            if el:
                txt = el.get_text(strip=True)
                if txt and len(txt) < 160:
                    data["department"] = txt
                    break

        # Full text from main content area only, with a raw-html fallback when
        # the cleaned tree over-prunes JS-heavy profile layouts.
        main_node = self.pick_main_content(soup, html)
        full_text = self.text_from_soup(main_node)
        if len(full_text) < 120:
            raw_soup = BeautifulSoup(html, "html.parser")
            raw_main = raw_soup.find("main") or raw_soup.find("article") or raw_soup.body
            raw_text = self.text_from_soup(raw_main)
            if len(raw_text) > len(full_text):
                full_text = raw_text
        data["full_text"] = full_text
        return data

    # ------------------------------------------------------------ link logic
    @staticmethod
    def is_blocked_domain(url: str) -> bool:
        blocked = [
            "facebook.com",
            "twitter.com",
            "x.com",
            "linkedin.com",
            "instagram.com",
            "youtube.com",
            "google.com/search",
            "scholar.google",
            "researchgate.net",
            "orcid.org",
            "ieee.org",
            "acm.org",
            "springer.com",
            "elsevier.com",
            "jstor.org",
        ]
        lower_url = url.lower()
        return any(b in lower_url for b in blocked)

    @staticmethod
    def is_cv_link(url: str, text: str, title: str) -> bool:
        url_l = url.lower()
        text_blob = f"{text} {title}".lower()

        bad_words = [
            "curriculum-program",
            "course",
            "program",
            "degree",
            "syllabus",
            "handbook",
            "catalog",
        ]
        if any(x in url_l for x in bad_words):
            return False

        cv_words = ["cv", "curriculum vitae", "curriculum-vitae", "vitae", "resume", "résumé"]
        if any(w in text_blob for w in cv_words):
            return True
        if any(w in url_l for w in cv_words):
            return True

        return False

    def classify_personal_site(self, base_domain: str, url: str, text: str, title: str, force: bool = False) -> bool:
        """
        Heuristic personal site detector with optional LLM re-ranking.
        The LLM path is guarded: cache decisions, cap call count, and short timeouts.
        """
        url = normalize_url(url)
        if url in self.personal_site_cache:
            return self.personal_site_cache[url]

        domain = urllib.parse.urlparse(url).netloc.lower()
        if not domain or self.is_blocked_domain(url):
            self.personal_site_cache[url] = False
            return False

        text_l = (text or "").lower()
        title_l = (title or "").lower()

        # force flag overrides domain/block heuristics
        if force:
            self.personal_site_cache[url] = True
            return True

        # Priority: explicit anchor mentions
        explicit_hits = ["personal website", "personal page", "homepage", "home page", "my website", "my site", "webpage"]
        if any(h in text_l for h in explicit_hits) or any(h in title_l for h in explicit_hits):
            self.personal_site_cache[url] = True
            return True

        # Domain signals
        personal_domains = [
            "github.io",
            "netlify.app",
            "vercel.app",
            "people.",
            "users.",
            "personal.",
            "faculty.",
            "cs.utexas.edu/~",
            "mit.edu/~",
        ]
        if any(p in domain for p in personal_domains):
            self.personal_site_cache[url] = True
            return True

        # External domain different from base; allow if contains name-like pattern
        if domain != base_domain:
            if any(k in text_l for k in ["website", "page", "portfolio", "lab", "group"]):
                self.personal_site_cache[url] = True
                return True

        # If undecided, be conservative and bound LLM usage
        if self.ollama_calls >= self.MAX_OLLAMA_CALLS:
            self.personal_site_cache[url] = False
            return False

        # Optional LLM check (best effort, silent on failure)
        if self.ollama_available is False:
            self.personal_site_cache[url] = False
            return False

        # Probe availability once with a tiny request
        if self.ollama_available is None:
            try:
                from ollama import chat  # type: ignore

                probe_q: Queue = Queue(maxsize=1)

                def _probe():
                    try:
                        chat(model="llama3", messages=[{"role": "user", "content": "ping"}], stream=False)
                        probe_q.put(True)
                    except Exception:
                        probe_q.put(False)

                t_probe = threading.Thread(target=_probe, daemon=True)
                t_probe.start()
                t_probe.join(self.OLLAMA_TIMEOUT)
                if not t_probe.is_alive() and not probe_q.empty():
                    self.ollama_available = bool(probe_q.get_nowait())
                else:
                    self.ollama_available = False
            except Exception:
                self.ollama_available = False

        if not self.ollama_available:
            self.personal_site_cache[url] = False
            return False

        try:
            from ollama import chat  # type: ignore

            q: Queue = Queue(maxsize=1)

            def _llm_call():
                try:
                    resp = chat(
                        model="llama3",
                        messages=[
                            {
                                "role": "user",
                                "content": (
                                    "Decide if the URL is the personal home page of an individual professor. "
                                    "Respond only yes or no.\n"
                                    f"URL: {url}\n"
                                    f"Link text: {text}\n"
                                    f"Title: {title}\n"
                                ),
                            }
                        ],
                        stream=False,
                    )
                    q.put(resp)
                except Exception as exc:
                    q.put(exc)

            self.ollama_calls += 1
            t = threading.Thread(target=_llm_call, daemon=True)
            t.start()
            t.join(self.OLLAMA_TIMEOUT)

            if t.is_alive():
                self.ollama_available = False
                self.personal_site_cache[url] = False
                return False

            if q.empty():
                self.personal_site_cache[url] = False
                return False

            resp = q.get_nowait()
            if isinstance(resp, Exception):
                self.ollama_available = False
                self.personal_site_cache[url] = False
                return False

            decision = (resp.get("message", {}) or {}).get("content", "").lower()
            if "yes" in decision:
                self.personal_site_cache[url] = True
                return True
        except Exception:
            self.ollama_available = False

        self.personal_site_cache[url] = False
        return False

    def extract_links(self, soup: BeautifulSoup, base_url: str) -> List[Dict]:
        """
        Extract anchors from given soup. Always include explicit personal-website
        anchors even if they would normally be filtered out by container.
        """
        links: List[Dict] = []
        explicit_personal_keywords = ["personal website", "personal page", "homepage", "home page", "my website", "my site"]
        for a in soup.find_all("a", href=True):
            if not isinstance(a, Tag):
                continue
            href = a.get("href", "").strip()
            if not href or href.startswith("#") or href.startswith("javascript:") or href.startswith("mailto:") or href.startswith("tel:"):
                continue
            text = a.get_text(strip=True) or ""
            title = a.get("title", "") or ""
            full_url = urllib.parse.urljoin(base_url, href)
            if is_media_url(full_url):
                continue
            # force-include explicit personal website anchors
            if any(k in text.lower() for k in explicit_personal_keywords) or any(k in title.lower() for k in explicit_personal_keywords):
                links.append({"url": normalize_url(full_url), "text": text, "title": title, "force_personal": True})
                continue
            links.append({"url": normalize_url(full_url), "text": text, "title": title})
        return links

    # ------------------------------------------------------- text quality gate
    @staticmethod
    def is_reasonable_text(text: str) -> bool:
        """Heuristic to reject mostly-binary / garbled text blobs."""
        if not text:
            return False
        length = len(text)
        if length < 50:
            return False
        if length > 120000:  # skip oversized blobs
            return False
        ascii_chars = sum(1 for ch in text if 32 <= ord(ch) < 127)
        ratio = ascii_chars / max(1, length)
        return ratio >= 0.6

    # ------------------------------------------------------- CV extraction
    def extract_cv(self, url: str) -> Dict:
        """Extract CV content, preserving the original link even if redirects occur."""
        original_url = url
        if url in self.cv_links_seen:
            return {"url": original_url, "status": "duplicate"}
        self.cv_links_seen.add(url)

        # Skip social media links entirely (do not attempt to scrape them)
        blocked_social = ["linkedin.com", "facebook.com", "twitter.com", "x.com", "instagram.com", "youtube.com"]
        if any(dom in url.lower() for dom in blocked_social):
            return {
                "url": original_url,
                "status": "skipped",
                "note": "social_media_link",
            }

        # Handle common cloud download toggles but keep the original visible URL
        if "sharepoint.com" in url and "download=1" not in url:
            url = f"{url}?download=1"
        if "drive.google.com" in url and "uc?export=download" not in url:
            url = url.replace("view?usp=sharing", "uc?export=download")

        # First try urllib fast path
        kind, html, final_url, binary, meta = self.fetch(url, timeout=30)
        resolved_url = final_url or url
        result = {
            "url": original_url,
            "resolved_url": resolved_url,
            "type": "cv_document",
            "status": "failed",
            "fetch_metadata": meta,
            "fetch_attempts": [meta] if meta else [],
        }

        def success_from_html(html_text: str) -> Dict:
            soup = self.clean_html(html_text)
            text = self.text_from_soup(soup)
            result.update(
                {
                    "status": "success",
                    "content": text[:20000],
                    "full_length": len(text),
                }
            )
            return result

        def success_from_binary(bin_data: Optional[bytes]) -> Dict:
            # Try to read document text if possible; otherwise return binary pointer
            if bin_data:
                text = self.extract_document_text(
                    bin_data,
                    source_url=resolved_url or original_url,
                    content_type=str((result.get("fetch_metadata") or {}).get("content_type") or ""),
                )
                if text:
                    result.update(
                        {
                            "status": "success",
                            "content": text[:20000],
                            "full_length": len(text),
                            "note": "Extracted from PDF binary",
                        }
                    )
                    return result
            result.update(
                {
                    "status": "binary_document",
                    "content_length": len(bin_data) if bin_data else 0,
                    "note": "Binary CV detected; downstream parser needed",
                }
            )
            return result

        # If urllib gave binary/html ok, return immediately
        if kind == "binary":
            return success_from_binary(binary)
        if (
            kind == "html"
            and html
            and "login.microsoftonline.com" not in (resolved_url or "")
            and not self.is_blocked_or_placeholder_text(html)
        ):
            return success_from_html(html)

        # If redirected to Microsoft login or fetch failed, try Playwright to get the document directly
        if kind == "error" or "login.microsoftonline.com" in (resolved_url or "") or (html and self.is_blocked_or_placeholder_text(html)):
            pkind, phtml, pfinal, pbinary, pmeta = self.fetch_playwright_cv(original_url)
            resolved_url = pfinal or resolved_url
            result["resolved_url"] = resolved_url
            if pmeta:
                result["fetch_metadata"] = pmeta
                result["fetch_attempts"].append(pmeta)

            if pkind == "binary":
                return success_from_binary(pbinary)
            if (
                pkind == "html"
                and phtml
                and "login.microsoftonline.com" not in (resolved_url or "")
                and not self.is_blocked_or_placeholder_text(phtml)
            ):
                return success_from_html(phtml)

            result.update(
                {
                    "status": "auth_required",
                    "note": "Requires authentication to access SharePoint",
                }
            )
            return result

        # Fallback unknown
        result["error"] = "unknown_format"
        return result

    # ------------------------------------------------------- PDF helper
    def extract_pdf_text(self, data: bytes) -> str:
        """
        PDF text extraction with PyMuPDF first, then pdfplumber / PyPDF2.
        Uses OCR as a last resort when Tesseract is installed.
        Returns empty string on failure.
        """
        text = ""

        try:
            import fitz  # type: ignore

            doc = fitz.open(stream=data, filetype="pdf")
            parts = []
            for page in doc:
                try:
                    page_text = page.get_text("text", sort=True) or ""
                except Exception:
                    page_text = ""
                if page_text.strip():
                    parts.append(page_text)
                    continue
                if self._tesseract_available:
                    try:
                        text_page = page.get_textpage_ocr()
                        ocr_text = page.get_text("text", textpage=text_page, sort=True) or ""
                    except Exception:
                        ocr_text = ""
                    if ocr_text.strip():
                        parts.append(ocr_text)
            text = "\n".join(part.strip() for part in parts if part and part.strip()).strip()
            if text:
                return text
        except Exception:
            pass

        # Try pdfplumber next
        try:
            import pdfplumber  # type: ignore

            with pdfplumber.open(io.BytesIO(data)) as pdf:
                parts = []
                for page in pdf.pages:
                    try:
                        parts.append(page.extract_text() or "")
                    except Exception:
                        continue
                text = "\n".join(parts).strip()
                if text:
                    return text
        except Exception:
            pass

        # Fallback to PyPDF2
        try:
            import PyPDF2  # type: ignore

            reader = PyPDF2.PdfReader(io.BytesIO(data))
            parts = []
            for page in reader.pages:
                try:
                    parts.append(page.extract_text() or "")
                except Exception:
                    continue
            text = "\n".join(parts).strip()
        except Exception:
            text = ""
        return text

    def extract_document_text(self, data: bytes, *, source_url: str = "", content_type: str = "") -> str:
        lower_url = (source_url or "").lower()
        lower_ct = (content_type or "").lower()
        if lower_url.endswith(".docx") or "officedocument" in lower_ct:
            return self._extract_text_from_docx(data)
        return self.extract_pdf_text(data)

    # ------------------------------------------------------- decode helper
    def decode_bytes(self, data: bytes, content_type: str) -> Optional[str]:
        """
        Try to decode bytes to text using charset hints, fallback to utf-8/latin-1.
        Returns None if it looks binary.
        """
        if not data:
            return ""

        # BOM sniffing
        if data.startswith(b"\xff\xfe") or data.startswith(b"\xfe\xff"):
            try:
                return data.decode("utf-16", errors="replace")
            except Exception:
                pass
        if data.startswith(b"\xff\xfe\x00\x00") or data.startswith(b"\x00\x00\xfe\xff"):
            try:
                return data.decode("utf-32", errors="replace")
            except Exception:
                pass

        # charset from content-type header
        charset = None
        if "charset=" in content_type:
            try:
                charset = content_type.split("charset=")[-1].split(";")[0].strip()
            except Exception:
                charset = None

        if charset:
            try:
                return data.decode(charset, errors="replace")
            except Exception:
                pass

        # Heuristic: many zero bytes -> utf-16le
        if sum(1 for b in data[:200] if b == 0) > 50:
            try:
                return data.decode("utf-16le", errors="replace")
            except Exception:
                pass

        # Try utf-8, then latin-1
        for enc in ("utf-8", "latin-1"):
            try:
                return data.decode(enc, errors="replace")
            except Exception:
                continue

        return None

    # ------------------------------------------------------- personal sites
    def crawl_site(self, url: str, depth: int, max_depth: int, base_domain: str, allow_garbled: bool = False) -> Dict:
        url = normalize_url(url)
        if url in self.visited or depth > max_depth:
            return {"url": url, "status": "skipped"}
        self.visited.add(url)

        kind, html, final_url, _, meta = self.fetch(url)
        # fallback to Playwright if fetch failed or very short
        if kind != "html" or not html or len(html) < 200 or self.is_blocked_or_placeholder_text(html):
            pkind, phtml, pfinal, _, pmeta = self.fetch_playwright(url)
            if pkind == "html" and phtml and not self.is_blocked_or_placeholder_text(phtml):
                kind, html, final_url, meta = pkind, phtml, pfinal, pmeta
            else:
                return {"url": url, "status": "failed"}

        soup = self.clean_html(html)
        text = self.text_from_soup(soup)
        if not allow_garbled and not self.is_reasonable_text(text):
            return {"url": url, "status": "skipped", "note": "Content not textual/garbled"}
        result = {
            "url": final_url or url,
            "source_url": url,
            "resolved_url": final_url or url,
            "status": "success",
            "content": (text[:15000] if text else ""),
            "full_length": len(text),
            "subpages": [],
            "fetch_metadata": meta,
        }

        # collect CVs on this page too
        links = self.extract_links(soup, final_url or url)
        cv_candidates = [
            l for l in links if self.is_cv_link(l["url"], l["text"], l["title"])
        ]
        nested_cv = []
        for cv in cv_candidates:
            nested_cv.append(self.extract_cv(cv["url"]))
        if nested_cv:
            result["cv_documents"] = nested_cv

        # recurse into a few relevant subpages (same domain, depth limited)
        subpages = []
        for l in links:
            sub_domain = urllib.parse.urlparse(l["url"]).netloc.lower()
            if sub_domain != base_domain:
                continue
            if any(k in l["url"].lower() for k in ["contact", "privacy", "terms", "login", "signin", "search"]):
                continue
            if len(subpages) >= 5:
                break
            subpage = self.crawl_site(l["url"], depth + 1, max_depth, base_domain)
            if subpage.get("status") == "success":
                subpages.append(subpage)
        result["subpages"] = subpages
        return result

    # -------------------------------------------------------------- run main
    def run(self, url: str) -> Dict:
        url = normalize_url(url)
        log(f"Job start: {url}")

        kind, html, final_url, binary, meta = self.fetch(url)
        if kind == "binary":
            document_text = self.extract_document_text(
                binary or b"",
                source_url=final_url or url,
                content_type=str((meta or {}).get("content_type") or ""),
            )
            if not document_text.strip():
                pkind, _, pfinal, pbinary, pmeta = self.fetch_playwright_cv(url)
                if pkind == "binary" and pbinary:
                    binary = pbinary
                    final_url = pfinal or final_url
                    meta = pmeta or meta
                    document_text = self.extract_document_text(
                        binary,
                        source_url=final_url or url,
                        content_type=str((meta or {}).get("content_type") or ""),
                    )
            if document_text.strip():
                base_url = final_url or url
                base_domain = urllib.parse.urlparse(base_url).netloc.lower()
                personal_sites: List[Dict] = []
                for candidate_url in self.extract_urls_from_text(document_text):
                    if len(personal_sites) >= 2:
                        break
                    if self.is_blocked_domain(candidate_url):
                        continue
                    site_data = self.crawl_site(
                        candidate_url,
                        depth=0,
                        max_depth=1,
                        base_domain=urllib.parse.urlparse(candidate_url).netloc.lower() or base_domain,
                        allow_garbled=False,
                    )
                    if site_data.get("status") == "success":
                        personal_sites.append(site_data)
                return {
                    "status": "success",
                    "profile_data": {
                        "name": "",
                        "email": "",
                        "position": "",
                        "department": "",
                        "full_text": document_text,
                    },
                    "profile_page": {
                        "source_url": url,
                        "resolved_url": base_url,
                        "content": document_text,
                        "page_title": "",
                        "fetch_metadata": meta,
                    },
                    "cv_documents": [
                        {
                            "url": url,
                            "resolved_url": base_url,
                            "type": "cv_document",
                            "status": "success",
                            "content": document_text[:20000],
                            "full_length": len(document_text),
                            "fetch_metadata": meta,
                            "note": "document_seed",
                        }
                    ],
                    "personal_websites": personal_sites,
                    "summary": {
                        "profile_text_length": len(document_text),
                        "cv_count": 1,
                        "website_count": len(personal_sites),
                        "total_subpages": sum(len(site.get("subpages", [])) for site in personal_sites),
                    },
                }
            self.stop_playwright()
            return {"status": "error", "error": "Failed to extract text from binary seed"}

        if kind != "html" or not html or len(html) < 300 or self.is_blocked_or_placeholder_text(html):
            # fallback to Playwright if available
            pkind, phtml, pfinal, _, pmeta = self.fetch_playwright(url)
            if pkind == "html" and phtml and not self.is_blocked_or_placeholder_text(phtml):
                kind, html, final_url, meta = pkind, phtml, pfinal, pmeta
            else:
                self.stop_playwright()
                return {"status": "error", "error": f"Failed to fetch main page ({kind})"}

        base_url = final_url or url
        base_domain = urllib.parse.urlparse(base_url).netloc.lower()

        soup = self.clean_html(html)
        main_content = self.pick_main_content(soup, html)
        profile_data = self.extract_profile_data(soup, html, base_url)

        # Extract links from main content; also pull explicit personal website anchors from whole page
        main_links = self.extract_links(main_content, base_url)
        all_links = self.extract_links(soup, base_url)
        # merge, preferring main_links order, and include any force_personal anchors from all_links
        seen = set()
        links: List[Dict] = []
        for l in main_links + all_links:
            key = (l.get("url"), l.get("text"), l.get("title"))
            if key in seen:
                continue
            seen.add(key)
            if l.get("force_personal") or l in main_links:
                links.append(l)
        cv_docs: List[Dict] = []
        personal_sites: List[Dict] = []

        # CV detection on main page
        for link in links:
            if self.is_cv_link(link["url"], link["text"], link["title"]):
                cv_docs.append(self.extract_cv(link["url"]))

        # Personal website detection + recursion
        for link in links:
            if len(personal_sites) >= 3:
                break
            force_personal = bool(link.get("force_personal"))
            if self.classify_personal_site(base_domain, link.get("url", ""), link.get("text", ""), link.get("title", ""), force=force_personal):
                site_data = self.crawl_site(
                    link["url"],
                    depth=0,
                    max_depth=2,
                    base_domain=urllib.parse.urlparse(link["url"]).netloc.lower() or base_domain,
                    allow_garbled=force_personal,
                )
                if site_data.get("status") == "success":
                    personal_sites.append(site_data)
                    # merge any CVs discovered inside personal site
                    for nested_cv in site_data.get("cv_documents", []):
                        if isinstance(nested_cv, dict) and nested_cv.get("url") not in {c.get("url") for c in cv_docs if isinstance(c, dict)}:
                            cv_docs.append(nested_cv)

        # Extract plain-text URLs from the profile and CVs to recover scholar-owned websites
        text_url_candidates: List[str] = []
        text_url_candidates.extend(self.extract_urls_from_text(profile_data.get("full_text", "")))
        for cv in cv_docs:
            if isinstance(cv, dict) and cv.get("status") == "success":
                text_url_candidates.extend(self.extract_urls_from_text(cv.get("content", "")))
        seen_text_urls: Set[str] = set()
        for candidate_url in text_url_candidates:
            candidate_url = normalize_url(candidate_url)
            if (
                not candidate_url
                or candidate_url in seen_text_urls
                or self.is_blocked_domain(candidate_url)
                or is_document_url(candidate_url)
            ):
                continue
            seen_text_urls.add(candidate_url)
            if len(personal_sites) >= 4:
                break
            site_data = self.crawl_site(
                candidate_url,
                depth=0,
                max_depth=1,
                base_domain=urllib.parse.urlparse(candidate_url).netloc.lower() or base_domain,
                allow_garbled=False,
            )
            if site_data.get("status") == "success":
                if site_data.get("url") not in {s.get("url") for s in personal_sites if isinstance(s, dict)}:
                    personal_sites.append(site_data)
                for nested_cv in site_data.get("cv_documents", []):
                    if isinstance(nested_cv, dict) and nested_cv.get("url") not in {c.get("url") for c in cv_docs if isinstance(c, dict)}:
                        cv_docs.append(nested_cv)

        summary = {
            "profile_text_length": len(profile_data.get("full_text", "")),
            "cv_count": len(cv_docs),
            "website_count": len(personal_sites),
            "total_subpages": sum(len(site.get("subpages", [])) for site in personal_sites),
        }

        return {
            "status": "success",
            "profile_data": profile_data,
            "profile_page": {
                "source_url": url,
                "resolved_url": base_url,
                "content": profile_data.get("full_text", ""),
                "page_title": (soup.title.string.strip() if soup.title and soup.title.string else ""),
                "fetch_metadata": meta,
            },
            "cv_documents": cv_docs,
            "personal_websites": personal_sites,
            "summary": summary,
        }


# ---------------------------------------------------------------------------
# Runtime patch: stricter CV detection that ignores image assets
# ---------------------------------------------------------------------------

def _patched_is_cv_link(url: str, text: str, title: str) -> bool:
    """
    Heuristic CV detector that:
    - Ignores obvious non-document assets (jpg/png/etc.)
    - Looks for CV/resume indicators in anchor text or URL.
    """
    url_l = (url or "").lower()
    text_blob = f"{text or ''} {title or ''}".lower()

    # Ignore image/icon assets outright
    try:
        import urllib.parse

        path = urllib.parse.urlparse(url_l).path or ""
    except Exception:
        path = url_l

    image_exts = (".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".bmp", ".tif", ".tiff", ".ico")
    if any(path.endswith(ext) for ext in image_exts):
        return False

    # Exclude course/program/handbook links
    bad_words = [
        "curriculum-program",
        "course",
        "program",
        "degree",
        "syllabus",
        "handbook",
        "catalog",
    ]
    if any(x in url_l for x in bad_words):
        return False

    # Positive CV indicators
    cv_words = ["cv", "curriculum vitae", "curriculum-vitae", "vitae", "resume", "résumé"]
    if any(w in text_blob for w in cv_words):
        return True
    if any(w in url_l for w in cv_words):
        return True

    return False


# Apply patch so all ScraperWorker instances use the stricter detector
ScraperWorker.is_cv_link = staticmethod(_patched_is_cv_link)


if __name__ == "__main__":
    try:
        if sys.platform == "win32":
            sys.stdout.reconfigure(encoding="utf-8")

        input_data = sys.stdin.read()
        if not input_data:
            raise ValueError("No input provided")

        data = json.loads(input_data)
        target_url = data.get("url")
        if not target_url:
            raise ValueError("Missing url in payload")

        worker = ScraperWorker()
        result = worker.run(target_url)
        print(json.dumps(result, ensure_ascii=False))
    except Exception as exc:
        import traceback

        error_trace = traceback.format_exc()
        log(f"Critical Error: {exc}")
        log(f"Traceback:\n{error_trace}")
        print(json.dumps({"status": "error", "error": str(exc), "traceback": error_trace}))
        sys.exit(1)
    finally:
        try:
            worker.stop_playwright()
        except Exception:
            pass
