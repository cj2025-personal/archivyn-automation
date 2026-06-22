"""
Web Search collector — the most accurate enrichment approach.

Strategy:
1. DuckDuckGo search: "professor name" + "Ohio State University" + department
2. Additional targeted searches for publications, research papers, grants
3. Visit each result URL — scrape the MAIN CONTENT only (strip nav/sidebar/footer/ads)
4. RECURSIVE HOPPING: extract outbound links from each page and follow ones that
   look like they belong to the professor (publications, CV, lab, project pages).
   Continues up to MAX_DEPTH hops from the original search results.
5. Download and parse PDFs (research papers, CVs, publications)
6. Detect and skip Cloudflare/captcha-protected pages gracefully
7. Validate every page mentions the correct professor

Requires: pip install duckduckgo-search
PDF parsing uses PyMuPDF (already installed).
"""

import asyncio
import io
import logging
import re
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urlparse, urljoin

from .base_collector import BaseCollector, CollectorResult, ProfessorQuery
from .validation import normalize, strict_identity_match

logger = logging.getLogger(__name__)

# ── Domain rules ───────────────────────────────────────────────────────

SKIP_DOMAINS = {
    "facebook.com", "twitter.com", "x.com", "instagram.com",
    "tiktok.com", "pinterest.com", "reddit.com",
    "amazon.com", "ebay.com", "walmart.com",
    "login.", "signin.", "accounts.",
    "play.google.com", "apps.apple.com",
    "fonts.googleapis.com", "cdn.", "static.",
    "google.com/recaptcha",
    # LinkedIn: ToS prohibits automated scraping. Public content we can see
    # without login is minimal (usually just name + headline); not worth
    # the ToS exposure for the small signal. Skip entirely.
    "linkedin.com",
}

PRIORITY_DOMAINS = [
    "osu.edu",
    "scholar.google.com",
    "researchgate.net",
    "pubmed.ncbi.nlm.nih.gov",
    "ncbi.nlm.nih.gov",
    "arxiv.org",
    "orcid.org",
    "semanticscholar.org",
    "openalex.org",
    "nsf.gov",
    "nih.gov",
    "acm.org",
    "ieee.org",
    "springer.com",
    "sciencedirect.com",
    "wiley.com",
    "nature.com",
    "plos.org",
    "mdpi.com",
    "frontiersin.org",
    "biorxiv.org",
    "medrxiv.org",
    "experts.osu.edu",
]

# Domains worth hopping INTO from a professor's page
HOP_WORTHY_DOMAINS = {
    "osu.edu", "scholar.google.com", "researchgate.net",
    "pubmed.ncbi.nlm.nih.gov", "ncbi.nlm.nih.gov",
    "arxiv.org", "orcid.org", "semanticscholar.org",
    "acm.org", "ieee.org", "springer.com", "sciencedirect.com",
    "wiley.com", "nature.com", "plos.org", "mdpi.com",
    "frontiersin.org", "biorxiv.org", "medrxiv.org",
    "tandfonline.com", "sagepub.com", "oup.com", "cell.com",
    "acs.org", "experts.osu.edu", "nsf.gov", "nih.gov",
}

# URL path segments that signal a link is about content, not navigation
HOP_WORTHY_PATHS = [
    "/publication", "/article", "/paper", "/abstract",
    "/doi/", "/abs/", "/full/", "/pdf",
    "/people/", "/faculty/", "/profile",
    "/project", "/lab", "/research",
    "/grant", "/award",
    "/cv", "/vita", "/resume",
    "/record/", "/works/",
]

# URL path segments that signal generic navigation — NEVER hop into these
SKIP_PATHS = [
    "/login", "/signin", "/signup", "/register", "/cart",
    "/search", "/contact", "/about", "/privacy", "/terms",
    "/help", "/faq", "/support", "/donate", "/give",
    "/apply", "/admissions", "/jobs", "/careers",
    "/sitemap", "/feed", "/rss",
    "#",  # anchor links on the same page
]

# Cloudflare / bot-protection signatures in HTML
BLOCK_SIGNATURES = [
    "just a moment",
    "checking your browser",
    "cf-browser-verification",
    "enable javascript and cookies",
    "captcha",
    "recaptcha",
    "hcaptcha",
    "access denied",
    "403 forbidden",
    "please verify you are a human",
    "ray id",
]

# HTML elements to STRIP (non-content chrome)
STRIP_TAGS = [
    "script", "style", "noscript", "svg", "img", "iframe",
    "video", "audio", "nav", "footer", "header", "aside", "form",
]
STRIP_ROLES = [
    "navigation", "banner", "contentinfo", "complementary",
    "search", "menu", "menubar", "toolbar",
]
STRIP_CLASSES = [
    r"nav\b", r"navbar", r"sidebar", r"side-bar", r"menu\b",
    r"footer", r"header", r"breadcrumb", r"pagination",
    r"cookie", r"banner", r"advert", r"social",
    r"share", r"related", r"comment", r"widget",
    r"popup", r"modal", r"overlay", r"toolbar",
    r"search-form", r"login", r"signup",
    r"skip-link", r"screen-reader",
]
STRIP_IDS = [
    r"nav", r"navbar", r"sidebar", r"menu",
    r"footer", r"header", r"cookie", r"banner",
    r"comment", r"ad-", r"social",
]

# ── Limits ─────────────────────────────────────────────────────────────

MAX_SEED_PAGES = 10        # max pages from DDG search results (depth 0)
MAX_TOTAL_PAGES = 25       # absolute max across all depths
MAX_DEPTH = 2              # how many hops from seed pages
MAX_HOP_LINKS = 8          # max links to follow from a single page
MAX_TEXT_PER_PAGE = 8000
MAX_PDF_PAGES = 20
MAX_PDF_TEXT = 15000


class WebSearchCollector(BaseCollector):
    """Search the web for a professor, scrape pages + PDFs, follow links recursively."""

    def __init__(self, **kwargs):
        kwargs.setdefault("rate_limit_delay", 2.5)
        kwargs.setdefault("timeout", 25.0)
        # Enable Cloudflare-bypass ladder so that when a scraped page returns a
        # cf-challenge body, we escalate to curl_cffi / playwright before
        # giving up (previous behaviour just returned None).
        kwargs.setdefault("bypass_tier", "curl_cffi")
        super().__init__(**kwargs)

    @property
    def source_name(self) -> str:
        return "web_search"

    async def collect(self, query: ProfessorQuery) -> CollectorResult:
        # Step 1: DDG search → seed URLs
        seed_urls = await self._search_ddg(query)
        if not seed_urls:
            return self._make_result(query, success=False,
                                     error="No search results from DuckDuckGo")
        print(f"    [web_search] Found {len(seed_urls)} seed URLs")

        # Step 2: Recursive scrape — seeds are depth 0, follow links up to MAX_DEPTH
        visited: Set[str] = set()
        all_pages: List[Dict] = []

        queue: List[tuple] = [(u, 0) for u in seed_urls]  # (url_info, depth)

        while queue and len(all_pages) < MAX_TOTAL_PAGES:
            url_info, depth = queue.pop(0)
            url = url_info["url"]

            # Skip already visited
            canon = self._canonical(url)
            if canon in visited:
                continue
            visited.add(canon)

            # Enforce seed limit
            if depth == 0 and sum(1 for p in all_pages if p.get("depth", 0) == 0) >= MAX_SEED_PAGES:
                continue

            try:
                page, raw_html = await self._fetch_and_extract(url, url_info, query)
            except Exception as e:
                logger.debug("[web_search] Error on %s: %s", url, e)
                continue

            if not page:
                continue

            page["depth"] = depth
            all_pages.append(page)

            # Step 3: Extract outbound links and queue for next depth
            if depth < MAX_DEPTH and raw_html and len(all_pages) < MAX_TOTAL_PAGES:
                hop_links = self._extract_hop_links(raw_html, url, query, visited)
                for link_url in hop_links[:MAX_HOP_LINKS]:
                    queue.append(({"url": link_url, "title": "", "snippet": ""}, depth + 1))

        if not all_pages:
            return self._make_result(query, success=False,
                                     error="Could not extract content from any result")

        # Stats
        html_ct = sum(1 for p in all_pages if p["content_type"] == "html")
        pdf_ct = sum(1 for p in all_pages if p["content_type"] == "pdf")
        total_chars = sum(p["text_length"] for p in all_pages)
        d0 = sum(1 for p in all_pages if p.get("depth", 0) == 0)
        d1_plus = len(all_pages) - d0
        print(f"    [web_search] Scraped {len(all_pages)} pages "
              f"({d0} seed + {d1_plus} hops, {html_ct} HTML + {pdf_ct} PDF) "
              f"— {total_chars:,} chars total")

        data = {
            "total_urls_found": len(seed_urls),
            "total_pages_scraped": len(all_pages),
            "seed_pages": d0,
            "hop_pages": d1_plus,
            "html_pages": html_ct,
            "pdf_pages": pdf_ct,
            "total_chars": total_chars,
            "pages": all_pages,
        }

        raw_text = self._to_text(data, query)
        return self._make_result(query, success=True, data=data, raw_text=raw_text)

    # ── Search ─────────────────────────────────────────────────────────

    async def _search_ddg(self, query: ProfessorQuery) -> List[Dict]:
        """Multi-query DuckDuckGo search covering profile, publications, grants.

        Two-tier: first the duckduckgo-search library (fast when it works),
        then a raw HTML fallback that routes through our curl_cffi bypass when
        DDG rate-limits the library's IP.
        """
        base = f'"{query.name}" "Ohio State"'
        search_queries = [
            base,
            f'{base} publications research',
            f'{base} Google Scholar',
        ]
        if query.department:
            search_queries.append(f'{base} {query.department}')

        all_results: List[Dict] = []
        seen_urls: Set[str] = set()

        # ── Tier 1: duckduckgo-search library ─────────────────────────────
        try:
            from duckduckgo_search import DDGS
            ddgs_available = True
        except ImportError:
            ddgs_available = False
            print("    [web_search] ⚠️ duckduckgo-search not installed; using HTML fallback only")

        if ddgs_available:
            for sq in search_queries:
                try:
                    results = await asyncio.get_event_loop().run_in_executor(
                        None, lambda q=sq: list(DDGS().text(q, max_results=12))
                    )
                    for r in results:
                        url = r.get("href", "")
                        if url and url not in seen_urls and not self._should_skip(url):
                            seen_urls.add(url)
                            all_results.append({
                                "url": url,
                                "title": r.get("title", ""),
                                "snippet": r.get("body", ""),
                            })
                except Exception as e:
                    logger.debug("[web_search] DDGS lib failed on '%s': %s", sq[:60], e)

                await asyncio.sleep(2.0)

        # ── Tier 2: HTML fallback through bypass ladder ───────────────────
        # If the library returned 0 results, try raw HTML queries — curl_cffi
        # often succeeds where the library's HTTPX calls get rate-limited.
        if not all_results:
            print("    [web_search] DDGS lib returned 0 results — trying HTML fallback via bypass")
            for sq in search_queries[:2]:  # fewer queries on fallback to save time
                html = await self._fetch_html(
                    "https://html.duckduckgo.com/html/",
                    params={"q": sq},
                )
                if not html:
                    continue
                parsed = self._parse_ddg_html(html)
                for r in parsed:
                    url = r["url"]
                    if url and url not in seen_urls and not self._should_skip(url):
                        seen_urls.add(url)
                        all_results.append(r)
                await asyncio.sleep(2.0)

        all_results.sort(key=lambda r: self._url_priority(r["url"]))
        return all_results[:MAX_SEED_PAGES + 8]

    def _parse_ddg_html(self, html: str) -> List[Dict]:
        """Extract result URLs from DuckDuckGo's HTML search page."""
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            return []
        from urllib.parse import unquote, parse_qs, urlparse
        try:
            soup = BeautifulSoup(html, "html.parser")
        except Exception:
            return []
        out: List[Dict] = []
        for a in soup.select("a.result__a"):
            href = a.get("href") or ""
            # DDG wraps URLs in /l/?uddg=<encoded>
            if "/l/?" in href:
                qs = parse_qs(urlparse(href).query)
                real = qs.get("uddg") or qs.get("u")
                if real:
                    href = unquote(real[0])
            if href.startswith("http"):
                out.append({"url": href, "title": a.get_text(strip=True), "snippet": ""})
        return out

    # ── Fetch + extract (returns page dict AND raw html for link extraction) ──

    async def _fetch_and_extract(
        self, url: str, url_info: Dict, query: ProfessorQuery
    ) -> tuple:
        """Fetch a URL (with Cloudflare bypass escalation), extract text.

        Returns (page_dict_or_None, raw_html_or_None).
        """
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": (
                "text/html,application/xhtml+xml,application/xml;"
                "q=0.9,application/pdf;q=0.8,*/*;q=0.7"
            ),
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate",
            "DNT": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "cross-site",
        }

        # Route through bypass-aware fetcher — auto-escalates on 403 / cf-challenge
        result = await self._fetch_any(url, headers=headers, accept_pdf=True)
        if not result:
            return None, None

        content_type_raw = result.get("content_type", "")
        raw_html = None

        # ── PDF ──
        if content_type_raw == "pdf":
            text = self._extract_pdf(result["content"])
            if not text:
                return None, None
            content_type = "pdf"
        # ── HTML / text ──
        elif content_type_raw in ("html",) or "text/html" in content_type_raw or "text/plain" in content_type_raw:
            raw_html = result["text"]
            if self._is_blocked(raw_html):
                return None, None
            text = self._extract_html(raw_html)
            if not text or len(text) < 80:
                return None, None
            content_type = "html"
        else:
            return None, None

        # ── STRICT IDENTITY GATE ──
        # Prior version accepted a page if ANY of: full name in text, OR
        # (first+last in text), OR (last-name in text AND URL contains name).
        # That's how "Courtenay Moore" matched www.courtenay.ca (a city site)
        # and "L Lee" matched a politician's Wikipedia page.
        #
        # New rule: require full name anchored in the page text PLUS an
        # OSU/department affiliation signal somewhere in the page. Exceptions
        # made only for publisher/repository pages on priority academic
        # domains (e.g. pubmed, openalex, orcid) where affiliation may be
        # mentioned on the paper rather than the landing page.
        text_check = text[:30000]
        domain_lc = urlparse(url).netloc.lower()

        # Priority academic domains where full-name + strong contextual
        # content is enough (affiliation check relaxed because their
        # metadata pages structurally don't always repeat "Ohio State")
        ACADEMIC_DOMAINS = (
            "pubmed.ncbi.nlm.nih.gov", "ncbi.nlm.nih.gov", "arxiv.org",
            "orcid.org", "openalex.org", "semanticscholar.org",
            "biorxiv.org", "medrxiv.org", "crossref.org",
            "osu.edu",  # Any OSU subdomain auto-qualifies
        )
        is_academic_domain = any(d in domain_lc for d in ACADEMIC_DOMAINS)

        passes_gate = strict_identity_match(
            query, text_check,
            require_full_name=True,
            require_affiliation=(not is_academic_domain),
            department_hint=query.department,
            min_name_density=(1 if is_academic_domain else 2),
        )
        if not passes_gate:
            return None, None

        text = text[:MAX_TEXT_PER_PAGE] if content_type == "html" else text[:MAX_PDF_TEXT]
        domain = urlparse(url).netloc
        source_type = self._classify_source(url, text)

        page = {
            "url": url,
            "title": url_info.get("title", ""),
            "snippet": url_info.get("snippet", ""),
            "domain": domain,
            "source_type": source_type,
            "content_type": content_type,
            "text": text,
            "text_length": len(text),
        }
        return page, raw_html

    # ── Recursive link extraction ──────────────────────────────────────

    def _extract_hop_links(
        self, html: str, base_url: str, query: ProfessorQuery, visited: Set[str]
    ) -> List[str]:
        """Extract outbound links worth following from an HTML page.

        Prioritizes:
        - Links on academic/publication domains
        - Links with path segments like /publication, /article, /paper, /profile, /lab
        - Links containing the professor's name in the URL
        - PDF links (likely research papers or CV)

        Filters out:
        - Already visited URLs
        - Generic nav links (/login, /about, /contact, etc.)
        - Non-academic social media / commerce
        """
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            return self._extract_hop_links_regex(html, base_url, query, visited)

        soup = BeautifulSoup(html, "html.parser")
        candidates = []
        name_lower = normalize(query.name)
        last_lower = normalize(query.last_name)

        for a_tag in soup.find_all("a", href=True):
            href = a_tag.get("href", "").strip()
            if not href or href.startswith("#") or href.startswith("javascript:"):
                continue

            # Resolve relative URLs
            full_url = urljoin(base_url, href)

            # Normalize away fragments
            full_url = full_url.split("#")[0]

            canon = self._canonical(full_url)
            if canon in visited:
                continue
            if self._should_skip(full_url):
                continue

            # Skip generic nav paths
            path = urlparse(full_url).path.lower()
            if any(sp in path for sp in SKIP_PATHS):
                continue

            # Score this link
            score = self._hop_score(full_url, a_tag.get_text(strip=True), query)
            if score > 0:
                candidates.append((score, full_url))

        # Sort by score descending, return URLs
        candidates.sort(key=lambda x: -x[0])
        return [url for _, url in candidates]

    def _extract_hop_links_regex(
        self, html: str, base_url: str, query: ProfessorQuery, visited: Set[str]
    ) -> List[str]:
        """Fallback link extraction without BeautifulSoup."""
        candidates = []
        for m in re.finditer(r'href=["\']([^"\']+)["\']', html):
            href = m.group(1).strip()
            if not href or href.startswith("#") or href.startswith("javascript:"):
                continue
            full_url = urljoin(base_url, href).split("#")[0]
            if self._canonical(full_url) in visited or self._should_skip(full_url):
                continue
            path = urlparse(full_url).path.lower()
            if any(sp in path for sp in SKIP_PATHS):
                continue
            score = self._hop_score(full_url, "", query)
            if score > 0:
                candidates.append((score, full_url))
        candidates.sort(key=lambda x: -x[0])
        return [url for _, url in candidates]

    def _hop_score(self, url: str, anchor_text: str, query: ProfessorQuery) -> int:
        """Score a link for how likely it is to contain useful professor data.
        Returns 0 to skip, higher = more valuable.
        """
        score = 0
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            path = parsed.path.lower()
        except Exception:
            return 0

        # Domain signals
        for hd in HOP_WORTHY_DOMAINS:
            if hd in domain:
                score += 3
                break

        # Path signals — publication, profile, lab, paper, CV, etc.
        for hp in HOP_WORTHY_PATHS:
            if hp in path:
                score += 4
                break

        # PDF link — likely a paper or CV
        if path.endswith(".pdf"):
            score += 5

        # Professor's name in the URL
        url_lower = url.lower()
        name_parts = query.name.lower().split()
        if any(part in url_lower for part in name_parts if len(part) > 2):
            score += 3

        # Professor's name in anchor text
        if anchor_text:
            anchor_lower = anchor_text.lower()
            last_lower = query.last_name.lower()
            if last_lower in anchor_lower:
                score += 2

        # Anchor text with academic keywords
        if anchor_text:
            al = anchor_text.lower()
            for kw in ["publication", "paper", "article", "research",
                        "lab", "project", "cv", "vita", "resume",
                        "grant", "profile", "scholar", "full text",
                        "pdf", "doi", "abstract", "proceedings"]:
                if kw in al:
                    score += 2
                    break

        # Same domain as base page — likely a subpage of their site
        # (but don't score if it's a giant portal like osu.edu home)
        if len(path) > 10:  # has a real path, not just "/"
            score += 1

        return score

    # ── HTML content extraction (main content only) ────────────────────

    def _extract_html(self, html: str) -> str:
        """Extract ONLY main body content, stripping all chrome."""
        try:
            from bs4 import BeautifulSoup, Comment
        except ImportError:
            return self._extract_html_fallback(html)

        soup = BeautifulSoup(html, "html.parser")

        for tag_name in STRIP_TAGS:
            for el in soup.find_all(tag_name):
                el.decompose()
        for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
            comment.extract()
        for role in STRIP_ROLES:
            for el in soup.find_all(attrs={"role": role}):
                el.decompose()
        class_re = re.compile("|".join(STRIP_CLASSES), re.I)
        for el in soup.find_all(class_=class_re):
            el.decompose()
        id_re = re.compile("|".join(STRIP_IDS), re.I)
        for el in soup.find_all(id=id_re):
            el.decompose()
        for el in soup.find_all(style=re.compile(r"display\s*:\s*none", re.I)):
            el.decompose()
        for el in soup.find_all(attrs={"hidden": True}):
            el.decompose()
        for el in soup.find_all(attrs={"aria-hidden": "true"}):
            el.decompose()

        main = self._find_main_content(soup)
        return self._get_clean_text(main)

    def _find_main_content(self, soup) -> Any:
        """Locate the primary content element."""
        for tag in ["main", "article"]:
            el = soup.find(tag)
            if el and len(el.get_text(strip=True)) > 200:
                return el
        for id_pattern in [
            "content", "main-content", "main_content", "page-content",
            "article-content", "post-content", "entry-content",
            "body-content", "primary",
        ]:
            el = soup.find(id=re.compile(rf"^{id_pattern}$", re.I))
            if el and len(el.get_text(strip=True)) > 200:
                return el
        for cls_pattern in [
            r"^content$", r"main.content", r"page.content",
            r"article.content", r"post.content", r"entry.content",
            r"^primary$", r"^main$",
        ]:
            el = soup.find(class_=re.compile(cls_pattern, re.I))
            if el and len(el.get_text(strip=True)) > 200:
                return el
        best = None
        best_len = 0
        for div in soup.find_all(["div", "section"]):
            txt = div.get_text(strip=True)
            if 200 < len(txt) < len(soup.get_text(strip=True)) * 0.95:
                if len(txt) > best_len:
                    best_len = len(txt)
                    best = div
        return best or soup.body or soup

    @staticmethod
    def _get_clean_text(element) -> str:
        """Extract readable text from leaf-level content elements only."""
        if element is None:
            return ""

        LEAF_TAGS = {"p", "li", "h1", "h2", "h3", "h4", "h5", "h6",
                     "blockquote", "dd", "dt", "figcaption", "caption", "td", "th"}
        lines = []
        for child in element.descendants:
            if isinstance(child, str):
                continue
            if child.name not in LEAF_TAGS:
                continue
            has_leaf_child = any(
                d.name in LEAF_TAGS
                for d in child.descendants
                if not isinstance(d, str)
            )
            if has_leaf_child:
                continue
            text = child.get_text(separator=" ", strip=True)
            if text and len(text) > 3:
                lines.append(text)

        if not lines:
            text = element.get_text(separator="\n", strip=True)
            lines = [l.strip() for l in text.split("\n")
                     if l.strip() and len(l.strip()) > 3]

        deduped = []
        seen = set()
        for line in lines:
            if line not in seen:
                seen.add(line)
                deduped.append(line)
        return "\n".join(deduped)

    @staticmethod
    def _extract_html_fallback(html: str) -> str:
        text = re.sub(r"<script[^>]*>.*?</script>", "", html,
                       flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text,
                       flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<(nav|footer|header|aside)[^>]*>.*?</\1>", "", text,
                       flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # ── PDF extraction ─────────────────────────────────────────────────

    @staticmethod
    def _extract_pdf(content: bytes) -> str:
        """Extract text from PDF bytes using PyMuPDF → pdfplumber → pypdf."""
        try:
            import fitz
            doc = fitz.open(stream=content, filetype="pdf")
            pages_to_read = min(len(doc), MAX_PDF_PAGES)
            parts = []
            for i in range(pages_to_read):
                page_text = doc[i].get_text("text")
                if page_text:
                    parts.append(page_text.strip())
            doc.close()
            text = "\n\n".join(parts)
            if text and len(text) > 50:
                return text[:MAX_PDF_TEXT]
        except Exception as e:
            logger.debug("[web_search] PyMuPDF failed: %s", e)
        try:
            import pdfplumber
            pdf = pdfplumber.open(io.BytesIO(content))
            parts = []
            for page in pdf.pages[:MAX_PDF_PAGES]:
                page_text = page.extract_text()
                if page_text:
                    parts.append(page_text.strip())
            pdf.close()
            text = "\n\n".join(parts)
            if text and len(text) > 50:
                return text[:MAX_PDF_TEXT]
        except Exception as e:
            logger.debug("[web_search] pdfplumber failed: %s", e)
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(content))
            parts = []
            for page in reader.pages[:MAX_PDF_PAGES]:
                page_text = page.extract_text()
                if page_text:
                    parts.append(page_text.strip())
            text = "\n\n".join(parts)
            if text and len(text) > 50:
                return text[:MAX_PDF_TEXT]
        except Exception as e:
            logger.debug("[web_search] pypdf failed: %s", e)
        return ""

    # ── Cloudflare / bot-protection detection ──────────────────────────

    @staticmethod
    def _is_blocked(html: str) -> bool:
        if not html:
            return True
        if len(html) < 1500:
            lower = html.lower()
            for sig in BLOCK_SIGNATURES:
                if sig in lower:
                    return True
        lower = html[:5000].lower()
        cf_markers = ["cf-browser-verification", "cf_chl_opt", "turnstile",
                       "__cf_chl_tk", "challenge-platform"]
        return any(m in lower for m in cf_markers)

    # ── URL utilities ──────────────────────────────────────────────────

    @staticmethod
    def _canonical(url: str) -> str:
        """Normalize URL for dedup: strip fragment, trailing slash, lowercase domain."""
        try:
            parsed = urlparse(url.split("#")[0])
            path = parsed.path.rstrip("/") or "/"
            return f"{parsed.scheme}://{parsed.netloc.lower()}{path}"
        except Exception:
            return url

    @staticmethod
    def _should_skip(url: str) -> bool:
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            path = parsed.path.lower()
        except Exception:
            return True
        for skip in SKIP_DOMAINS:
            if skip in domain:
                return True
        if any(path.endswith(ext) for ext in [".doc", ".docx", ".ppt", ".pptx",
                                               ".zip", ".gz", ".tar", ".mp4",
                                               ".mp3", ".wav", ".jpg", ".png",
                                               ".gif", ".svg", ".exe"]):
            return True
        return False

    @staticmethod
    def _url_priority(url: str) -> int:
        try:
            domain = urlparse(url).netloc.lower()
        except Exception:
            return 99
        for i, pd in enumerate(PRIORITY_DOMAINS):
            if pd in domain:
                return i
        return 50

    @staticmethod
    def _classify_source(url: str, text: str) -> str:
        domain = urlparse(url).netloc.lower()
        path = urlparse(url).path.lower()

        if "scholar.google" in domain:
            return "google_scholar_profile"
        if "researchgate.net" in domain:
            return "researchgate_profile"
        if "orcid.org" in domain:
            return "orcid_profile"
        if "linkedin.com" in domain:
            return "linkedin_profile"
        if "semanticscholar.org" in domain:
            return "semantic_scholar_profile"
        if "pubmed" in domain or "ncbi.nlm.nih.gov" in domain:
            return "pubmed"
        if "arxiv.org" in domain:
            return "arxiv_paper"
        if "nsf.gov" in domain:
            return "nsf_grant"
        if "nih.gov" in domain:
            return "nih_grant"
        if "experts.osu.edu" in domain:
            return "osu_experts_profile"
        if "osu.edu" in domain:
            if "news" in domain or "news" in path:
                return "osu_news"
            if any(k in path for k in ["/people/", "/faculty/", "/directory/",
                                        "/find-a-doctor/", "/staff/"]):
                return "osu_faculty_page"
            return "osu_page"
        if "ratemyprofessors" in domain:
            return "ratemyprofessors"
        if path.endswith(".pdf"):
            return "research_paper_pdf"
        pub_domains = ["springer.com", "wiley.com", "nature.com", "sciencedirect.com",
                        "ieee.org", "acm.org", "plos.org", "mdpi.com", "frontiersin.org",
                        "biorxiv.org", "medrxiv.org", "tandfonline.com", "sagepub.com",
                        "oup.com", "cell.com", "acs.org"]
        for pd in pub_domains:
            if pd in domain:
                return "journal_publication"
        text_lower = text[:3000].lower()
        if any(w in text_lower for w in ["abstract", "doi:", "10.", "journal", "et al"]):
            return "academic_publication"
        if any(w in text_lower for w in ["grant", "award", "nsf", "nih", "funding"]):
            return "grant_info"
        if any(w in text_lower for w in ["course", "syllabus", "credit hour"]):
            return "teaching_info"
        if any(w in text_lower for w in ["curriculum vitae", "c.v.", "resume"]):
            return "cv_resume"
        return "web_page"

    # ── Output formatting ──────────────────────────────────────────────

    def _to_text(self, data: Dict, query: ProfessorQuery) -> str:
        lines = []
        lines.append(f"=== Web Search Results: {query.name} ===")
        lines.append(f"URLs found: {data['total_urls_found']}")
        lines.append(f"Pages scraped: {data['total_pages_scraped']} "
                      f"({data['seed_pages']} seed + {data['hop_pages']} hops, "
                      f"{data['html_pages']} HTML + {data['pdf_pages']} PDF)")
        lines.append(f"Total content: {data['total_chars']:,} chars")
        lines.append("")

        for i, page in enumerate(data["pages"], 1):
            lines.append(f"\n{'─'*50}")
            ct_label = "PDF" if page["content_type"] == "pdf" else "HTML"
            depth_label = f"depth={page.get('depth', 0)}"
            lines.append(f"Source {i}: [{page['source_type']}] [{ct_label}] "
                          f"[{depth_label}] {page['domain']}")
            lines.append(f"URL: {page['url']}")
            if page.get("title"):
                lines.append(f"Title: {page['title']}")
            lines.append(f"Content ({page['text_length']:,} chars):")
            lines.append(page["text"])

        return "\n".join(lines)
