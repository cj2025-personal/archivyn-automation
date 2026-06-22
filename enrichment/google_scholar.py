"""
Google Scholar collector via the `scholarly` library.
Returns: h-index, i10-index, citation counts, top cited papers, co-authors.

NOTE: Google Scholar has aggressive anti-bot measures.
Uses free proxy rotation to avoid captcha blocks.
Install: pip install scholarly free-proxy
"""

import logging
import random
from typing import Any, Dict, List, Optional

from .base_collector import BaseCollector, CollectorResult, ProfessorQuery
from .validation import names_match_from_query

logger = logging.getLogger(__name__)

_PROXY_INITIALIZED = False


def _init_scholarly_proxy(scholarly_module):
    """Set up free proxy rotation once per process."""
    global _PROXY_INITIALIZED
    if _PROXY_INITIALIZED:
        return
    try:
        from scholarly import ProxyGenerator
        pg = ProxyGenerator()
        pg.FreeProxies()
        scholarly_module.use_proxy(pg)
        logger.info("[google_scholar] Free proxy rotation enabled")
    except Exception as e:
        logger.warning("[google_scholar] Could not set up free proxies: %s — running direct", e)
    _PROXY_INITIALIZED = True


class GoogleScholarCollector(BaseCollector):
    """Collect citation metrics and publication data from Google Scholar."""

    def __init__(self, **kwargs):
        # Google Scholar is aggressive with blocks — use generous delays.
        # Also enable the Cloudflare-bypass ladder so that if scholarly's
        # free-proxy pool fails, _fetch_html can still try curl_cffi / playwright
        # against scholar.google.com directly.
        kwargs.setdefault("rate_limit_delay", 5.0)
        kwargs.setdefault("timeout", 60.0)
        kwargs.setdefault("bypass_tier", "curl_cffi")
        super().__init__(**kwargs)
        self._max_attempts = 3

    @property
    def source_name(self) -> str:
        return "google_scholar"

    async def collect(self, query: ProfessorQuery) -> CollectorResult:
        try:
            from scholarly import scholarly
        except ImportError:
            return self._make_result(
                query, success=False,
                error="scholarly library not installed. Run: pip install scholarly"
            )

        import asyncio

        # Initialize proxy rotation on first use
        _init_scholarly_proxy(scholarly)

        last_error = "Google Scholar timed out (likely captcha-blocked)"
        for attempt in range(1, self._max_attempts + 1):
            # Add jittered delay between attempts (5s base + random 0-5s)
            if attempt > 1:
                wait = 5.0 + random.uniform(0, 5.0)
                print(f"    [google_scholar] Retry {attempt}/{self._max_attempts} after {wait:.1f}s...")
                await asyncio.sleep(wait)

            try:
                author_data = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None, self._search_author, query, scholarly
                    ),
                    timeout=45.0,
                )
            except asyncio.TimeoutError:
                last_error = f"Google Scholar timed out on attempt {attempt}/{self._max_attempts}"
                continue
            except Exception as e:
                last_error = f"Google Scholar error: {e}"
                continue

            # Success or author-not-found (no point retrying)
            if author_data is None:
                return self._make_result(query, success=False, error="Author not found on Google Scholar")

            raw_text = self._to_text(author_data, query)
            return self._make_result(query, success=True, data=author_data, raw_text=raw_text)

        return self._make_result(query, success=False, error=last_error)

    @staticmethod
    def _search_author(query: ProfessorQuery, scholarly_module) -> Optional[Dict]:
        """Search Google Scholar for the professor (synchronous).

        Strict matching: require first+last name AND Ohio State affiliation.
        """
        search_query = f"{query.name} Ohio State University"

        try:
            results = scholarly_module.search_author(search_query)
        except Exception as e:
            logger.warning("[google_scholar] Search failed for %s: %s", query.name, e)
            return None

        # Two-tier match: prefer name + OSU affiliation, accept name-only if unambiguous
        # Only check first 5 results — the generator can block on captchas
        best = None
        name_matched_no_affil = []
        checked = 0
        for author in results:
            checked += 1
            if checked > 5:
                break

            affiliation = (author.get("affiliation") or "").lower()
            found_name = author.get("name") or ""

            if not names_match_from_query(query, found_name):
                continue

            # Tier 1: name + Ohio State affiliation (best case)
            if "ohio state" in affiliation:
                best = author
                break

            # Track name-only matches for Tier 2 fallback
            name_matched_no_affil.append(author)

        # Tier 2: if no affiliation match, accept if exactly 1 name-matched candidate
        if not best and len(name_matched_no_affil) == 1:
            best = name_matched_no_affil[0]
            logger.info(
                "[google_scholar] Matched %s via name-only fallback (1 candidate)",
                query.name,
            )

        if not best:
            return None

        # Fill in detailed info
        try:
            author_detail = scholarly_module.fill(best, sections=["basics", "indices", "counts", "publications"])
        except Exception as e:
            logger.warning("[google_scholar] fill() failed for %s: %s", query.name, e)
            author_detail = best

        # Extract publications (top 20 by citations)
        pubs = author_detail.get("publications", [])
        # Sort by citations
        pubs_sorted = sorted(pubs, key=lambda p: p.get("num_citations", 0), reverse=True)

        publications = []
        for pub in pubs_sorted[:30]:
            bib = pub.get("bib", {})
            publications.append({
                "title": bib.get("title", ""),
                "year": bib.get("pub_year", ""),
                "venue": bib.get("venue", "") or bib.get("journal", "") or bib.get("conference", ""),
                "citations": pub.get("num_citations", 0),
                "abstract": bib.get("abstract", ""),
            })

        return {
            "scholar_id": author_detail.get("scholar_id", ""),
            "name": author_detail.get("name", query.name),
            "affiliation": author_detail.get("affiliation", ""),
            "interests": author_detail.get("interests", []),
            "h_index": author_detail.get("hindex", None),
            "h_index_5yr": author_detail.get("hindex5y", None),
            "i10_index": author_detail.get("i10index", None),
            "i10_index_5yr": author_detail.get("i10index5y", None),
            "total_citations": author_detail.get("citedby", 0),
            "total_citations_5yr": author_detail.get("citedby5y", 0),
            "citations_per_year": author_detail.get("cites_per_year", {}),
            "coauthors": [
                {"name": ca.get("name", ""), "affiliation": ca.get("affiliation", "")}
                for ca in (author_detail.get("coauthors") or [])[:20]
            ],
            "publications": publications,
            "url_picture": author_detail.get("url_picture", ""),
            "google_scholar_url": f"https://scholar.google.com/citations?user={author_detail.get('scholar_id', '')}",
        }

    def _to_text(self, data: Dict, query: ProfessorQuery) -> str:
        lines = []
        lines.append(f"=== Google Scholar Profile: {data['name']} ===")
        lines.append(f"Affiliation: {data.get('affiliation', 'N/A')}")
        if data.get("interests"):
            lines.append(f"Research interests: {', '.join(data['interests'])}")
        lines.append(f"h-index: {data.get('h_index', 'N/A')}")
        lines.append(f"i10-index: {data.get('i10_index', 'N/A')}")
        lines.append(f"Total citations: {data.get('total_citations', 'N/A')}")
        if data.get("h_index_5yr"):
            lines.append(f"h-index (last 5 years): {data['h_index_5yr']}")
        if data.get("total_citations_5yr"):
            lines.append(f"Citations (last 5 years): {data['total_citations_5yr']}")

        if data.get("coauthors"):
            lines.append("\n--- Co-authors ---")
            for ca in data["coauthors"]:
                lines.append(f"  - {ca['name']} ({ca.get('affiliation', '')})")

        lines.append("\n--- Top Publications (by citation count) ---")
        for i, pub in enumerate(data["publications"][:30], 1):
            lines.append(f"\n{i}. {pub['title']} ({pub.get('year', 'N/A')})")
            lines.append(f"   Citations: {pub['citations']}")
            if pub.get("venue"):
                lines.append(f"   Venue: {pub['venue']}")
            if pub.get("abstract"):
                lines.append(f"   Abstract: {pub['abstract'][:400]}")

        return "\n".join(lines)
