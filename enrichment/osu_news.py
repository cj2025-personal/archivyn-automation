"""
OSU News collector.
Scrapes news.osu.edu for press releases and articles mentioning a professor.
Also checks department-specific news pages.
"""

import logging
import re
from typing import Any, Dict, List, Optional

from .base_collector import BaseCollector, CollectorResult, ProfessorQuery
from .validation import filter_articles_by_name

logger = logging.getLogger(__name__)

OSU_NEWS_SEARCH = "https://news.osu.edu/"


class OSUNewsCollector(BaseCollector):
    """Collect news articles and press releases from OSU News."""

    def __init__(self, **kwargs):
        # OSU News WordPress API: be respectful to university servers
        kwargs.setdefault("rate_limit_delay", 2.0)
        super().__init__(**kwargs)

    @property
    def source_name(self) -> str:
        return "osu_news"

    async def collect(self, query: ProfessorQuery) -> CollectorResult:
        articles = await self._search_news(query)
        if not articles:
            return self._make_result(query, success=False, error="No OSU news articles found")

        data = {
            "total_articles": len(articles),
            "articles": articles,
        }

        raw_text = self._to_text(data, query)
        return self._make_result(query, success=True, data=data, raw_text=raw_text)

    async def _search_news(self, query: ProfessorQuery) -> List[Dict]:
        """Search OSU News for articles mentioning this professor."""
        articles = []

        # OSU News uses WordPress search
        search_url = f"{OSU_NEWS_SEARCH}"
        params = {
            "s": query.name,
        }

        try:
            client = await self.get_client()
            await self._rate_limit()
            resp = await client.get(search_url, params=params)
            if resp.status_code != 200:
                return []

            html = resp.text
            articles = self._parse_search_results(html, query)

        except Exception as e:
            logger.warning("[osu_news] Search failed: %s", e)

        # Also try the WordPress REST API
        wp_articles = await self._search_wp_api(query)
        articles.extend(wp_articles)

        # Post-filter: only keep articles that mention the professor's full name
        articles = filter_articles_by_name(articles, query, ["title", "excerpt"])

        # Deduplicate by URL
        seen = set()
        unique = []
        for a in articles:
            if a["url"] not in seen:
                seen.add(a["url"])
                unique.append(a)

        return unique[:30]

    async def _search_wp_api(self, query: ProfessorQuery) -> List[Dict]:
        """Try the WordPress REST API for OSU News."""
        url = "https://news.osu.edu/wp-json/wp/v2/posts"
        params = {
            "search": query.name,
            "per_page": 20,
            "_fields": "id,title,excerpt,link,date,modified",
        }

        try:
            resp = await self._get_json(url, params=params)
            if not resp or not isinstance(resp, list):
                return []
        except Exception:
            return []

        articles = []
        for post in resp:
            title = self._strip_html(post.get("title", {}).get("rendered", ""))
            excerpt = self._strip_html(post.get("excerpt", {}).get("rendered", ""))

            articles.append({
                "title": title,
                "excerpt": excerpt,
                "url": post.get("link", ""),
                "date": (post.get("date") or "")[:10],
                "source": "news.osu.edu",
            })

        return articles

    def _parse_search_results(self, html: str, query: ProfessorQuery) -> List[Dict]:
        """Parse search results from OSU News HTML."""
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            logger.warning("[osu_news] beautifulsoup4 not installed")
            return []

        soup = BeautifulSoup(html, "html.parser")
        articles = []

        # Find article entries
        for article in soup.find_all("article"):
            title_el = article.find(["h2", "h3"])
            link_el = article.find("a", href=True)
            excerpt_el = article.find(class_=re.compile(r"excerpt|summary|entry-content"))
            date_el = article.find("time") or article.find(class_=re.compile(r"date|time"))

            if not title_el or not link_el:
                continue

            title = title_el.get_text(strip=True)
            url = link_el.get("href", "")
            excerpt = excerpt_el.get_text(strip=True) if excerpt_el else ""
            date = date_el.get("datetime", date_el.get_text(strip=True)) if date_el else ""

            articles.append({
                "title": title,
                "excerpt": excerpt[:500],
                "url": url,
                "date": date[:10] if date else "",
                "source": "news.osu.edu",
            })

        return articles

    @staticmethod
    def _strip_html(text: str) -> str:
        """Remove HTML tags from text."""
        return re.sub(r"<[^>]+>", "", text).strip()

    def _to_text(self, data: Dict, query: ProfessorQuery) -> str:
        lines = []
        lines.append(f"=== OSU News & Media: {query.name} ===")
        lines.append(f"Total articles found: {data['total_articles']}")
        lines.append("")

        for i, article in enumerate(data["articles"], 1):
            lines.append(f"\n{i}. {article['title']}")
            if article.get("date"):
                lines.append(f"   Date: {article['date']}")
            lines.append(f"   Source: {article.get('source', 'news.osu.edu')}")
            lines.append(f"   URL: {article['url']}")
            if article.get("excerpt"):
                lines.append(f"   Excerpt: {article['excerpt'][:400]}")

        return "\n".join(lines)
