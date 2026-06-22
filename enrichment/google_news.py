"""
Google News collector via RSS feed scraping.
No API key required — uses the public Google News RSS feed.

Returns: news articles, media mentions, expert quotes, interviews.
"""

import logging
import re
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus

from .base_collector import BaseCollector, CollectorResult, ProfessorQuery
from .validation import filter_articles_by_name

logger = logging.getLogger(__name__)


class GoogleNewsCollector(BaseCollector):
    """Collect news articles mentioning a professor from Google News RSS."""

    def __init__(self, **kwargs):
        # Google News RSS: avoid triggering captchas
        kwargs.setdefault("rate_limit_delay", 4.0)
        super().__init__(**kwargs)

    @property
    def source_name(self) -> str:
        return "google_news"

    async def collect(self, query: ProfessorQuery) -> CollectorResult:
        articles = []

        # Only search with professor name + Ohio State — avoids noise
        search_queries = [
            f'"{query.name}" "Ohio State"',
        ]

        seen_titles = set()
        for sq in search_queries:
            results = await self._fetch_google_news_rss(sq)
            for article in results:
                title_key = article["title"].lower().strip()
                if title_key not in seen_titles:
                    seen_titles.add(title_key)
                    articles.append(article)

        # Post-filter: only keep articles that actually mention the professor's name
        articles = filter_articles_by_name(articles, query, ["title", "description"])

        if not articles:
            return self._make_result(query, success=False, error="No Google News articles found")

        data = {
            "total_articles": len(articles),
            "articles": articles[:40],
        }

        raw_text = self._to_text(data, query)
        return self._make_result(query, success=True, data=data, raw_text=raw_text)

    async def _fetch_google_news_rss(self, search_query: str) -> List[Dict]:
        """Fetch articles from Google News RSS feed."""
        encoded_query = quote_plus(search_query)
        rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"

        try:
            client = await self.get_client()
            await self._rate_limit()
            resp = await client.get(rss_url)
            if resp.status_code != 200:
                return []
            return self._parse_rss(resp.text)
        except Exception as e:
            logger.warning("[google_news] RSS fetch failed: %s", e)
            return []

    @staticmethod
    def _parse_rss(xml_text: str) -> List[Dict]:
        """Parse Google News RSS XML into article dicts."""
        articles = []

        # Simple XML parsing without importing xml.etree (avoid issues with malformed XML)
        items = re.findall(r"<item>(.*?)</item>", xml_text, re.DOTALL)

        for item in items:
            title_match = re.search(r"<title>(.*?)</title>", item, re.DOTALL)
            link_match = re.search(r"<link>(.*?)</link>", item, re.DOTALL)
            pub_date_match = re.search(r"<pubDate>(.*?)</pubDate>", item, re.DOTALL)
            desc_match = re.search(r"<description>(.*?)</description>", item, re.DOTALL)
            source_match = re.search(r"<source[^>]*>(.*?)</source>", item, re.DOTALL)

            title = title_match.group(1).strip() if title_match else ""
            link = link_match.group(1).strip() if link_match else ""
            pub_date = pub_date_match.group(1).strip() if pub_date_match else ""
            description = desc_match.group(1).strip() if desc_match else ""
            source = source_match.group(1).strip() if source_match else ""

            # Clean HTML from description
            description = re.sub(r"<[^>]+>", "", description).strip()
            # Decode HTML entities
            description = (
                description.replace("&amp;", "&")
                .replace("&lt;", "<")
                .replace("&gt;", ">")
                .replace("&quot;", '"')
                .replace("&#39;", "'")
            )

            if title:
                articles.append({
                    "title": title.replace("&amp;", "&").replace("&#39;", "'"),
                    "url": link,
                    "published_date": pub_date,
                    "description": description[:500],
                    "source": source,
                })

        return articles

    def _to_text(self, data: Dict, query: ProfessorQuery) -> str:
        lines = []
        lines.append(f"=== Google News: {query.name} ===")
        lines.append(f"Total articles found: {data['total_articles']}")
        lines.append("")

        for i, article in enumerate(data["articles"], 1):
            lines.append(f"\n{i}. {article['title']}")
            if article.get("source"):
                lines.append(f"   Source: {article['source']}")
            if article.get("published_date"):
                lines.append(f"   Date: {article['published_date']}")
            lines.append(f"   URL: {article['url']}")
            if article.get("description"):
                lines.append(f"   Summary: {article['description'][:400]}")

        return "\n".join(lines)
