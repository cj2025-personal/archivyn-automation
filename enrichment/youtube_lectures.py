"""
YouTube collector.
Uses the YouTube Data API v3 to find lectures, talks, and interviews
featuring a professor.

Requires: YOUTUBE_API_KEY environment variable.
Get one free at: https://console.cloud.google.com/apis/library/youtube.googleapis.com
"""

import logging
import os
from typing import Any, Dict, List, Optional

from .base_collector import BaseCollector, CollectorResult, ProfessorQuery
from .validation import filter_articles_by_name

logger = logging.getLogger(__name__)

SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"


class YouTubeLecturesCollector(BaseCollector):
    """Collect YouTube lectures and talks by/about a professor."""

    def __init__(self, **kwargs):
        # YouTube Data API: 10,000 units/day quota — be conservative
        kwargs.setdefault("rate_limit_delay", 2.0)
        super().__init__(**kwargs)
        self.api_key = os.getenv("YOUTUBE_API_KEY", "")

    @property
    def source_name(self) -> str:
        return "youtube_lectures"

    async def collect(self, query: ProfessorQuery) -> CollectorResult:
        if not self.api_key:
            return self._make_result(
                query, success=False,
                error="YOUTUBE_API_KEY not set. Get one at console.cloud.google.com"
            )

        videos = await self._search_videos(query)
        if not videos:
            return self._make_result(query, success=False, error="No YouTube videos found")

        # Get detailed info for found videos
        video_ids = [v["video_id"] for v in videos if v.get("video_id")]
        details = await self._get_video_details(video_ids) if video_ids else {}

        # Merge details
        for v in videos:
            vid = v.get("video_id", "")
            if vid in details:
                v.update(details[vid])

        data = {
            "total_videos": len(videos),
            "videos": videos,
        }

        raw_text = self._to_text(data, query)
        return self._make_result(query, success=True, data=data, raw_text=raw_text)

    async def _search_videos(self, query: ProfessorQuery) -> List[Dict]:
        """Search YouTube for videos featuring this professor."""
        all_videos = []

        # Multiple search queries for broader coverage
        search_queries = [
            f'"{query.name}" Ohio State University',
            f'"{query.name}" lecture',
            f'"{query.name}" {query.department}' if query.department else None,
        ]

        seen_ids = set()
        for sq in search_queries:
            if not sq:
                continue

            params = {
                "part": "snippet",
                "q": sq,
                "type": "video",
                "maxResults": 15,
                "order": "relevance",
                "key": self.api_key,
                "videoDuration": "medium",  # 4-20 minutes — filters out shorts
            }

            try:
                resp = await self._get_json(SEARCH_URL, params=params)
                if not resp or not resp.get("items"):
                    continue

                for item in resp["items"]:
                    vid = item.get("id", {}).get("videoId", "")
                    if not vid or vid in seen_ids:
                        continue
                    seen_ids.add(vid)

                    snippet = item.get("snippet", {})
                    all_videos.append({
                        "video_id": vid,
                        "title": snippet.get("title", ""),
                        "description": snippet.get("description", ""),
                        "channel": snippet.get("channelTitle", ""),
                        "published_at": snippet.get("publishedAt", ""),
                        "thumbnail": snippet.get("thumbnails", {}).get("high", {}).get("url", ""),
                        "url": f"https://www.youtube.com/watch?v={vid}",
                    })

            except Exception as e:
                logger.warning("[youtube] Search failed for '%s': %s", sq, e)

        # Post-filter: only keep videos that mention the professor's name
        all_videos = filter_articles_by_name(all_videos, query, ["title", "description"])
        return all_videos[:25]

    async def _get_video_details(self, video_ids: List[str]) -> Dict[str, Dict]:
        """Get detailed stats for videos."""
        if not video_ids:
            return {}

        params = {
            "part": "statistics,contentDetails",
            "id": ",".join(video_ids[:50]),
            "key": self.api_key,
        }

        try:
            resp = await self._get_json(VIDEOS_URL, params=params)
            if not resp or not resp.get("items"):
                return {}
        except Exception:
            return {}

        details = {}
        for item in resp["items"]:
            vid = item.get("id", "")
            stats = item.get("statistics", {})
            content = item.get("contentDetails", {})
            details[vid] = {
                "view_count": int(stats.get("viewCount", 0)),
                "like_count": int(stats.get("likeCount", 0)),
                "comment_count": int(stats.get("commentCount", 0)),
                "duration": content.get("duration", ""),
            }

        return details

    def _to_text(self, data: Dict, query: ProfessorQuery) -> str:
        lines = []
        lines.append(f"=== YouTube Videos: {query.name} ===")
        lines.append(f"Total videos found: {data['total_videos']}")
        lines.append("")

        for i, v in enumerate(data["videos"], 1):
            lines.append(f"\n{i}. {v['title']}")
            lines.append(f"   Channel: {v.get('channel', 'N/A')}")
            lines.append(f"   URL: {v.get('url', '')}")
            if v.get("published_at"):
                lines.append(f"   Published: {v['published_at'][:10]}")
            if v.get("view_count"):
                lines.append(f"   Views: {v['view_count']:,}")
            if v.get("duration"):
                lines.append(f"   Duration: {v['duration']}")
            if v.get("description"):
                lines.append(f"   Description: {v['description'][:300]}")

        return "\n".join(lines)
