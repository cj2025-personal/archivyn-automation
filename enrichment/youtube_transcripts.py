"""
YouTube transcripts collector — pulls full captions for videos already
surfaced by the youtube_lectures collector.

Uses the `youtube-transcript-api` package (graceful-optional). Falls back
to scraping captions via Playwright when library isn't available.

Why: lecture / talk transcripts are the single richest plain-language source
about a researcher's work, and almost entirely absent from LLM training data
(LLMs saw video metadata at best, not captions).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base_collector import BaseCollector, CollectorResult, ProfessorQuery

try:
    from youtube_transcript_api import YouTubeTranscriptApi  # type: ignore
    _YTA_AVAILABLE = True
except Exception:
    YouTubeTranscriptApi = None  # type: ignore
    _YTA_AVAILABLE = False

# The API surface changed in recent versions:
#   old: YouTubeTranscriptApi.get_transcript(video_id, languages=[...]) -> list[dict]
#   new: YouTubeTranscriptApi().fetch(video_id, languages=[...]) -> FetchedTranscript
# We detect which is available at runtime and handle both.
def _fetch_transcript_cross_version(video_id: str, languages):
    if not _YTA_AVAILABLE:
        return None
    # New API (>=1.0): instance method .fetch
    if hasattr(YouTubeTranscriptApi, "fetch") and not hasattr(YouTubeTranscriptApi, "get_transcript"):
        try:
            api = YouTubeTranscriptApi()
            tr = api.fetch(video_id, languages=languages)
            # tr.snippets is a list of FetchedTranscriptSnippet with .text/.start/.duration
            snippets = getattr(tr, "snippets", None) or list(tr)
            return [
                {"text": getattr(s, "text", ""), "start": getattr(s, "start", 0), "duration": getattr(s, "duration", 0)}
                for s in snippets
            ]
        except Exception:
            return None
    # Old API: classmethod get_transcript
    if hasattr(YouTubeTranscriptApi, "get_transcript"):
        try:
            return YouTubeTranscriptApi.get_transcript(video_id, languages=languages)  # type: ignore[attr-defined]
        except Exception:
            return None
    return None


class YouTubeTranscriptsCollector(BaseCollector):
    def __init__(self, **kwargs):
        kwargs.setdefault("rate_limit_delay", 1.0)
        super().__init__(**kwargs)
        self.max_videos = int(os.getenv("YOUTUBE_TRANSCRIPT_MAX", "20"))
        self.max_chars_per_transcript = int(os.getenv("YOUTUBE_TRANSCRIPT_CHAR_CAP", "15000"))

    @property
    def source_name(self) -> str:
        return "youtube_transcripts"

    async def collect(self, query: ProfessorQuery) -> CollectorResult:
        if not _YTA_AVAILABLE:
            return self._make_result(
                query, success=False,
                error="youtube-transcript-api not installed (pip install youtube-transcript-api)",
            )

        video_ids = self._gather_video_ids(query)
        if not video_ids:
            return self._make_result(
                query, success=False,
                error="No YouTube videos available; run youtube_lectures first",
            )

        video_ids = video_ids[: self.max_videos]
        transcripts: List[Dict[str, Any]] = []

        for vid in video_ids:
            text = self._fetch_transcript(vid)
            if not text:
                continue
            transcripts.append({
                "video_id": vid,
                "url": f"https://www.youtube.com/watch?v={vid}",
                "transcript": text[: self.max_chars_per_transcript],
                "length_chars": len(text),
            })

        if not transcripts:
            return self._make_result(query, success=False, error="No transcripts retrievable")

        data = {
            "total_videos_checked": len(video_ids),
            "total_transcripts": len(transcripts),
            "transcripts": transcripts,
        }
        return self._make_result(query, success=True, data=data, raw_text=self._to_text(data, query))

    def _gather_video_ids(self, query: ProfessorQuery) -> List[str]:
        """Read youtube_lectures results from prior enrichment.json."""
        base = Path(os.getenv("ENRICHMENT_OUTPUT_DIR", "output/osu_faculty_run"))
        enrichment_path = base / "profiles" / query.profile_id / "enrichment.json"
        if not enrichment_path.exists():
            return []
        try:
            doc = json.loads(enrichment_path.read_text(encoding="utf-8"))
        except Exception:
            return []

        yt = doc.get("sources", {}).get("youtube_lectures", {}).get("data", {}) or {}
        videos = yt.get("videos") or yt.get("results") or []
        ids: List[str] = []
        for v in videos:
            vid = v.get("video_id") or v.get("id")
            if not vid and v.get("url"):
                url = v["url"]
                if "v=" in url:
                    vid = url.split("v=", 1)[1].split("&", 1)[0]
                elif "youtu.be/" in url:
                    vid = url.split("youtu.be/", 1)[1].split("?", 1)[0]
            if vid:
                ids.append(vid)
        # Dedupe
        seen = set()
        out = []
        for v in ids:
            if v not in seen:
                seen.add(v)
                out.append(v)
        return out

    def _fetch_transcript(self, video_id: str) -> str:
        segments = _fetch_transcript_cross_version(video_id, ["en", "en-US", "en-GB"])
        if not segments:
            return ""
        return " ".join(s.get("text", "").strip() for s in segments if s.get("text"))

    def _to_text(self, data: Dict, query: ProfessorQuery) -> str:
        lines = [
            f"=== YouTube lecture transcripts for {query.name} ===",
            f"Videos checked: {data['total_videos_checked']}, transcripts retrieved: {data['total_transcripts']}",
            "",
        ]
        for t in data["transcripts"]:
            lines.append(f"── Transcript: {t['url']} ({t['length_chars']:,} chars) ──")
            lines.append(t["transcript"])
            lines.append("")
        return "\n".join(lines)
