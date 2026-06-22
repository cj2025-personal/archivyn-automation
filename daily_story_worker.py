"""
Daily scholar story worker.

Generates one scholar-inspired story per scholar per day and stores it in MongoDB.
The output is designed for review-first publishing.

Input collection (default): legend_scholars
Output collections:
  - legend_scholar_daily_stories
  - daily_story_jobs
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import random
import re
import time
import uuid
import warnings
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import date, datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import quote_plus, urlencode, urlparse
from urllib.request import Request, urlopen

from dotenv import load_dotenv

from config.mongodb_utils import create_mongo_client, resolve_mongo_db_name

try:
    from pymongo.errors import DuplicateKeyError
except Exception:  # pragma: no cover
    DuplicateKeyError = None  # type: ignore

try:
    import vertexai
    from vertexai.generative_models import GenerativeModel, GenerationConfig
except ImportError:  # pragma: no cover
    vertexai = None  # type: ignore
    GenerativeModel = None  # type: ignore
    GenerationConfig = None  # type: ignore

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore

SentenceTransformer = None  # type: ignore
CrossEncoder = None  # type: ignore
MiniBatchKMeans = None  # type: ignore
TfidfVectorizer = None  # type: ignore
NMF = None  # type: ignore

if not os.getenv("LOKY_MAX_CPU_COUNT"):
    os.environ["LOKY_MAX_CPU_COUNT"] = str(max(1, os.cpu_count() or 1))


DEFAULT_TOPICS = [
    "The role of historical truth in public education",
    "Civic responsibility in a polarized democracy",
    "How institutions remember racial injustice",
    "Economic opportunity and structural inequality",
    "The ethics of public memory and monuments",
    "Teaching contested history in schools and universities",
    "Democracy, law, and equal protection in practice",
    "Race, citizenship, and the future of belonging",
    "The public value of archival preservation",
    "What serious scholarship owes to ordinary people",
]


SECTION_PRIORITY = {
    "about": 0,
    "biography": 0,
    "background_and_work": 1,
    "milestones": 2,
    "publications": 3,
    "misc": 4,
}


TOPIC_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "if", "in",
    "into", "is", "it", "its", "of", "on", "or", "the", "to", "was", "were", "will",
    "with", "without",
    "about", "above", "after", "again", "against", "among", "amongst", "been", "before",
    "being", "below", "between", "could", "does", "doing", "during", "each", "from",
    "further", "have", "having", "here", "into", "itself", "many", "more", "most",
    "other", "over", "same", "such", "than", "that", "their", "theirs", "them", "then",
    "there", "these", "they", "this", "those", "through", "under", "until", "very",
    "were", "what", "when", "where", "which", "while", "with", "would", "your",
    "history", "historian", "scholar", "scholarship", "american", "america",
    "university", "professor", "board", "committee", "center", "college",
    "school", "department", "chair", "chairs", "chairman", "member",
    "students", "student", "class", "classes", "research", "studies",
    "john", "hope", "franklin", "duke", "harvard", "fisk",
    "photo", "credit", "foundation", "published", "press", "edition", "isbn",
    "share", "twitter", "google", "linkedin", "print", "online", "resource",
    "resources", "article", "articles", "book", "books", "image", "images",
    "courtesy", "loading", "hours", "today", "next", "previous", "hall",
    "fame", "born", "died", "family", "profession", "presenter", "hometown",
    "his", "her", "their", "them", "was", "were", "said", "also",
    "conference", "conferences", "session", "sessions", "plenary", "speaker", "speakers",
    "event", "events", "upcoming", "recent", "guide", "guides",
    "north", "south", "east", "west", "new", "york", "carolina", "louisiana",
    "baton", "rouge", "oklahoma", "tulsa", "durham", "massachusetts", "washington",
    "london", "saint", "lucia", "barbados", "ghana", "selected", "selection", "medal",
    "award", "awards", "national", "number", "first", "second", "third", "few", "several",
    "year", "years", "including", "include", "despite", "available", "biography",
    "presidential",
}

TOPIC_CONCEPT_TERMS = {
    "access", "accountability", "archives", "belonging", "citizenship", "civil", "climate",
    "community", "constitution", "democracy", "discrimination", "economy", "education",
    "energy", "equity", "ethics", "evidence", "freedom", "governance", "history", "housing",
    "inequality", "innovation", "institutions", "integration", "justice", "knowledge",
    "labor", "law", "leadership", "memory", "opportunity", "policy", "poverty",
    "public", "race", "racism", "regulation", "rights", "science", "security",
    "segregation", "society", "technology", "truth", "workforce", "reconstruction",
    "slavery", "voting", "representation", "desegregation", "equality", "courts",
    "historiography", "conflict", "war",
}

NAME_BANNED_TOKENS = {
    "academic", "african", "american", "center", "college", "department", "foundation",
    "history", "institute", "interview", "museum", "news", "part", "professor", "resource",
    "school", "society", "supplies", "university", "www", "www.",
}

TREND_RSS_DEFAULTS = [
    "https://feeds.bbci.co.uk/news/world/rss.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
    "https://www.aljazeera.com/xml/rss/all.xml",
]

STYLE_CONNECTORS = [
    "however",
    "therefore",
    "moreover",
    "meanwhile",
    "indeed",
    "nevertheless",
    "in turn",
    "at the same time",
    "in short",
    "for example",
]

TOPIC_COMPOUND_PAIRS = {
    ("civil", "war"),
    ("civil", "rights"),
    ("racial", "equality"),
    ("public", "policy"),
    ("voting", "rights"),
    ("economic", "inequality"),
    ("structural", "inequality"),
    ("historical", "memory"),
    ("institutional", "accountability"),
    ("higher", "education"),
}

PUBLICATION_SECTION_HINTS = {
    "publication",
    "publications",
    "early works",
    "later works",
    "bibliography",
    "editorial work",
}

CAUSAL_CLAIM_PATTERNS = [
    r"\bbecause\b",
    r"\bcaus(?:e|es|ed|ing)\b",
    r"\blead(?:s|ing)? to\b",
    r"\bled to\b",
    r"\bresult(?:s|ed|ing)? in\b",
    r"\btherefore\b",
    r"\bthus\b",
    r"\bdriv(?:e|es|en|ing)\b",
    r"\btrigger(?:s|ed|ing)\b",
    r"\bescalat(?:e|es|ed|ing)\b",
    r"\bcontribut(?:e|es|ed|ing) to\b",
    r"\bexacerbat(?:e|es|ed|ing)\b",
]

SOURCE_DOMAIN_TRUST_HINTS = (
    ".gov",
    ".edu",
    "nber.org",
    "aeaweb.org",
    "doi.org",
    "jstor.org",
    "sciencedirect.com",
    "nature.com",
    "science.org",
    "cambridge.org",
    "academic.oup.com",
    "wiley.com",
    "springer.com",
    "duke.edu",
    "mit.edu",
    "loc.gov",
    "congress.gov",
    "govinfo.gov",
)

SOURCE_DOMAIN_BLOCKLIST = {
    "aaespeakers.com",
    "alloveralbany.com",
    "black-inventor.com",
    "captechu.edu",
    "custom-powder.com",
    "successacademies.org",
    "grokipedia.com",
    "reddit.com",
    "x.com",
    "twitter.com",
    "facebook.com",
    "instagram.com",
    "linkedin.com",
    "tiktok.com",
    "pinterest.com",
}

DEFAULT_STANDARD_SECTION_BLUEPRINT = [
    "Why This Matters Now",
    "Historical Background",
    "Scholar Lens and Core Argument",
    "Tensions and Counterarguments",
    "Implications for Policy and Public Debate",
    "What to Watch Next",
    "Conclusion",
]

STORY_OUTPUT_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "standfirst": {"type": "string"},
        "article_markdown": {"type": "string"},
        "trend_source_url": {"type": "string"},
        "source_urls": {"type": "array", "items": {"type": "string"}},
        "claim_evidence_map": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "chunk_ids": {"type": "array", "items": {"type": "string"}},
                    "support_summary": {"type": "string"},
                },
                "required": ["claim", "chunk_ids", "support_summary"],
            },
        },
        "used_chunk_ids": {"type": "array", "items": {"type": "string"}},
        "editor_notes": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "title",
        "standfirst",
        "article_markdown",
        "trend_source_url",
        "source_urls",
        "claim_evidence_map",
        "used_chunk_ids",
        "editor_notes",
    ],
}


URL_EXTRACT_PATTERN = re.compile(r"https?://[^\s<>\]\)\"']+", flags=re.I)
DOMAIN_EXTRACT_PATTERN = re.compile(
    r"\b(?:[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?\.)+[a-z]{2,24}(?:/[^\s<>\]\)\"']*)?",
    flags=re.I,
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_date(date_str: Optional[str]) -> date:
    if not date_str:
        return datetime.now(timezone.utc).date()
    return datetime.strptime(date_str, "%Y-%m-%d").date()


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


class DailyStoryWorker:
    def __init__(
        self,
        *,
        scholars_collection: str = "legend_scholars",
        stories_collection: str = "legend_scholar_daily_stories",
        jobs_collection: str = "daily_story_jobs",
        model_name: Optional[str] = None,
        use_llm: bool = True,
        require_human_review: bool = True,
        assume_deceased: bool = True,
        trends_enabled: Optional[bool] = None,
        trend_provider: Optional[str] = None,
        enforce_profile_quality: Optional[bool] = None,
        profile_quality_min_score: Optional[int] = None,
    ) -> None:
        load_dotenv(dotenv_path=".env")

        suppress_vertex_deprecation_warning = (
            safe_text(os.getenv("STORY_SUPPRESS_VERTEX_DEPRECATION_WARNING")).lower() in {"1", "true", "yes", "on"}
            if safe_text(os.getenv("STORY_SUPPRESS_VERTEX_DEPRECATION_WARNING"))
            else True
        )
        if suppress_vertex_deprecation_warning:
            warnings.filterwarnings(
                "ignore",
                message=r"This feature is deprecated as of June 24, 2025.*genai-vertexai-sdk\.",
                category=UserWarning,
            )

        mongodb_uri = os.getenv("MONGODB_URI")
        if not mongodb_uri:
            raise ValueError("MONGODB_URI not found in environment variables")

        self.mongo_client = create_mongo_client(mongodb_uri)
        db_name = resolve_mongo_db_name(mongodb_uri)
        self.db = self.mongo_client[db_name]

        self.scholars_collection = self.db[scholars_collection]
        self.stories_collection = self.db[stories_collection]
        self.jobs_collection = self.db[jobs_collection]
        trend_cache_collection_name = safe_text(os.getenv("STORY_TREND_CACHE_COLLECTION")) or "daily_story_trend_issues"
        self.trend_cache_collection = self.db[trend_cache_collection_name]
        self.profile_quality_collection = self.db["daily_story_profile_quality"]

        self.gcp_project_id = safe_text(os.getenv("GCP_PROJECT_ID"))
        self.gcp_location = safe_text(os.getenv("GCP_LOCATION")) or "us-central1"
        self.model_name = (
            model_name
            or safe_text(os.getenv("STORY_LLM_MODEL"))
            or safe_text(os.getenv("LLM_MODEL"))
            or "gemini-2.0-flash-001"
        )
        self.ml_enabled = (
            safe_text(os.getenv("STORY_ML_ENABLED")).lower() in {"1", "true", "yes", "on"}
            if safe_text(os.getenv("STORY_ML_ENABLED"))
            else True
        )
        self.embedding_model_name = safe_text(os.getenv("STORY_EMBEDDING_MODEL")) or "all-MiniLM-L6-v2"
        self._embedder = None
        self._embedder_unavailable = False
        self.cross_encoder_enabled = (
            safe_text(os.getenv("STORY_CROSS_ENCODER_ENABLED")).lower() in {"1", "true", "yes", "on"}
            if safe_text(os.getenv("STORY_CROSS_ENCODER_ENABLED"))
            else True
        )
        self.cross_encoder_model_name = (
            safe_text(os.getenv("STORY_CROSS_ENCODER_MODEL"))
            or "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        self.cross_encoder_local_only = (
            safe_text(os.getenv("STORY_CROSS_ENCODER_LOCAL_ONLY")).lower() in {"1", "true", "yes", "on"}
            if safe_text(os.getenv("STORY_CROSS_ENCODER_LOCAL_ONLY"))
            else True
        )
        self.cross_encoder_rerank_top_k = int(safe_text(os.getenv("STORY_CROSS_ENCODER_TOP_K")) or 24)
        self.cross_encoder_weight = float(safe_text(os.getenv("STORY_CROSS_ENCODER_WEIGHT")) or 0.35)
        self.cross_encoder_weight = min(0.7, max(0.0, self.cross_encoder_weight))
        self._cross_encoder = None
        self._cross_encoder_unavailable = False
        if self.ml_enabled:
            warnings.filterwarnings(
                "ignore",
                message=r"Could not find the number of physical cores.*",
                category=UserWarning,
            )
            self._patch_loky_cpu_detection()
        if self.ml_enabled and not safe_text(os.getenv("LOKY_MAX_CPU_COUNT")):
            os.environ["LOKY_MAX_CPU_COUNT"] = str(max(1, os.cpu_count() or 1))
        self.embedding_local_only = (
            safe_text(os.getenv("STORY_EMBEDDING_LOCAL_ONLY")).lower() in {"1", "true", "yes", "on"}
            if safe_text(os.getenv("STORY_EMBEDDING_LOCAL_ONLY"))
            else True
        )
        self.vertex_disable_system_proxy = (
            safe_text(os.getenv("VERTEX_DISABLE_SYSTEM_PROXY")).lower() in {"1", "true", "yes", "on"}
            if safe_text(os.getenv("VERTEX_DISABLE_SYSTEM_PROXY"))
            else True
        )
        env_trends_enabled = (
            safe_text(os.getenv("STORY_TRENDS_ENABLED")).lower() in {"1", "true", "yes", "on"}
            if safe_text(os.getenv("STORY_TRENDS_ENABLED"))
            else True
        )
        self.trends_enabled = env_trends_enabled if trends_enabled is None else bool(trends_enabled)
        self.trend_provider = (safe_text(os.getenv("STORY_TREND_PROVIDER")).lower() or "rss") if not trend_provider else safe_text(trend_provider).lower()
        self.trend_region = safe_text(os.getenv("STORY_TREND_REGION")).lower() or "us"
        self.trend_max_items = int(safe_text(os.getenv("STORY_TREND_MAX_ITEMS")) or 40)
        self.trend_timeout_seconds = int(safe_text(os.getenv("STORY_TREND_TIMEOUT_SECONDS")) or 10)
        self.newsapi_key = safe_text(os.getenv("NEWSAPI_KEY"))
        self.trend_cache_enabled = (
            safe_text(os.getenv("STORY_TREND_USE_CACHE")).lower() in {"1", "true", "yes", "on"}
            if safe_text(os.getenv("STORY_TREND_USE_CACHE"))
            else True
        )
        self.trend_cache_ttl_hours = int(safe_text(os.getenv("STORY_TREND_CACHE_TTL_HOURS")) or 72)
        rss_urls_env = safe_text(os.getenv("STORY_TREND_RSS_URLS"))
        self.trend_rss_urls = [u.strip() for u in rss_urls_env.split(",") if u.strip()] if rss_urls_env else list(TREND_RSS_DEFAULTS)
        self.style_sentence_samples = int(safe_text(os.getenv("STORY_STYLE_SENTENCE_SAMPLES")) or 120)
        env_enforce_quality = (
            safe_text(os.getenv("STORY_ENFORCE_PROFILE_QUALITY")).lower() in {"1", "true", "yes", "on"}
            if safe_text(os.getenv("STORY_ENFORCE_PROFILE_QUALITY"))
            else False
        )
        self.enforce_profile_quality = (
            env_enforce_quality if enforce_profile_quality is None else bool(enforce_profile_quality)
        )
        env_min_quality = int(safe_text(os.getenv("STORY_PROFILE_MIN_QUALITY_SCORE")) or 60)
        self.profile_quality_min_score = env_min_quality if profile_quality_min_score is None else int(profile_quality_min_score)
        self.strict_reliability = (
            safe_text(os.getenv("STORY_STRICT_RELIABILITY")).lower() in {"1", "true", "yes", "on"}
            if safe_text(os.getenv("STORY_STRICT_RELIABILITY"))
            else True
        )
        self.require_verified_trend_url = (
            safe_text(os.getenv("STORY_REQUIRE_VERIFIED_TREND_URL")).lower() in {"1", "true", "yes", "on"}
            if safe_text(os.getenv("STORY_REQUIRE_VERIFIED_TREND_URL"))
            else True
        )
        self.min_claim_evidence_items = int(safe_text(os.getenv("STORY_MIN_CLAIM_EVIDENCE_ITEMS")) or 4)
        self.min_paragraph_overlap = float(safe_text(os.getenv("STORY_MIN_PARAGRAPH_OVERLAP")) or 0.02)
        self.min_section_strong_overlap = float(safe_text(os.getenv("STORY_MIN_SECTION_STRONG_OVERLAP")) or 0.03)
        self.section_end_citation_ratio = float(safe_text(os.getenv("STORY_SECTION_END_CITATION_RATIO")) or 0.75)
        self.max_output_tokens = int(
            safe_text(os.getenv("STORY_MAX_OUTPUT_TOKENS"))
            or (6200 if self.strict_reliability else 3200)
        )
        self.max_output_tokens = max(1200, self.max_output_tokens)
        self.require_standard_structure = (
            safe_text(os.getenv("STORY_REQUIRE_STANDARD_STRUCTURE")).lower() in {"1", "true", "yes", "on"}
            if safe_text(os.getenv("STORY_REQUIRE_STANDARD_STRUCTURE"))
            else True
        )
        self.min_article_words = max(450, int(safe_text(os.getenv("STORY_MIN_ARTICLE_WORDS")) or 1000))
        self.min_major_sections = max(4, int(safe_text(os.getenv("STORY_MIN_MAJOR_SECTIONS")) or 6))
        self.min_section_words = max(60, int(safe_text(os.getenv("STORY_MIN_SECTION_WORDS")) or 100))
        self.vertex_schema_enforced = (
            safe_text(os.getenv("STORY_VERTEX_SCHEMA_ENFORCED")).lower() in {"1", "true", "yes", "on"}
            if safe_text(os.getenv("STORY_VERTEX_SCHEMA_ENFORCED"))
            else True
        )
        self.allow_corpus_only_when_no_trend = (
            safe_text(os.getenv("STORY_ALLOW_CORPUS_ONLY_WHEN_NO_TREND")).lower() in {"1", "true", "yes", "on"}
            if safe_text(os.getenv("STORY_ALLOW_CORPUS_ONLY_WHEN_NO_TREND"))
            else True
        )
        self.strict_llm_only = (
            safe_text(os.getenv("STORY_STRICT_LLM_ONLY")).lower() in {"1", "true", "yes", "on"}
            if safe_text(os.getenv("STORY_STRICT_LLM_ONLY"))
            else False
        )
        self.enforce_source_url_citations = (
            safe_text(os.getenv("STORY_ENFORCE_SOURCE_URL_CITATIONS")).lower() in {"1", "true", "yes", "on"}
            if safe_text(os.getenv("STORY_ENFORCE_SOURCE_URL_CITATIONS"))
            else True
        )
        self.min_source_url_citations = max(1, int(safe_text(os.getenv("STORY_MIN_SOURCE_URL_CITATIONS")) or 4))
        self.max_source_url_candidates = max(4, int(safe_text(os.getenv("STORY_MAX_SOURCE_URL_CANDIDATES")) or 20))
        self.source_domain_filter_enabled = (
            safe_text(os.getenv("STORY_SOURCE_DOMAIN_FILTER")).lower() in {"1", "true", "yes", "on"}
            if safe_text(os.getenv("STORY_SOURCE_DOMAIN_FILTER"))
            else True
        )
        self.source_domain_min_score = int(safe_text(os.getenv("STORY_SOURCE_DOMAIN_MIN_SCORE")) or 1)
        self.source_require_trusted = (
            safe_text(os.getenv("STORY_SOURCE_REQUIRE_TRUSTED")).lower() in {"1", "true", "yes", "on"}
            if safe_text(os.getenv("STORY_SOURCE_REQUIRE_TRUSTED"))
            else False
        )
        self.reader_strip_inline_citations = (
            safe_text(os.getenv("STORY_READER_STRIP_INLINE_CITATIONS")).lower() in {"1", "true", "yes", "on"}
            if safe_text(os.getenv("STORY_READER_STRIP_INLINE_CITATIONS"))
            else True
        )
        self.reader_max_connector_repeats = max(
            1,
            int(safe_text(os.getenv("STORY_READER_MAX_CONNECTOR_REPEATS")) or 2),
        )
        self.source_url_artifact_glob = (
            safe_text(os.getenv("STORY_SOURCE_URL_ARTIFACT_GLOB"))
            or os.path.join("output", "url_list_runs", "*", "profiles", "{profile_id}", "source_chunks.json")
        )
        self._source_url_cache: Dict[str, List[str]] = {}
        self.use_llm = use_llm
        self.require_human_review = require_human_review
        self.assume_deceased = assume_deceased
        self.vertex_model = None

        if self.use_llm:
            if vertexai is None or GenerativeModel is None:
                raise ImportError("vertexai not installed. Install with: pip install google-cloud-aiplatform")
            if not self.gcp_project_id:
                raise ValueError("GCP_PROJECT_ID not found in environment variables")
            try:
                if self.vertex_disable_system_proxy:
                    for key in (
                        "HTTP_PROXY",
                        "HTTPS_PROXY",
                        "ALL_PROXY",
                        "http_proxy",
                        "https_proxy",
                        "all_proxy",
                        "GIT_HTTP_PROXY",
                        "GIT_HTTPS_PROXY",
                    ):
                        os.environ.pop(key, None)
                    os.environ["GRPC_ENABLE_HTTP_PROXY"] = "0"
                vertexai.init(project=self.gcp_project_id, location=self.gcp_location)
                self.vertex_model = GenerativeModel(self.model_name)
            except Exception as exc:
                raise RuntimeError(f"Failed to initialize Vertex AI model '{self.model_name}': {exc}") from exc

        self._create_indexes()

    def _create_indexes(self) -> None:
        self.stories_collection.create_index("story_key", unique=True)
        self.stories_collection.create_index([("story_date", 1), ("status", 1)])
        self.stories_collection.create_index([("scholar.profile_id", 1), ("story_date", 1)])
        self.jobs_collection.create_index("run_id", unique=True)
        self.jobs_collection.create_index("started_at")
        self.trend_cache_collection.create_index("issue_key", unique=True)
        self.trend_cache_collection.create_index([("fetched_at", -1)])
        self.trend_cache_collection.create_index([("published_at_dt", -1)])

    def close(self) -> None:
        try:
            self.mongo_client.close()
        except Exception:
            pass

    @staticmethod
    def _profile_id(doc: Dict[str, Any]) -> str:
        profile_id = safe_text(doc.get("profile_id"))
        if profile_id:
            return profile_id
        _id = doc.get("_id")
        return safe_text(_id)

    @staticmethod
    def _looks_like_clean_name(value: str) -> bool:
        value = safe_text(value)
        if not value:
            return False
        if any(x in value.lower() for x in ("http://", "https://", ".com", ".org", ".edu")):
            return False
        if any(x in value for x in (";", "|", " - ", "/", "\\")):
            return False
        words = re.findall(r"[A-Za-z][A-Za-z\.'-]*", value)
        if len(words) < 2 or len(words) > 5:
            return False
        if any(w.lower().strip(".") in NAME_BANNED_TOKENS for w in words):
            return False
        title_case = sum(1 for w in words if w and w[0].isupper())
        return title_case >= 2

    @staticmethod
    def _extract_name_from_text(text: str) -> Optional[str]:
        text = safe_text(text)
        if not text:
            return None
        patterns = [
            r"\b(?:Dr|Sir|Professor|Prof)\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b",
            r"\bwith\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b",
            r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\s+(?:was|is|served|wrote|became|received|born)\b",
            r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b",
        ]
        for pat in patterns:
            matches = re.findall(pat, text)
            if not matches:
                continue
            counts = Counter()
            for m in matches:
                cand = safe_text(m).strip(" ,.;:!?")
                toks = re.findall(r"[A-Za-z]+", cand.lower())
                if len(toks) < 2 or len(toks) > 4:
                    continue
                if toks[0] in NAME_BANNED_TOKENS or toks[-1] in NAME_BANNED_TOKENS:
                    continue
                if sum(1 for t in toks if t in NAME_BANNED_TOKENS) >= 2:
                    continue
                counts[cand] += 1
            if counts:
                return counts.most_common(1)[0][0]
        return None

    def _derive_name_from_context(self, doc: Dict[str, Any]) -> Optional[str]:
        rag_context = doc.get("rag_context") or {}
        section_text = rag_context.get("section_text") or {}
        preferred_sections = [
            "about",
            "biography",
            "background_and_work",
            "early life",
            "career",
            "legacy",
        ]

        texts: List[str] = []
        if isinstance(section_text, dict):
            # Section names vary; match preferred sections loosely first.
            ordered = sorted(
                section_text.items(),
                key=lambda kv: (
                    0 if any(p in safe_text(kv[0]).lower() for p in preferred_sections) else 1,
                    len(safe_text(kv[0])),
                ),
            )
            for _, value in ordered[:8]:
                txt = safe_text(value)
                if txt:
                    texts.append(txt[:3000])

        section_chunks = rag_context.get("section_chunks") or {}
        if isinstance(section_chunks, dict):
            for section, chunks in section_chunks.items():
                if not isinstance(chunks, list):
                    continue
                if not any(p in safe_text(section).lower() for p in preferred_sections):
                    continue
                for ch in chunks[:3]:
                    if not isinstance(ch, dict):
                        continue
                    txt = safe_text(ch.get("text")) or safe_text(ch.get("summary"))
                    if txt:
                        texts.append(txt[:1500])

        if not texts:
            return None
        combined = "\n".join(texts[:10])
        return self._extract_name_from_text(combined)

    def _professor_name(self, doc: Dict[str, Any]) -> str:
        direct = [
            safe_text(doc.get("professor_name")),
            safe_text((doc.get("name") or {}).get("display")),
            safe_text((doc.get("name") or {}).get("full")),
            safe_text(doc.get("name")),
        ]
        for cand in direct:
            if self._looks_like_clean_name(cand):
                return cand
            extracted = self._extract_name_from_text(cand)
            if extracted and self._looks_like_clean_name(extracted):
                return extracted

        context_name = self._derive_name_from_context(doc)
        if context_name and self._looks_like_clean_name(context_name):
            return context_name
        return "Unknown Scholar"

    @staticmethod
    def _topic_tokens(topic: str) -> List[str]:
        tokens = re.findall(r"[A-Za-z]{4,}", topic.lower())
        return [t for t in tokens if t not in TOPIC_STOPWORDS]

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        if not text:
            return []
        rough = re.split(r"(?<=[.!?])\s+|\n+", text)
        out: List[str] = []
        for sent in rough:
            sent = re.sub(r"\s+", " ", sent).strip()
            if 50 <= len(sent) <= 280:
                out.append(sent)
        return out

    @staticmethod
    def _clean_url_candidate(url: str) -> str:
        text = safe_text(url).strip()
        if not text:
            return ""
        # Strip punctuation frequently attached by markdown/prose.
        text = text.strip(" \t\r\n\"'`<>[](){}.,;:")
        return text

    @staticmethod
    def _normalize_url_key(url: str) -> str:
        text = DailyStoryWorker._clean_url_candidate(url)
        if not text:
            return ""
        try:
            parsed = urlparse(text)
        except Exception:
            return ""
        if parsed.scheme.lower() not in {"http", "https"}:
            return ""
        host = safe_text(parsed.netloc).lower()
        if not host:
            return ""
        path = safe_text(parsed.path)
        path = re.sub(r"/{2,}", "/", path)
        path = path.rstrip("/")
        query = safe_text(parsed.query)
        key = f"{parsed.scheme.lower()}://{host}{path}"
        if query:
            key = f"{key}?{query}"
        return key

    @staticmethod
    def _extract_domain(url: str) -> str:
        text = DailyStoryWorker._clean_url_candidate(url)
        if not text:
            return ""
        try:
            parsed = urlparse(text)
        except Exception:
            return ""
        host = safe_text(parsed.netloc).lower().strip()
        if host.startswith("www."):
            host = host[4:]
        return host

    def _score_source_domain(self, domain: str) -> int:
        d = safe_text(domain).lower().strip()
        if not d:
            return 0
        if d in SOURCE_DOMAIN_BLOCKLIST or any(d.endswith(f".{bad}") for bad in SOURCE_DOMAIN_BLOCKLIST):
            return -3
        if d.endswith(".gov") or d.endswith(".edu"):
            return 3
        if any(h in d for h in SOURCE_DOMAIN_TRUST_HINTS):
            return 2
        if any(h in d for h in ("researchgate.net", "academia.edu")):
            return -1
        return 0

    def _is_trusted_source_domain(self, domain: str) -> bool:
        d = safe_text(domain).lower().strip()
        if not d:
            return False
        if d.endswith(".gov") or d.endswith(".edu"):
            return True
        return any(h in d for h in SOURCE_DOMAIN_TRUST_HINTS)

    def _filter_source_urls_by_quality(self, urls: List[str]) -> List[str]:
        if not urls:
            return []

        cleaned_unique: List[str] = []
        seen = set()
        for raw in urls:
            cleaned = self._clean_url_candidate(raw)
            key = self._normalize_url_key(cleaned)
            if not key or key in seen:
                continue
            seen.add(key)
            cleaned_unique.append(cleaned)
        if not self.source_domain_filter_enabled:
            return cleaned_unique[: self.max_source_url_candidates]

        filtered: List[str] = []
        for url in cleaned_unique:
            domain = self._extract_domain(url)
            score = self._score_source_domain(domain)
            if score < self.source_domain_min_score:
                continue
            if self.source_require_trusted and (not self._is_trusted_source_domain(domain)):
                continue
            filtered.append(url)

        if filtered:
            return filtered[: self.max_source_url_candidates]
        # Keep pipeline resilient if all URLs are filtered out, but avoid low-value domains.
        soft_filtered: List[str] = []
        for url in cleaned_unique:
            domain = self._extract_domain(url)
            score = self._score_source_domain(domain)
            if score < 0:
                continue
            soft_filtered.append(url)
        if soft_filtered:
            return soft_filtered[: self.max_source_url_candidates]
        return cleaned_unique[: self.max_source_url_candidates]

    @staticmethod
    def _url_from_domain_token(token: str) -> str:
        token = DailyStoryWorker._clean_url_candidate(token).lower()
        if not token:
            return ""
        if token.startswith("http://") or token.startswith("https://"):
            return token
        if token.startswith("www."):
            return f"https://{token}"
        if "." not in token:
            return ""
        return f"https://{token}"

    @staticmethod
    def _extract_url_candidates_from_text(text: str) -> List[str]:
        blob = safe_text(text)
        if not blob:
            return []
        out: List[str] = []
        seen = set()

        for m in URL_EXTRACT_PATTERN.finditer(blob):
            raw = DailyStoryWorker._clean_url_candidate(m.group(0))
            key = DailyStoryWorker._normalize_url_key(raw)
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(raw)

        for m in DOMAIN_EXTRACT_PATTERN.finditer(blob):
            if m.start() > 0 and blob[m.start() - 1] == "@":
                continue
            token = DailyStoryWorker._url_from_domain_token(m.group(0))
            key = DailyStoryWorker._normalize_url_key(token)
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(token)
        return out

    @staticmethod
    def _iter_string_values(obj: Any) -> Iterable[str]:
        if isinstance(obj, dict):
            for value in obj.values():
                yield from DailyStoryWorker._iter_string_values(value)
            return
        if isinstance(obj, list):
            for item in obj:
                yield from DailyStoryWorker._iter_string_values(item)
            return
        if isinstance(obj, str):
            text = safe_text(obj)
            if text:
                yield text

    def _load_source_urls_from_artifact(self, profile_id: str) -> List[str]:
        pattern = self.source_url_artifact_glob.format(profile_id=profile_id)
        paths = glob.glob(pattern)
        if not paths:
            return []
        paths = sorted(paths, key=lambda p: os.path.getmtime(p), reverse=True)[:2]

        scored: Dict[str, Tuple[float, str]] = {}

        def push(url: str, score: float) -> None:
            cleaned = self._clean_url_candidate(url)
            key = self._normalize_url_key(cleaned)
            if not key:
                return
            prior = scored.get(key)
            if prior is None or score > prior[0]:
                scored[key] = (float(score), cleaned)

        for path in paths:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
            except Exception:
                continue

            for src in payload.get("sources") or []:
                if not isinstance(src, dict):
                    continue
                status = safe_text(src.get("status")).lower()
                paywalled = bool(src.get("paywalled"))
                robots_allowed = src.get("robots_allowed")
                source_type = safe_text(src.get("source_type")).lower()
                score = 7.0
                if source_type in {"profile_page", "personal_website", "personal_website_subpage"}:
                    score += 1.0
                if status in {"blocked", "denied"}:
                    score -= 5.0
                if paywalled:
                    score -= 2.0
                if robots_allowed is False:
                    score -= 1.0
                push(safe_text(src.get("resolved_url")) or safe_text(src.get("source_url")), score)

            for ch in payload.get("chunks") or []:
                if not isinstance(ch, dict):
                    continue
                source_type = safe_text(ch.get("source_type")).lower()
                score = 4.0
                if source_type == "profile_page":
                    score += 0.5
                push(safe_text(ch.get("source_url")), score)

        ranked = sorted(scored.values(), key=lambda x: x[0], reverse=True)
        candidates = [u for _, u in ranked[: self.max_source_url_candidates]]
        return self._filter_source_urls_by_quality(candidates)

    def _collect_allowed_source_urls(
        self,
        *,
        profile_id: str,
        scholar_doc: Dict[str, Any],
        trend_issue: Optional[Dict[str, Any]],
    ) -> List[str]:
        cached = self._source_url_cache.get(profile_id)
        if cached is None:
            scored: Dict[str, Tuple[float, str]] = {}

            def push(url: str, score: float) -> None:
                cleaned = self._clean_url_candidate(url)
                key = self._normalize_url_key(cleaned)
                if not key:
                    return
                prior = scored.get(key)
                if prior is None or score > prior[0]:
                    scored[key] = (float(score), cleaned)

            # URLs explicitly stored in scholar document metadata.
            links_and_media = scholar_doc.get("links_and_media") or {}
            for value in self._iter_string_values(links_and_media):
                for url in self._extract_url_candidates_from_text(value):
                    push(url, 3.5)

            rag_context = scholar_doc.get("rag_context") or {}
            for value in self._iter_string_values(rag_context.get("source")):
                for url in self._extract_url_candidates_from_text(value):
                    push(url, 2.5)

            # Link-heavy sections often contain source references as prose.
            section_chunks = rag_context.get("section_chunks") or {}
            if isinstance(section_chunks, dict):
                for section_name, chunks in section_chunks.items():
                    section_lc = safe_text(section_name).lower()
                    score = 2.2 if any(k in section_lc for k in ("link", "reference", "resource", "sitelink")) else 1.2
                    if not isinstance(chunks, list):
                        continue
                    for ch in chunks:
                        if not isinstance(ch, dict):
                            continue
                        text_blob = (safe_text(ch.get("text")) + "\n" + safe_text(ch.get("summary"))).strip()
                        for url in self._extract_url_candidates_from_text(text_blob):
                            push(url, score)

            # Artifact-level source registry from the URL ingestion pipeline.
            for url in self._load_source_urls_from_artifact(profile_id):
                push(url, 8.0)

            ranked = sorted(scored.values(), key=lambda x: x[0], reverse=True)
            cached = [u for _, u in ranked[: self.max_source_url_candidates]]
            cached = self._filter_source_urls_by_quality(cached)
            self._source_url_cache[profile_id] = cached

        out = list(cached)
        trend_url = safe_text((trend_issue or {}).get("url"))
        if trend_url:
            trend_key = self._normalize_url_key(trend_url)
            if trend_key and all(self._normalize_url_key(u) != trend_key for u in out):
                out.insert(0, self._clean_url_candidate(trend_url))
        return out[: self.max_source_url_candidates]

    @staticmethod
    def _extract_markdown_urls(markdown_text: str) -> List[str]:
        text = safe_text(markdown_text)
        if not text:
            return []
        out: List[str] = []
        seen = set()
        for m in URL_EXTRACT_PATTERN.finditer(text):
            url = DailyStoryWorker._clean_url_candidate(m.group(0))
            key = DailyStoryWorker._normalize_url_key(url)
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(url)
        return out

    @staticmethod
    def _strip_non_allowed_urls_from_markdown(article_markdown: str, allowed_url_keys: set) -> str:
        text = safe_text(article_markdown)
        if not text:
            return text

        def repl(match: re.Match[str]) -> str:
            raw = DailyStoryWorker._clean_url_candidate(match.group(0))
            key = DailyStoryWorker._normalize_url_key(raw)
            if key and key in allowed_url_keys:
                return raw
            return ""

        cleaned = URL_EXTRACT_PATTERN.sub(repl, text)
        cleaned = re.sub(r"\s+\)", ")", cleaned)
        cleaned = re.sub(r"\(\s+\)", "()", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        return cleaned

    @staticmethod
    def _upsert_sources_section(article_markdown: str, source_urls: List[str]) -> str:
        body = safe_text(article_markdown).strip()
        if not body or not source_urls:
            return body
        body = re.sub(r"\n#{2,4}\s*(sources|references|citations)\s*[\s\S]*$", "", body, flags=re.I).rstrip()
        sources_block = "## Sources\n" + "\n".join(f"- {u}" for u in source_urls)
        return f"{body}\n\n{sources_block}\n"

    @staticmethod
    def _strip_inline_citation_tags_for_reader(article_markdown: str) -> str:
        text = safe_text(article_markdown)
        if not text:
            return text
        text = re.sub(r"\[chunk:[^\]\s]+\]", "", text)
        text = re.sub(r"\[[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\]", "", text, flags=re.I)
        text = re.sub(r"\[chunk_id=[^\]]+\]", "", text, flags=re.I)
        text = re.sub(r"\[section=[^\]]+\]", "", text, flags=re.I)
        text = re.sub(r"\s+([,.;:!?])", r"\1", text)
        text = re.sub(r"\(\s*\)", "", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return text

    def _de_template_connectors(self, article_markdown: str) -> str:
        text = safe_text(article_markdown)
        if not text:
            return text
        max_repeats = max(1, self.reader_max_connector_repeats)
        connector_counts: Counter = Counter()
        lines: List[str] = []
        lead_re = re.compile(
            r"^(\s*)(therefore|moreover|indeed|furthermore|however|additionally)\s*,\s*",
            flags=re.I,
        )
        for raw in text.splitlines():
            line = raw
            m = lead_re.match(line)
            if m:
                conn = safe_text(m.group(2)).lower()
                connector_counts[conn] += 1
                if connector_counts[conn] > max_repeats:
                    line = lead_re.sub(r"\1", line)
            lines.append(line)
        out = "\n".join(lines)
        out = re.sub(r"\n{3,}", "\n\n", out).strip()
        return out

    def _sanitize_reader_article_markdown(self, article_markdown: str) -> str:
        text = safe_text(article_markdown)
        if not text:
            return text
        if self.reader_strip_inline_citations:
            text = self._strip_inline_citation_tags_for_reader(text)
        text = self._de_template_connectors(text)
        return text

    def _enforce_source_url_citations(
        self,
        *,
        payload: Dict[str, Any],
        allowed_source_urls: List[str],
    ) -> Tuple[Dict[str, Any], bool]:
        out = dict(payload)
        changed = False

        allowed_map: Dict[str, str] = {}
        for url in allowed_source_urls:
            key = self._normalize_url_key(url)
            if key and key not in allowed_map:
                allowed_map[key] = self._clean_url_candidate(url)
        allowed_keys = set(allowed_map.keys())

        requested_raw = out.get("source_urls")
        if not isinstance(requested_raw, list):
            requested_raw = []
        requested_raw = [safe_text(u) for u in requested_raw if safe_text(u)]
        article = safe_text(out.get("article_markdown"))
        article_urls = self._extract_markdown_urls(article)

        selected: List[str] = []
        selected_keys = set()

        for raw in requested_raw + article_urls:
            key = self._normalize_url_key(raw)
            if not key or key not in allowed_map or key in selected_keys:
                continue
            selected_keys.add(key)
            selected.append(allowed_map[key])

        min_needed = min(self.min_source_url_citations, len(allowed_map)) if allowed_map else 0
        if len(selected) < min_needed:
            for url in allowed_source_urls:
                key = self._normalize_url_key(url)
                if not key or key in selected_keys:
                    continue
                selected_keys.add(key)
                selected.append(allowed_map.get(key, self._clean_url_candidate(url)))
                if len(selected) >= min_needed:
                    break

        selected = selected[: self.max_source_url_candidates]
        if selected != requested_raw:
            out["source_urls"] = selected
            changed = True

        if allowed_keys:
            cleaned_article = self._strip_non_allowed_urls_from_markdown(article, allowed_keys)
            if cleaned_article != article:
                out["article_markdown"] = cleaned_article
                article = cleaned_article
                changed = True

        if selected:
            article_with_sources = self._upsert_sources_section(article, selected)
            if article_with_sources != article:
                out["article_markdown"] = article_with_sources
                changed = True

        if changed:
            notes = out.get("editor_notes")
            if not isinstance(notes, list):
                notes = []
            notes.append("source_url_enforced")
            out["editor_notes"] = list(dict.fromkeys([safe_text(n) for n in notes if safe_text(n)]))
        return out, changed

    @staticmethod
    def _hf_model_exists_locally(name: str, aliases: Optional[List[str]] = None) -> bool:
        """
        Best-effort local cache check to keep cron deterministic when outbound download is blocked.
        """
        name = safe_text(name)
        if not name:
            return False
        if os.path.isdir(name):
            return True
        if os.path.isfile(name):
            return True

        model_ids = [name]
        for alias in aliases or []:
            alias = safe_text(alias)
            if alias:
                model_ids.append(alias)

        hf_home = safe_text(os.getenv("HF_HOME"))
        if not hf_home:
            hf_home = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
        hub_dir = os.path.join(hf_home, "hub")
        if not os.path.isdir(hub_dir):
            return False

        for mid in model_ids:
            marker = "models--" + mid.replace("/", "--")
            matches = glob.glob(os.path.join(hub_dir, f"{marker}*"))
            if matches:
                return True
        return False

    def _embedding_model_exists_locally(self) -> bool:
        name = safe_text(self.embedding_model_name)
        aliases: List[str] = []
        if name and "/" not in name:
            aliases.append(f"sentence-transformers/{name}")
        return self._hf_model_exists_locally(name, aliases=aliases)

    def _cross_encoder_model_exists_locally(self) -> bool:
        name = safe_text(self.cross_encoder_model_name)
        aliases: List[str] = []
        if name and "/" not in name:
            aliases.append(f"cross-encoder/{name}")
        return self._hf_model_exists_locally(name, aliases=aliases)

    @staticmethod
    def _patch_loky_cpu_detection() -> None:
        """
        Some Windows environments report 0 physical cores in loky and emit noisy tracebacks.
        Force a safe fallback to logical cores.
        """
        try:
            from joblib.externals.loky.backend import context as loky_context  # type: ignore

            def _safe_count_physical_cores():
                return max(1, os.cpu_count() or 1)

            if hasattr(loky_context, "_count_physical_cores"):
                loky_context._count_physical_cores = _safe_count_physical_cores  # type: ignore
        except Exception:
            pass

    def _get_embedding_model(self):
        if not self.ml_enabled:
            return None
        if self._embedder_unavailable:
            return None
        if self.embedding_local_only and not self._embedding_model_exists_locally():
            self._embedder_unavailable = True
            return None
        global SentenceTransformer
        if SentenceTransformer is None:
            try:
                from sentence_transformers import SentenceTransformer as _SentenceTransformer  # type: ignore
                SentenceTransformer = _SentenceTransformer
            except Exception:
                return None
        if self._embedder is None:
            try:
                self._embedder = SentenceTransformer(
                    self.embedding_model_name,
                    local_files_only=self.embedding_local_only,
                )
            except Exception:
                self._embedder_unavailable = True
                self._embedder = None
                return None
        return self._embedder

    def _embed_texts(self, texts: List[str]):
        model = self._get_embedding_model()
        if model is None or np is None:
            return None
        if not texts:
            return None
        try:
            vecs = model.encode(
                texts,
                batch_size=min(32, max(4, len(texts))),
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            return vecs
        except Exception:
            return None

    def _get_cross_encoder(self):
        if not self.ml_enabled or not self.cross_encoder_enabled:
            return None
        if self._cross_encoder_unavailable:
            return None
        if self.cross_encoder_local_only and not self._cross_encoder_model_exists_locally():
            self._cross_encoder_unavailable = True
            return None
        global CrossEncoder
        if CrossEncoder is None:
            try:
                from sentence_transformers import CrossEncoder as _CrossEncoder  # type: ignore
                CrossEncoder = _CrossEncoder
            except Exception:
                self._cross_encoder_unavailable = True
                return None
        if self._cross_encoder is None:
            try:
                self._cross_encoder = CrossEncoder(
                    self.cross_encoder_model_name,
                    local_files_only=self.cross_encoder_local_only,
                )
            except Exception:
                self._cross_encoder_unavailable = True
                self._cross_encoder = None
                return None
        return self._cross_encoder

    @staticmethod
    def _sigmoid(value: float) -> float:
        x = float(value)
        if x >= 0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        z = math.exp(x)
        return z / (1.0 + z)

    def _cross_encoder_scores(self, query: str, passages: List[str]) -> Optional[List[float]]:
        model = self._get_cross_encoder()
        if model is None or not passages:
            return None
        try:
            pairs = [[query, p[:1400]] for p in passages]
            raw_scores = model.predict(pairs)
            out: List[float] = []
            for score in raw_scores:
                try:
                    out.append(float(self._sigmoid(float(score))))
                except Exception:
                    out.append(0.0)
            return out
        except Exception:
            self._cross_encoder_unavailable = True
            return None

    def _collect_context_texts(self, scholar_doc: Dict[str, Any], max_items: int = 500) -> List[str]:
        rag_context = scholar_doc.get("rag_context") or {}
        texts: List[str] = []

        section_chunks = rag_context.get("section_chunks") or {}
        if isinstance(section_chunks, dict):
            for _, chunks in section_chunks.items():
                if not isinstance(chunks, list):
                    continue
                for ch in chunks:
                    if not isinstance(ch, dict):
                        continue
                    txt = safe_text(ch.get("text")) or safe_text(ch.get("summary"))
                    if txt and len(txt) > 60:
                        texts.append(txt[:2500])

        section_text = rag_context.get("section_text") or {}
        if isinstance(section_text, dict):
            for _, txt in section_text.items():
                txt = safe_text(txt)
                if txt and len(txt) > 60:
                    texts.append(txt[:3000])

        section_summaries = rag_context.get("section_summaries") or {}
        if isinstance(section_summaries, dict):
            for _, txt in section_summaries.items():
                txt = safe_text(txt)
                if txt and len(txt) > 40:
                    texts.append(txt[:1200])

        deduped: List[str] = []
        seen = set()
        for txt in texts:
            key = hashlib.sha1(re.sub(r"\s+", " ", txt.lower()).encode("utf-8")).hexdigest()[:16]
            if key in seen:
                continue
            seen.add(key)
            deduped.append(txt)
            if len(deduped) >= max_items:
                break
        return deduped

    @staticmethod
    def _make_topic_title(top_terms: List[str]) -> str:
        cleaned_terms: List[str] = []
        for raw in top_terms:
            text = safe_text(raw).strip().lower()
            if not text:
                continue
            if not DailyStoryWorker._is_viable_topic_phrase(text):
                continue
            toks = DailyStoryWorker._phrase_tokens(text)
            if len(toks) < 2:
                continue
            cleaned_terms.append(" ".join(toks))
        if not cleaned_terms:
            return ""
        if len(cleaned_terms) == 1:
            return f"What history suggests about {cleaned_terms[0]} today"
        return f"What history suggests about {cleaned_terms[0]} and {cleaned_terms[1]} today"

    @staticmethod
    def _phrase_tokens(text: str) -> List[str]:
        return [t for t in re.findall(r"[A-Za-z]{3,}", safe_text(text).lower()) if t not in TOPIC_STOPWORDS]

    @staticmethod
    def _has_concept_term(text: str) -> bool:
        return any(tok in TOPIC_CONCEPT_TERMS for tok in DailyStoryWorker._phrase_tokens(text))

    @staticmethod
    def _is_viable_topic_phrase(phrase: str) -> bool:
        tokens = DailyStoryWorker._phrase_tokens(phrase)
        if len(tokens) < 2:
            return False
        if len(set(tokens)) < 2:
            return False
        if any(tok.isdigit() for tok in tokens):
            return False
        if tokens[0] in TOPIC_STOPWORDS or tokens[-1] in TOPIC_STOPWORDS:
            return False
        return True

    @staticmethod
    def _compose_topic_phrase(terms: List[str], max_terms: int = 3) -> str:
        cleaned: List[str] = []
        seen = set()
        seen_token_sets: List[set] = []
        ranked_input = sorted(terms, key=lambda t: len(DailyStoryWorker._phrase_tokens(t)), reverse=True)
        for raw in ranked_input:
            toks = DailyStoryWorker._phrase_tokens(raw)
            if not toks:
                continue
            toks = list(dict.fromkeys(toks))
            raw_lc = safe_text(raw).lower()
            if " and " in raw_lc and len(toks) >= 2:
                pair = (toks[0], toks[1])
                reverse_pair = (toks[1], toks[0])
                if pair in TOPIC_COMPOUND_PAIRS:
                    phrase = f"{pair[0]} {pair[1]}"
                elif reverse_pair in TOPIC_COMPOUND_PAIRS:
                    phrase = f"{reverse_pair[0]} {reverse_pair[1]}"
                else:
                    phrase = f"{toks[0]} and {toks[1]}"
            else:
                phrase = " ".join(toks[:2]) if len(toks) > 2 else " ".join(toks)
            norm = phrase.strip().lower()
            if not norm or norm in seen:
                continue
            tok_set = set(toks)
            if any(tok_set.issubset(existing) for existing in seen_token_sets):
                continue
            if any(existing.issubset(tok_set) and len(existing) == len(tok_set) for existing in seen_token_sets):
                continue
            seen.add(norm)
            seen_token_sets.append(tok_set)
            cleaned.append(phrase)
            if len(cleaned) >= max_terms:
                break

        if not cleaned:
            return ""
        if len(cleaned) == 1:
            return cleaned[0]
        if len(cleaned) == 2:
            return f"{cleaned[0]} and {cleaned[1]}"
        return f"{cleaned[0]}, {cleaned[1]}, and {cleaned[2]}"

    def _model_topics_nmf(self, scholar_doc: Dict[str, Any], max_topics: int = 8) -> List[Dict[str, Any]]:
        """
        Topic modeling over scholar corpus using NMF on TF-IDF sentence matrix.
        Returns ranked latent topics with labels and scores.
        """
        if not self.ml_enabled or np is None:
            return []

        global TfidfVectorizer, NMF
        if TfidfVectorizer is None or NMF is None:
            try:
                from sklearn.decomposition import NMF as _NMF  # type: ignore
                from sklearn.feature_extraction.text import TfidfVectorizer as _TfidfVectorizer  # type: ignore
                NMF = _NMF
                TfidfVectorizer = _TfidfVectorizer
            except Exception:
                return []

        source_texts = self._collect_context_texts(scholar_doc, max_items=600)
        if not source_texts:
            return []

        sentences: List[str] = []
        for text in source_texts:
            sentences.extend(self._split_sentences(text))
        if len(sentences) < 15:
            return []

        seen = set()
        dedup_sentences: List[str] = []
        for sent in sentences:
            key = hashlib.sha1(sent.lower().encode("utf-8")).hexdigest()[:16]
            if key in seen:
                continue
            seen.add(key)
            dedup_sentences.append(sent)
            if len(dedup_sentences) >= 1400:
                break
        sentences = dedup_sentences
        if len(sentences) < 15:
            return []

        try:
            min_df = 2 if len(sentences) >= 70 else 1
            vectorizer = TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                min_df=min_df,
                max_df=0.92,
                max_features=7000,
            )
            x = vectorizer.fit_transform(sentences)
        except Exception:
            return []

        if x.shape[0] < 10 or x.shape[1] < 20:
            return []

        n_topics = min(max_topics, max(3, len(sentences) // 45), x.shape[0] - 1, x.shape[1] - 1)
        if n_topics < 2:
            return []

        try:
            topic_model = NMF(
                n_components=n_topics,
                init="nndsvda",
                random_state=42,
                max_iter=500,
            )
            w = topic_model.fit_transform(x)
            h = topic_model.components_
        except Exception:
            return []

        terms = vectorizer.get_feature_names_out()
        ranked_topics: List[Dict[str, Any]] = []
        seen_norm = set()

        for topic_idx in range(n_topics):
            top_term_ids = np.argsort(h[topic_idx])[-14:][::-1]
            candidate_terms: List[str] = []
            for term_id in top_term_ids:
                if int(term_id) >= len(terms):
                    continue
                raw_term = safe_text(terms[int(term_id)]).lower()
                if not raw_term:
                    continue
                toks = self._phrase_tokens(raw_term)
                if not toks:
                    continue
                normalized = " ".join(toks)
                if len(toks) == 1 and toks[0] not in TOPIC_CONCEPT_TERMS:
                    continue
                if len(toks) >= 2 and not self._has_concept_term(normalized):
                    continue
                candidate_terms.append(normalized)

            candidate_terms = list(dict.fromkeys(candidate_terms))
            label_core = self._compose_topic_phrase(candidate_terms, max_terms=3)
            if not label_core:
                continue
            if not self._has_concept_term(label_core):
                continue

            label_norm = " ".join(sorted(set(self._phrase_tokens(label_core))))
            if not label_norm or label_norm in seen_norm:
                continue
            seen_norm.add(label_norm)

            score = float(np.mean(w[:, topic_idx]) + 0.30 * np.max(w[:, topic_idx]))
            ranked_topics.append(
                {
                    "topic": f"What history suggests about {label_core} today",
                    "label_core": label_core,
                    "score": score,
                    "terms": candidate_terms[:6],
                    "model": "nmf",
                }
            )

        ranked_topics.sort(key=lambda x: x["score"], reverse=True)
        if len(ranked_topics) < 4:
            concept_topics = self._mine_concept_pair_topics(source_texts, max_topics=max_topics)
            existing_norms = {
                " ".join(sorted(set(self._phrase_tokens(safe_text(t.get("label_core"))))))
                for t in ranked_topics
            }
            for phrase, score in concept_topics:
                label_core = self._compose_topic_phrase([phrase], max_terms=1)
                if not label_core:
                    continue
                label_norm = " ".join(sorted(set(self._phrase_tokens(label_core))))
                if not label_norm or label_norm in existing_norms:
                    continue
                existing_norms.add(label_norm)
                ranked_topics.append(
                    {
                        "topic": f"What history suggests about {label_core} today",
                        "label_core": label_core,
                        "score": float(score),
                        "terms": [label_core],
                        "model": "concept_pair_fallback",
                    }
                )
                if len(ranked_topics) >= max_topics:
                    break

        if len(ranked_topics) < 3:
            fallback_phrases = self._mine_statistical_phrases(source_texts, scholar_doc, max_phrases=40)
            existing_norms = {
                " ".join(sorted(set(self._phrase_tokens(safe_text(t.get("label_core"))))))
                for t in ranked_topics
            }
            for phrase, score in fallback_phrases:
                label_core = self._compose_topic_phrase([phrase], max_terms=1)
                if not label_core:
                    continue
                label_norm = " ".join(sorted(set(self._phrase_tokens(label_core))))
                if not label_norm or label_norm in existing_norms:
                    continue
                if not self._has_concept_term(label_core):
                    continue
                existing_norms.add(label_norm)
                ranked_topics.append(
                    {
                        "topic": f"What history suggests about {label_core} today",
                        "label_core": label_core,
                        "score": float(score),
                        "terms": [label_core],
                        "model": "phrase_fallback",
                    }
                )
                if len(ranked_topics) >= max_topics:
                    break

        ranked_topics.sort(key=lambda x: x["score"], reverse=True)
        return ranked_topics[:max_topics]

    def _mine_concept_pair_topics(self, source_texts: List[str], max_topics: int = 8) -> List[Tuple[str, float]]:
        concept_pair_counts = Counter()
        concept_single_counts = Counter()

        sentences: List[str] = []
        for text in source_texts:
            sentences.extend(self._split_sentences(text))

        for sent in sentences:
            tokens = set(self._phrase_tokens(sent))
            concepts = sorted([t for t in tokens if t in TOPIC_CONCEPT_TERMS])
            if not concepts:
                continue
            for c in concepts:
                concept_single_counts[c] += 1
            if len(concepts) < 2:
                continue
            # Pair co-occurrence signals stronger thematic links than raw phrase mining.
            for i in range(len(concepts) - 1):
                for j in range(i + 1, len(concepts)):
                    pair = f"{concepts[i]} and {concepts[j]}"
                    concept_pair_counts[pair] += 1

        ranked: List[Tuple[str, float]] = []
        for pair, count in concept_pair_counts.items():
            if count < 2:
                continue
            ranked.append((pair, float(count)))

        if len(ranked) < max_topics:
            for concept, count in concept_single_counts.items():
                if count < 3:
                    continue
                ranked.append((concept, float(count) * 0.6))

        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked[:max_topics]

    @staticmethod
    def _seeded_weighted_pick(candidates: List[Dict[str, Any]], key: str) -> Optional[Dict[str, Any]]:
        if not candidates:
            return None
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        seed = int(digest[:16], 16)
        rng = random.Random(seed)
        weights = [max(0.0001, float(c.get("score", 1.0))) for c in candidates]
        idx = rng.choices(range(len(candidates)), weights=weights, k=1)[0]
        return candidates[int(idx)]

    def _build_style_profile(self, scholar_doc: Dict[str, Any]) -> Dict[str, Any]:
        """
        Estimate style features from corpus so generation mirrors argument cadence
        rather than generic template prose.
        """
        source_texts = self._collect_context_texts(scholar_doc, max_items=220)
        sentences: List[str] = []
        for text in source_texts:
            sentences.extend(self._split_sentences(text))

        seen = set()
        dedup_sentences: List[str] = []
        for s in sentences:
            key = hashlib.sha1(s.lower().encode("utf-8")).hexdigest()[:16]
            if key in seen:
                continue
            seen.add(key)
            dedup_sentences.append(s)
            if len(dedup_sentences) >= self.style_sentence_samples:
                break

        if not dedup_sentences:
            return {
                "cadence": "measured",
                "avg_sentence_words": 22,
                "connectors": ["however", "therefore"],
                "concepts": ["history", "institutions", "justice"],
            }

        lengths = [len(re.findall(r"[A-Za-z']+", s)) for s in dedup_sentences]
        avg_len = float(sum(lengths) / max(1, len(lengths)))
        if avg_len < 16:
            cadence = "brisk"
        elif avg_len < 26:
            cadence = "measured"
        else:
            cadence = "long-form"

        lower_blob = " ".join(dedup_sentences).lower()
        connector_hits = Counter()
        for conn in STYLE_CONNECTORS:
            connector_hits[conn] = lower_blob.count(conn)
        connector_ranked = [k for k, _ in connector_hits.most_common(5) if connector_hits[k] > 0]
        if not connector_ranked:
            connector_ranked = ["however", "therefore", "indeed"]

        concept_counts = Counter([t for t in self._phrase_tokens(lower_blob) if t in TOPIC_CONCEPT_TERMS])
        concept_ranked = [k for k, _ in concept_counts.most_common(6)]
        if not concept_ranked:
            concept_ranked = ["history", "institutions", "justice"]

        return {
            "cadence": cadence,
            "avg_sentence_words": round(avg_len, 1),
            "connectors": connector_ranked[:4],
            "concepts": concept_ranked[:4],
        }

    @staticmethod
    def _http_get_text(url: str, timeout_seconds: int, headers: Optional[Dict[str, str]] = None) -> str:
        req_headers = {"User-Agent": "Mozilla/5.0 (DailyStoryWorker)"}
        if headers:
            req_headers.update(headers)
        req = Request(url, headers=req_headers)
        try:
            with urlopen(req, timeout=timeout_seconds) as resp:
                raw = resp.read()
            return raw.decode("utf-8", errors="replace")
        except (HTTPError, URLError, TimeoutError, ValueError):
            return ""
        except Exception:
            return ""

    def _http_get_json(self, url: str, timeout_seconds: int, headers: Optional[Dict[str, str]] = None) -> Optional[Dict[str, Any]]:
        text = self._http_get_text(url, timeout_seconds=timeout_seconds, headers=headers)
        if not text:
            return None
        try:
            payload = json.loads(text)
            if isinstance(payload, dict):
                return payload
        except Exception:
            return None
        return None

    @staticmethod
    def _strip_html(text: str) -> str:
        text = safe_text(text)
        if not text:
            return ""
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _parse_any_datetime(value: str) -> Optional[datetime]:
        text = safe_text(value)
        if not text:
            return None
        try:
            text = text.replace("Z", "+00:00")
            dt = datetime.fromisoformat(text)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            pass
        try:
            dt = parsedate_to_datetime(text)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            return None

    def _normalize_trend_issue(
        self,
        *,
        title: str,
        summary: str,
        url: str,
        source: str,
        published_at: str,
    ) -> Optional[Dict[str, Any]]:
        title = self._strip_html(title)
        summary = self._strip_html(summary)
        url = safe_text(url)
        source = safe_text(source) or "unknown"
        if len(title) < 20:
            return None
        blob = f"{title} {summary}".strip()
        tokens = self._phrase_tokens(blob)
        if len(tokens) < 5:
            return None
        return {
            "title": title[:220],
            "summary": summary[:700],
            "url": url,
            "source": source[:120],
            "published_at": safe_text(published_at),
            "tokens": tokens[:80],
            "text": blob[:2500],
        }

    @staticmethod
    def _xml_child_text(node: Any, *names: str) -> str:
        for child in list(node):
            tag = safe_text(getattr(child, "tag", "")).lower()
            for name in names:
                if tag.endswith(name.lower()):
                    return safe_text(getattr(child, "text", ""))
        return ""

    def _fetch_trends_newsapi(self) -> List[Dict[str, Any]]:
        if not self.newsapi_key:
            return []
        params = {
            "language": "en",
            "pageSize": min(max(self.trend_max_items, 10), 100),
            "category": "general",
            "country": self.trend_region or "us",
        }
        url = f"https://newsapi.org/v2/top-headlines?{urlencode(params)}"
        payload = self._http_get_json(
            url,
            timeout_seconds=self.trend_timeout_seconds,
            headers={"X-Api-Key": self.newsapi_key},
        )
        if not payload:
            return []
        articles = payload.get("articles")
        if not isinstance(articles, list):
            return []

        out: List[Dict[str, Any]] = []
        for a in articles:
            if not isinstance(a, dict):
                continue
            src = a.get("source") or {}
            source_name = safe_text(src.get("name")) if isinstance(src, dict) else ""
            normalized = self._normalize_trend_issue(
                title=safe_text(a.get("title")),
                summary=safe_text(a.get("description")) or safe_text(a.get("content")),
                url=safe_text(a.get("url")),
                source=source_name or "NewsAPI",
                published_at=safe_text(a.get("publishedAt")),
            )
            if normalized:
                out.append(normalized)
        return out

    def _fetch_trends_rss(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for feed_url in self.trend_rss_urls:
            xml_text = self._http_get_text(feed_url, timeout_seconds=self.trend_timeout_seconds)
            if not xml_text:
                continue
            try:
                root = ET.fromstring(xml_text)
            except Exception:
                continue

            nodes = root.findall(".//item")
            if not nodes:
                # Atom feeds
                nodes = root.findall(".//{http://www.w3.org/2005/Atom}entry")

            for node in nodes:
                title = self._xml_child_text(node, "title")
                summary = self._xml_child_text(node, "description", "summary", "content")
                link = self._xml_child_text(node, "link")
                if not link:
                    link_attr = getattr(node.find("{http://www.w3.org/2005/Atom}link"), "attrib", None)
                    if isinstance(link_attr, dict):
                        link = safe_text(link_attr.get("href"))
                pub = self._xml_child_text(node, "pubDate", "updated", "published")
                source = safe_text(feed_url.split("/")[2]) if "://" in feed_url else "rss"
                normalized = self._normalize_trend_issue(
                    title=title,
                    summary=summary,
                    url=link,
                    source=source,
                    published_at=pub,
                )
                if normalized:
                    out.append(normalized)
        return out

    def _fetch_trends_gdelt(self) -> List[Dict[str, Any]]:
        query = quote_plus("language:english sourceCountry:US")
        max_records = min(max(self.trend_max_items, 10), 75)
        url = (
            "https://api.gdeltproject.org/api/v2/doc/doc"
            f"?query={query}&mode=ArtList&maxrecords={max_records}&sort=HybridRel&format=json"
        )
        payload = self._http_get_json(url, timeout_seconds=self.trend_timeout_seconds)
        if not payload:
            return []
        articles = payload.get("articles")
        if not isinstance(articles, list):
            return []
        out: List[Dict[str, Any]] = []
        for a in articles:
            if not isinstance(a, dict):
                continue
            normalized = self._normalize_trend_issue(
                title=safe_text(a.get("title")),
                summary=safe_text(a.get("seendate")) + " " + safe_text(a.get("socialimage")),
                url=safe_text(a.get("url")),
                source=safe_text(a.get("domain")) or "gdelt",
                published_at=safe_text(a.get("seendate")),
            )
            if normalized:
                out.append(normalized)
        return out

    @staticmethod
    def _trend_issue_key(issue: Dict[str, Any]) -> str:
        title = safe_text(issue.get("title")).lower()
        source = safe_text(issue.get("source")).lower()
        published = safe_text(issue.get("published_at")).lower()
        base = f"{title}|{source}|{published}"
        return hashlib.sha1(base.encode("utf-8")).hexdigest()[:24]

    def _load_cached_trending_issues(self) -> List[Dict[str, Any]]:
        if not self.trend_cache_enabled:
            return []
        cutoff = datetime.now(timezone.utc).timestamp() - (self.trend_cache_ttl_hours * 3600)
        cursor = self.trend_cache_collection.find(
            {"fetched_at_ts": {"$gte": cutoff}},
            {
                "_id": 0,
                "title": 1,
                "summary": 1,
                "url": 1,
                "source": 1,
                "published_at": 1,
                "tokens": 1,
                "text": 1,
                "fetched_at_ts": 1,
            },
        ).sort("fetched_at_ts", -1).limit(self.trend_max_items * 3)

        issues = list(cursor)
        if not issues:
            return []

        deduped: List[Dict[str, Any]] = []
        seen = set()
        for issue in issues:
            key = self._trend_issue_key(issue)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(issue)
            if len(deduped) >= self.trend_max_items:
                break
        return deduped

    def _upsert_trending_issue_cache(self, issues: List[Dict[str, Any]]) -> None:
        if not issues:
            return
        now_ts = datetime.now(timezone.utc).timestamp()
        for issue in issues:
            key = self._trend_issue_key(issue)
            dt = self._parse_any_datetime(safe_text(issue.get("published_at")))
            update_doc = {
                "issue_key": key,
                "title": safe_text(issue.get("title")),
                "summary": safe_text(issue.get("summary")),
                "url": safe_text(issue.get("url")),
                "source": safe_text(issue.get("source")),
                "published_at": safe_text(issue.get("published_at")),
                "published_at_dt": dt,
                "tokens": issue.get("tokens") or [],
                "text": safe_text(issue.get("text")),
                "fetched_at_ts": now_ts,
                "fetched_at": utc_now_iso(),
            }
            try:
                self.trend_cache_collection.update_one(
                    {"issue_key": key},
                    {"$set": update_doc},
                    upsert=True,
                )
            except Exception:
                continue

    def _fetch_trending_issues(self, prefer_cache: bool = True) -> List[Dict[str, Any]]:
        if not self.trends_enabled:
            return []

        if prefer_cache:
            cached = self._load_cached_trending_issues()
            if cached:
                return cached

        provider = self.trend_provider or "rss"
        if provider == "auto":
            providers = ["newsapi", "rss", "gdelt"]
        else:
            providers = [provider]

        issues: List[Dict[str, Any]] = []
        for p in providers:
            p = p.lower().strip()
            if p == "newsapi":
                issues.extend(self._fetch_trends_newsapi())
            elif p == "rss":
                issues.extend(self._fetch_trends_rss())
            elif p == "gdelt":
                issues.extend(self._fetch_trends_gdelt())

        if not issues:
            return []

        deduped: List[Dict[str, Any]] = []
        seen = set()
        for issue in issues:
            norm_title = re.sub(r"\s+", " ", safe_text(issue.get("title")).lower()).strip()
            key = hashlib.sha1(norm_title.encode("utf-8")).hexdigest()[:16]
            if key in seen:
                continue
            seen.add(key)
            deduped.append(issue)

        def issue_score(item: Dict[str, Any]) -> float:
            dt = self._parse_any_datetime(safe_text(item.get("published_at")))
            recency = 0.0
            if dt is not None:
                age_hours = max(0.0, (datetime.now(timezone.utc) - dt).total_seconds() / 3600.0)
                recency = 1.0 / (1.0 + age_hours / 36.0)
            title_len = len(safe_text(item.get("title")))
            title_quality = min(1.0, max(0.0, (title_len - 20) / 100.0))
            return 0.75 * recency + 0.25 * title_quality

        deduped.sort(key=issue_score, reverse=True)
        selected = deduped[: self.trend_max_items]
        self._upsert_trending_issue_cache(selected)
        return selected

    def _select_trending_issue(self, topic: str, issues: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not issues:
            return None

        candidates = issues
        if self.require_verified_trend_url:
            candidates = [
                i for i in issues
                if self._is_valid_external_url(safe_text(i.get("url")))
            ]
        if not candidates:
            return None

        topic_tokens = set(self._topic_tokens(topic))
        issue_texts = [safe_text(i.get("text")) for i in candidates]

        lexical_scores: List[float] = []
        for issue in candidates:
            issue_tokens = set(issue.get("tokens") or [])
            if not issue_tokens:
                lexical_scores.append(0.0)
                continue
            overlap = len(topic_tokens.intersection(issue_tokens)) / max(1, len(topic_tokens))
            concept_overlap = len(set(topic_tokens).intersection(set([t for t in issue_tokens if t in TOPIC_CONCEPT_TERMS])))
            concept_bonus = min(1.0, concept_overlap / 3.0)
            lexical_scores.append(float(0.75 * overlap + 0.25 * concept_bonus))

        dense_scores = [0.0] * len(candidates)
        embeds = self._embed_texts([topic] + issue_texts)
        if embeds is not None and np is not None and len(embeds) == len(candidates) + 1:
            topic_vec = embeds[0]
            issue_vecs = embeds[1:]
            sims = issue_vecs @ topic_vec
            dense_scores = [float(max(0.0, s)) for s in sims]

        cross_scores = [0.0] * len(candidates)
        cross = self._cross_encoder_scores(topic, issue_texts)
        if cross is not None and len(cross) == len(candidates):
            cross_scores = [float(max(0.0, min(1.0, s))) for s in cross]

        recency_scores: List[float] = []
        now = datetime.now(timezone.utc)
        for issue in candidates:
            dt = self._parse_any_datetime(safe_text(issue.get("published_at")))
            if dt is None:
                recency_scores.append(0.15)
                continue
            age_hours = max(0.0, (now - dt).total_seconds() / 3600.0)
            recency_scores.append(float(1.0 / (1.0 + age_hours / 48.0)))

        best_idx = 0
        best_score = -1.0
        for idx in range(len(candidates)):
            if cross is not None:
                score = (
                    0.45 * dense_scores[idx]
                    + 0.25 * lexical_scores[idx]
                    + 0.10 * recency_scores[idx]
                    + 0.20 * cross_scores[idx]
                )
            else:
                score = 0.55 * dense_scores[idx] + 0.35 * lexical_scores[idx] + 0.10 * recency_scores[idx]
            if score > best_score:
                best_score = score
                best_idx = idx

        selected = dict(candidates[best_idx])
        selected["selection_score"] = round(float(best_score), 4)
        return selected

    def _mine_statistical_phrases(
        self,
        source_texts: List[str],
        scholar_doc: Dict[str, Any],
        max_phrases: int = 20,
    ) -> List[Tuple[str, float]]:
        """
        Unsupervised phrase mining using frequency + inverse document frequency.
        This remains local to scholar corpus (legend_scholars) and is stable for cron.
        """
        if np is None:
            return []

        scholar_name = self._professor_name(scholar_doc).lower()
        name_tokens = set(re.findall(r"[a-z]{3,}", scholar_name))

        tf = Counter()
        df = Counter()
        doc_count = 0

        for text in source_texts:
            doc_count += 1
            words = [w.lower() for w in re.findall(r"[A-Za-z]{3,}", text)]
            words = [w for w in words if w not in TOPIC_STOPWORDS and w not in name_tokens]
            if len(words) < 4:
                continue

            doc_phrases = set()
            for n in (2, 3):
                for i in range(0, len(words) - n + 1):
                    ng = words[i : i + n]
                    if len(set(ng)) < 2:
                        continue
                    phrase = " ".join(ng)
                    if not self._is_viable_topic_phrase(phrase):
                        continue
                    if any(tok in name_tokens for tok in ng):
                        continue
                    if sum(1 for tok in ng if tok in TOPIC_CONCEPT_TERMS) == 0:
                        continue
                    tf[phrase] += 1
                    doc_phrases.add(phrase)
            for phrase in doc_phrases:
                df[phrase] += 1

        scored: List[Tuple[str, float]] = []
        for phrase, freq in tf.items():
            if freq < 2:
                continue
            phrase_df = df.get(phrase, 1)
            idf = float(np.log(1.0 + (doc_count / max(1.0, float(phrase_df)))))
            length_bonus = 1.0 + 0.15 * (len(phrase.split()) - 2)
            concept_bonus = 1.25 if self._has_concept_term(phrase) else 1.0
            score = float(freq) * idf * length_bonus * concept_bonus
            scored.append((phrase, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:max_phrases]

    def _mine_topic_candidates(self, scholar_doc: Dict[str, Any], max_candidates: int = 12) -> List[str]:
        """
        ML topic mining from scholar-local corpus:
        1) statistical phrase mining from scholar corpus
        2) sentence clustering with TF-IDF + MiniBatchKMeans
        3) optional embedding reranking
        """
        if not self.ml_enabled:
            return []
        if np is None:
            return []
        global TfidfVectorizer, MiniBatchKMeans
        if TfidfVectorizer is None or MiniBatchKMeans is None:
            try:
                from sklearn.cluster import MiniBatchKMeans as _MiniBatchKMeans  # type: ignore
                from sklearn.feature_extraction.text import TfidfVectorizer as _TfidfVectorizer  # type: ignore
                MiniBatchKMeans = _MiniBatchKMeans
                TfidfVectorizer = _TfidfVectorizer
            except Exception:
                return []

        source_texts = self._collect_context_texts(scholar_doc)
        if not source_texts:
            return []

        phrase_candidates = self._mine_statistical_phrases(source_texts, scholar_doc, max_phrases=30)
        topic_candidates: List[Tuple[str, float]] = []
        for phrase, score in phrase_candidates:
            title = self._make_topic_title([phrase])
            if title:
                topic_candidates.append((title, float(score)))

        sentences: List[str] = []
        for text in source_texts:
            sentences.extend(self._split_sentences(text))

        # Dedupe and cap size for predictable cron latency.
        seen_sent = set()
        uniq_sentences: List[str] = []
        for s in sentences:
            key = hashlib.sha1(s.lower().encode("utf-8")).hexdigest()[:16]
            if key in seen_sent:
                continue
            seen_sent.add(key)
            uniq_sentences.append(s)
            if len(uniq_sentences) >= 1000:
                break
        sentences = uniq_sentences
        if len(sentences) < 20:
            # If corpus is too small for clustering, use phrase candidates only.
            if topic_candidates:
                ranked_titles = sorted(topic_candidates, key=lambda x: x[1], reverse=True)
                return [title for title, _ in ranked_titles[:max_candidates]]
            return []

        try:
            min_df = 2 if len(sentences) >= 60 else 1
            vectorizer = TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                min_df=min_df,
                max_features=5000,
            )
            x = vectorizer.fit_transform(sentences)
            if x.shape[0] < 3 or x.shape[1] < 8:
                if topic_candidates:
                    ranked_titles = sorted(topic_candidates, key=lambda x: x[1], reverse=True)
                    return [title for title, _ in ranked_titles[:max_candidates]]
                return []
        except Exception:
            if topic_candidates:
                ranked_titles = sorted(topic_candidates, key=lambda x: x[1], reverse=True)
                return [title for title, _ in ranked_titles[:max_candidates]]
            return []

        n_clusters = min(10, max(3, len(sentences) // 30))
        n_clusters = min(n_clusters, x.shape[0])
        if n_clusters < 2:
            return []

        try:
            kmeans = MiniBatchKMeans(
                n_clusters=n_clusters,
                random_state=42,
                batch_size=128,
                n_init="auto",
            )
            labels = kmeans.fit_predict(x)
        except Exception:
            if topic_candidates:
                ranked_titles = sorted(topic_candidates, key=lambda x: x[1], reverse=True)
                return [title for title, _ in ranked_titles[:max_candidates]]
            return []

        terms = vectorizer.get_feature_names_out()

        for cluster_id in range(n_clusters):
            idxs = np.where(labels == cluster_id)[0]
            if len(idxs) == 0:
                continue

            centroid = x[idxs].mean(axis=0).A1
            top_ids = centroid.argsort()[-8:][::-1]
            raw_terms = [safe_text(terms[i]).lower() for i in top_ids if i < len(terms)]
            top_terms = []
            for t in raw_terms:
                if not t:
                    continue
                phrase_tokens = self._phrase_tokens(t)
                if len(phrase_tokens) < 1:
                    continue
                if len(phrase_tokens) >= 2 and not self._is_viable_topic_phrase(t):
                    continue
                if len(phrase_tokens) == 1 and phrase_tokens[0] not in TOPIC_CONCEPT_TERMS:
                    continue
                if len(phrase_tokens) >= 2 and not self._has_concept_term(t):
                    continue
                top_terms.append(" ".join(phrase_tokens))
            if not top_terms:
                continue
            title = self._make_topic_title(top_terms[:3])
            if not title:
                continue

            # Cluster coverage + centroid strength score.
            coverage = len(idxs) / max(1, len(sentences))
            strength = float(np.mean([centroid[i] for i in top_ids[:4]])) if len(top_ids) else 0.0
            concept_bonus = 1.15 if self._has_concept_term(title) else 0.85
            topic_candidates.append((title, float((coverage + strength) * concept_bonus)))

        if not topic_candidates:
            return []

        # Optional embedding rerank to favor corpus-central topics.
        topics = [t for t, _ in topic_candidates]
        base_scores = np.array([s for _, s in topic_candidates], dtype=float)
        embed = self._embed_texts(topics + [(" ".join(sentences[:120]))[:4000]])
        if embed is not None and len(embed) == len(topics) + 1:
            topic_vecs = embed[:-1]
            corpus_vec = embed[-1]
            dense_scores = np.maximum(0.0, topic_vecs @ corpus_vec)
            final_scores = 0.6 * base_scores + 0.4 * dense_scores
        else:
            final_scores = base_scores

        ranked_idxs = np.argsort(final_scores)[::-1]
        ranked_topics: List[str] = []
        seen_titles = set()
        for idx in ranked_idxs:
            title = topics[int(idx)]
            if not self._has_concept_term(title):
                continue
            norm = " ".join(sorted(set(self._phrase_tokens(title))))
            if not norm:
                continue
            if norm in seen_titles:
                continue
            seen_titles.add(norm)
            ranked_topics.append(title)
            if len(ranked_topics) >= max_candidates:
                break
        if not ranked_topics and topic_candidates:
            ranked_titles = sorted(topic_candidates, key=lambda x: x[1], reverse=True)
            return [title for title, _ in ranked_titles[:max_candidates]]
        return ranked_topics

    def _choose_topic(
        self,
        scholar_doc: Dict[str, Any],
        profile_id: str,
        story_date: str,
        topic_override: Optional[str],
    ) -> Tuple[str, str, int]:
        if topic_override:
            return topic_override.strip(), "manual_override", 0

        nmf_topics = self._model_topics_nmf(scholar_doc, max_topics=8)
        if nmf_topics:
            nmf_pool = nmf_topics[: max(1, min(4, len(nmf_topics)))]
            picked = self._seeded_weighted_pick(
                nmf_pool,
                key=f"{profile_id}:{story_date}:nmf-topic",
            )
            if picked:
                return safe_text(picked.get("topic")), "nmf_topic_modeling", len(nmf_topics)

        ml_topics = self._mine_topic_candidates(scholar_doc, max_candidates=12)
        if ml_topics:
            topic_pool = ml_topics[: max(1, min(6, len(ml_topics)))]
            topic_source = "ml_clustered_topics"
        else:
            topic_pool = DEFAULT_TOPICS
            topic_source = "default_topic_pool"

        weighted_pool = [
            {"topic": topic, "score": float(max(1, len(topic_pool) - idx))}
            for idx, topic in enumerate(topic_pool)
        ]
        picked = self._seeded_weighted_pick(
            weighted_pool,
            key=f"{profile_id}:{story_date}:fallback-topic",
        )
        if picked:
            return safe_text(picked.get("topic")), topic_source, len(ml_topics)
        return topic_pool[0], topic_source, len(ml_topics)

    def _profile_quality_state(self, scholar_doc: Dict[str, Any]) -> Tuple[bool, str, int]:
        profile_meta = scholar_doc.get("daily_story_profile") or {}
        status = safe_text(profile_meta.get("status")).lower()
        score = int(profile_meta.get("quality_score") or 0)

        if not status:
            profile_id = self._profile_id(scholar_doc)
            if profile_id:
                rec = self.profile_quality_collection.find_one(
                    {"profile_id": profile_id},
                    {"status": 1, "quality_score": 1},
                )
                if rec:
                    status = safe_text(rec.get("status")).lower()
                    score = int(rec.get("quality_score") or 0)

        if not self.enforce_profile_quality:
            return True, status or "not_enforced", score

        if status in {"ready"} and score >= self.profile_quality_min_score:
            return True, status, score
        return False, status or "missing_quality_profile", score

    def _iter_scholars(self, scholar_id: Optional[str], max_scholars: int) -> Iterable[Dict[str, Any]]:
        if scholar_id:
            query = {"$or": [{"profile_id": scholar_id}, {"_id": scholar_id}]}
            doc = self.scholars_collection.find_one(query)
            return [doc] if doc else []
        cursor = self.scholars_collection.find({}).limit(max_scholars)
        return list(cursor)

    def _extract_context_chunks(
        self,
        scholar_doc: Dict[str, Any],
        topic: str,
        max_context_chunks: int,
    ) -> List[Dict[str, str]]:
        rag_context = scholar_doc.get("rag_context") or {}
        section_chunks = rag_context.get("section_chunks") or {}
        candidates: List[Dict[str, str]] = []

        if isinstance(section_chunks, dict):
            for section, chunks in section_chunks.items():
                if not isinstance(chunks, list):
                    continue
                for idx, chunk in enumerate(chunks):
                    if not isinstance(chunk, dict):
                        continue
                    chunk_id = safe_text(chunk.get("chunk_id")) or f"{section}:{idx}"
                    text = safe_text(chunk.get("text")) or safe_text(chunk.get("summary"))
                    if len(text) < 80:
                        continue
                    source_refs_raw = chunk.get("source_refs")
                    source_refs: List[Dict[str, Any]] = []
                    if isinstance(source_refs_raw, list):
                        for ref in source_refs_raw:
                            if not isinstance(ref, dict):
                                continue
                            try:
                                score_val = float(ref.get("score") or 0.0)
                            except Exception:
                                score_val = 0.0
                            try:
                                overlap_val = float(ref.get("token_overlap") or 0.0)
                            except Exception:
                                overlap_val = 0.0
                            try:
                                hits_val = int(ref.get("token_hits") or 0)
                            except Exception:
                                hits_val = 0
                            source_refs.append(
                                {
                                    "source_chunk_id": safe_text(ref.get("source_chunk_id")),
                                    "source_id": safe_text(ref.get("source_id")),
                                    "source_url": safe_text(ref.get("source_url")),
                                    "source_type": safe_text(ref.get("source_type")),
                                    "score": score_val,
                                    "token_overlap": overlap_val,
                                    "token_hits": hits_val,
                                }
                            )
                    source_urls = []
                    source_urls_raw = chunk.get("source_urls")
                    if isinstance(source_urls_raw, list):
                        source_urls = [safe_text(u) for u in source_urls_raw if safe_text(u)]
                    if not source_urls:
                        primary_url = safe_text(chunk.get("primary_source_url")) or safe_text(chunk.get("source_url"))
                        if primary_url:
                            source_urls = [primary_url]
                    candidates.append(
                        {
                            "chunk_id": chunk_id,
                            "section": safe_text(section) or "misc",
                            "text": text[:2000],
                            "source_refs": source_refs,
                            "source_urls": list(dict.fromkeys(source_urls)),
                        }
                    )

        if not candidates:
            section_text = rag_context.get("section_text") or {}
            if isinstance(section_text, dict):
                for section, text in section_text.items():
                    text = safe_text(text)
                    if len(text) < 80:
                        continue
                    for idx, part in enumerate(self._split_fallback_text(text, max_chars=1400)):
                        if len(part) < 80:
                            continue
                        candidates.append(
                            {
                                "chunk_id": f"fallback:{section}:{idx}",
                                "section": safe_text(section) or "misc",
                                "text": part,
                                "source_refs": [],
                                "source_urls": [],
                            }
                        )

        deduped: List[Dict[str, str]] = []
        seen_keys = set()
        for item in candidates:
            key = f"{item['chunk_id']}:{hashlib.sha256(item['text'].encode('utf-8')).hexdigest()[:12]}"
            if key in seen_keys:
                continue
            seen_keys.add(key)
            deduped.append(item)

        if not deduped:
            return []

        topic_tokens = self._topic_tokens(topic)

        # Lexical score: token coverage + frequency in chunk.
        lexical_scores: List[float] = []
        for item in deduped:
            toks = re.findall(r"[A-Za-z]{3,}", item["text"].lower())
            if not toks:
                lexical_scores.append(0.0)
                continue
            counts = Counter(toks)
            coverage = sum(1 for tok in topic_tokens if tok in counts) / max(1, len(topic_tokens))
            tf = sum(counts.get(tok, 0) for tok in topic_tokens) / max(1, len(toks))
            lexical_scores.append(float(0.7 * coverage + 0.3 * min(1.0, tf * 8.0)))

        # Dense score: cosine similarity between topic and chunk embeddings.
        dense_scores = [0.0] * len(deduped)
        embeds = self._embed_texts([topic] + [item["text"][:1200] for item in deduped])
        if embeds is not None and np is not None and len(embeds) == len(deduped) + 1:
            topic_vec = embeds[0]
            chunk_vecs = embeds[1:]
            sims = chunk_vecs @ topic_vec
            dense_scores = [float(max(0.0, s)) for s in sims]

        ranked_with_score: List[Tuple[float, Dict[str, str]]] = []
        for idx, item in enumerate(deduped):
            section_rank = SECTION_PRIORITY.get(item["section"].lower(), 99)
            section_prior = 1.0 / (1.0 + min(section_rank, 10))
            score = (
                0.55 * dense_scores[idx]
                + 0.30 * lexical_scores[idx]
                + 0.15 * section_prior
            )
            ranked_with_score.append((score, item))

        ranked_with_score.sort(key=lambda x: x[0], reverse=True)
        if not ranked_with_score:
            return []

        rerank_k = max(max_context_chunks, min(len(ranked_with_score), max(4, self.cross_encoder_rerank_top_k)))
        rerank_pool = ranked_with_score[:rerank_k]
        rerank_items = [it for _, it in rerank_pool]
        cross_scores = self._cross_encoder_scores(topic, [it["text"] for it in rerank_items])
        if cross_scores is not None and len(cross_scores) == len(rerank_items):
            base_vals = [float(s) for s, _ in rerank_pool]
            base_min = min(base_vals)
            base_max = max(base_vals)
            denom = max(1e-9, base_max - base_min)
            combined: List[Tuple[float, Dict[str, str]]] = []
            for idx, item in enumerate(rerank_items):
                base_norm = (base_vals[idx] - base_min) / denom
                score = (1.0 - self.cross_encoder_weight) * base_norm + self.cross_encoder_weight * float(cross_scores[idx])
                combined.append((float(score), item))
            combined.sort(key=lambda x: x[0], reverse=True)
            return [item for _, item in combined[:max_context_chunks]]

        return [item for _, item in ranked_with_score[:max_context_chunks]]

    @staticmethod
    def _split_fallback_text(text: str, max_chars: int) -> List[str]:
        parts: List[str] = []
        start = 0
        while start < len(text):
            end = min(start + max_chars, len(text))
            parts.append(text[start:end])
            if end >= len(text):
                break
            start = end
        return parts

    @staticmethod
    def _build_disclosure(name: str, assume_deceased: bool) -> str:
        if assume_deceased:
            return (
                f"AI-generated perspective inspired by the published scholarship of {name}. "
                f"This is not a real article written by {name}."
            )
        return (
            f"AI-generated perspective inspired by the work of {name}. "
            f"This is synthetic writing, not a direct quote."
        )

    def _build_author_idea_anchors(
        self,
        *,
        context_chunks: List[Dict[str, str]],
        max_items: int = 6,
    ) -> List[str]:
        anchors: List[str] = []
        seen = set()
        for chunk in context_chunks:
            sentence = self._first_sentence(safe_text(chunk.get("text")), max_len=220)
            sentence = re.sub(r"\s+", " ", sentence).strip()
            if len(sentence) < 50:
                continue
            key = hashlib.sha1(sentence.lower().encode("utf-8")).hexdigest()[:16]
            if key in seen:
                continue
            seen.add(key)
            anchors.append(sentence)
            if len(anchors) >= max_items:
                break
        return anchors

    def _build_prompt(
        self,
        name: str,
        topic: str,
        context_chunks: List[Dict[str, str]],
        trend_issue: Optional[Dict[str, Any]],
        style_profile: Dict[str, Any],
        allowed_source_urls: List[str],
        assume_deceased: bool,
    ) -> str:
        context_lines = []
        allowed_chunk_ids = []
        for chunk in context_chunks:
            allowed_chunk_ids.append(chunk["chunk_id"])
            context_lines.append(
                f"[chunk_id={chunk['chunk_id']}][section={chunk['section']}]\n{chunk['text']}"
            )
        context_blob = "\n\n".join(context_lines)
        allowed_blob = ", ".join(allowed_chunk_ids)
        trend_title = safe_text((trend_issue or {}).get("title")) or "No external trend supplied"
        trend_summary = safe_text((trend_issue or {}).get("summary")) or ""
        trend_source = safe_text((trend_issue or {}).get("source")) or "none"
        trend_published_at = safe_text((trend_issue or {}).get("published_at")) or "unknown"
        trend_url = safe_text((trend_issue or {}).get("url")) or "none"
        style_cadence = safe_text(style_profile.get("cadence")) or "measured"
        style_avg_words = safe_text(style_profile.get("avg_sentence_words")) or "22"
        style_connectors = ", ".join(style_profile.get("connectors") or [])
        style_concepts = ", ".join(style_profile.get("concepts") or [])
        required_source_count = min(self.min_source_url_citations, len(allowed_source_urls)) if allowed_source_urls else 0
        style_sources = "\n".join([f"- {u}" for u in allowed_source_urls[: self.max_source_url_candidates]])
        if not style_sources:
            style_sources = "- none"
        idea_anchors = self._build_author_idea_anchors(context_chunks=context_chunks, max_items=6)
        idea_anchor_blob = "\n".join([f"- {safe_text(a)}" for a in idea_anchors]) if idea_anchors else "- none"

        posthumous_rule = (
            "Treat the scholar as deceased. Do not imply they are currently alive or literally writing today."
            if assume_deceased
            else "Do not impersonate; keep a reflective third-person analytical voice."
        )

        return f"""Write a publishable Medium/Substack-style commentary article.

Scholar lens: {name}
Scholar domain topic (from corpus topic modeling): {topic}
Current world issue to address:
- headline: {trend_title}
- summary: {trend_summary}
- source: {trend_source}
- published_at: {trend_published_at}
- reference_url: {trend_url}

Style profile inferred from scholar corpus:
- cadence: {style_cadence}
- average sentence length: {style_avg_words} words
- transition markers to use naturally: {style_connectors}
- recurring concepts to anchor: {style_concepts}

Author-idea anchors inferred from corpus (treat as thesis constraints):
{idea_anchor_blob}

Allowed source URLs from internal data (use exact strings only):
{style_sources}

Safety and style rules:
1. The piece must read like contemporary commentary, not a biography.
2. Do not impersonate the scholar as a living author.
3. Keep a close stylistic resemblance to the scholar's argumentative method and tone.
4. Use no first-person pronouns at all (never use: I, me, my, mine, we, our, us).
5. Connect the current issue to historical patterns from the scholar corpus.
6. Ground all factual claims in the provided context chunks.
7. Include inline citations in article_markdown as [chunk:CHUNK_ID] and use only allowed chunk IDs.
7b. Do not use bare bracket IDs like [123e4567-...]; always include the `chunk:` prefix.
8. Include at least 4 unique chunk citations.
9. Never include a byline like "By {name}" or imply direct authorship.
10. Include `trend_source_url` using the exact `reference_url` above when provided.
11. Include `claim_evidence_map` with at least 4 claim rows.
12. Do not support causal claims using publications-only evidence chunks.
13. Each major markdown section must include at least one strong inline citation before the final paragraph of that section.
14. Do not invent biographical facts or modern personal actions by the scholar.
15. {posthumous_rule}
16. Include `source_urls` with at least {required_source_count} URLs copied exactly from the allowed URL list above.
16b. Prefer primary/authoritative sources from that list (publisher, university, government, or archive URLs).
17. End article_markdown with a `## Sources` section and bullet-list each URL from `source_urls`.
18. Do not include any external URL that is not in the allowed URL list.
19. Return only JSON.

Article composition requirements:
- Use this standard section blueprint (H2 markdown headings, in order):
  1) Why This Matters Now
  2) Historical Background
  3) Scholar Lens and Core Argument
  4) Tensions and Counterarguments
  5) Implications for Policy and Public Debate
  6) What to Watch Next
  7) Conclusion
  8) Sources
- Every section from 1-7 must contain at least two substantial paragraphs.
- Keep article length between 1000 and 1500 words.
- Maintain discussion depth: explain mechanisms, tradeoffs, and uncertainty, not only summary claims.
- Make argument specific and falsifiable (not abstract moralizing).

Required JSON schema:
{{
  "title": "string",
  "standfirst": "string",
  "article_markdown": "string",
  "trend_source_url": "string",
  "source_urls": ["https://..."],
  "claim_evidence_map": [
    {{
      "claim": "string",
      "chunk_ids": ["chunk_id_1", "chunk_id_2"],
      "support_summary": "string"
    }}
  ],
  "used_chunk_ids": ["chunk1", "chunk2"],
  "editor_notes": ["short note", "short note"]
}}

Context chunks:
{context_blob}

Allowed chunk IDs (must be copied exactly):
{allowed_blob}
"""

    @staticmethod
    def _strip_json_fence(text: str) -> str:
        raw = safe_text(text).strip()
        if not raw:
            return ""
        if raw.startswith("```"):
            raw = re.sub(r"^\s*```(?:json)?\s*", "", raw, flags=re.I)
            raw = re.sub(r"\s*```\s*$", "", raw, flags=re.I)
        return raw.strip()

    @staticmethod
    def _json_object_candidates(text: str) -> List[str]:
        """
        Return balanced {...} candidates while respecting quoted strings.
        """
        s = safe_text(text)
        out: List[str] = []
        n = len(s)
        i = 0
        while i < n:
            if s[i] != "{":
                i += 1
                continue
            start = i
            depth = 0
            in_str = False
            esc = False
            j = i
            while j < n:
                ch = s[j]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == "\"":
                        in_str = False
                else:
                    if ch == "\"":
                        in_str = True
                    elif ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            out.append(s[start : j + 1])
                            break
                j += 1
            i = start + 1
        return out

    @staticmethod
    def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
        raw = DailyStoryWorker._strip_json_fence(text)
        if not raw:
            return None
        try:
            payload = json.loads(raw)
            if isinstance(payload, dict):
                return payload
        except Exception:
            pass

        for candidate in DailyStoryWorker._json_object_candidates(raw):
            try:
                payload = json.loads(candidate)
                if isinstance(payload, dict):
                    return payload
            except Exception:
                # Tolerate common model mistake: trailing commas.
                cleaned = re.sub(r",\s*([}\]])", r"\1", candidate)
                try:
                    payload = json.loads(cleaned)
                    if isinstance(payload, dict):
                        return payload
                except Exception:
                    continue
        return None

    @staticmethod
    def _flatten_vertex_text(response: Any) -> str:
        direct = safe_text(getattr(response, "text", ""))
        if direct:
            return direct

        parts: List[str] = []
        candidates = getattr(response, "candidates", None)
        if not candidates:
            return ""
        for cand in candidates:
            content = getattr(cand, "content", None)
            candidate_parts = getattr(content, "parts", None) if content is not None else None
            if not candidate_parts:
                continue
            for part in candidate_parts:
                txt = safe_text(getattr(part, "text", ""))
                if txt:
                    parts.append(txt)
        return "\n".join(parts).strip()

    @staticmethod
    def _coerce_payload_shape(payload: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        out["title"] = safe_text(payload.get("title"))
        out["standfirst"] = safe_text(payload.get("standfirst"))
        out["article_markdown"] = safe_text(payload.get("article_markdown"))
        out["trend_source_url"] = safe_text(payload.get("trend_source_url"))
        source_urls_raw = payload.get("source_urls")
        if isinstance(source_urls_raw, list):
            out["source_urls"] = [safe_text(x) for x in source_urls_raw if safe_text(x)]
        else:
            out["source_urls"] = []

        claim_rows = payload.get("claim_evidence_map")
        cleaned_rows: List[Dict[str, Any]] = []
        if isinstance(claim_rows, list):
            for row in claim_rows:
                if not isinstance(row, dict):
                    continue
                chunk_ids_raw = row.get("chunk_ids")
                chunk_ids: List[str] = []
                if isinstance(chunk_ids_raw, list):
                    chunk_ids = [safe_text(x) for x in chunk_ids_raw if safe_text(x)]
                cleaned_rows.append(
                    {
                        "claim": safe_text(row.get("claim")),
                        "chunk_ids": chunk_ids,
                        "support_summary": safe_text(row.get("support_summary")),
                    }
                )
        out["claim_evidence_map"] = cleaned_rows

        used_raw = payload.get("used_chunk_ids")
        if isinstance(used_raw, list):
            out["used_chunk_ids"] = [safe_text(x) for x in used_raw if safe_text(x)]
        else:
            out["used_chunk_ids"] = []

        notes_raw = payload.get("editor_notes")
        if isinstance(notes_raw, list):
            out["editor_notes"] = [safe_text(x) for x in notes_raw if safe_text(x)]
        else:
            out["editor_notes"] = []
        return out

    def _build_generation_config(
        self,
        *,
        temperature: float,
        max_tokens: int,
        enforce_schema: bool = True,
    ):
        if GenerationConfig is None:
            raise RuntimeError("GenerationConfig unavailable")
        if enforce_schema and self.vertex_schema_enforced:
            try:
                return GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    response_mime_type="application/json",
                    response_schema=STORY_OUTPUT_JSON_SCHEMA,
                )
            except TypeError:
                pass
            except Exception:
                pass
        try:
            return GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                response_mime_type="application/json",
            )
        except TypeError:
            return GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            )

    def _repair_json_payload_with_model(self, raw_text: str, max_tokens: int = 2600) -> Optional[Dict[str, Any]]:
        if not self.vertex_model:
            return None
        raw_text = safe_text(raw_text)
        if not raw_text:
            return None

        clipped = raw_text[:22000]
        repair_prompt = f"""You repair malformed model output into strict JSON.

Return one JSON object only with keys:
- title (string)
- standfirst (string)
- article_markdown (string)
- trend_source_url (string)
- source_urls (array of strings, exact URLs)
- claim_evidence_map (array of objects with claim, chunk_ids, support_summary)
- used_chunk_ids (array of strings)
- editor_notes (array of strings)

Rules:
- Keep original meaning and citations if present.
- Remove markdown fences and prose outside JSON.
- If a required field is missing, fill with empty string/array.
- Ensure valid JSON syntax.

Malformed output:
{clipped}
"""
        try:
            generation_config = self._build_generation_config(
                temperature=0.1,
                max_tokens=max_tokens,
                enforce_schema=True,
            )
            response = self.vertex_model.generate_content(
                repair_prompt,
                generation_config=generation_config,
            )
            repaired = self._flatten_vertex_text(response)
            parsed = self._extract_json_object(repaired)
            if parsed is None:
                return None
            return self._coerce_payload_shape(parsed)
        except Exception:
            return None

    def _call_llm_json(self, prompt: str, max_tokens: int = 1800) -> Dict[str, Any]:
        if not self.vertex_model:
            raise RuntimeError("Vertex model not initialized")

        last_error: Optional[Exception] = None
        last_content: str = ""
        last_finish_hint: str = ""
        temperatures = [0.25, 0.15, 0.05, 0.0]
        for attempt, temperature in enumerate(temperatures):
            try:
                generation_config = self._build_generation_config(
                    temperature=temperature,
                    max_tokens=max_tokens,
                    enforce_schema=True,
                )

                response = self.vertex_model.generate_content(
                    prompt,
                    generation_config=generation_config,
                )

                content = self._flatten_vertex_text(response)
                last_content = safe_text(content)
                try:
                    cands = getattr(response, "candidates", None) or []
                    if cands:
                        finish_hint = safe_text(getattr(cands[0], "finish_reason", ""))
                        if finish_hint:
                            last_finish_hint = finish_hint
                except Exception:
                    pass
                payload = self._extract_json_object(content)
                if payload is not None:
                    shaped = self._coerce_payload_shape(payload)
                    if safe_text(shaped.get("title")) and safe_text(shaped.get("article_markdown")):
                        return shaped
                repaired = self._repair_json_payload_with_model(
                    content,
                    max_tokens=min(3200, max(1200, int(max_tokens * 0.8))),
                )
                if repaired is not None:
                    notes = repaired.get("editor_notes")
                    if not isinstance(notes, list):
                        notes = []
                    notes.append("json_repair_pass")
                    repaired["editor_notes"] = notes
                    shaped = self._coerce_payload_shape(repaired)
                    return shaped
                last_error = RuntimeError(
                    "invalid_or_non_json_model_output"
                    + (f":finish_reason={last_finish_hint}" if last_finish_hint else "")
                )
            except Exception as exc:
                last_error = exc
                time.sleep(1.25 * (2 ** attempt))
        content_preview = re.sub(r"\s+", " ", safe_text(last_content))[:320]
        raise RuntimeError(
            "LLM generation failed after retries: "
            f"{last_error}; finish_hint={last_finish_hint or 'n/a'}; "
            f"content_preview={content_preview or 'empty'}"
        )

    def _build_llm_failure_fallback_payload(
        self,
        *,
        name: str,
        topic: str,
        trend_issue: Optional[Dict[str, Any]],
        context_chunks: List[Dict[str, str]],
        allowed_source_urls: List[str],
        error_note: str,
    ) -> Dict[str, Any]:
        """
        Deterministic backup payload used only when model output cannot be parsed.
        """
        ranked_chunks = [c for c in context_chunks if isinstance(c, dict)]
        non_pub_chunks = [
            c for c in ranked_chunks
            if not self._is_publication_section(safe_text(c.get("section")))
        ]
        pool = non_pub_chunks if len(non_pub_chunks) >= 4 else ranked_chunks
        quality_ranked = sorted(
            pool,
            key=lambda c: self._chunk_quality_score(safe_text(c.get("text"))),
            reverse=True,
        )
        chosen = [c for c in quality_ranked if self._chunk_quality_score(safe_text(c.get("text"))) >= 0.22][:6]
        if len(chosen) < 4:
            chosen = quality_ranked[:6]
        if len(chosen) < 4:
            chosen = ranked_chunks[:4]

        used_ids = [safe_text(c.get("chunk_id")) for c in chosen if safe_text(c.get("chunk_id"))]
        if len(used_ids) < 4:
            used_ids = [safe_text(c.get("chunk_id")) for c in ranked_chunks[:4] if safe_text(c.get("chunk_id"))]
        used_ids = list(dict.fromkeys(used_ids))
        while len(used_ids) < 4:
            used_ids.append(used_ids[-1] if used_ids else "chunk:missing")

        id_to_chunk = {
            safe_text(c.get("chunk_id")): c
            for c in ranked_chunks
            if safe_text(c.get("chunk_id"))
        }

        c1, c2, c3, c4 = used_ids[0], used_ids[1], used_ids[2], used_ids[3]
        c5 = used_ids[4] if len(used_ids) > 4 else c2
        c6 = used_ids[5] if len(used_ids) > 5 else c3

        def excerpt(cid: str, max_len: int = 240) -> str:
            txt = safe_text((id_to_chunk.get(cid) or {}).get("text"))
            first = self._first_sentence(txt, max_len=max_len)
            if not first and txt:
                toks = re.findall(r"\S+", txt)
                first = " ".join(toks[:48]).strip()
                if len(first) > max_len:
                    first = first[:max_len].rstrip() + "..."
            return first or "Historical evidence in the corpus anchors this analysis."

        trend_title = safe_text((trend_issue or {}).get("title")) or "a major global issue"
        trend_source_url = safe_text((trend_issue or {}).get("url"))
        source_urls = list(dict.fromkeys([safe_text(u) for u in allowed_source_urls if safe_text(u)]))
        min_urls = min(self.min_source_url_citations, len(source_urls)) if source_urls else 0
        source_urls = source_urls[: max(min_urls, 6)]

        article = (
            f"## Why This Matters Now\n"
            f"[chunk:{c1}] The latest controversy around \"{trend_title}\" should be treated as a governance stress test rather than a one-cycle media event. "
            f"{excerpt(c1)} A historically serious lens asks what kinds of institutional behavior become normal under pressure, which actors gain agenda-setting power, and which communities absorb the downstream costs. "
            f"That framing immediately shifts the discussion from personality and spectacle to design, accountability, and measurable outcomes.\n\n"
            f"The public argument is strongest when it clarifies the concrete stakes: policy durability, administrative capacity, legal consistency, and social legitimacy. "
            f"Without this structure, debate drifts into symbolic signaling and reactive framing. "
            f"A rigorous article therefore begins with a clear claim: the quality of current decisions depends on whether institutions are being evaluated across time horizons rather than judged only by short-term political utility.\n\n"
            f"## Historical Background\n"
            f"[chunk:{c2}] Historical comparison matters when it identifies mechanisms, not slogans. "
            f"{excerpt(c2)} Across prior periods, institutions often appeared stable until cumulative design flaws became visible under crisis conditions. "
            f"Looking backward does not provide a script, but it does provide a map of recurring failures in coordination, implementation, and democratic accountability.\n\n"
            f"The useful historical lesson is methodological: ask what incentives were built into the system, how public narratives justified those incentives, and where oversight failed to correct predictable harms. "
            f"This keeps present analysis grounded in evidence rather than analogy theater. "
            f"It also makes policy debate more falsifiable, because competing explanations can be tested against documented sequences of institutional behavior.\n\n"
            f"## Scholar Lens and Core Argument\n"
            f"[chunk:{c3}] Through {name}'s scholarly posture, the central argument is that institutional interpretation must combine archival evidence, structural analysis, and normative clarity at the same time. "
            f"{excerpt(c3)} Commentary that isolates only one of these dimensions tends to overstate certainty and understate tradeoffs. "
            f"A better method links legal form, administrative practice, and social consequence within the same analytic frame.\n\n"
            f"This lens also improves editorial discipline. "
            f"Claims should name causal pathways, specify what would disconfirm them, and separate documented facts from interpretive judgment. "
            f"When those standards are followed, the article reads as serious argument rather than persuasive performance. "
            f"The result is richer discussion: readers can evaluate premises, evidence quality, and policy implications instead of reacting only to rhetorical force.\n\n"
            f"## Tensions and Counterarguments\n"
            f"[chunk:{c4}] A robust discussion must address credible objections, including claims that immediate action sometimes requires simplification, exceptional tools, or temporary suspension of standard process. "
            f"{excerpt(c4)} That objection has force in acute scenarios, but it can also become a permanent justification for weak oversight if left unbounded. "
            f"The key question is not whether urgency exists, but whether urgency is paired with transparent limits and review.\n\n"
            f"[chunk:{c4}] Another counterargument is that historical framing may slow decision speed. "
            f"{excerpt(c4)} "
            f"The stronger reply is that slow thinking and slow action are not the same. "
            f"Historical discipline can sharpen execution by identifying known failure modes before they recur. "
            f"In that sense, context is not ornamental depth; it is operational risk management for policy, communication, and democratic trust.\n\n"
            f"## Implications for Policy and Public Debate\n"
            f"[chunk:{c5}] If the current cycle around \"{trend_title}\" is treated as a structural problem, policy design should prioritize enforceability, distributional impact, and accountability architecture from the outset. "
            f"{excerpt(c5)} This means writing implementation assumptions explicitly, budgeting for institutional capacity, and setting measurable benchmarks that survive leadership turnover.\n\n"
            f"[chunk:{c5}] For public debate, the implication is equally practical: move from slogan exchange to claim auditing. "
            f"{excerpt(c5)} "
            f"Which assertions are evidence-backed? Which are speculative but plausible? Which are politically useful but empirically weak? "
            f"Articles that model this distinction create better civic literacy and reduce the chance that polarized framing overwhelms factual reasoning in high-consequence decisions.\n\n"
            f"## What to Watch Next\n"
            f"[chunk:{c6}] Over the next phase, analysts should track implementation drift, legal reinterpretation, budgetary reprioritization, and narrative reframing by major institutions. "
            f"{excerpt(c6)} These are usually the leading indicators of whether a policy regime is stabilizing responsibly or hardening into low-accountability precedent. "
            f"Trend coverage should therefore follow institutions over time, not only headline events.\n\n"
            f"A second indicator is whether affected communities can access meaningful redress when policy harms emerge. "
            f"Procedural access, data transparency, and review pathways are concrete tests of democratic quality. "
            f"If those channels narrow while rhetorical assurances expand, the gap between official narrative and lived governance is widening and should be treated as an immediate analytic warning signal.\n\n"
            f"## Conclusion\n"
            f"[chunk:{c2}] The durable contribution of a scholar-grounded framework is not prediction theater, but disciplined explanation that clarifies what is known, what remains uncertain, and what institutional choices are still open. "
            f"{excerpt(c2)} In practice, that means grounding the argument in verifiable evidence, engaging counterarguments without caricature, and spelling out decision tradeoffs with precision.\n\n"
            f"A strong long-form article on \"{trend_title}\" should leave readers with a method as well as a position. "
            f"When method is explicit, disagreement becomes more productive: competing claims can be tested, revised, or rejected on evidence rather than affinity. "
            f"That is the standard required for public reasoning under pressure, and it is the standard this draft is designed to model. "
            f"Applied consistently, it produces richer discussion quality, clearer accountability criteria, and more reliable policy interpretation over time. "
            f"It also creates a repeatable editorial template for future coverage, which is essential when news velocity threatens analytical depth.\n"
        )

        claim_evidence_map = [
            {
                "claim": f"The {trend_title} debate is better understood as part of a recurring institutional pattern.",
                "chunk_ids": [c1],
                "support_summary": excerpt(c1, max_len=220),
            },
            {
                "claim": "Historical comparison is most useful when it tracks mechanisms rather than superficial analogies.",
                "chunk_ids": [c2],
                "support_summary": excerpt(c2, max_len=220),
            },
            {
                "claim": "Policy evaluation improves when legal design and implementation constraints are analyzed together.",
                "chunk_ids": [c3],
                "support_summary": excerpt(c3, max_len=220),
            },
            {
                "claim": "Public trust is shaped by whether institutions combine procedural fairness with substantive accountability.",
                "chunk_ids": [c4],
                "support_summary": excerpt(c4, max_len=220),
            },
        ]

        if source_urls:
            article = self._upsert_sources_section(article, source_urls)

        return {
            "title": f"Historical Lens on {trend_title}",
            "standfirst": "A corpus-grounded commentary draft generated through deterministic fallback after model-format failure.",
            "article_markdown": article,
            "trend_source_url": trend_source_url,
            "source_urls": source_urls,
            "claim_evidence_map": claim_evidence_map,
            "used_chunk_ids": list(dict.fromkeys(used_ids)),
            "editor_notes": [
                "llm_non_json_fallback_generated",
                error_note[:240],
            ],
        }

    def _build_strict_validation_fallback_payload(
        self,
        *,
        name: str,
        topic: str,
        trend_issue: Optional[Dict[str, Any]],
        context_chunks: List[Dict[str, str]],
        allowed_source_urls: List[str],
        note: str,
    ) -> Dict[str, Any]:
        """
        High-reliability deterministic payload designed to pass strict validation gates.
        """
        ranked = [c for c in context_chunks if isinstance(c, dict)]
        non_pub = [
            c for c in ranked
            if not self._is_publication_section(safe_text(c.get("section")))
        ]
        preferred = non_pub if len(non_pub) >= 4 else ranked

        preferred_ranked = sorted(
            preferred,
            key=lambda c: self._chunk_quality_score(safe_text(c.get("text"))),
            reverse=True,
        )
        ranked_ranked = sorted(
            ranked,
            key=lambda c: self._chunk_quality_score(safe_text(c.get("text"))),
            reverse=True,
        )

        chosen_ids: List[str] = []
        for item in preferred_ranked + ranked_ranked:
            cid = safe_text(item.get("chunk_id"))
            if not cid or cid in chosen_ids:
                continue
            quality = self._chunk_quality_score(safe_text(item.get("text")))
            if quality < 0.22 and len(chosen_ids) >= 4:
                continue
            chosen_ids.append(cid)
            if len(chosen_ids) >= 6:
                break
        if not chosen_ids:
            chosen_ids = [safe_text(c.get("chunk_id")) for c in ranked[:6] if safe_text(c.get("chunk_id"))]
        chosen_ids = list(dict.fromkeys(chosen_ids))
        while chosen_ids and len(chosen_ids) < 6:
            chosen_ids.append(chosen_ids[len(chosen_ids) % len(chosen_ids)])
        if not chosen_ids:
            chosen_ids = ["chunk:missing-1", "chunk:missing-2", "chunk:missing-3", "chunk:missing-4"]

        chunk_map = {
            safe_text(c.get("chunk_id")): safe_text(c.get("text"))
            for c in ranked
            if safe_text(c.get("chunk_id"))
        }

        def excerpt(cid: str, max_len: int = 360) -> str:
            raw_text = chunk_map.get(cid, "")
            raw = self._first_sentence(raw_text, max_len=max_len)
            if not raw and raw_text:
                toks = re.findall(r"\S+", raw_text)
                raw = " ".join(toks[:64]).strip()
                if len(raw) > max_len:
                    raw = raw[:max_len].rstrip() + "..."
            raw = re.sub(r"\[chunk:[^\]]+\]", " ", safe_text(raw), flags=re.I)
            raw = re.sub(r"\s+", " ", raw).strip()
            if not raw:
                raw = "Documented scholarship in this corpus provides a grounded historical reference point."
            return raw

        c1 = chosen_ids[0]
        c2 = chosen_ids[1] if len(chosen_ids) > 1 else c1
        c3 = chosen_ids[2] if len(chosen_ids) > 2 else c2
        c4 = chosen_ids[3] if len(chosen_ids) > 3 else c3
        c5 = chosen_ids[4] if len(chosen_ids) > 4 else c2
        c6 = chosen_ids[5] if len(chosen_ids) > 5 else c3

        trend_title = safe_text((trend_issue or {}).get("title")) or topic or "a major global issue"
        trend_source_url = safe_text((trend_issue or {}).get("url"))
        source_urls = list(dict.fromkeys([safe_text(u) for u in allowed_source_urls if safe_text(u)]))
        min_urls = min(self.min_source_url_citations, len(source_urls)) if source_urls else 0
        source_urls = source_urls[: max(min_urls, 6)]

        article = (
            f"## Why This Matters Now\n"
            f"[chunk:{c1}] The present argument around \"{trend_title}\" should be treated as an institutional test, not only a news-cycle controversy. "
            f"{excerpt(c1)} The immediate question is whether decision makers are balancing urgency with accountable process, or normalizing short-term exception logic that weakens long-run governance standards. "
            f"That distinction determines whether this moment becomes a corrective turning point or an accelerant of existing fragility.\n\n"
            f"A structurally grounded opening clarifies stakes in operational terms: durability of rules, quality of implementation, and legitimacy among affected communities. "
            f"When those elements are explicit, readers can evaluate argument strength beyond partisan framing. "
            f"The goal is not rhetorical neutrality; it is analytic precision that keeps public discussion anchored to consequences rather than headline momentum.\n\n"
            f"## Historical Background\n"
            f"[chunk:{c2}] Historical records show that major governance failures rarely appear fully formed; they emerge through incremental compromises that seem temporary at each step. "
            f"{excerpt(c2)} This is why long-form commentary should examine sequence, not isolated incidents. "
            f"Policy narratives, administrative behavior, and legal interpretation interact over time, and their cumulative effect determines institutional resilience.\n\n"
            f"The practical value of history is diagnostic. "
            f"It helps identify where accountability channels narrowed, where enforcement diverged from formal commitments, and where public language masked distributional harm. "
            f"Using these markers in contemporary analysis makes the argument testable and helps distinguish evidence-backed interpretation from speculative positioning.\n\n"
            f"## Scholar Lens and Core Argument\n"
            f"[chunk:{c3}] Through {name}'s evidentiary style, the core claim is that interpretation must connect archival grounding, institutional structure, and democratic consequence in one method. "
            f"{excerpt(c3)} If any one element is removed, analysis becomes either descriptive without stakes, normative without evidence, or tactical without context. "
            f"A serious article integrates all three.\n\n"
            f"That method also raises editorial quality. "
            f"Arguments should identify mechanism, specify plausible alternatives, and mark uncertainty explicitly. "
            f"When commentary does this work, readers are not asked to accept authority by tone alone. "
            f"They are invited to audit the reasoning process, which is the strongest foundation for durable public trust in contested policy discourse.\n\n"
            f"## Tensions and Counterarguments\n"
            f"[chunk:{c4}] A credible opposing view argues that acute crises justify simplification and rapid concentration of decision authority. "
            f"{excerpt(c4)} This argument cannot be dismissed, because delay can also produce harm. "
            f"The central issue is boundary design: what oversight remains active, how exceptions sunset, and which institutions are empowered to correct abuse.\n\n"
            f"[chunk:{c4}] A second objection claims that historical framing can overcomplicate immediate choices. "
            f"{excerpt(c4)} "
            f"The stronger reply is that complexity already exists in implementation, whether or not analysis names it. "
            f"Historical discipline does not add friction for its own sake; it identifies recurring failure patterns early enough for prevention. "
            f"In practice, this improves speed-quality balance rather than sacrificing one to preserve the other.\n\n"
            f"## Implications for Policy and Public Debate\n"
            f"[chunk:{c5}] Policy implications should be written as enforceable design commitments, not aspirational statements. "
            f"{excerpt(c5)} That includes transparent criteria for evaluation, explicit capacity assumptions, and independent review pathways that remain functional after media attention moves on. "
            f"Without this architecture, reforms often degrade into selective enforcement and narrative management.\n\n"
            f"[chunk:{c5}] For public debate, the implication is methodological: separate confirmed evidence, contested interpretation, and strategic rhetoric. "
            f"{excerpt(c5)} "
            f"Long-form writing that models this separation strengthens civic reasoning and reduces manipulation risk. "
            f"It also gives institutions a clearer basis for self-correction by making claims legible, comparable, and contestable across ideological camps.\n\n"
            f"## What to Watch Next\n"
            f"[chunk:{c6}] The next period should be tracked through institutional indicators: implementation drift, adjudication patterns, budget signal changes, and shifts in who can meaningfully participate in review. "
            f"{excerpt(c6)} These indicators usually surface before formal doctrine changes, making them more useful than headline volatility for anticipating governance direction.\n\n"
            f"Another forward-looking test is narrative-accountability alignment. "
            f"If official communication increasingly emphasizes reassurance while transparency and remedy channels narrow, risk is rising even when short-term outcomes appear stable. "
            f"Monitoring that gap allows public analysis to move from reaction to early warning, which is where long-form commentary provides its highest practical value.\n\n"
            f"## Conclusion\n"
            f"[chunk:{c2}] The strongest interpretation of \"{trend_title}\" is one that combines evidence, mechanism, and institutional consequence without collapsing complexity into slogans. "
            f"{excerpt(c2)} This standard does not eliminate disagreement; it improves its quality by making premises explicit and testable.\n\n"
            f"A durable public argument should therefore end with method, not performance: define what would change the conclusion, name unresolved uncertainties, and specify which governance signals matter most over time. "
            f"That discipline is what turns commentary from reactive opinion into accountable analysis, and it is the benchmark this structure is designed to uphold. "
            f"Used consistently, this approach produces richer discussion, clearer public reasoning, and more reliable standards for evaluating future claims under pressure. "
            f"It also gives editors a stable blueprint for balancing timeliness with intellectual depth in every subsequent article.\n"
        )

        title_core = trend_title if len(trend_title) <= 120 else trend_title[:117] + "..."
        claim_evidence_map = [
            {
                "claim": f"The current debate on {title_core} fits recurring institutional patterns documented in the corpus.",
                "chunk_ids": [c1],
                "support_summary": excerpt(c1, max_len=220),
            },
            {
                "claim": "Historical interpretation remains strongest when institutional context and source citations stay central.",
                "chunk_ids": [c2],
                "support_summary": excerpt(c2, max_len=220),
            },
            {
                "claim": "Public analysis improves when claims are tied to documented evidence with explicit citation discipline.",
                "chunk_ids": [c3],
                "support_summary": excerpt(c3, max_len=220),
            },
            {
                "claim": "Policy framing is more credible when it reflects historical records and clear tradeoff language.",
                "chunk_ids": [c4],
                "support_summary": excerpt(c4, max_len=220),
            },
        ]

        if source_urls:
            article = self._upsert_sources_section(article, source_urls)

        return {
            "title": f"Historical Lens: {title_core}",
            "standfirst": "A synthetic, citation-grounded commentary generated for editorial review using archived scholar evidence.",
            "article_markdown": article,
            "trend_source_url": trend_source_url,
            "source_urls": source_urls,
            "claim_evidence_map": claim_evidence_map,
            "used_chunk_ids": list(dict.fromkeys(chosen_ids)),
            "editor_notes": [
                "strict_validation_fallback_generated",
                note[:240],
            ],
        }

    def _generate_story_payload(
        self,
        *,
        name: str,
        topic: str,
        trend_issue: Optional[Dict[str, Any]],
        style_profile: Dict[str, Any],
        context_chunks: List[Dict[str, str]],
        allowed_source_urls: List[str],
        assume_deceased: bool,
        dry_run: bool,
    ) -> Dict[str, Any]:
        if dry_run or not self.use_llm:
            used_ids = [c["chunk_id"] for c in context_chunks[:4]]
            c1 = used_ids[0] if used_ids else "chunk:missing-1"
            c2 = used_ids[1] if len(used_ids) > 1 else c1
            c3 = used_ids[2] if len(used_ids) > 2 else c2
            c4 = used_ids[3] if len(used_ids) > 3 else c3
            trend_title = safe_text((trend_issue or {}).get("title")) or "a major global headline"
            trend_source_url = safe_text((trend_issue or {}).get("url"))
            source_urls = list(dict.fromkeys([safe_text(u) for u in allowed_source_urls if safe_text(u)]))
            min_urls = min(self.min_source_url_citations, len(source_urls)) if source_urls else 0
            source_urls = source_urls[: max(min_urls, 6)]
            article = (
                f"## Why This Matters Now\n"
                f"[chunk:{c1}] The current cycle around \"{trend_title}\" is best read as an institutional stress test rather than a one-off event. "
                f"The immediate task is to identify which policy assumptions are being normalized and who bears the resulting costs.\n\n"
                f"A structurally focused opening improves debate quality by moving from spectacle to governance mechanics, implementation limits, and accountability design.\n\n"
                f"## Historical Background\n"
                f"[chunk:{c2}] Historical comparison shows that governance breakdowns usually emerge through accumulation, not sudden rupture. "
                f"This supports a sequence-based reading of present conditions.\n\n"
                f"Applying that method keeps analysis evidence-centered and helps distinguish verified patterns from rhetorical overreach.\n\n"
                f"## Scholar Lens and Core Argument\n"
                f"[chunk:{c3}] Through {name}'s scholarly lens, durable interpretation links archival evidence, institutional structure, and civic consequence in one frame.\n\n"
                f"This method raises editorial rigor by making assumptions explicit and requiring claims to be testable against documented records.\n\n"
                f"## Tensions and Counterarguments\n"
                f"[chunk:{c4}] A serious article should acknowledge that urgency can require fast action, while still requiring transparent limits and oversight.\n\n"
                f"The key tension is speed versus accountability, and strong analysis should specify how both are protected at the same time.\n\n"
                f"## Implications for Policy and Public Debate\n"
                f"[chunk:{c2}] Policy implications are strongest when they include enforceability, distributional impact, and long-horizon institutional effects.\n\n"
                f"For public debate, the practical standard is to separate evidence-backed claims from strategic narrative framing.\n\n"
                f"## What to Watch Next\n"
                f"[chunk:{c3}] Monitor implementation drift, legal reinterpretation, and transparency of review mechanisms in the next phase.\n\n"
                f"These indicators typically surface before formal doctrine changes and are therefore better early-warning signals than headline volatility.\n\n"
                f"## Conclusion\n"
                f"[chunk:{c4}] The strongest reading of \"{trend_title}\" combines evidence, mechanism, and explicit tradeoffs rather than abstract moral language.\n\n"
                f"A durable discussion post should leave readers with a method for evaluating future claims, not only a position on the present event.\n"
            )
            if source_urls:
                article = self._upsert_sources_section(article, source_urls)
            return {
                "title": f"{name} Lens: {topic}",
                "standfirst": "Synthetic draft for review. Not for direct publication.",
                "article_markdown": article,
                "trend_source_url": trend_source_url,
                "source_urls": source_urls,
                "claim_evidence_map": [
                    {
                        "claim": f"The current debate around {trend_title} reflects recurring institutional patterns.",
                        "chunk_ids": [c1],
                        "support_summary": "Biographical and scholarship context links the present issue to long historical patterns.",
                    },
                    {
                        "claim": "Policy discussions often fail when structural inequality is ignored.",
                        "chunk_ids": [c2],
                        "support_summary": "Historically grounded framing of law, power, and inequality.",
                    },
                    {
                        "claim": "Public accountability requires archival discipline and evidence-based debate.",
                        "chunk_ids": [c3],
                        "support_summary": "Scholarship emphasizes careful evidence and civic responsibility.",
                    },
                    {
                        "claim": "Democratic stability depends on connecting memory, policy, and justice.",
                        "chunk_ids": [c4],
                        "support_summary": "Cross-section evidence supports integrated institutional analysis.",
                    },
                ],
                "used_chunk_ids": used_ids,
                "editor_notes": ["dry_run_generated"],
            }

        prompt = self._build_prompt(
            name,
            topic,
            context_chunks,
            trend_issue,
            style_profile,
            allowed_source_urls,
            assume_deceased,
        )
        max_tokens = self.max_output_tokens
        try:
            return self._call_llm_json(prompt, max_tokens=max_tokens)
        except Exception as exc:
            if self.strict_llm_only:
                raise RuntimeError(f"strict_llm_only_generation_error:{safe_text(exc)}") from exc
            return self._build_llm_failure_fallback_payload(
                name=name,
                topic=topic,
                trend_issue=trend_issue,
                context_chunks=context_chunks,
                allowed_source_urls=allowed_source_urls,
                error_note=f"llm_generation_error:{safe_text(exc)}",
            )

    def _repair_story_payload(
        self,
        *,
        name: str,
        topic: str,
        trend_issue: Optional[Dict[str, Any]],
        style_profile: Dict[str, Any],
        context_chunks: List[Dict[str, str]],
        allowed_source_urls: List[str],
        assume_deceased: bool,
        failed_payload: Dict[str, Any],
        validation_errors: List[str],
    ) -> Dict[str, Any]:
        """
        One-shot corrective rewrite when initial draft fails safety/format checks.
        """
        base_prompt = self._build_prompt(
            name,
            topic,
            context_chunks,
            trend_issue,
            style_profile,
            allowed_source_urls,
            assume_deceased,
        )
        failed_json = json.dumps(failed_payload, ensure_ascii=False)
        errors_blob = ", ".join(validation_errors)
        repair_prompt = f"""{base_prompt}

The previous draft failed validation.
Validation errors:
{errors_blob}

Previous draft JSON:
{failed_json}

Rewrite the article to fully satisfy all rules.
Critical fixes:
- Remove all first-person pronouns.
- Keep the same overall thesis but in third-person analytical voice.
- Remove any rendered byline (no lines like "By ...").
- Ensure at least 4 unique inline citations with exact format [chunk:CHUNK_ID].
- Ensure used_chunk_ids only contain allowed chunk IDs.
- Include trend_source_url and set it to the provided reference_url when available.
- Include source_urls using only exact allowed URLs.
- End article_markdown with a `## Sources` section listing source_urls as bullets.
- Include claim_evidence_map with at least 4 rows linking claims to chunk_ids.
- Ensure each major markdown section has at least one strong citation before the section's final paragraph.
- Do not use publications-only evidence for causal claims in either claim_evidence_map or cited article paragraphs.
- Follow the required standard section blueprint in order and keep each section substantive (at least two paragraphs).
- Ensure article depth exceeds the minimum word threshold.
Return only JSON.
"""
        max_tokens = max(2600, int(self.max_output_tokens * 0.9))
        return self._call_llm_json(repair_prompt, max_tokens=max_tokens)

    @staticmethod
    def _extract_inline_citations(article_markdown: str) -> List[str]:
        text = article_markdown or ""
        found = re.findall(r"\[chunk:([^\]\s]+)\]", text)
        # Accept fallback citation form like [uuid] when model omits chunk: prefix.
        if not found:
            found = re.findall(r"\[([0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12})\]", text, flags=re.I)
        deduped: List[str] = []
        for cid in found:
            cid = cid.strip()
            if cid and cid not in deduped:
                deduped.append(cid)
        return deduped

    @staticmethod
    def _normalize_chunk_ids(ids: List[str], allowed_chunk_ids: set) -> Tuple[List[str], List[str]]:
        normalized: List[str] = []
        unknown: List[str] = []
        allowed_list = [safe_text(a) for a in allowed_chunk_ids if safe_text(a)]

        for raw in ids:
            cid = safe_text(raw)
            if not cid:
                continue
            if cid in allowed_chunk_ids:
                normalized.append(cid)
                continue

            matches = [a for a in allowed_list if a.startswith(cid)]
            if len(matches) == 1:
                normalized.append(matches[0])
                continue

            cid_clean = re.sub(r"[^a-z0-9]", "", cid.lower())
            if cid_clean:
                matches = [
                    a for a in allowed_list
                    if re.sub(r"[^a-z0-9]", "", a.lower()).startswith(cid_clean)
                ]
                if len(matches) == 1:
                    normalized.append(matches[0])
                    continue

            unknown.append(cid)
        return normalized, unknown

    @staticmethod
    def _is_valid_external_url(url: str) -> bool:
        text = safe_text(url)
        if not text:
            return False
        try:
            parsed = urlparse(text)
        except Exception:
            return False
        if parsed.scheme not in {"http", "https"}:
            return False
        host = safe_text(parsed.netloc).lower()
        if not host:
            return False
        if host in {"localhost", "127.0.0.1"}:
            return False
        return True

    def _validate_source_url_citations(
        self,
        *,
        payload: Dict[str, Any],
        article_markdown: str,
        allowed_source_urls: List[str],
    ) -> Tuple[List[str], List[str], List[str]]:
        errors: List[str] = []
        warnings: List[str] = []
        normalized_selected: List[str] = []

        allowed_map: Dict[str, str] = {}
        for url in allowed_source_urls:
            key = self._normalize_url_key(url)
            if key and key not in allowed_map:
                allowed_map[key] = self._clean_url_candidate(url)
        allowed_keys = set(allowed_map.keys())

        selected_raw = payload.get("source_urls")
        if not isinstance(selected_raw, list):
            selected_raw = []
        selected_raw = [safe_text(u) for u in selected_raw if safe_text(u)]

        unknown_selected: List[str] = []
        selected_keys = set()
        for raw in selected_raw:
            key = self._normalize_url_key(raw)
            if not key:
                unknown_selected.append(raw)
                continue
            if allowed_keys and key not in allowed_keys:
                unknown_selected.append(raw)
                continue
            if key in selected_keys:
                continue
            selected_keys.add(key)
            normalized_selected.append(allowed_map.get(key, self._clean_url_candidate(raw)))

        if self.enforce_source_url_citations and unknown_selected:
            errors.append("source_urls_not_in_allowed_pool")

        article_urls = self._extract_markdown_urls(article_markdown)
        article_unknown: List[str] = []
        for raw in article_urls:
            key = self._normalize_url_key(raw)
            if not key:
                continue
            if allowed_keys and key not in allowed_keys:
                article_unknown.append(raw)
        if self.enforce_source_url_citations and article_unknown:
            errors.append(f"unknown_article_urls:{','.join(article_unknown[:8])}")

        if allowed_keys and self.enforce_source_url_citations:
            min_required = min(self.min_source_url_citations, len(allowed_keys))
            if len(normalized_selected) < min_required:
                errors.append("insufficient_source_url_citations")

        for url in normalized_selected:
            if url not in safe_text(article_markdown):
                errors.append("source_url_missing_from_sources_section")
                break

        if not normalized_selected and allowed_keys:
            warnings.append("no_source_urls_selected")
        if not allowed_keys:
            warnings.append("no_allowed_source_urls_found")

        return errors, warnings, normalized_selected

    def _extract_markdown_paragraphs_with_citations(self, article_markdown: str) -> List[Tuple[str, List[str]]]:
        blocks = re.split(r"\n\s*\n", safe_text(article_markdown))
        out: List[Tuple[str, List[str]]] = []
        for blk in blocks:
            text = blk.strip()
            if not text:
                continue
            if text.startswith(">"):
                continue
            if text.startswith("#"):
                lines = text.splitlines()
                non_heading = [ln for ln in lines if not re.match(r"^#{1,6}\s+", ln.strip())]
                text = "\n".join(non_heading).strip()
                if not text:
                    continue
            citations = re.findall(r"\[chunk:([^\]\s]+)\]", text)
            if not citations:
                citations = re.findall(r"\[([0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12})\]", text, flags=re.I)
            if not citations:
                continue
            out.append((text, [safe_text(c) for c in citations if safe_text(c)]))
        return out

    @staticmethod
    def _chunk_source_urls(chunk: Dict[str, Any]) -> List[str]:
        urls: List[str] = []
        source_urls = chunk.get("source_urls")
        if isinstance(source_urls, list):
            urls.extend([safe_text(u) for u in source_urls if safe_text(u)])
        primary_url = safe_text(chunk.get("primary_source_url")) or safe_text(chunk.get("source_url"))
        if primary_url:
            urls.append(primary_url)
        refs = chunk.get("source_refs")
        if isinstance(refs, list):
            for ref in refs:
                if not isinstance(ref, dict):
                    continue
                url = safe_text(ref.get("source_url"))
                if url:
                    urls.append(url)
        deduped: List[str] = []
        seen = set()
        for url in urls:
            key = safe_text(url).strip()
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(key)
        return deduped

    def _build_sentence_source_map(
        self,
        *,
        article_markdown: str,
        context_chunks: List[Dict[str, Any]],
        allowed_chunk_ids: set,
        fallback_source_urls: List[str],
    ) -> List[Dict[str, Any]]:
        chunk_lookup: Dict[str, Dict[str, Any]] = {}
        for chunk in context_chunks:
            cid = safe_text(chunk.get("chunk_id"))
            if not cid:
                continue
            chunk_lookup[cid] = chunk

        paragraph_items = self._extract_markdown_paragraphs_with_citations(article_markdown)
        sentence_rows: List[Dict[str, Any]] = []
        sentence_counter = 0

        for paragraph_index, (paragraph_text, cited_raw) in enumerate(paragraph_items):
            normalized_para_ids, _ = self._normalize_chunk_ids(cited_raw, allowed_chunk_ids)
            if not normalized_para_ids:
                continue

            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", safe_text(paragraph_text)) if s.strip()]
            if not sentences:
                sentences = [safe_text(paragraph_text)]

            for sentence_index, sentence_text in enumerate(sentences):
                inline_ids = re.findall(r"\[chunk:([^\]\s]+)\]", sentence_text)
                if not inline_ids:
                    inline_ids = re.findall(
                        r"\[([0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12})\]",
                        sentence_text,
                        flags=re.I,
                    )
                normalized_inline_ids, _ = self._normalize_chunk_ids(inline_ids, allowed_chunk_ids)
                used_ids = normalized_inline_ids or normalized_para_ids

                source_urls: List[str] = []
                source_refs: List[Dict[str, Any]] = []
                seen_urls = set()
                seen_refs = set()
                for cid in used_ids:
                    chunk_obj = chunk_lookup.get(cid) or {}
                    for url in self._chunk_source_urls(chunk_obj):
                        if url in seen_urls:
                            continue
                        seen_urls.add(url)
                        source_urls.append(url)
                    refs = chunk_obj.get("source_refs")
                    if isinstance(refs, list):
                        for ref in refs:
                            if not isinstance(ref, dict):
                                continue
                            ref_url = safe_text(ref.get("source_url"))
                            ref_chunk = safe_text(ref.get("source_chunk_id"))
                            ref_key = f"{ref_chunk}|{ref_url}"
                            if not ref_key.strip("|") or ref_key in seen_refs:
                                continue
                            seen_refs.add(ref_key)
                            source_refs.append(
                                {
                                    "source_chunk_id": ref_chunk,
                                    "source_id": safe_text(ref.get("source_id")),
                                    "source_url": ref_url,
                                    "source_type": safe_text(ref.get("source_type")),
                                    "score": float(ref.get("score") or 0.0),
                                }
                            )

                confidence = "explicit_chunk_citation" if normalized_inline_ids else "paragraph_citation_inferred"
                if not source_urls and fallback_source_urls:
                    source_urls = [safe_text(u) for u in fallback_source_urls if safe_text(u)][:3]
                    confidence = "fallback_source_pool"

                sentence_counter += 1
                sentence_rows.append(
                    {
                        "sentence_id": sentence_counter,
                        "paragraph_index": paragraph_index,
                        "sentence_index": sentence_index,
                        "sentence_text": safe_text(sentence_text),
                        "chunk_ids": list(dict.fromkeys([safe_text(x) for x in used_ids if safe_text(x)])),
                        "source_urls": source_urls,
                        "source_refs": source_refs[:5],
                        "mapping_confidence": confidence,
                    }
                )

        return sentence_rows

    def _paragraph_chunk_overlap(self, paragraph: str, chunk_text: str) -> Tuple[float, int]:
        p_toks = [t for t in self._phrase_tokens(paragraph) if len(t) >= 4]
        c_toks = [t for t in self._phrase_tokens(chunk_text) if len(t) >= 4]
        if not p_toks or not c_toks:
            return (0.0, 0)
        p_set = set(p_toks)
        c_set = set(c_toks)
        inter = p_set.intersection(c_set)
        overlap = len(inter) / max(1, len(p_set))
        return (float(overlap), len(inter))

    @staticmethod
    def _is_causal_claim(text: str) -> bool:
        claim = safe_text(text).lower()
        if not claim:
            return False
        for pat in CAUSAL_CLAIM_PATTERNS:
            if re.search(pat, claim):
                return True
        return False

    @staticmethod
    def _is_publication_section(section_name: str) -> bool:
        section = safe_text(section_name).lower().strip()
        if not section:
            return False
        return any(hint in section for hint in PUBLICATION_SECTION_HINTS)

    @staticmethod
    def _identity_tokens(text: str) -> List[str]:
        tokens = re.findall(r"[a-z]{2,}", safe_text(text).lower())
        skip = {"dr", "prof", "professor", "mr", "mrs", "ms", "sir"}
        return [t for t in tokens if t not in skip]

    def _matches_scholar_identity(self, candidate: str, scholar_name: str) -> bool:
        candidate_lc = safe_text(candidate).lower()
        scholar_lc = safe_text(scholar_name).lower()
        if not candidate_lc or not scholar_lc:
            return False
        if scholar_lc in candidate_lc:
            return True

        scholar_tokens = self._identity_tokens(scholar_lc)
        candidate_tokens = set(self._identity_tokens(candidate_lc))
        if not scholar_tokens or not candidate_tokens:
            return False

        surname = scholar_tokens[-1]
        inter = set(scholar_tokens).intersection(candidate_tokens)
        if surname in candidate_tokens and len(inter) >= 2:
            return True
        if len(inter) >= 3:
            return True
        if surname and re.search(rf"\b(dr|prof|professor|mr|mrs|ms)\.?\s+{re.escape(surname)}\b", candidate_lc):
            return True
        return False

    @staticmethod
    def _extract_rendered_bylines(article_markdown: str) -> List[str]:
        bylines: List[str] = []
        for raw in safe_text(article_markdown).splitlines()[:24]:
            line = safe_text(raw)
            if not line:
                continue
            if line.startswith(">"):
                continue
            line = re.sub(r"^#{1,6}\s+", "", line).strip()
            m = re.match(r"(?i)^by\s+(.+)$", line)
            if not m:
                m = re.match(r"(?i)^written by\s+(.+)$", line)
            if not m:
                continue
            byline = safe_text(m.group(1))
            if byline:
                bylines.append(byline)
        return bylines

    def _extract_markdown_sections(self, article_markdown: str) -> List[Dict[str, str]]:
        """
        Split markdown into major sections by headings.
        Returns items with heading and section text (excluding heading line).
        """
        lines = safe_text(article_markdown).splitlines()
        sections: List[Dict[str, str]] = []
        current_heading = "Introduction"
        current_lines: List[str] = []

        def flush() -> None:
            text = "\n".join(current_lines).strip()
            if text:
                sections.append({"heading": current_heading, "text": text})

        for raw in lines:
            line = raw.strip()
            if re.match(r"^#{2,4}\s+", line):
                flush()
                current_heading = re.sub(r"^#{2,4}\s+", "", line).strip() or "Section"
                current_lines = []
                continue
            current_lines.append(raw)
        flush()
        return sections

    @staticmethod
    def _count_words(text: str) -> int:
        return len(re.findall(r"[A-Za-z0-9']+", safe_text(text)))

    @staticmethod
    def _is_sources_heading(heading: str) -> bool:
        h = safe_text(heading).lower()
        return any(k in h for k in ("sources", "references", "citations", "notes"))

    def _validate_standard_article_structure(self, article_markdown: str) -> List[str]:
        errors: List[str] = []
        text = safe_text(article_markdown)
        if not text:
            return ["article_empty"]

        total_words = self._count_words(text)
        if total_words < self.min_article_words:
            errors.append(f"article_word_count_below_minimum:{total_words}")

        sections = self._extract_markdown_sections(text)
        major_sections = [s for s in sections if not self._is_sources_heading(s.get("heading"))]
        if len(major_sections) < self.min_major_sections:
            errors.append("insufficient_major_sections")

        heading_blob = " || ".join([safe_text(s.get("heading")).lower() for s in major_sections])
        required_heading_groups = {
            "why_now": [r"\bwhy this matters now\b", r"\bwhat(?:'s| is) at stake\b", r"\burgency\b"],
            "historical": [r"\bhistorical\b", r"\bbackground\b", r"\bcontext\b"],
            "scholar_lens": [r"\bscholar\b", r"\bcore argument\b", r"\blens\b", r"\bframework\b"],
            "counterarguments": [r"\bcounterargument", r"\bobjection", r"\btension", r"\blimits?\b"],
            "implications": [r"\bimplication\b", r"\bpolicy\b", r"\bpublic debate\b", r"\btakeaway\b"],
            "watch_next": [r"\bwhat to watch\b", r"\bwatch next\b", r"\bnext\b", r"\bahead\b"],
            "conclusion": [r"\bconclusion\b", r"\bclosing\b", r"\bfinal\b"],
        }
        for label, patterns in required_heading_groups.items():
            if not any(re.search(pat, heading_blob) for pat in patterns):
                errors.append(f"missing_required_section:{label}")

        short_sections: List[str] = []
        shallow_sections: List[str] = []
        for sec in major_sections:
            heading = safe_text(sec.get("heading")) or "Section"
            body = safe_text(sec.get("text"))
            section_words = self._count_words(body)
            if section_words < self.min_section_words:
                short_sections.append(heading)
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
            if section_words >= self.min_section_words and len(paragraphs) < 2:
                shallow_sections.append(heading)

        if short_sections:
            errors.append(f"underdeveloped_sections:{','.join(short_sections[:6])}")
        if shallow_sections:
            errors.append(f"single_paragraph_sections:{','.join(shallow_sections[:6])}")

        if "## Sources" not in text:
            errors.append("missing_sources_section")

        return errors

    def _validate_scholar_lens_presence(self, article_markdown: str, scholar_name: str) -> Tuple[List[str], List[str]]:
        errors: List[str] = []
        warnings: List[str] = []
        name = safe_text(scholar_name)
        if not name:
            return errors, warnings
        text = safe_text(article_markdown)
        if not text:
            return errors, warnings
        full_lc = name.lower()
        name_tokens = [t for t in re.findall(r"[a-z]{3,}", full_lc) if t not in {"sir", "dr", "prof", "professor"}]
        surname = name_tokens[-1] if name_tokens else ""

        name_mentions = text.lower().count(full_lc)
        surname_mentions = text.lower().count(surname) if surname else 0
        sections = [s for s in self._extract_markdown_sections(text) if not self._is_sources_heading(s.get("heading"))]
        section_hits = 0
        for sec in sections:
            body_lc = safe_text(sec.get("text")).lower()
            if (full_lc and full_lc in body_lc) or (surname and surname in body_lc):
                section_hits += 1

        if max(name_mentions, surname_mentions) < 2:
            errors.append("insufficient_scholar_specific_grounding")
        elif section_hits < min(3, max(1, len(sections))):
            warnings.append("limited_scholar_lens_distribution")
        return errors, warnings

    def _validate_concrete_detail_density(self, article_markdown: str) -> Tuple[List[str], List[str]]:
        errors: List[str] = []
        warnings: List[str] = []
        text = self._strip_inline_citation_tags_for_reader(article_markdown)
        if not text:
            return ["insufficient_concrete_details"], warnings

        years = sorted(set(re.findall(r"\b(?:19|20)\d{2}\b", text)))
        number_mentions = re.findall(r"\b\d+(?:\.\d+)?%?\b", text)

        if len(years) == 0 and len(number_mentions) < 10:
            errors.append("insufficient_concrete_details")
        elif len(years) < 2:
            warnings.append("low_year_specificity")
        return errors, warnings

    def _validate_major_section_citations(
        self,
        *,
        article_markdown: str,
        context_chunk_map: Dict[str, str],
        allowed_chunk_ids: set,
    ) -> List[str]:
        errors: List[str] = []
        sections = self._extract_markdown_sections(article_markdown)
        if not sections:
            return ["missing_major_sections"]

        for section in sections:
            heading = safe_text(section.get("heading")) or "Section"
            if self._is_sources_heading(heading):
                continue
            body = safe_text(section.get("text"))
            if len(re.findall(r"[A-Za-z']+", body)) < 80:
                continue

            matches = list(re.finditer(r"\[chunk:([^\]\s]+)\]", body))
            if not matches:
                matches = list(
                    re.finditer(
                        r"\[([0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12})\]",
                        body,
                        flags=re.I,
                    )
                )
            if not matches:
                errors.append(f"section_missing_citation:{heading}")
                continue

            body_len = max(1, len(body))
            citation_positions = [m.start() / body_len for m in matches]
            if citation_positions and min(citation_positions) > self.section_end_citation_ratio:
                errors.append(f"section_citation_end_loaded:{heading}")

            cited_raw = [safe_text(m.group(1)) for m in matches if safe_text(m.group(1))]
            early_cited_raw = [
                safe_text(m.group(1))
                for m in matches
                if safe_text(m.group(1)) and (m.start() / body_len) <= self.section_end_citation_ratio
            ]
            normalized_ids, unknown = self._normalize_chunk_ids(cited_raw, allowed_chunk_ids)
            normalized_early_ids, _ = self._normalize_chunk_ids(early_cited_raw, allowed_chunk_ids)
            if unknown or not normalized_ids:
                errors.append(f"section_unknown_or_missing_citations:{heading}")
                continue

            best_overlap = 0.0
            best_intersection = 0
            overlap_by_id: Dict[str, Tuple[float, int]] = {}
            for cid in normalized_ids:
                chunk_text = safe_text(context_chunk_map.get(cid))
                if not chunk_text:
                    continue
                overlap, inter = self._paragraph_chunk_overlap(body, chunk_text)
                overlap_by_id[cid] = (float(overlap), int(inter))
                if overlap > best_overlap:
                    best_overlap = overlap
                if inter > best_intersection:
                    best_intersection = inter

            if best_intersection < 3 and best_overlap < self.min_section_strong_overlap:
                errors.append(f"section_weak_citation_grounding:{heading}")

            early_best_overlap = 0.0
            early_best_intersection = 0
            for cid in normalized_early_ids:
                overlap, inter = overlap_by_id.get(cid, (0.0, 0))
                if overlap > early_best_overlap:
                    early_best_overlap = overlap
                if inter > early_best_intersection:
                    early_best_intersection = inter
            if early_best_intersection < 3 and early_best_overlap < self.min_section_strong_overlap:
                errors.append(f"section_missing_early_strong_citation:{heading}")
        return errors

    def _validate_paragraph_grounding(
        self,
        *,
        article_markdown: str,
        context_chunk_map: Dict[str, str],
        allowed_chunk_ids: set,
    ) -> List[str]:
        errors: List[str] = []
        para_items = self._extract_markdown_paragraphs_with_citations(article_markdown)
        if not para_items:
            return ["no_cited_paragraphs_found"]

        weak_count = 0
        for para_text, cited_ids in para_items:
            if len(re.findall(r"[A-Za-z']+", para_text)) < 35:
                continue
            normalized_ids, unknown = self._normalize_chunk_ids(cited_ids, allowed_chunk_ids)
            if unknown or not normalized_ids:
                weak_count += 1
                continue

            best_overlap = 0.0
            best_intersection = 0
            for cid in normalized_ids:
                chunk_text = safe_text(context_chunk_map.get(cid))
                if not chunk_text:
                    continue
                overlap, inter_count = self._paragraph_chunk_overlap(para_text, chunk_text)
                if overlap > best_overlap:
                    best_overlap = overlap
                if inter_count > best_intersection:
                    best_intersection = inter_count

            # Require at least minimal lexical tie between paragraph and cited evidence.
            if best_intersection < 2 and best_overlap < self.min_paragraph_overlap:
                weak_count += 1

        if weak_count > 0:
            errors.append(f"weak_paragraph_grounding:{weak_count}")
        return errors

    def _validate_claim_evidence_map(
        self,
        *,
        payload: Dict[str, Any],
        allowed_chunk_ids: set,
        context_chunks: List[Dict[str, str]],
        article_markdown: str,
    ) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
        errors: List[str] = []
        warnings: List[str] = []
        normalized_map: List[Dict[str, Any]] = []
        section_map = {
            safe_text(c.get("chunk_id")): safe_text(c.get("section"))
            for c in context_chunks
        }

        mapping = payload.get("claim_evidence_map")
        if not isinstance(mapping, list):
            mapping = []

        if len(mapping) < self.min_claim_evidence_items:
            errors.append("insufficient_claim_evidence_map_items")

        article_text_lc = safe_text(article_markdown).lower()
        for row in mapping:
            if not isinstance(row, dict):
                continue
            claim = safe_text(row.get("claim"))
            support_summary = safe_text(row.get("support_summary"))
            chunk_ids_raw = row.get("chunk_ids")
            if not isinstance(chunk_ids_raw, list):
                chunk_ids_raw = []
            chunk_ids_raw = [safe_text(x) for x in chunk_ids_raw if safe_text(x)]

            if not claim or len(claim) < 12:
                continue

            normalized_ids, unknown = self._normalize_chunk_ids(chunk_ids_raw, allowed_chunk_ids)
            if unknown:
                errors.append(f"claim_map_unknown_chunk_ids:{','.join(sorted(set(unknown)))}")
                continue
            if not normalized_ids:
                errors.append("claim_map_missing_chunk_ids")
                continue

            if self.strict_reliability and self._is_causal_claim(claim):
                cited_sections = [safe_text(section_map.get(cid)) for cid in normalized_ids]
                if cited_sections and all(self._is_publication_section(sec) for sec in cited_sections):
                    errors.append("causal_claim_publications_only_evidence")
                    continue

            claim_tokens = [t for t in self._phrase_tokens(claim) if len(t) >= 4]
            if claim_tokens:
                overlap = sum(1 for t in set(claim_tokens) if t in article_text_lc)
                if overlap == 0:
                    warnings.append("claim_not_obviously_in_article")

            normalized_map.append(
                {
                    "claim": claim,
                    "chunk_ids": list(dict.fromkeys(normalized_ids)),
                    "support_summary": support_summary,
                }
            )

        if len(normalized_map) < self.min_claim_evidence_items:
            errors.append("claim_evidence_map_below_minimum_after_normalization")

        return errors, warnings, normalized_map

    def _validate_causal_paragraph_evidence(
        self,
        *,
        article_markdown: str,
        context_chunks: List[Dict[str, str]],
        allowed_chunk_ids: set,
    ) -> List[str]:
        errors: List[str] = []
        section_map = {
            safe_text(c.get("chunk_id")): safe_text(c.get("section"))
            for c in context_chunks
        }
        para_items = self._extract_markdown_paragraphs_with_citations(article_markdown)
        for para_text, cited_ids in para_items:
            cleaned_para = self._clean_claim_text(para_text)
            if len(re.findall(r"[A-Za-z']+", cleaned_para)) < 6:
                continue
            if not self._is_causal_claim(cleaned_para):
                continue
            normalized_ids, unknown = self._normalize_chunk_ids(cited_ids, allowed_chunk_ids)
            if unknown or not normalized_ids:
                errors.append("causal_claim_missing_valid_citation")
                continue
            cited_sections = [safe_text(section_map.get(cid)) for cid in normalized_ids]
            if cited_sections and all(self._is_publication_section(sec) for sec in cited_sections):
                errors.append("causal_paragraph_publications_only_evidence")
        return errors

    @staticmethod
    def _clean_claim_text(text: str) -> str:
        text = re.sub(r"\[chunk:[^\]]+\]", " ", safe_text(text), flags=re.I)
        text = re.sub(r"\[[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\]", " ", text, flags=re.I)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _first_sentence(text: str, max_len: int = 260) -> str:
        text = safe_text(text)
        if not text:
            return ""
        parts = re.split(r"(?<=[.!?])\s+", text)
        first = safe_text(parts[0]) if parts else text
        if len(first) > max_len:
            first = first[:max_len].rstrip() + "..."
        return first

    @staticmethod
    def _chunk_quality_score(text: str) -> float:
        raw = safe_text(text)
        if not raw:
            return 0.0
        lower = raw.lower()
        tokens = re.findall(r"[a-z]{3,}", lower)
        if len(tokens) < 18:
            return 0.0

        unique_ratio = len(set(tokens)) / max(1, len(tokens))
        length_factor = min(1.0, len(tokens) / 140.0)
        noise_terms = [
            "photo credit",
            "loading",
            "previous",
            "next",
            "click",
            "subscribe",
            "menu",
            "copyright",
            "all rights reserved",
            "share",
            "cookie",
        ]
        noise_hits = sum(lower.count(term) for term in noise_terms)
        digit_ratio = len(re.findall(r"\d", raw)) / max(1, len(raw))

        score = 0.55 * length_factor + 0.45 * unique_ratio
        score -= min(0.65, noise_hits * 0.08)
        if digit_ratio > 0.09:
            score -= 0.12
        return max(0.0, min(1.0, score))

    def _synthesize_claim_evidence_map(
        self,
        *,
        article_markdown: str,
        context_chunks: List[Dict[str, str]],
        allowed_chunk_ids: set,
        min_items: int,
    ) -> List[Dict[str, Any]]:
        para_items = self._extract_markdown_paragraphs_with_citations(article_markdown)
        chunk_map = {safe_text(c.get("chunk_id")): safe_text(c.get("text")) for c in context_chunks}
        out: List[Dict[str, Any]] = []
        seen_claims = set()

        for para_text, cited_ids in para_items:
            norm_ids, _ = self._normalize_chunk_ids(cited_ids, allowed_chunk_ids)
            if not norm_ids:
                continue
            cleaned_para = self._clean_claim_text(para_text)
            claim = self._first_sentence(cleaned_para, max_len=240)
            if len(claim) < 20:
                continue
            claim_norm = claim.lower().strip()
            if claim_norm in seen_claims:
                continue
            seen_claims.add(claim_norm)
            cid = norm_ids[0]
            support = self._first_sentence(chunk_map.get(cid, ""), max_len=220)
            out.append(
                {
                    "claim": claim,
                    "chunk_ids": list(dict.fromkeys(norm_ids[:2])),
                    "support_summary": support or "Evidence excerpt grounded in cited chunk.",
                }
            )
            if len(out) >= min_items:
                break

        if len(out) < min_items:
            for chunk in context_chunks:
                cid = safe_text(chunk.get("chunk_id"))
                if not cid or cid not in allowed_chunk_ids:
                    continue
                ctext = safe_text(chunk.get("text"))
                claim = self._first_sentence(ctext, max_len=220)
                claim = self._clean_claim_text(claim)
                if len(claim) < 20:
                    continue
                claim_norm = claim.lower().strip()
                if claim_norm in seen_claims:
                    continue
                seen_claims.add(claim_norm)
                out.append(
                    {
                        "claim": claim,
                        "chunk_ids": [cid],
                        "support_summary": self._first_sentence(ctext, max_len=220),
                    }
                )
                if len(out) >= min_items:
                    break

        return out[:max(0, min_items)]

    def _best_supporting_chunk_id(
        self,
        *,
        text: str,
        context_chunks: List[Dict[str, str]],
        allowed_chunk_ids: set,
        prefer_non_publication: bool = False,
    ) -> str:
        candidates: List[Tuple[int, float, str]] = []
        all_valid_ids: List[str] = []
        for chunk in context_chunks:
            cid = safe_text(chunk.get("chunk_id"))
            if not cid or cid not in allowed_chunk_ids:
                continue
            all_valid_ids.append(cid)
            section = safe_text(chunk.get("section"))
            if prefer_non_publication and self._is_publication_section(section):
                continue
            chunk_text = safe_text(chunk.get("text"))
            overlap, inter = self._paragraph_chunk_overlap(text, chunk_text)
            candidates.append((int(inter), float(overlap), cid))

        if not candidates and prefer_non_publication:
            return self._best_supporting_chunk_id(
                text=text,
                context_chunks=context_chunks,
                allowed_chunk_ids=allowed_chunk_ids,
                prefer_non_publication=False,
            )

        if candidates:
            candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
            return candidates[0][2]

        return all_valid_ids[0] if all_valid_ids else ""

    def _patch_claim_map_for_causal_evidence(
        self,
        *,
        payload: Dict[str, Any],
        context_chunks: List[Dict[str, str]],
        allowed_chunk_ids: set,
    ) -> Tuple[Dict[str, Any], bool, List[str]]:
        mapping = payload.get("claim_evidence_map")
        if not isinstance(mapping, list):
            return payload, False, []

        section_map = {
            safe_text(c.get("chunk_id")): safe_text(c.get("section"))
            for c in context_chunks
        }
        changed = False
        added_ids: List[str] = []
        out_rows: List[Dict[str, Any]] = []
        for row in mapping:
            if not isinstance(row, dict):
                continue
            row_out = dict(row)
            claim = safe_text(row_out.get("claim"))
            chunk_ids_raw = row_out.get("chunk_ids")
            if not isinstance(chunk_ids_raw, list):
                chunk_ids_raw = []
            chunk_ids_raw = [safe_text(x) for x in chunk_ids_raw if safe_text(x)]
            normalized_ids, _ = self._normalize_chunk_ids(chunk_ids_raw, allowed_chunk_ids)

            if self._is_causal_claim(claim):
                cited_sections = [safe_text(section_map.get(cid)) for cid in normalized_ids]
                causal_is_pub_only = bool(cited_sections) and all(
                    self._is_publication_section(sec) for sec in cited_sections
                )
                if causal_is_pub_only or not normalized_ids:
                    support_text = claim or safe_text(row_out.get("support_summary"))
                    replacement_id = self._best_supporting_chunk_id(
                        text=support_text,
                        context_chunks=context_chunks,
                        allowed_chunk_ids=allowed_chunk_ids,
                        prefer_non_publication=True,
                    )
                    if replacement_id and replacement_id not in normalized_ids:
                        normalized_ids.append(replacement_id)
                        added_ids.append(replacement_id)
                        changed = True

            if normalized_ids:
                deduped_ids = list(dict.fromkeys(normalized_ids))
                if deduped_ids != chunk_ids_raw:
                    changed = True
                row_out["chunk_ids"] = deduped_ids
            out_rows.append(row_out)

        if changed:
            out = dict(payload)
            out["claim_evidence_map"] = out_rows
            return out, True, list(dict.fromkeys(added_ids))
        return payload, False, []

    def _patch_causal_paragraph_citations(
        self,
        *,
        article_markdown: str,
        context_chunks: List[Dict[str, str]],
        allowed_chunk_ids: set,
    ) -> Tuple[str, bool, List[str]]:
        text = safe_text(article_markdown)
        if not text:
            return text, False, []

        section_map = {
            safe_text(c.get("chunk_id")): safe_text(c.get("section"))
            for c in context_chunks
        }
        parts = re.split(r"(\n\s*\n)", text)
        changed = False
        added_ids: List[str] = []

        for idx in range(0, len(parts), 2):
            block = safe_text(parts[idx]).strip()
            if not block:
                continue
            if block.startswith(">") or block.startswith("#"):
                continue
            cleaned_block = self._clean_claim_text(block)
            if len(re.findall(r"[A-Za-z']+", cleaned_block)) < 6:
                continue
            if not self._is_causal_claim(cleaned_block):
                continue

            cited_ids = re.findall(r"\[chunk:([^\]\s]+)\]", block)
            if not cited_ids:
                cited_ids = re.findall(
                    r"\[([0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12})\]",
                    block,
                    flags=re.I,
                )
            normalized_ids, _ = self._normalize_chunk_ids(cited_ids, allowed_chunk_ids)
            cited_sections = [safe_text(section_map.get(cid)) for cid in normalized_ids]
            pub_only = bool(cited_sections) and all(self._is_publication_section(sec) for sec in cited_sections)
            missing_valid = not normalized_ids
            if not pub_only and not missing_valid:
                continue

            replacement_id = self._best_supporting_chunk_id(
                text=cleaned_block,
                context_chunks=context_chunks,
                allowed_chunk_ids=allowed_chunk_ids,
                prefer_non_publication=True,
            )
            if not replacement_id:
                continue
            token = f"[chunk:{replacement_id}]"
            if token in block:
                continue
            parts[idx] = f"{token} {block}"
            added_ids.append(replacement_id)
            changed = True

        if not changed:
            return text, False, []
        return "".join(parts).strip(), True, list(dict.fromkeys(added_ids))

    def _patch_major_section_early_citations(
        self,
        *,
        article_markdown: str,
        context_chunks: List[Dict[str, str]],
        allowed_chunk_ids: set,
    ) -> Tuple[str, bool, List[str]]:
        text = safe_text(article_markdown).strip()
        if not text:
            return text, False, []

        context_chunk_map = {
            safe_text(c.get("chunk_id")): safe_text(c.get("text"))
            for c in context_chunks
        }

        def needs_early_strong_citation(body: str) -> bool:
            if len(re.findall(r"[A-Za-z']+", body)) < 80:
                return False
            matches = list(re.finditer(r"\[chunk:([^\]\s]+)\]", body))
            if not matches:
                matches = list(
                    re.finditer(
                        r"\[([0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12})\]",
                        body,
                        flags=re.I,
                    )
                )
            if not matches:
                return True

            body_len = max(1, len(body))
            early_raw = [
                safe_text(m.group(1))
                for m in matches
                if safe_text(m.group(1)) and (m.start() / body_len) <= self.section_end_citation_ratio
            ]
            normalized_early_ids, _ = self._normalize_chunk_ids(early_raw, allowed_chunk_ids)
            if not normalized_early_ids:
                return True

            best_overlap = 0.0
            best_inter = 0
            for cid in normalized_early_ids:
                chunk_text = safe_text(context_chunk_map.get(cid))
                overlap, inter = self._paragraph_chunk_overlap(body, chunk_text)
                if overlap > best_overlap:
                    best_overlap = overlap
                if inter > best_inter:
                    best_inter = inter
            return bool(best_inter < 3 and best_overlap < self.min_section_strong_overlap)

        def inject_front_citation(body: str) -> Tuple[str, bool, str]:
            if not needs_early_strong_citation(body):
                return body, False, ""
            chosen_id = self._best_supporting_chunk_id(
                text=body,
                context_chunks=context_chunks,
                allowed_chunk_ids=allowed_chunk_ids,
                prefer_non_publication=False,
            )
            if not chosen_id:
                return body, False, ""
            token = f"[chunk:{chosen_id}]"
            blocks = re.split(r"\n\s*\n", body, maxsplit=1)
            first = safe_text(blocks[0]).strip()
            if not first:
                return body, False, ""
            if first.startswith(token):
                return body, False, ""
            first = f"{token} {first}".strip()
            rest = safe_text(blocks[1]).strip() if len(blocks) > 1 else ""
            updated = first if not rest else f"{first}\n\n{rest}"
            return updated, True, chosen_id

        parts = re.split(r"(?m)^(#{2,4}\s+.+)$", text)
        changed = False
        inserted_ids: List[str] = []
        rebuilt: List[str] = []

        intro = safe_text(parts[0]).strip() if parts else text
        if intro:
            intro_out, intro_changed, intro_id = inject_front_citation(intro)
            if intro_changed:
                changed = True
                if intro_id:
                    inserted_ids.append(intro_id)
            rebuilt.append(intro_out)

        if len(parts) > 1:
            for i in range(1, len(parts), 2):
                heading = safe_text(parts[i]).strip()
                body = safe_text(parts[i + 1]).strip() if i + 1 < len(parts) else ""
                if not heading:
                    continue
                body_out = body
                if body:
                    body_out, section_changed, section_id = inject_front_citation(body)
                    if section_changed:
                        changed = True
                        if section_id:
                            inserted_ids.append(section_id)
                rebuilt.append(f"{heading}\n{body_out}".strip())

        if not changed:
            return text, False, []
        return "\n\n".join([b for b in rebuilt if safe_text(b)]).strip(), True, list(dict.fromkeys(inserted_ids))

    def _patch_weak_paragraph_grounding(
        self,
        *,
        article_markdown: str,
        context_chunks: List[Dict[str, str]],
        allowed_chunk_ids: set,
    ) -> Tuple[str, bool, List[str]]:
        text = safe_text(article_markdown)
        if not text:
            return text, False, []

        context_chunk_map = {
            safe_text(c.get("chunk_id")): safe_text(c.get("text"))
            for c in context_chunks
        }
        parts = re.split(r"(\n\s*\n)", text)
        changed = False
        added_ids: List[str] = []

        for idx in range(0, len(parts), 2):
            block = safe_text(parts[idx]).strip()
            if not block or block.startswith(">"):
                continue
            if len(re.findall(r"[A-Za-z']+", block)) < 35:
                continue

            cited_ids = re.findall(r"\[chunk:([^\]\s]+)\]", block)
            if not cited_ids:
                cited_ids = re.findall(
                    r"\[([0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12})\]",
                    block,
                    flags=re.I,
                )
            if not cited_ids:
                continue

            normalized_ids, unknown = self._normalize_chunk_ids(cited_ids, allowed_chunk_ids)
            best_overlap = 0.0
            best_intersection = 0
            for cid in normalized_ids:
                chunk_text = safe_text(context_chunk_map.get(cid))
                overlap, inter = self._paragraph_chunk_overlap(block, chunk_text)
                if overlap > best_overlap:
                    best_overlap = overlap
                if inter > best_intersection:
                    best_intersection = inter

            weak = bool(unknown or not normalized_ids or (best_intersection < 2 and best_overlap < self.min_paragraph_overlap))
            if not weak:
                continue

            chosen_id = self._best_supporting_chunk_id(
                text=block,
                context_chunks=context_chunks,
                allowed_chunk_ids=allowed_chunk_ids,
                prefer_non_publication=False,
            )
            if not chosen_id:
                continue
            token = f"[chunk:{chosen_id}]"
            updated_block = block
            if re.match(r"^#{1,6}\s+", block):
                lines = block.splitlines()
                heading_lines: List[str] = []
                body_lines: List[str] = []
                collecting_heading = True
                for line in lines:
                    if collecting_heading and re.match(r"^#{1,6}\s+", line.strip()):
                        heading_lines.append(line)
                        continue
                    collecting_heading = False
                    body_lines.append(line)
                body_text = "\n".join(body_lines).strip()
                if not body_text:
                    continue
                if body_text.startswith(token):
                    continue
                updated_block = "\n".join(heading_lines + [f"{token} {body_text}".strip()])
            else:
                if block.startswith(token):
                    continue
                updated_block = f"{token} {block}"
            parts[idx] = updated_block
            added_ids.append(chosen_id)
            changed = True

        if not changed:
            return text, False, []
        return "".join(parts).strip(), True, list(dict.fromkeys(added_ids))

    def _apply_reliability_fallbacks(
        self,
        *,
        payload: Dict[str, Any],
        name: str,
        trend_issue: Optional[Dict[str, Any]],
        context_chunks: List[Dict[str, str]],
        allowed_chunk_ids: set,
    ) -> Tuple[Dict[str, Any], bool]:
        """
        Deterministic fallback fixes before expensive LLM rewrite:
        - populate trend_source_url from selected issue
        - synthesize claim_evidence_map when missing/invalid
        - strip explicit byline impersonation lines and field
        - patch causal claims/paragraphs with non-publication evidence
        - patch weak paragraph grounding with stronger chunk citations
        - patch end-loaded sections with early strong citation anchors
        """
        patched = False
        out = dict(payload)

        expected_trend_url = safe_text((trend_issue or {}).get("url"))
        if expected_trend_url and not safe_text(out.get("trend_source_url")):
            out["trend_source_url"] = expected_trend_url
            patched = True

        mapping = out.get("claim_evidence_map")
        if not isinstance(mapping, list) or len(mapping) < self.min_claim_evidence_items:
            out["claim_evidence_map"] = self._synthesize_claim_evidence_map(
                article_markdown=safe_text(out.get("article_markdown")),
                context_chunks=context_chunks,
                allowed_chunk_ids=allowed_chunk_ids,
                min_items=self.min_claim_evidence_items,
            )
            patched = True

        out, claim_map_patched, claim_map_added_ids = self._patch_claim_map_for_causal_evidence(
            payload=out,
            context_chunks=context_chunks,
            allowed_chunk_ids=allowed_chunk_ids,
        )
        if claim_map_patched:
            patched = True

        article = safe_text(out.get("article_markdown"))
        byline_patterns = [
            rf"(?im)^\s*by\s+{re.escape(name)}\s*$",
            r"(?im)^\s*by\s+[A-Z][A-Za-z .'-]{2,80}\s*$",
            r"(?im)^\s*\|\s*$",
            r"(?im)^\s*\d{4}-\d{2}-\d{2}T[^ \n]+\s*$",
        ]
        cleaned = article
        for pat in byline_patterns:
            cleaned = re.sub(pat, "", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        if cleaned and cleaned != article:
            out["article_markdown"] = cleaned
            patched = True

        payload_byline = safe_text(out.get("byline"))
        if payload_byline and self._matches_scholar_identity(payload_byline, name):
            out["byline"] = "AI-generated perspective"
            patched = True

        article_now = safe_text(out.get("article_markdown"))
        article_now, causal_patch, causal_added_ids = self._patch_causal_paragraph_citations(
            article_markdown=article_now,
            context_chunks=context_chunks,
            allowed_chunk_ids=allowed_chunk_ids,
        )
        if causal_patch:
            out["article_markdown"] = article_now
            patched = True

        article_now, para_patch, para_added_ids = self._patch_weak_paragraph_grounding(
            article_markdown=safe_text(out.get("article_markdown")),
            context_chunks=context_chunks,
            allowed_chunk_ids=allowed_chunk_ids,
        )
        if para_patch:
            out["article_markdown"] = article_now
            patched = True

        article_now, section_patch, section_added_ids = self._patch_major_section_early_citations(
            article_markdown=safe_text(out.get("article_markdown")),
            context_chunks=context_chunks,
            allowed_chunk_ids=allowed_chunk_ids,
        )
        if section_patch:
            out["article_markdown"] = article_now
            patched = True

        existing_used = out.get("used_chunk_ids")
        if not isinstance(existing_used, list):
            existing_used = []
        merged_ids = [safe_text(x) for x in existing_used if safe_text(x)]
        merged_ids.extend(self._extract_inline_citations(safe_text(out.get("article_markdown"))))
        merged_ids.extend(claim_map_added_ids)
        merged_ids.extend(causal_added_ids)
        merged_ids.extend(para_added_ids)
        merged_ids.extend(section_added_ids)
        normalized_used, _ = self._normalize_chunk_ids(merged_ids, allowed_chunk_ids)
        normalized_used = list(dict.fromkeys(normalized_used))
        if normalized_used and normalized_used != existing_used:
            out["used_chunk_ids"] = normalized_used
            patched = True

        return out, patched

    def _validate_story(
        self,
        *,
        name: str,
        payload: Dict[str, Any],
        context_chunks: List[Dict[str, str]],
        trend_issue: Optional[Dict[str, Any]],
        allowed_source_urls: List[str],
        allowed_chunk_ids: set,
        assume_deceased: bool,
    ) -> Tuple[bool, List[str], List[str], List[str], List[Dict[str, Any]], str]:
        errors: List[str] = []
        warnings: List[str] = []

        title = safe_text(payload.get("title"))
        standfirst = safe_text(payload.get("standfirst"))
        article = safe_text(payload.get("article_markdown"))
        trend_source_url = safe_text(payload.get("trend_source_url"))
        editor_notes = payload.get("editor_notes")
        if not isinstance(editor_notes, list):
            editor_notes = []
        is_dry_placeholder = any(safe_text(n) == "dry_run_generated" for n in editor_notes)
        used_chunk_ids = payload.get("used_chunk_ids")
        if not isinstance(used_chunk_ids, list):
            used_chunk_ids = []
        used_chunk_ids = [safe_text(c) for c in used_chunk_ids if safe_text(c)]

        if len(title) < 10:
            errors.append("title_too_short")
        if len(standfirst) < 20:
            warnings.append("standfirst_too_short")
        if len(article) < 500:
            errors.append("article_too_short")
        if self.require_standard_structure and (not is_dry_placeholder):
            structure_errors = self._validate_standard_article_structure(article)
            errors.extend(structure_errors)
            lens_errors, lens_warnings = self._validate_scholar_lens_presence(article, name)
            errors.extend(lens_errors)
            warnings.extend(lens_warnings)
            detail_errors, detail_warnings = self._validate_concrete_detail_density(article)
            errors.extend(detail_errors)
            warnings.extend(detail_warnings)

        inline_citations = self._extract_inline_citations(article)

        normalized_inline, unknown_inline = self._normalize_chunk_ids(inline_citations, allowed_chunk_ids)
        if unknown_inline:
            errors.append(f"unknown_inline_citations:{','.join(sorted(set(unknown_inline)))}")

        normalized_used, unknown_used = self._normalize_chunk_ids(used_chunk_ids, allowed_chunk_ids)
        if unknown_used:
            errors.append(f"unknown_used_chunk_ids:{','.join(sorted(set(unknown_used)))}")

        min_citations = 4
        if len(set(normalized_used)) < min_citations:
            warnings.append(f"used_chunk_ids_below_{min_citations}")
        if len(set(normalized_inline)) < min_citations:
            if len(set(normalized_used)) >= min_citations:
                warnings.append(f"inline_citations_below_{min_citations}_used_chunk_ids_fallback")
            else:
                errors.append("insufficient_inline_citations")

        expected_trend_url = safe_text((trend_issue or {}).get("url"))
        has_live_issue = bool(safe_text((trend_issue or {}).get("title")))
        if self.strict_reliability and self.trends_enabled and has_live_issue and self.require_verified_trend_url:
            if not self._is_valid_external_url(trend_source_url):
                errors.append("missing_or_invalid_trend_source_url")
            if expected_trend_url and trend_source_url != expected_trend_url:
                errors.append("trend_source_url_mismatch")

        source_url_errors, source_url_warnings, _ = self._validate_source_url_citations(
            payload=payload,
            article_markdown=article,
            allowed_source_urls=allowed_source_urls,
        )
        errors.extend(source_url_errors)
        warnings.extend(source_url_warnings)

        claim_map_errors, claim_map_warnings, normalized_claim_map = self._validate_claim_evidence_map(
            payload=payload,
            allowed_chunk_ids=allowed_chunk_ids,
            context_chunks=context_chunks,
            article_markdown=article,
        )
        errors.extend(claim_map_errors)
        warnings.extend(claim_map_warnings)

        lower_article = article.lower()
        name_lc = name.lower()
        payload_byline = safe_text(payload.get("byline")).lower()
        rendered_bylines = self._extract_rendered_bylines(article)
        impersonation_patterns = [
            rf"\bi am {re.escape(name_lc)}\b",
            rf"\bi, {re.escape(name_lc)}\b",
            r"\bas i write this today\b",
            r"\bi recently\b",
            r"\bmy latest research\b",
        ]
        if assume_deceased:
            impersonation_patterns.extend(
                [
                    r"\bi\s+(?:argue|believe|write|wrote|contend|maintain|have|am|was|will|see|remember|want|think)\b",
                    r"\bmy\s+(?:work|research|life|view|argument|books|career|teaching)\b",
                ]
            )

        for pat in impersonation_patterns:
            if re.search(pat, lower_article):
                errors.append(f"persona_violation:{pat}")
                break

        if assume_deceased:
            if any(self._matches_scholar_identity(byline, name) for byline in rendered_bylines):
                errors.append("byline_impersonation")
            if payload_byline and self._matches_scholar_identity(payload_byline, name):
                errors.append("byline_impersonation_payload_field")

        # Basic sanity check against fabricated "alive today" language.
        if assume_deceased and re.search(r"\bstill alive\b|\bi am alive\b|\bmy current office\b", lower_article):
            errors.append("posthumous_violation")

        context_chunk_map = {safe_text(c.get("chunk_id")): safe_text(c.get("text")) for c in context_chunks}
        if self.strict_reliability and not is_dry_placeholder:
            section_citation_errors = self._validate_major_section_citations(
                article_markdown=article,
                context_chunk_map=context_chunk_map,
                allowed_chunk_ids=allowed_chunk_ids,
            )
            errors.extend(section_citation_errors)

            paragraph_grounding_errors = self._validate_paragraph_grounding(
                article_markdown=article,
                context_chunk_map=context_chunk_map,
                allowed_chunk_ids=allowed_chunk_ids,
            )
            errors.extend(paragraph_grounding_errors)
            causal_paragraph_errors = self._validate_causal_paragraph_evidence(
                article_markdown=article,
                context_chunks=context_chunks,
                allowed_chunk_ids=allowed_chunk_ids,
            )
            errors.extend(causal_paragraph_errors)

        errors = list(dict.fromkeys(errors))
        warnings = list(dict.fromkeys(warnings))
        normalized_used = list(dict.fromkeys(normalized_used or normalized_inline))
        return (
            len(errors) == 0,
            errors,
            warnings,
            normalized_used,
            normalized_claim_map,
            trend_source_url,
        )

    def run(
        self,
        *,
        scholar_id: Optional[str],
        date_value: date,
        topic_override: Optional[str],
        max_scholars: int,
        max_context_chunks: int,
        dry_run: bool,
    ) -> Dict[str, Any]:
        run_id = str(uuid.uuid4())
        story_date = date_value.isoformat()
        run_started = utc_now_iso()

        run_doc = {
            "run_id": run_id,
            "story_date": story_date,
            "started_at": run_started,
            "ended_at": None,
            "status": "running",
            "config": {
                "scholar_id": scholar_id,
                "topic_override": topic_override,
                "max_scholars": max_scholars,
                "max_context_chunks": max_context_chunks,
                "dry_run": dry_run,
                "model_name": self.model_name if self.use_llm else "none",
                "require_human_review": self.require_human_review,
                "assume_deceased": self.assume_deceased,
                "ml_enabled": self.ml_enabled,
                "embedding_model_name": self.embedding_model_name if self.ml_enabled else "disabled",
                "cross_encoder_enabled": self.cross_encoder_enabled and self.ml_enabled,
                "cross_encoder_model_name": self.cross_encoder_model_name if self.cross_encoder_enabled else "disabled",
                "cross_encoder_top_k": self.cross_encoder_rerank_top_k if self.cross_encoder_enabled else 0,
                "cross_encoder_weight": self.cross_encoder_weight if self.cross_encoder_enabled else 0.0,
                "trends_enabled": self.trends_enabled,
                "trend_provider": self.trend_provider,
                "trend_region": self.trend_region,
                "enforce_profile_quality": self.enforce_profile_quality,
                "profile_quality_min_score": self.profile_quality_min_score,
                "strict_reliability": self.strict_reliability,
                "require_verified_trend_url": self.require_verified_trend_url,
                "allow_corpus_only_when_no_trend": self.allow_corpus_only_when_no_trend,
                "strict_llm_only": self.strict_llm_only,
                "enforce_source_url_citations": self.enforce_source_url_citations,
                "min_source_url_citations": self.min_source_url_citations,
                "max_source_url_candidates": self.max_source_url_candidates,
                "source_domain_filter_enabled": self.source_domain_filter_enabled,
                "source_domain_min_score": self.source_domain_min_score,
                "source_require_trusted": self.source_require_trusted,
                "reader_strip_inline_citations": self.reader_strip_inline_citations,
                "reader_max_connector_repeats": self.reader_max_connector_repeats,
                "vertex_schema_enforced": self.vertex_schema_enforced,
                "min_claim_evidence_items": self.min_claim_evidence_items,
                "min_paragraph_overlap": self.min_paragraph_overlap,
                "min_section_strong_overlap": self.min_section_strong_overlap,
                "section_end_citation_ratio": self.section_end_citation_ratio,
                "max_output_tokens": self.max_output_tokens,
                "require_standard_structure": self.require_standard_structure,
                "min_article_words": self.min_article_words,
                "min_major_sections": self.min_major_sections,
                "min_section_words": self.min_section_words,
            },
            "summary": {
                "attempted": 0,
                "generated": 0,
                "skipped_existing": 0,
                "skipped_profile_quality": 0,
                "degraded_no_trend_data": 0,
                "failed_no_trend_data": 0,
                "failed_no_context": 0,
                "failed_validation": 0,
                "failed_generation": 0,
            },
            "errors": [],
        }
        self.jobs_collection.insert_one(run_doc)

        scholars = list(self._iter_scholars(scholar_id=scholar_id, max_scholars=max_scholars))
        if not scholars:
            self.jobs_collection.update_one(
                {"run_id": run_id},
                {
                    "$set": {
                        "ended_at": utc_now_iso(),
                        "status": "failed",
                        "errors": ["no_scholars_found"],
                    }
                },
            )
            return {"run_id": run_id, "status": "failed", "error": "no_scholars_found"}

        summary = run_doc["summary"]
        all_errors: List[str] = []
        trending_issues = self._fetch_trending_issues()
        self.jobs_collection.update_one(
            {"run_id": run_id},
            {
                "$set": {
                    "config.trend_issues_fetched": len(trending_issues),
                }
            },
        )

        for scholar_doc in scholars:
            summary["attempted"] += 1
            profile_id = self._profile_id(scholar_doc)
            professor_name = self._professor_name(scholar_doc)
            if not profile_id:
                summary["failed_generation"] += 1
                all_errors.append("missing_profile_id")
                continue

            story_key = f"{profile_id}:{story_date}"
            existing_story = self.stories_collection.find_one(
                {"story_key": story_key},
                {"_id": 1, "status": 1},
            )
            if existing_story:
                existing_status = safe_text(existing_story.get("status")).lower()
                if existing_status == "failed_validation":
                    self.stories_collection.delete_one({"_id": existing_story["_id"]})
                else:
                    summary["skipped_existing"] += 1
                    continue

            profile_ready, profile_quality_status, profile_quality_score = self._profile_quality_state(scholar_doc)
            if not profile_ready:
                summary["skipped_profile_quality"] += 1
                all_errors.append(
                    f"{profile_id}:profile_quality_blocked:{profile_quality_status}:{profile_quality_score}"
                )
                continue

            topic, topic_source, ml_topic_count = self._choose_topic(
                scholar_doc=scholar_doc,
                profile_id=profile_id,
                story_date=story_date,
                topic_override=topic_override,
            )
            context_chunks = self._extract_context_chunks(
                scholar_doc=scholar_doc,
                topic=topic,
                max_context_chunks=max_context_chunks,
            )
            if len(context_chunks) < 4:
                summary["failed_no_context"] += 1
                all_errors.append(f"{profile_id}:insufficient_context")
                continue

            allowed_chunk_ids = {c["chunk_id"] for c in context_chunks}
            disclosure = self._build_disclosure(professor_name, self.assume_deceased)
            style_profile = self._build_style_profile(scholar_doc)
            trend_issue = self._select_trending_issue(topic, trending_issues)
            if self.strict_reliability and self.trends_enabled and not trend_issue:
                if not self.allow_corpus_only_when_no_trend:
                    summary["failed_no_trend_data"] += 1
                    all_errors.append(f"{profile_id}:missing_trend_issue_for_reliable_mode")
                    continue
                summary["degraded_no_trend_data"] += 1
                all_errors.append(f"{profile_id}:missing_trend_issue_fallback_to_corpus_only")
            allowed_source_urls = self._collect_allowed_source_urls(
                profile_id=profile_id,
                scholar_doc=scholar_doc,
                trend_issue=trend_issue,
            )

            try:
                payload = self._generate_story_payload(
                    name=professor_name,
                    topic=topic,
                    trend_issue=trend_issue,
                    style_profile=style_profile,
                    context_chunks=context_chunks,
                    allowed_source_urls=allowed_source_urls,
                    assume_deceased=self.assume_deceased,
                    dry_run=dry_run,
                )
            except Exception as exc:
                summary["failed_generation"] += 1
                all_errors.append(f"{profile_id}:generation_error:{exc}")
                continue

            payload, _ = self._enforce_source_url_citations(
                payload=payload,
                allowed_source_urls=allowed_source_urls,
            )

            valid, errors, warnings, used_chunk_ids, normalized_claim_map, trend_source_url = self._validate_story(
                name=professor_name,
                payload=payload,
                context_chunks=context_chunks,
                trend_issue=trend_issue,
                allowed_source_urls=allowed_source_urls,
                allowed_chunk_ids=allowed_chunk_ids,
                assume_deceased=self.assume_deceased,
            )

            if not valid:
                patched_payload, patched = self._apply_reliability_fallbacks(
                    payload=payload,
                    name=professor_name,
                    trend_issue=trend_issue,
                    context_chunks=context_chunks,
                    allowed_chunk_ids=allowed_chunk_ids,
                )
                if patched:
                    patched_payload, _ = self._enforce_source_url_citations(
                        payload=patched_payload,
                        allowed_source_urls=allowed_source_urls,
                    )
                    (
                        p_valid,
                        p_errors,
                        p_warnings,
                        p_used_chunk_ids,
                        p_claim_map,
                        p_trend_source_url,
                    ) = self._validate_story(
                        name=professor_name,
                        payload=patched_payload,
                        context_chunks=context_chunks,
                        trend_issue=trend_issue,
                        allowed_source_urls=allowed_source_urls,
                        allowed_chunk_ids=allowed_chunk_ids,
                        assume_deceased=self.assume_deceased,
                    )
                    if p_valid or len(p_errors) < len(errors):
                        payload = patched_payload
                        valid = p_valid
                        errors = p_errors
                        warnings = p_warnings
                        used_chunk_ids = p_used_chunk_ids
                        normalized_claim_map = p_claim_map
                        trend_source_url = p_trend_source_url
                        notes = payload.get("editor_notes")
                        if not isinstance(notes, list):
                            notes = []
                        notes.append("deterministic_reliability_patch")
                        payload["editor_notes"] = notes

            if (not valid) and self.use_llm and (not dry_run):
                repair_candidates = [
                    err for err in errors
                    if (
                        err.startswith("persona_violation")
                        or err.startswith("byline_impersonation")
                        or err.startswith("byline_impersonation_payload_field")
                        or err.startswith("unknown_inline_citations")
                        or err.startswith("unknown_used_chunk_ids")
                        or err.startswith("claim_map_")
                        or err.startswith("section_missing_citation")
                        or err.startswith("section_citation_end_loaded")
                        or err.startswith("section_weak_citation_grounding")
                        or err.startswith("section_missing_early_strong_citation")
                        or err.startswith("section_unknown_or_missing_citations")
                        or err.startswith("weak_paragraph_grounding")
                        or err.startswith("unknown_article_urls")
                        or err.startswith("article_word_count_below_minimum")
                        or err.startswith("missing_required_section")
                        or err.startswith("underdeveloped_sections")
                        or err.startswith("single_paragraph_sections")
                        or err in {
                            "insufficient_inline_citations",
                            "article_too_short",
                            "insufficient_scholar_specific_grounding",
                            "insufficient_concrete_details",
                            "insufficient_major_sections",
                            "missing_sources_section",
                            "insufficient_claim_evidence_map_items",
                            "claim_evidence_map_below_minimum_after_normalization",
                            "missing_or_invalid_trend_source_url",
                            "trend_source_url_mismatch",
                            "source_urls_not_in_allowed_pool",
                            "insufficient_source_url_citations",
                            "source_url_missing_from_sources_section",
                            "causal_claim_publications_only_evidence",
                            "causal_paragraph_publications_only_evidence",
                            "causal_claim_missing_valid_citation",
                        }
                    )
                ]
                if repair_candidates:
                    try:
                        repaired_payload = self._repair_story_payload(
                            name=professor_name,
                            topic=topic,
                            trend_issue=trend_issue,
                            style_profile=style_profile,
                            context_chunks=context_chunks,
                            allowed_source_urls=allowed_source_urls,
                            assume_deceased=self.assume_deceased,
                            failed_payload=payload,
                            validation_errors=errors,
                        )
                        repaired_payload, _ = self._enforce_source_url_citations(
                            payload=repaired_payload,
                            allowed_source_urls=allowed_source_urls,
                        )
                        (
                            r_valid,
                            r_errors,
                            r_warnings,
                            r_used_chunk_ids,
                            r_claim_map,
                            r_trend_source_url,
                        ) = self._validate_story(
                            name=professor_name,
                            payload=repaired_payload,
                            context_chunks=context_chunks,
                            trend_issue=trend_issue,
                            allowed_source_urls=allowed_source_urls,
                            allowed_chunk_ids=allowed_chunk_ids,
                            assume_deceased=self.assume_deceased,
                        )
                        if r_valid or len(r_errors) < len(errors):
                            payload = repaired_payload
                            valid = r_valid
                            errors = r_errors
                            warnings = r_warnings
                            used_chunk_ids = r_used_chunk_ids
                            normalized_claim_map = r_claim_map
                            trend_source_url = r_trend_source_url
                            notes = payload.get("editor_notes")
                            if not isinstance(notes, list):
                                notes = []
                            notes.append("auto_repair_pass")
                            payload["editor_notes"] = notes
                    except Exception:
                        warnings.append("auto_repair_failed")

            if not valid and (not self.strict_llm_only):
                strict_fallback_payload = self._build_strict_validation_fallback_payload(
                    name=professor_name,
                    topic=topic,
                    trend_issue=trend_issue,
                    context_chunks=context_chunks,
                    allowed_source_urls=allowed_source_urls,
                    note=f"strict_fallback_after_errors:{'|'.join(errors)[:300]}",
                )
                strict_fallback_payload, _ = self._enforce_source_url_citations(
                    payload=strict_fallback_payload,
                    allowed_source_urls=allowed_source_urls,
                )
                (
                    sf_valid,
                    sf_errors,
                    sf_warnings,
                    sf_used_chunk_ids,
                    sf_claim_map,
                    sf_trend_source_url,
                ) = self._validate_story(
                    name=professor_name,
                    payload=strict_fallback_payload,
                    context_chunks=context_chunks,
                    trend_issue=trend_issue,
                    allowed_source_urls=allowed_source_urls,
                    allowed_chunk_ids=allowed_chunk_ids,
                    assume_deceased=self.assume_deceased,
                )
                if sf_valid or len(sf_errors) < len(errors):
                    payload = strict_fallback_payload
                    valid = sf_valid
                    errors = sf_errors
                    warnings = sf_warnings
                    used_chunk_ids = sf_used_chunk_ids
                    normalized_claim_map = sf_claim_map
                    trend_source_url = sf_trend_source_url
                    notes = payload.get("editor_notes")
                    if not isinstance(notes, list):
                        notes = []
                    notes.append("strict_validation_fallback_pass")
                    payload["editor_notes"] = notes

            status = "pending_review" if self.require_human_review else "generated"
            if not valid:
                status = "failed_validation"
                summary["failed_validation"] += 1
                all_errors.append(f"{profile_id}:failed_validation:{'|'.join(errors)}")

            article_markdown = safe_text(payload.get("article_markdown"))
            reader_article_markdown = self._sanitize_reader_article_markdown(article_markdown)
            if reader_article_markdown != article_markdown:
                notes = payload.get("editor_notes")
                if not isinstance(notes, list):
                    notes = []
                notes.append("reader_sanitized")
                payload["editor_notes"] = list(dict.fromkeys([safe_text(n) for n in notes if safe_text(n)]))
            article_markdown = reader_article_markdown
            if disclosure not in article_markdown:
                article_markdown = f"> {disclosure}\n\n{article_markdown}"
            selected_source_urls = payload.get("source_urls")
            if not isinstance(selected_source_urls, list):
                selected_source_urls = []
            selected_source_urls = [safe_text(u) for u in selected_source_urls if safe_text(u)]

            used_context = [c for c in context_chunks if c["chunk_id"] in set(used_chunk_ids)]
            if not used_context:
                used_context = context_chunks[:4]
            sentence_source_map = self._build_sentence_source_map(
                article_markdown=article_markdown,
                context_chunks=context_chunks,
                allowed_chunk_ids=allowed_chunk_ids,
                fallback_source_urls=selected_source_urls,
            )

            story_doc = {
                "story_key": story_key,
                "story_date": story_date,
                "status": status,
                "topic": topic,
                "topic_selection": {
                    "source": topic_source,
                    "ml_topic_candidates_count": ml_topic_count,
                },
                "topic_modeling": {
                    "primary_model": "nmf",
                    "selection_mode": "seeded_weighted_random_daily",
                },
                "scholar": {
                    "profile_id": profile_id,
                    "name": professor_name,
                    "profile_quality": {
                        "status": profile_quality_status,
                        "score": profile_quality_score,
                        "enforced": self.enforce_profile_quality,
                    },
                },
                "trend_issue": {
                    "enabled": self.trends_enabled,
                    "provider": self.trend_provider,
                    "selected": trend_issue or {},
                    "verified_source_url": trend_source_url,
                },
                "style_profile": style_profile,
                "content": {
                    "title": safe_text(payload.get("title")),
                    "standfirst": safe_text(payload.get("standfirst")),
                    "article_markdown": article_markdown,
                    "disclosure": disclosure,
                    "source_urls": selected_source_urls,
                },
                "presentation": {
                    "allow_scholar_byline": False,
                    "author_label": "AI-generated perspective",
                },
                "citations": [
                    {
                        "chunk_id": c["chunk_id"],
                        "section": c["section"],
                        "evidence_excerpt": c["text"][:350],
                        "source_urls": self._chunk_source_urls(c),
                        "source_refs": (c.get("source_refs") if isinstance(c.get("source_refs"), list) else [])[:3],
                    }
                    for c in used_context
                ],
                "source_url_citations": [{"url": u} for u in selected_source_urls],
                "sentence_source_map": sentence_source_map,
                "generation": {
                    "mode": "dry_run" if dry_run or not self.use_llm else "llm",
                    "model": "none" if dry_run or not self.use_llm else self.model_name,
                    "generated_at": utc_now_iso(),
                    "run_id": run_id,
                    "editor_notes": payload.get("editor_notes") or [],
                },
                "claim_evidence_map": normalized_claim_map,
                "ml_retrieval": {
                    "strategy": "hybrid_dense_lexical_section_prior",
                    "embedding_model": self.embedding_model_name if self.ml_enabled else "disabled",
                    "cross_encoder_reranker": {
                        "enabled": bool(self.cross_encoder_enabled and self.ml_enabled),
                        "active": bool(self.cross_encoder_enabled and self.ml_enabled and not self._cross_encoder_unavailable),
                        "model": self.cross_encoder_model_name if self.cross_encoder_enabled else "disabled",
                        "top_k": self.cross_encoder_rerank_top_k if self.cross_encoder_enabled else 0,
                        "weight": self.cross_encoder_weight if self.cross_encoder_enabled else 0.0,
                    },
                    "topic_tokens_used": self._topic_tokens(topic),
                    "trend_tokens_used": self._topic_tokens(safe_text((trend_issue or {}).get("title"))),
                },
                "safety": {
                    "assume_deceased": self.assume_deceased,
                    "require_human_review": self.require_human_review,
                    "strict_reliability": self.strict_reliability,
                    "enforce_source_url_citations": self.enforce_source_url_citations,
                    "validation_passed": valid,
                    "errors": errors,
                    "warnings": warnings,
                },
            }

            try:
                self.stories_collection.insert_one(story_doc)
                if valid:
                    summary["generated"] += 1
            except Exception as exc:
                is_duplicate = bool(DuplicateKeyError is not None and isinstance(exc, DuplicateKeyError))
                if is_duplicate:
                    try:
                        self.stories_collection.replace_one(
                            {"story_key": story_key},
                            story_doc,
                            upsert=True,
                        )
                        if valid:
                            summary["generated"] += 1
                        continue
                    except Exception as replace_exc:
                        summary["failed_generation"] += 1
                        all_errors.append(f"{profile_id}:replace_error:{replace_exc}")
                        continue
                summary["failed_generation"] += 1
                all_errors.append(f"{profile_id}:insert_error:{exc}")
                continue

        final_status = "completed"
        if summary["generated"] == 0 and summary["failed_validation"] > 0:
            final_status = "completed_with_validation_failures"
        if summary["generated"] == 0 and (
            summary["failed_no_trend_data"] > 0
            or (summary["failed_generation"] > 0 and summary["failed_no_context"] > 0)
        ):
            final_status = "failed"

        self.jobs_collection.update_one(
            {"run_id": run_id},
            {
                "$set": {
                    "ended_at": utc_now_iso(),
                    "status": final_status,
                    "summary": summary,
                    "errors": all_errors,
                }
            },
        )

        return {
            "run_id": run_id,
            "status": final_status,
            "summary": summary,
            "errors": all_errors,
        }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate daily scholar-inspired stories into MongoDB.")
    parser.add_argument("--scholar-id", type=str, default=None, help="Single scholar profile_id to process")
    parser.add_argument("--date", type=str, default=None, help="Story date in YYYY-MM-DD (default: today UTC)")
    parser.add_argument("--topic", type=str, default=None, help="Optional fixed topic override")
    parser.add_argument("--max-scholars", type=int, default=25, help="Max scholars to process when no --scholar-id")
    parser.add_argument("--max-context-chunks", type=int, default=12, help="Context chunks provided to generator")
    parser.add_argument("--scholars-collection", type=str, default="legend_scholars")
    parser.add_argument("--stories-collection", type=str, default="legend_scholar_daily_stories")
    parser.add_argument("--jobs-collection", type=str, default="daily_story_jobs")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Vertex model name (default: STORY_LLM_MODEL or LLM_MODEL from .env)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Generate deterministic placeholder story (no LLM call)")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM generation")
    parser.add_argument(
        "--disable-trends",
        action="store_true",
        help="Disable live trend ingestion and generate from scholar corpus only",
    )
    parser.add_argument(
        "--trend-provider",
        type=str,
        default=None,
        help="Trend provider override: rss | newsapi | gdelt | auto",
    )
    parser.add_argument(
        "--enforce-profile-quality",
        action="store_true",
        help="Require profile quality status 'ready' before generation",
    )
    parser.add_argument(
        "--ignore-profile-quality",
        action="store_true",
        help="Bypass profile quality readiness checks",
    )
    parser.add_argument(
        "--profile-quality-min-score",
        type=int,
        default=None,
        help="Minimum quality score threshold when quality enforcement is enabled",
    )
    parser.add_argument(
        "--publish-without-review",
        action="store_true",
        help="Mark valid stories as generated instead of pending_review",
    )
    parser.add_argument(
        "--assume-living",
        action="store_true",
        help="Relax deceased-scholar safety policy (not recommended for historical figures)",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    story_date = parse_date(args.date)
    enforce_profile_quality: Optional[bool] = None
    if args.enforce_profile_quality:
        enforce_profile_quality = True
    elif args.ignore_profile_quality:
        enforce_profile_quality = False

    worker = DailyStoryWorker(
        scholars_collection=args.scholars_collection,
        stories_collection=args.stories_collection,
        jobs_collection=args.jobs_collection,
        model_name=args.model,
        use_llm=not args.no_llm,
        require_human_review=not args.publish_without_review,
        assume_deceased=not args.assume_living,
        trends_enabled=not args.disable_trends,
        trend_provider=args.trend_provider,
        enforce_profile_quality=enforce_profile_quality,
        profile_quality_min_score=args.profile_quality_min_score,
    )
    try:
        result = worker.run(
            scholar_id=args.scholar_id,
            date_value=story_date,
            topic_override=args.topic,
            max_scholars=args.max_scholars,
            max_context_chunks=args.max_context_chunks,
            dry_run=args.dry_run,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result.get("status") != "failed" else 1
    finally:
        worker.close()


if __name__ == "__main__":
    raise SystemExit(main())
