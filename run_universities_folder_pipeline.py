"""
Resumable universities-folder ingestion runner.

Scans Excel workbooks under a directory, collects high-confidence scholar-owned
URLs for each row, seeds the scholar document into MongoDB immediately, then
runs the existing full pipeline one scholar at a time so MongoDB and Pinecone
are updated during the run instead of only at the end.

Design goals:
- Re-runnable against the same folder as more Excel files are added
- Stable profile IDs so MongoDB/Pinecone converge on one record per scholar
- Per-workbook checkpoints so new rows are processed and unchanged rows are skipped
- Strict URL identity checks to avoid contaminating the corpus with the wrong person

Usage:
    python run_universities_folder_pipeline.py
    python run_universities_folder_pipeline.py --aggressive
    python run_universities_folder_pipeline.py --workbook-glob "*Minnesota*.xlsx"
    python run_universities_folder_pipeline.py --limit-scholars 25 --dry-run
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

import httpx
import pandas as pd
from dotenv import load_dotenv

from api.utils.source_guardrails import is_noise_domain
from config.mongodb_utils import create_mongo_client, resolve_mongo_db_name
from unified_pipeline import UnifiedPipeline


DEFAULT_INPUT_DIR = "universities"
DEFAULT_OUTPUT_ROOT = "output/universities"
DEFAULT_MAX_URLS = 20
DEFAULT_PER_QUERY_RESULTS = 12
DEFAULT_SLEEP_SECONDS = 2.0
DEFAULT_HTTP_TIMEOUT_SECONDS = 25.0
DEFAULT_HTTP_RETRIES = 4

SKIP_DOMAINS = {
    "facebook.com",
    "twitter.com",
    "x.com",
    "instagram.com",
    "tiktok.com",
    "pinterest.com",
    "reddit.com",
    "quora.com",
    "linkedin.com",
    "youtube.com",
    "amazon.com",
    "wikipedia.org",
}

SKIP_EXACT_HOSTS = {
    "google.com",
    "www.google.com",
    "duckduckgo.com",
    "www.duckduckgo.com",
    "bing.com",
    "www.bing.com",
}

IDENTITY_DOMAINS = {
    "orcid.org",
    "scholar.google.com",
    "semanticscholar.org",
    "openalex.org",
    "researchgate.net",
    "dblp.org",
    "loop.frontiersin.org",
}

IDENTITY_PROFILE_URL_RULES: Dict[str, Tuple[str, ...]] = {
    "orcid.org": ("/0000-",),
    "scholar.google.com": ("/citations", "citations?user="),
    "semanticscholar.org": ("/author/",),
    "openalex.org": ("/authors/",),
    "researchgate.net": ("/profile/",),
    "dblp.org": ("/pid/", "/pers/hd/"),
    "loop.frontiersin.org": ("/people/", "/researcher/"),
}

BAD_RESULT_PATTERNS = (
    "obituary",
    "memorial",
    "funeral",
    "tributes",
    "legacy.com",
    "findagrave",
    "athletics",
    "sports",
)

NON_PERSON_NAME_TOKENS = {
    "program",
    "office",
    "department",
    "school",
    "center",
    "centre",
    "committee",
    "services",
}

BLOCKED_CONTENT_PATTERNS = (
    "request unsuccessful",
    "incapsula incident id",
    "access denied",
    "temporarily unavailable",
    "forbidden",
    "the system can't perform the operation now",
)

GENERIC_LISTING_TOKENS = {
    "faculty",
    "staff",
    "directory",
    "directories",
    "team",
    "people",
    "news",
    "profiles",
    "our-team",
}

URL_TOKEN_IGNORE = {
    "http",
    "https",
    "www",
    "edu",
    "org",
    "com",
    "net",
    "profile",
    "profiles",
    "people",
    "person",
    "directory",
    "citations",
    "user",
    "hl",
    "oi",
    "ao",
    "roleurl",
    "language",
    "lang",
    "id",
}

UNIVERSITY_STOPWORDS = {
    "the",
    "of",
    "and",
    "at",
    "for",
    "main",
    "campus",
    "university",
    "college",
    "school",
    "institute",
    "system",
    "cities",
    "twin",
}

EMPTY_MARKERS = {"", "na", "n/a", "nan", "none", "null"}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_event(stage: str, message: str, *, indent: int = 0) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    pad = " " * indent
    print(f"[{timestamp}] {pad}[{stage}] {message}", flush=True)


def configure_utf8_stdio() -> None:
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("PYTHONUTF8", "1")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def clean_cell(value: Any) -> str:
    if value is None:
        return ""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in EMPTY_MARKERS:
        return ""
    return text


def slugify(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", (text or "").strip().lower()).strip("-")
    return slug or "item"


def domain_of(url: str) -> str:
    try:
        netloc = (urlparse(url).netloc or "").lower()
    except Exception:
        return ""
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return netloc


def root_domain(host: str) -> str:
    parts = [part for part in (host or "").split(".") if part]
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return host or ""


def normalize_url(url: str) -> str:
    try:
        parsed = urlparse((url or "").strip())
    except Exception:
        return (url or "").strip()
    scheme = (parsed.scheme or "https").lower()
    netloc = (parsed.netloc or "").lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    path = (parsed.path or "/").rstrip("/") or "/"
    kept_query_parts: List[str] = []
    if parsed.query:
        for kv in parsed.query.split("&"):
            key = kv.split("=", 1)[0].lower()
            if key.startswith(("utm_", "fbclid", "gclid", "mc_", "ref")):
                continue
            kept_query_parts.append(kv)
    query = "&".join(kept_query_parts)
    suffix = f"?{query}" if query else ""
    return f"{scheme}://{netloc}{path}{suffix}"


def source_path_token(url: str) -> str:
    try:
        path = (urlparse(url).path or "").strip("/")
    except Exception:
        return ""
    if not path:
        return ""
    parts = [part for part in path.split("/") if part]
    return parts[-1].lower() if parts else ""


def is_identity_profile_url(url: str) -> bool:
    try:
        parsed = urlparse((url or "").strip())
    except Exception:
        return False
    host = domain_of(url)
    full = f"{(parsed.path or '').lower()}?{(parsed.query or '').lower()}"
    for domain, allowed_fragments in IDENTITY_PROFILE_URL_RULES.items():
        if host == domain or host.endswith("." + domain):
            return any(fragment in full for fragment in allowed_fragments)
    return False


def url_looks_scholar_owned(
    url: str,
    identity: Dict[str, str],
    *,
    title: str = "",
    snippet: str = "",
) -> bool:
    host = domain_of(url)
    if not host:
        return False
    if any(host == item or host.endswith("." + item) for item in IDENTITY_DOMAINS):
        if not is_identity_profile_url(url):
            return False
        if title or snippet:
            return strong_identity_text_match(" ".join([title or "", snippet or ""]), identity) and not result_conflicts_with_identity(
                {"title": title, "snippet": snippet, "url": ""},
                identity,
            )
        return True

    url_lc = url.lower()
    text_blob = normalize_name(" ".join([title or "", snippet or "", url or ""]))
    last = normalize_name(identity.get("last", ""))
    first = normalize_name(identity.get("first", ""))
    full_variants = identity_variants(identity)
    source_token = normalize_name(identity.get("source_path_token", ""))
    url_compact = re.sub(r"[^a-z0-9]+", "", url_lc)
    if any(variant and re.sub(r"[^a-z0-9]+", "", variant) in url_compact for variant in full_variants):
        return True
    if source_token and source_token.replace(" ", "") in url_compact:
        return True
    if any(variant and variant in text_blob for variant in full_variants):
        return True
    if first and last and first in url_lc and last in url_lc:
        return True
    if first and last and re.search(rf"\b{re.escape(first)}\b.*\b{re.escape(last)}\b", text_blob):
        return True
    if last and (url_lc.endswith(".pdf") or "/cv/" in url_lc or "/vita" in url_lc):
        return last in url_lc and bool(first in text_blob or any(variant and variant in text_blob for variant in full_variants))
    return False


def should_skip_url(url: str) -> bool:
    if not url or not url.startswith(("http://", "https://")):
        return True
    host = domain_of(url)
    if not host or host in SKIP_EXACT_HOSTS:
        return True
    if is_noise_domain(url):
        identity_override = any(host == item or host.endswith("." + item) for item in IDENTITY_DOMAINS)
        if not identity_override:
            return True
    for item in SKIP_DOMAINS:
        if host == item or host.endswith("." + item):
            return True
    return len(url) > 700


def url_priority(url: str, identity: Dict[str, str]) -> int:
    host = domain_of(url)
    source_host = identity.get("source_host", "")
    official_host = identity.get("official_host", "")
    url_lc = url.lower()
    if source_host and (host == source_host or host.endswith("." + source_host)):
        return 0
    if official_host and (host == official_host or host.endswith("." + official_host)):
        return 1
    if url_lc.endswith((".pdf", ".doc", ".docx")) or "/cv/" in url_lc or "curriculum-vitae" in url_lc or "resume" in url_lc:
        return 2
    if host.endswith(".edu"):
        return 3
    if host.endswith(".gov"):
        return 4
    if host == "orcid.org" or host.endswith(".orcid.org"):
        return 5
    if host == "scholar.google.com" or host.endswith(".scholar.google.com"):
        return 25
    if host == "researchgate.net" or host.endswith(".researchgate.net"):
        return 30
    if host == "semanticscholar.org" or host.endswith(".semanticscholar.org"):
        return 35
    if host == "openalex.org" or host.endswith(".openalex.org"):
        return 40
    if any(host == item or host.endswith("." + item) for item in IDENTITY_DOMAINS):
        return 20
    if host.endswith(".org"):
        return 45
    return 100


def normalize_name(name: str) -> str:
    text = (name or "").strip().lower()
    text = re.sub(r"^(dr\.?|prof\.?|professor)\s+", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def looks_like_person_name(name: str) -> bool:
    parts = [part for part in re.findall(r"[A-Za-z]+", (name or "").strip()) if part]
    if len(parts) < 2:
        return False
    lowered = {part.lower() for part in parts}
    if lowered.intersection(NON_PERSON_NAME_TOKENS):
        return False
    return True


def stable_profile_id(record: Dict[str, Any], workbook_slug: str) -> str:
    for key in ("Scholar Profile ID", "Object ID", "profile_id", "object_id"):
        value = clean_cell(record.get(key))
        if value:
            return value
    payload = "|".join(
        [
            workbook_slug,
            clean_cell(record.get("Name")),
            clean_cell(record.get("Email")),
            clean_cell(record.get("Source URL")),
            clean_cell(record.get("University")),
        ]
    )
    return str(uuid.uuid5(uuid.NAMESPACE_URL, payload))


def row_signature(record: Dict[str, Any], workbook_slug: str) -> str:
    payload = {
        "workbook": workbook_slug,
        "object_id": clean_cell(record.get("Object ID")),
        "name": clean_cell(record.get("Name")),
        "email": clean_cell(record.get("Email")),
        "source_url": clean_cell(record.get("Source URL")),
        "keywords": clean_cell(record.get("Keywords")),
        "expertise": clean_cell(record.get("Expertise")),
        "university": clean_cell(record.get("University")),
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha1(encoded.encode("utf-8")).hexdigest()


def normalized_tokens(text: str, *, min_len: int = 3) -> List[str]:
    return [tok for tok in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(tok) >= min_len]


def name_variants(name: str) -> List[str]:
    raw = (name or "").strip()
    if not raw:
        return []
    variants = [raw]
    no_periods = re.sub(r"\.", "", raw)
    if no_periods and no_periods != raw:
        variants.append(no_periods)
    compact_spaces = re.sub(r"\s+", " ", no_periods).strip()
    if compact_spaces and compact_spaces not in variants:
        variants.append(compact_spaces)
    parts = [part for part in compact_spaces.split() if part]
    if len(parts) >= 2:
        first = parts[0]
        last = parts[-1]
        if len(first) == 2 and first.isalpha():
            variants.append(f"{first[0]} {last}")
        if len(first) == 1 and first.isalpha():
            variants.append(f"{first} {last}")
    deduped: List[str] = []
    seen = set()
    for variant in variants:
        key = variant.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(variant)
    return deduped


def normalize_workbook(df: pd.DataFrame) -> pd.DataFrame:
    rename_map: Dict[str, str] = {}
    for col in df.columns:
        col_text = str(col).strip()
        col_lc = col_text.lower()
        if col_lc == "source":
            rename_map[col] = "Source URL"
        elif col_lc == "profile_url":
            rename_map[col] = "Source URL"
        elif col_lc == "name":
            rename_map[col] = "Name"
        elif col_lc == "email":
            rename_map[col] = "Email"
        elif col_lc == "keywords":
            rename_map[col] = "Keywords"
        elif col_lc == "expertise":
            rename_map[col] = "Expertise"
        elif col_lc == "university":
            rename_map[col] = "University"
        elif col_lc == "object id":
            rename_map[col] = "Object ID"
        elif col_lc == "scholar profile id":
            rename_map[col] = "Scholar Profile ID"
    df = df.rename(columns=rename_map).copy()
    for required in ("Name", "Email", "Source URL", "Keywords", "Expertise", "University", "Object ID"):
        if required not in df.columns:
            df[required] = ""
    df["_excel_row_number"] = df.index + 2
    return df


def university_tokens(university: str) -> List[str]:
    tokens = re.findall(r"[a-z0-9]{3,}", (university or "").lower())
    return [tok for tok in tokens if tok not in UNIVERSITY_STOPWORDS]


def identity_keywords(record: Dict[str, Any]) -> Dict[str, str]:
    name = clean_cell(record.get("Name"))
    parts = [part for part in name.split() if part]
    email = clean_cell(record.get("Email")).lower()
    email_local = email.split("@", 1)[0] if "@" in email else ""
    email_domain = email.split("@", 1)[1] if "@" in email else ""
    source_url_value = clean_cell(record.get("Source URL"))
    source_host = domain_of(source_url_value)
    official_host = email_domain or root_domain(source_host)
    return {
        "name": name,
        "first": parts[0] if parts else "",
        "last": parts[-1] if parts else "",
        "email": email,
        "email_local": email_local,
        "email_domain": email_domain,
        "source_url": source_url_value,
        "source_host": source_host,
        "source_path_token": source_path_token(source_url_value),
        "official_host": official_host,
        "university": clean_cell(record.get("University")),
        "keywords": clean_cell(record.get("Keywords")),
        "expertise": clean_cell(record.get("Expertise")),
    }


class RetryHttpJsonClient:
    def __init__(self, *, timeout_seconds: float, max_retries: int, user_agent: str):
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = max(1, int(max_retries))
        self.client = httpx.Client(
            timeout=self.timeout_seconds,
            follow_redirects=True,
            headers={
                "User-Agent": user_agent,
                "Accept": "application/json",
            },
        )

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:
            pass

    def get_json(
        self,
        url: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        last_error: Optional[BaseException] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.get(url, params=params, headers=extra_headers)
                if response.status_code == 429:
                    retry_after = response.headers.get("Retry-After")
                    delay = float(retry_after) if retry_after and retry_after.isdigit() else min(8.0, 1.5 * attempt)
                    time.sleep(delay)
                    continue
                response.raise_for_status()
                payload = response.json()
                return payload if isinstance(payload, dict) else {}
            except Exception as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    raise
                time.sleep(min(8.0, 1.2 * (2 ** (attempt - 1))))
        if last_error is not None:
            raise last_error
        return {}


def overlap_score(left: Iterable[str], right: Iterable[str]) -> int:
    left_set = {item for item in left if item}
    right_set = {item for item in right if item}
    return len(left_set.intersection(right_set))


class OpenAlexAuthorResolver:
    def __init__(self, http_client: RetryHttpJsonClient, api_key: str):
        self.http_client = http_client
        self.api_key = api_key

    def _request(self, path: str, *, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = dict(params or {})
        payload["api_key"] = self.api_key
        return self.http_client.get_json(f"https://api.openalex.org{path}", params=payload)

    @staticmethod
    def _extract_openalex_urls(author: Dict[str, Any]) -> List[str]:
        urls: List[str] = []
        openalex_id = str(author.get("id") or "").strip()
        if openalex_id:
            urls.append(openalex_id)

        ids = author.get("ids") or {}
        if isinstance(ids, dict):
            for value in ids.values():
                url = str(value or "").strip()
                if not url.startswith(("http://", "https://")):
                    continue
                host = domain_of(url)
                if any(host == item or host.endswith("." + item) for item in IDENTITY_DOMAINS):
                    urls.append(url)
        return urls

    @staticmethod
    def _author_identity_score(author: Dict[str, Any], identity: Dict[str, str]) -> Tuple[int, Dict[str, Any]]:
        display_name = str(author.get("display_name") or "").strip()
        display_name_norm = normalize_name(display_name)
        target_name_norm = normalize_name(identity.get("name", ""))
        first = identity.get("first", "").lower()
        last = identity.get("last", "").lower()
        exact_name = int(bool(display_name_norm and display_name_norm == target_name_norm))
        first_last = int(bool(first and last and first in display_name.lower() and last in display_name.lower()))

        institution_names: List[str] = []
        for inst in author.get("last_known_institutions") or []:
            if not isinstance(inst, dict):
                continue
            institution_names.append(str(inst.get("display_name") or ""))

        institution_token_overlap = max(
            [overlap_score(university_tokens(identity.get("university", "")), normalized_tokens(name)) for name in institution_names] or [0]
        )

        ids = author.get("ids") or {}
        orcid_url = str(ids.get("orcid") or "").strip() if isinstance(ids, dict) else ""
        orcid_present = int(bool(orcid_url))
        works_count = int(author.get("works_count") or 0)

        score = (
            exact_name * 100
            + first_last * 30
            + min(institution_token_overlap, 5) * 15
            + orcid_present * 10
            + min(works_count, 200) // 20
        )
        diagnostics = {
            "display_name": display_name,
            "institution_names": institution_names[:5],
            "exact_name": bool(exact_name),
            "first_last": bool(first_last),
            "institution_token_overlap": institution_token_overlap,
            "orcid_present": bool(orcid_present),
            "works_count": works_count,
        }
        return score, diagnostics

    def resolve(self, identity: Dict[str, str]) -> Dict[str, Any]:
        name = identity.get("name", "").strip()
        if not name:
            return {"matched": False, "urls": [], "meta": {}}

        params = {
            "search": name,
            "per_page": 10,
            "select": "id,display_name,ids,last_known_institutions,works_count,cited_by_count",
        }
        payload = self._request("/authors", params=params)
        results = payload.get("results") or []
        if not isinstance(results, list):
            return {"matched": False, "urls": [], "meta": {}}

        scored: List[Tuple[int, Dict[str, Any], Dict[str, Any]]] = []
        for author in results:
            if not isinstance(author, dict):
                continue
            score, diagnostics = self._author_identity_score(author, identity)
            if score < 100:
                continue
            scored.append((score, author, diagnostics))

        if not scored:
            return {"matched": False, "urls": [], "meta": {"candidate_count": len(results)}}

        scored.sort(key=lambda item: item[0], reverse=True)
        best_score, author, diagnostics = scored[0]
        urls = []
        seen = set()
        for url in self._extract_openalex_urls(author):
            norm = normalize_url(url)
            if not norm or norm in seen or should_skip_url(norm):
                continue
            seen.add(norm)
            urls.append(norm)

        meta = {
            "provider": "openalex",
            "score": best_score,
            "candidate_count": len(results),
            "matched_author_id": str(author.get("id") or ""),
            "matched_display_name": diagnostics.get("display_name", ""),
            "matched_institutions": diagnostics.get("institution_names", []),
            "orcid_present": diagnostics.get("orcid_present", False),
            "works_count": diagnostics.get("works_count", 0),
            "cited_by_count": int(author.get("cited_by_count") or 0),
        }
        return {"matched": True, "urls": urls, "meta": meta}


def build_queries(identity: Dict[str, str], aggressive: bool = False) -> List[str]:
    name = identity.get("name", "").strip()
    if not name:
        return []
    variants = name_variants(name)
    primary_name = variants[0]
    base = f'"{primary_name}"'
    university = identity.get("university", "").strip()
    university_phrase = f'"{university}"' if university else ""
    source_host = identity.get("source_host", "").strip()
    official_host = identity.get("official_host", "").strip()
    source_path_token = identity.get("source_path_token", "").strip()
    email_local = identity.get("email_local", "").strip()
    keywords = identity.get("keywords", "").strip()
    expertise = identity.get("expertise", "").strip()

    queries: List[str] = []
    if university_phrase:
        queries.extend(
            [
                f"{base} {university_phrase}",
                f"{base} {university_phrase} faculty profile",
                f"{base} {university_phrase} biography",
                f"{base} {university_phrase} publications research",
                f"{base} {university_phrase} cv OR curriculum vitae",
            ]
        )
    else:
        queries.extend(
            [
                f"{base} faculty profile",
                f"{base} biography",
                f"{base} publications research",
            ]
        )

    if source_host:
        queries.append(f"{base} site:{source_host}")
    if official_host and official_host != source_host:
        queries.append(f"{base} site:{official_host}")
    if source_path_token:
        token_query = f'"{source_path_token}"'
        queries.append(f"{token_query} {base}")
        if source_host:
            queries.append(f"{token_query} site:{source_host}")
        if official_host and official_host != source_host:
            queries.append(f"{token_query} site:{official_host}")
        if university_phrase:
            queries.append(f"{token_query} {university_phrase}")
        queries.append(f"{token_query} filetype:pdf")
    if email_local:
        queries.append(f'"{email_local}" {base}')
        if university_phrase:
            queries.append(f'"{email_local}" {base} {university_phrase}')
    if keywords:
        queries.append(f"{base} {keywords[:80]}")
    if expertise:
        queries.append(f"{base} {expertise[:80]}")

    queries.extend(
        [
            f"{base} site:orcid.org",
            f"{base} site:scholar.google.com",
            f"{base} site:semanticscholar.org",
            f"{base} site:openalex.org",
            f"{base} site:researchgate.net",
            f"{base} site:dblp.org",
            f"{base} filetype:pdf",
            f"{base} cv filetype:pdf",
        ]
    )

    if aggressive:
        extra = [
            f"{base} lab OR laboratory OR research group",
            f"{base} awards honors grants",
            f"{base} interview profile news",
            f"{base} site:pubmed.ncbi.nlm.nih.gov",
            f"{base} site:arxiv.org",
        ]
        if university_phrase:
            extra.extend(
                [
                    f"{base} {university_phrase} department directory",
                    f"{base} {university_phrase} google scholar",
                    f"{base} {university_phrase} orcid",
                ]
            )
        queries.extend(extra)

    # Initialed names often need punctuation-free variants to avoid SERP drift
    # toward single-letter results.
    for alt_name in variants[1:]:
        alt_base = f'"{alt_name}"'
        if university_phrase:
            queries.append(f"{alt_base} {university_phrase}")
            queries.append(f"{alt_base} {university_phrase} faculty profile")
        queries.append(f"{alt_base} site:orcid.org")
        queries.append(f"{alt_base} site:scholar.google.com")

    seen = set()
    deduped: List[str] = []
    for query in queries:
        query = re.sub(r"\s+", " ", query).strip()
        if not query:
            continue
        key = query.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(query)
    return deduped


def result_has_bad_pattern(result: Dict[str, str]) -> bool:
    text_blob = " ".join(
        [
            (result.get("title") or "").lower(),
            (result.get("snippet") or "").lower(),
            (result.get("url") or "").lower(),
        ]
    )
    return any(pattern in text_blob for pattern in BAD_RESULT_PATTERNS)


def university_signal(blob: str, url_lc: str, university: str) -> bool:
    university = (university or "").strip().lower()
    if university and university in blob:
        return True
    tokens = university_tokens(university)
    matched = sum(1 for tok in tokens if tok in blob or tok in url_lc)
    return matched >= 2 if len(tokens) >= 2 else matched >= 1


def keyword_signal(blob: str, identity: Dict[str, str]) -> bool:
    for field in ("keywords", "expertise"):
        raw = identity.get(field, "")
        tokens = [tok for tok in re.findall(r"[a-z0-9]{5,}", raw.lower()) if tok not in UNIVERSITY_STOPWORDS]
        for tok in tokens[:8]:
            if tok in blob:
                return True
    return False


def first_name_token_matches(token: str, target_first: str) -> bool:
    token = re.sub(r"[^a-z]", "", (token or "").lower())
    target_first = re.sub(r"[^a-z]", "", (target_first or "").lower())
    if not token or not target_first:
        return False
    if token == target_first:
        return True
    return len(token) == 1 and token == target_first[:1]


def identity_variants(identity: Dict[str, str]) -> List[str]:
    raw_name = identity.get("name", "")
    variants = [normalize_name(item) for item in name_variants(raw_name)]
    deduped: List[str] = []
    seen = set()
    for variant in variants:
        if not variant or variant in seen:
            continue
        seen.add(variant)
        deduped.append(variant)
    return deduped


def strong_identity_text_match(text: str, identity: Dict[str, str]) -> bool:
    text_norm = normalize_name(text)
    if not text_norm:
        return False

    variants = identity_variants(identity)
    if any(variant and variant in text_norm for variant in variants):
        return True

    first = normalize_name(identity.get("first", ""))
    last = normalize_name(identity.get("last", ""))
    if first and last and re.search(rf"\b{re.escape(first)}\b.*\b{re.escape(last)}\b", text_norm):
        return True

    name_parts = [part for part in variants[0].split() if part] if variants else []
    if len(name_parts) >= 2:
        initials = " ".join([p[:1] for p in name_parts[:-1]] + [name_parts[-1]])
        if initials and initials in text_norm:
            return True
    return False


def _tokens_near_last_conflict(tokens: List[str], *, target_first: str, target_last: str) -> bool:
    for idx, tok in enumerate(tokens):
        if tok != target_last:
            continue
        neighbors: List[str] = []
        for neighbor_idx in (idx - 1, idx + 1):
            if neighbor_idx < 0 or neighbor_idx >= len(tokens):
                continue
            neighbors.append(tokens[neighbor_idx])
        if any(first_name_token_matches(neighbor, target_first) for neighbor in neighbors):
            continue
        for neighbor in neighbors:
            if len(neighbor) < 3 or neighbor in UNIVERSITY_STOPWORDS or neighbor in GENERIC_LISTING_TOKENS or neighbor in URL_TOKEN_IGNORE:
                continue
            return True
    return False


def result_conflicts_with_identity(result: Dict[str, str], identity: Dict[str, str]) -> bool:
    first = normalize_name(identity.get("first", ""))
    last = normalize_name(identity.get("last", ""))
    if not first or not last:
        return False

    texts = [
        str(result.get("title") or ""),
        str(result.get("snippet") or ""),
        str(result.get("url") or ""),
    ]
    for text in texts:
        lowered_tokens = [tok for tok in re.findall(r"[a-z]+", text.lower()) if tok]
        if _tokens_near_last_conflict(lowered_tokens, target_first=first, target_last=last):
            return True
    return False


def url_conflicts_with_identity(url: str, identity: Dict[str, str]) -> bool:
    first = normalize_name(identity.get("first", ""))
    last = normalize_name(identity.get("last", ""))
    if not first or not last:
        return False
    try:
        parsed = urlparse((url or "").strip())
    except Exception:
        return False
    path_text = (parsed.path or "").replace("/", " ")
    query_values: List[str] = []
    if parsed.query:
        for item in parsed.query.split("&"):
            if "=" in item:
                _, value = item.split("=", 1)
            else:
                value = item
            if value:
                query_values.append(value)
    candidate_text = " ".join([path_text] + query_values)
    lowered_tokens = [
        tok
        for tok in re.findall(r"[a-z]+", candidate_text.lower())
        if tok and tok not in URL_TOKEN_IGNORE
    ]
    return _tokens_near_last_conflict(lowered_tokens, target_first=first, target_last=last)


def is_generic_listing_url(url: str, identity: Dict[str, str]) -> bool:
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    path = (parsed.path or "").lower()
    query = (parsed.query or "").lower()
    tokens = [tok for tok in re.findall(r"[a-z0-9]+", path) if tok]
    last = normalize_name(identity.get("last", ""))
    if last and last in tokens:
        return False
    if identity.get("source_path_token") and identity["source_path_token"].lower() in path:
        return False
    if any(tok in GENERIC_LISTING_TOKENS for tok in tokens):
        return True
    return "page=" in query


def identity_match(result: Dict[str, str], identity: Dict[str, str]) -> bool:
    if result_has_bad_pattern(result):
        return False
    if result_conflicts_with_identity(result, identity):
        return False

    title = result.get("title", "") or ""
    snippet = result.get("snippet", "") or ""
    url = result.get("url", "") or ""
    blob = f"{title} {snippet}".lower()
    url_lc = url.lower()
    domain = domain_of(url)

    raw_name = identity.get("name", "")
    name = raw_name.lower()
    first = identity.get("first", "").lower()
    last = identity.get("last", "").lower()
    email_local = identity.get("email_local", "").lower()
    source_host = identity.get("source_host", "").lower()
    official_host = identity.get("official_host", "").lower()

    if not name or not last:
        return False

    name_compact = re.sub(r"[^a-z0-9]+", "", name)
    url_compact = re.sub(r"[^a-z0-9]+", "", url_lc)

    variant_norms = [normalize_name(item) for item in name_variants(raw_name)]
    strong_name_signal = bool(name in blob or (name_compact and name_compact in url_compact))
    if not strong_name_signal:
        for variant in variant_norms[1:]:
            variant_compact = re.sub(r"[^a-z0-9]+", "", variant)
            if variant and (variant in blob or (variant_compact and variant_compact in url_compact)):
                strong_name_signal = True
                break
    first_last_blob = bool(first and first in blob and last in blob)
    first_last_url = bool(first and first in url_lc and last in url_lc)
    first_initial_last_blob = bool(first and last and first[:1] in blob and last in blob)
    first_initial_last_url = bool(first and last and first[:1] in url_lc and last in url_lc)
    path_token_match = bool(identity.get("source_path_token") and identity["source_path_token"] in url_lc)

    base_name_signal = (
        strong_name_signal
        or first_last_blob
        or first_last_url
        or first_initial_last_blob
        or first_initial_last_url
        or path_token_match
    )
    if not base_name_signal:
        return False

    email_signal = bool(email_local and len(email_local) >= 4 and (email_local in blob or email_local in url_lc))
    uni_signal = university_signal(blob, url_lc, identity.get("university", ""))
    kw_signal = keyword_signal(blob, identity)
    source_host_signal = bool(source_host and (domain == source_host or domain.endswith("." + source_host)))
    official_host_signal = bool(official_host and (domain == official_host or domain.endswith("." + official_host)))
    identity_domain_signal = any(domain == item or domain.endswith("." + item) for item in IDENTITY_DOMAINS)

    if source_host_signal or official_host_signal:
        return strong_name_signal or first_last_blob or first_last_url or first_initial_last_blob or first_initial_last_url

    if identity_domain_signal:
        return (strong_name_signal or first_last_blob or first_last_url or first_initial_last_blob or first_initial_last_url) and (uni_signal or email_signal or kw_signal)

    if domain.endswith(".edu") or domain.endswith(".gov"):
        return (strong_name_signal or first_last_blob or first_last_url or first_initial_last_blob or first_initial_last_url) and (uni_signal or email_signal or kw_signal)

    return False


async def ddg_search(query: str, max_results: int) -> List[Dict[str, str]]:
    try:
        from ddgs import DDGS  # type: ignore
    except ImportError:
        try:
            from duckduckgo_search import DDGS  # type: ignore
        except ImportError:
            raise RuntimeError("Install DDG search support with: pip install ddgs")

    loop = asyncio.get_event_loop()
    try:
        results = await loop.run_in_executor(None, lambda: list(DDGS().text(query, max_results=max_results)))
    except Exception as exc:
        log_event("DDG", f"query failed: {query!r} :: {exc}", indent=2)
        return []

    output: List[Dict[str, str]] = []
    for item in results:
        url = item.get("href") or item.get("url") or ""
        if not url:
            continue
        output.append(
            {
                "url": url,
                "title": item.get("title", "") or "",
                "snippet": item.get("body", "") or "",
            }
        )
    return output


async def collect_urls_for_record(
    record: Dict[str, Any],
    *,
    max_urls: int,
    per_query_results: int,
    sleep_seconds: float,
    aggressive: bool,
    openalex_result: Optional[Dict[str, Any]] = None,
    existing_urls: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    identity = identity_keywords(record)
    queries = build_queries(identity, aggressive=aggressive)
    log_event(
        "Collect",
        (
            f"identity name={identity.get('name', '')!r} "
            f"university={identity.get('university', '')!r} "
            f"source_host={identity.get('source_host', '')!r} "
            f"queries={len(queries)}"
        ),
        indent=2,
    )
    if not queries and not identity.get("source_url"):
        return {
            "queries": [],
            "raw_results": 0,
            "rejected_count": 0,
            "kept_urls": [],
            "results_detail": [],
        }

    candidates: Dict[str, Dict[str, Any]] = {}
    raw_results = 0
    rejected = 0

    for url in existing_urls or []:
        norm = normalize_url(url)
        if not norm or should_skip_url(norm) or url_conflicts_with_identity(norm, identity):
            continue
        if not url_looks_scholar_owned(norm, identity):
            continue
        candidates[norm] = {
            "url": url,
            "normalized_url": norm,
            "title": "existing",
            "snippet": "",
            "source_query": "seed_existing",
        }

    seed_url = identity.get("source_url", "")
    if (
        seed_url
        and not should_skip_url(seed_url)
        and not url_conflicts_with_identity(seed_url, identity)
        and url_looks_scholar_owned(seed_url, identity)
    ):
        norm = normalize_url(seed_url)
        candidates[norm] = {
            "url": seed_url,
            "normalized_url": norm,
            "title": "primary_profile",
            "snippet": "",
            "source_query": "seed_source_url",
        }
        log_event("Collect", f"seed source URL accepted: {norm}", indent=4)

    for url in (openalex_result or {}).get("urls", []) or []:
        norm = normalize_url(url)
        if not norm or should_skip_url(norm) or url_conflicts_with_identity(norm, identity):
            continue
        if not url_looks_scholar_owned(norm, identity):
            continue
        if norm in candidates:
            continue
        candidates[norm] = {
            "url": norm,
            "normalized_url": norm,
            "title": "openalex_identity",
            "snippet": "",
            "source_query": "openalex_identity",
        }
    if (openalex_result or {}).get("matched"):
        log_event(
            "OpenAlex",
            f"added {len((openalex_result or {}).get('urls', []) or [])} identity URL(s) from author match",
            indent=2,
        )

    for query_idx, query in enumerate(queries, start=1):
        log_event("Search", f"{query_idx}/{len(queries)} :: {query}", indent=2)
        results = await ddg_search(query, per_query_results)
        raw_results += len(results)
        accepted_this_query = 0
        duplicate_this_query = 0
        skipped_this_query = 0
        for result in results:
            url = result.get("url", "")
            if should_skip_url(url):
                rejected += 1
                skipped_this_query += 1
                continue
            if result_conflicts_with_identity(result, identity):
                rejected += 1
                skipped_this_query += 1
                continue
            if not identity_match(result, identity):
                rejected += 1
                skipped_this_query += 1
                continue
            if is_generic_listing_url(url, identity):
                rejected += 1
                skipped_this_query += 1
                continue
            norm = normalize_url(url)
            if not url_looks_scholar_owned(
                norm,
                identity,
                title=result.get("title", ""),
                snippet=result.get("snippet", ""),
            ):
                rejected += 1
                skipped_this_query += 1
                continue
            if norm in candidates:
                duplicate_this_query += 1
                continue
            result["normalized_url"] = norm
            result["source_query"] = query
            candidates[norm] = result
            accepted_this_query += 1
        log_event(
            "Search",
            (
                f"{query_idx}/{len(queries)} results={len(results)} "
                f"accepted={accepted_this_query} duplicates={duplicate_this_query} "
                f"rejected={skipped_this_query} candidate_pool={len(candidates)}"
            ),
            indent=4,
        )
        await asyncio.sleep(sleep_seconds)

    ranked = sorted(
        candidates.values(),
        key=lambda item: (
            url_priority(item["normalized_url"], identity),
            len(item.get("title", "")),
            item["normalized_url"],
        ),
    )
    final = ranked[: max(1, max_urls)]
    log_event(
        "Collect",
        f"ranked {len(ranked)} candidates, keeping {len(final)} URL(s)",
        indent=2,
    )

    return {
        "queries": queries,
        "raw_results": raw_results,
        "rejected_count": rejected,
        "kept_urls": [item["normalized_url"] for item in final],
        "openalex_meta": (openalex_result or {}).get("meta", {}),
        "results_detail": [
            {
                "url": item["normalized_url"],
                "title": item.get("title", ""),
                "snippet": (item.get("snippet", "") or "")[:200],
                "source_query": item.get("source_query", ""),
                "priority": url_priority(item["normalized_url"], identity),
            }
            for item in final
        ],
    }


def load_checkpoint(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_checkpoint(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(path)


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def ensure_env(var_name: str) -> None:
    if not os.getenv(var_name):
        raise ValueError(f"{var_name} not found in environment variables")


def build_seed_document(
    *,
    record: Dict[str, Any],
    profile_id: str,
    workbook_name: str,
    workbook_slug: str,
    row_signature_value: str,
) -> Dict[str, Any]:
    return {
        "profile_id": profile_id,
        "object_id": clean_cell(record.get("Object ID")),
        "scholar_profile_seed_id": clean_cell(record.get("Scholar Profile ID")) or profile_id,
        "professor_name": clean_cell(record.get("Name")) or "Unknown",
        "scholar_type": "university",
        "ingestion_source": "universities_folder",
        "university_name": clean_cell(record.get("University")),
        "origin_workbook": workbook_name,
        "origin_workbook_slug": workbook_slug,
        "origin_row_number": int(record.get("_excel_row_number") or 0),
        "seed_profile_url": clean_cell(record.get("Source URL")),
        "seed_email": clean_cell(record.get("Email")),
        "seed_keywords": clean_cell(record.get("Keywords")),
        "seed_expertise": clean_cell(record.get("Expertise")),
        "row_signature": row_signature_value,
        "processing_status": "queued",
        "processing_updated_at": utc_now_iso(),
    }


class UniversitiesFolderRunner:
    def __init__(self, args: argparse.Namespace):
        load_dotenv()
        ensure_env("MONGODB_URI")
        ensure_env("OPENAI_API_KEY")
        ensure_env("PINECONE_API_KEY")

        mongodb_uri = os.getenv("MONGODB_URI")
        assert mongodb_uri is not None
        self.mongo_client = create_mongo_client(mongodb_uri)
        db_name = resolve_mongo_db_name(mongodb_uri)
        self.db = self.mongo_client[db_name]
        self.scholars = self.db.scholars
        self.image_client = None
        self.image_coll = None
        self.args = args
        user_agent = os.getenv("SCRAPER_HTTP_USER_AGENT") or "NGOAutomationScholarIngest/1.0"
        self.http_client = RetryHttpJsonClient(
            timeout_seconds=float(args.http_timeout_seconds),
            max_retries=int(args.http_retries),
            user_agent=user_agent,
        )
        image_uri = os.getenv("MONGO_ATLAS_URI", "").strip()
        if image_uri:
            self.image_client = create_mongo_client(image_uri)
            image_db_name = resolve_mongo_db_name(image_uri, default="FacultyImages")
            image_coll_name = os.getenv("MONGODB_COLLECTION_NAME", "images")
            self.image_coll = self.image_client[image_db_name][image_coll_name]
        self.openalex_api_key = os.getenv("OPENALEX_API_KEY", "").strip()
        self.openalex_resolver = (
            OpenAlexAuthorResolver(self.http_client, self.openalex_api_key)
            if self.openalex_api_key and not args.disable_openalex
            else None
        )
        self._build_pipelines: Dict[str, UnifiedPipeline] = {}
        self._sync_pipelines: Dict[str, UnifiedPipeline] = {}

    def close(self) -> None:
        self.http_client.close()
        if self.image_client is not None:
            try:
                self.image_client.close()
            except Exception:
                pass
        try:
            self.mongo_client.close()
        except Exception:
            pass

    def get_build_pipeline(self, *, workbook_root: Path, chunks_root: Path) -> UnifiedPipeline:
        key = str(workbook_root)
        pipeline = self._build_pipelines.get(key)
        if pipeline is None:
            pipeline = UnifiedPipeline(
                output_dir=str(workbook_root),
                chunking_output_dir=str(chunks_root),
                use_llm_chunking=True,
                llm_provider="openai",
                llm_model="gpt-4o-mini",
                incremental_sync_enabled=False,
            )
            self._build_pipelines[key] = pipeline
        return pipeline

    def get_sync_pipeline(self, *, workbook_root: Path, chunks_root: Path) -> UnifiedPipeline:
        key = str(workbook_root)
        pipeline = self._sync_pipelines.get(key)
        if pipeline is None:
            pipeline = UnifiedPipeline(
                output_dir=str(workbook_root),
                chunking_output_dir=str(chunks_root),
                use_llm_chunking=True,
                llm_provider="openai",
                llm_model="gpt-4o-mini",
                incremental_sync_enabled=True,
                incremental_sync_batch_size=1,
                incremental_pinecone_batch_size=50,
                incremental_skip_pinecone=False,
                incremental_skip_mongo=False,
                incremental_skip_indexes=False,
            )
            self._sync_pipelines[key] = pipeline
        return pipeline

    def compute_resume_index(
        self,
        *,
        df: pd.DataFrame,
        workbook_slug: str,
        checkpoint: Dict[str, Any],
        output_root: Path,
    ) -> int:
        for idx, (_, row) in enumerate(df.iterrows(), start=0):
            record = {col: row[col] for col in df.columns}
            profile_id = stable_profile_id(record, workbook_slug)
            signature = row_signature(record, workbook_slug)
            output_paths = self.scholar_output_paths(output_root, workbook_slug, profile_id)
            existing_checkpoint = checkpoint.get(profile_id) or {}
            done_on_disk = output_paths["chunks_file"].exists() and output_paths["chunks_file"].stat().st_size > 0
            same_signature = existing_checkpoint.get("row_signature") == signature
            should_skip = (
                self.args.skip_existing
                and not self.args.force
                and same_signature
                and (existing_checkpoint.get("status") == "done" or done_on_disk)
            )
            if not should_skip:
                return idx
        return len(df)

    def validate_extracted_profile(
        self,
        *,
        output_paths: Dict[str, Path],
        profile_id: str,
        record: Dict[str, Any],
    ) -> Dict[str, Any]:
        identity = identity_keywords(record)
        profile_json_path = output_paths["profile_dir"] / f"{profile_id}.json"
        chunks_path = output_paths["chunks_file"]
        if not profile_json_path.exists():
            return {"ok": False, "reason": "profile_json_missing"}
        if not chunks_path.exists():
            return {"ok": False, "reason": "chunks_file_missing"}

        try:
            profile_payload = json.loads(profile_json_path.read_text(encoding="utf-8"))
            chunks_payload = json.loads(chunks_path.read_text(encoding="utf-8"))
        except Exception as exc:
            return {"ok": False, "reason": f"output_read_failed:{exc}"}

        all_urls = [str(u).strip() for u in (profile_payload.get("all_urls") or []) if str(u).strip()]
        bad_urls = [u for u in all_urls if url_conflicts_with_identity(u, identity)]
        if bad_urls:
            return {"ok": False, "reason": "conflicting_urls_detected", "bad_urls": bad_urls[:5]}

        sections = chunks_payload.get("sections") or {}
        chunk_texts: List[str] = []
        chunk_count = 0
        for section_chunks in sections.values():
            if not isinstance(section_chunks, list):
                continue
            for chunk in section_chunks:
                if not isinstance(chunk, dict):
                    continue
                text = str(chunk.get("text") or "").strip()
                if not text:
                    continue
                chunk_count += 1
                if len(chunk_texts) < 8:
                    chunk_texts.append(text)
        if chunk_count <= 0:
            return {"ok": False, "reason": "no_chunks_after_extraction"}

        raw_text = str(profile_payload.get("raw_text") or "")
        clean_text = str(profile_payload.get("clean_text") or "")
        combined_validation_text = "\n".join([clean_text, raw_text] + chunk_texts[:5]).strip()
        primary_snapshot = "\n".join([clean_text[:2500], raw_text[:2500]]).strip()
        combined_lc = combined_validation_text.lower()
        if any(marker in combined_lc for marker in BLOCKED_CONTENT_PATTERNS):
            return {"ok": False, "reason": "blocked_or_error_page_content"}
        if not strong_identity_text_match(combined_validation_text, identity):
            return {"ok": False, "reason": "no_strong_identity_match_in_extracted_content"}
        scholar_only_hosts = [domain_of(u) for u in all_urls]
        if scholar_only_hosts and all(host == "scholar.google.com" or host.endswith(".scholar.google.com") for host in scholar_only_hosts):
            if chunk_count < 2 or len(clean_text.strip()) < 700:
                return {"ok": False, "reason": "scholar_only_thin_content"}
        if result_conflicts_with_identity(
            {"title": "", "snippet": primary_snapshot, "url": profile_payload.get("profile_url", "")},
            identity,
        ):
            return {"ok": False, "reason": "extracted_content_mentions_conflicting_person"}

        return {
            "ok": True,
            "reason": "verified",
            "chunk_count": chunk_count,
            "url_count": len(all_urls),
        }

    def sync_verified_profile(
        self,
        *,
        sync_pipeline: UnifiedPipeline,
        profile_id: str,
        profile_name: str,
        chunks_path: Path,
    ) -> Dict[str, Any]:
        before = sync_pipeline._get_incremental_sync_stats_snapshot()
        sync_pipeline._flush_incremental_sync_batch(
            [
                {
                    "profile_id": profile_id,
                    "profile_name": profile_name,
                    "chunks_path": str(chunks_path),
                }
            ],
            is_final_flush=True,
        )
        after = sync_pipeline._get_incremental_sync_stats_snapshot()
        return {
            "mongo_synced_delta": int(after.get("profiles_synced_mongo", 0)) - int(before.get("profiles_synced_mongo", 0)),
            "mongo_failed_delta": int(after.get("profiles_failed_mongo", 0)) - int(before.get("profiles_failed_mongo", 0)),
            "vectors_uploaded_delta": int(after.get("vectors_uploaded", 0)) - int(before.get("vectors_uploaded", 0)),
            "vectors_failed_delta": int(after.get("vectors_failed", 0)) - int(before.get("vectors_failed", 0)),
            "last_error": str(after.get("last_error") or ""),
        }

    def workbook_paths(self) -> List[Path]:
        input_dir = Path(self.args.input_dir)
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        workbooks = sorted(input_dir.glob(self.args.workbook_glob))
        if self.args.workbook_name_contains:
            needle = self.args.workbook_name_contains.lower()
            workbooks = [path for path in workbooks if needle in path.name.lower()]
        if self.args.limit_workbooks is not None:
            workbooks = workbooks[: self.args.limit_workbooks]
        return workbooks

    def upsert_seed_doc(
        self,
        *,
        record: Dict[str, Any],
        profile_id: str,
        workbook_name: str,
        workbook_slug: str,
        signature: str,
    ) -> None:
        seed_doc = build_seed_document(
            record=record,
            profile_id=profile_id,
            workbook_name=workbook_name,
            workbook_slug=workbook_slug,
            row_signature_value=signature,
        )
        self.scholars.update_one(
            {"profile_id": profile_id},
            {
                "$set": seed_doc,
                "$setOnInsert": {
                    "_id": profile_id,
                    "created_at": utc_now_iso(),
                },
            },
            upsert=True,
        )

    def apply_image_mapping(self, *, record: Dict[str, Any], profile_id: str) -> Dict[str, Any]:
        if self.image_coll is None:
            return {"status": "disabled"}

        object_id = clean_cell(record.get("Object ID")) or profile_id
        if not object_id:
            return {"status": "missing_object_id"}

        image_doc = self.image_coll.find_one(
            {"profile_id": object_id},
            {
                "profile_id": 1,
                "name": 1,
                "university": 1,
                "source_url": 1,
                "image.s3_bucket": 1,
                "image.s3_key": 1,
                "image.content_type": 1,
                "image.status": 1,
                "YOLOv8n_human_detection.has_human": 1,
            },
        )
        if not image_doc:
            self.scholars.update_one(
                {"profile_id": profile_id},
                {
                    "$set": {
                        "image_mapping.status": "image_record_not_found",
                        "image_mapping.method": "object_id_profile_id",
                        "image_mapping.image_profile_id": object_id,
                        "image_mapping.matched_at": utc_now_iso(),
                    }
                },
                upsert=True,
            )
            return {"status": "image_record_not_found", "object_id": object_id, "source_url": ""}

        image = image_doc.get("image") or {}
        s3_bucket = clean_cell(image.get("s3_bucket"))
        s3_key = clean_cell(image.get("s3_key"))
        image_status = clean_cell(image.get("status"))
        has_human = bool((image_doc.get("YOLOv8n_human_detection") or {}).get("has_human"))
        source_url = clean_cell(image_doc.get("source_url"))

        update_doc: Dict[str, Any] = {
            "image_mapping.method": "object_id_profile_id",
            "image_mapping.image_profile_id": object_id,
            "image_mapping.image_name": clean_cell(image_doc.get("name")),
            "image_mapping.image_university": clean_cell(image_doc.get("university")),
            "image_mapping.source_url": source_url,
            "image_mapping.image_status": image_status,
            "image_mapping.has_human": has_human,
            "image_mapping.content_type": clean_cell(image.get("content_type")),
            "image_mapping.matched_at": utc_now_iso(),
        }
        if s3_bucket and s3_key:
            s3_uri = f"s3://{s3_bucket}/{s3_key}"
            update_doc.update(
                {
                    "about.avatar_url": s3_uri,
                    "display.profile_image_url": s3_uri,
                    "display.last_updated": utc_now_iso(),
                    "image_mapping.status": "matched",
                    "image_mapping.s3_bucket": s3_bucket,
                    "image_mapping.s3_key": s3_key,
                    "image_mapping.s3_uri": s3_uri,
                }
            )
            status = "matched"
        else:
            update_doc["image_mapping.status"] = "image_record_found_no_s3"
            status = "image_record_found_no_s3"

        self.scholars.update_one({"profile_id": profile_id}, {"$set": update_doc}, upsert=True)
        return {
            "status": status,
            "object_id": object_id,
            "source_url": source_url,
            "image_status": image_status,
            "has_human": has_human,
            "has_s3": bool(s3_bucket and s3_key),
        }

    def update_source_state(
        self,
        *,
        profile_id: str,
        urls: Sequence[str],
        collection_meta: Dict[str, Any],
        status: str,
        primary_url: str,
    ) -> None:
        self.scholars.update_one(
            {"profile_id": profile_id},
            {
                "$set": {
                    "source": list(urls),
                    "source_primary_url": primary_url,
                    "source_collected_at": utc_now_iso(),
                    "source_collection_meta": collection_meta,
                    "processing_status": status,
                    "processing_updated_at": utc_now_iso(),
                }
            },
            upsert=True,
        )

    def update_processing_state(self, *, profile_id: str, status: str, error: str = "") -> None:
        update_doc: Dict[str, Any] = {
            "processing_status": status,
            "processing_updated_at": utc_now_iso(),
        }
        if error:
            update_doc["processing_error"] = error
        else:
            update_doc["processing_error"] = ""
        if status == "completed":
            update_doc["last_ingested_at"] = utc_now_iso()
        self.scholars.update_one({"profile_id": profile_id}, {"$set": update_doc}, upsert=True)

    @staticmethod
    def scholar_output_paths(output_root: Path, workbook_slug: str, profile_id: str) -> Dict[str, Path]:
        workbook_root = output_root / workbook_slug
        return {
            "workbook_root": workbook_root,
            "profiles_root": workbook_root / "profiles",
            "chunks_root": workbook_root / "chunked_profiles",
            "url_lists_root": workbook_root / "url_lists",
            "profile_dir": workbook_root / "profiles" / profile_id,
            "chunks_file": workbook_root / "chunked_profiles" / profile_id / "chunks.json",
        }

    async def run(self) -> int:
        workbooks = self.workbook_paths()
        if not workbooks:
            log_event("Universities", "No workbooks matched the requested filters.")
            return 0

        log_event(
            "Universities",
            (
                f"Starting folder run | input_dir={self.args.input_dir} "
                f"output_root={self.args.output_root} "
                f"workbooks={len(workbooks)} "
                f"skip_existing={self.args.skip_existing} "
                f"force={self.args.force} "
                f"aggressive={self.args.aggressive}"
            ),
        )
        log_event(
            "Universities",
            f"Discovery stack | OpenAlex={'On' if self.openalex_resolver is not None else 'Off'} DDG=On",
        )
        for workbook_idx, workbook_path in enumerate(workbooks, start=1):
            log_event("Universities", f"Workbook {workbook_idx}/{len(workbooks)} queued: {workbook_path.name}")

        total_success = 0
        total_failed = 0
        total_skipped = 0
        total_no_urls = 0

        output_root = Path(self.args.output_root)
        checkpoint_root = output_root / "_checkpoints"
        logs_root = output_root / "_logs"

        for workbook_path in workbooks:
            workbook_name = workbook_path.name
            workbook_slug = slugify(workbook_path.stem)
            checkpoint_path = checkpoint_root / f"{workbook_slug}.json"
            log_path = logs_root / f"{workbook_slug}.jsonl"
            checkpoint = load_checkpoint(checkpoint_path)

            print()
            print("=" * 80)
            log_event("Workbook", f"Starting {workbook_name}")
            print("=" * 80)

            df = normalize_workbook(pd.read_excel(workbook_path))
            auto_resume_index = 0
            if self.args.start_from:
                auto_resume_index = max(0, int(self.args.start_from))
            elif self.args.skip_existing and not self.args.force:
                auto_resume_index = self.compute_resume_index(
                    df=df,
                    workbook_slug=workbook_slug,
                    checkpoint=checkpoint,
                    output_root=output_root,
                )
            if auto_resume_index:
                log_event("Resume", f"starting {workbook_name} from row index {auto_resume_index}", indent=2)
                df = df.iloc[auto_resume_index:].reset_index(drop=True)
            if self.args.limit_scholars is not None:
                df = df.head(self.args.limit_scholars)
            log_event(
                "Workbook",
                (
                    f"Loaded {len(df)} scholar row(s) | checkpoint_entries={len(checkpoint)} "
                    f"checkpoint={checkpoint_path}"
                ),
            )

            workbook_success = 0
            workbook_failed = 0
            workbook_skipped = 0
            workbook_no_urls = 0

            for idx, (_, row) in enumerate(df.iterrows(), start=1):
                record = {col: row[col] for col in df.columns}
                name = clean_cell(record.get("Name")) or "Unknown"
                profile_id = stable_profile_id(record, workbook_slug)
                signature = row_signature(record, workbook_slug)
                output_paths = self.scholar_output_paths(output_root, workbook_slug, profile_id)
                url_lists_root = output_paths["url_lists_root"]
                url_lists_root.mkdir(parents=True, exist_ok=True)

                existing_checkpoint = checkpoint.get(profile_id) or {}
                done_on_disk = output_paths["chunks_file"].exists() and output_paths["chunks_file"].stat().st_size > 0
                same_signature = existing_checkpoint.get("row_signature") == signature
                should_skip = (
                    self.args.skip_existing
                    and not self.args.force
                    and same_signature
                    and (
                        existing_checkpoint.get("status") == "done"
                        or done_on_disk
                    )
                )

                if should_skip:
                    workbook_skipped += 1
                    total_skipped += 1
                    log_event(
                        "Skip",
                        f"{idx}/{len(df)} {name} ({profile_id}) already completed with same row signature",
                        indent=2,
                    )
                    continue

                print()
                log_event("Scholar", f"{idx}/{len(df)} {name} ({profile_id})")
                if not looks_like_person_name(name):
                    error_text = "record_name_is_not_a_person_name"
                    workbook_failed += 1
                    total_failed += 1
                    checkpoint[profile_id] = {
                        "status": "failed",
                        "name": name,
                        "error": error_text,
                        "row_signature": signature,
                        "ts": utc_now_iso(),
                    }
                    save_checkpoint(checkpoint_path, checkpoint)
                    append_jsonl(
                        log_path,
                        {
                            "profile_id": profile_id,
                            "name": name,
                            "status": "failed",
                            "error": error_text,
                            "row_signature": signature,
                            "ts": utc_now_iso(),
                        },
                    )
                    self.update_processing_state(profile_id=profile_id, status="failed", error=error_text)
                    log_event("Validate", f"FAILED: {error_text}", indent=2)
                    continue
                if self.args.dry_run:
                    planned_identity = identity_keywords(record)
                    log_event(
                        "DryRun",
                        (
                            f"university={planned_identity.get('university', '')!r} "
                            f"source={planned_identity.get('source_url', '')!r}"
                        ),
                        indent=2,
                    )
                    workbook_skipped += 1
                    total_skipped += 1
                    continue

                log_event("Mongo", "Upserting seed scholar document", indent=2)
                self.upsert_seed_doc(
                    record=record,
                    profile_id=profile_id,
                    workbook_name=workbook_name,
                    workbook_slug=workbook_slug,
                    signature=signature,
                )
                image_mapping_result = self.apply_image_mapping(record=record, profile_id=profile_id)
                image_source_url = clean_cell(image_mapping_result.get("source_url"))
                if not clean_cell(record.get("Source URL")) and image_source_url.startswith(("http://", "https://")):
                    record["Source URL"] = image_source_url
                    self.scholars.update_one(
                        {"profile_id": profile_id},
                        {"$set": {"seed_profile_url": image_source_url}},
                        upsert=True,
                    )
                    log_event("Image", "using image-db source_url as official seed URL", indent=2)
                if image_mapping_result.get("status") == "matched":
                    log_event("Image", "matched S3 image via Object ID", indent=2)
                elif image_mapping_result.get("status") not in {"disabled", "image_record_not_found"}:
                    log_event("Image", str(image_mapping_result.get("status")), indent=2)

                scholar_slug = f"{slugify(name)}-{str(profile_id)[:8]}"
                urls_file = url_lists_root / f"{scholar_slug}.txt"
                meta_file = url_lists_root / f"{scholar_slug}.meta.json"
                existing_urls: List[str] = []
                if self.args.merge_existing and urls_file.exists():
                    try:
                        existing_urls = [
                            line.strip()
                            for line in urls_file.read_text(encoding="utf-8").splitlines()
                            if line.strip() and not line.strip().startswith("#")
                        ]
                    except Exception:
                        existing_urls = []
                if existing_urls:
                    log_event("Collect", f"loaded {len(existing_urls)} previously saved URL(s) from {urls_file}", indent=2)

                openalex_result: Dict[str, Any] = {"matched": False, "urls": [], "meta": {}}
                if self.openalex_resolver is not None:
                    try:
                        openalex_result = self.openalex_resolver.resolve(identity_keywords(record))
                        if openalex_result.get("matched"):
                            log_event(
                                "OpenAlex",
                                (
                                    f"matched {openalex_result.get('meta', {}).get('matched_display_name', '')!r} "
                                    f"urls={len(openalex_result.get('urls', []) or [])}"
                                ),
                                indent=2,
                            )
                        else:
                            log_event("OpenAlex", "no confident author match", indent=2)
                    except Exception as exc:
                        log_event("OpenAlex", f"lookup failed: {exc}", indent=2)

                try:
                    log_event("Collect", "starting URL discovery", indent=2)
                    collected = await collect_urls_for_record(
                        record,
                        max_urls=self.args.max_urls,
                        per_query_results=self.args.per_query_results,
                        sleep_seconds=self.args.sleep_seconds,
                        aggressive=self.args.aggressive,
                        openalex_result=openalex_result,
                        existing_urls=existing_urls,
                    )
                except Exception as exc:
                    error_text = str(exc)
                    workbook_failed += 1
                    total_failed += 1
                    checkpoint[profile_id] = {
                        "status": "failed",
                        "name": name,
                        "error": error_text,
                        "row_signature": signature,
                        "ts": utc_now_iso(),
                    }
                    save_checkpoint(checkpoint_path, checkpoint)
                    append_jsonl(
                        log_path,
                        {
                            "profile_id": profile_id,
                            "name": name,
                            "status": "failed",
                            "error": error_text,
                            "row_signature": signature,
                            "ts": utc_now_iso(),
                        },
                    )
                    self.update_processing_state(profile_id=profile_id, status="failed", error=error_text)
                    log_event("Collect", f"FAILED: {error_text}", indent=2)
                    continue

                urls = collected.get("kept_urls", [])
                primary_url = clean_cell(record.get("Source URL")) or (urls[0] if urls else "")

                collector_version = "universities_ddg_v1_aggressive" if self.args.aggressive else "universities_ddg_v1_precision"
                collection_meta = {
                    "collector_version": collector_version,
                    "raw_results": int(collected.get("raw_results", 0)),
                    "rejected_count": int(collected.get("rejected_count", 0)),
                    "queries_fired": len(collected.get("queries", []) or []),
                    "max_urls_cap": int(self.args.max_urls),
                    "workbook": workbook_name,
                    "row_signature": signature,
                    "openalex": collected.get("openalex_meta", {}),
                }

                if not urls:
                    workbook_no_urls += 1
                    total_no_urls += 1
                    checkpoint[profile_id] = {
                        "status": "no_urls",
                        "name": name,
                        "row_signature": signature,
                        "collection_meta": collection_meta,
                        "ts": utc_now_iso(),
                    }
                    save_checkpoint(checkpoint_path, checkpoint)
                    append_jsonl(
                        log_path,
                        {
                            "profile_id": profile_id,
                            "name": name,
                            "status": "no_urls",
                            "result": collected,
                            "ts": utc_now_iso(),
                        },
                    )
                    self.update_source_state(
                        profile_id=profile_id,
                        urls=[],
                        collection_meta=collection_meta,
                        status="no_urls",
                        primary_url=primary_url,
                    )
                    log_event("Collect", "0 URLs survived identity filtering", indent=2)
                    continue

                urls_file.write_text("\n".join(urls) + "\n", encoding="utf-8")
                meta_file.write_text(
                    json.dumps(
                        {
                            "profile_id": profile_id,
                            "profile_name": name,
                            "profile_url": primary_url,
                            "collector_version": collector_version,
                            "university": clean_cell(record.get("University")),
                            "email": clean_cell(record.get("Email")),
                            "row_signature": signature,
                            "results_detail": collected.get("results_detail", []),
                        },
                        indent=2,
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )

                self.update_source_state(
                    profile_id=profile_id,
                    urls=urls,
                    collection_meta=collection_meta,
                    status="sources_collected",
                    primary_url=primary_url,
                )

                log_event(
                    "Collect",
                    (
                        f"kept {len(urls)} URL(s) "
                        f"(raw={collected.get('raw_results', 0)}, rejected={collected.get('rejected_count', 0)})"
                    ),
                    indent=2,
                )
                for url_idx, url in enumerate(urls[:5], start=1):
                    log_event("Collect", f"URL {url_idx}: {url}", indent=4)
                if len(urls) > 5:
                    log_event("Collect", f"... plus {len(urls) - 5} more URL(s)", indent=4)
                log_event("Files", f"saved URL list to {urls_file}", indent=2)
                log_event("Files", f"saved URL metadata to {meta_file}", indent=2)
                self.update_processing_state(profile_id=profile_id, status="running")

                try:
                    os.environ.setdefault("INTENT_GATING_ENABLED", "0")
                    os.environ.setdefault("STRICT_SOURCE_POLICY", "0")
                    os.environ.setdefault("DEBUG_SCRAPER_LINKS", "1")
                    build_pipeline = self.get_build_pipeline(
                        workbook_root=output_paths["workbook_root"],
                        chunks_root=output_paths["chunks_root"],
                    )
                    log_event("Pipeline", f"starting scrape/chunk stage for {len(urls)} URL(s)", indent=2)
                    summary = await build_pipeline.run_from_urls(
                        urls=urls,
                        profile_name=name,
                        profile_id=profile_id,
                        profile_url=primary_url,
                    )
                except Exception as exc:
                    error_text = str(exc)
                    workbook_failed += 1
                    total_failed += 1
                    checkpoint[profile_id] = {
                        "status": "failed",
                        "name": name,
                        "error": error_text,
                        "row_signature": signature,
                        "collection_meta": collection_meta,
                        "file": str(urls_file),
                        "ts": utc_now_iso(),
                    }
                    save_checkpoint(checkpoint_path, checkpoint)
                    append_jsonl(
                        log_path,
                        {
                            "profile_id": profile_id,
                            "name": name,
                            "status": "failed",
                            "error": error_text,
                            "result": collected,
                            "ts": utc_now_iso(),
                        },
                    )
                    self.update_processing_state(profile_id=profile_id, status="failed", error=error_text)
                    log_event("Pipeline", f"FAILED: {error_text}", indent=2)
                    continue

                successful_profiles = int((summary or {}).get("successful", 0))
                if successful_profiles > 0:
                    validation = self.validate_extracted_profile(
                        output_paths=output_paths,
                        profile_id=profile_id,
                        record=record,
                    )
                    if not validation.get("ok"):
                        workbook_failed += 1
                        total_failed += 1
                        error_text = f"identity_validation_failed:{validation.get('reason')}"
                        checkpoint[profile_id] = {
                            "status": "failed",
                            "name": name,
                            "error": error_text,
                            "row_signature": signature,
                            "collection_meta": collection_meta,
                            "file": str(urls_file),
                            "summary": summary,
                            "validation": validation,
                            "ts": utc_now_iso(),
                        }
                        save_checkpoint(checkpoint_path, checkpoint)
                        append_jsonl(
                            log_path,
                            {
                                "profile_id": profile_id,
                                "name": name,
                                "status": "failed",
                                "error": error_text,
                                "result": collected,
                                "summary": summary,
                                "validation": validation,
                                "ts": utc_now_iso(),
                            },
                        )
                        self.update_processing_state(profile_id=profile_id, status="failed", error=error_text)
                        log_event("Validate", f"FAILED: {validation.get('reason')}", indent=2)
                        continue

                    sync_pipeline = self.get_sync_pipeline(
                        workbook_root=output_paths["workbook_root"],
                        chunks_root=output_paths["chunks_root"],
                    )
                    log_event(
                        "Validate",
                        (
                            f"passed identity validation | chunks={validation.get('chunk_count', 0)} "
                            f"urls={validation.get('url_count', 0)}"
                        ),
                        indent=2,
                    )
                    sync_result = self.sync_verified_profile(
                        sync_pipeline=sync_pipeline,
                        profile_id=profile_id,
                        profile_name=name,
                        chunks_path=output_paths["chunks_file"],
                    )
                    if sync_result.get("mongo_synced_delta", 0) <= 0 or sync_result.get("vectors_uploaded_delta", 0) <= 0:
                        workbook_failed += 1
                        total_failed += 1
                        error_text = f"sync_failed:{sync_result.get('last_error') or 'no_vectors_or_mongo_sync'}"
                        checkpoint[profile_id] = {
                            "status": "failed",
                            "name": name,
                            "error": error_text,
                            "row_signature": signature,
                            "collection_meta": collection_meta,
                            "file": str(urls_file),
                            "summary": summary,
                            "validation": validation,
                            "sync_result": sync_result,
                            "ts": utc_now_iso(),
                        }
                        save_checkpoint(checkpoint_path, checkpoint)
                        append_jsonl(
                            log_path,
                            {
                                "profile_id": profile_id,
                                "name": name,
                                "status": "failed",
                                "error": error_text,
                                "result": collected,
                                "summary": summary,
                                "validation": validation,
                                "sync_result": sync_result,
                                "ts": utc_now_iso(),
                            },
                        )
                        self.update_processing_state(profile_id=profile_id, status="failed", error=error_text)
                        log_event("Sync", f"FAILED: {error_text}", indent=2)
                        continue

                    workbook_success += 1
                    total_success += 1
                    checkpoint[profile_id] = {
                        "status": "done",
                        "name": name,
                        "row_signature": signature,
                        "url_count": len(urls),
                        "collection_meta": collection_meta,
                        "file": str(urls_file),
                        "summary": {
                            "successful": int((summary or {}).get("successful", 0)),
                            "failed": int((summary or {}).get("failed", 0)),
                            "ignored": int((summary or {}).get("ignored", 0)),
                            "incremental_sync": (summary or {}).get("incremental_sync", {}),
                        },
                        "validation": validation,
                        "sync_result": sync_result,
                        "ts": utc_now_iso(),
                    }
                    save_checkpoint(checkpoint_path, checkpoint)
                    append_jsonl(
                        log_path,
                        {
                            "profile_id": profile_id,
                            "name": name,
                            "status": "done",
                            "result": collected,
                            "summary": summary,
                            "validation": validation,
                            "sync_result": sync_result,
                            "image_mapping": image_mapping_result,
                            "ts": utc_now_iso(),
                        },
                    )
                    self.apply_image_mapping(record=record, profile_id=profile_id)
                    self.update_processing_state(profile_id=profile_id, status="completed")
                    log_event(
                        "Pipeline",
                        (
                            f"completed | successful={int((summary or {}).get('successful', 0))} "
                            f"failed={int((summary or {}).get('failed', 0))} "
                            f"ignored={int((summary or {}).get('ignored', 0))} "
                            f"vectors={sync_result.get('vectors_uploaded_delta', 0)} "
                            f"mongo={sync_result.get('mongo_synced_delta', 0)}"
                        ),
                        indent=2,
                    )
                else:
                    workbook_failed += 1
                    total_failed += 1
                    error_text = "run_full_pipeline returned 0 successful profiles"
                    checkpoint[profile_id] = {
                        "status": "failed",
                        "name": name,
                        "error": error_text,
                        "row_signature": signature,
                        "collection_meta": collection_meta,
                        "file": str(urls_file),
                        "summary": summary,
                        "ts": utc_now_iso(),
                    }
                    save_checkpoint(checkpoint_path, checkpoint)
                    append_jsonl(
                        log_path,
                        {
                            "profile_id": profile_id,
                            "name": name,
                            "status": "failed",
                            "error": error_text,
                            "result": collected,
                            "summary": summary,
                            "ts": utc_now_iso(),
                        },
                    )
                    self.update_processing_state(profile_id=profile_id, status="failed", error=error_text)
                    log_event("Pipeline", f"FAILED: {error_text}", indent=2)

            print()
            log_event(
                "Workbook",
                (
                    f"summary | success={workbook_success} "
                    f"failed={workbook_failed} "
                    f"no_urls={workbook_no_urls} "
                    f"skipped={workbook_skipped} "
                    f"total={len(df)}"
                ),
            )

        print()
        print("=" * 80)
        log_event(
            "Universities",
            (
                f"Overall summary | success={total_success} "
                f"failed={total_failed} "
                f"no_urls={total_no_urls} "
                f"skipped={total_skipped}"
            ),
        )
        print("=" * 80)
        return 0 if total_failed == 0 else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR, help=f"Directory containing workbook files (default: {DEFAULT_INPUT_DIR})")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, help=f"Base output directory (default: {DEFAULT_OUTPUT_ROOT})")
    parser.add_argument("--workbook-glob", default="*.xlsx", help='Workbook glob within --input-dir (default: "*.xlsx")')
    parser.add_argument("--workbook-name-contains", default=None, help="Only process workbooks whose filename contains this text")
    parser.add_argument("--limit-workbooks", type=int, default=None, help="Optional cap on number of workbook files to process")
    parser.add_argument("--limit-scholars", type=int, default=None, help="Optional cap on scholar rows per workbook")
    parser.add_argument("--start-from", type=int, default=0, help="Skip the first N scholar rows in each workbook")
    parser.add_argument("--max-urls", type=int, default=DEFAULT_MAX_URLS, help=f"Maximum collected URLs per scholar (default: {DEFAULT_MAX_URLS})")
    parser.add_argument("--per-query-results", type=int, default=DEFAULT_PER_QUERY_RESULTS, help=f"DuckDuckGo results per query (default: {DEFAULT_PER_QUERY_RESULTS})")
    parser.add_argument("--sleep-seconds", type=float, default=DEFAULT_SLEEP_SECONDS, help=f"Sleep between DDG queries (default: {DEFAULT_SLEEP_SECONDS})")
    parser.add_argument("--http-timeout-seconds", type=float, default=DEFAULT_HTTP_TIMEOUT_SECONDS, help=f"HTTP timeout for API discovery calls (default: {DEFAULT_HTTP_TIMEOUT_SECONDS})")
    parser.add_argument("--http-retries", type=int, default=DEFAULT_HTTP_RETRIES, help=f"HTTP retries for API discovery calls (default: {DEFAULT_HTTP_RETRIES})")
    parser.add_argument(
        "--skip-existing",
        dest="skip_existing",
        action="store_true",
        default=True,
        help="Skip scholars already completed with the same row signature (default: on)",
    )
    parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="Revisit completed scholars even if checkpoint/chunk files already exist",
    )
    parser.add_argument("--force", action="store_true", help="Reprocess even if a matching completed checkpoint exists")
    parser.add_argument("--aggressive", action="store_true", help="Use a wider query set across scholar-owned/identity domains")
    parser.add_argument("--disable-openalex", action="store_true", help="Disable OpenAlex author-identity discovery even if OPENALEX_API_KEY is set")
    parser.add_argument("--dry-run", action="store_true", help="Inspect what would run without searching or scraping")
    parser.add_argument("--merge-existing", dest="merge_existing", action="store_true", default=True, help="Merge with existing URL list files before recollecting (default on)")
    parser.add_argument("--no-merge-existing", dest="merge_existing", action="store_false", help="Do not merge previously collected URL files")
    return parser.parse_args()


async def main_async() -> int:
    args = parse_args()
    configure_utf8_stdio()
    runner = UniversitiesFolderRunner(args)
    try:
        return await runner.run()
    finally:
        runner.close()


def main() -> int:
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
