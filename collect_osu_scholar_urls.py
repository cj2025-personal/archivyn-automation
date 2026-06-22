"""
Collect URLs about each OSU scholar via DuckDuckGo and write per-scholar
.txt files that the existing legend pipeline can consume.

For each scholar in ``scholars`` where ``scholar_type == 'osu'`` we run a
suite of identity-anchored DDG queries (full name + Ohio State + dept +
field), filter the results against a noise/skip blocklist, rank by
domain authority, cap at ``--max-urls``, and write the result to
``osu_scholars/final/<slug>.txt``.

The same URL list is also written back to the scholar's Mongo doc as
the ``source`` array so the field is queryable even before the
downstream scrape pipeline runs.

Identity safety:
- Queries always quote the full name verbatim.
- Each candidate URL is identity-validated against title + snippet:
  the last name (or full name) must appear, and the result must mention
  Ohio State / OSU OR the scholar's department/field, OR live on a
  high-authority academic domain (.edu / orcid / scholar.google / …).
  This is what prevents another "Kevin Brown" from a different
  institution from polluting the corpus.

Resumable:
- A checkpoint JSON tracks per-scholar status (done / no_urls / failed).
- ``--skip-existing`` short-circuits scholars whose .txt already exists.

Usage:
    python collect_osu_scholar_urls.py                   # all 3,907 scholars
    python collect_osu_scholar_urls.py --limit 25        # smoke test
    python collect_osu_scholar_urls.py --start-from 500 --limit 100
    python collect_osu_scholar_urls.py --filter-field "Computer Science"
    python collect_osu_scholar_urls.py --skip-existing   # resume mode
    python collect_osu_scholar_urls.py --dry-run         # preview only

Targeted, exhaustive re-collection of weak scholars (accuracy preserved):
    python collect_osu_scholar_urls.py \
        --from-csv osu_scholars/weak_scholars.csv \
        --aggressive --max-urls 30 --per-query-results 25
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

from dotenv import load_dotenv
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config.mongodb_utils import create_mongo_client, resolve_mongo_db_name
from api.utils.source_guardrails import is_noise_domain


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("collect_osu")

DEFAULT_OUTPUT_DIR = "osu_scholars/final"
DEFAULT_CHECKPOINT = "osu_scholars/url_collection_checkpoint.json"
DEFAULT_LOG = "osu_scholars/url_collection_log.jsonl"
DEFAULT_EXCEL_PATH = "excel/OSU.xlsx"
DEFAULT_MAX_URLS = 50
DEFAULT_SLEEP_S = 2.5
DEFAULT_PER_QUERY = 15

# Domains to skip outright (mirrors legend pipeline blocklist + ToS-risky hosts).
# Note: ``google.com`` is intentionally NOT here — we want to keep
# ``scholar.google.com`` and ``patents.google.com``. SERP wrappers from
# ``www.google.com`` are caught by ``_should_skip`` via an exact-host
# check below.
SKIP_DOMAINS = {
    "facebook.com", "twitter.com", "x.com", "instagram.com",
    "tiktok.com", "pinterest.com", "reddit.com", "quora.com",
    "linkedin.com",  # ToS prohibits scraping
    "amazon.com", "ebay.com", "walmart.com",
    "play.google.com", "apps.apple.com",
    "wikipedia.org",
}

# Exact hostnames to reject even when their parent domain is permitted.
SKIP_EXACT_HOSTS = {
    "www.google.com", "google.com", "duckduckgo.com", "www.duckduckgo.com",
    "www.bing.com", "bing.com",
}

# Domains the *legend* pipeline marks as noise (listing pages with no
# article body) but that ARE valuable for an OSU faculty enrichment
# context — Google Scholar / Semantic Scholar profile pages are exactly
# what we want for an active researcher. We override the shared
# is_noise_domain blocklist for these.
NOISE_OVERRIDE_ALLOW = {
    "scholar.google.com",
    "semanticscholar.org",
}

# Domains we trust (ranked top → bottom). Lower index = higher priority.
PRIORITY_DOMAINS = [
    # OSU primary
    "experts.osu.edu", "osu.edu",
    # Author identity
    "orcid.org",
    # Citation databases
    "scholar.google.com",
    "semanticscholar.org",
    "openalex.org",
    "researchgate.net",
    # Preprint / OA
    "arxiv.org", "biorxiv.org", "medrxiv.org",
    "pubmed.ncbi.nlm.nih.gov", "ncbi.nlm.nih.gov", "pmc.ncbi.nlm.nih.gov",
    # Funding
    "nsf.gov", "nih.gov", "usaspending.gov",
    # Publishers
    "ieee.org", "acm.org",
    "springer.com", "sciencedirect.com", "wiley.com",
    "nature.com", "plos.org", "frontiersin.org", "mdpi.com",
    "tandfonline.com", "sagepub.com", "jstor.org", "cambridge.org",
    "oup.com", "academic.oup.com",
    # Patents
    "patents.google.com", "uspto.gov",
    # OSU-affiliated
    "u.osu.edu", "engineering.osu.edu", "medicine.osu.edu",
    "asc.osu.edu", "cfaes.osu.edu",
]

SCHOLAR_IDENTITY_DOMAINS = {
    "orcid.org",
    "scholar.google.com",
    "semanticscholar.org",
    "openalex.org",
    "researchgate.net",
    "dblp.org",
}

BAD_RESULT_PATTERNS = (
    "obituary",
    "memorial",
    "funeral",
    "tributes",
    "legacy.com",
    "findagrave",
    "wiktionary",
    "wiki",
    "athletics",
    "sports",
)


# ── Helpers ────────────────────────────────────────────────────────────


def _slugify(name: str, profile_id: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "-", name.strip().lower()).strip("-")
    if not s:
        s = "scholar"
    return f"{s}-{str(profile_id)[:8]}"


def _domain_of(url: str) -> str:
    try:
        d = (urlparse(url).netloc or "").lower()
    except Exception:
        return ""
    if d.startswith("www."):
        d = d[4:]
    return d


def _url_priority(url: str) -> int:
    d = _domain_of(url)
    if not d:
        return 999
    for i, p in enumerate(PRIORITY_DOMAINS):
        if d == p or d.endswith("." + p):
            return i
    if d.endswith(".edu"):
        return 50
    if d.endswith(".gov"):
        return 60
    if d.endswith(".org"):
        return 70
    return 100


def _should_skip(url: str) -> bool:
    if not url or not url.startswith(("http://", "https://")):
        return True
    d = _domain_of(url)
    if not d:
        return True
    if d in SKIP_EXACT_HOSTS:
        return True
    in_allow_override = any(
        d == ok or d.endswith("." + ok) for ok in NOISE_OVERRIDE_ALLOW
    )
    if not in_allow_override and is_noise_domain(url):
        return True
    for bad in SKIP_DOMAINS:
        if d == bad or d.endswith("." + bad):
            return True
    # Cap path length to avoid unbounded query-string SERP redirectors.
    if len(url) > 600:
        return True
    return False


def _normalize_url(url: str) -> str:
    """Strip tracking junk + trailing slash so duplicates collapse."""
    try:
        p = urlparse(url.strip())
    except Exception:
        return url.strip()
    scheme = (p.scheme or "https").lower()
    netloc = (p.netloc or "").lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    path = (p.path or "/").rstrip("/") or "/"
    # Drop common tracking params.
    qs_keep: List[str] = []
    if p.query:
        for kv in p.query.split("&"):
            k = kv.split("=", 1)[0].lower()
            if k.startswith(("utm_", "fbclid", "gclid", "mc_", "ref_src", "ref")):
                continue
            qs_keep.append(kv)
    query = "&".join(qs_keep)
    suffix = f"?{query}" if query else ""
    return f"{scheme}://{netloc}{path}{suffix}"


def _normalize_name(value: str) -> str:
    text = (value or "").strip().lower()
    text = re.sub(r"^(dr\.?|prof\.?|professor)\s+", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _source_path_token(url: str) -> str:
    try:
        path = (urlparse(url).path or "").strip("/")
    except Exception:
        return ""
    if not path:
        return ""
    parts = [part for part in path.split("/") if part]
    return parts[-1].lower() if parts else ""


def _load_excel_identity_map(excel_path: Path) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    if not excel_path.exists():
        raise FileNotFoundError(f"OSU workbook not found: {excel_path}")

    df = pd.read_excel(excel_path)
    by_profile_id: Dict[str, Dict[str, Any]] = {}
    by_name: Dict[str, Dict[str, Any]] = {}

    for _, row in df.iterrows():
        record = row.to_dict()
        name = str(record.get("Name") or "").strip()
        norm_name = _normalize_name(name)
        profile_id = str(record.get("Scholar Profile ID") or "").strip()
        if profile_id and profile_id.lower() != "nan":
            by_profile_id[profile_id] = record
        if norm_name and norm_name not in by_name:
            by_name[norm_name] = record
    return by_profile_id, by_name


def _attach_excel_identity(
    scholar: Dict[str, Any],
    by_profile_id: Dict[str, Dict[str, Any]],
    by_name: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    profile_id = str(scholar.get("profile_id") or "").strip()
    name = (scholar.get("name") or {}).get("full") or scholar.get("professor_name") or ""
    row = None
    if profile_id and profile_id in by_profile_id:
        row = by_profile_id[profile_id]
    else:
        row = by_name.get(_normalize_name(name))
    if not row:
        return scholar

    scholar = dict(scholar)
    scholar["_excel_source_url"] = str(row.get("source") or row.get("Source URL") or "").strip()
    scholar["_excel_email"] = str(row.get("Email") or "").strip()
    scholar["_excel_expertise"] = str(row.get("Expertise") or "").strip()
    scholar["_excel_university"] = str(row.get("University") or "").strip()
    scholar["_excel_race"] = str(row.get("Race") or "").strip()
    scholar["_excel_profile_id"] = str(row.get("Scholar Profile ID") or "").strip()
    if scholar.get("_excel_profile_id") and not scholar.get("profile_id"):
        scholar["profile_id"] = scholar["_excel_profile_id"]
    return scholar


def _identity_keywords(scholar: Dict[str, Any]) -> Dict[str, str]:
    """Pull every identity-disambiguating fact we can off the scholar doc."""
    name = (scholar.get("name") or {}).get("full") or scholar.get("professor_name") or ""
    md = scholar.get("metadata") or {}
    about = scholar.get("about") or {}
    bg = scholar.get("background_and_work") or {}
    source_url = str(scholar.get("_excel_source_url") or "").strip()
    email = str(scholar.get("_excel_email") or "").strip().lower()
    email_local = email.split("@", 1)[0] if "@" in email else ""
    expertise = str(scholar.get("_excel_expertise") or "").strip()
    return {
        "name": name.strip(),
        "first": (scholar.get("name") or {}).get("first", "").strip(),
        "last": (scholar.get("name") or {}).get("last", "").strip(),
        "field": (md.get("field_of_study") or about.get("field_of_study") or "").strip(),
        "department": (about.get("department") or md.get("department") or "").strip(),
        "institution": (
            about.get("institution")
            or md.get("university")
            or scholar.get("_excel_university")
            or "Ohio State University"
        ).strip(),
        "current_position": (about.get("current_position") or "").strip(),
        "research_focus": ", ".join(bg.get("research_focus") or [])[:120],
        "source_url": source_url,
        "source_host": _domain_of(source_url),
        "source_path_token": _source_path_token(source_url),
        "email": email,
        "email_local": email_local,
        "expertise": expertise,
    }


def _build_queries(idk: Dict[str, str], aggressive: bool = False) -> List[str]:
    """Rich query suite — accuracy is paramount, so we cast wide and
    rely on identity validation downstream.

    When ``aggressive`` is set we add a much broader set of angles
    (identity-domain site: queries, CV/lab/award/news/publication angles,
    etc.) to lift richness and quantity for weak scholars. Accuracy is
    unaffected — every candidate still passes ``_identity_match_osu``.
    """
    name = idk["name"]
    if not name:
        return []
    base = f'"{name}"'
    osu = '"Ohio State"'
    queries: List[str] = [
        f'{base} {osu}',
        f'{base} {osu} faculty profile',
        f'{base} {osu} CV biography',
        f'{base} {osu} publications research',
        f'{base} {osu} site:osu.edu',
        f'{base} site:experts.osu.edu',
        f'{base} {osu} ORCID',
        f'{base} {osu} "Google Scholar"',
    ]
    if idk["source_host"]:
        queries.append(f'{base} site:{idk["source_host"]}')
    if idk["email_local"]:
        queries.append(f'"{idk["email_local"]}" {base} {osu}')
    if idk["department"]:
        queries.append(f'{base} {osu} "{idk["department"]}"')
    if idk["field"]:
        queries.append(f'{base} {osu} "{idk["field"]}"')
    if idk["research_focus"]:
        # Truncate so DDG accepts it; keep it under ~250 chars.
        queries.append(f'{base} {osu} {idk["research_focus"][:80]}')
    if idk["current_position"]:
        queries.append(f'{base} {osu} {idk["current_position"][:60]}')
    if idk["expertise"]:
        queries.append(f'{base} {osu} {idk["expertise"][:80]}')

    if aggressive:
        # Identity-domain-targeted queries: these directly surface the
        # high-value academic profiles that drive "richness".
        queries += [
            f'{base} site:orcid.org',
            f'{base} site:scholar.google.com',
            f'{base} site:semanticscholar.org',
            f'{base} site:openalex.org',
            f'{base} {osu} site:researchgate.net',
            f'{base} site:dblp.org',
            f'{base} site:pubmed.ncbi.nlm.nih.gov',
            f'{base} site:arxiv.org',
            f'{base} {osu} site:u.osu.edu',
            f'{base} {osu} site:experts.osu.edu',
        ]
        # Content-angle queries: more pages about the same person.
        queries += [
            f'{base} {osu} curriculum vitae',
            f'{base} {osu} lab OR laboratory OR "research group"',
            f'{base} {osu} award OR grant OR fellowship OR honor',
            f'{base} {osu} interview OR news OR profile',
            f'{base} {osu} dissertation OR thesis OR PhD',
            f'{base} {osu} book OR author OR chapter',
            f'{base} {osu} conference OR keynote OR talk',
            f'{base} {osu} department directory',
            f'{base} {osu} biography',
            f'{base} {osu} publications list',
        ]
        if idk["department"]:
            queries.append(f'{base} site:osu.edu "{idk["department"]}"')
        if idk["field"]:
            queries.append(f'{base} {idk["field"]} Ohio State research')
        if idk["last"] and idk["email_local"]:
            queries.append(f'{base} {idk["email_local"]} site:osu.edu')

    # Dedupe while preserving order.
    seen = set()
    out: List[str] = []
    for q in queries:
        key = q.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(q)
    return out


def _identity_match(result: Dict[str, str], idk: Dict[str, str]) -> bool:
    """Does this DDG result actually look like our scholar?

    Identity signal sources (any of these counts as "subject named"):
      - full name appears in title/snippet
      - last+first appear in title/snippet
      - last alone appears in title/snippet (only when last name is ≥4 chars)
      - URL path/host contains the scholar's name slug (compressed) OR
        both first and last name tokens

    Authority signal (any of these qualifies the URL):
      - Host is on ``osu.edu`` or any of its subdomains
      - Title/snippet says "Ohio State" / "OSU"
      - Host is a top-tier academic / citation / preprint domain
      - Host ends in ``.edu`` or ``.gov``
      - Title/snippet mentions the scholar's department or field
    """
    title = (result.get("title") or "")
    snippet = (result.get("snippet") or "")
    blob = (title + " " + snippet).lower()
    url = result.get("url", "") or ""
    url_lc = url.lower()

    name = idk["name"].lower()
    last = idk["last"].lower()
    first = idk["first"].lower()
    if not name or not last:
        return True

    # Compressed forms of the name as they tend to appear in URL slugs.
    # "Rama Yedavalli" -> "ramayedavalli" / "rama-yedavalli" / "yedavalli".
    name_compact = re.sub(r"[^a-z0-9]+", "", name)
    url_path_compact = re.sub(r"[^a-z0-9]+", "", url_lc)

    # ── Identity signal ───────────────────────────────────────────────
    short_last = len(last) <= 3  # Tu, Wu, Ng, Lin, Liu, Yu, etc.

    name_in_blob = False
    if name in blob:
        name_in_blob = True
    elif last in blob and first and first in blob:
        name_in_blob = True
    elif last in blob and not short_last:
        # Last-name-alone is OK only for distinctive last names.
        name_in_blob = True

    name_in_url = False
    if name_compact and name_compact in url_path_compact:
        name_in_url = True
    elif first and last and (first in url_lc or first in url_path_compact) and \
            (last in url_lc or last in url_path_compact):
        name_in_url = True
    elif not short_last and last in url_path_compact and len(last) >= 5:
        # Last-name-only in URL slug is OK for unique surnames (≥5 chars).
        name_in_url = True

    if not (name_in_blob or name_in_url):
        return False

    # ── Authority signal ──────────────────────────────────────────────
    domain = _domain_of(url)

    # Strongest: any OSU host. Trust the URL completely; OSU subdomains
    # don't host content about random other people of the same name.
    if domain == "osu.edu" or domain.endswith(".osu.edu"):
        return True

    # Strong: explicit OSU mention in snippet/title.
    if "ohio state" in blob or " osu " in f" {blob} ":
        return True

    # Strong: priority academic / citation / preprint domains.
    if any(p == domain or domain.endswith("." + p) for p in PRIORITY_DOMAINS[:14]):
        return True

    # Medium: any other .edu / .gov where the name passed the identity
    # filter. Cross-institution co-author pages and former-affiliation
    # pages are legitimate corpus material.
    if domain.endswith(".edu") or domain.endswith(".gov"):
        return True

    # Weak: the snippet matches the scholar's known field or department.
    if idk["field"] and idk["field"].lower() in blob:
        return True
    if idk["department"] and idk["department"].lower() in blob:
        return True

    # Otherwise be conservative: reject. Better to drop a marginal match
    # than poison the corpus with the wrong person.
    return False


def _is_osu_domain(domain: str) -> bool:
    return bool(domain) and (domain == "osu.edu" or domain.endswith(".osu.edu"))


def _matches_identity_domain(domain: str) -> bool:
    return any(domain == item or domain.endswith("." + item) for item in SCHOLAR_IDENTITY_DOMAINS)


def _identity_match_osu(result: Dict[str, str], idk: Dict[str, str]) -> bool:
    """Higher-precision OSU-specific identity matcher."""
    title = (result.get("title") or "")
    snippet = (result.get("snippet") or "")
    blob = f"{title} {snippet}".lower()
    url = result.get("url", "") or ""
    url_lc = url.lower()
    domain = _domain_of(url)

    text_blob = f"{title} {snippet} {url_lc}"
    if any(pattern in text_blob for pattern in BAD_RESULT_PATTERNS):
        return False

    name = idk["name"].lower()
    first = idk["first"].lower()
    last = idk["last"].lower()
    if not name or not first or not last:
        return False

    name_compact = re.sub(r"[^a-z0-9]+", "", name)
    url_compact = re.sub(r"[^a-z0-9]+", "", url_lc)
    full_name_in_blob = name in blob
    full_name_in_url = bool(name_compact and name_compact in url_compact)
    first_last_in_blob = first in blob and last in blob
    first_last_in_url = first in url_lc and last in url_lc
    source_path_token = idk["source_path_token"].lower()
    source_path_match = bool(source_path_token) and source_path_token in url_lc

    strong_name_signal = full_name_in_blob or full_name_in_url
    base_name_signal = strong_name_signal or first_last_in_blob or first_last_in_url or source_path_match
    if not base_name_signal:
        return False

    field = idk["field"].lower()
    department = idk["department"].lower()
    expertise = idk["expertise"].lower()
    current_position = idk["current_position"].lower()
    source_host = idk["source_host"].lower()

    field_signal = bool(field and field in blob)
    department_signal = bool(department and department in blob)
    expertise_signal = bool(expertise and any(token in blob for token in expertise.split() if len(token) >= 6))
    position_signal = bool(current_position and current_position in blob)
    osu_signal = _is_osu_domain(domain) or "ohio state" in blob or " osu " in f" {blob} "
    source_host_signal = bool(source_host) and (domain == source_host or domain.endswith("." + source_host))

    if _is_osu_domain(domain):
        if source_host_signal and base_name_signal:
            return True
        if strong_name_signal:
            return True
        return first_last_in_blob and (field_signal or department_signal or position_signal or expertise_signal)

    if _matches_identity_domain(domain):
        return strong_name_signal and (osu_signal or field_signal or department_signal or expertise_signal)

    if domain.endswith(".edu") or domain.endswith(".gov"):
        return strong_name_signal and osu_signal and (field_signal or department_signal or position_signal or expertise_signal)

    return False


# ── DDG client (mirrors the existing two-tier pattern) ────────────────


async def _ddg_search(query: str, max_results: int) -> List[Dict[str, str]]:
    """Run one DDG query. Returns [] on rate-limit / failure."""
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        logger.error("duckduckgo-search not installed: pip install duckduckgo-search")
        return []
    loop = asyncio.get_event_loop()
    try:
        results = await loop.run_in_executor(
            None, lambda: list(DDGS().text(query, max_results=max_results))
        )
    except Exception as e:
        logger.warning("DDG query failed (%s): %s", query[:80], str(e)[:120])
        return []
    out: List[Dict[str, str]] = []
    for r in results:
        url = r.get("href") or r.get("url") or ""
        if not url:
            continue
        out.append({"url": url, "title": r.get("title", ""), "snippet": r.get("body", "")})
    return out


# ── Per-scholar collection ─────────────────────────────────────────────


async def collect_for_scholar(
    scholar: Dict[str, Any],
    *,
    max_urls: int,
    sleep_s: float,
    per_query: int,
    aggressive: bool = False,
    existing_urls: Optional[List[str]] = None,
) -> Dict[str, Any]:
    idk = _identity_keywords(scholar)
    queries = _build_queries(idk, aggressive=aggressive)
    if not queries:
        return {
            "name": idk["name"],
            "queries": [],
            "raw_results": 0,
            "kept_urls": [],
            "rejected_count": 0,
            "results_detail": [],
        }

    candidates: Dict[str, Dict[str, Any]] = {}  # normalised_url -> result
    rejected = 0
    raw = 0

    # Seed previously-collected URLs so a re-collection never regresses the
    # coverage we already had (they were accepted by the identity filter on a
    # prior pass; the official OSU page in particular must survive).
    for prev in existing_urls or []:
        if not prev or _should_skip(prev):
            continue
        norm = _normalize_url(prev)
        if norm in candidates:
            continue
        candidates[norm] = {
            "url": prev,
            "normalized_url": norm,
            "title": "Previously collected",
            "snippet": "",
            "source_query": "seed_existing",
        }

    if idk["source_url"] and not _should_skip(idk["source_url"]):
        norm = _normalize_url(idk["source_url"])
        candidates[norm] = {
            "url": idk["source_url"],
            "normalized_url": norm,
            "title": "Primary OSU profile",
            "snippet": "",
            "source_query": "seed_source_url",
        }

    for q in queries:
        results = await _ddg_search(q, per_query)
        raw += len(results)
        for r in results:
            url = r.get("url", "")
            if _should_skip(url):
                rejected += 1
                continue
            if not _identity_match_osu(r, idk):
                rejected += 1
                continue
            norm = _normalize_url(url)
            if norm in candidates:
                continue
            r["normalized_url"] = norm
            r["source_query"] = q
            candidates[norm] = r
        await asyncio.sleep(sleep_s)

    ranked = sorted(
        candidates.values(),
        key=lambda r: (_url_priority(r["normalized_url"]), len(r["normalized_url"])),
    )
    final = ranked[:max_urls]

    return {
        "name": idk["name"],
        "queries": queries,
        "raw_results": raw,
        "kept_urls": [r["normalized_url"] for r in final],
        "rejected_count": rejected,
        "results_detail": [
            {
                "url": r["normalized_url"],
                "title": r.get("title", ""),
                "snippet": r.get("snippet", "")[:200],
                "source_query": r.get("source_query", ""),
                "priority": _url_priority(r["normalized_url"]),
            }
            for r in final
        ],
    }


# ── Checkpoint / log helpers ───────────────────────────────────────────


def _load_ids_from_csv(path: Path, id_column: str) -> List[str]:
    """Read profile_ids from a CSV (e.g. osu_scholars/weak_scholars.csv).

    Handles cells that hold multiple '|'-joined ids. De-dupes, preserves order.
    """
    import csv as _csv

    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    ids: List[str] = []
    seen: Set[str] = set()
    with path.open(encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        if id_column not in (reader.fieldnames or []):
            raise ValueError(
                f"Column '{id_column}' not in {path}. Found: {reader.fieldnames}"
            )
        for row in reader:
            raw = (row.get(id_column) or "").strip()
            if not raw:
                continue
            for token in raw.split("|"):
                token = token.strip()
                if token and token not in seen:
                    seen.add(token)
                    ids.append(token)
    return ids


def _load_checkpoint(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_checkpoint(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _append_log(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ── Main ───────────────────────────────────────────────────────────────


async def main_async(args: argparse.Namespace) -> int:
    load_dotenv()
    uri = os.getenv("MONGODB_URI")
    if not uri:
        print("MONGODB_URI not set", file=sys.stderr)
        return 2

    client = create_mongo_client(uri)
    db = client[resolve_mongo_db_name(uri)]
    scholars_coll = db.scholars
    excel_by_profile_id, excel_by_name = _load_excel_identity_map(Path(args.excel_path))

    query: Dict[str, Any] = {"scholar_type": "osu"}
    if args.filter_field:
        query["metadata.field_of_study"] = {"$regex": args.filter_field, "$options": "i"}
    if args.filter_department:
        query["about.department"] = {"$regex": args.filter_department, "$options": "i"}
    if args.from_csv:
        target_ids = _load_ids_from_csv(Path(args.from_csv), args.csv_id_column)
        if not target_ids:
            print(f"[Collect] No ids found in {args.from_csv} (column '{args.csv_id_column}'). Nothing to do.")
            return 0
        query["_id"] = {"$in": target_ids}
        print(f"[Collect] CSV-targeted mode: {len(target_ids)} profile_id(s) from {args.from_csv}")

    cursor = scholars_coll.find(
        query,
        projection={
            "_id": 1,
            "professor_name": 1,
            "name": 1,
            "profile_id": 1,
            "metadata": 1,
            "about": 1,
            "background_and_work": 1,
        },
    ).sort("_id", 1)
    if args.start_from:
        cursor = cursor.skip(args.start_from)
    if args.limit:
        cursor = cursor.limit(args.limit)
    scholars = [_attach_excel_identity(sch, excel_by_profile_id, excel_by_name) for sch in cursor]

    print(f"[Collect] Targeting {len(scholars)} OSU scholars")
    if args.dry_run:
        print("[Collect] DRY RUN — no DDG calls, no writes.")
        for sch in scholars[:10]:
            idk = _identity_keywords(sch)
            print(f"  - {idk['name']!r:35s} field={idk['field']!r:30s} dept={idk['department']!r:30s}")
        if len(scholars) > 10:
            print(f"  … ({len(scholars) - 10} more)")
        return 0

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(args.checkpoint_file)
    log_path = Path(args.log_file)
    checkpoint = _load_checkpoint(ckpt_path)

    success = 0
    failed = 0
    skipped = 0
    no_urls = 0
    started_at = datetime.now(timezone.utc).isoformat()

    for idx, sch in enumerate(scholars, 1):
        pid = str(sch.get("profile_id") or sch["_id"])
        name = ((sch.get("name") or {}).get("full") or sch.get("professor_name") or "").strip()
        slug = _slugify(name or "scholar", pid)
        out_file = out_dir / f"{slug}.txt"
        meta_file = out_file.with_suffix(".meta.json")

        ckpt_entry = checkpoint.get(pid) or {}
        if args.skip_existing and ckpt_entry.get("status") == "done" and out_file.exists() and out_file.stat().st_size > 0:
            skipped += 1
            continue

        existing_urls: List[str] = []
        if args.merge_existing and out_file.exists():
            try:
                existing_urls = [
                    ln.strip() for ln in out_file.read_text(encoding="utf-8").splitlines()
                    if ln.strip() and not ln.strip().startswith("#")
                ]
            except Exception:
                existing_urls = []

        print(f"\n[{idx}/{len(scholars)}] {name!r} ({pid[:8]})"
              + (f" [aggressive, +{len(existing_urls)} existing]" if args.aggressive else ""))
        try:
            result = await collect_for_scholar(
                sch,
                max_urls=args.max_urls,
                sleep_s=args.sleep_seconds,
                per_query=args.per_query_results,
                aggressive=args.aggressive,
                existing_urls=existing_urls,
            )
        except Exception as e:
            print(f"  [ERROR] {e}")
            failed += 1
            checkpoint[pid] = {
                "status": "failed",
                "error": str(e),
                "name": name,
                "ts": datetime.now(timezone.utc).isoformat(),
            }
            _save_checkpoint(ckpt_path, checkpoint)
            continue

        urls = result["kept_urls"]
        if not urls:
            print(f"  [WARN] 0 URLs after identity filter (raw={result['raw_results']}, rejected={result['rejected_count']})")
            no_urls += 1
            checkpoint[pid] = {
                "status": "no_urls",
                "raw_results": result["raw_results"],
                "rejected_count": result["rejected_count"],
                "queries": result["queries"],
                "name": name,
                "ts": datetime.now(timezone.utc).isoformat(),
            }
            _save_checkpoint(ckpt_path, checkpoint)
            _append_log(log_path, {"profile_id": pid, "name": name, "result": result, "ts": started_at})
            continue

        # Write the .txt the legend pipeline consumes.
        with out_file.open("w", encoding="utf-8") as f:
            for u in urls:
                f.write(u + "\n")
        collector_version = "ddg_v3_osu_aggressive" if args.aggressive else "ddg_v2_osu_precision"
        meta_payload = {
            "profile_id": pid,
            "profile_name": name,
            "profile_url": sch.get("_excel_source_url") or (urls[0] if urls else ""),
            "email": sch.get("_excel_email") or "",
            "source_host": _domain_of(str(sch.get("_excel_source_url") or "")),
            "collector_version": collector_version,
        }
        meta_file.write_text(json.dumps(meta_payload, indent=2, ensure_ascii=False), encoding="utf-8")

        # Write `source` field back to the Mongo doc *immediately* so it's
        # queryable even before the downstream scrape pipeline runs.
        try:
            scholars_coll.update_one(
                {"_id": sch["_id"]},
                {
                    "$set": {
                        "source": urls,
                        "source_collected_at": datetime.now(timezone.utc).isoformat(),
                        "source_collection_meta": {
                            "raw_results": result["raw_results"],
                            "rejected_count": result["rejected_count"],
                            "queries_fired": len(result["queries"]),
                            "max_urls_cap": args.max_urls,
                            "collector_version": collector_version,
                        },
                    }
                },
            )
        except Exception as e:
            print(f"  [WARN] Mongo source-field update failed: {e}")

        success += 1
        print(f"  [OK] kept {len(urls)} URLs (raw={result['raw_results']}, rejected={result['rejected_count']}) -> {out_file.name}")
        checkpoint[pid] = {
            "status": "done",
            "url_count": len(urls),
            "raw_results": result["raw_results"],
            "rejected_count": result["rejected_count"],
            "name": name,
            "file": str(out_file),
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        _save_checkpoint(ckpt_path, checkpoint)
        _append_log(log_path, {"profile_id": pid, "name": name, "result": result, "ts": started_at})

    print()
    print(f"[Summary] success={success} no_urls={no_urls} failed={failed} skipped={skipped} total={len(scholars)}")
    print(f"[Summary] checkpoint: {ckpt_path}")
    print(f"[Summary] log:        {log_path}")
    print(f"[Summary] url files:  {out_dir}")
    return 0 if failed == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None, help="Process only first N scholars")
    parser.add_argument("--start-from", type=int, default=0, help="Skip first M scholars")
    parser.add_argument("--excel-path", default=DEFAULT_EXCEL_PATH, help="Canonical OSU workbook for identity context")
    parser.add_argument("--max-urls", type=int, default=DEFAULT_MAX_URLS, help=f"URL cap per scholar (default {DEFAULT_MAX_URLS})")
    parser.add_argument("--sleep-seconds", type=float, default=DEFAULT_SLEEP_S, help="Sleep between DDG queries")
    parser.add_argument("--per-query-results", type=int, default=DEFAULT_PER_QUERY, help="Results pulled per DDG query")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint-file", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--log-file", default=DEFAULT_LOG)
    parser.add_argument("--filter-field", default=None, help="Regex on metadata.field_of_study")
    parser.add_argument("--filter-department", default=None, help="Regex on about.department")
    parser.add_argument("--skip-existing", action="store_true", help="Skip scholars already collected")
    parser.add_argument("--dry-run", action="store_true", help="Print plan; don't search or write")
    parser.add_argument("--from-csv", default=None,
                        help="Only collect for profile_ids listed in this CSV "
                             "(e.g. osu_scholars/weak_scholars.csv). Targeted re-collection.")
    parser.add_argument("--csv-id-column", default="profile_id",
                        help="Column in --from-csv holding the profile_id (default: profile_id).")
    parser.add_argument("--aggressive", action="store_true",
                        help="Exhaustive query suite (identity-domain site: queries + "
                             "CV/lab/award/news/publication angles). Accuracy is unchanged: "
                             "every candidate still passes the strict identity filter.")
    parser.add_argument("--merge-existing", dest="merge_existing", action="store_true", default=True,
                        help="Merge with already-collected URLs so coverage never regresses (default on).")
    parser.add_argument("--no-merge-existing", dest="merge_existing", action="store_false",
                        help="Overwrite existing URL files instead of merging.")
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
