"""
Collect URLs about each legendary scholar via DuckDuckGo and write one
URL-list .txt file per scholar.

Input:
  - excel/legendary.xlsx

Output:
  - legendary_scholars/final/*.txt

The workbook is treated as the canonical scholar list. This script only
collects URL lists; downstream scraping/chunking/storage is handled by
run_legendary_enrichment.py.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import parse_qs, unquote, urlencode, urlparse
from urllib.request import Request, urlopen

import pandas as pd
from dotenv import load_dotenv

from collect_osu_scholar_urls import _append_log
from collect_osu_scholar_urls import _ddg_search
from collect_osu_scholar_urls import _domain_of
from collect_osu_scholar_urls import _normalize_url
from collect_osu_scholar_urls import _save_checkpoint
from collect_osu_scholar_urls import _should_skip


DEFAULT_EXCEL_PATH = "excel/legendary.xlsx"
DEFAULT_OUTPUT_DIR = "legendary_scholars/final"
DEFAULT_CHECKPOINT = "legendary_scholars/url_collection_checkpoint.json"
DEFAULT_LOG = "legendary_scholars/url_collection_log.jsonl"
DEFAULT_MAX_URLS = 50
DEFAULT_SLEEP_S = 2.0
DEFAULT_PER_QUERY = 15
WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"

TRUSTED_BIO_DOMAINS = (
    "wikipedia.org",
    "blackpast.org",
    "britannica.com",
    "encyclopedia.com",
    "biography.com",
    "poetryfoundation.org",
    "thehistorymakers.org",
    "federalreservehistory.org",
    "si.edu",
    "loc.gov",
    "archives.gov",
    "nasa.gov",
    "nih.gov",
    "nps.gov",
    "pbs.org",
)

LEGENDARY_PRIORITY_DOMAINS = (
    "blackpast.org",
    "britannica.com",
    "thehistorymakers.org",
    "wikipedia.org",
    "loc.gov",
    "archives.gov",
    "si.edu",
    "nasa.gov",
    "nih.gov",
    "federalreservehistory.org",
    "nps.gov",
    "pbs.org",
    "encyclopedia.com",
    "biography.com",
    "poetryfoundation.org",
)

LEGENDARY_EXCLUDE_DOMAINS = (
    "openalex.org",
    "orcid.org",
    "scholar.google.com",
    "semanticscholar.org",
    "researchgate.net",
    "academia.edu",
    "dblp.org",
)

WIKIPEDIA_EXTLINK_EXCLUDE_DOMAINS = (
    "viaf.org",
    "idref.fr",
    "isni.org",
    "snaccooperative.org",
    "books.google.com",
    "patents.google.com",
    "doi.org",
    "jstor.org",
    "newspapers.com",
    "amazon.com",
    "worldcat.org",
    "id.worldcat.org",
    "id.oclc.org",
    "id.loc.gov",
    "data.bibliotheken.nl",
    "d-nb.info",
)

FIELD_ALIASES = {
    "opthalmology": "ophthalmology",
}

BIOGRAPHY_KEYWORDS = (
    "biography",
    "biographical",
    "professor",
    "scholar",
    "scientist",
    "historian",
    "economist",
    "author",
    "researcher",
    "educator",
    "inventor",
    "physician",
    "chemist",
    "sociologist",
    "anthropologist",
    "lawyer",
    "jurist",
)

WORK_KEYWORDS = (
    "books",
    "works",
    "articles",
    "papers",
    "bibliography",
    "interview",
    "lecture",
    "oral history",
    "address",
    "essay",
    "manuscript",
    "collection",
)

HISTORICAL_KEYWORDS = (
    "archive",
    "archives",
    "collection",
    "papers",
    "oral history",
    "history",
    "legacy",
    "memorial",
    "obituary",
    "tribute",
    "tributes",
    "remembered",
    "honors",
)

LEGENDARY_BAD_RESULT_PATTERNS = (
    "athletics",
    "sports",
    "imdb",
    "findagrave",
    "legacy.com",
    "funeral home",
)


@dataclass
class ScholarRow:
    row_number: int
    scholar_id: str
    full_name: str
    field: str
    slug: str


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "legend-scholar"


def _clean_cell(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _load_slugs_from_file(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Slugs file not found: {path}")
    seen = set()
    slugs: List[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        slug = raw_line.strip()
        if not slug or slug.startswith("#") or slug in seen:
            continue
        seen.add(slug)
        slugs.append(slug)
    return slugs


def _field_variants(field: str) -> List[str]:
    base = field.strip().lower()
    if not base:
        return []
    variants = [base]
    alias = FIELD_ALIASES.get(base)
    if alias and alias not in variants:
        variants.append(alias)
    return variants


def _field_tokens(field: str) -> List[str]:
    tokens = set()
    for variant in _field_variants(field):
        for token in re.findall(r"[a-z0-9]+", variant):
            if len(token) >= 5:
                tokens.add(token)
    return sorted(tokens)


def _normalize_person_text(value: str) -> str:
    text = (value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _person_name_matches(candidate: str, scholar: ScholarRow) -> bool:
    candidate_norm = _normalize_person_text(candidate)
    target_norm = _normalize_person_text(scholar.full_name)
    if not candidate_norm or not target_norm:
        return False
    if candidate_norm == target_norm:
        return True

    candidate_parts = candidate_norm.split()
    target_parts = target_norm.split()
    if not candidate_parts or not target_parts:
        return False

    candidate_first = candidate_parts[0]
    target_first = target_parts[0]
    candidate_last = candidate_parts[-1]
    target_last = target_parts[-1]
    if candidate_last != target_last:
        return False
    return (
        candidate_first == target_first
        or candidate_first.startswith(target_first[:4])
        or target_first.startswith(candidate_first[:4])
    )


def _get_json(url: str, params: Dict[str, Any], headers: Dict[str, str] | None = None) -> Dict[str, Any]:
    req = Request(
        url + "?" + urlencode(params),
        headers=headers or {},
    )
    with urlopen(req, timeout=25) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _legendary_url_priority(url: str) -> int:
    domain = _domain_of(url)
    if not domain:
        return 999
    for idx, item in enumerate(LEGENDARY_PRIORITY_DOMAINS):
        if domain == item or domain.endswith("." + item):
            return idx
    if domain.endswith(".edu"):
        return 50
    if domain.endswith(".gov"):
        return 60
    if domain.endswith(".org"):
        return 70
    return 100


def _legendary_should_skip(url: str) -> bool:
    if not url or not url.startswith(("http://", "https://")):
        return True
    domain = _domain_of(url)
    if not domain:
        return True
    if any(domain == item or domain.endswith("." + item) for item in LEGENDARY_EXCLUDE_DOMAINS):
        return True
    if any(domain == item or domain.endswith("." + item) for item in TRUSTED_BIO_DOMAINS):
        return False
    return _should_skip(url)


def _filter_wikipedia_extlinks(urls: List[str]) -> List[str]:
    kept: List[str] = []
    seen = set()
    for url in urls:
        normalized = _normalize_url(url)
        domain = _domain_of(normalized)
        if not domain:
            continue
        if any(domain == item or domain.endswith("." + item) for item in WIKIPEDIA_EXTLINK_EXCLUDE_DOMAINS):
            continue
        if _legendary_should_skip(normalized):
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        kept.append(normalized)
    kept.sort(key=lambda item: (_legendary_url_priority(item), len(item)))
    return kept[:8]


def _wikipedia_seed_sync(scholar: ScholarRow) -> List[str]:
    email = os.getenv("OPENALEX_EMAIL") or "research@example.com"
    headers = {
        "User-Agent": f"legendary-scholar-enrichment/1.0 (research; {email})",
        "Accept": "application/json",
    }
    search_terms = [
        scholar.full_name,
        f"{scholar.full_name} scholar",
        f"{scholar.full_name} scientist",
        f"{scholar.full_name} professor",
    ]
    page_title = ""
    for term in search_terms:
        try:
            search = _get_json(
                WIKIPEDIA_API,
                {
                    "action": "query",
                    "list": "search",
                    "srsearch": term,
                    "srlimit": 8,
                    "format": "json",
                },
                headers=headers,
            )
        except Exception:
            continue
        candidates = (search.get("query") or {}).get("search") or []
        for candidate in candidates:
            title = candidate.get("title", "")
            if _person_name_matches(title, scholar):
                page_title = title
                break
        if page_title:
            break
    if not page_title:
        return []

    try:
        detail = _get_json(
            WIKIPEDIA_API,
            {
                "action": "query",
                "prop": "info",
                "titles": page_title,
                "inprop": "url",
                "format": "json",
            },
            headers=headers,
        )
    except Exception:
        return []
    pages = (detail.get("query") or {}).get("pages") or {}
    if not pages:
        return []
    page = next(iter(pages.values()))
    url = page.get("fullurl") or ""
    if not url:
        return []

    extlinks: List[str] = []
    try:
        extra = _get_json(
            WIKIPEDIA_API,
            {
                "action": "query",
                "prop": "extlinks",
                "titles": page_title,
                "ellimit": 50,
                "format": "json",
            },
            headers=headers,
        )
        extra_pages = (extra.get("query") or {}).get("pages") or {}
        extra_page = next(iter(extra_pages.values())) if extra_pages else {}
        extlinks = [item.get("*", "") for item in (extra_page.get("extlinks") or [])]
    except Exception:
        extlinks = []

    return [_normalize_url(url), *_filter_wikipedia_extlinks(extlinks)]


async def _seed_urls_for_scholar(scholar: ScholarRow) -> List[str]:
    loop = asyncio.get_event_loop()
    sources = await loop.run_in_executor(
        None,
        lambda: _wikipedia_seed_sync(scholar),
    )
    seen = set()
    ordered: List[str] = []
    for url in sources:
        normalized = _normalize_url(url)
        if normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def _parse_ddg_html(html: str) -> List[Dict[str, str]]:
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return []
    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception:
        return []
    out: List[Dict[str, str]] = []
    for a in soup.select("a.result__a"):
        href = a.get("href") or ""
        if "/l/?" in href:
            qs = parse_qs(urlparse(href).query)
            real = qs.get("uddg") or qs.get("u")
            if real:
                href = unquote(real[0])
        if href.startswith("http"):
            out.append({"url": href, "title": a.get_text(strip=True), "snippet": ""})
    return out


def _ddg_html_search_sync(query: str, max_results: int) -> List[Dict[str, str]]:
    url = "https://html.duckduckgo.com/html/?" + urlencode({"q": query})
    req = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    try:
        with urlopen(req, timeout=25) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
    except Exception:
        return []
    return _parse_ddg_html(html)[:max_results]


async def _search_query(query: str, max_results: int) -> List[Dict[str, str]]:
    results = await _ddg_search(query, max_results)
    if results:
        return results
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: _ddg_html_search_sync(query, max_results))


def _load_scholars(excel_path: Path) -> List[ScholarRow]:
    df = pd.read_excel(excel_path, header=2)
    required = ["Last Name", "First Name", "Field"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"legendary workbook missing required columns: {missing}")

    rows: List[ScholarRow] = []
    seen_slugs: Dict[str, int] = {}
    for idx, row in df.iterrows():
        last = _clean_cell(row.get("Last Name"))
        first = _clean_cell(row.get("First Name"))
        middle = _clean_cell(row.get("Middle Name"))
        field = _clean_cell(row.get("Field"))
        name_parts = [part for part in [first, middle, last] if part]
        full_name = " ".join(name_parts).strip()
        if not full_name:
            continue
        base_slug = _slugify(full_name)
        seen_slugs[base_slug] = seen_slugs.get(base_slug, 0) + 1
        slug = base_slug if seen_slugs[base_slug] == 1 else f"{base_slug}-{seen_slugs[base_slug]}"
        rows.append(
            ScholarRow(
                row_number=int(idx) + 4,
                scholar_id=slug,
                full_name=full_name,
                field=field,
                slug=slug,
            )
        )
    return rows


def _build_queries(scholar: ScholarRow) -> List[str]:
    name = scholar.full_name
    field_variants = _field_variants(scholar.field)
    queries = [
        f'"{name}" biography',
        f'"{name}" obituary',
        f'"{name}" archive',
        f'"{name}" "oral history"',
        f'"{name}" legacy',
        f'"{name}" papers',
        f'"{name}" site:.edu',
        f'"{name}" site:.gov',
        f'"{name}" site:blackpast.org',
        f'"{name}" site:britannica.com',
        f'"{name}" site:thehistorymakers.org',
        f'"{name}" site:loc.gov',
    ]
    for field in field_variants:
        queries.extend(
            [
                f"{name} {field}",
                f"{name} {field} history",
                f'"{name}" "{field}" biography',
                f'"{name}" "{field}" archive',
                f'"{name}" "{field}" legacy',
                f'"{name}" "{field}" site:.edu',
                f'"{name}" "{field}" site:.gov',
            ]
        )
    seen = set()
    out: List[str] = []
    for query in queries:
        key = query.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(query)
    return out


def _identity_match(result: Dict[str, str], scholar: ScholarRow) -> bool:
    title = (result.get("title") or "")
    snippet = (result.get("snippet") or "")
    blob = f"{title} {snippet}".lower()
    url = result.get("url", "") or ""
    url_lc = url.lower()
    text_blob = f"{title} {snippet} {url_lc}".lower()

    if any(pattern in text_blob for pattern in LEGENDARY_BAD_RESULT_PATTERNS):
        return False

    name = scholar.full_name.lower()
    parts = [part.lower() for part in scholar.full_name.split() if part]
    first = parts[0] if parts else ""
    last = parts[-1] if len(parts) >= 2 else ""
    name_compact = re.sub(r"[^a-z0-9]+", "", name)
    url_compact = re.sub(r"[^a-z0-9]+", "", url_lc)
    field_variants = _field_variants(scholar.field)
    field_tokens = _field_tokens(scholar.field)

    full_name_in_blob = bool(name and name in blob)
    first_last_in_blob = bool(first and last and first in blob and last in blob)
    full_name_in_url = bool(name_compact and name_compact in url_compact)
    first_last_in_url = bool(first and last and first in url_lc and last in url_lc)

    base_name_signal = full_name_in_blob or first_last_in_blob or full_name_in_url or first_last_in_url
    if not base_name_signal:
        return False

    domain = _domain_of(url)
    field_signal = any(field_variant in blob for field_variant in field_variants)
    if not field_signal and field_tokens:
        field_signal = any(token in blob for token in field_tokens)
    biography_signal = any(keyword in blob for keyword in BIOGRAPHY_KEYWORDS)
    work_signal = any(keyword in blob for keyword in WORK_KEYWORDS)
    historical_signal = any(keyword in blob for keyword in HISTORICAL_KEYWORDS)
    trusted_domain = any(domain == item or domain.endswith("." + item) for item in TRUSTED_BIO_DOMAINS)
    strong_name_signal = full_name_in_blob or full_name_in_url

    if any(domain == item or domain.endswith("." + item) for item in LEGENDARY_EXCLUDE_DOMAINS):
        return False
    if domain.endswith(".edu") or domain.endswith(".gov"):
        return strong_name_signal or (
            base_name_signal and (field_signal or biography_signal or work_signal or historical_signal)
        )
    if trusted_domain:
        return strong_name_signal and (field_signal or biography_signal or work_signal or historical_signal)
    if domain.endswith(".org"):
        return strong_name_signal and (field_signal or biography_signal or historical_signal)
    return strong_name_signal and field_signal and (biography_signal or work_signal or historical_signal)


async def collect_for_scholar(
    scholar: ScholarRow,
    *,
    max_urls: int,
    sleep_s: float,
    per_query: int,
) -> Dict[str, Any]:
    queries = _build_queries(scholar)
    candidates: Dict[str, Dict[str, Any]] = {}
    rejected = 0
    raw = 0

    for url in await _seed_urls_for_scholar(scholar):
        candidates[url] = {
            "normalized_url": url,
            "title": "Seed identity source",
            "snippet": "",
            "source_query": "seed_source",
        }

    enough_seed_sources = len(candidates) >= min(max_urls, 4)
    if enough_seed_sources:
        ranked = sorted(
            candidates.values(),
            key=lambda item: (_legendary_url_priority(item["normalized_url"]), len(item["normalized_url"])),
        )
        final = ranked[:max_urls]
        return {
            "queries": queries,
            "raw_results": raw,
            "kept_urls": [item["normalized_url"] for item in final],
            "rejected_count": rejected,
            "results_detail": [
                {
                    "url": item["normalized_url"],
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", "")[:200],
                    "source_query": item.get("source_query", ""),
                    "priority": _legendary_url_priority(item["normalized_url"]),
                }
                for item in final
            ],
        }

    for query in queries:
        results = await _search_query(query, per_query)
        raw += len(results)
        for result in results:
            url = result.get("url", "")
            if _legendary_should_skip(url):
                rejected += 1
                continue
            if not _identity_match(result, scholar):
                rejected += 1
                continue
            normalized = _normalize_url(url)
            if normalized in candidates:
                continue
            result["normalized_url"] = normalized
            result["source_query"] = query
            candidates[normalized] = result
        await asyncio.sleep(sleep_s)

    ranked = sorted(
        candidates.values(),
        key=lambda item: (_legendary_url_priority(item["normalized_url"]), len(item["normalized_url"])),
    )
    final = ranked[:max_urls]
    return {
        "queries": queries,
        "raw_results": raw,
        "kept_urls": [item["normalized_url"] for item in final],
        "rejected_count": rejected,
        "results_detail": [
            {
                "url": item["normalized_url"],
                "title": item.get("title", ""),
                "snippet": item.get("snippet", "")[:200],
                "source_query": item.get("source_query", ""),
                "priority": _legendary_url_priority(item["normalized_url"]),
            }
            for item in final
        ],
    }


async def main_async(args: argparse.Namespace) -> int:
    load_dotenv()
    excel_path = Path(args.excel_path)
    if not excel_path.exists():
        raise FileNotFoundError(f"Workbook not found: {excel_path}")

    scholars = _load_scholars(excel_path)
    if args.slugs_file:
        target_slugs = set(_load_slugs_from_file(Path(args.slugs_file)))
        scholars = [scholar for scholar in scholars if scholar.slug in target_slugs]
    if args.start_from:
        scholars = scholars[args.start_from:]
    if args.limit is not None:
        scholars = scholars[:args.limit]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(args.checkpoint_file)
    log_path = Path(args.log_file)
    checkpoint = {}
    if checkpoint_path.exists():
        try:
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        except Exception:
            checkpoint = {}

    success = 0
    failed = 0
    skipped = 0
    no_urls = 0
    started_at = datetime.now(timezone.utc).isoformat()

    print(f"[LegendCollect] Targeting {len(scholars)} legendary scholars")

    for idx, scholar in enumerate(scholars, 1):
        out_file = out_dir / f"{scholar.slug}.txt"
        meta_file = out_file.with_suffix(".meta.json")
        ckpt_entry = checkpoint.get(scholar.scholar_id) or {}
        if (
            args.skip_existing
            and ckpt_entry.get("status") == "done"
            and out_file.exists()
            and out_file.stat().st_size > 0
        ):
            skipped += 1
            continue

        print(f"\n[{idx}/{len(scholars)}] {scholar.full_name!r} field={scholar.field!r}")
        try:
            result = await collect_for_scholar(
                scholar,
                max_urls=args.max_urls,
                sleep_s=args.sleep_seconds,
                per_query=args.per_query_results,
            )
        except Exception as exc:
            failed += 1
            checkpoint[scholar.scholar_id] = {
                "status": "failed",
                "error": str(exc),
                "name": scholar.full_name,
                "field": scholar.field,
                "row_number": scholar.row_number,
                "ts": datetime.now(timezone.utc).isoformat(),
            }
            _save_checkpoint(checkpoint_path, checkpoint)
            print(f"  [ERROR] {exc}")
            continue

        urls = result["kept_urls"]
        if not urls:
            no_urls += 1
            checkpoint[scholar.scholar_id] = {
                "status": "no_urls",
                "name": scholar.full_name,
                "field": scholar.field,
                "row_number": scholar.row_number,
                "raw_results": result["raw_results"],
                "rejected_count": result["rejected_count"],
                "queries": result["queries"],
                "ts": datetime.now(timezone.utc).isoformat(),
            }
            _save_checkpoint(checkpoint_path, checkpoint)
            _append_log(
                log_path,
                {
                    "scholar_id": scholar.scholar_id,
                    "name": scholar.full_name,
                    "field": scholar.field,
                    "result": result,
                    "ts": started_at,
                },
            )
            print(
                f"  [WARN] 0 URLs after identity filter "
                f"(raw={result['raw_results']}, rejected={result['rejected_count']})"
            )
            continue

        with out_file.open("w", encoding="utf-8") as handle:
            for url in urls:
                handle.write(url + "\n")
        meta_payload = {
            "profile_id": scholar.scholar_id,
            "profile_name": scholar.full_name,
            "profile_url": urls[0] if urls else "",
            "field": scholar.field,
            "row_number": scholar.row_number,
            "collector_version": "ddg_v2_legendary_precision",
        }
        meta_file.write_text(json.dumps(meta_payload, indent=2, ensure_ascii=False), encoding="utf-8")

        checkpoint[scholar.scholar_id] = {
            "status": "done",
            "name": scholar.full_name,
            "field": scholar.field,
            "row_number": scholar.row_number,
            "file": str(out_file),
            "url_count": len(urls),
            "raw_results": result["raw_results"],
            "rejected_count": result["rejected_count"],
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        _save_checkpoint(checkpoint_path, checkpoint)
        _append_log(
            log_path,
            {
                "scholar_id": scholar.scholar_id,
                "name": scholar.full_name,
                "field": scholar.field,
                "result": result,
                "ts": started_at,
            },
        )
        success += 1
        print(
            f"  [OK] kept {len(urls)} URLs "
            f"(raw={result['raw_results']}, rejected={result['rejected_count']})"
        )

    print()
    print(
        f"[LegendCollect] success={success} no_urls={no_urls} "
        f"failed={failed} skipped={skipped} total={len(scholars)}"
    )
    print(f"[LegendCollect] checkpoint: {checkpoint_path}")
    print(f"[LegendCollect] log:        {log_path}")
    print(f"[LegendCollect] url files:  {out_dir}")
    return 0 if failed == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--excel-path", default=DEFAULT_EXCEL_PATH)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-from", type=int, default=0)
    parser.add_argument("--max-urls", type=int, default=DEFAULT_MAX_URLS)
    parser.add_argument("--sleep-seconds", type=float, default=DEFAULT_SLEEP_S)
    parser.add_argument("--per-query-results", type=int, default=DEFAULT_PER_QUERY)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint-file", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--log-file", default=DEFAULT_LOG)
    parser.add_argument(
        "--slugs-file",
        default=None,
        help="Optional newline-delimited slug file to target a subset of scholars.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
