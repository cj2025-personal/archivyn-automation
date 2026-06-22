"""
Source guardrail utilities for URL normalization, hashing, language/PII detection,
quality metrics, and similarity signatures.
"""
from __future__ import annotations

import hashlib
import re
from typing import Dict, Iterable, Optional, Tuple
from urllib.parse import urlparse, urlunparse


_EN_STOPWORDS = {
    "the", "and", "of", "to", "in", "for", "on", "with", "as", "by", "at",
    "from", "that", "this", "it", "is", "are", "was", "were", "be", "been",
    "or", "an", "a", "but", "if", "then", "so", "than", "which", "who",
    "whom", "their", "there", "these", "those", "its", "into", "about",
}

_BOILERPLATE_PATTERNS = [
    r"\bprivacy\b",
    r"\bterms\b",
    r"\bcookies?\b",
    r"\ball rights reserved\b",
    r"\bsubscribe\b",
    r"\blog\s*in\b",
    r"\bmenu\b",
    r"\bskip to content\b",
    r"\bback to top\b",
]

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE_RE = re.compile(r"\b(?:\+?1[\s\-\.])?\(?\d{3}\)?[\s\-\.]?\d{3}[\s\-\.]?\d{4}\b")
_ADDRESS_RE = re.compile(r"\b\d{1,6}\s+\w+(?:\s+\w+){0,3}\s+(?:st|street|ave|avenue|rd|road|blvd|boulevard|ln|lane|dr|drive)\b", re.I)


def normalize_url(url: str) -> str:
    """Normalize URL for stable hashing."""
    if not url:
        return ""
    url = url.strip()
    try:
        parsed = urlparse(url)
    except Exception:
        return url
    scheme = (parsed.scheme or "https").lower()
    netloc = (parsed.netloc or "").lower()
    # Drop default ports
    if netloc.endswith(":80") and scheme == "http":
        netloc = netloc[:-3]
    if netloc.endswith(":443") and scheme == "https":
        netloc = netloc[:-4]
    path = parsed.path or "/"
    # Remove fragment, keep query
    fragment = ""
    # Normalize trailing slash (but keep root)
    if path != "/" and path.endswith("/"):
        path = path[:-1]
    return urlunparse((scheme, netloc, path, "", parsed.query, fragment))


def make_source_id(url: str) -> str:
    canonical = normalize_url(url)
    digest = hashlib.sha256(canonical.encode("utf-8", errors="ignore")).hexdigest()
    return f"src_{digest}"


def normalize_text_for_hash(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text.strip())
    return text.lower()


def compute_text_hash(text: str) -> str:
    norm = normalize_text_for_hash(text)
    digest = hashlib.sha256(norm.encode("utf-8", errors="ignore")).hexdigest()
    return digest


def compute_simhash(tokens: Iterable[str]) -> str:
    """Compute a 64-bit simhash from tokens."""
    v = [0] * 64
    for tok in tokens:
        if not tok:
            continue
        h = hashlib.md5(tok.encode("utf-8", errors="ignore")).digest()
        h64 = int.from_bytes(h[:8], "big", signed=False)
        for i in range(64):
            bit = (h64 >> i) & 1
            v[i] += 1 if bit else -1
    fingerprint = 0
    for i, val in enumerate(v):
        if val > 0:
            fingerprint |= 1 << i
    return f"0x{fingerprint:016x}"


_FR_STOPWORDS = {
    "le", "la", "les", "et", "de", "des", "du", "un", "une", "que", "qui",
    "pour", "dans", "sur", "avec", "par", "ses", "son", "sa", "est", "sont",
    "ce", "ces", "cette", "au", "aux", "ne", "pas", "plus", "comme",
}
_ES_STOPWORDS = {
    "el", "la", "los", "las", "y", "de", "del", "que", "en", "un", "una",
    "por", "con", "para", "como", "es", "son", "su", "sus", "no", "se",
    "lo", "le", "al", "este", "esta",
}
_DE_STOPWORDS = {
    "der", "die", "das", "und", "ist", "sind", "ein", "eine", "von", "zu",
    "mit", "den", "des", "dem", "im", "auf", "fur", "nicht", "auch",
    "wird", "war", "als",
}


def detect_language(text: str) -> Tuple[str, float]:
    """Detect language, returning (lang_code, confidence).

    Goes beyond pure-English heuristics so that French/Spanish/German content
    is not silently labelled ``unknown`` (which previously caused all
    Francophone Diop sources to fall through with no language tag).
    """
    if not text or len(text) < 40:
        return "unknown", 0.0
    # Try langdetect if available
    try:
        from langdetect import detect_langs  # type: ignore
        langs = detect_langs(text)
        if langs:
            top = langs[0]
            return getattr(top, "lang", "unknown"), float(getattr(top, "prob", 0.0))
    except Exception:
        pass

    tokens = re.findall(r"[A-Za-zÀ-ÿ]{2,}", text.lower())
    if not tokens:
        return "unknown", 0.0
    counts = {
        "en": sum(1 for t in tokens if t in _EN_STOPWORDS),
        "fr": sum(1 for t in tokens if t in _FR_STOPWORDS),
        "es": sum(1 for t in tokens if t in _ES_STOPWORDS),
        "de": sum(1 for t in tokens if t in _DE_STOPWORDS),
    }
    total = max(1, len(tokens))
    best_lang = max(counts, key=counts.get)
    ratio = counts[best_lang] / total
    if ratio >= 0.10:
        # Mild floor avoids 0.0-confidence labels; cap so heuristic never
        # impersonates an actual classifier.
        return best_lang, min(0.85, 0.35 + ratio)
    return "unknown", round(0.1 * ratio, 4)


def detect_pii(text: str) -> Dict[str, bool]:
    if not text:
        return {"contains_email": False, "contains_phone": False, "contains_address": False}
    return {
        "contains_email": bool(_EMAIL_RE.search(text)),
        "contains_phone": bool(_PHONE_RE.search(text)),
        "contains_address": bool(_ADDRESS_RE.search(text)),
    }


def compute_quality_metrics(text: str) -> Dict[str, float]:
    if not text:
        return {
            "word_count": 0,
            "line_count": 0,
            "ascii_ratio": 0.0,
            "boilerplate_ratio": 0.0,
        }
    length = len(text)
    ascii_chars = sum(1 for ch in text if 32 <= ord(ch) < 127)
    ascii_ratio = ascii_chars / max(1, length)
    words = re.findall(r"\w+", text)
    lines = [ln for ln in text.splitlines() if ln.strip()]
    boilerplate_hits = 0
    for ln in lines:
        ln_lc = ln.lower()
        if any(re.search(pat, ln_lc) for pat in _BOILERPLATE_PATTERNS):
            boilerplate_hits += 1
    boilerplate_ratio = boilerplate_hits / max(1, len(lines))
    return {
        "word_count": float(len(words)),
        "line_count": float(len(lines)),
        "ascii_ratio": round(ascii_ratio, 4),
        "boilerplate_ratio": round(boilerplate_ratio, 4),
    }


_TITLE_NOISE_RE = re.compile(
    r"(?i)^("
    r"loading|sign in|log in|search\b|menu|skip to content|cookie|"
    r"shopping basket|shopping cart|home page|return to top|back to top|"
    r"site archived|=== ?profile page|=== ?seed url"
    r")"
)


def extract_title_from_text(text: str, max_len: int = 120) -> str:
    """Best-effort title extraction from text.

    Skips obvious nav/UI strings ("Shopping Basket", "Loading...", "Sign In",
    "=== PROFILE PAGE ===" markers etc.) so the returned title is more likely
    to be the real document title rather than chrome.
    """
    if not text:
        return ""
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        if len(ln) < 4:
            continue
        if _TITLE_NOISE_RE.match(ln):
            continue
        if len(ln) > max_len:
            return ln[:max_len].rstrip()
        return ln
    return ""


# Allowed-use tiers used downstream for ``quote_ok``.
ALLOWED_USE_FACTS_ONLY = "facts_only"
ALLOWED_USE_SHORT_QUOTES = "short_quotes"
ALLOWED_USE_FULL_TEXT = "full_text"


# Domains/path patterns that are reliably public-domain or government works.
_PUBLIC_DOMAIN_DOMAINS = {
    # US government — works of the federal government are public domain.
    "loc.gov", "tile.loc.gov", "www.loc.gov",
    "congress.gov", "www.congress.gov",
    "govinfo.gov", "www.govinfo.gov",
    "house.gov", "history.house.gov",
    "senate.gov",
    "nps.gov", "www.nps.gov",
    "nih.gov", "www.nih.gov", "www.nichd.nih.gov",
    "nist.gov", "www.nist.gov",
    "files.eric.ed.gov",
    # State / public archives that the project trusts as PD-equivalent.
    "aahc.nc.gov",
}

# Domains hosting CC-licensed or "fair-use-OK for short quotes" reference material.
_SHORT_QUOTES_DOMAINS = {
    "wikipedia.org", "en.wikipedia.org",
    "wikisource.org",
    "blackpast.org", "www.blackpast.org",
    "britannica.com", "www.britannica.com",
    "smithsonianmag.com", "americanhistory.si.edu", "learninglab.si.edu",
    "pbs.org", "www.pbs.org",
    "100.duke.edu", "today.duke.edu", "founders.duke.edu", "gradschool.duke.edu",
    "news.harvard.edu", "gsas.harvard.edu", "legacyofslavery.harvard.edu",
    "hutchinscenter.fas.harvard.edu", "www.gse.harvard.edu",
    "news.uchicago.edu", "bmrc.lib.uchicago.edu",
    "news.virginia.edu", "woodson.as.virginia.edu", "www.as.virginia.edu",
    "www.rutgers.edu", "africanastudies.rutgers.edu", "rutgersblackalumni.org",
    "docsouth.unc.edu", "findingaids.library.umass.edu",
    "africasocialwork.net",
    "www.historians.org", "www.aaihs.org",
    "www.berea.edu", "libraryguides.berea.edu",
    "aahc.nc.gov",
    # Indiana University faculty / school subdomains used by the IU
    # scholars run; without these every IU chunk stayed quote_ok=False.
    "iu.edu", "indiana.edu", "iub.edu",
    "law.indiana.edu", "spea.indiana.edu", "kelley.indiana.edu",
    "oneill.indiana.edu", "linguistics.indiana.edu",
    # Author-hosted preprint / faculty pages.
    "academia.edu",
    "ssrn.com", "papers.ssrn.com",
    "nber.org", "www.nber.org",
    "ideas.repec.org", "econpapers.repec.org",
}


def _domain_matches(domain: str, allowed: Iterable[str]) -> bool:
    if not domain:
        return False
    d = domain.lower().lstrip(".")
    for entry in allowed:
        e = entry.lower().lstrip(".")
        if d == e or d.endswith("." + e):
            return True
    return False


def infer_license_for_url(url: str) -> Dict[str, str]:
    """Heuristic license/usage inference based on URL.

    Returns a dict with ``license_type``, ``allowed_use``, ``rights_holder``
    and ``license_url``. Falls back to ``unknown`` / ``facts_only`` when the
    domain is not on the known list. Always conservative: a wrong call here
    only ever shrinks ``allowed_use`` downstream consumers can act on.
    """
    if not url:
        return {
            "license_type": "unknown",
            "allowed_use": ALLOWED_USE_FACTS_ONLY,
            "rights_holder": "",
            "license_url": "",
        }
    try:
        parsed = urlparse(url if "://" in url else f"https://{url}")
        domain = (parsed.netloc or "").lower()
    except Exception:
        domain = ""
    path = ""
    try:
        path = (urlparse(url).path or "").lower()
    except Exception:
        pass

    if domain.endswith(".gov") or _domain_matches(domain, _PUBLIC_DOMAIN_DOMAINS):
        return {
            "license_type": "public_domain_us_gov",
            "allowed_use": ALLOWED_USE_FULL_TEXT,
            "rights_holder": "U.S. Government",
            "license_url": "",
        }
    if domain.endswith("wikipedia.org") or domain.endswith("wikisource.org"):
        return {
            "license_type": "cc_by_sa",
            "allowed_use": ALLOWED_USE_SHORT_QUOTES,
            "rights_holder": "Wikimedia contributors",
            "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        }
    if domain == "archive.org" or domain.endswith(".archive.org"):
        # Archive.org streams may host PD or CC material; treat as short quotes.
        if "/stream/" in path or "/details/" in path:
            return {
                "license_type": "archive_org_streamed",
                "allowed_use": ALLOWED_USE_SHORT_QUOTES,
                "rights_holder": "Internet Archive (varies by item)",
                "license_url": "",
            }
    if _domain_matches(domain, _SHORT_QUOTES_DOMAINS):
        return {
            "license_type": "fair_use_short_quote",
            "allowed_use": ALLOWED_USE_SHORT_QUOTES,
            "rights_holder": "",
            "license_url": "",
        }
    return {
        "license_type": "unknown",
        "allowed_use": ALLOWED_USE_FACTS_ONLY,
        "rights_holder": "",
        "license_url": "",
    }


# Domains that almost never carry biographical content for an arbitrary subject
# and instead pollute the corpus with UI chrome, navigation, or unrelated material.
_NOISE_DOMAIN_PATTERNS = (
    # Help / support / asset hosts
    "help.pbs.org",
    # Paste / paper-thumbnail hosts (often serve listing pages, not full text)
    "scribd.com",
    "fr.scribd.com",
    "es.scribd.com",
    "pdfcoffee.com",
    "bookey.app",
    # Audio/podcast catalog pages — give us catalog cards, not transcripts
    "listennotes.com",
    "iheart.com",
    # Event listings, weeklies, light-news pages
    "watershed.co.uk",
    "noozhawk.com",
    "raggeduniversity.co.uk",
    "msn.com",
    # Social
    "facebook.com",
    "instagram.com",
    "tiktok.com",
    "pinterest.com",
    "x.com",
    "twitter.com",
    "reddit.com",
    "quora.com",
    "linkedin.com",
    # Search/aggregator landing pages
    "scholar.google.com",
    "semanticscholar.org",
    # Email-tracking / safelink wrappers (these don't return real content)
    "safelinks.protection.outlook.com",
)


def is_noise_domain(url: str) -> bool:
    """True for domains we want to skip during scrape — UI chrome, social, paste sites."""
    if not url:
        return False
    try:
        parsed = urlparse(url if "://" in url else f"https://{url}")
        domain = (parsed.netloc or "").lower()
    except Exception:
        return False
    if not domain:
        return False
    for noise in _NOISE_DOMAIN_PATTERNS:
        if domain == noise or domain.endswith("." + noise):
            return True
    return False
