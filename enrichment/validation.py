"""
Shared name-matching and affiliation-verification utilities.

Every collector MUST use these to validate that scraped data belongs
to the correct professor before accepting results.
"""

import re
import unicodedata
from typing import Any, Dict, List, Optional, Union


# ── Name normalization ─────────────────────────────────────────────────

def normalize(text: str) -> str:
    """Lowercase, strip accents, collapse whitespace, remove punctuation."""
    if not text:
        return ""
    # Strip accents: é→e, ü→u
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    # Lowercase, strip dots/commas, collapse whitespace
    text = text.lower()
    text = re.sub(r"[.\-,;:'\"()]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _split_name(name: str) -> tuple:
    """Return (first, last) from a full name string.
    Handles 'Last, First' and 'First Last' formats.
    """
    if not name:
        return ("", "")
    # Check for comma BEFORE normalizing (normalize strips commas)
    if "," in name:
        parts = [p.strip() for p in name.split(",", 1)]
        last = normalize(parts[0]).split()[-1] if parts[0] else ""
        first = normalize(parts[1]).split()[0] if len(parts) > 1 and parts[1] else ""
        return (first, last)
    name = normalize(name)
    parts = name.split()
    if len(parts) >= 2:
        return (parts[0], parts[-1])
    return (parts[0] if parts else "", "")


# ── Name matching ──────────────────────────────────────────────────────

def names_match(
    query_first: str,
    query_last: str,
    found_name: str,
    require_first_name: bool = True,
) -> bool:
    """
    Check whether `found_name` plausibly refers to the same person
    as (query_first, query_last).

    Rules:
    - Last name must match exactly (normalized).
    - If require_first_name=True, the first name must match by:
        (a) exact first name, OR
        (b) first initial match (e.g. 'J' matches 'John'), OR
        (c) found first name starts with query first name or vice-versa
    """
    q_first = normalize(query_first)
    q_last = normalize(query_last)
    found_first, found_last = _split_name(found_name)

    if not q_last or not found_last:
        return False

    # Last name must match exactly
    if q_last != found_last:
        return False

    if not require_first_name:
        return True

    if not q_first or not found_first:
        return False

    # Exact first name
    if q_first == found_first:
        return True

    # Initial match: "J" matches "John" or "John" matches "J"
    if len(q_first) == 1 and found_first.startswith(q_first):
        return True
    if len(found_first) == 1 and q_first.startswith(found_first):
        return True

    # Prefix match: "Jon" matches "Jonathan", "Chris" matches "Christopher"
    if q_first.startswith(found_first) or found_first.startswith(q_first):
        return True

    return False


def names_match_from_query(query, found_name: str, require_first_name: bool = True) -> bool:
    """Convenience: pass a ProfessorQuery directly."""
    return names_match(query.first_name, query.last_name, found_name, require_first_name)


# ── Affiliation verification ──────────────────────────────────────────

OSU_PATTERNS = [
    "ohio state",
    "osu",
    "ohio-state",
]

def has_osu_affiliation(text_or_list: Union[str, List, Dict, None]) -> bool:
    """
    Check whether the given text, list, or dict mentions Ohio State University.
    Handles nested structures by converting to JSON string.
    """
    if text_or_list is None:
        return False

    if isinstance(text_or_list, str):
        text = text_or_list.lower()
    elif isinstance(text_or_list, (list, dict)):
        import json
        text = json.dumps(text_or_list, default=str).lower()
    else:
        text = str(text_or_list).lower()

    # Check for "ohio state" — most reliable
    if "ohio state" in text:
        return True

    # "osu" alone is ambiguous (Oklahoma State, Oregon State)
    # Only accept if combined with other signals
    # Don't match on "osu" alone
    return False


def verify_affiliation_in_list(affiliations: List[str]) -> bool:
    """Check if any affiliation string mentions Ohio State."""
    for aff in affiliations:
        if has_osu_affiliation(aff):
            return True
    return False


# ── Combined validation ───────────────────────────────────────────────

def validate_match(
    query,
    found_name: str,
    found_affiliations: Optional[List[str]] = None,
    require_affiliation: bool = False,
    require_first_name: bool = True,
) -> bool:
    """
    Master validation: Is this the right person?

    Args:
        query: ProfessorQuery with .first_name, .last_name
        found_name: Name returned by the data source
        found_affiliations: List of affiliation strings (if available)
        require_affiliation: If True, must confirm OSU affiliation
        require_first_name: If True, first name must also match

    Returns:
        True if this is likely the correct professor.
    """
    # Name check
    if not names_match_from_query(query, found_name, require_first_name=require_first_name):
        return False

    # Affiliation check
    if require_affiliation and found_affiliations:
        if not verify_affiliation_in_list(found_affiliations):
            return False

    return True


def filter_articles_by_name(
    articles: List[Dict],
    query,
    text_keys: List[str] = None,
    require_first_name: bool = True,
) -> List[Dict]:
    """
    Filter news/article results to only those mentioning the professor's
    name in the specified text fields.

    Args:
        require_first_name: If True (default), requires full name match.
            If False, accepts last-name-only matches as a fallback.
    """
    if text_keys is None:
        text_keys = ["title", "description", "excerpt", "summary"]

    q_first = normalize(query.first_name)
    q_last = normalize(query.last_name)
    full_name = normalize(query.name)

    filtered = []
    for article in articles:
        combined = " ".join(
            normalize(str(article.get(k, ""))) for k in text_keys
        )
        # Check for full name or (first + last separately within text)
        if full_name in combined:
            filtered.append(article)
        elif q_first in combined and q_last in combined:
            filtered.append(article)
        elif not require_first_name and q_last in combined:
            filtered.append(article)

    return filtered


def strict_identity_match(
    query,
    text: str,
    *,
    require_full_name: bool = True,
    require_affiliation: bool = True,
    department_hint: str = "",
    min_name_density: int = 1,
) -> bool:
    """
    Hard gate: does this blob of text actually belong to *this* professor?

    This is the canonical check that every scraper/search result must pass
    before we accept it into enrichment. It catches the common failure modes:
      - Wikipedia article about a same-named politician
      - Web search result about a city/town sharing the name
      - GitHub user with the same last name but unrelated
      - ORCID record for someone at a different university

    Rules (all must hold):
      1. Full name (first + last) must appear in the text. Common-sense
         variants are accepted: "Jeff Volek", "Jeffrey Volek", "J. Volek",
         "Volek, Jeff". Pure last-name-only matches are REJECTED — that's
         how false positives get through.
      2. Text must show ≥ min_name_density last-name occurrences (dense
         biographical pages repeat the subject's name).
      3. Affiliation signal: one of
         - "ohio state" (case-insensitive) in text
         - ".osu.edu" domain reference
         - Known department name (if provided) appears in text
         Skipped only when require_affiliation=False.

    Args:
        query: ProfessorQuery with .first_name, .last_name, .name
        text: content to validate (HTML-stripped or raw)
        require_full_name: if False, first-initial+last match is accepted
        require_affiliation: if True, must see OSU/dept signal in text
        department_hint: extra department string to accept as affiliation signal
        min_name_density: minimum last-name occurrences in text

    Returns True only when ALL checks pass.
    """
    if not text or not query:
        return False

    q_first = normalize(query.first_name)
    q_last = normalize(query.last_name)
    if not q_last:
        return False

    norm_text = normalize(text)

    # Rule 2: minimum last-name density (kill near-empty pages)
    last_count = norm_text.count(" " + q_last + " ") + norm_text.count(q_last + " ") + (
        1 if norm_text.startswith(q_last + " ") else 0
    )
    if last_count < min_name_density:
        return False

    # Rule 1: full-name anchoring — require first+last together at least once,
    # OR a recognized variant (initial + last, "last, first", middle-initial).
    full_variants_present = (
        f"{q_first} {q_last}" in norm_text
        or f"{q_last} {q_first}" in norm_text  # "Volek, Jeff" after normalize = "volek jeff"
    )
    if not full_variants_present and require_full_name and q_first:
        # Allow first-initial variants: "j volek"
        initial = q_first[0]
        if f"{initial} {q_last}" not in norm_text and f"{q_last} {initial}" not in norm_text:
            # Allow middle-initial variant: "jeff s volek", "jeffrey s volek"
            middle_pattern = re.compile(rf"{re.escape(q_first)} [a-z] {re.escape(q_last)}")
            if not middle_pattern.search(norm_text):
                return False

    # Rule 3: affiliation signal (OSU or department)
    if require_affiliation:
        aff_ok = (
            "ohio state" in norm_text
            or "osu edu" in norm_text  # normalized form of .osu.edu
        )
        if not aff_ok and department_hint:
            dept_n = normalize(department_hint)
            if dept_n and len(dept_n) >= 3 and dept_n in norm_text:
                aff_ok = True
        if not aff_ok:
            return False

    return True


def names_match_fuzzy(
    query_first: str,
    query_last: str,
    found_name: str,
) -> bool:
    """
    Softer name matching that handles common format variations:
    - Hyphenated names: "Smith-Jones" matches "Smith Jones"
    - Suffixes: "John Smith Jr." matches "John Smith"
    - All-caps: "JOHN SMITH" matches "John Smith"
    - No-space comma: "Smith,John" matches "John Smith"
    """
    # Normalize both sides (handles accents, case, punctuation)
    q_first = normalize(query_first)
    q_last = normalize(query_last)

    # Clean up found_name: handle "Last,First" (no space) and "LAST, FIRST"
    cleaned = found_name.replace(",", ", ")  # ensure space after comma
    found_first, found_last = _split_name(cleaned)

    if not q_last or not found_last:
        return False

    # Last name: exact or hyphenation-collapsed match
    q_last_collapsed = q_last.replace(" ", "")
    found_last_collapsed = found_last.replace(" ", "")
    if q_last_collapsed != found_last_collapsed:
        return False

    if not q_first or not found_first:
        return True  # last name matched, no first name to check

    # Strip common suffixes for comparison
    suffixes = {"jr", "sr", "ii", "iii", "iv", "phd", "md", "do"}
    q_first_parts = [p for p in q_first.split() if p not in suffixes]
    found_first_parts = [p for p in found_first.split() if p not in suffixes]
    q_f = q_first_parts[0] if q_first_parts else q_first
    f_f = found_first_parts[0] if found_first_parts else found_first

    # Exact, initial, or prefix match (same as names_match)
    if q_f == f_f:
        return True
    if len(q_f) == 1 and f_f.startswith(q_f):
        return True
    if len(f_f) == 1 and q_f.startswith(f_f):
        return True
    if q_f.startswith(f_f) or f_f.startswith(q_f):
        return True

    return False
