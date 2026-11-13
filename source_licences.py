"""What is known about the terms of a legend source page, by its host.

A legend's passages were collected from pages found by searching its name,
from any site. Whether a page's text may be used to write from, and how it
must be credited, depends on the site. This table holds only what has been
checked; every other host is "unknown" (its terms were not checked), which is
not the same as refused. Refused hosts are the publishers the story worker
already keeps out (daily_story_worker.EXCLUDED_PUBLISHER_DOMAINS), whose
terms forbid this use.

The table is pilot plan B4's starting point (an allowlist with the licence
per domain), not a legal opinion: counsel signs off on it. Change the table
to change what every story's source record and the selection report say.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse

LICENCE_POLICY_VERSION = "source-licences-2026-11-12"

OPEN = "open"          # an open licence, or the site's own text is in the public domain
UNKNOWN = "unknown"    # the site's terms have not been checked
REFUSED = "refused"    # the site's terms forbid this use

# US federal agencies' own works have no copyright (17 U.S.C. 105). What an
# agency's site hosts from others (images, contributed text) can differ.
_US_FEDERAL = {
    "id": "US-PD-federal",
    "name": "US federal government work (17 U.S.C. 105)",
    "status": OPEN,
    "attribution": "Credit the agency and link the page; material the agency hosts from others keeps its own terms.",
}

# (host suffix, licence). The first suffix that matches wins, so list more
# specific hosts first.
KNOWN_HOSTS: List[Tuple[str, Dict[str, str]]] = [
    ("wikipedia.org", {
        "id": "CC-BY-SA-4.0",
        "name": "Creative Commons Attribution-ShareAlike 4.0 (Wikipedia text)",
        "status": OPEN,
        "attribution": "Name the article and Wikipedia, link it and the licence; an adaptation is shared under the same licence.",
    }),
    ("wikidata.org", {
        "id": "CC0-1.0",
        "name": "CC0 1.0 (Wikidata data)",
        "status": OPEN,
        "attribution": "None required; a link is kept as good practice.",
    }),
    ("globalvoices.org", {
        "id": "CC-BY-3.0",
        "name": "Creative Commons Attribution 3.0 (Global Voices)",
        "status": OPEN,
        "attribution": "Name Global Voices and the author, link the page and the licence.",
    }),
    *[(host, _US_FEDERAL) for host in (
        "loc.gov",
        "archives.gov",
        "nasa.gov",
        "nih.gov",
        "nsf.gov",
        "noaa.gov",
        "usgs.gov",
        "nps.gov",
        "cdc.gov",
        "ed.gov",
        "energy.gov",
        "history.navy.mil",
        "army.mil",
    )],
]

_UNKNOWN = {
    "id": "unknown",
    "name": "Terms not checked",
    "status": UNKNOWN,
    "attribution": "Link the page; check the site's terms before relying on it.",
}


def host_of(url: Any) -> str:
    text = str(url or "").strip()
    if not text:
        return ""
    try:
        host = (urlparse(text if "://" in text else f"https://{text}").hostname or "").lower()
    except ValueError:
        return ""
    return host[4:] if host.startswith("www.") else host


def _matches(host: str, suffix: str) -> bool:
    return host == suffix or host.endswith("." + suffix)


def licence_for(url: Any) -> Dict[str, str]:
    """The licence record for a page: its id, name, status (open, unknown,
    refused), what crediting it takes, the host and this table's version."""
    host = host_of(url)
    # Imported here: the worker imports this module.
    from daily_story_worker import EXCLUDED_PUBLISHER_DOMAINS

    if host and any(_matches(host, domain) for domain in EXCLUDED_PUBLISHER_DOMAINS):
        found = {
            "id": "refused",
            "name": "Publisher terms forbid this use",
            "status": REFUSED,
            "attribution": "Not used.",
        }
    else:
        found = next((licence for suffix, licence in KNOWN_HOSTS if host and _matches(host, suffix)), _UNKNOWN)
    return {**found, "host": host, "policy_version": LICENCE_POLICY_VERSION}
