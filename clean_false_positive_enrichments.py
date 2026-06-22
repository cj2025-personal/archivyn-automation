"""
One-off cleanup: scan every profile's enrichment.json, find sources whose
data fails the strict-identity gate, strip their data, and save the cleaned
file.

Targets the three false-positive-prone sources: wikipedia, web_search, github.
Other sources have their own stricter gates (DOI / ORCID / structured IDs)
and are left alone.

Usage:
    python clean_false_positive_enrichments.py                # dry-run
    python clean_false_positive_enrichments.py --apply        # actually strip

A cleaned profile is marked in-place: enrichment.json has the affected
source's `data` cleared and `error` set to "Stripped by cleanup: ...".

This is safe to run multiple times — it's idempotent on already-cleaned files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from enrichment.base_collector import ProfessorQuery
from enrichment.validation import strict_identity_match

ROOT = Path("output/osu_faculty_run/profiles")

# Sources we validate. Each has a `serialise` that produces the searchable
# text blob to feed strict_identity_match, plus a require_affiliation flag.
VALIDATORS = {
    "wikipedia": {
        "extract_text": lambda d: (d.get("extract") or ""),
        "require_affiliation": True,
        "min_name_density": 2,
    },
    "web_search": {
        # Concatenate all scraped page texts
        "extract_text": lambda d: " ".join(
            (p.get("text") or "")[:5000] for p in (d.get("pages") or [])
        ),
        # web_search can legitimately include academic-domain pages that don't
        # repeat OSU; so we relax affiliation gate at profile-level here — the
        # per-page gate inside the collector is the stricter guard going forward.
        "require_affiliation": False,
        "min_name_density": 2,
    },
    "github": {
        "extract_text": lambda d: " ".join([
            str(d.get("name") or ""),
            str(d.get("bio") or ""),
            str(d.get("company") or ""),
            str(d.get("location") or ""),
            str(d.get("blog") or ""),
        ] + [
            f"{r.get('description') or ''} {r.get('readme_excerpt') or ''}"
            for r in (d.get("top_repos") or [])
        ]),
        "require_affiliation": True,
        "min_name_density": 1,
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true",
                        help="Actually strip bad data (default: dry-run only)")
    parser.add_argument("--profiles-dir", default=str(ROOT))
    args = parser.parse_args()

    base = Path(args.profiles_dir)
    if not base.exists():
        print(f"Profiles dir not found: {base}")
        sys.exit(1)

    total_profiles = 0
    total_stripped = 0
    per_source_stripped = {s: 0 for s in VALIDATORS}
    samples_per_source = {s: [] for s in VALIDATORS}

    for pdir in sorted(base.iterdir()):
        if not pdir.is_dir():
            continue
        enr_path = pdir / "enrichment.json"
        if not enr_path.exists():
            continue
        try:
            doc = json.loads(enr_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        total_profiles += 1

        prof_name = doc.get("professor_name") or ""
        parts = prof_name.strip().split()
        if len(parts) < 2:
            continue
        # Synthesize a ProfessorQuery for the validator
        query = ProfessorQuery(
            profile_id=doc.get("profile_id", pdir.name),
            name=prof_name,
            university=doc.get("university", "Ohio State University"),
            department=doc.get("department", "") or "",
        )

        changed = False
        for src_name, cfg in VALIDATORS.items():
            src = doc.get("sources", {}).get(src_name)
            if not src or not src.get("success") or not src.get("data"):
                continue
            text = cfg["extract_text"](src["data"]) or ""
            if not text.strip():
                continue
            ok = strict_identity_match(
                query, text,
                require_full_name=True,
                require_affiliation=cfg["require_affiliation"],
                department_hint=query.department,
                min_name_density=cfg["min_name_density"],
            )
            if not ok:
                per_source_stripped[src_name] += 1
                if len(samples_per_source[src_name]) < 5:
                    preview = text[:160].replace("\n", " ")
                    samples_per_source[src_name].append(
                        f"{prof_name} → {preview}"
                    )
                if args.apply:
                    src["data"] = {}
                    src["success"] = False
                    src["error"] = (
                        "Stripped by cleanup: failed strict-identity gate "
                        "(wrong person / unrelated content)"
                    )
                    changed = True

        if changed:
            total_stripped += 1
            # Recompute summary counts
            srcs = doc.get("sources", {})
            ok_names = sorted(n for n, r in srcs.items() if r.get("success"))
            fail_names = sorted(n for n, r in srcs.items() if not r.get("success"))
            doc["summary"] = {
                "total_sources_queried": len(srcs),
                "successful_sources": len(ok_names),
                "failed_sources": len(fail_names),
                "successful_source_names": ok_names,
                "failed_source_names": fail_names,
            }
            enr_path.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")

    print(f"\n{'APPLIED' if args.apply else 'DRY RUN'} — scanned {total_profiles} profiles")
    print(f"Profiles with ≥1 stripped source: {total_stripped}")
    print()
    for src, cnt in per_source_stripped.items():
        pct = cnt / max(total_profiles, 1) * 100
        print(f"  {src:<15} {cnt:>5} failures ({pct:.1f}% of profiles)")
    print()
    for src, samples in samples_per_source.items():
        if not samples:
            continue
        print(f"\n--- Sample false positives for {src} ---")
        for s in samples:
            print(f"  • {s}")


if __name__ == "__main__":
    main()
