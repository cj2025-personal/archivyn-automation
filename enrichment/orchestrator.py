"""
Enrichment Orchestrator — runs all collectors for a professor and merges results.

Supports 13 public data sources, parallel execution, caching,
cross-source confidence scoring, and deduplication.

Usage:
    orchestrator = EnrichmentOrchestrator(output_dir="output/osu_faculty_run")
    results = await orchestrator.enrich_professor(query)
"""

import asyncio
import json
import logging
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .base_collector import BaseCollector, CollectorResult, ProfessorQuery
from .semantic_scholar import SemanticScholarCollector
from .openalex import OpenAlexCollector
from .google_scholar import GoogleScholarCollector
from .nsf_grants import NSFGrantsCollector
from .nih_grants import NIHGrantsCollector
from .rate_my_professor import RateMyProfessorCollector
from .osu_courses import OSUCoursesCollector
from .youtube_lectures import YouTubeLecturesCollector
from .osu_news import OSUNewsCollector
from .orcid import ORCIDCollector
from .crossref import CrossRefCollector
from .google_news import GoogleNewsCollector
from .osu_expertise import OSUExpertiseCollector
from .web_search_collector import WebSearchCollector

# -- New differentiator collectors --------------------------------------
from .unpaywall import UnpaywallCollector
from .arxiv import ArxivCollector
from .wikidata import WikidataCollector
from .wikipedia import WikipediaCollector
from .github import GitHubCollector
from .youtube_transcripts import YouTubeTranscriptsCollector
from .usaspending import USASpendingCollector
from .patentsview import PatentsViewCollector
from .clinicaltrials import ClinicalTrialsCollector
from .altmetric import AltmetricCollector
from .opencitations import OpenCitationsCollector
from .paperswithcode import PapersWithCodeCollector
from .huggingface import HuggingFaceCollector
from .biorxiv import BiorxivCollector
from .pmc_oa import PMCOpenAccessCollector
from .zenodo import ZenodoCollector
from .figshare import FigshareCollector
from .osf import OSFCollector
from .gdelt import GDELTCollector
from .core_api import CoreAPICollector

logger = logging.getLogger(__name__)

# ── All collectors (13 original + 20 new) ─────────────────────────────────
ALL_COLLECTORS = {
    # -- Web search (highest value — lets search engine disambiguate) --
    "web_search": WebSearchCollector,
    # -- Publication & citation data --
    "semantic_scholar": SemanticScholarCollector,
    "openalex": OpenAlexCollector,
    "google_scholar": GoogleScholarCollector,
    "crossref": CrossRefCollector,
    "orcid": ORCIDCollector,
    "opencitations": OpenCitationsCollector,          # NEW
    # -- Grants & funding --
    "nsf_grants": NSFGrantsCollector,
    "nih_grants": NIHGrantsCollector,
    "usaspending": USASpendingCollector,              # NEW
    # -- Patents & clinical --
    "patentsview": PatentsViewCollector,              # NEW
    "clinicaltrials": ClinicalTrialsCollector,        # NEW
    # -- Open Access full-text & preprints --
    "unpaywall": UnpaywallCollector,                  # NEW
    "arxiv": ArxivCollector,                          # NEW
    "biorxiv": BiorxivCollector,                      # NEW
    "pmc_oa": PMCOpenAccessCollector,                 # NEW
    "core_api": CoreAPICollector,                     # NEW
    # -- Code / ML / research artifacts --
    "github": GitHubCollector,                        # NEW
    "huggingface": HuggingFaceCollector,              # NEW
    "paperswithcode": PapersWithCodeCollector,        # NEW
    "zenodo": ZenodoCollector,                        # NEW
    "figshare": FigshareCollector,                    # NEW
    "osf": OSFCollector,                              # NEW
    # -- Impact / identity / lineage --
    "altmetric": AltmetricCollector,                  # NEW
    "wikidata": WikidataCollector,                    # NEW
    "wikipedia": WikipediaCollector,                  # NEW
    # -- Teaching & student experience --
    "rate_my_professor": RateMyProfessorCollector,
    "osu_courses": OSUCoursesCollector,
    # -- News & media --
    "osu_news": OSUNewsCollector,
    "google_news": GoogleNewsCollector,
    "gdelt": GDELTCollector,                          # NEW
    "youtube_lectures": YouTubeLecturesCollector,
    "youtube_transcripts": YouTubeTranscriptsCollector,  # NEW
    # -- OSU-specific --
    "osu_expertise": OSUExpertiseCollector,
}

# Sources that need specific API keys (rest are free)
# NOTE: New collectors gracefully degrade when their optional keys are missing;
# we only hard-disable here when the collector can't function at all without a key.
API_KEY_REQUIREMENTS = {
    "youtube_lectures": "YOUTUBE_API_KEY",
    "core_api": "CORE_API_KEY",
}

# Collectors that depend on other collectors' output (run after base sweep).
# Unpaywall/Altmetric/OpenCitations/YouTube-transcripts need DOIs or video IDs
# produced by earlier collectors.
DEPENDENT_SOURCES = {
    "unpaywall": ["openalex", "crossref"],
    "altmetric": ["openalex", "crossref"],
    "opencitations": ["openalex", "crossref"],
    "youtube_transcripts": ["youtube_lectures"],
}

# Priority order for text assembly (higher priority = listed first in output)
SOURCE_PRIORITY = [
    "web_search",           # Highest value — search engine finds actual pages about this person
    "wikipedia",            # Dense biographical summary if available
    "wikidata",             # Structured facts (awards, positions, lineage)
    "orcid",                # Authoritative career data
    "osu_expertise",        # OSU-specific profiles
    "youtube_transcripts",  # Differentiator: full lecture/talk transcripts
    "pmc_oa",               # Full-text biomedical papers
    "arxiv",                # Preprint full-text metadata
    "biorxiv",              # Life-sci preprints
    "semantic_scholar",
    "openalex",
    "crossref",
    "google_scholar",
    "core_api",             # OA aggregator
    "unpaywall",            # Legal OA PDF links for existing DOIs
    "opencitations",        # Supplementary citation graph
    "altmetric",            # Impact / policy / news per paper
    "nsf_grants",
    "nih_grants",
    "usaspending",          # Federal grants beyond NIH/NSF
    "clinicaltrials",       # Trials PI
    "patentsview",          # USPTO patents
    "paperswithcode",       # Code linked to papers
    "github",               # Repos, READMEs
    "huggingface",          # Models / datasets
    "zenodo",               # Datasets / software
    "figshare",             # Datasets / figures / posters
    "osf",                  # Projects / pre-registrations
    "osu_courses",
    "rate_my_professor",
    "osu_news",
    "google_news",
    "gdelt",                # Global news index
    "youtube_lectures",
]


class EnrichmentOrchestrator:
    """
    Runs multiple data collectors in parallel for each professor,
    saves results, cross-references data, and generates merged enrichment text.
    """

    def __init__(
        self,
        output_dir: str = "output/osu_faculty_run",
        cache_dir: Optional[str] = None,
        enabled_sources: Optional[List[str]] = None,
        disabled_sources: Optional[List[str]] = None,
        max_concurrent: int = 4,
    ):
        self.output_dir = Path(output_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else self.output_dir / "enrichment_cache"
        self.max_concurrent = max_concurrent

        # Determine which collectors to enable
        if enabled_sources:
            self.enabled_sources = set(enabled_sources) & set(ALL_COLLECTORS.keys())
        else:
            self.enabled_sources = set(ALL_COLLECTORS.keys())

        if disabled_sources:
            self.enabled_sources -= set(disabled_sources)

        # Remove sources missing required API keys
        for source, key_name in API_KEY_REQUIREMENTS.items():
            if source in self.enabled_sources and not os.getenv(key_name):
                logger.info("Disabling %s (missing %s)", source, key_name)
                self.enabled_sources.discard(source)

        # Instantiate collectors
        self.collectors: Dict[str, BaseCollector] = {}
        for name in self.enabled_sources:
            cls = ALL_COLLECTORS[name]
            self.collectors[name] = cls(cache_dir=self.cache_dir / name)

        print(f"[Enrichment] ✅ Orchestrator initialized with {len(self.collectors)}/{len(ALL_COLLECTORS)} sources")
        print(f"[Enrichment] Sources: {', '.join(sorted(self.collectors.keys()))}")
        print(f"[Enrichment] Cache: {self.cache_dir}")
        print(f"[Enrichment] Max concurrent: {self.max_concurrent}")

    async def enrich_professor(self, query: ProfessorQuery) -> Dict[str, CollectorResult]:
        """
        Run all enabled collectors for a single professor.

        Executes in two waves so that dependent collectors (unpaywall / altmetric
        / opencitations / youtube_transcripts) can read DOIs / video IDs from
        prior collectors via the enrichment.json file written after wave 1.

        Returns dict of source_name -> CollectorResult.
        """
        print(f"  [Enrich] Querying {len(self.collectors)} sources for: {query.name}")
        if query.department:
            print(f"  [Enrich] Department: {query.department}")

        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def _run(name: str, collector: BaseCollector) -> tuple:
            async with semaphore:
                result = await collector.safe_collect(query)
                return name, result

        # Partition collectors into wave 1 (independent) and wave 2 (dependent).
        dependents = set(DEPENDENT_SOURCES.keys()) & set(self.collectors.keys())
        wave1 = {n: c for n, c in self.collectors.items() if n not in dependents}
        wave2 = {n: c for n, c in self.collectors.items() if n in dependents}

        results: Dict[str, CollectorResult] = {}

        # Wave 1
        if wave1:
            print(f"  [Enrich] Wave 1 — {len(wave1)} independent sources")
            tasks = [_run(n, c) for n, c in wave1.items()]
            for item in await asyncio.gather(*tasks, return_exceptions=True):
                if isinstance(item, Exception):
                    print(f"    [???] ❌ Collector exception: {item}")
                    continue
                name, result = item
                results[name] = result

            # Persist partial enrichment.json so wave 2 collectors can read it
            self.save_enrichment(query, results)

        # Wave 2
        if wave2:
            print(f"  [Enrich] Wave 2 — {len(wave2)} dependent sources (DOIs / video IDs)")
            tasks = [_run(n, c) for n, c in wave2.items()]
            for item in await asyncio.gather(*tasks, return_exceptions=True):
                if isinstance(item, Exception):
                    print(f"    [???] ❌ Collector exception: {item}")
                    continue
                name, result = item
                results[name] = result

        # Summary
        ok = sum(1 for r in results.values() if r.success)
        fail = sum(1 for r in results.values() if not r.success)
        cached = sum(1 for r in results.values() if r.cached)
        total_text = sum(len(r.raw_text) for r in results.values() if r.success and r.raw_text)
        print(f"  [Enrich] Results: {ok} ✅ / {fail} ❌ / {cached} cached — {total_text:,} chars of enrichment text")

        return results

    def save_enrichment(
        self,
        query: ProfessorQuery,
        results: Dict[str, CollectorResult],
    ) -> Path:
        """Save enrichment results to a JSON file alongside the profile."""
        profile_dir = self.output_dir / "profiles" / query.profile_id
        profile_dir.mkdir(parents=True, exist_ok=True)

        enrichment_path = profile_dir / "enrichment.json"

        # Cross-reference and score
        confidence = self._compute_confidence(query, results)

        # Build enrichment document
        doc = {
            "profile_id": query.profile_id,
            "professor_name": query.name,
            "university": query.university,
            "department": query.department,
            "enriched_at": datetime.now(timezone.utc).isoformat(),
            "confidence": confidence,
            "sources": {},
            "summary": {
                "total_sources_queried": len(results),
                "successful_sources": sum(1 for r in results.values() if r.success),
                "failed_sources": sum(1 for r in results.values() if not r.success),
                "successful_source_names": sorted(
                    name for name, r in results.items() if r.success
                ),
                "failed_source_names": sorted(
                    name for name, r in results.items() if not r.success
                ),
            },
        }

        for source_name, result in results.items():
            doc["sources"][source_name] = {
                "success": result.success,
                "cached": result.cached,
                "error": result.error,
                "data": result.data if result.success else {},
                "timestamp": result.timestamp,
            }

        enrichment_path.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
        conf_score = confidence.get("overall_confidence", 0)
        print(f"  [Save] ✅ enrichment.json ({enrichment_path.stat().st_size:,} bytes, confidence={conf_score:.2f})")
        return enrichment_path

    def build_enrichment_text(
        self,
        query: ProfessorQuery,
        results: Dict[str, CollectorResult],
    ) -> str:
        """
        Merge all successful collector raw_text outputs into a single
        enrichment text block, ordered by source priority.
        """
        sections = []
        sections.append(f"{'='*80}")
        sections.append(f"ENRICHMENT DATA FOR: {query.name}")
        sections.append(f"University: {query.university}")
        if query.department:
            sections.append(f"Department: {query.department}")
        sections.append(f"Enriched at: {datetime.now(timezone.utc).isoformat()}")
        sections.append(f"Successful sources: {sum(1 for r in results.values() if r.success)}/{len(results)}")
        sections.append(f"{'='*80}")

        # Order by priority
        ordered_sources = sorted(
            results.keys(),
            key=lambda s: SOURCE_PRIORITY.index(s) if s in SOURCE_PRIORITY else 999,
        )

        for source_name in ordered_sources:
            result = results[source_name]
            if result.success and result.raw_text:
                sections.append(f"\n{'─'*60}")
                sections.append(f"Source: {source_name}")
                sections.append(f"{'─'*60}")
                sections.append(result.raw_text)

        return "\n".join(sections)

    def save_enrichment_text(
        self,
        query: ProfessorQuery,
        results: Dict[str, CollectorResult],
    ) -> Optional[Path]:
        """Save merged enrichment text to a file for chunking."""
        text = self.build_enrichment_text(query, results)
        if not text.strip():
            print(f"  [Save] ⚠️ No enrichment text to save (all sources empty)")
            return None

        profile_dir = self.output_dir / "profiles" / query.profile_id
        profile_dir.mkdir(parents=True, exist_ok=True)

        text_path = profile_dir / "enrichment_text.txt"
        text_path.write_text(text, encoding="utf-8")
        print(f"  [Save] ✅ enrichment_text.txt ({len(text):,} chars)")
        return text_path

    # ── Cross-source confidence scoring ──────────────────────────────────

    # Sources that apply to virtually all professors
    UNIVERSAL_SOURCES = {
        "web_search", "openalex", "osu_courses", "osu_news",
        "osu_expertise", "crossref", "orcid", "rate_my_professor",
        "wikidata", "gdelt",
    }
    # Domain-specific sources (only applicable if the professor is in STEM / biomed / CS)
    STEM_GRANT_SOURCES = {"nsf_grants", "nih_grants", "usaspending"}
    CLINICAL_SOURCES = {"clinicaltrials", "pmc_oa", "biorxiv"}
    CODE_SOURCES = {"github", "huggingface", "paperswithcode"}
    PATENT_SOURCES = {"patentsview"}
    # Data/artifact sources
    DATA_SOURCES = {"zenodo", "figshare", "osf"}
    # Media sources (hit-or-miss for any professor)
    MEDIA_SOURCES = {"google_news", "youtube_lectures", "youtube_transcripts"}
    # Dependent sources — scored only if the parent produced data
    DEPENDENT_SOURCES_SET = {"unpaywall", "altmetric", "opencitations", "youtube_transcripts"}

    def _compute_confidence(
        self,
        query: ProfessorQuery,
        results: Dict[str, CollectorResult],
    ) -> Dict[str, Any]:
        """
        Cross-reference data across sources to compute confidence scores.
        This helps identify cases where the wrong person was matched.

        Coverage is computed relative to applicable sources, not all 13.
        """
        signals = {
            "name_match_sources": 0,
            "osu_affiliation_confirmed": 0,
            "h_index_values": [],
            "citation_counts": [],
            "publication_counts": [],
            "overall_confidence": 0.0,
        }

        total_raw_text_len = 0

        for source_name, result in results.items():
            if not result.success:
                continue

            data = result.data
            total_raw_text_len += len(result.raw_text or "")

            # Check name consistency
            found_name = ""
            for key in ["name", "display_name", "author_name"]:
                if key in data:
                    found_name = data[key]
                    break
            if found_name and query.last_name.lower() in found_name.lower():
                signals["name_match_sources"] += 1

            # Check OSU affiliation
            for key in ["affiliations", "institutions", "affiliation", "organization"]:
                val = data.get(key)
                if val:
                    val_str = json.dumps(val).lower()
                    if "ohio state" in val_str:
                        signals["osu_affiliation_confirmed"] += 1
                        break

            # Collect h-index
            h = data.get("h_index") or data.get("hIndex")
            if h and isinstance(h, (int, float)):
                signals["h_index_values"].append(h)

            # Collect citation counts
            for key in ["citation_count", "cited_by_count", "total_citations"]:
                val = data.get(key)
                if val and isinstance(val, (int, float)) and val > 0:
                    signals["citation_counts"].append(val)
                    break

            # Collect publication counts
            for key in ["paper_count", "works_count", "publications_count", "total_works"]:
                val = data.get(key)
                if val and isinstance(val, (int, float)) and val > 0:
                    signals["publication_counts"].append(val)
                    break

        # Determine applicable sources for this professor
        queried_sources = set(results.keys())
        successful = sum(1 for r in results.values() if r.success)

        # A source is "applicable" if it was queried AND is either universal,
        # or succeeded (which proves the professor has data there),
        # or is a grant source for someone with publications (likely STEM).
        has_publications = len(signals["publication_counts"]) > 0
        applicable = set()
        for s in queried_sources:
            if s in self.UNIVERSAL_SOURCES:
                applicable.add(s)
            elif s in self.STEM_GRANT_SOURCES and has_publications:
                applicable.add(s)
            elif s in self.DEPENDENT_SOURCES_SET:
                # Dependent sources can't contribute to applicable unless parent succeeded
                parents = DEPENDENT_SOURCES.get(s) or []
                if any(results.get(p) and results[p].success for p in parents):
                    applicable.add(s)
            elif results[s].success:
                applicable.add(s)  # succeeded = clearly applicable
        applicable_count = max(len(applicable), 1)

        # Differentiator signal count (for richness_score)
        differentiator_hits = 0
        for s in ("youtube_transcripts", "patentsview", "clinicaltrials",
                  "github", "huggingface", "wikidata", "altmetric",
                  "pmc_oa", "arxiv", "biorxiv", "zenodo", "usaspending"):
            if results.get(s) and results[s].success:
                differentiator_hits += 1
        signals["differentiator_hits"] = differentiator_hits

        score = 0.0
        # Source coverage relative to applicable sources (up to 0.30)
        score += min(successful / applicable_count, 1.0) * 0.30
        # Name matches across successful sources (up to 0.20)
        score += min(signals["name_match_sources"] / max(successful, 1), 1.0) * 0.20
        # OSU affiliation confirmation (up to 0.15)
        score += min(signals["osu_affiliation_confirmed"] / max(successful, 1), 1.0) * 0.15
        # Data richness — text volume (up to 0.10)
        richness = min(total_raw_text_len / 20000, 1.0)  # 20K+ chars now for full score
        score += richness * 0.10
        # h-index consistency across sources (up to 0.10)
        if len(signals["h_index_values"]) >= 2:
            h_vals = signals["h_index_values"]
            h_range = max(h_vals) - min(h_vals)
            h_mean = sum(h_vals) / len(h_vals)
            if h_mean > 0:
                consistency = max(0, 1 - (h_range / h_mean))
                score += consistency * 0.10
        # Differentiator bonus — rewards signal absent from GPT/Claude training (up to 0.15)
        # 6+ differentiator hits = full bonus
        score += min(differentiator_hits / 6.0, 1.0) * 0.15

        signals["overall_confidence"] = round(score, 3)
        # Separate "richness" score summarising breadth of non-LLM-covered signal
        signals["richness_score"] = round(min(differentiator_hits / 10.0, 1.0), 3)
        return signals

    async def close(self):
        """Close all collector HTTP clients."""
        for collector in self.collectors.values():
            await collector.close()


# ── Non-professor role filter ─────────────────────────────────────────────
# Keywords indicating non-professor roles (PhD students, TAs, postdocs, etc.)
_EXCLUDE_KEYWORDS = [
    "phd student", "phd candidate", "ph.d. student", "ph.d. candidate",
    "doctoral student", "doctoral candidate",
    "graduate research associate", "graduate research assistant",
    "graduate teaching associate", "graduate teaching assistant",
    "graduate assistant", "graduate associate", "graduate fellow",
    "masters student", "master student",
    "visiting fellow", "visiting scholar",
    "postdoctoral", "post-doctoral", "postdoc ",
    "research assistant",
    "clinical research assistant",
]
# Keywords that override exclusion — keep anyone with professor/faculty rank
_KEEP_KEYWORDS = [
    "professor", "lecturer", "instructor", "faculty",
    "dean", "chair", "director", "endowed",
]


def _load_excel_expertise_lookup(profiles_dir: str) -> Dict[str, str]:
    """Build profile_id -> expertise and name -> expertise lookup from OSU.xlsx."""
    # Find OSU.xlsx relative to the profiles dir
    base = Path(profiles_dir).parent.parent  # output/osu_faculty_run -> project root
    excel_candidates = [
        base / "OSU.xlsx",
        base / "osu.xlsx",
        Path("OSU.xlsx"),
    ]
    excel_path = None
    for p in excel_candidates:
        if p.exists():
            excel_path = p
            break
    if not excel_path:
        logger.warning("OSU.xlsx not found, skipping professor-level filter")
        return {}

    try:
        import pandas as pd
        df = pd.read_excel(excel_path)
    except Exception as e:
        logger.warning("Failed to read OSU.xlsx: %s", e)
        return {}

    lookup = {}
    for _, row in df.iterrows():
        pid = str(row.get("Scholar Profile ID", "")).strip()
        name = str(row.get("Name", "")).strip().lower()
        exp = str(row.get("Expertise", "")).strip()
        if exp == "nan":
            exp = ""
        if pid and pid != "nan":
            lookup[f"pid:{pid}"] = exp
        if name:
            lookup[f"name:{name}"] = exp
    return lookup


def _is_professor_level(name: str, profile_id: str, expertise_lookup: Dict[str, str]) -> bool:
    """Return True if the profile is professor-level (not a grad student, TA, postdoc, etc.)."""
    if not expertise_lookup:
        return True  # No filter data — include everyone

    exp = expertise_lookup.get(f"pid:{profile_id}", "") or expertise_lookup.get(f"name:{name.lower()}", "")
    if not exp:
        return True  # No expertise info — include by default

    exp_lower = exp.lower()

    # If they have a professor/faculty title, always keep
    if any(kw in exp_lower for kw in _KEEP_KEYWORDS):
        return True

    # If they match an exclusion keyword, filter out
    for kw in _EXCLUDE_KEYWORDS:
        if kw in exp_lower:
            # "research assistant professor" should NOT be excluded
            if kw == "research assistant" and "research assistant professor" in exp_lower:
                continue
            return False

    return True


def load_professor_queries_from_profiles(
    profiles_dir: str,
    limit: Optional[int] = None,
    start_from: int = 0,
    filter_no_enrichment: bool = False,
    filter_professor_level: bool = True,
) -> List[ProfessorQuery]:
    """
    Load ProfessorQuery objects from existing profile JSON files.
    Reads profile_id, name, university, department from each profile.

    When filter_professor_level=True (default), filters out non-professor
    profiles (PhD students, TAs, postdocs, visiting fellows, etc.)
    """
    profiles_path = Path(profiles_dir)
    queries = []
    skipped_non_prof = 0

    # Load expertise lookup for professor-level filtering
    expertise_lookup = _load_excel_expertise_lookup(profiles_dir) if filter_professor_level else {}

    # Get all profile directories
    profile_dirs = sorted([d for d in profiles_path.iterdir() if d.is_dir()])

    for pdir in profile_dirs[start_from:]:
        if limit and len(queries) >= limit:
            break

        # Find the profile JSON (same name as directory)
        profile_json = pdir / f"{pdir.name}.json"
        if not profile_json.exists():
            # Fallback: find any JSON that isn't enrichment/claims/source_chunks
            skip_names = {"enrichment.json", "claims.json", "source_chunks.json"}
            for jf in pdir.glob("*.json"):
                if jf.name not in skip_names:
                    profile_json = jf
                    break

        if not profile_json or not profile_json.exists():
            continue

        # Skip if already enriched (when filter is on)
        if filter_no_enrichment and (pdir / "enrichment.json").exists():
            continue

        try:
            data = json.loads(profile_json.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("Failed to read %s: %s", profile_json, e)
            continue

        name = data.get("name", "")
        if not name or len(name.strip()) < 3:
            continue

        # Filter out non-professor profiles
        if filter_professor_level and not _is_professor_level(name, pdir.name, expertise_lookup):
            skipped_non_prof += 1
            continue

        # Try to extract department from profile data
        department = ""
        clean_text = data.get("clean_text", "") or data.get("raw_text", "")
        if clean_text:
            # Try explicit "Department of X" first
            dept_match = re.search(
                r"(?:department|dept\.?)\s+(?:of\s+)?([A-Z][A-Za-z\s&,]+?)(?:\n|\.|\|)",
                clean_text[:500],
            )
            if dept_match:
                department = dept_match.group(1).strip()
            else:
                # OSU profiles: "Title, Department Name" pattern
                # e.g. "Professor, Electrical and Computer Engineering"
                # e.g. "Graduate Research Associate, Chemical and Biomolecular Engineering"
                role_dept = re.search(
                    r"(?:Professor|Associate Professor|Assistant Professor|Instructor|"
                    r"Lecturer|Research (?:Associate|Scientist|Fellow)|Fellow|"
                    r"Graduate (?:Research|Teaching) (?:Associate|Assistant)|"
                    r"Visiting (?:Scholar|Fellow|Professor|Assistant Professor)),?\s+"
                    r"([A-Z][A-Za-z\s&,]+?)(?:\n|\.|$)",
                    clean_text[:500],
                )
                if role_dept:
                    department = role_dept.group(1).strip()
                    # Clean trailing commas or whitespace
                    department = department.rstrip(", ")

        queries.append(ProfessorQuery(
            profile_id=data.get("profile_id", pdir.name),
            name=name,
            university="Ohio State University",
            department=department,
            profile_url=data.get("profile_url", ""),
        ))

    if skipped_non_prof:
        logger.info("Filtered out %d non-professor profiles (PhD students, TAs, postdocs, etc.)", skipped_non_prof)

    return queries
