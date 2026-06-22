"""
GPT-4o-mini based enrichment text cleaner.

Cleans raw enrichment text before chunking/vectorization:
- Source-type-specific cleaning prompts
- Cross-source deduplication of publications
- Standardizes formatting across different source formats
- Removes noise while preserving all factual academic content
"""

import logging
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Source-type groupings for prompt specialization
PUBLICATION_SOURCES = {
    "semantic_scholar", "openalex", "crossref", "google_scholar",
}
GRANT_SOURCES = {"nsf_grants", "nih_grants"}
NEWS_SOURCES = {"osu_news", "google_news"}
WEB_SOURCES = {"web_search"}
PROFILE_SOURCES = {"orcid", "osu_expertise"}
TEACHING_SOURCES = {"osu_courses", "rate_my_professor"}
MEDIA_SOURCES = {"youtube_lectures"}


def _get_cleaning_prompt(source_type: str) -> str:
    """Return a source-type-specific cleaning system prompt."""
    base = (
        "You are an academic data cleaning assistant. "
        "Clean the following enrichment text while preserving ALL factual content. "
        "Return ONLY the cleaned text with no explanations.\n\n"
    )

    if source_type == "publication":
        return base + (
            "This text contains publication data from an academic database.\n"
            "- Remove duplicate entries (same paper title appearing multiple times)\n"
            "- Standardize publication format: Title (Year). Journal. DOI.\n"
            "- Merge citation counts if the same paper appears with different counts\n"
            "- Remove metadata noise (API IDs, internal identifiers, boolean flags)\n"
            "- Keep: all paper titles, authors, years, journals, DOIs, abstracts, citation counts\n"
            "- Keep: h-index, i10-index, total citations, research areas/concepts\n"
            "- Remove: raw JSON structure markers, internal field names"
        )
    elif source_type == "grant":
        return base + (
            "This text contains research grant data.\n"
            "- Standardize dollar amounts (e.g., $1,234,567)\n"
            "- Standardize date formats\n"
            "- Remove metadata noise and internal identifiers\n"
            "- Keep: grant titles, PIs, amounts, dates, abstracts, funding agencies\n"
            "- Consolidate duplicate grants"
        )
    elif source_type == "news":
        return base + (
            "This text contains news articles mentioning a professor.\n"
            "- Remove HTML/boilerplate remnants, navigation text, cookie notices\n"
            "- Remove 'read more', 'share this', social media buttons text\n"
            "- Remove duplicate articles (same content, different sources)\n"
            "- Keep: article titles, dates, substantive content about the professor\n"
            "- Keep: quotes, research descriptions, awards, achievements"
        )
    elif source_type == "web":
        return base + (
            "This text was scraped from web pages about a professor.\n"
            "- Heavy cleaning needed: remove navigation, sidebars, footers, headers\n"
            "- Remove cookie consent, privacy policy, terms of service text\n"
            "- Remove HTML/CSS/JS remnants, inline styles, class names\n"
            "- Remove anti-bot text, CAPTCHA notices\n"
            "- Remove repetitive boilerplate across pages\n"
            "- Keep: all academic content, research descriptions, biographical info\n"
            "- Keep: publications, awards, positions, courses, collaborations"
        )
    elif source_type == "profile":
        return base + (
            "This text contains structured academic profile data.\n"
            "- Light cleaning: normalize formatting, fix spacing\n"
            "- Remove empty fields and placeholder values\n"
            "- Keep: career history, education, employment, publications, biography\n"
            "- Keep: ORCID IDs, profile URLs, department affiliations\n"
            "- Standardize date formats and position titles"
        )
    elif source_type == "teaching":
        return base + (
            "This text contains teaching-related data (courses, student reviews).\n"
            "- Standardize course listing format\n"
            "- Remove duplicate course entries across semesters\n"
            "- Clean up student review text (remove profanity, keep substance)\n"
            "- Keep: course names, numbers, descriptions, ratings, review themes\n"
            "- Keep: teaching quality indicators, difficulty ratings"
        )
    else:
        return base + (
            "Clean this text by removing noise while preserving all academic content.\n"
            "- Remove HTML/boilerplate/navigation remnants\n"
            "- Remove metadata noise and internal identifiers\n"
            "- Keep all factual academic information"
        )


def _classify_source(source_name: str) -> str:
    """Map a source name to its cleaning category."""
    if source_name in PUBLICATION_SOURCES:
        return "publication"
    elif source_name in GRANT_SOURCES:
        return "grant"
    elif source_name in NEWS_SOURCES:
        return "news"
    elif source_name in WEB_SOURCES:
        return "web"
    elif source_name in PROFILE_SOURCES:
        return "profile"
    elif source_name in TEACHING_SOURCES:
        return "teaching"
    return "general"


def _split_by_source(text: str) -> List[Tuple[str, str]]:
    """Split enrichment text into (source_name, section_text) pairs."""
    # Enrichment text uses "--- source_name ---" as section markers
    # Also handle "=== Source: source_name ===" format
    pattern = re.compile(
        r"(?:^|\n)\s*(?:---\s*(\w+)\s*---|===\s*(?:Source:\s*)?(\w+)\s*===)\s*\n",
        re.MULTILINE,
    )

    sections = []
    matches = list(pattern.finditer(text))

    if not matches:
        # No source markers found — treat entire text as one section
        return [("general", text)]

    # Content before the first source marker (header)
    header = text[:matches[0].start()].strip()
    if header:
        sections.append(("header", header))

    for i, match in enumerate(matches):
        source_name = match.group(1) or match.group(2) or "unknown"
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        if section_text:
            sections.append((source_name, section_text))

    return sections


class EnrichmentCleaner:
    """Clean raw enrichment text using GPT-4o-mini."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        max_tokens_per_section: int = 4000,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens_per_section = max_tokens_per_section
        self._client = None
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")

    def _get_client(self):
        """Lazy-init OpenAI client."""
        if self._client is None:
            if not self._api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY not set — required for enrichment cleaning"
                )
            from openai import OpenAI
            try:
                import httpx
                http_client = httpx.Client(timeout=60.0)
                self._client = OpenAI(api_key=self._api_key, http_client=http_client)
            except Exception:
                self._client = OpenAI(api_key=self._api_key)
            print(f"    [GPT] OpenAI client initialized (model={self.model})")
        return self._client

    def clean_section(self, source_name: str, text: str) -> str:
        """Clean a single source section using GPT-4o-mini."""
        if not text or len(text.strip()) < 50:
            print(f"    [GPT] Skipping {source_name} (too short: {len(text)} chars)")
            return text

        source_type = _classify_source(source_name)
        system_prompt = _get_cleaning_prompt(source_type)

        t0 = time.perf_counter()
        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens_per_section,
                timeout=30,
            )
            cleaned = response.choices[0].message.content.strip()
            elapsed = time.perf_counter() - t0

            # Token usage
            usage = response.usage
            in_tok = usage.prompt_tokens if usage else 0
            out_tok = usage.completion_tokens if usage else 0

            # Validate: if LLM returned very little, keep original
            if not cleaned or len(cleaned) < len(text) * 0.1:
                print(f"    [GPT] {source_name}: LLM returned too short ({len(cleaned)} chars vs {len(text)} original), keeping raw")
                return text

            reduction = (1 - len(cleaned) / max(len(text), 1)) * 100
            print(f"    [GPT] {source_name} ({source_type}): {len(text)} -> {len(cleaned)} chars ({reduction:+.0f}%) [{in_tok}+{out_tok} tokens, {elapsed:.1f}s]")
            return cleaned

        except Exception as e:
            elapsed = time.perf_counter() - t0
            print(f"    [GPT] {source_name}: ERROR after {elapsed:.1f}s - {e}")
            return text

    def clean_enrichment_text(self, raw_text: str) -> str:
        """Clean a full enrichment text file."""
        sections = _split_by_source(raw_text)
        print(f"    [Clean] Found {len(sections)} sections in enrichment text ({len(raw_text):,} chars)")

        cleaned_parts = []
        for source_name, section_text in sections:
            if source_name == "header":
                cleaned_parts.append(section_text)
                continue

            cleaned = self.clean_section(source_name, section_text)
            cleaned_parts.append(f"--- {source_name} ---\n{cleaned}")

        result = "\n\n".join(cleaned_parts)
        total_reduction = (1 - len(result) / max(len(raw_text), 1)) * 100
        print(f"    [Clean] Total: {len(raw_text):,} -> {len(result):,} chars ({total_reduction:+.0f}% overall)")
        return result

    def clean_file(self, input_path: Path) -> Path:
        """Clean an enrichment_text.txt file and save as enrichment_text_cleaned.txt."""
        raw_text = input_path.read_text(encoding="utf-8")
        cleaned = self.clean_enrichment_text(raw_text)

        output_path = input_path.parent / "enrichment_text_cleaned.txt"
        output_path.write_text(cleaned, encoding="utf-8")

        return output_path
