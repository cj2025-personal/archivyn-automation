"""
Post-process existing chunks.json files to remove boilerplate and tiny chunks.

Usage:
  python clean_existing_chunks.py --chunks-root output/url_list_runs/20260205_213604/chunked_profiles
  python clean_existing_chunks.py --chunks-file output/.../chunks.json --profile-name "Person Name"
"""
import argparse
import json
import re
import html
import unicodedata
from pathlib import Path


NAV_TERMS_RE = re.compile(
    r"\b(home|about|contact|search|menu|navigation|skip to content|privacy|terms|"
    r"copyright|all rights reserved|sitemap|back to top|top of page|bottom of page|"
    r"login|sign in|sign up|site map|site navigation|breadcrumb)\b",
    re.I,
)

MARKER_RE = re.compile(
    r"(===\s*(seed url|profile page|webpage|source)\s*===)|"
    r"(cr[i|l]tical rules|output format|text segment to analyze|json output|return only valid json)",
    re.I,
)

POLICY_RE = re.compile(
    r"\b(california(n)? residents?|ccpa|gdpr|privacy policy|terms of service|"
    r"cookie(s)?|consent|personal information|data protection|tracking)\b",
    re.I,
)

RDF_RE = re.compile(
    r"\b(dbo|dbr|dbp|dbc|rdf|rdfs|owl|foaf|wikidata|schema|prov|dbt|gold|umbel-rc|dct)\b\s*[:]",
    re.I,
)
WIKI_RE = re.compile(
    r"\b(wikipedia-en|wiki-commons|wikidata)\b|special:filepath|oldid=\d+",
    re.I,
)
XSD_RE = re.compile(r"\b(xsd:date|xsd:integer)\b", re.I)
LANG_TAG_RE = re.compile(r"\(([a-z]{2,3})\)")

BASE64_RE = re.compile(r"[A-Za-z0-9+/]{50,}={0,2}")
HEX_RE = re.compile(r"\b[a-f0-9]{40,}\b", re.I)
HTML_TAG_RE = re.compile(r"<[^>]+>")
SCRIPT_STYLE_RE = re.compile(r"(?is)<(script|style)[^>]*>.*?</\1>")
BLOCK_TAG_RE = re.compile(r"(?is)</?(p|div|li|ul|ol|h[1-6]|br|hr|section|article|nav|header|footer|table|tr|td|th)[^>]*>")


def _normalize_html(text: str) -> str:
    if not text:
        return ""
    # Normalize unicode and HTML entities
    text = unicodedata.normalize("NFKC", text)
    text = html.unescape(text)
    # Drop script/style blocks
    text = SCRIPT_STYLE_RE.sub(" ", text)
    # Turn block tags into newlines to preserve segment boundaries
    text = BLOCK_TAG_RE.sub("\n", text)
    # Remove remaining tags
    text = HTML_TAG_RE.sub(" ", text)
    return text


def _is_garbled(line: str) -> bool:
    if not line:
        return True
    # High ratio of non-word characters
    non_word = re.sub(r"[\w\s]", "", line)
    if len(non_word) > max(10, int(len(line) * 0.4)):
        return True
    # Base64 or hex-like blobs
    if BASE64_RE.search(line) or HEX_RE.search(line):
        return True
    return False


def _is_rdf_noise(line: str) -> bool:
    if not line:
        return False
    if RDF_RE.search(line) or WIKI_RE.search(line) or XSD_RE.search(line):
        return True
    # If line is dominated by language tags like (en) (fr) (ru) etc.
    lang_tags = LANG_TAG_RE.findall(line)
    if len(lang_tags) >= 3:
        return True
    return False


def _is_non_english(line: str) -> bool:
    """
    Drop lines with high ratio of non-Latin letters.
    Keeps accents but filters heavy CJK/Arabic/Cyrillic blocks.
    """
    if not line:
        return False
    # Count letters
    letters = [ch for ch in line if ch.isalpha()]
    if not letters:
        return False
    non_latin = 0
    for ch in letters:
        code = ord(ch)
        # Basic Latin + Latin-1 Supplement + Latin Extended
        if 0x0000 <= code <= 0x024F:
            continue
        non_latin += 1
    ratio = non_latin / max(1, len(letters))
    return ratio >= 0.25


def _strip_non_english_prefix(line: str) -> str:
    """
    Strip leading non-Latin or language-tag prefixes and common DBpedia caption noise.
    """
    if not line:
        return line
    # Remove known caption noise
    line = re.sub(r"\bcaption\s+official\s+nobel\s+prize\s+photo,?\s*", "", line, flags=re.I)
    # If a line starts with non-Latin text and later has English, trim to first ASCII letter
    m = re.search(r"[A-Za-z]", line)
    if not m:
        return line
    prefix = line[:m.start()]
    if LANG_TAG_RE.search(prefix):
        return line[m.start():].lstrip()
    for ch in prefix:
        if ch.isalpha() and ord(ch) > 0x024F:
            return line[m.start():].lstrip()
    return line


def clean_chunk_text(text: str) -> str:
    if not text:
        return ""

    text = _normalize_html(text)
    # Normalize line breaks
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    lines = [ln.strip() for ln in text.split("\n")]
    cleaned_lines = []

    for line in lines:
        if not line:
            continue
        line = _strip_non_english_prefix(line)
        if MARKER_RE.search(line):
            continue
        if NAV_TERMS_RE.search(line):
            continue
        if POLICY_RE.search(line):
            # Drop policy/legal/cookie related lines
            continue
        if _is_rdf_noise(line):
            continue
        if _is_non_english(line):
            continue
        if _is_garbled(line):
            continue
        # Drop very short lines that are likely UI noise
        if len(line) < 3:
            continue
        cleaned_lines.append(line)

    cleaned = " ".join(cleaned_lines)
    # Remove lingering prompt artifacts inline
    cleaned = MARKER_RE.sub(" ", cleaned)
    # Remove base64/hex blobs inline
    cleaned = BASE64_RE.sub(" ", cleaned)
    cleaned = HEX_RE.sub(" ", cleaned)
    # Normalize whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def load_profile_name(chunks_path: Path, profile_name_arg: str | None) -> str:
    if profile_name_arg:
        return profile_name_arg.strip()
    # Try to locate profile JSON next to chunks
    # chunks path: .../chunked_profiles/<id>/chunks.json
    try:
        profile_id = chunks_path.parent.name
        profiles_root = chunks_path.parents[2] / "profiles"
        profile_json = profiles_root / profile_id / f"{profile_id}.json"
        if profile_json.exists():
            data = json.loads(profile_json.read_text(encoding="utf-8"))
            return (data.get("name") or "").strip()
    except Exception:
        pass
    return ""


def is_relevant(text: str, profile_name: str) -> bool:
    if not profile_name:
        return True
    name = profile_name.lower()
    parts = [p for p in name.split() if len(p) >= 3]
    last = parts[-1] if parts else ""
    text_lc = text.lower()
    return name in text_lc or (last and last in text_lc)


def should_drop(text: str, profile_name: str) -> bool:
    raw = text.strip()
    if not raw:
        return True
    if raw.startswith("===") and raw.endswith("==="):
        return True
    if raw.startswith("=== SEED URL") or raw.startswith("=== PROFILE PAGE"):
        return True
    if MARKER_RE.search(raw):
        return True
    if NAV_TERMS_RE.search(raw):
        return True
    if POLICY_RE.search(raw):
        return True
    if RDF_RE.search(raw) or WIKI_RE.search(raw) or XSD_RE.search(raw):
        return True
    if BASE64_RE.search(raw) or HEX_RE.search(raw):
        return True
    if _is_non_english(raw):
        return True
    words = raw.split()
    if len(words) < 30 and not is_relevant(raw, profile_name):
        return True
    return False


def clean_chunks_file(chunks_path: Path, profile_name_arg: str | None) -> None:
    data = json.loads(chunks_path.read_text(encoding="utf-8"))
    sections = data.get("sections", {})
    profile_name = load_profile_name(chunks_path, profile_name_arg)

    cleaned_sections = {}
    removed = 0
    kept = 0
    modified = 0

    # Support alternate list-style chunks (if present)
    if isinstance(data, list):
        for item in data:
            text = item.get("text", "") or ""
            cleaned_text = clean_chunk_text(text)
            if cleaned_text != text:
                item["text"] = cleaned_text
                modified += 1
            if should_drop(cleaned_text, profile_name):
                removed += 1
                item["text"] = ""
            else:
                kept += 1
        chunks_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{chunks_path}: kept={kept}, removed={removed}, modified={modified}, profile_name='{profile_name}'")
        return

    for section, chunks in sections.items():
        new_chunks = []
        for c in chunks:
            text = c.get("text", "") or ""
            cleaned_text = clean_chunk_text(text)
            if cleaned_text != text:
                c["text"] = cleaned_text
                modified += 1
            if should_drop(cleaned_text, profile_name):
                removed += 1
                continue
            new_chunks.append(c)
            kept += 1
        cleaned_sections[section] = new_chunks

    data["sections"] = cleaned_sections
    chunks_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{chunks_path}: kept={kept}, removed={removed}, modified={modified}, profile_name='{profile_name}'")


def main():
    parser = argparse.ArgumentParser(description="Clean existing chunks.json files")
    parser.add_argument("--chunks-root", type=str, default=None, help="Root folder to search for chunks.json")
    parser.add_argument("--chunks-file", type=str, default=None, help="Single chunks.json file to clean")
    parser.add_argument("--profile-name", type=str, default=None, help="Profile name override")
    args = parser.parse_args()

    if args.chunks_file:
        clean_chunks_file(Path(args.chunks_file), args.profile_name)
        return

    if not args.chunks_root:
        raise SystemExit("Provide --chunks-root or --chunks-file")

    root = Path(args.chunks_root)
    for cf in root.rglob("chunks.json"):
        clean_chunks_file(cf, args.profile_name)


if __name__ == "__main__":
    main()
