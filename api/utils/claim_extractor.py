"""
Lightweight claim extraction from cleaned chunks.
This is a heuristic extractor; for higher quality enable LLM-based extraction upstream.

The output schema is unchanged for downstream consumers:
    claim_id, researcher_id, claim_text, source_id, evidence_chunk_ids,
    confidence (float 0..1), tags (list[str]), paraphrase_quality.
"""
from __future__ import annotations

import re
import uuid
from typing import Dict, List, Optional


_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

_CLAIM_VERBS = {
    "is", "was", "are", "were", "served", "serves", "received", "earned", "joined",
    "appointed", "published", "authored", "wrote", "won", "led",
    "founded", "co-founded", "graduated", "completed", "holds", "born",
    "studied", "taught", "directed", "edited", "translated",
}

_TOPIC_TAGS = {
    "education": [
        "ph.d", "phd", "m.a", "m.s", "b.a", "b.s", "degree", "university",
        "college", "graduated", "doctorate", "alma mater",
    ],
    "awards": [
        "award", "honor", "honorary", "prize", "medal", "fellow", "fellowship",
        "laureate", "induct",
    ],
    "positions": [
        "professor", "chair", "director", "dean", "lecturer", "assistant",
        "associate", "founder", "co-founder", "president", "head of",
    ],
    "publications": [
        "published", "article", "journal", "book", "paper", "monograph",
        "edited volume", "author of",
    ],
    "speeches": ["speech", "remarks", "keynote", "lecture", "address", "talk"],
    "research": ["research", "study", "investigation", "experiment", "field work"],
    "biography": ["born", "grew up", "raised", "died", "passed away"],
}

_NOISE_PATTERNS = [
    r"verify you are human",
    r"captcha",
    r"cloudflare",
    r"attention required",
    r"access denied",
    r"loading\.\.\.",
    r"subscribe",
    r"sign in to continue",
    r"proof of work",
    r"anubis",
    r"jshelter",
    r"shopping basket",
    r"shopping cart",
    r"site archived",
    r"return to top",
    r"\bcookie\b",
]


def _sentence_tokens(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9']+", text.lower())


def _likely_claim(sentence: str) -> bool:
    if not sentence or len(sentence) < 40:
        return False
    if len(sentence) > 320:
        return False
    tokens = _sentence_tokens(sentence)
    if not tokens:
        return False
    s_lc = sentence.lower()
    if any(re.search(pat, s_lc) for pat in _NOISE_PATTERNS):
        return False
    if any(tok in _CLAIM_VERBS for tok in tokens):
        return True
    # Also allow if a year is present
    if re.search(r"\b(1[6-9]|20)\d{2}\b", sentence):
        return True
    return False


def _tag_topics(sentence: str) -> List[str]:
    s = sentence.lower()
    tags = []
    for tag, kws in _TOPIC_TAGS.items():
        if any(k in s for k in kws):
            tags.append(tag)
    return tags


def _name_match(sentence: str, profile_name: Optional[str]) -> bool:
    if not profile_name:
        return False
    name = profile_name.strip().lower()
    if not name:
        return False
    s = sentence.lower()
    if name in s:
        return True
    parts = [p for p in re.split(r"\s+", name) if len(p) >= 3]
    if not parts:
        return False
    last = parts[-1]
    return last in s


def _score_claim(
    sentence: str,
    *,
    profile_name: Optional[str],
    has_required_intent: bool,
    domain_quality: int,
    license_type: str,
) -> float:
    """Score a claim 0..1 based on signal strength.

    The previous extractor stamped everything 0.4. We now factor in:
      - whether the sentence mentions the profile subject (strongest signal),
      - whether the source has a passing required-intent classification,
      - source domain quality (.gov / .edu vs noise hosts),
      - license clarity (known PD/CC > unknown),
      - presence of a year, claim verb, or measurable noun (degree, award, year),
    so downstream filters can rank rather than treating all claims equally.
    """
    score = 0.20
    s_lc = sentence.lower()
    tokens = _sentence_tokens(sentence)

    if _name_match(sentence, profile_name):
        score += 0.30

    if any(tok in _CLAIM_VERBS for tok in tokens):
        score += 0.10

    if re.search(r"\b(1[6-9]|20)\d{2}\b", sentence):
        score += 0.10

    if has_required_intent:
        score += 0.10

    if domain_quality >= 2:
        score += 0.10
    elif domain_quality == 1:
        score += 0.05
    elif domain_quality < 0:
        score -= 0.20

    if license_type and license_type != "unknown":
        score += 0.05

    return round(min(1.0, max(0.0, score)), 3)


def extract_claims(
    researcher_id: str,
    chunks: List[Dict],
    max_claims_per_chunk: int = 3,
    *,
    profile_name: Optional[str] = None,
    sources_meta: Optional[List[Dict]] = None,
) -> List[Dict]:
    """
    Extract heuristic claims from cleaned chunk text.

    ``profile_name`` and ``sources_meta`` are optional. When supplied the
    extractor will (a) drop sentences that have no name overlap on low-quality
    sources, and (b) compute a real per-claim confidence rather than the
    legacy flat 0.4.
    """
    sources_index: Dict[str, Dict] = {}
    if sources_meta:
        for s in sources_meta:
            sid = s.get("source_id")
            if sid:
                sources_index[sid] = s

    claims: List[Dict] = []
    for chunk in chunks:
        text = chunk.get("text", "") or ""
        source_id = chunk.get("source_id", "")
        chunk_id = chunk.get("chunk_id", "")
        if not text:
            continue

        src_meta = sources_index.get(source_id, {})
        domain_quality = int(src_meta.get("intent_domain_quality", 0) or 0)
        license_type = str(src_meta.get("license_type", "") or "")
        has_required_intent = bool(src_meta.get("intent_required_hits"))

        # Split on sentence boundaries and newlines
        rough = []
        for part in text.splitlines():
            part = part.strip()
            if not part:
                continue
            rough.extend(_SENT_SPLIT_RE.split(part))
        sentences = [s.strip() for s in rough if s and s.strip()]

        extracted = 0
        for sent in sentences:
            sent = sent.strip()
            if not _likely_claim(sent):
                continue
            tokens = _sentence_tokens(sent)
            if not any(tok in _CLAIM_VERBS for tok in tokens) and not re.search(r"\b(1[6-9]|20)\d{2}\b", sent):
                continue

            # Strict per-claim subject filter: any claim that is going to
            # represent the legend in the chatbot/article generator must
            # mention them — pronoun-only claims are unattributable.
            if profile_name and not _name_match(sent, profile_name):
                # Drop from any non-premium source.
                if domain_quality <= 0:
                    continue
                # Even on .gov / .edu sources, require *some* claim signal.
                if not (
                    any(tok in _CLAIM_VERBS for tok in tokens)
                    and re.search(r"\b(1[6-9]|20)\d{2}\b", sent)
                ):
                    continue

            confidence = _score_claim(
                sent,
                profile_name=profile_name,
                has_required_intent=has_required_intent,
                domain_quality=domain_quality,
                license_type=license_type,
            )
            if confidence < 0.40:
                continue

            claim_id = f"clm_{uuid.uuid4()}"
            claims.append({
                "claim_id": claim_id,
                "researcher_id": researcher_id,
                "claim_text": sent,
                "source_id": source_id,
                "evidence_chunk_ids": [chunk_id] if chunk_id else [],
                "confidence": confidence,
                "tags": _tag_topics(sent),
                "paraphrase_quality": "heuristic",
                "name_match": _name_match(sent, profile_name) if profile_name else False,
            })
            extracted += 1
            if extracted >= max_claims_per_chunk:
                break
    return claims
