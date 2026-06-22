"""
Sync local chunked_profiles (chunks.json) into a separate MongoDB collection.
Creates documents in the same structure as the existing 'scholars' collection,
and additionally stores section-grouped chunks in rag_context.

Usage:
  python sync_local_chunked_to_mongodb.py --runs output/url_list_runs/20260205_204944 output/url_list_runs/20260205_231446 output/url_list_runs/20260207_003302 output/url_list_runs/20260207_142149
  python sync_local_chunked_to_mongodb.py --chunks-root output/url_list_runs/20260207_003302/chunked_profiles --profiles-root output/url_list_runs/20260207_003302/profiles

Flags:
  --collection legend_scholars        # target collection (default: legend_scholars)
  --no-llm                            # disable LLM summaries
"""
import os
import json
import time
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
from collections import defaultdict

from dotenv import load_dotenv

from config.mongodb_utils import create_mongo_client, resolve_mongo_db_name


class LocalChunkedMongoSync:
    def __init__(
        self,
        collection_name: str = "legend_scholars",
        use_llm: bool = True,
        scholar_type: Optional[str] = None,
    ):
        load_dotenv(dotenv_path=".env")

        mongodb_uri = os.getenv("MONGODB_URI")
        if not mongodb_uri:
            raise ValueError("MONGODB_URI not found in environment variables")

        self.mongo_client = create_mongo_client(mongodb_uri)
        db_name = resolve_mongo_db_name(mongodb_uri)
        self.db = self.mongo_client[db_name]
        self.collection = self.db[collection_name]
        self.collection_name = collection_name
        self.use_llm = use_llm
        # When set (e.g. "osu"), tag each upserted document so profiles in a
        # shared collection like ``scholars`` keep their category. Upserts use
        # $set, so this never clears unrelated fields on existing docs.
        self.scholar_type = scholar_type

        self.openai_client = None
        self.http_client = None
        self.chunk_summary_max_chars = 320
        self.section_summary_max_chars = 900
        self.long_bio_max_chars = 900
        if self.use_llm:
            openai_key = os.getenv("OPENAI_API_KEY")
            if not openai_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables")
            try:
                from openai import OpenAI
                import httpx
                self.http_client = httpx.Client(timeout=120.0)
                self.openai_client = OpenAI(api_key=openai_key, http_client=self.http_client)
            except Exception:
                from openai import OpenAI
                self.openai_client = OpenAI(api_key=openai_key)

    def _call_llm_json(self, prompt: str, max_tokens: int = 600) -> Dict[str, Any]:
        if not self.openai_client:
            return {}
        last_error = None
        for attempt in range(3):
            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "Return only valid JSON, no additional text."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                    response_format={"type": "json_object"},
                    max_tokens=max_tokens,
                )
                return json.loads(response.choices[0].message.content)
            except Exception as e:
                last_error = e
                time.sleep(1.5 * (2 ** attempt))
        print(f"[LLM] Error after retries: {last_error}")
        return {}

    @staticmethod
    def _load_json(path: Path) -> Dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _extract_name_parts(full_name: str) -> Dict[str, str]:
        if not full_name:
            return {"first": "", "middle": "", "last": "", "title": "", "suffix": ""}
        title = ""
        for t in ["Dr.", "Dr", "Professor", "Prof."]:
            if full_name.startswith(t):
                title = t
                full_name = full_name.replace(t, "").strip()
                break
        parts = full_name.split()
        if len(parts) == 0:
            return {"first": "", "middle": "", "last": "", "title": title, "suffix": ""}
        if len(parts) == 1:
            return {"first": parts[0], "middle": "", "last": "", "title": title, "suffix": ""}
        if len(parts) == 2:
            return {"first": parts[0], "middle": "", "last": parts[1], "title": title, "suffix": ""}
        return {
            "first": parts[0],
            "middle": " ".join(parts[1:-1]),
            "last": parts[-1],
            "title": title,
            "suffix": "",
        }

    @staticmethod
    def _avatar_initial(name_parts: Dict[str, str]) -> str:
        if name_parts.get("first"):
            return name_parts["first"][0].upper()
        if name_parts.get("last"):
            return name_parts["last"][0].upper()
        return "?"

    def _aggregate_chunks_by_section(self, sections: Dict[str, List[Dict[str, Any]]]) -> Dict[str, str]:
        aggregated = {}
        for section, chunks in sections.items():
            ordered = sorted(chunks, key=lambda c: c.get("order", c.get("chunk_index", 0)))
            texts = [c.get("text", "") for c in ordered if c.get("text")]
            aggregated[section] = "\n\n".join(texts)
        return aggregated

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[a-z0-9]{4,}", (text or "").lower())

    @staticmethod
    def _load_source_chunks_payload(profiles_root: Optional[Path], profile_id: str) -> Dict[str, Any]:
        if not profiles_root:
            return {}
        source_chunks_path = profiles_root / profile_id / "source_chunks.json"
        if not source_chunks_path.exists():
            return {}
        try:
            return json.loads(source_chunks_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    @staticmethod
    def _normalize_source_catalog(source_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for src in source_payload.get("sources") or []:
            if not isinstance(src, dict):
                continue
            out.append(
                {
                    "source_id": str(src.get("source_id") or ""),
                    "source_url": str(src.get("source_url") or ""),
                    "resolved_url": str(src.get("resolved_url") or ""),
                    "source_type": str(src.get("source_type") or ""),
                    "status": str(src.get("status") or ""),
                    "allowed_use": str(src.get("allowed_use") or ""),
                    "license_type": str(src.get("license_type") or ""),
                    "robots_allowed": src.get("robots_allowed"),
                    "paywalled": bool(src.get("paywalled")),
                }
            )
        return out

    def _prepare_source_chunks(self, source_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        prepared: List[Dict[str, Any]] = []
        for src_chunk in source_payload.get("chunks") or []:
            if not isinstance(src_chunk, dict):
                continue
            text = str(src_chunk.get("text") or "").strip()
            if len(text) < 80:
                continue
            tokens = set(self._tokenize(text))
            if not tokens:
                continue
            prepared.append(
                {
                    "source_chunk_id": str(src_chunk.get("chunk_id") or ""),
                    "source_id": str(src_chunk.get("source_id") or ""),
                    "source_url": str(src_chunk.get("source_url") or ""),
                    "source_type": str(src_chunk.get("source_type") or ""),
                    "tokens": tokens,
                }
            )
        return prepared

    def _best_source_refs_for_chunk(
        self,
        chunk_text: str,
        prepared_source_chunks: List[Dict[str, Any]],
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        tokens = set(self._tokenize(chunk_text))
        if not tokens:
            return []

        scored: List[Dict[str, Any]] = []
        token_count = len(tokens)
        for src in prepared_source_chunks:
            src_tokens = src["tokens"]
            hits = len(tokens.intersection(src_tokens))
            if hits == 0:
                continue
            overlap_ratio = hits / max(1, token_count)
            src_overlap = hits / max(1, len(src_tokens))
            score = 0.75 * overlap_ratio + 0.25 * src_overlap
            if hits < 6 and overlap_ratio < 0.12:
                continue
            scored.append(
                {
                    "source_chunk_id": src["source_chunk_id"],
                    "source_id": src["source_id"],
                    "source_url": src["source_url"],
                    "source_type": src["source_type"],
                    "score": round(float(score), 4),
                    "token_overlap": round(float(overlap_ratio), 4),
                    "token_hits": int(hits),
                }
            )

        scored.sort(key=lambda x: (x["score"], x["token_hits"]), reverse=True)
        return scored[:top_k]

    def _attach_source_provenance(
        self,
        section_chunks: Dict[str, List[Dict[str, Any]]],
        source_payload: Dict[str, Any],
    ) -> tuple[Dict[str, List[Dict[str, Any]]], Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]], Dict[str, Any]]:
        source_catalog = self._normalize_source_catalog(source_payload)
        prepared_source_chunks = self._prepare_source_chunks(source_payload)
        chunk_source_map: Dict[str, List[Dict[str, Any]]] = {}

        matched_chunks = 0
        total_chunks = 0
        for _, chunks in section_chunks.items():
            for chunk in chunks:
                total_chunks += 1
                text = str(chunk.get("text") or chunk.get("summary") or "").strip()
                refs = self._best_source_refs_for_chunk(text, prepared_source_chunks, top_k=3)
                source_urls = []
                seen_urls = set()
                for ref in refs:
                    url = str(ref.get("source_url") or "").strip()
                    if not url or url in seen_urls:
                        continue
                    seen_urls.add(url)
                    source_urls.append(url)
                chunk["source_refs"] = refs
                chunk["source_urls"] = source_urls
                chunk["primary_source_url"] = source_urls[0] if source_urls else ""
                chunk_id = str(chunk.get("chunk_id") or "")
                if refs and chunk_id:
                    matched_chunks += 1
                    chunk_source_map[chunk_id] = refs

        coverage = (matched_chunks / total_chunks) if total_chunks else 0.0
        provenance_stats = {
            "total_chunks": total_chunks,
            "matched_chunks": matched_chunks,
            "coverage_ratio": round(float(coverage), 4),
            "source_count": len(source_catalog),
            "source_chunk_count": len(prepared_source_chunks),
            "version": "chunk_source_v1",
        }
        return section_chunks, chunk_source_map, source_catalog, provenance_stats

    def _build_section_chunks(self, sections: Dict[str, List[Dict[str, Any]]], professor_name: str) -> Dict[str, List[Dict[str, Any]]]:
        section_chunks: Dict[str, List[Dict[str, Any]]] = {}
        for section, chunks in sections.items():
            ordered = sorted(chunks, key=lambda c: c.get("order", c.get("chunk_index", 0)))
            cleaned = []
            for c in ordered:
                summary = ""
                if self.use_llm:
                    summary = self._summarize_chunk(
                        c.get("text", ""),
                        professor_name,
                        section,
                    )
                cleaned.append({
                    "chunk_id": c.get("chunk_id", ""),
                    "order": c.get("order", c.get("chunk_index", 0)),
                    "text": c.get("text", ""),
                    "summary": summary,
                    "professor_name": professor_name,
                })
            section_chunks[section] = cleaned
        return section_chunks

    def _generate_section_summary_with_llm(
        self,
        section_name: str,
        context: Dict[str, str],
        professor_name: str,
    ) -> Dict[str, Any]:
        if not self.openai_client:
            return {}

        # Build context string (cap per section)
        context_parts = []
        for section, text in context.items():
            if text:
                context_parts.append(f"[Section: {section}]\n{text[:2000]}...")
        context_string = "\n\n".join(context_parts)

        if section_name == "about":
            prompt = f"""Extract a brief biography summary for {professor_name} based on the following context.

Context from profile:
{context_string}

Extract and return as JSON:
{{
  "short_bio": "1-2 sentence biography (max 300 characters) including current position, institution, and primary field",
  "current_position": "Current job title/position",
  "institution": "University or organization name",
  "department": "Department name if available",
  "field_of_study": "Primary field of study",
  "location": "City, State or location if available"
}}
"""
        elif section_name == "background_and_work":
            prompt = f"""Extract background and work information for {professor_name} based on the following context.

Context from profile:
{context_string}

Extract and return as JSON:
{{
  "background_summary": "2-3 sentence background summary (max 200 characters)",
  "education_summary": [
    {{
      "degree": "Degree type and field",
      "institution": "Institution name",
      "year": "Year if available, else null",
      "brief": "One-line description (max 100 characters)"
    }}
  ],
  "research_focus": ["Research area 1", "Research area 2", "Research area 3"],
  "current_work": "1-2 sentences about current research/work (max 200 characters)",
  "methodology": ["Methodology 1", "Methodology 2"]
}}
"""
        elif section_name == "milestones":
            prompt = f"""Extract 4-6 most significant milestones for {professor_name} based on the following context.

Context from profile:
{context_string}

Extract and return as JSON:
{{
  "milestones": [
    {{
      "title": "Milestone title",
      "year": "Year if available, else null",
      "type": "Fellowship, Award, Career, Publication, or Service",
      "description": "Brief description (max 150 characters)",
      "icon": "award, teaching, career, publication, or service",
      "order": 1
    }}
  ]
}}
"""
        elif section_name == "publications":
            prompt = f"""Extract 3-5 most important/featured publications for {professor_name} based on the following context.

Context from profile:
{context_string}

Extract and return as JSON:
{{
  "featured_publications": [
    {{
      "title": "Publication title",
      "publisher": "Publisher name",
      "year": "Year if available, else null",
      "type": "Book, Article, or Other",
      "brief_description": "Brief description (max 200 characters)",
      "is_featured": true,
      "order": 1
    }}
  ],
  "total_publications_count": "Total number of publications mentioned or estimated"
}}
"""
        else:
            return {}

        result = self._call_llm_json(prompt, max_tokens=2000)
        if not result:
            print(f"[LLM] Error generating {section_name} summary")
        return result

    def _summarize_chunk(self, text: str, professor_name: str, section_name: str) -> str:
        if not self.openai_client or not text:
            return ""
        chunk = text.strip()
        if len(chunk) > 3000:
            chunk = chunk[:3000]
        prompt = f"""Summarize the following chunk about {professor_name}.

Rules:
- 1–2 sentences, max {self.chunk_summary_max_chars} characters.
- Keep names, dates, titles, and concrete facts.
- Do not add new facts or guesses.
- Plain English only.

Section: {section_name}
Chunk:
{chunk}

Return JSON:
{{"summary": "..."}}"""
        result = self._call_llm_json(prompt, max_tokens=300)
        return (result.get("summary") or "").strip()

    def _summarize_section(self, section_name: str, chunk_summaries: List[str], professor_name: str) -> str:
        if not self.openai_client or not chunk_summaries:
            return ""
        context = "\n".join([s for s in chunk_summaries if s])
        if len(context) > 4000:
            context = context[:4000]
        prompt = f"""Create a concise section summary for {professor_name}.

Rules:
- 4–6 sentences, max {self.section_summary_max_chars} characters.
- Use only the provided chunk summaries.
- Plain English, no bullets.

Section: {section_name}
Chunk summaries:
{context}

Return JSON:
{{"summary": "..."}}"""
        result = self._call_llm_json(prompt, max_tokens=600)
        return (result.get("summary") or "").strip()

    def _summarize_long_bio(self, aggregated_context: Dict[str, str], professor_name: str) -> str:
        if not self.openai_client:
            return ""
        # Build short context from top sections
        context_parts = []
        for section, text in aggregated_context.items():
            if text:
                context_parts.append(f"[Section: {section}]\n{text[:1500]}...")
        context = "\n\n".join(context_parts)
        if len(context) > 6000:
            context = context[:6000]
        prompt = f"""Write a richer profile overview for {professor_name}.

Rules:
- 5–8 sentences, max {self.long_bio_max_chars} characters.
- Include current role/affiliation, key contributions, major awards, and notable positions if present.
- Plain English, no bullets, no speculation.

Context:
{context}

Return JSON:
{{"summary": "..."}}"""
        result = self._call_llm_json(prompt, max_tokens=700)
        return (result.get("summary") or "").strip()

    def _create_scholar_document(
        self,
        profile_id: str,
        professor_name: str,
        sections: Dict[str, List[Dict[str, Any]]],
        source_payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        name_parts = self._extract_name_parts(professor_name)
        aggregated_context = self._aggregate_chunks_by_section(sections)
        section_chunks = self._build_section_chunks(sections, professor_name)
        section_chunks, chunk_source_map, source_catalog, provenance_stats = self._attach_source_provenance(
            section_chunks=section_chunks,
            source_payload=source_payload or {},
        )
        sections_available = list(aggregated_context.keys())

        about_data = self._generate_section_summary_with_llm("about", aggregated_context, professor_name) if self.use_llm else {}
        background_data = self._generate_section_summary_with_llm("background_and_work", aggregated_context, professor_name) if self.use_llm else {}
        milestones_data = self._generate_section_summary_with_llm("milestones", aggregated_context, professor_name) if self.use_llm else {}
        publications_data = self._generate_section_summary_with_llm("publications", aggregated_context, professor_name) if self.use_llm else {}
        long_bio = self._summarize_long_bio(aggregated_context, professor_name) if self.use_llm else ""

        # Build section summaries from chunk summaries
        section_summaries = {}
        if self.use_llm:
            for section, chunk_list in section_chunks.items():
                chunk_summaries = [c.get("summary", "") for c in chunk_list if c.get("summary")]
                section_summaries[section] = self._summarize_section(section, chunk_summaries, professor_name) if chunk_summaries else ""

        now = datetime.now(timezone.utc).isoformat()

        document = {
            "_id": profile_id,
            "profile_id": profile_id,
            "professor_name": professor_name,
            "name": {
                "full": professor_name,
                "display": f"{name_parts.get('title', '')} {professor_name}".strip(),
                "title": name_parts.get("title", ""),
                "first": name_parts.get("first", ""),
                "middle": name_parts.get("middle", ""),
                "last": name_parts.get("last", ""),
                "suffix": name_parts.get("suffix", ""),
            },
            "about": {
                "short_bio": about_data.get("short_bio", ""),
                "long_bio": long_bio,
                "current_position": about_data.get("current_position", ""),
                "institution": about_data.get("institution", ""),
                "department": about_data.get("department", ""),
                "field_of_study": about_data.get("field_of_study", ""),
                "location": about_data.get("location", ""),
                "avatar_url": None,
                "avatar_initial": self._avatar_initial(name_parts),
            },
            "background_and_work": {
                "background_summary": background_data.get("background_summary", ""),
                "education_summary": background_data.get("education_summary", []),
                "research_focus": background_data.get("research_focus", []),
                "current_work": background_data.get("current_work", ""),
                "methodology": background_data.get("methodology", []),
            },
            "milestones": milestones_data.get("milestones", []),
            "publications": {
                "featured_publications": publications_data.get("featured_publications", []),
                "total_publications_count": int(publications_data.get("total_publications_count", 0)) if str(publications_data.get("total_publications_count", "0")).isdigit() else 0,
                "show_more_link": True,
            },
            "links_and_media": {
                "social_profiles": [],
                "references": [],
                "featured_video": None,
            },
            "display": {
                "avatar_initial": self._avatar_initial(name_parts),
                "last_name_initial": name_parts.get("last", "?")[0].upper() if name_parts.get("last") else "?",
                "profile_image_url": None,
                "is_featured": False,
                "display_order": 0,
                "visibility": "draft",
                "last_updated": now,
            },
            "metadata": {
                "search_keywords": self._generate_search_keywords(professor_name, about_data, background_data),
                "tags": background_data.get("research_focus", []),
                "field_of_study": about_data.get("field_of_study", ""),
                "university": about_data.get("institution", ""),
                "last_name_initial": name_parts.get("last", "?")[0].upper() if name_parts.get("last") else "?",
            },
            "rag_context": {
                "pinecone_indexed": False,
                "professor_id": profile_id,
                "professor_name": professor_name,
                "chunk_count": sum(len(v) for v in sections.values()),
                "sections_available": sections_available,
                "section_chunks": section_chunks,
                "section_text": aggregated_context,
                "section_summaries": section_summaries,
                "last_indexed_at": now,
                "source": "local_chunked_profiles",
                "source_data_hash": "",
                "source_catalog": source_catalog,
                "chunk_source_map": chunk_source_map,
                "provenance": provenance_stats,
            },
            "admin": {
                "created_by": "system",
                "created_at": now,
                "updated_by": None,
                "updated_at": None,
                "last_reviewed_at": None,
                "review_status": "pending_curation",
                "curation_progress": {
                    "sections_completed": 0,
                    "sections_total": 5,
                    "completion_percentage": 0,
                },
                "llm_generation_metadata": {
                    "model_used": "gpt-4o-mini" if self.use_llm else "none",
                    "prompt_version": "v1.0",
                    "generated_at": now,
                    "sections_generated": ["about", "background_and_work", "milestones", "publications"] if self.use_llm else [],
                },
            },
        }
        scholar_type = getattr(self, "scholar_type", None)
        if scholar_type:
            document["scholar_type"] = scholar_type
        return document

    @staticmethod
    def _generate_search_keywords(name: str, about_data: Dict, background_data: Dict) -> List[str]:
        keywords: List[str] = []
        name_parts = name.split()
        keywords.extend(name_parts)
        if about_data.get("institution"):
            keywords.append(about_data["institution"])
            keywords.extend(about_data["institution"].split())
        if about_data.get("field_of_study"):
            keywords.append(about_data["field_of_study"])
            keywords.extend(about_data["field_of_study"].split())
        if background_data.get("research_focus"):
            keywords.extend(background_data["research_focus"])
        # De-dupe
        cleaned = []
        for k in keywords:
            k = (k or "").strip()
            if not k:
                continue
            if k not in cleaned:
                cleaned.append(k)
        return cleaned[:20]

    @staticmethod
    def _extract_name_from_chunks_data(data: Dict[str, Any]) -> str:
        top_level_name = (data.get("professor_name") or "").strip()
        if top_level_name:
            return top_level_name

        sections = data.get("sections", {})
        if not isinstance(sections, dict):
            return ""

        for chunk_list in sections.values():
            if not isinstance(chunk_list, list):
                continue
            for chunk in chunk_list:
                if not isinstance(chunk, dict):
                    continue
                candidate = (chunk.get("professor_name") or "").strip()
                if candidate:
                    return candidate
        return ""

    def sync_chunks_file(self, chunks_path: Path, profiles_root: Optional[Path]) -> bool:
        try:
            data = self._load_json(chunks_path)
            profile_id = data.get("profile_id") or chunks_path.parent.name
            sections = data.get("sections", {})
            if not sections:
                print(f"[Skip] No sections in {chunks_path}")
                return False

            professor_name = ""
            if profiles_root:
                profile_json = profiles_root / profile_id / f"{profile_id}.json"
                if profile_json.exists():
                    prof_data = self._load_json(profile_json)
                    professor_name = (
                        prof_data.get("name")
                        or prof_data.get("professor_name")
                        or ""
                    ).strip()

            if not professor_name:
                professor_name = self._extract_name_from_chunks_data(data)

            if not professor_name:
                professor_name = "Unknown"

            source_payload = self._load_source_chunks_payload(profiles_root, profile_id)
            document = self._create_scholar_document(
                profile_id,
                professor_name,
                sections,
                source_payload=source_payload,
            )
            self.collection.update_one({"profile_id": profile_id}, {"$set": document}, upsert=True)
            print(f"[MongoDB] Upserted {professor_name} ({profile_id}) into {self.collection_name}")
            return True
        except Exception as e:
            print(f"[Error] Failed to sync {chunks_path}: {e}")
            return False

    def sync_from_roots(self, chunks_root: Path, profiles_root: Optional[Path]) -> None:
        if not chunks_root.exists():
            print(f"[Skip] Missing chunks root: {chunks_root}")
            return
        for chunks_path in chunks_root.rglob("chunks.json"):
            self.sync_chunks_file(chunks_path, profiles_root)
            time.sleep(0.2)

    def close(self) -> None:
        try:
            self.mongo_client.close()
        except Exception:
            pass
        http_client = getattr(self, "http_client", None)
        if http_client is not None:
            try:
                http_client.close()
            except Exception:
                pass


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Sync local chunked_profiles into MongoDB")
    parser.add_argument("--chunks-root", type=str, default=None, help="Path to chunked_profiles directory")
    parser.add_argument("--profiles-root", type=str, default=None, help="Path to profiles directory (for names)")
    parser.add_argument("--runs", nargs="*", default=None, help="One or more output/url_list_runs/<run_id> directories")
    parser.add_argument("--collection", type=str, default="legend_scholars", help="Target collection name")
    parser.add_argument("--scholar-type", type=str, default=None,
                        help="Tag each upserted doc with this scholar_type (e.g. 'osu').")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM summaries")
    args = parser.parse_args()

    sync = LocalChunkedMongoSync(
        collection_name=args.collection,
        use_llm=not args.no_llm,
        scholar_type=args.scholar_type,
    )
    try:
        if args.runs:
            for run_dir in args.runs:
                run_path = Path(run_dir)
                chunks_root = run_path / "chunked_profiles"
                profiles_root = run_path / "profiles"
                sync.sync_from_roots(chunks_root, profiles_root if profiles_root.exists() else None)
            return

        if not args.chunks_root:
            raise SystemExit("Provide --chunks-root or --runs")

        chunks_root = Path(args.chunks_root)
        profiles_root = Path(args.profiles_root) if args.profiles_root else None
        sync.sync_from_roots(chunks_root, profiles_root)
    finally:
        sync.close()


if __name__ == "__main__":
    main()
