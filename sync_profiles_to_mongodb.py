"""
Script to sync profiles from Pinecone to MongoDB
Creates 'scholars' collection with LLM-generated summaries for each section
"""
import os
import json
import re
import unicodedata
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from tqdm import tqdm
import time
from dotenv import load_dotenv
from collections import defaultdict, Counter

# Load environment variables
load_dotenv()

# Import services
from api.services.vector_db import get_vector_db
from api.services.embeddings import get_embeddings_service
from config.pinecone_config import INDEX_NAME, INDEX_DIMENSION
from config.mongodb_utils import create_mongo_client, resolve_mongo_db_name

# MongoDB
try:
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure, OperationFailure
except ImportError:
    raise ImportError("pymongo not installed. Install with: pip install pymongo")

# LLM clients
try:
    from openai import OpenAI
    import httpx
except ImportError:
    raise ImportError("openai not installed. Install with: pip install openai")


class MongoDBScholarSync:
    """Sync profiles from Pinecone to MongoDB with LLM-generated summaries"""
    
    def __init__(self):
        """Initialize connections"""
        # MongoDB connection
        mongodb_uri = os.getenv("MONGODB_URI")
        if not mongodb_uri:
            raise ValueError("MONGODB_URI not found in environment variables")
        
        try:
            self.mongo_client = create_mongo_client(mongodb_uri)
        except Exception as e:
            raise ConnectionError(f"Failed to connect to MongoDB: {e}") from e

        db_name = resolve_mongo_db_name(mongodb_uri)
        self.db = self.mongo_client[db_name]
        self.scholars_collection = self.db.scholars
        print(f"[MongoDB] âœ… Connected to MongoDB")
        print(f"[MongoDB] Database: {self.db.name}")
        print(f"[MongoDB] Collection: scholars")
        
        # Pinecone connection
        self.vector_db = get_vector_db(index_name=INDEX_NAME, dimension=INDEX_DIMENSION)
        print(f"[Pinecone] âœ… Connected to index: {INDEX_NAME}")
        
        # LLM client (OpenAI or Anthropic)
        self.llm_provider = os.getenv("LLM_SUMMARY_PROVIDER", "openai")

        if self.llm_provider == "anthropic":
            from anthropic import Anthropic
            anthropic_key = os.getenv("ANTHROPIC_API_KEY")
            if not anthropic_key:
                raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
            self.anthropic_client = Anthropic(api_key=anthropic_key)
            self.openai_client = None
            print(f"[Anthropic] ✅ Initialized Anthropic client")
            self.summary_model = os.getenv("MONGO_SUMMARY_MODEL", "claude-sonnet-4-20250514")
        else:
            openai_key = os.getenv("OPENAI_API_KEY")
            if not openai_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables")
            try:
                http_client = httpx.Client(timeout=120.0)
                self.openai_client = OpenAI(api_key=openai_key, http_client=http_client)
                print(f"[OpenAI] ✅ Initialized OpenAI client")
            except Exception:
                self.openai_client = OpenAI(api_key=openai_key)
                print(f"[OpenAI] ✅ Initialized OpenAI client (fallback)")
            self.anthropic_client = None
            self.summary_model = os.getenv("MONGO_SUMMARY_MODEL", "gpt-4o-mini")
        self.summary_max_section_chars = max(2000, int(os.getenv("MONGO_SUMMARY_MAX_SECTION_CHARS", "12000")))
        self.summary_max_total_chars = max(8000, int(os.getenv("MONGO_SUMMARY_MAX_TOTAL_CHARS", "60000")))
    def get_all_professor_ids_from_pinecone(self) -> List[str]:
        """Get all unique professor_ids from Pinecone"""
        print("\n[Discovery] Fetching all unique professor IDs from Pinecone...")
        
        # Query with a dummy vector to get metadata
        # We'll use a filter to get all records, but Pinecone doesn't support "get all"
        # So we'll query with a zero vector and high top_k
        try:
            # Get index stats first
            stats = self.vector_db.index.describe_index_stats()
            total_vectors = stats.total_vector_count
            print(f"[Discovery] Total vectors in index: {total_vectors}")
            
            if total_vectors == 0:
                print("[Discovery] No vectors found in Pinecone")
                return []
            
            # Query with zero vector to get all (or as many as possible)
            # Use a large top_k, but be aware of limits
            top_k = min(10000, total_vectors)  # Pinecone limit is usually 10000
            
            query_response = self.vector_db.index.query(
                vector=[0.0] * INDEX_DIMENSION,
                top_k=top_k,
                include_metadata=True
            )
            
            # Extract unique professor_ids
            professor_ids = set()
            for match in query_response.matches:
                metadata = match.metadata
                prof_id = metadata.get('professor_id') or metadata.get('profile_id')
                if prof_id:
                    professor_ids.add(prof_id)
            
            print(f"[Discovery] Found {len(professor_ids)} unique professor profiles")
            return list(professor_ids)
            
        except Exception as e:
            print(f"[Discovery] Error fetching professor IDs: {str(e)}")
            raise
    
    def get_chunks_for_profile(self, professor_id: str) -> List[Dict]:
        """Get all chunks for a specific professor from Pinecone"""
        try:
            # Query with filter
            query_response = self.vector_db.index.query(
                vector=[0.0] * INDEX_DIMENSION,
                top_k=1000,  # Get all chunks for this profile
                include_metadata=True,
                filter={"professor_id": professor_id}
            )
            
            chunks = []
            for match in query_response.matches:
                metadata = match.metadata
                chunks.append({
                    "section": metadata.get("section", "Unknown"),
                    "text": metadata.get("text", ""),
                    "order": metadata.get("order", 0),
                    "chunk_id": metadata.get("chunk_id", "")
                })
            
            # Sort by section and order
            chunks.sort(key=lambda x: (x["section"], x["order"]))
            return chunks
            
        except Exception as e:
            print(f"[Error] Failed to get chunks for {professor_id}: {str(e)}")
            return []
    
    def aggregate_chunks_by_section(self, chunks: List[Dict]) -> Dict[str, str]:
        """Aggregate chunks by section"""
        sections = defaultdict(list)
        for chunk in chunks:
            section = chunk.get("section", "Unknown")
            sections[section].append(chunk.get("text", ""))
        
        # Combine text for each section
        aggregated = {}
        for section, texts in sections.items():
            aggregated[section] = "\n\n".join(texts)
        
        return aggregated
    
    def extract_name_parts(self, full_name: str) -> Dict[str, str]:
        """Extract name parts from full name"""
        if not full_name:
            return {"first": "", "middle": "", "last": "", "title": "", "suffix": ""}
        
        # Remove common titles
        title = ""
        for t in ["Dr.", "Dr", "Professor", "Prof."]:
            if full_name.startswith(t):
                title = t
                full_name = full_name.replace(t, "").strip()
                break
        
        # Split name
        parts = full_name.split()
        
        if len(parts) == 0:
            return {"first": "", "middle": "", "last": "", "title": title, "suffix": ""}
        elif len(parts) == 1:
            return {"first": parts[0], "middle": "", "last": "", "title": title, "suffix": ""}
        elif len(parts) == 2:
            return {"first": parts[0], "middle": "", "last": parts[1], "title": title, "suffix": ""}
        else:
            # Assume first, middle(s), last
            return {
                "first": parts[0],
                "middle": " ".join(parts[1:-1]),
                "last": parts[-1],
                "title": title,
                "suffix": ""
            }
    
    def generate_avatar_initial(self, name: Dict[str, str]) -> str:
        """Generate avatar initial from name"""
        if name.get("first"):
            return name["first"][0].upper()
        elif name.get("last"):
            return name["last"][0].upper()
        else:
            return "?"

    def _strip_unicode_noise(self, text: str) -> str:
        """Normalize Unicode text to ASCII-safe form and remove control characters."""
        if not text:
            return ""
        normalized = unicodedata.normalize("NFKD", text)
        ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
        ascii_text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", " ", ascii_text)
        return ascii_text

    def _sanitize_text_for_llm(self, text: str) -> str:
        """Keep content detail while removing non-printable noise and Unicode artifacts."""
        if not text:
            return ""
        cleaned = self._strip_unicode_noise(text)
        cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    def _sanitize_value(self, value: Any) -> Any:
        """Recursively sanitize LLM output payload."""
        if isinstance(value, str):
            return self._sanitize_text_for_llm(value)
        if isinstance(value, list):
            return [self._sanitize_value(v) for v in value]
        if isinstance(value, dict):
            return {k: self._sanitize_value(v) for k, v in value.items()}
        return value

    def _build_context_string(self, context: Dict[str, str]) -> str:
        """Build a large, section-labeled context string without aggressive truncation."""
        if not context:
            return ""

        context_parts: List[str] = []
        total_chars = 0

        for section_name in sorted(context.keys(), key=lambda s: str(s).lower()):
            raw_text = context.get(section_name, "")
            cleaned_text = self._sanitize_text_for_llm(raw_text)
            if not cleaned_text:
                continue

            section_text = cleaned_text[: self.summary_max_section_chars]
            remaining_budget = self.summary_max_total_chars - total_chars
            if remaining_budget <= 0:
                break
            if len(section_text) > remaining_budget:
                section_text = section_text[:remaining_budget]

            context_parts.append(f"[Section: {section_name}]\n{section_text}")
            total_chars += len(section_text)

        return "\n\n".join(context_parts)

    @staticmethod
    def _parse_int(value: Any, default: int = 0) -> int:
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return int(value)
        text = str(value).strip()
        if not text:
            return default
        match = re.search(r"\d+", text)
        if not match:
            return default
        try:
            return int(match.group(0))
        except Exception:
            return default

    def _build_section_prompt(self, section_name: str, context_string: str, professor_name: str) -> str:
        if section_name == "about":
            return f"""Extract rich, explanatory profile information for {professor_name} from the context below.

Rules:
- Use ONLY information explicitly present in the context.
- Do NOT invent, infer, or guess missing facts.
- Keep all concrete details (roles, institutions, departments, fields, locations, dates).
- Do NOT abstract into a short summary. Keep it explanatory and detailed.
- If a field is not present, return an empty string or empty array.

Return ONLY valid JSON with this schema:
{{
  "short_bio": "Detailed 3-6 sentence biography grounded in context",
  "detailed_bio": "Long-form explanatory biography preserving all key details from context",
  "current_position": "Current role/title exactly as available",
  "institution": "Institution/organization name",
  "department": "Department/program/school name if available",
  "field_of_study": "Primary field or specialization if available",
  "location": "Location if available",
  "source_evidence": ["verbatim snippet 1", "verbatim snippet 2", "verbatim snippet 3"]
}}

Context:
{context_string}"""

        if section_name == "background_and_work":
            return f"""Extract comprehensive background and work details for {professor_name} from the context below.

Rules:
- Use ONLY explicit evidence from context.
- No hallucinations, no assumptions, no guessed years.
- Keep details complete and explanatory, not brief.
- Preserve all meaningful facts.
- If unknown, use empty string / empty list / null.

Return ONLY valid JSON:
{{
  "background_summary": "Detailed narrative of background and career progression",
  "education_summary": [
    {{
      "degree": "Degree/program exactly as stated",
      "institution": "Institution name",
      "year": "Year or null",
      "brief": "Detailed description of this education entry",
      "evidence": "short verbatim snippet supporting this entry"
    }}
  ],
  "research_focus": ["All explicit research focus areas mentioned"],
  "current_work": "Detailed explanation of current work/research themes",
  "methodology": ["All explicit methods/approaches mentioned"],
  "career_history": [
    {{
      "position": "Role title",
      "institution": "Organization/institution",
      "years": "Date range or null",
      "details": "Detailed description with concrete facts",
      "evidence": "short verbatim snippet supporting this entry"
    }}
  ]
}}

Context:
{context_string}"""

        if section_name == "milestones":
            return f"""Extract all major milestones for {professor_name} from the context below.

Rules:
- Use only explicit context evidence.
- No fabrication. If year/type is unknown, set it to null or empty string.
- Include all meaningful milestones you can identify from context, ordered earliest-to-latest when possible.
- Use explanatory descriptions (2-5 sentences each), not short abstracts.

Return ONLY valid JSON:
{{
  "milestones": [
    {{
      "title": "Milestone title",
      "year": "Year or null",
      "type": "Fellowship|Award|Career|Publication|Service|Education|Leadership|Other",
      "description": "Detailed, explanatory description with concrete facts",
      "icon": "award|teaching|career|publication|service",
      "order": 1,
      "evidence": "short verbatim snippet supporting this milestone"
    }}
  ]
}}

Context:
{context_string}"""

        if section_name == "publications":
            return f"""Extract publication details for {professor_name} from the context below.

Rules:
- Use only explicit publication evidence in context.
- Do not invent titles, years, publishers, counts, or descriptions.
- Keep descriptions explanatory and fact-rich.
- Include all notable publications that are explicitly present.

Return ONLY valid JSON:
{{
  "featured_publications": [
    {{
      "title": "Publication title",
      "publisher": "Publisher or venue",
      "year": "Year or null",
      "type": "Book|Article|Chapter|Report|Other",
      "brief_description": "Detailed explanation of topic/significance grounded in context",
      "details": "Additional concrete details if available",
      "is_featured": true,
      "order": 1,
      "evidence": "short verbatim snippet supporting this publication"
    }}
  ],
  "other_publications": [
    {{
      "title": "Publication title",
      "publisher": "Publisher or venue",
      "year": "Year or null",
      "type": "Book|Article|Chapter|Report|Other",
      "evidence": "short verbatim snippet supporting this publication"
    }}
  ],
  "total_publications_count": "integer if explicitly available, else null"
}}

Context:
{context_string}"""

        return ""
    
    def generate_section_summary_with_llm(
        self, 
        section_name: str, 
        context: Dict[str, str],
        professor_name: str
    ) -> Dict[str, Any]:
        """Generate rich, grounded section output using full extracted context."""
        context_string = self._build_context_string(context)
        if not context_string:
            return {}

        prompt = self._build_section_prompt(section_name, context_string, professor_name)
        if not prompt:
            return {}
        
        system_msg = (
            "You are a strict academic-profile extraction engine.\n"
            "Use only the provided context. Never hallucinate, infer, or guess.\n"
            "Keep outputs detailed and explanatory, preserving concrete facts.\n"
            "If a field is missing, return empty string, null, or empty list.\n"
            "Return only valid JSON."
        )

        try:
            if self.llm_provider == "anthropic" and self.anthropic_client:
                response = self.anthropic_client.messages.create(
                    model=self.summary_model,
                    max_tokens=3500,
                    temperature=0.0,
                    system=system_msg,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = response.content[0].text
            else:
                response = self.openai_client.chat.completions.create(
                    model=self.summary_model,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    response_format={"type": "json_object"},
                    max_tokens=3500,
                )
                raw = response.choices[0].message.content

            # Extract JSON from response (handle markdown fencing)
            raw = raw.strip()
            if raw.startswith("```"):
                raw = re.sub(r'^```(?:json)?\s*', '', raw)
                raw = re.sub(r'\s*```$', '', raw)

            result = json.loads(raw)
            return self._sanitize_value(result)

        except Exception as e:
            print(f"[LLM] Error generating {section_name} summary: {str(e)}")
            return {}
    
    def create_scholar_document(
        self, 
        professor_id: str, 
        professor_name: str,
        chunks: List[Dict],
        aggregated_context: Dict[str, str]
    ) -> Dict[str, Any]:
        """Create MongoDB document for a scholar"""
        
        # Extract name parts
        name_parts = self.extract_name_parts(professor_name)
        
        # Generate summaries for each section
        print(f"  [LLM] Generating summaries...")
        
        about_data = self.generate_section_summary_with_llm("about", aggregated_context, professor_name)
        background_data = self.generate_section_summary_with_llm("background_and_work", aggregated_context, professor_name)
        milestones_data = self.generate_section_summary_with_llm("milestones", aggregated_context, professor_name)
        publications_data = self.generate_section_summary_with_llm("publications", aggregated_context, professor_name)
        
        # Get sections available
        sections_available = list(set(chunk.get("section", "Unknown") for chunk in chunks))
        
        # Build document
        document = {
            "_id": professor_id,
            "profile_id": professor_id,
            
            # Identity
            "name": {
                "full": professor_name,
                "display": f"{name_parts.get('title', '')} {professor_name}".strip(),
                "title": name_parts.get("title", ""),
                "first": name_parts.get("first", ""),
                "middle": name_parts.get("middle", ""),
                "last": name_parts.get("last", ""),
                "suffix": name_parts.get("suffix", "")
            },
            
            # About section
            "about": {
                "short_bio": about_data.get("short_bio", ""),
                "detailed_bio": about_data.get("detailed_bio", ""),
                "current_position": about_data.get("current_position", ""),
                "institution": about_data.get("institution", ""),
                "department": about_data.get("department", ""),
                "field_of_study": about_data.get("field_of_study", ""),
                "location": about_data.get("location", ""),
                "source_evidence": about_data.get("source_evidence", []),
                "avatar_url": None,  # To be set by admin
                "avatar_initial": self.generate_avatar_initial(name_parts)
            },
            
            # Background and work
            "background_and_work": {
                "background_summary": background_data.get("background_summary", ""),
                "education_summary": background_data.get("education_summary", []),
                "research_focus": background_data.get("research_focus", []),
                "current_work": background_data.get("current_work", ""),
                "methodology": background_data.get("methodology", []),
                "career_history": background_data.get("career_history", []),
            },
            
            # Milestones
            "milestones": milestones_data.get("milestones", []),
            
            # Publications
            "publications": {
                "featured_publications": publications_data.get("featured_publications", []),
                "other_publications": publications_data.get("other_publications", []),
                "total_publications_count": self._parse_int(
                    publications_data.get("total_publications_count"),
                    default=len(publications_data.get("featured_publications", [])),
                ),
                "show_more_link": True
            },
            
            # Links and media (empty, to be filled by admin)
            "links_and_media": {
                "social_profiles": [],
                "references": [],
                "featured_video": None
            },
            
            # Display metadata
            "display": {
                "avatar_initial": self.generate_avatar_initial(name_parts),
                "last_name_initial": name_parts.get("last", "?")[0].upper() if name_parts.get("last") else "?",
                "profile_image_url": None,
                "is_featured": False,
                "display_order": 0,
                "visibility": "draft",
                "last_updated": datetime.now(timezone.utc).isoformat()
            },
            
            # Search metadata
            "metadata": {
                "search_keywords": self.generate_search_keywords(professor_name, about_data, background_data),
                "tags": background_data.get("research_focus", []),
                "field_of_study": about_data.get("field_of_study", ""),
                "university": about_data.get("institution", ""),
                "last_name_initial": name_parts.get("last", "?")[0].upper() if name_parts.get("last") else "?"
            },
            
            # RAG context
            "rag_context": {
                "pinecone_indexed": True,
                "professor_id": professor_id,
                "chunk_count": len(chunks),
                "sections_available": sections_available,
                "last_indexed_at": datetime.now(timezone.utc).isoformat(),
                "source_data_hash": ""  # Can be calculated if needed
            },
            
            # Admin workflow
            "admin": {
                "created_by": "system",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_by": None,
                "updated_at": None,
                "last_reviewed_at": None,
                "review_status": "pending_curation",
                "curation_progress": {
                    "sections_completed": 0,
                    "sections_total": 5,
                    "completion_percentage": 0
                },
                "llm_generation_metadata": {
                    "model_used": self.summary_model,
                    "prompt_version": "v2.0_rich_grounded",
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "sections_generated": ["about", "background_and_work", "milestones", "publications"]
                }
            }
        }
        
        return document
    
    def generate_search_keywords(self, name: str, about_data: Dict, background_data: Dict) -> List[str]:
        """Generate search keywords"""
        keywords = []
        
        # Add name parts
        name_parts = name.split()
        keywords.extend(name_parts)
        
        # Add institution
        if about_data.get("institution"):
            keywords.append(about_data["institution"])
            # Add institution words
            keywords.extend(about_data["institution"].split())
        
        # Add field of study
        if about_data.get("field_of_study"):
            keywords.append(about_data["field_of_study"])
            keywords.extend(about_data["field_of_study"].split())
        
        # Add research focus
        if background_data.get("research_focus"):
            keywords.extend(background_data["research_focus"])
        
        # Remove duplicates and empty strings
        keywords = list(set([k.strip() for k in keywords if k.strip()]))
        
        return keywords[:20]  # Limit to 20 keywords
    
    def sync_profile(self, professor_id: str, professor_name: str) -> bool:
        """Sync a single profile from Pinecone to MongoDB"""
        try:
            # Get chunks from Pinecone
            chunks = self.get_chunks_for_profile(professor_id)
            
            if not chunks:
                print(f"  âš ï¸ No chunks found for {professor_id}")
                return False
            
            # Aggregate chunks by section
            aggregated_context = self.aggregate_chunks_by_section(chunks)
            
            # Create MongoDB document
            document = self.create_scholar_document(
                professor_id, 
                professor_name,
                chunks,
                aggregated_context
            )

            existing_doc = self.scholars_collection.find_one(
                {"profile_id": professor_id},
                {
                    "about.avatar_url": 1,
                    "display.profile_image_url": 1,
                },
            ) or {}
            existing_about = existing_doc.get("about") or {}
            existing_display = existing_doc.get("display") or {}
            existing_avatar_url = existing_about.get("avatar_url")
            existing_profile_image_url = existing_display.get("profile_image_url")
            if existing_avatar_url:
                document.setdefault("about", {})["avatar_url"] = existing_avatar_url
            if existing_profile_image_url:
                document.setdefault("display", {})["profile_image_url"] = existing_profile_image_url
            
            # Upsert to MongoDB
            self.scholars_collection.update_one(
                {"profile_id": professor_id},
                {"$set": document},
                upsert=True
            )
            
            return True
            
        except Exception as e:
            print(f"  âŒ Error syncing {professor_id}: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_professor_names_from_pinecone(self) -> Dict[str, str]:
        """Get professor names from Pinecone metadata"""
        print("\n[Discovery] Fetching professor names from Pinecone...")
        
        try:
            stats = self.vector_db.index.describe_index_stats()
            total_vectors = stats.total_vector_count
            
            if total_vectors == 0:
                return {}
            
            top_k = min(10000, total_vectors)
            query_response = self.vector_db.index.query(
                vector=[0.0] * INDEX_DIMENSION,
                top_k=top_k,
                include_metadata=True
            )
            
            # Map professor_id to professor_name
            professor_map = {}
            for match in query_response.matches:
                metadata = match.metadata
                prof_id = metadata.get('professor_id') or metadata.get('profile_id')
                prof_name = metadata.get('professor_name')
                
                if prof_id and prof_name:
                    # Keep the most common name if multiple chunks have different names
                    if prof_id not in professor_map:
                        professor_map[prof_id] = prof_name
                    # Could also use Counter to find most common name
            
            print(f"[Discovery] Found names for {len(professor_map)} profiles")
            return professor_map
            
        except Exception as e:
            print(f"[Discovery] Error fetching names: {str(e)}")
            return {}
    
    def sync_all_profiles(self):
        """Sync all profiles from Pinecone to MongoDB"""
        print("="*60)
        print("MongoDB Scholars Collection Sync")
        print("="*60)
        
        # Get professor IDs and names
        professor_ids = self.get_all_professor_ids_from_pinecone()
        professor_names = self.get_professor_names_from_pinecone()

        # Optional resume support via environment variable
        # If SYNC_START_INDEX is set (e.g. 599), skip the first N profiles
        start_index_env = os.getenv("SYNC_START_INDEX")
        if start_index_env is not None:
            try:
                start_index = int(start_index_env)
                if start_index > 0:
                    print(f"\n[Sync] Resuming from profile index {start_index}")
                    professor_ids = professor_ids[start_index:]
            except ValueError:
                print(f"\n[Sync] Invalid SYNC_START_INDEX value '{start_index_env}', ignoring")
        
        if not professor_ids:
            print("No profiles found to sync")
            return
        
        print(f"\n[Sync] Syncing {len(professor_ids)} profiles to MongoDB...")
        print(f"[Sync] Collection: scholars")
        
        successful = 0
        failed = 0
        
        for prof_id in tqdm(professor_ids, desc="Syncing profiles"):
            prof_name = professor_names.get(prof_id, "Unknown")
            
            if prof_name == "Unknown":
                print(f"\n  âš ï¸ No name found for {prof_id}, using 'Unknown'")
            
            print(f"\n[Profile] {prof_name} ({prof_id})")
            
            if self.sync_profile(prof_id, prof_name):
                successful += 1
                print(f"  âœ… Synced successfully")
            else:
                failed += 1
                print(f"  âŒ Failed to sync")
            
            # Small delay to avoid rate limiting
            time.sleep(0.5)
        
        # Summary
        print(f"\n{'='*60}")
        print(f"[Summary] Sync Complete!")
        print(f"[Summary] Successful: {successful}")
        print(f"[Summary] Failed: {failed}")
        print(f"[Summary] Total: {len(professor_ids)}")
        print(f"{'='*60}")
        
        # Verify MongoDB
        try:
            count = self.scholars_collection.count_documents({})
            print(f"\n[MongoDB] Total documents in 'scholars' collection: {count}")
        except Exception as e:
            print(f"\n[MongoDB] Error counting documents: {str(e)}")
    
    def create_indexes(self):
        """Create indexes for the scholars collection"""
        print("\n[Indexes] Creating indexes...")
        
        try:
            # Primary index
            self.scholars_collection.create_index("profile_id", unique=True)
            print("  âœ… Created index on profile_id")
            
            # Search indexes
            self.scholars_collection.create_index([
                ("metadata.search_keywords", "text"),
                ("name.full", "text"),
                ("about.institution", "text")
            ])
            print("  âœ… Created text search index")
            
            # Filtering indexes
            self.scholars_collection.create_index("display.last_name_initial")
            print("  âœ… Created index on last_name_initial")
            
            self.scholars_collection.create_index("metadata.university")
            print("  âœ… Created index on university")
            
            self.scholars_collection.create_index("admin.review_status")
            print("  âœ… Created index on review_status")
            
            print("[Indexes] All indexes created successfully")
            
        except Exception as e:
            print(f"[Indexes] Error creating indexes: {str(e)}")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Sync profiles from Pinecone to MongoDB'
    )
    parser.add_argument(
        '--profile-id',
        type=str,
        default=None,
        help='Sync only a specific profile by profile_id (e.g., 74a44fc4-7ef5-47dd-8386-664c596e8f6c)'
    )
    parser.add_argument(
        '--skip-indexes',
        action='store_true',
        help='Skip creating indexes (faster for single profile sync)'
    )
    
    args = parser.parse_args()
    
    try:
        sync = MongoDBScholarSync()
        
        # Create indexes first (unless skipped)
        if not args.skip_indexes:
            sync.create_indexes()
        
        # Sync single profile or all profiles
        if args.profile_id:
            print("="*60)
            print(f"MongoDB Scholars Collection Sync - Single Profile")
            print("="*60)
            print(f"[Profile ID] {args.profile_id}")
            
            # Get professor name from Pinecone
            professor_names = sync.get_professor_names_from_pinecone()
            professor_name = professor_names.get(args.profile_id, "Unknown")
            
            if professor_name == "Unknown":
                print(f"  âš ï¸ No name found for {args.profile_id} in Pinecone")
                print(f"  [Info] Will use 'Unknown' as name")
            
            print(f"[Profile Name] {professor_name}")
            print()
            
            if sync.sync_profile(args.profile_id, professor_name):
                print(f"\nâœ… Profile synced successfully!")
            else:
                print(f"\nâŒ Failed to sync profile")
        else:
            # Sync all profiles
            sync.sync_all_profiles()
        
        print("\nâœ… Sync process completed!")
        
    except Exception as e:
        print(f"\nâŒ Error in sync process: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

