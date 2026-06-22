"""
Profile Chunking Pipeline
Processes large cleaned_text fields (10K-15K words) through a 3-stage pipeline
to produce section-aware chunks with correct overlap.
"""
import json
import os
import re
import uuid
import hashlib
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from urllib.parse import urlparse
from api.utils.source_guardrails import compute_text_hash

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional

try:
    from semantic_text_splitter import TextSplitter
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer
    import tiktoken
except ImportError:
    raise ImportError(
        "Required packages not installed. Run: pip install semantic-text-splitter sentence-transformers transformers tiktoken"
    )

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


_HEX_ID_RE = re.compile(r"^[A-Fa-f0-9]{8,}$")
_DOI_TAIL_RE = re.compile(r"^[0-9]{4,}([._-][0-9A-Za-z]+)*$")
_NUMERIC_ID_RE = re.compile(r"^[0-9]+$")


def _chunk_text_mentions(text: str, profile_name: str) -> bool:
    """Cheap last-ditch check that a chunk names the subject in some form."""
    if not text or not profile_name:
        return True
    text_lc = text.lower()
    name_lc = profile_name.strip().lower()
    if name_lc and name_lc in text_lc:
        return True
    parts = [p for p in name_lc.split() if len(p) >= 3]
    if not parts:
        return True
    last = parts[-1]
    if last and last in text_lc:
        return True
    return False


def _looks_like_id(value: str) -> bool:
    """True for strings that are obviously machine identifiers (DOI hashes,
    PBS asset IDs, numeric routing slugs) and so make terrible section
    labels for the chatbot to cite."""
    if not value:
        return True
    v = value.strip()
    if not v:
        return True
    if _NUMERIC_ID_RE.match(v):
        return True
    compact = v.replace(" ", "").replace("-", "").replace("_", "").replace(".", "")
    if _HEX_ID_RE.match(compact) and len(compact) >= 16:
        return True
    if _DOI_TAIL_RE.match(v):
        return True
    return False


@dataclass
class ChunkMetadata:
    """Metadata for a single chunk"""
    profile_id: str
    section: str
    chunk_id: str
    order: int
    text: str
    raw_text: str = ""
    source_id: str = ""
    text_hash: str = ""
    offset_start: int = -1
    offset_end: int = -1
    allowed_use: str = "facts_only"
    quote_ok: bool = False
    is_summary: bool = False
    language: str = "unknown"


class ProfileChunkingPipeline:
    """
    3-stage pipeline for processing large cleaned_text fields:
    1. Pre-segmentation (semantic split, no overlap)
    2. Section assignment (LLM per segment, no overlap)
    3. Global merge + final chunking (with overlap)
    """
    
    def __init__(
        self,
        output_dir: str = "output/chunked_profiles",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_provider: str = "openai",  # "ollama" or "openai"
        llm_model: str = "gpt-4o-mini",  # Forced model
        pre_segment_max_tokens: int = 2000,
        final_chunk_max_tokens: int = 200,
        final_chunk_overlap_tokens: int = 30,
        ollama_base_url: str = "http://localhost:11434"
    ):
        """
        Initialize the pipeline
        
        Args:
            output_dir: Directory to save chunked profiles
            embedding_model: Sentence transformer model for semantic splitting
            llm_provider: LLM provider ("ollama" or "openai")
            llm_model: Model name (e.g., "mistral:7b" for Ollama)
            pre_segment_max_tokens: Max tokens for pre-segmentation (3000-4000)
            final_chunk_max_tokens: Max tokens for final chunks (250-350)
            final_chunk_overlap_tokens: Overlap tokens for final chunks (40-60)
            ollama_base_url: Base URL for Ollama API
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.embedding_model_name = embedding_model
        self.llm_provider = llm_provider
        self.llm_model = "gpt-4o-mini"
        self.ollama_base_url = ollama_base_url
        
        # Token limits
        self.pre_segment_max_tokens = pre_segment_max_tokens
        self.final_chunk_max_tokens = final_chunk_max_tokens
        self.final_chunk_overlap_tokens = final_chunk_overlap_tokens
        # Large-section safeguards
        self.large_section_chars_threshold = int(os.getenv("LARGE_SECTION_CHARS_THRESHOLD", "15000"))
        self.large_section_timeout_seconds = int(os.getenv("LARGE_SECTION_TIMEOUT_SECONDS", "600"))
        self.force_simple_final_splitter = str(os.getenv("FORCE_SIMPLE_FINAL_SPLITTER", "1")).lower() in ("1", "true", "yes", "on")
        self.page_counter_re = re.compile(r"(?i)\bpage\s+\d+\s+of\s+\d+\b")
        self.reference_id_re = re.compile(r"(?i)\breference id\b[:\s-]*[a-z0-9-]{6,}")
        self.anti_bot_re = re.compile(
            r"(?i)(access to this page has been denied|verify you are human|captcha|cloudflare|"
            r"attention required|press\s*&\s*hold|press and hold|confirm you are a human|"
            r"checking your browser|proof of work|security check|not a bot)"
        )
        
        # Initialize models (lazy loading)
        self._embedding_model = None
        self._pre_segmenter = None
        self._final_segmenter = None
        
        logger.info(f"Pipeline initialized: output_dir={output_dir}, llm={llm_provider}:{llm_model}")

    def _is_noise_segment(self, segment_text: str) -> bool:
        """Detect segments dominated by anti-bot/pagination junk so we can skip them."""
        text = (segment_text or "").strip()
        if not text:
            return True
        lower = text.lower()
        if self.anti_bot_re.search(lower):
            return True
        if self.reference_id_re.search(lower):
            return True
        page_hits = len(self.page_counter_re.findall(lower))
        if page_hits >= 6:
            return True
        alpha_count = sum(1 for ch in text if ch.isalpha())
        digit_count = sum(1 for ch in text if ch.isdigit())
        if digit_count >= 80 and alpha_count <= digit_count:
            return True
        return False

    def _build_raw_token_index(self, raw_text: str) -> Optional[Dict[str, Any]]:
        """Build a lightweight token index for best-effort raw-text mapping."""
        if not raw_text:
            return None
        tokens = []
        spans = []
        for match in re.finditer(r"[A-Za-z0-9]+", raw_text):
            tokens.append(match.group(0).lower())
            spans.append((match.start(), match.end()))
        if not tokens:
            return None
        return {"text": raw_text, "tokens": tokens, "spans": spans}

    def _find_token_sequence(self, tokens: List[str], sequence: List[str], start_idx: int) -> Optional[int]:
        """Find the first occurrence of a token sequence starting at or after start_idx."""
        if not sequence:
            return None
        max_start = len(tokens) - len(sequence)
        first = sequence[0]
        for i in range(start_idx, max_start + 1):
            if tokens[i] != first:
                continue
            if tokens[i:i + len(sequence)] == sequence:
                return i
        return None

    def _extract_raw_chunk_text(
        self,
        cleaned_chunk_text: str,
        raw_index: Dict[str, Any],
        start_token_idx: int
    ) -> Tuple[str, int]:
        """
        Best-effort mapping of a cleaned chunk back to a raw-text substring.
        Uses token anchors to locate the chunk in the raw text.
        """
        if not cleaned_chunk_text or not raw_index:
            return "", start_token_idx

        raw_text = raw_index["text"]
        raw_tokens = raw_index["tokens"]
        raw_spans = raw_index["spans"]

        chunk_tokens = [m.group(0).lower() for m in re.finditer(r"[A-Za-z0-9]+", cleaned_chunk_text)]
        if not chunk_tokens:
            return "", start_token_idx

        anchor_len = min(6, len(chunk_tokens))
        anchor_pos = None
        for n in range(anchor_len, 1, -1):
            anchor = chunk_tokens[:n]
            anchor_pos = self._find_token_sequence(raw_tokens, anchor, start_token_idx)
            if anchor_pos is not None:
                anchor_len = n
                break

        if anchor_pos is None:
            return "", start_token_idx

        tail_anchor = chunk_tokens[-anchor_len:]
        tail_pos = self._find_token_sequence(raw_tokens, tail_anchor, anchor_pos + anchor_len)

        if tail_pos is not None:
            end_token_idx = tail_pos + anchor_len - 1
        else:
            end_token_idx = min(len(raw_tokens) - 1, anchor_pos + len(chunk_tokens) - 1)

        start_char = raw_spans[anchor_pos][0]
        end_char = raw_spans[end_token_idx][1]
        return raw_text[start_char:end_char].strip(), end_token_idx + 1
    
    @property
    def embedding_model(self) -> SentenceTransformer:
        """Lazy load embedding model"""
        if self._embedding_model is None:
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self._embedding_model = SentenceTransformer(self.embedding_model_name)
        return self._embedding_model
    
    def _create_tokenizer_wrapper(self, tokenizer):
        """Create a wrapper for the tokenizer that semantic-text-splitter can use"""
        class TokenizerWrapper:
            def __init__(self, tokenizer):
                self.tokenizer = tokenizer
            
            def encode(self, text: str):
                """Encode text to token IDs"""
                return self.tokenizer.encode(text, add_special_tokens=False)
            
            def decode(self, token_ids):
                """Decode token IDs to text"""
                return self.tokenizer.decode(token_ids, skip_special_tokens=True)
        
        return TokenizerWrapper(tokenizer)
    
    @property
    def pre_segmenter(self):
        """Lazy load pre-segmentation splitter (no overlap)"""
        if self._pre_segmenter is None:
            try:
                # Try using tiktoken for more reliable tokenization
                # Use cl100k_base encoding (used by GPT models, close enough for estimation)
                encoding = tiktoken.get_encoding("cl100k_base")
                self._pre_segmenter = TiktokenBasedSplitter(encoding, self.pre_segment_max_tokens, overlap=0)
                logger.debug(f"Created pre-segmenter using tiktoken with capacity: {self.pre_segment_max_tokens}")
            except Exception as e:
                logger.warning(f"Tiktoken-based splitting failed: {e}, using character-based fallback")
                # Fallback: Use character-based estimation (~4 chars per token)
                char_limit = self.pre_segment_max_tokens * 4
                self._pre_segmenter = CharacterBasedSplitter(char_limit, overlap=0)
        return self._pre_segmenter
    
    @property
    def final_segmenter(self):
        """Lazy load final chunking splitter (with overlap)"""
        if self._final_segmenter is None:
            try:
                # Use tiktoken for final chunking with overlap
                encoding = tiktoken.get_encoding("cl100k_base")
                self._final_segmenter = TiktokenBasedSplitter(
                    encoding, 
                    self.final_chunk_max_tokens, 
                    overlap=self.final_chunk_overlap_tokens
                )
                logger.debug(f"Created final segmenter using tiktoken with capacity: {self.final_chunk_max_tokens}, overlap: {self.final_chunk_overlap_tokens}")
            except Exception as e:
                logger.warning(f"Tiktoken-based splitting failed: {e}, using character-based fallback")
                char_limit = self.final_chunk_max_tokens * 4
                overlap_chars = self.final_chunk_overlap_tokens * 4
                self._final_segmenter = CharacterBasedSplitter(char_limit, overlap_chars)
        return self._final_segmenter
    
    def _generate_deterministic_uuid(self, seed: str) -> str:
        """Generate deterministic UUID from seed"""
        namespace = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')
        return str(uuid.uuid5(namespace, seed))

    def _simple_chunk_text(self, text: str, max_chars: int, overlap_chars: int) -> List[str]:
        """Simple character-based splitter with overlap (fallback)."""
        if not text:
            return []
        chunks: List[str] = []
        start = 0
        length = len(text)
        while start < length:
            end = min(start + max_chars, length)
            chunk = text[start:end]
            chunks.append(chunk)
            if end >= length:
                break
            start = max(0, end - overlap_chars)
        return chunks
    
    def pre_segment_text(self, cleaned_text: str) -> List[str]:
        """
        Stage 1: Pre-segmentation (semantic split, no overlap)
        
        Split cleaned_text into semantic segments of ~3000-4000 tokens.
        Uses semantic-text-splitter with sentence-transformers.
        
        Args:
            cleaned_text: Full cleaned text to segment
            
        Returns:
            List of pre-segmented text chunks (no overlap)
        """
        if not cleaned_text or not cleaned_text.strip():
            logger.warning("Empty cleaned_text provided")
            return []
        
        logger.info(f"Pre-segmenting text: {len(cleaned_text)} characters")
        
        # Use semantic splitter - chunks() returns an iterator
        try:
            segments = list(self.pre_segmenter.chunks(cleaned_text))
        except Exception as e:
            logger.error(f"Error in pre-segmentation: {e}")
            # Fallback: simple character-based split
            chunk_size = self.pre_segment_max_tokens * 4  # Rough estimate: 4 chars per token
            segments = [cleaned_text[i:i+chunk_size] for i in range(0, len(cleaned_text), chunk_size)]
        
        # Convert to list and filter empty segments
        segment_list = [seg.strip() for seg in segments if seg.strip()]
        
        logger.info(f"Pre-segmentation complete: {len(segment_list)} segments")
        for i, seg in enumerate(segment_list):
            logger.debug(f"Segment {i+1}: {len(seg)} chars")
        
        return segment_list
    
    def assign_sections(self, segment_text: str, segment_index: int) -> List[Dict[str, str]]:
        """
        Stage 2: Section assignment (LLM per segment, no overlap)
        
        Send segment to LLM for section classification.
        Output JSON structure: [{"section": "...", "content": "..."}]
        
        Args:
            segment_text: Single pre-segmented chunk
            segment_index: Index of segment (for logging)
            
        Returns:
            List of dicts with "section" and "content" keys
        """
        if not segment_text or not segment_text.strip():
            logger.warning(f"Empty segment {segment_index}")
            return []
        if self._is_noise_segment(segment_text):
            logger.warning(f"Segment {segment_index + 1}: Skipping noisy segment (anti-bot/pagination)")
            return []
        
        preview = segment_text[:200].replace("\n", " ")
        logger.info(f"Assigning sections to segment {segment_index + 1} ({len(segment_text)} chars) | preview: {preview!r}")
        
        # Store original segment text for validation
        original_segment_length = len(segment_text)
        
        # Build prompt for LLM
        prompt = self._build_section_prompt(segment_text)
        
        # Call LLM
        try:
            response = self._call_llm(prompt)
            section_data = self._parse_llm_response(response, segment_index)
            
            # Validate: Check if all content was preserved
            total_content_length = sum(len(item.get("content", "")) for item in section_data)
            if total_content_length < original_segment_length * 0.8:  # If we lost more than 20%
                logger.warning(f"Segment {segment_index + 1}: Content may be truncated. Original: {original_segment_length} chars, Returned: {total_content_length} chars")
                logger.warning(f"Segment {segment_index + 1}: Using fallback - preserving full original text")
                # Fallback: Use LLM classification but preserve original text
                # Try to use the section classification but replace content with original text
                if section_data:
                    # Use the section classification but replace content with original text
                    primary_section = section_data[0].get("section", "Misc")
                    logger.info(f"Segment {segment_index + 1}: Using section '{primary_section}' with full original text ({original_segment_length} chars)")
                    return [{"section": primary_section, "content": segment_text}]
                else:
                    return [{"section": "Misc", "content": segment_text}]
            
            logger.info(f"Segment {segment_index + 1}: Found {len(section_data)} sections, {total_content_length} chars preserved (original: {original_segment_length})")
            return section_data
            
        except Exception as e:
            logger.error(f"Error assigning sections to segment {segment_index + 1}: {e}")
            # Fallback: assign to "Misc" section with full original text
            return [{"section": "Misc", "content": segment_text}]
    
    def _build_section_prompt(self, segment_text: str) -> str:
        """Build prompt for LLM section classification"""
        prompt = f"""Analyze the following academic profile text segment and classify it into one or more sections.

CRITICAL RULES:
1. Only classify the text that is actually present in the segment
2. Do not create sections that don't exist in the text
3. Do not summarize, truncate, or omit any content - include ALL text exactly as provided
4. If text spans multiple sections, split it appropriately but preserve ALL content
5. Return ONLY valid JSON, no additional text
6. IMPORTANT: The "content" field must contain the COMPLETE, UNTRUNCATED text for that section

Output format (JSON array):
[
  {{"section": "SectionName", "content": "complete text content here - do not truncate"}},
  {{"section": "AnotherSection", "content": "complete more text here - do not truncate"}}
]

Text segment to analyze:
{segment_text}

JSON output (include ALL content, do not truncate):"""
        
        return prompt
    
    def _call_llm(self, prompt: str) -> str:
        """Call LLM (Ollama, OpenAI, or Anthropic)"""
        if self.llm_provider == "ollama":
            return self._call_ollama(prompt)
        elif self.llm_provider == "openai":
            return self._call_openai(prompt)
        elif self.llm_provider == "anthropic":
            return self._call_anthropic(prompt)
        else:
            raise ValueError(f"Unknown LLM provider: {self.llm_provider}")
    
    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API"""
        try:
            import requests
            
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": self.llm_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,  # Low temperature for classification
                        "top_p": 0.9
                    }
                },
                timeout=120
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "")
            
        except ImportError:
            raise ImportError("requests package required for Ollama. Install: pip install requests")
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            raise
    
    def _call_anthropic(self, prompt: str) -> str:
        """Call Anthropic Claude API"""
        try:
            from anthropic import Anthropic

            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError(
                    "ANTHROPIC_API_KEY not found in environment. "
                    "Set it in .env file or environment variables."
                )

            client = Anthropic(api_key=api_key)
            model_name = self.llm_model or "claude-sonnet-4-20250514"

            timeout_seconds = int(os.getenv("ANTHROPIC_TIMEOUT", "180"))
            max_retries = int(os.getenv("ANTHROPIC_MAX_RETRIES", "4"))
            base_backoff = float(os.getenv("ANTHROPIC_RETRY_BACKOFF", "1.5"))

            last_error = None
            for attempt in range(max_retries):
                try:
                    response = client.messages.create(
                        model=model_name,
                        max_tokens=16000,
                        temperature=0.1,
                        system="You are a helpful assistant that classifies academic profile text into sections. Always return valid JSON only. Include ALL content from the input text - do not truncate or summarize.",
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                    )
                    return response.content[0].text
                except Exception as e:
                    last_error = e
                    logger.warning(f"Anthropic error (attempt {attempt + 1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        sleep_seconds = base_backoff ** attempt
                        time.sleep(sleep_seconds)
                    else:
                        raise
            if last_error:
                raise last_error

        except ImportError:
            raise ImportError("anthropic package required. Install: pip install anthropic")
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise

    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API"""
        try:
            import openai
            from openai import OpenAI
            import httpx
            
            # Get API key from environment
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY not found in environment. "
                    "Set it in .env file or environment variables."
                )
            
            # Initialize OpenAI client
            # Use simple initialization to avoid proxy parameter conflicts
            # The newer OpenAI client doesn't accept 'proxies' as a parameter
            timeout_seconds = int(os.getenv("OPENAI_TIMEOUT", "180"))
            max_retries = int(os.getenv("OPENAI_MAX_RETRIES", "4"))
            base_backoff = float(os.getenv("OPENAI_RETRY_BACKOFF", "1.5"))
            try:
                import httpx
                # Create httpx client without proxy settings
                http_client = httpx.Client(timeout=httpx.Timeout(timeout_seconds))
                client = OpenAI(api_key=api_key, http_client=http_client)
            except (TypeError, AttributeError) as e:
                # If there's an issue with http_client (e.g., proxies parameter error),
                # fall back to simple initialization
                if 'proxies' in str(e).lower() or 'unexpected keyword' in str(e).lower():
                    logger.debug("Using simple OpenAI initialization to avoid proxy conflicts")
                    client = OpenAI(api_key=api_key)
                else:
                    raise
            except Exception as e:
                # Other errors - try simple initialization
                logger.warning(f"Failed to create custom http_client: {e}, using default")
                client = OpenAI(api_key=api_key)
            
            # Forced model name
            model_name = "gpt-4o-mini"
            
            # Calculate appropriate max_tokens based on input length
            # For gpt-4o-mini: max context is 128k tokens, but we need to leave room for response
            # Estimate: if input is long, we need more tokens for response
            # Use dynamic max_tokens: at least 4x the input length, but cap at 16000 (safe limit)
            import tiktoken
            try:
                encoding = tiktoken.encoding_for_model(model_name)
                input_tokens = len(encoding.encode(prompt))
                # Allocate enough tokens for response: input length + buffer, but cap at 16000
                dynamic_max_tokens = min(max(input_tokens * 2, 4000), 16000)
            except:
                # Fallback if tiktoken fails
                dynamic_max_tokens = 16000
            
            last_error = None
            for attempt in range(max_retries):
                try:
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that classifies academic profile text into sections. Always return valid JSON only. Include ALL content from the input text - do not truncate or summarize."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.1,
                        max_tokens=dynamic_max_tokens
                    )
                    return response.choices[0].message.content
                except Exception as e:
                    last_error = e
                    msg = str(e).lower()
                    if "timeout" in msg or "timed out" in msg or "request timed out" in msg:
                        logger.warning(f"OpenAI timeout (attempt {attempt + 1}/{max_retries})")
                    else:
                        logger.warning(f"OpenAI error (attempt {attempt + 1}/{max_retries}): {e}")

                    if attempt < max_retries - 1:
                        sleep_seconds = base_backoff ** attempt
                        time.sleep(sleep_seconds)
                    else:
                        raise
            if last_error:
                raise last_error
            
        except ImportError:
            raise ImportError("openai package required. Install: pip install openai")
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    def _extract_json_from_response(self, response: str) -> str:
        """Extract JSON from response, handling markdown and other formats"""
        response = response.strip()
        
        # Remove markdown code blocks
        if "```" in response:
            # Find JSON code block
            json_pattern = r'```(?:json)?\s*(\[.*?\])\s*```'
            match = re.search(json_pattern, response, re.DOTALL)
            if match:
                return match.group(1).strip()
            
            # Fallback: remove all code block markers
            lines = response.split("\n")
            cleaned_lines = []
            in_code_block = False
            for line in lines:
                if line.strip().startswith("```"):
                    in_code_block = not in_code_block
                    continue
                if not in_code_block:
                    cleaned_lines.append(line)
            response = "\n".join(cleaned_lines)
        
        # Try to find JSON array in the response
        # Look for JSON array pattern
        json_array_pattern = r'(\[[\s\S]*\])'
        match = re.search(json_array_pattern, response)
        if match:
            return match.group(1).strip()
        
        return response.strip()
    
    def _fix_json_errors(self, json_str: str) -> str:
        """Attempt to fix common JSON syntax errors"""
        # Fix unescaped newlines in strings (but not in actual JSON structure)
        # This is tricky - we need to be careful not to break valid JSON
        
        # Fix common issues:
        # 1. Unescaped quotes in strings (but be careful)
        # 2. Trailing commas
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)  # Remove trailing commas
        
        # Try to fix unterminated strings by finding the end of the content field
        # This is a heuristic approach
        try:
            # If we can find the pattern "content": "..." and the string is unterminated,
            # try to close it at a reasonable point
            pattern = r'"content"\s*:\s*"([^"]*(?:"[^,}\]]*)?)'
            def fix_content(match):
                content = match.group(1)
                # If the content doesn't end with a quote, try to find where it should end
                if not content.endswith('"'):
                    # Find the next }, ], or end of string
                    rest = json_str[match.end():]
                    # Look for the next structural element
                    next_brace = rest.find('}')
                    next_bracket = rest.find(']')
                    next_comma = rest.find(',')
                    
                    # Find the earliest structural break
                    breaks = [b for b in [next_brace, next_bracket, next_comma] if b >= 0]
                    if breaks:
                        end_pos = min(breaks)
                        # Extract up to that point and close the string
                        extracted = rest[:end_pos].rstrip()
                        # Remove any trailing structural chars
                        extracted = extracted.rstrip('},]')
                        return f'"content": "{content}{extracted}"'
                return match.group(0)
            
            # This is complex - let's try a simpler approach first
        except:
            pass
        
        return json_str
    
    def _parse_llm_response(self, response: str, segment_index: int) -> List[Dict[str, str]]:
        """Parse LLM response into section data with robust error handling"""
        original_response = response
        
        # Step 1: Extract JSON from response
        json_str = self._extract_json_from_response(response)
        
        # Step 2: Try to parse JSON
        data = None
        parse_attempts = [
            ("direct", json_str),
            ("fixed", self._fix_json_errors(json_str)),
        ]
        
        for attempt_name, attempt_json in parse_attempts:
            try:
                data = json.loads(attempt_json)
                if attempt_name != "direct":
                    logger.debug(f"Segment {segment_index + 1}: JSON parsed after {attempt_name} fix")
                break
            except json.JSONDecodeError as e:
                if attempt_name == "fixed":
                    logger.debug(f"Segment {segment_index + 1}: JSON fix attempt failed: {e}")
                continue
        
        # Step 3: If JSON parsing failed, try retry with stricter prompt
        if data is None:
            logger.warning(f"Segment {segment_index + 1}: JSON parsing failed, attempting retry with stricter prompt")
            try:
                retry_prompt = self._build_retry_prompt(original_response)
                retry_response = self._call_llm(retry_prompt)
                retry_json = self._extract_json_from_response(retry_response)
                data = json.loads(retry_json)
                logger.info(f"Segment {segment_index + 1}: JSON parsed successfully on retry")
            except Exception as retry_error:
                logger.error(f"Segment {segment_index + 1}: Retry also failed: {retry_error}")
                # Fallback: Try to extract sections using text-based parsing
                return self._parse_text_fallback(original_response, segment_index)
        
        # Step 4: Validate and process parsed data
        if data is None:
            return self._parse_text_fallback(original_response, segment_index)
        
        # Validate structure
        if not isinstance(data, list):
            logger.warning(f"Segment {segment_index + 1}: Expected list, got {type(data)}, trying fallback")
            return self._parse_text_fallback(original_response, segment_index)
        
        # Validate and filter sections
        valid_sections = []
        for item in data:
            if not isinstance(item, dict):
                continue
            
            section = (item.get("section", "") or "").strip()
            content = item.get("content", "").strip()
            
            if not section:
                section = "Misc"
            
            if content:
                valid_sections.append({
                    "section": section,
                    "content": content
                })
        
        if not valid_sections:
            logger.warning(f"Segment {segment_index + 1}: No valid sections found, trying fallback")
            return self._parse_text_fallback(original_response, segment_index)
        
        return valid_sections
    
    def _build_retry_prompt(self, failed_response: str) -> str:
        """Build a stricter prompt for retry when JSON parsing fails"""
        return f"""You must return ONLY valid JSON. No explanations, no markdown, no extra text.

The previous response had JSON parsing errors. Please return ONLY a valid JSON array in this exact format:

[
  {{"section": "SectionName", "content": "text content"}},
  {{"section": "AnotherSection", "content": "more text"}}
]

Rules:
1. Use double quotes for all strings
2. Escape all quotes inside content with backslash: \\"
3. No trailing commas
4. No comments
5. Return ONLY the JSON array, nothing else

Previous response (for reference - fix any errors):
{failed_response[:1000]}

Valid JSON output:"""
    
    def _parse_text_fallback(self, response: str, segment_index: int) -> List[Dict[str, str]]:
        """Fallback parser when JSON completely fails - extract sections from text"""
        logger.warning(f"Segment {segment_index + 1}: Using text-based fallback parser")
        
        # Try to extract section:content pairs from text
        sections = []
        
        # Look for patterns like "section": "content" or Section: content
        patterns = [
            r'"section"\s*:\s*"([^"]+)"\s*,\s*"content"\s*:\s*"([^"]+)"',
            r'"section"\s*:\s*"([^"]+)"[^}]*"content"\s*:\s*"([^"]+)"',
            r'Section:\s*([^\n]+)\nContent:\s*([^\n]+)',
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, response, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                section = match.group(1).strip()
                content = match.group(2).strip()
                
                if not section:
                    section = "Misc"
                
                if content and len(content) > 10:  # Minimum content length
                    sections.append({
                        "section": section,
                        "content": content
                    })
        
        if sections:
            logger.info(f"Segment {segment_index + 1}: Extracted {len(sections)} sections using fallback parser")
            return sections
        
        # Last resort: put everything in Misc
        logger.warning(f"Segment {segment_index + 1}: Could not extract sections, using Misc for entire segment")
        # Try to get some meaningful content from the response
        content = response.strip()
        if len(content) > 50:
            # Remove obvious JSON structure markers
            content = re.sub(r'[\[\]{}"]', '', content)
            content = re.sub(r'\s+', ' ', content).strip()
            if content:
                return [{"section": "Misc", "content": content[:5000]}]  # Limit content length
        
        return []
    
    def merge_sections(self, section_outputs: List[List[Dict[str, str]]]) -> Dict[str, List[str]]:
        """
        Stage 3: Global merge (non-LLM)
        
        Merge all segment-level JSON outputs into a single dictionary.
        Each section becomes a list of content strings.
        
        Args:
            section_outputs: List of segment outputs, each is List[Dict[str, str]]
            
        Returns:
            Dictionary mapping section names to lists of content strings
        """
        logger.info(f"Merging {len(section_outputs)} segment outputs")
        
        merged = {}
        
        for segment_idx, segment_output in enumerate(section_outputs):
            for item in segment_output:
                section = item.get("section", "Misc")
                content = item.get("content", "").strip()
                
                if not content:
                    continue
                
                if section not in merged:
                    merged[section] = []
                
                merged[section].append(content)
                logger.debug(f"Added to {section}: {len(content)} chars")
        
        # Log summary
        logger.info(f"Merged sections: {list(merged.keys())}")
        for section, contents in merged.items():
            logger.info(f"  {section}: {len(contents)} pieces, {sum(len(c) for c in contents)} total chars")
        
        return merged
    
    def chunk_sections(
        self,
        merged_sections: Dict[str, List[str]],
        profile_id: str,
        raw_index: Optional[Dict[str, Any]] = None
    ) -> Dict[str, List[ChunkMetadata]]:
        """
        Stage 4: Final chunking inside each section (with overlap)
        
        Chunk each merged section into smaller RAG-friendly chunks.
        Uses semantic-text-splitter with overlap.
        
        Args:
            merged_sections: Dictionary of section -> list of content strings
            profile_id: Profile ID for metadata
            
        Returns:
            Dictionary mapping section names to lists of ChunkMetadata
        """
        logger.info(f"Chunking {len(merged_sections)} sections for profile {profile_id}")
        
        chunked_sections = {}
        global_order = 0
        raw_search_start = 0
        
        for section, content_list in merged_sections.items():
            # Combine all content for this section
            combined_content = "\n\n".join(content_list)
            
            if not combined_content.strip():
                logger.warning(f"Empty section: {section}")
                chunked_sections[section] = []
                continue
            
            section_preview = combined_content[:200].replace("\n", " ")
            logger.info(
                f"Chunking section '{section}': {len(combined_content)} chars | preview: {section_preview!r}"
            )

            # Optionally force simple splitter for all sections
            if self.force_simple_final_splitter:
                max_chars = max(1000, self.final_chunk_max_tokens * 4)
                overlap_chars = max(100, self.final_chunk_overlap_tokens * 4)
                chunk_texts = self._simple_chunk_text(combined_content, max_chars, overlap_chars)
            # For large sections, skip semantic splitter entirely and use the fast fallback
            elif len(combined_content) >= self.large_section_chars_threshold:
                logger.warning(
                    f"Section '{section}' is large (>= {self.large_section_chars_threshold} chars); "
                    f"using simple splitter to avoid stalls"
                )
                max_chars = max(1000, self.final_chunk_max_tokens * 4)
                overlap_chars = max(100, self.final_chunk_overlap_tokens * 4)
                chunk_texts = self._simple_chunk_text(combined_content, max_chars, overlap_chars)
            else:
                chunks_iter = self.final_segmenter.chunks(combined_content)
                chunk_texts: List[str] = []
                timed_out = False
                start_time = time.time()

                for chunk_text in chunks_iter:
                    chunk_texts.append(chunk_text)
                    if (time.time() - start_time) > self.large_section_timeout_seconds:
                        timed_out = True
                        break

                if timed_out:
                    logger.warning(
                        f"Section '{section}' exceeded {self.large_section_timeout_seconds}s; "
                        f"falling back to simple splitter"
                    )
                    max_chars = max(1000, self.final_chunk_max_tokens * 4)
                    overlap_chars = max(100, self.final_chunk_overlap_tokens * 4)
                    chunk_texts = self._simple_chunk_text(combined_content, max_chars, overlap_chars)

            # Create chunk metadata
            section_chunks = []
            for chunk_idx, chunk_text in enumerate(chunk_texts):
                if not chunk_text.strip():
                    continue
                if chunk_idx > 0 and chunk_idx % 10 == 0:
                    logger.info(
                        f"Section '{section}': processed {chunk_idx} chunks so far (total chars: {len(combined_content)})"
                    )
                
                # Generate deterministic chunk ID
                chunk_seed = f"{profile_id}:{section}:{chunk_idx}"
                chunk_id = self._generate_deterministic_uuid(chunk_seed)
                
                raw_chunk_text = ""
                if raw_index:
                    raw_chunk_text, raw_search_start = self._extract_raw_chunk_text(
                        chunk_text,
                        raw_index,
                        raw_search_start
                    )

                chunk_meta = ChunkMetadata(
                    profile_id=profile_id,
                    section=section,
                    chunk_id=chunk_id,
                    order=global_order,
                    text=chunk_text.strip(),
                    raw_text=raw_chunk_text,
                    source_id="combined",
                    text_hash=compute_text_hash(chunk_text.strip()),
                    offset_start=-1,
                    offset_end=-1,
                    allowed_use="facts_only",
                    quote_ok=False,
                    is_summary=False,
                    language="unknown",
                )
                
                section_chunks.append(chunk_meta)
                global_order += 1
            
            chunked_sections[section] = section_chunks
            logger.info(f"Section '{section}': {len(section_chunks)} chunks created")
        
        total_chunks = sum(len(chunks) for chunks in chunked_sections.values())
        logger.info(f"Total chunks created: {total_chunks}")
        
        return chunked_sections
    
    def save_output(
        self,
        profile_id: str,
        structured_data: Dict[str, List[ChunkMetadata]],
        raw_text: Optional[str] = None,
        cleaned_text: Optional[str] = None
    ) -> Path:
        """
        Save chunked output to profile_id/chunks.json
        
        Args:
            profile_id: Profile ID
            structured_data: Dictionary of section -> list of ChunkMetadata
            
        Returns:
            Path to saved file
        """
        # Create profile directory
        profile_dir = self.output_dir / profile_id
        profile_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert ChunkMetadata to dict
        output_data = {
            "profile_id": profile_id,
            "sections": {}
        }

        if raw_text is not None:
            output_data["raw_text"] = raw_text
        if cleaned_text is not None:
            output_data["clean_text"] = cleaned_text
        
        for section, chunks in structured_data.items():
            output_data["sections"][section] = [asdict(chunk) for chunk in chunks]
        
        # Save to JSON
        output_file = profile_dir / "chunks.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved chunks to {output_file}")
        return output_file
    
    def build_chunks_from_source_chunks(
        self,
        *,
        profile_id: str,
        source_chunks: List[Dict[str, Any]],
        sources_meta: List[Dict[str, Any]],
        profile_name: str = "",
        raw_text: Optional[str] = None,
        cleaned_text: Optional[str] = None,
    ) -> Path:
        """Build the final ``chunks.json`` from per-source chunk records.

        Replaces the old "concatenate every source then re-section with the
        LLM" pipeline. The previous approach destroyed provenance
        (``source_id="combined"``, ``offset_start=-1``) and let the LLM
        re-section a giant blob — which is how Carter G. Woodson's *book*
        ended up filed under sections like "Quaker Restrictions" and Ida
        Stephens Owens' chunks were 80% co-honoree biographies.

        With per-source chunks supplied by the upstream guardrails layer,
        each chunk already carries its real ``source_id``, ``offset_start``,
        ``offset_end``, ``language``, ``allowed_use``, and ``quote_ok``.
        This builder simply groups them by a derived section label
        (source URL / link_text / source_type) and writes them in the
        existing ``chunks.json`` shape.
        """
        logger.info(
            f"Building chunks.json from {len(source_chunks)} per-source chunks "
            f"for profile {profile_id}"
        )

        sources_index: Dict[str, Dict[str, Any]] = {
            (s.get("source_id") or ""): s for s in (sources_meta or [])
        }

        sections: Dict[str, List[ChunkMetadata]] = {}
        order = 0
        seen_hashes: set = set()

        def _sort_key(c: Dict[str, Any]) -> Tuple[str, int]:
            sid = c.get("source_id", "") or ""
            try:
                off = int(c.get("offset_start", 0) if c.get("offset_start") is not None else 0)
            except (TypeError, ValueError):
                off = 0
            return (sid, off)

        # Stable iteration order so chunk `order` is deterministic
        # across runs even when sources arrive in different orders.
        ordered = sorted(source_chunks, key=_sort_key)

        # Defense-in-depth subject filter at the consolidator: any chunk
        # without ``subject_mention`` AND without an immediate
        # subject-mentioning neighbour in the same source has already
        # been pruned upstream, so anything that reaches here without a
        # subject_mention=True flag is suspicious. If `subject_mention`
        # is missing entirely (older source_chunks files), we leave the
        # chunk alone for back-compat.
        for ch in ordered:
            text = (ch.get("text") or "").strip()
            if not text:
                continue
            sm = ch.get("subject_mention")
            if sm is False and profile_name:
                # Re-check just in case upstream let it through.
                if not _chunk_text_mentions(text, profile_name):
                    continue
            text_hash = ch.get("text_hash") or compute_text_hash(text)
            if text_hash in seen_hashes:
                continue
            seen_hashes.add(text_hash)

            source_id = ch.get("source_id") or ""
            src_meta = sources_index.get(source_id, {})
            section_label = self._section_label_for_source(ch, src_meta)

            allowed_use = ch.get("allowed_use") or src_meta.get("allowed_use") or "facts_only"
            quote_ok = bool(allowed_use in ("short_quotes", "full_text"))

            # Mark as summary if the chunk is from a known summary/abstract
            # source (CV summary, biography lead-paragraph) — heuristic.
            is_summary = bool(
                src_meta.get("source_type") in ("biography_summary", "summary", "abstract")
            )

            # Preserve offsets faithfully (offset_start can legitimately be 0;
            # the previous ``or -1`` short-circuit would have wiped that out).
            raw_off_start = ch.get("offset_start", -1)
            raw_off_end = ch.get("offset_end", -1)
            try:
                offset_start = int(raw_off_start) if raw_off_start is not None else -1
            except (TypeError, ValueError):
                offset_start = -1
            try:
                offset_end = int(raw_off_end) if raw_off_end is not None else -1
            except (TypeError, ValueError):
                offset_end = -1

            chunk_meta = ChunkMetadata(
                profile_id=profile_id,
                section=section_label,
                chunk_id=ch.get("chunk_id") or f"{source_id}:{order}",
                order=order,
                text=text,
                raw_text=ch.get("raw_text") or "",
                source_id=source_id,
                text_hash=text_hash,
                offset_start=offset_start,
                offset_end=offset_end,
                allowed_use=allowed_use,
                quote_ok=quote_ok,
                is_summary=is_summary,
                language=ch.get("language") or "unknown",
            )
            sections.setdefault(section_label, []).append(chunk_meta)
            order += 1

        if not sections:
            logger.warning(
                f"No usable per-source chunks for profile {profile_id}; "
                "falling back to legacy combined-text chunker."
            )
            if cleaned_text:
                return self.process_profile(profile_id, cleaned_text, raw_text=raw_text)

        return self.save_output(
            profile_id,
            sections,
            raw_text=raw_text,
            cleaned_text=cleaned_text,
        )

    @staticmethod
    def _section_label_for_source(
        chunk: Dict[str, Any], src_meta: Dict[str, Any]
    ) -> str:
        """Pick a human-readable section label keyed off the source URL.

        Order of preference:
          1. ``link_text`` if present (the page's HTML <title> / scraper title)
          2. The first non-noise heading captured from chunk raw_text
          3. URL last-path-segment turned into a title (skipping numeric
             IDs / DOI hashes / trailing slashes)
          4. Domain-based label
          5. ``source_type``
        Two different scrapes of the same article collapse into one section
        because the chunks were already deduped by ``source_id``.
        """
        link_text = (src_meta.get("link_text") or "").strip()
        if link_text:
            # Trim trailing site suffixes like " | Site Name" or " - Site".
            for sep in (" | ", " — ", " – ", " - "):
                if sep in link_text:
                    head = link_text.split(sep, 1)[0].strip()
                    if head and len(head) >= 5:
                        link_text = head
                        break
            # Trim breadcrumb-style colon trails. IU directory pages
            # produce titles like
            #   "Kevin Brown: Directory: Faculty and staff: About us:
            #    Maurer School of Law: Indiana University Bloomington"
            # which become awful chatbot citations. If a colon-separated
            # tail is mostly short navigational fragments, keep only the
            # first segment.
            if link_text.count(":") >= 2:
                segs = [seg.strip() for seg in link_text.split(":") if seg.strip()]
                if len(segs) >= 3:
                    head = segs[0]
                    tail_segs = segs[1:]
                    tail_avg = sum(len(t) for t in tail_segs) / max(1, len(tail_segs))
                    if head and len(head) >= 4 and tail_avg <= 40:
                        link_text = head
            link_text = re.sub(r"\s+", " ", link_text).strip()
            if 5 <= len(link_text) <= 120 and not _looks_like_id(link_text):
                return link_text

        url = (
            src_meta.get("resolved_url")
            or src_meta.get("source_url")
            or chunk.get("source_url", "")
        )
        domain = ""
        slug = ""
        if url:
            try:
                parsed = urlparse(url)
                netloc = (parsed.netloc or "").lower()
                # Bug fix: ``lstrip("www.")`` was character-set stripping, so
                # ``woodson.as.virginia.edu`` became ``oodson.as.virginia.edu``.
                if netloc.startswith("www."):
                    netloc = netloc[4:]
                domain = netloc
                path = (parsed.path or "/").rstrip("/")
                segments = [seg for seg in path.split("/") if seg]
                # Walk segments from the right, skipping IDs/hashes/extensions
                # that aren't human-readable.
                for seg in reversed(segments):
                    cleaned = re.sub(r"\.(html?|pdf|aspx?|php|json|xml|htm)$", "", seg, flags=re.I)
                    cleaned = re.sub(r"[\-_]+", " ", cleaned).strip()
                    if cleaned and not _looks_like_id(cleaned):
                        slug = cleaned
                        break
            except Exception:
                pass

        if slug:
            label = f"{slug.title()} ({domain})" if domain else slug.title()
            return label[:120]
        if domain:
            return f"Source: {domain}"

        source_type = (src_meta.get("source_type") or chunk.get("source_type") or "Source").strip()
        return source_type.replace("_", " ").title() or "Source"

    def process_profile(self, profile_id: str, cleaned_text: str, raw_text: Optional[str] = None) -> Path:
        """
        Process a single profile through the complete pipeline

        Args:
            profile_id: Profile ID
            cleaned_text: Full cleaned text to process

        Returns:
            Path to saved chunks.json file
        """
        logger.info(f"Processing profile {profile_id}")
        logger.info(f"Input text: {len(cleaned_text)} characters")
        
        # Stage 1: Pre-segmentation
        segments = self.pre_segment_text(cleaned_text)
        
        if not segments:
            logger.warning(f"No segments created for profile {profile_id}")
            # Create empty output
            empty_data = {}
            return self.save_output(
                profile_id,
                empty_data,
                raw_text=raw_text,
                cleaned_text=cleaned_text
            )
        
        # Stage 2: Section assignment
        section_outputs = []
        for idx, segment in enumerate(segments):
            section_data = self.assign_sections(segment, idx)
            section_outputs.append(section_data)
        
        # Stage 3: Global merge
        merged_sections = self.merge_sections(section_outputs)
        
        # Stage 4: Final chunking
        raw_index = self._build_raw_token_index(raw_text) if raw_text else None
        chunked_sections = self.chunk_sections(merged_sections, profile_id, raw_index=raw_index)
        
        # Stage 5: Save output
        output_path = self.save_output(
            profile_id,
            chunked_sections,
            raw_text=raw_text,
            cleaned_text=cleaned_text
        )
        
        logger.info(f"Completed processing profile {profile_id}")
        return output_path


class TiktokenBasedSplitter:
    """Token-based splitter using tiktoken for accurate token counting"""
    
    def __init__(self, encoding, max_tokens: int, overlap: int = 0):
        self.encoding = encoding
        self.max_tokens = max_tokens
        self.overlap = overlap
    
    def chunks(self, text: str):
        """Split text into chunks based on token count"""
        if not text:
            return []
        
        # Encode entire text
        tokens = self.encoding.encode(text)
        
        if len(tokens) <= self.max_tokens:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(tokens):
            end = start + self.max_tokens
            
            # Try to break at sentence boundary (look for period, exclamation, question mark)
            if end < len(tokens):
                chunk_text = self.encoding.decode(tokens[start:end])
                # Find last sentence ending in the chunk
                for i in range(len(chunk_text) - 1, max(len(chunk_text) - 500, 0), -1):
                    if chunk_text[i] in '.!?\n':
                        # Re-encode to find the token position
                        partial_text = chunk_text[:i+1]
                        partial_tokens = self.encoding.encode(partial_text)
                        end = start + len(partial_tokens)
                        break
            
            # Decode chunk
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoding.decode(chunk_tokens).strip()
            
            if chunk_text:
                chunks.append(chunk_text)
            
            # Move start with overlap
            start = end - self.overlap if self.overlap > 0 else end
        
        return chunks


class CharacterBasedSplitter:
    """Fallback splitter using character-based estimation when tokenizer fails"""
    
    def __init__(self, max_chars: int, overlap: int = 0):
        self.max_chars = max_chars
        self.overlap = overlap
    
    def chunks(self, text: str):
        """Split text into chunks based on character count"""
        if not text:
            return []
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.max_chars
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings near the boundary
                for i in range(end, max(start + self.max_chars - 200, start), -1):
                    if i < len(text) and text[i] in '.!?\n':
                        end = i + 1
                        break
                else:
                    # Look for paragraph breaks
                    for i in range(end, max(start + self.max_chars - 100, start), -1):
                        if i < len(text) and text[i] == '\n':
                            end = i + 1
                            break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - self.overlap if self.overlap > 0 else end
        
        return chunks


# Example usage
if __name__ == "__main__":
    # Example: Process a profile
    pipeline = ProfileChunkingPipeline(
        output_dir="output/chunked_profiles",
        llm_provider="ollama",
        llm_model="mistral:7b"
    )
    
    # Example cleaned_text (replace with actual data)
    example_text = """
    Dr. Jane Smith is a Professor of Computer Science at Example University.
    She received her PhD from MIT in 2010.
    
    EDUCATION:
    - PhD in Computer Science, MIT, 2010
    - MS in Computer Science, Stanford, 2006
    - BS in Mathematics, UC Berkeley, 2004
    
    RESEARCH INTERESTS:
    Her research focuses on machine learning and natural language processing.
    She has published over 50 papers in top-tier conferences.
    
    PUBLICATIONS:
    1. Smith, J. (2023). "Advanced NLP Techniques". ACL.
    2. Smith, J. (2022). "ML for Text Analysis". NeurIPS.
    """
    
    profile_id = "example-profile-123"
    pipeline.process_profile(profile_id, example_text, raw_text=example_text)
