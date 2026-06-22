"""
LLM-based text cleaning utility
Uses OpenAI to intelligently clean and normalize text content
"""
import os
import re
import unicodedata
from typing import Optional, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class LLMTextCleaner:
    """Clean text using LLM (OpenAI) for intelligent cleaning"""
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        # Initialize OpenAI client
        try:
            from openai import OpenAI
            import httpx
            
            # Create custom httpx client to avoid proxy issues
            try:
                http_client = httpx.Client(timeout=60.0)
                self.client = OpenAI(api_key=self.api_key, http_client=http_client)
            except Exception:
                # Fallback without custom http_client
                self.client = OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("openai package not installed. Install with: pip install openai")

        self.page_counter_re = re.compile(r"(?i)\bpage\s+\d+\s+of\s+\d+\b")
        self.compact_page_line_re = re.compile(r"(?i)^(?:\s*(?:page\s*)?\d+\s*(?:of|/)\s*\d+\s*){1,}$")
        self.reference_id_re = re.compile(r"(?i)\breference id\b[:\s-]*[a-z0-9-]{6,}")
        self.anti_bot_re = re.compile(
            r"(?i)(access to this page has been denied|verify you are human|captcha|cloudflare|"
            r"attention required|press\s*&\s*hold|press and hold|confirm you are a human|"
            r"checking your browser|proof of work|security check|not a bot)"
        )
        self.menu_noise_re = re.compile(
            r"(?i)\b(filter|sort|topic|program|center|publication type|view all|results per page|"
            r"apply filters|clear filters)\b"
        )

    def _strip_unicode_noise(self, text: str) -> str:
        """Normalize Unicode to ASCII-safe text and remove non-printable controls."""
        if not text:
            return ""
        normalized = unicodedata.normalize("NFKD", text)
        ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
        ascii_text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", " ", ascii_text)
        return ascii_text

    def _is_noise_line(self, line: str) -> bool:
        line = (line or "").strip()
        if not line:
            return True
        if self.anti_bot_re.search(line):
            return True
        if self.reference_id_re.search(line):
            return True
        page_hits = len(self.page_counter_re.findall(line))
        if page_hits >= 2 or self.compact_page_line_re.match(line):
            return True
        if self.menu_noise_re.search(line):
            alpha = sum(1 for ch in line if ch.isalpha())
            # Skip short UI-like menu rows while keeping full prose.
            if alpha < 120:
                return True
        digit_count = sum(1 for ch in line if ch.isdigit())
        alpha_count = sum(1 for ch in line if ch.isalpha())
        if digit_count >= 12 and alpha_count <= digit_count:
            return True
        return False

    def _regex_prune_noise(self, text: str) -> str:
        if not text:
            return ""
        text = self._strip_unicode_noise(text)
        out_lines: List[str] = []
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            if self._is_noise_line(line):
                continue
            out_lines.append(line)
        cleaned = "\n".join(out_lines)
        # Catch inline pagination blocks even when OCR flattens lines.
        cleaned = re.sub(r"(?i)(?:\bpage\s+\d+\s+of\s+\d+\b[\s,;:]*){2,}", " ", cleaned)
        cleaned = re.sub(r"\s{2,}", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()
    
    def _chunk_text(self, text: str, chunk_size: int = 3000, overlap: int = 200) -> List[str]:
        """
        Split text into chunks at paragraph boundaries with overlap
        
        Args:
            text: Text to chunk
            chunk_size: Target chunk size in characters
            overlap: Overlap between chunks in characters
        
        Returns:
            List of text chunks
        """
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        paragraphs = text.split('\n\n')
        current_chunk = []
        current_length = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            para_length = len(para)
            
            # If adding this paragraph would exceed chunk size, start a new chunk
            if current_length + para_length > chunk_size and current_chunk:
                # Save current chunk
                chunk_text = '\n\n'.join(current_chunk)
                chunks.append(chunk_text)
                
                # Start new chunk with overlap (last part of previous chunk)
                if overlap > 0 and len(chunk_text) > overlap:
                    overlap_text = chunk_text[-overlap:]
                    current_chunk = [overlap_text, para]
                    current_length = len(overlap_text) + para_length
                else:
                    current_chunk = [para]
                    current_length = para_length
            else:
                current_chunk.append(para)
                current_length += para_length + 2  # +2 for '\n\n'
        
        # Add remaining chunk
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return chunks
    
    def clean_text(self, text: str, max_length: int = 1_000_000, timeout: int = 30, use_chunking: bool = True) -> str:
        """
        Clean text using LLM to remove cookie notices, boilerplate, and normalize content.
        Supports chunking for long texts.

        Args:
            text: Raw text to clean
            max_length: Hard upper bound on input characters. Default is intentionally
                large so the cleaner does not silently amputate profile content; per-chunk
                sizing is handled by ``_chunk_text`` below.
            timeout: Timeout in seconds for API call per chunk
            use_chunking: Whether to split long texts into chunks

        Returns:
            Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""

        # Quick regex-based pruning before LLM to remove obvious junk and reduce token waste.
        text = self._regex_prune_noise(text)

        # Skip if text is too short
        if len(text.strip()) < 10:
            return text.strip()

        original_length = len(text)
        if len(text) > max_length:
            # Only the most extreme overflows are dropped at the tail; never inject a
            # "... [truncated]" sentinel that downstream raw_text mapping would treat
            # as real source content.
            text = text[:max_length]
        
        # Determine if chunking is needed
        chunk_size = 3000  # Characters per chunk (leaves room for prompt + response)
        needs_chunking = use_chunking and len(text) > chunk_size
        
        try:
            if needs_chunking:
                # Split into chunks and process each
                chunks = self._chunk_text(text, chunk_size=chunk_size, overlap=200)
                print(f"[LLMCleaner] Cleaning text in {len(chunks)} chunks ({original_length} chars)...")
                
                cleaned_chunks = []
                for i, chunk in enumerate(chunks):
                    try:
                        cleaned_chunk = self._clean_single_chunk(chunk, timeout=timeout)
                        if cleaned_chunk:
                            cleaned_chunks.append(cleaned_chunk)
                    except Exception as e:
                        print(f"[LLMCleaner] Error cleaning chunk {i+1}/{len(chunks)}: {str(e)}")
                        # Keep original chunk if cleaning fails
                        cleaned_chunks.append(chunk)
                
                # Combine cleaned chunks
                cleaned_text = '\n\n'.join(cleaned_chunks)
            else:
                # Process as single chunk
                print(f"[LLMCleaner] Cleaning text ({original_length} chars)...")
                cleaned_text = self._clean_single_chunk(text, timeout=timeout)

            cleaned_text = self._regex_prune_noise(cleaned_text)
            
            # Fallback: if LLM returns empty or very short, return original
            if len(cleaned_text) < len(text) * 0.1:  # If cleaned text is less than 10% of original
                print(f"[LLMCleaner] Warning: LLM returned very short text, using original")
                return text.strip()
            
            print(f"[LLMCleaner] ✅ Cleaned text: {original_length} -> {len(cleaned_text)} chars")
            return cleaned_text
            
        except Exception as e:
            print(f"[LLMCleaner] Error cleaning text with LLM: {str(e)}")
            # Fallback to basic cleaning
            return self._basic_clean(text)
    
    def _clean_single_chunk(self, text: str, timeout: int = 30) -> str:
        """
        Clean a single chunk of text using LLM
        
        Args:
            text: Text chunk to clean
            timeout: Timeout in seconds for API call
        
        Returns:
            Cleaned text chunk
        """
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a STRICT, FACT-PRESERVING text cleaner for biography/profile content. "
                        "Rules (must follow exactly):\n"
                        "1) You MAY rewrite for clarity, but you MUST preserve every factual detail from kept text.\n"
                        "2) ONLY remove clearly irrelevant boilerplate, navigation, footer/header blocks, "
                        "cookie/privacy/terms notices, social icon lists, menu lists, repeated site chrome, "
                        "tracking strings, and policy/legal sections.\n"
                        "3) NEVER delete substantive profile content (name, roles, dates, education, awards, "
                        "publications, research, biography, quotes, achievements, affiliations, contact info).\n"
                        "4) If unsure whether a line is relevant, KEEP it.\n"
                        "5) Remove raw HTML/XML/CSS/JS markup and inline style/attribute noise (e.g., tags like <h2>, "
                        "attributes like style=, class=, font-size, padding, margin, text-align). If markup wraps "
                        "meaningful text, keep only the visible text, not tag/attribute words.\n"
                        "6) Remove privacy/terms/consent/CCPA/California-resident notices and cookie category sections.\n"
                        "7) Remove prompt/instruction artifacts or dataset markers (e.g., 'CRITICAL RULES', "
                        "'Output format', 'Text segment to analyze', 'JSON output', '=== SEED URL ===', "
                        "'=== PROFILE PAGE ===', '=== WEBPAGE ===', 'top of page', 'bottom of page', 'back to top').\n"
                        "8) Remove base64/hashed/garbled strings or lines that are mostly non-word tokens/IDs.\n"
                        "9) Remove anti-bot/challenge text (e.g., 'Access to this page has been denied', "
                        "'Press & Hold to confirm you are a human', 'Reference ID ...').\n"
                        "10) Remove OCR pagination/index noise (e.g., repeated 'Page 216 of 850').\n"
                        "11) Do NOT summarize, abstract, or shorten the kept factual content.\n"
                        "12) Preserve paragraph readability; use double line breaks between paragraphs.\n"
                        "13) Output ONLY the cleaned text, no commentary, no JSON, no markdown.\n"
                    )
                },
                {
                    "role": "user",
                    "content": (
                        "Clean the text by removing only irrelevant boilerplate, markup, and noise. "
                        "Rewrite for clarity only when needed, while preserving all factual details:\n\n"
                        f"{text}"
                    )
                }
            ],
            temperature=0.0,
            max_tokens=4000,
            timeout=timeout
        )
        
        return self._regex_prune_noise(response.choices[0].message.content.strip())
    
    def _basic_clean(self, text: str) -> str:
        """Basic fallback cleaning if LLM fails"""
        # Remove markdown formatting
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Remove **bold**
        text = re.sub(r'__([^_]+)__', r'\1', text)  # Remove __bold__
        text = re.sub(r'\*([^*]+)\*', r'\1', text)  # Remove *italic*
        text = re.sub(r'_([^_]+)_', r'\1', text)  # Remove _italic_
        text = re.sub(r'#{1,6}\s+', '', text)  # Remove markdown headers
        
        # Convert line breaks to paragraph structure
        # Split by paragraph breaks, then join sentences within paragraphs
        paragraphs = text.split('\n\n')
        cleaned_paragraphs = []
        for para in paragraphs:
            para = para.strip()
            if para:
                # Within a paragraph, replace single \n with space
                para = re.sub(r'\n+', ' ', para)
                # Normalize spaces
                para = re.sub(r'\s+', ' ', para)
                cleaned_paragraphs.append(para)
        text = '\n\n'.join(cleaned_paragraphs)
        
        # Remove excessive line breaks
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Normalize whitespace
        text = re.sub(r'[ \t]+', ' ', text)
        # Remove common boilerplate patterns
        text = re.sub(r'(?i)(cookie|privacy|terms).{0,100}?(?:\n|$)', '', text)
        text = self._regex_prune_noise(text)
        return text.strip()
    
    def clean_document_content(self, content: str) -> str:
        """
        Clean document content (PDFs, Word docs, etc.)
        
        Args:
            content: Raw document text
        
        Returns:
            Cleaned document text
        """
        return self.clean_text(content)
    
    def clean_structured_content(self, content: dict) -> dict:
        """
        Clean structured content (full_text, headings, paragraphs)
        
        Args:
            content: Dictionary with 'full_text', 'headings', 'paragraphs' keys
        
        Returns:
            Cleaned content dictionary
        """
        cleaned = {}
        
        if 'full_text' in content:
            cleaned['full_text'] = self.clean_text(content['full_text'])
        
        if 'headings' in content and isinstance(content['headings'], list):
            cleaned_headings = []
            for heading in content['headings']:
                if isinstance(heading, str) and len(heading.strip()) > 2:
                    cleaned_heading = self.clean_text(heading, max_length=500)
                    if cleaned_heading:
                        cleaned_headings.append(cleaned_heading)
            cleaned['headings'] = cleaned_headings
        
        if 'paragraphs' in content and isinstance(content['paragraphs'], list):
            cleaned_paragraphs = []
            for para in content['paragraphs']:
                if isinstance(para, str) and len(para.strip()) > 10:
                    cleaned_para = self.clean_text(para, max_length=2000)
                    if cleaned_para:
                        cleaned_paragraphs.append(cleaned_para)
            cleaned['paragraphs'] = cleaned_paragraphs
        
        return cleaned


# Singleton instance
_llm_cleaner_instance = None

def get_llm_text_cleaner() -> Optional[LLMTextCleaner]:
    """Get or create LLMTextCleaner instance"""
    global _llm_cleaner_instance
    if _llm_cleaner_instance is None:
        try:
            _llm_cleaner_instance = LLMTextCleaner()
        except Exception as e:
            print(f"[LLMCleaner] Failed to initialize LLM cleaner: {str(e)}")
            print(f"[LLMCleaner] Falling back to regex-based cleaning")
            return None
    return _llm_cleaner_instance

