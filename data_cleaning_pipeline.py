import re
import unicodedata
import json
import uuid
import os
from typing import Dict, List, Optional
from nltk.tokenize import sent_tokenize
import nltk

# Download required NLTK data if not available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt', quiet=True)


class DataCleaningPipeline:
    
    def __init__(
        self, 
        target_words_per_chunk: int = 325, 
        min_words_per_chunk: int = 250, 
        max_words_per_chunk: int = 400,
        use_llm_cleaning: bool = False,
        llm_provider: str = "openai",  # "openai" or "ollama"
        llm_model: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        ollama_host: Optional[str] = None
    ):
        
        self.target_words_per_chunk = target_words_per_chunk
        self.min_words_per_chunk = min_words_per_chunk
        self.max_words_per_chunk = max_words_per_chunk
        # Keep max_tokens_per_chunk for backward compatibility (used in old chunk_text method)
        self.max_tokens_per_chunk = max_words_per_chunk
        
        # LLM cleaning configuration
        self.use_llm_cleaning = use_llm_cleaning
        self.llm_provider = llm_provider.lower()
        self.llm_model = llm_model
        self.llm_api_key = llm_api_key
        self.ollama_host = ollama_host or os.getenv('OLLAMA_HOST', 'http://localhost:11434')
        
        # Initialize LLM client if enabled
        self._llm_client = None
        if self.use_llm_cleaning:
            self._init_llm_client()
        
        # Section headers to detect
        self.SECTION_HEADERS = [
            "RESEARCH INTERESTS",
            "RESEARCH EXPERIENCE",
            "EDUCATION",
            "AWARDS", "HONORS", "AWARDS AND HONORS",
            "TEACHING EXPERIENCE",
            "INDUSTRY EXPERIENCE",
            "PUBLICATIONS", "PUBLICATIONS AND PRESENTATIONS", "JOURNALS", "CONFERENCES",
            "RESEARCH PROJECTS",
            "TECHNICAL REPORTS",
            "BOOKS",
            "PROFESSIONAL ACTIVITIES", "PROFESSIONAL ACTIVITES",  # Handle typo
            "EDITORIAL BOARD",
            "LANGUAGES",
            "BIO", "BIOGRAPHY", "ABOUT",
            "EXPERIENCE",
            "SKILLS",
            "CONTACT",
        ]
        self.page_counter_re = re.compile(r"(?i)\bpage\s+\d+\s+of\s+\d+\b")
        self.compact_page_line_re = re.compile(r"(?i)^(?:\s*(?:page\s*)?\d+\s*(?:of|/)\s*\d+\s*){1,}$")
        self.reference_id_re = re.compile(r"(?i)\breference id\b[:\s-]*[a-z0-9-]{6,}")
        self.anti_bot_re = re.compile(
            r"(?i)(access to this page has been denied|verify you are human|captcha|cloudflare|"
            r"attention required|press\s*&\s*hold|press and hold|confirm you are a human|"
            r"checking your browser|proof of work|security check|not a bot)"
        )

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
        digit_count = sum(1 for ch in line if ch.isdigit())
        alpha_count = sum(1 for ch in line if ch.isalpha())
        if digit_count >= 12 and alpha_count <= digit_count:
            return True
        return False
    
    def normalize_text(self, text: str) -> str:
        """
        Step 1: Normalize raw text
        Remove noise so later chunking doesn't break
        
        Args:
            text: Raw text to normalize
            
        Returns:
            Normalized text
        """
        if not text:
            return ""
        
        # Unicode normalization
        text = unicodedata.normalize("NFKD", text)
        
        # Replace \n sequences with a single newline (preserve newlines for section detection)
        text = re.sub(r'\n+', '\n', text)
        
        # Remove leftover escape characters or weird sequences (but preserve newlines)
        text = re.sub(r'[^\S\r\n]+', ' ', text)   # collapse spaces (but not newlines)
        text = re.sub(r'[\t\r]', ' ', text)  # Replace tabs and carriage returns with spaces
        
        # Remove HTML leftovers if any
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'&[a-zA-Z]+;', '', text)
        
        # Remove smart quotes and em-dashes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        text = text.replace('—', '-').replace('–', '-')
        
        # IMPORTANT: Do NOT remove backslashes/forward slashes or collapse newlines here
        # This would break section detection. We'll clean symbols later in chunk text only.
        # Normalize multiple spaces (but preserve newlines)
        text = re.sub(r'[ ]+', ' ', text)  # Only collapse regular spaces, not newlines
        
        # Trim
        return text.strip()
    
    def split_into_sections(self, text: str) -> Dict[str, str]:
        """
        Step 2: Extract sections
        Split text into logical sections based on headers
        
        Args:
            text: Normalized text
            
        Returns:
            Dictionary mapping section headers to content
        """
        sections = {}
        
        # First, try to match known section headers (more accurate)
        # Match headers that are on their own line, optionally followed by colon
        # Pattern: start of line, header text, optional colon, whitespace/newline
        matches = []
        
        # Build pattern from known section headers
        # Sort by length (longest first) to match "RESEARCH INTERESTS" before "RESEARCH"
        sorted_headers = sorted(self.SECTION_HEADERS, key=len, reverse=True)
        
        for header in sorted_headers:
            # Escape special regex characters in header
            escaped_header = re.escape(header)
            # Match header at start of line or after newline, case-insensitive
            # Require it to be followed by colon or newline (not part of a word)
            pattern = rf"(?i)(?:^|\n)\s*({escaped_header})\s*:?\s*\n"
            header_matches = list(re.finditer(pattern, text, re.MULTILINE))
            matches.extend([(m.start(), m.end(), header.upper()) for m in header_matches])
        
        # Normalize section names (map variations to standard names)
        section_normalization = {
            "PUBLICATIONS AND PRESENTATIONS": "PUBLICATIONS",
            "AWARDS AND HONORS": "AWARDS",
            "PROFESSIONAL ACTIVITES": "PROFESSIONAL ACTIVITIES",  # Common typo
        }
        
        if matches:
            # Sort matches by position and remove duplicates
            matches = sorted(set(matches), key=lambda x: x[0])
            
            # Remove overlapping matches (keep the first one)
            filtered_matches = []
            for i, (start, end, header) in enumerate(matches):
                if i == 0 or start >= filtered_matches[-1][1]:
                    filtered_matches.append((start, end, header))
            
            # Extract sections
            for i, (start, end, header) in enumerate(filtered_matches):
                section_start = end
                section_end = filtered_matches[i+1][0] if i+1 < len(filtered_matches) else len(text)
                
                content = text[section_start:section_end].strip()
                if content:
                    # Normalize section name
                    normalized_header = section_normalization.get(header.upper(), header.upper())
                    sections[normalized_header] = content
        else:
            # No sections found, treat entire text as one section
            sections["CONTENT"] = text
        
        return sections
    
    def clean_section(self, text: str) -> str:
        """
        Step 3: Clean each section more deeply
        Remove bullet points, normalize dashes, remove duplicates
        
        Args:
            text: Section text to clean
            
        Returns:
            Cleaned section text
        """
        if not text:
            return ""
        
        # Remove bullet points (various unicode bullets)
        text = re.sub(r'[•·▪▫○●◘◙‣⁃∙⋅・◦▪■□]', '', text)
        
        # Remove leftover symbols like \t
        text = text.replace('\t', ' ')
        
        # Normalize dashes
        text = text.replace(" - ", ": ")
        text = text.replace(" — ", ": ")
        text = text.replace(" – ", ": ")
        
        # Remove repeated spaces
        text = re.sub(r'\s{2,}', ' ', text)
        
        # Remove PDF artifacts (common patterns)
        text = re.sub(r'\[PDF\]', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\(PDF\)', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\f', '\n', text)  # Form feed
        
        # Remove backslashes and forward slashes (replace with spaces)
        text = text.replace('\\', ' ')
        text = text.replace('/', ' ')
        
        # Remove duplicated words or titles (simple heuristic)
        lines = text.split('\n')
        cleaned_lines = []
        seen_lines = set()
        
        for line in lines:
            line_stripped = line.strip()
            if self._is_noise_line(line_stripped):
                continue
            
            # Clean symbols from each line individually
            line_stripped = line_stripped.replace('\\', ' ').replace('/', ' ')
            line_stripped = line_stripped.replace('\n', ' ').replace('\r', ' ')
            line_stripped = re.sub(r'\s+', ' ', line_stripped).strip()
            
            line_lower = line_stripped.lower()
            
            # Skip empty lines
            if not line_stripped:
                cleaned_lines.append('')
                continue
            
            # Skip duplicate lines (exact match)
            if line_lower not in seen_lines:
                seen_lines.add(line_lower)
                cleaned_lines.append(line_stripped)
        
        text = '\n'.join(cleaned_lines)
        
        # Final cleanup - remove any remaining symbols
        text = text.replace('\\', ' ').replace('/', ' ')
        text = re.sub(r'\n{3,}', '\n\n', text)  # Multiple blank lines to double
        text = re.sub(r'\s+', ' ', text)  # Normalize all whitespace
        text = text.strip()
        
        return text
    
    def micro_split(self, section: str) -> List[str]:
        """
        Step 4: Create micro-sections
        Split large sections into smaller, meaningful blocks
        
        Args:
            section: Section text to split
            
        Returns:
            List of micro-section texts
        """
        if not section:
            return []
        
        # Split by blank lines, bullets, or numbered items
        # Pattern: blank line, bullet point, or numbered item (1. 2. etc.)
        items = re.split(r"\n\s*\n|•|\d+\.\s+", section)
        
        # Filter out very short items
        micro_sections = [item.strip() for item in items if len(item.strip()) > 50]
        
        # If no good splits found, try splitting by sentences
        if len(micro_sections) < 2:
            # Split by periods followed by capital letters (sentence boundaries)
            sentences = re.split(r'\.\s+(?=[A-Z])', section)
            micro_sections = []
            for s in sentences:
                s_cleaned = s.strip()
                # Clean symbols from each micro-section
                s_cleaned = s_cleaned.replace('\\', ' ').replace('/', ' ')
                s_cleaned = s_cleaned.replace('\n', ' ').replace('\r', ' ')
                s_cleaned = re.sub(r'\s+', ' ', s_cleaned).strip()
                if len(s_cleaned) > 50:
                    if not s_cleaned.endswith('.'):
                        s_cleaned += '.'
                    micro_sections.append(s_cleaned)
        
        # Clean the final micro_sections and fallback section
        cleaned_micro_sections = []
        for ms in micro_sections:
            ms_cleaned = ms.replace('\\', ' ').replace('/', ' ')
            ms_cleaned = ms_cleaned.replace('\n', ' ').replace('\r', ' ')
            ms_cleaned = re.sub(r'\s+', ' ', ms_cleaned).strip()
            if ms_cleaned:
                cleaned_micro_sections.append(ms_cleaned)
        
        if cleaned_micro_sections:
            return cleaned_micro_sections
        
        # Fallback: clean the original section
        section_cleaned = section.replace('\\', ' ').replace('/', ' ')
        section_cleaned = section_cleaned.replace('\n', ' ').replace('\r', ' ')
        section_cleaned = re.sub(r'\s+', ' ', section_cleaned).strip()
        return [section_cleaned] if section_cleaned else []
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Step 5: Semantic chunking
        Split text into semantically coherent chunks
        
        Args:
            text: Text to chunk
            
        Returns:
            List of chunk texts
        """
        if not text:
            return []
        
        try:
            # Use NLTK sentence tokenizer
            sentences = sent_tokenize(text)
        except Exception as e:
            print(f"Warning: NLTK tokenization failed: {e}, using regex fallback")
            # Fallback to regex-based sentence splitting
            sentences = re.split(r'[.!?]+\s+', text)
            sentences = [s.strip() + '.' if s.strip() and not s.strip().endswith(('.', '!', '?')) else s.strip() 
                        for s in sentences if s.strip()]
        
        if not sentences:
            return [text] if text.strip() else []
        
        chunks = []
        buffer = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Estimate tokens (rough: 1 token ≈ 4 characters)
            buffer_tokens = len(buffer.split()) if buffer else 0
            sentence_tokens = len(sentence.split())
            
            if buffer_tokens + sentence_tokens < self.max_tokens_per_chunk:
                buffer += " " + sentence if buffer else sentence
            else:
                if buffer:
                    chunks.append(buffer.strip())
                buffer = sentence
        
        # Add remaining buffer
        if buffer:
            chunks.append(buffer.strip())
        
        return chunks

    def _sequential_chunk_text(self, text: str, profile_url: str = "", section_label: str = "CONTENT") -> List[Dict]:
        """
        Sequential chunking with overlap to preserve context.
        Uses a fixed window (target words) with ~15% overlap (bounded to 10-20%).
        """
        if not text or not text.strip():
            return []

        # Clean aggressively before splitting
        cleaned_text = self.clean_section(text)
        cleaned_text = self._final_cleanup(cleaned_text)
        words = cleaned_text.split()
        if not words:
            return []

        chunk_size = min(self.max_words_per_chunk, max(self.target_words_per_chunk, self.min_words_per_chunk))
        # Aim for 15% overlap, clamp to 10-20% and at least 1 word
        overlap = max(1, int(chunk_size * 0.15))
        overlap = min(overlap, max(1, int(chunk_size * 0.20)))
        step = max(1, chunk_size - overlap)

        chunks: List[Dict] = []
        idx = 0

        while idx < len(words):
            chunk_words = words[idx: idx + chunk_size]
            if not chunk_words:
                break

            chunk_text = ' '.join(chunk_words)
            chunk_text = self._final_cleanup(chunk_text)
            if not chunk_text:
                idx += step
                continue

            word_count = len(chunk_text.split())

            # Merge very small trailing chunks into the previous one when possible
            if word_count < self.min_words_per_chunk and chunks:
                prev_text = chunks[-1]["text"] + " " + chunk_text
                prev_words = len(prev_text.split())
                if prev_words <= self.max_words_per_chunk:
                    prev_text = self._final_cleanup(prev_text)
                    chunks[-1]["text"] = prev_text
                    chunks[-1]["metadata"]["length"] = len(prev_text.split())
                    chunks[-1]["metadata"]["char_count"] = len(prev_text)
                    idx += step
                    continue

            final_text = chunk_text
            if self.use_llm_cleaning:
                try:
                    final_text = self._clean_chunk_with_llm(final_text, section_name=section_label)
                    word_count = len(final_text.split())
                except Exception:
                    final_text = chunk_text  # fallback on any LLM issue

            chunks.append({
                "id": str(uuid.uuid4()),
                "text": final_text,
                "metadata": {
                    "source_profile": profile_url,
                    "section": section_label or "CONTENT",
                    "length": word_count,
                    "char_count": len(final_text),
                    "chunk_index": len(chunks)
                }
            })

            idx += step

        return chunks
    
    def combine_chunks_by_section(self, section_text: str, section_header: str) -> List[str]:
        """
        Combine section content into optimal-sized chunks (250-400 words, target 325)
        
        Args:
            section_text: Cleaned section text
            section_header: Section header name
            
        Returns:
            List of chunk texts with optimal word counts
        """
        if not section_text or not section_text.strip():
            return []
        
        # Split into sentences for semantic chunking
        try:
            sentences = sent_tokenize(section_text)
        except Exception:
            # Fallback to regex-based sentence splitting
            sentences = re.split(r'[.!?]+\s+', section_text)
            sentences = [s.strip() + '.' if s.strip() and not s.strip().endswith(('.', '!', '?')) else s.strip() 
                        for s in sentences if s.strip()]
        
        if not sentences:
            return [section_text] if section_text.strip() else []
        
        chunks = []
        current_chunk = []
        current_word_count = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Clean symbols from sentence before processing
            sentence = sentence.replace('\\', ' ').replace('/', ' ')
            sentence = sentence.replace('\n', ' ').replace('\r', ' ')
            sentence = re.sub(r'\s+', ' ', sentence).strip()
            
            if not sentence:
                continue
            
            sentence_words = len(sentence.split())
            
            # If adding this sentence would exceed max, finalize current chunk
            if current_word_count + sentence_words > self.max_words_per_chunk and current_chunk:
                # Create chunk and remove newlines and symbols
                chunk_text = ' '.join(current_chunk).strip()
                chunk_text = chunk_text.replace('\n', ' ').replace('\r', ' ')
                chunk_text = chunk_text.replace('\\', ' ').replace('/', ' ')
                # Normalize multiple spaces
                chunk_text = re.sub(r'\s+', ' ', chunk_text)
                if chunk_text:
                    chunks.append(chunk_text)
                
                # Start new chunk with current sentence
                current_chunk = [sentence]
                current_word_count = sentence_words
            
            # If we're below target, keep adding sentences
            elif current_word_count + sentence_words < self.target_words_per_chunk:
                current_chunk.append(sentence)
                current_word_count += sentence_words
            
            # If we're between target and max, check if we should finalize
            elif current_word_count >= self.min_words_per_chunk:
                # We have enough words, finalize this chunk
                chunk_text = ' '.join(current_chunk).strip()
                chunk_text = chunk_text.replace('\n', ' ').replace('\r', ' ')
                chunk_text = chunk_text.replace('\\', ' ').replace('/', ' ')
                # Normalize multiple spaces
                chunk_text = re.sub(r'\s+', ' ', chunk_text)
                if chunk_text:
                    chunks.append(chunk_text)
                
                # Start new chunk with current sentence
                current_chunk = [sentence]
                current_word_count = sentence_words
            else:
                # Below minimum, keep adding
                current_chunk.append(sentence)
                current_word_count += sentence_words
        
        # Add remaining chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk).strip()
            # Remove newlines, backslashes, forward slashes and normalize spaces
            chunk_text = chunk_text.replace('\n', ' ').replace('\r', ' ')
            chunk_text = chunk_text.replace('\\', ' ').replace('/', ' ')
            chunk_text = re.sub(r'\s+', ' ', chunk_text)
            word_count = len(chunk_text.split())
            
            # Always include remaining chunk if it has content
            # If it's below minimum but we have other chunks, try to merge
            if word_count >= self.min_words_per_chunk:
                chunks.append(chunk_text)
            elif chunks:
                # Try to merge small remaining chunk with last chunk if possible
                last_chunk_words = len(chunks[-1].split())
                if last_chunk_words + word_count <= self.max_words_per_chunk:
                    merged = chunks[-1] + ' ' + chunk_text
                    # Ensure merged chunk has no newlines, backslashes, or forward slashes
                    merged = merged.replace('\n', ' ').replace('\r', ' ')
                    merged = merged.replace('\\', ' ').replace('/', ' ')
                    merged = re.sub(r'\s+', ' ', merged).strip()
                    chunks[-1] = merged
                else:
                    # Can't merge, add as is (will be handled by min_threshold in process_text)
                    chunks.append(chunk_text)
            else:
                # No chunks yet, add this one (will be included if >= 50 words)
                chunks.append(chunk_text)
        
        return chunks
    
    def _final_cleanup(self, text: str) -> str:
        """
        Final comprehensive cleanup to remove all unwanted symbols
        This is the last line of defense to ensure clean chunks
        """
        if not text:
            return ""
        
        # Remove all backslashes (handle both single and escaped)
        # Replace all backslash characters with spaces
        text = text.replace('\\', ' ')
        text = text.replace(chr(92), ' ')  # ASCII backslash
        
        # Remove all forward slashes
        text = text.replace('/', ' ')
        text = text.replace(chr(47), ' ')  # ASCII forward slash
        
        # Remove all newlines and carriage returns
        text = text.replace('\n', ' ')
        text = text.replace('\r', ' ')
        text = text.replace('\t', ' ')
        text = text.replace(chr(10), ' ')  # Line feed
        text = text.replace(chr(13), ' ')  # Carriage return
        text = text.replace(chr(9), ' ')   # Tab
        
        # Remove any remaining escape sequences
        text = re.sub(r'\\[a-zA-Z0-9]', ' ', text)  # \ followed by alphanumeric
        text = re.sub(r'\\"', ' ', text)  # Escaped quotes
        text = re.sub(r"\\'", ' ', text)  # Escaped single quotes
        # Remove OCR pagination artifacts that survived line-level cleaning.
        text = re.sub(r"(?i)(?:\bpage\s+\d+\s+of\s+\d+\b[\s,;:]*){2,}", " ", text)
        text = re.sub(r"(?i)\breference id\b[:\s-]*[a-z0-9-]{6,}", " ", text)

        # Normalize all whitespace to single spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _init_llm_client(self):
        """Initialize LLM client for chunk cleaning"""
        try:
            if self.llm_provider == "openai":
                from openai import OpenAI
                api_key = self.llm_api_key or os.getenv('OPENAI_API_KEY')
                if not api_key:
                    print("[DataCleaningPipeline] Warning: OPENAI_API_KEY not set, LLM cleaning disabled")
                    self.use_llm_cleaning = False
                    return
                try:
                    import httpx
                    http_client = httpx.Client(timeout=60.0)
                    self._llm_client = OpenAI(api_key=api_key, http_client=http_client)
                except Exception:
                    self._llm_client = OpenAI(api_key=api_key)
                self.llm_model = self.llm_model or "gpt-4o-mini"
                
            elif self.llm_provider == "anthropic":
                from anthropic import Anthropic
                api_key = self.llm_api_key or os.getenv('ANTHROPIC_API_KEY')
                if not api_key:
                    print("[DataCleaningPipeline] Warning: ANTHROPIC_API_KEY not set, LLM cleaning disabled")
                    self.use_llm_cleaning = False
                    return
                self._llm_client = Anthropic(api_key=api_key)
                self.llm_model = self.llm_model or "claude-sonnet-4-20250514"

            elif self.llm_provider == "ollama":
                # Ollama uses HTTP requests directly
                self._llm_client = "ollama"  # Marker for Ollama
                self.llm_model = self.llm_model or os.getenv('OLLAMA_MODEL', 'llama3')
                self.ollama_api_key = self.llm_api_key or os.getenv('OLLAMA_API_KEY')
            else:
                print(f"[DataCleaningPipeline] Unknown LLM provider: {self.llm_provider}")
                self.use_llm_cleaning = False
        except ImportError as e:
            print(f"[DataCleaningPipeline] Failed to import LLM library: {e}")
            print("[DataCleaningPipeline] LLM cleaning disabled. Install with: pip install openai")
            self.use_llm_cleaning = False
        except Exception as e:
            print(f"[DataCleaningPipeline] Failed to initialize LLM client: {e}")
            self.use_llm_cleaning = False
    
    def _clean_chunk_with_llm(self, chunk_text: str, section_name: str = "") -> str:
        """
        Clean a single chunk using LLM
        
        Args:
            chunk_text: Text content of the chunk
            section_name: Name of the section (for context)
            
        Returns:
            Cleaned chunk text
        """
        if not self.use_llm_cleaning or not self._llm_client:
            return chunk_text
        
        try:
            # Create cleaning prompt
            system_prompt = """You are a text cleaning assistant. Your task is to remove unwanted content from academic/research text while preserving all meaningful information.

Remove:
- Cookie consent notices, privacy policy text, terms of service
- Navigation menus, headers, footers, UI elements
- Boilerplate text like 'click here', 'read more', 'link opens in new window'
- Cookie category descriptions
- Excessive whitespace and formatting artifacts
- HTML/XML tags if any remain
- HTML/CSS/JS markup, inline style attributes, and tag/attribute noise (style=, class=, font-size, padding, margin, text-align)
- Legal/policy sections (privacy/terms/consent/CCPA/California-resident notices)
- Prompt/instruction artifacts (CRITICAL RULES, Output format, Text segment to analyze, JSON output)
- Dataset markers (=== SEED URL ===, === PROFILE PAGE ===, === WEBPAGE ===, top of page, bottom of page, back to top)
- Base64/hashed/garbled strings and tracking IDs
- Unwanted symbols and noise
- Anti-bot/challenge text (e.g., 'Access to this page has been denied', 'Press & Hold to confirm you are a human', 'Reference ID ...')
- OCR pagination/index noise (e.g., repeated 'Page 216 of 850')

Keep:
- All academic content, research descriptions, publications
- Educational background, work experience, achievements
- Technical details, dates, names, titles
- All meaningful information

IMPORTANT:
- Preserve the exact meaning and content
- Maintain sentence structure and readability
- Do not add, modify, or summarize content
- Return ONLY the cleaned text with no explanations or meta-commentary
- Keep all backslashes, forward slashes, and newlines removed (use spaces instead)"""
            
            user_prompt = f"""Clean the following text{' from the ' + section_name + ' section' if section_name else ''} by removing unwanted content. Keep only the meaningful academic/research content:

{chunk_text}"""
            
            if self.llm_provider == "openai":
                response = self._llm_client.chat.completions.create(
                    model=self.llm_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1,  # Low temperature for consistent cleaning
                    max_tokens=4000,
                    timeout=30
                )
                cleaned_text = response.choices[0].message.content.strip()

            elif self.llm_provider == "anthropic":
                response = self._llm_client.messages.create(
                    model=self.llm_model,
                    max_tokens=4000,
                    temperature=0.1,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                )
                cleaned_text = response.content[0].text.strip()

            elif self.llm_provider == "ollama":
                import httpx
                
                # Prepare prompt for Ollama
                full_prompt = f"{system_prompt}\n\n{user_prompt}"
                
                # Determine API endpoint
                if self.ollama_api_key:
                    # Ollama Cloud API
                    api_url = 'https://api.ollama.com/v1/chat/completions'
                    headers = {
                        'Authorization': f'Bearer {self.ollama_api_key}',
                        'Content-Type': 'application/json'
                    }
                    payload = {
                        'model': self.llm_model,
                        'messages': [
                            {'role': 'user', 'content': full_prompt}
                        ],
                        'stream': False
                    }
                else:
                    # Local/remote Ollama server
                    api_url = f'{self.ollama_host.rstrip("/")}/api/generate'
                    headers = {'Content-Type': 'application/json'}
                    payload = {
                        'model': self.llm_model,
                        'prompt': full_prompt,
                        'stream': False
                    }
                
                # Make API request
                with httpx.Client(timeout=60.0) as http_client:
                    resp = http_client.post(api_url, json=payload, headers=headers)
                    resp.raise_for_status()
                    result = resp.json()
                    
                    # Handle different response formats
                    if self.ollama_api_key:
                        cleaned_text = result.get('choices', [{}])[0].get('message', {}).get('content', '')
                    else:
                        cleaned_text = result.get('response', '')
                
                cleaned_text = cleaned_text.strip()
            else:
                return chunk_text
            
            # Validate cleaned text - if it's too short or empty, use original
            if not cleaned_text or len(cleaned_text) < len(chunk_text) * 0.1:
                print(f"[DataCleaningPipeline] LLM returned very short text, using original")
                return chunk_text
            
            # Final cleanup to ensure no symbols remain
            cleaned_text = self._final_cleanup(cleaned_text)
            
            return cleaned_text
            
        except Exception as e:
            print(f"[DataCleaningPipeline] Error cleaning chunk with LLM: {str(e)}")
            # Fallback to original text
            return chunk_text
    
    def process_text(self, text: str, profile_url: str = "", section_header: str = "") -> List[Dict]:
        """
        Complete pipeline: normalize -> clean -> sequential chunk with overlap
        Maintains original order with fixed windows and overlap to preserve context
        
        Args:
            text: Raw text to process
            profile_url: Source profile URL (for metadata)
            section_header: Optional section header name (overrides detected sections)
            
        Returns:
            List of chunk dictionaries with optimal word counts (250-400, target 325)
        """
        if not text:
            return []
        
        normalized = self.normalize_text(text)
        section_label = section_header or "CONTENT"
        return self._sequential_chunk_text(
            text=normalized,
            profile_url=profile_url,
            section_label=section_label
        )
    
    def process_json_file(self, input_file: str, output_file: str = "chunks.json"):
        """
        Process extracted_content.json and create chunks.json
        
        Args:
            input_file: Path to input JSON file
            output_file: Path to output chunks JSON file
        """
        try:
            # Load input JSON
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"Error: File {input_file} not found")
            return
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in {input_file}: {e}")
            return
        
        final_chunks = []
        
        # Handle different JSON structures
        if isinstance(data, dict):
            # Check if it's the profile structure from json_writer
            if 'profiles' in data:
                for profile in data['profiles']:
                    profile_url = profile.get('profile_url', '')
                    raw_text = profile.get('raw_text', '')
                    
                    if raw_text:
                        chunks = self.process_text(raw_text, profile_url=profile_url)
                        final_chunks.extend(chunks)
            
            # Check if it has full_text, bio, etc. (scraped structure)
            elif 'full_text' in data or 'bio' in data:
                # Combine all text fields
                combined_text = ""
                if 'full_text' in data:
                    combined_text += data['full_text'] + "\n\n"
                if 'bio' in data:
                    combined_text += data['bio'] + "\n\n"
                
                profile_url = data.get('url', data.get('profile_url', ''))
                chunks = self.process_text(combined_text, profile_url=profile_url)
                final_chunks.extend(chunks)
            
            # Single text field
            elif 'text' in data:
                profile_url = data.get('url', data.get('profile_url', ''))
                chunks = self.process_text(data['text'], profile_url=profile_url)
                final_chunks.extend(chunks)
        
        elif isinstance(data, list):
            # List of items
            for item in data:
                if isinstance(item, dict):
                    text = item.get('text', item.get('content', item.get('full_text', '')))
                    profile_url = item.get('url', item.get('profile_url', ''))
                    if text:
                        chunks = self.process_text(text, profile_url=profile_url)
                        final_chunks.extend(chunks)
        
        # Save chunks to JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_chunks, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Processed {len(final_chunks)} chunks and saved to {output_file}")
        return final_chunks


def main():
    """Main function to run the pipeline"""
    import sys
    
    input_file = "extracted_content.json"
    output_file = "chunks.json"
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    pipeline = DataCleaningPipeline(target_words_per_chunk=325, min_words_per_chunk=250, max_words_per_chunk=400)
    pipeline.process_json_file(input_file, output_file)


if __name__ == "__main__":
    main()

